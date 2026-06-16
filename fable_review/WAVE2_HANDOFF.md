# Wave 2 — Architecture Refactors: Full Handoff (start in a fresh chat)

This is the complete brief to execute **Wave 2** of the architecture/maintainability work
on the H1 SMC alert system. Wave 1 status and the original plan live in
`fable_review/ARCHITECTURE_HANDOFF.md`; the ground truth on system behaviour lives in
`fable_review/FABLE_REFERENCE_LIVE.md`. **Read all three before touching code.**

Line numbers in the ORIGINAL handoff have already drifted once. **Every file:line in THIS
document was re-verified against source on 2026-06-16.** Re-verify again at the start of your
chat — the live modules get committed hourly by GitHub Actions (state files only, but rebases
can shift things).

---

## 0. Hard rules (binding — do not relax)

- **Behaviour-neutral.** Wave 2 is pure refactor. Zero trading-logic change. Any item that
  also carries a behaviour change (the 2B death-line buffer) lands the STRUCTURE first and gets
  a SEPARATE explicit sign-off for the behaviour flip.
- **No code edit without the owner seeing it first.** Trading logic always needs confirmation.
- **H1-only.** Ignore M5/M15, Phase 3. BUT the backtest imports the live modules — see §1.
- **Verify, don't trust** — including this doc.
- **One concept, one implementation.** Duplicate logic is a bug.
- **Python 3.14** for tests (`python`, not `python3`). **pytest is NOT installed** — repo tests
  are plain scripts with `if __name__ == "__main__": raise SystemExit(main())`. Run them
  directly: `python backtest/scanlog/test_scanlog_self.py`.
- Open files `encoding='utf-8'` in test scripts (source has unicode/HTML entities).
- **Commit/push only on "ship it" / "push".** Stage only relevant files, never `.claude/` or
  `backtest/out/` (generated run artifacts — not gitignored, but never commit them).
- **Every push collides with GitHub-Action state commits.** They touch DISJOINT files (state
  JSON vs code), so `git pull --rebase origin main` is always clean. Stash any `backtest/out/`
  artifact before rebasing.

---

## 1. The blast-radius constraint (READ FIRST — this shapes 2D and 2E)

The backtest imports the live modules and **reaches into private symbols**. Any split or
rename MUST keep ALL of these import paths working. This is the exact surface, verified:

**Public symbols the backtest uses:**
- `smc_radar.compute_pair_walls`, `smc_radar.detect_smc_radar`
- `smc_detector.compute_atr`, `smc_detector.run_scorecard`,
  `smc_detector.compute_phase2_levels`, `smc_detector.resolve_distal_mode`,
  `smc_detector.MIN_LEG_ATR_MULT`
- `dealing_range.detect_swings`, `dealing_range.MIN_LEG_ATR_MULT`,
  `dealing_range.BOS_ATR_MULT`
- `smc_detector` imported by `replay_engine.py`, `h1_only_simulator.py`, `run_backtest.py`

**PRIVATE symbols the backtest ALSO reaches into (the trap):**
- `smc_detector._ATR_CACHE`, `smc_detector._atr_cache_status`
- `dealing_range._filter_swings_by_leg_atr`

→ **2E (module split) and 2D (knobs.py) must re-export these private names too**, not just the
public ones. A facade that only re-exports the public API silently breaks the backtest.

---

## 2. Wave 2 items, in mandatory order

### 2A — Golden-file regression harness for `compute_structure` (DO FIRST — gates everything)

**Why first:** `compute_structure` (`dealing_range.py:451`) is one ~500-line pure forward
loop. Its block ORDER is load-bearing (failure-window `continue` pre-empts CHoCH/BOS). It has
NO tests. **No Wave-2 item that touches structure (2B touches the slate that structure feeds;
2F rewrites structure) may proceed until this harness is green.**

**What:** Record `(df window → full output)` for curated nasty cases, per pair, and assert
byte-identical output in CI.

- **Input fixtures:** real H1 df windows that exercise: V-reversal bar, CHoCH-then-revert
  (failure window fires), Range BOS (broken swing within `BOS_ATR_MULT` of H4 wall),
  birth/cold-start (`state==undefined` → first break), weekend gap (H4 gap-aware resample).
- **Captured output** (the full return of `compute_structure`): `state`, `trend`, `events`
  (the ring — every field per §4.4 of FABLE_REFERENCE), `swings`, `choch_flip_count`,
  `defended`, `flip_unconfirmed`, `ranging`.
- **Serialization:** dump output to JSON with a stable key order + rounded floats (ATR-derived
  values can have float noise across runs — round to a fixed dp, e.g. the pair's
  `decimal_places`, before hashing). Decide rounding ONCE and document it in the harness.
- **Assertion:** re-run `compute_structure` on the saved input, compare to saved output,
  fail on any diff. Print a readable diff (which event/field changed) on failure.
- **How to source fixtures:** pull real windows from the 30d H1 the backtest already fetches,
  OR hand-craft minimal OHLC frames. Real windows are higher-fidelity; hand-crafted are
  smaller and clearer. Recommend: 1–2 hand-crafted per case (clarity) + 1 real window per pair
  (fidelity).
- **Where:** new `tests/test_structure_golden.py` (or `backtest/` if you keep tests there —
  match the repo's existing location; scanlog tests live in `backtest/scanlog/`).

**Edge:** the harness itself must be deterministic. `compute_atr` is memoised on an OHLC
fingerprint (`_ATR_CACHE`) — clear or account for it between fixtures so cache state can't
leak. **Maintenance: LOW** (regenerate fixtures only on intentional structure changes — that
regeneration step IS the feature: it forces you to eyeball every behaviour change).

---

### 2B — Typed `Zone` dataclass (gated by 2A + a slate round-trip test)

**Why:** the slate is a hand-copied dict. Fields have died in hand-copies before — this caused
the BOS-count bug and the mitigation-window bug. A dataclass with ONE serialization path
(`to_dict`/`from_dict`) retires that entire bug class.

**Where (all re-verified 2026-06-16):**
- `smc_radar.py:3072` `load_slate()`
- `smc_radar.py:3080` `save_slate(slate)` (uses `save_json_atomic`)
- `smc_radar.py:3084` `find_matching_slate_zone(fresh_zone, slate_zones, pair_type)`
- `smc_radar.py:3113` `fresh_to_slate_zone(fresh_zone, zone_id, ist_now, current_price, dp)`
  — THE field-copy site. Every slate field is enumerated here.
- `smc_radar.py:3197` `resync_slate_zone_indices(slate_zone, df, pair_name="")`
- `smc_radar.py:3275` `refresh_slate_zone(slate_zone, fresh_zone, ist_now, current_price, dp)`
  — the OTHER field-copy site. Must stay in sync with `fresh_to_slate_zone`.
- Both phases READ the slate: Phase 1 reconcile loop in `run_radar`; Phase 2 reads it after
  the freshness gate (`Phase2_Alert_Engine.py`, slate load near line 1748+).

**Critical edge — round-trip EVERY existing slate field** or the migration itself drops one
(the exact bug it's meant to kill). Before writing the dataclass: dump a live `active_obs.json`,
enumerate every key on a zone (incl. recently-added `break_quality`, `bos_sequence_count`,
`status_label`, `h1_atr`, `touches`, the FVG/sweep/dealing_range snapshots). The dataclass
must serialize a byte-identical dict for an unchanged zone. **Gate behind: (a) 2A green, and
(b) a NEW slate round-trip test** that loads the current `active_obs.json`, runs it through
`from_dict → to_dict`, and asserts equality.

**Derived facts the dataclass should OWN** (single source — this is the point):
- `zone.mitigation_window_start()` — from its own `bos_timestamp` (event-candle+1). Today both
  phases compute this separately; the Phase-2 mismatch was a 2026-06-10 bug.
- `zone.death_line(pair_cfg)` — consumed by BOTH `is_ob_mitigated_phase1` AND the SL math.

**Land structure-only first (behaviour-neutral).** The **death-line buffer / any pair-aware SL
change is a BEHAVIOUR FLIP → separate explicit sign-off.** Do not bundle it into the dataclass
migration. **Maintenance: LOW** once done.

---

### 2C — Chart consolidation (independent of 2A/2B — can be done any time)

**Why:** TWO H1 chart renderers that must be kept in visual sync BY HAND, and have likely
drifted:
- `smc_radar.py:1206` `generate_h1_chart(df, ob, dp, pair_name, ist_timestamp, walls=None, ...)`
  — Phase 1 digest chart.
- `Phase2_Alert_Engine.py:908` `generate_h1_chart(df_h1, ob, pair_conf, title, levels=None,
  dealing_range=None)` — Phase 2 context chart.
- `Phase2_Alert_Engine.py:745` `generate_h1_zoomed_chart(df_h1, ob, pair_conf, title,
  levels=None)` — Phase 2 zoomed chart.

Note the signatures already differ (Phase 1 takes `dp`/`pair_name`/`walls`; Phase 2 takes
`pair_conf`/`title`/`levels`). They have NO shared helpers.

**Step zero (mandatory):** render BOTH renderers on ONE dataset and visually/pixel-diff them.
**That diff is your acceptance baseline** — it tells you what already drifted and what the
"correct" look is. Decide WITH THE OWNER which renderer is canonical before merging.

**Then:** one `charts.py` with shared style constants + three entry points
(digest / context / zoomed). Behaviour = visual only. **Maintenance: LOW** (drift becomes
impossible). **This is cosmetic but large** — budget accordingly; it's the safest item to do
in isolation if structure work is blocked.

---

### 2D — `knobs.py` centralization (do with or after 2E — they share the facade problem)

**What:** all 23 constants in one module, names matching FABLE_REFERENCE_LIVE.md §3.

**CRITICAL edge — re-export old names from their ORIGINAL modules** so every import in §1 keeps
working. Specifically `dealing_range.MIN_LEG_ATR_MULT`, `dealing_range.BOS_ATR_MULT`,
`smc_detector.MIN_LEG_ATR_MULT`, etc. must still resolve. Pattern: `knobs.py` is the source of
truth; each original module does `from knobs import MIN_LEG_ATR_MULT` (or
`MIN_LEG_ATR_MULT = knobs.MIN_LEG_ATR_MULT`) so the old path still works.

The 23 knobs + file:line + value are tabulated in **FABLE_REFERENCE_LIVE.md §3** — use that
table verbatim as the spec. ~18 are module-level constants; 5 are per-pair config. This is also
the future home for a second swing-filter pair (`SWING_FILTER_STRUCTURE` / `SWING_FILTER_TP`)
if a later trading change needs it. **Maintenance: LOW.**

---

### 2E — Facade-preserving `smc_detector` split (do with 2D)

**What:** move `smc_detector.py` internals into:
- `primitives.py` — `compute_atr` (`smc_detector.py:96`), `get_swing_points` (`:446`),
  `observe_phase1_sweep` (`:939`), `detect_fvg_in_zone` (`:1257`), `is_ob_mitigated_phase1`
  (`:2372`), `compute_break_quality` (`:2308`), `resolve_distal_mode`,
  `compute_bos_sequence_count` (`:527`).
- `phase2_math.py` — `run_scorecard` (`:1697`), `compute_phase2_levels` (`:1486`),
  `classify_setup` (`:2066`).
- `smc_detector.py` stays as a **thin explicit re-export facade** so backtest imports are
  untouched.

**CRITICAL edge — the facade must re-export the PRIVATE symbols too** (§1):
`_ATR_CACHE`, `_atr_cache_status` (used by the backtest's ATR-cache-status check). A facade that
only re-exports public functions WILL break the backtest silently. Verify with a smoke import:
`python -c "import smc_detector; smc_detector._ATR_CACHE; smc_detector.run_scorecard"` and run
the backtest's import-level checks.

**Pure moves, zero behaviour change. Maintenance: LOW.** Do AFTER 2D (or together) so the knob
imports and the function moves are reconciled once.

---

### 2F — Block-extraction of `compute_structure` (DEFER — only if a trading change needs it)

Break `compute_structure` into pure per-bar functions over an explicit `StructureState`, under
2A's green harness. **Do not do this speculatively.** It is the highest-risk item (the block
order is subtle and load-bearing). Defer until a real trading-logic change makes the monolith
the bottleneck. The handoff is explicit: purity is the RIGHT design — the only problem is the
missing tests, which 2A solves without a rewrite.

---

## 3. Behaviour-change items needing explicit sign-off (do NOT land silently)
- The **death-line buffer** + any pair-aware SL change carried by 2B.
- (From Wave 1, still pending) the **fail-loud half of 1C** schema_version.

---

## 4. Recommended execution order for the new chat
1. Re-read FABLE_REFERENCE_LIVE.md + ARCHITECTURE_HANDOFF.md + this file. Re-verify line numbers.
2. **2A first** (harness) — nothing structural moves without it. Get it green and committed.
3. **2C** any time (independent, cosmetic, safe) — good warm-up / good if structure work stalls.
4. **2D + 2E together** (facade + knobs) — reconcile imports once; smoke-test the private
   re-exports against the backtest.
5. **2B** behind 2A-green + the slate round-trip test. Structure-only; hold the SL buffer for
   sign-off.
6. **2F** only if a future trading change demands it.
7. Each item: small, tested (Python 3.14), shown to the owner BEFORE commit. Push only on
   "ship it". Never stack a refactor on unverified changes.

---

## 5. Wave 1 status carried in (so you don't re-do it)
- **1A** (per-alert structured log = `backtest/scanlog/`): SHIPPED. Behaviour-neutral verified
  (`Sum r_realised` identical with/without instrumentation; self-tests pass).
- **1B** (silent-degrade visibility): SHIPPED. Reused the existing heartbeat log-count pattern
  (`smc_radar.log_p1_degrade()` → P1-owned `p1_degrade_log.json`; Phase2 `_count_recent_by_kind`
  + heartbeat Rule 8). Wired at `p1_email_fail` (digest-send except), `walls_h4_error` and
  `walls_structure_error` (`compute_pair_walls` excepts). Added the log to `run_phase1.yml`
  commit list. `atr_fallback` + `state_push_fail` deliberately NOT done — see 1E note below.
- **1D** (truth pass): SHIPPED. `TEMP DIAG`→`OB BUILD LEDGER` rename; `config.json _key_labels`.
- Open trading item: the OB min-range filter (`OB_MIN_RANGE_ATR_MULT=0.3`) was TRIED & REMOVED
  2026-06-16 (backtest showed zero trade change). Do NOT re-add. It is correctly absent.

---

## 6. Wave 1 LEFTOVERS — full actionable briefs (do these too; do NOT lose them)

These two Wave-1 items were deferred, NOT cancelled. Full specs here so a fresh chat can do
them without re-deriving. Both are independent of Wave 2 ordering.

### 1C — `schema_version` on state files + tolerant reader

**State of the problem:** No state file carries a version. Readers do `json.load` +
`isinstance(dict)` only (verified: `dealing_range.load_state` `dealing_range.py:132`;
`smc_radar.load_json_safe` `smc_radar.py:126`, `load_slate` `smc_radar.py:3072`;
`Phase2.load_json` `Phase2_Alert_Engine.py:162`). If a schema ever changes shape, an old
reader silently mis-parses instead of failing.

**Two halves — they have DIFFERENT risk levels, do not conflate:**
1. **Stamp (pure, safe):** writers add `"schema_version": 1` to the dict before save. Writers:
   `dealing_range.save_state` (`:144`), `smc_radar.save_json_atomic` (`:119`) used by
   `save_slate` (`:3080`), `Phase2.save_json` (`:170`).
2. **Tolerant reader (BEHAVIOUR CHANGE → needs owner sign-off):** reader checks the version.
   **CRITICAL edge: MISSING version must be treated as v1**, or the FIRST run after deploy
   fails on every existing state file. Only a MISMATCH (present but ≠ expected) fails loud +
   sends a failure email.

**Decision already made:** the stamp alone is a field nothing reads = dead weight. This item is
only worth doing as ONE package (stamp + fail-loud reader). So it is a **single sign-off
conversation** — present both halves, get the owner's yes on the fail-loud behaviour, then land
together. **Do NOT half-ship the stamp.** Maintenance: LOW (bump only on real schema changes).

### 1E — immediate safe deletes / fences (counter-gated)

**State of the problem:** dead or M5/M15 remnants linger in the shared modules. Verify each is
truly dead BEFORE deleting; where unsure, fence + label instead of deleting.

**Concrete targets (verified 2026-06-16):**
- `FVG_WINDOW_M15_CANDLES = 40` (`smc_detector.py:171`) and any sibling M5/M15 constants — DEAD
  in the H1-only path. **Fence + label** (do not delete — shared module; a label is no-risk).
- `DEALING_RANGE_LOOKBACK_H1` (`smc_detector.py:137`) + the legacy fixed-lookback branch in
  `get_dealing_range` (`smc_detector.py:326`, the fallback at `:405`
  `lookback = DEALING_RANGE_LOOKBACK_H1.get(...)`). This is the legacy path used only when H4
  state is missing. **Counter-gate then delete:** confirm via live logs (now possible — see the
  1B `p1_degrade_log` and the scan logs) that the fallback has not fired in 30 days, THEN delete.
- The Phase-2 `bos_timestamp`-missing fallback in the still-alive gate (legacy zones only).
  **Counter-gate then delete** after one 15-day slate turnover shows zero use.
- Any remaining stale `lows[-2]` remnants from the fixed BOS path (the BOS-on-close fix should
  have removed these — verify none remain).

**Method:** add a one-line degrade-log call (reuse `log_p1_degrade` from 1B) on each legacy
branch you intend to delete; let it run live; delete only the branches that logged zero hits
over the window. **Do not bulk-delete on inspection alone.** Maintenance: NONE once removed.
