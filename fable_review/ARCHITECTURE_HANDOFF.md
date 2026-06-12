# Architecture Work — Handoff (execute Part 2 in a fresh chat)

This is the complete brief to execute the **architecture / maintainability** work on the
H1 SMC alert system without re-deriving anything. Read this top to bottom first. The repo is
accessible; verify every claim against source before acting (the rule on this project is
**verify, don't trust** — including this document).

---

## 0. Hard rules (binding — from CLAUDE.md + accumulated decisions)

- **No-maintenance solutions by default.** Only assign a maintenance tier (low/med/high) when
  a no-maintenance solution is genuinely impossible, and then say *why*. Prefer ATR-relative
  self-scaling values, a single source of truth, reading already-computed fields, contained
  changes. (Memory: `feedback_no_maintenance_solutions.md`.)
- **No trading-logic change without explicit approval.** This architecture work is meant to be
  **behaviour-neutral**. Where an item also carries a behaviour change (fail-loud, the
  death-line buffer), land the structure first and get a separate sign-off for the behaviour flip.
- **H1-only.** Ignore M5/M15, Phase 3, backtest as review *targets* — BUT the backtest IMPORTS
  the live modules (`smc_detector`, `smc_radar`, `dealing_range`, `h4_range`), so do not break
  their public import paths or rename/remove things they use. Prefer a facade.
- **Plain English, brutal honesty, no sycophancy.** Match length to the question.
- **Commit/push only on "ship it" / "push".** Stage only relevant files, never `.claude/`.
- **Python 3.14 is installed** (use it for tests). PowerShell on Windows; `bash` tool available.
- Open files with `encoding='utf-8'` in test scripts (the source has unicode/HTML entities;
  cp1252 decode errors otherwise).

---

## 1. Where the system stands RIGHT NOW (already shipped + committed)

A large trading-logic batch + a setup classifier + break-quality were implemented, tested, and
committed before this handoff. The architecture work happens ON TOP of this state. What's live:

- **Touch counter** is excursion-based (one touch per approach), `OB_TOUCH_REARM_ATR = 0.5`,
  in `is_ob_mitigated_phase1` (smc_detector.py). New `atr` param passed by all live callers.
- **Killzone scored** (+1) on the OB candle via `smc_detector._ts_in_killzone`; scorecard maxes
  are now **forex 9 / others 11** (`scorecard_real_max`). Approach-session shown as info.
- **Gold/NAS distal kill** is pair-aware close-vs-wick via `resolve_distal_mode` + `distal_mode`
  arg (done on a separate chat; it's in `is_ob_mitigated_phase1`).
- **Sweep window** widened ~3 candles before impulse, floored at the prior event
  (`SWEEP_LOOKBACK_BEFORE_IMPULSE = 3`, observe_phase1_sweep).
- **Forex sweep location tags** shown in `build_sweep_breakdown_html`.
- **TP ladder**: leg swing ≥1.5R → H4 confirmed wall ≥1.5R → no-trade with reason
  `no_qualifying_target` (compute_phase2_levels; adds `tp1_source`).
- **OB2 selection** prefers with-trend, then nearest (`_split_primary_alternative(trend=)`).
- **Per-OB BOS count** persisted on the slate (`fresh_to_slate_zone` + `refresh_slate_zone`) and
  read in Phase 2 instead of overwriting.
- **CHoCH tier**: from-zone = 4, mid-range = 3 (run_scorecard, reads `reversal_pct`).
- **Setup classifier** `smc_detector.classify_setup(ob, pd_position, trend_alignment)` →
  badge + note, rendered at the top of the trade email + in the subject. Patterns: "A+ Reversal
  at the Wall", "A First Pullback", "Caution: Late-Trend Chase". `scan_record["setup_badge"]`
  logged. **MEDIUM maintenance** (hand rules) — the no-maintenance version is data-driven once
  the per-alert outcome log exists (Wave 1 item below).
- **Break quality** `smc_detector.compute_break_quality(df, bos_idx, broken_price, direction,
  atr, event_type)` → bucket (marginal / solid / strong) judged EVENT-AWARE: the event minimum
  (CHoCH 1.0 ATR, BOS 0.4 ATR, read live from dealing_range constants) is the qualifying FLOOR;
  we grade `excess` = displacement ÷ floor (always ≥1.0). marginal <1.5× | solid 1.5–2.5× |
  strong ≥2.5× with body ≥1.0 ATR. Frozen on the OB (`break_quality`), persisted, shown as an
  info line ("Break: Solid · cleared 1.9× the required displacement"). Info-only (not scored).
  Cutoffs (1.5 / 2.5) are vet starting points — confirm with the Wave-1 outcome log before trusting.

Live data (2026-06-12) confirmed all of the above behave correctly. Treat this as the baseline.

---

## 2. Deployment model (verified — this shapes the failure-mode work)

- Both workflows are `on: [workflow_dispatch]` (`.github/workflows/run_phase1.yml`,
  `run_phase2.yml`). **No GitHub cron.** An **external scheduler (cron-job.org)** triggers them
  (P1 hourly — see the comment at `Phase2_Alert_Engine.py` near `P1_FRESHNESS_MAX_AGE_HOURS`).
- State is persisted by **git-committing the JSON files** at the end of each Actions run. P1 and
  P2 run as **separate isolated jobs** and commit **disjoint, owned file sets**:
  - P1 owns: `active_obs.json`, `state/structure_state.json`, `email_gate.json`.
  - P2 owns: `phase2_sent.json`, `active_watch_state.json`, `heartbeat_*.json`,
    `gemini_failure_log.json`, `p1_stale_alert_state.json`, `phase2_scan_log.jsonl`.
- **All state writes are atomic** (temp file + `os.replace`): `dealing_range.save_state`,
  `smc_radar.save_json_atomic` (used by `save_slate`), `Phase2.save_json`. Partial-write risk is
  already retired.
- **No file locking** anywhere — and none is needed (disjoint ownership + atomic writes +
  isolated checkouts mean the two jobs never write the same file in one process).

---

## 3. Verified facts the architecture review relied on (don't re-litigate)

- `smc_detector.py` is imported by Phase 1, Phase 2, AND the backtest (`replay_engine.py`,
  `h1_only_simulator.py`). Big blast radius → any split must keep import paths working (facade).
- **No `schema_version`** on any state file; readers do `json.load` + `isinstance(dict)` only.
- The **ATR fallback** in `dealing_range._compute_atr` is **value-identical** to the cached
  `smc_detector.compute_atr` (verified diff 0.0). It is not a math fork — its only gap is that it
  doesn't log when it fires.
- **TEMP DIAG is load-bearing** — `ob_build_diagnostics` feeds the Phase-1 scan log AND the
  "Last OB attempt" line via `_summarise_last_ob_attempt` (smc_radar.py). Do NOT delete; rename.
- `zone_fatigue_threshold` is consumed only by `weekly_review.py` (not the live path). Don't
  delete; relabel.
- Two independent H1 chart renderers exist: `smc_radar.generate_h1_chart` and
  `Phase2_Alert_Engine.generate_h1_chart` + `generate_h1_zoomed_chart`. No shared helpers →
  likely already drifted.
- `compute_structure` (dealing_range.py:440) is one ~500-line pure loop with load-bearing block
  order (failure-window → CHoCH → BOS → birth → ingest, with `continue`). Purity is the RIGHT
  design (reproducible, no state corruption) — keep it; the problem is it has no tests.
- ~18 of 23 knobs are module-level constants requiring a code edit (only per-pair
  `atr_multiplier`/`spread_pips`/`killzones_utc`/`decimal_places` + the scoring block are config).
- Phase-1 email-send failure is `logging.error` only — invisible to the heartbeat (which lives
  in Phase 2). Freshness gate handles weekends (`off_hours`) but has no holiday calendar.

Full ground truth: `fable_review/FABLE_REFERENCE_LIVE.md` (knob table in §3, function map in §4–5).

---

## 4. The plan — execute in this order (each item: what, where, edge guard, maintenance)

### WAVE 1 — safe, behaviour-neutral, high value (do first)

**1A. Per-alert structured log + lagged outcome appender.** THE foundation — unblocks the gate
decision, classifier validation, and knob tuning. One JSONL line per alert: zone_id, pair,
OB1/OB2 role, trigger type, every score component (incl. PD/killzone/reversal_pct/break_quality/
sweep components), levels (entry/SL/TP1/TP2/RR/tp1_source), setup_badge, timestamps. Then a
lagged job that re-reads H1 and appends TP1-before-SL-in-48-bars / MFE / MAE.
- Where: Phase 2 main loop already builds `scan_record["fired_levels"]` — extend to a dedicated
  per-alert record written to a new P2-owned JSONL. Outcome appender = its own small dispatched
  job or a P2 tail step (preserve disjoint file ownership).
- Edge: outcome job must not look ahead (only use bars strictly after the alert). Maintenance: NONE.

**1B. Degradation counters surfaced in the heartbeat.** Named counters on every degrade path:
`p1_email_fail`, `walls_cold_start`, `atr_fallback`, `gemini_fail`, `chart_fail`,
`state_push_fail`. Heartbeat reports nonzero counters since last beat. Fixes P1 invisibility +
partial-failure detection (the dead-man resolution — see §5).
- Where: degrade points = `compute_pair_walls` (smc_radar.py:1009, cold-start), the digest
  send try/except (smc_radar.py ~4144), `dealing_range._compute_atr` fallback branch,
  `call_gemini_flash`, `_log_chart_failure`. Heartbeat: `collect_heartbeat_diagnostics` /
  `build_heartbeat_email_html` / `send_heartbeat_if_due` (Phase2). Persist counters to a
  P1-owned + P2-owned counters file (disjoint ownership).
- Maintenance: NONE.

**1C. `schema_version` on every state file + tolerant reader.** Writer stamps a version; reader
validates. **Edge guard (critical): a MISSING version must be treated as v1**, or the first run
after deploy fails on existing files. Mismatch (not missing) → fail loud + failure email.
- Where: the save_* and load_* functions in §2. **The fail-loud half is a behaviour change →
  sign-off.** The version stamp itself is pure cleanup.
- Maintenance: LOW (one field; bump only on real schema changes).

**1D. Truth pass (text/labels only).** Delete the false killzone comment (ALREADY removed in the
trading batch — verify it's gone). Relabel the news/`hard_gates` config as `informational`.
Rename "TEMP DIAG" → a permanent name (it's load-bearing). Add a one-word label to each config
key: `live-gate` / `informational` / `weekly-review-only`.
- Maintenance: NONE.

**1E. Immediate safe deletes** (verify each is truly dead first, counter-gate if unsure): any
remaining stale `lows[-2]` remnants from the fixed BOS path. Fence + label (don't delete) the
M5/M15 constants in shared modules. Counter-gate then delete: the `get_dealing_range` legacy
fixed-lookback path (no live log evidence in 30 days) and the Phase-2 `bos_timestamp`-missing
fallback (after one 15-day slate turnover at zero).

### WAVE 2 — real refactors, behaviour-neutral but gated by tests (do after Wave 1)

**2A. Golden-file regression harness for `compute_structure`.** Record (df window → full output:
ring, trend, swings, flags) for curated nasty cases (V-reversal bar, CHoCH revert, Range BOS,
birth/cold-start, weekend gap) per pair. CI asserts byte-identical output. **Do this BEFORE any
structure edit.** Zero behaviour change. Maintenance: LOW (regenerate fixtures only on
intentional changes — a feature).

**2B. Typed `Zone` dataclass.** Single serialization path (`to_dict`/`from_dict` — a field can
never again die in a hand-copy, which caused the BOS-count + mitigation-window bugs). Owns
derived facts: `zone.mitigation_window_start()` (from its own `bos_timestamp`) and the shared
`zone.death_line(pair_cfg)` consumed by BOTH `is_ob_mitigated_phase1` and the SL math.
- Where: `fresh_to_slate_zone` (smc_radar.py:3081), `refresh_slate_zone` (3249), the slate I/O,
  both phases' reads. **Edge (critical): round-trip EVERY existing slate field** or the migration
  itself drops one — gate behind a slate round-trip test + 2A's harness. Land structure-only
  first (behaviour-neutral); the death-line buffer + any behaviour flip = separate sign-off.
- Maintenance: LOW once done (retires the whole dict-drift bug class).

**2C. Chart consolidation.** Step zero: render both renderers on ONE dataset and pixel/visual-diff
(they've likely drifted) — that diff is the acceptance baseline. Then one `charts.py` with shared
style constants + three entry points (digest / context / zoomed). Decide which renderer is the
canonical look. Behaviour = visual only. Maintenance: LOW (drift becomes impossible).

**2D. `knobs.py` centralization.** All 23 constants in one module, names matching
FABLE_REFERENCE_LIVE.md §3. **Edge: re-export old names from their original modules** so backtest
imports keep working. This is also the home for any future second swing-filter pair
(`SWING_FILTER_STRUCTURE` / `SWING_FILTER_TP`). Maintenance: LOW.

**2E. Facade-preserving `smc_detector` split.** Move into `primitives.py` (ATR, swings wrapper,
sweep, FVG, mitigation, break-quality) + `phase2_math.py` (scorecard, levels, classifier), keep
`smc_detector.py` as a thin explicit re-export facade so the backtest's imports are untouched.
Pure moves, zero behaviour change. Maintenance: LOW.

**2F. (only if a future trading change needs it) Block-extraction of `compute_structure`** into
pure per-bar functions over an explicit `StructureState`, under 2A's green harness. Defer until
needed.

### Behaviour-change items needing explicit sign-off (do NOT land silently)
- The **fail-loud** half of 1C.
- The **death-line buffer** + any pair-aware SL change carried by 2B.

---

## 5. Genuine compromises (cannot be fully fixed — accepted)

- **Dead-man (total dispatcher death on a quiet day).** The watcher (heartbeat) lives inside the
  watched system; if the external scheduler dies, everything including the heartbeat stops. The
  ONLY external fix (a healthchecks.io-style ping) was **declined by the owner** — he'll notice
  prolonged absence of near-zone (<2.5×ATR) emails. **Partial/silent failures ARE covered** by
  the Wave-1 counters (1B). Residual accepted risk: total death on a genuinely quiet day looks
  like a normal quiet day. Do NOT add the external ping unless the owner reverses this.
- **Git-push race.** Two jobs push to one branch; rarely the slower push is rejected and that
  run's state is lost (could repeat one email). Fix: detect rejection + `state_push_fail` counter
  + retry-with-rebase in the workflow. Cannot be made 100% race-proof without locking (which the
  clean disjoint design deliberately avoids). Accepted: low-probability, self-heals next run.
- **No holiday calendar.** Phase-2 freshness gate can false-warn on a weekday market holiday.
  Accepted: not worth a holiday-data dependency for a couple of false emails a year. Document.

---

## 6. Suggested working method for the new chat

1. Re-read `fable_review/FABLE_REFERENCE_LIVE.md` (ground truth) + this handoff.
2. Confirm §1 (current state) against source — the trading batch should all be present.
3. Execute Wave 1 in order; each change small, tested (Python 3.14), shown to the owner before
   commit. Get sign-off on the 1C fail-loud half.
4. Wave 2 only after Wave 1 is committed and has run once live. 2A (harness) before anything that
   touches structure; 2B behind the round-trip test.
5. Never stack a refactor on unverified changes. Commit + push only on "ship it".
