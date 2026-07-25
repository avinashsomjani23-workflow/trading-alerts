# OB Freeze / Recompute — Findings & Fix Plan

**Status:** findings captured, NO code changed yet. Trust rule: every claim here is
tagged with a live `file:line`; if a line disagrees with this doc, the code wins.

**Why this doc exists:** an audit of "is the backtest contaminated?" found that the
backtest **speed problem** and a **freeze-vs-live confusion** are the *same* root
cause. This is the durable record so the finding is not lost in a huge chat, and so
no future audit re-discovers it in a panic.

---

## 1. Verified freeze/live map (source: `zone.py`, `smc_radar.py`)

Every order-block field is in ONE of four buckets. Verified line-by-line against
`Zone.refresh` ([zone.py:235-312](../zone.py#L235-L312)) and the OB build
([smc_radar.py:1288-1346](../smc_radar.py#L1288-L1346)).

### FROZEN — identity (never assigned in refresh; defines *which* OB)
`zone_id`, `first_seen_iso`, `first_seen_label`, `ob_timestamp`, `direction`,
`bos_tag`, `bos_tier`.

### FROZEN — birth facts (backfill-only, locked at formation)
| field | line |
|---|---|
| body_ratio | zone.py:259-260 |
| walkback_depth | zone.py:261-262 |
| h1_atr (atr_at_ob) | zone.py:284-285 |
| sweep_v2 | zone.py:295-296 |
| bos_timestamp | zone.py:309-310 |

### LIVE — genuinely changes every bar
| field | line | why |
|---|---|---|
| last_seen_iso / last_seen_label | zone.py:244-245 | clock |
| is_new_this_scan | zone.py:246 | first-scan flag |
| bos_idx / ob_idx / impulse_start_idx | zone.py:265-267 | **position in the rolling 150-bar window** — same candle, new slot each bar |
| touches | zone.py:274 | grows |
| status_label / status | zone.py:275 / mitigation pass smc_radar.py:1347 | changes with price |
| current_price_at_scan | zone.py:286 | price now |
| distance_to_proximal_pips | zone.py:287-288 | moves with price |
| drop_reason | drop path | lifecycle |

### ⚠️ GREY ZONE — re-read every scan, but value is a birth fact that never changes
This is the crux. These are **birth facts** that are mechanically re-read (not
locked). Safe today because they track a fixed candle/leg, but they *could* drift on
a rolled/truncated window — the exact class the explicit freezes were added to kill.
| field | line | tied to |
|---|---|---|
| proximal_line / distal_line / high / low / ob_body | zone.py:248-252 | fixed OB candle |
| median_leg_body | zone.py:253 | impulse leg |
| impulse_start_price / bos_swing_price | zone.py:268-269 | fixed candles |
| bos_sequence_count | zone.py:271 | the break run |
| break_quality | zone.py:273 | fixed confirm candle + **birth ATR** (verified: smc_radar.py:1310 uses `h1_atr_for_leg` == the frozen `h1_atr` at smc_radar.py:1333) |
| broken_was_wall / reversal_pct | zone.py:307-308 | the break |

### RE-STAMPED by design (re-read AND meant to update)
`fvg` (zone.py:289), `sweep_observed` / v1 (zone.py:290), `dealing_range`
(zone.py:297), `role` (zone.py:311).

---

## 2. Root cause — "recompute then discard" (one cause, two symptoms)

- **Backtest** rebuilds the ENTIRE OB from scratch every bar inside detection —
  sweep, swings, break_quality, geometry — then the replay merge keeps **only fvg**
  ([replay_engine.py:418-429](../backtest/replay_engine.py#L418-L429)). So every
  frozen birth fact is rebuilt ~150× per OB and thrown away 149×.
- **Speed symptom:** that discarded recompute is the backtest slowness.
- **Correctness symptom:** recomputing on a rolled window is the drift risk (grey zone).
- **Fix once, fix both:** stop recomputing what is already discarded.

---

## 3. The fixes (all land BEFORE the next big backtest run)

### FIX A — reuse frozen birth facts for known OBs (speed + correctness)
- **What:** in the backtest replay, when an OB's `ob_timestamp` is already on the
  slate, skip recomputing its frozen birth facts (sweep v1+v2, swings, break_quality,
  geometry) and reuse the stamped values. Still recompute the genuinely-LIVE fields
  (window indices, touches, status, distance).
- **Scope:** **BACKTEST-REPLAY-ONLY. Live detection path untouched.** Live cost is
  irrelevant (hourly); this guarantees zero live drift.
- **Why byte-identical:** the replay already discards these recomputes
  (replay_engine.py:427), so reusing the frozen value changes nothing USED.
- **Proof obligation:** run the equivalence harness — alerts + every frozen column on
  filled trades must be sha-identical before/after, including a targeted old-OB /
  left-window-edge case (where a recompute is most likely to differ).
- **Also ship:** the safeguard test (below).

### FIX A-guard — the trust mechanism (safeguard test)
- **What:** one offline test that fails loudly if any FROZEN / birth field ever
  changes after formation, and if any LIVE field ever goes stale. Lives OUT of the
  live path (CI/offline only).
- **Why:** you stop trusting my labels and start trusting the system. This bites the
  moment a birth fact drifts.

### FIX B — retire sweep v1 ENTIRELY; v2 is the sole sweep (user pointer #6)
**STATUS: SHIPPED 2026-07-24.** v1 (`observe_phase1_sweep` / `sweep_observed` +
helpers `_equal_levels_score` / `_sweep_tier` / `_prior_trading_day_hl` /
`_compute_context_tags` + `SWEEP_SCORE_*` consts) DELETED. Every consumer (score
leg, `_count_confluences` OB2 rank, `classify_setup` badge [presence-only, any
tier], chart overlay, email narration, `sweep_present` col, zone plumbing, replay
`known_frozen` reuse) repointed to `ob['sweep_v2']`. Scoring decision (user):
v2 CONTRIBUTES to the current score (it already was the score input; v1 removal
finished it). TRUTH_LEDGER updated same change. Behaviour-changing → owns a fresh
baseline, NOT byte-identical.

- **User pointer:** *"Fix B is not just deleting sweep from scoring. V1 should not
  exist. Only V2 should exist and contribute."* (plus: the current score is not read
  and will be rebuilt after the backtest.)
- **What:** DELETE the v1 detector (`sweep_observed` / `observe_phase1_sweep`) and all
  its plumbing. `sweep_v2` becomes the **single** sweep source that feeds **every**
  consumer v1 used to feed. No v1 fallback, no "kept for parity."
- **Consumer sweep (do FIRST, exhaustively):** v1 currently feeds more than the score
  — OB2 confluence ranking + chart overlays + the score leg
  ([TRUTH_LEDGER.md:219](../TRUTH_LEDGER.md); `_count_confluences` / `sweep_pts` in
  smc_radar.py; `sweep_present` in h1_only_simulator.py:1845). Enumerate EVERY reader
  of `sweep_observed` (grep) and repoint each to `sweep_v2`, or remove it. Nothing may
  silently read a now-deleted field.
- **Why v2 can replace it:** v2 is a superset — v1's swing read is v2's SW tier
  ([liquidity_sweep.py:17-19](../liquidity_sweep.py#L17-L19)). v2 was parked
  observe-only mid-migration; this finishes the migration.
- **OPEN DECISION (confirm with user before wiring):** does v2 contribute to the
  *current* score now, or is sweep left out of the current score pending the scoring
  rebuild? User said both "delete from scoring" and "v2 should contribute" — resolve
  explicitly, do not guess.
- **This is BEHAVIOR-CHANGING** — v2 grades differently from v1 → alerts/ranking move
  → needs its own fresh baseline run. Do NOT bundle with Fix A's byte-identical proof.

### FIX D — canonical frozen/live/re-read list + consumer awareness (user pointer)
- **User pointer:** *"We need a documented list of frozen, live and re-read. And
  consumers should know that this list exists."*
- **What:** promote the §1 map to a canonical, discoverable list — the single place
  that answers "is this OB field frozen at birth, live, or re-read?"
- **Anti-stale (mandatory):** the list MUST be kept honest by a test that reads the
  actual freeze rules out of `zone.py` (`Zone.refresh`) and fails if the list and the
  code disagree. A hand-maintained MD that drifts is the exact trap CLAUDE.md forbids.
- **Consumer awareness:** add a short pointer comment at the source of truth
  (`zone.py`, near `_FIELD_ORDER`) and at the main OB-field consumers
  (`h1_only_simulator._build_row`, Phase 2 email build, `smc_radar` OB build) —
  "field freeze/live/re-read classification: see <list>." One line each, no logic.
- **Complexity:** LOW, no behaviour change. Ships with Fix C.

### FIX C — at-fill twin column for D1/W1 (user pointer #7)
- **User pointer:** *"At-fill drives the insights. Any function / Python / Claude /
  human who reads it must NOT confuse the at-fill column with the alert-time D1/W1
  column."*
- **What:** add a fill-anchored twin of the D1/W1 level reads, computed at `fill_ts`
  in the row build. Same level-read function (`levels_at` already takes an as-of
  timestamp), different timestamp.
- **NAMING (hard rule):** alert-time columns keep an explicit `*_at_alert` marker;
  fill-time columns MUST carry `*_at_fill`. Never a bare name. A one-line comment at
  the column source states which anchor and that at-fill is analysis-only.
- **LEAKAGE GUARD (hard rule):** at-fill columns drive **insights only** and NEVER
  enter scoring/selection/the alert decision — fill-time data cannot inform an
  alert-time choice. Keep them out of every screen.
- **Complexity:** LOW — isolated to row-build; touches no detection, live, or scoring.

---

## 4. Sequence & guardrails

1. **Now (this chat):** this doc. No behavior change.
2. **Backtest-speed chat — Fix A + A-guard.** Byte-identical, backtest-only, live
   untouched. Closes the speed topic. Prove sha-identical before trusting.
3. **Non-speed chat (SAFE bundle) — Fix D + Fix C.** Canonical frozen/live/re-read
   list + sync test + consumer pointers, then the at-fill twin column. No behaviour
   change; land + verify these before Fix B.
4. **Sweep-retirement chat — Fix B.** Behavior-changing (delete v1, v2 becomes sole
   sweep + contributor); exhaustive v1-consumer sweep first; own fresh baseline run.

**Split rationale:** Fix D+C are byte-safe (nothing scored/gated changes). Fix B moves
alerts and ranking, so it is isolated and gets its own baseline — never bundled with
work whose whole value is a byte-identical proof.

**Guardrails (CLAUDE.md):** live parity is sacred — Fix A must not touch live output;
Fix B is the only behavior change and needs a fresh baseline; no 18-yr run without an
explicit ask; every new/removed column updates `TRUTH_LEDGER.md` in the same edit.

**Also open (separate topic):** the two already-applied, proven byte-identical speed
fixes from earlier (within-bar frame reuse in smc_radar/liquidity_sweep; binary-search
`locate_ob_candle_idx` in smc_detector) — 46% faster, tests green, **uncommitted**.
Decide whether to commit them alongside Fix A.
