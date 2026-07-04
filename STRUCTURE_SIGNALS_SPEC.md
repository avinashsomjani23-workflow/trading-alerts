# STRUCTURE SIGNALS SPEC — Verify + Log the Four Structure Reads

**Created 2026-07-04. Executor: Opus. Reviewer: trader.**
Companion to `SIGNAL_CANDIDATES.md` (the why) and `DECISION_GUARDRAILS.md` (the rules).

## Goal

Surface the four structure reads the engine already computes — trend, pending flip,
ranging, leg verdict — into trades.csv so the edge engine can grade them, plus two
derived measurements (leg retracement, broken-wall PD). One small logic fix included
(S1). **No scoring, gating, filtering, alerting, or email behaviour changes.**

## Verification results (done 2026-07-04, live code trace — this spec's foundation)

| Read | Verdict | Evidence |
|---|---|---|
| Trend state machine | SOUND | Birth (`dealing_range.py:1173-1208`), flip only on Confirmation BOS (`:958-987`, `:1007-1035`), reclaim cancel (`:943-957`, `:992-1006`), continuation-BOS cancel of pending CHoCH (`:1141-1145`, `:1165-1169`), whipsaw guard `rearm_block_dir`, gap guard `_traded_through`, body gates. No premature-flip path exists. |
| Pending flip | SOUND | Armed at CHoCH (`:1063`, `:1094`), cleared on all three exits (reclaim / confirm / continuation BOS). Exposed as `flip_unconfirmed` + `choch_pending_dir` (`:1319-1320`). |
| Ranging flag | **DEFECT — S1** | `trend_dir_swings_since_extend` is NOT reset in either Confirmation-BOS flip branch. A stale counter (≥2) survives the trend flip, so a freshly confirmed reversal can be born labelled "ranging" until its first HL/LH confirms. Display-only today (label suffix), becomes a data defect the moment we log it. |
| Leg verdict (bos_verdict) | SOUND | `smc_detector.bos_leg_read` (`smc_detector.py:584-636`): pure, shared live+backtest, anchor-on-first-body decay rule, missing body → 'holding' (never claims unmeasurable exhaustion). Backtest stamps at fire + carries as payload scalar (T1 pattern, `replay_engine.py:567`, `:615`). |

Wiring verified: backtest walls = `smc_radar.compute_pair_walls` (`replay_engine.py:263`),
identical to live; the packed dict carries `structure_v2` (ranging, flip_unconfirmed,
choch_pending_dir) at `smc_radar.py:1284`. The frozen `ob["dealing_range"]` snapshot
(`smc_radar.py:1058-1066` → `smc_detector.get_dealing_range:407-416`) does NOT carry
per-wall broken flags today — only `tentative` (either wall). S4 adds them.

Minor observation, NOT in scope: on HL/LH confirmation the leg extreme resets to the
confirming bar's high/low (`dealing_range.py:1250`, `:1266`), skipping the `lookback`
bars between pivot and confirmation. Slightly understates the reclaim level. Cosmetic;
park it — any change there is a detection change (D1/D2 evidence required).

## Guardrail compliance

- **B6 window is open:** feature list is legal to extend — the baseline will be re-run
  for the Stage-0 fixes and no engine run has consumed features yet. This is the last
  cheap moment; that is why this ships now.
- **B5:** every new column needs (1) a TRUTH_LEDGER.md row, (2) a structural guard,
  (3) a disposition row in EDGE_ENGINE_SPEC.md §12. All three are items below, same
  change.
- **S1 is a logic fix, not a detection change:** it alters only the `ranging` output
  flag and label suffix. Events, trend, OBs, alerts, trades are untouched. The
  structure golden baseline may shift where labels carried "(ranging)" — the
  pre-commit hook re-baselines (gen_fixtures); expect and accept that diff, verify it
  is label/ranging-only.
- **A5:** sweep untouched.
- **Emails untouched:** no verdict text ships before Stage 1 validation
  (SIGNAL_CANDIDATES.md rule).

---

## S1 — Fix: reset the ranging counter on trend flip

**File:** `dealing_range.py`

- In BOTH Confirmation-BOS branches (UP flip ~`:969-979`, DOWN flip ~`:1017-1027`),
  add `trend_dir_swings_since_extend = 0` alongside the other leg re-seeds
  (`leg_extreme_*`, `leg_start`).
- Do NOT touch the CHoCH_FAILED / reclaim paths — there the old trend resumes and the
  stale count legitimately carries.
- Ranging semantics after fix (document in the docstring where `ranging` is computed,
  ~`:1277`): `ranging` = trend defined AND ≥ STRUCTURE_RANGING_STALE (=2) consecutive
  counter-trend swings confirmed without a trend extension (no new HL/LH `defended`
  reset, no continuation BOS) — counter starts at 0 on birth, extension, continuation
  BOS, and (new) trend flip.

**Guard (test):** extend the structure tests with a crafted candle sequence:
downtrend → two non-extending swing highs (counter ≥2, `ranging` True) → CHoCH up →
Confirmation BOS up → assert `structure_v2["ranging"]` is False on the flip bar and
stays False until two fresh non-extending lows confirm in the NEW trend. Also assert
trend/events output is byte-identical to pre-fix for the same input (proves
observe-only blast radius).

## S2 — Log the structure state at alert (3 columns)

**New trades.csv columns:** `structure_ranging_at_alert` (bool),
`flip_pending_at_alert` (bool), `flip_pending_dir_at_alert` (`bullish`/`bearish`/None).

- **Source:** at the fire block in `backtest/replay_engine.py` (~`:560-598`), read
  `walls.get("structure_v2") or {}`: keys `ranging`, `flip_unconfirmed`,
  `choch_pending_dir`. Map pending dir `up`→`bullish`, `down`→`bearish` (same map as
  `smc_detector.py:695`).
- **Stamp timing:** alert fire time, **payload scalars in the yield dict** (T1
  pattern — never stamped on the shared `ob` dict; rows are built post-walk and the
  next fire would overwrite). Add next to `h1_trend` / `trend_alignment` in the yield
  (~`:609-610`).
- **Row build:** `backtest/h1_only_simulator.py:_build_row` — read from `alert.get(...)`
  only, mirror how `trend_alignment` is read (`:1328`). Add to the STAMPED-AT-ALERT
  bucket in the FIX-3e classification comment (`:1053-1068`).
- **Population:** every row. `None` only if `structure_v2` missing (degraded walls).
- **Live is out of scope:** columns are backtest/edge-engine inputs. Live email wiring
  happens only after Stage 1 validates (SIGNAL_CANDIDATES rule).

**Guard (test):** re-fire freeze test, same pattern as `tests/test_ob_alert_freeze.py`:
one zone fires twice with structure state changed between fires (e.g. flip_pending
False → True); assert the first trade row carries the FIRST fire's values. This kills
the last-fire-stamp bug class for these columns.

## S3 — Log leg retracement at alert (3 columns)

**New trades.csv columns:** `leg_extreme_at_alert` (float),
`leg_retrace_pct_at_alert` (float 0–100+, uncapped), `leg_extreme_clipped` (bool).

- **Definition (stamp at alert, with the planned entry price — the a-priori read a
  trader would have when the alert lands):**
  - bullish OB: `leg_extreme = max(High)` of closed bars in `h1_slice` with
    `ts >= ob_timestamp`; `retrace_pct = (leg_extreme − entry) / (leg_extreme −
    impulse_start_price) × 100`.
  - bearish OB: `leg_extreme = min(Low)` same window; `retrace_pct = (entry −
    leg_extreme) / (impulse_start_price − leg_extreme) × 100`.
  - `entry` = the limit price the simulator will place (proximal or midpoint per
    `entry_zone`) — so compute `retrace_pct` in `_build_row` (which knows `entry`),
    from `leg_extreme_at_alert` carried as a **payload scalar** stamped in the replay
    fire block (which has `h1_slice`, ~`replay_engine.py:228` and `:560`).
- **Edge cases (all must be handled, all → `None` retrace, never a crash):**
  - `impulse_start_price` missing/None on the OB.
  - Degenerate denominator ≤ 0 (extreme not beyond impulse start).
  - `ob_timestamp` older than the 150-bar slice start → compute over the available
    slice AND set `leg_extreme_clipped = True` (extreme may be understated; the flag
    keeps the column honest). Otherwise `False`.
  - Retrace > 100 is VALID (price beyond the leg origin) — do not clamp.
- **Population:** every row; `None` per edge cases above.

**Guard (test):** unit tests on the retrace math: long normal (~50%), short normal,
degenerate denominator → None, missing impulse_start → None, clipped window sets the
flag, >100% not clamped. Plus the S2 re-fire freeze test extended to
`leg_extreme_at_alert` (second fire sees a higher extreme; first row must keep the
first-fire value).

## S4 — Log broken-wall PD flags at OB formation (2 columns)

**New trades.csv columns:** `dr_ceiling_broken_at_ob` (bool/None),
`dr_floor_broken_at_ob` (bool/None).

- **Source chain:** `h4_range.compute_h4_range` already emits `ceiling_broken` /
  `floor_broken` (`h4_range.py:233-234`) → `compute_pd_position` maps them to
  `ceiling_is_placeholder` / `floor_is_placeholder` in the h4_live branch
  (`dealing_range.py:469-479`) → **but** `get_dealing_range` drops them, returning
  only `tentative` (`smc_detector.py:407-416`).
- **Change:** additively extend the `get_dealing_range` valid-branch return with
  `"ceiling_broken": bool(pd_info.get("ceiling_is_placeholder", False))` and
  `"floor_broken": bool(pd_info.get("floor_is_placeholder", False))`. Additive keys
  only — no existing key changes, no consumer changes. The legacy/fallback branches
  set both to `None` (unknown).
- **Stamp timing:** these land in the frozen `ob["dealing_range"]` snapshot taken at
  OB build (`smc_radar.py:1058-1066`) — immutable after build, so `_build_row` may
  read them straight off `ob.get("dealing_range")` (same bucket as the existing
  FROZEN-BY-DESIGN `dealing_range` fields in the FIX-3e comment).
- **Population:** every row; `None` when the snapshot is invalid/legacy.
- **Why both walls:** for a long the meaningful flag is the floor, for a short the
  ceiling — but log both raw and let Stage 1 decide; deriving "relevant wall broken"
  in SQL is trivial, un-logging is impossible.

**Guard (test):** snapshot test — build an OB while a wall is broken (riding live
extreme), assert the flags land in `ob["dealing_range"]` and survive to the row; and
the inverse (intact range → both False).

## S5 — Bookkeeping (same change, not optional)

- **TRUTH_LEDGER.md:** one row per new column (8 total): source file:line, stamp
  timing (alert payload / OB-build snapshot), population, None-conditions. Follow the
  existing `trend_alignment` row format (`TRUTH_LEDGER.md:104`).
- **EDGE_ENGINE_SPEC.md §12:** one disposition row per new column. Proposed:
  `structure_ranging_at_alert`, `flip_pending_at_alert`, `flip_pending_dir_at_alert`,
  `leg_retrace_pct_at_alert`, `dr_*_broken_at_ob` → Stage 1 screen candidates;
  `leg_extreme_at_alert`, `leg_extreme_clipped` → audit/derivation support, not
  screened. A column with no disposition is a Stage-0 FAIL (B5).
- **`backtest/diagnostics/edge_engine.py:135` feature list:** add the five screen
  candidates. Do NOT add the two support columns.
- **Aggregator/reporting:** no changes. These columns ride trades.csv as-is; the
  email/report overhaul is a separate workstream.

## S6 — Acceptance checklist (Opus runs before handing back)

1. All new/changed tests green, including existing `tests/test_ob_alert_freeze.py`
   and the structure golden gate (re-baseline expected for S1; diff must be
   ranging/label-only — inspect it, state it in the handback).
2. Short proof run (~2–3 months, 2008) — per B2 style: all 8 columns present, sane
   values (ranging/flip booleans vary; retrace mostly 20–120; clipped rare; dr flags
   vary), no new None-floods, row counts unchanged vs a pre-change run of the same
   window (proves observe-only).
3. Determinism spot-check: same window run twice → identical trades.csv.
4. Grep proof there is NO new read of mutable `ob` state at row-build time (the FIX-3e
   rule): the only new `_build_row` inputs are `alert.get(...)` payload scalars and
   the frozen `ob["dealing_range"]`.
5. TRUTH_LEDGER + EDGE_ENGINE_SPEC §12 rows in the same commit as the code.

## Explicitly OUT of scope

- Any email/report content change (validation first).
- Any scoring, gate, filter, or alert behaviour change.
- Sweep detector (A5), leg-extreme lookback nuance (parked above), dealing-range
  redefinition (generation-2 question, needs Stage 1 evidence).
