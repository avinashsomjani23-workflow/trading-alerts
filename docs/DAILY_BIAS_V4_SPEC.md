# DAILY BIAS V4 — pool-anchored day context (design spec, NOT yet approved to build)

**Written 2026-07-05, cold-start per the trader's instruction. Supersedes the detection
DESIGN in `DAILY_BIAS_HANDOFF_V3.md` §4 (the engine-reuse approach); V3's DATA findings
(§5, §6, §11) remain valid and binding. No code exists for any of this — verified: zero
PDH/PWH references anywhere in `*.py`.**

---

## 0. The autopsy — what the null result actually killed (and what it didn't)

The V3 verdict "no daily-bias selection edge" is real, but it tested ONE construct:

- **What was tested:** D1 **swing-trend direction** from `compute_structure` —
  formation lag median **14 D1 bars**, confirmation lag median **31 D1 bars** after the
  true extreme (V3 §5, re-confirmed §11).
- **Why that construct was always going to fail as a bias:** a label that flips 14-31
  days after the turn is stale for most of the move. "With-bias" trades under it are
  dominated by late-trend entries — exactly where a vet stops trusting the trend. The
  null is not surprising in hindsight; it condemns the LAG, not the concept.
- **What a vet's daily bias actually is** (and ICT teaches): short-horizon, formed from
  yesterday's/last week's levels — did we break and hold PDH, or sweep it and close
  back? Is there an unswept draw above or below? None of that was in the tested
  construct.

**Settled and carried forward (do not relitigate):**
- D1 swing-trend direction as a selection filter: **dead** (two independent tests).
- Direction FORECASTING of any kind: dead (50% over 18y).
- The "ranging" stall flag: dead.

**Untested (this spec's subject):** pool-anchored day context. Levels, sweeps, breaks,
draw — facts, not forecasts. This is the construct that matches the methodology, and it
has never touched the data.

**Honesty line:** SMC says pools are real signals; the data has neither confirmed nor
denied — because it was never asked. Per G2, the V3 null vs SMC intuition is exactly a
"detector mismatch" discussion point, and this spec is the answer to it. No edge is
claimed until the machine confirms it (F-rule thresholds). If it comes back null too,
the layer dies for good and that verdict is final.

## 0b. Why NOT reuse `compute_structure` this time

- V3's reuse argument was language cohesion: D1 and H1 disagreeing as an artifact.
  Valid for a swing-trend bias — but the swing-trend bias is dead, so the argument
  now defends a corpse.
- Pools need no swing engine at all: PDH/PDL/PWH/PWL are `max/min` of fixed calendar
  windows. Stateless, deterministic, no path-dependent state to persist/replay, no
  lookback/leg-filter/lag knobs to mis-tune. The simplest correct mechanism wins.
- One optional bridge remains: a single observational `d1_trend_dir` column from
  `compute_structure` (buffer off, lb3) so the engine can falsify the old construct
  once more on 18 years. One column, zero behaviour. Trader may drop it.

---

## 1. What gets built (three small, separable pieces)

### 1.1 Pool builder (stateless, pure)

- **Inputs:** `backtest/mt5_data/<INST>_D1.csv` + H1 CSVs (MT5, 2008+). Day boundary =
  MT5 server midnight = **fixed 21:00 UTC** (proven to reproduce MT5 D1 from H1 100%).
  W1 = calendar-week roll-up of those D1 bars (proven lossless within the feed).
  **Never resample from a second feed; never chase NY-close DST (V3 §7).**
- **Pools, per instrument, at any H1 timestamp `t` (point-in-time, strictly prior
  data only):**
  - `PDH`/`PDL` — high/low of the last COMPLETED trading day before `t`.
  - `PWH`/`PWL` — high/low of the last COMPLETED week before `t`.
- **Status machine, evaluated on closed H1 bars since the pool was born:**
  - `intact` — never traded through.
  - `swept` — wick pierced the level but the same H1 bar CLOSED back on the origin
    side, and no later bar has broken it. (Fuel for reversal; the pool is spent.)
  - `broken` — H1 close beyond the level, held by the NEXT H1 close (N=1 confirm —
    justified by the measured 27% first-break fakeout rate). Expansion through the
    level; the pool is spent.
  - Precedence: `broken` overrides `swept` (a sweep followed by a genuine break is a
    break). Once spent, a pool never revives (a day's pool lives at most until its
    replacement is born; weekly pools live the week).
- **Parameters: none beyond N=1.** No ATR fuzz on the level (V3 §7: fuzz corrupts the
  sweep-vs-break distinction). Wick pierce = any trade through; no minimum pierce
  depth in v1 (add only if the data demands — pre-register first).

### 1.2 Day-state classifier (small, fixed vocabulary)

Derived per instrument per H1 bar from the four pool statuses. Exactly one of:

| state | definition |
|---|---|
| `INSIDE` | PDH and PDL both `intact` (price inside yesterday) |
| `EXPANSION_UP` | PDH `broken`, PDL `intact` |
| `EXPANSION_DOWN` | PDL `broken`, PDH `intact` |
| `SWEPT_HIGH` | PDH `swept` (not broken), PDL `intact` |
| `SWEPT_LOW` | PDL `swept` (not broken), PDH `intact` |
| `BOTH_SIDES` | both PDH and PDL spent in any way (chop / big-range day) |

- This replaces every dead abstraction: no trend label, no stall counter, no neutral
  bucket. `SWEPT_*` IS "exhausted = sweep-not-break" made concrete; `INSIDE` is real
  containment-ranging (the thing the stall counter failed to measure).
- Weekly pools do NOT get their own state machine vocabulary — they enter through the
  tier and DOL fields below (keep the state space small; 6 states × 5 instruments is
  already at the C3 thin-bucket edge).

### 1.3 Draw-on-liquidity (DOL) + per-trade features

Logged per trades.csv row, all stamped AT ALERT from strictly-prior bars:

| column | definition |
|---|---|
| `day_state_at_alert` | §1.2 state |
| `pdh_status_at_alert` / `pdl_status_at_alert` | intact / swept / broken |
| `pwh_status_at_alert` / `pwl_status_at_alert` | intact / swept / broken |
| `dist_next_pool_above_atr` / `below_atr` | distance to nearest UNSPENT pool, H1-ATR units (None if none) |
| `next_pool_above_tier` / `below_tier` | `PW` / `PD` (weekly outranks daily) |
| `trade_toward_pool` | does the trade's direction point at the nearest unspent pool? (bool; None if no pool either side) |
| `last_sweep_age_h1` / `last_sweep_tier` | bars since the most recent sweep event, and PW/PD |
| `d1_trend_dir_at_alert` | optional bridge column (§0b) — observational only |

- **Truth-ledger gate applies (B5):** each column ships with a ledger row
  (source file:line, stamp timing, population), a disposition row in
  `EDGE_ENGINE_SPEC.md` §12, and one structural guard for the class:
  a **no-look-ahead regression test** — synthetic H1/D1 series where the pool value at
  `t` provably requires only bars `< t`; plus a determinism check (same CSVs → same
  columns, twice).
- Storage is free; a missing column is a permanent blind spot (CLAUDE.md logging rule).

---

## 2. How its value gets judged (no ad-hoc study this time)

- **Wire the columns as LOGGING ONLY** — no score term, no gate, no email change
  beyond (optionally) one informational line. Prove P&L bit-identical to a pre-change
  short run on everything except the new columns.
- **Land BEFORE the 18-yr baseline run.** This is the whole timing argument: B6 freezes
  the feature list at first engine run. In the baseline, the engine's own
  discovery/validation split judges the pool features with the same F-rule bar as
  everything else: repeats in validation, same direction, ≥60% of quarters, ≥0.10R,
  ≥150 trades per bucket. No bespoke daily-bias study, no hand-slicing, no C5 risk.
- **Pre-registered hypotheses** (so nobody invents them after seeing tables):
  1. `SWEPT_*` against the trade direction (trade shorts after `SWEPT_HIGH`, longs
     after `SWEPT_LOW`) outperforms the base rate.
  2. `trade_toward_pool=True` with an unspent weekly-tier pool ahead outperforms
     `False`.
  3. `BOTH_SIDES` days underperform (the real "ranging" filter, if any exists).
  Anything else the engine surfaces is a v2 candidate, not a conclusion (C2).
- **Kill rule:** if none of the three survive validation at F thresholds, the daily
  layer is DEAD in full — no v5, no "maybe with different windows". The columns stay
  (they're free and honest); the idea stops.

---

## 3. Explicitly out of scope

- **Trade-setup sweep rebuild** (`SWEEP_REBUILD_HANDOFF.md`, guardrail A5) — SEPARATE
  workstream, per the trader 2026-07-05. It later CONSUMES `_sweep`/tier facts from the
  pool layer, but nothing here reads or edits `observe_phase1_sweep`.
- Live/email wiring of pools (waits on the Twelve Data daily-boundary fetch test, V3 §7).
- Session pools (Asia/London), BTC, NAS100, monthly levels (dropped — settled), any
  LLM involvement, any M5/M15 anything.
- Any gating/scoring — v1 is observation, period.

## 4. Build order

1. `pool_builder.py` (pure functions + unit tests incl. the no-look-ahead guard).
2. Day-state + DOL derivation on top (pure; same tests).
3. Backtest wiring: stamp the §1.3 columns at alert in the replay/simulator path;
   ledger rows + disposition rows; short-run P&L-identity proof.
4. Sequencing gate: land AFTER the live-bugs fixes (`LIVE_BUGS_FIX_SPEC.md`), BEFORE
   the 18-yr baseline. All three land as pre-declared changes ahead of the run (B1/B6).
5. Baseline runs → engine screens → F-rule verdict → ship / observe / kill.
