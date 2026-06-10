# FABLE REFERENCE — Backtest Diagnostic Harnesses

This file is the **ground truth** about a live SMC trading-alert codebase. You
(Fable) cannot see the repo. Everything below was read line-by-line from the
actual source and verified. Design against THIS, not against the brief's
assumptions — where the two conflict, this file wins (see §1).

Your job is to produce **methodology, architecture, an implementation plan, and
a detailed implementation spec** (pseudocode + edge cases + dos/don'ts) for
three diagnostic harnesses. A separate engineer (Claude Code, with repo access)
will implement from your spec. So your spec must be precise enough to implement
without guessing, but you do NOT need to emit final runnable Python.

---

## 0. System in one paragraph

Automated Smart Money Concepts (SMC) alert system. **H1 timeframe only** (M5/M15
and "Phase 3" are dormant — ignore them). 6 instruments: EURUSD, NZDUSD, USDJPY,
USDCHF, GOLD (XAUUSD), NAS100. Pipeline: **Phase 1** detects H1 market structure
(swings → trend → BOS/CHoCH) and builds Order Blocks (OBs); **Phase 2** fires an
alert when price approaches an OB within a proximity band, scores confluences,
and defines entry/SL/TP. A **backtest engine** replays history bar-by-bar through
the *same live Phase-1/Phase-2 functions* and simulates trades. The user does not
trust the backtest: he suspects look-ahead bias and silent divergence from live.

**Hard constraints for everything you design:**
- Change ZERO trading logic. Harnesses are new files only (under
  `backtest/diagnostics/`). They import and CALL the live functions; they never
  re-implement detection.
- Runs on **Python 3.14**, invoked from **PowerShell on Windows**.
- Output must be human-readable (CSV / Markdown table / console) the user can
  read without an assistant.
- Every P&L number must reconcile to `r_realised` (see §6). Never invent a P&L
  formula.

---

## 1. CORRECTIONS TO THE BRIEF (read first — the brief is partly wrong)

The brief you were given contains two factual errors. Both were verified against
source:

**1a. `atr_multiplier` in config.json is the alert-PROXIMITY cap, NOT a swing
multiplier.** It is consumed in exactly two places, both as
`prox_cap = pair_conf["atr_multiplier"] * h1_atr` — the band within which price
must approach an OB's proximal line before an alert fires:
- `Phase2_Alert_Engine.py:1934` (live)
- `backtest/replay_engine.py:343` (backtest)

It does **not** influence swing detection, structure, or OB building. The swing
size filter is a *different* constant, `MIN_LEG_ATR_MULT = 1.5`
(`dealing_range.py:70`), which is identical in live and backtest.

**Consequence you must bake into harness design:** the backtest calls the same
live detection functions, so **swings, structure, BOS, CHoCH, and OBs are
detected identically in backtest and live.** A sweep over the proximity cap
(`atr_multiplier` / `BACKTEST_ATR_MULT`) changes **only** which OB-touches fire
an alert and therefore P&L — the swing/OB/CHoCH/BOS counts are INVARIANT to it.
A sweep over `MIN_LEG_ATR_MULT` (or the other structure knobs) changes
everything downstream. Classify every knob by *what it can possibly move* (see
§3) or your comparison tables will show columns that look broken (all-identical)
and the user won't know if that's a bug or the truth.

**1b. The live config values changed.** `config.json` `atr_multiplier` is now
**2.5 (forex) / 3.0 (index + commodity)**, NOT 4.0/4.5 as the brief states.
`BACKTEST_ATR_MULT` is `{forex: 3.0, index: 3.5, commodity: 3.5}`
(`run_backtest.py:114`). So **today the backtest proximity band is WIDER than
live** (3.0 vs 2.5 forex; 3.5 vs 3.0 index/commodity) → the backtest fires
alerts *earlier / more often* than live would. This is a genuine
backtest≠live divergence, but it is an **alert-timing** divergence, not a
structure-detection one. Treat it as a real Harness-3 finding, but describe it
correctly.

---

## 2. Single-source-of-truth functions (call these; never re-derive)

Exact signatures and return shapes, verified from source. `df` is always a
UTC-indexed pandas DataFrame with columns `Open, High, Low, Close, Volume`; the
index is the bar OPEN time.

### Detection core
```
# smc_detector.py
compute_atr(df, period=14) -> Optional[float]
    # Mean True Range over last `period` bars. Memoised on an OHLC fingerprint
    # (global _ATR_CACHE). Pure for a given slice. Returns None if len<period+1.

get_swing_points(df, lookback=3, bounds=None, min_leg_atr_mult=1.5) -> list
    # Geometry (lb-3) swing detection, THEN delegates the ATR leg-size filter to
    # dealing_range._filter_swings_by_leg_atr. Each swing:
    #   {'type':'high'|'low', 'price':float, 'idx':int, 'ts': df.index[i]}
    # NOTE ts here is the raw index value (Timestamp), not an ISO string.

# dealing_range.py
detect_swings(df, lookback=3, min_leg_atr_mult=1.5) -> list
    # Same geometry + ATR filter. Each swing: {'type','idx','price','ts'(iso str)}.

_filter_swings_by_leg_atr(swings, df, period=14, min_mult=1.5) -> list
    # THE single ATR leg-size gate. A swing is kept only if the leg from the
    # previous kept opposite-type swing is >= min_mult * mean_TR_across_leg.
    # First swing passes free; same-type consecutive swings bypass; a dropped
    # swing does NOT advance the reference anchor.

compute_structure(df, h4_range, lookback=3, _min_leg_atr_mult=1.5,
                  choch_atr_mult=None, lock_atr_mult=None, _trace=None) -> dict
    # PURE H1 structure engine. Recomputes from the whole df each call (no
    # incremental state). Returns dict incl: 'state'(up/down/undefined),
    # 'trend'(bullish/bearish/None), 'choch'(bool), 'choch_flip_count'(int),
    # 'events'(ring of last 20 BOS/CHoCH dicts), 'swings'(confirmed pool), etc.
    # Knob fallbacks INSIDE the body read the module globals at call time:
    #   bos_disp   = BOS_ATR_MULT * atr
    #   choch_disp = (choch_atr_mult or STRUCTURE_CHOCH_ATR_MULT) * atr
    #   lock_dist  = (lock_atr_mult or STRUCTURE_LOCK_ATR_MULT) * atr

# h4_range.py
compute_h4_range(df, min_leg_atr_mult=1.5) -> dict
    # Resamples H1->H4 internally, detects H4 swings, returns the dealing-range
    # walls (confirmed_ceiling/floor etc.) consumed by compute_structure's PD gate.

# smc_radar.py  (the live Phase-1 assembler used by BOTH live and backtest)
compute_pair_walls(df, pair_name="") -> dict
    # Calls compute_h4_range(df) and compute_structure(df, _h4) with NO knob
    # args (all knobs use their internal defaults). Packs trend, walls, event
    # ring, swings into the 'walls' dict every consumer reads. Never raises.

detect_smc_radar(df, pair_type, events, walls, pair_name) -> dict
    # Builds OBs from the event ring. Returns:
    #   {"current_price": float, "active_zones": [ob, ...],
    #    "ob_build_diagnostics": [...]}
    # OB dict fields used downstream: proximal_line, distal_line,
    #   direction('bullish'|'bearish'), ob_timestamp, bos_timestamp,
    #   bos_tag, bos_tier, fvg(dict), sweep_observed(dict), dealing_range(dict).
```

### Backtest path
```
# backtest/data_loader.py
load_bars(symbol, interval, start, end, force_refresh=False) -> Optional[df]
    # yfinance + parquet cache (backtest/cache/). UTC-indexed. See §5 for limits.

# backtest/replay_engine.py
class ReplayState  # in-memory mirror of live JSON state; never writes live files
replay_pair(pair_conf, df_h1, state, walk_start_ts, walk_end_ts) -> generator
    # Walks H1 bars in window, yields event dicts: kind in
    #   {"ob_seen","alert","ob_mitigated"}. The "alert" events feed the simulator.

# backtest/h1_only_simulator.py
simulate_h1_only_dual(alert, pair_conf, df_h1, risk_usd=250.0) -> list[row]
    # Simulates BOTH entry zones (proximal + 50% midpoint) for one alert.
    # Returns 0/1/2 trade-row dicts. Row carries r_realised, r_if_exit_tp1,
    # r_if_exit_tp2, pnl_usd, exit_reason, mfe_r, mae_r, etc. (see §6).
```

---

## 3. ATR-knob inventory (verified file:line, value, scope, OVERRIDE MECHANISM)

The override mechanism is the make-or-break detail for the sweep harness. Some
knobs are read as a module global at call time (monkeypatch works). Others are
captured as a **default argument** bound at function-definition time —
monkeypatching the module constant does NOT change them, because the default was
already frozen. Those must be overridden by patching the function's
`__defaults__` (or passing the value explicitly through every call layer). The
live assembler `compute_pair_walls` passes NO knob args, so it always hits the
def-time defaults for the swing multiplier.

| # | Constant | file:line | Value (current) | Gates / SMC event | Phases | What it can move | Override mechanism |
|---|----------|-----------|-----------------|-------------------|--------|------------------|--------------------|
| 1 | `MIN_LEG_ATR_MULT` | dealing_range.py:70 | 1.5 | swing leg-size filter (the single H1 swing def) | P1 (+H4 range, all swing consumers) | swings, structure, BOS, CHoCH, OBs, alerts, P&L | **DEF-TIME DEFAULT** in `detect_swings`, `compute_structure(_min_leg_atr_mult=)`, `compute_h4_range(min_leg_atr_mult=)`. Monkeypatch of the constant does NOT flow through `compute_pair_walls`. Must patch `__defaults__` of all three OR drive a path that passes it explicitly. |
| 2 | `BOS_ATR_MULT` | dealing_range.py:75 | 0.4 | BOS displacement + Range-BOS wall proximity | P1 | BOS count, OBs, alerts, P&L | Read as module global at call time inside `compute_structure` → **monkeypatch `dealing_range.BOS_ATR_MULT` works**. |
| 3 | `STRUCTURE_CHOCH_ATR_MULT` | dealing_range.py:422 | 1.0 | CHoCH displacement | P1 | CHoCH count, OBs, alerts, P&L | None-fallback at call time → monkeypatch works, OR pass `choch_atr_mult=` to `compute_structure` (but `compute_pair_walls` doesn't, so monkeypatch is the only path through the live assembler). |
| 4 | `STRUCTURE_LOCK_ATR_MULT` | dealing_range.py:428 | 1.5 | CHoCH failure-window lock distance | P1 | CHoCH confirm/revert, downstream | Same as #3 (`lock_atr_mult=` param or monkeypatch). |
| 5 | `OB_MAX_RANGE_ATR_MULT` | smc_detector.py:177 | 2.0 | reject oversized OB candle (news spike) | P1 | OB count, alerts, P&L | Module global read at call time → monkeypatch works. |
| 6 | `FVG_NOISE_FLOOR_MULT` | smc_detector.py:149 | forex .08 / index .15 / commodity .12 | min FVG size to count | P1 (+P2/P3 dormant) | FVG presence/score, freshness, score | Dict `.get()` at call time → monkeypatch the dict works. |
| 7 | `SWEEP_EQUAL_LEVEL_TOLERANCE_ATR` | smc_detector.py:199 | forex .30 / index .40 / commodity .40 | "equal highs/lows" tolerance + context tags | sweep scoring | sweep score, P&L (via score only; H1-only has NO score gate, see §6) | Dict `.get()` at call time → monkeypatch works. |
| 8 | `SWEEP_WICK_PIERCE_MIN_ATR` | smc_detector.py:209 | forex .05 / index .08 / commodity .08 | wick must pierce swept level by this | sweep + `is_swing_active` | sweep detection, score | Dict `.get()` at call time → monkeypatch works. |
| 9 | `REARM_EXTRA_ATR` | replay_engine.py:153 | 1.0 | OB re-arm distance after a fire | backtest only | alert count (re-fires), P&L | Local var inside `replay_pair` (not a module global). Override = parametrise the harness's own copy of the walk, or patch the function. |
| 10 | `BACKTEST_ATR_MULT` | run_backtest.py:114 | forex 3.0 / index 3.5 / commodity 3.5 | overrides config `atr_multiplier` (proximity cap) in backtest | backtest only | **alerts + P&L only** (NOT swings/OBs/structure) | Local dict inside `_run_h1_only`; mutates `pair_conf["atr_multiplier"]` before the walk. |
| 11 | `atr_multiplier` (config) | config.json (per pair) | forex 2.5 / index+commodity 3.0 | LIVE proximity cap | P2 (live) + replay | alerts + P&L only | Per-pair config value. The proximity cap actually used in the walk = whatever `pair_conf["atr_multiplier"]` holds (overwritten by #10 in backtest). |

Knobs the brief did NOT list but exist (mention, don't necessarily sweep):
- `SWING_LOOKBACK = 3` (dealing_range.py:59) — the lb in "lb-3 geometry". Not an
  ATR knob but it's the OTHER half of the swing definition; Harness 2 sweeps it.
- `STRUCTURE_RANGING_STALE = 2` (swing-count, informational flag only).
- `epsilon = 0.0001 * tf_atr` inside `_rejection_score` — doji div-by-zero guard,
  not a tunable.

---

## 4. Slice discipline (the heart of the look-ahead problem)

The replay is built to feed each live call ONLY closed-bar history. Two helpers
enforce it (`replay_engine.py`):
- `_slice_closed_before(df, wall_clock_ts)` → returns `df[df.index < wall_clock_ts]`.
  yfinance H1 bars are OPEN-timestamped: a bar indexed `12:00` covers 12:00→13:00
  and is only known at 13:00. So at wall-clock T, closed bars are index < T.
- `_assert_no_lookahead(slice, replay_ts, tag)` → raises if `slice.index[-1] > replay_ts`.

**This is the single most important invariant for your harnesses.** Any harness
that recomputes detection must respect the SAME discipline, and you must decide —
explicitly, per harness — which of two questions it answers:

- **(A) "What exists in the full dataset at knob = X?"** Compute once over the
  whole final df. Fine for a static structural census (Harness 2's raw-vs-kept
  swing inspection). But a swing near the right edge is confirmed here that the
  live bar-by-bar system had not yet seen.
- **(B) "What would the live system actually have detected, bar by bar?"** You
  must walk bars and slice-closed-before each step, exactly like `replay_pair`.
  Slower, but it's the only honest answer for "alerts fired" / "P&L" / parity.

Never silently mix them. State which mode each table is in.

---

## 5. What the data loader can reach (don't fake output beyond this)

`data_loader.INTERVAL_MAX_DAYS = {"5m":58, "15m":58, "1h":720}`. For this project
only **1h** matters. yfinance 1h history is capped ~730 days; `load_bars` clamps
`needed_start` to `now - 720d` and logs `yfinance_clamp`. So:
- Usable timeframe: **H1 only.**
- Usable date range: roughly the **last ~2 years**, no earlier. A sweep "across
  date ranges" means sub-windows inside that span, not arbitrary history.
- Parquet cache in `backtest/cache/` makes repeated sweeps cheap (history is
  immutable; cache only refetches if the requested range exceeds cached range).
- `run_backtest` fetches with a 35-day warmup pad before `start`
  (`fetch_start = start - 35d`) so structure is warm at the walk start; the
  replay additionally requires `MIN_WARMUP = 50` closed H1 bars before it acts.

If a requested window can't be served, say so plainly in the harness output —
never emit fabricated rows.

---

## 6. How a "trade" is born + every existing look-ahead guard (audit these)

The brief's nightmare ("the system literally considered the event as the trade;
the trade always existed after the event on the retest to the OB") describes a
PRIOR engine. The current code CLAIMS to have fixed it. **Your Harness 3 must not
trust the comments — it must prove each guard holds on real data.** The guards,
verified present in source:

Replay (`replay_engine.replay_pair`):
1. Detection runs on `h1_slice = _slice_closed_before(df, h1_ts)` then
   `_assert_no_lookahead`. The forming bar is excluded.
2. Proximity is measured on the **just-closed** bar (`h1_slice.iloc[-1]`), wick-
   based: long OB → `just_closed_low - proximal`; short OB → `proximal - just_closed_high`.
3. Alert fires at `h1_ts` which is **strictly after** the just-closed bar, and a
   guard re-checks `h1_ts <= bos_ts` and blocks+logs `alert_lookahead_blocked`.
   So "trade the BOS candle" is structurally prevented.
4. Re-arm state machine: an OB fires once → `cooling` → re-arms only after price
   clears `(prox_cap + REARM_EXTRA_ATR) * ATR`. Stops hourly alert spam.
5. OB mitigation + a 15-day age cap drop dead OBs from the slate.

Run driver (`run_backtest._run_h1_only`):
6. **OB dedup**: only the FIRST alert per `(ob_timestamp, direction)` is
   simulated (`seen_obs`). Prior runs cloned 5–60 identical losers per OB.

Simulator (`h1_only_simulator._simulate_single_entry`):
7. Fill walk starts at `fill_walk_start = alert_ts + 1h` — the alert bar is NOT a
   fill candidate (the limit didn't exist while that bar was making its move).
8. Pending-limit fill: long fills on `bar.low <= entry`, short on `bar.high >= entry`.
9. **Fill-bar TP suppression**: on the bar the limit fills, TP is NOT credited
   (can't infer intra-bar fill→TP vs TP→fill order). SL on the fill bar IS taken
   (price had to pass entry first). This is the subtle one — audit it hard.
10. Same-bar SL+TP collision → **SL wins** (pessimistic).
11. Two independent clocks: pre-fill pend (≤48 bars) and post-fill hold (≤48
    bars). An earlier bug conflated them and produced false `window_end` exits.
12. Spread widens SL only (TPs not widened — pessimistic). Slippage/swap not modelled.

P&L source of truth (reconciliation contract — do not break):
- `r_realised` is the DEFAULT policy outcome (ride to TP2 with SL→BE after TP1).
- Every headline P&L / WR / expectancy must derive from `r_realised`; `pnl_usd =
  r_realised * risk_usd`. The `r_if_exit_tp1/_tp2` columns are hypotheticals used
  ONLY in their own explicit sentence — never in aggregates.
- The reporting layer asserts `headline_scoreboard == sum(pnl_usd for filled
  trades)`. Any P&L your harness reports MUST reconcile to the same `r_realised`.
- **H1-only mode has NO score gate**: every OB-touch fires regardless of
  confluence score (score is logged for later threshold discovery). So sweeping
  sweep/FVG knobs (#6,#7,#8) changes the *logged score* but does NOT change which
  trades exist in H1-only mode. Flag this — otherwise a sweep of those knobs will
  show identical trade counts/P&L and look broken. (It's correct: those knobs are
  P&L-neutral in the current no-gate H1-only backtest; they'd only bite if a score
  gate were turned on.)

---

## 7. My honest list of candidate divergence / look-ahead sites (starting points)

Verified or strongly suspected. Harness 3 should confirm/refute each with
file:line evidence and a numeric demonstration, and propose (not apply) a fix.

1. **Proximity-cap divergence (confirmed).** Backtest 3.0/3.5 vs live 2.5/3.0 →
   backtest fires more/earlier alerts. Knob #10/#11. Real, by design, but it
   means backtest alert set ⊋ live alert set.
2. **`compute_atr` global cache (`_ATR_CACHE`).** Pure on OHLC fingerprint, so
   correct — but it's process-global mutable state. A sweep that runs many
   configs in one process shares it. ATR doesn't depend on the swung knobs, so
   it's safe, but verify no harness assumes a clean cache per iteration.
3. **Fill-bar TP suppression (guard #9).** Conservative, but it means a trade
   that genuinely gapped to TP on the fill bar is denied — a *pessimistic* bias,
   not look-ahead. Quantify how often it bites.
4. **Same-bar SL-first (guard #10).** Pessimistic bias; quantify.
5. **MIN_LEG_ATR_MULT def-time-default trap (knob #1).** If any existing
   diagnostic or future harness monkeypatches the constant expecting it to flow
   through `compute_pair_walls`, it silently won't. Not a live bug, but a harness
   correctness trap — call it out loudly.
6. **`detect_smc_radar` return-shape normalisation** (`_normalize_obs_result`)
   tolerates dict/list/tuple. Confirm the live shape is always
   `{"active_zones": [...]}` so the backtest isn't silently reading a fallback
   branch that live never hits.
7. **OB dedup by `(ob_timestamp, direction)` (guard #6).** Verify the live system
   actually behaves this way (one trade per OB) — if live would re-take a
   re-approached un-mitigated OB, the backtest under-counts trades vs live.
8. **Warmup asymmetry.** Backtest needs 50 closed bars + 35d fetch pad; live runs
   on whatever persisted state exists. Confirm the first ~50 bars of any window
   aren't producing different structure than live would from warm state.

---

## 8. What to produce (your deliverables to the engineer)

For EACH harness: (a) the question it answers and which slice-mode (§4) it uses;
(b) architecture — modules, the shared detection-driver, data flow; (c) exact
inputs/outputs incl. the CSV/Markdown columns; (d) the one-line PowerShell run
command; (e) edge cases handled (empty data, clamp, NaN ATR, degenerate range,
zero-R, the def-time-default trap, cache); (f) how its numbers reconcile to
`r_realised`; (g) a sanity self-check the harness runs on itself before claiming
success. For Harness 3 additionally: the ranked divergence report format
(file:line + demonstration + proposed fix option, nothing applied).

Be meticulous about look-ahead. The single deepest requirement: **prove the
guards in §6 hold, don't assume them.** Where you propose a check, make it the
kind that would have caught "the event is the trade" — e.g. assert for every
filled trade that `fill_ts > alert_ts >= bos_ts` and that no fill price was
available on a bar at or before the alert.
