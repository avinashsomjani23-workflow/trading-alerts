# BACKTEST PERF SPEC — kill the 6-hour runtime (2026-07-26)

Owner: user. Executor: any session (written for Opus). Scope: **backtest path only** —
`backtest/h1_only_simulator.py`, `backtest/replay_engine.py`, `smc_detector.py` helper,
`smc_radar.py` lazy-build. **Zero live-behaviour change, zero number change.**

---

## 1. Measured facts (do not re-derive)

- CI run 30162233896 (`--start 2008-01-02 --end 2016-12-31 --pairs EURUSD`) hit the
  350-min timeout (`.github/workflows/backtest.yml:35`). Log shows `[parallel]
  launching 1 pairs across 1 workers` at 14:45:36, then **nothing until cancel at
  20:34:54** — one pair, 9 years, did not finish in 5h49m on the 2-core runner.
- Local clean timing, EURUSD 2 months (975 walked bars, 117 alerts, machine awake):
  - replay walk (detection): **16–72 s** total (~16–74 ms/bar; varies with host load)
  - simulator: **~0.8–4 s per alert** → sim is **85–95 % of total runtime**.
  - 9-yr extrapolation: ~55 k bars, ~6 000+ alerts → sim alone = hours. Sim is the target.
- Stage-timer split of the replay side (2-month run, ms per walked bar):
  `detect_smc_radar` 9.3 · `compute_pair_walls` 4.8 · sweep D1/W1 frame rebuild 3.6
  (+2 `detect_swings` calls ≈ 1) — the sweep frames are rebuilt **every bar** but only
  consumed when a *new* OB survives dedupe (64 new OBs in 975 bars ≈ 6 % of bars).
- Step 0 profile RESULT (2026-07-26, 10 EURUSD alerts, 2015 2-month window, py3.14,
  machine awake): 17.7 s / 10 alerts. **~89 % of sim time is one chain the original
  §3 menu missed:** `_session_level_features_at_alert` → `session_levels.
  build_session_level_event` → `pool_builder.pool_status` (~1 773 pool_status calls
  per 10 alerts). §3.1-3.5 did not appear in the top profile entries → §3.9 is now
  the primary fix; 3.1-3.5 are secondary. NOTE: profile ran on py3.14 (typing/
  annotationlib overhead is py3.14-specific); CI runs py3.11.15 — the exact ratio
  will differ, but §3.9's O(history²) growth dominates on both.
- The 2-month figure UNDERSTATES 9-yr cost: session-level pools accumulate with no
  lookback cap (`session_levels.py:173` — every completed session-day since data
  start), so per-alert cost grows ~quadratically with history. 1.77 s/alert in
  early 2015 is far more by 2016 — this, not a constant per-alert cost, is the
  timeout shape.
- CAUTION on local measurements: overnight numbers were poisoned by laptop sleep
  (one "312 s/alert" reading was sleep time booked to the sim stage — discard it).
  Measure only with the machine awake / plugged in.

## 2. Step 0 — MANDATORY micro-profile before editing (≈2 min runtime)

The per-alert seconds are NOT yet attributed line-by-line. Do this first, keep the
output in the PR description:

```python
# scratch script, 2-month EURUSD window (2015-01-01..2015-03-01, pad 35d)
# run replay_pair to collect alerts (as run_backtest._process_pair does), then:
import cProfile, pstats
cProfile.runctx("[sim.simulate_h1_only_dual(a, pc, df, risk_usd=250.0) for a in alerts[:10]]",
                globals(), locals(), sort="cumulative")
```

Fix what the profile names, in the order it names them, using §3 as the menu.
If the profile surfaces a hotspot NOT listed in §3, stop and flag it to the user
before inventing a fix.

## 3. Fix menu (each one is result-identical by construction)

**Execution order after the Step 0 profile: §3.9 FIRST (it is ~89 % of sim time),
then 3.1-3.5 as the profile ranks them, then 3.6. 3.7 only as fallback.**

### 3.1 `_atr_compute_raw` walks the whole frame for a 14-bar answer — CERTAIN, do always
`smc_detector.py:150-160`: builds 3 full-length arrays + a Python loop over **every
bar in the frame**, then means the **last 14** TRs. The simulator feeds it
full-history slices (`df_h1.loc[:fill_ts]` etc.), so near the end of a 9-yr run each
call is a ~55 k-iteration Python loop.
**Fix:** inside `compute_atr` (the public wrapper, so every caller benefits), slice
`df = df.tail(period + 1)` **before** the cache key is built and `_atr_compute_raw`
runs. The last `period` TRs need exactly the last `period+1` bars, so the returned
float is bit-identical. Keep the cache; key on the tailed slice.
**Guard:** one unit test: random 500-bar frame → `compute_atr(df) ==` old raw
formula on full frame (compute expected with the untailed loop inline in the test).

### 3.2 Replay per-bar slice copy is O(history) — CERTAIN, do always
`backtest/replay_engine.py:81-95` (`_slice_closed_before`): `df[df.index < ts]`
boolean-masks and **copies the entire prior history every bar** (~1 MB/bar → tens of
GB over a 9-yr walk) only for `.tail(150)` to be taken two lines later
(`replay_engine.py:230-234`).
**Fix:** assert once before the walk that `df_h1.index.is_monotonic_increasing`,
then per bar: `pos = df_h1.index.searchsorted(h1_ts)` (side='left') and
`h1_slice = df_h1.iloc[max(0, pos - detection_bars):pos]`. On a sorted index this is
the same set of bars as mask+tail — same slice, same asserts. Unclamped legacy mode
(`detection_bars=None`) keeps `df_h1.iloc[:pos]`.
**Guard:** the existing lookahead assert (`replay_engine.py:235`) plus the
slice-length assert (`replay_engine.py:245`) already police this — no new guard.

### 3.3 Sweep frames rebuilt every bar, used on ~6 % of bars — CERTAIN, do always
`smc_radar.py:928-939`: `_sweep_days/_sweep_weeks/_sweep_eq_swings/_sweep_sw_swings`
are built unconditionally per call; sole consumer is the deferred survivor pass at
`smc_radar.py:1437-1451` (plus the per-event `observe_pool_sweep` args). 
**Fix:** wrap the build in a lazy helper (`_ensure_sweep_frames()`), call it at the
first point a survivor OB actually reaches the `observe_pool_sweep` call. Bars that
build no new OB (~94 %) skip the resample + 2 swing scans entirely. Values when
built are identical (same `df`, same calls).
**Note:** live calls this too — laziness is a no-op change for live (it builds on
first use within the same call). No live divergence possible.

### 3.4 `iterrows` walks in the simulator — do if Step 0 names them
`backtest/h1_only_simulator.py:823` (realised walk), `:455`
(`_reference_touch_indices`), and the exit-lab replay loop. All are bounded by
`MAX_HOLD_H1_BARS = 48` (`h1_only_simulator.py:42`) so they are seconds only in
aggregate; `iterrows` boxes a Series per bar (~0.3 ms).
**Fix:** pre-extract `High/Low/Open/Close` numpy arrays + the DatetimeIndex once per
walk (`future` slice) and index them positionally in the same loop. Same floats,
same comparisons, same break order — line-for-line translation, no logic edits.
`ts.dayofweek/ts.hour` reads come from the index (`future.index[i]`), unchanged.

### 3.5 Full-history `.loc` copies inside the sim row build — do if Step 0 names them
Suspects (verify against profile before touching): `h1_only_simulator.py:394`
(`_closed_bars_at_alert` mask over the whole frame per call), `:1552`, `:1582`.
Same recipe as 3.2: `searchsorted` + `iloc` window sized to what the consumer
actually reads (e.g. `_closed_bars_at_alert` only needs the last
`LIVE_P2_H1_BARS` closed bars, `:418`).

### 3.6 CI run is a black box — heartbeat + unbuffered output — CERTAIN, do always
Verified from CI log 30162233896: after `[parallel] launching 1 pairs across 1
workers` (14:45:36) there is **zero output until the 20:34:54 cancel**. Two causes:
- The worker prints exactly once, AFTER the whole walk (`run_backtest.py:172`) —
  nothing during the walk, nothing during the ~6 000-alert sim loop.
- Worker stdout in CI is a block-buffered pipe; even that one print can sit unflushed
  (workflow env has no `PYTHONUNBUFFERED`). We cannot even tell if the 9-yr walk
  finished before the timeout.
**Fix:** (a) `PYTHONUNBUFFERED: 1` in `.github/workflows/backtest.yml` env;
(b) heartbeat print every N bars in the walk loop and every N alerts in the sim loop
(bars/alerts done, elapsed, rate, `flush=True`) — pick N so it fires ~ every 1-2 min,
e.g. 2 000 bars / 50 alerts; (c) print the existing stage-timer totals at pair end.
Pure logging, zero result impact. Without this, §6's "re-profile at 9-yr scale"
fallback is impossible on CI — a timed-out run teaches nothing (as just happened).

### 3.7 Sim loop is single-core — FALLBACK ONLY, needs user sign-off
The runner has 2 cores; one worker uses one. The walk is inherently sequential, but
the sim loop (`run_backtest.py:194-239`) sims ~6 000 independent alerts one by one.
The ONLY cross-alert coupling is `filled_obs` (`run_backtest.py:193-201`): a
re-alert is skipped once an earlier alert on the same OB has filled. Equivalent
parallel form: sim ALL alerts (workers), then apply the first-filled-per-OB drop in
alert-ts order as a post-filter — same rows out, some wasted sims. Exit-lab sink
ordering must be re-stitched per alert if armed (it was NOT armed in run 30162233896
— no `--exit-lab` flag in the CI invocation).
**Do NOT build this in the first pass** — max ~2× on CI vs 5-10× expected from
§3.1-3.5, and it is medium-maintenance. Only if §3.1-3.5 + a clean re-run still
miss the timeout, propose it explicitly and wait for approval.

### 3.8 Considered and REJECTED — do not re-propose
- **Chunking the 9 years across parallel CI matrix jobs:** `ReplayState` is
  path-dependent from walk start (OB registry, structure state accumulate over the
  whole walk). Chunks with pad warm-up are NOT byte-identical → violates the
  no-compromise rule. Rejected.
- **Bigger CI runner:** costs money, out of scope unless the user asks.

### 3.9 Session-levels / pool_status chain — DOMINANT (profiled ~89 % of sim), do FIRST
Four stacked O(history) costs, all per alert, verified against live code:
- `h1_only_simulator.py:2209` — `df_h1[df_h1.index < alert_ts]` full-frame mask +
  copy per alert (same disease as §3.2/§3.5).
- `session_levels.py:193-196` — Python loop over EVERY prior bar × 3 sessions per
  alert; each bar boxed via `ts.to_pydatetime()` + `astimezone()` (this boxing is
  the py3.14 typing/annotationlib call detonation in the profile).
- `session_levels.py:289` — `bars[bars.index >= p["close_utc"]]` full-length
  boolean mask per pool per alert (~180 pools at 2-month scale, ~14 000
  level-checks at 9-yr scale: ~2 300 session-days × 3 sessions × 2 levels).
- `pool_builder.py:328` — `pool_status` state-machine loop over all bars since
  pool birth, per level per alert → the quadratic term.

**Stage A — vectorise the session bucketing (result-identical, do always).**
Replace the per-bar `_local_hour_and_date` loop with pandas tz ops on the whole
index once: `local = idx.tz_localize("UTC").tz_convert(tz)`, then `local.hour` /
`local.date` arrays drive the same bucket logic. Same IANA tz database, same
values. **Guard:** unit test — random multi-year frame spanning DST transitions
both directions: old per-bar loop vs new vectorised buckets, identical pools out.

**Stage B — kill the per-pool masks (result-identical, do always).**
`session_levels.py:289`: `bars.index` is sorted (it comes from one H1 frame) →
`pos = index.searchsorted(close_utc, side="left")`, slice `iloc[pos:]` — same
recipe as §3.2. Also apply §3.2's recipe to the shim mask at
`h1_only_simulator.py:2209`. **Guard:** covered by the §4 byte-identical proof.

**Stage C — incremental session-level state (ONLY if A+B miss the target;
needs user sign-off).** The exact fix for the quadratic term: pools are
append-only once their session-day closes, and `pool_status` is a resumable state
machine (status/pending/timestamps), so a walk-scoped tracker could feed each pool
only the NEW bars since the previous alert instead of re-scanning from pool birth.
Byte-identical in principle but stateful and medium-maintenance — spec it,
propose it, wait for approval. Do NOT reach for it before measuring A+B.

**Scope note:** `session_levels` is imported ONLY by the backtest row build
(`backtest/h1_only_simulator.py`, `backtest/h1_only_reporting.py`) — re-verify
with a grep before editing. `pool_builder.pool_status` IS shared with live
(eq_pools, setup_liq, liquidity_sweep, pool emails) — Stages A/B deliberately do
NOT touch pool_status internals; only Stage C would wrap (not edit) it.
**Do-NOT:** no lookback cap on session pools — dropping old pools changes which
level is "nearest" and therefore changes column values. That is a trading-logic
question, not a perf fix.

## 4. Equivalence proof — REQUIRED before ship (repo standard)

1. Pick the frozen window: EURUSD 2015-01-01 → 2015-03-01 (parquet cache, no fetch).
2. Run `backtest/run_backtest.py` on it **before** the edits; save `trades.csv` +
   `summary.json` aside.
3. Run again **after** the edits. `trades.csv` must be **byte-identical** (allow the
   run-id/timestamp meta fields in summary.json to differ, nothing else).
4. Then run the same proof once on GOLD (different pip size / distal mode) for one
   month. Byte-identical or STOP and investigate — do not rationalise a diff.
5. `python preflight.py --fast` green before any push (user pushes, not the agent).

## 5. Do-NOTs (hard)

- No detection-value changes, no re-anchoring of any `*_at_alert` / `*_at_fill`
  stamp, no threshold/knob edits, no alert-count change. If a "perf" edit changes
  the alert count or any column value in §4, it is a bug — revert it.
- No live-path edits beyond the shared `compute_atr` tail (3.1) and lazy sweep
  frames (3.3), both proven no-op for callers.
- Do not "fix" the 117-alerts-per-2-months rate or the proximity cap — trading
  logic is out of scope for this spec.
- No new per-bar guards/asserts inside the walk (CLAUDE.md defensive-coding rule);
  guards live in the §3 unit tests + §4 proof only.
- Laptop must stay awake for any timing you quote (see §1 CAUTION).

## 6. Expected outcome

Replay side: ~30-40 % faster from 3.2+3.3. Sim side (the 85-95 % share): the win
now rests on §3.9 (~89 % of sim per the Step 0 profile) — Stages A+B should
collapse both the constant AND the quadratic-growth term; 3.1+3.4+3.5 add on top.
Target unchanged: 9-yr EURUSD from >6 h to well under 1.5 h on the CI runner.
After §3.9 A+B land, re-run the SAME 10-alert profile and paste both profiles in
the PR — if the session chain is still top-3, escalate to Stage C (with sign-off). If after §3 the 9-yr run still exceeds ~3 h, re-profile at 9-yr
scale (something scale-dependent survived) and report findings; do not guess.
