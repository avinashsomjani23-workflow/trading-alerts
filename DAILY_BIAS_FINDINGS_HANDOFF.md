# Daily Bias Detection — Findings & Build Handoff (fresh-chat summary)

**This consolidates a full investigation chat (2026-06-28/29). It REFINES and partly
OVERTURNS `DAILY_BIAS_BUILD_HANDOFF.md` (v2). Read this first. Every number here was
measured on real MT5 (2008+) / yfinance data — re-run the scratch scripts if in doubt.**

---

## 0. TL;DR — where daily bias detection actually stands
- **The build is NOT started.** This chat was pure validation. Zero Part-A code written.
- **The sticky daily bias does NOT predict direction.** Measured 50.4–50.6% directional
  accuracy (H5/H10/H20) over 18y — a coin flip. "100% sticky" is descriptive/tautological,
  not reliability.
- **No method predicts forward trend-vs-range >90%.** Four were tested; ceiling ~70–79%
  pooled, unstable 44–89% by month. >90% forward prediction is not real.
- **What IS >90% reliable is FACTS, not forecasts:** sweep-vs-break of a closed candle
  (~deterministic), and bias=structure (tautological). Build on facts; use bias as soft
  context/filter only.
- **The real blocker is the FEED, not the maths.** yfinance is too low-fidelity for exact
  pools. OANDA is the leading fix but UNPROVEN — gated experiment below.

---

## 1. What we are building (unchanged intent)
A daily CONTEXT layer on top of the H1 execution system: sticky daily bias, ranked
liquidity pools (PDH/PDL/PWH/PWL), draw-on-liquidity (DOL) + fuel, and sweep/break
classification. H1 stays the only execution clock. Observe-first: log signals next to
`r_realised`, score nothing until it earns it out-of-sample.

---

## 2. VALIDATED with data (the numbers)
- **Sticky bias directional accuracy:** pooled H5 50.6%, H10 50.4%, H20 50.6% (6 instruments,
  18y). Forward 10d drift in bias dir: mean +0.07 ATR (gold best +0.21, NZD negative),
  ~50% positive. → bias has ~no standalone directional edge.
- **Ranging "stale-count" (the v2 default) as a forward predictor:** 45–50% accuracy, BELOW
  the 51–55% majority base rate. Width/compression 36–39%. → predictive ranging is junk.
- **Reactive state read** ("price already beyond the bias wall"): 73–79% pooled vs forward
  expansion; confidence-gated multi-factor version 69% total, **44–89% per month** (unstable).
- **Break confirmation:** at the FIRST close beyond a prior-day level, **27% are fakeouts**
  (close back inside next H1), consistent across EURUSD/XAU/BTC. → N=1 confirm is justified;
  one H1 of latency is fine for a daily layer.
- **W1 from D1 is lossless:** building weekly high/low = max/min of that week's daily
  highs/lows matches MT5 native W1 **100% exact (0.00 pip)**, all pairs. (This was MT5→MT5 —
  it proves the METHOD within one feed, NOT cross-feed agreement.)
- **detect_swings / compute_pd_position / compute_h4_range reuse confirmed** (lb=3 +
  right_lookback exists; pd reads walls["h4_range"]; broken-wall tracking mirror-able to D1).

---

## 3. OVERTURNED / refined from v2
- **v2: "no resampling, real D1/W1 from both feeds."** OVERTURNED for POOLS. The two feeds
  use DIFFERENT native daily boundaries (below), so "real D1/W1 from both" guarantees mismatch.
  Pools must be built on ONE fixed boundary.
- **v2 ranging flag as a usable signal:** DEMOTED to logged-observation only — it does not
  predict (sec. 2).
- **Zones as the parity fix:** REJECTED by user and on merit — ATR-fuzzy levels corrupt the
  sweep-vs-break distinction (the whole point). The fix is a CLEAN, CONSISTENT FEED where the
  level is exact within-feed, not a fuzzy band.

---

## 4. THE FEED PROBLEM (the crux of the whole layer)
Backtest = MT5; live = yfinance. Bias DIRECTION is cross-source robust (98.8%, it's a
sequence). POOLS are absolute price lines and are NOT robust:
- **Boundaries differ.** MT5 rolls the day at **21:00 UTC** (00:00 server, fixed GMT+3 =
  ~5pm NY = the forex-standard "New York close"; verified: MT5 H1 reproduces MT5 D1 **100%**
  at server-midnight). yfinance FX `=X` rolls at **00:00 UTC** (verified: MT5-resampled best-
  matches yfinance native daily at the 00:00 anchor). 3-hour gap.
- **The forex boundary is NOT ambiguous.** It is 5pm NY — the same broker maintenance window
  seen at **2:30–3:30 AM IST** (the 1h spread = US DST moving 5pm NY between 21:00/22:00 UTC).
- **yfinance hourly is low-fidelity:** it cannot reproduce yfinance's OWN daily (agree ≤3p
  31%; on 41% of days the hourly misses the true daily high by >3 pips). So rebuilding clean
  pools from yfinance hourly is impossible.
- **Cross-feed daily-high gap (yfinance vs MT5), full distribution:**
  - resampled to 21:00 UTC: p50 2.6 / p90 4.0 / p95 5.2 / p99 8.4 / **max 51** pips.
  - yfinance native 00:00 UTC: p50 1.3 / p90 11.4 / p95 22 / p99 48 / **max 93** pips.
  - → typical ~3 pips, but a real fat tail. NOT good enough for exact sweep/break.
- **yfinance is even flaky to FETCH** ("possibly delisted" intermittently).
- **MT5 feed is NOT a live option:** `MetaTrader5` is Windows-only + needs a logged-in
  terminal; live runs on GitHub Actions `ubuntu-latest`. Ruled out.

### DST parity trap (must not miss)
MT5 server is fixed GMT+3 → its boundary is **21:00 UTC year-round** = 5pm NY in summer,
4pm NY in winter. To keep backtest = live, any live builder must mirror the **fixed 21:00
UTC**, NOT chase the shifting true-NY-close, or live silently desyncs from the backtest.

### BTC
Crypto-canonical day = **00:00 UTC** (every crypto chart's default; where retail stops sit).
MT5 rolls BTC at 21:00 UTC (non-canonical). Build BTC pools at 00:00 UTC from H1. Venue
divergence is large (~$40 median high gap, MT5 CFD vs yfinance index) → bucket BTC separately.

---

## 5. OANDA — the leading feed fix (UNPROVEN; gated)
Could not test in this chat (no API key, restricted sandbox network). Confirmed from v20 docs
at high confidence: daily candles align to **5pm New York** by default (configurable), pure
**REST/HTTP** (runs on Linux, no terminal), free practice account, real broker prices.
NOT confirmed: actual pip-variance vs MT5. **That is the decision gate.**

### Validation experiment (run BEFORE adopting)
1. Free OANDA practice account + token. `pip install oandapyV20` (or raw requests).
2. Pull ~2y H1 + D for EUR_USD, GBP_USD, USD_JPY, XAU_USD (+BTC if offered).
3. Build D1 at NY-close; build W1 FROM D1 (lossless).
4. Compare OANDA daily H/L vs MT5 native D1. Report p50/75/90/95/99/max pip gap — beat the
   yfinance table in sec. 4.
5. Self-check: does OANDA H1 reproduce OANDA D? (target ~98%+; yfinance scored 31%).
6. Fetch reliability over a day (yfinance is flaky).

### Decision criteria (adopt only if ALL hold)
- OANDA H1→D self-repro ≥ ~98%.
- vs-MT5 tail tighter than yfinance: target p95 ≤ ~5p, p99 ≤ ~10p (FX).
- Fetch clearly more reliable than yfinance.
- If the tail stays fat, OANDA does NOT solve parity — report honestly, don't force it.

### Integration (only after gate passes)
Feed is isolated: live flows through `fetch_with_retry()` → `clean_df()`
(Phase2_Alert_Engine.py:355, 424); live currently pulls only H1 30d (line 2367). Write ONE
adapter returning the SAME DataFrame shape (tz-aware UTC index, OHLC) behind a feed flag.
Needs: token in GH secrets; symbol map (config "symbol" `EURUSD=X` → `EUR_USD`); granularity
(H1→H1, D1→D, W1 from D1); forming-bar handling; staleness; weekend gaps. Risk contained to
the data layer; main work = a PARITY TEST proving identical structure decisions on overlap.

---

## 6. The honest design philosophy (carry this in)
- **You cannot forecast the daily regime >90%. Stop trying.** Build on FACTS:
  - sticky bias = structure (use as a soft FILTER, not a predictor),
  - sweep vs break of a closed candle (a deterministic fact; N=1 confirm),
  - liquidity pools as exact levels on ONE clean feed.
- **Observe-first still holds:** log daily_ctx next to r_realised on every trade AND every
  no-trade bar; let Part B score only what survives out-of-sample. This chat's data is exactly
  why — it would have killed the raw-bias and stale-count signals before they gated anything.
- **The layer's real hope is SELECTION value** (with-bias vs counter asymmetry: handoff's
  +0.153R vs +0.030R, in-sample, must re-prove OOS) and the DOL/fuel/sweep CONTEXT — NOT a
  directional arrow.

---

## 7. Build status & next steps (nothing built yet)
1. **Decide the feed** (OANDA gate, sec. 5) — blocks pool design. Until then, pools are
   unreliable for exact sweep/break.
2. **Decide the boundary** for backtest pools: mirror what live can actually deliver
   (parity-first) vs canonical 5pm NY. (If OANDA passes, both = 5pm NY and this dissolves.)
3. **Then Part A** (build from `DAILY_BIAS_BUILD_HANDOFF.md` §4, with these refinements):
   - `build_htf_candles(h1, instrument)`: H1→D1/W1 at fixed boundary (FX/Gold 21:00 UTC, BTC
     00:00 UTC). W1 from D1. NEW first component.
   - `compute_daily_bias` (sticky + ranging-as-logged) + **persistence + replay==incremental
     unit test** + **warm-up lead-in** (replay full D1 history before the window so state is
     mature at bar 0; cold-rebuild on poisoned state).
   - freshness (D1 live range, mirror broken-wall), liquidity pools (exact, per-feed),
     DOL/fuel, classify_level_interaction (log the FINAL N=1 label + separate provisional flag).
   - Phase-1 attach in a SEPARATE state namespace (no shared keys with h4_range).
   - Logging: daily_ctx on trades + no-trade scans; prove P&L identical to a pre-build baseline.

---

## 8. Open decisions for the user
- Run the OANDA gate? (recommended — it may dissolve the whole parity problem.)
- If OANDA fails: parity-first (mirror yfinance 00:00 UTC in backtest) vs canon (5pm NY, accept
  live can't match exactly). Don't use zones.
- Is the layer worth building given the bias is a coin flip directionally? (Value must come
  from selection/DOL context — confirm appetite before investing build time. Honest flag.)

---

## 9. Scratch scripts (in backtest/diagnostics/, uncommitted, re-runnable)
- scratch_pool_parity.py — native cross-feed pool gap.
- scratch_boundary.py / scratch_boundary2.py — boundary self-checks + which-anchor.
- scratch_resample_match.py — yfinance H1→D1 vs MT5 (+ native baseline).
- scratch_w1_from_d1.py — W1-from-D1 lossless proof + variance distribution.
- scratch_break_confirm.py — first-break fakeout rate (27%).
- scratch_ranging.py / scratch_state90.py — ranging predictors + gated reactive classifier.
- scratch_bias_reliability.py — the 50% directional-accuracy proof.

## 10. Carry-forward glossary
Buyside = above price (PWH/PDH/forming-high); sweep = wick through + close back (fuel);
break/displacement = close beyond + hold (N=1); DOL = nearest unswept pool in bias dir; pool
goes dead once swept. PWH/PWL > PDH/PDL > internal. NAS100 excluded. r_realised = sole P&L
truth. See also RECOMMENDATIONS.md (two caveats appended 2026-06-28).
```
