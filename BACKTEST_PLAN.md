# Backtest Validation Plan

**Goal:** Accumulate ≥200 trades across diverse market conditions, validate system edge with 95% statistical confidence, and define exact improvement triggers.

---

## Step 0 — Regime Classification (automated, no eyeballing)

For each week and each pair, compute:

```
ratio = (week's average H1 ATR-14) / (12-week rolling median H1 ATR-14 for that pair)
```

Per-pair label:
- `ratio > 1.25` → TRENDING
- `ratio < 0.75` → CHOPPY
- otherwise → NORMAL

Week-level label (across all 6 pairs):
- ≥ 4 pairs TRENDING → **TRENDING week**
- ≥ 4 pairs CHOPPY → **CHOPPY week**
- otherwise → **NORMAL week**

This is computed automatically before each backtest run and printed in the report header. No manual judgement involved.

---

## Step 1 — The 13-Week Run Schedule

Two already done; 11 remaining. Run in order. Do not skip.

### Tune window (Jan–Aug 2024) — builds the picture, tune nothing mid-way

| # | Start | End | Expected regime | Why |
|---|-------|-----|-----------------|-----|
| 1 | 2024-01-08 | 2024-01-12 | NORMAL | Quiet early-year, no major events |
| 2 | 2024-03-11 | 2024-03-15 | TRENDING | CPI + Fed commentary, strong USD move |
| 3 | 2024-04-08 | 2024-04-12 | TRENDING | Gold breakout, Middle East escalation |
| 4 | 2024-05-13 | 2024-05-17 | NORMAL | Post-FOMC settle, low-news week |
| 5 | 2024-06-10 | 2024-06-14 | NORMAL/TRENDING | ECB first cut, moderate activity |
| 6 | ~~2024-07-08~~ | ~~2024-07-12~~ | NORMAL/CHOPPY | **Done** (bau run exists) |
| 7 | ~~2024-08-19~~ | ~~2024-08-23~~ | CHOPPY | **Done** (h1only run exists, has bugs to fix first) |

### Validate window (Sep–Dec 2024) — out-of-sample, change nothing after this starts

| # | Start | End | Expected regime | Why |
|---|-------|-----|-----------------|-----|
| 8 | 2024-09-16 | 2024-09-20 | TRENDING | Fed first rate cut — major institutional move |
| 9 | 2024-10-07 | 2024-10-11 | NORMAL | Mid-quarter, no dominant theme |
| 10 | 2024-11-04 | 2024-11-08 | TRENDING | US election — extreme vol, strong direction |
| 11 | 2024-12-09 | 2024-12-13 | CHOPPY | Pre-holiday, thin liquidity |

### Live-era validation (Jan–Mar 2025) — proves edge didn't expire

| # | Start | End | Expected regime | Why |
|---|-------|-----|-----------------|-----|
| 12 | 2025-01-13 | 2025-01-17 | NORMAL | Post-inauguration settle |
| 13 | 2025-02-03 | 2025-02-07 | TRENDING | NFP + tariff announcements |
| 14 | 2025-03-10 | 2025-03-14 | NORMAL | Quiet pre-FOMC week |

**Target total:** 200–280 trades across all 13 active weeks (existing two included).

---

## Step 2 — Aggregate and Analyse

After all runs are complete:

1. Run `backtest/aggregate_runs.py` across all 13 run directories.
2. Run `backtest/insights.py` on the combined dataset.
3. The rich HTML report is generated once — this is the validation report.

The Excel files are there to manually verify individual trades. The HTML report is the verdict.

---

## Step 3 — What the Report Must Show and What Each Result Means

### A. Overall edge

| Metric | GREEN (proceed) | YELLOW (investigate) | RED (stop, redesign) |
|--------|----------------|----------------------|----------------------|
| Expectancy R (bootstrap 95% CI lower bound) | > 0.0R | 0.0R to -0.1R | < -0.1R |
| Win rate | ≥ 45% | 40–44% | < 40% |
| Max drawdown | ≤ 6R | 6–10R | > 10R |
| Sharpe (on trade R-series) | > 0.5 | 0.2–0.5 | < 0.2 |
| Longest losing streak | ≤ 6 | 7–9 | ≥ 10 |

If RED on any row: stop. Identify root cause before running more weeks.

### B. Score validation

The scoring system must prove it predicts outcomes — otherwise it's decoration.

| Verdict | Criteria | Meaning |
|---------|----------|---------|
| Score works | Spearman ρ > 0.25 AND score ≥ 3 bucket beats score < 2 bucket by ≥ 0.3R | Use current scoring, consider tightening the gate |
| Score is weak | ρ 0.10–0.25 OR bucket gap 0.1–0.3R | Recalibrate weights, don't raise the gate yet |
| Score is broken | ρ < 0.10 OR no monotonicity across buckets | Score redesign required before live trading |

### C. Confluence attribution

For each of: FVG present, liquidity sweep, kill zone, PD alignment, OB freshness, BOS tier (CHoCH vs BOS):

| Result | Criteria | Action |
|--------|----------|--------|
| Confluence earns its weight | ≥ 0.2R marginal uplift, n ≥ 30 | Keep in score |
| Confluence is noise | < 0.1R uplift regardless of n | Remove from score or reduce weight to 0 |
| Not enough data | n < 30 | Do not act yet; note for next run batch |

### D. Pair × session matrix

Each cell (pair + session combination) needs ≥ 20 trades before a conclusion is valid.

| Cell result | Action |
|-------------|--------|
| ≥ 20 trades, expectancy > 0.3R, CI excludes zero | Live-eligible cell |
| ≥ 20 trades, expectancy 0–0.3R | Paper trade only, monitor another 20 trades |
| ≥ 20 trades, expectancy < 0 | Filter this pair/session combination |
| < 20 trades | No conclusion; continue accumulating |

### E. Time-decay check

Split all trades into four quarters (Q1–Q4 of the test window). Compare expectancy per quarter.

| Pattern | Meaning |
|---------|---------|
| Consistent across all quarters | Edge is durable, not regime-specific |
| Strong Q1–Q2, weak Q3–Q4 | Edge may be fading; re-examine confluence weights |
| Erratic (up-down-up) | Small sample noise; run more weeks |
| Monotonically declining | System is regime-fit; requires methodology review |

---

## Step 4 — Out-of-Sample Lock Rules

**Tune window (runs 1–7):** You may adjust score thresholds, confluence weights, TP logic — but only between runs, never after seeing results of a run you're about to include.

**Lock point:** Before running week 8. Write down every parameter value in a `params_locked.json`. Do not change them after this.

**Validate window (runs 8–11):** No changes. Run as-is. Results stand regardless.

**Live-era (runs 12–14):** No changes. If edge holds here, the system is validated for paper trading on the current market.

If you change anything after the lock point, the entire validate window is void and must be re-run.

---

## Step 5 — Improvement Triggers and What to Do

| Observation | Root cause investigation | Fix |
|-------------|------------------------|-----|
| London session consistently negative | Kill zone timing off, or OB bias unreliable pre-NY | Add London session filter to Phase 2 gate |
| GOLD/NAS 0% win rate persists | Phase 3 CHoCH confirmation too slow on H1 bars | Reduce M5 confirmation window, or require liquidity sweep before entry |
| Score doesn't predict R | Individual confluence weights wrong | Reweight using confluence attribution table — highest-uplift confluences get highest weight |
| Wins captured < 80% MFE | TP levels too tight | Move TP2 to next liquidity level, not nearest 1.5R hurdle |
| Losses stopped shallow (MAE < 0.5R) | SL too tight, getting wicked out | Widen SL to OB distal + 10% buffer |
| Edge collapses in Q3–Q4 | Macro regime dependency | Add ATR regime filter — only trade TRENDING or NORMAL weeks |

---

## Slippage Stress (Backlog)

When ready: re-run the aggregate with 1.5 pip spread + 0.5 pip slippage for Forex, 3 pip for Gold, 1 point for NAS100 applied to every fill. Check if expectancy CI lower bound stays above zero. One function call on the trade dataset — no re-simulation needed.

---

## Current Status

| Run | Status | Notes |
|-----|--------|-------|
| bau_20240708_20240712 | Done | No Excel attachment in email |
| h1only_20240819_20240823 | Done (buggy) | Excel missing, 50% TP logic wrong, never_filled pollutes stats — fix first |
| All remaining runs | Not started | See schedule above |
