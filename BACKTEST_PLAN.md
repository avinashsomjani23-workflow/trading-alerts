# Backtest Plan — H1-Only SMC System

**Last updated:** 2026-05-22
**Goal:** Complete all 15 backtest runs in 2–3 days. Fix what is broken within 1 week. Trade the funded account after.

---

## What this document is for

This is a complete, standalone plan for backtesting the H1-only SMC trading system. A reviewer with no prior context should be able to read this document and tell whether the plan is statistically sound, practically executable, and likely to produce a trustworthy answer about whether the system is ready to trade live.

---

## What system is being tested

**Name:** H1-only SMC alert system
**Instruments:** EURUSD, NZDUSD, USDJPY, USDCHF, XAUUSD (Gold), NAS100
**Timeframe:** H1 (1-hour bars) only. No M15 or M5 data used anywhere in this plan.

**How it works:**
1. The system scans H1 price data for Order Blocks (OBs) — price levels where institutional buyers or sellers previously stepped in.
2. When price returns to an OB, the system fires an alert.
3. It scores the setup by counting how many confluences are present (FVG, liquidity sweep, kill zone, PD array alignment, OB freshness, structure tier).
4. A trade is simulated with two entry options: at the OB proximal edge (the near side of the block) or at the OB 50% midpoint (deeper into the block, fewer fills, higher reward-to-risk).
5. Stop loss is placed at the OB distal edge. Take profit targets are the nearest opposing H1 liquidity levels.

**What "H1-only" means in this context:**
All data, all detection, all entry and exit simulation runs on hourly bars. No intraday (15-minute or 5-minute) data is used. This is a deliberate choice — H1 setups are cleaner, more reliable, and the data availability is much broader.

---

## Glossary

| Term | Plain English meaning |
|------|----------------------|
| **R** | The amount you risk on one trade. If you risk $250 per trade, 1R = $250. A trade that makes +2R made $500. A trade that loses -1R lost $250. |
| **Expectancy** | Average R earned per trade across all trades. +0.3R means on average each trade made 30% of the risk amount. Negative = losing system. |
| **Win rate** | Percentage of trades that closed profitably. |
| **Max drawdown** | The largest peak-to-trough loss (in R) the system suffered in sequence. If you had 5 losses of 1R each with no wins in between, that's 5R drawdown. |
| **Proximal entry** | Entry at the near edge of the Order Block — the edge price first touches when approaching the OB. Standard SMC entry. |
| **50% entry** | Entry at the midpoint of the Order Block — requires price to penetrate deeper. Fewer trades fill (price sometimes reverses before reaching this level), but reward-to-risk is higher for those that do. |
| **Never-filled** | A 50% entry where price reversed before reaching the OB midpoint. Counted in fill rate statistics, not in win/loss statistics. |
| **OB (Order Block)** | The last bearish candle before a bullish move (bullish OB) or the last bullish candle before a bearish move (bearish OB). Represents where institutional orders were placed. |
| **CHoCH** | Change of Character — structure shifts from making lower lows to making a higher high (or vice versa). Signals a potential reversal. |
| **BOS** | Break of Structure — price breaks a previous swing high or low, confirming continuation. |
| **FVG** | Fair Value Gap — a price gap between two candles where price moved too fast for the market to fairly distribute orders. Often acts as a magnet for price to return. |
| **Confluence** | Any additional factor that strengthens the case for a trade — FVG, liquidity sweep, kill zone timing, etc. More confluences = higher score. |
| **Kill zone** | Specific time windows when institutional participants are most active: London open (07:00–10:00 UTC), New York open (13:00–16:00 UTC). |
| **PD array alignment** | Whether the OB sits in the right position within the dealing range (discount zone for longs, premium zone for shorts). |
| **Score** | A number (roughly 0–8) assigned to each OB-touch alert based on how many confluences are present. Higher score = more reasons to take the trade. |
| **BAU week** | Business as usual — a normal trading week with no major economic events, elections, or geopolitical shocks. Price moves based on existing trends and ordinary market dynamics. |
| **War week** | A high-volatility week driven by a major event (election, rate decision, geopolitical shock). Labeled `war` in run commands. |
| **Out-of-sample (OOS)** | Data that was never looked at during tuning. The OOS test is only valid if you commit to making no further changes after the OOS group starts. |

---

## The journey from here to live trading

| Stage | What happens | When you move forward |
|-------|-------------|----------------------|
| **1. Backtest** ← you are here | Run 15 specific weeks on historical data. Zero money at risk. | 200+ trades collected, conclusions clear |
| **2. Fix** | Change what the backtest reveals is broken | After Group 1 review. Lock before Group 2. |
| **3. Live trading** | Trade the funded account using validated pair × session cells and score thresholds | After all 15 weeks complete and verdict is GREEN |

The prop firm challenge serves as real-conditions validation. Treat it accordingly — use only the setups the backtest validated, at the exact score threshold the data supports.

---

## Data availability

**H1 data limit:** yfinance provides up to **720 days** of H1 historical data (confirmed in `backtest/data_loader.py`).

- Today: 2026-05-22
- Earliest available H1 date: approximately **2024-06-01**
- Every week in this plan falls within that window. The earliest week (2024-06-10) is approximately 712 days ago.

M15 and M5 data are available for only the last 60 days. Since this plan uses H1-only mode throughout, this limit is irrelevant.

---

## How to run each week

H1-only is now the default mode. No extra flags needed.

```
python backtest/run_backtest.py --start YYYY-MM-DD --end YYYY-MM-DD --regime LABEL --email
```

- `--regime war` for high-volatility event-driven weeks (marked in tables below)
- `--regime bau` for everything else
- `--email` sends the HTML report and Excel file to your inbox
- After each run completes, `BACKTEST_LOG.md` and `backtest/registry.json` update automatically

---

## Why 200+ trades?

With fewer trades, luck looks like skill. A single good week can make a broken system look profitable.

Consider: if the true win rate is 50% and you run 20 trades, a run of good luck could show 65%+ easily. With 200 trades, statistical noise shrinks enough that the real win rate becomes clear. 200 trades is the minimum for conclusions you can act on with reasonable confidence.

---

## ⚠️ Execution risk #1 — trade frequency is unknown until the system runs

**The 200-trade target assumes roughly 13–20 filled trades per week. This has no data behind it yet.** If the system fires 6 filled trades per week on average, 15 weeks produces ~90 trades — not enough to conclude anything.

**Checkpoint rule:** After Group 1 completes (6 weeks), count total filled trades.

| Avg filled trades/week | 15-week projection | Action |
|----------------------|-------------------|--------|
| ≥ 14 | 210+ | On track. Proceed. |
| 10–13 | 150–195 | Add 2–3 more weeks before Group 2. Re-check projection. |
| < 10 | < 150 | Stop. Investigate why the system is not firing. Likely a detection issue, not a strategy issue. Fix before continuing. |

Do not proceed to Group 2 until you have a trade frequency estimate from real Group 1 data.

---

## Why three groups?

**Group 1 (Study):** You are allowed to look at results, notice patterns, and make fixes. This is your learning data.

**Group 2 (Out-of-sample):** The real test. Run only after Group 1 is fully reviewed and all changes are locked in. If you change anything after Group 2 starts, the test is void. Whatever Group 2 shows is the truth about whether the system generalises — not just memorised Group 1 patterns.

**Group 3 (Live-era):** Run last. Tests whether the system still works in conditions closest to what you will actually trade.

**Why Group 1 must span different market environments — not just different calendar dates:**
Five weeks from the same macro era (same central bank policy, same equity trend) teach you how the system behaves in one specific environment. If you tune to that, the system may only work in that environment. Group 1 deliberately covers six different market conditions so any lessons apply broadly.

**Why Group 2 must be chronologically after all Group 1 dates:**
The out-of-sample test only means something if you never saw that data before locking parameters. Group 2 starts May 2025 — after the latest Group 1 week (April 2025). This is a hard constraint.

---

## Group 1 — Study window: 6 weeks across six distinct market environments

**Dates: June 2024 → April 2025**
**After reviewing all 6 results and committing any fixes → lock all parameters → start Group 2**

| # | Start | End | Regime | Market type | Why this week |
|---|-------|-----|--------|-------------|---------------|
| 1 | 2024-06-10 | 2024-06-14 | `bau` | BAU moderate — ECB first rate cut | Scheduled event, moderate EUR activity. Tests system under anticipated directional move. |
| 2 | 2024-07-14 | 2024-07-18 | `bau` | BAU trending — no major events | Mid-summer, no dominant news. Price moves on existing trends only. Purest test of OB quality. |
| 3 | 2024-08-19 | 2024-08-23 | `bau` | BAU choppy — post-Yen-carry settle | Thin, noisy, no clean direction. Tests whether system stays quiet when it should. |
| 4 | 2024-11-04 | 2024-11-08 | `war` | Extreme vol — US election 2024 | Strongest single-week move of 2024. Will OBs hold under institutional position-building? |
| 5 | 2025-02-10 | 2025-02-14 | `bau` | BAU 2025 conditions — normal week | Establishes how the system performs in 2025 macro conditions without a shock or event. |
| 6 | 2025-04-07 | 2025-04-11 | `war` | Shock — tariff Liberation Day aftermath | One of the largest macro shocks of 2025. Every pair moved significantly. |

**What Group 1 covers:** BAU trending, BAU choppy, BAU 2025-era normal, moderate event, extreme event, shock. Six distinct environments. No two weeks are from the same macro regime.

Commands for Group 1:
```
python backtest/run_backtest.py --start 2024-06-10 --end 2024-06-14 --regime bau --email
python backtest/run_backtest.py --start 2024-07-14 --end 2024-07-18 --regime bau --email
python backtest/run_backtest.py --start 2024-08-19 --end 2024-08-23 --regime bau --email
python backtest/run_backtest.py --start 2024-11-04 --end 2024-11-08 --regime war --email
python backtest/run_backtest.py --start 2025-02-10 --end 2025-02-14 --regime bau --email
python backtest/run_backtest.py --start 2025-04-07 --end 2025-04-11 --regime war --email
```

---

## Group 2 — Out-of-sample lock: 5 weeks

**Dates: May 2025 → November 2025**
**Hard rule: No parameter changes once Group 2 starts. Whatever it shows is the real answer.**

| # | Start | End | Regime | Market type | Why this week |
|---|-------|-----|--------|-------------|---------------|
| 7 | 2025-05-12 | 2025-05-16 | `bau` | BAU — post-tariff settle | Markets finding new equilibrium after the shock. Tests OB validity in a regime transition. |
| 8 | 2025-07-07 | 2025-07-11 | `bau` | BAU — mid-summer normal | Seasonal low activity. Clean test of whether system fires selectively when it should stay quiet. |
| 9 | 2025-09-15 | 2025-09-19 | `war` | War — Fed rate decision week | September FOMC meeting. Major institutional repositioning across all pairs. The only war week in OOS — necessary to catch whether system edge holds under event-driven flow. |
| 10 | 2025-10-27 | 2025-10-31 | `bau` | BAU — late Q4 | Pre-Fed October, typical institutional positioning week. |
| 11 | 2025-11-17 | 2025-11-21 | `bau` | BAU — pre-holiday | Liquidity beginning to thin ahead of December. |

Commands for Group 2:
```
python backtest/run_backtest.py --start 2025-05-12 --end 2025-05-16 --regime bau --email
python backtest/run_backtest.py --start 2025-07-07 --end 2025-07-11 --regime bau --email
python backtest/run_backtest.py --start 2025-09-15 --end 2025-09-19 --regime war --email
python backtest/run_backtest.py --start 2025-10-27 --end 2025-10-31 --regime bau --email
python backtest/run_backtest.py --start 2025-11-17 --end 2025-11-21 --regime bau --email
```

---

## Group 3 — Live-era validation: 4 weeks

**Dates: January 2026 → May 2026**
**Run last. These are closest to current conditions. Accept whatever they show.**

| # | Start | End | Regime | Market type | Why this week |
|---|-------|-----|--------|-------------|---------------|
| 12 | 2026-01-12 | 2026-01-16 | `bau` | BAU — early 2026 | Post-inauguration, new macro year settling. |
| 13 | 2026-03-09 | 2026-03-13 | `bau` | BAU — Q1 2026 normal | Pre-FOMC quiet week. |
| 14 | 2026-04-13 | 2026-04-17 | `bau` | Recent — re-run needed | Prior run used M15 (Phase 2) mode. Must re-run as H1-only. |
| 15 | 2026-05-05 | 2026-05-09 | `war` | Recent active — re-run needed | Prior run used M15 mode. Must re-run as H1-only. |

Commands for Group 3:
```
python backtest/run_backtest.py --start 2026-01-12 --end 2026-01-16 --regime bau --email
python backtest/run_backtest.py --start 2026-03-09 --end 2026-03-13 --regime bau --email
python backtest/run_backtest.py --start 2026-04-13 --end 2026-04-17 --regime bau --email
python backtest/run_backtest.py --start 2026-05-05 --end 2026-05-09 --regime war --email
```

---

## Week type distribution across all 15 weeks

| Type | Weeks | Which ones |
|------|-------|-----------|
| BAU trending (normal, no events) | 8 | #2, #5, #7, #8, #9, #10, #11, #12 |
| BAU choppy or quiet | 2 | #3, #13 |
| BAU moderate event (scheduled) | 1 | #1 |
| War — extreme directional event | 2 | #4, #15 |
| War — macro shock | 2 | #6, #14 |

**BAU weeks represent 10 of 15 (67%)** — the majority, reflecting real-world market conditions where most weeks are ordinary.

---

## After every run — what to check

Open the HTML report that arrives in your email. Answer these five questions and log the answers in the tracking table.

| # | Question | Healthy | Flag |
|---|----------|---------|------|
| 1 | Did trades fire (filled)? | 8+ per week | Under 5 — week too quiet, minimal data |
| 2 | Win rate? | ≥ 45% | < 40% |
| 3 | Expectancy (avg R per trade)? | +0.3R or better | Negative |
| 4 | Were losses clean? (how far price moved against before SL hit) | MAE close to 1.0R | MAE < 0.5R — stops may be too tight |
| 5 | Did winners capture most of the move? | ≥ 75% of available move | < 60% — TPs too tight |

**After Group 1 is complete, look across all 6 weeks together and ask:**
- Which pair × session combination appeared consistently positive?
- Did the war weeks (election, tariff) produce better or worse results than the BAU weeks?
- Do higher-score trades show better outcomes than lower-score trades, or is it random?
- What was the worst losing streak across Group 1?
- Is the proximal entry consistently outperforming the 50% entry, or vice versa?

These cross-week patterns are what determine what to fix before Group 2.

---

## Cross-run log

After every run, the registry updates automatically. Two files are written:

- **`BACKTEST_LOG.md`** — readable log showing every run side by side. This is the file to open when discussing what the data is telling us. It shows win rate, expectancy, per-pair breakdown, session breakdown, and score verdict for every completed run.
- **`backtest/registry.json`** — structured data used by the aggregate analysis script.

To add notes to a completed run, edit the `"notes"` field in `registry.json` for that run, then run:
```
python backtest/update_registry.py
```

---

## After all 15 weeks — the combined analysis

Run this once all weeks are complete:
```
python backtest/aggregate_runs.py
```

This pools every trade row from all 15 run folders into one dataset and produces a combined report.

### Five questions the combined report must answer

**A. Is the system profitable overall?**
Expectancy positive across Groups 1, 2, and 3. If only Group 1 is positive, the system was tuned to a specific era and the backtest has failed its core purpose.

**B. Does the score actually predict trade quality?**
Trades with higher scores should produce better outcomes. If there is no relationship, the scoring system needs a redesign before live trading.

**C. Which pair × session combinations are tradeable live?**
The heatmap shows every pair × session cell with trades, win rate, and expectancy. Only cells with **20+ trades and positive expectancy** are candidates for live trading. Everything else is unconfirmed — do not trade it.

**D. Which confluences are earning their weight?**
The attribution table shows whether each of the six confluences (FVG, liquidity sweep, kill zone, PD alignment, OB freshness, structure tier) actually improves results when present versus absent. Any confluence that shows no measurable uplift across 30+ trades should have its score weight reduced.

**E. Does performance hold across all three groups?**
Group 1 → Group 2 → Group 3 should all be positive. If Group 1 is strong but Group 2 or 3 weakens, the system may be era-fitted to mid-2024 conditions. That is not safe to trade.

---

## The three verdicts

### GREEN — trade the funded account

**Forex pairs (EURUSD, NZDUSD, USDJPY, USDCHF):**
- Expectancy ≥ +0.3R
- Win rate ≥ 45%
- 95% CI lower bound above zero

**Gold (XAUUSD) and NAS100 — higher bar, different cost structure:**
- Expectancy ≥ +0.5R (Gold and NAS100 have significantly wider spreads and higher slippage than Forex. A +0.3R edge on paper may be zero or negative after real-world costs.)
- Win rate ≥ 50%
- 95% CI lower bound above zero

**Both instrument groups must also satisfy:**
- Score predicts results in the right direction (higher score → better trades)
- At least 3 pair × session cells have 20+ trades and positive expectancy with CI above zero
- Performance positive in all three groups

**What you do:** Trade only the validated pair × session cells at the score threshold the data supports. No other setups.

---

### YELLOW — specific fix needed before trading

One or two things are broken. Fix the specific issue, re-run only the affected weeks, then re-check.

| What you see | Root cause | Fix |
|-------------|------------|-----|
| Score has no relationship to outcome | Confluence weights are wrong | Recalibrate using attribution table |
| London session consistently negative | Kill zone timing off, or OB bias unreliable before NY | Add London session filter |
| Gold or NAS100 consistently losing | OB detection quality on those instruments | Investigate dealing range and OB detection separately |
| Win capture below 60% (TPs too tight) | TP2 set to nearest 1.5R swing — sometimes not far enough | Move TP2 to next liquidity level |
| Losses stopping shallow (MAE < 0.5R) | SL too tight, getting wicked | Add a small buffer past OB distal |
| Good in Group 1, weaker in Group 2, weak in Group 3 | Era-fitted — tuned to mid-2024 conditions | Re-examine score thresholds; do not trade until stable across groups |
| BAU weeks perform fine, war/shock weeks consistently lose | System struggles with fast, erratic price action | Add a volatility filter — avoid war weeks or reduce position size |

---

### RED — do not trade yet

Any of these:
- Negative expectancy across all three groups
- Score completely random — no relationship to outcomes
- Longest losing streak exceeds 12 consecutive trades in any group

Identify root cause before running more data or trading. Do not try to trade through a fundamentally broken signal.

---

## Run tracking table

Update after every run. BACKTEST_LOG.md has the detail; this is the quick-look summary.

| # | Dates | Group | Regime | Type | Status | Trades | Win% | Avg R | Notes |
|---|-------|-------|--------|------|--------|--------|------|-------|-------|
| 1 | 2024-06-10 | 1 | bau | BAU moderate | Not started | — | — | — | |
| 2 | 2024-07-14 | 1 | bau | BAU trending | Not started | — | — | — | |
| 3 | 2024-08-19 | 1 | bau | BAU choppy | Not started | — | — | — | |
| 4 | 2024-11-04 | 1 | war | Extreme vol | Not started | — | — | — | |
| 5 | 2025-02-10 | 1 | bau | BAU 2025 normal | Not started | — | — | — | |
| 6 | 2025-04-07 | 1 | war | Shock | Not started | — | — | — | |
| 7 | 2025-05-12 | 2 | bau | BAU | Locked | — | — | — | |
| 8 | 2025-07-07 | 2 | bau | BAU | Locked | — | — | — | |
| 9 | 2025-09-08 | 2 | bau | BAU | Locked | — | — | — | |
| 10 | 2025-10-27 | 2 | bau | BAU | Locked | — | — | — | |
| 11 | 2025-11-17 | 2 | bau | BAU | Locked | — | — | — | |
| 12 | 2026-01-12 | 3 | bau | BAU | Locked | — | — | — | |
| 13 | 2026-03-09 | 3 | bau | BAU | Locked | — | — | — | |
| 14 | 2026-04-13 | 3 | bau | BAU | Re-run (mode fix) | — | — | — | |
| 15 | 2026-05-05 | 3 | war | Active | Re-run (mode fix) | — | — | — | |
| **Total** | | | | | | — | — | — | **Target: 200+ filled trades** |

---

## Regime label verification

Regime labels (BAU/war) in the plan are **pre-assigned based on known calendar events**, not verified from actual price data. A "BAU" week with a surprise macro print mid-week may behave like a war week.

**Post-run check (required for every run):**
After each run completes, look at the exit reason distribution in the HTML report:
- If > 60% of trades hit stop-loss: the week was more volatile than its BAU label suggests
- If average MAE on losing trades exceeds 1.2R: price was moving fast and far — atypical for BAU

If a BAU-labelled week shows these signals, flag it in the registry notes as "reclassified: likely war" and interpret results accordingly. The `aggregate_runs.py` script does this check automatically and lists any flagged weeks in the VERDICT.md.

Full ATR-based automated regime classification is a planned enhancement. For now, exit-reason distribution is the proxy.

---

## Known limitations to declare upfront

Any reviewer or new conversation should be aware of these before judging the results:

1. **No spread, slippage, or swap costs modelled.** Real P&L will be approximately 5–10% lower than the backtest shows.
2. **Same-bar SL + TP collision resolves as SL hit first.** This is the pessimistic assumption — better than assuming TP was hit.
3. **H1 bar-level resolution.** Entry and exit are simulated at H1 bar boundaries. Real execution happens tick-by-tick and may differ slightly.
4. **yfinance bars may differ from broker bars.** Small differences in OHLC values are expected.
5. **50% entry fill rate is approximate.** Whether a H1 bar actually fills a mid-OB limit depends on intrabar tick data we do not have.
6. **No position sizing progression.** Every trade risks the same 1R ($250). Real trading may use variable sizing.

---

## What still needs to be built before the final verdict

| Item | What it does | When needed |
|------|-------------|-------------|
| `backtest/aggregate_runs.py` | Pools all trade rows from all run folders for combined analysis | Before final verdict |
| `backtest/insights.py` | Calculates expectancy with confidence interval, Sharpe, drawdown, heatmap, confluence attribution, score validation | Before final verdict |
| Updated HTML email | Shows the 5 big-picture questions in plain English with heatmap and score verdict embedded | Before live trading decision |

---

## Current bugs fixed (as of 2026-05-22)

These were fixed before the backtest schedule began. All previous runs that used the old code are invalid and must be re-run.

| Bug | What was wrong | Fix |
|-----|---------------|-----|
| Excel missing from h1_only email | `reporting_email.py` only looked for `forex_trades.xlsx`, not `trades.xlsx` | Added `trades.xlsx` to attachment scan |
| Excel write failure silent | HTML report said nothing when openpyxl failed | Now shows a red warning in the HTML body |
| 50% entry using wrong TP levels | Each entry computed its own TP using the 1.5R gate measured from its own entry price. Since the entry differs, TP1 resolved to a different swing — making the A/B comparison meaningless | Proximal TP prices now shared with 50% entry. Same opposing liquidity target, different entry only |
| `never_filled` rows counted as breakevens | Confirmed this was already filtered in `h1_only_reporting.py`. No change needed | Verified — not an active bug |
| Default mode was `auto` (tried M15 first) | Runs under 60 days old used M15 data, older runs silently fell back to H1-only — inconsistent | Default changed to `h1_only`. All runs now consistent regardless of date |
