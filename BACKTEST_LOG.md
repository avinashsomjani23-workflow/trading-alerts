# Backtest Registry

*Last updated: 2026-06-16 16:57 UTC*

Each row is one backtest run. Use this to spot patterns across runs — not just within one.

## Cross-run snapshot

- Runs completed: **2** of 2
- Total filled trades across all runs: **23**
- Average expectancy across runs: **-0.42R**

- **Group 1 (Study):** 1 runs, 8 trades, avg expectancy -0.60R

---

## Individual run log

### h1only_20260209_20260213

**Unclassified** | 2026-02-09 to 2026-02-13 | Regime: `war`

| Metric | Value |
|--------|-------|
| Filled trades | 15 (of 48 rows) |
| Win rate | 26.7% |
| Expectancy | -0.24R |
| Max drawdown | +13.28R |
| Longest losing streak | 8 trades |
| Proximal fill rate | 62.5% |
| 50% fill rate | 58.3% |
| Score verdict | WEAK — partial relationship |

**By pair (proximal entry, TP2 exit):**

| Pair | Trades | Win rate | Expectancy |
|------|--------|----------|------------|
| NAS100 | 2 | 50.0% | +0.64R |
| EURUSD | 3 | 33.3% | +0.07R |
| USDCHF | 3 | 33.3% | -0.14R |
| NZDUSD | 3 | 33.3% | -0.22R |
| GOLD | 1 | 0.0% | -1.00R |
| USDJPY | 3 | 0.0% | -1.00R |

**By session:**

| Session | Trades | Win rate | Expectancy |
|---------|--------|----------|------------|
| Asia | 10 | 30.0% | +0.31R |
| London | 9 | 0.0% | -1.00R |
| NY | 14 | 42.9% | +0.12R |

**Score vs outcome:**

| Score bucket | Trades | Win rate | Expectancy |
|-------------|--------|----------|------------|
| 2-3 | 2 | 0.0% | -1.00R |
| 3-4 | 2 | 0.0% | -1.00R |
| 4-5 | 4 | 25.0% | -0.41R |
| 5-6 | 1 | 0.0% | -1.00R |
| 6-7 | 3 | 33.3% | +0.10R |
| 7+ | 3 | 66.7% | +0.93R |

---

### h1only_20240819_20240823

**Group 1 — Study** | 2024-08-19 to 2024-08-23 | Regime: `bau`
Market: *BAU choppy — post-Yen-carry-unwind settle*

| Metric | Value |
|--------|-------|
| Filled trades | 8 (of 38 rows) |
| Win rate | 12.5% |
| Expectancy | -0.60R |
| Max drawdown | +9.27R |
| Longest losing streak | 7 trades |
| Proximal fill rate | 42.1% |
| 50% fill rate | 42.1% |
| Score verdict | BROKEN — score does not predict outcome |

**By pair (proximal entry, TP2 exit):**

| Pair | Trades | Win rate | Expectancy |
|------|--------|----------|------------|
| GOLD | 2 | 50.0% | +0.59R |
| EURUSD | 2 | 0.0% | -1.00R |
| NAS100 | 2 | 0.0% | -1.00R |
| USDJPY | 2 | 0.0% | -1.00R |

**By session:**

| Session | Trades | Win rate | Expectancy |
|---------|--------|----------|------------|
| Asia | 7 | 14.3% | -0.80R |
| London | 3 | 33.3% | +0.59R |
| NY | 7 | 28.6% | -0.40R |
| Other | 1 | 100.0% | +1.71R |

**Score vs outcome:**

| Score bucket | Trades | Win rate | Expectancy |
|-------------|--------|----------|------------|
| 2-3 | 1 | 0.0% | -1.00R |
| 3-4 | 1 | 0.0% | -1.00R |
| 4-5 | 2 | 0.0% | -1.00R |
| 5-6 | 2 | 50.0% | +0.59R |
| 6-7 | 1 | 0.0% | -1.00R |
| 7+ | 1 | 0.0% | -1.00R |

---
