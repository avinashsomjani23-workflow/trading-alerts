# Backtest Harness

Replays historical bars through the live SMC detection modules to estimate what
the trading system *would have done* on a given week.

## Hard rules

- **Zero edits to live code.** This folder imports `smc_radar`,
  `smc_detector`, etc. read-only. It never writes to live JSON state files. All
  state writes go to `backtest/state/`.
- **No live email sent.** Backtest never calls live SMTP plumbing.
- **No lookahead.** Every dataframe slice fed to detection is asserted to end
  at or before the replay timestamp. Violation = hard fail.

## What it does

1. Fetch historical OHLC from yfinance for the requested date range.
2. Walk H1 bars one at a time. At each H1 close:
   - Slice H1 + M15 (+ M5 if available) up to that timestamp.
   - Feed slices to `smc_radar.compute_pair_walls` and `smc_radar.detect_smc_radar`.
   - Score the OB (mirrors Phase 2 confluence scoring — see compromise note below).
   - If score >= min_confidence, emit a "would-be alert".
3. For each emitted alert, simulate the trade:
   - Walk forward on M15 (or M5 for Phase 3) bars.
   - Track entry fill, SL hit, TP1/TP2 hit, MFE, MAE.
   - Conservative rule: same-bar SL + TP collision → SL hits first.
4. Generate Excel + HTML report with two regimes (per-pair view + Forex vs NAS/XAU split).

## Mode

H1-only is the only mode. Tests the SMC system on H1 data alone:
- Skips M15 / M5 fetches entirely.
- **No scoring gate** — every H1 OB-touch fires a trade regardless of
  confluence score. Score is still computed (via live
  `smc_detector.run_scorecard`, H1-only since 2026-05-26) and logged so the
  user can discover the optimal threshold empirically from trade outcomes.
- **Dual entry** — every OB-touch produces TWO trade rows: one with entry
  at the OB proximal edge, one at the OB 50% mean. Same SL (OB distal),
  same TP price levels (opposing H1 swing liquidity, reused from live
  `compute_phase2_levels`). R-distance halves on the 50% entry, so RR
  doubles for the same TP.
- Logs `r_if_exit_tp1` AND `r_if_exit_tp2` for every trade so the user can
  see TP1-only behaviour vs default TP2 side by side.

## Running

### Via GitHub Actions (recommended)
- Actions tab → "Backtest" workflow → "Run workflow" → fill date range.
- Output: artifacts attached, run log committed back to repo.

### Locally
```
python backtest/run_backtest.py --start 2026-05-12 --end 2026-05-16 --regime war
```

### H1-only sanity test
```
python -m backtest.test_h1_only
```
Asserts the level computation produces the expected proximal vs 50% entries
on a synthetic OB. Runs in <2s. Same script runs on CI before every backtest.

## Files

- `data_loader.py` — yfinance fetch + parquet cache.
- `replay_engine.py` — bar-by-bar H1 walk with lookahead guard.
- `h1_only_simulator.py` — H1-only dual-entry simulator.
- `h1_only_reporting.py` — report writer with TP1/TP2 side-by-side
  scoreboard and score-vs-winrate diagnostic table.
- `reporting_email.py` — own SMTP, no live email reuse.
- `run_backtest.py` — CLI entry.
- `test_h1_only.py` — synthetic-OB sanity tests for H1-only path.

## Output structure

### auto mode
```
backtest/results/<regime>_<start>_<end>/
  forex_trades.xlsx
  nas_xau_trades.xlsx
  zone_register.xlsx       # every OB ever active + disposition
  report.html
  raw_alerts.jsonl
  summary.json
```

### h1_only mode
```
backtest/results/h1only_<start>_<end>/
  trades.csv               # full column set, one row per (OB, entry zone)
  trades.xlsx              # same data, Excel-formatted
  report.html              # 4 scoreboards + score-vs-winrate buckets
  raw_alerts.jsonl
  summary.json
```
