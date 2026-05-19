# Backtest Harness

Replays historical bars through the live SMC detection modules to estimate what
the trading system *would have done* on a given week.

## Hard rules

- **Zero edits to live code.** This folder imports `smc_radar`, `dealing_range`,
  `smc_detector`, etc. read-only. It never writes to live JSON state files. All
  state writes go to `backtest/state/`.
- **No live email sent.** Backtest never calls live SMTP plumbing.
- **No lookahead.** Every dataframe slice fed to detection is asserted to end
  at or before the replay timestamp. Violation = hard fail.

## What it does

1. Fetch historical OHLC from yfinance for the requested date range.
2. Walk H1 bars one at a time. At each H1 close:
   - Slice H1 + M15 (+ M5 if available) up to that timestamp.
   - Feed slices to `dealing_range.update_pair` and `smc_radar.detect_smc_radar`.
   - Score the OB (mirrors Phase 2 confluence scoring — see compromise note below).
   - If score >= min_confidence, emit a "would-be alert".
3. For each emitted alert, simulate the trade:
   - Walk forward on M15 (or M5 for Phase 3) bars.
   - Track entry fill, SL hit, TP1/TP2 hit, MFE, MAE.
   - Conservative rule: same-bar SL + TP collision → SL hits first.
4. Generate Excel + HTML report with two regimes (per-pair view + Forex vs NAS/XAU split).

## Compromise: Phase 2 scoring

Phase 2's confluence scoring lives inline in the live `Phase2_Alert_Engine.py`
main block — not in a callable function. This harness mirrors that scoring
logic in `trade_simulator.py::score_ob_confluences()`. **If the live scoring
changes, this mirror must be updated.** See `KNOWN_LIMITATIONS.md`.

Why this compromise: refactoring the live scoring into a callable function
would mean editing the live codebase — explicit hard rule violation. The
alternative — running the live `Phase2_Alert_Engine.py` as a subprocess with
patched data sources — is fragile and slow. Mirroring the scoring is the
honest middle path. The harness logs every confluence input so divergence
from live is auditable.

## Running

### Via GitHub Actions (recommended)
- Actions tab → "Backtest" workflow → "Run workflow" → fill start date + end date.
- Output: Excel + HTML attached as run artifact; emailed if email is configured.

### Locally
```
python backtest/run_backtest.py --start 2025-09-15 --end 2025-09-19 --regime war
```

## Files

- `data_loader.py` — yfinance fetch + parquet cache.
- `replay_engine.py` — bar-by-bar walk with lookahead guard.
- `trade_simulator.py` — entry fill + SL/TP walk + MFE/MAE tracking.
- `reporting.py` — Excel + HTML report assembly.
- `reporting_email.py` — own SMTP, no live email reuse.
- `run_backtest.py` — CLI entry.

## Output structure

```
backtest/results/<run_id>/
  forex_trades.xlsx
  nas_xau_trades.xlsx
  report.html
  raw_alerts.jsonl    # every OB the system saw, scored or not
  summary.json        # headline metrics for the run
```
