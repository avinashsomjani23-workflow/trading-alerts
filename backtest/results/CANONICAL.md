# CANONICAL BACKTEST — the ONLY analysis CSV

Any analysis, finding, or column question uses THIS file and no other:

    backtest/results/h1only_20080102_20251231/trades.csv

- Commit: `1feb2db` (2026-07-09), git-clean / reproducible
- Shape: 108 columns, 33,838 rows
- Discovery-eligible: 11,366

> **STALE FOR DETECTION as of 2026-07-10.** This run was built with the break gates
> ON. The gates were removed that day (dealing_range.py + smc_radar.py — BOS/CHoCH
> distance + body). Detection-derived columns here NO LONGER match live code. Use
> this CSV only for detection-independent plumbing checks until a fresh canonical run
> is made. (2026-07-10: the earlier "fresh run adds break_dist_atr →109 cols" note is
> RETRACTED — break_dist_atr == break_close_atr, owner-confirmed; no new column, stays
> 108 unless something else is added.)

## Rules (non-negotiable)

- NEVER `glob` for `trades.csv` and use whatever turns up. There is exactly one truth file — this one.
- Before using it, confirm the header has **108 columns**. Wrong count = wrong file = STOP.
- Any other `trades.csv` that ever reappears under `backtest/results/` is a fresh run in progress or a stale artifact. It is NOT truth until this file is updated to name it.
- Column meanings come from `TRUTH_LEDGER.md` (file:line), read against live code. Not from any other doc.

## When a new canonical run is made

- Update the path, commit, and shape above IN THE SAME COMMIT that produces it.
- Delete or move out the superseded run — do not let old CSVs accumulate again (that was the July-2026 bad-data trap).
