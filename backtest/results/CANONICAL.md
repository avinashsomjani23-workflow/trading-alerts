# CANONICAL BACKTEST — the ONLY analysis CSV

Any analysis, finding, or column question uses THIS file and no other:

    backtest/results/h1only_20080102_20251231/trades.csv

- Commit: `1feb2db` (2026-07-09), git-clean / reproducible
- Shape: 113 columns, 33,838 rows. As of 2026-07-16 (Part D) all 113 are
  RUN-PRODUCED: the 5 news columns (`news_fill`…) are now stamped natively by
  the report writer (`h1_only_reporting._enrich_news_columns`, deterministic,
  idempotent join from `backtest/data/ff_calendar_2007_2026.csv`) — no separate
  post-hoc script. Re-scrape that events file before each baseline so it covers
  the run window (the one maintenance seam; the run fails loud otherwise).
  NOTE: THIS canonical CSV predates the clock fix — its news columns were
  stamped by the OLD post-hoc script (provisional labels + in-enrichment
  correction). The next fresh run is the first to pair true-UTC candles
  (import_mt5 Part B) with the no-double-correction enrichment (Part C).
- Discovery-eligible: 11,366

> **STALE FOR DETECTION as of 2026-07-10.** This run was built with the break gates
> ON. The gates were removed that day (dealing_range.py + smc_radar.py — BOS/CHoCH
> distance + body). Detection-derived columns here NO LONGER match live code. Use
> this CSV only for detection-independent plumbing checks until a fresh canonical run
> is made. (2026-07-10: the earlier "fresh run adds break_dist_atr →109 cols" note is
> RETRACTED — break_dist_atr == break_close_atr, owner-confirmed; no new column, stays
> 108 unless something else is added.)

> **PENDING SCHEMA (+2 cols) as of 2026-07-25.** Two new BACKTEST-ONLY fill-time
> columns were added to `_build_row` — `atr_regime_pct_at_fill` and
> `atr_regime_ratio_at_fill` (era-stable volatility-regime read; see TRUTH_LEDGER.md).
> They are NOT in THIS frozen CSV. The NEXT fresh canonical run is the first to
> emit them: when it lands, update the shape line above to the new count in the
> SAME commit that produces it. (The two columns land next to `atr_at_fill`.)

> **PENDING SCHEMA (+2 more cols) as of 2026-07-26.** Two new BACKTEST-ONLY
> OUTCOME-time columns were added to `_build_row` — `bars_to_mfe` and `bars_to_mae`
> (H1 bars from fill to the MFE/MAE bar; loser autopsy / time-stop input; see
> TRUTH_LEDGER.md). They are NOT in THIS frozen CSV. **NEVER a model/entry feature —
> pure look-ahead** (walled in `edge_lab.OUTCOME_TIME_FEATURES`). Combined with the
> +2 atr_regime note above, the next fresh canonical run emits **113 → 117 columns**:
> update the shape line and the header-count rule below to 117 in the SAME commit
> that produces it. (The two columns land next to `bars_to_exit`.)

## Rules (non-negotiable)

- NEVER `glob` for `trades.csv` and use whatever turns up. There is exactly one truth file — this one.
- Before using it, confirm the header has **113 columns**. Wrong count = wrong file = STOP.
- Any other `trades.csv` that ever reappears under `backtest/results/` is a fresh run in progress or a stale artifact. It is NOT truth until this file is updated to name it.
- Column meanings come from `TRUTH_LEDGER.md` (file:line), read against live code. Not from any other doc.

## When a new canonical run is made

- Update the path, commit, and shape above IN THE SAME COMMIT that produces it.
- Delete or move out the superseded run — do not let old CSVs accumulate again (that was the July-2026 bad-data trap).
