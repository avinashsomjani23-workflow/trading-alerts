# CANONICAL BACKTEST — the ONLY analysis CSV

Any analysis, finding, or column question uses THIS file and no other:

    backtest/results/h1only_20080102_20161231/trades.csv

- Run: `h1only_20080102_20161231` — EURUSD only, 2008-01-02 → 2016-12-31 (Discovery
  block). Rebuilt 2026-07-31 under the **FIXED 2R baseline**
  (`docs/FIXED_2R_BASELINE_SPEC.md`): exit = fixed +2R / −1R, no BE, no liquidity-pool
  TP; window-MFE decoupled from the exit; SL-anatomy re-anchored to +2R/+1R. All 10
  scan-log gates PASS. Local-only (OneDrive policy — not git-tracked; commit in CI).
- Shape: **180 columns, 3,322 rows.** All run-produced. (Corrected 2026-08-02 from a
  stale "182": the `ranging delete` dropped `structure_ranging_at_alert` and `entry_raw`
  was retired, but this shape line was never repointed. Verified by the CSV parser
  against the live header, not a naive comma count.)
- Per-year trade count: 346–391 (flat — noted, not yet acted on).
- Expectancy −0.09R (headline −$51,597.50) — NEGATIVE by design; the fixed 2R is a
  constant ruler for studying ENTRIES, not a profitable exit. Every pre-2026-07-31
  backtest number is VOID (exit changed).

## Canonical = the CURRENT active analysis run (repointed per run)

- Canonical is NOT one frozen CSV forever. Its job is to name the ONE truth file
  for the analysis in flight, and be REPOINTED whenever that moves.
- When the analysis moves to a different window or pair (e.g. EURUSD Validation
  years, or USDJPY Discovery), that run becomes a DIFFERENT file. Repoint this doc
  — path + shape — in the SAME commit that produces it, and archive the run it
  supersedes (see below). Never let two "truth" files coexist.

## Rules (non-negotiable)

- NEVER `glob` for `trades.csv` and use whatever turns up. There is exactly one
  truth file — the path above.
- Before using it, confirm the header has **180 columns**. Wrong count = wrong file
  = STOP. (This number changes when canonical is repointed — always re-confirm it
  against whatever this doc currently names. Count with the CSV parser, e.g.
  `python -m backtest.gen_column_buckets --check`, NOT a naive `tr ',' '\n' | wc -l`
  — quoted fields make the naive count over-read.)
- `backtest/archive/` is STALE by definition. NEVER read any CSV under it as an
  analysis source. It exists only for recoverability. See `backtest/archive/README.md`.
- Any other `trades.csv` that reappears under `backtest/results/` is a fresh run in
  progress or a stale artifact. It is NOT truth until this file is updated to name it.
- Column meanings come from `TRUTH_LEDGER.md` (file:line), read against live code.
  Not from any other doc.

## When a new canonical run is made

- Update the path and shape above IN THE SAME COMMIT that produces it.
- MOVE the superseded run to `backtest/archive/` — do not delete, do not let old
  CSVs accumulate under `results/` (that was the July-2026 bad-data trap).
