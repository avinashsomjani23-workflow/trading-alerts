# CANONICAL BACKTEST — the ONLY analysis CSV

Any analysis, finding, or column question uses THIS file and no other:

    backtest/results/h1only_20080102_20161231/trades.csv

- Run: `h1only_20080102_20161231` — EURUSD only, 2008-01-02 → 2016-12-31 (Discovery
  block). **Re-run 2026-08-11** with the structure fix `f302674e` ("never fire
  CHoCH/BOS on an already-broken swing") — false breaks on spent swings no longer
  create order blocks, so alerts dropped 3,322 → 3,041 (−8.5%); WR and expectancy held
  flat (a clean removal of noise, not a distortion). Every pre-2026-08-11 number on this
  window is VOID and must be re-derived from THIS CSV. Rebuilt 2026-07-31 under the
  **FIXED 2R baseline**
  (`docs/FIXED_2R_BASELINE_SPEC.md`): exit = fixed +2R / −1R, no BE, no liquidity-pool
  TP; window-MFE decoupled from the exit; SL-anatomy re-anchored to +2R/+1R. All 10
  scan-log gates PASS. Local-only (OneDrive policy — not git-tracked; commit in CI).
- Shape: **184 columns, 3,041 rows.** All run-produced. (Row count updated 2026-08-11
  from 3,322 after the structure re-run above; column count unchanged — the fix touched
  no columns, added no flag.) (Column count corrected 2026-08-02 from a
  stale "180": commit `c555fa71` added the four MFE_FIX_PLAN columns — `mfe_intrade_r`,
  `sl_bar_best_favor_r`, `sl_bar_reached_1r_ambiguous`, `ob_penetration_depth` (all
  `outcome`-timed) — taking 180 → 184, but this shape line was not repointed. Earlier
  the `ranging delete` dropped `structure_ranging_at_alert` and `entry_raw` was retired
  (182 → 180). Verified by the CSV parser against the live header, not a naive comma
  count.)
- Per-year alert count: 318–353 (flat — noted, not yet acted on).
- Expectancy −0.098R (headline −$50,293.50) — NEGATIVE by design; the fixed 2R is a
  constant ruler for studying ENTRIES, not a profitable exit. Every pre-2026-08-11
  backtest number is VOID (structure detection changed — see run note above).

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
- Before using it, confirm the header has **184 columns**. Wrong count = wrong file
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
