# Backtest Diagnostic Harnesses

Three diagnostic tools that test the backtest WITHOUT changing any trading
logic. They import and call the live functions; they never re-implement
detection. Built from `SPEC_driver_and_harness3.md` and
`SPEC_harness1_harness2_tierB.md` (Fable design), against the ground truth in
`FABLE_REFERENCE.md`.

All output lands in `backtest/diagnostics/out/`. Everything reconciles to
`r_realised` (the single P&L source of truth). Runs on Python 3.14 from
PowerShell.

> **Speed note:** a backtest walk is ~0.7s per H1 bar (the live structure engine
> recomputes over the whole slice each bar â the real backtest pays this too).
> A 1-month single-pair window â‰ˆ 6 min per walk. Use **short windows / few
> pairs** for quick looks; the data loader serves only ~the last 2 years (1h)
> and uses a parquet cache, so repeats are cheap.

---

## driver.py â shared engine (no CLI)
The foundation every harness uses. Bar-by-bar replay with strict slice
discipline, a knob-override context manager (handles the `MIN_LEG_ATR_MULT`
def-time-default trap), and a self-check. Sanity it directly:
```powershell
python -m backtest.diagnostics.driver
```

---

## Harness 1 â ATR-knob sweep
Sweep ONE knob across a grid; see how swings/OBs/CHoCH/BOS/alerts/P&L move.
Invariant columns are asserted identical (a moving invariant = escalates).
```powershell
python -m backtest.diagnostics.h1_knob_sweep --knob MIN_LEG_ATR_MULT --grid 1.0,1.5,2.0 --pairs EURUSD --start 2026-03-01 --end 2026-03-31
```
- `--knob` one of: MIN_LEG_ATR_MULT, BOS_ATR_MULT, STRUCTURE_CHOCH_ATR_MULT,
  STRUCTURE_LOCK_ATR_MULT, OB_MAX_RANGE_ATR_MULT, FVG_NOISE_FLOOR_MULT,
  SWEEP_EQUAL_LEVEL_TOLERANCE_ATR, SWEEP_WICK_PIERCE_MIN_ATR, PROXIMITY_CAP.
  (REARM_EXTRA_ATR is refused â see report; needs a 1-line live hoist first.)
- Dict knobs (FVG/sweep) take a uniform multiplier by default; add
  `--grid-mode absolute` to set the same absolute value per pair-type.
- Output: `out/h1_sweep_<knob>_<stamp>.csv` + `.md`.
- Remember: proximity/score knobs leave structure columns IDENTICAL BY DESIGN;
  that's truth, not a bug (the report annotates it).

---

## Harness 2 â swing-detection noise audit
Shows which swings the ATR gate keeps vs drops, each swing's gate-crossing
multiplier `M*`, and a variant grid. Fast (no replay).
```powershell
python -m backtest.diagnostics.h2_swing_audit --pairs EURUSD --start 2026-03-01 --end 2026-03-31 --lookbacks 3 --mults 1.0,1.25,1.5,1.75,2.0
```
- Output: `out/h2_swings_<pair>_<stamp>.csv` (per-swing eyeball file),
  `out/h2_variants_summary_<stamp>.csv`, `out/h2_borderline_<stamp>.md`
  (the swings near the live 1.5 gate worth a chart check).

---

## Harness 3 â parity / look-ahead audit (the important one)
Proves on real data that the backtest doesn't see the future and doesn't make
"the event the trade." Three pillars + read-only verifier + a pre-flight that
must catch a planted bug before it runs.
```powershell
python -m backtest.diagnostics.h3_parity_audit --pairs EURUSD --start 2026-03-01 --end 2026-03-31
```
- `--fast` skips the Tier-A second walk (quick smoke).
- `--max-truncation-samples N` bounds the truncation oracle (default 30).
- Output: `out/h3_report.md` (ranked, plain-English verdict) + `out/h3_findings.csv`.
- Every guard gets a row even when clean; nothing is ever auto-fixed
  (`status=NOT_APPLIED`). Proposed fixes are text only â you approve/reject.

---

## Tier-B â live vs backtest (real records)
Reconstructs historical LIVE alerts from git history (read-only) and lines them
up against the backtest.
```powershell
# 1) extract live records to out/live_alerts.csv
python -m backtest.diagnostics.h3_live_extract --out backtest\diagnostics\out
# 2) match a window
python -m backtest.diagnostics.h3_live_extract --match --pairs EURUSD --start 2026-05-01 --end 2026-05-15
```
- Live history before the parity field-add is **proximity-only** (no
  entry/SL/TP). The field-add (already in `Phase2_Alert_Engine.py`) records
  trade levels going forward, so trade-level parity improves over time.
- Below ~30 matched records, any verdict is "held on N=â€¦", never "proven."

---

## What was changed in the live system (additive logging only)
`Phase2_Alert_Engine.py` â the scan-log `zone_outcome` now also records
`distal` + `ob_timestamp`, and a `fired_levels` block (entry/sl/tp1/tp2/rr) for
alerts that fire. These are **log fields only** â no decision, branch, or
detection logic changed. They exist so Tier-B can match OB identity and trade
levels apples-to-apples on future live data.
