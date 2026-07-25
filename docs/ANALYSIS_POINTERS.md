# ANALYSIS POINTERS

Parked ideas + the working playbook, so nothing is forgotten at analysis time. Ideas only — no results, no findings.

---

## STANDING GUARDS

- **Per-pair isolation:** every pair's rules come ONLY from that pair's own Discovery block. Never from another pair, never from a later block.
- **Method-change log:** any change to the method made after seeing ANY pair's Holdout gets logged here, with date and reason.

---

## PLAYBOOK (EURUSD end-to-end first, then repeat per pair)

0. **Health only** — trade count sane, run ID, errors, count-vs-ATR flatness check. No edge reading.
1. **Pick the exit** — Discovery only, all trades, no signals. 13 recorded exit outcomes; winner must beat live baseline beyond its CI; tie → incumbent. Freeze.
2. **Relabel** — every Discovery trade → loss / breakeven / win under the frozen exit. No re-run.
3. **Loser autopsy** — died-fast vs gave-it-back split; full bucket curves (N, WR, mean R, straight-to-SL) per feature. Read, don't decide.
4. **Screening** — every logged + derived feature gets a curve with CI. Wide list.
5. **Model** — RF then XGBoost, 3-class (loss/BE/win). Tune knobs with expanding-window folds INSIDE Discovery (never on Validation — that stays sealed):
   - Fold 1: train 2008–2011 → test 2012
   - Fold 2: train 2008–2012 → test 2013
   - Fold 3: train 2008–2013 → test 2014
   - Fold 4: train 2008–2014 → test 2015
   - Fold 5: train 2008–2015 → test 2016
   - Always train earlier, test later. Knob setting with best AVERAGE test score across the 5 folds wins. (RF barely needs this; mainly for XGBoost, 2–3 knobs max.)
6. **EV** — EV = p(win)×tp1_rr − p(loss). EV floor = filter; EV tiers = sizing, capped 0.5×–1.5×.
7. **Exam sheet** — short pre-registered rule list + logbook count of everything tried. Written BEFORE touching Validation.
8. **Validation, one shot** — FDR q=0.10, calibration check, DSR on the assembled strategy.
9. **Cross-checks** — one pre-named alternate exit; sign check on other pairs' Discovery.
10. **Holdout 2022–2025, opened once.** Result stands. Next pair.

---

## POINTERS

**Format (fixed — every entry uses exactly these four lines):**
- **What:** one line.
- **Stage:** the playbook stage where it applies.
- **How:** one line — computation + data source.
- **Added:** date.

### Rolling win rate
- **What:** win rate of the last N trades at the moment of each alert — tests whether the system streaks hot/cold.
- **Stage:** 4 — Screening.
- **How:** sort the pair's trades by alert_ts; rolling mean of the win flag (start N=20), shifted by one so the current trade's own outcome is excluded. Derived from trades.csv — no run column needed.
- **Added:** 2026-07-25

### Weekly PD vs daily PD
- **What:** test which is the better setup-quality indicator per pair — position within the weekly range or within the prior daily range.
- **Stage:** 4 — Screening.
- **How:** bucket curves for both position features (weekly from the logged pool columns if present; daily from d1_pos_pct if logged); compare separation with CIs, per pair.
- **Added:** 2026-07-25

### Dealing range base timeframe: H4 vs D1
- **What:** DR is built on H4 swings today; candidate rebuild on D1 swings — a DETECTION change, not an analysis feature.
- **Stage:** post-10 — next generation, only if the autopsy shows DR-related misreads among losers.
- **How:** needs its own discovery+validation evidence; changes every downstream row, so never mid-analysis.
- **Added:** 2026-07-25
