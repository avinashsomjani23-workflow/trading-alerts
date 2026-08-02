# ANALYSIS POINTERS

Parked ideas + the working playbook, so nothing is forgotten at analysis time. Ideas only — no results, no findings.

> **Baseline exit = FIXED 2R** (2026-07-31, `docs/FIXED_2R_BASELINE_SPEC.md`). The run's exit is a
> constant +2R / −1R bracket (no BE, no trail, no liquidity-pool TP). Playbook step 1 ("pick the
> exit") is effectively frozen to fixed 2R for the entry-study phase; EV in step 6 is
> `p(win)×2 − p(loss)×1` (the fixed 2R:1R payoff), not a per-trade `tp1_rr`.

---

## STANDING GUARDS

- **Per-pair isolation:** every pair's rules come ONLY from that pair's own Discovery block. Never from another pair, never from a later block.
- **Method-change log:** any change to the method made after seeing ANY pair's Holdout gets logged here, with date and reason.
- **FDR/DSR never hide a result:** FDR and DSR are LABELS on the report, never a filter on what you see. Every signal — significant or not — is shown with its effect size and CI. A weak result gets TAGGED "thin / likely fluke", it is NEVER dropped or skipped from the report. Nothing is withheld from the human because of a significance test. (This is the "log the near-miss, don't bin it" rule in force.)
- **Regime-change is NOT permission to trust recent data:** if a signal works in one block and fails in a later one, that is a red flag about robustness — NOT a reason to re-weight toward the recent block. Whether recency matters is decided IN ADVANCE (before opening any sealed block) and baked into sampling. Deciding to trust recent data AFTER seeing it burns the one-shot test (Rule 3). Demand the signal hold across multiple regimes (per-quarter / per-decade consistency); a signal riding one regime is fragile no matter how good its numbers.

---

## HOW WE JUDGE SIGNIFICANCE — the ladder (plain English)

Read top to bottom. Effect size first; the p-value is the appetizer, out-of-sample is the verdict.

1. **Effect size — is it worth caring about?** The number itself, in money terms: win-rate lift (baseline 40% → 52% = **+12pp**) and mean-R lift (−0.09R → +0.15R = **+0.24R**). This is the "so what". Lead with it. A signal that is statistically solid but tiny is not worth trading.
2. **Bootstrap CI — is that size trustworthy or thin?** Resample our actual trades (no bell-curve assumption — our per-trade outcome is a 2-spike +2R/−1R world, not a bell curve). Tight CI → real. Wide CI straddling zero → thin, don't act.
3. **FDR (ranking dial only) — of the signals I flagged, which are most likely real?** Screen many columns and, by luck alone, ~1 in 20 looks good at p<0.05. FDR tags the likely flukes. It **ranks**, it never gates or hides (see Standing Guard). The p-value lives HERE, as a helper — never as the conclusion.
4. **Consistency (per-quarter / per-decade) — luck of one stretch, or robust?** Does the edge hold across regimes, or is it one lucky window? The Stage-5 expanding-window folds are the heavyweight ("walk-forward") version of this.
5. **Out-of-sample survival — the verdict.** Validation → Holdout, sealed. This is the conclusion, NOT the p-value. A signal that survives sealed data is real even with a mediocre p-value; one that fails sealed data is dead even with a beautiful one.
6. **DSR — final trophy only.** When ONE combined strategy is assembled, DSR discounts its headline number for how many strategies were tried to find it. Different job from FDR: **FDR sorts the pile of signals; DSR stress-tests the single trophy.** You need only one at a time — FDR while screening a list, DSR on the one final strategy.

---

## STAGE 0 SIGN-OFF — EURUSD Discovery (FIXED 2R run), closed 2026-08-01

Health only, no edge reading. Run `h1only_20080102_20161231`, EURUSD, 2008-01-11 → 2016-12-30.
Canonical was already repointed to this run in commit 71b08ea4 — no re-point needed this turn.

- **Shape:** 182 columns × 3,322 rows. Matches CANONICAL.md exactly. Header re-counted, not trusted.
- **Run errors:** zero error/exception/traceback lines in run_log.jsonl; console.log empty. Clean.
- **Trade count sane:** 3,322 alerts, one pair, LONG 1,682 / SHORT 1,640 (near-balanced). No null alert_ts.
- **Count-vs-ATR flatness:** per-year 346–391 (2008 379, 2009 349, 2010 381, 2011 391, 2012 360,
  2013 376, 2014 359, 2015 381, 2016 346), mean 369. Flat across the block — NOTED, not acted on
  (flatness itself carries no edge; it's just confirmation the detector fired evenly year to year).
- **Exit accounting (all 3,322):** sl 1,512 · never_filled 1,094 · tp 641 · friday_flat 68 · timeout 7.
  never_filled is audit-only, correctly excluded from P&L.
- **FIXED-2R invariants (spec §H) verified on live data:**
  - every tp row `r_realised == +2.0` (641/641); every sl row `r_realised == −1.0` (1,512/1,512). Clean bracket.
  - `mfe_r >= r_realised`: holds for ALL 641 tp + 1,512 sl trades. 4 exceptions, ALL `friday_flat`
    (booked a tiny +R at the Friday flat-close while window-MFE peak logged 0.0) — a flat-close quirk,
    not a decouple bug; touches no entry-study population. LOGGED, not a blocker.
  - Language grep on header: no EXIT/OUTCOME column whispers tp1/pool/wick/BE. The regex hits are all
    SETUP descriptors (pool-distance, liq-stop setup, sweep, sl_wick_depth_atr) — allowed per §H.
- **Headline reconciles:** resolved WR 29.8% (641/2,153); mean r_realised across booked trades −0.091R,
  matching CANONICAL.md's −0.09R. Negative BY DESIGN (constant 2R ruler; entry work must lift it).
- **Doc nit (not a blocker):** CANONICAL.md line ~11 says the CSV is "not git-tracked"; it actually IS
  tracked (the `backtest/results/` gitignore was overridden for this run). Cosmetic — flag if it matters.

**Verdict:** Stage 0 GREEN. Cleared to proceed to Stage 1 (pick the exit) when instructed. No edge read.

---

## PLAYBOOK (EURUSD end-to-end first, then repeat per pair)

0. **Health only** — trade count sane, run ID, errors, count-vs-ATR flatness check. No edge reading.
1. **Pick the exit** — Discovery only, all trades, no signals. 13 recorded exit outcomes; winner must beat live baseline beyond its CI; tie → incumbent. Freeze.
2. **Relabel** — every Discovery trade → loss / breakeven / win under the frozen exit. No re-run.
3. **Loser autopsy** — died-fast vs gave-it-back split; full bucket curves (N, WR, mean R, straight-to-SL) per feature. Read, don't decide.
4. **Screening** — every logged + derived feature gets a curve with CI. Wide list. **Lead with EFFECT SIZE**: report the WR-lift (in percentage points, e.g. 40%→52% = +12pp) AND the mean-R lift (e.g. −0.09R→+0.15R = +0.24R) — the number itself, not just "significant/not". Every feature is reported whether or not it clears significance; a weak one is TAGGED thin, never dropped (see Standing Guard).
5. **Model** — RF then XGBoost, 3-class (loss/BE/win). Tune knobs with expanding-window folds INSIDE Discovery (this is walk-forward — the gold-standard consistency test; always train earlier, test later; never on Validation — that stays sealed):
   - Fold 1: train 2008–2011 → test 2012
   - Fold 2: train 2008–2012 → test 2013
   - Fold 3: train 2008–2013 → test 2014
   - Fold 4: train 2008–2014 → test 2015
   - Fold 5: train 2008–2015 → test 2016
   - Always train earlier, test later. Knob setting with best AVERAGE test score across the 5 folds wins. (RF barely needs this; mainly for XGBoost, 2–3 knobs max.)
6. **EV** — EV = p(win)×tp1_rr − p(loss). EV floor = filter; EV tiers = sizing, capped 0.5×–1.5×.
7. **Exam sheet** — short pre-registered rule list + logbook count of everything tried. Written BEFORE touching Validation.
8. **Validation, one shot** — FDR q=0.10 (ranking/tagging the list of signals — never hides one), calibration check, DSR on the assembled strategy (stress-tests the ONE final trophy for how many strategies were tried). Different jobs: FDR sorts the pile, DSR discounts the trophy — not redundant.
9. **Cross-checks** — one pre-named alternate exit; sign check on other pairs' Discovery.
10. **Holdout 2022–2025, opened once.** Result stands. Next pair.

---

## POINTERS

**Plot rule (every result chart):** label BOTH axes (never leave X unnamed) and write the EFFECT SIZE on the chart itself (the WR-lift / mean-R-lift number), so a plot states its own "so what" at a glance.

**Format (fixed — every entry uses exactly these four lines):**
- **What:** one line.
- **Stage:** the playbook stage where it applies.
- **How:** one line — computation + data source.
- **Added:** date.

### PARKED — block bootstrap for drawdown/equity questions (NOT for screening)
- **What:** **PARKED, not needed now.** Plain bootstrap resamples single trades; block bootstrap resamples CHUNKS of consecutive trades, preserving the fact that losses CLUSTER in time. The 2R cap removes the single-trade monster but NOT the losing-streak monster (10 −1R in a row). Screening (does a filter separate winners from losers?) is a per-trade comparison where clustering barely matters — plain bootstrap is fine. Block bootstrap only earns its keep for equity-path / max-drawdown questions.
- **Stage:** post-8 — final-strategy drawdown/equity estimation only. Never for feature screening.
- **How:** resample overlapping blocks (chunk length ≈ typical streak) of the final strategy's trade sequence; read max-drawdown / equity-curve distribution, not WR separation.
- **Added:** 2026-08-02

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

### Big-winner anatomy (trades that reach 2R+) — the target-distance axis
- **What:** two Stage-1 exit recipes independently pointed at the SAME axis — a bigger, further target trades a lower hit-rate for larger wins and MORE positive quarters. Evidence (EURUSD Discovery h1only_20080102_20161231): (a) fixed-2R TP = 8/36 positive quarters vs baseline's 3/36; (b) single-target wick TP = lower WR 27.8% vs 33.3% but bigger avg win +1.86R vs +1.40R on the same 1401 trades. Goal: find what a big winner shares, so entries can be biased toward the setups that actually run far.
- **Stage:** 3 — Loser autopsy / 4 — Screening (winner side).
- **How:** flag trades whose real-order replay (walk_multileg, not raw MFE) reaches ≥2R; compare their feature distributions (break quality, PD zone, session, sweep, pool distance) vs the rest; bucket curves per feature with CI. Derived from trades.csv + exit replay.
- **Added:** 2026-07-27

### CAVEAT — exit choice must not be frozen on the raw (unfiltered) population
- **What:** every exit recipe LOSES on the Discovery block because it contains all bad entries. The mechanical ATR exit "won" (−0.138 vs baseline −0.188, paired CI [+0.003,+0.084]) largely because a wide 1.5-ATR stop tolerates the sweeps that stop out our tight structural stop on GARBAGE setups — i.e. it is compensating for bad entries, not proving a better exit. Its positive quarters also cluster post-2013 (2012Q3, then 2014–2016), so the edge may be regime-specific. Do NOT crown an exit at Stage 1 on this population.
- **Stage:** 1 — Pick the exit (deferral rule); re-decided after Stage 4 entry filtering.
- **How:** carry BOTH finalists (baseline + E_atr_sl1.5_tp2.5) forward. Re-run the paired exit contest on the FILTERED / EV-gated book and on Validation years. ATR ships only if it still beats baseline beyond CI once trash entries are removed; if its edge collapses, it was an entry-quality artifact and baseline holds.
- **Added:** 2026-07-27

### ATR-scaled position size (risk-per-trade tied to volatility)
- **What:** hypothesis — because the whole system is ATR-gated (ATR governs which setups even alert), position/risk sizing may belong on the same ATR axis rather than flat-per-trade. Test ONLY after bad entries are filtered (Stage 4+); a sizing edge on a losing raw population is meaningless.
- **Stage:** post-6 — EV/sizing, and only on the filtered book.
- **How:** compare flat-R sizing vs size ∝ f(ATR-at-fill) on the surviving (EV-gated) trades; measure expectancy and drawdown, not raw expR. Sizing never changes which trades are taken — outcome-neutral to entry selection.
- **Added:** 2026-07-27

### Dealing range base timeframe: H4 vs D1
- **What:** DR is built on H4 swings today; candidate rebuild on D1 swings — a DETECTION change, not an analysis feature.
- **Stage:** post-10 — next generation, only if the autopsy shows DR-related misreads among losers.
- **How:** needs its own discovery+validation evidence; changes every downstream row, so never mid-analysis.
- **Added:** 2026-07-25

### Garbage-first cut (define obviously-bad setups before fine screening)
- **What:** raise the baseline and de-noise every later screen by first removing setups a vet rejects on sight, defined by SMC MECHANISM (not data-mined). News was the first such cut. Candidates: no room to target (opposing pool immediately in front, direction-aware), SL sitting inside an opposing liquidity pool, structurally-weak break, dead-hours tiny-target.
- **Stage:** before 4 — pre-screen. Screening on a garbage-laden book is noisy (see the exit-choice CAVEAT).
- **How:** each garbage rule stated as an SMC reason first, then measured on news-clean WR/meanR with CIs; a rule only ships if SMC and data agree. Direction-aware.
- **Added:** 2026-07-28

### Within-bucket winner/loser separation (interaction / ML)
- **What:** buckets with equal WR (e.g. ob_in_killzone=False: 332 wins vs 506 losers) still hide a price/structure feature that splits winners from straight-to-stop losers. Find it in combination, not singly.
- **Stage:** 5 — Model (RF/XGBoost, interactions). Not a single-feature screen.
- **How:** train on news-clean; target = win vs loss; inspect splits inside equal-WR buckets. CI-overlap near-misses from Stage 4 feed here.
- **Added:** 2026-07-28

### Pool-distance features are direction-blind
- **What:** dist_next_pool_above / _below carry no LONG/SHORT meaning as-is. A pool above is a TARGET for a long but a STOP hazard for a short. Any pool-distance reading must be re-expressed as toward-target vs behind-stop, per direction.
- **Stage:** 4 — Screening (re-derive before use).
- **How:** on news-clean, fold `bias` (LONG/SHORT) into pool distance → distance-to-target-pool and distance-to-stop-side-pool; then curve WR/meanR with CIs.
- **Added:** 2026-07-28

### OB freshness: H4-OB-reversal-without-H1-touch hypothesis
- **What:** older OBs (ob_age_h1_bars = OB→ALERT bars) may lose more because price reversed off the H4 OB without ever touching the H1 OB, then returned days later once the move's fuel was spent → dies on arrival. Hypothesis only.
- **Stage:** 4 — Screening (freshness), news-clean.
- **How:** cross ob_age_h1_bars with whether the H4 level was hit first / displacement since OB; WR+meanR curves with CIs. Direction-aware.
- **Added:** 2026-07-28

### Trend alignment should carry a money signal if detection is right
- **What:** trend_alignment showed only death-texture separation so far, not money. If H1-trend detection is correct, with-trend should beat counter-trend on WR/meanR. If it doesn't, suspect the trend detector, not the market.
- **Stage:** 4 — Screening, news-clean. Data+SMC-disagreement → discussion point, not a filter.
- **How:** WR + meanR by trend_alignment on news-clean with CIs; if flat, audit the trend detector before concluding.
- **Added:** 2026-07-28

### PARKED — CHoCH "EITHER" confirmation (Path A + Path B, whichever fires first)
- **What:** **PARKED, not building.** Proposal (spec `docs/PATH_B_CONFIRM_SPEC.md`) adds a 2nd confirmation path — break the 2nd-prior counter-trend swing — and fires on whichever hits first, to catch the ~59% of CHoCHs Path A is blind to (straight momentum reversals; A's blindness verified live at dealing_range.py:1359-1364). Parked because: (1) the only evidence is a simple screen where A / B / EITHER all TIE (overlapping CIs, all near breakeven) — no proven edge; (2) EITHER's whole job is catching straight momentum drops, which is exactly where the big-single-break-candle loser texture lives, so the mechanism leans AGAINST it; (3) it's a detection change (voids canonical, forces a full rerun, changes which trades exist), not a quick build; (4) we are in analysis mode — measure what's built before adding detection. The coverage hole is real but "real hole ≠ filling it makes money."
- **Stage:** post-10 — next-generation detection, NOT a mid-analysis feature.
- **How to revisit:** only if current-book analysis shows momentum reversals are specifically where we lose edge. Then build EITHER behind an off-by-default flag (`CONFIRM_MODE`, live byte-identical), run one full Discovery rerun into a sibling folder (do NOT repoint canonical), head-to-head vs the A book, and specifically test whether the trades EITHER *adds* concentrate in the big-break loser cell. Revert the flag if it doesn't help.
- **Added:** 2026-07-31
