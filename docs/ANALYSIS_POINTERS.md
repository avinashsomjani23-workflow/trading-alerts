# ANALYSIS POINTERS

Parked ideas + the working playbook, so nothing is forgotten at analysis time. Ideas only — no results, no findings.

> **Baseline exit = FIXED 2R** (2026-07-31, `docs/FIXED_2R_BASELINE_SPEC.md`). The run's exit is a
> constant +2R / −1R bracket (no BE, no trail, no liquidity-pool TP). Playbook step 1 ("pick the
> exit") is effectively frozen to fixed 2R for the entry-study phase; EV in step 6 is
> `p(win)×2 − p(loss)×1` (the fixed 2R:1R payoff), not a per-trade `tp1_rr`.

---

## MANDATORY CHECKS — run EVERY time, not optional, not memory-dependent

These two run on EVERY setup we consider filtering and EVERY thing we propose to log as
"important to filter/keep later". They are STEPS in the work, not reminders. Skipping either =
the finding is incomplete and must not be logged or acted on. (Enforced, unlike the "gentle
reminder" guards below.)

- **CHECK 1 — Benchmark the bucket against the REST of the book (never against zero).** The whole
  fixed-2R book loses by design, so "this bucket has negative meanR" means nothing on its own. For
  EVERY bucket/filter, compute and SHOW: (a) bucket meanR; (b) rest-of-book meanR (everything
  except this bucket); (c) the excess = bucket − rest, per trade AND ×N as total-R; (d) whether the
  two CIs overlap. A filter is only "real" when the excess is meaningful AND its CI does not overlap
  the rest. Any keep/cut/log decision that does not display these four numbers is not finished.
- **CHECK 2 — Any web-search claim is split rigorous vs retail, every time.** No web finding is
  reported or logged without labelling each source rigorous-independent (peer-reviewed / academic /
  central-bank — method + sample + nothing to sell) vs retail-selling-something (vendor / course /
  signal group — for sale, tiny sample, no method), giving sample size, and — where they disagree —
  stating the rigorous conclusion and calling the retail number not-evidence. No rigorous source →
  say so; never dress up a vendor number as evidence.

Both checks are spelled out in full (with worked examples and how to invoke them) in the STANDING
GUARDS below and are wired into Playbook steps 3–4.

---

## PARKED KNOWLEDGE (read only when building the related thing)

- **Future to-do list → `docs/FUTURE_TODO_LIST.md`.** A running list of parked ideas/knowledge,
  each with a Trigger for when to open it. Item 1 = CTA / trend-following notes: what a CTA is
  and their 5-part recipe (trend signal, volatility-based sizing, diversification,
  let-winners-run exit, slow hold), for when we build a momentum/continuation entry, a
  risk-scaling/sizing layer, or a trailing/volatility exit. Key caveat baked in: their edge is
  proven SLOW (weeks–months) + diversified, so it does NOT prove an intraday continuation
  entry — that must be re-proven on our own Discovery data.

- **LOOK-AHEAD TRAP — continuation-fib fills must start at `alert_ts`, never `bos_timestamp`.**
  A continuation/fib entry study (shallow fib on the impulse leg; origin = OB proximal `entry`,
  extreme = `leg_extreme_at_alert`, r = 1.2×atr_at_ob, fixed 2R) once printed ~51% WR / +0.53R
  blind (60%+ at deep fib) — that was a LOOK-AHEAD ARTIFACT. The fib anchor `leg_extreme_at_alert`
  is the leg's structural top, confirmed only ~SWING_LOOKBACK (≈3) bars AFTER the peak forms
  (`displacement_leg.py` `_extreme_end_idx`) and re-stamped on every re-fire, so it is not known
  until `alert_ts` (always later than `bos_timestamp`: median 2h / mean 28h on EURUSD Discovery).
  Starting the fill walk at `bos_timestamp` lets the pre-alert rally fill a limit priced off a peak
  that had not happened yet. Tell-tale: WR climbs monotonically with fib depth (0.5→0.9 = 25%→61%
  under bos-start). The HONEST blind number (fills from `alert_ts`) on threshold ≥2.2 is ~30% WR,
  negative expR — below the 33.3% 2R breakeven. The 72–95% "never_filled-origin/runaway" bucket is
  outcome-selected = look-ahead, not selectable live; runner profit is real but needs an alert-time
  predictor tested with the alert_ts start.

---

## STANDING GUARDS

- **Per-pair isolation:** every pair's rules come ONLY from that pair's own Discovery block. Never from another pair, never from a later block.
- **Method-change log:** any change to the method made after seeing ANY pair's Holdout gets logged here, with date and reason.
- **FDR/DSR never hide a result:** FDR and DSR are LABELS on the report, never a filter on what you see. Every signal — significant or not — is shown with its effect size and CI. A weak result gets TAGGED "thin / likely fluke", it is NEVER dropped or skipped from the report. Nothing is withheld from the human because of a significance test. (This is the "log the near-miss, don't bin it" rule in force.)
- **Regime-change is NOT permission to trust recent data:** if a signal works in one block and fails in a later one, that is a red flag about robustness — NOT a reason to re-weight toward the recent block. Whether recency matters is decided IN ADVANCE (before opening any sealed block) and baked into sampling. Deciding to trust recent data AFTER seeing it burns the one-shot test (Rule 3). Demand the signal hold across multiple regimes (per-quarter / per-decade consistency); a signal riding one regime is fragile no matter how good its numbers.
- **No quartile/half bucketing, ever — not even a first pass.** Never screen a feature by splitting it into quartiles or halves. A quantile cut blurs straight through the real boundary and can mislabel a column "flat" when the effect lives inside one bin — throwing away a signal that would sharpen to significance once more data pools in (more pairs / the full 18-yr run). Look at the DISTRIBUTION and the win-rate trend across the raw values FIRST; that reveals whether there is a gradient and where it sits. Then bucket on the DOMAIN-MEANINGFUL threshold — the SMC boundary the number represents (e.g. body/wick at 0.5, stop-distance at 1.0 ATR, sweep "cleared the level" vs not) — and show a sensitivity check across ≥2 candidate edges. Bucket edges come from what the feature MEANS, never from where its percentiles fall.

- **Don't cut a bucket just because it loses — the whole 2R book loses by design (gentle reminder, not a gate).** On the fixed-2R ruler EVERY bucket has negative meanR, so "this bucket loses money" is never, on its own, a reason to drop it. Cut a bucket only when BOTH hold: (a) its meanR is worse than the REST of the book by a gap whose CIs do NOT overlap the rest — a REAL excess loss, not noise; AND (b) you have tried and failed to find a feature that separates its winners from its losers (Stage-5 within-bucket split). Cutting a bucket throws away its WINNERS too — always price that first (winners × 2R). Worked example (2026-08-04, RE-DERIVED 2026-08-11 on the structure-fix run): wicky OB meanR −0.199 vs rest −0.146 = only −0.053R/trade worse (×614 ≈ −32R), diffCI [−0.208,+0.100] overlaps 0 → NOT a real excess → do not cut, would forfeit 164 winners for a gap that may be zero. (High-vol-tight, once "better than rest", is now ≈EQUAL −0.168 vs −0.175 — so cutting it is roughly neutral, not clearly harmful; the point stands either way: no CI-separated excess to justify a cut. Old run: wicky −0.187/−0.141/−30R; high-vol-tight −0.117/−0.177.)

- **Web-evidence sourcing — separate rigorous from retail on EVERY web search.** RIGOROUS (weight it): peer-reviewed journals, academic / central-bank working papers — shows method + data + sample size, has nothing to sell, states uncertainty (CIs, out-of-sample caveats). RETAIL (report but don't weight): vendor blogs, course/indicator sellers, signal groups — red flags are something for sale, tiny/vague samples ("500 trades, 6 months"), no reproducible method, round confident win-rates, "discretionary / can't be backtested". PRESENT it by labelling each source rigorous-independent vs retail-selling-something, giving its sample size, and when they disagree stating the rigorous conclusion and calling the retail claim not-evidence. If no rigorous source exists, say so — never dress up a vendor number. (Trader can invoke with: "only cite peer-reviewed / label each source + sample size / separate rigorous from retail and tell me where they disagree".)

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

> **RE-DERIVED 2026-08-11 on the structure-fix re-run (3,322→3,041).** The 08-01 sign-off above is on the
> SUPERSEDED run; kept as the historical record. New health (all read against the current CSV):
> **184 cols × 3,041 rows.** Exit accounting: sl 1,401 · never_filled 988 · tp 589 · friday_flat 58 ·
> timeout 5. LONG 1,538 / SHORT 1,503 (near-balanced). Per-year 318–353 (still flat). Invariants hold:
> every tp `r_realised==+2.0` (589/589), every sl `==−1.0` (1,401/1,401). Resolved N=1,990, WR **29.6%**,
> mean r_realised **−0.112R** (LOOSE −0.104, STRICT −0.174). Still GREEN — clean removal of noise.

---

## PLAYBOOK (EURUSD end-to-end first, then repeat per pair)

0. **Health only** — trade count sane, run ID, errors, count-vs-ATR flatness check. No edge reading.
1. **Pick the exit** — Discovery only, all trades, no signals. 13 recorded exit outcomes; winner must beat live baseline beyond its CI; tie → incumbent. Freeze.
2. **Relabel** — every Discovery trade → loss / breakeven / win under the frozen exit. No re-run.
3. **Loser autopsy** — died-fast vs gave-it-back split; full bucket curves (N, WR, mean R, straight-to-SL) per feature. Read, don't decide. **MANDATORY CHECK 1** (bucket vs rest-of-book excess + CI-overlap) runs on any bucket flagged for cut/keep here.
4. **Screening** — every logged + derived feature gets a curve with CI. Wide list. **Lead with EFFECT SIZE**: report the WR-lift (in percentage points, e.g. 40%→52% = +12pp) AND the mean-R lift (e.g. −0.09R→+0.15R = +0.24R) — the number itself, not just "significant/not". Every feature is reported whether or not it clears significance; a weak one is TAGGED thin, never dropped (see Standing Guard). **MANDATORY CHECK 1 runs on every filter proposed here** (show bucket meanR, rest-of-book meanR, excess ×N, CI-overlap). Any web-sourced mechanism cited → **MANDATORY CHECK 2** (rigorous vs retail).
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
> **⚠️ NUMBERS STALE — old 3,322 run; NOT re-derived on the 2026-08-11 structure-fix run.** Needs the
> exit-replay harness (`walk_multileg` ≥2R), not yet rebuilt this session. Re-derive at Stage-1 exit work.
- **What:** two Stage-1 exit recipes independently pointed at the SAME axis — a bigger, further target trades a lower hit-rate for larger wins and MORE positive quarters. Evidence (EURUSD Discovery h1only_20080102_20161231, OLD RUN): (a) fixed-2R TP = 8/36 positive quarters vs baseline's 3/36; (b) single-target wick TP = lower WR 27.8% vs 33.3% but bigger avg win +1.86R vs +1.40R on the same 1401 trades. Goal: find what a big winner shares, so entries can be biased toward the setups that actually run far.
- **Stage:** 3 — Loser autopsy / 4 — Screening (winner side).
- **How:** flag trades whose real-order replay (walk_multileg, not raw MFE) reaches ≥2R; compare their feature distributions (break quality, PD zone, session, sweep, pool distance) vs the rest; bucket curves per feature with CI. Derived from trades.csv + exit replay.
- **Added:** 2026-07-27

### CAVEAT — exit choice must not be frozen on the raw (unfiltered) population
> **⚠️ NUMBERS STALE — old 3,322 run; NOT re-derived on the 2026-08-11 structure-fix run.** Needs the
> paired exit-contest re-walk, not yet rebuilt this session. The DEFERRAL RULE (don't crown an exit on the
> raw book) is method and stands regardless; only the ATR-vs-baseline numbers are stale.
- **What:** every exit recipe LOSES on the Discovery block because it contains all bad entries. The mechanical ATR exit "won" (−0.138 vs baseline −0.188, paired CI [+0.003,+0.084], OLD RUN) largely because a wide 1.5-ATR stop tolerates the sweeps that stop out our tight structural stop on GARBAGE setups — i.e. it is compensating for bad entries, not proving a better exit. Its positive quarters also cluster post-2013 (2012Q3, then 2014–2016), so the edge may be regime-specific. Do NOT crown an exit at Stage 1 on this population.
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

## STAGE 3/4 — STOP-HUNT WIDEN & GEOMETRY FINDINGS (EURUSD Discovery, logged 2026-08-04)

> **⚠️ ALL NUMBERS STALE — old 3,322 run (N=1254); NOT re-derived on the 2026-08-11 structure-fix run.**
> Every widen figure comes from the retired `scratchpad/resim2.py` re-walk, which no longer exists — the
> harness must be rebuilt to re-derive. These already ship nothing (all CIs straddle baseline); re-derive
> at the Stage-1 exit-geometry re-decide. The MECHANISM (widen only helps at imprecise/tight stops, hurts
> past ~0.25 ATR) is structural; the exact meanR/rescue counts are stale.

All numbers below: canonical `h1only_20080102_20161231`, EURUSD, STRICT news-clean
(`news_fill==0 & news_open==0`), resolved only (exit ∈ {tp,sl}), N=1254 (OLD RUN). Every widen number
is from a real H1-bar re-walk (`scratchpad/resim2.py`) that reproduces the CSV outcomes 100%
at width 0. Widen = stop moved D·ATR further, 2R target re-measured from the new (larger) risk,
constant-risk sizing (loss = −1R, win = +2R). LOGGED near-misses — every CI still straddles the
baseline, so nothing ships; carry to Validation / other pairs.

### Widen the stop a SMALL amount (~0.20 ATR) for messy / tight-stop setups
- **What:** re-walking the REAL H1 bars (`scratchpad/resim2.py`, reproduces the CSV outcomes 100%
  at width 0), a small stop-widen of ~0.15–0.20 ATR — 2R target re-measured from the new risk,
  constant-risk sizing (loss −1R, 2R win +2R) — lifts meanR only where the stop sits at an
  IMPRECISE or TIGHT level. It helps nowhere else and HURTS past ~0.25 ATR (the moved-out target
  breaks winners faster than the buffer rescues stop-outs).
- **Numbers (EURUSD Discovery, STRICT news-clean, resolved, N=1254; each at +0.20 ATR, 2R):**
  · wicky OB (`ob_body_ratio`<0.5), N=653: meanR **−0.187→−0.105**, WR 27.1→29.8%; rescued 31 of 476
    stop-outs, broke 13 winners → net +18 good flips × 3R = **+54R** (bucket book-R −122→−68).
  · ER≥0.5 & stop<1 ATR, N=296: meanR **−0.057→+0.02** — the ONLY above-water cell found.
  · high-vol (`atr_regime_pct_at_fill`≥50) & stop<1 ATR, N=248 (its shakeouts = 12% of ALL its
    trades): meanR **−0.117→−0.037**; at +0.50 ATR it is −0.190, WORSE than doing nothing.
  · decisive OB (body≥0.5): NO gain at any width — a clean level needs no buffer (the interaction proof).
- **1R vs 2R (settles it for the widen decision only — the general 2R>1R case is already established):**
  a nearer 1R target rescues MORE stop-outs (wicky OB 46 vs 31) but meanR is always WORSE (−0.177 vs
  −0.105) — the extra rescues limp to 1R and would die before 2R; half-payoff doesn't cover them.
  Keep 2R. Only the untested wide-SL-&-wide-OB cell could flip this (Prompt A).
- **Smart threshold (not a blanket number):** widen ≈ the bucket's OWN median shakeout-wick depth
  (`sl_wick_depth_atr` on `sl_swept_then_1r` rows): median ≈0.12–0.17, p75 ≈0.29–0.34 ATR → use
  ~0.15–0.20, never past p75. CEILING: only ~14% of losers are shakeouts (≈1 in 7); a safe buffer
  catches about half of those (≈1 in 15 of losers); the other ~86% are clean losses widening cannot
  save. Robust to counting unresolved trades as 0R (sensitivity ran).
- **Stage:** 3/4 — exit-geometry lever; revisit at Stage 1 exit re-decide. Every CI still TOUCHES
  the baseline → LOG, carry to Validation / other pairs, ship nothing. **Added:** 2026-08-04.

### PARKED — the wide-SL & wide-OB cell (the one 1R-could-still-win test)
- **What:** every widen test so far shows 2R beats 1R (1R rescues more stop-outs but the extra
  rescues limp to 1R and cost more than half-payoff covers). The ONE cell not yet tested is where
  the stop is ALREADY wide AND the OB is wide/imprecise — there a nearer 1R target might bank the
  move before a distant 2R gets swept back. Until that cell is run, "2R>1R" is proven everywhere
  EXCEPT here; do not over-generalise.
- **Stage:** 3/4 — exit-geometry lever, same re-walk harness (`scratchpad/resim2.py`).
- **How:** slice STRICT news-clean resolved on wide-SL (`sl_dist_atr_at_alert`≥1.0) AND wide-OB
  (`ob_body_ratio`<0.5 or a width proxy); re-walk 1R vs 2R at +0/0.20 ATR; compare meanR with
  bootstrap CIs. Ships nothing — measures whether 1R ever wins.
- **Added:** 2026-08-05

### Killzone marginals — LOG WITH A LOUD CAVEAT (unstable across exit changes)
- **OPEN FLAG (trader, 2026-08-05):** killzone marginals have now moved TWICE under changes that had
  NOTHING to do with killzone logic (the detection change; then the TP-location change) — each changes
  which trades exist, so the killzone read is FRAGILE. Treat any killzone number as provisional; do NOT
  build a killzone rule off one run. The old "OB-in-KZ wins better" is stale/quarantined, and the
  current numbers may be equally transient — re-measure after ANY exit/detection change before trusting.
- **DATA INTEGRITY — VERIFIED CLEAN (audit run on the prior 3,322-row run; the `ob_in_killzone` stamping
  logic is UNTOUCHED by the structure fix, so the "not a logging bug" conclusion still holds):** the
  sign-flip is NOT a logging bug. All rows
  with an `ob_timestamp` were checked THREE independent ways: (A) the live engine `smc_detector.ts_in_killzone`,
  (B) an independent hand-rolled NY-local-hour overlap (EURUSD KZ hours {2,3,4,7,8,9,10,11} America/New_York,
  DST self-resolving), (C) the logged `ob_in_killzone` column. **0 mismatches across all three.** The flip vs
  the old "OB-in-KZ wins better" is 100% the detection-change quarantine (Rule 1), not corruption. `ob_in_killzone`
  is stamped on OB-FORMATION time (`_ob_in_killzone` → `ob.ob_timestamp`, h1_only_simulator.py ~L164-243), so it
  is a true ALERT-time column. Script: scratchpad `kz_audit.py`.
- **Part of the KZ marginal is just zone-thickness, not time:** killzone OBs run thicker than dead-session OBs,
  and thickness ≈ stop-width (see thick-OB near-miss below) — so some of the OB-in-KZ WR gap is the wide-stop
  effect wearing a killzone costume, not a session-timing edge. Another reason not to build a KZ rule off one run.
- **Current run (do not carry forward without re-confirming; RE-DERIVED 2026-08-11 structure-fix run):**
  `killzone_alignment` — Both (OB+fill in KZ) WORST N=199 WR 21.1% meanR −0.367; Fill only BEST N=469
  WR 29.9% meanR −0.104; OB only N=203 WR 27.1% meanR −0.187; Neither N=291 WR 28.5% meanR −0.144.
  **Rank order IDENTICAL to 2026-08-04 (Both worst, Fill only best) — the numbers moved a THIRD time
  under a change unrelated to killzone logic, exactly confirming the fragility flag above. Old:
  Both N=220/22.3%/−0.332; Fill only N=486/30.2%/−0.093; OB only N=230/27.0%; Neither N=318/28.6%.**
- **"Both" is NOT the London–NY overlap:** a large share of Both trades fill 02:00–04:00 NY (London Open,
  before NY opens). So overlap-volatility literature does NOT explain Both=worst; mechanism unknown.
  (Exact fill-hour split was 67/220 on the old run; not re-counted on the 3,041-row run — directionally same.)
- **Evidence base:** rigorous (Ito & Hashimoto intraday seasonality) supports only "overlap = highest
  volatility"; the "killzone = higher WR" claim is retail/vendor (tiny samples, none peer-reviewed).
- **Stage:** 4 — screening. Log only; ship nothing; re-measure after any exit change. **Added:** 2026-08-04 (flag 2026-08-05).

### PARKED — CHoCH "EITHER" confirmation (Path A + Path B, whichever fires first)
- **What:** **PARKED, not building.** Proposal (spec `docs/PATH_B_CONFIRM_SPEC.md`) adds a 2nd confirmation path — break the 2nd-prior counter-trend swing — and fires on whichever hits first, to catch the ~59% of CHoCHs Path A is blind to (straight momentum reversals; A's blindness verified live at dealing_range.py:1359-1364). Parked because: (1) the only evidence is a simple screen where A / B / EITHER all TIE (overlapping CIs, all near breakeven) — no proven edge; (2) EITHER's whole job is catching straight momentum drops, which is exactly where the big-single-break-candle loser texture lives, so the mechanism leans AGAINST it; (3) it's a detection change (voids canonical, forces a full rerun, changes which trades exist), not a quick build; (4) we are in analysis mode — measure what's built before adding detection. The coverage hole is real but "real hole ≠ filling it makes money."
- **Stage:** post-10 — next-generation detection, NOT a mid-analysis feature.
- **How to revisit:** only if current-book analysis shows momentum reversals are specifically where we lose edge. Then build EITHER behind an off-by-default flag (`CONFIRM_MODE`, live byte-identical), run one full Discovery rerun into a sibling folder (do NOT repoint canonical), head-to-head vs the A book, and specifically test whether the trades EITHER *adds* concentrate in the big-break loser cell. Revert the flag if it doesn't help.
- **Added:** 2026-07-31

---

## TREND-FLIP / CHoCH-QUALITY FINDINGS (EURUSD Discovery, logged 2026-08-07; RE-DERIVED 2026-08-11)

Branch: "flip the H1 trend AT the CHoCH instead of waiting for the Confirmation BOS." All numbers:
canonical `h1only_20080102_20161231`, EURUSD, STRICT news-clean (`news_fill==0 & news_open==0`),
resolved (exit ∈ {tp,sl}). **RE-DERIVED 2026-08-11 on the structure-fix re-run (3,322→3,041): N=1254→
N=1162, book meanR −0.165→−0.174, WR 27.5%.** Every bucket below is a LOGGED near-miss — its CI
OVERLAPS the rest of the book (CHECK 1), so nothing ships; carry to Validation / a 2nd pair.

### 1. Flip-at-CHoCH is DEAD (KILL — not a near-miss, a dead end)
- Flipping the trend at the CHoCH is a pure RELABEL of the CHoCH trades: they currently land in
  `ambiguous` (the pending-flip demotion, `smc_detector.derive_trend_alignment` :2566-2580), NOT
  `counter_trend`. Relabelling folds the 364 ambiguous (meanR −0.159) into `with_trend` → moves
  with_trend −0.225 → −0.200 and leaves `counter_trend` (N=202, −0.050) UNTOUCHED. Money does not
  follow the label. Premise "CHoCH reversals hide in counter_trend" is FALSE. Do NOT build the flip.
  (2026-08-11: verdict UNCHANGED, arguably stronger — counter_trend is no longer even positive.)

### 2. Mid-range CHoCH is a TRADE problem, NOT a detection problem (trade-gate candidate)
- `reversed_from_extreme` = did price tag the top/bottom 25% of the **H4 dealing range**
  (dealing_range.py:1483) before the CHoCH. Flip-pending mid-range CHoCH (`ambiguous & reversed=False`)
  N=77 meanR −0.260 vs rest −0.168, EXCESS −0.092R ×77 = −7R. diffCI **[−0.379,+0.222] — straddles 0.**
- **2026-08-11 re-derive — SAME DIRECTION, magnitude is NOISE (tiny N).** Was −0.416 / −0.267 / −21R
  on the old run; the drop is ~4 trades flipping loss→win (15→19 winners of 77). At N=77 one winner =
  +0.039R on the mean, so the whole "−21R→−7R" swing is 4 trades. Effect was never stable; CI always
  straddled 0. Do NOT discount OR promote it — log as a thin near-miss, ship nothing, carry to Validation.
- FATE PROXY (same-dir Confirm before next CHoCH, from the alert stream): mid-range confirm **24.2%**
  vs from-extreme **25.4%** — IDENTICAL (OLD run; needs the alert stream, NOT re-derived 2026-08-11).
  Mid-range CHoCHs are NOT false flips; they become real
  reversals just as often. The gap is PAYOFF (extreme reversal has room to run; mid-range doesn't),
  not hit rate. → USE (if it validates): a TRADE-layer gate (don't trade a CHoCH OB that didn't reverse
  from the H4 extreme). Do NOT gate detection — you'd kill genuine reversals at the same rate and touch
  the frozen reclaim-disarm.

### 3. PD-alignment on CHoCH reversals — weak, SMC-coherent, UNCHANGED (near-miss)
- **2026-08-11 re-derive CORRECTS an earlier over-claim.** The only stable content is the direction-FOLDED
  read: `pd_alignment==aligned` (LONG&discount or SHORT&premium — the SMC-good side of a reversal) −0.146
  vs counter −0.192. Weak (+0.046/tr), CIs overlap, and **almost identical to the old run** (aligned
  −0.140 vs counter −0.191). This barely moved — the robust finding.
- **DROPPED as unreplicated noise:** the old "long-from-premium worst / short side shows NO PD effect"
  framing. Re-derived raw cells (within ambiguous, N=364): LONG premium −0.273 (EXC −0.105, diffCI
  [−0.415,+0.223]); LONG discount −0.040 ([−0.106,+0.411]); SHORT premium −0.244 ([−0.306,+0.167]);
  SHORT discount N=38 −0.053 ([−0.309,+0.584]). **EVERY cell's CI straddles 0** — the short side now
  shows a similar-sized gap too, so the long-specificity was thin-slicing, not a real direction split.
  (Short-discount N=38 = 12 winners; ±2 trades = ±0.16R. SMC cross-check: a short from premium reading
  "bad" is incoherent = the tell it was noise.) `pd_zone`/`pd_alignment` = H4-range read, H1-tracked.

### 4. Confirm × fill-speed sign-flip (near-miss; 6-bar threshold RETRACTED)
- Threshold-free Spearman(slower fill → R): BOS +0.018, CHoCH +0.087, Range +0.087 (patience helps a
  first break) vs Confirm −0.061 (patience hurts a confirmation break). Sign-flip real, magnitude tiny.
  (2026-08-11 re-derive; old 0808 values BOS +0.051/CHoCH +0.079/Range +0.039/Confirm −0.069.)
- Slow Confirm (`bars_break_to_pullback` > 13 = Confirm median, unchanged) N=83 meanR −0.313 vs rest
  −0.163, EXCESS −0.150R ×83 = −12R. **CIs OVERLAP.** Earlier t≤6 cut was outcome-mined → RETRACTED.
- USE (if it validates): TRADE-layer time-stop — cancel the Confirm limit if unfilled within ~13 bars.
  No detection change. Mechanism: first break → time = the level proving itself; confirmation break
  (already late) → time = the move running out of gas.
- THRESHOLD/AXIS (trader Q 2026-08-07): 13 bars is the Confirm MEDIAN, NOT a proven threshold — the
  finding is a near-miss, no cut is validated; do NOT hard-code 13. The sign-flip is axis-robust
  (Confirm rho: bars −0.069, hours −0.030, approach_speed-ATR −0.074; BOS/CHoCH all positive), so it
  is not a bars artefact. But raw bars is regime-blind (13 bars = a bigger move in high vol), so IF
  operationalized prefer an ATR-normalized lateness and re-test the cut. Bars-vs-ATR is UNTESTED.

### 5. "Late = bad" is FALSE book-wide (RULED OUT — do not re-test)
- Book-wide, SLOWER fills are BETTER: BOS+CHoCH slow (>8 bars) −0.112 vs fast −0.188. A blanket lateness
  filter backfires. Confirm (#4) is the lone exception, not the rule. (2026-08-11 re-derive; was −0.088/−0.190.)

### Rejected this branch (do not re-test)
- "Longer journey / distance to OB proximal → bigger loss": Spearman(distance → R) ≈ 0 on BOTH BOS and
  CHoCH across `impulse_leg_to_extreme_atr`, `ob_walkback_depth`, `reversal_pct`, `sl_dist_atr_at_alert`,
  `approach_speed_atr_at_fill` — mostly the WRONG sign (bigger impulse leg = mildly BETTER on BOS, +0.070).
- Direction-aware weekly PD: aligned −0.169 vs counter −0.161 = no edge.

### 6. D1-levels location is NOT sharper than H4 pd_alignment (near-miss log)
- QUESTION: does a direction-aware D1 (previous-day / weekly pool) location read separate winners from
  losers *sharper* than the H4-range `pd_alignment` — i.e. a bucket-vs-rest EXCESS whose CI SEPARATES
  where H4's did not? All features are **FILL-timed** (`*_at_fill`, pool distances/tiers, trade_toward_pool
  computed at the fill bar) — a live alert-time scorer cannot see them; label FILL. Built from
  `pdh/pdl_status_at_fill`, `dist_next_pool_above/below_atr`, `next_pool_above/below_tier`,
  `trade_toward_pool` (`pool_builder.trade_features` :458-528). Same pop: N=1254, book meanR −0.165.
- REFERENCE (H4): `pd_alignment==aligned` −0.140 vs rest −0.191, EXCESS +0.051/tr ×642 = +33R,
  diff boot95 **−0.096..+0.196 (overlaps 0)**. Bar to beat: a D1 read whose diff-CI EXCLUDES 0.
- **Every D1 cut's diff-CI still includes 0 — none separates. NOTHING is sharper. Verdict: NO.**
  - F1 `trade_toward_pool==True` (aimed at nearest untapped pool): N=501 WR 29.5%, meanR −0.114 vs
    rest −0.199, EXCESS **+0.085/tr ×501 = +43R**, diff boot95 −0.066..+0.237. Biggest CLEAN excess,
    larger than H4's — but CI overlaps. Best D1 near-miss; carry to Validation / 2nd pair.
  - F2a **D1 discount/premium (direct analog of pd_alignment)**: long lower-third / short upper-third of
    the intact-pool range (`range_pos=db/(da+db)`, both pools present N=953). N=357 meanR −0.202 vs
    rest −0.151, EXCESS **−0.051/tr (WRONG sign)**, diff −0.212..+0.112. The D1 discount read does the
    OPPOSITE of the (weak) H4 hint — buying the daily discount did NOT help here. Premium-fight F2b and
    middle F2c both ≈ flat (−0.020, −0.019). The apples-to-apples D1 analog is WORSE, not sharper.
  - F3 room-to-target-pool (dist to the pool the trade heads toward): NEAR ≤3.04ATR +0.050, MID −0.030,
    FAR >6.32ATR −0.045 — monotone-ish but every diff-CI spans 0. No sharp cut.
  - F4a target-side D1 extreme still intact (long&PDH intact / short&PDL intact): N=852 EXCESS +0.076/tr
    ×852 = +64R, diff −0.082..+0.230 — biggest total-R but widest overlap (it's most of the book).
  - F4b **reclaim-after-swept-extreme** (`status==broken` means yesterday's extreme was ALREADY taken
    out BEFORE this fill; going long once PDL is broken behind you / short once PDH is broken = the
    long-after-a-failed-downside-break setup, NOT a fresh breakdown short): N=107 WR 33.6%, meanR +0.009
    vs rest −0.181, EXCESS **+0.191/tr ×107 = +20R**, diff −0.082..+0.474. Biggest point-excess of the
    whole screen and the only ~breakeven bucket, but N=107 → CI huge. Second near-miss worth a Validation
    look; do NOT act on 107 trades of one pair.
- SMC cross-check: liquidity-DRAW (F1/F4b — aim at untapped liquidity / reclaim a swept D1 extreme)
  points the right way and beats the discount/premium-LOCATION read (F2), which was flat-to-wrong. If
  any D1 idea survives out-of-sample it is "trade toward untapped liquidity," not "buy the daily discount."
- **ALERT-TIME (live usability) — the D1-pool snapshot is NOT in this run.** `pdh/pdl_status`, pool
  distances/tiers and `trade_toward_pool` are logged `*_at_fill` ONLY; there is no `*_at_alert` twin, so
  F1/F4b CANNOT be scored live from the current CSV — needs a re-run that stamps the pool snapshot at
  `alert_ts`. Gap is material: alert→fill median 4h / mean 16h / p90 63h, only 37% fill within 1 bar.
  The alert-time daily/weekly location reads that DO exist are all flat: H4 `pd_alignment` +0.051/tr
  (reference, overlaps); weekly discount-aligned (`weekly_pd_position_at_alert`, long low/short high)
  N=483 +0.026/tr diff −0.130..+0.178; weekly zone-aligned (`weekly_pd_zone_at_alert`) N=646 −0.008/tr
  — no edge (matches the earlier "weekly PD = no edge" note under "Rejected this branch").

**Stage:** 4 — screening. Every bucket's CI overlaps the rest → LOG, carry to Validation / 2nd pair,
ship nothing. D1-levels sharpness test DONE (2026-08-07): no D1 read separates where H4 didn't; F1
trade_toward_pool (+43R) and F4b into-expansion (+20R) are the two logged near-misses. **Added:** 2026-08-07.
> **⚠️ NUMBERS IN ITEM 6 NOT RE-DERIVED on the 2026-08-11 structure-fix run (still the old 3,322/N=1254
> figures).** Deprioritised: this whole screen is FILL-timed (not live-usable) and its verdict is a NULL
> ("no D1 read sharper than H4"), which a −7% population trim does not flip. Re-derive only if item 6 is
> ever revisited for Validation; treat every number here as stale until then.

---

## OB PENETRATION FINDINGS (EURUSD Discovery, logged 2026-08-10)

Branch: "what does how-deep / how price interacts with the OB tell us about the trade." All numbers:
canonical `h1only_20080102_20161231`, EURUSD, STRICT news-clean (`news_fill==0 & news_open==0`),
resolved (exit ∈ {tp,sl}). Per-trade and
first-candle numbers come from a REAL H1-bar replay (fill_ts-anchored walk over
`backtest/cache/EURUSD_X_1h.parquet`, reproduces the CSV outcomes 100%; same method as the retired
`scratchpad/resim2.py`). Everything here is ENTRY-STUDY *understanding* on the fixed-2R ruler; the two
cut levers (#2, #3) are EXIT-engine candidates — live-observable but in-sample Discovery, one pair,
UNVALIDATED. Ship nothing.

> **RE-DERIVED 2026-08-11 on the structure-fix re-run (3,322 → 3,041 rows).** Population moved
> **N=1254 → N=1162** (320W / 842L, WR 27.8% → **27.5%**, book meanR −0.165 → **−0.174**). Items 2
> and 7 were fully re-run on the fresh CSV (numbers inline below, old → new) — **both verdicts hold,
> the signal barely moved.** Item 3 (depth-while-alive) is **NOT re-derived this session** — treat its
> figures as the old 3,322-run numbers until revisited (the monotone WR gradient and the D=0.4/T=2
> lever direction are structural, unlikely to flip on a −8.5% noise trim, but the exact cut-EV is stale).
> Re-derive scripts this session: `scratchpad/rebuild.py`, `stats2.py`, `rederive.py`.
>
> **⚠️ TP-on-fill-bar convention confirmed in code (2026-08-11):** the simulator NEVER credits a TP on
> the entry candle — `if is_fill_bar_this_iter: tp_hit_in_bar = False` (h1_only_simulator.py:818-822;
> exit_engine.py:233-235); SL *can* fire on the fill bar, TP cannot (intrabar wick order unknowable).
> Proof: **zero wins have `bars_to_exit==0`.** So "instant death" = SL on the entry candle = `bars_to_exit==0`
> = **256 trades = 22.0% of resolved / 30.4% of losses** (NOT the ~50% figure — that is the broader
> `bars_to_exit<=1` "died within one bar" = 57.6% of losses; different denominator + definition). The
> **actionable** population (survived the entry candle, can be read at bar-1 close) = **N=906, WR 35.3%,
> book meanR +0.060**. A fill-bar TP *touch* is NOT a win and must not be excluded as one.

### 1. `ob_penetration_depth` floors to 0 — read it with a companion, NEVER alone (column caveat)
- Definition: deepest adverse poke into the OB as a fraction of OB depth, from `mae_intrade_price`
  (`h1_only_simulator.py:902-924`). MAE tracking EXCLUDES the fill bar and the stop bar (`:787`), so any
  trade that fills-and-stops within ~1–2 bars never has its poke recorded → the column reads **0**.
- Result: **55.2% (642/1162) read exactly 0** (2026-08-11 re-derive; was 54.5% / 683/1254), and that 0
  conflates TWO OPPOSITE populations — clean runners that never dipped (~high WR) and instant-deaths
  that blew straight through the zone (0% WR).
- NOT "broken" — it faithfully records the deepest poke of the *tracked* bars. But it is USELESS read
  alone. Bifurcate the 0-bucket with a companion column: `bars_to_exit<=1` (or died on the stop bar) =
  instant death; else = clean runner. Or measure the real poke from raw bars.
- Its **median is literally 0**, so a median-split of this column is degenerate and INVERTS the true
  relationship (among trades that DID penetrate, deeper = monotonically worse). Domain thresholds /
  continuous only — never a median or quantile cut (Standing Guard).
- FIX options: (a) analysis-side, always pair it with `bars_to_exit`/`mfe_intrade_r` — no code; (b)
  code-side, add a `first_candle_penetration` column computed from the fill+stop bars (captures exactly
  what this column blanks out) — detection-layer change, rides next baseline, needs approval.

### 2. First-candle CLOSE (rejection vs acceptance) — the sharpest single read
- From real bars: does the FILL bar close back on our side (rejection) or inside/through the zone
  (acceptance)? **REJECTION** (closed back above entry, long) **[2026-08-11: N=533 WR 44.3% [40.1,48.5]
  meanR +0.328]** — was N=584 WR 45.2% +0.356. **ACCEPTANCE** (closed adverse) **[2026-08-11: N=629
  WR 13.4% [10.9,16.2] meanR −0.599]** — was N=670 WR 12.7% −0.619. Point-biserial corr(close-depth,
  win) = **−0.396** (was −0.41; LARGE by Cohen, ~16% of win/loss variance from one variable).
  First-candle WICK depth is the same signal weaker (continuous corr −0.217, monotonic shallow→deep:
  <0.5 OB 38.8% → 0.5–1 32.4% → >1 OB 14.3% WR). **Verdict unchanged — the signal barely moved
  through the structure re-run.**
- Acceptance includes the instant-deaths (256, already stopped, unsavable); among SURVIVORS acceptance
  is **20.9% WR [17.2,25.2]** (was 19.8%) — still far below rejection's 44%. Mechanism (SMC): the first
  candle is the market's first test of the zone — a wick that closes back out = zone holding; a close
  accepted inside = zone failing.
- LIVE STATUS: knowable ONE BAR AFTER the fill (you are already in) → an early-EXIT read, NOT an entry
  filter. Whole-book impact of acting on it: **realistic** (cut the acceptance survivors at the fill-bar
  close; instant-deaths stay −1R) book **[2026-08-11: −0.174 → −0.140 (+0.034R/tr, +39R)]** — was
  −0.165 → −0.122 (+0.043R, +54R). **Idealised ceiling** (never took any acceptance trade at all) =
  **+0.328R / 44.3% WR** — look-ahead, not a filter; shown only to size the effect.
- **STRONGER KEEP-SIGNAL — close-vs-OPEN beats close-vs-ENTRY (NEW 2026-08-11).** Keep only trades whose
  entry candle closed **above its own open** (candle direction, `close>open`), not merely back above
  entry: kept N=140, **WR 60.0% [51.7,67.7], held meanR +0.800**. Whole-book cut (keep those 140, exit
  every other survivor at the fill-bar close, instant-deaths −1R): book **−0.174 → −0.088 (+0.086R/tr,
  +100R)** — ~2.5× the acceptance-cut lever above, and POSITIVE in **all 9 Discovery years** (per-year
  ruleR min +0.003 in 2012, never negative). On the actionable-only book (N=906) it is +0.060 → +0.170
  [+0.121,+0.220], CI excludes 0. **COST / caveat:** this holds only ~12% of the resolved book (~15% of
  survivors) to target and flattens the other 85% near breakeven (cut group sits at +0.055R at the bar
  close but bleeds to −0.076R if held) — it is a strategy REDEFINITION ("only stay if the entry candle
  confirms"), not a small tweak, and is a bar-1-close management read, never an alert-time entry filter.
  In-sample, one pair, UNVALIDATED. Scripts: `scratchpad/stats2.py`, `rederive.py`.
- **Wick/body RATIO adds NOTHING beyond close-direction — no "smart threshold" exists (2026-08-11, NEGATIVE
  result, do not re-run).** Trader asked for a rejection-wick/body ratio threshold to manage a live trade after
  bar 1 closes. Population = alive N=979 (ratio undefined when body≈0, dropped). Continuous point-biserial
  win~ratio = **r=+0.055, p=0.083** (weak, could be luck). A raw threshold sweep LOOKS strong (ratio≥1.0 → 45.2%
  vs <1.0 → 31.1%) but that is pure confounding: a big rejection wick almost always means the bar CLOSED as a
  rejection. Held WITHIN close strata the ratio is FLAT — REJECTION rows: ratio≥1.0 WR 46.9% [41,53] vs <1.0
  48.5% [43,55] (CIs fully overlap); ACCEPTANCE rows: high-ratio N=30 too thin to read. **Conclusion: use the
  bar-1 CLOSE direction (this item), not the wick shape — the ratio is a weaker, noisier restatement of the same
  thing.** Script: scratchpad `h2_thresh.py`.

### 3. Depth-while-alive is the driver — "slow grind" was the WRONG label (CORRECTION)
> **⚠️ NOT re-derived on the 2026-08-11 structure-fix run — figures below are still the old 3,322/N=1254
> numbers.** Direction (deeper-while-alive = worse; D=0.4/T=2 sweet spot) is structural and unlikely to
> flip on a −8.5% noise trim, but the exact cut-EV is stale until re-walked. Re-derive if revisited.
- The deeper a trade pierces WHILE STILL ALIVE, the lower the WR — monotonic: reach 0.2=22.2%,
  0.3=19.6%, 0.4=15.9%, 0.5=12.6%, 0.6=9.8% WR.
- CORRECTION to an earlier read: the lever is DEPTH-while-alive, NOT speed. Splitting deep trades by
  slow-vs-fast (bars to reach the depth) is weak (at D=0.5: slow 11.1% vs fast 12.9%). An earlier
  "slow grind = bad" framing named the wrong driver — this is the depth version that replaces it.
- CUT-EARLIER EV (cut at market when alive, bar≥T, cumulative depth ≥D; books that bar's REAL close,
  no look-ahead): **D=0.4 / T=2 → −0.165 → −0.122 (+0.043R)**, beats D=0.5 (+0.037) — cut earlier AND
  better; D=0.3/T=2 = +0.028 (sacrifices more winners). **0.4 is the sweet spot.** Same magnitude as
  the #2 acceptance cut (they overlap — acceptance ≈ will-go-deep).
- CAVEATS: EXIT-engine lever on the fixed-2R ruler; in-sample Discovery, single pair. It makes a losing
  book LESS bad (−0.12, still negative), does NOT make it positive; belongs to the Stage-1 exit
  re-decide. Carry to Validation / 2nd pair; ship nothing.

### 4. Deeper entry for better RR — NOT well tested (whole-book only; needs slicing)
- Tested ONLY on the ENTIRE book: no signal there makes a deeper limit profitable — all losers fill
  regardless of entry depth (loser side can't improve), and the winner math `(2+p)/(1−p)` doesn't clear
  the bar. R stays $250 fixed (constant-risk sizing); that does not change it.
- OPEN — never sliced: the real question is whether SPECIFIC slices (by an alert-time signal) would
  benefit from a deeper limit. Needs a sliced re-test before any verdict. Do NOT conclude "deeper entry
  is dead" — only "dead on the whole book, UNTESTED on slices."

### 5. GH1 "deep and going nowhere" — outcome-based loser fingerprint (DATA ONLY, cannot filter)
- Trades that pierced deep (`ob_penetration_depth`≥0.5) AND never showed any profit
  (`mfe_intrade_r`≤0.1): **N=46, 100% losers, 0 winners, Wilson [92.3, 100]** (2026-08-11 re-derive;
  was N=49). A perfect loser fingerprint in hindsight (holds at ≥0.6 too; N stays 46 because the
  floored column caps the count).
- BOTH inputs are look-ahead/outcome (penetration = deepest post-hoc; mfe = in-trade peak). It LABELS
  losers after the fact — it CANNOT be scored at entry, and there is nothing to "cut" (they are already
  gone). Logged as BEHAVIOUR — how this pair's losers look — never as a filter.

### 6. GH2 "reclaim or die" — reclaim is a WEAK positive tell, not live-usable
> **⚠️ NOT re-derived on the 2026-08-11 structure-fix run — figures below are the old 3,322/N=1254 (book
> 27.8%) numbers.** Reclaim-by-bar-k needs the fill-anchored bar replay (same harness as #2/#3), not yet
> re-run this session. Direction (reclaim = mild positive, still EV-negative to cut) is structural; exact
> per-bar numbers stale until re-walked.
- The tautology is dropped (a trade that never closes back above entry loses ~always — circular,
  worthless). The real content: among trades that dipped then CLOSED BACK above entry (reclaimed),
  WR = 33–42% (reclaim by bar 1–2: 32.7% / −0.019; bar 3–5: 39.5% / +0.184; bar 6+: 42.5% / +0.275)
  vs book 27.8% and ~0% for never-reclaim (N=515, 0.2%). Reclaiming ~1.3–1.5× the odds — a MILD
  positive sign.
- BUT 58–67% of reclaimers still FAIL (reclaim, then roll back over to the stop). Reclaim = better
  odds, not safety; on its own too weak to act on. The reclaim-then-fail cohort is NOT yet understood
  (open sub-question).
- NOT live-cuttable: "never reclaimed" is END-of-trade knowledge; "not reclaimed YET by bar k" still
  holds many future winners, and the 2:1 stop makes cutting EV-NEGATIVE at every k (k=1..8 all worse;
  net-if-held stays positive because winners cut ×2R outweigh losers saved ×1R until ~k=10 where it is
  noise). Dead as a live cut; logged as behaviour + the geometry reason it cannot be one.

**Stage:** ENTRY-study understanding + two EXIT-engine cut candidates (#2 first-candle acceptance, #3
depth-while-alive @0.4) on the fixed-2R ruler. Both are live-observable but in-sample EURUSD Discovery,
one pair, UNVALIDATED → LOG, carry to Validation / 2nd pair, ship nothing. Column caveat (#1) governs
ALL future penetration analysis. Deeper-entry (#4) untested on slices; GH1/GH2 (#5/#6) are behaviour
only. **Added:** 2026-08-10.

### 7. Bar-2 CONTINUATION — a late-entry / trade-management tell (LOG for management, NOT an entry filter)
- **Idea:** after the fill candle (bar 1), does the NEXT candle (bar 2) closing FURTHER in our favour than
  bar-1's close lift the odds? A veteran adds/holds when the second candle confirms the rejection.
- **Data:** EURUSD Discovery, STRICT news-clean, resolved. Population = ALIVE (survived the fill candle),
  **[2026-08-11: N=906, baseWR 35.3%]** — was N=983, baseWR 35.5% (bar 2 is only observable if the trade
  didn't instant-die). Wilson 95% CIs.
- **Result [RE-DERIVED 2026-08-11, old → new]:** bar-2 continues → **55.5% WR (N=434) [50.8,60.1]** (was
  54.9%, N=472); bar-2 does NOT continue → **16.7% (N=472) [13.6,20.4]** (was 17.6%, N=511). A **+38.8pp**
  spread. Stacked with the bar-1 read: rejection+continue **65.7% (N=251)** (unchanged); closed-against+continue
  **41.5% (N=183)**; rejection+no-continue **28.0% (N=254)**; **closed-against + no-continue = DEAD 3.7%
  (N=218) [1.9,7.1]** — the clean "give up" cell. **Verdict unchanged.**
- **INDEPENDENCE (answers the trader's H1 question):** bar-1 does NOT predict WHETHER bar 2 continues —
  P(continue | rejection)=**0.497** vs P(continue | against)=**0.456**, corr(rejection,continue)=**0.040**
  (was 0.042). So bar-2 continuation is its own signal (~55% on its own), and it stacks ON TOP of the
  bar-1 close, not instead of it.
- **Timing class: FILL + 2 BARS** (~2h after fill). NOT an alert-time entry filter — a management/late-entry
  read: hold/add when bar 2 confirms, stand down on the dead cell. In-sample, one pair, UNVALIDATED.
  Script: scratchpad `h1_verify.py`.

### 8. OB THICKNESS (`ob_range_atr`) — NEAR-MISS, NOT a filter (retracts an earlier over-claim)
- **Idea (alert-time):** thicker order block = wider zone = worse trade? `ob_range_atr = (OB high−OB low)/ATR`,
  stamped at OB formation — genuinely ALERT-time (live-observable). [h1_only_simulator.py ~L1450-1453].
- **Data:** EURUSD Discovery, STRICT news-clean, resolved, ALERT-time population = instant-deaths INCLUDED
  (a live filter fires before the outcome is known), **N=1162, baseWR 27.5%** (2026-08-11 re-derive; was
  N=1254, 27.8%). Continuous + absolute ATR edges (NOT median/tertile — per the no-arbitrary-split rule).
- **Result — it does NOT hold:** continuous point-biserial win~thickness **r=−0.035, p=0.227** (nothing).
  Absolute edges are NON-monotonic: <0.6 tight 28.6%, 0.6–0.9 **31.1%**, 0.9–1.2 **22.3%**, ≥1.2 wide 26.8% —
  a zig-zag, no trend. CHECK 1 on a domain thick cut (≥1.2 ATR vs rest): meanR −0.195 vs −0.168,
  diffCI(meanR) **[−0.216,+0.162]** — straddles zero. (Old run: r=−0.035 p=0.212; edges 27.8/32.0/22.4/27.2.)
- **Why it's not real — CONFOUND:** thickness ≈ the stop distance. **corr(ob_range_atr, sl_dist_atr_at_alert)
  = 0.888** (was 0.886), and stop-width itself is also dead (r=−0.048, p=0.104). There is no independent
  "thickness" signal separate from stop-width (see the stop-widen section above).
- **RETRACTION:** an earlier pass called "thick OB = worse" an alert-time keeper — that was measured only on the
  post-fill ALIVE subset (look-ahead selection: thin/tight-stop OBs instant-die MORE, thick OBs survive-then-lose,
  and the two cancel on the full population). On the population a live filter actually sees, it washes out. Logged
  as a NEAR-MISS: carry to Validation / a 2nd pair, ship nothing. Scripts: scratchpad `thickob_clean.py`.
- **Added (items 7–8):** 2026-08-11.
