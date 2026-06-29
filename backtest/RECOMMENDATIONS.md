# Recommendations from the Back Desk

Durable lessons + methods from backtest audits. Read before judging any run.

---

## METHOD — Is the edge real, or luck? (Bootstrap 95% CI)

**Use this every time you want to trust a system's expectancy.**

### What it answers
"Is this system's average result really positive/negative, or did I just get a lucky/unlucky run of trades?"

### How to run it (plain steps)
1. Take the list of per-trade results in R (the `r_realised` of every eligible trade).
2. Draw a random sample of the SAME size, *with replacement* (some trades picked twice, some skipped).
3. Average that sample's R. Write it down.
4. Repeat 10,000 times → 10,000 averages.
5. The middle 95% of those averages = the **95% confidence interval (CI)**.

### How to read it
- **CI entirely above 0** → edge is real and positive. Trust it.
- **CI entirely below 0** → it's a real loser, not bad luck. Stop / rebuild.
- **CI crosses 0** → UNPROVEN. Could be zero. Do NOT trust either way — need more data.

### When to use
- Before calling any window/config/filter "good" or "bad."
- On any sub-group (per pair, per filter) BEFORE making it a rule. A split that looks
  great pooled but whose CI crosses zero is NOT a reliable signal (see PD-alignment, 2026-06).

### Reliability caveats
- Assumes trades are roughly independent and the future resembles the past.
- N matters: ~300+ trades gives a usable CI; <50 is almost always inconclusive.
- The CI is only as good as the input data. Garbage wicks → garbage CI.
- Also check **per-quarter sign consistency**: a signal that flips sign across quarters is
  noise even if the pooled CI looks ok.

---

## METHOD — Correlation & prediction toolkit (which stat answers which question)

Pick the tool by the question. All three need **one number per trade** for each side; a group
stat (a rate, a total) has no per-trade number and cannot be correlated.

- **Spearman (rank correlation):** "When X goes up, does the outcome go up too — in rank order?"
  Uses ranks, so one freak trade can't distort it. Returns −1..+1; **0 = no link**. Use for
  *feature vs continuous R* (score→R, fvg_size→R). Direction is free — testing "bigger X" also
  answers "smaller X", so never run both. **Cannot take a win RATE** (group stat).
- **Pearson (straight-line correlation):** same idea on the *raw values*, looking for a straight
  line. Outlier-sensitive (one +8R yanks it). For trade scoring we only trust order, so prefer
  Spearman. (Low priority here.)
- **Regression:** draws the best line/curve and gives the **equation to PREDICT** Y from X —
  and can take **many** features at once. Correlation says *if/how much*; regression *builds the
  predictor*. **Multiple regression** = several features → expected R. **Logistic regression** =
  features → win-probability (yes/no outcome). **The EV-score will be a regression.**
- **For feature vs win/lose (binary):** Spearman is the wrong tool — bucket the feature and
  compare win rates, or use logistic regression.

---

## CORE INSIGHT — A score concentrates an edge, it cannot create one

> A score can't create an edge — it can only concentrate one that already exists in the
> trades. An additive confluence score has zero predictive power (Spearman ~0.05) because
> it answers the wrong question: "how many confluences are stacked?" instead of
> "what is this setup's expected R, given its context?"

**Implication:** if the whole population of trades is a loser and no sub-group is *robustly*
positive (CI above zero, consistent across quarters), then no scoring threshold turns the
system positive — you can only lose less by trading less. Fix the edge (entry / regime /
exit), not the points. The score's only job is to estimate expected-R per setup and let you
trade/size the top slice — validated out-of-sample, net of spread.

**ACTION when scoring is rebuilt (decided 2026-06-24, user):** **drop the sweep detector from
the score** as part of the scoring rework. It currently inverts SMC (sweep_present = True →
16.7% WR, −0.440R) and is hard to manage. No change now — the removal happens inside the
scoring rebuild so live/backtest parity moves in one step. Confirm exactly where sweep points
enter `run_scorecard` (smc_detector.py) before pulling them, so the email/chart visual (if any)
is kept and only the scoring contribution is removed. See the sweep bullet under the
2026-06-23 audit for the detector-redesign option if it is ever wanted back.

---

## SCORING — Structure / BOS continuation (2026-06-24)

**The structure quality score is driven by event TIER + a continuation DRIVE
verdict. It is NOT driven by the raw BOS count.** Counting was the old design and
the wrong question: a trend does not age at the Nth break — it ages when
**displacement fades**. A clean wall break or a fresh confirmation re-commits
drive and should not be penalised for being "the 4th break."

**How it works now** (`smc_detector.compute_bos_sequence_count` → `run_scorecard`):

- **Clock resets** (new continuation leg starts) on: CHoCH, Confirmation BOS,
  Range BOS. Only a PLAIN internal BOS ages the leg.
- **Score by tier + verdict:**
  - CHoCH from zone / Range BOS / Confirmation BOS → 4 (strong).
  - CHoCH mid-range → 3.
  - plain BOS, drive **holding** → 3.
  - plain BOS, drive **fading** → 1 (the genuine "late continuation").
- **Verdict = displacement decay.** With ≥ 3 plain BOS in the leg, `fading` when
  `mean(last-2 break bodies) < 0.60 × mean(first-2 break bodies)`, where the body
  is the break-candle body in ATR (`break_body_atr`, the SAME window-aware measure
  the body gate fires on and `compute_break_quality` grades — one definition,
  three readers). < 3 breaks → `holding` (never claim exhaustion we can't measure).
- **`bos_sequence_count` ("#N") is LABEL CONTEXT only** — it appears in the email
  Structure row for the trader's eye; it has NO score weight on its own.

**Why 0.60, anchored on the EARLY mean:** live data showed single-bar / single-
strongest anchors are noise-jumpy (a mid-leg spike or one soft candle flips them).
Early-mean asks the vet's real question — "is recent drive weaker than how the leg
started?" 0.60 = recent push under ~two-thirds of the opening push. One number, no
per-pair tuning (momentum decay reads the same on EURUSD as on Gold).

**STILL OPEN (validate before trusting the magnitude):** the 0.60 ratio and the
1-vs-3 score split are SMC-reasoned, not CI-tested. `bos_sequence_count` is already
in the trade CSV; the next step is to tabulate WR/expR by verdict (holding vs
fading) and bootstrap-CI it — then decide whether the split should be wider/narrower.
This SUPERSEDES the old "exhaustion penalty may be inverted" note below: the penalty
is no longer count-based, so that proxy finding no longer maps cleanly — re-test on
the verdict, not on the raw count.

---

## RULE — Win rate is meaningless without R:R

Breakeven win rate = 1 / (1 + avg_win_R). With avg win 1.69R, breakeven WR = 37%.
A 37% WR is NOT "worse than a coin toss" — it is exactly breakeven for a system whose
winners are 1.69x its losers. To raise WR you must take profit sooner (smaller avg win,
higher WR) — it is a trade-off, not a free lunch. Judge expectancy (R), never WR alone.

The formula: over many trades, money from wins = `WR x avg_win`, money from losses =
`(1-WR) x avg_loss`. Breakeven is where they cancel: `WR x 1.69 = (1-WR) x 1`
→ `1.69 WR + WR = 1` → `WR = 1/2.69 = 37%`.

---

## PROVEN FINDINGS (audit 2026-06, H1 system, proximal = the model live trades)

All measured on 1 year of yfinance data (4 quarters, 340 proximal trades). Re-confirm on
broker/MT5 data before betting on them.

- **The system loses:** -53.5R, expectancy -0.157R, WR 30%, 95% CI [-0.28, -0.03] (real
  loss, not luck). Positive in only 1 of 4 quarters.
- **The confluence score is noise** (Spearman 0.05; every score bucket negative). See CORE
  INSIGHT above.
- **Counter-PD is robustly bad** (CI excludes 0). "PD-aligned = good" is NOT robust (diff CI
  crosses 0). Rule: **avoid counter-PD; do not reward "aligned" as a positive.**
- **The counter-PD damage is concentrated in CHoCH** (audit 2026-06-23, 4 quarters, proximal):
  counter-PD CHoCH = N=38, -0.433R, WR 19%, CI **[-0.715, -0.118]** (robust loser, but THIN N —
  confirm per-quarter + on MT5 before hard-gating). CHoCH-aligned = -0.069R, CI crosses 0 (near
  breakeven). Both BOS PD-splits cross 0. The CHoCH/BOS *geometry* is textbook-correct
  (`dealing_range.py:compute_structure` — break the last HL/LH by 1.0 ATR + 1.5 ATR body + gap
  guard + failure window). The bleed is a **missing PD-location gate, not a detection bug** —
  UNLESS the dealing range is mis-drawn on those 38 trades. OPEN (E2e): replay the DR on them to
  decide filter-vs-calc.
- **Losses are given-back winners:** 74-79% of full -1R losers were up >= +0.5R first
  (median +0.92R); winners dip only -0.40R. Fix = exit management, not entry selection.
- **Earlier break-even helps:** BE at +0.5R recovers ~14R/yr vs +1R (-53.5 -> -39.4).
  BE 0.5 > 0.7 > 1.0. Independent of the weekend rule.
- **NAS100 is the single biggest drag:** ~60% of total loss, negative every quarter.
  Dropping it = +26R.
- **Weekend-flat at Fri 18:00 UTC (IST midnight) costs ~0R** (vs -3R at 20:00 FX close).
  Free risk reduction at the user's session end.
- **Data risk:** yfinance FX `=X` wicks are unvalidated; futures (NQ/GC) cleaner. Validate
  against MT5 before trusting SL/TP precision.
- **Swing detection (lookback=3 fractal + ATR leg filter) is deterministic but NOT validated
  for correctness** vs a vet's marking. Untested foundation under BOS/CHoCH/OB.

---

## EXPERIMENTS TO RUN — TP / BE / exits

Trade-distribution facts driving these (filled trades): reach +0.5R 88%, +1R 69%, +1.5R 44%,
+2R 26%, +3R 10%; median MFE +1.37R; of trades reaching +1R, median peak +1.72R.

| Experiment | Why (from data) | Aim | How to run |
|---|---|---|---|
| Partial TP: 50% at +1R, BE the rest | 69% touch +1R but only 30% win — profit given back | Bank the meat on the 69%, keep upside on runners | Build multi-leg sim; close half at +1R, move stop to entry, run remainder to liquidity TP |
| BE trigger sweep | BE 0.5 already recovers +14R; find the floor | Max scratch-vs-cut without choking winners | Re-sim `be_r` over {0.3, 0.5, 0.7, 1.0} |
| Fixed-R TP (+1.5R / +2R) vs liquidity TP | only 44% reach +1.5R, 26% reach +2R — liquidity TP may be too greedy | Find the TP that maximises E[R] | Re-sim with fixed TP levels vs current target |
| Trail stop after +1R | +1R-touchers peak at median +1.72R; 37% reach +2R | Capture runners without giving it back | Multi-leg sim: once +1R hit, trail by ATR or last swing |
| Drop NAS100 | -0.45R, negative every quarter, ~60% of loss | Remove the structural bleeder | Filter the pair list; re-measure with the bootstrap CI |

### REQUIREMENT — TP1 placement, not just TP1 distance (user, 2026-06-24)

The "liquidity TP is too greedy" finding may be a **placement** problem, not only a
**distance** problem. The current TP1 is parked at the **edge of the opposing swing** — but
by SMC, price does **not** reverse from the raw swing edge. It reverses from a **structure /
order block** sitting before that edge. Parking TP at the edge asks price to travel to a level
it rarely turns from, so the runner gives back gains it could have banked one zone earlier.

- **Requirement:** TP1 should target the **reversal zone** (the opposing OB / structure on the
  path), **not** the swing edge. Track + build later — this is downstream of the exit-sim work.
- **Status:** NOT YET MEASURED. No conclusive data that a nearer TP1 wins; the SMC logic
  ("price turns at the OB, never the edge") is the driver. Verify where TP1 is currently placed
  (`compute_phase2_levels`, smc_detector.py) when this is picked up.
- Keep separate from the BE / partial experiments above — those change *distance*; this changes
  *where the target sits*.

### FINDING — reach-before-stop distribution (computed 2026-06-25, 4 quarters, proximal)

The decision-relevant number is **what fraction touch +X R BEFORE the −1R stop** (not raw MFE
at any time). `mfe_r` in the trade rows is the realised-path max (the walk breaks at the stop),
so for losers it IS the pre-stop max → `mfe_r >= X` answers it exactly for levels at/below the
typical liquidity TP1. (Above ~TP1 the figure is a FLOOR: winners bank at TP1 so higher levels
are undercounted.)

| Level | reach before −1R (all 6) | reach before −1R (ex-NAS) |
|---|---|---|
| +0.5R | 87.5% | 85.2% |
| +0.7R | 77.3% | 74.5% |
| +1.0R | 68.9% | 64.2% |
| +1.5R | 43.9% | 38.7% (floor) |
| +2.0R | 26.2% | 21.4% (floor) |

**Naive fixed-TP expectancy** (if it reaches +X it wins +X, else −1R; pre-spread, pre-CI,
ignores BE/timeout so the loss leg is slightly kinder than −1R), **ex-NAS**:

- TP +0.5R → ~**+0.28R**  | TP +1.0R → ~**+0.28R**  | TP +1.5R → ~**−0.03R**  | TP +2.0R → ~**−0.36R**

**Reading:** a fixed TP in the **+0.5R..+1.0R** band looks positive; **≥+1.5R loses** — direct,
quantified support for "the liquidity target is too greedy" and for the user's "win by volume"
instinct. **This is an ESTIMATE, not a verdict** — it is the most data-fragile config (tight
targets are the most wick-sensitive, spread eats a bigger share of a small target). Confirm with
the actual multi-leg sim + bootstrap CI + per-quarter sign + MT5 data. Drives experiment group C.

### PENDING — wire the winning exit recipe as the live policy (days away)

Once the exit lab picks a recipe (passes bootstrap CI + per-quarter sign on MT5 data, net of
spread), it becomes the new exit policy — this is the only step that touches live and the
parity path, and it needs explicit sign-off. To do it:
- Replace the single-exit walk in `backtest/h1_only_simulator.py` with the chosen recipe
  (it already lives as `exit_engine.walk_multileg`; flip `EXIT_MODE` from the experiment
  side-channel to the real path). One implementation — no second copy.
- Mirror the same exit in the live alert path (`compute_phase2_levels` / Phase2_Alert_Engine)
  so backtest = live stays true.
- Re-baseline the structure-golden / parity gates afterwards.
- **Status: NOT STARTED** — exits are still being measured; expect a few days before a winner
  is confirmed. Tracked here so it is not forgotten.

### TOOLING — exit lab (built 2026-06-25/26)

- `backtest/exit_engine.py` = the one multi-leg exit walker; `backtest/diagnostics/exit_lab.py`
  runs it over a fresh backtest via a side-channel in the simulator (OFF by default, never
  touches `r_realised` or live). Recipes: baseline, BE-sweep, fixed-TP {0.5/1/1.5/2}, partial+runner.
- **Do not replay exits over committed trades** — those were generated on yfinance; the cache is
  now MT5, so a reconstruction is unfaithful. Always run fresh so entries + exits share one feed.

---

## AUDIT 2026-06-23 — break gates, trend, NAS mechanism, BOS sequence

Same dataset as the 2026-06 audit: 1 year yfinance, 4 quarters, **340 proximal eligible
trades**. Re-confirm on MT5 before betting. Every claim below shows its data.

### PROVEN (verified facts — mechanical/structural, do not need a CI)

- **`trend_alignment` carries no information — it is ~93% one value by construction.**
  - Data: 318 of 340 proximal trades are "with_trend"; only 22 are "against_trend".
    By event: BOS 213/225 with_trend, CHoCH 100/110 with_trend.
  - Why: an OB's direction = the break's direction, and trend is read from the same swing
    structure that just broke — so OBs are with-trend *by construction*. Even CHoCH OBs read
    with-trend because the trend label flips with the CHoCH.
  - Trend is computed FRESH each bar (`replay_engine.py` calls `compute_pair_walls` per H1
    slice) — this is NOT stale walls.
  - **Rule:** drop `trend_alignment` as a scoring input. A feature that is 93% one value
    cannot rank trades. It is not a bug; it is structural.
  - **Update (2026-06-26, user):** the trend label now **flips only on the next BOS**, not the
    instant a CHoCH is detected (the old behaviour that made the label degenerate). Still drop
    it as a scoring input — but the flip timing is now correct for the structure engine.

- **NAS100's loss is a TIMEFRAME mismatch, not a filterable edge problem.**
  - Data: NAS100 proximal = 72 trades, 16.4% WR, expR -0.446, **-$8,022 (~60% of the year's
    -$13,385 loss)**. **47% of NAS losers (24/51) exit on the FILL bar** (bars_to_exit=0) vs
    26% (39/148) for every other pair. NAS losers avg MFE +2.32R AND avg MAE -1.78R — huge
    two-way swings inside single candles.
  - Mechanism: a single NAS H1 candle routinely spans entry -> +2R -> -1R. The OB-based stop
    distance is SMALLER than a typical NAS H1 candle's range, so one candle engulfs the whole
    trade; the engine books the loss and break-even never arms. No H1 filter fixes this.
  - **Rule:** NAS100 needs a finer entry timeframe (M5, the original methodology) or removal
    from the H1 book. Confirms the existing "drop NAS = +26R" finding with its root cause.

### TESTED BUT NOT PROVEN (guardrails — do not re-chase without new data)

- **Break displacement (body-ATR) gate is NOT a proven edge.** It LOOKS like the lever but
  fails the bootstrap bar.
  - Data (pooled looks good): BOS body>=1.5 = 113 trades, expR +0.004; body>=2.0 = 62 trades,
    +0.066; the marginal band (body<1.5) is the worst pooled bucket (-$11,484).
  - Data (fails verification): bootstrap 95% CI crosses zero for EVERY sub-group —
    body>=1.5 [-0.218,+0.233]; body>=2.0 [-0.247,+0.390]; body 1.5-2.5 [-0.172,+0.351];
    body>=1.5 no-NAS [-0.125,+0.374]. Per-quarter signs flip (body>=2.0: Q3 +0.26, Q4 -0.04,
    Q1 +0.25, Q2 -0.34).
  - Why not concluded: the positive expectancy sits INSIDE the noise band; once sliced to a
    band, N is too small for a tight CI; and the data is censored — every fired trade already
    passed the current gates, so we can only test TIGHTENING, never loosening.
  - **Loop closed:** do NOT treat a break-body threshold as a rule. To settle it: (a) more
    data (MT5, multi-year) for a tighter CI, then re-run the bootstrap; (b) if the CI clears
    zero AND per-quarter signs hold, it becomes real. Until then it is a candidate, not a rule.
  - **PLAN to conclude (2026-06-26, user):** BOTH displacement measures are already logged per
    trade — `break_close_atr` (distance the close cleared the level) AND `break_body_atr` (body
    size); `break_excess` is the ratio. Do NOT run this locally / on one year. Run the whole
    book over the **full ~18 years of MT5 history**, then bucket by `break_close_atr` and by
    `break_body_atr` and bootstrap each — with that much data the CI is tight enough to decide
    **which kind of break (distance vs body, and at what level) is actually best**. This is the
    way the break-quality question gets settled, not a 1-year in-sample slice.

- **The two break gates do different jobs; "body is the lever" is unproven; removing the
  distance gate is untestable here.**
  - Data: body and close-clear measure different things — body = candle size, close-clear =
    how far past the level the close finished. `body >= close_clear` always (gap guard,
    515/515 fired setups). The 0.4 distance gate is near-binding for 92/515 (18%) of fired
    setups (close-clear in [0.4,0.6)). Marginal test: holding body at its floor and raising
    the distance gate does NOT improve expR (0.4 -0.125, 0.8 -0.187, 1.2 -0.128, 1.6 -0.102).
  - Why not concluded: removing the distance gate would add setups with close-clear < 0.4 —
    those were rejected and have NO outcome rows, so the effect is unmeasurable from this data.
    And the "body is the lever" half rests on the break-displacement result above, which failed
    the CI.
  - **Loop closed:** keep the distance gate as a cheap guard against non-breaks; do not tune
    it (raising it adds no edge). To test removal: one re-run with `BOS_ATR_MULT = 0`. Design
    note: IF break-body is ever scored, score raw `break_body_atr`, NOT the ratio
    `break_excess` — the ratio is `body / floor`, so tuning the floor silently re-maps the
    score bands; raw is identical for BOS (floor 1.0) and only differs for CHoCH (floor 1.5).

- **BOS-sequence exhaustion penalty — REWORKED 2026-06-24, no longer count-based.**
  - SUPERSEDED by the `## SCORING — Structure / BOS continuation` section above.
    The penalty is no longer "BOS #>= caution -> score 1"; it is now a displacement-
    DECAY verdict (holding/fading). The old proxy finding below stands as the
    motivation but no longer maps to the live code.
  - Old data (proxy, for the record): "late" BOS (#>=3, structure=1) = 70 trades,
    39.7% WR, +0.042 expR, +$743 — the ONLY positive group; "early" BOS (#1-2,
    structure=3) = 83 trades, 37.1% WR, -0.057, -$1,184. This is what hinted a raw
    COUNT penalty was backwards — and is part of why count was dropped as the driver.
  - **Re-test the NEW design:** tabulate WR/expR by verdict (holding vs fading),
    bootstrap CI, then decide whether the 1-vs-3 split / 0.60 ratio should move.

- **Sweep: data contradicts SMC — detector suspect, kept out of the algorithm.**
  - Data: sweep_present = True is 40 proximal trades, 16.7% WR, expR -0.440; the fractional
    sweep-point setups went 0% across 13 trades. SMC says a pre-OB liquidity sweep is a
    POSITIVE confluence.
  - Why not concluded: data (inverted) vs SMC (positive) disagree, N is thin (40), not
    CI-tested. Likely the detector flags the wrong sweep (a sweep INTO the move = late entry,
    rather than a sweep of OPPOSING liquidity before the reversal).
  - **Loop closed:** audit the sweep detector before scoring it. Until then sweep stays OUT of
    the algorithm and is judged by eye. Do not reward or penalise it.

---

## KNOB SWEEP — MIN_LEG_ATR_MULT: lean 1.5 → 2.0 (2026-06-28, NOT yet shipped)

**The knob:** minimum swing-leg size to count a swing, in ATR units (`dealing_range.MIN_LEG_ATR_MULT`,
re-exported in `smc_detector`). Higher = ignore small wiggles, keep only meaningful legs → cleaner
OBs/structure, fewer trades. **Live = 1.5.** Grid tested: 1.0 / 1.5 / 2.0 / 2.5.

**Source:** 11 monthly Knob-Sweep runs on `origin/main`
(`backtest/diagnostics/sweeps/sweep_MIN_LEG_ATR_MULT_*`), all PASS / recon_ok / scope_ok. Months are
single calendars **scattered across 2008–2024** (varied regimes), NOT a contiguous two years — the
email's "two-year" label is a stale code header. Numbers below reproduced byte-for-byte from
`results.jsonl` (filled, proximal, score ≥ 4 — i.e. what live would have traded).

**Context first — the system is ~breakeven over these months, so all deltas are small:**

| value | filled trades | expectancy (R/trade) |
|---|---|---|
| 1.0 | 885 | −0.014 |
| 1.5 (live) | 866 | −0.022 |
| 2.0 | 802 | −0.012 |
| 2.5 | 747 | −0.001 |

**Consistency — did each value beat 1.5, month by month (all pairs pooled, Δexp vs 1.5):**

| Month | Market | 1.0 | 2.0 | 2.5 |
|---|---|---|---|---|
| 2008-10 | GFC crash | −0.068 | −0.004 | +0.097 |
| 2009-04 | recovery | +0.033 | +0.067 | −0.019 |
| 2011-08 | US downgrade | +0.043 | −0.070 | −0.063 |
| 2013-06 | calm | −0.002 | +0.058 | −0.046 |
| 2015-08 | China shock | +0.007 | +0.044 | +0.246 |
| 2016-11 | US election | −0.012 | +0.014 | −0.054 |
| 2018-12 | Q4 selloff | −0.031 | +0.023 | +0.085 |
| 2020-03 | COVID crash | +0.036 | +0.044 | +0.194 |
| 2021-07 | calm | −0.013 | −0.065 | −0.043 |
| 2023-03 | banking crisis | +0.046 | +0.047 | −0.028 |
| 2024-07 | calm | +0.022 | −0.051 | −0.098 |
| **won / 11** | | **6** | **7** | **4** |
| win_frac / median / mean | | 0.545 / +0.007 / +0.006 | **0.636 / +0.023 / +0.010** | 0.364 / −0.028 / +0.025 |

**By asset class (expectancy R/trade; trade counts show the weight):**

| Class | trades | 1.0 | 1.5 | 2.0 | 2.5 |
|---|---|---|---|---|---|
| Forex (bulk) | ~600 | +0.018 | +0.017 | **+0.046** | +0.054 (OOS-negative = luck) |
| Gold | ~130 | −0.024 | −0.027 | −0.037 | **+0.034** (OOS holds) |
| Index NAS100 | ~80 | −0.240 | −0.303 | −0.404 | −0.439 |

**Verdict — 2.0 is the best-supported single value, NOT a sure thing:**
- 2.0 wins the most months (7/11), has **no disaster month**, and is positive on median, mean, and
  the even-month OOS holdout. Its wins span crash *and* calm regimes → not regime-specific.
- 1.0 is barely different from 1.5 (weaker filter, near-zero effect).
- 2.5's high *mean* is a mirage: it won only 4/11; two fat months (2015 +0.246, 2020 +0.194) carry it
  while its median is negative. Don't be fooled by the average.
- **SMC agrees with the data** (bigger min leg = more significant structure, fewer junk OBs) → for
  forex + gold this is a "conclude" alignment, not just a discussion point.
- Index disagrees (wants the filter lower) but it's a thin (8 months, ~80 trades) **already-losing**
  book on this system — does not override the global call.

**If the knob is ever split per class:** forex 2.0, gold 2.5, index leave ≤1.5. That's where the edge
concentrates.

**Honest weaknesses:** edges are hundredths of R on a breakeven system; 3 of 2.0's months genuinely
lost; 11 months / 6 OOS is a modest jury. Treat 2.0 as **"the supported lean," not "proven."** More
months (full MT5 history) would tighten it. **Status: NOT shipped** — changing a live trading knob
needs explicit sign-off; one-line change in `dealing_range`.

---

## CANDIDATE — Asia-session high/low as a liquidity level (deferred, user 2026-06-26)

Add the **Asian-range high/low** to the liquidity hierarchy (alongside PDH/PDL/PWH/PWL) as a
draw-on-liquidity target / sweep level. ICT rationale: the Asian range is the canonical
intraday session pool — London's first move often raids it before the real direction.

- **Status: deferred, not built.** Intraday + DST-sensitive → higher maintenance than the
  daily/weekly levels, which go first.
- **When to pick up:** after the new score filters + the daily-bias narrative layer are live
  and producing numbers, so Asia's marginal value is measured ON TOP of them, not in a vacuum.
- **Scope note:** of the session levels (Asia/London/NY), **Asia is the most ICT-justified and
  should be first** (and possibly the only one added). London/NY standalone session highs/lows
  are emphasised far less. See `DAILY_BIAS_BUILD_HANDOFF.md` §2.7.

---

## CAVEAT — HTF liquidity pools: cross-source / boundary / feed fidelity (measured 2026-06-28)

Backtest = MT5; live = yfinance. The **bias DIRECTION** is cross-source robust (98.8%) because
it reads the *sequence* of swings. **Pools (PDH/PDL/PWH/PWL) are NOT** — they are absolute price
lines, sensitive to the candle boundary and to feed quality. Measured findings:

- **Daily boundary differs by feed.** MT5 rolls the day at **21:00 UTC** (00:00 server, GMT+3,
  DST-free = ~5pm NY = the forex-standard "New York close"; verified: MT5 H1 reproduces MT5 D1
  **100%** at server-midnight). yfinance FX (`=X`) rolls at **00:00 UTC** (verified: MT5-resampled
  best-matches yfinance native daily at the 00:00 anchor, falling off either side). 3-hour gap →
  different daily high/low on days whose extreme prints in that window.
- **The forex daily boundary is NOT ambiguous.** It is 5pm NY — the same broker maintenance
  window seen at **2:30–3:30 AM IST** (the 1-hour spread is US DST moving 5pm NY between 21:00
  and 22:00 UTC). yfinance simply ignores it.
- **DST parity trap.** MT5 server is fixed GMT+3, so its boundary is **21:00 UTC year-round** =
  5pm NY in summer, 4pm NY in winter. To keep backtest = live, any live D1 builder must mirror
  the **fixed 21:00 UTC**, NOT chase the shifting true-NY-close, or live silently desyncs.
- **yfinance hourly is low-fidelity — cannot rebuild clean pools.** yfinance `=X` hourly cannot
  even reproduce yfinance's OWN native daily (agree ≤3p only **31%**; on **41%** of days the
  hourly misses the true daily high by >3 pips). So resampling yfinance H1 → D1 does NOT give
  tick-accurate pools (vs MT5: ≤3p **5%**, ≤5p 67%, ≤10p **94%**). Weekly resample from yf hourly
  fails outright (sparse). Net: exact-tick pool parity with yfinance is **unattainable**.
- **MT5 feed is not an option for live.** Live runs on GitHub Actions (`ubuntu-latest`). The
  `MetaTrader5` package is **Windows-only** and needs an open terminal logged into a broker. Free,
  but not feasible without re-architecting live onto an always-on Windows host. → ruled out.
- **BTC is worse — venue divergence.** MT5 BTCUSD (broker CFD) vs yfinance BTC-USD (crypto index)
  differ by **~$40 median** on the daily high — far beyond FX feed noise — because crypto prices
  diverge across venues. Also MT5 rolls BTC at 21:00 UTC, but the **crypto-canonical day is 00:00
  UTC** (every crypto chart's default; where retail stops actually sit). So BTC pools must be
  built at **00:00 UTC** (not MT5's native 21:00), and BTC must be **bucketed separately** in
  analysis — the geometry is sound, the signal is noisier.

**Consequences for the build (durable):**
- Treat every pool as a **ZONE with ATR-scaled tolerance, not an exact line** — this is correct
  SMC anyway, and it absorbs the few-pip cross-feed divergence (≤10p captured 94% of FX days).
- Build D1/W1 from H1 at a **fixed, instrument-specific boundary** (FX/Gold 21:00 UTC; BTC 00:00
  UTC), the same construction on both feeds — boundary parity by construction; feed noise residual.
- **Open decision (parity vs canon):** the backtest must predict live. If live can only reliably
  deliver yfinance native daily (00:00 UTC), there is a real argument to mirror THAT boundary in
  the backtest (resample MT5 H1 → 00:00 UTC) so the backtest measures what live will see — at the
  cost of a non-canonical FX day. Decide before scoring pools. Don't chase tick parity; chase
  same-boundary + zones.

---

## CAVEAT — Daily "ranging" mode is not a forward predictor; play the bias REACTIVELY (2026-06-28)

The handoff's `STRUCTURE_RANGING_STALE` flag (predict ranging when N trend-swings fail to extend)
was tested as a forward regime predictor on D1, 5 instruments (EURUSD/XAUUSD/BTCUSD/USDJPY/GBPUSD,
N≈3,100 each). Ground truth = forward 10-day held-continuation vs containment.

- **Stale-count accuracy 45–50%, BELOW the 51–55% majority base rate.** Width/compression 36–39%.
  **Predictive ranging is coin-flip or worse — do not gate or score on it as a forecast.**
- **Reactive classifier (is price already expanding beyond the structural wall) = 73–79%,**
  consistent across all 5 instruments — far above base rate.
- **Lesson (veteran + data):** you cannot forecast the daily regime at high accuracy; you CAN
  read it from confirmed structure. Play the bias reactively: **break of the bias-direction wall
  that HOLDS → expansion → press toward the next draw; a SWEEP (wick beyond + close back) →
  exhaustion/range → demote/fade.** This is event-based (closed candle vs level), so the
  underlying sweep/break classification is near-deterministic — that is where >90% reliability
  actually exists, NOT in a forward-regime label.
- `mode=ranging` stays in Part A as **logged observational data**, but its role is an input to
  correlate, not a robust predictor. The robust play is the reactive sweep/break read.
- Next: layer the validated confirmations (N=1 held break, displacement body/ATR, sweep-and-
  reverse) on the reactive classifier to push precision toward mid-80s — test in Part B.
