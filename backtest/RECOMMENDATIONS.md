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
