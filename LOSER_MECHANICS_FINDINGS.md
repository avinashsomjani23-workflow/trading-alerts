# Loser Mechanics — Consolidated Findings Ledger

**Consolidates three handoff files** (`DISCOVERY_FINDINGS.md`, `INSTANT_DEATH_DECISION_TREE.md`, `BREAKEVEN_SWEEP_HANDOFF.md`) into one ledger after the column-correction backtest re-run. Those three source files were deleted after this merge.

**Status caveat:** these discoveries were run on the pre-correction column data. The user re-ran the backtest with a corrected column, so every number here is **suspect until re-verified against the fresh run.** Treat as direction/method, not as bankable figures.

---

## 0. Source of truth + scope

- **Run:** `backtest/results/h1only_20080102_20251231/trades.csv` — 2008–2025 (18-year) backtest. 33,494 rows / 106 cols, 23,371 filled.
- **DISCOVERY split (where ALL headline numbers below live):** `alert_ts` in **2008-01-01 → 2016-12-31** (9 years), date-based per `edge_engine.py:75`. **11,352 filled trades.**
- **VALIDATION 2017–2021 / HOLDOUT 2022–2025 stay untouched** — reserved, opened once, only after a rule is frozen (guardrail C5).
- **Why discovery-only:** testing filters on deploy years = in-sample overfitting.

### CONFLICT RESOLVED — which numbers are canonical

- The old `INSTANT_DEATH_DECISION_TREE.md` carried **18-year figures** (64.3% of losers = 7,867 deaths; −0.0437R; 14/18 years; 12,228 true losers). These are **SUPERSEDED.**
- The **discovery-scoped numbers are canonical** (66.4% = 4,012 deaths; −0.0527R; 9/9 years; 6,039 true losers). Where the decision tree and the findings ledger disagreed, the discovery-scoped value wins — the 18-yr figures were the pre-scope-fix version.
- 18-yr totals kept ONLY where explicitly labelled as reference (P&L reconciliation, ceiling math).

---

## 1. Definitions (fixed)

- **Instant death** = `exit_reason=="sl"` AND `mfe_r <= 0.05`. All are exactly −1.0R. Zero breakeven contamination. Verified against raw entry/SL/exit prices.
- **True loser** = `r_realised == -1.0`. The `sl` bucket ALSO holds breakeven-stop 0.0R exits — those are NOT losers. **Always filter on `r_realised`, never raw `exit_reason`.**
- **Break-even (BE) exit** = `exit_reason=="sl"` AND `r_realised==0.0`. Armed BE (reached +1R), then price returned to entry and stopped for 0R.
- **Small OB** = `ob_range_atr < 0.658` (~19% of discovery trades).
- **Fill in killzone** = `fill_in_killzone == True` (entry inside an ICT session window). NOT `ob_in_killzone` (formation).
- **R is truth.** `pnl_usd` is fixed ~$250 risk/trade, cumulative across 10 pairs — NOT an account balance.

### Denominator discipline (non-negotiable)

Every % states its denominator. A "death rate" is deaths ÷ trades-in-that-group. It is NEVER a share of the loser pile unless it says so.

---

## 2. Verified facts (discovery, 2008–2016)

### FACT 1 — system loses on raw fills
- Discovery expectancy = **−0.0527R / trade.**
- Discovery total = **−598.6R = −$149,660.**
- Full 18yr reference: −0.0437R, −1,022R, −$255,616. Also negative.
- **No positive headline P&L in this run.** Any memory of "thousands positive" is a different/older/contaminated run.

### P&L reconciles (verified)
- DISCOVERY −$149,661 + VALIDATION −$86,583 + HOLDOUT −$19,373 = **−$255,616 = 18yr total.**
- Not "double discovery" — the **loss is time-varying:** first 9 years bled hard (−$16.6k/yr avg; 2016 only positive year), last 9 bled slowly (2016, 2017, 2025 positive; holdout near flat).
- → **Any filter MUST be checked per-year** or it looks good just by fitting the worse early years.

### FACT 2 — most losers die instantly
- **66.4% of true losers die instantly** = 4,012 / 6,039 (discovery).
- MFE curve flat past 0.05 → deaths cluster at *exactly zero* favorable move. Hard instant reversal, not "small MFE."
- Stable across every discovery year and all pairs.

### FACT 3 — OB size predicts DEATH, not WINS
- OB size moves instant-death rate by ~29 pts across quintiles, but win rate by only ~7 pts. Losing is structural (predictable); winning is ~luck.
- Small OBs are a **barbell** — worst losers AND best wins:
  - Small-OB average WIN = **2.063R** (tighter SL → higher R payoff).
  - Large-OB average WIN = **1.589R.**
- → A pure OB-size gate does NOT fix it. Cutting small OBs removes the fat wins too. All size floors tested stay negative (13 floors tested).
- **Strategy consequence:** we cannot engineer wins. We CAN avoid structural deaths. That is the whole game.

### FACT 4 — fill-in-killzone dies more (DISCUSSION POINT, contradicts ICT)
- **Death RATE inside KZ = 35.6%; outside KZ = 25.8%.** (rate = deaths ÷ trades-filled-in-that-group.)
- Outside-KZ beats inside-KZ in **9/9 discovery years.** Rock solid.
- Independent of OB size (survives the 2×2 — not a confound).
- **Mechanism UNKNOWN.** Volatility theory tested and REJECTED (see N1).
- Fights ICT doctrine (killzones should be *better*). Per project rule, stays a **DISCUSSION POINT**, not a scored filter, until mechanism proven. → Owns its own investigation (`KILLZONE_DEATH_HANDOFF.md` — see §7, file not present in tree).

### FACT 5 — the MECHANISM of instant death (the big one)
- **92.9% of instant deaths die within 1 H1 bar of fill; ~half die on the fill bar itself** (median `bars_to_exit` = 0). Slow losers take median 4 bars.
- Median SL distance of deaths = **1.11 ATR** — about one H1 bar's range. So an instant death = the bar that touches the OB (or the next) runs straight through zone AND stop in one move.
- `mfe_r` excludes the exit bar → any fill-bar death records mfe≈0 **mechanically.** Half the "instant" label is measurement granularity, not a distinct trade type.
- **Why no strong pre-entry signal exists:** whether the touching bar penetrates ~1.1 ATR deeper is intrabar momentum AT the touch — info that does not exist in any H1 pre-entry column. The 66% is geometry (limit at zone edge + ~1-bar stop + H1 bars), not a hidden regime we failed to log.
- Null-model check: a drift-free walk would make only ~8% of losers "instant"; observed 66% — gap explained by 1-bar discreteness, not a mysterious force.

### FACT 6 — death IS predictable, but the kept set never goes positive
- Death-vs-rest model AUC = **0.64 out-of-year, stable 0.63–0.66 in all 9 discovery years.** Real signal, not luck.
- Separation test passes: death-vs-WIN AUC 0.61–0.62 (does not collapse to 0.5).
- Top model features = the known ones: `ob_range_atr`, `leg_retrace_pct_at_alert`, `alert_utc_hour`, `impulse_leg_atr`, `fill_in_killzone`, `sl_distance_atr`. Nothing new surfaced.
- **Kept-set expectancy curve (the honest metric): never positive.** Cut riskiest 25% → kept exp −0.031R (37% of deaths cut, 21% of winners lost). Cut 50% → −0.025R. Best point −0.021R at 40% cut. The barbell eats every cut.
- Direct expectancy model (predict `r_realised`): top-20%-predicted kept set positive in only **3/9 years.** No stable tradeable subset.

### FACT 7 — half our stops die on a wick
- **53.0% of true losers' stop candles are SWEEPS** — wick through the SL, close back on our side (`sl_bar_was_sweep`, walker def `h1_only_simulator.py:951`). Stable 50–57% all 9 years. Instant deaths: 53.8%.
- **30.0% of true losers: sweep AND TP1 touched within lookback after** (`sl_swept_then_tp1`). Stable 28–33% all 9 years.
- Ceiling if all flipped to TP1 win at the SAME tight stop: −599R → **+5,218R.** **UPPER-UPPER bound, user flagged as a stretch.** Flaw: to survive the wick, stop must sit BEYOND the wick → wider risk → every R multiple shrinks. Illustrative re-pricing (rough, needs replay): wider stop 1.25× risk ≈ +4,800R, 1.5× ≈ +2,970R, **2.0× ≈ −111R (edge gone).** Prize depends ENTIRELY on how deep wicks actually pierce — shallow = transformative, deep = nothing.
- **Wick depth beyond SL is NOT logged** → CSV cannot size a wider stop, which is exactly why the ceiling can't be pinned. New column `sl_wick_depth_atr` being added (`h1_only_simulator.py`, beside `sl_bar_was_sweep`). Only a replay variant (SL = distal + X·ATR grid) proves the real number. **Top open lever by ceiling size.**
- Secondary: **breakeven stops swept just as often** (53.2%; 34.1% then TP1). BE parks the stop at the most visible level on the chart (the OB edge). BE placement is donating winners. → See §4.

### FACT 8 — 50%-zone entry REJECTED (number corrected 2026-07-08)
- **CORRECTION (user caught):** the −0.427R headline was a METHOD ARTIFACT, retracted. Winners filled using `mae_r`, which the walker records as exactly 0.000 on the fill bar → any winner that dipped to midpoint DURING its fill hour is invisible and wrongly skipped. But −1R losers fill with certainty (they provably traverse the zone). So method under-fills winners and over-fills losers BY CONSTRUCTION. The −0.427R inherits that bias — do NOT quote.
- **What survives (arithmetic, doesn't depend on flawed counting):** every −1R loser crosses the entire zone → a 50% entry fills 100% of losers. Winners fill shallow: even generously only ~22–43% dip to midpoint after filling. Skipped trades average **+1.75R.** Deeper entry keeps all trash, drops most treasure = adverse selection.
- RR roughly doubles (median 1.76 → 4.10) but win count collapses ~5×; doesn't compensate.
- **Honest status:** cannot be measured cleanly from logged columns (fill-bar dip not recorded). Only a replay can. A replay would need ~47%+ of winners filling deep just to TIE a losing baseline. Verdict (deeper entry = adverse selection) holds; the specific −0.427R does not.

---

## 3. The 2×2 that matters (DISCOVERY only, all filled trades)

| OB size | fill in KZ | n | exp_R | death% (of cell) | win% (of cell) | total R |
|---|---|---|---|---|---|---|
| large | outside | 3783 | **−0.022** | 25.8 | 30.1 | −84 |
| large | inside  | 5414 | −0.059 | 35.6 | 29.7 | −318 |
| small | outside | 886  | −0.045 | 45.4 | 24.6 | −40 |
| **small** | **inside** | **1269** | **−0.123** | **55.7** | 25.4 | **−156** |

- **Worst cell = small OB + inside killzone: −0.123R, 55.7% cell death rate.** User's hypothesis "small OBs in killzone are to be killed" — data agrees on discovery.
- **55.7% is the death rate WITHIN that 1,269-trade cell (~11% of trades). NOT "55.7% of all losers."**
- **Cannot just delete the cell (barbell):** it holds 25.4% winners — real wins. Any filter MUST report effect on kept-winners AND kept-losers + net survivor expectancy. Nuking a cell is not allowed unless survivors net positive.

> **NOTE:** the old decision tree carried an 18-yr version of this 2×2 (large/outside n=7866 −0.011R … small/inside n=2672 −0.101R 53.6%). Superseded by the discovery-scoped table above; kept here only as a pointer that the 18-yr shape was the same direction.

### PARKED — small OB OUTSIDE killzone (do not act yet)
- n=886, exp **−0.045R**, death 45.4%, win 24.6%, total −40R.
- Still negative but **least-bad among small-OB cells.** A *survivor of elimination*, NOT a money-maker.
- **Parked:** killing it loses the 2.06R small-OB wins, and it's not the primary bleed. Revisit only after killzone mechanism understood.

---

## 4. Break-even stop sweep (open investigation)

**Mission:** the BE stop (moved to entry at +1R) gets swept — price wicks back to entry, stops us for 0R, then continues to target. Find under WHAT CONDITIONS this happens and whether a better BE rule (delay / offset / drop) saves those trades without breaking the ones BE protects.

### What's verified (discovery 2008–2016)
- **BE exits = 1,895 trades = 16.7% of filled.** Armed BE (reached +1R, so `mfe_r >= 1.0`; median mfe 1.31, min 1.00), then price returned to entry and stopped for 0R.
- **53.2% of BE stop-outs are sweeps** (`sl_bar_was_sweep==True`; walker def `h1_only_simulator.py:942-953`).
- **34.1% of BE stop-outs: swept AND then touched TP1** within lookback (`sl_swept_then_tp1==True`). A third of break-evens were "correct" trades wicked out at the worst spot.
- **NOT instant:** median `bars_to_exit` = 4 from fill. The trade earned +1R over several bars, then gave it back on a wick.
- **Ceiling (touch-based HINT, never bankable):** the 646 BE-swept-then-TP1 trades, if ridden to TP1 instead of BE-stop, = **+1,615R** (avg tp1_rr 2.50). Assumes same geometry and touch = fill; both optimistic. Real number needs a replay.

### Mechanism (hypothesis, not proven)
- BE parks the stop AT ENTRY = the OB proximal edge = **the most obvious level on the chart.** Price routinely returns to test a freshly-broken zone before continuing. Our BE stop sits exactly where a retest wicks.
- So the BE rule may be systematically donating winners: converts "pullback-then-continue" (normal SMC) into 0R exits.

### Conditions of BE-swept trades (descriptive only — thin, do NOT gate)
- Session: London 42%, NY 37%, Asia 16% (~base-rate).
- Killzone alignment: "Fill only" 32%, "Neither" 26%, "Both" 23% (spread out).
- Pair: USDCHF 14%, GOLD 13%, EURUSD 12% (~base-rate).
- **Read:** no single condition jumps out. The lever is likely the BE RULE itself (where/when the stop moves), not a sub-population to exclude.

### Candidate rules to test (rank by data, do not pre-commit)
1. **Delay BE arming** — arm at +1.25R or +1.5R instead of +1R so a shallow retest doesn't catch the stop. Test threshold as a grid. *(testable on current data)*
2. **Offset the BE stop** — move to entry MINUS a small buffer (e.g. entry − 0.1R) so a wick to the exact entry level survives. **Needs `sl_wick_depth_atr`** (see dependency).
3. **Drop BE entirely** — let the trade ride the original stop to TP1. Risks giving back more on trades that reverse hard after +1R. Measure both sides. *(testable on current data)*
4. **Time-conditioned BE** — only arm after N bars, or only in certain sessions. Lower priority (thin conditioning).

### HARD DEPENDENCY — the wick-depth column
- To offset the BE stop correctly (rule 2) you need **how far the wick pierced entry**, NOT in the current CSV.
- New column **`sl_wick_depth_atr`** being added to `h1_only_simulator.py` (logs, for every `sl` exit, how far past the stop the wick went, in ATR). **Wait for the fresh backtest including it** before testing rule 2. Rules 1 & 3 test on current data.
- **Shared dependency:** the wider-stop replay (for true −1R losers, FACT 7) and this BE work share `sl_wick_depth_atr` and the same wider-stop replay engine — coordinate so one run serves both.

### Method (same discipline as the death hunt)
- **Every rule reports BOTH sides:** trades saved (BE-swept-then-TP1 that now win) AND trades hurt (post-+1R reversals that now lose more). Net expectancy of changed set vs baseline −0.0527R.
- **Per-year stability across all 9 discovery years** — not one year carrying it.
- **Bootstrap CI** on expectancy delta must exclude 0 (`backtest/RECOMMENDATIONS.md` method).
- **Thin = under 150 trades / bucket** (100 for interaction cells) — labelled, never shipped from (C3).
- Freeze the rule, THEN check ONCE on validation. Only if it survives, look at holdout (C5).

---

## 5. Rejected / ruled out (do NOT re-run these as new)

- **OB-size gate alone — does NOT fix it.** Barbell (worst losers AND best wins, 2.06R vs 1.59R). All size floors stay negative. Method: quintile death/win rates + size-floor expectancy sweep.
- **Killzone as a "lose more" signal — WEAK.** Losers filled in KZ 59.9% vs winners 58.7% (base 58.9%) → 1.3pt delta. KZ shifts losses toward the *instant* kind; does NOT change *whether* you win/lose. Method: KZ share among winners vs losers.
- **Volatility mechanism for killzone (N1) — REJECTED.** Vol z-score inside KZ = −0.035; outside = +0.050 → inside is *calmer*, not wilder (opposite of theory). Death gap persists ~38% inside vs ~27.5% outside in BOTH hi/lo vol buckets — if vol caused it, controlling for vol would collapse the gap; it didn't. Only real diff of inside-KZ fills: arrive faster (median 4.0h vs 5.5h), lean London — a timing difference, not vol. Effect stands; no logged column explains it → hand to killzone chat.
- **Fast-fill mechanism — REJECTED as a standalone cause.** Outside-KZ fast is BETTER (−0.007 vs −0.046). Only fast+inside-KZ is worst (−0.094) — a 3-condition cell (fast × inside × discovery), win rate normal (28.5%). Likely overfit. **Status: parked/unproven, NOT proven useless** (never luck-tested). Method: median-split `alert_to_fill_hours` × KZ. [Q8 detail: fast+inside n=3,850 exp −0.094 win 28.5% loss 54.6% death 40.3%; slow+inside n=2,833 exp −0.040 win 29.4% death 38.2%.]
- **Confirmation entry on H1 (N2) — NOT VIABLE.** Real confirmation waits for price to react inside the zone on M5/M15 before entering. System is H1-only; an H1 close-into-zone confirms on the SAME bar that already reversed = weak proxy. Needs an M5/M15 feed = bigger scope decision, deferred. **This is the lever most likely to break the barbell** (changes WHEN/IF you enter, not WHICH trades) — parked on data availability, not on merit.
- **50%-zone / deeper entry — REJECTED (adverse selection).** See FACT 8. Verdict holds; the −0.427R figure does not.

---

## 6. N3 verdict — selection is the WRONG lever (closed for logged columns, 2026-07-07)

- No filter built from logged pre-entry columns turns the kept set positive on discovery. Honest null (C4), tested to the floor.
- Full blind sweep: univariate ranking of ~50 pre-entry features (quintile/level death-rate deltas, N≥150/bucket), then RandomForest + HistGradientBoosting with **year-grouped CV** (model never sees the year it predicts), permutation importance on unseen 2015–2016, pairwise interaction sweep (N≥100/cell), bootstrap CI + per-year on survivors. `sweep_present`/`sweep_pts` excluded per guardrail A5.
- Luck-test verdicts on the last leads:
  - **`alert_utc_hour` 13–18 — FAILS.** Bootstrap 95% CI of exp delta [−0.004, +0.091] includes 0; beats OUT 6/9 years but effect only exists 2012+ (direction drifts). Not a driver.
  - **`killzone_alignment` = "Fill only" — worse 7/9 years but CI [−0.095, +0.005] includes 0.** Stays a DISCUSSION POINT in the killzone chat; not shippable.
  - **Interactions: essentially none (with a correction).** First pass "all within noise" was an overstatement. Two cells cross ±2-SE: Gold × Asia-fill (−12.4 pts death but n=139 = thin, and PROTECTIVE not harmful) and "Fill only" × fill-in-KZ (−4.1 pts, n=4,052 solid sample but tiny in money). Neither tradeable. Worst-expectancy cells (small OB × walkback>1, −0.28R n=190) are additive stacks of known effects, not new interactions.
- **The mechanism (death happens INSIDE the touch bar) points at entry mechanics, not trade selection:** lower-TF confirmation (N2, parked on data availability, not merit), entry deeper in the zone, or stops sized beyond one bar's reach. Any of these changes the geometry the deaths live in; no selection rule can.

### Luck-test methods (kept for reference)
1. **Bootstrap CI** — resample trades 1,000× per candidate; effect is real only if the expectancy-delta CI excludes 0 (`backtest/RECOMMENDATIONS.md`).
2. **Death-vs-win separation** — does a signal that flags losers ALSO flag winners? If yes it's an activity/volatility proxy, not a death signal. Must be high on deaths, LOW on wins.

---

## 7. Guardrails (DECISION_GUARDRAILS.md)

- Every % must state its denominator. No bare percentages.
- Data-vs-SMC: any finding that fights methodology (e.g. killzone) stays a DISCUSSION POINT until mechanism is proven.
- No gate ships without per-year + out-of-sample stability and expectancy (not just win rate).
- Hunting to ELIMINATE structural deaths, not manufacture wins (wins are ~luck).
- **Changing BE is a trading-logic change** → needs explicit user confirmation before any code touches the walker's exit logic.
- **B5 / Truth Ledger:** any new column (like `sl_wick_depth_atr`) needs a ledger row + disposition + guard before it feeds a decision.
- **C6:** touch-based ceilings (+1,615R BE, +5,218R sweep) are hints, never bankable. Only a real-order replay counts.
- **C5:** holdout opens once — freeze the rule, check validation once, then holdout.
- **Backtest-vs-live timing gap** ([[backtest-fill-timing]]): backtest alerts ~1 bar later than live. A rule that looks good may behave differently live; shadow before shipping (E1).

---

## 8. The prize — ceiling math (whole 18yr, filled)

If we could remove instant deaths **for free** (hit only losers, touch zero winners). Each is exactly −1R, so clean but a **ceiling, not a forecast** — no real filter spares all winners (barbell). Real gain always less.

| Remove | expectancy | WR | total R | P&L |
|---|---|---|---|---|
| 0% (baseline) | −0.044R | 36.3% | −1,022R | −$255,616 |
| **25% of deaths** | **+0.044R** | **40.4%** | +945R | **+$236,134** (+$492k swing) |
| 50% of deaths | +0.150R | 45.6% | +2,912R | +$727,884 |
| 100% of deaths | +0.442R | 61.5% | +6,845R | +$1,711,134 |

- **Removing just 25% of instant deaths flips the whole system from losing to winning.** Size of the prize; why the hunt is worth it.
- **DO NOT quote as expected results.** Free-lunch ceiling. Real filter cost (lost winners) comes out of every number.

---

## 9. Open threads / next

- **Wider-stop replay** (FACT 7) — top open lever by ceiling size. Blocked on `sl_wick_depth_atr` + replay engine. Shares the engine with BE work.
- **Break-even rule rework** (§4) — rules 1 & 3 testable now; rule 2 blocked on `sl_wick_depth_atr`. Trading-logic change → needs user confirmation.
- **Killzone death mechanism** — real effect (9/9 yrs), no logged column explains it. The 1-bar geometry (FACT 5) may explain it (killzone = when the big bars print) but that chat owns the proof. **`KILLZONE_DEATH_HANDOFF.md` referenced but NOT present in the working tree** — either not yet written or deleted.
- **`INSTANT_DEATH_HANDOFF.md` referenced but NOT present in the working tree** (death hunt, closed for selection filters; mechanism = 1-bar geometry).
- **`exit_reason == "breakeven"` relabel** — proposed to stop the sl-bucket confusion. Spec pending, code not touched.
- **Entry-mechanics lever** (the real one, per N3): M5/M15 confirmation, deeper zone entry, or bar-aware stops. All are scope decisions, not screens. Parked on data availability, not merit.
