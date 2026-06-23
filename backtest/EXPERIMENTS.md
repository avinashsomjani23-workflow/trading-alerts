# EXPERIMENTS — End-to-End Plan to Figure Out Scoring

**Purpose:** a numbered catalog. Open a new chat and say *"run E4"* (or *"E4 + E5"*).
Each experiment is self-contained: goal, hypothesis, method, data, pass/fail rule, what it
depends on, what it outputs. Read `CLAUDE.md`, `HANDOFF.md`, `RECOMMENDATIONS.md` first.

**Rules that apply to every experiment (non-negotiable):**
- Judge **expectancy (R)**, never win rate alone.
- **Bootstrap 95% CI** every result. CI crosses 0 = UNPROVEN, trust neither way.
- **Per-quarter sign consistency** too — a signal that flips sign across quarters is noise.
- **Train on 2024, validate on 2025–26 holdout.** Never report only in-sample.
- Net of spread. Proximal entry only (live model). Never sum proximal + 50%.

---

## THE LOGIC (why the order is fixed)

A score **cannot create an edge — it can only concentrate one**. So we cannot "figure out
scoring" until a robustly-positive sub-group exists to concentrate.

1. **Phase A — Foundation.** Is the data real? Is structure detection right? If not, every
   number downstream is fiction. *Blocks everything.*
2. **Phase B — Exits.** The losses are given-back winners (74–79% of full losers were up
   ≥ +0.5R first). Fixing exits is the biggest, cheapest lever. Do this before scoring,
   because it changes which trades are winners.
3. **Phase C — Gates + Score.** Find the robustly-positive sub-group (gates), then build a
   score that ranks survivors by expected-R. This is the actual "scoring" work.
4. **Phase D — Final validation + sizing.** One clean out-of-sample test of the whole stack.

```
E1 (data) ──► everything
E2 (structure) ──► C
E3 (multi-leg sim) ──► E4, E5, E6
E7 (gates) ──► E9, E10
E8 (logger features) ──► E9 ──► E10 ──► E13
E11, E12 feed E10
```

---

## PHASE A — FOUNDATION (do first; nothing downstream is trustworthy until done)

### E1 — Data validation (broker data vs yfinance)
- **Goal:** does the −53.5R loss survive on real broker wicks, or is it a yfinance artifact?
- **Hypothesis:** the loss is real, but SL/TP precision shifts because every hit depends on
  the wick, and yfinance FX `=X` wicks are unverified.
- **Method:** pull H1 for all 6 pairs, the 4 clean quarters (2024Q3–2025Q2), from MT5
  (Funding Pips, `MetaTrader5` pkg, terminal must be open during the one-time pull) and/or
  Dukascopy (`dukascopy-python`, no account). Convert to UTC. Re-run the backtest. Compare
  expectancy + bootstrap CI + per-quarter signs vs yfinance.
- **Data:** broker H1; existing yfinance caches for diff.
- **Pass/fail:** numbers within sampling noise → trust yfinance going forward. Material
  divergence → broker data becomes the source of truth, re-baseline everything.
- **Output:** a side-by-side table (yfinance vs broker) + a verdict on data trust.
- **Note:** one-time download, NOT a 24×7 process. See HANDOFF §6.

### E2 — Structure correctness (swing / CHoCH / BOS) — NOT by hand-marking
- **Goal:** is `detect_swings` (lookback=3 fractal + ATR leg filter) actually right? User
  reports "CHoCH is failing." Golden tests prove determinism, not correctness.
- **Why not hand-marking:** an amateur eyeball is unreliable ground truth and slow. Use
  objective, low-maintenance cross-checks instead:
  - **E2a Independent algorithm cross-check:** run a second, well-known swing method
    (e.g. ZigZag, or a different fractal lookback) and measure *disagreement rate* on the
    same bars. Where they disagree, inspect.
  - **E2b Multi-lookback stability:** a swing that survives lookback {2,3,4,5} is robust;
    one that flips is noise. Score each detected swing's stability; check if unstable swings
    cluster on losers.
  - **E2c Outcome-anchored test:** a correct CHoCH should precede the move it claims. Measure:
    after a detected CHoCH, does price actually do what a CHoCH implies (vs random bars)? If
    detection is wrong, this signal is weak.
  - **E2d Fast visual spot-check:** auto-export chart PNGs with swings/CHoCH/OB marked for the
    ~20 worst losers. You confirm/reject in seconds — far faster than marking from scratch.
  - **E2e Dealing-range location check:** wrong-side CHoCH (bullish OB in premium / bearish in
    discount) was the worst cohort in an earlier audit. Replay the dealing-range construction on
    those trades: if the range is drawn correctly it's a *filter* fix (reject wrong-side CHoCH);
    if drawn wrong it's a *calc* fix. Decide which before touching the calc.
- **Pass/fail:** disagreement rate quantified; "CHoCH failing" reproduced or refuted with a
  named cause (lookback too tight/loose, ATR filter, on-close vs live break).
- **Output:** disagreement stats + a list of concrete detector fixes (if any).

---

## PHASE B — EXITS (biggest cheap P&L lever)

### E3 — Multi-leg exit simulator (infrastructure)
- **Goal:** current sim is single-exit. Build one that supports partial TP + BE arm + trailing,
  with **bar-level sequencing** (so same-bar +0.5-then-SL sweeps do NOT falsely arm BE).
- **Pass/fail:** on the no-change config it must reproduce the committed single-exit totals
  to the cent (regression guard).
- **Output:** the engine that E4–E6 run on. Without it, BE/partial/trail CIs are guesses
  (trades.csv has no per-bar MFE column → cannot reconstruct BE off the CSV; see ci_filter.py).

### E4 — BE-trigger sweep
- **Goal:** find the breakeven floor that scratches losers without choking winners.
- **Method:** re-sim BE arm at {0.3, 0.5, 0.7, 1.0}R. Bootstrap CI each. Per-quarter signs.
- **Data point we already have (BE+1R→0.5R recovers ~14R ≈ $3,525/yr) is suggestive but was
  the over-generous CSV estimate; E3 gives the honest number.**

### E5 — Take-profit study
- **Goal:** is liquidity-TP too greedy? (88% reach +0.5R, 69% +1R, 44% +1.5R, 26% +2R.)
- **Method:** re-sim partial TP (50% at +1R, run rest) vs fixed TP (+1.5R / +2R) vs current
  liquidity TP. Bootstrap CI each. Find the TP that maximises E[R].

### E6 — Trail-after-+1R
- **Goal:** capture runners (of +1R-touchers, median peak +1.72R; 37% reach +2R).
- **Method:** once +1R hit, trail by ATR or last swing. Bootstrap CI vs E5 winners.

---

## PHASE C — GATES + SCORE (the core "scoring" work)

### E7 — Establish the gate set (binary trade / skip)
- **Goal:** find filters that *robustly* remove more losers than winners.
- **Candidates:** drop NAS100, require fresh-FVG, avoid counter-PD, skip "ranging" regime.
- **Method:** replay each gate alone and stacked. For each: count winners vs losers removed,
  bootstrap CI on survivors, train-2024 / holdout-2025–26.
- **Pass/fail:** keep a gate only if survivors' holdout CI improves and stays consistent.
- **STATUS (measured 2026-06, committed exits, ci_filter.py):** best stack so far
  = ex-NAS + fresh-FVG + PD-aligned → N=77, +0.092R, **CI [−0.19, +0.39] crosses 0 (UNPROVEN)**,
  3/4 qtrs positive. Gets us from a *proven loser* to *unproven*, NOT to a proven winner.
  Needs E1 (real data) + E3 (better exits) to retest.

### E8 — Logger enrichment (feature engineering prerequisite)
- **Goal:** log the features a score could use but that aren't captured yet.
- **Add:** `fvg_size_atr = (ghost_top−ghost_bottom)/ATR`, regime label
  (`structure_v2.state` / `evaluate_ranging`), displacement ratio (`break_excess` exists).
- **Output:** richer trades.csv for E9. No scoring decisions here — just data.

### E9 — Feature mining (one feature at a time)
- **Goal:** which features actually separate winners from losers out-of-sample?
- **Method:** for each candidate feature, split E[R] by feature value, bootstrap CI on the
  split, per-quarter signs. Keep only features whose **holdout** CI separates.
- **Pass/fail:** a feature earns a place only if its edge survives holdout, not in-sample.

### E10 — Build the EV-score (replace additive confluence points)
- **Goal:** a score that estimates **expected-R per setup**, not "how many confluences stack."
- **Method:** on 2024 gated survivors, fit logistic regression (features → win-probability) or
  direct E[R] regression. Calibrate raw score → expected-R with isotonic regression. Require
  **monotonic** score → E[R] on the 2025–26 holdout, positive net of spread.
- **Pass/fail:** if no monotonic, holdout-positive score exists, STOP — there is no edge to
  concentrate (per CORE INSIGHT). Report that honestly rather than overfit.

### E11 — Regime gating
- **Goal:** does "ranging" predict worse outcomes? (Live computes it but gates nothing.)
- **Method:** log regime (E8), measure outcome by regime, bootstrap CI. Wire as a gate only
  if "ranging" is robustly worse.

### E12 — Per-instrument sweep relevance
- **Goal:** user believes liquidity-sweep matters more on NAS/Gold than FX. Don't assume.
- **Method:** test sweep-present vs absent E[R] split per pair, bootstrap CI. Pair-specific
  weighting only if the holdout CI supports it.

---

## PHASE D — FINAL VALIDATION + SIZING

### E13 — Full-stack out-of-sample test
- **Goal:** one clean test of gates (E7) + best exit (E4–E6) + score (E10) on unseen data
  (2026 windows + any held-out period).
- **Method:** run the whole pipeline, bootstrap CI, per-quarter consistency, net of spread.
- **Pass/fail:** trade-live decision + position sizing on the top score slice ONLY if the
  holdout CI excludes zero and signs are consistent.

---

## QUICK STATUS BOARD
| Exp | Title | Status |
|---|---|---|
| E1 | Data validation (broker) | not started — **blocks all** |
| E2 | Structure correctness (no hand-marking) | not started |
| E3 | Multi-leg exit sim | not started — blocks E4–E6 |
| E4 | BE sweep | not started (CSV estimate only) |
| E5 | TP study | not started |
| E6 | Trail after +1R | not started |
| E7 | Gate set | **partial — best stack CI crosses 0 (ci_filter.py)** |
| E8 | Logger features | not started |
| E9 | Feature mining | not started |
| E10 | EV-score build | not started |
| E11 | Regime gating | not started |
| E12 | Sweep relevance per pair | not started |
| E13 | Full-stack OOS validation | not started |
