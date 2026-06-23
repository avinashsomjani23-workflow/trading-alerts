# HANDOFF — Trading Alerts H1 System Audit & Rebuild Plan

**Written 2026-06-22, end of a deep audit + first-fixes session.**
Purpose: let a *fresh chat* (zero conversation history) become as sharp, honest, and
context-rich as the chat that produced this — instantly. Read this + `CLAUDE.md` +
`backtest/RECOMMENDATIONS.md` and you have the whole picture.

The chat history is NOT the memory. **These three files are:** `CLAUDE.md` (rules),
`backtest/RECOMMENDATIONS.md` (proven findings + methods + experiments), this `HANDOFF.md`
(state, open threads, how-to-think). Plus `~/.claude` auto-memory.

---

## 0. HOW TO USE THIS DOC
- New chat onboarding order: `CLAUDE.md` → this file → `backtest/RECOMMENDATIONS.md`.
- Confidence labels used throughout: **[PROVEN]** (measured, CI excludes zero / reconciled),
  **[SUGGESTIVE]** (in-sample signal, not robust), **[UNTESTED]** (we never checked — do not
  assume), **[ASSUMPTION]**.
- Every number below was computed this session from real run data and is reproducible.

---

## 1. THE ONE-PARAGRAPH TRUTH
The backtest **engine is honest** — it measures reality correctly (P&L reconciles to the
cent, 7 trades hand-traced with zero error, parity with live verified). The **strategy
loses money** on the one year of data we have: the model live actually trades (proximal
entry) returns **-53.5R over 340 trades, expectancy -0.157R, win rate 30%, 95% CI
[-0.28, -0.03]** — a statistically real loss, not bad luck. The **confluence score has zero
predictive power** (Spearman 0.05). BUT: the losses are mostly *given-back winners* (74-79%
of full losers were up >= +0.5R first), so **exit management + instrument selection are real,
cheap levers**, and the whole verdict rests on **one year of unvalidated yfinance data** that
must be re-checked against the user's broker (MT5) before anything is rebuilt.

---

## 2. TWO VERDICTS (keep them separate — always)
**Q1 VALIDITY (is the backtest measuring reality?) — YELLOW.**
- Engine machinery: GREEN. Reconciliation exact, hand-trace clean, fill/exit rules lean
  pessimistic (safe), open-stamped bars confirmed, parity real.
- Drags to YELLOW: (a) the headline *sums two entry models* but live trades only one
  (proximal) — judge by proximal; (b) **yfinance FX `=X` wick precision is unverified**;
  (c) small gap-at-stop optimism (~4R/yr); (d) **swing/CHoCH correctness never tested**.

**Q2 QUALITY (is the strategy good?) — RED.**
- Proximal (live model): -0.157R/trade, WR 30%, CI excludes zero on the negative side.
- Only 1 of 4 quarters positive; 2026 windows also negative; score is noise.
- This is **conditional on the data being clean** — see §6.

---

## 3. NUMBERS REFERENCE (all [PROVEN] this session)
Source = 4 clean consecutive quarters 2024Q3..2025Q2, proximal (= live) unless noted.
Risk $250/trade. avg win = 1.69R → **breakeven WR = 37%** (`WR x 1.69 = (1-WR) x 1`).

**Pooled scoreboards**
| Model | N | totR | exp_R | WR | 95% CI | verdict |
|---|---|---|---|---|---|---|
| proximal (LIVE) | 340 | -53.5 | -0.157 | 30% | [-0.28, -0.03] | real loss |
| 50pct (study only) | 311 | -23.3 | -0.075 | 21% | [-0.255, +0.115] | crosses 0 |

**Per quarter (proximal):** 2024Q3 -0.169 / 2024Q4 +0.004 / 2025Q1 -0.117 / 2025Q2 -0.342.
Only 2024Q4 ≈ breakeven; rest lose.

**Per pair (proximal, pooled):**
| Pair | N | exp_R | totR | note |
|---|---|---|---|---|
| NAS100 | 72 | -0.446 | -32.1 | **~60% of total loss, neg every qtr** |
| USDCHF | 58 | -0.235 | -13.6 | |
| EURUSD | 70 | -0.147 | -10.3 | |
| USDJPY | 39 | -0.086 | -3.4 | |
| GOLD | 46 | +0.066 | +3.1 | marginal, small N |
| NZDUSD | 55 | +0.051 | +2.8 | marginal, small N |

**Score validation:** Spearman(score, r) = 0.050, p=0.36. Buckets: 4→-0.075, 5→-0.394,
6→-0.025, 7+→-0.049. Non-monotonic = noise.

**Hard Truth 2 (the key insight):** 199 full -1R losers; **79% were up >= +0.5R first**
(62% >= +0.7R, 47% >= +1R), median MFE +0.92R. Winners dip only -0.40R median. Ex-NAS:
148 losers, 74% >= +0.5R. → losses are given-back winners → fix EXITS.

**Break-even re-sim (validated to 0.0005R vs committed):**
| Config | totR | exp_R |
|---|---|---|
| BE +1.0R (current) | -53.5 | -0.157 |
| BE +0.7R | -49.2 | -0.145 |
| BE +0.5R | -39.4 | -0.116 |
| BE +0.5R + Fri-flat @18:00 UTC | -39.5 | -0.116 |
| BE +0.5R + Fri-flat + ex-NAS | -16.4 | -0.061 |
| + fresh-FVG + PD-aligned | +11.4 | +0.148 (only 2/4 qtrs — fragile) |

**TP/BE distribution (filled):** reach +0.5R 88% / +1R 69% / +1.5R 44% / +2R 26% / +3R 10%.
Median MFE +1.37R. Of +1R-touchers: median peak +1.72R, 37% reach +2R, 14% reach +3R.

**PD alignment [SUGGESTIVE, NOT robust]:** aligned -0.09 CI[-0.26,+0.09] (crosses 0, pos
1/4 qtrs); counter -0.24 CI[-0.41,-0.06] (robust); diff +0.15 CI[-0.09,+0.40] crosses 0.
→ rule = **avoid counter-PD only**; do NOT reward "aligned".

---

## 4. WHAT'S VERIFIED ABOUT THE SYSTEM (don't re-audit these)
- **P&L is honest:** `pnl_usd == r_realised x 250` on every row; headline reconciles to the
  cent; gates G1-G9 PASS; 7 trades hand-traced from raw bars with 0 mismatch.
- **Bars are open-stamped UTC** (first FX bar Sun 23:00 = FX week open). `_slice_closed_before`
  (`index < ts`) is correct. No OHLC violations anywhere.
- **Parity with live is real:** backtest calls the live functions directly —
  `compute_pair_walls`, `detect_smc_radar`, `is_ob_mitigated_phase1`, `compute_phase2_levels`,
  `run_scorecard`. Knobs byte-identical (G5 pass).
- **Live trades ONLY the proximal limit** (`compute_phase2_levels` defaults
  `entry_zone="proximal"`, Phase2_Alert_Engine.py:2496). The 50% mean entry is a
  backtest-only A/B study. **Never sum the two as "the result."**
- **Killzone / news / IST are INFORMATION-ONLY in live** (never suppress an alert). The only
  live gate is the **score floor (min_score_to_email = 4)**. Backtest matches this.
- **Dedup = one trade per OB** matches live's "lifetime dedup with daily re-send" (one limit
  per zone; reminders are not new positions).
- **Config:** atr_multiplier 2.5 fx / 3.0 gold-nas; distal_invalidation_mode "close";
  min_score 4; spreads per pair; risk $250.

---

## 5. CODE / FILE CHANGES MADE THIS SESSION (local, not yet shipped)
1. **`backtest/h1_only_simulator.py`**
   - `_score_h1_only`: score now uses closed-only slice `df.index < alert_ts` (was
     `:alert_ts`, a 1-bar look-ahead). [fixed]
   - `_fvg_state`: window now excludes the forming bar (same leak class). [fixed]
   - **Weekend-flat:** `WEEKEND_FLAT=True`, `WEEKEND_FLAT_HOUR_UTC=18` (Fri 18:00 UTC =
     IST midnight = end of user's session). New exit reason `friday_flat`. Cost ~0R. Set
     `WEEKEND_FLAT=False` to disable.
2. **`backtest/scanlog/gates.py`** — `run_health.json` now also reports
   `live_proximal_headline_usd` (what live trades) and `study_fifty_pct_headline_usd`.
   Integrity gates untouched.
3. **`backtest/h1_only_reporting.py`** — `friday_flat` exit-reason labels added.
4. **`backtest/RECOMMENDATIONS.md`** — NEW. Reliability method (bootstrap CI), core insight,
   WR rule + formula, proven findings, TP/BE experiments table.
5. **Memory:** `backtest_reliability_method.md` + MEMORY.md index line.
- Safety net after all edits: **33 pytest PASS, 32/32 structure-golden, behaviour-neutral
  PASS, all files compile.**
- **NOT committed to git.** Ship when ready (stage only these files; skip settings.local.json).

---

## 6. DATA — the load-bearing caveat
- Source: yfinance. Forex = `=X` retail spot (lowest quality); index/metal = `NQ=F`/`GC=F`
  futures (cleaner). Flat/synthetic bars: NZDUSD 1.3%, EURUSD 0.8%, futures ~0%.
- **Wick precision is UNVERIFIED.** Every SL/TP hit depends on the wick. No second source in
  repo to cross-check. **This caps confidence in the entire Q2 verdict.**
- **ACTION (do before any rebuild):** pull H1 from the user's **MT5** (`MetaTrader5` pip pkg,
  free demo, Windows-only, broker server-time → convert to UTC) OR **Dukascopy**
  (`dukascopy-python`, free, no account, UTC, clean but not the user's exact broker). Re-run
  the 4 quarters on real data and see if the loss holds.
- User trades off **candle shapes at levels**, not exact prices → MT5 (their own feed) is the
  ideal match.

---

## 7. OPEN THREADS (the work — split across two chats)
**[UNTESTED] / [OPEN] — ranked by how foundational:**
1. **Swing/CHoCH correctness.** `SWING_LOOKBACK=3` (7-bar fractal) + ATR leg filter are
   hardcoded, never validated vs a vet's eye. User reports **"CHoCH is failing."** Everything
   (BOS/CHoCH/OB) sits on this. Golden tests prove *determinism*, NOT correctness.
2. **Data validation (MT5/Dukascopy).** §6.
3. **H4 vs D1 dealing-range walls** — user hypothesis that bigger TF = more stable. [UNTESTED]
   — second-order; test after 1-2.
4. **Exit redesign:** multi-leg simulator (current sim is single-exit) → partial TP at +1R +
   BE 0.5 + trail runner. Highest-potential P&L lever. See experiments table in RECs.
5. **EV-score (replace confluence points):** gate-then-rank, calibrate on 2024, validate on
   2025-26, require monotonic score→E[R] on holdout, positive net of spread. Start with GATES
   only (avoid counter-PD, drop NAS, skip "ranging" regime, require fresh FVG), then add one
   feature at a time keeping only those whose holdout CI improves.
6. **Regime gating:** live already computes `structure_v2.state` + `evaluate_ranging`
   (smc_radar.py:2703) but it **gates nothing**. Log it, test if "ranging" predicts worse
   outcomes, then wire as a gate.
7. **Per-instrument sweep test** — sweep relevance (FX vs NAS/Gold) is [UNTESTED]; user
   believes it matters more on NAS/Gold. Don't assume; test per pair.
8. **Add features to the logger:** `fvg_size_atr = (ghost_top-ghost_bottom)/ATR`, regime
   label, so they can be mined. (Displacement already exists: `compute_break_quality` returns
   raw `body_atr`/`close_beyond_atr` AND ratio `excess`.)

**Deprioritised (low ROI, don't over-engineer):** volatility-percentile feature.

---

## 8. THE TWO-CHAT PLAN
**Chat 1 — FOUNDATION (do first; nothing downstream is trustworthy until this is done):**
- MT5/Dukascopy data import + re-run 4 quarters on real data + report if loss holds.
- Swing/CHoCH correctness: build VET-LABELED fixtures (hand-mark correct swings/CHoCH on ~20
  windows), test algo against the labels. Investigate "CHoCH is failing".
- Optional: H4 vs D1 walls A/B.

**Chat 2 — EDGE (after foundation):**
- Build multi-leg exit simulator → run the TP/BE experiments table.
- Build EV-score (gates first, then calibrated ranking, holdout-validated).
- Wire regime gating from the existing `structure_v2`.
- Per-instrument sweep test; add `fvg_size_atr` + regime to logger.

**Ready-to-paste kickoff for a new chat:**
> "Read CLAUDE.md, HANDOFF.md, and backtest/RECOMMENDATIONS.md first. We are working on
> [Chat 1: data validation + swing/CHoCH correctness] / [Chat 2: exit redesign + EV-score].
> Keep the same rules: brutal honesty, no sycophancy, read code before claiming, bootstrap-CI
> before trusting any edge, gate-then-validate, plain English + bullets. Start by confirming
> the current state from the handoff, then [task]."

---

## 9. HOW WE THINK (the rules that made this chat good — keep them)
- **Anti-sycophancy (CLAUDE.md):** state disagreements; no praise-before-reasoning; read the
  code before claiming; hold position under pushback unless there's NEW evidence; point out
  flaws even if unwelcome. (This session: corrected own over-claims on PD-aligned and on
  sweep-FX when the data/user pushed back — that's the standard.)
- **Separate VALIDITY from QUALITY.** A correct backtest of a bad strategy is the common case.
- **Separate PESSIMISTIC bias (safe, understates) from OPTIMISTIC bias (dangerous, overstates).**
  Flag every optimistic one loudly.
- **Bootstrap CI + per-quarter sign consistency before trusting ANY edge or filter.** CI
  crosses zero = unproven = trust neither way. (Method in RECOMMENDATIONS.md.)
- **A score concentrates an edge; it cannot create one.** If no sub-group is robustly
  positive, no threshold makes the system positive. Fix the edge (entry/regime/exit), not
  the points.
- **Gate-then-validate:** never deploy a filter without replaying it and measuring
  winners-vs-losers removed.
- **WR is meaningless without R:R.** Breakeven WR = 1/(1+avg_win). Judge expectancy, never WR.
- **Don't hardcode weights from one year** — calibrate, then validate on unseen data.
- **Don't sum proximal + 50%** — live trades proximal only.
- **Every number must trace to a file/row/command. No guessing. Say "not verified" honestly.**

---

## 10. GLOSSARY (reusable plain-English explanations)
- **R:** one unit of risk (here $250). +2R = made twice what you risked.
- **Expectancy (exp_R):** average R per trade. The single honest scoreboard number.
- **MFE / MAE:** max favorable / adverse excursion — furthest a trade went in profit / against,
  before exit. Used to study where to put TP/BE.
- **95% CI (bootstrap):** resample the trades 10k times → range the true average lives in.
  NOT about re-running the backtest (that's deterministic); about whether THIS sample of
  trades is representative. CI below 0 = real loser; crosses 0 = unproven. Measures sampling
  luck, NOT future regime change.
- **Break-even (BE) arm:** once a trade is +X R, move stop to entry so it can't become a loss.
- **Partial TP:** close part of the position at a target, "banking" that profit; let the rest run.
- **Monotonic:** higher score must always mean higher real result — never zig-zag.
- **Logistic regression:** smooth S-curve turning features → win-probability. Simple, hard to
  overfit.
- **Isotonic regression:** a staircase only allowed to step up — forces "more score → never
  less E[R]". Good for calibrating a raw score into honest expected-R.
- **Holdout / unseen trades:** data from a period NOT used to build the score. Build on 2024,
  test on 2025-26.
- **Gate vs score:** gate = binary trade/skip; score = rank survivors by estimated E[R].

---

## 11. FILE MAP
- `CLAUDE.md` — project rules (format, anti-sycophancy, methodology). Read every session.
- `config.json` — pairs, atr_multiplier, distal mode, min_score, killzones, spreads.
- `smc_radar.py` — structure/DR engine, `detect_smc_radar`, `evaluate_ranging` (regime, unused).
- `dealing_range.py` — `compute_structure`, `detect_swings` (SWING_LOOKBACK=3), swing/BOS/CHoCH.
- `smc_detector.py` — `run_scorecard`, `compute_phase2_levels`, `compute_break_quality`,
  sweep (`sweep_observed`), `is_ob_mitigated_phase1`.
- `Phase2_Alert_Engine.py` — LIVE alert path (proximal-only, scores, emails). Parity reference.
- `backtest/h1_only_simulator.py` — fill/exit sim (proximal + 50%), the engine under audit.
- `backtest/replay_engine.py` — bar-by-bar replay, alert state machine, look-ahead asserts.
- `backtest/run_backtest.py` — orchestration, dedup, gates wiring.
- `backtest/h1_only_reporting.py` — scoreboards, eligibility (`_is_eligible`,
  `_EXCLUDE_REASONS`).
- `backtest/scanlog/gates.py` — G1-G9 health gates, run_health.json writer.
- `backtest/RECOMMENDATIONS.md` — proven findings, methods, experiments. **Keep updating it.**
- `backtest/results/<run_id>/` — trades.csv, summary.json per run.
- `backtest/out/scanlog/<run_id>/run_health.json` — gate verdicts per run.
