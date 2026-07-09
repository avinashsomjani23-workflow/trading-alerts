# DECISION GUARDRAILS — The Center File

**Written 2026-07-03, from the pre-baseline playbook chat. This file exists so the trader
never loses center during the edge-engine phase.**

## How this file is used

- Before any decision, the trader (or any Claude session) states the intended action and
  checks it against the rules below.
- **Any Claude session reading this file MUST flag a rule violation by its ID** (e.g.
  "this breaks C5") before helping with the action. Flag first, help second.
- Rules are frozen. Changing a rule requires new evidence, stated in writing, and the
  change logged at the bottom of this file with a date. A rule may never be changed in
  the same sitting as the decision it would block.

---

## A — SETTLED DECISIONS (do not reopen without new evidence)

- **A1 — Doji filter stays.** OB candle body must be >20% of range (`smc_radar.py
  is_valid_ob_candle`). A doji has no directional intent; it is not an order block in
  any SMC framework. Removing it also rewrites existing setups via the walk-back, so
  "remove and let data decide" cannot produce a clean comparison. Settled 2026-07-03.
- **A2 — Oversized cap (2× ATR) stays for the baseline.** A >2× ATR candle is a news
  bar; its zone implies a >2 ATR stop — different trade geometry, not a comparable row.
  Testable later ONLY as a generation-2 detection change with discovery+validation
  evidence (the `ob_range_atr` screen rising toward the cap).
- **A3 — Walk-back is an OPEN question, answered by logging, not by removal.** Three
  schools: substitute (current), skip, merge. Vet lean: substitute-with-a-gap is the
  weakest. Decision path: log `ob_walkback_depth` + `ob_body_ratio`, screen in Stage 1,
  decide in generation 2. No behaviour change before evidence.
- **A4 — NAS100 is out of trade scope.** Never judge or tune on it.
- **A5 — Sweep detector is excluded** (suspected inverted). Rebuild is its own
  workstream. `sweep_present` never enters any screen as a feature.
- **A6 — Detection window stays 150.** Two out-of-sample sweeps agree it is a dead knob.
- **A7 — No OB minimum-size floor.** Tried and removed 2026-06-16. The displacement leg
  is the quality signal, not the OB candle's size. Do not re-add without sweep evidence.
- **A8 — Unresolved trades (never_filled / timeout / window_end) never feed P&L.**
  Audit-only. `eligible_for_headline` is the one population rule — never re-derive it.

## B — BASELINE-RUN RULES (before and during the 18-yr run)

- **B1 — Clean tree first.** Never launch the baseline with uncommitted changes. The run
  must be reproducible from a commit hash.
- **B2 — Prove plumbing on a short run.** The new run-ID format (email subject,
  commit_logs, registry) is verified on a ~3-month run before the 18-yr run.
- **B3 — Verify gates-off config BEFORE launch,** not after. Stage 0 will fail the whole
  engine if the run was score-censored.
- **B4 — After the run: health checks only.** G1–G10, scanlog, trade-count sanity
  (~1,200 eligible/yr), run ID present. Do NOT read results for edge. Every early peek
  is a look that can't be taken back.
- **B5 — New trades.csv columns before the run need:** truth-ledger row (source
  file:line, stamp timing, population) + structural guard + a disposition row in
  EDGE_ENGINE_SPEC.md §12. A column with no disposition is a Stage-0 FAIL.
- **B6 — The feature list freezes at first engine run.** Adding features is legal only
  while no run has fed the engine. After that: v2 spec, fresh holdout.

## C — ENGINE RULES (while stages run)

- **C1 — Never use `--force`.** A failed stage means the input is broken. Fix the input.
  A forced output is stamped untrusted and must never drive a decision.
- **C2 — Never extend mid-run.** No new features, buckets, recipes, thresholds, or
  splits "because the data looks interesting". New ideas go to a v2 list, tested on
  data they have not seen.
- **C3 — Thin findings do not exist for decision purposes.** A bucket under 150 trades
  (100 for interaction cells) is labelled and kept visible, but no rule ships from it.
  A thin sample never overrides SMC method — and never confirms it either.
- **C4 — Honest nulls are results.** Empty survivors, NO_USABLE_EV,
  no_exit_improvement — all valid, all publishable. Never loosen a threshold to make a
  finding appear.
- **C5 — Holdout (2022–2025) is opened ONCE, by Stage 4.** Never open those years by
  hand. Any upstream change after Stage 4 ran = holdout burnt for that change.
- **C6 — Peak/MFE numbers are never bankable.** A touch is not an exit. Only real-order
  replays through the walker count (`verify_capturable` or silence).

## D — GENERATION DISCIPLINE (the outer loop)

- **D1 — One detection change per generation** (or one pre-declared bundle). Three knobs
  at once = no attribution = wasted generation.
- **D2 — A change may only be motivated by discovery + validation evidence.** An idea
  that came from a holdout table is tainted and may not seed a change validated on
  those same years.
- **D3 — Three generations max on the current holdout.** After that the exam is
  memorised: roll holdout forward (e.g. 2023–2026) and say so in the report.
- **D4 — "One more re-run" is arguing with the verdict.** Re-running with tweaks until
  the number improves is the exact failure this whole design prevents. The generation
  stamp makes it visible; the discipline is for the trader, against the trader.

## E — VERDICT REACTIONS (pre-committed 2026-07-03, before any result was seen)

- **E1 — ENTRY_AND_EXIT_EDGE:** do NOT wire live the same week. Shadow the recipe
  against live alerts first (known backtest-vs-live timing gap). Excitement is the
  enemy here, not doubt.
- **E2 — ENTRY_EDGE_ONLY:** ship the EV gate, keep baseline exits. Small clean change.
- **E3 — EXIT_EDGE_ONLY:** accept it — entries are a coin-flip and the legacy score
  predicts nothing. Arguing is allowed only with discovery/validation evidence, never
  with "it feels wrong". Ship exits; queue detection rework as generation 2.
- **E4 — NO_EDGE:** not four months wasted — it is a fact learned without losing money.
  Next move: the report's `detection` queue + the sweep rebuild. Tweak, don't rebuild.
- **E5 — FRAGILE stamp = "not yet",** whatever the verdict says. A fragile edge dies
  the month it gets funded.
- **E6 — No ship/don't-ship decision in the same sitting as reading the report.** Read,
  close, sleep, decide tomorrow. Nothing here has an hours-scale deadline.

## F — STATISTICAL CENTER (what thin / luck / driver mean — frozen)

- **Thin** = under 150 trades in a bucket. Two lucky trades move that average more than
  a real edge would.
- **Luck** = shows in 2008–2016, fails to repeat in 2017–2021. Luck doesn't repeat on
  unseen years.
- **Driver** = real in discovery AND repeats in validation with the same direction AND
  ≥60% of quarters agree AND effect ≥0.10R. All four, no exceptions.
- The exact numbers are judgment calls; the protection is that they were frozen before
  looking. **Renegotiating a threshold after seeing a result is self-deception by
  definition** — the number was chosen when there was nothing to flatter.
- **F-BUCKET — Never judge a feature on one summary number.** Every feature screened in
  discovery MUST print its full per-bucket curve — each bucket's N, win rate, mean R, and
  **straight-to-SL rate** — with the BEST and WORST buckets named. A single top-vs-bottom
  number, a correlation, or an MI value alone may NEVER be the verdict: a death cliff in
  one tail (small OBs) or an edge living in only one bucket both hide inside one number.
  The per-bucket SL rate is mandatory so an entry feature is always crossed against loser
  behaviour — a descriptive lens, never an entry input (the look-ahead wall stands).
  Enforced in code + test in the discovery module, out-of-band. (Full rule: EDGE LAB spec
  §3 "Bucket-reporting rule". Added 2026-07-09 after the Step-2 small-OB cliff miss.)

## G — EMOTIONAL RULES

- **G1 — The verdict judges one detection basis, not the trader.** The machine (18-yr
  MT5 data, truth-audited logging, pre-registered engine) is the asset and survives
  every verdict.
- **G2 — Data vs SMC disagreement = discussion point, never a conclusion.** Name the
  likely cause (detector bug, thin sample, timeframe mismatch), brainstorm, don't act.
- **G3 — No decisions tired.** Late-night reads are fine; late-night decisions are not.
- **G4 — The only real way to fail this phase is overriding the machine** built
  specifically so the answer would be trustworthy either way.

## RED-FLAG PHRASES (if the trader or Claude says one of these, stop and check the rules)

- "just to see" → probably B4 or C5
- "while we're here" / "let's also test" → C2
- "the data looks interesting" → C2
- "loosen it a little" → C4 or F
- "one more re-run" → D4
- "it's only a small sample but" → C3
- "I'm sure it's fine, ship it tonight" → E6

## OPEN QUESTIONS (parked, with their decision path)

- Walk-back: substitute vs skip vs merge → `ob_walkback_depth` screen → generation 2.
- Doji floor level (20%): validated or not by the `ob_body_ratio` gradient → generation 2.
- Oversized cap (2×): `ob_range_atr` trend toward the cap → generation 2.
- Wider-stop replay, trailing stops, time-stop knob → v2 spec, fresh pre-registration.

## CHANGE LOG

- 2026-07-03 — file created. No rules changed yet.
- 2026-07-09 — added **F-BUCKET** (full per-bucket curve + named best/worst + per-bucket
  straight-to-SL rate mandatory in discovery). New rule, not a change to an existing one;
  motivated by the Step-2 miss where a single top-vs-bottom number hid the small-OB death
  cliff. Enforced in code + test, out-of-band. Full text in EDGE LAB spec §3.
