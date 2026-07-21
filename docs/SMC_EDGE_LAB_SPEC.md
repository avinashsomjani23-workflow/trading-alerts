# SMC EDGE LAB — v2 Research Engine Spec

**Status:** FROZEN design, pre-registered before any v2 code. Supersedes the analysis
layer of `EDGE_ENGINE_SPEC.md` (v1). Written on Opus for build on Fable/Opus.
**Author decision date:** 2026-07-07. **Rev 2026-07-08** — operating model, 10-pair
per-pair discovery, effect-floor correction, two-pass exit, six new columns folded in.

This file is the single source of truth for the new chat. It states exactly what is
KEPT, REPLACED, and ADDED versus the live `backtest/diagnostics/edge_engine.py`, so
there is no contradiction between old and new.

---

## 0. Why v2 (the problem being fixed)

The v1 engine fuses **discovery** and **validation** into one binary survivor gate. A
feature is "candidate" or "noise". Everything else — break quality, OB age, FVG size,
walkback, SL anatomy — is computed then buried as `noise`. Result: the trader sees ~1
fragile candidate (`alert_utc_hour`) and none of the rich structure.

Three concrete v1 defects, evidenced on run `h1only_20080102_20251231`:
1. **Ordinal signal killed by the wrong test.** `ob_walkback_depth` tested with
   Kruskal-Wallis (unordered) — loses monotonic power. *(Note: its steep discovery
   gradient does NOT persist in the fuller sample; the fix is a correct test AND the
   two-layer split, not shipping walkback.)*
2. **Non-linear signal invisible.** Continuous features summarised by top-vs-bottom
   quintile Δ → U-shapes (`pd_pct`, `ob_range_atr`, `leg_retrace_pct`) read as null.
3. **No interaction discovery.** Only 5 pre-registered pairs; "most edge is in
   combinations" is never explored.
4. **No pair-level view.** FX+Gold pooled; a signal real for one instrument and dead
   for another is invisible. 10 instruments, never sliced.

Plus gaps: no exit/MFE research surfaced, no probability model, no calibration, no
whole-search overfitting control (PBO/DSR), no purged CV.

---

## 0.5 HOW TO RUN THIS WITHOUT GETTING LOST (read this first)

This is a big spec. The single biggest risk is not a wrong stat — it is the builder
drowning in methods, chats, and numbers. This section is the map. Obey it and you
cannot get lost.

### Where things run — settle it now
- **ALL discovery / model / exit research runs LOCALLY, in a chat, on the ONE existing
  18-yr `trades.csv`.** Not on GitHub Actions. Discovery is interactive: you read a
  table, ask a question, slice again. CI is for hands-off deterministic jobs (the knob
  sweeps) — the opposite of discovery.
- **CI keeps doing only:** the structure-golden gate + regression tests. Never the
  research.
- **You do NOT re-run the backtest per feature.** Every discovery question is answered
  by slicing the one frozen CSV. You re-run the backtest ONLY when a **column** changes
  — which is why column additions are batched (see §12, the six new columns) into ONE
  run before discovery starts.

### The mental model (three objects, nothing else)
1. **One frozen CSV = ground truth.** `backtest/results/<run_id>/trades.csv`. Re-made
   only when features/columns change, as a batch. Discovery never mutates it.
2. **`edge_lab.py` reads that CSV → ranked tables + JSON.** Deterministic, seed 42.
3. **Two buckets for every finding:** (a) SHIP-GATE QUEUE (passed discovery, being
   proven), (b) "interesting, not proven" (visible, never shipped). Nothing floats
   between them.

### One rule that keeps chats sane
- **One build step (§11) = one chat.** Never carry two steps in one conversation.
- Each step has a written **entry contract** (what must already exist) and **exit
  contract** (what it must produce, as a stamped file). When a chat finishes, it writes
  its findings to a ledger file; the next chat **reads that file first.** Same handoff
  pattern already used across this project — now mandatory per step.

### What you hold in your head at any moment
Only three things: (a) the current ranked table, (b) the ship-gate queue, (c) which
§11 step you are on. If you are holding more, stop — you have drifted.

### The discovery phase specifically (the long, dangerous one)
- Discovery outputs **one ranked table**, not a pile of scripts. Every feature = one
  row, richest signal on top. You read top-to-bottom.
- Each row is promoted to the ship-gate queue OR left as "interesting, not proven."
  Two destinations. No third.
- Per-pair discovery (§4b) is a **separate pass** built on top of the pooled pass —
  never mixed into it. Pooled first, then per-pair.

---

## 1. Architecture decision — KEEP the spine, REPLACE the analysis layer

**DO NOT delete `edge_engine.py`.** Reuse its correct machinery. Build the v2 engine as
a new module (`backtest/diagnostics/edge_lab.py`) that imports the spine helpers.

### KEEP (import/reuse verbatim — these are correct and hard-won)
- `load_population` + the eligibility filter order (`eligible_for_headline` is the ONE
  population rule; proximal-only; drop NAS100).
- `SPLITS` (Discovery 2008–2016 / Validation 2017–2021 / Holdout 2022–2025), `WAR_START`,
  `_split_of`, `_book_of`, pooled FX+Gold logic, `pair="GOLD"` mapping.
- Stage-0 TRUST GATE (all checks), **gates-off proof**, census, `MIN_SPLIT_N`,
  exploratory auto-degrade.
- Bar cache (`_ensure_bars`) + **real-order walker replay** (`_replay_recipe`,
  `walk_multileg`) + baseline self-check. This is the C6-compliant exit engine.
- `bootstrap_ci`, `bootstrap_diff_ci`, `_pos_quarters`, `_cell_stats`, seed 42,
  byte-identical determinism, "no number without N + window + scope".

### REPLACE (the analysis layer)
- **Binary survivor framing → two layers:** permissive DISCOVERY scorecard (rank
  everything, kill nothing) + strict SHIP gate (unchanged rigor, applied only to things
  we would actually ship).
- **Kruskal for ordinals → Spearman trend test.** Ordinals get an order-aware test.
- **Top-vs-bottom quintile Δ → add Mutual Information + full bucket curve** so U-shapes
  and monotonic trends both surface.
- **Ridge-only EV model → baseline ladder + meta-label probability model + calibration**
  (see §6).
- **5 fixed interactions → interaction DISCOVERY** (trees + XGBoost + SHAP), gated by
  validation (see §5).

### ADD
- **Pair-level discovery** across all 10 instruments (§4b).
- Exit/management research track, two-pass, walled off from entry (§7).
- Purged & embargoed CV; PBO + Deflated Sharpe on the whole search (§8).
- Bayesian shrinkage on thin buckets (§3).

---

## 2. The two layers (the core fix)

### Layer 1 — DISCOVERY SCORECARD (permissive, hypotheses only, 2008–2016)
- Purpose: SEE everything. Rank, never kill.
- Per feature, on its valid subpopulation (never impute):
  - effect size (top-vs-bottom Δ **and** full bucket/level curve),
  - 95% bootstrap CI,
  - **Mutual Information** (catches non-linear/U-shaped signal a Δ misses),
  - correct significance test by type (§3),
  - per-quarter consistency,
  - **Bayesian-shrunk** bucket means (thin buckets pulled toward the pooled mean).
- Output: one ranked table, richest signal on top. FDR shown as an **informational**
  column, not a gate. Every row carries N + window + scope.

### Layer 2 — SHIP GATE (strict, unchanged rigor)
A signal may become **eligible to change live trading** ONLY if it passes ALL five
(plain-English form the trader eyeballs):
1. **Makes SMC sense** — a real market reason exists.
2. **Real in discovery (2008–2016)** — clear gap, enough trades, not a fluke
   (CI excludes 0; correct test significant).
3. **Repeats in validation (2017–2021)** — same direction, ≥60% of quarters agree.
   *(60% is the LOOSEST of the five — eyeball anything in the 60–70% band harder.)*
4. **Big enough** — |effect| ≥ the effect floor (§10; provisional 0.05R for single
   features, NO fixed floor on the interaction track).
5. **Survives holdout (2022–2025)** — checked ONCE, at the very end.

Pass all → **eligible to ship.** Fail any → kept visible as "interesting, not proven",
never shipped.

**The gate PROPOSES; the trader DISPOSES.** Passing all five makes a signal *eligible*.
**Nothing is auto-shipped or auto-gated into live trading without the trader's explicit
approval.** The five points are the bar a signal must clear to even be *offered*.

This preserves v1's `MIN_BUCKET_N=150`, `QUARTER_SIGN_FRAC=0.60`, `FDR_Q=0.10`. The gate
is NOT loosened. Only the discovery layer is permissive, and the effect floor is
corrected (§10, with reasoning).

---

## 3. Correct test per feature type (statistician's call — frozen)

- **Ordinal** (ordered categories: `ob_walkback_depth`, `ob_touches`, `bos_tier`):
  **Spearman** rank-correlation trend test — "does result move steadily as the level
  climbs?". Chosen over Kruskal (throws away order) and Jonckheere (heavier, no gain
  here). Robust, already used elsewhere in the engine.
- **Continuous** (`break_close_atr`, `fvg_size_atr`, `ob_age_h1_bars`, `sl_distance_atr`,
  …): **Spearman on raw values + Mutual Information + decile curve.** MI is the U-shape
  catcher. Report the full curve, not just Δ(top−bottom).
- **Nominal/unordered** (`pair`, `session`, `pd_zone`): **Kruskal-Wallis** across levels
  (correct here — no order to exploit).
- **Binary** (`fvg_present`, `dr_ceiling_broken_at_ob`, `trend_pd_agree`): two-sample
  bootstrap diff-CI.

**Applicability flags (be honest where a stat is stretched):**
- Bootstrap CI assumes trades are independent draws. Overlapping trades violate this →
  CIs are slightly optimistic. Mitigation: purged CV (§8) for anything that ships; report
  the caveat on discovery numbers.
- FDR assumes an enumerable, independent test family. Tree/XGBoost fishing is neither →
  **do not trust FDR for the interaction/model search; use PBO/DSR there instead.**
- Per-quarter consistency needs N≥30/quarter (`MIN_QUARTER_N`) or the quarter is not
  counted — keep this.
- MI is biased upward on small samples → only report MI where the feature's valid subpop
  ≥ 500; below that, mark "MI unreliable". **This will silence MI on many thin features
  (specific killzone hours, deep walkback) — expected, not a bug; the decile curve and
  Spearman still speak there.**

**BUCKET-REPORTING RULE (mandatory, enforced in code + test — the §0.3 lesson made hard).**
No feature verdict may rest on a single summary number — not top-vs-bottom Δ, not Spearman
ρ, not MI alone. Every feature MUST print its FULL per-bucket / per-level curve: each
bucket's N, win rate, mean R, and **straight-to-SL rate**, with the BEST bucket and the
WORST bucket named explicitly.
- *Why:* a monotonic Δ and a correlation both hide (a) a death cliff in one tail (e.g. the
  smallest `ob_range_atr` bucket dying far more while the rest look flat) AND (b) an edge
  that lives in only one bucket (e.g. only the largest FVGs win). A single number buries
  both. This is the exact miss that made the small-OB death cliff invisible in Step 2.
- *The per-bucket straight-to-SL rate is required* so every entry feature is always crossed
  against loser behaviour. This is a DESCRIPTIVE lens (legal on all trades, §7 Pass 1),
  NEVER an entry input — the look-ahead wall stands (outcome-time may never select entries).
- *Enforcement:* a scorecard row missing the full curve, the named best/worst buckets, or
  the per-bucket SL rate is a build error. Guarded by a test in the discovery module,
  out-of-band — never in the live alert path.

---

## 4. Univariate discovery (Track A) — pooled

- Run the Layer-1 scorecard (§2, §3) over every entry-legal feature on the pooled
  FX+Gold discovery population (2008–2016), gates-off + gated views (§9).
- Outcome-time columns are **excluded here** (they belong to the exit track, §7). The
  timing classifier enforces this (`outcome_time` class — see §12).

## 4b. Pair-level discovery (Track B) — all 10 instruments

**The instruments (verified from both backtest runs, 2026-07-08):**
`AUDUSD, BTCUSD, EURJPY, EURUSD, GBPUSD, GOLD, NZDUSD, USDCAD, USDCHF, USDJPY`.
(NAS100 already out of scope, A4.) BTCUSD is the crypto pair (weekend-block aware).

- **Every scorecard feature is ALSO sliced per-pair.** A signal real for USDJPY and dead
  for EURUSD is invisible when pooled — this pass surfaces it. Built as a **separate
  pass on top of the pooled scorecard**, never mixed into it.
- Report per-pair: N, effect, CI, per-quarter consistency, same as pooled. Thin pairs
  (BTCUSD ~1,564 trades; NZDUSD/AUDUSD ~2,800) will have **wide CIs and few quarters** —
  flag this loudly on every thin-pair row. A thin per-pair slice is a hypothesis, not a
  verdict.

**Pair ELIMINATION — the hard rule (do NOT eliminate on data alone):**
- A pair may be dropped from trade scope ONLY when **data AND an SMC reason agree**
  (project rule: data-vs-SMC — a number that fights or lacks methodology is a discussion
  point, never a conclusion).
- A pair looking bad in-sample is NOT sufficient. Likely non-pair causes: thin sample,
  one bad regime stretch, a detector quirk on that instrument.
- **Precedent:** NAS100 was dropped for a *reason*, not a number. Same bar for the other
  nine. If a pair looks bad, surface it as a DISCUSSION POINT with the candidate SMC
  cause named; the trader decides.
- Sample-size cuts both ways: a thin pair's weak numbers do not disprove a sound SMC
  signal, and do not by themselves prove the pair worthless.

---

## 5. Interaction & model discovery (Track C)

Search wide, ship narrow. **Run the full stack even if simpler methods already find an
edge** — a single-feature edge does not mean there is no additional edge hiding in
combinations. The baseline ladder (§6) protects *shipping*, not *searching*.

- **Interpretable rules:** decision trees, **≤3 conditions deep**, every leaf
  **≥150 discovery / ≥75 validation trades**. Human-readable ("if X and Y → expR Z").
- **XGBoost discovery (signal-finder, NOT production):** standard shallow-tree ensemble.
  Noise controls (ALL required):
  1. Purged & embargoed CV (§8).
  2. **Importance stability** — a feature must rank important across most folds AND years;
     one-fold importance = discarded.
  3. **SHAP** to read leanings (explainability).
  4. Every surfaced candidate must pass the §2 SHIP GATE on validation.
  5. Must **beat the logistic baseline out-of-sample**, else its complexity is noise.
  6. **PBO/DSR** on the whole search.
- **Production model is simple + explainable** (logistic or shallow tree). XGBoost never
  gets the live keys — it only nominates candidates.

---

## 6. Probability model + Expected Value (meta-labeling — the trader's goal)

- **Meta-labeling:** SMC rules keep deciding direction + which trade. The model outputs
  **P(win)** (and P(TP1), P(SL), expected R, confidence) **for this specific trade.** It
  grades, never overrides SMC.
- **Baseline ladder (must beat each rung to justify the next):** random → legacy score →
  logistic regression → shallow tree → (only if it clearly wins OOS) XGBoost.
- **Calibration (makes the % trustworthy):** when the model says 70%, ~70% must actually
  win. Verify with a reliability curve + Brier score. An uncalibrated % is a lie and must
  not size trades. Ship only calibrated probabilities.
- **EV engine:** convert calibrated probabilities × avg win / avg loss − costs → expected
  value per trade. **EV, not win rate, is the optimisation target** (matches `r_realised`
  source-of-truth). This is the "EV score" that replaces the legacy score.
- **Drift (later, Phase 14):** re-check calibration + EV over time; markets change and a
  live model can quietly decay. Build AFTER a stable model exists, not before.

---

## 7. Exit / management research (Track D — TWO-PASS, walled from entry)

- **The wall (non-negotiable):** exit features are OUTCOME-time; they may NEVER select
  entries (look-ahead). Enforced by the timing classifier (`outcome_time` class).

- **Two-pass structure (settles the "eliminated trades still teach exits" point):**
  - **Pass 1 — DESCRIPTIVE, on ALL trades (BEFORE any entry filter).** MFE/MAE
    distributions (winners vs losers), `r_capture_ratio`, time-to-TP1, SL-sweep anatomy —
    measured on *everything*, including trades an entry filter would later eliminate.
    This closes the blind spot: eliminated trades still carry information about how stops
    and targets behave, and description must not be blind to them.
  - **Pass 2 — RULE TUNING, on SURVIVING trades ONLY (AFTER the entry filter is frozen).**
    Actual exit rules (stop size, break-even, partial, trail) are tuned only on the
    distribution actually traded — a good entry filter changes the excursion profile, so
    tuning on the wrong distribution overfits. Sequencing: **freeze the entry filter
    first, THEN tune exits on survivors.**
  - The wall the two-pass split does NOT touch: an outcome-time feature can never pick an
    entry. Two-pass governs *which trades* the exit research looks at, never *which
    features* select entries.

- **SL-sweep, measured correctly (fixes the trader's flagged gap):**
  - `sl_swept_then_tp1` is a TOUCH check (a hint), not proof — it may itself be a
    spike-and-fade. Do NOT bank it (C6).
  - The six new columns (§12) let the wider-stop replay be *designed and sanity-checked
    from data*: `sl_wick_depth_atr` (how far the wick pierced),
    `sl_max_adverse_after_sweep_atr` (did it recover or keep going against us),
    `sl_recovered_to_entry` (scratched-not-won), `bars_sl_to_tp1_touch` (how long a wider
    stop must endure). **None of these conclude anything** — they make the replay honest.
  - Correct measurement: re-run stopped trades through the **walker with a wider stop**
    sized by the `sl_wick_depth_atr` distribution, and see if the REAL order wins. Report
    the net-R delta of "distal + k·ATR" stops vs baseline, per split, via `_replay_recipe`.
- **Rules tested via real-order replay only** (break-even, partial, trail). A touch is
  not an exit.

---

## 8. Anti-luck machinery (published methods, replaces "FDR-only")

- **Purged & embargoed CV** (López de Prado): when splitting by time, drop training
  trades whose life overlaps the test window, plus a small embargo buffer after it. Kills
  the overlap leak. Use for anything that ships and for all model CV.
- **PBO (Probability of Backtest Overfitting):** re-run the whole selection on many
  shuffled splits; measure how often the best in-sample pick is below-median OOS. High
  PBO = the search is picking noise. This is the whole-search scorekeeper FDR can't be.
- **Deflated Sharpe Ratio:** shrink the strategy Sharpe for number of trials tried +
  return skew/kurtosis. "Is the Sharpe real after the search."
- **Division of labour:** FDR → univariate scorecard only. PBO/DSR → interaction/model
  search verdict. Purged CV → every OOS estimate.

---

## 9. Views

Run discovery on BOTH populations, reported side by side:
- **Gates-off** (all scores) — the trust/collection view (per the gates-off purpose:
  collect everything, then filter). This is the required Stage-0-proven input.
- **Live-gated subset** (what current live filtering would actually trade) — may tell a
  different story; this is where a positive-expectancy subset would show.

---

## 10. The effect floor (Gate 4) — CORRECTED, with reasoning

v1 used `MIN_EFFECT_R = 0.10R` everywhere. That is too high, and the reasoning below
holds independently of any single prior result (an earlier "daily-bias slice" number
once cited for this is UNVERIFIED and must not be used).

- **What the floor is:** the minimum effect size worth shipping — a guard against tiny,
  fragile effects that won't survive real costs and slippage.
- **Why 0.10R is too high for single features:** SMC on H1 has small per-trade
  expectancy (fractions of R). A single feature that must move a bucket ≥0.10R is asked
  to do a lot; real single-feature edges in liquid FX are usually smaller.
- **Why not just pick 0.075R either:** it's a round number with no derivation — same
  problem as 0.10R, one notch lower. The honest floor is tied to **cost**: a signal must
  clear round-trip cost (spread + slippage, in R) by a margin, or it's not edge, it's
  paying the broker. The walker already models costs (§6, "− costs").
- **The rule:** `effect floor = measured per-trade cost in R + safety margin.` For
  liquid H1 FX that cost is small (likely well under 0.05R).
- **Provisional value until measured:** **0.05R for single features**, and **NO fixed
  floor on the interaction track** (combined small effects are the whole point of that
  track — gate interactions on OOS + PBO/DSR, not on effect size). Replace 0.05R with the
  measured `cost + margin` once the walker's per-trade cost is pulled.
- **This is stricter reasoning than 0.10 or 0.075** — the floor earns its value from
  measured cost instead of being picked. It is NOT a loosening to fit a prior result.

---

## 11. Build order (each step = one chat; entry/exit contracts stated)

Each step reads the prior step's stamped output file first, and writes its own. No step
carries two conversations. Deliverables mirror the v1 stage files (JSON + md), stamped
scope/window/N, deterministic (seed 42).

1. **Scaffold + manifest.**
   - *Entry:* the six new columns (§12) are in code + the next 18-yr run has been
     generated (the clean baseline CSV). TRUTH_LEDGER rows exist for all six.
   - *Do:* `edge_lab.py` imports the v1 spine (§1 KEEP). Add the `outcome_time` timing
     class. Add the schema/dtype guard on load (see §12 guard).
   - *Exit:* module loads the CSV, census + Stage-0 trust gate pass, gates-off proven.
2. **Univariate discovery — pooled (Track A, §4).**
   - *Entry:* step 1 done.
   - *Do:* Layer-1 scorecard, gates-off + gated views.
   - *Exit:* one ranked table (JSON + md), every feature a row, two-bucket disposition.
3. **Pair-level discovery (Track B, §4b).**
   - *Entry:* step 2 done (do not mix into it).
   - *Do:* every feature sliced across all 10 pairs; thin-pair flags; any pair-elimination
     candidate surfaced as a DISCUSSION POINT with an SMC cause.
   - *Exit:* per-pair ranked tables; a written pair-elimination discussion (no pair killed
     without trader sign-off on data + SMC).
4. **Interaction + XGBoost/SHAP discovery (Track C, §5).**
   - *Entry:* steps 2–3 done.
   - *Do:* trees, XGBoost, SHAP, purged CV, PBO/DSR. Full stack runs regardless of
     earlier finds.
   - *Exit:* candidate interactions with stability + PBO/DSR verdicts; each queued for the
     ship gate.
5. **Meta-label probability model + calibration + EV (§6).**
   - *Entry:* step 4 done.
   - *Do:* baseline ladder, reliability curve, Brier, EV engine. Ship only calibrated.
   - *Exit:* a calibrated EV model + its calibration evidence; the chosen production model
     (logistic/shallow tree).
6. **Exit track — two-pass (Track D, §7).**
   - *Entry:* the entry filter is frozen (needed for Pass 2). Pass 1 can start earlier
     since it's descriptive on all trades.
   - *Do:* Pass 1 descriptive (all trades) → Pass 2 rule tuning (survivors) → wider-stop
     replay sized from `sl_wick_depth_atr`.
   - *Exit:* exit-rule recommendations with real-order-replay net-R deltas per split.
7. **Final SHIP GATE + one-shot holdout (§2, §9).**
   - *Entry:* everything above frozen.
   - *Do:* apply the five-point gate; open holdout 2022–2025 **once**.
   - *Exit:* the shortlist of eligible signals, holdout results, handed to the trader for
     the final ship decision.

---

## 12. Feature manifest — the six columns folded in (SHIPPED 2026-07-08)

All six are live in `backtest/h1_only_simulator.py`, logged via `front_cols` in
`h1_only_reporting.py`, tested (`test_h1_only.py::test_edge_lab_columns`), and carry
TRUTH_LEDGER rows. **They appear only in the NEXT backtest run** — the current
`trades.csv` still holds the OLD pasted/corrupted versions; do NOT slice these from it.

**Three DERIVED-IN-CODE (replace ex-pasted, CSV-corrupted columns):**
| column | formula | timing class |
|---|---|---|
| `sl_distance_atr` | `\|entry − sl_initial\| / atr_at_ob` (uses sl_initial, not sl_raw) | entry-legal (fill) |
| `r_capture_ratio` | `r_realised / mfe_r`; None when mfe_r ≤ 0 | **outcome_time** |
| `trend_pd_agree` | with-H1-trend AND `pd_alignment=="aligned"` | entry-legal (alert) |

**Three NEW OUTCOME-TIME (exit track only, §7):**
| column | meaning |
|---|---|
| `sl_max_adverse_after_sweep_atr` | furthest run AGAINST us beyond the fired stop after a sweep, in ATR — recovered (small) vs kept-losing (large) |
| `bars_sl_to_tp1_touch` | 1-indexed H1 bars from stop bar to first TP1 touch (None if never) |
| `sl_recovered_to_entry` | after a sweep, did price return to entry/BE within lookback |

**Timing classifier — third class `outcome_time`.** `r_capture_ratio`, `mfe_r`, `mae_r`,
`sl_swept_then_tp1`, `sl_wick_depth_atr`, and the three new SL columns are all
`outcome_time`: usable ONLY in the exit track (§7), NEVER as entry features. Leakage if
used to select entries.

**`sweep_present` stays decreed out (A5).** The `sl_bar_was_sweep` / `sl_swept_then_tp1`
/ `sl_max_adverse_after_sweep_atr` / `sl_recovered_to_entry` family is SL-anatomy,
unrelated to the excluded entry sweep — allowed in the exit track.

**Guard (warranted — silent + corrupts conclusions):**
- *Failure mode:* a hand-pasted or misaligned column silently shifts the CSV; discovery
  analyses garbage and ships a fake edge (exactly what happened to the three ex-pasted
  columns — text bled into numeric fields via the quoted `killzone_windows` comma).
- *Guard:* `edge_lab.py`'s CSV loader asserts (a) the column set matches what the
  simulator emits, and (b) every numeric feature parses as numeric. Lives in the analysis
  loader — out-of-band, never in the live trade path.
- *Why it matters:* a corrupted column that *looks* numeric is invisible until it
  produces a wrong trading rule.

---

## 13. Guardrail reconciliation (so nothing contradicts the frozen rules)

- This is a **v2 / generation-2 redesign** with a **fresh holdout** — legal under B6/C2.
  It is NOT a mid-run tweak to the v1 engine.
- **C5 is sacred:** Tracks A/B/C/D and all discovery/validation run on **2008–2021 only.**
  Holdout **2022–2025 is opened ONCE, at the very end,** by the final gate. Nothing above
  touches it early.
- **C4/F respected:** the SHIP GATE is NOT loosened. Only a permissive DESCRIPTIVE layer
  is added (C3 allows "thin findings labelled and kept visible, never shipped"), and the
  effect floor is *re-derived from cost*, not loosened to fit a result (§10).
- **C6 preserved:** MFE / `sl_swept_then_tp1` / the new SL columns are hints; only walker
  replays count.
- **A4/A5 preserved:** NAS out; `sweep_present` decreed out.
- **Data-vs-SMC preserved:** pair elimination needs data AND an SMC reason (§4b).
- **B5 / Truth Ledger:** every v2 column used has a ledger row before the v2 run — the six
  new columns already do (2026-07-08).

---

## 14. Frozen parameters (pre-registered; do not renegotiate after seeing results)

| knob | value | note |
|---|---|---|
| splits | 2008–2016 / 2017–2021 / 2022–2025 | unchanged from v1 |
| MIN_BUCKET_N | 150 | unchanged |
| interaction leaf | ≥150 disc / ≥75 val | v2: looser than v1's 100, deliberate |
| tree depth (readable rules) | ≤3 conditions | overfitting guard is OOS, not depth |
| effect floor (single feature) | 0.05R provisional | replace with measured cost + margin (§10) |
| effect floor (interaction track) | none | gate on OOS + PBO/DSR, not effect size |
| QUARTER_SIGN_FRAC | 0.60 | unchanged; loosest gate — eyeball 60–70% harder |
| FDR_Q | 0.10 | univariate scorecard only |
| MI reliability floor | subpop ≥500 | else mark unreliable |
| ship metric | Expected Value (R) | not win rate |
| production model | logistic / shallow tree | XGBoost = discovery only |
| pairs | 10 (NAS out) | AUD/BTC/EURJPY/EUR/GBP/GOLD/NZD/CAD/CHF/JPY |
| ship authority | trader (gate = eligibility only) | nothing auto-ships |
| seed | 42 | determinism preserved |

---

## 15. Inherited findings (from the instant-death / breakeven investigations)

These are ALREADY established on discovery (2008–2016) and carry into v2. Do NOT
re-discover the closed ones; DO carry the open ones into the exit track.

### INHERITED & CLOSED (do not re-run as new)
- **Selection is the wrong lever for instant death.** Full blind sweep + RF/GBM +
  interactions + luck tests: **no filter built from logged pre-entry columns turns the
  kept set positive** on discovery. Honest null (C4). Source: `DISCOVERY_FINDINGS.md`
  Fact 6, N3 verdict.
- **Mechanism of instant death = 1-bar geometry.** 92.9% of instant deaths die within 1
  H1 bar of fill; median SL distance ≈ 1.1 ATR ≈ one H1 bar's range. Not a hidden regime
  — it's limit-at-zone-edge + ~1-bar stop + H1 discreteness. Death is *predictable*
  (AUC 0.64, stable 9/9 yrs) but the kept set never goes positive (barbell). Source:
  `DISCOVERY_FINDINGS.md` Fact 5.
- **OB-size gate alone — does NOT fix it** (barbell: small OBs are worst losers AND best
  wins, 2.06R vs 1.59R). All floors tested stay negative.
- **50%-zone deeper entry — REJECTED** (adverse selection: fills 100% of losers, ~22–43%
  of winners; the specific −0.427R number was a method artifact, retracted — the
  *mechanism* holds).
- **Volatility mechanism for the killzone effect — REJECTED** (inside-KZ is calmer, gap
  persists in both vol buckets).
- **Confirmation entry on H1 — NOT VIABLE** (real confirmation needs M5/M15; H1-only can't
  simulate it). The lever most likely to break the barbell, parked on **data
  availability**, not merit.

### INHERITED & OPEN (carry into the exit track, §7)
- **Wider-stop replay (top open lever by ceiling).** 53% of true losers' stop candles are
  sweeps (wick through, close back); 30% swept then touched TP1. Whether a wider stop
  wins depends ENTIRELY on how deep wicks pierce — shallow = transformative, deep =
  nothing. The six new columns (§12) now let this be sized and replayed. Source:
  `DISCOVERY_FINDINGS.md` Fact 7; `INSTANT_DEATH_DECISION_TREE.md`.
- **Break-even sweep track.** BE stops (16.7% of filled) park the stop at entry = the OB
  proximal edge = the most obvious retest level; 53.2% of BE stop-outs are sweeps, 34.1%
  swept then TP1. Candidate rules: delay BE arming (+1.25R/+1.5R), offset the BE stop by
  wick depth, or drop BE. Needs `sl_wick_depth_atr` (now shipped). Source:
  `BREAKEVEN_SWEEP_HANDOFF.md`. Both share ONE wider-stop replay engine — coordinate.
- **Killzone death effect** — real (outside-KZ beats inside-KZ 9/9 discovery years) but
  mechanism UNKNOWN and it fights ICT doctrine → stays a DISCUSSION POINT, not a scored
  filter, until mechanism is proven. Owns its own investigation.

### CEILING DISCIPLINE (C6)
Every "prize" number in those files (remove-25%-of-deaths = +$236k, wider-stop +5,218R,
BE-sweep +1,615R) is a **touch-based free-lunch ceiling, never bankable.** Only a
real-order replay counts. Any wider-stop win shrinks as the stop widens (every R multiple
shrinks): illustratively 2.0× risk ≈ edge gone. Prove with replay, never quote the
ceiling as a forecast.

---

## 16. Open questions (parked, with decision path — not blockers)

- Production model choice (logistic vs shallow tree) → decided by calibration + OOS EV.
- Wider-stop size k·ATR → swept in the exit track via replay, sized from the
  `sl_wick_depth_atr` distribution, not guessed.
- Effect floor exact value → derived from the walker's measured per-trade cost in R
  (§10); 0.05R provisional until then.
- Regime as a feature → re-tested fresh in discovery (NOT assumed dead from memory).
- Pair elimination → per-pair discovery surfaces candidates; each needs data + SMC sign
  -off (§4b), decided by the trader.
