# EDGE ENGINE — IMPLEMENTATION SPEC (v1)

**For the implementer (Opus): build exactly this. Every decision is made. Where the data must
decide, the experiment is specified. Do not re-derive, do not add features, do not skip gates.**

- Deliverable: `backtest/diagnostics/edge_engine.py` — one script, five stages (0–4), stage-gated.
- It is a READ-ONLY post-processor: input = one completed backtest run dir + the frozen MT5
  parquet cache. It never touches git, registry, live state, or `r_realised`.
- Companion docs: `EDGE_ENGINE_HANDOFF.md` (mission), `backtest/RECOMMENDATIONS.md` (methods),
  `TRUTH_LEDGER.md` (column trust). This spec supersedes the handoff where they conflict (§1).

---

## 1. HANDOFF CORRECTIONS (verified against code 2026-07-03 — do not chase stale claims)

- **"Backtest models no spread" is HALF-WRONG.** SL is already spread-widened
  (`h1_only_simulator.py:515-543`; `sl_raw` vs `sl_initial` logged). TP fills are NOT
  spread-adjusted (`exit_engine.py` header: "TP levels are not widened"). → Cost model in §7
  charges spread on non-SL exits only. The SL side is pre-paid inside `r_distance`.
- **"exit_lab.py:109-114 neutralises persistence" is STALE.** Current `exit_lab.py` is a
  post-hoc replay script (reads `trades.csv` + frozen cache, self-checks baseline vs committed
  `r_realised`). No persistence to neutralise. The engine copies THIS pattern (§7.1). The
  separate in-run sink (`run_backtest.py:90-198`) stays untouched — it feeds the email only.
- **`atr_at_entry` does not exist as a column.** The logged column is `atr_at_ob`
  (`h1_only_simulator.py:1242`, formation ATR, frozen by design). Use `atr_at_ob`. If a true
  at-entry ATR is ever wanted, that is a logging change outside this engine.
- **`walk_multileg` supports legs + one shared break-even ONLY** (config keys: `legs`,
  `be_trigger_r`, `be_to_r` — `exit_engine.py:39-43`). No trailing stop exists. → Trailing is
  OUT of the v1 recipe grid (§7.3). Do not extend `exit_engine.py` in this build.
- **Verified as claimed:** `insights._spearmanr` (scipy, `insights.py:22`), `bootstrap_ci`
  (`:71`, seeded rng(42), returns `(None, None)` under n=5), `win_rate_pct` (`:43`, BE excluded),
  `is_peak_metric` / `verify_capturable` (`:708/:716`), `MAX_HOLD_H1_BARS = 48`
  (`h1_only_simulator.py:42`), all 11 pairs present in `config.json`.
- **No sklearn in this repo.** Regressions are numpy/scipy only (§6.2). Do not add dependencies.

---

## 2. ARCHITECTURE

```
input:  backtest/results/<run_id>/trades.csv   (the one 18-yr, gates-off baseline run)
        MT5 parquet cache (via backtest/data_loader.load_bars)
        config.json (spread_pips, pip sizes, pair symbols)

Stage 0  TRUST GATE      -> stage0_gate.json            (pass/fail; everything stops on fail)
Stage 1  UNIVARIATE      -> stage1_features.csv/.json   (survivors vs noise, ranked)
Stage 2  EV MODEL        -> stage2_model.json           (ridge coefs + calibration + verdict)
Stage 3  EXIT OPTIMISER  -> stage3_exits.csv/.json      (per-cluster winning recipe)
Stage 4  RECIPE + OOS    -> stage4_recipe.json          (THE deliverable) + edge_engine_report.md

output dir: backtest/results/<run_id>/edge_engine/
```

- CLI: `python -m backtest.diagnostics.edge_engine --run-dir <dir> [--stage N] [--force]`.
  Default runs all stages in order. Each stage writes `stageN_*.json` with `"pass": bool`;
  stage N+1 refuses to start unless stage N's file exists with `pass: true` (`--force` overrides,
  prints a red warning, stamps `"forced": true` into every downstream output).
- Determinism: every random draw seeded 42; all groupbys/iterations over sorted keys. Two runs
  on the same inputs must be byte-identical (except timestamps).
- Performance: load bars once per pair for the whole engine (exit_lab `_load_bars` pattern,
  window = trades' fill span + pads). ~40 recipes × ~15k trades × ≤50 bars ≈ minutes. Fine.

---

## 3. POPULATIONS AND SPLITS (fixed upfront — the anti-overfit backbone)

### 3.1 Row filters (in this order)

1. `eligible_for_headline == True` (kills never_filled / timeout / window_end / IST / weekend
   rows — the T2/D3 lesson; never re-derive eligibility yourself, the column is the one rule).
2. `entry_zone == "proximal"` (belt-and-braces; run should be proximal-only already).
3. Drop `pair == "NAS100"` entirely.
4. `BTCUSD` → separate bucket, never pooled (§3.3).

### 3.2 Time splits (by `alert_ts`, UTC)

| split | window | used for |
|---|---|---|
| DISCOVERY | 2008-01-01 .. 2016-12-31 | Stage 1 screening, Stage 2 fitting, Stage 3 recipe selection |
| VALIDATION | 2017-01-01 .. 2021-12-31 | Stage 1 survival confirm, Stage 2 calibration + threshold pick, Stage 3 recipe confirm |
| HOLDOUT | 2022-01-01 .. 2025-12-31 | Stage 4 ONLY — touched exactly once, by the final combined recipe |
| WAR | ≥ 2026-01-01 | never pooled; final recipe reported on it separately, label only |

- **Why three-way, not the handoff's two-way:** Stage 1 needs an unseen split to confirm
  survivors, but if that same split also validates the FINAL recipe, the verdict is contaminated
  by dozens of earlier looks. Discovery selects, validation confirms/selects hyper-choices,
  holdout is spent once on the finished recipe. This is the multiple-comparisons defence with
  teeth; per-quarter sign and min-N are the secondary guards.
- **Why contiguous-time, not odd/even years:** the question is "does the pattern persist into
  later, unseen regimes" — interleaved years leak regime information both ways.
- Per-quarter stats: a quarter counts only if it has ≥ 30 trades in the subset being measured;
  report `pos_quarters / counted_quarters`.

### 3.3 Books

- Pool all 9 FX+Gold pairs for screening and modelling (features are ATR-normalised — the
  handoff's own cross-instrument argument). Keep a `book` column (A = EURUSD NZDUSD USDJPY
  USDCHF XAUUSD; B = GBPUSD AUDUSD USDCAD EURJPY); every stage's output tables include per-book
  breakdown rows.
- BTCUSD: excluded from all pooled fits/screens. If its eligible N ≥ 300, run Stage 1 + Stage 3
  on it standalone and report in a `btc` section; otherwise report "BTC: N too thin" and stop.

### 3.4 Statistical conventions (used by every stage)

- expR = mean `r_realised` (or replayed net R). WR via `insights.win_rate_pct` (BE excluded).
- CI = `insights.bootstrap_ci` (10k, seed 42). `(None, None)` (n<5) → treat as THIN, never pass.
- MIN_N: a bucket/cell is testable at N ≥ 150 (pooled screen) or N ≥ 100 (interaction cell);
  below → verdict `thin`, never `survivor`.
- Peak-vs-fill law: `mfe_r`/`mae_r` are never an outcome target and never quoted as capturable.
  Stage 3 uses only real-order replays through `walk_multileg`. Any "reached X" narrative in the
  report must pass `insights.verify_capturable` or be omitted.

---

## 4. STAGE 0 — TRUST GATE (all checks must pass; each emits pass/fail + evidence)

1. **Columns exist:** every feature in §5.1 + outcomes (`r_realised`, `exit_reason`, `fill_ts`,
   `entry`, `sl_initial`, `tp1`, `eligible_for_headline`) present in trades.csv. Missing → FAIL
   with the list (means the logging workstream isn't done — stop, don't improvise).
2. **Ledger trust:** hardcode the §5.1 feature list (it encodes TRUTH_LEDGER verified/fixed
   status as of 2026-07-03). `sweep_present` must NOT appear in it (audit-only).
3. **Gates-off proof:** ≥ 10% of eligible filled rows have `score` below the live floor
   (floor read from the same constant the run used; default 4). Fewer → the run was censored
   → FAIL "input is a gated run, engine is blind to half the answer".
4. **Population census:** N by pair × year, per book, per split; war count. Emitted to the gate
   file. FAIL if any of DISCOVERY/VALIDATION/HOLDOUT has < 500 eligible trades total.
5. **Baseline exit self-check (the exit_lab pattern):** replay
   `{"legs": [(1.0, "tp1")], "be_trigger_r": 1.0, "be_to_r": 0.0}` over every eligible trade's
   post-fill bars from the frozen cache; require |mean(replay R) − mean(committed r_realised)|
   ≤ 0.01 AND per-trade |diff| > 0.02 on ≤ 1% of rows. FAIL otherwise — Stage 3 is worthless if
   the walker can't reproduce truth.
6. **News columns population:** report % non-null `news_event_title` on eligible rows. Not a
   pass/fail — but if 0%, stamp `news_usable: false` and Stage 1 skips the news confounder check.
7. **Duplicate/ordering sanity:** no duplicate `setup_id`; `alert_ts ≤ fill_ts ≤ exit_ts` on all
   eligible rows. FAIL on violation.
8. **BTC boundary note:** count BTC rows; stamp `btc_standalone: N ≥ 300`.

---

## 5. STAGE 1 — UNIVARIATE SCREEN

### 5.1 The feature list (exactly this; the engine's manifest)

- Continuous: `break_close_atr`, `break_body_atr`, `impulse_leg_atr`, `fvg_size_atr`,
  `ob_range_atr`, `atr_at_ob`, `pd_pct`, `reversal_pct`, `ob_age_h1_bars`, `ob_to_fill_hours`,
  `bars_break_to_pullback`, `bos_sequence_count`, `score` (legacy score, tested as a feature —
  expected noise, prove it), `alert_utc_hour` (treat as categorical via 4 session bins? NO —
  use `session` for that; screen `alert_utc_hour` as 24 categories, expect thin, fine).
- Categorical: `bos_tag`, `bos_tier`, `bos_verdict`, `event`, `reversed_from_extreme`,
  `fvg_present`, `fvg_mitigation`, `fvg_state`, `pd_zone`, `pd_alignment`, `session`,
  `ob_session`, `fill_session`, `killzone_alignment`, `ob_in_killzone`, `fill_in_killzone`,
  `trend_alignment`, `setup_badge`, `ob_touches`, `pair`.
- Excluded by decree: `sweep_present` (detector inverted, out-of-scope), all `news_*`
  (confounder check only, §5.5), NAS100 rows, `mfe_r`/`mae_r` (outcomes, peak law).
- Missing-value rule: `fvg_size_atr` is None when no FVG — screen it on the `fvg_present==True`
  subpopulation only. Same pattern for any None-by-construction feature (`pd_pct` when DR
  invalid → drop those rows from that feature's screen only). Never impute.

### 5.2 Bucketing

- Continuous → quintiles, **edges computed on DISCOVERY only**, then applied frozen to
  VALIDATION (fresh edges per split would leak and make buckets incomparable).
- Categorical → levels as-is; levels with discovery N < 150 merged into `other`.

### 5.3 Per-feature output (one row per feature × split × bucket)

- N, expR, bootstrap CI, WR, totR, pos_quarters/counted_quarters, per-book expR.
- Feature-level: Spearman(raw feature, r_realised) + p (continuous only, discovery and
  validation separately, via `insights._spearmanr`); top-vs-bottom-bucket ΔexpR + paired... not
  paired (different trades) — plain bootstrap CI on the difference of means (resample each
  bucket independently, 10k, seed 42).

### 5.4 Survivor criteria (ALL must hold — else `noise` / `thin` / `inverted`)

1. **Discovery signal:** Benjamini–Hochberg FDR at 10% across the full feature family on
   discovery p-values (Spearman p for continuous; Kruskal–Wallis p across buckets for
   categorical — scipy.stats.kruskal), AND top-vs-bottom ΔexpR CI excludes 0.
2. **Validation persistence:** same sign of top-vs-bottom ΔexpR, and per-quarter sign of the
   favoured bucket ≥ 60% positive-or-favoured in validation's counted quarters.
3. **Substance:** both extreme buckets N ≥ 150 in each split; |ΔexpR| ≥ 0.10R in discovery
   (below that, even a "real" effect isn't worth a rule on a ~breakeven system).
- Verdicts: `survivor`, `directional_thin` (passes 1, fails 2 or 3 on N only),
  `inverted` (validation sign flips), `noise`. Only `survivor` feeds Stage 2.
- Why BH-FDR on top of the split discipline: with ~35 features and thousands of rows, raw
  p-values go tiny for trivia; FDR prunes the family cheaply, but the REAL guard is criteria
  2–3. Rank the survivor table by validation ΔexpR, not by p.

### 5.5 Sub-screens (same machinery, different target)

- **Stop-out anatomy (the handoff's key split):** population = eligible SL exits. For every
  §5.1 feature, bucket and compare `P(sl_bar_was_sweep == False)` (clean-break rate) across
  buckets: rate, N, bootstrap CI on top-vs-bottom rate difference, validation persistence.
  Output table `stage1_sl_anatomy.csv`. Features that robustly predict clean-break stops are
  ENTRY-fault markers → auto-promoted as Stage-2 candidates even if they missed §5.4 (tag
  `anatomy_promoted`); features predicting sweep-stops are EXIT-fault markers → passed to
  Stage 3 as candidate cluster axes (tag only, no auto-action).
- **News confounder check** (only if `news_usable`): among clean-break SL exits, % within
  ±2h of a `news_event_ts` vs the same % among all eligible rows. Report only — context for
  reading the anatomy table, never a gate.
- **`sl_swept_then_tp1`:** report its rate per split as narrative. It is a HINT column
  (touch-based); the real answer is Stage 3's wider-stop… NOT in v1 grid (no stop-widening
  support in recipes — stop distance is set at entry, not a recipe knob). State in the report:
  "wider-stop replay = v2, requires re-simulating entries; out of scope here."

### 5.6 Two-way interactions

- Fixed, pre-registered list ONLY (no data-driven fishing): `pair × session`,
  `pd_zone × event`, `break_close_atr(quintile) × event`, `bos_verdict × bos_tier`,
  `killzone_alignment × session`.
- Cell rule: N ≥ 100, cell expR CI excludes 0 in discovery, same sign in validation.
- Output: flagged cells table. Interactions do NOT enter the Stage-2 model in v1 (keeps it
  interpretable); a flagged cell may become a discrete gate in Stage 4 (§8.2) if it also
  clears validation.

---

## 6. STAGE 2 — THE EV SCORE (multivariate)

### 6.1 Inputs

- Stage-1 `survivor` features + `anatomy_promoted` features. If the union is EMPTY → Stage 2
  verdict `NO_ENTRY_SIGNAL`, emit the file with `pass: true` (an honest null is a pass — the
  engine continues to Stage 3 with fallback clusters).
- PD status: `pd_alignment/pd_zone` enters as a candidate like any other. SEPARATELY, the
  proven-once counter-PD-CHoCH rule is re-tested as a standalone gate (§8.2) regardless of
  model fate.

### 6.2 Model (numpy/scipy only — no sklearn)

- **Primary: ridge regression** on `r_realised` (closed-form: `(XᵀX + λI)⁻¹Xᵀy`, intercept
  unpenalised). Continuous features standardised on discovery mean/std; categoricals one-hot
  (levels ≥ 5% discovery frequency, rest `other`, drop-first).
- λ from {0.01, 0.1, 1, 10, 100} by 5-fold **contiguous-time** CV inside discovery (folds =
  consecutive year blocks; random folds would leak regime).
- **Secondary: L2 logistic** (IRLS, ~30 lines, same λ grid) on win/loss (resolved trades only,
  BE excluded). Used only as a cross-check that the two models rank trades consistently
  (Spearman(ridge pred, logit pred) reported; < 0.5 → flag `models_disagree` in output).
- Why ridge over gradient boosting: ~≤15 features, interpretability is a hard requirement
  ("explain it to a vet"), collinearity (break_body/bos_tier/impulse_leg) is exactly what L2
  handles, and a black box can't be shipped into a live scorer anyway.
- **Collinearity hygiene:** VIF per feature (numpy); VIF > 5 → drop the member of the
  correlated group with the weaker Stage-1 validation ΔexpR (log what was dropped and why).
- **Sign sanity:** every coefficient's sign must match its Stage-1 bucket direction; a flipped
  sign after VIF hygiene → drop the feature (a sign you can't explain is a liability).

### 6.3 Calibration + pass bar (on VALIDATION)

- EV-score = ridge prediction. Decile table on validation: predicted-EV decile → N, realised
  expR, CI, WR.
- **Pass bar (all):** trade-level Spearman(EV, r_realised) ≥ 0.10 with p < 0.01 on validation;
  top-quintile realised expR CI lo > 0; top-quintile expR > population expR by ≥ 0.10R.
- Fail → verdict `NO_USABLE_EV` (pass:true, honest null). Pass → emit the model block
  (features, coefs, standardisation, λ) for Stage 4.
- Baseline comparison line (report only): same Spearman for the legacy `score` column — the
  "beats the old score" receipt.

---

## 7. STAGE 3 — SETUP-CONDITIONAL EXIT OPTIMISATION

### 7.1 Machinery

- Post-hoc replay, exit_lab pattern: one bar-load per pair from the frozen cache; for each
  eligible trade, `future = bars[pair][fill_ts:][: MAX_HOLD+2]`; every recipe runs through
  `exit_engine.walk_multileg` with `sim.MAX_HOLD_H1_BARS`, `sim.WEEKEND_FLAT`,
  `sim.WEEKEND_FLAT_HOUR_UTC`. Pessimism (SL-first, fill-bar TP suppressed) is inherited from
  the walker by construction — this IS the "uncapturable intrabar spike" discount.
- Population: all eligible trades (per §3.1). Replayed forced closes (`timeout` /
  `window_end` / `friday_flat`) stay IN, marked to their close price — dropping them would
  break pairing across recipes. Report per-recipe forced-close %, and a sensitivity table
  excluding the union of force-closed trades; if the winner changes in the sensitivity view,
  flag `forced_close_sensitive: true` and prefer the recipe that wins both views.

### 7.2 Cost model (the "net of spread" everywhere in Stages 3–4)

- `cost_r = (spread_pips × pip_size) / r_distance` per trade (crypto: spread read as dollars,
  pip_size 1.0 — mirror `h1_only_simulator.py:525-535`; read `spread_pips` from config.json).
- Charge per leg: leg net R = leg R − cost_r for every leg EXCEPT one that exited at the
  original `sl_initial` (reason `sl`, exit_price == sl_initial within 1e-9) — that fill
  pre-paid its spread via the widened stop. BE-stop exits, TP fills, market closes
  (timeout/window_end/friday_flat) all pay `cost_r`.
- All Stage-3/4 selection metrics are NET. Gross shown alongside in tables for transparency.

### 7.3 The recipe grid (FROZEN — pre-registered, never extended mid-run)

- Full-position: TP ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, "tp1"} × BE ∈ {None, 0.3, 0.5, 0.7,
  1.0}, `be_to_r` always 0.0, skipping combos where `be_trigger_r ≥ numeric TP` (BE can't arm
  before the target). 
- Partials (50/50): `[(0.5,1.0),(0.5,"tp1")]` × BE {None,0.5,1.0};
  `[(0.5,0.5),(0.5,1.5)]` × BE {None,0.3,0.5}; `[(0.5,1.0),(0.5,2.0)]` × BE {None,0.5,1.0};
  `[(0.5,0.5),(0.5,"tp1")]` × BE {None,0.3,0.5}.
- Plus baseline `[(1.0,"tp1")]` BE 1.0. Total ≈ 40. No trailing (§1). No stop-width changes
  (stop is an entry-time quantity; re-placing it means re-simulating fills — v2, own spec).

### 7.4 Clusters

- Primary (if Stage 2 passed): EV quintiles, edges from the DISCOVERY EV distribution, frozen.
- Fallback (if `NO_USABLE_EV`): `event` (BOS/CHoCH families via `bos_tag`) × `pd_zone` cells
  with N ≥ 300 in discovery; smaller cells merge into an `all_other` cluster.
- Always also compute the GLOBAL (un-clustered) sweep — the "one recipe for everything"
  candidate, and the yardstick that says whether per-cluster conditioning adds anything.

### 7.5 Selection + confirmation (per cluster; also for the global sweep)

- On DISCOVERY: for each recipe, per-trade **paired difference** vs baseline (same trades, so
  pair the diffs — this is the variance killer that makes ~40 comparisons survivable). Pick the
  recipe with max net expR whose paired-diff bootstrap CI (10k, seed 42) lo > 0.
- Tie-break (CIs overlapping the leader): fewer legs > no-BE > rounder numbers. Simplicity wins
  ties by decree.
- On VALIDATION: the ONE selected recipe per cluster gets a single confirmation look — paired
  diff vs baseline: same sign, per-quarter improvement sign ≥ 60% of counted quarters.
- Confirmed → cluster ships that recipe. Not confirmed (or nothing beat baseline on discovery)
  → cluster ships BASELINE, verdict `no_exit_improvement` for that cluster. Honest nulls
  propagate — never ship the second-best "because we did the work".
- Per-cluster conditioning is justified only if the confirmed per-cluster set beats the
  confirmed global recipe on validation (paired diff, CI lo > 0). Otherwise ship the global
  recipe alone — simpler is the default winner.

---

## 8. STAGE 4 — RECIPE SYNTHESIS + THE VERDICT

### 8.1 Combine (all choices made on discovery+validation, BEFORE holdout)

- **Pair set:** start = all 9 FX+Gold. Drop a pair only if BOTH discovery and validation show
  its expR CI entirely < 0 AND < 40% positive counted quarters. (BTC: never in the combined
  book; own section.)
- **Gates:** counter-PD-CHoCH skip — re-tested on 18-yr MT5: gate ships iff that cell's expR
  CI < 0 in discovery AND validation. Any §5.6 flagged interaction cell with CI < 0 in both
  splits may ship as an additional skip-gate (list them explicitly in the recipe).
- **EV threshold (only if Stage 2 passed):** candidates = discovery EV quintile edges
  {q20,q40,q60,q80} plus "no threshold". Pick the candidate with the highest validation net
  expR whose validation CI lo > 0 AND which keeps ≥ 150 validation trades/year equivalent…
  concretely: ≥ 30 trades/quarter average. None qualifies → no EV gate ships.
- **Exits:** Stage 3's confirmed output (per-cluster set or global).

### 8.2 The one holdout look

- Apply the frozen combined recipe to HOLDOUT (2022–2025). Report, net of cost: N, expR, CI,
  WR, totR, per-quarter signs, per-book, per-pair, max drawdown R, and the same for the
  BASELINE system on holdout (the "vs doing nothing new" line).
- **Robustness (computed, not selected on):**
  - Weekly-block bootstrap CI (resample ISO weeks with replacement, 10k, seed 42) alongside the
    iid CI — same-week trades are correlated (shared USD news); if the block CI flips the
    conclusion the iid CI gave, the block CI wins.
  - Expanding walk-forward, 3 folds: fit≤2015→test 2016-18, fit≤2018→test 2019-21,
    fit≤2021→test 2022-25 (re-run Stages 1–3 mechanically per fold with identical frozen
    rules). Edge labelled ROBUST if ≥ 2 of 3 fold tests are net-positive, else FRAGILE.
- WAR (2026): run the recipe, report separately, never pooled, one line in the verdict.

### 8.3 Verdict (mechanical decision tree — the script prints it, no judgment calls)

- ENTRY test: EV-gated holdout net expR CI lo > 0 (block CI) AND ≥ 60% positive quarters.
- EXIT test: chosen-exits-vs-baseline paired-diff on holdout: CI lo > 0 AND ≥ 60% positive
  quarters of the diff.
- Four outcomes: `ENTRY_AND_EXIT_EDGE`, `ENTRY_EDGE_ONLY`, `EXIT_EDGE_ONLY`
  ("entries are a coin-flip — stop pretending; ship exits"), `NO_EDGE` ("nothing survived OOS.
  Trade less or rebuild the entry basis. This verdict is trustworthy BECAUSE of the gates —
  killing it is the win."). Each outcome additionally stamped ROBUST/FRAGILE from walk-forward.

### 8.4 The recipe spec (stage4_recipe.json — the final deliverable schema)

```json
{
  "version": 1, "generated_utc": "...", "input_run": "<run_id>",
  "verdict": "ENTRY_AND_EXIT_EDGE | ENTRY_EDGE_ONLY | EXIT_EDGE_ONLY | NO_EDGE",
  "robustness": "ROBUST | FRAGILE",
  "pair_set": ["..."], "pairs_dropped": [{"pair": "...", "evidence": {...}}],
  "gates": [{"rule": "skip counter_pd_choch", "shipped": true, "evidence": {...}}],
  "ev_model": null | {"type": "ridge", "lambda": 1.0, "features": [...],
                       "standardization": {"feat": {"mean": 0, "std": 1}},
                       "coefficients": {"feat": 0.0}, "intercept": 0.0},
  "ev_threshold": null | {"value": 0.0, "discovery_quantile": "q60"},
  "exit_policy": {"mode": "global | per_cluster",
                   "clusters": [{"id": "...", "definition": "...",
                                  "recipe": {"legs": [[1.0, "tp1"]],
                                             "be_trigger_r": 0.5, "be_to_r": 0.0}}]},
  "cost_model": {"rule": "cost_r per non-initial-SL leg", "source": "config.json spread_pips"},
  "holdout": {"n": 0, "expR_net": 0.0, "ci_iid": [0,0], "ci_block": [0,0],
               "pos_quarters": "x/y", "vs_baseline_diff_ci": [0,0], "per_book": {}, "per_pair": {}},
  "walk_forward": [{"fold": "...", "test_expR_net": 0.0}],
  "war_2026": {"n": 0, "expR_net": 0.0, "note": "reported only, never pooled"},
  "btc": {"status": "standalone | too_thin", "detail": {}},
  "caveats": ["..."]
}
```

- Plus `edge_engine_report.md`: every stage's headline tables in markdown, dense, for pasting
  into chat. The judgment (ship / don't ship / discuss) happens in chat over this report —
  the script computes, humans decide (handoff §7).

---

## 9. ORDER OF OPERATIONS + TRUST CHAIN (what must be true before each stage is believed)

1. Logging workstream done → the 18-yr gates-off baseline run exists (post-detection-fix; all
   pre-fix runs are non-comparable per TRUTH_LEDGER).
2. Stage 0 green — else NOTHING downstream is real. Especially checks 3 (gates-off) and 5
   (walker reproduces `r_realised`).
3. Stage 1 → survivors may be EMPTY. That is a valid, publishable result; Stage 2 then emits
   its null and Stage 3 still runs (exits don't need entry survivors — fallback clusters).
4. Stage 2 → EV ships only past the §6.3 bar.
5. Stage 3 → recipes ship only past paired-CI + validation confirm; baseline is the default.
6. Stage 4 → holdout is opened ONCE. If anyone re-runs Stage 4 after changing upstream choices,
   holdout is burnt — the script stamps `holdout_opened_utc` into `stage4_recipe.json` and
   WARNS loudly on re-runs (`"holdout_reopened": true` + caveat auto-appended).

## 10. CONSTANTS (single block at the top of edge_engine.py)

- SPLITS as in §3.2 · MIN_BUCKET_N=150 · MIN_CELL_N=100 · MIN_QUARTER_N=30 ·
  MIN_EFFECT_R=0.10 · FDR_Q=0.10 · EV_SPEARMAN_FLOOR=0.10 · QUARTER_SIGN_FRAC=0.60 ·
  BOOT_N=10000 · SEED=42 · RIDGE_LAMBDAS=[0.01,0.1,1,10,100] · VIF_MAX=5.0 ·
  SCORE_FLOOR_LIVE=4 · MIN_SPLIT_N=500 · CLUSTER_MIN_N=300 · WF_FOLDS as §8.2.

## 11. WHAT THE ENGINE MUST NOT DO

- No new trades.csv columns, no writes outside `<run_dir>/edge_engine/` (truth-ledger gate
  does not apply — nothing here ships a column — but the report must label every number's
  population).
- Never quote `mfe_r`-derived numbers as capturable (peak law; `verify_capturable` or silence).
- Never extend the recipe grid, feature list, thresholds, or splits mid-run "because the data
  looks interesting" — that is the exact overfit this design exists to prevent. New ideas → a
  v2 spec with a fresh pre-registered grid, and holdout years it hasn't seen.
- Never pool WAR or BTC into anything.
- No sklearn, no new dependencies, no touching `exit_engine.py` / simulator / live code.
