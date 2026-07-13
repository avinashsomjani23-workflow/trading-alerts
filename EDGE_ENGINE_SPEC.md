# EDGE ENGINE — IMPLEMENTATION SPEC (v1)

**For the implementer (Opus): build exactly this. Every decision is made. Where the data must
decide, the experiment is specified. Do not re-derive, do not add features, do not skip gates.**

- Deliverable: `backtest/diagnostics/edge_engine.py` — one script, five stages (0–4), stage-gated.
- It is a READ-ONLY post-processor: input = one completed backtest run dir + the frozen MT5
  parquet cache. It never touches git, registry, live state, or `r_realised`.
- Companion docs: `EDGE_ENGINE_HANDOFF.md` (mission), `backtest/RECOMMENDATIONS.md` (methods),
  `TRUTH_LEDGER.md` (column trust). This spec supersedes the handoff where they conflict (§1).
- **Two workflows share this one engine (§14):** MAIN (the full 3-way-split verdict engine,
  Stages 0–4 below) and SHORT-RANGE (a curiosity/hypothesis tool — one pool, no split, soft
  floors). Stages 0–4 as written ARE the main workflow. Short-range is a thin mode on top (§15).
- **Activation is a GitHub Action, not a bare terminal command (§16).** The `python -m` CLI still
  exists (it is what the Action calls, and how it runs in-chat during iteration), but the user
  drives it from the Actions tab with date boxes + a mode selector, never a terminal.

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
- Stages 1–3 consume discovery+validation as specified; **ACTIVATION of Stage 1's validation
  half is gated by the §18 approval token** — see §18 (staged human review).

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
3. **Gates-off proof:** ≥ `MIN_BELOW_FLOOR_N` (default 50) eligible filled rows have `score`
   below the live floor (floor read from the same constant the run used; default 4). This
   tests PRESENCE of the sub-floor tail, not its share: a gated run holds ZERO sub-floor
   trades, so an absolute count proves the gate was off. A *fraction* threshold (the old
   ≥10%) wrongly failed detectors that emit few sub-floor setups — scores are not
   proportional to performance, so the size of the tail is not a trust signal, only its
   existence is. Fewer than the floor → the run was censored → FAIL "input is a gated run,
   engine is blind to half the answer". (Changed 2026-07-04 — see §17 change log.)
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
9. **Feature timing classes:** every §5.1 feature carries an alert-time/fill-time class per
   §5.1b; the hardcoded fill-time list matches §5.1b exactly. Unclassified feature → FAIL.

---

## 5. STAGE 1 — UNIVARIATE SCREEN

### 5.1 The feature list (exactly this; the engine's manifest)

- Continuous: `break_close_atr`, `break_body_atr`, `impulse_leg_atr`, `fvg_size_atr`,
  `ob_range_atr`, `atr_at_ob`, `pd_pct`, `reversal_pct`, `ob_age_h1_bars`, `ob_to_fill_hours`,
  `bars_break_to_pullback`, `bos_sequence_count`, `score` (legacy score, tested as a feature —
  expected noise, prove it), `alert_utc_hour` (treat as categorical via 4 session bins? NO —
  use `session` for that; screen `alert_utc_hour` as 24 categories, expect thin, fine),
  `ob_body_ratio` (A3 — the accepted OB candle's own body/range; doji floor at 0.20 already
  filters the tail, so the surviving gradient tests whether the floor level is validated).
- Categorical: `bos_tag`, `bos_tier`, `bos_verdict`, `event`, `reversed_from_extreme`,
  `fvg_present`, `fvg_mitigation`, `fvg_state`, `pd_zone`, `pd_alignment`, `session`,
  `ob_session`, `fill_session`, `killzone_alignment`, `ob_in_killzone`, `fill_in_killzone`,
  `trend_alignment`, `setup_badge`, `ob_touches`, `bias`, `pair`,
  `ob_walkback_depth` (A3 — raw integer skip-count logged; screened as levels-as-is like
  `ob_touches`, so deep levels (2, 3, …) auto-merge into `other` under the §5.2 discovery
  N<150 rule rather than a hard-coded `2+` bin — keeps the full gradient for the 18-yr run.
  Tests whether substitute-walk-back quality decays with depth, feeding the
  substitute-vs-skip-vs-merge decision).
- Excluded by decree: `sweep_present` (detector inverted, out-of-scope), all `news_*`
  (confounder check only, §5.5), NAS100 rows, `mfe_r`/`mae_r` (outcomes, peak law).
- Excluded as redundant-by-construction (underlying fact is already screened directly):
  `break_tier`/`break_excess` (graded/raw forms of `break_close_atr` — keep the ATR-normalised
  continuous), `structure_pts`/`fvg_pts`/`freshness_pts`/`killzone_pts`/`confluences_present`
  (legacy-score components; the composite `score` is screened once), `setup_badge_kind`
  (duplicate of `setup_badge`), `h1_trend`/`direction` (enter via `trend_alignment`/`bias`),
  `alert_seq` (constant 1 on traded rows — dedupe keeps first fire only).
- Excluded with a reason worth recording: `tp1_rr`/`tp2_rr` are entry-time-known but
  mechanically entangled with `r_realised` under tp1-based recipes (the win payoff IS tp1_rr
  by construction) — a screen against r_realised would flag payoff mechanics, not edge. If
  ever tested, test against WR via the logistic side only. v2 discussion, not in v1.
- Missing-value rule: `fvg_size_atr` is None when no FVG — screen it on the `fvg_present==True`
  subpopulation only. Same pattern for any None-by-construction feature (`pd_pct` when DR
  invalid → drop those rows from that feature's screen only). Never impute.

### 5.1b Feature timing class (stamped on every feature — decides where a survivor may ship)

- **ALERT-TIME** (known when the alert fires): everything in §5.1 EXCEPT the list below.
  Eligible for the Stage-2 EV model and Stage-4 alert-time gates.
- **FILL-TIME** (known only when/if the limit fills): `ob_to_fill_hours`,
  `bars_break_to_pullback`, `fill_session`, `fill_in_killzone`, `killzone_alignment`.
  These are NOT knowable at alert → they must NEVER enter the Stage-2 EV model (a live scorer
  could not compute them; using them there is a look-ahead leak into the entry score). They ARE
  screened in Stage 1 exactly like the rest; survivors route to the ORDER-RULE track (§8.1b),
  because live they are implementable as order-management rules (order TTL, arm-delay,
  cancel-outside-window), not as alert scores.
- Stage 0 addition (check 9): the engine asserts its fill-time list matches this section;
  any §5.1 feature not classified → FAIL (no silent class assignments).

### 5.1c Pre-registered hypothesis bins (frozen now, before any run — part of the family)

- **H-SNAPBACK (user hypothesis 2026-07-03):** immediate return to the OB after the break
  candle marks weak displacement → worse expR. Test: `bars_break_to_pullback` binned
  {1–2, 3–5, 6–12, >12} (in addition to its quintile screen; both enter the same BH-FDR
  family). SMC mechanism exists (no conviction behind the break), so a survivor here is a
  finding, not a fish. Expect collinearity with `break_body_atr`/`impulse_leg_atr` — if both
  survive, §6.2 VIF decides which carries the signal.
- CAVEAT for the 1–2 bin: the backtest alerts ~1 bar later than live (closed-bar vs live
  forming-bar proximity) — the fastest snapbacks may be under-sampled (alerted late or
  never_filled). Report the bin's N vs live-era alert data before trusting a null; a POSITIVE
  effect here is trustworthy, a "no effect on N=thin" is not.

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

- Stage-1 `survivor` features + `anatomy_promoted` features — **ALERT-TIME class only
  (§5.1b)**. Fill-time survivors never enter the model; they go to §8.1b. If the alert-time
  union is EMPTY → Stage 2 verdict `NO_ENTRY_SIGNAL`, emit the file with `pass: true` (an
  honest null is a pass — the engine continues to Stage 3 with fallback clusters).
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

### 7.6 Time-in-trade descriptives (REPORT ONLY — never a selection metric)

- Per cluster and for the winning recipe: p25/p50/p75 of `bars_to_exit` split by exit reason
  (tp/sl/be/forced), plus net expR of trades still open after {12, 24, 36} bars (from the
  replay — "what is a trade worth once it has dragged N bars").
- Purpose: the evidence base for a possible v2 time-stop knob. Outcome columns are legal here
  because Stage 3 is exit-side and this table selects nothing. The v1 grid stays frozen (§7.3);
  no time-stop ships from this table.

---

## 8. STAGE 4 — RECIPE SYNTHESIS + THE VERDICT

### 8.1 Combine (all choices made on discovery+validation, BEFORE holdout)

- **Pair set:** start = all 9 FX+Gold. Drop a pair only if BOTH discovery and validation show
  its expR CI entirely < 0 AND < 40% positive counted quarters. (BTC: never in the combined
  book; own section.)
- **Gates:** counter-PD-CHoCH skip — re-tested on 18-yr MT5: gate ships iff that cell's expR
  CI < 0 in discovery AND validation. Any §5.6 flagged interaction cell with CI < 0 in both
  splits may ship as an additional skip-gate (list them explicitly in the recipe).

### 8.1b Order-rule gates (from FILL-TIME survivors — §5.1b routing)

- A fill-time survivor whose bad bucket has expR CI < 0 in discovery AND validation may ship
  as an ORDER RULE, expressed live as order management: `ob_to_fill_hours > X` → order TTL of
  X hours; `bars_break_to_pullback ≤ 2` → don't arm the limit until bar 3 after the break;
  `fill_in_killzone == False` → cancel outside the window.
- In-engine test = row exclusion (drop trades the rule would have declined), same CI + quarter
  discipline as skip-gates. **Stated approximation:** exclusion assumes the declined trade
  never happens; live, price touching during a dead window and returning later would fill a
  later, different trade the backtest rows can't represent. Conservative for TTL/window rules;
  for arm-delay it slightly overstates the saving. Every shipped order rule carries this
  caveat + the affected N in the recipe JSON (`order_rules` list, same evidence schema as
  `gates`).
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
  "order_rules": [{"rule": "order_ttl_hours <= X", "shipped": false, "evidence": {...},
                    "approximation_caveat": "row-exclusion proxy, §8.1b"}],
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
  SCORE_FLOOR_LIVE=4 · MIN_BELOW_FLOOR_N=50 · MIN_SPLIT_N=500 · CLUSTER_MIN_N=300 ·
  WF_FOLDS as §8.2.

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

## 12. COLUMN COVERAGE MAP (every logged field has a disposition — nothing is silently ignored)

Where the confluence data actually gets used: §5.1 features → Stage 1 screen → Stage 2 EV
model → Stage 4 EV gate. The final recipe's `ev_model` block carries a coefficient per
surviving confluence — that IS the "best recipe from the data points".

| group | columns | disposition |
|---|---|---|
| Features | the §5.1 lists (37 fields: FVG size, break distance+displacement, OB age/freshness/range, OB walk-back geometry (`ob_body_ratio`/`ob_walkback_depth`, A3), fill timing, PD, sessions, killzones, structure tags, …) | Stage 1 screen → Stage 2 model → Stage 4 gate |
| Structure signals (STRUCTURE_SIGNALS_SPEC) | `structure_ranging_at_alert`, `flip_pending_at_alert`, `flip_pending_dir_at_alert`, `leg_retrace_pct_at_alert`, `dr_ceiling_broken_at_ob`, `dr_floor_broken_at_ob` | Stage 1 screen candidates (added to §5.1 manifest, edge_engine.py). No email/gate change until Stage 1 validates (SIGNAL_CANDIDATES rule) |
| Structure-signal support | `leg_extreme_at_alert`, `leg_extreme_clipped` | audit / derivation support for `leg_retrace_pct_at_alert`; NOT screened (kept out of the §5.1 manifest) |
| PD/PW liquidity pools (DAILY_BIAS_V4_SPEC §1.3, built 2026-07-13) | `day_state_at_alert`, `pdh_status_at_alert`, `pdl_status_at_alert`, `pwh_status_at_alert`, `pwl_status_at_alert`, `dist_next_pool_above_atr`, `dist_next_pool_below_atr`, `next_pool_above_tier`, `next_pool_below_tier`, `trade_toward_pool` | Stage 1 screen candidates, judged under the spec's three pre-registered hypotheses (SWEPT_* counter-trades / toward-unspent-weekly-pool / BOTH_SIDES underperformance) at F-rule thresholds. Kill rule: none survive validation → the daily layer dies in full, columns stay. No email/gate/score change until validated |
| PD/PW pool support | `last_sweep_age_h1`, `last_sweep_tier` | descriptive context for the sweep-recency read; screened only if a pool hypothesis survives (support, not a standalone candidate) |
| Outcomes | `r_realised`, `pnl_usd`, `exit_reason/price/ts`, `mfe_r`/`mae_r`, `bars_to_exit/tp1/tp2`, `sl_bar_was_sweep`, `sl_swept_then_tp1`, `be_arm_bar_touched_entry`, `r_if_exit_*`, `sl_collision` | targets + diagnostics (§5.5 anatomy, §7.6 time table); NEVER entry features (look-ahead) |
| Trade geometry | `entry`, `sl_raw/sl_initial`, `tp1/tp2`, `tp1_rr/tp2_rr` | define R and the replay; not features (§5.1 tp_rr note) |
| Identity / bookkeeping | `setup_id`, `alert_ts/alert_bar_ts/bos_timestamp/ob_timestamp`, `model`, `entry_zone` | keys; time fields enter as derived features (`session`, `ob_age_h1_bars`, …) |
| Population flags | `eligible_for_headline`, `headline_exclusion`, `ist_blocked`, `weekend_blocked`, `killzone_blocked`, `news_blocked` | row filters / confounder context (§3.1, §5.5); never features |
| Redundant | §5.1 redundant-by-construction list | excluded, reason recorded there |
| Decreed out | `sweep_*` (rebuild workstream), `news_event_*` (confounder only), NAS100 | §5.1 |

- Rule for the future: any NEW trades.csv feature column added by the logging workstream must
  be placed in this table (feature / outcome / flag / redundant) before a run feeds the engine;
  a column with no row here is a Stage-0 FAIL (extends check §4.1's spirit: no silent fields).

## 13. OUTER LOOP — DETECTION FEEDBACK PROTOCOL (the loop the engine cannot run itself)

The engine reads a finished run; it cannot tune detection. But its findings are allowed to
DRIVE detection changes — under these rules, or the anti-overfit discipline dies at the loop
boundary:

1. **Tagging:** Stage 1 stamps every survivor / anatomy-promoted feature with
   `actionable_at: "entry_gate" | "order_rule" | "detection"`. `detection` = exploiting it
   properly means changing what gets detected/alerted (e.g. a break-quality floor, an OB
   filter), not just declining alerts. The report lists the `detection` group in its own
   section — that list IS the "change detection, re-run" queue.
2. **Evidence rule:** a detection change may only be motivated by DISCOVERY + VALIDATION
   evidence. If the idea came from a holdout table, the change is tainted — holdout years can
   never seed changes that will later be validated on those same years.
3. **Re-run rule:** after a detection change → new baseline run → full engine re-run (Stage 0
   up). The new run's holdout verdict is legitimate only if rule 2 held. The engine stamps
   `generation: N` + `motivating_evidence: [...]` into `stage4_recipe.json` so the lineage is
   auditable.
4. **Generation budget:** every generation is another look at 2022–25. Cap: 3 generations on
   the current holdout window. Past that, the window is worn out — roll holdout forward (e.g.
   2023–26) and state it in the report. Track via the `generation` stamp.
5. **One change per generation** (or one pre-declared bundle): if generation N+1 changes three
   detection knobs at once and improves, nobody knows which knob did it — and knob-level
   attribution is exactly what the user needs for the next iteration.

---

## 14. RUN IDS + RUN SELECTION (fixes silent overwrite; decided 2026-07-03)

**The bug this fixes:** current run ID is `h1only_<start>_<end>` (`run_backtest.py:320`) — date
range only, no uniqueness. Re-running the same date range **overwrites the previous run's
folder** (verified on disk 2026-07-03: 49 folders, all date-range-only; e.g. four different 2008
runs would all collide on `h1only_20080102_20081231`). Silent data loss + no way to compare two
runs of the same years. This is a bug, not a feature.

### 14.1 New run-ID format

- **Drop the `h1only_` prefix** (redundant — every run is h1-only now).
- **Format:** `<start>_<end>__<runstamp>` where `runstamp = YYYYMMDD_HHMM` of when the run started
  (UTC). Example: `20080102_20081231__20260703_1425`.
- **Why timestamp, not a counter:** it self-sorts lexically by "most recent," which is exactly
  what the Option-A "latest wins" rule (§14.3) needs — no separate counter bookkeeping.
- **Change site:** `run_backtest.py:320`. This is a run-ID/logging change, NOT trading logic —
  but it touches `run_backtest.py`, so it ships with (a) the email-subject run-ID line (§14.2),
  (b) any registry/commit code that parses the old `h1only_` prefix. Grep for `h1only` before
  shipping; the run-ID string is consumed in `commit_logs.py`, registry build, and email subject.
- **No overwriting, ever:** every run gets its own folder locally. Old runs stay. Compare freely.

### 14.2 Run ID must appear everywhere it is consumed

- **Backtest email:** every backtest email carries its run ID (subject or first line). The user
  must always be able to tell which run an email came from.
- **Edge engine report:** `edge_engine_report.md` and `stage4_recipe.json` both name the exact
  run ID(s) they read (add `"input_run"` is already in the §8.4 schema — extend the report
  markdown header to print it prominently). Zero ambiguity about what was analysed.

### 14.3 Which run the engine reads (Option A — latest wins)

- The Action passes an explicit run dir (§16), so normal operation is unambiguous.
- **Default resolution when only a date range is given** (in-chat / CLI convenience): pick the
  folder matching that date range with the **newest runstamp**. Newest timestamp = latest run.
- **Override:** an exact `--run-dir` always wins over date-range resolution.
- The resolved run ID is echoed at the top of every stage's output and the report, so "which run"
  is answered in writing, not by folder-mtime guesswork.

---

## 15. SHORT-RANGE WORKFLOW (curiosity / hypothesis mode — decided 2026-07-03)

**Purpose:** a second, deliberately weaker tool for exploring short date ranges (the user has run
1-year and 2-year backtests). It exists to **generate hypotheses**, never to issue a verdict.

### 15.1 What it is NOT

- **NO discovery / validation / holdout split.** None. The 3-way split's only job is confirming a
  pattern survives on unseen data — a CONFIRMATION job. Short-range confirms nothing, so the
  split adds only complexity and empty buckets. (The proportional-split idea floated earlier was
  withdrawn — it was over-engineering.)
- **NO ship/don't-ship verdict.** No `ENTRY_AND_EXIT_EDGE` / `NO_EDGE` output. That verdict is the
  MAIN workflow's exclusive product.

### 15.2 What it IS

- **One pool.** Screen the entire short range as a single population (all eligible rows, §3.1
  filters still apply — NAS100 dropped, BTC separate, proximal-only, eligible-for-headline).
- **Reuses Stage 1 machinery** (univariate screen, §5) and MAY run Stage 3 exit replays (§7) as
  descriptives — but with the split-dependent steps removed (no "validation persistence", no
  "holdout look"). Stage 2 EV model and Stage 4 synthesis DO NOT run in this mode.
- **Soft floors — label, never abort.** MIN_BUCKET_N / MIN_CELL_N / MIN_QUARTER_N / MIN_SPLIT_N
  still compute and still TAG a cell `thin` when unmet, but in short-range mode a thin/low count
  **never stops the work and never fails a stage.** Stage 0's hard `MIN_SPLIT_N=500` FAIL (§4.4)
  is disabled in short-range; it becomes an informational census line instead.
- **Trade-count context (user-supplied, 2026-07-03):** ~1,200 eligible trades/year across the
  pooled pairs → a 2-year short run ≈ 2,400 pooled trades. At the POOL level that is plenty to
  screen; thinness only bites in rare sub-buckets (e.g. one session × one pair × top quintile),
  which is exactly what the soft `thin` tag is for.

### 15.3 Report language (mandatory — not optional phrasing)

- Every short-range report is stamped, top and bottom:
  `SHORT-RANGE MODE — EXPLORATORY. HYPOTHESES ONLY, NOT A SHIPPABLE VERDICT.`
- Every finding is worded as a hypothesis to test on full data, e.g. "worth testing on the full
  18-yr run", never "this is the edge". Any survivor here must go get confirmed by the MAIN
  workflow (full 3-way split) before it can become a rule — same discipline as §13.2.

### 15.4 Mode flag

- CLI/Action gains `--mode main | short_range` (default `main`). `short_range` selects the
  single-pool path above. The mode is stamped into every output file and the report header.

---

## 16. ACTIVATION + DELIVERY (GitHub Action, no terminal — decided 2026-07-03)

### 16.1 The engine core is unchanged

- Stages 0–4 stay a read-only library pointed at a run folder. The Action is a thin trigger
  wrapper on top — the smart part is untouched. `python -m backtest.diagnostics.edge_engine`
  still exists; it is what the Action invokes and how the engine runs in-chat during iteration.

### 16.2 GitHub Action — two buttons (workflow_dispatch inputs)

- **The user drives it from the Actions tab with input boxes, never a terminal:**
  - `start_date` (YYYY-MM-DD), `end_date` (YYYY-MM-DD)
  - `mode` = `main | short_range`
- **Button 1 — FULL RUN:** run the backtest for the given dates → then run the edge engine on
  that fresh run folder. Expensive (the backtest is the long pole; the engine is minutes). Use
  when fresh data is wanted.
- **Button 2 — ENGINE ONLY:** skip the backtest; point the engine at an existing committed run
  folder (resolved via §14.3 latest-wins, or an explicit run ID input). Cheap. Use when iterating
  on engine logic so Action minutes are not burned re-running the backtest.
- **Minutes note:** the private-account 2,000-min budget is spent almost entirely by the backtest,
  not the engine — hence the engine-only button. If it ever gets tight the user switches to the
  public account. (Do not build minute-optimisation beyond the two-button split without asking.)

### 16.3 Report delivery — all three, they do not conflict

1. **Committed report file:** `edge_engine_report.md` written into the run folder and committed
   (same `git add -f` path as other backtest logs, §backtest_logs_pipeline). Permanent, versioned,
   renders in GitHub, and readable directly in chat — this is the full dense detail.
2. **Email:** headline verdict + key numbers + the run ID land in the inbox (the "ping"). Summary
   only — the file carries the tables the email would truncate.
3. **Chat:** the committed markdown is read on request to discuss ("look at the last edge report").
- **Division of labour:** email carries the summary, the committed file carries the detail, chat
  is where the ship/don't-ship judgment happens over that file (handoff §7: script computes,
  human decides).

## 18. STAGED HUMAN REVIEW (discovery → approve → confirm → holdout)

Full implementer spec: `EDGE_ENGINE_STAGED_REVIEW_SPEC.md`. Summary:

**The three phases** (verdict scope only; exploratory/short-range is exempt — see below):

- **PHASE A — `--phase discovery`.** Runs Stage 0, then a **discovery-only** preview of Stage 1
  (`stage1_discovery`). The validation frame is NEVER built in this function, so it cannot leak.
  Output = **CANDIDATES** (verdicts: `candidate`, `candidate_thin`, `noise`, `thin` — the words
  `survivor` / `hypothesis` / `inverted` are impossible here). Prints + emails an **approval
  token** and stops. The human reads for as long as they want.
- **PHASE B — `--phase confirm`.** Canonical Stage 1, unchanged math (discovery + real validation,
  same `_apply_survivor_criteria`, same FDR), behind the **approval gate**. Output = **SURVIVORS**
  plus a **died-in-validation table** (candidates that did not repeat — the system working, not a
  bug). Logs the validation spend to the ledger and stamps it.
- **PHASE C — `--phase final`.** Stages 2, 3, 4. Holdout opens once, exactly as today.

**The token mechanism (the lock).** `token = sha256(discovery_sha + code_sha)[:12]`, where
`discovery_sha` is the sha256 of the token-less `stage1_discovery.json` and `code_sha` is the
sha256 of `edge_engine.py` on disk. Every discovery run mints a fresh token bound to that exact
output + that exact engine code. `--approve <token>` recomputes and, on a match, writes
`approval.json` (`consumed: false`). The gate at the top of canonical `stage1()` passes iff
`approval.json` exists, all three hashes match a fresh recompute, and `consumed == false`; on
pass it sets `consumed: true` (single-use). The refusal happens **before** the validation frame is
built.

**The ledger + stamps (the loud part).** Every validation activation appends one line to
`validation_ledger.jsonl` (append-only in code, committed by the Action → git preserves every
spend even if the file is later hand-edited). `stage1()` and `stage4()` gain `validation_runs: N`
and `validation_burned: bool` (true iff any line is a burn OR N > 1). When burned, `stage4()`
prepends a caveat and the report header shows a RE-OPENED block; confirm/final email subjects are
prefixed `[VALIDATION RE-RUN N]`.

**Re-opening.** `--force` does **not** bypass the gate. The only sanctioned re-open is
`--burn-validation "<non-empty reason>"`, which is by construction loud and permanently stamped.

**Honest limitation (§5.6).** The trader owns the machine: they can delete `approval.json` / the
ledger or edit the code. The mechanism makes silent burning **impossible to do accidentally and
impossible to hide from git history** — it does not make it physically impossible. Same trust
model as C5 (holdout) and D4 today.

**Exploratory / short-range exemption.** When Stage 0 stamps `scope: exploratory` (a split below
`MIN_SPLIT_N`), NONE of this applies: no gate, no token, no phases (`--phase` errors out). §15's
hypothesis language stays its own thing. The legacy full-run loop works exactly as before.

Structural guard: `tests/test_staged_review.py` (kills the class "validation spent without a ledger
line / stamp"). No new `trades.csv` columns; no live / simulator / exit changes.

## 17. CHANGE LOG

- **2026-07-04 — Staged human review (discovery → approve → confirm → holdout).** New §18. Splits
  the one-shot run into three human-gated phases so the trader is accustomed to the discovery data
  BEFORE validation runs, and a full participant before holdout. Discovery is read freely;
  validation activates only behind a single-use approval token and every spend is stamped forever
  (`validation_runs`, `validation_burned`) in an append-only ledger + git history. **Canonical
  Stage 1–4 math is untouched** — the only additions are the gate, the ledger, and the stamp
  fields; a passed gate yields byte-identical stage outputs (plus stamps). Why: trader
  participation + catching bugs before Stage 2 consumes survivors, without re-opening the
  iterate-until-it-passes door (D4). New files: `edge_email.py`, `.github/workflows/edge_engine.yml`,
  `tests/test_staged_review.py`. Guardrail impact: enforces D4; a **D5 draft** ("validation is
  opened by token, once") is prepared but NOT committed to `DECISION_GUARDRAILS.md` in this sitting
  (per that file's own change procedure — a rule may not be added in the sitting it first gates).

- **2026-07-04 — Gates-off proof: fraction → absolute count.** Check 3 (§4.3) changed from
  "≥10% of eligible filled rows below the live floor" to "≥`MIN_BELOW_FLOOR_N` (50) below the
  floor". Reason: the check exists to prove the run was executed gates-OFF (a gated run holds
  ZERO sub-floor trades). It was mis-specified as a *share* of the population. Scores are not
  proportional to performance, so the SIZE of the sub-floor tail is not a trust signal — only
  its EXISTENCE is. The 2010-2025 baseline run (`h1only_20100101_20251231`) has 2,023 sub-floor
  setups (6.7% overall, 8% of filled) — genuinely gates-off, but failed the old 10% fraction.
  Constant `MIN_BELOW_FLOOR_N=50` added to §10. Approved by the trader, who chose to make the
  fix in the same sitting the check was blocking (an explicit override of the "next-sitting"
  timing rule in DECISION_GUARDRAILS E6; the fix direction — presence not share — was decided
  before re-running).
