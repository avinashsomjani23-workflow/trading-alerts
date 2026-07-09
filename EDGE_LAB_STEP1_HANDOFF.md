# Edge Lab v2 — Step 1 (Scaffold) HANDOFF

**Read this whole file before starting Step 2.** This is the stamped exit file for
Step 1 (spec §11). Step 2 (univariate pooled discovery) reads it first.

**Stamped:** 2026-07-08. **Built on:** Opus. **Spec:** `SMC_EDGE_LAB_SPEC.md` (frozen).

---

## What Step 1 built

- **`backtest/diagnostics/edge_lab.py`** — the v2 research module. It IMPORTS the v1
  spine from `edge_engine.py` (never copies it) and REPLACES the v1 analysis layer
  (none of v1's survivor/EV/recipe/interaction code lives in edge_lab.py).
- **`tests/test_edge_lab_scaffold.py`** — 11 guards, all passing. Runs standalone,
  no CSV on disk needed.

### What is IN edge_lab.py (Step 1 surface)
1. **Spine imports** (§1 KEEP): `SEED`, `load_population`, `stage0`, `resolve_run_dir`,
   `_now_utc`. The rest of the KEEP list (splits, stats, replay) is imported by the
   LATER step that first uses it — Step 1 carries ZERO dead imports (verified).
2. **v2 frozen params** (§14): `EFFECT_FLOOR_SINGLE_R=0.05` (provisional, replace with
   measured cost+margin), `EFFECT_FLOOR_INTERACTION_R=None`, interaction leaf mins
   (150 disc / 75 val), tree depth ≤3, `MI_RELIABILITY_MIN_N=500`, the 10 `PAIRS`.
3. **Three-class timing classifier** `classify_timing()`: `alert_time` / `fill_time`
   / `outcome_time`. This is the look-ahead WALL. outcome_time is NEVER entry-legal.
4. **`entry_features(df)` / `exit_features(df)`**: the Track-A/B entry universe vs the
   §7 exit-track universe. Verified: no overlap, no outcome-time leak, `sweep_present`
   decreed out.
5. **Schema/dtype load guard** (§12): `_run_schema_guard()` raises `SchemaGuardError`
   if (a) the CSV lacks the 7 baseline-signal columns (= it's the OLD/stale CSV) or
   (b) any numeric feature holds an unparseable string (= paste corruption). Lives in
   the analysis loader only — out-of-band, can never kill a live alert.
6. **`scaffold()`** + CLI: runs the guard, the spine's Stage-0 trust gate, and the
   timing-wall check; writes `<run_dir>/edge_lab/scaffold.json`; exit 0 on PASS, 1 on
   FAIL.

---

## Entry contract for Step 1 — STATUS

| requirement | status |
|---|---|
| six new columns in code (sim + reporting + test + ledger) | ✅ verified present |
| clean 18-yr run generated (the baseline CSV) | ⏳ **PENDING** — user started it 2026-07-08, ~5–6h |

**The clean baseline CSV does NOT exist yet.** The current
`backtest/results/h1only_20080102_20251231/trades.csv` is the OLD one (Jul 6, 106
cols, only 3 of 6 new columns — and those 3 are the corrupted-paste versions). The
schema guard **correctly REFUSES it** (proven: missing
`sl_wick_depth_atr`, `sl_max_adverse_after_sweep_atr`, `bars_sl_to_tp1_touch`,
`sl_recovered_to_entry`).

---

## Verification done (what was actually tested, not claimed)

- **Module imports, compiles, zero unused imports.** (ast check + py_compile.)
- **Classifier**: alert/fill/outcome all class correctly (unit tests).
- **Guard REJECTS the old CSV** (real old CSV, real run) — exit 1, clean abort, JSON
  written. This is the guard biting on the exact stale-CSV case.
- **Guard RAISES on a text value in a numeric column** (synthetic paste-corruption).
- **Guard TOLERATES None/blank/nan** in None-by-construction numeric columns.
- **Look-ahead wall**: 0 outcome-time features in the entry universe; 0 entry/exit
  overlap; `sweep_present` not entry-legal. (unit tests + live check on old CSV.)
- **Entry manifest = 44 features**, all legitimate SMC signals (no raw price levels
  leaked — those are in the corruption-guard set, not the screen set).

### Verification NOT yet complete (honest gap)
- **The scaffold GREEN path (guard PASS + trust gate PASS + wall intact) has not been
  confirmed on a real clean CSV**, because that CSV doesn't exist. A synthetic clean
  CSV (old CSV + 4 columns added) was run through `scaffold()` to exercise the spine's
  Stage-0 replay against REAL bars; that run is slow (~23k-trade replay) and was still
  running at handoff. **First action when the real run lands: run the scaffold on it
  and confirm PASS.** That signs the Step-1 exit contract.

---

## Step 1 EXIT CONTRACT — how to sign it (do this first in the Step-2 chat, once the run lands)

```
python -m backtest.diagnostics.edge_lab \
    --run-dir backtest/results/h1only_20080102_20251231 --stage scaffold
```
Expect: `[PASS]`, exit 0, `edge_lab/scaffold.json` with `"pass": true`,
`schema_guard.pass=true`, `trust_gate.pass=true`, `timing_wall.wall_intact=true`,
scope `verdict` (all splits ≥ MIN_SPLIT_N). ONLY after this is green does Step 2
analysis begin.

---

## Prerequisites flagged for LATER steps (not blockers now)

- **Step 4 (interaction/XGBoost/SHAP) needs `xgboost` installed** — NOT currently
  installed (`ModuleNotFoundError: xgboost`, verified 2026-07-08). `scipy`/`sklearn`
  are present. Install before Step 4; irrelevant to Steps 1–3.
- **Effect floor 0.05R is provisional** — replace with the walker's measured per-trade
  cost in R + margin (§10) when the exit track pulls cost.

---

## Guardrails honored (so nothing contradicts the frozen rules)

- **C5 sacred**: scaffold opens NO holdout. It only loads + trust-gates.
- **§1 KEEP/REPLACE**: spine imported, v1 analysis layer absent from edge_lab.py.
- **§12 guard**: warranted, out-of-band, tested. Full column-set match deliberately
  NOT asserted (brittle, no new bug class — reasoning in the code comment).
- **B5/Truth Ledger**: the six columns already carry ledger rows (2026-07-08). No new
  trades.csv column was added by Step 1 (analysis-only module).

---

## Next: Step 2 — Univariate pooled discovery (Track A, §4)

- Entry: Step 1 exit contract signed (above).
- Do: Layer-1 DISCOVERY scorecard on pooled FX+Gold, 2008–2016, gates-off + gated
  views. Per feature: effect size + full bucket curve, 95% bootstrap CI, Mutual
  Information (MI floor N≥500), correct test by type (Spearman ordinal/continuous,
  Kruskal nominal, diff-CI binary), per-quarter consistency, Bayesian-shrunk buckets.
  FDR is an INFORMATIONAL column, not a gate.
- Exit: ONE ranked table (JSON + md), every feature a row, two-bucket disposition
  (ship-gate queue vs "interesting, not proven"). Write `EDGE_LAB_STEP2_HANDOFF.md`.
- Reuse from spine for Step 2: `bootstrap_ci`, `bootstrap_diff_ci`, `_ci_excludes_zero`,
  `_pos_quarters`, `_cell_stats`, `benjamini_hochberg`, `pooled_fx_gold`, `split_frame`,
  `SPLITS`, `MIN_BUCKET_N`, `QUARTER_SIGN_FRAC`, `FDR_Q`. Add fresh: MI, decile curve,
  Bayesian shrinkage (NOT in the spine — v2 new).
