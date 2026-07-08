"""
EDGE LAB — v2 research engine (SMC_EDGE_LAB_SPEC.md).

Supersedes the ANALYSIS layer of edge_engine.py (v1). This module is the single
entry point for the v2 discovery/validation/exit research. It REUSES v1's correct,
hard-won spine (population loading, splits, trust gate, bootstrap, bar-cache +
real-order replay) by IMPORTING it — never by copying it — and REPLACES v1's binary
"survivor" analysis layer, which does not live here at all.

Design rules that never bend (from the spec):
  - ALL discovery/model/exit research runs LOCALLY on ONE frozen 18-yr trades.csv.
    This module reads that CSV; it never mutates it and never re-runs the backtest.
  - Seed 42, deterministic, byte-identical between runs (inherited from the spine).
  - No number leaves without N + window + scope (inherited from the spine).
  - C5 is sacred: discovery/validation touch 2008–2021 only. HOLDOUT 2022–2025 is
    opened exactly once, by the final ship gate — never by this scaffold.

THIS FILE = STEP 1 (scaffold + manifest) ONLY.
  It stands up the module, the v2 feature manifest with the THREE-class timing
  classifier (alert_time / fill_time / outcome_time), the schema/dtype load guard,
  and a trust-gate wrapper around the spine's Stage 0. Steps 2–7 (univariate,
  pair-level, interaction/model, EV, exit track, final gate) are separate chats and
  are NOT implemented here.

CLI:
    python -m backtest.diagnostics.edge_lab --run-dir <dir> [--stage scaffold]
    python -m backtest.diagnostics.edge_lab --start 2008-01-02 --end 2025-12-31

What v2 does NOT import from v1 (the REPLACED analysis layer — deliberately absent):
    survivor framing, Kruskal-for-ordinals, top-vs-bottom quintile-Δ screens, ridge
    EV model, fixed pre-registered interactions, the approval-token machinery, and
    all of stage1/2/3/4. None of that contaminates v2. The v2 analysis is written
    fresh in later steps (two-layer discovery + strict ship gate + MI/Spearman + PBO
    /DSR + purged CV + meta-label EV), per the spec.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

import pandas as pd

# ── The v1 SPINE (SMC_EDGE_LAB_SPEC §1 KEEP). Import, never copy. ────────────
# These are the correct, hard-won machinery pieces. If any of them changes in
# edge_engine.py, v2 inherits the change — one implementation, one truth.
#
# Step 1 (scaffold) uses only the names below. The rest of the KEEP list —
# SPLITS/WAR_START/_split_of/_book_of (splits), pooled_fx_gold/split_frame
# (population views), bootstrap_ci/bootstrap_diff_ci/_ci_excludes_zero/
# benjamini_hochberg/_pos_quarters/_cell_stats (stats), _replay_recipe/_ensure_bars
# (exit-track replay) — are imported by the LATER steps that use them, not here, so
# this module never carries a dead import. All remain reachable via `spine.<name>`.
from backtest.diagnostics import edge_engine as spine
from backtest.diagnostics.edge_engine import (
    SEED,
    load_population,
    stage0 as _spine_stage0,
    resolve_run_dir,
    _now_utc,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "backtest", "results")


# ═══════════════════════════════════════════════════════════════════════════
# V2 FROZEN PARAMETERS (SMC_EDGE_LAB_SPEC §14). Pre-registered — do not
# renegotiate after seeing results. Values that are UNCHANGED from v1 are
# imported from the spine above (MIN_BUCKET_N, MIN_SPLIT_N, QUARTER_SIGN_FRAC,
# FDR_Q, SPLITS, seed). Values v2 CHANGES or ADDS live here, with the reasoning.
# ═══════════════════════════════════════════════════════════════════════════

# Effect floor CORRECTED (§10). v1 used 0.10R everywhere; v2 ties the single-
# feature floor to measured cost + margin. 0.05R is PROVISIONAL until the walker's
# per-trade cost in R is pulled; replace then. The interaction track has NO fixed
# floor (combined small effects are its whole point — gate on OOS + PBO/DSR).
EFFECT_FLOOR_SINGLE_R = 0.05          # provisional; replace with measured cost+margin
EFFECT_FLOOR_INTERACTION_R = None     # none — gate on OOS + PBO/DSR, not effect size

# Interaction-tree leaf minimums (§14): v2 is DELIBERATELY looser than v1's 100.
INTERACTION_LEAF_MIN_DISC = 150
INTERACTION_LEAF_MIN_VAL = 75
INTERACTION_TREE_MAX_DEPTH = 3        # "≤3 conditions" readable rules

# Mutual-Information reliability floor (§3, §14): MI is upward-biased on small n.
# Below this valid-subpop size, MI is reported as "unreliable" (Spearman + the
# decile curve still speak). Expected to silence MI on thin features — not a bug.
MI_RELIABILITY_MIN_N = 500

# The 10 in-scope instruments (§4b, §14). NAS100 is out of scope (project A4) and
# is dropped by the spine's load_population. Verified against both backtest runs
# 2026-07-08. BTCUSD is the crypto pair (weekend-block aware; thin ~1,564 trades).
PAIRS = [
    "AUDUSD", "BTCUSD", "EURJPY", "EURUSD", "GBPUSD",
    "GOLD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]

# The clean 18-yr baseline this whole spec runs on (§0.5). Discovery never slices
# any other CSV. Named here so every step resolves the same run.
BASELINE_RUN_ID = "h1only_20080102_20251231"


# ═══════════════════════════════════════════════════════════════════════════
# V2 FEATURE MANIFEST + THREE-CLASS TIMING CLASSIFIER (§4, §7, §12).
#
# The timing class decides WHERE a feature may be used, and is the wall that
# stops look-ahead:
#   alert_time   — known at signal time. Entry-legal: may select entries.
#   fill_time    — known only if/when the limit fills. Entry-legal at FILL, but a
#                  live *pre-fill* scorer cannot see it; routed to the order-rule
#                  track, never the pre-entry EV model (same rule as v1).
#   outcome_time — known only after the trade resolves. EXIT TRACK ONLY (§7).
#                  Using one to select entries is look-ahead. Hard-walled.
#
# This is the v1 classifier EXTENDED with the third class. v1 had only
# alert_time/fill_time; v2 adds outcome_time for the six new columns + the
# pre-existing MFE/MAE/sweep family, so the exit track can be walled off cleanly.
# ═══════════════════════════════════════════════════════════════════════════

# OUTCOME-TIME (§12 + spec §7): usable ONLY in the exit track, NEVER as entry
# features. Leakage if used to select entries. This set is the wall.
OUTCOME_TIME_FEATURES = {
    # pre-existing outcome-time columns (were 'noise' in v1, now correctly classed)
    "mfe_r", "mae_r", "sl_bar_was_sweep", "sl_swept_then_tp1",
    "sl_wick_depth_atr", "r_capture_ratio",
    # three NEW outcome-time columns shipped 2026-07-08 (§12)
    "sl_max_adverse_after_sweep_atr", "bars_sl_to_tp1_touch",
    "sl_recovered_to_entry",
    # bars-to-* are all measured after the trade runs → outcome-time
    "bars_to_exit", "bars_to_tp1", "bars_to_tp2",
    "exit_reason", "exit_price", "exit_ts",
    "r_realised", "r_if_exit_tp1", "r_if_exit_tp2", "pnl_usd",
}

# FILL-TIME (§12; same set as v1's FILL_TIME_FEATURES + fill geometry): known only
# at/after fill. Entry-legal at fill, but never a pre-fill EV-model input.
FILL_TIME_FEATURES = {
    "ob_to_fill_hours", "bars_break_to_pullback", "fill_session",
    "fill_in_killzone", "killzone_alignment",
    "sl_distance_atr",  # |entry − sl_initial|/atr_at_ob — known once the fill sets entry
    "fill_ts",
}


def classify_timing(feat: str) -> str:
    """Three-class timing (§12). outcome_time wins over fill_time wins over the
    default alert_time. This is the single source of truth for 'may this feature
    pick an entry?' — only alert_time and (at fill) fill_time may; outcome_time
    never may."""
    if feat in OUTCOME_TIME_FEATURES:
        return "outcome_time"
    if feat in FILL_TIME_FEATURES:
        return "fill_time"
    return "alert_time"


# ── The entry-legal feature manifest (Track A/B input, §4). ─────────────────
# These are ALERT-TIME or FILL-TIME features that MAY drive entry discovery. The
# manifest is inherited from v1's ALL_FEATURES (the vetted, TRUTH_LEDGER-mapped
# set) MINUS anything the spec decrees out, PLUS the two entry-legal new columns.
# Outcome-time columns are excluded here BY CONSTRUCTION (they live in the exit
# track). sweep_present stays decreed out (A5, §12).
_V1_ENTRY_MANIFEST = list(spine.ALL_FEATURES)  # v1's vetted feature list

# The entry-legal additions from §12 that are NOT already in v1's manifest.
_NEW_ENTRY_LEGAL = ["trend_pd_agree"]  # alert-time (with-H1-trend AND pd aligned)

DECREED_OUT = {"sweep_present"}  # (A5) — must never be screened as an entry feature


def entry_features(df: pd.DataFrame) -> List[str]:
    """The entry-legal feature list actually present in this CSV, with outcome-time
    columns and decreed-out columns removed. Sorted, deterministic. This is the
    Track A/B universe; the exit track (§7) uses OUTCOME_TIME_FEATURES instead."""
    universe = list(dict.fromkeys(_V1_ENTRY_MANIFEST + _NEW_ENTRY_LEGAL))
    out = [
        f for f in universe
        if f in df.columns
        and f not in DECREED_OUT
        and classify_timing(f) in ("alert_time", "fill_time")
    ]
    return sorted(out)


def exit_features(df: pd.DataFrame) -> List[str]:
    """Outcome-time features present in this CSV (§7 exit-track universe). Never
    entry-legal. Kept separate so no code path can hand one to an entry screen."""
    return sorted(f for f in OUTCOME_TIME_FEATURES if f in df.columns)


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA / DTYPE LOAD GUARD (§12 — WARRANTED guard, out-of-band).
#
# Failure mode: a hand-pasted or misaligned column silently shifts the CSV;
#   discovery then analyses garbage and could ship a fake edge. This exact thing
#   happened to the three ex-pasted columns (text bled into numeric fields via the
#   quoted killzone_windows comma).
# Guard: on load, assert (a) the column set matches what the simulator emits, and
#   (b) every numeric feature parses as numeric. Lives ONLY in this analysis loader
#   — out-of-band, never in the live trade path, so it can never kill a real alert.
# Why it matters: a corrupted column that *looks* numeric is invisible until it
#   produces a wrong trading rule.
# ═══════════════════════════════════════════════════════════════════════════

# Numeric features whose values MUST parse as numbers. If a non-numeric string is
# present in any of these (past the pandas NA/None sentinels), the CSV is corrupt
# and the guard raises — better a loud stop here than a silent fake edge. Built
# from the entry-legal continuous set + the numeric outcome-time columns.
_NUMERIC_MUST_PARSE = sorted(set(spine.CONTINUOUS_FEATURES) | {
    "r_realised", "mfe_r", "mae_r", "sl_distance_atr", "r_capture_ratio",
    "sl_wick_depth_atr", "sl_max_adverse_after_sweep_atr", "bars_sl_to_tp1_touch",
    "atr_at_ob", "entry", "sl_initial", "tp1",
})

# The six new columns (§12) that PROVE this is the clean post-2026-07-08 baseline,
# not the old corrupted CSV. Their presence is the entry-contract signal.
SIX_NEW_COLUMNS = [
    "sl_distance_atr", "r_capture_ratio", "trend_pd_agree",       # 3 derived-in-code
    "sl_max_adverse_after_sweep_atr", "bars_sl_to_tp1_touch",     # 2 of 3 new outcome
    "sl_recovered_to_entry",                                      # 3rd new outcome
]
# sl_wick_depth_atr is the 7th sizing column (shipped alongside); require it too.
_BASELINE_SIGNAL_COLUMNS = SIX_NEW_COLUMNS + ["sl_wick_depth_atr"]


# NOTE on the §12 "column set matches the simulator" clause: we deliberately do
# NOT assert the FULL column list. front_cols is a local inside the reporter (not
# an importable value), its order varies, and optional columns come and go — a full
# set-match would be brittle without catching a new bug class. The real silent-
# failure class (paste corruption shifting the file, or a stale pre-2026-07-08 CSV)
# is caught by the two checks below: baseline-signal columns present + every numeric
# feature parses. That is the cheapest guard that actually bites.

class SchemaGuardError(RuntimeError):
    """Raised when the loaded CSV fails the schema/dtype guard. A hard stop — the
    analysis must not run on a corrupt or stale CSV."""


def _run_schema_guard(df: pd.DataFrame, run_id: str) -> Dict[str, Any]:
    """Assert the CSV is the clean, current baseline and every numeric feature
    parses. Raises SchemaGuardError on failure. Returns an evidence dict on pass."""
    evidence: Dict[str, Any] = {"run_id": run_id, "n_rows": int(len(df)),
                                "n_cols": int(df.shape[1])}

    # (1) Baseline-signal columns present → this is the post-2026-07-08 clean run,
    #     not the old corrupted CSV (§12: the six new cols "appear only in the NEXT
    #     backtest run"). A missing one = stale/old CSV = refuse.
    missing = [c for c in _BASELINE_SIGNAL_COLUMNS if c not in df.columns]
    evidence["baseline_signal_columns"] = _BASELINE_SIGNAL_COLUMNS
    evidence["missing_baseline_columns"] = missing
    if missing:
        raise SchemaGuardError(
            f"CSV is missing baseline columns {missing} — this is the OLD/corrupted "
            f"trades.csv, not the clean post-2026-07-08 run. Regenerate the 18-yr "
            f"backtest before discovery (spec §0.5, §12)."
        )

    # (2) Every numeric feature must parse as numeric. A corrupted paste leaves a
    #     stray string in a numeric column; coercion to NaN of a value that is NOT
    #     a recognised NA token = corruption. We compare pre/post-coercion non-null
    #     counts on the raw string view.
    bad_numeric: Dict[str, Any] = {}
    NA_TOKENS = {"", "nan", "none", "na", "null", "<na>"}
    for col in _NUMERIC_MUST_PARSE:
        if col not in df.columns:
            continue
        raw = df[col]
        coerced = pd.to_numeric(raw, errors="coerce")
        # rows that were non-null strings but became NaN AND are not NA tokens
        raw_str = raw.astype(str).str.strip().str.lower()
        looked_present = ~raw_str.isin(NA_TOKENS) & raw.notna()
        became_nan = coerced.isna()
        corrupt_mask = looked_present & became_nan
        n_corrupt = int(corrupt_mask.sum())
        if n_corrupt > 0:
            sample = raw[corrupt_mask].astype(str).head(3).tolist()
            bad_numeric[col] = {"n_unparseable": n_corrupt, "sample": sample}
    evidence["unparseable_numeric_columns"] = bad_numeric
    if bad_numeric:
        raise SchemaGuardError(
            f"Numeric columns hold unparseable strings (CSV corruption): "
            f"{bad_numeric}. Refusing to analyse a shifted/pasted CSV (spec §12 guard)."
        )

    evidence["pass"] = True
    return evidence


def load_lab_population(run_dir: str):
    """Load the frozen CSV through the v1 spine's population loader (the ONE
    eligibility rule + proximal-only + drop-NAS + split/book stamps), then run the
    v2 schema/dtype guard on top. Returns (eligible_guarded_population, guard_evidence).

    The guard runs on the RAW CSV columns (before eligibility filtering) so a
    corrupted column is caught even if every corrupt row is later filtered out —
    corruption anywhere means the paste shifted the file. The raw CSV is read ONCE
    here for the guard; the spine loader then re-reads to apply eligibility (its own
    read is unavoidable without changing the spine, and this is a local research
    load, not a hot path)."""
    trades_p = os.path.join(run_dir, "trades.csv")
    if not os.path.exists(trades_p):
        raise SystemExit(f"no trades.csv at {trades_p} — run the backtest first")
    raw = pd.read_csv(trades_p, low_memory=False)
    guard = _run_schema_guard(raw, os.path.basename(run_dir))  # raises on corruption
    return load_population(run_dir), guard                     # spine: eligible pop


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: SCAFFOLD (Step 1 exit contract, §11).
# Exit contract: module loads the CSV, census + Stage-0 trust gate pass,
# gates-off proven. We reuse the spine's stage0 verbatim (it already runs census,
# gates-off proof, baseline self-check, ordering/dupe checks) and ADD the v2
# schema guard + the three-class timing coverage check on top.
# ═══════════════════════════════════════════════════════════════════════════

def _lab_dir(run_dir: str) -> str:
    """v2 outputs live beside the run, under edge_lab/ (separate from v1's
    edge_engine/ dir so the two never collide)."""
    d = os.path.join(run_dir, "edge_lab")
    os.makedirs(d, exist_ok=True)
    return d


def _timing_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """Every column in the entry manifest classifies to exactly one timing class,
    and no outcome-time column leaked into the entry universe. This is the v2
    look-ahead wall, checked at scaffold time."""
    ef = entry_features(df)
    xf = exit_features(df)
    leaked = [f for f in ef if classify_timing(f) == "outcome_time"]
    overlap = sorted(set(ef) & set(xf))
    return {
        "n_entry_features": len(ef),
        "n_exit_features": len(xf),
        "entry_features": ef,
        "exit_features": xf,
        "outcome_leak_into_entry": leaked,     # must be []
        "entry_exit_overlap": overlap,         # must be []
        "wall_intact": not leaked and not overlap,
    }


def scaffold(run_dir: str, forced: bool = False) -> Dict[str, Any]:
    """Step-1 scaffold verification. Runs the schema guard, the spine's Stage-0
    trust gate, and the v2 timing-wall check. Writes edge_lab/scaffold.json.

    PASS means: this is the clean baseline CSV, the trust gate is green (gates-off
    proven, census healthy, baseline replay reproduces r_realised), and the
    look-ahead wall is intact. That is the entry ticket for Steps 2–7."""
    lab_dir = _lab_dir(run_dir)

    # 1) Schema guard + eligible population (raises loudly on a corrupt/old CSV).
    try:
        df, guard = load_lab_population(run_dir)
        guard_pass = True
        guard_error = None
    except SchemaGuardError as e:
        guard_pass = False
        guard_error = str(e)
        df = None
        guard = None

    result: Dict[str, Any] = {
        "stage": "scaffold", "run_dir": run_dir,
        "run_id": os.path.basename(run_dir),
        "generated_utc": _now_utc(), "forced": forced, "seed": SEED,
        "schema_guard": {"pass": guard_pass, "error": guard_error,
                         "evidence": guard if guard_pass else None},
    }
    if not guard_pass:
        result["pass"] = False
        result["abort_reason"] = "schema guard failed — see error"
        spine._write_json(os.path.join(lab_dir, "scaffold.json"), result)
        return result

    # 2) Spine Stage-0 trust gate (census, gates-off, baseline self-check, dupes).
    #    Reused verbatim — it is the C6-compliant, hard-won trust machinery.
    s0 = _spine_stage0(run_dir, lab_dir, forced)
    result["trust_gate"] = {
        "pass": bool(s0.get("pass")),
        "scope": s0.get("scope"),
        "verdict_capable": s0.get("verdict_capable"),
        "census": s0.get("census"),
        "checks": s0.get("checks"),
    }

    # 3) v2 timing wall (three-class classifier covers everything; no leak).
    cov = _timing_coverage(df)
    result["timing_wall"] = cov

    result["pass"] = bool(guard_pass and s0.get("pass") and cov["wall_intact"])
    result["scope"] = s0.get("scope")
    spine._write_json(os.path.join(lab_dir, "scaffold.json"), result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _print_scaffold(res: Dict[str, Any]) -> None:
    ok = "PASS" if res.get("pass") else "FAIL"
    print(f"\n=== EDGE LAB - SCAFFOLD [{ok}] ===")
    print(f"run_id : {res.get('run_id')}")
    print(f"scope  : {res.get('scope')}")
    sg = res.get("schema_guard", {})
    print(f"schema guard : {'pass' if sg.get('pass') else 'FAIL — ' + str(sg.get('error'))}")
    if sg.get("pass"):
        ev = sg.get("evidence", {})
        print(f"  rows={ev.get('n_rows')} cols={ev.get('n_cols')} "
              f"baseline cols present, numeric parse clean")
    tg = res.get("trust_gate", {})
    if tg:
        print(f"trust gate  : {'pass' if tg.get('pass') else 'FAIL'} "
              f"(scope={tg.get('scope')})")
        for c in tg.get("checks", []):
            mark = "ok " if c.get("pass") else "XX "
            print(f"    [{mark}] {c.get('check')}")
    tw = res.get("timing_wall", {})
    if tw:
        print(f"timing wall : {'intact' if tw.get('wall_intact') else 'BREACHED'} "
              f"({tw.get('n_entry_features')} entry / "
              f"{tw.get('n_exit_features')} exit features)")
        if tw.get("outcome_leak_into_entry"):
            print(f"    LEAK: {tw['outcome_leak_into_entry']}")
    if res.get("abort_reason"):
        print(f"ABORT: {res['abort_reason']}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Edge Lab v2 — research engine (SMC_EDGE_LAB_SPEC.md).")
    ap.add_argument("--run-dir", default=None,
                    help="explicit run dir; else use --start/--end to resolve one")
    ap.add_argument("--start", default=None, help="run start date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="run end date YYYY-MM-DD")
    ap.add_argument("--stage", default="scaffold", choices=["scaffold"],
                    help="only 'scaffold' exists in Step 1; later steps add stages")
    ap.add_argument("--force", action="store_true",
                    help="run even if a prior gate did not pass (stamped forced)")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir, args.start, args.end)

    if args.stage == "scaffold":
        res = scaffold(run_dir, forced=args.force)
        _print_scaffold(res)
        sys.exit(0 if res.get("pass") else 1)


if __name__ == "__main__":
    main()
