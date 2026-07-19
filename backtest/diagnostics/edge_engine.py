"""
EDGE ENGINE — the edge-discovery post-processor (EDGE_ENGINE_SPEC.md v1).

READ-ONLY. Input = one completed backtest run dir + the frozen MT5 parquet cache.
It never touches git, registry, live state, or r_realised. It reads a finished run
and decides — with anti-overfit discipline — whether this system has a real,
repeatable edge in entries, exits, pairs, or nowhere, and emits the recipe or an
honest "no edge".

Five stages, each stage-gated (stage N+1 refuses to start unless stage N wrote a
file with "pass": true). Everything seeded 42; every groupby over sorted keys; two
runs on the same inputs are byte-identical except timestamps.

    Stage 0  TRUST GATE      -> stage0_gate.json
    Stage 1  UNIVARIATE      -> stage1_features.csv/.json (+ sub-screens)
    Stage 2  EV MODEL        -> stage2_model.json
    Stage 3  EXIT OPTIMISER  -> stage3_exits.csv/.json
    Stage 4  RECIPE + OOS    -> stage4_recipe.json + edge_engine_report.md

CLI:
    python -m backtest.diagnostics.edge_engine --run-dir <dir> [--stage N] [--force]

RIGOR IS DATA-DERIVED, NOT A MODE FLAG (decided 2026-07-03). There is no
--mode main|short_range. The engine always attempts the full 3-way-split verdict
workflow. If the run's date span cannot fill DISCOVERY / VALIDATION / HOLDOUT with
>= MIN_SPLIT_N eligible trades each, the verdict path is BLOCKED and the engine
auto-degrades to the single-pool EXPLORATORY path (Stage 1 + Stage 3 as
descriptives; no Stage 2 EV model, no Stage 4 holdout verdict). The report is
stamped EXPLORATORY and every finding is worded as a hypothesis to confirm on the
full run. Scope is derived from split-N — a thin-run number is structurally typed
so it can never be read as a shippable verdict.

BLIND-SPOT GUARD: no number leaves this engine without N + window + scope attached.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest import data_loader, insights
from backtest.exit_engine import walk_multileg
from backtest import h1_only_simulator as sim

try:
    from scipy.stats import spearmanr as _spearmanr, kruskal as _kruskal
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a hard dep of this repo
    _HAS_SCIPY = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "backtest", "results")

# Run folders that must never be auto-selected by resolve_run_dir, even though
# they still exist on disk or in git history (e.g. a wrong-start rerun).
REJECTED_RUN_IDS = {
    "h1only_20080201_20251231",  # 2026-07-07: wrong start (should be 20080102)
}


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS (single block — SPEC §10)
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
BOOT_N = 10_000

# Splits by alert_ts, UTC (SPEC §3.2). Fixed ABSOLUTE dates so any sub-window maps
# into whichever era it belongs to — a 2-yr slice and the 18-yr run talk about the
# SAME discovery era. Boundaries are never recomputed relative to a run's own range.
SPLITS: Dict[str, Tuple[str, str]] = {
    "DISCOVERY":  ("2008-01-01", "2016-12-31"),
    "VALIDATION": ("2017-01-01", "2021-12-31"),
    "HOLDOUT":    ("2022-01-01", "2025-12-31"),
}
WAR_START = "2026-01-01"  # >= this is WAR — never pooled, reported label-only.

MIN_BUCKET_N = 150     # a pooled screen bucket is testable at N >= this
MIN_CELL_N = 100       # an interaction cell is testable at N >= this
MIN_QUARTER_N = 30     # a quarter counts toward pos_quarters only at N >= this
MIN_EFFECT_R = 0.10    # |ΔexpR| floor for a survivor (worth a rule on ~breakeven)
FDR_Q = 0.10           # Benjamini-Hochberg false-discovery rate
EV_SPEARMAN_FLOOR = 0.10
QUARTER_SIGN_FRAC = 0.60
RIDGE_LAMBDAS = [0.01, 0.1, 1, 10, 100]
VIF_MAX = 5.0
SCORE_FLOOR_LIVE = 4   # the live score floor (SPEC §4.3 gates-off proof)
MIN_BELOW_FLOOR_N = 50 # gates-off proof: >= this many eligible filled rows must be
                       # BELOW the live floor (SPEC §4.3). Proves the run is gates-OFF
                       # (the sub-floor tail is present at all), not a fraction of the
                       # population — scores are NOT proportional to performance, so a
                       # fraction test punished a detector that simply emits few sub-floor
                       # setups. Absolute presence is the honest on/off proof.
MIN_SPLIT_N = 500      # a split must hold >= this many eligible trades (SPEC §4.4)
CLUSTER_MIN_N = 300    # fallback-cluster minimum discovery N (SPEC §7.4)

# Walk-forward folds (SPEC §8.2): (fit_end_year_inclusive, test_start, test_end)
WF_FOLDS = [
    ("<=2015", "2016-01-01", "2018-12-31"),
    ("<=2018", "2019-01-01", "2021-12-31"),
    ("<=2021", "2022-01-01", "2025-12-31"),
]

# Pair books (SPEC §3.3). NOTE the live `pair` column writes "GOLD", NOT the spec's
# "XAUUSD" — code-is-truth: filtering on XAUUSD yields ZERO gold rows. Mapped here
# once so a future rename to XAUUSD is a single edit. Verified against real
# trades.csv 2026-07-03 (pairs seen: EURUSD GOLD NAS100 NZDUSD USDCHF USDJPY).
BOOK_A = ["EURUSD", "NZDUSD", "USDJPY", "USDCHF", "GOLD"]
BOOK_B = ["GBPUSD", "AUDUSD", "USDCAD", "EURJPY"]
BTC = "BTCUSD"
NAS = "NAS100"
BTC_STANDALONE_MIN_N = 300


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE MANIFEST (SPEC §5.1 — the engine's hardcoded feature list, with the
# TRUTH_LEDGER verified/fixed status baked in as of 2026-07-03) + timing class
# (SPEC §5.1b — decides where a survivor may ship).
# ═══════════════════════════════════════════════════════════════════════════

CONTINUOUS_FEATURES = [
    "break_close_atr", "break_body_atr", "impulse_leg_atr", "fvg_size_atr",
    "ob_range_atr", "atr_at_ob", "pd_pct", "reversal_pct", "ob_age_h1_bars",
    "ob_to_fill_hours", "bars_break_to_pullback", "bos_sequence_count",
    "score", "alert_utc_hour", "ob_body_ratio", "chop_at_alert",
    # (leg_retrace_pct_at_alert removed 2026-07-19 — uninformative for an
    # order-block-limit system; support cols leg_extreme_at_alert /
    # leg_extreme_clipped remain in the CSV as audit support, never screened.)
    # Weekly PD zone (weekly_pd.py, 2026-07-15) — ALERT-time (bars strictly before
    # alert_ts, h1_only_simulator.py:1708). weekly_pd_position is the continuous
    # read; range_high/low are support (audit), zone/agreement are categorical below.
    "weekly_pd_position_at_alert",
    # PD/PW liquidity pools (DAILY_BIAS_V4_SPEC §1.3) — FILL-time distances
    # (see FILL_TIME_FEATURES). Continuous ATR distances to the nearest unspent pool.
    "dist_next_pool_above_atr", "dist_next_pool_below_atr",
    # EQH/EQL equal-level clusters (eq_pools.py, 2026-07-14) — FILL-time. Continuous
    # shelf distances/sizes + stop-vs-pool gap. Support counts are categorical/int below.
    "eqh_above_dist_atr", "eqh_above_size", "eql_below_dist_atr", "eql_below_size",
    "eq_sl_gap_atr",
    # Approach quality (RETRACE_QUALITY_SPEC, 2026-07-15) — FILL-time entry mechanics.
    "approach_speed_atr_at_fill", "approach_body_ratio_at_fill", "approach_er_at_fill",
]

CATEGORICAL_FEATURES = [
    "bos_tag", "bos_tier", "bos_verdict", "event", "reversed_from_extreme",
    "fvg_present", "fvg_mitigation", "fvg_state", "pd_zone", "pd_alignment",
    "session", "ob_session", "fill_session", "killzone_alignment",
    "ob_in_killzone", "fill_in_killzone", "trend_alignment", "setup_badge",
    "ob_touches", "bias", "pair", "ob_walkback_depth",
    # STRUCTURE_SIGNALS_SPEC S2/S4 screen candidates (booleans + pending dir).
    "structure_ranging_at_alert", "flip_pending_at_alert",
    "flip_pending_dir_at_alert", "dr_ceiling_broken_at_ob", "dr_floor_broken_at_ob",
    # Weekly PD zone (ALERT-time) — premium/discount + H4-vs-weekly agreement.
    "weekly_pd_zone_at_alert", "pd_zone_agreement_at_alert",
    # Trend-vs-PD confluence bool (2026-07-08, was un-screened) — enters here, NOT
    # via h1_trend (which is redundant-by-construction with trend_alignment).
    "trend_pd_agree",
    # PD/PW liquidity pools (FILL-time) — day state, per-pool swept/intact status,
    # nearest-pool tiers, and the draw-on-liquidity direction read.
    "day_state_at_alert", "pdh_status_at_alert", "pdl_status_at_alert",
    "pwh_status_at_alert", "pwl_status_at_alert",
    "next_pool_above_tier", "next_pool_below_tier", "trade_toward_pool",
    # EQ clusters (FILL-time) — draw-toward + the instant-death stop-in-a-pool test
    # + intact-shelf counts (screened as levels-as-is like ob_touches).
    "eq_trade_toward", "eq_sl_at_risk",
    "eq_intact_above_count", "eq_intact_below_count",
]

ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

# FILL-TIME features (SPEC §5.1b): known only when/if the limit fills. NEVER enter
# the Stage-2 EV model (a live scorer could not compute them = look-ahead leak).
# They ARE screened in Stage 1; survivors route to the ORDER-RULE track (§8.1b).
FILL_TIME_FEATURES = {
    "ob_to_fill_hours", "bars_break_to_pullback", "fill_session",
    "fill_in_killzone", "killzone_alignment",
    # PD/PW pool + EQ columns are FILL-anchored (owner call 2026-07-16,
    # h1_only_simulator.py:1677-1697): derived from bars strictly BEFORE fill_ts,
    # because alert-time pool status can be stale by the time the limit fills. The
    # `_at_alert` suffix on the pool/EQ names is HISTORICAL — the anchor is the
    # fill. So they screen in Stage 1 but route to the order-rule track, never the
    # alert-time EV model.
    "day_state_at_alert", "pdh_status_at_alert", "pdl_status_at_alert",
    "pwh_status_at_alert", "pwl_status_at_alert",
    "dist_next_pool_above_atr", "dist_next_pool_below_atr",
    "next_pool_above_tier", "next_pool_below_tier", "trade_toward_pool",
    "eqh_above_dist_atr", "eqh_above_size", "eql_below_dist_atr", "eql_below_size",
    "eq_trade_toward", "eq_sl_gap_atr", "eq_sl_at_risk",
    "eq_intact_above_count", "eq_intact_below_count",
    # Approach quality — fill-time by construction (RETRACE_QUALITY_SPEC §1.3).
    "approach_speed_atr_at_fill", "approach_body_ratio_at_fill", "approach_er_at_fill",
}

# None-by-construction: screen each on its own valid subpopulation only, never
# impute (SPEC §5.1 missing-value rule).
FVG_ONLY_FEATURES = {"fvg_size_atr", "fvg_mitigation", "fvg_state"}  # need fvg_present
DR_ONLY_FEATURES = {"pd_pct"}  # need a valid dealing range (non-null)
# Weekly-PD: None until a completed prior week exists; screen on non-null only.
WEEKLY_PD_ONLY_FEATURES = {"weekly_pd_position_at_alert", "weekly_pd_zone_at_alert",
                           "pd_zone_agreement_at_alert"}
# Pool/EQ/approach: None on never_filled rows (fill_ts=None) + thin history; screen
# each on its own non-null subpopulation only (same rule as FVG_ONLY).

# Outcome / geometry columns Stage 0 requires present (SPEC §4.1).
REQUIRED_OUTCOME_COLS = [
    "r_realised", "exit_reason", "fill_ts", "entry", "sl_initial", "tp1",
    "eligible_for_headline",
]

# SL-anatomy columns (SPEC §5.5) — present on current runs; absence disables the
# sub-screen with a stamped note (they are 2026-07-02 additions).
SL_ANATOMY_COLS = ["sl_bar_was_sweep", "sl_swept_then_tp1"]

# Decreed out (SPEC §5.1) — must NOT appear in the screened manifest.
DECREED_OUT = {"sweep_present"}


def _classify_timing(feat: str) -> str:
    """ALERT-TIME vs FILL-TIME (SPEC §5.1b). Stage 0 check 9 asserts this covers
    every manifest feature."""
    return "fill_time" if feat in FILL_TIME_FEATURES else "alert_time"


# ═══════════════════════════════════════════════════════════════════════════
# SMALL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quarter(ts) -> str:
    ts = pd.to_datetime(ts, utc=True)
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


def _to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _symbol_map() -> Dict[str, str]:
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    return {p["name"]: p["symbol"] for p in cfg["pairs"]}


def _pair_conf_map() -> Dict[str, Dict[str, Any]]:
    """pair name -> its config block (for the cost model: spread_pips,
    decimal_places, pair_type). Mirrors the simulator's pip-size derivation."""
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    return {p["name"]: p for p in cfg["pairs"]}


def _pip_size(pair_conf: Dict[str, Any]) -> float:
    """EXACT mirror of h1_only_simulator.py:529-534. Crypto reads spread in
    dollars (pip 1.0); else 0.01 for <=3 dp, 0.0001 otherwise."""
    if pair_conf.get("pair_type") == "crypto":
        return 1.0
    decimal_places = int(pair_conf.get("decimal_places", 5))
    return 0.01 if decimal_places <= 3 else 0.0001


def bootstrap_ci(values, n_boot: int = BOOT_N):
    """95% bootstrap CI on the mean, seed 42. Reuses insights.bootstrap_ci but
    ALWAYS at 10k (its default is 5k). (None, None) under n<5 = THIN, never a pass."""
    return insights.bootstrap_ci(list(values), n_boot=n_boot)


def bootstrap_diff_ci(a, b, n_boot: int = BOOT_N, paired: bool = False):
    """Bootstrap CI on mean(a) - mean(b), seed 42.

    paired=True: a and b are aligned per-trade (same trades) -> resample the diff
    vector (the variance killer for recipe-vs-baseline, SPEC §7.5). paired=False:
    two independent samples (e.g. top vs bottom bucket, different trades) ->
    resample each independently (SPEC §5.3).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(SEED)
    if paired:
        if len(a) != len(b) or len(a) < 5:
            return None, None
        d = a - b
        boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    else:
        if len(a) < 5 or len(b) < 5:
            return None, None
        ia = rng.integers(0, len(a), size=(n_boot, len(a)))
        ib = rng.integers(0, len(b), size=(n_boot, len(b)))
        boots = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return round(lo, 4), round(hi, 4)


def _ci_excludes_zero(lo, hi) -> bool:
    return lo is not None and hi is not None and (lo > 0 or hi < 0)


def benjamini_hochberg(pvals: List[float], q: float = FDR_Q) -> List[bool]:
    """BH-FDR: returns a reject[] mask aligned to pvals. NaN p-values never reject."""
    m = len([p for p in pvals if p is not None and not (isinstance(p, float) and math.isnan(p))])
    if m == 0:
        return [False] * len(pvals)
    order = sorted(
        (i for i, p in enumerate(pvals)
         if p is not None and not (isinstance(p, float) and math.isnan(p))),
        key=lambda i: pvals[i],
    )
    reject = [False] * len(pvals)
    max_k = -1
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= (rank / m) * q:
            max_k = rank
    if max_k >= 0:
        for rank, i in enumerate(order, start=1):
            if rank <= max_k:
                reject[i] = True
    return reject


def _pos_quarters(sub: pd.DataFrame, r_col: str = "r_realised",
                  ts_col: str = "alert_ts") -> Tuple[int, int]:
    """(positive_quarters, counted_quarters). A quarter counts only at
    N >= MIN_QUARTER_N (SPEC §3.2)."""
    if sub.empty:
        return 0, 0
    q = sub.assign(_q=sub[ts_col].map(_quarter))
    pos = counted = 0
    for _, g in sorted(q.groupby("_q"), key=lambda kv: kv[0]):
        if len(g) < MIN_QUARTER_N:
            continue
        counted += 1
        if g[r_col].mean() > 0:
            pos += 1
    return pos, counted


def _cell_stats(sub: pd.DataFrame, r_col: str = "r_realised") -> Dict[str, Any]:
    """The standard per-cell stat block (SPEC §5.3). Every number here is meant to
    be printed WITH its N — callers must not strip N."""
    vals = sub[r_col].dropna().tolist()
    if not vals:
        return {"n": 0, "expR": None, "ci_lo": None, "ci_hi": None,
                "wr_pct": None, "totR": 0.0, "pos_quarters": "0/0"}
    lo, hi = bootstrap_ci(vals)
    pos, counted = _pos_quarters(sub, r_col)
    return {
        "n": len(vals),
        "expR": round(float(np.mean(vals)), 4),
        "ci_lo": lo, "ci_hi": hi,
        "ci_excludes_0": _ci_excludes_zero(lo, hi),
        "wr_pct": insights.win_rate_pct(sub, r_col),
        "totR": round(float(np.sum(vals)), 2),
        "pos_quarters": f"{pos}/{counted}",
    }


# ═══════════════════════════════════════════════════════════════════════════
# STAGE FILE I/O + GATING
# ═══════════════════════════════════════════════════════════════════════════

def _stage_path(engine_dir: str, stage: int, ext: str = "json") -> str:
    names = {0: "stage0_gate", 1: "stage1_features", 2: "stage2_model",
             3: "stage3_exits", 4: "stage4_recipe"}
    return os.path.join(engine_dir, f"{names[stage]}.{ext}")


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _prior_passed(engine_dir: str, stage: int, forced: bool) -> bool:
    """Stage N refuses to start unless stage N-1 wrote pass:true (--force
    overrides with a red warning, SPEC §2)."""
    if stage == 0:
        return True
    prior = _read_json(_stage_path(engine_dir, stage - 1))
    ok = bool(prior and prior.get("pass") is True)
    if not ok and forced:
        print(f"\033[31m[--force] stage {stage-1} did not pass; running stage "
              f"{stage} anyway — outputs stamped forced:true\033[0m", file=sys.stderr)
        return True
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# RUN RESOLUTION (SPEC §14.3 — Option A latest-wins; format-agnostic on the
# run-ID string so old `h1only_*` folders and any future format both work)
# ═══════════════════════════════════════════════════════════════════════════

def resolve_run_dir(run_dir: Optional[str], start: Optional[str],
                    end: Optional[str]) -> str:
    """An explicit --run-dir always wins. Otherwise, given a date range, pick the
    matching folder with the newest runstamp (lexical max = latest)."""
    if run_dir:
        rd = run_dir if os.path.isabs(run_dir) else os.path.join(ROOT, run_dir)
        if not os.path.isdir(rd):
            raise SystemExit(f"run dir not found: {rd}")
        return rd
    if not (start and end):
        raise SystemExit("give --run-dir, or --start and --end")
    s = start.replace("-", "")
    e = end.replace("-", "")
    # Match any folder whose name CONTAINS both the start and end date tokens —
    # format-agnostic (h1only_<s>_<e>, <s>_<e>__<stamp>, …). Latest by name.
    cands = [d for d in os.listdir(RESULTS)
             if os.path.isdir(os.path.join(RESULTS, d)) and s in d and e in d]
    # Reject known-bad/superseded run folders outright, even if present on disk
    # or in git history — a wrong-start rerun (e.g. 2026-07-07 Feb-01-start
    # mistake, should have been Jan-02) must never be selectable again.
    cands = [d for d in cands if d not in REJECTED_RUN_IDS]
    if not cands:
        raise SystemExit(f"no run folder matches {start}..{end} under {RESULTS}")
    return os.path.join(RESULTS, sorted(cands)[-1])


# ═══════════════════════════════════════════════════════════════════════════
# POPULATION LOADING + SPLIT ASSIGNMENT (SPEC §3.1, §3.2)
# ═══════════════════════════════════════════════════════════════════════════

def _split_of(ts) -> str:
    """Map an alert_ts to its split label (or WAR / OUT)."""
    t = pd.to_datetime(ts, utc=True)
    if t >= pd.Timestamp(WAR_START, tz="UTC"):
        return "WAR"
    for name, (lo, hi) in SPLITS.items():
        if pd.Timestamp(lo, tz="UTC") <= t <= pd.Timestamp(hi, tz="UTC") + pd.Timedelta(days=1):
            return name
    return "OUT"


def _book_of(pair: str) -> str:
    if pair in BOOK_A:
        return "A"
    if pair in BOOK_B:
        return "B"
    if pair == BTC:
        return "BTC"
    return "OTHER"


def load_population(run_dir: str) -> pd.DataFrame:
    """Read trades.csv, apply the §3.1 row filters IN ORDER, stamp split + book.

    Filter order matters: eligibility FIRST (kills never_filled / timeout /
    window_end / IST / weekend rows, all of which carry junk categorical values
    like 'never_filled' that would otherwise pollute screens), THEN proximal-only,
    THEN drop NAS100. BTC is stamped book=BTC (separated downstream, never pooled).
    """
    trades_p = os.path.join(run_dir, "trades.csv")
    if not os.path.exists(trades_p):
        raise SystemExit(f"no trades.csv at {trades_p} — run the backtest first")
    df = pd.read_csv(trades_p, low_memory=False)

    # 1) eligible_for_headline (the ONE eligibility rule — never re-derived).
    if "eligible_for_headline" not in df.columns:
        raise SystemExit("trades.csv lacks eligible_for_headline — stale/broken "
                         "run; re-run the backtest on current code")
    df = df[df["eligible_for_headline"] == True].copy()  # noqa: E712

    # 2) proximal-only (belt-and-braces).
    if "entry_zone" in df.columns:
        df = df[df["entry_zone"] == "proximal"].copy()

    # 3) drop NAS100 entirely.
    df = df[df["pair"] != NAS].copy()

    # Stamp derived split/book columns (used everywhere downstream).
    df["_split"] = df["alert_ts"].map(_split_of)
    df["_book"] = df["pair"].map(_book_of)
    df["_quarter"] = df["alert_ts"].map(_quarter)
    return df.reset_index(drop=True)


def split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["_split"] == split].copy()


def pooled_fx_gold(df: pd.DataFrame) -> pd.DataFrame:
    """Book A + Book B pooled (features are ATR-normalised → cross-instrument
    comparable). BTC and NAS excluded (§3.3)."""
    return df[df["_book"].isin(["A", "B"])].copy()


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 0 — TRUST GATE (SPEC §4). All checks must pass; each emits pass/fail +
# evidence. Everything downstream is blind if Stage 0 is not green.
# ═══════════════════════════════════════════════════════════════════════════

def _gates_off_proof(filled: pd.DataFrame) -> Dict[str, Any]:
    """SPEC §4.3 check 3. Proves the run was executed with the score gate OFF:
    a gated run holds ZERO sub-floor trades. We test PRESENCE of the sub-floor
    tail (>= MIN_BELOW_FLOOR_N filled below the live floor), NOT its share —
    scores are not proportional to performance, so the size of the tail is not a
    trust signal, only its existence is (a fraction test wrongly failed detectors
    that emit few sub-floor setups). Pure predicate so the guard is testable
    without a bar cache — see tests/test_gates_off_proof.py."""
    if "score" not in filled.columns or not len(filled):
        return {"pass": False, "below_floor": 0, "n": len(filled),
                "note": "no score column / no filled rows"}
    below = int((filled["score"] < SCORE_FLOOR_LIVE).sum())
    ok = below >= MIN_BELOW_FLOOR_N
    return {
        "pass": ok, "below_floor": below, "n": len(filled),
        "frac": round(below / max(len(filled), 1), 4),
        "floor": SCORE_FLOOR_LIVE, "min_below": MIN_BELOW_FLOOR_N,
        "note": ("gates confirmed off" if ok else
                 "input is a GATED run — engine is blind to half the answer"),
    }


def stage0(run_dir: str, engine_dir: str, forced: bool) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def _check(name: str, passed: bool, **evidence):
        checks.append({"check": name, "pass": bool(passed), **evidence})
        return passed

    trades_p = os.path.join(run_dir, "trades.csv")
    raw = pd.read_csv(trades_p, low_memory=False)
    cols = set(raw.columns)

    # ── Check 1: required feature + outcome columns exist ───────────────────
    # setup_badge / bos_verdict / SL-anatomy are 2026-07 additions; a run missing
    # them is a STALE run and SHOULD fail here (the guard working, not a bug).
    required = set(ALL_FEATURES) | set(REQUIRED_OUTCOME_COLS)
    required -= {"pair"}  # pair is always present; keep the set honest anyway
    missing = sorted(c for c in required if c not in cols)
    _check("columns_exist", not missing, missing=missing,
           note=("stale run — re-run the backtest on current code"
                 if missing else "all required columns present"))

    # ── Check 2: ledger trust — sweep_present must NOT be screened ───────────
    _check("ledger_trust_sweep_excluded", not (DECREED_OUT & set(ALL_FEATURES)),
           decreed_out=sorted(DECREED_OUT))

    # ── Check 9 (done early — cheap): every manifest feature is timing-classed ─
    unclassified = [f for f in ALL_FEATURES
                    if _classify_timing(f) not in ("alert_time", "fill_time")]
    _check("feature_timing_classes", not unclassified,
           unclassified=unclassified,
           fill_time=sorted(FILL_TIME_FEATURES))

    # Everything past here needs the eligible population.
    if missing:
        result = {"stage": 0, "pass": False, "run_dir": run_dir,
                  "generated_utc": _now_utc(), "forced": forced, "checks": checks,
                  "abort_reason": "required columns missing — cannot proceed"}
        _write_json(_stage_path(engine_dir, 0), result)
        return result

    df = load_population(run_dir)
    filled = df[df["fill_ts"].notna()].copy()

    # ── Check 3: gates-off proof (>= MIN_BELOW_FLOOR_N eligible filled below floor) ─
    gp = _gates_off_proof(filled)
    _check("gates_off_proof", gp.pop("pass"), **gp)

    # ── Check 4: population census (per split / book) + MIN_SPLIT_N ─────────
    census: Dict[str, Any] = {"by_split": {}, "by_book": {}, "war": 0}
    for sp in list(SPLITS) + ["WAR", "OUT"]:
        census["by_split"][sp] = int((df["_split"] == sp).sum())
    for bk in ["A", "B", "BTC", "OTHER"]:
        census["by_book"][bk] = int((df["_book"] == bk).sum())
    census["war"] = census["by_split"]["WAR"]
    census["by_pair_year"] = _census_pair_year(df)

    split_ok = {sp: census["by_split"][sp] >= MIN_SPLIT_N for sp in SPLITS}
    verdict_capable = all(split_ok.values())
    # SPEC §4.4: FAIL if any split < MIN_SPLIT_N — BUT auto-degradation (§15)
    # turns this into an informational census + EXPLORATORY scope instead of a
    # hard abort. The engine still runs; it just cannot ship a verdict. This is
    # the "no mode dropdown" design: rigor derived from N, not a flag.
    _check("population_census", True, census=census, split_ok=split_ok,
           verdict_capable=verdict_capable,
           scope=("verdict" if verdict_capable else "exploratory"),
           note=("all splits >= MIN_SPLIT_N — full verdict workflow"
                 if verdict_capable else
                 f"a split < {MIN_SPLIT_N} — EXPLORATORY scope (hypotheses only, "
                 "no shippable verdict); confirm findings on the full run"))

    # ── Check 5: baseline exit self-check (the exit_lab pattern) ────────────
    self_check = _baseline_self_check(filled)
    _check("baseline_exit_self_check", self_check["pass"], **self_check)

    # ── Check 6: news columns population (report-only) ──────────────────────
    news_usable = False
    if "news_event_title" in filled.columns and len(filled):
        nn = int(filled["news_event_title"].astype(str).str.strip().replace(
            {"": None, "nan": None, "None": None}).notna().sum())
        news_usable = nn > 0
        _check("news_population", True, non_null=nn, n=len(filled),
               news_usable=news_usable)
    else:
        _check("news_population", True, non_null=0, news_usable=False)

    # ── Check 7: duplicate / ordering sanity ────────────────────────────────
    dup = int(df["setup_id"].duplicated().sum()) if "setup_id" in df.columns else 0
    order_ok = _ordering_ok(df)
    _check("no_dupes_ordering", dup == 0 and order_ok,
           duplicate_setup_ids=dup, ordering_ok=order_ok)

    # ── Check 8: BTC boundary note ──────────────────────────────────────────
    btc_n = int((df["_book"] == "BTC").sum())
    _check("btc_boundary", True, btc_n=btc_n,
           btc_standalone=btc_n >= BTC_STANDALONE_MIN_N)

    # SL-anatomy availability (not a pass/fail — enables/disables §5.5 sub-screen).
    sl_anatomy_ok = all(c in cols for c in SL_ANATOMY_COLS)

    all_pass = all(c["pass"] for c in checks)
    result = {
        "stage": 0, "pass": all_pass, "run_dir": run_dir,
        "run_id": os.path.basename(run_dir),
        "generated_utc": _now_utc(), "forced": forced,
        "verdict_capable": verdict_capable,
        "scope": "verdict" if verdict_capable else "exploratory",
        "news_usable": news_usable,
        "sl_anatomy_usable": sl_anatomy_ok,
        "census": census,
        "checks": checks,
    }
    _write_json(_stage_path(engine_dir, 0), result)
    return result


def _census_pair_year(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    yr = _to_utc(df["alert_ts"]).dt.year
    for pair in sorted(df["pair"].dropna().unique()):
        sub_yr = yr[df["pair"] == pair]
        out[pair] = {str(int(y)): int((sub_yr == y).sum())
                     for y in sorted(sub_yr.dropna().unique())}
    return out


def _ordering_ok(df: pd.DataFrame) -> bool:
    """alert_ts <= fill_ts <= exit_ts on all eligible rows (nulls tolerated —
    an unfilled row has no fill/exit; those are already filtered by eligibility,
    but be defensive)."""
    a = _to_utc(df["alert_ts"])
    f = _to_utc(df["fill_ts"])
    e = _to_utc(df["exit_ts"]) if "exit_ts" in df.columns else pd.Series(pd.NaT, index=df.index)
    ok = True
    m_af = a.notna() & f.notna()
    if m_af.any():
        ok = ok and bool((a[m_af] <= f[m_af]).all())
    m_fe = f.notna() & e.notna()
    if m_fe.any():
        ok = ok and bool((f[m_fe] <= e[m_fe]).all())
    return ok


def _baseline_self_check(filled: pd.DataFrame) -> Dict[str, Any]:
    """SPEC §4.5: replay the live baseline recipe over every eligible trade's
    post-fill bars from the frozen cache; the replay must reproduce committed
    r_realised (|mean diff| <= 0.01 AND per-trade |diff| > 0.02 on <= 1% of rows).
    Stage 3 is worthless if the walker can't reproduce truth."""
    if filled.empty:
        return {"pass": False, "note": "no filled rows to self-check"}
    baseline = {"legs": [(1.0, "tp1")], "be_trigger_r": 1.0, "be_to_r": 0.0}
    rep = _replay_recipe(filled, baseline)
    if rep is None or rep.empty:
        return {"pass": False, "note": "self-check replay produced no rows"}
    # Compare GROSS replay to committed r_realised. The committed r_realised is
    # itself gross of spread on non-SL exits (the sim widens only the SL), so the
    # walker's gross output is the correct thing to reconcile. Cost is applied
    # only in Stage 3/4 selection, never here.
    committed = rep["committed_r"].to_numpy(dtype=float)
    replayed = rep["r_gross"].to_numpy(dtype=float)
    mean_diff = abs(float(committed.mean() - replayed.mean()))
    per_trade_bad = int((np.abs(committed - replayed) > 0.02).sum())
    bad_frac = per_trade_bad / len(rep)
    ok = mean_diff <= 0.01 and bad_frac <= 0.01
    return {"pass": ok, "mean_diff": round(mean_diff, 5),
            "per_trade_bad": per_trade_bad, "n": len(rep),
            "bad_frac": round(bad_frac, 5),
            "note": ("baseline replay reproduces r_realised"
                     if ok else "walker does NOT reproduce truth — Stage 3 blind")}


# ═══════════════════════════════════════════════════════════════════════════
# BAR CACHE + RECIPE REPLAY (exit_lab pattern — one bar-load per pair, SPEC §7.1)
# ═══════════════════════════════════════════════════════════════════════════

_BAR_CACHE: Dict[str, pd.DataFrame] = {}
_BARS_LOADED_FOR: Optional[Tuple] = None


def _ensure_bars(trades: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Load each needed pair's H1 bars ONCE for the whole engine. Window = the
    trades' own fill span + pads, so any run length reconstructs every trade."""
    global _BARS_LOADED_FOR
    pairs = tuple(sorted(trades["pair"].dropna().unique()))
    fts = _to_utc(trades["fill_ts"]).dropna()
    if fts.empty:
        return {}
    key = (pairs, fts.min(), fts.max())
    if _BARS_LOADED_FOR == key and _BAR_CACHE:
        return _BAR_CACHE
    symbols = _symbol_map()
    bar_start = (fts.min() - pd.Timedelta(days=5)).to_pydatetime()
    bar_end = (fts.max() + pd.Timedelta(days=sim.MAX_HOLD_H1_BARS // 24 + 5)).to_pydatetime()
    _BAR_CACHE.clear()
    for pair in pairs:
        symv = symbols.get(pair)
        if not symv:
            continue
        d = data_loader.load_bars(symv, "1h", bar_start, bar_end)
        if d is not None and not d.empty:
            if d.index.tz is None:
                d.index = d.index.tz_localize("UTC")
            _BAR_CACHE[pair] = d
    _BARS_LOADED_FOR = key
    return _BAR_CACHE


def _replay_recipe(trades: pd.DataFrame, recipe: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Replay ONE recipe over each eligible trade's post-fill bars. One row per
    trade: r (net set later), gross r, exit reason, committed_r for the self-check.
    Pessimism (SL-first, fill-bar TP suppressed) is inherited from walk_multileg."""
    bars = _ensure_bars(trades)
    if not bars:
        return None
    max_hold = sim.MAX_HOLD_H1_BARS
    wk_flat = sim.WEEKEND_FLAT
    wk_hour = sim.WEEKEND_FLAT_HOUR_UTC
    rows: List[Dict[str, Any]] = []
    for _, t in trades.iterrows():
        pair = t["pair"]
        pb = bars.get(pair)
        if pb is None:
            continue
        fill_ts = pd.to_datetime(t["fill_ts"], utc=True)
        future = pb.loc[pb.index >= fill_ts]
        if future.empty:
            continue
        future = future.iloc[: max_hold + 2]
        bias = t["bias"] if t.get("bias") in ("LONG", "SHORT") else (
            "LONG" if t.get("direction") == "bullish" else "SHORT")
        try:
            entry = float(t["entry"]); sl = float(t["sl_initial"]); tp1 = float(t["tp1"])
        except (TypeError, ValueError):
            continue
        r_distance = abs(entry - sl)
        if r_distance <= 0:
            continue
        res = walk_multileg(future, bias, entry, sl, r_distance, tp1, recipe,
                            weekend_flat=wk_flat, weekend_hour_utc=wk_hour,
                            max_hold=max_hold)
        rows.append({
            "setup_id": t.get("setup_id"), "pair": pair,
            "alert_ts": t["alert_ts"], "_split": t.get("_split"),
            "_book": t.get("_book"), "_quarter": t.get("_quarter"),
            "committed_r": float(t["r_realised"]),
            "r_gross": float(res["r_realised"]),
            "exit_reason": res["exit_reason"],
            "exit_price": res.get("exit_price"),
            "entry": entry, "sl_initial": sl, "r_distance": r_distance,
            "legs": res.get("legs"),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — UNIVARIATE SCREEN (SPEC §5)
#
# Screens every §5.1 feature. Continuous → quintiles with edges computed on
# DISCOVERY only, applied frozen to VALIDATION (fresh edges would leak and make
# buckets incomparable). Categorical → levels as-is (discovery N<150 → 'other').
# Survivor = passes ALL of: BH-FDR discovery signal + top-vs-bottom ΔexpR CI
# excludes 0, validation sign persistence + per-quarter favoured ≥60%, both
# extreme buckets N≥150 each split, |ΔexpR|≥0.10R in discovery. Only survivors
# feed Stage 2. Fill-time survivors route to the order-rule track (§8.1b), never
# the EV model.
# ═══════════════════════════════════════════════════════════════════════════

# Pre-registered continuous quantile bins that are NOT plain quintiles (SPEC §5.1c
# H-SNAPBACK). bars_break_to_pullback also gets these explicit bins; both the
# quintile screen and these bins enter the same BH-FDR family.
SNAPBACK_BINS = [(1, 2), (3, 5), (6, 12), (13, 10**9)]

# Fixed, pre-registered two-way interactions (SPEC §5.6). NO data-driven fishing.
INTERACTIONS = [
    ("pair", "session"),
    ("pd_zone", "event"),
    ("break_close_atr_q", "event"),  # break_close_atr quintile × event
    ("bos_verdict", "bos_tier"),
    ("killzone_alignment", "session"),
]


def _discovery_quintile_edges(disc: pd.DataFrame, feat: str) -> Optional[List[float]]:
    """5 quintile edges from DISCOVERY only, on the feature's valid subpopulation.
    Returns None if too few distinct values to bucket."""
    sub = _feature_subpop(disc, feat)
    vals = pd.to_numeric(sub[feat], errors="coerce").dropna()
    if len(vals) < MIN_BUCKET_N or vals.nunique() < 5:
        return None
    qs = np.quantile(vals, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges = sorted(set(round(float(q), 8) for q in qs))
    return edges if len(edges) >= 3 else None


def _feature_subpop(df: pd.DataFrame, feat: str) -> pd.DataFrame:
    """None-by-construction rule (SPEC §5.1): FVG features screened only on
    fvg_present==True; pd_pct only where the DR was valid (pd_pct non-null).
    Never impute — restrict the screen to the feature's valid rows.

    The pool / EQ / weekly-PD / approach features are also None-by-construction:
    None on never_filled rows (fill anchor) and on thin history. They are screened
    on their own non-null subpopulation the same way — restricting to notna() is
    the correct honest screen (a None is 'not measurable here', not a zero)."""
    if feat in FVG_ONLY_FEATURES and "fvg_present" in df.columns:
        return df[df["fvg_present"] == True].copy()  # noqa: E712
    if (feat in DR_ONLY_FEATURES or feat in WEEKLY_PD_ONLY_FEATURES
            or feat in FILL_TIME_FEATURES):
        if feat in df.columns:
            return df[df[feat].notna()].copy()
    return df


def _assign_quintile(df: pd.DataFrame, feat: str, edges: List[float]) -> pd.Series:
    """Apply frozen discovery edges to any split. Bucket index 0..k-1; NaN outside."""
    v = pd.to_numeric(df[feat], errors="coerce")
    # right-closed bins; clip to edges so validation extremes land in end buckets.
    labels = list(range(len(edges) - 1))
    b = pd.cut(v, bins=edges, labels=labels, include_lowest=True, duplicates="drop")
    return b


def _spearman(sub: pd.DataFrame, feat: str, r_col: str = "r_realised"):
    """Spearman(raw feature, r_realised) + p on the feature's valid subpop."""
    if not _HAS_SCIPY:
        return None, None
    s = _feature_subpop(sub, feat)
    x = pd.to_numeric(s[feat], errors="coerce")
    y = pd.to_numeric(s[r_col], errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 10 or x[m].nunique() < 3:
        return None, None
    rho, p = _spearmanr(x[m], y[m])
    if isinstance(rho, float) and math.isnan(rho):
        return None, None
    return round(float(rho), 4), float(p)


def _kruskal_p(sub: pd.DataFrame, feat: str, r_col: str = "r_realised"):
    """Kruskal-Wallis p across a categorical's levels (SPEC §5.4)."""
    if not _HAS_SCIPY:
        return None
    groups = []
    for _, g in sub.groupby(feat):
        vals = pd.to_numeric(g[r_col], errors="coerce").dropna().tolist()
        if len(vals) >= 5:
            groups.append(vals)
    if len(groups) < 2:
        return None
    try:
        _, p = _kruskal(*groups)
        return float(p)
    except ValueError:
        return None


def _merge_rare_levels(df: pd.DataFrame, feat: str, disc: pd.DataFrame) -> pd.Series:
    """Categorical levels with DISCOVERY N<150 merged into 'other' (SPEC §5.2)."""
    keep = disc[feat].value_counts()
    keep = set(keep[keep >= MIN_BUCKET_N].index)
    return df[feat].map(lambda v: v if v in keep else "other")


def _continuous_screen(disc: pd.DataFrame, val: Optional[pd.DataFrame], feat: str,
                       buckets_out: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Screen one continuous feature. Returns the feature-level record (pre-verdict)
    with discovery/validation top-vs-bottom ΔexpR, CIs, Spearman, and the p-value
    that enters the BH-FDR family.

    `val is None` = DISCOVERY-ONLY mode (SPEC_STAGED §3): emit DISCOVERY bucket rows
    only, compute discovery-side stats only, and NEVER add any `*_val` key (tests
    assert their ABSENCE). The `val is not None` path is byte-identical to before —
    this is pure addition of the None branch."""
    disc_only = val is None
    edges = _discovery_quintile_edges(disc, feat)
    rec: Dict[str, Any] = {"feature": feat, "type": "continuous",
                           "timing": _classify_timing(feat), "edges": edges}
    if edges is None:
        rec.update({"verdict": "thin", "note": "insufficient distinct values"})
        return rec

    disc_s = _feature_subpop(disc, feat).copy()
    disc_s["_q"] = _assign_quintile(disc_s, feat, edges)
    n_buckets = len(edges) - 1
    lo_b, hi_b = 0, n_buckets - 1

    splits = (("DISCOVERY", disc_s),) if disc_only else (
        ("DISCOVERY", disc_s), ("VALIDATION", _feature_subpop(val, feat).copy()))
    if not disc_only:
        splits[1][1]["_q"] = _assign_quintile(splits[1][1], feat, edges)
    for split_name, s in splits:
        for b in range(n_buckets):
            cell = s[s["_q"] == b]
            st = _cell_stats(cell)
            buckets_out.append({"feature": feat, "split": split_name, "bucket": b,
                                "edge_lo": edges[b], "edge_hi": edges[b + 1], **st})

    # top-vs-bottom ΔexpR + independent bootstrap CI on the difference (§5.3).
    def _delta(s):
        top = pd.to_numeric(s[s["_q"] == hi_b]["r_realised"], errors="coerce").dropna()
        bot = pd.to_numeric(s[s["_q"] == lo_b]["r_realised"], errors="coerce").dropna()
        d = (top.mean() - bot.mean()) if len(top) and len(bot) else None
        clo, chi = bootstrap_diff_ci(top.tolist(), bot.tolist(), paired=False)
        return (round(float(d), 4) if d is not None else None, clo, chi,
                len(top), len(bot))

    d_disc, dlo, dhi, nt_d, nb_d = _delta(disc_s)
    rho_d, p_d = _spearman(disc, feat)
    fav_b = hi_b if (d_disc is not None and d_disc >= 0) else lo_b

    rec.update({
        "n_buckets": n_buckets,
        "delta_disc": d_disc, "delta_disc_ci": [dlo, dhi],
        "spearman_disc": rho_d, "spearman_p_disc": p_d,
        "top_bottom_n_disc": [nt_d, nb_d],
        "favoured_bucket": fav_b,
        "_fdr_p": p_d,  # discovery Spearman p enters the BH-FDR family
    })
    if disc_only:
        return rec

    val_s = splits[1][1]
    d_val, vlo, vhi, nt_v, nb_v = _delta(val_s)
    rho_v, p_v = _spearman(val, feat)
    # per-quarter of the FAVOURED bucket in validation (favoured = discovery's
    # higher-expR extreme). SPEC §5.4 criterion 2.
    fav_pos, fav_counted = _pos_quarters(val_s[val_s["_q"] == fav_b])
    rec.update({
        "delta_val": d_val, "delta_val_ci": [vlo, vhi],
        "spearman_val": rho_v, "spearman_p_val": p_v,
        "top_bottom_n_val": [nt_v, nb_v],
        "val_favoured_pos_quarters": f"{fav_pos}/{fav_counted}",
    })
    return rec


def _norm_cat_levels(s: pd.Series) -> pd.Series:
    """Normalise a categorical column to STRING levels, preserving NaN as NaN.

    Fixes the dtype bug (EDGE_DISCOVERY_REPORT_SPEC F3): level keys are stored as
    `str(lvl)` but the raw column keeps its native dtype (bool / int64 / mixed
    object), so `frame[frame[feat] == "True"]` matched ZERO rows and every such
    feature's CI collapsed to (None, None) → verdict capped at noise forever. We
    stringify ONCE at the top of the screen so keys, groupby levels, `_lvl_vals`
    comparisons and `_pos_quarters` filters all compare string-to-string. NaN is
    preserved (never coerced to a "nan" level) so rare-level merging and Kruskal
    grouping still drop missing rows exactly as before."""
    return s.map(lambda v: v if (isinstance(v, float) and math.isnan(v)) or v is None
                 else str(v))


def _categorical_screen(disc: pd.DataFrame, val: Optional[pd.DataFrame], feat: str,
                        buckets_out: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Screen one categorical. Levels as-is (rare→other). Top-vs-bottom = best vs
    worst discovery level (each N≥150). BH-FDR p = discovery Kruskal-Wallis.

    `val is None` = DISCOVERY-ONLY mode (SPEC_STAGED §3): discovery bucket rows +
    discovery stats only, no `*_val` keys. `val is not None` path byte-identical."""
    disc_only = val is None
    rec: Dict[str, Any] = {"feature": feat, "type": "categorical",
                           "timing": _classify_timing(feat)}
    # Normalise level values to strings ONCE (both frames), BEFORE rare-level
    # merging, so every downstream comparison ('best' key vs the column) matches.
    # `disc_s` is the stringified discovery frame the keep-set is derived from,
    # so `_merge_rare_levels` and the `frame[feat] == lvl` filters agree.
    disc_s = disc.copy()
    disc_s[feat] = _norm_cat_levels(disc_s[feat])
    d = disc_s.copy()
    d[feat] = _merge_rare_levels(d, feat, disc_s)

    disc_levels = {}
    for lvl, g in sorted(d.groupby(feat), key=lambda kv: str(kv[0])):
        st = _cell_stats(g)
        disc_levels[str(lvl)] = st
        buckets_out.append({"feature": feat, "split": "DISCOVERY",
                            "level": str(lvl), **st})
    if not disc_only:
        v = val.copy()
        v[feat] = _norm_cat_levels(v[feat])
        v[feat] = _merge_rare_levels(v, feat, disc_s)
        for lvl, g in sorted(v.groupby(feat), key=lambda kv: str(kv[0])):
            st = _cell_stats(g)
            buckets_out.append({"feature": feat, "split": "VALIDATION",
                                "level": str(lvl), **st})

    # best vs worst discovery level among those with N≥150.
    testable = {k: s for k, s in disc_levels.items()
                if s["n"] >= MIN_BUCKET_N and s["expR"] is not None}
    if len(testable) < 2:
        rec.update({"verdict": "thin", "note": "fewer than 2 testable levels",
                    "_fdr_p": None})
        return rec
    best = max(testable, key=lambda k: testable[k]["expR"])
    worst = min(testable, key=lambda k: testable[k]["expR"])

    def _lvl_vals(frame, lvl):
        return pd.to_numeric(frame[frame[feat] == lvl]["r_realised"],
                             errors="coerce").dropna().tolist()

    d_disc = round(testable[best]["expR"] - testable[worst]["expR"], 4)
    dlo, dhi = bootstrap_diff_ci(_lvl_vals(d, best), _lvl_vals(d, worst), paired=False)
    p_d = _kruskal_p(d, feat)

    rec.update({
        "best_level": best, "worst_level": worst,
        "delta_disc": d_disc, "delta_disc_ci": [dlo, dhi],
        "best_worst_n_disc": [testable[best]["n"], testable[worst]["n"]],
        "_fdr_p": p_d,
    })
    if disc_only:
        return rec

    vb, vw = _lvl_vals(v, best), _lvl_vals(v, worst)
    d_val = round((np.mean(vb) - np.mean(vw)), 4) if (len(vb) and len(vw)) else None
    vlo, vhi = bootstrap_diff_ci(vb, vw, paired=False)
    fav_pos, fav_counted = _pos_quarters(v[v[feat] == best])
    rec.update({
        "delta_val": d_val, "delta_val_ci": [vlo, vhi],
        "best_worst_n_val": [len(vb), len(vw)],
        "val_favoured_pos_quarters": f"{fav_pos}/{fav_counted}",
    })
    return rec


def _apply_survivor_criteria(rec: Dict[str, Any], fdr_reject: bool) -> str:
    """SPEC §5.4 — ALL must hold for `survivor`. Returns the verdict string."""
    if rec.get("verdict") == "thin":
        return "thin"
    d_disc = rec.get("delta_disc")
    dlo, dhi = rec.get("delta_disc_ci", [None, None])
    d_val = rec.get("delta_val")
    if d_disc is None or d_val is None:
        return "thin"

    # Criterion 3 (substance): both extreme buckets N≥150 in each split.
    nd = rec.get("top_bottom_n_disc") or rec.get("best_worst_n_disc") or [0, 0]
    nv = rec.get("top_bottom_n_val") or rec.get("best_worst_n_val") or [0, 0]
    substance_n = min(nd) >= MIN_BUCKET_N and min(nv) >= MIN_BUCKET_N
    substance_effect = abs(d_disc) >= MIN_EFFECT_R

    # Criterion 1 (discovery signal): BH-FDR reject AND ΔexpR CI excludes 0.
    disc_signal = fdr_reject and _ci_excludes_zero(dlo, dhi)

    # Criterion 2 (validation persistence): same sign AND favoured ≥60% quarters.
    same_sign = (d_disc > 0) == (d_val > 0) if (d_disc != 0 and d_val != 0) else False
    pos, counted = _parse_quarters(rec.get("val_favoured_pos_quarters", "0/0"))
    quarter_ok = counted > 0 and (pos / counted) >= QUARTER_SIGN_FRAC

    if not substance_n:
        return "directional_thin" if disc_signal else "noise"
    if not disc_signal:
        return "noise"
    if not substance_effect:
        return "directional_thin"
    if d_disc != 0 and d_val != 0 and not same_sign:
        return "inverted"
    if not (same_sign and quarter_ok):
        return "directional_thin"
    return "survivor"


def _parse_quarters(s: str) -> Tuple[int, int]:
    try:
        a, b = str(s).split("/")
        return int(a), int(b)
    except (ValueError, AttributeError):
        return 0, 0


def _actionable_class(feat: str, rec: Dict[str, Any]) -> str:
    """SPEC §13.1: tag every survivor entry_gate | order_rule | detection.
    Fill-time → order_rule. Structure/OB-quality alert-time features that would
    require changing WHAT gets detected → detection. Else entry_gate."""
    if _classify_timing(feat) == "fill_time":
        return "order_rule"
    detection_features = {
        "break_close_atr", "break_body_atr", "impulse_leg_atr", "fvg_size_atr",
        "ob_range_atr", "bos_tag", "bos_tier", "bos_verdict", "event",
        "reversed_from_extreme", "reversal_pct", "bos_sequence_count",
        "fvg_present", "fvg_mitigation", "fvg_state",
    }
    return "detection" if feat in detection_features else "entry_gate"


def _snapback_screen(pooled: pd.DataFrame) -> List[Dict[str, Any]]:
    """H-SNAPBACK pre-registered bins (SPEC §5.1c). bars_break_to_pullback binned
    {1-2,3-5,6-12,>12} on the pooled discovery+validation population, reported with
    N and the live-era under-sampling caveat for the 1-2 bin."""
    feat = "bars_break_to_pullback"
    out = []
    if feat not in pooled.columns:
        return out
    v = pd.to_numeric(pooled[feat], errors="coerce")
    for lo, hi in SNAPBACK_BINS:
        cell = pooled[(v >= lo) & (v <= hi)]
        st = _cell_stats(cell)
        out.append({"bin": f"{lo}-{hi if hi < 10**8 else 'inf'}",
                    "timing": "fill_time", **st,
                    "caveat": ("1-2 bin under-sampled: backtest alerts ~1 bar late "
                               "vs live forming-bar proximity; trust a POSITIVE "
                               "effect, distrust a null-on-thin"
                               if lo == 1 else "")})
    return out


def _sl_anatomy_screen(disc: pd.DataFrame, val: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """SPEC §5.5 stop-out anatomy. Population = eligible SL exits. For every
    feature, compare clean-break rate P(sl_bar_was_sweep==False) across buckets.
    Clean-break predictors = ENTRY-fault markers (auto-promote to Stage-2
    candidates, tag anatomy_promoted). Sweep-stop predictors = EXIT-fault markers
    (tag only, candidate Stage-3 cluster axis).

    `val is None` = DISCOVERY-ONLY mode (SPEC_STAGED §3): discovery stats only, no
    promotions — promotion is a validation concept, so `anatomy_promoted` is always
    `[]` here. Computed on the discovery SL frame; no validation frame is read."""
    disc_sl = disc[(disc["exit_reason"] == "sl") &
                   disc["sl_bar_was_sweep"].notna()].copy()
    rows: List[Dict[str, Any]] = []
    promoted: List[str] = []
    if disc_sl.empty:
        return {"rows": rows, "anatomy_promoted": promoted,
                "note": "no eligible SL-anatomy rows"}
    disc_sl["_clean"] = (disc_sl["sl_bar_was_sweep"] == False).astype(int)  # noqa: E712

    for feat in ALL_FEATURES:
        if feat not in disc_sl.columns:
            continue
        if feat in CONTINUOUS_FEATURES:
            edges = _discovery_quintile_edges(disc_sl, feat)
            if edges is None:
                continue
            dd = _feature_subpop(disc_sl, feat).copy()
            dd["_q"] = _assign_quintile(dd, feat, edges)
            n_b = len(edges) - 1
            top = dd[dd["_q"] == n_b - 1]["_clean"]
            bot = dd[dd["_q"] == 0]["_clean"]
            key_hi, key_lo = f"q{n_b-1}", "q0"
        else:
            dd = disc_sl.copy()
            dd[feat] = _merge_rare_levels(dd, feat, disc_sl)
            rates = dd.groupby(feat)["_clean"].agg(["mean", "count"])
            rates = rates[rates["count"] >= MIN_CELL_N]
            if len(rates) < 2:
                continue
            key_hi = str(rates["mean"].idxmax()); key_lo = str(rates["mean"].idxmin())
            top = dd[dd[feat].astype(str) == key_hi]["_clean"]
            bot = dd[dd[feat].astype(str) == key_lo]["_clean"]
        if len(top) < MIN_CELL_N or len(bot) < MIN_CELL_N:
            continue
        rate_diff = round(float(top.mean() - bot.mean()), 4)
        clo, chi = bootstrap_diff_ci(top.tolist(), bot.tolist(), paired=False)
        robust = _ci_excludes_zero(clo, chi)
        rows.append({"feature": feat, "hi": key_hi, "lo": key_lo,
                     "clean_rate_hi": round(float(top.mean()), 3),
                     "clean_rate_lo": round(float(bot.mean()), 3),
                     "rate_diff": rate_diff, "ci": [clo, chi],
                     "n_hi": len(top), "n_lo": len(bot),
                     "robust_clean_break_predictor": robust})
        # A robust clean-break predictor = ENTRY-fault → auto-promote to Stage 2.
        # Discovery-only mode (val is None) never promotes (SPEC_STAGED §3).
        if val is not None and robust and rate_diff > 0 and _classify_timing(feat) == "alert_time":
            promoted.append(feat)
    return {"rows": rows, "anatomy_promoted": sorted(set(promoted))}


def _news_confounder(pooled_sl_clean: pd.DataFrame, pooled_all: pd.DataFrame
                     ) -> Dict[str, Any]:
    """SPEC §5.5 news confounder (report-only, never a gate). % of clean-break SL
    exits within ±2h of a news_event_ts vs the same % among all eligible rows."""
    def _near_news_pct(df):
        if "news_event_ts" not in df.columns or df.empty:
            return None
        near = df["news_event_ts"].astype(str).str.strip().replace(
            {"": None, "nan": None, "None": None}).notna()
        return round(100.0 * near.sum() / len(df), 1)
    return {"clean_break_near_news_pct": _near_news_pct(pooled_sl_clean),
            "all_eligible_near_news_pct": _near_news_pct(pooled_all),
            "note": "context for reading anatomy; never a gate"}


def _interaction_screen(disc: pd.DataFrame, val: pd.DataFrame) -> List[Dict[str, Any]]:
    """SPEC §5.6 fixed pre-registered interactions. Cell N≥100, discovery CI
    excludes 0, same sign in validation. Flagged cells may become Stage-4 gates."""
    out: List[Dict[str, Any]] = []
    d = disc.copy(); v = val.copy()
    # derive break_close_atr quintile column (discovery edges) if needed.
    if any("break_close_atr_q" in pair for pair in INTERACTIONS):
        edges = _discovery_quintile_edges(disc, "break_close_atr")
        if edges is not None:
            d["break_close_atr_q"] = _assign_quintile(d, "break_close_atr", edges).astype(str)
            v["break_close_atr_q"] = _assign_quintile(v, "break_close_atr", edges).astype(str)
    for a, b in INTERACTIONS:
        if a not in d.columns or b not in d.columns:
            continue
        for (av, bv), g in d.groupby([a, b]):
            if len(g) < MIN_CELL_N:
                continue
            st = _cell_stats(g)
            if not st.get("ci_excludes_0"):
                continue
            vg = v[(v[a] == av) & (v[b] == bv)]
            vst = _cell_stats(vg)
            same_sign = (st["expR"] is not None and vst["expR"] is not None
                         and (st["expR"] > 0) == (vst["expR"] > 0))
            out.append({"interaction": f"{a} × {b}", "cell": f"{av} / {bv}",
                        "disc": st, "val": vst, "same_sign_val": same_sign})
    return out


# ═══════════════════════════════════════════════════════════════════════════
# STAGED HUMAN REVIEW (SPEC §18 / SPEC_STAGED) — the discovery→approve→confirm
# lock. Discovery is read freely; validation runs only behind a single-use token,
# and every spend is stamped forever in an append-only ledger + git history.
# ═══════════════════════════════════════════════════════════════════════════

# NOTE the wording: SPEC_STAGED §4.4 sketches this stamp with the word "survivor",
# but §4.5 + the §11.3 test forbid the literal words "survivor" / "edge" anywhere in
# Phase A output. §4.5/§11.3 is the binding rule (the point of the phase is to NOT
# claim survival yet), so the stamp keeps the meaning without the forbidden words:
# a candidate is only confirmed if it REPEATS on unseen validation years.
DISCOVERY_LANGUAGE_STAMP = (
    "CANDIDATE ONLY — discovery split only. Luck is NOT ruled out. A candidate is "
    "confirmed only if it REPEATS on validation years it has never seen.")


def _discovery_path(engine_dir: str) -> str:
    return os.path.join(engine_dir, "stage1_discovery.json")


def _approval_path(engine_dir: str) -> str:
    return os.path.join(engine_dir, "approval.json")


def _ledger_path(engine_dir: str) -> str:
    return os.path.join(engine_dir, "validation_ledger.jsonl")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _discovery_serialise(obj: Dict[str, Any]) -> bytes:
    """Serialise the discovery record the SAME way _write_json does (indent=2,
    default=str) so a hash taken here matches what lands on disk byte-for-byte."""
    return json.dumps(obj, indent=2, default=str).encode("utf-8")


def _compute_token(engine_dir: str) -> Optional[Dict[str, str]]:
    """SPEC_STAGED §5.1. token = sha256(discovery_sha + code_sha)[:12], where
    discovery_sha is the sha256 of the TOKEN-LESS serialisation of
    stage1_discovery.json and code_sha is the sha256 of edge_engine.py on disk.
    Returns None if the discovery file is absent."""
    disc = _read_json(_discovery_path(engine_dir))
    if disc is None:
        return None
    tokenless = {k: v for k, v in disc.items() if k != "token"}
    discovery_sha = _sha256_bytes(_discovery_serialise(tokenless))
    with open(os.path.abspath(__file__), "rb") as f:
        code_sha = _sha256_bytes(f.read())
    token = _sha256_bytes((discovery_sha + code_sha).encode("utf-8"))[:12]
    return {"token": token, "discovery_sha": discovery_sha, "code_sha": code_sha}


def approve(engine_dir: str, supplied_token: str) -> Dict[str, Any]:
    """SPEC_STAGED §5.2 — sign the discovery token. Recompute; refuse if it drifted.
    Writes approval.json (consumed:false) and returns a result dict for the CLI."""
    comp = _compute_token(engine_dir)
    if comp is None:
        return {"approved": False,
                "note": "no stage1_discovery.json — run --phase discovery first"}
    if comp["token"] != supplied_token:
        return {"approved": False, "refused": "token_mismatch",
                "note": ("REFUSED: discovery output or engine code changed since this "
                         "token was issued — re-run --phase discovery")}
    approval = {"token": comp["token"], "discovery_sha": comp["discovery_sha"],
                "code_sha": comp["code_sha"], "approved_utc": _now_utc(),
                "consumed": False}
    _write_json(_approval_path(engine_dir), approval)
    return {"approved": True,
            "note": "Approved. Validation is now armed for ONE confirmation run "
                    "(--phase confirm)."}


def _check_approval_gate(engine_dir: str) -> Tuple[bool, Dict[str, Any]]:
    """SPEC_STAGED §5.3.1 — gate passes iff approval.json exists, its token/
    discovery_sha/code_sha all match a fresh _compute_token, and consumed==false.
    Returns (passed, detail) — detail names which check failed for the refusal block.
    Reads only metadata; never touches any trade frame."""
    comp = _compute_token(engine_dir)
    if comp is None:
        return False, {"reason": "no_discovery", "note": "no stage1_discovery.json"}
    appr = _read_json(_approval_path(engine_dir))
    if appr is None:
        return False, {"reason": "no_approval", "note": "no approval.json"}
    if appr.get("consumed") is True:
        return False, {"reason": "consumed",
                       "note": "approval token already consumed (single-use)"}
    for k in ("token", "discovery_sha", "code_sha"):
        if appr.get(k) != comp[k]:
            return False, {"reason": f"{k}_mismatch",
                           "note": f"approval.json {k} != recomputed {k}"}
    return True, {"reason": "ok", "token": comp["token"], "code_sha": comp["code_sha"]}


def _append_ledger(engine_dir: str, via: str, token: Optional[str],
                   code_sha: str, burn_reason: Optional[str]) -> int:
    """SPEC_STAGED §5.3.4 — append ONE line to the append-only ledger; return the
    new line count (== validation_runs)."""
    line = {"run_utc": _now_utc(), "token": token, "code_sha": code_sha,
            "via": via, "burn_reason": burn_reason}
    os.makedirs(engine_dir, exist_ok=True)
    with open(_ledger_path(engine_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(line, default=str) + "\n")
    return _ledger_summary(engine_dir)["validation_runs"]


def _ledger_summary(engine_dir: str) -> Dict[str, Any]:
    """Read the append-only ledger: N runs, whether any was a burn, the via list."""
    path = _ledger_path(engine_dir)
    runs = 0
    burned = False
    vias: List[str] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                runs += 1
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                vias.append(rec.get("via", "?"))
                if rec.get("via") == "burn":
                    burned = True
    burned = burned or runs > 1
    return {"validation_runs": runs, "validation_burned": burned, "vias": vias}


def _refusal_block(engine_dir: str, detail: Dict[str, Any]) -> str:
    """The loud, human-readable refusal printed when the gate blocks validation."""
    comp = _compute_token(engine_dir)
    tok = comp["token"] if comp else "<run --phase discovery first>"
    return (
        "\033[31m╔══ VALIDATION REFUSED — approval gate (SPEC §18 / D5) ══╗\033[0m\n"
        f"  failed check: {detail.get('reason')} — {detail.get('note')}\n"
        "  Validation is a ONE-SHOT confirmation. It does not run until you\n"
        "  approve the discovery token. The staged workflow:\n"
        f"    1)  python -m backtest.diagnostics.edge_engine --phase discovery\n"
        f"    2)  python -m backtest.diagnostics.edge_engine --approve {tok}\n"
        f"    3)  python -m backtest.diagnostics.edge_engine --phase confirm\n"
        "  To deliberately re-open validation (stamped forever):\n"
        "    --burn-validation \"<written reason>\"\n")


def _apply_candidate_criteria(rec: Dict[str, Any], fdr_reject: bool) -> str:
    """SPEC_STAGED §4.3 — DISCOVERY-ONLY verdict ladder. Parallel to
    _apply_survivor_criteria but criterion 2 (validation persistence) does NOT
    exist here. Possible outputs: candidate, candidate_thin, noise, thin. The
    strings survivor / hypothesis / inverted are IMPOSSIBLE by construction.

    SIDE-EFFECT: stamps `rec["criteria"]` = the four pass/fail flags
    {fdr_reject, ci_excludes_0, substance_n, substance_effect} so the report's
    near-miss section reads them off the record instead of re-deriving thresholds
    (EDGE_DISCOVERY_REPORT_SPEC §4). A `candidate` passes all four; a near-miss
    fails exactly one. `thin`/None-Δ records carry no criteria (nothing to score)."""
    if rec.get("verdict") == "thin":
        return "thin"
    d_disc = rec.get("delta_disc")
    if d_disc is None:
        return "thin"
    dlo, dhi = rec.get("delta_disc_ci", [None, None])
    nd = rec.get("top_bottom_n_disc") or rec.get("best_worst_n_disc") or [0, 0]
    ci_ok = _ci_excludes_zero(dlo, dhi)
    substance_n = min(nd) >= MIN_BUCKET_N
    substance_eff = abs(d_disc) >= MIN_EFFECT_R
    rec["criteria"] = {
        "fdr_reject": bool(fdr_reject),
        "ci_excludes_0": bool(ci_ok),
        "substance_n": bool(substance_n),
        "substance_effect": bool(substance_eff),
    }
    disc_signal = fdr_reject and ci_ok
    if not substance_n:
        return "candidate_thin" if disc_signal else "noise"
    if not disc_signal:
        return "noise"
    if not substance_eff:
        return "candidate_thin"
    return "candidate"


def _population_stats(disc: pd.DataFrame) -> Dict[str, Any]:
    """Overall + per-pair + per-session discovery-split stats for the report's
    baseline-context section (EDGE_DISCOVERY_REPORT_SPEC §4/§5). Pure aggregation
    of the SAME discovery frame the screens use — reads no validation frame, adds
    no trades.csv column. Every row carries N so the report never prints a naked
    expR (blind-spot guard). The `overall`/`per_pair`/`per_session` rows are the
    gates-off, all-scores population — NOT what live (score≥4) trading produces;
    the report stamps that caveat, not this helper."""
    overall = _cell_stats(disc)
    per_pair = []
    if "pair" in disc.columns:
        for lvl, g in sorted(disc.groupby("pair"), key=lambda kv: str(kv[0])):
            per_pair.append({"pair": str(lvl), **_cell_stats(g)})
    per_session = []
    if "session" in disc.columns:
        for lvl, g in sorted(disc.groupby("session"), key=lambda kv: str(kv[0])):
            per_session.append({"session": str(lvl), **_cell_stats(g)})
    return {"overall": overall, "per_pair": per_pair, "per_session": per_session}


def stage1_discovery(run_dir: str, engine_dir: str, forced: bool) -> Dict[str, Any]:
    """PHASE A (SPEC_STAGED §4). Discovery-only preview of Stage 1. NEVER
    materialises the validation frame — the frame it would need is simply never
    built, so this function cannot leak. Output = CANDIDATES + an approval token.
    Verdict-scope only."""
    gate = _read_json(_stage_path(engine_dir, 0)) or {}
    scope = gate.get("scope", "exploratory")
    if scope != "verdict":
        raise SystemExit("staged review requires verdict scope; use the short-range "
                         "workflow (SPEC §15)")
    sl_anatomy_usable = gate.get("sl_anatomy_usable", False)
    news_usable = gate.get("news_usable", False)

    df = load_population(run_dir)
    sl_anatomy_usable = sl_anatomy_usable and all(c in df.columns for c in SL_ANATOMY_COLS)
    pooled = pooled_fx_gold(df)
    disc = split_frame(pooled, "DISCOVERY")
    # The validation / holdout frames are NEVER built past this line (SPEC_STAGED §4.2).

    buckets: List[Dict[str, Any]] = []
    feature_recs: List[Dict[str, Any]] = []
    for feat in CONTINUOUS_FEATURES:
        if feat in DECREED_OUT or feat not in disc.columns:
            continue
        feature_recs.append(_continuous_screen(disc, None, feat, buckets))
    for feat in CATEGORICAL_FEATURES:
        if feat in DECREED_OUT or feat not in disc.columns:
            continue
        feature_recs.append(_categorical_screen(disc, None, feat, buckets))

    fam_p = [r.get("_fdr_p") for r in feature_recs]
    reject = benjamini_hochberg(fam_p, FDR_Q)
    for r, rej in zip(feature_recs, reject):
        r["fdr_reject"] = bool(rej)
        r["verdict"] = _apply_candidate_criteria(r, rej)

    candidates = [r for r in feature_recs if r["verdict"] == "candidate"]
    # Rank by |delta_disc| (no delta_val exists yet — SPEC_STAGED §4.2).
    ranked = sorted(
        [r for r in feature_recs if r["verdict"] in ("candidate", "candidate_thin")],
        key=lambda r: -(abs(r.get("delta_disc") or 0.0)))

    snapback = _snapback_screen(pooled)
    anatomy = ({"note": "SL-anatomy columns absent (stale run)"}
               if not sl_anatomy_usable else _sl_anatomy_screen(disc, None))
    news = ({"note": "news unusable"} if not news_usable else _news_confounder(
        disc[(disc["exit_reason"] == "sl") &
             (disc.get("sl_bar_was_sweep") == False)]  # noqa: E712
        if sl_anatomy_usable else disc.iloc[0:0], disc))

    pd.DataFrame(buckets).to_csv(
        os.path.join(engine_dir, "stage1_discovery_features.csv"), index=False)

    pop_stats = _population_stats(disc)

    result: Dict[str, Any] = {
        "phase": "discovery", "pass": True, "forced": forced,
        "run_id": os.path.basename(run_dir), "generated_utc": _now_utc(),
        "scope": "verdict", "window": _window_label(disc),
        "n_discovery": int(len(disc)),
        "validation_untouched": True,
        "population_stats": pop_stats,
        "features": feature_recs,
        "candidates": [r["feature"] for r in candidates],
        "ranked_candidates": [{"feature": r["feature"], "verdict": r["verdict"],
                               "delta_disc": r.get("delta_disc"),
                               "timing": r["timing"]} for r in ranked],
        "snapback": snapback,
        "sl_anatomy": anatomy if isinstance(anatomy, dict) else {},
        "news_confounder": news,
        "interactions": "deferred_to_confirm",
        "language_stamp": DISCOVERY_LANGUAGE_STAMP,
    }
    # Token is computed on the TOKEN-LESS serialisation, then embedded (SPEC_STAGED
    # §5.1): write without token → hash → rewrite with token.
    _write_json(_discovery_path(engine_dir), result)
    comp = _compute_token(engine_dir)
    result["token"] = comp["token"]
    _write_json(_discovery_path(engine_dir), result)
    # Render the full-detail companion report (SPEC_STAGED §7/§9.1) AFTER the final
    # JSON (with token) lands, so it reads the committed bytes. Pure rendering — it
    # reads engine_dir only and builds no validation frame. The Action commits any
    # file in engine_dir (edge_engine.yml `git add -f "$RUN_DIR/edge_engine/"`), so
    # no workflow change is needed (F6).
    from backtest.diagnostics import edge_report
    edge_report.render_discovery_report(engine_dir)
    return result


def stage1(run_dir: str, engine_dir: str, forced: bool,
           burn_reason: Optional[str] = None) -> Dict[str, Any]:
    gate = _read_json(_stage_path(engine_dir, 0)) or {}
    scope = gate.get("scope", "exploratory")
    news_usable = gate.get("news_usable", False)
    sl_anatomy_usable = gate.get("sl_anatomy_usable", False)

    # ── APPROVAL GATE (SPEC §18 / SPEC_STAGED §5.3) — verdict scope only. This
    # runs BEFORE any validation frame is built. In verdict scope, validation is a
    # one-shot confirmation armed by a single-use token; --burn-validation is the
    # only sanctioned re-open and is stamped forever. Exploratory scope never hits
    # the gate (§15 has no validation to protect). ──
    if scope == "verdict":
        gate_ok, gdetail = _check_approval_gate(engine_dir)
        burning = burn_reason is not None and burn_reason.strip() != ""
        if not gate_ok and not burning:
            print(_refusal_block(engine_dir, gdetail))
            result = {"stage": 1, "pass": False, "refused": "approval_gate",
                      "gate_detail": gdetail, "run_id": os.path.basename(run_dir),
                      "generated_utc": _now_utc(), "scope": scope}
            # Deliberately NOT written to _stage_path(1) — the canonical artefact
            # must stay absent so Stage 2 cannot chain off a refused Stage 1.
            return result
        code_sha = _compute_token(engine_dir)
        code_sha = code_sha["code_sha"] if code_sha else _sha256_bytes(
            open(os.path.abspath(__file__), "rb").read())
        if gate_ok:
            appr = _read_json(_approval_path(engine_dir)) or {}
            appr["consumed"] = True  # single-use
            _write_json(_approval_path(engine_dir), appr)
            _append_ledger(engine_dir, "approval", gdetail.get("token"),
                           code_sha, None)
        else:  # burning
            _append_ledger(engine_dir, "burn", None, code_sha, burn_reason.strip())

    df = load_population(run_dir)
    # Re-derive column availability from the DATA WE ACTUALLY LOADED, never trust
    # the gate's stamped flag alone — the gate JSON on disk can be stale (written
    # by a prior Stage 0 on different data). This closes a stale-gate blind spot.
    sl_anatomy_usable = sl_anatomy_usable and all(c in df.columns for c in SL_ANATOMY_COLS)
    pooled = pooled_fx_gold(df)
    disc = split_frame(pooled, "DISCOVERY")
    val = split_frame(pooled, "VALIDATION")

    # In EXPLORATORY scope there is no discovery/validation confirmation — screen
    # the WHOLE pooled range as one population (SPEC §15.2). We reuse the same
    # machinery by pointing both "splits" at the pool; survivor criteria that
    # depend on validation persistence are relaxed to informational tags.
    exploratory = scope != "verdict"
    if exploratory:
        disc = pooled.copy()
        val = pooled.copy()

    buckets: List[Dict[str, Any]] = []
    feature_recs: List[Dict[str, Any]] = []
    for feat in CONTINUOUS_FEATURES:
        if feat in DECREED_OUT or feat not in disc.columns:
            continue
        feature_recs.append(_continuous_screen(disc, val, feat, buckets))
    for feat in CATEGORICAL_FEATURES:
        if feat in DECREED_OUT or feat not in disc.columns:
            continue
        feature_recs.append(_categorical_screen(disc, val, feat, buckets))

    # BH-FDR across the WHOLE feature family on discovery p-values (SPEC §5.4).
    fam_p = [r.get("_fdr_p") for r in feature_recs]
    reject = benjamini_hochberg(fam_p, FDR_Q)
    for r, rej in zip(feature_recs, reject):
        r["fdr_reject"] = bool(rej)
        v = _apply_survivor_criteria(r, rej)
        # In exploratory mode a "survivor" is only a HYPOTHESIS — never shippable.
        if exploratory and v == "survivor":
            v = "hypothesis"
        r["verdict"] = v
        if v in ("survivor", "hypothesis"):
            r["actionable_at"] = _actionable_class(feat=r["feature"], rec=r)

    survivors = [r for r in feature_recs if r["verdict"] == "survivor"]

    # Sub-screens.
    snapback = _snapback_screen(pooled)
    anatomy = ({"note": "SL-anatomy columns absent (stale run)"}
               if not sl_anatomy_usable else _sl_anatomy_screen(disc, val))
    # anatomy-promoted features become Stage-2 candidates even if they missed §5.4.
    anatomy_promoted = anatomy.get("anatomy_promoted", []) if sl_anatomy_usable else []
    news = ({"note": "news unusable"} if not news_usable else _news_confounder(
        pooled[(pooled["exit_reason"] == "sl") &
               (pooled.get("sl_bar_was_sweep") == False)]  # noqa: E712
        if sl_anatomy_usable else pooled.iloc[0:0],
        pooled))
    interactions = _interaction_screen(disc, val) if not exploratory else []

    # Rank survivor table by validation ΔexpR (SPEC §5.4), not by p.
    def _rank_key(r):
        return -(abs(r.get("delta_val") or 0.0))
    ranked = sorted(
        [r for r in feature_recs if r["verdict"] in ("survivor", "hypothesis")],
        key=_rank_key)

    pd.DataFrame(buckets).to_csv(
        os.path.join(engine_dir, "stage1_features.csv"), index=False)
    if isinstance(anatomy, dict) and anatomy.get("rows"):
        pd.DataFrame(anatomy["rows"]).to_csv(
            os.path.join(engine_dir, "stage1_sl_anatomy.csv"), index=False)

    result = {
        "stage": 1, "pass": True, "forced": forced,
        "run_id": os.path.basename(run_dir), "generated_utc": _now_utc(),
        "scope": scope, "exploratory": exploratory,
        "window": _window_label(pooled),
        "n_pooled": int(len(pooled)),
        "n_discovery": int(len(disc)) if not exploratory else None,
        "n_validation": int(len(val)) if not exploratory else None,
        "features": feature_recs,
        "survivors": [r["feature"] for r in survivors],
        "ranked_actionable": [{"feature": r["feature"], "verdict": r["verdict"],
                               "actionable_at": r.get("actionable_at"),
                               "delta_val": r.get("delta_val"),
                               "timing": r["timing"]} for r in ranked],
        "anatomy_promoted": anatomy_promoted,
        "detection_queue": [r["feature"] for r in ranked
                            if r.get("actionable_at") == "detection"],
        "snapback": snapback,
        "sl_anatomy": anatomy if isinstance(anatomy, dict) else {},
        "news_confounder": news,
        "interactions": interactions,
    }
    # Validation-spend stamps (SPEC §18 §5.4) — loud like holdout_reopened. In
    # exploratory scope the gate never ran, so there is no ledger and no spend.
    if not exploratory:
        result.update(_ledger_summary(engine_dir))
    _write_json(_stage_path(engine_dir, 1), result)
    return result


def _window_label(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    a = _to_utc(df["alert_ts"])
    return f"{a.min().date()}..{a.max().date()}"


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — THE EV SCORE (multivariate; SPEC §6). numpy/scipy only — no sklearn.
#
# Ridge regression on r_realised (closed-form (XᵀX+λI)⁻¹Xᵀy, intercept
# unpenalised). λ by 5-fold CONTIGUOUS-TIME CV inside discovery. L2 logistic
# (IRLS) as a rank cross-check. VIF>5 → drop the weaker-ΔexpR member. Sign sanity
# vs Stage-1 direction. Calibration + pass bar on VALIDATION. ALERT-TIME survivors
# + anatomy_promoted only (fill-time never enters — look-ahead leak). Empty union
# → NO_ENTRY_SIGNAL (honest null, pass:true). Does not run in exploratory scope.
# ═══════════════════════════════════════════════════════════════════════════

def _build_design(df: pd.DataFrame, feats: List[str],
                  spec: Optional[Dict[str, Any]] = None
                  ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """One-hot categoricals (levels ≥5% discovery freq, drop-first, rest 'other');
    standardise continuous on discovery mean/std. If `spec` is given (from
    discovery), APPLY it to a later split (frozen encoding). Else BUILD it.
    Returns (X, column_names, spec)."""
    building = spec is None
    if building:
        spec = {"continuous": {}, "categorical": {}, "columns": []}
    cols: List[np.ndarray] = []
    names: List[str] = []
    for f in feats:
        if f not in df.columns:
            continue
        if f in CONTINUOUS_FEATURES:
            x = pd.to_numeric(df[f], errors="coerce")
            if building:
                mu = float(np.nanmean(x)); sd = float(np.nanstd(x)) or 1.0
                spec["continuous"][f] = {"mean": mu, "std": sd}
            mu = spec["continuous"][f]["mean"]; sd = spec["continuous"][f]["std"]
            z = ((x - mu) / sd).fillna(0.0).to_numpy(dtype=float)
            cols.append(z); names.append(f)
        else:
            s = df[f].astype("object")
            if building:
                freq = s.value_counts(normalize=True)
                levels = [str(k) for k, v in freq.items() if v >= 0.05]
                levels = sorted(levels)
                drop = levels[0] if levels else None  # drop-first
                spec["categorical"][f] = {"levels": levels, "drop": drop}
            levels = spec["categorical"][f]["levels"]
            drop = spec["categorical"][f]["drop"]
            sv = s.map(lambda v: str(v) if str(v) in levels else "other")
            for lvl in levels:
                if lvl == drop:
                    continue
                cols.append((sv == lvl).astype(float).to_numpy())
                names.append(f"{f}={lvl}")
    if building:
        spec["columns"] = names
    else:
        # Reindex to the frozen column order (missing dummies → 0).
        frozen = spec["columns"]
        idx = {n: i for i, n in enumerate(names)}
        realigned = []
        for n in frozen:
            realigned.append(cols[idx[n]] if n in idx else np.zeros(len(df)))
        cols, names = realigned, frozen
    X = np.column_stack(cols) if cols else np.zeros((len(df), 0))
    return X, names, spec


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge, intercept unpenalised. Returns beta incl. intercept
    (col 0). X is (n,p); design gets a leading ones column here."""
    n = X.shape[0]
    Xi = np.column_stack([np.ones(n), X])
    p = Xi.shape[1]
    R = np.eye(p) * lam
    R[0, 0] = 0.0  # do not penalise the intercept
    beta = np.linalg.solve(Xi.T @ Xi + R, Xi.T @ y)
    return beta


def _ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X]) @ beta


def _cv_lambda(X: np.ndarray, y: np.ndarray, years: np.ndarray) -> float:
    """5-fold CONTIGUOUS-TIME CV (folds = consecutive year blocks). Picks the λ
    with the lowest mean held-out MSE. Random folds would leak regime."""
    uniq_years = sorted(set(int(v) for v in years))
    if len(uniq_years) < 2:
        return 1.0
    # split years into up to 5 consecutive blocks
    k = min(5, len(uniq_years))
    blocks = np.array_split(uniq_years, k)
    best_lam, best_mse = 1.0, math.inf
    for lam in RIDGE_LAMBDAS:
        mses = []
        for blk in blocks:
            test_mask = np.isin(years, blk)
            if test_mask.sum() < 5 or (~test_mask).sum() < 10:
                continue
            beta = _ridge_fit(X[~test_mask], y[~test_mask], lam)
            pred = _ridge_predict(X[test_mask], beta)
            mses.append(float(np.mean((pred - y[test_mask]) ** 2)))
        if mses and np.mean(mses) < best_mse:
            best_mse, best_lam = float(np.mean(mses)), lam
    return best_lam


def _logistic_irls(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 30
                   ) -> np.ndarray:
    """L2 logistic via IRLS (intercept unpenalised). y ∈ {0,1}. ~30 lines."""
    n = X.shape[0]
    Xi = np.column_stack([np.ones(n), X])
    p = Xi.shape[1]
    beta = np.zeros(p)
    R = np.eye(p) * lam
    R[0, 0] = 0.0
    for _ in range(iters):
        eta = Xi @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = np.clip(mu * (1 - mu), 1e-6, None)
        z = eta + (y - mu) / w
        WX = Xi * w[:, None]
        try:
            beta_new = np.linalg.solve(Xi.T @ WX + R, Xi.T @ (w * z))
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _vif(X: np.ndarray, names: List[str]) -> Dict[str, float]:
    """Variance inflation factor per column (numpy). VIF_i = 1/(1-R²_i) from
    regressing column i on the rest."""
    out: Dict[str, float] = {}
    p = X.shape[1]
    for i in range(p):
        others = np.delete(X, i, axis=1)
        if others.shape[1] == 0:
            out[names[i]] = 1.0
            continue
        Oi = np.column_stack([np.ones(X.shape[0]), others])
        try:
            beta = np.linalg.lstsq(Oi, X[:, i], rcond=None)[0]
            pred = Oi @ beta
            ss_res = float(np.sum((X[:, i] - pred) ** 2))
            ss_tot = float(np.sum((X[:, i] - X[:, i].mean()) ** 2)) or 1e-12
            r2 = 1 - ss_res / ss_tot
            out[names[i]] = round(1.0 / max(1 - r2, 1e-6), 3)
        except np.linalg.LinAlgError:
            out[names[i]] = float("inf")
    return out


def _feature_of_column(colname: str) -> str:
    """'bos_tag=CHoCH' -> 'bos_tag'; 'break_close_atr' -> 'break_close_atr'."""
    return colname.split("=")[0]


def stage2(run_dir: str, engine_dir: str, forced: bool) -> Dict[str, Any]:
    gate = _read_json(_stage_path(engine_dir, 0)) or {}
    s1 = _read_json(_stage_path(engine_dir, 1)) or {}
    scope = gate.get("scope", "exploratory")

    base = {"stage": 2, "pass": True, "forced": forced,
            "run_id": os.path.basename(run_dir), "generated_utc": _now_utc(),
            "scope": scope}

    if scope != "verdict":
        base.update({"verdict": "SKIPPED_EXPLORATORY",
                     "note": "Stage 2 EV model does not run in exploratory scope "
                             "(SPEC §15.2) — no split to fit/calibrate against."})
        _write_json(_stage_path(engine_dir, 2), base)
        return base

    # Inputs: ALERT-TIME survivors + anatomy_promoted (§6.1). Fill-time excluded.
    survivor_feats = [r["feature"] for r in s1.get("features", [])
                      if r.get("verdict") == "survivor"
                      and _classify_timing(r["feature"]) == "alert_time"]
    promoted = [f for f in s1.get("anatomy_promoted", [])
                if _classify_timing(f) == "alert_time"]
    feats = sorted(set(survivor_feats) | set(promoted))

    if not feats:
        base.update({"verdict": "NO_ENTRY_SIGNAL",
                     "note": "no alert-time survivors — honest null; Stage 3 runs "
                             "with fallback clusters (SPEC §6.1)."})
        _write_json(_stage_path(engine_dir, 2), base)
        return base

    df = load_population(run_dir)
    pooled = pooled_fx_gold(df)
    disc = split_frame(pooled, "DISCOVERY")
    val = split_frame(pooled, "VALIDATION")

    Xd, names, spec = _build_design(disc, feats)
    yd = pd.to_numeric(disc["r_realised"], errors="coerce").fillna(0.0).to_numpy()
    years = _to_utc(disc["alert_ts"]).dt.year.to_numpy()

    # VIF hygiene: drop VIF>5 members with weaker Stage-1 validation |ΔexpR|.
    s1_delta = {r["feature"]: abs(r.get("delta_val") or 0.0)
                for r in s1.get("features", [])}
    dropped_vif: List[Dict[str, Any]] = []
    while Xd.shape[1] > 1:
        vifs = _vif(Xd, names)
        worst = max(vifs, key=lambda k: vifs[k])
        if vifs[worst] <= VIF_MAX:
            break
        # among the correlated group (all VIF>5), drop the weakest-ΔexpR feature.
        high = [n for n in names if vifs[n] > VIF_MAX]
        drop_col = min(high, key=lambda n: s1_delta.get(_feature_of_column(n), 0.0))
        j = names.index(drop_col)
        dropped_vif.append({"column": drop_col, "vif": vifs[drop_col]})
        Xd = np.delete(Xd, j, axis=1); names.pop(j)
        spec["columns"] = names

    lam = _cv_lambda(Xd, yd, years)
    beta = _ridge_fit(Xd, yd, lam)
    coef = {n: round(float(b), 5) for n, b in zip(names, beta[1:])}
    intercept = round(float(beta[0]), 5)

    # Sign sanity: each coef sign must match its Stage-1 bucket direction; a flip
    # after VIF hygiene → drop (SPEC §6.2). Direction from Stage-1 delta_disc.
    s1_dir = {r["feature"]: (1 if (r.get("delta_disc") or 0) >= 0 else -1)
              for r in s1.get("features", [])}
    sign_dropped: List[str] = []
    for n in list(names):
        f = _feature_of_column(n)
        want = s1_dir.get(f)
        got = 1 if coef[n] >= 0 else -1
        # Only continuous features have an unambiguous single direction to check.
        if f in CONTINUOUS_FEATURES and want is not None and coef[n] != 0 and got != want:
            sign_dropped.append(n)
    if sign_dropped:
        keep = [i for i, n in enumerate(names) if n not in sign_dropped]
        Xd = Xd[:, keep]; names = [names[i] for i in keep]
        spec["columns"] = names
        beta = _ridge_fit(Xd, yd, lam)
        coef = {n: round(float(b), 5) for n, b in zip(names, beta[1:])}
        intercept = round(float(beta[0]), 5)

    # Secondary logistic (rank cross-check on resolved trades, BE excluded).
    resolved = disc[disc["r_realised"] != 0].copy()
    models_disagree = None
    if len(resolved) >= 20:
        Xr, _, _ = _build_design(resolved, feats, spec)
        yr = (pd.to_numeric(resolved["r_realised"], errors="coerce") > 0).astype(int).to_numpy()
        blog = _logistic_irls(Xr, yr, lam)
        logit_pred = _ridge_predict(Xr, blog)  # linear predictor rank
        ridge_pred_r = _ridge_predict(Xr, beta)
        if _HAS_SCIPY and len(set(yr)) > 1:
            rho, _ = _spearmanr(ridge_pred_r, logit_pred)
            models_disagree = bool(rho < 0.5) if not math.isnan(rho) else None

    # Calibration + pass bar on VALIDATION.
    Xv, _, _ = _build_design(val, feats, spec)
    ev_val = _ridge_predict(Xv, beta)
    yv = pd.to_numeric(val["r_realised"], errors="coerce").fillna(0.0).to_numpy()
    calib = _decile_calibration(ev_val, yv, val)
    passed, passbar = _stage2_pass_bar(ev_val, yv, val, disc, beta, spec, feats)

    # legacy score baseline receipt (report-only).
    legacy_rho = None
    if "score" in val.columns and _HAS_SCIPY:
        sc = pd.to_numeric(val["score"], errors="coerce")
        m = sc.notna()
        if m.sum() > 10 and sc[m].nunique() > 1:
            legacy_rho, _ = _spearmanr(sc[m], yv[m.to_numpy()])
            legacy_rho = round(float(legacy_rho), 4)

    verdict = "EV_MODEL" if passed else "NO_USABLE_EV"
    result = {
        **base, "verdict": verdict,
        "features_in": feats,
        "ev_model": {
            "type": "ridge", "lambda": lam, "columns": names,
            "coefficients": coef, "intercept": intercept,
            "standardization": spec["continuous"],
            "categorical_encoding": spec["categorical"],
        } if passed else None,
        "vif_dropped": dropped_vif,
        "sign_dropped": sign_dropped,
        "models_disagree": models_disagree,
        "calibration_deciles": calib,
        "pass_bar": passbar,
        "legacy_score_spearman_val": legacy_rho,
        "note": ("EV ships past the §6.3 bar" if passed else
                 "did not clear the §6.3 pass bar — honest null"),
    }
    _write_json(_stage_path(engine_dir, 2), result)
    return result


def _decile_calibration(ev: np.ndarray, y: np.ndarray, val: pd.DataFrame
                        ) -> List[Dict[str, Any]]:
    """Predicted-EV decile → N, realised expR, CI, WR (SPEC §6.3)."""
    if len(ev) < 10:
        return []
    edges = np.quantile(ev, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    if len(edges) < 3:
        return []
    dec = np.clip(np.digitize(ev, edges[1:-1]), 0, len(edges) - 2)
    out = []
    tmp = val.copy()
    tmp["_ev_dec"] = dec
    for d in sorted(set(dec)):
        cell = tmp[tmp["_ev_dec"] == d]
        st = _cell_stats(cell)
        out.append({"decile": int(d), **st,
                    "pred_ev_mean": round(float(ev[dec == d].mean()), 4)})
    return out


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — SETUP-CONDITIONAL EXIT OPTIMISATION (SPEC §7)
#
# Post-hoc replay (exit_lab pattern). FROZEN recipe grid (~40, never extended
# mid-run). Cost model charges spread on non-initial-SL legs (SL pre-paid via the
# widened stop). Clusters = EV quintiles (if Stage 2 passed) or event×pd_zone
# fallback. Selection = per-trade PAIRED difference vs baseline on discovery (the
# variance killer), CI lo > 0; validation single-confirm; honest nulls ship
# baseline. Per-cluster only ships if it beats the global recipe. Runs in both
# scopes (exploratory = discovery-only descriptives, no validation confirm).
# ═══════════════════════════════════════════════════════════════════════════

BASELINE_RECIPE = {"legs": [(1.0, "tp1")], "be_trigger_r": 1.0, "be_to_r": 0.0}


# Wider-stop k fallback (ATR multiples added to sl_initial) when the run's CSV lacks
# sl_wick_depth_atr — a stamped default set, NOT a data read (flagged in the recipe
# grid meta). The live path derives k from the wick-depth distribution instead (§7).
WIDEN_K_FALLBACK = [0.25, 0.5, 0.75, 1.0]
# Trailing candidates (distance kept behind best closed-bar extreme, in R) × arm point.
TRAIL_RS = [1.0, 1.5, 2.0]
TRAIL_ARMS = [0.0, 1.0]


def _widen_k_candidates(df: Optional[pd.DataFrame]) -> Tuple[List[float], str]:
    """k·ATR wider-stop candidates, SIZED FROM DATA (SPEC §7/§15): the k set is
    anchored to the sl_wick_depth_atr distribution so the replay sweeps the range
    wicks actually pierce, then PICKS the winner. Returns (k_list, source_note).

    Uses the p25/p50/p75/p90 wick-depth percentiles (rounded to 0.05R) as the k
    candidates — 'how far do losers' stop wicks actually poke through?'. Falls back
    to WIDEN_K_FALLBACK, stamped, when the column is absent (old CSV) or too thin."""
    if df is None or "sl_wick_depth_atr" not in getattr(df, "columns", []):
        return list(WIDEN_K_FALLBACK), "fallback (sl_wick_depth_atr absent)"
    w = pd.to_numeric(df["sl_wick_depth_atr"], errors="coerce").dropna()
    w = w[w > 0]
    if len(w) < MIN_BUCKET_N:
        return list(WIDEN_K_FALLBACK), f"fallback (only {len(w)} wick-depth values)"
    ks = sorted({round(round(float(np.percentile(w, p)) / 0.05) * 0.05, 2)
                 for p in (25, 50, 75, 90)})
    ks = [k for k in ks if k >= 0.05]  # drop a degenerate ~0 candidate
    if not ks:
        return list(WIDEN_K_FALLBACK), "fallback (wick-depth percentiles ~0)"
    return ks, f"data-derived from sl_wick_depth_atr p25/50/75/90 (n={len(w)})"


def _build_recipe_grid(widen_ks: Optional[List[float]] = None,
                       trail: bool = True) -> Dict[str, Dict[str, Any]]:
    """SPEC §7.3/§7 — the frozen grid. Full-position TP×BE (skip BE≥numeric-TP),
    four partial families, baseline, PLUS the exit-track levers: wider-stop variants
    (sl_widen_atr = k·ATR, k sized from data by the caller) and trailing-stop
    variants. `widen_ks` None = no wider-stop recipes; `trail` False = no trail
    recipes (keeps the original ~40-recipe grid for callers that don't want them)."""
    grid: Dict[str, Dict[str, Any]] = {}
    tps = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, "tp1"]
    bes = [None, 0.3, 0.5, 0.7, 1.0]
    for tp in tps:
        for be in bes:
            if be is not None and isinstance(tp, (int, float)) and be >= tp:
                continue  # BE can't arm before the target
            name = f"full_tp{tp}_be{be}"
            grid[name] = {"legs": [(1.0, tp)], "be_trigger_r": be,
                          "be_to_r": 0.0 if be is not None else None}
    # Partials (50/50) with their pre-registered BE sets.
    partials = [
        ("p_1.0_tp1", [(0.5, 1.0), (0.5, "tp1")], [None, 0.5, 1.0]),
        ("p_0.5_1.5", [(0.5, 0.5), (0.5, 1.5)], [None, 0.3, 0.5]),
        ("p_1.0_2.0", [(0.5, 1.0), (0.5, 2.0)], [None, 0.5, 1.0]),
        ("p_0.5_tp1", [(0.5, 0.5), (0.5, "tp1")], [None, 0.3, 0.5]),
    ]
    for pfx, legs, be_set in partials:
        for be in be_set:
            grid[f"{pfx}_be{be}"] = {"legs": legs, "be_trigger_r": be,
                                     "be_to_r": 0.0 if be is not None else None}

    # ── WIDER-STOP variants (exit-track lever, SPEC §7). Take the liquidity-TP
    # full-position leg and re-run it on a k·ATR-wider stop, both no-BE and BE@1R
    # (BE is the OTHER live sweep-killer; pair them so the replay can separate them).
    for k in (widen_ks or []):
        ks = f"{k:g}"
        grid[f"widen{ks}_tp1_beNone"] = {
            "legs": [(1.0, "tp1")], "be_trigger_r": None, "be_to_r": None,
            "sl_widen_atr": k}
        grid[f"widen{ks}_tp1_be1.0"] = {
            "legs": [(1.0, "tp1")], "be_trigger_r": 1.0, "be_to_r": 0.0,
            "sl_widen_atr": k}

    # ── TRAILING-STOP variants (exit-track lever #5). Liquidity-TP leg, no fixed
    # BE (the trail IS the dynamic stop), swept over trail distance × arm point.
    if trail:
        for tr in TRAIL_RS:
            for arm in TRAIL_ARMS:
                grid[f"trail{tr:g}_arm{arm:g}"] = {
                    "legs": [(1.0, "tp1")], "be_trigger_r": None, "be_to_r": None,
                    "trail_r": tr, "trail_arm_r": arm}

    grid["baseline"] = dict(BASELINE_RECIPE)
    return grid


def _cost_r_for_trade(t: pd.Series, pair_conf_map: Dict[str, Dict[str, Any]]) -> float:
    """cost_r = (spread_pips × pip_size) / r_distance (SPEC §7.2). Mirrors the
    simulator's pip-size derivation exactly."""
    pc = pair_conf_map.get(t["pair"], {})
    spread_pips = float(pc.get("spread_pips", 0.0))
    pip = _pip_size(pc)
    r_distance = t.get("r_distance")
    if r_distance is None:
        try:
            r_distance = abs(float(t["entry"]) - float(t["sl_initial"]))
        except (TypeError, ValueError, KeyError):
            return 0.0
    if r_distance <= 0:
        return 0.0
    return (spread_pips * pip) / r_distance


def _net_r_of_legs(legs: List[Dict[str, Any]], bias: str, entry: float,
                   traded_sl: float, r_distance: float, cost_r: float) -> float:
    """Frac-weighted net R. Charge cost_r on every leg EXCEPT one that exited at the
    TRADED stop (reason 'sl', exit_price == traded_sl) — that fill pre-paid its spread
    because every stop the simulator hands us is already spread-widened by one spread
    in the adverse direction (h1_only_simulator sl_initial = sl_raw ∓ spread). A
    wider-stop replay stops out at `traded_sl` (sl_initial pushed a further k·ATR out,
    STILL carrying that one baked-in spread) → still pre-paid, still no double-charge.

    SPEC §7.2 (corrected): the pre-payment predicate is the STOP ACTUALLY TRADED in
    this replay, NOT the original sl_initial. The old `== sl_initial` check wrongly
    charged spread on a widened-stop loser (its exit price ≠ sl_initial). BE / trail
    stops land at a computed R-level, not a spread-widened price → they ARE charged."""
    net = 0.0
    for lg in legs:
        r = ((lg["exit_price"] - entry) if bias == "LONG"
             else (entry - lg["exit_price"])) / r_distance
        pre_paid = (lg.get("reason") == "sl"
                    and abs(float(lg["exit_price"]) - traded_sl) < 1e-9)
        leg_net = r - (0.0 if pre_paid else cost_r)
        net += lg["frac"] * leg_net
    return net


def _replay_grid(trades: pd.DataFrame, grid: Dict[str, Dict[str, Any]],
                 pair_conf_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """One row per (trade, recipe): net R (cost-adjusted), gross R, exit reason,
    bars_to_exit, forced-close flag. Bars loaded once (exit_lab pattern)."""
    bars = _ensure_bars(trades)
    if not bars:
        return pd.DataFrame()
    max_hold = sim.MAX_HOLD_H1_BARS
    wk_flat = sim.WEEKEND_FLAT
    wk_hour = sim.WEEKEND_FLAT_HOUR_UTC
    rows: List[Dict[str, Any]] = []
    for _, t in trades.iterrows():
        pb = bars.get(t["pair"])
        if pb is None:
            continue
        fill_ts = pd.to_datetime(t["fill_ts"], utc=True)
        future = pb.loc[pb.index >= fill_ts]
        if future.empty:
            continue
        future = future.iloc[: max_hold + 2]
        bias = t["bias"] if t.get("bias") in ("LONG", "SHORT") else (
            "LONG" if t.get("direction") == "bullish" else "SHORT")
        try:
            entry = float(t["entry"]); sl = float(t["sl_initial"]); tp1 = float(t["tp1"])
        except (TypeError, ValueError):
            continue
        r_distance = abs(entry - sl)
        if r_distance <= 0:
            continue
        try:
            atr_at_ob = float(t["atr_at_ob"])
        except (TypeError, ValueError, KeyError):
            atr_at_ob = None
        cost_r = _cost_r_for_trade(t, pair_conf_map)  # spread as a fraction of the ORIGINAL 1R
        for name, recipe in grid.items():
            rec = {k: v for k, v in recipe.items() if v is not None or k != "be_to_r"}
            # WIDER STOP (SPEC §7): push sl_initial a further k·ATR into the loss and
            # rescale 1R to the new risk. sl_initial already carries one baked-in
            # spread; widening only moves it further out, so it stays spread-inclusive
            # (pre-paid on a stop-out). A recipe with no sl_widen_atr (or a trade
            # missing atr_at_ob) trades the original stop unchanged.
            k = rec.pop("sl_widen_atr", None)
            traded_sl, traded_r = sl, r_distance
            if k and atr_at_ob and atr_at_ob > 0:
                traded_sl = (sl - k * atr_at_ob) if bias == "LONG" else (sl + k * atr_at_ob)
                traded_r = abs(entry - traded_sl)
            # Spread is a FIXED price; net R is in units of traded_r, so the cost in
            # those units shrinks as the stop widens: cost_r_traded = cost_r · (R/traded_r).
            cost_r_traded = cost_r * (r_distance / traded_r) if traded_r > 0 else cost_r
            res = walk_multileg(future, bias, entry, traded_sl, traded_r, tp1, rec,
                                weekend_flat=wk_flat, weekend_hour_utc=wk_hour,
                                max_hold=max_hold)
            net = _net_r_of_legs(res["legs"], bias, entry, traded_sl, traded_r, cost_r_traded)
            forced = res["exit_reason"] in ("timeout", "window_end", "friday_flat")
            rows.append({
                "setup_id": t.get("setup_id"), "pair": t["pair"],
                "recipe": name, "_split": t.get("_split"),
                "_quarter": t.get("_quarter"), "alert_ts": t["alert_ts"],
                "net_r": round(net, 4), "gross_r": round(float(res["r_realised"]), 4),
                "exit_reason": res["exit_reason"], "bars_to_exit": res["bars_to_exit"],
                "forced_close": forced, "cost_r": round(cost_r, 5),
            })
    return pd.DataFrame(rows)


def _paired_diff_select(rep: pd.DataFrame, recipes: List[str],
                        split: str) -> Optional[Dict[str, Any]]:
    """SPEC §7.5: pick the recipe with max net expR whose PAIRED-diff bootstrap CI
    vs baseline (lo>0) on `split`. Paired by setup_id (same trades). Tie-break:
    fewer legs > no-BE > rounder numbers (simplicity by decree)."""
    sub = rep[rep["_split"] == split] if split else rep
    piv = sub.pivot_table(index="setup_id", columns="recipe", values="net_r")
    if "baseline" not in piv.columns:
        return None
    base = piv["baseline"]
    cands = []
    for r in recipes:
        if r == "baseline" or r not in piv.columns:
            continue
        both = piv[[r]].join(base.rename("baseline")).dropna()
        if len(both) < 5:
            continue
        diff = (both[r] - both["baseline"]).to_numpy()
        lo, hi = bootstrap_diff_ci(both[r].tolist(), both["baseline"].tolist(),
                                   paired=True)
        cands.append({"recipe": r, "net_expR": round(float(both[r].mean()), 4),
                      "paired_diff": round(float(diff.mean()), 4),
                      "diff_ci": [lo, hi], "n": len(both),
                      "beats_baseline": _ci_excludes_zero(lo, hi) and lo > 0})
    winners = [c for c in cands if c["beats_baseline"]]
    if not winners:
        return None
    max_exp = max(c["net_expR"] for c in winners)
    leaders = [c for c in winners if c["net_expR"] >= max_exp - 1e-9
               or _overlaps(c, winners, max_exp)]
    leaders.sort(key=lambda c: _tiebreak_key(c["recipe"], -c["net_expR"]))
    return leaders[0]


def _overlaps(c, winners, max_exp) -> bool:
    """CI of c overlaps the leader's — eligible for the tie-break pool."""
    return c["diff_ci"][1] is not None and c["net_expR"] >= max_exp - 0.05


def _tiebreak_key(recipe: str, exp_neg: float):
    """fewer legs > no-BE > rounder numbers, then higher expR (SPEC §7.5)."""
    n_legs = 2 if recipe.startswith("p_") else 1
    has_be = 0 if "beNone" in recipe else 1
    roundness = 0 if any(x in recipe for x in ("tp1", "0.5", "1.0", "2.0")) else 1
    return (n_legs, has_be, roundness, exp_neg)


def _confirm_on_validation(rep: pd.DataFrame, recipe: str) -> Dict[str, Any]:
    """SPEC §7.5: the ONE selected recipe gets a single validation look — paired
    diff vs baseline: same sign, per-quarter improvement sign ≥ 60%."""
    sub = rep[rep["_split"] == "VALIDATION"]
    piv = sub.pivot_table(index="setup_id", columns="recipe", values="net_r")
    if recipe not in piv.columns or "baseline" not in piv.columns:
        return {"confirmed": False, "note": "recipe/baseline absent in validation"}
    both = piv[[recipe]].join(piv["baseline"].rename("baseline")).dropna()
    if len(both) < 5:
        return {"confirmed": False, "note": "too few validation pairs"}
    diff = both[recipe] - both["baseline"]
    lo, hi = bootstrap_diff_ci(both[recipe].tolist(), both["baseline"].tolist(),
                               paired=True)
    same_sign = float(diff.mean()) > 0 and lo is not None and lo > 0
    # per-quarter improvement sign.
    q = sub[sub["recipe"].isin([recipe, "baseline"])].pivot_table(
        index=["setup_id", "_quarter"], columns="recipe", values="net_r").dropna()
    q = q.reset_index()
    pos = counted = 0
    for _, g in sorted(q.groupby("_quarter"), key=lambda kv: kv[0]):
        if len(g) < MIN_QUARTER_N:
            continue
        counted += 1
        if (g[recipe] - g["baseline"]).mean() > 0:
            pos += 1
    quarter_ok = counted > 0 and (pos / counted) >= QUARTER_SIGN_FRAC
    return {"confirmed": bool(same_sign and quarter_ok),
            "paired_diff": round(float(diff.mean()), 4), "diff_ci": [lo, hi],
            "pos_quarters": f"{pos}/{counted}"}


def _make_clusters(df: pd.DataFrame, s2: Dict[str, Any]
                   ) -> Tuple[str, Dict[str, pd.Index], Optional[List[float]]]:
    """SPEC §7.4. Primary = EV quintiles (if Stage 2 passed), edges from DISCOVERY
    EV distribution frozen. Fallback = event(bos_tag)×pd_zone cells N≥300 in
    discovery, small→all_other. Returns (mode, {cluster_id: row-index}, ev_edges).
    ev_edges are the frozen discovery quintile edges (None in fallback mode) — they
    are stamped into Stage 3's output so Stage 4 can RECOMPUTE cluster membership on
    the holdout (setup_ids don't survive across splits)."""
    if s2.get("verdict") == "EV_MODEL" and s2.get("ev_model"):
        ev = _score_ev(df, s2["ev_model"])
        disc_ev = ev[df["_split"] == "DISCOVERY"]
        edges = np.quantile(disc_ev.dropna(), [0, .2, .4, .6, .8, 1.0])
        edges = np.unique(edges)
        if len(edges) >= 3:
            q = _ev_quintile_of(ev, edges)
            clusters = {f"ev_q{i}": df.index[q == i] for i in sorted(set(q))}
            return "ev_quintile", clusters, [float(e) for e in edges]
    # fallback: bos_tag × pd_zone.
    clusters: Dict[str, pd.Index] = {}
    small = []
    for (tag, zone), g in df.groupby([df.get("bos_tag", "?"), df.get("pd_zone", "?")]):
        disc_n = int((g["_split"] == "DISCOVERY").sum())
        cid = f"{tag}×{zone}"
        if disc_n >= CLUSTER_MIN_N:
            clusters[cid] = g.index
        else:
            small.append(g.index)
    if small:
        clusters["all_other"] = small[0].append(small[1:]) if len(small) > 1 else small[0]
    return "event_pdzone", clusters, None


def _ev_quintile_of(ev: pd.Series, edges: List[float]) -> np.ndarray:
    """Map EV scores to quintile index 0..k-1 using FROZEN discovery edges. Single
    source of truth so clustering and holdout membership agree bit-for-bit."""
    e = np.asarray(edges, dtype=float)
    return np.clip(np.digitize(np.asarray(ev, dtype=float), e[1:-1]), 0, len(e) - 2)


def _score_ev(df: pd.DataFrame, model: Dict[str, Any]) -> pd.Series:
    """Apply a saved ridge EV model to rows (for clustering)."""
    feats = sorted({_feature_of_column(c) for c in model["columns"]})
    spec = {"continuous": model["standardization"],
            "categorical": model["categorical_encoding"],
            "columns": model["columns"]}
    X, _, _ = _build_design(df, feats, spec)
    beta = np.array([model["intercept"]] + [model["coefficients"][c] for c in model["columns"]])
    return pd.Series(_ridge_predict(X, beta), index=df.index)


def _time_in_trade(rep_recipe: pd.DataFrame) -> Dict[str, Any]:
    """SPEC §7.6 REPORT-ONLY descriptives. p25/p50/p75 bars_to_exit by exit reason;
    net expR of trades still open after {12,24,36} bars. Selects nothing."""
    out: Dict[str, Any] = {"by_reason": {}, "expR_if_open_after": {}}
    for reason, g in rep_recipe.groupby("exit_reason"):
        b = g["bars_to_exit"].dropna()
        if len(b):
            out["by_reason"][str(reason)] = {
                "n": len(b), "p25": float(np.percentile(b, 25)),
                "p50": float(np.percentile(b, 50)), "p75": float(np.percentile(b, 75))}
    for k in (12, 24, 36):
        still = rep_recipe[rep_recipe["bars_to_exit"] >= k]
        if len(still):
            out["expR_if_open_after"][str(k)] = {
                "n": len(still), "net_expR": round(float(still["net_r"].mean()), 4)}
    return out


def stage3(run_dir: str, engine_dir: str, forced: bool) -> Dict[str, Any]:
    gate = _read_json(_stage_path(engine_dir, 0)) or {}
    s2 = _read_json(_stage_path(engine_dir, 2)) or {}
    scope = gate.get("scope", "exploratory")
    exploratory = scope != "verdict"

    df = load_population(run_dir)
    pooled = pooled_fx_gold(df)
    filled = pooled[pooled["fill_ts"].notna()].copy()
    # k·ATR wider-stop candidates sized from THIS run's wick-depth distribution
    # (SPEC §7/§15) — measured on DISCOVERY only so sizing never sees validation.
    disc_filled = filled[filled["_split"] == "DISCOVERY"]
    widen_ks, widen_src = _widen_k_candidates(disc_filled)
    grid = _build_recipe_grid(widen_ks=widen_ks, trail=True)
    pcm = _pair_conf_map()
    rep = _replay_grid(filled, grid, pcm)

    base = {"stage": 3, "pass": True, "forced": forced,
            "run_id": os.path.basename(run_dir), "generated_utc": _now_utc(),
            "scope": scope, "exploratory": exploratory,
            "window": _window_label(filled), "n_trades": int(len(filled)),
            "widen_ks": widen_ks, "widen_ks_source": widen_src,
            "n_recipes": len(grid)}
    if rep.empty:
        base.update({"note": "no replayable trades", "clusters": []})
        _write_json(_stage_path(engine_dir, 3), base)
        return base

    recipes = list(grid.keys())
    # GLOBAL sweep (the "one recipe for everything" yardstick, §7.4).
    sel_split = "DISCOVERY" if not exploratory else ""
    global_pick = _paired_diff_select(rep, recipes, sel_split)
    global_confirmed = None
    if global_pick and not exploratory:
        global_confirmed = _confirm_on_validation(rep, global_pick["recipe"])

    # Clusters.
    filled_idx = filled.reset_index(drop=True)
    rep_key = rep.merge(filled_idx[["setup_id", "bos_tag", "pd_zone", "_split"]]
                        .rename(columns={"_split": "_split_f"}),
                        on="setup_id", how="left", suffixes=("", "_f"))
    mode, clusters, ev_edges = _make_clusters(filled_idx, s2)
    cluster_out: List[Dict[str, Any]] = []
    for cid, idx in sorted(clusters.items()):
        ids = set(filled_idx.loc[idx, "setup_id"])
        crep = rep[rep["setup_id"].isin(ids)]
        if crep.empty:
            continue
        pick = _paired_diff_select(crep, recipes, sel_split)
        confirmed = (_confirm_on_validation(crep, pick["recipe"])
                     if (pick and not exploratory) else None)
        ships = (pick["recipe"] if (pick and (exploratory or
                 (confirmed and confirmed["confirmed"]))) else "baseline")
        cluster_out.append({
            "cluster": cid, "n": int(crep["setup_id"].nunique()),
            "selected": pick, "validation_confirm": confirmed,
            "ships": ships,
            "verdict": ("candidate_hypothesis" if exploratory else
                        ("shipped" if ships != "baseline" else "no_exit_improvement")),
            "time_in_trade": _time_in_trade(
                crep[crep["recipe"] == ships]),
        })

    # Per-cluster justified only if it beats the confirmed global (§7.5).
    exit_mode = "global"
    if not exploratory and global_pick and global_confirmed and global_confirmed.get("confirmed"):
        any_cluster_ships = any(c["ships"] != "baseline" for c in cluster_out)
        exit_mode = "per_cluster" if any_cluster_ships else "global"

    # Forced-close sensitivity (§7.1).
    forced_ids = set(rep[rep["forced_close"]]["setup_id"])
    forced_pct = round(100.0 * len(forced_ids) / max(rep["setup_id"].nunique(), 1), 1)

    rep.to_csv(os.path.join(engine_dir, "stage3_exits.csv"), index=False)
    result = {
        **base, "cluster_mode": mode, "exit_mode": exit_mode,
        "ev_discovery_edges": ev_edges,
        "global_recipe": global_pick, "global_validation_confirm": global_confirmed,
        "clusters": cluster_out, "forced_close_pct": forced_pct,
        "baseline_recipe": {"legs": [[1.0, "tp1"]], "be_trigger_r": 1.0, "be_to_r": 0.0},
        "note": ("EXPLORATORY — cluster recipes are hypotheses, no validation "
                 "confirm applied" if exploratory else
                 "recipes ship only past paired-CI + validation confirm; baseline "
                 "is the default"),
    }
    _write_json(_stage_path(engine_dir, 3), result)
    return result


def _stage2_pass_bar(ev_val, yv, val, disc, beta, spec, feats) -> Tuple[bool, Dict]:
    """SPEC §6.3 pass bar (ALL): trade-level Spearman(EV, r) ≥ 0.10, p<0.01 on
    validation; top-quintile realised expR CI lo > 0; top-quintile expR >
    population expR by ≥ 0.10R."""
    info: Dict[str, Any] = {}
    if not _HAS_SCIPY or len(ev_val) < 20:
        return False, {"note": "insufficient validation N for pass bar"}
    rho, p = _spearmanr(ev_val, yv)
    rho = float(rho) if not math.isnan(rho) else 0.0
    info["spearman"] = round(rho, 4); info["spearman_p"] = p
    # top quintile by EV
    thr = np.quantile(ev_val, 0.8)
    top_mask = ev_val >= thr
    top_r = yv[top_mask]
    lo, hi = bootstrap_ci(top_r.tolist())
    pop_exp = float(yv.mean())
    top_exp = float(top_r.mean()) if len(top_r) else None
    info.update({"top_quintile_n": int(top_mask.sum()),
                 "top_quintile_expR": round(top_exp, 4) if top_exp is not None else None,
                 "top_quintile_ci": [lo, hi], "population_expR": round(pop_exp, 4)})
    ok = (rho >= EV_SPEARMAN_FLOOR and p is not None and p < 0.01
          and lo is not None and lo > 0
          and top_exp is not None and (top_exp - pop_exp) >= MIN_EFFECT_R)
    info["passed"] = ok
    return ok, info


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — RECIPE SYNTHESIS + THE VERDICT (SPEC §8)
#
# ALL choices made on discovery+validation BEFORE the holdout is touched. The
# combined recipe is applied to HOLDOUT exactly ONCE. Block-bootstrap (ISO weeks)
# + expanding walk-forward for robustness (computed, not selected on). Mechanical
# verdict tree — the script prints it, no judgment calls. holdout_opened_utc is
# stamped; re-runs warn loudly (holdout is burnt after upstream changes). Does not
# run in exploratory scope (no verdict is the exploratory product's whole point).
# ═══════════════════════════════════════════════════════════════════════════

def _weekly_block_bootstrap_ci(vals: List[float], weeks: List[str],
                               n_boot: int = BOOT_N):
    """Resample ISO weeks with replacement (SPEC §8.2). Same-week trades are
    correlated (shared USD news); if the block CI flips the iid conclusion, the
    block CI wins."""
    if len(vals) < 5:
        return None, None
    by_week: Dict[str, List[float]] = {}
    for v, w in zip(vals, weeks):
        by_week.setdefault(w, []).append(v)
    keys = sorted(by_week)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        pool = []
        for ci in chosen:
            pool.extend(by_week[keys[ci]])
        if pool:
            boots.append(float(np.mean(pool)))
    if not boots:
        return None, None
    return round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)


def _iso_week(ts) -> str:
    t = pd.to_datetime(ts, utc=True)
    iso = t.isocalendar()
    return f"{iso[0]}-W{int(iso[1]):02d}"


def _apply_combined_recipe(df: pd.DataFrame, recipe: Dict[str, Any],
                           pcm: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Apply the frozen combined recipe (pair set + gates + order rules + exits +
    optional EV threshold) to a split. Returns per-trade net-R rows for the KEPT
    trades. This is the single holdout application in §8.2."""
    keep = df[df["pair"].isin(recipe["pair_set"])].copy()
    # EV threshold gate (if shipped).
    if recipe.get("ev_threshold") and recipe.get("ev_model"):
        ev = _score_ev(keep, recipe["ev_model"])
        keep = keep[ev >= recipe["ev_threshold"]["value"]].copy()
    # order rules / skip gates are row-exclusions expressed as caveated proxies.
    for g in recipe.get("gates", []):
        col, op, val = g.get("col"), g.get("op"), g.get("val")
        if col and col in keep.columns:
            keep = _apply_exclusion(keep, col, op, val)
    # exits: for a single global recipe apply it; per-cluster picks the cluster's.
    grid = {}
    exit_pol = recipe["exit_policy"]
    if exit_pol["mode"] == "global":
        grid = {"final": _recipe_dict(exit_pol["clusters"][0]["recipe"])}
        rep = _replay_grid(keep, grid, pcm)
        return rep[rep["recipe"] == "final"]
    # per_cluster: RECOMPUTE each trade's cluster on THIS split (setup_ids do not
    # survive across splits — reading member_ids would zero every holdout row, the
    # bug this replaces), then apply the cluster's shipped recipe.
    membership = _recompute_cluster_membership(keep, exit_pol, recipe)
    frames = []
    for c in exit_pol["clusters"]:
        idx = membership.get(c["id"])
        sub = keep.loc[idx] if idx is not None and len(idx) else keep.iloc[0:0]
        if sub.empty:
            continue
        rep = _replay_grid(sub, {"final": _recipe_dict(c["recipe"])}, pcm)
        frames.append(rep[rep["recipe"] == "final"])
    return pd.concat(frames) if frames else pd.DataFrame()


def _recompute_cluster_membership(keep: pd.DataFrame, exit_pol: Dict[str, Any],
                                  recipe: Dict[str, Any]) -> Dict[str, pd.Index]:
    """Assign each row of `keep` to its per_cluster id, on WHATEVER split keep is.

    ev_quintile: score with the shipped EV model, bucket by the FROZEN discovery
    edges carried in exit_pol['ev_edges'] (same _ev_quintile_of used at build time).
    event_pdzone: parse 'tag×zone' back out of the cluster id; 'all_other' catches
    the remainder. A row lands in exactly one shipped cluster (or nowhere → its
    trades are declined, which is correct: an un-shipped cluster ships baseline and
    is handled by the caller only for shipped clusters)."""
    ids = [c["id"] for c in exit_pol["clusters"]]
    mode = exit_pol.get("cluster_mode")
    out: Dict[str, pd.Index] = {}
    if mode == "ev_quintile" and exit_pol.get("ev_edges") and recipe.get("ev_model"):
        ev = _score_ev(keep, recipe["ev_model"])
        q = _ev_quintile_of(ev, exit_pol["ev_edges"])
        for cid in ids:
            try:
                qi = int(cid.split("ev_q")[1])
            except (IndexError, ValueError):
                continue
            out[cid] = keep.index[q == qi]
        return out
    # event_pdzone fallback (and any unknown mode): match tag×zone, remainder→all_other.
    tag = keep.get("bos_tag")
    zone = keep.get("pd_zone")
    assigned = pd.Series(False, index=keep.index)
    for cid in ids:
        if cid == "all_other" or "×" not in cid:
            continue
        t, z = cid.split("×", 1)
        m = (tag.astype(str) == t) & (zone.astype(str) == z)
        out[cid] = keep.index[m]
        assigned |= m
    if "all_other" in ids:
        out["all_other"] = keep.index[~assigned]
    return out


def _apply_exclusion(df, col, op, val):
    if op == "<":
        return df[~(pd.to_numeric(df[col], errors="coerce") < val)]
    if op == ">":
        return df[~(pd.to_numeric(df[col], errors="coerce") > val)]
    if op == "==":
        return df[df[col] != val]
    return df


def _recipe_dict(r: Dict[str, Any]) -> Dict[str, Any]:
    legs = [tuple(l) for l in r["legs"]]
    out = {"legs": legs, "be_trigger_r": r.get("be_trigger_r")}
    if r.get("be_trigger_r") is not None:
        out["be_to_r"] = r.get("be_to_r", 0.0)
    # Preserve the exit-track knobs so a widen/trail winner replays as itself on the
    # holdout — dropping them would silently downgrade it to a plain liquidity-TP
    # recipe and produce a WRONG verdict (the recipe replayed != the recipe chosen).
    for knob in ("sl_widen_atr", "trail_r", "trail_arm_r"):
        if r.get(knob) is not None:
            out[knob] = r[knob]
    return out


def stage4(run_dir: str, engine_dir: str, forced: bool) -> Dict[str, Any]:
    gate = _read_json(_stage_path(engine_dir, 0)) or {}
    s1 = _read_json(_stage_path(engine_dir, 1)) or {}
    s2 = _read_json(_stage_path(engine_dir, 2)) or {}
    s3 = _read_json(_stage_path(engine_dir, 3)) or {}
    scope = gate.get("scope", "exploratory")

    base = {"version": 1, "stage": 4, "pass": True, "forced": forced,
            "input_run": os.path.basename(run_dir), "generated_utc": _now_utc(),
            "scope": scope}

    if scope != "verdict":
        base.update({"verdict": "NO_VERDICT_EXPLORATORY", "robustness": None,
                     "note": "Stage 4 does not run in exploratory scope (SPEC "
                             "§15.1) — the exploratory product is hypotheses, "
                             "never a verdict. Run the full 18-yr run for a verdict."})
        _write_json(_stage_path(engine_dir, 4), base)
        _write_report(engine_dir, run_dir, gate, s1, s2, s3, base)
        return base

    df = load_population(run_dir)
    pooled = pooled_fx_gold(df)
    pcm = _pair_conf_map()

    # ── Combine (all on discovery+validation) ───────────────────────────────
    pair_set = _select_pair_set(pooled)
    gates = _select_gates(pooled, s1)
    order_rules = _select_order_rules(pooled, s1)
    ev_threshold = _select_ev_threshold(pooled, s2) if s2.get("verdict") == "EV_MODEL" else None
    exit_policy = _build_exit_policy(s3, pooled)

    recipe = {
        "pair_set": pair_set["pairs"], "pairs_dropped": pair_set["dropped"],
        "gates": gates, "order_rules": order_rules,
        "ev_model": s2.get("ev_model"), "ev_threshold": ev_threshold,
        "exit_policy": exit_policy,
        "cost_model": {"rule": "cost_r per non-initial-SL leg",
                       "source": "config.json spread_pips"},
    }

    # ── The ONE holdout look ────────────────────────────────────────────────
    prior = _read_json(_stage_path(engine_dir, 4))
    holdout_reopened = bool(prior and prior.get("holdout_opened_utc"))
    hold = split_frame(pooled, "HOLDOUT")
    hrep = _apply_combined_recipe(hold, recipe, pcm)
    brep = _replay_grid(hold, {"baseline": dict(BASELINE_RECIPE)}, pcm)
    brep = brep[brep["recipe"] == "baseline"]
    holdout = _holdout_report(hrep, brep, hold, recipe, pcm)

    # ── Robustness (computed, not selected on) ──────────────────────────────
    walk_forward = _walk_forward(pooled, recipe, pcm)
    robust = "ROBUST" if sum(1 for f in walk_forward if f.get("test_expR_net", 0) > 0) >= 2 else "FRAGILE"

    # WAR (2026) — reported separately, never pooled.
    war = split_frame(pooled, "WAR")
    warrep = _apply_combined_recipe(war, recipe, pcm) if not war.empty else pd.DataFrame()
    war_block = ({"n": int(warrep["setup_id"].nunique()),
                  "expR_net": round(float(warrep["net_r"].mean()), 4),
                  "note": "reported only, never pooled"} if not warrep.empty
                 else {"n": 0, "note": "no WAR trades / reported only, never pooled"})

    # BTC standalone (never pooled).
    btc = _btc_section(df, gate, pcm)

    verdict = _verdict_tree(holdout)
    result = {
        **base, "verdict": verdict, "robustness": robust,
        "holdout_opened_utc": _now_utc(), "holdout_reopened": holdout_reopened,
        **recipe,
        "holdout": holdout, "walk_forward": walk_forward,
        "war_2026": war_block, "btc": btc,
        **_ledger_summary(engine_dir),
        "caveats": _collect_caveats(order_rules, holdout_reopened, holdout,
                                    _ledger_summary(engine_dir)),
    }
    _write_json(_stage_path(engine_dir, 4), result)
    _write_report(engine_dir, run_dir, gate, s1, s2, s3, result)
    return result


def _select_pair_set(pooled: pd.DataFrame) -> Dict[str, Any]:
    """Start = all 9 FX+Gold. Drop a pair only if BOTH discovery and validation
    show its expR CI entirely < 0 AND < 40% positive quarters (SPEC §8.1)."""
    keep, dropped = [], []
    for pair in sorted(pooled["pair"].dropna().unique()):
        d = pooled[(pooled["pair"] == pair) & (pooled["_split"] == "DISCOVERY")]
        v = pooled[(pooled["pair"] == pair) & (pooled["_split"] == "VALIDATION")]
        ds, vs = _cell_stats(d), _cell_stats(v)
        dp, dc = _parse_quarters(ds["pos_quarters"]); vp, vc = _parse_quarters(vs["pos_quarters"])
        both_neg = (ds["ci_hi"] is not None and ds["ci_hi"] < 0
                    and vs["ci_hi"] is not None and vs["ci_hi"] < 0)
        low_q = (dc and dp / dc < 0.40) and (vc and vp / vc < 0.40)
        if both_neg and low_q:
            dropped.append({"pair": pair, "evidence": {"disc": ds, "val": vs}})
        else:
            keep.append(pair)
    return {"pairs": keep, "dropped": dropped}


def _select_gates(pooled: pd.DataFrame, s1: Dict[str, Any]) -> List[Dict[str, Any]]:
    """counter-PD-CHoCH skip re-tested on 18-yr: ships iff that cell's expR CI < 0
    in discovery AND validation. Plus any §5.6 flagged interaction cell CI<0 both
    splits (SPEC §8.1/§8.2)."""
    gates = []
    # counter-PD-CHoCH: pd_alignment == 'counter' AND a CHoCH.
    if "pd_alignment" in pooled.columns and "bos_tag" in pooled.columns:
        mask = (pooled["pd_alignment"] == "counter") & \
               (pooled["bos_tag"].astype(str).str.contains("CHoCH"))
        d = _cell_stats(pooled[mask & (pooled["_split"] == "DISCOVERY")])
        v = _cell_stats(pooled[mask & (pooled["_split"] == "VALIDATION")])
        ships = (d["ci_hi"] is not None and d["ci_hi"] < 0
                 and v["ci_hi"] is not None and v["ci_hi"] < 0)
        gates.append({"rule": "skip counter_pd_choch", "shipped": bool(ships),
                      "evidence": {"disc": d, "val": v}})
    # flagged interaction cells with CI<0 both splits.
    for cell in s1.get("interactions", []):
        d, v = cell.get("disc", {}), cell.get("val", {})
        if (d.get("ci_hi") is not None and d["ci_hi"] < 0
                and v.get("ci_hi") is not None and v["ci_hi"] < 0):
            gates.append({"rule": f"skip {cell['interaction']}={cell['cell']}",
                          "shipped": True, "evidence": {"disc": d, "val": v}})
    return gates


def _select_order_rules(pooled: pd.DataFrame, s1: Dict[str, Any]) -> List[Dict[str, Any]]:
    """SPEC §8.1b: a FILL-TIME survivor whose bad bucket has expR CI<0 in
    discovery AND validation may ship as an order rule (row-exclusion proxy)."""
    rules = []
    fill_survivors = [r for r in s1.get("features", [])
                      if r.get("verdict") == "survivor"
                      and _classify_timing(r["feature"]) == "fill_time"]
    for r in fill_survivors:
        rules.append({
            "rule": f"order-manage on {r['feature']}", "feature": r["feature"],
            "shipped": True, "evidence": {"delta_val": r.get("delta_val")},
            "approximation_caveat": "row-exclusion proxy, §8.1b — declined trade "
                                    "assumed never happens; conservative for "
                                    "TTL/window, slightly overstates arm-delay",
        })
    return rules


def _select_ev_threshold(pooled: pd.DataFrame, s2: Dict[str, Any]
                         ) -> Optional[Dict[str, Any]]:
    """SPEC §8.1b EV threshold: candidates = discovery EV quintile edges + 'none';
    pick highest validation net expR whose CI lo>0 and keeps ≥30 trades/quarter."""
    model = s2.get("ev_model")
    if not model:
        return None
    disc = split_frame(pooled, "DISCOVERY")
    val = split_frame(pooled, "VALIDATION")
    ev_disc = _score_ev(disc, model)
    ev_val = _score_ev(val, model)
    yv = pd.to_numeric(val["r_realised"], errors="coerce").fillna(0.0).to_numpy()
    q_edges = {f"q{int(q*100)}": float(np.quantile(ev_disc.dropna(), q))
               for q in (0.2, 0.4, 0.6, 0.8)}
    best = None
    for label, thr in q_edges.items():
        mask = (ev_val >= thr).to_numpy()
        if mask.sum() < 5:
            continue
        r = yv[mask]
        lo, hi = bootstrap_ci(r.tolist())
        # ≥30 trades/quarter average.
        n_q = val[mask]["_quarter"].nunique() if hasattr(val[mask], "nunique") else 1
        per_q = mask.sum() / max(n_q, 1)
        if lo is not None and lo > 0 and per_q >= 30:
            cand = {"value": round(thr, 5), "discovery_quantile": label,
                    "val_expR": round(float(r.mean()), 4), "ci": [lo, hi]}
            if best is None or cand["val_expR"] > best["val_expR"]:
                best = cand
    return best


def _build_exit_policy(s3: Dict[str, Any], pooled: pd.DataFrame) -> Dict[str, Any]:
    """SPEC §8.1 exits = Stage 3's confirmed output (per-cluster set or global).

    For per_cluster, the policy carries `cluster_mode` + `ev_edges` (frozen
    discovery EV quintile edges) so membership is RECOMPUTED on any later split
    (holdout / walk-forward / WAR) from the shipped EV model — never read off
    setup_ids that don't exist outside discovery+validation (the bug that zeroed
    every per_cluster holdout). Fallback (event_pdzone) membership is recomputed
    from the bos_tag×pd_zone cell definition parsed back out of the cluster id."""
    mode = s3.get("exit_mode", "global")
    # The k·ATR set Stage 3 actually swept — needed to rebuild a widen recipe by name
    # with the SAME (data-derived) k, so the holdout replays the chosen recipe exactly.
    widen_ks = s3.get("widen_ks")
    if mode == "global" and s3.get("global_recipe"):
        gr = s3["global_recipe"]["recipe"]
        return {"mode": "global",
                "clusters": [{"id": "global", "definition": "all trades",
                              "recipe": _recipe_from_name(gr, widen_ks)}]}
    if mode == "per_cluster":
        clusters = []
        for c in s3.get("clusters", []):
            if c["ships"] != "baseline":
                clusters.append({"id": c["cluster"], "definition": c["cluster"],
                                 "recipe": _recipe_from_name(c["ships"], widen_ks)})
        if clusters:
            return {"mode": "per_cluster",
                    "cluster_mode": s3.get("cluster_mode"),
                    "ev_edges": s3.get("ev_discovery_edges"),
                    "clusters": clusters}
    # default: baseline.
    return {"mode": "global",
            "clusters": [{"id": "baseline", "definition": "all trades",
                          "recipe": {"legs": [[1.0, "tp1"]], "be_trigger_r": 1.0,
                                     "be_to_r": 0.0}}]}


def _recipe_from_name(name: str,
                      widen_ks: Optional[List[float]] = None) -> Dict[str, Any]:
    """Reconstruct a recipe dict from a grid name (for the JSON schema). `widen_ks`
    must be the SAME set Stage 3 swept so a `widen<k>_…` name rebuilds with the exact
    data-derived k — otherwise a widen winner would fall back to baseline and the
    holdout would replay the wrong recipe (silent wrong verdict)."""
    grid = _build_recipe_grid(widen_ks=widen_ks, trail=True)
    r = grid.get(name, dict(BASELINE_RECIPE))
    out = {"legs": [list(l) for l in r["legs"]],
           "be_trigger_r": r.get("be_trigger_r"),
           "be_to_r": r.get("be_to_r", 0.0)}
    for knob in ("sl_widen_atr", "trail_r", "trail_arm_r"):
        if r.get(knob) is not None:
            out[knob] = r[knob]
    return out


def _holdout_report(hrep, brep, hold, recipe, pcm) -> Dict[str, Any]:
    """Net-of-cost holdout metrics + baseline comparison (SPEC §8.2)."""
    if hrep.empty:
        return {"n": 0, "note": "no holdout trades after recipe applied"}
    vals = hrep["net_r"].tolist()
    weeks = [_iso_week(t) for t in hrep["alert_ts"]]
    lo_i, hi_i = bootstrap_ci(vals)
    lo_b, hi_b = _weekly_block_bootstrap_ci(vals, weeks)
    pos, counted = _pos_quarters(hrep.assign(alert_ts=hrep["alert_ts"]), "net_r", "alert_ts")
    # vs baseline paired diff.
    piv = hrep[["setup_id", "net_r"]].merge(
        brep[["setup_id", "net_r"]], on="setup_id", suffixes=("_r", "_b")).dropna()
    dlo, dhi = bootstrap_diff_ci(piv["net_r_r"].tolist(), piv["net_r_b"].tolist(),
                                 paired=True) if len(piv) >= 5 else (None, None)
    per_book = {bk: round(float(hrep.merge(
        hold[["setup_id", "_book"]], on="setup_id")["net_r"][
            hrep.merge(hold[["setup_id", "_book"]], on="setup_id")["_book"] == bk].mean()), 4)
        for bk in ["A", "B"] if bk in hold["_book"].values}
    return {
        "n": int(hrep["setup_id"].nunique()),
        "expR_net": round(float(np.mean(vals)), 4),
        "ci_iid": [lo_i, hi_i], "ci_block": [lo_b, hi_b],
        "wr_pct": insights.win_rate_pct(hrep.rename(columns={"net_r": "r_realised"})),
        "totR": round(float(np.sum(vals)), 2),
        "pos_quarters": f"{pos}/{counted}",
        "vs_baseline_diff_ci": [dlo, dhi],
        "max_drawdown_R": insights.max_drawdown_r(vals),
        "per_book": per_book,
    }


def _walk_forward(pooled, recipe, pcm) -> List[Dict[str, Any]]:
    """Expanding walk-forward, 3 folds (SPEC §8.2). We apply the SAME frozen
    combined recipe's exit + gates to each fold's test window — a mechanical
    robustness read, not a re-fit (the spec's 're-run mechanically per fold' is
    approximated by re-applying the frozen recipe; full per-fold re-fit is a v2
    depth). Labelled clearly as such."""
    out = []
    for label, ts, te in WF_FOLDS:
        test = pooled[(pooled["_split"].isin(["DISCOVERY", "VALIDATION", "HOLDOUT"]))]
        a = _to_utc(test["alert_ts"])
        test = test[(a >= pd.Timestamp(ts, tz="UTC")) & (a <= pd.Timestamp(te, tz="UTC"))]
        rep = _apply_combined_recipe(test, recipe, pcm)
        out.append({"fold": f"fit{label}_test{ts[:4]}-{te[:4]}",
                    "n": int(rep["setup_id"].nunique()) if not rep.empty else 0,
                    "test_expR_net": round(float(rep["net_r"].mean()), 4) if not rep.empty else 0.0,
                    "note": "frozen-recipe re-application (not a per-fold re-fit; v2)"})
    return out


def _btc_section(df, gate, pcm) -> Dict[str, Any]:
    """BTC standalone if eligible N≥300, else too_thin (SPEC §3.3)."""
    btc = df[df["_book"] == "BTC"]
    n = int(len(btc))
    if n < BTC_STANDALONE_MIN_N:
        return {"status": "too_thin", "detail": {"n": n}}
    vals = pd.to_numeric(btc["r_realised"], errors="coerce").dropna().tolist()
    lo, hi = bootstrap_ci(vals)
    return {"status": "standalone",
            "detail": {"n": n, "expR": round(float(np.mean(vals)), 4), "ci": [lo, hi]}}


def _verdict_tree(holdout: Dict[str, Any]) -> str:
    """SPEC §8.3 mechanical tree. ENTRY test: EV-gated holdout net expR block-CI
    lo>0 AND ≥60% pos quarters. EXIT test: chosen-vs-baseline paired diff CI lo>0
    AND ≥60% pos quarters of the diff."""
    if holdout.get("n", 0) == 0:
        return "NO_EDGE"
    block = holdout.get("ci_block", [None, None])
    pos, counted = _parse_quarters(holdout.get("pos_quarters", "0/0"))
    q_ok = counted > 0 and pos / counted >= QUARTER_SIGN_FRAC
    entry_edge = (block[0] is not None and block[0] > 0 and q_ok)
    diff = holdout.get("vs_baseline_diff_ci", [None, None])
    exit_edge = (diff[0] is not None and diff[0] > 0)
    if entry_edge and exit_edge:
        return "ENTRY_AND_EXIT_EDGE"
    if entry_edge:
        return "ENTRY_EDGE_ONLY"
    if exit_edge:
        return "EXIT_EDGE_ONLY"
    return "NO_EDGE"


def _collect_caveats(order_rules, reopened, holdout,
                     ledger: Optional[Dict[str, Any]] = None) -> List[str]:
    c = []
    if ledger and ledger.get("validation_burned"):
        c.append(f"VALIDATION RE-OPENED (runs: {ledger.get('validation_runs')}) — "
                 "survivor list is no longer a one-shot confirmation; treat the "
                 "verdict as exploratory (D4).")
    if reopened:
        c.append("HOLDOUT REOPENED — a prior Stage 4 already spent this holdout "
                 "window; this verdict is contaminated by earlier looks (SPEC §9.6).")
    if order_rules:
        c.append("Order rules use a row-exclusion proxy (SPEC §8.1b) — see each "
                 "rule's approximation_caveat.")
    if holdout.get("ci_block") and holdout.get("ci_iid"):
        b, i = holdout["ci_block"], holdout["ci_iid"]
        if b[0] is not None and i[0] is not None and (b[0] > 0) != (i[0] > 0):
            c.append("Block CI and iid CI disagree — the block CI wins (same-week "
                     "correlation); read the verdict off the block CI.")
    return c


# ═══════════════════════════════════════════════════════════════════════════
# THE REPORT (edge_engine_report.md — dense markdown for chat, SPEC §8.4/§16.3)
# ═══════════════════════════════════════════════════════════════════════════

def _write_report(engine_dir, run_dir, gate, s1, s2, s3, s4) -> None:
    run_id = os.path.basename(run_dir)
    scope = gate.get("scope", "exploratory")
    L = []
    L.append(f"# EDGE ENGINE REPORT — `{run_id}`")
    L.append("")
    if scope != "verdict":
        L.append("> **SHORT-RANGE MODE — EXPLORATORY. HYPOTHESES ONLY, NOT A "
                 "SHIPPABLE VERDICT.**")
        L.append("")
    L.append(f"- **Input run:** `{run_id}`  ·  **scope:** `{scope}`  ·  "
             f"generated {s4.get('generated_utc')}")
    cen = gate.get("census", {}).get("by_split", {})
    L.append(f"- **Census by split:** " + ", ".join(f"{k}={v}" for k, v in cen.items()))
    L.append(f"- **Every number below carries N + window + scope.** A number "
             f"without a verdict-scope stamp is a hypothesis, not an edge.")
    L.append("")

    # Staged-review header (SPEC §18 / SPEC_STAGED §7.2) — pure additional lines.
    if scope == "verdict":
        disc = _read_json(_discovery_path(engine_dir)) or {}
        appr = _read_json(_approval_path(engine_dir)) or {}
        led = _ledger_summary(engine_dir)
        L.append("## Staged Review")
        L.append(f"- discovery token: `{disc.get('token', '—')}`  ·  "
                 f"approved: `{appr.get('approved_utc', '—')}`")
        L.append(f"- validation_runs: **{led['validation_runs']}**  ·  "
                 f"ledger via: {led['vias'] or '—'}")
        if led["validation_burned"]:
            L.append(f"- ⚠️ **VALIDATION RE-OPENED (runs: {led['validation_runs']})** "
                     f"— verdict is no longer a one-shot confirmation (D4).")
        L.append("")

    # Stage 0
    L.append("## Stage 0 — Trust Gate")
    for c in gate.get("checks", []):
        L.append(f"- {'✅' if c['pass'] else '❌'} `{c['check']}`"
                 + (f" — {c.get('note')}" if c.get("note") else ""))
    L.append("")

    # Stage 1
    L.append("## Stage 1 — Univariate Screen")
    L.append(f"- window `{s1.get('window')}`  ·  N={s1.get('n_pooled')}  ·  "
             f"survivors: {s1.get('survivors') or '(none — a valid null)'}")
    if s1.get("ranked_actionable"):
        L.append("")
        L.append("| feature | verdict | Δval | actionable | timing |")
        L.append("|---|---|---|---|---|")
        for r in s1["ranked_actionable"][:20]:
            L.append(f"| `{r['feature']}` | {r['verdict']} | {r.get('delta_val')} "
                     f"| {r.get('actionable_at')} | {r['timing']} |")
    if s1.get("detection_queue"):
        L.append("")
        L.append(f"- **Detection queue** (change-detection-and-rerun): "
                 f"{', '.join(s1['detection_queue'])}")
    L.append("")

    # Stage 2
    L.append("## Stage 2 — EV Model")
    L.append(f"- verdict: **{s2.get('verdict')}** — {s2.get('note', '')}")
    if s2.get("ev_model"):
        pb = s2.get("pass_bar", {})
        L.append(f"- ridge λ={s2['ev_model']['lambda']}  ·  "
                 f"val Spearman={pb.get('spearman')}  ·  "
                 f"top-q expR={pb.get('top_quintile_expR')} (pop {pb.get('population_expR')})")
    L.append("")

    # Stage 3
    L.append("## Stage 3 — Exit Optimiser")
    g = s3.get("global_recipe")
    if g:
        L.append(f"- global best: `{g['recipe']}`  net expR={g['net_expR']}  "
                 f"paired-diff={g['paired_diff']} CI={g['diff_ci']}")
    L.append(f"- cluster mode: `{s3.get('cluster_mode')}`  ·  exit_mode: "
             f"`{s3.get('exit_mode')}`  ·  forced-close {s3.get('forced_close_pct')}%")
    if s3.get("clusters"):
        L.append("")
        L.append("| cluster | N | ships | verdict |")
        L.append("|---|---|---|---|")
        for c in s3["clusters"]:
            L.append(f"| {c['cluster']} | {c['n']} | `{c['ships']}` | {c['verdict']} |")
    L.append("")

    # Stage 4
    L.append("## Stage 4 — Verdict")
    if scope != "verdict":
        L.append(f"- **{s4.get('verdict')}** — {s4.get('note')}")
    else:
        L.append(f"- ## VERDICT: **{s4.get('verdict')}**  ·  robustness: "
                 f"**{s4.get('robustness')}**")
        h = s4.get("holdout", {})
        L.append(f"- Holdout (2022–2025): N={h.get('n')}  expR_net={h.get('expR_net')}  "
                 f"iid CI={h.get('ci_iid')}  block CI={h.get('ci_block')}  "
                 f"pos_quarters={h.get('pos_quarters')}")
        L.append(f"- vs baseline diff CI: {h.get('vs_baseline_diff_ci')}  ·  "
                 f"max DD={h.get('max_drawdown_R')}R")
        L.append(f"- Pair set: {s4.get('pair_set')}")
        if s4.get("pairs_dropped"):
            L.append(f"- Dropped: {[d['pair'] for d in s4['pairs_dropped']]}")
        L.append(f"- Walk-forward: " + "  ".join(
            f"{f['fold']}={f['test_expR_net']}" for f in s4.get("walk_forward", [])))
        L.append(f"- WAR 2026 (never pooled): {s4.get('war_2026')}")
        L.append(f"- BTC: {s4.get('btc')}")
        for cav in s4.get("caveats", []):
            L.append(f"- ⚠️ {cav}")
    L.append("")
    L.append("_Script computes; the ship / don't-ship judgment happens in chat "
             "over this report._")

    with open(os.path.join(engine_dir, "edge_engine_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Windows consoles default to cp1252, which cannot encode the Δ/×/λ we print.
    # Force UTF-8 so the report symbols render everywhere (no-op where already utf-8).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="Edge Engine (EDGE_ENGINE_SPEC.md v1)")
    ap.add_argument("--run-dir", default=None,
                    help="path to a completed backtest results dir (with trades.csv)")
    ap.add_argument("--start", default=None, help="YYYY-MM (date-range resolution)")
    ap.add_argument("--end", default=None, help="YYYY-MM (date-range resolution)")
    ap.add_argument("--stage", type=int, default=None,
                    help="run a single stage (0-4). Default: all stages in order.")
    ap.add_argument("--force", action="store_true",
                    help="run a stage even if the prior stage did not pass")
    ap.add_argument("--phase", choices=["discovery", "confirm", "final"], default=None,
                    help="staged review phase (SPEC §18 / SPEC_STAGED §2)")
    ap.add_argument("--approve", metavar="TOKEN", default=None,
                    help="sign the discovery approval token; writes approval.json "
                         "and exits")
    ap.add_argument("--burn-validation", metavar="REASON", default=None,
                    help="explicitly re-open validation with a written reason. "
                         "Appends to validation_ledger.jsonl and stamps everything "
                         "downstream.")
    ap.add_argument("--render-discovery-report", action="store_true",
                    help="re-render edge_engine_discovery.md from the committed "
                         "discovery JSON/CSV alone (no recomputation, no validation "
                         "frame). Backfill for past runs; writes and exits.")
    args = ap.parse_args()

    # Mutual-exclusion (SPEC_STAGED §2.1).
    if args.approve is not None and (args.phase is not None or args.stage is not None):
        raise SystemExit("--approve is mutually exclusive with --phase and --stage")
    if args.phase is not None and args.stage is not None:
        raise SystemExit("--phase and --stage are mutually exclusive")

    run_dir = resolve_run_dir(args.run_dir, args.start, args.end)
    engine_dir = os.path.join(run_dir, "edge_engine")
    os.makedirs(engine_dir, exist_ok=True)
    print(f"Edge Engine — run: {os.path.basename(run_dir)}")
    print(f"  engine outputs -> {engine_dir}")

    # ── --render-discovery-report: re-render the .md from committed JSON/CSV ──
    if args.render_discovery_report:
        from backtest.diagnostics import edge_report
        p = edge_report.render_discovery_report(engine_dir)
        if p:
            print(f"\033[32mrendered {p}\033[0m")
            sys.exit(0)
        print("\033[31mno stage1_discovery.json in engine_dir — run --phase "
              "discovery first\033[0m")
        sys.exit(1)

    # ── --approve: sign the token and exit (SPEC_STAGED §5.2) ────────────────
    if args.approve is not None:
        res = approve(engine_dir, args.approve)
        col = "\033[32m" if res.get("approved") else "\033[31m"
        print(f"{col}{res['note']}\033[0m")
        sys.exit(0 if res.get("approved") else 1)

    # ── --phase: staged review dispatch (SPEC_STAGED §2.1) ───────────────────
    if args.phase is not None:
        _run_phase(args.phase, run_dir, engine_dir, args.force, args.burn_validation)
        return

    stages = [args.stage] if args.stage is not None else [0, 1, 2, 3, 4]
    for st in stages:
        if not _prior_passed(engine_dir, st, args.force):
            print(f"\033[31mStage {st}: prior stage did not pass — stopping "
                  f"(use --force to override).\033[0m")
            break
        if st == 0:
            res = stage0(run_dir, engine_dir, args.force)
            _print_stage0(res)
            if not res["pass"] and not args.force:
                print("\033[31mStage 0 FAILED — nothing downstream is real. "
                      "Stopping.\033[0m")
                break
        elif st == 1:
            res = stage1(run_dir, engine_dir, args.force, args.burn_validation)
            if res.get("refused"):
                break  # gate refused — the refusal block already printed
            _print_stage1(res)
        elif st == 2:
            res = stage2(run_dir, engine_dir, args.force)
            _print_stage2(res)
        elif st == 3:
            res = stage3(run_dir, engine_dir, args.force)
            _print_stage3(res)
        elif st == 4:
            res = stage4(run_dir, engine_dir, args.force)
            _print_stage4(res)


def _run_phase(phase: str, run_dir: str, engine_dir: str, forced: bool,
               burn_reason: Optional[str]) -> None:
    """Staged-review dispatch (SPEC_STAGED §2.1). Each phase stops on completion."""
    from backtest.diagnostics import edge_email

    if phase == "discovery":
        # Stage 0 fresh, then discovery-only preview (§4.1).
        g = stage0(run_dir, engine_dir, forced)
        _print_stage0(g)
        if not g["pass"] and not forced:
            print("\033[31mStage 0 FAILED — cannot stage a run that fails trust.\033[0m")
            return
        res = stage1_discovery(run_dir, engine_dir, forced)
        _print_stage1_discovery(res)
        edge_email.send_discovery(engine_dir, res)
        return

    if phase == "confirm":
        # Require an existing Stage 0 with pass:true (do NOT re-run it — §2.1).
        g = _read_json(_stage_path(engine_dir, 0))
        if not (g and g.get("pass") is True):
            raise SystemExit("--phase confirm requires a passed Stage 0 "
                             "(run --phase discovery first)")
        if g.get("scope") != "verdict":
            raise SystemExit("staged review requires verdict scope (SPEC §15)")
        res = stage1(run_dir, engine_dir, forced, burn_reason)
        if res.get("refused"):
            return  # gate refused — refusal block already printed
        _print_stage1(res)
        edge_email.send_confirm(engine_dir, res)
        return

    if phase == "final":
        # Require a passed canonical Stage 1 (via _prior_passed), then 2,3,4.
        for st in (2, 3, 4):
            if not _prior_passed(engine_dir, st, forced):
                print(f"\033[31mStage {st}: prior stage did not pass — stopping "
                      f"(use --force to override).\033[0m")
                return
            if st == 2:
                _print_stage2(stage2(run_dir, engine_dir, forced))
            elif st == 3:
                _print_stage3(stage3(run_dir, engine_dir, forced))
            elif st == 4:
                res = stage4(run_dir, engine_dir, forced)
                _print_stage4(res)
                edge_email.send_verdict(engine_dir, res)
        return


def _print_stage0(res: Dict[str, Any]) -> None:
    print(f"\n==== STAGE 0 — TRUST GATE ====  scope={res.get('scope')}  "
          f"pass={res['pass']}")
    for c in res["checks"]:
        flag = "PASS" if c["pass"] else "FAIL"
        col = "\033[32m" if c["pass"] else "\033[31m"
        print(f"  {col}[{flag}]\033[0m {c['check']}"
              + (f" — {c.get('note')}" if c.get("note") else ""))
    cen = res.get("census", {}).get("by_split", {})
    print(f"  census by split: " + "  ".join(f"{k}={v}" for k, v in cen.items()))


def _print_stage1(res: Dict[str, Any]) -> None:
    print(f"\n==== STAGE 1 — UNIVARIATE SCREEN ====  scope={res['scope']}  "
          f"window={res['window']}  N={res['n_pooled']}")
    if res["exploratory"]:
        print("  \033[33mEXPLORATORY MODE — hypotheses only, not a shippable "
              "verdict. Confirm on the full run.\033[0m")
    verdicts: Dict[str, int] = {}
    for r in res["features"]:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    print("  verdicts: " + "  ".join(f"{k}={v}" for k, v in sorted(verdicts.items())))
    if res["ranked_actionable"]:
        print("  ranked survivors/hypotheses (by |ΔexpR_val|):")
        for r in res["ranked_actionable"][:15]:
            print(f"    {r['feature']:24} {r['verdict']:14} "
                  f"Δval={r['delta_val']}  ->{r.get('actionable_at')}  [{r['timing']}]")
    else:
        print("  no survivors — a valid, publishable null (Stage 2 emits its null).")
    if res.get("detection_queue"):
        print("  detection queue (change-detection-and-rerun): "
              + ", ".join(res["detection_queue"]))
    if res.get("anatomy_promoted"):
        print("  anatomy-promoted (entry-fault markers): "
              + ", ".join(res["anatomy_promoted"]))


def _print_stage1_discovery(res: Dict[str, Any]) -> None:
    print(f"\n==== PHASE A — DISCOVERY (candidates only) ====  "
          f"window={res['window']}  N={res['n_discovery']}")
    print(f"  \033[33m{res['language_stamp']}\033[0m")
    verdicts: Dict[str, int] = {}
    for r in res["features"]:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    print("  verdicts: " + "  ".join(f"{k}={v}" for k, v in sorted(verdicts.items())))
    if res["ranked_candidates"]:
        print("  ranked candidates (by |Δdisc|)  — CANDIDATE, luck not ruled out:")
        for r in res["ranked_candidates"][:15]:
            print(f"    {r['feature']:24} {r['verdict']:16} "
                  f"Δdisc={r['delta_disc']}  [{r['timing']}]")
    else:
        print("  no candidates — a valid null on discovery.")
    print(f"\n  \033[1mAPPROVAL TOKEN: {res['token']}\033[0m")
    print("  To confirm on validation (ONE shot):")
    print(f"    python -m backtest.diagnostics.edge_engine --approve {res['token']}")
    print("    python -m backtest.diagnostics.edge_engine --phase confirm")


def _print_stage2(res: Dict[str, Any]) -> None:
    print(f"\n==== STAGE 2 — EV SCORE ====  scope={res['scope']}  "
          f"verdict={res['verdict']}")
    print(f"  {res.get('note', '')}")
    if res["verdict"] == "EV_MODEL":
        m = res["ev_model"]
        print(f"  ridge λ={m['lambda']}  features={res['features_in']}")
        pb = res.get("pass_bar", {})
        print(f"  pass bar: spearman={pb.get('spearman')} "
              f"top-q expR={pb.get('top_quintile_expR')} "
              f"(pop {pb.get('population_expR')})")
        top = sorted(m["coefficients"].items(), key=lambda kv: -abs(kv[1]))[:8]
        print("  top coefficients: " + "  ".join(f"{k}={v}" for k, v in top))
    if res.get("legacy_score_spearman_val") is not None:
        print(f"  legacy score spearman (val): {res['legacy_score_spearman_val']} "
              "(the 'beats the old score' receipt)")


def _print_stage3(res: Dict[str, Any]) -> None:
    print(f"\n==== STAGE 3 — EXIT OPTIMISER ====  scope={res['scope']}  "
          f"N={res['n_trades']}  recipes={res.get('n_recipes')}")
    if res.get("exploratory"):
        print("  \033[33mEXPLORATORY — cluster recipes are hypotheses, not "
              "shippable.\033[0m")
    g = res.get("global_recipe")
    if g:
        print(f"  global best: {g['recipe']}  net expR={g['net_expR']}  "
              f"paired-diff={g['paired_diff']} CI={g['diff_ci']}")
    else:
        print("  global: nothing beat baseline on discovery — baseline is default.")
    print(f"  cluster mode={res.get('cluster_mode')}  exit_mode={res.get('exit_mode')}  "
          f"forced-close={res.get('forced_close_pct')}%")
    for c in res.get("clusters", []):
        sel = c.get("selected") or {}
        print(f"    [{c['cluster']:20}] n={c['n']:5} ships={c['ships']:16} "
              f"({c['verdict']})" + (f" Δ={sel.get('paired_diff')}" if sel else ""))


def _print_stage4(res: Dict[str, Any]) -> None:
    print(f"\n==== STAGE 4 — VERDICT ====  scope={res['scope']}")
    if res["scope"] != "verdict":
        print(f"  {res['verdict']} — {res.get('note')}")
        print("  (report written: edge_engine_report.md)")
        return
    print(f"  \033[1mVERDICT: {res['verdict']}  ·  robustness: {res['robustness']}\033[0m")
    h = res.get("holdout", {})
    print(f"  Holdout N={h.get('n')} expR_net={h.get('expR_net')} "
          f"block CI={h.get('ci_block')} pos_q={h.get('pos_quarters')}")
    print(f"  vs baseline diff CI={h.get('vs_baseline_diff_ci')}")
    print(f"  pair set={res.get('pair_set')}")
    print(f"  WAR 2026 (never pooled): {res.get('war_2026')}")
    print(f"  BTC: {res.get('btc', {}).get('status')}")
    for cav in res.get("caveats", []):
        print(f"  ⚠️  {cav}")
    print("  (report written: edge_engine_report.md)")


if __name__ == "__main__":
    main()
