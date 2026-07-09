"""
EDGE LAB — v2 STEP 2: Univariate pooled DISCOVERY scorecard (Track A, spec §4, §2, §3).

WHAT THIS IS (plain English)
  Look at every entry-legal feature ONE AT A TIME, on the pooled FX+Gold DISCOVERY
  population (2008-2016), and rank them by how much real signal each carries. This
  layer is PERMISSIVE: it ranks everything and kills nothing (spec §2 Layer 1). The
  strict SHIP GATE (spec §2 Layer 2) is a LATER step; Step 2 only sorts features into
  two visible buckets — "ship-gate queue" (worth proving) vs "interesting, not proven".

WHAT THIS IS NOT
  - Not validation. Touches DISCOVERY (2008-2016) only. C5 holdout stays sealed.
  - Not a filter. No feature is dropped, gated, or shipped here.
  - Not BTC. BTC is standalone and starts 2017 (no discovery-era data) — deferred to
    Step 3 per the trader's call. This runs Book A + Book B (FX + Gold) only.

METHODOLOGY (each choice justified against spec §3 — "correct test per feature type")
  Target variable: r_realised (source-of-truth per-trade R). Every stat is on R, seed 42.

  Per feature, classified by type, we compute:
    - CONTINUOUS  (spec §3): Spearman rank-corr trend test (monotonic signal) + Mutual
                   Information (the U-shape / non-linear catcher a linear corr misses)
                   + a DECILE curve (mean R per decile, so the shape is visible) +
                   top-vs-bottom-decile diff-CI (bootstrap). MI reported only where the
                   feature's valid subpop >= 500 (spec §3: MI is upward-biased on thin
                   samples) — else marked "MI unreliable".
    - ORDINAL     (ordered categories: ob_walkback_depth, ob_touches, bos_tier):
                   Spearman on the ordered code — NOT Kruskal. This is the exact v1 bug
                   the spec flags (§0.3): Kruskal throws away order, losing monotonic
                   power. Full per-level curve reported.
    - NOMINAL     (unordered: pair, session, ob_session, event, ...): Kruskal-Wallis
                   across levels (correct — no order to exploit) + per-level curve.
    - BINARY      (fvg_present, pd_alignment, trend_pd_agree, ...): two-sample bootstrap
                   diff-CI between the two groups (spec §3 binary rule).

  EFFECT SIZE (spec §2/§10): the effect a feature moves R by, in R units. For
    continuous/ordinal = |mean R(top bucket) - mean R(bottom bucket)|; for nominal =
    |best level mean - worst level mean|; for binary = |group1 - group0|. Compared to
    the provisional 0.05R single-feature effect floor (spec §14) as an INFORMATIONAL
    column here (the floor is a Layer-2 SHIP gate, not applied in discovery).

  CONSISTENCY (spec §3): per-quarter pos-fraction on the strongest bucket vs weakest,
    reusing the spine's _pos_quarters (a quarter counts only at N>=MIN_QUARTER_N=30).

  SHRINKAGE (spec §2/§3 ADD): thin buckets' mean R is Bayesian-shrunk toward the pooled
    mean (empirical-Bayes; shrink weight = n/(n+K), K = median bucket N). Reported
    alongside the raw bucket mean so a thin bucket cannot masquerade as a strong signal.
    This is NEW in v2 (not in the spine).

  FDR (spec §2/§8): Benjamini-Hochberg across the univariate p-values, shown as an
    INFORMATIONAL column only — never a gate here (spec §8: FDR -> univariate scorecard
    only; the search-level scorekeeper PBO/DSR belongs to the interaction track).

  VIEWS (spec §9): the scorecard runs on BOTH populations, side by side —
    gates-off (all scores, the trust/collection view) and live-gated (score >= floor,
    what current live filtering would actually trade). A signal can differ between them.

DISPOSITION (spec §2/§0.5 — two buckets, nothing floats):
  A feature is queued to the SHIP-GATE QUEUE (worth proving in later steps) when, on the
  gates-off discovery view, ALL of:
    (a) its type-correct test is significant (p < 0.05), AND
    (b) its effect-size CI excludes 0 (a real gap, not noise), AND
    (c) |effect| >= EFFECT_FLOOR_SINGLE_R (0.05R provisional).
  Everything else is "interesting, not proven" — kept fully visible, never shipped.
  This is a DISCOVERY triage, not the ship gate; the strict five-point gate (§2 Layer 2)
  is applied later, and the trader disposes.

REUSE (spec §11 Step-2 reuse list): bootstrap_ci, bootstrap_diff_ci, _ci_excludes_zero,
  _pos_quarters, _cell_stats, benjamini_hochberg, pooled_fx_gold, split_frame, SPLITS,
  MIN_BUCKET_N, QUARTER_SIGN_FRAC, FDR_Q — all imported from the spine. FRESH in v2:
  Mutual Information, decile curve, Bayesian shrinkage, Spearman-for-ordinals.

OUTPUT (spec §11 Step-2 exit): one ranked table as JSON + markdown under
  <run_dir>/edge_lab/step2_scorecard.{json,md}, every feature a row, richest on top,
  two-bucket disposition. Deterministic (seed 42). No number without N + window + scope.

CLI:
    python -m backtest.diagnostics.edge_lab_step2 \
        --run-dir backtest/results/h1only_20080102_20251231
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

from backtest.diagnostics import edge_engine as spine
from backtest.diagnostics import edge_lab as scaffold
from backtest.diagnostics.edge_engine import (
    SEED,
    MIN_BUCKET_N,
    MIN_QUARTER_N,
    FDR_Q,
    SCORE_FLOOR_LIVE,
    bootstrap_diff_ci,
    _ci_excludes_zero,
    _pos_quarters,
    benjamini_hochberg,
    pooled_fx_gold,
    split_frame,
    _now_utc,
)

R_COL = "r_realised"
TS_COL = "alert_ts"
DISCOVERY_SPLIT = "DISCOVERY"          # 2008-2016 (spec §14 splits, C5)
N_DECILES = 10
MI_MIN_N = scaffold.MI_RELIABILITY_MIN_N       # 500 (spec §3 / §14)
EFFECT_FLOOR = scaffold.EFFECT_FLOOR_SINGLE_R  # 0.05R provisional (spec §14)

# ── Feature-type overrides (spec §3). The spine buckets everything as continuous
# or "categorical"; §3 needs a finer split. ORDINAL features are ORDERED categories
# that MUST get Spearman (the v1 Kruskal bug the spec calls out, §0.3). NOMINAL are
# genuinely unordered -> Kruskal. Anything with <=2 non-null levels is treated BINARY.
ORDINAL_FEATURES = {
    # ordered levels; None = already numeric-ish (coerced to its own order).
    "ob_walkback_depth": None,   # 0,1,2,... deeper = higher
    "ob_touches": None,          # 1,2,3 touches
    "bos_sequence_count": None,  # count -> ordered
    # NOTE: bos_tier is NOT ordinal — its levels are BOS/CHoCH/Confirm/Range, which
    # are unordered STRUCTURE TYPES, not ranked tiers. It is handled as NOMINAL
    # (Kruskal), the correct §3 test for unordered categories.
}
# Continuous set comes straight from the spine (§3 "continuous"): Spearman + MI + decile.
CONTINUOUS_FEATURES = set(spine.CONTINUOUS_FEATURES)


# ═══════════════════════════════════════════════════════════════════════════
# TYPE CLASSIFICATION (spec §3). Decides which test a feature gets.
# ═══════════════════════════════════════════════════════════════════════════

def _feature_kind(feat: str, s: pd.Series) -> str:
    """Return one of: continuous | ordinal | binary | nominal (spec §3)."""
    nu = int(s.dropna().nunique())
    if nu <= 2:
        return "binary"
    if feat in ORDINAL_FEATURES:
        return "ordinal"
    if feat in CONTINUOUS_FEATURES:
        # a "continuous" column that only ever takes 2 values is really binary
        return "continuous" if nu > 2 else "binary"
    return "nominal"


def _ordinal_codes(feat: str, s: pd.Series) -> Optional[pd.Series]:
    """Map an ordinal feature to its ordered numeric code for Spearman. Returns a
    numeric Series aligned to s (NaN where unmappable), or None if it can't be ordered."""
    mapping = ORDINAL_FEATURES.get(feat)
    if mapping is None:
        # already numeric-ish (walkback depth / touches / counts) -> coerce
        return pd.to_numeric(s, errors="coerce")
    return s.map(mapping)


# ═══════════════════════════════════════════════════════════════════════════
# STAT PRIMITIVES (fresh in v2 where the spine has none; §3 ADD).
# ═══════════════════════════════════════════════════════════════════════════

def _mutual_information(x: np.ndarray, y_r: np.ndarray, discrete: bool) -> Optional[float]:
    """MI between a feature and the SIGN of R (win vs loss) — the U-shape catcher
    (spec §3). Sign, not raw R, so MI reads 'does this feature separate winners from
    losers' regardless of shape. sklearn, seed 42. None if subpop < MI_MIN_N (spec §3:
    MI is upward-biased on thin samples -> unreliable below the floor)."""
    from sklearn.feature_selection import mutual_info_classif
    mask = ~(np.isnan(x) if not discrete else pd.isna(x)) & ~np.isnan(y_r)
    xv, rv = x[mask], y_r[mask]
    if len(xv) < MI_MIN_N:
        return None
    y_sign = (rv > 0).astype(int)
    if len(np.unique(y_sign)) < 2:
        return None
    X = xv.reshape(-1, 1)
    mi = mutual_info_classif(
        X, y_sign, discrete_features=[discrete], random_state=SEED, n_neighbors=3
    )
    return round(float(mi[0]), 5)


def _decile_curve(sub: pd.DataFrame, feat: str) -> List[Dict[str, Any]]:
    """Mean R per decile of a continuous feature (spec §3 'report the full curve').
    Uses rank-based deciles so ties/skew don't collapse bins. Each cell carries N."""
    s = pd.to_numeric(sub[feat], errors="coerce")
    r = sub[R_COL]
    d = pd.DataFrame({"x": s, "r": r}).dropna()
    if len(d) < N_DECILES * 5:
        return []
    d = d.sort_values("x")
    d["bin"] = pd.qcut(d["x"].rank(method="first"), N_DECILES, labels=False)
    out = []
    for b, g in d.groupby("bin"):
        out.append({
            "decile": int(b) + 1,
            "n": int(len(g)),
            "x_lo": round(float(g["x"].min()), 4),
            "x_hi": round(float(g["x"].max()), 4),
            "expR": round(float(g["r"].mean()), 4),
        })
    return out


def _level_curve(sub: pd.DataFrame, feat: str) -> List[Dict[str, Any]]:
    """Mean R per level of a nominal/ordinal/binary feature. Each cell carries N.
    Levels below MIN_BUCKET_N are still shown but flagged thin."""
    out = []
    for lvl, g in sub.groupby(sub[feat].astype("object")):
        vals = g[R_COL].dropna()
        if len(vals) == 0:
            continue
        out.append({
            "level": str(lvl),
            "n": int(len(vals)),
            "expR": round(float(vals.mean()), 4),
            "thin": bool(len(vals) < MIN_BUCKET_N),
        })
    return sorted(out, key=lambda c: c["expR"], reverse=True)


def _bayes_shrink(level_curve: List[Dict[str, Any]], pooled_mean: float) -> None:
    """Empirical-Bayes shrink each bucket mean toward the pooled mean (spec §2/§3 ADD).
    weight = n/(n+K), K = median bucket N. Mutates each cell with expR_shrunk. A thin
    bucket is pulled hard toward pooled -> cannot masquerade as a strong signal."""
    if not level_curve:
        return
    ns = [c["n"] for c in level_curve]
    K = float(np.median(ns)) if ns else 1.0
    for c in level_curve:
        w = c["n"] / (c["n"] + K) if (c["n"] + K) > 0 else 0.0
        c["expR_shrunk"] = round(w * c["expR"] + (1 - w) * pooled_mean, 4)


# ═══════════════════════════════════════════════════════════════════════════
# PER-FEATURE SCORECARD ROW (spec §2 Layer 1 — rank, never kill).
# ═══════════════════════════════════════════════════════════════════════════

def _score_continuous(sub: pd.DataFrame, feat: str, pooled_mean: float) -> Dict[str, Any]:
    s = pd.to_numeric(sub[feat], errors="coerce")
    r = sub[R_COL]
    d = pd.DataFrame({"x": s, "r": r}).dropna()
    n = len(d)
    row: Dict[str, Any] = {"kind": "continuous", "n_valid": int(n)}
    if n < MIN_BUCKET_N * 2:
        row["note"] = "subpop too thin for a decile split"
        return row
    rho, p = spearmanr(d["x"], d["r"])
    row["spearman_rho"] = round(float(rho), 4)
    row["p_value"] = float(p)
    row["mi"] = _mutual_information(d["x"].to_numpy(), d["r"].to_numpy(), discrete=False)
    row["mi_reliable"] = row["mi"] is not None
    curve = _decile_curve(sub, feat)
    row["curve"] = curve
    if curve:
        top, bot = curve[-1], curve[0]
        # effect = top-decile mean minus bottom-decile mean, with a bootstrap diff-CI
        top_r = d[d["x"] >= top["x_lo"]]["r"].to_numpy()
        bot_r = d[d["x"] <= bot["x_hi"]]["r"].to_numpy()
        lo, hi = bootstrap_diff_ci(top_r, bot_r)
        row["effect_r"] = round(top["expR"] - bot["expR"], 4)
        row["effect_ci_lo"], row["effect_ci_hi"] = lo, hi
        row["effect_ci_excludes_0"] = _ci_excludes_zero(lo, hi)
        # consistency: pos-quarters on top vs bottom decile subpop
        top_sub = sub.loc[d[d["x"] >= top["x_lo"]].index]
        pos, cnt = _pos_quarters(top_sub, R_COL, TS_COL)
        row["top_pos_quarters"] = f"{pos}/{cnt}"
    return row


def _score_ordinal(sub: pd.DataFrame, feat: str, pooled_mean: float) -> Dict[str, Any]:
    codes = _ordinal_codes(feat, sub[feat])
    r = sub[R_COL]
    d = pd.DataFrame({"x": codes, "r": r}).dropna()
    n = len(d)
    row: Dict[str, Any] = {"kind": "ordinal", "n_valid": int(n)}
    if n < MIN_BUCKET_N:
        row["note"] = "subpop too thin"
        return row
    rho, p = spearmanr(d["x"], d["r"])          # spec §3: Spearman, NOT Kruskal
    row["spearman_rho"] = round(float(rho), 4)
    row["p_value"] = float(p)
    row["test"] = "spearman(ordinal)"
    curve = _level_curve(sub, feat)
    _bayes_shrink(curve, pooled_mean)
    row["curve"] = curve
    # Effect = ΔR between the ordered EXTREME buckets, but ONLY over levels that clear
    # MIN_BUCKET_N. A raw extreme level with N=1 (e.g. bos_sequence_count==10) must not
    # define the effect — it produced a bogus −0.92R before this guard. Walk inward to
    # the nearest testable levels; if fewer than two qualify, report no effect (the
    # Spearman ρ/p still carries the monotonic signal).
    codemap = {c["level"]: _ordinal_codes(feat, pd.Series([c["level"]])).iloc[0]
               for c in curve}
    ordered = [c for c in sorted(
        curve, key=lambda c: (codemap.get(c["level"]) if not pd.isna(
            codemap.get(c["level"])) else -1e9))
        if c["n"] >= MIN_BUCKET_N]
    if len(ordered) >= 2:
        lo_c, hi_c = ordered[0], ordered[-1]
        lo_r = sub[sub[feat].astype(str) == lo_c["level"]][R_COL].to_numpy()
        hi_r = sub[sub[feat].astype(str) == hi_c["level"]][R_COL].to_numpy()
        clo, chi = bootstrap_diff_ci(hi_r, lo_r)
        row["effect_r"] = round(hi_c["expR"] - lo_c["expR"], 4)
        row["effect_ci_lo"], row["effect_ci_hi"] = clo, chi
        row["effect_ci_excludes_0"] = _ci_excludes_zero(clo, chi)
        row["effect_levels"] = f"{lo_c['level']}..{hi_c['level']}"
    return row


def _score_nominal(sub: pd.DataFrame, feat: str, pooled_mean: float) -> Dict[str, Any]:
    d = sub[[feat, R_COL]].dropna()
    n = len(d)
    row: Dict[str, Any] = {"kind": "nominal", "n_valid": int(n)}
    groups = [g[R_COL].to_numpy() for _, g in d.groupby(d[feat].astype("object"))
              if len(g) >= MIN_QUARTER_N]
    if len(groups) < 2:
        row["note"] = "fewer than 2 testable levels"
        return row
    h, p = kruskal(*groups)                      # spec §3: Kruskal for unordered
    row["kruskal_h"] = round(float(h), 4)
    row["p_value"] = float(p)
    row["test"] = "kruskal(nominal)"
    curve = _level_curve(sub, feat)
    _bayes_shrink(curve, pooled_mean)
    row["curve"] = curve
    # Effect = best minus worst level mean, but ONLY across levels clearing
    # MIN_BUCKET_N (a thin best/worst bucket would inflate the gap). curve is already
    # sorted by expR descending, so filter then take the extremes.
    testable = [c for c in curve if c["n"] >= MIN_BUCKET_N]
    if len(testable) >= 2:
        best, worst = testable[0], testable[-1]
        br = sub[sub[feat].astype(str) == best["level"]][R_COL].to_numpy()
        wr = sub[sub[feat].astype(str) == worst["level"]][R_COL].to_numpy()
        clo, chi = bootstrap_diff_ci(br, wr)
        row["effect_r"] = round(best["expR"] - worst["expR"], 4)
        row["effect_ci_lo"], row["effect_ci_hi"] = clo, chi
        row["effect_ci_excludes_0"] = _ci_excludes_zero(clo, chi)
        row["effect_levels"] = f"{worst['level']}..{best['level']}"
        pos, cnt = _pos_quarters(
            sub[sub[feat].astype(str) == best["level"]], R_COL, TS_COL)
        row["best_pos_quarters"] = f"{pos}/{cnt}"
    return row


def _score_binary(sub: pd.DataFrame, feat: str, pooled_mean: float) -> Dict[str, Any]:
    d = sub[[feat, R_COL]].dropna()
    n = len(d)
    row: Dict[str, Any] = {"kind": "binary", "n_valid": int(n)}
    levels = sorted(d[feat].astype("object").unique(), key=lambda v: str(v))
    if len(levels) != 2:
        row["note"] = f"expected 2 levels, saw {len(levels)}"
        return row
    a = d[d[feat] == levels[1]][R_COL].to_numpy()
    b = d[d[feat] == levels[0]][R_COL].to_numpy()
    clo, chi = bootstrap_diff_ci(a, b)           # spec §3: binary diff-CI
    row["test"] = "bootstrap_diff_ci(binary)"
    row["level_hi"], row["level_lo"] = str(levels[1]), str(levels[0])
    row["expR_hi"] = round(float(a.mean()), 4) if len(a) else None
    row["expR_lo"] = round(float(b.mean()), 4) if len(b) else None
    row["n_hi"], row["n_lo"] = int(len(a)), int(len(b))
    row["effect_r"] = (round(float(a.mean()) - float(b.mean()), 4)
                       if len(a) and len(b) else None)
    row["effect_ci_lo"], row["effect_ci_hi"] = clo, chi
    row["effect_ci_excludes_0"] = _ci_excludes_zero(clo, chi)
    # p-value proxy for binary: CI-excludes-0 IS the significance call; also give a
    # sign-consistency read via pos-quarters on the stronger group.
    stronger_level = levels[1] if (row["expR_hi"] or -9) >= (row["expR_lo"] or -9) else levels[0]
    pos, cnt = _pos_quarters(sub[sub[feat] == stronger_level], R_COL, TS_COL)
    row["strong_pos_quarters"] = f"{pos}/{cnt}"
    curve = _level_curve(sub, feat)
    _bayes_shrink(curve, pooled_mean)
    row["curve"] = curve
    return row


def score_feature(sub: pd.DataFrame, feat: str, pooled_mean: float) -> Dict[str, Any]:
    """One scorecard row for one feature on one population view (spec §2 Layer 1)."""
    s = sub[feat]
    kind = _feature_kind(feat, s)
    if kind == "continuous":
        row = _score_continuous(sub, feat, pooled_mean)
    elif kind == "ordinal":
        row = _score_ordinal(sub, feat, pooled_mean)
    elif kind == "binary":
        row = _score_binary(sub, feat, pooled_mean)
    else:
        row = _score_nominal(sub, feat, pooled_mean)
    row["feature"] = feat
    row["timing"] = scaffold.classify_timing(feat)
    return row


# ═══════════════════════════════════════════════════════════════════════════
# RANKING + DISPOSITION (spec §2/§0.5 — two buckets, richest on top).
# ═══════════════════════════════════════════════════════════════════════════

def _rank_key(row: Dict[str, Any]) -> Tuple:
    """Richest signal on top: (a) significant + CI-excludes-0 + over floor first, then
    (b) by |effect|, then (c) by MI (when present). Rows with no computable effect sink."""
    eff = row.get("effect_r")
    eff_abs = abs(eff) if isinstance(eff, (int, float)) else -1.0
    sig = 1 if row.get("effect_ci_excludes_0") else 0
    mi = row.get("mi") or 0.0
    return (sig, eff_abs, mi)


def _disposition(row: Dict[str, Any]) -> str:
    """Two buckets, nothing floats (spec §0.5). Ship-gate queue requires: type-test
    significant (p<0.05 where a p exists) AND effect-CI excludes 0 AND |effect|>=floor.
    Binary has no p (its CI IS the test) -> CI-excludes-0 + floor suffices."""
    eff = row.get("effect_r")
    if not isinstance(eff, (int, float)):
        return "interesting_not_proven"
    over_floor = abs(eff) >= EFFECT_FLOOR
    ci_ok = bool(row.get("effect_ci_excludes_0"))
    p = row.get("p_value")
    p_ok = True if p is None else (p < 0.05)
    if over_floor and ci_ok and p_ok:
        return "ship_gate_queue"
    return "interesting_not_proven"


def build_scorecard(sub: pd.DataFrame, features: List[str], view: str,
                    window: str) -> List[Dict[str, Any]]:
    """Score every feature on one view, attach FDR (informational), rank, dispose."""
    pooled_mean = float(sub[R_COL].dropna().mean())
    rows = [score_feature(sub, f, pooled_mean) for f in features]
    # FDR across the univariate p-values (informational only, spec §8).
    pvals = [r.get("p_value") for r in rows]
    reject = benjamini_hochberg(pvals, q=FDR_Q)
    for r, rej in zip(rows, reject):
        r["fdr_reject"] = bool(rej)      # informational column, NOT a gate
        r["view"] = view
        r["window"] = window
        r["scope"] = "pooled_fx_gold"
        r["disposition"] = _disposition(r)
    rows.sort(key=_rank_key, reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# REPORT WRITERS (JSON + markdown; spec §11 Step-2 exit).
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _md_table(rows: List[Dict[str, Any]], view: str) -> str:
    lines = [f"### View: {view}", "",
             "| # | feature | kind | timing | N | effect R | eff CI | sig | MI | "
             "test stat | p | FDR | consistency | disposition |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = (f"[{_fmt(r.get('effect_ci_lo'))}, {_fmt(r.get('effect_ci_hi'))}]"
              if r.get("effect_ci_lo") is not None else "—")
        stat = (f"ρ={_fmt(r.get('spearman_rho'))}" if "spearman_rho" in r
                else f"H={_fmt(r.get('kruskal_h'))}" if "kruskal_h" in r
                else r.get("test", "—"))
        cons = (r.get("top_pos_quarters") or r.get("best_pos_quarters")
                or r.get("strong_pos_quarters") or "—")
        mi = _fmt(r.get("mi"), 5) if r.get("mi") is not None else (
            "n/a" if r["kind"] not in ("continuous",) else "thin")
        disp = "🟢 QUEUE" if r["disposition"] == "ship_gate_queue" else "· not proven"
        lines.append(
            f"| {r['rank']} | {r['feature']} | {r['kind']} | {r['timing']} | "
            f"{r.get('n_valid','—')} | {_fmt(r.get('effect_r'))} | {ci} | "
            f"{'✅' if r.get('effect_ci_excludes_0') else '—'} | {mi} | {stat} | "
            f"{_fmt(r.get('p_value'), 4) if r.get('p_value') is not None else '—'} | "
            f"{'rej' if r.get('fdr_reject') else '—'} | {cons} | {disp} |")
    return "\n".join(lines)


def write_reports(run_dir: str, results: Dict[str, Any]) -> Tuple[str, str]:
    out_dir = os.path.join(run_dir, "edge_lab")
    os.makedirs(out_dir, exist_ok=True)
    json_p = os.path.join(out_dir, "step2_scorecard.json")
    md_p = os.path.join(out_dir, "step2_scorecard.md")
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    meta = results["meta"]
    md = [
        "# Edge Lab v2 — Step 2: Univariate Pooled Discovery Scorecard",
        "",
        f"- **Scope:** {meta['scope']} (Book A + Book B; BTC deferred to Step 3)",
        f"- **Window:** {meta['window']} (DISCOVERY split; C5 holdout sealed)",
        f"- **Target:** `{R_COL}` (per-trade R, source of truth)",
        f"- **Seed:** {meta['seed']} · **Generated:** {meta['generated_utc']}",
        f"- **N (gates-off):** {meta['n_gates_off']} · "
        f"**N (live-gated, score≥{SCORE_FLOOR_LIVE}):** {meta['n_gated']}",
        f"- **Effect floor (informational):** {EFFECT_FLOOR}R · "
        f"**MI reliability floor:** subpop≥{MI_MIN_N}",
        "",
        "**Layer 1 = permissive discovery — ranks everything, kills nothing. "
        "Disposition is a triage into two visible buckets; the strict SHIP GATE and "
        "the holdout are LATER steps. Nothing here changes live trading.**",
        "",
        "Test per type (spec §3): continuous → Spearman + MI + decile curve; "
        "ordinal → **Spearman (not Kruskal)**; nominal → Kruskal-Wallis; "
        "binary → bootstrap diff-CI. Effect = top-vs-bottom bucket ΔR. "
        "FDR & effect-floor columns are informational, not gates.",
        "",
    ]
    for view in ("gates_off", "live_gated"):
        md.append(_md_table(results["views"][view], view))
        md.append("")
    with open(md_p, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return json_p, md_p


# ═══════════════════════════════════════════════════════════════════════════
# DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def run(run_dir: str) -> Dict[str, Any]:
    pop, guard = scaffold.load_lab_population(run_dir)
    pooled = pooled_fx_gold(pop)
    disc = split_frame(pooled, DISCOVERY_SPLIT)
    features = scaffold.entry_features(pop)

    gates_off = disc
    gated = disc[pd.to_numeric(disc["score"], errors="coerce") >= SCORE_FLOOR_LIVE]

    window = "2008-01-02..2016-12-31"
    views = {
        "gates_off": build_scorecard(gates_off, features, "gates_off", window),
        "live_gated": build_scorecard(gated, features, "live_gated", window),
    }
    results = {
        "meta": {
            "step": "step2_univariate_pooled_discovery",
            "spec": "SMC_EDGE_LAB_SPEC.md §4 / §2 / §3 / §9",
            "run_id": os.path.basename(run_dir.rstrip("/\\")),
            "scope": "pooled_fx_gold",
            "window": window,
            "seed": SEED,
            "generated_utc": _now_utc(),
            "n_features": len(features),
            "n_gates_off": int(len(gates_off)),
            "n_gated": int(len(gated)),
            "score_floor_live": SCORE_FLOOR_LIVE,
            "effect_floor_single_r": EFFECT_FLOOR,
            "mi_min_n": MI_MIN_N,
            "schema_guard": guard,
        },
        "views": views,
    }
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Edge Lab v2 Step 2 — univariate pooled discovery")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = args.run_dir
    results = run(run_dir)
    json_p, md_p = write_reports(run_dir, results)

    # Console summary — the ship-gate queue, richest on top (gates-off view).
    off = results["views"]["gates_off"]
    queue = [r for r in off if r["disposition"] == "ship_gate_queue"]
    print(f"[STEP2] features scored: {results['meta']['n_features']} · "
          f"N gates-off={results['meta']['n_gates_off']} gated={results['meta']['n_gated']}")
    print(f"[STEP2] ship-gate QUEUE (gates-off): {len(queue)}")
    for r in queue:
        print(f"   #{r['rank']:>2} {r['feature']:28} eff={_fmt(r.get('effect_r'))}R "
              f"CI=[{_fmt(r.get('effect_ci_lo'))},{_fmt(r.get('effect_ci_hi'))}] "
              f"MI={_fmt(r.get('mi'),5) if r.get('mi') is not None else '—'} "
              f"kind={r['kind']}")
    print(f"[STEP2] wrote {json_p}")
    print(f"[STEP2] wrote {md_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
