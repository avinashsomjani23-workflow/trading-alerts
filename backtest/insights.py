"""Backtest analytics. Pure functions — no file I/O, no side effects.

Input:  a pandas DataFrame of combined trades from one or more runs.
        Expected columns (all produced by h1_only_reporting):
          pair, session, entry_zone, exit_reason, r_realised,
          r_if_exit_tp1, r_if_exit_tp2, mfe_r, mae_r,
          score, fvg_present, sweep_present,
          killzone_pts, freshness_pts, structure_pts,
          pd_zone, bos_tier, group (added by aggregate_runs)

Output: plain Python dicts and lists — JSON-serializable, rendered by aggregate_runs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr as _spearmanr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filled(df: pd.DataFrame) -> pd.DataFrame:
    """Drop never-filled rows. These are valid fill-rate data but not outcomes."""
    if "exit_reason" in df.columns:
        return df[df["exit_reason"] != "never_filled"].copy()
    return df.copy()


def _fmt_r(v: float) -> str:
    return f"{'+' if v >= 0 else ''}{v:.2f}R"


def win_rate_pct(sub: pd.DataFrame, r_col: str = "r_realised") -> Optional[float]:
    """Directional hit rate on RESOLVED trades: wins / (wins + losses).

    Breakevens (r == 0) are scratches — risk removed, nothing won or lost — so
    they are excluded from BOTH the numerator and the denominator. They are not
    losses and must never drag win rate down. Counting them as the denominator
    understates the strategy's real accuracy; counting them as losses is plain
    wrong. (Expectancy keeps every trade — see compute_overall — so the BE
    slots are never hidden from the bottom line.)

    Returns None when there are no resolved trades (all BE or empty). Callers
    render None as an em-dash, never as 0%. Matches weekly_review.py's
    wins/(wins+losses) convention so backtest and live reports agree.
    """
    if sub.empty or r_col not in sub.columns:
        return None
    wins = int((sub[r_col] > 0).sum())
    losses = int((sub[r_col] < 0).sum())
    resolved = wins + losses
    if resolved == 0:
        return None
    return round(wins / resolved * 100, 1)


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    n_boot: int = 5000,
    ci: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """95% CI on the mean via bootstrap resampling.

    Returns (lower, upper) or (None, None) if fewer than 5 values.
    """
    if len(values) < 5:
        return None, None
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(42)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return round(lo, 3), round(hi, 3)


def sharpe(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    std = float(arr.std())
    return round(float(arr.mean()) / std, 3) if std > 0 else 0.0


def max_drawdown_r(values: List[float]) -> float:
    equity = np.cumsum(np.array(values, dtype=float))
    peak = np.maximum.accumulate(equity)
    return round(float((peak - equity).max()), 3)


def longest_losing_streak(values: List[float]) -> int:
    streak = max_streak = 0
    for v in values:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def capture_pct(wins: pd.DataFrame, r_col: str = "r_realised", mfe_col: str = "mfe_r") -> float:
    """How much of the available move did winners actually capture?"""
    if wins.empty or mfe_col not in wins.columns:
        return 0.0
    avg_booked = wins[r_col].mean()
    avg_mfe = wins[mfe_col].mean()
    if avg_mfe <= 0:
        return 0.0
    return round(avg_booked / avg_mfe * 100, 1)


# ---------------------------------------------------------------------------
# Overall metrics
# ---------------------------------------------------------------------------

def compute_overall(df: pd.DataFrame, r_col: str = "r_realised") -> Dict[str, Any]:
    f = _filled(df)
    if f.empty or r_col not in f.columns:
        return {"n": 0}

    vals = f[r_col].tolist()
    wins = f[f[r_col] > 0]
    losses = f[f[r_col] < 0]
    ci_lo, ci_hi = bootstrap_ci(vals)

    return {
        "n":                     len(f),
        "wins":                  len(wins),
        "losses":                len(losses),
        "breakevens":            len(f) - len(wins) - len(losses),
        "win_rate_pct":          win_rate_pct(f, r_col),
        "expectancy_r":          round(float(f[r_col].mean()), 3),
        "ci_lo_95":              ci_lo,
        "ci_hi_95":              ci_hi,
        "ci_excludes_zero":      ci_lo is not None and ci_lo > 0,
        "sharpe":                sharpe(vals),
        "max_dd_r":              max_drawdown_r(vals),
        "longest_losing_streak": longest_losing_streak(vals),
        "avg_win_r":             round(float(wins[r_col].mean()), 3) if len(wins) else 0.0,
        "avg_loss_r":            round(float(losses[r_col].mean()), 3) if len(losses) else 0.0,
        "avg_mfe_r":             round(float(wins["mfe_r"].mean()), 3) if (len(wins) and "mfe_r" in wins.columns) else None,
        "avg_mae_r":             round(float(losses["mae_r"].mean()), 3) if (len(losses) and "mae_r" in losses.columns) else None,
        "win_capture_pct":       capture_pct(wins, r_col),
    }


# ---------------------------------------------------------------------------
# Pair × session matrix
# ---------------------------------------------------------------------------

def pair_session_matrix(df: pd.DataFrame, r_col: str = "r_realised") -> List[Dict[str, Any]]:
    f = _filled(df)
    if f.empty or "pair" not in f.columns or "session" not in f.columns:
        return []

    rows = []
    for (pair, session), grp in f.groupby(["pair", "session"]):
        n = len(grp)
        exp = float(grp[r_col].mean())
        ci_lo, ci_hi = bootstrap_ci(grp[r_col].tolist())

        # Confidence badge: green ≥ 20, yellow 10-19, red < 10
        if n >= 20:
            badge = "green"
        elif n >= 10:
            badge = "yellow"
        else:
            badge = "red"

        rows.append({
            "pair":          pair,
            "session":       session,
            "n":             n,
            "win_rate_pct":  win_rate_pct(grp, r_col),
            "expectancy_r":  round(exp, 3),
            "ci_lo_95":      ci_lo,
            "ci_hi_95":      ci_hi,
            "live_eligible": badge == "green" and exp > 0 and (ci_lo is not None and ci_lo > 0),
            "confidence":    badge,
        })

    return sorted(rows, key=lambda r: r["expectancy_r"], reverse=True)


# ---------------------------------------------------------------------------
# Instrument-specific verdict thresholds
# ---------------------------------------------------------------------------

FOREX_PAIRS   = {"EURUSD", "NZDUSD", "USDJPY", "USDCHF"}
PREMIUM_PAIRS = {"XAUUSD", "GOLD", "NAS100"}  # higher thresholds

THRESHOLDS = {
    "forex": {
        "green":  {"expectancy_r": 0.30, "win_rate_pct": 45.0, "ci_excludes_zero": True},
        "yellow": {"expectancy_r": 0.10, "win_rate_pct": 40.0},
    },
    "premium": {
        "green":  {"expectancy_r": 0.50, "win_rate_pct": 50.0, "ci_excludes_zero": True},
        "yellow": {"expectancy_r": 0.20, "win_rate_pct": 45.0},
    },
}


def instrument_verdicts(df: pd.DataFrame, r_col: str = "r_realised") -> Dict[str, Dict]:
    """Per-instrument-group breakdown with instrument-appropriate thresholds."""
    f = _filled(df)
    if f.empty or "pair" not in f.columns:
        return {}

    results = {}
    for group_name, pairs in [("forex", FOREX_PAIRS), ("premium", PREMIUM_PAIRS)]:
        sub = f[f["pair"].isin(pairs)]
        if sub.empty:
            continue
        m = compute_overall(sub, r_col)
        t = THRESHOLDS[group_name]

        # win_rate_pct is None when no trade resolved (all breakevens). A group
        # with no resolved trades cannot clear a WR gate -> treat as 0 for the
        # comparison, which lands it in RED.
        wr = m["win_rate_pct"] if m["win_rate_pct"] is not None else 0.0

        if (m["expectancy_r"] >= t["green"]["expectancy_r"] and
                wr >= t["green"]["win_rate_pct"] and
                m.get("ci_excludes_zero", False)):
            verdict = "GREEN"
        elif (m["expectancy_r"] >= t["yellow"]["expectancy_r"] and
              wr >= t["yellow"]["win_rate_pct"]):
            verdict = "YELLOW"
        else:
            verdict = "RED"

        results[group_name] = {
            **m,
            "thresholds": t,
            "verdict":    verdict,
            "note": (
                "Forex: thresholds +0.3R / 45% WR (post-spread survival approx)"
                if group_name == "forex"
                else "Gold/NAS100: thresholds +0.5R / 50% WR (higher real-world spread and slippage cost)"
            ),
        }
    return results


# ---------------------------------------------------------------------------
# Confluence attribution
# ---------------------------------------------------------------------------

def _confluence_mask(df: pd.DataFrame, name: str) -> pd.Series:
    mapping = {
        "fvg":          ("fvg_present",     lambda s: s == True),
        "sweep":        ("sweep_present",   lambda s: s == True),
        "killzone":     ("killzone_pts",    lambda s: s > 0),
        "freshness":    ("freshness_pts",   lambda s: s > 0),
        "structure":    ("structure_pts",   lambda s: s > 0),
        # Direction-aware: a trade only earns PD credit when it is on the right
        # side of the range for its direction (long in discount / short in
        # premium). The old mask counted any discount/premium regardless of
        # direction, so a short in discount (counter-PD, a red flag) scored the
        # same as a short in premium. Falls back to pd_zone for older runs that
        # predate the pd_alignment column.
        "pd_alignment": ("pd_alignment",    lambda s: s == "aligned"),
    }
    col, fn = mapping.get(name, (None, None))
    if col is None or col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    try:
        return fn(df[col])
    except Exception:
        return pd.Series([False] * len(df), index=df.index)


def confluence_attribution(df: pd.DataFrame, r_col: str = "r_realised") -> Dict[str, Any]:
    f = _filled(df)
    if f.empty:
        return {}

    results = {}
    for name in ["fvg", "sweep", "killzone", "freshness", "structure", "pd_alignment"]:
        mask = _confluence_mask(f, name)
        with_c  = f[mask]
        without = f[~mask]

        n_with    = len(with_c)
        n_without = len(without)
        exp_with    = round(float(with_c[r_col].mean()), 3)  if n_with    > 0 else None
        exp_without = round(float(without[r_col].mean()), 3) if n_without > 0 else None
        uplift = (round(exp_with - exp_without, 3)
                  if exp_with is not None and exp_without is not None else None)

        if n_with < 15:
            verdict = "insufficient data (< 15 trades with confluence)"
        elif uplift is None:
            verdict = "insufficient data"
        elif uplift >= 0.20:
            verdict = "EARNS ITS WEIGHT — keep score points"
        elif uplift >= 0.05:
            verdict = "MARGINAL — consider reducing score weight"
        else:
            verdict = "NOISE — remove from scoring or set weight to 0"

        results[name] = {
            "n_with":    n_with,
            "n_without": n_without,
            "exp_with":  exp_with,
            "exp_without": exp_without,
            "uplift_r":  uplift,
            "verdict":   verdict,
        }
    return results


# ---------------------------------------------------------------------------
# Score validation
# ---------------------------------------------------------------------------

def score_validation(df: pd.DataFrame, r_col: str = "r_realised") -> Dict[str, Any]:
    f = _filled(df)
    if f.empty or "score" not in f.columns or len(f) < 10:
        return {"verdict": "insufficient data"}

    # Spearman correlation
    if _HAS_SCIPY:
        r_stat, p_val = _spearmanr(f["score"], f[r_col])
    else:
        # Fallback: rank correlation manually
        score_rank = f["score"].rank()
        r_rank = f[r_col].rank()
        r_stat = float(score_rank.corr(r_rank, method="spearman"))
        p_val = None

    # Bucket analysis
    edges = [(0, 2, "0–2"), (2, 3, "2–3"), (3, 4, "3–4"), (4, 5, "4–5"), (5, 99, "5+")]
    buckets = []
    for lo, hi, label in edges:
        sub = f[(f["score"] >= lo) & (f["score"] < hi)]
        if sub.empty:
            continue
        buckets.append({
            "bucket":       label,
            "n":            len(sub),
            "win_rate_pct": win_rate_pct(sub, r_col),
            "expectancy_r": round(float(sub[r_col].mean()), 3),
        })

    # Monotonicity — do expectancy values rise bucket-to-bucket?
    exp_vals = [b["expectancy_r"] for b in buckets]
    pairs = list(zip(exp_vals, exp_vals[1:]))
    rises = sum(1 for a, b in pairs if b > a)
    monotone_pct = rises / len(pairs) if pairs else 0

    if r_stat > 0.25 and monotone_pct >= 0.70:
        verdict = "WORKS — higher score reliably leads to better trades"
    elif r_stat > 0.10 or monotone_pct >= 0.50:
        verdict = "WEAK — partial relationship; score needs recalibration"
    else:
        verdict = "BROKEN — score does not predict trade outcome; redesign required"

    return {
        "spearman_r":    round(float(r_stat), 3),
        "p_value":       round(float(p_val), 4) if p_val is not None else None,
        "monotone_pct":  round(monotone_pct, 2),
        "buckets":       buckets,
        "verdict":       verdict,
    }


# ---------------------------------------------------------------------------
# Setup badge validation (Phase 2 email banners — A+/First Pullback/Late-Trend
# Chase). smc_detector.classify_setup output, logged per EDGE_ENGINE_HANDOFF.md
# §3 "Setup badges". A banner name is a CLAIM, not a signal, until this runs.
# ---------------------------------------------------------------------------

def setup_badge_validation(df: pd.DataFrame, r_col: str = "r_realised") -> Dict[str, Any]:
    """Bucket trades by setup_badge, test whether the badge predicts outcome.

    Per-badge N/WR/expectancy/bootstrap CI, plus Spearman's rank correlation
    between badge conviction (caution=-1, none=0, premium=+1) and r_realised —
    the standing Stage-1 rule (EDGE_ENGINE_HANDOFF.md §4): any new insight must
    clear this bar before it's trusted, same as the confidence score was.
    """
    f = _filled(df)
    if f.empty or "setup_badge" not in f.columns:
        return {"verdict": "insufficient data"}

    f = f.copy()
    f["setup_badge"] = f["setup_badge"].fillna("(none)")
    kind_col = f["setup_badge_kind"] if "setup_badge_kind" in f.columns else pd.Series(
        [None] * len(f), index=f.index)
    _kind_rank = {"caution": -1, None: 0, "premium": 1}
    f["_badge_rank"] = kind_col.map(lambda k: _kind_rank.get(k, 0))

    buckets = []
    for badge in f["setup_badge"].unique():
        sub = f[f["setup_badge"] == badge]
        if sub.empty:
            continue
        vals = sub[r_col].tolist()
        ci_lo, ci_hi = bootstrap_ci(vals)
        buckets.append({
            "badge":        badge,
            "n":            len(sub),
            "win_rate_pct": win_rate_pct(sub, r_col),
            "expectancy_r": round(float(sub[r_col].mean()), 3),
            "ci_lo_95":     ci_lo,
            "ci_hi_95":     ci_hi,
        })
    buckets.sort(key=lambda b: -b["n"])

    n_badged = int((f["setup_badge"] != "(none)").sum())
    if n_badged < 15 or f["_badge_rank"].nunique() < 2:
        return {"verdict": "insufficient data (< 15 badged trades, or only one badge fired)",
                "buckets": buckets}

    if _HAS_SCIPY:
        r_stat, p_val = _spearmanr(f["_badge_rank"], f[r_col])
        r_stat, p_val = float(r_stat), float(p_val)
    else:
        r_stat = float(f["_badge_rank"].rank().corr(f[r_col].rank(), method="spearman"))
        p_val = None

    if p_val is not None and p_val < 0.05 and abs(r_stat) > 0.10:
        verdict = "SIGNAL — badge rank correlates with outcome (p < 0.05)"
    elif abs(r_stat) > 0.10:
        verdict = "DIRECTIONAL, UNCONFIRMED — correlation present but not significant (p >= 0.05)"
    else:
        verdict = "NOISE — badge does not predict outcome"

    return {
        "spearman_r": round(r_stat, 3),
        "p_value":    round(p_val, 4) if p_val is not None else None,
        "n_badged":   n_badged,
        "n_total":    len(f),
        "buckets":    buckets,
        "verdict":    verdict,
    }


# ---------------------------------------------------------------------------
# Group-by-group consistency
# ---------------------------------------------------------------------------

def group_comparison(df: pd.DataFrame, r_col: str = "r_realised") -> Dict[int, Dict]:
    f = _filled(df)
    if f.empty or "group" not in f.columns:
        return {}

    results = {}
    for g in sorted(f["group"].unique()):
        sub = f[f["group"] == g]
        if sub.empty:
            continue
        vals = sub[r_col].tolist()
        ci_lo, ci_hi = bootstrap_ci(vals)
        results[int(g)] = {
            "n":            len(sub),
            "expectancy_r": round(float(sub[r_col].mean()), 3),
            "win_rate_pct": win_rate_pct(sub, r_col),
            "ci_lo_95":     ci_lo,
            "ci_hi_95":     ci_hi,
            "max_dd_r":     max_drawdown_r(vals),
        }
    return results


# ---------------------------------------------------------------------------
# Proximal vs 50% entry comparison
# ---------------------------------------------------------------------------

def entry_zone_comparison(df: pd.DataFrame, r_col: str = "r_realised") -> Dict[str, Any]:
    f = _filled(df)
    if f.empty or "entry_zone" not in f.columns:
        return {}

    results = {}
    for zone, sub in f.groupby("entry_zone"):
        vals = sub[r_col].tolist()
        ci_lo, ci_hi = bootstrap_ci(vals)

        # Fill rate: how many alerts for this zone actually filled?
        if "exit_reason" in df.columns:
            zone_total = df[df.get("entry_zone", pd.Series()) == zone] if "entry_zone" in df.columns else df
            total_for_zone = len(df[df["entry_zone"] == zone]) if "entry_zone" in df.columns else len(sub)
            fill_rate = round(len(sub) / total_for_zone * 100, 1) if total_for_zone > 0 else 0
        else:
            fill_rate = None

        results[str(zone)] = {
            "n":            len(sub),
            "fill_rate_pct": fill_rate,
            "win_rate_pct": win_rate_pct(sub, r_col),
            "expectancy_r": round(float(sub[r_col].mean()), 3) if len(sub) else 0,
            "ci_lo_95":     ci_lo,
            "ci_hi_95":     ci_hi,
        }
    return results


# ---------------------------------------------------------------------------
# OB freshness by touch count
# ---------------------------------------------------------------------------

# alert_seq is the OB's touch number at fire time (excursion-based, same re-arm
# rule as live mitigation): 1 = fresh OB, 2 = touched once before, 3 = touched
# twice before. The engine kills a zone on the 3rd proximal touch, so 3 is the
# deepest a trade can fire on.
_FRESHNESS_BUCKETS = [
    (1, "Fresh (1st touch)"),
    (2, "Touched once (2nd)"),
    (3, "Touched twice (3rd)"),
]


def ob_freshness_comparison(df: pd.DataFrame, r_col: str = "r_realised") -> List[Dict[str, Any]]:
    """Win/loss by OB touch count at fire time.

    One row per touch bucket (fresh / touched-once / touched-twice), ALWAYS all
    three — an empty bucket reports zeros, which is itself the finding: the
    system almost never trades a re-touched OB. Filled+resolved trades only
    (win_rate convention: wins/(wins+losses), breakevens excluded).
    """
    f = _filled(df)
    rows: List[Dict[str, Any]] = []
    for seq, label in _FRESHNESS_BUCKETS:
        sub = f[f["alert_seq"] == seq] if ("alert_seq" in f.columns and not f.empty) else f.iloc[0:0]
        wins = int((sub[r_col] > 0).sum()) if not sub.empty else 0
        losses = int((sub[r_col] < 0).sum()) if not sub.empty else 0
        be = int((sub[r_col] == 0).sum()) if not sub.empty else 0
        rows.append({
            "touch":        seq,
            "label":        label,
            "n":            len(sub),
            "wins":         wins,
            "losses":       losses,
            "breakevens":   be,
            "win_rate_pct": win_rate_pct(sub, r_col),
            "expectancy_r": round(float(sub[r_col].mean()), 3) if len(sub) else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Regime label verification (proxy — uses exit reason distribution)
# ---------------------------------------------------------------------------

def regime_verification(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Regime labels in the plan are pre-assigned from calendar knowledge.
    This function provides a data-driven proxy check using exit reason
    distribution and MFE/MAE patterns to flag potential mislabelled weeks.

    A week labelled BAU that shows mostly SL hits + high MAE is behaving
    like a war week regardless of what was on the calendar.

    Full ATR-based verification is a future enhancement.
    """
    if df.empty or "run_id" not in df.columns:
        return {}

    results = {}
    for run_id, grp in df.groupby("run_id"):
        f = _filled(grp)
        if f.empty:
            continue

        # SL rate as proxy for volatility — high SL% = price moving fast/far
        total = len(f)
        sl_rate = round((f.get("exit_reason", pd.Series()) == "sl").sum() / total * 100, 1) if "exit_reason" in f.columns else None

        # Mean MAE on losing trades — large MAE = deep adverse excursion
        losses = f[f["r_realised"] < 0]
        avg_mae = round(float(losses["mae_r"].mean()), 3) if (len(losses) and "mae_r" in losses.columns) else None

        flag = None
        if sl_rate is not None and sl_rate > 60:
            flag = "high SL rate — may be behaving as war week regardless of label"
        elif avg_mae is not None and avg_mae < -0.5:
            flag = "deep MAE on losses — price moving fast, verify regime label"

        results[str(run_id)] = {
            "sl_rate_pct":    sl_rate,
            "avg_mae_losses": avg_mae,
            "regime_flag":    flag,
        }
    return results


# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

def generate_verdict(
    overall: Dict,
    instrument_v: Dict,
    score_v: Dict,
    pair_sess: List[Dict],
    group_comp: Dict,
) -> Dict[str, Any]:
    """
    Returns a verdict dict: {"overall": "GREEN"|"YELLOW"|"RED", "issues": [...], "live_cells": [...]}
    """
    issues = []

    # 1. Overall CI
    if overall.get("n", 0) < 100:
        issues.append(f"Only {overall.get('n', 0)} trades — well below 200 target. Conclusions are preliminary.")

    if not overall.get("ci_excludes_zero", False):
        ci_lo = overall.get("ci_lo_95")
        issues.append(
            f"Expectancy CI includes zero (low end: {ci_lo}R). "
            "Edge is not proven — could be luck."
        )

    if overall.get("expectancy_r", 0) < 0.3:
        issues.append(f"Expectancy {overall.get('expectancy_r')}R below Forex threshold of +0.3R.")

    wr_overall = overall.get("win_rate_pct")
    if wr_overall is None:
        issues.append("No trade resolved (all breakevens) — win rate undefined.")
    elif wr_overall < 45:
        issues.append(f"Win rate {wr_overall}% below 45% threshold.")

    # 2. Instrument-specific verdicts
    for instr, iv in instrument_v.items():
        if iv["verdict"] == "RED":
            issues.append(f"{instr.upper()} instruments: RED — expectancy {iv.get('expectancy_r')}R, "
                          f"win rate {iv.get('win_rate_pct')}% (thresholds: {iv['thresholds']['green']['expectancy_r']}R / "
                          f"{iv['thresholds']['green']['win_rate_pct']}% WR).")

    # 3. Score
    if "BROKEN" in score_v.get("verdict", ""):
        issues.append("Score does not predict trade outcome. Scoring system needs redesign before live trading.")

    # 4. Live-eligible cells
    live_cells = [c for c in pair_sess if c.get("live_eligible")]
    if len(live_cells) < 3:
        issues.append(f"Only {len(live_cells)} pair×session cells are live-eligible (need ≥ 3: 20+ trades, positive expectancy, CI excludes zero).")

    # 5. Group consistency
    negative_groups = [g for g, s in group_comp.items() if s.get("expectancy_r", 0) < 0]
    if negative_groups:
        labels = {1: "Study", 2: "Out-of-sample", 3: "Live-era"}
        names = [labels.get(g, f"Group {g}") for g in negative_groups]
        issues.append(f"Negative expectancy in: {', '.join(names)}. System may be era-fitted or broken.")

    if len(issues) == 0:
        verdict = "GREEN"
    elif len(issues) <= 2:
        verdict = "YELLOW"
    else:
        verdict = "RED"

    return {
        "overall":    verdict,
        "issues":     issues,
        "live_cells": [f"{c['pair']} × {c['session']} ({c['n']} trades, {_fmt_r(c['expectancy_r'])})"
                       for c in live_cells],
    }


# ---------------------------------------------------------------------------
# PEAK-vs-FILL GUARD  (hardcoded lesson — read the docstring before deleting)
# ---------------------------------------------------------------------------
# THE MISTAKE THIS EXISTS TO PREVENT (made twice, both cost trust):
#   1. "47% of stop-outs reached +1R before stopping -> we leave money."
#   2. "93% of trades reversed from +0.5R."
# Both read a column that records where price *touched* (mfe_r / mae_r / any
# "price reached level L" count) and treated that touch as an *outcome we could
# have banked*. It is not. A touch is a peak on the path; banking it needs a
# resting ORDER at L that fills BEFORE the stop, under the sim's pessimistic
# same-bar rule (SL is checked first). When claim (1) was re-run with a REAL fixed
# exit at +1R (exit_lab C_fullTP_1.0R, 2024-07..2025-06), 23.5% of live-stopped
# trades booked +1R vs 41.4% that merely TOUCHED it — an 18pp gap. That gap is
# same-candle +1R-and-stop collisions a real order cannot bank. So the touch
# number is still a trap, but the capturable figure is ~23%, not the ~1-2% once
# claimed here.
#
# THE GENERAL LAW (applies far beyond MFE):
#   An EXTREMUM or a THRESHOLD-CROSSING is never an ACHIEVABLE OUTCOME until it
#   is replayed as a real order and reconciled to r_realised. This covers:
#     - mfe_r as "profit we could have taken"        (TP that fills before SL?)
#     - mae_r as "a wider stop would have survived"  (does the wider stop trade?)
#     - "price reached X% N times"                   (a limit at X — does it fill?)
#     - "reversed from level L"                       (entry at L — fills, then?)
#     - any "best/worst/max/min ... before exit" stat.
#   If a claim's number comes from a peak/touch column and NOT from an exit
#   replay (exit_engine.walk_multileg / exit_lab), it is UNVERIFIED. Say so.

# Column-name fragments that signal a peak/touch metric (not a bankable outcome).
_PEAK_METRIC_TOKENS = ("mfe", "mae", "reached", "touched", "excursion",
                       "peak", "max_r", "min_r", "reversed_from", "best_", "worst_")


def is_peak_metric(name: str) -> bool:
    """True if `name` looks like a peak/touch/extremum metric whose value is a
    point on the price PATH, not an achievable exit. Such metrics must never be
    quoted as bankable P&L without a real-exit replay (see verify_capturable)."""
    n = str(name).lower()
    return any(tok in n for tok in _PEAK_METRIC_TOKENS)


def verify_capturable(claim: str,
                      peak_count: int,
                      captured_count: int,
                      total: int,
                      *,
                      tolerance: float = 0.15) -> Dict[str, Any]:
    """HARD GATE for any claim of the form "X% of trades reached level L".

    You MUST supply BOTH:
      - peak_count     : how many trades TOUCHED L (from an mfe/mae/reached column).
      - captured_count : how many ACTUALLY EXIT with the L-outcome when replayed
                         as a real order (exit_engine.walk_multileg fixed-L / exit_lab
                         C_fullTP_*). If you have not run that replay, DO NOT call
                         this — the claim is unverified by definition.

    Returns {'verified': bool, 'severity', 'peak_pct', 'captured_pct',
             'gap_pct', 'message'}. `verified` is True only when the capturable
            fraction tracks the touch fraction within `tolerance`. A large gap
            means the touches are uncapturable intrabar spikes — the classic
            MFE trap — and the claim is REJECTED.

    Use it as a tripwire in any analysis script:
        r = verify_capturable("47% reached +1R", peak, captured, n)
        assert r['verified'], r['message']
    """
    total = max(int(total), 1)
    peak_pct = 100.0 * peak_count / total
    captured_pct = 100.0 * captured_count / total
    gap = peak_pct - captured_pct
    # A claim is only capturable if what actually fills tracks what merely touched.
    verified = (peak_count == 0) or (gap <= tolerance * peak_pct)
    if verified:
        sev, msg = "ok", (
            f"OK: '{claim}' — {captured_pct:.0f}% capturable vs {peak_pct:.0f}% "
            f"touched (gap {gap:.0f}pp within tolerance).")
    else:
        sev, msg = "reject", (
            f"REJECT: '{claim}' — {peak_pct:.0f}% merely TOUCHED the level but only "
            f"{captured_pct:.0f}% is CAPTURABLE with a real exit (gap {gap:.0f}pp). "
            f"This is the peak-vs-fill trap: a touch is not an outcome. Re-state the "
            f"claim using the capturable number, or drop it.")
    return {"verified": verified, "severity": sev,
            "peak_pct": round(peak_pct, 1), "captured_pct": round(captured_pct, 1),
            "gap_pct": round(gap, 1), "message": msg}
