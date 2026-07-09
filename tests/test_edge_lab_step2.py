"""
Verification tests for edge_lab_step2 (v2 Step 2 — univariate pooled discovery).

These are OUT-OF-BAND guards (project rule): they run on tiny, hand-computed synthetic
frames, never on the live trade path, and fail loudly if the scorecard math drifts.
Each test encodes an answer I can verify by hand, so a silent regression in the
type-classification, effect calc, or thin-bucket guard is caught here — not discovered
by eyeballing a 44-row table.

Covers, specifically, the three bug classes fixed on 2026-07-08:
  1. bos_tier wrongly treated as ORDINAL (its levels are unordered structure types).
  2. a fake effect pulled from an N=1 extreme bucket (thin-bucket guard).
  3. nominal best/worst effect pulled from a thin bucket (same guard, nominal path).
Plus the core: continuous Spearman sign, binary diff, and disposition triage.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.diagnostics import edge_lab_step2 as s2


# ── helpers ─────────────────────────────────────────────────────────────────
def _frame(feat_name, feat_vals, r_vals, ts=None):
    """Minimal frame the scorecard needs: the feature, r_realised, alert_ts."""
    n = len(r_vals)
    if ts is None:
        ts = pd.date_range("2010-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        feat_name: feat_vals,
        s2.R_COL: r_vals,
        s2.TS_COL: ts,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 1. TYPE CLASSIFICATION (the bos_tier bug).
# ═══════════════════════════════════════════════════════════════════════════

def test_bos_tier_is_nominal_not_ordinal():
    """bos_tier's levels are BOS/CHoCH/Confirm/Range — unordered. It must classify
    as nominal (Kruskal), NOT ordinal (Spearman). This is the exact 2026-07-08 bug."""
    s = pd.Series(["BOS", "CHoCH", "Confirm", "Range"] * 50)
    assert s2._feature_kind("bos_tier", s) == "nominal"
    assert "bos_tier" not in s2.ORDINAL_FEATURES


def test_true_ordinal_stays_ordinal():
    """A genuinely ordered count (bos_sequence_count) with >2 levels stays ordinal."""
    s = pd.Series([0, 1, 2, 3, 4, 5] * 40)
    assert s2._feature_kind("bos_sequence_count", s) == "ordinal"


def test_two_value_continuous_becomes_binary():
    """A 'continuous' column that only ever takes 2 values is really binary (§3)."""
    s = pd.Series([0.0, 1.0] * 100)   # reversal_pct behaves like this in the data
    assert s2._feature_kind("reversal_pct", s) == "binary"


def test_nominal_default():
    s = pd.Series(["asia", "london", "ny"] * 60)
    assert s2._feature_kind("session", s) == "nominal"


# ═══════════════════════════════════════════════════════════════════════════
# 2. THIN-BUCKET GUARD on the ORDINAL effect (the bos_sequence_count −0.92R bug).
# ═══════════════════════════════════════════════════════════════════════════

def test_ordinal_effect_ignores_thin_extreme_bucket():
    """An ordinal feature whose EXTREME level has 1 trade must NOT let that bucket
    define the effect. Before the guard, level 10 (n=1, r=+4.8) produced a bogus
    effect. After: effect is computed only over levels with n>=MIN_BUCKET_N, so the
    huge thin-bucket value cannot appear."""
    big = s2.MIN_BUCKET_N + 20
    # levels 0 and 1 are fat and near zero; a lone level 9 sits at +9.0R (n=1).
    feat = [0] * big + [1] * big + [9] * 1
    r = [0.01] * big + [-0.02] * big + [9.0]
    df = _frame("bos_sequence_count", feat, r)
    row = s2.score_feature(df, "bos_sequence_count", pooled_mean=float(np.mean(r)))
    assert row["kind"] == "ordinal"
    # effect must come from levels 0..1 only (both fat); never touch the +9.0 outlier.
    assert row.get("effect_levels") == "0..1"
    assert abs(row["effect_r"]) < 0.1, row  # ~ -0.03, NOT +9
    # sanity: the +9 bucket never leaked into the effect magnitude
    assert row["effect_r"] > -0.1


def test_ordinal_no_effect_when_fewer_than_two_fat_levels():
    """If only ONE level clears MIN_BUCKET_N, there is no defensible effect -> None
    (the Spearman rho/p still carries the trend)."""
    big = s2.MIN_BUCKET_N + 20
    feat = [0] * big + [1] * 3 + [2] * 3   # only level 0 is fat
    r = [0.05] * big + [0.5, -0.5, 0.1] + [0.2, -0.2, 0.0]
    df = _frame("bos_sequence_count", feat, r)
    row = s2.score_feature(df, "bos_sequence_count", pooled_mean=float(np.mean(r)))
    assert row.get("effect_r") is None
    assert "spearman_rho" in row   # trend stat still present


# ═══════════════════════════════════════════════════════════════════════════
# 3. THIN-BUCKET GUARD on the NOMINAL effect (the setup_badge bug).
# ═══════════════════════════════════════════════════════════════════════════

def test_nominal_effect_ignores_thin_level():
    """A nominal feature's best/worst effect must be taken over fat levels only. A
    thin level with a wild mean must not define the gap."""
    big = s2.MIN_BUCKET_N + 20
    feat = (["a"] * big) + (["b"] * big) + (["c"] * 2)  # c is thin, wild
    r = ([0.10] * big) + ([-0.10] * big) + [5.0, 5.0]
    df = _frame("event", feat, r)
    row = s2.score_feature(df, "event", pooled_mean=float(np.mean(r)))
    assert row["kind"] == "nominal"
    # effect spans a..b (both fat): 0.10 - (-0.10) = 0.20; c(+5) excluded.
    assert row.get("effect_levels") in ("b..a", "a..b")
    assert abs(row["effect_r"] - 0.20) < 1e-6, row


# ═══════════════════════════════════════════════════════════════════════════
# 4. CORE MATH — continuous, binary, disposition.
# ═══════════════════════════════════════════════════════════════════════════

def test_continuous_spearman_sign_positive():
    """A monotonically increasing r-vs-x relationship -> positive Spearman rho."""
    x = list(range(400))
    r = [0.001 * i for i in range(400)]   # r rises with x
    df = _frame("impulse_leg_atr", x, r)
    row = s2.score_feature(df, "impulse_leg_atr", pooled_mean=float(np.mean(r)))
    assert row["kind"] == "continuous"
    assert row["spearman_rho"] > 0.9
    assert row["p_value"] < 0.01
    # effect = top-decile mean minus bottom-decile mean, must be positive here
    assert row["effect_r"] > 0


def test_binary_diff_effect_direction():
    """Binary effect = mean(level_hi) - mean(level_lo), with hi/lo sorted by str.
    Group '1' at +0.2R, group '0' at -0.2R -> effect = 0.4."""
    feat = ([0] * 200) + ([1] * 200)
    r = ([-0.2] * 200) + ([0.2] * 200)
    df = _frame("fvg_present", feat, r)
    row = s2.score_feature(df, "fvg_present", pooled_mean=0.0)
    assert row["kind"] == "binary"
    assert abs(row["effect_r"] - 0.4) < 1e-9
    # a clean ±0.2 split with n=200 each has a tight CI that excludes 0
    assert row["effect_ci_excludes_0"] is True


def test_disposition_queue_requires_all_three():
    """ship_gate_queue needs: p<0.05 (where p exists) AND CI-excludes-0 AND
    |effect|>=floor. A big effect with a CI touching 0 stays 'not proven'."""
    # over floor + CI excludes 0 + significant -> QUEUE
    q = {"effect_r": 0.2, "effect_ci_excludes_0": True, "p_value": 0.001}
    assert s2._disposition(q) == "ship_gate_queue"
    # over floor but CI includes 0 -> not proven
    n1 = {"effect_r": 0.2, "effect_ci_excludes_0": False, "p_value": 0.001}
    assert s2._disposition(n1) == "interesting_not_proven"
    # under floor -> not proven even if significant
    n2 = {"effect_r": 0.01, "effect_ci_excludes_0": True, "p_value": 0.001}
    assert s2._disposition(n2) == "interesting_not_proven"
    # binary (p_value None): CI-excludes-0 + floor is enough
    b = {"effect_r": 0.2, "effect_ci_excludes_0": True, "p_value": None}
    assert s2._disposition(b) == "ship_gate_queue"


def test_effect_floor_boundary():
    """Exactly-at-floor counts as over the floor (>=)."""
    at = {"effect_r": s2.EFFECT_FLOOR, "effect_ci_excludes_0": True, "p_value": 0.01}
    assert s2._disposition(at) == "ship_gate_queue"


# ═══════════════════════════════════════════════════════════════════════════
# 5. BAYESIAN SHRINKAGE — a thin bucket is pulled toward pooled.
# ═══════════════════════════════════════════════════════════════════════════

def test_bayes_shrink_pulls_thin_bucket():
    """A thin bucket's shrunk mean sits between its raw mean and the pooled mean,
    and closer to pooled than a fat bucket with the same raw mean would be."""
    curve = [
        {"level": "fat", "n": 1000, "expR": 1.0},
        {"level": "thin", "n": 2, "expR": 1.0},
    ]
    pooled = 0.0
    s2._bayes_shrink(curve, pooled)
    fat = next(c for c in curve if c["level"] == "fat")
    thin = next(c for c in curve if c["level"] == "thin")
    assert thin["expR_shrunk"] < fat["expR_shrunk"]   # thin pulled harder to pooled
    assert 0.0 < thin["expR_shrunk"] < 1.0            # between pooled and raw


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
