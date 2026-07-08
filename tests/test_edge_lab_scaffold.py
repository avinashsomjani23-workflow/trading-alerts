"""Edge Lab v2 scaffold guards (SMC_EDGE_LAB_SPEC.md §11 step 1, §12).

These tests protect the three things in the scaffold that can fail SILENTLY and
corrupt every downstream conclusion:

  1. SCHEMA / DTYPE GUARD (§12). A hand-pasted or misaligned column shifts the CSV
     and discovery ships a fake edge. The guard must (a) REFUSE the old/stale CSV
     that lacks the six new columns, and (b) RAISE when a numeric feature holds an
     unparseable string. Both are out-of-band (analysis loader only) — they can
     never kill a live alert, only stop a bad research run.

  2. THREE-CLASS TIMING CLASSIFIER (§12). outcome_time features must NEVER be
     entry-legal (look-ahead). The classifier is the wall; a misclassification is
     a silent leak that would let an exit feature pick entries.

  3. THE LOOK-AHEAD WALL. entry_features() and exit_features() must never overlap,
     and no outcome_time feature may appear in the entry universe.

All synthetic — no CSV on disk required, runnable standalone.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest.diagnostics import edge_lab as lab


# ── helpers ─────────────────────────────────────────────────────────────────

def _clean_frame(n=10):
    """A minimal frame that carries all baseline-signal columns and clean numerics
    so the guard PASSES. Only the columns the guard inspects are needed.

    dtype=object mirrors how pd.read_csv presents a CSV where a text value has bled
    into a numeric column (the whole column becomes object) — the exact corruption
    the guard exists to catch. It also lets a test assign a string into a 'numeric'
    column, which newer pandas forbids on a typed float64 column."""
    row = {c: 0.0 for c in lab._BASELINE_SIGNAL_COLUMNS}
    row.update({c: 1.0 for c in lab._NUMERIC_MUST_PARSE})
    return pd.DataFrame([row] * n, dtype=object)


# ── 1. schema guard: stale CSV ──────────────────────────────────────────────

def test_guard_refuses_stale_csv_missing_new_columns():
    df = _clean_frame()
    df = df.drop(columns=["sl_wick_depth_atr"])  # drop one baseline-signal column
    with pytest.raises(lab.SchemaGuardError) as e:
        lab._run_schema_guard(df, "stale_run")
    assert "missing baseline columns" in str(e.value)
    assert "sl_wick_depth_atr" in str(e.value)


def test_guard_passes_clean_baseline_frame():
    ev = lab._run_schema_guard(_clean_frame(), "clean_run")
    assert ev["pass"] is True
    assert ev["missing_baseline_columns"] == []
    assert ev["unparseable_numeric_columns"] == {}


# ── 1b. schema guard: corrupt numeric (the paste-shift bug class) ────────────

def test_guard_raises_on_unparseable_numeric():
    df = _clean_frame()
    # Simulate the killzone_windows comma bleed: a text value in a numeric column.
    df.loc[0, "ob_range_atr"] = "London,NewYork"
    with pytest.raises(lab.SchemaGuardError) as e:
        lab._run_schema_guard(df, "corrupt_run")
    assert "unparseable" in str(e.value).lower()
    assert "ob_range_atr" in str(e.value)


def test_guard_tolerates_na_tokens_in_numeric():
    # None/blank/nan are legitimate missing values (None-by-construction columns),
    # NOT corruption. They must not trip the guard.
    df = _clean_frame()
    df.loc[0, "fvg_size_atr"] = None
    df.loc[1, "pd_pct"] = ""
    df.loc[2, "reversal_pct"] = "nan"
    ev = lab._run_schema_guard(df, "na_ok_run")
    assert ev["pass"] is True


# ── 2. three-class timing classifier ────────────────────────────────────────

def test_outcome_time_features_are_classed_outcome_time():
    for f in ["r_capture_ratio", "sl_wick_depth_atr", "mfe_r", "mae_r",
              "sl_max_adverse_after_sweep_atr", "bars_sl_to_tp1_touch",
              "sl_recovered_to_entry", "r_realised", "exit_reason"]:
        assert lab.classify_timing(f) == "outcome_time", f

def test_fill_time_features_are_classed_fill_time():
    for f in ["sl_distance_atr", "ob_to_fill_hours", "fill_in_killzone",
              "killzone_alignment", "fill_session"]:
        assert lab.classify_timing(f) == "fill_time", f

def test_alert_time_is_the_default():
    for f in ["break_close_atr", "ob_range_atr", "bos_verdict", "trend_pd_agree",
              "pd_zone", "session"]:
        assert lab.classify_timing(f) == "alert_time", f


# ── 3. the look-ahead wall ──────────────────────────────────────────────────

def _wall_frame():
    """A frame carrying entry-legal + outcome-time + decreed-out columns, so the
    split functions have something to select from."""
    cols = (lab._V1_ENTRY_MANIFEST + lab._NEW_ENTRY_LEGAL
            + sorted(lab.OUTCOME_TIME_FEATURES) + list(lab.DECREED_OUT))
    return pd.DataFrame([{c: 0.0 for c in cols}])

def test_no_outcome_time_feature_is_entry_legal():
    df = _wall_frame()
    ef = lab.entry_features(df)
    leak = [f for f in ef if lab.classify_timing(f) == "outcome_time"]
    assert leak == [], f"outcome-time features leaked into entry universe: {leak}"

def test_entry_and_exit_universes_never_overlap():
    df = _wall_frame()
    overlap = set(lab.entry_features(df)) & set(lab.exit_features(df))
    assert overlap == set(), f"entry/exit overlap: {sorted(overlap)}"

def test_sweep_present_is_decreed_out_of_entries():
    df = _wall_frame()
    assert "sweep_present" not in lab.entry_features(df)

def test_pairs_are_the_ten_in_scope():
    assert len(lab.PAIRS) == 10
    assert "NAS100" not in lab.PAIRS
    assert "BTCUSD" in lab.PAIRS


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
