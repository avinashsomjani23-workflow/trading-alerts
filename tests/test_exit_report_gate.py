"""EXIT-LEVER HONESTY GATE regression test (trader-approved 2026-07-09).

Kills three silent report-layer bug classes that all occurred in the Step-B
exit session (STEP_B_EXIT_TRACK_HANDOFF.md §7):
  (a) an exit result emitted without avg R:R next to net expR — hides an expR
      "win" bought by collapsing R:R (2.08 -> 0.80);
  (b) a rule conditioning on an outcome-time column (sl_*, exit_*, r_*)
      without a non-tradeable flag — outcome-conditioning sold as tradeable;
  (c) r == 0 breakeven scratches counted in the loser denominator — poisons
      every stopout/sweep percentage.

Out-of-band: the gate lives in backtest/diagnostics, never in the live alert
path.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest.diagnostics.exit_report_gate import (
    assert_exit_report, check_exit_result, is_outcome_time,
)


def _clean(**over):
    res = {
        "recipe": "widen0.5_be1.0",
        "net_expR": -0.08,
        "avg_RR": 1.62,
        "conditions_on": [],
        "loser_def": "r<0",
    }
    res.update(over)
    return res


def test_clean_result_passes():
    assert check_exit_result(_clean()) == []
    assert_exit_report([_clean(), _clean(recipe="be_None")])


def test_missing_rr_fails():          # rule (a)
    bad = _clean()
    del bad["avg_RR"]
    v = check_exit_result(bad)
    assert len(v) == 1 and "avg_RR" in v[0]


def test_outcome_conditioning_needs_flag():   # rule (b)
    bad = _clean(conditions_on=["sl_bar_was_sweep"])
    v = check_exit_result(bad)
    assert len(v) == 1 and "sl_bar_was_sweep" in v[0]
    # flagged non-tradeable -> allowed (visible, honest)
    ok = _clean(conditions_on=["sl_bar_was_sweep"], non_tradeable=True)
    assert check_exit_result(ok) == []


def test_missing_conditions_on_fails():       # rule (b) — unverifiable
    bad = _clean()
    del bad["conditions_on"]
    v = check_exit_result(bad)
    assert len(v) == 1 and "conditions_on" in v[0]


def test_scratches_in_loser_denominator_fail():   # rule (c)
    bad = _clean(loser_def="exit_reason=='sl'")
    v = check_exit_result(bad)
    assert len(v) == 1 and "loser_def" in v[0]


def test_batch_raises_with_every_violation():
    batch = [_clean(),
             _clean(recipe="sweep_widen", conditions_on=["sl_bar_was_sweep"]),
             _clean(recipe="old_style", loser_def="exit_reason=='sl'")]
    with pytest.raises(ValueError) as e:
        assert_exit_report(batch)
    msg = str(e.value)
    assert "sweep_widen" in msg and "old_style" in msg


def test_timing_classifier_edges():
    # outcome-time families
    assert is_outcome_time("sl_bar_was_sweep")
    assert is_outcome_time("exit_reason")
    assert is_outcome_time("r_realised")
    assert is_outcome_time("mfe_r")
    # entry-legal exceptions with a matching prefix
    assert not is_outcome_time("sl_distance_atr")
    assert not is_outcome_time("r_distance")
    # plain entry features
    assert not is_outcome_time("ob_range_atr")
