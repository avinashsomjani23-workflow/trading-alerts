"""Aggregate population guard (TRUTH_FIXES_SPEC.md T2).

The cross-run aggregator must never feed force-closed audit rows
(timeout / window_end — arbitrary-price exits, audit-only by the settled
unresolved-trade policy) or hard-blocked rows into any metric. trades.csv
ships eligible_for_headline for exactly this; legacy CSVs are reconstructed
from the same inputs. This test kills the class: a population that
re-derives eligibility wrongly fails here before it can publish a number.
"""

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest.aggregate_runs import _eligible_mask
from backtest import insights as ins


def _df(rows):
    return pd.DataFrame(rows)


def test_force_closed_rows_are_excluded():
    df = _df([
        {"exit_reason": "tp1",        "r_realised":  1.0, "eligible_for_headline": True},
        {"exit_reason": "timeout",    "r_realised": -0.4, "eligible_for_headline": False},
        {"exit_reason": "window_end", "r_realised":  0.7, "eligible_for_headline": False},
        {"exit_reason": "never_filled", "r_realised": 0.0, "eligible_for_headline": False},
    ])
    eligible = df[_eligible_mask(df)]
    filled = ins._filled(eligible)
    assert len(filled) == 1
    assert filled.iloc[0]["exit_reason"] == "tp1"
    # The timeout row's R must not be in the expectancy input.
    assert -0.4 not in filled["r_realised"].tolist()
    assert 0.7 not in filled["r_realised"].tolist()


def test_string_bools_from_csv_roundtrip():
    # pandas reads mixed bool columns back as strings — masks must survive it.
    df = _df([
        {"exit_reason": "tp1", "r_realised": 1.0, "eligible_for_headline": "True"},
        {"exit_reason": "sl",  "r_realised": -1.0, "eligible_for_headline": "False"},
    ])
    eligible = df[_eligible_mask(df)]
    assert len(eligible) == 1
    assert eligible.iloc[0]["exit_reason"] == "tp1"


def test_legacy_csv_without_column_reconstructs_from_rule_inputs():
    df = _df([
        {"exit_reason": "tp1",     "r_realised": 1.0, "ist_blocked": "False", "weekend_blocked": False},
        {"exit_reason": "tp1",     "r_realised": 1.0, "ist_blocked": "True",  "weekend_blocked": False},
        {"exit_reason": "timeout", "r_realised": 0.3, "ist_blocked": "False", "weekend_blocked": False},
        {"exit_reason": "sl",      "r_realised": -1.0, "ist_blocked": "False", "weekend_blocked": "True"},
    ])
    eligible = df[_eligible_mask(df)]
    assert len(eligible) == 1
    assert eligible.iloc[0]["exit_reason"] == "tp1"


if __name__ == "__main__":
    test_force_closed_rows_are_excluded()
    test_string_bools_from_csv_roundtrip()
    test_legacy_csv_without_column_reconstructs_from_rule_inputs()
    print("aggregate eligibility guard: OK")
