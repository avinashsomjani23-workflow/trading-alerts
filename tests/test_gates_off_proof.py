"""Gates-off proof guard (EDGE_ENGINE_SPEC.md §4.3, change log §17).

Stage 0 check 3 proves the 18-yr baseline run was executed with the score gate
OFF. A gated run holds ZERO sub-floor trades, so the engine would be blind to
half the answer (never sees what the score filter would have dropped).

The test kills two bug classes:
  1. A GATED run (no sub-floor trades) must FAIL — the original purpose.
  2. The check must key on the PRESENCE of the sub-floor tail (absolute count),
     NOT its share. Scores are not proportional to performance, so a *fraction*
     threshold wrongly failed a gates-off run whose detector simply emits few
     sub-floor setups (the 2026-07-04 fix). A run with plenty of sub-floor
     trades that are a small % of the population must still PASS.
"""

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest.diagnostics.edge_engine import (
    _gates_off_proof, SCORE_FLOOR_LIVE, MIN_BELOW_FLOOR_N,
)


def _filled(scores):
    """A filled population with the given scores (fill_ts already applied upstream)."""
    return pd.DataFrame({"score": [float(s) for s in scores]})


def test_gated_run_fails():
    # Every trade at or above the floor => the run was score-censored => FAIL.
    df = _filled([SCORE_FLOOR_LIVE + 1] * 5000)
    res = _gates_off_proof(df)
    assert res["pass"] is False
    assert res["below_floor"] == 0
    assert "GATED" in res["note"]


def test_gates_off_run_passes_even_on_a_small_share():
    # The fix's core case: sub-floor trades are PRESENT (>= MIN_BELOW_FLOOR_N)
    # but a tiny fraction of the population. Old 10% fraction test failed this;
    # the presence test must PASS it.
    below = [SCORE_FLOOR_LIVE - 1.0] * MIN_BELOW_FLOOR_N          # exactly the floor N
    above = [SCORE_FLOOR_LIVE + 1.0] * (MIN_BELOW_FLOOR_N * 200)  # ~0.5% below-floor
    res = _gates_off_proof(_filled(below + above))
    assert res["pass"] is True
    assert res["below_floor"] == MIN_BELOW_FLOOR_N
    assert res["frac"] < 0.10          # would have FAILED the old fraction test
    assert res["note"] == "gates confirmed off"


def test_one_below_the_floor_still_fails():
    # A near-off config leaking a handful of sub-floor trades is not proof of
    # a gates-off run. Below the minimum count => FAIL.
    df = _filled([SCORE_FLOOR_LIVE - 1.0] * (MIN_BELOW_FLOOR_N - 1)
                 + [SCORE_FLOOR_LIVE + 1.0] * 1000)
    res = _gates_off_proof(df)
    assert res["pass"] is False


def test_floor_is_strict_less_than():
    # A trade exactly AT the floor is NOT below it (it would pass the live gate).
    df = _filled([float(SCORE_FLOOR_LIVE)] * (MIN_BELOW_FLOOR_N + 100))
    res = _gates_off_proof(df)
    assert res["below_floor"] == 0
    assert res["pass"] is False


def test_empty_or_missing_score_fails():
    assert _gates_off_proof(pd.DataFrame({"score": []}))["pass"] is False
    assert _gates_off_proof(pd.DataFrame({"other": [1, 2, 3]}))["pass"] is False


if __name__ == "__main__":
    test_gated_run_fails()
    test_gates_off_run_passes_even_on_a_small_share()
    test_one_below_the_floor_still_fails()
    test_floor_is_strict_less_than()
    test_empty_or_missing_score_fails()
    print("gates-off proof guard: OK")
