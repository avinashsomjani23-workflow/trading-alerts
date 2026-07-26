"""compute_atr tail-slice perf guard (BACKTEST_PERF_SPEC §3.1).

§3.1 tails the frame to period+1 bars inside compute_atr before the raw loop, so
a 9-yr slice does O(period) work instead of O(history). This test pins that the
optimisation is RESULT-IDENTICAL: compute_atr on a full frame must equal the raw
ATR formula (mean of the last `period` true ranges) computed inline over the
whole frame. If the tail ever drops a bar it should have kept, the float moves
and this fails loud.

Run:  python -m pytest tests/test_compute_atr_tail.py -q
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import smc_detector  # noqa: E402


def _raw_atr_full(df, period):
    """The old untailed formula, inline: mean of the last `period` TRs computed
    over the ENTIRE frame. This is the ground truth the tail must not change."""
    H = df["High"].values.astype(float)
    L = df["Low"].values.astype(float)
    C = df["Close"].values.astype(float)
    trs = []
    for i in range(1, len(C)):
        tr = max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))
        trs.append(tr)
    if len(trs) < period:
        return None
    return float(np.mean(trs[-period:]))


def test_compute_atr_tail_is_bit_identical_to_full_frame():
    rng = np.random.default_rng(14)
    n = 500
    idx = pd.date_range("2015-01-01", periods=n, freq="h")
    close = 1.10 + np.cumsum(rng.normal(0, 0.0006, n))
    high = close + rng.uniform(0, 0.001, n)
    low = close - rng.uniform(0, 0.001, n)
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close},
                      index=idx)
    smc_detector._ATR_CACHE.clear()
    for period in (14, 20, 50):
        got = smc_detector.compute_atr(df, period=period)
        expected = _raw_atr_full(df, period)
        assert got == expected, f"period={period}: {got!r} != {expected!r}"


def test_compute_atr_short_frame_returns_none():
    idx = pd.date_range("2015-01-01", periods=10, freq="h")
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0},
                      index=idx)
    # 10 bars < period+1 for the default period -> None (unchanged contract).
    assert smc_detector.compute_atr(df, period=14) is None
