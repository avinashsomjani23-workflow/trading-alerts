"""Guard tests for the shared MT5 candle-clock era table (backtest/mt5_clock.py).

Pure unit tests of the era logic — no events, no trades, no I/O. Moved here from
test_news_enrichment.py (Part A, 2026-07-16) so the one implementation has one
home for its tests. Behavior tests that exercise enrich() on era-B / flip-window
candles stay in test_news_enrichment.py.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mt5_clock import (  # noqa: E402
    is_flip_window,
    mt5_label_error_hours as err,
    true_utc,
)


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_clock_correction_eras():
    assert err(_ts("2010-07-15 12:00")) == 0      # era A, EU summer
    assert err(_ts("2010-01-15 12:00")) == -1     # era A, EU winter
    assert err(_ts("2016-07-15 12:00")) == 1      # era B, EU summer
    assert err(_ts("2016-01-15 12:00")) == 0      # era B, EU winter
    assert err(_ts("2025-07-15 12:00")) == 0      # era C, fixed UTC+3
    assert err(_ts("2024-11-15 12:00")) == 0      # era C starts 2024-10-27
    assert err(_ts("2014-11-20 12:00")) is None   # regime-flip ambiguity


def test_era_boundaries_are_inclusive_exclusive():
    # era C begins exactly 2024-10-27 (the EU DST-end weekend).
    assert err(_ts("2024-10-27 00:00")) == 0
    # last era-B instant (just before era C) is winter -> 0 either way; use a
    # summer era-B day to prove +1 still applies right up to era C's edge.
    assert err(_ts("2024-08-15 12:00")) == 1
    # flip window is [2014-11-01, 2014-12-08): era B begins 2014-12-08.
    assert err(_ts("2014-12-08 00:00")) is not None
    assert is_flip_window(_ts("2014-11-01 00:00")) is True
    assert is_flip_window(_ts("2014-12-07 23:00")) is True
    assert is_flip_window(_ts("2014-12-08 00:00")) is False
    assert is_flip_window(_ts("2014-10-31 23:00")) is False


def test_true_utc_subtracts_error_and_passes_through_nat():
    # era B summer label 12:00 -> true 11:00 (error +1)
    assert true_utc(_ts("2016-07-15 12:00")) == _ts("2016-07-15 11:00")
    # era A winter label 12:00 -> true 13:00 (error -1)
    assert true_utc(_ts("2010-01-15 12:00")) == _ts("2010-01-15 13:00")
    # era C -> unchanged
    assert true_utc(_ts("2025-07-15 12:00")) == _ts("2025-07-15 12:00")
    # flip window / NaT -> None
    assert true_utc(_ts("2014-11-20 12:00")) is None
    assert true_utc(pd.NaT) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
