"""Part B-guard (2026-07-16): prove the import clock fix BITES on real candles.

Rebuilds EURUSD H1 through import_mt5.load_one (the corrected path) and spike-
aligns high-impact ForexFactory events (epoch = exact UTC) against it: the fattest
H1 bar in a +/-3h window around each event's TRUE-UTC hour must land at offset 0,
per era-season cell. This is the bite-proof from MT5_CANDLE_CLOCK_AUDIT.md,
re-pointed at the fix. It FAILS if the era table is ever reverted (uncorrected
candles peak at -1 in era-A winter, +1 in era-B summer).

Skips (not fails) if the raw MT5 CSV or the events file is absent, so CI without
the data blob stays green — the era-table math itself is pinned in test_mt5_clock.py.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mt5_data"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSV = os.path.join(_HERE, "mt5_data", "EURUSD_H1.csv")
_EVENTS = os.path.join(_HERE, "data", "ff_calendar_2007_2026.csv")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_CSV) and os.path.exists(_EVENTS)),
    reason="raw MT5 CSV or events file not present",
)

_EU = "Europe/Athens"


def _load_candles():
    import import_mt5
    return import_mt5.load_one(__import__("pathlib").Path(_CSV), interval="1h")


def _high_usd_eur_events():
    from news_enrichment import load_events
    ev = load_events(_EVENTS)
    ev = ev[ev["high"] & ev["timed"] & ~ev["speech"]]
    return ev[ev["currency"].isin(("USD", "EUR"))].copy()


def _modal_offset(candles: pd.DataFrame, events: pd.DataFrame) -> int:
    """For each event, the hour offset (event_hour -> fattest nearby bar), mode.

    Bar 'fatness' = High-Low range; a high-impact release fattens the bar of its
    own hour. If the candle labels are TRUE UTC, the fattest bar sits at the
    event's own hour -> offset 0.
    """
    rng = (candles["High"] - candles["Low"]).astype(float)
    offsets = []
    for t in events["utc"]:
        hour = t.floor("h")
        window = rng.loc[hour - pd.Timedelta(hours=3): hour + pd.Timedelta(hours=3)]
        if len(window) < 5:
            continue
        peak = window.idxmax()
        offsets.append(int((peak - hour) / pd.Timedelta(hours=1)))
    if not offsets:
        return None
    vals, counts = np.unique(offsets, return_counts=True)
    return int(vals[counts.argmax()])


# (label year-month range, EU season) cells that were WRONG before the fix.
# era A winter -> was -1, era B summer -> was +1; both must now peak at 0.
_CELLS = [
    ("era-A winter", "2010-01-01", "2010-02-28"),
    ("era-A summer", "2010-06-01", "2010-08-31"),
    ("era-B summer", "2016-06-01", "2016-08-31"),
    ("era-B winter", "2016-01-01", "2016-02-28"),
    ("era-C", "2025-06-01", "2025-08-31"),
]


def test_spike_alignment_peaks_at_zero_per_era():
    candles = _load_candles()
    events = _high_usd_eur_events()
    failures = []
    for label, lo, hi in _CELLS:
        cell_ev = events[(events["utc"] >= pd.Timestamp(lo, tz="UTC")) &
                         (events["utc"] <= pd.Timestamp(hi, tz="UTC"))]
        off = _modal_offset(candles, cell_ev)
        if off is None:
            continue
        if off != 0:
            failures.append(f"{label}: modal offset {off:+d} (expected 0)")
    assert not failures, "clock fix did not bite:\n  " + "\n  ".join(failures)


def test_flip_window_rows_kept_provisional_and_flaggable():
    """Flip-window candles keep the provisional label -> still in the flip range."""
    from mt5_clock import is_flip_window
    candles = _load_candles()
    flip = candles.loc["2014-11-01":"2014-12-07"]
    assert len(flip) > 0
    # every retained flip-window label is still detectable as flip (re-derivable).
    assert all(is_flip_window(ts) for ts in flip.index)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
