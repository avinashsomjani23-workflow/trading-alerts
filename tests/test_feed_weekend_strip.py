"""Regression: the live feed strips synthetic FX-closed (weekend) bars so the
H1 frame matches MT5's gapped shape.

Run:  python tests/test_feed_weekend_strip.py
Exit 0 iff:
  1) _strip_market_closed drops every bar in the closed window
     (Saturday any hour, OR Sunday before 21:00 UTC) and keeps everything else;
  2) after stripping, a Fri->Mon boundary shows a real >1h gap (so the gap-aware
     detectors in dealing_range/h4_range see the session break they were built
     for) instead of a continuous run of filler bars;
  3) the strip is wired into _to_dataframe (the single network path), not just an
     unused helper.

WHY THIS EXISTS: Twelve Data's free tier pads the weekend with ~40 low-range
filler bars. Verified impossible in real data (18yr MT5 USDJPY H1: longest
sub-0.5-ATR run is ~10 bars, never 40). Those fillers (a) render a band of dead
micro-candles in every chart and (b) break resync_slate_zone_indices' uniform-
delta assumption, pushing FVG/BOS/OB boxes off their candles. If the strip ever
regresses, both bug classes return silently — this test is the tripwire.

Out-of-band by design: pure function test, never runs in the live alert path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import inspect

import pandas as pd

import feed_adapter


def _fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def _make_frame():
    """Continuous hourly UTC frame spanning a full Fri->Mon weekend, as Twelve
    Data would deliver it WITH weekend filler. Covers Fri 20:00 UTC through
    Mon 02:00 UTC so every edge of the closed window is exercised."""
    # 2026-07-03 is a Friday.
    idx = pd.date_range("2026-07-03 20:00", "2026-07-06 02:00", freq="h", tz="UTC")
    n = len(idx)
    df = pd.DataFrame(
        {
            "Open": [1.0] * n, "High": [1.0] * n,
            "Low": [1.0] * n, "Close": [1.0] * n, "Volume": [0.0] * n,
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


def main():
    df = _make_frame()
    out = feed_adapter._strip_market_closed(df)

    dow = out.index.dayofweek
    hour = out.index.hour

    # 1) nothing in the closed window survives.
    bad = (dow == 5) | ((dow == 6) & (hour < 21))
    if bad.any():
        _fail(f"{int(bad.sum())} closed-window bar(s) survived the strip: "
              f"{list(out.index[bad])}")

    # ...and legit bars are kept: Fri<=23:00, Sun>=21:00, all of Mon.
    kept_fri = ((df.index.dayofweek == 4)).sum()
    if (out.index.dayofweek == 4).sum() != kept_fri:
        _fail("a legit Friday bar was dropped")
    if not (out.index.dayofweek == 0).any():
        _fail("Monday bars were dropped")
    if not ((out.index.dayofweek == 6) & (out.index.hour >= 21)).any():
        _fail("legit Sunday >=21:00 UTC reopen bar was dropped")

    # 2) the strip creates a real session gap the detectors can see.
    deltas = out.index.to_series().diff().dropna()
    max_gap_h = deltas.max().total_seconds() / 3600.0
    if max_gap_h < 2.0:
        _fail(f"no session gap after strip (max delta {max_gap_h:.1f}h) — "
              f"gap-aware structure detection would not see the weekend")

    # 3) the strip is actually wired into the network path.
    src = inspect.getsource(feed_adapter._to_dataframe)
    if "_strip_market_closed" not in src:
        _fail("_to_dataframe does not call _strip_market_closed — strip is dead code")

    print(f"PASS: stripped {len(df) - len(out)} weekend bars, "
          f"session gap {max_gap_h:.0f}h restored, wired into _to_dataframe.")
    sys.exit(0)


if __name__ == "__main__":
    main()
