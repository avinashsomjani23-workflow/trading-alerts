"""Regression: FVG geometry is CLOSED-BARS-ONLY. A still-forming current-hour
bar must never be used as c1/c2/c3 or as a mitigation candle.

Run:  python tests/test_fvg_forming_bar.py
Exit 0 iff:
  (1) drop_forming_bar removes the last row when its open-hour == current UTC
      hour, and keeps it otherwise; and
  (2) detect_fvg_in_zone does NOT report an FVG whose 3rd candle only exists on
      the forming (open, still-moving) bar — i.e. running on the closed frame
      (what smc_radar now passes) yields no FVG, while running on the frame WITH
      the forming bar spuriously would (proving the bar was the culprit).

Why this bug matters: Twelve Data returns the current hour's bar (open, still
moving) as the newest row. Its mid-flight wick can fabricate a 3-candle
imbalance that vanishes the instant the bar closes and the wick fills in. A
phantom FVG inflates the confidence score and can push a non-signal into a live
alert. smc_radar.detect_smc_radar now feeds detect_fvg_in_zone the output of
drop_forming_bar(df); this test is the tripwire if the forming bar ever re-enters
FVG geometry.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import smc_detector
import smc_radar

_FAILS = []


def _ok(m): print(f"  OK:   {m}")
def _bad(m):
    print(f"  FAIL: {m}")
    _FAILS.append(m)


# ---------------------------------------------------------------------------
# Part 1: drop_forming_bar drops the still-forming bar, keeps closed bars.
# ---------------------------------------------------------------------------
def _hourly_frame(open_hours):
    """Build a reset-index frame with a UTC 'Datetime' column at the given
    hour-boundary timestamps (list of pandas Timestamps)."""
    rows = []
    for i, ts in enumerate(open_hours):
        base = 1.0 + i * 0.01
        rows.append({"Datetime": ts, "Open": base, "High": base + 0.005,
                     "Low": base - 0.005, "Close": base + 0.002})
    return pd.DataFrame(rows)


def test_drop_forming_bar():
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)  # noqa: matches drop_forming_bar's own utcnow()
    # Last bar OPENS this hour -> still forming -> must be dropped.
    hours = [pd.Timestamp(now - timedelta(hours=h), tz="UTC")
             for h in range(4, -1, -1)]  # ..., now-1h, now(open, forming)
    df = _hourly_frame(hours)
    out = smc_radar.drop_forming_bar(df)
    if len(out) == len(df) - 1:
        _ok("drop_forming_bar removed the forming current-hour bar")
    else:
        _bad(f"forming bar NOT dropped: in={len(df)} out={len(out)}")

    # Last bar OPENED an hour ago -> already closed -> must be kept.
    hours_closed = [pd.Timestamp(now - timedelta(hours=h), tz="UTC")
                    for h in range(5, 0, -1)]  # newest = now-1h (closed)
    df_closed = _hourly_frame(hours_closed)
    out_closed = smc_radar.drop_forming_bar(df_closed)
    if len(out_closed) == len(df_closed):
        _ok("drop_forming_bar kept an already-closed last bar")
    else:
        _bad(f"closed bar wrongly dropped: in={len(df_closed)} out={len(out_closed)}")


# ---------------------------------------------------------------------------
# Part 2: the forming bar is what fabricates the FVG. On the closed frame the
# FVG does not exist; append a forming c3 and it appears -> confirms the bar was
# the culprit and that closed-only detection kills it.
# ---------------------------------------------------------------------------
def _build_fvg_case():
    """LONG-bias frame. Closed bars have NO gap between bar0.High and bar2.Low.
    An appended forming bar (index 4) creates a valid bullish FVG only via its
    mid-flight Low. bias='LONG' FVG condition: H[k] < L[k+2].

    We anchor the scan at k=2 so c3 = index 4 (the appended bar). On the closed
    frame index 4 does not exist -> no FVG. With it -> FVG."""
    # 5 closed bars whose ranges all OVERLAP heavily -> NO bullish gap anywhere
    # (for every k, H[k] >= L[k+2], so the LONG condition H[k] < L[k+2] fails).
    closed = pd.DataFrame({
        "Open":  [1.00, 1.00, 1.00, 1.00, 1.00],
        "High":  [1.05, 1.05, 1.05, 1.05, 1.05],
        "Low":   [0.95, 0.95, 0.95, 0.95, 0.95],
        "Close": [1.00, 1.00, 1.00, 1.00, 1.00],
    })
    # Forming bar (index 5) gaps far above bar 3's High -> bullish FVG at k=3
    # (H[3]=1.05 < L[5]=1.20). Only exists if this forming bar is included.
    forming = pd.DataFrame({
        "Open":  [1.20], "High": [1.25], "Low": [1.20], "Close": [1.22],
    })
    with_forming = pd.concat([closed, forming], ignore_index=True)
    return closed, with_forming


def test_forming_bar_fabricates_fvg():
    closed, with_forming = _build_fvg_case()
    atr_floor = 0.001  # small; the gap (~0.165) dwarfs it
    # Scan window covers k=3 (c3 = index 5, the forming bar in with_forming).
    kwargs = dict(bias="LONG", zone_top=1.05, zone_bottom=0.99,
                  atr_floor=atr_floor, leg_start_idx=0,
                  pair_type="forex")

    res_forming = smc_detector.detect_fvg_in_zone(
        with_forming, leg_end_idx=len(with_forming) - 1, **kwargs)
    if res_forming.get("exists"):
        _ok("forming-bar frame DOES fabricate an FVG (bug reproduced)")
    else:
        _bad("could not reproduce phantom FVG — test setup is wrong, fix the case")

    res_closed = smc_detector.detect_fvg_in_zone(
        closed, leg_end_idx=len(closed) - 1, **kwargs)
    if not res_closed.get("exists"):
        _ok("closed-only frame reports NO FVG (forming bar excluded = fixed)")
    else:
        _bad(f"closed frame still reports an FVG: {res_closed}")


# ---------------------------------------------------------------------------
# Part 3: source guard — detect_smc_radar must feed the FVG detector the
# drop_forming_bar output, not the raw df. Trips if the wiring is reverted.
# ---------------------------------------------------------------------------
def test_source_wires_closed_df():
    src = (Path(_ROOT) / "smc_radar.py").read_text(encoding="utf-8")
    if "df_fvg = drop_forming_bar(df)" not in src:
        _bad("df_fvg = drop_forming_bar(df) missing from detect_smc_radar")
        return
    # The detect_fvg_in_zone call must pass df_fvg, never the raw df.
    import re
    m = re.search(r"detect_fvg_in_zone\(\s*([A-Za-z_]+)", src)
    if not m:
        _bad("could not locate detect_fvg_in_zone call in smc_radar.py")
    elif m.group(1) == "df_fvg":
        _ok("detect_fvg_in_zone is called with the closed-only df_fvg")
    else:
        _bad(f"detect_fvg_in_zone called with '{m.group(1)}', expected 'df_fvg'")


if __name__ == "__main__":
    print("test_fvg_forming_bar:")
    test_drop_forming_bar()
    test_forming_bar_fabricates_fvg()
    test_source_wires_closed_df()
    if _FAILS:
        print(f"\n{len(_FAILS)} FAILURE(S)")
        sys.exit(1)
    print("\nALL PASS")
    sys.exit(0)
