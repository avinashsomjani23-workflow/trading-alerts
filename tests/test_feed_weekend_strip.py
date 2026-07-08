"""Regression: the live feed strips synthetic FX-closed (weekend) bars so the
H1 frame matches MT5's gapped shape.

Run:  python tests/test_feed_weekend_strip.py
Exit 0 iff:
  1) _strip_market_closed drops EVERY Saturday bar and EVERY Sunday bar (the
     broker week is Mon 00:00 -> Fri 23:00 UTC) and keeps all of Fri/Mon;
  2) after stripping, the Fri 23:00 -> Mon 00:00 boundary shows the real ~49h
     weekend gap the gap-aware detectors in dealing_range/h4_range were built
     for — not a partial Sunday-evening reopen;
  3) the strip is wired into _to_dataframe (the single network path), not just an
     unused helper.

WHY THIS EXISTS: Twelve Data pads the FX-closed weekend with filler bars that MT5
never had. Verified against MT5 directly (1 full year, all 5 live pairs): ZERO
Saturday bars and ZERO Sunday bars — the broker prints Mon 00:00 -> Fri 23:00 UTC
only. Those fillers (a) render dead micro-candles AND, because both renderers plot
on positional x, shove every post-weekend candle sideways vs MT5; and (b) break
resync_slate_zone_indices' uniform-delta assumption, pushing FVG/BOS/OB boxes off
their candles. The FIRST fix kept Sunday >= 21:00 UTC on a generic-FX reopen
assumption — but this broker's MT5 does not print until Monday 00:00, so 3 phantom
Sunday-evening bars leaked through every weekend. This test now fails if ANY Sat/Sun
bar survives, so both bug classes (and the phantom-Sunday regression) can't return
silently.

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
    out = feed_adapter._strip_market_closed(df)   # FX default (no td_symbol)

    dow = out.index.dayofweek

    # 1) the broker week is Mon 00:00 -> Fri 23:00 UTC (verified against MT5:
    #    1yr, all 5 pairs, ZERO Sat and ZERO Sun bars). NOTHING on Sat or Sun
    #    may survive — including Sunday >= 21:00, which the old rule wrongly kept
    #    (3 phantom Sunday-evening bars every weekend shoved every post-weekend
    #    candle sideways vs MT5 on the positional-x charts).
    sat = (dow == 5)
    if sat.any():
        _fail(f"{int(sat.sum())} Saturday bar(s) survived the strip: "
              f"{list(out.index[sat])}")
    sun = (dow == 6)
    if sun.any():
        _fail(f"{int(sun.sum())} Sunday bar(s) survived the strip (MT5 has ZERO "
              f"Sunday bars — this is the phantom-Sunday regression): "
              f"{list(out.index[sun])}")

    # ...and legit bars are kept: all of Friday, all of Monday.
    kept_fri = (df.index.dayofweek == 4).sum()
    if (out.index.dayofweek == 4).sum() != kept_fri:
        _fail("a legit Friday bar was dropped")
    if not (out.index.dayofweek == 0).any():
        _fail("Monday bars were dropped")
    # Fri 23:00 (last legit bar of the week) must survive.
    if not ((out.index.dayofweek == 4) & (out.index.hour == 23)).any():
        _fail("Friday 23:00 UTC (last bar before the weekend gap) was dropped")

    # 2) the strip creates the real Fri23:00 -> Mon00:00 session gap (>= 49h) the
    #    gap-aware detectors were built for — not a partial Sunday-evening reopen.
    deltas = out.index.to_series().diff().dropna()
    max_gap_h = deltas.max().total_seconds() / 3600.0
    if max_gap_h < 25.0:
        _fail(f"no full weekend gap after strip (max delta {max_gap_h:.1f}h) — "
              f"expected the Fri 23:00 -> Mon 00:00 break (~49h); a smaller gap "
              f"means Sunday-evening phantom bars leaked through")

    # 2b) FX keeps its weekday 00:00 UTC bars (FX prints all 24 hours in MT5).
    if not ((out.index.dayofweek == 0) & (out.index.hour == 0)).any():
        _fail("FX Monday 00:00 UTC bar was dropped — FX has no daily 00:00 gap")

    # --- GOLD daily-break rule -------------------------------------------------
    # MT5 Gold omits the 00:00 UTC hour on EVERY weekday (verified 264/264 weekdays
    # over 1yr). Twelve Data pads it, so for XAU the strip must drop weekday 00:00
    # too — else Gold drifts one bar per day vs MT5. FX must NOT get this rule.
    gout = feed_adapter._strip_market_closed(df, td_symbol="XAU/USD")
    gdow = gout.index.dayofweek
    bad_gold_midnight = (gdow < 5) & (gout.index.hour == 0)
    if bad_gold_midnight.any():
        _fail(f"{int(bad_gold_midnight.sum())} Gold weekday 00:00 UTC bar(s) survived — "
              f"MT5 Gold has none: {list(gout.index[bad_gold_midnight])}")
    # Gold still keeps a non-midnight weekday bar (didn't over-strip the day).
    if not ((gdow < 5) & (gout.index.hour == 1)).any():
        _fail("Gold strip removed a legit weekday 01:00 UTC bar (over-stripping)")
    # And the FX default did NOT strip weekday 00:00 (rule is Gold-only).
    fx_midnight_kept = ((dow < 5) & (out.index.hour == 0)).any()
    if not fx_midnight_kept:
        _fail("FX default wrongly stripped weekday 00:00 — Gold rule leaked to FX")

    # 3) the strip is actually wired into the network path.
    src = inspect.getsource(feed_adapter._to_dataframe)
    if "_strip_market_closed" not in src:
        _fail("_to_dataframe does not call _strip_market_closed — strip is dead code")

    print(f"PASS: FX stripped {len(df) - len(out)} weekend bars (gap {max_gap_h:.0f}h "
          f"restored); Gold additionally strips weekday 00:00 UTC; FX untouched; "
          f"wired into _to_dataframe.")
    sys.exit(0)


if __name__ == "__main__":
    main()
