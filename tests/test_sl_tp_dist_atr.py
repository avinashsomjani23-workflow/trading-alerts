"""sl_dist_atr_at_alert / tp_dist_atr_at_alert — structural guards.

Observe-only feature: how big this trade's STOP / TARGET is versus NORMAL recent
movement, judged AT THE ALERT. Anchor = OB proximal line; ruler = a FRESH ATR(14)
on the last 14 CLOSED H1 candles as of the alert (drops the forming bar).

Out-of-band; zero live-path asserts (CLAUDE.md guard rule — a guard must never
sit inside live alert generation). Failure mode guarded: a silent anchor drift
(entry instead of proximal) or a stale-ATR regression (formation ATR instead of
the fresh closed-bar one) would quietly mislabel every trade's stop/target size
in the next canonical run — exactly the staleness this feature exists to fix.

  1  Math — proximal-anchored ratio, both columns, known ATR
  2  Anchor is PROXIMAL, not entry (bite-proven: entry != proximal)
  3  Ruler is the FRESH closed-bar ATR(14), not the stale formation ATR
  4  Forming bar excluded — appending a wild forming bar leaves the value unchanged
  5  None-ability — short slice / missing level -> None, never a fake 0.0

Run:  python -m pytest tests/test_sl_tp_dist_atr.py -q
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "backtest"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import smc_detector  # noqa: E402
from h1_only_simulator import _closed_bars_at_alert  # noqa: E402


# ---------------------------------------------------------------------------
# The feature's computation, mirrored from _build_row (same formula, same
# anchor, same fresh closed-bar ATR). Tested in isolation so a change to the
# arithmetic bites here rather than only inside the full row build.
# ---------------------------------------------------------------------------
def _dist_atr(df_h1, alert_ts, proximal, sl, tp1):
    _slice = _closed_bars_at_alert(df_h1, alert_ts)
    atr = smc_detector.compute_atr(_slice, period=14) if _slice is not None else None
    if not (atr and atr > 0 and proximal is not None):
        return None, None
    sl_d = round(abs(proximal - sl) / atr, 3) if sl is not None else None
    tp_d = round(abs(proximal - tp1) / atr, 3) if tp1 is not None else None
    return sl_d, tp_d


def _frame(n=30, start="2019-01-01 00:00", rng=1.0, base=100.0):
    """n hourly bars, each with High-Low == rng (so ATR == rng), flat closes."""
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    highs = np.full(n, base + rng / 2)
    lows = np.full(n, base - rng / 2)
    closes = np.full(n, base)
    opens = np.full(n, base)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx
    )


def test_1_math_both_columns():
    df = _frame(rng=1.0)                       # ATR == 1.0
    alert_ts = df.index[-1]                    # last row = forming bar
    # proximal 100.0, SL 98.5 -> 1.5 ATR; TP1 104.0 -> 4.0 ATR
    sl_d, tp_d = _dist_atr(df, alert_ts, proximal=100.0, sl=98.5, tp1=104.0)
    assert sl_d == 1.5
    assert tp_d == 4.0


def test_2_anchor_is_proximal_not_entry():
    df = _frame(rng=1.0)
    alert_ts = df.index[-1]
    # If the anchor were entry (say 99.0) the SL distance would be 0.5, not 1.5.
    sl_d, _ = _dist_atr(df, alert_ts, proximal=100.0, sl=98.5, tp1=104.0)
    assert sl_d == 1.5  # proximal 100.0, NOT an entry at 99.0


def test_3_ruler_is_fresh_not_formation():
    # A formation ATR frozen at OB birth could be anything; the fresh ruler is
    # derived from THIS slice. Double the slice's range -> the ratio halves.
    df = _frame(rng=2.0)                       # ATR == 2.0
    alert_ts = df.index[-1]
    sl_d, tp_d = _dist_atr(df, alert_ts, proximal=100.0, sl=98.0, tp1=104.0)
    assert sl_d == 1.0   # |100-98| / 2.0
    assert tp_d == 2.0   # |100-104| / 2.0


def test_4_forming_bar_excluded():
    df = _frame(rng=1.0)
    alert_ts = df.index[-1]
    base_sl, base_tp = _dist_atr(df, alert_ts, 100.0, 98.5, 104.0)
    # Poison ONLY the forming (last) bar with a huge range. If it leaked into the
    # ATR the ratio would shrink. Closed-bar exclusion => identical result.
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("High")] = 200.0
    df2.iloc[-1, df2.columns.get_loc("Low")] = 0.0
    sl_d, tp_d = _dist_atr(df2, alert_ts, 100.0, 98.5, 104.0)
    assert (sl_d, tp_d) == (base_sl, base_tp)


def test_5_none_ability():
    # Too few closed bars for ATR(14): 8 rows -> last dropped -> 7 < 15 needed.
    short = _frame(n=8, rng=1.0)
    sl_d, tp_d = _dist_atr(short, short.index[-1], 100.0, 98.5, 104.0)
    assert sl_d is None and tp_d is None
    # Healthy ATR but a missing level -> that side None, never a fake 0.0.
    df = _frame(rng=1.0)
    sl_d, tp_d = _dist_atr(df, df.index[-1], 100.0, None, None)
    assert sl_d is None and tp_d is None
