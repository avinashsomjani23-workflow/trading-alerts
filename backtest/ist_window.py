"""IST trading-window gate for the backtest.

Mirrors the live system's blackout: outside the user's IST trading window,
live suppresses everything (no scans, no alerts, no trades). The backtest
must do the same -- otherwise WR/PnL include hours the user never trades.

Window (single source of truth, expressed in UTC):
  forex / commodity : 03:30 .. 18:30 UTC  (=  09:00 .. 24:00 IST)
  index             : 13:00 .. 20:00 UTC  (=  18:30 .. 01:30 IST)

The IST gate is a *hard filter*: alerts that fall fully outside the
trading window above are dropped.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def _utc_minute(ts: pd.Timestamp) -> int:
    """Minutes-since-UTC-midnight for ts. Assumes ts is tz-aware UTC."""
    return ts.hour * 60 + ts.minute


def in_user_trading_window(ts: pd.Timestamp, pair_type: str) -> bool:
    """True iff alert timestamp falls inside the user's IST trading window
    for this pair_type. ts MUST be tz-aware UTC."""
    if ts.tzinfo is None:
        raise ValueError("in_user_trading_window requires tz-aware ts")
    m = _utc_minute(ts)
    if pair_type == "index":
        # 13:00 .. 20:00 UTC inclusive on the open, exclusive on the close.
        return 13 * 60 <= m < 20 * 60
    # forex / commodity / default.
    # 03:30 .. 18:30 UTC inclusive on the open, exclusive on the close.
    return 3 * 60 + 30 <= m < 18 * 60 + 30


def window_label(pair_type: str) -> str:
    """Human-readable window string for the report copy."""
    if pair_type == "index":
        return "UTC 13:00-20:00 (IST 18:30-01:30)"
    return "UTC 03:30-18:30 (IST 09:00-24:00)"
