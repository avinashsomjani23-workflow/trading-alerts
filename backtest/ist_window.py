"""IST trading-window gate for the backtest.

Mirrors the live system's blackout: outside the user's IST trading window,
live suppresses everything (no scans, no alerts, no trades). The backtest
must do the same -- otherwise WR/PnL include hours the user never trades.

Window (single source of truth, expressed in UTC):
  all instruments : 03:30 .. 18:30 UTC  (=  09:00 .. 24:00 IST)

Live blocks the WHOLE scan before 09:00 IST (smc_radar.py: "do nothing before
09:00 IST"), for every pair. The trading window is 09:00 -> 24:00 (midnight)
IST; its complement is 00:00 .. 09:00 IST, which is exactly what live's
`hour < 9` gate suppresses. There is NO separate index window in live, so we
use one window for all instruments here too.

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
    """True iff alert timestamp falls inside the user's IST trading window.
    ts MUST be tz-aware UTC. `pair_type` is accepted for signature stability
    but ignored: live blocks all instruments before 09:00 IST with no separate
    index window, so one window (09:00-24:00 IST = 03:30-18:30 UTC) covers all."""
    if ts.tzinfo is None:
        raise ValueError("in_user_trading_window requires tz-aware ts")
    m = _utc_minute(ts)
    # 03:30 .. 18:30 UTC inclusive on the open, exclusive on the close.
    return 3 * 60 + 30 <= m < 18 * 60 + 30


def window_label(pair_type: str) -> str:
    """Human-readable window string for the report copy. One window for all
    instruments (live has no separate index window)."""
    return "UTC 03:30-18:30 (IST 09:00-24:00)"
