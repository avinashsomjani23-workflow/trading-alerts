"""Per-pair killzone hard filter for the backtest.

A killzone is the window in which institutional desks are active and SMC
structure actually carries weight. Outside it, price drifts and OB tests
are noise. We drop those alerts entirely -- they never enter the
simulator, never appear in aggregates, never appear in the Excel.

Windows live in config.json (pair_conf['killzones']) as SESSION-LOCAL time +
an IANA timezone, e.g. {"tz": "America/New_York", "start": "07:00", "end":
"10:00"}. The shared smc_detector engine resolves them to UTC per candle DATE
so DST is handled automatically -- the same window the live engine uses. This
module never parses windows itself; it delegates to keep one source of truth.

Half-open at the close: 07:00 inclusive, 10:00 exclusive.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import smc_detector


def in_pair_killzone(ts: pd.Timestamp, pair_conf: dict) -> bool:
    """True iff `ts` falls inside any configured killzone window for this pair.

    - ts must be tz-aware UTC (raises otherwise -- silent tz bugs are worse
      than a loud failure).
    - If pair_conf has no `killzones`, return True (no filter). A missing
      config never blocks data; it just means no killzone gate.
    """
    if ts.tzinfo is None:
        raise ValueError("in_pair_killzone requires tz-aware ts")
    killzones = pair_conf.get("killzones")
    if not killzones:
        return True
    return smc_detector.ts_in_killzone(ts.isoformat(), killzones)


def windows_label(pair_conf: dict) -> str:
    """Human-readable label of configured session-local windows, for report copy."""
    killzones = pair_conf.get("killzones") or []
    if not killzones:
        return "no killzone configured"
    return ", ".join(
        f"{w.get('start')}-{w.get('end')} {w.get('tz')}"
        for w in killzones if isinstance(w, dict)
    )
