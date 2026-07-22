"""Per-pair killzone membership tag for the backtest.

A killzone is the window in which institutional desks are active and SMC
structure actually carries weight. Outside it, price drifts and OB tests
are noise.

NOT a drop. This module only ANSWERS "is this ts in a killzone?" — it is a
TAG, not a gate. run_backtest.py:209 sets `killzone_blocked` on the row and
increments a counter, but the trade is still simulated and appended
(run_backtest.py:219,246). Killzone never removes a trade from the run,
the aggregates, or the Excel (the only real drop is the score floor; see
memory Backtest Live Parity). The old header claimed "we drop those alerts
entirely" — that was false and is corrected here.

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
    """Human-readable label of configured session-local windows, for report copy.

    Each window carries a session name (config `label`, e.g. "New York
    (forex)", "Asian Range"). We surface it so the report shows WHICH session a
    window is, not just raw times -- different pairs run different sessions
    (USDJPY uses Asian Range; USDCHF uses London Open), and a bare time range
    can't tell them apart.
    """
    killzones = pair_conf.get("killzones") or []
    if not killzones:
        return "no killzone configured"

    def _one(w: dict) -> str:
        times = f"{w.get('start')}-{w.get('end')} {w.get('tz')}"
        name = w.get("label")
        return f"{name}: {times}" if name else times

    return ", ".join(_one(w) for w in killzones if isinstance(w, dict))
