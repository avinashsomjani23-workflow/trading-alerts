"""Per-pair killzone hard filter for the backtest.

A killzone is the window in which institutional desks are active and SMC
structure actually carries weight. Outside it, price drifts and OB tests
are noise. We drop those alerts entirely -- they never enter the
simulator, never appear in aggregates, never appear in the Excel.

Windows come from config.json (pair_conf['killzones_utc']), expressed as a
list of [start_hhmm, end_hhmm] pairs in UTC. Multiple windows per pair are
supported so split-session pairs (USDJPY, NZDUSD) work.

A 30-minute buffer is already baked into the configured windows -- the
config values are the *effective* killzone, not the raw exchange session.

Half-open at the close: a window of ["07:00", "16:30"] means 07:00:00
inclusive, 16:30:00 exclusive.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd


def _parse_hhmm(s: str) -> int:
    """Convert 'HH:MM' to minutes-since-midnight. Raises on malformed input."""
    h_str, m_str = s.split(":")
    return int(h_str) * 60 + int(m_str)


def _normalise_windows(raw: Sequence) -> List[Tuple[int, int]]:
    """Convert config list to a list of (start_min, end_min) integer tuples.
    Returns an empty list if the config entry is missing or malformed --
    caller decides whether that means 'no filter' or 'block all'."""
    out: List[Tuple[int, int]] = []
    if not raw:
        return out
    for window in raw:
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            continue
        try:
            start = _parse_hhmm(str(window[0]))
            end   = _parse_hhmm(str(window[1]))
        except (ValueError, TypeError):
            continue
        if not (0 <= start < 24 * 60) or not (0 < end <= 24 * 60):
            continue
        out.append((start, end))
    return out


def in_pair_killzone(ts: pd.Timestamp, pair_conf: dict) -> bool:
    """True iff `ts` falls inside any configured killzone window for this pair.

    Behaviour:
      - ts must be tz-aware UTC (raises otherwise -- silent tz bugs are worse
        than a loud failure).
      - If pair_conf has no `killzones_utc` or it's empty/malformed, return
        True (i.e. no filter applied). This is the safe default: a missing
        config never blocks live data; it just means no killzone gate.
    """
    if ts.tzinfo is None:
        raise ValueError("in_pair_killzone requires tz-aware ts")
    windows = _normalise_windows(pair_conf.get("killzones_utc") or [])
    if not windows:
        return True  # no killzone configured -> no filter
    m = ts.hour * 60 + ts.minute
    return any(start <= m < end for start, end in windows)


def windows_label(pair_conf: dict) -> str:
    """Human-readable label of configured windows, for report copy."""
    windows = _normalise_windows(pair_conf.get("killzones_utc") or [])
    if not windows:
        return "no killzone configured"
    return ", ".join(
        f"{s // 60:02d}:{s % 60:02d}-{e // 60:02d}:{e % 60:02d} UTC"
        for s, e in windows
    )
