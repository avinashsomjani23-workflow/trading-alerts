"""MT5 candle-label clock correction — the single source of the era table.

Cached MT5 H1 candles are labeled with the BROKER's server clock, not true UTC.
The server clock changed twice over 18 years, so the fixed `-3h` conversion in
`mt5_data/import_mt5.py` is right only for the most recent era. Prices are fine;
only the HOUR LABEL is wrong, by +/-1h, for ~5 months/year in most years.

Empirically proven (2026-07-16) by spike-aligning 13,839 high-impact ForexFactory
events (page-embedded epoch = absolute UTC) against the cached H1 candles: all 36
year-season cells 2008-2025 peak at offset 0 ONLY AFTER this correction.

label - true_utc, in hours. Seasons follow Europe/Athens DST:
    era A   ..2014-10-31         0 in EU-DST summer, -1 in winter  (EET+DST)
    flip    2014-11-01..12-07    regime flip, per-event votes conflict -> None
    era B   2014-12-08..2024-10-26   +1 in EU-DST summer, 0 in winter (UTC+3+DST)
    era C   2024-10-27..         0 year-round (fixed UTC+3; the -3h pin era)

Evidence + the "isn't it just delayed news?" rebuttal: MT5_CANDLE_CLOCK_AUDIT.md.
This module is the ONE implementation — both import_mt5.py (source fix) and
news_enrichment.py import from here (one concept, one implementation, per CLAUDE.md).

IMPORTANT — what a caller feeds in: the input `label_ts` is a cached-candle label,
i.e. the PROVISIONAL `-3h` UTC stamp import_mt5 currently produces. The era table
was calibrated against exactly those provisional labels, so it composes at import:
    true_utc = provisional_label - mt5_label_error_hours(provisional_label)
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from zoneinfo import ZoneInfo

_EU = ZoneInfo("Europe/Athens")
_ERA_B_START = pd.Timestamp("2014-12-08", tz="UTC")
_AMBIG_START = pd.Timestamp("2014-11-01", tz="UTC")
_ERA_C_START = pd.Timestamp("2024-10-27", tz="UTC")


def is_flip_window(label_ts: pd.Timestamp) -> bool:
    """True if the label falls in the 2014-11-01..12-07 regime-flip window.

    These bars cannot be resolved to the hour (per-event votes conflict), so
    every clock column derived from them stays NULL, never a guessed hour.
    """
    return bool(_AMBIG_START <= label_ts < _ERA_B_START)


def mt5_label_error_hours(label_ts: pd.Timestamp) -> Optional[int]:
    """Hours to SUBTRACT from a cached-candle label to get true UTC.

    None = inside the 2014 regime-flip ambiguity window (unknowable).
    """
    if label_ts >= _ERA_C_START:
        return 0
    if is_flip_window(label_ts):
        return None
    eu_dst = bool(label_ts.tz_convert(_EU).dst())
    if label_ts >= _ERA_B_START:
        return 1 if eu_dst else 0
    return 0 if eu_dst else -1


def true_utc(label_ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Cached-candle label -> true UTC bar-open, or None if ambiguous."""
    if pd.isna(label_ts):
        return None
    err = mt5_label_error_hours(label_ts)
    if err is None:
        return None
    return label_ts - pd.Timedelta(hours=err)
