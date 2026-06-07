"""Auto-detects backtest regime (WAR vs BAU) from a curated date-range file.

Source of truth: backtest/WAR_REGIME_WEEKS.json. The file is hand-curated --
this module does NOT infer regime from news feeds. If a run window overlaps
ANY day inside a WAR range, the run is tagged WAR; otherwise BAU.

Why the "any overlap" rule: a backtest week that contains even one day of
WAR-regime price action carries that regime's behaviour (vol spike,
safe-haven flow). Tagging by majority would hide it.
"""

from __future__ import annotations
import json
from datetime import date, datetime
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

_WAR_FILE = Path(__file__).parent / "WAR_REGIME_WEEKS.json"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _load_ranges() -> List[Dict[str, Any]]:
    if not _WAR_FILE.exists():
        return []
    with open(_WAR_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("war_ranges", []) or []


def detect_regime(start: str, end: str) -> Tuple[str, Optional[str]]:
    """Return (regime, matched_label).

    regime is 'war' or 'bau'. matched_label is the WAR range label that
    triggered the WAR tag, or None for BAU runs. Inputs are YYYY-MM-DD.
    """
    s = _parse_date(start)
    e = _parse_date(end)
    if e < s:
        raise ValueError(f"end {end} is before start {start}")

    for r in _load_ranges():
        try:
            rs = _parse_date(r["start"])
            re_ = _parse_date(r["end"])
        except (KeyError, ValueError):
            continue
        # Inclusive overlap: ranges touch if max(starts) <= min(ends).
        if max(s, rs) <= min(e, re_):
            return "war", r.get("label")
    return "bau", None


