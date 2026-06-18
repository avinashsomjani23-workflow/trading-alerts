"""Auto-detects backtest regime (WAR vs BAU) from a curated week-by-week map.

Source of truth: backtest/WAR_REGIME_WEEKS.json. The file is hand-curated --
this module does NOT infer regime from news feeds. It is a non-overlapping
Monday-Sunday week map; every week carries an explicit regime ('war' or 'bau')
and, for war weeks, the specific conflict label.

Lookup rule: a run's regime is the regime of the week containing the run's
START date. Multi-week runs therefore inherit the start week's label, matching
the email-subject convention ("show the event for the start of the window").
No overlap/priority logic is needed -- weeks do not overlap, so exactly one
week matches the start date.
"""

from __future__ import annotations
import json
from datetime import date, datetime
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

_WAR_FILE = Path(__file__).parent / "WAR_REGIME_WEEKS.json"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _load_weeks() -> List[Dict[str, Any]]:
    if not _WAR_FILE.exists():
        return []
    with open(_WAR_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("weeks", []) or []


def detect_regime(start: str, end: str) -> Tuple[str, Optional[str]]:
    """Return (regime, matched_label).

    regime is 'war' or 'bau'. matched_label is the conflict label for war
    weeks, or None for BAU. Inputs are YYYY-MM-DD.

    The regime is that of the week containing the START date. If the start
    date falls outside the curated map (run window predates or postdates the
    file), regime is 'bau' and label is None -- the safe default.
    """
    s = _parse_date(start)
    e = _parse_date(end)
    if e < s:
        raise ValueError(f"end {end} is before start {start}")

    for w in _load_weeks():
        try:
            ws = _parse_date(w["start"])
            we = _parse_date(w["end"])
        except (KeyError, ValueError):
            continue
        if ws <= s <= we:
            regime = w.get("regime", "bau")
            label = w.get("label") if regime == "war" else None
            return regime, label
    return "bau", None
