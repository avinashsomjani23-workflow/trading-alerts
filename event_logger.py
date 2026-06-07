"""
Structure event logger.

Append-only JSON log of every BOS / CHoCH / range-break detection — qualified
or rejected — across all instruments and timeframes. Lets us answer questions
like "price reversed from 0.4x ATR but ATR threshold is 0.6x — did we filter
out a real CHoCH?" by reading back the rejection reason later.

File layout:
    logs/structure_events_YYYY-MM.json     (monthly rotation)

Logs are gitignored. Email/archive cadence is handled by weekly_review.py
(weekly cleanse) so the working directory does not bloat over time.

Each entry captures:
    log_ts            ISO timestamp the entry was written
    candle_ts         ISO timestamp of the candle the event is about
    pair              instrument name (EURUSD / NAS100 / etc.)
    timeframe         'H1' (only H1 for now; M15/M5 hooks can follow later)
    event_kind        'BOS' | 'CHoCH' | 'RangeBreak' | 'BREAK_REJECTED'
                       | 'FALLBACK_USED'
    direction         'bullish' | 'bearish'  (for break events)
    swing_price       price level that was tested / broken
    swing_ts          ISO timestamp of the swing being tested (when known)
    close_price       break candle close
    displacement      |close - swing_price|, raw price units
    displacement_atr  displacement / atr  (so threshold comparison is portable)
    threshold_atr     ATR multiplier required for this event type
    atr               raw ATR value at the candle
    wall_status       'confirmed' | 'tentative' (whether the broken level was
                      a real swing or a rolling extreme)
    reject_reason     only present for BREAK_REJECTED
                      'atr_threshold' | 'no_swing' | other
    candle_body       break-candle body (|close - open|), price units
    candle_range      break-candle range (high - low), price units
    body_range_pct    candle_body / candle_range (impulsive observation)

Logging failures NEVER propagate. The trading flow must not depend on the
log file being writable. All exceptions are swallowed silently.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any

LOG_DIR = "logs"


def _ensure_log_dir() -> None:
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def _log_path_for_now() -> str:
    """Monthly rotation: logs/structure_events_2026-05.json"""
    now = datetime.utcnow()
    return os.path.join(LOG_DIR, f"structure_events_{now.strftime('%Y-%m')}.json")


def log_event(event: Dict[str, Any]) -> None:
    """
    Append a structure event to the current month's log file.

    Read existing list, append, atomic write back. Single-process safe; we
    are not optimising for concurrent writers (the scanner runs as one
    process).

    Any exception is swallowed — logging must never break the system.
    """
    try:
        _ensure_log_dir()
        path = _log_path_for_now()
        if 'log_ts' not in event:
            event['log_ts'] = datetime.utcnow().isoformat()

        existing = []
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []

        existing.append(event)
        tmp = path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(existing, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        # Logging must never break trading. Swallow silently.
        pass


