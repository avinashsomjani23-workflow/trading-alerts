"""
trend_state.py — H1 trend label from pure swing structure.

ISOLATED, ADDITIVE, INFORMATION-ONLY.
  - Reads the SINGLE shared swing pool (dealing_range.detect_swings: lb-3 +
    ATR1.5). It does NOT define its own swings.
  - It does NOT read, write, or depend on dealing_range's wall `trend`,
    `trend_start_idx`, BOS/CHoCH events, or structure_state.json. The wall
    engine and this module never touch each other's state.
  - Its output GATES NOTHING. No OB selection, no trade filtering, no scoring.
    It is a label the trader reads and collects data on.

WHAT IT COMPUTES
  A single state, computed by walking ALL confirmed swings in time order and
  reporting the state AFTER THE LATEST SWING / LATEST CANDLE (never frozen
  early). Four states:

    UP        : higher-high + higher-low structure intact.
    DOWN      : lower-high + lower-low structure intact.
    RANGING   : a confirmed trend's defended level was broken (a candle CLOSED
                beyond it) but the opposite trend has NOT yet confirmed, AND
                price has not reclaimed. Carries the PRIOR trend as context
                ("up / ranging" or "down / ranging"). This is the change-of-
                character / transition window. Information only.
    UNDEFINED : cold start only — no trend has ever been established yet.
                Has NO prior-trend memory. Reached only at the very start of
                the walk; once a trend forms it is never re-entered.

DEFENDED LEVEL  (the whole machine)
  UP   : the most recent CONFIRMED higher-low. Break = a candle CLOSE below it.
  DOWN : the most recent CONFIRMED lower-high. Break = a candle CLOSE above it.
  The level only ratchets in the trend's favour (up: low rises; down: high
  falls). Break is checked on the LIVE close against the ALREADY-CONFIRMED
  level. Per-candle order is: CHECK BREAK FIRST, THEN ratchet (so a flip is
  never masked by same-bar ratcheting).

NO swing counts anywhere. A single counter-swing cannot flip the trend; only a
close beyond the defended level can.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional

import dealing_range as _dr  # SINGLE source of the swing pool. No other coupling.

# Public state constants.
UP        = "up"
DOWN      = "down"
RANGING   = "ranging"
UNDEFINED = "undefined"


def _confirm_idx(swing: Dict[str, Any], lookback: int) -> int:
    """The candle index at which a swing becomes KNOWN (lb-3 confirmation lag).
    A swing at idx i is only confirmed after `lookback` further candles."""
    return int(swing["idx"]) + lookback


def compute_trend(df, lookback: int = _dr.SWING_LOOKBACK) -> Dict[str, Any]:
    """Compute the current H1 trend state from the shared swing pool.

    Args:
      df: H1 OHLC DataFrame (DatetimeIndex). Same df the wall engine scans.
      lookback: swing lookback (default = the shared SWING_LOOKBACK=3).

    Returns a dict (never raises on empty/short data — returns UNDEFINED):
      {
        'state':        'up'|'down'|'ranging'|'undefined',
        'prior_trend':  'up'|'down'|None,   # only set when state=='ranging'
        'label':        human string for the email banner,
        'defended':     float|None,         # active defended level (up: HL, down: LH)
        'last_swing_ts':str|None,
        'broke_ts':     str|None,           # ts of the candle that broke into RANGING
      }
    """
    empty = {"state": UNDEFINED, "prior_trend": None,
             "label": "H1 trend undefined (insufficient structure)",
             "defended": None, "last_swing_ts": None, "broke_ts": None}
    if df is None or len(df) < (lookback * 2 + 1):
        return empty

    swings: List[Dict[str, Any]] = _dr.detect_swings(df, lookback=lookback)
    if not swings:
        return empty

    closes = df["Close"].values.astype(float)
    n = len(df)

    # Index swings by the candle at which they become known (confirmation lag),
    # so the walk only "sees" a swing once it would actually be confirmed live.
    by_known: Dict[int, List[Dict[str, Any]]] = {}
    for s in swings:
        by_known.setdefault(_confirm_idx(s, lookback), []).append(s)

    state = UNDEFINED
    prior_trend: Optional[str] = None
    defended: Optional[float] = None       # active defended level
    broke_ts: Optional[str] = None

    # Rolling memory of the last two CONFIRMED highs and lows (for HH/HL reads).
    highs: List[Dict[str, Any]] = []
    lows: List[Dict[str, Any]] = []
    last_swing_ts: Optional[str] = None

    for ci in range(n):
        # ---- 1. BREAK CHECK FIRST (live close vs already-confirmed level) ----
        c = closes[ci]
        if state == UP and defended is not None and c < defended:
            prior_trend = UP
            state = RANGING
            broke_ts = _dr._ts_iso(df, ci)
            defended = None
        elif state == DOWN and defended is not None and c > defended:
            prior_trend = DOWN
            state = RANGING
            broke_ts = _dr._ts_iso(df, ci)
            defended = None

        # ---- 2. INGEST any swings confirmed AT this candle ----
        for s in by_known.get(ci, ()):
            last_swing_ts = s["ts"]
            if s["type"] == "high":
                highs.append(s)
            else:
                lows.append(s)

            # Pattern read needs two of each.
            HH = HL = LH = LL = False
            if len(highs) >= 2 and len(lows) >= 2:
                HH = highs[-1]["price"] > highs[-2]["price"]
                HL = lows[-1]["price"]  > lows[-2]["price"]
                LH = highs[-1]["price"] < highs[-2]["price"]
                LL = lows[-1]["price"]  < lows[-2]["price"]

            # --- Birth / confirmation / reclaim of a trend ---
            if HH and HL:
                # Up-pattern confirmed. From UNDEFINED, DOWN, or RANGING -> UP.
                # (RANGING from a prior down OR a prior up that reclaimed both
                #  resolve here cleanly — first full pattern wins.)
                state = UP
                prior_trend = None
                broke_ts = None
                defended = lows[-1]["price"]            # most recent HL
            elif LH and LL:
                state = DOWN
                prior_trend = None
                broke_ts = None
                defended = highs[-1]["price"]           # most recent LH
            else:
                # No fresh full pattern this swing — only RATCHET an existing
                # live trend (never loosen). Done AFTER the break check above,
                # so a same-bar break is never masked.
                if state == UP and s["type"] == "low" and len(lows) >= 2:
                    if lows[-1]["price"] > lows[-2]["price"]:
                        defended = lows[-1]["price"]
                elif state == DOWN and s["type"] == "high" and len(highs) >= 2:
                    if highs[-1]["price"] < highs[-2]["price"]:
                        defended = highs[-1]["price"]

    # ---- Build label ----
    if state == UP:
        label = "H1 trend UP (higher-highs / higher-lows)"
    elif state == DOWN:
        label = "H1 trend DOWN (lower-highs / lower-lows)"
    elif state == RANGING:
        label = f"H1 ranging — prior trend {prior_trend.upper()} (change of character; not yet reversed or resumed)"
    else:
        label = "H1 trend undefined (insufficient structure)"

    return {
        "state": state,
        "prior_trend": prior_trend if state == RANGING else None,
        "label": label,
        "defended": defended,
        "last_swing_ts": last_swing_ts,
        "broke_ts": broke_ts if state == RANGING else None,
    }
