"""
Dealing Range — single source of truth for H1 dealing range walls.

Concepts (plain English):

A dealing range has two walls: a CEILING (upper) and a FLOOR (lower). Walls
are anchored to confirmed swing highs / swing lows on H1. They update only
when a wall is broken by a candle's CLOSE (not wick).

A wall break is classified as:
  - BOS   (Break of Structure)    — break in the direction of the current trend
  - CHoCH (Change of Character)  — break against the current trend (reversal)

Each wall is either real (a confirmed swing) or a placeholder (rolling-max /
rolling-min while a new swing has not yet formed). PD scoring is suspended
on the placeholder side.

Phase 1 calls:
    dealing_range.update_pair(df_h1, prior_state, pair_conf)

This either cold-starts (if no prior state) by walking forward through the
fetched H1 window, or runs incrementally (if prior state exists). Returns
the new state for that pair plus a fallback flag.

Phase 2 reads `state/structure_state.json` and consumes:
    dealing_range.compute_pd_position(price, walls)

Phase 2 NEVER writes state. One writer (Phase 1), many readers.

Design decisions (locked):
  - Lookback for swing confirmation: 3 (3 before + swing + 3 after = 7 candles)
  - BOS leg threshold:   0.4 x H1 ATR
  - CHoCH leg threshold: 0.75 x H1 ATR
  - Trailing wall on BOS = most recent confirmed swing INSIDE the just-completed leg.
    No ATR pullback threshold. If no confirmed swing is inside the leg, wall stays.
  - CHoCH wall reset = most recent confirmed swing of the just-completed trend
    (the swing that formed just before the swing whose break was the CHoCH).
    NOT the absolute trend extreme. This kills the runaway-extreme problem.
  - Cold-start window: 150 H1 candles.
  - Internal swings inside the range generate no structural events.
  - No silent fallback. If no event fires across the cold-start window,
    fallback to window high/low and flag fallback_active = True.
  - Atomic writes (temp + rename), like every other state file in this codebase.
"""

import json
import os
from typing import Optional, Tuple, List, Dict, Any

# --- Tunables (locked) -------------------------------------------------------

SWING_LOOKBACK = 3
BOS_ATR_MULT   = 0.4
CHOCH_ATR_MULT = 0.75

# Cold-start window — number of most recent H1 candles to walk forward.
COLDSTART_WINDOW_H1 = 150

# State file path. Lives in a dedicated directory outside any purge scope.
STATE_DIR  = "state"
STATE_PATH = os.path.join(STATE_DIR, "structure_state.json")


# --- ATR (local copy to avoid import cycle) ----------------------------------

def _compute_atr(df, period: int = 14) -> Optional[float]:
    """Mirror of smc_detector.compute_atr — duplicated to avoid circular import."""
    if df is None or len(df) < period + 1:
        return None
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    C = df['Close'].values.astype(float)
    trs = []
    for i in range(1, len(C)):
        tr = max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))
        trs.append(tr)
    if len(trs) < period:
        return None
    sumv = 0.0
    for v in trs[-period:]:
        sumv += v
    return sumv / period


# --- Atomic JSON I/O ---------------------------------------------------------

def _ensure_state_dir():
    if not os.path.isdir(STATE_DIR):
        os.makedirs(STATE_DIR, exist_ok=True)


def load_state() -> Dict[str, Any]:
    """Load structure_state.json. Returns empty dict on any failure."""
    try:
        with open(STATE_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def save_state(state: Dict[str, Any]) -> None:
    """Atomic write: temp file then rename. Same pattern used elsewhere."""
    _ensure_state_dir()
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


# --- Timestamp helper --------------------------------------------------------

def _ts_iso(df, idx: int) -> Optional[str]:
    """Return ISO timestamp string for df row at positional idx."""
    if df is None or idx is None:
        return None
    try:
        idx = int(idx)
        if idx < 0 or idx >= len(df):
            return None
        if 'Datetime' in df.columns:
            raw = df['Datetime'].iloc[idx]
        elif 'Date' in df.columns:
            raw = df['Date'].iloc[idx]
        else:
            raw = df.index[idx]
        if hasattr(raw, 'isoformat'):
            return raw.isoformat()
        return str(raw)
    except Exception:
        return None


# --- Swing detection ---------------------------------------------------------

def detect_swings(df, lookback: int = SWING_LOOKBACK) -> List[Dict[str, Any]]:
    """
    Find confirmed swing highs and swing lows over the entire df.

    A candle at idx i is:
      - a swing high if H[i] is the strict max over [i-lookback, i+lookback]
      - a swing low  if L[i] is the strict min over [i-lookback, i+lookback]

    Both can fire on the same candle (rare but possible). Returns list sorted
    by idx, each entry: {'type': 'high'|'low', 'idx': i, 'price': float, 'ts': iso}.
    """
    if df is None or len(df) < lookback * 2 + 1:
        return []
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    n = len(df)
    out = []
    for i in range(lookback, n - lookback):
        wh = H[i - lookback: i + lookback + 1]
        wl = L[i - lookback: i + lookback + 1]
        if H[i] == max(wh):
            out.append({'type': 'high', 'idx': i, 'price': float(H[i]), 'ts': _ts_iso(df, i)})
        if L[i] == min(wl):
            out.append({'type': 'low',  'idx': i, 'price': float(L[i]), 'ts': _ts_iso(df, i)})
    out.sort(key=lambda s: s['idx'])
    return out


# --- Core: walk forward and build state -------------------------------------

def _empty_state() -> Dict[str, Any]:
    return {
        "trend": None,                        # 'bullish' | 'bearish' | None
        "ceiling_price": None,
        "ceiling_ts": None,
        "ceiling_is_placeholder": True,
        "floor_price": None,
        "floor_ts": None,
        "floor_is_placeholder": True,
        "last_event_type": None,              # 'BOS' | 'CHoCH' | None
        "last_event_ts": None,
        "last_scanned_ts": None,
        "fallback_active": False,
    }


def _resolve_placeholder(side: str, df, leg_start_idx: int, leg_end_idx: int,
                         all_swings: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Try to resolve a placeholder wall on `side` ('ceiling' or 'floor') by
    finding the highest swing high (for ceiling) or lowest swing low (for
    floor) inside [leg_start_idx, leg_end_idx]. Returns (swing_dict_or_None,
    is_placeholder).

    If no confirmed swing is present in the range, returns (rolling_extreme,
    True) where rolling_extreme is a synthetic dict with the highest H or
    lowest L in the range (visualization only).
    """
    if df is None or leg_start_idx >= leg_end_idx:
        return None, True

    target_type = 'high' if side == 'ceiling' else 'low'
    candidates = [s for s in all_swings
                  if s['type'] == target_type
                  and leg_start_idx <= s['idx'] <= leg_end_idx]
    if candidates:
        if side == 'ceiling':
            best = max(candidates, key=lambda s: s['price'])
        else:
            best = min(candidates, key=lambda s: s['price'])
        return {'idx': best['idx'], 'price': best['price'], 'ts': best['ts']}, False

    # No confirmed swing yet — produce rolling extreme for visualization.
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    if side == 'ceiling':
        rng_idx = leg_start_idx + int(H[leg_start_idx: leg_end_idx + 1].argmax())
        return {'idx': rng_idx, 'price': float(H[rng_idx]), 'ts': _ts_iso(df, rng_idx)}, True
    else:
        rng_idx = leg_start_idx + int(L[leg_start_idx: leg_end_idx + 1].argmin())
        return {'idx': rng_idx, 'price': float(L[rng_idx]), 'ts': _ts_iso(df, rng_idx)}, True


def _trail_inside_leg(side: str, swings: List[Dict[str, Any]],
                      leg_start_idx: int, leg_end_idx: int) -> Optional[Dict[str, Any]]:
    """
    Pick the trailing wall on the OPPOSITE side of a BOS.

    For a bullish BOS: side='floor', returns the LOWEST confirmed swing low
    inside (leg_start_idx, leg_end_idx). Plain English: deepest pullback
    inside the leg = new floor.

    For a bearish BOS: side='ceiling', returns the HIGHEST confirmed swing
    high inside that range.

    Strict interior — endpoints excluded (the leg-bounding swings themselves
    are NOT pullbacks).

    Returns None if no qualifying confirmed swing exists. Caller keeps the
    prior wall in that case.
    """
    target_type = 'low' if side == 'floor' else 'high'
    interior = [s for s in swings
                if s['type'] == target_type
                and leg_start_idx < s['idx'] < leg_end_idx]
    if not interior:
        return None
    if side == 'floor':
        best = min(interior, key=lambda s: s['price'])
    else:
        best = max(interior, key=lambda s: s['price'])
    return {'idx': best['idx'], 'price': best['price'], 'ts': best['ts']}


def _most_recent_swing_before(swings: List[Dict[str, Any]], swing_type: str,
                              before_idx: int) -> Optional[Dict[str, Any]]:
    """Most recent confirmed swing of `swing_type` strictly before `before_idx`."""
    matches = [s for s in swings if s['type'] == swing_type and s['idx'] < before_idx]
    if not matches:
        return None
    matches.sort(key=lambda s: s['idx'], reverse=True)
    s = matches[0]
    return {'idx': s['idx'], 'price': s['price'], 'ts': s['ts']}


def _walk_forward(df, prior_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Walk forward through df starting from either:
      - candle 0 (cold start, prior_state is empty/None), or
      - the candle just after prior_state['last_scanned_ts'] (incremental).

    Apply BOS/CHoCH detection against current walls (NOT against latest
    swings). Update walls per locked rules. Return updated state.

    Cold start fallback: if no event fires across the entire window, walls
    fall back to window high/low and fallback_active = True.
    """
    n = len(df) if df is not None else 0
    if n == 0:
        return prior_state or _empty_state()

    atr = _compute_atr(df)
    if atr is None or atr <= 0:
        # Can't classify legs without ATR — hold prior state if any, else empty.
        return prior_state or _empty_state()

    swings = detect_swings(df, lookback=SWING_LOOKBACK)

    # Determine starting candle for the forward walk.
    state = prior_state if prior_state else _empty_state()
    is_cold_start = (
        prior_state is None
        or prior_state.get("ceiling_price") is None
        or prior_state.get("floor_price") is None
    )

    start_i = 0
    if not is_cold_start and prior_state.get("last_scanned_ts"):
        # Find first candle whose ts > last_scanned_ts.
        last_ts = prior_state["last_scanned_ts"]
        for i in range(n):
            ts = _ts_iso(df, i)
            if ts and ts > last_ts:
                start_i = i
                break
        else:
            # No new candles since last scan — just refresh last_scanned_ts.
            new_state = dict(state)
            new_state["last_scanned_ts"] = _ts_iso(df, n - 1)
            return new_state

    C = df['Close'].values.astype(float)

    trend = state.get("trend")
    ceiling = {
        "price": state.get("ceiling_price"),
        "ts":    state.get("ceiling_ts"),
        "idx":   None,  # idx is local to the simulation; we re-resolve where needed
        "is_placeholder": state.get("ceiling_is_placeholder", True),
    }
    floor = {
        "price": state.get("floor_price"),
        "ts":    state.get("floor_ts"),
        "idx":   None,
        "is_placeholder": state.get("floor_is_placeholder", True),
    }

    last_event_type = state.get("last_event_type")
    last_event_ts   = state.get("last_event_ts")
    leg_start_idx   = 0  # idx of the most recent structural anchor in this df
    fallback_active = state.get("fallback_active", False)

    # Map walls back to df indices when possible (resolve "idx" from ts).
    def _idx_from_ts(ts_iso: Optional[str]) -> Optional[int]:
        if not ts_iso:
            return None
        for i in range(n):
            if _ts_iso(df, i) == ts_iso:
                return i
        return None

    if ceiling["ts"]:
        ceiling["idx"] = _idx_from_ts(ceiling["ts"])
    if floor["ts"]:
        floor["idx"] = _idx_from_ts(floor["ts"])

    # leg_start = the most recent of (last event, ceiling_idx, floor_idx).
    candidates_for_leg_start = []
    if ceiling["idx"] is not None:
        candidates_for_leg_start.append(ceiling["idx"])
    if floor["idx"] is not None:
        candidates_for_leg_start.append(floor["idx"])
    if last_event_ts:
        ev_idx = _idx_from_ts(last_event_ts)
        if ev_idx is not None:
            candidates_for_leg_start.append(ev_idx)
    if candidates_for_leg_start:
        leg_start_idx = max(candidates_for_leg_start)

    # Helpers to resolve placeholders given current candle position.
    def _try_promote_placeholder(side: str, current_i: int):
        """If side is placeholder, see if a confirmed swing now exists in
        (leg_start_idx, current_i] and promote the wall. Otherwise refresh
        rolling extreme."""
        if side == 'ceiling':
            if not ceiling["is_placeholder"]:
                return
            promoted, is_ph = _resolve_placeholder('ceiling', df, leg_start_idx + 1, current_i, swings)
            if promoted is not None:
                ceiling["price"] = promoted["price"]
                ceiling["ts"]    = promoted["ts"]
                ceiling["idx"]   = promoted["idx"]
                ceiling["is_placeholder"] = is_ph
        else:
            if not floor["is_placeholder"]:
                return
            promoted, is_ph = _resolve_placeholder('floor', df, leg_start_idx + 1, current_i, swings)
            if promoted is not None:
                floor["price"] = promoted["price"]
                floor["ts"]    = promoted["ts"]
                floor["idx"]   = promoted["idx"]
                floor["is_placeholder"] = is_ph

    # Walk forward.
    for i in range(start_i, n):
        # 1. Try to promote any placeholders using newly-confirmed swings up to i.
        _try_promote_placeholder('ceiling', i)
        _try_promote_placeholder('floor',   i)

        close_i = float(C[i])

        # 2. Check for wall break — closed candle close beyond a REAL wall.
        # Placeholder walls cannot be broken (they are visualization only).
        broke_ceiling = (
            ceiling["price"] is not None
            and not ceiling["is_placeholder"]
            and close_i > ceiling["price"]
        )
        broke_floor = (
            floor["price"] is not None
            and not floor["is_placeholder"]
            and close_i < floor["price"]
        )
        if not broke_ceiling and not broke_floor:
            continue

        # If both broke (extreme single-candle move): use the larger displacement.
        if broke_ceiling and broke_floor:
            if (close_i - ceiling["price"]) >= (floor["price"] - close_i):
                broke_floor = False
            else:
                broke_ceiling = False

        break_dir = 'bullish' if broke_ceiling else 'bearish'
        # 3. Classify BOS vs CHoCH against current trend.
        if trend is None:
            event = 'BOS'   # first event ever — initialize trend
        elif trend == break_dir:
            event = 'BOS'
        else:
            event = 'CHoCH'

        # 4. Leg-size filter — distance from prior opposite wall to this close.
        prior_opposite = floor["price"] if break_dir == 'bullish' else ceiling["price"]
        if prior_opposite is None:
            continue  # cannot measure — skip
        threshold_mult = BOS_ATR_MULT if event == 'BOS' else CHOCH_ATR_MULT
        if abs(close_i - prior_opposite) < threshold_mult * atr:
            continue  # leg too small — non-event

        # 5. Apply wall update rules.
        if event == 'BOS':
            if break_dir == 'bullish':
                # Floor trails up: most recent confirmed swing low INSIDE the leg
                # (leg = from prior leg_start to break candle i).
                trailed = _trail_inside_leg('floor', swings, leg_start_idx, i)
                if trailed is not None:
                    floor["price"] = trailed["price"]
                    floor["ts"]    = trailed["ts"]
                    floor["idx"]   = trailed["idx"]
                    floor["is_placeholder"] = False
                # else: floor unchanged (no qualifying interior swing).

                # Ceiling broke -> placeholder until new swing high confirms.
                ceiling["price"] = None
                ceiling["ts"]    = None
                ceiling["idx"]   = None
                ceiling["is_placeholder"] = True
            else:
                # Bearish BOS — mirrored.
                trailed = _trail_inside_leg('ceiling', swings, leg_start_idx, i)
                if trailed is not None:
                    ceiling["price"] = trailed["price"]
                    ceiling["ts"]    = trailed["ts"]
                    ceiling["idx"]   = trailed["idx"]
                    ceiling["is_placeholder"] = False
                floor["price"] = None
                floor["ts"]    = None
                floor["idx"]   = None
                floor["is_placeholder"] = True

            trend = break_dir

        else:  # CHoCH
            # The wall that just broke: placeholder.
            # The OPPOSITE wall resets to the most recent confirmed swing of the
            # just-completed trend, strictly BEFORE the swing whose break = CHoCH.
            #
            # The swing whose break = CHoCH is the wall that just broke, i.e.
            # the most recent swing high (bullish CHoCH) or swing low (bearish
            # CHoCH) before i. The opposite-wall reset target is therefore the
            # most recent swing of the OPPOSITE type strictly before that swing.
            if break_dir == 'bullish':
                # The broken ceiling was anchored at ceiling["idx"] (the lower
                # high of the bearish trend). New floor = most recent swing LOW
                # strictly before that ceiling swing.
                ref_idx = ceiling["idx"] if ceiling["idx"] is not None else i
                new_floor = _most_recent_swing_before(swings, 'low', ref_idx)
                if new_floor is not None:
                    floor["price"] = new_floor["price"]
                    floor["ts"]    = new_floor["ts"]
                    floor["idx"]   = new_floor["idx"]
                    floor["is_placeholder"] = False
                # else floor unchanged — extremely rare; held until next event.
                ceiling["price"] = None
                ceiling["ts"]    = None
                ceiling["idx"]   = None
                ceiling["is_placeholder"] = True
            else:
                # Bearish CHoCH — mirrored.
                ref_idx = floor["idx"] if floor["idx"] is not None else i
                new_ceiling = _most_recent_swing_before(swings, 'high', ref_idx)
                if new_ceiling is not None:
                    ceiling["price"] = new_ceiling["price"]
                    ceiling["ts"]    = new_ceiling["ts"]
                    ceiling["idx"]   = new_ceiling["idx"]
                    ceiling["is_placeholder"] = False
                floor["price"] = None
                floor["ts"]    = None
                floor["idx"]   = None
                floor["is_placeholder"] = True

            trend = break_dir

        last_event_type = event
        last_event_ts   = _ts_iso(df, i)
        leg_start_idx   = i
        fallback_active = False  # any real event clears fallback

        # After the event, immediately try to promote the broken side's
        # placeholder using any swings already confirmed since leg_start.
        _try_promote_placeholder('ceiling', i)
        _try_promote_placeholder('floor',   i)

    # End of walk. Final placeholder promotion attempt at last candle.
    _try_promote_placeholder('ceiling', n - 1)
    _try_promote_placeholder('floor',   n - 1)

    # Cold-start fallback: if neither wall was ever set and no event fired
    # across the entire window, fall back to window high/low.
    if is_cold_start and last_event_type is None:
        H_arr = df['High'].values.astype(float)
        L_arr = df['Low'].values.astype(float)
        rng_hi = float(H_arr.max())
        rng_lo = float(L_arr.min())
        hi_idx = int(H_arr.argmax())
        lo_idx = int(L_arr.argmin())
        ceiling = {
            "price": rng_hi, "ts": _ts_iso(df, hi_idx), "idx": hi_idx,
            "is_placeholder": False
        }
        floor = {
            "price": rng_lo, "ts": _ts_iso(df, lo_idx), "idx": lo_idx,
            "is_placeholder": False
        }
        fallback_active = True

    # Build returned state.
    new_state = {
        "trend":                  trend,
        "ceiling_price":          ceiling["price"],
        "ceiling_ts":             ceiling["ts"],
        "ceiling_is_placeholder": bool(ceiling["is_placeholder"]),
        "floor_price":            floor["price"],
        "floor_ts":               floor["ts"],
        "floor_is_placeholder":   bool(floor["is_placeholder"]),
        "last_event_type":        last_event_type,
        "last_event_ts":          last_event_ts,
        "last_scanned_ts":        _ts_iso(df, n - 1),
        "fallback_active":        bool(fallback_active),
    }
    return new_state


# --- Public API used by Phase 1 ---------------------------------------------

def update_pair(df, prior_state: Optional[Dict[str, Any]],
                pair_conf: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Phase 1 entrypoint. Cold-starts on first call (prior_state empty) or runs
    incrementally otherwise. Always returns a complete state dict.

    Caveat handled: if df has fewer than ~30 candles, return prior_state
    unchanged (or empty state). Caller decides whether to skip emission.
    """
    if df is None or len(df) < (SWING_LOOKBACK * 2 + 5):
        return prior_state or _empty_state()

    # For cold-start cap the window. After cold-start, the full df is fine
    # because incremental walk only processes new candles.
    is_cold = (
        prior_state is None
        or prior_state.get("ceiling_price") is None
        or prior_state.get("floor_price") is None
    )
    if is_cold:
        df_used = df.tail(COLDSTART_WINDOW_H1).copy().reset_index(drop=True) \
                  if len(df) > COLDSTART_WINDOW_H1 else df
        return _walk_forward(df_used, prior_state=None)
    return _walk_forward(df, prior_state=prior_state)


# --- Public API used by Phase 1 + Phase 2 (read-only) ------------------------

def compute_pd_position(price: float, walls: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a price and the wall state for a pair, return PD positioning.

    Returns dict:
      {
        "valid":         bool,      # False if either wall is placeholder OR walls invalid
        "range_high":    float,
        "range_low":     float,
        "equilibrium":   float,
        "pd_position":   float,     # 0.0 = at floor, 1.0 = at ceiling, None if invalid
        "ceiling_is_placeholder": bool,
        "floor_is_placeholder":   bool,
        "fallback_active":        bool,
        "source":        str        # human label
      }

    PD scoring is suspended (valid=False) if EITHER wall is a placeholder.
    Phase 2 callers should treat valid=False as "no PD score this scan".
    """
    if not walls:
        return {"valid": False, "source": "no_state",
                "ceiling_is_placeholder": True, "floor_is_placeholder": True,
                "fallback_active": False, "pd_position": None,
                "range_high": 0.0, "range_low": 0.0, "equilibrium": 0.0}

    ceiling = walls.get("ceiling_price")
    floor   = walls.get("floor_price")
    cph = bool(walls.get("ceiling_is_placeholder", True))
    fph = bool(walls.get("floor_is_placeholder", True))
    fb  = bool(walls.get("fallback_active", False))

    base = {
        "ceiling_is_placeholder": cph,
        "floor_is_placeholder":   fph,
        "fallback_active":        fb,
        "range_high": float(ceiling) if ceiling is not None else 0.0,
        "range_low":  float(floor)   if floor   is not None else 0.0,
        "equilibrium": 0.0,
        "pd_position": None,
    }

    if ceiling is None or floor is None:
        base.update({"valid": False, "source": "incomplete_walls"})
        return base

    if ceiling <= floor:
        base.update({"valid": False, "source": "degenerate_range"})
        return base

    if cph or fph:
        # Wall geometry exists for visualization but PD scoring is suspended.
        base["equilibrium"] = (float(ceiling) + float(floor)) / 2.0
        base.update({"valid": False, "source": "placeholder_active"})
        return base

    eq = (float(ceiling) + float(floor)) / 2.0
    width = float(ceiling) - float(floor)
    pos = (float(price) - float(floor)) / width if width > 0 else None
    src = "fallback_window" if fb else "structural"
    return {
        "valid": True,
        "range_high": float(ceiling),
        "range_low":  float(floor),
        "equilibrium": eq,
        "pd_position": pos,
        "ceiling_is_placeholder": cph,
        "floor_is_placeholder":   fph,
        "fallback_active":        fb,
        "source": src,
    }
