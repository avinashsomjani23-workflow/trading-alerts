"""
Dealing Range — single source of truth for H1 dealing range walls AND
BOS / CHoCH detection.

Concepts (plain English):

A dealing range has two walls: a CEILING (upper) and a FLOOR (lower). Walls
are anchored to CONFIRMED swing highs / lows on H1 (lookback=3 strict).

Two structural events are detected:
  - BOS   (Break of Structure)   — close past a wall in the trend direction.
  - CHoCH (Change of Character) — close past a confirmed swing AGAINST trend,
                                    with a premium/discount-zone reversal.

CHoCH has two tiers:
  - Major: broken pivot is a lookback=3 swing (or the opposite wall when no
    internal lookback=3 swing exists). Flips trend.
  - Minor: broken pivot is a lookback=2 swing inside the trend. Does NOT flip
    trend; does NOT move walls. Informational weakening flag only.

Wall update rule (LOCKED): walls move ONLY when a wall is broken.
  - BOS: trend-direction wall trails to break-candle extreme (tentative
    until promoted); opposite wall trails to deepest pullback inside the
    just-completed leg.
  - Major CHoCH AT WALL (no internal pivot existed; close past opposite wall):
    broken wall → tentative; opposite wall stays put (it's the prior trend
    anchor = new trend's starting extreme). NO history search.
    EXCEPTION — chop CHoCH-at-wall: if chop=True on this event, the BOS
    that set the opposite wall just before was a fakeout. The opposite
    wall also resets to tentative at the break-candle extreme. Both walls
    re-anchor inside the new leg via _try_promote_placeholder /
    _refresh_tentative.
  - Major CHoCH on internal HL/LH: walls do NOT move; trend flips.
  - Minor CHoCH: walls do NOT move; trend does NOT flip.

Premium / discount gate (LOCKED, CHoCH only): the reversal high (for down
CHoCH in an uptrend) must lie in the top 25% of the dealing range; the
reversal low (for up CHoCH) must lie in the bottom 25%. Without this, an
"internal" close past a swing in mid-range is just noise, not a CHoCH.

Phase 1 calls:
    dealing_range.update_pair(df_h1, prior_state, pair_conf)

Cold-starts on first call, runs incrementally otherwise. Returns the new
state for that pair (walls + last_event metadata + event ring).

Phase 2 reads `state/structure_state.json` and consumes:
    dealing_range.compute_pd_position(price, walls)

Phase 2 NEVER writes state. One writer (Phase 1), many readers.

Design decisions (locked):
  - SWING_LOOKBACK = 3 for walls, BOS, Major CHoCH.
  - MINOR_SWING_LOOKBACK = 2 for Minor CHoCH detection only.
  - BOS displacement >= 0.40 × H1 ATR past broken wall.
  - CHoCH displacement >= 0.60 × H1 ATR past broken pivot (Major and Minor
    use the same threshold).
  - Premium / discount threshold = 25% of dealing range.
  - Tentative (placeholder) walls give geometry only — events do NOT fire
    on placeholder-wall break.
  - Event ring: last 20 qualified events kept on state for Phase 2 readers
    (BOS sequence count; zone invalidation).
  - Chop flag: CHoCH within 5 candles of prior event marks last_event_chop.
  - Atomic state writes (temp + rename).
"""

import json
import os
from typing import Optional, Tuple, List, Dict, Any

# Optional structure event logger. Logging failures must never break the
# trading flow — every call site below wraps in try/except.
try:
    import event_logger as _event_logger
except Exception:
    _event_logger = None

# --- Tunables (locked) -------------------------------------------------------

# Two parallel swing pools.
#   SWING_LOOKBACK (3) is the "Major" / wall-grade swing. Used for walls,
#   BOS detection, and Major CHoCH detection.
#   MINOR_SWING_LOOKBACK (2) is used ONLY for Minor CHoCH detection. A Minor
#   CHoCH break is informational (weakening flag) — it does NOT flip trend
#   and does NOT move walls.
SWING_LOOKBACK       = 3
MINOR_SWING_LOOKBACK = 2

# Leg-size threshold = displacement of the break candle's CLOSE PAST the wall
# (or pivot) it just broke. Major and Minor CHoCH share the same threshold
# (0.60× ATR) — the depth of the broken swing differs, but a noise filter
# at the candle level is the same problem.
BOS_ATR_MULT   = 0.4
CHOCH_ATR_MULT = 0.6

# Premium / discount thresholds for the CHoCH zone gate. A CHoCH is valid
# only if the reversal high (down CHoCH) sits in the top 25% of the dealing
# range, or the reversal low (up CHoCH) sits in the bottom 25%.
PREMIUM_PCT  = 0.75
DISCOUNT_PCT = 0.25

# Cold-start window — number of most recent H1 candles to walk forward
# looking for events. If no events fire across this whole window, fallback
# walls are computed from FALLBACK_WINDOW_H1 only (a tighter recent window).
COLDSTART_WINDOW_H1 = 150
FALLBACK_WINDOW_H1  = 72   # last 72 H1 candles ≈ 3 trading days

# Chop flag: a CHoCH within this many candles of the previous event tags
# the event as possible-chop in audit + email.
CHOP_LOOKBACK_CANDLES = 5

# Event ring — last N qualified events kept on state for downstream readers
# (Phase 2 BOS-sequence count, zone invalidation). Cap chosen well above
# any realistic per-trend BOS run (caution thresholds are 3–5).
EVENT_RING_MAX = 20

# State file path. Lives in a dedicated directory outside any purge scope.
STATE_DIR  = "state"
STATE_PATH = os.path.join(STATE_DIR, "structure_state.json")

# Non-persisted key attached to update_pair()'s returned state. Carries the
# placeholder-not-promoted diagnostic for downstream scan logging. MUST be
# stripped before save_state() so structure_state.json stays clean.
PLACEHOLDER_DIAG_KEY = "_diagnostic_last_walk"


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
      - a swing high if H[i] is STRICTLY GREATER than every other high in
        the window [i-lookback, i+lookback].
      - a swing low  if L[i] is STRICTLY LESS than every other low in
        the window [i-lookback, i+lookback].

    Strict comparison: equal highs / equal lows do NOT register as swings.
    A flat top / flat bottom across the window correctly produces no swing.

    Returns list sorted by idx, each entry:
      {'type': 'high'|'low', 'idx': i, 'price': float, 'ts': iso}
    """
    if df is None or len(df) < lookback * 2 + 1:
        return []
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    n = len(df)
    out = []
    for i in range(lookback, n - lookback):
        # Compare against every neighbour in window EXCLUDING i itself.
        wh_left  = H[i - lookback: i]
        wh_right = H[i + 1: i + lookback + 1]
        wl_left  = L[i - lookback: i]
        wl_right = L[i + 1: i + lookback + 1]
        # Use Python max/min on numpy slices — fine for small windows.
        max_neighbour_h = max(max(wh_left), max(wh_right)) if len(wh_left) and len(wh_right) else None
        min_neighbour_l = min(min(wl_left), min(wl_right)) if len(wl_left) and len(wl_right) else None
        if max_neighbour_h is not None and H[i] > max_neighbour_h:
            out.append({'type': 'high', 'idx': i, 'price': float(H[i]), 'ts': _ts_iso(df, i)})
        if min_neighbour_l is not None and L[i] < min_neighbour_l:
            out.append({'type': 'low',  'idx': i, 'price': float(L[i]), 'ts': _ts_iso(df, i)})
    out.sort(key=lambda s: s['idx'])
    return out


def detect_minor_swings(df) -> List[Dict[str, Any]]:
    """Lookback=2 swing pool. Used ONLY for Minor CHoCH detection.

    Same shape and rules as detect_swings, but with a tighter window. A
    lookback=2 swing requires the 2 candles on each side to be strictly lower
    (for highs) or higher (for lows). These swings appear more frequently and
    represent shallower, internal structure shifts.
    """
    return detect_swings(df, lookback=MINOR_SWING_LOOKBACK)


def _most_recent_swing_in_window(swings: List[Dict[str, Any]], swing_type: str,
                                  start_idx: int, end_idx: int) -> Optional[Dict[str, Any]]:
    """Most recent swing of the given type with start_idx <= idx <= end_idx.

    Used by CHoCH detection to find the most recent confirmed pivot inside
    the current trend (between leg_start_idx and the current candle).
    Returns the swing dict, or None if none qualify.
    """
    if start_idx > end_idx:
        return None
    matches = [s for s in swings
               if s['type'] == swing_type and start_idx <= s['idx'] <= end_idx]
    if not matches:
        return None
    matches.sort(key=lambda s: s['idx'], reverse=True)
    return matches[0]


def _pick_choch_pivot(swings_lb3: List[Dict[str, Any]],
                      swings_lb2: List[Dict[str, Any]],
                      trend: str,
                      leg_start_idx: int,
                      current_idx: int,
                      ceiling: Dict[str, Any],
                      floor: Dict[str, Any]
                      ) -> Optional[Tuple[Dict[str, Any], str, bool]]:
    """Pick the pivot whose break would constitute a CHoCH at this candle.

    Returns (pivot_dict, tier, at_wall) or None if no candidate.

    Picks the MOST RECENT confirmed pivot of the relevant type inside the
    current trend. lookback=3 swings are a SUBSET of lookback=2 swings, so
    we use the lb2 pool for "most recent" and tag tier=Major if that swing
    also appears in lb3, else Minor.

    Why most-recent: in an uptrend, sequential HLs rise — the most recent HL
    is the highest. Price closing below the most recent HL IS a structural
    break; an older lb3 HL further down is still un-broken at that close.
    Picking the older lb3 swing would miss the real break.

    Fallback: if no internal pivot exists, the opposite wall (floor in
    uptrend, ceiling in downtrend) is the pivot — break of a wall is Major
    CHoCH at_wall=True. Only valid if the wall is CONFIRMED (not placeholder).

    Search window: (leg_start_idx, current_idx-1] — strictly inside the
    current trend and strictly before the candle being tested.
    """
    if trend == 'bullish':
        target_type = 'low'        # Down CHoCH breaks an HL (swing low).
        wall = floor
    elif trend == 'bearish':
        target_type = 'high'       # Up CHoCH breaks an LH (swing high).
        wall = ceiling
    else:
        return None

    lo, hi = leg_start_idx + 1, current_idx - 1
    if lo > hi:
        return None

    pivot = _most_recent_swing_in_window(swings_lb2, target_type, lo, hi)
    if pivot is not None:
        # Tier = Major if this swing is also in the lb3 pool, else Minor.
        is_lb3 = any(s['idx'] == pivot['idx'] and s['type'] == pivot['type']
                     for s in swings_lb3)
        return (pivot, 'Major' if is_lb3 else 'Minor', False)

    # Fall back to the opposite wall — but only if it's confirmed.
    if wall.get('price') is None or wall.get('is_placeholder', True):
        return None
    wall_pivot = {'idx': wall.get('idx'), 'price': wall['price'], 'ts': wall.get('ts')}
    return (wall_pivot, 'Major', True)


def _resolve_impulse_start(event_kind: str, direction: str,
                            swings_lb3: List[Dict[str, Any]],
                            df, broken_pivot_idx: int, break_idx: int,
                            leg_start_idx: int) -> Optional[Dict[str, Any]]:
    """Find the impulse-start anchor for an event so the OB-finder downstream
    knows the leg to walk back through.

    BOS: impulse_start = most recent confirmed lookback=3 swing of the opposite
         type between leg_start_idx and break_idx (the latest HL before a
         bullish BOS, the latest LH before a bearish BOS).

    CHoCH: impulse_start = the index of the reversal extreme between the
         broken pivot and the break candle (the highest High for a down CHoCH,
         the lowest Low for an up CHoCH). Always exists by construction (the
         premium-zone gate already implied a non-empty window).

    Returns dict with idx/price/ts, or None if not resolvable.
    """
    if event_kind == 'BOS':
        # Opposite-type swing for the impulse leg's anchor.
        target_type = 'low' if direction == 'bullish' else 'high'
        lo, hi = leg_start_idx + 1, break_idx - 1
        if lo > hi:
            return None
        candidates = [s for s in swings_lb3
                      if s['type'] == target_type and lo <= s['idx'] <= hi]
        if not candidates:
            return None
        candidates.sort(key=lambda s: s['idx'], reverse=True)
        s = candidates[0]
        return {'idx': s['idx'], 'price': s['price'], 'ts': s['ts']}

    # CHoCH: reversal extreme between broken pivot idx and break_idx-1.
    lo, hi = broken_pivot_idx, break_idx - 1
    if lo > hi or lo < 0 or hi >= len(df):
        return None
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    if direction == 'bearish':       # uptrend reversal — find reversal HIGH
        slc = H[lo: hi + 1]
        rng_idx = lo + int(slc.argmax())
        return {'idx': rng_idx, 'price': float(H[rng_idx]), 'ts': _ts_iso(df, rng_idx)}
    else:                            # downtrend reversal — find reversal LOW
        slc = L[lo: hi + 1]
        rng_idx = lo + int(slc.argmin())
        return {'idx': rng_idx, 'price': float(L[rng_idx]), 'ts': _ts_iso(df, rng_idx)}


def _premium_zone_satisfied(direction_of_choch: str,
                             df,
                             pivot_idx: int,
                             break_idx: int,
                             ceiling_price: Optional[float],
                             floor_price: Optional[float]
                             ) -> Tuple[bool, Optional[float]]:
    """Verify the reversal happened from premium (down CHoCH) or discount (up CHoCH).

    For a down CHoCH (uptrend reversing): the highest High over
        [pivot_idx .. break_idx - 1]
    must satisfy:  reversal_high >= floor + 0.75 * (ceiling - floor)
        (i.e. reversal high lies in the top 25% of the dealing range).

    For an up CHoCH (downtrend reversing): mirror with the lowest Low and the
    bottom 25%.

    Returns (gate_passed, reversal_pct) where reversal_pct is the fractional
    position of the reversal extreme in the dealing range (None if walls
    degenerate, in which case gate FAILS CLOSED).
    """
    if (ceiling_price is None or floor_price is None
            or ceiling_price <= floor_price):
        return False, None
    if pivot_idx is None or break_idx is None or pivot_idx >= break_idx:
        return False, None

    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    lo, hi = pivot_idx, break_idx - 1
    if lo < 0 or hi >= len(df) or lo > hi:
        return False, None

    range_size = ceiling_price - floor_price
    if direction_of_choch == 'bearish':       # down CHoCH (uptrend reversal)
        reversal_high = float(H[lo: hi + 1].max())
        pct = (reversal_high - floor_price) / range_size
        threshold = floor_price + PREMIUM_PCT * range_size
        return reversal_high >= threshold, pct
    elif direction_of_choch == 'bullish':     # up CHoCH (downtrend reversal)
        reversal_low = float(L[lo: hi + 1].min())
        pct = (reversal_low - floor_price) / range_size
        threshold = floor_price + DISCOUNT_PCT * range_size
        return reversal_low <= threshold, pct
    return False, None


# --- Core: walk forward and build state -------------------------------------

def _empty_state() -> Dict[str, Any]:
    return {
        "trend": None,                        # 'bullish' | 'bearish' | None
        "ceiling_price": None,
        "ceiling_ts": None,
        "ceiling_is_placeholder": True,       # True = tentative wall (rolling extreme)
        "floor_price": None,
        "floor_ts": None,
        "floor_is_placeholder": True,         # True = tentative wall (rolling extreme)
        "last_event_type": None,              # 'BOS' | 'CHoCH' | None
        "last_event_tier": None,              # 'Major' | 'Minor' | None  (Major for BOS too)
        "last_event_direction": None,         # 'bullish' | 'bearish' | None
        "last_event_ts": None,
        "last_event_idx_iso": None,           # ISO ts of the break candle (used for chop gap calc)
        "last_event_chop": False,             # True if this CHoCH fired within CHOP_LOOKBACK_CANDLES of prior event
        "last_scanned_ts": None,
        "fallback_active": False,
        "events": [],                         # ring of last EVENT_RING_MAX qualified events
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


def _walk_forward(df, prior_state: Optional[Dict[str, Any]] = None,
                  pair_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Walk forward through df starting from either:
      - candle 0 (cold start, prior_state is empty/None), or
      - the candle just after prior_state['last_scanned_ts'] (incremental).

    Detects BOS, Major CHoCH (lookback=3 internal pivot or wall break), and
    Minor CHoCH (lookback=2 internal pivot break) per the locked rules.
    Updates walls only when a wall is broken (BOS or Major CHoCH-at-wall).
    Internal-pivot CHoCH does not move walls; Minor CHoCH does not flip trend.

    Cold start fallback: if no event fires across the entire window, walls
    fall back to window high/low and fallback_active = True.
    """
    n = len(df) if df is not None else 0
    if n == 0:
        return prior_state or _empty_state()

    atr = _compute_atr(df)
    if atr is None or atr <= 0:
        return prior_state or _empty_state()

    # Two parallel swing pools (computed once per walk).
    swings_lb3 = detect_swings(df, lookback=SWING_LOOKBACK)
    swings_lb2 = detect_swings(df, lookback=MINOR_SWING_LOOKBACK)

    state = prior_state if prior_state else _empty_state()
    is_cold_start = (
        prior_state is None
        or prior_state.get("ceiling_price") is None
        or prior_state.get("floor_price") is None
    )

    start_i = 0
    if not is_cold_start and prior_state.get("last_scanned_ts"):
        last_ts = prior_state["last_scanned_ts"]
        for i in range(n):
            ts = _ts_iso(df, i)
            if ts and ts > last_ts:
                start_i = i
                break
        else:
            new_state = dict(state)
            new_state["last_scanned_ts"] = _ts_iso(df, n - 1)
            return new_state

    C = df['Close'].values.astype(float)

    trend = state.get("trend")
    ceiling = {
        "price": state.get("ceiling_price"),
        "ts":    state.get("ceiling_ts"),
        "idx":   None,
        "is_placeholder": state.get("ceiling_is_placeholder", True),
    }
    floor = {
        "price": state.get("floor_price"),
        "ts":    state.get("floor_ts"),
        "idx":   None,
        "is_placeholder": state.get("floor_is_placeholder", True),
    }

    last_event_type      = state.get("last_event_type")
    last_event_tier      = state.get("last_event_tier")
    last_event_direction = state.get("last_event_direction")
    last_event_ts        = state.get("last_event_ts")
    last_event_idx_iso   = state.get("last_event_idx_iso")
    last_event_chop      = bool(state.get("last_event_chop", False))
    leg_start_idx        = 0
    fallback_active      = state.get("fallback_active", False)
    events_ring          = list(state.get("events", []))

    last_event_local_idx: Optional[int] = None

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

    if last_event_idx_iso:
        last_event_local_idx = _idx_from_ts(last_event_idx_iso)

    # Promote a placeholder wall if a confirmed swing now exists in
    # (leg_start_idx, current_i].
    def _try_promote_placeholder(side: str, current_i: int):
        if side == 'ceiling':
            if not ceiling["is_placeholder"]:
                return
            promoted, is_ph = _resolve_placeholder('ceiling', df, leg_start_idx + 1, current_i, swings_lb3)
            if promoted is not None:
                ceiling["price"] = promoted["price"]
                ceiling["ts"]    = promoted["ts"]
                ceiling["idx"]   = promoted["idx"]
                ceiling["is_placeholder"] = is_ph
        else:
            if not floor["is_placeholder"]:
                return
            promoted, is_ph = _resolve_placeholder('floor', df, leg_start_idx + 1, current_i, swings_lb3)
            if promoted is not None:
                floor["price"] = promoted["price"]
                floor["ts"]    = promoted["ts"]
                floor["idx"]   = promoted["idx"]
                floor["is_placeholder"] = is_ph

    # Refresh a still-tentative wall to the rolling extreme since leg_start.
    # Geometry only — events do NOT fire on placeholder-wall break.
    def _refresh_tentative(side: str, current_i: int):
        if side == 'ceiling':
            if not ceiling["is_placeholder"]:
                return
            lo, hi = leg_start_idx + 1, current_i
            if lo > hi:
                return
            H_arr = df['High'].values.astype(float)
            rng_idx = lo + int(H_arr[lo: hi + 1].argmax())
            ceiling["price"] = float(H_arr[rng_idx])
            ceiling["ts"]    = _ts_iso(df, rng_idx)
            ceiling["idx"]   = rng_idx
        else:
            if not floor["is_placeholder"]:
                return
            lo, hi = leg_start_idx + 1, current_i
            if lo > hi:
                return
            L_arr = df['Low'].values.astype(float)
            rng_idx = lo + int(L_arr[lo: hi + 1].argmin())
            floor["price"] = float(L_arr[rng_idx])
            floor["ts"]    = _ts_iso(df, rng_idx)
            floor["idx"]   = rng_idx

    def _log_safe(payload: Dict[str, Any]):
        if _event_logger is None or not pair_name:
            return
        try:
            _event_logger.log_event(payload)
        except Exception:
            pass

    def _ring_append(event_dict: Dict[str, Any]):
        events_ring.append(event_dict)
        if len(events_ring) > EVENT_RING_MAX:
            del events_ring[0: len(events_ring) - EVENT_RING_MAX]

    # ------------------------------------------------------------------
    # Walk forward.
    # ------------------------------------------------------------------
    for i in range(start_i, n):
        # 1. Promote / refresh placeholders (geometry maintenance).
        _try_promote_placeholder('ceiling', i)
        _try_promote_placeholder('floor',   i)
        _refresh_tentative('ceiling', i)
        _refresh_tentative('floor',   i)

        close_i = float(C[i])
        event_kind = None       # 'BOS' | 'CHoCH' | None
        event_tier = None       # 'Major' | 'Minor' | None  (Major for BOS)
        event_direction = None  # 'bullish' | 'bearish'
        event_at_wall = False
        event_pivot = None      # the pivot/wall that broke — dict with idx/price/ts
        event_displacement = 0.0
        event_reversal_pct = None  # CHoCH only

        # 2. BOS check — ONLY on confirmed walls.
        # In trend direction:
        #   bullish trend -> close > confirmed ceiling
        #   bearish trend -> close < confirmed floor
        # Cold-start (trend is None): a confirmed wall break in either direction
        # initialises the trend as a BOS.
        if (trend in ('bullish', None)
                and ceiling["price"] is not None
                and not ceiling["is_placeholder"]
                and close_i > ceiling["price"]):
            disp = close_i - ceiling["price"]
            if disp >= BOS_ATR_MULT * atr:
                event_kind = 'BOS'
                event_tier = 'Major'
                event_direction = 'bullish'
                event_at_wall = True
                event_pivot = {
                    'idx': ceiling["idx"], 'price': ceiling["price"], 'ts': ceiling["ts"]
                }
                event_displacement = disp
            else:
                _log_safe({
                    'candle_ts': _ts_iso(df, i), 'pair': pair_name, 'timeframe': 'H1',
                    'event_kind': 'BREAK_REJECTED', 'reject_reason': 'atr_threshold',
                    'attempted_event': 'BOS', 'direction': 'bullish',
                    'swing_price': float(ceiling["price"]), 'swing_ts': ceiling["ts"],
                    'close_price': float(close_i), 'displacement': float(disp),
                    'displacement_atr': float(disp / atr),
                    'threshold_atr': float(BOS_ATR_MULT), 'atr': float(atr),
                })

        elif (trend in ('bearish', None)
                and floor["price"] is not None
                and not floor["is_placeholder"]
                and close_i < floor["price"]):
            disp = floor["price"] - close_i
            if disp >= BOS_ATR_MULT * atr:
                event_kind = 'BOS'
                event_tier = 'Major'
                event_direction = 'bearish'
                event_at_wall = True
                event_pivot = {
                    'idx': floor["idx"], 'price': floor["price"], 'ts': floor["ts"]
                }
                event_displacement = disp
            else:
                _log_safe({
                    'candle_ts': _ts_iso(df, i), 'pair': pair_name, 'timeframe': 'H1',
                    'event_kind': 'BREAK_REJECTED', 'reject_reason': 'atr_threshold',
                    'attempted_event': 'BOS', 'direction': 'bearish',
                    'swing_price': float(floor["price"]), 'swing_ts': floor["ts"],
                    'close_price': float(close_i), 'displacement': float(disp),
                    'displacement_atr': float(disp / atr),
                    'threshold_atr': float(BOS_ATR_MULT), 'atr': float(atr),
                })

        # 3. CHoCH check — only when there's an established trend AND no BOS already
        # fired this candle. Direction is AGAINST trend.
        if event_kind is None and trend in ('bullish', 'bearish'):
            choch_dir = 'bearish' if trend == 'bullish' else 'bullish'
            picked = _pick_choch_pivot(swings_lb3, swings_lb2, trend,
                                       leg_start_idx, i, ceiling, floor)
            if picked is not None:
                pivot, tier, at_wall = picked
                pivot_price = pivot['price']
                # Direction-aware break test:
                if choch_dir == 'bearish':       # uptrend reversing
                    broke = close_i < pivot_price
                    disp = pivot_price - close_i
                else:                            # downtrend reversing
                    broke = close_i > pivot_price
                    disp = close_i - pivot_price

                if broke:
                    if disp >= CHOCH_ATR_MULT * atr:
                        # Premium / discount zone gate.
                        gate_passed, reversal_pct = _premium_zone_satisfied(
                            choch_dir, df, pivot['idx'], i,
                            ceiling.get("price"), floor.get("price")
                        )
                        if gate_passed:
                            event_kind = 'CHoCH'
                            event_tier = tier
                            event_direction = choch_dir
                            event_at_wall = at_wall
                            event_pivot = pivot
                            event_displacement = disp
                            event_reversal_pct = reversal_pct
                        else:
                            _log_safe({
                                'candle_ts': _ts_iso(df, i), 'pair': pair_name, 'timeframe': 'H1',
                                'event_kind': 'BREAK_REJECTED',
                                'reject_reason': 'not_in_premium_zone',
                                'attempted_event': f'CHoCH-{tier}',
                                'direction': choch_dir,
                                'swing_price': float(pivot_price), 'swing_ts': pivot.get('ts'),
                                'close_price': float(close_i), 'displacement': float(disp),
                                'displacement_atr': float(disp / atr),
                                'reversal_pct': reversal_pct,
                                'threshold_pct': PREMIUM_PCT if choch_dir == 'bearish' else DISCOUNT_PCT,
                                'broken_was_wall': bool(at_wall),
                            })
                    else:
                        _log_safe({
                            'candle_ts': _ts_iso(df, i), 'pair': pair_name, 'timeframe': 'H1',
                            'event_kind': 'BREAK_REJECTED', 'reject_reason': 'atr_threshold',
                            'attempted_event': f'CHoCH-{tier}',
                            'direction': choch_dir,
                            'swing_price': float(pivot_price), 'swing_ts': pivot.get('ts'),
                            'close_price': float(close_i), 'displacement': float(disp),
                            'displacement_atr': float(disp / atr),
                            'threshold_atr': float(CHOCH_ATR_MULT), 'atr': float(atr),
                            'broken_was_wall': bool(at_wall),
                        })

        if event_kind is None:
            continue

        # 4. Chop flag — CHoCH only, gap from prior event in candles.
        chop_this_event = False
        if event_kind == 'CHoCH' and last_event_local_idx is not None:
            gap = i - last_event_local_idx
            if 0 < gap <= CHOP_LOOKBACK_CANDLES:
                chop_this_event = True

        # 5. Apply wall update rules per event type.
        broken_swing_price = float(event_pivot['price'])
        broken_swing_ts    = event_pivot.get('ts')
        broken_was_wall    = event_at_wall

        if event_kind == 'BOS':
            # Trend-direction wall just broke -> tentative at break-candle extreme.
            # Opposite wall trails to deepest pullback inside just-completed leg.
            if event_direction == 'bullish':
                trailed = _trail_inside_leg('floor', swings_lb3, leg_start_idx, i)
                if trailed is not None:
                    floor["price"] = trailed["price"]
                    floor["ts"]    = trailed["ts"]
                    floor["idx"]   = trailed["idx"]
                    floor["is_placeholder"] = False
                ceiling["price"] = float(df['High'].iloc[i])
                ceiling["ts"]    = _ts_iso(df, i)
                ceiling["idx"]   = i
                ceiling["is_placeholder"] = True
            else:
                trailed = _trail_inside_leg('ceiling', swings_lb3, leg_start_idx, i)
                if trailed is not None:
                    ceiling["price"] = trailed["price"]
                    ceiling["ts"]    = trailed["ts"]
                    ceiling["idx"]   = trailed["idx"]
                    ceiling["is_placeholder"] = False
                floor["price"] = float(df['Low'].iloc[i])
                floor["ts"]    = _ts_iso(df, i)
                floor["idx"]   = i
                floor["is_placeholder"] = True
            trend = event_direction

        elif event_kind == 'CHoCH' and event_tier == 'Major':
            if event_at_wall:
                # The wall we just closed past becomes tentative.
                # Opposite wall normally STAYS — it's the prior trend anchor
                # and the new trend's starting extreme.
                # EXCEPTION: chop CHoCH-at-wall (chop_this_event=True). The
                # BOS that set the opposite wall just before was a fakeout.
                # Reset BOTH walls so they re-anchor inside the new leg
                # (avoids inheriting spike-derived walls like USDJPY 160.7).
                if event_direction == 'bearish':       # down CHoCH at floor wall
                    floor["price"] = float(df['Low'].iloc[i])
                    floor["ts"]    = _ts_iso(df, i)
                    floor["idx"]   = i
                    floor["is_placeholder"] = True
                    if chop_this_event:
                        ceiling["price"] = float(df['High'].iloc[i])
                        ceiling["ts"]    = _ts_iso(df, i)
                        ceiling["idx"]   = i
                        ceiling["is_placeholder"] = True
                else:                                  # up CHoCH at ceiling wall
                    ceiling["price"] = float(df['High'].iloc[i])
                    ceiling["ts"]    = _ts_iso(df, i)
                    ceiling["idx"]   = i
                    ceiling["is_placeholder"] = True
                    if chop_this_event:
                        floor["price"] = float(df['Low'].iloc[i])
                        floor["ts"]    = _ts_iso(df, i)
                        floor["idx"]   = i
                        floor["is_placeholder"] = True
            # else: internal-pivot Major CHoCH -> walls do NOT change.
            trend = event_direction

        else:  # Minor CHoCH
            # Walls do NOT change. Trend does NOT flip. Informational only.
            pass

        # 6. Resolve impulse-start anchor (used downstream by smc_radar to
        # build the OB by walking back through the impulse leg).
        impulse_start = _resolve_impulse_start(
            event_kind, event_direction, swings_lb3, df,
            event_pivot.get('idx') if event_pivot else None, i, leg_start_idx
        )
        impulse_start_ts    = impulse_start.get('ts') if impulse_start else None
        impulse_start_price = (float(impulse_start['price'])
                                if impulse_start else None)

        # 7. Commit event metadata + ring append.
        last_event_type      = event_kind
        last_event_tier      = event_tier
        last_event_direction = event_direction
        last_event_ts        = _ts_iso(df, i)
        last_event_idx_iso   = _ts_iso(df, i)
        last_event_local_idx = i
        last_event_chop      = chop_this_event
        leg_start_idx        = i
        fallback_active      = False

        _ring_append({
            'type':                 event_kind,
            'tier':                 event_tier,
            'direction':            event_direction,
            'candle_ts':            _ts_iso(df, i),
            'broken_swing_price':   broken_swing_price,
            'broken_swing_ts':      broken_swing_ts,
            'broken_was_wall':      bool(broken_was_wall),
            'displacement_atr':     float(event_displacement / atr),
            'chop':                 bool(chop_this_event),
            'reversal_pct':         (float(event_reversal_pct)
                                      if event_reversal_pct is not None else None),
            'impulse_start_ts':     impulse_start_ts,
            'impulse_start_price':  impulse_start_price,
            'trend_after':          trend,
        })

        # Log qualified event for offline review.
        cb = abs(float(df['Close'].iloc[i]) - float(df['Open'].iloc[i]))
        cr = float(df['High'].iloc[i]) - float(df['Low'].iloc[i])
        _log_safe({
            'candle_ts':       _ts_iso(df, i),
            'pair':            pair_name,
            'timeframe':       'H1',
            'event_kind':      event_kind,
            'tier':            event_tier,
            'direction':       event_direction,
            'swing_price':     broken_swing_price,
            'swing_ts':        broken_swing_ts,
            'close_price':     float(close_i),
            'displacement':    float(event_displacement),
            'displacement_atr': float(event_displacement / atr),
            'threshold_atr':   float(BOS_ATR_MULT if event_kind == 'BOS' else CHOCH_ATR_MULT),
            'atr':             float(atr),
            'broken_was_wall': bool(broken_was_wall),
            'reversal_pct':    (float(event_reversal_pct)
                                 if event_reversal_pct is not None else None),
            'chop_flag':       bool(chop_this_event),
            'candle_body':     float(cb),
            'candle_range':    float(cr),
            'body_range_pct':  float(cb / cr) if cr > 0 else 0.0,
            'trend_after':     trend,
        })

        # After the event, try to promote / refresh again — the new tentative
        # wall (if any) needs initial geometry.
        _try_promote_placeholder('ceiling', i)
        _try_promote_placeholder('floor',   i)
        _refresh_tentative('ceiling', i)
        _refresh_tentative('floor',   i)

    # End of walk: final promote / refresh pass.
    _try_promote_placeholder('ceiling', n - 1)
    _try_promote_placeholder('floor',   n - 1)
    _refresh_tentative('ceiling', n - 1)
    _refresh_tentative('floor',   n - 1)

    # --- Placeholder-not-promoted diagnostic ----------------------------------
    # For any wall still tentative at end-of-walk, explain WHY no lookback=3
    # swing in (leg_start_idx, n-1] was promotable. Three outcomes per side:
    #   - anchored          : wall is confirmed, no diagnostic needed
    #   - no_interior_window: leg_start_idx == n-1 (just had an event, no room)
    #   - swings_present    : N lb3 swings existed but none was picked (bug path)
    #   - blocked_by_idx    : the highest H / lowest L in the window is not
    #                          a confirmed swing because a neighbour candle
    #                          violates the strict-greater/less rule. Names
    #                          the blocking candle.
    def _diagnose_placeholder(side: str) -> Dict[str, Any]:
        wall = ceiling if side == 'ceiling' else floor
        if not wall["is_placeholder"]:
            return {
                "side": side,
                "is_placeholder": False,
                "reason": "anchored",
            }
        current_i = n - 1
        lo, hi = leg_start_idx + 1, current_i
        out: Dict[str, Any] = {
            "side": side,
            "is_placeholder": True,
            "leg_start_idx": int(leg_start_idx),
            "leg_start_ts":  _ts_iso(df, leg_start_idx) if 0 <= leg_start_idx < n else None,
            "current_idx":   int(current_i),
            "current_ts":    _ts_iso(df, current_i),
            "wall_price":    float(wall["price"]) if wall["price"] is not None else None,
            "wall_ts":       wall["ts"],
        }
        if lo > hi:
            out["reason"] = "no_interior_window"
            return out

        target_type = 'high' if side == 'ceiling' else 'low'
        lb3_in_window = [s for s in swings_lb3
                         if s['type'] == target_type and lo <= s['idx'] <= hi]
        out["lb3_swings_in_window"] = len(lb3_in_window)

        if lb3_in_window:
            # Should have been promoted — record the candidate the resolver
            # WOULD have picked so we can compare with what's on state.
            if side == 'ceiling':
                best = max(lb3_in_window, key=lambda s: s['price'])
            else:
                best = min(lb3_in_window, key=lambda s: s['price'])
            out["reason"] = "swings_present_but_not_promoted"
            out["candidate_idx"]   = int(best['idx'])
            out["candidate_ts"]    = best['ts']
            out["candidate_price"] = float(best['price'])
            return out

        # No lb3 swing exists in window. Find the candle that holds the
        # rolling extreme and identify what neighbour blocks it from being
        # a confirmed swing.
        H_arr = df['High'].values.astype(float)
        L_arr = df['Low'].values.astype(float)
        arr = H_arr if side == 'ceiling' else L_arr
        if side == 'ceiling':
            ext_idx = lo + int(arr[lo: hi + 1].argmax())
        else:
            ext_idx = lo + int(arr[lo: hi + 1].argmin())
        ext_price = float(arr[ext_idx])
        out["rolling_extreme_idx"]   = int(ext_idx)
        out["rolling_extreme_ts"]    = _ts_iso(df, ext_idx)
        out["rolling_extreme_price"] = ext_price
        out["candles_after_extreme"] = int(current_i - ext_idx)

        # Right-edge check: does the extreme even have SWING_LOOKBACK candles
        # to its right? If not, it cannot be confirmed yet regardless of values.
        right_edge_needed = ext_idx + SWING_LOOKBACK
        if right_edge_needed >= n:
            out["reason"] = "right_edge_insufficient"
            out["right_candles_needed"]  = int(SWING_LOOKBACK)
            out["right_candles_present"] = int(n - 1 - ext_idx)
            return out

        # Left/right neighbour windows used by detect_swings.
        l_lo = max(0, ext_idx - SWING_LOOKBACK)
        l_hi = ext_idx                       # exclusive
        r_lo = ext_idx + 1
        r_hi = min(n, ext_idx + SWING_LOOKBACK + 1)  # exclusive

        blocker_side = None
        blocker_idx  = None
        blocker_val  = None
        if side == 'ceiling':
            for j in range(l_lo, l_hi):
                if H_arr[j] >= ext_price:
                    blocker_side = "left"
                    blocker_idx  = j
                    blocker_val  = float(H_arr[j])
                    break
            if blocker_idx is None:
                for j in range(r_lo, r_hi):
                    if H_arr[j] >= ext_price:
                        blocker_side = "right"
                        blocker_idx  = j
                        blocker_val  = float(H_arr[j])
                        break
        else:
            for j in range(l_lo, l_hi):
                if L_arr[j] <= ext_price:
                    blocker_side = "left"
                    blocker_idx  = j
                    blocker_val  = float(L_arr[j])
                    break
            if blocker_idx is None:
                for j in range(r_lo, r_hi):
                    if L_arr[j] <= ext_price:
                        blocker_side = "right"
                        blocker_idx  = j
                        blocker_val  = float(L_arr[j])
                        break

        if blocker_idx is not None:
            out["reason"] = "blocked_by_neighbour"
            out["blocker_side"]  = blocker_side
            out["blocker_idx"]   = int(blocker_idx)
            out["blocker_ts"]    = _ts_iso(df, blocker_idx)
            out["blocker_value"] = blocker_val
        else:
            # No neighbour violates the rule — extreme should be a swing.
            # Means detect_swings has a logic gap (shouldn't happen).
            out["reason"] = "rule_satisfied_but_no_swing_detected"

        return out

    placeholder_diag = {
        "ceiling": _diagnose_placeholder('ceiling'),
        "floor":   _diagnose_placeholder('floor'),
        "leg_start_idx": int(leg_start_idx),
        "n_candles_in_df": int(n),
    }

    # Cold-start fallback: no event fired across entire walk.
    if is_cold_start and last_event_type is None:
        H_arr = df['High'].values.astype(float)
        L_arr = df['Low'].values.astype(float)
        C_arr = df['Close'].values.astype(float)
        fb_lo_idx = max(0, n - FALLBACK_WINDOW_H1)
        fb_hi_arr = H_arr[fb_lo_idx:]
        fb_lo_arr = L_arr[fb_lo_idx:]
        rng_hi = float(fb_hi_arr.max())
        rng_lo = float(fb_lo_arr.min())
        hi_idx = fb_lo_idx + int(fb_hi_arr.argmax())
        lo_idx = fb_lo_idx + int(fb_lo_arr.argmin())
        ceiling = {
            "price": rng_hi, "ts": _ts_iso(df, hi_idx), "idx": hi_idx,
            "is_placeholder": False
        }
        floor = {
            "price": rng_lo, "ts": _ts_iso(df, lo_idx), "idx": lo_idx,
            "is_placeholder": False
        }
        first_c = float(C_arr[fb_lo_idx])
        last_c  = float(C_arr[n - 1])
        trend = 'bearish' if last_c < first_c else 'bullish'
        fallback_active = True
        _log_safe({
            'candle_ts':    _ts_iso(df, n - 1),
            'pair':         pair_name,
            'timeframe':    'H1',
            'event_kind':   'FALLBACK_USED',
            'direction':    trend,
            'fallback_window_h1': FALLBACK_WINDOW_H1,
            'range_high':   float(rng_hi),
            'range_low':    float(rng_lo),
        })

    new_state = {
        "trend":                  trend,
        "ceiling_price":          ceiling["price"],
        "ceiling_ts":             ceiling["ts"],
        "ceiling_is_placeholder": bool(ceiling["is_placeholder"]),
        "floor_price":            floor["price"],
        "floor_ts":               floor["ts"],
        "floor_is_placeholder":   bool(floor["is_placeholder"]),
        "last_event_type":        last_event_type,
        "last_event_tier":        last_event_tier,
        "last_event_direction":   last_event_direction,
        "last_event_ts":          last_event_ts,
        "last_event_idx_iso":     last_event_idx_iso,
        "last_event_chop":        bool(last_event_chop),
        "last_scanned_ts":        _ts_iso(df, n - 1),
        "fallback_active":        bool(fallback_active),
        "events":                 events_ring,
        # Non-persisted diagnostic — caller must strip before save_state().
        # See PLACEHOLDER_DIAG_KEY constant.
        PLACEHOLDER_DIAG_KEY:     placeholder_diag,
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
    pair_name = (pair_conf or {}).get("name") if pair_conf else None
    if is_cold:
        df_used = df.tail(COLDSTART_WINDOW_H1).copy().reset_index(drop=True) \
                  if len(df) > COLDSTART_WINDOW_H1 else df
        return _walk_forward(df_used, prior_state=None, pair_name=pair_name)
    return _walk_forward(df, prior_state=prior_state, pair_name=pair_name)


# --- Public API used by Phase 1 + Phase 2 (read-only) ------------------------

def compute_pd_position(price: float, walls: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a price and the wall state for a pair, return PD positioning.

    Returns dict:
      {
        "valid":         bool,      # True if both walls have prices and range is non-degenerate
        "range_high":    float,
        "range_low":     float,
        "equilibrium":   float,
        "pd_position":   float,     # 0.0 = at floor, 1.0 = at ceiling, None if invalid
        "ceiling_is_placeholder": bool,
        "floor_is_placeholder":   bool,
        "tentative":     bool,      # True if either wall is currently tentative (rolling extreme)
        "fallback_active":        bool,
        "source":        str        # human label
      }

    Geometry is valid whenever both walls have prices (confirmed OR tentative)
    and the range is non-degenerate. Whether either wall is tentative is
    surfaced via the `tentative` flag for downstream consumers to label.
    PD scoring is no longer driven by this function's `valid` (PD removed
    from scorecard); `valid` here only gates whether geometry renders.
    """
    if not walls:
        return {"valid": False, "source": "no_state",
                "ceiling_is_placeholder": True, "floor_is_placeholder": True,
                "tentative": True, "fallback_active": False, "pd_position": None,
                "range_high": 0.0, "range_low": 0.0, "equilibrium": 0.0}

    ceiling = walls.get("ceiling_price")
    floor   = walls.get("floor_price")
    cph = bool(walls.get("ceiling_is_placeholder", True))
    fph = bool(walls.get("floor_is_placeholder", True))
    fb  = bool(walls.get("fallback_active", False))
    tentative = cph or fph

    base = {
        "ceiling_is_placeholder": cph,
        "floor_is_placeholder":   fph,
        "tentative":              tentative,
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

    eq = (float(ceiling) + float(floor)) / 2.0
    width = float(ceiling) - float(floor)
    pos = (float(price) - float(floor)) / width if width > 0 else None
    if fb:
        src = "fallback_window"
    elif tentative:
        src = "tentative_walls"
    else:
        src = "structural"
    return {
        "valid": True,
        "range_high": float(ceiling),
        "range_low":  float(floor),
        "equilibrium": eq,
        "pd_position": pos,
        "ceiling_is_placeholder": cph,
        "floor_is_placeholder":   fph,
        "tentative":              tentative,
        "fallback_active":        fb,
        "source": src,
    }
