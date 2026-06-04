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

CHoCH has two tiers — both use lookback=3 swings; geometry distinguishes them:
  - Major: close past the OPPOSITE confirmed wall. The dealing range itself
    is broken; trend flips. No precondition beyond displacement + premium/
    discount gate.
  - Minor: close past the most recent unbroken lookback=3 swing of the
    relevant type INSIDE the current leg, AFTER price has tested the
    trend-direction wall within MINOR_CHOCH_WALL_TOUCH_ATR * ATR somewhere
    in the same leg. Informational weakening flag; does NOT flip trend;
    does NOT move walls.

  Major has precedence: a candle that qualifies for both fires Major.

Wall update rule (LOCKED): walls move ONLY when a wall is broken.
  - BOS: trend-direction wall trails to break-candle extreme (tentative
    until promoted); opposite wall trails to deepest pullback inside the
    just-completed leg.
  - Major CHoCH AT WALL (close past opposite confirmed wall): broken wall
    → tentative; opposite wall stays put (it's the prior trend anchor =
    new trend's starting extreme). NO history search.
    EXCEPTION — chop CHoCH-at-wall: if chop=True on this event, the BOS
    that set the opposite wall just before was a fakeout. The opposite
    wall also resets to tentative at the break-candle extreme. Both walls
    re-anchor inside the new leg via _try_promote_placeholder /
    _refresh_tentative.
  - Minor CHoCH: walls do NOT move; trend does NOT flip.

Wall-stale rule: when the non-trend-side wall is confirmed and price closes
beyond it by >= STALE_WALL_ATR_MULT × ATR without firing a CHoCH (e.g. no
qualifying internal pivot yet), the wall is relabelled. It moves to the
most recent confirmed lookback=3 swing of the right type in the current
leg (strictly past the old wall on the breached side). If none exists yet,
the wall falls back to the breach candle's extreme as a placeholder that
refreshes forward via _refresh_tentative. No event fires; trend does NOT
flip. Geometry-only relabel that keeps the dealing range visually honest.

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
  - SWING_LOOKBACK = 3 — single swing pool, drives walls, BOS, and both
    CHoCH tiers. No separate lb-2 pool.
  - BOS displacement >= 0.40 × H1 ATR past broken wall.
  - CHoCH displacement >= 0.60 × H1 ATR past broken pivot (Major and Minor
    use the same threshold).
  - Minor CHoCH wall-touch precondition: 0.35 × H1 ATR proximity to the
    trend-direction wall, anywhere in the current leg.
  - Premium / discount threshold = 25% of dealing range.
  - Tentative (placeholder) walls give geometry only — events do NOT fire
    on placeholder-wall break.
  - Event ring: last 20 qualified events kept on state for Phase 2 readers
    (BOS sequence count; zone invalidation).
  - Chop flag: CHoCH within 5 candles of prior event marks last_event_chop.
  - Atomic state writes (temp + rename).
  - Event detection order per candle: BOS first, then CHoCH; stale-wall
    relabel runs only when no event fired (prevents stale from swallowing
    a Major CHoCH whose displacement crossed both thresholds).
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

# Optional debug-module override hook. The debug package is a sidecar
# (read-only against Phase 1); the only point of contact is here, where
# an operator-issued one-shot override file can request a forced range
# rebuild from a specific timestamp. Import + call are wrapped — if the
# package is absent OR raises for any reason, this resolves to None and
# Phase 1 runs exactly as before.
try:
    from debug import overrides as _debug_overrides  # type: ignore
except Exception:
    _debug_overrides = None

# --- Tunables (locked) -------------------------------------------------------

# Single swing pool. lookback=3 wall-grade swings drive walls, BOS detection,
# and both Major and Minor CHoCH detection. Minor CHoCH is distinguished
# from Major by event geometry, not by a separate swing pool — see
# _pick_choch_pivot for the Major (wall break) vs Minor (internal break
# after wall touch) rule. Minor CHoCH does not flip trend and does not move
# walls; informational weakening flag only.
SWING_LOOKBACK = 3

# Minimum swing leg size in ATR(14) units, applied AFTER lookback=3 geometric
# detection. A confirmed pivot is only kept if the leg into it (distance from
# the previous kept opposite-type swing) is >= MIN_LEG_ATR_MULT * the average
# H1 ATR across that leg. This is the SINGLE definition of an H1 swing: every
# consumer (trend, CHoCH, BOS, walls, sweep scoring, charts) reads swings that
# have passed BOTH the lb-3 geometry gate AND this ATR leg-size gate. Tiny
# triangles never become structural swings. Owned here (lowest layer);
# smc_detector.get_swing_points calls down into _filter_swings_by_leg_atr so
# there is exactly one implementation.
MIN_LEG_ATR_MULT = 1.5

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

# Wall-stale rule: in a bullish trend, if close drops below the confirmed
# floor by >= STALE_WALL_ATR_MULT * ATR, the floor is no longer respected.
# Symmetric for bearish (close above confirmed ceiling). Does NOT fire an
# event and does NOT flip trend — geometry-only relabel of a wall that the
# market has already invalidated without a CHoCH. Same threshold as BOS
# (0.40) keeps the "meaningful displacement" bar consistent.
STALE_WALL_ATR_MULT = 0.4

# Janitor rule (LOCKED): re-anchor an abandoned opposite-side wall when the
# market has clearly trended away from it. Complements STALE_WALL_ATR_MULT,
# which only fires when price closes PAST a wall — that never triggers for a
# wall the trend has run AWAY from. Fires when:
#   1. opposite-side wall predates trend_start_idx (set before current trend),
#   2. current trend has produced >= N confirmed lb-3 swings of the relevant
#      type strictly inside (trend_start_idx, current_i],
#   3. no BOS/CHoCH fired this candle AND the stale-wall rule did not relabel.
# Wall is re-anchored to the MOST EXTREME of those in-trend swings (deepest
# pullback low in a bullish trend; highest pullback high in a bearish trend).
# Old wall is discarded — OB store preserves zone memory independently.
# Pair-aware threshold: Forex pairs trend slower per H1 candle than indices
# / commodities, so Forex uses a tighter swing count; NAS100/Gold need one
# more swing to avoid firing on a single impulsive session.
#
# Long-trend handling: when the Major CHoCH that started the current trend
# is older than the visible rolling H1 window, trend_start_idx falls back to
# 0 (whole window belongs to this trend). The rule still fires in that case
# — the "wall predates trend" gate is bypassed because every in-window wall
# necessarily has idx >= 0. Without this, long-running trends could never
# be cleaned up (which was the original observed bug).
JANITOR_SWING_MIN_FOREX        = 3
JANITOR_SWING_MIN_INDEX_COMMOD = 4
JANITOR_INDEX_COMMOD_PAIRS = frozenset({"NAS100", "GOLD", "XAUUSD"})

# Minor CHoCH wall-touch precondition. A Minor CHoCH (internal lb-3 break
# inside the current trend) is only valid if price tested the trend-direction
# wall within MINOR_CHOCH_WALL_TOUCH_ATR * ATR somewhere in the current leg
# (since the last structural event). Filters mid-range chop where an internal
# pivot breaks without the trend ever actually testing the boundary. Loose
# enough to absorb data-feed wick noise (~0.10-0.15*ATR on forex H1); tight
# enough to require genuine proximity. Does NOT gate Major CHoCH.
MINOR_CHOCH_WALL_TOUCH_ATR = 0.35

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
    """Return ISO timestamp string for df row at positional idx.

    Refuses to fall back to str(idx) when no datetime source exists — returning
    an integer-string ts (e.g. "146") silently corrupts persisted state with
    positional indices that look like timestamps but compare wrong and never
    resolve back via _idx_from_ts. Better to return None and let callers see
    a missing ts than to write garbage."""
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
        # No datetime source — refuse to return a positional-index string.
        return None
    except Exception:
        return None


# --- Swing detection ---------------------------------------------------------

def _true_range_series(df) -> List[float]:
    """Per-candle True Range, length == len(df). Element 0 is NaN (no prior
    close). Pure-python True Range; lives here (lowest layer) so the ATR
    leg-size filter has no cross-import. smc_detector delegates to it.
    """
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    C = df['Close'].values.astype(float)
    n = len(df)
    tr = [float('nan')] * n
    for i in range(1, n):
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))
    return tr


def _filter_swings_by_leg_atr(swings: List[Dict[str, Any]], df,
                              period: int = 14,
                              min_mult: float = MIN_LEG_ATR_MULT
                              ) -> List[Dict[str, Any]]:
    """Filter a time-ordered swing list by minimum leg size in ATR units.

    The first swing passes unconditionally (no prior opposite swing to measure
    a leg against). Each subsequent swing must satisfy
        |price - prev_kept_opposite.price| >= min_mult * mean_TR_across_leg
    where the leg spans (prev_kept_opposite.idx, this_swing.idx].

    Same-type consecutive swings (high->high, low->low) bypass the test — they
    describe trend continuation, not a measurable leg; only opposite-type
    transitions form a leg whose size we can measure. A swing that fails is
    dropped WITHOUT updating the reference, so the next opposite swing is still
    measured against the same surviving anchor.

    SINGLE SOURCE OF TRUTH for the ATR leg-size gate. smc_detector calls this.
    """
    if not swings or df is None or len(df) < period + 1 or min_mult <= 0:
        return swings
    tr = _true_range_series(df)
    kept: List[Dict[str, Any]] = []
    last_opp_by_type: Dict[str, Dict[str, Any]] = {}  # 'high'->last kept low, 'low'->last kept high
    for s in swings:
        opp = 'low' if s['type'] == 'high' else 'high'
        ref = last_opp_by_type.get(opp)
        if ref is None:
            kept.append(s)
            last_opp_by_type[s['type']] = s
            continue
        a, b = ref['idx'], s['idx']
        if b <= a:
            # Defensive: out-of-order swing. Drop rather than divide by zero.
            continue
        leg_tr = [v for v in tr[a + 1: b + 1] if v == v]  # drop NaN (v==v is False for NaN)
        if not leg_tr:
            kept.append(s)
            last_opp_by_type[s['type']] = s
            continue
        avg_atr = sum(leg_tr) / len(leg_tr)
        leg_size = abs(s['price'] - ref['price'])
        if leg_size >= min_mult * avg_atr:
            kept.append(s)
            last_opp_by_type[s['type']] = s
        # else: leg too small -> discard as noise; reference unchanged.
    return kept


def detect_swings(df, lookback: int = SWING_LOOKBACK,
                  min_leg_atr_mult: Optional[float] = MIN_LEG_ATR_MULT
                  ) -> List[Dict[str, Any]]:
    """
    Find confirmed swing highs and swing lows over the entire df.

    A candle at idx i is:
      - a swing high if H[i] is STRICTLY GREATER than every other high in
        the window [i-lookback, i+lookback].
      - a swing low  if L[i] is STRICTLY LESS than every other low in
        the window [i-lookback, i+lookback].

    Strict comparison: equal highs / equal lows do NOT register as swings.
    A flat top / flat bottom across the window correctly produces no swing.

    After geometric detection, the ATR leg-size filter is applied (unless
    min_leg_atr_mult is None or <= 0). This is the single H1 swing definition:
    lb-3 geometry PLUS a leg that is large enough in ATR terms. Tiny triangles
    are removed. Pass min_leg_atr_mult=None ONLY for non-H1 / diagnostic use
    where the H1-tuned multiple does not apply.

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
    if min_leg_atr_mult is not None and min_leg_atr_mult > 0:
        out = _filter_swings_by_leg_atr(out, df, min_mult=min_leg_atr_mult)
    return out


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


def _wall_touched_in_leg(side: str, df, leg_start_idx: int, current_idx: int,
                         wall_price: Optional[float], atr: float) -> bool:
    """Return True if any candle in (leg_start_idx, current_idx] tested the
    given wall within MINOR_CHOCH_WALL_TOUCH_ATR * ATR.

    side: 'ceiling' or 'floor'. Caller passes the trend-direction wall (ceiling
    for bullish trend, floor for bearish) — this is the wall whose rejection
    qualifies a subsequent internal break as a Minor CHoCH.

    Evaluated against the CURRENT wall_price: trend-direction walls only
    tighten inward within a leg (placeholder -> confirmed via promote) and
    are never relabelled by stale-wall (stale targets the non-trend side),
    so a candle that satisfied the touch precondition against the looser
    placeholder also satisfies it against the tightened confirmed wall in
    every case where it matters.
    """
    if wall_price is None or atr is None or atr <= 0:
        return False
    if leg_start_idx + 1 > current_idx:
        return False
    threshold = MINOR_CHOCH_WALL_TOUCH_ATR * atr
    if side == 'ceiling':
        H = df['High'].values.astype(float)
        for j in range(leg_start_idx + 1, current_idx + 1):
            if H[j] >= wall_price - threshold:
                return True
        return False
    if side == 'floor':
        L = df['Low'].values.astype(float)
        for j in range(leg_start_idx + 1, current_idx + 1):
            if L[j] <= wall_price + threshold:
                return True
        return False
    return False


def _pick_choch_pivot(swings_lb3: List[Dict[str, Any]],
                      trend: str,
                      leg_start_idx: int,
                      current_idx: int,
                      ceiling: Dict[str, Any],
                      floor: Dict[str, Any],
                      df=None,
                      atr: Optional[float] = None,
                      ) -> Optional[Tuple[Dict[str, Any], str, bool]]:
    """Pick the pivot whose break would constitute a CHoCH at this candle.

    Returns (pivot_dict, tier, at_wall) or None if no candidate.

    Two tiers, both using lookback=3 swings:

      Major CHoCH — close past the opposite confirmed wall (floor in an
        uptrend, ceiling in a downtrend). The dealing range itself is broken;
        trend flips. No wall-touch precondition: a confirmed wall break
        with displacement is sufficient on its own.

      Minor CHoCH — close past the most recent unbroken lb-3 swing of the
        relevant type INSIDE the current leg, AFTER price has tested the
        trend-direction wall within MINOR_CHOCH_WALL_TOUCH_ATR * ATR
        somewhere in the same leg. Informational weakening flag; trend
        does NOT flip, walls do NOT move. The wall-touch precondition
        filters mid-range internal breaks where the trend never actually
        engaged its boundary.

    Major has precedence over Minor: when a candle qualifies for both (close
    past the opposite wall AND past an internal lb-3 pivot), Major fires.
    The internal break is implied by the wall break.

    Why most-recent (Minor): in an uptrend, sequential HLs rise — the most
    recent HL is the highest. Price closing below it IS a structural break.
    An older HL further down is still un-broken at that close; picking it
    would miss the real break.

    Search window: (leg_start_idx, current_idx-1] — strictly inside the
    current trend and strictly before the candle being tested.
    """
    if trend == 'bullish':
        target_type = 'low'           # Down CHoCH breaks an HL (swing low).
        opp_wall    = floor           # Wall whose break = Major CHoCH down.
        trend_wall_side = 'ceiling'   # Wall to test for the Minor precondition.
        trend_wall  = ceiling
    elif trend == 'bearish':
        target_type = 'high'          # Up CHoCH breaks an LH (swing high).
        opp_wall    = ceiling
        trend_wall_side = 'floor'
        trend_wall  = floor
    else:
        return None

    # 1. Major check first — opposite confirmed wall break has precedence
    #    over any internal pivot break on the same candle.
    if (opp_wall.get('price') is not None
            and not opp_wall.get('is_placeholder', True)):
        wall_pivot = {
            'idx':   opp_wall.get('idx'),
            'price': opp_wall['price'],
            'ts':    opp_wall.get('ts'),
        }
        return (wall_pivot, 'Major', True)

    # 2. Minor check — most recent unbroken lb-3 pivot inside the current
    #    leg. Requires wall-touch precondition on the trend-direction wall.
    lo, hi = leg_start_idx + 1, current_idx - 1
    if lo > hi:
        return None

    pivot = _most_recent_swing_in_window(swings_lb3, target_type, lo, hi)
    if pivot is None:
        return None

    # Wall-touch precondition. Skipped only when we lack the inputs to
    # evaluate it (df or atr missing) — caller always supplies them in the
    # live path; defensive None-guard preserves the historical signature
    # contract.
    if df is None or atr is None or atr <= 0:
        return None
    if not _wall_touched_in_leg(trend_wall_side, df, leg_start_idx,
                                current_idx, trend_wall.get('price'), atr):
        return None

    return (pivot, 'Minor', False)


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
        "trend_start_ts": None,               # ISO ts of the last Major CHoCH candle (or earliest candle on cold start). Anchors the opposite-wall trail across the WHOLE trend, not just the latest leg.
        # Minor BOS de-dup: ISO ts of the lb-3 pivot most recently consumed by
        # a Minor BOS. Subsequent closes that "break" the SAME pivot do not
        # re-fire. Only a NEWER (higher-idx) pivot can trigger the next Minor
        # BOS. Resets on Major event (trend flip or Major BOS) since the leg
        # boundary moves and prior continuation memory is no longer relevant.
        "last_minor_bos_pivot_ts": None,
        "events": [],                         # ring of last EVENT_RING_MAX qualified events
        "swings": [],                         # persisted lb-3+ATR swing pool (single source for chart markers)
    }


def _resolve_placeholder(side: str, df, leg_start_idx: int, leg_end_idx: int,
                         swings: List[Dict[str, Any]]
                         ) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Try to resolve a placeholder wall on `side` ('ceiling' or 'floor') by
    finding the highest swing high (for ceiling) or lowest swing low (for
    floor) inside [leg_start_idx, leg_end_idx]. Returns (swing_dict_or_None,
    is_placeholder).

    DETECT-ONCE-THEN-PICK: swings are NOT re-detected here. The caller passes
    the single full-df swing pool (`swings`, already lb-3 + ATR-filtered) and
    this function SELECTS the qualifying swing from it by absolute index range.
    This is what guarantees the wall is anchored to a real (ATR-qualified)
    swing — the same pool trend/CHoCH/BOS use — and removes the old per-slice
    re-detection where a tiny window could have nothing for the ATR filter to
    measure against.

    The break candle at leg_start_idx is naturally excluded: candidates are
    taken from (leg_start_idx, leg_end_idx], so the break candle is never a
    candidate, and full-df detection already gave every interior swing its true
    lookback-3 neighbours.

    If no confirmed leg-internal swing is present, returns (rolling_extreme,
    True) where rolling_extreme is a synthetic dict with the highest H or
    lowest L in the leg (visualization only — wall stays placeholder).
    """
    if df is None or leg_start_idx >= leg_end_idx:
        return None, True

    target_type = 'high' if side == 'ceiling' else 'low'
    # Select from the shared pool: same-type swings strictly inside the leg.
    # (leg_start_idx, leg_end_idx] — break candle (leg_start_idx) excluded.
    candidates = [
        {'type': s['type'], 'idx': s['idx'], 'price': float(s['price']), 'ts': s['ts']}
        for s in swings
        if s['type'] == target_type and leg_start_idx < s['idx'] <= leg_end_idx
    ]
    if candidates:
        if side == 'ceiling':
            best = max(candidates, key=lambda s: s['price'])
        else:
            best = min(candidates, key=lambda s: s['price'])
        return {'idx': best['idx'], 'price': best['price'], 'ts': best['ts']}, False

    # No confirmed leg-internal swing yet — produce rolling extreme.
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
    inside (leg_start_idx, leg_end_idx).

    For a bearish BOS: side='ceiling', returns the HIGHEST confirmed swing
    high inside that range.

    The opposite wall represents the TREND ORIGIN (deepest unmitigated pullback
    of the entire trend), not just the latest leg. Callers therefore pass
    trend_start_idx (the most recent Major CHoCH candle, or cold-start anchor)
    as `leg_start_idx`. SMC semantics: a CHoCH against the trend should require
    breaking the trend's origin, not the most recent pullback.

    Strict interior — endpoints excluded (the trend-bounding swings themselves
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


def _needs_cold_start(prior_state: Optional[Dict[str, Any]]) -> bool:
    """True if prior_state cannot support an incremental walk and must be
    rebuilt from scratch.

    Cold when prices are missing (genuine first run / empty state) OR when
    prices exist but the anchoring timestamps are null. The latter is a
    POISONED state: a wipe or any write through a df without a usable datetime
    source leaves prices intact but nulls every ts (ceiling/floor/
    last_scanned). Incremental mode then carries un-anchorable events forward
    forever — events never resolve to a candle, so no OB is ever built and the
    pair emits 0 zones permanently. Treating null-ts-with-prices as cold makes
    the system self-heal on the next scan instead of staying dead."""
    if (prior_state is None
            or prior_state.get("ceiling_price") is None
            or prior_state.get("floor_price") is None):
        return True
    # Priced but unanchorable -> poisoned. A genuine warm state always has at
    # least one wall ts and a last_scanned_ts to drive the incremental walk.
    if (prior_state.get("ceiling_ts") is None
            and prior_state.get("floor_ts") is None):
        return True
    if prior_state.get("last_scanned_ts") is None:
        return True
    return False


def _walk_forward(df, prior_state: Optional[Dict[str, Any]] = None,
                  pair_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Walk forward through df starting from either:
      - candle 0 (cold start, prior_state is empty/None), or
      - the candle just after prior_state['last_scanned_ts'] (incremental).

    Detects Major BOS (confirmed wall break in trend direction), Minor BOS
    (trend-direction internal lb-3 swing break inside the current leg,
    informational continuation flag), Major CHoCH (opposite confirmed wall
    break), and Minor CHoCH (lookback=3 internal pivot break, gated by
    wall-touch precondition on the trend-direction wall) per the locked
    rules. Updates walls only when a wall is broken (Major BOS or Major
    CHoCH-at-wall). Minor BOS and Minor CHoCH do not flip trend and do not
    move walls.

    Cold start fallback: if no event fires across the entire window, walls
    fall back to window high/low and fallback_active = True.
    """
    n = len(df) if df is not None else 0
    if n == 0:
        return prior_state or _empty_state()

    atr = _compute_atr(df)
    if atr is None or atr <= 0:
        return prior_state or _empty_state()

    # Single swing pool (computed once per walk). Drives walls, BOS, and
    # both CHoCH tiers.
    swings_lb3 = detect_swings(df, lookback=SWING_LOOKBACK)

    state = prior_state if prior_state else _empty_state()
    is_cold_start = _needs_cold_start(prior_state)

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
    trend_start_idx      = 0  # Anchor for OPPOSITE-wall trailing on BOS Major and stale-wall replacement. Resets ONLY on Major CHoCH (or cold start). leg_start_idx still advances on every Major event for promote/refresh of the trend-direction wall.
    fallback_active      = state.get("fallback_active", False)
    last_minor_bos_pivot_ts = state.get("last_minor_bos_pivot_ts")
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

    # Leg start = the candle of the most recent BOS / Major CHoCH-at-wall.
    # Walls are NOT leg boundaries — a placeholder wall trails forward each
    # scan, so including it here would drag leg_start with it and shrink the
    # leg over time. Confirmed walls sit inside the leg, not at its edge.
    # No event yet (cold start) -> leg starts at candle 0.
    if last_event_ts:
        ev_idx = _idx_from_ts(last_event_ts)
        if ev_idx is not None:
            leg_start_idx = ev_idx

    if last_event_idx_iso:
        last_event_local_idx = _idx_from_ts(last_event_idx_iso)

    # Resolve trend_start_idx from persisted state. Priority:
    #   1. state['trend_start_ts'] (authoritative if present).
    #   2. Most recent Major CHoCH ts in the events ring (back-fill for states
    #      written before trend_start_ts was added).
    #   3. 0 (cold start — no Major CHoCH yet; the whole window is the trend).
    # If a ts is present but not resolvable inside df (out of window), clamp
    # to 0 — safer to over-include than to mis-anchor inside the latest leg.
    trend_start_ts_persisted = state.get("trend_start_ts")
    if trend_start_ts_persisted:
        resolved = _idx_from_ts(trend_start_ts_persisted)
        trend_start_idx = resolved if resolved is not None else 0
    else:
        last_major_choch_ts: Optional[str] = None
        for ev in reversed(events_ring):
            if ev.get('type') == 'CHoCH' and ev.get('tier') == 'Major':
                last_major_choch_ts = ev.get('candle_ts')
                break
        if last_major_choch_ts:
            resolved = _idx_from_ts(last_major_choch_ts)
            trend_start_idx = resolved if resolved is not None else 0
        else:
            trend_start_idx = 0

    # Promote a placeholder wall if a confirmed swing now exists in
    # (leg_start_idx, current_i].
    #
    # Two distinct outcomes from _resolve_placeholder:
    #   - Confirmed lb-3 swing found (is_ph=False) -> structural anchor,
    #     wall becomes ANCHORED. Replaces unconditionally — an lb-3 pivot
    #     inside the leg is the meaningful new structural reference for
    #     this trend, even if it sits inward of the break-candle extreme.
    #   - No lb-3 swing yet (is_ph=True) -> rolling extreme. Acts as the
    #     same one-way ratchet as _refresh_tentative: only update if the
    #     rolling extreme is FURTHER from mid-range than the current
    #     placeholder. Otherwise we'd overwrite a meaningful anchor (BOS
    #     break-candle extreme; surviving wall demoted across a Major
    #     CHoCH) with a single-candle OHLC of the latest bar.
    def _try_promote_placeholder(side: str, current_i: int):
        if side == 'ceiling':
            if not ceiling["is_placeholder"]:
                return
            promoted, is_ph = _resolve_placeholder('ceiling', df, leg_start_idx + 1, current_i, swings_lb3)
            if promoted is None:
                return
            if is_ph:
                # Rolling-extreme outcome — apply ratchet up only.
                if ceiling["price"] is not None and promoted["price"] <= ceiling["price"]:
                    return
            ceiling["price"] = promoted["price"]
            ceiling["ts"]    = promoted["ts"]
            ceiling["idx"]   = promoted["idx"]
            ceiling["is_placeholder"] = is_ph
        else:
            if not floor["is_placeholder"]:
                return
            promoted, is_ph = _resolve_placeholder('floor', df, leg_start_idx + 1, current_i, swings_lb3)
            if promoted is None:
                return
            if is_ph:
                # Rolling-extreme outcome — apply ratchet down only.
                if floor["price"] is not None and promoted["price"] >= floor["price"]:
                    return
            floor["price"] = promoted["price"]
            floor["ts"]    = promoted["ts"]
            floor["idx"]   = promoted["idx"]
            floor["is_placeholder"] = is_ph

    # Refresh a still-tentative wall to the rolling extreme since leg_start.
    # Geometry only — events do NOT fire on placeholder-wall break.
    #
    # One-way ratchet (LOCKED): a placeholder wall represents the most extreme
    # price reached since the wall was last set. Refresh may only EXTEND the
    # placeholder further from mid-range (ceiling can only ratchet UP, floor
    # can only ratchet DOWN). It must NEVER pull the wall back toward price.
    #
    # Why: the wall's current price is a real anchor — either the break-candle
    # extreme set on BOS/Major CHoCH, or the surviving wall from the prior
    # trend that was demoted to placeholder. Replacing it with a rolling
    # extreme of a shorter slice (e.g. when leg_start advanced to the latest
    # candle on a fresh event) collapses the dealing range to single-candle
    # OHLC. Observed bug: post-Major-CHoCH at candle N, the trend-flipped
    # floor (surviving from prior bearish trend, real anchor) was overwritten
    # to Low(N+1) — a few pips below current price — destroying the dealing
    # range geometry.
    def _refresh_tentative(side: str, current_i: int):
        if side == 'ceiling':
            if not ceiling["is_placeholder"]:
                return
            lo, hi = leg_start_idx + 1, current_i
            if lo > hi:
                return
            H_arr = df['High'].values.astype(float)
            rng_idx = lo + int(H_arr[lo: hi + 1].argmax())
            rng_price = float(H_arr[rng_idx])
            # Ratchet up only — never retreat.
            if ceiling["price"] is None or rng_price > ceiling["price"]:
                ceiling["price"] = rng_price
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
            rng_price = float(L_arr[rng_idx])
            # Ratchet down only — never retreat.
            if floor["price"] is None or rng_price < floor["price"]:
                floor["price"] = rng_price
                floor["ts"]    = _ts_iso(df, rng_idx)
                floor["idx"]   = rng_idx

    # Relabel a confirmed non-trend-side wall as stale when price has already
    # closed beyond it by >= STALE_WALL_ATR_MULT * ATR. No event fires; trend
    # does NOT flip. The wall moves to the most recent confirmed lb-3 swing
    # of the right type in (leg_start_idx, current_i]; if none exists yet,
    # falls back to the breach candle's extreme as a placeholder that will
    # refresh forward via _refresh_tentative. One-way ratchet: in bullish
    # trend the floor can only move DOWN here; in bearish the ceiling can
    # only move UP.
    def _stale_wall_check(current_i: int, close_i: float):
        nonlocal_trend = trend
        if nonlocal_trend not in ('bullish', 'bearish'):
            return
        if atr is None or atr <= 0:
            return

        if nonlocal_trend == 'bullish':
            # Floor stale when close drops below confirmed floor by >= N*ATR.
            if (floor["price"] is None
                    or floor["is_placeholder"]
                    or close_i >= floor["price"] - STALE_WALL_ATR_MULT * atr):
                return
            new_floor_price = float(close_i)  # tentative target before swing search
            # Most recent confirmed lb-3 swing low strictly inside the TREND
            # (since last Major CHoCH, not just latest leg) and STRICTLY BELOW
            # the current floor (the wall we're discarding). Trend-scoped so
            # the replacement floor stays anchored at trend origin, not at a
            # mid-trend pullback.
            cand = [
                s for s in swings_lb3
                if s['type'] == 'low'
                and trend_start_idx < s['idx'] <= current_i
                and s['price'] < floor["price"]
            ]
            if cand:
                cand.sort(key=lambda s: s['idx'], reverse=True)
                best = cand[0]
                floor["price"] = float(best['price'])
                floor["ts"]    = best['ts']
                floor["idx"]   = best['idx']
                floor["is_placeholder"] = False
            else:
                floor["price"] = float(df['Low'].iloc[current_i])
                floor["ts"]    = _ts_iso(df, current_i)
                floor["idx"]   = current_i
                floor["is_placeholder"] = True
            _log_safe({
                'candle_ts':       _ts_iso(df, current_i),
                'pair':            pair_name,
                'timeframe':       'H1',
                'event_kind':      'WALL_STALE',
                'wall_side':       'floor',
                'trend':           nonlocal_trend,
                'close_price':     float(close_i),
                'new_wall_price':  float(floor["price"]),
                'new_wall_is_placeholder': bool(floor["is_placeholder"]),
                'atr':             float(atr),
                'threshold_atr':   float(STALE_WALL_ATR_MULT),
            })
            return

        # bearish: ceiling stale when close rises above confirmed ceiling.
        if (ceiling["price"] is None
                or ceiling["is_placeholder"]
                or close_i <= ceiling["price"] + STALE_WALL_ATR_MULT * atr):
            return
        cand = [
            s for s in swings_lb3
            if s['type'] == 'high'
            and trend_start_idx < s['idx'] <= current_i
            and s['price'] > ceiling["price"]
        ]
        if cand:
            cand.sort(key=lambda s: s['idx'], reverse=True)
            best = cand[0]
            ceiling["price"] = float(best['price'])
            ceiling["ts"]    = best['ts']
            ceiling["idx"]   = best['idx']
            ceiling["is_placeholder"] = False
        else:
            ceiling["price"] = float(df['High'].iloc[current_i])
            ceiling["ts"]    = _ts_iso(df, current_i)
            ceiling["idx"]   = current_i
            ceiling["is_placeholder"] = True
        _log_safe({
            'candle_ts':       _ts_iso(df, current_i),
            'pair':            pair_name,
            'timeframe':       'H1',
            'event_kind':      'WALL_STALE',
            'wall_side':       'ceiling',
            'trend':           nonlocal_trend,
            'close_price':     float(close_i),
            'new_wall_price':  float(ceiling["price"]),
            'new_wall_is_placeholder': bool(ceiling["is_placeholder"]),
            'atr':             float(atr),
            'threshold_atr':   float(STALE_WALL_ATR_MULT),
        })

    def _janitor_wall_check(current_i: int):
        # Diagnostic skip-reason logger. Fires once per candle when the janitor
        # exits without re-anchoring. Lets us see which guard blocked the rule
        # in production without re-running anything.
        def _skip(reason: str, extra: Optional[Dict[str, Any]] = None):
            payload = {
                'candle_ts':  _ts_iso(df, current_i),
                'pair':       pair_name,
                'timeframe':  'H1',
                'event_kind': 'WALL_JANITOR_SKIP',
                'reason':     reason,
                'trend':      trend,
                'trend_start_idx': int(trend_start_idx),
                'current_i':       int(current_i),
            }
            if extra:
                payload.update(extra)
            _log_safe(payload)

        if trend not in ('bullish', 'bearish'):
            _skip('no_trend')
            return
        # trend_start_idx == 0 is VALID — it means the trend origin (last Major
        # CHoCH) is older than the visible rolling window, so the whole window
        # belongs to this trend. Only exit if start has caught up with current.
        if trend_start_idx >= current_i:
            _skip('trend_start_at_or_after_current')
            return

        # Flag set when trend_start_idx fell back to 0 because the persisted
        # trend_start_ts couldn't be resolved inside the visible df window.
        # Used purely for diagnostics — does not change behaviour.
        trend_start_out_of_window = (
            bool(state.get("trend_start_ts")) and trend_start_idx == 0
        )

        swing_min = (JANITOR_SWING_MIN_INDEX_COMMOD
                     if pair_name in JANITOR_INDEX_COMMOD_PAIRS
                     else JANITOR_SWING_MIN_FOREX)

        if trend == 'bullish':
            wall = floor
            opposite_type = 'low'
        else:
            wall = ceiling
            opposite_type = 'high'

        if wall["price"] is None:
            _skip('wall_price_none')
            return
        if wall["is_placeholder"]:
            _skip('wall_is_placeholder')
            return
        # Predates current trend? Compare against trend_start_idx, not ts —
        # idx may be None on resumed state, so resolve from wall ts when needed.
        wall_idx = wall.get("idx")
        if wall_idx is None:
            wall_ts = wall.get("ts")
            if wall_ts is None:
                _skip('wall_ts_missing')
                return
            resolved = _idx_from_ts(wall_ts)
            if resolved is None:
                # Wall was set before the visible window — by construction it
                # predates trend_start_idx. Treat it as eligible by setting
                # wall_idx to -1 (older than any in-window candle).
                wall_idx = -1
            else:
                wall_idx = resolved
        if wall_idx >= trend_start_idx and not trend_start_out_of_window:
            # When trend origin is out-of-window (trend_start_idx==0), the
            # "wall predates trend" check is meaningless — every in-window
            # wall has idx >= 0. Skip the gate so long trends remain eligible.
            _skip('wall_inside_current_trend',
                  {'wall_idx': int(wall_idx)})
            return

        in_trend_swings = [
            s for s in swings_lb3
            if s['type'] == opposite_type
            and trend_start_idx < s['idx'] <= current_i
        ]
        if len(in_trend_swings) < swing_min:
            _skip('not_enough_swings',
                  {'swing_count': int(len(in_trend_swings)),
                   'swing_min':   int(swing_min),
                   'trend_start_out_of_window': bool(trend_start_out_of_window)})
            return

        if trend == 'bullish':
            best = min(in_trend_swings, key=lambda s: s['price'])
        else:
            best = max(in_trend_swings, key=lambda s: s['price'])

        old_price = float(wall["price"])
        old_ts    = wall["ts"]
        wall["price"] = float(best['price'])
        wall["ts"]    = best['ts']
        wall["idx"]   = best['idx']
        wall["is_placeholder"] = False

        _log_safe({
            'candle_ts':       _ts_iso(df, current_i),
            'pair':            pair_name,
            'timeframe':       'H1',
            'event_kind':      'WALL_JANITOR',
            'wall_side':       'floor' if trend == 'bullish' else 'ceiling',
            'trend':           trend,
            'old_wall_price':  old_price,
            'old_wall_ts':     old_ts,
            'new_wall_price':  float(wall["price"]),
            'new_wall_ts':     wall["ts"],
            'swing_count':     int(len(in_trend_swings)),
            'swing_min':       int(swing_min),
            'trend_start_out_of_window': bool(trend_start_out_of_window),
        })

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

        # Stale-wall check is deferred to AFTER BOS/CHoCH detection on this
        # candle. Running it first would mutate the wall to a placeholder
        # before the event tests run, silently swallowing a Major CHoCH at
        # wall whose displacement crossed both the stale (0.4*ATR) and CHoCH
        # (0.6*ATR) thresholds. Event detection has first refusal; stale is
        # the no-event cleanup. See block after `if event_kind is None:`.

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

        # 2b. Minor BOS — trend-direction internal lb-3 swing break inside the
        # current leg. Continuation signal that today's Major-BOS path misses
        # because the wall hasn't broken yet. Same 0.4*ATR displacement bar
        # as Major BOS. Walls do NOT move, trend does NOT flip, leg_start does
        # NOT advance — informational sub-event inside the active leg. No
        # wall-touch precondition (continuation doesn't require the trend to
        # have engaged the boundary; the displacement filter alone suffices).
        # Direction must match trend (locked rule: BOS minor or major fires
        # only in the system-recorded trend direction).
        if event_kind is None and trend in ('bullish', 'bearish'):
            target_type = 'high' if trend == 'bullish' else 'low'
            lo, hi = leg_start_idx + 1, i - 1
            if lo <= hi:
                pivot = _most_recent_swing_in_window(swings_lb3, target_type, lo, hi)
                # De-dup against already-consumed pivot. Once a Minor BOS has
                # broken pivot P, subsequent closes past the SAME P must not
                # fire a fresh event — pivot is "spent." Compare on ISO ts
                # (stable across slice rewindowing) not idx (slice-relative).
                # Only a strictly NEWER pivot ts can trigger the next event.
                if pivot is not None and last_minor_bos_pivot_ts is not None:
                    p_ts = pivot.get('ts')
                    if p_ts is not None and p_ts <= last_minor_bos_pivot_ts:
                        pivot = None
                if pivot is not None:
                    pivot_price = float(pivot['price'])
                    if trend == 'bullish':
                        broke = close_i > pivot_price
                        disp = close_i - pivot_price
                    else:
                        broke = close_i < pivot_price
                        disp = pivot_price - close_i
                    if broke:
                        if disp >= BOS_ATR_MULT * atr:
                            event_kind = 'BOS'
                            event_tier = 'Minor'
                            event_direction = trend
                            event_at_wall = False
                            event_pivot = {
                                'idx':   pivot['idx'],
                                'price': pivot_price,
                                'ts':    pivot.get('ts'),
                            }
                            event_displacement = disp
                            # Pivot is now spent. Subsequent bars cannot re-
                            # break the same swing high/low; the next Minor
                            # BOS needs a fresh, later lb-3 swing.
                            last_minor_bos_pivot_ts = pivot.get('ts')
                        else:
                            _log_safe({
                                'candle_ts': _ts_iso(df, i), 'pair': pair_name, 'timeframe': 'H1',
                                'event_kind': 'BREAK_REJECTED', 'reject_reason': 'atr_threshold',
                                'attempted_event': 'BOS-Minor', 'direction': trend,
                                'swing_price': float(pivot_price), 'swing_ts': pivot.get('ts'),
                                'close_price': float(close_i), 'displacement': float(disp),
                                'displacement_atr': float(disp / atr),
                                'threshold_atr': float(BOS_ATR_MULT), 'atr': float(atr),
                                'broken_was_wall': False,
                            })

        # 3. CHoCH check — only when there's an established trend AND no BOS already
        # fired this candle. Direction is AGAINST trend.
        if event_kind is None and trend in ('bullish', 'bearish'):
            choch_dir = 'bearish' if trend == 'bullish' else 'bullish'
            picked = _pick_choch_pivot(swings_lb3, trend,
                                       leg_start_idx, i, ceiling, floor,
                                       df=df, atr=atr)
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
            # No BOS/CHoCH fired. Only now run the stale-wall relabel — if
            # price has closed past the non-trend-side wall by
            # >= STALE_WALL_ATR_MULT * ATR without a qualifying event, the
            # wall is no longer respected and is relabelled. Geometry-only;
            # no event, no trend flip.
            _stale_wall_check(i, close_i)
            _janitor_wall_check(i)
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

        if event_kind == 'BOS' and event_tier == 'Major':
            # Trend-direction wall just broke -> tentative at break-candle extreme.
            # Opposite wall = TREND ORIGIN: deepest unmitigated lb-3 swing across
            # the whole current trend (since the last Major CHoCH), NOT just the
            # latest leg. SMC: a counter-trend CHoCH should require breaking
            # where the trend started, not the most recent pullback.
            #
            # Non-regression guard: in bullish trend, the new floor must be <=
            # current floor (the trend cannot lose its deepest defense by
            # extending). Mirror for bearish ceiling. The deepest-of-trend
            # search across (trend_start_idx, i) typically satisfies this
            # automatically, but the current wall may have been set by a
            # rolling-extreme refresh that found a deeper low than the lb-3
            # pool currently exposes — in that case we keep the existing wall.
            if event_direction == 'bullish':
                trailed = _trail_inside_leg('floor', swings_lb3, trend_start_idx, i)
                # Non-regression: confirmed floor cannot ratchet UP (would lose
                # the deepest defense of the trend). A PLACEHOLDER floor has no
                # such standing — any confirmed lb-3 swing replaces it cleanly.
                if trailed is not None and (
                    floor["price"] is None
                    or floor.get("is_placeholder", True)
                    or trailed["price"] <= floor["price"]
                ):
                    floor["price"] = trailed["price"]
                    floor["ts"]    = trailed["ts"]
                    floor["idx"]   = trailed["idx"]
                    floor["is_placeholder"] = False
                ceiling["price"] = float(df['High'].iloc[i])
                ceiling["ts"]    = _ts_iso(df, i)
                ceiling["idx"]   = i
                ceiling["is_placeholder"] = True
            else:
                trailed = _trail_inside_leg('ceiling', swings_lb3, trend_start_idx, i)
                # Non-regression: confirmed ceiling cannot ratchet DOWN. A
                # placeholder ceiling is replaced unconditionally.
                if trailed is not None and (
                    ceiling["price"] is None
                    or ceiling.get("is_placeholder", True)
                    or trailed["price"] >= ceiling["price"]
                ):
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
            # Under the locked rules, Major CHoCH always fires at the opposite
            # confirmed wall (event_at_wall=True). _pick_choch_pivot enforces
            # this — internal-pivot CHoCH is always Minor.
            # The broken wall becomes tentative. Opposite wall normally STAYS
            # (it's the prior trend anchor = new trend's starting extreme).
            # EXCEPTION: chop CHoCH-at-wall (chop_this_event=True). The BOS
            # that set the opposite wall just before was a fakeout. Reset
            # BOTH walls so they re-anchor inside the new leg (avoids
            # inheriting spike-derived walls like USDJPY 160.7).
            # Both walls become placeholders on Major CHoCH:
            #   - Broken wall: re-anchored to break-candle extreme (the new
            #     trend-direction wall, will trail forward as the new leg
            #     builds).
            #   - Surviving wall: keeps its PRICE (real lb-3 swing, prior trend
            #     anchor) but flips to placeholder so the new trend can
            #     re-promote / replace it as its own structure develops. Without
            #     this, the surviving wall stays locked at the prior trend's
            #     extreme (e.g. USDJPY 160.7) and never moves even as price
            #     spends days far away from it. The non-regression guard on
            #     subsequent BOS also skips placeholders, freeing the wall to
            #     anchor at a tighter trend-internal lb-3 swing once one forms.
            # Chop case: BOTH walls fully reset to break-candle extremes
            # (the BOS that set the surviving wall was a fakeout — don't carry
            # its spike price forward at all).
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
                else:
                    # Surviving ceiling: keep price/ts/idx, demote to placeholder.
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
                else:
                    # Surviving floor: keep price/ts/idx, demote to placeholder.
                    floor["is_placeholder"] = True
            trend = event_direction
            # Trend just flipped. Anchor trend_start_idx at the CHoCH candle.
            # The surviving wall was demoted to placeholder above, so the next
            # BOS's opposite-wall trail (which skips the non-regression guard
            # for placeholder walls) can freely re-anchor to a real lb-3 swing
            # built INSIDE the new trend.
            trend_start_idx = i

        elif event_kind == 'BOS' and event_tier == 'Minor':
            # Walls do NOT change. Trend does NOT flip. Leg_start does NOT
            # advance (Minor BOS is a sub-event inside the active leg —
            # treating it as a leg boundary would shrink the swing pool used
            # for the next Major event). Ring-only continuation flag; OB
            # build proceeds via smc_radar's existing event-ring consumer.
            pass

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
        # Minor BOS is a sub-event inside the active leg: it does NOT advance
        # leg_start_idx and does NOT overwrite the last_event_* pointers
        # (which are used downstream to identify the active leg's anchoring
        # event). It is only appended to the events ring so consumers
        # (smc_radar OB builder, Phase 2) can see continuation prints.
        is_minor_bos = (event_kind == 'BOS' and event_tier == 'Minor')
        if not is_minor_bos:
            last_event_type      = event_kind
            last_event_tier      = event_tier
            last_event_direction = event_direction
            last_event_ts        = _ts_iso(df, i)
            last_event_idx_iso   = _ts_iso(df, i)
            last_event_local_idx = i
            last_event_chop      = chop_this_event
            leg_start_idx        = i
            fallback_active      = False
            # Major event resets Minor BOS continuation memory. New leg gets
            # a clean swing pool — any earlier "spent" pivot is no longer
            # inside the active leg anyway.
            last_minor_bos_pivot_ts = None

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
        leg_len = max(0, hi - lo + 1)
        out: Dict[str, Any] = {
            "side": side,
            "is_placeholder": True,
            "leg_start_idx": int(leg_start_idx),
            "leg_start_ts":  _ts_iso(df, leg_start_idx) if 0 <= leg_start_idx < n else None,
            "leg_slice_len": int(leg_len),
            "current_idx":   int(current_i),
            "current_ts":    _ts_iso(df, current_i),
            "wall_price":    float(wall["price"]) if wall["price"] is not None else None,
            "wall_ts":       wall["ts"],
            "last_event_ts": last_event_ts,
        }
        if lo > hi:
            out["reason"] = "no_interior_window"
            return out

        # Mirror the resolver: SELECT from the shared full-df swing pool
        # (swings_lb3, lb-3 + ATR filtered) by absolute index range, exactly as
        # _resolve_placeholder now does. Same-type swings in (leg_start_idx,
        # current_i] == (lo-1, hi]; lo == leg_start_idx+1, hi == current_i.
        target_type = 'high' if side == 'ceiling' else 'low'
        leg_swings_typed = [
            s for s in swings_lb3
            if s['type'] == target_type and (lo - 1) < s['idx'] <= hi
        ]
        out["leg_lb3_swings"] = len(leg_swings_typed)

        if leg_swings_typed:
            # Should have been promoted by the resolver — record what it would
            # have picked. Reaching this branch means a bug.
            if side == 'ceiling':
                best = max(leg_swings_typed, key=lambda s: s['price'])
            else:
                best = min(leg_swings_typed, key=lambda s: s['price'])
            abs_idx = int(best['idx'])
            out["reason"] = "swings_present_but_not_promoted"
            out["candidate_idx"]   = int(abs_idx)
            out["candidate_ts"]    = _ts_iso(df, abs_idx)
            out["candidate_price"] = float(best['price'])
            return out

        # No leg-internal swing was promotable. Identify the rolling extreme
        # inside the leg and why it isn't an ATR-qualified lb-3 swing. The
        # resolver now uses FULL-DF neighbour windows (detect-once-then-pick),
        # so neighbour checks below run on the full df, not a leg slice.
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

        # Need SWING_LOOKBACK candles on each side WITHIN the full df.
        if ext_idx < SWING_LOOKBACK:
            out["reason"] = "left_edge_insufficient_in_df"
            out["left_candles_needed"]  = int(SWING_LOOKBACK)
            out["left_candles_present"] = int(ext_idx)
            return out
        if ext_idx + SWING_LOOKBACK >= n:
            out["reason"] = "right_edge_insufficient_in_df"
            out["right_candles_needed"]  = int(SWING_LOOKBACK)
            out["right_candles_present"] = int(n - 1 - ext_idx)
            return out

        # Full-df neighbour windows around the extreme.
        l_lo = ext_idx - SWING_LOOKBACK
        l_hi = ext_idx                       # exclusive
        r_lo = ext_idx + 1
        r_hi = ext_idx + SWING_LOOKBACK + 1  # exclusive

        blocker_side = None
        blocker_idx  = None
        blocker_val  = None
        if side == 'ceiling':
            for j in range(l_lo, l_hi):
                if H_arr[j] >= ext_price:
                    blocker_side = "left"; blocker_idx = j; blocker_val = float(H_arr[j]); break
            if blocker_idx is None:
                for j in range(r_lo, r_hi):
                    if H_arr[j] >= ext_price:
                        blocker_side = "right"; blocker_idx = j; blocker_val = float(H_arr[j]); break
        else:
            for j in range(l_lo, l_hi):
                if L_arr[j] <= ext_price:
                    blocker_side = "left"; blocker_idx = j; blocker_val = float(L_arr[j]); break
            if blocker_idx is None:
                for j in range(r_lo, r_hi):
                    if L_arr[j] <= ext_price:
                        blocker_side = "right"; blocker_idx = j; blocker_val = float(L_arr[j]); break

        if blocker_idx is not None:
            out["reason"] = "blocked_by_neighbour"
            out["blocker_side"]  = blocker_side
            out["blocker_idx"]   = int(blocker_idx)
            out["blocker_ts"]    = _ts_iso(df, blocker_idx)
            out["blocker_value"] = blocker_val
        else:
            # Extreme passes lb-3 geometry on the full df, yet wasn't promoted.
            # With detect-once-then-pick the only remaining reason is the ATR
            # leg-size filter: the extreme IS a geometric pivot but its leg was
            # too small, so it never entered the shared pool.
            out["reason"] = "geometric_swing_dropped_by_atr_filter"

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

    # --- Persisted swing list (SINGLE SOURCE OF TRUTH for chart markers) ----
    # The ONE filtered swing pool (lb-3 + ATR) that drove trend / CHoCH / BOS /
    # walls in this walk, emitted so every chart consumes the SAME swings rather
    # than re-detecting. Keyed by ts (window-independent). `broken` is True when
    # this swing's ts matches a structural event's broken_swing_ts in the ring
    # (any kind: Major/Minor BOS, Major/Minor CHoCH) -> rendered as an X; all
    # others render as triangles. No chart calls a swing detector of its own.
    _broken_ts = {e.get("broken_swing_ts") for e in events_ring if e.get("broken_swing_ts")}
    swings_persisted = [
        {
            "ts":     s["ts"],
            "type":   s["type"],
            "price":  float(s["price"]),
            "broken": s["ts"] in _broken_ts,
        }
        for s in swings_lb3 if s.get("ts")
    ]

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
        "trend_start_ts":         _ts_iso(df, trend_start_idx) if 0 <= trend_start_idx < n else None,
        "last_minor_bos_pivot_ts": last_minor_bos_pivot_ts,
        "events":                 events_ring,
        "swings":                 swings_persisted,
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

    pair_name = (pair_conf or {}).get("name") if pair_conf else None

    # Debug-module override hook (sidecar, one-shot, read-and-delete).
    # If an operator has issued force_new_range for this pair, trim df to
    # candles >= force_from_ts and force a cold rebuild. Wrapped — any
    # failure is silently ignored so Phase 1 cannot break on debug code.
    if _debug_overrides is not None and pair_name:
        try:
            ov = _debug_overrides.consume_override(pair_name)
            if ov:
                trimmed = _trim_df_from(df, ov.get("force_new_range_from"))
                if trimmed is not None and len(trimmed) >= (SWING_LOOKBACK * 2 + 5):
                    df = trimmed
                    prior_state = None  # force cold-start rebuild from this point
        except Exception:
            pass

    # For cold-start cap the window. After cold-start, the full df is fine
    # because incremental walk only processes new candles.
    is_cold = _needs_cold_start(prior_state)
    if is_cold:
        df_used = df.tail(COLDSTART_WINDOW_H1).copy().reset_index(drop=True) \
                  if len(df) > COLDSTART_WINDOW_H1 else df
        return _walk_forward(df_used, prior_state=None, pair_name=pair_name)
    return _walk_forward(df, prior_state=prior_state, pair_name=pair_name)


def _trim_df_from(df, force_from_iso: Optional[str]):
    """Return df sliced to rows with ts >= force_from_iso. None if invalid.

    Defensive: if the ts column isn't recognisable or no rows match, returns
    None and the caller falls back to normal flow. Uses _ts_iso so this stays
    consistent with how Phase 1 reads timestamps elsewhere.
    """
    if not force_from_iso:
        return None
    try:
        n = len(df)
        keep_from = None
        for i in range(n):
            ts = _ts_iso(df, i)
            if ts and ts >= force_from_iso:
                keep_from = i
                break
        if keep_from is None:
            return None
        return df.iloc[keep_from:].copy().reset_index(drop=True)
    except Exception:
        return None


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

    # STAGE 1: prefer the H4-derived dealing range when present and valid. This
    # is the LIVE range (broken-wall live tracking applied) — used for display,
    # containment, and pd_position. The CHoCH 25% gate must NOT read this; it
    # reads the frozen `confirmed_*` block via compute_pd_confirmed() instead.
    # Legacy walls remain the fallback until the old engine is removed (Stage 3).
    h4 = walls.get("h4_range")
    if isinstance(h4, dict) and h4.get("valid") and h4.get("ceiling") is not None \
            and h4.get("floor") is not None and h4["ceiling"] > h4["floor"]:
        c = float(h4["ceiling"]); f = float(h4["floor"])
        width = c - f
        pos = (float(price) - f) / width if width > 0 else None
        # "tentative" now means: a wall is riding the live extreme (broken),
        # i.e. not a confirmed swing. Surfaced for labelling, mirrors the old
        # placeholder semantics so downstream consumers keep working unchanged.
        cbrk = bool(h4.get("ceiling_broken"))
        fbrk = bool(h4.get("floor_broken"))
        return {
            "valid": True,
            "range_high": c,
            "range_low":  f,
            "equilibrium": (c + f) / 2.0,
            "pd_position": pos,
            "ceiling_is_placeholder": cbrk,
            "floor_is_placeholder":   fbrk,
            "tentative":              cbrk or fbrk,
            "fallback_active":        False,
            "source": "h4_live",
        }

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


def compute_pd_confirmed(price: float, walls: Dict[str, Any]) -> Dict[str, Any]:
    """STABLE premium/discount reference for the CHoCH 25% gate (Stage 2).

    Reads ONLY the H4 `confirmed_*` range (both walls are confirmed swings, NO
    broken-wall live tracking). A moving range makes the 25% gate jitter and
    fire false CHoCHs, so the gate must read this frozen range — never the live
    one from compute_pd_position.

    Returns:
      {
        "valid":       bool,    # confirmed range present and non-degenerate
        "range_high":  float,
        "range_low":   float,
        "equilibrium": float,
        "pd_position": float|None,  # 0=floor, 1=ceiling
        "premium_floor":  float,    # >= this = top 25% (premium gate line)
        "discount_ceil":  float,    # <= this = bottom 25% (discount gate line)
        "source":      str,
      }
    """
    h4 = (walls or {}).get("h4_range")
    if not (isinstance(h4, dict) and h4.get("confirmed_valid")
            and h4.get("confirmed_ceiling") is not None
            and h4.get("confirmed_floor") is not None
            and h4["confirmed_ceiling"] > h4["confirmed_floor"]):
        return {"valid": False, "source": "no_confirmed_h4", "pd_position": None}
    c = float(h4["confirmed_ceiling"]); f = float(h4["confirmed_floor"])
    width = c - f
    pos = (float(price) - f) / width if width > 0 else None
    return {
        "valid": True,
        "range_high": c,
        "range_low":  f,
        "equilibrium": (c + f) / 2.0,
        "pd_position": pos,
        "premium_floor": f + PREMIUM_PCT * width,   # top 25% line
        "discount_ceil": f + DISCOUNT_PCT * width,  # bottom 25% line
        "source": "h4_confirmed",
    }
