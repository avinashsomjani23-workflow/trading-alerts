"""
Dealing Range / H1 Structure — single source of truth for H1 swing structure
(trend, BOS, CHoCH) and the swing pool every consumer reads.

Plain English:

The H1 trend is read directly from swing structure: higher-highs + higher-lows
= up, lower-highs + lower-lows = down. Two structural events are detected:

  - BOS   (Break of Structure)   — a trend-direction swing is taken out with
                                    displacement. Continuation.
  - CHoCH (Change of Character)   — the defended swing (last HL in an uptrend /
                                    last LH in a downtrend) is closed through
                                    against the trend by >= the CHoCH
                                    displacement. ARMS a pending reversal; the
                                    trend flips only on a later Confirmation BOS.

Dealing-range WALLS are not computed here. They come from the H4 swing
high / low (`h4_range.py`). This module only consumes the H4 range to (a) gate
CHoCH by premium / discount and (b) tag a BOS as 'Range' when it breaks an H4
wall.

Single swing definition (LOCKED): a confirmed swing must pass BOTH the
lookback=3 geometry test (detect_swings) AND the ATR leg-size filter
(_filter_swings_by_leg_atr). Every consumer — trend, BOS, CHoCH, charts, sweep
scoring, h4_range — reads this one pool. One definition, one filter.

CHoCH premium / discount (LOCKED): the reversal extreme's position in the FROZEN
confirmed H4 range (top 25% / bottom 25%) is recorded as a QUALITY TAG
(reversal_pct / choch_from_zone) only — it is NOT a gate. A CHoCH arms regardless
of zone. Read via compute_pd_confirmed(), never the live range, so the tag does
not jitter.

CHoCH confirmation (LOCKED): a CHoCH is PENDING when it arms — state stays the
old trend. It CONFIRMS (flips the trend) only when a Confirmation BOS breaks the
first post-CHoCH swing in the pending direction by >= bos_disp. It is CANCELLED
(reverts, no flip) only if price closes back past the origin extreme (reclaim).
There is NO timeout and NO lock distance. A reverted direction cannot re-arm a
CHoCH until a fresh confirmed swing forms in that direction (re-arm guard).

Detection engine: compute_structure(). It is PURE and recomputes from the full
H1 df every scan — there is no incremental wall state to corrupt. It emits an
event ring (last EVENT_RING_MAX BOS/CHoCH events) consumed by smc_radar to
build OBs and by Phase 2 (trend + BOS sequence count).

Phase 1 (smc_radar) is the only writer of `state/structure_state.json`. Phase 2
is read-only and consumes:
    dealing_range.compute_pd_position(price, walls)   # dealing-range geometry
One writer (Phase 1), many readers. Atomic state writes (temp + rename).
"""

import json
import os
from typing import Optional, Tuple, List, Dict, Any

import schema as _schema  # schema_version stamp/check for state files (Wave 1 item 1C)

# --- Tunables (locked) -------------------------------------------------------

# Single swing pool. lookback=3 swings drive the H1 trend, BOS, and CHoCH
# detection. There is no separate pool per event type — event geometry (not a
# distinct pool) distinguishes BOS from CHoCH.
SWING_LOOKBACK = 3

# Minimum swing leg size in ATR(14) units, applied AFTER lookback=3 geometric
# detection. A confirmed pivot is only kept if the leg into it (distance from
# the previous kept opposite-type swing) is >= MIN_LEG_ATR_MULT * the average
# H1 ATR across that leg. This is the SINGLE definition of an H1 swing: every
# consumer (trend, CHoCH, BOS, sweep scoring, charts, h4_range) reads swings
# that have passed BOTH the lb-3 geometry gate AND this ATR leg-size gate. Tiny
# triangles never become structural swings. Owned here (lowest layer);
# smc_detector.get_swing_points calls down into _filter_swings_by_leg_atr so
# there is exactly one implementation.
MIN_LEG_ATR_MULT = 1.5

# BOS displacement gate: the break candle's CLOSE must clear the broken swing
# by >= BOS_ATR_MULT * H1 ATR. (The CHoCH displacement gate is separate —
# STRUCTURE_CHOCH_ATR_MULT, defined with the v2 engine below.)
BOS_ATR_MULT = 0.4

# BODY GATE (replaces distance as the strength test). A break only fires if the
# CANDLE BODY (|close-open|) of the break candle OR the candle right after it
# (whichever is larger) is >= the event's body multiple * H1 ATR. Body measures
# displacement/conviction; distance-past-the-level does not. The close-beyond
# check (BOS_ATR_MULT / STRUCTURE_CHOCH_ATR_MULT) is RETAINED as the validity
# trigger (did a body close past the swing at all); the body gate is the added
# strength filter. Window = [break, break+1]; at the live edge (no break+1 bar
# yet) the gate uses the break candle alone. See STRUCTURE_CHOCH_BODY_ATR_MULT.
BOS_BODY_ATR_MULT = 1.0

# Premium / discount thresholds for the CHoCH zone gate. A CHoCH is valid
# only if the reversal high (down CHoCH) sits in the top 25% of the dealing
# range, or the reversal low (up CHoCH) sits in the bottom 25%.
PREMIUM_PCT  = 0.75
DISCOUNT_PCT = 0.25

# Event ring — last N qualified events kept on state for downstream readers
# (Phase 2 BOS-sequence count, zone invalidation). Cap chosen well above
# any realistic per-trend BOS run (caution thresholds are 3–5).
EVENT_RING_MAX = 20

# State file path. Lives in a dedicated directory outside any purge scope.
STATE_DIR  = "state"
STATE_PATH = os.path.join(STATE_DIR, "structure_state.json")


# --- ATR (local copy to avoid import cycle) ----------------------------------

def _compute_atr(df, period: int = 14) -> Optional[float]:
    """ATR(14). Single source of truth is smc_detector.compute_atr (which is
    memoised). We delegate to it via a LAZY import — a top-level import would be
    circular (smc_detector imports this module at load time). The previous
    duplicated body has been removed so there is one ATR implementation, one
    value. A raw fallback below preserves "never raises" if the import ever
    fails, and computes the identical mean-of-last-`period`-TR.
    """
    if df is None or len(df) < period + 1:
        return None
    try:
        import smc_detector as _sd
        return _sd.compute_atr(df, period=period)
    except Exception:
        # Defensive raw fallback (import path broken). Identical math to
        # smc_detector._atr_compute_raw: mean of the last `period` true ranges.
        try:
            H = df['High'].values.astype(float)
            L = df['Low'].values.astype(float)
            C = df['Close'].values.astype(float)
            trs = []
            for i in range(1, len(C)):
                trs.append(max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1])))
            if len(trs) < period:
                return None
            return sum(trs[-period:]) / period
        except Exception:
            return None


# --- Atomic JSON I/O ---------------------------------------------------------

def _ensure_state_dir():
    if not os.path.isdir(STATE_DIR):
        os.makedirs(STATE_DIR, exist_ok=True)


def load_state() -> Dict[str, Any]:
    """Load structure_state.json. Returns empty dict on any failure.

    Schema check (Wave 1 item 1C): a present-but-mismatched schema_version
    raises SchemaVersionError (fail-loud) rather than letting a misread file
    drive detection. A MISSING version is treated as v1 (deploy-safe). The
    SchemaVersionError is deliberately NOT swallowed by the except below — it
    must propagate so the scan stops red.
    """
    try:
        with open(STATE_PATH, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return _schema.check(data, name="structure_state.json")


def save_state(state: Dict[str, Any]) -> None:
    """Atomic write: temp file then rename. Same pattern used elsewhere.

    Stamps the current schema_version before writing (Wave 1 item 1C).
    """
    _ensure_state_dir()
    _schema.stamp(state)
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


def _iso_source(df):
    """Return the tz/column datetime source for df (DatetimeIndex), or None.

    Same source precedence as _ts_iso: 'Datetime' column -> 'Date' column ->
    DatetimeIndex. None when no datetime source exists. Centralises the lookup
    so the batched stamper and _ts_iso agree exactly.
    """
    if df is None:
        return None
    try:
        import pandas as _pd  # lazy import, matches module style
        cols = getattr(df, 'columns', [])
        if 'Datetime' in cols:
            return _pd.DatetimeIndex(_pd.to_datetime(df['Datetime']))
        if 'Date' in cols:
            return _pd.DatetimeIndex(_pd.to_datetime(df['Date']))
        if isinstance(df.index, _pd.DatetimeIndex):
            return df.index
        return None
    except Exception:
        return None


def _iso_for_indices(df, indices) -> Optional[List[Optional[str]]]:
    """Vectorized ISO timestamps for the given positional `indices`, or None
    when no datetime source exists.

    PERF: resolves only the requested rows (the handful of emitted swings) in a
    single boxing pass via DatetimeIndex.take, instead of one boxed-Timestamp
    access per swing. Result entry k == _ts_iso(df, indices[k]); same datetime
    source precedence, same "no source => None (list-level)" contract. Returns
    None (not a list) only when there is no datetime source at all, so the
    caller falls back to per-row _ts_iso and behaviour never changes.
    """
    src = _iso_source(df)
    if src is None:
        return None
    try:
        taken = src.take(list(int(i) for i in indices))
        return [t.isoformat() if t is not None and hasattr(t, 'isoformat') else None
                for t in taken]
    except Exception:
        return None


# --- Swing detection ---------------------------------------------------------

def _true_range_series(df) -> List[float]:
    """Per-candle True Range, length == len(df). Element 0 is NaN (no prior
    close). Pure-python True Range; lives here (lowest layer) so the ATR
    leg-size filter has no cross-import. smc_detector delegates to it.
    """
    import numpy as _np
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    C = df['Close'].values.astype(float)
    n = len(df)
    # Vectorized True Range (PERF; bit-identical to the per-bar max loop).
    # Element 0 stays NaN (no prior close). For i>=1, TR = max(H-L, |H-prevC|,
    # |L-prevC|) computed as a single numpy reduce over the three components.
    tr = _np.full(n, _np.nan)
    if n > 1:
        prevC = C[:-1]
        tr[1:] = _np.maximum.reduce([H[1:] - L[1:],
                                     _np.abs(H[1:] - prevC),
                                     _np.abs(L[1:] - prevC)])
    return tr.tolist()


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
                  min_leg_atr_mult: Optional[float] = MIN_LEG_ATR_MULT,
                  right_lookback: Optional[int] = None
                  ) -> List[Dict[str, Any]]:
    """
    Find confirmed swing highs and swing lows over the entire df.

    A candle at idx i is:
      - a swing high if H[i] is STRICTLY GREATER than every other high in
        the window [i-lookback, i+right_lookback].
      - a swing low  if L[i] is STRICTLY LESS than every other low in
        the window [i-lookback, i+right_lookback].

    Strict comparison: equal highs / equal lows do NOT register as swings.
    A flat top / flat bottom across the window correctly produces no swing.

    right_lookback (default = lookback): how many bars to the RIGHT a pivot
    must dominate to be detected. The left side stays = lookback (the real
    pivot geometry). A SMALLER right_lookback lets the most recent pivot be
    seen sooner (fewer bars must print after it) — used by h4_range to make
    the newest dealing-range wall appear earlier without changing the level a
    fully-confirmed pivot resolves to. Symmetric (right=left) is the H1
    default and is unchanged.

    After geometric detection, the ATR leg-size filter is applied (unless
    min_leg_atr_mult is None or <= 0). This is the single H1 swing definition:
    lb-3 geometry PLUS a leg that is large enough in ATR terms. Tiny triangles
    are removed. Pass min_leg_atr_mult=None ONLY for non-H1 / diagnostic use
    where the H1-tuned multiple does not apply.

    Returns list sorted by idx, each entry:
      {'type': 'high'|'low', 'idx': i, 'price': float, 'ts': iso}
    """
    rb = lookback if right_lookback is None else right_lookback
    if df is None or len(df) < lookback + rb + 1:
        return []
    import numpy as _np
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    n = len(df)

    # --- Vectorized geometry (PERF; byte-identical to the old per-bar loop) ---
    # A candle i is a swing high iff H[i] is STRICTLY greater than every
    # neighbour in [i-lookback, i+rb] (excluding i); swing low symmetric. We
    # build the per-index neighbour max/min with `lookback`+`rb` shifted
    # numpy comparisons (both are tiny: 1 or 3) instead of a Python max()/min()
    # over slices on every bar. The emit window [lookback, n-rb) and the STRICT
    # `>` / `<` comparison are preserved exactly. Verified identical (incl. the
    # right_lookback=1 early pass) across EURUSD / NZDUSD / GOLD up to 80k bars.
    neg_inf = -_np.inf
    pos_inf = _np.inf
    left_max_h = _np.full(n, neg_inf); right_max_h = _np.full(n, neg_inf)
    left_min_l = _np.full(n, pos_inf); right_min_l = _np.full(n, pos_inf)
    for d in range(1, lookback + 1):
        left_max_h[d:] = _np.maximum(left_max_h[d:], H[:-d])
        left_min_l[d:] = _np.minimum(left_min_l[d:], L[:-d])
    for d in range(1, rb + 1):
        right_max_h[:-d] = _np.maximum(right_max_h[:-d], H[d:])
        right_min_l[:-d] = _np.minimum(right_min_l[:-d], L[d:])
    neigh_max_h = _np.maximum(left_max_h, right_max_h)
    neigh_min_l = _np.minimum(left_min_l, right_min_l)
    valid = _np.zeros(n, dtype=bool)
    valid[lookback: n - rb] = True
    is_high = (H > neigh_max_h) & valid
    is_low  = (L < neigh_min_l) & valid
    hi_idx = _np.nonzero(is_high)[0]
    lo_idx = _np.nonzero(is_low)[0]

    # Stamp ISO timestamps ONLY for emitted swing indices, in one boxing pass
    # (PERF). The old code called _ts_iso(df, i) per swing (one boxed Timestamp
    # each); here we resolve all emitted indices via one vectorized lookup. The
    # union of indices preserves the original emit order (idx-sorted, high
    # before low at the same idx — matched by the merge below).
    all_idx = _np.unique(_np.concatenate([hi_idx, lo_idx])) if (len(hi_idx) or len(lo_idx)) else _np.array([], dtype=int)
    iso_by_idx: Dict[int, Optional[str]] = {}
    if len(all_idx):
        iso_vals = _iso_for_indices(df, all_idx)
        if iso_vals is not None:
            iso_by_idx = {int(ix): iso_vals[k] for k, ix in enumerate(all_idx)}
        else:
            # No datetime source — fall back to the exact per-row path.
            iso_by_idx = {int(ix): _ts_iso(df, int(ix)) for ix in all_idx}

    out = []
    hi_set = set(int(i) for i in hi_idx)
    lo_set = set(int(i) for i in lo_idx)
    for i in sorted(hi_set | lo_set):
        # Preserve original per-i append order: high appended before low.
        if i in hi_set:
            out.append({'type': 'high', 'idx': i, 'price': float(H[i]), 'ts': iso_by_idx.get(i)})
        if i in lo_set:
            out.append({'type': 'low',  'idx': i, 'price': float(L[i]), 'ts': iso_by_idx.get(i)})
    if min_leg_atr_mult is not None and min_leg_atr_mult > 0:
        out = _filter_swings_by_leg_atr(out, df, min_mult=min_leg_atr_mult)
    return out


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


# ---------------------------------------------------------------------------
# H1 Structure Engine (v2) — single CHoCH / BOS / trend model.
#
# Locked spec (decided with the trader):
#   - One trend state: up | down | undefined (undefined only before birth).
#   - CHoCH ARMS a pending reversal; it does NOT flip the trend on its own candle.
#     The trend flips only on a Confirmation BOS (sec. 2c).
#   - Premium/discount is a QUALITY TAG only, never a flip gate.
#   - Pending CHoCH cancels = close back past the origin extreme (reclaim). There
#     is NO lock distance and NO timeout (decided with the trader).
#   - Re-arm guard: a reverted/invalidated CHoCH direction cannot re-arm until a
#     fresh confirmed swing forms in the reverted trend direction.
# ---------------------------------------------------------------------------

# CHoCH displacement (H1 ATR units). A close must clear the defended swing
# by >= this to count as a CHoCH. 1.0 ATR chosen on replay.
STRUCTURE_CHOCH_ATR_MULT  = 1.0

# CHoCH body gate (see BOS_BODY_ATR_MULT). A CHoCH flips the whole bias, so it
# demands more force than a continuation BOS — 1.5 ATR (ICT displacement default).
STRUCTURE_CHOCH_BODY_ATR_MULT = 1.5

# DEAD knob (retained for signature / golden-test compatibility, intentionally
# unused — see compute_structure: `_ = lock_atr_mult`). The old lock-distance
# confirmation was replaced by the Confirmation-BOS model: a pending CHoCH now
# confirms only on a BOS and cancels only on reclaim, with no lock and no timeout.
STRUCTURE_LOCK_ATR_MULT   = 1.5

# Ranging flag: trend intact but no trend-direction swing extended for at
# least this many confirmed swings. Informational only.
STRUCTURE_RANGING_STALE   = 2


def _structure_confirm_idx(swing: Dict[str, Any], lookback: int) -> int:
    """Candle index at which a swing becomes known (lookback confirmation lag)."""
    return int(swing["idx"]) + lookback


def compute_structure(df, h4_range: Optional[Dict[str, Any]],
                      lookback: int = SWING_LOOKBACK,
                      _min_leg_atr_mult: Optional[float] = MIN_LEG_ATR_MULT,
                      choch_atr_mult: Optional[float] = None,
                      lock_atr_mult: Optional[float] = None,
                      _trace: Optional[list] = None) -> Dict[str, Any]:
    """Compute H1 swing structure (v2 engine). Pure; never raises on thin data.

    Trend is determined solely by swing structure: HH+HL = up, LH+LL = down.
    There is no wall-break or event-based trend detection. The event ring
    emitted here feeds detect_smc_radar (OB building) and Phase 2 (trend +
    BOS sequence count) — replacing the legacy dealing_range wall engine.

    Event ring schema (matches detect_smc_radar expectations exactly):
      type:               'BOS' | 'CHoCH'
      tier:               'BOS' | 'Range' for BOS events ('Range' = H4 wall
                          break); an internal placeholder for CHoCH (never
                          surfaced — there is NO Major/Minor in v2; the only
                          user-facing event types are BOS / Range BOS / CHoCH).
      direction:          'bullish' | 'bearish'   (direction of the move)
      candle_ts:          iso — the bar on which the event fired
      impulse_start_ts:   iso — start of the impulse leg (for OB walk-back)
      broken_swing_price: float — the swing level that was broken
      broken_was_wall:    False  (no walls in v2; level is a confirmed swing)
      reversal_pct:       1.0 if reversed from premium/discount zone, else 0.0
      trend_after:        'bullish' | 'bearish' — trend after this event

    Returns:
      {
        'state':            'up'|'down'|'undefined',
        'ranging':          bool,
        'prior_trend':      'up'|'down'|None,
        'flip_unconfirmed': bool,
        'choch':            bool,
        'choch_ts':         str|None,
        'choch_level':      float|None,
        'choch_from_zone':  bool,
        'choch_flip_count': int,
        'defended':         float|None,
        'broken_swing_ts':  str|None,
        'last_bos':         dict|None,
        'label':            str,
        'events':           list,   # event ring (last EVENT_RING_MAX entries)
        'trend':            str|None,  # 'bullish'|'bearish'|None (Phase 2 compat)
        'swings':           list,   # confirmed swings for chart/debug
      }
    """
    _UP   = "up"
    _DOWN = "down"
    _UNDEF = "undefined"

    empty = {"state": _UNDEF, "ranging": False, "prior_trend": None,
             "flip_unconfirmed": False, "choch": False, "choch_ts": None,
             "choch_level": None, "choch_from_zone": False, "choch_flip_count": 0,
             "defended": None, "broken_swing_ts": None, "last_bos": None,
             "label": "H1 structure undefined (insufficient data)",
             "events": [], "trend": None, "swings": []}
    if df is None or len(df) < (lookback * 2 + 5):
        return empty

    atr = _compute_atr(df)
    if atr is None or atr <= 0:
        return empty

    swings: List[Dict[str, Any]] = detect_swings(
        df, lookback=lookback, min_leg_atr_mult=_min_leg_atr_mult)
    if not swings:
        return empty

    closes = df["Close"].to_numpy(dtype=float)
    opens  = df["Open"].to_numpy(dtype=float)
    n = len(df)

    # ---- Session-gap detection (shared, time-based) ------------------------
    # A break "gap-open guard" must reject ONLY a price teleport across a level
    # that happened over a SESSION/WEEKEND gap (no candle traded the in-between
    # prices). It must NOT reject a gradual continuous slide through the level
    # (price drifted there candle-by-candle — that IS real trading).
    # The two are distinguished by TIME, not price: a weekend/holiday gap shows
    # as a multi-hour jump between consecutive H1 bars, while a continuous candle
    # (even a violent 50-pip news bar) is exactly one bar-interval after the
    # prior. `is_session_gap[ci]` is True iff bar ci is separated from bar ci-1
    # by MORE than the bar interval (so no continuous candle traded between them).
    # Interval is derived from the data (median delta) so it self-adjusts and is
    # never hardcoded. Index 0 is False (no prior bar). `gaps_known` records
    # whether we could compute this at all: if timestamps are missing/odd we
    # CANNOT prove continuity, so `_traded_through` falls back to the original
    # open-side guard on every candle (conservative — never loosens what worked).
    is_session_gap = [False] * n
    gaps_known = False
    try:
        import pandas as _pd
        if 'Datetime' in df.columns:
            _ts_vals = df['Datetime']
        elif 'Date' in df.columns:
            _ts_vals = df['Date']
        else:
            _ts_vals = df.index.to_series()
        # Normalise to int64 nanoseconds (tz-aware safe). NaT -> abort below.
        # .values -> datetime64[ns] (UTC instants) -> int64 ns; works across
        # pandas versions (Series.view was removed in pandas 3).
        _dt = _pd.to_datetime(_ts_vals, utc=True, errors='coerce')
        _ts_ns = _dt.values.astype('datetime64[ns]').astype('int64')  # ns epoch
        if len(_ts_ns) == n and n >= 3 and not _pd.isna(_dt).any():
            _deltas = _ts_ns[1:] - _ts_ns[:-1]
            _pos = _deltas[_deltas > 0]
            if len(_pos):
                import numpy as _np
                _bar_ns = float(_np.median(_pos))
                # Tolerance 1.5x the bar interval: a normal bar is exactly 1x;
                # a weekend gap is >=24x. 1.5x cleanly separates without being
                # tripped by DST shifts (1h) or minor feed jitter.
                _gap_thresh = 1.5 * _bar_ns
                for _i in range(1, n):
                    if float(_ts_ns[_i] - _ts_ns[_i - 1]) > _gap_thresh:
                        is_session_gap[_i] = True
                gaps_known = True
    except Exception:
        is_session_gap = [False] * n
        gaps_known = False

    def _traded_through(break_idx: int, level: float, side: str) -> bool:
        """Did price TRADE through `level` on candle break_idx (vs teleport over
        a session gap)? `side`: 'down' = a downward break (close below level),
        'up' = an upward break (close above level).

        Continuous candle (gaps known, not a session gap): always True — a
        decisive close beyond the level means price traded through it, whether in
        one candle or as the last step of a gradual slide. Session-gap candle (or
        gaps NOT known — conservative fallback): require the OPEN to be on the
        NEAR side (open >= level for a down-break / open <= level for an up-break)
        so a pure weekend teleport across the level cannot count; a gap candle
        that itself opened on the near side and closed through is real intra-
        candle trading and qualifies. The fallback path reproduces the ORIGINAL
        open-side guard exactly, so behaviour is never loosened when continuity
        cannot be proven.

        SINGLE definition of the gap-open guard — every BOS / CHoCH / Confirmation
        break calls this so the rule can never diverge between event types.
        """
        if gaps_known and not is_session_gap[break_idx]:
            return True
        if side == 'down':
            return opens[break_idx] >= level
        return opens[break_idx] <= level

    def _body_gate_ok(break_idx: int, body_mult: float) -> bool:
        """Body gate (option B): the break candle OR the candle right after it
        must have a body (|close-open|) >= body_mult * atr. At the live edge
        (no break+1 bar) the gate uses the break candle alone. Fails open if
        atr is missing/non-positive (never silently drop a structural break)."""
        if atr is None or atr <= 0:
            return True
        floor = body_mult * atr
        b0 = abs(closes[break_idx] - opens[break_idx])
        if b0 >= floor:
            return True
        nxt = break_idx + 1
        if nxt < n and abs(closes[nxt] - opens[nxt]) >= floor:
            return True
        return False

    price_now = float(closes[-1])
    pdc = compute_pd_confirmed(price_now, {"h4_range": h4_range or {}})
    gate_valid    = bool(pdc.get("valid"))
    premium_floor = pdc.get("premium_floor")
    discount_ceil = pdc.get("discount_ceil")

    # H4 dealing range walls for Range-BOS tagging.
    # A BOS whose broken swing is within BOS_ATR_MULT * ATR of the H4
    # ceiling (bullish) or H4 floor (bearish) is tagged tier='Range'.
    _h4 = h4_range or {}
    h4_ceiling: Optional[float] = _h4.get("ceiling")
    h4_floor:   Optional[float] = _h4.get("floor")

    bos_disp = BOS_ATR_MULT * atr

    by_known: Dict[int, List[Dict[str, Any]]] = {}
    for s in swings:
        by_known.setdefault(_structure_confirm_idx(s, lookback), []).append(s)

    H = df["High"].to_numpy(dtype=float)
    L = df["Low"].to_numpy(dtype=float)

    state       = _UNDEF
    prior_trend: Optional[str]   = None
    defended:    Optional[float] = None
    # The swing dict whose price == `defended`, or None when `defended` was set
    # from a raw leg extreme (failure-window reversal). Travels with `defended`
    # so a CHoCH can report the EXACT swing it broke (by ts), not a guess.
    defended_swing: Optional[Dict[str, Any]] = None
    choch                        = False
    choch_ts:   Optional[str]   = None
    broken_swing_ts: Optional[str] = None
    last_bos:   Optional[Dict[str, Any]] = None
    choch_level:   Optional[float] = None
    choch_from_zone: bool          = False
    choch_flip_count               = 0
    choch_origin:  Optional[float] = None
    rearm_block_dir: Optional[str] = None

    highs: List[Dict[str, Any]] = []
    lows:  List[Dict[str, Any]] = []
    recent_low:  Optional[Dict[str, Any]] = None
    recent_high: Optional[Dict[str, Any]] = None
    trend_dir_swings_since_extend = 0
    leg_extreme_high: Optional[float] = None
    leg_extreme_low:  Optional[float] = None
    leg_start = 0

    # BOS break target (Option B — fire on-close, no swing-confirm lag).
    # In a downtrend `bos_break_low` is the most-recent CONFIRMED swing low; a
    # close below its price - bos_disp fires a BOS on that candle (mirrors how
    # CHoCH fires on-close vs `defended`). In an uptrend `bos_break_high` is the
    # most-recent confirmed swing high. After a BOS fires the target is cleared
    # (set None) so the same spent swing cannot re-fire; it re-seeds when the
    # next confirmed trend-direction swing forms. None = no valid target yet
    # (e.g. just after a CHoCH flip / birth, before the next swing confirms).
    bos_break_low:  Optional[Dict[str, Any]] = None
    bos_break_high: Optional[Dict[str, Any]] = None

    # CONFIRMATION-BOS model (locked with the trader 2026-06-23).
    # A CHoCH does NOT flip the trend on its own candle. It is PROVISIONAL: the
    # trend stays the OLD direction and the CHoCH only flips it once a BOS prints
    # in the CHoCH's direction (a "Confirmation BOS"). This removes the whole
    # class of "frozen unconfirmed flip" bugs — there is no premature flip to get
    # stuck in. State stays honest to confirmed structure.
    #
    # While a CHoCH is pending, these describe it (state is UNCHANGED — still the
    # old trend). `choch_pending_dir` is the direction the CHoCH is trying to flip
    # TO (always the opposite of `state`); kept explicit for readability though it
    # is derivable. `confirm_break_swing` is the FIRST confirmed swing that prints
    # AFTER the CHoCH candle in the pending direction (Option B) — the swing whose
    # break by >= bos_disp confirms the reversal:
    #   - CHoCH UP  pending (state DOWN) -> confirm on break of a swing HIGH
    #   - CHoCH DOWN pending (state UP)  -> confirm on break of a swing LOW
    # The pending CHoCH is CANCELLED (no flip) only if price closes back past
    # `choch_origin` (reclaim). There is NO timeout — a pending CHoCH persists
    # until confirmed-by-BOS or cancelled-by-reclaim (decided with the trader).
    # `prior_trend` / `choch_level` / `choch_origin` (declared above) are the
    # pending descriptors and gate `flip_unconfirmed` reporting.
    choch_pending_dir:   Optional[str] = None
    confirm_break_swing: Optional[Dict[str, Any]] = None
    choch_arm_idx:       Optional[int] = None   # candle index the CHoCH armed on
    # The CHoCH's own attributes, stashed at arm-time so the Confirmation BOS /
    # cancel paths can reference them (impulse start for OB walk-back, etc.).
    choch_impulse_ts:    Optional[str] = None
    choch_broken_ts:     Optional[str] = None

    # impulse_start_ts: the timestamp of the swing that anchors the start of
    # the current impulse leg (last swing in the trend direction before current
    # move). This is what detect_smc_radar uses to walk back and find the OB.
    impulse_start_ts: Optional[str] = None

    # Event ring — accumulated during the walk, trimmed to EVENT_RING_MAX.
    events_ring: List[Dict[str, Any]] = []

    def _break_body_atr(break_idx: Optional[int]) -> Optional[float]:
        """Window-aware break-candle body in ATR units — the SAME displacement
        measure the body gate fires on (`_body_gate_ok`) and that
        smc_detector.compute_break_quality grades. Window = max body of the break
        candle and the candle right after it (option B), exactly as both. Returns
        None when ATR or the index is unavailable (callers store None, and the
        continuation-decay read in compute_bos_sequence_count treats a missing
        body as 'cannot measure' rather than 'fading'). One body definition,
        three readers — no duplicated numbers."""
        if break_idx is None or atr is None or atr <= 0:
            return None
        i = int(break_idx)
        if i < 0 or i >= n:
            return None
        b0 = abs(closes[i] - opens[i])
        nxt = i + 1
        b1 = abs(closes[nxt] - opens[nxt]) if nxt < n else 0.0
        # Cast to plain float (closes/opens are numpy) — match the float() casts
        # on the other event fields; keeps persisted JSON clean and consumer math
        # numpy-free.
        return round(float(max(b0, b1)) / float(atr), 2)

    def _push_event(ev_type: str, direction: str, candle_ts: Optional[str],
                    broken_price: Optional[float], imp_start_ts: Optional[str],
                    from_zone: bool, trend_after: Optional[str],
                    tier: str = "BOS",
                    broken_swing_ts_arg: Optional[str] = None,
                    break_idx: Optional[int] = None) -> None:
        # Tier is a BOS sub-type only: 'BOS' (internal swing break) or 'Range'
        # (H4 dealing-range wall break). A CHoCH has NO BOS sub-tier — its tier
        # mirrors its type ('CHoCH'). The dead legacy value 'Major'/'Minor' must
        # never be emitted: it leaked into last_event_tier, the scan log and the
        # audit rows, and there is no Major/Minor in the v2 engine. Forcing the
        # tier here at the single emit point makes that leak structurally
        # impossible regardless of what any caller passes.
        if ev_type == "CHoCH":
            tier = "CHoCH"
        elif ev_type == "CHoCH_FAILED":
            # A failed CHoCH is NOT a BOS sub-type — never let it carry tier='BOS'
            # (the default), which would make last_event_tier / the scan log read
            # it as a continuation break. Tier mirrors the type, same rule as CHoCH.
            tier = "CHoCH_FAILED"
        ev = {
            "type":               ev_type,
            "tier":               tier,
            "direction":          direction,
            "candle_ts":          candle_ts,
            "impulse_start_ts":   imp_start_ts,
            "broken_swing_price": float(broken_price) if broken_price is not None else None,
            # ISO ts of the EXACT swing object that was broken, captured at
            # detection time. None when the broken level was a raw leg extreme
            # (failure-window reversal) rather than a confirmed swing — in that
            # case there is no swing to mark. This is the ONLY reliable handle
            # on the broken swing: price-matching is ambiguous (equal highs/lows
            # are common) and same-direction type rules are wrong (a CHoCH can
            # break either a high or a low depending on how `defended` was set).
            "broken_swing_ts":    broken_swing_ts_arg,
            "broken_was_wall":    True if tier == "Range" else False,
            "reversal_pct":       1.0 if from_zone else 0.0,
            "trend_after":        trend_after,
            "chop":               False,
            # Displacement (break-candle body in ATR) — same window-aware measure
            # as the body gate and compute_break_quality. Consumed by
            # compute_bos_sequence_count to judge continuation drive decay. None
            # when ATR/idx unavailable (treated as 'cannot measure', never 'fading').
            "break_body_atr":     _break_body_atr(break_idx),
        }
        events_ring.append(ev)
        while len(events_ring) > EVENT_RING_MAX:
            events_ring.pop(0)

    def _reversed_from_premium(idx_from: int, idx_to: int) -> bool:
        if not gate_valid or premium_floor is None or idx_from > idx_to:
            return False
        return float(H[idx_from:idx_to + 1].max()) >= premium_floor

    def _reversed_from_discount(idx_from: int, idx_to: int) -> bool:
        if not gate_valid or discount_ceil is None or idx_from > idx_to:
            return False
        return float(L[idx_from:idx_to + 1].min()) <= discount_ceil

    _mult     = choch_atr_mult if choch_atr_mult is not None else STRUCTURE_CHOCH_ATR_MULT
    choch_disp = _mult * atr
    # NOTE: lock_atr_mult / STRUCTURE_LOCK_ATR_MULT are RETAINED only for the
    # diagnostics knob-sweep harness signature. In the Confirmation-BOS model a
    # pending CHoCH has NO timeout/lock — it resolves only by Confirmation BOS or
    # reclaim — so this knob is now inert (no lock distance is computed/used).
    _ = lock_atr_mult  # intentionally unused (see note)

    for ci in range(n):
        c    = closes[ci]
        hi_i = float(H[ci])
        lo_i = float(L[ci])

        if leg_extreme_high is None or hi_i > leg_extreme_high:
            leg_extreme_high = hi_i
        if leg_extreme_low is None or lo_i < leg_extreme_low:
            leg_extreme_low = lo_i

        # ---- 1. PENDING CHoCH RESOLUTION ---------------------------------------
        # Only active while a CHoCH is pending (armed in sec. 2, not yet confirmed
        # or cancelled). `state` is STILL the old trend here. Two exits:
        #   (1a) RECLAIM cancel — price closes back past choch_origin -> the CHoCH
        #        failed; no flip (state was never changed). No timeout exists.
        #   (1b) CONFIRMATION BOS — a close breaks the first post-CHoCH confirmed
        #        swing in the pending direction by >= bos_disp -> the reversal is
        #        confirmed; flip state NOW and emit a 'Confirmation BOS' event.
        # A continuation BOS in the OLD direction (sec. 2b) is the third exit and
        # is handled there (down/up-trend extended => pending CHoCH failed).
        if choch_pending_dir is not None and choch_origin is not None \
                and choch_level is not None:
            if choch_pending_dir == _UP:
                # CHoCH UP pending, state is DOWN. Reclaim = close BELOW the
                # down-leg low (choch_origin). Confirm = close breaks a swing
                # HIGH upward by bos_disp.
                if c < choch_origin:
                    _failed_level = choch_level
                    prior_trend = None; choch = False
                    choch_origin = None; choch_level = None; choch_from_zone = False
                    choch_pending_dir = None; confirm_break_swing = None
                    choch_arm_idx = None
                    last_bos = {"kind": "CHoCH_FAILED", "direction": _DOWN,
                                "ts": _ts_iso(df, ci)}
                    # Resumed trend = DOWN. Chart: 'CHoCH failed (trend resumed)'.
                    _push_event("CHoCH_FAILED", "bearish", _ts_iso(df, ci),
                                _failed_level, impulse_start_ts, False, "bearish",
                                break_idx=ci)
                    if _trace is not None:
                        _trace.append(state)
                    continue
                elif (confirm_break_swing is not None
                      and c > confirm_break_swing["price"] + bos_disp
                      and _traded_through(ci, confirm_break_swing["price"], 'up')
                      and _body_gate_ok(ci, BOS_BODY_ATR_MULT)):
                    # CONFIRMATION BOS up — the CHoCH-up is confirmed. Flip to UP
                    # now and seed the new (up) trend exactly as the old CHoCH flip
                    # did. Emit a Confirmation BOS (type BOS, tier 'Confirm') that
                    # breaks the post-CHoCH swing high.
                    broken_price = confirm_break_swing["price"]
                    broken_ts    = confirm_break_swing["ts"]
                    ts_now = _ts_iso(df, ci)
                    state = _UP; prior_trend = None; choch = False
                    choch_origin = None; choch_level = None; choch_from_zone = False
                    choch_pending_dir = None; confirm_break_swing = None
                    choch_arm_idx = None
                    defended = leg_extreme_low if leg_extreme_low is not None else defended
                    defended_swing = None  # raw leg extreme until next HL confirms
                    leg_extreme_high = hi_i; leg_extreme_low = lo_i
                    leg_start = ci
                    impulse_start_ts = choch_impulse_ts or ts_now
                    rearm_block_dir = _DOWN
                    bos_break_high = None; bos_break_low = None
                    last_bos = {"kind": "BOS", "direction": _UP,
                                "ts": ts_now, "tier": "Confirm"}
                    _push_event("BOS", "bullish", ts_now, broken_price,
                                impulse_start_ts, False, "bullish", "Confirm",
                                broken_swing_ts_arg=broken_ts, break_idx=ci)
                    if _trace is not None:
                        _trace.append(state)
                    continue
            elif choch_pending_dir == _DOWN:
                # CHoCH DOWN pending, state is UP. Reclaim = close ABOVE the
                # up-leg high (choch_origin). Confirm = close breaks a swing LOW
                # downward by bos_disp.
                if c > choch_origin:
                    _failed_level = choch_level
                    prior_trend = None; choch = False
                    choch_origin = None; choch_level = None; choch_from_zone = False
                    choch_pending_dir = None; confirm_break_swing = None
                    choch_arm_idx = None
                    last_bos = {"kind": "CHoCH_FAILED", "direction": _UP,
                                "ts": _ts_iso(df, ci)}
                    # Resumed trend = UP. Chart: 'CHoCH failed (trend resumed)'.
                    _push_event("CHoCH_FAILED", "bullish", _ts_iso(df, ci),
                                _failed_level, impulse_start_ts, False, "bullish",
                                break_idx=ci)
                    if _trace is not None:
                        _trace.append(state)
                    continue
                elif (confirm_break_swing is not None
                      and c < confirm_break_swing["price"] - bos_disp
                      and _traded_through(ci, confirm_break_swing["price"], 'down')
                      and _body_gate_ok(ci, BOS_BODY_ATR_MULT)):
                    # CONFIRMATION BOS down — the CHoCH-down is confirmed. Flip to
                    # DOWN now and seed the new (down) trend. Emit a Confirmation
                    # BOS that breaks the post-CHoCH swing low.
                    broken_price = confirm_break_swing["price"]
                    broken_ts    = confirm_break_swing["ts"]
                    ts_now = _ts_iso(df, ci)
                    state = _DOWN; prior_trend = None; choch = False
                    choch_origin = None; choch_level = None; choch_from_zone = False
                    choch_pending_dir = None; confirm_break_swing = None
                    choch_arm_idx = None
                    defended = leg_extreme_high if leg_extreme_high is not None else defended
                    defended_swing = None  # raw leg extreme until next LH confirms
                    leg_extreme_high = hi_i; leg_extreme_low = lo_i
                    leg_start = ci
                    impulse_start_ts = choch_impulse_ts or ts_now
                    rearm_block_dir = _UP
                    bos_break_high = None; bos_break_low = None
                    last_bos = {"kind": "BOS", "direction": _DOWN,
                                "ts": ts_now, "tier": "Confirm"}
                    _push_event("BOS", "bearish", ts_now, broken_price,
                                impulse_start_ts, False, "bearish", "Confirm",
                                broken_swing_ts_arg=broken_ts, break_idx=ci)
                    if _trace is not None:
                        _trace.append(state)
                    continue

        # ---- 2. CHoCH (ARM — provisional, does NOT flip the trend) -------------
        # A CHoCH no longer flips `state`. It ARMS a pending reversal: the CHoCH
        # event is pushed (so its OB still builds — unchanged), but the trend
        # stays the OLD direction until a Confirmation BOS prints (section 2c).
        # `defended` / `leg_extreme_*` / `leg_start` / `impulse_start_ts` are NOT
        # reset here — they still belong to the live (old) trend. Gated by
        # `choch_pending_dir is None` so a second CHoCH cannot arm while one is
        # already pending.
        if state == _UP and defended is not None and rearm_block_dir != _DOWN \
                and choch_pending_dir is None:
            # Gap guard via _traded_through: a down-break of defended only counts
            # if price actually traded through it (continuous candle), not a
            # weekend teleport across it.
            if c < defended - choch_disp and _traded_through(ci, defended, 'down') \
                    and _body_gate_ok(ci, STRUCTURE_CHOCH_BODY_ATR_MULT):
                rev_idx = recent_high["idx"] if recent_high else leg_start
                choch = True
                ts_now = _ts_iso(df, ci)
                choch_ts = ts_now
                broken_defended_ts = defended_swing["ts"] if defended_swing else None
                broken_swing_ts = broken_defended_ts
                # Pending CHoCH DOWN (state still UP). Confirm on a BOS down
                # (break of a swing LOW). Reclaim = close back ABOVE choch_origin
                # (the up-leg high). State, defended, leg extremes UNCHANGED.
                choch_level  = defended
                choch_origin = leg_extreme_high
                choch_pending_dir = _DOWN
                choch_arm_idx = ci
                confirm_break_swing = None    # first post-CHoCH swing LOW (sec. 4)
                choch_from_zone = _reversed_from_premium(rev_idx, ci)
                prior_trend  = _UP
                choch_impulse_ts = impulse_start_ts
                choch_broken_ts  = broken_defended_ts
                choch_flip_count += 1
                last_bos = {"kind": "CHoCH", "direction": _DOWN,
                            "ts": ts_now, "from_zone": choch_from_zone}
                _push_event("CHoCH", "bearish", ts_now, choch_level,
                            impulse_start_ts, choch_from_zone, "bearish",
                            broken_swing_ts_arg=broken_defended_ts, break_idx=ci)
        elif state == _DOWN and defended is not None and rearm_block_dir != _UP \
                and choch_pending_dir is None:
            # Gap guard via _traded_through: an up-break of defended only counts
            # if price actually traded through it (continuous candle), not a
            # weekend teleport across it.
            if c > defended + choch_disp and _traded_through(ci, defended, 'up') \
                    and _body_gate_ok(ci, STRUCTURE_CHOCH_BODY_ATR_MULT):
                rev_idx = recent_low["idx"] if recent_low else leg_start
                choch = True
                ts_now = _ts_iso(df, ci)
                choch_ts = ts_now
                broken_defended_ts = defended_swing["ts"] if defended_swing else None
                broken_swing_ts = broken_defended_ts
                # Pending CHoCH UP (state still DOWN). Confirm on a BOS up (break
                # of a swing HIGH). Reclaim = close back BELOW choch_origin (the
                # down-leg low). State, defended, leg extremes UNCHANGED.
                choch_level  = defended
                choch_origin = leg_extreme_low
                choch_pending_dir = _UP
                choch_arm_idx = ci
                confirm_break_swing = None    # first post-CHoCH swing HIGH (sec. 4)
                choch_from_zone = _reversed_from_discount(rev_idx, ci)
                prior_trend  = _DOWN
                choch_impulse_ts = impulse_start_ts
                choch_broken_ts  = broken_defended_ts
                choch_flip_count += 1
                last_bos = {"kind": "CHoCH", "direction": _UP,
                            "ts": ts_now, "from_zone": choch_from_zone}
                _push_event("CHoCH", "bullish", ts_now, choch_level,
                            impulse_start_ts, choch_from_zone, "bullish",
                            broken_swing_ts_arg=broken_defended_ts, break_idx=ci)

        # ---- 2b. BOS (continuation, fire ON-CLOSE) -----------------------------
        # Symmetric to the CHoCH check above: fire the instant a close clears the
        # most-recent confirmed swing in the trend direction by >= bos_disp. No
        # swing-confirm lag (the old `made_ll`/`made_hh` path waited for the NEXT
        # swing to confirm, then reported `lows[-2]` — a stale, often already-
        # broken level). Gated by `rearm_block_dir` exactly like CHoCH so a
        # failed-CHoCH whipsaw cannot fire a false BOS. Runs AFTER the failure-
        # window block (which `continue`s) so it can never preempt a reversal.
        # Does NOT touch `defended`/`leg_extreme_*` — continuation must not reset
        # the protected swing. Target is cleared after firing so the same spent
        # swing cannot re-fire; it re-seeds on the next confirmed swing (sec. 4).
        if state == _DOWN and bos_break_low is not None and rearm_block_dir != _UP:
            broken_price = bos_break_low["price"]
            # Gap guard via _traded_through: a down-break only counts if price
            # traded through the level (continuous candle), not a weekend gap.
            if c < broken_price - bos_disp and _traded_through(ci, broken_price, 'down') \
                    and _body_gate_ok(ci, BOS_BODY_ATR_MULT):
                ts_now = _ts_iso(df, ci)
                bos_tier = ("Range" if h4_floor is not None
                            and abs(broken_price - h4_floor) <= bos_disp
                            else "BOS")
                last_bos = {"kind": "BOS", "direction": _DOWN,
                            "ts": ts_now, "tier": bos_tier}
                _push_event("BOS", "bearish", ts_now, broken_price,
                            impulse_start_ts, False, "bearish", bos_tier,
                            broken_swing_ts_arg=bos_break_low["ts"], break_idx=ci)
                trend_dir_swings_since_extend = 0
                bos_break_low = None  # spent — re-seeds on next confirmed low
                # A continuation BOS DOWN = the down-trend extended = any pending
                # CHoCH-UP failed (price resumed down instead of confirming up).
                # Silently cancel it — this BOS event already tells the story; a
                # separate CHoCH_FAILED on the SAME candle would just be noise.
                # State already DOWN — no flip.
                if choch_pending_dir == _UP:
                    prior_trend = None; choch = False
                    choch_origin = None; choch_level = None; choch_from_zone = False
                    choch_pending_dir = None; confirm_break_swing = None
                    choch_arm_idx = None
        elif state == _UP and bos_break_high is not None and rearm_block_dir != _DOWN:
            broken_price = bos_break_high["price"]
            # Gap guard via _traded_through: an up-break only counts if price
            # traded through the level (continuous candle), not a weekend gap.
            if c > broken_price + bos_disp and _traded_through(ci, broken_price, 'up') \
                    and _body_gate_ok(ci, BOS_BODY_ATR_MULT):
                ts_now = _ts_iso(df, ci)
                bos_tier = ("Range" if h4_ceiling is not None
                            and abs(broken_price - h4_ceiling) <= bos_disp
                            else "BOS")
                last_bos = {"kind": "BOS", "direction": _UP,
                            "ts": ts_now, "tier": bos_tier}
                _push_event("BOS", "bullish", ts_now, broken_price,
                            impulse_start_ts, False, "bullish", bos_tier,
                            broken_swing_ts_arg=bos_break_high["ts"], break_idx=ci)
                trend_dir_swings_since_extend = 0
                # A continuation BOS UP = the up-trend extended = any pending
                # CHoCH-DOWN failed (price resumed up instead of confirming down).
                # Silently cancel it — this BOS event already tells the story.
                if choch_pending_dir == _DOWN:
                    prior_trend = None; choch = False
                    choch_origin = None; choch_level = None; choch_from_zone = False
                    choch_pending_dir = None; confirm_break_swing = None
                    choch_arm_idx = None
                bos_break_high = None  # spent — re-seeds on next confirmed high

        # ---- 3. BIRTH (cold start) ---------------------------------------------
        if state == _UNDEF:
            if recent_high is not None and c > recent_high["price"]:
                state    = _UP
                defended = recent_low["price"] if recent_low else recent_high["price"]
                # defended is a real swing (recent_low, or recent_high fallback).
                defended_swing = recent_low if recent_low else recent_high
                leg_start = ci
                ts_now = _ts_iso(df, ci)
                impulse_start_ts = (recent_low["ts"] if recent_low
                                    else recent_high["ts"] if recent_high
                                    else ts_now)
                last_bos = {"kind": "BOS_BIRTH", "direction": _UP, "ts": ts_now}
                # Birth BOS breaks recent_high (the high price just exceeded).
                _push_event("BOS", "bullish", ts_now, recent_high["price"],
                            impulse_start_ts, False, "bullish", "BOS",
                            broken_swing_ts_arg=recent_high["ts"], break_idx=ci)
                # recent_high is now spent (just broken). Next BOS-up target is
                # the next confirmed swing high; cleared until it forms.
                bos_break_high = None; bos_break_low = None
            elif recent_low is not None and c < recent_low["price"]:
                state    = _DOWN
                defended = recent_high["price"] if recent_high else recent_low["price"]
                defended_swing = recent_high if recent_high else recent_low
                leg_start = ci
                ts_now = _ts_iso(df, ci)
                impulse_start_ts = (recent_high["ts"] if recent_high
                                    else recent_low["ts"] if recent_low
                                    else ts_now)
                last_bos = {"kind": "BOS_BIRTH", "direction": _DOWN, "ts": ts_now}
                # Birth BOS breaks recent_low (the low price just broken).
                _push_event("BOS", "bearish", ts_now, recent_low["price"],
                            impulse_start_ts, False, "bearish", "BOS",
                            broken_swing_ts_arg=recent_low["ts"], break_idx=ci)
                # recent_low is now spent (just broken). Next BOS-down target is
                # the next confirmed swing low; cleared until it forms.
                bos_break_low = None; bos_break_high = None

        # ---- 4. INGEST swings (MAINTAIN + BOS only) ----------------------------
        for s in by_known.get(ci, ()):
            if s["type"] == "high":
                highs.append(s)
                recent_high = s
            else:
                lows.append(s)
                recent_low = s

            # CONFIRMATION-BOS: while a CHoCH is pending, capture the FIRST
            # confirmed swing that prints in the PENDING direction AFTER the CHoCH
            # candle (s["idx"] > choch_arm_idx, the pivot's actual bar). This is
            # the swing whose break (by >= bos_disp) becomes the Confirmation BOS
            # (sec. 1b). Option B: the FIRST such swing only (set once, while
            # None) — a fixed, structurally-meaningful level, not a moving target.
            #   - CHoCH UP  pending -> first post-CHoCH swing HIGH (break upward)
            #   - CHoCH DOWN pending -> first post-CHoCH swing LOW (break downward)
            if choch_pending_dir is not None and confirm_break_swing is None \
                    and choch_arm_idx is not None and s["idx"] > choch_arm_idx:
                if choch_pending_dir == _UP and s["type"] == "high":
                    confirm_break_swing = s
                elif choch_pending_dir == _DOWN and s["type"] == "low":
                    confirm_break_swing = s

            # Maintain the BOS break target + the protected (defended) swing.
            # The BOS itself no longer fires here — it fires ON-CLOSE in sec. 2b.
            # Here we only keep the targets current as confirmed swings arrive:
            #   - trend-direction swing (low in DOWN / high in UP) => new BOS
            #     break target (most-recent confirmed swing point to break next).
            #   - counter-trend swing (HL in UP / LH in DOWN) => new `defended`
            #     swing for the CHoCH check (unchanged from before).
            if state == _UP:
                made_hl = (s["type"] == "low"  and len(lows)  >= 2
                           and lows[-1]["price"]  > lows[-2]["price"])
                if s["type"] == "high":
                    # New confirmed swing high = next BOS-up break target.
                    bos_break_high = highs[-1]
                if made_hl:
                    defended = lows[-1]["price"]
                    defended_swing = lows[-1]  # confirmed swing low (the new HL)
                    leg_extreme_high = float(H[ci])
                    impulse_start_ts = lows[-1]["ts"]
                    trend_dir_swings_since_extend = 0
                    if rearm_block_dir == _DOWN:
                        rearm_block_dir = None
                elif s["type"] == "low":
                    trend_dir_swings_since_extend += 1
            elif state == _DOWN:
                made_lh = (s["type"] == "high" and len(highs) >= 2
                           and highs[-1]["price"] < highs[-2]["price"])
                if s["type"] == "low":
                    # New confirmed swing low = next BOS-down break target.
                    bos_break_low = lows[-1]
                if made_lh:
                    defended = highs[-1]["price"]
                    defended_swing = highs[-1]  # confirmed swing high (the new LH)
                    leg_extreme_low = float(L[ci])
                    impulse_start_ts = highs[-1]["ts"]
                    trend_dir_swings_since_extend = 0
                    if rearm_block_dir == _UP:
                        rearm_block_dir = None
                elif s["type"] == "high":
                    trend_dir_swings_since_extend += 1

        if _trace is not None:
            _trace.append(state)

    ranging = (state in (_UP, _DOWN)
               and trend_dir_swings_since_extend >= STRUCTURE_RANGING_STALE)

    # CONFIRMATION-BOS model: a pending CHoCH means a reversal is ARMED but NOT
    # yet confirmed. `state` is the confirmed trend (UNCHANGED by the CHoCH).
    # `flip_unconfirmed` now means "a CHoCH is pending a Confirmation BOS" — the
    # field name is kept so downstream consumers (Phase 2 / chart) keep working.
    choch_pending = (choch_pending_dir is not None and choch_level is not None)
    flip_unconfirmed = bool(choch_pending and state in (_UP, _DOWN))
    # Direction the pending CHoCH is trying to flip TO (up/down), for labelling.
    _pend_to = {_UP: "UP", _DOWN: "DOWN"}.get(choch_pending_dir) if choch_pending else None

    if state == _UP:
        label = ("H1 trend UP" + (f" (CHoCH {_pend_to} pending — needs Confirmation BOS)"
                                   if flip_unconfirmed else
                                   (" (ranging)" if ranging else "")))
    elif state == _DOWN:
        label = ("H1 trend DOWN" + (f" (CHoCH {_pend_to} pending — needs Confirmation BOS)"
                                    if flip_unconfirmed else
                                    (" (ranging)" if ranging else "")))
    else:
        label = "H1 structure undefined (insufficient structure)"

    # trend key in 'bullish'/'bearish'/None for Phase 2 compatibility
    _trend_map = {_UP: "bullish", _DOWN: "bearish"}
    trend_out = _trend_map.get(state)

    # `is_setup_break` was previously tagged here on the most-recent event — wrong
    # when multiple OBs are shown (each OB has its own defining event). Tagging is
    # now done per-OB at render time via smc_detector.ob_broken_swing_ts().
    for s in swings:
        if "is_setup_break" in s:
            del s["is_setup_break"]

    return {
        "state":            state,
        "ranging":          ranging,
        # `prior_trend` = the confirmed trend the pending CHoCH would flip FROM
        # (i.e. the current `state`). Only meaningful while a CHoCH is pending.
        "prior_trend":      state if flip_unconfirmed else None,
        # Direction the pending CHoCH would flip TO ('up'|'down'); None when no
        # CHoCH is pending. New field — lets consumers name the pending reversal.
        "choch_pending_dir": choch_pending_dir if flip_unconfirmed else None,
        "flip_unconfirmed": flip_unconfirmed,
        "choch":            choch,
        "choch_ts":         choch_ts if flip_unconfirmed else None,
        "choch_level":      choch_level if flip_unconfirmed else None,
        "choch_from_zone":  bool(choch_from_zone) if flip_unconfirmed else False,
        "choch_flip_count": int(choch_flip_count),
        "defended":         defended,
        "broken_swing_ts":  broken_swing_ts if flip_unconfirmed else None,
        "last_bos":         last_bos,
        "label":            label,
        "events":           events_ring,
        "trend":            trend_out,
        "swings":           swings,
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
