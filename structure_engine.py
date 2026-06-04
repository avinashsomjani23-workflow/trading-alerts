"""
structure_engine.py — Stage 2: single H1 structure engine (CHoCH + BOS + trend).

ADDITIVE / PARALLEL. Runs ALONGSIDE the old dealing_range wall engine. It does
NOT move walls, does NOT change OB building, does NOT delete anything. It emits
a labelled structure read for the digest and for verification against the old
engine. Cutover + deletion of the old engine is Stage 3.

THE MODEL (locked with the trader — plain English)

  TREND STATE is one of:
    up         : higher-highs / higher-lows intact.
    down       : lower-highs / lower-lows intact.
    transition : a CHoCH has fired (warning the trend may be ending) but the
                 trend has NOT flipped yet. Resolves on the next BOS.
    undefined  : cold start, no trend established yet.

  RANGING is NOT a state. It is a FLAG that can sit inside up or down (price
  chopping while the trend is still technically intact). Exposed as `ranging`.

  CHoCH — single definition, no Major/Minor (H1):
    In an UP trend: price reverses from the TOP 25% of the (confirmed H4)
      dealing range, then CLOSES below the most recent confirmed H1 swing LOW
      (the "defended low"). -> CHoCH fires. State -> transition (prior=up).
    In a DOWN trend: reverse from BOTTOM 25%, then CLOSE above the most recent
      confirmed H1 swing HIGH (defended high). -> CHoCH. State -> transition.
    CHoCH does NOT flip the trend. It is a signal.

  BOS — two tiers (H1):
    Minor BOS : H1 close past a confirmed H1 swing (in the direction tested).
    Major BOS : H1 close past the H4-derived WALL by >= BOS_ATR_MULT * H1 ATR.
    (Both are BOS. Tier is informational, mirrors the old system's split.)

  TRANSITION resolves via TWO-WAY BOS (the blind spot the old engine had —
  old BOS only fired WITH the trend; in transition there is no trend, so BOS
  must be allowed in BOTH directions):
    - BOS in the NEW direction (same side as the CHoCH) -> FLIP confirmed.
      Trend becomes the new direction.
    - BOS in the OLD direction (reclaim — price closes back past the swing the
      CHoCH broke / makes a with-old-trend BOS) -> CHoCH FAILED. Resume the
      prior trend; clear transition.
    Whichever fires first wins. Until one fires, state stays transition.

  ALL premium/discount (the 25% gate) reads the FROZEN confirmed H4 range via
  dealing_range.compute_pd_confirmed — never the live/moving range. ALL ATR is
  H1 ATR.

COUPLING
  - dealing_range.detect_swings  (shared H1 swing definition: lb-3 + ATR)
  - dealing_range.compute_pd_confirmed  (frozen H4 25% lines)
  - dealing_range._compute_atr   (H1 ATR)
  Writes no state. Pure function of (H1 df, H4 range block).
"""

from typing import Dict, Any, List, Optional

import dealing_range as _dr

UP         = "up"
DOWN       = "down"
TRANSITION = "transition"
UNDEFINED  = "undefined"

# Tier thresholds — H1 ATR units (reuse the locked dealing_range constants so
# there is one source of truth for the displacement bar).
BOS_ATR_MULT = _dr.BOS_ATR_MULT       # Major BOS: close past H4 wall by >= this * H1 ATR

# Ranging flag: trend intact but price has not extended the trend for at least
# this many confirmed swings of the trend-direction type (pure chop read).
# Informational only; does not change the state.
RANGING_STALE_SWINGS = 2


def _confirm_idx(swing: Dict[str, Any], lookback: int) -> int:
    """Candle index at which a swing becomes KNOWN (lb confirmation lag)."""
    return int(swing["idx"]) + lookback


def compute_structure(df, h4_range: Optional[Dict[str, Any]],
                      lookback: int = _dr.SWING_LOOKBACK,
                      _min_leg_atr_mult: Optional[float] = _dr.MIN_LEG_ATR_MULT) -> Dict[str, Any]:
    """Compute the H1 structure read. Pure; never raises on thin data.

    Args:
      df:        H1 OHLC DataFrame (live shape: 'Datetime' col or DatetimeIndex).
      h4_range:  the block from h4_range.compute_h4_range(df) — carries the
                 confirmed (frozen) 25% reference and the live wall. May be None.
      lookback:  swing lookback (shared default 3).

    Returns:
      {
        'state':        'up'|'down'|'transition'|'undefined',
        'ranging':      bool,                 # flag inside up/down
        'prior_trend':  'up'|'down'|None,     # set in transition
        'choch':        bool,                 # CHoCH fired into the current transition
        'choch_ts':     str|None,
        'defended':     float|None,           # active defended level (up: HL, down: LH)
        'broken_swing_ts': str|None,          # the H1 swing the CHoCH broke
        'last_bos':     dict|None,            # most recent BOS {tier,direction,ts,wall}
        'label':        str,
      }
    """
    empty = {"state": UNDEFINED, "ranging": False, "prior_trend": None,
             "choch": False, "choch_ts": None, "defended": None,
             "broken_swing_ts": None, "last_bos": None,
             "label": "H1 structure undefined (insufficient data)"}
    if df is None or len(df) < (lookback * 2 + 5):
        return empty

    atr = _dr._compute_atr(df)
    if atr is None or atr <= 0:
        return empty

    swings: List[Dict[str, Any]] = _dr.detect_swings(
        df, lookback=lookback, min_leg_atr_mult=_min_leg_atr_mult)
    if not swings:
        return empty

    closes = df["Close"].to_numpy(dtype=float)
    n = len(df)

    # Frozen 25% gate lines from the confirmed H4 range. Computed ONCE at the
    # latest price; the gate is a property of the (slow) H4 range, so using the
    # current confirmed lines for the whole H1 walk is correct — the confirmed
    # H4 range only changes when a new H4 swing confirms, far slower than H1.
    price_now = float(closes[-1])
    pdc = _dr.compute_pd_confirmed(price_now, {"h4_range": h4_range or {}})
    gate_valid = bool(pdc.get("valid"))
    premium_floor = pdc.get("premium_floor")   # >= this == top 25%
    discount_ceil = pdc.get("discount_ceil")   # <= this == bottom 25%

    # Live H4 wall (for Major BOS). May be None if H4 range absent.
    wall_ceiling = (h4_range or {}).get("ceiling")
    wall_floor = (h4_range or {}).get("floor")

    by_known: Dict[int, List[Dict[str, Any]]] = {}
    for s in swings:
        by_known.setdefault(_confirm_idx(s, lookback), []).append(s)

    # Highs/Lows of df for reversal-extreme checks.
    H = df["High"].to_numpy(dtype=float)
    L = df["Low"].to_numpy(dtype=float)

    state = UNDEFINED
    prior_trend: Optional[str] = None
    defended: Optional[float] = None
    choch = False
    choch_ts: Optional[str] = None
    broken_swing_ts: Optional[str] = None
    last_bos: Optional[Dict[str, Any]] = None

    highs: List[Dict[str, Any]] = []
    lows: List[Dict[str, Any]] = []
    # Most recent confirmed H1 swing low / high "known" so far (defended levels).
    recent_low: Optional[Dict[str, Any]] = None
    recent_high: Optional[Dict[str, Any]] = None
    # The swing extreme the CHoCH must reverse FROM (premium/discount origin).
    # We track the running reversal extreme since the last leg for the gate.
    trend_dir_swings_since_extend = 0  # for the ranging flag

    def _reversed_from_premium(idx_from: int, idx_to: int) -> bool:
        """Down-CHoCH precondition: the REVERSAL HIGH — the highest High in
        [idx_from, idx_to] (the leg before the break) — sat in the TOP 25% of
        the confirmed H4 range AT THE TIME IT FORMED. We read the reversal
        extreme's position in the FROZEN confirmed range, NOT the live price.

        This is what makes the gate robust: it looks BACKWARD at the high price
        rejected from (which was inside the range), so a subsequent breakdown
        that carries price outside the range never invalidates the gate, and a
        mid-range break that never tagged premium never passes it. Mirrors the
        old engine's _premium_zone_satisfied, against the confirmed H4 range."""
        if not gate_valid or premium_floor is None or idx_from > idx_to:
            return False
        return float(H[idx_from:idx_to + 1].max()) >= premium_floor

    def _reversed_from_discount(idx_from: int, idx_to: int) -> bool:
        """Up-CHoCH precondition: the reversal LOW (lowest Low in the leg) sat
        in the BOTTOM 25% of the confirmed H4 range. Backward-looking, frozen
        range — symmetric with _reversed_from_premium."""
        if not gate_valid or discount_ceil is None or idx_from > idx_to:
            return False
        return float(L[idx_from:idx_to + 1].min()) <= discount_ceil

    # leg_start: index where the current trend leg began (last flip/birth), used
    # as the window for the premium/discount reversal check.
    leg_start = 0
    # Flip tracking: a trend flip out of transition requires a NEW swing in the
    # new direction, formed AFTER the CHoCH. choch_idx marks the CHoCH bar;
    # flip_low/flip_high hold the most recent post-CHoCH swing of each type.
    choch_idx: Optional[int] = None
    flip_low: Optional[Dict[str, Any]] = None
    flip_high: Optional[Dict[str, Any]] = None

    for ci in range(n):
        c = closes[ci]

        # ---- 1. CHoCH check (signal) — only from a live up/down trend -------
        # The premium/discount window is anchored at the most recent confirmed
        # opposite-side swing (the reversal extreme candidate), NOT at the
        # pattern-confirmation bar. In an uptrend the reversal HIGH that price
        # rejected from is recent_high — formed BEFORE the trend pattern even
        # confirmed (confirmation lags by `lookback` bars). Windowing from
        # leg_start (the confirmation bar) would miss that high. So we window
        # from the reversal swing's own idx forward to now.
        if state == UP and defended is not None and gate_valid:
            rev_idx = recent_high["idx"] if recent_high else leg_start
            if c < defended and _reversed_from_premium(rev_idx, ci):
                choch = True
                choch_ts = _dr._ts_iso(df, ci)
                broken_swing_ts = recent_low["ts"] if recent_low else None
                prior_trend = UP
                state = TRANSITION
                # defended stays as the broken low — used as the reclaim line.
                leg_start = ci
                choch_idx = ci
                flip_low = flip_high = None   # await a NEW post-CHoCH swing
        elif state == DOWN and defended is not None and gate_valid:
            rev_idx = recent_low["idx"] if recent_low else leg_start
            if c > defended and _reversed_from_discount(rev_idx, ci):
                choch = True
                choch_ts = _dr._ts_iso(df, ci)
                broken_swing_ts = recent_high["ts"] if recent_high else None
                prior_trend = DOWN
                state = TRANSITION
                leg_start = ci
                choch_idx = ci
                flip_low = flip_high = None   # await a NEW post-CHoCH swing

        # ---- 2. TRANSITION resolution — two-way BOS ------------------------
        elif state == TRANSITION:
            # New-direction BOS (flip) vs old-direction BOS (reclaim).
            if prior_trend == UP:
                # CHoCH was DOWN. New-direction = down. Flip ONLY on a close
                # below a NEW post-CHoCH swing low (flip_low) — the low of the
                # first lower-high leg — not the higher-low the CHoCH broke.
                flipped, tier, wall = _flip_down(c, flip_low, wall_floor, atr)
                if flipped:
                    state = DOWN
                    prior_trend = None
                    choch = False
                    defended = recent_high["price"] if recent_high else None
                    leg_start = ci
                    last_bos = {"tier": tier, "direction": DOWN,
                                "ts": _dr._ts_iso(df, ci), "wall": wall}
                else:
                    # Reclaim: with-old-trend (up) BOS cancels the CHoCH.
                    reclaimed, tier, wall = _bos_up(c, recent_high, wall_ceiling, atr)
                    if reclaimed:
                        state = UP
                        prior_trend = None
                        choch = False
                        defended = recent_low["price"] if recent_low else None
                        leg_start = ci
                        last_bos = {"tier": tier, "direction": UP,
                                    "ts": _dr._ts_iso(df, ci), "wall": wall}
            elif prior_trend == DOWN:
                # New-direction = up. Flip ONLY on a close above a NEW post-CHoCH
                # swing high (flip_high) — the high of the first higher-low leg.
                flipped, tier, wall = _flip_up(c, flip_high, wall_ceiling, atr)
                if flipped:
                    state = UP
                    prior_trend = None
                    choch = False
                    defended = recent_low["price"] if recent_low else None
                    leg_start = ci
                    last_bos = {"tier": tier, "direction": UP,
                                "ts": _dr._ts_iso(df, ci), "wall": wall}
                else:
                    reclaimed, tier, wall = _bos_down(c, recent_low, wall_floor, atr)
                    if reclaimed:
                        state = DOWN
                        prior_trend = None
                        choch = False
                        defended = recent_high["price"] if recent_high else None
                        leg_start = ci
                        last_bos = {"tier": tier, "direction": DOWN,
                                    "ts": _dr._ts_iso(df, ci), "wall": wall}

        # ---- 3. INGEST swings confirmed AT this candle ---------------------
        for s in by_known.get(ci, ()):
            if s["type"] == "high":
                highs.append(s)
                recent_high = s
            else:
                lows.append(s)
                recent_low = s

            # Track NEW post-CHoCH swings for the flip test. A swing whose own
            # idx is after the CHoCH bar belongs to the new (reversal) leg — its
            # break is the first BOS in the new direction. Swings confirmed at
            # the CHoCH bar formed earlier (old structure) and are excluded.
            if state == TRANSITION and choch_idx is not None and s["idx"] > choch_idx:
                if s["type"] == "low":
                    flip_low = s
                else:
                    flip_high = s

            # While in TRANSITION the trend is owned by step 2 (the close-break
            # flip / reclaim). The swing-pattern reassignment below must NOT flip
            # the trend on an LH&LL / HH&HL pattern — that would bypass the
            # "close below the new low" rule. So skip the pattern block entirely
            # in transition; only the post-CHoCH swing tracking above runs.
            if state == TRANSITION:
                continue

            HH = HL = LH = LL = False
            if len(highs) >= 2 and len(lows) >= 2:
                HH = highs[-1]["price"] > highs[-2]["price"]
                HL = lows[-1]["price"] > lows[-2]["price"]
                LH = highs[-1]["price"] < highs[-2]["price"]
                LL = lows[-1]["price"] < lows[-2]["price"]

            if HH and HL:
                if state != UP:
                    leg_start = ci
                state = UP
                prior_trend = None
                choch = False
                defended = lows[-1]["price"]
                trend_dir_swings_since_extend = 0
            elif LH and LL:
                if state != DOWN:
                    leg_start = ci
                state = DOWN
                prior_trend = None
                choch = False
                defended = highs[-1]["price"]
                trend_dir_swings_since_extend = 0
            else:
                # Ratchet defended in the trend's favour only (never loosen).
                if state == UP and s["type"] == "low" and len(lows) >= 2:
                    if lows[-1]["price"] > lows[-2]["price"]:
                        defended = lows[-1]["price"]
                        trend_dir_swings_since_extend = 0
                    else:
                        trend_dir_swings_since_extend += 1
                elif state == DOWN and s["type"] == "high" and len(highs) >= 2:
                    if highs[-1]["price"] < highs[-2]["price"]:
                        defended = highs[-1]["price"]
                        trend_dir_swings_since_extend = 0
                    else:
                        trend_dir_swings_since_extend += 1

    ranging = (state in (UP, DOWN)
               and trend_dir_swings_since_extend >= RANGING_STALE_SWINGS)

    # ---- Label ----
    if state == UP:
        label = "H1 trend UP" + (" (ranging)" if ranging else "")
    elif state == DOWN:
        label = "H1 trend DOWN" + (" (ranging)" if ranging else "")
    elif state == TRANSITION:
        label = f"H1 TRANSITION — CHoCH from {prior_trend.upper()} (awaiting BOS to flip or reclaim)"
    else:
        label = "H1 structure undefined (insufficient structure)"

    return {
        "state": state,
        "ranging": ranging,
        "prior_trend": prior_trend if state == TRANSITION else None,
        "choch": choch,
        "choch_ts": choch_ts if state == TRANSITION else None,
        "defended": defended,
        "broken_swing_ts": broken_swing_ts if state == TRANSITION else None,
        "last_bos": last_bos,
        "label": label,
    }


def _flip_down(close_price: float, flip_low: Optional[Dict[str, Any]],
               wall_floor: Optional[float], atr: float):
    """FLIP BOS (down) out of transition. Textbook SMC: after a bearish CHoCH
    the trend flips ONLY when price closes below a NEW swing low formed AFTER
    the CHoCH (the low under the first lower-high of the new leg) — never the
    higher-low the CHoCH already broke. `flip_low` is that post-CHoCH swing low;
    if it is None no new low exists yet, so no flip. Tier is Major when the same
    close also clears the H4 floor by >= BOS_ATR_MULT*ATR, else Minor."""
    if flip_low is None or close_price >= flip_low["price"]:
        return False, None, None
    if (wall_floor is not None
            and close_price < wall_floor
            and (wall_floor - close_price) >= BOS_ATR_MULT * atr):
        return True, "Major", float(wall_floor)
    return True, "Minor", None


def _flip_up(close_price: float, flip_high: Optional[Dict[str, Any]],
             wall_ceiling: Optional[float], atr: float):
    """FLIP BOS (up) out of transition — symmetric with _flip_down. Requires a
    close above a NEW swing high formed AFTER the bullish CHoCH (the high above
    the first higher-low of the new up-leg), never the lower-high the CHoCH
    already broke."""
    if flip_high is None or close_price <= flip_high["price"]:
        return False, None, None
    if (wall_ceiling is not None
            and close_price > wall_ceiling
            and (close_price - wall_ceiling) >= BOS_ATR_MULT * atr):
        return True, "Major", float(wall_ceiling)
    return True, "Minor", None


def _bos_up(close_price: float, recent_high: Optional[Dict[str, Any]],
            wall_ceiling: Optional[float], atr: float):
    """Return (fired, tier, wall_used). Up BOS = close past confirmed H1 swing
    high (Minor) or past the H4 ceiling by >= BOS_ATR_MULT*ATR (Major).
    Major has precedence when both fire. Used for the RECLAIM (with-old-trend)
    leg of transition resolution, not the flip."""
    if (wall_ceiling is not None
            and close_price > wall_ceiling
            and (close_price - wall_ceiling) >= BOS_ATR_MULT * atr):
        return True, "Major", float(wall_ceiling)
    if recent_high is not None and close_price > recent_high["price"]:
        return True, "Minor", None
    return False, None, None


def _bos_down(close_price: float, recent_low: Optional[Dict[str, Any]],
              wall_floor: Optional[float], atr: float):
    if (wall_floor is not None
            and close_price < wall_floor
            and (wall_floor - close_price) >= BOS_ATR_MULT * atr):
        return True, "Major", float(wall_floor)
    if recent_low is not None and close_price < recent_low["price"]:
        return True, "Minor", None
    return False, None, None
