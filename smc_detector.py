"""
smc_detector.py — Pure Python SMC Pattern Detection Engine v2
==============================================================
Zero Gemini calls. Every function is deterministic and testable.
Pair-type aware: forex, forex_jpy, index, commodity.

v2 changes from v1:
  - Strong Point / Weak Point tracking for iBOS vs BOS vs CHoCH
  - Partial liquidity sweep scoring (2.5 / 1.5 / 0)
  - M15 opposing OBs + FVGs as TP targets (H1 OB/FVG dropped entirely)
  - Multi-TF scoring: 4-tier (1.5 / 1.0 / 0.5 / 0)
  - Pair-type-aware proximity thresholds for sweeps

Functions:
    1.  get_swing_points          — swing high/low detection
    2.  _determine_trend          — HH/HL/LH/LL trend classification
    3.  _find_strong_weak_points  — Strong Point (ERL) and Weak Point
    4.  detect_bos_choch          — BOS / CHoCH / iBOS with displacement
    5.  detect_fvgs               — Fair Value Gaps with mitigation tracking
    6.  get_unmitigated_fvgs      — filter helper
    7.  detect_obs                — Order Blocks from FVG backward-walk
    8.  detect_liquidity_sweep    — wick-rejection sweep (2.5 / 1.5 / 0)
    9.  compute_premium_discount  — 60/40 equilibrium check
    10. select_trade_ob           — pick best OB for zone + bias
    11. compute_levels            — entry / SL / TP1 / TP2 / RR
    12. check_entry_readiness     — engulfing or BOS/CHoCH trigger
    13. compute_invalidation      — direction-aware invalidation levels
    14. compute_score             — full weighted scorecard (news → Gemini)
    15. run_analysis              — main orchestrator
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# PAIR-AWARE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# BOS/CHoCH displacement: breaking candle body must be >= this fraction of
# H1 ATR(14). Filters doji and weak grinds.
# 0.25 → EURUSD ATR 0.0040 needs ~10 pip body. NAS100 ATR 200 needs ~50 pts.
DISPLACEMENT_ATR_FRACTION = 0.25

# BOS tolerance: how far past the swing body extreme the close must be.
# Prevents false breaks where close barely nicks the level.
BOS_TOLERANCE = {
    "forex":     0.00010,   # 1 pip
    "forex_jpy": 0.010,     # 1 pip (JPY scale)
    "index":     5.0,       # 5 points (NAS100)
    "commodity": 0.50,      # $0.50 (Gold)
}

# Liquidity sweep: wick must be >= this multiple of body for rejection.
SWEEP_WICK_BODY_RATIO = 1.5

# Sweep: minimum body size as fraction of ATR (anti-doji guard).
# When body ~ 0 the wick/body ratio explodes → false sweep.
SWEEP_MIN_BODY_ATR = 0.10

# Sweep: how many recent candles to check for sweep evidence.
SWEEP_CANDLE_WINDOW = 3

# Sweep: proximity threshold for near-miss scoring (1.5 pts).
# Wick approached within this % of the liquidity level but did not pierce.
SWEEP_PROXIMITY_PCT = {
    "forex":     0.06,   # ~6.5 pips on EURUSD
    "forex_jpy": 0.08,   # ~11.6 pips on USDJPY
    "index":     0.12,   # ~23 pts on NAS100
    "commodity": 0.10,   # ~$2.40 on Gold
}

# Premium/Discount: tolerance band around equilibrium.
# 0.10 → LONG valid below 60 %, SHORT valid above 40 %.
PD_TOLERANCE = 0.10

# OB: max candles to walk backward from FVG origin.
OB_MAX_WALK_BACK = 5

# M15 analysis window (candles from end of DataFrame).
FVG_M15_WINDOW = 100   # ~25 hours

# TP fallback R-multiples when no opposing structure found.
TP1_FALLBACK_R = 2.5
TP2_FALLBACK_R = 4.0

# Engulfing confirmation: body must be >= this fraction of ATR.
ENGULF_MIN_BODY_ATR = 0.20


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _pair_tol_key(pair_conf):
    """Map pair config to tolerance-type key."""
    pt = pair_conf.get("pair_type", "forex")
    name = pair_conf.get("name", "")
    if pt == "forex" and "JPY" in name:
        return "forex_jpy"
    if pt == "index":
        return "index"
    if pt == "commodity":
        return "commodity"
    return "forex"


def _dp(pair_conf):
    """Shorthand for decimal places."""
    return pair_conf.get("decimal_places", 5)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SWING POINT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_swing_points(df, lookback=5):
    """
    Detect swing highs and swing lows from OHLC data.

    A swing high at index i: High[i] is the max in [i-lookback : i+lookback+1].
    A swing low  at index i: Low[i]  is the min in [i-lookback : i+lookback+1].

    Returns list of dicts sorted by index:
        {type, price, body_extreme, idx, ts}

    body_extreme = max(Open, Close) for highs, min(Open, Close) for lows.
    Used for BOS body-break detection (not wick-break).
    """
    if df is None or len(df) < lookback * 2 + 1:
        return []

    H = df['High'].values.flatten().astype(float)
    L = df['Low'].values.flatten().astype(float)
    O = df['Open'].values.flatten().astype(float)
    C = df['Close'].values.flatten().astype(float)

    swings = []
    for i in range(lookback, len(H) - lookback):
        window_h = H[i - lookback: i + lookback + 1]
        window_l = L[i - lookback: i + lookback + 1]

        if H[i] == max(window_h):
            swings.append({
                "type": "high",
                "price": float(H[i]),
                "body_extreme": float(max(O[i], C[i])),
                "idx": i,
                "ts": str(df.index[i])
            })
        if L[i] == min(window_l):
            swings.append({
                "type": "low",
                "price": float(L[i]),
                "body_extreme": float(min(O[i], C[i])),
                "idx": i,
                "ts": str(df.index[i])
            })

    swings.sort(key=lambda s: s["idx"])
    return swings


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TREND DETERMINATION
# ═══════════════════════════════════════════════════════════════════════════════

def _determine_trend(swings):
    """
    Classify trend from the last two swing highs and two swing lows.

    bullish  = Higher High AND Higher Low  (HH + HL)
    bearish  = Lower High  AND Lower Low   (LH + LL)
    mixed    = anything else (range / transition)

    Returns: "bullish", "bearish", or "mixed"
    """
    sh = [s for s in swings if s["type"] == "high"]
    sl = [s for s in swings if s["type"] == "low"]

    if len(sh) < 2 or len(sl) < 2:
        return "mixed"

    hh = sh[-1]["price"] > sh[-2]["price"]
    hl = sl[-1]["price"] > sl[-2]["price"]

    if hh and hl:
        return "bullish"
    if not hh and not hl:
        return "bearish"
    return "mixed"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. STRONG POINT / WEAK POINT (ERL / IRL)
# ═══════════════════════════════════════════════════════════════════════════════

def _find_strong_weak_points(swings, trend):
    """
    Identify the External Range Liquidity boundaries.

    Strong Point = origin of the last validated BOS.
        Bullish: swing LOW between the second-to-last and last swing HIGH
                 (the low that launched the move breaking the prior high).
                 Breaking the Strong Point = CHoCH (trend reversal).

        Bearish: swing HIGH between the second-to-last and last swing LOW.

    Weak Point = the most recent extreme in the trend direction.
        Bullish: most recent swing HIGH (breaking it = continuation BOS).
        Bearish: most recent swing LOW.

    Mixed / insufficient data: returns (None, None) — all breaks treated as
    external (cannot determine internal range).

    Returns: (strong_point_dict, weak_point_dict) or (None, None)
    """
    sh = [s for s in swings if s["type"] == "high"]
    sl = [s for s in swings if s["type"] == "low"]

    if trend == "bullish" and len(sh) >= 2 and len(sl) >= 1:
        latest_sh = sh[-1]
        prev_sh = sh[-2]
        candidates = [s for s in sl
                      if prev_sh["idx"] < s["idx"] < latest_sh["idx"]]
        if candidates:
            strong_point = min(candidates, key=lambda s: s["price"])
        else:
            strong_point = sl[-1]
        weak_point = latest_sh
        return strong_point, weak_point

    if trend == "bearish" and len(sl) >= 2 and len(sh) >= 1:
        latest_sl = sl[-1]
        prev_sl = sl[-2]
        candidates = [s for s in sh
                      if prev_sl["idx"] < s["idx"] < latest_sl["idx"]]
        if candidates:
            strong_point = max(candidates, key=lambda s: s["price"])
        else:
            strong_point = sh[-1]
        weak_point = latest_sl
        return strong_point, weak_point

    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BOS / CHoCH / iBOS DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_bos_choch(df, pair_conf, atr_value=None, lookback=5):
    """
    Detect the most recent structural break on the given timeframe.

    Method:
      1. Find swing points → determine trend → find Strong/Weak Points
      2. Scan last 5 CLOSED candles for a body close past a swing level
         (body_extreme +/- pair-aware tolerance)
      3. Classify:
         - Close past Weak Point  → BOS   (continuation, external)
         - Close past Strong Point → CHoCH (reversal, external)
         - Close past internal swing → iBOS (internal)
      4. Displacement: breaking candle body >= 0.25 x ATR

    Priority: external breaks over internal.
    Among same level, most recent candle wins.

    Returns dict:
        confirmed, type (bos/choch/ibos), direction (bullish/bearish),
        break_level (external/internal), break_price, break_candle_ts,
        break_candle_idx, swing_broken, has_displacement, reason
    """
    empty = {
        "confirmed": False, "type": None, "direction": None,
        "break_level": None, "break_price": None,
        "break_candle_ts": None, "break_candle_idx": None,
        "swing_broken": None, "has_displacement": False, "reason": ""
    }

    if df is None or len(df) < lookback * 2 + 5:
        empty["reason"] = "Insufficient data for BOS/CHoCH"
        return empty

    swings = get_swing_points(df, lookback)
    sh = [s for s in swings if s["type"] == "high"]
    sl = [s for s in swings if s["type"] == "low"]

    if len(sh) < 2 or len(sl) < 2:
        empty["reason"] = (f"Need >=2 swing highs and >=2 swing lows, "
                           f"got {len(sh)}H {len(sl)}L")
        return empty

    trend = _determine_trend(swings)
    strong_point, weak_point = _find_strong_weak_points(swings, trend)

    tol = BOS_TOLERANCE.get(_pair_tol_key(pair_conf), 0.0001)
    disp_thresh = (DISPLACEMENT_ATR_FRACTION * atr_value
                   if atr_value and atr_value > 0 else 0)

    C_arr = df['Close'].values.flatten().astype(float)
    O_arr = df['Open'].values.flatten().astype(float)
    n = len(df)

    # ── Build target list with classification ─────────────────────────────
    targets = []

    if strong_point is not None and weak_point is not None:
        # Weak Point (continuation)
        if weak_point["type"] == "high":
            targets.append({
                "body_extreme": weak_point["body_extreme"],
                "price": weak_point["price"],
                "test": "above",
                "break_type": "bos",
                "direction": "bullish",
                "break_level": "external",
                "swing": weak_point
            })
        else:
            targets.append({
                "body_extreme": weak_point["body_extreme"],
                "price": weak_point["price"],
                "test": "below",
                "break_type": "bos",
                "direction": "bearish",
                "break_level": "external",
                "swing": weak_point
            })

        # Strong Point (reversal)
        if strong_point["type"] == "low":
            targets.append({
                "body_extreme": strong_point["body_extreme"],
                "price": strong_point["price"],
                "test": "below",
                "break_type": "choch",
                "direction": "bearish",
                "break_level": "external",
                "swing": strong_point
            })
        else:
            targets.append({
                "body_extreme": strong_point["body_extreme"],
                "price": strong_point["price"],
                "test": "above",
                "break_type": "choch",
                "direction": "bullish",
                "break_level": "external",
                "swing": strong_point
            })

        # Internal swings (iBOS)
        sp_idx = strong_point["idx"]
        wp_idx = weak_point["idx"]
        lo_bound = min(sp_idx, wp_idx)
        hi_bound = max(sp_idx, wp_idx)

        for s in swings:
            if s["idx"] <= lo_bound or s["idx"] >= hi_bound:
                continue
            if s is strong_point or s is weak_point:
                continue
            if s["type"] == "high":
                targets.append({
                    "body_extreme": s["body_extreme"],
                    "price": s["price"],
                    "test": "above",
                    "break_type": "ibos",
                    "direction": "bullish",
                    "break_level": "internal",
                    "swing": s
                })
            else:
                targets.append({
                    "body_extreme": s["body_extreme"],
                    "price": s["price"],
                    "test": "below",
                    "break_type": "ibos",
                    "direction": "bearish",
                    "break_level": "internal",
                    "swing": s
                })
    else:
        # Mixed / no SP-WP → every swing break is external
        for s in sh:
            targets.append({
                "body_extreme": s["body_extreme"],
                "price": s["price"],
                "test": "above",
                "break_type": "bos",
                "direction": "bullish",
                "break_level": "external",
                "swing": s
            })
        for s in sl:
            targets.append({
                "body_extreme": s["body_extreme"],
                "price": s["price"],
                "test": "below",
                "break_type": "bos",
                "direction": "bearish",
                "break_level": "external",
                "swing": s
            })

    if not targets:
        empty["reason"] = "No swing targets for break detection"
        return empty

    # ── Scan last 5 closed candles ────────────────────────────────────────
    best = None

    for i in range(n - 2, max(n - 7, -1), -1):
        body = abs(C_arr[i] - O_arr[i])
        has_disp = body >= disp_thresh if disp_thresh > 0 else True

        for t in targets:
            if t["swing"]["idx"] >= i:
                continue

            broken = False
            if t["test"] == "above" and C_arr[i] > t["body_extreme"] + tol:
                broken = True
            elif t["test"] == "below" and C_arr[i] < t["body_extreme"] - tol:
                broken = True

            if not broken:
                continue

            candidate = {
                "confirmed": True,
                "type": t["break_type"],
                "direction": t["direction"],
                "break_level": t["break_level"],
                "break_price": float(C_arr[i]),
                "break_candle_ts": str(df.index[i]),
                "break_candle_idx": i,
                "swing_broken": t["price"],
                "has_displacement": has_disp,
                "reason": (
                    f"{t['break_type'].upper()} {t['direction']}: "
                    f"close {C_arr[i]:.6f} "
                    f"{'>' if t['test'] == 'above' else '<'} "
                    f"swing body {t['body_extreme']:.6f} "
                    f"(trend was {trend}, {t['break_level']})")
            }

            if best is None:
                best = candidate
            elif (candidate["break_level"] == "external"
                  and best["break_level"] == "internal"):
                best = candidate
            elif (candidate["break_level"] == best["break_level"]
                  and candidate["break_candle_idx"] >
                  best["break_candle_idx"]):
                best = candidate

    if best is not None:
        return best

    empty["reason"] = "No structural break in last 5 closed candles"
    return empty


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FVG DETECTION + DIRECTION-AWARE MITIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_fvgs(df, window=None):
    """
    Detect Fair Value Gaps with direction-aware mitigation tracking.

    Bullish FVG (gap UP):  candle[i].High < candle[i+2].Low
        fvg_top    = candle[i+2].Low
        fvg_bottom = candle[i].High
        Mitigated when any later candle's Low drops INTO gap (<= fvg_top).

    Bearish FVG (gap DOWN): candle[i].Low > candle[i+2].High
        fvg_top    = candle[i].Low
        fvg_bottom = candle[i+2].High
        Mitigated when any later candle's High rises INTO gap (>= fvg_bottom).
    """
    fvgs = []
    if df is None or len(df) < 3:
        return fvgs

    n = len(df)
    start = max(0, n - (window or FVG_M15_WINDOW))

    H = df['High'].values.flatten().astype(float)
    L = df['Low'].values.flatten().astype(float)

    for i in range(start, n - 2):
        # Bearish FVG: gap down
        if L[i] > H[i + 2]:
            fvg = {
                'type': 'bearish',
                'fvg_top':    round(float(L[i]), 6),
                'fvg_bottom': round(float(H[i + 2]), 6),
                'c1_ts': str(df.index[i]),
                'c3_ts': str(df.index[i + 2]),
                'idx': i,
                'end_idx': i + 2,
                'mitigated': False,
                'mitigated_at_ts': None
            }
            for j in range(i + 3, n):
                if H[j] >= fvg['fvg_bottom']:
                    fvg['mitigated'] = True
                    fvg['mitigated_at_ts'] = str(df.index[j])
                    break
            fvgs.append(fvg)

        # Bullish FVG: gap up
        elif H[i] < L[i + 2]:
            fvg = {
                'type': 'bullish',
                'fvg_top':    round(float(L[i + 2]), 6),
                'fvg_bottom': round(float(H[i]), 6),
                'c1_ts': str(df.index[i]),
                'c3_ts': str(df.index[i + 2]),
                'idx': i,
                'end_idx': i + 2,
                'mitigated': False,
                'mitigated_at_ts': None
            }
            for j in range(i + 3, n):
                if L[j] <= fvg['fvg_top']:
                    fvg['mitigated'] = True
                    fvg['mitigated_at_ts'] = str(df.index[j])
                    break
            fvgs.append(fvg)

    return fvgs


def get_unmitigated_fvgs(fvgs):
    """Return only FVGs not yet filled by subsequent price action."""
    return [f for f in fvgs if not f['mitigated']]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ORDER BLOCK DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_obs(df, fvgs, max_walk_back=None):
    """
    Detect Order Blocks by walking backward from each FVG.

    OB = last opposing candle before the impulse that created the FVG.
      Bearish FVG → last BULLISH candle = bearish OB (for SHORTs)
      Bullish FVG → last BEARISH candle = bullish OB (for LONGs)

    Mitigated OBs (current price closed through body) excluded.
    Walk-back capped at OB_MAX_WALK_BACK.
    """
    obs = []
    if df is None or len(df) == 0 or not fvgs:
        return obs

    max_wb = max_walk_back or OB_MAX_WALK_BACK
    seen_idx = set()
    current_close = float(df['Close'].iloc[-1])

    O = df['Open'].values.flatten().astype(float)
    C = df['Close'].values.flatten().astype(float)
    H = df['High'].values.flatten().astype(float)
    L = df['Low'].values.flatten().astype(float)

    for fvg in fvgs:
        fvg_idx = fvg['idx']
        fvg_type = fvg['type']
        walk_end = max(fvg_idx - max_wb, -1)

        for j in range(fvg_idx, walk_end, -1):
            if j in seen_idx:
                break

            o, c, h, low = O[j], C[j], H[j], L[j]

            if fvg_type == 'bearish':
                is_opposing = c > o
                ob_type = 'bearish'
            else:
                is_opposing = c < o
                ob_type = 'bullish'

            if not is_opposing:
                continue

            ob_body_top = max(o, c)
            ob_body_bot = min(o, c)

            if ob_type == 'bearish' and current_close > ob_body_top:
                break
            if ob_type == 'bullish' and current_close < ob_body_bot:
                break

            seen_idx.add(j)
            obs.append({
                'type': ob_type,
                'ob_top': round(h, 6),
                'ob_bottom': round(low, 6),
                'ob_body_top': round(ob_body_top, 6),
                'ob_body_bottom': round(ob_body_bot, 6),
                'ts': str(df.index[j]),
                'idx': j,
                'related_fvg_top': fvg['fvg_top'],
                'related_fvg_bottom': fvg['fvg_bottom'],
                'fvg_mitigated': fvg.get('mitigated', False)
            })
            break

    return obs


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LIQUIDITY SWEEP DETECTION (2.5 / 1.5 / 0)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_liquidity_sweep(df, swing_points, pair_conf, atr_value=None):
    """
    Detect liquidity sweeps with graded scoring.

    CONFIRMED (2.5 pts):
      Wick pierced PAST swing point, close back inside,
      wick >= 1.5x body, body >= 10% ATR.

    NEAR-MISS (1.5 pts):
      Wick within proximity % of swing point (did NOT pierce),
      close on correct side, wick >= 1.5x body, body >= 10% ATR.

    MULTI-CANDLE (2.5 pts):
      Any of last 3 candles wicked past level AND latest close back inside.

    Returns: {detected, sweep_score, sweep_type, sweep_price, level_swept,
              candle_ts, method}
    """
    result = {
        "detected": False, "sweep_score": 0.0, "sweep_type": None,
        "sweep_price": None, "level_swept": None,
        "candle_ts": None, "method": None
    }

    if df is None or len(df) < 3 or not swing_points:
        return result

    sh = [s for s in swing_points if s["type"] == "high"]
    sl = [s for s in swing_points if s["type"] == "low"]
    if not sh and not sl:
        return result

    H = df['High'].values.flatten().astype(float)
    L = df['Low'].values.flatten().astype(float)
    O = df['Open'].values.flatten().astype(float)
    C = df['Close'].values.flatten().astype(float)
    n = len(df)

    min_body = (SWEEP_MIN_BODY_ATR * atr_value
                if atr_value and atr_value > 0 else 0)
    prox_pct = SWEEP_PROXIMITY_PCT.get(_pair_tol_key(pair_conf), 0.06) / 100

    best = result

    # ── Single-candle sweeps ──────────────────────────────────────────────
    for i in range(n - 1, max(n - SWEEP_CANDLE_WINDOW - 1, -1), -1):
        raw_body = abs(C[i] - O[i])
        body = max(raw_body, min_body)
        upper_wick = H[i] - max(O[i], C[i])
        lower_wick = min(O[i], C[i]) - L[i]
        reject_up = upper_wick >= body * SWEEP_WICK_BODY_RATIO
        reject_dn = lower_wick >= body * SWEEP_WICK_BODY_RATIO

        # ── Swing HIGH sweeps ─────────────────────────────────────────
        for s in sh:
            if s["idx"] >= i:
                continue
            level = s["price"]

            # Confirmed: wick above level, close below
            if H[i] > level and C[i] < level and reject_up:
                if best["sweep_score"] < 2.5:
                    best = {
                        "detected": True, "sweep_score": 2.5,
                        "sweep_type": "high",
                        "sweep_price": float(H[i]),
                        "level_swept": level,
                        "candle_ts": str(df.index[i]),
                        "method": "single_confirmed"
                    }

            # Near-miss: wick approached but didn't pierce
            elif reject_up and best["sweep_score"] < 1.5:
                distance = abs(H[i] - level)
                threshold = level * prox_pct
                if (distance <= threshold and H[i] <= level
                        and C[i] < level):
                    best = {
                        "detected": True, "sweep_score": 1.5,
                        "sweep_type": "high",
                        "sweep_price": float(H[i]),
                        "level_swept": level,
                        "candle_ts": str(df.index[i]),
                        "method": "single_near_miss"
                    }

        # ── Swing LOW sweeps ──────────────────────────────────────────
        for s in sl:
            if s["idx"] >= i:
                continue
            level = s["price"]

            # Confirmed: wick below level, close above
            if L[i] < level and C[i] > level and reject_dn:
                if best["sweep_score"] < 2.5:
                    best = {
                        "detected": True, "sweep_score": 2.5,
                        "sweep_type": "low",
                        "sweep_price": float(L[i]),
                        "level_swept": level,
                        "candle_ts": str(df.index[i]),
                        "method": "single_confirmed"
                    }

            # Near-miss
            elif reject_dn and best["sweep_score"] < 1.5:
                distance = abs(L[i] - level)
                threshold = level * prox_pct
                if (distance <= threshold and L[i] >= level
                        and C[i] > level):
                    best = {
                        "detected": True, "sweep_score": 1.5,
                        "sweep_type": "low",
                        "sweep_price": float(L[i]),
                        "level_swept": level,
                        "candle_ts": str(df.index[i]),
                        "method": "single_near_miss"
                    }

    # ── Multi-candle sweep ────────────────────────────────────────────────
    if best["sweep_score"] < 2.5:
        latest_close = C[-1]
        for i in range(n - 1, max(n - SWEEP_CANDLE_WINDOW - 1, -1), -1):
            for s in sh:
                if s["idx"] >= i:
                    continue
                if H[i] > s["price"] and latest_close < s["price"]:
                    if best["sweep_score"] < 2.5:
                        best = {
                            "detected": True, "sweep_score": 2.5,
                            "sweep_type": "high",
                            "sweep_price": float(H[i]),
                            "level_swept": s["price"],
                            "candle_ts": str(df.index[i]),
                            "method": "multi_candle"
                        }
            for s in sl:
                if s["idx"] >= i:
                    continue
                if L[i] < s["price"] and latest_close > s["price"]:
                    if best["sweep_score"] < 2.5:
                        best = {
                            "detected": True, "sweep_score": 2.5,
                            "sweep_type": "low",
                            "sweep_price": float(L[i]),
                            "level_swept": s["price"],
                            "candle_ts": str(df.index[i]),
                            "method": "multi_candle"
                        }

    return best


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PREMIUM / DISCOUNT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_premium_discount(swing_points, current_price, bias):
    """
    Check if price is in the correct zone for the given bias.
    Uses 60/40 split (10% tolerance) for institutional front-running.

    LONG:  valid if price is below 60% of range
    SHORT: valid if price is above 40% of range
    """
    sh = [s for s in swing_points if s["type"] == "high"]
    sl = [s for s in swing_points if s["type"] == "low"]

    fail = {"valid": False, "position_pct": 50.0, "range_high": 0,
            "range_low": 0, "equilibrium": 0, "score": 0.0}

    if not sh or not sl:
        return fail

    range_high = max(s["price"] for s in sh)
    range_low = min(s["price"] for s in sl)

    if range_high <= range_low:
        return fail

    eq = (range_high + range_low) / 2
    position = (current_price - range_low) / (range_high - range_low)
    position_pct = round(position * 100, 1)

    if bias == "LONG":
        valid = position < (0.50 + PD_TOLERANCE)
    elif bias == "SHORT":
        valid = position > (0.50 - PD_TOLERANCE)
    else:
        valid = False

    return {
        "valid": valid,
        "position_pct": position_pct,
        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": eq,
        "score": 1.5 if valid else 0.0
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SELECT BEST OB FOR ZONE
# ═══════════════════════════════════════════════════════════════════════════════

def select_trade_ob(obs, zone_level, bias, pair_conf):
    """
    Pick the best OB for this zone and bias direction.
    OB type must match bias (bullish OB → LONG, bearish OB → SHORT).
    Closest to zone_level wins.
    """
    matching = []
    for ob in obs:
        if bias == "LONG" and ob["type"] != "bullish":
            continue
        if bias == "SHORT" and ob["type"] != "bearish":
            continue
        matching.append(ob)

    if not matching:
        return None

    matching.sort(key=lambda o: abs(
        ((o["ob_top"] + o["ob_bottom"]) / 2) - zone_level))
    return matching[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 10. COMPUTE ENTRY / SL / TP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_levels(ob, fvgs, all_obs, zones, bias, current_price,
                   pair_conf, zone_level):
    """
    Compute entry, SL, TP1, TP2.

    ENTRY: OB body midpoint, or FVG edge if unmitigated FVG overlaps OB.
    SL:    OB wick extreme (low for LONG, high for SHORT).
    TP1:   Nearest OPPOSING structure:
             1. H1 zone clusters  2. M15 opposing FVGs  3. M15 opposing OBs
             4. Fallback 2.5R
    TP2:   Next opposing structure beyond TP1, or fallback 4R.

    Returns dict or None if risk is zero.
    """
    if ob is None:
        return None

    dp = _dp(pair_conf)
    ob_body_mid = (ob["ob_body_top"] + ob["ob_body_bottom"]) / 2

    # ── Entry ─────────────────────────────────────────────────────────────
    entry = ob_body_mid
    entry_source = "ob_body_midpoint"

    unmitigated = get_unmitigated_fvgs(fvgs) if fvgs else []
    for fvg in unmitigated:
        if (fvg["fvg_bottom"] <= ob["ob_top"]
                and fvg["fvg_top"] >= ob["ob_bottom"]):
            if bias == "LONG":
                entry = fvg["fvg_top"]
            else:
                entry = fvg["fvg_bottom"]
            entry_source = "fvg_edge"
            break

    # ── SL ────────────────────────────────────────────────────────────────
    if bias == "LONG":
        sl = ob["ob_bottom"]
    else:
        sl = ob["ob_top"]

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    # ── Collect opposing TP candidates ────────────────────────────────────
    candidates = []

    # Source 1: H1 zone clusters
    if zones:
        for z_level, z_touches in zones:
            if (zone_level > 0
                    and abs(z_level - zone_level) / zone_level * 100 < 0.3):
                continue
            if bias == "LONG" and z_level > entry:
                candidates.append((z_level, "h1_zone"))
            elif bias == "SHORT" and z_level < entry:
                candidates.append((z_level, "h1_zone"))

    # Source 2: M15 unmitigated OPPOSING FVGs
    for fvg in unmitigated:
        fvg_mid = (fvg["fvg_top"] + fvg["fvg_bottom"]) / 2
        if bias == "LONG" and fvg["type"] == "bearish" and fvg_mid > entry:
            candidates.append((fvg_mid, "m15_opposing_fvg"))
        elif (bias == "SHORT" and fvg["type"] == "bullish"
              and fvg_mid < entry):
            candidates.append((fvg_mid, "m15_opposing_fvg"))

    # Source 3: M15 opposing OBs
    if all_obs:
        for o in all_obs:
            if o is ob:
                continue
            ob_mid = (o["ob_top"] + o["ob_bottom"]) / 2
            if bias == "LONG" and o["type"] == "bearish" and ob_mid > entry:
                candidates.append((ob_mid, "m15_opposing_ob"))
            elif (bias == "SHORT" and o["type"] == "bullish"
                  and ob_mid < entry):
                candidates.append((ob_mid, "m15_opposing_ob"))

    # ── Sort by proximity ─────────────────────────────────────────────────
    if bias == "LONG":
        candidates.sort(key=lambda c: c[0])
    else:
        candidates.sort(key=lambda c: c[0], reverse=True)

    # ── Assign TP1, TP2 ──────────────────────────────────────────────────
    tp1, tp1_source = None, "none"
    tp2, tp2_source = None, "none"

    if len(candidates) >= 1:
        tp1 = candidates[0][0]
        tp1_source = candidates[0][1]
    if len(candidates) >= 2:
        tp2 = candidates[1][0]
        tp2_source = candidates[1][1]

    # ── Fallbacks ─────────────────────────────────────────────────────────
    if tp1 is None:
        if bias == "LONG":
            tp1 = entry + (risk * TP1_FALLBACK_R)
        else:
            tp1 = entry - (risk * TP1_FALLBACK_R)
        tp1_source = f"fallback_{TP1_FALLBACK_R}R"

    if tp2 is None:
        if bias == "LONG":
            tp2 = entry + (risk * TP2_FALLBACK_R)
        else:
            tp2 = entry - (risk * TP2_FALLBACK_R)
        tp2_source = f"fallback_{TP2_FALLBACK_R}R"

    rr1 = abs(tp1 - entry) / risk
    rr2 = abs(tp2 - entry) / risk

    if rr2 <= rr1:
        target_rr2 = max(rr1 + 1.0, TP2_FALLBACK_R)
        if bias == "LONG":
            tp2 = entry + (risk * target_rr2)
        else:
            tp2 = entry - (risk * target_rr2)
        rr2 = abs(tp2 - entry) / risk
        tp2_source = "adjusted_beyond_tp1"

    return {
        "entry": round(entry, 6),
        "sl": round(sl, 6),
        "tp1": round(tp1, 6),
        "tp2": round(tp2, 6),
        "risk": round(risk, 6),
        "rr_tp1": round(rr1, 2),
        "rr_tp2": round(rr2, 2),
        "entry_source": entry_source,
        "tp1_source": tp1_source,
        "tp2_source": tp2_source
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 11. ENTRY READINESS
# ═══════════════════════════════════════════════════════════════════════════════

def check_entry_readiness(df, ob, zone_level, bias, pair_conf,
                          atr_value=None, bos_choch_result=None):
    """
    Check if the LATEST CLOSED candle shows a confirmation pattern.

    Ready if ANY of:
      1. BOS/CHoCH/iBOS in last 2 candles, direction matches bias
      2. Engulfing on last closed candle, correct direction, body >= 20% ATR

    Only closed candles. Current forming candle ignored.
    """
    result = {
        "ready": False, "trigger_kind": None,
        "trigger_tf": "M15", "trigger_candle_ts": None, "reason": ""
    }

    if df is None or len(df) < 3:
        result["reason"] = "Insufficient data for trigger check"
        return result

    n = len(df)

    # ── Check 1: Recent structural break ──────────────────────────────────
    if bos_choch_result and bos_choch_result.get("confirmed"):
        break_idx = bos_choch_result.get("break_candle_idx")
        if break_idx is not None and break_idx >= n - 3:
            bdir = bos_choch_result.get("direction", "")
            if ((bias == "LONG" and bdir == "bullish") or
                    (bias == "SHORT" and bdir == "bearish")):
                btype = bos_choch_result["type"]
                result["ready"] = True
                result["trigger_kind"] = btype
                result["trigger_candle_ts"] = (
                    bos_choch_result["break_candle_ts"])
                result["reason"] = (
                    f"{btype.upper()} {bdir} confirmed near zone")
                return result

    # ── Check 2: Engulfing ────────────────────────────────────────────────
    O = df['Open'].values.flatten().astype(float)
    C = df['Close'].values.flatten().astype(float)

    i = n - 2
    prev = i - 1

    if prev >= 0:
        curr_body_top = max(O[i], C[i])
        curr_body_bot = min(O[i], C[i])
        prev_body_top = max(O[prev], C[prev])
        prev_body_bot = min(O[prev], C[prev])
        curr_body = curr_body_top - curr_body_bot

        min_body = (ENGULF_MIN_BODY_ATR * atr_value
                    if atr_value and atr_value > 0 else 0)

        is_engulfing = (curr_body_top >= prev_body_top and
                        curr_body_bot <= prev_body_bot and
                        curr_body >= min_body)

        if is_engulfing:
            if bias == "LONG" and C[i] > O[i]:
                result["ready"] = True
                result["trigger_kind"] = "engulf"
                result["trigger_candle_ts"] = str(df.index[i])
                result["reason"] = (
                    "Bullish engulfing on last closed M15 candle")
                return result
            elif bias == "SHORT" and C[i] < O[i]:
                result["ready"] = True
                result["trigger_kind"] = "engulf"
                result["trigger_candle_ts"] = str(df.index[i])
                result["reason"] = (
                    "Bearish engulfing on last closed M15 candle")
                return result

    result["reason"] = "No trigger pattern on last closed candle"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 12. INVALIDATION LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_invalidation(ob, bias, pair_conf):
    """
    Direction-aware invalidation from OB extremes.

    SHORT: invalidate_above = OB top
    LONG:  invalidate_below = OB bottom
    Opposite side always None.
    """
    if ob is None:
        return {"invalidate_above": None, "invalidate_below": None}

    if bias == "SHORT":
        return {"invalidate_above": round(ob["ob_top"], 6),
                "invalidate_below": None}
    elif bias == "LONG":
        return {"invalidate_above": None,
                "invalidate_below": round(ob["ob_bottom"], 6)}
    return {"invalidate_above": None, "invalidate_below": None}


# ═══════════════════════════════════════════════════════════════════════════════
# 13. SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════

def compute_score(analysis, fatigue_count, in_session, fatigue_threshold=3):
    """
    Weighted scorecard from Python-detected patterns.

    Python (6 of 7 items, max 9.0):
      1. liquidity_swept:      2.5 / 1.5 / 0
      2. fvg_overlaps_ob:      2.0 / 0
      3. premium_discount:     1.5 / 0
      4. multi_tf_alignment:   1.5 / 1.0 / 0.5 / 0
      5. zone_freshness:       1.0 / 0
      6. session_alignment:    0.5 / 0

    Gemini (1 item):
      7. no_high_impact_news:  1.0 / 0 (default 1.0, Gemini overrides)

    multi_tf tiers:
      1.5 = M15 external break + H1 same direction
      1.0 = M15 iBOS + H1 same direction
      0.5 = only one TF confirms
      0.0 = neither confirms
    """
    breakdown = {}
    bias = analysis.get("bias", "")

    # 1. Liquidity swept (graded)
    sweep = analysis.get("sweep", {})
    breakdown["liquidity_swept"] = sweep.get("sweep_score", 0.0)

    # 2. FVG overlaps OB
    breakdown["fvg_overlaps_ob"] = (
        2.0 if analysis.get("fvg_overlaps_ob", False) else 0.0)

    # 3. Premium/Discount
    pd = analysis.get("premium_discount", {})
    breakdown["premium_discount"] = pd.get("score", 0.0)

    # 4. Multi-TF alignment
    m15 = analysis.get("m15_structure", {})
    h1 = analysis.get("h1_structure", {})

    m15_confirmed = (m15.get("confirmed", False) and
                     ((bias == "LONG" and m15.get("direction") == "bullish")
                      or (bias == "SHORT"
                          and m15.get("direction") == "bearish")))
    h1_confirmed = (h1.get("confirmed", False) and
                    ((bias == "LONG" and h1.get("direction") == "bullish")
                     or (bias == "SHORT"
                         and h1.get("direction") == "bearish")))
    m15_external = m15.get("break_level") == "external"

    if m15_confirmed and m15_external and h1_confirmed:
        breakdown["multi_tf_alignment"] = 1.5
    elif m15_confirmed and h1_confirmed:
        breakdown["multi_tf_alignment"] = 1.0
    elif m15_confirmed or h1_confirmed:
        breakdown["multi_tf_alignment"] = 0.5
    else:
        breakdown["multi_tf_alignment"] = 0.0

    # 5. Zone freshness
    breakdown["zone_freshness"] = (
        0.0 if fatigue_count >= fatigue_threshold else 1.0)

    # 6. Session alignment
    breakdown["session_alignment"] = 0.5 if in_session else 0.0

    # 7. News — placeholder, Gemini overrides
    breakdown["no_high_impact_news"] = 1.0

    total = round(sum(breakdown.values()), 1)

    return {
        "breakdown": breakdown,
        "total": total,
        "news_score_placeholder": True
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 14. MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(pair_conf, df_h1, df_m15, zones, current_price, zone_level,
                 zone_label, atr_value, fatigue_count, in_session,
                 fatigue_threshold=3, lookback=5):
    """
    Complete Python-side SMC analysis for one zone.

    Flow: bias → gates → detectors → score → levels → readiness → result.

    gemini_needed = True when gates pass and score meets threshold.
    Gemini fills text fields + news scoring only.
    """
    dp = _dp(pair_conf)
    min_conf = pair_conf.get("min_confidence", 7)

    # ── Bias ──────────────────────────────────────────────────────────────
    if "Demand" in zone_label:
        bias = "LONG"
    elif "Supply" in zone_label:
        bias = "SHORT"
    else:
        return _fail_result("Cannot determine bias from zone label")

    # ══════════════════════════════════════════════════════════════════════
    # GATE 1: Structure on M15
    # ══════════════════════════════════════════════════════════════════════
    m15_struct = detect_bos_choch(df_m15, pair_conf, atr_value, lookback)

    if not m15_struct["confirmed"]:
        return _gate_fail("structure", m15_struct["reason"], bias)

    if bias == "LONG" and m15_struct["direction"] != "bullish":
        return _gate_fail(
            "structure",
            f"M15 {m15_struct['type']} is {m15_struct['direction']}, "
            f"need bullish for LONG",
            bias)
    if bias == "SHORT" and m15_struct["direction"] != "bearish":
        return _gate_fail(
            "structure",
            f"M15 {m15_struct['type']} is {m15_struct['direction']}, "
            f"need bearish for SHORT",
            bias)

    if not m15_struct["has_displacement"]:
        return _gate_fail(
            "structure",
            f"M15 {m15_struct['type']} lacks displacement "
            f"(body < {DISPLACEMENT_ATR_FRACTION} x ATR)",
            bias)

    gate_structure = (
        f"M15 {m15_struct['type'].upper()} {m15_struct['direction']} "
        f"({m15_struct['break_level']}) at "
        f"{m15_struct['break_candle_ts']}, "
        f"broke {m15_struct['swing_broken']:.{dp}f}")

    # ══════════════════════════════════════════════════════════════════════
    # GATE 2: Valid OB on M15
    # ══════════════════════════════════════════════════════════════════════
    m15_fvgs = detect_fvgs(df_m15, FVG_M15_WINDOW)
    m15_obs = detect_obs(df_m15, m15_fvgs, OB_MAX_WALK_BACK)

    ob = select_trade_ob(m15_obs, zone_level, bias, pair_conf)

    if ob is None:
        return _gate_fail(
            "ob",
            f"No valid {bias} OB near zone {zone_level:.{dp}f} "
            f"({len(m15_obs)} OBs detected, none match bias/zone)",
            bias)

    gate_ob = (
        f"{ob['type']} OB at "
        f"{ob['ob_top']:.{dp}f}/{ob['ob_bottom']:.{dp}f}, "
        f"candle {ob['ts']}")

    # ══════════════════════════════════════════════════════════════════════
    # GATE 3: RR >= 2.0
    # ══════════════════════════════════════════════════════════════════════
    unmitigated_fvgs = get_unmitigated_fvgs(m15_fvgs)
    levels = compute_levels(ob, m15_fvgs, m15_obs, zones, bias,
                            current_price, pair_conf, zone_level)

    if levels is None:
        return _gate_fail("rr", "Could not compute levels (zero risk)",
                          bias)

    if levels["rr_tp1"] < 2.0:
        return _gate_fail(
            "rr",
            f"RR {levels['rr_tp1']:.2f}:1 < 2.0 | "
            f"entry={levels['entry']:.{dp}f} sl={levels['sl']:.{dp}f} "
            f"tp1={levels['tp1']:.{dp}f} risk={levels['risk']:.{dp}f}",
            bias)

    gate_rr = (
        f"RR={levels['rr_tp1']:.2f}:1 | "
        f"entry={levels['entry']:.{dp}f} "
        f"sl={levels['sl']:.{dp}f} tp1={levels['tp1']:.{dp}f}")

    # ══════════════════════════════════════════════════════════════════════
    # ALL GATES PASSED
    # ══════════════════════════════════════════════════════════════════════
    h1_struct = detect_bos_choch(df_h1, pair_conf, atr_value, lookback)

    m15_swings = get_swing_points(df_m15, lookback)
    h1_swings = get_swing_points(df_h1, lookback)

    sweep = detect_liquidity_sweep(df_m15, m15_swings, pair_conf, atr_value)
    pd_result = compute_premium_discount(h1_swings, current_price, bias)

    # FVG-OB overlap check
    fvg_overlap = levels["entry_source"] == "fvg_edge"
    matched_fvg = {"top": 0.0, "bot": 0.0, "type": ""}
    for fvg in unmitigated_fvgs:
        if (fvg["fvg_bottom"] <= ob["ob_top"]
                and fvg["fvg_top"] >= ob["ob_bottom"]):
            if not fvg_overlap:
                fvg_overlap = True
            matched_fvg = {"top": fvg["fvg_top"],
                           "bot": fvg["fvg_bottom"],
                           "type": fvg["type"]}
            break

    # ── Score ─────────────────────────────────────────────────────────────
    score_result = compute_score(
        {
            "sweep": sweep,
            "fvg_overlaps_ob": fvg_overlap,
            "premium_discount": pd_result,
            "m15_structure": m15_struct,
            "h1_structure": h1_struct,
            "bias": bias
        },
        fatigue_count, in_session, fatigue_threshold)

    # ── Entry readiness ───────────────────────────────────────────────────
    readiness = check_entry_readiness(
        df_m15, ob, zone_level, bias, pair_conf, atr_value, m15_struct)

    # ── Invalidation ─────────────────────────────────────────────────────
    inv = compute_invalidation(ob, bias, pair_conf)

    # ── Confluences ───────────────────────────────────────────────────────
    confluences = []
    if sweep["detected"]:
        label = ("confirmed" if sweep["sweep_score"] >= 2.5
                 else "near-miss")
        confluences.append(
            f"Liquidity swept ({label}): {sweep['sweep_type']} at "
            f"{sweep['level_swept']:.{dp}f}, "
            f"wick to {sweep['sweep_price']:.{dp}f}")
    if fvg_overlap:
        confluences.append(
            f"FVG overlaps OB: {matched_fvg['type']} FVG "
            f"{matched_fvg['bot']:.{dp}f}-{matched_fvg['top']:.{dp}f}")
    if pd_result["valid"]:
        confluences.append(
            f"Price in {'discount' if bias == 'LONG' else 'premium'} "
            f"({pd_result['position_pct']}% of range)")
    mtf = score_result["breakdown"]["multi_tf_alignment"]
    if mtf >= 1.5:
        confluences.append(
            f"Both H1 and M15 aligned "
            f"(M15 {m15_struct['type']} {m15_struct['break_level']})")
    elif mtf >= 1.0:
        confluences.append("M15 iBOS + H1 aligned (internal pullback)")
    elif mtf >= 0.5:
        tf = "M15" if m15_struct["confirmed"] else "H1"
        confluences.append(f"Single TF aligned ({tf})")

    # ── Missing ───────────────────────────────────────────────────────────
    missing = []
    if not sweep["detected"]:
        missing.append({
            "item": "liquidity_swept",
            "reason": "No wick-rejection sweep of prior swing point"})
    elif sweep["sweep_score"] < 2.5:
        missing.append({
            "item": "liquidity_swept_partial",
            "reason": "Near-miss sweep — scored 1.5/2.5"})
    if not fvg_overlap:
        missing.append({
            "item": "fvg_overlaps_ob",
            "reason": "No unmitigated FVG overlapping the selected OB"})
    if not pd_result["valid"]:
        side = "premium" if bias == "LONG" else "discount"
        missing.append({
            "item": "premium_discount",
            "reason": (f"Price at {pd_result['position_pct']}% "
                       f"of range ({side})")})
    if mtf < 1.5:
        if mtf >= 1.0:
            missing.append({
                "item": "multi_tf_alignment",
                "reason": "M15 is iBOS (internal) — not full external"})
        elif mtf >= 0.5:
            missing.append({
                "item": "multi_tf_alignment",
                "reason": "H1 and M15 not both confirming"})
        else:
            missing.append({
                "item": "multi_tf_alignment",
                "reason": "No structural alignment on either TF"})
    if score_result["breakdown"]["zone_freshness"] == 0:
        missing.append({
            "item": "zone_freshness",
            "reason": f"Zone tested {fatigue_count}x in 30 days"})

    # ══════════════════════════════════════════════════════════════════════
    # RESULT
    # ══════════════════════════════════════════════════════════════════════
    return {
        "send_alert": True,
        "gates_passed": True,
        "gate_structure": gate_structure,
        "gate_ob": gate_ob,
        "gate_rr": gate_rr,

        "bias": bias,
        "confidence_score": score_result["total"],
        "score_breakdown": score_result["breakdown"],
        "news_score_placeholder": True,

        "entry": levels["entry"],
        "entry_model": "limit",
        "entry_ready_now": readiness["ready"],
        "entry_source": levels["entry_source"],

        "trigger_status": (
            "ready" if readiness["ready"] else "not_ready"),
        "trigger_tf": readiness["trigger_tf"],
        "trigger_kind": readiness.get("trigger_kind") or "",

        "sl": levels["sl"],
        "tp1": levels["tp1"],
        "tp2": levels["tp2"],
        "rr_tp1": f"{levels['rr_tp1']:.2f}",
        "rr_tp2": f"{levels['rr_tp2']:.2f}",
        "tp1_source": levels["tp1_source"],
        "tp2_source": levels["tp2_source"],

        "invalidate_above": inv["invalidate_above"],
        "invalidate_below": inv["invalidate_below"],

        "ob_top": ob["ob_top"],
        "ob_bottom": ob["ob_bottom"],
        "ob_type": ob["type"],
        "ob_confirmed": True,
        "ob_body_top": ob["ob_body_top"],
        "ob_body_bottom": ob["ob_body_bottom"],

        "fvg_top": matched_fvg["top"],
        "fvg_bottom": matched_fvg["bot"],
        "fvg_type": matched_fvg["type"],
        "fvg_confirmed": fvg_overlap,

        "lq_sweep_price": sweep.get("sweep_price") or 0,

        "premium_discount_pct": pd_result["position_pct"],
        "premium_discount_valid": pd_result["valid"],

        "m15_structure_detail": m15_struct,
        "h1_structure_detail": h1_struct,
        "m15_break_level": m15_struct.get("break_level", "unknown"),
        "sweep_detail": sweep,
        "readiness_detail": readiness,

        "gemini_needed": True,

        # Placeholder text — Gemini fills in Phase 3
        "confidence_reason": readiness.get("reason", ""),
        "bias_reason": "",
        "trigger": readiness.get("reason", ""),
        "invalid_if": "",
        "macro_line1": "",
        "macro_line2": "",
        "mindset": "",
        "missing": missing,
        "news_flag": "none",
        "geo_flag": False,
        "confluences": confluences,
        "sl_note": (f"SL at OB {'low' if bias == 'LONG' else 'high'} "
                    f"({levels['sl']:.{dp}f})")
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GATE-FAIL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _gate_fail(gate_name, reason, bias):
    """Standard gate-fail — short-form, no Gemini needed."""
    return {
        "send_alert": False,
        "gates_passed": False,
        "gate_structure": (
            False if gate_name == "structure" else "not_checked"),
        "gate_ob": False if gate_name == "ob" else "not_checked",
        "gate_rr": False if gate_name == "rr" else "not_checked",
        "confidence_score": 0.0,
        "confidence_reason": reason,
        "bias": bias,
        "gemini_needed": False
    }


def _fail_result(reason):
    """Generic fail — cannot determine bias."""
    return {
        "send_alert": False,
        "gates_passed": False,
        "gate_structure": False,
        "gate_ob": False,
        "gate_rr": False,
        "confidence_score": 0.0,
        "confidence_reason": reason,
        "bias": "",
        "gemini_needed": False
    }
