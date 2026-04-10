"""
smc_detector.py — Pure Python SMC Pattern Detection Engine v2.2
================================================================
Zero Gemini calls. Every function is deterministic and testable.
Pair-type aware: forex, forex_jpy, index, commodity.

v2.2 changes from v2.1:
  - BOS/CHoCH scan window now configurable via scan_window param (was hardcoded 15)
    Default: 100 for M15 (~25h), 50 for H1 (~50h)
  - select_trade_ob: range overlap replaces midpoint distance
    OB range (ob_bottom to ob_top) must overlap zone band (zone ± prox%)
  - Swing lookback default reduced from 15 to 4 (config-driven)
  - run_analysis accepts bos_scan_m15 / bos_scan_h1 params

Functions:
    1.  get_swing_points          — swing high/low detection
    2.  _determine_trend          — HH/HL/LH/LL trend classification
    3.  _find_strong_weak_points  — Strong Point (ERL) and Weak Point
    4.  detect_bos_choch          — BOS / CHoCH / iBOS (single best)
    5.  detect_bos_choch_all      — ALL confirmed breaks (for scoring)
    6.  detect_fvgs               — Fair Value Gaps with mitigation tracking
    7.  get_unmitigated_fvgs      — filter helper
    8.  detect_obs_from_structure  — OB from structural break candle
    9.  detect_liquidity_sweep    — wick-rejection sweep (2.5 / 1.5 / 0)
    10. compute_premium_discount  — 60/40 equilibrium check
    11. select_trade_ob           — validate OB overlap with zone band
    12. compute_levels            — entry / SL / TP1 / TP2 / RR
    13. check_entry_readiness     — engulfing or BOS/CHoCH trigger
    14. compute_invalidation      — direction-aware invalidation levels
    15. compute_capped_atr        — ATR capped at median * 1.5
    16. compute_score             — full weighted scorecard (news → Gemini)
    17. run_analysis              — main orchestrator
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# PAIR-AWARE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# BOS/CHoCH displacement: breaking candle body must be >= this fraction of
# H1 ATR(14). Lowered from 0.25 to catch more valid breaks.
DISPLACEMENT_ATR_FRACTION = 0.15

# BOS tolerance: how far past the swing body extreme the close must be.
BOS_TOLERANCE = {
    "forex":     0.00010,   # 1 pip
    "forex_jpy": 0.010,     # 1 pip (JPY scale)
    "index":     5.0,       # 5 points (NAS100)
    "commodity": 0.50,      # $0.50 (Gold)
}

# Liquidity sweep: wick must be >= this multiple of body for rejection.
SWEEP_WICK_BODY_RATIO = 1.5

# Sweep: minimum body size as fraction of ATR (anti-doji guard).
SWEEP_MIN_BODY_ATR = 0.10

# Sweep: how many recent candles to check for sweep evidence.
SWEEP_CANDLE_WINDOW = 3

# Sweep: proximity threshold for near-miss scoring (1.5 pts).
SWEEP_PROXIMITY_PCT = {
    "forex":     0.06,
    "forex_jpy": 0.08,
    "index":     0.12,
    "commodity": 0.10,
}

# Premium/Discount: tolerance band around equilibrium.
PD_TOLERANCE = 0.10

# OB: max candles to walk backward from structural break candle.
OB_MAX_WALK_BACK = 15

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

def get_swing_points(df, lookback=4):
    """
    Detect swing highs and swing lows from OHLC data.

    A swing high at index i: High[i] is the max in [i-lookback : i+lookback+1].
    A swing low  at index i: Low[i]  is the min in [i-lookback : i+lookback+1].

    Returns list of dicts sorted by index:
        {type, price, body_extreme, idx, ts}

    body_extreme = max(Open, Close) for highs, min(Open, Close) for lows.
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
# SHARED: Build break targets
# ═══════════════════════════════════════════════════════════════════════════════

def _build_break_targets(swings, strong_point, weak_point, sh, sl, trend):
    """Build target list for break detection. Used by both detect_bos_choch and detect_bos_choch_all."""
    targets = []

    if strong_point is not None and weak_point is not None:
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

    return targets


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BOS / CHoCH / iBOS DETECTION (single best)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_bos_choch(df, pair_conf, atr_value=None, lookback=4, scan_window=100):
    """
    Detect the most recent structural break on the given timeframe.
    Scans last `scan_window` closed candles. Priority: external > internal,
    most recent wins for same level.

    scan_window: how many candles to scan backward (default 100 for M15, use 50 for H1).
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

    targets = _build_break_targets(swings, strong_point, weak_point, sh, sl, trend)

    if not targets:
        empty["reason"] = "No swing targets for break detection"
        return empty

    best = None

    for i in range(n - 2, max(n - scan_window - 2, -1), -1):
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

    empty["reason"] = f"No structural break in last {scan_window} closed candles"
    return empty


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BOS / CHoCH — ALL BREAKS (for scoring)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_bos_choch_all(df, pair_conf, atr_value=None, lookback=4, scan_window=100):
    """
    Returns ALL confirmed structural breaks (with displacement) in the
    last `scan_window` closed candles. Used for scoring item 1 — determines
    if both BOS and CHoCH are present for the given bias.
    """
    result = {
        "has_bos_bullish": False,
        "has_bos_bearish": False,
        "has_choch_bullish": False,
        "has_choch_bearish": False,
        "all_breaks": []
    }

    if df is None or len(df) < lookback * 2 + 5:
        return result

    swings = get_swing_points(df, lookback)
    sh = [s for s in swings if s["type"] == "high"]
    sl = [s for s in swings if s["type"] == "low"]

    if len(sh) < 2 or len(sl) < 2:
        return result

    trend = _determine_trend(swings)
    strong_point, weak_point = _find_strong_weak_points(swings, trend)

    tol = BOS_TOLERANCE.get(_pair_tol_key(pair_conf), 0.0001)
    disp_thresh = (DISPLACEMENT_ATR_FRACTION * atr_value
                   if atr_value and atr_value > 0 else 0)

    C_arr = df['Close'].values.flatten().astype(float)
    O_arr = df['Open'].values.flatten().astype(float)
    n = len(df)

    targets = _build_break_targets(swings, strong_point, weak_point, sh, sl, trend)

    if not targets:
        return result

    all_breaks = []

    for i in range(n - 2, max(n - scan_window - 2, -1), -1):
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

            if not broken or not has_disp:
                continue

            all_breaks.append({
                "type": t["break_type"],
                "direction": t["direction"],
                "break_level": t["break_level"],
                "break_price": float(C_arr[i]),
                "break_candle_idx": i,
                "swing_broken": t["price"]
            })

    for brk in all_breaks:
        if brk["type"] == "bos" and brk["direction"] == "bullish":
            result["has_bos_bullish"] = True
        elif brk["type"] == "bos" and brk["direction"] == "bearish":
            result["has_bos_bearish"] = True
        elif brk["type"] == "choch" and brk["direction"] == "bullish":
            result["has_choch_bullish"] = True
        elif brk["type"] == "choch" and brk["direction"] == "bearish":
            result["has_choch_bearish"] = True

    result["all_breaks"] = all_breaks
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FVG DETECTION + DIRECTION-AWARE MITIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_fvgs(df, window=None):
    """
    Detect Fair Value Gaps with direction-aware mitigation tracking.
    """
    fvgs = []
    if df is None or len(df) < 3:
        return fvgs

    n = len(df)
    start = max(0, n - (window or FVG_M15_WINDOW))

    H = df['High'].values.flatten().astype(float)
    L = df['Low'].values.flatten().astype(float)

    for i in range(start, n - 2):
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
# 7. ORDER BLOCK DETECTION (structure-anchored)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_obs_from_structure(df, bos_choch_result, bias, pair_conf):
    """
    Find the OB by walking backward from the structural break candle.

    The OB = last opposing candle before the break.
    For LONG bias: last bearish candle (Close < Open) → bullish OB
    For SHORT bias: last bullish candle (Close > Open) → bearish OB

    Walk back up to OB_MAX_WALK_BACK (15) candles from break candle.

    Mitigation check: if current price has closed through OB body, discard.

    Returns OB dict or None.
    """
    if not bos_choch_result or not bos_choch_result.get("confirmed"):
        return None

    if df is None or len(df) == 0:
        return None

    break_idx = bos_choch_result.get("break_candle_idx")
    if break_idx is None or break_idx < 1:
        return None

    O = df['Open'].values.flatten().astype(float)
    C = df['Close'].values.flatten().astype(float)
    H = df['High'].values.flatten().astype(float)
    L = df['Low'].values.flatten().astype(float)
    current_close = float(C[-1])

    if bias == "LONG":
        ob_type = "bullish"
    elif bias == "SHORT":
        ob_type = "bearish"
    else:
        return None

    walk_end = max(break_idx - OB_MAX_WALK_BACK - 1, -1)

    for j in range(break_idx - 1, walk_end, -1):
        if j < 0:
            break

        o, c, h, low = O[j], C[j], H[j], L[j]

        if bias == "LONG" and c >= o:
            continue
        if bias == "SHORT" and c <= o:
            continue

        ob_body_top = max(o, c)
        ob_body_bot = min(o, c)

        if ob_type == "bullish" and current_close < ob_body_bot:
            return None
        if ob_type == "bearish" and current_close > ob_body_top:
            return None

        return {
            "type": ob_type,
            "ob_top": round(float(h), 6),
            "ob_bottom": round(float(low), 6),
            "ob_body_top": round(float(ob_body_top), 6),
            "ob_body_bottom": round(float(ob_body_bot), 6),
            "ts": str(df.index[j]),
            "idx": j,
            "has_fvg_overlap": False,
            "source": "structure"
        }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. LIQUIDITY SWEEP DETECTION (2.5 / 1.5 / 0)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_liquidity_sweep(df, swing_points, pair_conf, atr_value=None):
    """
    Detect liquidity sweeps with graded scoring.
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

    for i in range(n - 1, max(n - SWEEP_CANDLE_WINDOW - 1, -1), -1):
        raw_body = abs(C[i] - O[i])
        body = max(raw_body, min_body)
        upper_wick = H[i] - max(O[i], C[i])
        lower_wick = min(O[i], C[i]) - L[i]
        reject_up = upper_wick >= body * SWEEP_WICK_BODY_RATIO
        reject_dn = lower_wick >= body * SWEEP_WICK_BODY_RATIO

        for s in sh:
            if s["idx"] >= i:
                continue
            level = s["price"]
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

        for s in sl:
            if s["idx"] >= i:
                continue
            level = s["price"]
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
# 9. PREMIUM / DISCOUNT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_premium_discount(swing_points, current_price, bias):
    """
    Check if price is in the correct zone for the given bias.
    Uses 60/40 split (10% tolerance).
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
        "score": 1.0 if valid else 0.0
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SELECT / VALIDATE OB (range overlap with zone band)
# ═══════════════════════════════════════════════════════════════════════════════

def select_trade_ob(ob, zone_level, pair_conf):
    """
    Validate the structure-anchored OB against the zone using range overlap.

    Zone band = zone_level ± proximity_pct%.
    OB range  = ob_bottom to ob_top.
    If these two ranges overlap at all → valid OB for this zone.

    Returns ob if valid, None if no overlap or None input.
    """
    if ob is None:
        return None

    prox_pct = pair_conf.get("proximity_pct", 0.3)
    zone_top = zone_level * (1 + prox_pct / 100)
    zone_bot = zone_level * (1 - prox_pct / 100)

    # Ranges overlap if ob_top >= zone_bot AND ob_bottom <= zone_top
    if ob["ob_top"] < zone_bot or ob["ob_bottom"] > zone_top:
        return None

    return ob


# ═══════════════════════════════════════════════════════════════════════════════
# 11. COMPUTE ENTRY / SL / TP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_levels(ob, fvgs, all_obs, zones, bias, current_price,
                   pair_conf, zone_level):
    """
    Compute entry, SL, TP1, TP2. Returns dict or None if risk is zero.
    """
    if ob is None:
        return None

    dp = _dp(pair_conf)
    ob_body_mid = (ob["ob_body_top"] + ob["ob_body_bottom"]) / 2

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

    if bias == "LONG":
        sl = ob["ob_bottom"]
    else:
        sl = ob["ob_top"]

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    candidates = []

    if zones:
        for z_level, z_touches in zones:
            if (zone_level > 0
                    and abs(z_level - zone_level) / zone_level * 100 < 0.3):
                continue
            if bias == "LONG" and z_level > entry:
                candidates.append((z_level, "h1_zone"))
            elif bias == "SHORT" and z_level < entry:
                candidates.append((z_level, "h1_zone"))

    for fvg in unmitigated:
        fvg_mid = (fvg["fvg_top"] + fvg["fvg_bottom"]) / 2
        if bias == "LONG" and fvg["type"] == "bearish" and fvg_mid > entry:
            candidates.append((fvg_mid, "m15_opposing_fvg"))
        elif (bias == "SHORT" and fvg["type"] == "bullish"
              and fvg_mid < entry):
            candidates.append((fvg_mid, "m15_opposing_fvg"))

    if all_obs:
        for o in all_obs:
            if o is ob:
                continue
            o_mid = (o["ob_top"] + o["ob_bottom"]) / 2
            if bias == "LONG" and o["type"] == "bearish" and o_mid > entry:
                candidates.append((o_mid, "m15_opposing_ob"))
            elif (bias == "SHORT" and o["type"] == "bullish"
                  and o_mid < entry):
                candidates.append((o_mid, "m15_opposing_ob"))

    if bias == "LONG":
        candidates.sort(key=lambda c: c[0])
    else:
        candidates.sort(key=lambda c: c[0], reverse=True)

    tp1, tp1_source = None, "none"
    tp2, tp2_source = None, "none"

    if len(candidates) >= 1:
        tp1 = candidates[0][0]
        tp1_source = candidates[0][1]
    if len(candidates) >= 2:
        tp2 = candidates[1][0]
        tp2_source = candidates[1][1]

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
# 12. ENTRY READINESS
# ═══════════════════════════════════════════════════════════════════════════════

def check_entry_readiness(df, ob, zone_level, bias, pair_conf,
                          atr_value=None, bos_choch_result=None):
    """
    Check if the LATEST CLOSED candle shows a confirmation pattern.
    """
    result = {
        "ready": False, "trigger_kind": None,
        "trigger_tf": "M15", "trigger_candle_ts": None, "reason": ""
    }

    if df is None or len(df) < 3:
        result["reason"] = "Insufficient data for trigger check"
        return result

    n = len(df)

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
# 13. INVALIDATION LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_invalidation(ob, bias, pair_conf):
    """Direction-aware invalidation from OB extremes."""
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
# 14. ATR WITH CAP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_capped_atr(df_h1, period=14):
    """
    Compute ATR(14) on H1 data, capped at median_atr * 1.5.
    Prevents single spike days from inflating the ATR ruler.
    Returns float or None.
    """
    if df_h1 is None or len(df_h1) < period + 1:
        return None

    try:
        highs = df_h1['High'].values.flatten().astype(float)
        lows = df_h1['Low'].values.flatten().astype(float)
        closes = df_h1['Close'].values.flatten().astype(float)

        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
            trs.append(tr)

        if len(trs) < period:
            return None

        latest_atr = float(np.mean(trs[-period:]))

        median_window = min(120, len(trs))
        median_atr = float(np.median(trs[-median_window:]))

        cap = median_atr * 1.5
        capped = min(latest_atr, cap)

        return capped
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 15. SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════

def compute_score(analysis, fatigue_count, fatigue_threshold=5):
    """
    Weighted scorecard — 8 items, max 10.0.
    Session alignment removed. News placeholder = 1.0, Gemini overrides.
    """
    breakdown = {}
    bias = analysis.get("bias", "")

    # 1. Structure M15 — 2.5 max
    bos_all = analysis.get("bos_choch_all", {})
    has_bos = (bos_all.get("has_bos_bullish") if bias == "LONG"
               else bos_all.get("has_bos_bearish"))
    has_choch = (bos_all.get("has_choch_bullish") if bias == "LONG"
                 else bos_all.get("has_choch_bearish"))
    if has_bos and has_choch:
        breakdown["structure_m15"] = 2.5
    elif has_choch:
        breakdown["structure_m15"] = 1.5
    elif has_bos:
        breakdown["structure_m15"] = 1.0
    else:
        breakdown["structure_m15"] = 0.0

    # 2. OB near zone — 2.0 max
    ob = analysis.get("ob")
    ob_has_fvg = analysis.get("ob_has_fvg_overlap", False)
    if ob is not None and ob_has_fvg:
        breakdown["ob_near_zone"] = 2.0
    elif ob is not None:
        breakdown["ob_near_zone"] = 1.0
    else:
        breakdown["ob_near_zone"] = 0.0

    # 3. FVG in zone — 1.0 max (simple existence check)
    breakdown["fvg_in_zone"] = analysis.get("fvg_in_zone_score", 0.0)

    # 4. Liquidity sweep (bias-matched) — 1.5 max
    sweep = analysis.get("sweep", {})
    sweep_correct = False
    if sweep.get("detected"):
        if bias == "LONG" and sweep.get("sweep_type") == "low":
            sweep_correct = True
        elif bias == "SHORT" and sweep.get("sweep_type") == "high":
            sweep_correct = True
    if sweep_correct:
        breakdown["liquidity_sweep"] = (1.5 if sweep["sweep_score"] >= 2.5
                                        else 1.0)
    else:
        breakdown["liquidity_sweep"] = 0.0

    # 5. Premium/Discount — 1.0 max
    pd = analysis.get("premium_discount", {})
    breakdown["premium_discount"] = 1.0 if pd.get("valid") else 0.0

    # 6. H1 alignment — 0.5 max
    h1 = analysis.get("h1_structure", {})
    h1_correct = (h1.get("confirmed") and
                  ((bias == "LONG" and h1.get("direction") == "bullish") or
                   (bias == "SHORT" and h1.get("direction") == "bearish")))
    breakdown["h1_alignment"] = 0.5 if h1_correct else 0.0

    # 7. Zone freshness — 0.5 max
    breakdown["zone_freshness"] = (0.0 if fatigue_count >= fatigue_threshold
                                   else 0.5)

    # 8. News — placeholder, Gemini overrides
    breakdown["no_high_impact_news"] = 1.0

    total = round(sum(breakdown.values()), 1)

    return {
        "breakdown": breakdown,
        "total": total,
        "news_score_placeholder": True
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 16. MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(pair_conf, df_h1, df_m15, zones, current_price, zone_level,
                 zone_label, atr_value, fatigue_count,
                 fatigue_threshold=5, lookback=4,
                 bos_scan_m15=100, bos_scan_h1=50):
    """
    Complete Python-side SMC analysis for one zone.
    bos_scan_m15 / bos_scan_h1: how many candles to scan for structural breaks.
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

    # ── Capped ATR ────────────────────────────────────────────────────────
    capped_atr = compute_capped_atr(df_h1)
    if capped_atr is None:
        capped_atr = atr_value

    # ══════════════════════════════════════════════════════════════════════
    # STRUCTURE on M15 (not a hard gate — feeds OB detection)
    # ══════════════════════════════════════════════════════════════════════
    m15_struct = detect_bos_choch(df_m15, pair_conf, capped_atr, lookback,
                                  scan_window=bos_scan_m15)

    # Structure must be confirmed + bias-matching + displaced for OB anchoring.
    # If any condition fails, OB detection gets an empty struct → returns None →
    # levels = None → fails at RR/OB gate naturally.
    struct_usable = (
        m15_struct["confirmed"]
        and m15_struct["has_displacement"]
        and ((bias == "LONG" and m15_struct["direction"] == "bullish")
             or (bias == "SHORT" and m15_struct["direction"] == "bearish"))
    )

    if struct_usable:
        gate_structure = (
            f"M15 {m15_struct['type'].upper()} {m15_struct['direction']} "
            f"({m15_struct['break_level']}) at "
            f"{m15_struct['break_candle_ts']}, "
            f"broke {m15_struct['swing_broken']:.{dp}f}")
    else:
        gate_structure = (
            f"No usable M15 structure: "
            f"{m15_struct.get('reason', 'direction/displacement mismatch')}")
    # ── All M15 breaks (for scoring) ─────────────────────────────────────
    bos_all = detect_bos_choch_all(df_m15, pair_conf, capped_atr, lookback,
                                    scan_window=bos_scan_m15)

    # ── M15 FVGs ──────────────────────────────────────────────────────────
    m15_fvgs = detect_fvgs(df_m15, FVG_M15_WINDOW)
    unmitigated_fvgs = get_unmitigated_fvgs(m15_fvgs)

    # ══════════════════════════════════════════════════════════════════════
    # OB from structure
    # ══════════════════════════════════════════════════════════════════════
    raw_ob = detect_obs_from_structure(
        df_m15, m15_struct if struct_usable else {"confirmed": False},
        bias, pair_conf)
    ob = select_trade_ob(raw_ob, zone_level, pair_conf)

    # ══════════════════════════════════════════════════════════════════════
    # LEVELS (no OB → None → gate fail)
    # ══════════════════════════════════════════════════════════════════════
    levels = compute_levels(ob, m15_fvgs, [], zones, bias,
                            current_price, pair_conf, zone_level)

    if levels is None:
        if ob is None:
            return _gate_fail(
                "ob",
                f"No valid {bias} OB near zone {zone_level:.{dp}f}",
                bias)
        return _gate_fail("rr", "Could not compute levels (zero risk)", bias)

    gate_ob = (
        f"{ob['type']} OB at "
        f"{ob['ob_top']:.{dp}f}/{ob['ob_bottom']:.{dp}f}, "
        f"candle {ob['ts']}")

    # ══════════════════════════════════════════════════════════════════════
    # HARD GATE: RR >= 1.5
    # ══════════════════════════════════════════════════════════════════════
    if levels["rr_tp1"] < 1.5:
        return _gate_fail(
            "rr",
            f"RR {levels['rr_tp1']:.2f}:1 < 1.5 | "
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

    # ── FVG overlap on OB body ────────────────────────────────────────────
    ob_has_fvg_overlap = False
    matched_fvg = {"top": 0.0, "bot": 0.0, "type": ""}
    if ob is not None:
        for fvg in unmitigated_fvgs:
            if (fvg["fvg_bottom"] <= ob["ob_body_top"]
                    and fvg["fvg_top"] >= ob["ob_body_bottom"]):
                ob_has_fvg_overlap = True
                matched_fvg = {"top": fvg["fvg_top"],
                               "bot": fvg["fvg_bottom"],
                               "type": fvg["type"]}
                break

    # ── FVG in zone (independent — simple existence) ──────────────────────
    fvg_in_zone_score = 0.0
    prox_pct = pair_conf.get("proximity_pct", 0.3)
    for fvg in m15_fvgs:
        fvg_mid = (fvg["fvg_top"] + fvg["fvg_bottom"]) / 2
        dist_pct = abs(fvg_mid - zone_level) / zone_level * 100
        if dist_pct <= prox_pct:
            fvg_in_zone_score = 1.0
            break

    # ── H1 structure ──────────────────────────────────────────────────────
    h1_struct = detect_bos_choch(df_h1, pair_conf, capped_atr, lookback,
                                 scan_window=bos_scan_h1)

    m15_swings = get_swing_points(df_m15, lookback)
    h1_swings = get_swing_points(df_h1, lookback)

    sweep = detect_liquidity_sweep(df_m15, m15_swings, pair_conf, capped_atr)
    pd_result = compute_premium_discount(h1_swings, current_price, bias)

    # ── Score ─────────────────────────────────────────────────────────────
    score_result = compute_score(
        {
            "bos_choch_all": bos_all,
            "ob": ob,
            "ob_has_fvg_overlap": ob_has_fvg_overlap,
            "fvg_in_zone_score": fvg_in_zone_score,
            "sweep": sweep,
            "premium_discount": pd_result,
            "h1_structure": h1_struct,
            "bias": bias
        },
        fatigue_count, fatigue_threshold)

    # ── Entry readiness ───────────────────────────────────────────────────
    readiness = check_entry_readiness(
        df_m15, ob, zone_level, bias, pair_conf, capped_atr, m15_struct)

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
    if ob_has_fvg_overlap:
        confluences.append(
            f"FVG overlaps OB body: {matched_fvg['type']} FVG "
            f"{matched_fvg['bot']:.{dp}f}-{matched_fvg['top']:.{dp}f}")
    if fvg_in_zone_score > 0:
        confluences.append("FVG present in zone proximity")
    if pd_result["valid"]:
        confluences.append(
            f"Price in {'discount' if bias == 'LONG' else 'premium'} "
            f"({pd_result['position_pct']}% of range)")

    struct_score = score_result["breakdown"]["structure_m15"]
    if struct_score >= 2.5:
        confluences.append(
            f"Both BOS + CHoCH on M15 "
            f"(M15 {m15_struct['type']} {m15_struct['break_level']})")
    elif struct_score >= 1.5:
        confluences.append(
            f"M15 CHoCH confirmed ({m15_struct['direction']})")
    elif struct_score >= 1.0:
        confluences.append(
            f"M15 BOS confirmed ({m15_struct['direction']})")

    h1_score = score_result["breakdown"]["h1_alignment"]
    if h1_score > 0:
        confluences.append(
            f"H1 aligned: {h1_struct['type']} {h1_struct['direction']}")

    # ── Missing ───────────────────────────────────────────────────────────
    missing = []
    if not sweep["detected"]:
        missing.append({
            "item": "liquidity_sweep",
            "reason": "No wick-rejection sweep of prior swing point"})
    elif sweep["sweep_score"] < 2.5:
        missing.append({
            "item": "liquidity_sweep_partial",
            "reason": "Near-miss sweep — scored 1.0/1.5"})
    if not ob_has_fvg_overlap:
        missing.append({
            "item": "ob_fvg_overlap",
            "reason": "No unmitigated FVG overlapping the OB body"})
    if fvg_in_zone_score == 0:
        missing.append({
            "item": "fvg_in_zone",
            "reason": "No FVG found within zone proximity"})
    if not pd_result["valid"]:
        side = "premium" if bias == "LONG" else "discount"
        missing.append({
            "item": "premium_discount",
            "reason": (f"Price at {pd_result['position_pct']}% "
                       f"of range ({side})")})
    if struct_score < 2.5:
        if struct_score >= 1.5:
            missing.append({
                "item": "structure_m15",
                "reason": "Only CHoCH confirmed — no BOS yet"})
        elif struct_score >= 1.0:
            missing.append({
                "item": "structure_m15",
                "reason": "Only BOS confirmed — no CHoCH yet"})
        else:
            missing.append({
                "item": "structure_m15",
                "reason": f"No bias-matching BOS or CHoCH in last {bos_scan_m15} candles"})
    if h1_score == 0:
        missing.append({
            "item": "h1_alignment",
            "reason": "H1 structure not confirming bias direction"})
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
        "fvg_confirmed": ob_has_fvg_overlap,

        "lq_sweep_price": sweep.get("sweep_price") or 0,

        "premium_discount_pct": pd_result["position_pct"],
        "premium_discount_valid": pd_result["valid"],

        "m15_structure_detail": m15_struct,
        "h1_structure_detail": h1_struct,
        "m15_break_level": m15_struct.get("break_level", "unknown"),
        "sweep_detail": sweep,
        "readiness_detail": readiness,

        "gemini_needed": True,

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
