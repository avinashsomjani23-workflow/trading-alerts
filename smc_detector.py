import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _dp(pair_conf):
    return pair_conf.get("decimal_places", 5)


def compute_atr(df, period=14):
    """ATR computation used across phases."""
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
    return float(np.mean(trs[-period:]))


# ---------------------------------------------------------------------------
# Dealing range lookback per pair type (in H1 candles).
# Veteran-calibrated: forex ~3 trading days, indices ~2, commodities ~4.
# ---------------------------------------------------------------------------
DEALING_RANGE_LOOKBACK_H1 = {
    "forex": 72,
    "index": 48,
    "commodity": 96
}


def get_dealing_range(ob, df_h1, h1_atr, pair_conf=None, current_price=None):
    """
    Compute the dealing range using the most recent HTF swing high and swing low
    within a pair-aware lookback window on H1.

    This is what a veteran does on a chart: scroll back a few days, identify the
    highest swing high and lowest swing low that enclose current price, and use
    that as the dealing range. Range is recomputed each scan (not frozen at OB
    creation) so it stays fresh as structure evolves.

    If current price sits outside the detected range, the range is extended to
    include current_price ± 0.5x ATR so pd_position stays within [0, 1].

    Signature keeps backward-compatible defaults: if pair_conf or current_price
    are not passed, falls back to window-min/max of last 72 candles.

    Returns dict: valid, range_high, range_low, equilibrium, source.
    """
    if df_h1 is None or len(df_h1) < 20:
        return {"valid": False, "source": "insufficient_data"}
    if h1_atr is None or h1_atr == 0:
        return {"valid": False, "source": "no_atr"}

    # Pair-aware lookback window
    pair_type = pair_conf.get("pair_type", "forex") if pair_conf else "forex"
    lookback = DEALING_RANGE_LOOKBACK_H1.get(pair_type, 72)
    lookback = min(lookback, len(df_h1))

    df_window = df_h1.tail(lookback)

    # Find fractal swings within the window. Use a copy with reset index so
    # get_swing_points returns indices local to the window.
    df_window_reset = df_window.reset_index(drop=True)
    swings = get_swing_points(df_window_reset, lookback=5)

    if swings:
        highs = [s['price'] for s in swings if s['type'] == 'high']
        lows = [s['price'] for s in swings if s['type'] == 'low']
        if highs and lows:
            range_high = max(highs)
            range_low = min(lows)
            source = "swings_window"
        else:
            # Only one side has swings — fall back to window extremes
            range_high = float(df_window['High'].max())
            range_low = float(df_window['Low'].min())
            source = "window_extremes_partial"
    else:
        # No swings detected at all — use window extremes
        range_high = float(df_window['High'].max())
        range_low = float(df_window['Low'].min())
        source = "window_extremes"

    # Safety: ensure range is positive-width
    if range_high <= range_low:
        return {"valid": False, "source": "degenerate_range"}

    # If current price sits outside the range, extend to include it with a
    # half-ATR buffer. This keeps pd_position math sane and represents "range
    # just broke — redraw from the new extreme" in veteran terms.
    if current_price is not None:
        buf = 0.5 * h1_atr
        if current_price > range_high:
            range_high = current_price + buf
            source = source + "+extended_high"
        elif current_price < range_low:
            range_low = current_price - buf
            source = source + "+extended_low"

    eq = (range_high + range_low) / 2.0
    return {
        "valid": True,
        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": eq,
        "source": source
    }


def get_swing_points(df, lookback=4, bounds=None):
    if df is None or len(df) < lookback * 2 + 1:
        return []
    H, L = df['High'].values.astype(float), df['Low'].values.astype(float)
    swings = []
    for i in range(lookback, len(H) - lookback):
        if bounds and (H[i] > bounds['max'] or L[i] < bounds['min']):
            continue
        if H[i] == max(H[i - lookback: i + lookback + 1]):
            swings.append({"type": "high", "price": float(H[i]), "idx": i, "ts": df.index[i]})
        if L[i] == min(L[i - lookback: i + lookback + 1]):
            swings.append({"type": "low", "price": float(L[i]), "idx": i, "ts": df.index[i]})
    return sorted(swings, key=lambda s: s["idx"])


def compute_bos_sequence_count(df_h1, lookback=4):
    """
    Count how many consecutive BOS events have printed since the last CHoCH on H1.
    Returns the count for the most recent directional trend.

    Used by Phase 2 to ensure BOS sequence scoring is always derived from the
    latest H1 data, not a potentially stale count from active_obs.json.

    Returns dict: {'count': int, 'trend': 'bullish'|'bearish'|None}
    """
    if df_h1 is None or len(df_h1) < lookback * 2 + 2:
        return {'count': 1, 'trend': None}

    C = df_h1['Close'].values.astype(float)
    n = len(df_h1)
    swings = get_swing_points(df_h1, lookback=lookback)

    trend_state = None
    bos_seq_counter = 0

    for i in range(lookback + 1, n):
        past_swings = [s for s in swings if s['idx'] < i]
        if len(past_swings) < 2:
            continue
        latest_high = [s for s in past_swings if s['type'] == 'high']
        latest_low = [s for s in past_swings if s['type'] == 'low']
        if not latest_high or not latest_low:
            continue

        sh, sl = latest_high[-1], latest_low[-1]
        bos_detected, bos_type = False, None

        if C[i] > sh['price'] and C[i - 1] <= sh['price']:
            bos_detected, bos_type = True, 'bullish'
        elif C[i] < sl['price'] and C[i - 1] >= sl['price']:
            bos_detected, bos_type = True, 'bearish'

        if not bos_detected:
            continue

        if trend_state is None or trend_state != bos_type:
            # CHoCH — reset counter
            bos_seq_counter = 0
        else:
            # BOS continuation
            bos_seq_counter += 1
        trend_state = bos_type

    # Counter stored zero-indexed (0 = first BOS after CHoCH). Return 1-indexed.
    return {'count': max(1, bos_seq_counter + 1), 'trend': trend_state}


def stack_labels(labels, pair_conf):
    """
    Prevent label overlap on charts by offsetting prices that are too close.

    labels: list of (price, text, color) tuples.
    Returns: list of (adjusted_price, text, color) tuples, sorted by price.
    """
    if not labels:
        return labels

    dp = _dp(pair_conf)
    pair_type = pair_conf.get("pair_type", "forex")

    thresholds = {
        "forex": 0.00030 if dp == 5 else 0.030,
        "index": 15.0,
        "commodity": 2.0
    }
    min_gap = thresholds.get(pair_type, 0.00030)

    sorted_labels = sorted(labels, key=lambda x: x[0])
    adjusted = [sorted_labels[0]]

    for i in range(1, len(sorted_labels)):
        price, text, color = sorted_labels[i]
        prev_price = adjusted[-1][0]
        if abs(price - prev_price) < min_gap:
            price = prev_price + min_gap
        adjusted.append((price, text, color))

    return adjusted


def detect_sweep_decay(df, swings, current_idx, bias=None):
    score, sweep_price = 0.0, None
    H, L, C = df['High'].values, df['Low'].values, df['Close'].values
    current_ts = df.index[current_idx]
    for i in range(max(0, current_idx - 8), current_idx + 1):
        for s in swings:
            if s['idx'] >= i:
                continue
            hours_old = (current_ts - s['ts']).total_seconds() / 3600
            if hours_old > 72:
                continue
            if s['type'] == 'low' and L[i] < s['price'] and C[i] > s['price']:
                if bias is not None and bias != 'LONG':
                    continue
                pts = 2.5 if hours_old <= 24 else 1.5
                if pts > score:
                    score, sweep_price = pts, float(L[i])
            elif s['type'] == 'high' and H[i] > s['price'] and C[i] < s['price']:
                if bias is not None and bias != 'SHORT':
                    continue
                pts = 2.5 if hours_old <= 24 else 1.5
                if pts > score:
                    score, sweep_price = pts, float(H[i])
    return score, sweep_price


def detect_fvg_in_zone(df, bias, zone_top, zone_bottom):
    if df is None or len(df) < 5:
        return {"exists": False, "fvg_top": None, "fvg_bottom": None}
    H, L = df['High'].values.astype(float), df['Low'].values.astype(float)
    n = len(df)
    for k in range(n - 3, max(0, n - 30), -1):
        if bias == "LONG" and H[k] < L[k + 2]:
            ft, fb = float(L[k + 2]), float(H[k])
            if fb > zone_top or ft < zone_bottom:
                continue
            filled = any(L[m] <= fb for m in range(k + 3, n))
            if not filled:
                return {"exists": True, "fvg_top": ft, "fvg_bottom": fb}
        elif bias == "SHORT" and L[k] > H[k + 2]:
            ft, fb = float(L[k]), float(H[k + 2])
            if fb > zone_top or ft < zone_bottom:
                continue
            filled = any(H[m] >= ft for m in range(k + 3, n))
            if not filled:
                return {"exists": True, "fvg_top": ft, "fvg_bottom": fb}
    return {"exists": False, "fvg_top": None, "fvg_bottom": None}


def compute_dynamic_levels(pair_conf, bias, ob, fvg, current_price, df_trigger):
    dp = _dp(pair_conf)
    spread_val = pair_conf.get("spread_pips", 2) * (0.0001 if dp == 5 else 0.01)

    ob_top = float(ob.get('ob_high', ob.get('high', ob.get('proximal_line', 0))))
    ob_bottom = float(ob.get('ob_low', ob.get('low', ob.get('distal_line', 0))))

    sl = ob_bottom - spread_val if bias == "LONG" else ob_top + spread_val
    ob_prox = ob_top if bias == "LONG" else ob_bottom
    ob_mean = (ob_top + ob_bottom) / 2.0

    fvg_prox = None
    if fvg and fvg.get('exists'):
        fvg_prox = float(fvg['fvg_top']) if bias == "LONG" else float(fvg['fvg_bottom'])

    swings = get_swing_points(df_trigger, lookback=10)
    tp_targets = [
        s['price'] for s in swings
        if (bias == "LONG" and s['type'] == 'high' and s['price'] > ob_prox)
        or (bias == "SHORT" and s['type'] == 'low' and s['price'] < ob_prox)
    ]
    tp_targets.sort(reverse=(bias == "SHORT"))

    entry_model = pair_conf.get("pair_type", "forex")
    final_entry, final_rr, entry_source, tp1 = None, 0.0, "", None

    def check_rr(entry_test):
        risk = abs(entry_test - sl)
        if risk == 0:
            return 0.0, None
        for target in tp_targets:
            rr = abs(target - entry_test) / risk
            if rr >= 1.5:
                return rr, target
        fallback_tp = entry_test + (risk * 2.0) if bias == "LONG" else entry_test - (risk * 2.0)
        return 2.0, fallback_tp

    if entry_model == "forex":
        attempts = []
        if fvg_prox:
            attempts.append((fvg_prox, "FVG Proximal"))
        attempts.append((ob_prox, "OB Proximal"))
        attempts.append((ob_mean, "OB 50% Mean"))
        for price, name in attempts:
            rr, tp_val = check_rr(price)
            if rr >= 1.5:
                final_entry, entry_source, final_rr, tp1 = price, name, rr, tp_val
                break
    else:
        final_entry, entry_source = ob_prox, "OB Proximal Limit"
        final_rr, tp1 = check_rr(final_entry)

    if final_entry is None or final_rr < 1.5:
        return {"valid": False, "reason": "R:R < 1.5 on all cascade attempts"}

    # Entry-side validation (forex limit orders only).
    # A BUY LIMIT must sit at or below current price; a SELL LIMIT at or above.
    # If the cascade picked an entry on the wrong side of price, price has already
    # moved through the zone — skip the alert rather than chase.
    # Small tolerance = 0.5x spread to avoid false fails on rounding / tick noise.
    if entry_model == "forex":
        tolerance = 0.5 * spread_val
        if bias == "LONG" and final_entry > current_price + tolerance:
            return {
                "valid": False,
                "reason": (f"Entry {round(final_entry, dp)} is above current price "
                           f"{round(current_price, dp)} — LONG limit would chase price.")
            }
        if bias == "SHORT" and final_entry < current_price - tolerance:
            return {
                "valid": False,
                "reason": (f"Entry {round(final_entry, dp)} is below current price "
                           f"{round(current_price, dp)} — SHORT limit would chase price.")
            }

    risk = abs(final_entry - sl)
    tp2 = final_entry + (risk * 4.0) if bias == "LONG" else final_entry - (risk * 4.0)

    return {
        "valid": True,
        "entry": round(final_entry, dp),
        "sl": round(sl, dp),
        "tp1": round(tp1, dp),
        "tp2": round(tp2, dp),
        "rr": round(final_rr, 2),
        "entry_source": entry_source
    }


def _killzone_hit(ist_hour, pair_type):
    """Widened session windows (hour-level precision)."""
    if pair_type == "forex":
        return (12 <= ist_hour <= 16) or (ist_hour >= 18) or (ist_hour == 0)
    if pair_type == "index":
        return (ist_hour >= 19) or (ist_hour <= 1)
    if pair_type == "commodity":
        return (12 <= ist_hour <= 16) or (ist_hour >= 18) or (ist_hour == 0)
    return False


def run_scorecard(bias, df_h1, ob, fvg, current_price, pair_conf=None, df_m15=None, macro_score=1.0):
    bos_tag = ob.get('bos_tag', 'BOS')

    # Structure — pair-aware BOS sequence penalty
    if bos_tag == 'CHoCH':
        bd = {"structure": 2.5}
    else:
        bos_seq = ob.get('bos_sequence_count', 1)
        pair_type = pair_conf.get('pair_type', 'forex') if pair_conf else 'forex'
        caution_threshold = {'forex': 3, 'index': 5, 'commodity': 4}.get(pair_type, 3)
        bd = {"structure": 1.0 if bos_seq >= caution_threshold else 1.5}

    # Sweep — take the best of H1 / M15
    swings_h1 = get_swing_points(df_h1, lookback=5)
    h1_sweep_score, h1_sweep_price = detect_sweep_decay(df_h1, swings_h1, len(df_h1) - 1, bias)

    m15_sweep_score, m15_sweep_price = 0.0, None
    if df_m15 is not None and len(df_m15) > 20:
        swings_m15 = get_swing_points(df_m15, lookback=4)
        m15_sweep_score, m15_sweep_price = detect_sweep_decay(df_m15, swings_m15, len(df_m15) - 1, bias)

    if m15_sweep_score > h1_sweep_score:
        bd["sweep"], sweep_price, sweep_tf = m15_sweep_score, m15_sweep_price, "M15"
    else:
        bd["sweep"], sweep_price, sweep_tf = h1_sweep_score, h1_sweep_price, "H1"

    # FVG
    if fvg and fvg.get('exists'):
        bd["fvg"] = 1.5 if fvg.get('touches_ob', False) else 1.0
    else:
        bd["fvg"] = 0.0

    # Freshness — has zone been retouched on H1 in last 5 candles?
    is_fresh = True
    ob_top = float(ob.get('proximal_line', 0))
    ob_bottom = float(ob.get('distal_line', 0))
    for i in range(max(0, len(df_h1) - 5), len(df_h1) - 1):
        if bias == "LONG" and df_h1['Low'].iloc[i] <= ob_top:
            is_fresh = False
            break
        if bias == "SHORT" and df_h1['High'].iloc[i] >= ob_bottom:
            is_fresh = False
            break
    bd["freshness"] = 0.5 if is_fresh else 0.0

    # Premium / Discount — graded scoring on impulse-leg dealing range.
    # LONG wants price in bottom of range (discount):
    #   position <= 0.30  → 1.0 (deep discount)
    #   0.30 < position <= 0.45 → 0.5 (discount, not deep)
    #   position > 0.45 → 0.0 (not in discount)
    # SHORT wants price in top of range (premium):
    #   position >= 0.70  → 1.0 (deep premium)
    #   0.55 <= position < 0.70 → 0.5 (premium, not deep)
    #   position < 0.55 → 0.0 (not in premium)
    # `pd_position` stored in breakdown so scorecard row can show exact percentage.
    h1_atr_val = compute_atr(df_h1)
    dr = get_dealing_range(ob, df_h1, h1_atr_val,
                           pair_conf=pair_conf, current_price=current_price)
    pd_position = None
    if dr["valid"]:
        rng_width = dr["range_high"] - dr["range_low"]
        if rng_width > 0:
            pd_position = (current_price - dr["range_low"]) / rng_width
            if bias == "LONG":
                if pd_position <= 0.30:
                    bd["pd"] = 1.0
                elif pd_position <= 0.45:
                    bd["pd"] = 0.5
                else:
                    bd["pd"] = 0.0
            else:  # SHORT
                if pd_position >= 0.70:
                    bd["pd"] = 1.0
                elif pd_position >= 0.55:
                    bd["pd"] = 0.5
                else:
                    bd["pd"] = 0.0
        else:
            bd["pd"] = 0.0
    else:
        bd["pd"] = 0.0
        dr = {"valid": False, "range_high": 0, "range_low": 0, "equilibrium": 0,
              "source": dr.get("source", "unknown")}

    # Killzone — IST-based (hardcoded IST clock hours; no DST drift needed since
    # windows are defined in local IST and IST has no DST).
    ist_hour = (datetime.utcnow() + timedelta(hours=5, minutes=30)).hour
    pair_type = pair_conf.get("pair_type", "forex") if pair_conf else "forex"
    bd["killzone"] = 1.0 if _killzone_hit(ist_hour, pair_type) else 0.0

    bd["macro"] = macro_score

    return {
        "total": round(sum(bd.values()), 1),
        "breakdown": bd,
        "sweep_price": sweep_price,
        "sweep_tf": sweep_tf,
        "dealing_range": dr,
        "pd_position": pd_position
    }


def generate_scorecard_rows(bias, breakdown, ob, sweep_price, sweep_tf, pair_conf,
                            dealing_range=None, fvg_source=None, pd_position=None):
    """Return list of (label, score, max_score, status, explanation) for email rendering."""
    dp = _dp(pair_conf)
    rows = []

    # 1. Structure — pair-aware BOS sequence
    s = breakdown.get("structure", 0)
    bos_seq = ob.get('bos_sequence_count', 1)
    if s >= 2.5:
        rows.append(("Structure", s, 2.5, "ok", "Trend has shifted in our favor (CHoCH confirmed)."))
    elif s >= 1.5:
        rows.append(("Structure", s, 2.5, "warn",
                      f"Trend continuation (BOS #{bos_seq} since CHoCH)."))
    elif s >= 1.0:
        rows.append(("Structure", s, 2.5, "fail",
                      f"Late trend continuation (BOS #{bos_seq} since CHoCH) — trend may be exhausted."))
    else:
        rows.append(("Structure", s, 2.5, "fail", "No confirmed BOS or CHoCH."))

    # 2. Liquidity Sweep
    s = breakdown.get("sweep", 0)
    if s >= 2.5 and sweep_price is not None:
        rows.append((
            "Liquidity Sweep", s, 2.5, "ok",
            f"Price pierced a recent level and reversed — smart money grabbed stop-losses "
            f"({sweep_tf} sweep at {sweep_price:.{dp}f})."
        ))
    elif s >= 1.5 and sweep_price is not None:
        rows.append((
            "Liquidity Sweep", s, 2.5, "warn",
            f"Sweep happened but is older than 24h — signal is weaker "
            f"({sweep_tf} sweep at {sweep_price:.{dp}f})."
        ))
    else:
        rows.append(("Liquidity Sweep", s, 2.5, "fail", "No recent stop-hunt sweep detected near the zone."))

    # 3. FVG — with source label (M15 / H1)
    s = breakdown.get("fvg", 0)
    src_label = f"{fvg_source} " if fvg_source else ""
    if s >= 1.5:
        rows.append(("FVG", s, 1.5, "ok",
                      f"{src_label}FVG overlaps the Order Block — strong displacement confluence."))
    elif s >= 1.0:
        rows.append(("FVG", s, 1.5, "warn",
                      f"{src_label}FVG exists inside the zone but does not overlap the OB."))
    else:
        rows.append(("FVG", s, 1.5, "fail", "No unmitigated FVG inside the zone."))

    # 4. Freshness — based on H1 candle proximity, not touch count
    s = breakdown.get("freshness", 0)
    if s >= 0.5:
        rows.append(("Freshness", s, 0.5, "ok",
                      "Zone is untouched in last 5 H1 candles — first-touch entry."))
    else:
        rows.append(("Freshness", s, 0.5, "warn",
                      "Price has been near this zone in the last 5 H1 candles — not a first-touch setup."))

    # 5. Premium / Discount — graded, with exact percentage and dealing range details
    s = breakdown.get("pd", 0)
    dr_src = ""
    if dealing_range and dealing_range.get("valid"):
        dr_src = (f" (range: {dealing_range['range_low']:.{dp}f}"
                  f"–{dealing_range['range_high']:.{dp}f},"
                  f" EQ: {dealing_range['equilibrium']:.{dp}f})")
        src_val = dealing_range.get("source", "")
        if "extended" in src_val:
            dr_src += " [range extended to include current price]"

    pd_pct_str = f"{pd_position * 100:.0f}%" if pd_position is not None else "n/a"

    if not dealing_range or not dealing_range.get("valid"):
        rows.append(("Premium / Discount", s, 1.0, "warn",
                      "Dealing range too narrow to score. Neutral."))
    elif bias == "LONG":
        if s >= 1.0:
            rows.append(("Premium / Discount", s, 1.0, "ok",
                          f"Price at {pd_pct_str} of dealing range (deep discount).{dr_src}"))
        elif s >= 0.5:
            rows.append(("Premium / Discount", s, 1.0, "warn",
                          f"Price at {pd_pct_str} of dealing range (discount, not deep).{dr_src}"))
        else:
            rows.append(("Premium / Discount", s, 1.0, "fail",
                          f"Price at {pd_pct_str} of dealing range (above equilibrium — not optimal for LONG).{dr_src}"))
    else:  # SHORT
        if s >= 1.0:
            rows.append(("Premium / Discount", s, 1.0, "ok",
                          f"Price at {pd_pct_str} of dealing range (deep premium).{dr_src}"))
        elif s >= 0.5:
            rows.append(("Premium / Discount", s, 1.0, "warn",
                          f"Price at {pd_pct_str} of dealing range (premium, not deep).{dr_src}"))
        else:
            rows.append(("Premium / Discount", s, 1.0, "fail",
                          f"Price at {pd_pct_str} of dealing range (below equilibrium — not optimal for SHORT).{dr_src}"))

    # 6. Killzone
    s = breakdown.get("killzone", 0)
    if s >= 1.0:
        rows.append(("Killzone", s, 1.0, "ok", "Inside active trading window."))
    else:
        rows.append(("Killzone", s, 1.0, "fail",
                      "Outside main trading window — lower volume expected."))

    # 7. Macro / News
    s = breakdown.get("macro", 0)
    if s >= 1.0:
        rows.append(("Macro / News", s, 1.0, "ok", "No Tier-1 news expected in the 2h window."))
    else:
        rows.append(("Macro / News", s, 1.0, "fail", "High-impact news imminent — risk of whipsaw."))

    return rows


def detect_ltf_choch(df_m5, bias, bounds):
    """
    Detect M5 CHoCH where the BREAK LEVEL is near the HTF zone.

    Old behavior (dropped): required the M5 swing to sit *inside* the zone.
    This failed on tight zones (NAS100/GOLD) where M5 swings formed just outside.

    New behavior:
    - Scan M5 swings across the full window (no bounds filter).
    - For LONG: the most recent swing high that gets broken must sit inside the zone
      OR within 1x M5 ATR above the zone's top.
    - For SHORT: the most recent swing low that gets broken must sit inside the zone
      OR within 1x M5 ATR below the zone's bottom.
    - Break = current close crosses the swing level and previous close was on the other side.
    """
    if df_m5 is None or len(df_m5) < 10:
        return {"fired": False, "level": None}

    # Swings across full window — no bounds filter
    swings = get_swing_points(df_m5, lookback=3)
    if len(swings) < 2:
        return {"fired": False, "level": None}

    C = df_m5['Close'].values
    m5_atr = compute_atr(df_m5) or 0.0

    # Grace band: 1x M5 ATR above (LONG) or below (SHORT) the zone
    zone_max = bounds['max']
    zone_min = bounds['min']

    if bias == 'LONG':
        long_grace_top = zone_max + m5_atr
        # Latest swing high whose PRICE sits inside zone OR within grace above it
        eligible_highs = [
            s for s in swings
            if s['type'] == 'high' and zone_min <= s['price'] <= long_grace_top
        ]
        if not eligible_highs:
            return {"fired": False, "level": None}
        latest = eligible_highs[-1]
        if C[-1] > latest['price'] and C[-2] <= latest['price']:
            return {"fired": True, "level": float(latest['price'])}
    elif bias == 'SHORT':
        short_grace_bottom = zone_min - m5_atr
        eligible_lows = [
            s for s in swings
            if s['type'] == 'low' and short_grace_bottom <= s['price'] <= zone_max
        ]
        if not eligible_lows:
            return {"fired": False, "level": None}
        latest = eligible_lows[-1]
        if C[-1] < latest['price'] and C[-2] >= latest['price']:
            return {"fired": True, "level": float(latest['price'])}

    return {"fired": False, "level": None}


def check_opposite_bos(df_h1, bias, since_ts=None):
    """True if H1 printed a BOS in the opposite direction since `since_ts`."""
    if df_h1 is None or len(df_h1) < 10:
        return False

    swings = get_swing_points(df_h1, lookback=5)
    C = df_h1['Close'].values
    n = len(df_h1)
    start_idx = max(1, n - 24)

    if since_ts is not None:
        try:
            since_utc = since_ts - timedelta(hours=5, minutes=30)
            for i in range(n):
                ts = df_h1.index[i]
                if hasattr(ts, 'tz') and ts.tz is not None:
                    ts_cmp = ts.tz_convert('UTC').tz_localize(None).to_pydatetime()
                elif hasattr(ts, 'to_pydatetime'):
                    ts_cmp = ts.to_pydatetime()
                else:
                    ts_cmp = ts
                if ts_cmp >= since_utc:
                    start_idx = max(1, i)
                    break
        except Exception:
            start_idx = max(1, n - 24)

    for i in range(start_idx, n):
        past_swings = [s for s in swings if s['idx'] < i]
        latest_high = [s for s in past_swings if s['type'] == 'high']
        latest_low = [s for s in past_swings if s['type'] == 'low']

        if bias == "LONG" and latest_low:
            if C[i] < latest_low[-1]['price'] and C[i - 1] >= latest_low[-1]['price']:
                return True
        elif bias == "SHORT" and latest_high:
            if C[i] > latest_high[-1]['price'] and C[i - 1] <= latest_high[-1]['price']:
                return True
    return False
