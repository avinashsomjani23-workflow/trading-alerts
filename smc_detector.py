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


def detect_sweep_decay(df, swings, current_idx):
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
                pts = 2.5 if hours_old <= 24 else 1.5
                if pts > score:
                    score, sweep_price = pts, float(L[i])
            elif s['type'] == 'high' and H[i] > s['price'] and C[i] < s['price']:
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
        # London 12:30-16:30 IST + NY 18:30-00:30 IST
        return (12 <= ist_hour <= 16) or (ist_hour >= 18) or (ist_hour == 0)
    if pair_type == "index":
        # US cash 19:00-01:30 IST
        return (ist_hour >= 19) or (ist_hour <= 1)
    if pair_type == "commodity":
        # Gold follows London/NY
        return (12 <= ist_hour <= 16) or (ist_hour >= 18) or (ist_hour == 0)
    return False


def run_scorecard(bias, df_h1, ob, fvg, current_price, pair_conf=None, df_m15=None, macro_score=1.0):
    bos_tag = ob.get('bos_tag', 'BOS')
    bd = {"structure": 2.5 if bos_tag == 'CHoCH' else 1.5}

    # Sweep — take the best of H1 / M15
    swings_h1 = get_swing_points(df_h1, lookback=5)
    h1_sweep_score, h1_sweep_price = detect_sweep_decay(df_h1, swings_h1, len(df_h1) - 1)

    m15_sweep_score, m15_sweep_price = 0.0, None
    if df_m15 is not None and len(df_m15) > 20:
        swings_m15 = get_swing_points(df_m15, lookback=4)
        m15_sweep_score, m15_sweep_price = detect_sweep_decay(df_m15, swings_m15, len(df_m15) - 1)

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

    # Premium / Discount
    eq = (df_h1['High'].max() + df_h1['Low'].min()) / 2.0
    pd_hit = (bias == "LONG" and current_price <= eq) or (bias == "SHORT" and current_price >= eq)
    bd["pd"] = 1.0 if pd_hit else 0.0

    # Killzone — widened, pair-aware
    ist_hour = (datetime.utcnow() + timedelta(hours=5, minutes=30)).hour
    pair_type = pair_conf.get("pair_type", "forex") if pair_conf else "forex"
    bd["killzone"] = 1.0 if _killzone_hit(ist_hour, pair_type) else 0.0

    bd["macro"] = macro_score

    return {
        "total": round(sum(bd.values()), 1),
        "breakdown": bd,
        "sweep_price": sweep_price,
        "sweep_tf": sweep_tf
    }


def generate_scorecard_rows(bias, breakdown, ob, sweep_price, sweep_tf, pair_conf):
    """Return list of (label, score, max_score, status, explanation) for email rendering."""
    dp = _dp(pair_conf)
    rows = []

    # 1. Structure
    s = breakdown.get("structure", 0)
    if s >= 2.5:
        rows.append(("Structure", s, 2.5, "ok", "Trend has shifted in our favor (CHoCH confirmed)."))
    elif s >= 1.5:
        rows.append(("Structure", s, 2.5, "warn", "Trend continuation confirmed (BOS), but no fresh shift."))
    else:
        rows.append(("Structure", s, 2.5, "fail", "No confirmed BOS or CHoCH."))

    # 2. Liquidity Sweep
    s = breakdown.get("sweep", 0)
    if s >= 2.5 and sweep_price is not None:
        rows.append((
            "Liquidity Sweep", s, 2.5, "ok",
            f"Price pierced a recent level and reversed — smart money grabbed stop-losses before reversing "
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

    # 3. FVG
    s = breakdown.get("fvg", 0)
    if s >= 1.5:
        rows.append(("FVG", s, 1.5, "ok", "Fair Value Gap overlaps the Order Block — strong displacement confluence."))
    elif s >= 1.0:
        rows.append(("FVG", s, 1.5, "warn", "FVG exists inside the zone but does not overlap the OB."))
    else:
        rows.append(("FVG", s, 1.5, "fail", "No unmitigated FVG inside the zone."))

    # 4. Freshness
    s = breakdown.get("freshness", 0)
    touches = ob.get("touches", 0)
    if s >= 0.5:
        rows.append(("Freshness", s, 0.5, "ok", "Zone is pristine — not touched yet."))
    else:
        rows.append((
            "Freshness", s, 0.5, "warn",
            f"Zone has been tested {touches}x already — reaction may be weaker."
        ))

    # 5. Premium / Discount
    s = breakdown.get("pd", 0)
    if s >= 1.0:
        if bias == "LONG":
            rows.append(("Premium / Discount", s, 1.0, "ok", "Price is in discount — good area to buy."))
        else:
            rows.append(("Premium / Discount", s, 1.0, "ok", "Price is in premium — good area to sell."))
    else:
        rows.append(("Premium / Discount", s, 1.0, "fail", "Price is on the wrong side of equilibrium."))

    # 6. Killzone
    s = breakdown.get("killzone", 0)
    if s >= 1.0:
        rows.append(("Killzone", s, 1.0, "ok", "Inside active trading window."))
    else:
        rows.append(("Killzone", s, 1.0, "fail", "Outside main trading window — lower volume, weaker follow-through expected."))

    # 7. Macro / News
    s = breakdown.get("macro", 0)
    if s >= 1.0:
        rows.append(("Macro / News", s, 1.0, "ok", "No Tier-1 news expected in the 2h window."))
    else:
        rows.append(("Macro / News", s, 1.0, "fail", "High-impact news imminent — risk of whipsaw."))

    return rows


def detect_ltf_choch(df_m5, bias, bounds):
    """Detect M5 CHoCH strictly inside HTF zone bounds. Returns dict with fired flag and level."""
    swings = get_swing_points(df_m5, lookback=3, bounds=bounds)
    if len(swings) < 2:
        return {"fired": False, "level": None}

    C = df_m5['Close'].values
    latest_high = [s for s in swings if s['type'] == 'high']
    latest_low = [s for s in swings if s['type'] == 'low']

    if bias == 'LONG' and latest_high:
        if C[-1] > latest_high[-1]['price'] and C[-2] <= latest_high[-1]['price']:
            return {"fired": True, "level": float(latest_high[-1]['price'])}
    elif bias == 'SHORT' and latest_low:
        if C[-1] < latest_low[-1]['price'] and C[-2] >= latest_low[-1]['price']:
            return {"fired": True, "level": float(latest_low[-1]['price'])}
    return {"fired": False, "level": None}


def check_opposite_bos(df_h1, bias, since_ts=None):
    """True if H1 printed a BOS in the opposite direction since `since_ts` (naive IST datetime)."""
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
