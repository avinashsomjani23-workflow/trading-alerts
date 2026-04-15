import numpy as np
import pandas as pd

def _dp(pair_conf): return pair_conf.get("decimal_places", 5)

def get_swing_points(df, lookback=4, bounds=None):
    if df is None or len(df) < lookback * 2 + 1: return []
    H, L = df['High'].values.astype(float), df['Low'].values.astype(float)
    swings = []
    for i in range(lookback, len(H) - lookback):
        # Strict boundary check if provided (M15 inside H1)
        if bounds and (H[i] > bounds['max'] or L[i] < bounds['min']): continue
        
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
            if s['idx'] >= i: continue
            hours_old = (current_ts - s['ts']).total_seconds() / 3600
            if hours_old > 72: continue
            if s['type'] == 'low' and L[i] < s['price'] and C[i] > s['price']:
                pts = 2.5 if hours_old <= 24 else 1.5
                if pts > score: score, sweep_price = pts, float(L[i])
            elif s['type'] == 'high' and H[i] > s['price'] and C[i] < s['price']:
                pts = 2.5 if hours_old <= 24 else 1.5
                if pts > score: score, sweep_price = pts, float(H[i])
    return score, sweep_price
def detect_fvg_in_zone(df, bias, zone_top, zone_bottom, impulse_start=None, bos_idx=None):
    if df is None or len(df) < 5:
        return {"exists": False, "fvg_top": None, "fvg_bottom": None}
    H, L = df['High'].values.astype(float), df['Low'].values.astype(float)
    n = len(df)
    scan_start = impulse_start if impulse_start is not None else max(0, n - 30)
    scan_end   = min(bos_idx - 1 if bos_idx is not None else n - 3, n - 3)
    for k in range(scan_start, scan_end + 1):
        if k + 2 >= n: break
        if bias == "LONG" and H[k] < L[k + 2]:
            ft, fb = float(L[k + 2]), float(H[k])
            if fb > zone_top or ft < zone_bottom: continue
            if not any(L[m] <= fb for m in range(k + 3, n)):
                return {"exists": True, "fvg_top": ft, "fvg_bottom": fb}
        elif bias == "SHORT" and L[k] > H[k + 2]:
            ft, fb = float(L[k]), float(H[k + 2])
            if fb > zone_top or ft < zone_bottom: continue
            if not any(H[m] >= ft for m in range(k + 3, n)):
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
    tp_targets = [s['price'] for s in swings if (bias == "LONG" and s['type'] == 'high' and s['price'] > ob_prox) or (bias == "SHORT" and s['type'] == 'low' and s['price'] < ob_prox)]
    tp_targets.sort(reverse=(bias == "SHORT"))

    entry_model = pair_conf.get("pair_type", "forex")
    final_entry, final_rr, entry_source, tp1 = None, 0.0, "", None

    def check_rr(entry_test):
        risk = abs(entry_test - sl)
        if risk == 0: return 0.0, None
        for target in tp_targets:
            rr = abs(target - entry_test) / risk
            if rr >= 1.5: return rr, target
        return 2.0, entry_test + (risk * 2.0) if bias == "LONG" else entry_test - (risk * 2.0)

    if entry_model == "forex":
        attempts = []
        if fvg_prox: attempts.append((fvg_prox, "FVG Proximal"))
        attempts.append((ob_prox, "OB Proximal"))
        attempts.append((ob_mean, "OB 50% Mean"))
        for price, name in attempts:
            rr, tp_val = check_rr(price)
            if rr >= 1.5:
                final_entry, entry_source, final_rr, tp1 = price, name, rr, tp_val
                break
    else:
        # Indices and Commodities wait for LTF CHoCH
        final_entry, entry_source = ob_prox, "OB Proximal Limit"
        final_rr, tp1 = check_rr(final_entry)

    if final_entry is None or final_rr < 1.5:
        return {"valid": False, "reason": "R:R < 1.5 on all cascade attempts"}

    risk = abs(final_entry - sl)
    tp2 = final_entry + (risk * 4.0) if bias == "LONG" else final_entry - (risk * 4.0)

    return {"valid": True, "entry": round(final_entry, dp), "sl": round(sl, dp), "tp1": round(tp1, dp), "tp2": round(tp2, dp), "rr": round(final_rr, 2), "entry_source": entry_source}

def run_scorecard(bias, df_h1, ob, fvg, current_price, pair_conf=None, df_m15=None, macro_score=1.0):
    bos_tag = ob.get('bos_tag', 'BOS')
    if bos_tag == 'CHoCH':
        bd = {"structure": 2.5}
    else:
        bd = {"structure": 1.5}
    
    # H1 sweep
    swings_h1 = get_swing_points(df_h1, lookback=5)
    h1_sweep_score, h1_sweep_price = detect_sweep_decay(df_h1, swings_h1, len(df_h1)-1)
    
    # M15 sweep (if data provided)
    m15_sweep_score, m15_sweep_price = 0.0, None
    if df_m15 is not None:
        swings_m15 = get_swing_points(df_m15, lookback=4)
        m15_sweep_score, m15_sweep_price = detect_sweep_decay(df_m15, swings_m15, len(df_m15)-1)
    
    # Take the best sweep from either timeframe
    if m15_sweep_score > h1_sweep_score:
        bd["sweep"] = m15_sweep_score
        sweep_price = m15_sweep_price
    else:
        bd["sweep"] = h1_sweep_score
        sweep_price = h1_sweep_price
    
    if fvg and fvg.get('exists'):
        bd["fvg"] = 1.5 if fvg.get('touches_ob', False) else 1.0
    else:
        bd["fvg"] = 0.0

    is_fresh = True
    ob_top, ob_bottom = float(ob.get('proximal_line', 0)), float(ob.get('distal_line', 0))
    for i in range(len(df_h1) - 5, len(df_h1) - 1): 
        if bias == "LONG" and df_h1['Low'].iloc[i] <= ob_top: is_fresh = False; break
        elif bias == "SHORT" and df_h1['High'].iloc[i] >= ob_bottom: is_fresh = False; break
    bd["freshness"] = 0.5 if is_fresh else 0.0

    eq = (df_h1['High'].max() + df_h1['Low'].min()) / 2.0
    bd["pd"] = 1.0 if (bias == "LONG" and current_price <= eq) or (bias == "SHORT" and current_price >= eq) else 0.0
    
    # Killzone — pair-aware
    from datetime import datetime, timedelta
    ist_hour = (datetime.utcnow() + timedelta(hours=5, minutes=30)).hour
    pair_type = "forex"
    if pair_conf:
        pair_type = pair_conf.get("pair_type", "forex")
    
    kz_hit = False
    if pair_type == "forex":
        kz_hit = (12 <= ist_hour <= 15) or (18 <= ist_hour <= 21)
    elif pair_type == "index":
        kz_hit = (19 <= ist_hour <= 23) or (0 <= ist_hour <= 1)
    elif pair_type == "commodity":
        kz_hit = (12 <= ist_hour <= 15) or (18 <= ist_hour <= 21)
    bd["killzone"] = 1.0 if kz_hit else 0.0
    
    bd["macro"] = macro_score

    return {"total": round(sum(bd.values()), 1), "breakdown": bd, "sweep_price": sweep_price}
# Phase 3 CHoCH Extractor
def detect_ltf_choch(df_m5, bias, bounds):
    # Strictly isolates CHoCH within H1/M15 distal boundaries
    swings = get_swing_points(df_m5, lookback=3, bounds=bounds)
    if len(swings) < 2: return False
    
    C = df_m5['Close'].values
    latest_high = [s for s in swings if s['type'] == 'high']
    latest_low  = [s for s in swings if s['type'] == 'low']
    
    if bias == 'LONG' and latest_high:
        if C[-1] > latest_high[-1]['price'] and C[-2] <= latest_high[-1]['price']:
            return True # CHoCH prioritized over standard BOS
    elif bias == 'SHORT' and latest_low:
        if C[-1] < latest_low[-1]['price'] and C[-2] >= latest_low[-1]['price']:
            return True
    return False
