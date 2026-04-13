"""
smc_detector.py — v3.0 (Strict SMC Execution Engine)
Handles: Step 3 Risk Math, Dynamic Entry Shifting, and 10-Point Scorecard
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def _dp(pair_conf): return pair_conf.get("decimal_places", 5)

def get_swing_points(df, lookback=4):
    """Maps structural swings for BSL/SSL targets and sweeping."""
    if df is None or len(df) < lookback * 2 + 1: return []
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    swings = []
    for i in range(lookback, len(H) - lookback):
        if H[i] == max(H[i - lookback: i + lookback + 1]):
            swings.append({"type": "high", "price": float(H[i]), "idx": i, "ts": df.index[i]})
        if L[i] == min(L[i - lookback: i + lookback + 1]):
            swings.append({"type": "low", "price": float(L[i]), "idx": i, "ts": df.index[i]})
    return sorted(swings, key=lambda s: s["idx"])

def detect_sweep_decay(df, swings, current_idx):
    """Rule 2: Time-Decay Sweeps (24h = 2.5, 72h = 1.5)"""
    score = 0.0
    H, L = df['High'].values, df['Low'].values
    O, C = df['Open'].values, df['Close'].values
    current_ts = df.index[current_idx]
    
    # Check last 3 candles for a wick sweep
    for i in range(max(0, current_idx - 3), current_idx + 1):
        for s in swings:
            if s['idx'] >= i: continue
            hours_old = (current_ts - s['ts']).total_seconds() / 3600
            if hours_old > 72: continue
            
            # Bullish Sweep (Sweeping SSL)
            if s['type'] == 'low' and L[i] < s['price'] and C[i] > s['price']:
                pts = 2.5 if hours_old <= 24 else 1.5
                score = max(score, pts)
            # Bearish Sweep (Sweeping BSL)
            elif s['type'] == 'high' and H[i] > s['price'] and C[i] < s['price']:
                pts = 2.5 if hours_old <= 24 else 1.5
                score = max(score, pts)
    return score

def compute_dynamic_levels(pair_conf, bias, ob, fvg, current_price, df_trigger):
    """Step 3: Risk, Entry Shifting, and TP Math"""
    dp = _dp(pair_conf)
    spread = pair_conf.get("spread_pips", 0)
    if dp == 5: spread_val = spread * 0.0001
    elif dp == 3: spread_val = spread * 0.01
    else: spread_val = spread

    # 1. SL Math (Distal ± Spread, NO ATR)
    if bias == "LONG":
        sl = float(ob['ob_bottom']) - spread_val
    else:
        sl = float(ob['ob_top']) + spread_val

    # 2. Entry Coordinates
    ob_prox = float(ob['ob_top']) if bias == "LONG" else float(ob['ob_bottom'])
    ob_mean = (float(ob['ob_top']) + float(ob['ob_bottom'])) / 2.0
    fvg_prox = None
    
    if fvg and fvg.get('exists'):
        fvg_prox = float(fvg['fvg_top']) if bias == "LONG" else float(fvg['fvg_bottom'])

    # 3. TP Mapping (Internal Liquidity)
    swings = get_swing_points(df_trigger, lookback=10)
    tp_targets = []
    for s in swings:
        if bias == "LONG" and s['type'] == 'high' and s['price'] > ob_prox:
            tp_targets.append(s['price'])
        elif bias == "SHORT" and s['type'] == 'low' and s['price'] < ob_prox:
            tp_targets.append(s['price'])
    
    tp_targets.sort(reverse=(bias == "LONG")) # Closest targets first

    # 4. Entry Cascade & R:R Verification
    entry_model = pair_conf.get("pair_type", "forex")
    final_entry, final_rr = None, 0.0
    entry_source = ""
    tp1 = None

    def check_rr(entry_test):
        risk = abs(entry_test - sl)
        if risk == 0: return 0.0, None
        for target in tp_targets:
            rr = abs(target - entry_test) / risk
            if rr >= 1.5: return rr, target
        # Fallback if no structural targets exist
        fallback_tp = entry_test + (risk * 2.0) if bias == "LONG" else entry_test - (risk * 2.0)
        return 2.0, fallback_tp

    # Cascade Logic
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
    elif entry_model == "commodity": # Gold
        final_entry, entry_source = ob_mean, "OB 50% Mean"
        final_rr, tp1 = check_rr(final_entry)
    elif entry_model == "index": # NAS100
        final_entry, entry_source = ob_prox, "OB Proximal"
        final_rr, tp1 = check_rr(final_entry)

    # Hard Gate Abort
    if final_entry is None or final_rr < 1.5:
        return {"valid": False, "reason": "R:R < 1.5 on all entry cascade attempts"}

    # TP2 (External Liquidity / Runner)
    risk = abs(final_entry - sl)
    tp2 = final_entry + (risk * 4.0) if bias == "LONG" else final_entry - (risk * 4.0)

    return {
        "valid": True, "entry": round(final_entry, dp), "sl": round(sl, dp), 
        "tp1": round(tp1, dp), "tp2": round(tp2, dp), 
        "rr": round(final_rr, 2), "entry_source": entry_source
    }

def run_scorecard(bias, df_h1, ob, fvg, current_price):
    """Step 2: The 10-Point Scorecard Math"""
    score = 0.0
    bd = {}
    
    swings = get_swing_points(df_h1, lookback=5)
    
    # 1. Structure (Assume Phase 1 passed us an OB, so BOS exists. Default 1.5. If CHoCH+BOS, 2.5)
    # Note: Full robust structure tracking sits in Phase 1. For Phase 2, we default to 1.5, upgrade if sweep + BOS.
    bd["structure"] = 1.5 
    
    # 2. Liquidity Sweep Time-Decay
    sweep_score = detect_sweep_decay(df_h1, swings, len(df_h1)-1)
    bd["sweep"] = sweep_score
    
    # 3. OB + FVG Touch
    if fvg and fvg.get('exists'):
        if fvg.get('touches_ob', False): bd["fvg"] = 1.5
        else: bd["fvg"] = 1.0
    else: bd["fvg"] = 0.0

    # 4. OB Freshness (+0.5)
    # Check if price has entered the OB since its creation index
    ob_idx = ob.get('idx', 0)
    is_fresh = True
    for i in range(ob_idx + 3, len(df_h1) - 1):
        if bias == "LONG" and df_h1['Low'].iloc[i] <= float(ob['ob_top']):
            is_fresh = False; break
        elif bias == "SHORT" and df_h1['High'].iloc[i] >= float(ob['ob_bottom']):
            is_fresh = False; break
    bd["freshness"] = 0.5 if is_fresh else 0.0

    # 5. Premium/Discount (ERL 50%)
    h1_high = df_h1['High'].max()
    h1_low = df_h1['Low'].max()
    eq = (h1_high + h1_low) / 2.0
    if bias == "LONG" and current_price <= eq: bd["pd"] = 1.0
    elif bias == "SHORT" and current_price >= eq: bd["pd"] = 1.0
    else: bd["pd"] = 0.0

    # 6 & 7. Killzone & Macro (Handled via alert_engine, defaults here)
    bd["killzone"] = 1.0 
    bd["macro"] = 1.0

    total_score = sum(bd.values())
    return {"total": round(total_score, 1), "breakdown": bd}
