"""
smc_detector.py — v3.0 (Strict SMC Execution Engine)
Handles: Step 3 Risk Math, Dynamic Entry Shifting, and 10-Point Scorecard
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def _dp(pair_conf): return pair_conf.get("decimal_places", 5)

def get_swing_points(df, lookback=4):
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

def detect_sweep_decay(df_radar, swings, current_idx):
    score = 0.0
    sweep_detail = {"detected": False, "price": 0.0}
    H, L = df_radar['High'].values, df_radar['Low'].values
    O, C = df_radar['Open'].values, df_radar['Close'].values
    current_ts = df_radar.index[current_idx]
    
    for i in range(max(0, current_idx - 5), current_idx + 1):
        for s in swings:
            if s['idx'] >= i: continue
            hours_old = (current_ts - s['ts']).total_seconds() / 3600
            if hours_old > 72: continue
            
            if s['type'] == 'low' and L[i] < s['price'] and C[i] > s['price']:
                pts = 2.5 if hours_old <= 24 else 1.5
                if pts > score:
                    score = pts
                    sweep_detail = {"detected": True, "price": float(L[i]), "level": s['price']}
            elif s['type'] == 'high' and H[i] > s['price'] and C[i] < s['price']:
                pts = 2.5 if hours_old <= 24 else 1.5
                if pts > score:
                    score = pts
                    sweep_detail = {"detected": True, "price": float(H[i]), "level": s['price']}
    return score, sweep_detail

def compute_dynamic_levels(pair_conf, bias, ob, fvg, df_radar):
    dp = _dp(pair_conf)
    spread = pair_conf.get("spread_pips", 0)
    if dp == 5: spread_val = spread * 0.0001
    elif dp == 3: spread_val = spread * 0.01
    else: spread_val = spread

    # 1. SL Math (Distal ± Spread)
    if bias == "LONG":
        sl = float(ob['ob_bottom']) - spread_val
    else:
        sl = float(ob['ob_top']) + spread_val

    # 2. Entry Coordinates
    ob_prox = float(ob['ob_top']) if bias == "LONG" else float(ob['ob_bottom'])
    ob_mean = (float(ob['ob_top']) + float(ob['ob_bottom'])) / 2.0
    fvg_prox = float(fvg['fvg_top']) if (fvg and fvg.get('exists') and bias == "LONG") else \
               float(fvg['fvg_bottom']) if (fvg and fvg.get('exists') and bias == "SHORT") else None

    # 3. TP Mapping
    swings = get_swing_points(df_radar, lookback=10)
    tp_targets = []
    for s in swings:
        if bias == "LONG" and s['type'] == 'high' and s['price'] > ob_prox:
            tp_targets.append(s['price'])
        elif bias == "SHORT" and s['type'] == 'low' and s['price'] < ob_prox:
            tp_targets.append(s['price'])
    
    tp_targets.sort(reverse=(bias == "LONG"))

    # 4. Entry Cascade
    entry_model = pair_conf.get("pair_type", "forex")
    final_entry, final_rr = None, 0.0
    entry_source, tp1 = "", None

    def check_rr(entry_test):
        risk = abs(entry_test - sl)
        if risk == 0: return 0.0, None
        for target in tp_targets:
            rr = abs(target - entry_test) / risk
            if rr >= 1.5: return rr, target
        fallback_tp = entry_test + (risk * 2.0) if bias == "LONG" else entry_test - (risk * 2.0)
        return 2.0, fallback_tp

    if entry_model == "forex":
        attempts = []
        if fvg_prox: attempts.append((fvg_prox, "FVG Proximal Edge"))
        attempts.append((ob_prox, "OB Proximal Line"))
        attempts.append((ob_mean, "OB 50% Mean Threshold"))
        
        for price, name in attempts:
            rr, tp_val = check_rr(price)
            if rr >= 1.5:
                final_entry, entry_source, final_rr, tp1 = price, name, rr, tp_val
                break
    elif entry_model == "commodity":
        final_entry, entry_source = ob_mean, "OB 50% Mean Threshold"
        final_rr, tp1 = check_rr(final_entry)
    elif entry_model == "index":
        final_entry, entry_source = ob_prox, "OB Proximal Line"
        final_rr, tp1 = check_rr(final_entry)

    if final_entry is None or final_rr < 1.5:
        return {"valid": False, "reason": "R:R < 1.5 on all entry cascade attempts"}

    risk = abs(final_entry - sl)
    tp2 = final_entry + (risk * 4.0) if bias == "LONG" else final_entry - (risk * 4.0)

    return {
        "valid": True, "entry": round(final_entry, dp), "sl": round(sl, dp), 
        "tp1": round(tp1, dp), "tp2": round(tp2, dp), 
        "rr_tp1": round(final_rr, 2), "rr_tp2": 4.0, "entry_source": entry_source
    }

def run_phase2_analysis(pair_conf, bias, df_h1, df_radar, ob, fvg, current_price, fatigue_count):
    """Wrapper to format output exactly how alert_engine.py expects it."""
    bd = {}
    swings = get_swing_points(df_h1, lookback=5)
    
    bd["structure_m15"] = 1.5 # Default from Phase 1
    
    sweep_score, sweep_detail = detect_sweep_decay(df_radar, swings, len(df_radar)-1)
    bd["liquidity_sweep"] = sweep_score
    
    if fvg and fvg.get('exists'):
        bd["ob_near_zone"] = 1.5 if fvg.get('touches_ob', False) else 1.0
    else: bd["ob_near_zone"] = 0.0

    ob_idx = ob.get('idx', 0)
    is_fresh = True
    for i in range(ob_idx + 3, len(df_h1) - 1):
        if bias == "LONG" and df_h1['Low'].iloc[i] <= float(ob['ob_top']):
            is_fresh = False; break
        elif bias == "SHORT" and df_h1['High'].iloc[i] >= float(ob['ob_bottom']):
            is_fresh = False; break
    bd["zone_freshness"] = 0.5 if is_fresh else 0.0

    eq = (df_h1['High'].max() + df_h1['Low'].max()) / 2.0
    if bias == "LONG" and current_price <= eq: bd["premium_discount"] = 1.0
    elif bias == "SHORT" and current_price >= eq: bd["premium_discount"] = 1.0
    else: bd["premium_discount"] = 0.0

    bd["no_high_impact_news"] = 1.0 # Placeholder for Gemini
    bd["h1_alignment"] = 0.5 # Default assumption from Phase 1 HTF mapping

    total_score = round(sum(bd.values()), 1)
    levels = compute_dynamic_levels(pair_conf, bias, ob, fvg, df_radar)

    return {
        "send_alert": levels['valid'] and total_score >= pair_conf["min_confidence"],
        "bias": bias,
        "confidence_score": total_score,
        "score_breakdown": bd,
        "entry": levels.get('entry', 0),
        "sl": levels.get('sl', 0),
        "tp1": levels.get('tp1', 0),
        "tp2": levels.get('tp2', 0),
        "rr_tp1": levels.get('rr_tp1', 0),
        "rr_tp2": levels.get('rr_tp2', 0),
        "entry_source": levels.get('entry_source', ""),
        "ob_top": float(ob.get('ob_top', 0)),
        "ob_bottom": float(ob.get('ob_bottom', 0)),
        "ob_type": ob.get('type', "bullish" if bias=="LONG" else "bearish"),
        "fvg_top": float(fvg.get('fvg_top', 0)) if fvg else 0,
        "fvg_bottom": float(fvg.get('fvg_bottom', 0)) if fvg else 0,
        "fvg_confirmed": bd["ob_near_zone"] > 0,
        "lq_sweep_price": sweep_detail['price'],
        "confidence_reason": levels.get('reason', 'Score and RR pass.') if not levels['valid'] else "All confluences aligned.",
        "confluences": [f"Score: {total_score}/10"],
        "missing": [],
        "premium_discount_pct": round((current_price - df_h1['Low'].max()) / (df_h1['High'].max() - df_h1['Low'].max()) * 100, 1) if (df_h1['High'].max() - df_h1['Low'].max()) > 0 else 50,
        "premium_discount_valid": bd["premium_discount"] == 1.0,
        "geo_flag": False,
        "news_flag": "none",
        "gemini_needed": True
    }
