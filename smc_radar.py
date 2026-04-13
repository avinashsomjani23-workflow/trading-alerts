import yfinance as yf
import pandas as pd
import numpy as np
import json
import smtplib
import logging
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logging.basicConfig(filename="smc_radar.log", level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

with open("config.json") as f:
    config_master = json.load(f)

def fetch_data(ticker, interval, period):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
    return df.tail(150).copy().reset_index()

def is_valid_ob_candle(open_p, close_p, high_p, low_p):
    body = abs(open_p - close_p)
    rng  = high_p - low_p
    if rng == 0: return False
    return body > (rng * 0.15)

def detect_smc_radar(df, lookback):
    n = len(df)
    O, C, H, L = df['Open'].values, df['Close'].values, df['High'].values, df['Low'].values
    swings, active_obs, bos_events = [], [], []
    trend_state = None   

    for i in range(lookback, n - lookback):
        if H[i] == max(H[i - lookback: i + lookback + 1]): swings.append({'type': 'high', 'idx': i, 'price': float(H[i])})
        elif L[i] == min(L[i - lookback: i + lookback + 1]): swings.append({'type': 'low',  'idx': i, 'price': float(L[i])})
    
    swings = sorted(swings, key=lambda x: x['idx'])

    for i in range(lookback + 1, n):
        past_swings = [s for s in swings if s['idx'] < i]
        if len(past_swings) < 2: continue
        latest_high = [s for s in past_swings if s['type'] == 'high']
        latest_low  = [s for s in past_swings if s['type'] == 'low']
        if not latest_high or not latest_low: continue

        sh, sl = latest_high[-1], latest_low[-1]
        bos_detected, bos_type = False, None

        if C[i] > sh['price'] and C[i - 1] <= sh['price']:
            bos_detected, bos_type = True, 'bullish'
        elif C[i] < sl['price'] and C[i - 1] >= sl['price']:
            bos_detected, bos_type = True, 'bearish'

        if bos_detected:
            bos_tag = 'CHoCH+BOS' if (trend_state is None or trend_state != bos_type) else 'continuation_BOS'
            trend_state = bos_type
            bos_events.append({'type': bos_type, 'tag': bos_tag, 'idx': i, 'price': C[i]})

            ob_idx = -1
            impulse_start_idx = sl['idx'] if bos_type == 'bullish' else sh['idx']
            for j in range(i - 1, impulse_start_idx - 1, -1):
                if (bos_type == 'bullish' and C[j] < O[j]) or (bos_type == 'bearish' and C[j] > O[j]):
                    if is_valid_ob_candle(O[j], C[j], H[j], L[j]):
                        ob_idx = j; break
            
            if ob_idx == -1: continue

            fvg_valid, fvg_top, fvg_bot = False, None, None
            for k in range(ob_idx, min(ob_idx + 6, n) - 2):
                if bos_type == 'bullish' and H[k] < L[k + 2]:
                    fvg_valid, fvg_top, fvg_bot = True, float(L[k + 2]), float(H[k])
                    break
                elif bos_type == 'bearish' and L[k] > H[k + 2]:
                    fvg_valid, fvg_top, fvg_bot = True, float(H[k]), float(L[k + 2])
                    break

            if not fvg_valid: continue

            ob_high, ob_low = float(H[ob_idx]), float(L[ob_idx])
            fvg_touches = True if (fvg_bot <= ob_high and fvg_top >= ob_low) else False

            active_obs.append({
                'bos_idx': i, 'ob_idx': ob_idx, 'direction': bos_type, 'bos_tag': bos_tag,
                'high': ob_high, 'low': ob_low, 'mean': float((ob_high + ob_low) / 2),
                'proximal_line': ob_high if bos_type == 'bullish' else ob_low,
                'distal_line': ob_low if bos_type == 'bullish' else ob_high,
                'fvg': {'exists': True, 'fvg_top': fvg_top, 'fvg_bottom': fvg_bot, 'touches_ob': fvg_touches}
            })

    pristine_obs, current_price = [], float(C[-1])
    for ob in active_obs:
        mitigated = False
        for m in range(ob['ob_idx'] + 2, n):
            if (ob['direction'] == 'bullish' and C[m] < ob['distal_line']) or (ob['direction'] == 'bearish' and C[m] > ob['distal_line']):
                mitigated = True; break
        if not mitigated:
            ob['dist_to_price'] = abs(current_price - ob['proximal_line'])
            ob['direction_label'] = "Demand" if ob['direction'] == 'bullish' else "Supply"
            pristine_obs.append(ob)

    return {"active_unmitigated_obs": pristine_obs}

def run_radar():
    print(f"Running Phase 1 Scout (smc_radar) at {datetime.utcnow().strftime('%H:%M')} UTC")
    export_payload = {}
    for pair in config_master["pairs"]:
        ticker, name, lookback = pair["symbol"], pair["name"], 4
        if name in ["NZDUSD", "XAUUSD"]: lookback = 5
        elif name == "NAS100": lookback = 6
        
        df = fetch_data(ticker, pair["map_tf"], "15d")
        if df is not None:
            result = detect_smc_radar(df, lookback)
            export_payload[name] = result["active_unmitigated_obs"]
            print(f"  Mapped {len(result['active_unmitigated_obs'])} pristine OBs for {name}")

    with open("active_obs.json", "w") as f:
        json.dump(export_payload, f, indent=2)
    print("Phase 1 Complete. Saved to active_obs.json.")

if __name__ == "__main__":
    run_radar()
