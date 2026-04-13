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

EMAIL_CONFIG = {
    "sender":      ["avinash.somjani23@gmail.com"],
    "recipient":   ["avinash.somjani23@gmail.com", "fernandesbrezhnev@gmail.com"],
    "smtp_server": "smtp.gmail.com",
    "smtp_port":   587,
    "password":    os.environ.get("GMAIL_APP_PASSWORD")
}

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
    O = df['Open'].values
    C = df['Close'].values
    H = df['High'].values
    L = df['Low'].values

    swings = []
    for i in range(lookback, n - lookback):
        window_highs = H[i - lookback: i + lookback + 1]
        window_lows  = L[i - lookback: i + lookback + 1]

        if H[i] == max(window_highs): swings.append({'type': 'high', 'idx': i, 'price': float(H[i])})
        elif L[i] == min(window_lows): swings.append({'type': 'low',  'idx': i, 'price': float(L[i])})

    swings = sorted(swings, key=lambda x: x['idx'])

    active_obs  = []
    bos_events  = []
    trend_state = None   

    for i in range(lookback + 1, n):
        past_swings = [s for s in swings if s['idx'] < i]
        if len(past_swings) < 2: continue

        latest_high = [s for s in past_swings if s['type'] == 'high']
        latest_low  = [s for s in past_swings if s['type'] == 'low']

        if not latest_high or not latest_low: continue

        sh = latest_high[-1]
        sl = latest_low[-1]

        bos_detected = False
        bos_type     = None

        if C[i] > sh['price'] and C[i - 1] <= sh['price']:
            bos_detected = True; bos_type = 'bullish'
        elif C[i] < sl['price'] and C[i - 1] >= sl['price']:
            bos_detected = True; bos_type = 'bearish'

        if bos_detected:
            bos_tag = 'CHoCH+BOS' if (trend_state is None or trend_state != bos_type) else 'continuation_BOS'
            trend_state = bos_type
            bos_events.append({'type': bos_type, 'tag':  bos_tag, 'idx':  i, 'price': C[i]})

            ob_idx = -1
            impulse_start_idx = sl['idx'] if bos_type == 'bullish' else sh['idx']

            if impulse_start_idx >= i: continue

            leg_bodies = [abs(C[k] - O[k]) for k in range(impulse_start_idx, i + 1)]
            median_leg_body = float(np.median(leg_bodies)) if leg_bodies else 0.0001
            if median_leg_body == 0: median_leg_body = 0.0001

            for j in range(i - 1, impulse_start_idx - 1, -1):
                if (bos_type == 'bullish' and C[j] < O[j]) or (bos_type == 'bearish' and C[j] > O[j]):
                    if is_valid_ob_candle(O[j], C[j], H[j], L[j]):
                        candle_body = abs(C[j] - O[j])
                        if candle_body <= (2.0 * median_leg_body):
                            ob_idx = j
                            break

            if ob_idx == -1: continue

            ob_high = float(H[ob_idx])
            ob_low  = float(L[ob_idx])

            fvg_valid  = False
            fvg_top = None
            fvg_bottom = None
            window_end = min(ob_idx + 6, n)

            for k in range(ob_idx, window_end - 2):
                if bos_type == 'bullish' and H[k] < L[k + 2]:
                    fvg_valid = True
                    fvg_top = float(L[k + 2])
                    fvg_bottom = float(H[k])
                    break
                elif bos_type == 'bearish' and L[k] > H[k + 2]:
                    fvg_valid = True
                    fvg_top = float(H[k])
                    fvg_bottom = float(L[k + 2])
                    break

            if not fvg_valid: continue

            active_obs.append({
                'bos_idx': i, 'ob_idx': ob_idx, 'direction': bos_type, 'bos_tag': bos_tag,
                'high': ob_high, 'low': ob_low, 'mean': float((ob_high + ob_low) / 2),
                'proximal_line': ob_high if bos_type == 'bullish' else ob_low,
                'distal_line': ob_low if bos_type == 'bullish' else ob_high,
                'fvg': {'exists': True, 'fvg_top': fvg_top, 'fvg_bottom': fvg_bottom}
            })

    pristine_obs  = []
    current_price = float(C[-1])

    for ob in active_obs:
        mitigated = False
        for m in range(ob['ob_idx'] + 2, n):
            if (ob['direction'] == 'bullish' and C[m] < ob['distal_line']) or \
               (ob['direction'] == 'bearish' and C[m] > ob['distal_line']):
                mitigated = True; break

        if not mitigated:
            dist = abs(current_price - ob['proximal_line'])
            pristine_obs.append({
                "direction": "Bullish (Demand)" if ob['direction'] == 'bullish' else "Bearish (Supply)",
                "bos_tag": ob['bos_tag'],
                "proximal_line": round(ob['proximal_line'], 5),
                "distal_line": round(ob['distal_line'], 5),
                "dist_to_price": round(dist, 5),
                "ob_high": round(ob['high'], 5),
                "ob_low": round(ob['low'], 5),
                "ob_mean": round(ob['mean'], 5),
                "fvg": ob['fvg']
            })

    return {
        "current_price": round(current_price, 5),
        "active_unmitigated_obs": pristine_obs
    }

def send_summary_email(payload):
    try:
        subject = f"Phase 1 Scout Summary — {datetime.utcnow().strftime('%H:%M')} UTC"
        lines = ["SMC Radar (Phase 1) Active Order Blocks:\n" + "="*40]
        total_obs = 0
        for pair, obs in payload.items():
            lines.append(f"{pair}: {len(obs)} pristine OB(s) mapped.")
            total_obs += len(obs)
        lines.append("="*40)
        lines.append(f"Total active zones being tracked by Phase 2: {total_obs}")
        body = "\n".join(lines)

        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender'][0]
        msg['To'] = ", ".join(EMAIL_CONFIG['recipient'])
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender'][0], EMAIL_CONFIG['password'])
            server.sendmail(EMAIL_CONFIG['sender'][0], EMAIL_CONFIG['recipient'], msg.as_string())
        logging.info("2-Hour Summary Email dispatched successfully.")
    except Exception as e:
        logging.error(f"Summary email failed (System continued safely): {e}")

def run_radar():
    print(f"Running Phase 1 Scout (smc_radar) at {datetime.utcnow().strftime('%H:%M')} UTC")
    export_payload = {}
    
    for pair in config_master["pairs"]:
        ticker, name = pair["symbol"], pair["name"]
        lookback = 5 if name in ["NZDUSD", "GOLD"] else 6 if name == "NAS100" else 4
        
        df = fetch_data(ticker, pair["map_tf"], "15d")
        if df is not None:
            result = detect_smc_radar(df, lookback)
            export_payload[name] = result["active_unmitigated_obs"]
            print(f"  Mapped {len(result['active_unmitigated_obs'])} pristine OBs for {name}")

    with open("active_obs.json", "w") as f:
        json.dump(export_payload, f, indent=2)
    print("Phase 1 Complete. Saved to active_obs.json.")

    run_time = datetime.utcnow()
    if run_time.hour % 2 == 0:
        send_summary_email(export_payload)

if __name__ == "__main__":
    run_radar()
