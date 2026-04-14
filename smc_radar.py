import yfinance as yf
import pandas as pd
import numpy as np
import json
import smtplib
import logging
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import base64
from io import BytesIO

logging.basicConfig(filename="smc_radar.log", level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

with open("config.json") as f:
    config_master = json.load(f)

EMAIL_CONFIG = {
    "sender":      ["avinash.somjani23@gmail.com"],
    "recipient":   config_master["account"].get("alert_emails", ["avinash.somjani23@gmail.com"]),
    "smtp_server": "smtp.gmail.com",
    "smtp_port":   587,
    "password":    os.environ.get("GMAIL_APP_PASSWORD", "dummy")
}

def get_ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def get_last_9am_ist():
    now_ist = get_ist_now()
    if now_ist.hour >= 9:
        return now_ist.replace(hour=9, minute=0, second=0, microsecond=0)
    else:
        return (now_ist - timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)

def load_audit_log():
    try:
        with open("emailed_zones.json", "r") as f: return json.load(f)
    except FileNotFoundError:
        return {}

def save_audit_log(log_data):
    with open("emailed_zones.json", "w") as f: json.dump(log_data, f, indent=2)

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

    swings = []
    for i in range(lookback, n - lookback):
        window_highs = H[i - lookback: i + lookback + 1]
        window_lows  = L[i - lookback: i + lookback + 1]
        if H[i] == max(window_highs): swings.append({'type': 'high', 'idx': i, 'price': float(H[i])})
        elif L[i] == min(window_lows): swings.append({'type': 'low',  'idx': i, 'price': float(L[i])})

    swings = sorted(swings, key=lambda x: x['idx'])
    active_obs, bos_events, trend_state = [], [], None   

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
            bos_tag = 'CHoCH' if (trend_state is None or trend_state != bos_type) else 'BOS'
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
                        if abs(C[j] - O[j]) <= (2.0 * median_leg_body):
                            ob_idx = j
                            break

            if ob_idx == -1: continue

            ob_high, ob_low = float(H[ob_idx]), float(L[ob_idx])
            
            # Strict C2-C5 FVG Window
            fvg_valid, fvg_top, fvg_bottom = False, None, None
            c2_idx = ob_idx + 1
            max_c5_idx = min(ob_idx + 4, n - 1)
            
            for k in range(c2_idx, max_c5_idx):
                if k + 2 >= n: break
                if bos_type == 'bullish' and H[k] < L[k + 2]:
                    fvg_valid = True
                    fvg_top, fvg_bottom = float(L[k + 2]), float(H[k])
                    break
                elif bos_type == 'bearish' and L[k] > H[k + 2]:
                    fvg_valid = True
                    fvg_top, fvg_bottom = float(H[k]), float(L[k + 2])
                    break

            active_obs.append({
                'bos_idx': i, 'ob_idx': ob_idx, 'direction': bos_type, 'bos_tag': bos_tag,
                'high': ob_high, 'low': ob_low, 'mean': float((ob_high + ob_low) / 2),
                'proximal_line': ob_high if bos_type == 'bullish' else ob_low,
                'distal_line': ob_low if bos_type == 'bullish' else ob_high,
                'median_leg_body': median_leg_body, 'ob_body': abs(C[ob_idx]-O[ob_idx]),
                'fvg': {'exists': fvg_valid, 'fvg_top': fvg_top, 'fvg_bottom': fvg_bottom}
            })

    tracked_obs = []
    for ob in active_obs:
        mitigated = False
        touches = 0
        
        # Walk forward from the OB to current price
        for m in range(ob['ob_idx'] + 2, n):
            if ob['direction'] == 'bullish':
                # 1. Dead? (Body CLOSES below distal line)
                if C[m] < ob['distal_line']:
                    mitigated = True
                    break
                # 2. Touched? (Wick PIERCES proximal line)
                elif L[m] <= ob['proximal_line']:
                    touches += 1
            
            elif ob['direction'] == 'bearish':
                # 1. Dead? (Body CLOSES above distal line)
                if C[m] > ob['distal_line']:
                    mitigated = True
                    break
                # 2. Touched? (Wick PIERCES proximal line)
                elif H[m] >= ob['proximal_line']:
                    touches += 1
            
            # 3. Exhaustion Dead? (More than 3 touches kills the zone)
            if touches > 3:
                mitigated = True
                break
                    
        if not mitigated:
            ob['touches'] = touches
            ob['status'] = 'Pristine' if touches == 0 else f'Tested ({touches} touches)'
            tracked_obs.append(ob)

    return {"current_price": float(C[-1]), "active_unmitigated_obs": tracked_obs}

def generate_h1_chart(df, title, ob, dp, pair_conf):
    try:
        df_plot = df.tail(60).copy().reset_index(drop=True)
        fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), facecolor='#131722')
        ax.set_facecolor('#131722')
        for s in ax.spines.values(): s.set_color('#2a2a3e')

        for i, row in df_plot.iterrows():
            o, h, l, c = float(row['Open']), float(row['High']), float(row['Low']), float(row['Close'])
            col_c = '#26a69a' if c >= o else '#ef5350'
            ax.plot([i,i], [l,h], color=col_c, linewidth=1.5, zorder=2)
            body = abs(c-o) or (h-l) * 0.02
            ax.add_patch(patches.Rectangle((i-0.4), min(o,c), 0.8, body, facecolor=col_c, linewidth=0, alpha=0.95, zorder=3))

        n = len(df_plot)
        proximal, distal = float(ob['proximal_line']), float(ob['distal_line'])
        ax.add_patch(patches.Rectangle((0, min(proximal, distal)), n+5, abs(proximal-distal), fill=False, edgecolor='#aaaaaa', linestyle=':', linewidth=1.5, zorder=2))
        ax.add_patch(patches.Rectangle((0, min(proximal, distal)), n+5, abs(proximal-distal), facecolor='#9b59b6', alpha=0.15, zorder=1))

        if ob['fvg']['exists']:
            ft, fb = float(ob['fvg']['fvg_top']), float(ob['fvg']['fvg_bottom'])
            ax.add_patch(patches.Rectangle((0, fb), n+5, ft - fb, facecolor='#2ecc71', alpha=0.15, zorder=1))

        # Dynamic Scaling based on pip value
        pip_val = pair_conf.get("spread_pips", 2) * (0.0001 if dp == 5 else 0.01)
        padding = pip_val * 10 
        ax.set_ylim(min(proximal, distal) - padding, max(proximal, distal) + padding)
        
        ax.set_title(title, color='#dddddd', fontsize=12, pad=10, loc='left')
        ax.tick_params(colors='#888', labelsize=9); ax.yaxis.tick_right()
        ax.set_xlim(-1, n + 5)
        plt.tight_layout(pad=0.5)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#131722', edgecolor='none')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return b64
    except Exception as e:
        print(f"Chart err: {e}"); return None

def send_scout_email(pair, ob, dp, chart_b64, pair_conf):
    ist_time = get_ist_now().strftime('%H:%M IST')
    dir_text = "Bullish (Demand)" if ob['direction'] == 'bullish' else "Bearish (Supply)"
    status_label = f"<span style='color:{'#27ae60' if ob['touches'] == 0 else '#e67e22'};'><b>{ob['status']}</b></span>"
    
    # Logic Translation
    fvg_text = f"A 3-candle FVG confirmed the displacement between {ob['fvg']['fvg_bottom']:.{dp}f} and {ob['fvg']['fvg_top']:.{dp}f}." if ob['fvg']['exists'] else "No immediate FVG detected in the C2-C5 window."
    explanation = f"<b>Methodology:</b> A {ob['bos_tag']} was detected. We walked backward from the impulse leg (median body size: {ob['median_leg_body']:.{dp}f}) to find the accumulation candle. The opposing candle body ({ob['ob_body']:.{dp}f}) strictly satisfied the &lt; 2.0x rule. {fvg_text}<br><br><b>Zone Status:</b> {status_label}"

    html = f"""<html><body style="font-family:Arial;background:#f0f2f5;padding:12px;">
    <div style="max-width:620px;margin:auto;background:white;border-radius:10px;padding:20px;">
        <h3 style="margin:0;">Phase 1 Scout | {pair} | {dir_text}</h3>
        <p style="color:#555;font-size:12px;">Scanned at {ist_time}</p>
        <p style="font-size:14px;background:#f8f9fa;padding:10px;border-left:4px solid #9b59b6;">{explanation}</p>
        <p style="font-size:12px;"><b>Zone:</b> {ob['proximal_line']:.{dp}f} to {ob['distal_line']:.{dp}f}</p>
        <img src="cid:chart_h1" style="width:100%;border-radius:8px;margin-top:10px;" />
    </div></body></html>"""

    msg = MIMEMultipart("related")
    msg['From'], msg['To'], msg['Subject'] = EMAIL_CONFIG['sender'][0], ", ".join(EMAIL_CONFIG['recipient']), f"Scout | {pair} | {dir_text} | {ist_time}"
    msg.attach(MIMEText(html, 'html'))
    if chart_b64:
        img = MIMEImage(base64.b64decode(chart_b64))
        img.add_header("Content-ID", "<chart_h1>")
        msg.attach(img)
    
    with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
        server.starttls()
        server.login(EMAIL_CONFIG['sender'][0], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['sender'][0], EMAIL_CONFIG['recipient'], msg.as_string())

def run_radar():
    ist_now = get_ist_now()
    print(f"Running Phase 1 Scout at {ist_now.strftime('%H:%M')} IST")
    
    export_payload = {}
    audit_log = load_audit_log()
    cutoff_time = get_last_9am_ist().isoformat()
    spam_count = 0

    for pair in config_master["pairs"]:
        ticker, name, dp = pair["symbol"], pair["name"], pair.get("decimal_places", 5)
        lookback = 5 if name in ["NZDUSD", "GOLD"] else 6 if name == "NAS100" else 4
        
        df = fetch_data(ticker, pair["map_tf"], "15d")
        if df is not None:
            result = detect_smc_radar(df, lookback)
            export_payload[name] = result["active_unmitigated_obs"]
            
            for ob in result["active_unmitigated_obs"]:
                zone_id = f"{name}_{ob['direction']}_{round(ob['proximal_line'], dp)}"
                last_sent = audit_log.get(zone_id)
                
                # Check if it was sent TODAY (after the most recent 9 AM IST)
                if last_sent and last_sent > cutoff_time:
                    spam_count += 1
                    continue
                
                chart_b64 = generate_h1_chart(df, f"{name} H1 POI", ob, dp, pair)
                send_scout_email(name, ob, dp, chart_b64, pair)
                audit_log[zone_id] = ist_now.isoformat()
                print(f"  Alerted fresh H1 zone for {name}")

    save_audit_log(audit_log)
    with open("active_obs.json", "w") as f: json.dump(export_payload, f, indent=2)
    print(f"Phase 1 Complete. {spam_count} charts skipped (already sent today).")

if __name__ == "__main__":
    run_radar()
