import yfinance as yf
import pandas as pd
import json, os, smtplib
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
import smc_detector

with open("config.json") as f: config = json.load(f)

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "dummy@gmail.com")
GMAIL_PASS    = os.environ.get("GMAIL_APP_PASSWORD", "dummy")

def get_ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def generate_m5_chart(df, title, levels, ob, pair_conf):
    try:
        dp = pair_conf.get("decimal_places", 5)
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
        proximal, distal = float(ob.get('proximal_line', 0)), float(ob.get('distal_line', 0))
        if proximal > 0 and distal > 0:
            ax.add_patch(patches.Rectangle((0, min(proximal, distal)), n+5, abs(proximal-distal), fill=False, edgecolor='#aaaaaa', linestyle=':', linewidth=1.5, zorder=2))
            ax.add_patch(patches.Rectangle((0, min(proximal, distal)), n+5, abs(proximal-distal), facecolor='#9b59b6', alpha=0.15, zorder=1))

        # Plot Levels
        for key, (color, lbl) in {'tp1': ('#27ae60', 'TP1'), 'entry': ('#e67e22', 'ENTRY'), 'sl': ('#e74c3c', 'SL')}.items():
            price = float(levels.get(key, 0))
            if price > 0:
                ax.axhline(y=price, color=color, linestyle='-', linewidth=1.5, alpha=0.9, zorder=4)
                ax.text(n + 1, price, f" {lbl}", color=color, fontsize=9, va='center', fontweight='bold', zorder=5)

        # Dynamic Scaling
        sl_p, tp1_p = float(levels.get('sl', 0)), float(levels.get('tp1', 0))
        if sl_p > 0 and tp1_p > 0:
            pip_val = pair_conf.get("spread_pips", 2) * (0.0001 if dp == 5 else 0.01)
            ax.set_ylim(min(sl_p, tp1_p) - (pip_val*10), max(sl_p, tp1_p) + (pip_val*10))

        ax.set_title(title, color='#dddddd', fontsize=12, pad=10, loc='left')
        ax.tick_params(colors='#888', labelsize=9); ax.yaxis.tick_right()
        ax.set_xlim(-1, n + 8); plt.tight_layout(pad=0.5)
        
        buf = BytesIO(); fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#131722')
        buf.seek(0); b64 = base64.b64encode(buf.read()).decode(); plt.close(fig)
        return b64
    except Exception as e: 
        print(f"Chart error: {e}")
        return None

def send_html_email(subject, html_body, chart_b64=None):
    recipients = config["account"].get("alert_emails", [])
    for recipient in recipients:
        msg = MIMEMultipart("related")
        msg["Subject"], msg["From"], msg["To"] = subject, GMAIL_ADDRESS, recipient
        msg.attach(MIMEText(html_body, "html"))
        if chart_b64:
            img = MIMEImage(base64.b64decode(chart_b64))
            img.add_header("Content-ID", "<chart_m5>")
            img.add_header("Content-Disposition", "inline", filename="chart_m5.png") # THE GMAIL FIX
            msg.attach(img)
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(GMAIL_ADDRESS, GMAIL_PASS)
                server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
        except Exception as e:
            print(f"Email failed: {e}")

def run_phase3():
    ist_now = get_ist_now()
    print(f"Phase 3 (M5 Trigger) started at {ist_now.strftime('%H:%M')} IST")
    try:
        with open("active_watch_state.json") as f: watch_state = json.load(f)
    except Exception:
        print("No active watch state found. Exiting.")
        return

    if not watch_state:
        print("Watch state empty. No pairs approaching LTF triggers.")
        return

    keys_to_delete = []

    for key, data in watch_state.items():
        pair_name, bias, ob = data["pair"], data["bias"], data.get("ob", {})
        distal, proximal = float(ob.get("distal_line", 0)), float(ob.get("proximal_line", 0))

        pair_conf = next((p for p in config["pairs"] if p["name"] == pair_name), None)
        if not pair_conf: continue

        df_m5 = yf.download(pair_conf["symbol"], period="3d", interval="5m", progress=False)
        if df_m5 is None or df_m5.empty: continue
        if isinstance(df_m5.columns, pd.MultiIndex): df_m5.columns = [col[0] for col in df_m5.columns]
        
        current_close = float(df_m5['Close'].iloc[-1])
        invalidated = (bias == "LONG" and current_close < distal) or (bias == "SHORT" and current_close > distal)

        if invalidated:
            print(f"  [X] {pair_name} INVALIDATED: M5 closed beyond distal line ({distal}).")
            html_inv = f"""<html><body style="font-family:Arial;padding:20px;"><h2 style="color:#e74c3c;">INVALIDATED: {pair_name}</h2><p>Price breached distal line at {distal}. Setup canceled.</p></body></html>"""
            send_html_email(f"INVALIDATED | {pair_name}", html_inv)
            keys_to_delete.append(key)
            continue

        tapped = (bias == "LONG" and df_m5['Low'].min() <= proximal) or (bias == "SHORT" and df_m5['High'].max() >= proximal)
        if not tapped:
            print(f"  [-] {pair_name}: Waiting for price to tap proximal line ({proximal}).")
            continue

        # Strict M5 boundaries based on the higher timeframe zone
        bounds = {'max': max(proximal, distal), 'min': min(proximal, distal)}
        
        # Check for the M5 CHoCH
        if smc_detector.detect_ltf_choch(df_m5, bias, bounds):
            print(f"  [✓] LTF TRIGGER FIRED: {pair_name} M5 CHoCH detected!")
            levels = data.get("levels", {})
            dp = pair_conf.get("decimal_places", 5)
            
            chart_b64 = generate_m5_chart(df_m5, f"{pair_name} M5 SNIPER TRIGGER", levels, ob, pair_conf)
            
            html_body = f"""<html><body style="font-family:Arial;background:#f0f2f5;padding:12px;">
            <div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;">
                <div style="background:#1a1a2e;padding:16px 20px;">
                    <h2 style="color:white;margin:0;">TRADE READY (M5 SNIPER): {pair_name}</h2>
                    <p style="color:#8899bb;margin:4px 0 0;font-size:11px;">{ist_now.strftime('%H:%M IST')}</p>
                </div>
                <div style="padding:16px 20px;">
                    <div style="background:#27ae60;padding:16px 20px;border-radius:10px;margin-bottom:16px;">
                        <p style="color:white;font-size:16px;font-weight:bold;margin:0;">{'SELL' if bias=='SHORT' else 'BUY'} MARKET/LIMIT at {levels.get('entry', proximal):,.{dp}f}</p>
                        <p style="color:white;margin:4px 0 0;">SL: {levels.get('sl', distal):,.{dp}f} &nbsp;|&nbsp; TP1: {levels.get('tp1', 0):,.{dp}f}</p>
                    </div>
                    <p style="font-size:13px;background:#f8f9fa;padding:10px;border-left:4px solid #27ae60;"><b>Logic Translation:</b> M5 Change of Character (CHoCH) detected strictly inside the Higher Timeframe Order Block constraints. Stop-loss remains protected beyond the distal line.</p>
                    <img src="cid:chart_m5" style="width:100%;border-radius:8px;margin-bottom:12px;" />
                    <p><b>Macro Context:</b> {data.get('macro_summary', 'N/A')}</p>
                </div>
            </div></body></html>"""
            
            send_html_email(f"TRADE READY (M5 SNIPER) | {pair_name} | {bias}", html_body, chart_b64)
            keys_to_delete.append(key)

    for k in keys_to_delete: del watch_state[k]
    with open("active_watch_state.json", "w") as f: json.dump(watch_state, f, indent=2)

if __name__ == "__main__": 
    run_phase3()
