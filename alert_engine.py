import yfinance as yf
import pandas as pd
import json, os, smtplib, requests, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import numpy as np
import base64
from io import BytesIO
import smc_detector

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "dummy")
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "dummy@gmail.com")
GMAIL_PASS    = os.environ.get("GMAIL_APP_PASSWORD", "dummy")
ALERT_EMAIL   = os.environ.get("ALERT_EMAIL", "dummy@gmail.com")

def load_json(path, default):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return default

def save_json(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=2)

def utc_now(): return datetime.utcnow()
def ist_now(): return utc_now() + timedelta(hours=5, minutes=30)
def ist_str(): return ist_now().strftime("%d %b %Y %H:%M IST")
def utc_str(): return utc_now().strftime("%Y-%m-%d %H:%M")
def fmt_price(price, pair_conf):
    dp = pair_conf.get("decimal_places", 5)
    return f"{float(price):,.{dp}f}"

def clean_df(df):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def get_atr(df, period=14):
    try:
        highs  = df['High'].values.flatten().astype(float)
        lows   = df['Low'].values.flatten().astype(float)
        closes = df['Close'].values.flatten().astype(float)
        trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(closes))]
        if len(trs) < period: return None
        return float(np.mean(trs[-period:]))
    except Exception: return None

def generate_chart(df, title, levels, data, pair_conf):
    """Institutional Matplotlib Generator"""
    try:
        if df is None or df.empty: return None
        dp = pair_conf.get("decimal_places", 5)
        df_plot = df.tail(50).copy().reset_index(drop=True)
        fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), facecolor='#131722')
        ax.set_facecolor('#131722')
        for s in ax.spines.values(): s.set_color('#2a2a3e')

        for i, row in df_plot.iterrows():
            o, h, l, c = float(row['Open']), float(row['High']), float(row['Low']), float(row['Close'])
            col_c = '#26a69a' if c >= o else '#ef5350'
            ax.plot([i,i], [l,h], color=col_c, linewidth=1.2, zorder=2)
            body = abs(c-o) or (h-l) * 0.02
            ax.add_patch(patches.Rectangle((i-0.35, min(o,c)), 0.7, body, facecolor=col_c, linewidth=0, alpha=0.95, zorder=3))

        n = len(df_plot)
        ob_top, ob_bottom = float(data.get('ob_top', 0)), float(data.get('ob_bottom', 0))
        if ob_top > 0 and ob_bottom > 0:
            ob_color = '#E8A838'
            ax.add_patch(patches.Rectangle((max(0, n-15), ob_bottom), min(n, 15), ob_top - ob_bottom, facecolor=ob_color, edgecolor=ob_color, linewidth=2.0, alpha=0.20, zorder=1))
            ax.text(max(0.5, n-14.5), ob_top, "OB", color=ob_color, fontsize=9, va='bottom', fontweight='bold', zorder=5)

        level_cfg = {
            'tp2': ('#1e8449', '--', 1.0, 'TP2'), 'tp1': ('#27ae60', '-', 1.5, 'TP1'),
            'entry': ('#e67e22', '-', 1.5, 'ENTRY'), 'sl': ('#e74c3c', '-', 1.5, 'SL')
        }
        
        for key, (color, style, width, lbl) in level_cfg.items():
            price = float(levels.get(key, 0))
            if price > 0:
                ax.axhline(y=price, color=color, linestyle=style, linewidth=width, alpha=0.85, zorder=4)
                ax.text(n + 1, price, f" {lbl}: {price:,.{dp}f}", color=color, fontsize=8, va='center', fontweight='bold', zorder=5)

        ax.set_title(title, color='#dddddd', fontsize=11, pad=8, fontweight='bold', loc='left')
        ax.tick_params(colors='#888', labelsize=8)
        ax.yaxis.tick_right()
        ax.set_xlim(-1, n + 10)

        plt.tight_layout(pad=0.5)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#131722', edgecolor='none')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return b64
    except Exception as e:
        print(f"Chart error: {e}")
        return None

def build_trade_ready_email_html(data, pair, pair_conf, current_price, chart1_b64):
    dp = pair_conf.get("decimal_places", 5)
    bias = data.get("bias", "—")
    entry_p = fmt_price(data.get("entry", 0), pair_conf)
    sl_p = fmt_price(data.get("sl", 0), pair_conf)
    tp1_p = fmt_price(data.get("tp1", 0), pair_conf)
    tp2_p = fmt_price(data.get("tp2", 0), pair_conf)

    chart1_html = f'<img src="cid:chart_m15" style="width:100%;border-radius:8px;display:block;margin-bottom:12px;" />' if chart1_b64 else ''

    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:12px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;">
  <div style="background:#1a1a2e;padding:16px 20px;"><h2 style="color:white;margin:0;">TRADE READY: {pair}</h2></div>
  <div style="padding:16px 20px;">
    <div style="background:#27ae60;padding:16px 20px;border-radius:10px;margin-bottom:16px;">
      <p style="color:white;font-size:16px;font-weight:bold;margin:0;">{'SELL' if bias=='SHORT' else 'BUY'} LIMIT at {entry_p}</p>
      <p style="color:white;margin:4px 0 0;">SL: {sl_p} &nbsp;|&nbsp; TP1: {tp1_p} &nbsp;|&nbsp; TP2: {tp2_p}</p>
    </div>
    {chart1_html}
    <p>Score: {data.get('confidence_score')}/10 | Entry Model: {data.get('entry_source')}</p>
  </div>
</div></body></html>"""

def send_email(subject, html_body, chart1_b64=None):
    recipients = config["account"].get("alert_emails", [])
    for recipient in recipients:
        msg = MIMEMultipart("related")
        msg["Subject"], msg["From"], msg["To"] = subject, GMAIL_ADDRESS, recipient
        msg.attach(MIMEText(html_body, "html"))
        if chart1_b64:
            img = MIMEImage(base64.b64decode(chart1_b64))
            img.add_header("Content-ID", f"<chart_m15>")
            msg.attach(img)
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(GMAIL_ADDRESS, GMAIL_PASS)
                server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
            print(f"    Sent to {recipient}")
        except Exception as e: print(f"    Failed to send email: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 & 3: EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Phase 2 Engine started {ist_str()} ({utc_str()} UTC)")
    
    active_obs = load_json("active_obs.json", {})
    watch_state = load_json("active_watch_state.json", {})

    for pair_conf in config["pairs"]:
        symbol = pair_conf["symbol"]
        name = pair_conf["name"]
        entry_model = pair_conf.get("entry_model", "limit")
        
        pair_obs = active_obs.get(name, [])
        if not pair_obs: continue

        radar_interval = pair_conf.get("radar_tf", "15m")
        df_radar = clean_df(yf.download(symbol, period="5d", interval=radar_interval, progress=False))
        df_h1 = clean_df(yf.download(symbol, period="15d", interval="1h", progress=False))
        
        if df_radar is None or df_h1 is None or df_h1.empty: continue
        
        current_price = float(df_radar['Close'].iloc[-1])
        h1_atr = get_atr(df_h1)
        if not h1_atr: continue
        
        warning_dist = pair_conf["atr_multiplier"] * h1_atr

        for ob in pair_obs:
            ob_proximal = float(ob['proximal_line'])
            distance = abs(current_price - ob_proximal)
            
            if distance <= warning_dist:
                bias = "LONG" if "Demand" in ob['direction'] else "SHORT"
                fvg_data = ob.get("fvg", {"exists": False}) 
                
                # Phase 2: Math & Scorecard (smc_detector v3.0)
                result = smc_detector.run_phase2_analysis(pair_conf, bias, df_h1, df_radar, ob, fvg_data, current_price, 0)
                
                if not result['send_alert']:
                    print(f"  [X] {name} Aborted: {result.get('confidence_reason')}")
                    continue

                if entry_model == "limit":
                    levels = {'entry': result['entry'], 'sl': result['sl'], 'tp1': result['tp1'], 'tp2': result['tp2']}
                    chart1 = generate_chart(df_radar, f"{name} — {radar_interval}", levels, result, pair_conf)
                    html = build_trade_ready_email_html(result, name, pair_conf, current_price, chart1)
                    
                    print(f"  [✓] TRADE READY (FOREX): {name} | Entry: {result['entry']} | R:R: {result['rr_tp1']}")
                    send_email(f"TRADE READY | {name} | {'SELL' if bias=='SHORT' else 'BUY'}", html, chart1)
                
                elif entry_model == "ltf_choch":
                    watch_state[f"{name}_{ob_proximal}"] = {"pair": name, "bias": bias, "score": result['confidence_score'], "levels": result}
                    print(f"  [>] LOGGED FOR PHASE 3 (CHoCH): {name} approaching {ob_proximal}")

    save_json("active_watch_state.json", watch_state)
    print("Phase 2 complete.")
