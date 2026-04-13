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
import base64
from io import BytesIO
import xml.etree.ElementTree as ET
import smc_detector

with open("config.json") as f: config = json.load(f)

GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "dummy")
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "dummy@gmail.com")
GMAIL_PASS    = os.environ.get("GMAIL_APP_PASSWORD", "dummy")

def load_json(path, default):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return default

def save_json(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=2)

def clean_df(df):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
    return df

def get_atr(df, period=14):
    try:
        highs, lows, closes = df['High'].values.astype(float), df['Low'].values.astype(float), df['Close'].values.astype(float)
        trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(closes))]
        if len(trs) < period: return None
        import numpy as np
        return float(np.mean(trs[-period:]))
    except Exception: return None

def fetch_macro_news(pair_name):
    try:
        url = "https://www.forexlive.com/feed/news" 
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)
        headlines = []
        for item in root.findall('.//item')[:10]:
            title = item.find('title').text
            headlines.append(f"- {title}")
        return "\n".join(headlines)
    except Exception as e:
        print(f"News fetch failed: {e}")
        return "Could not fetch latest news."

def build_gemini_prompt(pair, bias, news_headlines):
    return f"""
    You are a strict Risk Management AI for an algorithmic trading desk.
    DO NOT analyze the chart. DO NOT calculate math. DO NOT offer trading advice.
    
    TRADE DETAILS: Pair: {pair} | Direction: {bias}
    
    RECENT NEWS HEADLINES (Last 24 Hours):
    {news_headlines}
    
    TASK: Analyze headlines for high-impact, Tier-1 economic events affecting the {pair} currencies. 
    
    OUTPUT FORMAT (Valid JSON ONLY):
    {{
        "high_impact_news_detected": true/false,
        "macro_score": 1.0,
        "macro_summary": "<Exactly 2 sentences summarizing the news risk.>"
    }}
    """

def call_gemini_flash(prompt, max_retries=2):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.15, 
            "maxOutputTokens": 250
        }
    }
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, json=body, timeout=20)
            result = r.json()
            if "candidates" not in result:
                if attempt < max_retries:
                    time.sleep(5)
                    continue
                return None
            raw_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            return json.loads(raw_text)
        except Exception:
            if attempt < max_retries:
                time.sleep(3)
                continue
    return None

def generate_chart(df, title, levels, ob, pair_conf):
    try:
        if df is None or df.empty: return None
        dp = pair_conf.get("decimal_places", 5)
        df_plot = df.tail(60).copy().reset_index(drop=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), facecolor='#131722')
        ax.set_facecolor('#131722')
        for s in ax.spines.values(): s.set_color('#2a2a3e')
        ax.grid(False)

        for i, row in df_plot.iterrows():
            o, h, l, c = float(row['Open']), float(row['High']), float(row['Low']), float(row['Close'])
            col_c = '#26a69a' if c >= o else '#ef5350'
            ax.plot([i,i], [l,h], color=col_c, linewidth=1.5, zorder=2)
            body = abs(c-o) or (h-l) * 0.02
            ax.add_patch(patches.Rectangle((i-0.4), min(o,c), 0.8, body, facecolor=col_c, linewidth=0, alpha=0.95, zorder=3))

        n = len(df_plot)
        ob_top, ob_bottom = float(ob.get('ob_high', 0)), float(ob.get('ob_low', 0))
        if ob_top > 0 and ob_bottom > 0:
            ob_color = '#9b59b6' # Light Purple for OB
            ax.add_patch(patches.Rectangle((0, ob_bottom), n+5, ob_top - ob_bottom, facecolor=ob_color, alpha=0.15, zorder=1))
            
        level_cfg = {'tp1': ('#27ae60', '-', 1.5, 'TP1'), 'entry': ('#e67e22', '-', 1.5, 'ENTRY'), 'sl': ('#e74c3c', '-', 1.5, 'SL')}
        for key, (color, style, width, lbl) in level_cfg.items():
            price = float(levels.get(key, 0))
            if price > 0:
                ax.axhline(y=price, color=color, linestyle=style, linewidth=width, alpha=0.9, zorder=4)
                ax.text(n + 1, price, f" {lbl}: {price:,.{dp}f}", color=color, fontsize=9, va='center', fontweight='bold', zorder=5)

        # Clamp Y-Axis to hide TP2 and make candles large
        sl_price = float(levels.get('sl', 0))
        tp1_price = float(levels.get('tp1', 0))
        if sl_price > 0 and tp1_price > 0:
            padding = abs(tp1_price - sl_price) * 0.2
            ax.set_ylim(min(sl_price, tp1_price) - padding, max(sl_price, tp1_price) + padding)

        ax.set_title(title, color='#dddddd', fontsize=12, pad=10, fontweight='bold', loc='left')
        ax.tick_params(colors='#888', labelsize=9)
        ax.yaxis.tick_right()
        ax.set_xlim(-1, n + 12)
        plt.tight_layout(pad=0.5)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#131722', edgecolor='none')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return b64
    except Exception as e:
        print(f"Chart error: {e}"); return None

def build_trade_email(data, pair, pair_conf, chart_b64):
    dp = pair_conf.get("decimal_places", 5)
    bias = data.get("bias", "—")
    entry_p = f"{float(data['levels'].get('entry', 0)):,.{dp}f}"
    sl_p = f"{float(data['levels'].get('sl', 0)):,.{dp}f}"
    tp1_p = f"{float(data['levels'].get('tp1', 0)):,.{dp}f}"
    tp2_p = f"{float(data['levels'].get('tp2', 0)):,.{dp}f}"
    chart_html = f'<img src="cid:chart_m15" style="width:100%;border-radius:8px;display:block;margin-bottom:12px;" />' if chart_b64 else ''

    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:12px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;">
  <div style="background:#1a1a2e;padding:16px 20px;"><h2 style="color:white;margin:0;">TRADE READY: {pair}</h2></div>
  <div style="padding:16px 20px;">
    <div style="background:#27ae60;padding:16px 20px;border-radius:10px;margin-bottom:16px;">
      <p style="color:white;font-size:16px;font-weight:bold;margin:0;">{'SELL' if bias=='SHORT' else 'BUY'} LIMIT at {entry_p}</p>
      <p style="color:white;margin:4px 0 0;">SL: {sl_p} &nbsp;|&nbsp; TP1: {tp1_p} &nbsp;|&nbsp; TP2: {tp2_p}</p>
    </div>
    {chart_html}
    <p><b>Final Score:</b> {data.get('score')}/10 | <b>Model:</b> {data['levels'].get('entry_source')}</p>
    <p><b>Macro Context:</b> {data.get('macro_summary')}</p>
  </div></div></body></html>"""

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

if __name__ == "__main__":
    print(f"Phase 2 Engine started {datetime.utcnow().strftime('%H:%M')} UTC")
    
    active_obs = load_json("active_obs.json", {})
    watch_state = load_json("active_watch_state.json", {})

    for pair_conf in config["pairs"]:
        symbol, name, entry_model = pair_conf["symbol"], pair_conf["name"], pair_conf.get("entry_model", "limit")
        pair_obs = active_obs.get(name, [])
        if not pair_obs: continue

        radar_interval = pair_conf.get("trigger_tf", "15m")
        df_trigger = clean_df(yf.download(symbol, period="5d", interval=radar_interval, progress=False))
        df_h1 = clean_df(yf.download(symbol, period="15d", interval="1h", progress=False))
        
        if df_trigger is None or df_h1 is None or df_h1.empty: continue
        
        current_price = float(df_trigger['Close'].iloc[-1])
        h1_atr = get_atr(df_h1)
        if not h1_atr: continue
        
        warning_dist = pair_conf["atr_multiplier"] * h1_atr

        for ob in pair_obs:
            ob_proximal = float(ob['proximal_line'])
            if abs(current_price - ob_proximal) <= warning_dist:
                bias = "LONG" if "Demand" in ob['direction'] else "SHORT"
                fvg_data = ob.get("fvg", {"exists": False})
                
                score_res = smc_detector.run_scorecard(bias, df_h1, ob, fvg_data, current_price)
                if score_res['total'] < pair_conf["min_confidence"]:
                    print(f"  [X] {name} Score {score_res['total']} < 7.0. Aborted.")
                    continue
                    
                levels = smc_detector.compute_dynamic_levels(pair_conf, bias, ob, fvg_data, current_price, df_trigger)
                if not levels['valid']:
                    print(f"  [X] {name} {levels['reason']}. Aborted.")
                    continue

                # --- GEMINI EXECUTION ---
                print(f"  [*] Fetching Macro News & Calling Gemini for {name}...")
                news = fetch_macro_news(name)
                gemini_prompt = build_gemini_prompt(name, bias, news)
                gemini_risk = call_gemini_flash(gemini_prompt)
                
                if not gemini_risk:
                    gemini_risk = {"macro_score": 1.0, "macro_summary": "Gemini API failed. Defaulting to safe."}
                
                final_score = round(score_res['total'] + gemini_risk['macro_score'], 1)

                trade_data = {
                    "pair": name,
                    "bias": bias,
                    "score": final_score,
                    "macro_summary": gemini_risk["macro_summary"],
                    "levels": levels,
                    "ob": ob
                }

                if entry_model == "limit":
                    chart1 = generate_chart(df_trigger, f"{name} — {radar_interval}", levels, ob, pair_conf)
                    html = build_trade_email(trade_data, name, pair_conf, chart1)
                    print(f"  [✓] TRADE READY (FOREX): {name} | Entry: {levels['entry']} | R:R: {levels['rr']}")
                    send_email(f"TRADE READY | {name} | {'SELL' if bias=='SHORT' else 'BUY'}", html, chart1)
                
                elif entry_model == "ltf_choch":
                    watch_state[f"{name}_{ob_proximal}"] = trade_data
                    print(f"  [>] LOGGED FOR PHASE 3 (CHoCH): {name} approaching {ob_proximal}")

    save_json("active_watch_state.json", watch_state)
    print("Phase 2 complete.")
