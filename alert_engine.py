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

def generate_chart(df, title, levels, pair_conf):
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
        level_cfg = {'tp2': ('#1e8449', '--', 1.0, 'TP2'), 'tp1': ('#27ae60', '-', 1.5, 'TP1'), 'entry': ('#e67e22', '-', 1.5, 'ENTRY'), 'sl': ('#e74c3c', '-', 1.5, 'SL')}
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
        print(f"Chart error: {e}"); return None

def build_trade_email(data, pair, pair_conf, chart_b64):
    dp = pair_conf.get("decimal_places", 5)
    bias = data.get("bias", "—")
    entry_p = f"{float(data.get('entry', 0)):,.{dp}f}"
    sl_p = f"{float(data.get('sl', 0)):,.{dp}f}"
    tp1_p = f"{float(data.get('tp1', 0)):,.{dp}f}"
    tp2_p = f"{float(data.get('tp2', 0)):,.{dp}f}"
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
    <p>Score: {data.get('confidence_score')}/10 | Entry Model: {data.get('entry_source')}</p>
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
import xml.etree.ElementTree as ET # Required for parsing RSS news

def fetch_macro_news(pair_name):
    """
    Your original news fetcher. (Using a standard ForexLive/Investing RSS fallback here).
    If you have your specific old code, replace the URL/Parsing inside this function.
    """
    try:
        # Example using a generic high-impact forex news RSS feed
        url = "https://www.forexlive.com/feed/news" 
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)
        
        headlines = []
        # Grab the latest 10 news headlines
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
    
    TASK: Analyze headlines for high-impact, Tier-1 economic events (e.g., CPI, NFP, Central Bank) affecting the {pair} currencies. 
    
    OUTPUT FORMAT (Valid JSON ONLY):
    {{
        "high_impact_news_detected": true/false,
        "macro_score": <float 0.0 to 1.0. 1.0=Safe, 0.0=Extreme volatility imminent>,
        "macro_summary": "<Exactly 2 sentences summarizing the news risk.>"
    }}
    """

def call_gemini_flash(prompt, max_retries=2):
    """Executes Gemini 2.5 Flash strictly via JSON."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.15,  # Sweet spot for safe JSON + natural phrasing
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
if __name__ == "__main__":
    print(f"Phase 2 Engine started {datetime.utcnow().strftime('%H:%M')} UTC")
    
    active_obs = load_json("active_obs.json", {})
    watch_state = load_json("active_watch_state.json", {})

    for pair_conf in config["pairs"]:
        symbol, name, entry_model = pair_conf["symbol"], pair_conf["name"], pair_conf.get("entry_model", "limit")
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
            if abs(current_price - ob_proximal) <= warning_dist:
                bias = "LONG" if ob['direction'] == 'bullish' else "SHORT"
                
                result = smc_detector.run_phase2_analysis(pair_conf, bias, df_h1, df_radar, ob, current_price, 0)
                
                if not result['send_alert']:
                    print(f"  [X] {name} Aborted: {result.get('confidence_reason')}")
                    continue

                if entry_model == "limit":
                    levels = {'entry': result['entry'], 'sl': result['sl'], 'tp1': result['tp1'], 'tp2': result['tp2']}
                    chart1 = generate_chart(df_radar, f"{name} — {radar_interval}", levels, pair_conf)
                    html = build_trade_email(result, name, pair_conf, chart1)
                    
                    print(f"  [✓] TRADE READY (FOREX): {name} | Entry: {result['entry']} | R:R: {result['rr_tp1']}")
                    send_email(f"TRADE READY | {name} | {'SELL' if bias=='SHORT' else 'BUY'}", html, chart1)
                
                elif entry_model == "ltf_choch":
                    watch_state[f"{name}_{ob_proximal}"] = {"pair": name, "bias": bias, "score": result['confidence_score'], "levels": result}
                    print(f"  [>] LOGGED FOR PHASE 3 (CHoCH): {name} approaching {ob_proximal}")
# ... (Inside your Phase 2 for-loop over the active zones) ...
            
            # Step 3: Risk Math & Entry Cascade
            levels = smc_detector.compute_dynamic_levels(pair_conf, bias, ob, fvg_data, current_price, df_trigger)
            
            if not levels['valid']:
                print(f"  [X] {name} {levels['reason']}. Aborted.")
                continue

            # ─────────────────────────────────────────────────────────────
            # STEP 4: GEMINI MACRO ANALYSIS (Paste this block here)
            # ─────────────────────────────────────────────────────────────
            print(f"  [*] Fetching Macro News & Calling Gemini for {name}...")
            news = fetch_macro_news(name)
            gemini_prompt = build_gemini_prompt(name, bias, news)
            gemini_risk = call_gemini_flash(gemini_prompt)
            
            # Failsafe if API crashes, so your trade isn't missed
            if not gemini_risk:
                gemini_risk = {
                    "high_impact_news_detected": False,
                    "macro_score": 1.0,
                    "macro_summary": "Gemini API failed. Defaulting to safe macro conditions."
                }
            
            # Apply Gemini's score to your Python Scorecard
            final_score = round(score_res['total'] + gemini_risk['macro_score'], 1)
            # ─────────────────────────────────────────────────────────────

            trade_data = {
                "pair": name,
                "bias": bias,
                "score": final_score, # Updated with Gemini's score
                "macro_summary": gemini_risk["macro_summary"], # To be injected into your HTML email
                "levels": levels,
                "ob": ob
            }
    save_json("active_watch_state.json", watch_state)
    print("Phase 2 complete.")
