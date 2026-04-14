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

def load_phase2_sent():
    try:
        with open("phase2_sent.json") as f: return json.load(f)
    except: return {}

def save_phase2_sent(data):
    with open("phase2_sent.json", "w") as f: json.dump(data, f, indent=2)
with open("config.json") as f: config = json.load(f)

GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "dummy")
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "dummy@gmail.com")
GMAIL_PASS    = os.environ.get("GMAIL_APP_PASSWORD", "dummy")

def get_ist_now(): return datetime.utcnow() + timedelta(hours=5, minutes=30)
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
        import numpy as np
        trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(closes))]
        if len(trs) < period: return None
        return float(np.mean(trs[-period:]))
    except Exception: return None

def fetch_macro_news(pair_name):
    try:
        r = requests.get("https://www.forexlive.com/feed/news", timeout=10)
        headlines = [f"- {item.find('title').text}" for item in ET.fromstring(r.content).findall('.//item')[:10]]
        return "\n".join(headlines)
    except Exception: return "Could not fetch latest news."

def call_gemini_flash(pair, bias, news_headlines):
    prompt = f"""
    You are a strict Risk Management AI. DO NOT analyze the chart. DO NOT calculate math.
    TRADE DETAILS: Pair: {pair} | Direction: {bias}
    RECENT NEWS: {news_headlines}
    
    TASK:
    1. Identify any Tier-1 economic events (e.g., CPI, NFP) affecting {pair}.
    2. Assign a macro_score: 1.0 if safe, 0.0 if a high-impact event is imminent.
    
    OUTPUT FORMAT (Strict JSON):
    {{
        "high_impact_news_detected": boolean,
        "macro_score": float,
        "macro_summary": "Exactly 2 concise sentences summarizing the risk specific to {pair}."
    }}
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0.15}}
    for _ in range(3):
        try:
            r = requests.post(url, json=body, timeout=20).json()
            if "candidates" in r: return json.loads(r["candidates"][0]["content"]["parts"][0]["text"].strip())
        except Exception: time.sleep(3)
    return {"macro_score": 1.0, "macro_summary": "Gemini API failed. Defaulting to safe."}

def generate_chart(df, title, levels, ob, pair_conf, fvg_data, sweep_price):
    try:
        dp = pair_conf.get("decimal_places", 5)
        # 1. STRIP NaNs HERE TO PREVENT CRASHES ON NQ=F / GC=F
        df_plot = df.dropna(subset=['Open', 'High', 'Low', 'Close']).tail(60).copy().reset_index(drop=True)
        fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), facecolor='#131722')
        ax.set_facecolor('#131722')
        for s in ax.spines.values(): s.set_color('#2a2a3e')

        for i, row in df_plot.iterrows():
            o, h, l, c = float(row['Open']), float(row['High']), float(row['Low']), float(row['Close'])
            # 2. DOUBLE CHECK FOR NaNs HERE
            import numpy as np
            if any(np.isnan(v) for v in [o, h, l, c]): continue
            col_c = '#26a69a' if c >= o else '#ef5350'
            ax.plot([i,i], [l,h], color=col_c, linewidth=1.5, zorder=2)
            body = abs(c-o) or (h-l) * 0.02
            ax.add_patch(patches.Rectangle((i-0.4, min(o,c)), 0.8, body, facecolor=col_c, linewidth=0, alpha=0.95, zorder=3))

        n = len(df_plot)
        proximal, distal = float(ob.get('proximal_line', 0)), float(ob.get('distal_line', 0))
        if proximal > 0 and distal > 0:
            ax.add_patch(patches.Rectangle((0, min(proximal, distal)), n+5, abs(proximal-distal), fill=False, edgecolor='#aaaaaa', linestyle=':', linewidth=1.5, zorder=2))
            ax.add_patch(patches.Rectangle((0, min(proximal, distal)), n+5, abs(proximal-distal), facecolor='#9b59b6', alpha=0.15, zorder=1))

        if fvg_data and fvg_data.get('exists'):
            ft, fb = float(fvg_data.get('fvg_top', 0)), float(fvg_data.get('fvg_bottom', 0))
            ax.add_patch(patches.Rectangle((0, fb), n+5, ft - fb, facecolor='#2ecc71', alpha=0.15, zorder=1))

        if sweep_price: ax.scatter([n-2], [sweep_price], color='red', marker='x', s=100, linewidth=2, zorder=5)

        for key, (color, lbl) in {'tp1': ('#27ae60', 'TP1'), 'entry': ('#e67e22', 'ENTRY'), 'sl': ('#e74c3c', 'SL')}.items():
            price = float(levels.get(key, 0))
            if price > 0:
                ax.axhline(y=price, color=color, linestyle='-', linewidth=1.5, alpha=0.9, zorder=4)
                ax.text(n + 1, price, f" {lbl}", color=color, fontsize=9, va='center', fontweight='bold', zorder=5)

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
    except Exception: return None

def build_trade_email(data, pair, pair_conf, chart_b64, state_msg="TRADE READY"):
    dp = pair_conf.get("decimal_places", 5)
    bias = data.get("bias", "—")
    ist_time = get_ist_now().strftime('%H:%M IST')
    
    explanation = f"<b>Logic Translation:</b> A {bias} Order Block was identified and structurally graded ({data.get('score')}/10). The entry model targets the {data['levels'].get('entry_source')} to mathematically guarantee a 1.5+ R:R to TP1. Stop loss is strictly placed beyond the distal line wick."

    return f"""<html><body style="font-family:Arial;background:#f0f2f5;padding:12px;">
    <div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;">
        <div style="background:#1a1a2e;padding:16px 20px;">
            <h2 style="color:white;margin:0;">{state_msg}: {pair}</h2>
            <p style="color:#8899bb;margin:4px 0 0;font-size:11px;">{ist_time}</p>
        </div>
        <div style="padding:16px 20px;">
            <div style="background:#27ae60;padding:16px 20px;border-radius:10px;margin-bottom:16px;">
                <p style="color:white;font-size:16px;font-weight:bold;margin:0;">{'SELL' if bias=='SHORT' else 'BUY'} LIMIT at {data['levels'].get('entry'):,.{dp}f}</p>
                <p style="color:white;margin:4px 0 0;">SL: {data['levels'].get('sl'):,.{dp}f} &nbsp;|&nbsp; TP1: {data['levels'].get('tp1'):,.{dp}f}</p>
            </div>
            <p style="font-size:13px;background:#f8f9fa;padding:10px;border-left:4px solid #27ae60;">{explanation}</p>
            <img src="cid:chart_m15" style="width:100%;border-radius:8px;margin-bottom:12px;" />
            <p><b>Macro Context:</b> {data.get('macro_summary')}</p>
        </div>
    </div></body></html>"""

def send_email(subject, html_body, chart_b64):
    for recipient in config["account"].get("alert_emails", []):
        msg = MIMEMultipart("related")
        msg["Subject"], msg["From"], msg["To"] = subject, GMAIL_ADDRESS, recipient
        msg.attach(MIMEText(html_body, "html"))
        if chart_b64:
            img = MIMEImage(base64.b64decode(chart_b64))
            img.add_header("Content-ID", "<chart_m15>")
            img.add_header("Content-Disposition", "inline", filename="chart_m15.png") # ADD THIS LINE
            msg.attach(img)
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(GMAIL_ADDRESS, GMAIL_PASS)
                server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
        except Exception: pass

if __name__ == "__main__":
    print(f"Phase 2 Engine started {get_ist_now().strftime('%H:%M')} IST")
    active_obs = load_json("active_obs.json", {})
    watch_state = load_json("active_watch_state.json", {})
    new_watch_state = {}

    # --- PHASE 2: H1 to M15 Cascade ---
    for pair_conf in config["pairs"]:
        symbol, name, entry_model = pair_conf["symbol"], pair_conf["name"], pair_conf.get("entry_model", "limit")
        pair_obs = active_obs.get(name, [])
        if not pair_obs: continue

        df_trigger = clean_df(yf.download(symbol, period="5d", interval=pair_conf.get("trigger_tf", "15m"), progress=False))
        df_h1 = clean_df(yf.download(symbol, period="15d", interval="1h", progress=False))
        if df_trigger is None or df_h1 is None: continue
        
        current_price = float(df_trigger['Close'].iloc[-1])
        h1_atr = get_atr(df_h1)
        if not h1_atr: continue
        
        for ob in pair_obs:
            if abs(current_price - float(ob['proximal_line'])) <= (pair_conf["atr_multiplier"] * h1_atr):
                bias = "LONG" if ob['direction'] == 'bullish' else "SHORT"
                fvg_data = ob.get("fvg", {"exists": False})
                phase2_sent = load_phase2_sent()
                score_res = smc_detector.run_scorecard(bias, df_h1, ob, fvg_data, current_price)
                if score_res['total'] < pair_conf["min_confidence"]: continue
                    
                levels = smc_detector.compute_dynamic_levels(pair_conf, bias, ob, fvg_data, current_price, df_trigger)
                if not levels['valid']: continue

                gemini_risk = call_gemini_flash(name, bias, fetch_macro_news(name))
                trade_data = {"pair": name, "bias": bias, "score": round(score_res['total'] + gemini_risk['macro_score'], 1), "macro_summary": gemini_risk["macro_summary"], "levels": levels, "ob": ob}

                if entry_model == "limit":
                    chart = generate_chart(df_trigger, f"{name} M15 LIMIT", levels, ob, pair_conf, fvg_data, score_res.get('sweep_price'))
                    send_email(f"TRADE READY | {name} | {bias} | {get_ist_now().strftime('%H:%M IST')}", build_trade_email(trade_data, name, pair_conf, chart), chart)
                    print(f"  [✓] TRADE READY (FOREX): {name}")
                elif entry_model == "ltf_choch":
                    zone_id = f"{name}_{bias}_{ob['proximal_line']}"
                if zone_id in phase2_sent:
                print(f"  [—] Already sent: {zone_id}")
                continue
                # ... existing send logic ...
                phase2_sent[zone_id] = get_ist_now().isoformat()
                save_phase2_sent(phase2_sent)
                    watch_id = f"{name}_{ob['proximal_line']}"
                    if watch_id not in watch_state:
                        chart = generate_chart(df_trigger, f"{name} M15 APPROACHING", levels, ob, pair_conf, fvg_data, None)
                        send_email(f"APPROACHING | {name} | {bias}", build_trade_email(trade_data, name, pair_conf, chart, "APPROACHING"), chart)
                    new_watch_state[watch_id] = trade_data
                    print(f"  [>] LOGGED FOR PHASE 3: {name}")

    save_json("active_watch_state.json", new_watch_state)
    print("Execution complete.")
