import yfinance as yf
import pandas as pd
import json, os, smtplib, requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import base64
from io import BytesIO

# ── Load config ───────────────────────────────────────────────
with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY    = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_PASS    = os.environ["GMAIL_APP_PASSWORD"]
ALERT_EMAIL   = os.environ["ALERT_EMAIL"]

# ── Alert log ─────────────────────────────────────────────────
ALERT_LOG_FILE = "alert_log.json"
try:
    with open(ALERT_LOG_FILE) as f:
        alert_log = json.load(f)
except:
    alert_log = []

# ── Cooldown ──────────────────────────────────────────────────
COOLDOWN_FILE = "cooldown_state.json"
try:
    with open(COOLDOWN_FILE) as f:
        cooldown_state = json.load(f)
except:
    cooldown_state = {}

def is_on_cooldown(pair, zone_level):
    key = f"{pair}_{round(zone_level, 4)}"
    if key in cooldown_state:
        fired_at = datetime.fromisoformat(cooldown_state[key])
        if datetime.utcnow() - fired_at < timedelta(hours=config["zone_detection"]["cooldown_hours"]):
            return True
    return False

def set_cooldown(pair, zone_level):
    key = f"{pair}_{round(zone_level, 4)}"
    cooldown_state[key] = datetime.utcnow().isoformat()
    with open(COOLDOWN_FILE, "w") as f:
        json.dump(cooldown_state, f)

# ── Market hours (IST) ────────────────────────────────────────
def is_market_open():
    ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    wd, h, m = ist_now.weekday(), ist_now.hour, ist_now.minute
    if wd == 5: return False, "Saturday IST — closed."
    if wd == 6: return False, "Sunday IST — closed."
    if wd == 0 and (h < 2 or (h == 2 and m < 30)): return False, "Monday before 2:30 AM IST."
    if wd == 4 and h >= 23 and m >= 30: return False, "Friday after 11:30 PM IST."
    return True, f"Open — {ist_now.strftime('%A %H:%M')} IST"

# ── Clean yfinance DataFrame ──────────────────────────────────
def clean_df(df):
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

# ── Zone detection + candle fetch ─────────────────────────────
def detect_zones_and_candles(symbol, min_touches, style):
    if style == "swing":
        tf1, tf2, p1, p2 = "4h", "1h", "60d", "15d"
        tf1_label, tf2_label = "H4", "H1"
    else:
        tf1, tf2, p1, p2 = "1h", "15m", "15d", "5d"
        tf1_label, tf2_label = "H1", "M15"

    df1 = clean_df(yf.download(symbol, period=p1, interval=tf1, progress=False))
    df2 = clean_df(yf.download(symbol, period=p2, interval=tf2, progress=False))

    if df1 is None or df1.empty:
        return [], None, None, None, tf1_label, tf2_label

    current_price = float(df1['Close'].iloc[-1])
    lb = config["zone_detection"]["swing_lookback"]
    highs = df1['High'].values.flatten()
    lows  = df1['Low'].values.flatten()

    swing_points = []
    for i in range(lb, len(highs) - lb):
        if highs[i] == max(highs[i-lb:i+lb+1]):
            swing_points.append(float(highs[i]))
        if lows[i] == min(lows[i-lb:i+lb+1]):
            swing_points.append(float(lows[i]))

    if not swing_points:
        return [], current_price, df1, df2, tf1_label, tf2_label

    swing_points = sorted(swing_points)
    clusters = [[swing_points[0]]]
    for lvl in swing_points[1:]:
        if (lvl - clusters[-1][-1]) / clusters[-1][-1] * 100 < 0.3:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    zones = [(float(np.mean(c)), len(c)) for c in clusters if len(c) >= min_touches]
    return zones, current_price, df1, df2, tf1_label, tf2_label

def get_zone_label(zone_level, current_price):
    return "Demand / Support" if zone_level < current_price else "Supply / Resistance"

# ── Macro news ────────────────────────────────────────────────
def fetch_macro_news():
    try:
        url = "https://api.rss2json.com/v1/api.json?rss_url=https://www.fxstreet.com/rss/news&count=5"
        r = requests.get(url, timeout=10)
        items = r.json().get("items", [])
        return "\n".join([f"- {i['title']}" for i in items])
    except:
        return "Macro news unavailable."

# ── Format candles for Gemini ─────────────────────────────────
def format_candles(df, label, n=20):
    if df is None or df.empty:
        return f"{label}: No data\n"
    result = f"{label} (last {n} candles):\n"
    for i in range(max(0, len(df)-n), len(df)):
        try:
            ts = df.index[i]
            ts_str = ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts, 'strftime') else str(ts)[:16]
            o = float(df['Open'].iloc[i])
            h = float(df['High'].iloc[i])
            l = float(df['Low'].iloc[i])
            c = float(df['Close'].iloc[i])
            result += f"{ts_str} O:{o:.5f} H:{h:.5f} L:{l:.5f} C:{c:.5f}\n"
        except:
            pass
    return result

# ── Build Gemini prompt ───────────────────────────────────────
def build_prompt(pair, zone_level, zone_label, current_price,
                 macro_news, df1, df2, tf1_label, tf2_label, min_confidence):
    risk_dollar = config["account"]["balance"] * (config["account"]["risk_percent"] / 100)
    ist_time = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")
    utc_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    candles1 = format_candles(df1, tf1_label)
    candles2 = format_candles(df2, tf2_label)

    return f"""
You are a professional SMC trader. A zone alert has triggered.

ALERT:
- Pair: {pair}
- Zone: {zone_label} at {zone_level}
- Current Price: {current_price}
- Time: {utc_time} UTC | {ist_time} IST
- Account: ${config["account"]["balance"]} | Risk: {config["account"]["risk_percent"]}% = ${risk_dollar:.0f}

CANDLE DATA:
{candles1}
{candles2}

MACRO:
{macro_news}

SMC SCORECARD (score /10):
STRUCTURE (3pts): +1 H4/Daily BOS confirms trend | +1 Price in Premium/Discount | +1 CHoCH confirmed H1/M15
ZONE QUALITY (3pts): +1 Valid OB at zone | +1 FVG overlaps OB | +1 Zone fresh (≤2 tests)
LIQUIDITY (2pts): +1 Liquidity swept before zone | +1 Entry in Discount(long) or Premium(short)
RISK/MACRO (2pts): +1 R:R ≥ 2:1 | +1 No high-impact news next 2h

MIN CONFIDENCE FOR THIS PAIR: {min_confidence}/10
If score < {min_confidence}: set send_alert=false. If >= {min_confidence}: set send_alert=true.

LEVEL RULES — use ONLY candle data:
- SL: below OB wick (long) / above OB wick (short)
- Entry: inside OB or FVG range
- TP1: nearest swing high/low
- TP2: next major structure level
- Never guess. State if data is insufficient.

Return ONLY raw JSON, no markdown:
{{
  "send_alert": true,
  "confidence_score": 0,
  "confidence_reason": "one sentence",
  "news_flag": "none or event description",
  "bias": "LONG or SHORT or WAIT",
  "bias_reason": "max 12 words",
  "confluences": ["item 1", "item 2", "item 3"],
  "missing": [{{"item": "name", "reason": "plain English why this matters"}}],
  "entry": "price or range from candle data",
  "sl": 0.0,
  "tp1": 0.0,
  "tp2": 0.0,
  "rr_tp1": "x.x",
  "rr_tp2": "x.x",
  "lot_size": "x.x",
  "sl_pts": 0,
  "trigger": "exact M15/H1 price action before entry",
  "invalid_if": "exact condition that kills trade",
  "macro_line1": "main macro driver for {pair}",
  "macro_line2": "key upcoming event for {pair}",
  "mindset": "one sharp psychological reminder for this setup",
  "ob_top": 0.0,
  "ob_bottom": 0.0,
  "ob_type": "bullish or bearish",
  "ob_confirmed": true,
  "fvg_top": 0.0,
  "fvg_bottom": 0.0,
  "fvg_type": "bullish or bearish",
  "fvg_confirmed": true,
  "chart_annotations": [
    {{"label": "short label", "price": 0.0, "status": "confirmed or missing"}}
  ]
}}"""

# ── Generate candlestick chart ────────────────────────────────
def generate_chart(df, title, levels, data):
    try:
        df_plot = df.tail(60).copy().reset_index(drop=True)
        required = ['Open','High','Low','Close']
        for col in required:
            if col not in df_plot.columns:
                return None

        fig = plt.figure(figsize=(13, 7), facecolor='#131722')
        gs = GridSpec(4, 1, figure=fig, hspace=0.04)
        ax = fig.add_subplot(gs[:3, 0])
        ax_vol = fig.add_subplot(gs[3, 0], sharex=ax)

        for a in [ax, ax_vol]:
            a.set_facecolor('#131722')
            for s in a.spines.values():
                s.set_color('#2a2a3e')

        # Candlesticks
        for i, row in df_plot.iterrows():
            try:
                o = float(row['Open']); h = float(row['High'])
                l = float(row['Low']);  c = float(row['Close'])
                if any(np.isnan(v) for v in [o,h,l,c]):
                    continue
                color = '#26a69a' if c >= o else '#ef5350'
                ax.plot([i,i], [l,h], color=color, linewidth=0.8, zorder=2)
                body = abs(c-o) or (h-l)*0.01
                ax.add_patch(patches.Rectangle((i-0.35, min(o,c)), 0.7, body,
                             facecolor=color, linewidth=0, alpha=0.9, zorder=3))
            except:
                continue

        n = len(df_plot)

        # OB zone
        ob_top    = float(data.get('ob_top', 0) or 0)
        ob_bottom = float(data.get('ob_bottom', 0) or 0)
        if ob_top > 0 and ob_bottom > 0 and abs(ob_top - ob_bottom) > 0:
            ob_color = '#26a69a' if data.get('ob_type','') == 'bullish' else '#ef5350'
            confirmed = data.get('ob_confirmed', True)
            ax.add_patch(patches.Rectangle((0, ob_bottom), n, ob_top-ob_bottom,
                         facecolor=ob_color, edgecolor=ob_color,
                         linewidth=1, alpha=0.2 if confirmed else 0.08,
                         linestyle='-' if confirmed else '--', zorder=1))
            ax.text(1, ob_top, f" OB {'✓' if confirmed else '⚠'}",
                    color=ob_color, fontsize=8, va='bottom', fontweight='bold', zorder=5)

        # FVG zone
        fvg_top    = float(data.get('fvg_top', 0) or 0)
        fvg_bottom = float(data.get('fvg_bottom', 0) or 0)
        if fvg_top > 0 and fvg_bottom > 0 and abs(fvg_top - fvg_bottom) > 0:
            fvg_confirmed = data.get('fvg_confirmed', True)
            ax.add_patch(patches.Rectangle((0, fvg_bottom), n, fvg_top-fvg_bottom,
                         facecolor='#3498db', edgecolor='#3498db',
                         linewidth=1, alpha=0.18 if fvg_confirmed else 0.07,
                         linestyle='-' if fvg_confirmed else '--', zorder=1))
            ax.text(1, fvg_top, f" FVG {'✓' if fvg_confirmed else '⚠'}",
                    color='#3498db', fontsize=8, va='bottom', fontweight='bold', zorder=5)

        # Horizontal levels
        level_cfg = {
            'tp2':     ('#1e8449', '--', 1.2, 'TP2'),
            'tp1':     ('#27ae60', '-',  1.8, 'TP1'),
            'entry':   ('#e67e22', '-',  1.8, 'Entry'),
            'zone':    ('#9b59b6', '--', 1.5, 'Zone'),
            'current': ('#ffffff', ':',  1.0, 'Now'),
            'sl':      ('#e74c3c', '-',  1.8, 'SL'),
        }
        for key, (color, style, width, lbl) in level_cfg.items():
            val = levels.get(key, 0)
            try:
                price = float(str(val).split('-')[0].strip()) if val else 0
            except:
                price = 0
            if price > 0:
                ax.axhline(y=price, color=color, linestyle=style,
                           linewidth=width, alpha=0.85, zorder=4)
                ax.text(n+0.3, price, f"{lbl}: {price:,.5f}",
                        color=color, fontsize=7.5, va='center',
                        fontweight='bold', zorder=5)

        # Confluence / missing annotations
        annotations = data.get('chart_annotations', [])
        placed_y = []
        all_prices = [v for k,v in levels.items()
                      if v and float(str(v).split('-')[0] or 0) > 0]
        price_range = (max(all_prices) - min(all_prices)) if len(all_prices) >= 2 else 1

        for ann in annotations:
            try:
                price = float(ann.get('price', 0) or 0)
            except:
                continue
            if price <= 0:
                continue
            status = ann.get('status', 'confirmed')
            color  = '#2ecc71' if status == 'confirmed' else '#f39c12'
            marker = '✓' if status == 'confirmed' else '⚠'
            label_text = f"{marker} {ann['label']}"

            # Avoid label overlap
            y_pos = price
            for py in placed_y:
                if abs(y_pos - py) < price_range * 0.012:
                    y_pos += price_range * 0.015
            placed_y.append(y_pos)

            ax.annotate(label_text, xy=(int(n*0.05), y_pos),
                        color=color, fontsize=7.5,
                        bbox=dict(boxstyle='round,pad=0.25',
                                  facecolor='#1a1a2e', edgecolor=color, alpha=0.85),
                        zorder=6)

        # Volume
        for i, row in df_plot.iterrows():
            try:
                vol = float(row.get('Volume', 0) or 0)
                if np.isnan(vol): vol = 0
                vc = '#26a69a' if float(row['Close']) >= float(row['Open']) else '#ef5350'
                ax_vol.bar(i, vol, color=vc, alpha=0.5, width=0.7)
            except:
                continue

        # Styling
        ax.set_title(title, color='#dddddd', fontsize=11, pad=8,
                     fontweight='bold', loc='left')
        ax.tick_params(colors='#666', labelsize=7.5)
        ax.yaxis.tick_right()
        ax.yaxis.set_tick_params(labelcolor='#888')
        ax.xaxis.set_visible(False)
        ax.set_xlim(-1, n+14)

        ax_vol.tick_params(colors='#555', labelsize=6)
        ax_vol.set_ylabel('Vol', color='#555', fontsize=7)
        ax_vol.yaxis.tick_right()
        ax_vol.xaxis.set_visible(False)
        ax_vol.set_xlim(-1, n+14)

        legend_items = [
            patches.Patch(facecolor='#26a69a', alpha=0.8, label='Bullish candle'),
            patches.Patch(facecolor='#ef5350', alpha=0.8, label='Bearish candle'),
            patches.Patch(facecolor='#3498db', alpha=0.4, label='FVG'),
            plt.Line2D([0],[0], color='#2ecc71', linewidth=1.5, label='✓ Confirmed'),
            plt.Line2D([0],[0], color='#f39c12', linewidth=1.5, label='⚠ Missing'),
        ]
        ax.legend(handles=legend_items, loc='upper left',
                  facecolor='#1a1a2e', edgecolor='#333',
                  labelcolor='white', fontsize=7, framealpha=0.9)

        plt.tight_layout(pad=0.3)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='#131722', edgecolor='none')
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return chart_b64

    except Exception as e:
        print(f"    Chart error: {e}")
        plt.close('all')
        return None

# ── Call Gemini ───────────────────────────────────────────────
def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, json=body, timeout=60)
        result = r.json()
        if "candidates" not in result:
            return None, f"Gemini error: {result}"
        raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw), None
    except Exception as e:
        return None, f"Gemini error: {str(e)}"

# ── Build HTML email ──────────────────────────────────────────
def build_html(data, pair, zone_level, zone_label, current_price,
               chart1_b64, chart2_b64, tf1_label, tf2_label):
    ist_time = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")
    utc_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    bias_color  = "#e74c3c" if data["bias"]=="SHORT" else "#27ae60" if data["bias"]=="LONG" else "#f39c12"
    score       = data.get("confidence_score", 0)
    score_color = "#27ae60" if score>=8 else "#f39c12" if score>=6 else "#e74c3c"

    news_flag   = data.get("news_flag","none")
    news_banner = f"""<div style="background:#fff3cd;padding:10px 24px;border-left:4px solid #f39c12;font-size:12px;color:#856404;">⚠️ <b>NEWS:</b> {news_flag}</div>""" if news_flag and news_flag.lower()!="none" else ""

    confluences_html = "".join([
        f"<li style='margin-bottom:6px;padding:7px 10px;background:#f0fff4;border-radius:6px;font-size:13px;'>✅ {c}</li>"
        for c in data.get("confluences",[])
    ])
    missing_html = "".join([
        f"<li style='margin-bottom:6px;padding:7px 10px;background:#fff8f0;border-radius:6px;font-size:13px;'>❌ <b>{m['item']}</b> — <span style='color:#777;font-style:italic;'>{m['reason']}</span></li>"
        for m in data.get("missing",[])
    ])

    # Price map
    try:
        ep = float(str(data["entry"]).split("-")[0].strip())
        lvls = {"TP2":float(data["tp2"]),"TP1":float(data["tp1"]),"Entry":ep,
                "Current":float(current_price),"Zone":float(zone_level),"SL":float(data["sl"])}
        lc   = {"SL":"#e74c3c","Zone":"#9b59b6","Current":"#3498db","Entry":"#e67e22","TP1":"#27ae60","TP2":"#1e8449"}
        vals = [v for v in lvls.values() if v>0]
        pmin, pmax = min(vals), max(vals)
        pr = pmax-pmin or 1
        rows = ""
        for lbl, price in sorted([(k,v) for k,v in lvls.items() if v>0], key=lambda x:x[1], reverse=True):
            c   = lc.get(lbl,"#888")
            bar = int(((price-pmin)/pr)*75)+15
            rows += f"""<tr><td style="padding:5px 10px;color:{c};font-weight:bold;font-size:12px;width:65px;">{lbl}</td><td style="padding:5px 6px;"><div style="background:{c};height:10px;border-radius:4px;width:{bar}%;"></div></td><td style="padding:5px 10px;color:{c};font-weight:bold;font-size:12px;text-align:right;white-space:nowrap;">{price:,.5f}</td></tr>"""
        price_map = f"""<h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">📊 PRICE MAP</h3><table style="width:100%;border-collapse:collapse;background:#f8f9fa;border-radius:8px;margin-bottom:20px;overflow:hidden;">{rows}</table>"""
    except:
        price_map = ""

    chart1_html = f"""<h3 style="color:#1a1a2e;font-size:13px;margin:20px 0 6px;">📈 {tf1_label} CHART</h3><img src="data:image/png;base64,{chart1_b64}" style="width:100%;border-radius:8px;margin-bottom:4px;" />""" if chart1_b64 else ""
    chart2_html = f"""<h3 style="color:#1a1a2e;font-size:13px;margin:16px 0 6px;">📈 {tf2_label} CHART</h3><img src="data:image/png;base64,{chart2_b64}" style="width:100%;border-radius:8px;" />""" if chart2_b64 else ""

    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:16px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">
<div style="background:#1a1a2e;padding:18px 24px;">
  <h2 style="color:white;margin:0;font-size:17px;">🔔 {pair} — {zone_label}</h2>
  <p style="color:#8899bb;margin:5px 0 0;font-size:12px;">{utc_time} UTC &nbsp;|&nbsp; {ist_time} IST</p>
</div>
<div style="background:{bias_color};padding:12px 24px;">
  <p style="color:white;margin:0;font-size:15px;font-weight:bold;">⚡ {data["bias"]} — {data["bias_reason"]}</p>
</div>
<div style="padding:10px 24px;background:#f8f9fa;border-bottom:1px solid #eee;">
  <span style="font-size:12px;color:#666;">Confidence: </span>
  <span style="font-size:18px;font-weight:bold;color:{score_color};">{score}/10</span>
  <span style="font-size:12px;color:#888;margin-left:8px;">— {data.get("confidence_reason","")}</span>
</div>
{news_banner}
<div style="padding:20px 24px;">
  <p style="background:#f4f4f8;padding:10px 14px;border-radius:8px;font-size:13px;color:#333;margin:0 0 16px;">
    📍 <b>{zone_label}</b> at <b>{zone_level}</b> &nbsp;|&nbsp; Now: <b>{current_price}</b>
  </p>
  {chart1_html}
  {chart2_html}
  {price_map}
  <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">🎯 TRADE</h3>
  <table style="width:100%;font-size:13px;margin-bottom:20px;">
    <tr><td style="padding:5px 0;color:#555;width:110px;">Entry</td><td style="padding:5px 0;font-weight:bold;">{data["entry"]}</td></tr>
    <tr><td style="padding:5px 0;color:#e74c3c;">Stop Loss</td><td style="padding:5px 0;font-weight:bold;color:#e74c3c;">{data["sl"]}</td></tr>
    <tr><td style="padding:5px 0;color:#27ae60;">TP1</td><td style="padding:5px 0;font-weight:bold;color:#27ae60;">{data["tp1"]} &nbsp;<span style="font-weight:normal;color:#888;">(R:R {data["rr_tp1"]})</span></td></tr>
    <tr><td style="padding:5px 0;color:#1e8449;">TP2</td><td style="padding:5px 0;font-weight:bold;color:#1e8449;">{data["tp2"]} &nbsp;<span style="font-weight:normal;color:#888;">(R:R {data["rr_tp2"]})</span></td></tr>
    <tr><td style="padding:5px 0;color:#555;">Lot Size</td><td style="padding:5px 0;font-weight:bold;">{data["lot_size"]} lots &nbsp;<span style="font-weight:normal;color:#888;">({data["sl_pts"]} pts risk)</span></td></tr>
  </table>
  <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">✅ CONFLUENCES</h3>
  <ul style="list-style:none;padding:0;margin:0 0 20px;">{confluences_html}</ul>
  <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">❌ MISSING</h3>
  <ul style="list-style:none;padding:0;margin:0 0 20px;">{missing_html}</ul>
  <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 6px;">⏳ TRIGGER</h3>
  <p style="font-size:13px;color:#333;background:#fffbea;padding:10px 14px;border-radius:8px;border-left:4px solid #f39c12;margin:0 0 20px;">{data["trigger"]}</p>
  <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 6px;">🛑 INVALID IF</h3>
  <p style="font-size:13px;color:#c0392b;background:#fef0f0;padding:10px 14px;border-radius:8px;border-left:4px solid #e74c3c;margin:0 0 20px;">{data["invalid_if"]}</p>
  <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 6px;">🌍 MACRO</h3>
  <p style="font-size:13px;color:#444;margin:0 0 20px;line-height:1.7;">{data["macro_line1"]}<br>{data["macro_line2"]}</p>
  <div style="background:#1a1a2e;padding:14px 18px;border-radius:10px;">
    <p style="color:#8899bb;font-size:10px;margin:0 0 4px;text-transform:uppercase;letter-spacing:1px;">🧠 Mindset</p>
    <p style="color:white;font-size:13px;margin:0;font-style:italic;line-height:1.6;">{data["mindset"]}</p>
  </div>
</div>
</div></body></html>"""

# ── Log alert ─────────────────────────────────────────────────
def log_alert(pair, zone_level, zone_label, current_price, data):
    entry = {
        "id": f"{pair}_{int(datetime.utcnow().timestamp())}",
        "pair": pair,
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "zone_level": zone_level,
        "zone_label": zone_label,
        "bias": data.get("bias",""),
        "entry": data.get("entry",""),
        "sl": data.get("sl", 0),
        "tp1": data.get("tp1", 0),
        "tp2": data.get("tp2", 0),
        "confidence_score": data.get("confidence_score", 0),
        "confluences": data.get("confluences", []),
        "outcome": "pending",
        "outcome_price": None,
        "outcome_checked_at": None
    }
    alert_log.append(entry)
    with open(ALERT_LOG_FILE, "w") as f:
        json.dump(alert_log, f, indent=2)

# ── Send email ────────────────────────────────────────────────
def send_email(subject, data, pair, zone_level, zone_label, current_price,
               chart1_b64, chart2_b64, tf1_label, tf2_label):
    html_body = build_html(data, pair, zone_level, zone_label, current_price,
                           chart1_b64, chart2_b64, tf1_label, tf2_label)
    recipients = config["account"].get("alert_emails", [ALERT_EMAIL])
    for recipient in recipients:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_ADDRESS
        msg["To"]      = recipient
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_PASS)
            server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
        print(f"    Email sent to {recipient}")

# ── Main ──────────────────────────────────────────────────────
print(f"Alert engine started {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

market_open, market_status = is_market_open()
print(f"  Market: {market_status}")
if not market_open:
    print("  Exiting — market closed.")
    exit(0)

macro_news  = fetch_macro_news()
alerts_fired = 0

for pair_conf in config["pairs"]:
    symbol      = pair_conf["symbol"]
    name        = pair_conf["name"]
    prox        = pair_conf["proximity_pct"]
    min_touches = pair_conf.get("min_touches", 2)
    min_conf    = pair_conf.get("min_confidence", 5)
    style       = pair_conf.get("style", "intraday")

    print(f"  Scanning {name} ({style})...")
    zones, current_price, df1, df2, tf1_label, tf2_label = detect_zones_and_candles(symbol, min_touches, style)

    if current_price is None:
        print(f"    No data for {name}. Skipping.")
        continue

    for zone_level, touches in zones:
        dist_pct = abs(current_price - zone_level) / zone_level * 100
        if dist_pct > prox:
            continue
        if is_on_cooldown(name, zone_level):
            print(f"    {name} @ {zone_level:.5f} — cooldown.")
            continue

        zone_label = get_zone_label(zone_level, current_price)
        print(f"    ZONE HIT: {name} {zone_label} at {zone_level:.5f}")

        prompt = build_prompt(name, round(zone_level,5), zone_label,
                              round(current_price,5), macro_news,
                              df1, df2, tf1_label, tf2_label, min_conf)
        data, error = call_gemini(prompt)

        if error:
            print(f"    Gemini error: {error}")
            continue

        score = data.get("confidence_score", 0)
        if not data.get("send_alert", False):
            print(f"    {name} skipped — {score}/10 below {min_conf}. {data.get('confidence_reason','')}")
            continue

        levels = {
            'zone': zone_level, 'current': current_price,
            'entry': data.get('entry',''), 'sl': data.get('sl',0),
            'tp1': data.get('tp1',0), 'tp2': data.get('tp2',0)
        }
        print(f"    Generating charts...")
        chart1 = generate_chart(df1, f"{name} — {tf1_label}", levels, data) if df1 is not None else None
        chart2 = generate_chart(df2, f"{name} — {tf2_label}", levels, data) if df2 is not None else None

        subject = f"[{score}/10] {name} | {zone_label} | {round(zone_level,5)} | {datetime.utcnow().strftime('%H:%M')} UTC"
        send_email(subject, data, name, round(zone_level,5), zone_label,
                   round(current_price,5), chart1, chart2, tf1_label, tf2_label)
        log_alert(name, round(zone_level,5), zone_label, round(current_price,5), data)
        set_cooldown(name, zone_level)
        alerts_fired += 1
        print(f"    ✅ Sent: {name} [{score}/10]")

print(f"Done. {alerts_fired} alert(s) fired.")
