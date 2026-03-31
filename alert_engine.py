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
from email.mime.image import MIMEImage
import numpy as np
import base64
from io import BytesIO

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY    = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_PASS    = os.environ["GMAIL_APP_PASSWORD"]
ALERT_EMAIL   = os.environ["ALERT_EMAIL"]

# ── Alert log ─────────────────────────────────────────────────────────────────
ALERT_LOG_FILE = "alert_log.json"
SCAN_LOG_FILE = "scan_log.json"
SYSTEM_STATUS_FILE = "system_status.json"
VISIT_FILE = "zone_visit_state.json"

def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
def save_alert_log():
    save_json(ALERT_LOG_FILE, alert_log)

alert_log = load_json(ALERT_LOG_FILE, [])
scan_log = load_json(SCAN_LOG_FILE, [])
system_status = load_json(SYSTEM_STATUS_FILE, {
    "last_ok_email_utc": None,
    "last_error_email_utc": None,
    "last_trade_alert_utc": None
})
visit_state = load_json(VISIT_FILE, {})

def utc_now():
    return datetime.utcnow()

def utc_str():
    return utc_now().strftime("%Y-%m-%d %H:%M")

def hours_since(ts):
    if not ts:
        return None
    try:
        t = datetime.strptime(ts, "%Y-%m-%d %H:%M")
        return (utc_now() - t).total_seconds() / 3600
    except:
        return None

def should_send_ok():
    last_ok = hours_since(system_status.get("last_ok_email_utc"))
    last_trade = hours_since(system_status.get("last_trade_alert_utc"))

    return (last_ok is None or last_ok >= 3) and (last_trade is None or last_trade >= 3)

def should_send_error():
    last_err = hours_since(system_status.get("last_error_email_utc"))
    return (last_err is None or last_err >= 3)

def log_scan(pair, status, reason, zone=None):
    scan_log.append({
        "time": utc_str(),
        "pair": pair,
        "zone": round(zone, 5) if zone is not None else None,
        "status": status,
        "reason": reason
    })
    save_json(SCAN_LOG_FILE, scan_log)

# ── Zone visit state ──────────────────────────────────────────────────────────
# No cooldown hours. No time tracking.
# A zone re-alerts ONLY when price has moved more than 1.5x proximity_pct
# away from zone since the last alert on that zone. That is the only rule.
def save_visit_state():
    save_json(VISIT_FILE, visit_state)

def should_alert_zone(pair, zone_level, current_price, proximity_pct):
    key = f"{pair}_{round(zone_level, 4)}"
    if key not in visit_state:
        return True
    last_price = float(visit_state[key].get("last_alert_price", current_price))
    dist_pct   = abs(current_price - last_price) / zone_level * 100
    return dist_pct > proximity_pct * 1.5

def record_zone_alert(pair, zone_level, current_price):
    key = f"{pair}_{round(zone_level, 4)}"
    visit_state[key] = {"last_alert_price": current_price}
    save_visit_state()

# ── Zone fatigue (for SMC scorecard deduction only) ───────────────────────────
def count_zone_alerts(pair, zone_level, days=30):
    cutoff = datetime.utcnow() - timedelta(days=days)
    count  = 0
    for a in alert_log:
        try:
            if a.get("alert_type", "zone") != "zone":
                continue
            if datetime.strptime(a["timestamp_utc"], "%Y-%m-%d %H:%M") < cutoff:
                continue
            if a["pair"] != pair:
                continue
            if abs(float(a.get("zone_level", 0)) - zone_level) / zone_level * 100 < 0.3:
                count += 1
        except:
            pass
    return count

# ── Market hours (IST) ────────────────────────────────────────────────────────
def is_market_open():
    ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    wd, h, m = ist.weekday(), ist.hour, ist.minute
    if wd == 5: return False, "Saturday — closed."
    if wd == 6: return False, "Sunday — closed."
    if h < 8: return False, f"Before 8:00 AM IST — {ist.strftime('%A %H:%M')} IST."
    if wd == 4 and h >= 23 and m >= 30: return False, "Friday after 11:30 PM IST."
    return True, f"Open — {ist.strftime('%A %H:%M')} IST"

# ── Data helpers ──────────────────────────────────────────────────────────────
def clean_df(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def get_atr(df, period=14):
    try:
        highs  = df['High'].values.flatten().astype(float)
        lows   = df['Low'].values.flatten().astype(float)
        closes = df['Close'].values.flatten().astype(float)
        trs = [max(highs[i]-lows[i],
                   abs(highs[i]-closes[i-1]),
                   abs(lows[i]-closes[i-1]))
               for i in range(1, len(closes))]
        if len(trs) < period:
            return None
        return float(np.mean(trs[-period:]))
    except:
        return None

# All pairs use intraday: H1 primary. M15 fetched only after zone is in proximity.
def detect_zones_and_candles(symbol, min_touches):
    df1 = clean_df(yf.download(symbol, period="15d", interval="1h", progress=False))

    if df1 is None:
        return [], None, None

    current_price = float(df1['Close'].iloc[-1])
    lb    = config["zone_detection"]["swing_lookback"]
    highs = df1['High'].values.flatten()
    lows  = df1['Low'].values.flatten()

    swing_points = []
    for i in range(lb, len(highs) - lb):
        if highs[i] == max(highs[i-lb:i+lb+1]):
            swing_points.append(float(highs[i]))
        if lows[i] == min(lows[i-lb:i+lb+1]):
            swing_points.append(float(lows[i]))

    if not swing_points:
        return [], current_price, df1

    swing_points = sorted(swing_points)
    clusters = [[swing_points[0]]]
    for lvl in swing_points[1:]:
        if (lvl - clusters[-1][-1]) / clusters[-1][-1] * 100 < 0.3:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    zones = [(float(np.mean(c)), len(c)) for c in clusters if len(c) >= min_touches]
    return zones, current_price, df1


def fetch_m15_data(symbol):
    return clean_df(yf.download(symbol, period="5d", interval="15m", progress=False))
def get_zone_label(zone_level, current_price):
    return "Demand / Support" if zone_level < current_price else "Supply / Resistance"
# ── Macro news ────────────────────────────────────────────────────────────────
def fetch_macro_news():
    """
    Two-source news fetch:
    1. FXStreet RSS — forex-specific macro (rate decisions, CPI, etc.)
    2. Google News RSS — geopolitical headlines (Iran, war, sanctions, tariffs, oil supply)
    Both sources are combined and passed to Gemini as context.
    """
    headlines = []

    # Source 1 — FXStreet (forex macro)
    try:
        url   = "https://api.rss2json.com/v1/api.json?rss_url=https://www.fxstreet.com/rss/news&count=5"
        r     = requests.get(url, timeout=10)
        items = r.json().get("items", [])
        for i in items:
            headlines.append(f"[FXStreet] {i['title']}")
    except Exception:
        headlines.append("[FXStreet] Unavailable")

    # Source 2 — Google News RSS (geopolitical: US-Iran, sanctions, oil, war, tariffs)
    geo_query = "Iran+war+OR+military+strike+OR+sanctions+OR+oil+supply+OR+tariff+OR+ceasefire"
    geo_url   = (f"https://news.google.com/rss/search?q={geo_query}"
                 f"&hl=en-US&gl=US&ceid=US:en")
    try:
        import xml.etree.ElementTree as ET
        r   = requests.get(geo_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.content)
        items = root.findall(".//item")[:5]
        for item in items:
            title = item.findtext("title", "").strip()
            if title:
                headlines.append(f"[GeoNews] {title}")
    except Exception:
        headlines.append("[GeoNews] Unavailable")

    return "\n".join(headlines) if headlines else "Macro news unavailable."

def format_candles(df, label, n=20):
    if df is None or df.empty:
        return f"{label}: No data\n"
    result = f"{label} (last {n} candles):\n"
    for i in range(max(0, len(df)-n), len(df)):
        try:
            ts  = df.index[i]
            tss = ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts,'strftime') else str(ts)[:16]
            result += (f"{tss} O:{float(df['Open'].iloc[i]):.5f} "
                       f"H:{float(df['High'].iloc[i]):.5f} "
                       f"L:{float(df['Low'].iloc[i]):.5f} "
                       f"C:{float(df['Close'].iloc[i]):.5f}\n")
        except:
            pass
    return result

# ── Gemini prompt ─────────────────────────────────────────────────────────────
def build_zone_prompt(pair, zone_level, zone_label, current_price,
                      macro_news, df1, df2, min_confidence, fatigue_count):
    risk_dollar = config["account"]["balance"] * config["account"]["risk_percent"] / 100
    ist_time    = (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M")
    utc_time    = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    if fatigue_count >= 5:
        fatigue_rule = (f"Zone alerted {fatigue_count} times in 30 days. HEAVILY FATIGUED. "
                        "Deduct 2 from Zone Quality score. Require extra confluence to pass.")
    elif fatigue_count >= 3:
        fatigue_rule = (f"Zone alerted {fatigue_count} times in 30 days. SHOWING FATIGUE. "
                        "Deduct 1 from Zone Quality score.")
    else:
        fatigue_rule = f"Zone alerted {fatigue_count} times in 30 days. FRESH. No deduction."

    return f"""You are a highly skilled professional SMC trader. A zone alert has triggered.

PAIR: {pair} | ZONE: {zone_label} at {zone_level} | PRICE NOW: {current_price}
TIME: {utc_time} UTC | {ist_time} IST
ACCOUNT: ${config["account"]["balance"]} | RISK: {config["account"]["risk_percent"]}% = ${risk_dollar:.0f} | FIRM: {config["account"]["firm"]}

CANDLE DATA:
{format_candles(df1, "H1")}
{format_candles(df2, "M15")}

MACRO: {macro_news}

SMC SCORECARD (score out of 10 — this is the ONLY scorecard):
STRUCTURE (3pts):     +1 H1 BOS confirms trend direction | +1 Price in Premium or Discount zone | +1 CHoCH confirmed on H1 or M15
ZONE QUALITY (3pts):  +1 Valid OB at zone | +1 FVG overlaps OB | +1 Zone freshness — rule: {fatigue_rule}
LIQUIDITY (2pts):     +1 Liquidity swept before zone | +1 Entry on correct side (Discount for long, Premium for short)
RISK/MACRO (2pts):    +1 RR at least 2:1 achievable | +1 No high-impact news in next 2 hours

ENTRY: 50pct midpoint of OB candle body. If FVG overlaps OB, use FVG edge (top for longs, bottom for shorts).
SL: OB wick extreme plus ATR buffer (3-5 pips forex, 10-15pts NAS100, 50-100pts BTC/Gold/Silver).
After trigger fires: refine SL to trigger candle extreme only if it gives a tighter SL than OB wick.

MIN CONFIDENCE TO SEND: {min_confidence}/10. Below this, set send_alert to false.

Return ONLY raw JSON. No markdown. No code fences. No text outside the JSON.
{{
  "send_alert": true,
  "confidence_score": 0,
  "confidence_reason": "one sentence including fatigue impact if applicable",
  "news_flag": "none or describe the event",
  "bias": "LONG or SHORT or WAIT",
  "bias_reason": "max 12 words",
  "entry": "price or range",
  "sl": 0.0,
  "sl_note": "one sentence on SL placement logic and refinement after trigger",
  "tp1": 0.0,
  "tp2": 0.0,
  "rr_tp1": "x.x",
  "rr_tp2": "x.x",
  "lot_size": "x.x",
  "sl_pts": 0,
  "trigger": "exact M15 or H1 candle pattern required before entry",
  "invalid_if": "exact price action that cancels this trade",
  "confluences": ["item1", "item2", "item3"],
  "missing": [{{"item": "name", "reason": "why it matters"}}],
  "macro_line1": "main macro driver for {pair} right now",
  "macro_line2": "key upcoming event for {pair} this week",
  "mindset": "one sharp psychological trap to avoid on this exact setup",
  "ob_top": 0.0,
  "ob_bottom": 0.0,
  "ob_type": "bullish or bearish",
  "ob_confirmed": true,
  "fvg_top": 0.0,
  "fvg_bottom": 0.0,
  "fvg_type": "bullish or bearish",
  "fvg_confirmed": true,
 "chart_annotations": [{{"label": "short label", "price": 0.0, "status": "confirmed or missing"}}],
  "geo_flag": false
}}"""

# ── Gemini call ───────────────────────────────────────────────────────────────
def call_gemini(prompt):
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r      = requests.post(url, json=body, timeout=90)
        result = r.json()
        if "candidates" not in result:
            return None, f"Gemini error: {result}"
        raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n",1)[1].rsplit("```",1)[0]
        return json.loads(raw), None
    except Exception as e:
        return None, f"Gemini error: {str(e)}"

# ── Intra-week outcome checking ───────────────────────────────────────────────
def get_week_start_utc():
    """Returns UTC datetime for Monday 00:00 IST of the current week."""
    ist_now           = datetime.utcnow() + timedelta(hours=5, minutes=30)
    days_since_monday = ist_now.weekday()          # Monday=0 … Sunday=6
    monday_ist        = ist_now.replace(hour=0, minute=0, second=0, microsecond=0) \
                        - timedelta(days=days_since_monday)
    monday_utc        = monday_ist - timedelta(hours=5, minutes=30)
    return monday_ist, monday_utc   # returns both for display and comparison


def check_outcome_for_alert(alert):
    """
    Fetches M15 candles after alert_time and returns the first SL or TP1 hit.
    M15 used (not H1) — same single fetch, finer resolution.
    Returns: ('win_tp1', price) | ('loss', price) | ('pending', None)
    Sequential yfinance fetch — never call in parallel.
    """
    pair   = alert.get('pair', '')
    symbol = next((p['symbol'] for p in config['pairs'] if p['name'] == pair), None)
    if not symbol:
        return 'pending', None
    try:
        # Normalise bias — handles strings like "WAIT for pullback, then LONG"
        raw_bias = str(alert.get('bias', '')).upper()
        if   'LONG'  in raw_bias: bias = 'LONG'
        elif 'SHORT' in raw_bias: bias = 'SHORT'
        else: return 'pending', None

        sl  = float(alert.get('sl',  0) or 0)
        tp1 = float(alert.get('tp1', 0) or 0)
        if sl <= 0 or tp1 <= 0:
            return 'pending', None

        alert_time = datetime.strptime(alert['timestamp_utc'], "%Y-%m-%d %H:%M")

        # Single M15 fetch — one call, finer resolution than H1
        df = clean_df(yf.download(symbol,
            start=(alert_time - timedelta(hours=1)).strftime('%Y-%m-%d'),
            interval="15m", progress=False))
        if df is None:
            return 'pending', None

        for ts, row in df.iterrows():
            try:
                ts_n = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
                if ts_n < alert_time:
                    continue
                h = float(row['High'])
                l = float(row['Low'])
            except Exception:
                continue
            if bias == 'LONG':
                if l <= sl:  return 'loss',    sl
                if h >= tp1: return 'win_tp1', tp1
            else:
                if h >= sl:  return 'loss',    sl
                if l <= tp1: return 'win_tp1', tp1

        return 'pending', None

    except Exception as e:
        print(f"    Outcome check error ({pair}): {e}")
        return 'pending', None


def run_intraweek_outcome_check():
    """
    Scans all pending alerts fired since Monday 00:00 IST.
    Updates outcomes in alert_log and saves. Sequential fetches only.
    """
    monday_ist, monday_utc = get_week_start_utc()
    print(f"  Intra-week outcome check — from "
          f"{monday_ist.strftime('%a %d %b %H:%M IST')} "
          f"/ {monday_utc.strftime('%d %b %H:%M UTC')} to now...")

    updated = 0
    for alert in alert_log:
        if alert.get('outcome') in ('win_tp1', 'loss', 'invalidated'):
            continue
        if alert.get('alert_type') not in ('zone', 'zone_intraday'):
            continue

        try:
            alert_time = datetime.strptime(alert['timestamp_utc'], "%Y-%m-%d %H:%M")
        except Exception:
            continue

        if alert_time < monday_utc:
            continue

        # Skip alerts with no trade levels (pure breakout notifications)
        sl  = float(alert.get('sl',  0) or 0)
        tp1 = float(alert.get('tp1', 0) or 0)
        if sl <= 0 or tp1 <= 0:
            continue

        print(f"    Checking {alert['pair']} @ {alert['timestamp_utc']} UTC...", end=" ")
        outcome, outcome_price = check_outcome_for_alert(alert)

        if outcome != 'pending':
            alert['outcome']            = outcome
            alert['outcome_price']      = outcome_price
            alert['outcome_checked_at'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            print(f"→ {outcome} at {outcome_price}")
            updated += 1
        else:
            print("→ still pending.")

    save_alert_log()
    print(f"  Done. {updated} outcome(s) updated.")


# ── Chart generation ──────────────────────────────────────────────────────────
def generate_chart(df, title, levels, data):
    try:
        if df is None or df.empty:
            return None
        df_plot = df.tail(40).copy().reset_index(drop=True)
        for col in ['Open','High','Low','Close']:
            if col not in df_plot.columns:
                return None

        fig    = plt.figure(figsize=(10,5), facecolor='#131722')
        gs     = GridSpec(4, 1, figure=fig, hspace=0.04)
        ax     = fig.add_subplot(gs[:3,0])
        ax_vol = fig.add_subplot(gs[3,0], sharex=ax)

        for a in [ax, ax_vol]:
            a.set_facecolor('#131722')
            for s in a.spines.values():
                s.set_color('#2a2a3e')

        for i, row in df_plot.iterrows():
            try:
                o=float(row['Open']); h=float(row['High'])
                l=float(row['Low']);  c=float(row['Close'])
                if any(np.isnan(v) for v in [o,h,l,c]):
                    continue
                col = '#26a69a' if c >= o else '#ef5350'
                ax.plot([i,i],[l,h], color=col, linewidth=0.8, zorder=2)
                body = abs(c-o) or (h-l)*0.01
                ax.add_patch(patches.Rectangle((i-0.35,min(o,c)),0.7,body,
                    facecolor=col,linewidth=0,alpha=0.9,zorder=3))
            except:
                continue

        n = len(df_plot)

        ob_top    = float(data.get('ob_top',0) or 0)
        ob_bottom = float(data.get('ob_bottom',0) or 0)
        if ob_top>0 and ob_bottom>0 and abs(ob_top-ob_bottom)>0:
            oc = '#26a69a' if data.get('ob_type','')=='bullish' else '#ef5350'
            ok = data.get('ob_confirmed',True)
            ax.add_patch(patches.Rectangle((0,ob_bottom),n,ob_top-ob_bottom,
                facecolor=oc,edgecolor=oc,linewidth=1,
                alpha=0.2 if ok else 0.08,linestyle='-' if ok else '--',zorder=1))
            ax.text(1,ob_top,f" OB {'OK' if ok else '?'}",
                color=oc,fontsize=7,va='bottom',fontweight='bold',zorder=5)

        fvg_top    = float(data.get('fvg_top',0) or 0)
        fvg_bottom = float(data.get('fvg_bottom',0) or 0)
        if fvg_top>0 and fvg_bottom>0 and abs(fvg_top-fvg_bottom)>0:
            fok = data.get('fvg_confirmed',True)
            ax.add_patch(patches.Rectangle((0,fvg_bottom),n,fvg_top-fvg_bottom,
                facecolor='#3498db',edgecolor='#3498db',linewidth=1,
                alpha=0.18 if fok else 0.07,linestyle='-' if fok else '--',zorder=1))
            ax.text(1,fvg_top,f" FVG {'OK' if fok else '?'}",
                color='#3498db',fontsize=7,va='bottom',fontweight='bold',zorder=5)

        level_cfg = {
            'tp2':     ('#1e8449','--',1.0,'TP2'),
            'tp1':     ('#27ae60','-', 1.5,'TP1'),
            'entry':   ('#e67e22','-', 1.5,'Entry'),
            'zone':    ('#9b59b6','--',1.2,'Zone'),
            'current': ('#ffffff',':' ,0.8,'Now'),
            'sl':      ('#e74c3c','-', 1.5,'SL'),
        }
        for key,(color,style,width,lbl) in level_cfg.items():
            val = levels.get(key,0)
            try:
                price = float(str(val).split('-')[0].strip()) if val else 0
            except:
                price = 0
            if price > 0:
                ax.axhline(y=price,color=color,linestyle=style,linewidth=width,alpha=0.85,zorder=4)
                ax.text(n+0.3,price,f"{lbl}: {price:,.5f}",
                    color=color,fontsize=7,va='center',fontweight='bold',zorder=5)

        for i, row in df_plot.iterrows():
            try:
                vol = float(row.get('Volume',0) or 0)
                if np.isnan(vol): vol=0
                vc = '#26a69a' if float(row['Close'])>=float(row['Open']) else '#ef5350'
                ax_vol.bar(i,vol,color=vc,alpha=0.5,width=0.7)
            except:
                continue

        ax.set_title(title,color='#dddddd',fontsize=10,pad=6,fontweight='bold',loc='left')
        ax.tick_params(colors='#666',labelsize=7)
        ax.yaxis.tick_right()
        ax.yaxis.set_tick_params(labelcolor='#888')
        ax.xaxis.set_visible(False)
        ax.set_xlim(-1,n+12)
        ax_vol.tick_params(colors='#555',labelsize=6)
        ax_vol.set_ylabel('Vol',color='#555',fontsize=6)
        ax_vol.yaxis.tick_right()
        ax_vol.xaxis.set_visible(False)
        ax_vol.set_xlim(-1,n+12)

        plt.tight_layout(pad=0.3)
        buf = BytesIO()
        fig.savefig(buf,format='png',dpi=72,bbox_inches='tight',
            facecolor='#131722',edgecolor='none')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        print(f"    Chart ok: {len(b64)//1024}KB")
        return b64
    except Exception as e:
        print(f"    Chart error: {e}")
        plt.close('all')
        return None

# ── Zone email HTML ───────────────────────────────────────────────────────────
def build_zone_email_html(data, pair, zone_level, zone_label, current_price,
                          chart1_b64, chart2_b64):
    ist_time    = (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M")
    utc_time    = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    score       = data.get("confidence_score",0)
    bias        = data.get("bias","WAIT")
    bias_color  = "#e74c3c" if bias=="SHORT" else "#27ae60" if bias=="LONG" else "#f39c12"
    score_color = "#27ae60" if score>=8 else "#f39c12" if score>=6 else "#e74c3c"

    news_flag = data.get("news_flag","none")
    news_html = (f'<div style="background:#fff3cd;padding:9px 24px;border-left:4px solid #f39c12;'
                 f'font-size:12px;color:#856404;"><b>NEWS:</b> {news_flag}</div>'
                 if news_flag and news_flag.lower()!="none" else "")

    conf_items = "".join([
        f'<li style="margin-bottom:5px;padding:7px 10px;background:#f0fff4;border-radius:6px;font-size:13px;">&#10003; {c}</li>'
        for c in data.get("confluences",[])])
    miss_items = "".join([
        f'<li style="margin-bottom:5px;padding:7px 10px;background:#fff8f0;border-radius:6px;font-size:13px;">&#10007; <b>{m["item"]}</b> — <span style="color:#777;font-style:italic;">{m["reason"]}</span></li>'
        for m in data.get("missing",[])])

    price_map = ""
    try:
        ep   = float(str(data["entry"]).split("-")[0].strip())
        sl_v = float(data.get("sl",0))
        lvls_map = {"TP2":float(data["tp2"]),"TP1":float(data["tp1"]),
                    "Entry":ep,"Current":float(current_price),
                    "Zone":float(zone_level),"SL":sl_v}
        lc   = {"SL":"#e74c3c","Zone":"#9b59b6","Current":"#3498db",
                "Entry":"#e67e22","TP1":"#27ae60","TP2":"#1e8449"}
        vals = [v for v in lvls_map.values() if v>0]
        pmin,pmax = min(vals),max(vals)
        pr   = pmax-pmin or 1
        rows = ""
        for lbl,price in sorted([(k,v) for k,v in lvls_map.items() if v>0],
                                  key=lambda x:x[1],reverse=True):
            c   = lc.get(lbl,"#888")
            bar = int(((price-pmin)/pr)*75)+15
            rows += (f'<tr>'
                     f'<td style="padding:5px 10px;color:{c};font-weight:bold;font-size:12px;width:65px;">{lbl}</td>'
                     f'<td style="padding:5px 6px;"><div style="background:{c};height:10px;border-radius:4px;width:{bar}%;"></div></td>'
                     f'<td style="padding:5px 10px;color:{c};font-weight:bold;font-size:12px;text-align:right;white-space:nowrap;">{price:,.5f}</td>'
                     f'</tr>')
        price_map = (f'<h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">PRICE MAP</h3>'
                     f'<table style="width:100%;border-collapse:collapse;background:#f8f9fa;border-radius:8px;margin-bottom:20px;">{rows}</table>')
    except:
        pass

    chart1_html = (f'<h3 style="color:#1a1a2e;font-size:13px;margin:20px 0 6px;">H1 CHART</h3>'
                   f'<img src="cid:chart_h1" style="width:100%;border-radius:8px;display:block;" />'
                   if chart1_b64 else '<p style="color:#aaa;font-size:12px;">H1 chart unavailable.</p>')
    chart2_html = (f'<h3 style="color:#1a1a2e;font-size:13px;margin:16px 0 6px;">M15 CHART</h3>'
                   f'<img src="cid:chart_m15" style="width:100%;border-radius:8px;display:block;" />'
                   if chart2_b64 else '<p style="color:#aaa;font-size:12px;">M15 chart unavailable.</p>')

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:16px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">

  <div style="background:#1a1a2e;padding:18px 24px;">
    <h2 style="color:white;margin:0;font-size:17px;">ZONE ALERT: {pair} — {zone_label}</h2>
    <p style="color:#8899bb;margin:5px 0 0;font-size:12px;">{utc_time} UTC | {ist_time} IST</p>
  </div>

  <div style="background:{bias_color};padding:12px 24px;">
    <p style="color:white;margin:0;font-size:15px;font-weight:bold;">{bias} — {data.get("bias_reason","")}</p>
  </div>

  <div style="padding:10px 24px;background:#f8f9fa;border-bottom:1px solid #eee;">
    <span style="font-size:12px;color:#666;">SMC Confidence: </span>
    <span style="font-size:18px;font-weight:bold;color:{score_color};">{score}/10</span>
    <span style="font-size:12px;color:#888;margin-left:8px;">— {data.get("confidence_reason","")}</span>
  </div>

  {news_html}

  <div style="padding:20px 24px;">

    <p style="background:#f4f4f8;padding:10px 14px;border-radius:8px;font-size:13px;color:#333;margin:0 0 16px;">
      Zone: <b>{zone_label}</b> at <b>{zone_level}</b> &nbsp;|&nbsp; Now: <b>{current_price}</b>
    </p>

    {chart1_html}
    {chart2_html}

    {price_map}

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">TRADE LEVELS</h3>
    <table style="width:100%;font-size:13px;margin-bottom:16px;">
      <tr><td style="padding:5px 0;color:#555;width:130px;">Entry</td><td style="padding:5px 0;font-weight:bold;">{data.get("entry","")}</td></tr>
      <tr><td style="padding:5px 0;color:#e74c3c;">Stop Loss</td><td style="padding:5px 0;font-weight:bold;color:#e74c3c;">{data.get("sl","")}</td></tr>
      <tr><td style="padding:5px 0;color:#555;">SL Note</td><td style="padding:5px 0;font-size:12px;color:#777;">{data.get("sl_note","")}</td></tr>
      <tr><td style="padding:5px 0;color:#27ae60;">TP1</td><td style="padding:5px 0;font-weight:bold;color:#27ae60;">{data.get("tp1","")} &nbsp;<span style="font-weight:normal;color:#888;">(R:R {data.get("rr_tp1","")})</span></td></tr>
      <tr><td style="padding:5px 0;color:#1e8449;">TP2</td><td style="padding:5px 0;font-weight:bold;color:#1e8449;">{data.get("tp2","")} &nbsp;<span style="font-weight:normal;color:#888;">(R:R {data.get("rr_tp2","")})</span></td></tr>
      <tr><td style="padding:5px 0;color:#555;">Lot Size</td><td style="padding:5px 0;font-weight:bold;">{data.get("lot_size","")} lots &nbsp;<span style="font-weight:normal;color:#888;">({data.get("sl_pts","")} pts risk)</span></td></tr>
    </table>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">CONFLUENCES</h3>
    <ul style="list-style:none;padding:0;margin:0 0 20px;">{conf_items}</ul>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">MISSING</h3>
    <ul style="list-style:none;padding:0;margin:0 0 20px;">{miss_items}</ul>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 6px;">TRIGGER</h3>
    <p style="font-size:13px;color:#333;background:#fffbea;padding:10px 14px;border-radius:8px;border-left:4px solid #f39c12;margin:0 0 20px;">{data.get("trigger","")}</p>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 6px;">INVALID IF</h3>
    <p style="font-size:13px;color:#c0392b;background:#fef0f0;padding:10px 14px;border-radius:8px;border-left:4px solid #e74c3c;margin:0 0 20px;">{data.get("invalid_if","")}</p>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 6px;">MACRO</h3>
    <p style="font-size:13px;color:#444;margin:0 0 20px;line-height:1.7;">{data.get("macro_line1","")}<br>{data.get("macro_line2","")}</p>

    <div style="background:#1a1a2e;padding:14px 18px;border-radius:10px;">
      <p style="color:#8899bb;font-size:10px;margin:0 0 4px;text-transform:uppercase;letter-spacing:1px;">MINDSET</p>
      <p style="color:white;font-size:13px;margin:0;font-style:italic;line-height:1.6;">{data.get("mindset","")}</p>
    </div>

  </div>
</div>
</body>
</html>"""

# ── Log alert ─────────────────────────────────────────────────────────────────
def log_alert(pair, zone_level, zone_label, current_price, data, alert_type="zone", geo_flag=False):
    ist_time = (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M")
    alert_log.append({
        "id":               f"{pair}_{int(datetime.utcnow().timestamp())}",
        "alert_type":       alert_type,
        "pair":             pair,
        "timestamp_utc":    datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "ist_time":         ist_time,
        "zone_level":       zone_level,
        "zone_label":       zone_label,
        "bias":             data.get("bias","") if data else "",
        "entry":            data.get("entry","") if data else "",
        "sl":               data.get("sl",0) if data else 0,
        "tp1":              data.get("tp1",0) if data else 0,
        "tp2":              data.get("tp2",0) if data else 0,
        "confidence_score": data.get("confidence_score",0) if data else 0,
        "confluences":      data.get("confluences",[]) if data else [],
        "trigger":          data.get("trigger","") if data else "",
        "invalid_if":       data.get("invalid_if","") if data else "",
        "geo_flag":         geo_flag,
        "outcome":          "pending",
        "outcome_price":    None,
        "outcome_checked_at": None
    })
    save_alert_log()

# ── Send email ────────────────────────────────────────────────────────────────
def send_email(subject, html_body, chart1_b64=None, chart2_b64=None):
    recipients = config["account"].get("alert_emails", [ALERT_EMAIL])
    for recipient in recipients:
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_ADDRESS
        msg["To"]      = recipient
        msg.attach(MIMEText(html_body, "html"))
        for cid, b64data in [("chart_h1", chart1_b64), ("chart_m15", chart2_b64)]:
            if b64data:
                img = MIMEImage(base64.b64decode(b64data))
                img.add_header("Content-ID", f"<{cid}>")
                img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
                msg.attach(img)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_PASS)
            server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
        print(f"    Sent to {recipient}")
def send_simple_email(subject, html_body):
    send_email(subject, html_body, None, None)

def build_ok_email_html():
    return f"""
    <html>
      <body style="font-family: Arial, sans-serif; background:#f6f8fb; padding:24px;">
        <div style="max-width:600px; margin:auto; background:white; border-radius:12px; padding:24px; border:1px solid #e5e7eb;">
          <h2 style="margin-top:0; color:#1f2937;">System OK — No valid trade in the last 3 hours</h2>
          <p style="color:#374151; font-size:14px;">The trading system ran correctly.</p>
          <p style="color:#374151; font-size:14px;">No valid trade setup passed all filters in the last 3 hours.</p>
          <p style="color:#6b7280; font-size:12px; margin-top:20px;">{utc_str()} UTC</p>
        </div>
      </body>
    </html>
    """

def build_error_email_html(error_lines):
    items = "".join([f"<li>{line}</li>" for line in error_lines[:10]])
    return f"""
    <html>
      <body style="font-family: Arial, sans-serif; background:#fff7f7; padding:24px;">
        <div style="max-width:600px; margin:auto; background:white; border-radius:12px; padding:24px; border:1px solid #fecaca;">
          <h2 style="margin-top:0; color:#991b1b;">System Error — Trading engine needs attention</h2>
          <p style="color:#374151; font-size:14px;">The system hit one or more errors in the last run.</p>
          <ul style="color:#374151; font-size:14px;">{items}</ul>
          <p style="color:#6b7280; font-size:12px; margin-top:20px;">{utc_str()} UTC</p>
        </div>
      </body>
    </html>
    """

# ── MAIN ──────────────────────────────────────────────────────────────────────
print(f"Alert engine started {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

run_errors = []
alerts_fired = 0

# Outcome check runs first — regardless of market hours.
run_intraweek_outcome_check()

market_open, market_status = is_market_open()
print(f"  Market: {market_status}")
if not market_open:
    print("  Exiting — market closed.")
    exit(0)

macro_news = fetch_macro_news()
for pair_conf in config["pairs"]:
    symbol      = pair_conf["symbol"]
    name        = pair_conf["name"]
    prox        = pair_conf["proximity_pct"]
    min_touches = pair_conf.get("min_touches", 1)
    min_conf    = pair_conf.get("min_confidence", 5)

    print(f"  Scanning {name}...")

    try:
        zones, current_price, df1 = detect_zones_and_candles(symbol, min_touches)

        if current_price is None:
            print(f"    No data for {name}. Skipping.")
            log_scan(name, "error", "No market data returned for this pair.")
            run_errors.append(f"{name}: No market data returned for this pair.")
            continue

        if not zones:
            log_scan(name, "no_zone_found", "No valid zones detected in current scan.")
            continue

        zones_in_proximity = 0

        for zone_level, touches in zones:
            dist_pct = abs(current_price - zone_level) / zone_level * 100

            if dist_pct > prox:
                continue

            zones_in_proximity += 1
            zone_label = get_zone_label(zone_level, current_price)

            if not should_alert_zone(name, zone_level, current_price, prox):
                print(f"    {name} @ {zone_level:.5f} — price hasn't moved enough since last alert.")
                log_scan(name, "blocked_revisit", "Zone already alerted and price has not moved far enough away yet.", zone_level)
                continue

            df2 = fetch_m15_data(symbol)
            if df2 is None:
                print(f"    No M15 data for {name}. Skipping zone.")
                log_scan(name, "error", "No M15 market data returned once zone entered proximity.", zone_level)
                continue

            fatigue = count_zone_alerts(name, zone_level)
            print(f"    ZONE HIT: {name} {zone_label} @ {zone_level:.5f} dist:{dist_pct:.2f}% fatigue:{fatigue}")

            prompt = build_zone_prompt(
                name,
                round(zone_level, 5),
                zone_label,
                round(current_price, 5),
                macro_news,
                df1,
                df2,
                min_conf,
                fatigue
            )

            data, error = call_gemini(prompt)

            if error:
                print(f"    {error}")
                log_scan(name, "error", error, zone_level)
                run_errors.append(f"{name}: {error}")
                continue

            score = data.get("confidence_score", 0)

            if not data.get("send_alert", False):
                reason = data.get("confidence_reason", "Gemini rejected this setup.")
                print(f"    {name} skipped — Gemini said no alert. {reason}")
                log_scan(name, "rejected_gemini_no_alert", reason, zone_level)
                continue

            if score < min_conf:
                reason = f"Score {score}/10 below minimum {min_conf}. {data.get('confidence_reason', '')}".strip()
                print(f"    {name} skipped — {reason}")
                log_scan(name, "rejected_low_confidence", reason, zone_level)
                continue

            levels = {
                'zone': zone_level,
                'current': current_price,
                'entry': data.get('entry', ''),
                'sl': data.get('sl', 0),
                'tp1': data.get('tp1', 0),
                'tp2': data.get('tp2', 0)
            }

            chart1 = generate_chart(df1, f"{name} — H1", levels, data)
            chart2 = generate_chart(df2, f"{name} — M15", levels, data)

            html = build_zone_email_html(
                data,
                name,
                round(zone_level, 5),
                zone_label,
                round(current_price, 5),
                chart1,
                chart2
            )
            subject = f"[{score}/10] {name} | {zone_label} | {round(zone_level, 5)} | {datetime.utcnow().strftime('%H:%M')} UTC"

            send_email(subject, html, chart1, chart2)
            log_alert(name, round(zone_level, 5), zone_label, round(current_price, 5), data, "zone", geo_flag=bool(data.get('geo_flag', False)))
            log_scan(name, "alert_sent", f"Trade alert sent successfully at score {score}/10.", zone_level)
            record_zone_alert(name, zone_level, current_price)

            system_status["last_trade_alert_utc"] = utc_str()
            alerts_fired += 1
            print(f"    Sent: {name} SMC[{score}/10]")
            break

        if zones_in_proximity == 0:
            log_scan(name, "zone_outside_proximity", "Zones were detected, but none are close enough to current price.")

    except Exception as e:
        print(f"    Unexpected pair-level error: {str(e)}")
        log_scan(name, "error", f"Unexpected pair-level error: {str(e)}")
        run_errors.append(f"{name}: {str(e)}")
   
if alerts_fired == 0 and not run_errors and should_send_ok():
    print("  Sending 3-hour OK email...")
    send_simple_email(
        "System OK — No valid trade in the last 3 hours",
        build_ok_email_html()
    )
    system_status["last_ok_email_utc"] = utc_str()

if run_errors and should_send_error():
    print("  Sending 3-hour error email...")
    send_simple_email(
        "System Error — Trading engine needs attention",
        build_error_email_html(run_errors)
    )
    system_status["last_error_email_utc"] = utc_str()

save_alert_log()
save_json(SCAN_LOG_FILE, scan_log)
save_json(SYSTEM_STATUS_FILE, system_status)
save_visit_state()

print(f"Alert log saved: {len(alert_log)} total entries.")
print(f"Scan log saved: {len(scan_log)} total entries.")
print(f"Scan complete. {alerts_fired} alert(s) fired.")
