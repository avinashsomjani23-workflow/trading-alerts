import yfinance as yf
import pandas as pd
import json, os, smtplib, requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import base64
from io import BytesIO

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

def save_alert_log():
    with open(ALERT_LOG_FILE, "w") as f:
        json.dump(alert_log, f, indent=2)
    print(f"  Alert log saved: {len(alert_log)} total entries")

# ── Market hours ──────────────────────────────────────────────
def is_market_open():
    ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    wd, h, m = ist_now.weekday(), ist_now.hour, ist_now.minute
    if wd == 5: return False, "Saturday IST — closed."
    if wd == 6: return False, "Sunday IST — closed."
    if wd == 0 and (h < 2 or (h == 2 and m < 30)):
        return False, "Monday before 2:30 AM IST."
    if wd == 4 and h >= 23 and m >= 30:
        return False, "Friday after 11:30 PM IST."
    return True, f"Open — {ist_now.strftime('%A %H:%M')} IST"

# ── Clean DataFrame ───────────────────────────────────────────
def clean_df(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    for col in ['Open','High','Low','Close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Open','High','Low','Close'])
    return df if not df.empty else None

# ── Zone detection ────────────────────────────────────────────
def detect_zones_and_candles(symbol, min_touches, style):
    if style == "swing":
        tf1, tf2, p1, p2 = "4h", "1h", "60d", "15d"
        tf1_label, tf2_label = "H4", "H1"
    else:
        tf1, tf2, p1, p2 = "1h", "15m", "15d", "5d"
        tf1_label, tf2_label = "H1", "M15"

    df1 = clean_df(yf.download(symbol, period=p1, interval=tf1, progress=False))
    df2 = clean_df(yf.download(symbol, period=p2, interval=tf2, progress=False))

    if df1 is None:
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

SMC SCORECARD (score out of 10):
STRUCTURE (3pts): +1 H4/Daily BOS confirms trend | +1 Price in Premium/Discount | +1 CHoCH confirmed H1/M15
ZONE QUALITY (3pts): +1 Valid OB at zone |
