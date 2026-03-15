import yfinance as yf
import pandas as pd
import json, os, smtplib, requests, time
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
from io import BytesIO


# ── Config ────────────────────────────────────────────────────────────────────
with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY    = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_PASS    = os.environ["GMAIL_APP_PASSWORD"]
ALERT_EMAIL   = os.environ["ALERT_EMAIL"]
NEWS_API_KEY  = os.environ.get("NEWS_API_KEY", "")

# Forex pairs — used for liquidity sweep exclusion and volume bar suppression
FOREX_PAIRS = {"EURUSD", "USDJPY", "NZDUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF"}

# ── Alert log ─────────────────────────────────────────────────────────────────
ALERT_LOG_FILE = "alert_log.json"
try:
    with open(ALERT_LOG_FILE) as f:
        alert_log = json.load(f)
    print(f"  Loaded {len(alert_log)} existing log entries")
except Exception:
    alert_log = []

def save_alert_log():
    with open(ALERT_LOG_FILE, "w") as f:
        json.dump(alert_log, f, indent=2)

# ── Zone visit state ──────────────────────────────────────────────────────────
VISIT_FILE = "zone_visit_state.json"
try:
    with open(VISIT_FILE) as f:
        visit_state = json.load(f)
except Exception:
    visit_state = {}

def save_visit_state():
    with open(VISIT_FILE, "w") as f:
        json.dump(visit_state, f)

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

def should_alert_breakout(pair, broken_level, current_price, proximity_pct):
    key = f"{pair}_bo_{round(broken_level, 4)}"
    if key not in visit_state:
        return True
    last_price = float(visit_state[key].get("last_alert_price", current_price))
    dist_pct   = abs(current_price - last_price) / broken_level * 100
    return dist_pct > proximity_pct * 1.5

def record_breakout_alert(pair, broken_level, current_price):
    key = f"{pair}_bo_{round(broken_level, 4)}"
    visit_state[key] = {"last_alert_price": current_price}
    save_visit_state()

# ── Zone touch count — OHLC based (replaces alert-log counting) ───────────────
# A candle "touches" the zone if its High or Low entered the zone boundary.
# This is more accurate than counting email alerts — it uses actual price data.
def count_zone_touches_ohlc(df1, zone_level, proximity_pct):
    """
    Count how many H1 candles in df1 had a High or Low that entered the zone.
    Demand zone: candle High >= zone_level - proximity_pct%
    Supply zone: candle Low  <= zone_level + proximity_pct%
    We use a symmetric check: candle touched zone if either
      High >= zone_level * (1 - proximity_pct/100)
      OR
      Low  <= zone_level * (1 + proximity_pct/100)
    """
    try:
        if df1 is None or df1.empty:
            return 0
        highs  = df1['High'].values.flatten().astype(float)
        lows   = df1['Low'].values.flatten().astype(float)
        upper  = zone_level * (1 + proximity_pct / 100)
        lower  = zone_level * (1 - proximity_pct / 100)
        count  = 0
        for h, l in zip(highs, lows):
            if h >= lower and l <= upper:
                count += 1
        return count
    except Exception:
        return 0

def zone_depletion_label(touch_count):
    """Plain-language label for zone depletion state."""
    if touch_count <= 2:
        return "Fresh zone — tested 1–2 times, first reaction likely strong."
    elif touch_count <= 4:
        return f"Showing depletion — zone tested {touch_count} times, trade with moderate caution."
    else:
        return f"Heavily depleted — zone tested {touch_count} times, trade with caution, require extra confluence."

# ── Market hours — blocks 00:00–08:00 IST weekdays + full weekend ─────────────
def is_market_open():
    ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    wd, h = ist.weekday(), ist.hour  # Mon=0 … Fri=4, Sat=5, Sun=6

    if wd == 5:
        return False, "Saturday — market closed."
    if wd == 6:
        return False, "Sunday — market closed."
    if h < 8:
        return False, f"Quiet hours (midnight–08:00 IST) — {ist.strftime('%A %H:%M')} IST."
    if wd == 4 and h >= 23:
        return False, "Friday after 11:00 PM IST — market closing."

    return True, f"Open — {ist.strftime('%A %H:%M')} IST"

# ── Data helpers ──────────────────────────────────────────────────────────────
def clean_df(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def check_data_freshness(df, label, max_age_minutes=45):
    """
    Returns True if data is fresh enough to use.
    Returns False (and prints a warning) if the last candle is older than max_age_minutes.
    This is critical — stale data leads to alerts on prices that no longer exist.
    """
    if df is None or df.empty:
        return False
    try:
        last_ts = df.index[-1]
        if hasattr(last_ts, 'tzinfo') and last_ts.tzinfo is not None:
            last_ts = last_ts.replace(tzinfo=None)
        age_minutes = (datetime.utcnow() - last_ts).total_seconds() / 60
        if age_minutes > max_age_minutes:
            print(f"    ⚠ STALE DATA [{label}]: last candle is {age_minutes:.0f} min old (limit: {max_age_minutes} min). Skipping pair.")
            return False
        return True
    except Exception as e:
        print(f"    Freshness check error [{label}]: {e}")
        return False

def get_atr(df, period=14):
    try:
        highs  = df['High'].values.flatten().astype(float)
        lows   = df['Low'].values.flatten().astype(float)
        closes = df['Close'].values.flatten().astype(float)
        trs = [max(highs[i] - lows[i],
                   abs(highs[i] - closes[i-1]),
                   abs(lows[i]  - closes[i-1]))
               for i in range(1, len(closes))]
        if len(trs) < period:
            return None
        return float(np.mean(trs[-period:]))
    except Exception:
        return None

# ── Detect zones + candles — intraday (H1 + M15) ─────────────────────────────
def detect_zones_and_candles(symbol, min_touches):
    """Sequential fetch — yfinance is NOT thread-safe, never parallelize."""
    df1 = clean_df(yf.download(symbol, period="15d", interval="1h",  progress=False))
    df2 = clean_df(yf.download(symbol, period="5d",  interval="15m", progress=False))

    if df1 is None:
        return [], None, None, None

    # Freshness check on H1 — primary data
    if not check_data_freshness(df1, f"{symbol} H1"):
        return [], None, None, None

    current_price = float(df1['Close'].iloc[-1])
    lb    = config["zone_detection"]["swing_lookback"]
    highs = df1['High'].values.flatten()
    lows  = df1['Low'].values.flatten()

    swing_points = []
    for i in range(lb, len(highs) - lb):
        if highs[i] == max(highs[i-lb:i+lb+1]):
            swing_points.append(float(highs[i]))
        if lows[i]  == min(lows[i-lb:i+lb+1]):
            swing_points.append(float(lows[i]))

    if not swing_points:
        return [], current_price, df1, df2

    swing_points = sorted(swing_points)
    clusters = [[swing_points[0]]]
    for lvl in swing_points[1:]:
        if (lvl - clusters[-1][-1]) / clusters[-1][-1] * 100 < 0.3:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    zones = [(float(np.mean(c)), len(c)) for c in clusters if len(c) >= min_touches]
    return zones, current_price, df1, df2

# ── Detect zones + candles — swing (D1 + H4) ─────────────────────────────────
def detect_zones_and_candles_swing(symbol, min_touches):
    """Sequential fetch — yfinance is NOT thread-safe, never parallelize."""
    df1 = clean_df(yf.download(symbol, period="90d", interval="1d",  progress=False))
    df2 = clean_df(yf.download(symbol, period="30d", interval="1h",  progress=False))
    # Note: yfinance does not support "4h" interval directly; using "1h" as H4 proxy
    # and we'll downsample if needed — for prompt context raw H4-equivalent 1h is sufficient

    if df1 is None:
        return [], None, None, None

    if not check_data_freshness(df1, f"{symbol} D1", max_age_minutes=90):
        return [], None, None, None

    current_price = float(df1['Close'].iloc[-1])
    lb    = config["zone_detection"]["swing_lookback"]
    highs = df1['High'].values.flatten()
    lows  = df1['Low'].values.flatten()

    swing_points = []
    for i in range(lb, len(highs) - lb):
        if highs[i] == max(highs[i-lb:i+lb+1]):
            swing_points.append(float(highs[i]))
        if lows[i]  == min(lows[i-lb:i+lb+1]):
            swing_points.append(float(lows[i]))

    if not swing_points:
        return [], current_price, df1, df2

    swing_points = sorted(swing_points)
    clusters = [[swing_points[0]]]
    for lvl in swing_points[1:]:
        if (lvl - clusters[-1][-1]) / clusters[-1][-1] * 100 < 0.3:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    zones = [(float(np.mean(c)), len(c)) for c in clusters if len(c) >= min_touches]
    return zones, current_price, df1, df2

def get_zone_label(zone_level, current_price):
    return "Demand / Support" if zone_level < current_price else "Supply / Resistance"

# ── Breakout detection — minimum score 4/5, candle quality check added ────────
def detect_breakout(df1, pair_name):
    try:
        if df1 is None or len(df1) < 30:
            return None

        highs  = df1['High'].values.flatten().astype(float)
        lows   = df1['Low'].values.flatten().astype(float)
        closes = df1['Close'].values.flatten().astype(float)
        opens  = df1['Open'].values.flatten().astype(float)

        atr = get_atr(df1, 14)
        if atr is None or atr == 0:
            return None

        last   = len(df1) - 2
        if last < 20:
            return None

        l_open  = opens[last]
        l_close = closes[last]
        l_high  = highs[last]
        l_low   = lows[last]
        l_range = l_high - l_low

        # Pre-filter: candle must be at least 1.5x ATR to qualify as a breakout candle
        if l_range < 1.5 * atr:
            return None

        lb = config["zone_detection"]["swing_lookback"]

        sig_highs = []
        sig_lows  = []
        for i in range(lb, last - lb):
            if highs[i] == max(highs[max(0,i-lb):i+lb+1]):
                touches = sum(1 for j in range(last)
                              if abs(highs[j]-highs[i])/highs[i] < 0.002)
                sig_highs.append((float(highs[i]), touches))
            if lows[i]  == min(lows[max(0,i-lb):i+lb+1]):
                touches = sum(1 for j in range(last)
                              if abs(lows[j]-lows[i])/lows[i] < 0.002)
                sig_lows.append((float(lows[i]), touches))

        ma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else float(np.mean(closes))
        buf  = atr * 0.3
        is_forex = pair_name in FOREX_PAIRS

        def score_breakout(direction, level, touches, body_ok, close_quality_ok):
            score   = 0
            reasons = []

            # Break Quality — 2pts (both body close + close position required)
            if body_ok and close_quality_ok:
                score += 2
                reasons.append(
                    f"+2 H1 body closed {'above' if direction=='BULLISH' else 'below'} "
                    f"{round(level,5)} in outer 35% of candle range ({round(l_range/atr,1)}x ATR)")
            elif body_ok and not close_quality_ok:
                reasons.append(
                    f"+0 H1 body closed beyond level but close in middle of candle — "
                    f"long wick signals rejection, not a true breakout")
            else:
                reasons.append("+0 H1 body did not fully close beyond level")

            # Volume — 1pt (skipped for forex with explanation)
            if is_forex:
                reasons.append("+0 Volume not tracked for forex pairs (excluded from scoring)")
            elif 'Volume' in df1.columns:
                try:
                    vols    = df1['Volume'].values.flatten().astype(float)
                    avg_vol = float(np.nanmean(vols[-21:-1]))
                    if avg_vol > 0 and vols[last] >= 1.3 * avg_vol:
                        score += 1
                        reasons.append(
                            f"+1 📊 VOLUME-BASED SIGNAL: Volume {round(vols[last]/avg_vol,1)}x average — "
                            f"institutional participation confirmed")
                    else:
                        reasons.append("+0 Volume below 1.3x average — no institutional confirmation")
                except Exception:
                    reasons.append("+0 Volume data unavailable")
            else:
                reasons.append("+0 Volume not available for this instrument")

            # Level Significance — 1pt
            if touches >= 3:
                score += 1
                reasons.append(f"+1 Level tested {touches} times before — well-established level")
            else:
                reasons.append(f"+0 Level only {touches} prior touch(es) — needs 3+ for significance")

            # Trend Alignment — 1pt
            aligned = (closes[-1] > ma50 if direction == "BULLISH" else closes[-1] < ma50)
            if aligned:
                score += 1
                reasons.append("+1 H1 break direction matches 50-period MA trend")
            else:
                reasons.append("+0 Counter-trend break — price on wrong side of 50-period MA")

            return score, reasons

        # ── Bullish BOS ──
        if l_close > l_open:
            # Candle quality: close must be in upper 35% of the H1 candle range
            close_quality_ok = (l_close >= l_low + 0.65 * l_range) if l_range > 0 else False
            candidates = [(lvl,tc) for lvl,tc in sig_highs
                          if l_close > lvl and l_open <= lvl * 1.005]
            if candidates:
                level, touches = max(candidates, key=lambda x: x[0])
                score, reasons = score_breakout("BULLISH", level, touches, True, close_quality_ok)
                if score >= 4:
                    return {
                        "direction":        "BULLISH",
                        "timeframe":        "H1",
                        "broken_level":     round(level, 5),
                        "break_size_atr":   round(l_range/atr, 1),
                        "level_touches":    touches,
                        "bo_score":         score,
                        "bo_reasons":       reasons,
                        "retest_zone_top":  round(level+buf, 5),
                        "retest_zone_bot":  round(level-buf, 5),
                        "momentum_entry":   round(closes[-1], 5),
                        "retest_entry":     round(level+buf, 5),
                        "sl_momentum":      round(l_low-buf, 5),
                        "sl_retest":        round(level-buf*2, 5),
                        "description":      (f"Bullish BOS on H1: H1 body closed "
                                             f"{round((l_close-level)/level*100,3)}% above {round(level,5)}")
                    }

        # ── Bearish BOS ──
        elif l_close < l_open:
            # Candle quality: close must be in lower 35% of the H1 candle range
            close_quality_ok = (l_close <= l_low + 0.35 * l_range) if l_range > 0 else False
            candidates = [(lvl,tc) for lvl,tc in sig_lows
                          if l_close < lvl and l_open >= lvl * 0.995]
            if candidates:
                level, touches = min(candidates, key=lambda x: x[0])
                score, reasons = score_breakout("BEARISH", level, touches, True, close_quality_ok)
                if score >= 4:
                    return {
                        "direction":        "BEARISH",
                        "timeframe":        "H1",
                        "broken_level":     round(level, 5),
                        "break_size_atr":   round(l_range/atr, 1),
                        "level_touches":    touches,
                        "bo_score":         score,
                        "bo_reasons":       reasons,
                        "retest_zone_top":  round(level+buf, 5),
                        "retest_zone_bot":  round(level-buf, 5),
                        "momentum_entry":   round(closes[-1], 5),
                        "retest_entry":     round(level-buf, 5),
                        "sl_momentum":      round(l_high+buf, 5),
                        "sl_retest":        round(level+buf*2, 5),
                        "description":      (f"Bearish BOS on H1: H1 body closed "
                                             f"{round((level-l_close)/level*100,3)}% below {round(level,5)}")
                    }

        return None

    except Exception as e:
        print(f"    Breakout detection error: {e}")
        return None

# ── Macro news — dual source (Reuters RSS + NewsAPI) ─────────────────────────
def fetch_macro_news():
    """
    Fetches from two sources:
    1. Reuters RSS — full article descriptions
    2. NewsAPI — business headlines + geopolitical keywords
    Returns a structured string fed into the Gemini prompt.
    """
    news_items = []

    # Source 1: Reuters RSS
    try:
        r = requests.get(
            "https://feeds.reuters.com/reuters/businessNews",
            timeout=10
        )
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)
        for item in root.findall('.//item')[:5]:
            title = item.findtext('title', '').strip()
            desc  = (item.findtext('description', '') or
                     item.findtext('{http://purl.org/rss/1.0/modules/content/}encoded', '')).strip()
            if not desc:
                desc = title
            if title:
                news_items.append(f"[Reuters] {title}: {desc[:200]}")
    except Exception as e:
        print(f"    Reuters RSS error: {e}")

    # Source 2: NewsAPI — business headlines
    if NEWS_API_KEY:
        try:
            r = requests.get(
                f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=5&apiKey={NEWS_API_KEY}",
                timeout=10
            )
            for art in r.json().get("articles", [])[:5]:
                title = art.get("title", "").strip()
                desc  = art.get("description", "").strip()
                if title and desc:
                    news_items.append(f"[NewsAPI/Business] {title}: {desc[:200]}")
        except Exception as e:
            print(f"    NewsAPI business error: {e}")

        # Source 2b: NewsAPI — geopolitical/macro keywords
        try:
            r = requests.get(
                f"https://newsapi.org/v2/everything?q=Trump+OR+Fed+OR+war+OR+sanctions+OR+trade"
                f"&language=en&sortBy=publishedAt&pageSize=3&apiKey={NEWS_API_KEY}",
                timeout=10
            )
            for art in r.json().get("articles", [])[:3]:
                title = art.get("title", "").strip()
                desc  = art.get("description", "").strip()
                if title and desc:
                    news_items.append(f"[NewsAPI/Macro] {title}: {desc[:200]}")
        except Exception as e:
            print(f"    NewsAPI macro error: {e}")
    else:
        print("    NEWS_API_KEY not set — skipping NewsAPI source.")

    if not news_items:
        return "Macro news unavailable from all sources."

    return "\n".join(news_items)

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
        except Exception:
            pass
    return result

# ── Day-of-week context for Gemini prompts ────────────────────────────────────
def get_day_of_week_context(pair):
    ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    dow = ist.strftime("%A")  # e.g. "Friday"
    notes = [f"Today is {dow}."]
    if dow == "Friday" and "NAS100" in pair:
        notes.append(
            "Friday NAS100: elevated stop hunt risk in the last 2 hours of the NY session "
            "(after 20:00 UTC) due to end-of-week position squaring. "
            "Use a tighter SL or avoid new entries after 20:00 UTC."
        )
    if dow == "Monday" and pair in FOREX_PAIRS:
        notes.append(
            "Monday forex: weekend gap risk. The first H1 candle may be anomalous. "
            "Wait for the second H1 candle confirmation before entry."
        )
    return " ".join(notes)

# ── Gemini zone prompt ────────────────────────────────────────────────────────
def build_zone_prompt(pair, zone_level, zone_label, current_price,
                      macro_news, df1, df2, min_confidence,
                      touch_count, timeframe_label="INTRADAY"):
    risk_dollar  = config["account"]["balance"] * config["account"]["risk_percent"] / 100
    ist_time     = (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M")
    utc_time     = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    dow_context  = get_day_of_week_context(pair)
    depletion    = zone_depletion_label(touch_count)

    # Zone quality deduction rules based on OHLC touch count
    if touch_count >= 5:
        fatigue_rule = (f"Zone tested {touch_count} times using H1 candle OHLC data. "
                        "HEAVILY DEPLETED — deduct 2 from Zone Quality score. "
                        "Require extra confluence to pass minimum confidence threshold.")
    elif touch_count >= 3:
        fatigue_rule = (f"Zone tested {touch_count} times using H1 candle OHLC data. "
                        "SHOWING DEPLETION — deduct 1 from Zone Quality score.")
    else:
        fatigue_rule = (f"Zone tested {touch_count} times using H1 candle OHLC data. "
                        "FRESH — no deduction.")

    # Liquidity scoring — forex uses H4/D1 structure alignment instead of sweep
    if pair in FOREX_PAIRS:
        liquidity_section = (
            "LIQUIDITY (2pts):     "
            "+1 Entry on correct side (Discount for long, Premium for short) | "
            "+1 H4 or D1 trend structure aligns with trade direction "
            "[IMPORTANT: Do NOT score or penalise for liquidity sweeps on forex — "
            "sweep scoring is excluded for this system on forex pairs]"
        )
    else:
        liquidity_section = (
            "LIQUIDITY (2pts):     "
            "+1 Liquidity swept before zone | "
            "+1 Entry on correct side (Discount for long, Premium for short)"
        )

    tf_primary   = "D1" if timeframe_label == "SWING" else "H1"
    tf_secondary = "H4" if timeframe_label == "SWING" else "M15"
    sl_note_tf   = "SL widened for swing timeframe — use D1 OB wick extreme + larger ATR buffer." if timeframe_label == "SWING" else "SL: OB wick extreme plus ATR buffer (3-5 pips forex, 10-15pts NAS100, 50-100pts BTC/Gold/Silver)."

    return f"""You are a highly skilled professional SMC trader. A {timeframe_label} zone alert has triggered.

PAIR: {pair} | ZONE: {zone_label} at {zone_level} | PRICE NOW: {current_price}
TIME: {utc_time} UTC | {ist_time} IST
ALERT TYPE: {timeframe_label} | PRIMARY TF: {tf_primary} | SECONDARY TF: {tf_secondary}
ACCOUNT: ${config["account"]["balance"]} | RISK: {config["account"]["risk_percent"]}% = ${risk_dollar:.0f} | FIRM: {config["account"]["firm"]}
DAY CONTEXT: {dow_context}

CANDLE DATA:
{format_candles(df1, tf_primary)}
{format_candles(df2, tf_secondary)}

MACRO NEWS (use full descriptions below — do NOT base any insight on headline alone):
{macro_news}

SMC SCORECARD (score out of 10 — this is the ONLY scorecard):
STRUCTURE (3pts):     +1 {tf_primary} BOS confirms trend direction | +1 Price in Premium or Discount zone | +1 CHoCH confirmed on {tf_primary} or {tf_secondary}
ZONE QUALITY (3pts):  +1 Valid OB at zone | +1 FVG overlaps OB | +1 Zone freshness — rule: {fatigue_rule}
{liquidity_section}
RISK/MACRO (2pts):    +1 RR at least 2:1 achievable | +1 No high-impact news in next 2 hours

ZONE DEPLETION STATE: {depletion}
ENTRY: 50pct midpoint of OB candle body. If FVG overlaps OB, use FVG edge (top for longs, bottom for shorts).
{sl_note_tf}
After trigger fires: refine SL to trigger candle extreme only if it gives a tighter SL than OB wick.

MACRO INSTRUCTION: For each relevant macro item, state: (1) the news event in 5 words, (2) directional impact on {pair}, (3) confidence high/medium/low. Maximum 3 bullet points. If nothing is materially relevant to {pair} in the next 4–8 hours, say exactly: "No material macro driver in the next 4–8 hours." Do not pad with general commentary.

MIN CONFIDENCE TO SEND: {min_confidence}/10. Below this, set send_alert to false.

Every candle reference you make MUST include its timeframe ({tf_primary} or {tf_secondary}). Never say "the last candle" without specifying which timeframe.

Return ONLY raw JSON. No markdown. No code fences. No text outside the JSON.
{{
  "send_alert": true,
  "confidence_score": 0,
  "confidence_reason": "one sentence including depletion impact if applicable",
  "news_flag": "none or describe the event",
  "bias": "LONG or SHORT or WAIT",
  "bias_reason": "max 12 words",
  "entry": "price or range",
  "sl": 0.0,
  "sl_note": "one sentence on SL placement — include timeframe of the candle used",
  "tp1": 0.0,
  "tp2": 0.0,
  "rr_tp1": "x.x",
  "rr_tp2": "x.x",
  "lot_size": "x.x",
  "sl_pts": 0,
  "trigger": "exact {tf_secondary} or {tf_primary} candle pattern required before entry — must name timeframe",
  "invalid_if": "exact price action that cancels this trade — must name timeframe",
  "confluences": ["item1", "item2", "item3"],
  "missing": [{{"item": "name", "reason": "why it matters"}}],
  "macro_bullets": ["bullet 1 (5-word event | direction | confidence)", "bullet 2", "bullet 3"],
  "mindset": "one sharp psychological trap to avoid on this exact setup",
  "ob_top": 0.0,
  "ob_bottom": 0.0,
  "ob_type": "bullish or bearish",
  "ob_confirmed": true,
  "fvg_top": 0.0,
  "fvg_bottom": 0.0,
  "fvg_type": "bullish or bearish",
  "fvg_confirmed": true,
  "chart_annotations": [{{"label": "short label", "price": 0.0, "status": "confirmed or missing"}}]
}}"""

# ── Gemini breakout prompt ────────────────────────────────────────────────────
def build_breakout_prompt(pair, bo, current_price, macro_news, df1):
    risk_dollar = config["account"]["balance"] * config["account"]["risk_percent"] / 100
    ist_time    = (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M")
    utc_time    = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    dow_context = get_day_of_week_context(pair)
    d           = bo.get("direction","BULLISH")

    return f"""You are a highly skilled professional breakout trader. A scored H1 breakout alert has triggered.

PAIR: {pair} | DIRECTION: {d} | BROKEN LEVEL: {bo.get('broken_level','')} | PRICE NOW: {current_price}
TIME: {utc_time} UTC | {ist_time} IST
BREAKOUT SCORE: {bo.get('bo_score','')}/5
BREAK DESCRIPTION: {bo.get('description','')}
ACCOUNT: ${config["account"]["balance"]} | RISK: {config["account"]["risk_percent"]}% = ${risk_dollar:.0f} | FIRM: {config["account"]["firm"]}
DAY CONTEXT: {dow_context}

CANDLE DATA:
{format_candles(df1, "H1")}

MACRO NEWS (use full descriptions — do NOT base any insight on headline alone):
{macro_news}

BREAKOUT SCORING ALREADY DONE:
{chr(10).join(bo.get('bo_reasons', []))}

RETEST ZONE: {bo.get('retest_zone_bot','')} — {bo.get('retest_zone_top','')}
MOMENTUM ENTRY: {bo.get('momentum_entry','')} (aggressive — enter now)
RETEST ENTRY: {bo.get('retest_entry','')} (preferred — wait for pullback to broken level)
SL IF MOMENTUM: {bo.get('sl_momentum','')}
SL IF RETEST: {bo.get('sl_retest','')}

MACRO INSTRUCTION: For each relevant macro item, state: (1) the news event in 5 words, (2) directional impact on {pair}, (3) confidence high/medium/low. Maximum 3 bullet points. If nothing is materially relevant to {pair} in the next 4–8 hours, say exactly: "No material macro driver in the next 4–8 hours."

Every candle reference MUST include "H1" or "M15" — never say "the candle" without a timeframe.

MIN CONFIDENCE TO SEND: 4/5 (already scored — confirm validity and add context below).

Return ONLY raw JSON. No markdown. No code fences. No text outside the JSON.
{{
  "confirmed": true,
  "confidence_score": {bo.get('bo_score','')},
  "confidence_reason": "one sentence on why this breakout is or isn't reliable",
  "news_flag": "none or describe the event",
  "trigger": "exact H1 or M15 condition to confirm entry — must name timeframe",
  "invalid_if": "exact H1 price action that cancels this breakout — must name timeframe",
  "fakeout_warning": "one sentence on the most likely fakeout scenario here",
  "macro_bullets": ["bullet 1 (5-word event | direction | confidence)", "bullet 2", "bullet 3"],
  "mindset": "one sharp psychological trap to avoid on this breakout"
}}"""

# ── Gemini call — 120s timeout, 1 automatic retry ────────────────────────────
def call_gemini(prompt, retries=1):
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    for attempt in range(retries + 1):
        try:
            r      = requests.post(url, json=body, timeout=120)
            result = r.json()
            if "candidates" not in result:
                raise ValueError(f"No candidates in response: {result}")
            raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            if raw.startswith("```"):
                raw = raw.split("\n",1)[1].rsplit("```",1)[0]
            return json.loads(raw), None
        except Exception as e:
            if attempt < retries:
                print(f"    Gemini attempt {attempt+1} failed ({e}), retrying in 3s...")
                time.sleep(3)
            else:
                return None, f"Gemini error after {retries+1} attempts: {str(e)}"

# ── Pre-send sanity checks ────────────────────────────────────────────────────
def run_sanity_checks(data, zone_level, proximity_pct):
    """
    Returns (critical_fail: bool, critical_reason: str, warnings: list[str])
    Critical fail = suppress the alert entirely.
    Warnings = include a red banner at the top of the email.
    """
    critical_fail   = False
    critical_reason = ""
    warnings        = []

    try:
        bias  = data.get("bias","")
        entry = float(str(data.get("entry","0")).split("-")[0].strip() or 0)
        sl    = float(data.get("sl", 0) or 0)
        tp1   = float(data.get("tp1", 0) or 0)
        tp2   = float(data.get("tp2", 0) or 0)
        score = data.get("confidence_score", 0)
        trig  = data.get("trigger","")
        inv   = data.get("invalid_if","")

        # Critical: no NaN or zero in key levels
        if any(v <= 0 for v in [entry, sl, tp1]):
            critical_fail   = True
            critical_reason = f"Critical values missing: entry={entry}, sl={sl}, tp1={tp1}. Alert suppressed."
            return critical_fail, critical_reason, warnings

        # Critical: SL on correct side of entry
        if bias == "LONG" and sl >= entry:
            critical_fail   = True
            critical_reason = f"SL ({sl}) is ABOVE entry ({entry}) on a LONG trade. Alert suppressed — invalid setup."
            return critical_fail, critical_reason, warnings
        if bias == "SHORT" and sl <= entry:
            critical_fail   = True
            critical_reason = f"SL ({sl}) is BELOW entry ({entry}) on a SHORT trade. Alert suppressed — invalid setup."
            return critical_fail, critical_reason, warnings

        # Critical: TP1 gives at least 2:1 RR
        if bias == "LONG":
            rr = (tp1 - entry) / (entry - sl) if (entry - sl) > 0 else 0
        else:
            rr = (entry - tp1) / (sl - entry) if (sl - entry) > 0 else 0
        if rr < 2.0:
            critical_fail   = True
            critical_reason = f"TP1 R:R is {rr:.1f} — below 2:1 minimum. Alert suppressed."
            return critical_fail, critical_reason, warnings

        # Critical: entry near zone level
        if zone_level > 0:
            dist = abs(entry - zone_level) / zone_level * 100
            if dist > proximity_pct * 2:
                critical_fail   = True
                critical_reason = f"Entry ({entry}) is {dist:.2f}% from zone ({zone_level}) — too far. Alert suppressed."
                return critical_fail, critical_reason, warnings

        # Non-critical: TP2 at least 3:1
        if tp2 > 0:
            if bias == "LONG":
                rr2 = (tp2 - entry) / (entry - sl) if (entry - sl) > 0 else 0
            else:
                rr2 = (entry - tp2) / (sl - entry) if (sl - entry) > 0 else 0
            if rr2 < 3.0:
                warnings.append(f"TP2 R:R is {rr2:.1f} — below preferred 3:1 minimum.")

        # Non-critical: confidence at or above min
        min_conf = data.get("_min_confidence", 5)
        if score < min_conf:
            warnings.append(f"Confidence score {score}/10 is below threshold {min_conf}/10.")

        # Non-critical: trigger and invalid_if contain timeframe reference
        tf_keywords = ["H1","M15","D1","H4","1h","15m","daily","hourly"]
        if not any(kw in trig for kw in tf_keywords):
            warnings.append("Trigger condition does not specify a timeframe (H1/M15/D1/H4).")
        if not any(kw in inv for kw in tf_keywords):
            warnings.append("Invalid-if condition does not specify a timeframe (H1/M15/D1/H4).")

    except Exception as e:
        warnings.append(f"Sanity check error (non-critical): {e}")

    return critical_fail, critical_reason, warnings

# ── Chart generation — 150 DPI, returns PNG bytes ─────────────────────────────
def generate_chart(df, title, levels, data, pair_name="", is_breakout=False, bo=None):
    """
    Returns raw PNG bytes at 150 DPI.
    Charts are sent as proper email attachments (not inline/base64).
    Volume bars: shown for BTC, NAS100, GOLD, SILVER only. Hidden for forex.
    Annotations: OB, FVG, BOS, CHoCH, Liquidity Sweep, Premium/Discount zones,
                 Swing Highs/Lows, Entry/SL/TP lines.
    """
    try:
        if df is None or df.empty:
            return None
        df_plot = df.tail(40).copy().reset_index(drop=True)
        for col in ['Open','High','Low','Close']:
            if col not in df_plot.columns:
                return None

        show_volume = pair_name not in FOREX_PAIRS

        if show_volume:
            fig = plt.figure(figsize=(10, 5.5), facecolor='#131722')
            gs  = GridSpec(4, 1, figure=fig, hspace=0.04)
            ax  = fig.add_subplot(gs[:3, 0])
            ax_vol = fig.add_subplot(gs[3, 0], sharex=ax)
            ax_vol.set_facecolor('#131722')
            for s in ax_vol.spines.values():
                s.set_color('#2a2a3e')
        else:
            fig = plt.figure(figsize=(10, 4.5), facecolor='#131722')
            ax  = fig.add_subplot(111)

        ax.set_facecolor('#131722')
        for s in ax.spines.values():
            s.set_color('#2a2a3e')

        n = len(df_plot)

        # ── Candles ──
        for i, row in df_plot.iterrows():
            try:
                o=float(row['Open']); h=float(row['High'])
                l=float(row['Low']);  c=float(row['Close'])
                if any(np.isnan(v) for v in [o,h,l,c]):
                    continue
                col = '#26a69a' if c >= o else '#ef5350'
                ax.plot([i,i],[l,h], color=col, linewidth=0.8, zorder=2)
                body = abs(c-o) or (h-l)*0.01
                ax.add_patch(patches.Rectangle((i-0.35,min(o,c)), 0.7, body,
                    facecolor=col, linewidth=0, alpha=0.9, zorder=3))
            except Exception:
                continue

        # ── Compute price range for premium/discount shading ──
        try:
            all_highs = df_plot['High'].values.flatten().astype(float)
            all_lows  = df_plot['Low'].values.flatten().astype(float)
            p_range_h = float(np.nanmax(all_highs))
            p_range_l = float(np.nanmin(all_lows))
            midpoint  = (p_range_h + p_range_l) / 2

            ax.add_patch(patches.Rectangle(
                (0, midpoint), n, p_range_h - midpoint,
                facecolor='#e74c3c', alpha=0.05, zorder=0, label='Premium'))
            ax.add_patch(patches.Rectangle(
                (0, p_range_l), n, midpoint - p_range_l,
                facecolor='#27ae60', alpha=0.05, zorder=0, label='Discount'))
            ax.text(0.5, p_range_h * 0.9995, 'Premium', color='#e74c3c',
                    fontsize=6, alpha=0.6, va='top')
            ax.text(0.5, p_range_l * 1.0005, 'Discount', color='#27ae60',
                    fontsize=6, alpha=0.6, va='bottom')
        except Exception:
            pass

        # ── Swing Highs and Lows (markers) ──
        try:
            lb = config["zone_detection"]["swing_lookback"]
            hs = df_plot['High'].values.flatten().astype(float)
            ls = df_plot['Low'].values.flatten().astype(float)
            for i in range(lb, n - lb):
                if hs[i] == max(hs[i-lb:i+lb+1]):
                    ax.plot(i, hs[i], marker='v', color='#888888',
                            markersize=5, zorder=5, markeredgewidth=0)
                    ax.text(i, hs[i], ' SH', color='#888888',
                            fontsize=6, va='bottom', zorder=6)
                if ls[i] == min(ls[i-lb:i+lb+1]):
                    ax.plot(i, ls[i], marker='^', color='#888888',
                            markersize=5, zorder=5, markeredgewidth=0)
                    ax.text(i, ls[i], ' SL', color='#888888',
                            fontsize=6, va='top', zorder=6)
        except Exception:
            pass

        # ── OB box ──
        ob_top    = float(data.get('ob_top', 0) or 0)
        ob_bottom = float(data.get('ob_bottom', 0) or 0)
        if ob_top > 0 and ob_bottom > 0 and abs(ob_top - ob_bottom) > 0:
            oc  = '#26a69a' if data.get('ob_type','') == 'bullish' else '#ef5350'
            ok  = data.get('ob_confirmed', True)
            ls  = '-' if ok else '--'
            ax.add_patch(patches.Rectangle(
                (0, ob_bottom), n, ob_top - ob_bottom,
                facecolor=oc, edgecolor=oc, linewidth=1.2,
                alpha=0.20 if ok else 0.08, linestyle=ls, zorder=1))
            ax.text(1, ob_top, f" OB ({data.get('ob_type','')}) {'✓' if ok else '?'}",
                    color=oc, fontsize=7, va='bottom', fontweight='bold', zorder=5)

        # ── FVG box ──
        fvg_top    = float(data.get('fvg_top', 0) or 0)
        fvg_bottom = float(data.get('fvg_bottom', 0) or 0)
        if fvg_top > 0 and fvg_bottom > 0 and abs(fvg_top - fvg_bottom) > 0:
            fok = data.get('fvg_confirmed', True)
            fls = '-' if fok else '--'
            ax.add_patch(patches.Rectangle(
                (0, fvg_bottom), n, fvg_top - fvg_bottom,
                facecolor='#3498db', edgecolor='#3498db', linewidth=1.2,
                alpha=0.18 if fok else 0.07, linestyle=fls, zorder=1))
            ax.text(1, fvg_top, f" FVG ({data.get('fvg_type','')}) {'✓' if fok else '?'}",
                    color='#3498db', fontsize=7, va='bottom', fontweight='bold', zorder=5)

        # ── Breakout-specific annotations ──
        if is_breakout and bo:
            bl = bo.get('broken_level', 0)
            d  = bo.get('direction', 'BULLISH')
            bc = '#26a69a' if d == 'BULLISH' else '#ef5350'
            if bl:
                ax.axhline(y=bl, color=bc, linestyle='-', linewidth=2.0,
                           alpha=0.9, zorder=4, label='Broken Level')
                ax.text(n+0.3, bl, f"Broken Level: {bl:,.5f}",
                        color=bc, fontsize=7, va='center', fontweight='bold', zorder=5)

            rz_top = bo.get('retest_zone_top', 0)
            rz_bot = bo.get('retest_zone_bot', 0)
            if rz_top and rz_bot:
                ax.add_patch(patches.Rectangle(
                    (0, rz_bot), n, rz_top - rz_bot,
                    facecolor=bc, alpha=0.15, zorder=1))
                ax.text(1, rz_top, ' Retest Zone',
                        color=bc, fontsize=6, va='bottom', alpha=0.8, zorder=5)

            # Projected path arrow
            try:
                y_start = float(df_plot['Close'].iloc[-1])
                y_end   = y_start * 1.002 if d == 'BULLISH' else y_start * 0.998
                ax.annotate('', xy=(n+4, y_end), xytext=(n, y_start),
                            arrowprops=dict(arrowstyle='->', color=bc, lw=1.5,
                                           linestyle='dotted'))
            except Exception:
                pass

        # ── Price levels (Entry, SL, TP1, TP2, Zone, Current) ──
        level_cfg = {
            'tp2':     ('#1e8449', '--', 1.0, 'TP2'),
            'tp1':     ('#27ae60', '-',  1.5, 'TP1'),
            'entry':   ('#e67e22', '-',  1.5, 'Entry'),
            'zone':    ('#9b59b6', '--', 1.2, 'Zone'),
            'current': ('#ffffff', ':',  0.8, 'Now'),
            'sl':      ('#e74c3c', '-',  1.5, 'SL'),
        }
        for key, (color, style, width, lbl) in level_cfg.items():
            val = levels.get(key, 0)
            try:
                price = float(str(val).split('-')[0].strip()) if val else 0
            except Exception:
                price = 0
            if price > 0:
                ax.axhline(y=price, color=color, linestyle=style,
                           linewidth=width, alpha=0.85, zorder=4)
                ax.text(n+0.3, price, f"{lbl}: {price:,.5f}",
                        color=color, fontsize=7, va='center',
                        fontweight='bold', zorder=5)

        # ── Volume bars (non-forex only) ──
        if show_volume:
            for i, row in df_plot.iterrows():
                try:
                    vol = float(row.get('Volume', 0) or 0)
                    if np.isnan(vol): vol = 0
                    vc = '#26a69a' if float(row['Close']) >= float(row['Open']) else '#ef5350'
                    ax_vol.bar(i, vol, color=vc, alpha=0.5, width=0.7)
                except Exception:
                    continue
            ax_vol.tick_params(colors='#555', labelsize=6)
            ax_vol.set_ylabel('Vol', color='#555', fontsize=6)
            ax_vol.yaxis.tick_right()
            ax_vol.xaxis.set_visible(False)
            ax_vol.set_xlim(-1, n+12)
            for s in ax_vol.spines.values():
                s.set_color('#2a2a3e')

        ax.set_title(title, color='#dddddd', fontsize=10, pad=6,
                     fontweight='bold', loc='left')
        ax.tick_params(colors='#666', labelsize=7)
        ax.yaxis.tick_right()
        ax.yaxis.set_tick_params(labelcolor='#888')
        ax.xaxis.set_visible(False)
        ax.set_xlim(-1, n+12)

        plt.tight_layout(pad=0.3)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='#131722', edgecolor='none')
        buf.seek(0)
        img_bytes = buf.read()
        plt.close(fig)
        print(f"    Chart ok: {len(img_bytes)//1024}KB")
        return img_bytes
    except Exception as e:
        print(f"    Chart error: {e}")
        plt.close('all')
        return None

# ── Glossary footer ────────────────────────────────────────────────────────────
TERM_DEFINITIONS = {
    "OB":            "OB (Order Block): A candle where institutions placed large buy or sell orders — price tends to react when it returns to this level.",
    "FVG":           "FVG (Fair Value Gap): A price gap between three candles that price often returns to fill — acts like a magnet.",
    "BOS":           "BOS (Break of Structure): Price broke a previous swing high or low, confirming the trend direction.",
    "CHoCH":         "CHoCH (Change of Character): The first sign that the trend may be reversing — price breaks a structure in the opposite direction.",
    "Liquidity":     "Liquidity Sweep: Price briefly spiked beyond a key level to trigger stop-losses before sharply reversing.",
    "Premium":       "Premium Zone: The upper half of a price range — expensive area; prefer shorts here.",
    "Discount":      "Discount Zone: The lower half of a price range — cheap area; prefer longs here.",
    "Breakout":      "Breakout (BOS): When price decisively closes beyond a key level, signalling a potential new trend direction.",
    "Supply":        "Supply Zone: A price area where sellers previously overwhelmed buyers — price may reject here again.",
    "Demand":        "Demand Zone: A price area where buyers previously overwhelmed sellers — price may bounce here again.",
    "R:R":           "R:R (Risk-to-Reward): Ratio of potential profit to potential loss. 2:1 means you risk 1 to make 2.",
    "SL":            "SL (Stop Loss): The price level where the trade is automatically closed to limit your loss.",
    "TP":            "TP (Take Profit): The target price where you plan to close the trade and take your profit.",
}

def build_glossary_footer(text_to_scan):
    """Build a compact glossary of only the terms that appear in this email."""
    found = []
    for key, definition in TERM_DEFINITIONS.items():
        if key in text_to_scan:
            found.append(definition)
    if not found:
        return ""
    items = "".join([
        f'<li style="font-size:11px;color:#888;margin-bottom:4px;">• {d}</li>'
        for d in found
    ])
    return f"""<div style="background:#f8f9fa;padding:14px 18px;border-radius:10px;margin-top:20px;border-top:1px solid #eee;">
  <p style="font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;">QUICK REFERENCE — TERMS USED IN THIS ALERT</p>
  <ul style="list-style:none;padding:0;margin:0;">{items}</ul>
</div>"""

# ── Breakout HTML block ────────────────────────────────────────────────────────
def build_breakout_html_block(bo, gemini_bo=None):
    if not bo:
        return ""
    d      = bo.get("direction", "BULLISH")
    color  = "#26a69a" if d == "BULLISH" else "#ef5350"
    arrow  = "▲" if d == "BULLISH" else "▼"
    score  = bo.get("bo_score", 0)
    sc     = "#27ae60" if score >= 5 else "#f39c12"
    reasons_html = "".join([
        f'<li style="font-size:11px;color:#aaa;margin-bottom:3px;">{r}</li>'
        for r in bo.get("bo_reasons", [])
    ])

    gemini_html = ""
    if gemini_bo:
        trigger_txt  = gemini_bo.get("trigger", "")
        invalid_txt  = gemini_bo.get("invalid_if", "")
        fakeout_txt  = gemini_bo.get("fakeout_warning", "")
        mindset_txt  = gemini_bo.get("mindset", "")
        macro_items  = gemini_bo.get("macro_bullets", [])
        macro_html   = "".join([f'<li style="font-size:12px;color:#ccc;margin-bottom:4px;">• {m}</li>'
                                 for m in macro_items])
        gemini_html  = f"""
  <div style="margin-top:12px;padding-top:12px;border-top:1px solid #2a2a3e;">
    <p style="color:#f39c12;font-size:10px;text-transform:uppercase;margin:0 0 6px;">TRIGGER (Gemini)</p>
    <p style="color:#fff;font-size:12px;background:#1a1a2e;padding:8px 10px;border-radius:6px;border-left:3px solid #f39c12;margin:0 0 10px;">{trigger_txt}</p>
    <p style="color:#e74c3c;font-size:10px;text-transform:uppercase;margin:0 0 6px;">INVALID IF</p>
    <p style="color:#fff;font-size:12px;background:#1a1a2e;padding:8px 10px;border-radius:6px;border-left:3px solid #e74c3c;margin:0 0 10px;">{invalid_txt}</p>
    <p style="color:#e67e22;font-size:10px;text-transform:uppercase;margin:0 0 6px;">FAKEOUT WARNING</p>
    <p style="color:#f39c12;font-size:12px;margin:0 0 10px;">{fakeout_txt}</p>
    <p style="color:#3498db;font-size:10px;text-transform:uppercase;margin:0 0 4px;">MACRO</p>
    <ul style="list-style:none;padding:0;margin:0 0 10px;">{macro_html}</ul>
    <div style="background:#1a1a2e;padding:10px 12px;border-radius:8px;">
      <p style="color:#8899bb;font-size:9px;text-transform:uppercase;margin:0 0 3px;">MINDSET</p>
      <p style="color:white;font-size:12px;margin:0;font-style:italic;">{mindset_txt}</p>
    </div>
  </div>"""

    return f"""<div style="background:#0d1117;border:2px solid {color};border-radius:10px;padding:14px 16px;margin-bottom:20px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <p style="color:{color};font-weight:bold;font-size:14px;margin:0;">{arrow} BREAKOUT ALERT — {d} BOS ({bo.get("timeframe","")})</p>
    <span style="background:{sc};color:white;font-weight:bold;font-size:13px;padding:3px 10px;border-radius:20px;">{score}/5</span>
  </div>
  <p style="color:#aaa;font-size:12px;margin:0 0 10px;">{bo.get("description","")}</p>
  <table style="width:100%;font-size:12px;margin-bottom:8px;color:white;">
    <tr><td style="color:#888;padding:3px 0;width:160px;">Broken Level (H1)</td><td style="font-weight:bold;">{bo.get("broken_level","")}</td></tr>
    <tr><td style="color:#888;padding:3px 0;">Break Candle (H1)</td><td style="font-weight:bold;">{bo.get("break_size_atr","")}x ATR — strong momentum</td></tr>
    <tr><td style="color:#888;padding:3px 0;">Retest Zone</td><td style="font-weight:bold;">{bo.get("retest_zone_bot","")} — {bo.get("retest_zone_top","")}</td></tr>
    <tr><td style="color:#888;padding:3px 0;">Momentum Entry</td><td style="color:{color};font-weight:bold;">{bo.get("momentum_entry","")} &nbsp;<span style="color:#555;font-size:10px;">aggressive — enter now</span></td></tr>
    <tr><td style="color:#888;padding:3px 0;">Retest Entry</td><td style="color:{color};font-weight:bold;">{bo.get("retest_entry","")} &nbsp;<span style="color:#555;font-size:10px;">preferred — wait for pullback to broken level</span></td></tr>
    <tr><td style="color:#888;padding:3px 0;">SL if momentum</td><td style="color:#ef5350;font-weight:bold;">{bo.get("sl_momentum","")}</td></tr>
    <tr><td style="color:#888;padding:3px 0;">SL if retest</td><td style="color:#ef5350;font-weight:bold;">{bo.get("sl_retest","")}</td></tr>
  </table>
  <ul style="list-style:none;padding:0;margin:0 0 6px;">{reasons_html}</ul>
  <p style="color:#856404;background:#fff3cd;padding:7px 10px;border-radius:6px;font-size:11px;margin:8px 0 0;">
    FAKEOUT RULE: If price closes back inside the broken H1 level within 2 H1 candles, the breakout has failed. Do not enter, or exit immediately.
  </p>
  {gemini_html}
</div>"""

# ── Warning banner HTML ────────────────────────────────────────────────────────
def build_warnings_banner(warnings):
    if not warnings:
        return ""
    items = "".join([f'<li style="margin-bottom:4px;">⚠ {w}</li>' for w in warnings])
    return (f'<div style="background:#fef0f0;padding:12px 16px;border-left:4px solid #e74c3c;'
            f'border-radius:6px;margin-bottom:16px;">'
            f'<p style="color:#c0392b;font-weight:bold;font-size:12px;margin:0 0 6px;">ALERT WARNINGS</p>'
            f'<ul style="color:#c0392b;font-size:12px;padding-left:16px;margin:0;">{items}</ul></div>')

# ── Zone email HTML ────────────────────────────────────────────────────────────
def build_zone_email_html(data, pair, zone_level, zone_label, current_price,
                          has_chart1, has_chart2, breakout, timeframe_label="INTRADAY",
                          warnings=None, touch_count=0, depletion_label=""):
    ist_time    = (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M")
    utc_time    = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    score       = data.get("confidence_score", 0)
    bias        = data.get("bias", "WAIT")
    bias_color  = "#e74c3c" if bias=="SHORT" else "#27ae60" if bias=="LONG" else "#f39c12"
    score_color = "#27ae60" if score>=8 else "#f39c12" if score>=6 else "#e74c3c"
    tf_badge    = ("SWING" if timeframe_label=="SWING" else "INTRADAY")
    tf_color    = "#9b59b6" if tf_badge=="SWING" else "#3498db"

    news_flag = data.get("news_flag", "none")
    news_html = (f'<div style="background:#fff3cd;padding:9px 24px;border-left:4px solid #f39c12;'
                 f'font-size:12px;color:#856404;"><b>NEWS:</b> {news_flag}</div>'
                 if news_flag and news_flag.lower() != "none" else "")

    # Macro bullets (max 3)
    macro_items = data.get("macro_bullets", [])
    if not macro_items:
        # Fallback to legacy fields
        ml1 = data.get("macro_line1","")
        ml2 = data.get("macro_line2","")
        macro_items = [x for x in [ml1, ml2] if x]
    macro_html = "".join([
        f'<li style="font-size:13px;color:#444;margin-bottom:6px;line-height:1.5;">• {m}</li>'
        for m in macro_items[:3]
    ]) or '<li style="font-size:13px;color:#aaa;">No material macro driver in the next 4–8 hours.</li>'

    conf_items = "".join([
        f'<li style="margin-bottom:5px;padding:7px 10px;background:#f0fff4;border-radius:6px;font-size:13px;">&#10003; {c}</li>'
        for c in data.get("confluences", [])])
    miss_items = "".join([
        f'<li style="margin-bottom:5px;padding:7px 10px;background:#fff8f0;border-radius:6px;font-size:13px;">&#10007; <b>{m["item"]}</b> — <span style="color:#777;font-style:italic;">{m["reason"]}</span></li>'
        for m in data.get("missing", [])])

    price_map = ""
    try:
        ep   = float(str(data["entry"]).split("-")[0].strip())
        sl_v = float(data.get("sl", 0))
        lvls_map = {"TP2": float(data["tp2"]), "TP1": float(data["tp1"]),
                    "Entry": ep, "Current": float(current_price),
                    "Zone": float(zone_level), "SL": sl_v}
        lc = {"SL":"#e74c3c","Zone":"#9b59b6","Current":"#3498db",
              "Entry":"#e67e22","TP1":"#27ae60","TP2":"#1e8449"}
        vals = [v for v in lvls_map.values() if v>0]
        pmin,pmax = min(vals),max(vals)
        pr = pmax-pmin or 1
        rows = ""
        for lbl, price in sorted([(k,v) for k,v in lvls_map.items() if v>0],
                                   key=lambda x: x[1], reverse=True):
            c   = lc.get(lbl, "#888")
            bar = int(((price-pmin)/pr)*75)+15
            rows += (f'<tr>'
                     f'<td style="padding:5px 10px;color:{c};font-weight:bold;font-size:12px;width:65px;">{lbl}</td>'
                     f'<td style="padding:5px 6px;"><div style="background:{c};height:10px;border-radius:4px;width:{bar}%;"></div></td>'
                     f'<td style="padding:5px 10px;color:{c};font-weight:bold;font-size:12px;text-align:right;white-space:nowrap;">{price:,.5f}</td>'
                     f'</tr>')
        price_map = (f'<h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">PRICE MAP</h3>'
                     f'<table style="width:100%;border-collapse:collapse;background:#f8f9fa;border-radius:8px;margin-bottom:20px;">{rows}</table>')
    except Exception:
        pass

    # Chart note (charts are attachments)
    tf_primary   = "D1" if timeframe_label=="SWING" else "H1"
    tf_secondary = "H4" if timeframe_label=="SWING" else "M15"
    chart1_html = (
        f'<p style="background:#e8f4fd;padding:8px 12px;border-radius:6px;font-size:12px;color:#2980b9;margin:16px 0 4px;">'
        f'📎 {tf_primary} chart attached to this email as chart_h1.png</p>'
        if has_chart1 else
        f'<p style="color:#aaa;font-size:12px;margin:16px 0 4px;">{tf_primary} chart unavailable.</p>'
    )
    chart2_html = (
        f'<p style="background:#e8f4fd;padding:8px 12px;border-radius:6px;font-size:12px;color:#2980b9;margin:4px 0 16px;">'
        f'📎 {tf_secondary} chart attached to this email as chart_m15.png</p>'
        if has_chart2 else
        f'<p style="color:#aaa;font-size:12px;margin:4px 0 16px;">{tf_secondary} chart unavailable.</p>'
    )

    depletion_html = ""
    if depletion_label:
        dep_color = "#e74c3c" if "Heavily" in depletion_label else "#f39c12" if "depletion" in depletion_label else "#27ae60"
        depletion_html = (
            f'<div style="padding:8px 14px;border-radius:6px;border-left:4px solid {dep_color};'
            f'background:#f8f9fa;margin-bottom:12px;font-size:12px;color:{dep_color};">'
            f'<b>Zone Status:</b> {depletion_label}</div>'
        )

    bo_html      = build_breakout_html_block(breakout) if breakout else ""
    warn_html    = build_warnings_banner(warnings or [])

    # Build full content string for glossary scanning
    full_content = json.dumps(data) + zone_label + (data.get("trigger","")) + (data.get("invalid_if",""))
    glossary     = build_glossary_footer(full_content)

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:16px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">

  <div style="background:#1a1a2e;padding:18px 24px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <h2 style="color:white;margin:0;font-size:17px;">ZONE ALERT: {pair} — {zone_label}</h2>
      <span style="background:{tf_color};color:white;font-size:10px;font-weight:bold;padding:2px 8px;border-radius:10px;">{tf_badge}</span>
    </div>
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

    {warn_html}

    <p style="background:#f4f4f8;padding:10px 14px;border-radius:8px;font-size:13px;color:#333;margin:0 0 12px;">
      Zone: <b>{zone_label}</b> at <b>{zone_level}</b> &nbsp;|&nbsp; Now: <b>{current_price}</b>
    </p>

    {depletion_html}

    {bo_html}

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
    <ul style="list-style:none;padding:0;margin:0 0 20px;">{macro_html}</ul>

    <div style="background:#1a1a2e;padding:14px 18px;border-radius:10px;margin-bottom:16px;">
      <p style="color:#8899bb;font-size:10px;margin:0 0 4px;text-transform:uppercase;letter-spacing:1px;">MINDSET</p>
      <p style="color:white;font-size:13px;margin:0;font-style:italic;line-height:1.6;">{data.get("mindset","")}</p>
    </div>

    {glossary}

  </div>
</div>
</body>
</html>"""

# ── Breakout-only email HTML ───────────────────────────────────────────────────
def build_breakout_only_email_html(bo, pair, current_price, gemini_bo=None, has_chart=False, warnings=None):
    ist_time  = (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M")
    utc_time  = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    bo_block  = build_breakout_html_block(bo, gemini_bo)
    warn_html = build_warnings_banner(warnings or [])
    chart_note = (
        '<p style="background:#e8f4fd;padding:8px 12px;border-radius:6px;font-size:12px;'
        'color:#2980b9;margin:0 0 16px;">📎 H1 chart attached to this email as chart_h1.png</p>'
        if has_chart else ""
    )

    full_content = json.dumps(bo) + (json.dumps(gemini_bo) if gemini_bo else "")
    glossary     = build_glossary_footer(full_content)

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:16px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">
  <div style="background:#1a1a2e;padding:18px 24px;">
    <h2 style="color:white;margin:0;font-size:17px;">BREAKOUT ALERT: {pair}</h2>
    <p style="color:#8899bb;margin:5px 0 0;font-size:12px;">{utc_time} UTC | {ist_time} IST &nbsp;|&nbsp; Now: {current_price}</p>
  </div>
  <div style="padding:20px 24px;">
    {warn_html}
    {chart_note}
    {bo_block}
    {glossary}
  </div>
</div>
</body>
</html>"""

# ── Log alert ──────────────────────────────────────────────────────────────────
def log_alert(pair, zone_level, zone_label, current_price, data,
              alert_type="zone_intraday"):
    """
    alert_type must be one of: zone_intraday, zone_swing, breakout
    """
    ist_time = (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M")
    alert_log.append({
        "id":                 f"{pair}_{int(datetime.utcnow().timestamp())}",
        "alert_type":         alert_type,
        "pair":               pair,
        "timestamp_utc":      datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "ist_time":           ist_time,
        "zone_level":         zone_level,
        "zone_label":         zone_label,
        "bias":               data.get("bias","") if data else "",
        "entry":              data.get("entry","") if data else "",
        "sl":                 data.get("sl",0) if data else 0,
        "tp1":                data.get("tp1",0) if data else 0,
        "tp2":                data.get("tp2",0) if data else 0,
        "confidence_score":   data.get("confidence_score",0) if data else 0,
        "confluences":        data.get("confluences",[]) if data else [],
        "outcome":            "pending",
        "outcome_price":      None,
        "outcome_checked_at": None
    })
    save_alert_log()

# ── Send email — charts as proper attachments (MIMEMultipart mixed) ────────────
def send_email(subject, html_body, chart_attachments=None):
    """
    chart_attachments: dict mapping filename → PNG bytes
    e.g. {"chart_h1.png": bytes, "chart_m15.png": bytes}
    Charts sent as proper downloadable attachments — NOT inline/base64/CID.
    Gmail will display them as attached files.
    """
    recipients = config["account"].get("alert_emails", [ALERT_EMAIL])
    for recipient in recipients:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_ADDRESS
        msg["To"]      = recipient
        msg.attach(MIMEText(html_body, "html"))

        if chart_attachments:
            for filename, img_bytes in chart_attachments.items():
                img = MIMEImage(img_bytes, _subtype="png")
                img.add_header("Content-Disposition", "attachment", filename=filename)
                msg.attach(img)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_PASS)
            server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
        print(f"    Sent to {recipient}")

# ── MAIN ───────────────────────────────────────────────────────────────────────
print(f"Alert engine started {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

market_open, market_status = is_market_open()
print(f"  Market: {market_status}")
if not market_open:
    print("  Exiting — outside trading hours.")
    save_alert_log()
    exit(0)

macro_news   = fetch_macro_news()
alerts_fired = 0

# Sequential fetch — yfinance is NOT thread-safe. Never use ThreadPoolExecutor here.
for pair_conf in config["pairs"]:
    name        = pair_conf["name"]
    symbol      = pair_conf["symbol"]
    prox        = pair_conf["proximity_pct"]
    min_touches = pair_conf.get("min_touches", 1)
    min_conf    = pair_conf.get("min_confidence", 5)

    print(f"  Scanning {name}...")

    # ── INTRADAY (H1 + M15) ──────────────────────────────────────────────────
    zones_intra, current_price, df1, df2 = detect_zones_and_candles(symbol, min_touches)

    if current_price is None:
        print(f"    No data / stale data for {name}. Skipping.")
        continue

    breakout = detect_breakout(df1, name) if df1 is not None else None

    if breakout and not should_alert_breakout(name, breakout["broken_level"], current_price, prox):
        print(f"    Breakout suppressed — price hasn't moved enough since last BO alert.")
        breakout = None

    intraday_zone_fired = False

    for zone_level, _ in zones_intra:
        dist_pct = abs(current_price - zone_level) / zone_level * 100
        if dist_pct > prox:
            continue
        if not should_alert_zone(name, zone_level, current_price, prox):
            print(f"    {name} @ {zone_level:.5f} — suppressed (not moved enough).")
            continue

        zone_label   = get_zone_label(zone_level, current_price)
        touch_count  = count_zone_touches_ohlc(df1, zone_level, prox)
        depletion    = zone_depletion_label(touch_count)
        print(f"    INTRADAY ZONE HIT: {name} {zone_label} @ {zone_level:.5f} dist:{dist_pct:.2f}% touches:{touch_count}")

        prompt = build_zone_prompt(name, round(zone_level,5), zone_label,
                                   round(current_price,5), macro_news,
                                   df1, df2, min_conf, touch_count, "INTRADAY")
        data, error = call_gemini(prompt)
        if error:
            print(f"    {error}")
            continue

        score = data.get("confidence_score", 0)
        if not data.get("send_alert", False):
            print(f"    {name} INTRADAY skipped — {score}/10 below {min_conf}. {data.get('confidence_reason','')}")
            continue

        data["_min_confidence"] = min_conf
        critical_fail, critical_reason, warnings = run_sanity_checks(data, zone_level, prox)
        if critical_fail:
            print(f"    CRITICAL SANITY FAIL: {critical_reason}")
            continue

        levels = {'zone': zone_level, 'current': current_price,
                  'entry': data.get('entry',''), 'sl': data.get('sl',0),
                  'tp1': data.get('tp1',0), 'tp2': data.get('tp2',0)}

        chart1_bytes = generate_chart(df1, f"{name} — H1 (INTRADAY)", levels, data, name)
        chart2_bytes = generate_chart(df2, f"{name} — M15 (INTRADAY)", levels, data, name)

        attachments = {}
        if chart1_bytes: attachments["chart_h1.png"]  = chart1_bytes
        if chart2_bytes: attachments["chart_m15.png"] = chart2_bytes

        html = build_zone_email_html(
            data, name, round(zone_level,5), zone_label, round(current_price,5),
            has_chart1=bool(chart1_bytes), has_chart2=bool(chart2_bytes),
            breakout=breakout, timeframe_label="INTRADAY",
            warnings=warnings, touch_count=touch_count, depletion_label=depletion
        )
        subject = f"[INTRADAY {score}/10] {name} | {zone_label} | {round(zone_level,5)} | {datetime.utcnow().strftime('%H:%M')} UTC"
        if breakout:
            subject = f"[INTRADAY SMC {score}/10 + BO {breakout['bo_score']}/5] {name} | {zone_label} | {datetime.utcnow().strftime('%H:%M')} UTC"

        send_email(subject, html, chart_attachments=attachments or None)
        log_alert(name, round(zone_level,5), zone_label, round(current_price,5), data, "zone_intraday")
        record_zone_alert(name, zone_level, current_price)
        if breakout:
            record_breakout_alert(name, breakout["broken_level"], current_price)

        alerts_fired += 1
        intraday_zone_fired = True
        print(f"    Sent INTRADAY: {name} [{score}/10]" + (f" + BO[{breakout['bo_score']}/5]" if breakout else ""))
        break  # one zone email per pair per scan

    # ── SWING (D1 + H4) ──────────────────────────────────────────────────────
    zones_swing, cp_swing, df1_swing, df2_swing = detect_zones_and_candles_swing(symbol, min_touches)

    if cp_swing is not None:
        for zone_level_s, _ in zones_swing:
            dist_pct_s = abs(cp_swing - zone_level_s) / zone_level_s * 100
            if dist_pct_s > prox:
                continue

            # Do not fire swing alert if intraday alert already fired for same zone
            intraday_zone_nearby = any(
                abs(zl - zone_level_s) / zone_level_s * 100 < prox * 0.5
                for zl, _ in zones_intra
                if abs(current_price - zl) / zl * 100 <= prox
            )
            if intraday_zone_fired and intraday_zone_nearby:
                print(f"    {name} SWING zone {zone_level_s:.5f} skipped — intraday alert already fired for same zone.")
                continue

            swing_key = f"{name}_swing_{round(zone_level_s,4)}"
            if not should_alert_zone(swing_key, zone_level_s, cp_swing, prox):
                print(f"    {name} SWING @ {zone_level_s:.5f} — suppressed.")
                continue

            zone_label_s  = get_zone_label(zone_level_s, cp_swing)
            touch_count_s = count_zone_touches_ohlc(df1_swing, zone_level_s, prox)
            depletion_s   = zone_depletion_label(touch_count_s)
            print(f"    SWING ZONE HIT: {name} {zone_label_s} @ {zone_level_s:.5f} dist:{dist_pct_s:.2f}% touches:{touch_count_s}")

            prompt_s = build_zone_prompt(name, round(zone_level_s,5), zone_label_s,
                                         round(cp_swing,5), macro_news,
                                         df1_swing, df2_swing, min_conf,
                                         touch_count_s, "SWING")
            data_s, error_s = call_gemini(prompt_s)
            if error_s:
                print(f"    {error_s}")
                continue

            score_s = data_s.get("confidence_score", 0)
            if not data_s.get("send_alert", False):
                print(f"    {name} SWING skipped — {score_s}/10. {data_s.get('confidence_reason','')}")
                continue

            data_s["_min_confidence"] = min_conf
            cf, cr, warn_s = run_sanity_checks(data_s, zone_level_s, prox)
            if cf:
                print(f"    CRITICAL SANITY FAIL (SWING): {cr}")
                continue

            levels_s = {'zone': zone_level_s, 'current': cp_swing,
                        'entry': data_s.get('entry',''), 'sl': data_s.get('sl',0),
                        'tp1': data_s.get('tp1',0), 'tp2': data_s.get('tp2',0)}

            chart1_s = generate_chart(df1_swing, f"{name} — D1 (SWING)", levels_s, data_s, name)
            chart2_s = generate_chart(df2_swing, f"{name} — H4 (SWING)", levels_s, data_s, name)

            att_s = {}
            if chart1_s: att_s["chart_h1.png"]  = chart1_s
            if chart2_s: att_s["chart_m15.png"] = chart2_s

            html_s  = build_zone_email_html(
                data_s, name, round(zone_level_s,5), zone_label_s, round(cp_swing,5),
                has_chart1=bool(chart1_s), has_chart2=bool(chart2_s),
                breakout=None, timeframe_label="SWING",
                warnings=warn_s, touch_count=touch_count_s, depletion_label=depletion_s
            )
            subj_s = f"[SWING {score_s}/10] {name} | {zone_label_s} | {round(zone_level_s,5)} | {datetime.utcnow().strftime('%H:%M')} UTC"
            send_email(subj_s, html_s, chart_attachments=att_s or None)
            log_alert(name, round(zone_level_s,5), zone_label_s, round(cp_swing,5), data_s, "zone_swing")
            record_zone_alert(swing_key, zone_level_s, cp_swing)
            alerts_fired += 1
            print(f"    Sent SWING: {name} [{score_s}/10]")
            break

    # ── Standalone breakout email (only if no zone alert fired for this pair) ──
    if breakout and not intraday_zone_fired:
        # Call Gemini for breakout analysis
        bo_prompt   = build_breakout_prompt(name, breakout, round(current_price,5), macro_news, df1)
        gemini_bo, bo_err = call_gemini(bo_prompt)
        if bo_err:
            print(f"    Gemini breakout analysis failed: {bo_err}")
            gemini_bo = None

        chart_bo_bytes = generate_chart(
            df1, f"{name} — H1 (BREAKOUT)", {
                'current': current_price,
                'entry': breakout.get('momentum_entry',0),
                'sl': breakout.get('sl_momentum',0),
                'zone': breakout.get('broken_level',0)
            }, {}, name, is_breakout=True, bo=breakout
        )

        att_bo = {}
        if chart_bo_bytes: att_bo["chart_h1.png"] = chart_bo_bytes

        html_bo = build_breakout_only_email_html(
            breakout, name, round(current_price,5),
            gemini_bo=gemini_bo, has_chart=bool(chart_bo_bytes)
        )
        subj_bo = (f"[BO {breakout['bo_score']}/5] {name} | {breakout['direction']} Breakout | "
                   f"{round(breakout['broken_level'],5)} | {datetime.utcnow().strftime('%H:%M')} UTC")
        send_email(subj_bo, html_bo, chart_attachments=att_bo or None)
        log_alert(name, breakout["broken_level"], f"{breakout['direction']} Breakout",
                  round(current_price,5), None, "breakout")
        record_breakout_alert(name, breakout["broken_level"], current_price)
        alerts_fired += 1
        print(f"    Sent standalone breakout: {name} [{breakout['bo_score']}/5]")

save_alert_log()
print(f"Log saved: {len(alert_log)} total entries.")
print(f"Scan complete. {alerts_fired} alert(s) fired.")
