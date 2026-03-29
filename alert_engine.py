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
try:
    with open(ALERT_LOG_FILE) as f:
        alert_log = json.load(f)
    print(f"  Loaded {len(alert_log)} existing log entries")
except:
    alert_log = []

def save_alert_log():
    with open(ALERT_LOG_FILE, "w") as f:
        json.dump(alert_log, f, indent=2)

# ── Zone visit state ──────────────────────────────────────────────────────────
# No cooldown hours. No time tracking.
# A zone re-alerts ONLY when price has moved more than 1.5x proximity_pct
# away from zone since the last alert on that zone. That is the only rule.
VISIT_FILE = "zone_visit_state.json"
try:
    with open(VISIT_FILE) as f:
        visit_state = json.load(f)
except:
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

# All pairs use intraday: H1 primary, M15 secondary
def detect_zones_and_candles(symbol, min_touches):
    df1 = clean_df(yf.download(symbol, period="15d", interval="1h",  progress=False))
    df2 = clean_df(yf.download(symbol, period="5d",  interval="15m", progress=False))

    if df1 is None:
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

# ── Breakout detection — scored out of 5 ─────────────────────────────────────
# Break Quality (2pts): +1 body closes beyond level | +1 break candle >= 1.5x ATR14
# Volume      (1pt):    +1 volume >= 1.3x 20-period avg (gracefully skipped for forex)
# Level Sig.  (1pt):    +1 level had 3+ prior touches
# Trend Align (1pt):    +1 break direction matches 50-period MA
# Minimum score to send: 3/5
def detect_breakout(df1):
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

        last    = len(df1) - 2
        if last < 20:
            return None

        l_open  = opens[last]
        l_close = closes[last]
        l_high  = highs[last]
        l_low   = lows[last]
        l_range = l_high - l_low

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
            if lows[i] == min(lows[max(0,i-lb):i+lb+1]):
                touches = sum(1 for j in range(last)
                              if abs(lows[j]-lows[i])/lows[i] < 0.002)
                sig_lows.append((float(lows[i]), touches))

        ma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else float(np.mean(closes))
        buf  = atr * 0.3

        def score_breakout(direction, level, touches, body_ok):
            score     = 0
            reasons   = []
            scorecard = []

            # Break Quality — 2pts
            bq_s = 0
            if body_ok:
                bq_s   = 2
                score += 2
                bq_det = (f"The H1 candle body fully closed "
                          f"{'above' if direction=='BULLISH' else 'below'} {round(level,5)}. "
                          f"The candle was {round(l_range/atr,1)}x the average H1 candle size — "
                          f"showing {'strong buying' if direction=='BULLISH' else 'strong selling'} momentum.")
                reasons.append(f"+2 H1 body closed {'above' if direction=='BULLISH' else 'below'} {round(level,5)}, candle {round(l_range/atr,1)}x ATR")
            else:
                bq_det = "The H1 candle body did not fully close beyond the level. Weak break — price may snap back inside."
                reasons.append("+0 Body did not fully close beyond level")
            scorecard.append({"criterion": "Break Quality", "score": bq_s, "max": 2, "detail": bq_det})

            # Volume — 1pt
            vol_s   = 0
            vol_det = ""
            if 'Volume' in df1.columns:
                try:
                    vols    = df1['Volume'].values.flatten().astype(float)
                    avg_vol = float(np.nanmean(vols[-21:-1]))
                    if avg_vol > 0 and vols[last] >= 1.3 * avg_vol:
                        vol_s   = 1
                        score  += 1
                        ratio   = round(vols[last]/avg_vol, 1)
                        vol_det = (f"Volume on the H1 break candle was {ratio}x the 20-period average. "
                                   f"Higher volume means large institutions were behind this move — "
                                   f"increasing the probability the break is genuine.")
                        reasons.append(f"+1 Volume {ratio}x average (institutional)")
                    else:
                        ratio   = round(vols[last]/avg_vol, 1) if avg_vol > 0 else 0
                        vol_det = (f"Volume on the H1 break candle was only {ratio}x average (need 1.3x or higher). "
                                   f"No institutional confirmation visible on this break. "
                                   f"Be extra cautious — retail-driven breaks fail more often.")
                        reasons.append("+0 Volume below 1.3x average")
                except:
                    vol_det = "Volume data could not be read for this instrument."
                    reasons.append("+0 Volume unavailable")
            else:
                vol_det = ("Volume is not tracked for this forex/commodity pair by our data source. "
                           "This criterion is skipped — score is neutral, not negative.")
                reasons.append("+0 Volume not tracked for this pair (forex/commodity)")
            scorecard.append({"criterion": "Volume Confirmation", "score": vol_s, "max": 1, "detail": vol_det})

            # Level Significance — 1pt
            ls_s = 0
            if touches >= 3:
                ls_s   = 1
                score += 1
                ls_det = (f"This H1 level was tested and held {touches} times before finally breaking. "
                          f"The more times a level holds, the more orders are stacked there — "
                          f"making the eventual break more powerful and significant.")
                reasons.append(f"+1 Level tested {touches} times before (well-established)")
            else:
                ls_det = (f"This H1 level only had {touches} prior touch(es). We need at least 3 touches "
                          f"for a level to be considered well-established. Fewer touches = fewer orders stacked "
                          f"= weaker break confirmation.")
                reasons.append(f"+0 Level only {touches} prior touch(es) — needs 3+")
            scorecard.append({"criterion": "Level Significance", "score": ls_s, "max": 1, "detail": ls_det})

            # Trend Alignment — 1pt
            aligned = (closes[-1] > ma50 if direction == "BULLISH" else closes[-1] < ma50)
            ta_s    = 0
            if aligned:
                ta_s   = 1
                score += 1
                ta_det = (f"The {'bullish' if direction=='BULLISH' else 'bearish'} break direction matches "
                          f"the 50-period Moving Average trend on H1. "
                          f"Trading with the broader trend improves the probability of a sustained move.")
                reasons.append("+1 Break direction matches 50-period MA trend")
            else:
                ta_det = (f"This is a counter-trend break — price is on the wrong side of the H1 50-period "
                          f"Moving Average. Counter-trend breakouts fail more often. "
                          f"Extra caution required — prefer smaller position size.")
                reasons.append("+0 Counter-trend break — price on wrong side of 50-MA")
            scorecard.append({"criterion": "Trend Alignment", "score": ta_s, "max": 1, "detail": ta_det})

            return score, reasons, scorecard

        # Bullish BOS
        if l_close > l_open:
            candidates = [(lvl,tc) for lvl,tc in sig_highs
                          if l_close > lvl and l_open <= lvl * 1.005]
            if candidates:
                level, touches = max(candidates, key=lambda x: x[0])
                score, reasons, scorecard = score_breakout("BULLISH", level, touches, True)
                if score >= 4:
                    _re   = round(level+buf, 5)
                    _sl   = round(level-buf*2, 5)
                    _dist = round(abs(_re - _sl), 5)
                    _tp1  = round(_re + 2.0 * _dist, 5)
                    _tp2  = round(_re + 3.0 * _dist, 5)
                    return {
                        "direction":            "BULLISH",
                        "timeframe":            "H1",
                        "broken_level":         round(level, 5),
                        "break_size_atr":       round(l_range/atr, 1),
                        "level_touches":        touches,
                        "bo_score":             score,
                        "bo_reasons":           reasons,
                        "bo_scorecard":         scorecard,
                        "retest_zone_top":      round(level+buf, 5),
                        "retest_zone_bot":      round(level-buf, 5),
                        "retest_entry":         _re,
                        "sl_retest":            _sl,
                        "sl_distance_retest":   _dist,
                        "tp1_retest":           _tp1,
                        "tp2_retest":           _tp2,
                        "rr_tp1":               "2.0",
                        "rr_tp2":               "3.0",
                        "trigger_text":         (f"Wait for price to pull back into the retest zone "
                                                 f"({round(level-buf,5)} – {round(level+buf,5)}). "
                                                 f"Enter long after a bullish confirmation candle on the H1 or M15 timeframe — "
                                                 f"look for a bullish engulfing candle, a pin bar, or an M15 Change of Character (CHoCH) to the upside."),
                        "invalid_if_text":      (f"Any H1 candle closes back below {round(level,5)} (the broken level). "
                                                 f"This signals a fakeout — the breakout has failed and you should not enter, "
                                                 f"or exit immediately if already in the trade."),
                        "description":          f"Bullish BOS on H1: body closed {round((l_close-level)/level*100,3)}% above {round(level,5)}"
                    }

        # Bearish BOS
        elif l_close < l_open:
            candidates = [(lvl,tc) for lvl,tc in sig_lows
                          if l_close < lvl and l_open >= lvl * 0.995]
            if candidates:
                level, touches = min(candidates, key=lambda x: x[0])
                score, reasons, scorecard = score_breakout("BEARISH", level, touches, True)
                if score >= 4:
                    _re   = round(level-buf, 5)
                    _sl   = round(level+buf*2, 5)
                    _dist = round(abs(_sl - _re), 5)
                    _tp1  = round(_re - 2.0 * _dist, 5)
                    _tp2  = round(_re - 3.0 * _dist, 5)
                    return {
                        "direction":            "BEARISH",
                        "timeframe":            "H1",
                        "broken_level":         round(level, 5),
                        "break_size_atr":       round(l_range/atr, 1),
                        "level_touches":        touches,
                        "bo_score":             score,
                        "bo_reasons":           reasons,
                        "bo_scorecard":         scorecard,
                        "retest_zone_top":      round(level+buf, 5),
                        "retest_zone_bot":      round(level-buf, 5),
                        "retest_entry":         _re,
                        "sl_retest":            _sl,
                        "sl_distance_retest":   _dist,
                        "tp1_retest":           _tp1,
                        "tp2_retest":           _tp2,
                        "rr_tp1":               "2.0",
                        "rr_tp2":               "3.0",
                        "trigger_text":         (f"Wait for price to pull back into the retest zone "
                                                 f"({round(level-buf,5)} – {round(level+buf,5)}). "
                                                 f"Enter short after a bearish confirmation candle on the H1 or M15 timeframe — "
                                                 f"look for a bearish engulfing candle, a pin bar, or an M15 Change of Character (CHoCH) to the downside."),
                        "invalid_if_text":      (f"Any H1 candle closes back above {round(level,5)} (the broken level). "
                                                 f"This signals a fakeout — the breakout has failed and you should not enter, "
                                                 f"or exit immediately if already in the trade."),
                        "description":          f"Bearish BOS on H1: body closed {round((level-l_close)/level*100,3)}% below {round(level,5)}"
                    }

        return None

    except Exception as e:
        print(f"    Breakout detection error: {e}")
        return None

# ── Geopolitical context detection (breakout alerts only) ─────────────────────
# Zone alerts: Gemini reads full news and returns geo_flag directly in its JSON.
# Breakout alerts: no Gemini call, so we scan full headline lines for phrases.
# We match on PHRASES (not single words) to avoid false positives like "oil prices".
GEO_PHRASES = [
    "military operation", "air strike", "airstrike", "missile attack", "missile strike",
    "troops deployed", "troops advancing", "war declared", "invasion", "ceasefire",
    "sanctions imposed", "new sanctions", "oil field", "oil supply", "pipeline attack",
    "port blockade", "energy embargo", "trade war", "tariff imposed", "tariff hike",
    "nuclear threat", "nato response", "nato deployment", "conflict escalat",
    "coup attempt", "government collapse", "regime change", "political crisis",
    "terrorist attack", "bombing campaign", "blockade", "geopolit",
    "supply disruption", "refugee crisis", "humanitarian crisis",
    "armed conflict", "cross-border", "warship", "naval blockade",
]

def detect_geo_flag_phrases(news_text):
    """
    Contextual phrase scan for breakout alerts where Gemini is not called.
    Checks each full headline line — not individual words — to reduce false positives.
    Returns True if any headline contains a recognised geopolitical phrase.
    """
    if not news_text or news_text.strip() == "Macro news unavailable.":
        return False
    lower = news_text.lower()
    for line in lower.splitlines():
        line = line.strip()
        if not line:
            continue
        for phrase in GEO_PHRASES:
            if phrase in line:
                return True
    return False

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
        if alert.get('alert_type') not in ('zone', 'zone_intraday', 'breakout'):
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

# ── Breakout HTML block ───────────────────────────────────────────────────────
def build_breakout_html_block(bo):
    if not bo:
        return ""
    d      = bo.get("direction","BULLISH")
    color  = "#26a69a" if d=="BULLISH" else "#ef5350"
    arrow  = "▲" if d=="BULLISH" else "▼"
    score  = bo.get("bo_score",0)
    sc     = "#27ae60" if score>=4 else "#f39c12"

    scorecard_rows = ""
    for item in bo.get("bo_scorecard",[]):
        s    = item["score"]
        m    = item["max"]
        icon = "✓" if s==m else ("~" if s>0 else "✗")
        ic   = "#27ae60" if s==m else ("#f39c12" if s>0 else "#e74c3c")
        scorecard_rows += (
            f'<tr style="border-bottom:1px solid #2a2a3e;">'
            f'<td style="padding:7px 10px;color:#aaa;font-size:11px;width:160px;">{item["criterion"]}</td>'
            f'<td style="padding:7px 8px;text-align:center;width:50px;"><span style="color:{ic};font-weight:bold;font-size:11px;">{icon} {s}/{m}</span></td>'
            f'<td style="padding:7px 10px;color:#666;font-size:10px;line-height:1.5;">{item["detail"]}</td>'
            f'</tr>'
        )

    return f"""<div style="background:#0d1117;border:2px solid {color};border-radius:10px;padding:14px 16px;margin-bottom:20px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <p style="color:{color};font-weight:bold;font-size:13px;margin:0;">{arrow} BREAKOUT — {d} BOS (H1)</p>
    <span style="background:{sc};color:white;font-weight:bold;font-size:12px;padding:2px 9px;border-radius:20px;">{score}/5</span>
  </div>
  <p style="color:#666;font-size:11px;margin:0 0 10px;">{bo.get("description","")}</p>

  <table style="width:100%;border-collapse:collapse;margin-bottom:12px;">
    <tr style="border-bottom:1px solid #2a2a3e;">
      <td style="padding:7px 10px;color:#666;font-size:11px;width:160px;">H1 Level Broken</td>
      <td style="padding:7px 10px;color:white;font-weight:bold;font-size:12px;">{bo.get("broken_level","")}</td>
    </tr>
    <tr style="border-bottom:1px solid #2a2a3e;">
      <td style="padding:7px 10px;color:#666;font-size:11px;">Retest Zone</td>
      <td style="padding:7px 10px;color:white;font-weight:bold;font-size:12px;">{bo.get("retest_zone_bot","")} – {bo.get("retest_zone_top","")}</td>
    </tr>
    <tr style="border-bottom:1px solid #2a2a3e;">
      <td style="padding:7px 10px;color:#666;font-size:11px;">Retest Entry</td>
      <td style="padding:7px 10px;color:{color};font-weight:bold;font-size:12px;">{bo.get("retest_entry","")}</td>
    </tr>
    <tr style="border-bottom:1px solid #2a2a3e;">
      <td style="padding:7px 10px;color:#666;font-size:11px;">Stop Loss</td>
      <td style="padding:7px 10px;color:#ef5350;font-weight:bold;font-size:12px;">{bo.get("sl_retest","")}</td>
    </tr>
    <tr style="border-bottom:1px solid #2a2a3e;">
      <td style="padding:7px 10px;color:#666;font-size:11px;">TP1 (R:R {bo.get("rr_tp1","2.0")})</td>
      <td style="padding:7px 10px;color:#27ae60;font-weight:bold;font-size:12px;">{bo.get("tp1_retest","")}</td>
    </tr>
    <tr>
      <td style="padding:7px 10px;color:#666;font-size:11px;">TP2 (R:R {bo.get("rr_tp2","3.0")})</td>
      <td style="padding:7px 10px;color:#1e8449;font-weight:bold;font-size:12px;">{bo.get("tp2_retest","")}</td>
    </tr>
  </table>

  <p style="color:#555;font-size:9px;text-transform:uppercase;letter-spacing:0.5px;margin:0 0 4px;">SCORECARD</p>
  <table style="width:100%;border-collapse:collapse;margin-bottom:10px;">{scorecard_rows}</table>

  <div style="background:#0a1a0a;border-left:3px solid #27ae60;padding:8px 10px;border-radius:4px;margin-bottom:8px;">
    <p style="color:#555;font-size:9px;text-transform:uppercase;margin:0 0 3px;">WHEN TO ENTER</p>
    <p style="color:#a3e6b3;font-size:11px;margin:0;line-height:1.5;">{bo.get("trigger_text","Wait for a confirmation candle in the retest zone.")}</p>
  </div>
  <div style="background:#1a0a0a;border-left:3px solid #ef5350;padding:8px 10px;border-radius:4px;">
    <p style="color:#555;font-size:9px;text-transform:uppercase;margin:0 0 3px;">TRADE IS INVALID IF</p>
    <p style="color:#f5a0a0;font-size:11px;margin:0;line-height:1.5;">{bo.get("invalid_if_text","Price closes back inside the broken level.")}</p>
  </div>
</div>"""

# ── Zone email HTML ───────────────────────────────────────────────────────────
def build_zone_email_html(data, pair, zone_level, zone_label, current_price,
                          chart1_b64, chart2_b64, breakout):
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

    bo_html = build_breakout_html_block(breakout) if breakout else ""

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
    <p style="font-size:13px;color:#444;margin:0 0 20px;line-height:1.7;">{data.get("macro_line1","")}<br>{data.get("macro_line2","")}</p>

    <div style="background:#1a1a2e;padding:14px 18px;border-radius:10px;">
      <p style="color:#8899bb;font-size:10px;margin:0 0 4px;text-transform:uppercase;letter-spacing:1px;">MINDSET</p>
      <p style="color:white;font-size:13px;margin:0;font-style:italic;line-height:1.6;">{data.get("mindset","")}</p>
    </div>

  </div>
</div>
</body>
</html>"""

# ── Breakout-only email HTML ──────────────────────────────────────────────────
def build_breakout_only_email_html(bo, pair, current_price, chart_b64=None):
    ist_now     = datetime.utcnow() + timedelta(hours=5, minutes=30)
    ist_time    = ist_now.strftime("%H:%M")
    ist_date    = ist_now.strftime("%d %b %Y")
    risk_amount = config["account"]["balance"] * config["account"]["risk_percent"] / 100
    risk_pct    = config["account"]["risk_percent"]

    d           = bo.get("direction", "BULLISH")
    color       = "#26a69a" if d == "BULLISH" else "#ef5350"
    arrow       = "▲" if d == "BULLISH" else "▼"
    score       = bo.get("bo_score", 0)
    score_color = "#27ae60" if score >= 4 else "#f39c12"
    score_label = ("Strong breakout — retest entry advised." if score >= 4
                   else "Moderate breakout — be selective. Wait for a clean retest confirmation candle.")

    # ── Scorecard table ────────────────────────────────────────────────────────
    scorecard_rows = ""
    for item in bo.get("bo_scorecard", []):
        s      = item["score"]
        m      = item["max"]
        icon   = "✓" if s == m else ("~" if s > 0 else "✗")
        ic     = "#27ae60" if s == m else ("#f39c12" if s > 0 else "#e74c3c")
        border = "border-bottom:1px solid #2a2a3e;"
        scorecard_rows += (
            f'<tr style="{border}">'
            f'<td style="padding:10px 14px;color:#ccc;font-size:12px;font-weight:bold;width:170px;">{item["criterion"]}</td>'
            f'<td style="padding:10px 10px;text-align:center;width:55px;">'
            f'<span style="color:{ic};font-weight:bold;font-size:13px;">{icon} {s}/{m}</span></td>'
            f'<td style="padding:10px 14px;color:#aaa;font-size:11px;line-height:1.6;">{item["detail"]}</td>'
            f'</tr>'
        )

    # ── Trade levels ───────────────────────────────────────────────────────────
    entry       = bo.get("retest_entry", "—")
    sl          = bo.get("sl_retest", "—")
    tp1         = bo.get("tp1_retest", "—")
    tp2         = bo.get("tp2_retest", "—")
    broken      = bo.get("broken_level", "—")
    r_bot       = bo.get("retest_zone_bot", "—")
    r_top       = bo.get("retest_zone_top", "—")
    sl_dist     = bo.get("sl_distance_retest", 0)
    rr1         = bo.get("rr_tp1", "2.0")
    rr2         = bo.get("rr_tp2", "3.0")
    trigger     = bo.get("trigger_text", "Wait for a confirmation candle in the retest zone before entering.")
    invalid_if  = bo.get("invalid_if_text", f"Price closes back inside the broken level {broken}.")

    # ── Price map ──────────────────────────────────────────────────────────────
    price_map_html = ""
    try:
        lvls = {"SL": float(sl), "Entry": float(entry), "TP1": float(tp1), "TP2": float(tp2)}
        lc   = {"SL": "#ef5350", "Entry": color, "TP1": "#27ae60", "TP2": "#1e8449"}
        vals = [v for v in lvls.values() if v > 0]
        if vals:
            pmin, pmax = min(vals), max(vals)
            pr = pmax - pmin or 1
            rows_pm = ""
            for lbl, price in sorted(lvls.items(), key=lambda x: x[1], reverse=True):
                if price <= 0:
                    continue
                c   = lc.get(lbl, "#888")
                bar = int(((price - pmin) / pr) * 75) + 15
                rows_pm += (
                    f'<tr><td style="padding:5px 10px;color:{c};font-weight:bold;font-size:12px;width:55px;">{lbl}</td>'
                    f'<td style="padding:5px 6px;"><div style="background:{c};height:9px;border-radius:3px;width:{bar}%;"></div></td>'
                    f'<td style="padding:5px 10px;color:{c};font-weight:bold;font-size:12px;text-align:right;">{price:,.5f}</td></tr>'
                )
            price_map_html = (
                f'<h3 style="color:#8899bb;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;">PRICE MAP</h3>'
                f'<table style="width:100%;border-collapse:collapse;background:#0d1117;border-radius:8px;margin-bottom:20px;">{rows_pm}</table>'
            )
    except:
        pass

    # ── Chart ──────────────────────────────────────────────────────────────────
    chart_html = ""
    if chart_b64:
        chart_html = (
            f'<h3 style="color:#8899bb;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;">'
            f'H1 CHART — RETEST LEVELS</h3>'
            f'<img src="cid:chart_h1" style="width:100%;border-radius:8px;display:block;margin-bottom:20px;" />'
        )

    # ── Glossary ───────────────────────────────────────────────────────────────
    glossary = [
        ("BOS — Break of Structure",
         "Price breaks through a previous swing high (bullish) or swing low (bearish) that was holding as a key level. "
         "This confirms a shift in market direction."),
        ("ATR — Average True Range",
         f"Measures how much {pair} moves on average per H1 candle. "
         "A break of 1.6x ATR means the break candle was 60% larger than the average candle — unusually strong move."),
        ("Retest Entry",
         "After breaking a level, price often comes back to 'test' that same level again before continuing in the break direction. "
         "Entering at the retest is safer than chasing price immediately after the break."),
        ("Retest Zone",
         "The price range just around the broken level where the retest is expected to happen. "
         "This is where you watch for a confirmation candle before entering."),
        ("Fakeout",
         "When price appears to break a level but then reverses back inside it — a false signal. "
         "The fakeout rule protects you: if price closes back inside the level, the break has failed, do not enter."),
        ("50-Period Moving Average (50 MA)",
         "The average closing price of the last 50 H1 candles. Price above this line = broader uptrend. "
         "Price below = broader downtrend. Trading in the same direction as this average improves probability."),
        ("Volume Confirmation",
         "The number of contracts/transactions during the break candle. "
         "High volume (1.3x+ the 20-period average) means large institutions drove the move — more reliable. "
         "Low volume breaks fail more often."),
        ("CHoCH — Change of Character",
         "A small-scale shift in price direction on the M15 (15-minute) timeframe, used as a confirmation trigger. "
         "Example: in a bearish setup, a CHoCH to the downside on M15 confirms the retest is over and the sell continuation has begun."),
        ("R:R — Risk to Reward Ratio",
         "How much you can potentially win compared to how much you risk. "
         "R:R 2:1 means: risk 1 unit to potentially gain 2 units. Higher R:R = more profit for the same risk."),
        ("Bearish / Bullish",
         "Bearish = price is expected to fall (short trade). Bullish = price is expected to rise (long trade)."),
    ]
    glossary_html = "".join([
        f'<div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #1a2030;">'
        f'<p style="font-size:11px;font-weight:bold;color:#8899bb;margin:0 0 3px;">{term}</p>'
        f'<p style="font-size:11px;color:#666;margin:0;line-height:1.5;">{defn}</p>'
        f'</div>'
        for term, defn in glossary
    ])

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#0a0e1a;padding:16px;margin:0;">
<div style="max-width:640px;margin:auto;background:#131722;border-radius:14px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.5);">

  <!-- ── HEADER ── -->
  <div style="background:#0d1117;padding:20px 24px;border-bottom:2px solid {color};">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
      <div>
        <h2 style="color:white;margin:0;font-size:18px;font-weight:bold;">{arrow} BREAKOUT ALERT: {pair}</h2>
        <p style="color:#8899bb;margin:5px 0 2px;font-size:12px;">{ist_date} &nbsp;|&nbsp; {ist_time} IST &nbsp;|&nbsp; Price now: <b style="color:white;">{current_price}</b></p>
        <p style="color:{color};font-size:12px;margin:4px 0 0;">{bo.get("description","")}</p>
      </div>
      <div style="text-align:center;flex-shrink:0;margin-left:16px;">
        <div style="background:{score_color};color:white;font-weight:bold;font-size:18px;padding:8px 16px;border-radius:20px;">{score}/5</div>
        <p style="color:#888;font-size:9px;margin:4px 0 0;text-transform:uppercase;">Score</p>
      </div>
    </div>
    <p style="color:#888;font-size:11px;margin:10px 0 0;font-style:italic;">{score_label}</p>
  </div>

  <!-- ── SETUP SCORECARD ── -->
  <div style="padding:20px 24px;border-bottom:1px solid #2a2a3e;">
    <h3 style="color:#8899bb;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin:0 0 12px;">SETUP SCORECARD — WHY THIS BREAKOUT QUALIFIES</h3>
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr style="background:#0d1117;">
          <th style="padding:8px 14px;text-align:left;color:#555;font-size:9px;text-transform:uppercase;font-weight:normal;">Criterion</th>
          <th style="padding:8px 10px;text-align:center;color:#555;font-size:9px;text-transform:uppercase;font-weight:normal;">Score</th>
          <th style="padding:8px 14px;text-align:left;color:#555;font-size:9px;text-transform:uppercase;font-weight:normal;">What This Means For This Trade</th>
        </tr>
      </thead>
      <tbody>{scorecard_rows}</tbody>
    </table>
  </div>

  <!-- ── FULL TRADE PLAN ── -->
  <div style="padding:20px 24px;border-bottom:1px solid #2a2a3e;">
    <h3 style="color:#8899bb;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin:0 0 14px;">FULL TRADE PLAN — RETEST ENTRY ONLY</h3>

    <!-- Broken level + retest zone context -->
    <div style="background:#0d1117;border-radius:8px;padding:12px 16px;margin-bottom:16px;">
      <p style="color:#555;font-size:9px;text-transform:uppercase;margin:0 0 8px;letter-spacing:0.5px;">WHAT HAPPENED &amp; WHERE TO LOOK</p>
      <p style="color:white;font-size:13px;margin:0;">H1 level broken: <b style="color:{color};">{broken}</b></p>
      <p style="color:white;font-size:13px;margin:6px 0 0;">Retest zone: <b style="color:{color};">{r_bot} – {r_top}</b></p>
      <p style="color:#666;font-size:11px;margin:8px 0 0;line-height:1.5;">
        Price has broken this H1 level. It is now expected to pull back and retest it. 
        The retest zone is the price range where you wait — do NOT enter before price reaches this zone.
      </p>
    </div>

    <!-- Trade levels -->
    <table style="width:100%;border-collapse:collapse;">
      <tr style="border-bottom:1px solid #2a2a3e;">
        <td style="padding:11px 0;color:#666;font-size:12px;width:160px;">Entry — Retest</td>
        <td style="padding:11px 0;color:{color};font-weight:bold;font-size:15px;">{entry}</td>
      </tr>
      <tr style="border-bottom:1px solid #2a2a3e;">
        <td style="padding:11px 0;color:#666;font-size:12px;">Stop Loss</td>
        <td style="padding:11px 0;">
          <span style="color:#ef5350;font-weight:bold;font-size:15px;">{sl}</span>
          <span style="color:#555;font-size:11px;margin-left:8px;">({sl_dist} pts from entry)</span>
        </td>
      </tr>
      <tr style="border-bottom:1px solid #2a2a3e;">
        <td style="padding:11px 0;color:#666;font-size:12px;">TP1 &nbsp;<span style="color:#555;font-size:10px;">R:R {rr1}</span></td>
        <td style="padding:11px 0;color:#27ae60;font-weight:bold;font-size:15px;">{tp1}</td>
      </tr>
      <tr style="border-bottom:1px solid #2a2a3e;">
        <td style="padding:11px 0;color:#666;font-size:12px;">TP2 &nbsp;<span style="color:#555;font-size:10px;">R:R {rr2}</span></td>
        <td style="padding:11px 0;color:#1e8449;font-weight:bold;font-size:15px;">{tp2}</td>
      </tr>
      <tr>
        <td style="padding:11px 0;color:#666;font-size:12px;">Risk Amount</td>
        <td style="padding:11px 0;color:white;font-weight:bold;font-size:13px;">${risk_amount:.0f} &nbsp;<span style="color:#555;font-size:10px;">({risk_pct}% of account — adjust lot size at your broker accordingly)</span></td>
      </tr>
    </table>
  </div>

  <!-- ── PRICE MAP ── -->
  <div style="padding:20px 24px;border-bottom:1px solid #2a2a3e;">
    {price_map_html}
  </div>

  <!-- ── CHART ── -->
  {'<div style="padding:0 24px 20px;border-bottom:1px solid #2a2a3e;">' + chart_html + '</div>' if chart_b64 else ''}

  <!-- ── TRIGGER ── -->
  <div style="padding:20px 24px;border-bottom:1px solid #2a2a3e;">
    <h3 style="color:#8899bb;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;">WHEN TO ENTER — TRIGGER</h3>
    <div style="background:#0a1a0a;border-left:4px solid #27ae60;padding:12px 16px;border-radius:6px;">
      <p style="color:#a3e6b3;font-size:13px;margin:0;line-height:1.7;">{trigger}</p>
    </div>
  </div>

  <!-- ── INVALID IF ── -->
  <div style="padding:20px 24px;border-bottom:1px solid #2a2a3e;">
    <h3 style="color:#8899bb;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;">DO NOT ENTER — TRADE IS INVALID IF</h3>
    <div style="background:#1a0a0a;border-left:4px solid #ef5350;padding:12px 16px;border-radius:6px;">
      <p style="color:#f5a0a0;font-size:13px;margin:0;line-height:1.7;">{invalid_if}</p>
    </div>
  </div>

  <!-- ── GLOSSARY ── -->
  <div style="padding:20px 24px;background:#0d1117;">
    <h3 style="color:#555;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin:0 0 14px;">QUICK REFERENCE — TERMS USED IN THIS ALERT</h3>
    {glossary_html}
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

# ── MAIN ──────────────────────────────────────────────────────────────────────
print(f"Alert engine started {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

# Outcome check runs first — regardless of market hours.
# This means manual triggers on weekends still update this week's results.
run_intraweek_outcome_check()

market_open, market_status = is_market_open()
print(f"  Market: {market_status}")
if not market_open:
    print("  Exiting — market closed.")
    exit(0)

macro_news   = fetch_macro_news()
alerts_fired = 0

for pair_conf in config["pairs"]:
    symbol      = pair_conf["symbol"]
    name        = pair_conf["name"]
    prox        = pair_conf["proximity_pct"]
    min_touches = pair_conf.get("min_touches", 1)
    min_conf    = pair_conf.get("min_confidence", 5)

    print(f"  Scanning {name}...")
    zones, current_price, df1, df2 = detect_zones_and_candles(symbol, min_touches)

    if current_price is None:
        print(f"    No data for {name}. Skipping.")
        continue

    # ── Breakout check (independent of zone alert) ──
    breakout = detect_breakout(df1)
    if breakout:
        if should_alert_breakout(name, breakout["broken_level"], current_price, prox):
            print(f"    BREAKOUT: {name} {breakout['direction']} [{breakout['bo_score']}/5]")
            # Breakout stored — will be attached to zone email if zone fires,
            # or sent standalone if no zone fires
        else:
            print(f"    Breakout detected for {name} but price hasn't moved enough since last breakout alert.")
            breakout = None  # suppress repeat

    # ── Zone check ──
    zone_alert_fired = False
    for zone_level, touches in zones:
        dist_pct = abs(current_price - zone_level) / zone_level * 100
        if dist_pct > prox:
            continue

        if not should_alert_zone(name, zone_level, current_price, prox):
            print(f"    {name} @ {zone_level:.5f} — price hasn't moved enough since last alert.")
            continue

        zone_label = get_zone_label(zone_level, current_price)
        fatigue    = count_zone_alerts(name, zone_level)
        print(f"    ZONE HIT: {name} {zone_label} @ {zone_level:.5f} dist:{dist_pct:.2f}% fatigue:{fatigue}")

        prompt = build_zone_prompt(name, round(zone_level,5), zone_label,
                                   round(current_price,5), macro_news,
                                   df1, df2, min_conf, fatigue)
        data, error = call_gemini(prompt)

        if error:
            print(f"    {error}")
            continue

        score = data.get("confidence_score",0)
        if not data.get("send_alert", False):
            print(f"    {name} skipped — {score}/10 below {min_conf}. {data.get('confidence_reason','')}")
            continue

        levels = {'zone':zone_level,'current':current_price,
                  'entry':data.get('entry',''),'sl':data.get('sl',0),
                  'tp1':data.get('tp1',0),'tp2':data.get('tp2',0)}

        chart1 = generate_chart(df1, f"{name} — H1",  levels, data)
        chart2 = generate_chart(df2, f"{name} — M15", levels, data)

        html    = build_zone_email_html(data, name, round(zone_level,5), zone_label,
                                        round(current_price,5), chart1, chart2, breakout)
        subject = f"[{score}/10] {name} | {zone_label} | {round(zone_level,5)} | {datetime.utcnow().strftime('%H:%M')} UTC"
        if breakout:
            subject = f"[SMC {score}/10 + BO {breakout['bo_score']}/5] {name} | {zone_label} | {datetime.utcnow().strftime('%H:%M')} UTC"

        send_email(subject, html, chart1, chart2)
        log_alert(name, round(zone_level,5), zone_label, round(current_price,5), data, "zone", geo_flag=bool(data.get('geo_flag', False)))
        record_zone_alert(name, zone_level, current_price)
        if breakout:
            record_breakout_alert(name, breakout["broken_level"], current_price)

        alerts_fired += 1
        zone_alert_fired = True
        print(f"    Sent: {name} SMC[{score}/10]" + (f" + BO[{breakout['bo_score']}/5]" if breakout else ""))
        break  # one zone email per pair per run

    # ── Standalone breakout email (only if no zone alert fired) ──
    if breakout and not zone_alert_fired:
        if should_alert_breakout(name, breakout["broken_level"], current_price, prox):
            bo_levels = {
                'zone':    breakout["broken_level"],
                'current': current_price,
                'entry':   breakout.get("retest_entry", 0),
                'sl':      breakout.get("sl_retest", 0),
                'tp1':     breakout.get("tp1_retest", 0),
                'tp2':     breakout.get("tp2_retest", 0),
            }
            bo_chart = generate_chart(df1, f"{name} — H1 Retest Setup", bo_levels, {})
            html    = build_breakout_only_email_html(breakout, name, round(current_price,5), bo_chart)
            _ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
            subject = f"[BO {breakout['bo_score']}/5] {name} | {breakout['direction']} Breakout | {round(breakout['broken_level'],5)} | {_ist_now.strftime('%H:%M')} IST"
            send_email(subject, html, bo_chart)
            bo_bias = "LONG" if breakout["direction"] == "BULLISH" else "SHORT"
            bo_data = {
                "bias":             bo_bias,
                "entry":            breakout.get("retest_entry", 0),
                "sl":               breakout.get("sl_retest", 0),
                "tp1":              breakout.get("tp1_retest", 0),
                "tp2":              breakout.get("tp2_retest", 0),
                "confidence_score": breakout.get("bo_score", 0),
                "confluences":      breakout.get("bo_reasons", []),
                "trigger":          breakout.get("trigger_text", ""),
                "invalid_if":       breakout.get("invalid_if_text", ""),
            }
            log_alert(name, breakout["broken_level"], f"{breakout['direction']} Breakout",
                      round(current_price,5), bo_data, "breakout",
                      geo_flag=detect_geo_flag_phrases(macro_news))
            record_breakout_alert(name, breakout["broken_level"], current_price)
            alerts_fired += 1
            print(f"    Sent standalone breakout: {name} [{breakout['bo_score']}/5]")

save_alert_log()
print(f"Log saved: {len(alert_log)} total entries.")
print(f"Scan complete. {alerts_fired} alert(s) fired.")
