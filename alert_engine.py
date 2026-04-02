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

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY    = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_PASS    = os.environ["GMAIL_APP_PASSWORD"]
ALERT_EMAIL   = os.environ["ALERT_EMAIL"]
# ── Scoring maximums — Python-enforced caps per item ──────────────────────────
SCORE_MAX = {
    "liquidity_swept":      2.5,
    "fvg_overlaps_ob":      2.0,
    "premium_discount":     1.5,
    "premium_discount_zone":1.5,
    "multi_tf_alignment":   1.5,
    "zone_freshness":       1.0,
    "no_high_impact_news":  1.0,
    "session_alignment":    0.5,
}
MACRO_CACHE_FILE = "macro_cache.json"

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def utc_now():
    return datetime.utcnow()

def ist_now():
    return utc_now() + timedelta(hours=5, minutes=30)

def ist_str():
    return ist_now().strftime("%d %b %Y %H:%M IST")

def utc_str():
    return utc_now().strftime("%Y-%m-%d %H:%M")

def hours_since(ts):
    if not ts:
        return None
    try:
        return (utc_now() - datetime.strptime(ts, "%Y-%m-%d %H:%M")).total_seconds() / 3600
    except Exception:
        return None

def fmt_price(price, pair_conf):
    """Format price to correct decimal places for this pair."""
    dp = pair_conf.get("decimal_places", 5)
    return f"{float(price):,.{dp}f}"

def get_pair_conf(pair_name):
    return next((p for p in config["pairs"] if p["name"] == pair_name), None)

def parse_entry_mid(entry_value):
    try:
        s = str(entry_value).strip()
        if not s:
            return None
        if "-" in s:
            nums = [float(x.strip()) for x in s.split("-") if x.strip()]
            if len(nums) == 2:
                return (nums[0] + nums[1]) / 2
        return float(s)
    except Exception:
        return None

# ── File state ────────────────────────────────────────────────────────────────
ALERT_LOG_FILE     = "alert_log.json"
SCAN_LOG_FILE      = "scan_log.json"
SYSTEM_STATUS_FILE = "system_status.json"
VISIT_FILE         = "zone_visit_state.json"

alert_log      = load_json(ALERT_LOG_FILE, [])
scan_log       = load_json(SCAN_LOG_FILE, [])
system_status  = load_json(SYSTEM_STATUS_FILE, {
    "last_ok_email_utc": None,
    "last_error_email_utc": None,
    "last_trade_alert_utc": None
})
visit_state = load_json(VISIT_FILE, {})

def save_alert_log():
    save_json(ALERT_LOG_FILE, alert_log)

def should_send_ok():
    last_ok    = hours_since(system_status.get("last_ok_email_utc"))
    last_trade = hours_since(system_status.get("last_trade_alert_utc"))
    return (last_ok is None or last_ok >= 3) and (last_trade is None or last_trade >= 3)

def should_send_error():
    last_err = hours_since(system_status.get("last_error_email_utc"))
    return last_err is None or last_err >= 3

def log_scan(pair, status, reason, zone=None):
    scan_log.append({
        "time_ist": ist_now().strftime("%d %b %H:%M IST"),
        "time_utc": utc_str(),
        "pair": pair,
        "zone": round(zone, 5) if zone is not None else None,
        "status": status,
        "reason": reason
    })
    save_json(SCAN_LOG_FILE, scan_log)

# ── Zone visit state ──────────────────────────────────────────────────────────
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

# ── Zone fatigue ──────────────────────────────────────────────────────────────
def count_zone_alerts(pair, zone_level, days=30):
    cutoff = utc_now() - timedelta(days=days)
    count  = 0
    for a in alert_log:
        try:
            if a.get("alert_type") != "zone":
                continue
            if datetime.strptime(a["timestamp_utc"], "%Y-%m-%d %H:%M") < cutoff:
                continue
            if a["pair"] != pair:
                continue
            if abs(float(a.get("zone_level", 0)) - zone_level) / zone_level * 100 < 0.3:
                count += 1
        except Exception:
            pass
    return count

# ── Market hours (IST) ────────────────────────────────────────────────────────
def is_market_open():
    ist = ist_now()
    wd, h, m = ist.weekday(), ist.hour, ist.minute
    if wd == 5: return False, "Saturday — closed."
    if wd == 6: return False, "Sunday — closed."
    if h < 8:   return False, f"Before 8:00 AM IST — {ist.strftime('%A %H:%M')} IST."
    if wd == 4 and h >= 23 and m >= 30:
        return False, "Friday after 11:30 PM IST."
    return True, f"Open — {ist.strftime('%A %H:%M')} IST"

# ── Session alignment check ──────────────────────────────────────────────────
def is_in_session(pair_conf):
    """Check if current UTC hour falls within this pair's primary session."""
    session_name = pair_conf.get("primary_session", "any")
    session_def  = config.get("sessions", {}).get(session_name)
    if not session_def:
        return True
    h = utc_now().hour
    s, e = session_def["utc_start"], session_def["utc_end"]
    if s <= e:
        return s <= h < e
    else:
        return h >= s or h < e

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
    except Exception:
        return None

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
    return "Demand" if zone_level < current_price else "Supply"

# ── Macro news ────────────────────────────────────────────────────────────────
def fetch_macro_news():
    headlines = []
    # Source 1 — FXStreet RSS
    try:
        url   = "https://api.rss2json.com/v1/api.json?rss_url=https://www.fxstreet.com/rss/news&count=5"
        r     = requests.get(url, timeout=10)
        items = r.json().get("items", [])
        for i in items:
            headlines.append(f"[FXStreet] {i['title']}")
    except Exception:
        headlines.append("[FXStreet] Unavailable")
    # Source 2 — Google News RSS (geopolitical)
    geo_query = "Iran+war+OR+military+strike+OR+sanctions+OR+oil+supply+OR+tariff+OR+ceasefire+OR+invasion+OR+embargo"
    geo_url   = f"https://news.google.com/rss/search?q={geo_query}&hl=en-US&gl=US&ceid=US:en"
    try:
        import xml.etree.ElementTree as ET
        r    = requests.get(geo_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.content)
        items = root.findall(".//item")[:5]
        for item in items:
            title = item.findtext("title", "").strip()
            if title:
                headlines.append(f"[GeoNews] {title}")
    except Exception:
        headlines.append("[GeoNews] Unavailable")
    news_text = "\n".join(headlines) if headlines else "Macro news unavailable."
    # Cache for merged Alert 2 to use without re-fetching
    save_json(MACRO_CACHE_FILE, {"cached_at": utc_str(), "news": news_text})
    return news_text

def get_cached_macro_news():
    """Read cached macro news. Used by Job 2 (entry/invalidation checks)."""
    cache = load_json(MACRO_CACHE_FILE, {})
    return cache.get("news", "Macro news unavailable (cache miss).")

def format_candles(df, label, n=20):
    if df is None or df.empty:
        return f"{label}: No data\n"
    lines = [f"{label} (last {n} candles):"]
    for i in range(max(0, len(df)-n), len(df)):
        try:
            ts  = df.index[i]
            # Convert to IST for display
            if hasattr(ts, 'tz_localize') and ts.tzinfo is None:
                ts_ist = ts + timedelta(hours=5, minutes=30)
            elif hasattr(ts, 'tz_convert'):
                ts_ist = ts.tz_convert(None) + timedelta(hours=5, minutes=30)
            else:
                ts_ist = ts + timedelta(hours=5, minutes=30)
            tss = ts_ist.strftime('%d %b %H:%M IST')
            lines.append(f"{tss} O:{float(df['Open'].iloc[i]):.5f} "
                         f"H:{float(df['High'].iloc[i]):.5f} "
                         f"L:{float(df['Low'].iloc[i]):.5f} "
                         f"C:{float(df['Close'].iloc[i]):.5f}")
        except Exception:
            pass
    return "\n".join(lines) + "\n"


# ── Gemini prompt ─────────────────────────────────────────────────────────────
def build_zone_prompt(pair_conf, zone_level, zone_label, current_price,
                      macro_news, df1, df2, fatigue_count):
    name        = pair_conf["name"]
    pair_type   = pair_conf.get("pair_type", "forex")
    dp          = pair_conf.get("decimal_places", 5)
    min_conf    = pair_conf.get("min_confidence", 7)
    min_sl      = pair_conf.get("min_sl_pips", 15)
    structure_tf = pair_conf.get("structure_tf", ["H1"])
    extra_gate  = pair_conf.get("extra_gate")
    risk_dollar = config["account"]["balance"] * config["account"]["risk_percent"] / 100
    scoring     = config.get("scoring", {})
    ist_time    = ist_now().strftime("%H:%M IST, %d %b %Y")

    # Structure TF instruction
    if "M15" in structure_tf and "H1" in structure_tf:
        tf_rule = "H1 BOS or H1 CHoCH or M15 BOS or M15 CHoCH — any ONE of these passes the gate."
    else:
        tf_rule = "H1 BOS or H1 CHoCH ONLY. M15 structure is NOT accepted for this pair."

    # Extra gate instruction
    extra_gate_text = ""
    if extra_gate == "liquidity_swept_required":
        extra_gate_text = (f"\nADDITIONAL HARD GATE FOR {name}: Liquidity MUST be swept before zone entry. "
                           "If no liquidity sweep is visible in the candle data, set send_alert to false regardless of score.")
    elif extra_gate == "macro_alignment_required":
        extra_gate_text = (f"\nADDITIONAL HARD GATE FOR {name}: Macro context MUST align with technical bias. "
                           "If DXY direction, safe-haven flows, or geopolitical context contradicts the trade direction, "
                           "set send_alert to false. Gold is driven by macro first, technicals second.")

    # Fatigue text
    fatigue_thresh = scoring.get("zone_fatigue_threshold", 3)
    if fatigue_count >= fatigue_thresh + 2:
        fatigue_rule = f"Zone alerted {fatigue_count} times in 30 days. HEAVILY FATIGUED. Score zone_freshness as 0."
    elif fatigue_count >= fatigue_thresh:
        fatigue_rule = f"Zone alerted {fatigue_count} times in 30 days. FATIGUED. Score zone_freshness as 0."
    else:
        fatigue_rule = f"Zone alerted {fatigue_count} times in 30 days. FRESH. Score zone_freshness as 1.0."

    # News blackout
    nb_before = scoring.get("news_blackout_hours_before", 2)
    nb_after  = scoring.get("news_blackout_hours_after", 1)

    return f"""You are an elite institutional SMC trader. Analyze this zone setup with extreme precision.

PAIR: {name} ({pair_type}) | ZONE: {zone_label} at {zone_level} | PRICE NOW: {current_price}
TIME: {ist_time}
ACCOUNT: ${config["account"]["balance"]} | RISK: {config["account"]["risk_percent"]}% = ${risk_dollar:.0f}
PRICE FORMAT: {dp} decimal places for this pair.

CANDLE DATA:
{format_candles(df1, "H1")}
{format_candles(df2, "M15")}

MACRO HEADLINES:
{macro_news}

═══════════════════════════════════════════════════════════════
HARD GATES — ALL THREE MUST PASS OR SET send_alert TO false:
═══════════════════════════════════════════════════════════════
GATE 1 — STRUCTURE: {tf_rule}
  You MUST cite the specific candle timestamp and price where BOS or CHoCH occurred.
  If you cannot identify a specific candle, this gate FAILS.

GATE 2 — VALID ORDER BLOCK: A valid OB must exist at or near the zone.
  You MUST cite ob_top, ob_bottom, and the candle that formed the OB.
  If no OB is identifiable, this gate FAILS.

GATE 3 — RISK:REWARD: R:R to TP1 must be at least 2.0:1.
  Calculate from entry to TP1 vs entry to SL. If below 2:1, this gate FAILS.
{extra_gate_text}

═══════════════════════════════════════════════════════════════
WEIGHTED SCORECARD — SCORE OUT OF 10 (minimum {min_conf} to send):
═══════════════════════════════════════════════════════════════

1. LIQUIDITY SWEPT (2.5 pts):
   2.5 = Price clearly spiked beyond a prior swing high/low then reversed. Cite the sweep candle timestamp and wick price.
   1.5 = Price approached but did not clearly exceed the prior swing extreme.
   0   = No visible sweep. Prior swing extremes untested.

2. FVG OVERLAPS OB (2.0 pts):
   2.0 = A Fair Value Gap overlaps with the Order Block range. Cite fvg_top and fvg_bottom.
   0   = No FVG found within or overlapping the OB range.

3. PREMIUM/DISCOUNT ZONE (1.5 pts):
   1.5 = For SHORT: price is above the 50% mark of the recent range (premium). For LONG: below 50% (discount).
   0   = Price is on the wrong side (buying in premium or selling in discount).
   State the range high, low, midpoint, and current position.

4. MULTI-TIMEFRAME ALIGNMENT (1.5 pts):
   1.5 = Both H1 AND M15 show structural confirmation in the same direction.
   0.5 = Only one timeframe confirms.
   0   = Neither confirms or they conflict.

5. ZONE FRESHNESS (1.0 pt): {fatigue_rule}

6. NO HIGH-IMPACT NEWS (1.0 pt):
   1.0 = No high-impact event within {nb_before} hours BEFORE or {nb_after} hour AFTER current time.
   0   = A high-impact event falls within the blackout window.
   HIGH-IMPACT EVENTS include: rate decisions, CPI, NFP, GDP, PMI, AND geopolitical events
   (military conflict, sanctions, trade war escalation, tariff announcements, energy supply
   disruptions, ceasefire negotiations, diplomatic breakdowns, invasion, retaliation, embargo).
   If ANY of these are present, score 0. Cite the specific event.

7. SESSION ALIGNMENT (0.5 pts):
   0.5 = {name} is trading during its primary session window.
   0   = Outside primary session.

TOTAL POSSIBLE: 10.0 | MINIMUM TO SEND: {min_conf}
Your confidence_score MUST equal the sum of the individual scores above.

═══════════════════════════════════════════════════════════════
ENTRY, SL, TP RULES:
═══════════════════════════════════════════════════════════════
ENTRY: 50% midpoint of OB candle body. If FVG overlaps OB, use FVG edge (top for longs, bottom for shorts).
SL: OB wick extreme + buffer. MINIMUM SL distance for {name}: {min_sl} pips/points.
  If your calculated SL is tighter than {min_sl}, widen it to {min_sl}.
TP1: Next significant opposing zone or liquidity pool. Must achieve 2:1 RR minimum.
TP2: Second opposing zone beyond TP1.
PARTIAL CLOSE: Close 50% at TP1, move SL to breakeven, let remaining run to TP2.

═══════════════════════════════════════════════════════════════
TRIGGER & AUTOMATION RULES:
═══════════════════════════════════════════════════════════════
- trigger_status: "not_ready" (waiting for confirmation) | "ready" (confirmed) | "invalidated" (setup broken)
- entry_ready_now: true ONLY if trigger candle pattern is ALREADY confirmed on latest candles
- trigger_level: the EXACT price level that must be broken for trigger (e.g., M15 swing low for CHoCH)
  This will be used to set a TradingView alert. It must be a single numeric value.
- invalidate_above / invalidate_below: numeric price where setup is cancelled
- bias MUST be "LONG" or "SHORT". Never "WAIT". If the setup is not tradeable, set send_alert to false.

THIS SYSTEM ONLY TRADES ZONE RETESTS (OB + FVG setups).
NEVER suggest breakout entries, momentum entries, or trend-following entries.
If current price action is a breakout without a valid zone retest, set send_alert to false.

Return ONLY raw JSON. No markdown. No code fences. No text outside the JSON.
{{
  "send_alert": true,
  "gates_passed": true,
  "gate_structure": "cite specific candle or false",
  "gate_ob": "cite OB candle or false",
  "gate_rr": "cite RR value or false",
  "confidence_score": 0.0,
  "score_breakdown": {{
    "liquidity_swept": 0.0,
    "fvg_overlaps_ob": 0.0,
    "premium_discount": 0.0,
    "multi_tf_alignment": 0.0,
    "zone_freshness": 0.0,
    "no_high_impact_news": 0.0,
    "session_alignment": 0.0
  }},
  "confidence_reason": "one sentence",
  "news_flag": "none or describe the event",
  "geo_flag": false,
  "bias": "LONG or SHORT",
  "bias_reason": "max 12 words",

  "entry": 0.0,
  "entry_model": "limit",
  "entry_ready_now": false,

  "trigger_status": "not_ready",
  "trigger_tf": "M15 or H1",
  "trigger_kind": "choch or bos or engulf or break_retest",
  "trigger_level": 0.0,
  "trigger": "exact candle pattern required — cite specific price and what must happen",

  "invalidate_above": null,
  "invalidate_below": null,
  "invalid_if": "exact condition that cancels this trade",

  "sl": 0.0,
  "sl_note": "one sentence on SL logic",
  "tp1": 0.0,
  "tp2": 0.0,
  "rr_tp1": "x.x",
  "rr_tp2": "x.x",

  "confluences": ["item1 — cite evidence", "item2 — cite evidence"],
  "missing": [{{"item": "name", "reason": "why it matters"}}],
  "macro_line1": "main macro driver right now",
  "macro_line2": "key upcoming event this week",
  "mindset": "one sharp psychological trap to avoid",

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

# ── Gemini call with retry ────────────────────────────────────────────────────
def call_gemini(prompt, max_retries=2):
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        }
    }
    for attempt in range(max_retries + 1):
        try:
            r      = requests.post(url, json=body, timeout=90)
            result = r.json()
            if "candidates" not in result:
                err_code = result.get("error", {}).get("code", 0)
                if err_code == 429 and attempt < max_retries:
                    wait = 10 * (attempt + 1)
                    print(f"    Gemini rate limit — retrying in {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    continue
                return None, f"Gemini error (code {err_code}): {result.get('error', {}).get('message', 'Unknown')}"
            raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(raw), None
        except json.JSONDecodeError as e:
            return None, f"Gemini returned invalid JSON: {str(e)[:100]}"
        except Exception as e:
            if attempt < max_retries:
                time.sleep(5)
                continue
            return None, f"Gemini error: {str(e)[:100]}"
    return None, "Gemini failed after all retries"

# ── Python post-validation ────────────────────────────────────────────────────
def validate_gemini_response(data, pair_conf, zone_label, current_price):
    """
    Validates Gemini's JSON response. Returns (is_valid, reason).
    Checks: gates, score math, SL sanity, bias sanity, minimum SL distance.
    """
    name   = pair_conf["name"]
    min_sl = pair_conf.get("min_sl_pips", 15)
    dp     = pair_conf.get("decimal_places", 5)

    # 1. Gates must all pass
    if not data.get("gates_passed", False):
        return False, "Hard gates not passed"

    # 2. Bias must be LONG or SHORT (not WAIT)
    bias = str(data.get("bias", "")).upper()
    if bias not in ("LONG", "SHORT"):
        return False, f"Bias is '{bias}' — must be LONG or SHORT"

    # 3. Clamp each score item to its maximum, recalculate, override Gemini's total
    breakdown = data.get("score_breakdown", {})
    clamped = False
    for key in list(breakdown.keys()):
        raw_val = float(breakdown[key])
        max_val = SCORE_MAX.get(key, 0)
        if max_val > 0 and raw_val > max_val:
            print(f"    Score clamped: {key} was {raw_val}, max is {max_val}")
            breakdown[key] = max_val
            clamped = True
    data["score_breakdown"] = breakdown
    calc_sum = round(sum(float(v) for v in breakdown.values()), 1)
    reported = float(data.get("confidence_score", 0))
    if clamped or abs(calc_sum - reported) >= 0.3:
        print(f"    Score corrected: Gemini said {reported}, clamped breakdown sums to {calc_sum}")
        data["confidence_score"] = calc_sum

    # 4. SL on correct side of entry
    try:
        entry = float(str(data.get("entry", 0)).split("-")[0].strip() or 0)
        sl    = float(data.get("sl", 0) or 0)
        tp1   = float(data.get("tp1", 0) or 0)
        if entry <= 0 or sl <= 0 or tp1 <= 0:
            return False, "Missing entry/SL/TP1 values"
        if bias == "LONG" and sl >= entry:
            return False, f"SL ({sl}) above entry ({entry}) for LONG"
        if bias == "SHORT" and sl <= entry:
            return False, f"SL ({sl}) below entry ({entry}) for SHORT"
    except Exception:
        return False, "Could not parse entry/SL/TP1"

    # 5. Minimum SL distance enforcement
    sl_dist = abs(entry - sl)
    # Convert min_sl_pips to price distance
    if pair_conf.get("pair_type") == "forex":
        if "JPY" in name:
            min_sl_price = min_sl * 0.01  # JPY pips = 0.01
        else:
            min_sl_price = min_sl * 0.0001  # Standard forex pips = 0.0001
    else:
        min_sl_price = min_sl  # Indices/BTC/Gold: pips = points

    if sl_dist < min_sl_price:
        # Widen SL to minimum
        if bias == "LONG":
            data["sl"] = round(entry - min_sl_price, dp)
        else:
            data["sl"] = round(entry + min_sl_price, dp)
        data["sl_note"] = f"SL widened to minimum {min_sl} pips/pts for {name}. " + data.get("sl_note", "")
        print(f"    SL widened: {sl_dist:.5f} → {min_sl_price:.5f} (min for {name})")
        # Recalculate RR
        new_sl   = float(data["sl"])
        new_risk = abs(entry - new_sl)
        new_rr   = abs(tp1 - entry) / new_risk if new_risk > 0 else 0
        data["rr_tp1"] = f"{new_rr:.1f}"
        if new_rr < 2.0:
            return False, f"After SL widening, RR dropped to {new_rr:.1f} (below 2:1)"

    # 6. Bias matches zone type
    if "Demand" in zone_label and bias == "SHORT":
        return False, "SHORT bias at Demand zone — contradictory"
    if "Supply" in zone_label and bias == "LONG":
        return False, "LONG bias at Supply zone — contradictory"

    return True, "OK"


# ── Chart generation (improved) ───────────────────────────────────────────────
def generate_chart(df, title, levels, data, pair_conf):
    try:
        if df is None or df.empty:
            return None
        dp = pair_conf.get("decimal_places", 5)
        df_plot = df.tail(30).copy().reset_index(drop=True)
        for col in ['Open','High','Low','Close']:
            if col not in df_plot.columns:
                return None

        fig, (ax, ax_vol) = plt.subplots(2, 1, figsize=(12, 6),
            gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05},
            facecolor='#131722')

        for a in [ax, ax_vol]:
            a.set_facecolor('#131722')
            for s in a.spines.values():
                s.set_color('#2a2a3e')

        # Draw candles — wider bodies, thicker wicks
        for i, row in df_plot.iterrows():
            try:
                o = float(row['Open']); h = float(row['High'])
                l = float(row['Low']);  c = float(row['Close'])
                if any(np.isnan(v) for v in [o,h,l,c]):
                    continue
                col_c = '#26a69a' if c >= o else '#ef5350'
                ax.plot([i,i], [l,h], color=col_c, linewidth=1.2, zorder=2)
                body = abs(c-o) or (h-l) * 0.02  # Minimum body height for dojis
                ax.add_patch(patches.Rectangle(
                    (i-0.4, min(o,c)), 0.8, body,
                    facecolor=col_c, linewidth=0, alpha=0.95, zorder=3))
            except Exception:
                continue

        n = len(df_plot)

        # OB zone shading
        ob_top    = float(data.get('ob_top', 0) or 0)
        ob_bottom = float(data.get('ob_bottom', 0) or 0)
        if ob_top > 0 and ob_bottom > 0 and abs(ob_top - ob_bottom) > 0:
            oc = '#26a69a' if data.get('ob_type','') == 'bullish' else '#ef5350'
            ax.add_patch(patches.Rectangle(
                (0, ob_bottom), n, ob_top - ob_bottom,
                facecolor=oc, edgecolor=oc, linewidth=1.5,
                alpha=0.25, zorder=1))
            ax.text(1, ob_top + (ob_top - ob_bottom) * 0.1, "OB",
                color=oc, fontsize=10, va='bottom', fontweight='bold', zorder=5)

        # FVG zone shading
        fvg_top    = float(data.get('fvg_top', 0) or 0)
        fvg_bottom = float(data.get('fvg_bottom', 0) or 0)
        if fvg_top > 0 and fvg_bottom > 0 and abs(fvg_top - fvg_bottom) > 0:
            ax.add_patch(patches.Rectangle(
                (0, fvg_bottom), n, fvg_top - fvg_bottom,
                facecolor='#3498db', edgecolor='#3498db', linewidth=1.5,
                alpha=0.22, zorder=1))
            ax.text(1, fvg_top + (fvg_top - fvg_bottom) * 0.1, "FVG",
                color='#3498db', fontsize=10, va='bottom', fontweight='bold', zorder=5)

        # Price levels — clear labels on right margin
        level_cfg = {
            'tp2':     ('#1e8449', '--', 1.2, 'TP2'),
            'tp1':     ('#27ae60', '-',  1.8, 'TP1'),
            'entry':   ('#e67e22', '-',  1.8, 'ENTRY'),
            'zone':    ('#9b59b6', '--', 1.0, 'ZONE'),
            'current': ('#ffffff', ':',  1.0, 'NOW'),
            'sl':      ('#e74c3c', '-',  1.8, 'SL'),
        }
        for key, (color, style, width, lbl) in level_cfg.items():
            val = levels.get(key, 0)
            try:
                price = float(str(val).split('-')[0].strip()) if val else 0
            except Exception:
                price = 0
            if price > 0:
                ax.axhline(y=price, color=color, linestyle=style, linewidth=width, alpha=0.9, zorder=4)
                ax.text(n + 0.5, price, f" {lbl}: {price:,.{dp}f}",
                    color=color, fontsize=9, va='center', fontweight='bold', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#131722', edgecolor='none', alpha=0.8))

        # Trigger level (if present)
        trig_level = float(data.get('trigger_level', 0) or 0)
        if trig_level > 0:
            ax.axhline(y=trig_level, color='#f39c12', linestyle='-.', linewidth=1.5, alpha=0.8, zorder=4)
            ax.text(n + 0.5, trig_level, f" TRIGGER: {trig_level:,.{dp}f}",
                color='#f39c12', fontsize=9, va='center', fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#131722', edgecolor='none', alpha=0.8))

        # Volume bars
        for i, row in df_plot.iterrows():
            try:
                vol = float(row.get('Volume', 0) or 0)
                if np.isnan(vol): vol = 0
                vc = '#26a69a' if float(row['Close']) >= float(row['Open']) else '#ef5350'
                ax_vol.bar(i, vol, color=vc, alpha=0.5, width=0.7)
            except Exception:
                continue

        # IST timestamps on x-axis (every 5 candles)
        tick_positions = list(range(0, n, max(1, n // 6)))
        tick_labels = []
        for pos in tick_positions:
            try:
                ts = df.iloc[-(n - pos)].name
                if hasattr(ts, 'strftime'):
                    ts_ist = ts.tz_localize(None) + timedelta(hours=5, minutes=30) if hasattr(ts, 'tz_localize') and ts.tzinfo is None else ts + timedelta(hours=5, minutes=30)
                    tick_labels.append(ts_ist.strftime('%d %b\n%H:%M'))
                else:
                    tick_labels.append('')
            except Exception:
                tick_labels.append('')

        ax.set_title(title, color='#dddddd', fontsize=11, pad=8, fontweight='bold', loc='left')
        ax.tick_params(colors='#888', labelsize=8)
        ax.yaxis.tick_right()
        ax.yaxis.set_tick_params(labelcolor='#aaa', labelsize=8)
        ax.xaxis.set_visible(False)
        ax.set_xlim(-1, n + 14)

        ax_vol.set_xticks(tick_positions)
        ax_vol.set_xticklabels(tick_labels, fontsize=7, color='#888')
        ax_vol.tick_params(colors='#666', labelsize=7)
        ax_vol.yaxis.tick_right()
        ax_vol.set_xlim(-1, n + 14)
        ax_vol.set_ylabel('Vol', color='#666', fontsize=7)

        plt.tight_layout(pad=0.5)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
            facecolor='#131722', edgecolor='none')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        print(f"    Chart ok: {len(b64)//1024}KB")
        return b64
    except Exception as e:
        print(f"    Chart error: {e}")
        plt.close('all')
        return None


# ── Zone Alert email HTML ─────────────────────────────────────────────────────
def build_zone_email_html(data, pair, pair_conf, zone_level, zone_label,
                          current_price, chart1_b64, chart2_b64):
    dp          = pair_conf.get("decimal_places", 5)
    ist_time    = ist_now().strftime("%H:%M IST, %d %b %Y")
    score       = data.get("confidence_score", 0)
    bias        = data.get("bias", "—")
    bias_color  = "#e74c3c" if bias == "SHORT" else "#27ae60"
    score_color = "#27ae60" if score >= 8 else "#e67e22" if score >= 7 else "#e74c3c"

    # News flag
    news_flag = data.get("news_flag", "none")
    news_html = ""
    if news_flag and news_flag.lower() != "none":
        news_html = (f'<div style="background:#fff3cd;padding:10px 20px;border-left:4px solid #f39c12;'
                     f'font-size:12px;color:#856404;"><b>⚠ NEWS:</b> {news_flag}</div>')

    # Format prices
    entry_p = fmt_price(data.get("entry", 0), pair_conf)
    sl_p    = fmt_price(data.get("sl", 0), pair_conf)
    tp1_p   = fmt_price(data.get("tp1", 0), pair_conf)
    tp2_p   = fmt_price(data.get("tp2", 0), pair_conf)
    zone_p  = fmt_price(zone_level, pair_conf)
    now_p   = fmt_price(current_price, pair_conf)

    # Trigger instruction for TradingView
    trigger_level = data.get("trigger_level", 0)
    trigger_kind  = data.get("trigger_kind", "")
    trigger_tf    = data.get("trigger_tf", "M15")
    trig_price_f  = fmt_price(trigger_level, pair_conf) if trigger_level else "—"

    if data.get("entry_ready_now"):
        action_box = f"""<div style="background:#27ae60;padding:16px 20px;border-radius:10px;margin-bottom:16px;">
          <p style="color:white;margin:0;font-size:11px;text-transform:uppercase;letter-spacing:1px;opacity:0.8;">✅ TRIGGER CONFIRMED — PLACE ORDER NOW</p>
          <p style="color:white;margin:8px 0 0;font-size:16px;font-weight:bold;">
            {'SELL' if bias=='SHORT' else 'BUY'} LIMIT at {entry_p}</p>
          <p style="color:white;margin:4px 0 0;font-size:14px;">SL: {sl_p} &nbsp;|&nbsp; TP1: {tp1_p} &nbsp;|&nbsp; TP2: {tp2_p}</p>
          <p style="color:#d5f5e3;margin:8px 0 0;font-size:11px;">Close 50% at TP1, move SL to breakeven, let rest run to TP2</p>
        </div>"""
    else:
        action_box = f"""<div style="background:#1a1a2e;padding:16px 20px;border-radius:10px;margin-bottom:16px;">
          <p style="color:#f39c12;margin:0;font-size:11px;text-transform:uppercase;letter-spacing:1px;">⏳ WAITING FOR TRIGGER — DO NOT PLACE ORDER YET</p>
          <p style="color:white;margin:10px 0 0;font-size:13px;">
            <b>Step 1:</b> Set TradingView alert → {trigger_tf} close {'below' if bias=='SHORT' else 'above'} {trig_price_f}</p>
          <p style="color:white;margin:4px 0 0;font-size:13px;">
            <b>Step 2:</b> When alert fires → place {'SELL' if bias=='SHORT' else 'BUY'} LIMIT at {entry_p}</p>
          <p style="color:white;margin:4px 0 0;font-size:13px;">
            <b>Levels:</b> SL: {sl_p} &nbsp;|&nbsp; TP1: {tp1_p} &nbsp;|&nbsp; TP2: {tp2_p}</p>
          <p style="color:#8899bb;margin:10px 0 0;font-size:11px;">
            Cancel if: {trigger_tf} closes {'above' if bias=='SHORT' else 'below'} {sl_p}</p>
          <p style="color:#8899bb;margin:2px 0 0;font-size:11px;">
            Close 50% at TP1, move SL to breakeven, let rest run to TP2</p>
          <p style="color:#445566;margin:8px 0 0;font-size:10px;">
            System will send Entry Alert when trigger is confirmed, or Invalidation if setup breaks.</p>
        </div>"""

    # Confluence table with weights and colors
    breakdown = data.get("score_breakdown", {})
    weights   = config.get("scoring", {}).get("confluences", {})
    conf_rows = ""
    conf_order = [
        ("liquidity_swept",     "Liquidity Swept"),
        ("fvg_overlaps_ob",     "FVG Overlaps OB"),
        ("premium_discount",    "Premium/Discount Zone"),
        ("multi_tf_alignment",  "Multi-TF Alignment"),
        ("zone_freshness",      "Zone Freshness"),
        ("no_high_impact_news", "No High-Impact News"),
        ("session_alignment",   "Session Alignment"),
    ]
    # Map breakdown keys (Gemini might use premium_discount or premium_discount_zone)
    for key, label in conf_order:
        max_pts = weights.get(key, weights.get(key.replace("premium_discount", "premium_discount_zone"), 0))
        scored  = float(breakdown.get(key, breakdown.get(key.replace("premium_discount", "premium_discount_zone"), 0)))
        if scored >= max_pts * 0.8:
            bg_c, txt_c, icon = "#e8f5e9", "#2e7d32", "✓"
        elif scored > 0:
            bg_c, txt_c, icon = "#fff8e1", "#f57f17", "◐"
        else:
            bg_c, txt_c, icon = "#ffebee", "#c62828", "✗"
        conf_rows += (f'<tr style="background:{bg_c};">'
                      f'<td style="padding:6px 10px;font-size:12px;color:{txt_c};font-weight:bold;width:30px;">{icon}</td>'
                      f'<td style="padding:6px 10px;font-size:12px;color:#333;">{label}</td>'
                      f'<td style="padding:6px 10px;font-size:12px;color:{txt_c};font-weight:bold;text-align:center;">{scored}/{max_pts}</td>'
                      f'</tr>')

    conf_table = (f'<table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;margin-bottom:16px;">'
                  f'<tr style="background:#1a1a2e;">'
                  f'<td colspan="3" style="padding:8px 10px;color:white;font-size:11px;font-weight:bold;letter-spacing:1px;">SCORECARD BREAKDOWN</td>'
                  f'</tr>{conf_rows}'
                  f'<tr style="background:#f4f4f8;">'
                  f'<td colspan="2" style="padding:8px 10px;font-size:13px;font-weight:bold;color:#1a1a2e;">TOTAL</td>'
                  f'<td style="padding:8px 10px;font-size:14px;font-weight:bold;color:{score_color};text-align:center;">{score}/10</td>'
                  f'</tr></table>')

    # Charts
    chart1_html = (f'<img src="cid:chart_h1" style="width:100%;border-radius:8px;display:block;margin-bottom:12px;" />'
                   if chart1_b64 else '<p style="color:#aaa;font-size:11px;">H1 chart unavailable.</p>')
    chart2_html = (f'<img src="cid:chart_m15" style="width:100%;border-radius:8px;display:block;margin-bottom:12px;" />'
                   if chart2_b64 else '<p style="color:#aaa;font-size:11px;">M15 chart unavailable.</p>')

    # Missing items
    miss_items = ""
    for m in data.get("missing", []):
        if isinstance(m, dict):
            miss_items += (f'<p style="font-size:12px;color:#c62828;margin:4px 0;padding:6px 10px;background:#ffebee;border-radius:6px;">'
                           f'✗ <b>{m.get("item","")}</b> — {m.get("reason","")}</p>')

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:12px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">

  <div style="background:#1a1a2e;padding:16px 20px;">
    <h2 style="color:white;margin:0;font-size:16px;">ZONE ALERT: {pair} — {zone_label}</h2>
    <p style="color:#8899bb;margin:4px 0 0;font-size:11px;">{ist_time}</p>
  </div>

  <div style="background:{bias_color};padding:10px 20px;">
    <p style="color:white;margin:0;font-size:14px;font-weight:bold;">{bias} — {data.get("bias_reason","")}</p>
  </div>

  {news_html}

  <div style="padding:16px 20px;">

    {action_box}

    <p style="font-size:12px;color:#666;margin:0 0 12px;">Zone: <b>{zone_label}</b> at {zone_p} &nbsp;|&nbsp; Now: {now_p} &nbsp;|&nbsp; R:R {data.get("rr_tp1","—")}:1</p>

    {chart1_html}
    {chart2_html}

    {conf_table}

    <table style="width:100%;font-size:12px;margin-bottom:14px;border-collapse:collapse;">
      <tr style="background:#f8f9fa;"><td style="padding:6px 10px;color:#555;width:100px;">Entry</td><td style="padding:6px 10px;font-weight:bold;">{entry_p}</td></tr>
      <tr><td style="padding:6px 10px;color:#e74c3c;">Stop Loss</td><td style="padding:6px 10px;font-weight:bold;color:#e74c3c;">{sl_p}</td></tr>
      <tr style="background:#f8f9fa;"><td style="padding:6px 10px;color:#555;">SL Note</td><td style="padding:6px 10px;font-size:11px;color:#777;">{data.get("sl_note","")}</td></tr>
      <tr><td style="padding:6px 10px;color:#27ae60;">TP1</td><td style="padding:6px 10px;font-weight:bold;color:#27ae60;">{tp1_p} (R:R {data.get("rr_tp1","")})</td></tr>
      <tr style="background:#f8f9fa;"><td style="padding:6px 10px;color:#1e8449;">TP2</td><td style="padding:6px 10px;font-weight:bold;color:#1e8449;">{tp2_p} (R:R {data.get("rr_tp2","")})</td></tr>
    </table>

    {miss_items}

    <div style="background:#fffbea;padding:10px 14px;border-radius:8px;border-left:4px solid #f39c12;margin-bottom:14px;">
      <p style="font-size:10px;color:#f39c12;margin:0 0 4px;font-weight:bold;">TRIGGER CONDITION</p>
      <p style="font-size:12px;color:#333;margin:0;">{data.get("trigger","")}</p>
    </div>

    <div style="background:#fef0f0;padding:10px 14px;border-radius:8px;border-left:4px solid #e74c3c;margin-bottom:14px;">
      <p style="font-size:10px;color:#e74c3c;margin:0 0 4px;font-weight:bold;">INVALIDATION</p>
      <p style="font-size:12px;color:#c0392b;margin:0;">{data.get("invalid_if","")}</p>
    </div>

    <p style="font-size:11px;color:#666;margin:0 0 4px;"><b>Macro:</b> {data.get("macro_line1","")}</p>
    <p style="font-size:11px;color:#666;margin:0 0 14px;">{data.get("macro_line2","")}</p>

    <div style="background:#1a1a2e;padding:12px 16px;border-radius:8px;">
      <p style="color:#8899bb;font-size:9px;margin:0 0 3px;text-transform:uppercase;letter-spacing:1px;">MINDSET</p>
      <p style="color:white;font-size:12px;margin:0;font-style:italic;">{data.get("mindset","")}</p>
    </div>

    <p style="font-size:9px;color:#bbb;margin:12px 0 0;text-align:center;">
      Prices from market data feed. Your broker prices may differ slightly. Adjust levels to match your MT5 chart.</p>
  </div>
</div>
</body>
</html>"""

# ── Entry Alert email (Stage 2) ───────────────────────────────────────────────
def build_entry_email_html(original_alert, refreshed_data, pair_conf, current_price):
    dp    = pair_conf.get("decimal_places", 5)
    score = refreshed_data.get("confidence_score", 0)
    bias  = refreshed_data.get("bias", original_alert.get("bias", "—"))
    bias_color  = "#e74c3c" if bias == "SHORT" else "#27ae60"
    score_color = "#27ae60" if score >= 8 else "#e67e22" if score >= 7 else "#e74c3c"
    ist_time = ist_now().strftime("%H:%M IST, %d %b %Y")

    entry_p = fmt_price(refreshed_data.get("entry", 0), pair_conf)
    sl_p    = fmt_price(refreshed_data.get("sl", 0), pair_conf)
    tp1_p   = fmt_price(refreshed_data.get("tp1", 0), pair_conf)
    tp2_p   = fmt_price(refreshed_data.get("tp2", 0), pair_conf)

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:12px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">

  <div style="background:#1a1a2e;padding:16px 20px;">
    <h2 style="color:white;margin:0;font-size:16px;">ENTRY ALERT: {original_alert['pair']}</h2>
    <p style="color:#8899bb;margin:4px 0 0;font-size:11px;">{ist_time}</p>
  </div>

  <div style="background:{bias_color};padding:14px 20px;">
    <p style="color:white;margin:0;font-size:16px;font-weight:bold;">
      {'SELL' if bias=='SHORT' else 'BUY'} LIMIT at {entry_p}</p>
    <p style="color:rgba(255,255,255,0.9);margin:6px 0 0;font-size:14px;">
      SL: {sl_p} &nbsp;|&nbsp; TP1: {tp1_p} &nbsp;|&nbsp; TP2: {tp2_p}</p>
    <p style="color:rgba(255,255,255,0.7);margin:6px 0 0;font-size:11px;">
      Close 50% at TP1, move SL to breakeven, let rest run to TP2</p>
  </div>

  <div style="padding:14px 20px;background:#f8f9fa;border-bottom:1px solid #eee;">
    <span style="font-size:11px;color:#666;">Updated confidence: </span>
    <span style="font-size:16px;font-weight:bold;color:{score_color};">{score}/10</span>
    <span style="font-size:11px;color:#888;margin-left:6px;">— {refreshed_data.get("confidence_reason","")}</span>
  </div>

  <div style="padding:14px 20px;">
    <p style="font-size:12px;color:#666;margin:0 0 10px;">
      Original zone: <b>{original_alert.get("zone_label","")}</b> at {original_alert.get("zone_level","")}
      &nbsp;|&nbsp; Now: {fmt_price(current_price, pair_conf)}
      &nbsp;|&nbsp; R:R {refreshed_data.get("rr_tp1","—")}:1</p>

    <p style="font-size:11px;color:#888;margin:0;">
      Refer to your earlier Zone Alert for full chart and analysis.</p>
  </div>
</div>
</body>
</html>"""

# ── Invalidation email (Stage 3) ──────────────────────────────────────────────
def build_invalidation_email_html(alert, refreshed_data, pair_conf, current_price):
    ist_time = ist_now().strftime("%H:%M IST, %d %b %Y")
    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#fff7f7;padding:12px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">
  <div style="background:#7f1d1d;padding:16px 20px;">
    <h2 style="color:white;margin:0;font-size:16px;">TRADE INVALIDATED: {alert['pair']}</h2>
    <p style="color:#fecaca;margin:4px 0 0;font-size:11px;">{ist_time}</p>
  </div>
  <div style="background:#dc2626;padding:14px 20px;">
    <p style="color:white;margin:0;font-size:14px;font-weight:bold;">Cancel any pending {alert['pair']} orders</p>
  </div>
  <div style="padding:14px 20px;">
    <p style="font-size:13px;color:#374151;margin:0 0 10px;">The earlier setup is no longer valid.</p>
    <table style="width:100%;font-size:12px;">
      <tr><td style="padding:4px 0;color:#555;width:120px;">Pair</td><td style="padding:4px 0;font-weight:bold;">{alert.get("pair","")}</td></tr>
      <tr><td style="padding:4px 0;color:#555;">Bias was</td><td style="padding:4px 0;font-weight:bold;">{alert.get("bias","")}</td></tr>
      <tr><td style="padding:4px 0;color:#555;">Current price</td><td style="padding:4px 0;font-weight:bold;">{fmt_price(current_price, pair_conf)}</td></tr>
      <tr><td style="padding:4px 0;color:#555;">Reason</td><td style="padding:4px 0;">{refreshed_data.get("invalid_if", refreshed_data.get("confidence_reason","Setup no longer valid"))}</td></tr>
    </table>
  </div>
</div>
</body>
</html>"""

# ── System emails ─────────────────────────────────────────────────────────────
def build_ok_email_html():
    return f"""<html><body style="font-family:Arial,sans-serif;background:#f6f8fb;padding:16px;">
<div style="max-width:500px;margin:auto;background:white;border-radius:12px;padding:20px;border:1px solid #e5e7eb;">
  <h2 style="margin:0 0 8px;color:#1f2937;font-size:15px;">System OK</h2>
  <p style="color:#374151;font-size:13px;margin:0;">No trade setups passed all gates + scoring filters in the last 3 hours. The system is monitoring normally.</p>
  <p style="color:#9ca3af;font-size:11px;margin:12px 0 0;">{ist_str()}</p>
</div></body></html>"""

def build_error_email_html(error_lines, pairs_ok):
    """Plain English error email with pair status and guidance."""
    error_items = ""
    for line in error_lines[:10]:
        # Parse pair name and simplify error message
        parts  = line.split(":", 1)
        p_name = parts[0].strip() if len(parts) > 1 else "Unknown"
        err    = parts[1].strip() if len(parts) > 1 else line

        # Simplify Gemini errors
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            simple = "Gemini API rate limit reached. Will retry automatically on next scan."
        elif "Gemini error" in err:
            simple = "Gemini could not analyze this pair. Will retry on next scan."
        elif "No market data" in err:
            simple = "Market data unavailable (yfinance). Likely a temporary data feed issue."
        else:
            simple = err[:120]

        error_items += (f'<div style="background:#fef2f2;padding:8px 12px;border-radius:6px;margin-bottom:6px;">'
                        f'<p style="margin:0;font-size:12px;"><b style="color:#dc2626;">{p_name}:</b> '
                        f'<span style="color:#374151;">{simple}</span></p></div>')

    ok_items = ""
    if pairs_ok:
        ok_list = ", ".join([f"<b>{p}</b> ✓" for p in pairs_ok])
        ok_items = f'<p style="font-size:12px;color:#059669;margin:10px 0 0;">Completed OK: {ok_list}</p>'

    return f"""<html><body style="font-family:Arial,sans-serif;background:#fff7f7;padding:16px;">
<div style="max-width:540px;margin:auto;background:white;border-radius:12px;padding:20px;border:1px solid #fecaca;">
  <h2 style="margin:0 0 4px;color:#991b1b;font-size:15px;">System Issue — Some pairs missed this scan</h2>
  <p style="color:#6b7280;font-size:11px;margin:0 0 12px;">{ist_str()}</p>
  {error_items}
  {ok_items}
  <p style="color:#9ca3af;font-size:11px;margin:14px 0 0;">
    The system will retry affected pairs on the next scan (15 minutes).
    If this error repeats for more than 1 hour, check Gemini billing at ai.google.dev.</p>
</div></body></html>"""


# ── Log alert ─────────────────────────────────────────────────────────────────
def log_alert(pair, zone_level, zone_label, current_price, data, pair_conf):
    alert_log.append({
        "id":                 f"{pair}_{int(utc_now().timestamp())}",
        "alert_type":         "zone",
        "pair":               pair,
        "pair_type":          pair_conf.get("pair_type", "forex"),
        "timestamp_utc":      utc_str(),
        "ist_time":           ist_now().strftime("%H:%M"),
        "zone_level":         zone_level,
        "zone_label":         zone_label,
        "bias":               data.get("bias", ""),
        "entry":              data.get("entry", ""),
        "entry_model":        data.get("entry_model", ""),
        "entry_ready_now":    data.get("entry_ready_now", False),
        "trigger_status":     data.get("trigger_status", "not_ready"),
        "trigger_tf":         data.get("trigger_tf", ""),
        "trigger_kind":       data.get("trigger_kind", ""),
        "trigger_level":      data.get("trigger_level", 0),
        "trigger":            data.get("trigger", ""),
        "invalidate_above":   data.get("invalidate_above"),
        "invalidate_below":   data.get("invalidate_below"),
        "invalid_if":         data.get("invalid_if", ""),
        "sl":                 data.get("sl", 0),
        "tp1":                data.get("tp1", 0),
        "tp2":                data.get("tp2", 0),
        "rr_tp1":             data.get("rr_tp1", ""),
        "confidence_score":   data.get("confidence_score", 0),
        "score_breakdown":    data.get("score_breakdown", {}),
        "confluences":        data.get("confluences", []),
        "missing":            data.get("missing", []),
        "geo_flag":           data.get("geo_flag", False),
        "news_flag":          data.get("news_flag", "none"),
        "outcome":            "pending",
        "outcome_price":      None,
        "outcome_checked_at": None,
        "entry_alert_sent":   False,
        "entry_alert_sent_at": None,
        "entry_alert_price":  None,
        "invalidation_email_sent": False
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

# ── Entry-gated outcome check (aligned with weekly_review.py) ─────────────────
def check_entry_gated_outcome(alert):
    """
    Three-step entry-gated outcome logic:
    Step 1: SL hit before price reached entry → not_triggered
    Step 2: Price never reached entry after 48h → not_triggered
    Step 3: Entry reached, then SL or TP1 hit first → loss or win_tp1
    """
    pair   = alert.get('pair', '')
    symbol = next((p['symbol'] for p in config['pairs'] if p['name'] == pair), None)
    if not symbol:
        return 'pending', None
    try:
        bias = str(alert.get('bias', '')).upper()
        if bias not in ('LONG', 'SHORT'):
            return 'pending', None
        sl  = float(alert.get('sl', 0) or 0)
        tp1 = float(alert.get('tp1', 0) or 0)
        if sl <= 0 or tp1 <= 0:
            return 'pending', None
        try:
            entry = float(str(alert.get('entry', '0')).split('-')[0].strip() or 0)
        except Exception:
            entry = 0
        if entry <= 0:
            return 'pending', None

        alert_time = datetime.strptime(alert['timestamp_utc'], "%Y-%m-%d %H:%M")
        df = clean_df(yf.download(symbol,
            start=(alert_time - timedelta(hours=1)).strftime('%Y-%m-%d'),
            interval="15m", progress=False))
        if df is None:
            return 'pending', None

        entry_reached = False
        for ts, row in df.iterrows():
            try:
                ts_n = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
                if ts_n < alert_time:
                    continue
                h = float(row['High'])
                l = float(row['Low'])
            except Exception:
                continue
            if not entry_reached:
                if bias == "LONG"  and l <= sl:  return "not_triggered", None
                if bias == "SHORT" and h >= sl:  return "not_triggered", None
                if bias == "LONG"  and l <= entry <= h: entry_reached = True
                if bias == "SHORT" and l <= entry <= h: entry_reached = True
            else:
                if bias == "LONG":
                    if l <= sl:  return "loss",    sl
                    if h >= tp1: return "win_tp1", tp1
                elif bias == "SHORT":
                    if h >= sl:  return "loss",    sl
                    if l <= tp1: return "win_tp1", tp1

        if not entry_reached:
            age_h = (utc_now() - alert_time).total_seconds() / 3600
            if age_h >= 48:
                return "not_triggered", None
        return "pending", None
    except Exception as e:
        print(f"    Outcome check error ({pair}): {e}")
        return "pending", None

# ── Job 2: Entry/Invalidation monitoring (merged from alert2_engine) ──────────
def run_entry_invalidation_checks(macro_news):
    """
    Checks pending zone alerts where entry hasn't been sent yet and setup
    hasn't been invalidated. Uses cached macro news from Job 1.
    Only monitors alerts between 'Zone Alert sent' and 'Entry Alert sent or Invalidated'.
    """
    fired = 0
    for alert in alert_log:
        try:
            if alert.get("alert_type") != "zone":
                continue
            if alert.get("outcome") in ("win_tp1", "loss", "invalidated", "not_triggered"):
                continue
            if alert.get("entry_alert_sent"):
                continue
            if alert.get("invalidation_email_sent"):
                continue

            pair_conf = get_pair_conf(alert.get("pair"))
            if not pair_conf:
                continue

            symbol   = pair_conf["symbol"]
            min_conf = pair_conf.get("min_confidence", 7)

            # Fetch current price (H1 only — lightweight)
            df1 = clean_df(yf.download(symbol, period="15d", interval="1h", progress=False))
            if df1 is None:
                continue
            current_price = float(df1["Close"].iloc[-1])

            # Check proximity to entry
            entry_mid = parse_entry_mid(alert.get("entry", ""))
            if entry_mid is None:
                continue
            near_pct   = pair_conf.get("near_entry_pct", 0.08)
            entry_dist = abs(current_price - entry_mid) / entry_mid * 100
            if entry_dist > near_pct:
                # Also check invalidation levels even when price isn't near entry
                inv_above = alert.get("invalidate_above")
                inv_below = alert.get("invalidate_below")
                invalidated = False
                if inv_above and current_price > float(inv_above):
                    invalidated = True
                if inv_below and current_price < float(inv_below):
                    invalidated = True
                if invalidated:
                    send_simple_email(
                        f"INVALIDATED | {alert['pair']} | {ist_now().strftime('%H:%M IST')}",
                        build_invalidation_email_html(alert, {"invalid_if": alert.get("invalid_if", "Price breached invalidation level")},
                                                     pair_conf, round(current_price, pair_conf.get("decimal_places", 5)))
                    )
                    alert["outcome"] = "invalidated"
                    alert["outcome_checked_at"] = utc_str()
                    alert["invalidation_email_sent"] = True
                    save_alert_log()
                    fired += 1
                continue

            # Price is near entry — call Gemini for fresh trigger check
            df2 = fetch_m15_data(symbol)
            if df2 is None:
                continue

            zone_level = float(alert.get("zone_level", 0) or 0)
            zone_label = alert.get("zone_label", get_zone_label(zone_level, current_price))
            fatigue    = count_zone_alerts(alert["pair"], zone_level) if zone_level > 0 else 0

            prompt = build_zone_prompt(
                pair_conf, round(zone_level, 5), zone_label,
                round(current_price, 5), macro_news, df1, df2, fatigue
            )

            refreshed_data, error = call_gemini(prompt)
            if error or not refreshed_data:
                continue

            trigger_status = str(refreshed_data.get("trigger_status", "not_ready")).lower()
            score          = refreshed_data.get("confidence_score", 0)

            # Invalidated
            if trigger_status == "invalidated":
                send_simple_email(
                    f"INVALIDATED | {alert['pair']} | {ist_now().strftime('%H:%M IST')}",
                    build_invalidation_email_html(alert, refreshed_data, pair_conf,
                                                 round(current_price, pair_conf.get("decimal_places", 5)))
                )
                alert["outcome"] = "invalidated"
                alert["outcome_checked_at"] = utc_str()
                alert["invalidation_email_sent"] = True
                save_alert_log()
                fired += 1
                continue

            # Trigger ready
            if trigger_status != "ready":
                continue
            if not refreshed_data.get("entry_ready_now", False):
                continue
            if not refreshed_data.get("send_alert", False):
                continue
            if score < min_conf:
                continue

            # Validate response
            is_valid, reason = validate_gemini_response(
                refreshed_data, pair_conf, zone_label, current_price)
            if not is_valid:
                print(f"    Entry alert blocked for {alert['pair']}: {reason}")
                continue

            subject = f"[{score}/10] ENTRY | {alert['pair']} | {ist_now().strftime('%H:%M IST')}"
            html    = build_entry_email_html(alert, refreshed_data, pair_conf,
                                             round(current_price, pair_conf.get("decimal_places", 5)))
            send_simple_email(subject, html)

            alert["entry_alert_sent"]    = True
            alert["entry_alert_sent_at"] = utc_str()
            alert["entry_alert_price"]   = round(current_price, 5)
            save_alert_log()
            fired += 1

        except Exception as e:
            print(f"    Entry check error ({alert.get('pair', '?')}): {e}")

    return fired


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
ist_start = ist_now().strftime("%H:%M IST, %d %b %Y")
print(f"Alert engine started {ist_start} ({utc_str()} UTC)")

run_errors  = []
pairs_ok    = []
alerts_fired = 0

market_open, market_status = is_market_open()
print(f"  Market: {market_status}")
if not market_open:
    print("  Exiting — market closed.")
    exit(0)

# ── Job 1: Scan for new zone alerts ──────────────────────────────────────────
macro_news = fetch_macro_news()

for pair_conf in config["pairs"]:
    symbol      = pair_conf["symbol"]
    name        = pair_conf["name"]
    prox        = pair_conf["proximity_pct"]
    min_touches = pair_conf.get("min_touches", 1)
    min_conf    = pair_conf.get("min_confidence", 7)

    print(f"  Scanning {name}...")

    try:
        zones, current_price, df1 = detect_zones_and_candles(symbol, min_touches)

        if current_price is None:
            print(f"    No data for {name}. Skipping.")
            log_scan(name, "error", "No market data returned.")
            run_errors.append(f"{name}: No market data returned for this pair.")
            continue

        if not zones:
            log_scan(name, "no_zone_found", "No valid zones detected.")
            pairs_ok.append(name)
            continue

        zones_in_proximity = 0
        zone_alerted = False

        for zone_level, touches in zones:
            if zone_alerted:
                break

            dist_pct = abs(current_price - zone_level) / zone_level * 100
            if dist_pct > prox:
                continue

            zones_in_proximity += 1
            zone_label = get_zone_label(zone_level, current_price)

            if not should_alert_zone(name, zone_level, current_price, prox):
                log_scan(name, "blocked_revisit",
                         "Zone already alerted, price hasn't moved enough.", zone_level)
                continue

            df2 = fetch_m15_data(symbol)
            if df2 is None:
                log_scan(name, "error", "No M15 data.", zone_level)
                continue

            fatigue = count_zone_alerts(name, zone_level)
            dp = pair_conf.get("decimal_places", 5)
            print(f"    ZONE HIT: {name} {zone_label} @ {zone_level:.{dp}f} "
                  f"dist:{dist_pct:.2f}% fatigue:{fatigue}")

            prompt = build_zone_prompt(
                pair_conf,
                round(zone_level, dp),
                zone_label,
                round(current_price, dp),
                macro_news, df1, df2, fatigue
            )

            data, error = call_gemini(prompt)

            if error:
                print(f"    {error}")
                log_scan(name, "error", error, zone_level)
                run_errors.append(f"{name}: {error}")
                continue

            # ── Python post-validation ────────────────────────────────────
            is_valid, reason = validate_gemini_response(
                data, pair_conf, zone_label, current_price)

            if not is_valid:
                print(f"    {name} blocked by validation: {reason}")
                log_scan(name, "rejected_validation", reason, zone_level)
                continue

            score = data.get("confidence_score", 0)

            if not data.get("send_alert", False):
                reason = data.get("confidence_reason", "Gemini rejected setup.")
                log_scan(name, "rejected_gemini", reason, zone_level)
                continue

            if not data.get("gates_passed", False):
                reason = f"Hard gates failed. {data.get('confidence_reason', '')}"
                log_scan(name, "rejected_gates", reason, zone_level)
                continue

            if score < min_conf:
                reason = f"Score {score}/10 below {min_conf}. {data.get('confidence_reason', '')}"
                log_scan(name, "rejected_low_score", reason, zone_level)
                continue

            # ── Generate charts ───────────────────────────────────────────
            levels = {
                'zone': zone_level,
                'current': current_price,
                'entry': data.get('entry', ''),
                'sl': data.get('sl', 0),
                'tp1': data.get('tp1', 0),
                'tp2': data.get('tp2', 0)
            }

            chart1 = generate_chart(df1, f"{name} — H1", levels, data, pair_conf)
            chart2 = generate_chart(df2, f"{name} — M15", levels, data, pair_conf)

            # ── Build and send email ──────────────────────────────────────
            html = build_zone_email_html(
                data, name, pair_conf,
                round(zone_level, dp),
                zone_label,
                round(current_price, dp),
                chart1, chart2
            )

            subject = f"[{score}/10] {name} | {zone_label} | {ist_now().strftime('%H:%M IST')}"

            send_email(subject, html, chart1, chart2)
            log_alert(name, round(zone_level, dp), zone_label,
                      round(current_price, dp), data, pair_conf)
            log_scan(name, "alert_sent",
                     f"Zone alert sent at score {score}/10.", zone_level)
            record_zone_alert(name, zone_level, current_price)

            system_status["last_trade_alert_utc"] = utc_str()
            alerts_fired += 1
            zone_alerted = True
            print(f"    ✓ Sent: {name} [{score}/10]")

        if zones_in_proximity == 0:
            log_scan(name, "zone_outside_proximity",
                     "Zones detected but none near current price.")

        if not zone_alerted:
            pairs_ok.append(name)

    except Exception as e:
        print(f"    Error: {str(e)}")
        log_scan(name, "error", f"Pair-level error: {str(e)}")
        run_errors.append(f"{name}: {str(e)}")

# ── Job 2: Check pending alerts for entry/invalidation ────────────────────────
print(f"\n  Job 2: Checking pending alerts for entry/invalidation...")
cached_news = get_cached_macro_news()
entry_fired = run_entry_invalidation_checks(cached_news)
print(f"  Job 2 complete. {entry_fired} entry/invalidation alert(s) sent.")

# ── System status emails ──────────────────────────────────────────────────────
if alerts_fired == 0 and entry_fired == 0 and not run_errors and should_send_ok():
    print("  Sending system OK email...")
    send_simple_email("System OK — No trade setups in last 3 hours", build_ok_email_html())
    system_status["last_ok_email_utc"] = utc_str()

if run_errors and should_send_error():
    print("  Sending system error email...")
    send_simple_email(
        f"System Issue — {len(run_errors)} pair(s) had errors",
        build_error_email_html(run_errors, pairs_ok)
    )
    system_status["last_error_email_utc"] = utc_str()

# ── Save all state ────────────────────────────────────────────────────────────
save_alert_log()
save_json(SCAN_LOG_FILE, scan_log)
save_json(SYSTEM_STATUS_FILE, system_status)
save_visit_state()

total_actions = alerts_fired + entry_fired
print(f"\nAlert log: {len(alert_log)} entries | Scan log: {len(scan_log)} entries")
print(f"Scan complete. {alerts_fired} zone alert(s), {entry_fired} entry/invalidation(s).")
