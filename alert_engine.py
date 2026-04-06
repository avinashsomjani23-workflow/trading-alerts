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

def spread_to_price(pair_conf):
    """Convert spread_pips to price distance for this pair."""
    spread = pair_conf.get("spread_pips", 0)
    if pair_conf.get("pair_type") == "forex":
        if "JPY" in pair_conf["name"]:
            return spread * 0.01
        else:
            return spread * 0.0001
    else:
        return spread

def min_sl_to_price(pair_conf):
    """Convert min_sl_pips (fallback) to price distance."""
    min_sl = pair_conf.get("min_sl_pips", 15)
    if pair_conf.get("pair_type") == "forex":
        if "JPY" in pair_conf["name"]:
            return min_sl * 0.01
        else:
            return min_sl * 0.0001
    else:
        return min_sl

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

# ── Zone cooldown — prevents spam, allows re-entry after resolution ───────────
def save_visit_state():
    save_json(VISIT_FILE, visit_state)

def has_active_trade(pair, zone_level):
    """Check if there's an unresolved TRADE READY alert on this zone."""
    for a in alert_log:
        if a.get("pair") != pair:
            continue
        if a.get("alert_type") != "zone":
            continue
        if a.get("alert_stage") != "trade_ready":
            continue
        if a.get("outcome") != "pending":
            continue
        try:
            logged_zone = float(a.get("zone_level", 0))
            if logged_zone > 0 and abs(logged_zone - zone_level) / zone_level * 100 < 0.3:
                return True
        except Exception:
            pass
    return False

def should_send_approaching(pair, zone_level):
    """Allow APPROACHING if no active trade and last approaching was 2+ hours ago."""
    if has_active_trade(pair, zone_level):
        return False
    key = f"{pair}_{round(zone_level, 4)}"
    state = visit_state.get(key, {})
    last_appr = state.get("approaching_at_utc")
    if last_appr:
        elapsed = hours_since(last_appr)
        if elapsed is not None and elapsed < 2.0:
            return False
    return True

def should_send_trade_ready(pair, zone_level):
    """Allow TRADE READY if no active trade on this zone."""
    return not has_active_trade(pair, zone_level)

def record_approaching(pair, zone_level):
    key = f"{pair}_{round(zone_level, 4)}"
    state = visit_state.get(key, {})
    state["approaching_at_utc"] = utc_str()
    visit_state[key] = state
    save_visit_state()

def record_trade_ready(pair, zone_level):
    key = f"{pair}_{round(zone_level, 4)}"
    state = visit_state.get(key, {})
    state["trade_ready_at_utc"] = utc_str()
    visit_state[key] = state
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

def compute_sl_buffer(pair_conf, atr_value):
    """Dynamic SL buffer = spread + (multiplier × ATR14_H1).
    Falls back to min_sl_pips if ATR unavailable."""
    multiplier = config.get("scoring", {}).get("atr_buffer_multiplier", 0.2)
    spread_price = spread_to_price(pair_conf)
    if atr_value is not None and atr_value > 0:
        buffer = spread_price + (multiplier * atr_value)
    else:
        buffer = min_sl_to_price(pair_conf)
        print(f"    ATR unavailable for {pair_conf['name']}, using min_sl fallback: {buffer}")
    return buffer

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
    try:
        url   = "https://api.rss2json.com/v1/api.json?rss_url=https://www.fxstreet.com/rss/news&count=5"
        r     = requests.get(url, timeout=10)
        items = r.json().get("items", [])
        for i in items:
            headlines.append(f"[FXStreet] {i['title']}")
    except Exception:
        headlines.append("[FXStreet] Unavailable")
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
    save_json(MACRO_CACHE_FILE, {"cached_at": utc_str(), "news": news_text})
    return news_text

def get_cached_macro_news():
    cache = load_json(MACRO_CACHE_FILE, {})
    return cache.get("news", "Macro news unavailable (cache miss).")

def format_candles(df, label, n=20):
    if df is None or df.empty:
        return f"{label}: No data\n"
    lines = [f"{label} (last {n} candles):"]
    for i in range(max(0, len(df)-n), len(df)):
        try:
            ts  = df.index[i]
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
                      macro_news, df1, df2, fatigue_count, atr_value):
    name        = pair_conf["name"]
    pair_type   = pair_conf.get("pair_type", "forex")
    dp          = pair_conf.get("decimal_places", 5)
    min_conf    = pair_conf.get("min_confidence", 7)
    structure_tf = pair_conf.get("structure_tf", ["H1"])
    extra_gate  = pair_conf.get("extra_gate")
    risk_dollar = config["account"]["balance"] * config["account"]["risk_percent"] / 100
    scoring     = config.get("scoring", {})
    ist_time    = ist_now().strftime("%H:%M IST, %d %b %Y")
    atr_str     = f"{atr_value:.5f}" if atr_value else "unavailable"

    if "M15" in structure_tf and "H1" in structure_tf:
        tf_rule = "H1 BOS or H1 CHoCH or M15 BOS or M15 CHoCH — any ONE of these passes the gate."
    else:
        tf_rule = "H1 BOS or H1 CHoCH ONLY. M15 structure is NOT accepted for this pair."

    extra_gate_text = ""
    if extra_gate == "liquidity_swept_required":
        extra_gate_text = (f"\nADDITIONAL HARD GATE FOR {name}: Liquidity MUST be swept before zone entry. "
                           "If no liquidity sweep is visible in the candle data, set send_alert to false regardless of score.")
    elif extra_gate == "macro_alignment_required":
        extra_gate_text = (f"\nADDITIONAL HARD GATE FOR {name}: Macro context MUST align with technical bias. "
                           "If DXY direction, safe-haven flows, or geopolitical context contradicts the trade direction, "
                           "set send_alert to false. Gold is driven by macro first, technicals second.")

    fatigue_thresh = scoring.get("zone_fatigue_threshold", 3)
    if fatigue_count >= fatigue_thresh + 2:
        fatigue_rule = f"Zone alerted {fatigue_count} times in 30 days. HEAVILY FATIGUED. Score zone_freshness as 0."
    elif fatigue_count >= fatigue_thresh:
        fatigue_rule = f"Zone alerted {fatigue_count} times in 30 days. FATIGUED. Score zone_freshness as 0."
    else:
        fatigue_rule = f"Zone alerted {fatigue_count} times in 30 days. FRESH. Score zone_freshness as 1.0."

    nb_before = scoring.get("news_blackout_hours_before", 2)
    nb_after  = scoring.get("news_blackout_hours_after", 1)

    return f"""You are an elite institutional SMC trader. Analyze this zone setup with extreme precision.

PAIR: {name} ({pair_type}) | ZONE: {zone_label} at {zone_level} | PRICE NOW: {current_price}
TIME: {ist_time}
ACCOUNT: ${config["account"]["balance"]} | RISK: {config["account"]["risk_percent"]}% = ${risk_dollar:.0f}
PRICE FORMAT: {dp} decimal places for this pair.
ATR(14, H1): {atr_str}

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

HARD SCORING RULES — VIOLATIONS WILL BE REJECTED:
- NEVER score ANY item above its stated maximum. Session Alignment max is 0.5. Liquidity Swept max is 2.5. Etc.
- If you score a partial value (e.g. 0.5 on Multi-TF Alignment), you MUST explain which timeframe confirmed and which did not.
- confidence_score MUST equal the exact arithmetic sum of all 7 items above. Python will verify and reject mismatches.
TOTAL POSSIBLE: 10.0 | MINIMUM TO SEND: {min_conf}

═══════════════════════════════════════════════════════════════
ENTRY, SL, TP RULES:
═══════════════════════════════════════════════════════════════
ENTRY: 50% midpoint of OB candle body. If FVG overlaps OB, use FVG edge (top for longs, bottom for shorts).
SL: Place SL at OB wick extreme (low of OB candle for LONG, high for SHORT).
  Python will add a dynamic volatility buffer on top of your SL. Do NOT add buffer yourself.
  Just place SL at the exact OB wick extreme.
TP1: Next significant opposing zone or liquidity pool. Must achieve 2:1 RR minimum.
TP2: Second opposing zone beyond TP1.
PARTIAL CLOSE: Close 50% at TP1, move SL to breakeven, let remaining run to TP2.

═══════════════════════════════════════════════════════════════
TRIGGER & ENTRY READINESS:
═══════════════════════════════════════════════════════════════
- entry_ready_now: Set to true ONLY if a trigger candle pattern is ALREADY confirmed on the
  LATEST CLOSED candle (M15 or H1). This means the most recent closed candle shows:
  - Bullish/bearish engulfing at the zone, OR
  - CHoCH (change of character) at the zone, OR
  - BOS (break of structure) confirming zone defense.
  If the latest candle does NOT show a completed confirmation pattern, set entry_ready_now to false.
  Do NOT set true based on the current forming candle — only closed candles count.
- trigger_status: "ready" if entry_ready_now is true, "not_ready" if still waiting.
- invalidate_above / invalidate_below: numeric price where setup is cancelled.
- bias MUST be "LONG" or "SHORT". Never "WAIT". If the setup is not tradeable, set send_alert to false.

THIS SYSTEM ONLY TRADES ZONE RETESTS (OB + FVG setups).
NEVER suggest breakout entries, momentum entries, or trend-following entries.
If current price action is a breakout without a valid zone retest, set send_alert to false.

Return ONLY raw JSON. No markdown. No code fences. No text outside the JSON.

CRITICAL OUTPUT RULE: If ANY hard gate fails, return ONLY this short-form JSON and STOP:
{{
  "send_alert": false,
  "gates_passed": false,
  "gate_structure": "cite specific candle or false",
  "gate_ob": "cite OB candle or false",
  "gate_rr": "cite RR value or false",
  "confidence_score": 0.0,
  "confidence_reason": "one sentence why gates failed",
  "bias": "LONG or SHORT"
}}

If ALL gates pass, return the full JSON:
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

  "trigger_status": "not_ready or ready",
  "trigger_tf": "M15 or H1",
  "trigger_kind": "choch or bos or engulf or break_retest",
  "trigger": "exact candle pattern needed — cite what the latest candle shows or what must happen next",

  "invalidate_above": null,
  "invalidate_below": null,
  "invalid_if": "exact condition that cancels this trade",

  "sl": 0.0,
  "sl_note": "one sentence on SL logic — place at OB wick extreme, Python adds buffer",
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
  "lq_sweep_price": 0.0,
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
            },
            "maxOutputTokens": 2500
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
def validate_gemini_response(data, pair_conf, zone_label, current_price, atr_value):
    """
    Validates Gemini's JSON response. Returns (is_valid, reason, buffer_applied).
    Checks: gates, score math, SL sanity, bias sanity, ATR-based buffer.
    """
    name   = pair_conf["name"]
    dp     = pair_conf.get("decimal_places", 5)

    if not data.get("gates_passed", False):
        return False, "Hard gates not passed", 0

    bias = str(data.get("bias", "")).upper()
    if bias not in ("LONG", "SHORT"):
        return False, f"Bias is '{bias}' — must be LONG or SHORT", 0

    # Clamp each score item to its maximum, recalculate
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

    # Parse entry/SL/TP
    try:
        entry = float(str(data.get("entry", 0)).split("-")[0].strip() or 0)
        sl    = float(data.get("sl", 0) or 0)
        tp1   = float(data.get("tp1", 0) or 0)
        if entry <= 0 or sl <= 0 or tp1 <= 0:
            return False, "Missing entry/SL/TP1 values", 0
        if bias == "LONG" and sl >= entry:
            return False, f"SL ({sl}) above entry ({entry}) for LONG", 0
        if bias == "SHORT" and sl <= entry:
            return False, f"SL ({sl}) below entry ({entry}) for SHORT", 0
    except Exception:
        return False, "Could not parse entry/SL/TP1", 0

    # SL widening removed — SL is taken as Gemini places it at OB wick extreme
    buffer = 0

    # Bias matches zone type
    if "Demand" in zone_label and bias == "SHORT":
        return False, "SHORT bias at Demand zone — contradictory", buffer
    if "Supply" in zone_label and bias == "LONG":
        return False, "LONG bias at Supply zone — contradictory", buffer

    return True, "OK", buffer


# ── Chart generation — clean, no volume, smart labels ─────────────────────────
def generate_chart(df, title, levels, data, pair_conf):
    try:
        if df is None or df.empty:
            return None
        dp = pair_conf.get("decimal_places", 5)
        df_plot = df.tail(30).copy().reset_index(drop=True)
        for col in ['Open','High','Low','Close']:
            if col not in df_plot.columns:
                return None

        fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), facecolor='#131722')
        ax.set_facecolor('#131722')
        for s in ax.spines.values():
            s.set_color('#2a2a3e')

        # Draw candles
        for i, row in df_plot.iterrows():
            try:
                o = float(row['Open']); h = float(row['High'])
                l = float(row['Low']);  c = float(row['Close'])
                if any(np.isnan(v) for v in [o,h,l,c]):
                    continue
                col_c = '#26a69a' if c >= o else '#ef5350'
                ax.plot([i,i], [l,h], color=col_c, linewidth=1.2, zorder=2)
                body = abs(c-o) or (h-l) * 0.02
                ax.add_patch(patches.Rectangle(
                    (i-0.35, min(o,c)), 0.7, body,
                    facecolor=col_c, linewidth=0, alpha=0.95, zorder=3))
            except Exception:
                continue

        n = len(df_plot)

        # OB zone shading — uniform amber color regardless of direction
        ob_top    = float(data.get('ob_top', 0) or 0)
        ob_bottom = float(data.get('ob_bottom', 0) or 0)
        if ob_top > 0 and ob_bottom > 0 and abs(ob_top - ob_bottom) > 0:
            ob_color = '#E8A838'
            ax.add_patch(patches.Rectangle(
                (max(0, n-15), ob_bottom), min(n, 15), ob_top - ob_bottom,
                facecolor=ob_color, edgecolor=ob_color, linewidth=2.0,
                alpha=0.20, zorder=1, linestyle='-'))
            ax.text(max(0.5, n-14.5), ob_top + (ob_top - ob_bottom) * 0.08, "OB",
                color=ob_color, fontsize=9, va='bottom', fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#131722', edgecolor=ob_color, alpha=0.9))

        # FVG zone shading — purple with dashed border
        fvg_top    = float(data.get('fvg_top', 0) or 0)
        fvg_bottom = float(data.get('fvg_bottom', 0) or 0)
        if fvg_top > 0 and fvg_bottom > 0 and abs(fvg_top - fvg_bottom) > 0:
            fvg_color = '#7C6FD4'
            ax.add_patch(patches.Rectangle(
                (max(0, n-15), fvg_bottom), min(n, 15), fvg_top - fvg_bottom,
                facecolor=fvg_color, edgecolor=fvg_color, linewidth=1.5,
                alpha=0.15, zorder=1, linestyle='--'))
            ax.text(max(0.5, n-14.5), fvg_bottom - (fvg_top - fvg_bottom) * 0.08, "FVG",
                color=fvg_color, fontsize=9, va='top', fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#131722', edgecolor=fvg_color, alpha=0.9))

        # Liquidity sweep marker
        lq_price = float(data.get('lq_sweep_price', 0) or 0)
        if lq_price > 0:
            ax.plot(n-2, lq_price, marker='v', color='#F39C12', markersize=10, zorder=6)
            ax.text(n-1, lq_price, " LQ", color='#F39C12', fontsize=8, va='center',
                fontweight='bold', zorder=5)

        # ── Smart label stacking ──────────────────────────────────────────
        level_cfg = {
            'tp2':     ('#1e8449', '--', 1.0, 'TP2'),
            'tp1':     ('#27ae60', '-',  1.5, 'TP1'),
            'entry':   ('#e67e22', '-',  1.5, 'ENTRY'),
            'zone':    ('#9b59b6', '--', 0.8, 'ZONE'),
            'current': ('#ffffff', ':',  0.8, 'NOW'),
            'sl':      ('#e74c3c', '-',  1.5, 'SL'),
        }

        # Collect all valid levels
        label_items = []
        for key, (color, style, width, lbl) in level_cfg.items():
            val = levels.get(key, 0)
            try:
                price = float(str(val).split('-')[0].strip()) if val else 0
            except Exception:
                price = 0
            if price > 0:
                label_items.append({'price': price, 'label': lbl, 'color': color,
                                    'style': style, 'width': width})

        # Sort by price ascending
        label_items.sort(key=lambda x: x['price'])

        # Compute minimum spacing (3.5% of visible price range)
        all_prices = [li['price'] for li in label_items]
        if all_prices:
            price_range = max(all_prices) - min(all_prices)
            if price_range <= 0:
                price_range = max(all_prices) * 0.01
            min_spacing = price_range * 0.035
        else:
            min_spacing = 0

        # Resolve overlaps by shifting display position
        display_positions = []
        for i, item in enumerate(label_items):
            display_y = item['price']
            for prev_y in display_positions:
                if abs(display_y - prev_y) < min_spacing:
                    display_y = prev_y + min_spacing
            display_positions.append(display_y)

        # Draw level lines and labels
        for i, item in enumerate(label_items):
            actual_price = item['price']
            display_y    = display_positions[i]
            color  = item['color']
            style  = item['style']
            width  = item['width']
            lbl    = item['label']

            ax.axhline(y=actual_price, color=color, linestyle=style,
                       linewidth=width, alpha=0.85, zorder=4)

            # If display position differs from actual, draw a thin connector
            if abs(display_y - actual_price) > min_spacing * 0.1:
                ax.plot([n+0.5, n+2], [actual_price, display_y],
                        color=color, linewidth=0.6, alpha=0.4, zorder=4)

            ax.text(n + 2.5, display_y, f" {lbl}: {actual_price:,.{dp}f}",
                color=color, fontsize=8, va='center', fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e',
                          edgecolor=color, alpha=0.9, linewidth=0.8))

        # IST timestamps on x-axis
        tick_positions = list(range(0, n, max(1, n // 6)))
        tick_labels = []
        for pos in tick_positions:
            try:
                ts = df.iloc[-(n - pos)].name
                if hasattr(ts, 'strftime'):
                    if hasattr(ts, 'tz_localize') and ts.tzinfo is None:
                        ts_ist = ts + timedelta(hours=5, minutes=30)
                    else:
                        ts_ist = ts + timedelta(hours=5, minutes=30)
                    tick_labels.append(ts_ist.strftime('%d %b\n%H:%M'))
                else:
                    tick_labels.append('')
            except Exception:
                tick_labels.append('')

        ax.set_title(title, color='#dddddd', fontsize=11, pad=8, fontweight='bold', loc='left')
        ax.tick_params(colors='#888', labelsize=8)
        ax.yaxis.tick_right()
        ax.yaxis.set_tick_params(labelcolor='#aaa', labelsize=8)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7, color='#888')
        ax.set_xlim(-1, n + 16)

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


# ── APPROACHING email HTML ────────────────────────────────────────────────────
def build_approaching_email_html(data, pair, pair_conf, zone_level, zone_label,
                                  current_price, chart1_b64, chart2_b64, fatigue_count):
    dp          = pair_conf.get("decimal_places", 5)
    ist_time    = ist_now().strftime("%H:%M IST, %d %b %Y")
    score       = data.get("confidence_score", 0)
    bias        = data.get("bias", "—")
    bias_color  = "#e74c3c" if bias == "SHORT" else "#27ae60"
    score_color = "#27ae60" if score >= 8 else "#e67e22" if score >= 7 else "#e74c3c"

    news_flag = data.get("news_flag", "none")
    news_html = ""
    if news_flag and news_flag.lower() != "none":
        news_html = (f'<div style="background:#fff3cd;padding:10px 20px;border-left:4px solid #f39c12;'
                     f'font-size:12px;color:#856404;"><b>⚠ NEWS:</b> {news_flag}</div>')

    entry_p = fmt_price(data.get("entry", 0), pair_conf)
    sl_p    = fmt_price(data.get("sl", 0), pair_conf)
    tp1_p   = fmt_price(data.get("tp1", 0), pair_conf)
    tp2_p   = fmt_price(data.get("tp2", 0), pair_conf)
    zone_p  = fmt_price(zone_level, pair_conf)
    now_p   = fmt_price(current_price, pair_conf)

    fatigue_label = f"Zone tested {fatigue_count}× in 30 days"

    action_box = f"""<div style="background:#1a1a2e;padding:16px 20px;border-radius:10px;margin-bottom:16px;">
      <p style="color:#f39c12;margin:0;font-size:11px;text-transform:uppercase;letter-spacing:1px;">⏳ APPROACHING ZONE — LEVELS READY, WAITING FOR CONFIRMATION</p>
      <p style="color:white;margin:10px 0 0;font-size:13px;">
        <b>Pre-set your order:</b> {'SELL' if bias=='SHORT' else 'BUY'} LIMIT at {entry_p}</p>
      <p style="color:white;margin:4px 0 0;font-size:13px;">
        <b>Levels:</b> SL: {sl_p} &nbsp;|&nbsp; TP1: {tp1_p} &nbsp;|&nbsp; TP2: {tp2_p}</p>
      <p style="color:#8899bb;margin:10px 0 0;font-size:11px;">
        Place limit order now OR wait for the TRADE READY email to confirm trigger.</p>
      <p style="color:#8899bb;margin:2px 0 0;font-size:11px;">
        Close 50% at TP1, move SL to breakeven, let rest run to TP2</p>
      <p style="color:#445566;margin:8px 0 0;font-size:10px;">
        System checks every 15 min. You'll receive TRADE READY or INVALIDATION next.</p>
    </div>"""

    conf_table = _build_scorecard_html(data, score, score_color)
    chart1_html = (f'<img src="cid:chart_h1" style="width:100%;border-radius:8px;display:block;margin-bottom:12px;" />'
                   if chart1_b64 else '')
    chart2_html = (f'<img src="cid:chart_m15" style="width:100%;border-radius:8px;display:block;margin-bottom:12px;" />'
                   if chart2_b64 else '')
    miss_items = _build_missing_html(data)

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:12px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">
  <div style="background:#1a1a2e;padding:16px 20px;">
    <h2 style="color:white;margin:0;font-size:16px;">APPROACHING: {pair} — {zone_label}</h2>
    <p style="color:#8899bb;margin:4px 0 0;font-size:11px;">{ist_time}</p>
  </div>
  <div style="background:{bias_color};padding:10px 20px;">
    <p style="color:white;margin:0;font-size:14px;font-weight:bold;">{bias} — {data.get("bias_reason","")}</p>
  </div>
  {news_html}
  <div style="padding:16px 20px;">
    {action_box}
    <p style="font-size:12px;color:#666;margin:0 0 4px;">Zone: <b>{zone_label}</b> at {zone_p} &nbsp;|&nbsp; Now: {now_p} &nbsp;|&nbsp; R:R {data.get("rr_tp1","—")}:1</p>
    <p style="font-size:11px;color:#888;margin:0 0 12px;">{fatigue_label}</p>
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
      <p style="font-size:10px;color:#f39c12;margin:0 0 4px;font-weight:bold;">WAITING FOR</p>
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


# ── TRADE READY email HTML ────────────────────────────────────────────────────
def build_trade_ready_email_html(data, pair, pair_conf, zone_level, zone_label,
                                  current_price, chart1_b64, chart2_b64, fatigue_count):
    dp          = pair_conf.get("decimal_places", 5)
    ist_time    = ist_now().strftime("%H:%M IST, %d %b %Y")
    score       = data.get("confidence_score", 0)
    bias        = data.get("bias", "—")
    bias_color  = "#e74c3c" if bias == "SHORT" else "#27ae60"
    score_color = "#27ae60" if score >= 8 else "#e67e22" if score >= 7 else "#e74c3c"

    news_flag = data.get("news_flag", "none")
    news_html = ""
    if news_flag and news_flag.lower() != "none":
        news_html = (f'<div style="background:#fff3cd;padding:10px 20px;border-left:4px solid #f39c12;'
                     f'font-size:12px;color:#856404;"><b>⚠ NEWS:</b> {news_flag}</div>')

    entry_p = fmt_price(data.get("entry", 0), pair_conf)
    sl_p    = fmt_price(data.get("sl", 0), pair_conf)
    tp1_p   = fmt_price(data.get("tp1", 0), pair_conf)
    tp2_p   = fmt_price(data.get("tp2", 0), pair_conf)
    zone_p  = fmt_price(zone_level, pair_conf)
    now_p   = fmt_price(current_price, pair_conf)

    fatigue_label = f"Zone tested {fatigue_count}× in 30 days"

    action_box = f"""<div style="background:#27ae60;padding:16px 20px;border-radius:10px;margin-bottom:16px;">
      <p style="color:white;margin:0;font-size:11px;text-transform:uppercase;letter-spacing:1px;opacity:0.8;">✅ TRADE READY — PLACE ORDER NOW</p>
      <p style="color:white;margin:8px 0 0;font-size:16px;font-weight:bold;">
        {'SELL' if bias=='SHORT' else 'BUY'} LIMIT at {entry_p}</p>
      <p style="color:white;margin:4px 0 0;font-size:14px;">SL: {sl_p} &nbsp;|&nbsp; TP1: {tp1_p} &nbsp;|&nbsp; TP2: {tp2_p}</p>
      <p style="color:#d5f5e3;margin:8px 0 0;font-size:11px;">Close 50% at TP1, move SL to breakeven, let rest run to TP2</p>
    </div>"""

    conf_table = _build_scorecard_html(data, score, score_color)
    chart1_html = (f'<img src="cid:chart_h1" style="width:100%;border-radius:8px;display:block;margin-bottom:12px;" />'
                   if chart1_b64 else '')
    chart2_html = (f'<img src="cid:chart_m15" style="width:100%;border-radius:8px;display:block;margin-bottom:12px;" />'
                   if chart2_b64 else '')
    miss_items = _build_missing_html(data)

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:12px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">
  <div style="background:#1a1a2e;padding:16px 20px;">
    <h2 style="color:white;margin:0;font-size:16px;">TRADE READY: {pair} — {zone_label}</h2>
    <p style="color:#8899bb;margin:4px 0 0;font-size:11px;">{ist_time}</p>
  </div>
  <div style="background:{bias_color};padding:10px 20px;">
    <p style="color:white;margin:0;font-size:14px;font-weight:bold;">{bias} — {data.get("bias_reason","")}</p>
  </div>
  {news_html}
  <div style="padding:16px 20px;">
    {action_box}
    <p style="font-size:12px;color:#666;margin:0 0 4px;">Zone: <b>{zone_label}</b> at {zone_p} &nbsp;|&nbsp; Now: {now_p} &nbsp;|&nbsp; R:R {data.get("rr_tp1","—")}:1</p>
    <p style="font-size:11px;color:#888;margin:0 0 12px;">{fatigue_label}</p>
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


# ── Shared email builder helpers ──────────────────────────────────────────────
def _build_scorecard_html(data, score, score_color):
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
    return (f'<table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;margin-bottom:16px;">'
            f'<tr style="background:#1a1a2e;">'
            f'<td colspan="3" style="padding:8px 10px;color:white;font-size:11px;font-weight:bold;letter-spacing:1px;">SCORECARD BREAKDOWN</td>'
            f'</tr>{conf_rows}'
            f'<tr style="background:#f4f4f8;">'
            f'<td colspan="2" style="padding:8px 10px;font-size:13px;font-weight:bold;color:#1a1a2e;">TOTAL</td>'
            f'<td style="padding:8px 10px;font-size:14px;font-weight:bold;color:{score_color};text-align:center;">{score}/10</td>'
            f'</tr></table>')

def _build_missing_html(data):
    miss_items = ""
    for m in data.get("missing", []):
        if isinstance(m, dict):
            miss_items += (f'<p style="font-size:12px;color:#c62828;margin:4px 0;padding:6px 10px;background:#ffebee;border-radius:6px;">'
                           f'✗ <b>{m.get("item","")}</b> — {m.get("reason","")}</p>')
    return miss_items


# ── Invalidation email — references original alert ───────────────────────────
def build_invalidation_email_html(alert, reason_text, pair_conf, current_price):
    ist_time = ist_now().strftime("%H:%M IST, %d %b %Y")
    # Reference back to original alert
    orig_ist = alert.get("ist_time", "")
    orig_utc = alert.get("timestamp_utc", "")
    try:
        orig_dt = datetime.strptime(orig_utc, "%Y-%m-%d %H:%M") + timedelta(hours=5, minutes=30)
        orig_ref = orig_dt.strftime("%d %b %Y, %H:%M IST")
    except Exception:
        orig_ref = f"{orig_ist} (UTC: {orig_utc})"

    entry_p = fmt_price(alert.get("entry", 0), pair_conf)
    sl_p    = fmt_price(alert.get("sl", 0), pair_conf)
    tp1_p   = fmt_price(alert.get("tp1", 0), pair_conf)

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
      <tr><td style="padding:4px 0;color:#555;width:130px;">Original Alert</td><td style="padding:4px 0;font-weight:bold;">{orig_ref}</td></tr>
      <tr><td style="padding:4px 0;color:#555;">Pair</td><td style="padding:4px 0;font-weight:bold;">{alert.get("pair","")}</td></tr>
      <tr><td style="padding:4px 0;color:#555;">Bias was</td><td style="padding:4px 0;font-weight:bold;">{alert.get("bias","")}</td></tr>
      <tr><td style="padding:4px 0;color:#555;">Entry was</td><td style="padding:4px 0;">{entry_p}</td></tr>
      <tr><td style="padding:4px 0;color:#555;">SL was</td><td style="padding:4px 0;">{sl_p}</td></tr>
      <tr><td style="padding:4px 0;color:#555;">TP1 was</td><td style="padding:4px 0;">{tp1_p}</td></tr>
      <tr><td style="padding:4px 0;color:#555;">Current price</td><td style="padding:4px 0;font-weight:bold;">{fmt_price(current_price, pair_conf)}</td></tr>
      <tr><td style="padding:4px 0;color:#555;">Reason</td><td style="padding:4px 0;color:#dc2626;font-weight:bold;">{reason_text}</td></tr>
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
    error_items = ""
    for line in error_lines[:10]:
        parts  = line.split(":", 1)
        p_name = parts[0].strip() if len(parts) > 1 else "Unknown"
        err    = parts[1].strip() if len(parts) > 1 else line
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


# ── Log alert (TRADE READY only — APPROACHING goes to scan_log) ───────────────
def log_alert(pair, zone_level, zone_label, current_price, data, pair_conf,
              alert_stage, atr_value, buffer_applied, fatigue_count):
    # Check if an approaching was sent earlier for this zone
    key = f"{pair}_{round(zone_level, 4)}"
    approaching_at = visit_state.get(key, {}).get("approaching_at_utc")

    alert_log.append({
        "id":                 f"{pair}_{int(utc_now().timestamp())}",
        "alert_type":         "zone",
        "alert_stage":        alert_stage,
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
        "atr_h1":             round(atr_value, 6) if atr_value else None,
        "buffer_applied":     round(buffer_applied, 6) if buffer_applied else None,
        "zone_tests_30d":     fatigue_count,
        "approaching_sent_at": approaching_at,
        "outcome":            "pending",
        "outcome_price":      None,
        "outcome_checked_at": None,
        "entry_alert_sent":   alert_stage == "trade_ready",
        "entry_alert_sent_at": utc_str() if alert_stage == "trade_ready" else None,
        "entry_alert_price":  round(current_price, 5) if alert_stage == "trade_ready" else None,
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


# ── Job 2: Pure Python invalidation — ZERO Gemini calls ──────────────────────
def run_invalidation_checks(current_prices):
    """
    Checks ALL pending alerts for invalidation using pure Python price checks.
    No Gemini calls. Uses current prices already fetched in Job 1.

    CRITICAL: If entry price was reached AFTER the Trade Ready alert was sent,
    the trade is LIVE. Invalidation is permanently skipped — the outcome checker
    (SL vs TP1) handles live trades from here.

    Checks (only if entry NOT yet filled):
      1. Invalidation levels breached
      2. Price ran away (2x risk distance beyond entry without filling)
      3. 48-hour timeout (entry never reached)
    """
    fired = 0

    # Pre-fetch M15 data once per pair that has pending alerts
    pairs_with_pending = set()
    for alert in alert_log:
        if (alert.get("alert_type") == "zone"
            and alert.get("outcome") == "pending"
            and not alert.get("invalidation_email_sent")):
            pairs_with_pending.add(alert.get("pair"))

    m15_cache = {}
    for pair_name in pairs_with_pending:
        pc = get_pair_conf(pair_name)
        if not pc:
            continue
        try:
            df = clean_df(yf.download(pc["symbol"], period="5d",
                                      interval="15m", progress=False))
            if df is not None:
                m15_cache[pair_name] = df
        except Exception:
            pass

    for alert in alert_log:
        try:
            if alert.get("alert_type") != "zone":
                continue
            if alert.get("outcome") in ("win_tp1", "loss", "invalidated", "not_triggered"):
                continue
            if alert.get("invalidation_email_sent"):
                continue

            # Already marked as filled on a previous scan — skip permanently
            if alert.get("entry_filled"):
                continue

            pair = alert.get("pair")
            pair_conf = get_pair_conf(pair)
            if not pair_conf:
                continue

            current_price = current_prices.get(pair)
            if current_price is None:
                continue

            dp = pair_conf.get("decimal_places", 5)

            # ── Check if entry was filled AFTER the Trade Ready alert ─────
            try:
                entry_val = float(str(alert.get('entry', '0')).split('-')[0].strip() or 0)
                bias_val  = str(alert.get('bias', '')).upper()
                if entry_val > 0 and bias_val in ('LONG', 'SHORT'):
                    alert_time = datetime.strptime(alert['timestamp_utc'], "%Y-%m-%d %H:%M")
                    df_m15 = m15_cache.get(pair)
                    if df_m15 is not None:
                        for ts_c, row_c in df_m15.iterrows():
                            ts_n = ts_c.replace(tzinfo=None) if hasattr(ts_c, 'tzinfo') and ts_c.tzinfo else ts_c
                            if ts_n < alert_time:
                                continue
                            h_c = float(row_c['High']); l_c = float(row_c['Low'])
                            if l_c <= entry_val <= h_c:
                                alert["entry_filled"] = True
                                alert["entry_filled_at"] = ts_n.strftime("%Y-%m-%d %H:%M")
                                save_alert_log()
                                print(f"    {pair}: Entry filled at {entry_val} on {ts_n.strftime('%d %b %H:%M')} — trade live, skipping invalidation")
                                break
            except Exception:
                pass

            if alert.get("entry_filled"):
                continue

            # ── Entry NOT filled — run invalidation checks ────────────────
            reason = None

            # Check 1: Invalidation levels breached
            inv_above = alert.get("invalidate_above")
            inv_below = alert.get("invalidate_below")
            if inv_above and current_price > float(inv_above):
                reason = f"Price ({fmt_price(current_price, pair_conf)}) broke above invalidation level ({fmt_price(float(inv_above), pair_conf)})"
            if inv_below and current_price < float(inv_below):
                reason = f"Price ({fmt_price(current_price, pair_conf)}) broke below invalidation level ({fmt_price(float(inv_below), pair_conf)})"

            # Check 2: Price ran away (2x risk distance beyond entry without filling)
            if not reason:
                try:
                    entry = float(str(alert.get('entry', '0')).split('-')[0].strip() or 0)
                    sl    = float(alert.get('sl', 0) or 0)
                    bias  = str(alert.get('bias', '')).upper()
                    if entry > 0 and sl > 0 and bias in ('LONG', 'SHORT'):
                        risk_dist = abs(entry - sl)
                        if bias == "LONG" and current_price > entry + (2 * risk_dist):
                            reason = f"Price ran away above entry — now {fmt_price(current_price, pair_conf)}, entry was {fmt_price(entry, pair_conf)}. Retest window closed."
                        elif bias == "SHORT" and current_price < entry - (2 * risk_dist):
                            reason = f"Price ran away below entry — now {fmt_price(current_price, pair_conf)}, entry was {fmt_price(entry, pair_conf)}. Retest window closed."
                except Exception:
                    pass

            # Check 3: 48-hour timeout (entry never reached)
            if not reason:
                try:
                    alert_time = datetime.strptime(alert['timestamp_utc'], "%Y-%m-%d %H:%M")
                    age_h = (utc_now() - alert_time).total_seconds() / 3600
                    if age_h >= 48:
                        reason = "Setup expired — 48 hours without entry being reached."
                except Exception:
                    pass

            if reason:
                send_simple_email(
                    f"INVALIDATED | {pair} | Ref: {alert.get('ist_time','')} | {ist_now().strftime('%H:%M IST')}",
                    build_invalidation_email_html(alert, reason, pair_conf,
                                                 round(current_price, dp))
                )
                alert["outcome"] = "invalidated"
                alert["outcome_checked_at"] = utc_str()
                alert["invalidation_email_sent"] = True
                save_alert_log()
                fired += 1
                print(f"    INVALIDATED: {pair} — {reason}")

        except Exception as e:
            print(f"    Invalidation check error ({alert.get('pair', '?')}): {e}")

    return fired

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
ist_start = ist_now().strftime("%H:%M IST, %d %b %Y")
print(f"Alert engine started {ist_start} ({utc_str()} UTC)")

run_errors    = []
pairs_ok      = []
alerts_fired  = 0
current_prices = {}

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

        # Cache current price for Job 2 invalidation checks
        current_prices[name] = current_price

        # Compute ATR for this pair
        atr_value = get_atr(df1) if df1 is not None else None

        if not zones:
            log_scan(name, "no_zone_found", "No valid zones detected.")
            pairs_ok.append(name)
            continue

        zones_in_proximity = 0
        zone_alerted = False

        zones_by_dist = sorted(zones, key=lambda z: abs(current_price - z[0]))
        for zone_level, touches in zones_by_dist[:1]:
            if zone_alerted:
                break

            dist_pct = abs(current_price - zone_level) / zone_level * 100
            if dist_pct > prox:
                continue

            zones_in_proximity += 1
            zone_label = get_zone_label(zone_level, current_price)

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
                macro_news, df1, df2, fatigue, atr_value
            )

            data, error = call_gemini(prompt)

            if error:
                print(f"    {error}")
                log_scan(name, "error", error, zone_level)
                run_errors.append(f"{name}: {error}")
                continue

            # ── Python post-validation ────────────────────────────────────
            is_valid, reason, buffer_applied = validate_gemini_response(
                data, pair_conf, zone_label, current_price, atr_value)

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

            # ── Determine email type: TRADE READY or APPROACHING ──────────
            entry_ready = data.get("entry_ready_now", False)
            trigger_ok  = str(data.get("trigger_status", "")).lower() == "ready"
            is_trade_ready = entry_ready and trigger_ok

            if is_trade_ready:
                # ── TRADE READY ───────────────────────────────────────────
                if not should_send_trade_ready(name, zone_level):
                    log_scan(name, "blocked_active_trade",
                             "Active trade on this zone — blocked.", zone_level)
                    continue

                levels = {
                    'zone': zone_level, 'current': current_price,
                    'entry': data.get('entry', ''), 'sl': data.get('sl', 0),
                    'tp1': data.get('tp1', 0), 'tp2': data.get('tp2', 0)
                }
                chart1 = generate_chart(df1, f"{name} — H1", levels, data, pair_conf)
                chart2 = generate_chart(df2, f"{name} — M15", levels, data, pair_conf)

                html = build_trade_ready_email_html(
                    data, name, pair_conf, round(zone_level, dp),
                    zone_label, round(current_price, dp), chart1, chart2, fatigue
                )
                subject = f"[{score}/10] TRADE READY | {name} | {'SELL' if data.get('bias')=='SHORT' else 'BUY'} | {ist_now().strftime('%H:%M IST')}"
                send_email(subject, html, chart1, chart2)

                log_alert(name, round(zone_level, dp), zone_label,
                          round(current_price, dp), data, pair_conf,
                          "trade_ready", atr_value, buffer_applied, fatigue)
                log_scan(name, "trade_ready_sent",
                         f"Trade ready alert sent at score {score}/10.", zone_level)
                record_trade_ready(name, zone_level)

                system_status["last_trade_alert_utc"] = utc_str()
                alerts_fired += 1
                zone_alerted = True
                print(f"    ✓ TRADE READY: {name} [{score}/10]")

            else:
                # ── APPROACHING ───────────────────────────────────────────
                if not should_send_approaching(name, zone_level):
                    log_scan(name, "approaching_cooldown",
                             "Approaching already sent within 2h or active trade.", zone_level)
                    continue

                levels = {
                    'zone': zone_level, 'current': current_price,
                    'entry': data.get('entry', ''), 'sl': data.get('sl', 0),
                    'tp1': data.get('tp1', 0), 'tp2': data.get('tp2', 0)
                }
                chart1 = generate_chart(df1, f"{name} — H1", levels, data, pair_conf)
                chart2 = generate_chart(df2, f"{name} — M15", levels, data, pair_conf)

                html = build_approaching_email_html(
                    data, name, pair_conf, round(zone_level, dp),
                    zone_label, round(current_price, dp), chart1, chart2, fatigue
                )
                subject = f"APPROACHING | {name} | {zone_label} | {ist_now().strftime('%H:%M IST')}"
                send_email(subject, html, chart1, chart2)

                log_scan(name, "approaching_sent",
                         f"Approaching alert sent at score {score}/10.", zone_level)
                record_approaching(name, zone_level)

                zone_alerted = True
                print(f"    → APPROACHING: {name} [{score}/10]")

        if zones_in_proximity == 0:
            log_scan(name, "zone_outside_proximity",
                     "Zones detected but none near current price.")

        if not zone_alerted:
            pairs_ok.append(name)

    except Exception as e:
        print(f"    Error: {str(e)}")
        log_scan(name, "error", f"Pair-level error: {str(e)}")
        run_errors.append(f"{name}: {str(e)}")

# ── Job 2: Pure Python invalidation checks ────────────────────────────────────
print(f"\n  Job 2: Checking pending alerts for invalidation (Python only, 0 Gemini calls)...")
inv_fired = run_invalidation_checks(current_prices)
print(f"  Job 2 complete. {inv_fired} invalidation(s) sent.")

# ── System status emails ──────────────────────────────────────────────────────
if alerts_fired == 0 and inv_fired == 0 and not run_errors and should_send_ok():
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

total_actions = alerts_fired + inv_fired
print(f"\nAlert log: {len(alert_log)} entries | Scan log: {len(scan_log)} entries")
print(f"Scan complete. {alerts_fired} zone alert(s), {inv_fired} invalidation(s).")
