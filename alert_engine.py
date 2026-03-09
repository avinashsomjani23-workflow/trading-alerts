import yfinance as yf
import json, os, smtplib, requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np

# ── Load config ───────────────────────────────────────────────
with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY    = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_PASS    = os.environ["GMAIL_APP_PASSWORD"]
ALERT_EMAIL   = os.environ["ALERT_EMAIL"]

# ── Cooldown system ───────────────────────────────────────────
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

# ── Market hours check (IST based) ───────────────────────────
def is_market_open():
    ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    ist_weekday = ist_now.weekday()
    ist_hour = ist_now.hour
    ist_minute = ist_now.minute
    if ist_weekday == 5:
        return False, f"Saturday IST — market closed."
    if ist_weekday == 6:
        return False, f"Sunday IST — market closed."
    if ist_weekday == 0 and (ist_hour < 2 or (ist_hour == 2 and ist_minute < 30)):
        return False, f"Monday before 2:30 AM IST — market not yet open."
    if ist_weekday == 4 and ist_hour >= 23 and ist_minute >= 30:
        return False, f"Friday after 11:30 PM IST — market closing."
    return True, f"Market open. IST: {ist_now.strftime('%A %H:%M')}"

# ── Zone detection + real candle data feed ───────────────────
def detect_zones_and_candles(symbol, min_touches):
    df_h4 = yf.download(symbol, period="60d", interval="4h", progress=False)
    df_h1 = yf.download(symbol, period="10d", interval="1h", progress=False)

    if df_h4.empty:
        return [], None, "", ""

    highs  = df_h4["High"].values.flatten()
    lows   = df_h4["Low"].values.flatten()
    closes = df_h4["Close"].values.flatten()
    current_price = float(closes[-1])
    lb = config["zone_detection"]["swing_lookback"]

    swing_points = []
    for i in range(lb, len(highs) - lb):
        if highs[i] == max(highs[i-lb:i+lb+1]):
            swing_points.append(float(highs[i]))
        if lows[i] == min(lows[i-lb:i+lb+1]):
            swing_points.append(float(lows[i]))

    if not swing_points:
        return [], current_price, "", ""

    swing_points = sorted(swing_points)
    clusters = [[swing_points[0]]]
    for lvl in swing_points[1:]:
        if (lvl - clusters[-1][-1]) / clusters[-1][-1] * 100 < 0.3:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    significant_zones = [
        (float(np.mean(c)), len(c))
        for c in clusters if len(c) >= min_touches
    ]

    # Last 20 H4 candles — real OHLC data for Gemini
    h4_candles = ""
    for i in range(max(0, len(df_h4)-20), len(df_h4)):
        try:
            h4_candles += (
                f"H4 | {df_h4.index[i].strftime('%Y-%m-%d %H:%M')} | "
                f"O:{float(df_h4['Open'].values.flatten()[i]):.5f} "
                f"H:{float(df_h4['High'].values.flatten()[i]):.5f} "
                f"L:{float(df_h4['Low'].values.flatten()[i]):.5f} "
                f"C:{float(df_h4['Close'].values.flatten()[i]):.5f}\n"
            )
        except:
            pass

    # Last 20 H1 candles — real OHLC data for Gemini
    h1_candles = ""
    if not df_h1.empty:
        for i in range(max(0, len(df_h1)-20), len(df_h1)):
            try:
                h1_candles += (
                    f"H1 | {df_h1.index[i].strftime('%Y-%m-%d %H:%M')} | "
                    f"O:{float(df_h1['Open'].values.flatten()[i]):.5f} "
                    f"H:{float(df_h1['High'].values.flatten()[i]):.5f} "
                    f"L:{float(df_h1['Low'].values.flatten()[i]):.5f} "
                    f"C:{float(df_h1['Close'].values.flatten()[i]):.5f}\n"
                )
            except:
                pass

    return significant_zones, current_price, h4_candles, h1_candles

def get_zone_label(zone_level, current_price):
    return "Demand / Support" if zone_level < current_price else "Supply / Resistance"

# ── Macro news ────────────────────────────────────────────────
def fetch_macro_news():
    try:
        url = "https://api.rss2json.com/v1/api.json?rss_url=https://www.fxstreet.com/rss/news&count=5"
        response = requests.get(url, timeout=10)
        items = response.json().get("items", [])
        return "\n".join([f"- {item['title']}" for item in items])
    except:
        return "Macro news feed unavailable right now."

# ── Build Gemini prompt with real candle data + SMC scorecard ─
def build_prompt(pair, zone_level, zone_label, current_price,
                 macro_news, h4_candles, h1_candles, min_confidence):
    risk_dollar = config["account"]["balance"] * (config["account"]["risk_percent"] / 100)
    ist_time = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")
    utc_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    prompt = f"""
You are a professional SMC (Smart Money Concepts) trader and analyst.
You are briefing a part-time trading entrepreneur. They understand SMC but have no time to waste.

---
ALERT DETAILS:
- Pair: {pair}
- Zone Type: {zone_label}
- Key Level: {zone_level}
- Current Price: {current_price}
- Time: {utc_time} UTC | {ist_time} IST
- Account: ${config["account"]["balance"]} | Risk: {config["account"]["risk_percent"]}% = ${risk_dollar:.0f}

---
REAL CANDLE DATA (use this to identify actual OBs, FVGs, structure, and accurate price levels):

{h4_candles}
{h1_candles}

---
MACRO HEADLINES:
{macro_news}

---
SMC CONFIDENCE SCORECARD — score this setup honestly out of 10:

STRUCTURE (max 3 points):
+1 if H4 or Daily BOS confirms trend direction
+1 if price is in correct Premium (for shorts) or Discount (for longs) zone
+1 if CHoCH is confirmed on H1 or M15 as entry signal

ZONE QUALITY (max 3 points):
+1 if a valid Order Block exists at or near the zone (identify from candle data above)
+1 if a Fair Value Gap overlaps or sits inside the OB (identify from candle data)
+1 if this zone is fresh — tested only once or twice before

LIQUIDITY (max 2 points):
+1 if liquidity (stop hunt above swing high or below swing low) has already been swept
+1 if entry would be in Discount zone for longs or Premium zone for shorts

RISK AND MACRO (max 2 points):
+1 if R:R is 2:1 or better based on actual candle structure
+1 if no high-impact news in next 2 hours (flag it either way — email still sends)

MINIMUM CONFIDENCE REQUIRED FOR THIS PAIR: {min_confidence}/10
If score is below {min_confidence}, set send_alert to false.
If score is {min_confidence} or above, set send_alert to true.

---
ACCURACY RULES FOR PRICE LEVELS:
- Identify Stop Loss from actual candle data: place BELOW the wick of the OB candle for longs, ABOVE for shorts
- Identify Entry from actual OB or FVG range visible in candle data
- Identify TP1 at the nearest swing high/low visible in candle data
- Identify TP2 at the next significant structure level visible in candle data
- Calculate R:R precisely from these levels
- Do NOT guess. If candle data does not support a level, say so.

---
Return ONLY a valid JSON object. No explanation. No markdown. No code blocks. Just raw JSON.

{{
  "send_alert": true,
  "confidence_score": 0,
  "confidence_reason": "one sentence explaining the score honestly",
  "news_flag": "none OR describe the upcoming news event and time",
  "bias": "SHORT or LONG or WAIT",
  "bias_reason": "max 12 words, plain English",
  "confluences": [
    "confirmed SMC item 1",
    "confirmed SMC item 2",
    "confirmed SMC item 3"
  ],
  "missing": [
    {{
      "item": "what SMC element is missing",
      "reason": "one plain English sentence — why does waiting for this matter"
    }}
  ],
  "entry": "exact price or range from candle data",
  "sl": 0.0,
  "tp1": 0.0,
  "tp2": 0.0,
  "rr_tp1": "x.x",
  "rr_tp2": "x.x",
  "lot_size": "x.x",
  "sl_pts": 0,
  "trigger": "exact price action required on M15 or H1 before entry",
  "invalid_if": "exact price action that kills this trade",
  "macro_line1": "most relevant macro driver for {pair} right now",
  "macro_line2": "key event or risk to watch this week for {pair}",
  "mindset": "one sharp sentence — specific psychological trap on this exact setup"
}}
"""
    return prompt

# ── Call Gemini API ───────────────────────────────────────────
def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, json=body, timeout=60)
        result = response.json()
        if "candidates" not in result:
            return None, f"Gemini error: {result}"
        raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
        data = json.loads(raw)
        return data, None
    except Exception as e:
        return None, f"Gemini API error: {str(e)}"

# ── Build HTML email ──────────────────────────────────────────
def build_html(data, pair, zone_level, zone_label, current_price):
    ist_time = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")
    utc_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    bias_color = "#e74c3c" if data["bias"] == "SHORT" else "#27ae60" if data["bias"] == "LONG" else "#f39c12"
    score = data.get("confidence_score", 0)
    score_color = "#27ae60" if score >= 8 else "#f39c12" if score >= 6 else "#e74c3c"

    news_flag = data.get("news_flag", "none")
    news_banner = ""
    if news_flag and news_flag.lower() != "none":
        news_banner = f"""
        <div style="background:#fff3cd;padding:10px 24px;border-left:4px solid #f39c12;font-size:12px;color:#856404;">
          ⚠️ <b>NEWS ALERT:</b> {news_flag}
        </div>"""

    confluences_html = "".join([
        f"<li style='margin-bottom:6px;'>✅ {c}</li>"
        for c in data.get("confluences", [])
    ])
    missing_html = "".join([
        f"<li style='margin-bottom:6px;'>❌ <b>{m['item']}</b> — <span style='color:#666;'>{m['reason']}</span></li>"
        for m in data.get("missing", [])
    ])

    # Price ladder — proportional bars
    try:
        entry_price = float(str(data["entry"]).split("-")[0].strip())
        levels = {
            "TP2": float(data["tp2"]),
            "TP1": float(data["tp1"]),
            "Entry": entry_price,
            "Current": float(current_price),
            "Zone": float(zone_level),
            "SL": float(data["sl"]),
        }
        level_colors = {
            "SL": "#e74c3c", "Zone": "#9b59b6", "Current": "#3498db",
            "Entry": "#e67e22", "TP1": "#27ae60", "TP2": "#1e8449"
        }
        all_prices = list(levels.values())
        price_min = min(all_prices)
        price_max = max(all_prices)
        price_range = price_max - price_min if price_max != price_min else 1

        sorted_levels = sorted(levels.items(), key=lambda x: x[1], reverse=True)
        ladder_rows = ""
        for label, price in sorted_levels:
            color = level_colors.get(label, "#888")
            bar_pct = int(((price - price_min) / price_range) * 75) + 15
            ladder_rows += f"""
            <tr>
              <td style="padding:5px 10px;color:{color};font-weight:bold;font-size:12px;width:65px;">{label}</td>
              <td style="padding:5px 6px;">
                <div style="background:{color};height:10px;border-radius:4px;width:{bar_pct}%;"></div>
              </td>
              <td style="padding:5px 10px;color:{color};font-weight:bold;font-size:12px;text-align:right;white-space:nowrap;">{price:,.5f}</td>
            </tr>"""
        price_map_section = f"""
        <h3 style="color:#1a1a2e;font-size:13px;margin:20px 0 8px;">📊 PRICE MAP</h3>
        <table style="width:100%;border-collapse:collapse;background:#f8f9fa;border-radius:8px;margin-bottom:20px;overflow:hidden;">
          {ladder_rows}
        </table>"""
    except:
        price_map_section = "<p style='color:#888;font-size:12px;'>Price map unavailable for this alert.</p>"

    html = f"""
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:20px;margin:0;">
<div style="max-width:600px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">

  <div style="background:#1a1a2e;padding:20px 24px;">
    <h2 style="color:white;margin:0;font-size:17px;letter-spacing:0.5px;">🔔 {pair} — {zone_label}</h2>
    <p style="color:#8899bb;margin:5px 0 0;font-size:12px;">{utc_time} UTC &nbsp;|&nbsp; {ist_time} IST</p>
  </div>

  <div style="background:{bias_color};padding:13px 24px;">
    <p style="color:white;margin:0;font-size:15px;font-weight:bold;">⚡ {data["bias"]} — {data["bias_reason"]}</p>
  </div>

  <div style="padding:12px 24px;background:#f8f9fa;border-bottom:1px solid #eee;display:flex;align-items:center;">
    <span style="font-size:12px;color:#666;">Confidence: </span>
    <span style="font-size:18px;font-weight:bold;color:{score_color};margin:0 8px;">{score}/10</span>
    <span style="font-size:12px;color:#888;">— {data.get("confidence_reason","")}</span>
  </div>

  {news_banner}

  <div style="padding:20px 24px;">

    <p style="background:#f8f9fa;padding:10px 14px;border-radius:8px;font-size:13px;color:#333;margin:0 0 16px 0;">
      📍 Zone: <b>{zone_label}</b> at <b>{zone_level}</b> &nbsp;|&nbsp; Current Price: <b>{current_price}</b>
    </p>

    {price_map_section}

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">🎯 TRADE</h3>
    <table style="width:100%;font-size:13px;margin-bottom:20px;">
      <tr>
        <td style="padding:4px 0;color:#555;width:110px;">Entry</td>
        <td style="padding:4px 0;font-weight:bold;">{data["entry"]}</td>
      </tr>
      <tr>
        <td style="padding:4px 0;color:#e74c3c;">Stop Loss</td>
        <td style="padding:4px 0;font-weight:bold;color:#e74c3c;">{data["sl"]}</td>
      </tr>
      <tr>
        <td style="padding:4px 0;color:#27ae60;">TP1</td>
        <td style="padding:4px 0;font-weight:bold;color:#27ae60;">{data["tp1"]} &nbsp;<span style="font-weight:normal;color:#888;">(R:R {data["rr_tp1"]})</span></td>
      </tr>
      <tr>
        <td style="padding:4px 0;color:#1e8449;">TP2</td>
        <td style="padding:4px 0;font-weight:bold;color:#1e8449;">{data["tp2"]} &nbsp;<span style="font-weight:normal;color:#888;">(R:R {data["rr_tp2"]})</span></td>
      </tr>
      <tr>
        <td style="padding:4px 0;color:#555;">Lot Size</td>
        <td style="padding:4px 0;font-weight:bold;">{data["lot_size"]} lots &nbsp;<span style="font-weight:normal;color:#888;">({data["sl_pts"]} pts risk)</span></td>
      </tr>
    </table>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">✅ CONFLUENCES</h3>
    <ul style="font-size:13px;color:#333;padding-left:20px;margin:0 0 20px;">{confluences_html}</ul>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">❌ MISSING</h3>
    <ul style="font-size:13px;color:#333;padding-left:20px;margin:0 0 20px;">{missing_html}</ul>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 6px;">⏳ TRIGGER</h3>
    <p style="font-size:13px;color:#333;background:#fff8e1;padding:10px 14px;border-radius:8px;border-left:4px solid #f39c12;margin:0 0 20px;">{data["trigger"]}</p>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 6px;">🛑 INVALID IF</h3>
    <p style="font-size:13px;color:#c0392b;background:#fdf0f0;padding:10px 14px;border-radius:8px;border-left:4px solid #e74c3c;margin:0 0 20px;">{data["invalid_if"]}</p>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 6px;">🌍 MACRO</h3>
    <p style="font-size:13px;color:#333;margin:0 0 20px;">{data["macro_line1"]}<br><br>{data["macro_line2"]}</p>

    <div style="background:#1a1a2e;padding:16px 18px;border-radius:10px;">
      <p style="color:#8899bb;font-size:11px;margin:0 0 5px;text-transform:uppercase;letter-spacing:1px;">🧠 Mindset</p>
      <p style="color:white;font-size:13px;margin:0;font-style:italic;line-height:1.5;">{data["mindset"]}</p>
    </div>

  </div>
</div>
</body>
</html>"""
    return html

# ── Send email ────────────────────────────────────────────────
def send_email(subject, data, pair, zone_level, zone_label, current_price):
    html_body = build_html(data, pair, zone_level, zone_label, current_price)
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

# ── Main loop ─────────────────────────────────────────────────
print(f"Alert engine started at {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

market_open, market_status = is_market_open()
print(f"  Market status: {market_status}")
if not market_open:
    print("  No alerts will be sent. Exiting.")
    exit(0)

macro_news = fetch_macro_news()
alerts_fired = 0

for pair_conf in config["pairs"]:
    symbol      = pair_conf["symbol"]
    name        = pair_conf["name"]
    prox        = pair_conf["proximity_pct"]
    min_touches = pair_conf.get("min_touches", 2)
    min_conf    = pair_conf.get("min_confidence", 5)

    print(f"  Scanning {name}...")
    zones, current_price, h4_candles, h1_candles = detect_zones_and_candles(symbol, min_touches)

    if current_price is None:
        print(f"    Could not fetch data for {name}. Skipping.")
        continue

    for zone_level, touches in zones:
        distance_pct = abs(current_price - zone_level) / zone_level * 100
        if distance_pct <= prox:
            if is_on_cooldown(name, zone_level):
                print(f"    {name} @ {zone_level:.5f} — cooldown active, skipping.")
                continue

            zone_label = get_zone_label(zone_level, current_price)
            print(f"    ZONE HIT: {name} near {zone_label} at {zone_level:.5f} (dist: {distance_pct:.2f}%)")

            prompt = build_prompt(
                name, round(zone_level, 5), zone_label,
                round(current_price, 5), macro_news,
                h4_candles, h1_candles, min_conf
            )
            data, error = call_gemini(prompt)

            if error:
                print(f"    Gemini failed: {error}")
                continue

            score = data.get("confidence_score", 0)

            if not data.get("send_alert", False):
                print(f"    {name} skipped — score {score}/10 below threshold {min_conf}. {data.get('confidence_reason','')}")
                continue

            subject = f"[{score}/10] Trade Alert: {name} | {zone_label} | {round(zone_level, 5)} | {datetime.utcnow().strftime('%H:%M')} UTC"
            send_email(subject, data, name, round(zone_level, 5), zone_label, round(current_price, 5))
            set_cooldown(name, zone_level)
            alerts_fired += 1
            print(f"    ✅ Alert sent for {name} — score {score}/10")

print(f"Scan complete. {alerts_fired} alert(s) fired.")
