import yfinance as yf
import json, os, smtplib, requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np

# Load config
with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY    = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_PASS    = os.environ["GMAIL_APP_PASSWORD"]
ALERT_EMAIL   = os.environ["ALERT_EMAIL"]

# Cooldown system — prevents repeat emails for same zone
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

# Zone detection — finds key levels from price history
def detect_zones(symbol):
    df = yf.download(symbol, period="60d", interval="4h", progress=False)
    if df.empty:
        return [], None
    highs  = df["High"].values.flatten()
    lows   = df["Low"].values.flatten()
    closes = df["Close"].values.flatten()
    current_price = float(closes[-1])
    lb = config["zone_detection"]["swing_lookback"]

    swing_points = []
    for i in range(lb, len(highs) - lb):
        if highs[i] == max(highs[i-lb : i+lb+1]):
            swing_points.append(float(highs[i]))
        if lows[i] == min(lows[i-lb : i+lb+1]):
            swing_points.append(float(lows[i]))

    if not swing_points:
        return [], current_price

    swing_points = sorted(swing_points)
    clusters = [[swing_points[0]]]
    for lvl in swing_points[1:]:
        if (lvl - clusters[-1][-1]) / clusters[-1][-1] * 100 < 0.3:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    min_touches = config["zone_detection"]["min_zone_touches"]
    significant_zones = [
        (float(np.mean(c)), len(c))
        for c in clusters if len(c) >= min_touches
    ]
    return significant_zones, current_price

def get_zone_label(zone_level, current_price):
    return "Demand / Support" if zone_level < current_price else "Supply / Resistance"

# Fetch macro news headlines
def fetch_macro_news():
    try:
        url = "https://api.rss2json.com/v1/api.json?rss_url=https://www.fxstreet.com/rss/news&count=5"
        response = requests.get(url, timeout=10)
        items = response.json().get("items", [])
        return "\n".join([f"- {item['title']}" for item in items])
    except:
        return "Macro news feed unavailable right now."

# Build the Gemini prompt
def build_prompt(pair, zone_level, zone_label, current_price, macro_news):
    risk_dollar = config["account"]["balance"] * (config["account"]["risk_percent"] / 100)
    ist_time = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")
    utc_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    prompt = f"""
You are a professional SMC trader briefing a busy entrepreneur who trades part-time.
They understand SMC concepts but have limited time.
Your job: deliver a briefing they can read in 60 seconds and act on.

STRICT RULES:
- Maximum 200 words total across the entire briefing
- Zero filler words. Zero explaining what SMC terms mean.
- Use symbols: ✅ ❌ ⏳ 🎯 🛑 📐 — they save space and add scannability
- Bold only the numbers that matter: entry, SL, TP, lot size
- One emoji per section header only
- If a confluence IS confirmed, state it in one line.
- If a confluence is MISSING, write ❌ followed by one plain English sentence (max 10 words) explaining why waiting for it matters. No jargon.
- The BIAS line must be the very first thing they read
- End with one sentence max under MINDSET — sharp, specific, no motivation speech

ALERT:
- Pair: {pair}
- Zone: {zone_label}
- Key Level: {zone_level}
- Current Price: {current_price}
- Time: {utc_time} UTC | {ist_time} IST

Macro Headlines:
{macro_news}

Account: ${config["account"]["balance"]} | Risk: {config["account"]["risk_percent"]}% = ${risk_dollar:.0f}

---

Generate the briefing in EXACTLY this format and nothing else:

⚡ BIAS: [LONG / SHORT / WAIT] — [one line reason, max 12 words]

📍 ZONE: [zone_label] at [zone_level] | Price: [current_price] | [utc_time] UTC | [ist_time] IST

✅ CONFLUENCES
[List only what IS confirmed. One line each. Max 4 items.]

❌ MISSING
[For each missing confluence: one ❌ + one plain English sentence on why it matters to wait for it. Max 2 items.]

🎯 TRADE
Entry: [price range]
SL: [price] | TP1: [price] | TP2: [price]
R:R: [x:x to TP1] / [x:x to TP2]

📐 SIZE
Lot size: [X lots] (${risk_dollar:.0f} risk / [SL points] pts)

⏳ TRIGGER
[One sentence. Exactly what must happen before entry. Be specific.]

🛑 INVALID IF
[One sentence. The exact price action that kills this trade.]

🌍 MACRO
[Two lines max. What matters for this pair right now.]

🧠 MINDSET
[One sentence. The specific psychological trap to avoid on this exact setup.]

---
Total word count must not exceed 200 words.
"""
    return prompt
# Call Gemini API
def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, json=body, timeout=45)
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini API error: {str(e)}"

# Send email
def send_email(subject, body_text):
    recipients = config["account"].get("alert_emails", [ALERT_EMAIL])
    for recipient in recipients:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_ADDRESS
        msg["To"]      = recipient
        msg.attach(MIMEText(body_text, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_PASS)
            server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
        print(f"    Email sent to {recipient}")

# Main loop
print(f"Alert engine started at {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
macro_news = fetch_macro_news()
alerts_fired = 0

for pair_conf in config["pairs"]:
    symbol = pair_conf["symbol"]
    name   = pair_conf["name"]
    prox   = pair_conf["proximity_pct"]

    print(f"  Scanning {name}...")
    zones, current_price = detect_zones(symbol)

    if current_price is None:
        print(f"    Could not fetch data for {name}. Skipping.")
        continue

    for zone_level, touches in zones:
        distance_pct = abs(current_price - zone_level) / zone_level * 100
        if distance_pct <= prox:
            if is_on_cooldown(name, zone_level):
                print(f"    {name} @ {zone_level:.4f} - cooldown active, skipping.")
                continue

            zone_label = get_zone_label(zone_level, current_price)
            print(f"    ZONE HIT: {name} near {zone_label} at {zone_level:.4f}")

            prompt   = build_prompt(name, round(zone_level, 5), zone_label, round(current_price, 5), macro_news)
            briefing = call_gemini(prompt)
            subject  = f"Trade Alert: {name} | {zone_label} at {round(zone_level, 5)} | {datetime.utcnow().strftime('%H:%M')} UTC"
            send_email(subject, briefing)
            set_cooldown(name, zone_level)
            alerts_fired += 1
            print(f"    Alert sent for {name}")

print(f"Scan complete. {alerts_fired} alert(s) fired.")
