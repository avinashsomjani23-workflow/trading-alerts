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
    s = config["email_sections"]
    risk_dollar = config["account"]["balance"] * (config["account"]["risk_percent"] / 100)

    prompt = f"""
You are a highly technical professional trader specialising in Smart Money Concepts (SMC) and ICT methodology.
A price alert has triggered. Generate a complete, precise trading briefing.
Write as if briefing a professional prop trader at {config["account"]["firm"]}.
No filler. No vague language. Every section must contain specific, actionable insight.

ALERT DETAILS:
- Pair: {pair}
- Zone Type: {zone_label}
- Key Level: {zone_level}
- Current Price: {current_price}
- Trading Style: Intraday (M15-H1) and Swing (H4-Daily)
- Time (UTC): {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}

Recent Macro Headlines:
{macro_news}

---
ALERT: {pair} has touched a {zone_label} zone at {zone_level}
---
"""

    if s.get("micro_view"):
        prompt += """
MICRO VIEW

Multi-Timeframe Structure:
- Daily: [Trend direction, major structure, swing highs/lows, where price sits in the range]
- H4: [Intermediate structure, order blocks, fair value gaps, key levels nearby]
- H1: [Intraday structure, entry zones, liquidity pools, current momentum]
- M15: [Confirmation signals needed, entry refinement, current candle behaviour]
"""

    if s.get("confluence_checklist"):
        prompt += """
Confluence Analysis at This Zone:
- Order Block (OB): [Valid OB at this level? Bullish or bearish? Mitigated?]
- Fair Value Gap (FVG): [Any FVG within or near this zone? Direction?]
- RSI: [Expected condition - oversold, overbought, divergence, midline rejection]
- Structure Signal: [What BOS or CHoCH to watch for as entry confirmation]
- Liquidity: [Where are stops resting? Have equal highs/lows been swept?]

Confluence Checklist:
[ ] Price inside or reacting from a valid OB
[ ] FVG present and aligning with zone direction
[ ] RSI divergence or extreme reading present
[ ] BOS or CHoCH confirmation on M15 or H1
[ ] Session aligns - London/NY for intraday, H4/D close for swing
[ ] Liquidity swept before entry
[ ] No high-impact news in next 30 minutes
"""

    if s.get("macro_view"):
        prompt += f"""
---
MACRO VIEW

What is driving {pair} right now:
[Central bank stance, risk sentiment, recent or upcoming economic data, geopolitical factors. Be specific.]

Sentiment Bias: [Bullish / Bearish / Neutral - one sharp reason]
Key Events This Week: [2-3 specific upcoming catalysts for this pair]
"""

    if s.get("trade_rationale"):
        prompt += f"""
---
TRADE RATIONALE

Directional Bias: [Long / Short / Wait for confirmation - and the specific reason]

Why this zone matters:
[2-3 sentences on the structural significance of this exact level]

Ideal Setup:
- Entry Trigger: [Exact confirmation required]
- Entry Zone: [Specific price range]
- Stop Loss: [Level and reason]
- TP1: [First target]
- TP2: [Swing target]
- Estimated R:R: [Based on the above levels]

Invalidation: [One sentence - what makes this setup void]
"""

    if s.get("position_size"):
        prompt += f"""
---
POSITION SIZE

Account Balance: ${config["account"]["balance"]}
Risk Per Trade: {config["account"]["risk_percent"]}% = ${risk_dollar:.2f}
Stop Loss Distance: [Estimate in pips or points]
Recommended Lot Size: [Lot Size = ${risk_dollar:.2f} divided by (SL distance x pip value)]
Pip Value Note: [State pip value for {pair}]
"""

    if s.get("pretrade_checklist"):
        prompt += f"""
---
PRE-TRADE CHECKLIST - {config["account"]["firm"]} Rules

[ ] NOT within 1% of daily drawdown limit
[ ] No high-impact news in next 2 minutes - confirm on Forex Factory
[ ] This is a pre-identified zone - not chasing price
[ ] Invalidation level set before entry
[ ] Lot size within firm maximum
[ ] Within allowed trading hours for this instrument
[ ] Directional bias aligns with at least H4 structure
"""

    if s.get("mindset_note"):
        prompt += f"""
---
MINDSET NOTE

[One specific psychological reminder for THIS setup on {pair}. 
Tailored to the zone type and likely emotional trap. Sharp and relevant, not generic.]
"""

    prompt += "\n---\nEnd of briefing. Be precise and specific throughout."
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
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = ALERT_EMAIL
    msg.attach(MIMEText(body_text, "plain"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_PASS)
        server.sendmail(GMAIL_ADDRESS, ALERT_EMAIL, msg.as_string())

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
