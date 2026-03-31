import yfinance as yf
import pandas as pd
import json, os, smtplib, requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY    = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_PASS    = os.environ["GMAIL_APP_PASSWORD"]
ALERT_EMAIL   = os.environ["ALERT_EMAIL"]

ALERT_LOG_FILE = "alert_log.json"

def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

alert_log = load_json(ALERT_LOG_FILE, [])

def save_alert_log():
    save_json(ALERT_LOG_FILE, alert_log)

def utc_now():
    return datetime.utcnow()

def utc_str():
    return utc_now().strftime("%Y-%m-%d %H:%M")

def clean_df(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def fetch_m15_data(symbol):
    return clean_df(yf.download(symbol, period="5d", interval="15m", progress=False))

def get_zone_label(zone_level, current_price):
    return "Demand / Support" if zone_level < current_price else "Supply / Resistance"

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

def format_candles(df, label, n=20):
    if df is None or df.empty:
        return f"{label}: No data\n"
    result = f"{label} (last {n} candles):\n"
    for i in range(max(0, len(df)-n), len(df)):
        try:
            ts  = df.index[i]
            tss = ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts, 'strftime') else str(ts)[:16]
            result += (f"{tss} O:{float(df['Open'].iloc[i]):.5f} "
                       f"H:{float(df['High'].iloc[i]):.5f} "
                       f"L:{float(df['Low'].iloc[i]):.5f} "
                       f"C:{float(df['Close'].iloc[i]):.5f}\n")
        except:
            pass
    return result

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

IMPORTANT FOR AUTOMATION:
- trigger_status must be one of: "not_ready", "ready", "invalidated"
- entry_ready_now must be true only if the trigger condition has ALREADY been confirmed on the latest candles
- Do not mark "ready" just because price touched entry. Retest entries need confirmation.
- invalidate_above / invalidate_below must be numeric whenever possible.
- Use null only if a boundary truly cannot be defined from the setup.
- Keep the human-readable "trigger" and "invalid_if" too.

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
  "entry_model": "retest or limit or market",
  "entry_ready_now": false,

  "trigger_status": "not_ready",
  "trigger_tf": "M15 or H1",
  "trigger_kind": "retest or choch or engulf or break_retest or none",
  "trigger_level": 0.0,
  "trigger": "exact M15 or H1 candle pattern required before entry",

  "invalidate_above": null,
  "invalidate_below": null,
  "invalid_if": "exact price action that cancels this trade",

  "sl": 0.0,
  "sl_note": "one sentence on SL placement logic and refinement after trigger",
  "tp1": 0.0,
  "tp2": 0.0,
  "rr_tp1": "x.x",
  "rr_tp2": "x.x",
  "lot_size": "x.x",
  "sl_pts": 0,

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

def send_email(subject, html_body):
    recipients = [ALERT_EMAIL]
    if "account" in config and "alert_emails" in config["account"]:
        recipients = config["account"]["alert_emails"]

    for recipient in recipients:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = GMAIL_ADDRESS
        msg["To"] = recipient
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_PASS)
            server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
        print(f"    Sent to {recipient}")

def send_simple_email(subject, html_body):
    send_email(subject, html_body)

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
    except:
        return None

def get_alert2_near_entry_pct(pair_name):
    custom = {
        "NAS100": 0.18,
        "XAUUSD": 0.12,
        "US30":   0.15
    }
    return custom.get(pair_name, 0.08)

def build_alert2_email_html(original_alert, refreshed_data, current_price):
    score = refreshed_data.get("confidence_score", 0)
    bias = refreshed_data.get("bias", original_alert.get("bias", "WAIT"))
    bias_color = "#e74c3c" if bias == "SHORT" else "#27ae60" if bias == "LONG" else "#f39c12"
    score_color = "#27ae60" if score >= 8 else "#f39c12" if score >= 6 else "#e74c3c"

    conf_items = "".join([
        f'<li style="margin-bottom:5px;padding:7px 10px;background:#f0fff4;border-radius:6px;font-size:13px;">&#10003; {c}</li>'
        for c in refreshed_data.get("confluences", [])
    ])
    miss_items = "".join([
        f'<li style="margin-bottom:5px;padding:7px 10px;background:#fff8f0;border-radius:6px;font-size:13px;">&#10007; <b>{m["item"]}</b> — <span style="color:#777;font-style:italic;">{m["reason"]}</span></li>'
        for m in refreshed_data.get("missing", [])
    ])

    ist_time = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")
    utc_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:16px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">
  <div style="background:#1a1a2e;padding:18px 24px;">
    <h2 style="color:white;margin:0;font-size:17px;">ENTRY ALERT: {original_alert['pair']}</h2>
    <p style="color:#8899bb;margin:5px 0 0;font-size:12px;">{utc_time} UTC | {ist_time} IST</p>
  </div>

  <div style="background:{bias_color};padding:12px 24px;">
    <p style="color:white;margin:0;font-size:15px;font-weight:bold;">{bias} — trigger ready now</p>
  </div>

  <div style="padding:10px 24px;background:#f8f9fa;border-bottom:1px solid #eee;">
    <span style="font-size:12px;color:#666;">Updated SMC Confidence: </span>
    <span style="font-size:18px;font-weight:bold;color:{score_color};">{score}/10</span>
    <span style="font-size:12px;color:#888;margin-left:8px;">— {refreshed_data.get("confidence_reason","")}</span>
  </div>

  <div style="padding:20px 24px;">
    <p style="background:#f4f4f8;padding:10px 14px;border-radius:8px;font-size:13px;color:#333;margin:0 0 16px;">
      Original zone: <b>{original_alert.get("zone_label","")}</b> at <b>{original_alert.get("zone_level","")}</b>
      &nbsp;|&nbsp; Now: <b>{current_price}</b>
    </p>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">TRADE LEVELS</h3>
    <table style="width:100%;font-size:13px;margin-bottom:16px;">
      <tr><td style="padding:5px 0;color:#555;width:140px;">Entry</td><td style="padding:5px 0;font-weight:bold;">{refreshed_data.get("entry","")}</td></tr>
      <tr><td style="padding:5px 0;color:#555;">Entry model</td><td style="padding:5px 0;font-weight:bold;">{refreshed_data.get("entry_model","")}</td></tr>
      <tr><td style="padding:5px 0;color:#555;">Trigger TF</td><td style="padding:5px 0;font-weight:bold;">{refreshed_data.get("trigger_tf","")}</td></tr>
      <tr><td style="padding:5px 0;color:#555;">Trigger kind</td><td style="padding:5px 0;font-weight:bold;">{refreshed_data.get("trigger_kind","")}</td></tr>
      <tr><td style="padding:5px 0;color:#e74c3c;">Stop Loss</td><td style="padding:5px 0;font-weight:bold;color:#e74c3c;">{refreshed_data.get("sl","")}</td></tr>
      <tr><td style="padding:5px 0;color:#27ae60;">TP1</td><td style="padding:5px 0;font-weight:bold;color:#27ae60;">{refreshed_data.get("tp1","")} (R:R {refreshed_data.get("rr_tp1","")})</td></tr>
      <tr><td style="padding:5px 0;color:#1e8449;">TP2</td><td style="padding:5px 0;font-weight:bold;color:#1e8449;">{refreshed_data.get("tp2","")} (R:R {refreshed_data.get("rr_tp2","")})</td></tr>
    </table>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">CONFLUENCES</h3>
    <ul style="list-style:none;padding:0;margin:0 0 20px;">{conf_items}</ul>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 8px;">MISSING</h3>
    <ul style="list-style:none;padding:0;margin:0 0 20px;">{miss_items}</ul>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 6px;">TRIGGER</h3>
    <p style="font-size:13px;color:#333;background:#fffbea;padding:10px 14px;border-radius:8px;border-left:4px solid #f39c12;margin:0 0 20px;">{refreshed_data.get("trigger","")}</p>

    <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 6px;">INVALID IF</h3>
    <p style="font-size:13px;color:#c0392b;background:#fef0f0;padding:10px 14px;border-radius:8px;border-left:4px solid #e74c3c;margin:0;">{refreshed_data.get("invalid_if","")}</p>
  </div>
</div>
</body>
</html>"""

def build_invalidation_email_html(alert, refreshed_data, current_price):
    ist_time = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M")
    utc_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#fff7f7;padding:16px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">
  <div style="background:#7f1d1d;padding:18px 24px;">
    <h2 style="color:white;margin:0;font-size:17px;">TRADE INVALIDATED: {alert['pair']}</h2>
    <p style="color:#fecaca;margin:5px 0 0;font-size:12px;">{utc_time} UTC | {ist_time} IST</p>
  </div>
  <div style="padding:20px 24px;">
    <p style="font-size:14px;color:#374151;">The earlier setup is no longer valid.</p>
    <table style="width:100%;font-size:13px;margin-top:12px;">
      <tr><td style="padding:5px 0;color:#555;width:140px;">Pair</td><td style="padding:5px 0;font-weight:bold;">{alert.get("pair","")}</td></tr>
      <tr><td style="padding:5px 0;color:#555;">Bias</td><td style="padding:5px 0;font-weight:bold;">{alert.get("bias","")}</td></tr>
      <tr><td style="padding:5px 0;color:#555;">Current price</td><td style="padding:5px 0;font-weight:bold;">{current_price}</td></tr>
      <tr><td style="padding:5px 0;color:#555;">Trigger status</td><td style="padding:5px 0;font-weight:bold;">{refreshed_data.get("trigger_status","")}</td></tr>
      <tr><td style="padding:5px 0;color:#555;">Invalid if</td><td style="padding:5px 0;">{refreshed_data.get("invalid_if","")}</td></tr>
    </table>
  </div>
</div>
</body>
</html>"""

def run_second_alert_checks(macro_news):
    fired = 0

    for alert in alert_log:
        try:
            if alert.get("alert_type") != "zone":
                continue
            if alert.get("outcome") in ("win_tp1", "loss", "invalidated"):
                continue
            if alert.get("entry_alert_sent"):
                continue
            if alert.get("invalidation_email_sent"):
                continue

            pair_conf = get_pair_conf(alert.get("pair"))
            if not pair_conf:
                continue

            symbol = pair_conf["symbol"]
            min_conf = pair_conf.get("min_confidence", 5)

            df1 = clean_df(yf.download(symbol, period="15d", interval="1h", progress=False))
            df2 = fetch_m15_data(symbol)
            if df1 is None or df2 is None:
                continue

            current_price = float(df1["Close"].iloc[-1])
            entry_mid = parse_entry_mid(alert.get("entry", ""))
            if entry_mid is None:
                continue

            near_entry_pct = get_alert2_near_entry_pct(alert["pair"])
            entry_dist_pct = abs(current_price - entry_mid) / entry_mid * 100
            if entry_dist_pct > near_entry_pct:
                continue

            zone_level = float(alert.get("zone_level", 0) or 0)
            zone_label = alert.get("zone_label", get_zone_label(zone_level, current_price))
            fatigue = count_zone_alerts(alert["pair"], zone_level) if zone_level > 0 else 0

            prompt = build_zone_prompt(
                alert["pair"],
                round(zone_level, 5),
                zone_label,
                round(current_price, 5),
                macro_news,
                df1,
                df2,
                min_conf,
                fatigue
            )

            refreshed_data, error = call_gemini(prompt)
            if error or not refreshed_data:
                continue

            trigger_status = str(refreshed_data.get("trigger_status", "not_ready")).lower()
            score = refreshed_data.get("confidence_score", 0)

            if trigger_status == "invalidated":
                send_simple_email(
                    f"INVALIDATED | {alert['pair']} | {datetime.utcnow().strftime('%H:%M')} UTC",
                    build_invalidation_email_html(alert, refreshed_data, round(current_price, 5))
                )
                alert["outcome"] = "invalidated"
                alert["outcome_checked_at"] = utc_str()
                alert["invalidation_email_sent"] = True
                save_alert_log()
                fired += 1
                continue

            if trigger_status != "ready":
                continue
            if not refreshed_data.get("entry_ready_now", False):
                continue
            if not refreshed_data.get("send_alert", False):
                continue
            if score < min_conf:
                continue

            subject = f"[{score}/10] ENTRY ALERT | {alert['pair']} | {datetime.utcnow().strftime('%H:%M')} UTC"
            html = build_alert2_email_html(alert, refreshed_data, round(current_price, 5))
            send_simple_email(subject, html)

            alert["entry_alert_sent"] = True
            alert["entry_alert_sent_at"] = utc_str()
            alert["entry_alert_price"] = round(current_price, 5)
            save_alert_log()
            fired += 1

        except Exception as e:
            print(f"    Alert 2 error ({alert.get('pair', 'unknown')}): {e}")

    return fired

# ── MAIN ──────────────────────────────────────────────────────────────────────
print(f"Alert2 engine started {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

pending_alerts = [
    a for a in alert_log
    if a.get("alert_type") == "zone"
    and a.get("outcome") not in ("win_tp1", "loss", "invalidated")
    and not a.get("entry_alert_sent")
    and not a.get("invalidation_email_sent")
]

if not pending_alerts:
    print("No pending Alert 1 setups. Exiting.")
    raise SystemExit(0)

fired = run_second_alert_checks("Macro refresh skipped in alert2 lightweight check.")
save_alert_log()
print(f"Alert2 complete. {fired} alert(s) fired.")
