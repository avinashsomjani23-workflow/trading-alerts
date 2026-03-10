import yfinance as yf
import pandas as pd
import json, os, smtplib, requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY    = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_PASS    = os.environ["GMAIL_APP_PASSWORD"]
ALERT_EMAIL   = os.environ["ALERT_EMAIL"]

ALERT_LOG_FILE = "alert_log.json"
try:
    with open(ALERT_LOG_FILE) as f:
        alert_log = json.load(f)
    print(f"  Loaded {len(alert_log)} total log entries")
except:
    alert_log = []
    print("  No alert log found")

def clean_df(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def check_outcome(alert):
    pair   = alert['pair']
    symbol = next((p['symbol'] for p in config['pairs'] if p['name'] == pair), None)
    if not symbol:
        return "pending", None
    try:
        alert_time = datetime.strptime(alert['timestamp_utc'], "%Y-%m-%d %H:%M")
        bias  = alert.get('bias', 'LONG')
        sl    = float(alert.get('sl', 0) or 0)
        tp1   = float(alert.get('tp1', 0) or 0)
        if sl <= 0 or tp1 <= 0:
            return "pending", None

        df = clean_df(yf.download(
            symbol,
            start=(alert_time - timedelta(hours=1)).strftime('%Y-%m-%d'),
            interval="1h", progress=False
        ))
        if df is None or df.empty:
            return "pending", None

        for ts, row in df.iterrows():
            try:
                ts_naive = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
                if ts_naive < alert_time:
                    continue
                h = float(row['High'])
                l = float(row['Low'])
            except:
                continue
            if bias == "LONG":
                if l <= sl:  return "loss", sl
                if h >= tp1: return "win_tp1", tp1
            elif bias == "SHORT":
                if h >= sl:  return "loss", sl
                if l <= tp1: return "win_tp1", tp1
        return "pending", None
    except Exception as e:
        print(f"  Outcome error {pair}: {e}")
        return "pending", None

def update_outcomes():
    updated = 0
    for alert in alert_log:
        if alert.get('outcome') == 'pending':
            try:
                alert_time = datetime.strptime(alert['timestamp_utc'], "%Y-%m-%d %H:%M")
                age_hours  = (datetime.utcnow() - alert_time).total_seconds() / 3600
                if 4 <= age_hours <= 336:
                    outcome, outcome_price = check_outcome(alert)
                    if outcome != 'pending':
                        alert['outcome']           = outcome
                        alert['outcome_price']      = outcome_price
                        alert['outcome_checked_at'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
                        updated += 1
            except:
                pass
    with open(ALERT_LOG_FILE, "w") as f:
        json.dump(alert_log, f, indent=2)
    print(f"  Updated {updated} outcomes")

def get_weekly_alerts():
    cutoff = datetime.utcnow() - timedelta(days=7)
    weekly = [
        a for a in alert_log
        if datetime.strptime(a['timestamp_utc'], "%Y-%m-%d %H:%M") >= cutoff
    ]
    print(f"  {len(weekly)} alerts in last 7 days")
    return weekly

def get_session(utc_hour):
    if 2 <= utc_hour < 8:   return "Asian"
    elif 8 <= utc_hour < 13: return "London"
    elif 13 <= utc_hour < 21: return "New York"
    else:                     return "Off-Hours"

def build_weekly_analysis(weekly_alerts, wins, losses, pending, win_rate, pair_stats):
    high_conf = [a for a in weekly_alerts if a.get('confidence_score', 0) >= 8]
    low_conf  = [a for a in weekly_alerts if a.get('confidence_score', 0) < 8]

    # Session breakdown
    session_stats = {}
    for a in weekly_alerts:
        try:
            utc_hour = datetime.strptime(a['timestamp_utc'], "%Y-%m-%d %H:%M").hour
            session  = get_session(utc_hour)
            if session not in session_stats:
                session_stats[session] = {'alerts':0,'wins':0,'losses':0,'pending':0}
            session_stats[session]['alerts'] += 1
            o = a.get('outcome', 'pending')
            if o == 'win_tp1':  session_stats[session]['wins']    += 1
            elif o == 'loss':   session_stats[session]['losses']  += 1
            else:               session_stats[session]['pending'] += 1
        except:
            pass
    for s in session_stats.values():
        w = s.get('wins', 0)
        l = s.get('losses', 0)
        s['win_rate'] = round((w/(w+l)*100), 1) if (w+l) > 0 else None

    # Timing clusters (IST hours with 2+ losses)
    hour_buckets = {}
    for a in weekly_alerts:
        try:
            utc_dt   = datetime.strptime(a['timestamp_utc'], "%Y-%m-%d %H:%M")
            ist_hour = (utc_dt + timedelta(hours=5, minutes=30)).hour
            bucket   = f"{ist_hour:02d}:00 IST"
            if bucket not in hour_buckets:
                hour_buckets[bucket] = {'wins':0,'losses':0}
            o = a.get('outcome','pending')
            if o == 'win_tp1': hour_buckets[bucket]['wins']   += 1
            elif o == 'loss':  hour_buckets[bucket]['losses'] += 1
        except:
            pass
    loss_clusters = [
        f"{h} ({v['losses']} losses)"
        for h, v in sorted(hour_buckets.items()) if v['losses'] >= 2
    ]

    prompt = f"""
You are a professional SMC trading analyst reviewing one week of automated alerts.

WEEKLY SUMMARY:
- Total alerts: {len(weekly_alerts)}
- Wins (hit TP1): {wins} | Losses (hit SL): {losses} | Pending: {pending}
- Win rate: {win_rate:.1f}%
- High confidence (>=8/10): {len(high_conf)} alerts, {sum(1 for a in high_conf if a.get('outcome')=='win_tp1')} wins, {sum(1 for a in high_conf if a.get('outcome')=='loss')} losses
- Low confidence (<8/10): {len(low_conf)} alerts, {sum(1 for a in low_conf if a.get('outcome')=='win_tp1')} wins, {sum(1 for a in low_conf if a.get('outcome')=='loss')} losses

SESSION BREAKDOWN:
{json.dumps(session_stats, indent=2)}

LOSS TIME CLUSTERS (IST hours with 2+ losses):
{loss_clusters if loss_clusters else "None identified this week"}

PAIR PERFORMANCE:
{json.dumps(pair_stats, indent=2)}

FULL ALERT DATA:
{json.dumps([{{
  "pair": a['pair'],
  "time": a['timestamp_utc'],
  "ist_time": a.get('ist_time',''),
  "bias": a.get('bias'),
  "entry": a.get('entry'),
  "sl": a.get('sl'),
  "tp1": a.get('tp1'),
  "confidence": a.get('confidence_score'),
  "confluences": a.get('confluences',[]),
  "outcome": a.get('outcome','pending')
}} for a in weekly_alerts], indent=2)}

Return ONLY raw JSON, no markdown:
{{
  "overall_grade": "A",
  "grade_comment": "one sentence on overall performance this week",
  "best_pair": "pair name",
  "worst_pair": "pair name",
  "pattern_insight": "which confluence combination appeared most in winning trades",
  "confidence_calibration": "one sentence — are higher scored alerts winning more than lower ones",
  "session_summary": "which session had best win rate and which had worst, with numbers",
  "session_recommendation": "one actionable sentence on sessions",
  "timing_observation": "specific IST time windows where losses cluster this week",
  "timing_recommendation": "one actionable sentence on time filtering",
  "zone_fatigue_flags": ["pair at level: description — if any zone alerted 3+ times weakly, else empty array"],
  "streak_summary": "notable winning or losing streaks by pair this week",
  "drawdown_flag": "none or describe if 3 or more consecutive losses occurred",
  "improvement_suggestion": "one specific actionable system change based purely on this week data",
  "scorecard_rows": [
    {{
      "pair": "pair",
      "time": "Day HH:MM IST",
      "bias": "LONG or SHORT",
      "entry": "price",
      "sl": "price",
      "tp1": "price",
      "confidence": 0,
      "session": "London or NY or Asian or Off-Hours",
      "outcome": "Win or Loss or Pending",
      "comment": "one sharp observation on this specific trade"
    }}
  ]
}}"""

    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, json=body, timeout=60)
        result = r.json()
        if "candidates" not in result:
            print(f"  Gemini error: {result}")
            return None
        raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
    except Exception as e:
        print(f"  Gemini error: {e}")
        return None

def insight_card(emoji, title, color, content):
    return f'<div style="background:#f8f9fa;padding:13px 15px;border-radius:10px;margin-bottom:12px;border-left:4px solid {color};"><p style="font-size:10px;color:{color};margin:0 0 4px;text-transform:uppercase;font-weight:bold;letter-spacing:0.5px;">{emoji} {title}</p><p style="font-size:13px;color:#333;margin:0;line-height:1.5;">{content}</p></div>'

def build_weekly_html(data, weekly_alerts, wins, losses, pending, win_rate):
    ist_now     = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%A, %d %b %Y")
    grade       = data.get('overall_grade', 'B')
    grade_color = "#27ae60" if grade=="A" else "#f39c12" if grade in ["B","C"] else "#e74c3c"

    scorecard_rows = ""
    for row in data.get("scorecard_rows", []):
        outcome       = row.get('outcome','Pending')
        bg            = '#f0fff4' if 'Win' in outcome else '#fff0f0' if 'Loss' in outcome else '#fffbea'
        outcome_emoji = '✅' if 'Win' in outcome else '❌' if 'Loss' in outcome else '⏳'
        conf          = row.get('confidence', 0)
        cc            = "#27ae60" if conf>=8 else "#f39c12" if conf>=6 else "#e74c3c"
        scorecard_rows += f"""
        <tr style="background:{bg};border-bottom:1px solid #eee;">
          <td style="padding:7px 10px;font-weight:bold;font-size:12px;">{row.get('pair','')}</td>
          <td style="padding:7px 10px;font-size:11px;color:#666;">{row.get('time','')}</td>
          <td style="padding:7px 10px;font-size:12px;font-weight:bold;">{row.get('bias','')}</td>
          <td style="padding:7px 10px;font-size:11px;">{row.get('entry','')}</td>
          <td style="padding:7px 10px;font-size:11px;color:#e74c3c;">{row.get('sl','')}</td>
          <td style="padding:7px 10px;font-size:11px;color:#27ae60;">{row.get('tp1','')}</td>
          <td style="padding:7px 10px;font-weight:bold;color:{cc};">{conf}/10</td>
          <td style="padding:7px 10px;font-size:11px;color:#666;">{row.get('session','')}</td>
          <td style="padding:7px 10px;font-size:14px;">{outcome_emoji}</td>
          <td style="padding:7px 10px;font-size:11px;color:#888;font-style:italic;">{row.get('comment','')}</td>
        </tr>"""

    zone_flags = data.get('zone_fatigue_flags', [])
    zone_html  = "".join([
        f"<li style='font-size:12px;color:#e67e22;margin-bottom:4px;'>⚠️ {z}</li>"
        for z in zone_flags
    ]) if zone_flags else "<li style='font-size:12px;color:#aaa;'>No fatigued zones this week.</li>"

    drawdown      = data.get('drawdown_flag', 'none')
    drawdown_html = f'<div style="background:#fef0f0;padding:12px 16px;border-radius:8px;border-left:4px solid #e74c3c;margin-bottom:16px;"><p style="margin:0;font-size:13px;color:#c0392b;"><b>⚠️ Drawdown Alert:</b> {drawdown}</p></div>' if drawdown and drawdown.lower() != 'none' else ""

    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:16px;margin:0;">
<div style="max-width:720px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">

<div style="background:#1a1a2e;padding:20px 24px;">
  <h2 style="color:white;margin:0;font-size:18px;">📊 Weekly Trading Review</h2>
  <p style="color:#8899bb;margin:5px 0 0;font-size:12px;">{ist_now}</p>
</div>

<div style="display:flex;background:#f8f9fa;border-bottom:1px solid #eee;">
  <div style="flex:1;padding:14px 16px;text-align:center;border-right:1px solid #eee;">
    <p style="margin:0;font-size:10px;color:#888;text-transform:uppercase;">Grade</p>
    <p style="margin:4px 0 0;font-size:26px;font-weight:bold;color:{grade_color};">{grade}</p>
    <p style="margin:2px 0 0;font-size:10px;color:#aaa;">{data.get('grade_comment','')[:50]}</p>
  </div>
  <div style="flex:1;padding:14px 16px;text-align:center;border-right:1px solid #eee;">
    <p style="margin:0;font-size:10px;color:#888;text-transform:uppercase;">Win Rate</p>
    <p style="margin:4px 0 0;font-size:24px;font-weight:bold;color:#1a1a2e;">{win_rate:.0f}%</p>
  </div>
  <div style="flex:1;padding:14px 16px;text-align:center;border-right:1px solid #eee;">
    <p style="margin:0;font-size:10px;color:#888;text-transform:uppercase;">Alerts</p>
    <p style="margin:4px 0 0;font-size:24px;font-weight:bold;color:#1a1a2e;">{len(weekly_alerts)}</p>
  </div>
  <div style="flex:1;padding:14px 16px;text-align:center;">
    <p style="margin:0;font-size:10px;color:#888;text-transform:uppercase;">W / L / P</p>
    <p style="margin:4px 0 0;font-size:18px;font-weight:bold;">
      <span style="color:#27ae60;">{wins}</span>&thinsp;/&thinsp;<span style="color:#e74c3c;">{losses}</span>&thinsp;/&thinsp;<span style="color:#999;">{pending}</span>
    </p>
  </div>
</div>

<div style="display:flex;background:#f8f9fa;border-bottom:1px solid #eee;">
  <div style="flex:1;padding:12px 20px;text-align:center;border-right:1px solid #eee;">
    <p style="margin:0;font-size:10px;color:#888;">🏆 Best Pair</p>
    <p style="margin:4px 0 0;font-size:16px;font-weight:bold;color:#27ae60;">{data.get('best_pair','—')}</p>
  </div>
  <div style="flex:1;padding:12px 20px;text-align:center;">
    <p style="margin:0;font-size:10px;color:#888;">⚠️ Weakest Pair</p>
    <p style="margin:4px 0 0;font-size:16px;font-weight:bold;color:#e74c3c;">{data.get('worst_pair','—')}</p>
  </div>
</div>

<div style="padding:20px 24px;">
  {drawdown_html}

  <h3 style="color:#1a1a2e;font-size:13px;margin:0 0 10px;">📋 WEEKLY SCORECARD</h3>
  <div style="overflow-x:auto;margin-bottom:24px;">
  <table style="width:100%;border-collapse:collapse;font-size:12px;min-width:600px;">
    <thead><tr style="background:#1a1a2e;color:white;">
      <th style="padding:8px 10px;text-align:left;">Pair</th>
      <th style="padding:8px 10px;text-align:left;">Time IST</th>
      <th style="padding:8px 10px;text-align:left;">Dir</th>
      <th style="padding:8px 10px;text-align:left;">Entry</th>
      <th style="padding:8px 10px;text-align:left;">SL</th>
      <th style="padding:8px 10px;text-align:left;">TP1</th>
      <th style="padding:8px 10px;text-align:left;">Score</th>
      <th style="padding:8px 10px;text-align:left;">Session</th>
      <th style="padding:8px 10px;text-align:left;">Result</th>
      <th style="padding:8px 10px;text-align:left;">Note</th>
    </tr></thead>
    <tbody>{scorecard_rows}</tbody>
  </table>
  </div>

  {insight_card("🧠","Pattern Intelligence","#3498db", data.get('pattern_insight','—'))}
  {insight_card("📊","Confidence Calibration","#27ae60", data.get('confidence_calibration','—'))}
  {insight_card("🕐","Session Win Rate","#3498db", data.get('session_summary','—'))}
  {insight_card("📌","Session Recommendation","#27ae60", data.get('session_recommendation','—'))}
  {insight_card("⏰","Timing Clusters","#f39c12", data.get('timing_observation','—'))}
  {insight_card("🚦","Timing Recommendation","#e67e22", data.get('timing_recommendation','—'))}
  {insight_card("🔥","Streak Awareness","#9b59b6", data.get('streak_summary','—'))}

  <div style="background:#fff8f0;padding:13px 15px;border-radius:10px;margin-bottom:12px;border-left:4px solid #e67e22;">
    <p style="font-size:10px;color:#e67e22;margin:0 0 6px;text-transform:uppercase;font-weight:bold;letter-spacing:0.5px;">📍 Zone Fatigue</p>
    <ul style="list-style:none;padding:0;margin:0;">{zone_html}</ul>
  </div>

  <div style="background:#1a1a2e;padding:16px 18px;border-radius:10px;">
    <p style="color:#8899bb;font-size:10px;margin:0 0 5px;text-transform:uppercase;letter-spacing:1px;">💡 System Improvement This Week</p>
    <p style="color:white;font-size:13px;margin:0;line-height:1.6;">{data.get('improvement_suggestion','—')}</p>
  </div>
</div>
</div></body></html>"""

def send_status_email(message):
    ist_date   = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%d %b")
    subject    = f"📊 Weekly Review | {ist_date}"
    html_body  = f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:20px;">
<div style="max-width:500px;margin:auto;background:white;border-radius:12px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.1);">
<div style="background:#1a1a2e;padding:20px 24px;">
  <h2 style="color:white;margin:0;font-size:17px;">📊 Weekly Review</h2>
</div>
<div style="padding:24px;">
  <p style="font-size:14px;color:#333;">{message}</p>
  <p style="font-size:13px;color:#27ae60;font-weight:bold;margin-top:16px;">✅ System is running correctly.</p>
</div>
</div></body></html>"""
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
        print(f"  Status email sent to {recipient}")

def send_weekly_email(html_body, total, wins, losses, win_rate):
    ist_date   = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%d %b")
    subject    = f"📊 Weekly Review | {total} alerts | {wins}W {losses}L | {win_rate:.0f}% WR | {ist_date}"
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
        print(f"  Weekly review sent to {recipient}")

# ── Main ──────────────────────────────────────────────────────
print(f"Weekly review started {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
print("  Updating outcomes...")
update_outcomes()

weekly_alerts = get_weekly_alerts()

if not weekly_alerts:
    send_status_email("No trade alerts were logged in the past 7 days. Either no zones were hit, all setups scored below threshold, or the alert log is still building up.")
    exit(0)

wins     = sum(1 for a in weekly_alerts if a.get('outcome') == 'win_tp1')
losses   = sum(1 for a in weekly_alerts if a.get('outcome') == 'loss')
pending  = sum(1 for a in weekly_alerts if a.get('outcome') == 'pending')
win_rate = (wins / (wins+losses) * 100) if (wins+losses) > 0 else 0

pair_stats = {}
for a in weekly_alerts:
    p = a['pair']
    if p not in pair_stats:
        pair_stats[p] = {'alerts':0,'wins':0,'losses':0,'pending':0}
    pair_stats[p]['alerts'] += 1
    o = a.get('outcome','pending')
    if o == 'win_tp1':  pair_stats[p]['wins']    += 1
    elif o == 'loss':   pair_stats[p]['losses']  += 1
    else:               pair_stats[p]['pending'] += 1

print("  Calling Gemini for analysis...")
analysis = build_weekly_analysis(weekly_alerts, wins, losses, pending, win_rate, pair_stats)

if not analysis:
    send_status_email(f"Found {len(weekly_alerts)} alerts this week but Gemini analysis failed. Raw stats: {wins}W / {losses}L / {pending} pending.")
    exit(0)

html = build_weekly_html(analysis, weekly_alerts, wins, losses, pending, win_rate)
send_weekly_email(html, len(weekly_alerts), wins, losses, win_rate)
print("  Done.")
