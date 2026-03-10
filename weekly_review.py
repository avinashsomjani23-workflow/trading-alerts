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
except:
    alert_log = []

def clean_df(df):
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def check_outcome(alert):
    pair = alert['pair']
    symbol = next((p['symbol'] for p in config['pairs'] if p['name'] == pair), None)
    if not symbol:
        return "pending", None
    try:
        alert_time = datetime.strptime(alert['timestamp_utc'], "%Y-%m-%d %H:%M")
        bias  = alert.get('bias', 'LONG')
        entry_str = str(alert.get('entry','0')).split('-')[0].strip()
        sl    = float(alert.get('sl', 0) or 0)
        tp1   = float(alert.get('tp1', 0) or 0)
        if sl <= 0 or tp1 <= 0:
            return "pending", None

        df = clean_df(yf.download(symbol,
                                   start=(alert_time - timedelta(hours=1)).strftime('%Y-%m-%d'),
                                   interval="1h", progress=False))
        if df is None or df.empty:
            return "pending", None

        for ts, row in df.iterrows():
            ts_naive = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
            if ts_naive < alert_time:
                continue
            try:
                h = float(row['High'])
                l = float(row['Low'])
            except:
                continue
            if bias == "LONG":
                if l <= sl:   return "loss", sl
                if h >= tp1:  return "win_tp1", tp1
            elif bias == "SHORT":
                if h >= sl:   return "loss", sl
                if l <= tp1:  return "win_tp1", tp1
        return "pending", None
    except Exception as e:
        print(f"  Outcome check error {pair}: {e}")
        return "pending", None

def update_outcomes():
    for alert in alert_log:
        if alert.get('outcome') == 'pending':
            alert_time = datetime.strptime(alert['timestamp_utc'], "%Y-%m-%d %H:%M")
            age_hours = (datetime.utcnow() - alert_time).total_seconds() / 3600
            if 4 <= age_hours <= 336:  # 4h to 14 days
                outcome, outcome_price = check_outcome(alert)
                if outcome != 'pending':
                    alert['outcome'] = outcome
                    alert['outcome_price'] = outcome_price
                    alert['outcome_checked_at'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    with open(ALERT_LOG_FILE, "w") as f:
        json.dump(alert_log, f, indent=2)

def get_weekly_alerts():
    cutoff = datetime.utcnow() - timedelta(days=7)
    return [a for a in alert_log
            if datetime.strptime(a['timestamp_utc'], "%Y-%m-%d %H:%M") >= cutoff]

def build_weekly_analysis(weekly_alerts, wins, losses, pending, win_rate, pair_stats):
    high_conf = [a for a in weekly_alerts if a.get('confidence_score',0) >= 8]
    low_conf  = [a for a in weekly_alerts if a.get('confidence_score',0) < 8]

    prompt = f"""
You are a professional SMC trading analyst reviewing one week of automated trade alerts.

WEEKLY DATA:
- Total alerts: {len(weekly_alerts)}
- Wins (hit TP1): {wins} | Losses (hit SL): {losses} | Pending: {pending}
- Win rate: {win_rate:.1f}%
- High confidence (>=8): {len(high_conf)} alerts, {sum(1 for a in high_conf if a.get('outcome')=='win_tp1')} wins, {sum(1 for a in high_conf if a.get('outcome')=='loss')} losses
- Low confidence (<8): {len(low_conf)} alerts, {sum(1 for a in low_conf if a.get('outcome')=='win_tp1')} wins, {sum(1 for a in low_conf if a.get('outcome')=='loss')} losses

PAIR STATS:
{json.dumps(pair_stats, indent=2)}

FULL ALERT DATA:
{json.dumps([{{"pair":a['pair'],"time":a['timestamp_utc'],"bias":a.get('bias'),"entry":a.get('entry'),"sl":a.get('sl'),"tp1":a.get('tp1'),"confidence":a.get('confidence_score'),"confluences":a.get('confluences',[]),"outcome":a.get('outcome','pending')}} for a in weekly_alerts], indent=2)}

Return ONLY raw JSON:
{{
  "overall_grade": "A",
  "grade_comment": "one sentence on overall performance",
  "win_rate_comment": "one sentence on win rate",
  "best_pair": "pair name",
  "worst_pair": "pair name",
  "pattern_insight": "which confluence combo appeared most in wins",
  "confidence_calibration": "one sentence — are high scores winning more than low scores",
  "timing_observation": "any time-of-day patterns in wins vs losses",
  "zone_fatigue_flags": ["pair: zone alerted 3+ times weakly — if any"],
  "streak_summary": "notable streaks by pair",
  "drawdown_flag": "none or describe 3+ consecutive losses",
  "improvement_suggestion": "one specific actionable system change for next week",
  "scorecard_rows": [
    {{
      "pair": "pair",
      "time": "Day HH:MM",
      "bias": "LONG/SHORT",
      "entry": "price",
      "sl": "price",
      "tp1": "price",
      "confidence": 0,
      "outcome": "Win / Loss / Pending",
      "comment": "one sharp observation"
    }}
  ]
}}"""

    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, json=body, timeout=60)
        result = r.json()
        if "candidates" not in result:
            return None
        raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n",1)[1].rsplit("```",1)[0]
        return json.loads(raw)
    except Exception as e:
        print(f"  Gemini error: {e}")
        return None

def build_weekly_html(data, weekly_alerts, wins, losses, pending, win_rate):
    ist_now     = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%A, %d %b %Y")
    grade       = data.get('overall_grade','B')
    grade_color = "#27ae60" if grade=="A" else "#f39c12" if grade in ["B","C"] else "#e74c3c"

    scorecard_rows = ""
    for row in data.get("scorecard_rows",[]):
        outcome = row.get('outcome','Pending')
        bg      = '#f0fff4' if 'Win' in outcome else '#fff0f0' if 'Loss' in outcome else '#fffbea'
        outcome_emoji = '✅' if 'Win' in outcome else '❌' if 'Loss' in outcome else '⏳'
        conf    = row.get('confidence',0)
        cc      = "#27ae60" if conf>=8 else "#f39c12" if conf>=6 else "#e74c3c"
        scorecard_rows += f"""
        <tr style="background:{bg};border-bottom:1px solid #eee;">
          <td style="padding:7px 10px;font-weight:bold;font-size:12px;">{row.get('pair','')}</td>
          <td style="padding:7px 10px;font-size:11px;color:#666;">{row.get('time','')}</td>
          <td style="padding:7px 10px;font-size:12px;font-weight:bold;">{row.get('bias','')}</td>
          <td style="padding:7px 10px;font-size:11px;">{row.get('entry','')}</td>
          <td style="padding:7px 10px;font-size:11px;color:#e74c3c;">{row.get('sl','')}</td>
          <td style="padding:7px 10px;font-size:11px;color:#27ae60;">{row.get('tp1','')}</td>
          <td style="padding:7px 10px;font-weight:bold;color:{cc};">{conf}/10</td>
          <td style="padding:7px 10px;font-size:14px;">{outcome_emoji}</td>
          <td style="padding:7px 10px;font-size:11px;color:#888;font-style:italic;">{row.get('comment','')}</td>
        </tr>"""

    zone_flags   = data.get('zone_fatigue_flags',[])
    zone_html    = "".join([f"<li style='font-size:12px;color:#e67e22;margin-bottom:4px;'>⚠️ {z}</li>" for z in zone_flags]) if zone_flags else "<li style='font-size:12px;color:#aaa;'>No fatigued zones this week.</li>"
    drawdown     = data.get('drawdown_flag','none')
    drawdown_html= f"""<div style="background:#fef0f0;padding:12px 16px;border-radius:8px;border-left:4px solid #e74c3c;margin-bottom:16px;"><p style="margin:0;font-size:13px;color:#c0392b;"><b>⚠️ Drawdown Alert:</b> {drawdown}</p></div>""" if drawdown and drawdown.lower()!='none' else ""

    def insight_card(emoji, title, color, content):
        return f"""<div style="background:#f8f9fa;padding:13px 15px;border-radius:10px;margin-bottom:12px;border-left:4px solid {color};"><p style="font-size:10px;color:{color};margin:0 0 4px;text-transform:uppercase;font-weight:bold;letter-spacing:0.5px;">{emoji} {title}</p><p style="font-size:13px;color:#333;margin:0;line-height:1.5;">{content}</p></div>"""

    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:16px;margin:0;">
<div style="max-width:680px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">

<div style="background:#1a1a2e;padding:20px 24px;">
  <h2 style="color:white;margin:0;font-size:18px;">📊 Weekly Trading Review</h2>
  <p style="color:#8899bb;margin:5px 0 0;font-size:12px;">{ist_now}</p>
</div>

<div style="display:flex;background:#f8f9fa;border-bottom:1px solid #eee;">
  <div style="flex:1;padding:14px 16px;text-align:center;border-right:1px solid #eee;">
    <p style="margin:0;font-size:10px;color:#888;text-transform:uppercase;">Grade</p>
    <p style="margin:4px 0 0;font-size:26px;font-weight:bold;color:{grade_color};">{grade}</p>
    <p style="margin:2px 0 0;font-size:10px;color:#aaa;">{data.get('grade_comment','')[:40]}</p>
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
  <table style="width:100%;border-collapse:collapse;font-size:12px;min-width:560px;">
    <thead><tr style="background:#1a1a2e;color:white;">
      <th style="padding:8px 10px;text-align:left;">Pair</th>
      <th style="padding:8px 10px;text-align:left;">Time</th>
      <th style="padding:8px 10px;text-align:left;">Dir</th>
      <th style="padding:8px 10px;text-align:left;">Entry</th>
      <th style="padding:8px 10px;text-align:left;">SL</th>
      <th style="padding:8px 10px;text-align:left;">TP1</th>
      <th style="padding:8px 10px;text-align:left;">Score</th>
      <th style="padding:8px 10px;text-align:left;">Result</th>
      <th style="padding:8px 10px;text-align:left;">Note</th>
    </tr></thead>
    <tbody>{scorecard_rows}</tbody>
  </table>
  </div>

  {insight_card("🧠","Pattern Intelligence","#3498db", data.get('pattern_insight','—'))}
  {insight_card("📊","Confidence Calibration","#27ae60", data.get('confidence_calibration','—'))}
  {insight_card("⏰","Timing Observation","#f39c12", data.get('timing_observation','—'))}
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

def send_weekly_email(html_body, total, wins, losses, win_rate):
    ist_date = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%d %b")
    subject  = f"📊 Weekly Review | {total} alerts | {wins}W {losses}L | {win_rate:.0f}% WR | {ist_date}"
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
        print(f"  Sent to {recipient}")

# ── Main ──────────────────────────────────────────────────────
print(f"Weekly review started {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
print("  Updating outcomes...")
update_outcomes()

weekly_alerts = get_weekly_alerts()
print(f"  {len(weekly_alerts)} alerts this week.")

if not weekly_alerts:
    print("  No alerts — no email sent.")
    exit(0)

wins    = sum(1 for a in weekly_alerts if a.get('outcome')=='win_tp1')
losses  = sum(1 for a in weekly_alerts if a.get('outcome')=='loss')
pending = sum(1 for a in weekly_alerts if a.get('outcome')=='pending')
win_rate = (wins/(wins+losses)*100) if (wins+losses)>0 else 0

pair_stats = {}
for a in weekly_alerts:
    p = a['pair']
    if p not in pair_stats:
        pair_stats[p] = {'alerts':0,'wins':0,'losses':0,'pending':0}
    pair_stats[p]['alerts'] += 1
    o = a.get('outcome','pending')
    if o=='win_tp1': pair_stats[p]['wins'] += 1
    elif o=='loss':  pair_stats[p]['losses'] += 1
    else:            pair_stats[p]['pending'] += 1

print("  Building Gemini analysis...")
analysis = build_weekly_analysis(weekly_alerts, wins, losses, pending, win_rate, pair_stats)
if not analysis:
    print("  Gemini failed.")
    exit(1)

html = build_weekly_html(analysis, weekly_alerts, wins, losses, pending, win_rate)
send_weekly_email(html, len(weekly_alerts), wins, losses, win_rate)
print("  Done.")
