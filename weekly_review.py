import yfinance as yf
import pandas as pd
import json, os, smtplib, requests, time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from io import BytesIO

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY    = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_PASS    = os.environ["GMAIL_APP_PASSWORD"]
ALERT_EMAIL   = os.environ["ALERT_EMAIL"]

RISK_PER_TRADE = config["account"]["balance"] * config["account"]["risk_percent"] / 100

ALERT_LOG_FILE = "alert_log.json"
try:
    with open(ALERT_LOG_FILE) as f:
        alert_log = json.load(f)
    print(f"  Loaded {len(alert_log)} total log entries")
except Exception:
    alert_log = []
    print("  No alert log found")


# ── Helpers ────────────────────────────────────────────────────────────────────
def clean_df(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def format_candles_for_review(df, label, alert_time, n=60):
    """Returns formatted candle string + count of candles strictly after alert_time."""
    if df is None or df.empty:
        return f"{label}: No data\n", 0
    result = f"{label} candles after alert ({alert_time.strftime('%Y-%m-%d %H:%M')} UTC):\n"
    count  = 0
    for i in range(len(df)):
        try:
            ts   = df.index[i]
            ts_n = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
            if ts_n < alert_time:
                continue
            tss = ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts, 'strftime') else str(ts)[:16]
            result += (f"{tss}  O:{float(df['Open'].iloc[i]):.5f} "
                       f"H:{float(df['High'].iloc[i]):.5f} "
                       f"L:{float(df['Low'].iloc[i]):.5f} "
                       f"C:{float(df['Close'].iloc[i]):.5f}\n")
            count += 1
            if count >= n:
                break
        except Exception:
            continue
    return result, count


def get_session(utc_hour):
    if   2  <= utc_hour < 8:  return "Asian"
    elif 8  <= utc_hour < 13: return "London"
    elif 13 <= utc_hour < 21: return "New York"
    else:                     return "Off-Hours"


# ── Gemini call ────────────────────────────────────────────────────────────────
def call_gemini(prompt, retries=1):
    url  = (f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}")
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    for attempt in range(retries + 1):
        try:
            r      = requests.post(url, json=body, timeout=120)
            result = r.json()
            if "candidates" not in result:
                raise ValueError(f"No candidates: {result}")
            raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(raw), None
        except Exception as e:
            if attempt < retries:
                print(f"    Gemini attempt {attempt+1} failed ({e}), retrying in 3s...")
                time.sleep(3)
            else:
                return None, f"Gemini error: {str(e)}"


# ── Trigger + Invalidation detection — Option 5 (Gemini reads actual candles) ─
#
# Logic:
#   1. Fetch H1 + M15 candles after alert_time
#   2. Send to Gemini: trigger_text, invalid_if text, actual candle data
#   3. Gemini determines:
#      a. Did the TRIGGER CONDITION form? (yes/no, timestamp, confidence)
#      b. Did the INVALID-IF condition fire BEFORE the trigger? (yes/no, timestamp)
#   4. Safety rule: if Gemini says both fired, the earlier timestamp wins
#   5. If < 3 candles available after alert_time → stay pending (not enough data)
#   6. Low-confidence rows are flagged in journal with amber highlighting
#
def detect_trigger_and_invalidation(alert):
    """
    Returns a dict with:
      trigger_met, trigger_candle_time, trigger_confidence, trigger_reasoning,
      invalidated, invalid_candle_time, invalid_reasoning, gemini_note
    """
    pair         = alert['pair']
    symbol       = next((p['symbol'] for p in config['pairs'] if p['name'] == pair), None)
    trigger_text = alert.get('trigger', '').strip()
    invalid_text = alert.get('invalid_if', '').strip()
    bias         = alert.get('bias', '')

    blank = {
        "trigger_met": False, "trigger_candle_time": None,
        "trigger_confidence": "unknown", "trigger_reasoning": "Skipped — missing data.",
        "invalidated": False, "invalid_candle_time": None,
        "invalid_reasoning": "Skipped.", "gemini_note": ""
    }

    if not symbol or not trigger_text or not invalid_text:
        blank["gemini_note"] = "Trigger or invalid_if text missing from alert — cannot evaluate."
        return blank

    try:
        alert_time = datetime.strptime(alert['timestamp_utc'], "%Y-%m-%d %H:%M")

        # Sequential fetch — yfinance NOT thread-safe
        df1 = clean_df(yf.download(symbol,
            start=alert_time.strftime('%Y-%m-%d'),
            interval="1h", progress=False))
        df2 = clean_df(yf.download(symbol,
            start=alert_time.strftime('%Y-%m-%d'),
            interval="15m", progress=False))

        candles_h1,  count_h1  = format_candles_for_review(df1, "H1",  alert_time)
        candles_m15, count_m15 = format_candles_for_review(df2, "M15", alert_time)

        # Need at least 3 candles in at least one timeframe
        if count_h1 < 3 and count_m15 < 3:
            blank["gemini_note"] = "Fewer than 3 post-alert candles available — staying pending."
            return blank

        prompt = f"""You are reviewing a live trade alert to determine exactly two things:
1. Did the TRIGGER CONDITION form in the candle data after the alert time?
2. Did the INVALID-IF CONDITION fire BEFORE the trigger formed?

PAIR: {pair} | BIAS: {bias}
ALERT TIME: {alert['timestamp_utc']} UTC

TRIGGER CONDITION — the exact candle pattern or price action required to enter this trade:
"{trigger_text}"

INVALID-IF CONDITION — the exact price action that cancels this setup before entry:
"{invalid_text}"

CANDLE DATA AFTER THE ALERT:
{candles_h1}
{candles_m15}

INSTRUCTIONS:
- Read BOTH conditions literally. Do not generalise or paraphrase them.
- Scan candles in strict time order. The FIRST condition satisfied determines the outcome.
- If invalid-if fired BEFORE the trigger formed → set invalidated=true, trigger_met=false.
- If the trigger formed before or without the invalid-if firing → set trigger_met=true, invalidated=false.
- If neither has clearly occurred in the available data → both false, gemini_note="pending".
- Set trigger_confidence="low" if the candle data is ambiguous or the match is approximate.
  Still record your best determination — just flag low confidence so the trader can verify.
- Every timestamp you cite MUST appear in the candle data above. Never invent timestamps.

Return ONLY raw JSON. No markdown. No code fences.
{{
  "trigger_met": false,
  "trigger_candle_time": "YYYY-MM-DD HH:MM UTC or null",
  "trigger_confidence": "high or medium or low",
  "trigger_reasoning": "one sentence — cite the candle time and what it showed",
  "invalidated": false,
  "invalid_candle_time": "YYYY-MM-DD HH:MM UTC or null",
  "invalid_reasoning": "one sentence — cite the candle time and what it showed",
  "gemini_note": "one sentence summary of what happened to this setup after the alert"
}}"""

        result, err = call_gemini(prompt)
        if err or result is None:
            blank["gemini_note"] = f"Gemini detection failed: {err}"
            return blank

        trigger_met  = bool(result.get('trigger_met', False))
        invalidated  = bool(result.get('invalidated', False))
        trigger_time = result.get('trigger_candle_time')
        invalid_time = result.get('invalid_candle_time')

        # Safety: if Gemini says both fired, earlier timestamp wins
        if trigger_met and invalidated:
            try:
                t_dt = datetime.strptime(trigger_time[:16], "%Y-%m-%d %H:%M") if trigger_time else None
                i_dt = datetime.strptime(invalid_time[:16], "%Y-%m-%d %H:%M") if invalid_time else None
                if t_dt and i_dt:
                    invalidated = i_dt < t_dt
                    trigger_met = not invalidated
                elif i_dt:
                    trigger_met = False
                else:
                    invalidated = False
            except Exception:
                invalidated = False

        return {
            "trigger_met":         trigger_met,
            "trigger_candle_time": trigger_time,
            "trigger_confidence":  result.get('trigger_confidence', 'unknown'),
            "trigger_reasoning":   result.get('trigger_reasoning', ''),
            "invalidated":         invalidated,
            "invalid_candle_time": invalid_time,
            "invalid_reasoning":   result.get('invalid_reasoning', ''),
            "gemini_note":         result.get('gemini_note', '')
        }

    except Exception as e:
        blank["gemini_note"] = f"Detection error: {str(e)}"
        return blank


# ── SL/TP outcome check — only runs after trigger confirmed ───────────────────
def check_sl_tp_outcome(alert):
    pair   = alert['pair']
    symbol = next((p['symbol'] for p in config['pairs'] if p['name'] == pair), None)
    if not symbol:
        return "pending", None
    try:
        bias = alert.get('bias', 'LONG')
        sl   = float(alert.get('sl',  0) or 0)
        tp1  = float(alert.get('tp1', 0) or 0)
        if sl <= 0 or tp1 <= 0:
            return "pending", None

        # Start scanning from trigger candle time if known, else from alert time
        alert_time   = datetime.strptime(alert['timestamp_utc'], "%Y-%m-%d %H:%M")
        trigger_ts   = alert.get('trigger_candle_time')
        scan_from    = alert_time
        if trigger_ts:
            try:
                scan_from = datetime.strptime(trigger_ts[:16], "%Y-%m-%d %H:%M")
            except Exception:
                pass

        df = clean_df(yf.download(symbol,
            start=(scan_from - timedelta(hours=1)).strftime('%Y-%m-%d'),
            interval="1h", progress=False))
        if df is None:
            return "pending", None

        for ts, row in df.iterrows():
            try:
                ts_n = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
                if ts_n < scan_from:
                    continue
                h = float(row['High'])
                l = float(row['Low'])
            except Exception:
                continue
            if bias == "LONG":
                if l <= sl:  return "loss",    sl
                if h >= tp1: return "win_tp1", tp1
            elif bias == "SHORT":
                if h >= sl:  return "loss",    sl
                if l <= tp1: return "win_tp1", tp1
        return "pending", None

    except Exception as e:
        print(f"  SL/TP check error {pair}: {e}")
        return "pending", None


# ── Full outcome update pipeline ───────────────────────────────────────────────
def update_outcomes():
    """
    For each pending alert:
    1. Gemini reads trigger + invalid_if text against actual post-alert candles
    2. If invalid-if fired before trigger → 'invalidated' → excluded from win rate
    3. If trigger confirmed → SL/TP scan from trigger candle time
    4. If neither confirmed yet → stay pending
    Sequential yfinance fetches only — NOT thread-safe.
    """
    updated = 0
    for alert in alert_log:
        if alert.get('outcome') in ('win_tp1', 'loss', 'invalidated'):
            continue
        try:
            alert_time = datetime.strptime(alert['timestamp_utc'], "%Y-%m-%d %H:%M")
            age_hours  = (datetime.utcnow() - alert_time).total_seconds() / 3600
            if not (4 <= age_hours <= 336):
                continue

            print(f"    Evaluating {alert['pair']} ({alert['timestamp_utc']})...")
            det = detect_trigger_and_invalidation(alert)

            # Store all detection fields on the alert record
            alert['trigger_met']          = det['trigger_met']
            alert['trigger_candle_time']  = det['trigger_candle_time']
            alert['trigger_confidence']   = det['trigger_confidence']
            alert['trigger_reasoning']    = det['trigger_reasoning']
            alert['invalid_candle_time']  = det['invalid_candle_time']
            alert['invalid_reasoning']    = det['invalid_reasoning']
            alert['gemini_setup_note']    = det['gemini_note']

            if det['invalidated']:
                alert['outcome']            = 'invalidated'
                alert['outcome_price']      = None
                alert['outcome_checked_at'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
                print(f"      → INVALIDATED before trigger. {det['gemini_note']}")
                updated += 1

            elif det['trigger_met']:
                outcome, outcome_price = check_sl_tp_outcome(alert)
                if outcome != 'pending':
                    alert['outcome']            = outcome
                    alert['outcome_price']      = outcome_price
                    alert['outcome_checked_at'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
                    print(f"      → Trigger met ({det['trigger_confidence']} confidence). "
                          f"Outcome: {outcome}")
                    updated += 1
                else:
                    print(f"      → Trigger met, SL/TP not yet hit. Staying pending.")
            else:
                print(f"      → Trigger not yet met. Staying pending.")

        except Exception as e:
            print(f"  Update error {alert.get('pair','?')}: {e}")

    with open(ALERT_LOG_FILE, "w") as f:
        json.dump(alert_log, f, indent=2)
    print(f"  Updated {updated} outcomes")


def get_weekly_alerts():
    cutoff = datetime.utcnow() - timedelta(days=7)
    weekly = [a for a in alert_log
              if datetime.strptime(a['timestamp_utc'], "%Y-%m-%d %H:%M") >= cutoff]
    print(f"  {len(weekly)} alerts in last 7 days")
    return weekly


# ── Estimated P&L ──────────────────────────────────────────────────────────────
def estimate_pnl(alert):
    outcome = alert.get('outcome', 'pending')
    if outcome not in ('win_tp1', 'loss'):
        return 0.0, "—"
    try:
        bias  = alert.get('bias', '')
        entry = float(str(alert.get('entry', '0')).split('-')[0].strip() or 0)
        sl    = float(alert.get('sl',  0) or 0)
        tp1   = float(alert.get('tp1', 0) or 0)
        if entry <= 0 or sl <= 0 or tp1 <= 0:
            return 0.0, "—"
        risk_pts   = (entry - sl)   if bias == "LONG" else (sl    - entry)
        reward_pts = (tp1   - entry) if bias == "LONG" else (entry - tp1)
        if risk_pts <= 0:
            return 0.0, "—"
        rr = reward_pts / risk_pts
        if outcome == 'win_tp1':
            return round(RISK_PER_TRADE * rr, 2), f"+{round(reward_pts, 5)}"
        else:
            return round(-RISK_PER_TRADE, 2), f"-{round(risk_pts, 5)}"
    except Exception:
        return 0.0, "—"


# ── Gemini weekly analysis (email body cards) ──────────────────────────────────
def build_weekly_analysis(weekly_alerts, wins, losses, invalidated_count,
                          pending, win_rate, pair_stats):
    triggered = [a for a in weekly_alerts if a.get('outcome') in ('win_tp1', 'loss')]

    intraday_count = len([a for a in weekly_alerts if 'intraday'  in a.get('alert_type','')])
    swing_count    = len([a for a in weekly_alerts if 'swing'     in a.get('alert_type','')])
    breakout_count = len([a for a in weekly_alerts if a.get('alert_type','') == 'breakout'])

    session_stats = {}
    for a in triggered:
        try:
            utc_hour = datetime.strptime(a['timestamp_utc'], "%Y-%m-%d %H:%M").hour
            session  = get_session(utc_hour)
            if session not in session_stats:
                session_stats[session] = {'wins': 0, 'losses': 0}
            if a.get('outcome') == 'win_tp1': session_stats[session]['wins']   += 1
            else:                             session_stats[session]['losses'] += 1
        except Exception:
            pass
    for s in session_stats.values():
        w = s['wins']; l = s['losses']
        s['win_rate'] = round(w/(w+l)*100, 1) if (w+l) > 0 else None

    hour_buckets = {}
    for a in triggered:
        try:
            utc_dt   = datetime.strptime(a['timestamp_utc'], "%Y-%m-%d %H:%M")
            ist_hour = (utc_dt + timedelta(hours=5, minutes=30)).hour
            bucket   = f"{ist_hour:02d}:00"
            if bucket not in hour_buckets:
                hour_buckets[bucket] = {'wins': 0, 'losses': 0}
            if a.get('outcome') == 'win_tp1': hour_buckets[bucket]['wins']   += 1
            else:                             hour_buckets[bucket]['losses'] += 1
        except Exception:
            pass
    loss_clusters = [f"{h} IST ({v['losses']} losses)"
                     for h, v in sorted(hour_buckets.items()) if v['losses'] >= 2]

    invalidation_rate = round(invalidated_count / len(weekly_alerts) * 100, 1) if weekly_alerts else 0

    alert_summary = []
    for a in weekly_alerts:
        alert_summary.append({
            "pair":               a.get('pair',''),
            "ist_time":           a.get('ist_time',''),
            "alert_type":         a.get('alert_type','zone_intraday'),
            "bias":               a.get('bias',''),
            "confidence":         a.get('confidence_score',0),
            "confluences":        a.get('confluences',[]),
            "outcome":            a.get('outcome','pending'),
            "trigger_met":        a.get('trigger_met', None),
            "trigger_confidence": a.get('trigger_confidence','unknown'),
            "gemini_note":        a.get('gemini_setup_note','')
        })

    prompt = f"""You are a highly skilled SMC trading analyst reviewing one week of automated alerts.
Win rate is calculated on TRIGGERED trades only (wins + losses). Invalidated and pending excluded.

WEEKLY NUMBERS:
- Total alerts fired: {len(weekly_alerts)} (Intraday: {intraday_count}, Swing: {swing_count}, Breakout: {breakout_count})
- Triggered and resolved: Wins {wins} | Losses {losses} | Win Rate {win_rate:.1f}%
- Invalidated (setup broke before trigger — not taken): {invalidated_count} ({invalidation_rate}% of all alerts)
- Still pending resolution: {pending}

SESSION BREAKDOWN (triggered trades only):
{json.dumps(session_stats, indent=2)}

LOSS TIME CLUSTERS (IST hours with 2+ losses):
{str(loss_clusters) if loss_clusters else "None identified"}

PAIR PERFORMANCE:
{json.dumps(pair_stats, indent=2)}

FULL ALERT DATA:
{json.dumps(alert_summary, indent=2)}

Return ONLY raw JSON. No markdown. No code fences.
{{
  "overall_grade": "A/B/C/D",
  "grade_comment": "one sentence based on triggered trades only",
  "best_pair": "pair name or none",
  "worst_pair": "pair name or none",
  "pattern_insight": "which confluence combination appeared most in winning triggered trades",
  "confidence_calibration": "are high scored alerts winning more — cite numbers",
  "session_summary": "best and worst session with win rates and trade counts",
  "session_recommendation": "one actionable sentence",
  "timing_observation": "IST windows where losses cluster",
  "timing_recommendation": "one actionable sentence on time filtering",
  "intraday_vs_swing": "one sentence comparing intraday vs swing triggered performance",
  "invalidation_insight": "what the {invalidation_rate}% invalidation rate tells us about setup quality this week",
  "zone_fatigue_flags": ["list any pair+zone invalidated or alerted 3+ times — empty array if none"],
  "streak_summary": "notable winning or losing streaks in the triggered set",
  "drawdown_flag": "none or describe if 3+ consecutive losses in triggered trades",
  "improvement_suggestion": "one specific actionable change based on this week only"
}}"""

    result, err = call_gemini(prompt)
    if err:
        print(f"  Gemini analysis error: {err}")
        return None
    return result


# ── Excel journal builder — 30 columns + Summary tab ──────────────────────────
def build_excel_journal(weekly_alerts, analysis):
    wb = openpyxl.Workbook()

    # Colour palette
    C_HDR_BG    = "1A1A2E"
    C_HDR_FG    = "FFFFFF"
    C_SEC_BG    = "2C3E6B"
    C_WIN       = "D5F5E3"
    C_LOSS      = "FADBD8"
    C_INVALID   = "FDEBD0"
    C_PENDING   = "FDFEFE"
    C_LOWCONF   = "FFF3CD"   # amber — low confidence trigger
    C_ALT       = "F2F3F4"
    C_ID        = "D6EAF8"   # identity cols
    C_SETUP     = "D5F5E3"   # setup quality cols
    C_RISK      = "FDEDEC"   # risk cols
    C_TRIG      = "FEF9E7"   # trigger/invalidation cols
    C_OUT       = "EBF5FB"   # outcome cols
    C_LEARN     = "F5EEF8"   # learning cols

    thin   = Side(style='thin', color='CCCCCC')
    bdr    = Border(left=thin, right=thin, top=thin, bottom=thin)

    def hfont(size=10): return Font(name='Arial', bold=True, color=C_HDR_FG, size=size)
    def cfont(bold=False, color="000000", size=9):
        return Font(name='Arial', bold=bold, color=color, size=size)
    def bg(hex_c): return PatternFill("solid", fgColor=hex_c)
    def align(h='left', wrap=True):
        return Alignment(horizontal=h, vertical='center', wrap_text=wrap)
    def bc(cell): cell.border = bdr

    # ═══════════════════════════════════════════════
    # TAB 1 — WEEKLY SUMMARY
    # ═══════════════════════════════════════════════
    ws_s = wb.active
    ws_s.title = "Weekly Summary"
    ws_s.sheet_view.showGridLines = False
    ws_s.column_dimensions['A'].width = 38
    ws_s.column_dimensions['B'].width = 26

    ist_now = (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%d %b %Y")

    # Compute summary metrics
    wins_list    = [a for a in weekly_alerts if a.get('outcome') == 'win_tp1']
    losses_list  = [a for a in weekly_alerts if a.get('outcome') == 'loss']
    inv_list     = [a for a in weekly_alerts if a.get('outcome') == 'invalidated']
    pend_list    = [a for a in weekly_alerts if a.get('outcome') == 'pending']
    triggered    = wins_list + losses_list

    wins_n       = len(wins_list)
    losses_n     = len(losses_list)
    inv_n        = len(inv_list)
    pend_n       = len(pend_list)
    wr           = round(wins_n/(wins_n+losses_n)*100, 1) if (wins_n+losses_n) > 0 else 0
    inv_rate     = round(inv_n/len(weekly_alerts)*100, 1) if weekly_alerts else 0
    net_pnl      = sum(estimate_pnl(a)[0] for a in triggered)

    rrs = []
    for a in wins_list:
        try:
            entry = float(str(a.get('entry','0')).split('-')[0].strip() or 0)
            sl    = float(a.get('sl', 0) or 0)
            tp1   = float(a.get('tp1', 0) or 0)
            bias  = a.get('bias','')
            if entry > 0 and sl > 0 and tp1 > 0:
                rp = (entry-sl)   if bias == "LONG" else (sl-entry)
                rw = (tp1-entry)  if bias == "LONG" else (entry-tp1)
                if rp > 0: rrs.append(rw/rp)
        except Exception:
            pass
    avg_rr    = round(sum(rrs)/len(rrs), 2) if rrs else 0.0
    avg_cw    = round(sum(a.get('confidence_score',0) for a in wins_list)  /len(wins_list),  1) if wins_list   else 0
    avg_cl    = round(sum(a.get('confidence_score',0) for a in losses_list)/len(losses_list),1) if losses_list else 0

    def safe(key): return (analysis.get(key,'—') or '—') if analysis else '—'

    # Title
    ws_s.merge_cells('A1:B1')
    ws_s['A1'].value     = f"Weekly Trading Review — {ist_now}"
    ws_s['A1'].font      = Font(name='Arial', bold=True, color=C_HDR_FG, size=13)
    ws_s['A1'].fill      = bg(C_HDR_BG)
    ws_s['A1'].alignment = align('center')
    ws_s.row_dimensions[1].height = 28

    rows = [
        ("PERFORMANCE",                  None,                   C_SEC_BG),
        ("Total Alerts Fired",            len(weekly_alerts),     None),
        ("  — Intraday",                  len([a for a in weekly_alerts if 'intraday' in a.get('alert_type','')]), None),
        ("  — Swing",                     len([a for a in weekly_alerts if 'swing'    in a.get('alert_type','')]), None),
        ("  — Breakout",                  len([a for a in weekly_alerts if a.get('alert_type','')=='breakout']),   None),
        ("Triggered Trades (W + L)",      wins_n + losses_n,      None),
        ("Wins",                          wins_n,                  None),
        ("Losses",                        losses_n,                None),
        ("Win Rate (triggered trades only)", f"{wr}%",            None),
        ("Invalidated (not taken)",       inv_n,                   None),
        ("Invalidation Rate",             f"{inv_rate}%",          None),
        ("Still Pending",                 pend_n,                  None),
        ("Net Estimated P&L (USD)",       f"${net_pnl:+.2f}",      None),
        ("QUALITY METRICS",              None,                   C_SEC_BG),
        ("Avg R:R on Winning Trades",     f"{avg_rr}x",            None),
        ("Avg Confidence Score — Wins",   f"{avg_cw}/10",          None),
        ("Avg Confidence Score — Losses", f"{avg_cl}/10",          None),
        ("ANALYSIS",                     None,                   C_SEC_BG),
        ("Overall Grade",                 safe('overall_grade'),   None),
        ("Grade Comment",                 safe('grade_comment'),   None),
        ("Best Pair",                     safe('best_pair'),       None),
        ("Weakest Pair",                  safe('worst_pair'),      None),
        ("Pattern Insight",               safe('pattern_insight'), None),
        ("Confidence Calibration",        safe('confidence_calibration'), None),
        ("Session Summary",               safe('session_summary'), None),
        ("Session Recommendation",        safe('session_recommendation'), None),
        ("Timing Observation",            safe('timing_observation'), None),
        ("Timing Recommendation",         safe('timing_recommendation'), None),
        ("Intraday vs Swing",             safe('intraday_vs_swing'), None),
        ("Invalidation Insight",          safe('invalidation_insight'), None),
        ("Streak Summary",                safe('streak_summary'),  None),
        ("Drawdown Flag",                 safe('drawdown_flag'),   None),
        ("Improvement Suggestion",        safe('improvement_suggestion'), None),
    ]

    ri = 2
    for label, value, sec in rows:
        if sec:
            ws_s.merge_cells(f'A{ri}:B{ri}')
            ws_s[f'A{ri}'].value     = label
            ws_s[f'A{ri}'].font      = hfont(10)
            ws_s[f'A{ri}'].fill      = bg(sec)
            ws_s[f'A{ri}'].alignment = align()
            ws_s.row_dimensions[ri].height = 18
        else:
            ws_s[f'A{ri}'].value     = label
            ws_s[f'B{ri}'].value     = value
            ws_s[f'A{ri}'].font      = cfont(bold=True, size=10)
            ws_s[f'B{ri}'].font      = cfont(size=10)
            row_bg = C_ALT if ri % 2 == 0 else "FFFFFF"
            ws_s[f'A{ri}'].fill      = bg(row_bg)
            ws_s[f'B{ri}'].fill      = bg(row_bg)
            ws_s[f'A{ri}'].alignment = align()
            ws_s[f'B{ri}'].alignment = align('left')
            ws_s.row_dimensions[ri].height = 20
        bc(ws_s[f'A{ri}'])
        bc(ws_s[f'B{ri}'])
        ri += 1

    # Zone flags
    flags = safe('zone_fatigue_flags')
    if isinstance(flags, list) and flags:
        ws_s.merge_cells(f'A{ri}:B{ri}')
        ws_s[f'A{ri}'].value     = "ZONE DEPLETION FLAGS"
        ws_s[f'A{ri}'].font      = hfont(10)
        ws_s[f'A{ri}'].fill      = bg(C_SEC_BG)
        ws_s[f'A{ri}'].alignment = align()
        bc(ws_s[f'A{ri}']); bc(ws_s[f'B{ri}'])
        ri += 1
        for flag in flags:
            ws_s.merge_cells(f'A{ri}:B{ri}')
            ws_s[f'A{ri}'].value     = f"⚠ {flag}"
            ws_s[f'A{ri}'].font      = cfont(color="E67E22", size=10)
            ws_s[f'A{ri}'].fill      = bg("FFF3CD")
            ws_s[f'A{ri}'].alignment = align()
            bc(ws_s[f'A{ri}']); bc(ws_s[f'B{ri}'])
            ri += 1

    # ═══════════════════════════════════════════════
    # TAB 2 — TRADE JOURNAL (30 columns)
    # ═══════════════════════════════════════════════
    ws = wb.create_sheet("Trade Journal")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = 'A3'

    # (column_name, width, group_colour)
    columns = [
        # IDENTITY (1-5)
        ("Date (IST)",                16, C_ID),
        ("Pair",                      10, C_ID),
        ("Session",                   12, C_ID),
        ("Timeframe",                 13, C_ID),
        ("Direction",                 11, C_ID),
        # SETUP QUALITY (6-10)
        ("Scoring Type",              16, C_SETUP),
        ("Confidence Score",          14, C_SETUP),
        ("Key Confluences",           40, C_SETUP),
        ("What Was Missing",          32, C_SETUP),
        ("Trigger Condition",         40, C_SETUP),
        # RISK LEVELS (11-17)
        ("Entry Price",               13, C_RISK),
        ("Stop Loss",                 13, C_RISK),
        ("TP1",                       13, C_RISK),
        ("TP2",                       13, C_RISK),
        ("Risk:Reward (TP1)",         16, C_RISK),
        ("Lot Size",                  11, C_RISK),
        ("Risk (USD)",                12, C_RISK),
        # TRIGGER & INVALIDATION (18-21)
        ("Trigger Met",               13, C_TRIG),
        ("Trigger Candle Time",       20, C_TRIG),
        ("Trigger Confidence",        17, C_TRIG),
        ("Invalidated",               13, C_TRIG),
        # OUTCOME (22-25)
        ("Outcome",                   14, C_OUT),
        ("Outcome Price",             14, C_OUT),
        ("Points +/-",                13, C_OUT),
        ("Est. P&L (USD)",            14, C_OUT),
        # LEARNING (26-30)
        ("Gemini Setup Note",         42, C_LEARN),
        ("Session Note",              36, C_LEARN),
        ("Lesson",                    36, C_LEARN),
        ("System Flag",               24, C_LEARN),
        ("Why This Trade Was Taken",  42, C_LEARN),
    ]

    # Group header row (row 1)
    groups = [
        ("IDENTITY",               1,  5,  C_ID),
        ("SETUP QUALITY",          6,  10, C_SETUP),
        ("RISK LEVELS",            11, 17, C_RISK),
        ("TRIGGER & INVALIDATION", 18, 21, C_TRIG),
        ("OUTCOME",                22, 25, C_OUT),
        ("LEARNING",               26, 30, C_LEARN),
    ]
    for grp, c_start, c_end, grp_col in groups:
        sl = get_column_letter(c_start)
        el = get_column_letter(c_end)
        ws.merge_cells(f'{sl}1:{el}1')
        cell = ws[f'{sl}1']
        cell.value     = grp
        cell.font      = Font(name='Arial', bold=True, color=C_HDR_FG, size=10)
        cell.fill      = bg(C_HDR_BG)
        cell.alignment = align('center')
        bc(cell)
    ws.row_dimensions[1].height = 22

    # Column header row (row 2)
    for ci, (col_name, col_w, col_col) in enumerate(columns, start=1):
        lt = get_column_letter(ci)
        cell = ws[f'{lt}2']
        cell.value     = col_name
        cell.font      = Font(name='Arial', bold=True, color="1A1A2E", size=9)
        cell.fill      = bg(col_col)
        cell.alignment = align('center')
        bc(cell)
        ws.column_dimensions[lt].width = col_w
    ws.row_dimensions[2].height = 36

    # Data rows starting from row 3
    dr = 3
    for a in weekly_alerts:
        outcome   = a.get('outcome', 'pending')
        trig_conf = (a.get('trigger_confidence') or 'unknown').lower()

        # Row background
        if   outcome == 'win_tp1':    row_bg = C_WIN
        elif outcome == 'loss':       row_bg = C_LOSS
        elif outcome == 'invalidated':row_bg = C_INVALID
        else:                         row_bg = C_PENDING
        # Amber override for low-confidence trigger detection
        if trig_conf == 'low' and outcome not in ('invalidated','pending'):
            row_bg = C_LOWCONF

        pnl_usd, pts_str = estimate_pnl(a)

        # R:R
        rr_str = "—"
        try:
            entry = float(str(a.get('entry','0')).split('-')[0].strip() or 0)
            sl_v  = float(a.get('sl',  0) or 0)
            tp1_v = float(a.get('tp1', 0) or 0)
            bias  = a.get('bias','')
            if entry > 0 and sl_v > 0 and tp1_v > 0:
                rp = (entry-sl_v)  if bias=="LONG" else (sl_v-entry)
                rw = (tp1_v-entry) if bias=="LONG" else (entry-tp1_v)
                if rp > 0: rr_str = f"{round(rw/rp,2)}:1"
        except Exception:
            pass

        # Alert type / scoring label
        atype = a.get('alert_type','zone_intraday')
        if atype == 'breakout':
            score_type = f"Breakout {a.get('confidence_score',0)}/5"
            tf_label   = "Breakout"
        elif 'swing' in atype:
            score_type = f"SMC {a.get('confidence_score',0)}/10"
            tf_label   = "Swing"
        else:
            score_type = f"SMC {a.get('confidence_score',0)}/10"
            tf_label   = "Intraday"

        # Date + session
        try:
            utc_dt   = datetime.strptime(a['timestamp_utc'], "%Y-%m-%d %H:%M")
            ist_dt   = utc_dt + timedelta(hours=5, minutes=30)
            date_str = ist_dt.strftime("%d %b %Y %H:%M")
            session  = get_session(utc_dt.hour)
        except Exception:
            date_str = a.get('timestamp_utc','')
            session  = "—"

        # Confluences
        conf_str = "; ".join(a.get('confluences', []))

        # Missing
        miss_raw = a.get('missing', [])
        if isinstance(miss_raw, list):
            miss_str = "; ".join(
                m.get('item','') if isinstance(m, dict) else str(m)
                for m in miss_raw)
        else:
            miss_str = str(miss_raw)

        # Outcome label
        outcome_label = {
            'win_tp1':    'WIN (TP1)',
            'loss':       'LOSS (SL)',
            'invalidated':'INVALIDATED',
            'pending':    'PENDING'
        }.get(outcome, outcome.upper())

        trig_met_lbl  = {True:'YES', False:'NO', None:'UNKNOWN'}.get(a.get('trigger_met'), 'UNKNOWN')
        inv_lbl       = 'YES' if outcome == 'invalidated' else 'NO'
        pnl_display   = f"${pnl_usd:+.2f}" if pnl_usd != 0.0 else "—"
        op            = a.get('outcome_price')
        op_str        = str(round(float(op), 5)) if op is not None else "—"

        # Lot size — stored on alert if available
        lot_str = str(a.get('lot_size','—')) if a.get('lot_size') else "—"

        # Gemini setup note also serves as session note if no separate one stored
        setup_note   = a.get('gemini_setup_note','')
        session_note = a.get('session_note', setup_note)

        row_vals = [
            date_str,                              # 1  Date (IST)
            a.get('pair',''),                      # 2  Pair
            session,                               # 3  Session
            tf_label,                              # 4  Timeframe
            a.get('bias',''),                      # 5  Direction
            score_type,                            # 6  Scoring Type
            a.get('confidence_score', 0),          # 7  Confidence Score
            conf_str,                              # 8  Key Confluences
            miss_str,                              # 9  What Was Missing
            a.get('trigger',''),                   # 10 Trigger Condition
            a.get('entry',''),                     # 11 Entry Price
            a.get('sl',''),                        # 12 Stop Loss
            a.get('tp1',''),                       # 13 TP1
            a.get('tp2',''),                       # 14 TP2
            rr_str,                                # 15 Risk:Reward (TP1)
            lot_str,                               # 16 Lot Size
            f"${RISK_PER_TRADE:.0f}",              # 17 Risk (USD)
            trig_met_lbl,                          # 18 Trigger Met
            a.get('trigger_candle_time','—') or '—', # 19 Trigger Candle Time
            trig_conf.upper(),                     # 20 Trigger Confidence
            inv_lbl,                               # 21 Invalidated
            outcome_label,                         # 22 Outcome
            op_str,                                # 23 Outcome Price
            pts_str,                               # 24 Points +/-
            pnl_display,                           # 25 Est. P&L (USD)
            setup_note,                            # 26 Gemini Setup Note
            session_note,                          # 27 Session Note
            a.get('lesson',''),                    # 28 Lesson
            a.get('system_flag',''),               # 29 System Flag
            a.get('why_taken',''),                 # 30 Why This Trade Was Taken
        ]

        for ci, val in enumerate(row_vals, start=1):
            lt   = get_column_letter(ci)
            cell = ws[f'{lt}{dr}']
            cell.value     = val
            cell.font      = cfont(size=9)
            cell.fill      = bg(row_bg)
            cell.alignment = align()
            bc(cell)

            # Column-specific formatting
            if ci == 7:   # Confidence score
                conf = a.get('confidence_score', 0)
                cell.font = cfont(bold=True, size=9,
                    color=("27AE60" if conf>=8 else "F39C12" if conf>=6 else "E74C3C"))
            if ci == 18:  # Trigger Met
                cell.font = cfont(bold=True, size=9,
                    color=("1E8449" if trig_met_lbl=='YES' else
                           "C0392B" if trig_met_lbl=='NO' else "888888"))
            if ci == 22:  # Outcome
                clr = {"WIN (TP1)":"1E8449","LOSS (SL)":"C0392B",
                       "INVALIDATED":"E67E22","PENDING":"888888"}
                cell.font = cfont(bold=True, size=9, color=clr.get(outcome_label,"000000"))
            if ci == 25:  # Est. P&L
                cell.font = cfont(bold=True, size=9,
                    color=("1E8449" if pnl_usd>0 else "C0392B" if pnl_usd<0 else "888888"))

        ws.row_dimensions[dr].height = 58
        dr += 1

    # Legend
    dr += 1
    ws.merge_cells(f'A{dr}:C{dr}')
    ws[f'A{dr}'].value = "COLOUR LEGEND"
    ws[f'A{dr}'].font  = cfont(bold=True, size=9)
    ws[f'A{dr}'].fill  = bg(C_HDR_BG)
    ws[f'A{dr}'].font  = Font(name='Arial', bold=True, color=C_HDR_FG, size=9)
    ws[f'A{dr}'].alignment = align()
    dr += 1

    legend = [
        (C_WIN,     "Green  — Trade triggered and won (TP1 hit). Counted in win rate."),
        (C_LOSS,    "Red    — Trade triggered and lost (SL hit). Counted in win rate."),
        (C_INVALID, "Orange — Setup invalidated before trigger. NOT counted in win rate."),
        (C_LOWCONF, "Amber  — Low-confidence trigger detection. Verify this row manually."),
        (C_PENDING, "White  — Outcome still pending resolution."),
    ]
    for leg_bg, leg_text in legend:
        ws.merge_cells(f'A{dr}:F{dr}')
        ws[f'A{dr}'].value     = leg_text
        ws[f'A{dr}'].fill      = bg(leg_bg)
        ws[f'A{dr}'].font      = cfont(size=9)
        ws[f'A{dr}'].alignment = align()
        ws.row_dimensions[dr].height = 15
        dr += 1

    # Save to bytes
    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


# ── Email HTML — analysis insight cards only, no scorecard table ───────────────
def insight_card(title, color, content):
    return (f'<div style="background:#f8f9fa;padding:13px 15px;border-radius:10px;'
            f'margin-bottom:12px;border-left:4px solid {color};">'
            f'<p style="font-size:10px;color:{color};margin:0 0 4px;text-transform:uppercase;'
            f'font-weight:bold;letter-spacing:0.5px;">{title}</p>'
            f'<p style="font-size:13px;color:#333;margin:0;line-height:1.5;">{content}</p></div>')


def build_weekly_email_html(data, weekly_alerts, wins, losses,
                            invalidated_count, pending, win_rate):
    ist_now    = (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%A, %d %b %Y")
    grade      = (data.get('overall_grade','—') or '—') if data else '—'
    grade_color = "#27ae60" if grade=="A" else "#f39c12" if grade in ("B","C") else "#e74c3c"
    total       = len(weekly_alerts)
    net_pnl     = sum(estimate_pnl(a)[0] for a in weekly_alerts
                      if a.get('outcome') in ('win_tp1','loss'))
    pnl_color   = "#27ae60" if net_pnl >= 0 else "#e74c3c"

    def safe(key): return (data.get(key,'—') or '—') if data else '—'

    drawdown = safe('drawdown_flag')
    ddhtml   = (f'<div style="background:#fef0f0;padding:12px 16px;border-radius:8px;'
                f'border-left:4px solid #e74c3c;margin-bottom:16px;">'
                f'<p style="margin:0;font-size:13px;color:#c0392b;">'
                f'<b>⚠ DRAWDOWN ALERT:</b> {drawdown}</p></div>'
                if drawdown and drawdown.lower() != 'none' else "")

    flags     = safe('zone_fatigue_flags')
    flags_html = ""
    if isinstance(flags, list) and flags:
        flags_html = "".join([
            f'<li style="font-size:12px;color:#e67e22;margin-bottom:4px;">⚠ {z}</li>'
            for z in flags
        ])
    else:
        flags_html = '<li style="font-size:12px;color:#aaa;">No depletion flags this week.</li>'

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:16px;margin:0;">
<div style="max-width:640px;margin:auto;background:white;border-radius:14px;
     overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">

  <div style="background:#1a1a2e;padding:20px 24px;">
    <h2 style="color:white;margin:0;font-size:18px;">Weekly Trading Review</h2>
    <p style="color:#8899bb;margin:5px 0 0;font-size:12px;">{ist_now}</p>
  </div>

  <div style="display:flex;background:#f8f9fa;border-bottom:1px solid #eee;flex-wrap:wrap;">
    <div style="flex:1;min-width:80px;padding:14px 10px;text-align:center;border-right:1px solid #eee;">
      <p style="margin:0;font-size:9px;color:#888;text-transform:uppercase;">Grade</p>
      <p style="margin:4px 0 0;font-size:26px;font-weight:bold;color:{grade_color};">{grade}</p>
    </div>
    <div style="flex:1;min-width:80px;padding:14px 10px;text-align:center;border-right:1px solid #eee;">
      <p style="margin:0;font-size:9px;color:#888;text-transform:uppercase;">Win Rate</p>
      <p style="margin:4px 0 0;font-size:22px;font-weight:bold;color:#1a1a2e;">{win_rate:.0f}%</p>
      <p style="margin:2px 0 0;font-size:9px;color:#aaa;">triggered only</p>
    </div>
    <div style="flex:1;min-width:80px;padding:14px 10px;text-align:center;border-right:1px solid #eee;">
      <p style="margin:0;font-size:9px;color:#888;text-transform:uppercase;">W / L</p>
      <p style="margin:4px 0 0;font-size:20px;font-weight:bold;">
        <span style="color:#27ae60;">{wins}</span> /
        <span style="color:#e74c3c;">{losses}</span>
      </p>
    </div>
    <div style="flex:1;min-width:80px;padding:14px 10px;text-align:center;border-right:1px solid #eee;">
      <p style="margin:0;font-size:9px;color:#888;text-transform:uppercase;">Invalidated</p>
      <p style="margin:4px 0 0;font-size:22px;font-weight:bold;color:#e67e22;">{invalidated_count}</p>
      <p style="margin:2px 0 0;font-size:9px;color:#aaa;">not taken</p>
    </div>
    <div style="flex:1;min-width:80px;padding:14px 10px;text-align:center;border-right:1px solid #eee;">
      <p style="margin:0;font-size:9px;color:#888;text-transform:uppercase;">Alerts</p>
      <p style="margin:4px 0 0;font-size:22px;font-weight:bold;color:#1a1a2e;">{total}</p>
    </div>
    <div style="flex:1;min-width:80px;padding:14px 10px;text-align:center;">
      <p style="margin:0;font-size:9px;color:#888;text-transform:uppercase;">Est. P&L</p>
      <p style="margin:4px 0 0;font-size:20px;font-weight:bold;color:{pnl_color};">${net_pnl:+.0f}</p>
    </div>
  </div>

  <div style="padding:20px 24px;">
    {ddhtml}

    <p style="font-size:12px;color:#2980b9;background:#e8f4fd;padding:10px 14px;
       border-radius:8px;margin:0 0 20px;">
      📎 Full trade journal attached — 30-column Excel workbook including trigger detection,
      invalidation status, R:R, P&L, lessons, and why each trade was taken.
    </p>

    {insight_card("PATTERN INTELLIGENCE", "#3498db", safe('pattern_insight'))}
    {insight_card("CONFIDENCE CALIBRATION", "#27ae60", safe('confidence_calibration'))}
    {insight_card("SESSION WIN RATE", "#3498db", safe('session_summary'))}
    {insight_card("SESSION RECOMMENDATION", "#27ae60", safe('session_recommendation'))}
    {insight_card("TIMING CLUSTERS", "#f39c12", safe('timing_observation'))}
    {insight_card("TIMING RECOMMENDATION", "#e67e22", safe('timing_recommendation'))}
    {insight_card("INTRADAY vs SWING", "#9b59b6", safe('intraday_vs_swing'))}
    {insight_card("INVALIDATION INSIGHT", "#e67e22", safe('invalidation_insight'))}
    {insight_card("STREAK AWARENESS", "#9b59b6", safe('streak_summary'))}

    <div style="background:#fff8f0;padding:13px 15px;border-radius:10px;
         margin-bottom:12px;border-left:4px solid #e67e22;">
      <p style="font-size:10px;color:#e67e22;margin:0 0 6px;text-transform:uppercase;
         font-weight:bold;letter-spacing:0.5px;">ZONE DEPLETION FLAGS</p>
      <ul style="list-style:none;padding:0;margin:0;">{flags_html}</ul>
    </div>

    <div style="background:#1a1a2e;padding:16px 18px;border-radius:10px;">
      <p style="color:#8899bb;font-size:10px;margin:0 0 5px;text-transform:uppercase;
         letter-spacing:1px;">SYSTEM IMPROVEMENT THIS WEEK</p>
      <p style="color:white;font-size:13px;margin:0;line-height:1.6;">
        {safe('improvement_suggestion')}</p>
    </div>
  </div>
</div>
</body>
</html>"""


# ── Email senders ──────────────────────────────────────────────────────────────
def send_status_email(message):
    ist_date  = (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%d %b")
    subject   = f"Weekly Review | {ist_date}"
    html_body = (f'<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;'
                 f'background:#f0f2f5;padding:20px;">'
                 f'<div style="max-width:500px;margin:auto;background:white;'
                 f'border-radius:12px;overflow:hidden;">'
                 f'<div style="background:#1a1a2e;padding:20px 24px;">'
                 f'<h2 style="color:white;margin:0;">Weekly Review</h2></div>'
                 f'<div style="padding:24px;">'
                 f'<p style="font-size:14px;color:#333;">{message}</p>'
                 f'</div></div></body></html>')
    recipients = config["account"].get("alert_emails", [ALERT_EMAIL])
    for recipient in recipients:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_ADDRESS
        msg["To"]      = recipient
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_PASS)
            server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())


def send_weekly_email(html_body, excel_bytes, total, wins, losses,
                      invalidated_count, win_rate):
    ist_dt   = datetime.utcnow() + timedelta(hours=5, minutes=30)
    filename = f"Trading_Journal_{ist_dt.strftime('%d_%b_%Y')}.xlsx"
    subject  = (f"Weekly Review | {total} alerts | "
                f"{wins}W {losses}L {invalidated_count}INV | "
                f"{win_rate:.0f}% WR | {ist_dt.strftime('%d %b')}")
    recipients = config["account"].get("alert_emails", [ALERT_EMAIL])
    for recipient in recipients:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_ADDRESS
        msg["To"]      = recipient
        msg.attach(MIMEText(html_body, "html"))

        part = MIMEBase("application",
                        "vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        part.set_payload(excel_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=filename)
        msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_PASS)
            server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
        print(f"  Weekly review + journal sent to {recipient}")


# ── MAIN ───────────────────────────────────────────────────────────────────────
print(f"Weekly review started {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
print("  Running outcome update (Gemini trigger detection + SL/TP check)...")
update_outcomes()

weekly_alerts = get_weekly_alerts()

if not weekly_alerts:
    send_status_email(
        "No alerts logged in the past 7 days. "
        "System is still building history — expected in early weeks."
    )
    exit(0)

wins              = sum(1 for a in weekly_alerts if a.get('outcome') == 'win_tp1')
losses            = sum(1 for a in weekly_alerts if a.get('outcome') == 'loss')
invalidated_count = sum(1 for a in weekly_alerts if a.get('outcome') == 'invalidated')
pending           = sum(1 for a in weekly_alerts if a.get('outcome') == 'pending')
win_rate          = (wins/(wins+losses)*100) if (wins+losses) > 0 else 0

pair_stats = {}
for a in weekly_alerts:
    p = a['pair']
    if p not in pair_stats:
        pair_stats[p] = {'alerts':0,'wins':0,'losses':0,'invalidated':0,'pending':0}
    pair_stats[p]['alerts'] += 1
    o = a.get('outcome','pending')
    if   o == 'win_tp1':    pair_stats[p]['wins']        += 1
    elif o == 'loss':       pair_stats[p]['losses']      += 1
    elif o == 'invalidated':pair_stats[p]['invalidated'] += 1
    else:                   pair_stats[p]['pending']     += 1

print("  Calling Gemini for weekly analysis...")
analysis = build_weekly_analysis(
    weekly_alerts, wins, losses, invalidated_count, pending, win_rate, pair_stats
)

if not analysis:
    send_status_email(
        f"Found {len(weekly_alerts)} alerts but Gemini analysis failed. "
        f"Raw: {wins}W / {losses}L / {invalidated_count} Invalidated / {pending} Pending."
    )
    exit(0)

print("  Building Excel journal...")
excel_bytes = build_excel_journal(weekly_alerts, analysis)

print("  Building email...")
html = build_weekly_email_html(
    analysis, weekly_alerts, wins, losses, invalidated_count, pending, win_rate
)

send_weekly_email(html, excel_bytes, len(weekly_alerts),
                  wins, losses, invalidated_count, win_rate)
print("  Done.")
