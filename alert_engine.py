import yfinance as yf
import pandas as pd
import json, os, smtplib, requests, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage          # FIX ④ — CID image attachments
import numpy as np
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed  # FIX ⑤ — parallel fetch

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY    = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_PASS    = os.environ["GMAIL_APP_PASSWORD"]
ALERT_EMAIL   = os.environ["ALERT_EMAIL"]

# FIX ③ — identifies forex pairs for liquidity sweep exclusion
FOREX_PAIRS = {"EURUSD", "USDJPY", "NZDUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF"}

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

# ── Zone fatigue ──────────────────────────────────────────────────────────────
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

# ── Market hours ──────────────────────────────────────────────────────────────
# FIX ⑦ — blocks 00:00–08:00 IST every weekday + full weekend
def is_market_open():
    ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    wd, h = ist.weekday(), ist.hour   # Mon=0 … Fri=4, Sat=5, Sun=6

    if wd == 5:
        return False, "Saturday — market closed."
    if wd == 6:
        return False, "Sunday — market closed."
    if h < 8:
        return False, f"Quiet hours (midnight–08:00 IST) — {ist.strftime('%A %H:%M')} IST."
    if wd == 4 and h >= 23:
        return False, "Friday after 11:00 PM IST — market closing."

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

# ── Breakout detection — minimum score raised to 4/5 ─────────────────────────
# FIX ② — threshold changed from 3 to 4 (two places, clearly marked below)
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

        last   = len(df1) - 2
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
            score   = 0
            reasons = []

            if body_ok:
                score += 2
                reasons.append(
                    f"+2 Body closed {'above' if direction=='BULLISH' else 'below'} "
                    f"{round(level,5)}, candle {round(l_range/atr,1)}x ATR")
            else:
                reasons.append("+0 Body did not fully close beyond level")

            if 'Volume' in df1.columns:
                try:
                    vols    = df1['Volume'].values.flatten().astype(float)
                    avg_vol = float(np.nanmean(vols[-21:-1]))
                    if avg_vol > 0 and vols[last] >= 1.3 * avg_vol:
                        score += 1
                        reasons.append(f"+1 Volume {round(vols[last]/avg_vol,1)}x average (institutional)")
                    else:
                        reasons.append("+0 Volume below 1.3x average")
                except:
                    reasons.append("+0 Volume unavailable")
            else:
                reasons.append("+0 Volume not tracked for this pair (forex/commodity)")

            if touches >= 3:
                score += 1
                reasons.append(f"+1 Level tested {touches} times before (well-established)")
            else:
                reasons.append(f"+0 Level only {touches} prior touch(es) — needs 3+")

            aligned = (closes[-1] > ma50 if direction == "BULLISH" else closes[-1] < ma50)
            if aligned:
                score += 1
                reasons.append("+1 Break direction matches 50-period MA trend")
            else:
                reasons.append("+0 Counter-trend break — price on wrong side of 50-MA")

            return score, reasons

        # ── Bullish BOS ──
        if l_close > l_open:
            candidates = [(lvl,tc) for lvl,tc in sig_highs
                          if l_close > lvl and l_open <= lvl * 1.005]
            if candidates:
                level, touches = max(candidates, key=lambda x: x[0])
                score, reasons = score_breakout("BULLISH", level, touches, True)
                if score >= 4:   # ← FIX ② was 3
                    return {
                        "direction":       "BULLISH",
                        "timeframe":       "H1",
                        "broken_level":    round(level, 5),
                        "break_size_atr":  round(l_range/atr, 1),
                        "level_touches":   touches,
                        "bo_score":        score,
                        "bo_reasons":      reasons,
                        "retest_zone_top": round(level+buf, 5),
                        "retest_zone_bot": round(level-buf, 5),
                        "momentum_entry":  round(closes[-1], 5),
                        "retest_entry":    round(level+buf, 5),
                        "sl_momentum":     round(l_low-buf, 5),
                        "sl_retest":       round(level-buf*2, 5),
                        "description":     (f"Bullish BOS on H1: body closed "
                                            f"{round((l_close-level)/level*100,3)}% above {round(level,5)}")
                    }

        # ── Bearish BOS ──
        elif l_close < l_open:
            candidates = [(lvl,tc) for lvl,tc in sig_lows
                          if l_close < lvl and l_open >= lvl * 0.995]
            if candidates:
                level, touches = min(candidates, key=lambda x: x[0])
                score, reasons = score_breakout("BEARISH", level, touches, True)
                if score >= 4:   # ← FIX ② was 3
                    return {
                        "direction":       "BEARISH",
                        "timeframe":       "H1",
                        "broken_level":    round(level, 5),
                        "break_size_atr":  round(l_range/atr, 1),
                        "level_touches":   touches,
                        "bo_score":        score,
                        "bo_reasons":      reasons,
                        "retest_zone_top": round(level+buf, 5),
                        "retest_zone_bot": round(level-buf, 5),
                        "momentum_entry":  round(closes[-1], 5),
                        "retest_entry":    round(level-buf, 5),
                        "sl_momentum":     round(l_high+buf, 5),
                        "sl_retest":       round(level+buf*2, 5),
                        "description":     (f"Bearish BOS on H1: body closed "
                                            f"{round((level-l_close)/level*100,3)}% below {round(level,5)}")
                    }

        return None

    except Exception as e:
        print(f"    Breakout detection error: {e}")
        return None

# ── Parallel data fetch ───────────────────────────────────────────────────────
# FIX ⑤ — runs detect_zones_and_candles + detect_breakout in parallel for all pairs
def fetch_pair_data(pair_conf):
    name        = pair_conf["name"]
    symbol      = pair_conf["symbol"]
    min_touches = pair_conf.get("min_touches", 1)
    try:
        zones, current_price, df1, df2 = detect_zones_and_candles(symbol, min_touches)
        breakout = detect_breakout(df1) if df1 is not None else None
        return name, zones, current_price, df1, df2, breakout
    except Exception as e:
        print(f"    Fetch error {name}: {e}")
        return name, [], None, None, None, None

# ── Macro news ────────────────────────────────────────────────────────────────
def fetch_macro_news():
    try:
        url   = "https://api.rss2json.com/v1/api.json?rss_url=https://www.fxstreet.com/rss/news&count=5"
        r     = requests.get(url, timeout=10)
        items = r.json().get("items", [])
        return "\n".join([f"- {i['title']}" for i in items])
    except:
        return "Macro news unavailable."

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
# FIX ③ — forex pairs get H4/D1 structure alignment instead of liquidity sweep
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

    # Forex pairs: replace liquidity sweep with H4/D1 structure alignment
    # (liquidity sweep signals are unreliable on H1 forex data via yfinance)
    if pair in FOREX_PAIRS:
        liquidity_section = (
            "LIQUIDITY (2pts):     "
            "+1 Entry on correct side (Discount for long, Premium for short) | "
            "+1 H4 or D1 trend structure aligns with trade direction "
            "[IMPORTANT: Do NOT score or penalise for liquidity sweeps — "
            "sweep scoring is excluded for forex pairs in this system]"
        )
    else:
        liquidity_section = (
            "LIQUIDITY (2pts):     "
            "+1 Liquidity swept before zone | "
            "+1 Entry on correct side (Discount for long, Premium for short)"
        )

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
{liquidity_section}
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
  "chart_annotations": [{{"label": "short label", "price": 0.0, "status": "confirmed or missing"}}]
}}"""

# ── Gemini call ───────────────────────────────────────────────────────────────
# FIX ⑥ — timeout raised to 120s, plus one automatic retry on failure
def call_gemini(prompt, retries=1):
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    for attempt in range(retries + 1):
        try:
            r      = requests.post(url, json=body, timeout=120)
            result = r.json()
            if "candidates" not in result:
                raise ValueError(f"No candidates in response: {result}")
            raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            if raw.startswith("```"):
                raw = raw.split("\n",1)[1].rsplit("```",1)[0]
            return json.loads(raw), None
        except Exception as e:
            if attempt < retries:
                print(f"    Gemini attempt {attempt+1} failed ({e}), retrying in 3s...")
                time.sleep(3)
            else:
                return None, f"Gemini error after {retries+1} attempts: {str(e)}"

# ── Chart generation ──────────────────────────────────────────────────────────
# FIX ④ — returns raw PNG bytes (not base64); bytes are attached via MIMEImage
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
        fig.savefig(buf, format='png', dpi=72, bbox_inches='tight',
            facecolor='#131722', edgecolor='none')
        buf.seek(0)
        img_bytes = buf.read()   # raw bytes — NOT base64
        plt.close(fig)
        print(f"    Chart ok: {len(img_bytes)//1024}KB")
        return img_bytes
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
    sc     = "#27ae60" if score >= 5 else "#f39c12"  # 5=green, 4=amber
    reasons_html = "".join([
        f'<li style="font-size:11px;color:#aaa;margin-bottom:3px;">{r}</li>'
        for r in bo.get("bo_reasons",[])])
    return f"""<div style="background:#0d1117;border:2px solid {color};border-radius:10px;padding:14px 16px;margin-bottom:20px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <p style="color:{color};font-weight:bold;font-size:14px;margin:0;">{arrow} BREAKOUT ALERT — {d} BOS ({bo.get("timeframe","")})</p>
    <span style="background:{sc};color:white;font-weight:bold;font-size:13px;padding:3px 10px;border-radius:20px;">{score}/5</span>
  </div>
  <p style="color:#aaa;font-size:12px;margin:0 0 10px;">{bo.get("description","")}</p>
  <table style="width:100%;font-size:12px;margin-bottom:8px;color:white;">
    <tr><td style="color:#888;padding:3px 0;width:160px;">Broken Level</td><td style="font-weight:bold;">{bo.get("broken_level","")}</td></tr>
    <tr><td style="color:#888;padding:3px 0;">Break Candle</td><td style="font-weight:bold;">{bo.get("break_size_atr","")}x ATR — strong momentum</td></tr>
    <tr><td style="color:#888;padding:3px 0;">Retest Zone</td><td style="font-weight:bold;">{bo.get("retest_zone_bot","")} — {bo.get("retest_zone_top","")}</td></tr>
    <tr><td style="color:#888;padding:3px 0;">Momentum Entry</td><td style="color:{color};font-weight:bold;">{bo.get("momentum_entry","")} &nbsp;<span style="color:#555;font-size:10px;">aggressive — enter now</span></td></tr>
    <tr><td style="color:#888;padding:3px 0;">Retest Entry</td><td style="color:{color};font-weight:bold;">{bo.get("retest_entry","")} &nbsp;<span style="color:#555;font-size:10px;">preferred — wait for pullback to broken level</span></td></tr>
    <tr><td style="color:#888;padding:3px 0;">SL if momentum</td><td style="color:#ef5350;font-weight:bold;">{bo.get("sl_momentum","")}</td></tr>
    <tr><td style="color:#888;padding:3px 0;">SL if retest</td><td style="color:#ef5350;font-weight:bold;">{bo.get("sl_retest","")}</td></tr>
  </table>
  <ul style="list-style:none;padding:0;margin:0 0 6px;">{reasons_html}</ul>
  <p style="color:#856404;background:#fff3cd;padding:7px 10px;border-radius:6px;font-size:11px;margin:8px 0 0;">
    FAKEOUT RULE: If price closes back inside the broken level within 2 candles, the breakout has failed. Do not enter, or exit immediately.
  </p>
</div>"""

# ── Zone email HTML ───────────────────────────────────────────────────────────
# FIX ④ — uses cid: references; actual PNG bytes are attached separately via MIMEImage
def build_zone_email_html(data, pair, zone_level, zone_label, current_price,
                          has_chart1, has_chart2, breakout):
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

    # CID references — Gmail renders these when PNGs are MIME-attached with Content-ID headers
    chart1_html = (
        '<h3 style="color:#1a1a2e;font-size:13px;margin:20px 0 6px;">H1 CHART</h3>'
        '<img src="cid:chart_h1" style="width:100%;border-radius:8px;display:block;" />'
        if has_chart1 else
        '<p style="color:#aaa;font-size:12px;margin:20px 0 6px;">H1 chart unavailable.</p>'
    )
    chart2_html = (
        '<h3 style="color:#1a1a2e;font-size:13px;margin:16px 0 6px;">M15 CHART</h3>'
        '<img src="cid:chart_m15" style="width:100%;border-radius:8px;display:block;" />'
        if has_chart2 else
        '<p style="color:#aaa;font-size:12px;margin:16px 0 6px;">M15 chart unavailable.</p>'
    )

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
def build_breakout_only_email_html(bo, pair, current_price):
    ist_time = (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M")
    utc_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    bo_block = build_breakout_html_block(bo)
    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:16px;margin:0;">
<div style="max-width:620px;margin:auto;background:white;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">
  <div style="background:#1a1a2e;padding:18px 24px;">
    <h2 style="color:white;margin:0;font-size:17px;">BREAKOUT ALERT: {pair}</h2>
    <p style="color:#8899bb;margin:5px 0 0;font-size:12px;">{utc_time} UTC | {ist_time} IST &nbsp;|&nbsp; Now: {current_price}</p>
  </div>
  <div style="padding:20px 24px;">
    {bo_block}
  </div>
</div>
</body>
</html>"""

# ── Log alert ─────────────────────────────────────────────────────────────────
def log_alert(pair, zone_level, zone_label, current_price, data, alert_type="zone"):
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
        "outcome":          "pending",
        "outcome_price":    None,
        "outcome_checked_at": None
    })
    save_alert_log()

# ── Send email ────────────────────────────────────────────────────────────────
# FIX ④ — MIMEMultipart("related") + MIMEImage attachments for CID chart rendering
def send_email(subject, html_body, chart_images=None):
    """
    chart_images: dict mapping Content-ID name → PNG bytes
    e.g. {"chart_h1": bytes, "chart_m15": bytes}
    Gmail renders <img src="cid:chart_h1"> correctly when the image
    is attached as MIMEImage with a matching Content-ID header.
    """
    recipients = config["account"].get("alert_emails", [ALERT_EMAIL])
    for recipient in recipients:
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_ADDRESS
        msg["To"]      = recipient
        msg.attach(MIMEText(html_body, "html"))

        if chart_images:
            for cid, img_bytes in chart_images.items():
                img = MIMEImage(img_bytes, _subtype="png")
                img.add_header("Content-ID", f"<{cid}>")
                img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
                msg.attach(img)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_PASS)
            server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
        print(f"    Sent to {recipient}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
print(f"Alert engine started {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

market_open, market_status = is_market_open()
print(f"  Market: {market_status}")
if not market_open:
    print("  Exiting — outside trading hours.")
    save_alert_log()
    exit(0)

macro_news   = fetch_macro_news()
alerts_fired = 0

# ── Phase 1: Parallel data fetch for all pairs ────────────────────────────────
# FIX ⑤ — all 7 pairs download simultaneously; cuts ~21s serial wait to ~3s
print(f"  Fetching data for {len(config['pairs'])} pairs in parallel...")
all_pair_data = {}
with ThreadPoolExecutor(max_workers=len(config["pairs"])) as executor:
    futures = {executor.submit(fetch_pair_data, pc): pc["name"] for pc in config["pairs"]}
    for future in as_completed(futures):
        name = futures[future]
        try:
            _, zones, current_price, df1, df2, breakout = future.result()
            all_pair_data[name] = (zones, current_price, df1, df2, breakout)
            status = f"{len(zones)} zones, price={round(current_price,5)}" if current_price else "no data"
            print(f"    {name}: {status}")
        except Exception as e:
            print(f"    {name}: fetch failed — {e}")
            all_pair_data[name] = ([], None, None, None, None)

# ── Phase 2: Analyse and fire alerts (sequential — safe for shared state) ─────
for pair_conf in config["pairs"]:
    name     = pair_conf["name"]
    prox     = pair_conf["proximity_pct"]
    min_conf = pair_conf.get("min_confidence", 5)

    zones, current_price, df1, df2, breakout = all_pair_data.get(
        name, ([], None, None, None, None))

    if current_price is None:
        print(f"  {name}: no data, skipping.")
        continue

    print(f"  Processing {name}...")

    # Suppress breakout repeat if price hasn't moved enough since last BO alert
    if breakout:
        if not should_alert_breakout(name, breakout["broken_level"], current_price, prox):
            print(f"    Breakout suppressed — price hasn't moved enough since last BO alert.")
            breakout = None

    # ── Zone check ────────────────────────────────────────────────────────────
    zone_alert_fired = False
    for zone_level, touches in zones:
        dist_pct = abs(current_price - zone_level) / zone_level * 100
        if dist_pct > prox:
            continue

        if not should_alert_zone(name, zone_level, current_price, prox):
            print(f"    {name} @ {zone_level:.5f} — suppressed (price not moved enough).")
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

        chart1_bytes = generate_chart(df1, f"{name} — H1",  levels, data)
        chart2_bytes = generate_chart(df2, f"{name} — M15", levels, data)

        # Build CID image dict — only include charts that rendered successfully
        chart_images = {}
        if chart1_bytes:
            chart_images["chart_h1"]  = chart1_bytes
        if chart2_bytes:
            chart_images["chart_m15"] = chart2_bytes

        html = build_zone_email_html(
            data, name, round(zone_level,5), zone_label, round(current_price,5),
            has_chart1=bool(chart1_bytes), has_chart2=bool(chart2_bytes),
            breakout=breakout
        )
        subject = (f"[{score}/10] {name} | {zone_label} | "
                   f"{round(zone_level,5)} | {datetime.utcnow().strftime('%H:%M')} UTC")
        if breakout:
            subject = (f"[SMC {score}/10 + BO {breakout['bo_score']}/5] {name} | "
                       f"{zone_label} | {datetime.utcnow().strftime('%H:%M')} UTC")

        send_email(subject, html, chart_images=chart_images if chart_images else None)
        log_alert(name, round(zone_level,5), zone_label, round(current_price,5), data, "zone")
        record_zone_alert(name, zone_level, current_price)
        if breakout:
            record_breakout_alert(name, breakout["broken_level"], current_price)

        alerts_fired += 1
        zone_alert_fired = True
        print(f"    Sent: {name} SMC[{score}/10]" + (f" + BO[{breakout['bo_score']}/5]" if breakout else ""))
        break  # one zone email per pair per run

    # ── Standalone breakout email (only if no zone alert fired) ──────────────
    if breakout and not zone_alert_fired:
        html    = build_breakout_only_email_html(breakout, name, round(current_price,5))
        subject = (f"[BO {breakout['bo_score']}/5] {name} | {breakout['direction']} Breakout | "
                   f"{round(breakout['broken_level'],5)} | {datetime.utcnow().strftime('%H:%M')} UTC")
        send_email(subject, html)  # No charts on standalone breakout emails
        log_alert(name, breakout["broken_level"], f"{breakout['direction']} Breakout",
                  round(current_price,5), None, "breakout")
        record_breakout_alert(name, breakout["broken_level"], current_price)
        alerts_fired += 1
        print(f"    Sent standalone breakout: {name} [{breakout['bo_score']}/5]")

save_alert_log()
print(f"Log saved: {len(alert_log)} total entries.")
print(f"Scan complete. {alerts_fired} alert(s) fired.")
