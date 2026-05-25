import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import smtplib
import requests
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import base64
from io import BytesIO
import xml.etree.ElementTree as ET
import smc_detector

with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "dummy")
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "avinash.somjani23@gmail.com")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD", "dummy")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def get_day_id_ist(now=None):
    """Trading-day identifier. Day starts at 09:00 IST. Mirrors Phase 1.
    Pre-09:00 IST belongs to the previous day's slate."""
    if now is None:
        now = get_ist_now()
    if now.hour >= 9:
        return now.strftime("%Y-%m-%d")
    return (now - timedelta(days=1)).strftime("%Y-%m-%d")


def score_to_int(score):
    """Integer floor of score. Stabilises 7.4 and 7.9 to the same bucket (7).
    Used to decide whether a re-emerging zone has materially changed."""
    try:
        return int(float(score))
    except Exception:
        return -1


# Hysteresis thresholds for re-emailing the same zone. Asymmetric on purpose:
# losing a confluence is a stronger reason to re-alert than gaining one.
# A zone wobbling 6.4 <-> 7.0 doesn't cross either threshold and stays silent.
SCORE_REEMAIL_UP_THRESHOLD   = 0.7   # current >= prior + 0.7 -> re-email up
SCORE_REEMAIL_DOWN_THRESHOLD = 0.5   # current <= prior - 0.5 -> re-email down


def _ob_in_killzone_label(ob, pair_conf):
    """Render a human-readable label for whether the OB candle landed in a
    configured killzone window. Used in the live alert email so the trader
    can eyeball the SMC alignment hypothesis as data accumulates.

    Returns:
      - "in killzone" (green) if ob_timestamp hour overlaps any killzones_utc window
      - "outside killzone" (amber) if it doesn't
      - "unknown" if either input is missing
    """
    try:
        ob_ts_iso = (ob or {}).get("ob_timestamp")
        if not ob_ts_iso:
            return "<span style='color:#888;'>unknown</span>"
        import pandas as pd
        ts = pd.Timestamp(ob_ts_iso)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        hour_start = ts.hour * 60
        hour_end   = hour_start + 60
        windows = (pair_conf or {}).get("killzones_utc") or []
        in_kz = False
        for w in windows:
            if not isinstance(w, (list, tuple)) or len(w) != 2:
                continue
            try:
                sh, sm = (int(x) for x in str(w[0]).split(":"))
                eh, em = (int(x) for x in str(w[1]).split(":"))
            except (ValueError, AttributeError):
                continue
            start_min = sh * 60 + sm
            end_min   = eh * 60 + em
            if hour_start < end_min and hour_end > start_min:
                in_kz = True
                break
        if in_kz:
            return "<b style='color:#27ae60;'>in killzone</b>"
        return "<b style='color:#e67e22;'>outside killzone</b>"
    except Exception:
        return "<span style='color:#888;'>unknown</span>"


def hysteresis_should_reemail(current_score, prior_score):
    """
    Decide whether a same-zone re-sighting deserves an update email.

    Returns 'up' if score crossed the upward threshold (confluence gained),
    'down' if it crossed the downward threshold (confluence lost), or
    None if inside the dead band (silent).
    """
    if prior_score is None:
        return 'up'  # first sighting — always email (caller treats as new)
    delta = current_score - prior_score
    if delta >= SCORE_REEMAIL_UP_THRESHOLD:
        return 'up'
    if delta <= -SCORE_REEMAIL_DOWN_THRESHOLD:
        return 'down'
    return None


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    """Atomic save: write to temp file then rename."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    
def format_score_driver_line(prior_raw, current_raw, prior_breakdown, current_breakdown):
    """
    One-line summary explaining what drove the score change.
    Compares prior vs current breakdown component-by-component, surfaces
    the top movers (largest absolute deltas).

    Falls back gracefully if prior breakdown is missing (older state files
    written before this field was tracked).
    """
    delta_total = round(current_raw - prior_raw, 2)
    sign = "+" if delta_total >= 0 else ""
    header = f"Score {prior_raw:.1f} → {current_raw:.1f} ({sign}{delta_total:+.1f})."

    if not isinstance(prior_breakdown, dict) or not isinstance(current_breakdown, dict):
        return f"{header} Drivers: not available (first update post-deploy)."

    component_deltas = []
    keys = set(prior_breakdown.keys()) | set(current_breakdown.keys())
    for k in keys:
        try:
            p = float(prior_breakdown.get(k, 0.0))
            c = float(current_breakdown.get(k, 0.0))
        except (TypeError, ValueError):
            continue
        d = round(c - p, 2)
        if abs(d) >= 0.05:  # ignore noise-level shifts
            component_deltas.append((k, d))

    if not component_deltas:
        return f"{header} Drivers: rounding-level adjustments only."

    # Sort by absolute delta descending; show top 2.
    component_deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    top = component_deltas[:2]
    parts = [f"{name} {d:+.2f}" for name, d in top]
    return f"{header} Drivers: {', '.join(parts)}."
    
def load_slate_as_pair_map(path="active_obs.json"):
    """
    Backward-compatible reader. Phase 1 daily-slate schema is:
      {"slate_date": "...", "slate_started_iso": "...",
       "pairs": {"EURUSD": {"next_id_counter": N, "zones": [...]}, ...}}

    Legacy flat schema:
      {"EURUSD": [...], "USDJPY": [...], ...}

    Phase 2's runtime expects flat {pair_name: [zones]}. This adapter
    detects the schema version and returns the flat map regardless.

    For new schema: only zones with status="active" are returned.
    Dropped zones are excluded — Phase 2 must never alert on a dropped zone.
    """
    raw = load_json(path, {})
    if not isinstance(raw, dict):
        return {}
    # New schema detection: presence of "pairs" key with dict value
    if "pairs" in raw and isinstance(raw.get("pairs"), dict):
        flat = {}
        for pair_name, pblock in raw["pairs"].items():
            if not isinstance(pblock, dict):
                continue
            zones = pblock.get("zones", [])
            if not isinstance(zones, list):
                continue
            # Filter to active only — dropped zones must not feed Phase 2
            flat[pair_name] = [z for z in zones if z.get("status") == "active"]
        return flat
    # Legacy schema: assume already flat
    return raw
    
def append_scan_log(entry):
    """Append one JSON line per pair per scan to phase2_scan_log.jsonl."""
    try:
        with open("phase2_scan_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"  [SCANLOG ERR] {e}")


SCAN_LOG_RETENTION_DAYS = 14


def rotate_scan_log(ist_now):
    """Trim phase2_scan_log.jsonl to entries within retention window.

    Called once at P2 start. Filters lines by their 'ts_ist' field; entries
    older than SCAN_LOG_RETENTION_DAYS are dropped. Unparseable lines are
    kept (we don't silently delete data we can't read).
    """
    path = "phase2_scan_log.jsonl"
    if not os.path.exists(path):
        return
    try:
        cutoff = ist_now - timedelta(days=SCAN_LOG_RETENTION_DAYS)
        kept = []
        dropped = 0
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts_str = entry.get("ts_ist")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str)
                        if ts < cutoff:
                            dropped += 1
                            continue
                except Exception:
                    pass  # unparseable — keep it
                kept.append(line)
        if dropped == 0:
            return
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for line in kept:
                f.write(line + "\n")
        os.replace(tmp, path)
        print(f"  [SCANLOG ROT] Dropped {dropped} entries older than {SCAN_LOG_RETENTION_DAYS}d.")
    except Exception as e:
        print(f"  [SCANLOG ROT ERR] {e}")

def clean_df(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def pip_size(pair_conf):
    dp = pair_conf.get("decimal_places", 5)
    if dp == 5:
        return 0.0001
    if dp == 3:
        return 0.01
    return 1.0


def pip_unit_label(pair_conf):
    return "pips" if pair_conf.get("pair_type") == "forex" else "points"


def atr_distance_label(distance, atr, tf_label="M15"):
    if atr is None or atr == 0:
        return ""
    ratio = distance / atr
    if ratio < 0.5:
        return f"Price is at the zone now (under 0.5x {tf_label} ATR)."
    if ratio < 1.5:
        return f"Price is approx one candle away at {tf_label} TF."
    if ratio < 3:
        return f"Price is {ratio:.1f}x {tf_label} ATR away (2-3 candles)."
    return f"Price is {ratio:.1f}x {tf_label} ATR away — still distant."


# ---------------------------------------------------------------------------
# Fetch with retry + staleness check (B3, B5)
# ---------------------------------------------------------------------------

# Staleness thresholds in hours. If latest candle is older than this, data is
# considered stale and fetch is skipped (after retries).
STALENESS_HOURS = {
    "1h": 2.0,
    "15m": 0.75,
    "5m": 0.30
}


def _check_staleness(df, interval):
    """Return (is_stale, age_hours). age_hours is None if df is empty."""
    if df is None or df.empty:
        return True, None
    try:
        last_ts = df.index[-1]
        if hasattr(last_ts, 'tz_convert') and last_ts.tzinfo is not None:
            last_utc = last_ts.tz_convert('UTC').tz_localize(None).to_pydatetime()
        elif hasattr(last_ts, 'to_pydatetime'):
            last_utc = last_ts.to_pydatetime()
            if last_utc.tzinfo is not None:
                last_utc = last_utc.replace(tzinfo=None)
        else:
            last_utc = last_ts

        age_hours = (datetime.utcnow() - last_utc).total_seconds() / 3600
        max_age = STALENESS_HOURS.get(interval, 2.0)
        return age_hours > max_age, age_hours
    except Exception:
        return False, None  # Can't determine — don't block


def fetch_with_retry(symbol, period, interval, retries=2):
    """
    Fetch yfinance data with retry + staleness check.
    Returns cleaned df on success, None on persistent failure/staleness.
    Logs staleness skips to yfinance_stale_log.json for weekly review.
    """
    last_error = None
    last_age = None

    for attempt in range(retries + 1):
        try:
            raw = yf.download(symbol, period=period, interval=interval,
                              progress=False, timeout=20)
            df = clean_df(raw)
            is_stale, age_hours = _check_staleness(df, interval)

            if df is not None and not is_stale:
                return df

            last_age = age_hours
            if is_stale:
                last_error = f"stale data (age {age_hours:.2f}h, limit {STALENESS_HOURS.get(interval, 2.0)}h)"
            else:
                last_error = "empty dataframe"
        except Exception as e:
            last_error = str(e)

        if attempt < retries:
            wait = 2 ** attempt  # 1s, 2s
            print(f"  [RETRY {attempt + 1}/{retries}] {symbol} {interval}: {last_error}. Waiting {wait}s.")
            time.sleep(wait)

    # All retries exhausted — log and return None
    print(f"  [SKIP] {symbol} {interval}: {last_error}")
    _log_stale_skip(symbol, interval, last_error, last_age)
    return None


def _log_stale_skip(symbol, interval, reason, age_hours):
    """Log fetch failures for weekly review."""
    try:
        log = load_json("yfinance_stale_log.json", [])
        log.append({
            "ts": get_ist_now().isoformat(),
            "symbol": symbol,
            "interval": interval,
            "reason": reason,
            "age_hours": round(age_hours, 2) if age_hours is not None else None
        })
        # Keep only last 200 entries
        log = log[-200:]
        save_json("yfinance_stale_log.json", log)
    except Exception as e:
        print(f"  [LOG ERR] stale log: {e}")


def _log_gemini_failure(pair, reason):
    """Log Gemini API failures for weekly review (B1)."""
    try:
        log = load_json("gemini_failure_log.json", [])
        log.append({
            "ts": get_ist_now().isoformat(),
            "pair": pair,
            "reason": reason
        })
        log = log[-200:]
        save_json("gemini_failure_log.json", log)
    except Exception as e:
        print(f"  [LOG ERR] gemini log: {e}")


def _log_chart_failure(pair, chart_type):
    """Log chart render failures for weekly review (B8)."""
    try:
        log = load_json("chart_failure_log.json", [])
        log.append({
            "ts": get_ist_now().isoformat(),
            "pair": pair,
            "chart_type": chart_type
        })
        log = log[-200:]
        save_json("chart_failure_log.json", log)
    except Exception as e:
        print(f"  [LOG ERR] chart log: {e}")


def _log_smtp_failure(recipient, reason):
    """Log SMTP send failures so heartbeat can surface them. Without this,
    a bad app password or transient SMTP outage produces a silent system."""
    try:
        log = load_json("smtp_failure_log.json", [])
        log.append({
            "ts": get_ist_now().isoformat(),
            "recipient": recipient,
            "reason": reason
        })
        log = log[-200:]
        save_json("smtp_failure_log.json", log)
    except Exception as e:
        print(f"  [LOG ERR] smtp log: {e}")


# ---------------------------------------------------------------------------
# Macro news + Gemini
# ---------------------------------------------------------------------------

def fetch_macro_news(pair_name):
    try:
        r = requests.get("https://www.forexlive.com/feed/news", timeout=10)
        headlines = [
            f"- {item.find('title').text}"
            for item in ET.fromstring(r.content).findall('.//item')[:10]
        ]
        return "\n".join(headlines)
    except Exception:
        return "Could not fetch latest news."


def call_gemini_flash(pair, bias, news_headlines):
    prompt = f"""
    You are a Macro Context Writer. DO NOT analyze the chart. DO NOT score the trade.
    Your only job is to summarize macro risk for the trader to read.
    TRADE DETAILS: Pair: {pair} | Direction: {bias}
    RECENT NEWS: {news_headlines}

    TASK:
    Identify any Tier-1 economic events (e.g., CPI, NFP) affecting {pair} and
    summarize them in plain language. The trader will decide what to do with this.

    OUTPUT FORMAT (Strict JSON):
    {{
        "macro_summary": "Exactly 2 concise sentences summarizing the macro risk specific to {pair}. If no Tier-1 events, say so explicitly."
    }}
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.15,
            "thinkingConfig": {"thinkingBudget": 0}
        }
    }
    last_err = "unknown"
    for attempt in range(3):
        try:
            r = requests.post(url, json=body, timeout=20).json()
            if "candidates" in r:
                return json.loads(r["candidates"][0]["content"]["parts"][0]["text"].strip())
            last_err = f"no candidates field, attempt {attempt + 1}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:80]}"
            time.sleep(3)

    # All attempts failed — log and surface the failure HONESTLY to the
    # trader. Previously this returned a fake "safe" summary that read like
    # "no Tier-1 events" — misleading. macro_summary=None + macro_unavailable
    # lets the email render a distinct "unavailable" banner.
    _log_gemini_failure(pair, last_err)
    return {"macro_summary": None, "macro_unavailable": True}


# ---------------------------------------------------------------------------
# Chart generators — H1 context + M15 approach
# ---------------------------------------------------------------------------

def _base_canvas():
    fig, ax = plt.subplots(1, 1, figsize=(12, 5.0), facecolor='#131722')
    ax.set_facecolor('#131722')
    for s in ax.spines.values():
        s.set_color('#2a2a3e')
    return fig, ax


def _draw_candles(ax, df_plot):
    for i, row in df_plot.iterrows():
        o, h, l, c = float(row['Open']), float(row['High']), float(row['Low']), float(row['Close'])
        if any(np.isnan(v) for v in [o, h, l, c]):
            continue
        col = '#26a69a' if c >= o else '#ef5350'
        ax.plot([i, i], [l, h], color=col, linewidth=1.2, zorder=2)
        body = abs(c - o) or (h - l) * 0.02
        ax.add_patch(patches.Rectangle(
            (i - 0.4, min(o, c)), 0.8, body,
            facecolor=col, linewidth=0, alpha=0.9, zorder=3
        ))


def _fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#131722')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def build_m15_chart_ob(h1_ob, levels):
    """
    When a nested M15 OB was found inside the H1 OB, the M15 chart and the
    email zone label must mark the M15 OB (that's where we actually enter),
    NOT the H1 OB. This builds a shallow overlay dict: it inherits H1's
    metadata (BOS line, FVG, sweep, direction) but overrides the geometry
    fields (proximal/distal/high/low/ob_timestamp) to the M15 OB.

    Returns the overlay dict, or the original H1 ob if no M15 OB nested.
    Rule per project owner: BOS/CHoCH stays H1; only geometry switches.
    """
    m15_ob = levels.get("m15_ob") if isinstance(levels, dict) else None
    if not m15_ob:
        return h1_ob
    direction = h1_ob.get("direction")  # bullish / bearish
    m15_high = float(m15_ob["high"])
    m15_low  = float(m15_ob["low"])
    # proximal_line is set the same way Phase 1 sets it on the H1 OB:
    # bullish -> high, bearish -> low. See smc_radar.py:866.
    if direction == "bullish":
        prox, dist = m15_high, m15_low
    else:
        prox, dist = m15_low, m15_high
    overlay = dict(h1_ob)
    overlay["proximal_line"] = prox
    overlay["distal_line"]   = dist
    overlay["high"]          = m15_high
    overlay["low"]           = m15_low
    overlay["ob_timestamp"]  = m15_ob["ts"]
    return overlay


def generate_h1_chart(df_h1, ob, pair_conf, title, levels=None, dealing_range=None):
    try:
        dp = pair_conf.get("decimal_places", 5)
        df_plot = df_h1.dropna(subset=['Open', 'High', 'Low', 'Close']).tail(80).copy().reset_index(drop=True)
        n = len(df_plot)
        if n < 5:
            return None

        fig, ax = _base_canvas()
        _draw_candles(ax, df_plot)

        tail_n = 80
        full_n = len(df_h1)
        window_start = max(0, full_n - tail_n)

        proximal = float(ob.get('proximal_line', 0))
        distal = float(ob.get('distal_line', 0))
        zone_hi, zone_lo = max(proximal, distal), min(proximal, distal)

        # --- Zone band ---
        if zone_hi > 0 and zone_lo > 0:
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                facecolor='#9b59b6', alpha=0.15, zorder=1
            ))
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                fill=False, edgecolor='#bb8fce', linestyle=':', linewidth=1.5, zorder=2
            ))

        # --- BOS / CHoCH horizontal line ---
        # Palette:
        #   BOS              -> yellow #f1c40f
        #   Major CHoCH      -> pink   #e91e63
        #   Minor CHoCH      -> purple #9c27b0  (internal lb-3 break after wall touch — weakening flag)
        bos_price = float(ob.get('bos_swing_price', 0))
        bos_tag = ob.get('bos_tag', 'BOS')
        bos_tier = ob.get('bos_tier', 'Major')
        if bos_tag == 'BOS':
            bos_color = '#f1c40f'
        elif bos_tier == 'Minor':
            bos_color = '#9c27b0'
        else:
            bos_color = '#e91e63'
        if bos_price > 0:
            ax.axhline(y=bos_price, color=bos_color, linewidth=0.8, linestyle='--', alpha=0.7, zorder=2)

        # --- Liquidity sweep wick highlight (H1 only) ---
        # Draws a dotted rectangle around the wick portion of the sweep candle.
        # Drawn only if sweep occurred on H1 timeframe.
        sweep_tf = ob.get('sweep_tf')
        sweep_ts = ob.get('sweep_timestamp')
        if sweep_tf == 'H1' and sweep_ts:
            sw_abs_idx, sw_found = smc_detector.locate_ob_candle_idx(df_h1, sweep_ts)
            if sw_found and sw_abs_idx >= window_start:
                sw_local = sw_abs_idx - window_start
                if 0 <= sw_local < n:
                    sw_o = float(df_plot['Open'].iloc[sw_local])
                    sw_h = float(df_plot['High'].iloc[sw_local])
                    sw_l = float(df_plot['Low'].iloc[sw_local])
                    sw_c = float(df_plot['Close'].iloc[sw_local])
                    body_top = max(sw_o, sw_c)
                    body_bot = min(sw_o, sw_c)
                    # LONG sweep -> wick pierces below: lower wick
                    # SHORT sweep -> wick pierces above: upper wick
                    if ob.get('direction') == 'bullish':
                        wick_lo, wick_hi = sw_l, body_bot
                    else:
                        wick_lo, wick_hi = body_top, sw_h
                    if wick_hi > wick_lo:
                        ax.add_patch(patches.Rectangle(
                            (sw_local - 0.45, wick_lo), 0.9, wick_hi - wick_lo,
                            fill=False, edgecolor='#00e5ff', linestyle=':',
                            linewidth=1.5, zorder=5
                        ))

        # --- FVG: outline middle (displacement) candle only, slightly wider for mitigation visibility ---
        # Cross-phase safety: resolve c1 candle position via timestamp (c1_idx
        # from Phase 1 is NOT portable to Phase 2's fresh dataframe).
        fvg = ob.get('fvg', {}) or {}
        if fvg.get('exists'):
            ft, fb = float(fvg.get('fvg_top', 0)), float(fvg.get('fvg_bottom', 0))
            c1_ts = fvg.get('c1_timestamp')
            c1_resolved = None
            if c1_ts:
                idx_c1, found_c1 = smc_detector.locate_ob_candle_idx(df_h1, c1_ts)
                if found_c1:
                    c1_resolved = idx_c1
            if c1_resolved is None:
                # Legacy fallback (only valid same-run; Phase 2 with no timestamp
                # means the ob predates this fix — outline likely off but harmless).
                c1_idx_legacy = fvg.get('c1_idx')
                if c1_idx_legacy is not None:
                    try:
                        c1_resolved = int(c1_idx_legacy)
                    except Exception:
                        c1_resolved = None

            if ft > 0 and fb > 0 and c1_resolved is not None:
                mid_abs = c1_resolved + 1  # displacement candle (middle of c1-c2-c3)
                mid_local = mid_abs - window_start
                if 0 <= mid_local < n:
                    mit = fvg.get('mitigation', 'pristine')
                    if mit == 'partial':
                        # Amber — partial mitigation (caution).
                        face_col, edge_col = '#f4d03f', '#f1c40f'
                    else:
                        face_col, edge_col = '#27ae60', '#2ecc71'
                    fvg_x_start = mid_local - 0.6
                    fvg_width = 1.8 + 1.2  # 1.8 candle widths + extra right-side mitigation visibility
                    ax.add_patch(patches.Rectangle(
                        (fvg_x_start, fb), fvg_width, ft - fb,
                        facecolor=face_col, alpha=0.15, zorder=1
                    ))
                    ax.add_patch(patches.Rectangle(
                        (fvg_x_start, fb), fvg_width, ft - fb,
                        fill=False, edgecolor=edge_col, linestyle='--', linewidth=1.0, zorder=2
                    ))
        # --- Dealing range band + equilibrium ---
        dr_eq = None
        if dealing_range and dealing_range.get('valid'):
            dr_hi = float(dealing_range['range_high'])
            dr_lo = float(dealing_range['range_low'])
            dr_eq = float(dealing_range['equilibrium'])
            y_min_candle = float(df_plot['Low'].min())
            y_max_candle = float(df_plot['High'].max())
            candle_range = y_max_candle - y_min_candle
            if candle_range > 0 and (dr_hi - dr_lo) < candle_range * 3:
                ax.add_patch(patches.Rectangle(
                    (0, dr_lo), n + 5, dr_hi - dr_lo,
                    facecolor='#3498db', alpha=0.06, zorder=0
                ))
                ax.add_patch(patches.Rectangle(
                    (0, dr_lo), n + 5, dr_hi - dr_lo,
                    fill=False, edgecolor='#5dade2', linestyle='-.', linewidth=0.8, zorder=1
                ))
            ax.axhline(y=dr_eq, color='#5dade2', linewidth=0.9, linestyle='-.', alpha=0.6, zorder=2)

        # --- Entry / SL horizontal lines ---
        entry_p = 0
        sl_p = 0
        tp1_p = 0
        if levels and levels.get('valid', True):
            entry_p = float(levels.get('entry', 0))
            sl_p = float(levels.get('sl', 0))
            tp1_p = float(levels.get('tp1', 0))
            if entry_p > 0:
                ax.axhline(y=entry_p, color='#e67e22', linewidth=1.0, linestyle='--', alpha=0.8, zorder=3)
            if sl_p > 0:
                ax.axhline(y=sl_p, color='#e74c3c', linewidth=1.0, linestyle='--', alpha=0.8, zorder=3)

        # --- Current price line ---
        current = float(df_plot['Close'].iloc[-1])
        ax.axhline(y=current, color='#ffffff', linewidth=0.8, linestyle='-', alpha=0.5, zorder=2)

        # --- OB candle outline (white) ---
        ob_ts_iso = ob.get('ob_timestamp')
        if ob_ts_iso:
            abs_idx, found = smc_detector.locate_ob_candle_idx(df_h1, ob_ts_iso)
            if found and abs_idx >= window_start:
                local_idx = abs_idx - window_start
                if 0 <= local_idx < n:
                    ob_c_h = float(df_plot['High'].iloc[local_idx])
                    ob_c_l = float(df_plot['Low'].iloc[local_idx])
                    ax.add_patch(patches.Rectangle(
                        (local_idx - 0.5, ob_c_l), 1.0, ob_c_h - ob_c_l,
                        fill=False, edgecolor='#ffffff', linewidth=1.5, zorder=5
                    ))

        # --- BOS/CHoCH break candle outline (cyan or orange) ---
        br_start, br_end = smc_detector.compute_h1_break_candle_span(df_h1, ob, None)
        if br_start is not None and br_end is not None:
            for abs_i in range(br_start, br_end + 1):
                if abs_i < window_start:
                    continue
                local_i = abs_i - window_start
                if 0 <= local_i < n:
                    c_h = float(df_plot['High'].iloc[local_i])
                    c_l = float(df_plot['Low'].iloc[local_i])
                    ax.add_patch(patches.Rectangle(
                        (local_i - 0.5, c_l), 1.0, c_h - c_l,
                        fill=False, edgecolor=bos_color, linewidth=1.5, zorder=5
                    ))

        # --- Right-edge tags: ENTRY, SL, TP1, TP2 (numbers only, colour-matched) ---
        tp2_p = float(levels.get('tp2', 0)) if isinstance(levels, dict) else 0
        right_labels = []
        if entry_p > 0:
            right_labels.append((entry_p, f" {entry_p:.{dp}f}", '#e67e22'))
        if sl_p > 0:
            right_labels.append((sl_p, f" {sl_p:.{dp}f}", '#e74c3c'))
        if tp1_p > 0:
            right_labels.append((tp1_p, f" {tp1_p:.{dp}f}", '#27ae60'))
        if tp2_p > 0:
            right_labels.append((tp2_p, f" {tp2_p:.{dp}f}", '#1e8449'))
        right_stacked = smc_detector.stack_labels(right_labels, pair_conf)
        for adj_price, text, color in right_stacked:
            ax.text(n + 1, adj_price, text, color=color, fontsize=10, va='center',
                    fontweight='bold', zorder=5)

        # --- Left-edge tags: proximal, distal, BOS/CHoCH ---
        left_labels = []
        if zone_hi > 0:
            left_labels.append((proximal, f"{proximal:.{dp}f}", '#bb8fce'))
            left_labels.append((distal, f"{distal:.{dp}f}", '#bb8fce'))
        if bos_price > 0:
            left_labels.append((bos_price, f"{bos_price:.{dp}f}", bos_color))
        left_stacked = smc_detector.stack_labels(left_labels, pair_conf)
        for adj_price, text, color in left_stacked:
            ax.text(-1, adj_price, text, color=color, fontsize=8, va='center',
                    ha='left', fontweight='bold', zorder=5,
                    bbox=dict(facecolor='#131722', edgecolor='none', pad=1.5, alpha=0.75))

        # --- Mid-chart tags: current price, EQ ---
        mid_x = n / 2.0
        center_labels = []
        center_labels.append((current, f"{current:.{dp}f}", '#ffffff'))
        if dr_eq is not None:
            center_labels.append((dr_eq, f"{dr_eq:.{dp}f}", '#5dade2'))
        center_stacked = smc_detector.stack_labels(center_labels, pair_conf)
        for adj_price, text, color in center_stacked:
            ax.text(mid_x, adj_price, text, color=color, fontsize=8, va='center',
                    ha='center', fontweight='bold', zorder=5,
                    bbox=dict(facecolor='#131722', edgecolor='none', pad=1.5, alpha=0.75))

        # --- Y-axis range ---
        y_min, y_max = float(df_plot['Low'].min()), float(df_plot['High'].max())
        for val in [zone_lo, zone_hi]:
            if val > 0:
                y_min = min(y_min, val)
                y_max = max(y_max, val)
        if sl_p > 0:
            y_min = min(y_min, sl_p)
        if entry_p > 0:
            y_max = max(y_max, entry_p)
        pad = (y_max - y_min) * 0.08
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlim(-1, n + 14)
        ax.set_title(title, color='#dddddd', fontsize=11, pad=8, loc='left')
        ax.tick_params(colors='#888', labelsize=9)
        ax.yaxis.tick_right()
        ax.set_xticks([])
        plt.tight_layout(pad=0.5)
        return _fig_to_b64(fig)
    except Exception as e:
        print(f"H1 chart error: {e}")
        plt.close('all')
        return None
        
def generate_m15_chart(df_m15, title, levels, ob, pair_conf, fvg_data, sweep_price,
                       dealing_range=None):
    try:
        dp = pair_conf.get("decimal_places", 5)
        df_plot = df_m15.dropna(subset=['Open', 'High', 'Low', 'Close']).tail(60).copy().reset_index(drop=True)
        n = len(df_plot)
        if n < 5:
            return None

        fig, ax = _base_canvas()
        _draw_candles(ax, df_plot)

        tail_n = 60
        full_n = len(df_m15)
        window_start = max(0, full_n - tail_n)

        proximal = float(ob.get('proximal_line', 0))
        distal = float(ob.get('distal_line', 0))
        zone_hi, zone_lo = max(proximal, distal), min(proximal, distal)

        # --- Zone band ---
        if zone_hi > 0 and zone_lo > 0:
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                facecolor='#9b59b6', alpha=0.15, zorder=1
            ))
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                fill=False, edgecolor='#bb8fce', linestyle=':', linewidth=1.5, zorder=2
            ))

        # --- Dealing range EQ line only ---
        dr_eq = None
        if dealing_range and dealing_range.get('valid'):
            dr_eq = float(dealing_range['equilibrium'])
            ax.axhline(y=dr_eq, color='#5dade2', linewidth=0.9, linestyle='-.', alpha=0.6, zorder=2)

        # --- FVG: outline middle (displacement) candle only, slightly wider for mitigation visibility ---
        if fvg_data and fvg_data.get('exists'):
            ft, fb = float(fvg_data.get('fvg_top', 0)), float(fvg_data.get('fvg_bottom', 0))
            c1_idx = fvg_data.get('c1_idx')
            if ft > 0 and fb > 0 and c1_idx is not None:
                mid_abs = int(c1_idx) + 1
                mid_local = mid_abs - window_start
                if 0 <= mid_local < n:
                    fvg_x_start = mid_local - 0.6
                    fvg_width = 1.8 + 1.2
                    ax.add_patch(patches.Rectangle(
                        (fvg_x_start, fb), fvg_width, ft - fb,
                        facecolor='#27ae60', alpha=0.10, zorder=1
                    ))
                    ax.add_patch(patches.Rectangle(
                        (fvg_x_start, fb), fvg_width, ft - fb,
                        fill=False, edgecolor='#2ecc71', linestyle='--', linewidth=1.0, zorder=2
                    ))

        # --- Liquidity sweep wick highlight (M15 only) ---
        # Draws a dotted rectangle around the wick portion of the sweep candle.
        # Drawn only if sweep occurred on M15 timeframe.
        sweep_tf = ob.get('sweep_tf')
        sweep_ts = ob.get('sweep_timestamp')
        if sweep_tf == 'M15' and sweep_ts:
            sw_abs_idx, sw_found = smc_detector.locate_ob_candle_idx(df_m15, sweep_ts)
            if sw_found and sw_abs_idx >= window_start:
                sw_local = sw_abs_idx - window_start
                if 0 <= sw_local < n:
                    sw_o = float(df_plot['Open'].iloc[sw_local])
                    sw_h = float(df_plot['High'].iloc[sw_local])
                    sw_l = float(df_plot['Low'].iloc[sw_local])
                    sw_c = float(df_plot['Close'].iloc[sw_local])
                    body_top = max(sw_o, sw_c)
                    body_bot = min(sw_o, sw_c)
                    if ob.get('direction') == 'bullish':
                        wick_lo, wick_hi = sw_l, body_bot
                    else:
                        wick_lo, wick_hi = body_top, sw_h
                    if wick_hi > wick_lo:
                        ax.add_patch(patches.Rectangle(
                            (sw_local - 0.45, wick_lo), 0.9, wick_hi - wick_lo,
                            fill=False, edgecolor='#00e5ff', linestyle=':',
                            linewidth=1.5, zorder=5
                        ))

        # --- BOS / CHoCH horizontal line ---
        # Palette:
        #   BOS              -> yellow #f1c40f
        #   Major CHoCH      -> pink   #e91e63
        #   Minor CHoCH      -> purple #9c27b0  (internal lb-3 break after wall touch — weakening flag)
        bos_price = float(ob.get('bos_swing_price', 0))
        bos_tag = ob.get('bos_tag', 'BOS')
        bos_tier = ob.get('bos_tier', 'Major')
        if bos_tag == 'BOS':
            bos_color = '#f1c40f'
        elif bos_tier == 'Minor':
            bos_color = '#9c27b0'
        else:
            bos_color = '#e91e63'
        if bos_price > 0:
            ax.axhline(y=bos_price, color=bos_color, linewidth=0.8, linestyle='--', alpha=0.7, zorder=2)

        # --- Entry / SL horizontal lines ---
        # Dashed to match H1 chart; solid lines were visually competing
        # with TP lines and crowding the small M15 chart.
        entry_p = float(levels.get('entry', 0))
        sl_p = float(levels.get('sl', 0))
        if entry_p > 0:
            ax.axhline(y=entry_p, color='#e67e22', linestyle='--', linewidth=1.1, alpha=0.85, zorder=4)
        if sl_p > 0:
            ax.axhline(y=sl_p, color='#e74c3c', linestyle='--', linewidth=1.1, alpha=0.85, zorder=4)

        # --- Current price line ---
        current = float(df_plot['Close'].iloc[-1])
        ax.axhline(y=current, color='#ffffff', linewidth=0.8, linestyle='-', alpha=0.5, zorder=2)

        # --- OB candle outline (white) ---
        ob_ts_iso = ob.get('ob_timestamp')
        if ob_ts_iso:
            abs_idx, found = smc_detector.locate_ob_candle_idx(df_m15, ob_ts_iso)
            if found and abs_idx >= window_start:
                local_idx = abs_idx - window_start
                if 0 <= local_idx < n:
                    ob_c_h = float(df_plot['High'].iloc[local_idx])
                    ob_c_l = float(df_plot['Low'].iloc[local_idx])
                    ax.add_patch(patches.Rectangle(
                        (local_idx - 0.5, ob_c_l), 1.0, ob_c_h - ob_c_l,
                        fill=False, edgecolor='#ffffff', linewidth=1.5, zorder=5
                    ))

        # --- Right-edge tags: ENTRY, SL, TP1, TP2 (numbers only, colour-matched) ---
        tp1_p = float(levels.get('tp1', 0))
        tp2_p = float(levels.get('tp2', 0)) if isinstance(levels, dict) else 0
        right_labels = []
        if entry_p > 0:
            right_labels.append((entry_p, f" {entry_p:.{dp}f}", '#e67e22'))
        if sl_p > 0:
            right_labels.append((sl_p, f" {sl_p:.{dp}f}", '#e74c3c'))

        # --- Y-axis: candle range + SL + entry + zone. ---
        y_min, y_max = float(df_plot['Low'].min()), float(df_plot['High'].max())
        if sl_p > 0:
            y_min = min(y_min, sl_p)
        if entry_p > 0:
            y_max = max(y_max, entry_p)
            y_min = min(y_min, entry_p)
        y_min = min(y_min, zone_lo)
        y_max = max(y_max, zone_hi)
        pad = (y_max - y_min) * 0.08
        ax.set_ylim(y_min - pad, y_max + pad)

        # TP1: line if visible, arrow if outside
        if tp1_p > 0:
            y_lo, y_hi = ax.get_ylim()
            if tp1_p > y_hi:
                ax.text(n + 1, y_hi - pad * 0.3, f" \u2191 {tp1_p:.{dp}f}",
                        color='#27ae60', fontsize=10, va='top', fontweight='bold', zorder=5)
            elif tp1_p < y_lo:
                ax.text(n + 1, y_lo + pad * 0.3, f" \u2193 {tp1_p:.{dp}f}",
                        color='#27ae60', fontsize=10, va='bottom', fontweight='bold', zorder=5)
            else:
                ax.axhline(y=tp1_p, color='#2ecc71', linestyle='-', linewidth=1.5, alpha=0.95, zorder=4)
                right_labels.append((tp1_p, f" {tp1_p:.{dp}f}", '#2ecc71'))

        # TP2: line if visible, arrow if outside (same logic as TP1, darker green)
        if tp2_p > 0:
            y_lo, y_hi = ax.get_ylim()
            if tp2_p > y_hi:
                ax.text(n + 1, y_hi - pad * 0.6, f" \u2191 {tp2_p:.{dp}f}",
                        color='#1e8449', fontsize=10, va='top', fontweight='bold', zorder=5)
            elif tp2_p < y_lo:
                ax.text(n + 1, y_lo + pad * 0.6, f" \u2193 {tp2_p:.{dp}f}",
                        color='#1e8449', fontsize=10, va='bottom', fontweight='bold', zorder=5)
            else:
                ax.axhline(y=tp2_p, color='#27ae60', linestyle='-', linewidth=1.5, alpha=0.95, zorder=4)
                right_labels.append((tp2_p, f" {tp2_p:.{dp}f}", '#27ae60'))

        right_stacked = smc_detector.stack_labels(right_labels, pair_conf)
        for adj_price, text, color in right_stacked:
            ax.text(n + 1, adj_price, text, color=color, fontsize=10, va='center',
                    fontweight='bold', zorder=5)

        # --- Left-edge tags: proximal, distal, BOS/CHoCH ---
        left_labels = []
        if zone_hi > 0:
            left_labels.append((proximal, f"{proximal:.{dp}f}", '#bb8fce'))
            left_labels.append((distal, f"{distal:.{dp}f}", '#bb8fce'))
        if bos_price > 0:
            left_labels.append((bos_price, f"{bos_price:.{dp}f}", bos_color))
        left_stacked = smc_detector.stack_labels(left_labels, pair_conf)
        for adj_price, text, color in left_stacked:
            ax.text(-1, adj_price, text, color=color, fontsize=8, va='center',
                    ha='left', fontweight='bold', zorder=5,
                    bbox=dict(facecolor='#131722', edgecolor='none', pad=1.5, alpha=0.75))

        # --- Mid-chart tags: current price, EQ ---
        mid_x = n / 2.0
        center_labels = []
        center_labels.append((current, f"{current:.{dp}f}", '#ffffff'))
        if dr_eq is not None:
            center_labels.append((dr_eq, f"{dr_eq:.{dp}f}", '#5dade2'))
        center_stacked = smc_detector.stack_labels(center_labels, pair_conf)
        for adj_price, text, color in center_stacked:
            ax.text(mid_x, adj_price, text, color=color, fontsize=8, va='center',
                    ha='center', fontweight='bold', zorder=5,
                    bbox=dict(facecolor='#131722', edgecolor='none', pad=1.5, alpha=0.75))

        ax.set_xlim(-1, n + 14)
        ax.set_title(title, color='#dddddd', fontsize=11, pad=8, loc='left')
        ax.tick_params(colors='#888', labelsize=9)
        ax.yaxis.tick_right()
        ax.set_xticks([])
        plt.tight_layout(pad=0.5)
        return _fig_to_b64(fig)
    except Exception as e:
        print(f"M15 chart error: {e}")
        plt.close('all')
        return None


# ---------------------------------------------------------------------------
# Email assembly
# ---------------------------------------------------------------------------

def build_scorecard_html(rows, total, total_max=10.0):
    body = ""
    for label, score, max_score, status, expl in rows:
        if status == "info":
            # Display-only row (e.g. PD post-removal). No score column shown.
            body += f"""
            <tr>
              <td style="padding:5px 8px;color:#888;font-weight:bold;font-size:13px;white-space:nowrap;vertical-align:top;">&bull;</td>
              <td style="padding:5px 8px;color:#eee;font-size:12px;white-space:nowrap;vertical-align:top;">{label}</td>
              <td style="padding:5px 8px;color:#888;font-size:11px;font-family:monospace;white-space:nowrap;vertical-align:top;">info</td>
              <td style="padding:5px 8px;color:#bbb;font-size:12px;line-height:1.45;">{expl}</td>
            </tr>"""
            continue
        icon = {"ok": "&#10003;", "warn": "!", "fail": "&#10007;"}.get(status, "&bull;")
        color = {"ok": "#27ae60", "warn": "#e67e22", "fail": "#e74c3c"}.get(status, "#888")
        body += f"""
        <tr>
          <td style="padding:5px 8px;color:{color};font-weight:bold;font-size:13px;white-space:nowrap;vertical-align:top;">{icon}</td>
          <td style="padding:5px 8px;color:#eee;font-size:12px;white-space:nowrap;vertical-align:top;">{label}</td>
          <td style="padding:5px 8px;color:#aaa;font-size:11px;font-family:monospace;white-space:nowrap;vertical-align:top;">{score}/{max_score}</td>
          <td style="padding:5px 8px;color:#bbb;font-size:12px;line-height:1.45;">{expl}</td>
        </tr>"""

    return f"""
    <div style="margin-bottom:14px;">
      <div style="color:#aaa;font-size:11px;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase;">
        Confluence Scorecard &mdash; <span style="color:#eee;font-size:14px;font-weight:bold;">{total}/{total_max}</span>
      </div>
      <table style="width:100%;border-collapse:collapse;background:#1a1a2e;border-radius:6px;">
        <tbody>{body}</tbody>
      </table>
    </div>"""

def _chart_legend_html(bos_tag="BOS", bos_tier="Major"):
    """Colour-code legend rendered below each chart. Cosmetic only."""
    if bos_tag == 'BOS':
        bos_color, bos_label = '#f1c40f', 'BOS'
    elif bos_tier == 'Minor':
        bos_color, bos_label = '#9c27b0', 'Minor CHoCH'
    else:
        bos_color, bos_label = '#e91e63', 'Major CHoCH'
    items = [
        ('#bb8fce', 'Zone band (proximal/distal)'),
        ('#2ecc71', 'FVG pristine (displacement)'),
        ('#f1c40f', 'FVG partial (proximal touched)'),
        (bos_color, f'{bos_label} break candle / level'),
        ('#ffffff', 'OB candle / current price'),
        ('#e67e22', 'Entry'),
        ('#e74c3c', 'Stop loss'),
        ('#27ae60', 'TP1'),
        ('#1e8449', 'TP2'),
        ('#5dade2', 'Equilibrium (50% of range)'),
        ('#00e5ff', 'Liquidity sweep (wick highlight)'),
    ]
    rows = "".join(
        f'<span style="display:inline-block;margin:2px 10px 2px 0;font-size:11px;color:#bbb;">'
        f'<span style="display:inline-block;width:10px;height:10px;background:{c};'
        f'border-radius:2px;vertical-align:middle;margin-right:5px;"></span>{txt}</span>'
        for c, txt in items
    )
    return (
        f'<div style="margin:4px 0 12px 0;padding:8px 10px;background:#0d0d1a;'
        f'border-radius:4px;line-height:1.8;">{rows}</div>'
    )
def build_trade_email(data, pair, pair_conf, state_msg, scorecard_rows, total_score,
                      atr_label, distance_str, dollar_risk_str, scan_start_ist,
                      h1_chart_ok=True, m15_chart_ok=True):
    dp = pair_conf.get("decimal_places", 5)
    bias = data.get("bias", "-")
    sent_ist = get_ist_now().strftime('%H:%M IST')
    scan_ist = scan_start_ist.strftime('%H:%M IST')
    ist_time = f"Scanned {scan_ist} · Sent {sent_ist}"
    ob = data.get('ob', {})
    levels = data.get('levels', {})
    # Freshness display handled by scorecard row alone; no separate context line.
    bos_tag = ob.get('bos_tag', 'BOS')
    bos_tier = ob.get('bos_tier', 'Major')
    entry_model = pair_conf.get('entry_model', 'limit')
    if entry_model == "limit":
        action_word = "SELL LIMIT" if bias == "SHORT" else "BUY LIMIT"
        tp2_val = levels.get('tp2')
        tp2_html = f"TP2: {tp2_val:,.{dp}f} &nbsp;|&nbsp; " if tp2_val is not None else ""
        action_block = f"""
            <div style="background:#27ae60;padding:14px 18px;border-radius:10px;margin-bottom:14px;">
                <p style="color:white;font-size:15px;font-weight:bold;margin:0;">{action_word} at {levels.get('entry'):,.{dp}f}</p>
                <p style="color:white;margin:4px 0 0;font-size:12px;">
                    SL: {levels.get('sl'):,.{dp}f} &nbsp;|&nbsp;
                    TP1: {levels.get('tp1'):,.{dp}f} &nbsp;|&nbsp;
                    {tp2_html}Risk: {dollar_risk_str}
                </p>
            </div>"""
    else:
        # Zone label shows the OB we ACTUALLY enter on -- M15 OB if nested,
        # else H1 OB. Matches what the M15 chart marks.
        zone_ob = data.get('m15_chart_ob') or ob
        proximal = float(zone_ob.get('proximal_line', 0))
        distal = float(zone_ob.get('distal_line', 0))
        action_block = f"""
            <div style="background:#e67e22;padding:14px 18px;border-radius:10px;margin-bottom:14px;">
                <p style="color:white;font-size:15px;font-weight:bold;margin:0;">APPROACHING &mdash; WAIT FOR M5 CHoCH</p>
                <p style="color:white;margin:6px 0 2px;font-size:12px;">
                    Zone: {min(proximal, distal):,.{dp}f} &rarr; {max(proximal, distal):,.{dp}f} &nbsp;|&nbsp;
                    Direction: {bias}
                </p>
                <p style="color:white;margin:2px 0 0;font-size:11px;opacity:0.9;">
                    Projected SL: {levels.get('sl'):,.{dp}f} &nbsp;|&nbsp;
                    TP1: {levels.get('tp1'):,.{dp}f} &nbsp;|&nbsp;
                    {(f"TP2: {levels.get('tp2'):,.{dp}f} &nbsp;|&nbsp; " if levels.get('tp2') is not None else "")}Risk: {dollar_risk_str}
                </p>
                <p style="color:white;margin:4px 0 0;font-size:11px;opacity:0.85;">
                    Final entry will be confirmed at M5 CHoCH inside the zone.
                </p>
            </div>"""

    # Total max is 8.0 for non-JPY forex (sweep collapsed to presence-only 1.0),
    # 10.0 elsewhere. Mirrors the scoring rule in smc_detector.run_scorecard.
    _pname = pair_conf.get('name', '') if pair_conf else ''
    _ptype = pair_conf.get('pair_type', 'forex') if pair_conf else 'forex'
    total_max_for_card = 8.0 if (_ptype == 'forex' and 'JPY' not in _pname) else 10.0
    scorecard_html = build_scorecard_html(scorecard_rows, total_score, total_max_for_card)

    distance_html = f"""
    <div style="margin-bottom:12px;padding:8px 12px;background:#0d0d1a;border-left:3px solid #00bcd4;border-radius:4px;font-size:12px;color:#bbb;">
        <b style="color:#eee;">Distance:</b> {distance_str} &nbsp;&middot;&nbsp; {atr_label}
    </div>"""

    # Killzone-alignment annotation: did the OB candle land in a killzone?
    # Fill side is unknown until we actually fill, so we only show the OB
    # side here. SMC hypothesis: OB-in-killzone setups outperform.
    ob_in_kz_label = _ob_in_killzone_label(ob, pair_conf)

    context_html = f"""
    <div style="margin-bottom:12px;font-size:11px;color:#888;">
        <b style="color:#aaa;">Zone:</b> {bos_tag}
        &nbsp;&middot;&nbsp; Proximal {ob.get('proximal_line', 0):.{dp}f}
        / Distal {ob.get('distal_line', 0):.{dp}f}
        &nbsp;&middot;&nbsp; <b style="color:#aaa;">OB candle:</b> {ob_in_kz_label}
    </div>"""

    # Trend banner (information only — trader decides whether to take counter-trend)
    trend_alignment = data.get("trend_alignment", "ambiguous")
    trend_label = data.get("trend_label", "H1 trend unavailable")
    if trend_alignment == "with_trend":
        trend_bg, trend_border, trend_icon = "#1a3a1a", "#27ae60", "&#9989;"  # green ✅
    elif trend_alignment == "counter_trend":
        trend_bg, trend_border, trend_icon = "#3a1a1a", "#e74c3c", "&#9888;"  # red ⚠️
    else:
        trend_bg, trend_border, trend_icon = "#2a2a1a", "#f39c12", "&#10067;"  # amber ❓
    trend_banner_html = f"""
    <div style="margin-bottom:12px;padding:10px 14px;background:{trend_bg};border-left:4px solid {trend_border};border-radius:4px;font-size:13px;color:#eee;font-weight:bold;">
        {trend_icon} Trend Context: {trend_label}
    </div>"""

    # Chart blocks with fallback banner (B8)
    if h1_chart_ok:
        h1_chart_block = '<img src="cid:chart_h1" style="width:100%;border-radius:6px;margin-bottom:12px;" />'
    else:
        h1_chart_block = '<div style="padding:10px 12px;background:#2d1a1a;border-left:3px solid #e74c3c;border-radius:4px;color:#e74c3c;font-size:12px;margin-bottom:12px;">&#9888; H1 chart failed to render for this alert. Check GitHub Actions logs.</div>'

    if m15_chart_ok:
        m15_chart_block = '<img src="cid:chart_m15" style="width:100%;border-radius:6px;margin-bottom:12px;" />'
    else:
        m15_chart_block = '<div style="padding:10px 12px;background:#2d1a1a;border-left:3px solid #e74c3c;border-radius:4px;color:#e74c3c;font-size:12px;margin-bottom:12px;">&#9888; M15 chart failed to render for this alert. Check GitHub Actions logs.</div>'

    # Score-change driver line: only shown on update emails.
    score_change_line = data.get("score_change_line")
    if score_change_line:
        score_change_html = f"""
    <div style="margin-bottom:12px;padding:10px 14px;background:#1a1a2e;border-left:3px solid #f1c40f;border-radius:4px;font-size:12px;color:#eee;line-height:1.5;">
        <b style="color:#f1c40f;">Score change:</b> {score_change_line}
    </div>"""
    else:
        score_change_html = ""

    sweep_breakdown_html = build_sweep_breakdown_html(data, dp)

    # Macro context block. If Gemini failed, render a distinct unavailable
    # banner so the trader knows to manually check news — NOT a fake
    # "no events" summary that misleads.
    if data.get('macro_unavailable'):
        macro_html = (
            '<div style="margin-top:12px;padding:10px 12px;background:#2d1a1a;'
            'border-left:3px solid #e74c3c;border-radius:4px;font-size:12px;'
            'color:#ffb3b3;line-height:1.5;">'
            '<b style="color:#e74c3c;">&#9888; Macro Context Unavailable:</b> '
            'Gemini API failed for this scan. Check macro news manually before '
            'entering.</div>'
        )
    else:
        _ms = data.get('macro_summary') or 'N/A'
        macro_html = (
            f'<div style="margin-top:12px;padding:10px 12px;background:#0d0d1a;'
            f'border-left:3px solid #888;border-radius:4px;font-size:12px;'
            f'color:#bbb;line-height:1.5;">'
            f'<b style="color:#eee;">Macro Context:</b> {_ms}</div>'
        )

    return f"""<html><body style="font-family:Arial,sans-serif;background:#0d0d1a;padding:12px;margin:0;">
    <div style="max-width:650px;margin:auto;background:#13131f;border-radius:14px;overflow:hidden;">
        <div style="background:#1a1a2e;padding:14px 18px;">
            <h2 style="color:#eee;margin:0;font-size:16px;">{state_msg}: {pair} &middot; {bias}</h2>
            <p style="color:#888;margin:4px 0 0;font-size:11px;">{ist_time}</p>
        </div>
        <div style="padding:14px 18px;">
            {action_block}
            {trend_banner_html}
            {distance_html}
            {context_html}
            {scorecard_html}
            <div style="margin:14px 0 6px 0;color:#aaa;font-size:11px;letter-spacing:1px;text-transform:uppercase;">H1 Context</div>
            {h1_chart_block}
            {_chart_legend_html(bos_tag, bos_tier)}
            <div style="margin:14px 0 6px 0;color:#aaa;font-size:11px;letter-spacing:1px;text-transform:uppercase;">M15 Approach</div>
            {m15_chart_block}
            {_chart_legend_html(bos_tag, bos_tier)}
            {sweep_breakdown_html}
            {macro_html}
        </div>
    </div></body></html>"""


def build_sweep_breakdown_html(data, dp):
    """
    Render the Sweep Quality Breakdown banner. Placed below M15 chart, above
    Macro Context. Shows the three components that make up the sweep score:
    Presence, Equal Levels, Rejection Quality.
    """
    comps = data.get('sweep_components') or {}
    sweep_price = data.get('sweep_price')
    sweep_tf    = data.get('sweep_tf', 'H1')
    hrs_before  = data.get('sweep_hours_before_ob')
    base        = comps.get('base', 0.0)
    eq_score    = comps.get('equal_levels', 0.0)
    eq_matches  = comps.get('equal_levels_matches', 0)
    rej_score   = comps.get('rejection', 0.0)
    wb_ratio    = comps.get('wick_body_ratio', 0.0)

    if base <= 0:
        return f"""
    <div style="margin-top:12px;padding:10px 12px;background:#0d0d1a;border-left:3px solid #e74c3c;border-radius:4px;font-size:12px;color:#bbb;line-height:1.6;">
        <div style="color:#eee;font-weight:bold;margin-bottom:6px;letter-spacing:0.5px;">SWEEP QUALITY BREAKDOWN</div>
        <div>No qualifying sweep detected within the recency window before the OB.</div>
    </div>"""

    presence_icon = "&#10003;" if base > 0 else "&#10007;"      # ✓ / ✗
    eq_icon       = "&#10003;" if eq_score > 0 else "&#10007;"
    rej_icon      = "&#10003;" if rej_score > 0 else "&#10007;"

    hrs_str = f"{hrs_before:.0f}h before OB" if hrs_before is not None else "n/a"
    sweep_price_str = f"{sweep_price:.{dp}f}" if sweep_price is not None else "n/a"

    if rej_score >= 1.0:
        rej_label = "textbook rejection (wick:body > 3)"
    elif rej_score >= 0.66:
        rej_label = "strong rejection (wick:body 2-3)"
    elif rej_score >= 0.33:
        rej_label = "weak rejection (wick:body 1-2)"
    else:
        rej_label = "no real rejection (wick:body < 1)"

    if eq_matches >= 2:
        eq_label = f"{eq_matches} equal levels matched"
    elif eq_matches == 1:
        eq_label = "1 equal level matched"
    else:
        eq_label = "0 equal levels matched"

    total = base + eq_score + rej_score

    return f"""
    <div style="margin-top:12px;padding:10px 12px;background:#0d0d1a;border-left:3px solid #00bcd4;border-radius:4px;font-size:12px;color:#bbb;line-height:1.6;">
        <div style="color:#eee;font-weight:bold;margin-bottom:6px;letter-spacing:0.5px;">SWEEP QUALITY BREAKDOWN</div>
        <div>{presence_icon} <b style="color:#eee;">Presence:</b> {base:.2f}/1.5 &middot; {sweep_tf} sweep at {sweep_price_str}, {hrs_str}</div>
        <div>{eq_icon} <b style="color:#eee;">Equal Levels:</b> {eq_score:.2f}/0.5 &middot; {eq_label}</div>
        <div>{rej_icon} <b style="color:#eee;">Rejection Quality:</b> {rej_score:.2f}/1.0 &middot; {rej_label} (ratio {wb_ratio:.1f})</div>
        <div style="margin-top:4px;color:#eee;"><b>Total: {total:.2f}/3.0</b></div>
    </div>"""
def send_email(subject, html_body, h1_chart_b64, m15_chart_b64):
    for recipient in config["account"].get("alert_emails", []):
        msg = MIMEMultipart("related")
        msg["Subject"], msg["From"], msg["To"] = subject, GMAIL_ADDRESS, recipient
        msg.attach(MIMEText(html_body, "html"))
        if h1_chart_b64:
            img = MIMEImage(base64.b64decode(h1_chart_b64))
            img.add_header("Content-ID", "<chart_h1>")
            img.add_header("Content-Disposition", "inline", filename="chart_h1.png")
            msg.attach(img)
        if m15_chart_b64:
            img = MIMEImage(base64.b64decode(m15_chart_b64))
            img.add_header("Content-ID", "<chart_m15>")
            img.add_header("Content-Disposition", "inline", filename="chart_m15.png")
            msg.attach(img)
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(GMAIL_ADDRESS, GMAIL_PASS)
                server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
        except Exception as e:
            reason = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"Email failed: {reason}")
            _log_smtp_failure(recipient, reason)


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

HEARTBEAT_INTERVAL_HOURS = 3
HEARTBEAT_WINDOW_HOURS = 3  # Window to count recent failures for rules 2, 3, 6


def _count_recent_log_entries(log_path, window_hours, ist_now):
    """Count entries in a log file whose 'ts' is within the last `window_hours`."""
    try:
        entries = load_json(log_path, [])
        if not isinstance(entries, list):
            return 0
        cutoff = ist_now - timedelta(hours=window_hours)
        count = 0
        for e in entries:
            ts_str = e.get("ts") if isinstance(e, dict) else None
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts >= cutoff:
                    count += 1
            except Exception:
                continue
        return count
    except Exception:
        return 0


def _get_active_obs_mtime_hours(ist_now):
    """Return hours since active_obs.json was last modified. None if missing."""
    try:
        if not os.path.exists("active_obs.json"):
            return None
        mtime_utc = datetime.utcfromtimestamp(os.path.getmtime("active_obs.json"))
        mtime_ist = mtime_utc + timedelta(hours=5, minutes=30)
        return (ist_now - mtime_ist).total_seconds() / 3600
    except Exception:
        return None


def _is_weekday_market_hours(ist_now):
    """Rough check: weekday and not deep weekend. FX runs Mon-Fri IST."""
    # Monday=0 ... Sunday=6. Forex is closed Sat 03:30 IST through Mon 03:30 IST roughly.
    wd = ist_now.weekday()
    if wd < 5:  # Mon-Fri
        return True
    if wd == 6 and ist_now.hour >= 4:  # Sunday after ~04:00 IST markets reopen
        return True
    return False


# Phase 1 freshness gate threshold. P1 runs hourly via cron-job.org. We give
# 75 minutes of slack — one P1 cycle (60m) + 15m grace for slow runs / commit
# lag. If active_obs.json hasn't been touched in this window during market
# hours, P2 refuses to scan and emails a one-shot "P1 stale" warning.
P1_FRESHNESS_MAX_AGE_HOURS = 1.25


def check_p1_freshness(ist_now):
    """
    Return (is_fresh, reason). Fresh means BOTH conditions hold:
      (a) active_obs.json was modified within P1_FRESHNESS_MAX_AGE_HOURS, AND
      (b) the slate's slate_date matches today's trading day.

    Either fails => stale. Phase 2 will skip scanning + emit one alert. Off
    on weekends / outside market hours so we don't spam during legitimate
    quiet periods.
    """
    if not _is_weekday_market_hours(ist_now):
        return True, "off_hours"

    age_hrs = _get_active_obs_mtime_hours(ist_now)
    if age_hrs is None:
        return False, "active_obs_missing"
    if age_hrs > P1_FRESHNESS_MAX_AGE_HOURS:
        return False, f"active_obs_age_{age_hrs:.1f}h_over_{P1_FRESHNESS_MAX_AGE_HOURS:.2f}h"

    # Slate-date check: the file may have been touched recently (e.g. by a
    # P1 run that crashed before writing the new slate) but still hold a
    # stale slate_date. Compare to today's trading day.
    try:
        raw = load_json("active_obs.json", {})
        slate_date = raw.get("slate_date") if isinstance(raw, dict) else None
    except Exception:
        slate_date = None
    today_str = get_day_id_ist(ist_now)
    if slate_date and slate_date != today_str:
        return False, f"slate_date_{slate_date}_not_{today_str}"

    return True, "ok"


def emit_p1_stale_alert(ist_now, reason):
    """
    Send a one-shot 'P1 stale' email and persist a flag so subsequent P2
    runs don't spam. Flag clears automatically on the next fresh scan.
    """
    state = load_json("p1_stale_alert_state.json", {})
    if not isinstance(state, dict):
        state = {}
    if state.get("alerted"):
        # Already notified for the current stale streak. Stay silent.
        print(f"  [STALE] P1 still stale ({reason}); already notified — silent.")
        return

    ts_str = ist_now.strftime("%H:%M IST, %d %b")
    subject = f"P2 PAUSED | Phase 1 data is stale | {ts_str}"
    html = f"""<html><body style="background:#131722;font-family:Arial,sans-serif;padding:20px;">
        <div style="max-width:640px;margin:auto;background:#1e222d;padding:20px;border-radius:8px;">
            <div style="color:#eee;font-size:16px;font-weight:bold;margin-bottom:14px;">
                Phase 2 paused — {ts_str}
            </div>
            <div style="padding:14px;background:#3a1b1b;border-left:4px solid #ef5350;border-radius:4px;color:#eee;font-size:14px;line-height:1.6;">
                <b style="color:#ef5350;">&#9888; Phase 1 data is stale.</b><br>
                Phase 2 refused to scan. No trade alerts will be sent until P1 recovers.<br><br>
                <b>Reason:</b> {reason}<br>
                <b>Action:</b> Check P1 Actions tab on GitHub. Investigate why the hourly run isn't producing fresh active_obs.json.
            </div>
            <div style="padding:10px 12px;margin-top:12px;background:#0d0d1a;border-left:3px solid #555;border-radius:4px;font-size:12px;color:#aaa;line-height:1.5;">
                You will receive ONE alert per stale streak. The next P2 run that sees fresh data will resume scanning silently.
            </div>
        </div>
    </body></html>"""
    try:
        send_email(subject, html, None, None)
    except Exception as e:
        print(f"  [STALE ALERT ERR] {e}")
    state["alerted"] = True
    state["since_ist"] = ist_now.isoformat()
    state["reason"] = reason
    save_json("p1_stale_alert_state.json", state)
    print(f"  [STALE] One-shot P1-stale alert sent: {reason}")


def clear_p1_stale_flag():
    """Called once P1 is fresh again. Resets the one-shot alert flag."""
    state = load_json("p1_stale_alert_state.json", {})
    if isinstance(state, dict) and state.get("alerted"):
        save_json("p1_stale_alert_state.json", {"alerted": False})
        print("  [STALE] P1 fresh again — stale-alert flag cleared.")


def collect_heartbeat_diagnostics(ist_now, active_obs):
    """Return dict with issues list and context fields."""
    issues = []

    # Rule 1: P1 stale during market hours
    ob_age_hrs = _get_active_obs_mtime_hours(ist_now)
    if _is_weekday_market_hours(ist_now):
        if ob_age_hrs is None:
            issues.append({
                "title": "active_obs.json is missing",
                "action": "Check P1 Actions tab for last successful run."
            })
        elif ob_age_hrs > 3:
            issues.append({
                "title": f"P1 has not updated active_obs.json in {ob_age_hrs:.1f} hours",
                "action": "Check P1 Actions tab for last successful run."
            })

    # Rule 2: Gemini failures in window
    gemini_fails = _count_recent_log_entries(
        "gemini_failure_log.json", HEARTBEAT_WINDOW_HOURS, ist_now
    )
    if gemini_fails >= 3:
        issues.append({
            "title": f"{gemini_fails} Gemini API failures in last {HEARTBEAT_WINDOW_HOURS}h",
            "action": "Check gemini_failure_log.json — likely rate limit or key issue."
        })

    # Rule 3: yfinance stale skips in window
    yf_stale = _count_recent_log_entries(
        "yfinance_stale_log.json", HEARTBEAT_WINDOW_HOURS, ist_now
    )
    if yf_stale >= 3:
        issues.append({
            "title": f"{yf_stale} yfinance stale/failed fetches in last {HEARTBEAT_WINDOW_HOURS}h",
            "action": "Check yfinance_stale_log.json. If persistent, yfinance may be rate-limiting."
        })

    # Rule 5: active_obs empty
    ob_count = 0
    if isinstance(active_obs, dict):
        for pair_list in active_obs.values():
            if isinstance(pair_list, list):
                ob_count += len(pair_list)
    if ob_count == 0:
        issues.append({
            "title": "active_obs.json has zero zones across all pairs",
            "action": "Verify P1 is detecting structure. Check smc_radar.log in last P1 run."
        })

    # Rule 6: Chart failures in window
    chart_fails = _count_recent_log_entries(
        "chart_failure_log.json", HEARTBEAT_WINDOW_HOURS, ist_now
    )
    if chart_fails > 5:
        issues.append({
            "title": f"{chart_fails} chart render failures in last {HEARTBEAT_WINDOW_HOURS}h",
            "action": "Check chart_failure_log.json. Charts failing silently; emails still send with banner."
        })

    # Rule 7: SMTP failures in window. Even one means an alert may have been
    # dropped silently. The heartbeat itself rides the same SMTP path, so the
    # log file is the only reliable signal of a degraded send path.
    smtp_fails = _count_recent_log_entries(
        "smtp_failure_log.json", HEARTBEAT_WINDOW_HOURS, ist_now
    )
    if smtp_fails >= 1:
        issues.append({
            "title": f"{smtp_fails} SMTP send failures in last {HEARTBEAT_WINDOW_HOURS}h",
            "action": "Check smtp_failure_log.json. Likely bad GMAIL_APP_PASSWORD or Gmail outage. Alerts may have been lost."
        })

    return {
        "issues": issues,
        "ob_count": ob_count,
        "ob_age_hrs": ob_age_hrs,
        "gemini_fails": gemini_fails,
        "yf_stale": yf_stale,
        "chart_fails": chart_fails,
        "smtp_fails": smtp_fails,
    }


def build_heartbeat_email_html(diag, ist_now):
    """Return (subject, html_body)."""
    issues = diag["issues"]
    ts_str = ist_now.strftime("%H:%M IST, %d %b")
    # Guard formatting against None ob_age_hrs (active_obs.json missing).
    # Previous code formatted '.1f' unconditionally and crashed the heartbeat
    # in the exact scenario where it was most needed.
    ob_age_str = (f"{diag['ob_age_hrs']:.1f}h" if diag['ob_age_hrs'] is not None
                  else "unknown")
    smtp_fails = diag.get('smtp_fails', 0)

    if not issues:
        subject = f"P2 HEARTBEAT | {ts_str} | ✅ ALL CLEAR"
        body_inner = f"""
            <div style="padding:14px;background:#1b3a1b;border-left:4px solid #26a69a;border-radius:4px;color:#eee;font-size:14px;line-height:1.6;">
                <b style="color:#26a69a;font-size:15px;">✅ All systems green.</b><br>
                Phase 2 is scanning on schedule. No errors in last {HEARTBEAT_WINDOW_HOURS}h.<br>
                Active zones across all pairs: <b>{diag['ob_count']}</b>.
            </div>
            <div style="padding:10px 12px;margin-top:12px;background:#0d0d1a;border-left:3px solid #555;border-radius:4px;font-size:12px;color:#aaa;line-height:1.5;">
                <b style="color:#ccc;">FYI:</b>
                P1 last updated active_obs.json {ob_age_str} ago.
                Gemini failures: {diag['gemini_fails']}. yfinance stale: {diag['yf_stale']}. Chart failures: {diag['chart_fails']}. SMTP failures: {smtp_fails}.
            </div>
        """
    else:
        n = len(issues)
        subject = f"P2 HEARTBEAT | {ts_str} | ⚠️ {n} ISSUE{'S' if n > 1 else ''}"
        issue_blocks = ""
        for i, iss in enumerate(issues, 1):
            issue_blocks += f"""
            <div style="padding:12px 14px;margin-bottom:10px;background:#3a1b1b;border-left:4px solid #ef5350;border-radius:4px;color:#eee;font-size:14px;line-height:1.6;">
                <b style="color:#ef5350;">⚠️ ISSUE {i}:</b> {iss['title']}<br>
                <span style="color:#ffb74d;">→ ACTION:</span> {iss['action']}
            </div>
            """
        body_inner = issue_blocks + f"""
            <div style="padding:10px 12px;margin-top:12px;background:#0d0d1a;border-left:3px solid #555;border-radius:4px;font-size:12px;color:#aaa;line-height:1.5;">
                <b style="color:#ccc;">Context:</b>
                Active zones: {diag['ob_count']}.
                P1 last update: {ob_age_str} ago.
                Gemini fails (3h): {diag['gemini_fails']}. yfinance stale (3h): {diag['yf_stale']}. Chart fails (3h): {diag['chart_fails']}. SMTP fails (3h): {smtp_fails}.
            </div>
        """

    html = f"""<html><body style="background:#131722;font-family:Arial,sans-serif;padding:20px;">
        <div style="max-width:640px;margin:auto;background:#1e222d;padding:20px;border-radius:8px;">
            <div style="color:#eee;font-size:16px;font-weight:bold;margin-bottom:14px;letter-spacing:0.5px;">
                Phase 2 Heartbeat — {ts_str}
            </div>
            {body_inner}
        </div>
    </body></html>"""
    return subject, html


def send_heartbeat_if_due(ist_now, active_obs):
    """Check timestamp gate, run diagnostics, send email if due. Never raises."""
    try:
        hb_state = load_json("heartbeat_state.json", {})
        last_iso = hb_state.get("last_sent_ist")
        if last_iso:
            try:
                last_dt = datetime.fromisoformat(last_iso)
                hrs_since = (ist_now - last_dt).total_seconds() / 3600
                if hrs_since < HEARTBEAT_INTERVAL_HOURS:
                    return
            except Exception:
                pass  # Corrupt timestamp -> treat as due

        diag = collect_heartbeat_diagnostics(ist_now, active_obs)
        email_sent = False
        if diag["issues"]:
            subject, html = build_heartbeat_email_html(diag, ist_now)
            send_email(subject, html, None, None)
            email_sent = True

        hb_state["last_sent_ist"] = ist_now.isoformat()
        save_json("heartbeat_state.json", hb_state)

        # --- Rolling diagnostic log: append snapshot of this heartbeat ---
        try:
            ob_per_pair = {}
            if isinstance(active_obs, dict):
                for pair_name, pair_list in active_obs.items():
                    ob_per_pair[pair_name] = len(pair_list) if isinstance(pair_list, list) else 0

            log_entry = {
                "ts": ist_now.isoformat(),
                "ts_human": ist_now.strftime("%H:%M IST, %d %b %Y"),
                "ob_count_total": diag["ob_count"],
                "ob_count_per_pair": ob_per_pair,
                "ob_age_hrs": diag["ob_age_hrs"],
                "gemini_fails_3h": diag["gemini_fails"],
                "yfinance_stale_3h": diag["yf_stale"],
                "chart_fails_3h": diag["chart_fails"],
                "smtp_fails_3h": diag.get("smtp_fails", 0),
                "issue_count": len(diag["issues"]),
                "issues": [{"title": i["title"], "action": i["action"]} for i in diag["issues"]],
            }
            hb_log = load_json("heartbeat_log.json", [])
            if not isinstance(hb_log, list):
                hb_log = []
            hb_log.append(log_entry)
            hb_log = hb_log[-50:]  # keep last 50 heartbeats (~6 days at 3h interval)
            save_json("heartbeat_log.json", hb_log)
        except Exception as e:
            print(f"  [HEARTBEAT LOG ERR] {e}")

        status = "Sent" if email_sent else "Silent (all clear)"
        print(f"  [HEARTBEAT] {status}. Issues: {len(diag['issues'])}. OB count: {diag['ob_count']}.")
    except Exception as e:
        print(f"  [HEARTBEAT ERR] {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# NEW
if __name__ == "__main__":
    ist_now = get_ist_now()
    scan_start_ts = ist_now  # Captured at scan start; separate from per-alert send time.
    print(f"Phase 2 Engine started {ist_now.strftime('%H:%M')} IST")

    # --- Phase 1 freshness gate -----------------------------------------
    # Phase 2 must NEVER alert on stale Phase 1 data. If P1 hasn't refreshed
    # active_obs.json within the freshness window during market hours, we
    # send one warning email and exit. The flag self-clears on recovery.
    is_fresh, fresh_reason = check_p1_freshness(ist_now)
    if not is_fresh:
        print(f"  [STALE GATE] Refusing to scan — {fresh_reason}")
        emit_p1_stale_alert(ist_now, fresh_reason)
        import sys
        sys.exit(0)
    else:
        clear_p1_stale_flag()

    # Rotate scan log first so a long-running system doesn't accumulate
    # unbounded JSONL. Keeps last 14 days.
    rotate_scan_log(ist_now)

    active_obs = load_slate_as_pair_map("active_obs.json")
    watch_state = load_json("active_watch_state.json", {})

    # --- active_watch_state stale-entry GC ---
    # Evict watches whose first_seen_ist (or legacy alert_ist) is older than
    # WATCH_STATE_RETENTION_DAYS. A 15-day-old zone is no longer a respectable
    # SMC setup; if price returns, P2 will re-register it fresh on next scan.
    # Anchor on first_seen rather than last_seen so a re-emailed wobbling zone
    # eventually expires regardless of refresh activity.
    WATCH_STATE_RETENTION_DAYS = 15
    watch_cutoff = ist_now - timedelta(days=WATCH_STATE_RETENTION_DAYS)
    watch_kept = {}
    watch_evicted = 0
    for k, v in (watch_state or {}).items():
        if not isinstance(v, dict):
            watch_evicted += 1
            continue
        anchor_iso = v.get("first_seen_ist") or v.get("alert_ist")
        if not anchor_iso:
            watch_kept[k] = v
            continue
        try:
            anchor_dt = datetime.fromisoformat(anchor_iso)
        except Exception:
            watch_kept[k] = v
            continue
        if anchor_dt < watch_cutoff:
            watch_evicted += 1
            continue
        watch_kept[k] = v
    if watch_evicted:
        print(f"  [WATCH GC] Evicted {watch_evicted} watches older than {WATCH_STATE_RETENTION_DAYS}d.")
        # Persist eviction immediately so an orphaned key can't sneak back via
        # the end-of-scan concurrency-safe merge.
        save_json("active_watch_state.json", watch_kept)
    watch_state = watch_kept
    # --- Phase 2 dedup state — lifetime model (no daily reset) ---
    # Structure: { "day_id": "YYYY-MM-DD",  # retained for backward-compat reads; no longer used to wipe
    #              "zones": { zone_id: {"score_int": ..., "score_raw": ...,
    #                                    "alert_ist": ..., "last_seen_ist": ...,
    #                                    "breakdown": ..., ...} } }
    #
    # A zone's dedup entry lives as long as the zone itself lives in P1's
    # active_obs slate. Re-emails are governed ONLY by the asymmetric
    # hysteresis band (+0.7 / -0.5). Calendar rollovers do NOT re-spam.
    #
    # Stale-entry garbage collection: any entry whose last_seen_ist is older
    # than DEDUP_STALE_DAYS days is evicted at load time. Conservative window
    # so a transient yfinance hiccup that skips one P2 scan never evicts a
    # live zone's dedup state.
    DEDUP_STALE_DAYS = 7
    phase2_state = load_json("phase2_sent.json", {"day_id": None, "zones": {}})

    # Defensive: handle legacy schema (flat zone_id -> iso) gracefully.
    if not isinstance(phase2_state, dict) or "zones" not in phase2_state:
        phase2_state = {"day_id": None, "zones": {}}

    # Stale-entry GC. An entry's freshness anchor is last_seen_ist; older
    # entries fall back to alert_ist (entries written before this field
    # existed).
    stale_cutoff = ist_now - timedelta(days=DEDUP_STALE_DAYS)
    zones_in = phase2_state.get("zones") or {}
    zones_kept = {}
    evicted = 0
    for zid, entry in zones_in.items():
        if not isinstance(entry, dict):
            evicted += 1
            continue
        anchor_iso = entry.get("last_seen_ist") or entry.get("alert_ist")
        if not anchor_iso:
            # No anchor — keep (cannot prove stale). Will get one on next match.
            zones_kept[zid] = entry
            continue
        try:
            anchor_dt = datetime.fromisoformat(anchor_iso)
        except Exception:
            zones_kept[zid] = entry
            continue
        if anchor_dt < stale_cutoff:
            evicted += 1
            continue
        zones_kept[zid] = entry
    if evicted:
        print(f"  [DEDUP GC] Evicted {evicted} dedup entries older than {DEDUP_STALE_DAYS}d.")
    phase2_state["zones"] = zones_kept
    phase2_state["day_id"] = get_day_id_ist(ist_now)  # informational only; no longer triggers wipe

    phase2_zones = phase2_state["zones"]

    # CONCURRENCY: track watch_state writes; merge on save so P3 deletions stick.
    watch_writes = {}
    balance = config["account"]["balance"]
    risk_pct = config["account"]["risk_percent"]
    dollar_risk = balance * (risk_pct / 100.0)
    dollar_risk_str = f"${dollar_risk:,.0f}"

    for pair_conf in config["pairs"]:
        symbol = pair_conf["symbol"]
        name = pair_conf["name"]
        dp = pair_conf.get("decimal_places", 5)
        entry_model = pair_conf.get("entry_model", "limit")
        pair_obs = active_obs.get(name, [])

        scan_record = {
            "ts_ist": ist_now.isoformat(),
            "pair": name,
            "zones_in_active_obs": len(pair_obs),
            "current_price": None,
            "h1_trend": None,
            "zone_outcomes": [],
            "final_action": None
        }

        if not pair_obs:
            scan_record["final_action"] = "no_zones_in_active_obs"
            append_scan_log(scan_record)
            continue

        radar_tf = pair_conf.get("radar_tf", "15m")
        df_m15 = fetch_with_retry(symbol, "5d", radar_tf)
        # 30d H1 mirrors Phase 1's MAX_OB_AGE_DAYS window. If a zone is
        # still active in P1's slate, its OB candle must be locatable on
        # P2's H1 frame too — no silent fallback to "latest candle" inside
        # the scorer.
        df_h1 = fetch_with_retry(symbol, "30d", "1h")
        if df_m15 is None or df_h1 is None:
            print(f"  [SKIP] {name}: data unavailable after retries")
            continue

        current_price = float(df_m15['Close'].iloc[-1])
        h1_atr = smc_detector.compute_atr(df_h1)
        m15_atr = smc_detector.compute_atr(df_m15)
        if not h1_atr:
            continue

        # BOS sequence count is read from dealing_range state (single source of
        # truth). Counter resets on Major CHoCH; Minor CHoCH does NOT reset it.
        bos_counter = smc_detector.compute_bos_sequence_count(name)

        scan_record["current_price"] = current_price

        surviving_obs = []
        for ob in pair_obs:
            proximal = float(ob['proximal_line'])
            distal = float(ob['distal_line'])
            bias = "LONG" if ob['direction'] == 'bullish' else "SHORT"
            distance = abs(current_price - proximal)
            prox_cap = pair_conf["atr_multiplier"] * h1_atr

            zone_outcome = {
                "direction": ob['direction'],
                "proximal": proximal,
                "distance": round(distance, dp),
                "proximity_cap": round(prox_cap, dp),
                "result": None
            }

            # 1. Proximity gate
            if distance > prox_cap:
                zone_outcome["result"] = "dropped_proximity"
                scan_record["zone_outcomes"].append(zone_outcome)
                continue

            # 2. OB still-active gate. Phase 1 owns the canonical drop decision,
            # but P1 only runs hourly. Between P1 cycles, price can close beyond
            # the distal — that invalidates the zone. We replay candles from
            # the OB candle to NOW on P2's H1 frame, using the same mitigation
            # rule as P1. If invalidated, drop without scoring/alerting.
            ob_ts_iso_gate = ob.get('ob_timestamp')
            if ob_ts_iso_gate:
                ob_idx_gate, on_chart_gate = smc_detector.locate_ob_candle_idx(
                    df_h1, ob_ts_iso_gate
                )
                if on_chart_gate:
                    mitigated, mit_reason, _touches = smc_detector.is_ob_mitigated_phase1(
                        ob['direction'], distal, proximal, df_h1,
                        start_idx=ob_idx_gate + 1,
                    )
                    if mitigated:
                        zone_outcome["result"] = f"dropped_invalidated_{mit_reason}"
                        scan_record["zone_outcomes"].append(zone_outcome)
                        continue
                else:
                    # OB older than 30d H1 fetch — should be vanishingly rare
                    # given P1's MAX_OB_AGE_DAYS guard. Skip the alert rather
                    # than score against an OB we can't locate.
                    zone_outcome["result"] = "dropped_ob_off_chart"
                    scan_record["zone_outcomes"].append(zone_outcome)
                    continue

            zone_outcome["result"] = "passed_to_trend_gate"
            scan_record["zone_outcomes"].append(zone_outcome)
            surviving_obs.append(ob)
        if not surviving_obs:
            scan_record["final_action"] = "no_zones_passed_proximity"
            append_scan_log(scan_record)
            continue

        # 3. Per-pair single-alert: pick nearest-to-price zone. No trend gating.
        # Trend is computed as INFORMATION ONLY and surfaced in the email banner
        # so the trader can decide with-trend vs counter-trend at execution time.
        current_trend = bos_counter.get('trend')  # 'bullish' | 'bearish' | None
        scan_record["h1_trend"] = current_trend

        # Nearest-to-price wins across ALL surviving zones, regardless of direction
        surviving_obs.sort(key=lambda o: abs(current_price - float(o['proximal_line'])))
        selected_ob = surviving_obs[0]

        # Classify trend alignment for the selected zone (information only)
        zone_dir = selected_ob.get('direction')
        if current_trend is None:
            trend_alignment = "ambiguous"
            trend_label = "H1 trend ambiguous — no clear BOS sequence"
        elif current_trend == zone_dir:
            trend_alignment = "with_trend"
            trend_label = f"WITH H1 trend (H1 is {current_trend})"
        else:
            trend_alignment = "counter_trend"
            trend_label = f"AGAINST H1 trend (H1 is {current_trend}, zone is {zone_dir})"

        scan_record["trend_alignment"] = trend_alignment

        # Inject fresh BOS count + ring-overflow flag onto selected zone.
        # Structure is scored from the zone's own bos_tag/bos_tier — the
        # zone's structural identity does not depend on whether it agrees
        # with the current dominant trend. Trend alignment is rendered as
        # an information-only banner in the email. Previously this injection
        # was gated on `current_trend == zone_dir`, which silently gave
        # counter-trend zones a phantom "BOS #1" score (the scorecard's
        # default). Removed the gating.
        selected_ob['bos_sequence_count'] = bos_counter['count']
        selected_ob['bos_count_maxed'] = bool(bos_counter.get('count_maxed', False))

        # Now score and alert only the selected zone
        for ob in [selected_ob]:
            proximal = float(ob['proximal_line'])
            bias = "LONG" if ob['direction'] == 'bullish' else "SHORT"
            zone_top = max(proximal, float(ob['distal_line']))
            zone_bottom = min(proximal, float(ob['distal_line']))

            # FVG: detect H1 + M15 INDEPENDENTLY in the same time window
            # anchored to the OB candle. Veteran SMC: H1 FVG is the macro
            # displacement signature; M15 FVG is the finer-resolution
            # confirmation in the same window. Both are scored separately.
            #
            # H1 FVG: inherited from Phase 1 (radar already detected it with
            # the OB→OB+7 window). We do NOT redetect on H1.
            # M15 FVG: detected here on the M15 dataframe using OB→OB+28
            # M15 candles. Window anchored to OB timestamp.
            ptype = pair_conf.get("pair_type", "forex")
            fvg_floor_mult = smc_detector.FVG_NOISE_FLOOR_MULT.get(ptype, 0.20)
            m15_atr_for_fvg = m15_atr if m15_atr else 0.0
            atr_floor_m15 = fvg_floor_mult * m15_atr_for_fvg

            # Locate OB candle on M15 via its absolute timestamp.
            ob_ts_iso = ob.get('ob_timestamp')
            ob_idx_m15 = None
            if ob_ts_iso:
                idx_found_m15, on_chart_m15 = smc_detector.locate_ob_candle_idx(
                    df_m15, ob_ts_iso
                )
                if on_chart_m15:
                    ob_idx_m15 = idx_found_m15

            fvg_m15 = {"exists": False, "was_detected": False, "mitigation": "none"}
            if ob_idx_m15 is not None and atr_floor_m15 > 0:
                m15_window_end = min(
                    ob_idx_m15 + smc_detector.FVG_WINDOW_M15_CANDLES,
                    len(df_m15) - 1
                )
                fvg_m15 = smc_detector.detect_fvg_in_zone(
                    df_m15, bias, zone_top, zone_bottom, atr_floor_m15,
                    leg_start_idx=ob_idx_m15, leg_end_idx=m15_window_end,
                    pair_type=ptype
                )

            fvg_h1 = ob.get("fvg", {"exists": False, "was_detected": False,
                                    "mitigation": "none"})

            # Bundle for scorecard + downstream consumers.
            fvg_data = {"h1": fvg_h1, "m15": fvg_m15}

            # Source label for email rendering: which timeframes contributed.
            srcs = []
            if fvg_h1.get('exists'):
                srcs.append("H1")
            if fvg_m15.get('exists'):
                srcs.append("M15")
            fvg_source = "+".join(srcs) if srcs else None

            # Score FIRST. Gemini macro is display-only — don't pay for the
            # API call until we know the zone is alert-worthy.
            score_res = smc_detector.run_scorecard(
                bias, df_h1, ob, fvg_data, current_price, pair_conf, df_m15
            )
            if score_res['total'] < pair_conf["min_confidence"]:
                scan_record["final_action"] = f"score_below_min ({score_res['total']:.1f}<{pair_conf['min_confidence']})"
                append_scan_log(scan_record)
                continue

            levels = smc_detector.compute_phase2_levels(pair_conf, bias, ob, current_price, df_h1, df_m15)
            if not levels['valid']:
                scan_record["final_action"] = "levels_invalid"
                append_scan_log(scan_record)
                continue

            # Score + levels passed. NOW fetch macro context — only spent on
            # zones that will actually email.
            gemini_risk = call_gemini_flash(name, bias, fetch_macro_news(name))

            # B2: Pass fvg_source, dealing_range, pd_position, plus new sweep
            # tier + components + age into scorecard rows for richer narration.
            scorecard_rows = smc_detector.generate_scorecard_rows(
                bias, score_res['breakdown'], ob,
                score_res.get('sweep_price'), score_res.get('sweep_tf', 'H1'), pair_conf,
                score_res.get('dealing_range'), fvg_source, score_res.get('pd_position'),
                sweep_tier=score_res.get('sweep_tier'),
                sweep_components=score_res.get('sweep_components'),
                sweep_hours_before_ob=score_res.get('sweep_hours_before_ob'),
                fvg=fvg_data
            )
            # Resolve sweep candle timestamp for chart rendering.
            #
            # H1 sweep: consumed from P1's snapshot. P1's sweep_idx points
            # into P1's H1 dataframe, NOT P2's (different fetches). Always
            # use the timestamp the snapshot carries.
            #
            # M15 sweep: detected by P2 just now against THIS df_m15. Both
            # the timestamp (set inside observe_phase1_sweep) and the idx
            # are valid against P2's frame. Prefer timestamp for uniformity.
            sweep_tf_resolved = score_res.get('sweep_tf')
            sweep_ts_iso = score_res.get('sweep_timestamp_iso')
            # Legacy fallback: if timestamp missing (snapshot pre-dates this
            # field), best-effort idx lookup. Off for H1 (cross-frame), but
            # harmless for M15 where the idx is from this scan's frame.
            if not sweep_ts_iso:
                sweep_idx_resolved = score_res.get('sweep_idx')
                if sweep_idx_resolved is not None and sweep_tf_resolved == 'M15':
                    try:
                        sw_ts = df_m15.index[int(sweep_idx_resolved)]
                        sweep_ts_iso = sw_ts.isoformat() if hasattr(sw_ts, 'isoformat') else str(sw_ts)
                    except Exception:
                        sweep_ts_iso = None

            # Inject onto the ob dict so chart functions can locate the sweep
            # candle by timestamp. Non-invasive: ob is reused only for charts.
            ob['sweep_tf'] = sweep_tf_resolved
            ob['sweep_timestamp'] = sweep_ts_iso

            distance = abs(current_price - proximal)
            pip = pip_size(pair_conf)
            distance_pips_num = round(distance / pip, 1)
            distance_str = f"{distance_pips_num} {pip_unit_label(pair_conf)}"
            atr_label = atr_distance_label(distance, m15_atr, "M15")

            trade_data = {
                "pair": name,
                "bias": bias,
                "score": score_res['total'],
                "breakdown": score_res['breakdown'],
                "sweep_price": score_res.get('sweep_price'),
                "sweep_tf": score_res.get('sweep_tf', 'H1'),
                "sweep_tier": score_res.get('sweep_tier'),
                "sweep_components": score_res.get('sweep_components'),
                "sweep_hours_before_ob": score_res.get('sweep_hours_before_ob'),
                "macro_summary": gemini_risk.get("macro_summary"),
                "macro_unavailable": bool(gemini_risk.get("macro_unavailable", False)),
                "levels": levels,
                "ob": ob,
                "alert_ist": ist_now.isoformat(),
                "scorecard_version": "v2",
                "trend_alignment": trend_alignment,
                "trend_label": trend_label,
                "h1_trend": current_trend
            }
            dr = score_res.get('dealing_range')

            if entry_model == "limit":
                # Structural dedup key: uses BOS swing price (stable across scans)
                # instead of OB proximal (which drifts as Phase 1 reselects OB candle).
                # ob_timestamp hour-bucket disambiguates two structurally distinct
                # zones that happen to share a swing price (e.g. re-CHoCH at the
                # same level after a Major CHoCH wipe).
                bos_swing_px = float(ob.get('bos_swing_price', proximal))
                bos_tag = ob.get('bos_tag', 'BOS')
                key_dp = max(0, dp - 1)
                ob_ts_iso = ob.get('ob_timestamp') or ''
                ts_bucket = ob_ts_iso[:13] if len(ob_ts_iso) >= 13 else ob_ts_iso
                zone_id = f"{name}_{bias}_{bos_tag}_{round(bos_swing_px, key_dp)}_{ts_bucket}"

                # Asymmetric hysteresis dedup.
                # First sighting today → email.
                # Re-sighting inside the dead band (+0.7 / -0.5) → silent;
                # we still track high/low water so we can see how much the
                # score wobbled between emails (diagnostic-only).
                # Re-sighting crossing the upward or downward threshold →
                # email with UPDATED prefix + score-drivers line.
                current_score_raw = float(score_res['total'])
                current_breakdown = score_res.get('breakdown', {})
                prior = phase2_zones.get(zone_id)
                is_update = False
                prior_score_raw = None
                prior_breakdown = None
                hi_water = current_score_raw
                lo_water = current_score_raw
                if prior is not None:
                    prior_score_raw = float(prior.get("score_raw", 0.0))
                    prior_breakdown = prior.get("breakdown")
                    prior_hi = float(prior.get("score_high_water", prior_score_raw))
                    prior_lo = float(prior.get("score_low_water",  prior_score_raw))
                    hi_water = max(prior_hi, current_score_raw)
                    lo_water = min(prior_lo, current_score_raw)
                    direction = hysteresis_should_reemail(current_score_raw, prior_score_raw)
                    if direction is None:
                        # Silent — but persist updated watermarks for diagnostics.
                        prior["score_high_water"] = hi_water
                        prior["score_low_water"]  = lo_water
                        # Refresh last_seen_ist so stale-entry GC doesn't evict
                        # a live silent-band zone.
                        prior["last_seen_ist"] = ist_now.isoformat()
                        save_json("phase2_sent.json", phase2_state)
                        scan_record["final_action"] = (
                            f"dedup_hysteresis_silent "
                            f"({prior_score_raw:.1f}->{current_score_raw:.1f}, "
                            f"band -{SCORE_REEMAIL_DOWN_THRESHOLD:.1f}/+{SCORE_REEMAIL_UP_THRESHOLD:.1f})"
                        )
                        append_scan_log(scan_record)
                        continue
                    is_update = True

                # Register zone state in-memory. Persisted AFTER send_email so
                # a crash mid-send re-attempts the alert next scan (duplicate
                # is recoverable; silent loss is not). Stores breakdown so
                # future updates can show driver deltas. Watermarks reset to
                # the current score at email time.
                phase2_zones[zone_id] = {
                    "score_int": score_to_int(current_score_raw),
                    "score_raw": current_score_raw,
                    "score_high_water": current_score_raw,
                    "score_low_water":  current_score_raw,
                    "breakdown": current_breakdown,
                    "alert_ist": ist_now.isoformat(),
                    "last_seen_ist": ist_now.isoformat(),
                    "bias": bias,
                    "pair": name
                }
                # Limit zones do NOT write to active_watch_state.json.
                # Phase 3 only handles ltf_choch zones (NAS100, GOLD).
                m15_chart_ob = build_m15_chart_ob(ob, levels)
                h1_chart = generate_h1_chart(df_h1, ob, pair_conf,
                                             f"{name} H1 - {bias} zone context", levels, dr)
                m15_chart = generate_m15_chart(
                    df_m15, f"{name} M15 - Approach and entry",
                    levels, m15_chart_ob, pair_conf, fvg_data.get('m15') if isinstance(fvg_data, dict) and 'm15' in fvg_data else fvg_data, score_res.get('sweep_price'), dr
                )
                trade_data["m15_chart_ob"] = m15_chart_ob

                h1_ok = h1_chart is not None
                m15_ok = m15_chart is not None
                if not h1_ok:
                    _log_chart_failure(name, "h1_phase2_limit")
                if not m15_ok:
                    _log_chart_failure(name, "m15_phase2_limit")

                if is_update:
                    drivers_line = format_score_driver_line(
                        prior_score_raw, current_score_raw,
                        prior_breakdown, current_breakdown
                    )
                    subject_prefix = "TRADE READY (UPDATED)"
                    email_label = (
                        f"TRADE READY — Score updated {prior_score_raw:.1f} → {current_score_raw:.1f}"
                    )
                    trade_data["score_change_line"] = drivers_line
                    log_action = (
                        f"alert_sent_TRADE_READY_UPDATED "
                        f"({prior_score_raw:.1f}->{current_score_raw:.1f})"
                    )
                    print_label = (
                        f"TRADE READY UPDATE (FOREX): {name} "
                        f"score {prior_score_raw:.1f}->{current_score_raw:.1f}"
                    )
                else:
                    subject_prefix = "TRADE READY"
                    email_label = "TRADE READY"
                    log_action = "alert_sent_TRADE_READY"
                    print_label = f"TRADE READY (FOREX): {name}"

                html = build_trade_email(
                    trade_data, name, pair_conf, email_label,
                    scorecard_rows, score_res['total'],
                    atr_label, distance_str, dollar_risk_str, scan_start_ts,
                    h1_chart_ok=h1_ok, m15_chart_ok=m15_ok
                )
                _subj_max = 8.0 if (pair_conf.get('pair_type') == 'forex' and 'JPY' not in name) else 10.0
                send_email(
                    f"{subject_prefix} | {name} | {bias} | Score {score_res['total']:.1f}/{_subj_max} | {ist_now.strftime('%H:%M IST')}",
                    html, h1_chart, m15_chart
                )
                # Persist dedup state AFTER send. See comment above.
                save_json("phase2_sent.json", phase2_state)
                print(f"  [OK] {print_label}")
                scan_record["final_action"] = log_action
                append_scan_log(scan_record)
            elif entry_model == "ltf_choch":
                # Structural dedup key: BOS swing price + ob_timestamp hour-bucket.
                # watch_id reuses the same recipe so Phase 3's watch survives
                # P1 reselecting the OB candle within the same structural zone.
                # Previously watch_id was keyed on proximal, which drifted with
                # every P1 OB-candle reselection and orphaned the prior watch.
                bos_swing_px = float(ob.get('bos_swing_price', proximal))
                bos_tag = ob.get('bos_tag', 'BOS')
                key_dp = max(0, dp - 1)
                ob_ts_iso = ob.get('ob_timestamp') or ''
                ts_bucket = ob_ts_iso[:13] if len(ob_ts_iso) >= 13 else ob_ts_iso
                zone_id = f"{name}_{bias}_{bos_tag}_{round(bos_swing_px, key_dp)}_{ts_bucket}"
                watch_id = zone_id  # same recipe; stable across OB reselection

                # Preserve first_seen_ist anchor for 15d watch-state GC.
                # New watch -> now. Existing watch -> carry through.
                prior_watch = watch_state.get(watch_id) if isinstance(watch_state, dict) else None
                first_seen_ist = (prior_watch.get("first_seen_ist") if isinstance(prior_watch, dict)
                                  else None) or ist_now.isoformat()
                trade_data["first_seen_ist"] = first_seen_ist

                # Asymmetric hysteresis dedup. Same model as limit branch.
                current_score_raw = float(score_res['total'])
                current_breakdown = score_res.get('breakdown', {})
                prior = phase2_zones.get(zone_id)
                is_update = False
                prior_score_raw = None
                prior_breakdown = None
                hi_water = current_score_raw
                lo_water = current_score_raw
                if prior is not None:
                    prior_score_raw = float(prior.get("score_raw", 0.0))
                    prior_breakdown = prior.get("breakdown")
                    prior_hi = float(prior.get("score_high_water", prior_score_raw))
                    prior_lo = float(prior.get("score_low_water",  prior_score_raw))
                    hi_water = max(prior_hi, current_score_raw)
                    lo_water = min(prior_lo, current_score_raw)
                    direction = hysteresis_should_reemail(current_score_raw, prior_score_raw)
                    if direction is None:
                        # Same zone, inside hysteresis band — silent. But keep
                        # watch_state fresh so Phase 3 keeps monitoring, and
                        # persist updated watermarks for diagnostics.
                        if watch_id in watch_state:
                            trade_data["alert_ist"] = watch_state[watch_id].get(
                                "alert_ist", ist_now.isoformat()
                            )
                        watch_writes[watch_id] = trade_data
                        prior["score_high_water"] = hi_water
                        prior["score_low_water"]  = lo_water
                        # Refresh last_seen_ist so stale-entry GC doesn't evict
                        # a live silent-band zone.
                        prior["last_seen_ist"] = ist_now.isoformat()
                        save_json("phase2_sent.json", phase2_state)
                        scan_record["final_action"] = (
                            f"dedup_hysteresis_silent_ltf "
                            f"({prior_score_raw:.1f}->{current_score_raw:.1f}, "
                            f"band -{SCORE_REEMAIL_DOWN_THRESHOLD:.1f}/+{SCORE_REEMAIL_UP_THRESHOLD:.1f})"
                        )
                        append_scan_log(scan_record)
                        continue
                    is_update = True

                # Register zone state in-memory. Persisted AFTER send_email.
                phase2_zones[zone_id] = {
                    "score_int": score_to_int(current_score_raw),
                    "score_raw": current_score_raw,
                    "score_high_water": current_score_raw,
                    "score_low_water":  current_score_raw,
                    "breakdown": current_breakdown,
                    "alert_ist": ist_now.isoformat(),
                    "last_seen_ist": ist_now.isoformat(),
                    "bias": bias,
                    "pair": name
                }

                m15_chart_ob = build_m15_chart_ob(ob, levels)
                h1_chart = generate_h1_chart(df_h1, ob, pair_conf,
                                             f"{name} H1 - {bias} zone context", levels, dr)
                m15_chart = generate_m15_chart(
                    df_m15, f"{name} M15 - Approach",
                    levels, m15_chart_ob, pair_conf, fvg_data.get('m15') if isinstance(fvg_data, dict) and 'm15' in fvg_data else fvg_data, score_res.get('sweep_price'), dr
                )
                trade_data["m15_chart_ob"] = m15_chart_ob

                h1_ok = h1_chart is not None
                m15_ok = m15_chart is not None
                if not h1_ok:
                    _log_chart_failure(name, "h1_phase2_ltf")
                if not m15_ok:
                    _log_chart_failure(name, "m15_phase2_ltf")

                if is_update:
                    drivers_line = format_score_driver_line(
                        prior_score_raw, current_score_raw,
                        prior_breakdown, current_breakdown
                    )
                    subject_prefix = "APPROACHING (UPDATED)"
                    email_label = (
                        f"APPROACHING — Score updated {prior_score_raw:.1f} → {current_score_raw:.1f}"
                    )
                    trade_data["score_change_line"] = drivers_line
                    log_action = (
                        f"alert_sent_APPROACHING_UPDATED "
                        f"({prior_score_raw:.1f}->{current_score_raw:.1f})"
                    )
                    print_label = (
                        f"APPROACHING UPDATE: {name} "
                        f"score {prior_score_raw:.1f}->{current_score_raw:.1f}"
                    )
                else:
                    subject_prefix = "APPROACHING"
                    email_label = "APPROACHING"
                    log_action = "alert_sent_APPROACHING"
                    print_label = f"LOGGED FOR PHASE 3: {name}"

                html = build_trade_email(
                    trade_data, name, pair_conf, email_label,
                    scorecard_rows, score_res['total'],
                    atr_label, distance_str, dollar_risk_str, scan_start_ts,
                    h1_chart_ok=h1_ok, m15_chart_ok=m15_ok
                )
                _subj_max = 8.0 if (pair_conf.get('pair_type') == 'forex' and 'JPY' not in name) else 10.0
                send_email(
                    f"{subject_prefix} | {name} | {bias} | Score {score_res['total']:.1f}/{_subj_max} | {ist_now.strftime('%H:%M IST')}",
                    html, h1_chart, m15_chart
                )
                # Persist dedup state AFTER send. See comment in limit branch.
                save_json("phase2_sent.json", phase2_state)
                watch_writes[watch_id] = trade_data
                print(f"  [>] {print_label}")
                scan_record["final_action"] = log_action
                append_scan_log(scan_record)

    save_json("phase2_sent.json", phase2_state)

    # CONCURRENCY-SAFE SAVE: re-read latest disk state, apply only our upserts.
    # If P3 deleted keys mid-run, those deletions are preserved.
    #
    # Sticky fields preserved across upserts: a small set of fields that
    # Phase 3 owns and Phase 2 must never wipe. Currently: 'tapped',
    # 'tapped_ist'. Without this carry-over, every P2 hourly upsert would
    # reset Phase 3's sticky tap flag, defeating fix #5.
    STICKY_FROM_P3 = ("tapped", "tapped_ist")
    fresh_disk = load_json("active_watch_state.json", {})
    for k, v in watch_writes.items():
        prior = fresh_disk.get(k)
        if isinstance(prior, dict) and isinstance(v, dict):
            for sf in STICKY_FROM_P3:
                if sf in prior and sf not in v:
                    v[sf] = prior[sf]
        fresh_disk[k] = v
    save_json("active_watch_state.json", fresh_disk)
    print(f"Phase 2 complete. Watch upserts: {len(watch_writes)}")

    # Heartbeat — runs after main scan is fully saved. Wrapped in try/except.
    send_heartbeat_if_due(ist_now, active_obs)
