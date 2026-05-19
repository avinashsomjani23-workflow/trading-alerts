import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import smtplib
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
import smc_detector

with open("config.json") as f:
    config = json.load(f)

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "avinash.somjani23@gmail.com")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD", "dummy")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def parse_iso(ts_str):
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
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


def clean_df(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


# ---------------------------------------------------------------------------
# Fetch with retry + staleness check (B3, B5)
# ---------------------------------------------------------------------------

STALENESS_HOURS = {
    "1h": 2.0,
    "15m": 0.75,
    "5m": 0.30
}


def _check_staleness(df, interval):
    """Return (is_stale, age_hours). age_hours None if df empty."""
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
        return False, None


def _log_stale_skip(symbol, interval, reason, age_hours):
    try:
        log = load_json("yfinance_stale_log.json", [])
        log.append({
            "ts": get_ist_now().isoformat(),
            "symbol": symbol,
            "interval": interval,
            "reason": reason,
            "age_hours": round(age_hours, 2) if age_hours is not None else None
        })
        log = log[-200:]
        save_json("yfinance_stale_log.json", log)
    except Exception as e:
        print(f"  [LOG ERR] stale log: {e}")


def _log_chart_failure(pair, chart_type):
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


def fetch_with_retry(symbol, period, interval, retries=2):
    """Fetch yfinance data with retry + staleness check."""
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
            wait = 2 ** attempt
            print(f"  [RETRY {attempt + 1}/{retries}] {symbol} {interval}: {last_error}. Waiting {wait}s.")
            time.sleep(wait)

    print(f"  [SKIP] {symbol} {interval}: {last_error}")
    _log_stale_skip(symbol, interval, last_error, last_age)
    return None


# ---------------------------------------------------------------------------
# M5 chart with full confluence overlay
# ---------------------------------------------------------------------------

def generate_m5_chart(df_m5, title, levels, ob, pair_conf, m5_fvg, choch_level, sweep_price):
    try:
        dp = pair_conf.get("decimal_places", 5)
        df_plot = df_m5.dropna(subset=['Open', 'High', 'Low', 'Close']).tail(80).copy().reset_index(drop=True)
        n = len(df_plot)
        if n < 5:
            return None

        fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), facecolor='#131722')
        ax.set_facecolor('#131722')
        for s in ax.spines.values():
            s.set_color('#2a2a3e')

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

        tail_n = 80
        full_n = len(df_m5)
        window_start = max(0, full_n - tail_n)

        proximal = float(ob.get('proximal_line', 0))
        distal = float(ob.get('distal_line', 0))
        zone_hi, zone_lo = max(proximal, distal), min(proximal, distal)

        # H1 zone band context
        if zone_hi > 0 and zone_lo > 0:
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                facecolor='#9b59b6', alpha=0.15, zorder=1
            ))
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                fill=False, edgecolor='#bb8fce', linestyle=':', linewidth=1.5, zorder=2
            ))

        # --- M5 FVG: outline middle (displacement) candle only, slightly wider for mitigation visibility ---
        if m5_fvg and m5_fvg.get('exists'):
            ft, fb = float(m5_fvg.get('fvg_top', 0)), float(m5_fvg.get('fvg_bottom', 0))
            c1_idx = m5_fvg.get('c1_idx')
            if ft > 0 and fb > 0 and c1_idx is not None:
                mid_abs = int(c1_idx) + 1
                mid_local = mid_abs - window_start
                if 0 <= mid_local < n:
                    mit = m5_fvg.get('mitigation', 'pristine')
                    if mit == 'partial':
                        # Amber — partial mitigation (caution).
                        face_col, edge_col = '#f4d03f', '#f1c40f'
                    else:
                        face_col, edge_col = '#27ae60', '#2ecc71'
                    fvg_x_start = mid_local - 0.6
                    fvg_width = 1.8 + 1.2
                    ax.add_patch(patches.Rectangle(
                        (fvg_x_start, fb), fvg_width, ft - fb,
                        facecolor=face_col, alpha=0.25, zorder=1
                    ))
                    ax.add_patch(patches.Rectangle(
                        (fvg_x_start, fb), fvg_width, ft - fb,
                        fill=False, edgecolor=edge_col, linestyle='--', linewidth=1.0, zorder=2
                    ))

        # M5 CHoCH horizontal line
        if choch_level is not None and choch_level > 0:
            ax.axhline(y=choch_level, color='#e91e63', linestyle='--', linewidth=1.0, alpha=0.85, zorder=3)

        # M5 sweep is intentionally not drawn. M5 sweep is hardcoded to None
        # in run_phase3 (no M5 confluences beyond FVG by design).

        # OB candle outline (white)
        ob_ts_iso = ob.get('ob_timestamp')
        if ob_ts_iso:
            abs_idx, found = smc_detector.locate_ob_candle_idx(df_m5, ob_ts_iso)
            if found and abs_idx >= window_start:
                local_idx = abs_idx - window_start
                if 0 <= local_idx < n:
                    ob_c_h = float(df_plot['High'].iloc[local_idx])
                    ob_c_l = float(df_plot['Low'].iloc[local_idx])
                    ax.add_patch(patches.Rectangle(
                        (local_idx - 0.5, ob_c_l), 1.0, ob_c_h - ob_c_l,
                        fill=False, edgecolor='#ffffff', linewidth=1.5, zorder=5
                    ))

        # Execution lines: ENTRY, SL
        sl_p = float(levels.get('sl', 0))
        entry_p = float(levels.get('entry', 0))
        if entry_p > 0:
            ax.axhline(y=entry_p, color='#e67e22', linestyle='-', linewidth=1.3, alpha=0.9, zorder=4)
        if sl_p > 0:
            ax.axhline(y=sl_p, color='#e74c3c', linestyle='-', linewidth=1.3, alpha=0.9, zorder=4)

        # Current price line
        current = float(df_plot['Close'].iloc[-1])
        ax.axhline(y=current, color='#ffffff', linewidth=0.8, linestyle='-', alpha=0.5, zorder=2)

        # --- Y-axis range ---
        y_min = float(df_plot['Low'].min())
        y_max = float(df_plot['High'].max())
        if sl_p > 0:
            y_min = min(y_min, sl_p)
        if entry_p > 0:
            y_max = max(y_max, entry_p)
            y_min = min(y_min, entry_p)
        y_min = min(y_min, zone_lo)
        y_max = max(y_max, zone_hi)
        pad = (y_max - y_min) * 0.08
        ax.set_ylim(y_min - pad, y_max + pad)

        # --- Right-edge tags: ENTRY, SL, TP1, TP2 (numbers only, colour-matched) ---
        right_labels = []
        if entry_p > 0:
            right_labels.append((entry_p, f" {entry_p:.{dp}f}", '#e67e22'))
        if sl_p > 0:
            right_labels.append((sl_p, f" {sl_p:.{dp}f}", '#e74c3c'))

        for tp_key, tp_color in [('tp1', '#27ae60'), ('tp2', '#1abc9c')]:
            tp_raw = levels.get(tp_key)
            tp_p = float(tp_raw) if tp_raw is not None else 0.0
            if tp_p > 0:
                y_lo, y_hi = ax.get_ylim()
                if y_lo <= tp_p <= y_hi:
                    ax.axhline(y=tp_p, color=tp_color, linestyle='-', linewidth=1.3, alpha=0.9, zorder=4)
                    right_labels.append((tp_p, f" {tp_p:.{dp}f}", tp_color))
                elif tp_p > y_hi:
                    ax.text(n + 1, y_hi - pad * 0.3, f" \u2191 {tp_p:.{dp}f}",
                            color=tp_color, fontsize=10, va='top', fontweight='bold', zorder=5)
                elif tp_p < y_lo:
                    ax.text(n + 1, y_lo + pad * 0.3, f" \u2193 {tp_p:.{dp}f}",
                            color=tp_color, fontsize=10, va='bottom', fontweight='bold', zorder=5)

        right_stacked = smc_detector.stack_labels(right_labels, pair_conf)
        for adj_price, text, color in right_stacked:
            ax.text(n + 1, adj_price, text, color=color, fontsize=10,
                    va='center', fontweight='bold', zorder=5)

        # --- Mid-chart tags: proximal, distal, CHoCH, current (numbers only, colour-matched) ---
        mid_x = n / 2.0
        mid_labels = []
        if zone_hi > 0:
            mid_labels.append((proximal, f"{proximal:.{dp}f}", '#bb8fce'))
            mid_labels.append((distal, f"{distal:.{dp}f}", '#bb8fce'))
        if choch_level is not None and choch_level > 0:
            mid_labels.append((float(choch_level), f"{float(choch_level):.{dp}f}", '#e91e63'))
        mid_labels.append((current, f"{current:.{dp}f}", '#ffffff'))
        mid_stacked = smc_detector.stack_labels(mid_labels, pair_conf)
        for adj_price, text, color in mid_stacked:
            ax.text(mid_x, adj_price, text, color=color, fontsize=10, va='center',
                    ha='center', fontweight='bold', zorder=5,
                    bbox=dict(facecolor='#131722', edgecolor='none', pad=1.5, alpha=0.75))

        ax.set_xlim(-1, n + 14)
        ax.set_title(title, color='#dddddd', fontsize=11, pad=8, loc='left')
        ax.tick_params(colors='#888', labelsize=9)
        ax.yaxis.tick_right()
        ax.set_xticks([])
        plt.tight_layout(pad=0.5)

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#131722')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return b64
    except Exception as e:
        print(f"M5 chart error: {e}")
        plt.close('all')
        return None

# ---------------------------------------------------------------------------
# Email senders + builders
# ---------------------------------------------------------------------------

def send_email(subject, html_body, chart_b64=None):
    for recipient in config["account"].get("alert_emails", []):
        msg = MIMEMultipart("related")
        msg["Subject"], msg["From"], msg["To"] = subject, GMAIL_ADDRESS, recipient
        msg.attach(MIMEText(html_body, "html"))
        if chart_b64:
            img = MIMEImage(base64.b64decode(chart_b64))
            img.add_header("Content-ID", "<chart_m5>")
            img.add_header("Content-Disposition", "inline", filename="chart_m5.png")
            msg.attach(img)
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(GMAIL_ADDRESS, GMAIL_PASS)
                server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
        except Exception as e:
            print(f"Email failed: {e}")

def _chart_legend_html():
    """Colour-code legend rendered below M5 chart. Cosmetic only."""
    items = [
        ('#bb8fce', 'H1 zone band (proximal/distal)'),
        ('#2ecc71', 'M5 FVG pristine (displacement)'),
        ('#f1c40f', 'M5 FVG partial (proximal touched)'),
        ('#e91e63', 'M5 CHoCH level'),
        ('#ffffff', 'OB candle / current price'),
        ('#e67e22', 'Entry'),
        ('#e74c3c', 'Stop loss'),
        ('#27ae60', 'TP1'),
        ('#1abc9c', 'TP2'),
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
    
def build_trigger_email(pair, bias, ob, levels, m5_fvg, choch_level, pair_conf, ist_now,
                       dollar_risk_str, macro_summary, scan_start_ist, chart_ok=True,
                       rr_after=None):
    dp = pair_conf.get("decimal_places", 5)
    sent_ist = get_ist_now().strftime('%H:%M IST')
    scan_ist = scan_start_ist.strftime('%H:%M IST')
    ist_str = f"Scanned {scan_ist} · Sent {sent_ist}"
    action_word = "SELL" if bias == "SHORT" else "BUY"

    fvg_line = ""
    if m5_fvg and m5_fvg.get('exists'):
        fvg_line = f"<br>M5 FVG: {m5_fvg.get('fvg_bottom',0):.{dp}f} &rarr; {m5_fvg.get('fvg_top',0):.{dp}f}"

    choch_line = ""
    if choch_level:
        choch_line = f"<br>M5 CHoCH level: {choch_level:.{dp}f}"

    rr_line = ""
    if rr_after is not None:
        rr_line = f"<br>RR (post-slippage): {rr_after:.2f}"

    # B8: chart fallback banner
    if chart_ok:
        chart_block = '<img src="cid:chart_m5" style="width:100%;border-radius:6px;margin-bottom:12px;" />'
    else:
        chart_block = '<div style="padding:10px 12px;background:#2d1a1a;border-left:3px solid #e74c3c;border-radius:4px;color:#e74c3c;font-size:12px;margin-bottom:12px;">&#9888; M5 chart failed to render. Trade data above is valid; check GitHub Actions logs.</div>'

    return f"""<html><body style="font-family:Arial,sans-serif;background:#0d0d1a;padding:12px;margin:0;">
    <div style="max-width:650px;margin:auto;background:#13131f;border-radius:14px;overflow:hidden;">
        <div style="background:#1a1a2e;padding:14px 18px;">
            <h2 style="color:#eee;margin:0;font-size:16px;">TRADE READY (M5 SNIPER): {pair} &middot; {bias}</h2>
            <p style="color:#888;margin:4px 0 0;font-size:11px;">{ist_str}</p>
        </div>
        <div style="padding:14px 18px;">
            <div style="background:#27ae60;padding:14px 18px;border-radius:10px;margin-bottom:14px;">
                <p style="color:white;font-size:15px;font-weight:bold;margin:0;">{action_word} MARKET at {levels.get('entry'):,.{dp}f}</p>
                <p style="color:white;margin:4px 0 0;font-size:12px;">
                    SL: {levels.get('sl'):,.{dp}f} &nbsp;|&nbsp;
                    TP1: {levels.get('tp1'):,.{dp}f}{(f" &nbsp;|&nbsp; TP2: " + format(levels.get('tp2'), f',.{dp}f')) if levels.get('tp2') is not None else ""}
                </p>
                <p style="color:white;margin:4px 0 0;font-size:12px;">Risk: {dollar_risk_str}{rr_line}</p>
            </div>
            <div style="background:#0d0d1a;padding:10px 14px;border-left:3px solid #00bcd4;border-radius:4px;font-size:12px;color:#bbb;margin-bottom:14px;line-height:1.5;">
                <b style="color:#eee;">M5 Confluences:</b>
                M5 CHoCH confirmed inside H1 zone bounds.{choch_line}{fvg_line}
            </div>
            <div style="margin:14px 0 6px 0;color:#aaa;font-size:11px;letter-spacing:1px;text-transform:uppercase;">M5 Execution Chart</div>
            {chart_block}
            {_chart_legend_html()}
            <div style="padding:10px 12px;background:#0d0d1a;border-left:3px solid #888;border-radius:4px;font-size:12px;color:#bbb;line-height:1.5;">
                <b style="color:#eee;">Macro Context:</b> {macro_summary or 'N/A'}
            </div>
        </div>
    </div></body></html>"""


# ---------------------------------------------------------------------------
# Invalidation logic
# ---------------------------------------------------------------------------

def is_invalidated(bias, current_close, distal):
    """Uniform invalidation rule across all pairs: M5 close beyond H1 OB distal.

    No ATR buffer, no time expiry, no opposite-structure check (handled
    upstream / by the watch itself). Watch dies silently — no email.
    """
    if bias == "LONG":
        return current_close < distal
    if bias == "SHORT":
        return current_close > distal
    return False


# ---------------------------------------------------------------------------
# Phase 1 freshness gate (mirrors Phase 2's gate). Phase 3 trade triggers must
# never fire on stale upstream data.
# ---------------------------------------------------------------------------

P1_FRESHNESS_MAX_AGE_HOURS = 1.25  # Same threshold as Phase 2; P1 runs hourly.


def _is_weekday_market_hours(ist_now):
    """Rough check: weekday and not deep weekend. FX runs Mon-Fri IST.
    Mirror of Phase 2's helper; kept local to avoid an import dependency."""
    wd = ist_now.weekday()
    if wd < 5:
        return True
    if wd == 6 and ist_now.hour >= 4:
        return True
    return False


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


def check_p1_freshness(ist_now):
    """
    Return (is_fresh, reason). Fresh means active_obs.json was modified within
    P1_FRESHNESS_MAX_AGE_HOURS during market hours. Off-hours bypass keeps the
    gate from spamming during legitimate quiet periods.

    Slate-date is NOT checked here (Phase 2's job). Phase 3 only needs to
    confirm the underlying structure feed is alive; a stale slate during a
    fresh mtime window is Phase 2's responsibility to flag.
    """
    if not _is_weekday_market_hours(ist_now):
        return True, "off_hours"
    age_hrs = _get_active_obs_mtime_hours(ist_now)
    if age_hrs is None:
        return False, "active_obs_missing"
    if age_hrs > P1_FRESHNESS_MAX_AGE_HOURS:
        return False, f"active_obs_age_{age_hrs:.1f}h_over_{P1_FRESHNESS_MAX_AGE_HOURS:.2f}h"
    return True, "ok"


def emit_p3_stale_alert(ist_now, reason):
    """
    Send a one-shot 'P3 paused — P1 stale' email and persist a flag so
    subsequent P3 runs don't spam. Flag clears on the next fresh scan.

    Distinct from P2's stale-alert state (different file) so P2 and P3
    each get their own one-shot per stale streak.
    """
    state = load_json("p3_stale_alert_state.json", {})
    if not isinstance(state, dict):
        state = {}
    if state.get("alerted"):
        print(f"  [STALE] P1 still stale ({reason}); P3 already notified — silent.")
        return

    ts_str = ist_now.strftime("%H:%M IST, %d %b")
    subject = f"P3 PAUSED | Phase 1 data is stale | {ts_str}"
    html = f"""<html><body style="background:#131722;font-family:Arial,sans-serif;padding:20px;">
        <div style="max-width:640px;margin:auto;background:#1e222d;padding:20px;border-radius:8px;">
            <div style="color:#eee;font-size:16px;font-weight:bold;margin-bottom:14px;">
                Phase 3 paused — {ts_str}
            </div>
            <div style="padding:14px;background:#3a1b1b;border-left:4px solid #ef5350;border-radius:4px;color:#eee;font-size:14px;line-height:1.6;">
                <b style="color:#ef5350;">&#9888; Phase 1 data is stale.</b><br>
                Phase 3 refused to fire triggers. No M5 entry alerts will be sent until P1 recovers.<br><br>
                <b>Reason:</b> {reason}<br>
                <b>Action:</b> Check P1 Actions tab on GitHub. Investigate why the hourly run isn't producing fresh active_obs.json.
            </div>
            <div style="padding:10px 12px;margin-top:12px;background:#0d0d1a;border-left:3px solid #555;border-radius:4px;font-size:12px;color:#aaa;line-height:1.5;">
                You will receive ONE alert per stale streak. The next P3 run that sees fresh data will resume silently.
            </div>
        </div>
    </body></html>"""
    try:
        send_email(subject, html, None)
    except Exception as e:
        print(f"  [STALE ALERT ERR] {e}")
    state["alerted"] = True
    state["since_ist"] = ist_now.isoformat()
    state["reason"] = reason
    save_json("p3_stale_alert_state.json", state)
    print(f"  [STALE] One-shot P3-stale alert sent: {reason}")


def clear_p3_stale_flag():
    """Called once P1 is fresh again. Resets the P3 one-shot alert flag."""
    state = load_json("p3_stale_alert_state.json", {})
    if isinstance(state, dict) and state.get("alerted"):
        save_json("p3_stale_alert_state.json", {"alerted": False})
        print("  [STALE] P1 fresh again — P3 stale-alert flag cleared.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# NEW
def run_phase3():
    ist_now = get_ist_now()
    scan_start_ts = ist_now  # Scan start; per-alert send time re-captured at email build.
    print(f"Phase 3 (M5 Trigger) started at {ist_now.strftime('%H:%M')} IST")

    # Phase 1 freshness gate. Phase 3 must never fire entry triggers on stale
    # upstream data — the watch state may reference zones P1 has already killed.
    is_fresh, fresh_reason = check_p1_freshness(ist_now)
    if not is_fresh:
        print(f"  [STALE GATE] Refusing to fire triggers — {fresh_reason}")
        emit_p3_stale_alert(ist_now, fresh_reason)
        return
    else:
        clear_p3_stale_flag()

    watch_state = load_json("active_watch_state.json", {})
    if not watch_state:
        print("Watch state empty. No pairs approaching LTF triggers.")
        return

    balance = config["account"]["balance"]
    risk_pct = config["account"]["risk_percent"]
    dollar_risk = balance * (risk_pct / 100.0)
    dollar_risk_str = f"${dollar_risk:,.0f}"

    keys_to_delete = []
    # Per-watch updates we want merged into the live disk state at save time
    # (e.g. setting tapped=True). Concurrency-safe: at save we re-read disk and
    # apply only these field-level merges, so P2 can write fresh data mid-run
    # without us clobbering it.
    watch_field_updates = {}  # key -> {"tapped": True, ...}

    # Post-slippage RR floor: trade rejected if recomputed RR (using current M5
    # close as MARKET entry) drops below this. Veteran-tuned for Gold/NAS
    # market-on-CHoCH-close behavior.
    min_rr_after_slippage = config.get("phase3", {}).get("min_rr_after_slippage", 1.2)

    for key, data in watch_state.items():
        pair_name = data.get("pair")
        bias = data.get("bias")
        ob = data.get("ob", {})
        phase2_levels = data.get("levels", {}) or {}

        pair_conf = next((p for p in config["pairs"] if p["name"] == pair_name), None)
        if not pair_conf:
            continue

        dp = pair_conf.get("decimal_places", 5)
        proximal = float(ob.get("proximal_line", 0))
        distal = float(ob.get("distal_line", 0))

        trigger_tf = pair_conf.get("trigger_tf", "5m")
        df_m5 = fetch_with_retry(pair_conf["symbol"], "3d", trigger_tf)
        if df_m5 is None or df_m5.empty:
            print(f"  [!] {pair_name}: no M5 data after retries")
            continue

        current_close = float(df_m5['Close'].iloc[-1])
        m5_atr = smc_detector.compute_atr(df_m5)

        # Invalidation: uniform rule, no email. Silent watch deletion.
        if is_invalidated(bias, current_close, distal):
            print(f"  [X] {pair_name} INVALIDATED (close {current_close:.{dp}f} beyond distal {distal:.{dp}f}). Silent delete.")
            keys_to_delete.append(key)
            continue

        # Sticky tap flag. Read once and reuse below — both the TP-side
        # invalidation and the tap gate depend on this.
        already_tapped = bool(data.get("tapped", False))

        # TP-side invalidation. If price reached TP1 BEFORE we ever tapped the
        # zone, the planned move has already paid out without us — the setup
        # is consumed. Kill the watch silently.
        #
        # Two guards prevent murdering tight-RR setups on noise:
        #   1. Only applies when `already_tapped` is False (a tapped watch is
        #      "in play" — TP1 wicks are normal market noise inside the trade).
        #   2. Requires an M5 CLOSE past TP1, not just a wick. A wick to TP1
        #      followed by close back below (LONG case) is not enough.
        #
        # If TP1 is missing from the levels (legacy watch), this rule is
        # skipped — only the distal-side invalidation applies.
        tp1_val = phase2_levels.get('tp1')
        if tp1_val is not None and not already_tapped:
            try:
                tp1_for_check = float(tp1_val)
            except (TypeError, ValueError):
                tp1_for_check = None
            if tp1_for_check is not None:
                recent_closes = df_m5['Close'].tail(30)
                tp1_hit = False
                if bias == "LONG":
                    tp1_hit = bool((recent_closes >= tp1_for_check).any())
                elif bias == "SHORT":
                    tp1_hit = bool((recent_closes <= tp1_for_check).any())
                if tp1_hit:
                    print(f"  [X] {pair_name} TP1-CONSUMED (M5 close past TP1 {tp1_for_check:.{dp}f} before tap). Silent delete.")
                    keys_to_delete.append(key)
                    continue

        # Tap check — sticky flag.
        # Phase 3 used to recompute tap from the last 30 M5 candles (~2.5h)
        # every scan, which meant a watch that tapped 3h ago would silently
        # "untap" itself and miss the CHoCH that followed the second tap.
        # Fix: once tapped, persist tapped=True onto the watch entry and
        # treat it as sticky forever. Fresh-tap detection still uses the
        # last 30 M5 candles to flip the flag the first time.
        if already_tapped:
            tapped_now = True
        else:
            recent_m5 = df_m5.tail(30)
            tapped_now = (
                (bias == "LONG" and recent_m5['Low'].min() <= proximal)
                or (bias == "SHORT" and recent_m5['High'].max() >= proximal)
            )
            if tapped_now:
                # Newly tapped this scan. Queue a field-level merge so disk
                # state retains the flag across runs without overwriting any
                # other fields P2 may have written.
                watch_field_updates[key] = {"tapped": True,
                                            "tapped_ist": ist_now.isoformat()}
        if not tapped_now:
            print(f"  [-] {pair_name}: waiting for tap of proximal ({proximal:.{dp}f})")
            continue

        # M5 CHoCH inside zone bounds + 0.75x M5 ATR grace (logic in smc_detector)
        bounds = {'max': max(proximal, distal), 'min': min(proximal, distal)}
        choch_res = smc_detector.detect_ltf_choch(df_m5, bias, bounds)

        if not choch_res.get("fired"):
            print(f"  [-] {pair_name}: tapped but M5 CHoCH not yet fired")
            continue

        choch_level = choch_res.get("level")
        print(f"  [OK] LTF TRIGGER: {pair_name} M5 CHoCH at {choch_level:.{dp}f}")

        # M5 FVG retained for chart context only (does NOT gate the alert).
        zone_top = max(proximal, distal)
        zone_bottom = min(proximal, distal)
        ptype = pair_conf.get("pair_type", "forex")
        fvg_floor_mult = smc_detector.FVG_NOISE_FLOOR_MULT.get(ptype, 0.20)
        m5_atr_for_fvg = m5_atr if m5_atr else 0.0
        atr_floor_m5 = fvg_floor_mult * m5_atr_for_fvg
        m5_fvg = smc_detector.detect_fvg_in_zone(
            df_m5, bias, zone_top, zone_bottom, atr_floor_m5,
            pair_type=ptype
        )

        # Recompute levels fresh at trigger time using current H1 + M15 data.
        # Never use frozen Phase 2 levels — they may be days old and computed
        # by a superseded code path. SL and TPs must reflect live market structure.
        df_h1 = fetch_with_retry(pair_conf["symbol"], "60d", "1h")
        df_m15 = fetch_with_retry(pair_conf["symbol"], "10d", "15m")
        if df_h1 is None or df_h1.empty or df_m15 is None or df_m15.empty:
            print(f"  [!] {pair_name}: could not fetch H1/M15 for level recompute. Skipping.")
            continue

        fresh_computed = smc_detector.compute_phase2_levels(
            pair_conf, bias, ob, current_close, df_h1, df_m15
        )
        if not fresh_computed.get("valid"):
            print(f"  [X] {pair_name}: fresh level recompute invalid ({fresh_computed.get('reason')}). Killing watch.")
            keys_to_delete.append(key)
            continue

        sl  = float(fresh_computed['sl'])
        tp1 = float(fresh_computed['tp1'])
        tp2_raw = fresh_computed.get('tp2')
        tp2 = float(tp2_raw) if tp2_raw is not None else None

        # MARKET entry: fill at current M5 close, recompute risk/RR from fresh levels.
        entry = current_close
        risk = abs(entry - sl)
        if risk <= 0:
            print(f"  [!] {pair_name}: zero risk at entry. Skipping.")
            continue
        rr_after = abs(tp1 - entry) / risk

        # Sanity: entry must be on the correct side of SL. If price blew past
        # SL before we triggered, kill the watch.
        if (bias == "LONG" and entry <= sl) or (bias == "SHORT" and entry >= sl):
            print(f"  [X] {pair_name}: current close {entry:.{dp}f} on wrong side of SL {sl:.{dp}f}. Killing watch.")
            keys_to_delete.append(key)
            continue

        if rr_after < min_rr_after_slippage:
            print(f"  [!] {pair_name}: RR after slippage {rr_after:.2f} < floor {min_rr_after_slippage}. Killing watch.")
            keys_to_delete.append(key)
            continue

        fresh_levels = {
            "valid": True,
            "entry": round(entry, dp),
            "sl": round(sl, dp),
            "tp1": round(tp1, dp),
            "tp2": round(tp2, dp) if tp2 is not None else None,
            "rr": round(rr_after, 2),
            "entry_source": "MARKET @ M5 close"
        }

        chart_b64 = generate_m5_chart(
            df_m5, f"{pair_name} M5 - Sniper Trigger",
            fresh_levels, ob, pair_conf, m5_fvg, choch_level, None
        )
        chart_ok = chart_b64 is not None
        if not chart_ok:
            _log_chart_failure(pair_name, "m5_phase3_trigger")

        html = build_trigger_email(
            pair_name, bias, ob, fresh_levels, m5_fvg, choch_level, pair_conf,
            ist_now, dollar_risk_str, data.get("macro_summary"), scan_start_ts,
            chart_ok=chart_ok, rr_after=rr_after
        )
        send_email(f"TRADE READY (M5 SNIPER) | {pair_name} | {bias}", html, chart_b64)
        keys_to_delete.append(key)

    # CONCURRENCY-SAFE SAVE: re-read latest disk state, apply field-level
    # merges (e.g. tapped=True) FIRST, then delete only our processed keys.
    # Field merges respect P2's mid-run writes — we never overwrite a whole
    # entry, only the specific fields we own (tapped, tapped_ist).
    fresh_disk = load_json("active_watch_state.json", {})
    merged = 0
    for k, fields in watch_field_updates.items():
        if k in keys_to_delete:
            # Watch will be deleted in the same save; merge is moot.
            continue
        live = fresh_disk.get(k)
        if isinstance(live, dict):
            live.update(fields)
            merged += 1
    deleted = 0
    for k in keys_to_delete:
        if k in fresh_disk:
            del fresh_disk[k]
            deleted += 1
    save_json("active_watch_state.json", fresh_disk)
    print(f"Phase 3 complete. Watch deletions: {deleted}. Field merges: {merged}.")


if __name__ == "__main__":
    run_phase3()
