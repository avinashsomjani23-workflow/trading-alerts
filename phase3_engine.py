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
                        face_col, edge_col = '#a8e6a1', '#7ed67e'
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
            ax.axhline(y=choch_level, color='#ff9800', linestyle='--', linewidth=1.0, alpha=0.85, zorder=3)

        # M5 sweep marker (X only, no text)
        if sweep_price is not None:
            ax.scatter([n - 2], [sweep_price], color='#ff5370', marker='x', s=120, linewidth=2, zorder=5)

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
            tp_p = float(levels.get(tp_key, 0))
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
            mid_labels.append((float(choch_level), f"{float(choch_level):.{dp}f}", '#ff9800'))
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

def build_invalidation_email(pair, bias, reason, reason_detail, ob, levels, alert_ts, ist_now, pair_conf, scan_start_ist):
    dp = pair_conf.get("decimal_places", 5)
    sent_ist = get_ist_now().strftime('%H:%M IST')
    scan_ist = scan_start_ist.strftime('%H:%M IST')
    ist_str = f"Scanned {scan_ist} · Sent {sent_ist}"
    age_hrs = 0
    if alert_ts:
        age_hrs = round((ist_now - alert_ts).total_seconds() / 3600, 1)

    return f"""<html><body style="font-family:Arial,sans-serif;background:#0d0d1a;padding:12px;margin:0;">
    <div style="max-width:620px;margin:auto;background:#13131f;border-radius:14px;overflow:hidden;">
        <div style="background:#2d1a1a;padding:14px 18px;border-bottom:1px solid #4a2020;">
            <h2 style="color:#e74c3c;margin:0;font-size:16px;">INVALIDATED: {pair} &middot; {bias}</h2>
            <p style="color:#888;margin:4px 0 0;font-size:11px;">{ist_str}</p>
        </div>
        <div style="padding:16px 18px;color:#ccc;font-size:13px;line-height:1.6;">
            <div style="background:#2d1a1a;padding:10px 14px;border-left:3px solid #e74c3c;border-radius:4px;margin-bottom:12px;">
                <b style="color:#eee;">Reason:</b> {reason}<br>
                <span style="color:#aaa;font-size:12px;">{reason_detail}</span>
            </div>
            <table style="width:100%;font-size:12px;color:#aaa;">
                <tr><td style="padding:3px 0;">Zone proximal:</td><td style="font-family:monospace;color:#ddd;">{ob.get('proximal_line',0):.{dp}f}</td></tr>
                <tr><td style="padding:3px 0;">Zone distal:</td><td style="font-family:monospace;color:#ddd;">{ob.get('distal_line',0):.{dp}f}</td></tr>
                <tr><td style="padding:3px 0;">Projected entry:</td><td style="font-family:monospace;color:#ddd;">{levels.get('entry',0):.{dp}f}</td></tr>
                <tr><td style="padding:3px 0;">Setup age:</td><td style="color:#ddd;">{age_hrs} hours</td></tr>
            </table>
            <p style="color:#888;font-size:11px;margin-top:12px;">Setup canceled. No further action.</p>
        </div>
    </div></body></html>"""


def _chart_legend_html():
    """Colour-code legend rendered below M5 chart. Cosmetic only."""
    items = [
        ('#bb8fce', 'H1 zone band (proximal/distal)'),
        ('#2ecc71', 'M5 FVG (displacement)'),
        ('#ff9800', 'M5 CHoCH level'),
        ('#ffffff', 'OB candle / current price'),
        ('#e67e22', 'Entry'),
        ('#e74c3c', 'Stop loss'),
        ('#27ae60', 'TP1'),
        ('#1abc9c', 'TP2'),
        ('#ff5370', 'Sweep (X marker)'),
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
                       dollar_risk_str, macro_summary, scan_start_ist, chart_ok=True):
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
                    TP1: {levels.get('tp1'):,.{dp}f} &nbsp;|&nbsp;
                    TP2: {levels.get('tp2'):,.{dp}f}
                </p>
                <p style="color:white;margin:4px 0 0;font-size:12px;">Risk: {dollar_risk_str}</p>
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

def check_invalidation(bias, current_close, distal, m5_atr, pair_conf, alert_ts, ist_now, df_h1):
    """Return (reason, detail) tuple if invalidated, else (None, None)."""
    dp = pair_conf.get("decimal_places", 5)

    # B9: Corrupt state — timestamp missing/unparseable → force invalidation.
    # Prevents watch entries from persisting forever if their timestamps get corrupted.
    if alert_ts is None:
        return (
            "Corrupt state",
            "Alert timestamp missing or unparseable. Forcing invalidation to prevent stuck watch entry."
        )

    # 1. ATR-buffered distal breach (with naive fallback if ATR unavailable)
    atr_buffer = pair_conf.get("invalidation_atr_multiplier", 0.3)
    if m5_atr is not None:
        buf = atr_buffer * m5_atr
        if bias == "LONG" and current_close < (distal - buf):
            return (
                "Distal breached with buffer",
                f"M5 closed at {current_close:.{dp}f}, more than {atr_buffer}x M5 ATR ({buf:.{dp}f}) below distal {distal:.{dp}f}."
            )
        if bias == "SHORT" and current_close > (distal + buf):
            return (
                "Distal breached with buffer",
                f"M5 closed at {current_close:.{dp}f}, more than {atr_buffer}x M5 ATR ({buf:.{dp}f}) above distal {distal:.{dp}f}."
            )
    else:
        if bias == "LONG" and current_close < distal:
            return ("Distal breached (naive)", f"M5 closed at {current_close:.{dp}f} below distal {distal:.{dp}f}. ATR buffer unavailable.")
        if bias == "SHORT" and current_close > distal:
            return ("Distal breached (naive)", f"M5 closed at {current_close:.{dp}f} above distal {distal:.{dp}f}. ATR buffer unavailable.")

    # 2. Time expiry
    max_hrs = pair_conf.get("invalidation_time_hours", 24)
    age_hrs = (ist_now - alert_ts).total_seconds() / 3600
    if age_hrs > max_hrs:
        return ("Time expiry", f"Setup has been active {round(age_hrs,1)} hours without trigger (limit: {max_hrs}h).")

    # NEW
    # 3. Opposite H1 BOS. check_opposite_bos now expects UTC-naive since_ts.
    # alert_ts is IST-naive (parsed from ist timestamps) -> subtract 5h30m.
    since_utc = alert_ts - timedelta(hours=5, minutes=30) if alert_ts else None
    if df_h1 is not None and smc_detector.check_opposite_bos(df_h1, bias, since_ts=since_utc):
        return (
            "Opposite H1 structure shift",
            "H1 printed a Break of Structure in the opposite direction since the alert fired. The setup's foundation is broken."
        )

    return (None, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# NEW
def run_phase3():
    ist_now = get_ist_now()
    scan_start_ts = ist_now  # Scan start; per-alert send time re-captured at email build.
    print(f"Phase 3 (M5 Trigger) started at {ist_now.strftime('%H:%M')} IST")

    watch_state = load_json("active_watch_state.json", {})
    if not watch_state:
        print("Watch state empty. No pairs approaching LTF triggers.")
        return

    balance = config["account"]["balance"]
    risk_pct = config["account"]["risk_percent"]
    dollar_risk = balance * (risk_pct / 100.0)
    dollar_risk_str = f"${dollar_risk:,.0f}"

    keys_to_delete = []

    for key, data in watch_state.items():
        pair_name = data.get("pair")
        bias = data.get("bias")
        ob = data.get("ob", {})
        levels = data.get("levels", {})
        alert_ts = parse_iso(data.get("alert_ist"))

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

        # Fetch H1 once for opposite-BOS check
        df_h1 = fetch_with_retry(pair_conf["symbol"], "10d", "1h")

        # Invalidation first
        inv_reason, inv_detail = check_invalidation(
            bias, current_close, distal, m5_atr, pair_conf, alert_ts, ist_now, df_h1
        )

        if inv_reason:
            print(f"  [X] {pair_name} INVALIDATED: {inv_reason}")
            html_inv = build_invalidation_email(
                pair_name, bias, inv_reason, inv_detail, ob, levels, alert_ts, ist_now, pair_conf, scan_start_ts
            )
            send_email(f"INVALIDATED | {pair_name} | {bias}", html_inv)
            keys_to_delete.append(key)
            continue

        # Tap check
        recent_m5 = df_m5.tail(30)
        tapped = (bias == "LONG" and recent_m5['Low'].min() <= proximal) or \
                 (bias == "SHORT" and recent_m5['High'].max() >= proximal)
        if not tapped:
            print(f"  [-] {pair_name}: waiting for tap of proximal ({proximal:.{dp}f})")
            continue

        # M5 CHoCH inside zone bounds (A1: grace band logic in smc_detector)
        bounds = {'max': max(proximal, distal), 'min': min(proximal, distal)}
        choch_res = smc_detector.detect_ltf_choch(df_m5, bias, bounds)

        if not choch_res.get("fired"):
            print(f"  [-] {pair_name}: tapped but M5 CHoCH not yet fired")
            continue

        choch_level = choch_res.get("level")
        print(f"  [OK] LTF TRIGGER: {pair_name} M5 CHoCH at {choch_level:.{dp}f}")

        # NEW
        # M5 FVG stays: provides chart context and lets compute_dynamic_levels
        # anchor entry precisely on the OB candle. It does NOT gate email on its
        # own — email gating is RR-only (fresh_levels['valid']). If OB Proximal
        # fallback yields RR >= 1.5, the alert fires regardless of M5 FVG.
        # M5 sweep dropped entirely per design (no M5 confluences beyond FVG).
        zone_top = max(proximal, distal)
        zone_bottom = min(proximal, distal)

        ptype = pair_conf.get("pair_type", "forex")
        fvg_floor_mult = smc_detector.FVG_NOISE_FLOOR_MULT.get(ptype, 0.20)
        m5_atr_for_fvg = m5_atr if m5_atr else 0.0
        atr_floor_m5 = fvg_floor_mult * m5_atr_for_fvg

        m5_fvg = smc_detector.detect_fvg_in_zone(
            df_m5, bias, zone_top, zone_bottom, atr_floor_m5
        )
        m5_sweep_price = None  # M5 sweep not used on chart or in levels.

        # Recompute levels with fresh M5 FVG — entry must land on the OB candle.
        fresh_levels = smc_detector.compute_dynamic_levels(pair_conf, bias, ob, m5_fvg, current_close, df_m5)
        if not fresh_levels['valid']:
            print(f"  [!] {pair_name}: CHoCH fired but RR invalid. Skipping.")
            continue

        chart_b64 = generate_m5_chart(
            df_m5, f"{pair_name} M5 - Sniper Trigger",
            fresh_levels, ob, pair_conf, m5_fvg, choch_level, m5_sweep_price
        )
        chart_ok = chart_b64 is not None
        if not chart_ok:
            _log_chart_failure(pair_name, "m5_phase3_trigger")
            
        html = build_trigger_email(
            pair_name, bias, ob, fresh_levels, m5_fvg, choch_level, pair_conf,
            ist_now, dollar_risk_str, data.get("macro_summary"), scan_start_ts, chart_ok=chart_ok
        )
        send_email(f"TRADE READY (M5 SNIPER) | {pair_name} | {bias}", html, chart_b64)
        keys_to_delete.append(key)

    for k in keys_to_delete:
        if k in watch_state:
            del watch_state[k]
    save_json("active_watch_state.json", watch_state)
    print("Phase 3 complete.")


if __name__ == "__main__":
    run_phase3()
