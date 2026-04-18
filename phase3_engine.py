import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import smtplib
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

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "dummy@gmail.com")
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
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def clean_df(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


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
            ax.text(1, zone_hi, ' H1 Zone', color='#bb8fce', fontsize=8, va='bottom', zorder=5)

        # M5 FVG overlay
        if m5_fvg and m5_fvg.get('exists'):
            ft, fb = float(m5_fvg.get('fvg_top', 0)), float(m5_fvg.get('fvg_bottom', 0))
            if ft > 0 and fb > 0:
                ax.add_patch(patches.Rectangle(
                    (0, fb), n + 5, ft - fb,
                    facecolor='#27ae60', alpha=0.25, zorder=1
                ))
                ax.text(1, ft, ' M5 FVG', color='#2ecc71', fontsize=8, va='bottom', zorder=5)

        # M5 CHoCH level
        if choch_level is not None and choch_level > 0:
            ax.axhline(y=choch_level, color='#ff9800', linestyle='--', linewidth=1.0, alpha=0.85, zorder=3)
            ax.text(n + 1, choch_level, f" M5 CHoCH {choch_level:.{dp}f}",
                    color='#ff9800', fontsize=8, va='center', fontweight='bold', zorder=5)

        # M5 sweep marker
        if sweep_price is not None:
            ax.scatter([n - 2], [sweep_price], color='#ff5370', marker='x', s=120, linewidth=2, zorder=5)
            ax.text(n - 2, sweep_price, ' sweep', color='#ff5370', fontsize=8, va='bottom', zorder=5)

        # Execution levels — draw lines, collect labels for stacking
        level_styles = {
            'entry': ('#e67e22', 'ENTRY'),
            'sl': ('#e74c3c', 'SL')
        }
        for key, (color, lbl) in level_styles.items():
            price = float(levels.get(key, 0))
            if price > 0:
                ax.axhline(y=price, color=color, linestyle='-', linewidth=1.3, alpha=0.9, zorder=4)

        # Collect all labels for stacking
        raw_labels = []
        for key, (color, lbl) in level_styles.items():
            price = float(levels.get(key, 0))
            if price > 0:
                raw_labels.append((price, f" {lbl} {price:.{dp}f}", color))
        if choch_level is not None and choch_level > 0:
            raw_labels.append((choch_level, f" M5 CHoCH {choch_level:.{dp}f}", '#ff9800'))
        # TP1/TP2 added as labels only (lines drawn separately below)
        for tp_key, tp_color, tp_lbl in [('tp1', '#27ae60', 'TP1'), ('tp2', '#1abc9c', 'TP2')]:
            tp_p = float(levels.get(tp_key, 0))
            if tp_p > 0:
                raw_labels.append((tp_p, f" {tp_lbl} {tp_p:.{dp}f}", tp_color))

        stacked = smc_detector.stack_labels(raw_labels, pair_conf)
        for adj_price, text, color in stacked:
            ax.text(n + 1, adj_price, text, color=color, fontsize=8,
                    va='center', fontweight='bold', zorder=5)

        y_min = float(df_plot['Low'].min())
        y_max = float(df_plot['High'].max())
        sl_p = float(levels.get('sl', 0))
        entry_p = float(levels.get('entry', 0))
        if sl_p > 0:
            y_min = min(y_min, sl_p)
        if entry_p > 0:
            y_max = max(y_max, entry_p)
            y_min = min(y_min, entry_p)
        y_min = min(y_min, zone_lo)
        y_max = max(y_max, zone_hi)
        pad = (y_max - y_min) * 0.08
        ax.set_ylim(y_min - pad, y_max + pad)

        # TP1/TP2 lines: draw only if within visible range, else edge arrow
        for tp_key, tp_color, tp_lbl in [('tp1', '#27ae60', 'TP1'), ('tp2', '#1abc9c', 'TP2')]:
            tp_p = float(levels.get(tp_key, 0))
            if tp_p > 0:
                y_lo, y_hi = ax.get_ylim()
                if y_lo <= tp_p <= y_hi:
                    ax.axhline(y=tp_p, color=tp_color, linestyle='-', linewidth=1.3, alpha=0.9, zorder=4)
                elif tp_p > y_hi:
                    ax.text(n + 1, y_hi - pad * 0.3, f" \u2191 {tp_lbl} {tp_p:.{dp}f}",
                            color=tp_color, fontsize=8, va='top', fontweight='bold', zorder=5)
                elif tp_p < y_lo:
                    ax.text(n + 1, y_lo + pad * 0.3, f" \u2193 {tp_lbl} {tp_p:.{dp}f}",
                            color=tp_color, fontsize=8, va='bottom', fontweight='bold', zorder=5)

        ax.set_xlim(-1, n + 14)
        ax.set_title(title, color='#dddddd', fontsize=11, pad=8, loc='left')
        ax.tick_params(colors='#888', labelsize=8)
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


def build_invalidation_email(pair, bias, reason, reason_detail, ob, levels, alert_ts, ist_now, pair_conf):
    dp = pair_conf.get("decimal_places", 5)
    ist_str = ist_now.strftime('%H:%M IST')
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


def build_trigger_email(pair, bias, ob, levels, m5_fvg, choch_level, pair_conf, ist_now, dollar_risk_str, macro_summary):
    dp = pair_conf.get("decimal_places", 5)
    ist_str = ist_now.strftime('%H:%M IST')
    action_word = "SELL" if bias == "SHORT" else "BUY"

    fvg_line = ""
    if m5_fvg and m5_fvg.get('exists'):
        fvg_line = f"<br>M5 FVG: {m5_fvg.get('fvg_bottom',0):.{dp}f} &rarr; {m5_fvg.get('fvg_top',0):.{dp}f}"

    choch_line = ""
    if choch_level:
        choch_line = f"<br>M5 CHoCH level: {choch_level:.{dp}f}"

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
            <img src="cid:chart_m5" style="width:100%;border-radius:6px;margin-bottom:12px;" />
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
    if alert_ts:
        max_hrs = pair_conf.get("invalidation_time_hours", 24)
        age_hrs = (ist_now - alert_ts).total_seconds() / 3600
        if age_hrs > max_hrs:
            return ("Time expiry", f"Setup has been active {round(age_hrs,1)} hours without trigger (limit: {max_hrs}h).")

    # 3. Opposite H1 BOS
    if alert_ts and df_h1 is not None and smc_detector.check_opposite_bos(df_h1, bias, since_ts=alert_ts):
        return (
            "Opposite H1 structure shift",
            "H1 printed a Break of Structure in the opposite direction since the alert fired. The setup's foundation is broken."
        )

    return (None, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase3():
    ist_now = get_ist_now()
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
        df_m5 = clean_df(yf.download(pair_conf["symbol"], period="3d", interval=trigger_tf, progress=False))
        if df_m5 is None or df_m5.empty:
            print(f"  [!] {pair_name}: no M5 data")
            continue

        current_close = float(df_m5['Close'].iloc[-1])
        m5_atr = smc_detector.compute_atr(df_m5)

        # Fetch H1 once for opposite-BOS check (also reused if invalidated)
        df_h1 = clean_df(yf.download(pair_conf["symbol"], period="10d", interval="1h", progress=False))

        # Invalidation first
        inv_reason, inv_detail = check_invalidation(
            bias, current_close, distal, m5_atr, pair_conf, alert_ts, ist_now, df_h1
        )

        if inv_reason:
            print(f"  [X] {pair_name} INVALIDATED: {inv_reason}")
            html_inv = build_invalidation_email(
                pair_name, bias, inv_reason, inv_detail, ob, levels, alert_ts, ist_now, pair_conf
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

        # M5 CHoCH inside zone bounds
        bounds = {'max': max(proximal, distal), 'min': min(proximal, distal)}
        choch_res = smc_detector.detect_ltf_choch(df_m5, bias, bounds)

        if not choch_res.get("fired"):
            print(f"  [-] {pair_name}: tapped but M5 CHoCH not yet fired")
            continue

        choch_level = choch_res.get("level")
        print(f"  [OK] LTF TRIGGER: {pair_name} M5 CHoCH at {choch_level:.{dp}f}")

        # M5 confluences for chart
        zone_top = max(proximal, distal)
        zone_bottom = min(proximal, distal)
        m5_fvg = smc_detector.detect_fvg_in_zone(df_m5, bias, zone_top, zone_bottom)

        swings_m5 = smc_detector.get_swing_points(df_m5, lookback=3)
        _, m5_sweep_price = smc_detector.detect_sweep_decay(df_m5, swings_m5, len(df_m5) - 1)

        # Recompute levels with fresh M5 FVG
        fresh_levels = smc_detector.compute_dynamic_levels(pair_conf, bias, ob, m5_fvg, current_close, df_m5)
        if not fresh_levels['valid']:
            print(f"  [!] {pair_name}: CHoCH fired but RR invalid. Skipping.")
            continue

        chart_b64 = generate_m5_chart(
            df_m5, f"{pair_name} M5 - Sniper Trigger",
            fresh_levels, ob, pair_conf, m5_fvg, choch_level, m5_sweep_price
        )
        html = build_trigger_email(
            pair_name, bias, ob, fresh_levels, m5_fvg, choch_level, pair_conf,
            ist_now, dollar_risk_str, data.get("macro_summary")
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
