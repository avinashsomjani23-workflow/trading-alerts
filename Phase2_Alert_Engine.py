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
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "dummy@gmail.com")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD", "dummy")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


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
    You are a strict Risk Management AI. DO NOT analyze the chart. DO NOT calculate math.
    TRADE DETAILS: Pair: {pair} | Direction: {bias}
    RECENT NEWS: {news_headlines}

    TASK:
    1. Identify any Tier-1 economic events (e.g., CPI, NFP) affecting {pair}.
    2. Assign a macro_score: 1.0 if safe, 0.0 if a high-impact event is imminent.

    OUTPUT FORMAT (Strict JSON):
    {{
        "high_impact_news_detected": boolean,
        "macro_score": float,
        "macro_summary": "Exactly 2 concise sentences summarizing the risk specific to {pair}."
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
    for _ in range(3):
        try:
            r = requests.post(url, json=body, timeout=20).json()
            if "candidates" in r:
                return json.loads(r["candidates"][0]["content"]["parts"][0]["text"].strip())
        except Exception:
            time.sleep(3)
    return {"macro_score": 1.0, "macro_summary": "Gemini API failed. Defaulting to safe."}


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


def generate_h1_chart(df_h1, ob, pair_conf, title):
    try:
        dp = pair_conf.get("decimal_places", 5)
        df_plot = df_h1.dropna(subset=['Open', 'High', 'Low', 'Close']).tail(80).copy().reset_index(drop=True)
        n = len(df_plot)
        if n < 5:
            return None

        fig, ax = _base_canvas()
        _draw_candles(ax, df_plot)

        proximal = float(ob.get('proximal_line', 0))
        distal = float(ob.get('distal_line', 0))
        zone_hi, zone_lo = max(proximal, distal), min(proximal, distal)

        if zone_hi > 0 and zone_lo > 0:
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                facecolor='#9b59b6', alpha=0.15, zorder=1
            ))
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                fill=False, edgecolor='#bb8fce', linestyle=':', linewidth=1.5, zorder=2
            ))
            ax.text(n + 1, proximal, f" P {proximal:.{dp}f}", color='#bb8fce', fontsize=8, va='center', zorder=5)
            ax.text(n + 1, distal, f" D {distal:.{dp}f}", color='#bb8fce', fontsize=8, va='center', zorder=5)

        bos_price = float(ob.get('bos_swing_price', 0))
        bos_tag = ob.get('bos_tag', 'BOS')
        if bos_price > 0:
            bos_color = '#00bcd4' if bos_tag == 'BOS' else '#ff9800'
            ax.axhline(y=bos_price, color=bos_color, linewidth=0.8, linestyle='--', alpha=0.7, zorder=2)
            ax.text(n + 1, bos_price, f" {bos_tag} {bos_price:.{dp}f}",
                    color=bos_color, fontsize=8, va='center', fontweight='bold', zorder=5)

        fvg = ob.get('fvg', {}) or {}
        if fvg.get('exists'):
            ft, fb = float(fvg.get('fvg_top', 0)), float(fvg.get('fvg_bottom', 0))
            if ft > 0 and fb > 0:
                ax.add_patch(patches.Rectangle(
                    (0, fb), n + 5, ft - fb,
                    facecolor='#27ae60', alpha=0.2, zorder=1
                ))

        current = float(df_plot['Close'].iloc[-1])
        ax.axhline(y=current, color='#ffffff', linewidth=0.8, linestyle='-', alpha=0.5, zorder=2)
        ax.text(n + 1, current, f" {current:.{dp}f}", color='#ffffff', fontsize=8, va='center', zorder=5)

        y_min, y_max = float(df_plot['Low'].min()), float(df_plot['High'].max())
        pad = (y_max - y_min) * 0.08
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlim(-1, n + 10)
        ax.set_title(title, color='#dddddd', fontsize=11, pad=8, loc='left')
        ax.tick_params(colors='#888', labelsize=8)
        ax.yaxis.tick_right()
        ax.set_xticks([])
        plt.tight_layout(pad=0.5)
        return _fig_to_b64(fig)
    except Exception as e:
        print(f"H1 chart error: {e}")
        plt.close('all')
        return None


def generate_m15_chart(df_m15, title, levels, ob, pair_conf, fvg_data, sweep_price):
    try:
        dp = pair_conf.get("decimal_places", 5)
        df_plot = df_m15.dropna(subset=['Open', 'High', 'Low', 'Close']).tail(60).copy().reset_index(drop=True)
        n = len(df_plot)
        if n < 5:
            return None

        fig, ax = _base_canvas()
        _draw_candles(ax, df_plot)

        proximal = float(ob.get('proximal_line', 0))
        distal = float(ob.get('distal_line', 0))
        zone_hi, zone_lo = max(proximal, distal), min(proximal, distal)

        if zone_hi > 0 and zone_lo > 0:
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                facecolor='#9b59b6', alpha=0.15, zorder=1
            ))
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                fill=False, edgecolor='#bb8fce', linestyle=':', linewidth=1.5, zorder=2
            ))

        if fvg_data and fvg_data.get('exists'):
            ft, fb = float(fvg_data.get('fvg_top', 0)), float(fvg_data.get('fvg_bottom', 0))
            if ft > 0 and fb > 0:
                ax.add_patch(patches.Rectangle(
                    (0, fb), n + 5, ft - fb,
                    facecolor='#27ae60', alpha=0.2, zorder=1
                ))
                ax.text(0.5, ft, ' FVG', color='#2ecc71', fontsize=8, va='bottom', zorder=5)

        if sweep_price is not None:
            ax.scatter([n - 2], [sweep_price], color='#ff5370', marker='x', s=120, linewidth=2, zorder=5)
            ax.text(n - 2, sweep_price, ' sweep', color='#ff5370', fontsize=8, va='bottom', zorder=5)

        level_styles = {
            'tp1': ('#27ae60', 'TP1'),
            'entry': ('#e67e22', 'ENTRY'),
            'sl': ('#e74c3c', 'SL')
        }
        for key, (color, lbl) in level_styles.items():
            price = float(levels.get(key, 0))
            if price > 0:
                ax.axhline(y=price, color=color, linestyle='-', linewidth=1.3, alpha=0.85, zorder=4)
                ax.text(n + 1, price, f" {lbl} {price:.{dp}f}",
                        color=color, fontsize=8, va='center', fontweight='bold', zorder=5)

        y_min, y_max = float(df_plot['Low'].min()), float(df_plot['High'].max())
        sl_p = float(levels.get('sl', 0))
        tp1_p = float(levels.get('tp1', 0))
        if sl_p > 0:
            y_min = min(y_min, sl_p)
        if tp1_p > 0:
            y_max = max(y_max, tp1_p)
        pad = (y_max - y_min) * 0.08
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlim(-1, n + 10)
        ax.set_title(title, color='#dddddd', fontsize=11, pad=8, loc='left')
        ax.tick_params(colors='#888', labelsize=8)
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

def build_scorecard_html(rows, total):
    body = ""
    for label, score, max_score, status, expl in rows:
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
        Confluence Scorecard &mdash; <span style="color:#eee;font-size:14px;font-weight:bold;">{total}/10</span>
      </div>
      <table style="width:100%;border-collapse:collapse;background:#1a1a2e;border-radius:6px;">
        <tbody>{body}</tbody>
      </table>
    </div>"""


def build_trade_email(data, pair, pair_conf, state_msg, scorecard_rows, total_score,
                      atr_label, distance_str, dollar_risk_str):
    dp = pair_conf.get("decimal_places", 5)
    bias = data.get("bias", "-")
    ist_time = get_ist_now().strftime('%H:%M IST')
    ob = data.get('ob', {})
    levels = data.get('levels', {})
    fresh = "Pristine" if ob.get('touches', 0) == 0 else f"Tested {ob.get('touches', 0)}x"
    bos_tag = ob.get('bos_tag', 'BOS')
    entry_model = pair_conf.get('entry_model', 'limit')

    if entry_model == "limit":
        action_word = "SELL LIMIT" if bias == "SHORT" else "BUY LIMIT"
        action_block = f"""
            <div style="background:#27ae60;padding:14px 18px;border-radius:10px;margin-bottom:14px;">
                <p style="color:white;font-size:15px;font-weight:bold;margin:0;">{action_word} at {levels.get('entry'):,.{dp}f}</p>
                <p style="color:white;margin:4px 0 0;font-size:12px;">
                    SL: {levels.get('sl'):,.{dp}f} &nbsp;|&nbsp;
                    TP1: {levels.get('tp1'):,.{dp}f} &nbsp;|&nbsp;
                    Risk: {dollar_risk_str}
                </p>
            </div>"""
    else:
        proximal = float(ob.get('proximal_line', 0))
        distal = float(ob.get('distal_line', 0))
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
                    Risk: {dollar_risk_str}
                </p>
                <p style="color:white;margin:4px 0 0;font-size:11px;opacity:0.85;">
                    Final entry will be confirmed at M5 CHoCH inside the zone.
                </p>
            </div>"""

    scorecard_html = build_scorecard_html(scorecard_rows, total_score)

    distance_html = f"""
    <div style="margin-bottom:12px;padding:8px 12px;background:#0d0d1a;border-left:3px solid #00bcd4;border-radius:4px;font-size:12px;color:#bbb;">
        <b style="color:#eee;">Distance:</b> {distance_str} &nbsp;&middot;&nbsp; {atr_label}
    </div>"""

    context_html = f"""
    <div style="margin-bottom:12px;font-size:11px;color:#888;">
        <b style="color:#aaa;">Zone:</b> {bos_tag} &nbsp;&middot;&nbsp; {fresh}
        &nbsp;&middot;&nbsp; Proximal {ob.get('proximal_line', 0):.{dp}f}
        / Distal {ob.get('distal_line', 0):.{dp}f}
    </div>"""

    return f"""<html><body style="font-family:Arial,sans-serif;background:#0d0d1a;padding:12px;margin:0;">
    <div style="max-width:650px;margin:auto;background:#13131f;border-radius:14px;overflow:hidden;">
        <div style="background:#1a1a2e;padding:14px 18px;">
            <h2 style="color:#eee;margin:0;font-size:16px;">{state_msg}: {pair} &middot; {bias}</h2>
            <p style="color:#888;margin:4px 0 0;font-size:11px;">{ist_time}</p>
        </div>
        <div style="padding:14px 18px;">
            {action_block}
            {distance_html}
            {context_html}
            {scorecard_html}
            <div style="margin:14px 0 6px 0;color:#aaa;font-size:11px;letter-spacing:1px;text-transform:uppercase;">H1 Context</div>
            <img src="cid:chart_h1" style="width:100%;border-radius:6px;margin-bottom:12px;" />
            <div style="margin:14px 0 6px 0;color:#aaa;font-size:11px;letter-spacing:1px;text-transform:uppercase;">M15 Approach</div>
            <img src="cid:chart_m15" style="width:100%;border-radius:6px;margin-bottom:12px;" />
            <div style="margin-top:12px;padding:10px 12px;background:#0d0d1a;border-left:3px solid #888;border-radius:4px;font-size:12px;color:#bbb;line-height:1.5;">
                <b style="color:#eee;">Macro Context:</b> {data.get('macro_summary', 'N/A')}
            </div>
        </div>
    </div></body></html>"""


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
            print(f"Email failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ist_now = get_ist_now()
    print(f"Phase 2 Engine started {ist_now.strftime('%H:%M')} IST")

    active_obs = load_json("active_obs.json", {})
    watch_state = load_json("active_watch_state.json", {})
    phase2_sent = load_json("phase2_sent.json", {})
    new_watch_state = dict(watch_state)

    balance = config["account"]["balance"]
    risk_pct = config["account"]["risk_percent"]
    dollar_risk = balance * (risk_pct / 100.0)
    dollar_risk_str = f"${dollar_risk:,.0f}"

    for pair_conf in config["pairs"]:
        symbol = pair_conf["symbol"]
        name = pair_conf["name"]
        entry_model = pair_conf.get("entry_model", "limit")
        pair_obs = active_obs.get(name, [])
        if not pair_obs:
            continue

        radar_tf = pair_conf.get("radar_tf", "15m")
        df_m15 = clean_df(yf.download(symbol, period="5d", interval=radar_tf, progress=False))
        df_h1 = clean_df(yf.download(symbol, period="15d", interval="1h", progress=False))
        if df_m15 is None or df_h1 is None:
            continue

        current_price = float(df_m15['Close'].iloc[-1])
        h1_atr = smc_detector.compute_atr(df_h1)
        m15_atr = smc_detector.compute_atr(df_m15)
        if not h1_atr:
            continue

        for ob in pair_obs:
            proximal = float(ob['proximal_line'])
            if abs(current_price - proximal) > (pair_conf["atr_multiplier"] * h1_atr):
                continue

            bias = "LONG" if ob['direction'] == 'bullish' else "SHORT"
            zone_top = max(proximal, float(ob['distal_line']))
            zone_bottom = min(proximal, float(ob['distal_line']))

            fvg_data = smc_detector.detect_fvg_in_zone(df_m15, bias, zone_top, zone_bottom)
            if not fvg_data['exists']:
                fvg_data = ob.get("fvg", {"exists": False})

            gemini_risk = call_gemini_flash(name, bias, fetch_macro_news(name))
            macro_score = gemini_risk.get('macro_score', 1.0)

            score_res = smc_detector.run_scorecard(
                bias, df_h1, ob, fvg_data, current_price, pair_conf, df_m15, macro_score
            )
            if score_res['total'] < pair_conf["min_confidence"]:
                continue

            levels = smc_detector.compute_dynamic_levels(pair_conf, bias, ob, fvg_data, current_price, df_m15)
            if not levels['valid']:
                continue

            scorecard_rows = smc_detector.generate_scorecard_rows(
                bias, score_res['breakdown'], ob,
                score_res.get('sweep_price'), score_res.get('sweep_tf', 'H1'), pair_conf
            )

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
                "macro_summary": gemini_risk.get("macro_summary", ""),
                "levels": levels,
                "ob": ob,
                "alert_ist": ist_now.isoformat()
            }

            if entry_model == "limit":
                zone_id = f"{name}_{bias}_{proximal}"
                if zone_id in phase2_sent:
                    print(f"  [-] {name}: already alerted (dedup). Skipping.")
                    continue

                phase2_sent[zone_id] = ist_now.isoformat()

                h1_chart = generate_h1_chart(df_h1, ob, pair_conf, f"{name} H1 - {bias} zone context")
                m15_chart = generate_m15_chart(
                    df_m15, f"{name} M15 - Approach and entry",
                    levels, ob, pair_conf, fvg_data, score_res.get('sweep_price')
                )
                html = build_trade_email(
                    trade_data, name, pair_conf, "TRADE READY",
                    scorecard_rows, score_res['total'],
                    atr_label, distance_str, dollar_risk_str
                )
                send_email(
                    f"TRADE READY | {name} | {bias} | {ist_now.strftime('%H:%M IST')}",
                    html, h1_chart, m15_chart
                )
                print(f"  [OK] TRADE READY (FOREX): {name}")
            elif entry_model == "ltf_choch":
                zone_id = f"{name}_{bias}_{proximal}"
                watch_id = f"{name}_{proximal}"

                if zone_id in phase2_sent:
                    # Already alerted. Refresh watch_state but preserve original alert timestamp.
                    if watch_id in new_watch_state:
                        trade_data["alert_ist"] = new_watch_state[watch_id].get(
                            "alert_ist", ist_now.isoformat()
                        )
                    new_watch_state[watch_id] = trade_data
                    continue

                phase2_sent[zone_id] = ist_now.isoformat()
                h1_chart = generate_h1_chart(df_h1, ob, pair_conf, f"{name} H1 - {bias} zone context")
                m15_chart = generate_m15_chart(
                    df_m15, f"{name} M15 - Approach",
                    levels, ob, pair_conf, fvg_data, score_res.get('sweep_price')
                )
                html = build_trade_email(
                    trade_data, name, pair_conf, "APPROACHING",
                    scorecard_rows, score_res['total'],
                    atr_label, distance_str, dollar_risk_str
                )
                send_email(
                    f"APPROACHING | {name} | {bias} | {ist_now.strftime('%H:%M IST')}",
                    html, h1_chart, m15_chart
                )

                new_watch_state[watch_id] = trade_data
                print(f"  [>] LOGGED FOR PHASE 3: {name}")

    save_json("phase2_sent.json", phase2_sent)
    save_json("active_watch_state.json", new_watch_state)
    print("Phase 2 complete.")
