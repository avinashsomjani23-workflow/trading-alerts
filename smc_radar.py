import yfinance as yf
import pandas as pd
import numpy as np
import json
import smtplib
import logging
import os
import google.generativeai as genai
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

logging.basicConfig(
    filename="smc_radar.log", level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

with open("config.json") as f:
    config_master = json.load(f)

EMAIL_CONFIG = {
    "sender":      "avinash.somjani23@gmail.com",
    "recipient":   config_master["account"].get("alert_emails", ["avinash.somjani23@gmail.com"]),
    "smtp_server": "smtp.gmail.com",
    "smtp_port":   587,
    "password":    os.environ.get("GMAIL_APP_PASSWORD", "dummy")
}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# ZONE IDENTITY THRESHOLDS — price-range overlap matching (not string match)
# ---------------------------------------------------------------------------
ZONE_PROXIMITY_THRESHOLDS = {
    "forex":     0.00030,   # 3 pips
    "index":     30.0,      # 30 points NAS100
    "commodity": 3.0        # $3 Gold
}

# ---------------------------------------------------------------------------
# TIME HELPERS
# ---------------------------------------------------------------------------

def get_ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def get_day_start_ist():
    """Return today's 09:00 IST as isoformat string — daily slate boundary."""
    now = get_ist_now()
    if now.hour >= 9:
        return now.replace(hour=9, minute=0, second=0, microsecond=0).isoformat()
    else:
        return (now - timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0).isoformat()

# ---------------------------------------------------------------------------
# ZONE STATE PERSISTENCE
# emailed_zones.json structure:
# {
#   "EURUSD": [
#     {
#       "proximal": 1.15487, "distal": 1.15407, "direction": "bullish",
#       "bos_tag": "BOS", "first_seen_ist": "2026-04-14T09:00:00",
#       "last_seen_ist": "2026-04-14T11:00:00",
#       "touches": 0, "fvg_valid": true, "status": "Pristine",
#       "changes": []
#     }, ...
#   ], ...
# }
# ---------------------------------------------------------------------------

def load_zone_state():
    try:
        with open("emailed_zones.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_zone_state(state):
    with open("emailed_zones.json", "w") as f:
        json.dump(state, f, indent=2)

def zones_match(ob, stored, pair_type):
    """True if ob and stored zone are the same zone (price-range overlap check)."""
    if ob["direction"] != stored["direction"]:
        return False
    threshold = ZONE_PROXIMITY_THRESHOLDS.get(pair_type, 0.0003)
    return abs(ob["proximal_line"] - stored["proximal"]) <= threshold

def detect_zone_changes(ob, stored, dp):
    """
    Compare current scan result vs stored zone state.
    Returns list of plain-English change strings. Empty list = no change.
    """
    changes = []
    if ob["touches"] > stored["touches"]:
        diff = ob["touches"] - stored["touches"]
        new_status = ob["status"]
        changes.append(
            f"New touch detected (+{diff}). Status now: {new_status}."
        )
    if ob["fvg"]["exists"] != stored["fvg_valid"]:
        if not ob["fvg"]["exists"] and stored["fvg_valid"]:
            changes.append("FVG mitigated since last scan.")
        elif ob["fvg"]["exists"] and not stored["fvg_valid"]:
            changes.append("FVG now confirmed (was absent).")
    return changes

# ---------------------------------------------------------------------------
# DATA FETCH
# ---------------------------------------------------------------------------

def fetch_data(ticker, interval, period):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df.tail(150).copy().reset_index()

# ---------------------------------------------------------------------------
# SMC DETECTION
# ---------------------------------------------------------------------------

def is_valid_ob_candle(open_p, close_p, high_p, low_p):
    body = abs(open_p - close_p)
    rng  = high_p - low_p
    if rng == 0:
        return False
    return body > (rng * 0.15)

def detect_smc_radar(df, lookback):
    n = len(df)
    O = df['Open'].values
    C = df['Close'].values
    H = df['High'].values
    L = df['Low'].values

    swings = []
    for i in range(lookback, n - lookback):
        window_highs = H[i - lookback: i + lookback + 1]
        window_lows  = L[i - lookback: i + lookback + 1]
        if H[i] == max(window_highs):
            swings.append({'type': 'high', 'idx': i, 'price': float(H[i])})
        elif L[i] == min(window_lows):
            swings.append({'type': 'low',  'idx': i, 'price': float(L[i])})

    swings = sorted(swings, key=lambda x: x['idx'])
    active_obs = []
    trend_state = None

    for i in range(lookback + 1, n):
        past_swings = [s for s in swings if s['idx'] < i]
        if len(past_swings) < 2:
            continue
        latest_high = [s for s in past_swings if s['type'] == 'high']
        latest_low  = [s for s in past_swings if s['type'] == 'low']
        if not latest_high or not latest_low:
            continue

        sh, sl = latest_high[-1], latest_low[-1]
        bos_detected, bos_type = False, None

        if C[i] > sh['price'] and C[i - 1] <= sh['price']:
            bos_detected, bos_type = True, 'bullish'
        elif C[i] < sl['price'] and C[i - 1] >= sl['price']:
            bos_detected, bos_type = True, 'bearish'

        if not bos_detected:
            continue

        bos_tag = 'CHoCH' if (trend_state is None or trend_state != bos_type) else 'BOS'
        trend_state = bos_type
        bos_swing_price = sh['price'] if bos_type == 'bullish' else sl['price']

        ob_idx = -1
        impulse_start_idx = sl['idx'] if bos_type == 'bullish' else sh['idx']
        if impulse_start_idx >= i:
            continue

        leg_bodies = [abs(C[k] - O[k]) for k in range(impulse_start_idx, i + 1)]
        median_leg_body = float(np.median(leg_bodies)) if leg_bodies else 0.0001
        if median_leg_body == 0:
            median_leg_body = 0.0001

        for j in range(i - 1, impulse_start_idx - 1, -1):
            if (bos_type == 'bullish' and C[j] < O[j]) or \
               (bos_type == 'bearish' and C[j] > O[j]):
                if is_valid_ob_candle(O[j], C[j], H[j], L[j]):
                    if abs(C[j] - O[j]) <= (2.0 * median_leg_body):
                        ob_idx = j
                        break

        if ob_idx == -1:
            continue

        ob_high = float(H[ob_idx])
        ob_low  = float(L[ob_idx])

        # FVG detection — store candle indices for chart positioning
        fvg_valid = False
        fvg_top = fvg_bottom = None
        fvg_c1_idx = fvg_c3_idx = None
        c2_idx = ob_idx + 1
        max_c5_idx = min(ob_idx + 5, i + 1, n - 1)

        for k in range(c2_idx, max_c5_idx):
            if k + 2 >= n:
                break
            if bos_type == 'bullish' and H[k] < L[k + 2]:
                fvg_top    = float(L[k + 2])
                fvg_bottom = float(H[k])
                filled = any(L[m] <= fvg_bottom for m in range(k + 3, n))
                if not filled:
                    fvg_valid   = True
                    fvg_c1_idx  = k
                    fvg_c3_idx  = k + 2
                break
            elif bos_type == 'bearish' and L[k] > H[k + 2]:
                fvg_top    = float(L[k])
                fvg_bottom = float(H[k + 2])
                filled = any(H[m] >= fvg_top for m in range(k + 3, n))
                if not filled:
                    fvg_valid   = True
                    fvg_c1_idx  = k
                    fvg_c3_idx  = k + 2
                break

        active_obs.append({
            'bos_idx':         i,
            'bos_swing_price': bos_swing_price,
            'ob_idx':          ob_idx,
            'direction':       bos_type,
            'bos_tag':         bos_tag,
            'high':            ob_high,
            'low':             ob_low,
            'proximal_line':   ob_high if bos_type == 'bullish' else ob_low,
            'distal_line':     ob_low  if bos_type == 'bullish' else ob_high,
            'median_leg_body': median_leg_body,
            'ob_body':         abs(C[ob_idx] - O[ob_idx]),
            'fvg': {
                'exists':     fvg_valid,
                'fvg_top':    fvg_top,
                'fvg_bottom': fvg_bottom,
                'c1_idx':     fvg_c1_idx,
                'c3_idx':     fvg_c3_idx
            }
        })

    # Mitigation + touch tracking
    tracked_obs = []
    for ob in active_obs:
        mitigated = False
        touches   = 0
        for m in range(ob['ob_idx'] + 2, n):
            if ob['direction'] == 'bullish':
                if C[m] < ob['distal_line']:
                    mitigated = True
                    break
                elif L[m] <= ob['proximal_line']:
                    touches += 1
            else:
                if C[m] > ob['distal_line']:
                    mitigated = True
                    break
                elif H[m] >= ob['proximal_line']:
                    touches += 1
            if touches > 3:
                mitigated = True
                break

        if not mitigated:
            ob['touches'] = touches
            ob['status']  = 'Pristine' if touches == 0 else f'Tested ({touches}x)'
            tracked_obs.append(ob)

    return {"current_price": float(C[-1]), "active_unmitigated_obs": tracked_obs}

# ---------------------------------------------------------------------------
# CHART GENERATION
# ---------------------------------------------------------------------------

def generate_h1_chart(df, ob, dp, pair_name, ist_timestamp):
    """
    Generates a clean SMC chart showing:
    - Price context (40 candles or enough to include OB candle)
    - Zone band (dotted rectangle, starts at OB candle)
    - OB candle highlight (purple border box)
    - FVG rectangle (green, at correct candle positions)
    - BOS/CHoCH horizontal line
    - Current price line with label
    - Proximal/Distal labels
    - IST timestamp in chart title
    """
    try:
        full_df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy().reset_index(drop=True)
        n_full  = len(full_df)
        ob_abs  = ob['ob_idx']

        # Window: 40 candles back from end, but must include OB candle
        window_start = max(0, n_full - 40)
        if ob_abs < window_start:
            window_start = max(0, ob_abs - 5)   # 5 candles of left-context before OB
        window_start = min(window_start, n_full - 1)

        df_plot = full_df.iloc[window_start:].copy().reset_index(drop=True)
        n_plot  = len(df_plot)

        # OB candle position in plot window
        ob_plot_idx = ob_abs - window_start

        O = df_plot['Open'].values
        C = df_plot['Close'].values
        H = df_plot['High'].values
        L = df_plot['Low'].values

        fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), facecolor='#131722')
        ax.set_facecolor('#131722')
        for spine in ax.spines.values():
            spine.set_color('#2a2a3e')

        # --- Candles ---
        for i in range(n_plot):
            o, h, l, c = float(O[i]), float(H[i]), float(L[i]), float(C[i])
            col = '#26a69a' if c >= o else '#ef5350'
            ax.plot([i, i], [l, h], color=col, linewidth=1.2, zorder=2)
            body = abs(c - o) or (h - l) * 0.02
            ax.add_patch(patches.Rectangle(
                (i - 0.4, min(o, c)), 0.8, body,
                facecolor=col, linewidth=0, alpha=0.9, zorder=3
            ))

        proximal = float(ob['proximal_line'])
        distal   = float(ob['distal_line'])
        zone_lo  = min(proximal, distal)
        zone_hi  = max(proximal, distal)

        # --- Zone band: starts at OB candle, extends to right edge ---
        zone_x_start = max(0, ob_plot_idx - 0.5)
        zone_width   = (n_plot + 2) - zone_x_start
        ax.add_patch(patches.Rectangle(
            (zone_x_start, zone_lo), zone_width, zone_hi - zone_lo,
            facecolor='#9b59b6', alpha=0.12, zorder=1
        ))
        ax.add_patch(patches.Rectangle(
            (zone_x_start, zone_lo), zone_width, zone_hi - zone_lo,
            fill=False, edgecolor='#bb8fce', linestyle=':', linewidth=1.5, zorder=2
        ))

        # --- OB candle highlight: purple border around OB candle full range ---
        if 0 <= ob_plot_idx < n_plot:
            ob_h = float(H[ob_plot_idx])
            ob_l = float(L[ob_plot_idx])
            ax.add_patch(patches.Rectangle(
                (ob_plot_idx - 0.5, ob_l), 1.0, ob_h - ob_l,
                fill=False, edgecolor='#d7bde2', linewidth=2.0, zorder=4,
                linestyle='-'
            ))
            # Small OB label
            ax.text(
                ob_plot_idx, ob_h, 'OB',
                color='#d7bde2', fontsize=7, ha='center', va='bottom',
                fontweight='bold', zorder=5
            )

        # --- FVG rectangle: exact candle positions ---
        if ob['fvg']['exists'] and ob['fvg']['c1_idx'] is not None:
            ft = float(ob['fvg']['fvg_top'])
            fb = float(ob['fvg']['fvg_bottom'])
            fvg_c1_plot = ob['fvg']['c1_idx'] - window_start
            fvg_c3_plot = ob['fvg']['c3_idx'] - window_start
            if 0 <= fvg_c1_plot < n_plot:
                fvg_x_start = fvg_c1_plot - 0.4
                fvg_width   = (fvg_c3_plot + 0.4) - fvg_x_start
                ax.add_patch(patches.Rectangle(
                    (fvg_x_start, fb), fvg_width, ft - fb,
                    facecolor='#27ae60', alpha=0.25, zorder=1
                ))
                ax.add_patch(patches.Rectangle(
                    (fvg_x_start, fb), fvg_width, ft - fb,
                    fill=False, edgecolor='#2ecc71', linewidth=1.0,
                    linestyle='--', zorder=2
                ))
                ax.text(
                    fvg_c1_plot, ft, 'FVG',
                    color='#2ecc71', fontsize=7, ha='left', va='bottom', zorder=5
                )

        # --- BOS/CHoCH line ---
        bos_price = float(ob['bos_swing_price'])
        bos_color = '#00bcd4' if ob['bos_tag'] == 'BOS' else '#ff9800'
        ax.axhline(
            y=bos_price, color=bos_color, linewidth=0.8,
            linestyle='--', alpha=0.7, zorder=2
        )
        ax.text(
            n_plot + 1.5, bos_price, ob['bos_tag'],
            color=bos_color, fontsize=7, va='center', fontweight='bold', zorder=5
        )

        # --- Current price line ---
        current_price = float(C[-1])
        ax.axhline(
            y=current_price, color='#ffffff', linewidth=0.8,
            linestyle='-', alpha=0.5, zorder=2
        )
        ax.text(
            n_plot + 1.5, current_price,
            f"{current_price:.{dp}f}",
            color='#ffffff', fontsize=7, va='center', zorder=5
        )

        # --- Proximal / Distal labels ---
        ax.text(
            n_plot + 1.5, proximal,
            f"P {proximal:.{dp}f}",
            color='#bb8fce', fontsize=7, va='center', zorder=5
        )
        ax.text(
            n_plot + 1.5, distal,
            f"D {distal:.{dp}f}",
            color='#bb8fce', fontsize=7, va='center', zorder=5
        )

        # --- Y-axis: full candle range with 8% padding ---
        y_min = float(np.min(L))
        y_max = float(np.max(H))
        pad   = (y_max - y_min) * 0.08
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlim(-1, n_plot + 10)

        # --- Title with IST timestamp ---
        direction_label = "Demand" if ob['direction'] == 'bullish' else "Supply"
        title = (
            f"{pair_name} | {direction_label} Zone | {ob['bos_tag']} | "
            f"{ob['status']}   —   {ist_timestamp} IST"
        )
        ax.set_title(title, color='#dddddd', fontsize=10, pad=8, loc='left')
        ax.tick_params(colors='#888', labelsize=8)
        ax.yaxis.tick_right()
        ax.set_xticks([])

        plt.tight_layout(pad=0.5)
        buf = BytesIO()
        fig.savefig(
            buf, format='png', dpi=150, bbox_inches='tight',
            facecolor='#131722', edgecolor='none'
        )
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return b64

    except Exception as e:
        logging.error(f"Chart error: {e}")
        plt.close('all')
        return None

# ---------------------------------------------------------------------------
# GEMINI NARRATIVE — new zones only
# ---------------------------------------------------------------------------

def generate_zone_narrative(ob, name, dp, current_price):
    """
    Calls Gemini 2.5 Flash (no thinking) to produce a 4-sentence plain-English
    zone briefing. All math pre-computed in Python — Gemini writes text only.
    Returns string. Falls back to Python-generated text on any error.
    """
    if not GEMINI_API_KEY:
        return _fallback_narrative(ob, name, dp, current_price)

    direction    = "bullish demand" if ob['direction'] == 'bullish' else "bearish supply"
    proximal     = ob['proximal_line']
    distal       = ob['distal_line']
    pip_unit     = 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)
    zone_pips    = round(abs(proximal - distal) / pip_unit, 1)
    dist_pips    = round(abs(current_price - proximal) / pip_unit, 1)
    fvg_status   = (
        f"confirmed FVG between {ob['fvg']['fvg_bottom']:.{dp}f} "
        f"and {ob['fvg']['fvg_top']:.{dp}f}"
        if ob['fvg']['exists']
        else "no FVG present"
    )
    ratio        = round(ob['ob_body'] / ob['median_leg_body'], 2) if ob['median_leg_body'] > 0 else 0

    prompt = f"""You are a veteran SMC (Smart Money Concepts) prop trader writing a zone briefing for another experienced SMC trader.
Be direct. No fluff. No pleasantries. Four sentences only. One paragraph.

ZONE DATA — use these exact values, do not recalculate:
- Pair: {name}
- Bias: {direction} | Structure event: {ob['bos_tag']}
- Proximal: {proximal:.{dp}f} | Distal: {distal:.{dp}f}
- Zone width: {zone_pips} pips
- OB body vs median impulse leg: {ob['ob_body']:.{dp}f} vs {ob['median_leg_body']:.{dp}f} (ratio: {ratio}x — valid because <2.0x)
- FVG: {fvg_status}
- Zone status: {ob['status']}
- Current price: {current_price:.{dp}f} | Distance to proximal: {dist_pips} pips

WRITE EXACTLY FOUR SENTENCES IN THIS ORDER:
1. What structure event ({ob['bos_tag']}) created this zone and why institutional accumulation is likely here.
2. OB quality: assess tightness (ratio {ratio}x), and whether pristine or tested means strength or caution.
3. FVG assessment: displacement confirmation present or absent, and what that means for zone conviction.
4. Current price context: distance to zone ({dist_pips} pips), whether price is approaching or far, and what to watch for.

STRICT OUTPUT RULES:
- Plain text only
- No bullet points, no headers, no markdown, no bold, no numbers
- Four sentences, one paragraph
- Do not repeat the zone levels in every sentence"""

    try:
        model    = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=300,
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
            )
        )
        response = model.generate_content(prompt)
        text     = response.text.strip()
        if len(text) < 50:
            return _fallback_narrative(ob, name, dp, current_price)
        return text
    except Exception as e:
        logging.error(f"Gemini narrative error for {name}: {e}")
        return _fallback_narrative(ob, name, dp, current_price)

def _fallback_narrative(ob, name, dp, current_price):
    """Python-only fallback if Gemini is unavailable."""
    pip_unit  = 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)
    dist_pips = round(abs(current_price - ob['proximal_line']) / pip_unit, 1)
    fvg_line  = (
        f"FVG confirmed between {ob['fvg']['fvg_bottom']:.{dp}f}–{ob['fvg']['fvg_top']:.{dp}f}, adding displacement confluence."
        if ob['fvg']['exists']
        else "No FVG present — zone relies on OB alone for confluence."
    )
    return (
        f"{ob['bos_tag']} confirmed the {ob['direction']} shift; this OB marks the last institutional "
        f"accumulation before the break. "
        f"OB body ratio {round(ob['ob_body']/ob['median_leg_body'],2):.2f}x vs median leg — tight and valid. "
        f"{fvg_line} "
        f"Current price is {dist_pips} pips from proximal — "
        f"{'approaching zone, watch for reaction.' if dist_pips < 50 else 'still distant, no action yet.'}"
    )

# ---------------------------------------------------------------------------
# EMAIL ASSEMBLY
# ---------------------------------------------------------------------------

def build_summary_table_html(all_zones_for_table, dp_map):
    """
    Builds the top-level summary table: one row per zone, all pairs.
    Columns: Pair | Bias | Levels | Status | FVG | Since
    """
    rows = ""
    for z in all_zones_for_table:
        name      = z['name']
        dp        = dp_map[name]
        direction = "▲ Bullish" if z['direction'] == 'bullish' else "▼ Bearish"
        dir_color = '#27ae60'   if z['direction'] == 'bullish' else '#e74c3c'
        status    = z['status']
        stat_col  = '#27ae60'   if 'Pristine' in status else '#e67e22'
        fvg_cell  = "✓" if z['fvg_valid'] else "–"
        fvg_col   = '#27ae60' if z['fvg_valid'] else '#888'
        since     = z['first_seen_ist']
        tag_badge = (
            f"<span style='background:#00bcd4;color:#000;font-size:9px;"
            f"padding:1px 4px;border-radius:3px;font-weight:bold;'>{z['bos_tag']}</span>"
            if z['bos_tag'] == 'BOS' else
            f"<span style='background:#ff9800;color:#000;font-size:9px;"
            f"padding:1px 4px;border-radius:3px;font-weight:bold;'>{z['bos_tag']}</span>"
        )
        is_new    = z.get('is_new', False)
        is_changed = z.get('is_changed', False)
        row_bg    = '#1e3a2f' if is_new else ('#2d2a1a' if is_changed else 'transparent')
        new_badge = ""
        if is_new:
            new_badge = "<span style='background:#27ae60;color:#fff;font-size:9px;padding:1px 4px;border-radius:3px;margin-left:4px;'>NEW</span>"
        elif is_changed:
            new_badge = "<span style='background:#e67e22;color:#fff;font-size:9px;padding:1px 4px;border-radius:3px;margin-left:4px;'>UPD</span>"

        rows += f"""
        <tr style="background:{row_bg};border-bottom:1px solid #2a2a3e;">
          <td style="padding:6px 8px;font-weight:bold;color:#eee;font-size:12px;white-space:nowrap;">
            {name}{new_badge}
          </td>
          <td style="padding:6px 8px;color:{dir_color};font-size:12px;white-space:nowrap;">
            {direction} {tag_badge}
          </td>
          <td style="padding:6px 8px;color:#ccc;font-size:11px;font-family:monospace;white-space:nowrap;">
            {z['proximal']:.{dp}f} / {z['distal']:.{dp}f}
          </td>
          <td style="padding:6px 8px;color:{stat_col};font-size:11px;white-space:nowrap;">{status}</td>
          <td style="padding:6px 8px;color:{fvg_col};font-size:12px;text-align:center;">{fvg_cell}</td>
          <td style="padding:6px 8px;color:#888;font-size:10px;white-space:nowrap;">{since}</td>
        </tr>"""

    return f"""
    <div style="margin-bottom:24px;">
      <h3 style="color:#aaa;font-size:12px;letter-spacing:1px;margin:0 0 8px 0;text-transform:uppercase;">
        Active Zone Map
      </h3>
      <table style="width:100%;border-collapse:collapse;background:#1a1a2e;border-radius:6px;overflow:hidden;">
        <thead>
          <tr style="background:#0d0d1a;">
            <th style="padding:7px 8px;text-align:left;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">Pair</th>
            <th style="padding:7px 8px;text-align:left;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">Bias</th>
            <th style="padding:7px 8px;text-align:left;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">Prox / Dist</th>
            <th style="padding:7px 8px;text-align:left;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">Status</th>
            <th style="padding:7px 8px;text-align:center;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">FVG</th>
            <th style="padding:7px 8px;text-align:left;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">First Seen</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

def build_new_zone_card_html(ob, name, dp, narrative, cid, ist_timestamp):
    """Full detail card for a new zone — includes chart and Gemini narrative."""
    direction  = "Bullish (Demand)" if ob['direction'] == 'bullish' else "Bearish (Supply)"
    dir_color  = '#27ae60' if ob['direction'] == 'bullish' else '#e74c3c'
    stat_color = '#27ae60' if 'Pristine' in ob['status'] else '#e67e22'
    fvg_line   = (
        f"FVG: <span style='color:#2ecc71;'>✓ {ob['fvg']['fvg_bottom']:.{dp}f} – "
        f"{ob['fvg']['fvg_top']:.{dp}f}</span>"
        if ob['fvg']['exists']
        else "FVG: <span style='color:#888;'>None</span>"
    )
    pip_unit  = 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)
    zone_pips = round(abs(ob['proximal_line'] - ob['distal_line']) / pip_unit, 1)

    chart_html = (
        f'<img src="cid:{cid}" style="width:100%;max-width:600px;border-radius:6px;'
        f'border:1px solid #2a2a3e;display:block;" />'
        if cid else
        '<p style="color:#888;font-size:11px;">[Chart unavailable]</p>'
    )

    return f"""
    <div style="margin-bottom:28px;padding:16px;background:#1a1a2e;border-radius:8px;
                border-left:4px solid {dir_color};">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;
                  margin-bottom:10px;flex-wrap:wrap;gap:6px;">
        <div>
          <span style="font-size:14px;font-weight:bold;color:#eee;">{name}</span>
          <span style="font-size:12px;color:{dir_color};margin-left:8px;">{direction}</span>
          <span style="background:#27ae60;color:#fff;font-size:9px;padding:2px 6px;
                       border-radius:3px;margin-left:6px;font-weight:bold;">NEW</span>
        </div>
        <span style="font-size:10px;color:#666;">{ist_timestamp} IST</span>
      </div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;">
        <span style="font-size:11px;color:#aaa;">
          <b style="color:#bb8fce;">Proximal</b> {ob['proximal_line']:.{dp}f}
        </span>
        <span style="font-size:11px;color:#aaa;">
          <b style="color:#bb8fce;">Distal</b> {ob['distal_line']:.{dp}f}
        </span>
        <span style="font-size:11px;color:#aaa;">
          <b>Width</b> {zone_pips} pips
        </span>
        <span style="font-size:11px;color:{stat_color};">
          <b>Status</b> {ob['status']}
        </span>
        <span style="font-size:11px;color:#aaa;">{fvg_line}</span>
      </div>
      <p style="font-size:12px;color:#bbb;line-height:1.6;margin:0 0 12px 0;
                border-left:3px solid #2a2a3e;padding-left:10px;">
        {narrative}
      </p>
      {chart_html}
    </div>"""

def build_changed_zone_html(stored_zone, changes, name, dp):
    """Compact update block for a zone that changed since last email."""
    direction = "Bullish" if stored_zone['direction'] == 'bullish' else "Bearish"
    dir_color = '#27ae60' if stored_zone['direction'] == 'bullish' else '#e74c3c'
    change_items = "".join(
        f'<li style="margin:2px 0;">{c}</li>' for c in changes
    )
    return f"""
    <div style="margin-bottom:10px;padding:10px 12px;background:#2d2a1a;
                border-left:4px solid #e67e22;border-radius:4px;">
      <div style="margin-bottom:4px;">
        <span style="font-size:12px;font-weight:bold;color:#eee;">{name}</span>
        <span style="font-size:11px;color:{dir_color};margin-left:8px;">{direction}</span>
        <span style="background:#e67e22;color:#fff;font-size:9px;padding:1px 5px;
                     border-radius:3px;margin-left:6px;">UPDATE</span>
        <span style="font-size:11px;color:#888;margin-left:8px;font-family:monospace;">
          {stored_zone['proximal']:.{dp}f} / {stored_zone['distal']:.{dp}f}
        </span>
      </div>
      <ul style="margin:0;padding-left:16px;color:#e0c080;font-size:11px;">
        {change_items}
      </ul>
    </div>"""

def build_repeat_zone_html(stored_zone, name, dp):
    """Single-line reference for unchanged existing zones — no chart."""
    direction = "Bullish" if stored_zone['direction'] == 'bullish' else "Bearish"
    dir_color = '#27ae60' if stored_zone['direction'] == 'bullish' else '#e74c3c'
    return f"""
    <div style="margin-bottom:6px;padding:7px 10px;background:#111827;
                border-left:3px solid #2a2a3e;border-radius:3px;">
      <span style="font-size:11px;font-weight:bold;color:#aaa;">{name}</span>
      <span style="font-size:11px;color:{dir_color};margin-left:6px;">{direction}</span>
      <span style="font-size:11px;color:#666;margin-left:6px;font-family:monospace;">
        {stored_zone['proximal']:.{dp}f} / {stored_zone['distal']:.{dp}f}
      </span>
      <span style="font-size:10px;color:#555;margin-left:8px;">
        since {stored_zone['first_seen_ist']} · unchanged
      </span>
    </div>"""

def send_master_digest(summary_table_html, new_zone_cards, changed_zone_blocks,
                       repeat_zone_lines, attachments, zone_count, ist_time):
    new_cards_html     = "".join(new_zone_cards)
    changed_block_html = "".join(changed_zone_blocks)
    repeat_html        = "".join(repeat_zone_lines)

    sections = ""
    if new_zone_cards:
        sections += f"""
        <div style="margin-bottom:20px;">
          <h3 style="color:#27ae60;font-size:12px;letter-spacing:1px;margin:0 0 10px 0;
                     text-transform:uppercase;border-bottom:1px solid #1e3a2f;padding-bottom:6px;">
            New Zones ({len(new_zone_cards)})
          </h3>
          {new_cards_html}
        </div>"""

    if changed_zone_blocks:
        sections += f"""
        <div style="margin-bottom:20px;">
          <h3 style="color:#e67e22;font-size:12px;letter-spacing:1px;margin:0 0 10px 0;
                     text-transform:uppercase;border-bottom:1px solid #2d2a1a;padding-bottom:6px;">
            Zone Updates ({len(changed_zone_blocks)})
          </h3>
          {changed_block_html}
        </div>"""

    if repeat_zone_lines:
        sections += f"""
        <div style="margin-bottom:8px;">
          <h3 style="color:#555;font-size:11px;letter-spacing:1px;margin:0 0 8px 0;
                     text-transform:uppercase;">
            Active · Unchanged ({len(repeat_zone_lines)})
          </h3>
          {repeat_html}
        </div>"""

    master_html = f"""<html>
<body style="font-family:Arial,sans-serif;background:#0d0d1a;padding:16px;margin:0;">
<div style="max-width:650px;margin:auto;background:#13131f;border-radius:10px;
            overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.4);">

  <!-- Header -->
  <div style="background:#0d0d1a;padding:18px 20px;border-bottom:1px solid #1a1a2e;">
    <h2 style="color:#eee;margin:0;font-size:18px;letter-spacing:1px;">
      PHASE 1 SCOUT DIGEST
    </h2>
    <p style="color:#555;margin:4px 0 0;font-size:11px;">
      {zone_count} active zones · updated {ist_time} IST
    </p>
  </div>

  <!-- Body -->
  <div style="padding:18px 20px;">
    {summary_table_html}
    {sections}
  </div>

  <!-- Footer -->
  <div style="background:#0d0d1a;padding:12px 20px;border-top:1px solid #1a1a2e;text-align:center;">
    <p style="color:#333;font-size:10px;margin:0;">
      SMC Alert Engine v2.0 · Institutional Order Flow · {ist_time} IST
    </p>
  </div>
</div>
</body></html>"""

    msg           = MIMEMultipart("related")
    msg['From']   = EMAIL_CONFIG['sender']
    msg['To']     = ", ".join(EMAIL_CONFIG['recipient'])
    msg['Subject']= f"Scout Digest | {zone_count} Zones | {ist_time}"
    msg.attach(MIMEText(master_html, 'html'))
    for img in attachments:
        msg.attach(img)

    with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
        server.starttls()
        server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['sender'], EMAIL_CONFIG['recipient'], msg.as_string())

    logging.info(f"Digest sent: {zone_count} zones, {len(new_zone_cards)} new, "
                 f"{len(changed_zone_blocks)} updated, {len(repeat_zone_lines)} unchanged.")
    print(f"Digest sent: {zone_count} zones total.")

# ---------------------------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------------------------

def run_radar():
    ist_now      = get_ist_now()
    ist_time_str = ist_now.strftime('%H:%M')
    ist_ts_full  = ist_now.strftime('%d %b %Y, %H:%M')

    # --- Blackout: do nothing before 09:00 IST ---
    if ist_now.hour < 9:
        print(f"Blackout active. Suppressed until 09:00 IST.")
        return

    # --- 2-hour email gate: only send on even hours ---
    send_email_this_run = (ist_now.hour % 2 == 0)
    print(f"Scout running at {ist_time_str} IST | Email: {'YES' if send_email_this_run else 'NO (odd hour scan)'}")

    day_start    = get_day_start_ist()    # daily slate boundary
    zone_state   = load_zone_state()      # {pair_name: [zone_dicts]}
    export_payload = {}

    # dp lookup map for HTML builders
    dp_map = {p['name']: p.get('decimal_places', 5) for p in config_master['pairs']}

    # Collectors for email assembly
    all_zones_for_table  = []   # every active zone this run — for summary table
    new_zone_cards       = []
    changed_zone_blocks  = []
    repeat_zone_lines    = []
    attachments          = []
    chart_counter        = 0

    for pair in config_master["pairs"]:
        ticker   = pair["symbol"]
        name     = pair["name"]
        dp       = pair.get("decimal_places", 5)
        ptype    = pair.get("pair_type", "forex")
        lookback = 5 if name in ["NZDUSD", "GOLD"] else 6 if name == "NAS100" else 4

        df = fetch_data(ticker, pair["map_tf"], "15d")
        if df is None:
            logging.warning(f"No data for {name}. Skipped.")
            continue

        result       = detect_smc_radar(df, lookback)
        current_price = result["current_price"]
        scanned_obs  = result["active_unmitigated_obs"]
        export_payload[name] = scanned_obs

        # Stored zones for this pair today
        stored_today = [
            z for z in zone_state.get(name, [])
            if z.get("first_seen_ist", "") >= day_start
        ]

        matched_stored_ids = set()   # indices into stored_today that were matched

        for ob in scanned_obs:
            # --- Find matching stored zone ---
            matched_idx = None
            for si, sz in enumerate(stored_today):
                if zones_match(ob, sz, ptype):
                    matched_idx = si
                    break

            if matched_idx is None:
                # ---- BRAND NEW ZONE ----
                first_seen_label = ist_now.strftime('%H:%M IST')
                new_stored = {
                    "proximal":       ob['proximal_line'],
                    "distal":         ob['distal_line'],
                    "direction":      ob['direction'],
                    "bos_tag":        ob['bos_tag'],
                    "first_seen_ist": first_seen_label,
                    "last_seen_ist":  first_seen_label,
                    "touches":        ob['touches'],
                    "fvg_valid":      ob['fvg']['exists'],
                    "status":         ob['status'],
                    "changes":        []
                }
                stored_today.append(new_stored)

                # Table row data
                all_zones_for_table.append({
                    "name": name, "direction": ob['direction'],
                    "proximal": ob['proximal_line'], "distal": ob['distal_line'],
                    "bos_tag": ob['bos_tag'], "status": ob['status'],
                    "fvg_valid": ob['fvg']['exists'],
                    "first_seen_ist": first_seen_label,
                    "is_new": True, "is_changed": False
                })

                if send_email_this_run:
                    narrative = generate_zone_narrative(ob, name, dp, current_price)
                    cid       = f"chart_{name}_{chart_counter}"
                    chart_b64 = generate_h1_chart(df, ob, dp, name, ist_ts_full)

                    new_zone_cards.append(
                        build_new_zone_card_html(ob, name, dp, narrative, cid, ist_ts_full)
                    )

                    if chart_b64:
                        img_mime = MIMEImage(base64.b64decode(chart_b64))
                        img_mime.add_header("Content-ID", f"<{cid}>")
                        img_mime.add_header("Content-Disposition", "inline",
                                            filename=f"{cid}.png")
                        attachments.append(img_mime)
                        chart_counter += 1

                print(f"  NEW zone: {name} {ob['direction']} @ {ob['proximal_line']:.{dp}f}")

            else:
                # ---- EXISTING ZONE ----
                matched_stored_ids.add(matched_idx)
                sz      = stored_today[matched_idx]
                changes = detect_zone_changes(ob, sz, dp)

                # Always update last_seen and live state
                sz["last_seen_ist"] = ist_now.strftime('%H:%M IST')
                sz["touches"]       = ob['touches']
                sz["fvg_valid"]     = ob['fvg']['exists']
                sz["status"]        = ob['status']

                is_changed = len(changes) > 0

                all_zones_for_table.append({
                    "name": name, "direction": ob['direction'],
                    "proximal": ob['proximal_line'], "distal": ob['distal_line'],
                    "bos_tag": ob['bos_tag'], "status": ob['status'],
                    "fvg_valid": ob['fvg']['exists'],
                    "first_seen_ist": sz['first_seen_ist'],
                    "is_new": False, "is_changed": is_changed
                })

                if send_email_this_run:
                    if is_changed:
                        sz["changes"].extend(changes)
                        changed_zone_blocks.append(
                            build_changed_zone_html(sz, changes, name, dp)
                        )
                        print(f"  UPDATED zone: {name} — {changes}")
                    else:
                        repeat_zone_lines.append(
                            build_repeat_zone_html(sz, name, dp)
                        )

        # Update stored state for this pair
        zone_state[name] = stored_today

    # --- Send email if this is an even-hour run and there's something to send ---
    if send_email_this_run:
        if all_zones_for_table:
            summary_table = build_summary_table_html(all_zones_for_table, dp_map)
            send_master_digest(
                summary_table, new_zone_cards, changed_zone_blocks,
                repeat_zone_lines, attachments,
                len(all_zones_for_table), ist_time_str
            )
        else:
            print("  No active zones detected. Digest skipped.")
    else:
        print(f"  Scan complete. {len(all_zones_for_table)} zones cached. Next email on even hour.")

    save_zone_state(zone_state)
    with open("active_obs.json", "w") as f:
        json.dump(export_payload, f, indent=2)
    print(f"Phase 1 complete at {ist_time_str} IST.")

if __name__ == "__main__":
    run_radar()
