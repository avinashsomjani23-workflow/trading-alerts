import yfinance as yf
import pandas as pd
import numpy as np
import json
import smtplib
import logging
import os
import time
import google.generativeai as genai
import smc_detector
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
    "sender":      os.environ.get("GMAIL_ADDRESS", "avinash.somjani23@gmail.com"),
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
    "forex":     0.00030,
    "index":     30.0,
    "commodity": 3.0
}

# ---------------------------------------------------------------------------
# STALENESS THRESHOLDS (B3)
# ---------------------------------------------------------------------------
STALENESS_HOURS = {
    "1h": 2.0,
    "15m": 0.75,
    "5m": 0.30
}

# ---------------------------------------------------------------------------
# EMAIL GATE CONFIG (B6) — state-based timer
# ---------------------------------------------------------------------------
EMAIL_GATE_MINUTES = 100  # Email sent if ≥100 min since last email

# ---------------------------------------------------------------------------
# STATE COMPACTION (B7)
# ---------------------------------------------------------------------------
ZONE_MAX_AGE_DAYS = 14  # Drop zones whose last_seen is older than this

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
# ATOMIC SAVE (A3) + LOGGING HELPERS (B3)
# ---------------------------------------------------------------------------

def save_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_json_safe(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _log_stale_skip(symbol, interval, reason, age_hours):
    try:
        log = load_json_safe("yfinance_stale_log.json", [])
        log.append({
            "ts": get_ist_now().isoformat(),
            "symbol": symbol,
            "interval": interval,
            "reason": reason,
            "age_hours": round(age_hours, 2) if age_hours is not None else None
        })
        log = log[-200:]
        save_json_atomic("yfinance_stale_log.json", log)
    except Exception as e:
        logging.error(f"Stale log write failed: {e}")


def _check_staleness(df, interval):
    if df is None or df.empty:
        return True, None
    try:
        last_ts = df['Datetime'].iloc[-1] if 'Datetime' in df.columns else df.index[-1]
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

# ---------------------------------------------------------------------------
# ZONE STATE PERSISTENCE
# ---------------------------------------------------------------------------

def load_zone_state():
    return load_json_safe("emailed_zones.json", {})


def save_zone_state(state):
    save_json_atomic("emailed_zones.json", state)


def zones_match(ob, stored, pair_type):
    if ob["direction"] != stored["direction"]:
        return False
    threshold = ZONE_PROXIMITY_THRESHOLDS.get(pair_type, 0.0003)
    return abs(ob["proximal_line"] - stored["proximal"]) <= threshold


def detect_zone_changes(ob, stored, dp):
    changes = []
    if ob["touches"] > stored["touches"]:
        diff = ob["touches"] - stored["touches"]
        new_status = ob["status"]
        changes.append(f"New touch detected (+{diff}). Status now: {new_status}.")
    if ob["fvg"]["exists"] != stored["fvg_valid"]:
        if not ob["fvg"]["exists"] and stored["fvg_valid"]:
            changes.append("FVG mitigated since last scan.")
        elif ob["fvg"]["exists"] and not stored["fvg_valid"]:
            changes.append("FVG now confirmed (was absent).")
    return changes


def compact_zone_state(zone_state):
    """
    B7: Drop zones whose last_seen is older than ZONE_MAX_AGE_DAYS.
    Keeps counter keys and __day__ marker intact.
    """
    cutoff = get_ist_now() - timedelta(days=ZONE_MAX_AGE_DAYS)
    removed_count = 0

    for key in list(zone_state.keys()):
        if key.startswith("__"):
            continue
        if not isinstance(zone_state[key], list):
            continue

        kept = []
        for z in zone_state[key]:
            last_seen_str = z.get("last_seen_ist") or z.get("first_seen_ist")
            if not last_seen_str:
                kept.append(z)
                continue
            # last_seen_ist stored as "HH:MM IST" label, not full ISO, so
            # fall back to first_seen_ist if stored as ISO, else keep.
            try:
                if "T" in last_seen_str:
                    last_dt = datetime.fromisoformat(last_seen_str)
                    if last_dt < cutoff:
                        removed_count += 1
                        continue
                # Label format "HH:MM IST" — cannot age-test; keep
                kept.append(z)
            except Exception:
                kept.append(z)

        if kept:
            zone_state[key] = kept
        else:
            del zone_state[key]

    if removed_count > 0:
        logging.info(f"Compaction: removed {removed_count} stale zones.")
        print(f"  [COMPACT] Removed {removed_count} zones older than {ZONE_MAX_AGE_DAYS} days.")

# ---------------------------------------------------------------------------
# ZONE REFERENCE ID
# ---------------------------------------------------------------------------

ZONE_ID_PREFIX = {
    "EURUSD": "EUR", "USDJPY": "JPY", "NZDUSD": "NZD",
    "USDCHF": "CHF", "NAS100": "NAS", "GOLD":   "XAU"
}

def get_next_zone_id(zone_state, name):
    counter_key = f"__counter_{name}__"
    current = zone_state.get(counter_key, 0) + 1
    zone_state[counter_key] = current
    prefix = ZONE_ID_PREFIX.get(name, name[:3].upper())
    return f"{prefix}{current:02d}"

def reset_daily_counters(zone_state, names):
    for name in names:
        counter_key = f"__counter_{name}__"
        zone_state[counter_key] = 0

# ---------------------------------------------------------------------------
# DATA FETCH WITH RETRY + STALENESS (B3, B5)
# ---------------------------------------------------------------------------

def fetch_data(ticker, interval, period, retries=2):
    last_error = None
    last_age = None

    for attempt in range(retries + 1):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=20)
            if df is None or df.empty:
                last_error = "empty dataframe"
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                tailed = df.tail(150).copy().reset_index()
                is_stale, age_hours = _check_staleness(tailed, interval)
                if not is_stale:
                    return tailed
                last_age = age_hours
                last_error = f"stale data (age {age_hours:.2f}h, limit {STALENESS_HOURS.get(interval, 2.0)}h)"
        except Exception as e:
            last_error = str(e)

        if attempt < retries:
            wait = 2 ** attempt
            logging.warning(f"Retry {attempt + 1}/{retries} for {ticker} {interval}: {last_error}. Waiting {wait}s.")
            print(f"  [RETRY {attempt + 1}/{retries}] {ticker} {interval}: {last_error}")
            time.sleep(wait)

    logging.warning(f"Fetch failed after retries: {ticker} {interval}: {last_error}")
    print(f"  [SKIP] {ticker} {interval}: {last_error}")
    _log_stale_skip(ticker, interval, last_error, last_age)
    return None

# ---------------------------------------------------------------------------
# SMC DETECTION
# ---------------------------------------------------------------------------

def is_valid_ob_candle(open_p, close_p, high_p, low_p):
    body = abs(open_p - close_p)
    rng  = high_p - low_p
    if rng == 0:
        return False
    return body > (rng * 0.15)

# NEW
def detect_smc_radar(df, lookback):
    """
    A4 applied: OB walk-back includes the swing candle itself (impulse_start_idx).
    Verification: range(5-1, 0-1, -1) = range(4, -1, -1) = [4, 3, 2, 1, 0]. Includes 0.

    Leg-size filter (Round 2): a break is only emitted as a structure event
    (CHoCH or BOS) if the net price displacement from the prior opposite swing
    to the break candle's close is at least LEG_SIZE_MIN_ATR * H1 ATR. Breaks
    below threshold are treated as non-events: no zone emitted, trend_state and
    bos_seq_counter are not updated. This filters micro-swing noise in ranging
    markets without rejecting large multi-candle moves.
    """
    n = len(df)
    O = df['Open'].values
    C = df['Close'].values
    H = df['High'].values
    L = df['Low'].values

    # H1 ATR for leg-size threshold. Computed once per scan.
    h1_atr_for_leg = smc_detector.compute_atr(df)
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
    bos_seq_counter = 0
    last_choch_idx = None

    for i in range(lookback + 1, n):
        past_swings = [s for s in swings if s['idx'] < i]
        if len(past_swings) < 2:
            continue
        latest_high = [s for s in past_swings if s['type'] == 'high']
        latest_low  = [s for s in past_swings if s['type'] == 'low']
        if not latest_high or not latest_low:
            continue

        # NEW
        sh, sl = latest_high[-1], latest_low[-1]
        bos_detected, bos_type = False, None

        if C[i] > sh['price'] and C[i - 1] <= sh['price']:
            bos_detected, bos_type = True, 'bullish'
        elif C[i] < sl['price'] and C[i - 1] >= sl['price']:
            bos_detected, bos_type = True, 'bearish'

        if not bos_detected:
            continue

        # Provisional classification — tells us which threshold to apply.
        provisional_tag = 'CHoCH' if (trend_state is None or trend_state != bos_type) else 'BOS'
        threshold_mult = smc_detector.LEG_SIZE_MIN_ATR.get(provisional_tag, 0.6)

        # Prior opposite swing that this break flipped:
        # bullish break -> swung off the prior low (sl['price'])
        # bearish break -> swung off the prior high (sh['price'])
        prior_opposite_swing_price = sl['price'] if bos_type == 'bullish' else sh['price']

        if not smc_detector.validate_leg_distance(
            prior_opposite_swing_price, C[i], h1_atr_for_leg, threshold_mult
        ):
            # Leg too small — treat as non-event. Do NOT update state.
            continue

        # Passed leg-size filter. Now commit state.
        bos_tag = provisional_tag
        if bos_tag == 'CHoCH':
            bos_seq_counter = 0
            last_choch_idx = i
        else:
            bos_seq_counter += 1
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

        # A4: Walk back from BOS candle through impulse leg, INCLUDING swing candle.
        # range(i-1, impulse_start_idx - 1, -1) stops BEFORE impulse_start_idx - 1,
        # so the swing candle at impulse_start_idx IS included in scan.
        for j in range(i - 1, impulse_start_idx - 1, -1):
            if (bos_type == 'bullish' and C[j] < O[j]) or \
               (bos_type == 'bearish' and C[j] > O[j]):
                if is_valid_ob_candle(O[j], C[j], H[j], L[j]):
                    if abs(C[j] - O[j]) <= (1.5 * median_leg_body):
                        ob_idx = j
                        break

        if ob_idx == -1:
            continue

        ob_high = float(H[ob_idx])
        ob_low  = float(L[ob_idx])

        # FVG detection — leg-bounded via smc_detector (uniform across phases).
        # Window: OB candle to BOS candle + 1 (catches displacement gaps on/just-after BOS).
        # Walks BACKWARD → first unmitigated FVG closest to BOS wins.
        # If candidate is filled, returns ghost data for grey-box rendering.
        ob_high_for_fvg = ob_high
        ob_low_for_fvg  = ob_low
        bias_label      = "LONG" if bos_type == 'bullish' else "SHORT"
        h1_atr_for_fvg  = h1_atr_for_leg if h1_atr_for_leg else 0.0
        fvg_floor_mult  = smc_detector.FVG_NOISE_FLOOR_MULT.get("forex", 0.20)
        # NOTE: pair_type is known to the caller in run_radar; we use the OB's
        # own high/low as the zone overlap window, and let Phase 1's caller
        # apply pair-type-aware floor below. For this helper we use H1 ATR
        # with a neutral 0.20 floor — safe across all pairs on H1.
        atr_floor_h1    = fvg_floor_mult * h1_atr_for_fvg

        fvg_result = smc_detector.detect_fvg_in_zone(
            df, bias_label, ob_high_for_fvg, ob_low_for_fvg, atr_floor_h1,
            leg_start_idx=ob_idx, leg_end_idx=i
        )

        # Absolute timestamp of the OB candle — required by Phase 2/3 to locate
        # the OB candle in their own dataframes (ob_idx is not portable across phases).
        try:
            if 'Datetime' in df.columns:
                ob_ts_raw = df['Datetime'].iloc[ob_idx]
            elif 'Date' in df.columns:
                ob_ts_raw = df['Date'].iloc[ob_idx]
            else:
                ob_ts_raw = df.index[ob_idx]
            if hasattr(ob_ts_raw, 'isoformat'):
                ob_timestamp_str = ob_ts_raw.isoformat()
            else:
                ob_timestamp_str = str(ob_ts_raw)
        except Exception:
            ob_timestamp_str = None

        # Build fvg dict — pristine (dark green), partial (light green),
        # full (grey/ghost), or absent.
        fvg_dict = {
            'exists':       fvg_result.get('exists', False),
            'fvg_top':      fvg_result.get('fvg_top'),
            'fvg_bottom':   fvg_result.get('fvg_bottom'),
            'c1_idx':       fvg_result.get('c1_idx'),
            'c3_idx':       fvg_result.get('c3_idx'),
            'was_detected': fvg_result.get('was_detected', False),
            'mitigation':   fvg_result.get('mitigation', 'none'),
            'ghost_top':    fvg_result.get('ghost_top'),
            'ghost_bottom': fvg_result.get('ghost_bottom'),
            'ghost_c1_idx': fvg_result.get('ghost_c1_idx'),
            'ghost_c3_idx': fvg_result.get('ghost_c3_idx'),
            'mitigated_at_idx': fvg_result.get('mitigated_at_idx')
        }

        active_obs.append({
            'bos_idx':           i,
            'bos_swing_price':   bos_swing_price,
            'impulse_start_idx': impulse_start_idx,
            'impulse_start_price': float(L[impulse_start_idx]) if bos_type == 'bullish' else float(H[impulse_start_idx]),
            'bos_sequence_count': bos_seq_counter,
            'last_choch_idx':    last_choch_idx,
            'ob_idx':            ob_idx,
            'ob_timestamp':      ob_timestamp_str,
            'direction':       bos_type,
            'bos_tag':         bos_tag,
            'high':            ob_high,
            'low':             ob_low,
            'proximal_line':   ob_high if bos_type == 'bullish' else ob_low,
            'distal_line':     ob_low  if bos_type == 'bullish' else ob_high,
            'median_leg_body': median_leg_body,
            'ob_body':         abs(C[ob_idx] - O[ob_idx]),
            'fvg': fvg_dict
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
            if touches >= 3:
                mitigated = True
                break

        if not mitigated:
            ob['touches'] = touches
            ob['status']  = 'Pristine' if touches == 0 else f'Tested ({touches}x)'
            tracked_obs.append(ob)

    # Latest OB per bias; same-leg fallback only.
    latest, filtered = {}, []
    for ob in sorted(tracked_obs, key=lambda x: x['bos_idx'], reverse=True):
        d = ob['direction']
        if d not in latest:
            latest[d] = ob
            filtered.append(ob)
        elif ob['impulse_start_idx'] == latest[d]['impulse_start_idx']:
            filtered.append(ob)
            break

    # Same-leg dedupe: if two OBs in `filtered` share a direction AND their
    # proximal lines sit within the pair-type proximity threshold, they are
    # the same visual zone. Keep the one with an unmitigated FVG; if tied,
    # keep the one with the higher bos_idx (freshest structure).
    # NOTE: proximity_threshold is resolved in the caller (run_radar) via
    # pair_type — here we pass the threshold via ob['_dedupe_thresh'] which
    # the caller injects. If absent, we fall back to forex default.
    def _dedupe_same_leg(obs):
        if len(obs) < 2:
            return obs
        by_dir = {}
        for o in obs:
            by_dir.setdefault(o['direction'], []).append(o)
        kept = []
        for direction, group in by_dir.items():
            if len(group) == 1:
                kept.extend(group)
                continue
            group_sorted = sorted(group, key=lambda x: x['bos_idx'], reverse=True)
            survivors = []
            for cand in group_sorted:
                is_dup = False
                thresh = cand.get('_dedupe_thresh', 0.00030)
                for surv in survivors:
                    if abs(cand['proximal_line'] - surv['proximal_line']) <= thresh:
                        # Duplicate. Prefer the one with unmitigated FVG.
                        cand_has_fvg = cand['fvg'].get('exists', False)
                        surv_has_fvg = surv['fvg'].get('exists', False)
                        if cand_has_fvg and not surv_has_fvg:
                            # Replace survivor with cand.
                            survivors.remove(surv)
                            survivors.append(cand)
                        # else: keep current survivor (freshest by bos_idx).
                        is_dup = True
                        break
                if not is_dup:
                    survivors.append(cand)
            kept.extend(survivors)
        return kept

    filtered = _dedupe_same_leg(filtered)

    # Strip the private dedupe hint before returning.
    for o in filtered:
        o.pop('_dedupe_thresh', None)

    return {"current_price": float(C[-1]), "active_unmitigated_obs": filtered}

# ---------------------------------------------------------------------------
# CHART GENERATION
# ---------------------------------------------------------------------------

def generate_h1_chart(df, ob, dp, pair_name, ist_timestamp):
    try:
        full_df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy().reset_index(drop=True)
        n_full  = len(full_df)
        ob_abs  = ob['ob_idx']

        window_start = max(0, ob_abs - 15)
        window_start = min(window_start, n_full - 1)

        df_plot = full_df.iloc[window_start:].copy().reset_index(drop=True)
        n_plot  = len(df_plot)

        ob_plot_idx = ob_abs - window_start

        O = df_plot['Open'].values
        C = df_plot['Close'].values
        H = df_plot['High'].values
        L = df_plot['Low'].values

        fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), facecolor='#131722')
        ax.set_facecolor('#131722')
        for spine in ax.spines.values():
            spine.set_color('#2a2a3e')

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

        if 0 <= ob_plot_idx < n_plot:
            ob_h = float(H[ob_plot_idx])
            ob_l = float(L[ob_plot_idx])
            ax.add_patch(patches.Rectangle(
                (ob_plot_idx - 0.5, ob_l), 1.0, ob_h - ob_l,
                fill=False, edgecolor='#d7bde2', linewidth=2.0, zorder=4,
                linestyle='-'
            ))
            ax.text(
                ob_plot_idx, ob_h, 'OB',
                color='#d7bde2', fontsize=7, ha='center', va='bottom',
                fontweight='bold', zorder=5
            )

        fvg_active    = ob['fvg']['exists'] and ob['fvg']['c1_idx'] is not None
        fvg_ghost     = (not ob['fvg']['exists']) and ob['fvg'].get('was_detected') and ob['fvg'].get('ghost_c1_idx') is not None

        if fvg_active:
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
        elif fvg_ghost:
            ft = float(ob['fvg']['ghost_top'])
            fb = float(ob['fvg']['ghost_bottom'])
            fvg_c1_plot = ob['fvg']['ghost_c1_idx'] - window_start
            fvg_c3_plot = ob['fvg']['ghost_c3_idx'] - window_start
            if 0 <= fvg_c1_plot < n_plot:
                fvg_x_start = fvg_c1_plot - 0.4
                fvg_width   = (fvg_c3_plot + 0.4) - fvg_x_start
                ax.add_patch(patches.Rectangle(
                    (fvg_x_start, fb), fvg_width, ft - fb,
                    facecolor='#888888', alpha=0.10, zorder=1
                ))
                ax.add_patch(patches.Rectangle(
                    (fvg_x_start, fb), fvg_width, ft - fb,
                    fill=False, edgecolor='#888888', linewidth=1.0,
                    linestyle=':', zorder=2
                ))
                ax.text(
                    fvg_c1_plot, ft, 'FVG ✓',
                    color='#888888', fontsize=7, ha='left', va='bottom', zorder=5
                )
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

        y_min = float(np.min(L))
        y_max = float(np.max(H))
        pad   = (y_max - y_min) * 0.08
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlim(-1, n_plot + 10)

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
# GEMINI NARRATIVE
# ---------------------------------------------------------------------------

def generate_zone_narrative(ob, name, dp, current_price):
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
    rows = ""
    for z in all_zones_for_table:
        name      = z['name']
        dp        = dp_map[name]
        direction = "&#9650; Bullish" if z['direction'] == 'bullish' else "&#9660; Bearish"
        dir_color = '#27ae60'   if z['direction'] == 'bullish' else '#e74c3c'
        status    = z['status']
        stat_col  = '#27ae60'   if 'Pristine' in status else '#e67e22'
        if z['fvg_valid']:
            fvg_cell = "&#10003;"
            fvg_col  = '#27ae60'
        elif z.get('fvg_ghost'):
            fvg_cell = "&#9675;"   # empty circle = ghost
            fvg_col  = '#888888'
        else:
            fvg_cell = "&ndash;"
            fvg_col  = '#888'
        tag_badge = (
            f"<span style='background:#00bcd4;color:#000;font-size:9px;"
            f"padding:1px 4px;border-radius:3px;font-weight:bold;'>{z['bos_tag']}</span>"
            if z['bos_tag'] == 'BOS' else
            f"<span style='background:#ff9800;color:#000;font-size:9px;"
            f"padding:1px 4px;border-radius:3px;font-weight:bold;'>{z['bos_tag']}</span>"
        )
        is_new     = z.get('is_new', False)
        is_changed = z.get('is_changed', False)
        row_bg     = '#1e3a2f' if is_new else ('#2d2a1a' if is_changed else 'transparent')
        new_badge  = ""
        if is_new:
            new_badge = "<span style='background:#27ae60;color:#fff;font-size:9px;padding:1px 4px;border-radius:3px;margin-left:4px;'>NEW</span>"
        elif is_changed:
            new_badge = "<span style='background:#e67e22;color:#fff;font-size:9px;padding:1px 4px;border-radius:3px;margin-left:4px;'>UPD</span>"

        zone_label = f"<span style='color:#555;font-size:10px;font-family:monospace;'>{z.get('zone_id','&mdash;')}&nbsp;</span>"

        rows += f"""
        <tr style="background:{row_bg};border-bottom:1px solid #2a2a3e;">
          <td style="padding:6px 8px;font-weight:bold;color:#eee;font-size:12px;white-space:nowrap;">
            {zone_label}{name}{new_badge}
          </td>
          <td style="padding:6px 8px;color:{dir_color};font-size:12px;white-space:nowrap;">
            {direction}&nbsp;{tag_badge}
          </td>
          <td style="padding:6px 8px;color:{stat_col};font-size:11px;white-space:nowrap;">{status}</td>
          <td style="padding:6px 8px;color:{fvg_col};font-size:12px;text-align:center;">{fvg_cell}</td>
          <td style="padding:6px 8px;color:#888;font-size:10px;white-space:nowrap;">{z['first_seen_ist']}</td>
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
            <th style="padding:7px 8px;text-align:left;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">Status</th>
            <th style="padding:7px 8px;text-align:center;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">FVG</th>
            <th style="padding:7px 8px;text-align:left;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">First Seen</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def build_new_zone_card_html(ob, name, dp, narrative, cid, ist_timestamp, zone_id="—"):
    direction  = "Bullish (Demand)" if ob['direction'] == 'bullish' else "Bearish (Supply)"
    dir_color  = '#27ae60' if ob['direction'] == 'bullish' else '#e74c3c'
    stat_color = '#27ae60' if 'Pristine' in ob['status'] else '#e67e22'
    mit = ob['fvg'].get('mitigation', 'none')
    if ob['fvg'].get('exists') and mit == 'partial':
        # Partial — price crossed proximal but not distal. Light green.
        fvg_line = (
            f"FVG: <span style='color:#7ed67e;'>◐ Partial "
            f"{ob['fvg']['fvg_bottom']:.{dp}f} – {ob['fvg']['fvg_top']:.{dp}f}</span>"
        )
    elif ob['fvg'].get('exists'):
        # Pristine — untouched. Dark green.
        fvg_line = (
            f"FVG: <span style='color:#27ae60;'>✓ {ob['fvg']['fvg_bottom']:.{dp}f} – "
            f"{ob['fvg']['fvg_top']:.{dp}f}</span>"
        )
    elif ob['fvg'].get('was_detected'):
        gb = ob['fvg'].get('ghost_bottom')
        gt = ob['fvg'].get('ghost_top')
        fvg_line = (
            f"FVG: <span style='color:#888;'>✗ Mitigated "
            f"({gb:.{dp}f} – {gt:.{dp}f})</span>"
        )
    else:
        fvg_line = "FVG: <span style='color:#888;'>None</span>"
    pip_unit  = 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)
    zone_pips = round(abs(ob['proximal_line'] - ob['distal_line']) / pip_unit, 1)

    chart_html = (
        f'<img src="cid:{cid}" style="width:100%;max-width:600px;border-radius:6px;'
        f'border:1px solid #2a2a3e;display:block;" />'
        if cid else
        '<div style="padding:8px 12px;background:#2d1a1a;border-left:3px solid #e74c3c;border-radius:4px;color:#e74c3c;font-size:11px;">&#9888; Chart failed to render.</div>'
    )

    return f"""
    <div style="margin-bottom:28px;padding:16px;background:#1a1a2e;border-radius:8px;
                border-left:4px solid {dir_color};">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;
                  margin-bottom:10px;flex-wrap:wrap;gap:6px;">
        <div>
          <span style="font-size:10px;color:#888;font-family:monospace;margin-right:6px;">{zone_id}</span>
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


def build_changed_zone_html(stored_zone, changes, name, dp, cid=None):
    direction = "Bullish" if stored_zone['direction'] == 'bullish' else "Bearish"
    dir_color = '#27ae60' if stored_zone['direction'] == 'bullish' else '#e74c3c'
    change_items = "".join(
        f'<li style="margin:3px 0;color:#e0c080;">{c}</li>' for c in changes
    )
    zone_id = stored_zone.get('zone_id', '—')
    chart_html = (
        f'<img src="cid:{cid}" style="width:100%;max-width:600px;border-radius:6px;'
        f'border:1px solid #2a2a3e;display:block;margin-top:10px;" />'
        if cid else ''
    )
    return f"""
    <div style="margin-bottom:16px;padding:12px 14px;background:#2d2a1a;
                border-left:4px solid #e67e22;border-radius:6px;">
      <div style="margin-bottom:6px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        <span style="font-size:10px;color:#888;font-family:monospace;">{zone_id}</span>
        <span style="font-size:12px;font-weight:bold;color:#eee;">{name}</span>
        <span style="font-size:11px;color:{dir_color};">{direction}</span>
        <span style="background:#e67e22;color:#fff;font-size:9px;padding:1px 5px;
                     border-radius:3px;">UPDATE</span>
        <span style="font-size:11px;color:#888;font-family:monospace;">
          {stored_zone['proximal']:.{dp}f} / {stored_zone['distal']:.{dp}f}
        </span>
      </div>
      <ul style="margin:0;padding-left:16px;font-size:11px;line-height:1.7;">
        {change_items}
      </ul>
      {chart_html}
    </div>"""


def build_repeat_zone_html(stored_zone, name, dp):
    direction = "Bullish" if stored_zone['direction'] == 'bullish' else "Bearish"
    dir_color = '#27ae60' if stored_zone['direction'] == 'bullish' else '#e74c3c'
    return f"""
    <div style="margin-bottom:6px;padding:7px 10px;background:#111827;
                border-left:3px solid #2a2a3e;border-radius:3px;">
      <span style="font-size:10px;color:#666;font-family:monospace;margin-right:6px;">{stored_zone.get('zone_id','—')}</span>
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
  <div style="background:#0d0d1a;padding:18px 20px;border-bottom:1px solid #1a1a2e;">
    <h2 style="color:#eee;margin:0;font-size:18px;letter-spacing:1px;">
      PHASE 1 SCOUT DIGEST
    </h2>
    <p style="color:#555;margin:4px 0 0;font-size:11px;">
      {zone_count} active zones · updated {ist_time} IST
    </p>
  </div>
  <div style="padding:18px 20px;">
    {summary_table_html}
    {sections}
  </div>
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

# ---------------------------------------------------------------------------
# STATELESS EMAIL GATE (replaces emailed_zones.json)
# ---------------------------------------------------------------------------
EMAIL_GATE_FILE = "email_gate.json"

# Pair-type-aware same-leg OB dedupe thresholds.
# Matches ZONE_PROXIMITY_THRESHOLDS semantic but applied intra-scan
# between two OB candidates from the same leg.
OB_DEDUPE_THRESHOLDS = {
    "forex":     0.00030,   # 3 pips
    "index":     10.0,      # 10 points on NAS100
    "commodity": 1.0        # $1 on GOLD
}


def load_email_gate():
    return load_json_safe(EMAIL_GATE_FILE, {})


def save_email_gate(gate):
    save_json_atomic(EMAIL_GATE_FILE, gate)


def append_audit_log(zones_this_scan, ist_now):
    """
    Append-only audit log for weekly review.
    Writes one JSON object per scan to zone_audit_log.jsonl.
    Runtime logic NEVER reads this file. Weekly review reads it only.
    """
    try:
        entry = {
            "ts_iso": ist_now.isoformat(),
            "ts_label": ist_now.strftime('%d %b %Y, %H:%M IST'),
            "zones": zones_this_scan
        }
        with open("zone_audit_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logging.error(f"Audit log write failed: {e}")


# ---------------------------------------------------------------------------
# MAIN RUNNER — STATELESS
# ---------------------------------------------------------------------------

def run_radar():
    ist_now      = get_ist_now()
    ist_time_str = ist_now.strftime('%H:%M')
    ist_ts_full  = ist_now.strftime('%d %b %Y, %H:%M')

    # Blackout: do nothing before 09:00 IST
    if ist_now.hour < 9:
        print(f"Blackout active. Suppressed until 09:00 IST.")
        return

    # Email gate — only persisted state across runs.
    gate = load_email_gate()
    last_email_ts_str = gate.get("last_email_ts")
    send_email_this_run = True
    if last_email_ts_str:
        try:
            last_email_dt = datetime.fromisoformat(last_email_ts_str)
            minutes_since = (ist_now - last_email_dt).total_seconds() / 60
            send_email_this_run = minutes_since >= EMAIL_GATE_MINUTES
        except Exception:
            send_email_this_run = True

    print(f"Scout running at {ist_time_str} IST | Email: {'YES' if send_email_this_run else 'NO (within gate)'}")

    export_payload = {}
    dp_map = {p['name']: p.get('decimal_places', 5) for p in config_master['pairs']}

    # Stateless: every scan builds its own fresh zone list from scratch.
    all_zones_for_table = []
    zone_cards          = []
    attachments         = []
    chart_counter       = 0
    audit_rows          = []

    # Per-scan zone ID counters (reset every scan — stateless).
    per_scan_counters = {}

    def _next_id(pair_name):
        per_scan_counters[pair_name] = per_scan_counters.get(pair_name, 0) + 1
        prefix = ZONE_ID_PREFIX.get(pair_name, pair_name[:3].upper())
        return f"{prefix}{per_scan_counters[pair_name]:02d}"

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

        result        = detect_smc_radar(df, lookback)
        current_price = result["current_price"]
        scanned_obs   = result["active_unmitigated_obs"]

        # Inject pair-type-aware dedupe threshold for same-leg dedupe.
        # detect_smc_radar already ran — but we re-dedupe here with pair-aware
        # thresholds since detect_smc_radar used the default. Idempotent.
        dedupe_thresh = OB_DEDUPE_THRESHOLDS.get(ptype, 0.00030)
        if len(scanned_obs) > 1:
            by_dir = {}
            for o in scanned_obs:
                by_dir.setdefault(o['direction'], []).append(o)
            re_filtered = []
            for direction, group in by_dir.items():
                if len(group) == 1:
                    re_filtered.extend(group)
                    continue
                group_sorted = sorted(group, key=lambda x: x['bos_idx'], reverse=True)
                survivors = []
                for cand in group_sorted:
                    is_dup = False
                    for surv in survivors:
                        if abs(cand['proximal_line'] - surv['proximal_line']) <= dedupe_thresh:
                            cand_has_fvg = cand['fvg'].get('exists', False)
                            surv_has_fvg = surv['fvg'].get('exists', False)
                            if cand_has_fvg and not surv_has_fvg:
                                survivors.remove(surv)
                                survivors.append(cand)
                            is_dup = True
                            break
                    if not is_dup:
                        survivors.append(cand)
                re_filtered.extend(survivors)
            scanned_obs = re_filtered
            
        # Proximity filter — drop zones further than 4× H1 ATR from current price.
        # Gives ~4 hours of price-movement heads-up before zone is reachable.
        # Pair-agnostic multiplier because ATR is already pair-scaled.
        h1_atr_proximity = smc_detector.compute_atr(df)
        if h1_atr_proximity and h1_atr_proximity > 0:
            proximity_cap = 4.0 * h1_atr_proximity
            before_count = len(scanned_obs)
            scanned_obs = [
                o for o in scanned_obs
                if abs(current_price - float(o['proximal_line'])) <= proximity_cap
            ]
            dropped = before_count - len(scanned_obs)
            if dropped > 0:
                print(f"  [PROX] {name}: dropped {dropped} distant zones "
                      f"(beyond {proximity_cap:.{dp}f}).")
                
        export_payload[name] = scanned_obs

        for ob in scanned_obs:
            zone_id = _next_id(name)
            first_seen_label = ist_now.strftime('%H:%M IST')

            all_zones_for_table.append({
                "name": name, "zone_id": zone_id, "direction": ob['direction'],
                "proximal": ob['proximal_line'], "distal": ob['distal_line'],
                "bos_tag": ob['bos_tag'], "status": ob['status'],
                "fvg_valid": ob['fvg'].get('exists', False),
                "fvg_ghost": (not ob['fvg'].get('exists', False))
                              and ob['fvg'].get('was_detected', False),
                "first_seen_ist": first_seen_label,
                "is_new": False, "is_changed": False
            })

            audit_rows.append({
                "zone_id": zone_id, "pair": name,
                "direction": ob['direction'], "bos_tag": ob['bos_tag'],
                "proximal": ob['proximal_line'], "distal": ob['distal_line'],
                "status": ob['status'], "touches": ob.get('touches', 0),
                "fvg_exists": ob['fvg'].get('exists', False),
                "fvg_was_detected": ob['fvg'].get('was_detected', False),
                "current_price": current_price
            })

            if send_email_this_run:
                narrative = generate_zone_narrative(ob, name, dp, current_price)
                cid       = f"chart_{name}_{chart_counter}"
                chart_b64 = generate_h1_chart(df, ob, dp, name, ist_ts_full)

                zone_cards.append(
                    build_new_zone_card_html(
                        ob, name, dp, narrative,
                        cid if chart_b64 else None,
                        ist_ts_full, zone_id
                    )
                )

                if chart_b64:
                    img_mime = MIMEImage(base64.b64decode(chart_b64))
                    img_mime.add_header("Content-ID", f"<{cid}>")
                    img_mime.add_header("Content-Disposition", "inline",
                                        filename=f"{cid}.png")
                    attachments.append(img_mime)
                    chart_counter += 1

            print(f"  Zone: {name} {zone_id} {ob['direction']} @ {ob['proximal_line']:.{dp}f} "
                  f"| FVG: {'✓' if ob['fvg'].get('exists') else ('ghost' if ob['fvg'].get('was_detected') else '—')} "
                  f"| Status: {ob['status']}")

    # Always write audit log (used by weekly review only).
    if audit_rows:
        append_audit_log(audit_rows, ist_now)

    # Send email if gate allows
    if send_email_this_run:
        if all_zones_for_table:
            summary_table = build_summary_table_html(all_zones_for_table, dp_map)
            try:
                send_master_digest(
                    summary_table, zone_cards, [], [],
                    attachments, len(all_zones_for_table), ist_time_str
                )
                gate["last_email_ts"] = ist_now.isoformat()
                save_email_gate(gate)
            except Exception as e:
                logging.error(f"Digest send failed: {e}")
                print(f"  [EMAIL ERR] {e}")
        else:
            print("  No active zones detected. Digest skipped.")
    else:
        print(f"  Scan complete. {len(all_zones_for_table)} zones cached. Email gate active.")

    save_json_atomic("active_obs.json", export_payload)
    print(f"Phase 1 complete at {ist_time_str} IST.")

if __name__ == "__main__":
    run_radar()
