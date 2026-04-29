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
    return body > (rng * 0.10)

# NEW
def detect_smc_radar(df, lookback, pair_type="forex"):
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
        ob_proximal = ob_high if bos_type == 'bullish' else ob_low

        # PROXIMITY GATE — applied immediately after OB found, before any
        # further work on this zone. Drop zones whose proximal sits more than
        # 4× H1 ATR from current price. Saves compute (no FVG calc) and email
        # noise (no card for unreachable zones).
        current_price_now = float(C[-1])
        if h1_atr_for_leg and h1_atr_for_leg > 0:
            proximity_cap = 4.0 * h1_atr_for_leg
            if abs(current_price_now - ob_proximal) > proximity_cap:
                continue  # zone too far — drop silently

        # FVG detection — H1 OB internal gap. Kept as Phase 1 badge.
        # Window: OB candle to BOS candle + 1 (catches displacement gaps on/just-after BOS).
        # Walks BACKWARD → first unmitigated FVG closest to BOS wins.
        bias_label      = "LONG" if bos_type == 'bullish' else "SHORT"
        h1_atr_for_fvg  = h1_atr_for_leg if h1_atr_for_leg else 0.0
        fvg_floor_mult  = smc_detector.FVG_NOISE_FLOOR_MULT.get("forex", 0.20)
        atr_floor_h1    = fvg_floor_mult * h1_atr_for_fvg

        # FVG search window: OB candle through OB + FVG_WINDOW_H1_CANDLES.
        # Anchored to OB so window is consistent regardless of OB→BOS leg
        # length. Captures the post-BOS displacement gap which is where the
        # confirming FVG most often forms.
        h1_fvg_window_end = min(ob_idx + smc_detector.FVG_WINDOW_H1_CANDLES,
                                len(df) - 1)
        fvg_result = smc_detector.detect_fvg_in_zone(
            df, bias_label, ob_high, ob_low, atr_floor_h1,
            leg_start_idx=ob_idx, leg_end_idx=h1_fvg_window_end
        )

        # Absolute timestamp helpers — used by Phase 2 to locate candles in
        # their own dataframes, since integer indices are NOT portable across
        # phases (rolling yfinance window shifts the df start point between scans).
        def _ts_for_idx(idx_val):
            if idx_val is None:
                return None
            try:
                idx_val = int(idx_val)
                if idx_val < 0 or idx_val >= len(df):
                    return None
                if 'Datetime' in df.columns:
                    raw = df['Datetime'].iloc[idx_val]
                elif 'Date' in df.columns:
                    raw = df['Date'].iloc[idx_val]
                else:
                    raw = df.index[idx_val]
                if hasattr(raw, 'isoformat'):
                    return raw.isoformat()
                return str(raw)
            except Exception:
                return None

        ob_timestamp_str = _ts_for_idx(ob_idx)
        bos_timestamp_str = _ts_for_idx(i)

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

        # NOTE: liquidity sweep grading removed from Phase 1.
        # Phase 2 grades sweep on fresh M15 + H1 data when zone is approached.

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
            'h1_atr':          float(h1_atr_for_leg) if h1_atr_for_leg else 0.0,
            'fvg': fvg_dict
        })
# Mitigation + touch tracking. Sets 'touches' and 'status' on each OB.
    # Must run BEFORE dedupe so the touch-state test in dedupe is meaningful.
    tracked_obs = []
    for ob in active_obs:
        mitigated = False
        touches   = 0
        for m in range(ob['bos_idx'] + 1, n):
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

    # Same-leg dedupe: 4-test ladder applied in strict order.
    # Pristine > FVG-holder > Freshest > Defensive.
    # Touch state ('touches' field) is set above in mitigation block.
    def _dedupe_same_leg(obs):
        """
        Same-leg dedupe — 4-test ladder applied in strict order.
        Two OBs in same direction with proximal lines within pair-aware
        threshold are the same visual zone. Keep one using:

          Test 1: Pristine (0 touches) beats Tested.
          Test 2: Both same touch state — FVG-holder wins.
          Test 3: Tied on touch and FVG — freshest (higher bos_idx) wins.
          Test 4: Defensive — identical bos_idx, keep first encountered.
        """
        if len(obs) < 2:
            return obs

        def _pick_winner(a, b):
            a_touches = a.get('touches', 0)
            b_touches = b.get('touches', 0)
            a_pristine = (a_touches == 0)
            b_pristine = (b_touches == 0)
            if a_pristine and not b_pristine:
                return a
            if b_pristine and not a_pristine:
                return b

            a_fvg = a['fvg'].get('exists', False)
            b_fvg = b['fvg'].get('exists', False)
            if a_fvg and not b_fvg:
                return a
            if b_fvg and not a_fvg:
                return b

            if a['bos_idx'] > b['bos_idx']:
                return a
            if b['bos_idx'] > a['bos_idx']:
                return b

            logging.warning(
                f"Dedupe Test 4 triggered — identical bos_idx {a['bos_idx']} "
                f"for direction {a['direction']}. Keeping first."
            )
            return a

        by_dir = {}
        for o in obs:
            by_dir.setdefault(o['direction'], []).append(o)

        kept = []
        for direction, group in by_dir.items():
            if len(group) == 1:
                kept.extend(group)
                continue
            survivors = []
            for cand in group:
                thresh = cand.get('_dedupe_thresh', 0.00030)
                merged = False
                for idx, surv in enumerate(survivors):
                    if abs(cand['proximal_line'] - surv['proximal_line']) <= thresh:
                        winner = _pick_winner(cand, surv)
                        survivors[idx] = winner
                        merged = True
                        break
                if not merged:
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

        # --- Zone band ---
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

        # --- OB candle outline (light purple, no text label) ---
        if 0 <= ob_plot_idx < n_plot:
            ob_h = float(H[ob_plot_idx])
            ob_l = float(L[ob_plot_idx])
            ax.add_patch(patches.Rectangle(
                (ob_plot_idx - 0.5, ob_l), 1.0, ob_h - ob_l,
                fill=False, edgecolor='#d7bde2', linewidth=2.0, zorder=4,
                linestyle='-'
            ))

        # --- FVG: outline middle (displacement) candle only, slightly wider for mitigation visibility ---
        fvg_active = ob['fvg']['exists'] and ob['fvg']['c1_idx'] is not None
        fvg_ghost  = (not ob['fvg']['exists']) and ob['fvg'].get('was_detected') and ob['fvg'].get('ghost_c1_idx') is not None
        fvg_partial = fvg_active and ob['fvg'].get('mitigation') == 'partial'

        if fvg_active:
            ft = float(ob['fvg']['fvg_top'])
            fb = float(ob['fvg']['fvg_bottom'])
            mid_abs = ob['fvg']['c1_idx'] + 1
            mid_local = mid_abs - window_start
            if 0 <= mid_local < n_plot:
                if fvg_partial:
                    face_col, edge_col = '#a8e6a1', '#7ed67e'
                else:
                    face_col, edge_col = '#27ae60', '#2ecc71'
                fvg_x_start = mid_local - 0.6
                fvg_width   = 1.8 + 1.2
                ax.add_patch(patches.Rectangle(
                    (fvg_x_start, fb), fvg_width, ft - fb,
                    facecolor=face_col, alpha=0.25, zorder=1
                ))
                ax.add_patch(patches.Rectangle(
                    (fvg_x_start, fb), fvg_width, ft - fb,
                    fill=False, edgecolor=edge_col, linewidth=1.0,
                    linestyle='--', zorder=2
                ))
        elif fvg_ghost:
            ft = float(ob['fvg']['ghost_top'])
            fb = float(ob['fvg']['ghost_bottom'])
            mid_abs = ob['fvg']['ghost_c1_idx'] + 1
            mid_local = mid_abs - window_start
            if 0 <= mid_local < n_plot:
                fvg_x_start = mid_local - 0.6
                fvg_width   = 1.8 + 1.2
                ax.add_patch(patches.Rectangle(
                    (fvg_x_start, fb), fvg_width, ft - fb,
                    facecolor='#888888', alpha=0.10, zorder=1
                ))
                ax.add_patch(patches.Rectangle(
                    (fvg_x_start, fb), fvg_width, ft - fb,
                    fill=False, edgecolor='#888888', linewidth=1.0,
                    linestyle=':', zorder=2
                ))

        # --- BOS/CHoCH horizontal line ---
        bos_price = float(ob['bos_swing_price'])
        bos_color = '#00bcd4' if ob['bos_tag'] == 'BOS' else '#ff9800'
        ax.axhline(
            y=bos_price, color=bos_color, linewidth=0.8,
            linestyle='--', alpha=0.7, zorder=2
        )

        # --- BOS/CHoCH break candle outline ---
        br_start, br_end = smc_detector.compute_h1_break_candle_span(full_df, ob, None)
        if br_start is not None and br_end is not None:
            for abs_i in range(br_start, br_end + 1):
                if abs_i < window_start:
                    continue
                local_i = abs_i - window_start
                if 0 <= local_i < n_plot:
                    c_h = float(H[local_i])
                    c_l = float(L[local_i])
                    ax.add_patch(patches.Rectangle(
                        (local_i - 0.5, c_l), 1.0, c_h - c_l,
                        fill=False, edgecolor=bos_color, linewidth=1.5, zorder=5
                    ))

        # --- Current price line ---
        current_price = float(C[-1])
        ax.axhline(
            y=current_price, color='#ffffff', linewidth=0.8,
            linestyle='-', alpha=0.5, zorder=2
        )

        # --- Mid-chart tags: proximal, distal, BOS/CHoCH, current (numbers only, colour-matched) ---
        # Build pair_conf shim for stack_labels (it needs decimal_places + pair_type).
        pair_type_guess = "forex"
        if dp == 3:
            pair_type_guess = "forex"  # JPY-style
        elif dp == 0 or dp == 1:
            pair_type_guess = "index"
        elif dp == 2:
            pair_type_guess = "commodity"
        pair_conf_shim = {"decimal_places": dp, "pair_type": pair_type_guess}

        mid_x = n_plot / 2.0
        mid_labels = [
            (proximal, f"{proximal:.{dp}f}", '#bb8fce'),
            (distal, f"{distal:.{dp}f}", '#bb8fce'),
            (bos_price, f"{bos_price:.{dp}f}", bos_color),
            (current_price, f"{current_price:.{dp}f}", '#ffffff'),
        ]
        mid_stacked = smc_detector.stack_labels(mid_labels, pair_conf_shim)
        for adj_price, text, color in mid_stacked:
            ax.text(mid_x, adj_price, text, color=color, fontsize=10, va='center',
                    ha='center', fontweight='bold', zorder=5,
                    bbox=dict(facecolor='#131722', edgecolor='none', pad=1.5, alpha=0.75))

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
        ax.tick_params(colors='#888', labelsize=9)
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
    if ob['fvg'].get('exists'):
        mit = ob['fvg'].get('mitigation', 'pristine')
        if mit == 'partial':
            fvg_status = (
                f"partially mitigated FVG between {ob['fvg']['fvg_bottom']:.{dp}f} "
                f"and {ob['fvg']['fvg_top']:.{dp}f} (price tagged proximal, distal intact)"
            )
        else:
            fvg_status = (
                f"pristine FVG between {ob['fvg']['fvg_bottom']:.{dp}f} "
                f"and {ob['fvg']['fvg_top']:.{dp}f}"
            )
    elif ob['fvg'].get('was_detected'):
        fvg_status = "FVG fully mitigated — zone relies on OB alone"
    else:
        fvg_status = "no FVG present"
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
    if ob['fvg'].get('exists'):
        mit = ob['fvg'].get('mitigation', 'pristine')
        if mit == 'partial':
            fvg_line = (
                f"FVG partially mitigated between {ob['fvg']['fvg_bottom']:.{dp}f}–{ob['fvg']['fvg_top']:.{dp}f} "
                f"— proximal tagged, distal still intact."
            )
        else:
            fvg_line = (
                f"FVG confirmed between {ob['fvg']['fvg_bottom']:.{dp}f}–{ob['fvg']['fvg_top']:.{dp}f}, "
                f"adding displacement confluence."
            )
    elif ob['fvg'].get('was_detected'):
        fvg_line = "FVG fully mitigated — zone relies on OB alone for confluence."
    else:
        fvg_line = "No FVG present — zone relies on OB alone for confluence."
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
        if z['fvg_valid'] and z.get('fvg_mitigation') == 'partial':
            fvg_cell = "&#9680;"   # half-filled circle = partial
            fvg_col  = '#7ed67e'   # light green
        elif z['fvg_valid']:
            fvg_cell = "&#10003;"
            fvg_col  = '#27ae60'
        elif z.get('fvg_ghost'):
            fvg_cell = "&#9675;"   # empty circle = fully mitigated ghost
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

def _phase1_chart_legend_html(bos_tag="BOS"):
    """Colour-code legend rendered below H1 zone chart in Phase 1 digest. Cosmetic only."""
    bos_color = '#00bcd4' if bos_tag == 'BOS' else '#ff9800'
    items = [
        ('#bb8fce', 'Zone band (proximal/distal)'),
        ('#d7bde2', 'OB candle outline'),
        ('#2ecc71', 'FVG (displacement)'),
        ('#888888', 'FVG mitigated (ghost)'),
        (bos_color, f'{bos_tag} break candle / level'),
        ('#ffffff', 'Current price'),
    ]
    rows = "".join(
        f'<span style="display:inline-block;margin:2px 10px 2px 0;font-size:11px;color:#bbb;">'
        f'<span style="display:inline-block;width:10px;height:10px;background:{c};'
        f'border-radius:2px;vertical-align:middle;margin-right:5px;"></span>{txt}</span>'
        for c, txt in items
    )
    return (
        f'<div style="margin:8px 0 0 0;padding:8px 10px;background:#0d0d1a;'
        f'border-radius:4px;line-height:1.8;">{rows}</div>'
    )
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

    # Sweep observation badge (display-only; Phase 2 re-grades independently)
    sweep_obs = ob.get('sweep_observed', {}) or {}
    if sweep_obs.get('exists'):
        tier = sweep_obs.get('tier', 'weak')
        tier_color = {'textbook': '#27ae60', 'decent': '#e67e22', 'weak': '#888'}.get(tier, '#888')
        tier_emoji = {'textbook': '🎯', 'decent': '◐', 'weak': '·'}.get(tier, '·')
        sweep_line = (
            f"Sweep: <span style='color:{tier_color};'>"
            f"{tier_emoji} {tier.title()} ({sweep_obs.get('tf', 'H1')} "
            f"@ {sweep_obs.get('price', 0):.{dp}f})</span>"
        )
    else:
        sweep_line = "Sweep: <span style='color:#888;'>None observed</span>"

    pip_unit  = 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)
    zone_pips = round(abs(ob['proximal_line'] - ob['distal_line']) / pip_unit, 1)

    chart_html = (
        f'<img src="cid:{cid}" style="width:100%;max-width:600px;border-radius:6px;'
        f'border:1px solid #2a2a3e;display:block;" />'
        if cid else
        '<div style="padding:8px 12px;background:#2d1a1a;border-left:3px solid #e74c3c;border-radius:4px;color:#e74c3c;font-size:11px;">&#9888; Chart failed to render.</div>'
    )
    legend_html = _phase1_chart_legend_html(ob.get('bos_tag', 'BOS'))

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
        <span style="font-size:11px;color:#aaa;">{sweep_line}</span>
      </div>
      <p style="font-size:12px;color:#bbb;line-height:1.6;margin:0 0 12px 0;
                border-left:3px solid #2a2a3e;padding-left:10px;">
        {narrative}
      </p>
      {chart_html}
      {legend_html}
    </div>"""

def build_active_zone_card_html(sz, name, dp, narrative, cid, ist_timestamp):
    """
    Render an active zone card. Used for both NEW and UNCHANGED active zones.
    NEW badge is rendered inline based on sz['is_new_this_scan'].
    H1 ATR shown in zone metrics.
    """
    direction  = "Bullish (Demand)" if sz['direction'] == 'bullish' else "Bearish (Supply)"
    dir_color  = '#27ae60' if sz['direction'] == 'bullish' else '#e74c3c'
    status_label = sz.get('status_label', 'Pristine')
    stat_color = '#27ae60' if 'Pristine' in status_label else '#e67e22'

    fvg = sz.get('fvg', {})
    mit = fvg.get('mitigation', 'none')
    if fvg.get('exists') and mit == 'partial':
        fvg_line = (
            f"FVG: <span style='color:#7ed67e;'>◐ Partial "
            f"{fvg['fvg_bottom']:.{dp}f} – {fvg['fvg_top']:.{dp}f}</span>"
        )
    elif fvg.get('exists'):
        fvg_line = (
            f"FVG: <span style='color:#27ae60;'>✓ {fvg['fvg_bottom']:.{dp}f} – "
            f"{fvg['fvg_top']:.{dp}f}</span>"
        )
    elif fvg.get('was_detected'):
        gb = fvg.get('ghost_bottom')
        gt = fvg.get('ghost_top')
        fvg_line = (
            f"FVG: <span style='color:#888;'>✗ Mitigated "
            f"({gb:.{dp}f} – {gt:.{dp}f})</span>"
        )
    else:
        fvg_line = "FVG: <span style='color:#888;'>None</span>"

    pip_unit  = 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)
    zone_pips = round(abs(sz['proximal_line'] - sz['distal_line']) / pip_unit, 1)
    h1_atr_val = sz.get('h1_atr', 0.0)
    atr_display = f"{h1_atr_val:.{dp}f}" if h1_atr_val > 0 else "—"

    is_new = sz.get('is_new_this_scan', False)
    new_badge = ""
    if is_new:
        new_badge = (
            "<span style='background:#27ae60;color:#fff;font-size:9px;padding:2px 6px;"
            "border-radius:3px;margin-left:6px;font-weight:bold;'>NEW</span>"
        )

    chart_html = (
        f'<img src="cid:{cid}" style="width:100%;max-width:600px;border-radius:6px;'
        f'border:1px solid #2a2a3e;display:block;" />'
        if cid else
        '<div style="padding:8px 12px;background:#2d1a1a;border-left:3px solid #e74c3c;'
        'border-radius:4px;color:#e74c3c;font-size:11px;">&#9888; Chart unavailable.</div>'
    )
    legend_html = _phase1_chart_legend_html(sz.get('bos_tag', 'BOS'))

    return f"""
    <div style="margin-bottom:28px;padding:16px;background:#1a1a2e;border-radius:8px;
                border-left:4px solid {dir_color};">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;
                  margin-bottom:10px;flex-wrap:wrap;gap:6px;">
        <div>
          <span style="font-size:10px;color:#888;font-family:monospace;margin-right:6px;">{sz['zone_id']}</span>
          <span style="font-size:14px;font-weight:bold;color:#eee;">{name}</span>
          <span style="font-size:12px;color:{dir_color};margin-left:8px;">{direction}</span>
          {new_badge}
        </div>
        <span style="font-size:10px;color:#666;">{ist_timestamp} IST</span>
      </div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;">
        <span style="font-size:11px;color:#aaa;">
          <b style="color:#bb8fce;">Proximal</b> {sz['proximal_line']:.{dp}f}
        </span>
        <span style="font-size:11px;color:#aaa;">
          <b style="color:#bb8fce;">Distal</b> {sz['distal_line']:.{dp}f}
        </span>
        <span style="font-size:11px;color:#aaa;">
          <b>Width</b> {zone_pips} pips
        </span>
        <span style="font-size:11px;color:{stat_color};">
          <b>Status</b> {status_label}
        </span>
        <span style="font-size:11px;color:#aaa;">
          <b style="color:#bb8fce;">H1 ATR</b> {atr_display}
        </span>
        <span style="font-size:11px;color:#aaa;">{fvg_line}</span>
      </div>
      <p style="font-size:12px;color:#bbb;line-height:1.6;margin:0 0 12px 0;
                border-left:3px solid #2a2a3e;padding-left:10px;">
        {narrative}
      </p>
      {chart_html}
      {legend_html}
    </div>"""


def build_dropped_zone_line(sz, name, dp):
    """One-line note for a dropped zone."""
    reason_map = {
        "mitigated_distal_break": "mitigated — close broke distal",
        "mitigated_three_touches": "mitigated — proximal hit 3 times",
        "out_of_proximity": "moved beyond 4× H1 ATR from price",
        "structure_supplanted": "replaced by fresher structure (same leg)",
        "aged_out_of_window": "OB candle aged out of H1 data window",
        "data_unavailable": "pair fetch failed — zone unverifiable",
        "data_stale": "yfinance data stale — zone unverifiable"
    }
    reason_text = reason_map.get(sz.get("drop_reason", ""), sz.get("drop_reason", "unknown"))
    direction = "Bullish" if sz['direction'] == 'bullish' else "Bearish"
    dir_color = '#27ae60' if sz['direction'] == 'bullish' else '#e74c3c'
    return f"""
    <div style="margin-bottom:6px;padding:7px 10px;background:#2a1a1a;
                border-left:3px solid #e74c3c;border-radius:3px;opacity:0.7;">
      <span style="font-size:10px;color:#888;font-family:monospace;margin-right:6px;">{sz['zone_id']}</span>
      <span style="font-size:11px;font-weight:bold;color:#aaa;">{name}</span>
      <span style="font-size:11px;color:{dir_color};margin-left:6px;">{direction}</span>
      <span style="font-size:11px;color:#666;margin-left:6px;font-family:monospace;">
        {sz['proximal_line']:.{dp}f} / {sz['distal_line']:.{dp}f}
      </span>
      <span style="font-size:11px;color:#e74c3c;margin-left:8px;">
        DROPPED — {reason_text}
      </span>
    </div>"""


def generate_zone_narrative_with_atr(ob, name, dp, current_price, h1_atr):
    """
    Wrapper around generate_zone_narrative that injects H1 ATR into the prompt
    and fallback. Calls Gemini with ATR-aware prompt; falls back to local
    narrative on failure.
    """
    if not GEMINI_API_KEY:
        return _fallback_narrative_with_atr(ob, name, dp, current_price, h1_atr)

    direction    = "bullish demand" if ob['direction'] == 'bullish' else "bearish supply"
    proximal     = ob['proximal_line']
    distal       = ob['distal_line']
    pip_unit     = 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)
    zone_pips    = round(abs(proximal - distal) / pip_unit, 1)
    dist_pips    = round(abs(current_price - proximal) / pip_unit, 1)
    atr_display  = f"{h1_atr:.{dp}f}" if h1_atr > 0 else "n/a"

    if ob['fvg'].get('exists'):
        mit = ob['fvg'].get('mitigation', 'pristine')
        if mit == 'partial':
            fvg_status = (
                f"partially mitigated FVG between {ob['fvg']['fvg_bottom']:.{dp}f} "
                f"and {ob['fvg']['fvg_top']:.{dp}f} (price tagged proximal, distal intact)"
            )
        else:
            fvg_status = (
                f"pristine FVG between {ob['fvg']['fvg_bottom']:.{dp}f} "
                f"and {ob['fvg']['fvg_top']:.{dp}f}"
            )
    elif ob['fvg'].get('was_detected'):
        fvg_status = "FVG fully mitigated — zone relies on OB alone"
    else:
        fvg_status = "no FVG present"
    ratio = round(ob['ob_body'] / ob['median_leg_body'], 2) if ob['median_leg_body'] > 0 else 0

    prompt = f"""You are a veteran SMC (Smart Money Concepts) prop trader writing a zone briefing for another experienced SMC trader.
Be direct. No fluff. No pleasantries. Four sentences only. One paragraph.

ZONE DATA — use these exact values, do not recalculate:
- Pair: {name}
- Bias: {direction} | Structure event: {ob['bos_tag']}
- Proximal: {proximal:.{dp}f} | Distal: {distal:.{dp}f}
- Zone width: {zone_pips} pips
- OB body vs median impulse leg: {ob['ob_body']:.{dp}f} vs {ob['median_leg_body']:.{dp}f} (ratio: {ratio}x)
- FVG: {fvg_status}
- Zone status: {ob.get('status', 'Pristine')}
- Current H1 ATR: {atr_display}
- Current price: {current_price:.{dp}f} | Distance to proximal: {dist_pips} pips

WRITE EXACTLY FOUR SENTENCES IN THIS ORDER:
1. What structure event ({ob['bos_tag']}) created this zone and why institutional accumulation is likely here.
2. OB quality: assess tightness (ratio {ratio}x), and whether pristine or tested means strength or caution.
3. FVG assessment: displacement confirmation present or absent, and what that means for zone conviction.
4. Current price context: distance to zone ({dist_pips} pips) measured against H1 ATR ({atr_display}), whether price is approaching or far, and what to watch for.

STRICT OUTPUT RULES:
- Plain text only
- No bullet points, no headers, no markdown, no bold, no numbers
- Four sentences, one paragraph
- Do not repeat the zone levels in every sentence"""

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=300,
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
            )
        )
        response = model.generate_content(prompt)
        text = response.text.strip()
        if len(text) < 50:
            return _fallback_narrative_with_atr(ob, name, dp, current_price, h1_atr)
        return text
    except Exception as e:
        logging.error(f"Gemini narrative error for {name}: {e}")
        return _fallback_narrative_with_atr(ob, name, dp, current_price, h1_atr)


def _fallback_narrative_with_atr(ob, name, dp, current_price, h1_atr):
    pip_unit  = 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)
    dist_pips = round(abs(current_price - ob['proximal_line']) / pip_unit, 1)
    atr_display = f"{h1_atr:.{dp}f}" if h1_atr > 0 else "n/a"
    if ob['fvg'].get('exists'):
        mit = ob['fvg'].get('mitigation', 'pristine')
        if mit == 'partial':
            fvg_line = (
                f"FVG partially mitigated between {ob['fvg']['fvg_bottom']:.{dp}f}–{ob['fvg']['fvg_top']:.{dp}f} "
                f"— proximal tagged, distal still intact."
            )
        else:
            fvg_line = (
                f"FVG confirmed between {ob['fvg']['fvg_bottom']:.{dp}f}–{ob['fvg']['fvg_top']:.{dp}f}, "
                f"adding displacement confluence."
            )
    elif ob['fvg'].get('was_detected'):
        fvg_line = "FVG fully mitigated — zone relies on OB alone for confluence."
    else:
        fvg_line = "No FVG present — zone relies on OB alone for confluence."
    return (
        f"{ob['bos_tag']} confirmed the {ob['direction']} shift; this OB marks the last institutional "
        f"accumulation before the break. "
        f"OB body ratio {round(ob['ob_body']/ob['median_leg_body'],2):.2f}x vs median leg. "
        f"{fvg_line} "
        f"Current price is {dist_pips} pips from proximal (H1 ATR {atr_display}) — "
        f"{'approaching zone, watch for reaction.' if dist_pips < 50 else 'still distant, no immediate action.'}"
    )
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

def send_master_digest_v2(summary_table_html, new_zone_cards, unchanged_zone_cards,
                          dropped_lines, attachments, zone_count, ist_time):
    """
    Daily-slate digest. Three sections:
      - NEW (full cards with charts)
      - UNCHANGED ACTIVE (full cards with charts — every active zone gets chart)
      - DROPPED (one-line notes with reason)
    """
    new_html      = "".join(new_zone_cards)
    unchanged_html = "".join(unchanged_zone_cards)
    dropped_html  = "".join(dropped_lines)

    sections = ""
    if new_zone_cards:
        sections += f"""
        <div style="margin-bottom:20px;">
          <h3 style="color:#27ae60;font-size:12px;letter-spacing:1px;margin:0 0 10px 0;
                     text-transform:uppercase;border-bottom:1px solid #1e3a2f;padding-bottom:6px;">
            New Zones ({len(new_zone_cards)})
          </h3>
          {new_html}
        </div>"""

    if unchanged_zone_cards:
        sections += f"""
        <div style="margin-bottom:20px;">
          <h3 style="color:#7a8aa6;font-size:12px;letter-spacing:1px;margin:0 0 10px 0;
                     text-transform:uppercase;border-bottom:1px solid #1a2a3a;padding-bottom:6px;">
            Active Zones — Refreshed ({len(unchanged_zone_cards)})
          </h3>
          {unchanged_html}
        </div>"""

    if dropped_lines:
        sections += f"""
        <div style="margin-bottom:8px;">
          <h3 style="color:#e74c3c;font-size:11px;letter-spacing:1px;margin:0 0 8px 0;
                     text-transform:uppercase;border-bottom:1px solid #2a1a1a;padding-bottom:6px;">
            Dropped This Scan ({len(dropped_lines)})
          </h3>
          {dropped_html}
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
      SMC Alert Engine v2.1 · Daily Slate · {ist_time} IST
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

    logging.info(f"Digest v2 sent: {zone_count} active zones, "
                 f"{len(new_zone_cards)} new, {len(unchanged_zone_cards)} unchanged, "
                 f"{len(dropped_lines)} dropped.")
    print(f"Digest sent: {zone_count} active, {len(dropped_lines)} dropped.")
                              
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
# DAILY SLATE — state file structure
# ---------------------------------------------------------------------------
# active_obs.json schema (overwritten only on first scan of each trading day):
#
# {
#   "slate_date": "YYYY-MM-DD",                    # IST date of the slate
#   "slate_started_iso": "...",                    # First-scan timestamp
#   "pairs": {
#     "EURUSD": {
#       "next_id_counter": int,                    # Per-pair counter
#       "zones": [
#         {
#           "zone_id": "EUR01",
#           "status": "active" | "dropped",
#           "drop_reason": null | "mitigated" | "out_of_proximity" | "structure_invalidated" | "stale_data",
#           "first_seen_iso": "...", "first_seen_label": "HH:MM IST",
#           "last_seen_iso": "...", "last_seen_label": "HH:MM IST",
#           "is_new_this_scan": bool,
#           "ob_timestamp": "...",                 # Identity anchor
#           "direction": "bullish" | "bearish",
#           "bos_tag": "BOS" | "CHoCH",
#           "proximal_line": float, "distal_line": float,
#           "high": float, "low": float,
#           "ob_body": float, "median_leg_body": float,
#           "touches": int, "status_label": "Pristine" | "Tested (Nx)",
#           "h1_atr": float,                       # Refreshed every scan
#           "current_price_at_scan": float,
#           "distance_to_proximal_pips": float,
#           "fvg": {...}
#         }
#       ]
#     }
#   }
# }

SLATE_FILE = "active_obs.json"


def get_today_ist_date_str():
    """Return today's date string in IST as YYYY-MM-DD."""
    return get_ist_now().strftime('%Y-%m-%d')


def load_slate():
    """Load slate, return empty structure if missing or malformed."""
    raw = load_json_safe(SLATE_FILE, {})
    if not isinstance(raw, dict) or "pairs" not in raw:
        return {"slate_date": None, "slate_started_iso": None, "pairs": {}}
    return raw


def save_slate(slate):
    save_json_atomic(SLATE_FILE, slate)


def init_fresh_slate(ist_now, pair_names):
    """Build empty slate for a new trading day. Counters at zero per pair."""
    return {
        "slate_date": ist_now.strftime('%Y-%m-%d'),
        "slate_started_iso": ist_now.isoformat(),
        "pairs": {name: {"next_id_counter": 0, "zones": []} for name in pair_names}
    }


def find_matching_slate_zone(fresh_zone, slate_zones, pair_type):
    """
    Identity match: same direction AND
      (a) same ob_timestamp, OR
      (b) proximal_line within pair-aware threshold (timestamp drift fallback).
    Returns the matching slate zone dict (still in 'active' state) or None.
    """
    threshold = ZONE_PROXIMITY_THRESHOLDS.get(pair_type, 0.0003)
    for sz in slate_zones:
        if sz.get("status") != "active":
            continue
        if sz.get("direction") != fresh_zone["direction"]:
            continue
        # (a) Exact timestamp match
        if sz.get("ob_timestamp") and fresh_zone.get("ob_timestamp"):
            if sz["ob_timestamp"] == fresh_zone["ob_timestamp"]:
                return sz
        # (b) Proximal-line proximity fallback
        if abs(sz["proximal_line"] - fresh_zone["proximal_line"]) <= threshold:
            return sz
    return None


def assign_new_zone_id(slate_pair_block, pair_name):
    slate_pair_block["next_id_counter"] = slate_pair_block.get("next_id_counter", 0) + 1
    prefix = ZONE_ID_PREFIX.get(pair_name, pair_name[:3].upper())
    return f"{prefix}{slate_pair_block['next_id_counter']:02d}"


def fresh_to_slate_zone(fresh_zone, zone_id, ist_now, current_price, dp):
    """Materialize a fresh-detection dict into a slate zone record."""
    pip_unit = 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)
    dist_pips = round(abs(current_price - fresh_zone['proximal_line']) / pip_unit, 1)
    label = ist_now.strftime('%H:%M IST')
    iso = ist_now.isoformat()
    return {
        "zone_id": zone_id,
        "status": "active",
        "drop_reason": None,
        "first_seen_iso": iso,
        "first_seen_label": label,
        "last_seen_iso": iso,
        "last_seen_label": label,
        "is_new_this_scan": True,
        "ob_timestamp": fresh_zone.get("ob_timestamp"),
        "direction": fresh_zone["direction"],
        "bos_tag": fresh_zone["bos_tag"],
        "proximal_line": fresh_zone["proximal_line"],
        "distal_line": fresh_zone["distal_line"],
        "high": fresh_zone["high"],
        "low": fresh_zone["low"],
        "ob_body": fresh_zone["ob_body"],
        "median_leg_body": fresh_zone["median_leg_body"],
        "bos_idx": fresh_zone["bos_idx"],
        "ob_idx": fresh_zone["ob_idx"],
        "impulse_start_idx": fresh_zone["impulse_start_idx"],
        "impulse_start_price": fresh_zone["impulse_start_price"],
        "bos_swing_price": fresh_zone["bos_swing_price"],
        "touches": fresh_zone.get("touches", 0),
        "status_label": fresh_zone.get("status", "Pristine"),
        "h1_atr": fresh_zone.get("h1_atr", 0.0),
        "current_price_at_scan": current_price,
        "distance_to_proximal_pips": dist_pips,
        "fvg": fresh_zone["fvg"]
    }


def refresh_slate_zone(slate_zone, fresh_zone, ist_now, current_price, dp):
    """
    Update an existing slate zone with fresh-scan data. Identity (zone_id,
    first_seen) preserved. Mutable state (touches, status, fvg, atr, price)
    refreshed from latest scan.
    """
    pip_unit = 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)
    dist_pips = round(abs(current_price - fresh_zone['proximal_line']) / pip_unit, 1)
    slate_zone["last_seen_iso"] = ist_now.isoformat()
    slate_zone["last_seen_label"] = ist_now.strftime('%H:%M IST')
    slate_zone["is_new_this_scan"] = False
    # Refresh structural fields (in case proximal drifted within threshold).
    slate_zone["proximal_line"] = fresh_zone["proximal_line"]
    slate_zone["distal_line"]   = fresh_zone["distal_line"]
    slate_zone["high"]          = fresh_zone["high"]
    slate_zone["low"]           = fresh_zone["low"]
    slate_zone["ob_body"]       = fresh_zone["ob_body"]
    slate_zone["median_leg_body"] = fresh_zone["median_leg_body"]
    slate_zone["bos_idx"]       = fresh_zone["bos_idx"]
    slate_zone["ob_idx"]        = fresh_zone["ob_idx"]
    slate_zone["touches"]       = fresh_zone.get("touches", 0)
    slate_zone["status_label"]  = fresh_zone.get("status", "Pristine")
    slate_zone["h1_atr"]        = fresh_zone.get("h1_atr", 0.0)
    slate_zone["current_price_at_scan"] = current_price
    slate_zone["distance_to_proximal_pips"] = dist_pips
    slate_zone["fvg"]           = fresh_zone["fvg"]


def determine_drop_reason(slate_zone, current_price, df, h1_atr, fresh_zones_in_pair, pair_type):
    """
    Return concrete drop reason. Every drop must map to ONE of these checks.
    No "unknown" or "structure_invalidated" fallback — silent fails are the
    failure mode we are explicitly preventing.

    Returns one of:
      'mitigated_distal_break'
      'mitigated_three_touches'
      'out_of_proximity'
      'structure_supplanted'
      'aged_out_of_window'
      'data_unavailable'
      'data_stale'
      None  -> zone should NOT be dropped; caller keeps it alive and logs.
    """
    # --- data_unavailable handled by caller before this is called ---

    # --- out_of_proximity ---
    if h1_atr and h1_atr > 0:
        if abs(current_price - slate_zone['proximal_line']) > 4.0 * h1_atr:
            return "out_of_proximity"

    # --- aged_out_of_window ---
    # Slate zone's OB candle is older than the oldest candle in fresh df.
    if df is not None and len(df) > 0 and slate_zone.get("ob_timestamp"):
        try:
            slate_ob_dt = datetime.fromisoformat(slate_zone["ob_timestamp"])
            if slate_ob_dt.tzinfo is not None:
                slate_ob_dt = slate_ob_dt.replace(tzinfo=None)
            first_ts_raw = df['Datetime'].iloc[0] if 'Datetime' in df.columns else df.index[0]
            if hasattr(first_ts_raw, 'to_pydatetime'):
                first_dt = first_ts_raw.to_pydatetime()
            else:
                first_dt = first_ts_raw
            if hasattr(first_dt, 'tzinfo') and first_dt.tzinfo is not None:
                first_dt = first_dt.replace(tzinfo=None)
            if slate_ob_dt < first_dt:
                return "aged_out_of_window"
        except Exception:
            pass  # if we can't compare, fall through to other checks

    # --- mitigated_distal_break / mitigated_three_touches ---
    # Replay candles from OB onwards (or whole df if OB not locatable) and
    # check distal break OR 3-touch proximal mitigation.
    if df is not None and len(df) > 0:
        try:
            C = df['Close'].values.astype(float)
            H = df['High'].values.astype(float)
            L = df['Low'].values.astype(float)
            distal = slate_zone['distal_line']
            proximal = slate_zone['proximal_line']
            direction = slate_zone['direction']

            # Find scan start index — from OB candle if locatable, else whole df.
            scan_start = 0
            if slate_zone.get("ob_timestamp"):
                try:
                    target_ts = datetime.fromisoformat(slate_zone["ob_timestamp"])
                    if target_ts.tzinfo is not None:
                        target_ts = target_ts.replace(tzinfo=None)
                    ts_col = df['Datetime'] if 'Datetime' in df.columns else pd.Series(df.index)
                    for k, t in enumerate(ts_col):
                        kt = t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t
                        if hasattr(kt, 'tzinfo') and kt.tzinfo is not None:
                            kt = kt.replace(tzinfo=None)
                        if kt >= target_ts:
                            scan_start = k + 1  # candles AFTER the OB candle
                            break
                except Exception:
                    scan_start = 0

            touches = 0
            for m in range(scan_start, len(C)):
                if direction == 'bullish':
                    if C[m] < distal:
                        return "mitigated_distal_break"
                    if L[m] <= proximal:
                        touches += 1
                else:
                    if C[m] > distal:
                        return "mitigated_distal_break"
                    if H[m] >= proximal:
                        touches += 1
                if touches >= 3:
                    return "mitigated_three_touches"
        except Exception:
            pass

    # --- structure_supplanted ---
    # Same direction, overlapping by proximity threshold, fresh zone has
    # HIGHER bos_idx (fresher structure). Means same-leg dedupe in the new
    # scan picked a different OB representing the same zone.
    threshold = ZONE_PROXIMITY_THRESHOLDS.get(pair_type, 0.0003)
    slate_bos_idx = slate_zone.get("bos_idx", -1)
    for fz in fresh_zones_in_pair:
        if fz.get("direction") != slate_zone.get("direction"):
            continue
        if abs(fz.get("proximal_line", 0) - slate_zone.get("proximal_line", 0)) > threshold:
            continue
        if fz.get("bos_idx", -1) > slate_bos_idx:
            return "structure_supplanted"

    # --- No concrete reason matched ---
    # Do NOT drop. Log diagnostic so we can investigate.
    return None


def log_unverified_drop_attempt(slate_zone, pair_name, ist_now):
    """A zone disappeared from fresh scan but no concrete drop reason fires.
    Log it; keep zone in slate. This is our silent-fail guardrail."""
    try:
        log = load_json_safe("drop_diagnostic_log.json", [])
        log.append({
            "ts": ist_now.isoformat(),
            "pair": pair_name,
            "zone_id": slate_zone.get("zone_id"),
            "direction": slate_zone.get("direction"),
            "proximal": slate_zone.get("proximal_line"),
            "distal": slate_zone.get("distal_line"),
            "ob_timestamp": slate_zone.get("ob_timestamp"),
            "note": "Zone missing from fresh scan but no concrete drop reason matched. Kept in slate."
        })
        log = log[-300:]
        save_json_atomic("drop_diagnostic_log.json", log)
    except Exception as e:
        logging.error(f"Drop diagnostic log write failed: {e}")

# ---------------------------------------------------------------------------
# MAIN RUNNER — DAILY SLATE MODEL
# ---------------------------------------------------------------------------

def run_radar():
    ist_now      = get_ist_now()
    ist_time_str = ist_now.strftime('%H:%M')
    ist_ts_full  = ist_now.strftime('%d %b %Y, %H:%M')
    today_str    = get_today_ist_date_str()

    # Blackout: do nothing before 09:00 IST
    if ist_now.hour < 9:
        print(f"Blackout active. Suppressed until 09:00 IST.")
        return

    pair_names = [p['name'] for p in config_master['pairs']]

    # --- LOAD SLATE; START FRESH IF NEW TRADING DAY ---
    slate = load_slate()
    is_new_slate = (slate.get("slate_date") != today_str)
    if is_new_slate:
        slate = init_fresh_slate(ist_now, pair_names)
        print(f"  [SLATE] New trading day {today_str} — slate reset, counters at zero.")
    else:
        # Mark every existing zone is_new_this_scan = False at start of scan.
        for pname, pblock in slate.get("pairs", {}).items():
            for z in pblock.get("zones", []):
                z["is_new_this_scan"] = False

    # Ensure all configured pairs have a block (handles config additions mid-day).
    for name in pair_names:
        if name not in slate["pairs"]:
            slate["pairs"][name] = {"next_id_counter": 0, "zones": []}

    # --- EMAIL GATE ---
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

    dp_map = {p['name']: p.get('decimal_places', 5) for p in config_master['pairs']}
    pair_type_map = {p['name']: p.get('pair_type', 'forex') for p in config_master['pairs']}

    # Track per-scan fetch outcomes — for skipping zones with no fresh data
    pairs_with_fresh_data = set()
    pair_dfs = {}  # pair_name -> df (for drop-reason analysis)
    pair_atrs = {}  # pair_name -> h1_atr
    pair_prices = {}  # pair_name -> current_price

    # --- SCAN EACH PAIR, RECONCILE WITH SLATE ---
    audit_rows = []

    for pair in config_master["pairs"]:
        ticker   = pair["symbol"]
        name     = pair["name"]
        dp       = pair.get("decimal_places", 5)
        ptype    = pair.get("pair_type", "forex")
        lookback = 5 if name in ["NZDUSD", "GOLD"] else 6 if name == "NAS100" else 4

        df = fetch_data(ticker, pair["map_tf"], "15d")
        if df is None:
            # Cannot verify slate zones for this pair. Keep them alive but
            # tag last_seen so weekly review can spot persistent fetch failures.
            logging.warning(f"No data for {name}. Slate zones held; awaiting next scan.")
            print(f"  [NO DATA] {name}: slate zones held (data_unavailable this scan).")
            slate_pair = slate["pairs"].get(name, {})
            for sz in slate_pair.get("zones", []):
                if sz["status"] == "active":
                    sz["last_data_unavailable_at"] = ist_now.isoformat()
            continue

        pairs_with_fresh_data.add(name)
        pair_dfs[name] = df

        result        = detect_smc_radar(df, lookback, pair_type=ptype)
        current_price = result["current_price"]
        fresh_zones   = result["active_unmitigated_obs"]
        h1_atr        = smc_detector.compute_atr(df) or 0.0

        pair_atrs[name] = h1_atr
        pair_prices[name] = current_price

        # Stamp fresh zones with h1_atr from this scan (already done in detect_smc_radar
        # via 'h1_atr' field, but we ensure consistency here).
        for fz in fresh_zones:
            if not fz.get('h1_atr'):
                fz['h1_atr'] = h1_atr

        # --- RECONCILE FRESH ZONES WITH SLATE ---
        slate_pair = slate["pairs"][name]
        slate_zones = slate_pair["zones"]
        active_slate_zones = [z for z in slate_zones if z["status"] == "active"]

        matched_slate_ids = set()

        # 1. For each fresh zone, find match in slate or add as NEW.
        for fz in fresh_zones:
            match = find_matching_slate_zone(fz, active_slate_zones, ptype)
            if match:
                refresh_slate_zone(match, fz, ist_now, current_price, dp)
                matched_slate_ids.add(match["zone_id"])
            else:
                new_id = assign_new_zone_id(slate_pair, name)
                new_record = fresh_to_slate_zone(fz, new_id, ist_now, current_price, dp)
                slate_zones.append(new_record)
                matched_slate_ids.add(new_id)

        # 2. For each active slate zone NOT matched, determine concrete drop reason.
        # If no concrete reason fires, KEEP the zone alive and log diagnostic.
        # This prevents silent fails masquerading as "unknown" drops.
        for sz in active_slate_zones:
            if sz["zone_id"] in matched_slate_ids:
                continue
            reason = determine_drop_reason(
                sz, current_price, df, h1_atr, fresh_zones, ptype
            )
            if reason is None:
                # No concrete reason. Keep zone alive, log for investigation.
                log_unverified_drop_attempt(sz, name, ist_now)
                # Refresh last_seen so it doesn't accumulate as ghost.
                sz["last_seen_iso"] = ist_now.isoformat()
                sz["last_seen_label"] = ist_now.strftime('%H:%M IST')
                print(f"  [HOLD] {name} {sz['zone_id']} missing from scan but no concrete drop reason — kept in slate, logged.")
                continue
            sz["status"] = "dropped"
            sz["drop_reason"] = reason
            sz["last_seen_iso"] = ist_now.isoformat()
            sz["last_seen_label"] = ist_now.strftime('%H:%M IST')
            print(f"  [DROP] {name} {sz['zone_id']} dropped — {reason}.")

        # Audit log row per active zone post-reconcile
        for sz in slate_zones:
            if sz["status"] == "active":
                audit_rows.append({
                    "zone_id": sz["zone_id"], "pair": name,
                    "direction": sz["direction"], "bos_tag": sz["bos_tag"],
                    "proximal": sz["proximal_line"], "distal": sz["distal_line"],
                    "status": sz["status_label"], "touches": sz.get("touches", 0),
                    "fvg_exists": sz["fvg"].get("exists", False),
                    "fvg_was_detected": sz["fvg"].get("was_detected", False),
                    "current_price": current_price,
                    "h1_atr": sz.get("h1_atr", 0.0),
                    "is_new_this_scan": sz.get("is_new_this_scan", False)
                })

    # --- BUILD EMAIL ---
    if send_email_this_run:
        # Collect renderable zones across all pairs.
        new_zone_cards     = []
        unchanged_zone_cards = []
        dropped_lines      = []
        all_zones_for_table = []
        attachments        = []
        chart_counter      = 0

        for pair_name in pair_names:
            dp = dp_map[pair_name]
            pblock = slate["pairs"].get(pair_name, {})
            for sz in pblock.get("zones", []):
                if sz["status"] == "active":
                    # Build OB-shaped dict for narrative + chart helpers.
                    ob_for_render = {
                        "direction": sz["direction"],
                        "bos_tag": sz["bos_tag"],
                        "proximal_line": sz["proximal_line"],
                        "distal_line": sz["distal_line"],
                        "high": sz["high"], "low": sz["low"],
                        "ob_body": sz["ob_body"],
                        "median_leg_body": sz["median_leg_body"],
                        "ob_idx": sz["ob_idx"],
                        "bos_idx": sz["bos_idx"],
                        "bos_swing_price": sz["bos_swing_price"],
                        "impulse_start_idx": sz["impulse_start_idx"],
                        "impulse_start_price": sz["impulse_start_price"],
                        "fvg": sz["fvg"],
                        "touches": sz.get("touches", 0),
                        "status": sz.get("status_label", "Pristine"),
                        "h1_atr": sz.get("h1_atr", 0.0)
                    }

                    # Summary table row
                    all_zones_for_table.append({
                        "name": pair_name, "zone_id": sz["zone_id"],
                        "direction": sz["direction"],
                        "proximal": sz["proximal_line"], "distal": sz["distal_line"],
                        "bos_tag": sz["bos_tag"], "status": sz.get("status_label", "Pristine"),
                        "fvg_valid": sz["fvg"].get("exists", False),
                        "fvg_ghost": (not sz["fvg"].get("exists", False))
                                      and sz["fvg"].get("was_detected", False),
                        "fvg_mitigation": sz["fvg"].get("mitigation", "none"),
                        "first_seen_ist": sz.get("first_seen_label", "—"),
                        "is_new": sz.get("is_new_this_scan", False),
                        "is_changed": False
                    })

                    # Chart + narrative — only if we have fresh df for this pair.
                    df = pair_dfs.get(pair_name)
                    current_price = pair_prices.get(pair_name, sz.get("current_price_at_scan", 0))
                    chart_b64 = None
                    cid = None
                    if df is not None:
                        cid = f"chart_{pair_name}_{chart_counter}"
                        chart_b64 = generate_h1_chart(df, ob_for_render, dp, pair_name, ist_ts_full)

                    narrative = generate_zone_narrative_with_atr(
                        ob_for_render, pair_name, dp, current_price, sz.get("h1_atr", 0.0)
                    )

                    card_html = build_active_zone_card_html(
                        sz, pair_name, dp, narrative,
                        cid if chart_b64 else None,
                        ist_ts_full
                    )

                    if sz.get("is_new_this_scan", False):
                        new_zone_cards.append(card_html)
                    else:
                        unchanged_zone_cards.append(card_html)

                    if chart_b64:
                        img_mime = MIMEImage(base64.b64decode(chart_b64))
                        img_mime.add_header("Content-ID", f"<{cid}>")
                        img_mime.add_header("Content-Disposition", "inline",
                                            filename=f"{cid}.png")
                        attachments.append(img_mime)
                        chart_counter += 1

                elif sz["status"] == "dropped":
                    dropped_lines.append(build_dropped_zone_line(sz, pair_name, dp))

        if all_zones_for_table or dropped_lines:
            summary_table = build_summary_table_html(all_zones_for_table, dp_map)
            try:
                send_master_digest_v2(
                    summary_table, new_zone_cards, unchanged_zone_cards,
                    dropped_lines, attachments,
                    len(all_zones_for_table), ist_time_str
                )
                gate["last_email_ts"] = ist_now.isoformat()
                save_email_gate(gate)
            except Exception as e:
                logging.error(f"Digest send failed: {e}")
                print(f"  [EMAIL ERR] {e}")
        else:
            print("  No zones in slate. Digest skipped.")

        # 3. Drop-cycle cleanup: remove dropped zones from slate AFTER email sent.
        for pname in pair_names:
            pblock = slate["pairs"].get(pname, {})
            pblock["zones"] = [z for z in pblock.get("zones", []) if z["status"] == "active"]

    else:
        active_count = sum(
            1 for pname in pair_names
            for z in slate["pairs"].get(pname, {}).get("zones", [])
            if z["status"] == "active"
        )
        print(f"  Scan complete. {active_count} active zones in slate. Email gate active.")

    # Always write audit log (used by weekly review only).
    if audit_rows:
        append_audit_log(audit_rows, ist_now)

    # Persist slate (overwrites file — but only mutates within-day, never resets
    # mid-day; new-day reset happens via slate_date check at scan start).
    save_slate(slate)
    print(f"Phase 1 complete at {ist_time_str} IST.")


if __name__ == "__main__":
    run_radar()
