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
import dealing_range
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

# Route runtime logs to stdout — GitHub Actions captures stdout into run logs,
# and locally it's still readable from the terminal. The structured Phase 1
# scan log (state/phase1_scans/YYYY-MM.jsonl) is the source of truth for
# forensic analysis; this stream is for live tail / error reporting only.
logging.basicConfig(
    level=logging.INFO,
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


# ---------------------------------------------------------------------------
# PHASE 1 SCAN LOG (forensic trail)
# ---------------------------------------------------------------------------
# One JSONL line per pair per scan, written to state/phase1_scans/YYYY-MM.jsonl.
# Monthly rotation keeps individual files small and greppable. Records are
# structured so post-hoc analysis can answer "why didn't this fire?" without
# re-running anything.
#
# Record schema (one line):
#   {
#     "scan_ts": "<IST iso>", "source": "scan" | "backfill",
#     "pair": "USDJPY",
#     "current_price": float | null,
#     "trend": "bullish" | "bearish" | null,
#     "walls": {ceiling_price, ceiling_is_placeholder, floor_price,
#               floor_is_placeholder, fallback_active},
#     "last_event": {type, tier, direction, ts, chop},
#     "active_zones": [{id, direction, status, touches, fvg_mitigation,
#                       proximal, distal, distance_to_proximal_pips,
#                       sweep_observed: {exists, tier, price} | null}],
#     "dropped_this_scan": [{id, reason}],
#     "diagnostics": [str]   # optional one-line notes
#   }
# ---------------------------------------------------------------------------
PHASE1_SCAN_DIR = os.path.join("state", "phase1_scans")

def _phase1_scan_log_path(ist_now=None):
    """Return the path for the current month's scan log file."""
    if ist_now is None:
        ist_now = get_ist_now()
    month_str = ist_now.strftime('%Y-%m')
    return os.path.join(PHASE1_SCAN_DIR, f"{month_str}.jsonl")


def _write_phase1_scan_records(records, ist_now=None):
    """Append a batch of scan records to the current month's log file."""
    if not records:
        return
    try:
        os.makedirs(PHASE1_SCAN_DIR, exist_ok=True)
        path = _phase1_scan_log_path(ist_now)
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")
    except Exception as e:
        logging.warning(f"phase1 scan log write failed: {e}")


def _backfill_phase1_scan_log():
    """One-time backfill from structure_state.json events.

    Idempotent: writes only if the current month's log file does not already
    exist. Emits one record per historical structure event (Phase 1 has
    finer-grained scans, but pre-feature we only have event-level history).
    """
    path = _phase1_scan_log_path()
    if os.path.exists(path):
        return
    try:
        state = dealing_range.load_state() or {}
        if not state:
            return
        records = []
        for pair_name, pair_state in state.items():
            events = pair_state.get('events', []) or []
            for ev in events:
                ts = ev.get('candle_ts')
                if not ts:
                    continue
                records.append({
                    'scan_ts': ts,
                    'source': 'backfill',
                    'pair': pair_name,
                    'current_price': None,
                    'trend': ev.get('trend_after'),
                    'walls': {
                        # Backfill cannot reconstruct walls-at-time-of-event
                        # from the snapshot alone — only event facts are kept.
                        'ceiling_price': None,
                        'ceiling_is_placeholder': None,
                        'floor_price': None,
                        'floor_is_placeholder': None,
                        'fallback_active': None,
                    },
                    'last_event': {
                        'type': ev.get('type'),
                        'tier': ev.get('tier'),
                        'direction': ev.get('direction'),
                        'ts': ts,
                        'chop': bool(ev.get('chop', False)),
                        'broken_was_wall': bool(ev.get('broken_was_wall', False)),
                        'broken_swing_price': ev.get('broken_swing_price'),
                        'displacement_atr': ev.get('displacement_atr'),
                        'reversal_pct': ev.get('reversal_pct'),
                    },
                    'active_zones': [],
                    'dropped_this_scan': [],
                    'diagnostics': ['backfilled from structure_state.events'],
                })
        records.sort(key=lambda r: (r.get('scan_ts') or '', r.get('pair') or ''))
        _write_phase1_scan_records(records)
        if records:
            logging.info(f"Phase 1 scan log backfilled with {len(records)} event records.")
    except Exception as e:
        logging.warning(f"phase1 scan log backfill failed: {e}")


def _build_phase1_scan_record(pair_name, ist_now, current_price, walls,
                              slate_zones, fresh_zones, dropped_this_scan,
                              diagnostics=None, placeholder_diagnostic=None):
    """Construct a single per-pair scan record (snapshot of decisions).

    `placeholder_diagnostic`: structured per-side explanation of why a wall
    remains tentative (or that it's anchored). Sourced from
    `dealing_range.update_pair()` under PLACEHOLDER_DIAG_KEY. May be None
    when the walk took an early-return path (no new candle / empty df).
    """
    walls = walls or {}
    active = []
    for sz in slate_zones:
        if sz.get('status') != 'active':
            continue
        sw = sz.get('sweep_observed') or {}
        active.append({
            'id': sz.get('zone_id'),
            'direction': sz.get('direction'),
            'bos_tag': sz.get('bos_tag'),
            'bos_tier': sz.get('bos_tier'),
            'status': sz.get('status_label'),
            'touches': sz.get('touches', 0),
            'proximal': sz.get('proximal_line'),
            'distal': sz.get('distal_line'),
            'distance_to_proximal_pips': sz.get('distance_to_proximal_pips'),
            'fvg_mitigation': (sz.get('fvg') or {}).get('mitigation', 'none'),
            'sweep_observed': {
                'exists': bool(sw.get('exists')),
                'tier': sw.get('tier'),
                'price': sw.get('price'),
            } if sw else None,
        })
    return {
        'scan_ts': ist_now.isoformat(),
        'source': 'scan',
        'pair': pair_name,
        'current_price': current_price,
        'trend': walls.get('trend'),
        'walls': {
            'ceiling_price': walls.get('ceiling_price'),
            'ceiling_is_placeholder': walls.get('ceiling_is_placeholder'),
            'floor_price': walls.get('floor_price'),
            'floor_is_placeholder': walls.get('floor_is_placeholder'),
            'fallback_active': bool(walls.get('fallback_active', False)),
        },
        'last_event': {
            'type': walls.get('last_event_type'),
            'tier': walls.get('last_event_tier'),
            'direction': walls.get('last_event_direction'),
            'ts': walls.get('last_event_ts'),
            'chop': bool(walls.get('last_event_chop', False)),
        },
        'active_zones': active,
        'dropped_this_scan': dropped_this_scan or [],
        'diagnostics': diagnostics or [],
        'placeholder_diagnostic': placeholder_diagnostic,
    }


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
    # Body minimum 20% of candle range. Below this is near-doji territory —
    # no clear directional intent. The backward walk continues to older
    # opposing candles if the most recent one fails this check.
    body = abs(open_p - close_p)
    rng  = high_p - low_p
    if rng == 0:
        return False
    return body > (rng * 0.20)


def _event_label(bos_tag, bos_tier):
    # Single source of truth for the human-readable structural-event label.
    # BOS is single-tier (no Major/Minor split); CHoCH carries the tier.
    if bos_tag == 'BOS':
        return 'BOS'
    if bos_tier == 'Minor':
        return 'Minor CHoCH'
    return 'Major CHoCH'


# ---------------------------------------------------------------------------
# PHASE 1 OB MITIGATION (single source of truth)
# ---------------------------------------------------------------------------
# Rule (LOCKED):
#   - Distal mitigation = CLOSE strictly beyond distal.
#       Bullish OB:  C[m] < distal  → mitigated
#       Bearish OB:  C[m] > distal  → mitigated
#     Strict. No ATR buffer. Wick-only breaches do NOT mitigate.
#   - Touches counted by wick at proximal (a touch = a visit).
#   - 3 touches at proximal = mitigated (overuse mitigation).
#
# Phase 2 / Phase 3 invalidation paths have their own rules (Phase 3 adds
# an ATR buffer on M5 close). This function is Phase 1 only.
# ---------------------------------------------------------------------------
def is_ob_mitigated_phase1(direction, distal, proximal, df, start_idx, end_idx=None):
    """
    Replay candles in [start_idx, end_idx) and apply Phase 1 mitigation.

    Args:
        direction: 'bullish' | 'bearish'.
        distal:    OB distal price.
        proximal:  OB proximal price.
        df:        H1 OHLC dataframe.
        start_idx: first idx to inspect (exclusive of OB candle is up to caller).
        end_idx:   one past last idx to inspect; defaults to len(df).

    Returns:
        (mitigated: bool, reason: Optional[str], touches: int)
        reason ∈ {'mitigated_distal_break', 'mitigated_three_touches'} or None.
    """
    if df is None or len(df) == 0:
        return False, None, 0
    if end_idx is None:
        end_idx = len(df)
    start_idx = max(0, int(start_idx))
    end_idx = min(int(end_idx), len(df))
    if start_idx >= end_idx:
        return False, None, 0

    C = df['Close'].values.astype(float)
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)

    touches = 0
    for m in range(start_idx, end_idx):
        if direction == 'bullish':
            # CLOSE below distal = mitigated. Wick alone does not mitigate.
            if C[m] < distal:
                return True, 'mitigated_distal_break', touches
            # Touch = wick into proximal (visit, not invalidation).
            if L[m] <= proximal:
                touches += 1
        else:
            if C[m] > distal:
                return True, 'mitigated_distal_break', touches
            if H[m] >= proximal:
                touches += 1
        if touches >= 3:
            return True, 'mitigated_three_touches', touches

    return False, None, touches


def detect_smc_radar(df, pair_type="forex", events=None, walls=None, pair_name=None):
    """
    Build OB zones from BOS / CHoCH events emitted by dealing_range.py.

    No detection happens here — the events list is the single source of
    truth for structural events (BOS, Major CHoCH, Minor CHoCH). For each
    event in the ring, we resolve the break candle and impulse-leg start
    from absolute timestamps (idx is NOT portable across scans since the
    yfinance window rolls), then walk back through the impulse leg to find
    the order block, attach an FVG badge, and proximity-gate.

    Args:
        df: H1 OHLC dataframe for the pair.
        pair_type: 'forex' | 'index' | 'commodity' (used for FVG noise floor).
        events: list of event dicts from dealing_range state. If None, the
                function returns no zones (defensive fallback).
        walls: dealing_range pair-state dict for the pair (ceiling/floor +
                placeholder/fallback flags). Used for the PD-array gate
                (bullish OBs only valid in discount, bearish in premium).
                When None or fallback is active, the gate fails open.
    """
    n = len(df)
    O = df['Open'].values
    C = df['Close'].values
    H = df['High'].values
    L = df['Low'].values
    h1_atr_for_leg = smc_detector.compute_atr(df)
    current_price_now = float(C[-1]) if n > 0 else 0.0

    if not events:
        return {"current_price": current_price_now, "active_unmitigated_obs": []}

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

    def _idx_from_ts(ts_iso):
        if not ts_iso:
            return None
        for k in range(len(df)):
            if _ts_for_idx(k) == ts_iso:
                return k
        return None

    # BOS sequence count is recomputed by Phase 2 fresh from state — but we
    # also stamp it on each fresh OB at emit time so downstream slate dedupe /
    # logging has a value. Counter resets on Major CHoCH only (Minor CHoCH
    # does NOT flip trend, so does NOT reset BOS chain).
    bos_seq_counter = 0
    last_choch_local_idx: Optional[int] = None  # idx in this df, or None
    active_obs = []

    for ev in events:
        ev_type = ev.get('type')           # 'BOS' | 'CHoCH'
        ev_tier = ev.get('tier')           # 'Major' | 'Minor'
        ev_dir  = ev.get('direction')      # 'bullish' | 'bearish'
        if ev_type not in ('BOS', 'CHoCH'):
            continue

        # Update BOS chain bookkeeping.
        if ev_type == 'CHoCH' and ev_tier == 'Major':
            bos_seq_counter = 0
        elif ev_type == 'BOS':
            bos_seq_counter += 1
        # Minor CHoCH: do not touch the counter.

        # Locate the break candle and impulse-leg start in CURRENT df.
        bos_idx = _idx_from_ts(ev.get('candle_ts'))
        impulse_start_idx = _idx_from_ts(ev.get('impulse_start_ts'))
        if bos_idx is None or impulse_start_idx is None:
            # Event references a candle outside the current df window. Common
            # for older events in the ring — they have no fresh OB to build.
            continue
        if impulse_start_idx >= bos_idx:
            continue

        if ev_type == 'CHoCH':
            last_choch_local_idx = bos_idx

        bos_swing_price = float(ev.get('broken_swing_price') or 0.0)

        # Median leg body (used downstream by some cosmetic features).
        leg_bodies = [abs(C[k] - O[k]) for k in range(impulse_start_idx, bos_idx + 1)]
        median_leg_body = float(np.median(leg_bodies)) if leg_bodies else 0.0001
        if median_leg_body == 0:
            median_leg_body = 0.0001

        # Walk back from break candle through impulse leg to find OB.
        # First opposing candle that passes range cap + body minimum wins.
        ob_idx = -1
        for j in range(bos_idx - 1, impulse_start_idx - 1, -1):
            if (ev_dir == 'bullish' and C[j] < O[j]) or \
               (ev_dir == 'bearish' and C[j] > O[j]):
                if h1_atr_for_leg and h1_atr_for_leg > 0:
                    if (H[j] - L[j]) > smc_detector.OB_MAX_RANGE_ATR_MULT * h1_atr_for_leg:
                        continue
                if is_valid_ob_candle(O[j], C[j], H[j], L[j]):
                    ob_idx = j
                    break
        if ob_idx == -1:
            continue

        ob_high = float(H[ob_idx])
        ob_low  = float(L[ob_idx])
        ob_proximal = ob_high if ev_dir == 'bullish' else ob_low

        # PD-array gate. Vet rule: bullish OBs are only valid when the
        # entry side (proximal) sits in the discount half of the dealing
        # range; bearish OBs only valid in the premium half. Applied to
        # all OBs (BOS and CHoCH alike) — buying in premium / selling in
        # discount is the wrong side of the range regardless of event.
        # Fails open when geometry is unavailable (no walls / cold-start
        # fallback): we cannot trust the gate, so we let the OB through.
        if walls:
            pd_info = dealing_range.compute_pd_position(ob_proximal, walls)
            if pd_info.get('valid') and not pd_info.get('fallback_active'):
                eq = pd_info['equilibrium']
                if ev_dir == 'bullish' and ob_proximal > eq:
                    continue
                if ev_dir == 'bearish' and ob_proximal < eq:
                    continue

        # Proximity gate.
        if h1_atr_for_leg and h1_atr_for_leg > 0:
            if abs(current_price_now - ob_proximal) > 4.0 * h1_atr_for_leg:
                continue

        # FVG detection — H1 internal gap, anchored to the displacement leg.
        # Window = [ob_idx, bos_idx + 1]: covers the OB candle through one
        # candle past BOS confirmation (catches late displacement). Self-
        # adjusts to leg length. Soft cap of FVG_WINDOW_H1_CANDLES guards
        # against runaway windows on slow grinds.
        bias_label    = "LONG" if ev_dir == 'bullish' else "SHORT"
        h1_atr_for_fvg = h1_atr_for_leg if h1_atr_for_leg else 0.0
        fvg_floor_mult = smc_detector.FVG_NOISE_FLOOR_MULT.get(pair_type, 0.20)
        atr_floor_h1   = fvg_floor_mult * h1_atr_for_fvg
        leg_window_end = min(
            bos_idx + 1,
            ob_idx + smc_detector.FVG_WINDOW_H1_CANDLES,
            n - 1
        )
        fvg_result = smc_detector.detect_fvg_in_zone(
            df, bias_label, ob_high, ob_low, atr_floor_h1,
            leg_start_idx=ob_idx, leg_end_idx=leg_window_end,
            pair_type=pair_type
        )

        ob_timestamp_str  = _ts_for_idx(ob_idx)
        bos_timestamp_str = _ts_for_idx(bos_idx)

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

        # Phase 1 sweep observation — leg-anchored snapshot.
        # Scans [impulse_start_idx, ob_idx] for the most-recent qualifying
        # sweep. Snapshot semantics: written once at OB build, never re-graded.
        try:
            sweep_obs = smc_detector.observe_phase1_sweep(
                df, ob_idx, impulse_start_idx, ev_dir,
                h1_atr_for_leg, pair_type, pair_name, tf_label='H1'
            )
        except Exception as _sweep_err:
            logging.warning(f"[sweep_observed] OB build failed sweep observation: {_sweep_err}")
            sweep_obs = {'exists': False}

        # 'bos_tag' is the legacy field name for the structural event type.
        # Kept for backwards compatibility with chart / scoring / dedupe code.
        # 'bos_tier' carries the Major / Minor distinction (Major for BOS).
        active_obs.append({
            'bos_idx':            bos_idx,
            'bos_timestamp':      bos_timestamp_str,
            'bos_swing_price':    bos_swing_price,
            'impulse_start_idx':  impulse_start_idx,
            'impulse_start_price': float(L[impulse_start_idx]) if ev_dir == 'bullish'
                                    else float(H[impulse_start_idx]),
            'bos_sequence_count': bos_seq_counter,
            'last_choch_idx':     last_choch_local_idx,
            'ob_idx':             ob_idx,
            'ob_timestamp':       ob_timestamp_str,
            'direction':          ev_dir,
            'bos_tag':            ev_type,
            'bos_tier':           ev_tier,
            'broken_was_wall':    bool(ev.get('broken_was_wall', False)),
            'reversal_pct':       ev.get('reversal_pct'),
            'high':               ob_high,
            'low':                ob_low,
            'proximal_line':      ob_high if ev_dir == 'bullish' else ob_low,
            'distal_line':        ob_low  if ev_dir == 'bullish' else ob_high,
            'median_leg_body':    median_leg_body,
            'ob_body':            abs(C[ob_idx] - O[ob_idx]),
            'h1_atr':             float(h1_atr_for_leg) if h1_atr_for_leg else 0.0,
            'fvg':                fvg_dict,
            'sweep_observed':     sweep_obs
        })
# Mitigation + touch tracking. Sets 'touches' and 'status' on each OB.
    # Must run BEFORE dedupe so the touch-state test in dedupe is meaningful.
    # Mitigation rule: see is_ob_mitigated_phase1 (close-based, strict, no ATR).
    tracked_obs = []
    for ob in active_obs:
        mitigated, _reason, touches = is_ob_mitigated_phase1(
            ob['direction'], ob['distal_line'], ob['proximal_line'],
            df, start_idx=ob['bos_idx'] + 1, end_idx=n
        )
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

def generate_h1_chart(df, ob, dp, pair_name, ist_timestamp, walls=None):
    """
    H1 zone chart.

    Visual elements:
      - Candles (thin body 0.55, fat wick 1.5) — last 40 candles + 6 right margin
      - Zone band (proximal/distal, purple)
      - OB candle outline
      - FVG (active or ghost)
      - BOS/CHoCH horizontal line + break candle outline
      - Current price (white)
      - DR ceiling/floor (always drawn): solid dotted if confirmed,
        dashed-gap if placeholder; faded color
      - Equilibrium line (only when both DR walls on-screen)
      - Swing markers: filled triangle for lookback-3, hollow for lookback-2,
        muted yellow, placed outside the candle
      - Adaptive figure height when DR is wide vs candle range
    """
    try:
        full_df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy().reset_index(drop=True)
        n_full  = len(full_df)
        ob_abs  = ob['ob_idx']

        # --- Plot window: 40 back from current, but always include OB candle.
        WINDOW_BACK = 40
        RIGHT_MARGIN = 6
        window_start = max(0, n_full - WINDOW_BACK)
        # If OB sits earlier than that, extend window back to include it.
        if ob_abs < window_start:
            window_start = max(0, ob_abs - 3)

        df_plot = full_df.iloc[window_start:].copy().reset_index(drop=True)
        n_plot  = len(df_plot)
        ob_plot_idx = ob_abs - window_start

        O = df_plot['Open'].values
        C = df_plot['Close'].values
        H = df_plot['High'].values
        L = df_plot['Low'].values

        proximal = float(ob['proximal_line'])
        distal   = float(ob['distal_line'])
        zone_lo  = min(proximal, distal)
        zone_hi  = max(proximal, distal)

        # --- Walls (optional) ---
        ceiling_price = None
        floor_price   = None
        ceiling_ph    = True
        floor_ph      = True
        if walls and isinstance(walls, dict):
            ceiling_price = walls.get("ceiling_price")
            floor_price   = walls.get("floor_price")
            ceiling_ph    = bool(walls.get("ceiling_is_placeholder", True))
            floor_ph      = bool(walls.get("floor_is_placeholder", True))
        eq_price = None
        if ceiling_price is not None and floor_price is not None and ceiling_price > floor_price:
            eq_price = (ceiling_price + floor_price) / 2.0

        # --- Y-limits: candles + zone + FVG + current price ALWAYS included.
        # Walls included ONLY when within a reasonable proximity to the candle
        # range; far walls (e.g. a stale 4-ATR-away ceiling) get rendered as
        # off-chart annotations so candles stay tall and readable.
        candle_lo = float(np.min(L))
        candle_hi = float(np.max(H))
        candle_span = max(candle_hi - candle_lo, 1e-9)
        h1_atr = ob.get('h1_atr', 0.0) or 0.0
        # Proximity threshold: 2.5×ATR or 1.5×candle_range, whichever is larger.
        # Wall must sit within this band of the candle range to be drawn on-chart.
        if h1_atr > 0:
            wall_proximity_max = max(2.5 * h1_atr, 1.5 * candle_span)
        else:
            wall_proximity_max = 1.5 * candle_span

        def _wall_in_view(wall_price):
            if wall_price is None:
                return False
            return (wall_price <= candle_hi + wall_proximity_max
                    and wall_price >= candle_lo - wall_proximity_max)

        ceiling_in_view = _wall_in_view(ceiling_price)
        floor_in_view   = _wall_in_view(floor_price)

        ymin_candidates = [candle_lo, zone_lo]
        ymax_candidates = [candle_hi, zone_hi]
        if ob['fvg'].get('exists') or ob['fvg'].get('was_detected'):
            ft = ob['fvg'].get('fvg_top') or ob['fvg'].get('ghost_top')
            fb = ob['fvg'].get('fvg_bottom') or ob['fvg'].get('ghost_bottom')
            if ft is not None and fb is not None:
                ymin_candidates.append(float(fb))
                ymax_candidates.append(float(ft))
        ymin_candidates.append(float(C[-1]))
        ymax_candidates.append(float(C[-1]))
        if ceiling_in_view:
            ymax_candidates.append(float(ceiling_price))
        if floor_in_view:
            ymin_candidates.append(float(floor_price))

        y_min_raw = min(ymin_candidates)
        y_max_raw = max(ymax_candidates)

        # H1 ATR-based minimum padding (0.5 ATR) so tight pairs aren't cramped.
        atr_pad = 0.5 * h1_atr if h1_atr > 0 else (y_max_raw - y_min_raw) * 0.05
        # General padding 6% of required range.
        gen_pad = (y_max_raw - y_min_raw) * 0.06
        pad = max(atr_pad, gen_pad)
        y_min = y_min_raw - pad
        y_max = y_max_raw + pad

        # --- Adaptive figure height.
        candle_range = max(candle_hi - candle_lo, 1e-9)
        required_range = y_max - y_min
        ratio = required_range / candle_range
        # Base height 6.0; bump up to 9.0 as DR/zone forces a wider y span.
        base_h = 6.0
        if ratio > 1.5:
            base_h = min(9.0, 6.0 + (ratio - 1.5) * 1.5)
        fig, ax = plt.subplots(1, 1, figsize=(14, base_h), facecolor='#131722')
        ax.set_facecolor('#131722')
        for spine in ax.spines.values():
            spine.set_color('#2a2a3e')

        # --- Candles (thin bodies, fat wicks) ---
        BODY_W = 0.55
        WICK_W = 1.5
        for i in range(n_plot):
            o, h, l, c = float(O[i]), float(H[i]), float(L[i]), float(C[i])
            col = '#26a69a' if c >= o else '#ef5350'
            ax.plot([i, i], [l, h], color=col, linewidth=WICK_W, zorder=2,
                    solid_capstyle='butt')
            body = abs(c - o) or (h - l) * 0.02
            ax.add_patch(patches.Rectangle(
                (i - BODY_W/2, min(o, c)), BODY_W, body,
                facecolor=col, linewidth=0, alpha=0.95, zorder=3
            ))

        # --- Zone band ---
        zone_x_start = max(0, ob_plot_idx - 0.5)
        zone_width   = (n_plot + RIGHT_MARGIN - 1) - zone_x_start
        ax.add_patch(patches.Rectangle(
            (zone_x_start, zone_lo), zone_width, zone_hi - zone_lo,
            facecolor='#9b59b6', alpha=0.12, zorder=1
        ))
        ax.add_patch(patches.Rectangle(
            (zone_x_start, zone_lo), zone_width, zone_hi - zone_lo,
            fill=False, edgecolor='#bb8fce', linestyle=':', linewidth=1.5, zorder=2
        ))

        # --- OB candle outline ---
        if 0 <= ob_plot_idx < n_plot:
            ob_h = float(H[ob_plot_idx])
            ob_l = float(L[ob_plot_idx])
            ax.add_patch(patches.Rectangle(
                (ob_plot_idx - 0.5, ob_l), 1.0, ob_h - ob_l,
                fill=False, edgecolor='#d7bde2', linewidth=2.0, zorder=4,
                linestyle='-'
            ))

        # --- FVG outline ---
        fvg_active  = ob['fvg']['exists'] and ob['fvg']['c1_idx'] is not None
        fvg_ghost   = (not ob['fvg']['exists']) and ob['fvg'].get('was_detected') and ob['fvg'].get('ghost_c1_idx') is not None
        fvg_partial = fvg_active and ob['fvg'].get('mitigation') == 'partial'
        if fvg_active:
            ft = float(ob['fvg']['fvg_top'])
            fb = float(ob['fvg']['fvg_bottom'])
            mid_abs = ob['fvg']['c1_idx'] + 1
            mid_local = mid_abs - window_start
            if 0 <= mid_local < n_plot:
                if fvg_partial:
                    # Amber — partial mitigation. Distinct from pristine green
                    # and ghost grey: reads as caution at a glance.
                    face_col, edge_col = '#f4d03f', '#f1c40f'
                else:
                    face_col, edge_col = '#27ae60', '#2ecc71'
                ax.add_patch(patches.Rectangle(
                    (mid_local - 0.6, fb), 3.0, ft - fb,
                    facecolor=face_col, alpha=0.25, zorder=1
                ))
                ax.add_patch(patches.Rectangle(
                    (mid_local - 0.6, fb), 3.0, ft - fb,
                    fill=False, edgecolor=edge_col, linewidth=1.0,
                    linestyle='--', zorder=2
                ))
        elif fvg_ghost:
            ft = float(ob['fvg']['ghost_top'])
            fb = float(ob['fvg']['ghost_bottom'])
            mid_abs = ob['fvg']['ghost_c1_idx'] + 1
            mid_local = mid_abs - window_start
            if 0 <= mid_local < n_plot:
                ax.add_patch(patches.Rectangle(
                    (mid_local - 0.6, fb), 3.0, ft - fb,
                    facecolor='#888888', alpha=0.10, zorder=1
                ))
                ax.add_patch(patches.Rectangle(
                    (mid_local - 0.6, fb), 3.0, ft - fb,
                    fill=False, edgecolor='#888888', linewidth=1.0,
                    linestyle=':', zorder=2
                ))

        # --- BOS / CHoCH line + break-candle outline ---
        bos_price = float(ob['bos_swing_price'])
        _btag = ob.get('bos_tag', 'BOS')
        _btier = ob.get('bos_tier', 'Major')
        if _btag == 'BOS':
            bos_color = '#00bcd4'
        elif _btier == 'Minor':
            bos_color = '#9c27b0'
        else:
            bos_color = '#ff9800'
        ax.axhline(y=bos_price, color=bos_color, linewidth=0.8,
                   linestyle='--', alpha=0.7, zorder=2)

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

        # --- Dealing Range walls + Equilibrium ---
        # Anchored = solid dotted, alpha 0.85; Placeholder = dashed-gap, alpha 0.45.
        # Walls outside the candle proximity band render as off-chart edge
        # annotations (decided earlier as ceiling_in_view / floor_in_view).
        DR_COLOR = '#5dade2'   # muted blue, doesn't clash with palette
        EQ_COLOR = '#85c1e9'   # lighter than DR walls
        DR_LW = 1.0
        if ceiling_price is not None and ceiling_in_view:
            ax.axhline(
                y=ceiling_price, color=DR_COLOR, linewidth=DR_LW,
                linestyle=':' if not ceiling_ph else (0, (4, 4)),
                alpha=0.85 if not ceiling_ph else 0.45, zorder=2
            )
        if floor_price is not None and floor_in_view:
            ax.axhline(
                y=floor_price, color=DR_COLOR, linewidth=DR_LW,
                linestyle=':' if not floor_ph else (0, (4, 4)),
                alpha=0.85 if not floor_ph else 0.45, zorder=2
            )
        # EQ only drawn when both walls are on-chart — otherwise it's
        # geometrically meaningful only with respect to the off-chart wall.
        if eq_price is not None and ceiling_in_view and floor_in_view:
            ax.axhline(
                y=eq_price, color=EQ_COLOR, linewidth=0.8,
                linestyle=':', alpha=0.6, zorder=2
            )

        # --- Swing triangles ---
        # lookback-3 filled triangles render across the visible window
        # (structural swings; drive walls / BOS / Major CHoCH).
        # lookback-2 hollow triangles render ONLY inside the OB impulse leg
        # [impulse_start_idx, ob_idx] — that's the slice where they drive
        # Phase 1 sweep observation. Outside the leg they're decision-
        # irrelevant noise.
        SWING_COLOR = '#d4a017'
        try:
            swings_lb3 = smc_detector.get_swing_points(full_df, lookback=3)
            swings_lb2 = smc_detector.get_swing_points(full_df, lookback=2)
        except Exception:
            swings_lb3, swings_lb2 = [], []
        lb3_keys = {(s['idx'], s['type']) for s in swings_lb3}
        # Visual offset based on chart vertical span (not pixels).
        marker_offset = (y_max - y_min) * 0.012
        for s in swings_lb3:
            xi = s['idx'] - window_start
            if not (0 <= xi < n_plot):
                continue
            if s['type'] == 'high':
                ax.scatter([xi], [s['price'] + marker_offset], marker='v',
                           s=42, color=SWING_COLOR, edgecolors=SWING_COLOR,
                           linewidths=1.0, zorder=6)
            else:
                ax.scatter([xi], [s['price'] - marker_offset], marker='^',
                           s=42, color=SWING_COLOR, edgecolors=SWING_COLOR,
                           linewidths=1.0, zorder=6)
        # lb2 scoped to the OB impulse leg only.
        impulse_start_abs = ob.get('impulse_start_idx')
        ob_idx_abs = ob.get('ob_idx')
        leg_valid = (
            impulse_start_abs is not None
            and ob_idx_abs is not None
            and impulse_start_abs <= ob_idx_abs
        )
        if leg_valid:
            for s in swings_lb2:
                if (s['idx'], s['type']) in lb3_keys:
                    continue  # already drawn as filled lb3
                # Scope: idx must lie inside the impulse leg.
                if not (impulse_start_abs <= s['idx'] <= ob_idx_abs):
                    continue
                xi = s['idx'] - window_start
                if not (0 <= xi < n_plot):
                    continue
                if s['type'] == 'high':
                    ax.scatter([xi], [s['price'] + marker_offset], marker='v',
                               s=36, facecolors='none', edgecolors=SWING_COLOR,
                               linewidths=1.2, zorder=6)
                else:
                    ax.scatter([xi], [s['price'] - marker_offset], marker='^',
                               s=36, facecolors='none', edgecolors=SWING_COLOR,
                               linewidths=1.2, zorder=6)

        # --- Sweep candle marker (Phase 1 sweep observation) ---
        # Highlights the swept candle's wick in a distinct color, places a
        # star at the wick tip, and draws a short level line at the swept
        # price. Sweep_idx is refreshed each scan via refresh_slate_zone, so
        # this maps to the current df frame correctly.
        sw = ob.get('sweep_observed') or {}
        if sw.get('exists'):
            sw_abs_idx = sw.get('sweep_idx')
            sw_level = sw.get('price')
            sw_tier = sw.get('tier', 'weak')
            SWEEP_COLOR_MAP = {
                'textbook': '#00e5ff',
                'decent':   '#26c6da',
                'weak':     '#80deea',
            }
            SWEEP_COLOR = SWEEP_COLOR_MAP.get(sw_tier, '#00e5ff')
            if sw_abs_idx is not None and sw_level is not None:
                sw_local = sw_abs_idx - window_start
                if 0 <= sw_local < n_plot:
                    sw_h = float(H[sw_local])
                    sw_l = float(L[sw_local])
                    # Highlight the sweep candle wick (thick line in sweep color).
                    ax.plot([sw_local, sw_local], [sw_l, sw_h],
                            color=SWEEP_COLOR, linewidth=2.8, alpha=0.85,
                            zorder=7, solid_capstyle='butt')
                    # Mark the swept wick tip with a star.
                    wick_tip = sw_l if ob['direction'] == 'bullish' else sw_h
                    ax.scatter([sw_local], [wick_tip], marker='*', s=140,
                               color=SWEEP_COLOR, edgecolors='#001f24',
                               linewidths=0.8, zorder=8)
                    # Short dashed line at the swept price level (extends
                    # back from the sweep candle to show the swept swing).
                    lvl_x_start = max(0, sw_local - 6)
                    ax.plot([lvl_x_start, sw_local], [sw_level, sw_level],
                            color=SWEEP_COLOR, linewidth=1.0,
                            linestyle=(0, (3, 2)), alpha=0.7, zorder=4)
                    # Compact label near the wick tip.
                    label_dy = -14 if ob['direction'] == 'bullish' else 14
                    label_va = 'top' if ob['direction'] == 'bullish' else 'bottom'
                    ax.annotate('Sweep', xy=(sw_local, wick_tip),
                                xytext=(0, label_dy), textcoords='offset points',
                                color=SWEEP_COLOR, fontsize=8, fontweight='bold',
                                ha='center', va=label_va, zorder=8)

        # --- Current price line ---
        current_price = float(C[-1])
        ax.axhline(y=current_price, color='#ffffff', linewidth=0.8,
                   linestyle='-', alpha=0.5, zorder=2)

        # --- Mid-chart numeric tags (price units; chart axis stays in price) ---
        pair_type_guess = "forex"
        if dp == 3:
            pair_type_guess = "forex"
        elif dp == 0 or dp == 1:
            pair_type_guess = "index"
        elif dp == 2:
            pair_type_guess = "commodity"
        pair_conf_shim = {"decimal_places": dp, "pair_type": pair_type_guess}

        mid_x = n_plot / 2.0
        mid_labels = [
            (proximal, f"{proximal:.{dp}f}", '#bb8fce'),
            (distal,   f"{distal:.{dp}f}",   '#bb8fce'),
            (bos_price, f"{bos_price:.{dp}f}", bos_color),
            (current_price, f"{current_price:.{dp}f}", '#ffffff'),
        ]
        # Wall labels go in mid-chart ONLY when the wall is on-chart. Off-
        # chart walls get edge annotations below.
        if ceiling_price is not None and ceiling_in_view:
            anchor_tag = '(placeholder)' if ceiling_ph else '(anchored)'
            mid_labels.append(
                (ceiling_price,
                 f"DR↑ {ceiling_price:.{dp}f} {anchor_tag}",
                 DR_COLOR)
            )
        if floor_price is not None and floor_in_view:
            anchor_tag = '(placeholder)' if floor_ph else '(anchored)'
            mid_labels.append(
                (floor_price,
                 f"DR↓ {floor_price:.{dp}f} {anchor_tag}",
                 DR_COLOR)
            )
        # EQ label only when EQ line is drawn (both walls on-chart).
        if eq_price is not None and ceiling_in_view and floor_in_view:
            mid_labels.append((eq_price, f"EQ {eq_price:.{dp}f}", EQ_COLOR))

        mid_stacked = smc_detector.stack_labels(mid_labels, pair_conf_shim)
        for adj_price, text, color in mid_stacked:
            ax.text(mid_x, adj_price, text, color=color, fontsize=10, va='center',
                    ha='center', fontweight='bold', zorder=7,
                    bbox=dict(facecolor='#131722', edgecolor='none', pad=1.5, alpha=0.78))

        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-1, n_plot + RIGHT_MARGIN)

        # --- Off-chart wall annotations ---
        # Far walls (not in candle proximity band) render as compact edge
        # labels so the trader still knows where the structural reference
        # sits, without sacrificing candle height.
        if dp == 5:
            pip_unit_local = 0.0001
        elif dp == 3:
            pip_unit_local = 0.01
        else:
            pip_unit_local = 1.0

        def _off_chart_label(direction, wall_price, is_placeholder):
            anchor_tag = 'placeholder' if is_placeholder else 'anchored'
            if direction == 'up':
                dist = (wall_price - candle_hi) / pip_unit_local
                arrow = '↑'
                ref = 'above'
            else:
                dist = (candle_lo - wall_price) / pip_unit_local
                arrow = '↓'
                ref = 'below'
            dist_str = f"{abs(dist):.0f}p {ref}"
            return f"DR{arrow} {wall_price:.{dp}f} ({anchor_tag}, {dist_str})"

        if ceiling_price is not None and not ceiling_in_view:
            txt = _off_chart_label('up', ceiling_price, ceiling_ph)
            ax.text(
                n_plot / 2.0, y_max, txt,
                color=DR_COLOR, fontsize=9, fontweight='bold',
                ha='center', va='top', zorder=7,
                bbox=dict(facecolor='#131722', edgecolor=DR_COLOR,
                          linewidth=0.7, pad=2.5, alpha=0.92)
            )
        if floor_price is not None and not floor_in_view:
            txt = _off_chart_label('down', floor_price, floor_ph)
            ax.text(
                n_plot / 2.0, y_min, txt,
                color=DR_COLOR, fontsize=9, fontweight='bold',
                ha='center', va='bottom', zorder=7,
                bbox=dict(facecolor='#131722', edgecolor=DR_COLOR,
                          linewidth=0.7, pad=2.5, alpha=0.92)
            )

        direction_label = "Demand" if ob['direction'] == 'bullish' else "Supply"
        event_label = _event_label(ob.get('bos_tag', 'BOS'), ob.get('bos_tier', 'Major'))
        title = (
            f"{pair_name} | {direction_label} Zone | {event_label} | "
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

    event_label = _event_label(ob.get('bos_tag', 'BOS'), ob.get('bos_tier', 'Major'))
    prompt = f"""You are a veteran SMC (Smart Money Concepts) prop trader writing a zone briefing for another experienced SMC trader.
Be direct. No fluff. No pleasantries. Four sentences only. One paragraph.

ZONE DATA — use these exact values, do not recalculate:
- Pair: {name}
- Bias: {direction} | Structure event: {event_label}
- Proximal: {proximal:.{dp}f} | Distal: {distal:.{dp}f}
- Zone width: {zone_pips} pips
- OB body vs median impulse leg: {ob['ob_body']:.{dp}f} vs {ob['median_leg_body']:.{dp}f} (ratio: {ratio}x — valid because <2.0x)
- FVG: {fvg_status}
- Zone status: {ob['status']}
- Current price: {current_price:.{dp}f} | Distance to proximal: {dist_pips} pips

WRITE EXACTLY FOUR SENTENCES IN THIS ORDER:
1. What structure event ({event_label}) created this zone and why institutional accumulation is likely here.
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
    event_label = _event_label(ob.get('bos_tag', 'BOS'), ob.get('bos_tier', 'Major'))
    return (
        f"{event_label} confirmed the {ob['direction']} shift; this OB marks the last institutional "
        f"accumulation before the break. "
        f"OB body ratio {round(ob['ob_body']/ob['median_leg_body'],2):.2f}x vs median leg — tight and valid. "
        f"{fvg_line} "
        f"Current price is {dist_pips} pips from proximal — "
        f"{'approaching zone, watch for reaction.' if dist_pips < 50 else 'still distant, no action yet.'}"
    )

# ---------------------------------------------------------------------------
# EMAIL ASSEMBLY
# ---------------------------------------------------------------------------

def _format_dr_walls_cell(walls):
    """Render the DR Walls cell. Per-wall anchored/placeholder status."""
    if not walls or not isinstance(walls, dict):
        return "<span style='color:#888;'>&mdash;</span>"
    if walls.get("fallback_active"):
        return "<span style='color:#e74c3c;'>&#9888; Fallback</span>"
    ceiling_ph = walls.get("ceiling_is_placeholder", True)
    floor_ph   = walls.get("floor_is_placeholder", True)
    if not ceiling_ph and not floor_ph:
        return "<span style='color:#27ae60;'>&#10003; Both anchored</span>"
    if ceiling_ph and floor_ph:
        return "<span style='color:#e67e22;'>&#9888; Both placeholder</span>"
    if ceiling_ph:
        # Ceiling tentative, floor anchored.
        return (
            "<span style='color:#e67e22;'>&#9888; Ceiling placeholder</span>"
            "<span style='color:#666;font-size:10px;'>&nbsp;/ floor anchored</span>"
        )
    return (
        "<span style='color:#e67e22;'>&#9888; Floor placeholder</span>"
        "<span style='color:#666;font-size:10px;'>&nbsp;/ ceiling anchored</span>"
    )


def _pip_unit(dp):
    return 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)


def build_summary_table_html(all_zones_for_table, dp_map, pair_prices=None):
    pair_prices = pair_prices or {}
    rows = ""
    for z in all_zones_for_table:
        name      = z['name']
        dp        = dp_map[name]
        walls     = z.get('walls', {})
        dr_cell   = _format_dr_walls_cell(walls)

        # Placeholder row: pair has no active zone (or data unavailable).
        # Columns: Pair | Bias | Status | Dist | DR Walls | FVG | First Seen
        if z.get('is_placeholder_row'):
            rows += f"""
        <tr style="background:transparent;border-bottom:1px solid #2a2a3e;opacity:0.55;">
          <td style="padding:6px 8px;font-weight:bold;color:#aaa;font-size:12px;white-space:nowrap;">
            <span style='color:#555;font-size:10px;font-family:monospace;'>&mdash;&nbsp;</span>{name}
          </td>
          <td style="padding:6px 8px;color:#888;font-size:11px;white-space:nowrap;">&mdash;</td>
          <td style="padding:6px 8px;color:#888;font-size:11px;white-space:nowrap;font-style:italic;">{z['status']}</td>
          <td style="padding:6px 8px;color:#888;font-size:11px;text-align:right;">&mdash;</td>
          <td style="padding:6px 8px;font-size:11px;white-space:nowrap;">{dr_cell}</td>
          <td style="padding:6px 8px;color:#888;font-size:12px;text-align:center;">&mdash;</td>
          <td style="padding:6px 8px;color:#666;font-size:10px;white-space:nowrap;">&mdash;</td>
        </tr>"""
            continue

        direction = "&#9650; Bullish" if z['direction'] == 'bullish' else "&#9660; Bearish"
        dir_color = '#27ae60'   if z['direction'] == 'bullish' else '#e74c3c'
        status    = z['status']
        stat_col  = '#27ae60'   if 'Pristine' in status else '#e67e22'

        # Distance to proximal in pips. If price is inside zone, show "in zone".
        cur_price = pair_prices.get(name)
        if cur_price is not None and z.get('proximal') is not None:
            pip_unit = _pip_unit(dp)
            if z.get('in_progress'):
                dist_cell = "<span style='color:#f1c40f;'>in zone</span>"
            else:
                dpips = round(abs(cur_price - z['proximal']) / pip_unit, 1)
                dist_cell = f"{dpips} p"
        else:
            dist_cell = "&mdash;"

        if z['fvg_valid'] and z.get('fvg_mitigation') == 'partial':
            fvg_cell = "&#9680;"
            fvg_col  = '#f1c40f'   # amber — partial (caution)
        elif z['fvg_valid']:
            fvg_cell = "&#10003;"
            fvg_col  = '#27ae60'   # green — pristine
        elif z.get('fvg_ghost'):
            fvg_cell = "&#9675;"
            fvg_col  = '#888888'   # grey — ghost (mitigated)
        else:
            fvg_cell = "&ndash;"
            fvg_col  = '#888'

        _tier = z.get('bos_tier', 'Major')
        if z['bos_tag'] == 'BOS':
            _badge_color, _badge_text = '#00bcd4', 'BOS'
        elif _tier == 'Minor':
            _badge_color, _badge_text = '#9c27b0', 'CHoCH'
        else:
            _badge_color, _badge_text = '#ff9800', 'CHoCH'
        tag_badge = (
            f"<span style='background:{_badge_color};color:#000;font-size:9px;"
            f"padding:1px 4px;border-radius:3px;font-weight:bold;'>{_badge_text}</span>"
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
          <td style="padding:6px 8px;color:#ddd;font-size:11px;text-align:right;white-space:nowrap;">{dist_cell}</td>
          <td style="padding:6px 8px;font-size:11px;white-space:nowrap;">{dr_cell}</td>
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
            <th style="padding:7px 8px;text-align:right;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">Dist</th>
            <th style="padding:7px 8px;text-align:left;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">DR Walls</th>
            <th style="padding:7px 8px;text-align:center;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">FVG</th>
            <th style="padding:7px 8px;text-align:left;color:#666;font-size:10px;font-weight:normal;text-transform:uppercase;letter-spacing:0.5px;">First Seen</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

def _phase1_chart_legend_html(bos_tag="BOS", bos_tier="Major"):
    """Colour-code legend rendered below H1 zone chart in Phase 1 digest. Cosmetic only."""
    if bos_tag == 'BOS':
        bos_color, bos_label = '#00bcd4', 'BOS'
    elif bos_tier == 'Minor':
        bos_color, bos_label = '#9c27b0', 'Minor CHoCH'
    else:
        bos_color, bos_label = '#ff9800', 'Major CHoCH'
    items = [
        ('#bb8fce', 'Zone band (proximal/distal)'),
        ('#d7bde2', 'OB candle outline'),
        ('#2ecc71', 'FVG pristine (displacement)'),
        ('#f1c40f', 'FVG partial (proximal touched)'),
        ('#888888', 'FVG mitigated (ghost)'),
        (bos_color, f'{bos_label} break candle / level'),
        ('#5dade2', 'Dealing range walls (dotted=anchored, dashed=placeholder; far walls render as edge labels)'),
        ('#85c1e9', 'Equilibrium (50%)'),
        ('#d4a017', 'Swing: ▲▼ filled = lookback-3 (visible window); △▽ hollow = lookback-2 (OB impulse leg only)'),
        ('#00e5ff', 'Sweep candle (★ at wick tip, line at swept level) — Phase 1 sweep observation'),
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


def _render_sweep_observation_html(sweep_obs, dp):
    """
    Render the Phase 1 sweep observation badge HTML. Single source of truth
    used by both NEW-zone and active-zone card builders so the format stays
    consistent. Snapshot semantics: timestamp / numbers / tags are frozen at
    OB build time.

    Returns the HTML string starting with "Sweep:".
    """
    sweep_obs = sweep_obs or {}
    if not sweep_obs.get('exists'):
        return "Sweep: <span style='color:#888;'>None observed in this leg</span>"

    tier = sweep_obs.get('tier', 'weak')
    tier_color = {'textbook': '#27ae60', 'decent': '#e67e22', 'weak': '#888'}.get(tier, '#888')
    tier_emoji = {'textbook': '🎯', 'decent': '◐', 'weak': '·'}.get(tier, '·')
    tier_explainer = {
        'textbook': 'Strong rejection wick with multiple prior swings at the same level.',
        'decent':   'Either strong rejection OR equal-level cluster — one of the two signals.',
        'weak':     'Sweep present but minimal rejection and no equal-level cluster.'
    }.get(tier, '')

    ts_raw = sweep_obs.get('timestamp', '')
    ts_short = ''
    try:
        if ts_raw:
            ts_short = datetime.fromisoformat(ts_raw.replace('Z', '')).strftime('%H:%M UTC on %d-%b')
    except Exception:
        ts_short = ts_raw[:16] if ts_raw else ''

    tag_pretty = {
        'round_number':    'round number',
        'prior_day_high':  'prior day high',
        'prior_day_low':   'prior day low',
        'asia_high':       'Asia high',
        'asia_low':        'Asia low',
        'london_high':     'London high',
        'london_low':      'London low',
        'ny_high':         'NY high',
        'ny_low':          'NY low'
    }
    tags = sweep_obs.get('context_tags', []) or []
    tags_text = ', '.join(tag_pretty.get(t, t) for t in tags) if tags else 'none'
    wick_pips = sweep_obs.get('wick_distance_pips', 0)
    wick_body = sweep_obs.get('wick_body_ratio', 0)
    eq_count  = sweep_obs.get('equal_levels_count', 0)

    return (
        f"Sweep: <span style='color:{tier_color};'>"
        f"{tier_emoji} {tier.title()} ({sweep_obs.get('tf', 'H1')} "
        f"@ {sweep_obs.get('price', 0):.{dp}f} at {ts_short})</span>"
        f"<div style='font-size:10px;color:#aaa;margin-top:3px;line-height:1.5;'>"
        f"Wick {wick_pips} past level &middot; wick:body {wick_body}x &middot; "
        f"equal levels {eq_count}<br>"
        f"Context: {tags_text}<br>"
        f"<span style='color:#888;font-style:italic;'>{tier_explainer}</span>"
        f"</div>"
    )


def build_new_zone_card_html(ob, name, dp, narrative, cid, ist_timestamp, zone_id="—"):
    direction  = "Bullish (Demand)" if ob['direction'] == 'bullish' else "Bearish (Supply)"
    dir_color  = '#27ae60' if ob['direction'] == 'bullish' else '#e74c3c'
    stat_color = '#27ae60' if 'Pristine' in ob['status'] else '#e67e22'
    mit = ob['fvg'].get('mitigation', 'none')
    if ob['fvg'].get('exists') and mit == 'partial':
        # Partial — price crossed proximal but not distal. Amber (caution).
        fvg_line = (
            f"FVG: <span style='color:#f1c40f;'>◐ Partial "
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

    # Sweep observation badge — Phase 1 snapshot. Same render path used by
    # build_active_zone_card_html for consistency across NEW and active cards.
    sweep_line = _render_sweep_observation_html(ob.get('sweep_observed'), dp)

    pip_unit  = 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)
    zone_pips = round(abs(ob['proximal_line'] - ob['distal_line']) / pip_unit, 1)

    chart_html = (
        f'<img src="cid:{cid}" style="width:100%;max-width:600px;border-radius:6px;'
        f'border:1px solid #2a2a3e;display:block;" />'
        if cid else
        '<div style="padding:8px 12px;background:#2d1a1a;border-left:3px solid #e74c3c;border-radius:4px;color:#e74c3c;font-size:11px;">&#9888; Chart failed to render.</div>'
    )
    legend_html = _phase1_chart_legend_html(ob.get('bos_tag', 'BOS'),
                                              ob.get('bos_tier', 'Major'))

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

def build_active_zone_card_html(sz, name, dp, narrative, cid, ist_timestamp,
                                 current_price=None, in_progress=False):
    """
    Render an active zone card. Used for both NEW and UNCHANGED active zones.
    NEW badge is rendered inline based on sz['is_new_this_scan'].
    Distance to proximal shown in pips. 'In zone' label when in_progress.
    """
    direction  = "Bullish (Demand)" if sz['direction'] == 'bullish' else "Bearish (Supply)"
    dir_color  = '#27ae60' if sz['direction'] == 'bullish' else '#e74c3c'
    status_label = sz.get('status_label', 'Pristine')
    stat_color = '#27ae60' if 'Pristine' in status_label else '#e67e22'

    fvg = sz.get('fvg', {})
    mit = fvg.get('mitigation', 'none')
    if fvg.get('exists') and mit == 'partial':
        fvg_line = (
            f"FVG: <span style='color:#f1c40f;'>◐ Partial "
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

    pip_unit  = _pip_unit(dp)
    zone_pips = round(abs(sz['proximal_line'] - sz['distal_line']) / pip_unit, 1)
    h1_atr_val = sz.get('h1_atr', 0.0)
    # H1 ATR rendered in pips (was raw price units — confusing across pairs).
    atr_pips_display = (
        f"{round(h1_atr_val / pip_unit, 1)} p" if h1_atr_val > 0 else "—"
    )
    if current_price is not None:
        if in_progress:
            dist_text = "<span style='color:#f1c40f;font-weight:bold;'>in zone</span>"
        else:
            dist_pips_val = round(abs(current_price - sz['proximal_line']) / pip_unit, 1)
            dist_text = f"{dist_pips_val} pips"
    else:
        dist_text = "—"

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
    legend_html = _phase1_chart_legend_html(sz.get('bos_tag', 'BOS'),
                                              sz.get('bos_tier', 'Major'))

    # Phase 1 sweep snapshot — same renderer as NEW card.
    sweep_line = _render_sweep_observation_html(sz.get('sweep_observed'), dp)

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
          <b style="color:#bb8fce;">Dist</b> {dist_text}
        </span>
        <span style="font-size:11px;color:#aaa;">
          <b style="color:#bb8fce;">H1 ATR</b> {atr_pips_display}
        </span>
        <span style="font-size:11px;color:#aaa;">{fvg_line}</span>
      </div>
      <div style="margin-bottom:10px;font-size:11px;color:#aaa;">{sweep_line}</div>
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
        "mitigated_distal_break": "invalidated — price touched distal",
        "mitigated_three_touches": "mitigated — proximal hit 3 times",
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
    pip_unit     = _pip_unit(dp)
    zone_pips    = round(abs(proximal - distal) / pip_unit, 1)
    dist_pips    = round(abs(current_price - proximal) / pip_unit, 1)
    atr_pips     = round(h1_atr / pip_unit, 1) if h1_atr > 0 else None
    atr_display  = f"{atr_pips} pips" if atr_pips is not None else "n/a"

    z_lo = min(proximal, distal)
    z_hi = max(proximal, distal)
    in_zone = z_lo <= current_price <= z_hi

    if ob['fvg'].get('exists'):
        mit = ob['fvg'].get('mitigation', 'pristine')
        fvg_top_pips = round(ob['fvg']['fvg_top'] / pip_unit, 1)
        fvg_bot_pips = round(ob['fvg']['fvg_bottom'] / pip_unit, 1)
        if mit == 'partial':
            fvg_status = (
                f"partially mitigated FVG ({fvg_bot_pips}–{fvg_top_pips} pips zone) "
                f"— price tagged proximal, distal intact"
            )
        else:
            fvg_status = (
                f"pristine FVG ({fvg_bot_pips}–{fvg_top_pips} pips zone)"
            )
    elif ob['fvg'].get('was_detected'):
        fvg_status = "FVG fully mitigated — zone relies on OB alone"
    else:
        fvg_status = "no FVG present"
    ratio = round(ob['ob_body'] / ob['median_leg_body'], 2) if ob['median_leg_body'] > 0 else 0

    event_label = _event_label(ob.get('bos_tag', 'BOS'), ob.get('bos_tier', 'Major'))
    distance_brief = "price is INSIDE the zone (mitigation in progress)" if in_zone \
                     else f"price is {dist_pips} pips from proximal"
    prompt = f"""You are a veteran SMC (Smart Money Concepts) prop trader writing a zone briefing for another experienced SMC trader.
Be direct. No fluff. No pleasantries. Four sentences only. One paragraph.

ZONE DATA — use these exact values, do not recalculate. ALL distances in PIPS.
- Pair: {name}
- Bias: {direction} | Structure event: {event_label}
- Zone width: {zone_pips} pips
- OB body vs median impulse leg ratio: {ratio}x
- FVG: {fvg_status}
- Zone status: {ob.get('status', 'Pristine')}
- H1 ATR: {atr_display}
- {distance_brief}

WRITE EXACTLY FOUR SENTENCES IN THIS ORDER:
1. What structure event ({event_label}) created this zone and why institutional accumulation is likely here.
2. OB quality: assess tightness (ratio {ratio}x), and whether pristine or tested means strength or caution.
3. FVG assessment: displacement confirmation present or absent, and what that means for zone conviction.
4. Current price context: state distance in pips, compare to H1 ATR ({atr_display}), and what to watch for. Use only pips for distance, never raw price.

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
    pip_unit  = _pip_unit(dp)
    dist_pips = round(abs(current_price - ob['proximal_line']) / pip_unit, 1)
    atr_pips  = round(h1_atr / pip_unit, 1) if h1_atr > 0 else None
    atr_display = f"{atr_pips} pips" if atr_pips is not None else "n/a"
    z_lo = min(ob['proximal_line'], ob['distal_line'])
    z_hi = max(ob['proximal_line'], ob['distal_line'])
    in_zone = z_lo <= current_price <= z_hi

    if ob['fvg'].get('exists'):
        mit = ob['fvg'].get('mitigation', 'pristine')
        fvg_w_pips = round(abs(ob['fvg']['fvg_top'] - ob['fvg']['fvg_bottom']) / pip_unit, 1)
        if mit == 'partial':
            fvg_line = (
                f"FVG partially mitigated ({fvg_w_pips} pips wide) "
                f"— proximal tagged, distal still intact."
            )
        else:
            fvg_line = (
                f"FVG confirmed ({fvg_w_pips} pips wide), adding displacement confluence."
            )
    elif ob['fvg'].get('was_detected'):
        fvg_line = "FVG fully mitigated — zone relies on OB alone for confluence."
    else:
        fvg_line = "No FVG present — zone relies on OB alone for confluence."
    event_label = _event_label(ob.get('bos_tag', 'BOS'), ob.get('bos_tier', 'Major'))
    if in_zone:
        distance_line = f"Price is INSIDE the zone — mitigation in progress, watch for reaction (H1 ATR {atr_display})."
    elif dist_pips < 50:
        distance_line = f"Current price is {dist_pips} pips from proximal (H1 ATR {atr_display}) — approaching zone, watch for reaction."
    else:
        distance_line = f"Current price is {dist_pips} pips from proximal (H1 ATR {atr_display}) — still distant, no immediate action."
    return (
        f"{event_label} confirmed the {ob['direction']} shift; this OB marks the last institutional "
        f"accumulation before the break. "
        f"OB body ratio {round(ob['ob_body']/ob['median_leg_body'],2):.2f}x vs median leg. "
        f"{fvg_line} "
        f"{distance_line}"
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
#           "drop_reason": null | "mitigated_distal_break" | "mitigated_three_touches" | "structure_supplanted" | "aged_out_of_window" | "data_unavailable" | "data_stale",
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
        "bos_tier": fresh_zone.get("bos_tier", "Major"),
        "broken_was_wall": fresh_zone.get("broken_was_wall", False),
        "reversal_pct": fresh_zone.get("reversal_pct"),
        "bos_timestamp": fresh_zone.get("bos_timestamp"),
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
        "fvg": fresh_zone["fvg"],
        # Phase 1 sweep snapshot — preserved across scans; never re-evaluated.
        "sweep_observed": fresh_zone.get("sweep_observed", {"exists": False})
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
    # All idx-bearing fields MUST be refreshed together — they belong to the
    # same df frame, which rolls each scan as yfinance returns a new window.
    # Bug history: previously only bos_idx + ob_idx were refreshed, leaving
    # impulse_start_idx stale (pointing past ob_idx after enough scans). This
    # silently disabled sweep observation since the leg slice became empty.
    slate_zone["bos_idx"]              = fresh_zone["bos_idx"]
    slate_zone["ob_idx"]               = fresh_zone["ob_idx"]
    slate_zone["impulse_start_idx"]    = fresh_zone["impulse_start_idx"]
    slate_zone["impulse_start_price"]  = fresh_zone["impulse_start_price"]
    slate_zone["bos_swing_price"]      = fresh_zone["bos_swing_price"]
    slate_zone["touches"]       = fresh_zone.get("touches", 0)
    slate_zone["status_label"]  = fresh_zone.get("status", "Pristine")
    slate_zone["h1_atr"]        = fresh_zone.get("h1_atr", 0.0)
    slate_zone["current_price_at_scan"] = current_price
    slate_zone["distance_to_proximal_pips"] = dist_pips
    slate_zone["fvg"]           = fresh_zone["fvg"]
    # Sweep observation: refresh with the fresh observation. Original design
    # was "snapshot, never re-evaluate", but that assumed df-frame stability
    # which doesn't hold. Sweep observation is deterministic given the leg,
    # so re-observing each scan yields the same answer in stable conditions
    # and self-corrects if yfinance revises wicks.
    slate_zone["sweep_observed"] = fresh_zone.get(
        "sweep_observed", slate_zone.get("sweep_observed", {"exists": False})
    )
    # Tier / context refresh (Major / Minor distinction may change if a
    # later re-emission upgrades the structural classification).
    slate_zone["bos_tier"]      = fresh_zone.get("bos_tier", slate_zone.get("bos_tier", "Major"))
    slate_zone["broken_was_wall"] = fresh_zone.get("broken_was_wall",
                                                    slate_zone.get("broken_was_wall", False))
    slate_zone["reversal_pct"]  = fresh_zone.get("reversal_pct", slate_zone.get("reversal_pct"))
    slate_zone["bos_timestamp"] = fresh_zone.get("bos_timestamp", slate_zone.get("bos_timestamp"))


def determine_drop_reason(slate_zone, current_price, df, h1_atr, fresh_zones_in_pair, pair_type):
    """
    Return concrete drop reason. Every drop must map to ONE of these checks.
    No "unknown" or "structure_invalidated" fallback — silent fails are the
    failure mode we are explicitly preventing.

    Returns one of:
      'mitigated_distal_break'
      'mitigated_three_touches'
      'structure_supplanted'
      'aged_out_of_window'
      'data_unavailable'
      'data_stale'
      None  -> zone should NOT be dropped; caller keeps it alive and logs.

    Note: proximity gating was removed. Phase 1 surfaces all 6 pairs every run;
    only invalidation drops a zone.
    """
    # --- data_unavailable handled by caller before this is called ---

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
    # Uses is_ob_mitigated_phase1 — single source of truth for Phase 1.
    # Rule: close beyond distal (strict, no ATR buffer). Wick alone never
    # invalidates. 3 wick touches at proximal = mitigated.
    if df is not None and len(df) > 0:
        try:
            distal = slate_zone['distal_line']
            proximal = slate_zone['proximal_line']
            direction = slate_zone['direction']

            # Find scan start index — from the candle AFTER OB if locatable,
            # else whole df.
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

            mitigated, reason, _touches = is_ob_mitigated_phase1(
                direction, distal, proximal, df,
                start_idx=scan_start, end_idx=len(df)
            )
            if mitigated and reason:
                return reason
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


def select_relevant_zone_for_pair(active_zones, current_price, dp):
    """
    Phase 1 highlights ONE zone per pair, recomputed every scan.

    Selection rule (locked):
      1. If price is INSIDE any active zone (between proximal and distal,
         inclusive), that zone wins. Tag it 'zone-in-progress'. If multiple
         zones contain price (rare; nested zones), pick the one whose proximal
         is closest to price.
      2. Otherwise: pick the zone with smallest |current_price - proximal|.

    No bias filter (counter-trend OBs can flag the next CHoCH).
    No pristine preference (closeness wins).
    No hysteresis (closest each scan; flapping = real signal that price sits
    between zones).

    Returns (selected_zone, in_progress_flag) or (None, False) if no zones.
    """
    if not active_zones:
        return None, False

    # In-zone check first.
    inside = []
    for z in active_zones:
        prox = z['proximal_line']
        dist = z['distal_line']
        lo, hi = (dist, prox) if z['direction'] == 'bullish' else (prox, dist)
        # Bullish demand: distal below proximal, both below price normally.
        # Bearish supply: proximal below distal, both above price normally.
        # Generic containment by min/max handles both.
        z_lo = min(prox, dist)
        z_hi = max(prox, dist)
        if z_lo <= current_price <= z_hi:
            inside.append(z)

    if inside:
        winner = min(inside, key=lambda z: abs(current_price - z['proximal_line']))
        return winner, True

    # Otherwise closest by pip distance to proximal.
    winner = min(active_zones, key=lambda z: abs(current_price - z['proximal_line']))
    return winner, False


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

    # One-time backfill of the Phase 1 scan log from structure_state events.
    # Idempotent: skips if the current month's log file already exists.
    _backfill_phase1_scan_log()

    pair_names = [p['name'] for p in config_master['pairs']]
    # Per-pair scan records accumulated across the scan; written once at end.
    phase1_scan_records = []

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
            # Scan log still records the gap.
            try:
                _walls_for_log = dealing_range.load_state().get(name, {})
            except Exception:
                _walls_for_log = {}
            phase1_scan_records.append(_build_phase1_scan_record(
                name, ist_now, None, _walls_for_log,
                slate_pair.get("zones", []), [], [],
                diagnostics=['data_unavailable: fetch_data returned None']
            ))
            continue

        pairs_with_fresh_data.add(name)
        pair_dfs[name] = df

        # --- DEALING RANGE WALL UPDATE (single source of truth) ---
        # Load prior state for this pair, walk forward through fresh H1 data,
        # update walls if any breaks fired, save back. Phase 2 reads only.
        new_walls = {}
        placeholder_diag = None
        try:
            structure_state_all = dealing_range.load_state()
            prior_walls = structure_state_all.get(name)
            new_walls = dealing_range.update_pair(df, prior_walls, pair) or {}
            # Strip non-persisted diagnostic before saving — it stays in
            # the in-memory copy for downstream scan logging only.
            placeholder_diag = new_walls.pop(dealing_range.PLACEHOLDER_DIAG_KEY, None)
            structure_state_all[name] = new_walls
            dealing_range.save_state(structure_state_all)
            if new_walls.get("fallback_active"):
                logging.warning(
                    f"[{name}] Dealing-range fallback active — no BOS/CHoCH in cold-start window."
                )
                print(f"  [WALLS] {name}: fallback active (last 72h high/low used).")
            elif new_walls.get("last_event_chop"):
                print(f"  [WALLS] {name}: rapid CHoCH within 5 candles — possible chop.")
        except Exception as _dr_err:
            logging.error(f"[{name}] dealing_range update failed: {_dr_err}")
            print(f"  [WALLS ERR] {name}: {_dr_err}")

        result        = detect_smc_radar(df, pair_type=ptype,
                                          events=new_walls.get('events', []),
                                          walls=new_walls, pair_name=name)
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
        drops_this_pair = []
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
            drops_this_pair.append({"id": sz["zone_id"], "reason": reason})
            print(f"  [DROP] {name} {sz['zone_id']} dropped — {reason}.")

        # Phase 1 scan log — one record per pair per scan (forensic trail).
        phase1_scan_records.append(_build_phase1_scan_record(
            name, ist_now, current_price, new_walls,
            slate_zones, fresh_zones, drops_this_pair,
            placeholder_diagnostic=placeholder_diag
        ))

        # Audit log row per active zone post-reconcile.
        # Shadow-logs dealing range source + PD position for retrospective analysis.
        try:
            _walls_for_audit = dealing_range.load_state().get(name, {})
            _pd_audit = dealing_range.compute_pd_position(current_price, _walls_for_audit)
        except Exception:
            _walls_for_audit = {}
            _pd_audit = {"valid": False, "source": "error", "pd_position": None}
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
                    "is_new_this_scan": sz.get("is_new_this_scan", False),
                    "dr_source": _pd_audit.get("source"),
                    "dr_valid":  _pd_audit.get("valid"),
                    "dr_pd_position": _pd_audit.get("pd_position"),
                    "dr_fallback": _pd_audit.get("fallback_active", False),
                    "dr_tentative": _pd_audit.get("tentative", False),
                    "dr_event_type":  _walls_for_audit.get("last_event_type"),
                    "dr_event_tier":  _walls_for_audit.get("last_event_tier"),
                    "dr_event_dir":   _walls_for_audit.get("last_event_direction"),
                    "dr_chop_flag":   _walls_for_audit.get("last_event_chop", False)
                })

    # --- BUILD EMAIL ---
    if send_email_this_run:
        # Structure status banner: tells the trader at a glance which pairs
        # have a fully-anchored dealing range vs. tentative side vs. fallback.
        try:
            _structure_state_for_banner = dealing_range.load_state()
        except Exception:
            _structure_state_for_banner = {}
        _fb_pairs = []
        _tent_pairs = []
        _ok_pairs = []
        _chop_pairs = []
        for _pn in pair_names:
            _ws = _structure_state_for_banner.get(_pn, {})
            if _ws.get("fallback_active"):
                _fb_pairs.append(_pn)
            elif _ws.get("ceiling_is_placeholder") or _ws.get("floor_is_placeholder"):
                _tent_pairs.append(_pn)
            elif _ws.get("ceiling_price") is not None and _ws.get("floor_price") is not None:
                _ok_pairs.append(_pn)
            if _ws.get("last_event_chop"):
                _chop_pairs.append(_pn)
        _structure_banner_lines = [
            f"Structure: {len(_ok_pairs)}/{len(pair_names)} pairs fully anchored."
        ]
        if _tent_pairs:
            _structure_banner_lines.append(
                f"  Tentative side (one wall pending swing confirmation): {', '.join(_tent_pairs)}"
            )
        if _fb_pairs:
            _structure_banner_lines.append(
                f"  FALLBACK (no recent BOS/CHoCH — using last 72h high/low): {', '.join(_fb_pairs)}"
            )
        if _chop_pairs:
            _structure_banner_lines.append(
                f"  \u26a0\ufe0f Rapid CHoCH (possible chop, low conviction): {', '.join(_chop_pairs)}"
            )
        structure_banner_text = "\n".join(_structure_banner_lines)
        print(structure_banner_text)
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
            zones_in_pair = pblock.get("zones", [])

            # Dropped lines first (independent of selection).
            for sz in zones_in_pair:
                if sz["status"] == "dropped":
                    dropped_lines.append(build_dropped_zone_line(sz, pair_name, dp))

            active_zones = [z for z in zones_in_pair if z["status"] == "active"]
            current_price = pair_prices.get(pair_name)
            df = pair_dfs.get(pair_name)

            # Pair-level walls for table + chart context.
            # Reuse the structure state already loaded for the banner above —
            # avoids N+1 disk reads.
            pair_walls = _structure_state_for_banner.get(pair_name, {}) or {}

            # No active zones for this pair — render placeholder row.
            if not active_zones:
                all_zones_for_table.append({
                    "name": pair_name, "zone_id": "—",
                    "direction": None,
                    "proximal": None, "distal": None,
                    "bos_tag": None, "status": "No active zone",
                    "fvg_valid": False, "fvg_ghost": False,
                    "fvg_mitigation": "none",
                    "first_seen_ist": "—",
                    "is_new": False, "is_changed": False,
                    "is_placeholder_row": True,
                    "walls": pair_walls
                })
                continue

            # Price unknown (fetch failed this scan) — fall back to last
            # recorded price-at-scan from the most recent zone, else skip
            # selection and render placeholder.
            if current_price is None:
                fallback_price = None
                for z in active_zones:
                    if z.get("current_price_at_scan"):
                        fallback_price = z["current_price_at_scan"]
                        break
                if fallback_price is None:
                    all_zones_for_table.append({
                        "name": pair_name, "zone_id": "—",
                        "direction": None,
                        "proximal": None, "distal": None,
                        "bos_tag": None, "status": "Data unavailable",
                        "fvg_valid": False, "fvg_ghost": False,
                        "fvg_mitigation": "none",
                        "first_seen_ist": "—",
                        "is_new": False, "is_changed": False,
                        "is_placeholder_row": True,
                        "walls": pair_walls
                    })
                    continue
                current_price = fallback_price

            sz, in_progress = select_relevant_zone_for_pair(
                active_zones, current_price, dp
            )
            if sz is None:
                continue  # defensive — active_zones non-empty so shouldn't fire

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

            # Summary table row for the SELECTED zone.
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
                "is_changed": False,
                "is_placeholder_row": False,
                "in_progress": in_progress,
                "walls": pair_walls
            })

            chart_b64 = None
            cid = None
            if df is not None:
                cid = f"chart_{pair_name}_{chart_counter}"
                chart_b64 = generate_h1_chart(
                    df, ob_for_render, dp, pair_name, ist_ts_full,
                    walls=pair_walls
                )

            narrative = generate_zone_narrative_with_atr(
                ob_for_render, pair_name, dp, current_price, sz.get("h1_atr", 0.0)
            )

            card_html = build_active_zone_card_html(
                sz, pair_name, dp, narrative,
                cid if chart_b64 else None,
                ist_ts_full,
                current_price=current_price,
                in_progress=in_progress
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

        if all_zones_for_table or dropped_lines:
            summary_table = build_summary_table_html(all_zones_for_table, dp_map, pair_prices)
            try:
                # Prepend structure banner to the email content.
                if new_zone_cards:
                    _banner_html = (
                        f'<div style="background:#1a1f2e;border-left:3px solid #5dade2;'
                        f'padding:8px 12px;margin:0 0 10px 0;color:#dddddd;'
                        f'font-family:monospace;font-size:12px;white-space:pre-wrap;">'
                        f'{structure_banner_text}</div>'
                    )
                    new_zone_cards = [_banner_html] + new_zone_cards
                elif unchanged_zone_cards:
                    _banner_html = (
                        f'<div style="background:#1a1f2e;border-left:3px solid #5dade2;'
                        f'padding:8px 12px;margin:0 0 10px 0;color:#dddddd;'
                        f'font-family:monospace;font-size:12px;white-space:pre-wrap;">'
                        f'{structure_banner_text}</div>'
                    )
                    unchanged_zone_cards = [_banner_html] + unchanged_zone_cards
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

    # Phase 1 scan log — flush all records collected this scan in one append.
    # Single fsync, single rotation check. File path = current month.
    _write_phase1_scan_records(phase1_scan_records, ist_now)

    # Persist slate (overwrites file — but only mutates within-day, never resets
    # mid-day; new-day reset happens via slate_date check at scan start).
    save_slate(slate)
    print(f"Phase 1 complete at {ist_time_str} IST.")


if __name__ == "__main__":
    run_radar()
