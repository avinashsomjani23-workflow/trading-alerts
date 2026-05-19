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
import shutil
import sys

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
# Two-OB system. OB1 (Primary) = closest bias-correct OB within
# OB1_INNER_LIMIT_ATR x H1 ATR. OB2 (Alternative) = best Pristine OB in the
# ring [OB1_INNER_LIMIT_ATR, OB2_OUTER_LIMIT_ATR] x H1 ATR. OB2 surfaces
# the next zone above/below if price travels past OB1, even when no OB1
# exists. OBs beyond OB2_OUTER_LIMIT_ATR are dropped at proximity_gate.
# Phase 2 / Phase 3 only consume OB1 (primary). When price moves into OB2's
# range, OB2 naturally re-classifies as OB1 on the next scan.
# ---------------------------------------------------------------------------
OB1_INNER_LIMIT_ATR = 4.0
OB2_OUTER_LIMIT_ATR = 8.0

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
# OB PERSISTENCE — hard age cap on slate OBs
# ---------------------------------------------------------------------------
# Slate zones (active OBs) are persisted across scans and re-checked for
# mitigation each scan. Independent of the H1 data window: an OB built 10
# days ago stays alive as long as mitigation hasn't fired, even though its
# OB candle is long gone from the rolling 150-candle fetch.
#
# This cap retires OBs that have neither been tested nor invalidated after
# OB_MAX_AGE_DAYS. Vet rationale: a zone untouched for two weeks is
# functionally stale — institutions don't re-defend levels indefinitely.
OB_MAX_AGE_DAYS = 15

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
                              diagnostics=None, placeholder_diagnostic=None,
                              ob_build_diagnostics=None):  # TEMP DIAG — remove with OB build verification
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
        # TEMP DIAG — remove with OB build verification.
        # Per-event ledger of how detect_smc_radar handled each BOS/CHoCH:
        # built or dropped (with the gate that fired). One entry per event in
        # dealing_range.events for this pair, in chronological order.
        'ob_build_diagnostics': ob_build_diagnostics or [],
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
    # Both BOS and CHoCH carry a Major/Minor tier. Minor BOS = continuation
    # break of an internal lb-3 swing inside the active leg (walls don't move,
    # trend doesn't flip); Major BOS = trend-direction wall break.
    if bos_tag == 'BOS':
        if bos_tier == 'Minor':
            return 'Minor BOS'
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
# OB mitigation rule moved to smc_detector.is_ob_mitigated_phase1 so Phase 2
# can call the SAME function for its mid-day still-active gate. Re-exported
# here so existing call sites continue to work unchanged.
is_ob_mitigated_phase1 = smc_detector.is_ob_mitigated_phase1


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

    # TEMP DIAG — remove with OB build verification.
    # Per-event ledger of how each event was handled (built / dropped + gate).
    # Surfaced via the result dict and persisted in the Phase 1 scan log.
    ob_build_diagnostics = []

    def _diag_short(payload):
        """Compact one-line console string for an OB-drop event."""
        ts = (payload.get('event_ts') or '')[:16] or '?'
        et = payload.get('event_type', '?')
        di = payload.get('event_dir', '?')
        gate = payload.get('drop_gate', '?')
        det = payload.get('drop_detail', {}) or {}
        det_str = ", ".join(f"{k}={v}" for k, v in det.items())
        return f"  [OB-DROP] {pair_name or '?'} {di} {et} @ {ts} -> {gate} ({det_str})"

    if not events:
        return {"current_price": current_price_now,
                "active_unmitigated_obs": [],
                "ob_build_diagnostics": ob_build_diagnostics}  # TEMP DIAG

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

    for ev_pos, ev in enumerate(events):
        ev_type = ev.get('type')           # 'BOS' | 'CHoCH'
        ev_tier = ev.get('tier')           # 'Major' | 'Minor'
        ev_dir  = ev.get('direction')      # 'bullish' | 'bearish'

        # Most recent OPPOSING-direction structural event (Major/Minor BOS
        # or Major/Minor CHoCH against this event's direction). Used to
        # widen the sweep search window — the opposing event marks the
        # start of the current trend leg, and the catalysing sweep often
        # sits at or near that turn. Resolved from the events ring.
        # None when no opposing prior event exists yet, or when its candle
        # falls outside the df window.
        prior_event_idx = None
        for back_pos in range(ev_pos - 1, -1, -1):
            cand = events[back_pos]
            if cand.get('type') not in ('BOS', 'CHoCH'):
                continue
            if cand.get('direction') == ev_dir:
                continue
            prior_event_idx = _idx_from_ts(cand.get('candle_ts'))
            break
        if ev_type not in ('BOS', 'CHoCH'):
            continue

        # TEMP DIAG — remove with OB build verification.
        # Seed a diagnostic stub for this event; gates below fill drop_gate
        # and drop_detail when they reject, or flip outcome to 'built' on
        # success (post-build mitigation has its own pass below).
        diag = {
            'event_ts':    ev.get('candle_ts'),
            'event_type':  ev_type,
            'event_tier':  ev_tier,
            'event_dir':   ev_dir,
            'outcome':     'dropped',
            'drop_gate':   None,
            'drop_detail': {},
            'ob_idx_ts':   None,
            'ob_proximal': None,
            'ob_distal':   None,
        }

        # Update BOS chain bookkeeping. Only Major BOS contributes to the
        # chain count — Minor BOS is a continuation sub-event, not a fresh
        # leg, so it must not inflate the count Phase 2 uses to score caution.
        if ev_type == 'CHoCH' and ev_tier == 'Major':
            bos_seq_counter = 0
        elif ev_type == 'BOS' and ev_tier == 'Major':
            bos_seq_counter += 1
        # Minor BOS, Minor CHoCH: do not touch the counter.

        # Locate the break candle and impulse-leg start in CURRENT df.
        bos_idx = _idx_from_ts(ev.get('candle_ts'))
        impulse_start_idx = _idx_from_ts(ev.get('impulse_start_ts'))
        if bos_idx is None or impulse_start_idx is None:
            # Event references a candle outside the current df window. Common
            # for older events in the ring — they have no fresh OB to build.
            diag['drop_gate'] = 'event_outside_window'
            diag['drop_detail'] = {
                'candle_ts_resolved': bos_idx is not None,
                'impulse_start_ts_resolved': impulse_start_idx is not None,
                'df_first_ts': _ts_for_idx(0),
                'df_last_ts':  _ts_for_idx(n - 1),
            }
            ob_build_diagnostics.append(diag)
            print(_diag_short(diag))
            continue
        if impulse_start_idx >= bos_idx:
            diag['drop_gate'] = 'degenerate_leg'
            diag['drop_detail'] = {
                'impulse_start_idx': int(impulse_start_idx),
                'bos_idx': int(bos_idx),
            }
            ob_build_diagnostics.append(diag)
            print(_diag_short(diag))
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
        # TEMP DIAG — remove with OB build verification.
        # Track which sub-gate rejected the leg so we can tell whether the
        # walk found no opposing candle at all, all of them were oversized,
        # or all of them were near-doji.
        ob_idx = -1
        opposing_count = 0
        oversized_count = 0
        doji_count = 0
        for j in range(bos_idx - 1, impulse_start_idx - 1, -1):
            if (ev_dir == 'bullish' and C[j] < O[j]) or \
               (ev_dir == 'bearish' and C[j] > O[j]):
                opposing_count += 1
                if h1_atr_for_leg and h1_atr_for_leg > 0:
                    if (H[j] - L[j]) > smc_detector.OB_MAX_RANGE_ATR_MULT * h1_atr_for_leg:
                        oversized_count += 1
                        continue
                if is_valid_ob_candle(O[j], C[j], H[j], L[j]):
                    ob_idx = j
                    break
                else:
                    doji_count += 1
        if ob_idx == -1:
            diag['drop_gate'] = 'no_qualifying_ob_candle'
            diag['drop_detail'] = {
                'leg_len': int(bos_idx - impulse_start_idx),
                'opposing_candles_in_leg': opposing_count,
                'oversized_rejected': oversized_count,
                'doji_rejected': doji_count,
                'h1_atr': float(h1_atr_for_leg) if h1_atr_for_leg else 0.0,
            }
            ob_build_diagnostics.append(diag)
            print(_diag_short(diag))
            continue

        ob_high = float(H[ob_idx])
        ob_low  = float(L[ob_idx])
        ob_proximal = ob_high if ev_dir == 'bullish' else ob_low
        # TEMP DIAG — stamp resolved OB candle on the diag now that we have it.
        diag['ob_idx_ts'] = _ts_for_idx(ob_idx)
        diag['ob_proximal'] = ob_proximal
        diag['ob_distal'] = ob_low if ev_dir == 'bullish' else ob_high

        # Proximity gate. Two-OB system: keep OBs out to OB2_OUTER_LIMIT_ATR
        # so the outer-ring (alternative) OB is preserved. The OB1 vs OB2
        # split happens post-dedupe via _split_primary_alternative below.
        # Anything beyond OB2_OUTER_LIMIT_ATR is dropped — those OBs aren't
        # actionable in the current chart context regardless of quality.
        if h1_atr_for_leg and h1_atr_for_leg > 0:
            if abs(current_price_now - ob_proximal) > OB2_OUTER_LIMIT_ATR * h1_atr_for_leg:
                diag['drop_gate'] = 'proximity_gate'
                diag['drop_detail'] = {
                    'ob_proximal': ob_proximal,
                    'current_price': current_price_now,
                    'h1_atr': float(h1_atr_for_leg),
                    'distance_atr': abs(current_price_now - ob_proximal) / h1_atr_for_leg,
                    'limit_atr': OB2_OUTER_LIMIT_ATR,
                }
                ob_build_diagnostics.append(diag)
                print(_diag_short(diag))
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
            # c1_timestamp is the absolute ISO timestamp of the c1 candle.
            # Cross-phase chart rendering (Phase 2 H1 chart) must use this
            # because c1_idx is local to P1's H1 dataframe and does not
            # translate to P2's separate H1 fetch.
            'c1_timestamp': fvg_result.get('c1_timestamp'),
            'was_detected': fvg_result.get('was_detected', False),
            'mitigation':   fvg_result.get('mitigation', 'none'),
            'ghost_top':    fvg_result.get('ghost_top'),
            'ghost_bottom': fvg_result.get('ghost_bottom'),
            'ghost_c1_idx': fvg_result.get('ghost_c1_idx'),
            'ghost_c3_idx': fvg_result.get('ghost_c3_idx'),
            'ghost_c1_timestamp': fvg_result.get('ghost_c1_timestamp'),
            'mitigated_at_idx': fvg_result.get('mitigated_at_idx')
        }

        # Phase 1 sweep observation — snapshot semantics. Symmetric window
        # for BOS and CHoCH: [prior_opposing_event_idx, ob_idx]. The opposing
        # event marks where the current trend leg started; the catalysing
        # sweep often sits at or near that turn. Falls back to a 48-candle
        # lookback inside observe_phase1_sweep when prior_event_idx is None.
        try:
            sweep_obs = smc_detector.observe_phase1_sweep(
                df, ob_idx, impulse_start_idx, ev_dir,
                h1_atr_for_leg, pair_type, pair_name, tf_label='H1',
                event_type=ev_type,
                prior_event_idx=prior_event_idx,
            )
        except Exception as _sweep_err:
            logging.warning(f"[sweep_observed] OB build failed sweep observation: {_sweep_err}")
            sweep_obs = {'exists': False}

        # Dealing range snapshot — Phase 1 is the single source of truth
        # for DR. Computed once at OB build time using this scan's H1 frame,
        # then frozen on the zone. Phase 2 consumes ob['dealing_range']
        # directly and does NOT recompute. Same scan, same frame, same DR.
        ob_proximal_for_dr = ob_high if ev_dir == 'bullish' else ob_low
        ob_for_dr = {
            'proximal_line': ob_proximal_for_dr,
            'distal_line':   ob_low if ev_dir == 'bullish' else ob_high,
            'direction':     ev_dir,
            'ob_idx':        ob_idx,
            'bos_idx':       bos_idx,
            'bos_swing_price': bos_swing_price,
            'bos_tag':       ev_type,
            'bos_tier':      ev_tier,
        }
        try:
            dealing_range_snapshot = smc_detector.get_dealing_range(
                ob_for_dr, df, h1_atr_for_leg,
                pair_conf={'pair_type': pair_type, 'name': pair_name},
                current_price=ob_proximal_for_dr,
            )
        except Exception as _dr_snap_err:
            logging.warning(f"[dealing_range] OB snapshot failed: {_dr_snap_err}")
            dealing_range_snapshot = {'valid': False, 'source': 'snapshot_error'}

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
            'sweep_observed':     sweep_obs,
            'dealing_range':      dealing_range_snapshot,
            # TEMP DIAG — back-reference so post-build mitigation can amend
            # the right diag entry. Stripped before returning.
            '_diag_ref':          diag,
        })
        # TEMP DIAG — survived all construction gates; mark built. The
        # post-build mitigation pass below may flip this back to 'dropped'
        # with drop_gate='post_build_mitigation'.
        diag['outcome'] = 'built'
        ob_build_diagnostics.append(diag)
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
        else:
            # TEMP DIAG — flip the previously-built diag entry to dropped.
            _d = ob.get('_diag_ref')
            if _d is not None:
                _d['outcome'] = 'dropped'
                _d['drop_gate'] = 'post_build_mitigation'
                _d['drop_detail'] = {
                    'reason': _reason,
                    'touches': int(touches),
                    'proximal': ob['proximal_line'],
                    'distal':   ob['distal_line'],
                    'scanned_from_idx': int(ob['bos_idx'] + 1),
                    'scanned_to_idx':   int(n),
                }
                print(_diag_short(_d))

    # Two-OB system: keep ALL OBs through dedupe. The split into one
    # primary (OB1) and one alternative (OB2) per direction happens AFTER
    # dedupe in _split_primary_alternative. We can no longer pre-filter to
    # "latest OB per direction" because the alternative may come from an
    # earlier leg sitting in the 4-8 x ATR ring.
    filtered = list(tracked_obs)

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

    # Two-OB split per direction.
    #   OB1 (Primary):     bias-correct OB closest to current price within
    #                      OB1_INNER_LIMIT_ATR x H1 ATR. Touch state ignored
    #                      (current behaviour preserved — Pristine or Tested
    #                      both qualify; closest wins).
    #   OB2 (Alternative): best Pristine OB in the outer ring
    #                      (OB1_INNER_LIMIT_ATR, OB2_OUTER_LIMIT_ATR] x H1 ATR.
    #                      Same direction as bias. Tested OBs do not qualify.
    #                      Pick by confluence count, freshness tiebreak. OB2
    #                      surfaces independently of OB1 — when no OB1 exists,
    #                      OB2 still shows.
    # Each OB carries 'role' = 'primary' | 'alternative' for downstream
    # consumers (Phase 2/3 only consume primary; chart renders both).
    cur_price = float(C[-1])
    filtered = _split_primary_alternative(filtered, cur_price)

    # Strip the private dedupe hint before returning.
    for o in filtered:
        o.pop('_dedupe_thresh', None)
        o.pop('_diag_ref', None)  # TEMP DIAG — internal back-reference

    return {
        "current_price": cur_price,
        "active_unmitigated_obs": filtered,
        "ob_build_diagnostics": ob_build_diagnostics,  # TEMP DIAG
    }


def _count_confluences(ob):
    """
    Count distinct SMC confluences present on an OB.

    Confluences considered (each contributes 1 to the count):
      - FVG exists (any state — pristine or partial; ghost does not count)
      - Sweep observed (active/unbroken target — already filtered by
        observe_phase1_sweep, so any sweep present here qualifies)
      - broken_was_wall (the BOS that birthed this OB broke a dealing-range
        wall — meaningful structural displacement)

    Used to rank OB2 candidates. Higher count wins. Tie-breaks by freshness
    (higher bos_idx) live in the caller.
    """
    n = 0
    fvg = ob.get('fvg') or {}
    if fvg.get('exists'):
        n += 1
    sw = ob.get('sweep_observed') or {}
    if sw.get('exists'):
        n += 1
    if ob.get('broken_was_wall'):
        n += 1
    return n


def _split_primary_alternative(obs, cur_price):
    """
    Partition OBs into one Primary (OB1) + one Alternative (OB2) per
    direction. Tags each kept OB with `role` ∈ {'primary', 'alternative'}
    and stamps `_distance_atr` for downstream sorting / rendering.

    Rules — see docstring at call site for full SMC rationale:
      - Primary: bias-correct, distance ≤ OB1_INNER_LIMIT_ATR x H1 ATR,
        closest to current price. Pristine or Tested both qualify.
      - Alternative: bias-correct, OB1_INNER_LIMIT_ATR < distance ≤
        OB2_OUTER_LIMIT_ATR x H1 ATR, Pristine only (touches == 0). Pick by
        confluence count, freshness tiebreak.

    OB1 and OB2 are independent — OB2 surfaces even when no OB1 exists.
    """
    by_dir = {}
    for ob in obs:
        by_dir.setdefault(ob['direction'], []).append(ob)

    kept = []
    for direction, group in by_dir.items():
        primary = None
        primary_dist_atr = None
        alt_candidates = []

        for ob in group:
            atr = float(ob.get('h1_atr') or 0.0)
            if atr <= 0:
                continue  # cannot classify without ATR
            dist = abs(cur_price - float(ob['proximal_line']))
            dist_atr = dist / atr
            ob['_distance_atr'] = round(dist_atr, 3)

            if dist_atr <= OB1_INNER_LIMIT_ATR:
                # Primary candidate: closest to price wins.
                if primary is None or dist_atr < primary_dist_atr:
                    primary = ob
                    primary_dist_atr = dist_atr
            elif dist_atr <= OB2_OUTER_LIMIT_ATR:
                # Alternative candidate. Pristine gate.
                if int(ob.get('touches', 0)) == 0:
                    alt_candidates.append(ob)
            # else: outside outer limit — should have been gated at build,
            # defensive skip here.

        if primary is not None:
            primary['role'] = 'primary'
            kept.append(primary)

        if alt_candidates:
            # Sort: confluence count desc, then freshness (bos_idx desc).
            alt_candidates.sort(
                key=lambda o: (_count_confluences(o), int(o.get('bos_idx', 0))),
                reverse=True
            )
            alt = alt_candidates[0]
            alt['role'] = 'alternative'
            kept.append(alt)

    return kept

# ---------------------------------------------------------------------------
# CHART GENERATION
# ---------------------------------------------------------------------------

def generate_h1_chart(df, ob, dp, pair_name, ist_timestamp, walls=None,
                      is_invalidated=False, last_event=None, alt_ob=None):
    """
    H1 zone chart. Used for active zones, invalidated zones, and structure-only
    (no zone) views.

    Args:
      ob: PRIMARY zone dict (OB1) OR None for structure-only.
      walls: dealing_range walls dict.
      is_invalidated: True to render the OB band greyed out (dead zone).
      last_event: optional dict {type, tier, direction, ts} — when ob is None,
                  used to draw a marker at the most recent BOS/CHoCH so the
                  vet can see what defined the current dealing range.
      alt_ob: ALTERNATIVE zone dict (OB2) — same direction as ob, sits in
              the 4-8 x H1 ATR ring from current price. Rendered with reduced
              opacity and a "Alt" label so OB1 stays the loud focus. None
              when no alternative exists.

    Visual elements:
      - Candles (thin body 0.55, fat wick 1.5) — last 130 candles + 8 right margin
      - Zone band (proximal/distal, purple — greyed when is_invalidated)
      - OB candle outline (greyed when is_invalidated)
      - FVG (active or ghost) — skipped when ob is None or is_invalidated
      - BOS/CHoCH horizontal line + break candle outline (greyed when invalidated)
      - Current price (white)
      - DR ceiling/floor (always drawn): solid dotted if confirmed,
        dashed-gap if placeholder; faded color
      - Equilibrium line (only when both DR walls on-screen)
      - Swing markers: filled triangle for lookback-3 structural swings,
        muted yellow, placed outside the candle
      - Adaptive figure height when DR is wide vs candle range
    """
    try:
        full_df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy().reset_index(drop=True)
        n_full  = len(full_df)
        has_ob  = ob is not None
        ob_abs  = ob['ob_idx'] if has_ob else None

        # --- Plot window: 130 back from current, but always include OB candle.
        # Wider window (~5 days of H1) gives the vet enough context to trace
        # CHoCH breaks, the swing that was taken, and the levels in play.
        WINDOW_BACK = 130
        RIGHT_MARGIN = 8
        window_start = max(0, n_full - WINDOW_BACK)
        # If OB sits earlier than that, extend window back to include it.
        if has_ob and ob_abs < window_start:
            window_start = max(0, ob_abs - 3)

        df_plot = full_df.iloc[window_start:].copy().reset_index(drop=True)
        n_plot  = len(df_plot)
        ob_plot_idx = (ob_abs - window_start) if has_ob else None

        O = df_plot['Open'].values
        C = df_plot['Close'].values
        H = df_plot['High'].values
        L = df_plot['Low'].values

        if has_ob:
            proximal = float(ob['proximal_line'])
            distal   = float(ob['distal_line'])
            zone_lo  = min(proximal, distal)
            zone_hi  = max(proximal, distal)
        else:
            proximal = distal = zone_lo = zone_hi = None

        # Alt OB (OB2) — same direction, sits in the 4-8xATR ring. Optional.
        has_alt_ob = (alt_ob is not None) and has_ob
        if has_alt_ob:
            alt_proximal = float(alt_ob['proximal_line'])
            alt_distal   = float(alt_ob['distal_line'])
            alt_zone_lo  = min(alt_proximal, alt_distal)
            alt_zone_hi  = max(alt_proximal, alt_distal)
            alt_ob_abs   = alt_ob.get('ob_idx')
            alt_ob_plot_idx = (alt_ob_abs - window_start) if alt_ob_abs is not None else None
        else:
            alt_proximal = alt_distal = alt_zone_lo = alt_zone_hi = None
            alt_ob_plot_idx = None

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
        h1_atr = (ob.get('h1_atr', 0.0) or 0.0) if has_ob else 0.0
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

        ymin_candidates = [candle_lo]
        ymax_candidates = [candle_hi]
        if has_ob:
            ymin_candidates.append(zone_lo)
            ymax_candidates.append(zone_hi)
            if ob['fvg'].get('exists') or ob['fvg'].get('was_detected'):
                ft = ob['fvg'].get('fvg_top') or ob['fvg'].get('ghost_top')
                fb = ob['fvg'].get('fvg_bottom') or ob['fvg'].get('ghost_bottom')
                if ft is not None and fb is not None:
                    ymin_candidates.append(float(fb))
                    ymax_candidates.append(float(ft))
        if has_alt_ob:
            ymin_candidates.append(alt_zone_lo)
            ymax_candidates.append(alt_zone_hi)
        ymin_candidates.append(float(C[-1]))
        ymax_candidates.append(float(C[-1]))
        if ceiling_in_view:
            ymax_candidates.append(float(ceiling_price))
        if floor_in_view:
            ymin_candidates.append(float(floor_price))

        # Off-window event broken-swing price: include in y-range so the
        # horizontal level annotation (drawn below) has room to render. Only
        # added when no OB is on this chart and the level falls within the
        # same proximity band used for walls (so a wildly stale level can't
        # drag the y-axis far from current candles).
        _off_window_bws_for_y = None
        if (not has_ob) and last_event is not None:
            _le_bws_y = last_event.get('broken_swing_price')
            if _le_bws_y is not None:
                try:
                    _le_bws_y_f = float(_le_bws_y)
                except Exception:
                    _le_bws_y_f = None
                if _le_bws_y_f is not None and _wall_in_view(_le_bws_y_f):
                    ymin_candidates.append(_le_bws_y_f)
                    ymax_candidates.append(_le_bws_y_f)
                    _off_window_bws_for_y = _le_bws_y_f

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
        # Taller, slightly narrower frame (12 wide × 7.5+ tall) so 130 candles
        # render with TradingView-style proportions: narrow bodies, tall wicks,
        # vertical price space. Base height 7.5; bump up to 11 as DR/zone
        # forces a wider y span.
        candle_range = max(candle_hi - candle_lo, 1e-9)
        required_range = y_max - y_min
        ratio = required_range / candle_range
        base_h = 7.5
        if ratio > 1.5:
            base_h = min(11.0, 7.5 + (ratio - 1.5) * 1.7)
        fig, ax = plt.subplots(1, 1, figsize=(12, base_h), facecolor='#131722')
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
        # Greyed out when invalidated (dead zone — corpse only).
        if has_ob:
            if is_invalidated:
                band_face, band_edge, band_alpha = '#666666', '#888888', 0.10
            else:
                band_face, band_edge, band_alpha = '#9b59b6', '#bb8fce', 0.12
            zone_x_start = max(0, ob_plot_idx - 0.5)
            zone_width   = (n_plot + RIGHT_MARGIN - 1) - zone_x_start
            ax.add_patch(patches.Rectangle(
                (zone_x_start, zone_lo), zone_width, zone_hi - zone_lo,
                facecolor=band_face, alpha=band_alpha, zorder=1
            ))
            ax.add_patch(patches.Rectangle(
                (zone_x_start, zone_lo), zone_width, zone_hi - zone_lo,
                fill=False, edgecolor=band_edge, linestyle=':', linewidth=1.5, zorder=2
            ))

            # --- OB candle outline ---
            if 0 <= ob_plot_idx < n_plot:
                ob_h = float(H[ob_plot_idx])
                ob_l = float(L[ob_plot_idx])
                ob_outline = '#888888' if is_invalidated else '#d7bde2'
                ax.add_patch(patches.Rectangle(
                    (ob_plot_idx - 0.5, ob_l), 1.0, ob_h - ob_l,
                    fill=False, edgecolor=ob_outline, linewidth=2.0, zorder=4,
                    linestyle='-'
                ))

            # --- Alternative zone band (OB2) ---
            # Reduced opacity so OB1 stays the loud focus. Same band style,
            # paler fill / paler edge. Tag "Alt" on the right edge so the
            # vet immediately knows which zone is the alternative.
            if has_alt_ob and alt_ob_plot_idx is not None:
                alt_x_start = max(0, alt_ob_plot_idx - 0.5)
                alt_width   = (n_plot + RIGHT_MARGIN - 1) - alt_x_start
                if alt_width > 0:
                    ax.add_patch(patches.Rectangle(
                        (alt_x_start, alt_zone_lo), alt_width, alt_zone_hi - alt_zone_lo,
                        facecolor='#9b59b6', alpha=0.06, zorder=1
                    ))
                    ax.add_patch(patches.Rectangle(
                        (alt_x_start, alt_zone_lo), alt_width, alt_zone_hi - alt_zone_lo,
                        fill=False, edgecolor='#bb8fce', linestyle=(0, (4, 3)),
                        linewidth=1.0, alpha=0.55, zorder=2
                    ))
                    # Alt OB candle outline (lighter than primary).
                    if 0 <= alt_ob_plot_idx < n_plot:
                        a_h = float(H[alt_ob_plot_idx])
                        a_l = float(L[alt_ob_plot_idx])
                        ax.add_patch(patches.Rectangle(
                            (alt_ob_plot_idx - 0.5, a_l), 1.0, a_h - a_l,
                            fill=False, edgecolor='#bb8fce', linewidth=1.2,
                            alpha=0.6, zorder=4, linestyle='-'
                        ))
                    # "Alt" tag on the right edge of the band, aligned to
                    # the alt zone midline — keeps it readable.
                    alt_mid = (alt_zone_lo + alt_zone_hi) / 2.0
                    ax.annotate(
                        'Alt OB', xy=(n_plot + RIGHT_MARGIN - 1, alt_mid),
                        xytext=(-4, 0), textcoords='offset points',
                        color='#bb8fce', fontsize=8, alpha=0.7,
                        ha='right', va='center', zorder=5,
                    )

        # --- FVG outline ---
        # Skipped entirely for structure-only (no ob) and invalidated views —
        # FVG is meaningful only for live zones.
        fvg_active = fvg_ghost = fvg_partial = False
        if has_ob and not is_invalidated:
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
        bos_price = None
        bos_color = '#00bcd4'  # default; reassigned below from ob or last_event
        if has_ob:
            bos_price = float(ob['bos_swing_price'])
            _btag = ob.get('bos_tag', 'BOS')
            _btier = ob.get('bos_tier', 'Major')
            if _btag == 'BOS':
                bos_color = '#00897b' if _btier == 'Minor' else '#00bcd4'
            elif _btier == 'Minor':
                bos_color = '#9c27b0'
            else:
                bos_color = '#ff9800'
            # Greyed for invalidated zones (the structure is historical now).
            line_alpha = 0.3 if is_invalidated else 0.7
            edge_color = '#888888' if is_invalidated else bos_color
            ax.axhline(y=bos_price, color=edge_color, linewidth=0.8,
                       linestyle='--', alpha=line_alpha, zorder=2)

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
                            fill=False, edgecolor=edge_color, linewidth=1.5, zorder=5
                        ))
        elif last_event is not None:
            # Structure-only: mark the last BOS/CHoCH so the vet can see what
            # defined the current dealing range.
            #   - In-window: vertical guide on the event candle column.
            #   - Off-window (event candle older than n_full - window_start):
            #     horizontal line at the broken-swing price with a "N candles
            #     ago" label at the left edge. The level matters more than the
            #     candle — drawing it on-chart lets the vet trace what's
            #     defining the current dealing range without us widening the
            #     candle window (which would shrink price detail).
            le_type = last_event.get('type')
            le_tier = last_event.get('tier', 'Major')
            le_dir  = last_event.get('direction')
            le_ts   = last_event.get('ts')
            le_bws  = last_event.get('broken_swing_price')
            if le_type == 'BOS':
                ev_color = '#00897b' if le_tier == 'Minor' else '#00bcd4'
            elif le_tier == 'Minor':
                ev_color = '#9c27b0'
            else:
                ev_color = '#ff9800'
            le_idx = None
            if le_ts:
                # Match against full_df since window_start may chop history.
                ts_col = full_df['Datetime'] if 'Datetime' in full_df.columns else full_df.index
                for k in range(len(full_df)):
                    raw = ts_col.iloc[k] if hasattr(ts_col, 'iloc') else ts_col[k]
                    raw_iso = raw.isoformat() if hasattr(raw, 'isoformat') else str(raw)
                    if raw_iso == le_ts:
                        le_idx = k
                        break
            if le_idx is not None and le_idx >= window_start:
                local_i = le_idx - window_start
                if 0 <= local_i < n_plot:
                    ax.axvline(x=local_i, color=ev_color, linewidth=0.8,
                               linestyle='--', alpha=0.45, zorder=2)
                    ax.text(local_i, y_max, f"  {le_type} {le_tier}",
                            color=ev_color, fontsize=8, fontweight='bold',
                            ha='left', va='top', zorder=7,
                            bbox=dict(facecolor='#131722', edgecolor='none',
                                      pad=1.5, alpha=0.78))
            elif le_bws is not None:
                # Event is off-window. Draw the broken-swing as a horizontal
                # level with a left-edge "N candles ago" label so the vet
                # still sees the level defining the current dealing range.
                # Only when the level sits inside the candle proximity band
                # (mirrors _wall_in_view) — a wildly stale level would just
                # be a stray line; skip rendering in that case.
                try:
                    bws_f = float(le_bws)
                except Exception:
                    bws_f = None
                if bws_f is not None and _wall_in_view(bws_f):
                    candles_ago = None
                    if le_idx is not None:
                        candles_ago = max(0, n_full - 1 - le_idx)
                    ax.axhline(y=bws_f, color=ev_color, linewidth=0.8,
                               linestyle='--', alpha=0.55, zorder=2)
                    if le_type == 'BOS':
                        ev_name = "Minor BOS" if le_tier == 'Minor' else "BOS"
                    else:
                        ev_name = f"{le_tier} CHoCH"
                    dir_part = f" {le_dir}" if le_dir else ""
                    age_part = (f" · {candles_ago}c ago"
                                if candles_ago is not None else "")
                    ax.text(0, bws_f, f"  {ev_name}{dir_part}{age_part}",
                            color=ev_color, fontsize=8, fontweight='bold',
                            ha='left', va='center', zorder=7,
                            bbox=dict(facecolor='#131722', edgecolor='none',
                                      pad=1.5, alpha=0.78))

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
        # (structural swings; drive walls / BOS / both CHoCH tiers).
        SWING_COLOR = '#d4a017'
        try:
            swings_lb3 = smc_detector.get_swing_points(full_df, lookback=3)
        except Exception:
            swings_lb3 = []
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

        # --- Sweep candle marker (Phase 1 sweep observation) ---
        # Star at the wick tip + dotted level line from sweep candle BACK
        # to the swept swing's idx. NO wick highlight — the bar bloom hid
        # the candle itself. Star is drawn for EVERY tier (textbook /
        # decent / weak); color varies so visual intensity still hints at
        # quality, but presence is consistent.
        sw = (ob.get('sweep_observed') or {}) if (has_ob and not is_invalidated) else {}
        if sw.get('exists'):
            sw_abs_idx = sw.get('sweep_idx')
            sw_level = sw.get('price')
            sw_tier = sw.get('tier', 'weak')
            sw_swept_idx = sw.get('swept_swing_idx')  # may be None on legacy snapshots
            SWEEP_COLOR_MAP = {
                'textbook': '#00e5ff',
                'decent':   '#26c6da',
                'weak':     '#80deea',
            }
            SWEEP_COLOR = SWEEP_COLOR_MAP.get(sw_tier, '#80deea')
            if sw_abs_idx is not None and sw_level is not None:
                sw_local = sw_abs_idx - window_start
                if 0 <= sw_local < n_plot:
                    sw_h = float(H[sw_local])
                    sw_l = float(L[sw_local])
                    # Star at the wick tip — every sweep tier.
                    wick_tip = sw_l if ob['direction'] == 'bullish' else sw_h
                    ax.scatter([sw_local], [wick_tip], marker='*', s=140,
                               color=SWEEP_COLOR, edgecolors='#001f24',
                               linewidths=0.8, zorder=8)
                    # Dotted line at the swept price level — extends from
                    # the sweep candle BACK to the swept swing's idx so
                    # the vet can see exactly which swing was taken.
                    # Falls back to a 6-candle stub when swept_swing_idx
                    # isn't on the snapshot (legacy zones).
                    if sw_swept_idx is not None:
                        lvl_x_start = max(0, int(sw_swept_idx) - window_start)
                    else:
                        lvl_x_start = max(0, sw_local - 6)
                    if lvl_x_start < sw_local:
                        ax.plot([lvl_x_start, sw_local], [sw_level, sw_level],
                                color=SWEEP_COLOR, linewidth=1.0,
                                linestyle=(0, (3, 2)), alpha=0.8, zorder=4)
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
        mid_labels = [(current_price, f"{current_price:.{dp}f}", '#ffffff')]
        if has_ob:
            zone_label_color = '#888888' if is_invalidated else '#bb8fce'
            mid_labels.append((proximal, f"{proximal:.{dp}f}", zone_label_color))
            mid_labels.append((distal,   f"{distal:.{dp}f}",   zone_label_color))
            if bos_price is not None:
                bos_label_color = '#888888' if is_invalidated else bos_color
                mid_labels.append((bos_price, f"{bos_price:.{dp}f}", bos_label_color))
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

        if has_ob:
            direction_label = "Demand" if ob['direction'] == 'bullish' else "Supply"
            event_label = _event_label(ob.get('bos_tag', 'BOS'), ob.get('bos_tier', 'Major'))
            state_label = "INVALIDATED" if is_invalidated else ob.get('status', 'Pristine')
            title = (
                f"{pair_name} | {direction_label} Zone | {event_label} | "
                f"{state_label}   —   {ist_timestamp} IST"
            )
        else:
            title = f"{pair_name} | No Active Zone — Structure View   —   {ist_timestamp} IST"
        title_color = '#888888' if is_invalidated else '#dddddd'
        ax.set_title(title, color=title_color, fontsize=10, pad=8, loc='left')
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
    logging.info(f"[OB_BODY_RATIO] {name} zone {ob.get('zone_id','?')}: ob_body={ob['ob_body']:.{dp}f} median_leg={ob['median_leg_body']:.{dp}f} ratio={ratio}x")
    prompt = f"""You are a veteran SMC (Smart Money Concepts) prop trader writing a zone briefing for another experienced SMC trader.
Be direct. No fluff. No pleasantries. Four sentences only. One paragraph.

ZONE DATA — use these exact values, do not recalculate:
- Pair: {name}
- Bias: {direction} | Structure event: {event_label}
- Proximal: {proximal:.{dp}f} | Distal: {distal:.{dp}f}
- Zone width: {zone_pips} pips
- FVG: {fvg_status}
- Zone status: {ob['status']}
- Current price: {current_price:.{dp}f} | Distance to proximal: {dist_pips} pips

WRITE EXACTLY FOUR SENTENCES IN THIS ORDER:
1. What structure event ({event_label}) created this zone and why institutional accumulation is likely here.
2. OB quality: assess whether pristine or tested means strength or caution.
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
    logging.info(f"[OB_BODY_RATIO] {name} zone {ob.get('zone_id','?')}: ob_body={ob['ob_body']:.{dp}f} median_leg={ob['median_leg_body']:.{dp}f} ratio={round(ob['ob_body']/ob['median_leg_body'],2):.2f}x (fallback narrative)")
    return (
        f"{event_label} confirmed the {ob['direction']} shift. "
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


def _walls_icon_cell(walls):
    """Compact icon for the Walls column in the phone summary table.

    ✓✓ both anchored, ✓– one anchored, –– cold-start fallback. Tooltip via
    title= holds the long-form label so a long-press on phones reveals it.
    """
    if not walls or not isinstance(walls, dict):
        return ("<span style='color:#888;' title='No wall data'>&mdash;&mdash;</span>")
    if walls.get("fallback_active"):
        return ("<span style='color:#e74c3c;' title='Fallback — no recent BOS/CHoCH'>"
                "&mdash;&mdash;</span>")
    cph = walls.get("ceiling_is_placeholder", True)
    fph = walls.get("floor_is_placeholder", True)
    if not cph and not fph:
        return "<span style='color:#27ae60;' title='Both walls anchored'>&#10003;&#10003;</span>"
    if cph and fph:
        return "<span style='color:#e67e22;' title='Both walls placeholder'>&#9888;&#9888;</span>"
    if cph:
        return "<span style='color:#e67e22;' title='Ceiling placeholder, floor anchored'>&#9888;&#10003;</span>"
    return "<span style='color:#e67e22;' title='Floor placeholder, ceiling anchored'>&#10003;&#9888;</span>"


def _pip_unit(dp):
    return 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)


def build_summary_table_html(all_zones_for_table, dp_map, pair_prices=None):
    """Phone-friendly summary table — icons only, no prices.

    Columns: Pair | Bias | Event | OB | FVG | Sweep | Walls

    Numbers / prices live in the zone card. Trend (Bias) and Structure type
    (Event) are preserved per user requirement. First Seen moved into the
    zone card header so phone view stays compact.
    """
    pair_prices = pair_prices or {}
    rows = ""
    for z in all_zones_for_table:
        name      = z['name']
        walls     = z.get('walls', {})
        walls_cell = _walls_icon_cell(walls)

        # Placeholder row: pair has no active zone (or data unavailable).
        # All confluence cells dashed; walls cell still drawn so user can
        # see structure health at a glance.
        if z.get('is_placeholder_row'):
            rows += f"""
        <tr style="background:transparent;border-bottom:1px solid #2a2a3e;opacity:0.55;">
          <td style="padding:6px 6px;font-weight:bold;color:#aaa;font-size:12px;white-space:nowrap;">{name}</td>
          <td style="padding:6px 4px;text-align:center;color:#666;font-size:13px;">&mdash;</td>
          <td style="padding:6px 4px;text-align:center;color:#666;font-size:10px;">&mdash;</td>
          <td style="padding:6px 4px;text-align:center;color:#666;font-size:13px;">&mdash;</td>
          <td style="padding:6px 4px;text-align:center;color:#666;font-size:13px;">&mdash;</td>
          <td style="padding:6px 4px;text-align:center;color:#666;font-size:13px;">&mdash;</td>
          <td style="padding:6px 4px;text-align:center;font-size:11px;">{walls_cell}</td>
        </tr>"""
            continue

        # --- Bias glyph (kept per user requirement: trend direction stays).
        if z['direction'] == 'bullish':
            bias_glyph, bias_col = '&#9650;', '#27ae60'
        else:
            bias_glyph, bias_col = '&#9660;', '#e74c3c'

        # --- Event chip (kept per user requirement: structure type stays).
        _tier = z.get('bos_tier', 'Major')
        if z['bos_tag'] == 'BOS':
            if _tier == 'Minor':
                ev_color, ev_text = '#00897b', 'mBOS'
            else:
                ev_color, ev_text = '#00bcd4', 'BOS'
        elif _tier == 'Minor':
            ev_color, ev_text = '#9c27b0', 'mCH'
        else:
            ev_color, ev_text = '#ff9800', 'CHoCH'
        event_chip = (f"<span style='background:{ev_color};color:#000;font-size:9px;"
                      f"padding:1px 4px;border-radius:3px;font-weight:bold;'>{ev_text}</span>")

        # --- OB cell: ✓ pristine, ◐ tested, – none.
        touches = z.get('touches', 0)
        if touches == 0:
            ob_glyph, ob_col, ob_title = '&#10003;', '#27ae60', 'Pristine OB'
        else:
            ob_glyph, ob_col, ob_title = '&#9680;', '#e67e22', f'Tested {touches}x'

        # --- FVG cell: ✓ pristine, ◐ partial, ○ ghost, – none.
        if z['fvg_valid'] and z.get('fvg_mitigation') == 'partial':
            fvg_glyph, fvg_col, fvg_title = '&#9680;', '#f1c40f', 'FVG partial'
        elif z['fvg_valid']:
            fvg_glyph, fvg_col, fvg_title = '&#10003;', '#27ae60', 'FVG pristine'
        elif z.get('fvg_ghost'):
            fvg_glyph, fvg_col, fvg_title = '&#9675;', '#888', 'FVG ghost (mitigated)'
        else:
            fvg_glyph, fvg_col, fvg_title = '&ndash;', '#666', 'No FVG'

        # --- Sweep cell: star for every sweep, color encodes tier.
        # textbook = green, decent = yellow, weak = grey.
        # Detail (tier name) sits in the chart legend + zone narrative.
        sw = z.get('sweep_observed') or {}
        if not sw.get('exists'):
            sw_glyph, sw_col, sw_title = '&ndash;', '#666', 'No sweep'
        else:
            tier = (sw.get('tier') or 'weak').lower()
            if tier == 'textbook':
                sw_glyph, sw_col, sw_title = '&#9733;', '#27ae60', 'Textbook sweep'
            elif tier == 'decent':
                sw_glyph, sw_col, sw_title = '&#9733;', '#f1c40f', 'Decent sweep'
            else:
                sw_glyph, sw_col, sw_title = '&#9733;', '#888', 'Weak sweep'

        is_new     = z.get('is_new', False)
        is_changed = z.get('is_changed', False)
        row_bg     = '#1e3a2f' if is_new else ('#2d2a1a' if is_changed else 'transparent')
        # NEW / UPD badge inline next to pair name — small, doesn't blow up width.
        suffix_badge = ""
        if is_new:
            suffix_badge = (" <span style='background:#27ae60;color:#fff;font-size:8px;"
                            "padding:0 3px;border-radius:2px;'>N</span>")
        elif is_changed:
            suffix_badge = (" <span style='background:#e67e22;color:#fff;font-size:8px;"
                            "padding:0 3px;border-radius:2px;'>U</span>")

        rows += f"""
        <tr style="background:{row_bg};border-bottom:1px solid #2a2a3e;">
          <td style="padding:6px 6px;font-weight:bold;color:#eee;font-size:12px;white-space:nowrap;">{name}{suffix_badge}</td>
          <td style="padding:6px 4px;text-align:center;color:{bias_col};font-size:13px;">{bias_glyph}</td>
          <td style="padding:6px 4px;text-align:center;">{event_chip}</td>
          <td style="padding:6px 4px;text-align:center;color:{ob_col};font-size:13px;" title="{ob_title}">{ob_glyph}</td>
          <td style="padding:6px 4px;text-align:center;color:{fvg_col};font-size:13px;" title="{fvg_title}">{fvg_glyph}</td>
          <td style="padding:6px 4px;text-align:center;color:{sw_col};font-size:13px;" title="{sw_title}">{sw_glyph}</td>
          <td style="padding:6px 4px;text-align:center;font-size:11px;">{walls_cell}</td>
        </tr>"""

    return f"""
    <div style="margin-bottom:18px;">
      <h3 style="color:#aaa;font-size:12px;letter-spacing:1px;margin:0 0 6px 0;text-transform:uppercase;">
        Active Zone Map
      </h3>
      <table style="width:100%;border-collapse:collapse;background:#1a1a2e;border-radius:6px;overflow:hidden;">
        <thead>
          <tr style="background:#0d0d1a;">
            <th style="padding:6px 6px;text-align:left;color:#666;font-size:9px;font-weight:normal;text-transform:uppercase;letter-spacing:0.4px;">Pair</th>
            <th style="padding:6px 4px;text-align:center;color:#666;font-size:9px;font-weight:normal;text-transform:uppercase;letter-spacing:0.4px;">Bias</th>
            <th style="padding:6px 4px;text-align:center;color:#666;font-size:9px;font-weight:normal;text-transform:uppercase;letter-spacing:0.4px;">Event</th>
            <th style="padding:6px 4px;text-align:center;color:#666;font-size:9px;font-weight:normal;text-transform:uppercase;letter-spacing:0.4px;">OB</th>
            <th style="padding:6px 4px;text-align:center;color:#666;font-size:9px;font-weight:normal;text-transform:uppercase;letter-spacing:0.4px;">FVG</th>
            <th style="padding:6px 4px;text-align:center;color:#666;font-size:9px;font-weight:normal;text-transform:uppercase;letter-spacing:0.4px;">Sweep</th>
            <th style="padding:6px 4px;text-align:center;color:#666;font-size:9px;font-weight:normal;text-transform:uppercase;letter-spacing:0.4px;">Walls</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

def _phase1_chart_legend_html(bos_tag="BOS", bos_tier="Major"):
    """Colour-code legend rendered ONCE at the bottom of the Phase 1 digest.

    Args kept for backwards-compat call sites, but ignored — the global legend
    surfaces ALL three structure-event colours at once instead of switching
    based on a single zone's event.
    """
    items = [
        ('#bb8fce', 'Primary zone band (OB1 — closest to price) / greyed when invalidated'),
        ('#d7bde2', 'OB candle outline — greyed when invalidated'),
        ('#bb8fce', 'Alternative zone band (OB2 — best Pristine OB further out, dashed + faded)'),
        ('#2ecc71', 'FVG pristine (displacement)'),
        ('#f1c40f', 'FVG partial (proximal touched)'),
        ('#888888', 'FVG mitigated (ghost) / Invalidated zone'),
        ('#00bcd4', 'Major BOS break candle / level (wall break)'),
        ('#00897b', 'Minor BOS break candle / level (internal continuation)'),
        ('#ff9800', 'Major CHoCH break candle / level'),
        ('#9c27b0', 'Minor CHoCH break candle / level'),
        ('#5dade2', 'Dealing range walls (dotted=anchored, dashed=placeholder; far walls render as edge labels)'),
        ('#85c1e9', 'Equilibrium (50%)'),
        ('#d4a017', 'Swing: ▲▼ lookback-3 (structural swings — walls / BOS / CHoCH)'),
        ('#00e5ff', 'Sweep — ★ at wick tip + dotted line back to the swept swing (color: textbook=cyan, decent=teal, weak=pale)'),
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
    # Legend rendered ONCE at the end of the email (see send_master_digest_v2),
    # not per-card — frees up vertical space on phones.

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
    # Legend rendered ONCE at the end of the email (see send_master_digest_v2),
    # not per-card — frees up vertical space on phones.

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
    </div>"""


def build_dropped_zone_line(sz, name, dp):
    """One-line note for a dropped zone."""
    reason_map = {
        "mitigated_distal_break": "invalidated — price touched distal",
        "mitigated_three_touches": "mitigated — proximal hit 3 times",
        "structure_supplanted": "replaced by fresher structure (same leg)",
        "aged_out_of_window": f"OB older than {OB_MAX_AGE_DAYS} days — retired",
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


def _slate_zone_to_ob_shape(sz):
    """Reshape a slate zone dict into the OB-shape dict generate_h1_chart wants.

    Slate zones carry the same field names as fresh OBs, so this is a thin
    adaptor — exists so callers don't repeat the same dict-shaping inline.
    """
    return {
        "direction": sz["direction"],
        "bos_tag":   sz.get("bos_tag", "BOS"),
        "bos_tier":  sz.get("bos_tier", "Major"),
        "proximal_line": sz["proximal_line"],
        "distal_line":   sz["distal_line"],
        "high":      sz["high"],
        "low":       sz["low"],
        "ob_body":   sz["ob_body"],
        "median_leg_body": sz["median_leg_body"],
        "ob_idx":    sz["ob_idx"],
        "bos_idx":   sz["bos_idx"],
        "bos_swing_price":   sz["bos_swing_price"],
        "impulse_start_idx": sz["impulse_start_idx"],
        "impulse_start_price": sz["impulse_start_price"],
        "fvg":       sz["fvg"],
        "touches":   sz.get("touches", 0),
        "status":    sz.get("status_label", "Pristine"),
        "h1_atr":    sz.get("h1_atr", 0.0),
        "sweep_observed": sz.get("sweep_observed", {"exists": False}),
    }


_INVALIDATION_REASON_LONG = {
    "mitigated_distal_break":  "price closed beyond distal — zone is dead",
    "mitigated_three_touches": "proximal was wicked three times — zone is mitigated",
    "structure_supplanted":    "fresher structure on the same leg replaced this OB",
    "aged_out_of_window":      f"OB older than {OB_MAX_AGE_DAYS} days — auto-retired",
    "data_unavailable":        "pair data feed failed — could not verify the zone",
    "data_stale":              "yfinance data went stale — could not verify the zone",
}


def build_invalidation_card_html(sz, name, dp, cid, ist_timestamp):
    """One-paragraph card stating WHY the OB was invalidated. Chart sits below.

    Per CLAUDE.md: plain English, vet voice, no fluff. The chart (greyed-out OB)
    carries the visual; this card carries the reason.
    """
    direction = "Bullish demand" if sz['direction'] == 'bullish' else "Bearish supply"
    dir_color = '#27ae60' if sz['direction'] == 'bullish' else '#e74c3c'
    reason_key = sz.get('drop_reason', '')
    reason_text = _INVALIDATION_REASON_LONG.get(reason_key, reason_key or 'unknown reason')
    last_seen = sz.get('last_seen_label', '—')
    first_seen = sz.get('first_seen_label', '—')
    chart_html = (
        f'<img src="cid:{cid}" style="width:100%;border-radius:6px;display:block;margin-top:10px;" />'
        if cid else
        '<div style="padding:8px 12px;background:#2d1a1a;border-left:3px solid #e74c3c;'
        'border-radius:4px;color:#e74c3c;font-size:11px;">&#9888; Chart unavailable.</div>'
    )
    zone_id = sz.get('zone_id', '—')
    return f"""
    <div style="margin-bottom:14px;padding:12px 14px;background:#1a1a2e;
                border-left:3px solid #888;border-radius:6px;opacity:0.92;">
      <div style="margin-bottom:6px;">
        <span style="font-size:10px;color:#666;font-family:monospace;margin-right:6px;">{zone_id}</span>
        <span style="font-size:13px;font-weight:bold;color:#bbb;">{name}</span>
        <span style="font-size:11px;color:{dir_color};margin-left:8px;">{direction}</span>
        <span style="background:#888;color:#000;font-size:9px;padding:1px 5px;
                     border-radius:3px;font-weight:bold;margin-left:8px;">INVALIDATED</span>
      </div>
      <p style="font-size:12px;color:#bbb;line-height:1.5;margin:0 0 6px 0;">
        Zone is dead — {reason_text}. Lived from {first_seen} to {last_seen}.
      </p>
      {chart_html}
    </div>"""


_OB_DROP_GATE_LABELS = {
    "event_outside_window":   "event candle outside H1 data window",
    "degenerate_leg":         "impulse leg too short to form an OB",
    "no_qualifying_ob_candle": "no opposing candle in the leg qualified",
    "pd_array_gate":          "OB sat on the wrong side of equilibrium",
    "proximity_gate":          "price too far from OB to be actionable",
    "post_build_mitigation":   "OB built then died (close beyond distal or 3 touches)",
}


def _summarise_last_ob_attempt(ob_build_diagnostics):
    """Pick the most recent meaningful OB build attempt and return a short
    English line.

    Used in the 'no active OB' card so the vet can see WHY nothing is being
    surfaced. Filters out:
      - Built (non-dropped) entries — they aren't a 'rejected attempt'.
      - 'event_outside_window' drops — plumbing artefact (event candle rolled
        out of the H1 fetch window), tells the vet nothing actionable.
    Returns "" when nothing informative remains after filtering.
    """
    if not ob_build_diagnostics:
        return ""
    relevant = [
        d for d in ob_build_diagnostics
        if d.get('drop_gate') and d.get('drop_gate') != 'event_outside_window'
    ]
    if not relevant:
        return ""
    try:
        latest = max(relevant, key=lambda d: d.get('event_ts') or '')
    except Exception:
        latest = relevant[-1]
    gate = latest.get('drop_gate')
    ev_ts = latest.get('event_ts') or ''
    ev_type = latest.get('event_type') or '?'
    ev_dir = latest.get('event_dir') or ''
    gate_text = _OB_DROP_GATE_LABELS.get(gate, gate.replace('_', ' '))
    try:
        ts_short = ev_ts[:10]
    except Exception:
        ts_short = str(ev_ts)
    return (
        f"Last OB attempt: {ev_type} {ev_dir} on {ts_short} — "
        f"dropped: {gate_text}."
    )


def build_inactive_pair_card_html(name, dp, cid, ist_timestamp, walls,
                                  last_event, ob_build_diagnostics=None):
    """Card for a pair with no active OB and no recently-dropped OB.

    The vet wants to see WHY there's nothing to trade and what to wait for.
    Caption is plain-English: stop-and-think trigger for both a pullback OR a
    range-fail CHoCH. When OB build attempts exist, surface the most recent
    drop reason so the vet knows the system saw the setup and rejected it.
    """
    chart_html = (
        f'<img src="cid:{cid}" style="width:100%;border-radius:6px;display:block;margin-top:10px;" />'
        if cid else
        '<div style="padding:8px 12px;background:#1a1f2e;border-left:3px solid #5dade2;'
        'border-radius:4px;color:#5dade2;font-size:11px;">Structure-only chart unavailable.</div>'
    )
    le_type  = (last_event or {}).get('type')
    le_tier  = (last_event or {}).get('tier', 'Major')
    le_dir   = (last_event or {}).get('direction')
    le_ts    = (last_event or {}).get('ts')
    le_label = "—"
    if le_type and le_dir and le_ts:
        try:
            ts_short = le_ts[:10]
        except Exception:
            ts_short = str(le_ts)
        if le_type == 'BOS':
            ev_name = "Minor BOS" if le_tier == 'Minor' else "BOS"
        else:
            ev_name = f"{le_tier} CHoCH"
        le_label = f"{ev_name} {le_dir} on {ts_short}"
    caption = (
        f"No active OB. Last structure event: {le_label}. "
        f"Wait for price to return into the dealing range — could form a "
        f"continuation OB on the next leg, or a CHoCH if the range fails."
    )
    ob_attempt_line = _summarise_last_ob_attempt(ob_build_diagnostics)
    attempt_html = (
        f'<p style="font-size:11px;color:#888;line-height:1.5;'
        f'margin:0 0 6px 0;font-style:italic;">{ob_attempt_line}</p>'
        if ob_attempt_line else ""
    )
    return f"""
    <div style="margin-bottom:14px;padding:12px 14px;background:#13131f;
                border-left:3px solid #5dade2;border-radius:6px;">
      <div style="margin-bottom:6px;">
        <span style="font-size:13px;font-weight:bold;color:#dddddd;">{name}</span>
        <span style="background:#5dade2;color:#000;font-size:9px;padding:1px 5px;
                     border-radius:3px;font-weight:bold;margin-left:8px;">NO ACTIVE OB</span>
      </div>
      <p style="font-size:12px;color:#bbb;line-height:1.5;margin:0 0 6px 0;">{caption}</p>
      {attempt_html}
      {chart_html}
    </div>"""


def build_zone_changed_notice_html(prev_zone_id, new_zone_id, prev_zone_status, name):
    """One-line notice when the displayed zone for a pair changes between scans.

    Triggered when select_relevant_zone_for_pair picks a different zone than
    last scan (e.g. price moved closer to OB2 than OB1, or OB1 was invalidated).
    """
    if prev_zone_status == 'invalidated':
        msg = f"{prev_zone_id} was invalidated. Now showing {new_zone_id}."
    else:
        msg = (f"{prev_zone_id} is no longer the closest zone — price moved. "
               f"Showing {new_zone_id}. {prev_zone_id} stays alive in the slate "
               f"until it's touched or invalidated.")
    return (f'<div style="margin:6px 0 8px;padding:6px 10px;background:#2a2a1a;'
            f'border-left:3px solid #e67e22;border-radius:3px;font-size:11px;'
            f'color:#e67e22;"><b>{name}:</b> {msg}</div>')


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
    logging.info(f"[OB_BODY_RATIO] {name} zone {ob.get('zone_id','?')}: ob_body={ob['ob_body']:.{dp}f} median_leg={ob['median_leg_body']:.{dp}f} ratio={ratio}x")
    prompt = f"""You are a veteran SMC (Smart Money Concepts) prop trader writing a zone briefing for another experienced SMC trader.
Be direct. No fluff. No pleasantries. Four sentences only. One paragraph.

ZONE DATA — use these exact values, do not recalculate. ALL distances in PIPS.
- Pair: {name}
- Bias: {direction} | Structure event: {event_label}
- Zone width: {zone_pips} pips
- FVG: {fvg_status}
- Zone status: {ob.get('status', 'Pristine')}
- H1 ATR: {atr_display}
- {distance_brief}

WRITE EXACTLY FOUR SENTENCES IN THIS ORDER:
1. What structure event ({event_label}) created this zone and why institutional accumulation is likely here.
2. OB quality: assess whether pristine or tested means strength or caution.
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
    logging.info(f"[OB_BODY_RATIO] {name} zone {ob.get('zone_id','?')}: ob_body={ob['ob_body']:.{dp}f} median_leg={ob['median_leg_body']:.{dp}f} ratio={round(ob['ob_body']/ob['median_leg_body'],2):.2f}x (fallback narrative)")
    return (
        f"{event_label} confirmed the {ob['direction']} shift. "
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
                          invalidation_cards, inactive_pair_cards,
                          zone_change_notices, dropped_lines,
                          attachments, zone_count, ist_time):
    """
    Daily-slate digest sections:
      - Zone-change notices (when displayed zone for a pair flips)
      - NEW active zones (full cards with charts)
      - REFRESHED active zones (full cards with charts)
      - INVALIDATED (paragraph + greyed-OB chart per pair)
      - INACTIVE PAIRS (structure-only chart per pair)
      - DROPPED (one-line notes — book-keeping reasons only)
      - Legend (rendered once at the bottom)

    By design every pair gets exactly ONE chart per email: an active card,
    an invalidation card, or an inactive-pair card.
    """
    new_html         = "".join(new_zone_cards)
    unchanged_html   = "".join(unchanged_zone_cards)
    invalidated_html = "".join(invalidation_cards)
    inactive_html    = "".join(inactive_pair_cards)
    dropped_html     = "".join(dropped_lines)
    notices_html     = "".join(zone_change_notices)

    sections = ""
    if zone_change_notices:
        sections += f"""
        <div style="margin-bottom:14px;">{notices_html}</div>"""

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

    if invalidation_cards:
        sections += f"""
        <div style="margin-bottom:20px;">
          <h3 style="color:#e74c3c;font-size:12px;letter-spacing:1px;margin:0 0 10px 0;
                     text-transform:uppercase;border-bottom:1px solid #2a1a1a;padding-bottom:6px;">
            Invalidated Zones ({len(invalidation_cards)})
          </h3>
          {invalidated_html}
        </div>"""

    if inactive_pair_cards:
        sections += f"""
        <div style="margin-bottom:20px;">
          <h3 style="color:#5dade2;font-size:12px;letter-spacing:1px;margin:0 0 10px 0;
                     text-transform:uppercase;border-bottom:1px solid #1a2a3a;padding-bottom:6px;">
            Pairs Without an Active OB ({len(inactive_pair_cards)})
          </h3>
          {inactive_html}
        </div>"""

    if dropped_lines:
        sections += f"""
        <div style="margin-bottom:8px;">
          <h3 style="color:#888;font-size:11px;letter-spacing:1px;margin:0 0 8px 0;
                     text-transform:uppercase;border-bottom:1px solid #2a2a3e;padding-bottom:6px;">
            Other Drops ({len(dropped_lines)})
          </h3>
          {dropped_html}
        </div>"""

    # Single legend at the very bottom — replaces the per-card legend that
    # previously bloated every chart card.
    legend_html = _phase1_chart_legend_html('BOS', 'Major')

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
    <div style="margin-top:18px;">
      <h3 style="color:#666;font-size:10px;letter-spacing:1px;margin:0 0 6px 0;
                 text-transform:uppercase;border-bottom:1px solid #2a2a3e;padding-bottom:4px;">
        Legend
      </h3>
      {legend_html}
    </div>
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

    logging.info(
        f"Digest v2 sent: {zone_count} active zones, "
        f"{len(new_zone_cards)} new, {len(unchanged_zone_cards)} refreshed, "
        f"{len(invalidation_cards)} invalidated, "
        f"{len(inactive_pair_cards)} inactive-pair, "
        f"{len(dropped_lines)} other-drops."
    )
    print(
        f"Digest sent: {zone_count} active, {len(invalidation_cards)} invalidated, "
        f"{len(inactive_pair_cards)} inactive."
    )
                              
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
        "pairs": {name: {"next_id_counter": 0, "zones": [],
                          "last_displayed_zone_id": None}
                  for name in pair_names}
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
        "sweep_observed": fresh_zone.get("sweep_observed", {"exists": False}),
        # Phase 1 dealing range snapshot — single source of truth. Phase 2
        # consumes this directly and does NOT recompute.
        "dealing_range": fresh_zone.get("dealing_range", {"valid": False}),
        # Two-OB role classification — refreshed each scan as price moves.
        # 'primary' = OB1 (closest within 4xATR), 'alternative' = OB2
        # (best Pristine in 4-8xATR ring). Phase 2/3 only consume primary.
        "role": fresh_zone.get("role", "primary"),
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
    # Dealing range — refresh with the latest snapshot. Recomputed each scan
    # by Phase 1 against its own H1 frame; Phase 2 consumes whatever this
    # last write produced.
    slate_zone["dealing_range"] = fresh_zone.get(
        "dealing_range", slate_zone.get("dealing_range", {"valid": False})
    )
    # Tier / context refresh (Major / Minor distinction may change if a
    # later re-emission upgrades the structural classification).
    slate_zone["bos_tier"]      = fresh_zone.get("bos_tier", slate_zone.get("bos_tier", "Major"))
    slate_zone["broken_was_wall"] = fresh_zone.get("broken_was_wall",
                                                    slate_zone.get("broken_was_wall", False))
    slate_zone["reversal_pct"]  = fresh_zone.get("reversal_pct", slate_zone.get("reversal_pct"))
    slate_zone["bos_timestamp"] = fresh_zone.get("bos_timestamp", slate_zone.get("bos_timestamp"))
    # Two-OB role refreshed each scan — depends on current price relative
    # to the OB's proximal_line. As price moves, OB2 may re-classify to OB1.
    slate_zone["role"] = fresh_zone.get("role", slate_zone.get("role", "primary"))


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
      'out_of_proximity'
      'data_unavailable'
      'data_stale'
      None  -> zone should NOT be dropped; caller keeps it alive and logs.
    """
    # --- data_unavailable handled by caller before this is called ---

    # --- aged_out_of_window ---
    # Hard age cap on slate OBs. Retire any zone whose OB candle is older
    # than OB_MAX_AGE_DAYS. Independent of the H1 data window — an OB whose
    # candle has rolled out of the 150-candle fetch is still tracked here
    # for mitigation against the latest df slice, until the age cap fires.
    if slate_zone.get("ob_timestamp"):
        try:
            slate_ob_dt = datetime.fromisoformat(slate_zone["ob_timestamp"])
            if slate_ob_dt.tzinfo is not None:
                slate_ob_dt = slate_ob_dt.replace(tzinfo=None)
            now_utc = datetime.utcnow()
            age_days = (now_utc - slate_ob_dt).total_seconds() / 86400.0
            if age_days >= OB_MAX_AGE_DAYS:
                return "aged_out_of_window"
        except Exception:
            pass  # if we can't compare, fall through to other checks

    # --- out_of_proximity ---
    # Drop zones that have drifted beyond OB2_OUTER_LIMIT_ATR from current
    # price. Replaces the daily slate wipe's cleanup of distant zones.
    # Only fires when h1_atr is known and positive.
    if h1_atr and h1_atr > 0 and current_price is not None:
        proximal = slate_zone.get('proximal_line')
        if proximal is not None:
            if abs(current_price - proximal) > OB2_OUTER_LIMIT_ATR * h1_atr:
                return 'out_of_proximity'

    # --- mitigated_distal_break / mitigated_three_touches ---
    # Uses is_ob_mitigated_phase1 — single source of truth for Phase 1.
    # Rule: close beyond distal (strict, no ATR buffer). Wick alone never
    # invalidates. 3 wick touches at proximal = mitigated.
    if df is not None and len(df) > 0:
        try:
            distal = slate_zone['distal_line']
            proximal = slate_zone['proximal_line']
            direction = slate_zone['direction']

            # Find scan start index — from the candle AFTER the BOS candle.
            # Single source of truth: a touch is a return visit AFTER the OB
            # is confirmed by the break. Candles inside the displacement leg
            # (between OB and BOS) are part of the impulse, not tests.
            scan_start = 0
            if slate_zone.get("bos_timestamp"):
                try:
                    target_ts = datetime.fromisoformat(slate_zone["bos_timestamp"])
                    if target_ts.tzinfo is not None:
                        target_ts = target_ts.replace(tzinfo=None)
                    ts_col = df['Datetime'] if 'Datetime' in df.columns else pd.Series(df.index)
                    for k, t in enumerate(ts_col):
                        kt = t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t
                        if hasattr(kt, 'tzinfo') and kt.tzinfo is not None:
                            kt = kt.replace(tzinfo=None)
                        if kt >= target_ts:
                            scan_start = k + 1  # candles AFTER the BOS candle
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
    Phase 1 highlights ONE primary zone + optionally ONE alternative zone
    per pair, recomputed every scan.

    Selection (two-OB system):
      Primary headline (OB1):
        1. If price is INSIDE any active primary zone, that zone wins
           ('zone-in-progress'). Tie -> closest proximal.
        2. Else: among primaries (role=='primary'), pick smallest distance.
        3. Else (no primary exists this scan): fall back to the alternative
           (role=='alternative') closest to price. This handles the case
           where price ran past OB1 and only OB2 remains.

      Alternative (OB2) — returned alongside for chart rendering:
        - The single role=='alternative' zone (if any) in the same
          direction as the chosen primary. None if no alternative or if
          its direction conflicts.

    Returns (selected_zone, in_progress_flag, alt_zone)
            or (None, False, None) if no zones.
    """
    if not active_zones:
        return None, False, None

    primaries    = [z for z in active_zones if z.get('role', 'primary') == 'primary']
    alternatives = [z for z in active_zones if z.get('role') == 'alternative']

    inside = []
    for z in primaries:
        prox = z['proximal_line']
        dist = z['distal_line']
        z_lo = min(prox, dist)
        z_hi = max(prox, dist)
        if z_lo <= current_price <= z_hi:
            inside.append(z)

    selected = None
    in_progress = False
    if inside:
        selected = min(inside, key=lambda z: abs(current_price - z['proximal_line']))
        in_progress = True
    elif primaries:
        selected = min(primaries, key=lambda z: abs(current_price - z['proximal_line']))
    elif alternatives:
        # Fallback: only alternative zones exist this scan. Promote one to
        # the headline so the pair isn't silently empty.
        selected = min(alternatives, key=lambda z: abs(current_price - z['proximal_line']))

    if selected is None:
        return None, False, None

    # Pair the chosen primary with an alternative in the SAME direction.
    alt = None
    same_dir_alts = [a for a in alternatives
                     if a.get('direction') == selected.get('direction')
                     and a.get('zone_id') != selected.get('zone_id')]
    if same_dir_alts:
        alt = same_dir_alts[0]

    return selected, in_progress, alt


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

    # --- LOAD SLATE; UPDATE DATE IF NEW TRADING DAY (zones persist across days) ---
    slate = load_slate()
    if slate.get("slate_date") != today_str:
        slate["slate_date"] = today_str
        slate["slate_started_iso"] = ist_now.isoformat()
        print(f"  [SLATE] New trading day {today_str} — date updated, zones carried forward.")
    # Mark every existing zone is_new_this_scan = False at start of scan.
    for pname, pblock in slate.get("pairs", {}).items():
        for z in pblock.get("zones", []):
            z["is_new_this_scan"] = False

    # Ensure all configured pairs have a block (handles config additions mid-day).
    for name in pair_names:
        if name not in slate["pairs"]:
            slate["pairs"][name] = {"next_id_counter": 0, "zones": [],
                                     "last_displayed_zone_id": None}

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
        # TEMP DIAG — remove with OB build verification.
        ob_build_diag = result.get("ob_build_diagnostics", [])
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
            placeholder_diagnostic=placeholder_diag,
            ob_build_diagnostics=ob_build_diag,  # TEMP DIAG
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
        # Collect renderable items across all pairs.
        # Per requirement: every pair gets EXACTLY ONE chart — active card,
        # invalidation card, or inactive-pair structure card.
        new_zone_cards       = []
        unchanged_zone_cards = []
        invalidation_cards   = []
        inactive_pair_cards  = []
        zone_change_notices  = []
        dropped_lines        = []   # bookkeeping reasons only (supplanted/aged/data_*)
        all_zones_for_table  = []
        attachments          = []
        chart_counter        = 0

        # Reasons that warrant a full invalidation card (paragraph + chart).
        # Other reasons stay as one-line drops (bookkeeping noise).
        INVALIDATION_REASONS = {"mitigated_distal_break", "mitigated_three_touches"}

        def _attach_chart(b64_str, cid_str):
            """Attach a base64 PNG as inline MIME with the given content-id."""
            img_mime = MIMEImage(base64.b64decode(b64_str))
            img_mime.add_header("Content-ID", f"<{cid_str}>")
            img_mime.add_header("Content-Disposition", "inline",
                                filename=f"{cid_str}.png")
            attachments.append(img_mime)

        for pair_name in pair_names:
            dp = dp_map[pair_name]
            pblock = slate["pairs"].get(pair_name, {})
            zones_in_pair = pblock.get("zones", [])
            prev_displayed_id = pblock.get("last_displayed_zone_id")

            active_zones = [z for z in zones_in_pair if z["status"] == "active"]
            dropped_today = [z for z in zones_in_pair if z["status"] == "dropped"]
            current_price = pair_prices.get(pair_name)
            df = pair_dfs.get(pair_name)

            # Pair-level walls for table + chart context.
            # Reuse the structure state already loaded for the banner above —
            # avoids N+1 disk reads.
            pair_walls = _structure_state_for_banner.get(pair_name, {}) or {}
            # Enrich last_event with the broken-swing price by scanning the
            # event ring for the matching candle_ts. Needed by generate_h1_chart
            # to draw an off-window event level (horizontal line + "N candles
            # ago" label) when the event candle has rolled out of the visible
            # 130-candle window.
            _le_ts = pair_walls.get("last_event_ts")
            _le_bws = None
            if _le_ts:
                for _e in (pair_walls.get("events") or []):
                    if _e.get("candle_ts") == _le_ts:
                        _le_bws = _e.get("broken_swing_price")
                        break
            last_event = {
                "type":      pair_walls.get("last_event_type"),
                "tier":      pair_walls.get("last_event_tier"),
                "direction": pair_walls.get("last_event_direction"),
                "ts":        _le_ts,
                "broken_swing_price": _le_bws,
            }

            # Bookkeeping drops (structure_supplanted / aged_out / data_*) get
            # the one-liner regardless of whether we render an active card.
            for sz in dropped_today:
                if sz.get("drop_reason") not in INVALIDATION_REASONS:
                    dropped_lines.append(build_dropped_zone_line(sz, pair_name, dp))

            # Price-unavailable fallback (preserved from prior behaviour).
            if active_zones and current_price is None:
                for z in active_zones:
                    if z.get("current_price_at_scan"):
                        current_price = z["current_price_at_scan"]
                        break

            new_displayed_id = None
            displayed_status = None  # 'active' | 'invalidated' | 'inactive'

            if active_zones and current_price is not None:
                # --- Active path -------------------------------------------
                sz, in_progress, alt_sz = select_relevant_zone_for_pair(
                    active_zones, current_price, dp
                )
                if sz is None:
                    continue  # defensive — active_zones non-empty so shouldn't fire

                ob_for_render = _slate_zone_to_ob_shape(sz)
                alt_ob_for_render = _slate_zone_to_ob_shape(alt_sz) if alt_sz else None
                new_displayed_id = sz["zone_id"]
                displayed_status = "active"

                # Summary table row for the SELECTED zone.
                all_zones_for_table.append({
                    "name": pair_name, "zone_id": sz["zone_id"],
                    "direction": sz["direction"],
                    "proximal": sz["proximal_line"], "distal": sz["distal_line"],
                    "bos_tag": sz["bos_tag"],
                    "bos_tier": sz.get("bos_tier", "Major"),
                    "status": sz.get("status_label", "Pristine"),
                    "touches": sz.get("touches", 0),
                    "fvg_valid": sz["fvg"].get("exists", False),
                    "fvg_ghost": (not sz["fvg"].get("exists", False))
                                  and sz["fvg"].get("was_detected", False),
                    "fvg_mitigation": sz["fvg"].get("mitigation", "none"),
                    "sweep_observed": sz.get("sweep_observed", {"exists": False}),
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
                        walls=pair_walls,
                        alt_ob=alt_ob_for_render,
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
                    _attach_chart(chart_b64, cid)
                    chart_counter += 1

            elif dropped_today:
                # --- Invalidation path -------------------------------------
                # No active zones, but at least one was dropped today.
                # Pick the most recent invalidation (mitigation reason); fall
                # back to the most recent drop overall if none were invalidated.
                invalidations = [z for z in dropped_today
                                 if z.get("drop_reason") in INVALIDATION_REASONS]
                target = (invalidations or dropped_today)[-1]

                # Summary table: placeholder row (no active zone for the
                # confluence cells; the invalidation card carries the detail).
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

                if df is not None and target.get("drop_reason") in INVALIDATION_REASONS:
                    cid = f"chart_{pair_name}_{chart_counter}"
                    ob_for_chart = _slate_zone_to_ob_shape(target)
                    chart_b64 = generate_h1_chart(
                        df, ob_for_chart, dp, pair_name, ist_ts_full,
                        walls=pair_walls, is_invalidated=True
                    )
                    if chart_b64:
                        invalidation_cards.append(
                            build_invalidation_card_html(
                                target, pair_name, dp, cid, ist_ts_full
                            )
                        )
                        _attach_chart(chart_b64, cid)
                        chart_counter += 1
                    else:
                        invalidation_cards.append(
                            build_invalidation_card_html(
                                target, pair_name, dp, None, ist_ts_full
                            )
                        )
                # Else: no invalidation candidate (only bookkeeping drops),
                # which already went into dropped_lines above. Render an
                # inactive-pair card so the pair still gets a chart.
                else:
                    if df is not None:
                        cid = f"chart_{pair_name}_{chart_counter}"
                        chart_b64 = generate_h1_chart(
                            df, None, dp, pair_name, ist_ts_full,
                            walls=pair_walls, last_event=last_event
                        )
                        if chart_b64:
                            inactive_pair_cards.append(
                                build_inactive_pair_card_html(
                                    pair_name, dp, cid, ist_ts_full,
                                    pair_walls, last_event,
                                    ob_build_diagnostics=ob_build_diag,
                                )
                            )
                            _attach_chart(chart_b64, cid)
                            chart_counter += 1
                        else:
                            inactive_pair_cards.append(
                                build_inactive_pair_card_html(
                                    pair_name, dp, None, ist_ts_full,
                                    pair_walls, last_event,
                                    ob_build_diagnostics=ob_build_diag,
                                )
                            )

                displayed_status = "invalidated"
                # new_displayed_id stays None — there's no active zone to track.

            else:
                # --- Inactive-pair path ------------------------------------
                # No active zones, no recent drops. Pure structure-only view.
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

                if df is not None:
                    cid = f"chart_{pair_name}_{chart_counter}"
                    chart_b64 = generate_h1_chart(
                        df, None, dp, pair_name, ist_ts_full,
                        walls=pair_walls, last_event=last_event
                    )
                    if chart_b64:
                        inactive_pair_cards.append(
                            build_inactive_pair_card_html(
                                pair_name, dp, cid, ist_ts_full,
                                pair_walls, last_event,
                                ob_build_diagnostics=ob_build_diag,
                            )
                        )
                        _attach_chart(chart_b64, cid)
                        chart_counter += 1
                    else:
                        inactive_pair_cards.append(
                            build_inactive_pair_card_html(
                                pair_name, dp, None, ist_ts_full,
                                pair_walls, last_event,
                                ob_build_diagnostics=ob_build_diag,
                            )
                        )

                displayed_status = "inactive"

            # --- Zone-change notice ----------------------------------------
            # Tell the user when the displayed zone for a pair flipped between
            # scans. Two cases trigger:
            #   1. Active path: closest zone changed (price moved closer to a
            #      different OB). Old OB stays alive in slate.
            #   2. Invalidated path: the previously displayed OB was dropped
            #      with an invalidation reason this scan.
            if prev_displayed_id and prev_displayed_id != new_displayed_id:
                # Was the previous one invalidated this scan?
                prev_zone_obj = next(
                    (z for z in dropped_today if z.get("zone_id") == prev_displayed_id),
                    None
                )
                prev_was_invalidated = (
                    prev_zone_obj is not None
                    and prev_zone_obj.get("drop_reason") in INVALIDATION_REASONS
                )
                if displayed_status == "active":
                    zone_change_notices.append(build_zone_changed_notice_html(
                        prev_displayed_id, new_displayed_id,
                        'invalidated' if prev_was_invalidated else 'still_active',
                        pair_name
                    ))
                elif displayed_status == "invalidated" and prev_was_invalidated:
                    zone_change_notices.append(build_zone_changed_notice_html(
                        prev_displayed_id, "—", 'invalidated', pair_name
                    ))

            # Persist for next scan's comparison.
            pblock["last_displayed_zone_id"] = new_displayed_id

        if (all_zones_for_table or dropped_lines or
                invalidation_cards or inactive_pair_cards):
            summary_table = build_summary_table_html(all_zones_for_table, dp_map, pair_prices)
            try:
                # Prepend structure banner to the first section that has content.
                _banner_html = (
                    f'<div style="background:#1a1f2e;border-left:3px solid #5dade2;'
                    f'padding:8px 12px;margin:0 0 10px 0;color:#dddddd;'
                    f'font-family:monospace;font-size:12px;white-space:pre-wrap;">'
                    f'{structure_banner_text}</div>'
                )
                if new_zone_cards:
                    new_zone_cards = [_banner_html] + new_zone_cards
                elif unchanged_zone_cards:
                    unchanged_zone_cards = [_banner_html] + unchanged_zone_cards
                elif invalidation_cards:
                    invalidation_cards = [_banner_html] + invalidation_cards
                elif inactive_pair_cards:
                    inactive_pair_cards = [_banner_html] + inactive_pair_cards
                # Active count for the header line — number of pairs with a
                # live OB (not the placeholder rows).
                active_pair_count = sum(
                    1 for z in all_zones_for_table if not z.get("is_placeholder_row")
                )
                send_master_digest_v2(
                    summary_table, new_zone_cards, unchanged_zone_cards,
                    invalidation_cards, inactive_pair_cards,
                    zone_change_notices, dropped_lines,
                    attachments, active_pair_count, ist_time_str
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

    # Persist slate. Zones accumulate across days; only invalidation drops them.
    # slate_date is updated on day rollover (for Phase 2 freshness gate) without
    # wiping zones.
    save_slate(slate)
    print(f"Phase 1 complete at {ist_time_str} IST.")

    # Nightly archive: snapshot structure_state.json after the 23:00 IST scan.
    # Copy only — live state untouched. Idempotent within the evening window.
    _maybe_archive_state(ist_now)


# ---------------------------------------------------------------------------
# STATE ARCHIVAL + MANUAL RESET
# ---------------------------------------------------------------------------

STATE_ARCHIVE_DIR = os.path.join("state", "archives")


def _archive_state_file(label_date):
    """Copy live structure_state.json to state/archives/structure_state_<date>.json.

    label_date is a `date` object — used to name the archive. Returns the
    archive path on success, None when there is no live state file to copy.
    Overwrites an existing archive for the same date (idempotent).
    """
    live_path = dealing_range.STATE_PATH
    if not os.path.exists(live_path):
        return None
    os.makedirs(STATE_ARCHIVE_DIR, exist_ok=True)
    archive_path = os.path.join(
        STATE_ARCHIVE_DIR,
        f"structure_state_{label_date.isoformat()}.json"
    )
    shutil.copy2(live_path, archive_path)
    return archive_path


def _maybe_archive_state(ist_now):
    """Run an archive copy if this scan falls in the 23:00-23:59 IST window.

    Triggered automatically from run_radar(). Silently no-ops outside that
    window. Failures are logged but do NOT abort the scan — archival is
    secondary to live state integrity.
    """
    if ist_now.hour != 23:
        return
    try:
        archive_path = _archive_state_file(ist_now.date())
        if archive_path:
            print(f"  [ARCHIVE] structure_state snapshot -> {archive_path}")
    except Exception as e:
        logging.warning(f"Nightly state archive failed: {e}")
        print(f"  [ARCHIVE ERR] {e}")


def _manual_reset_state():
    """Manually wipe structure_state.json. NEVER called automatically.

    Behaviour:
      1. Archive the current live state to state/archives/ first (safety
         net, named with today's date + '_pre_reset' suffix).
      2. Delete the live state file.
      3. Exit. No scan runs.

    On next scan, dealing_range.load_state() returns {} and the system
    rebuilds walls/events from scratch on H1 data.
    """
    ist_now = get_ist_now()
    live_path = dealing_range.STATE_PATH
    print(f"[RESET] Manual state reset requested at {ist_now.isoformat()} IST.")
    if not os.path.exists(live_path):
        print(f"[RESET] No live state file at {live_path}. Nothing to reset.")
        return
    try:
        os.makedirs(STATE_ARCHIVE_DIR, exist_ok=True)
        backup_path = os.path.join(
            STATE_ARCHIVE_DIR,
            f"structure_state_{ist_now.date().isoformat()}_pre_reset.json"
        )
        shutil.copy2(live_path, backup_path)
        print(f"[RESET] Backup written: {backup_path}")
    except Exception as e:
        print(f"[RESET] Backup FAILED ({e}). Aborting — refusing to wipe without a backup.")
        return
    try:
        os.remove(live_path)
        print(f"[RESET] Wiped {live_path}. Next scan starts with empty state.")
    except Exception as e:
        print(f"[RESET] Wipe FAILED: {e}")


if __name__ == "__main__":
    if "--reset-state" in sys.argv:
        _manual_reset_state()
        sys.exit(0)
    run_radar()
