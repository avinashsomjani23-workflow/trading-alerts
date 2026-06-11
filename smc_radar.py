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
import h4_range  # H4-derived dealing range (built from H1, mapped onto H1)
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
# Two-OB system (single proximity window, no inner/outer ring, no trend gate).
# Both surfaced OBs use ONE gate: proximal within OB_PROXIMITY_ATR x H1 ATR.
# Selection: keep the nearest TWO OBs inside the window; on a tie/ranking,
# Pristine (touches == 0) is preferred over Tested. No direction/trend gating —
# buy and sell zones compete purely on distance + pristine-ness. Phase 2 fires
# on whichever OBs are within its tighter gate (see Phase2_Alert_Engine).
# OBs beyond OB_PROXIMITY_ATR are dropped at proximity_gate.
# ---------------------------------------------------------------------------
OB_PROXIMITY_ATR = 5.0   # Phase 1 surfacing window (both OBs)
OB_MAX_KEEP = 2          # hard cap on surfaced OBs

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
                        # NOTE: 'displacement_atr' removed — the v2 event ring
                        # (dealing_range._push_event) does not emit it; the read
                        # was always None. It only existed in pre-2026-05-25
                        # archived state. Don't reintroduce without the engine
                        # emitting it.
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
                              ob_build_diagnostics=None):
    """Construct a single per-pair scan record (snapshot of decisions)."""
    walls = walls or {}
    sv2 = walls.get('structure_v2') or {}
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
        # trend from v2 swing engine: 'bullish' | 'bearish' | None
        'trend': walls.get('trend'),
        # H1 structure detail for forensics
        'structure': {
            'state':            sv2.get('state'),
            'flip_unconfirmed': sv2.get('flip_unconfirmed'),
            'ranging':          sv2.get('ranging'),
            'defended':         sv2.get('defended'),
            'choch_flip_count': sv2.get('choch_flip_count'),
            'label':            sv2.get('label'),
        },
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
        # Live state-machine signal (may be a transient non-event like
        # 'CHoCH_FAILED'/'BOS_BIRTH'). Separate from last_event so the two
        # never collide. Forensics only.
        'last_structure_signal': walls.get('last_structure_signal'),
        # Full event ring snapshot. structure_state.json is overwritten every
        # scan, so without this the BOS/CHoCH history is unrecoverable from the
        # logs. Persisting it per scan lets any past event be reconstructed.
        'events': sv2.get('events', []),
        'active_zones': active,
        'dropped_this_scan': dropped_this_scan or [],
        'diagnostics': diagnostics or [],
        'h4_range': walls.get('h4_range'),
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


# ---------------------------------------------------------------------------
# ZONE REFERENCE ID
# ---------------------------------------------------------------------------

ZONE_ID_PREFIX = {
    "EURUSD": "EUR", "USDJPY": "JPY", "NZDUSD": "NZD",
    "USDCHF": "CHF", "NAS100": "NAS", "GOLD":   "XAU"
}


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
                # Normalize the datetime column to UTC at the fetch boundary so
                # detection, persisted swing ts, and the chart all speak one
                # timezone. yfinance returns GC=F (gold) in US/Eastern on some
                # fetches and UTC on others; without this, persisted swings and
                # freshly fetched candles can disagree on tz and markers misplace.
                _dt_col = 'Datetime' if 'Datetime' in tailed.columns else (
                    'Date' if 'Date' in tailed.columns else None)
                if _dt_col is not None:
                    _s = pd.to_datetime(tailed[_dt_col], utc=True)
                    tailed[_dt_col] = _s
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
# is_valid_ob_candle (20%-body anti-doji gate) — RESTORED 2026-06-10 by trader
# decision. The deletion was never approved. An order block must show
# directional intent; a near-doji opposing candle is indecision, not an
# institutional block. The walk-back continues to older opposing candles when
# the most recent one fails this check. The detection METHOD is unchanged
# (still the first opposing candle from the break) — this filter only skips
# dojis, exactly as before.


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
    # Human-readable structural-event label.
    # Two BOS tiers: 'BOS' (internal swing break) and 'Range' (H4 wall break).
    # NOTE: 'Major'/'Minor' do not exist in the v2 engine — only BOS / Range BOS
    # / CHoCH. Any 'Major' reaching here is a legacy default and is treated as
    # plain BOS.
    #
    # `bos_tag` arrives from two sources with different vocabularies:
    #   - OB zones use the bos_tag field: 'BOS' | 'CHoCH'.
    #   - The zoneless-pair structure marker passes last_bos.kind, which is
    #     'BOS' | 'BOS_BIRTH' (cold-start birth break) | 'CHoCH' |
    #     'CHoCH_FAILED' (attempted reversal that price reclaimed — trend
    #     resumed, so structurally a continuation, NOT a flip).
    # A BIRTH break is a BOS, not a trend flip. The previous `else -> CHoCH`
    # mislabeled BOTH 'BOS_BIRTH' (a BOS) and 'CHoCH_FAILED' (a continuation) as
    # CHoCH — a veteran would read either as a reversal that never happened.
    # Match every known tag explicitly so the label is always the honest event.
    if bos_tag in ('BOS', 'BOS_BIRTH'):
        return 'Range BOS' if bos_tier == 'Range' else 'BOS'
    if bos_tag == 'CHoCH':
        return 'CHoCH'
    if bos_tag == 'CHoCH_FAILED':
        # The flip was attempted and failed; the prior trend resumed. Name it
        # for what it is so the marker never implies a reversal occurred.
        return 'CHoCH failed (trend resumed)'
    # Unknown/None tag: do not silently call it a CHoCH. Fall back to the
    # neutral 'BOS' label rather than inventing a reversal.
    return 'BOS'


def _dir_arrow(direction):
    """Up/down glyph for a structural event direction. Empty when unknown.

    'bullish'/'up' -> ' ↑ (up)' ; 'bearish'/'down' -> ' ↓ (down)'.
    Lets every CHoCH/BOS mention read its direction without the chart, per the
    trader's requirement to never have to verify direction off the candles.
    """
    d = (direction or '').lower()
    if d in ('bullish', 'up'):
        return ' ↑ (up)'
    if d in ('bearish', 'down'):
        return ' ↓ (down)'
    return ''


def _event_label_dir(bos_tag, bos_tier, direction):
    """Event label WITH direction, e.g. 'BOS ↑ (up)', 'CHoCH ↓ (down)'."""
    return _event_label(bos_tag, bos_tier) + _dir_arrow(direction)


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
    truth for structural events (BOS, Range BOS, CHoCH — no Major/Minor). For each
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
                "active_zones": [],
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
    # logging has a value. Counter resets on any CHoCH (CHoCH flips the trend,
    # which restarts the BOS continuation chain).
    bos_seq_counter = 0
    last_choch_local_idx: Optional[int] = None  # idx in this df, or None
    active_obs = []

    # Candle ts of the MOST RECENT real structural event (BOS|CHoCH). Its OB is
    # OB1 (the last-event order block) and must NOT be dropped by the build-loop
    # proximity gate — OB1 surfaces regardless of distance (trader decision
    # 2026-06-10). All other events' OBs are still proximity-gated here.
    last_event_ts = None
    for _ev in reversed(events):
        if _ev.get('type') in ('BOS', 'CHoCH'):
            last_event_ts = _ev.get('candle_ts')
            break

    for ev_pos, ev in enumerate(events):
        ev_type = ev.get('type')           # 'BOS' | 'CHoCH'
        ev_tier = ev.get('tier')           # 'BOS' | 'Range' | 'Major'(legacy)
        ev_dir  = ev.get('direction')      # 'bullish' | 'bearish'

        # Most recent OPPOSING-direction structural event (a BOS or CHoCH
        # against this event's direction). Used to
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

        # BOS chain count: CHoCH resets, any BOS (plain or Range) increments.
        if ev_type == 'CHoCH':
            bos_seq_counter = 0
        elif ev_type == 'BOS' and ev_tier in ('BOS', 'Range', 'Major'):
            bos_seq_counter += 1

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

        # Walk back from the break candle through the impulse leg to find the OB.
        #
        # SMC definition (LOCKED 2026-06, decided with the trader): the order
        # block is the FIRST opposing candle walking back from the break — i.e.
        # the last opposing candle before the displacement leg. For a bullish
        # move that is the last down-candle before the up-impulse; for a bearish
        # move, the last up-candle before the down-impulse. Walking back from the
        # break, "last before the impulse" and "first from the break" are the
        # SAME candle on a clean leg — so the first opposing candle wins.
        #
        # Two rejections while walking back:
        #   - oversized guard: a candle wider than OB_MAX_RANGE_ATR_MULT * ATR
        #     is a news/volatility bar, not a clean institutional block.
        #   - body/doji guard (is_valid_ob_candle, 20% of range): a near-doji
        #     opposing candle is indecision, not an order block — skip it and
        #     keep walking to the next opposing candle.
        # Neither changes the detection METHOD (still the first qualifying
        # opposing candle from the break) — they only skip news bars and dojis.
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

        # Proximity gate. Drop any OB whose proximal sits beyond
        # OB_PROXIMITY_ATR from current price — EXCEPT the last-event OB (OB1),
        # which surfaces regardless of distance (trader decision 2026-06-10).
        # OB2 selection (closest, gated) happens post-dedupe in
        # _split_primary_alternative.
        is_last_event_ob = (last_event_ts is not None
                            and ev.get('candle_ts') == last_event_ts)
        if h1_atr_for_leg and h1_atr_for_leg > 0 and not is_last_event_ob:
            if abs(current_price_now - ob_proximal) > OB_PROXIMITY_ATR * h1_atr_for_leg:
                diag['drop_gate'] = 'proximity_gate'
                diag['drop_detail'] = {
                    'ob_proximal': ob_proximal,
                    'current_price': current_price_now,
                    'h1_atr': float(h1_atr_for_leg),
                    'distance_atr': abs(current_price_now - ob_proximal) / h1_atr_for_leg,
                    'limit_atr': OB_PROXIMITY_ATR,
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

        # Phase 1 sweep observation — snapshot semantics. Window is the OB's
        # OWN impulse leg [impulse_start_idx, ob_idx] (LOCKED 2026-06): the
        # validating sweep is the local liquidity run that fuels the displacement
        # which built this zone, so it can only live inside the leg. prior_event_idx
        # is still passed for signature compatibility but no longer sets the lower
        # bound — see observe_phase1_sweep's window comment for why the prior
        # opposing/same-direction event anchors were wrong on continuation BOS.
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

        # 'bos_tag' is the legacy field name for the structural event type
        # ('BOS' | 'CHoCH'). Kept for backwards compatibility with chart /
        # scoring / dedupe code.
        # 'bos_tier' carries the BOS sub-type: 'BOS' (internal swing break) or
        # 'Range' (H4 dealing-range wall break). No Major/Minor in v2.
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
            # Touches counted by is_ob_mitigated_phase1 are PROXIMAL wick-touches
            # (a distal wick is terminal — it kills the zone, it never accrues as
            # a "touch"). Label says "proximal" so the email is unambiguous about
            # which line was hit (trader request 2026-06-10).
            ob['status']  = 'Pristine' if touches == 0 else f'Tested ({touches}x proximal)'
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

    # Surface the nearest two OBs within OB_PROXIMITY_ATR of price, any
    # direction, pristine preferred on ties. No trend gating. Each kept OB
    # carries 'role' = 'primary' (nearest) | 'alternative' (next) for
    # downstream consumers (chart renders both; Phase 2 fires per its own gate).
    cur_price = float(C[-1])
    filtered = _split_primary_alternative(filtered, cur_price)

    # Strip the private dedupe hint before returning.
    for o in filtered:
        o.pop('_dedupe_thresh', None)
        o.pop('_diag_ref', None)  # TEMP DIAG — internal back-reference

    return {
        "current_price": cur_price,
        "active_zones": filtered,
        "ob_build_diagnostics": ob_build_diagnostics,  # TEMP DIAG
    }


def compute_pair_walls(df, pair_name=""):
    """Build the per-pair structure-state ('walls') dict from an H1 df.

    SINGLE SOURCE for assembling the dict every downstream consumer reads
    (trend, dealing-range walls, event ring, swings). Runs the H4 dealing range
    (h4_range) + the v2 structure engine (dealing_range.compute_structure) and
    packs the result. Pure read over `df`; never writes state. Inner H4 /
    structure failures degrade to a cold-start dict — never raises.

    Used by live Phase 1 (run_radar) and the backtest replay engine so both
    produce identical state. ceiling_price / floor_price come from the H4
    confirmed swing high / low; None until the H4 range is valid (cold start).
    """
    try:
        _h4 = h4_range.compute_h4_range(df)
    except Exception as _h4err:
        logging.warning(f"[h4_range] {pair_name} compute failed: {_h4err}")
        _h4 = {"valid": False, "source": "error"}

    try:
        _sv2 = dealing_range.compute_structure(df, _h4)
    except Exception as _sverr:
        logging.error(f"[structure] {pair_name} compute failed: {_sverr}")
        _sv2 = {"state": "error", "events": [], "trend": None, "swings": []}

    _h4_valid = _h4.get("valid") and _h4.get("ceiling") is not None and _h4.get("floor") is not None
    # last_event_* describes the most recent REAL structural event. ALL of its
    # fields (type / tier / direction / ts) are sourced from ONE place — the
    # event ring's last entry — so they can never describe two different events.
    #
    # The previous code split the sources: type/direction/ts came from
    # `last_bos` (the live state-machine scratchpad, which also holds transient
    # non-events like 'CHoCH_FAILED' and 'BOS_BIRTH') while tier came from the
    # ring. When last_bos held a CHoCH_FAILED, the four fields described two
    # unrelated events at different times/directions (verified live on USDJPY).
    #
    # The ring only ever contains real, fired events ('BOS'|'CHoCH', tier
    # 'BOS'|'Range'|'CHoCH'), so it is the single correct source. The live
    # state-machine scratchpad value (incl. transient CHoCH_FAILED / BOS_BIRTH)
    # is exposed SEPARATELY as `last_structure_signal` for forensics — never
    # collides with last_event.
    _events = _sv2.get("events", []) or []
    _last_ev = _events[-1] if _events else {}
    _last_bos = _sv2.get("last_bos") or {}
    return {
        "trend":                  _sv2.get("trend"),
        "ceiling_price":          float(_h4["ceiling"]) if _h4_valid else None,
        "ceiling_is_placeholder": bool(_h4.get("ceiling_broken", False)) if _h4_valid else True,
        "floor_price":            float(_h4["floor"])   if _h4_valid else None,
        "floor_is_placeholder":   bool(_h4.get("floor_broken", False))   if _h4_valid else True,
        "last_event_type":        _last_ev.get("type"),
        "last_event_tier":        _last_ev.get("tier"),
        "last_event_direction":   _last_ev.get("direction"),
        "last_event_ts":          _last_ev.get("candle_ts"),
        # Live state-machine signal (may be a transient non-event such as
        # 'CHoCH_FAILED' / 'BOS_BIRTH'). Kept separate so it never corrupts the
        # last_event_* fields above. Forensics only.
        "last_structure_signal":  {
            "kind":      _last_bos.get("kind"),
            "direction": _last_bos.get("direction"),
            "ts":        _last_bos.get("ts"),
        },
        "last_event_chop":        False,
        "last_scanned_ts":        None,
        "fallback_active":        not _h4_valid,
        "events":                 _sv2.get("events", []),
        "swings":                 _sv2.get("swings", []),
        "h4_range":               _h4,
        "structure_v2":           _sv2,
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
    Surface up to two OBs (trader spec, LOCKED 2026-06-10):

      OB1 = 'primary'     — the LAST-EVENT order block: the OB built from the
                            most recent structural event (highest bos_idx),
                            regardless of type (BOS or CHoCH). NOT gated by
                            proximity — it always surfaces while it exists, so
                            the vet can always see where the last structural OB
                            sits, even when price has run far from it.

      OB2 = 'alternative' — the CLOSEST-DISTANCE order block to current price,
                            across ALL directions (buy and sell compete; OB2
                            may be the opposite direction to OB1). Proximity-
                            GATED at OB_PROXIMITY_ATR; pristine breaks ties.
                            Chosen from the remaining OBs (OB1 excluded) so OB1
                            and OB2 are always distinct when both exist.

    Edge cases (all confirmed with trader):
      - Last event produced no OB (walk-back failed / out of window): OB1 empty,
        OB2 still surfaces. (no OB to show is the honest outcome)
      - OB1 and the closest OB are the SAME OB: show it once as OB1; no OB2.
      - No OB within the proximity window: OB2 empty (OB1 may still show).

    Distance (in ATR) is stamped on every OB for downstream display/logging.
    """
    # Stamp distance on all OBs (used for OB2 ranking + display).
    for ob in obs:
        atr = float(ob.get('h1_atr') or 0.0)
        ob['_distance_atr'] = (round(abs(cur_price - float(ob['proximal_line'])) / atr, 3)
                               if atr > 0 else None)

    # --- OB1: last-event OB = highest bos_idx (most recent event). Ungated. ---
    ob1 = None
    obs_with_idx = [o for o in obs if o.get('bos_idx') is not None]
    if obs_with_idx:
        ob1 = max(obs_with_idx, key=lambda o: int(o['bos_idx']))

    # --- OB2: closest OB within proximity, excluding OB1, any direction. ------
    in_window = []
    for ob in obs:
        if ob is ob1:
            continue
        d = ob.get('_distance_atr')
        if d is not None and d <= OB_PROXIMITY_ATR:
            in_window.append(ob)
    # Nearest first; pristine (touches == 0) breaks ties.
    in_window.sort(key=lambda o: (o['_distance_atr'], int(o.get('touches', 0))))
    ob2 = in_window[0] if in_window else None

    kept = []
    if ob1 is not None:
        ob1['role'] = 'primary'
        kept.append(ob1)
    if ob2 is not None:
        ob2['role'] = 'alternative'
        kept.append(ob2)
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
        # Same extension for Alt OB (OB2) — otherwise the alt zone band shows
        # without its candle outline when the alt OB is older than 130 bars.
        # Capped at the data limit (n_full=150); we never fetch more than that,
        # so we can't show what doesn't exist.
        if alt_ob is not None:
            alt_ob_abs_for_window = alt_ob.get('ob_idx')
            if alt_ob_abs_for_window is not None and alt_ob_abs_for_window < window_start:
                window_start = max(0, alt_ob_abs_for_window - 3)

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
                    _off_window_bws_for_y = _le_bws_y_f  # kept for off-window branch only

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
        # SAFETY: if the OB candle has rolled out of the visible window
        # (ob_plot_idx < 0 — possible on stale invalidated zones where the
        # df has shifted further than the OB age), DO NOT render the band.
        # Drawing the band with zone_x_start clamped to 0 while the OB candle
        # outline is suppressed produces an orphan box anchored to nothing.
        # Caller (slate render path) should have skipped this chart entirely
        # via resync; this guard is the second line of defence.
        ob_in_window = has_ob and ob_plot_idx is not None and 0 <= ob_plot_idx < n_plot
        if has_ob and ob_in_window:
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
            _btier = ob.get('bos_tier', 'BOS')
            if _btag == 'CHoCH':
                bos_color = '#ff9800'
            elif _btier == 'Range':
                bos_color = '#00897b'  # teal = Range BOS (H4 wall break)
            else:
                bos_color = '#00bcd4'  # cyan = plain BOS
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
            le_tier = last_event.get('tier', 'BOS')
            le_dir  = last_event.get('direction')
            le_ts   = last_event.get('ts')
            le_bws  = last_event.get('broken_swing_price')
            if le_type == 'CHoCH':
                ev_color = '#ff9800'
            elif le_tier == 'Range':
                ev_color = '#00897b'
            else:
                ev_color = '#00bcd4'
            le_idx = None
            if le_ts:
                # Match against full_df since window_start may chop history.
                # Compare UTC instants, not isoformat strings — the event ts may
                # be persisted in a different timezone than the chart's candles
                # (see smc_detector.ts_to_utc_instant).
                le_inst = smc_detector.ts_to_utc_instant(le_ts)
                if le_inst is not None:
                    ts_col = full_df['Datetime'] if 'Datetime' in full_df.columns else full_df.index
                    for k in range(len(full_df)):
                        raw = ts_col.iloc[k] if hasattr(ts_col, 'iloc') else ts_col[k]
                        if smc_detector.ts_to_utc_instant(raw) == le_inst:
                            le_idx = k
                            break
            if le_idx is not None and le_idx >= window_start:
                local_i = le_idx - window_start
                if 0 <= local_i < n_plot:
                    ax.axvline(x=local_i, color=ev_color, linewidth=0.8,
                               linestyle='--', alpha=0.45, zorder=2)
                    ax.text(local_i, y_max,
                            "  " + _event_label_dir(le_type, le_tier, le_dir),
                            color=ev_color, fontsize=8, fontweight='bold',
                            ha='left', va='top', zorder=7,
                            bbox=dict(facecolor='#131722', edgecolor='none',
                                      pad=1.5, alpha=0.78))
                    # Broken-swing level: horizontal dashed line so the vet
                    # can see exactly what price level the event broke.
                    if le_bws is not None:
                        try:
                            le_bws_f = float(le_bws)
                        except Exception:
                            le_bws_f = None
                        if le_bws_f is not None and _wall_in_view(le_bws_f):
                            ax.axhline(y=le_bws_f, color=ev_color,
                                       linewidth=0.8, linestyle='--',
                                       alpha=0.55, zorder=2)
                            ax.text(n_plot - 1, le_bws_f,
                                    f"  broken {le_bws_f:.5g}",
                                    color=ev_color, fontsize=7,
                                    ha='left', va='center', zorder=7,
                                    bbox=dict(facecolor='#131722',
                                              edgecolor='none',
                                              pad=1.0, alpha=0.78))
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
                    ev_name = _event_label_dir(le_type, le_tier, le_dir)
                    age_part = (f" · {candles_ago}c ago"
                                if candles_ago is not None else "")
                    ax.text(0, bws_f, f"  {ev_name}{age_part}",
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

        # --- Swing markers (triangles + current-setup broken-swing X) ---
        # SINGLE SOURCE: consume the persisted swing pool from dealing_range
        # state (walls['swings']) — the SAME lb-3+ATR swings that drove trend /
        # CHoCH / BOS / walls. The chart does NOT detect swings itself and does
        # NOT decide which break matters — dealing_range flags the one
        # current-setup break as `is_setup_break`. The X is drawn on exactly
        # that swing; every other swing (including older broken ones) renders as
        # its plain triangle. Positioned by ts.
        SWING_COLOR = '#d4a017'
        SETUP_BREAK_COLOR = '#ffffff'  # max contrast on dark bg + red/green candles; the X is a bold marker, never confused with the thin white price line
        marker_offset = (y_max - y_min) * 0.012
        swings_persisted = smc_detector.swings_for_chart(walls)
        if swings_persisted:
            # ts -> absolute idx over full_df, then shift to local plot x.
            ts_to_abs = smc_detector.build_ts_to_local_x(full_df)
            for s in swings_persisted:
                abs_i = ts_to_abs.get(smc_detector.ts_to_utc_instant(s.get('ts')))
                if abs_i is None:
                    continue  # ts not in this df at all
                xi = abs_i - window_start
                if not (0 <= xi < n_plot):
                    continue  # swing outside the plotted window
                # Anchor the marker to the candle's ACTUAL extreme at the
                # matched column — high swing -> H[xi], low swing -> L[xi] —
                # not the price stored in state. yfinance revises recent
                # intraday bars between the detection run and the render run;
                # the stored price then no longer equals the bar at that
                # column, which floats the X off its candle. The column comes
                # from the timestamp; the price must come from the same bar.
                is_high = (s['type'] == 'high')
                candle_price = float(H[xi]) if is_high else float(L[xi])
                if s.get('is_setup_break'):
                    ax.scatter([xi], [candle_price], marker='x',
                               s=70, color=SETUP_BREAK_COLOR, linewidths=2.0, zorder=8)
                elif is_high:
                    ax.scatter([xi], [candle_price + marker_offset], marker='v',
                               s=42, color=SWING_COLOR, edgecolors=SWING_COLOR,
                               linewidths=1.0, zorder=6)
                else:
                    ax.scatter([xi], [candle_price - marker_offset], marker='^',
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

        # --- Edge tags: proximal, distal, BOS/CHoCH ---
        # Default to left edge. If a label would visually collide with
        # an FVG box / ghost FVG / sweep marker / break-candle outline /
        # swing marker that also sits near the left edge, flip THAT label
        # to the right edge. OB zone band and OB candle outline are NOT
        # counted as obstacles (the band by design covers the left labels,
        # and the OB candle is structurally tied to the labels).
        edge_labels = []  # (price, text, color)
        if has_ob:
            zone_label_color = '#888888' if is_invalidated else '#bb8fce'
            edge_labels.append((proximal, f"{proximal:.{dp}f}", zone_label_color))
            edge_labels.append((distal,   f"{distal:.{dp}f}",   zone_label_color))
            if bos_price is not None:
                bos_label_color = '#888888' if is_invalidated else bos_color
                edge_labels.append((bos_price, f"{bos_price:.{dp}f}", bos_label_color))

        # Build obstacle rectangles in DATA coords. Each entry: (x_lo, x_hi, y_lo, y_hi).
        # We check overlap against a label's bbox at the left edge: x in [-1, LEFT_BAND_W].
        # Approx label width in data x-units. Labels are ~7 chars at fontsize=10;
        # 6 x-units is a conservative band covering the rendered text width.
        LEFT_BAND_W = 6.0
        RIGHT_BAND_LO = (n_plot + RIGHT_MARGIN) - LEFT_BAND_W - 1
        RIGHT_BAND_HI = (n_plot + RIGHT_MARGIN)
        # Label vertical half-height in price units — approximate using 1.2%
        # of the y-axis span. Same value used for all labels (uniform font).
        label_half_h = max((y_max - y_min) * 0.012, 1e-9)

        obstacles = []
        # FVG (active) — drawn at (mid_local - 0.6, fb) width 3.0, height ft-fb.
        if has_ob and not is_invalidated and ob['fvg']['exists'] and ob['fvg']['c1_idx'] is not None:
            _ft = float(ob['fvg']['fvg_top'])
            _fb = float(ob['fvg']['fvg_bottom'])
            _mid = ob['fvg']['c1_idx'] + 1 - window_start
            if 0 <= _mid < n_plot:
                obstacles.append((_mid - 0.6, _mid - 0.6 + 3.0, _fb, _ft))
        # FVG (ghost) — same geometry as active.
        if has_ob and not is_invalidated and (not ob['fvg']['exists']) \
                and ob['fvg'].get('was_detected') and ob['fvg'].get('ghost_c1_idx') is not None:
            _ft = float(ob['fvg']['ghost_top'])
            _fb = float(ob['fvg']['ghost_bottom'])
            _mid = ob['fvg']['ghost_c1_idx'] + 1 - window_start
            if 0 <= _mid < n_plot:
                obstacles.append((_mid - 0.6, _mid - 0.6 + 3.0, _fb, _ft))
        # Sweep — dotted level line from swept_swing back to sweep_idx, plus
        # star + label at the wick tip. Treat the level line span as obstacle.
        _sw = (ob.get('sweep_observed') or {}) if (has_ob and not is_invalidated) else {}
        if _sw.get('exists') and _sw.get('sweep_idx') is not None and _sw.get('price') is not None:
            _sw_local = _sw['sweep_idx'] - window_start
            if 0 <= _sw_local < n_plot:
                _swept = _sw.get('swept_swing_idx')
                _x_lo = max(0, _swept - window_start) if _swept is not None else max(0, _sw_local - 6)
                _x_hi = _sw_local
                if _x_lo > _x_hi:
                    _x_lo, _x_hi = _x_hi, _x_lo
                _sw_p = float(_sw['price'])
                obstacles.append((_x_lo, _x_hi + 0.5, _sw_p - label_half_h, _sw_p + label_half_h))
        # Break candle outline — spans br_start..br_end at ±0.5 around column.
        if has_ob:
            try:
                _br_start, _br_end = smc_detector.compute_h1_break_candle_span(full_df, ob, None)
            except Exception:
                _br_start = _br_end = None
            if _br_start is not None and _br_end is not None:
                for _abs_i in range(_br_start, _br_end + 1):
                    if _abs_i < window_start:
                        continue
                    _li = _abs_i - window_start
                    if 0 <= _li < n_plot:
                        _ch = float(H[_li]); _cl = float(L[_li])
                        obstacles.append((_li - 0.5, _li + 0.5, _cl, _ch))
        # Alt OB candle outline — only when alt has its own candle drawn in window.
        if (alt_ob is not None) and has_ob:
            _alt_idx = alt_ob.get('ob_idx')
            if _alt_idx is not None:
                _ali = _alt_idx - window_start
                if 0 <= _ali < n_plot:
                    _ah = float(H[_ali]); _al = float(L[_ali])
                    obstacles.append((_ali - 0.5, _ali + 0.5, _al, _ah))
        # Alt OB right-edge tag — sits at right edge, alt_zone midline.
        if (alt_ob is not None) and has_ob:
            _ap = float(alt_ob['proximal_line']); _ad = float(alt_ob['distal_line'])
            _amid = (_ap + _ad) / 2.0
            obstacles.append((RIGHT_BAND_LO, RIGHT_BAND_HI,
                              _amid - label_half_h * 2, _amid + label_half_h * 2))

        def _band_overlap(price, x_lo, x_hi):
            """True if a label centered at price within [x_lo, x_hi] overlaps any obstacle."""
            y_lo = price - label_half_h
            y_hi = price + label_half_h
            for (ox_lo, ox_hi, oy_lo, oy_hi) in obstacles:
                if x_hi < ox_lo or x_lo > ox_hi:
                    continue
                if y_hi < oy_lo or y_lo > oy_hi:
                    continue
                return True
            return False

        # Decide per-label side. Default left; if left collides AND right is clear, flip.
        # If both collide, keep left (least-bad fallback, never worse than today).
        left_labels = []
        right_labels = []
        for (price, text, color) in edge_labels:
            left_blocked = _band_overlap(price, -1, -1 + LEFT_BAND_W)
            if left_blocked:
                right_blocked = _band_overlap(price, RIGHT_BAND_LO, RIGHT_BAND_HI)
                if not right_blocked:
                    right_labels.append((price, text, color))
                    continue
            left_labels.append((price, text, color))

        # Render left group.
        left_stacked = smc_detector.stack_labels(left_labels, pair_conf_shim)
        for adj_price, text, color in left_stacked:
            ax.text(-1, adj_price, text, color=color, fontsize=10, va='center',
                    ha='left', fontweight='bold', zorder=7,
                    bbox=dict(facecolor='#131722', edgecolor='none', pad=1.5, alpha=0.78))
        # Render right group.
        right_stacked = smc_detector.stack_labels(right_labels, pair_conf_shim)
        for adj_price, text, color in right_stacked:
            ax.text(n_plot + RIGHT_MARGIN - 1, adj_price, text, color=color,
                    fontsize=10, va='center', ha='right', fontweight='bold', zorder=7,
                    bbox=dict(facecolor='#131722', edgecolor='none', pad=1.5, alpha=0.78))

        # --- Mid-chart tags: current price, DR walls, EQ ---
        mid_labels = [(current_price, f"{current_price:.{dp}f}", '#ffffff')]
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
            event_label = _event_label(ob.get('bos_tag', 'BOS'), ob.get('bos_tier', 'BOS'))
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


# ---------------------------------------------------------------------------
# EMAIL ASSEMBLY
# ---------------------------------------------------------------------------


def _walls_icon_cell(walls):
    """Compact icon for the Walls column in the phone summary table.

    ✓ = anchored wall, ○ = tentative (placeholder, pending swing confirmation),
    — — = cold-start fallback. The hollow circle reads as "not yet confirmed"
    rather than the old ⚠ which looked like an error. Tooltip via title= holds
    the long-form label so a long-press on phones reveals it.
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
        return "<span style='color:#e67e22;' title='Both walls tentative (pending swing confirmation)'>&#9675;&#9675;</span>"
    if cph:
        return "<span style='color:#e67e22;' title='Ceiling tentative, floor anchored'>&#9675;&#10003;</span>"
    return "<span style='color:#e67e22;' title='Floor tentative, ceiling anchored'>&#10003;&#9675;</span>"


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
            # Bias cell: show H1 trend even with no zone, from the same
            # structure_v2.state source the zone card uses (▲ up / ▼ down /
            # — undefined). Lets the user read direction on zoneless pairs.
            _ph_state = ((walls.get('structure_v2') or {}).get('state'))
            if _ph_state == 'up':
                ph_bias = "<span style='color:#27ae60;'>&#9650;</span>"
            elif _ph_state == 'down':
                ph_bias = "<span style='color:#e74c3c;'>&#9660;</span>"
            else:
                ph_bias = "<span style='color:#666;'>&mdash;</span>"
            rows += f"""
        <tr style="background:transparent;border-bottom:1px solid #2a2a3e;opacity:0.55;">
          <td style="padding:6px 6px;font-weight:bold;color:#aaa;font-size:12px;white-space:nowrap;">{name}</td>
          <td style="padding:6px 4px;text-align:center;font-size:13px;">{ph_bias}</td>
          <td style="padding:6px 4px;text-align:center;color:#666;font-size:10px;">&mdash;</td>
          <td style="padding:6px 4px;text-align:center;color:#666;font-size:13px;">&mdash;</td>
          <td style="padding:6px 4px;text-align:center;color:#666;font-size:13px;">&mdash;</td>
          <td style="padding:6px 4px;text-align:center;color:#666;font-size:13px;">&mdash;</td>
          <td style="padding:6px 4px;text-align:right;color:#666;font-size:11px;">&mdash;</td>
          <td style="padding:6px 4px;text-align:center;font-size:11px;">{walls_cell}</td>
        </tr>"""
            continue

        # --- Dist cell: ATR-distance to proximal — primary tradeability rank.
        # 'in zone' when price is inside; — when ATR or price unavailable.
        _atr = z.get('h1_atr', 0.0) or 0.0
        _cp  = z.get('current_price')
        if z.get('in_progress'):
            dist_cell = "<span style='color:#f1c40f;font-weight:bold;'>in zone</span>"
        elif _atr > 0 and _cp is not None and z.get('proximal') is not None:
            _d = abs(_cp - z['proximal']) / _atr
            _dc = '#27ae60' if _d <= 1.5 else ('#f1c40f' if _d <= 3 else '#888')
            dist_cell = f"<span style='color:{_dc};'>{_d:.1f}&#215;</span>"
        else:
            dist_cell = "<span style='color:#666;'>&mdash;</span>"

        # --- Bias glyph = H1 TREND (structure_v2.state), NOT the zone's own
        # direction. Bias and Trend are the same concept and MUST share one
        # source — the card's Trend field reads structure_v2.state too, and the
        # placeholder rows above do as well. Reading z['direction'] here (the
        # supply/demand side of the zone) made a bearish zone in an uptrend show
        # ▼ while the card showed Up — the contradiction the trader flagged. The
        # zone's own side is already conveyed by the Event chip + zone card.
        _bias_state = (walls.get('structure_v2') or {}).get('state')
        if _bias_state == 'up':
            bias_glyph, bias_col = '&#9650;', '#27ae60'
        elif _bias_state == 'down':
            bias_glyph, bias_col = '&#9660;', '#e74c3c'
        else:
            bias_glyph, bias_col = '&mdash;', '#666'

        # --- Event chip: BOS / Range BOS / CHoCH.
        _tier = z.get('bos_tier', 'BOS')
        if z['bos_tag'] == 'BOS':
            if _tier == 'Range':
                ev_color, ev_text = '#00897b', 'RangeBOS'
            else:
                ev_color, ev_text = '#00bcd4', 'BOS'
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
          <td style="padding:6px 4px;text-align:right;font-size:11px;">{dist_cell}</td>
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
            <th style="padding:6px 4px;text-align:right;color:#666;font-size:9px;font-weight:normal;text-transform:uppercase;letter-spacing:0.4px;">Dist</th>
            <th style="padding:6px 4px;text-align:center;color:#666;font-size:9px;font-weight:normal;text-transform:uppercase;letter-spacing:0.4px;">Walls</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

def _phase1_chart_legend_html(bos_tag="BOS", bos_tier=None):
    """Colour-code legend rendered ONCE at the bottom of the Phase 1 digest.

    Args kept for backwards-compat call sites, but ignored — the global legend
    surfaces ALL structure-event colours at once instead of switching based on
    a single zone's event.

    The v2 engine has ONE structural tier — there is no Major/Minor. The only
    event types are BOS (internal swing break), Range BOS (H4 dealing-range wall
    break) and CHoCH (trend flip). Colours below match the chart exactly:
    plain BOS #00bcd4, Range BOS #00897b, CHoCH #ff9800.
    """
    items = [
        ('#bb8fce', 'Primary zone band (OB1 — closest to price) / greyed when invalidated'),
        ('#d7bde2', 'OB candle outline — greyed when invalidated'),
        ('#bb8fce', 'Alternative zone band (OB2 — best Pristine OB further out, dashed + faded)'),
        ('#2ecc71', 'FVG pristine (displacement)'),
        ('#f1c40f', 'FVG partial (proximal touched)'),
        ('#888888', 'FVG mitigated (ghost) / Invalidated zone'),
        ('#00bcd4', 'BOS break candle / level (internal swing break)'),
        ('#00897b', 'Range BOS break candle / level (H4 dealing-range wall break)'),
        ('#ff9800', 'CHoCH break candle / level (trend flip)'),
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

    return (
        f"Sweep: <span style='color:{tier_color};'>"
        f"{tier_emoji} {tier.title()}</span>"
    )


def build_active_zone_card_html(sz, name, dp, narrative, cid, ist_timestamp,
                                 current_price=None, in_progress=False,
                                 walls=None):
    """
    Render an active zone card. Used for both NEW and UNCHANGED active zones.
    NEW badge is rendered inline based on sz['is_new_this_scan'].
    Distance to proximal shown in pips. 'In zone' label when in_progress.

    `walls`: the per-pair structure dict (compute_pair_walls output) for THIS
    scan. The slate zone `sz` persisted in active_obs.json carries no `walls`
    key, so the H1-trend read below must take it from the live scan, not from
    the zone — otherwise Trend always renders "Undefined" while the summary
    table (which is fed pair_walls directly) shows the real bias, a direct
    contradiction. Falls back to sz['walls'] for any caller that still embeds it.
    """
    direction  = "Bullish (Demand)" if sz['direction'] == 'bullish' else "Bearish (Supply)"
    dir_color  = '#27ae60' if sz['direction'] == 'bullish' else '#e74c3c'
    status_label = sz.get('status_label', 'Pristine')
    stat_color = '#27ae60' if 'Pristine' in status_label else '#e67e22'

    fvg = sz.get('fvg', {})
    mit = fvg.get('mitigation', 'none')
    if fvg.get('exists') and mit == 'partial':
        fvg_line = "FVG: <span style='color:#f1c40f;'>◐ Partial</span>"
    elif fvg.get('exists'):
        fvg_line = "FVG: <span style='color:#27ae60;'>✓ Pristine</span>"
    elif fvg.get('was_detected'):
        fvg_line = "FVG: <span style='color:#888;'>✗ Mitigated</span>"
    else:
        fvg_line = "FVG: <span style='color:#888;'>None</span>"

    pip_unit  = _pip_unit(dp)
    zone_pips = round(abs(sz['proximal_line'] - sz['distal_line']) / pip_unit, 1)
    h1_atr_val = sz.get('h1_atr', 0.0)
    # Width rendered as ATR multiple — raw pips aren't comparable across
    # instruments (a 4-pip NZD zone is tight, a 190-pip NAS zone is too).
    # Pips kept secondary so the absolute size is still on the card.
    if h1_atr_val > 0:
        width_atr = abs(sz['proximal_line'] - sz['distal_line']) / h1_atr_val
        width_text = f"{width_atr:.1f}× ATR ({zone_pips}p)"
    else:
        width_text = f"{zone_pips} pips"

    # Dist as ATR multiple (primary decision driver) + pips secondary. Sorting
    # of cards keys off the same ratio (see send path).
    if current_price is not None:
        if in_progress:
            dist_text = "<span style='color:#f1c40f;font-weight:bold;'>in zone</span>"
        else:
            dist_pips_val = round(abs(current_price - sz['proximal_line']) / pip_unit, 1)
            if h1_atr_val > 0:
                dist_atr = abs(current_price - sz['proximal_line']) / h1_atr_val
                dist_text = f"{dist_atr:.1f}× ATR ({dist_pips_val}p)"
            else:
                dist_text = f"{dist_pips_val} pips"
    else:
        dist_text = "—"

    # H1 trend state — canonical three-state field from structure_v2
    # (up | down | undefined). Distinct from the OB's own bias. ranging /
    # transition / unconfirmed sub-flags are intentionally NOT shown here.
    _sv2 = ((walls or sz.get('walls')) or {}).get('structure_v2') or {}
    _state = _sv2.get('state')
    if _state == 'up':
        trend_text, trend_col = 'Up', '#27ae60'
    elif _state == 'down':
        trend_text, trend_col = 'Down', '#e74c3c'
    else:
        trend_text, trend_col = 'Undefined', '#888'

    # Sweep status for the chip row. The standalone "Sweep: None" line below
    # the chips is dropped — absence is shown here and in the table dash.
    _sw = sz.get('sweep_observed') or {}
    if _sw.get('exists'):
        _sw_tier = (_sw.get('tier') or 'weak').lower()
        _sw_col = {'textbook': '#27ae60', 'decent': '#f1c40f', 'weak': '#888'}.get(_sw_tier, '#888')
        sweep_chip = f"<span style='color:{_sw_col};'>★ {_sw_tier.title()}</span>"
    else:
        sweep_chip = "<span style='color:#888;'>None</span>"

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
    # not per-card — frees up vertical space on phones. The standalone
    # "Sweep: None" line is dropped — sweep state now lives in the chip row.

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
          <b>Width</b> {width_text}
        </span>
        <span style="font-size:11px;color:{stat_color};">
          <b>Status</b> {status_label}
        </span>
        <span style="font-size:11px;color:#aaa;">
          <b style="color:#bb8fce;">Dist</b> {dist_text}
        </span>
        <span style="font-size:11px;color:#aaa;">
          <b style="color:#bb8fce;">Trend</b> <span style="color:{trend_col};">{trend_text}</span>
        </span>
        <span style="font-size:11px;color:#aaa;">
          <b>Sweep</b> {sweep_chip}
        </span>
        <span style="font-size:11px;color:#aaa;">{fvg_line}</span>
      </div>
      <p style="font-size:12px;color:#bbb;line-height:1.6;margin:0 0 12px 0;
                border-left:3px solid #2a2a3e;padding-left:10px;">
        {narrative}
      </p>
      {chart_html}
    </div>"""


def build_dropped_zone_line(sz, name, dp):
    """One-line note for a dropped zone."""
    reason_map = {
        "mitigated_distal_break": "invalidated — a wick hit the DISTAL line (zone dead)",
        "mitigated_three_touches": "mitigated — the PROXIMAL line was wicked 3 times (exhausted)",
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
        "bos_tier":  sz.get("bos_tier", "BOS"),
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
    "mitigated_distal_break":  "a wick pierced the DISTAL line — zone is dead",
    "mitigated_three_touches": "the PROXIMAL line was wicked three times — zone is mitigated (exhausted)",
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
    "post_build_mitigation":   "OB built then died (wick to distal, or proximal wicked 3 times)",
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
    detail = latest.get('drop_detail') or {}
    try:
        ts_short = ev_ts[:10]
    except Exception:
        ts_short = str(ev_ts)

    # Build a gate-specific reason from drop_detail so the vet can see WHY,
    # not just WHICH gate fired. For no_qualifying_ob_candle, the leg had
    # candidates but each was killed by size or doji filters — that's a real
    # signal about volatility, not an opaque "no candle qualified".
    if gate == 'no_qualifying_ob_candle':
        opposing = int(detail.get('opposing_candles_in_leg') or 0)
        oversized = int(detail.get('oversized_rejected') or 0)
        doji = int(detail.get('doji_rejected') or 0)
        leg_len = int(detail.get('leg_len') or 0)
        if opposing == 0:
            reason = (
                f"impulse leg had no opposing candle ({leg_len}-candle leg, "
                f"all in trend direction)"
            )
        else:
            parts = []
            if oversized:
                parts.append(f"{oversized} rejected (body > 2x H1 ATR)")
            if doji:
                parts.append(f"{doji} rejected (near-doji)")
            sub = "; ".join(parts) if parts else "all failed body/wick checks"
            reason = (
                f"{opposing} opposing candle(s) found in {leg_len}-candle leg — {sub}"
            )
    elif gate == 'post_build_mitigation':
        touches = detail.get('touches')
        reason = (
            f"OB built then mitigated ({touches} touches)"
            if touches is not None else "OB built then mitigated"
        )
    elif gate == 'degenerate_leg':
        leg_len = detail.get('leg_len')
        reason = (
            f"impulse leg too short ({leg_len} candle(s))"
            if leg_len is not None else "impulse leg too short to form an OB"
        )
    else:
        reason = _OB_DROP_GATE_LABELS.get(gate, gate.replace('_', ' '))

    return (
        f"Last OB attempt: {ev_type} {ev_dir} on {ts_short} — dropped: {reason}."
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
    le_tier  = (last_event or {}).get('tier', 'BOS')
    le_dir   = (last_event or {}).get('direction')
    le_ts    = (last_event or {}).get('ts')
    le_label = "—"
    if le_type and le_dir and le_ts:
        try:
            ts_short = le_ts[:10]
        except Exception:
            ts_short = str(le_ts)
        if le_type == 'BOS':
            ev_name = "Range BOS" if le_tier == 'Range' else "BOS"
        else:
            ev_name = "CHoCH"
        le_label = f"{ev_name} {le_dir} on {ts_short}"

    # Facts only — dealing range bounds + H1 trend. No generic guidance about
    # what "might" happen next; the OB-attempt line below tells the vet what
    # the system tried and rejected.
    w = walls or {}
    ceil_px = w.get('ceiling_price')
    floor_px = w.get('floor_price')
    # H1 trend comes from the single structure engine (structure_v2 on state):
    # up / down / transition, with `ranging` as a flag inside up/down. The old
    # wall `trend` field is NOT used.
    range_parts = []
    if ceil_px is not None and floor_px is not None:
        range_parts.append(
            f"DR {floor_px:.{dp}f} → {ceil_px:.{dp}f} "
            f"({ceil_px - floor_px:.{dp}f} pts)"
        )
    sv2 = (walls or {}).get('structure_v2') or {}
    if sv2.get('state'):
        state = sv2['state']
        if state in ('up', 'down'):
            if sv2.get('flip_unconfirmed') and sv2.get('prior_trend'):
                trend_txt = f"{state.upper()} (CHoCH from {sv2['prior_trend'].upper()}, unconfirmed)"
            else:
                trend_txt = state.upper() + (" (ranging)" if sv2.get('ranging') else "")
        else:
            trend_txt = "undefined"
        range_parts.append(f"H1 trend: {trend_txt}")
    range_line = " &middot; ".join(range_parts) if range_parts else ""
    caption = (
        f"No active OB. Last structure event: {le_label}."
        + (f" {range_line}." if range_line else "")
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
    Generates the LLM zone narrative with H1 ATR injected into the prompt.
    Calls Gemini with the ATR-aware prompt; falls back to the local
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

    event_label = _event_label(ob.get('bos_tag', 'BOS'), ob.get('bos_tier', 'BOS'))
    # Qualitative proximity band only — the prompt forbids quoting numbers, so
    # feed the model a word, not a pip figure it might parrot back.
    if in_zone:
        distance_brief = "price is INSIDE the zone (mitigation in progress)"
    elif atr_pips and dist_pips <= 1.5 * atr_pips:
        distance_brief = "price is arriving at the zone"
    elif atr_pips and dist_pips <= 3 * atr_pips:
        distance_brief = "price is nearing the zone"
    else:
        distance_brief = "price is still distant from the zone"
    logging.info(f"[OB_BODY_RATIO] {name} zone {ob.get('zone_id','?')}: ob_body={ob['ob_body']:.{dp}f} median_leg={ob['median_leg_body']:.{dp}f} ratio={ratio}x")
    prompt = f"""You are a veteran SMC (Smart Money Concepts) prop trader writing a zone briefing for another experienced SMC trader.
Be direct. No fluff. No pleasantries. Three sentences only. One paragraph.

The reader already sees the numbers (zone width, distance, H1 ATR, FVG state, OB status) on the card above your text. DO NOT restate them. Give judgment the numbers cannot — the structural read and what would confirm or kill the setup.

ZONE CONTEXT (for your reasoning only — do not quote these figures back):
- Pair: {name}
- Bias: {direction} | Structure event: {event_label}
- FVG: {fvg_status}
- Zone status: {ob.get('status', 'Pristine')}
- {distance_brief}

WRITE EXACTLY THREE SENTENCES IN THIS ORDER:
1. The structural read: what the {event_label} tells you about where this zone sits (with trend / counter-trend, premium/discount) and why smart money may defend it.
2. Conviction: weigh OB freshness and FVG displacement together — is this a clean setup or one to distrust, and why.
3. What to watch: the single condition that would confirm the trade, and the one that would invalidate the zone.

STRICT OUTPUT RULES:
- Plain text only
- No bullet points, no headers, no markdown, no bold, no numbers, no pip figures, no distances
- Three sentences, one paragraph
- Judgment only — never restate the card's metrics"""

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
    # Judgment-only fallback. The card chips already carry width, distance,
    # H1 ATR, FVG state and OB status — this paragraph must NOT restate them.
    # It gives the structural read + what confirms / kills the zone.
    z_lo = min(ob['proximal_line'], ob['distal_line'])
    z_hi = max(ob['proximal_line'], ob['distal_line'])
    in_zone = z_lo <= current_price <= z_hi
    is_tested = 'Pristine' not in ob.get('status', 'Pristine')

    if ob['fvg'].get('exists'):
        mit = ob['fvg'].get('mitigation', 'pristine')
        if mit == 'partial':
            conviction = ("OB carries partial-FVG displacement — proximal already "
                          "tagged, so conviction is moderate; the distal half is "
                          "where the unfilled imbalance still sits.")
        else:
            conviction = ("OB is backed by a clean displacement FVG — the strongest "
                          "form of this setup.")
    elif ob['fvg'].get('was_detected'):
        conviction = ("The FVG has been filled, so the zone rests on the OB alone — "
                      "treat conviction as reduced.")
    else:
        conviction = ("No FVG formed, so the zone rests on the OB alone — demand "
                      "tighter confirmation before trusting it.")
    if is_tested:
        conviction += " The OB has already been tested, so expect a weaker reaction than a pristine block."

    event_label = _event_label(ob.get('bos_tag', 'BOS'), ob.get('bos_tier', 'BOS'))
    side = "demand" if ob['direction'] == 'bullish' else "supply"
    # A CHoCH flips structure; a BOS (incl. Range BOS / birth) continues it.
    # Using the right verb keeps the narrative honest — a BOS card must never
    # read "flipped structure". `event_label` is the single source of truth for
    # the event type here.
    structural_verb = "flipped structure" if event_label == 'CHoCH' else "continued structure"
    structural = (f"{event_label} {structural_verb} {ob['direction']}, leaving this "
                  f"{side} zone as the origin smart money may defend on the retrace.")
    if in_zone:
        watch = ("Price is mitigating it now — watch the reaction candle for a "
                 "rejection that holds, and stand down if it closes through the distal.")
    else:
        watch = ("Wait for price to reach the zone and reject; a clean close beyond "
                 "the distal invalidates it.")
    logging.info(f"[OB_BODY_RATIO] {name} zone {ob.get('zone_id','?')}: ob_body={ob['ob_body']:.{dp}f} median_leg={ob['median_leg_body']:.{dp}f} ratio={round(ob['ob_body']/ob['median_leg_body'],2):.2f}x (fallback narrative)")
    return f"{structural} {conviction} {watch}"


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
    legend_html = _phase1_chart_legend_html('BOS', 'BOS')

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
        "bos_tier": fresh_zone.get("bos_tier", "BOS"),
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


def _df_ts_iso(df, idx):
    """Return the ISO timestamp at row idx in df, or None if out of range."""
    try:
        idx = int(idx)
        if idx < 0 or idx >= len(df):
            return None
        if 'Datetime' in df.columns:
            raw = df['Datetime'].iloc[idx]
        elif 'Date' in df.columns:
            raw = df['Date'].iloc[idx]
        else:
            raw = df.index[idx]
        return raw.isoformat() if hasattr(raw, 'isoformat') else str(raw)
    except Exception:
        return None


def _df_idx_from_iso(df, ts_iso):
    """Return the row idx in df whose timestamp matches ts_iso, or None."""
    if not ts_iso:
        return None
    for k in range(len(df)):
        if _df_ts_iso(df, k) == ts_iso:
            return k
    return None


def resync_slate_zone_indices(slate_zone, df, pair_name=""):
    """
    Re-anchor a slate zone's index fields to the current df frame using stored
    timestamps. Each scan adds new candles and the frame shifts left; without
    this, ob_idx / bos_idx / fvg.c1_idx / impulse_start_idx drift and point
    to wrong candles. Charts then draw boxes on unrelated candles and Phase 2/3
    walk the wrong slices for break-candle / sweep / FVG mitigation logic.

    Source of truth: ob_timestamp + bos_timestamp (stored as ISO strings at
    build time). impulse_start_idx and fvg.c1_idx/c3_idx are derived by
    preserving their original offset from ob_idx (the leg's relative spacing
    is invariant across scans).

    Returns True on success, False if timestamps could not be located in df
    (e.g., yfinance dropped the candle). Caller decides what to do on False —
    typically log and skip rendering rather than draw with stale indices.
    """
    ob_ts  = slate_zone.get("ob_timestamp")
    bos_ts = slate_zone.get("bos_timestamp")
    new_ob_idx  = _df_idx_from_iso(df, ob_ts)
    new_bos_idx = _df_idx_from_iso(df, bos_ts)
    if new_ob_idx is None or new_bos_idx is None:
        logging.warning(
            f"[resync] {pair_name} zone {slate_zone.get('zone_id')} — "
            f"could not locate ob_ts={ob_ts} or bos_ts={bos_ts} in current df. "
            f"Leaving indices stale."
        )
        return False

    old_ob_idx = slate_zone.get("ob_idx")
    if old_ob_idx is None:
        # Nothing to derive offset from for impulse_start; set defensively.
        slate_zone["ob_idx"]  = new_ob_idx
        slate_zone["bos_idx"] = new_bos_idx
        return True

    delta = new_ob_idx - int(old_ob_idx)
    if delta == 0:
        return True  # already aligned

    # Apply the shift uniformly. Relative spacing is preserved.
    slate_zone["ob_idx"]  = new_ob_idx
    slate_zone["bos_idx"] = new_bos_idx
    old_impulse = slate_zone.get("impulse_start_idx")
    if old_impulse is not None:
        shifted = int(old_impulse) + delta
        slate_zone["impulse_start_idx"] = shifted if 0 <= shifted < len(df) else None
    # FVG indices live under slate_zone["fvg"].
    fvg = slate_zone.get("fvg") or {}
    for key in ("c1_idx", "c3_idx", "ghost_c1_idx", "ghost_c3_idx", "mitigated_at_idx"):
        v = fvg.get(key)
        if v is None:
            continue
        try:
            shifted = int(v) + delta
        except (TypeError, ValueError):
            continue
        fvg[key] = shifted if 0 <= shifted < len(df) else None
    slate_zone["fvg"] = fvg
    # Sweep observation idx fields — these are absolute df indices too.
    sw = slate_zone.get("sweep_observed") or {}
    for key in ("sweep_idx", "swept_swing_idx"):
        v = sw.get(key)
        if v is None:
            continue
        try:
            shifted = int(v) + delta
        except (TypeError, ValueError):
            continue
        sw[key] = shifted if 0 <= shifted < len(df) else None
    slate_zone["sweep_observed"] = sw
    logging.info(
        f"[resync] {pair_name} zone {slate_zone.get('zone_id')} — "
        f"shifted indices by {delta} (ob_idx {old_ob_idx} -> {new_ob_idx})."
    )
    return True


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
    # Tier / context refresh (BOS vs Range BOS may change if a later
    # re-emission reclassifies the break against the H4 wall).
    slate_zone["bos_tier"]      = fresh_zone.get("bos_tier", slate_zone.get("bos_tier", "BOS"))
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
    # Drop zones that have drifted beyond OB_PROXIMITY_ATR from current price.
    # Replaces the daily slate wipe's cleanup of distant zones.
    # Only fires when h1_atr is known and positive.
    #
    # NO OB1 exemption here — and that is correct. This function is ONLY called
    # for slate zones that did NOT match a fresh OB this scan (see run_radar
    # reconcile: matched zones `continue` before this). The CURRENT last-event
    # OB (OB1) is exempt from the build-loop proximity gate, so it is always
    # produced fresh and therefore always MATCHES — it never reaches here.
    # Any zone that does reach here is, by definition, NOT this scan's OB1
    # (at most a FORMER OB1, now superseded). Exempting on the slate's stale
    # `role` would make a superseded far OB1 immortal (role is never refreshed
    # on unmatched zones). So a former OB1 that drifted away is dropped here,
    # exactly as it should be.
    if h1_atr and h1_atr > 0 and current_price is not None:
        proximal = slate_zone.get('proximal_line')
        if proximal is not None:
            if abs(current_price - proximal) > OB_PROXIMITY_ATR * h1_atr:
                return 'out_of_proximity'

    # --- mitigated_distal_break / mitigated_three_touches ---
    # Uses is_ob_mitigated_phase1 — single source of truth for Phase 1.
    # Rule (WICK-based, no ATR buffer): a wick to/through the distal line kills
    # the zone; 3 wick touches at the proximal line = mitigated. (Earlier docs
    # here said "close beyond distal" — stale; the function is wick-based.)
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

    # Return the alternative zone (OB2). Direction may differ from OB1 —
    # no type filter applied. OB2 was already picked globally as the best
    # pristine zone in the outer ring by _split_primary_alternative.
    alt = None
    valid_alts = [a for a in alternatives
                  if a.get('zone_id') != selected.get('zone_id')]
    if valid_alts:
        alt = valid_alts[0]

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

        # --- STRUCTURE + DEALING RANGE UPDATE (v2 engine, single source of truth) ---
        # compute_structure is the sole trend/CHoCH/BOS engine. It emits the
        # event ring that detect_smc_radar uses to build OBs, and the trend
        # that Phase 2 reads via compute_bos_sequence_count. No legacy wall
        # engine. Dealing range walls come from H4 swing highs/lows only.
        new_walls = {}
        placeholder_diag = None
        try:
            structure_state_all = dealing_range.load_state()

            # Single source: H4 range + v2 structure engine + walls assembly,
            # extracted to compute_pair_walls so live Phase 1 and the backtest
            # replay engine build identical state.
            new_walls = compute_pair_walls(df, name)

            structure_state_all[name] = new_walls
            dealing_range.save_state(structure_state_all)

            if new_walls["fallback_active"]:
                logging.warning(f"[{name}] H4 range not yet valid — ceiling/floor absent (cold start).")
                print(f"  [STRUCTURE] {name}: H4 range cold-starting, walls absent.")
            else:
                sv2_label = (new_walls.get("structure_v2") or {}).get("label", "")
                print(f"  [STRUCTURE] {name}: {sv2_label}  events={len(new_walls['events'])}")

        except Exception as _dr_err:
            logging.error(f"[{name}] structure update failed: {_dr_err}")
            print(f"  [STRUCTURE ERR] {name}: {_dr_err}")

        result        = detect_smc_radar(df, pair_type=ptype,
                                          events=new_walls.get('events', []),
                                          walls=new_walls, pair_name=name)
        current_price = result["current_price"]
        fresh_zones   = result["active_zones"]
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
                # Re-anchor indices to the current df frame so chart rendering
                # and Phase 2/3 consumers don't walk stale positions. Without
                # this, every new H1 candle silently shifts ob_idx away from
                # the real OB candle while the zone sits unrefreshed in slate.
                try:
                    resync_slate_zone_indices(sz, df, pair_name=name)
                except Exception as _resync_err:
                    logging.warning(f"[resync] {name} {sz.get('zone_id')} failed: {_resync_err}")
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

            # H1 trend / CHoCH / transition is surfaced from structure_v2,
            # computed once in the wall-update block and carried on pair_walls.
            # The banner reads it directly off walls — no recompute here.

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
                    "bos_tier": sz.get("bos_tier", "BOS"),
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
                    "h1_atr": sz.get("h1_atr", 0.0),
                    "current_price": current_price,
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
                    in_progress=in_progress,
                    walls=pair_walls
                )

                # Sort key: ATR-distance to proximal, closest first. In-zone
                # cards rank first (key -1). Cards without ATR/price sink last.
                _h1_atr_for_sort = sz.get("h1_atr", 0.0) or 0.0
                if in_progress:
                    sort_key = -1.0
                elif _h1_atr_for_sort > 0:
                    sort_key = abs(current_price - sz["proximal_line"]) / _h1_atr_for_sort
                else:
                    sort_key = float("inf")

                if sz.get("is_new_this_scan", False):
                    new_zone_cards.append((sort_key, card_html))
                else:
                    unchanged_zone_cards.append((sort_key, card_html))

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
                    # Re-anchor the invalidated zone's df indices to the
                    # current frame BEFORE rendering. Slate zones can sit
                    # with stale ob_idx after enough scans (each new H1 candle
                    # shifts the frame left; the drop path doesn't refresh
                    # the indices the way the HOLD path does). Stale ob_idx
                    # causes the chart to draw a grey box clamped to x=0 with
                    # NO OB candle outline (the outline guard `0 <= ob_plot_idx
                    # < n_plot` skips negative indices). Result: an orphan
                    # box anchored to nothing — visually misleading.
                    try:
                        resync_slate_zone_indices(target, df, pair_name=pair_name)
                    except Exception as _resync_err:
                        logging.warning(
                            f"[resync] {pair_name} invalidated zone "
                            f"{target.get('zone_id')} failed: {_resync_err}"
                        )
                    ob_for_chart = _slate_zone_to_ob_shape(target)
                    # If the OB candle is no longer locatable in the visible
                    # df window (rolled out of yfinance's rolling fetch), do
                    # NOT render a chart with phantom indices. Skip the chart
                    # entirely; the invalidation card still publishes the
                    # textual reason. Better to lose one chart than to print
                    # an orphan box.
                    ob_ts_check = target.get("ob_timestamp")
                    ob_idx_resolved = (
                        _df_idx_from_iso(df, ob_ts_check)
                        if ob_ts_check else None
                    )
                    if ob_idx_resolved is None:
                        logging.warning(
                            f"[invalidated-chart] {pair_name} "
                            f"{target.get('zone_id')} skipped — OB candle "
                            f"ts={ob_ts_check} no longer in df window."
                        )
                        chart_b64 = None
                    else:
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
            # Active-zone cards were collected as (atr_distance, html) tuples.
            # Sort closest-first so the actionable zone sits at the top, then
            # drop the keys back to plain HTML strings before the banner prepend.
            new_zone_cards = [h for _, h in sorted(new_zone_cards, key=lambda t: t[0])]
            unchanged_zone_cards = [h for _, h in sorted(unchanged_zone_cards, key=lambda t: t[0])]
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
