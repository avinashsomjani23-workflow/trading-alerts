import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone


# ============================================================================
# PHASE SCOPE MAP — read this before editing any function or constant below.
# ============================================================================
# This module is the historical shared detection library. The phase
# boundaries are convention only — there is no language-level isolation.
# Touching a function without checking its scope label can silently break
# a phase that imports it. Every public function and module-level constant
# carries one of the scope labels defined here.
#
# Scope labels:
#
#   PHASE 1 ONLY      — called only by smc_radar.py (Phase 1 scout).
#                       Safe to edit without checking Phase 2/3.
#
#   PHASE 2 ONLY      — called only by Phase2_Alert_Engine.py.
#                       Safe to edit without checking Phase 1/3.
#
#   PHASE 3 ONLY      — called only by phase3_engine.py.
#                       Safe to edit without checking Phase 1/2.
#
#   SHARED P1+P2      — called by Phase 1 and Phase 2 only. Behavior change
#                       ripples to both. Phase 3 unaffected.
#
#   SHARED P2+P3      — called by Phase 2 and Phase 3 only. Phase 1 unaffected.
#
#   SHARED P1+P2+P3   — called by all three phases. Highest blast radius.
#                       Detection uniformity across phases depends on this
#                       single definition staying consistent. Any signature
#                       or behavior change must be verified at every callsite.
#
#   INTERNAL          — private helper (leading underscore) or non-public
#                       function used by other functions in this file only.
#                       No phase imports it directly.
#
#   UNUSED            — defined but no callsite anywhere in the codebase.
#                       Safe to delete; kept for now to avoid churn risk.
#
# Editing protocol:
#   1. Check the scope label on the thing you are editing.
#   2. Verify the label is still accurate — grep the symbol across
#      smc_radar.py, Phase2_Alert_Engine.py, phase3_engine.py.
#      If the label is stale, fix it as part of the same edit.
#   3. For SHARED labels, walk every callsite and confirm the change is
#      intended for all callers. Different phases use different timeframes
#      (Phase 1 = H1, Phase 2 = M15, Phase 3 = M5) but call the same
#      function — make sure the change is timeframe-agnostic.
#   4. Never copy a SHARED function into a phase file to "isolate" it.
#      That destroys detection uniformity. Phases are isolated by which
#      functions they call, not by duplication.
# ============================================================================


# INTERNAL — decimal places helper, used inside this file for price rounding.
def _dp(pair_conf):
    return pair_conf.get("decimal_places", 5)


# SHARED P1+P2+P3 — ATR computation. Called by Phase 1 (H1), Phase 2 (H1+M15),
# Phase 3 (M5). Timeframe is determined by the df passed in. Behavior change
# affects FVG noise floor, OB filtering, and proximity calculations everywhere.
#
# Memoization (added 2026-05-23 to cut backtest runtime):
# The function is pure for a given OHLC slice. During backtest replay it gets
# called ~480x per scan, each time iterating the full history. We cache on a
# content fingerprint (first_ts, last_ts, len, last OHLC values, period).
# Collision across instruments is effectively impossible because the OHLC
# anchors differ. If anything in the cache path raises, we set a module flag
# that the email layer reads — backtest email will be tagged [ATR-CACHE-FAIL]
# so the user knows to fall back. Cache miss path always recomputes from
# scratch, so a buggy cache cannot silently corrupt results — at worst it
# fails loudly via the flag.
_ATR_CACHE = {}
_ATR_CACHE_ERROR = None  # str message if memoization path ever raised

def _atr_cache_status():
    """Returns None if cache healthy, else error message. Read by email layer."""
    return _ATR_CACHE_ERROR

def _atr_compute_raw(df, period):
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    C = df['Close'].values.astype(float)
    trs = []
    for i in range(1, len(C)):
        tr = max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))
        trs.append(tr)
    if len(trs) < period:
        return None
    return float(np.mean(trs[-period:]))

def compute_atr(df, period=14):
    """ATR computation used across phases. Memoized on slice fingerprint."""
    global _ATR_CACHE_ERROR
    if df is None or len(df) < period + 1:
        return None
    try:
        n = len(df)
        first_ts = df.index[0]
        last_ts = df.index[-1]
        last_close = float(df['Close'].iat[-1])
        last_high = float(df['High'].iat[-1])
        last_low = float(df['Low'].iat[-1])
        key = (first_ts, last_ts, n, period, last_close, last_high, last_low)
        cached = _ATR_CACHE.get(key)
        if cached is not None:
            return cached
        value = _atr_compute_raw(df, period)
        if value is not None:
            # Bound cache: 4096 entries is more than enough for any single
            # backtest run (240 bars x 6 pairs x ~3 callsites = ~4300 worst case).
            if len(_ATR_CACHE) > 4096:
                _ATR_CACHE.clear()
            _ATR_CACHE[key] = value
        return value
    except Exception as e:
        # Cache path broke — record loudly and fall back to raw compute.
        # This NEVER returns wrong values; worst case is no caching this turn.
        _ATR_CACHE_ERROR = f"{type(e).__name__}: {e}"
        try:
            return _atr_compute_raw(df, period)
        except Exception as e2:
            _ATR_CACHE_ERROR = f"raw compute also failed: {type(e2).__name__}: {e2}"
            return None


# ---------------------------------------------------------------------------
# PHASE 1 ONLY — Dealing range lookback per pair type (in H1 candles).
# Used internally by get_dealing_range(). Trader-set: Forex 5 trading days,
# Index 3 days, Gold 5 days. 1 trading day = 24 H1 candles. Weekend candles
# already excluded by yfinance.
# ---------------------------------------------------------------------------
DEALING_RANGE_LOOKBACK_H1 = {
    "forex": 120,
    "index": 72,
    "commodity": 120
}

# ---------------------------------------------------------------------------
# SHARED P1+P2+P3 — FVG noise floor multipliers (pair-type aware, TF-agnostic).
# Multiplier applies to whatever TF ATR the caller passes in:
#   Phase 1 -> H1 ATR  |  Phase 2 -> M15 ATR  |  Phase 3 -> M5 ATR
# Any change here shifts FVG detection thresholds across all three phases.
# ---------------------------------------------------------------------------
FVG_NOISE_FLOOR_MULT = {
    "forex":     0.08,
    "index":     0.15,
    "commodity": 0.12
}
# ---------------------------------------------------------------------------
# FVG search window — number of candles past the OB candle to scan for a
# 3-candle FVG. The FVG is the displacement signature that confirms the OB.
# Veteran SMC: displacement happens during OB→BOS leg AND can extend 1 candle
# past BOS. The actual window used is [ob_idx, bos_idx+1], i.e. the leg
# itself. These constants are SOFT CAPS to prevent runaway windows on slow
# grinds.
#
# H1 = 10 candles soft cap (was 7; widened because the leg-anchored window
#      self-adjusts and 7 was occasionally tighter than the leg).
# M15 = 40 candles. Matches H1's 10-candle window in time-span (10 H1 = 40
#       M15). Phase 2 uses a fixed window — Phase 2 doesn't have a BOS index
#       on M15.
# ---------------------------------------------------------------------------
# PHASE 1 ONLY — H1 FVG window soft cap.
FVG_WINDOW_H1_CANDLES  = 10
# PHASE 2 ONLY — M15 FVG window soft cap.
FVG_WINDOW_M15_CANDLES = 40

# PHASE 1 ONLY — OB candidate range cap. Reject candles where (high - low) > N x ATR.
# Filters volatility spikes (news bars) from being picked as OBs.
# Adopted from LuxAlgo SMC ob_coord. Not the inverse of the removed 1.5x
# median minimum — that one rejected small candles; this rejects oversized.
OB_MAX_RANGE_ATR_MULT = 2.0

# Touch re-arm distance (ATR units). A proximal touch is counted once per
# APPROACH: after a touch, price must pull back away from the proximal by this
# distance before another touch can register. Without this, a multi-hour
# consolidation resting on the proximal racked up 3 per-bar "touches" and
# falsely mitigated a zone that a vet calls ONE test. Self-scaling on ATR.
OB_TOUCH_REARM_ATR = 0.5

# Minimum swing leg size in ATR(14) units, applied AFTER lookback-3 geometric
# swing detection. SINGLE SOURCE OF TRUTH lives in dealing_range (the lowest
# layer); re-exported here so existing references keep working and there is one
# value, one filter implementation. See dealing_range.MIN_LEG_ATR_MULT and
# dealing_range._filter_swings_by_leg_atr. Reasoning / benchmark catalogue in
# Benchmarking.md section 8 (1.5x cuts 6-20% of swings across pairs).
import dealing_range as _dr_const
MIN_LEG_ATR_MULT = _dr_const.MIN_LEG_ATR_MULT

# ---------------------------------------------------------------------------
# Liquidity sweep — pair-aware tolerance for "equal highs / equal lows" detection.
# Two prior swings (out of last 3 same-type swings near the swept swing) are
# considered "equal" if they sit within this multiple of the TF's own ATR.
# Forex: tighter — pairs respect levels precisely.
# Index/commodity: looser — wider noise around levels.
# Widened (2026-05) from 0.15/0.25 -> 0.30/0.40 — vet feedback that a sub-2
# pip spread between forex lows still reads as "equal" in practice. Also
# reused as the context-tag proximity band (prior-day H/L, session H/L).
# ---------------------------------------------------------------------------
# INTERNAL — consumed by sweep scoring helpers below.
SWEEP_EQUAL_LEVEL_TOLERANCE_ATR = {
    "forex":     0.30,
    "index":     0.40,
    "commodity": 0.40
}

# INTERNAL — wick pierce minimum — the sweep wick must extend BEYOND the swept level by
# at least this multiple of the TF ATR. Decoupled from equal-level tolerance:
# they measure different things (level identity vs. clear pierce). A below-
# noise poke is not a sweep.
SWEEP_WICK_PIERCE_MIN_ATR = {
    "forex":     0.05,
    "index":     0.08,
    "commodity": 0.08
}

# Sweep window: how many candles to extend the search BEFORE the impulse-leg
# start, to catch the stop-run that TURNED the market (the classic
# sweep -> base -> impulse sequence, where the sweep precedes the leg by a candle
# or two). Kept small and ALWAYS floored at the prior structural event so it can
# never reach an earlier leg's unrelated liquidity (the 2026-06 over-reach that
# the impulse-leg-only lock fixed). Survivorship + active-target filters still
# discard anything that isn't the local fueling sweep.
SWEEP_LOOKBACK_BEFORE_IMPULSE = 3

# INTERNAL — round-number grid used by Phase 1 context tagging. Tight tolerance —
# being "near a round number" must mean within a few pips, not 30. yfinance wick
# revisions are typically sub-pip, well inside this band; weekly_review must
# log tag accuracy to confirm the band holds up.
ROUND_NUMBER_GRID = {
    "forex":     0.0050,   # 50 pips on 5-dp pairs
    "forex_jpy": 0.50,     # 50 pips on 3-dp JPY pairs
    "index":     50.0,     # 50 points on NAS100
    "commodity": 5.0       # $5 on Gold
}
ROUND_NUMBER_TOLERANCE = {
    "forex":     0.0005,   # 5 pips
    "forex_jpy": 0.05,     # 5 pips (JPY)
    "index":     5.0,      # 5 points
    "commodity": 0.50      # $0.50
}

# INTERNAL — session windows in UTC. Asia wraps midnight (22 prev-day -> 07 same-day).
# Phase 1 displays IST equivalents in the email but computes in UTC.
# Asia    ~03:30-12:30 IST | London ~12:30-17:30 IST | NY ~17:30-22:30 IST
SESSION_WINDOWS_UTC = {
    "asia":   (22, 7),
    "london": (7, 12),
    "ny":     (12, 17)
}

# INTERNAL — per-pair session tags shown in Phase 1 sweep badge (handoff table).
PAIR_SESSION_TAGS = {
    "EURUSD": ["asia", "london"],
    "USDJPY": ["asia", "london"],
    "NZDUSD": ["asia"],
    "USDCHF": ["london"],
    "GOLD":   ["london", "ny"],
    "NAS100": ["ny"]
}

# PHASE 1 ONLY — Sweep observation window for the Phase 1 display-only badge.
# Same 72 trading-hour rule applied during OB construction. Kept as a separate
# constant so Phase 1 observation logic is decoupled from Phase 2 grading.
PHASE1_SWEEP_OBS_TRADING_HOURS = 72

# INTERNAL — scoring caps for the sweep score. Consumed by run_scorecard()
# (Phase 2 only). Sums to 3.0 by construction. Vet's allocation: Presence
# carries the trade (1.5), Rejection confirms it (1.0), Equal Levels is the
# "nice to have" bigger-pool bonus (0.5).
SWEEP_SCORE_BASE_MAX        = 1.5   # presence (wick + close-back, bias-aligned, within recency)
SWEEP_SCORE_EQUAL_LEVEL_MAX = 0.5   # 0 / 0.25 / 0.5 for 0 / 1 / 2 prior matches in last 3 swings
SWEEP_SCORE_REJECTION_MAX   = 1.0   # 0 / 0.33 / 0.66 / 1.0 for wick:body ratio < 1 / 1-2 / 2-3 / >3


def trading_hours_between(ts_earlier, ts_later):
    """
    Count Mon–Fri hours between two timestamps. Both treated as naive UTC.
    Weekends excluded entirely (Saturday + Sunday do not contribute hours).

    A best-effort approximation: walks day-by-day, counts 24h for each weekday
    fully covered, plus partial-day fractions for the start/end days.

    Returns float hours. Returns None on bad input.
    """
    if ts_earlier is None or ts_later is None:
        return None
    if ts_later < ts_earlier:
        return None
    try:
        # Strip tz if present, treat as UTC-naive
        if hasattr(ts_earlier, 'tzinfo') and ts_earlier.tzinfo is not None:
            ts_earlier = ts_earlier.replace(tzinfo=None)
        if hasattr(ts_later, 'tzinfo') and ts_later.tzinfo is not None:
            ts_later = ts_later.replace(tzinfo=None)

        total_hours = 0.0
        cursor = ts_earlier
        while cursor < ts_later:
            day_start = cursor.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end   = day_start + timedelta(days=1)
            slice_end = min(day_end, ts_later)
            # weekday(): Mon=0 .. Sun=6 -> include 0..4 only
            if cursor.weekday() < 5:
                seconds = (slice_end - cursor).total_seconds()
                total_hours += seconds / 3600.0
            cursor = day_end
        return total_hours
    except Exception:
        return None

# PHASE 1 ONLY — fetches dealing range walls (structural state primary,
# legacy lookback fallback). Called only by smc_radar.py.
def get_dealing_range(ob, df_h1, h1_atr, pair_conf=None, current_price=None):
    """
    Wrapper over dealing_range.py — the new single-source-of-truth module.

    Resolution order:
      1. If structure_state.json has walls with prices for this pair, return
         them. Geometry is valid whenever both walls have prices (confirmed
         OR tentative). The `tentative` flag indicates whether either wall
         is currently a rolling extreme.
      2. If state is missing for this pair, fall back to the legacy window
         high/low so nothing downstream breaks.

    The `ob`, `h1_atr`, `current_price` params are kept for signature
    compatibility with every existing call site. Only `pair_conf['name']`
    is consumed by the new logic.
    """
    pair_name = (pair_conf or {}).get("name")
    pair_type = (pair_conf or {}).get("pair_type", "forex")

    # 1. Try structure_state.json
    try:
        import dealing_range as _DR
        state = _DR.load_state()
        walls = state.get(pair_name) if pair_name else None
    except Exception:
        walls = None

    proximal = float(ob.get("proximal_line", 0.0)) if ob else 0.0
    # Enter the structural-PD branch when EITHER a valid H4 dealing range exists
    # OR the legacy walls are present. The H4 range is the primary source now
    # (compute_pd_position prefers it internally), so gating this branch on the
    # legacy walls alone would discard a valid H4 range whenever the old engine
    # had no walls — letting the dead old engine veto the live one. compute_pd_
    # position returns valid=False safely if neither source is usable.
    _h4 = walls.get("h4_range") if walls else None
    _h4_ok = isinstance(_h4, dict) and bool(_h4.get("valid"))
    _legacy_ok = bool(walls and walls.get("ceiling_price") is not None
                      and walls.get("floor_price") is not None)
    if walls and (_h4_ok or _legacy_ok):
        pd_info = _DR.compute_pd_position(proximal, walls)
        chop_flag = bool(walls.get("last_event_chop", False))
        last_event_type = walls.get("last_event_type")
        if pd_info.get("valid"):
            tentative = bool(pd_info.get("tentative", False))
            if pd_info.get("source") == "h4_live":
                # Honest label: range came from the H4 dealing range, not walls.
                source = "h4_dealing_range"
            elif pd_info.get("fallback_active"):
                source = "structural_fallback_window"
            elif tentative:
                source = "structural_tentative"
            else:
                source = "structural_walls"
            return {
                "valid":       True,
                "range_high":  pd_info["range_high"],
                "range_low":   pd_info["range_low"],
                "equilibrium": pd_info["equilibrium"],
                "tentative":   tentative,
                "chop_flag":   chop_flag,
                "last_event_type": last_event_type,
                "source":      source
            }
        # Walls present but degenerate / incomplete — return geometry-only block.
        return {
            "valid":       False,
            "range_high":  pd_info.get("range_high", 0.0),
            "range_low":   pd_info.get("range_low",  0.0),
            "equilibrium": pd_info.get("equilibrium", 0.0),
            "tentative":   bool(pd_info.get("tentative", True)),
            "chop_flag":   chop_flag,
            "last_event_type": last_event_type,
            "source":      pd_info.get("source", "incomplete_walls")
        }

    # 2. Legacy fallback — preserves prior behaviour when state is missing.
    if df_h1 is None or len(df_h1) < 20:
        return {"valid": False, "tentative": False, "chop_flag": False,
                "last_event_type": None, "source": "insufficient_data"}
    lookback = DEALING_RANGE_LOOKBACK_H1.get(pair_type, 120)
    lookback = min(lookback, len(df_h1))
    df_window = df_h1.tail(lookback)
    range_high = float(df_window['High'].max())
    range_low = float(df_window['Low'].min())
    if range_high <= range_low:
        return {"valid": False, "tentative": False, "chop_flag": False,
                "last_event_type": None, "source": "degenerate_range"}
    eq = (range_high + range_low) / 2.0
    return {
        "valid": True,
        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": eq,
        "tentative": False,
        "chop_flag": False,
        "last_event_type": None,
        "source": f"legacy_window_{lookback}h"
    }
# PHASE 1 ONLY — extracts swing highs/lows for context tagging in smc_radar.py.
# Default lookback was changed 4->3 in commit 8300876 (2026-05-18).
# `min_leg_atr_mult`: if set, applies the ATR leg-size filter after geometric
# detection. Pass None to skip (e.g. M5 / Phase 3, where the H1-tuned multiple
# does not apply).
#
# SINGLE SOURCE (2026-06): geometry is NOT reimplemented here. This is a thin
# wrapper over dealing_range.detect_swings — the one lb-3 geometry + ATR
# leg-filter definition every consumer reads. Previously this function carried
# a second, parallel copy of the geometry loop; the two agreed today but could
# silently desync on any future edit to one and not the other (the exact
# "one concept, two implementations" trap). Now there is one implementation.
#
# `bounds` (optional {'max','min'}): legacy filter that drops swings whose
# high/low pierces the bounds. No live caller passes it; retained for signature
# compatibility and applied post-detection if ever supplied.
#
# Return shape is identical to detect_swings: {'type','idx','price','ts'} with
# `ts` as an ISO string. (The old wrapper returned `ts` as a pandas Timestamp;
# no caller reads `ts` from this function, so the representation change is inert.
# Verified callsites: smc_detector sweep observation, Phase 2 TP swings, Phase 3
# CHoCH — all read only idx/type/price.)
def get_swing_points(df, lookback=3, bounds=None, min_leg_atr_mult=MIN_LEG_ATR_MULT):
    import dealing_range as _dr
    swings = _dr.detect_swings(df, lookback=lookback, min_leg_atr_mult=min_leg_atr_mult)
    if bounds:
        swings = [s for s in swings
                  if not (s['price'] > bounds['max'] or s['price'] < bounds['min'])]
    return swings


# --- Shared chart-swing helpers ---------------------------------------------
# SINGLE SOURCE consumers. Charts (Phase 1 + Phase 2) read the persisted swing
# pool from dealing_range state and position it by timestamp. No chart detects
# swings itself, so every chart shows identical triangles / X markers.

def swings_for_chart(walls):
    """Return the persisted lb-3+ATR swing list from dealing_range state.

    `walls` is the per-pair structure-state dict (carries 'swings'). Returns a
    list of {ts, type, price, broken} or [] when absent (e.g. legacy state
    written before swings were persisted — caller renders no markers, never
    crashes)."""
    if not isinstance(walls, dict):
        return []
    sw = walls.get('swings')
    return sw if isinstance(sw, list) else []


def ts_to_utc_instant(raw):
    """Normalize any timestamp form to a UTC-aware pandas Timestamp, or None.

    The same instant can be serialized in different timezones across runs —
    yfinance returns GC=F (gold) in US/Eastern on some fetches and UTC on
    others. String equality on isoformat() then fails ('...-04:00' vs
    '...+00:00') even though both name the same moment, so persisted-swing
    markers silently drop or land on the wrong candle. Comparing the UTC
    instant instead is tz-agnostic. A tz-naive input is assumed UTC (the feed
    is normalized to UTC at the fetch boundary). Returns None on any failure so
    callers simply skip rather than crash."""
    if raw is None:
        return None
    try:
        ts = pd.Timestamp(raw)
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    try:
        if ts.tzinfo is None:
            return ts.tz_localize('UTC')
        return ts.tz_convert('UTC')
    except Exception:
        return None


def build_ts_to_local_x(df_plot):
    """Map each plotted candle's UTC instant -> its local x index (0-based).

    Lets a chart place a persisted swing (keyed by ts) at the right candle
    regardless of how the plot window was sliced. Keyed by UTC instant (not the
    raw isoformat string) so a swing persisted in one timezone still matches a
    chart rendered in another — see ts_to_utc_instant. Uses 'Datetime' column
    if present, else the index. Returns {} on any failure (markers simply
    skip)."""
    out = {}
    try:
        if 'Datetime' in df_plot.columns:
            ts_seq = df_plot['Datetime']
        else:
            ts_seq = df_plot.index
        for x in range(len(df_plot)):
            raw = ts_seq.iloc[x] if hasattr(ts_seq, 'iloc') else ts_seq[x]
            inst = ts_to_utc_instant(raw)
            if inst is not None:
                out[inst] = x
    except Exception:
        return {}
    return out


# PHASE 2 ONLY — counts BOS events since the last CHoCH. Reads dealing_range
# event state. Called only by Phase2_Alert_Engine.py.
def compute_bos_sequence_count(pair_name):
    """Count BOS events since the most recent CHoCH for the given pair.

    Reads from dealing_range's structure_state.json — single source of truth.
    Detection lives in dealing_range.py; this function does NOT walk-forward.

    Counter rules (matches dealing_range event semantics):
      - BOS  (plain or Range) -> increments counter
      - CHoCH                 -> resets counter (trend flipped)

    Returns:
      {
        'count':       int   (1 if no events / no BOS yet),
        'trend':       'bullish' | 'bearish' | None,
        'count_maxed': bool  (True if event ring is full — count may be capped),
      }

    `count` reports the position of the LATEST BOS (i.e. how many BOS events
    have printed since the last CHoCH, including the latest). If the most
    recent event is a CHoCH, count is reported as 1 (the CHoCH "is" the
    structure event of interest at that moment; downstream scoring uses tier).
    """
    try:
        import dealing_range as _dr
        state_all = _dr.load_state()
        pair_state = state_all.get(pair_name) or {}
    except Exception:
        return {'count': 1, 'trend': None, 'count_maxed': False}

    events = pair_state.get('events', []) or []
    trend = pair_state.get('trend')
    if not events:
        return {'count': 1, 'trend': trend, 'count_maxed': False}

    # Walk events forward.
    # CHoCH resets the counter. Both BOS and Range BOS increment it.
    # Range BOS is the more significant break (at the dealing range wall)
    # but both count — the caution threshold applies to the full sequence.
    count = 0
    for ev in events:
        kind = ev.get('type')
        tier = ev.get('tier')
        if kind == 'CHoCH':
            count = 0
        elif kind == 'BOS' and tier in ('BOS', 'Range', 'Major'):
            count += 1

    # If no BOS has fired since the last CHoCH, report count=1 so callers
    # treating the latest event as "the structural anchor" don't divide-by-zero.
    if count == 0:
        count = 1

    count_maxed = (len(events) >= _dr.EVENT_RING_MAX)
    return {'count': count, 'trend': trend, 'count_maxed': count_maxed}

# SHARED P1+P2+P3 — chart label stacking utility. Used by every chart-rendering
# path across all three phases. Cosmetic only — chart label overlap if broken,
# but alerts still fire correctly.
def stack_labels(labels, pair_conf):
    """
    Prevent label overlap on charts by offsetting prices that are too close.

    labels: list of (price, text, color) tuples.
    Returns: list of (adjusted_price, text, color) tuples, sorted by price.
    """
    if not labels:
        return labels

    dp = _dp(pair_conf)
    pair_type = pair_conf.get("pair_type", "forex")

    # Thresholds widened ~40% to prevent visual overlap at fontsize 10 (cosmetic only).
    thresholds = {
        "forex": 0.00042 if dp == 5 else 0.042,
        "index": 21.0,
        "commodity": 2.8
    }
    min_gap = thresholds.get(pair_type, 0.00042)
    sorted_labels = sorted(labels, key=lambda x: x[0])
    adjusted = [sorted_labels[0]]

    for i in range(1, len(sorted_labels)):
        price, text, color = sorted_labels[i]
        prev_price = adjusted[-1][0]
        if abs(price - prev_price) < min_gap:
            price = prev_price + min_gap
        adjusted.append((price, text, color))

    return adjusted


# INTERNAL — swing liveness check used by sweep scoring helpers in this file.
def is_swing_active(swing, df, pierce_min, before_idx=None):
    """
    A swing is ACTIVE (= unbroken AND unswept) if no candle between its
    formation and `before_idx` has either:
      - pierced its level by >= pierce_min (wick — drained the resting
        liquidity sitting at the level), OR
      - closed beyond it (body — broke the structural significance).

    SMC view: a swing's value as a sweep target is the untouched stop
    liquidity parked at it. Once price has wicked past it meaningfully,
    those orders are filled — the level is drained and no longer a
    valid sweep target.

    Args:
      swing:      {'type': 'high'|'low', 'price': float, 'idx': int}
      df:         OHLC dataframe.
      pierce_min: minimum pierce in price units to count as a sweep
                  (typically SWEEP_WICK_PIERCE_MIN_ATR * tf_atr).
      before_idx: only consider candles strictly before this idx.
                  Defaults to len(df). Used so the candle currently being
                  evaluated as a sweep candidate doesn't disqualify its
                  own target.

    Returns: bool — True if active (still valid as a sweep target).
    """
    if df is None or len(df) == 0 or not swing:
        return False
    if before_idx is None:
        before_idx = len(df)
    j_start = int(swing['idx']) + 1
    j_end   = int(before_idx)
    if j_start >= j_end:
        return True  # nothing has happened after the swing yet
    H = df['High'].values
    L = df['Low'].values
    C = df['Close'].values
    level = float(swing['price'])
    if swing['type'] == 'high':
        for j in range(j_start, j_end):
            if H[j] > level + pierce_min:  # wicked through (swept)
                return False
            if C[j] > level:                # closed beyond (broken)
                return False
    else:  # 'low'
        for j in range(j_start, j_end):
            if L[j] < level - pierce_min:
                return False
            if C[j] < level:
                return False
    return True


def _equal_levels_score(swept_swing, all_swings, pair_type, tf_atr,
                        df=None, before_idx=None,
                        recency_floor_idx=None):
    """
    Score the 'equal highs/lows' confluence around the swept swing.

    SMC-faithful rules (rewritten):
      - Pool: same-type swings within recency window. If `recency_floor_idx`
        is provided, includes swings with idx >= recency_floor_idx. Else
        includes swings within the last 50 candles before the swept swing.
      - ONLY counts swings that are ACTIVE (unbroken AND unswept) as of
        the swept swing's idx. Drained equal-level swings have no
        liquidity left and don't add confluence.
      - Counts how many active swings sit within the pair-aware
        equal-level tolerance of the swept swing's price.
      - Score is capped at 2 matches → max score 0.5 (unchanged tier semantics).

    `df` is required to evaluate active-ness. If not provided, falls back
    to the old "last 3 swings, no active filter" behaviour for backwards
    compat with any callers that don't yet pass df (defensive).

    Returns:
      (score, match_count)  where score in {0.0, 0.25, 0.5}
                            and match_count in {0, 1, 2}.
    """
    if not swept_swing or not all_swings or tf_atr is None or tf_atr <= 0:
        return 0.0, 0
    tol_mult = SWEEP_EQUAL_LEVEL_TOLERANCE_ATR.get(pair_type, 0.25)
    tolerance = tol_mult * tf_atr
    pierce_min = SWEEP_WICK_PIERCE_MIN_ATR.get(pair_type, 0.05) * tf_atr

    swept_idx   = int(swept_swing['idx'])
    anchor_price = float(swept_swing['price'])

    # Recency window for the pool.
    if recency_floor_idx is None:
        recency_floor_idx = max(0, swept_idx - 50)

    same_type = [s for s in all_swings
                 if s['type'] == swept_swing['type']
                 and recency_floor_idx <= s['idx'] <= swept_idx
                 and s['idx'] != swept_idx]

    if not same_type:
        return 0.0, 0

    # Filter to active (unbroken + unswept) up to the swept swing's idx.
    # `df` is the active-ness oracle. Without df, skip the filter
    # (degraded mode — preserves old behaviour for legacy callers).
    if df is not None:
        same_type = [s for s in same_type
                     if is_swing_active(s, df, pierce_min, before_idx=swept_idx)]

    if not same_type:
        return 0.0, 0

    matches = sum(1 for s in same_type if abs(s['price'] - anchor_price) <= tolerance)
    matches = min(matches, 2)

    if matches == 0:
        return 0.0, 0
    if matches == 1:
        return 0.25, 1
    return 0.5, 2


def _rejection_score(df, sweep_idx, swept_type, tf_atr):
    """
    Score the rejection quality of the sweep candle.

    For a bullish sweep (LONG, swept a low):
        wick = min(open, close) - low  (lower wick)
    For a bearish sweep (SHORT, swept a high):
        wick = high - max(open, close) (upper wick)
    body = abs(close - open)
    ratio = wick / max(body, 0.0001 * tf_atr)  -- epsilon prevents doji div-by-zero

    Tiers (4-tier, max 1.0):
      ratio < 1.0           -> 0.0   (no real rejection — body dominates)
      1.0 <= ratio < 2.0    -> 0.33  (weak rejection — ambiguous)
      2.0 <= ratio < 3.0    -> 0.66  (strong rejection — institutional signature)
      ratio >= 3.0          -> 1.0   (textbook rejection — clear stop run + reversal)

    Returns: (score, ratio_value)
    """
    if df is None or sweep_idx < 0 or sweep_idx >= len(df):
        return 0.0, 0.0
    if tf_atr is None or tf_atr <= 0:
        return 0.0, 0.0
    O = float(df['Open'].iloc[sweep_idx])
    C = float(df['Close'].iloc[sweep_idx])
    H = float(df['High'].iloc[sweep_idx])
    L = float(df['Low'].iloc[sweep_idx])
    body = abs(C - O)
    epsilon = 0.0001 * tf_atr
    if swept_type == 'low':       # bullish sweep -> lower wick matters
        wick = min(O, C) - L
    else:                         # 'high' -> bearish sweep -> upper wick
        wick = H - max(O, C)
    if wick < 0:
        wick = 0.0
    ratio = wick / max(body, epsilon)
    if ratio < 1.0:
        return 0.0, ratio
    if ratio < 2.0:
        return 0.33, ratio
    if ratio < 3.0:
        return 0.66, ratio
    return 1.0, ratio

def _sweep_tier(score):
    """Classify final sweep score into a label for narration. Max 3.0."""
    if score >= 2.4:
        return 'textbook'
    if score >= 1.8:
        return 'decent'
    if score > 0.0:
        return 'weak'
    return 'none'


# ============================================================================
# Phase 1 sweep observation (display-only, snapshot semantics)
# ============================================================================
# Anchored to the leg `[impulse_start_idx, ob_idx]` already on every OB. Scans
# the most-recent qualifying sweep in that leg (newest first). Records a
# snapshot onto ob['sweep_observed']. Phase 2 consumes the snapshot — it does
# NOT re-grade. Past observations are not re-evaluated when yfinance revises.
# ----------------------------------------------------------------------------

def _round_number_key(pair_name, pair_type):
    """Round-number grid lookup — JPY pairs need their own bucket."""
    if pair_name == 'USDJPY':
        return 'forex_jpy'
    return pair_type if pair_type in ROUND_NUMBER_GRID else 'forex'


def _nearest_round_number(price, grid):
    """Return the nearest grid level to `price`."""
    if grid <= 0:
        return price
    return round(price / grid) * grid


def _prior_trading_day_hl(df, anchor_ts):
    """
    Return (high, low) of the prior trading day for the candle containing
    `anchor_ts`. Trading day = Mon-Fri UTC. Sunday's "prior day" is Friday.

    Returns (None, None) if df has no candles in the prior trading day.
    """
    if df is None or len(df) == 0 or anchor_ts is None:
        return None, None
    try:
        anchor_dt = anchor_ts.to_pydatetime() if hasattr(anchor_ts, 'to_pydatetime') else anchor_ts
        if hasattr(anchor_dt, 'tzinfo') and anchor_dt.tzinfo is not None:
            anchor_dt = anchor_dt.replace(tzinfo=None)
        cursor = anchor_dt - timedelta(days=1)
        # Walk back across weekend if needed (Sat=5, Sun=6)
        while cursor.weekday() >= 5:
            cursor = cursor - timedelta(days=1)
        target_date = cursor.date()
        # Slice df rows whose index falls on target_date
        prior_mask = []
        for ts in df.index:
            t = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
            if hasattr(t, 'tzinfo') and t.tzinfo is not None:
                t = t.replace(tzinfo=None)
            prior_mask.append(t.date() == target_date)
        if not any(prior_mask):
            return None, None
        prior_df = df[prior_mask]
        return float(prior_df['High'].max()), float(prior_df['Low'].min())
    except Exception:
        return None, None


def _session_hl_until(df, anchor_ts, session_key):
    """
    High/low of the named session on `anchor_ts`'s UTC date, clipped to
    `anchor_ts` (we never include candles in the session's future relative
    to the OB candle). Asia wraps midnight — we use the asia window that
    *ends* on the anchor date.

    Returns (high, low) or (None, None) if no candles fall in window.
    """
    if df is None or len(df) == 0 or anchor_ts is None:
        return None, None
    if session_key not in SESSION_WINDOWS_UTC:
        return None, None
    try:
        anchor_dt = anchor_ts.to_pydatetime() if hasattr(anchor_ts, 'to_pydatetime') else anchor_ts
        if hasattr(anchor_dt, 'tzinfo') and anchor_dt.tzinfo is not None:
            anchor_dt = anchor_dt.replace(tzinfo=None)
        start_h, end_h = SESSION_WINDOWS_UTC[session_key]
        anchor_date = anchor_dt.date()
        if session_key == 'asia':
            # 22:00 (prev day) -> 07:00 (anchor date). If anchor_dt is before 07,
            # use the window ending today; else use the window starting today 22:00
            # which is in the FUTURE — instead use yesterday-22 -> today-07.
            sess_start = datetime.combine(anchor_date, datetime.min.time()).replace(hour=start_h) - timedelta(days=1)
            sess_end   = datetime.combine(anchor_date, datetime.min.time()).replace(hour=end_h)
        else:
            sess_start = datetime.combine(anchor_date, datetime.min.time()).replace(hour=start_h)
            sess_end   = datetime.combine(anchor_date, datetime.min.time()).replace(hour=end_h)
        # Clip end to anchor (we cannot see the future)
        sess_end = min(sess_end, anchor_dt)
        if sess_end <= sess_start:
            return None, None
        mask = []
        for ts in df.index:
            t = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
            if hasattr(t, 'tzinfo') and t.tzinfo is not None:
                t = t.replace(tzinfo=None)
            mask.append(sess_start <= t < sess_end)
        if not any(mask):
            return None, None
        sess_df = df[mask]
        return float(sess_df['High'].max()), float(sess_df['Low'].min())
    except Exception:
        return None, None


def _compute_context_tags(swept_price, swept_type, df, anchor_ts,
                          pair_name, pair_type, tf_atr):
    """
    Return list of human-readable tags describing what the swept level is
    aligned with. Tags use the same widened ATR tolerance as equal-levels,
    except round-number which uses its own tight bucket.
    """
    tags = []
    if tf_atr is None or tf_atr <= 0:
        return tags
    tol = SWEEP_EQUAL_LEVEL_TOLERANCE_ATR.get(pair_type, 0.30) * tf_atr

    # Round number
    rn_key = _round_number_key(pair_name, pair_type)
    grid   = ROUND_NUMBER_GRID.get(rn_key, 0.0)
    rn_tol = ROUND_NUMBER_TOLERANCE.get(rn_key, 0.0)
    if grid > 0:
        nearest = _nearest_round_number(swept_price, grid)
        if abs(swept_price - nearest) <= rn_tol:
            tags.append('round_number')

    # Prior day H/L
    pd_high, pd_low = _prior_trading_day_hl(df, anchor_ts)
    if swept_type == 'low' and pd_low is not None:
        if abs(swept_price - pd_low) <= tol:
            tags.append('prior_day_low')
    if swept_type == 'high' and pd_high is not None:
        if abs(swept_price - pd_high) <= tol:
            tags.append('prior_day_high')

    # Per-pair session H/L
    for sess in PAIR_SESSION_TAGS.get(pair_name, []):
        s_high, s_low = _session_hl_until(df, anchor_ts, sess)
        if swept_type == 'low' and s_low is not None:
            if abs(swept_price - s_low) <= tol:
                tags.append(f'{sess}_low')
        if swept_type == 'high' and s_high is not None:
            if abs(swept_price - s_high) <= tol:
                tags.append(f'{sess}_high')

    return tags


# PHASE 1 ONLY — Phase 1 sweep observation (display badge in scout email).
# Called only by smc_radar.py.
def observe_phase1_sweep(df, ob_idx, impulse_start_idx, direction,
                         tf_atr, pair_type, pair_name, tf_label='H1',
                         event_type='BOS', prior_event_idx=None,
                         fallback_lookback=48):
    """
    Sweep observation — uniform detection used by BOTH Phase 1 (H1 snapshot
    at OB formation) and Phase 2 (M15 entry-time sweep).

    Swings are computed on the FULL df at lookback=3 and filtered to the
    search window. Full-df detection gives every candle in the window its
    true neighbours (computing on the slice silently drops 3 candidates at
    each edge).

    Search window:
      - CHoCH: `[prior_event_idx, ob_idx]` when prior_event_idx is given.
        Falls back to leg-anchored when no prior event exists.
      - BOS:   `[prior_event_idx, ob_idx]` when prior_event_idx is given
        (symmetric with CHoCH — covers the entire trend leg the BOS is
        continuing). Caller passes the most recent OPPOSING-direction
        structural event (a BOS / Range BOS / CHoCH that reversed the
        trend; v2 has no Major/Minor). Fallback when prior_event_idx is None or
        unresolvable: `max(0, ob_idx - fallback_lookback)`.

    `fallback_lookback`: candle count used when no structural anchor is
    available. Phase 1 (H1) uses default 6 (≈6 trading hours). Phase 2 (M15)
    has no structural events of its own, so it always hits this fallback —
    callers should pass a larger value (e.g. 20 M15 candles = ~5 hours).

    A qualifying sweep candle:
      - Bullish OB: candle's low pierces a prior pivot LOW by at least the
        pair's wick-pierce minimum, AND closes back above that low.
      - Bearish OB: candle's high pierces a prior pivot HIGH by the pierce
        minimum, AND closes back below that high.

    Target eligibility (SMC-faithful, NEW):
      - Only ACTIVE swings qualify as sweep targets. Active = unbroken AND
        unswept by any candle between the swing's birth and the candidate
        sweep candle's idx. Drained / broken swings have no liquidity left.
      - If no active target exists in the window for any candidate, the
        snapshot returns `exists=False`. No fallback to swept targets.

    Selection:
      - For each candidate candle in the window, find the deepest pierce
        among all ACTIVE prior same-type pivots. Score the candidate.
      - Across all candidates with a qualifying active target, the
        HIGHEST-SCORED candidate wins. Tie-break: more recent.
      - The OB candle itself is allowed to be the sweep candle (per ICT
        methodology — engulfing / rejection-block patterns).

    Returns the snapshot dict written to ob['sweep_observed'] (Phase 1) or
    consumed live by Phase 2's M15 detection. Canonical empty shape carries
    `components` and `hours_before_anchor` so downstream consumers can read
    a uniform schema regardless of whether a sweep was found.
    """
    not_observed = {
        'exists': False, 'tf': tf_label, 'tier': 'none', 'score': 0.0,
        'price': None, 'sweep_idx': None,
        'swept_swing_idx': None, 'swept_swing_ts': None,
        'timestamp': None,
        'wick_distance_pips': 0.0, 'wick_body_ratio': 0.0,
        'equal_levels_count': 0, 'context_tags': [],
        'components': {
            'base': 0.0,
            'equal_levels': 0.0, 'equal_levels_matches': 0,
            'rejection': 0.0, 'wick_body_ratio': 0.0,
        },
        'hours_before_anchor': None,
    }

    if df is None or len(df) == 0:
        return not_observed
    if ob_idx is None or impulse_start_idx is None:
        return not_observed
    if direction not in ('bullish', 'bearish'):
        return not_observed
    if tf_atr is None or tf_atr <= 0:
        return not_observed
    if impulse_start_idx < 0 or ob_idx >= len(df) or impulse_start_idx > ob_idx:
        return not_observed

    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    C = df['Close'].values.astype(float)

    bias_low = (direction == 'bullish')  # bullish OB -> hunt low sweeps
    swing_type_we_want = 'low' if bias_low else 'high'
    pierce_min = SWEEP_WICK_PIERCE_MIN_ATR.get(pair_type, 0.05) * tf_atr

    # Swing pool — computed once on the full df at lookback=3.
    all_swings = get_swing_points(df, lookback=3)
    if not all_swings:
        return not_observed

    # Search window low bound.
    #
    # LOCKED 2026-06 (decided with the trader, verified on real USDJPY swings):
    # the sweep that VALIDATES an order block is the local liquidity run inside
    # that OB's OWN impulse leg PLUS a small lookback before it — the stop-run
    # price took immediately before the displacement that built the zone. The OB
    # is the origin of this leg, so its fueling sweep lives in
    # [impulse_start - SWEEP_LOOKBACK_BEFORE_IMPULSE, ob_idx], FLOORED at the prior
    # structural event. The lookback (2026-06) catches the common case where the
    # sweep candle is a candle or two BEFORE the impulse start (sweep -> base ->
    # impulse); the prior-event floor keeps it from reaching an earlier leg.
    #
    # WHY THE OLD UNBOUNDED PRIOR ANCHORS WERE WRONG (tested + rejected on USDJPY):
    #   - prior OPPOSING event: on a continuation BOS deep in a trend this reached
    #     back to the trend's origin and grabbed unrelated old liquidity (picked
    #     159.09 / 159.531, candles before this leg existed).
    #   - prior SAME-direction break: still sat before this leg (159.574).
    #   The fix is NOT "impulse leg only" but "impulse leg + a few candles, hard-
    #   floored at the prior event" — local enough to stay on the fueling run
    #   (159.706), bounded enough to never reach the earlier leg.
    #
    # `prior_event_idx` is now used as the LOWER-BOUND FLOOR (not the anchor):
    # the lookback can extend before impulse_start but never past the prior event.
    #
    # REGRESSION NOTE (still relevant): get_swing_points tags the sweep candle
    # itself (the leg's terminal extreme) as a swing. That does NOT break the
    # leg-bounded window: the candidate loop below requires the swept target to
    # be a swing with idx STRICTLY less than the candidate candle (s['idx'] < i),
    # so a sweep candle can never sweep its own level. The earlier failure was a
    # backward "find pullback before impulse_start" walk that latched search_lo
    # ONTO the sweep candle; a forward leg window has no such collision.
    #
    # `fallback_lookback` is retained for Phase 2 (M15), which has no structural
    # impulse leg of its own and walks back from impulse_start by this many
    # candles. Phase 1 (H1) always has the leg, so it never hits the fallback.
    if impulse_start_idx is not None:
        # Extend a few candles BEFORE the impulse start to catch the sweep that
        # turned the market (sweep -> base -> impulse). FLOOR at the prior
        # structural event (prior_event_idx + 1) so the window can never reach an
        # earlier leg's unrelated liquidity — the exact over-reach the
        # impulse-leg-only lock fixed. With no prior event, just clamp at 0.
        search_lo = int(impulse_start_idx) - SWEEP_LOOKBACK_BEFORE_IMPULSE
        if prior_event_idx is not None:
            try:
                search_lo = max(search_lo, int(prior_event_idx) + 1)
            except (TypeError, ValueError):
                pass
    else:
        search_lo = max(0, int(ob_idx) - int(fallback_lookback))

    if search_lo < 0:
        search_lo = 0
    if search_lo > ob_idx:
        return not_observed

    # Filter swings to the search window (same-type only). idx is absolute.
    swings_in_window = [
        s for s in all_swings
        if s['type'] == swing_type_we_want
        and search_lo <= s['idx'] <= ob_idx
    ]
    if not swings_in_window:
        return not_observed

    swept_type = 'low' if bias_low else 'high'

    # Score every candidate. The HIGHEST-SCORED candidate with an ACTIVE
    # (unbroken + unswept) target wins. NOT first-match.
    # Tie-break: more recent (higher candidate idx).
    best = None  # (total, candidate_idx, payload_dict)

    for i in range(search_lo, ob_idx + 1):
        prior_swings = [s for s in swings_in_window if s['idx'] < i]
        if not prior_swings:
            continue

        # Find the deepest pierce among all ACTIVE prior same-type pivots.
        # `before_idx=i` ensures the candidate doesn't disqualify its own
        # target by counting itself as a sweep of the level.
        winning_swing = None
        winning_depth = 0.0
        for s in prior_swings:
            level = s['price']
            if bias_low:
                pierced = (L[i] < level - pierce_min) and (C[i] > level)
                depth   = level - L[i] if pierced else 0.0
            else:
                pierced = (H[i] > level + pierce_min) and (C[i] < level)
                depth   = H[i] - level if pierced else 0.0
            if not pierced:
                continue
            if not is_swing_active(s, df, pierce_min, before_idx=i):
                continue
            if depth > winning_depth:
                winning_swing = s
                winning_depth = depth

        if winning_swing is None:
            continue

        # ------------------------------------------------------------------
        # Sweep-candle survivorship — SMC-faithful right-side check.
        #
        # The sweep candle's pierce extreme must remain the extreme of the
        # leg from the sweep candle through to the OB candle (inclusive).
        # If any later candle in [i+1, ob_idx] wicks STRICTLY deeper than
        # the sweep candle's pierce extreme, the original "sweep" was just
        # a wick on the way to a real extreme — fresh liquidity has been
        # parked beyond it and the impulse into the OB is not fueled by
        # this candidate's stop-run. Reject and let the loop find the
        # genuine sweep candle (which may be the OB candle itself —
        # engulfing / rejection-block pattern, already supported).
        #
        # Strictly deeper only: equal-depth later candles are allowed
        # (they form the "equal levels" confluence the scorer rewards).
        # ------------------------------------------------------------------
        if bias_low:
            sweep_extreme = L[i]
            disqualified  = any(L[j] < sweep_extreme for j in range(i + 1, ob_idx + 1))
        else:
            sweep_extreme = H[i]
            disqualified  = any(H[j] > sweep_extreme for j in range(i + 1, ob_idx + 1))
        if disqualified:
            continue

        level = winning_swing['price']
        eq_score, eq_matches = _equal_levels_score(
            winning_swing, swings_in_window, pair_type, tf_atr,
            df=df, before_idx=i,
            recency_floor_idx=search_lo,
        )
        rej_score, wb_ratio  = _rejection_score(df, i, swept_type, tf_atr)
        total = SWEEP_SCORE_BASE_MAX + eq_score + rej_score
        tier  = _sweep_tier(total)

        # Pierce distance in display units (pips for forex, points/$ for others).
        if pair_type == 'forex':
            pip_unit = 0.01 if pair_name == 'USDJPY' else 0.0001
        elif pair_type == 'index':
            pip_unit = 1.0
        else:  # commodity (Gold)
            pip_unit = 1.0
        if bias_low:
            raw_distance = level - L[i]
        else:
            raw_distance = H[i] - level
        wick_distance_pips = round(max(raw_distance, 0.0) / pip_unit, 2)

        sweep_ts = df.index[i]
        try:
            sweep_ts_iso = sweep_ts.isoformat() if hasattr(sweep_ts, 'isoformat') else str(sweep_ts)
        except Exception:
            sweep_ts_iso = str(sweep_ts)
        try:
            swept_swing_ts = df.index[int(winning_swing['idx'])]
            swept_swing_ts_iso = (
                swept_swing_ts.isoformat() if hasattr(swept_swing_ts, 'isoformat')
                else str(swept_swing_ts)
            )
        except Exception:
            swept_swing_ts_iso = None

        context_tags = _compute_context_tags(
            level, swept_type, df, sweep_ts, pair_name, pair_type, tf_atr
        )

        # Hours from the sweep candle to the OB anchor. Trading-hours
        # (Mon-Fri) so weekends don't inflate the gap. Used by Phase 2 to
        # narrate sweep recency in the email.
        try:
            ob_ts_for_hours = df.index[int(ob_idx)]
            if hasattr(ob_ts_for_hours, 'to_pydatetime'):
                ob_dt_for_hours = ob_ts_for_hours.to_pydatetime()
            else:
                ob_dt_for_hours = ob_ts_for_hours
            if hasattr(ob_dt_for_hours, 'tzinfo') and ob_dt_for_hours.tzinfo is not None:
                ob_dt_for_hours = ob_dt_for_hours.replace(tzinfo=None)
            sw_dt_for_hours = sweep_ts.to_pydatetime() if hasattr(sweep_ts, 'to_pydatetime') else sweep_ts
            if hasattr(sw_dt_for_hours, 'tzinfo') and sw_dt_for_hours.tzinfo is not None:
                sw_dt_for_hours = sw_dt_for_hours.replace(tzinfo=None)
            hrs_before = trading_hours_between(sw_dt_for_hours, ob_dt_for_hours)
            hrs_before = round(hrs_before, 1) if hrs_before is not None else None
        except Exception:
            hrs_before = None

        payload = {
            'exists':              True,
            'tf':                  tf_label,
            'tier':                tier,
            'score':               round(total, 3),
            'price':               float(level),
            'sweep_idx':           int(i),
            'swept_swing_idx':     int(winning_swing['idx']),
            'swept_swing_ts':      swept_swing_ts_iso,
            'timestamp':           sweep_ts_iso,
            'wick_distance_pips':  wick_distance_pips,
            'wick_body_ratio':     round(wb_ratio, 2),
            'equal_levels_count':  int(eq_matches),
            'context_tags':        context_tags,
            'observed_at':         datetime.utcnow().isoformat(),
            # Uniform-schema fields for Phase 2 consumers.
            'components': {
                'base':                 SWEEP_SCORE_BASE_MAX,
                'equal_levels':         eq_score,
                'equal_levels_matches': int(eq_matches),
                'rejection':            rej_score,
                'wick_body_ratio':      round(wb_ratio, 2),
            },
            'hours_before_anchor': hrs_before,
        }

        if best is None or total > best[0] or (total == best[0] and i > best[1]):
            best = (total, i, payload)

    if best is None:
        return not_observed
    return best[2]


# NEW
# NEW
# SHARED P1+P2+P3 — 3-candle FVG detection inside a zone. Called by:
#   Phase 1 (H1 FVG inside dealing range), Phase 2 (M15 FVG for scorecard input),
#   Phase 3 (M5 FVG for chart context). Timeframe determined by caller's df.
#   Any change to detection logic affects FVG identification in all three phases.
def detect_fvg_in_zone(df, bias, zone_top, zone_bottom, atr_floor,
                       leg_start_idx=None, leg_end_idx=None,
                       pair_type="forex"):
    """
    Find the most relevant 3-candle FVG inside the displacement leg.

    Selection rule:
      Scan oldest-first within the window. Return the FIRST live (pristine or
      partial) FVG — i.e. the FVG closest to the OB candle. This is the FVG
      price hits first on retrace. If no live FVG exists, fall back to the
      most-recent ghost (latest fully-mitigated FVG) so chart context is kept.

    Mitigation states:
      - 'pristine' : price has NOT touched FVG proximal since formation.
      - 'partial'  : price touched proximal but NOT distal. Full score.
      - 'full'     : FVG is dead. Definition is pair-aware (see below).

    Pair-aware full mitigation:
      - Forex: touch-based full mit. Any wick through distal kills the FVG.
        Forex respects levels precisely.
      - Index (NAS100) / Commodity (Gold): close-based full mit. A wick spike
        does not kill the FVG; only a candle close past distal does. Both
        instruments routinely wick through levels on news without genuine
        fills.

    Partial mitigation is touch-based for all pairs (proximal touch is the
    proximal touch regardless of close).

    Proximal / Distal by bias:
      LONG  FVG: proximal = fvg_top    | distal = fvg_bottom
      SHORT FVG: proximal = fvg_bottom | distal = fvg_top

    Return shape:
      Live (pristine/partial) -> {"exists": True, "fvg_top": ft, "fvg_bottom": fb,
                                  "c1_idx": k, "c3_idx": k+2,
                                  "c1_timestamp": iso_str,
                                  "mitigation": "pristine" | "partial",
                                  "was_detected": True}
      Ghost only (full mit)   -> {"exists": False, "fvg_top": None, "fvg_bottom": None,
                                  "was_detected": True, "mitigation": "full",
                                  "ghost_top": ft, "ghost_bottom": fb,
                                  "ghost_c1_idx": k, "ghost_c3_idx": k+2,
                                  "ghost_c1_timestamp": iso_str,
                                  "mitigated_at_idx": m}
      Nothing                 -> {"exists": False, "fvg_top": None, "fvg_bottom": None,
                                  "was_detected": False, "mitigation": "none"}

    `c1_timestamp` / `ghost_c1_timestamp` carry the absolute ISO timestamp of
    the c1 (left) candle. Cross-phase chart rendering must use the timestamp
    because c1_idx is local to the detection df and does not translate across
    different yfinance fetches.
    """
    _empty = {"exists": False, "fvg_top": None, "fvg_bottom": None,
              "was_detected": False, "mitigation": "none"}

    if df is None or len(df) < 5:
        return _empty
    if atr_floor is None or atr_floor <= 0:
        return _empty

    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)
    C = df['Close'].values.astype(float)
    n = len(df)

    # Resolve a candle's absolute timestamp once per FVG return. Cross-phase
    # callers (Phase 2 / Phase 3) cannot rely on c1_idx because their df is a
    # different fetch than Phase 1's; the timestamp is the only stable anchor.
    def _idx_to_iso(k):
        try:
            ts = df.index[int(k)]
            return ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
        except Exception:
            return None

    # Close-based full mit for index (NAS) and commodity (Gold) — both wick
    # through levels on news without genuine fills. Forex keeps touch-based.
    close_based_full_mit = pair_type in ("index", "commodity")

    # Session-gap guard. A textbook 3-candle FVG requires candle 2 to be a
    # real traded candle whose body+wicks didn't fill the imbalance between
    # candle 1 and candle 3. When candle 2's timestamp is more than one bar
    # after candle 1 (weekend close, holiday, exchange halt), there IS no
    # real candle 2 — the "gap" is just an unrelated price jump across a
    # market-closed period. Veterans do not respect these as FVGs.
    #
    # Detection: infer the dominant timestamp delta from the median pairwise
    # gap of the df index, then reject any 3-candle pattern where ts(k+1)-ts(k)
    # exceeds 1.5x that delta. Applied to BOTH LONG and SHORT branches below.
    def _session_gap_between(a: int, b: int) -> bool:
        try:
            ta = df.index[int(a)]
            tb = df.index[int(b)]
            if not (hasattr(ta, 'to_pydatetime') and hasattr(tb, 'to_pydatetime')):
                return False
            delta_seconds = (tb - ta).total_seconds()
            return delta_seconds > _bar_seconds * 1.5
        except Exception:
            return False

    _bar_seconds = None
    try:
        idx = df.index
        if len(idx) >= 5 and hasattr(idx[0], 'to_pydatetime'):
            diffs = []
            for k in range(1, min(len(idx), 30)):
                d = (idx[k] - idx[k - 1]).total_seconds()
                if d > 0:
                    diffs.append(d)
            if diffs:
                diffs.sort()
                _bar_seconds = diffs[len(diffs) // 2]  # median
    except Exception:
        _bar_seconds = None
    # If we can't infer the bar size, the gap guard is a no-op (return False).
    if _bar_seconds is None or _bar_seconds <= 0:
        _bar_seconds = float('inf')

    # Oldest-first scan: returns FVG closest to OB. When a live FVG exists,
    # it is preferred over later ghosts (price hits the deepest live FVG first
    # on retrace).
    if leg_start_idx is not None and leg_end_idx is not None:
        k_lo = max(leg_start_idx, 0)
        k_hi = min(leg_end_idx - 1, n - 3)
        if k_hi < k_lo:
            return _empty
        scan_range = range(k_lo, k_hi + 1)
    else:
        # Default fallback (Phase 3 M5 path): scan last ~30 bars oldest-first.
        scan_range = range(max(0, n - 30), n - 2)

    last_ghost = None  # remember the most-recent ghost for fallback display

    for k in scan_range:
        if k + 2 >= n or k < 0:
            continue

        if bias == "LONG" and H[k] < L[k + 2]:
            ft, fb = float(L[k + 2]), float(H[k])
            if (ft - fb) < atr_floor:
                continue
            # Reject weekend/holiday gaps masquerading as FVGs (see comment
            # near _session_gap_between definition above). Either edge with a
            # session-close gap disqualifies the pattern.
            if _session_gap_between(k, k + 1) or _session_gap_between(k + 1, k + 2):
                continue
            # LONG: proximal = ft, distal = fb.
            full_fill_idx = None
            partial_hit = False
            for m in range(k + 3, n):
                # Full mit check (pair-aware).
                if close_based_full_mit:
                    if C[m] < fb:
                        full_fill_idx = m
                        break
                else:
                    if L[m] <= fb:
                        full_fill_idx = m
                        break
                # Partial check (touch-based for all pairs).
                if L[m] <= ft:
                    partial_hit = True
            if full_fill_idx is not None:
                # Track latest ghost; keep scanning for a live FVG closer to BOS.
                last_ghost = {
                    "exists": False, "fvg_top": None, "fvg_bottom": None,
                    "was_detected": True, "mitigation": "full",
                    "ghost_top": ft, "ghost_bottom": fb,
                    "ghost_c1_idx": k, "ghost_c3_idx": k + 2,
                    "ghost_c1_timestamp": _idx_to_iso(k),
                    "mitigated_at_idx": full_fill_idx
                }
                continue
            return {
                "exists": True, "fvg_top": ft, "fvg_bottom": fb,
                "c1_idx": k, "c3_idx": k + 2,
                "c1_timestamp": _idx_to_iso(k),
                "mitigation": "partial" if partial_hit else "pristine",
                "was_detected": True
            }

        elif bias == "SHORT" and L[k] > H[k + 2]:
            ft, fb = float(L[k]), float(H[k + 2])
            if (ft - fb) < atr_floor:
                continue
            if _session_gap_between(k, k + 1) or _session_gap_between(k + 1, k + 2):
                continue
            # SHORT: proximal = fb, distal = ft.
            full_fill_idx = None
            partial_hit = False
            for m in range(k + 3, n):
                if close_based_full_mit:
                    if C[m] > ft:
                        full_fill_idx = m
                        break
                else:
                    if H[m] >= ft:
                        full_fill_idx = m
                        break
                if H[m] >= fb:
                    partial_hit = True
            if full_fill_idx is not None:
                last_ghost = {
                    "exists": False, "fvg_top": None, "fvg_bottom": None,
                    "was_detected": True, "mitigation": "full",
                    "ghost_top": ft, "ghost_bottom": fb,
                    "ghost_c1_idx": k, "ghost_c3_idx": k + 2,
                    "ghost_c1_timestamp": _idx_to_iso(k),
                    "mitigated_at_idx": full_fill_idx
                }
                continue
            return {
                "exists": True, "fvg_top": ft, "fvg_bottom": fb,
                "c1_idx": k, "c3_idx": k + 2,
                "c1_timestamp": _idx_to_iso(k),
                "mitigation": "partial" if partial_hit else "pristine",
                "was_detected": True
            }

    # No live FVG found in the window. Return the latest ghost if any.
    if last_ghost is not None:
        return last_ghost
    return _empty


# PHASE 2 ONLY — computes SL/TP1/TP2 from H1 structure.
# (Phase 3 historically called this too. Phase 3 is disabled in the H1-only
# migration; if it is ever re-enabled, this function's H1-only behaviour is
# what it will get.)
def compute_phase2_levels(pair_conf, bias, ob, current_price, df_h1,
                          entry_zone="proximal"):
    """
    Phase 2 entry / SL / TP computation. H1-only since 2026-05-26 migration.

    Entry: always taken from H1 OB geometry. `entry_zone` selects the line:
      - "proximal" (default, live) -> OB proximal edge. Standard SMC limit.
      - "50pct" (backtest only)    -> OB midpoint. Deeper fill, tighter R.
    Both entries share the same SL (OB distal +/- 1x spread).

    SL: H1 OB distal +/- 1x spread.

    TP swings: H1 swings at lookback=3.
      TP1 = nearest opposing H1 swing past entry that clears 1.5R. If NO swing
            clears 1.5R, fall back to the H4 dealing-range wall (ceiling for
            LONG / floor for SHORT) if it clears 1.5R. If neither qualifies ->
            no trade (reason 'no_qualifying_target', logged for counting).
      TP2 = next opposing H1 swing past TP1 (no RR gate). None when TP1 is the
            wall, or when no further swing exists; simulator rides TP1 + BE-stop.

    Limit-order chase guard (proximal entry only):
      If the proximal entry sits on the wrong side of current price (LONG
      proximal above current, SHORT proximal below), the alert is invalid --
      price has already moved through the zone, the limit would chase.
      The 50pct entry skips the chase guard because by construction it sits
      deeper inside the OB than the proximal -- it's a pending limit waiting
      for further penetration, not a market-time decision.
    """
    dp = _dp(pair_conf)
    # Spread unit per decimal_places. dp=5 forex (0.0001/pip), dp=3 JPY (0.01/pip),
    # dp=2 commodity/index (1.0/point). Hard-coded forex-only fallback collapsed
    # GOLD/NAS spread to 0.005/0.02 -- effectively no SL buffer.
    _pip_for_dp = {5: 0.0001, 3: 0.01, 2: 1.0}.get(dp, 0.0001)
    spread_val = pair_conf.get("spread_pips", 2) * _pip_for_dp

    # H1 OB geometry. Strict read -- schema drift returns invalid rather than guess.
    try:
        h1_top = float(ob['high'])
        h1_bot = float(ob['low'])
    except (KeyError, TypeError, ValueError):
        return {"valid": False,
                "reason": "H1 OB geometry missing -- zone schema drift."}

    # H1-only entry: H1 OB geometry, no M15 nest lookup.
    # Proximal = OB edge nearest to price; 50pct = OB midpoint.
    # SL = OB distal +/- 1x spread (same for both entry zones; R-distance
    # naturally differs because entry differs).
    ob_top, ob_bot = h1_top, h1_bot
    entry_source = "H1 OB " + ("50% Mean" if entry_zone == "50pct" else "Proximal")
    ob_mid = (ob_top + ob_bot) / 2.0
    if bias == "LONG":
        entry = ob_mid if entry_zone == "50pct" else ob_top
        sl = ob_bot - spread_val
    else:
        entry = ob_mid if entry_zone == "50pct" else ob_bot
        sl = ob_top + spread_val

    # Limit-order chase guard — proximal entry only.
    # If the proximal entry sits on the wrong side of current price, the
    # limit would chase (LONG buying above market, SHORT selling below).
    # Tolerance = 0.5x spread for rounding / tick noise.
    # The 50pct entry skips this guard: it sits deeper in the OB by
    # construction, so it's a pending limit waiting for further penetration,
    # not a market-time decision. The backtest dual-entry A/B comparison
    # depends on both entries getting the same chase-policy treatment.
    if entry_zone == "proximal":
        tolerance = 0.5 * spread_val
        if bias == "LONG" and entry > current_price + tolerance:
            return {
                "valid": False,
                "reason": (f"Entry {round(entry, dp)} is above current price "
                           f"{round(current_price, dp)} -- LONG limit would chase price.")
            }
        if bias == "SHORT" and entry < current_price - tolerance:
            return {
                "valid": False,
                "reason": (f"Entry {round(entry, dp)} is below current price "
                           f"{round(current_price, dp)} -- SHORT limit would chase price.")
            }

    risk = abs(entry - sl)
    if risk == 0:
        return {"valid": False, "reason": "Zero risk -- entry == SL."}

    # TP swings from H1 at lookback=3. Opposing swings only, past entry.
    h1_swings = get_swing_points(df_h1, lookback=3)
    if bias == "LONG":
        opposing = [s['price'] for s in h1_swings
                    if s['type'] == 'high' and s['price'] > entry]
        opposing.sort()  # ascending -- nearest first
    else:
        opposing = [s['price'] for s in h1_swings
                    if s['type'] == 'low' and s['price'] < entry]
        opposing.sort(reverse=True)  # descending -- nearest first

    # TP1: nearest opposing swing past entry clearing 1.5R.
    tp1 = None
    tp1_rr = 0.0
    tp1_idx_in_opposing = None
    for i, target in enumerate(opposing):
        rr = abs(target - entry) / risk
        if rr >= 1.5:
            tp1 = target
            tp1_rr = rr
            tp1_idx_in_opposing = i
            break

    # TP1 fallback ladder (2026-06). A fresh CHoCH often has NO opposing
    # leg-filtered swing >= 1.5R yet, which silently no-traded the best setups.
    # Fallback to the H4 dealing-range WALL — the institutional draw on liquidity
    # (LONG aims at the ceiling, SHORT at the floor). The wall is frozen per-OB on
    # ob['dealing_range'] (no state load, no wobble) and still must clear 1.5R.
    tp1_source = "swing"
    if tp1 is None:
        dr_for_tp = ob.get('dealing_range') if isinstance(ob, dict) else None
        wall = None
        if isinstance(dr_for_tp, dict) and dr_for_tp.get('valid'):
            wall = dr_for_tp.get('range_high') if bias == "LONG" else dr_for_tp.get('range_low')
        try:
            wall = float(wall) if wall is not None else None
        except (TypeError, ValueError):
            wall = None
        if wall is not None:
            on_side = (wall > entry) if bias == "LONG" else (wall < entry)
            wall_rr = abs(wall - entry) / risk
            if on_side and wall_rr >= 1.5:
                tp1 = wall
                tp1_rr = wall_rr
                tp1_idx_in_opposing = None  # wall has no swing index -> no swing TP2
                tp1_source = "h4_wall"

    if tp1 is None:
        return {
            "valid": False,
            # Distinct, machine-readable reason so the scan log can COUNT how often
            # a real setup is dropped purely for lack of a target (the quiet
            # no-trade hole). Was previously two prose strings that were hard to tally.
            "reason": ("no_qualifying_target -- no opposing H1 swing and no H4 wall "
                       ">= 1.5R past entry."),
            "entry": round(entry, dp),
            "sl": round(sl, dp),
        }

    # TP2: next opposing swing past TP1. No RR gate, no fallback.
    # Textbook SMC: TP2 is the next real liquidity pool past TP1. If none
    # exists in the swing series, the trade rides TP1 -> BE-stop (handled
    # by the simulator). When TP1 is the H4 wall (no swing index), there is no
    # swing-based TP2 -- the wall is the final target.
    tp2 = None
    if tp1_idx_in_opposing is not None and tp1_idx_in_opposing + 1 < len(opposing):
        tp2 = opposing[tp1_idx_in_opposing + 1]

    out = {
        "valid": True,
        "entry": round(entry, dp),
        "sl": round(sl, dp),
        "tp1": round(tp1, dp),
        "rr": round(tp1_rr, 2),
        "entry_source": entry_source,
        "tp1_source": tp1_source,  # "swing" | "h4_wall"
    }
    # Post-rounding direction check. Pre-round, the swing list ordering
    # guarantees tp2 > tp1 (LONG) or tp2 < tp1 (SHORT). After rounding to
    # `dp` decimals, two nearby swings can collapse to the same price
    # (e.g. 1.23459 and 1.23456 both round to 1.23456 at dp=5).
    # If that happens, drop TP2 -- the trade still has TP1 + BE-stop policy.
    if tp2 is not None:
        tp1_r = round(tp1, dp)
        tp2_r = round(tp2, dp)
        if (bias == "LONG" and tp2_r > tp1_r) or (bias == "SHORT" and tp2_r < tp1_r):
            out["tp2"] = tp2_r
            out["tp2_rr"] = round(abs(tp2 - entry) / risk, 2)
        # else: tp2 collapsed onto tp1 (or wrong side) -> emit no tp2.
    return out


# SHARED — killzone membership test. Single source of truth for killzone
# scoring (used by run_scorecard on the OB candle) AND the live "approaching
# during killzone" info line. Mirrors Phase2._ob_in_killzone_label's window
# math, but returns a plain bool and lives in the lower layer so the scorer
# (and the backtest) can call it without importing Phase 2.
def _ts_in_killzone(ts_iso, killzones_utc):
    """True if the H1 candle at ts_iso (UTC ISO) overlaps any killzone window.
    killzones_utc = [["07:00","10:00"], ...] in UTC. Treats the candle as a
    1-hour block. Returns False on any missing/bad input (never raises)."""
    if not ts_iso or not killzones_utc:
        return False
    try:
        dt = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        hour_start = dt.hour * 60 + dt.minute
        hour_end = hour_start + 60
        for w in killzones_utc:
            if not isinstance(w, (list, tuple)) or len(w) != 2:
                continue
            try:
                sh, sm = (int(x) for x in str(w[0]).split(":"))
                eh, em = (int(x) for x in str(w[1]).split(":"))
            except (ValueError, AttributeError):
                continue
            start_min = sh * 60 + sm
            end_min = eh * 60 + em
            if hour_start < end_min and hour_end > start_min:
                return True
        return False
    except Exception:
        return False


# PHASE 2 ONLY — sweep quality scorecard. Called only by Phase2_Alert_Engine.py.
def run_scorecard(bias, df_h1, ob, fvg, current_price, pair_conf=None):
    bos_tag = ob.get('bos_tag', 'BOS')
    pair_type = pair_conf.get('pair_type', 'forex') if pair_conf else 'forex'

    # Structure score grid (max 4). The v2 engine emits exactly THREE
    # structural event types — CHoCH, BOS, Range BOS — with NO Major/Minor
    # tier (see smc_radar._event_label). The grid maps directly onto them:
    #   CHoCH                       -> 4  (trend reversal confirmed)
    #   Range BOS                   -> 4  (clean break-off through the H4
    #                                      dealing-range wall — reversal-grade)
    #   plain BOS #1-2 since CHoCH  -> 3  (early continuation, smart money still loading)
    #   plain BOS #3 .. caution-1   -> 2  (mid-trend)
    #   plain BOS >= caution        -> 1  (exhausted)
    # Caution thresholds reflect typical pair behaviour: forex mean-reverts
    # faster, indices sustain trends longer.
    bos_tier = ob.get('bos_tier', 'BOS')
    if bos_tag == 'CHoCH':
        # CHoCH tiering (2026-06): a CHoCH that reversed FROM the premium/discount
        # extreme of the H4 range is the textbook reversal -> 4. A mid-range CHoCH
        # is still a confirmed reversal (so it must rank >= an early continuation
        # BOS, which is 3), but it lacks the premium/discount location quality -> 3.
        # reversal_pct == 1.0 is the from-zone flag carried on the OB by Phase 1.
        choch_from_zone = float(ob.get('reversal_pct') or 0.0) >= 1.0
        bd = {"structure": 4 if choch_from_zone else 3}
    elif bos_tier == 'Range':
        # Range BOS: broke through the H4 dealing range wall. Higher-conviction
        # than a plain BOS — score same as CHoCH entry (4).
        bd = {"structure": 4}
    else:
        bos_seq = ob.get('bos_sequence_count', 1)
        caution_threshold = {'forex': 3, 'index': 5, 'commodity': 4}.get(pair_type, 3)
        if bos_seq >= caution_threshold:
            structure_score = 1
        elif bos_seq <= 2:
            structure_score = 3
        else:
            structure_score = 2
        bd = {"structure": structure_score}

    # ------------------------------------------------------------------
    # Sweep — H1-only (M15 removed 2026-05-26 in H1-only migration).
    #   H1: consumed from ob['sweep_observed'] (frozen by Phase 1 at OB
    #       formation). Phase 2 does NOT re-detect; Phase 1 is source of truth.
    #
    # If the H1 snapshot is missing from the zone, treat as zero — this is a
    # schema-drift signal worth investigating, not a reason to fabricate one.
    # ------------------------------------------------------------------
    ob_ts_iso = ob.get('ob_timestamp')

    sweep_obs_snapshot = ob.get('sweep_observed')
    if isinstance(sweep_obs_snapshot, dict) and sweep_obs_snapshot.get('exists'):
        # Consume P1's frozen H1 sweep. We also carry the sweep TIMESTAMP
        # because P1's sweep_idx points into P1's H1 dataframe, NOT P2's
        # (different fetches, indices don't align). Chart rendering must
        # resolve the candle by timestamp on P2's frame.
        h1_sweep = {
            'score':               float(sweep_obs_snapshot.get('score', 0.0)),
            'tier':                sweep_obs_snapshot.get('tier', 'none'),
            'price':               sweep_obs_snapshot.get('price'),
            'sweep_idx':           sweep_obs_snapshot.get('sweep_idx'),
            'sweep_timestamp_iso': sweep_obs_snapshot.get('timestamp'),
            'tf':                  'H1',
            'components':          sweep_obs_snapshot.get('components', {
                'base': 0.0, 'equal_levels': 0.0, 'equal_levels_matches': 0,
                'rejection': 0.0, 'wick_body_ratio': 0.0,
            }),
            'hours_before_anchor': sweep_obs_snapshot.get('hours_before_anchor'),
        }
    else:
        h1_sweep = {
            'score': 0.0, 'tier': 'none', 'price': None, 'sweep_idx': None,
            'sweep_timestamp_iso': None, 'tf': 'H1',
            'components': {'base': 0.0, 'equal_levels': 0.0, 'equal_levels_matches': 0,
                           'rejection': 0.0, 'wick_body_ratio': 0.0},
            'hours_before_anchor': None,
        }

    chosen_sweep = h1_sweep
    # Non-JPY forex collapse: sweep is presence-only (1.0 if a qualifying
    # sweep exists, else 0.0). Equal-levels and rejection-quality components
    # are detected and rendered but do NOT add points on these pairs.
    # Rationale: spot forex has no centralized stop pool, so a qualifying
    # sweep's *fact* carries some signal but its quality grading is noise.
    # JPY keeps full 3.0 (BoJ levels, carry-flow stops are real). Gold/NAS
    # keep full 3.0 (indices/commodities behave like the original SMC model).
    pair_name_for_sweep_scoring = pair_conf.get('name', '') if pair_conf else ''
    is_non_jpy_forex = (pair_type == 'forex' and 'JPY' not in pair_name_for_sweep_scoring)
    if is_non_jpy_forex:
        bd["sweep"] = 1.0 if chosen_sweep['tier'] != 'none' else 0.0
    else:
        bd["sweep"] = chosen_sweep['score']
    sweep_price  = chosen_sweep['price']
    sweep_tf     = chosen_sweep['tf']

    # FVG — H1 only (2026-05-26 scoring rewrite — max 2):
    # H1 FVG = macro displacement at OB formation, the structural signal.
    #
    # Per-mitigation scoring (max 2):
    #   pristine                       -> 2  (fresh imbalance still intact)
    #   partial mitigation             -> 1  (institutional intent partly filled)
    #   fully mitigated (was_detected) -> 0  (historical evidence only -- not scored)
    #   never detected                 -> 0
    #
    # Grading rationale: FVG is evidence of institutional displacement at OB
    # formation. We score current usability: only currently-present FVGs count.
    # A "ghost" FVG (was_detected=True but already filled) carries no live edge.
    def _grade_single(fvg_obj, pristine_pts, partial_pts):
        if not fvg_obj or not fvg_obj.get('exists'):
            return 0
        mit = fvg_obj.get('mitigation', 'pristine')
        if mit == 'partial':
            return partial_pts
        return pristine_pts

    fvg_h1 = fvg.get('h1') if isinstance(fvg, dict) else None

    # Backward-compat: if caller passes a single FVG object (legacy shape with
    # 'exists' at top level), treat it as H1.
    if fvg and isinstance(fvg, dict) and 'exists' in fvg and fvg_h1 is None:
        fvg_h1 = fvg

    bd["fvg"] = _grade_single(fvg_h1, 2, 1)
    # Freshness — binary (2026-05-26 scoring rewrite — max 1):
    # Phase 1 invalidates OBs at 3 touches; we only ever see touches=0,1,2.
    #   0 touches  -> 1 (pristine; institutional order untouched)
    #   1+ touches -> 0 (used; smart money interest diminishes after first tap)
    # The previous half-credit at 1 touch muddied the signal. A zone is either
    # untouched or it isn't.
    touches = int(ob.get('touches', 0))
    bd["freshness"] = 1 if touches == 0 else 0

    # Premium / Discount — REMOVED FROM SCORING (May 2026 overhaul).
    # Geometry is consumed from Phase 1's snapshot on the zone — Phase 1 is
    # the single source of truth for dealing range. Phase 2 does NOT redraw.
    # If the snapshot is missing (legacy zone written before this change),
    # fall back to a one-shot compute so the email still renders; weekly
    # review picks up the missing-snapshot signal via diagnostics.
    proximal = float(ob['proximal_line'])
    dr = ob.get('dealing_range') if isinstance(ob, dict) else None
    if not isinstance(dr, dict) or not dr.get('valid'):
        # Legacy fallback — compute once. Log via the missing-snapshot path
        # so we can detect schema drift.
        h1_atr_val = compute_atr(df_h1)
        dr = get_dealing_range(ob, df_h1, h1_atr_val,
                               pair_conf=pair_conf, current_price=proximal)
    pd_position = None
    if dr.get("valid"):
        rng_width = dr["range_high"] - dr["range_low"]
        if rng_width > 0:
            pd_position = (proximal - dr["range_low"]) / rng_width
    bd["pd"] = 0.0  # PD removed from scoring; display-only (trader decision).
    # Killzone — RE-ADDED to scoring 2026-06. There is NO hard killzone filter in
    # the live path: alerts fire at all hours, so OB-in-killzone is a real, varying
    # confluence (the old "the filter drops every off-session alert" comment was
    # false and has been removed). We score the OB CANDLE's killzone membership —
    # i.e. was the order block formed during institutional (London/NY) hours, a
    # zone-quality signal. The entry/approach-time killzone is shown as INFO only
    # (it flickers per scan and the limit's fill time is unknown), never scored.
    kz_windows = pair_conf.get("killzones_utc") if pair_conf else None
    bd["killzone"] = 1.0 if _ts_in_killzone(ob.get("ob_timestamp"), kz_windows) else 0.0

    # Macro removed from scorecard. Still surfaced as email-only context.

    return {
        "total": round(sum(bd.values()), 1),
        "breakdown": bd,
        "sweep_price": sweep_price,
        "sweep_tf": sweep_tf,
        "sweep_idx": chosen_sweep.get('sweep_idx'),
        # Authoritative for chart rendering: timestamp survives the cross-
        # phase / cross-fetch boundary, idx does not.
        "sweep_timestamp_iso": chosen_sweep.get('sweep_timestamp_iso'),
        "sweep_tier": chosen_sweep['tier'],
        "sweep_components": chosen_sweep['components'],
        "sweep_hours_before_ob": chosen_sweep['hours_before_anchor'],
        "dealing_range": dr,
        "pd_position": pd_position
    }


# PHASE 2 ONLY — generates scorecard HTML rows for Phase 2 alert email.
def generate_scorecard_rows(bias, breakdown, ob, sweep_price, sweep_tf, pair_conf,
                            dealing_range=None, fvg_source=None, pd_position=None,
                            sweep_tier=None, sweep_components=None,
                            sweep_hours_before_ob=None, fvg=None):
    """
    Return list of (label, score, max_score, status, explanation) for email rendering.

    Scorecard maxima (killzone +1 re-added 2026-06):
      Non-JPY forex (EURUSD, NZDUSD, USDCHF):
        Structure 4 | Sweep 1 (presence-only) | FVG 2 | Freshness 1 | Killzone 1
        Total = 9.
      JPY / Gold / NAS:
        Structure 4 | Sweep 3 (quality-graded) | FVG 2 | Freshness 1 | Killzone 1
        Total = 11.
    The forex/non-forex sweep asymmetry is intentional: forex pairs lack
    centralized stop pools so sweep quality is noise; presence alone signals.
    Gold/NAS/JPY have real institutional stop levels so quality matters.
    No confidence gate — every zone passing proximity + still-valid alerts.
    PD is rendered as a display-only row (max_score = 0) showing range
    geometry and PD% so the trader can use it for judgement, but it
    contributes no points. Macro removed from scoring; still rendered
    as email-only context block.

    Sweep row shows ✓/✗ based on PRESENCE only (base component). Detailed
    breakdown rendered separately in build_sweep_breakdown_html banner.
    """
    dp = _dp(pair_conf)
    rows = []

    # 1. Structure — pair-aware BOS sequence + event type (CHoCH / BOS /
    #    Range BOS; v2 has no Major/Minor tier).
    s = breakdown.get("structure", 0)
    bos_seq = ob.get('bos_sequence_count', 1)
    bos_tag_local = ob.get('bos_tag', 'BOS')
    bos_tier_local = ob.get('bos_tier', 'BOS')
    bos_count_maxed = bool(ob.get('bos_count_maxed', False))
    seq_str = f"#{bos_seq}+" if bos_count_maxed else f"#{bos_seq}"
    if bos_tag_local == 'CHoCH':
        choch_from_zone = float(ob.get('reversal_pct') or 0.0) >= 1.0
        if choch_from_zone:
            rows.append(("Structure", s, 4, "ok",
                         "CHoCH from premium/discount extreme — textbook reversal."))
        else:
            rows.append(("Structure", s, 4, "warn",
                         "CHoCH (mid-range) — confirmed reversal, but not from a range extreme."))
    elif bos_tag_local == 'BOS' and bos_tier_local == 'Range':
        rows.append(("Structure", s, 4, "ok",
                     f"Range BOS {seq_str} — broke through H4 dealing range wall with displacement."))
    elif s >= 3:
        rows.append(("Structure", s, 4, "ok",
                      f"Early continuation (BOS {seq_str} since last CHoCH) — smart money still loading."))
    elif s >= 2:
        rows.append(("Structure", s, 4, "warn",
                      f"Mid-trend continuation (BOS {seq_str} since last CHoCH)."))
    elif s >= 1:
        rows.append(("Structure", s, 4, "fail",
                      f"Late continuation (BOS {seq_str} since last CHoCH) — trend may be exhausted."))
    else:
        rows.append(("Structure", s, 4, "fail", "No confirmed BOS or CHoCH."))

    # 2. Liquidity Sweep — scorecard shows PRESENCE only.
    # Quality breakdown rendered separately above Macro Context in email.
    # Max is 1 (presence-only) for non-JPY forex, 3 (quality-graded) elsewhere.
    s = breakdown.get("sweep", 0)
    comps = sweep_components or {}
    presence = comps.get('base', 0.0)
    pair_name_for_row = pair_conf.get('name', '') if pair_conf else ''
    pair_type_for_row = pair_conf.get('pair_type', 'forex') if pair_conf else 'forex'
    sweep_max = 1 if (pair_type_for_row == 'forex' and 'JPY' not in pair_name_for_row) else 3
    if presence > 0 and sweep_price is not None:
        rows.append(("Liquidity Sweep", s, sweep_max, "ok",
                     f"{sweep_tf} sweep confirmed at {sweep_price:.{dp}f}. See breakdown below charts."))
    else:
        rows.append(("Liquidity Sweep", s, sweep_max, "fail",
                     "No qualifying sweep within recency window before the OB."))

    # 3. FVG — H1 only (max 2). pristine 2 | partial 1 | none/mitigated 0
    s = breakdown.get("fvg", 0)
    fvg_h1 = fvg.get('h1') if isinstance(fvg, dict) else None
    # Legacy single-FVG fallback (treat as H1)
    if fvg and isinstance(fvg, dict) and 'exists' in fvg and fvg_h1 is None:
        fvg_h1 = fvg

    def _state(f):
        if not f or not f.get('exists'):
            # was_detected + mitigation tells us if ghost was seen
            if f and f.get('was_detected') and f.get('mitigation') == 'full':
                return "fully mitigated"
            return "absent"
        return "partial" if f.get('mitigation') == 'partial' else "pristine"

    h1_state = _state(fvg_h1)
    desc = f"H1 FVG {h1_state}."

    if s >= 2:
        rows.append(("FVG", s, 2, "ok",
                     f"{desc} Fresh institutional displacement at OB formation."))
    elif s >= 1:
        rows.append(("FVG", s, 2, "warn",
                     f"{desc} Displacement present but partly mitigated."))
    else:
        rows.append(("FVG", s, 2, "fail",
                     f"{desc} No qualifying displacement (absent or fully filled)."))
    # 4. Freshness — binary (max 1). Phase 1 invalidates at 3 touches.
    s = breakdown.get("freshness", 0)
    touches = int(ob.get('touches', 0))
    if touches == 0:
        rows.append(("Freshness", s, 1, "ok",
                     "Pristine — zone untouched since it was formed."))
    elif touches == 1:
        rows.append(("Freshness", s, 1, "fail",
                     "Tested 1x since formation — institutional order partly filled."))
    elif touches == 2:
        rows.append(("Freshness", s, 1, "fail",
                     "Tested 2x since formation — one more touch invalidates."))
    else:
        rows.append(("Freshness", s, 1, "fail",
                     f"Tested {touches}x since formation — zone fatigued (legacy)."))

    # 5. Premium / Discount — DISPLAY ONLY (no longer contributes points).
    # Geometry (range, equilibrium, %) is surfaced for trader judgement.
    # Tentative-range flag rendered when applicable.
    dr_src = ""
    dr_tag = ""
    chop_tag = ""
    if dealing_range and dealing_range.get("valid"):
        src_raw = dealing_range.get("source", "structural")
        if "fallback" in src_raw:
            src_label = "FALLBACK — no recent BOS/CHoCH"
        elif "tentative" in src_raw:
            src_label = "tentative — one wall not yet swing-confirmed"
        elif "structural" in src_raw:
            src_label = "structural"
        elif "legacy" in src_raw:
            src_label = "legacy window"
        else:
            src_label = src_raw
        dr_src = (f"Range: {dealing_range['range_low']:.{dp}f}"
                  f"–{dealing_range['range_high']:.{dp}f}, "
                  f"EQ: {dealing_range['equilibrium']:.{dp}f}, "
                  f"src: {src_label}.")
        if dealing_range.get("tentative"):
            dr_tag = " (tentative range — wall pending swing confirmation)"
    if dealing_range and dealing_range.get("chop_flag"):
        chop_tag = (" \u26a0\ufe0f Rapid CHoCH within 5 candles of prior event — "
                    "possible ranging conditions, low conviction.")

    pd_pct_str = f"{pd_position * 100:.0f}%" if pd_position is not None else "n/a"

    if not dealing_range or not dealing_range.get("valid"):
        rows.append(("Premium / Discount", 0.0, 0.0, "info",
                      f"Dealing range not available — skipped (display only).{chop_tag}"))
    else:
        if bias == "LONG":
            zone_label = ("very deep discount" if pd_position is not None and pd_position <= 0.25
                          else "deep discount"  if pd_position is not None and pd_position <= 0.35
                          else "mid discount"   if pd_position is not None and pd_position <= 0.45
                          else "above equilibrium")
        else:
            zone_label = ("very deep premium" if pd_position is not None and pd_position >= 0.75
                          else "deep premium"  if pd_position is not None and pd_position >= 0.65
                          else "mid premium"   if pd_position is not None and pd_position >= 0.55
                          else "below equilibrium")
        rows.append(("Premium / Discount", 0.0, 0.0, "info",
                      f"H1 OB proximal at {pd_pct_str} of H1 dealing range ({zone_label}). {dr_src}{dr_tag}{chop_tag}"))
    
    # 6. Killzone — OB formed during the pair's institutional killzone (max 1).
    # Re-added to scoring 2026-06 (no hard killzone filter exists, so this varies).
    kz = breakdown.get("killzone", 0.0)
    if kz >= 1.0:
        rows.append(("Killzone", int(kz), 1, "ok",
                     "OB formed inside a killzone window — institutional hours."))
    else:
        rows.append(("Killzone", int(kz), 1, "warn",
                     "OB formed outside killzone hours."))

    return rows


# PHASE 2 ONLY — setup classifier. A thin layer over fields ALREADY computed:
# recognises when the confluences line up into a named, textbook SMC pattern and
# returns a badge + a one-line mentor note. No detection, no new data. The score
# is a linear sum (it can't see HOW the ingredients relate); this encodes the
# interaction logic the sum misses — WHERE the event happened (range extreme),
# WHAT sequence it's in (first leg vs late), and WHICH direction vs trend/range.
def classify_setup(ob, pd_position, trend_alignment):
    """Return (name, note, kind) for a recognised setup, or (None, None, None).

    kind is 'premium' (high-conviction, green) or 'caution' (avoid-ish, red).
    Pure — reads only fields on the OB plus the PD position. Edge guards:
      - pd_position may be None (range invalid) -> the location test fails
        CLOSED for the positive 'First Pullback' badge (we never claim "cheap
        side" we can't prove), and is simply skipped for the caution.
      - reversal_pct is only meaningful on a CHoCH; the from-zone test is only
        reached on CHoCH events.
      - sweep TIER is graded for every pair (the non-JPY-forex presence-only
        collapse affects the SCORE, not the tier), so the tier test is valid
        across instruments.
    """
    bos_tag   = ob.get('bos_tag')
    bos_tier  = ob.get('bos_tier')
    direction = ob.get('direction')
    touches   = int(ob.get('touches', 0) or 0)
    seq       = int(ob.get('bos_sequence_count', 0) or 0)
    fvg       = ob.get('fvg') or {}
    fvg_exists = bool(fvg.get('exists'))
    fvg_mit    = fvg.get('mitigation')
    sweep      = ob.get('sweep_observed') or {}
    sweep_exists = bool(sweep.get('exists'))
    sweep_tier   = (sweep.get('tier') or 'none')
    from_zone    = float(ob.get('reversal_pct') or 0.0) >= 1.0

    # --- A+ "Reversal at the Wall" -------------------------------------------
    # CHoCH that reversed from the range extreme, with a real sweep, an untouched
    # zone, and a live gap. Every ingredient tells the SAME story: smart money
    # loaded at the extreme. The richest SMC reversal there is.
    if (bos_tag == 'CHoCH' and from_zone
            and sweep_exists and sweep_tier in ('textbook', 'decent')
            and touches == 0
            and fvg_exists and fvg_mit != 'full'):
        return ("A+ Reversal at the Wall",
                "Liquidity was swept at the range extreme, structure flipped, and "
                "the zone is untouched with a live gap. This is the textbook SMC "
                "reversal — the cleanest setup the system flags.",
                "premium")

    # --- A "First Pullback" ---------------------------------------------------
    # The first BOS after a CHoCH is the youngest, least-crowded point of a new
    # trend — maximum runway. Require entry from the CHEAP side (discount for
    # longs / premium for shorts); pd unknown => no badge (we don't guess).
    pd_cheap_side = (pd_position is not None and (
        (direction == 'bullish' and pd_position <= 0.5) or
        (direction == 'bearish' and pd_position >= 0.5)))
    if (bos_tag == 'BOS' and bos_tier != 'Range'
            and seq == 1
            and trend_alignment == 'with_trend'
            and touches == 0
            and fvg_exists and fvg_mit == 'pristine'
            and pd_cheap_side):
        return ("A First Pullback",
                "The first pullback of a brand-new trend, entering from the "
                "discounted side with a fresh, untouched gap — maximum runway and "
                "minimum trend-exhaustion risk.",
                "premium")

    # --- Caution "Late-Trend Chase" ------------------------------------------
    # Third-or-later leg of a mature trend, entered from the EXPENSIVE side.
    pd_expensive_side = (pd_position is not None and (
        (direction == 'bullish' and pd_position >= 0.5) or
        (direction == 'bearish' and pd_position <= 0.5)))
    if bos_tag == 'BOS' and seq >= 3 and pd_expensive_side:
        return ("Caution: Late-Trend Chase",
                "This is the third-or-later leg of a mature trend, entered from "
                "the expensive side — you may be buying exactly what smart money "
                "is distributing into. Treat with suspicion.",
                "caution")

    return (None, None, None)


# PHASE 3 ONLY — detects M5 CHoCH inside H1 zone bounds (Phase 3 entry trigger).
def detect_ltf_choch(df_m5, bias, bounds):
    """
    Detect M5 CHoCH where the BREAK LEVEL is near the HTF zone.

    Grace band rationale: on Gold/NAS the bounce inside an H1 OB tap commonly
    pokes a few ticks above proximal (LONG) before rolling over to form the
    swing low. That bounce print IS the swing high that CHoCH must break.
    Strict zone-bound rejects it; 0.75x M5 ATR catches genuine
    wick-just-above-proximal swings without admitting swings clearly outside
    the reaction context.

    Logic:
    - Scan M5 swings across the full window at lookback=3.
    - LONG: take the single most recent M5 swing high. If it sits within
      zone_min to zone_max + 0.75 * M5 ATR, check for break. If not, no fire.
    - SHORT: mirror — single most recent swing low.
    - Break = current close crosses the swing level and previous close was on the other side.
    """
    if df_m5 is None or len(df_m5) < 10:
        return {"fired": False, "level": None}

    # Swings across full window — no bounds filter.
    # Pass min_leg_atr_mult=None: the H1-tuned MIN_LEG_ATR_MULT does not apply
    # to M5 (different volatility magnitude). Phase 3 is currently dormant; if
    # M5 trading is reactivated, calibrate a separate M5 multiplier.
    swings = get_swing_points(df_m5, lookback=3, min_leg_atr_mult=None)
    if len(swings) < 1:
        return {"fired": False, "level": None}

    C = df_m5['Close'].values
    m5_atr = compute_atr(df_m5) or 0.0

    # Grace band: 0.75x M5 ATR above (LONG) or below (SHORT) the zone
    grace_mult = 0.75
    zone_max = bounds['max']
    zone_min = bounds['min']

    if bias == 'LONG':
        long_grace_top = zone_max + grace_mult * m5_atr
        # Only the single most recent swing high. If it is outside the grace
        # band the market structure is not in reaction context — no fire.
        all_highs = [s for s in swings if s['type'] == 'high']
        if not all_highs:
            return {"fired": False, "level": None}
        latest = all_highs[-1]
        if not (zone_min <= latest['price'] <= long_grace_top):
            return {"fired": False, "level": None}
        if C[-1] > latest['price'] and C[-2] <= latest['price']:
            return {"fired": True, "level": float(latest['price'])}
    elif bias == 'SHORT':
        short_grace_bottom = zone_min - grace_mult * m5_atr
        # Only the single most recent swing low.
        all_lows = [s for s in swings if s['type'] == 'low']
        if not all_lows:
            return {"fired": False, "level": None}
        latest = all_lows[-1]
        if not (short_grace_bottom <= latest['price'] <= zone_max):
            return {"fired": False, "level": None}
        if C[-1] < latest['price'] and C[-2] >= latest['price']:
            return {"fired": True, "level": float(latest['price'])}

    return {"fired": False, "level": None}


# SHARED P1+P2 — H1 break candle span for BOS distance. Phase 1 uses it for
# context tagging; Phase 2 uses it for chart annotation. Phase 3 unaffected.
def compute_h1_break_candle_span(df_h1, ob, h1_atr):
    """
    Return (start_idx, end_idx) of consecutive H1 candles from the first
    structural break candle through the candle that satisfies the ATR-based
    break threshold. Inclusive on both ends. Max 3 candles.

    A pure VISUAL helper — does not touch trading logic. Reads bos_timestamp
    (preferred) or bos_idx (fallback for same-run charts only) from `ob`.

    Cross-phase safety: Phase 1's bos_idx is NOT portable to Phase 2 because
    the rolling yfinance window shifts the df start point. We resolve the
    BOS candle's current position via bos_timestamp using locate_ob_candle_idx.

    The resolved BOS candle is always the qualifying break candle (Phase 1
    only commits state when the ATR threshold is met). We walk backward up
    to 2 candles to capture earlier closes that crossed the swing but did
    not yet meet ATR threshold — those are the "almost broke" candles the
    trader wants to see. Stops at first candle whose Close did NOT cross
    the swing in the same direction.

    Returns (start_idx, end_idx) on success, (None, None) on failure.
    """
    if df_h1 is None or len(df_h1) == 0:
        return (None, None)
    swing_price = ob.get('bos_swing_price')
    direction = ob.get('direction')
    if swing_price is None or direction not in ('bullish', 'bearish'):
        return (None, None)
    try:
        swing_price = float(swing_price)
    except Exception:
        return (None, None)

    # Prefer timestamp lookup (cross-phase safe). Fall back to bos_idx only
    # if no timestamp is stored (e.g. legacy ob entries from before this fix).
    bos_ts = ob.get('bos_timestamp')
    resolved_idx = None
    if bos_ts:
        idx, found = locate_ob_candle_idx(df_h1, bos_ts)
        if found:
            resolved_idx = idx
    if resolved_idx is None:
        # Fallback: use bos_idx (only safe in the same run that emitted ob).
        bos_idx = ob.get('bos_idx')
        if bos_idx is None:
            return (None, None)
        try:
            bos_idx = int(bos_idx)
        except Exception:
            return (None, None)
        if bos_idx < 0 or bos_idx >= len(df_h1):
            return (None, None)
        resolved_idx = bos_idx

    C = df_h1['Close'].values
    start_idx = resolved_idx
    # Walk back at most 2 candles. Include candles whose Close crossed swing
    # in the break direction (these are early-cross candles that didn't yet
    # hit ATR threshold but are part of the visual break sequence).
    for k in range(1, 3):
        j = resolved_idx - k
        if j < 0:
            break
        try:
            cj = float(C[j])
        except Exception:
            break
        if direction == 'bullish' and cj > swing_price:
            start_idx = j
        elif direction == 'bearish' and cj < swing_price:
            start_idx = j
        else:
            break
    return (start_idx, resolved_idx)
    
# SHARED P1+P2 — OB mitigation check (close beyond distal OR 3 wick touches).
# Phase 1 uses it inside determine_drop_reason(); Phase 2 uses it as a
# mitigation gate before alerting. Despite the name, this is NOT Phase 1 only.
def resolve_distal_mode(pair_conf):
    """
    Resolve the per-instrument distal-invalidation mode from a pair config dict.
    Returns 'close' (default) or 'wick'. Single resolver so Phase 1 and Phase 2
    can never disagree on the rule for a given instrument.

      - 'close': a candle must CLOSE beyond the distal to invalidate the zone.
                 A wick that pierces the distal and closes back inside is a
                 liquidity grab the OB absorbs — zone stays alive. This is the
                 documented SMC rule and the default.
      - 'wick':  a single wick to/through the distal kills the zone. Stricter;
                 retained only as a backtest-tunable experiment.

    Anything other than the literal string 'wick' resolves to 'close', so a
    missing/typo'd config value fails safe to the documented behaviour.
    """
    if not isinstance(pair_conf, dict):
        return "close"
    mode = str(pair_conf.get("distal_invalidation_mode", "close")).strip().lower()
    return "wick" if mode == "wick" else "close"


def is_ob_mitigated_phase1(direction, distal, proximal, df, start_idx,
                           end_idx=None, distal_mode="close", atr=None):
    """
    Single-source-of-truth OB mitigation check. Used by Phase 1 (canonical
    drop reason) AND Phase 2 (mid-day still-active gate before scoring).

    Replay candles in [start_idx, end_idx) and apply Phase 1 mitigation:
      - DISTAL break -> mitigated_distal_break (zone dead). Governed by
        `distal_mode` (per-instrument, see resolve_distal_mode):
          * 'close' (default): bullish C[m] < distal; bearish C[m] > distal.
            A wick alone never invalidates — it's the liquidity grab the OB
            is built to absorb.
          * 'wick': bullish L[m] <= distal; bearish H[m] >= distal. A single
            wick to the distal kills the zone. Backtest-tunable only.
      - WICK into proximal counts as a touch; 3 touches -> mitigated_three_touches.
        The proximal touch count is ALWAYS wick-based regardless of distal_mode.
        Touches are counted ONCE PER APPROACH (excursion-based, 2026-06): after a
        touch, price must pull back beyond proximal by OB_TOUCH_REARM_ATR * atr
        before another touch registers. A zone price merely sits on for hours is
        therefore ONE test, not three — fixing the per-bar over-count that used to
        delete fresh, coiling zones (and zero their freshness score).

    The `touches` returned are PROXIMAL touches only (a distal break is terminal,
    it never accrues as a touch).

    `atr` is the H1 ATR used to scale the re-arm distance. If None/<=0, the re-arm
    falls back to OB_TOUCH_REARM_ATR * OB width so the function still works.

    Args:
        direction:   'bullish' | 'bearish'.
        distal:      OB distal price.
        proximal:    OB proximal price.
        df:          H1 OHLC dataframe.
        start_idx:   first idx to inspect (exclusive of OB candle is up to caller).
        end_idx:     one past last idx to inspect; defaults to len(df).
        distal_mode: 'close' (default) | 'wick'. Resolve via resolve_distal_mode
                     so live and backtest read the SAME per-instrument value.

    Returns:
        (mitigated: bool, reason: Optional[str], touches: int)
        reason in {'mitigated_distal_break', 'mitigated_three_touches'} or None.
    """
    if df is None or len(df) == 0:
        return False, None, 0
    if end_idx is None:
        end_idx = len(df)
    start_idx = max(0, int(start_idx))
    end_idx = min(int(end_idx), len(df))
    if start_idx >= end_idx:
        return False, None, 0

    use_wick_distal = (str(distal_mode).strip().lower() == "wick")

    C = df['Close'].values.astype(float)
    H = df['High'].values.astype(float)
    L = df['Low'].values.astype(float)

    # Re-arm distance for excursion-based touch counting. Self-scaling on ATR;
    # falls back to a fraction of OB width when ATR is unavailable.
    if atr and atr > 0:
        rearm = OB_TOUCH_REARM_ATR * float(atr)
    else:
        rearm = OB_TOUCH_REARM_ATR * abs(float(proximal) - float(distal))

    touches = 0
    armed = True   # True => a fresh approach can register a touch
    for m in range(start_idx, end_idx):
        if direction == 'bullish':
            distal_broken = (L[m] <= distal) if use_wick_distal else (C[m] < distal)
            if distal_broken:
                return True, 'mitigated_distal_break', touches
            # Proximal touch is always wick-based, counted once per approach.
            if armed and L[m] <= proximal:
                touches += 1
                armed = False
            elif (not armed) and L[m] > proximal + rearm:
                armed = True   # price pulled clearly away -> next visit is a new touch
        else:
            distal_broken = (H[m] >= distal) if use_wick_distal else (C[m] > distal)
            if distal_broken:
                return True, 'mitigated_distal_break', touches
            if armed and H[m] >= proximal:
                touches += 1
                armed = False
            elif (not armed) and H[m] < proximal - rearm:
                armed = True
        if touches >= 3:
            return True, 'mitigated_three_touches', touches

    return False, None, touches


# SHARED P2+P3 — locates a candle by ISO timestamp on a fetched df. Used by
# Phase 2 and Phase 3 for charting (locating OB / sweep / C1 candles). Phase 1
# does not call this; Phase 1 uses local idx values from its own df.
def locate_ob_candle_idx(df, ob_timestamp_iso):
    """
    Find the OB candle's positional index in `df` using its absolute timestamp.
    Returns: (idx, on_chart)
      idx: integer index into df (0 <= idx < len(df)), or 0 if off-chart/not found.
      on_chart: True if the timestamp is within df's time range, False if earlier.

    Caller then clips idx against its chart-visible window (e.g. tail(50)) to
    decide whether to draw the rectangle at its true position or at the left edge.
    """
    if df is None or len(df) == 0 or not ob_timestamp_iso:
        return 0, False

    try:
        target = datetime.fromisoformat(ob_timestamp_iso)
        if target.tzinfo is not None:
            # Normalize to UTC-naive to match the candle side below (line ~2167
            # converts each df timestamp to UTC-naive). astimezone(None) would
            # convert to the LOCAL machine tz, skewing the comparison by the
            # machine's UTC offset for any tz-aware ts (e.g. gold's -04:00).
            target = target.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return 0, False

    # Build a list of naive-UTC datetimes from df's index
    try:
        for i in range(len(df) - 1, -1, -1):
            ts = df.index[i]
            if hasattr(ts, 'tz_convert') and ts.tzinfo is not None:
                ts_cmp = ts.tz_convert('UTC').tz_localize(None).to_pydatetime()
            elif hasattr(ts, 'to_pydatetime'):
                ts_cmp = ts.to_pydatetime()
                if ts_cmp.tzinfo is not None:
                    ts_cmp = ts_cmp.replace(tzinfo=None)
            else:
                ts_cmp = ts
            # Match when candle's open time equals (or is just before) target
            if ts_cmp <= target:
                return i, True
        return 0, False
    except Exception:
        return 0, False
