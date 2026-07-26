"""Session high/low SWEEP + BREAK — backtest study columns (SESSION_SWEEP_STUDY_SPEC).

"Before this trade filled, did price grab (sweep) or clear (break) the high/low
of a prior trading session — Asia / London / NY?" A session H/L is a known pool
of stop orders; the vets say the effect is PAIR-SPECIFIC (London sweeps the Asian
range on EURUSD/GBP; JPY moves DURING Asia), so this module stamps the raw fact
per trade and the study slices it per pair. It does NOT pool pairs.

OBSERVATION ONLY. Nothing here gates, scores, or filters a trade (same discipline
as pool_builder / weekly_pd / eq_pools). The backtest stamps the
SESSION_LEVEL_FEATURE_COLUMNS onto trades.csv for the edge engine to judge. There
is no live consumer and no email line — this is a measure-first study.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY A FRESH BACKTEST MODULE AND NOT smc_detector._session_hl_until
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  This module is the BACKTEST sweep/break study: it classifies a session H/L as
  swept/broken and stamps trade-row columns. smc_detector._session_hl_until
  (smc_detector.py:1195) is the LIVE Phase-1 email badge — it only measures a
  session H/L for a text tag; it does not classify sweep/break and has no place in
  the row build. So the two are separate JOBS, not a duplicate. (History note: the
  live helper WAS DST-broken — frozen-UTC SESSION_WINDOWS_UTC — which is why this
  module was built DST-honest from the start; the live helper was fixed the same
  way on 2026-07-22, so both now share the identical per-candle-ZoneInfo mechanism
  and the market-standard 08-16 / 08-17 / 00-09 hours. See SESSION_WINDOWS_LOCAL.)

  This module resolves each candle's session by converting its timestamp to the
  session's OWN local timezone via ZoneInfo — DST resolved PER CANDLE — the same
  per-candle-date resolution the config killzones use (smc_detector.ts_in_killzone /
  resolve_killzone_windows_utc) and the live _session_hl_until now uses. One
  DST-honest session definition across live and backtest.

  SWEEP vs BREAK is decided by REUSING pool_builder.pool_status verbatim (the SAME
  N=1-confirm status machine the PD/PW daily/weekly pools use) — one concept, one
  implementation. We feed the session level in as the pool level; `broken` -> break,
  `swept` -> sweep, `intact` -> none. No new geometry, no tolerance band: pool_status
  tests a pierce/close against the EXACT level, so it is ATR-neutral by construction
  and never distorts Gold-vs-FX (SESSION_SWEEP_STUDY_SPEC §5).

SESSION WINDOW DEFINITIONS (local, DST self-resolves; SESSION_SWEEP_STUDY_SPEC §3a)
  FULL session ranges (not the narrow killzones), each in its OWN local timezone at
  the market-standard convention, so DST is carried by the zone conversion, not us:
    - Asia   : Asia/Tokyo      00:00 -> 09:00 JST  (Japan has NO DST — stable, but
                                                    resolved through the same path
                                                    for consistency)
    - London : Europe/London   08:00 -> 16:00 local (observes BST — this is where the
                                                    ±1h UTC drift lives; the zone
                                                    conversion absorbs it)
    - New York: America/New_York 08:00 -> 17:00 local (observes EDT)
  These are the standard cash-session hours, NOT the simulator's internal NY-hour
  bucket blocks (h1_only_simulator._session_from_ny_hour uses London 02-08 NY / NY
  08-16 NY for its session TAG columns — a coarser bucketing for a different job).
  This study measures the real session H/L, so it uses the real session hours. The
  DST mechanism is identical (per-candle ZoneInfo), only the local hours differ.
  The tests in tests/test_session_levels.py pin the DST behaviour at these hours.

POINT-IN-TIME (no future leak; SESSION_SWEEP_STUDY_SPEC §4.3)
  Frozen at ALERT time. The session H/L pools are built only from CLOSED H1 bars
  STRICTLY BEFORE alert_ts, then classified by pool_status over those same prior
  bars. A session's H/L is used only if that session has COMPLETED before the alert
  bar (its window's local end has passed on the pool's day) — a still-forming
  session is not yet a pool. Only the MOST-RECENTLY-CLOSED completed session is
  reported (the last range a real trader was watching) — nothing older, so the pick
  is bounded and point-in-time, never a nearest-in-price search across all history.
  Same recency discipline as the PD/PW pools' previous-completed-period rule.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from pool_builder import _naive_utc_index, pool_status

# The trades.csv column set this module owns. One list, one implementation — the
# backtest row build, the reporting front_cols and the None-fallback all key off it.
SESSION_LEVEL_FEATURE_COLUMNS = (
    "session_level_event",           # 'sweep' / 'break' / 'none' (nearest session level pre-entry)
    "session_level_which",           # 'asia' / 'london' / 'ny' / 'none'
    "session_level_side",            # 'high' / 'low' / 'none'
    "session_level_pair_relevant",   # bool: is 'which' a session the pair actually trades?
)

# Per-pair relevant sessions. Sourced VERBATIM from smc_detector.PAIR_SESSION_TAGS
# (smc_detector.py:456) — the same map the live Phase-1 sweep badge uses. We do NOT
# invent a second map: one source of truth. It is a FLAG only, never a filter — every
# pair logs ALL three sessions' events, and this bool records whether the reported
# session is one PAIR_SESSION_TAGS marks relevant. The 18-yr study decides whether
# relevance matters (e.g. "does NY still work on USD pairs it isn't tagged for?"), so
# we must keep the off-tag events too. Imported lazily in _pair_relevant to avoid a
# module-load cycle (smc_detector is heavy) and to degrade to False if it moves.
# All 11 config.json instruments — mirror of smc_detector.PAIR_SESSION_TAGS. Only
# used if that live import fails; kept in lock-step so the fallback never disagrees.
_PAIR_RELEVANT_SESSIONS_FALLBACK = {
    "EURUSD": ("asia", "london"),
    "USDJPY": ("asia", "london"),
    "NZDUSD": ("asia",),
    "USDCHF": ("london",),
    "GOLD":   ("london", "ny"),
    "NAS100": ("ny",),
    "GBPUSD": ("london", "ny"),
    "AUDUSD": ("asia", "ny"),
    "USDCAD": ("ny",),
    "EURJPY": ("asia", "london"),
    "BTCUSD": ("asia", "london", "ny"),
}

# Session windows: (ZoneInfo tz, local_start_hour, local_end_hour). end is
# EXCLUSIVE. A session may wrap past local midnight (start > end) — Asia does not
# here (00->09 JST) but the classifier handles wrap defensively. DST is resolved
# per candle by converting the candle's UTC time into `tz` (ZoneInfo carries the
# offset for that calendar date), so these constant LOCAL hours are correct in
# both summer and winter. See module docstring for the NY-block cross-check.
_SESSIONS = {
    "asia":   (ZoneInfo("Asia/Tokyo"),       0, 9),   # 00:00-09:00 JST (no DST)
    "london": (ZoneInfo("Europe/London"),    8, 16),  # 08:00-16:00 local (BST-aware)
    "ny":     (ZoneInfo("America/New_York"), 8, 17),  # 08:00-17:00 local (EDT-aware)
}

_UTC = timezone.utc


def features_none():
    """All-None session-level feature dict (spelled 'none' for the categorical
    columns, matching how the study reads a no-event row). The honest value when
    history is too thin (no completed prior session in frame) or the layer errored."""
    return {
        "session_level_event": "none",
        "session_level_which": "none",
        "session_level_side": "none",
        "session_level_pair_relevant": False,
    }


def _pair_relevant(pair, sess_key):
    """True iff `sess_key` is a session `pair` actually trades, per the LIVE
    PAIR_SESSION_TAGS map (smc_detector.py:456). Reads the live map first so this
    tracks the badge; falls back to the frozen copy above only if the import fails
    or the pair is absent. A flag, never a gate."""
    if pair is None:
        return False
    try:
        import smc_detector
        tags = smc_detector.PAIR_SESSION_TAGS.get(pair)
    except Exception:
        tags = None
    if tags is None:
        tags = _PAIR_RELEVANT_SESSIONS_FALLBACK.get(pair)
    return bool(tags) and sess_key in tags


# ---------------------------------------------------------------------------
# DST-honest per-candle session assignment (the real work — SESSION_SWEEP_STUDY §3a)
# ---------------------------------------------------------------------------

def _local_hour_and_date(ts_utc_naive, tz):
    """Convert a naive-UTC timestamp to (local_hour, local_date) in `tz`, DST
    resolved for that timestamp's calendar date by ZoneInfo. `local_date` is the
    session-local calendar date, which is what groups a session's bars into one
    day-pool (a London bar at 07:00 UTC belongs to that day's London session)."""
    aware_utc = ts_utc_naive.replace(tzinfo=_UTC) if ts_utc_naive.tzinfo is None else ts_utc_naive
    local = aware_utc.astimezone(tz)
    return local.hour, local.date()


def _in_session(local_hour, start_h, end_h):
    """True if a local hour falls in [start_h, end_h) (end exclusive). Handles a
    window that wraps past local midnight (start > end)."""
    if start_h <= end_h:
        return start_h <= local_hour < end_h
    return local_hour >= start_h or local_hour < end_h


def _session_hl_pools(bars, sess_key, alert_ts_utc):
    """Every COMPLETED session-`sess_key` day-pool strictly before the alert.

    Groups the closed H1 `bars` (naive-UTC DatetimeIndex) by session-LOCAL calendar
    date, keeps only bars whose session-LOCAL hour is inside this session's window,
    and returns one (high, low) per local session-day whose window has fully closed
    before `alert_ts_utc`. Point-in-time: only bars strictly before the alert are
    passed in, AND a session-day is emitted only if its local end-hour has already
    passed (a still-forming session is not a pool yet).

    Returns a list of dicts: {high, low} — most recent day-pool last.
    """
    tz, start_h, end_h = _SESSIONS[sess_key]
    idx = bars.index
    if len(idx) == 0:
        return []

    # Per bar: (session_local_date, in_window?, high, low)
    highs = bars["High"].to_numpy()
    lows = bars["Low"].to_numpy()
    # PERF (§3.9 Stage A, 2026-07-26): the local hour+date for EVERY bar is a
    # single vectorised tz conversion of the whole naive-UTC index, NOT a per-bar
    # ts.to_pydatetime()+astimezone() call. `tz` is a ZoneInfo; pandas tz_convert
    # uses the SAME IANA tz database, so `local.hour` / `local.date` are bit-
    # identical to _local_hour_and_date() (which does astimezone(tz)) on every
    # bar, DST boundaries included. This kills the py3.14 typing/annotationlib
    # detonation the Step 0 profile showed came from per-element boxing. The
    # bucketing loop below is unchanged — it now reads precomputed arrays.
    local = idx.tz_localize("UTC").tz_convert(tz.key)
    local_hours = local.hour.to_numpy()
    local_dates = local.date  # numpy array of datetime.date, one per bar
    buckets = {}  # local_date -> [high, low]
    for i in range(len(idx)):
        lh = int(local_hours[i])
        if not _in_session(lh, start_h, end_h):
            continue
        ld = local_dates[i]
        b = buckets.get(ld)
        if b is None:
            buckets[ld] = [float(highs[i]), float(lows[i])]
        else:
            if highs[i] > b[0]:
                b[0] = float(highs[i])
            if lows[i] < b[1]:
                b[1] = float(lows[i])

    # Keep only day-pools whose session window has fully CLOSED before the alert:
    # the local end-hour on that local date, back in UTC, must be <= alert_ts_utc.
    # (end_h is exclusive, so the session's last bar opens at end_h-1 local and the
    #  window closes at end_h:00 local.)
    alert_naive = alert_ts_utc.replace(tzinfo=None) if alert_ts_utc.tzinfo is not None else alert_ts_utc
    out = []
    for ld in sorted(buckets):
        # local end instant for this session-day, in the session tz
        end_local = datetime(ld.year, ld.month, ld.day, 0, 0, tzinfo=tz)
        # add end_h hours; for a wrapping window (start>end) the close is next day —
        # not the case for the three sessions here, but keep it correct.
        end_local = end_local + pd.Timedelta(hours=end_h)
        end_utc_naive = end_local.astimezone(_UTC).replace(tzinfo=None)
        if end_utc_naive <= alert_naive:
            hi, lo = buckets[ld]
            out.append({"high": hi, "low": lo, "close_utc": end_utc_naive})
    return out


# ---------------------------------------------------------------------------
# Sweep/break classification — REUSE pool_builder.pool_status (one implementation)
# ---------------------------------------------------------------------------

def _event_for_level(bars_after_pool, level, side):
    """'sweep' / 'break' / 'none' for one session level, via pool_builder.pool_status.

    `bars_after_pool` = closed H1 bars AFTER the pool was born (the session closed),
    up to the alert. pool_status returns swept/broken/intact; we map:
      broken -> 'break', swept -> 'sweep', intact -> 'none'.
    Reused verbatim so session levels share the EXACT N=1-confirm / failed-break /
    reclaim logic the PD/PW pools use (pool_builder.py:297)."""
    st = pool_status(bars_after_pool, level, side)["status"]
    if st == "broken":
        return "break"
    if st == "swept":
        return "sweep"
    return "none"


# (2026-07-27) RECENCY-ONLY pick — the point-in-time fix.
# build_session_level_event reports the event on the MOST-RECENTLY-CLOSED completed
# session before the alert, and NOTHING older. Recency is what a trader watches: the
# last Asia / London / NY range that closed is the live pool, exactly as the PD/PW
# pools report the PREVIOUS completed day/week and nothing further back
# (pool_builder.levels_at:270). This is naturally bounded — only the last day or two
# of sessions can be "most recent" — so it is BOTH correct and O(window) fast (the
# old code's O(history^2) all-history scan was THE backtest timeout hotspot).
#
# HISTORY (why there is no lookback_days knob any more): the original code scanned
# EVERY session-day since 2008 and reported the level NEAREST IN PRICE to entry — a
# 2019 level could win purely on price coincidence, which is meaningless. That was a
# design bug (nearest-in-price across all history), not just slow. The recency pick
# below replaces it outright: same discipline as every other study column (bounded,
# point-in-time). The values CHANGE vs the old nearest-price pick — these columns have
# never been in a canonical run, so nothing downstream is invalidated.


def build_session_level_event(df_h1_prior, alert_ts, ref_price, pair=None):
    """SESSION_LEVEL_FEATURE_COLUMNS for one alert, from bars strictly before it.

    df_h1_prior : naive-UTC-indexed H1 OHLC, bars strictly BEFORE alert_ts only.
    alert_ts    : the alert bar timestamp (naive UTC).
    ref_price   : the placed entry — used ONLY as a WITHIN-SESSION tiebreak: if the
                  most-recent completed session had an event on BOTH its high and its
                  low, the side nearer to entry is reported. It never reaches across
                  sessions/history and never changes whether a level was swept/broken.
    pair        : instrument name, for the pair-relevance FLAG (PAIR_SESSION_TAGS).
                  A recorded flag only — it never filters which sessions are scanned
                  (all three always are) and never reorders the recency pick.

    RECENCY PICK (point-in-time, bounded — matches PD/PW):
      1. Across asia/london/ny, take the ONE day-pool whose window CLOSED LATEST
         before the alert (the last session a trader was watching). Nothing older is
         considered — we NEVER scan back to find an event on an earlier session.
      2. Classify that session's high and low with pool_status over the bars since it
         closed. If neither fired, the answer is 'none' (we do NOT fall back to an
         older session — that would be the "reach back for something interesting"
         disease). If exactly one side fired, report it. If BOTH fired, report the
         side nearer to ref_price.

    Pure of feed I/O beyond the frame passed in. Never raises — returns the all-
    'none' dict on any internal failure so a study bug can never kill a run row.
    """
    try:
        if df_h1_prior is None or len(df_h1_prior) == 0 or ref_price is None:
            return features_none()
        bars = _naive_utc_index(df_h1_prior)
        alert = pd.Timestamp(alert_ts)
        if alert.tzinfo is not None:
            alert = alert.tz_convert("UTC").tz_localize(None)

        # Collect every completed day-pool across the three sessions, then keep ONLY
        # the single most-recently-closed one (max close_utc). All older pools are
        # dropped unseen — recency, never a search for the nearest/most-interesting
        # older level.
        newest = None  # (close_utc, sess_key, high, low)
        for sess_key in ("asia", "london", "ny"):
            for p in _session_hl_pools(bars, sess_key, alert):
                if newest is None or p["close_utc"] > newest[0]:
                    newest = (p["close_utc"], sess_key, p["high"], p["low"])

        if newest is None:
            return features_none()

        close_utc, sess_key, hi, lo = newest
        # Bars strictly AFTER this pool closed, up to the alert. The pool is spent by
        # bars that came after the session ended — never by the session's own bars
        # (that would let the session sweep itself). bars.index is sorted, so
        # searchsorted+iloc selects the same rows as a `>= close_utc` mask, no copy.
        _pos = bars.index.searchsorted(close_utc, side="left")
        after = bars.iloc[_pos:]

        # Classify each side of THIS session only. Collect the ones that fired.
        fired = []  # (side_key, event, dist_to_entry)
        for side_key, side_arg, level in (
            ("high", "above", hi),
            ("low", "below", lo),
        ):
            event = _event_for_level(after, level, side_arg)
            if event == "none":
                continue
            fired.append((side_key, event, abs(float(ref_price) - level)))

        if not fired:
            return features_none()
        # One side fired -> that side. Both fired -> the side nearer to entry (a pure
        # within-session tiebreak; ref_price never reaches across sessions/history).
        best = min(fired, key=lambda f: f[2])
        return {
            "session_level_event": best[1],
            "session_level_which": sess_key,
            "session_level_side": best[0],
            "session_level_pair_relevant": _pair_relevant(pair, sess_key),
        }
    except Exception as e:  # never let the session-level study kill a backtest row
        print(f"  [SESSION_LEVEL WARN] build failed at {alert_ts}: "
              f"{type(e).__name__}: {e}")
        return features_none()
