"""PD/PW liquidity pools — previous day / previous week high & low, with a
sweep/break status machine. Implements DAILY_BIAS_V4_SPEC §1.1–1.3.

OBSERVATION ONLY. Nothing here gates, scores, or filters a trade. Live callers
(smc_radar Phase 1, Phase2_Alert_Engine) surface the facts in emails and scan
logs; the backtest stamps them as trades.csv columns for the edge engine to
judge (guardrail B5/B6 discipline applies before any canonical run).

DAY BOUNDARY — the one rule everything hangs on:
  MT5 server clock = UTC+3, fixed, no DST (pinned empirically in
  backtest/mt5_data/import_mt5.py). A trading day is a SERVER calendar date,
  i.e. the UTC window [D-1 21:00, D 21:00). Grouping H1 bars by server date
  reproduces MT5's own D1 highs/lows (verified 2026-07-13: 99.2–100% exact on
  EURUSD/XAUUSD H1-vs-D1, backtest/mt5_data). A week is the Mon–Fri run of
  those server dates, keyed by its Monday.

WHY RESAMPLE INSTEAD OF FETCHING TWELVE DATA'S NATIVE 1day/1week BARS
(tested 2026-07-13 against live TD data + MT5 CSVs, both EURUSD and XAUUSD):
  - TD's native daily bars sit on a NY-close-style boundary that drifts with
    DST. They disagreed with the MT5-day resample of TD'S OWN H1 on ~10–14% of
    days — worst single-day disagreement 73 points on Gold. The backtest can
    only use the MT5 21:00 UTC boundary, so native TD dailies would make live
    levels structurally different from backtested levels.
  - Resampling our own H1 uses the SAME code path for live (Twelve Data H1)
    and backtest (MT5 H1) — the only residual difference is feed quote noise
    (known MT5-vs-TD gap: FX p50 ~1 pip).
  - DAILY_BIAS_V4_SPEC §1.1 already pinned this for the backtest ("never
    resample from a second feed; never chase NY-close DST").

STATUS MACHINE (spec §1.1), evaluated on CLOSED H1 bars since the pool was born:
  intact — never traded through.
  swept  — wick pierced the level but the bar CLOSED back on the origin side,
           OR a close beyond the level failed its N=1 confirm (next close came
           back). Fuel for reversal; the pool is spent.
  broken — H1 close beyond the level, HELD by the next H1 close (N=1 confirm,
           justified by the measured 27% first-break fakeout rate). Expansion;
           the pool is spent.
  Precedence: broken overrides swept. Once spent, never revives. An
  unconfirmed close-beyond (no next closed bar yet) reports the PRIOR status —
  never a look-ahead.
  Note the failed-break rule extends the spec's sweep wording (which covers
  only the same-bar wick reject): a close-through that reverses on the next
  close is liquidity taken + rejected, so it lands as `swept`. Approved by
  the trader 2026-07-13.

Pool birth: PD pools are born at the server-day roll (21:00 UTC) and live one
day; PW pools are born at the week roll and live the week. Bars before birth
never count against the pool.
"""

import json
import os
import tempfile
from datetime import datetime, timezone

import pandas as pd

# MT5 server clock = UTC+3, fixed, no DST. Single source for the boundary;
# proof and pinning live in backtest/mt5_data/import_mt5.py.
SERVER_UTC_OFFSET_HOURS = 3

# Wide H1 fetch for the LIVE level build. Needs to cover the previous full
# week from a late-Friday vantage point: ~5 current-week days + 5 prior-week
# days = ~240 trading bars; 450 covers ~3.5 weeks with holiday slack. One
# Twelve Data credit per call, refreshed once per server day per pair.
LIVE_LEVELS_H1_BARS = 450

# Live per-day level cache + last-seen statuses (for NEW-event markers).
POOL_LEVELS_STATE_PATH = os.path.join("state", "pool_levels.json")
POOL_STATUS_STATE_PATH = os.path.join("state", "pool_status.json")

# The trades.csv column set this module owns (DAILY_BIAS_V4_SPEC §1.3 core).
# One list, one implementation — the backtest row build and the None-fallback
# both key off it.
POOL_FEATURE_COLUMNS = (
    "day_state_at_alert",
    "pdh_status_at_alert", "pdl_status_at_alert",
    "pwh_status_at_alert", "pwl_status_at_alert",
    "dist_next_pool_above_atr", "dist_next_pool_below_atr",
    "next_pool_above_tier", "next_pool_below_tier",
    "trade_toward_pool",
    "last_sweep_age_h1", "last_sweep_tier",
)

POOL_KEYS = ("pdh", "pdl", "pwh", "pwl")
_POOL_TIER = {"pdh": "PD", "pdl": "PD", "pwh": "PW", "pwl": "PW"}
_POOL_SIDE = {"pdh": "above", "pdl": "below", "pwh": "above", "pwl": "below"}


def features_none():
    """All-None pool feature dict — the honest value when history is too thin
    (no completed prior day/week in frame) or the pool layer errored."""
    return {col: None for col in POOL_FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# Time plumbing: UTC H1 index -> server days / weeks
# ---------------------------------------------------------------------------

def _naive_utc_index(df):
    """Return df with a tz-naive UTC DatetimeIndex (accepts aware or naive).

    Both feeds arrive as UTC: Twelve Data (feed_adapter requests timezone=UTC)
    and the backtest parquet cache (import_mt5 stores server -3h, tz='UTC').

    Shape tolerance: the live Phase 1 engine passes a reset-index frame with the
    timestamp in a `Datetime` column (smc_radar.fetch_data does .reset_index()),
    while the backtest passes a DatetimeIndexed frame. Restore the index from the
    column when needed so BOTH callers funnel through the same UTC-index gate
    (fixed 2026-07-14: live pool context crashed with
    "'RangeIndex' object has no attribute 'tz'" — the pool layer never ran live).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("Datetime", "datetime"):
            if col in df.columns:
                df = df.set_index(col)
                break
    if getattr(df.index, "tz", None) is not None:
        out = df.copy()
        out.index = out.index.tz_convert("UTC").tz_localize(None)
        return out
    return df


def _server_date(ts):
    """Server calendar date (naive midnight Timestamp) for a UTC timestamp."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return (ts + pd.Timedelta(hours=SERVER_UTC_OFFSET_HOURS)).normalize()


def _week_monday(server_day):
    """Monday (server date) of the week containing server_day."""
    return server_day - pd.Timedelta(days=int(server_day.dayofweek))


def _day_start_utc(server_day):
    """Naive-UTC timestamp where this server day begins (prev day 21:00 UTC)."""
    return server_day - pd.Timedelta(hours=SERVER_UTC_OFFSET_HOURS)


def server_days(h1):
    """H1 (UTC-indexed OHLC) -> one row per server trading day: high/low/n_bars.

    Server-weekend dates (Sat/Sun) are dropped defensively — MT5 prints none,
    and the live strip below removes them before this runs.
    """
    h1 = _naive_utc_index(h1)
    day = (h1.index + pd.Timedelta(hours=SERVER_UTC_OFFSET_HOURS)).normalize()
    g = h1.groupby(day)
    out = pd.DataFrame({
        "high": g["High"].max(),
        "low": g["Low"].min(),
        "n_bars": g["High"].count(),
    })
    return out[out.index.dayofweek < 5]


def server_weeks(days):
    """Daily frame (from server_days) -> one row per week, keyed by its Monday."""
    if days.empty:
        return pd.DataFrame(columns=["high", "low", "n_days"])
    monday = days.index - pd.to_timedelta(days.index.dayofweek, unit="D")
    g = days.groupby(monday)
    return pd.DataFrame({
        "high": g["high"].max(),
        "low": g["low"].min(),
        "n_days": g["high"].count(),
    })


def drop_forming(df, now_utc=None):
    """Drop the last H1 bar if it is still forming at `now_utc`.

    A bar stamped T covers [T, T+1h): it is closed only once now >= T+1h.
    Self-contained so Phase 2 (which has no drop_forming_bar helper in scope)
    and the live pool path share one rule.
    """
    if df is None or len(df) == 0:
        return df
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_ts = pd.Timestamp(now_utc)
    if now_ts.tzinfo is not None:
        now_ts = now_ts.tz_convert("UTC").tz_localize(None)
    last_ts = pd.Timestamp(df.index[-1])
    if last_ts.tzinfo is not None:
        last_ts = last_ts.tz_convert("UTC").tz_localize(None)
    if now_ts < last_ts + pd.Timedelta(hours=1):
        return df.iloc[:-1]
    return df


def strip_for_pools(df, is_gold=False):
    """Closed-market strip for the RAW live level fetch, in SERVER time.

    feed_adapter.fetch_h1's strip works in UTC weekdays (chart parity with the
    engines' frames) — but that drops Sunday-evening UTC bars, which are the
    first bars of MONDAY'S server day, and keeps Friday 21:00+ UTC bars, which
    fall on server-Saturday (MT5 prints neither). For level building we strip
    by SERVER date so day highs/lows aggregate exactly the bars MT5 would:
      - drop server-Sat/Sun dates;
      - gold only: drop the 00:00 UTC weekday pad bar (MT5 gold's daily
        maintenance gap — same rule as feed_adapter);
      - drop runs of >= 2 consecutive flat bars (Twelve Data holiday filler —
        same proven rule as feed_adapter; real flat bars are always isolated).
    """
    df = _naive_utc_index(df)
    idx = df.index
    server_idx = idx + pd.Timedelta(hours=SERVER_UTC_OFFSET_HOURS)
    closed = server_idx.dayofweek >= 5
    if is_gold:
        closed = closed | ((idx.dayofweek < 5) & (idx.hour == 0))
    flat = df["High"] == df["Low"]
    if flat.any():
        prev_flat = flat.shift(1, fill_value=False)
        next_flat = flat.shift(-1, fill_value=False)
        closed = closed | (flat & (prev_flat | next_flat)).to_numpy()
    return df[~closed]


# ---------------------------------------------------------------------------
# Levels (pure) — the four pool prices at a point in time
# ---------------------------------------------------------------------------

def levels_at(h1, asof_ts=None, days=None, weeks=None):
    """PDH/PDL/PWH/PWL as of `asof_ts` (default: just after the last bar).

    Point-in-time correct: only bars strictly BEFORE asof_ts feed the values,
    and only fully COMPLETED days/weeks qualify (a day D-1 is complete once
    asof sits in day D, because days end at the 21:00 UTC roll).

    `days`/`weeks` may be passed pre-computed on the FULL frame (backtest fast
    path): selecting completed periods strictly before asof's day/week cannot
    look ahead, because those periods closed entirely in the past.

    Returns dict with pdh/pdl/pwh/pwl (floats or None), prev_day / prev_week
    (period labels), cur_day / cur_week (the running period asof sits in).
    Values are None when the frame holds no completed prior day/week.
    """
    h1 = _naive_utc_index(h1)
    if asof_ts is None:
        if len(h1) == 0:
            return _levels_none()
        asof_ts = h1.index[-1] + pd.Timedelta(hours=1)  # after last closed bar
    asof_ts = pd.Timestamp(asof_ts)
    if asof_ts.tzinfo is not None:
        asof_ts = asof_ts.tz_convert("UTC").tz_localize(None)

    if days is None:
        days = server_days(h1.loc[h1.index < asof_ts])
    if weeks is None:
        weeks = server_weeks(days)
    if days.empty:
        return _levels_none()

    cur_day = _server_date(asof_ts)
    if cur_day.dayofweek >= 5:
        # Weekend vantage point: the last trading day BEFORE the weekend is
        # the running "current" day (its pools stay live until Monday's roll).
        # Strictly-before selection keeps this look-ahead-safe even when
        # `days` was precomputed on the full backtest frame.
        past = days[days.index < cur_day]
        if past.empty:
            return _levels_none()
        cur_day = past.index[-1]
    cur_week = _week_monday(cur_day)

    prev_days = days[days.index < cur_day]
    prev_weeks = weeks[weeks.index < cur_week]

    out = _levels_none()
    out["cur_day"] = str(cur_day.date())
    out["cur_week"] = str(cur_week.date())
    if not prev_days.empty:
        d = prev_days.iloc[-1]
        out["pdh"], out["pdl"] = float(d["high"]), float(d["low"])
        out["prev_day"] = str(prev_days.index[-1].date())
    if not prev_weeks.empty:
        w = prev_weeks.iloc[-1]
        out["pwh"], out["pwl"] = float(w["high"]), float(w["low"])
        out["prev_week"] = str(prev_weeks.index[-1].date())
    return out


def _levels_none():
    return {"pdh": None, "pdl": None, "pwh": None, "pwl": None,
            "prev_day": None, "prev_week": None,
            "cur_day": None, "cur_week": None}


# ---------------------------------------------------------------------------
# Status machine (pure)
# ---------------------------------------------------------------------------

def pool_status(bars, level, side):
    """Status of one pool level over `bars` (closed H1 bars since pool birth).

    side: 'above' (a high pool — pierced upward) or 'below' (a low pool).
    Returns dict(status, swept_ts, broken_ts, last_sweep_ts).

    Rules (module docstring has the full story):
      wick through + close back        -> swept
      close beyond + next close beyond -> broken (terminal; overrides swept)
      close beyond + next close back   -> swept (failed break)
      close beyond, no next bar yet    -> prior status (no look-ahead)
    """
    status, swept_ts, broken_ts, last_sweep_ts = "intact", None, None, None
    pending_ts = None  # unconfirmed close-beyond awaiting its N=1 confirm

    if bars is None or len(bars) == 0 or level is None:
        return {"status": status, "swept_ts": None, "broken_ts": None,
                "last_sweep_ts": None}

    highs = bars["High"].to_numpy()
    lows = bars["Low"].to_numpy()
    closes = bars["Close"].to_numpy()
    index = bars.index

    for i in range(len(bars)):
        if side == "above":
            pierced = highs[i] > level
            closed_beyond = closes[i] > level
        else:
            pierced = lows[i] < level
            closed_beyond = closes[i] < level

        if pending_ts is not None:
            if closed_beyond:
                status, broken_ts = "broken", index[i]
                break  # spent — terminal, later bars are irrelevant
            # Failed break: liquidity taken and rejected -> swept.
            status = "swept"
            swept_ts = swept_ts or pending_ts
            last_sweep_ts = pending_ts
            pending_ts = None
            # fall through: this bar may itself pierce or close beyond

        if closed_beyond:
            pending_ts = index[i]
        elif pierced:
            status = "swept"
            swept_ts = swept_ts or index[i]
            last_sweep_ts = index[i]

    return {"status": status, "swept_ts": _iso(swept_ts),
            "broken_ts": _iso(broken_ts), "last_sweep_ts": _iso(last_sweep_ts)}


def _iso(ts):
    return None if ts is None else str(ts)


def day_state(pdh_status, pdl_status):
    """Six-state day classifier (spec §1.2). None if either status is missing."""
    if pdh_status is None or pdl_status is None:
        return None
    hi_spent = pdh_status in ("swept", "broken")
    lo_spent = pdl_status in ("swept", "broken")
    if not hi_spent and not lo_spent:
        return "INSIDE"
    if hi_spent and lo_spent:
        return "BOTH_SIDES"
    if hi_spent:
        return "EXPANSION_UP" if pdh_status == "broken" else "SWEPT_HIGH"
    return "EXPANSION_DOWN" if pdl_status == "broken" else "SWEPT_LOW"


# ---------------------------------------------------------------------------
# Snapshot: levels + statuses at a point in time
# ---------------------------------------------------------------------------

def snapshot(h1_closed, asof_ts=None, levels=None, days=None, weeks=None):
    """Full pool picture at `asof_ts` from closed H1 bars.

    h1_closed must contain CLOSED bars only (caller drops the forming bar).
    `levels` may be injected (live path: levels come from the wide cached
    fetch); default derives them from the same frame (backtest path).

    Returns {"levels": ..., "pools": {pdh|pdl|pwh|pwl: {level, tier, status,
    swept_ts, broken_ts, last_sweep_ts}}, "day_state": ..., "asof": ...}.
    """
    h1_closed = _naive_utc_index(h1_closed)
    if asof_ts is None:
        if len(h1_closed) == 0:
            return {"levels": _levels_none(), "pools": {}, "day_state": None,
                    "asof": None}
        asof_ts = h1_closed.index[-1] + pd.Timedelta(hours=1)
    asof_ts = pd.Timestamp(asof_ts)
    if asof_ts.tzinfo is not None:
        asof_ts = asof_ts.tz_convert("UTC").tz_localize(None)

    if levels is None:
        levels = levels_at(h1_closed, asof_ts, days=days, weeks=weeks)

    cur_day = _server_date(asof_ts)
    if cur_day.dayofweek >= 5 and levels.get("cur_day"):
        cur_day = pd.Timestamp(levels["cur_day"])
    cur_week = _week_monday(cur_day)

    # Pool-birth boundaries in UTC. Bars strictly before asof and at/after
    # birth are the pool's life so far.
    day_birth = _day_start_utc(cur_day)
    week_birth = _day_start_utc(cur_week)
    day_bars = h1_closed.loc[(h1_closed.index >= day_birth)
                             & (h1_closed.index < asof_ts)]
    week_bars = h1_closed.loc[(h1_closed.index >= week_birth)
                              & (h1_closed.index < asof_ts)]

    pools = {}
    for key in POOL_KEYS:
        level = levels.get(key)
        bars = day_bars if _POOL_TIER[key] == "PD" else week_bars
        st = (pool_status(bars, level, _POOL_SIDE[key]) if level is not None
              else {"status": None, "swept_ts": None, "broken_ts": None,
                    "last_sweep_ts": None})
        pools[key] = {"level": level, "tier": _POOL_TIER[key], **st}

    return {
        "levels": levels,
        "pools": pools,
        "day_state": day_state(pools["pdh"]["status"], pools["pdl"]["status"]),
        "asof": str(asof_ts),
    }


def trade_features(snap, ref_price, atr, direction, h1_index=None):
    """Per-trade feature dict (spec §1.3) from a snapshot.

    ref_price — the entry (backtest) or current price (live).
    atr       — H1 ATR for distance normalisation (backtest: ob['h1_atr'],
                the frozen formation ATR, matching the other *_atr columns).
    direction — 'bullish' / 'bearish' (trade_toward_pool is None otherwise).
    h1_index  — closed-bar index used to express last-sweep age in H1 bars;
                None degrades last_sweep_age_h1 to None.
    """
    out = features_none()
    if not snap or not snap.get("pools"):
        return out
    pools = snap["pools"]

    out["day_state_at_alert"] = snap.get("day_state")
    out["pdh_status_at_alert"] = pools["pdh"]["status"]
    out["pdl_status_at_alert"] = pools["pdl"]["status"]
    out["pwh_status_at_alert"] = pools["pwh"]["status"]
    out["pwl_status_at_alert"] = pools["pwl"]["status"]

    # Nearest UNSPENT (intact) pool each side of ref_price, ATR-normalised.
    # Position is by price relative to ref_price, not by the pool's high/low
    # identity. Tie at equal distance: the weekly pool outranks the daily.
    if ref_price is not None:
        above, below = [], []
        for key in POOL_KEYS:
            p = pools[key]
            if p["status"] != "intact" or p["level"] is None:
                continue
            dist = float(p["level"]) - float(ref_price)
            (above if dist >= 0 else below).append((abs(dist), p["tier"]))
        rank = {"PW": 0, "PD": 1}
        _atr = float(atr) if atr else None
        if above:
            d, tier = sorted(above, key=lambda t: (t[0], rank[t[1]]))[0]
            out["dist_next_pool_above_atr"] = round(d / _atr, 3) if _atr else None
            out["next_pool_above_tier"] = tier
        if below:
            d, tier = sorted(below, key=lambda t: (t[0], rank[t[1]]))[0]
            out["dist_next_pool_below_atr"] = round(d / _atr, 3) if _atr else None
            out["next_pool_below_tier"] = tier

        # Does the trade point at the nearest unspent pool overall?
        best_above = min(above)[0] if above else None
        best_below = min(below)[0] if below else None
        if direction in ("bullish", "bearish") and (above or below):
            if best_above is not None and (best_below is None
                                           or best_above <= best_below):
                nearest_side = "above"
            else:
                nearest_side = "below"
            out["trade_toward_pool"] = (
                (direction == "bullish" and nearest_side == "above")
                or (direction == "bearish" and nearest_side == "below"))

    # Most recent sweep event across all four pools, in H1 bars before asof.
    last_ts, last_tier = None, None
    for key in POOL_KEYS:
        ts = pools[key]["last_sweep_ts"]
        if ts is not None and (last_ts is None or ts > last_ts):
            last_ts, last_tier = ts, pools[key]["tier"]
    if last_ts is not None:
        out["last_sweep_tier"] = last_tier
        if h1_index is not None and len(h1_index):
            idx = h1_index
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            pos = idx.searchsorted(pd.Timestamp(last_ts))
            out["last_sweep_age_h1"] = int(len(idx) - 1 - min(pos, len(idx) - 1))
    return out


# ---------------------------------------------------------------------------
# Backtest fast path: per-frame cache so 18 years of rows don't re-resample
# ---------------------------------------------------------------------------

_FRAME_CACHE = {}


def _frame_key(df):
    return (id(df), len(df), str(df.index[0]), str(df.index[-1]))


def _cached_days_weeks(df_h1):
    key = _frame_key(df_h1)
    hit = _FRAME_CACHE.get(key)
    if hit is None:
        h1 = _naive_utc_index(df_h1)
        days = server_days(h1)
        weeks = server_weeks(days)
        hit = {"h1": h1, "days": days, "weeks": weeks}
        _FRAME_CACHE.clear()  # one live frame per pair at a time is enough
        _FRAME_CACHE[key] = hit
    return hit


def features_at_alert(df_h1, alert_ts, direction, ref_price, atr):
    """Backtest row-build entry point: the §1.3 columns at one alert.

    Uses only bars strictly BEFORE alert_ts (same rule as
    _closed_bars_at_alert in the simulator). days/weeks are pre-computed once
    per frame; selecting completed periods before the alert's day cannot look
    ahead (those periods closed entirely in the past). Never raises — returns
    the all-None dict on any internal failure so a pool bug can never kill a
    run row.
    """
    try:
        cached = _cached_days_weeks(df_h1)
        h1 = cached["h1"]
        ts = pd.Timestamp(alert_ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        pos = h1.index.searchsorted(ts)  # bars strictly before ts = iloc[:pos]
        if pos == 0:
            return features_none()
        lv = levels_at(h1, ts, days=cached["days"], weeks=cached["weeks"])
        # The status machine only ever looks at bars since the CURRENT week's
        # birth (PD pools live a day, PW pools a week), so hand snapshot just
        # that slice instead of copying 18 years of history per row. Sweep age
        # is measured on the same slice — identical to the full-frame answer,
        # since any sweep this day/week sits inside the current week.
        week_birth = _day_start_utc(_week_monday(_server_date(ts)))
        wb = h1.index.searchsorted(week_birth)
        closed_week = h1.iloc[wb:pos]
        snap = snapshot(closed_week, asof_ts=ts, levels=lv)
        return trade_features(snap, ref_price=ref_price, atr=atr,
                              direction=direction, h1_index=closed_week.index)
    except Exception as e:  # never let the pool layer kill a backtest row
        print(f"  [POOL WARN] features_at_alert failed at {alert_ts}: "
              f"{type(e).__name__}: {e}")
        return features_none()


# ---------------------------------------------------------------------------
# Live path: per-day cached wide fetch for levels + snapshot on engine bars
# ---------------------------------------------------------------------------

def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json_atomic(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def get_live_levels(pair_name, config_symbol, is_gold, now_utc=None):
    """PDH/PDL/PWH/PWL for a live pair, cached one entry per server day.

    Levels only change at the 21:00 UTC day roll, so the wide 450-bar fetch
    (one Twelve Data credit) runs at most once per pair per server day; every
    other scan reads state/pool_levels.json. On a fetch failure the previous
    day's cached levels are returned stamped stale=True (visible downstream)
    rather than nothing — and None only when there is no cache at all.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    today_server = str(_server_date(pd.Timestamp(now_utc)).date())

    state = _load_json(POOL_LEVELS_STATE_PATH, {})
    entry = state.get(pair_name)
    if entry and entry.get("server_day") == today_server:
        return entry

    import feed_adapter  # lazy: keeps this module import-light for the backtest
    df = feed_adapter.fetch_h1_unstripped(config_symbol,
                                          outputsize=LIVE_LEVELS_H1_BARS)
    if df is None:
        if entry:
            stale = dict(entry)
            stale["stale"] = True
            return stale
        return None

    df = strip_for_pools(df, is_gold=is_gold)
    df = drop_forming(df, now_utc)
    lv = levels_at(df, asof_ts=pd.Timestamp(now_utc))
    entry = {
        "server_day": today_server,
        "fetched_at": pd.Timestamp(now_utc).isoformat(),
        "stale": False,
        **lv,
    }
    state[pair_name] = entry
    _save_json_atomic(POOL_LEVELS_STATE_PATH, state)
    return entry


def live_pool_context(pair_name, config_symbol, df_h1, is_gold,
                      now_utc=None, mark_transitions=False):
    """One-call live entry point for Phase 1 / Phase 2.

    Levels come from the per-day cache (get_live_levels); statuses are
    evaluated on the caller's engine H1 frame (closed bars — the forming bar
    is dropped here). Returns the snapshot dict plus levels metadata, or None
    when no levels are available (cold cache + failed fetch).

    mark_transitions=True (Phase 1 only): diff the four statuses against
    state/pool_status.json and stamp `changed` per pool, so the email can
    mark sweep/break events that happened since the last scan. Phase 2 leaves
    the marker state alone.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    levels = get_live_levels(pair_name, config_symbol, is_gold, now_utc)
    if not levels or levels.get("pdh") is None:
        return None

    closed = drop_forming(_naive_utc_index(df_h1), now_utc)
    snap = snapshot(closed, asof_ts=pd.Timestamp(now_utc), levels=levels)
    snap["levels_stale"] = bool(levels.get("stale"))
    snap["h1_index_len"] = len(closed)

    if mark_transitions:
        st = _load_json(POOL_STATUS_STATE_PATH, {})
        prev = st.get(pair_name, {})
        new_state = {}
        for key in POOL_KEYS:
            cur_status = snap["pools"][key]["status"]
            # A pool is "changed" when its status differs from the last scan
            # ON THE SAME pool (same level-birth day/week); a day/week roll
            # resets the comparison (new pool, no marker).
            birth = (levels.get("prev_day") if _POOL_TIER[key] == "PD"
                     else levels.get("prev_week"))
            prev_rec = prev.get(key) or {}
            snap["pools"][key]["changed"] = bool(
                prev_rec.get("birth") == birth
                and prev_rec.get("status") is not None
                and prev_rec.get("status") != cur_status
            )
            new_state[key] = {"status": cur_status, "birth": birth}
        st[pair_name] = new_state
        try:
            _save_json_atomic(POOL_STATUS_STATE_PATH, st)
        except Exception:
            pass  # marker state is cosmetic; never fail the scan for it
    return snap


# ---------------------------------------------------------------------------
# Presentation helpers (shared by the P1 banner and the P2 email line)
# ---------------------------------------------------------------------------

# Compact status glyphs for the P1 digest (codes stay in logs/trades.csv).
# The long "what it means" phrasing moved to a one-time legend at the foot of
# the digest — repeating it on every pool line was the bulk of the P1 bloat.
_STATUS_GLYPH = {
    "intact": "✓",       # ✓ untouched (the boring default)
    "swept": "swept⚠",   # ⚠ grabbed and rejected — possible reversal fuel
    "broken": "broke→",  # → closed through and held
}

# One-time legend line for the P1 digest (printed once, not per pair).
POOL_LEGEND = ("Legend: ✓ untouched · swept⚠ = grabbed & "
               "rejected (reversal fuel) · broke→ = closed through "
               "and held · * = changed this scan.")

# Plain-English day-state phrasing (full sentence — kept for any long-form use).
_DAY_STATE_PHRASE = {
    "INSIDE": "trading inside yesterday's range",
    "EXPANSION_UP": "broke above yesterday's range and holding",
    "EXPANSION_DOWN": "broke below yesterday's range and holding",
    "SWEPT_HIGH": "yesterday's high swept — grabbed and rejected",
    "SWEPT_LOW": "yesterday's low swept — grabbed and rejected",
    "BOTH_SIDES": "both sides of yesterday's range taken — choppy day",
}

# Short day-state tag for the compact P1 digest line (no full sentence).
_DAY_STATE_TAG = {
    "INSIDE": "inside range",
    "EXPANSION_UP": "broke up, holding",
    "EXPANSION_DOWN": "broke down, holding",
    "SWEPT_HIGH": "high swept",
    "SWEPT_LOW": "low swept",
    "BOTH_SIDES": "both sides taken (choppy)",
}

# Full pool name for the P2 email (which exact level the trade points at).
_POOL_NAME = {
    "pdh": "yesterday's high (PDH)", "pdl": "yesterday's low (PDL)",
    "pwh": "last week's high (PWH)", "pwl": "last week's low (PWL)",
}


def _pool_glyph(p):
    """Compact status glyph for one pool, with the '*' new-this-scan marker.
    None if the pool has no level/status."""
    if p["level"] is None or p["status"] is None:
        return None
    g = _STATUS_GLYPH.get(p["status"], p["status"])
    star = "*" if p.get("changed") else ""
    return f"{g}{star}"


def format_pool_line(snap, dp):
    """One compact glyph line for the P1 digest, e.g.
    'PDH swept⚠ PDL ✓ · PWH ✓ PWL ✓ · inside range'.
    '*' marks a status change since the previous scan. None for an empty
    snapshot. The long meaning of each glyph lives once in POOL_LEGEND.

    `dp` is kept in the signature for caller parity; prices live in the
    logs/trades.csv columns, and the email text stays code-free.
    """
    if not snap or not snap.get("pools"):
        return None
    pools = snap["pools"]
    segments = []

    # Prior day: PDH / PDL glyphs side by side.
    pdh, pdl = _pool_glyph(pools["pdh"]), _pool_glyph(pools["pdl"])
    day_bits = []
    if pdh:
        day_bits.append(f"PDH {pdh}")
    if pdl:
        day_bits.append(f"PDL {pdl}")
    if day_bits:
        segments.append(" ".join(day_bits))

    # Prior week: PWH / PWL glyphs side by side.
    pwh, pwl = _pool_glyph(pools["pwh"]), _pool_glyph(pools["pwl"])
    wk_bits = []
    if pwh:
        wk_bits.append(f"PWH {pwh}")
    if pwl:
        wk_bits.append(f"PWL {pwl}")
    if wk_bits:
        segments.append(" ".join(wk_bits))

    if not segments:
        return None

    # Day-state as a short tag, not a sentence (full phrasing dropped —
    # the glyphs above already say what was swept/broken).
    ds = snap.get("day_state")
    if ds:
        segments.append(_DAY_STATE_TAG.get(ds, ds.lower()))

    line = " · ".join(segments)
    if snap.get("levels_stale"):
        line += "  [levels stale — using cache]"
    return line


# Plain-word status for the P1 zone-card Liquidity bullet — no glyphs, so a
# non-trader reads it with no legend to cross-reference.
_STATUS_WORD = {
    "intact": "untouched",
    "swept": "swept (grabbed & rejected)",
    "broken": "broken (closed through)",
}
_POOL_WORD = {
    "pdh": "Yesterday's high", "pdl": "Yesterday's low",
    "pwh": "Last week's high", "pwl": "Last week's low",
}


def format_pool_words(snap):
    """Plain-English P1 pool status, e.g.
    'Yesterday's high swept (grabbed & rejected); yesterday's low untouched;
    last week's high untouched; last week's low untouched.'
    No glyphs, no legend needed. None for an empty snapshot."""
    if not snap or not snap.get("pools"):
        return None
    pools = snap["pools"]
    parts = []
    for key in ("pdh", "pdl", "pwh", "pwl"):
        p = pools.get(key) or {}
        if p.get("level") is None or p.get("status") is None:
            continue
        word = _STATUS_WORD.get(p["status"], p["status"])
        parts.append(f"{_POOL_WORD[key]} {word}")
    if not parts:
        return None
    return "; ".join(parts) + "."


def _pool_key_from(tier, side):
    """The exact pool key (pdh/pdl/pwh/pwl) from its tier (PD/PW) and its
    side of price (above/below). A pool sitting above price is a high, below
    is a low — so tier + side names the level unambiguously."""
    if tier not in ("PD", "PW") or side not in ("above", "below"):
        return None
    return {("PD", "above"): "pdh", ("PD", "below"): "pdl",
            ("PW", "above"): "pwh", ("PW", "below"): "pwl"}[(tier, side)]


def format_liquidity_inference(features, bias):
    """P2 only: one plain-English "what it means for THIS trade" line, in three
    parts — the data, what it means, what to do — for a 5-year-old reader.

    TWO cases only, toward or away (no near/mid/far buckets — the ATR number
    already carries the distance):
      - toward: the nearest untouched pool sits in the trade's direction, so it
        is a natural take-profit target price is pulled toward.
      - away:   the nearest untouched pool sits BEHIND the trade, so price is
        likely to grab it FIRST — a dip/shakeout risk before any move.

    All distances are ATR from the CURRENT price (ref_price=current_price at the
    P2 call site), never the OB or the alert candle — the wording says so.

    Side selection is by trade_toward_pool, not by bias alone: a LONG that is
    running AWAY names the pool BELOW (the magnet behind it), not the pool above
    (older code named the above-pool on the away branch — the wrong level).

    Returns None when there is no untouched pool to speak to. Information only —
    never gates or scores.
    """
    if not features:
        return None
    toward = features.get("trade_toward_pool")
    if toward is None:
        return None

    long = (bias == "LONG")
    # The magnet this line is about: the pool in the trade's direction when
    # toward, the pool behind the trade when away.
    if (toward and long) or (not toward and not long):
        tier, dist, side, arrow = (features.get("next_pool_above_tier"),
                                   features.get("dist_next_pool_above_atr"),
                                   "above", "up")
    else:
        tier, dist, side, arrow = (features.get("next_pool_below_tier"),
                                   features.get("dist_next_pool_below_atr"),
                                   "below", "down")
    key = _pool_key_from(tier, side)
    name = _POOL_NAME.get(key, "the nearest untouched level")
    dist_str = f"{dist} ATR {arrow}" if dist is not None else "nearby"

    if toward:
        # Data → meaning → what to do (target).
        return (f"<b>{side.title()} the current price:</b> {name}, "
                f"{dist_str}, never touched.<br>"
                f"Price is pulled toward untouched levels to grab the stops "
                f"resting there.<br>"
                f"<b>Your {bias.lower()} points straight at it → use {name} "
                f"as your take-profit.</b>")

    # Away: the magnet is behind the trade — a shakeout risk first. The move to
    # reach it is `arrow` (down for a long's below-pool, up for a short's
    # above-pool), so the wording stays directionally correct for both sides.
    return (f"<b>{side.title()} the current price:</b> {name}, "
            f"{dist_str}, never touched.<br>"
            f"Price usually grabs an untouched level like this FIRST, before "
            f"moving the other way.<br>"
            f"<b>Your {bias.lower()} is fighting this pull → expect price to "
            f"go {arrow} to {name} first. Put your stop beyond {name}, not just "
            f"at the zone, and take less out of the move.</b>")
