"""EQH/EQL — equal-highs / equal-lows liquidity clusters.

Two or more swing highs (lows) printing within a small ATR tolerance of each
other form a CLUSTER: a shelf of resting stop liquidity just beyond the shared
level. This module detects those clusters point-in-time, tracks their
intact / swept / broken lifecycle, and emits observation-only features.

OBSERVATION ONLY. Nothing here gates, scores, or filters a trade (approved
2026-07-14, same discipline as pool_builder). Live callers surface one plain-
English line in the P1 digest and the P2 trade email; the backtest stamps the
EQ_FEATURE_COLUMNS onto trades.csv for the edge engine to judge.

SINGLE-IMPLEMENTATION REUSE (one concept, one implementation):
  - Swings: dealing_range.detect_swings at lookback 3 with
    min_leg_atr_mult=None. RAW GEOMETRY ON PURPOSE — the 1.5-ATR leg filter
    would delete exactly the small equal swings EQ clusters are made of.
    The raw pool is used for THIS module only; every other consumer of
    detect_swings keeps the ATR-filtered definition (owner decision
    2026-07-14: "nothing else should be contaminated").
  - Lifecycle: pool_builder.pool_status — the SAME wick-vs-close status
    machine PD/PW pools use (wick+close-back = swept, close+hold(N=1) =
    broken, failed break = swept, broken overrides swept, no look-ahead).
  - Frame plumbing: pool_builder._naive_utc_index / drop_forming.

CLUSTER RULES:
  - Members: confirmed same-type swings (a swing at idx i is confirmed once
    3 bars have closed after it, i.e. i <= asof_pos - 4).
  - A swing joins a cluster when its price is within EQ_TOL_ATR * atr of the
    cluster's running level AND it prints within EQ_MAX_MEMBER_GAP_BARS of
    the cluster's last member. Otherwise it seeds a new cluster.
  - Cluster level = the EXTREME of its members (max high for EQH, min low
    for EQL) — that is where the stops actually sit. The equality tolerance
    is for level IDENTITY only; sweep judgment uses the strict extreme
    (any trade beyond the extreme drains the stops parked there).
  - A cluster is real once it has >= 2 members. Pairs count (owner: "even a
    pair should be okay"); member count is logged, never gated.
  - Status bars start AFTER the last member's pivot bar (a member's own
    formation cannot sweep its own cluster; lookback-3 geometry guarantees
    the 3 bars after a pivot never pierce it).

SWEEP-REBUILD NOTE: these clusters are the liquidity REFERENCE the sweep
rebuild consumes (a sweep = trade through a ranked pool + rejection). This
module only builds and observes the reference; sweep grading stays in its own
workstream.
"""

from datetime import datetime, timezone

import pandas as pd

from dealing_range import detect_swings
from pool_builder import _naive_utc_index, drop_forming, pool_status

# Swing geometry for the raw pool (matches the system-wide H1 lookback).
EQ_SWING_LOOKBACK = 3

# Level identity: two swings are "equal" within this multiple of the H1 ATR.
# LuxAlgo ships 0.1 x ATR(200); we key off our ATR(14) — current-noise units
# beat a 200-bar average for judging whether two nearby extremes are one
# shelf. 0.20 (not 0.10) because of LIVE/BACKTEST FEED VARIANCE: MT5-vs-
# TwelveData quote gaps run p50 ~1 pip on FX, and 0.10 x a typical 10-15 pip
# H1 ATR is ~1-1.5 pips — the same size as the feed disagreement, so cluster
# membership would flip between the live feed and the backtest feed at the
# boundary. 0.20 (~2-3 pips) absorbs the median gap and matches what a
# trader eyeballs as "equal". Knob-sweep candidates: 0.10 / 0.20 / 0.30.
EQ_TOL_ATR = 0.20

# How far back (H1 bars) we look for cluster members at all. Older shelves
# are PWH/PWL-class territory already covered by the weekly pools.
EQ_LOOKBACK_BARS = 240  # ~2 trading weeks

# Max bars between CONSECUTIVE members of one cluster. Equal extremes more
# than ~a trading week apart are two separate stories, not one shelf.
EQ_MAX_MEMBER_GAP_BARS = 120

# "Stop is bait" band: the stop counts as at-risk only when the pool sits
# beyond it by no more than this many ATR. Instant-death context: the median
# SL is about one H1 bar's range (~1 ATR), so a pool within 1 ATR beyond the
# stop is inside one bar's stop-run reach.
EQ_SL_RISK_MAX_ATR = 1.0

# The trades.csv column set this module owns. One list, one implementation —
# the backtest row build, the CSV writer and the None-fallback all key off it.
EQ_FEATURE_COLUMNS = (
    "eqh_above_dist_atr", "eqh_above_size",
    "eql_below_dist_atr", "eql_below_size",
    "eq_trade_toward",
    "eq_sl_gap_atr", "eq_sl_at_risk",
    "eq_last_sweep_age_h1", "eq_last_sweep_side",
    "eq_intact_above_count", "eq_intact_below_count",
)


def features_none():
    """All-None EQ feature dict — the honest value when history is too thin,
    ATR is missing, or the EQ layer errored."""
    return {col: None for col in EQ_FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# Per-frame cache (naive frame + raw swings) — mirror of
# pool_builder._cached_days_weeks. Keyed off the CALLER's frame so the
# UTC-naive copy and the full-frame swing detection run ONCE per pair frame,
# not once per row (the naive copy alone is a 100k+-bar frame copy).
# ---------------------------------------------------------------------------

_EQ_CACHE = {}


def _frame_key(df):
    return (id(df), len(df), str(df.index[0]), str(df.index[-1]))


def _cached_frame(df):
    """{"h1": naive-UTC frame, "swings": raw-geometry lookback-3 swings}
    computed once per frame. Point-in-time selection happens later by idx
    (a lookback-3 pivot is a purely local fact — the full-frame list filtered
    to confirmed-before-asof indices equals a slice recompute)."""
    key = _frame_key(df)
    hit = _EQ_CACHE.get(key)
    if hit is None:
        h1 = _naive_utc_index(df)
        swings = detect_swings(h1, lookback=EQ_SWING_LOOKBACK,
                               min_leg_atr_mult=None)
        hit = {"h1": h1, "swings": swings}
        _EQ_CACHE.clear()  # one frame per pair at a time is enough
        _EQ_CACHE[key] = hit
    return hit


# ---------------------------------------------------------------------------
# Cluster detection (pure, point-in-time)
# ---------------------------------------------------------------------------

def _chain_clusters(swings, swing_type, tol):
    """Greedy oldest-first chaining of same-type swings into clusters.

    A swing joins the OPEN cluster when it is within `tol` of the cluster's
    running level and within EQ_MAX_MEMBER_GAP_BARS of the last member; else
    it seeds a new cluster. Running level = extreme of members so far (max
    for highs, min for lows). Only clusters with >= 2 members are returned.

    Known drift bound: membership compares against the running EXTREME, so a
    staircase of sub-tol steps can span more than one tol total — bounded in
    practice by the gap cap and logged member count.
    """
    take_max = (swing_type == "high")
    clusters, open_cl = [], None
    for s in (x for x in swings if x["type"] == swing_type):
        price, idx = float(s["price"]), int(s["idx"])
        if open_cl is not None \
                and abs(price - open_cl["level"]) <= tol \
                and (idx - open_cl["last_idx"]) <= EQ_MAX_MEMBER_GAP_BARS:
            open_cl["level"] = (max if take_max else min)(open_cl["level"], price)
            open_cl["last_idx"] = idx
            open_cl["size"] += 1
        else:
            if open_cl is not None and open_cl["size"] >= 2:
                clusters.append(open_cl)
            open_cl = {"side": swing_type, "level": price,
                       "first_idx": idx, "last_idx": idx, "size": 1}
    if open_cl is not None and open_cl["size"] >= 2:
        clusters.append(open_cl)
    return clusters


def clusters_at(df_h1, asof_pos, atr, swings=None):
    """All EQ clusters visible at bar position `asof_pos` (exclusive), with
    lifecycle status evaluated on bars strictly before asof_pos.

    df_h1 must be UTC-indexed closed bars (callers funnel through
    _naive_utc_index). Returns None when the layer cannot run (missing ATR /
    history too thin) and [] when it ran and found no cluster — the feature
    builder maps None to all-None and [] to real zero counts.
    Each cluster dict carries: side ('high'|'low'), level, size,
    first_idx, last_idx, status ('intact'|'swept'|'broken'),
    last_sweep_pos (bar position of the most recent sweep, or None).
    """
    if atr is None or atr <= 0 or df_h1 is None:
        return None  # couldn't run — distinct from "ran, found none" ([])
    n = len(df_h1)
    asof_pos = min(int(asof_pos), n)
    # Confirmed = 3 closed bars after the pivot, all before asof.
    max_swing_idx = asof_pos - (EQ_SWING_LOOKBACK + 1)
    if max_swing_idx < 0:
        return None  # too thin to even hold one confirmed swing
    lo = max(0, asof_pos - EQ_LOOKBACK_BARS)
    if swings is None:
        swings = _cached_frame(df_h1)["swings"]
    swings = [s for s in swings if lo <= s["idx"] <= max_swing_idx]
    if not swings:
        return []

    tol = EQ_TOL_ATR * float(atr)
    out = []
    for swing_type in ("high", "low"):
        for cl in _chain_clusters(swings, swing_type, tol):
            # Lifecycle on bars AFTER the last member's pivot bar, up to asof.
            # SAME status machine as PD/PW pools (pool_builder.pool_status).
            bars = df_h1.iloc[cl["last_idx"] + 1: asof_pos]
            side = "above" if swing_type == "high" else "below"
            st = pool_status(bars, cl["level"], side)
            cl["status"] = st["status"]
            ts = st["last_sweep_ts"]
            cl["last_sweep_pos"] = (
                int(df_h1.index.searchsorted(pd.Timestamp(ts)))
                if ts is not None else None)
            out.append(cl)
    return out


# ---------------------------------------------------------------------------
# Features (pure) — trade-relative view of the clusters
# ---------------------------------------------------------------------------

def features_from_clusters(clusters, ref_price, sl, direction, atr, asof_pos):
    """EQ_FEATURE_COLUMNS dict from a cluster list.

    ref_price — entry (backtest) or current price (live).
    sl        — the traded stop (sl_initial); None degrades the sl columns.
    atr       — frozen ob['h1_atr'] in the backtest (same denominator as
                every other *_atr column); live current H1 ATR for display.

    Semantics (liquidity sits BEYOND the shared extreme):
      EQH clusters = buy-stop liquidity ABOVE price -> counted when level >
      ref_price. EQL clusters = sell-stop liquidity BELOW. Intact only —
      swept/broken shelves are spent.

      eq_sl_gap_atr — signed distance from the traded stop to the nearest
      intact same-side pool BEYOND the entry-to-stop path:
        bullish: nearest intact EQL below entry, gap = (sl - level)/atr
        bearish: nearest intact EQH above entry, gap = (level - sl)/atr
      POSITIVE gap = the pool sits beyond our stop — any run to the pool
      takes our stop out first (we are the liquidity). NEGATIVE = the stop
      is tucked behind the pool (protected side).

      eq_sl_at_risk — True only when 0 < gap <= EQ_SL_RISK_MAX_ATR: the stop
      is bait sitting within one stop-run's reach of the magnet. The raw gap
      column keeps the full information either way.
    """
    out = features_none()
    if clusters is None or ref_price is None:
        return out  # layer couldn't run — every column stays None
    # Detection RAN: the counts are real zeros even when no cluster formed.
    out["eq_intact_above_count"] = 0
    out["eq_intact_below_count"] = 0
    if not clusters:
        return out
    _atr = float(atr) if atr else None
    ref = float(ref_price)

    intact = [c for c in clusters if c["status"] == "intact"]
    above = sorted((c for c in intact
                    if c["side"] == "high" and c["level"] > ref),
                   key=lambda c: c["level"] - ref)
    below = sorted((c for c in intact
                    if c["side"] == "low" and c["level"] < ref),
                   key=lambda c: ref - c["level"])

    out["eq_intact_above_count"] = len(above)
    out["eq_intact_below_count"] = len(below)

    if above:
        out["eqh_above_size"] = int(above[0]["size"])
        if _atr:
            out["eqh_above_dist_atr"] = round((above[0]["level"] - ref) / _atr, 3)
    if below:
        out["eql_below_size"] = int(below[0]["size"])
        if _atr:
            out["eql_below_dist_atr"] = round((ref - below[0]["level"]) / _atr, 3)

    # Does the trade point at the nearest intact shelf overall?
    if direction in ("bullish", "bearish") and (above or below):
        d_above = (above[0]["level"] - ref) if above else None
        d_below = (ref - below[0]["level"]) if below else None
        if d_above is not None and (d_below is None or d_above <= d_below):
            nearest_side = "above"
        else:
            nearest_side = "below"
        out["eq_trade_toward"] = (
            (direction == "bullish" and nearest_side == "above")
            or (direction == "bearish" and nearest_side == "below"))

    # Stop-vs-pool geometry (the instant-death hypothesis column).
    if sl is not None and _atr and direction in ("bullish", "bearish"):
        stop = float(sl)
        if direction == "bullish":
            pool = below[0] if below else None
        else:
            pool = above[0] if above else None
        if pool is not None:
            gap = ((stop - pool["level"]) if direction == "bullish"
                   else (pool["level"] - stop)) / _atr
            out["eq_sl_gap_atr"] = round(gap, 3)
            out["eq_sl_at_risk"] = bool(0.0 < gap <= EQ_SL_RISK_MAX_ATR)

    # Most recent sweep across ALL clusters (spent shelves included — the
    # sweep EVENT is the signal), as an age in closed H1 bars before asof.
    last_pos, last_side = None, None
    for c in clusters:
        p = c.get("last_sweep_pos")
        if p is not None and (last_pos is None or p > last_pos):
            last_pos, last_side = p, c["side"]
    if last_pos is not None:
        out["eq_last_sweep_age_h1"] = int(asof_pos - 1 - last_pos)
        out["eq_last_sweep_side"] = last_side
    return out


# ---------------------------------------------------------------------------
# Backtest entry point (mirror of pool_builder.features_at_alert)
# ---------------------------------------------------------------------------

def features_at_alert(df_h1, alert_ts, direction, entry, sl, atr):
    """Backtest row-build entry point: the EQ columns at one alert.

    Uses only bars strictly BEFORE alert_ts (same rule as
    _closed_bars_at_alert). The raw swing pool is cached once per frame;
    a lookback-3 pivot confirmed before the alert is immutable, so the
    full-frame cache filtered by idx cannot look ahead. Never raises —
    returns the all-None dict on any internal failure so an EQ bug can
    never kill a run row.
    """
    try:
        cached = _cached_frame(df_h1)
        h1, swings = cached["h1"], cached["swings"]
        ts = pd.Timestamp(alert_ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        pos = int(h1.index.searchsorted(ts))  # bars strictly before ts
        if pos == 0:
            return features_none()
        clusters = clusters_at(h1, pos, atr, swings=swings)
        return features_from_clusters(clusters, ref_price=entry, sl=sl,
                                      direction=direction, atr=atr,
                                      asof_pos=pos)
    except Exception as e:  # never let the EQ layer kill a backtest row
        print(f"  [EQ WARN] features_at_alert failed at {alert_ts}: "
              f"{type(e).__name__}: {e}")
        return features_none()


# ---------------------------------------------------------------------------
# Live entry point + plain-English email line
# ---------------------------------------------------------------------------

def live_eq_context(df_h1, atr, now_utc=None):
    """Clusters on the live engine frame (forming bar dropped here).
    Returns {"clusters": [...], "asof_pos": int} or None on thin/broken
    input. Same detection code path as the backtest."""
    try:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        closed = drop_forming(_naive_utc_index(df_h1), now_utc)
        if closed is None or len(closed) == 0:
            return None
        pos = len(closed)
        clusters = clusters_at(closed, pos, atr)
        if not clusters:
            return None
        return {"clusters": clusters, "asof_pos": pos}
    except Exception as e:
        print(f"  [EQ WARN] live_eq_context failed: {type(e).__name__}: {e}")
        return None


def format_eq_line(ctx, ref_price, atr):
    """One plain-English digest line, mirroring format_pool_line's tone.
    Prices stay out of the text (same rule as the pool line); distances are
    given in ATR because that is how the trader sizes a move. None when
    there is nothing worth saying."""
    if not ctx or not ctx.get("clusters") or ref_price is None:
        return None
    f = features_from_clusters(ctx["clusters"], ref_price=ref_price, sl=None,
                               direction=None, atr=atr,
                               asof_pos=ctx["asof_pos"])
    bits = []
    if f["eqh_above_dist_atr"] is not None:
        bits.append(f"stops above: equal highs {f['eqh_above_dist_atr']} ATR "
                    f"away ({f['eqh_above_size']} touches)")
    if f["eql_below_dist_atr"] is not None:
        bits.append(f"stops below: equal lows {f['eql_below_dist_atr']} ATR "
                    f"away ({f['eql_below_size']} touches)")
    if f["eq_last_sweep_age_h1"] is not None and f["eq_last_sweep_age_h1"] <= 24:
        side = "highs" if f["eq_last_sweep_side"] == "high" else "lows"
        bits.append(f"equal {side} were just taken ({f['eq_last_sweep_age_h1']}h "
                    f"ago) — that pool is used up, no longer a draw")
    if not bits:
        return None
    return "Equal levels: " + " · ".join(bits) + "."


def format_eq_inference(ctx, ref_price, atr, direction):
    """Direction-aware 3-part EQ bullet (data -> meaning -> what to do), used by
    the P1 zone card and shaped for a 5-year-old reader. Distances are ATR from
    the CURRENT price (ref_price). None when there is no intact cluster in the
    trade's direction to speak to. Info only — never gates or scores.

    A stack of equal highs/lows is a pile of resting stops = a magnet. In the
    trade's direction it is a take-profit target (expect a wick, not a clean
    stop); behind the trade it is a shakeout risk (give the stop room).
    """
    if not ctx or not ctx.get("clusters") or ref_price is None:
        return None
    f = features_from_clusters(ctx["clusters"], ref_price=ref_price, sl=None,
                               direction=direction, atr=atr,
                               asof_pos=ctx["asof_pos"])
    long = (direction == "bullish")
    # The cluster in the trade's direction (a target) if one exists, else the
    # cluster behind the trade (a risk).
    if long:
        tgt_dist, tgt_size = f["eqh_above_dist_atr"], f["eqh_above_size"]
        risk_dist, risk_size = f["eql_below_dist_atr"], f["eql_below_size"]
        tgt_word, risk_word = "equal highs", "equal lows"
        tgt_dir, risk_dir = "up", "down"
    else:
        tgt_dist, tgt_size = f["eql_below_dist_atr"], f["eql_below_size"]
        risk_dist, risk_size = f["eqh_above_dist_atr"], f["eqh_above_size"]
        tgt_word, risk_word = "equal lows", "equal highs"
        tgt_dir, risk_dir = "down", "up"
    bias = "long" if long else "short"

    if tgt_dist is not None:
        return (f"<b>{'Above' if long else 'Below'} the current price:</b> "
                f"{tgt_word} ({tgt_size} touches), {tgt_dist} ATR {tgt_dir}.<br>"
                f"A stack of stops like this pulls price toward it.<br>"
                f"<b>Good take-profit for your {bias} — but expect a sharp "
                f"wick, not a clean stop.</b>")
    if risk_dist is not None:
        return (f"<b>{'Below' if long else 'Above'} the current price:</b> "
                f"{risk_word} ({risk_size} touches), {risk_dist} ATR {risk_dir}."
                f"<br>Price often grabs a stack like this before turning.<br>"
                f"<b>Your {bias} may get shaken out first — give the stop room "
                f"beyond these {risk_word}.</b>")
    return None


def format_eq_sl_warning(features):
    """P2 trade-email warning when the stop sits in front of an equal-level
    pool. Plain English, one sentence, None when not at risk."""
    if not features or not features.get("eq_sl_at_risk"):
        return None
    gap = features.get("eq_sl_gap_atr")
    return (f"⚠ Stop-placement note: your stop sits {gap} ATR in FRONT of a "
            f"resting equal-level pool — a stop-run into that pool takes this "
            f"trade out first. Consider whether the stop belongs on the far "
            f"side of the pool.")
