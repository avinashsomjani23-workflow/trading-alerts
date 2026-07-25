"""Rebuilt liquidity-sweep detector (sweep v2) — pool-anchored, observe-only.

WHY THIS EXISTS (SWEEP_REBUILD_HANDOFF.md / SWEEP_V2_DESIGN.md):
The legacy detector (smc_detector.observe_phase1_sweep) rewards any
wick+close-back off ANY minor lookback-3 swing. Verified on the canonical run:
that signal is noise at the presence level and mildly INVERSE at the quality
tail (the harder it graded a sweep, the worse the trade). SMC-wise most minor
swings hold no meaningful liquidity, so the real signal — a genuine stop-run
of a REAL pool — was diluted to nothing.

THE REBUILT DEFINITION — a sweep exists only when, inside the OB's own leg
window, price raided a RANKED liquidity pool that existed and was INTACT when
the leg began, and the raid resolved as a rejection (not a break):

  TARGET   PW (last week's H/L) > PD (yesterday's H/L) > EQ cluster
           (eq_pools shelf) > SW (a bare lb-3+1.5-ATR swing). SW is the
           weakest tier — the normal-swing fuel read folded in from the
           retired legacy observer (2026-07-20): it shows the leg ran a stop,
           just not a mapped pool. Sub-1.5-ATR triangles still never qualify.
  EVENT    _first_sweep_ts walks the leg window — the ONE sweep judge. The
           sweep is the FIRST touch of the level (later touches are ghosts on
           already-taken liquidity), kept only if the level was not
           broken-and-held first (unbroken AND unspent). A single close-beyond
           that reverses next bar is a FAILED break = a real sweep and is kept;
           a break that held its confirm bar is excluded.
  REJECTION raw wick:body ratio of the sweep bar (logged raw, never tiered).
  FOLLOW-THROUGH displacement beyond the swept level from the sweep bar to
           the break-confirmation bar, in formation ATR.
  RELEVANCE fuel side only (bullish OB <- low-side pools); window =
           [impulse_start - 3, ob_idx] floored at the prior structural event
           (the SAME leg-lock window as the legacy detector, LOCKED 2026-06).
  ROUND NUMBERS a swept level sitting on the RN grid outranks same/lower
           tiers; alignment uses a FEED-BUFFERED tolerance (see below), the
           tight legacy context-tag tolerance is untouched.

OBSERVATION ONLY. Nothing here gates, scores, ranks, or filters (guardrail A5
discipline; handoff rule 4). The legacy detector keeps running untouched for
the score leg / OB2 ranking / chart overlays — live behaviour is byte-identical.

ANCHORING — SPLIT FREEZE (WS2, 2026-07-25). Two different clocks, on purpose:

  SW + EQ — FORMATION-FROZEN. observe_pool_sweep runs ONCE at OB build inside
    detect_smc_radar (live scan AND backtest replay — same function, same
    150-bar clamped frame, same picture). The SW/EQ result is stamped
    ob['sweep_v2'] and never re-graded: live Zone.refresh only back-fills it
    once, the replay zone merge refreshes only fvg. SW is the OB's own leg
    swing and EQ a formation-time shelf — neither ROLLS forward in time, so
    freezing them is truthful. A later re-compute of SW/EQ would see a rolled,
    truncated frame and could silently differ — that is the re-grade bug class
    the SW/EQ freeze kills.

  PW + PD — FILL-ANCHORED FOR THE BACKTEST CSV; birth-frozen for LIVE.
    "Yesterday's low" / "last week's low" ROLL as days/weeks pass; a formation-
    frozen PW/PD reads the wrong price by fill time, and a sweep can UN-happen
    (wick-reject at formation that price later closes back through). So for the
    BACKTEST feature-usability read, PW/PD are re-judged at the FILL bar in
    features_from_snapshot -> pw_pd_at_fill, on the SAME fuel window (the frozen
    leg anchors below), using level values + validity as they stood at fill,
    reading ONLY closed bars strictly before fill_ts. No look-ahead: the window
    is not widened and no post-fill bar is read; a live trader at fill genuinely
    knows where the pool sits and whether it is still taken.

    *** LIVE-vs-BACKTEST PARITY NOTE (WS2, 2026-07-25) — READ BEFORE "FIXING" ***
    This is a DELIBERATE, permanent split, not a bug:
      - LIVE (email describe_pool + score_inputs off ob['sweep_v2'],
        Phase2_Alert_Engine) reads the BIRTH-FROZEN winner, which DOES include
        PW/PD computed at formation. Live is byte-identical to before WS2.
      - The BACKTEST CSV IGNORES the frozen winner cols (sweep2_tier/level/
        pierce/rejection/follow/rn/eq_size were DROPPED from the CSV in WS2) and
        instead reads fill-anchored per-tier blocks (sweep2_{sw,eq,pw,pd}_*).
    So PW/PD ARE computed twice by design: once at birth (for live), once at
    fill (for the backtest feature test). They answer DIFFERENT questions at
    DIFFERENT times — one frozen fact, one live-at-fill read — and cannot be a
    single number. Do NOT "de-duplicate" them; that reintroduces the exact
    live-behaviour change WS2 was built to avoid. observe-only (A5) — nothing
    here gates/scores/ranks in the backtest.

  LEG ANCHORS on the snapshot — impulse_start_idx / ob_idx / prior_event_idx /
    break_idx / direction / pair_type / pair_name are frozen onto the snapshot
    at birth so the fill-time PW/PD recompute rebuilds the EXACT same fuel
    window. These are point-in-time integers/strings fixed at formation — they
    do not roll, so freezing them carries no re-grade risk.

HONESTY BOUNDS (150-bar detection frame, identical live and backtest):
  - PD is provable almost always; PW only when the frame covers the FULL
    prior week. Provability is checked geometrically (frame start <= pool
    birth) and logged in tiers_checked — an unprovable tier is labelled,
    never silently mis-measured. The fill-time PW/PD recompute re-runs this
    same provability gate against the fill frame.
  - Pools are referenced as they existed at the LEG START (the resting
    liquidity the raid took). A leg spanning the 21:00 UTC day-roll checks
    the pre-roll generation; EQ/PW cover most of the sliver this misses.
  - Live feeds carry server-Saturday bars MT5 lacks (pip-class feed
    asymmetry, accepted system-wide).
"""

from datetime import datetime, timezone

import pandas as pd

import dealing_range
import smc_detector
from pool_builder import (
    _day_start_utc,
    _naive_utc_index,
    levels_at,
    pool_status,
)
from eq_pools import EQ_SWING_LOOKBACK, clusters_at

# Same small pre-leg lookback as the legacy detector (LOCKED 2026-06 with the
# trader): the catalysing sweep often prints a candle or two before the
# impulse start (sweep -> base -> impulse). One value, re-used, so the two
# detectors always search the same window.
LOOKBACK_BEFORE_IMPULSE = smc_detector.SWEEP_LOOKBACK_BEFORE_IMPULSE

# Round-number tolerance, FEED-BUFFERED (sweep v2 only — the tight legacy
# ROUND_NUMBER_TOLERANCE stays as-is for the context tags). The legacy 5-pip
# FX tolerance was set for MT5 data; live runs on Twelve Data and the logged
# MT5-vs-TD quote gap is p50 ~1 pip but 5-12 pips at p95 (memories:
# project_oanda_twelvedata_eval / feed_hole_diagnosis_2026-07-08). At 5 pips a
# genuine round-number touch seen through the live feed can land 6-10 pips off
# the grid line and be MISSED, so the alignment FEATURE needs room to register
# on BOTH feeds. 8 pips ~ covers the p95 gap without swallowing half the
# 50-pip grid cell (16% of it). Other classes scaled to the same fraction.
# Same reasoning as eq_pools.EQ_TOL_ATR going 0.10 -> 0.20.
RN_TOLERANCE_BUFFERED = {
    "forex":     0.0008,   # 8 pips on 5-dp pairs   (legacy tag tol: 5)
    "forex_jpy": 0.08,     # 8 pips on 3-dp JPY     (legacy: 0.05)
    "index":     8.0,      # 8 points               (legacy: 5)
    "commodity": 0.80,     # $0.80 on Gold          (legacy: 0.50)
    "crypto":    80.0,     # $80 on BTC             (legacy: 50)
}

# Winner ranking among swept pools: tier weight, then deepest pierce
# (resolved in _rank_key). PW outranks PD outranks EQ outranks SW — the bigger
# the pool, the more meaningful the raid. SW (a bare lb-3+1.5-ATR swing with no
# mapped pool on it) is the weakest fuel and ranks last (2026-07-20, owner): it
# still shows the leg ran a stop, just not a ranked one.
#
# ROUND NUMBER IS DELIBERATELY NOT A RANKING INPUT (2026-07-19, owner call).
# It was originally ranked ABOVE tier on the practitioner hunch that "on FX a
# sweep only holds at a round number". A 2016-17 sample slice did NOT support
# that (RN-aligned FX raids did mildly WORSE, not better), so promoting RN
# would put a thumb on the scale the data does not back. rn_aligned / rn_dist
# stay LOGGED as pure facts for the full-baseline run to judge — they just no
# longer change which pool wins. Zero-noise, still measurable.
_TIER_RANK = {"PW": 0, "PD": 1, "EQ": 2, "SW": 3}

# The trades.csv column set this module owns. One list, one implementation —
# the backtest row build and the None-fallback both key off it (EQ precedent).
#
# WS2 (2026-07-25) reshaped this to a FOUR-BLOCK per-tier layout:
#   - The old birth-frozen WINNER cols (sweep2_tier/level/pierce_atr/
#     rejection_ratio/follow_atr/rn_aligned/rn_dist_atr/eq_size) were DROPPED —
#     they are fully reconstructable from the four always-on per-tier blocks
#     below (winner = highest-ranked present tier, PW>PD>EQ>SW), so keeping them
#     was pure redundancy in the CSV. The winner still exists on the FROZEN
#     snapshot for the LIVE email/score read (see docstring parity note); it is
#     just no longer a CSV column.
#   - Kept roll-ups NOT derivable from one tier block: sweep2_present (any tier),
#     sweep2_pools_swept (birth count), sweep2_tiers_checked, sweep2_age_at_fill.
#   - SW + EQ blocks are BIRTH-FROZEN; PW + PD blocks are re-judged at the FILL
#     bar (pw_pd_at_fill) on the same fuel window. All four blocks are always-on
#     (present even when hidden behind a higher tier), so no per-tier signal is
#     collapsed by a winner pick.
SWEEP2_FEATURE_COLUMNS = (
    # Roll-ups.
    "sweep2_present",         # any tier swept (birth: SW/EQ/PW/PD; see note)
    "sweep2_pools_swept",     # birth-time count of swept candidates
    "sweep2_age_at_fill_h1",  # winner sweep_ts -> fill bar, closed H1 bars
    "sweep2_tiers_checked",   # provability/ran labels, incl. fill pw/pd tags
    # SW block — BIRTH-FROZEN (OB's own leg swing; never rolls). Always-on.
    "sweep2_sw_present",
    "sweep2_sw_pierce_atr",
    "sweep2_sw_rejection_ratio",
    "sweep2_sw_follow_atr",
    "sweep2_sw_rn_aligned",
    # EQ block — BIRTH-FROZEN (formation-time shelf; never rolls). Always-on.
    "sweep2_eq_present",
    "sweep2_eq_pierce_atr",
    "sweep2_eq_rejection_ratio",
    "sweep2_eq_follow_atr",
    "sweep2_eq_rn_aligned",
    "sweep2_eq_size",         # cluster touch count (EQ-only geometry)
    # PW block — FILL-ANCHORED (last week's H/L rolls; re-judged at fill).
    "sweep2_pw_present",
    "sweep2_pw_pierce_atr",
    "sweep2_pw_rejection_ratio",
    "sweep2_pw_follow_atr",
    "sweep2_pw_rn_aligned",
    # PD block — FILL-ANCHORED (yesterday's H/L rolls; re-judged at fill).
    "sweep2_pd_present",
    "sweep2_pd_pierce_atr",
    "sweep2_pd_rejection_ratio",
    "sweep2_pd_follow_atr",
    "sweep2_pd_rn_aligned",
)


def snapshot_none(tiers_checked=""):
    """Canonical empty snapshot. pools_swept=0 + a tiers_checked string means
    the detector RAN and found no qualifying raid; pools_swept=None +
    tiers_checked='failed' (via snapshot_failed) means it could not run at all.
    An EMPTY tiers_checked ('') is the ran-but-no-pre-window case."""
    return {
        "exists": False,
        "tier": None,
        "side": None,
        "level": None,
        "sweep_ts": None,
        "pierce_atr": None,
        "rejection_ratio": None,
        "follow_atr": None,
        "pools_swept": 0,
        "rn_aligned": None,
        "rn_dist_atr": None,
        "eq_size": None,
        "tiers_checked": tiers_checked,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }


def snapshot_failed():
    """Layer-couldn't-run shape (bad inputs / internal error). Distinct from
    'ran, found none' so the columns stay honest (EQ None-vs-[] precedent).

    pools_swept=None is the machine-readable failure flag (drives the all-None
    columns in features_from_snapshot). tiers_checked='failed' is the HUMAN
    flag: it is emitted to sweep2_tiers_checked so a failure reads as the word
    'failed' in the CSV, never as a blank that could be mistaken for the
    ran-but-empty-window case ('') — a failure must SAY it failed."""
    snap = snapshot_none()
    snap["pools_swept"] = None
    snap["tiers_checked"] = "failed"
    return snap


# ---------------------------------------------------------------------------
# Detection (pure) — runs once at OB build, result frozen on the zone
# ---------------------------------------------------------------------------

def _rn_key(pair_name, pair_type):
    """Round-number grid bucket — same JPY special-case as the legacy tags."""
    return smc_detector._round_number_key(pair_name, pair_type)


def _first_sweep_ts(leg_bars, level, side):
    """Timestamp of the ONE valid sweep of `level` inside the leg window, or
    None if there is no valid sweep.

    WHY THIS GUARD EXISTS (2026-07-23): a sweep is only valid on a swing/pool
    that is still UNBROKEN **and UNSPENT**. Two ways the old code broke that:

      1. BROKEN — price closed beyond the level and HELD there (a real break).
         pool_status's reclaim rule (pool_builder.py:342) later downgrades that
         broken level to 'swept' the moment price closes back across it, so a
         decisive break read as a fresh sweep. (EURUSD 2026-07-23: swing highs
         closed above for six straight H1 bars, then were tagged swept.)

      2. SPENT — a sweep takes the resting liquidity ONCE. The first
         wick-through-and-reject fills those stops; the level is spent. But
         pool_status returns last_sweep_ts = the LATEST touch, so if price
         poked the level, rejected, drifted back and poked it AGAIN, the SECOND
         (ghost) poke was reported as the sweep. There is no liquidity left on a
         re-poke — it is not a sweep.

    So the ONLY valid sweep is the FIRST touch of the level in the leg. This
    helper finds that first touch and returns its timestamp — UNLESS the level
    was truly broken (closed-beyond-and-held) at or before that first touch, in
    which case the level was never resting liquidity to raid and None is
    returned.

    A single close-beyond that reverses on the very next bar is a FAILED break =
    a real sweep and is kept (that is the wick/failed-break case). Only a break
    that HELD for its confirm bar disqualifies.
    """
    if leg_bars is None or len(leg_bars) == 0 or level is None:
        return None
    highs = leg_bars["High"].to_numpy()
    lows = leg_bars["Low"].to_numpy()
    closes = leg_bars["Close"].to_numpy()
    index = leg_bars.index
    n = len(closes)
    for i in range(n):
        if side == "above":
            touched = highs[i] > level        # wick pierced the high pool
            closed_beyond = closes[i] > level
            held = closed_beyond and (i + 1 < n) and closes[i + 1] > level
        else:
            touched = lows[i] < level         # wick pierced the low pool
            closed_beyond = closes[i] < level
            held = closed_beyond and (i + 1 < n) and closes[i + 1] < level
        if held:
            # Level truly broke and held before any valid sweep — spent as a
            # break, no resting liquidity to raid.
            return None
        if touched:
            # First touch of the level. This is the one and only sweep; every
            # later touch is a ghost on already-taken liquidity.
            return index[i]
    return None


def _pw_pd_candidates(h1, lo_pos, leg_bars, side, frame_start_ts,
                      days=None, weeks=None):
    """PW/PD fuel-side pools swept by the leg + which tiers were provable.

    A tier is PROVABLE only when the frame's first bar is at or before the
    pool's birth, i.e. the frame contains the pool's whole source period —
    otherwise the level computed from partial bars would be silently wrong
    (the 150-bar frame rarely covers a full prior week). Unprovable tiers are
    reported in `checked` as absent, never guessed.

    `days`/`weeks`, when passed, are the full-frame resample precomputed once
    per bar; levels_at selects periods strictly-before lo_ts internally, so the
    result is byte-identical to rebuilding them from a truncated frame here.
    None => levels_at rebuilds them (live path).
    """
    candidates, checked = [], []
    lo_ts = h1.index[lo_pos]
    lv = levels_at(h1, lo_ts, days=days, weeks=weeks)

    key_map = {"PD": ("pdl" if side == "below" else "pdh", "prev_day"),
               "PW": ("pwl" if side == "below" else "pwh", "prev_week")}
    for tier, (key, prev_label) in key_map.items():
        level = lv.get(key)
        prev = lv.get(prev_label)
        if level is None or prev is None:
            continue
        birth = _day_start_utc(pd.Timestamp(prev))
        if frame_start_ts > birth:
            continue  # frame can't prove the full source period — unprovable
        checked.append(tier.lower())
        # Intact when the leg began: walk the pool's life BEFORE the window.
        # The pool lives during the CURRENT period as of lo_ts — take the
        # period labels levels_at already computed (they carry its weekend
        # adjustment, so a weekend-vantage lo_ts still walks the true period).
        cur_label = lv.get("cur_day") if tier == "PD" else lv.get("cur_week")
        if cur_label is None:
            continue
        life_start = _day_start_utc(pd.Timestamp(cur_label))
        pre_bars = h1.loc[(h1.index >= life_start) & (h1.index < lo_ts)]
        if pool_status(pre_bars, level, side)["status"] != "intact":
            continue  # already drained before the leg — not resting liquidity
        # The ONE valid sweep = FIRST touch of the level, and only if the level
        # was not broken-and-held first (unbroken AND unspent). See
        # _first_sweep_ts. Replaces trusting pool_status' last_sweep_ts (which
        # points at the latest/ghost touch on already-taken liquidity).
        sweep_ts = _first_sweep_ts(leg_bars, level, side)
        if sweep_ts is not None:
            candidates.append({"tier": tier, "level": float(level),
                               "sweep_ts": sweep_ts,
                               "eq_size": None})
    return candidates, checked


def _eq_candidates(h1, lo_pos, leg_bars, side, atr, swings):
    """EQ-shelf fuel-side clusters swept by the leg. Shelves are taken AS OF
    the leg start (clusters_at at lo_pos: members confirmed and status walked
    on bars strictly before the window), so only pre-existing intact shelves
    qualify — a shelf the sweep bar itself joins can never hide its own raid."""
    clusters = clusters_at(h1, lo_pos, atr, swings=swings)
    if clusters is None:
        return [], False  # EQ layer couldn't run (thin history)
    want_type = "low" if side == "below" else "high"
    out = []
    for cl in clusters:
        if cl["side"] != want_type or cl["status"] != "intact":
            continue
        sweep_ts = _first_sweep_ts(leg_bars, cl["level"], side)
        if sweep_ts is not None:
            out.append({"tier": "EQ", "level": float(cl["level"]),
                        "sweep_ts": sweep_ts,
                        "eq_size": int(cl["size"])})
    return out, True


def _sw_candidates(h1, lo_pos, leg_bars, side, atr, pair_type, swings=None):
    """Bare-swing (tier SW) fuel-side pivots swept by the leg — the normal-swing
    fuel read the ranked tiers (PW/PD/EQ) can't see (2026-07-20, owner).

    A tier SW candidate is a lookback-3 + 1.5-ATR swing (the ONE H1 swing
    definition — dealing_range.detect_swings, NOT the raw EQ pool) that:
      - is on the fuel side (LONG raids lows / SHORT raids highs),
      - was ACTIVE (unbroken AND unswept) at the LEG START (is_swing_active
        before_idx=lo_pos) — the same "resting liquidity when the leg began"
        rule the PW/PD/EQ tiers enforce, so a swing already drained pre-leg is
        not counted, and
      - was SWEPT inside the leg window by _first_sweep_ts (the ONE sweep
        judge): the FIRST touch of the swing, kept only if it was not
        broken-and-held first. A re-poke on the same swing later in the leg is a
        ghost on already-taken liquidity, not a sweep.

    Ranked BELOW EQ (weakest fuel: a local stop-run on no mapped pool). This is
    the observation the retired legacy observe_phase1_sweep used to make; folded
    in here so there is ONE sweep detector, one window, one judge.
    """
    # The single H1 swing definition (lb-3 + 1.5-ATR). Never touches the
    # eq_pools per-frame cache (perf trap). The caller may pass this same
    # window-constant set in via `swings` (within-bar reuse); identical values,
    # computed once per bar instead of once per event.
    if swings is None:
        swings = dealing_range.detect_swings(h1, lookback=dealing_range.SWING_LOOKBACK)
    if not swings:
        return []
    want_type = "low" if side == "below" else "high"
    pierce_min = (smc_detector.SWEEP_WICK_PIERCE_MIN_ATR.get(pair_type, 0.05) * atr)
    out = []
    for s in swings:
        if s.get("type") != want_type:
            continue
        s_idx = int(s["idx"])
        if s_idx >= lo_pos:
            continue  # swing must predate the window to be resting liquidity it raids
        # Resting when the leg began: unbroken AND unswept up to the leg start.
        if not smc_detector.is_swing_active(s, h1, pierce_min, before_idx=lo_pos):
            continue
        # The ONE valid sweep = FIRST touch of the swing, and only if it was not
        # broken-and-held first (unbroken AND unspent). This is the EURUSD
        # 2026-07-23 bug: the swing highs closed above for six bars (broken)
        # before price fell back, and pool_status still reported them swept.
        sweep_ts = _first_sweep_ts(leg_bars, float(s["price"]), side)
        if sweep_ts is not None:
            out.append({"tier": "SW", "level": float(s["price"]),
                        "sweep_ts": sweep_ts,
                        "eq_size": None})
    return out


# ---------------------------------------------------------------------------
# Per-tier ALWAYS-ON metric block — ONE implementation, reused by every tier
# (SW/EQ at birth, PW/PD at fill). "Best" candidate = the EARLIEST swept one
# (chronological fact, "first touch is the sweep"; matches _first_sweep_ts —
# NOT a quality pick). pierce/rn are stamped per-candidate before this call;
# rejection + follow are computed here with the SAME helpers/span the winner
# uses, so a tier's block equals the winner cols when that tier wins.
# ---------------------------------------------------------------------------

def _tier_block(tier_candidates, h1, side, swept_type, tf_atr, ft_end, H, L):
    """Return the always-on (present, pierce_atr, rejection_ratio, follow_atr,
    rn_aligned) dict for one tier's candidate list. Empty list -> all None/False.
    Every candidate must already carry sweep_pos / pierce_atr / rn_aligned
    (stamped by the per-candidate loop in observe_pool_sweep / pw_pd_at_fill)."""
    if not tier_candidates:
        return {"present": False, "pierce_atr": None, "rejection_ratio": None,
                "follow_atr": None, "rn_aligned": None, "sweep_ts": None}
    best = min(tier_candidates, key=lambda c: pd.Timestamp(c["sweep_ts"]))
    b_pos = best["sweep_pos"]
    _, rej = smc_detector._rejection_score(h1, b_pos, swept_type, tf_atr)
    follow = None
    if b_pos + 1 <= ft_end:
        if side == "below":
            exc = float(H[b_pos + 1: ft_end + 1].max()) - best["level"]
        else:
            exc = best["level"] - float(L[b_pos + 1: ft_end + 1].min())
        follow = round(exc / tf_atr, 3)
    # sweep_ts carried for the fill-age anchor (earliest sweep across tiers).
    return {"present": True, "pierce_atr": best["pierce_atr"],
            "rejection_ratio": round(float(rej), 3), "follow_atr": follow,
            "rn_aligned": best["rn_aligned"],
            "sweep_ts": pd.Timestamp(best["sweep_ts"]).isoformat()}


def _stamp_candidate_metrics(candidates, h1, side, tf_atr, pair_name, pair_type):
    """Stamp sweep_pos / pierce_atr / rn_aligned / rn_dist_atr on each candidate
    IN PLACE. Shared by birth (observe_pool_sweep) and fill (pw_pd_at_fill) so
    the two paths grade a swept level identically."""
    rn_bucket = _rn_key(pair_name, pair_type)
    grid = smc_detector.ROUND_NUMBER_GRID.get(rn_bucket, 0.0)
    rn_tol = RN_TOLERANCE_BUFFERED.get(rn_bucket, 0.0)
    H = h1["High"].values
    L = h1["Low"].values
    for c in candidates:
        ts = pd.Timestamp(c["sweep_ts"])
        pos = int(h1.index.searchsorted(ts))
        c["sweep_pos"] = pos
        if side == "below":
            pierce = c["level"] - float(L[pos])
        else:
            pierce = float(H[pos]) - c["level"]
        # A reclaim-bar sweep (broken then given back inside the window) stamps
        # a bar that need not wick beyond the level — clamp to 0.
        c["pierce_atr"] = round(max(pierce, 0.0) / tf_atr, 3)
        if grid > 0:
            nearest = smc_detector._nearest_round_number(c["level"], grid)
            rn_dist = c["level"] - nearest
            c["rn_aligned"] = bool(abs(rn_dist) <= rn_tol)
            c["rn_dist_atr"] = round(rn_dist / tf_atr, 3)
        else:
            c["rn_aligned"] = None
            c["rn_dist_atr"] = None


def observe_pool_sweep(df, ob_idx, impulse_start_idx, direction, tf_atr,
                       pair_type, pair_name, prior_event_idx=None,
                       break_idx=None, days=None, weeks=None,
                       eq_swings=None, sw_swings=None):
    """The sweep-v2 observation for one OB. Returns the ob['sweep_v2'] dict.

    Args mirror the legacy observer's call site in detect_smc_radar:
      df                — the detection frame (live reset-index or backtest
                          DatetimeIndex; funnelled through _naive_utc_index).
      ob_idx            — OB candle position in df.
      impulse_start_idx — leg start position.
      direction         — 'bullish' | 'bearish' (fuel side = low | high).
      tf_atr            — frozen formation H1 ATR (ob['h1_atr'] source; the
                          shared *_atr denominator).
      prior_event_idx   — window floor (never reach an earlier leg).
      break_idx         — break-confirmation candle position; follow-through
                          is measured sweep bar -> this bar. Falls back to
                          ob_idx when absent/invalid.
      days/weeks        — PERF (within-bar reuse): the daily/weekly resample
                          FRAME, precomputed ONCE on the whole df window by the
                          caller and shared across every event in this bar.
                          levels_at still selects periods strictly-before each
                          event's asof internally, so the PDH/PDL/PWH/PWL are
                          byte-identical to the per-event rebuild — nothing is
                          stamped or frozen. None => rebuild locally (live path).
      eq_swings/sw_swings — the EQ-lookback and lb-3 swing sets, likewise
                          window-constant across this bar's events, precomputed
                          once by the caller. None => compute locally.

    Never raises on degraded input — returns snapshot_failed() so a sweep bug
    can never kill an OB build (guard lives OUT of the live alert path).
    """
    try:
        if df is None or len(df) == 0:
            return snapshot_failed()
        if ob_idx is None or impulse_start_idx is None:
            return snapshot_failed()
        if direction not in ("bullish", "bearish"):
            return snapshot_failed()
        if tf_atr is None or tf_atr <= 0:
            return snapshot_failed()
        n = len(df)
        ob_pos = int(ob_idx)
        if not (0 <= int(impulse_start_idx) <= ob_pos < n):
            return snapshot_failed()

        # Same window rule as the legacy detector (one concept, one window):
        # a few candles before the impulse start, hard-floored at the prior
        # structural event so the search can never reach an earlier leg.
        lo_pos = int(impulse_start_idx) - LOOKBACK_BEFORE_IMPULSE
        if prior_event_idx is not None:
            try:
                lo_pos = max(lo_pos, int(prior_event_idx) + 1)
            except (TypeError, ValueError):
                pass
        lo_pos = max(lo_pos, 0)
        if lo_pos > ob_pos:
            # The prior structural event sits at/after the OB candle — the
            # leg has NO pre-window at all (~5% of real OBs, measured on
            # random cached windows). No room for a fueling raid is a REAL
            # negative ("ran, found none"), same as the legacy detector's
            # empty observation — never a layer failure. tiers_checked=''
            # says nothing was checkable.
            return snapshot_none("")

        h1 = _naive_utc_index(df)
        if not isinstance(h1.index, pd.DatetimeIndex):
            return snapshot_failed()
        side = "below" if direction == "bullish" else "above"
        leg_bars = h1.iloc[lo_pos: ob_pos + 1]

        # WS2 leg anchors — TIMESTAMPS (not birth-frame index positions), so the
        # BACKTEST fill-time PW/PD recompute can re-find the SAME fuel window
        # inside the DIFFERENT fill frame by searchsorting these boundaries.
        # Frozen point-in-time facts; they do not roll. leg_end_ts is the OB
        # candle (window is [leg_start, ob] inclusive); break_ts anchors
        # follow-through at fill. LIVE ignores these — it reads the birth winner.
        break_pos = ob_pos
        if break_idx is not None:
            try:
                bi = int(break_idx)
                if ob_pos <= bi < n:
                    break_pos = bi
            except (TypeError, ValueError):
                pass
        leg_anchors = {
            "leg_start_ts": h1.index[lo_pos].isoformat(),
            "leg_end_ts": h1.index[ob_pos].isoformat(),
            "break_ts": h1.index[break_pos].isoformat(),
            "direction": direction,
            "pair_type": pair_type,
            "pair_name": pair_name,
            "tf_atr": float(tf_atr),
        }

        # Raw-geometry swings for the EQ reference (approved for the sweep/EQ
        # use-case only). Computed locally and PASSED IN to _eq_candidates so
        # this call never touches eq_pools' per-frame cache — evicting that cache
        # with a 150-bar detection frame would force the backtest row build to
        # re-derive full-frame swings per row (the proven perf trap). The caller
        # may hand this same window-constant set in via eq_swings (within-bar
        # reuse); identical values, just computed once per bar instead of once
        # per event.
        swings = eq_swings
        if swings is None:
            swings = dealing_range.detect_swings(h1, lookback=EQ_SWING_LOOKBACK,
                                                 min_leg_atr_mult=None)

        # Birth-time candidates — ALL FOUR tiers (PW/PD/EQ/SW). This frozen
        # snapshot is what the LIVE path reads (email describe_pool + score_inputs
        # off ob['sweep_v2']); it must keep its full birth winner so live stays
        # byte-identical (WS2 parity note — see leg_anchors comment above and the
        # module docstring). The BACKTEST CSV re-judges PW/PD at fill separately
        # (pw_pd_at_fill) and does NOT read these frozen winner cols.
        pwpd, checked = _pw_pd_candidates(h1, lo_pos, leg_bars, side,
                                          h1.index[0], days=days, weeks=weeks)
        eq, eq_ran = _eq_candidates(h1, lo_pos, leg_bars, side, tf_atr, swings)
        if eq_ran:
            checked.append("eq")
        # Tier SW — bare lb-3+1.5-ATR swings (the normal-swing fuel read the
        # ranked tiers can't see). Always checkable on H1 (no provability gate),
        # so it always joins tiers_checked.
        sw = _sw_candidates(h1, lo_pos, leg_bars, side, tf_atr, pair_type,
                            swings=sw_swings)
        checked.append("sw")
        candidates = pwpd + eq + sw
        tiers_checked = ",".join(checked)
        if not candidates:
            # Nothing swept at birth — but PW/PD are re-judged at FILL for the
            # backtest, so the snapshot still carries the leg anchors so
            # pw_pd_at_fill can run there.
            snap = snapshot_none(tiers_checked)
            snap["_leg"] = leg_anchors
            return snap

        # Stamp RN alignment + sweep-bar metrics per candidate (shared helper —
        # birth and fill grade a swept level identically).
        _stamp_candidate_metrics(candidates, h1, side, tf_atr,
                                 pair_name, pair_type)
        H = h1["High"].values
        L = h1["Low"].values

        # Winner = biggest pool (PW>PD>EQ), tie broken by deepest pierce.
        # Round-number alignment is NOT in the key (see _TIER_RANK comment) —
        # it stays a logged fact, never a ranking thumb. This BIRTH winner is
        # what LIVE reads (email/score); the backtest ignores it (WS2 parity).
        def _rank_key(c):
            return (_TIER_RANK.get(c["tier"], 9), -c["pierce_atr"])

        winner = sorted(candidates, key=_rank_key)[0]
        w_pos = winner["sweep_pos"]

        # Raw wick:body of the sweep bar. Reuses the legacy geometry helper
        # (one implementation); only the RAW ratio is kept — no tier caps.
        swept_type = "low" if side == "below" else "high"
        _, rej_ratio = smc_detector._rejection_score(h1, w_pos, swept_type,
                                                     tf_atr)

        # Follow-through: displacement beyond the swept level from the bar
        # after the sweep to the break-confirmation bar (the leg's own end —
        # no arbitrary N-bar knob). None when no bar exists in that span.
        ft_end = ob_pos
        if break_idx is not None:
            try:
                bi = int(break_idx)
                if ob_pos <= bi < n:
                    ft_end = bi
            except (TypeError, ValueError):
                pass
        follow_atr = None
        if w_pos + 1 <= ft_end:
            if side == "below":
                excursion = float(H[w_pos + 1: ft_end + 1].max()) - winner["level"]
            else:
                excursion = winner["level"] - float(L[w_pos + 1: ft_end + 1].min())
            follow_atr = round(excursion / tf_atr, 3)

        # ── BIRTH-FROZEN per-tier ALWAYS-ON blocks: SW + EQ (2026-07-25) ─────
        # SW and EQ never roll (own-leg swing / formation shelf), so they freeze
        # here. Each block is surfaced even when hidden behind a higher-tier
        # winner (best = earliest swept; ONE _tier_block implementation). PW/PD
        # are NOT blocked here — they are re-judged at fill (pw_pd_at_fill).
        sw_blk = _tier_block(sw, h1, side, swept_type, tf_atr, ft_end, H, L)
        eq_blk = _tier_block(eq, h1, side, swept_type, tf_atr, ft_end, H, L)
        # EQ cluster touch-count (EQ-only geometry) — earliest swept EQ, matching
        # the _tier_block "best" pick so size lines up with the block metrics.
        eq_size_out = None
        if eq:
            eq_size_out = min(eq, key=lambda c: pd.Timestamp(c["sweep_ts"]))["eq_size"]

        return {
            "exists": True,
            # ── LIVE winner fields (birth-frozen) — read by describe_pool /
            # score_inputs off ob['sweep_v2']. The backtest CSV IGNORES these
            # (WS2 parity note in the module docstring). Kept byte-identical.
            "tier": winner["tier"],
            "side": swept_type,
            "level": winner["level"],
            "sweep_ts": pd.Timestamp(winner["sweep_ts"]).isoformat(),
            "pierce_atr": winner["pierce_atr"],
            "rejection_ratio": round(float(rej_ratio), 3),
            "follow_atr": follow_atr,
            "pools_swept": len(candidates),
            "rn_aligned": winner["rn_aligned"],
            "rn_dist_atr": winner["rn_dist_atr"],
            "eq_size": winner["eq_size"],
            "tiers_checked": tiers_checked,
            # ── BIRTH-FROZEN per-tier blocks for the backtest CSV: SW + EQ.
            # Always-on (present even when hidden behind a higher-tier winner).
            # Nested dicts so features_from_snapshot re-labels them straight,
            # never re-detects. PW/PD are absent here — added at fill.
            "sw_block": sw_blk,
            "eq_block": eq_blk,
            "eq_block_size": eq_size_out,
            # WS2 leg anchors — backtest-only, for pw_pd_at_fill's window rebuild.
            # Live never reads this; the frozen winner cols above serve live.
            "_leg": leg_anchors,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        # Never let the sweep layer kill an OB build / backtest bar.
        print(f"  [SWEEP2 WARN] observe_pool_sweep failed: "
              f"{type(e).__name__}: {e}")
        return snapshot_failed()


# ---------------------------------------------------------------------------
# Backtest row-build entry point (spread precedent: eq_pools / pool_builder)
# ---------------------------------------------------------------------------

def features_none():
    """All-None sweep2 dict — the honest value when the snapshot is missing
    (legacy zone) or the layer failed."""
    return {col: None for col in SWEEP2_FEATURE_COLUMNS}


def _naive_fill_ts(fill_ts):
    """Coerce fill_ts to a tz-naive UTC Timestamp (searchsort key), or None."""
    if fill_ts is None:
        return None
    f = pd.Timestamp(fill_ts)
    if f.tzinfo is not None:
        f = f.tz_convert("UTC").tz_localize(None)
    return f


def pw_pd_at_fill(snap, df_h1, fill_ts):
    """Re-judge the PW/PD tiers AT THE FILL BAR on the OB's frozen fuel window.

    Returns (pw_block, pd_block, pw_pd_tags, first_sweep_ts) where each block is
    the _tier_block always-on dict and pw_pd_tags is the list of provable tiers
    ('pw'/'pd') to fold into tiers_checked. first_sweep_ts is the EARLIEST swept
    PW/PD sweep_ts (for the age anchor), or None.

    WHY FILL, NOT BIRTH (WS2): "yesterday's / last week's low" ROLL as time
    passes and a sweep can UN-happen (price closes back through). This reads the
    pool levels + validity as they stood at the fill bar. NO LOOK-AHEAD: only
    closed bars STRICTLY BEFORE fill_ts are used (df_h1 sliced < fill_ts), and
    the fuel window is the SAME [leg_start, leg_end] the OB froze — never widened.

    Never raises — returns empty blocks on any failure (a sweep bug can't kill a
    run row). Empty blocks when: no _leg anchors (legacy/no-window snapshot),
    never_filled (fill_ts None), or the anchors fall outside the fill frame.
    """
    empty = {"present": False, "pierce_atr": None, "rejection_ratio": None,
             "follow_atr": None, "rn_aligned": None}
    try:
        leg = snap.get("_leg") if isinstance(snap, dict) else None
        f_ts = _naive_fill_ts(fill_ts)
        if leg is None or f_ts is None or df_h1 is None:
            return dict(empty), dict(empty), [], None

        # Closed bars strictly before the fill bar — the no-look-ahead clamp.
        h1 = _naive_utc_index(df_h1)
        if not isinstance(h1.index, pd.DatetimeIndex):
            return dict(empty), dict(empty), [], None
        h1 = h1[h1.index < f_ts]
        if len(h1) == 0:
            return dict(empty), dict(empty), [], None

        # Re-find the frozen fuel window inside the fill frame by timestamp.
        side = "below" if leg["direction"] == "bullish" else "above"
        tf_atr = float(leg["tf_atr"])
        start_ts = _naive_fill_ts(leg["leg_start_ts"])
        end_ts = _naive_fill_ts(leg["leg_end_ts"])
        break_ts = _naive_fill_ts(leg["break_ts"])
        lo_pos = int(h1.index.searchsorted(start_ts))
        ob_pos = int(h1.index.searchsorted(end_ts))
        # The window must sit fully inside the (pre-fill) frame. A fill so soon
        # after formation that the OB bar itself is not yet closed cannot carry a
        # fill-anchored PW/PD read — honest empty, not a guess.
        if not (0 <= lo_pos <= ob_pos < len(h1)):
            return dict(empty), dict(empty), [], None
        leg_bars = h1.iloc[lo_pos: ob_pos + 1]
        ft_end = ob_pos
        bpos = int(h1.index.searchsorted(break_ts))
        if ob_pos <= bpos < len(h1):
            ft_end = bpos

        # Same PW/PD detector as birth, but on the fill frame: levels_at picks
        # yesterday's/last week's H/L as of the leg start relative to THIS frame,
        # re-runs the provability gate, and _first_sweep_ts re-judges validity
        # (a sweep that later un-happened no longer qualifies).
        pwpd, checked = _pw_pd_candidates(h1, lo_pos, leg_bars, side,
                                          h1.index[0])
        _stamp_candidate_metrics(pwpd, h1, side, tf_atr,
                                 leg["pair_name"], leg["pair_type"])
        H = h1["High"].values
        L = h1["Low"].values
        swept_type = "low" if side == "below" else "high"
        pw = [c for c in pwpd if c["tier"] == "PW"]
        pd_ = [c for c in pwpd if c["tier"] == "PD"]
        pw_blk = _tier_block(pw, h1, side, swept_type, tf_atr, ft_end, H, L)
        pd_blk = _tier_block(pd_, h1, side, swept_type, tf_atr, ft_end, H, L)
        first_ts = None
        if pwpd:
            first_ts = min(pwpd, key=lambda c: pd.Timestamp(c["sweep_ts"]))["sweep_ts"]
        return pw_blk, pd_blk, checked, first_ts
    except Exception as e:
        print(f"  [SWEEP2 WARN] pw_pd_at_fill failed: {type(e).__name__}: {e}")
        return dict(empty), dict(empty), [], None


def _block_cols(out, prefix, blk):
    """Re-label one _tier_block dict into the sweep2_<prefix>_* CSV columns."""
    out[f"sweep2_{prefix}_present"] = bool(blk.get("present"))
    out[f"sweep2_{prefix}_pierce_atr"] = blk.get("pierce_atr")
    out[f"sweep2_{prefix}_rejection_ratio"] = blk.get("rejection_ratio")
    out[f"sweep2_{prefix}_follow_atr"] = blk.get("follow_atr")
    out[f"sweep2_{prefix}_rn_aligned"] = blk.get("rn_aligned")


def features_from_snapshot(snap, df_h1, fill_ts):
    """SWEEP2_FEATURE_COLUMNS dict — FOUR always-on per-tier blocks + roll-ups.

    WS2 (2026-07-25) split freeze:
      - SW + EQ blocks: re-labelled STRAIGHT off the birth-frozen snapshot
        (snap['sw_block'] / snap['eq_block']) — never re-detected.
      - PW + PD blocks: RE-JUDGED at the fill bar (pw_pd_at_fill) on the OB's
        frozen fuel window, closed bars strictly before fill_ts only. No
        look-ahead (see pw_pd_at_fill).
      - The old birth-frozen WINNER cols are NOT emitted — dropped in WS2 as
        redundant (reconstructable from the four blocks). The winner still lives
        on the snapshot for the LIVE read; it is just not a CSV column.

    sweep2_age_at_fill_h1 = closed H1 bars from the EARLIEST sweep across all
    present tiers (SW/EQ frozen + PW/PD at fill) to the fill bar — the first
    stop-run that fuelled the setup. None when never_filled or no sweep.
    sweep2_tiers_checked folds the birth eq/sw tags with the fill pw/pd tags.
    Never raises — all-None dict on any failure so a sweep bug can't kill a row.
    """
    out = features_none()
    try:
        if not isinstance(snap, dict):
            return out  # legacy zone / no snapshot at all — every column None
        if snap.get("pools_swept") is None:
            # Layer failed at birth. Numeric columns stay None; the honesty
            # label carries the explicit 'failed' word so it is never a silent
            # blank. PW/PD cannot be re-judged (no leg anchors) — stay None.
            out["sweep2_tiers_checked"] = snap.get("tiers_checked")
            return out

        out["sweep2_pools_swept"] = snap.get("pools_swept")

        # Frozen SW/EQ blocks (birth). Missing block => absent tier (None/False).
        sw_blk = snap.get("sw_block") or {}
        eq_blk = snap.get("eq_block") or {}
        _block_cols(out, "sw", sw_blk)
        _block_cols(out, "eq", eq_blk)
        out["sweep2_eq_size"] = snap.get("eq_block_size")

        # Fill-anchored PW/PD blocks (re-judged on the fill frame).
        pw_blk, pd_blk, pwpd_tags, pwpd_first_ts = pw_pd_at_fill(
            snap, df_h1, fill_ts)
        _block_cols(out, "pw", pw_blk)
        _block_cols(out, "pd", pd_blk)

        # sweep2_present = any tier present (birth SW/EQ OR fill PW/PD).
        out["sweep2_present"] = bool(
            sw_blk.get("present") or eq_blk.get("present")
            or pw_blk.get("present") or pd_blk.get("present"))

        # tiers_checked = birth eq/sw tags (from the snapshot) folded with the
        # fill pw/pd provability tags. Birth tags already carry eq/sw; strip any
        # stale pw/pd from the frozen string (birth ran them for the live winner)
        # and replace with the fill-frame provability verdict.
        birth_tags = [t for t in (snap.get("tiers_checked") or "").split(",")
                      if t and t not in ("pw", "pd")]
        out["sweep2_tiers_checked"] = ",".join(birth_tags + list(pwpd_tags))

        # Age — earliest sweep across ALL present tiers to the fill bar.
        sweep_tss = [b.get("sweep_ts") for b in (sw_blk, eq_blk)
                     if b.get("sweep_ts")]
        if pwpd_first_ts:
            sweep_tss.append(pwpd_first_ts)
        f_ts = _naive_fill_ts(fill_ts)
        if sweep_tss and df_h1 is not None and f_ts is not None:
            h1 = _naive_utc_index(df_h1)
            if isinstance(h1.index, pd.DatetimeIndex):
                earliest = min(_naive_fill_ts(t) for t in sweep_tss)
                f_pos = int(h1.index.searchsorted(f_ts))  # bars before fill
                s_pos = int(h1.index.searchsorted(earliest))
                if f_pos > s_pos:
                    out["sweep2_age_at_fill_h1"] = int(f_pos - 1 - s_pos)
        return out
    except Exception as e:
        print(f"  [SWEEP2 WARN] features_from_snapshot failed: "
              f"{type(e).__name__}: {e}")
        return features_none()


# ---------------------------------------------------------------------------
# Plain-English narration (shared by the P1 chip/title and the P2 banner)
# ---------------------------------------------------------------------------

_POOL_PHRASE = {
    ("PD", "low"): "yesterday's low", ("PD", "high"): "yesterday's high",
    ("PW", "low"): "last week's low", ("PW", "high"): "last week's high",
}


# Tier -> 0-3 quality grade for the JPY/Gold score leg (2026-07-20). The legacy
# observer fed run_scorecard a 0-3 score (base + equal_levels + rejection); the
# scorecard scales it x2/3 into a 0-2 budget. The merged sweep v2 replaces that
# input for the SCORE ONLY (owner "Option 1": score-only rewire, legacy stays
# alive for badge/OB2-rank/sweep_present).
#
# The grade is TIER-ANCHORED and nothing else. Rejection ratio / follow-through
# are DELIBERATELY not additive points here: the sweep-v2 rebuild proved that
# grading a sweep's *quality* beyond "which pool" was noise/inverse, so folding
# those back in as score would re-introduce the exact tuned thumb that rebuild
# removed. Tier (pool meaningfulness) IS the quality signal; the raw metrics stay
# logged (the per-tier sweep2_{sw,eq,pw,pd}_rejection_ratio / _follow_atr blocks)
# for the full run to judge, never scored. This grade serves the LIVE score leg
# only (score_inputs off the frozen snapshot) — unchanged by the WS2 CSV reshape.
# PW(3) > PD(2.5) > EQ(2) > SW(1): a bigger raided pool is stronger
# fuel; SW (a bare swing, weakest tier) grades lowest but non-zero (it is a real,
# if minor, stop-run).
_TIER_GRADE_0_3 = {"PW": 3.0, "PD": 2.5, "EQ": 2.0, "SW": 1.0}


def score_inputs(snap):
    """(exists, grade_0_3) for run_scorecard's sweep leg, from the FROZEN
    ob['sweep_v2'] snapshot. exists drives the non-JPY-FX presence collapse;
    grade_0_3 (tier-anchored, see _TIER_GRADE_0_3) drives the JPY/Gold quality
    leg. A missing/failed/empty snapshot -> (False, 0.0), matching the legacy
    'schema drift -> treat as zero' behaviour. Never raises."""
    if not isinstance(snap, dict) or not snap.get("exists"):
        return (False, 0.0)
    grade = _TIER_GRADE_0_3.get(snap.get("tier"), 0.0)
    return (True, grade)


def describe_pool(snap):
    """Short plain-English name of the raided pool, e.g. "yesterday's low",
    "an equal-lows shelf (3 touches)", "a local swing low". None when no raid."""
    if not isinstance(snap, dict) or not snap.get("exists"):
        return None
    tier, side = snap.get("tier"), snap.get("side")
    if tier == "EQ":
        word = "equal-lows" if side == "low" else "equal-highs"
        size = snap.get("eq_size")
        touches = f" ({size} touches)" if size else ""
        return f"an {word} shelf{touches}"
    if tier == "SW":
        # Bare swing — no mapped pool, the weakest fuel tier. Named as a plain
        # local swing so the email never implies a ranked pool was taken.
        return "a local swing low" if side == "low" else "a local swing high"
    return _POOL_PHRASE.get((tier, side), "a mapped pool")
