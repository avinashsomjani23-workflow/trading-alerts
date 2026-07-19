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
  EVENT    pool_builder.pool_status walked over the leg window — the ONE
           existing sweep judge (wick+close-back = swept, failed break =
           swept, close-and-hold = broken -> excluded). No second judge.
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

ANCHORING. observe_pool_sweep runs ONCE at OB build inside detect_smc_radar
(live scan AND backtest replay — the replay drives the same function on the
same 150-bar clamped frame, so the two paths see the same picture). The
returned snapshot is stamped ob['sweep_v2'] and is FORMATION-FROZEN: live
Zone.refresh never re-stamps it (one-time back-fill only) and the replay's
zone merge refreshes only fvg. A later re-compute would see a rolled frame
with truncated history and could silently differ — that is the re-grade bug
class the freeze kills.

HONESTY BOUNDS (150-bar detection frame, identical live and backtest):
  - PD is provable almost always; PW only when the frame covers the FULL
    prior week. Provability is checked geometrically (frame start <= pool
    birth) and logged in tiers_checked — an unprovable tier is labelled,
    never silently mis-measured.
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
SWEEP2_FEATURE_COLUMNS = (
    "sweep2_present",
    "sweep2_tier",
    "sweep2_level",
    "sweep2_pierce_atr",
    "sweep2_rejection_ratio",
    "sweep2_follow_atr",
    "sweep2_pools_swept",
    "sweep2_rn_aligned",
    "sweep2_rn_dist_atr",
    "sweep2_eq_size",
    "sweep2_age_at_alert_h1",
    "sweep2_tiers_checked",
)


def snapshot_none(tiers_checked=""):
    """Canonical empty snapshot. pools_swept=0 + a tiers_checked string means
    the detector RAN and found no qualifying raid; pools_swept=None (via
    snapshot_failed) means it could not run at all."""
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
    'ran, found none' so the columns stay honest (EQ None-vs-[] precedent)."""
    snap = snapshot_none()
    snap["pools_swept"] = None
    snap["tiers_checked"] = None
    return snap


# ---------------------------------------------------------------------------
# Detection (pure) — runs once at OB build, result frozen on the zone
# ---------------------------------------------------------------------------

def _rn_key(pair_name, pair_type):
    """Round-number grid bucket — same JPY special-case as the legacy tags."""
    return smc_detector._round_number_key(pair_name, pair_type)


def _pw_pd_candidates(h1, lo_pos, leg_bars, side, frame_start_ts):
    """PW/PD fuel-side pools swept by the leg + which tiers were provable.

    A tier is PROVABLE only when the frame's first bar is at or before the
    pool's birth, i.e. the frame contains the pool's whole source period —
    otherwise the level computed from partial bars would be silently wrong
    (the 150-bar frame rarely covers a full prior week). Unprovable tiers are
    reported in `checked` as absent, never guessed.
    """
    candidates, checked = [], []
    lo_ts = h1.index[lo_pos]
    lv = levels_at(h1, lo_ts)

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
        st = pool_status(leg_bars, level, side)
        if st["status"] == "swept" and st["last_sweep_ts"] is not None:
            candidates.append({"tier": tier, "level": float(level),
                               "sweep_ts": st["last_sweep_ts"],
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
        st = pool_status(leg_bars, cl["level"], side)
        if st["status"] == "swept" and st["last_sweep_ts"] is not None:
            out.append({"tier": "EQ", "level": float(cl["level"]),
                        "sweep_ts": st["last_sweep_ts"],
                        "eq_size": int(cl["size"])})
    return out, True


def _sw_candidates(h1, lo_pos, leg_bars, side, atr, pair_type):
    """Bare-swing (tier SW) fuel-side pivots swept by the leg — the normal-swing
    fuel read the ranked tiers (PW/PD/EQ) can't see (2026-07-20, owner).

    A tier SW candidate is a lookback-3 + 1.5-ATR swing (the ONE H1 swing
    definition — dealing_range.detect_swings, NOT the raw EQ pool) that:
      - is on the fuel side (LONG raids lows / SHORT raids highs),
      - was ACTIVE (unbroken AND unswept) at the LEG START (is_swing_active
        before_idx=lo_pos) — the same "resting liquidity when the leg began"
        rule the PW/PD/EQ tiers enforce, so a swing already drained pre-leg is
        not counted, and
      - was SWEPT inside the leg window by pool_status (the ONE sweep judge —
        wick+close-back / failed break; close-and-hold is a break, excluded).

    Ranked BELOW EQ (weakest fuel: a local stop-run on no mapped pool). This is
    the observation the retired legacy observe_phase1_sweep used to make; folded
    in here so there is ONE sweep detector, one window, one judge.
    """
    # The single H1 swing definition (lb-3 + 1.5-ATR). Computed locally and used
    # only here — never touches the eq_pools per-frame cache (perf trap).
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
        st = pool_status(leg_bars, float(s["price"]), side)
        if st["status"] == "swept" and st["last_sweep_ts"] is not None:
            out.append({"tier": "SW", "level": float(s["price"]),
                        "sweep_ts": st["last_sweep_ts"],
                        "eq_size": None})
    return out


def observe_pool_sweep(df, ob_idx, impulse_start_idx, direction, tf_atr,
                       pair_type, pair_name, prior_event_idx=None,
                       break_idx=None):
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

        # Raw-geometry swings for the EQ reference (approved for the sweep/EQ
        # use-case only). Computed locally and PASSED IN so this per-OB call
        # never touches eq_pools' per-frame cache — evicting that cache with a
        # 150-bar detection frame would force the backtest row build to
        # re-derive full-frame swings per row (the proven perf trap).
        swings = dealing_range.detect_swings(h1, lookback=EQ_SWING_LOOKBACK,
                                             min_leg_atr_mult=None)

        pwpd, checked = _pw_pd_candidates(h1, lo_pos, leg_bars, side,
                                          h1.index[0])
        eq, eq_ran = _eq_candidates(h1, lo_pos, leg_bars, side, tf_atr, swings)
        if eq_ran:
            checked.append("eq")
        # Tier SW — bare lb-3+1.5-ATR swings (the normal-swing fuel read the
        # ranked tiers can't see). Always checkable on H1 (no provability gate),
        # so it always joins tiers_checked.
        sw = _sw_candidates(h1, lo_pos, leg_bars, side, tf_atr, pair_type)
        checked.append("sw")
        candidates = pwpd + eq + sw
        tiers_checked = ",".join(checked)
        if not candidates:
            return snapshot_none(tiers_checked)

        # Stamp RN alignment + sweep-bar metrics per candidate.
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
            # A reclaim-bar sweep (broken then given back inside the window)
            # stamps a bar that need not wick beyond the level — clamp to 0.
            c["pierce_atr"] = round(max(pierce, 0.0) / tf_atr, 3)
            if grid > 0:
                nearest = smc_detector._nearest_round_number(c["level"], grid)
                rn_dist = c["level"] - nearest
                c["rn_aligned"] = bool(abs(rn_dist) <= rn_tol)
                c["rn_dist_atr"] = round(rn_dist / tf_atr, 3)
            else:
                c["rn_aligned"] = None
                c["rn_dist_atr"] = None

        # Winner = biggest pool (PW>PD>EQ), tie broken by deepest pierce.
        # Round-number alignment is NOT in the key (see _TIER_RANK comment) —
        # it stays a logged fact, never a ranking thumb.
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

        return {
            "exists": True,
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


def features_from_snapshot(snap, df_h1, alert_ts):
    """SWEEP2_FEATURE_COLUMNS dict from the FROZEN ob['sweep_v2'] snapshot.

    Pure re-labelling of frozen fields — nothing is re-detected or re-graded
    post-alert. The one derived value, sweep2_age_at_alert_h1, is arithmetic
    on the frozen sweep_ts against the alert bar (same class as
    ob_age_h1_bars): closed H1 bars strictly before alert_ts, minus one, minus
    the sweep bar's position. Never raises — returns the all-None dict on any
    failure so a sweep bug can never kill a run row.
    """
    out = features_none()
    try:
        if not isinstance(snap, dict) or snap.get("pools_swept") is None:
            return out  # legacy zone / layer failed — every column stays None
        out["sweep2_present"] = bool(snap.get("exists"))
        out["sweep2_tier"] = snap.get("tier")
        out["sweep2_level"] = snap.get("level")
        out["sweep2_pierce_atr"] = snap.get("pierce_atr")
        out["sweep2_rejection_ratio"] = snap.get("rejection_ratio")
        out["sweep2_follow_atr"] = snap.get("follow_atr")
        out["sweep2_pools_swept"] = snap.get("pools_swept")
        out["sweep2_rn_aligned"] = snap.get("rn_aligned")
        out["sweep2_rn_dist_atr"] = snap.get("rn_dist_atr")
        out["sweep2_eq_size"] = snap.get("eq_size")
        out["sweep2_tiers_checked"] = snap.get("tiers_checked")
        sweep_ts = snap.get("sweep_ts")
        if sweep_ts is not None and df_h1 is not None and alert_ts is not None:
            h1 = _naive_utc_index(df_h1)
            if isinstance(h1.index, pd.DatetimeIndex):
                a_ts = pd.Timestamp(alert_ts)
                if a_ts.tzinfo is not None:
                    a_ts = a_ts.tz_convert("UTC").tz_localize(None)
                s_ts = pd.Timestamp(sweep_ts)
                if s_ts.tzinfo is not None:
                    s_ts = s_ts.tz_convert("UTC").tz_localize(None)
                a_pos = int(h1.index.searchsorted(a_ts))  # bars before alert
                s_pos = int(h1.index.searchsorted(s_ts))
                if a_pos > s_pos:
                    out["sweep2_age_at_alert_h1"] = int(a_pos - 1 - s_pos)
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
# logged (sweep2_rejection_ratio / sweep2_follow_atr) for the full run to judge,
# never scored. PW(3) > PD(2.5) > EQ(2) > SW(1): a bigger raided pool is stronger
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
