"""Setup-liquidity reads — this trade's own STOP and TARGET vs resting swing
liquidity, plus whether the OB leg's own extreme was itself a sweep.

WHY THIS EXISTS (SWING_SWEEP_SPEC.md, corrected by the owner in discussion —
the corrections WIN over the spec; spec is background only):
  Sweep v2 (liquidity_sweep.py) answers "how was this OB built" — did the leg
  raid a resting pool (PW/PD/EQ) or bare swing (SW). What it does NOT answer is
  the liquidity picture around THIS trade's own stop and target. Three
  observe-only reads fill that gap:

    Read 1 — STOP-side liquidity. Is there an ACTIVE (unbroken AND unspent)
             swing that could be swept in the region of my stop? For a LONG we
             hunt sell-side liquidity = active swing LOWS near the SL. The
             SIGNED offset from the SL is the signal:
               - negative (swing below SL): a sweep must blow through the stop
                 then recover — "stopped out, then it ran without me".
               - ~0 (swing at SL): exact stop-hunt hit.
               - positive (swing inside risk, between entry and SL): the sweep +
                 reverse happens BEFORE the stop is hit — the survive-the-hunt
                 case, the profitable one. A SYMMETRIC band is mandatory or we
                 blind ourselves to it.

    Read 2 — TP-side magnet (Draw on Liquidity). Is my target sitting on UNSPENT
             liquidity that pulls price toward it? For a LONG the target is
             above -> active swing HIGHS near the TP. Unspent = positive (fresh
             fuel, price is drawn to it); a 1:1-fallback TP sits on no swing by
             construction and correctly reads "no magnet".

    Read 3.2 — Leg-extreme-was-a-sweep. Did the OB's displacement leg END by
             grabbing fresh liquidity — i.e. was the leg's own terminal extreme
             itself a sweep of an active swing, then a reversal? An
             institutional-intent tell on the setup itself. This is Q1 ("was the
             extreme itself a sweep"), knowable at OB build — NOT Q2 ("has the
             extreme been swept since", which needs post-leg bars).

ONE non-negotiable rule across every read: only ACTIVE swings count
(smc_detector.is_swing_active — unbroken AND unspent). A swing already wicked or
broken holds no liquidity.

ANCHORING / TIMING (owner-corrected):
  - Reads 1 & 2 anchor on the SL / TP1, which are BORN when
    compute_phase2_levels runs (live: alert; backtest: level-calc step). There
    is nothing to freeze at OB build because the anchor does not exist yet.
    compute_phase2_levels is deterministic on the closed-bar frame, so these
    reads are computed ONCE with the levels — no per-scan re-scan (rule F
    intent honoured), and live/backtest stay symmetric because both use the
    same frozen closed-bar frame the levels themselves use.
  - Read 3.2 is pure leg geometry (no SL/TP), so it FREEZES at OB build on the
    zone, exactly like sweep_v2.

SWING SOURCE: dealing_range.detect_swings(lookback=3) — the ONE H1 swing
definition (lb-3 geometry + 1.5-ATR leg filter). Never the raw EQ pool.

ONE sweep judge: smc_detector.is_swing_active (the unspent filter) +
pool_builder.pool_status (the leg-extreme sweep event). No new sweep definition.

OBSERVATION ONLY. Nothing here gates, scores, ranks or filters. Never raises —
degraded input returns the all-None feature dict / None reads, so a bug here can
never kill a level computation or an OB build (guard lives OUT of the live
path).
"""

from datetime import datetime, timezone

import pandas as pd

import dealing_range
import smc_detector
from pool_builder import _naive_utc_index, levels_at, pool_status
from eq_pools import clusters_at


# Band half-width in ATR either side of the SL / TP anchor (SWING_SWEEP_SPEC §4,
# owner-LOCKED). Symmetric. WHY 0.5 ATR:
#   - Floor: must exceed the p95 MT5-vs-TwelveData feed gap (5-12 pips) so feed
#     variance never flips a flag. On EURUSD 0.5 ATR ~ 25-40 pips, safely above.
#   - Ceiling: must stay LOCAL to the anchor. Wider starts catching swings that
#     belong to the other end of the trade (double-counting).
BAND_ATR_MULT = 0.5

# Tier ladder for the stop-swing's pool coincidence (Read 1 setup_liq_stop_tier).
# The same ranking sweep v2 uses: a swing that also sits on a ranked pool is a
# more meaningful stop-hunt than a bare swing. PW > PD > EQ > bare.
_STOP_TIER_RANK = {"PW": 0, "PD": 1, "EQ": 2, "bare": 3}

# The trades.csv column set this module owns. One list, one implementation — the
# backtest row build and the None-fallback both key off it (EQ/sweep2 precedent).
SETUP_LIQ_FEATURE_COLUMNS = (
    "setup_liq_stop_present",
    "setup_liq_stop_offset_atr",
    "setup_liq_stop_tier",
    "setup_liq_tp_present",
    "setup_liq_tp_offset_atr",
    "setup_liq_legextreme_swept",
)


def features_none():
    """All-None dict — the honest value when a read could not run (degraded
    input / legacy zone / failed layer)."""
    return {col: None for col in SETUP_LIQ_FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pierce_min(pair_type, atr):
    """Minimum wick pierce (price units) to count a swing as swept — the SAME
    constant the sweep engine and the TP unbroken-filter use (no new knob)."""
    return smc_detector.SWEEP_WICK_PIERCE_MIN_ATR.get(pair_type, 0.05) * atr


def _swings_lb3(h1):
    """The ONE H1 swing definition: lb-3 geometry + 1.5-ATR leg filter."""
    return dealing_range.detect_swings(h1, lookback=dealing_range.SWING_LOOKBACK)


def _active_swing_in_band(h1, swings, want_type, anchor, atr, pierce_min):
    """Nearest ACTIVE (unbroken + unspent) swing of `want_type` whose price is
    within BAND_ATR_MULT*atr of `anchor`. Returns the swing dict or None.

    "Active" is judged as of the LAST closed bar in h1 (before_idx defaults to
    len(h1) inside is_swing_active) — the frame passed here is already
    closed-only and point-in-time, so this is look-ahead-safe. Nearest = the
    smallest |price - anchor|, so the offset we log is the closest resting
    liquidity to the stop/target, not a farther one that happens to be active.
    """
    band = BAND_ATR_MULT * atr
    best, best_dist = None, None
    for s in swings:
        if s.get("type") != want_type:
            continue
        dist = abs(float(s["price"]) - anchor)
        if dist > band:
            continue
        if not smc_detector.is_swing_active(s, h1, pierce_min):
            continue
        if best is None or dist < best_dist:
            best, best_dist = s, dist
    return best


def _stop_tier(h1, swing, side, atr, swings):
    """Pool coincidence of the stop swing: 'PW' | 'PD' | 'EQ' | 'bare'.

    Does the stop swing's price sit on a ranked pool (a bigger, more meaningful
    stop-hunt) or is it a bare swing? Reuses the ONE judges: levels_at for PW/PD
    prices, clusters_at for EQ shelves. Tolerance = the same feed-buffered
    equal-level band the sweep engine uses (SWEEP_EQUAL_LEVEL_TOLERANCE_ATR), so
    a swing a pip off the pool still counts as coincident. Returns 'bare' when
    the swing lands on no ranked pool. Never raises -> 'bare' on any failure."""
    try:
        level = float(swing["price"])
        tol = smc_detector.SWEEP_EQUAL_LEVEL_TOLERANCE_ATR.get(
            "forex", 0.30) * atr  # pair_type-agnostic band; ATR carries the pair scale
        s_ts = pd.Timestamp(swing["ts"]) if swing.get("ts") else h1.index[-1]
        if s_ts.tzinfo is not None:
            s_ts = s_ts.tz_convert("UTC").tz_localize(None)
        lv = levels_at(h1, s_ts)
        # PW outranks PD; check the fuel side matching the swing type.
        want_low = (side == "below")
        pw_key = "pwl" if want_low else "pwh"
        pd_key = "pdl" if want_low else "pdh"
        for tier, key in (("PW", pw_key), ("PD", pd_key)):
            lvl = lv.get(key)
            if lvl is not None and abs(float(lvl) - level) <= tol:
                return tier
        # EQ shelf coincidence — clusters as of the swing bar.
        s_pos = int(h1.index.searchsorted(s_ts))
        clusters = clusters_at(h1, s_pos, atr, swings=swings)
        if clusters:
            want_type = "low" if want_low else "high"
            for cl in clusters:
                if cl["side"] == want_type and abs(float(cl["level"]) - level) <= tol:
                    return "EQ"
        return "bare"
    except Exception:
        return "bare"


# ---------------------------------------------------------------------------
# Read 1 & 2 — computed WITH the trade levels (compute_phase2_levels output)
# ---------------------------------------------------------------------------

def reads_stop_and_tp(df_h1, direction, sl, tp1, atr, pair_type,
                      tp1_is_fallback=False):
    """Reads 1 (stop-side) & 2 (tp-side magnet) as a dict of the four columns
    they own. Anchored on the SL / TP1 born from compute_phase2_levels.

    Args:
      df_h1        — the closed-bar H1 frame the levels were computed on (same
                     point-in-time slice; look-ahead-safe).
      direction    — 'bullish' | 'bearish' (LONG hunts stop-side lows / tp-side
                     highs; SHORT mirrors).
      sl, tp1      — the trade's stop / target prices (compute_phase2_levels).
      atr          — H1 ATR for the band + offset normalisation (ob['h1_atr'],
                     the shared *_atr denominator).
      pair_type    — for the sweep pierce constant.
      tp1_is_fallback — True when TP1 is the mechanical 1:1 (no swing anchor);
                     the tp-side magnet then reads absent by construction (there
                     is no resting pool behind a 1:1), logged honestly as False.

    Returns {setup_liq_stop_present, setup_liq_stop_offset_atr,
             setup_liq_stop_tier, setup_liq_tp_present, setup_liq_tp_offset_atr}.
    Never raises — any failure returns those five keys as None.
    """
    out = {
        "setup_liq_stop_present": None,
        "setup_liq_stop_offset_atr": None,
        "setup_liq_stop_tier": None,
        "setup_liq_tp_present": None,
        "setup_liq_tp_offset_atr": None,
    }
    try:
        if df_h1 is None or len(df_h1) == 0:
            return out
        if direction not in ("bullish", "bearish"):
            return out
        if atr is None or atr <= 0 or sl is None:
            return out
        h1 = _naive_utc_index(df_h1)
        if not isinstance(h1.index, pd.DatetimeIndex):
            return out
        swings = _swings_lb3(h1)
        pierce_min = _pierce_min(pair_type, atr)

        # ── Read 1 — STOP-side ────────────────────────────────────────────────
        # LONG hunts sell-side liquidity (swing LOWS) near the SL; SHORT hunts
        # buy-side (swing HIGHS). Signed offset = (swing - SL)/ATR: for a LONG,
        # positive = swing ABOVE the SL = INSIDE risk (survive-the-hunt);
        # negative = swing BELOW the SL (stopped-then-runs).
        stop_type = "low" if direction == "bullish" else "high"
        stop_side = "below" if direction == "bullish" else "above"
        stop_sw = _active_swing_in_band(h1, swings, stop_type, float(sl),
                                        atr, pierce_min)
        out["setup_liq_stop_present"] = stop_sw is not None
        if stop_sw is not None:
            out["setup_liq_stop_offset_atr"] = round(
                (float(stop_sw["price"]) - float(sl)) / atr, 3)
            out["setup_liq_stop_tier"] = _stop_tier(h1, stop_sw, stop_side,
                                                     atr, swings)

        # ── Read 2 — TP-side magnet ───────────────────────────────────────────
        # LONG's target sits above -> hunt active swing HIGHS near TP1; SHORT
        # mirrors. A 1:1-fallback TP1 sits on no pool -> no magnet (False), the
        # honest read (never a bug). Signed offset = (swing - TP1)/ATR.
        if tp1 is not None and not tp1_is_fallback:
            tp_type = "high" if direction == "bullish" else "low"
            tp_sw = _active_swing_in_band(h1, swings, tp_type, float(tp1),
                                          atr, pierce_min)
            out["setup_liq_tp_present"] = tp_sw is not None
            if tp_sw is not None:
                out["setup_liq_tp_offset_atr"] = round(
                    (float(tp_sw["price"]) - float(tp1)) / atr, 3)
        else:
            # No pool behind a 1:1 fallback -> magnet absent by construction.
            out["setup_liq_tp_present"] = False
        return out
    except Exception as e:
        print(f"  [SETUP_LIQ WARN] reads_stop_and_tp failed: "
              f"{type(e).__name__}: {e}")
        return {k: None for k in out}


# ---------------------------------------------------------------------------
# Read 3.2 — leg-extreme-was-a-sweep — FROZEN at OB build
# ---------------------------------------------------------------------------

def read_legextreme_swept(df, leg_extreme, extreme_end_idx, direction,
                          pair_type, atr):
    """Read 3.2: did the OB leg's OWN terminal extreme sweep an active swing?

    Q1 (owner): "was the extreme ITSELF a sweep" — the leg ended by grabbing
    fresh liquidity at an active swing, then reversed. Knowable at OB build (the
    leg is complete). NOT Q2 ("has the extreme been swept since" — that needs
    post-leg bars and is a different, alert-time question we are NOT asking).

    Definition: the leg's extreme candle (at extreme_end_idx) pierced an ACTIVE
    same-side swing beyond it by >= pierce_min, then the extreme was NOT extended
    (the leg turned) — i.e. the extreme candle itself is the deepest point. We
    ask pool_status over the leg's tail whether that extreme took a resting
    swing's liquidity.

    Args:
      df               — the OB-build detection frame (funnelled through
                         _naive_utc_index).
      leg_extreme      — the leg's terminal extreme PRICE
                         (displacement_leg.compute_leg_extreme_er).
      extreme_end_idx  — position of the extreme bar in df (same core).
      direction        — 'bullish' | 'bearish'.
      pair_type, atr   — pierce constant + normalisation.

    Returns bool (swept / not) or None on degraded input. Never raises.
    """
    try:
        if df is None or len(df) == 0 or direction not in ("bullish", "bearish"):
            return None
        if leg_extreme is None or extreme_end_idx is None:
            return None
        if atr is None or atr <= 0:
            return None
        h1 = _naive_utc_index(df)
        if not isinstance(h1.index, pd.DatetimeIndex):
            return None
        end_pos = int(extreme_end_idx)
        if not (0 <= end_pos < len(h1)):
            return None
        pierce_min = _pierce_min(pair_type, atr)
        swings = _swings_lb3(h1)
        # A bullish leg drives UP -> its extreme is a HIGH that may have swept an
        # active swing HIGH (buy-side liquidity above); bearish mirrors.
        want_type = "high" if direction == "bullish" else "low"
        side = "above" if direction == "bullish" else "below"
        # Consider only swings that PREDATE the extreme bar and were ACTIVE as of
        # the extreme bar (resting liquidity the extreme could take).
        for s in swings:
            if s.get("type") != want_type:
                continue
            s_idx = int(s["idx"])
            if s_idx >= end_pos:
                continue
            if not smc_detector.is_swing_active(s, h1, pierce_min,
                                                before_idx=end_pos):
                continue
            # Did the extreme bar itself sweep this swing (wick+close-back)? Walk
            # pool_status over just the extreme bar (one-bar window) — the ONE
            # sweep judge, no forked definition.
            leg_tail = h1.iloc[end_pos: end_pos + 1]
            st = pool_status(leg_tail, float(s["price"]), side)
            if st["status"] == "swept":
                return True
        return False
    except Exception as e:
        print(f"  [SETUP_LIQ WARN] read_legextreme_swept failed: "
              f"{type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Backtest row-build entry point (spread precedent: eq_pools / liquidity_sweep)
# ---------------------------------------------------------------------------

def features_from_reads(reads_dict, legextreme_swept):
    """Assemble the SETUP_LIQ_FEATURE_COLUMNS dict from the Read 1/2 dict
    (reads_stop_and_tp output) and the frozen Read 3.2 bool. Pure re-labelling —
    nothing re-detected here. Never raises."""
    out = features_none()
    try:
        if isinstance(reads_dict, dict):
            for k in ("setup_liq_stop_present", "setup_liq_stop_offset_atr",
                      "setup_liq_stop_tier", "setup_liq_tp_present",
                      "setup_liq_tp_offset_atr"):
                out[k] = reads_dict.get(k)
        out["setup_liq_legextreme_swept"] = legextreme_swept
        return out
    except Exception as e:
        print(f"  [SETUP_LIQ WARN] features_from_reads failed: "
              f"{type(e).__name__}: {e}")
        return features_none()


# ---------------------------------------------------------------------------
# Plain-English narration (P2 email)
# ---------------------------------------------------------------------------

def describe_stop(present, offset_atr, tier):
    """One-line stop-side read for the email. None when not present."""
    if not present or offset_atr is None:
        return None
    pool = "" if tier in (None, "bare") else f" ({tier})"
    if offset_atr > 0.05:
        # swing inside risk (LONG: above SL) -> survive-the-hunt
        return (f"Sell-side liquidity sits inside your risk{pool} — a stop-hunt "
                f"here reverses BEFORE your stop is hit (survive-the-hunt).")
    if offset_atr < -0.05:
        return (f"Sell-side liquidity sits beyond your stop{pool} — a sweep must "
                f"blow through the stop, then recover.")
    return f"Liquidity sits right on your stop{pool} — an exact stop-hunt level."


def describe_tp(present, offset_atr):
    """One-line tp-side magnet read for the email. None when the read did not
    run (present is None -> legacy/degraded, render nothing); a real False (the
    read ran and found no magnet) narrates the no-draw case."""
    if present is None:
        return None
    if not present:
        return "Target sits on no resting swing — no liquidity magnet drawing price toward it."
    return "Target sits on unspent resting liquidity — a Draw-on-Liquidity magnet toward the TP."


def describe_legextreme(swept):
    """One-line leg-extreme read for the email. None when unknown."""
    if swept is None:
        return None
    if swept:
        return "The leg ENDED on a sweep — it grabbed fresh liquidity at its extreme, then reversed (institutional intent)."
    return "The leg's extreme took no resting liquidity — a plain displacement, not a stop-run finish."
