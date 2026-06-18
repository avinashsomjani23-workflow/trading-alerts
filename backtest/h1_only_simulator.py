"""H1-only dual-entry trade simulator.

Tests the SMC system using ONLY H1 data — H1 finds the OB, entry happens at
the OB, SL/TP are sized off the H1 OB and H1 swing liquidity. No M15, no M5.

For every H1 OB-touch alert, this simulator fires TWO trade rows:
  1. Proximal entry  — fills the bar the OB is touched (entry = OB proximal).
  2. 50% mean entry  — pending limit at OB midpoint; fills only if price
                       penetrates deep enough.

Both share the same SL (OB distal +/- spread) and the same TP price levels
(liquidity-based opposing H1 swings, reused from live compute_phase2_levels).
R-distance and RR-multiples differ per entry zone by construction.

No scoring gate — every OB-touch fires regardless of confluence score. Score
is logged for post-run analysis (discover the optimal threshold empirically).

Per-trade outputs cover both exit policies (TP1 vs TP2) so the user can
see what their real-life TP1-only behaviour would produce vs the system's
default TP2 target.

Hard rule (matches live simulator): same-bar SL+TP collision resolves SL-first.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import timedelta
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import smc_detector  # live, read-only

from backtest.run_logger import log_event


# H1 trade hold limit. 48 H1 bars = 2 trading days. Long enough for a swing
# setup to play out; short enough to release capital and avoid degenerate
# trades that never resolve.
MAX_HOLD_H1_BARS = 48
DEFAULT_RISK_USD = 250.0


def _session_from_utc_hour(h: int) -> str:
    """Map UTC hour -> trading session label. Matches reporting._classify_session_utc."""
    if 0 <= h < 7:
        return "Asia"
    if 7 <= h < 13:
        return "London"
    if 13 <= h < 21:
        return "NY"
    return "Other"


def _ts_hour_utc(ts_val) -> Optional[int]:
    """Coerce ts (str / pd.Timestamp / None) to UTC hour, or None if unparseable."""
    if ts_val is None or ts_val == "":
        return None
    try:
        ts = pd.Timestamp(ts_val)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.hour)
    except Exception:
        return None


def _ob_session(ob: Dict[str, Any]) -> str:
    """Session label for the OB candle itself (when the institutional move
    that created the zone happened). 'unknown' if ob_timestamp missing."""
    h = _ts_hour_utc(ob.get("ob_timestamp"))
    return _session_from_utc_hour(h) if h is not None else "unknown"


def _fill_session(fill_ts, alert_ts) -> str:
    """Session at fill (when capital was at work). Falls back to alert hour
    for never_filled rows so the column is never empty."""
    h = _ts_hour_utc(fill_ts) if fill_ts is not None else None
    if h is None:
        h = _ts_hour_utc(alert_ts)
    return _session_from_utc_hour(h) if h is not None else "unknown"


def _ts_in_killzone(ts_val, pair_conf: Dict[str, Any]) -> bool:
    """DST-aware killzone membership for a full timestamp. Routes through the
    shared smc_detector engine so the backtest resolves the SAME UTC windows
    the live engine does, per candle date. The full date matters: the same UTC
    hour can be in/out of a killzone depending on the EDT/EST season."""
    if ts_val is None or ts_val == "":
        return False
    killzones = pair_conf.get("killzones")
    if not killzones:
        return False
    try:
        ts = pd.Timestamp(ts_val)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return smc_detector.ts_in_killzone(ts.isoformat(), killzones)
    except Exception:
        return False


def _ob_in_killzone(ob: Dict[str, Any], pair_conf: Dict[str, Any]) -> bool:
    return _ts_in_killzone(ob.get("ob_timestamp"), pair_conf)


def _fill_in_killzone(fill_ts, pair_conf: Dict[str, Any]) -> bool:
    return _ts_in_killzone(fill_ts, pair_conf)


def _killzone_alignment(ob: Dict[str, Any], fill_ts, alert_ts,
                        pair_conf: Dict[str, Any]) -> str:
    """4-way bucket for the SMC veteran hypothesis test:
       - 'Both'    : OB candle AND fill candle both fell in a killzone window
       - 'OB only' : OB in killzone, fill outside
       - 'Fill only': fill in killzone, OB outside
       - 'Neither' : both outside
       - 'never_filled': fill_ts is None (no fill happened)
    """
    if fill_ts is None:
        return "never_filled"
    ob_kz = _ob_in_killzone(ob, pair_conf)
    fl_kz = _fill_in_killzone(fill_ts, pair_conf)
    if ob_kz and fl_kz:
        return "Both"
    if ob_kz:
        return "OB only"
    if fl_kz:
        return "Fill only"
    return "Neither"


def _pd_zone_from_dr(price: float, dr: Optional[Dict[str, Any]]) -> str:
    """Where in the dealing range is `price`?
       discount = lower 40%, premium = upper 40%, equilibrium = middle 20%.
       Returns 'unknown' if dealing range data is missing/invalid.
    """
    if not isinstance(dr, dict) or not dr.get("valid"):
        return "unknown"
    try:
        rng_low = float(dr["range_low"])
        rng_high = float(dr["range_high"])
    except (KeyError, TypeError, ValueError):
        return "unknown"
    width = rng_high - rng_low
    if width <= 0:
        return "unknown"
    pos = (price - rng_low) / width  # 0.0 at low, 1.0 at high
    if pos < 0.40:
        return "discount"
    if pos > 0.60:
        return "premium"
    return "equilibrium"


def _pd_alignment(bias: str, pd_zone: str) -> str:
    """Direction-aware PD-array read. Raw discount/premium is meaningless
    without the trade direction: SMC wants LONGS in discount and SHORTS in
    premium. The opposite (long in premium / short in discount) is a red flag,
    not a confluence -- the old pd_zone column could not tell them apart.

       aligned  = with the draw on liquidity (long+discount / short+premium)
       counter  = against it (long+premium / short+discount)
       neutral  = equilibrium (middle of the range)
       unknown  = no valid dealing range
    """
    if pd_zone in (None, "unknown"):
        return "unknown"
    if pd_zone == "equilibrium":
        return "neutral"
    if bias == "LONG":
        return "aligned" if pd_zone == "discount" else "counter"
    return "aligned" if pd_zone == "premium" else "counter"


def _confluences_present(breakdown: Dict[str, float]) -> str:
    """Comma-separated list of confluences that scored > 0 on this OB.
    Killzone removed 2026-05-25 (no longer a scoring input)."""
    names = []
    if breakdown.get("structure", 0) > 0:
        names.append("structure")
    if breakdown.get("sweep", 0) > 0:
        names.append("sweep")
    if breakdown.get("fvg", 0) > 0:
        names.append("fvg")
    if breakdown.get("freshness", 0) > 0:
        names.append("freshness")
    return ",".join(names) if names else "none"


def _event_label(bos_tag: Optional[str], bos_tier: Optional[str]) -> str:
    """One-column event label: 'Major BOS' / 'Minor BOS' / 'Major CHoCH' / 'Minor CHoCH'."""
    tag = bos_tag or "BOS"
    tier = bos_tier or "Major"
    return f"{tier} {tag}"


# FVG re-arm distance for fresh-vs-stale classification. Mirrors
# REARM_EXTRA_ATR in replay_engine.py (=1.0); that one is defined inside a
# function so it can't be imported cleanly. Same number, anchored to the FVG
# band here instead of the OB proximal. If the replay constant changes, change
# this too.
_FVG_REARM_ATR = 1.0


def _fvg_state(ob: Dict[str, Any], df_h1: pd.DataFrame,
               alert_ts: pd.Timestamp) -> str:
    """Classify the FVG at trigger time: 'fresh' | 'stale' | 'no_fvg'.

    no_fvg : no FVG ever formed in this zone -> excluded from the headline.
    fresh  : FVG still live at trigger (incl. partial), OR it was filled during
             THIS approach to the zone. First-approach pass-through is fresh:
             price must cross the FVG to reach the OB, so a same-visit fill is
             healthy, not stale.
    stale  : FVG was fully filled, price then LEFT the FVG band (cleared it by
             the re-arm distance) and RETURNED to trigger. The imbalance was
             already discharged on an earlier trip.

    Anchored to the FVG band (ghost_top/ghost_bottom), NOT the OB proximal, so a
    fill-then-reverse-before-the-OB-then-return is correctly stale. Uses
    mitigated_at_iso plumbed from smc_detector. Never raises -> defaults 'fresh'
    (the non-penalising bucket) on any missing data."""
    fvg = ob.get("fvg") or {}
    if not fvg.get("was_detected"):
        return "no_fvg"
    if fvg.get("exists"):
        return "fresh"                      # live at trigger (incl. partial)
    fill_iso = fvg.get("mitigated_at_iso")
    top, bot = fvg.get("ghost_top"), fvg.get("ghost_bottom")
    if not fill_iso or top is None or bot is None:
        return "fresh"                      # filled but no timing/band -> don't penalise
    try:
        fill_ts = pd.Timestamp(fill_iso)
        if fill_ts.tzinfo is None:
            fill_ts = fill_ts.tz_localize("UTC")
        rearm = _FVG_REARM_ATR * float(ob.get("h1_atr") or 0.0)
        win = df_h1.loc[fill_ts:alert_ts]   # bars from fill up to the trigger
        if win.empty:
            return "fresh"
        # Did price pull clear of the FVG band by the re-arm distance after
        # filling it? Above the top or below the bottom counts as "left".
        left = ((win["Low"] > top + rearm) | (win["High"] < bot - rearm)).any()
        return "stale" if bool(left) else "fresh"
    except Exception:
        return "fresh"


def _ob_age_h1_bars(ob: Dict[str, Any], df_h1: pd.DataFrame,
                    alert_ts: pd.Timestamp) -> int:
    """How many H1 bars old is this OB at the alert moment?"""
    ob_ts_iso = ob.get("ob_timestamp")
    if not ob_ts_iso:
        return -1
    try:
        ob_ts = pd.Timestamp(ob_ts_iso)
        if ob_ts.tzinfo is None:
            ob_ts = ob_ts.tz_localize("UTC")
        # Use df_h1 index to count actual H1 bars between ob_ts and alert_ts,
        # not wall-clock hours (avoids weekend / data-gap inflation).
        in_window = df_h1.loc[ob_ts:alert_ts]
        # subtract 1 so age=0 means "alert on the OB-formation bar itself"
        return max(0, len(in_window) - 1)
    except Exception:
        return -1


def _score_h1_only(alert: Dict[str, Any], pair_conf: Dict[str, Any],
                   df_h1: pd.DataFrame, alert_ts: pd.Timestamp
                   ) -> Tuple[float, Dict[str, float]]:
    """Score the OB using live run_scorecard (H1-only since 2026-05-26).
    Returns (total, breakdown). Never raises — bad scores log and return 0.
    """
    ob = alert["ob"]
    bias = "LONG" if ob.get("direction") == "bullish" else "SHORT"
    h1_slice = df_h1.loc[:alert_ts]
    fvg_h1 = ob.get("fvg", {"exists": False, "was_detected": False,
                            "mitigation": "none"})
    fvg_data = {"h1": fvg_h1}
    try:
        score_res = smc_detector.run_scorecard(
            bias, h1_slice, ob, fvg_data, alert["current_price"],
            pair_conf,
        )
    except Exception as e:
        log_event("h1only_scorecard_error", level="warn",
                  pair=alert.get("pair"),
                  error=f"{type(e).__name__}: {e}")
        return 0.0, {}
    breakdown = dict(score_res.get("breakdown", {}))
    # Killzone IS scored (2026-06-18) on the OB-FORMATION candle. The hard
    # filter gates the entry/alert time, NOT the OB candle, so this score is
    # independent of the filter and must flow through to the backtest total.
    total = round(sum(float(v) for v in breakdown.values()), 1)
    return total, breakdown


def _simulate_single_entry(
    alert: Dict[str, Any],
    pair_conf: Dict[str, Any],
    df_h1: pd.DataFrame,
    entry_zone: str,
    score: float,
    breakdown: Dict[str, float],
    risk_usd: float,
    forced_tp1: Optional[float] = None,
    forced_tp2: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Simulate one trade for one entry zone ('proximal' or '50pct').

    Returns a row dict or None if the trade is invalid (e.g. no TP1 clearing
    1.5R — same gate as live). Returns "never_filled" row for 50pct entries
    that don't get filled within the hold window, so we can count the miss.
    """
    ob = alert["ob"]
    pair = alert["pair"]
    bias = "LONG" if ob.get("direction") == "bullish" else "SHORT"
    alert_ts = alert["ts"]
    if not isinstance(alert_ts, pd.Timestamp):
        alert_ts = pd.Timestamp(alert_ts)
    if alert_ts.tzinfo is None:
        alert_ts = alert_ts.tz_localize("UTC")
    # Fill walk starts on the bar that OPENS at alert_ts (the bar still
    # forming when the alert fires). The just-closed bar that triggered the
    # alert is NOT a fill candidate — at the moment its wick was making the
    # move that triggered proximity, the limit order didn't exist yet (the
    # alert hadn't fired). The earliest a live broker could fill the limit
    # is during the bar that starts at alert_ts. Same-bar fills (within
    # this opening bar) ARE allowed and common — market momentum that
    # carries straight into the zone.
    current_price = alert["current_price"]

    # Lookahead guard (2026-06): TP/SL levels must be computed from ONLY the
    # bars a live trader could see at the alert -- bars that had already CLOSED.
    # The alert fires at alert_ts (the bar opening then is still forming), so
    # closed bars are those indexed strictly before alert_ts. Passing the full
    # df_h1 let compute_phase2_levels.get_swing_points pick opposing swings that
    # formed AFTER the alert (future liquidity), biasing both TP selection and
    # the 1.5R validity gate optimistically. This mirrors the slice the scoring
    # path already uses (_score_h1_only) and live behaviour. The forward
    # fill-walk below intentionally keeps the FULL df_h1 -- it must see the
    # future to simulate how the trade plays out.
    df_h1_at_alert = df_h1.loc[df_h1.index < alert_ts]
    if df_h1_at_alert.empty:
        df_h1_at_alert = df_h1.loc[:alert_ts]  # degenerate guard, never empty

    try:
        levels = smc_detector.compute_phase2_levels(
            pair_conf, bias, ob, current_price, df_h1_at_alert,
            entry_zone=entry_zone,
        )
    except Exception as e:
        log_event("h1only_levels_error", level="error", pair=pair,
                  entry_zone=entry_zone, alert_ts=str(alert_ts),
                  error=f"{type(e).__name__}: {e}")
        return None

    if forced_tp1 is not None:
        # Shared-TP mode: only need entry and sl from levels (may be "invalid"
        # because the TP gate failed from the 50pct entry, but entry/sl are
        # still present in the dict after the smc_detector change).
        if not isinstance(levels, dict) or "entry" not in levels:
            log_event("h1only_sim_skip", level="info", pair=pair,
                      entry_zone=entry_zone, alert_ts=str(alert_ts),
                      reason="entry_not_computable_for_forced_tp_mode")
            return None
        entry  = float(levels["entry"])
        sl     = float(levels["sl"])
        tp1    = float(forced_tp1)
        tp2    = float(forced_tp2) if forced_tp2 is not None else None
        r_dist = abs(entry - sl)
        tp1_rr = abs(tp1 - entry) / r_dist if r_dist > 0 else 0.0
        tp2_rr = abs(tp2 - entry) / r_dist if (r_dist > 0 and tp2 is not None) else 0.0
    else:
        if not levels or not levels.get("valid", False):
            log_event("h1only_sim_skip", level="info", pair=pair,
                      entry_zone=entry_zone, alert_ts=str(alert_ts),
                      reason=levels.get("reason", "levels_invalid")
                             if isinstance(levels, dict) else "levels_none")
            return None
        entry  = float(levels["entry"])
        sl     = float(levels["sl"])
        tp1    = float(levels["tp1"])
        tp2    = float(levels["tp2"]) if levels.get("tp2") is not None else None
        tp1_rr = float(levels.get("rr", 0.0))
        tp2_rr = float(levels.get("tp2_rr", 0.0)) if tp2 is not None else 0.0

    # Apply pair spread to widen SL (worst-case execution). spread_pips is
    # the pair's typical broker spread. pip_size derived from decimal_places:
    # 4-5dp instruments (EURUSD, NZDUSD, USDCHF) -> pip = 0.0001
    # 2-3dp instruments (USDJPY, GOLD, NAS100)   -> pip = 0.01
    # For a LONG, SL sits below entry; spread pushes SL further down (worse).
    # For a SHORT, SL sits above entry; spread pushes SL further up (worse).
    # TP levels are NOT widened -- pessimistic, matches what the user gets
    # at the bid/ask after entering. Slippage and swap are NOT modelled
    # (user decision; revisit when needed). RCA #9.
    spread_pips = float(pair_conf.get("spread_pips", 0.0))
    decimal_places = int(pair_conf.get("decimal_places", 5))
    pip_size = 0.01 if decimal_places <= 3 else 0.0001
    spread_price = spread_pips * pip_size
    if spread_price > 0:
        if bias == "LONG":
            sl = sl - spread_price
        else:
            sl = sl + spread_price

    r_distance = abs(entry - sl)
    if r_distance <= 0:
        log_event("h1only_sim_skip", level="warn", pair=pair,
                  entry_zone=entry_zone, alert_ts=str(alert_ts),
                  reason="zero_r_distance")
        return None

    # Defense in depth: drop the trade if TP2 is on the wrong side of TP1.
    # compute_phase2_levels already filters this; this guard catches any
    # future regression or forced-TP path where the upstream check is bypassed.
    if tp2 is not None:
        bad = (bias == "LONG" and tp2 <= tp1) or (bias == "SHORT" and tp2 >= tp1)
        if bad:
            log_event("h1only_sim_skip", level="error", pair=pair,
                      entry_zone=entry_zone, alert_ts=str(alert_ts),
                      reason="tp_order_invalid",
                      tp1=tp1, tp2=tp2, bias=bias)
            return None

    # Walk H1 bars from (alert_ts + 1H) forward up to MAX_HOLD_H1_BARS bars.
    # The alert fires when its bar CLOSES; a live broker can only place the
    # limit AFTER close, so the earliest legal fill is the NEXT bar's open.
    # Including the alert bar caused 283/1792 cloned-fill rows in the
    # 2026-03 backtest -- physically impossible "fills" on the bar that
    # only just published the alert (see RCA #2, #4).
    # Two separate clocks:
    #   - Pre-fill:  limit pends at most MAX_HOLD_H1_BARS bars after alert.
    #                If price never comes back to entry inside that window,
    #                emit never_filled.
    #   - Post-fill: once filled, trade runs at most MAX_HOLD_H1_BARS bars
    #                before forced timeout. Independent of pre-fill wait.
    # Earlier version pre-sliced future to MAX_HOLD_H1_BARS bars TOTAL from
    # alert, conflating the two clocks. A trade that pended N bars only got
    # MAX_HOLD - N bars to play out, producing false window_end exits at
    # whatever R the position happened to sit at when the slice ran out
    # (often a positive number for trades that would have hit SL one bar
    # later). See RCA Mar 9 EURUSD long, filled bar 46/48 -> window_end at
    # +0.125R when real SL hit 8 bars after fill.
    fill_walk_start = alert_ts + pd.Timedelta(hours=1)
    future = df_h1.loc[fill_walk_start:]
    if future.empty:
        return None

    filled = False
    fill_ts: Optional[pd.Timestamp] = None
    fill_bar_idx = -1
    # Both proximal and 50pct entries are modelled as pre-placed pending limits
    # sitting at their respective levels for the OB's lifetime. Fill when price
    # first crosses the entry level (long fills on bar low <= entry; short on
    # bar high >= entry). This handles three cases uniformly:
    #   - alert bar exactly touched the level   -> fills on alert bar
    #   - alert bar approaching but not yet at  -> fills on subsequent bar
    #   - alert bar overshot past the level     -> fills when price pulls back
    # Same logic for both entry zones means proximal vs 50% is a clean A/B.

    exit_ts: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None
    tp1_hit_bar_idx = -1
    tp2_hit_bar_idx = -1
    mfe_price = entry
    mae_price = entry
    sl_collision = False
    bars_walked_post_fill = 0
    bars_to_tp1 = -1
    bars_to_tp2 = -1
    sl_after_tp1 = sl  # snapshot for the "default policy" walk (TP1 -> BE)

    for i, (ts, bar) in enumerate(future.iterrows()):
        bar_hi = float(bar["High"])
        bar_lo = float(bar["Low"])

        is_fill_bar_this_iter = False
        if not filled:
            # Pending limit fill: long fills when bar.low <= entry,
            # short fills when bar.high >= entry. Applies to both proximal
            # and 50pct entries (pre-placed limit semantics).
            if (bias == "LONG" and bar_lo <= entry) or \
               (bias == "SHORT" and bar_hi >= entry):
                filled = True
                fill_ts = ts
                fill_bar_idx = i
                mfe_price = entry
                mae_price = entry
                is_fill_bar_this_iter = True
            else:
                # Pre-fill cap: limit pends at most MAX_HOLD_H1_BARS bars.
                # i is 0-indexed bars-since-alert, so >= cap - 1 means we've
                # already waited the full window without a touch -> give up.
                if i >= MAX_HOLD_H1_BARS - 1:
                    break
                continue

        bars_walked_post_fill = i - fill_bar_idx
        if bars_walked_post_fill > MAX_HOLD_H1_BARS and exit_reason is None:
            exit_ts = ts
            exit_reason = "timeout"
            exit_price = float(bar["Close"])
            break

        if bias == "LONG":
            mfe_price = max(mfe_price, bar_hi)
            mae_price = min(mae_price, bar_lo)
            sl_hit_in_bar = bar_lo <= sl_after_tp1
            tp1_hit_in_bar = bar_hi >= tp1
            tp2_hit_in_bar = (tp2 is not None) and (bar_hi >= tp2)
        else:
            mfe_price = min(mfe_price, bar_lo)
            mae_price = max(mae_price, bar_hi)
            sl_hit_in_bar = bar_hi >= sl_after_tp1
            tp1_hit_in_bar = bar_lo <= tp1
            tp2_hit_in_bar = (tp2 is not None) and (bar_lo <= tp2)

        # Fill-bar rule (2026-05-25):
        # On the bar where the limit just filled, we cannot infer intra-bar
        # sequence of fill -> TP vs fill -> SL. SL-side: if the bar pierced
        # SL, price had to travel through entry first (limit fills, then SL),
        # so SL is the honest outcome. TP-side: bar high reaching TP could
        # mean (a) price ticked up to TP before pulling down to fill, OR (b)
        # filled then rallied to TP. Can't tell. Conservative call: do NOT
        # credit TP on the fill bar. Walk forward.
        if is_fill_bar_this_iter:
            tp1_hit_in_bar = False
            tp2_hit_in_bar = False

        # Record first-touch bar indices for diagnostic columns.
        if tp1_hit_in_bar and tp1_hit_bar_idx == -1:
            tp1_hit_bar_idx = bars_walked_post_fill
        if tp2_hit_in_bar and tp2_hit_bar_idx == -1:
            tp2_hit_bar_idx = bars_walked_post_fill

        # Worst-case same-bar resolution: SL wins. Small OBs may trigger this;
        # user decision is to take the loss rather than tag inconclusive, so
        # there's no special exit_reason -- it's a plain SL.
        if sl_hit_in_bar and (tp1_hit_in_bar or tp2_hit_in_bar):
            sl_collision = True
            exit_ts = ts
            exit_reason = "sl"
            exit_price = sl_after_tp1
            break
        if sl_hit_in_bar:
            exit_ts = ts
            exit_reason = "sl"
            exit_price = sl_after_tp1
            break
        if tp2_hit_in_bar:
            exit_ts = ts
            exit_reason = "tp2"
            exit_price = tp2
            break
        if tp1_hit_in_bar and sl_after_tp1 != entry:
            # First TP1 hit — default policy moves SL to breakeven and keeps
            # tracking for TP2. We do NOT exit here for the default-policy walk.
            sl_after_tp1 = entry

    if not filled:
        # 50pct entry that never filled. Emit a row with exit_reason="never_filled"
        # so the report can count "would-have-missed" trades.
        bars_to_exit = bars_walked_post_fill
        return _build_row(
            alert=alert, pair_conf=pair_conf, ob=ob,
            entry_zone=entry_zone, entry=entry, sl=sl, tp1=tp1, tp2=tp2,
            tp1_rr=tp1_rr, tp2_rr=tp2_rr,
            score=score, breakdown=breakdown,
            df_h1=df_h1, alert_ts=alert_ts,
            fill_ts=None, exit_ts=None, exit_reason="never_filled",
            exit_price=None,
            r_realised=0.0, r_if_exit_tp1=0.0, r_if_exit_tp2=0.0,
            mfe_r=0.0, mae_r=0.0, bars_to_exit=0,
            bars_to_tp1=-1, bars_to_tp2=-1,
            sl_collision=False, risk_usd=risk_usd,
        )

    if exit_reason is None:
        # Window exhausted with position open and no SL/TP hit.
        last = future.iloc[-1]
        exit_ts = future.index[-1]
        exit_reason = "window_end"
        exit_price = float(last["Close"])

    # R outcomes — note we compute three:
    #   r_realised: under DEFAULT policy (TP2 unless SL/timeout/collision)
    #   r_if_exit_tp1: hypothetical TP1-only exit
    #   r_if_exit_tp2: hypothetical TP2-only exit (== r_realised when no early hit)
    if bias == "LONG":
        r_realised = (exit_price - entry) / r_distance
        mfe_r = (mfe_price - entry) / r_distance
        mae_r = -(entry - mae_price) / r_distance
    else:
        r_realised = (entry - exit_price) / r_distance
        mfe_r = (entry - mfe_price) / r_distance
        mae_r = -(mae_price - entry) / r_distance

    # r_if_exit_tp1: if TP1 was ever touched, this trade closes at TP1 (=tp1_rr).
    # If never touched, it's whatever r_realised ended at (SL/timeout/window_end).
    if tp1_hit_bar_idx >= 0:
        r_if_exit_tp1 = round(tp1_rr, 3)
    else:
        r_if_exit_tp1 = round(r_realised, 3)

    # r_if_exit_tp2: if TP2 was ever touched, closes at TP2 (=tp2_rr).
    # Else if TP1 was touched then SL/timeout, the default-policy SL was moved
    # to breakeven after TP1, so we either book 0R or the actual r_realised.
    # This branch already matches r_realised in those cases.
    if tp2_hit_bar_idx >= 0:
        r_if_exit_tp2 = round(tp2_rr, 3) if tp2 is not None else round(r_realised, 3)
    else:
        r_if_exit_tp2 = round(r_realised, 3)

    bars_to_exit = max(0, bars_walked_post_fill)

    return _build_row(
        alert=alert, pair_conf=pair_conf, ob=ob,
        entry_zone=entry_zone, entry=entry, sl=sl, tp1=tp1, tp2=tp2,
        tp1_rr=tp1_rr, tp2_rr=tp2_rr,
        score=score, breakdown=breakdown,
        df_h1=df_h1, alert_ts=alert_ts,
        fill_ts=fill_ts, exit_ts=exit_ts, exit_reason=exit_reason,
        exit_price=exit_price,
        r_realised=round(r_realised, 3),
        r_if_exit_tp1=r_if_exit_tp1,
        r_if_exit_tp2=r_if_exit_tp2,
        mfe_r=round(mfe_r, 3), mae_r=round(mae_r, 3),
        bars_to_exit=bars_to_exit,
        bars_to_tp1=tp1_hit_bar_idx,
        bars_to_tp2=tp2_hit_bar_idx,
        sl_collision=sl_collision, risk_usd=risk_usd,
    )


def _build_row(*, alert, pair_conf, ob, entry_zone, entry, sl, tp1, tp2,
               tp1_rr, tp2_rr, score, breakdown, df_h1, alert_ts,
               fill_ts, exit_ts, exit_reason, exit_price,
               r_realised, r_if_exit_tp1, r_if_exit_tp2,
               mfe_r, mae_r, bars_to_exit, bars_to_tp1, bars_to_tp2,
               sl_collision, risk_usd) -> Dict[str, Any]:
    """Assemble the final trade row dict in stable column order."""
    direction = ob.get("direction", "?")
    bos_tag = ob.get("bos_tag", "BOS")
    bos_tier = ob.get("bos_tier", "Major")
    dr = ob.get("dealing_range")
    pd_zone = _pd_zone_from_dr(entry, dr)
    pd_alignment = _pd_alignment("LONG" if direction == "bullish" else "SHORT",
                                 pd_zone)
    pnl_usd = round(r_realised * risk_usd, 2)
    return {
        "pair":          alert["pair"],
        "alert_ts":      alert_ts.isoformat() if hasattr(alert_ts, "isoformat") else str(alert_ts),
        "alert_bar_ts":  (alert.get("alert_bar_ts").isoformat()
                          if hasattr(alert.get("alert_bar_ts"), "isoformat")
                          else str(alert.get("alert_bar_ts")) if alert.get("alert_bar_ts") is not None
                          else None),
        "alert_seq":     int(alert.get("alert_seq", 1)),
        "bos_timestamp": ob.get("bos_timestamp"),
        "fill_ts":       fill_ts.isoformat() if (fill_ts is not None and hasattr(fill_ts, "isoformat")) else None,
        "exit_ts":       exit_ts.isoformat() if (exit_ts is not None and hasattr(exit_ts, "isoformat")) else None,
        "direction":     direction,
        "bias":          "LONG" if direction == "bullish" else "SHORT",
        "model":         "h1_only",
        "event":         _event_label(bos_tag, bos_tier),
        "entry_zone":    entry_zone,
        "entry":         entry,
        "sl_initial":    sl,
        "tp1":           tp1,
        "tp2":           tp2,
        "tp1_rr":        round(tp1_rr, 3),
        "tp2_rr":        round(tp2_rr, 3) if tp2 is not None else None,
        "exit_price":    exit_price,
        "exit_reason":   exit_reason,
        "r_realised":    r_realised,
        "r_if_exit_tp1": r_if_exit_tp1,
        "r_if_exit_tp2": r_if_exit_tp2,
        "pnl_usd":       pnl_usd,
        "mfe_r":         mfe_r,
        "mae_r":         mae_r,
        "bars_to_exit":  bars_to_exit,
        "bars_to_tp1":   bars_to_tp1,
        "bars_to_tp2":   bars_to_tp2,
        "ob_age_h1_bars": _ob_age_h1_bars(ob, df_h1, alert_ts),
        "ob_timestamp":  ob.get("ob_timestamp"),
        "pd_zone":       pd_zone,
        "pd_alignment":  pd_alignment,
        "score":         round(float(score), 2),
        "structure_pts": round(float(breakdown.get("structure", 0.0)), 2),
        "sweep_pts":     round(float(breakdown.get("sweep", 0.0)), 2),
        "fvg_pts":       round(float(breakdown.get("fvg", 0.0)), 2),
        "freshness_pts": round(float(breakdown.get("freshness", 0.0)), 2),
        "killzone_pts":  round(float(breakdown.get("killzone", 0.0)), 2),
        "confluences_present": _confluences_present(breakdown),
        "session":       _session_from_utc_hour(alert_ts.hour),
        "sl_collision":  sl_collision,
        "bos_tag":       bos_tag,
        "bos_tier":      bos_tier,
        "fvg_present":   bool((ob.get("fvg") or {}).get("exists")),
        # fresh / stale / no_fvg — was the FVG already discharged on an earlier
        # approach before this trigger? Feeds the FVG-staleness breakdown.
        "fvg_state":     _fvg_state(ob, df_h1, alert_ts),
        "sweep_present": bool((ob.get("sweep_observed") or {}).get("exists")),
        # Session breakdown — OB formation vs fill, plus killzone alignment.
        # Fill session is the more honest label (when capital was actually
        # at work). OB session captures setup quality (institutional vs not).
        # Alignment buckets: Both / OB only / Fill only / Neither -- used by
        # email and Excel reporting to test the SMC veteran hypothesis that
        # both-in-killzone trades have a higher win rate.
        "ob_session":          _ob_session(ob),
        "fill_session":        _fill_session(fill_ts, alert_ts),
        "ob_in_killzone":      _ob_in_killzone(ob, pair_conf),
        "fill_in_killzone":    _fill_in_killzone(fill_ts, pair_conf),
        "killzone_alignment":  _killzone_alignment(ob, fill_ts, alert_ts, pair_conf),
    }


def simulate_h1_only_dual(
    alert: Dict[str, Any],
    pair_conf: Dict[str, Any],
    df_h1: pd.DataFrame,
    risk_usd: float = DEFAULT_RISK_USD,
) -> List[Dict[str, Any]]:
    """Public entry point: simulate BOTH entry zones for one OB-touch alert.

    Returns a list of 0, 1, or 2 trade row dicts:
      - 0 rows: both entry zones returned levels-invalid (e.g. no TP1 >= 1.5R).
      - 1 row : only one entry zone produced valid levels.
      - 2 rows: standard case, proximal + 50pct rows.
    """
    alert_ts = alert["ts"]
    if not isinstance(alert_ts, pd.Timestamp):
        alert_ts = pd.Timestamp(alert_ts)
    if alert_ts.tzinfo is None:
        alert_ts = alert_ts.tz_localize("UTC")

    score, breakdown = _score_h1_only(alert, pair_conf, df_h1, alert_ts)

    # Proximal entry defines the trade structure — simulate it first.
    # If proximal levels are invalid (no TP1 clears 1.5R), skip both entries.
    prox_row = _simulate_single_entry(
        alert, pair_conf, df_h1, "proximal", score, breakdown, risk_usd,
    )
    if prox_row is None:
        return []

    # 50pct entry reuses proximal TP prices — same opposing liquidity target,
    # only the entry zone differs. This makes the A/B comparison clean.
    mid_row = _simulate_single_entry(
        alert, pair_conf, df_h1, "50pct", score, breakdown, risk_usd,
        forced_tp1=prox_row["tp1"],
        forced_tp2=prox_row.get("tp2"),
    )

    rows: List[Dict[str, Any]] = [prox_row]
    if mid_row is not None:
        rows.append(mid_row)
    return rows
