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


def _confluences_present(breakdown: Dict[str, float]) -> str:
    """Comma-separated list of confluences that scored > 0 on this OB."""
    names = []
    if breakdown.get("structure", 0) > 0:
        names.append("structure")
    if breakdown.get("sweep", 0) > 0:
        names.append("sweep")
    if breakdown.get("fvg", 0) > 0:
        names.append("fvg")
    if breakdown.get("freshness", 0) > 0:
        names.append("freshness")
    if breakdown.get("killzone", 0) > 0:
        names.append("killzone")
    return ",".join(names) if names else "none"


def _event_label(bos_tag: Optional[str], bos_tier: Optional[str]) -> str:
    """One-column event label: 'Major BOS' / 'Minor BOS' / 'Major CHoCH' / 'Minor CHoCH'."""
    tag = bos_tag or "BOS"
    tier = bos_tier or "Major"
    return f"{tier} {tag}"


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
    """Score the OB using live run_scorecard with df_m15=None (H1-only).
    Returns (total, breakdown). Never raises — bad scores log and return 0.
    """
    ob = alert["ob"]
    bias = "LONG" if ob.get("direction") == "bullish" else "SHORT"
    h1_slice = df_h1.loc[:alert_ts]
    fvg_h1 = ob.get("fvg", {"exists": False, "was_detected": False,
                            "mitigation": "none"})
    fvg_data = {"h1": fvg_h1, "m15": {"exists": False, "was_detected": False,
                                       "mitigation": "none"}}
    try:
        score_res = smc_detector.run_scorecard(
            bias, h1_slice, ob, fvg_data, alert["current_price"],
            pair_conf, df_m15=None,
        )
    except Exception as e:
        log_event("h1only_scorecard_error", level="warn",
                  pair=alert.get("pair"),
                  error=f"{type(e).__name__}: {e}")
        return 0.0, {}
    breakdown = dict(score_res.get("breakdown", {}))
    # Override killzone using the alert bar's hour (live wallclock would be
    # wrong in a backtest).
    pair_type = pair_conf.get("pair_type", "forex")
    breakdown["killzone"] = (
        0.5 if smc_detector._killzone_hit(alert_ts.hour, pair_type) else 0.0
    )
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

    try:
        levels = smc_detector.compute_phase2_levels(
            pair_conf, bias, ob, current_price, df_h1, df_m15=None,
            h1_only=True, entry_zone=entry_zone,
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

    # Walk H1 bars from alert_ts forward up to MAX_HOLD_H1_BARS bars.
    # alert_ts is the bar OPENING at P2 fire moment — the first bar on which
    # a live broker could fill the limit order. The just-closed bar that
    # triggered the proximity check is NOT included: at the moment its wick
    # was creating the move, the limit order didn't exist yet.
    future = df_h1.loc[alert_ts:]
    if future.empty:
        return None
    future = future.iloc[: MAX_HOLD_H1_BARS + 1]  # +1 to include alert bar

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
            else:
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

        # Record first-touch bar indices for diagnostic columns.
        if tp1_hit_in_bar and tp1_hit_bar_idx == -1:
            tp1_hit_bar_idx = bars_walked_post_fill
        if tp2_hit_in_bar and tp2_hit_bar_idx == -1:
            tp2_hit_bar_idx = bars_walked_post_fill

        # Pessimistic same-bar resolution: SL first.
        if sl_hit_in_bar and (tp1_hit_in_bar or tp2_hit_in_bar):
            sl_collision = True
            exit_ts = ts
            exit_reason = "sl_collision"
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
        "sweep_present": bool((ob.get("sweep_observed") or {}).get("exists")),
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
