"""Trade simulation: entry fill, SL/TP walk, MFE/MAE tracking.

Given a would-be alert (from replay_engine), this module:
1. Computes entry / SL / TP via live `smc_detector.compute_phase2_levels`.
2. Walks forward on M15 (or M5 for Phase 3) bars until SL, TP1, TP2, or
   time-stop hits. TP2 only after TP1.
3. Records MFE (max favourable excursion) and MAE (max adverse excursion)
   in R units.

Hard rule: same-bar SL + TP collision -> SL hits first (pessimistic).
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import timedelta
from typing import Dict, Any, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import smc_detector  # live module, read-only

from backtest.run_logger import log_event


MAX_HOLD_HOURS = 72            # time-stop
DEFAULT_RISK_USD = 250.0       # 1R = $250
MIN_RR_AFTER_SLIPPAGE = 1.2    # mirrors live phase3 config



def compute_levels(pair_conf, bias, ob, current_price, df_h1, df_trigger):
    """Wrap live level computation. Returns dict with entry/sl/tp1/tp2."""
    try:
        levels = smc_detector.compute_phase2_levels(
            pair_conf, bias, ob, current_price, df_h1, df_trigger
        )
        return levels if isinstance(levels, dict) else None
    except Exception as e:
        log_event("levels_error", level="error",
                  pair=pair_conf.get("name"),
                  error=f"{type(e).__name__}: {e}")
        return None


def score_ob_confluences(ob, pair_conf, current_price, h1_atr, walls):
    """Mirror of live Phase 2 confluence scoring. See KNOWN_LIMITATIONS.md.

    v1 implementation: baseline confluences with pragmatic weights.
    Live scoring lives inline in Phase2_Alert_Engine.py main block; this
    will drift if live changes. Logs every input so divergence is auditable.
    """
    score = 0.0
    breakdown = {}

    # FVG present
    fvg = ob.get("fvg", {}) or {}
    if fvg.get("exists"):
        mit = fvg.get("mitigation", "pristine")
        v = 2.0 if mit == "pristine" else 1.0
        score += v
        breakdown["fvg"] = v

    # Liquidity sweep
    if ob.get("sweep_timestamp"):
        score += 1.5
        breakdown["sweep"] = 1.5

    # Event quality
    bos_tag = ob.get("bos_tag", "BOS")
    bos_tier = ob.get("bos_tier", "Major")
    if bos_tag == "CHoCH" and bos_tier == "Major":
        score += 2.5
        breakdown["event"] = 2.5
    elif bos_tag == "CHoCH" and bos_tier == "Minor":
        score += 1.0
        breakdown["event"] = 1.0
    else:  # BOS
        score += 1.5
        breakdown["event"] = 1.5

    # PD alignment (premium/discount)
    try:
        pd_info = (
            __import__("dealing_range").compute_pd_position(current_price, walls or {})
            if walls else {}
        )
        if pd_info and pd_info.get("valid"):
            zone = pd_info.get("zone")
            direction = ob.get("direction")
            if (direction == "bullish" and zone == "discount") or \
               (direction == "bearish" and zone == "premium"):
                score += 1.5
                breakdown["pd"] = 1.5
            else:
                breakdown["pd"] = 0.0
    except Exception:
        pass

    # OB freshness — penalty if old
    breakdown["freshness"] = 1.0
    score += 1.0

    # Cap at 10
    score = min(score, 10.0)
    return round(score, 2), breakdown


def simulate_trade_h1only(
    alert: Dict[str, Any],
    pair_conf: Dict[str, Any],
    df_h1: pd.DataFrame,
    risk_usd: float = DEFAULT_RISK_USD,
) -> Optional[Dict[str, Any]]:
    """H1-only fallback simulation — used when M15 data is unavailable
    (yfinance 60d intraday limit means weeks > 60d ago have no M15).

    Uses H1 bars for both level computation and trade walk. Less granular
    than M15 — one H1 bar = up to 4 M15 bars — so same-bar SL+TP
    collisions are more frequent. Pessimistic SL-first rule mitigates this.
    Report labels every such trade 'h1_only' so results are read in context.
    """
    result = simulate_trade(alert, pair_conf, df_h1, df_h1, risk_usd=risk_usd)
    if result:
        result["model"] = "h1_only"
    return result


def simulate_trade(
    alert: Dict[str, Any],
    pair_conf: Dict[str, Any],
    df_h1: pd.DataFrame,
    df_trigger: pd.DataFrame,
    risk_usd: float = DEFAULT_RISK_USD,
) -> Optional[Dict[str, Any]]:
    """Walk df_trigger forward from alert.ts and return trade outcome dict.

    Returns None if levels invalid or no fill within hold window.
    """
    ob = alert["ob"]
    pair = alert["pair"]
    alert_ts = pd.Timestamp(alert["ts"]).tz_convert("UTC") if alert["ts"].tzinfo else pd.Timestamp(alert["ts"], tz="UTC")
    current_price = alert["current_price"]

    bias = "LONG" if ob.get("direction") == "bullish" else "SHORT"
    levels = compute_levels(pair_conf, bias, ob, current_price, df_h1, df_trigger)
    if not levels or not levels.get("valid", True):
        return None

    entry = float(levels.get("entry", 0))
    sl    = float(levels.get("sl", 0))
    tp1   = float(levels.get("tp1", 0))
    tp2   = float(levels.get("tp2", 0)) if levels.get("tp2") else None
    if entry <= 0 or sl <= 0 or tp1 <= 0:
        return None

    r_distance = abs(entry - sl)
    if r_distance <= 0:
        return None

    # Walk M15/M5 bars from alert_ts onward.
    future = df_trigger.loc[alert_ts:]
    if future.empty:
        return None

    deadline = alert_ts + timedelta(hours=MAX_HOLD_HOURS)
    filled = False
    fill_ts = None
    exit_ts = None
    exit_reason = None
    exit_price = None
    tp1_hit = False
    mfe_price = entry
    mae_price = entry
    sl_collisions = 0

    for ts, bar in future.iterrows():
        if ts > deadline and filled:
            exit_ts = ts
            exit_reason = "time_stop"
            exit_price = float(bar["Close"])
            break

        bar_hi = float(bar["High"])
        bar_lo = float(bar["Low"])

        if not filled:
            # Limit fill: long fills when low <= entry; short fills when high >= entry.
            if bias == "LONG" and bar_lo <= entry:
                filled = True
                fill_ts = ts
            elif bias == "SHORT" and bar_hi >= entry:
                filled = True
                fill_ts = ts
            if not filled:
                continue

        # Track MFE / MAE post-fill.
        if bias == "LONG":
            mfe_price = max(mfe_price, bar_hi)
            mae_price = min(mae_price, bar_lo)
            sl_hit_in_bar = bar_lo <= sl
            tp1_hit_in_bar = bar_hi >= tp1
            tp2_hit_in_bar = (tp2 is not None) and (bar_hi >= tp2)
        else:
            mfe_price = min(mfe_price, bar_lo)
            mae_price = max(mae_price, bar_hi)
            sl_hit_in_bar = bar_hi >= sl
            tp1_hit_in_bar = bar_lo <= tp1
            tp2_hit_in_bar = (tp2 is not None) and (bar_lo <= tp2)

        # Pessimistic same-bar resolution: SL first.
        if sl_hit_in_bar and (tp1_hit_in_bar or tp2_hit_in_bar):
            sl_collisions += 1
            exit_ts = ts
            exit_reason = "sl_collision"
            exit_price = sl
            break
        if sl_hit_in_bar:
            exit_ts = ts
            exit_reason = "sl"
            exit_price = sl
            break
        if tp2_hit_in_bar:
            exit_ts = ts
            exit_reason = "tp2"
            exit_price = tp2
            tp1_hit = True
            break
        if tp1_hit_in_bar and not tp1_hit:
            tp1_hit = True
            # Move SL to breakeven after TP1.
            sl = entry

    if not filled:
        return None

    if exit_reason is None:
        # Window exhausted with position open.
        last = future.iloc[-1]
        exit_ts = future.index[-1]
        exit_reason = "window_end"
        exit_price = float(last["Close"])

    # R-realised
    if bias == "LONG":
        r_realised = (exit_price - entry) / r_distance
        mfe_r = (mfe_price - entry) / r_distance
        mae_r = (entry - mae_price) / r_distance * -1
    else:
        r_realised = (entry - exit_price) / r_distance
        mfe_r = (entry - mfe_price) / r_distance
        mae_r = (mae_price - entry) / r_distance * -1

    pnl_usd = round(r_realised * risk_usd, 2)
    hold_minutes = int((exit_ts - fill_ts).total_seconds() / 60) if fill_ts else 0

    return _build_result(
        pair=pair,
        bias=bias,
        ob=ob,
        levels=levels,
        alert_ts=alert_ts,
        fill_ts=fill_ts,
        exit_ts=exit_ts,
        exit_reason=exit_reason,
        exit_price=exit_price,
        tp1_hit=tp1_hit,
        r_realised=r_realised,
        pnl_usd=pnl_usd,
        mfe_r=mfe_r,
        mae_r=mae_r,
        hold_minutes=hold_minutes,
        sl_collision=(exit_reason == "sl_collision"),
        model="phase2",
    )


def simulate_phase3_trade(
    trigger: Dict[str, Any],
    pair_conf: Dict[str, Any],
    df_h1: pd.DataFrame,
    df_m15: pd.DataFrame,
    df_m5: pd.DataFrame,
    risk_usd: float = DEFAULT_RISK_USD,
) -> Optional[Dict[str, Any]]:
    """Simulate a Phase 3 trade from a phase3_trigger event.

    Entry: market at M5 CHoCH level (mirrors live: entry = choch_level).
    SL/TP: recomputed via smc_detector.compute_phase2_levels using H1+M15.
    Walk: M5 bars forward from trigger timestamp.
    """
    ob = trigger["ob"]
    bias = trigger["bias"]
    choch_level = trigger["choch_level"]
    pair = trigger["pair"]
    trigger_ts = pd.Timestamp(trigger["ts"])
    if trigger_ts.tzinfo is None:
        trigger_ts = trigger_ts.tz_localize("UTC")

    dp = pair_conf.get("decimal_places", 5)

    # Slice H1 + M15 up to trigger time for level computation.
    h1_slice = df_h1.loc[:trigger_ts] if df_h1 is not None else None
    m15_slice = df_m15.loc[:trigger_ts] if df_m15 is not None else None

    if h1_slice is None or h1_slice.empty or m15_slice is None or m15_slice.empty:
        return None

    try:
        levels = smc_detector.compute_phase2_levels(
            pair_conf, bias, ob, choch_level, h1_slice, m15_slice
        )
    except Exception as e:
        log_event("p3_levels_error", level="error", pair=pair,
                  error=f"{type(e).__name__}: {e}")
        return None

    if not levels or not levels.get("valid", True):
        return None

    entry = choch_level  # market entry at CHoCH level
    sl    = float(levels.get("sl", 0))
    tp1   = float(levels.get("tp1", 0))
    tp2   = float(levels.get("tp2")) if levels.get("tp2") else None

    if sl <= 0 or tp1 <= 0:
        return None

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    # RR gate: mirrors live min_rr_after_slippage check.
    rr = abs(tp1 - entry) / risk
    if rr < MIN_RR_AFTER_SLIPPAGE:
        return None

    # Walk M5 bars forward.
    if df_m5 is None or df_m5.empty:
        return None
    future = df_m5.loc[trigger_ts:]
    if future.empty:
        return None

    deadline = trigger_ts + timedelta(hours=MAX_HOLD_HOURS)
    fill_ts = trigger_ts   # market order — filled immediately at trigger bar
    exit_ts = None
    exit_reason = None
    exit_price = None
    tp1_hit = False
    mfe_price = entry
    mae_price = entry
    sl_collision = False

    for ts, bar in future.iterrows():
        if ts > deadline:
            exit_ts = ts
            exit_reason = "time_stop"
            exit_price = float(bar["Close"])
            break

        bar_hi = float(bar["High"])
        bar_lo = float(bar["Low"])

        if bias == "LONG":
            mfe_price = max(mfe_price, bar_hi)
            mae_price = min(mae_price, bar_lo)
            sl_hit = bar_lo <= sl
            tp1_hit_bar = bar_hi >= tp1
            tp2_hit_bar = (tp2 is not None) and (bar_hi >= tp2)
        else:
            mfe_price = min(mfe_price, bar_lo)
            mae_price = max(mae_price, bar_hi)
            sl_hit = bar_hi >= sl
            tp1_hit_bar = bar_lo <= tp1
            tp2_hit_bar = (tp2 is not None) and (bar_lo <= tp2)

        if sl_hit and (tp1_hit_bar or tp2_hit_bar):
            sl_collision = True
            exit_ts, exit_reason, exit_price = ts, "sl_collision", sl
            break
        if sl_hit:
            exit_ts, exit_reason, exit_price = ts, "sl", sl
            break
        if tp2_hit_bar:
            exit_ts, exit_reason, exit_price = ts, "tp2", tp2
            tp1_hit = True
            break
        if tp1_hit_bar and not tp1_hit:
            tp1_hit = True
            sl = entry  # move to breakeven

    if exit_reason is None:
        last = future.iloc[-1]
        exit_ts = future.index[-1]
        exit_reason = "window_end"
        exit_price = float(last["Close"])

    if bias == "LONG":
        r_realised = (exit_price - entry) / risk
        mfe_r = (mfe_price - entry) / risk
        mae_r = (entry - mae_price) / risk * -1
    else:
        r_realised = (entry - exit_price) / risk
        mfe_r = (entry - mfe_price) / risk
        mae_r = (mae_price - entry) / risk * -1

    pnl_usd = round(r_realised * risk_usd, 2)
    hold_minutes = int((exit_ts - fill_ts).total_seconds() / 60) if fill_ts else 0

    return _build_result(
        pair=pair,
        bias=bias,
        ob=ob,
        levels={"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2},
        alert_ts=trigger_ts,
        fill_ts=fill_ts,
        exit_ts=exit_ts,
        exit_reason=exit_reason,
        exit_price=exit_price,
        tp1_hit=tp1_hit,
        r_realised=r_realised,
        pnl_usd=pnl_usd,
        mfe_r=mfe_r,
        mae_r=mae_r,
        hold_minutes=hold_minutes,
        sl_collision=sl_collision,
        model="phase3",
        extra={"choch_level": choch_level, "rr_at_entry": round(rr, 2)},
    )


def _build_result(pair, bias, ob, levels, alert_ts, fill_ts, exit_ts,
                  exit_reason, exit_price, tp1_hit, r_realised, pnl_usd,
                  mfe_r, mae_r, hold_minutes, sl_collision, model, extra=None):
    _entry = float(levels.get("entry", 0))
    _sl    = float(levels.get("sl", 0))
    _tp1   = float(levels.get("tp1", 0))
    _tp2   = levels.get("tp2")
    result = {
        "pair": pair,
        "alert_ts": alert_ts.isoformat() if hasattr(alert_ts, "isoformat") else str(alert_ts),
        "fill_ts": fill_ts.isoformat() if fill_ts else None,
        "exit_ts": exit_ts.isoformat() if exit_ts else None,
        "direction": ob.get("direction"),
        "bias": bias,
        "model": model,
        "entry": _entry,
        "sl_initial": _sl,
        "tp1": _tp1,
        "tp2": _tp2,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "tp1_hit": tp1_hit,
        "r_realised": round(r_realised, 3),
        "pnl_usd": pnl_usd,
        "mfe_r": round(mfe_r, 3),
        "mae_r": round(mae_r, 3),
        "hold_minutes": hold_minutes,
        "sl_collision": sl_collision,
        "ob_timestamp": ob.get("ob_timestamp"),
        "bos_tag": ob.get("bos_tag"),
        "bos_tier": ob.get("bos_tier"),
        "fvg_present": bool((ob.get("fvg") or {}).get("exists")),
        "sweep_present": bool(ob.get("sweep_timestamp")),
    }
    if extra:
        result.update(extra)
    return result
