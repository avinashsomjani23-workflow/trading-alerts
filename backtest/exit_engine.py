"""
MULTI-LEG EXIT ENGINE  — the single source of truth for exit simulation.

Why this exists
---------------
The live system and the current backtest both use a SINGLE-exit policy (one TP +
break-even at +1R). The audit showed losses are mostly *given-back winners*, so the
real lever is the EXIT, not the entry. To test exit ideas we need to close a
position in PIECES ("legs") at different targets — partial TP, runner, trailing.

This module generalises the single-exit walk to N legs. The current live behaviour
is just the special case ``legs=[(1.0, "tp1")], be_trigger_r=1.0`` — so there is ONE
exit implementation, not two. The backtest's `_simulate_single_entry` is NOT changed
by this file (parity preserved); experiments call `walk_multileg` directly over the
already-committed trades (entry/SL/TP1/fill are exit-policy-independent).

Fidelity rules (identical to backtest/h1_only_simulator.py, stay pessimistic)
-----------------------------------------------------------------------------
- Only OHLC is known -> intrabar wick order is unknowable -> never assume the good fill.
- FILL BAR (future.iloc[0]): SL CAN fire (price travelled through entry to reach the
  stop, so the limit filled first), but TP and break-even CANNOT be credited.
- SL + TP in the same bar -> SL wins (cannot prove the TP printed before the stop).
  This applies to EVERY still-open leg in that bar.
- Break-even: once +be_trigger_r R is reached, the SHARED stop moves to be_to_r R
  (0R = entry). After that, a later collision resolves at the moved stop.
- Spread is already baked into the `sl` passed in (the committed `sl_initial`).
  TP levels are not widened — matches the simulator.

Inputs
------
walk_multileg(future, bias, entry, sl, r_distance, tp1, config, ...)
  future      : post-fill H1 bars, future.iloc[0] == the fill bar. DatetimeIndex
                (tz-aware UTC), columns Open/High/Low/Close.
  bias        : "LONG" | "SHORT".
  entry,sl    : floats (sl already spread-widened).
  r_distance  : abs(entry - sl) (1R in price).
  tp1         : the liquidity TP price (used by legs whose target spec is "tp1").
  config      : {
      "legs": [(fraction, target_spec), ...]   # fractions sum to 1.0
                 target_spec = float R-multiple (e.g. 1.0) OR "tp1" (liquidity).
      "be_trigger_r": float | None,            # arm BE at +X R; None = no BE.
      "be_to_r": float,                        # move stop to this R (default 0.0).
    }

Returns a dict: r_realised, exit_reason, exit_price, exit_ts, mfe_r, mae_r,
bars_to_exit, n_legs, legs (per-leg detail).
"""
from typing import Any, Dict, List, Optional, Union

import pandas as pd

TargetSpec = Union[float, str]


def _target_price(spec: TargetSpec, bias: str, entry: float,
                  r_distance: float, tp1: float) -> float:
    """Resolve a leg target spec to an absolute price."""
    if isinstance(spec, str):
        if spec == "tp1":
            return tp1
        raise ValueError(f"unknown target spec {spec!r}")
    # numeric R-multiple
    return entry + spec * r_distance if bias == "LONG" else entry - spec * r_distance


def _r_of_price(price: float, bias: str, entry: float, r_distance: float) -> float:
    """Signed R of an exit price."""
    return ((price - entry) if bias == "LONG" else (entry - price)) / r_distance


def walk_multileg(
    future: pd.DataFrame,
    bias: str,
    entry: float,
    sl: float,
    r_distance: float,
    tp1: float,
    config: Dict[str, Any],
    *,
    weekend_flat: bool = True,
    weekend_hour_utc: int = 18,
    max_hold: int = 48,
) -> Dict[str, Any]:
    long = bias == "LONG"

    # ── Build legs ──────────────────────────────────────────────────────────
    raw_legs = config["legs"]
    frac_sum = sum(f for f, _ in raw_legs)
    if abs(frac_sum - 1.0) > 1e-6:
        raise ValueError(f"leg fractions must sum to 1.0, got {frac_sum}")
    legs: List[Dict[str, Any]] = []
    for frac, spec in raw_legs:
        tprice = _target_price(spec, bias, entry, r_distance, tp1)
        legs.append({
            "frac": float(frac),
            "target_spec": spec,
            "target": float(tprice),
            "target_r": _r_of_price(tprice, bias, entry, r_distance),
            "closed": False,
            "exit_price": None,
            "reason": None,
            "exit_ts": None,
        })
    # Process TP touches nearest-target-first within a bar.
    tp_order = sorted(range(len(legs)), key=lambda k: legs[k]["target_r"])

    be_trigger_r = config.get("be_trigger_r")
    be_to_r = float(config.get("be_to_r", 0.0))
    be_trigger_price = (
        None if be_trigger_r is None
        else (entry + be_trigger_r * r_distance if long
              else entry - be_trigger_r * r_distance)
    )
    be_to_price = entry + be_to_r * r_distance if long else entry - be_to_r * r_distance
    # FP-boundary tolerance: be_trigger_price = entry +/- be_trigger_r * r_distance
    # carries accumulated float error, so a bar that touches the trigger EXACTLY can
    # fail `hi >= be_trigger_price` by ~2e-16 while MFE credits the raw high -- the
    # same split that produced the G10 full_sl_loser_with_1R_mfe row in the live
    # walk (h1_only_simulator). Arm within this tolerance so the two agree.
    be_eps = r_distance * 1e-6

    cur_sl = sl
    be_armed = False
    mfe_price = entry
    mae_price = entry
    bars_to_exit = 0
    last_exit_ts = None
    last_exit_price = None
    last_reason = None

    def _close(leg: Dict[str, Any], price: float, reason: str, ts) -> None:
        leg["closed"] = True
        leg["exit_price"] = float(price)
        leg["reason"] = reason
        leg["exit_ts"] = ts

    def _close_all_open(price: float, reason: str, ts) -> None:
        nonlocal last_exit_ts, last_exit_price, last_reason
        for leg in legs:
            if not leg["closed"]:
                _close(leg, price, reason, ts)
        last_exit_ts, last_exit_price, last_reason = ts, float(price), reason

    n = len(future)
    for i in range(n):
        ts = future.index[i]
        bar = future.iloc[i]
        hi = float(bar["High"]); lo = float(bar["Low"])
        op = float(bar["Open"]); cl = float(bar["Close"])
        is_fill_bar = (i == 0)
        bars_to_exit = i

        # 1. Timeout — same order as the simulator (checked before mfe/exit).
        if i > max_hold and any(not lg["closed"] for lg in legs):
            _close_all_open(cl, "timeout", ts)
            break

        # 2. Weekend-flat — never on the fill bar; close at the bar OPEN.
        if (weekend_flat and not is_fill_bar
                and ts.dayofweek == 4 and ts.hour >= weekend_hour_utc
                and any(not lg["closed"] for lg in legs)):
            _close_all_open(op, "friday_flat", ts)
            break

        # 3. Stop check first, then excursions only on non-SL, non-fill bars.
        # MFE/MAE must not include the SL bar (the wick that touches the stop
        # also touches the opposite extreme on the same bar) NOR the fill bar
        # (a LONG fills on the bar low, so that bar's high printed BEFORE the
        # fill — pre-fill price, not excursion). Same rule as the simulator.
        if long:
            stop_hit = lo <= cur_sl
            if not stop_hit and not is_fill_bar:
                mfe_price = max(mfe_price, hi); mae_price = min(mae_price, lo)
        else:
            stop_hit = hi >= cur_sl
            if not stop_hit and not is_fill_bar:
                mfe_price = min(mfe_price, lo); mae_price = max(mae_price, hi)

        # 4. Stop wins any collision (pessimistic), and CAN fire on the fill bar.
        if stop_hit:
            _close_all_open(cur_sl, "sl", ts)
            break

        # 5. Fill bar: SL handled above; TP/BE suppressed.
        if is_fill_bar:
            continue

        # 6. TP legs (nearest target first).
        for k in tp_order:
            leg = legs[k]
            if leg["closed"]:
                continue
            tp_hit = (hi >= leg["target"]) if long else (lo <= leg["target"])
            if tp_hit:
                _close(leg, leg["target"], "tp", ts)
                last_exit_ts, last_exit_price, last_reason = ts, leg["target"], "tp1"

        # 7. Arm break-even.
        if not be_armed and be_trigger_price is not None:
            reached = ((hi >= be_trigger_price - be_eps) if long
                       else (lo <= be_trigger_price + be_eps))
            if reached:
                be_armed = True
                cur_sl = be_to_price

        # 8. All legs closed -> done.
        if all(lg["closed"] for lg in legs):
            break

    # Window exhausted with legs still open -> close at last bar close.
    if any(not lg["closed"] for lg in legs):
        last_bar = future.iloc[-1]
        _close_all_open(float(last_bar["Close"]), "window_end", future.index[-1])

    r_realised = sum(
        lg["frac"] * _r_of_price(lg["exit_price"], bias, entry, r_distance)
        for lg in legs
    )
    mfe_r = (mfe_price - entry) / r_distance if long else (entry - mfe_price) / r_distance
    mae_r = (-(entry - mae_price) / r_distance if long
             else -(mae_price - entry) / r_distance)

    return {
        "r_realised": round(r_realised, 4),
        "exit_reason": last_reason,
        "exit_price": last_exit_price,
        "exit_ts": last_exit_ts,
        "mfe_r": round(mfe_r, 3),
        "mae_r": round(mae_r, 3),
        "bars_to_exit": bars_to_exit,
        "n_legs": len(legs),
        "legs": legs,
    }
