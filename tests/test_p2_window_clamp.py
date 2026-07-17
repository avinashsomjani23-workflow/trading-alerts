"""Regression: the backtest clamps its scorecard + levels input to live P2's
200-bar H1 window (TRUTH_FIXES_SPEC_2 T5 — the class-killer test).

Run:  python tests/test_p2_window_clamp.py
Exit 0 iff:
  1) the shared constant LIVE_P2_H1_BARS is 200 and lives in smc_detector;
  2) _closed_bars_at_alert clamps to that window AND never leaks future bars;
  3) TP selection is DIFFERENT clamped vs unclamped — the clamp is LIVE, not a
     no-op (a nearer opposing swing 300 bars back is invisible in the 200 window,
     so the clamped run picks the farther 50-bar swing instead);
  4) both live and backtest fetch/slice the SAME window (source tripwire).

Why this shape: compute_phase2_levels' TP1 = nearest opposing swing clearing the
RR floor. Put a nearer qualifying swing OUTSIDE the 200-bar window and a farther
one INSIDE it. Unbounded history sees the nearer one -> wrong (future-of-live)
TP1; the 200-bar clamp cannot see it -> live-parity TP1. Any revert of the clamp
flips this back and fails here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

import smc_detector
from backtest import h1_only_simulator

_FAILS = []


def _ok(m): print(f"  OK:   {m}")
def _bad(m):
    # RAISE, don't just collect: CI runs these via `pytest tests/ -q`, which
    # never calls main(). A print-and-append _bad is invisible to pytest -> the
    # guard is green even when the code is broken (Deep Value A, 2026-07-10).
    print(f"  FAIL: {m}")
    _FAILS.append(m)
    raise AssertionError(m)


def _build_frame():
    """400-bar LONG frame with ONE isolated, UNBROKEN opposing swing HIGH:
      - idx 100 (300 bars back, OUTSIDE the 200 window): price 1.0080.
    It is the highest bar in the frame, so nothing later pierces it -> it stays
    UNBROKEN (`is_swing_active` keeps it). The in-window region (last 200 bars) has
    NO qualifying opposing high. Entry = OB top 1.0000.

    So: UNCLAMPED sees the 1.0080 pool -> TP1 is a real swing target. CLAMPED (last
    200 bars only) cannot see idx 100 and finds no pool -> falls back to a mechanical
    1:1. The clamp flips tp1_source swing -> fallback_1to1, proving it is live.

    (Rewritten 2026-07-17 for the unbroken-pool filter: the old fixture used a
    NEARER out-of-window high [1.0040] + a FARTHER in-window high [1.0080], but a
    later higher high necessarily SWEEPS the nearer one, so `is_swing_active` now
    correctly drains 1.0040 and the old "unclamped picks 1.0040" premise is
    geometrically impossible. This fixture keeps the out-of-window pool unbroken.)
    """
    N = 400
    idx = pd.date_range("2020-01-01", periods=N, freq="h", tz="UTC")
    base = 1.0000
    close = np.full(N, base)
    high = np.full(N, base + 0.0002)
    low = np.full(N, base - 0.0002)
    for i, p in [(100, 1.0080)]:  # the ONE unbroken swing high, outside the window
        high[i] = p
        close[i] = p - 0.0001
        low[i] = p - 0.0003
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close},
                      index=idx)
    # alert fires on the bar AFTER the last printed bar (all 400 already closed).
    alert_ts = idx[-1] + pd.Timedelta(hours=1)
    return df, alert_ts


def _tp1(frame):
    pair_conf = {"decimal_places": 5, "spread_pips": 2}
    ob = {"high": 1.0000, "low": 0.9980, "direction": "bullish"}
    res = smc_detector.compute_phase2_levels(
        pair_conf, "LONG", ob, 1.0000, frame,
        entry_zone="proximal", tp1_min_rr=1.5)
    return res.get("valid"), res.get("tp1"), res.get("tp1_source")


# --- 1) constant is 200 and lives in smc_detector ---------------------------

def test_constant_is_200():
    if getattr(smc_detector, "LIVE_P2_H1_BARS", None) == 200:
        _ok("smc_detector.LIVE_P2_H1_BARS == 200 (shared live/backtest constant)")
    else:
        _bad(f"LIVE_P2_H1_BARS is {getattr(smc_detector,'LIVE_P2_H1_BARS',None)}, want 200")


# --- 2) helper clamps to the window and never leaks future ------------------

def test_helper_clamps_and_no_lookahead():
    df, alert_ts = _build_frame()
    clamped = h1_only_simulator._closed_bars_at_alert(df, alert_ts)
    if len(clamped) != 200:
        _bad(f"_closed_bars_at_alert returned {len(clamped)} bars, want 200")
        return
    if clamped.index.max() >= alert_ts:
        _bad("clamped frame leaked a bar at/after alert_ts (lookahead)")
        return
    # tail(200) of a strictly-before slice = the newest 200 closed bars.
    if clamped.index.min() == df.index[200]:
        _ok("helper returns the last 200 closed bars, no future leak")
    else:
        _bad(f"clamped window start wrong: {clamped.index.min()} vs {df.index[200]}")


def test_helper_clamp_holds_under_short_history():
    # Fewer than 200 bars available -> return all of them, tripwire must not fire.
    df, alert_ts = _build_frame()
    short = df.head(50)
    out = h1_only_simulator._closed_bars_at_alert(short, alert_ts)
    if len(out) == 50:
        _ok("helper returns all bars when history < 200 (no crash, assert holds)")
    else:
        _bad(f"short-history helper returned {len(out)} bars, want 50")


# --- 3) the clamp CHANGES TP selection (proves it is live, not a no-op) -----

def test_clamp_changes_tp1():
    df, alert_ts = _build_frame()
    clamped = h1_only_simulator._closed_bars_at_alert(df, alert_ts)
    unclamped = df.loc[df.index < alert_ts]  # full closed history, no tail

    valid_c, tp1_c, src_c = _tp1(clamped)
    valid_u, tp1_u, src_u = _tp1(unclamped)

    # Unclamped sees the unbroken out-of-window pool (1.0080) -> real swing TP.
    # Clamped (last 200 bars) cannot see idx 100 -> no pool -> mechanical 1:1.
    # The clamp flips tp1_source swing -> fallback_1to1, proving it is live.
    if not (valid_c and valid_u):
        _bad(f"expected both valid; clamped={valid_c}, unclamped={valid_u}")
        return
    if (src_u == "swing" and abs(tp1_u - 1.0080) < 1e-9
            and src_c == "fallback_1to1"):
        _ok("clamp is LIVE: unclamped tp1=1.0080 (unbroken out-of-window swing), "
            "clamped falls back to 1:1 (pool invisible in the 200-bar window)")
    else:
        _bad(f"clamp did not change TP1 selection as expected: "
             f"unclamped tp1={tp1_u} src={src_u} (want 1.0080/swing); "
             f"clamped tp1={tp1_c} src={src_c} (want fallback_1to1)")


# --- 4) source tripwires: live + backtest use the SAME window ---------------

def test_live_fetches_shared_constant():
    src = (_ROOT / "Phase2_Alert_Engine.py").read_text(encoding="utf-8")
    if "smc_detector.LIVE_P2_H1_BARS" in src:
        _ok("live Phase 2 fetch uses smc_detector.LIVE_P2_H1_BARS")
    else:
        _bad("live Phase 2 no longer fetches smc_detector.LIVE_P2_H1_BARS "
             "— parity with backtest window broken")


def test_backtest_uses_helper_at_both_sites():
    src = (_ROOT / "backtest" / "h1_only_simulator.py").read_text(encoding="utf-8")
    # helper defined once, called at BOTH the scoring and levels slices.
    if src.count("_closed_bars_at_alert(df_h1, alert_ts)") >= 2:
        _ok("backtest routes BOTH scorecard + levels through _closed_bars_at_alert")
    else:
        _bad("backtest does not use _closed_bars_at_alert at both call sites — "
             "the unbounded-history slice (T5 bug) may have returned")


def main():
    print("== T5: constant ==")
    test_constant_is_200()
    print("\n== T5: helper clamp + lookahead ==")
    test_helper_clamps_and_no_lookahead()
    test_helper_clamp_holds_under_short_history()
    print("\n== T5: clamp changes TP selection (live, not no-op) ==")
    test_clamp_changes_tp1()
    print("\n== T5: source parity tripwires ==")
    test_live_fetches_shared_constant()
    test_backtest_uses_helper_at_both_sites()
    print()
    if _FAILS:
        print(f"FAILED: {len(_FAILS)} problem(s)")
        return 1
    print("PASSED: backtest scorecard + levels input is clamped to live's 200-bar window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
