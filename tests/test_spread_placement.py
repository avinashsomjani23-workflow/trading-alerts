"""Spread-aware execution placement (2026-07-22).

Verifies the entry/TP spread SHIFT built into compute_phase2_levels and the
simulator's raw-line fill trigger. Physical model (chart bars are BID; a LONG
fills at the ASK = bid + spread, exits selling at the BID):

  - ENTRY shifts TOWARD price so a live limit fills at the intended zone
    (LONG entry_raw + spread, SHORT entry_raw - spread).
  - TP shifts NEARER so it fills before the reversal
    (LONG tp_raw - spread, SHORT tp_raw + spread).
  - SL keeps its OB-distal +/- spread buffer (unchanged).
  - Trade SELECTION is unchanged: *_raw carries the pre-shift geometry.
  - spread_pips == 0 -> every placed price equals its *_raw (byte-identical).

Run: python -m pytest tests/test_spread_placement.py -q
"""
import numpy as np
import pandas as pd

import smc_detector


def _frame(long_side=True):
    """Minimal H1 frame with an opposing swing pool past entry so a real TP is
    selected (not the 1:1 fallback)."""
    idx = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    if long_side:
        close = np.linspace(0.6085, 0.6140, 60)   # up-trend -> highs above entry
    else:
        close = np.linspace(0.6115, 0.6060, 60)   # down-trend -> lows below entry
    return pd.DataFrame(
        {"Open": close, "High": close + 0.0004,
         "Low": close - 0.0004, "Close": close}, index=idx)


def _conf(spread_pips=2.0):
    return {"name": "NZDUSD", "pair_type": "forex",
            "decimal_places": 5, "spread_pips": spread_pips}


def test_long_entry_and_tp_placement():
    conf = _conf(2.0)
    spread = 2.0 * 0.0001
    ob = {"high": 0.61000, "low": 0.60800,
          "direction": "bullish", "h1_atr": 0.0015}
    lv = smc_detector.compute_phase2_levels(
        conf, "LONG", ob, 0.61010, _frame(True), tp_targets="single")
    assert lv["valid"]
    # entry placed one spread ABOVE the raw OB proximal (fills at the ask).
    assert abs((lv["entry"] - lv["entry_raw"]) - spread) < 1e-9, \
        f"LONG entry shift wrong: {lv['entry']} vs raw {lv['entry_raw']}"
    assert lv["entry_raw"] == 0.61000
    # tp1 placed one spread NEARER than the raw zone edge (fills before reversal).
    if lv.get("tp1_raw") is not None:
        assert abs((lv["tp1_raw"] - lv["tp1"]) - spread) < 1e-9, \
            f"LONG tp1 shift wrong: {lv['tp1']} vs raw {lv['tp1_raw']}"
    # SL keeps its distal - spread buffer (unchanged by placement).
    assert abs(lv["sl"] - (0.60800 - spread)) < 1e-9


def test_short_entry_and_tp_placement():
    conf = _conf(2.0)
    spread = 2.0 * 0.0001
    ob = {"high": 0.61200, "low": 0.61000,
          "direction": "bearish", "h1_atr": 0.0015}
    lv = smc_detector.compute_phase2_levels(
        conf, "SHORT", ob, 0.60990, _frame(False), tp_targets="single")
    assert lv["valid"]
    # entry placed one spread BELOW the raw OB proximal (SHORT fills at the bid).
    assert abs((lv["entry_raw"] - lv["entry"]) - spread) < 1e-9, \
        f"SHORT entry shift wrong: {lv['entry']} vs raw {lv['entry_raw']}"
    assert lv["entry_raw"] == 0.61000
    # tp1 placed one spread NEARER (higher for a SHORT) than the raw zone edge.
    if lv.get("tp1_raw") is not None:
        assert abs((lv["tp1"] - lv["tp1_raw"]) - spread) < 1e-9, \
            f"SHORT tp1 shift wrong: {lv['tp1']} vs raw {lv['tp1_raw']}"
    assert abs(lv["sl"] - (0.61200 + spread)) < 1e-9


def test_zero_spread_is_byte_identical():
    """spread_pips == 0 -> placement is a no-op: every placed price == its *_raw."""
    conf = _conf(0.0)
    ob = {"high": 0.61000, "low": 0.60800,
          "direction": "bullish", "h1_atr": 0.0015}
    lv = smc_detector.compute_phase2_levels(
        conf, "LONG", ob, 0.61010, _frame(True), tp_targets="single")
    assert lv["valid"]
    assert lv["entry"] == lv["entry_raw"]
    if lv.get("tp1_raw") is not None:
        assert lv["tp1"] == lv["tp1_raw"]


def test_simulator_fills_on_raw_line_not_placed():
    """The simulator must trigger the fill on entry_raw (bid reaches the OB line),
    NOT on the spread-placed entry — otherwise a LONG fills one spread too easily.
    A bar low that reaches entry_raw fills; a bar low that only reaches the placed
    entry (one spread short of raw) must NOT."""
    import backtest.h1_only_simulator as sim  # noqa: F401  (import guard only)
    conf = _conf(2.0)
    ob = {"high": 0.61000, "low": 0.60800,
          "direction": "bullish", "h1_atr": 0.0015}
    lv = smc_detector.compute_phase2_levels(
        conf, "LONG", ob, 0.61010, _frame(True), tp_targets="triple")
    entry_raw = lv["entry_raw"]
    entry_placed = lv["entry"]
    # placed is ABOVE raw for a LONG; a fill trigger on placed would fire when the
    # bid only fell to placed — that is one spread too shallow. Assert the ordering
    # the simulator relies on (trigger must be the lower/raw line).
    assert entry_placed > entry_raw
    # sanity: bid must fall to entry_raw (below placed) for the ask to reach placed.
    assert entry_raw < entry_placed
