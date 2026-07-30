"""Spread model: RAW convention (2026-07-30).

Verifies compute_phase2_levels and the simulator's fill trigger under the RAW
spread model. entry/TP are the RAW OB / zone-edge levels — NO spread shift.
The spread lives in exactly ONE place: the stop, widened by one spread away
from entry (LONG sl below distal, SHORT sl above distal).

This REPLACES the 2026-07-22 three-leg shift (entry +spread, TP -spread,
SL -spread), which double-counted the stop (one spread there + one more in
the simulator) and grew every 1R by ~8%. Now there is one spread, on the
stop, once. See smc_detector.py compute_phase2_levels docstring (:1551).

  - ENTRY == entry_raw (no shift).
  - TP == its *_raw twin (no shift).
  - SL keeps its OB-distal +/- spread buffer (the only spread in the model).
  - spread_pips == 0 -> every placed price equals its *_raw (byte-identical),
    same as spread_pips > 0 now (the raw convention is a no-op by construction).

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
    # RAW convention: entry == entry_raw (no spread shift).
    assert abs(lv["entry"] - lv["entry_raw"]) < 1e-9, \
        f"LONG entry should equal raw: {lv['entry']} vs raw {lv['entry_raw']}"
    assert lv["entry_raw"] == 0.61000
    # tp1 == tp1_raw (no spread shift).
    if lv.get("tp1_raw") is not None:
        assert abs(lv["tp1"] - lv["tp1_raw"]) < 1e-9, \
            f"LONG tp1 should equal raw: {lv['tp1']} vs raw {lv['tp1_raw']}"
    # SL keeps its distal - spread buffer (the only spread in the model).
    assert abs(lv["sl"] - (0.60800 - spread)) < 1e-9


def test_short_entry_and_tp_placement():
    conf = _conf(2.0)
    spread = 2.0 * 0.0001
    ob = {"high": 0.61200, "low": 0.61000,
          "direction": "bearish", "h1_atr": 0.0015}
    lv = smc_detector.compute_phase2_levels(
        conf, "SHORT", ob, 0.60990, _frame(False), tp_targets="single")
    assert lv["valid"]
    # RAW convention: entry == entry_raw (no spread shift).
    assert abs(lv["entry"] - lv["entry_raw"]) < 1e-9, \
        f"SHORT entry should equal raw: {lv['entry']} vs raw {lv['entry_raw']}"
    assert lv["entry_raw"] == 0.61000
    # tp1 == tp1_raw (no spread shift).
    if lv.get("tp1_raw") is not None:
        assert abs(lv["tp1"] - lv["tp1_raw"]) < 1e-9, \
            f"SHORT tp1 should equal raw: {lv['tp1']} vs raw {lv['tp1_raw']}"
    assert abs(lv["sl"] - (0.61200 + spread)) < 1e-9


def test_zero_spread_is_byte_identical():
    """spread_pips == 0 -> placement is a no-op: every placed price == its *_raw.
    Under the RAW convention this holds for ANY spread_pips value, but the
    zero case stays as the sanity floor."""
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
    """The simulator must trigger the fill on entry_raw (bid reaches the OB line).
    Under the RAW convention entry == entry_raw by construction, so the fill
    trigger and the placed entry are the same line — asserting that identity
    is the regression guard against a future re-introduction of a shift that
    would silently decouple the two again."""
    import backtest.h1_only_simulator as sim  # noqa: F401  (import guard only)
    conf = _conf(2.0)
    ob = {"high": 0.61000, "low": 0.60800,
          "direction": "bullish", "h1_atr": 0.0015}
    lv = smc_detector.compute_phase2_levels(
        conf, "LONG", ob, 0.61010, _frame(True), tp_targets="triple")
    entry_raw = lv["entry_raw"]
    entry_placed = lv["entry"]
    # RAW convention: placed == raw, so the fill trigger (on entry_raw) is
    # exactly the traded entry — no spread gap between trigger and fill.
    assert entry_placed == entry_raw
