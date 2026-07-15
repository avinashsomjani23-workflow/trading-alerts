"""Approach quality — structural guards (observe-only, RETRACE_QUALITY_SPEC §7).

All out-of-band; zero live-path asserts (CLAUDE.md guard rule — a guard must
never sit inside live alert generation). Failure mode being guarded: a silent
look-ahead or a silently-flipped sign would poison the next canonical run's
entry-mechanics verdict — wrong rule ships, real money mis-filtered.

  1  Column contract — exactly 3 keys; all-None on empty / fill_ts None / <7 bars
  2  Math, speed — bullish AND bearish sign convention (fall into bull zone = +)
  3  Math, body — known bodies; zero-range skipped; all-zero-range -> None
  4  Math, ER — known path; flat closes (0 denom) -> None, never 0-fake
  5  No look-ahead (bite-proven) — garbage fill bar + all later bars -> identical
  6  Determinism — same inputs twice -> identical dict
  7  ATR scaling (Area B) — double atr -> speed halves, body/ER unchanged
  8  Independent None-ability — atr None -> speed None, body/ER real; dir None too
  9  Live/backtest parity — features_now(truncated) == features_at_fill(full)
 10  Never raises — garbage frame / NaT / strings -> all-None dict

(Item 11, simulator integration, lives in this file too — added after §2 wiring.)

Run:  python -m pytest tests/test_approach_quality.py -q
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import approach_quality as aq  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic H1 builder. Each row is (open, high, low, close). Index = hourly
# UTC timestamps starting 2019-01-01 00:00 (chosen far from any real run window).
# ---------------------------------------------------------------------------

def frame(rows, start="2019-01-01 00:00", tz="UTC"):
    idx = pd.date_range(start, periods=len(rows), freq="h", tz=tz)
    return pd.DataFrame(
        rows, columns=["Open", "High", "Low", "Close"], index=idx
    )


def ohlc(o, h, l, c):
    return (o, h, l, c)


# A clean 8-bar frame: 7 approach bars (idx 0..6) then a fill bar (idx 7).
# Closes descend 110,109,108,107,106,105,104 over the 7 approach bars, so a
# BULLISH trade (zone below) sees a positive "toward zone" speed.
def descending_frame():
    rows = [
        ohlc(110, 110.5, 109.5, 110),   # b1  idx0
        ohlc(110, 110.0, 108.5, 109),   # b2  idx1
        ohlc(109, 109.0, 107.5, 108),   # b3  idx2
        ohlc(108, 108.0, 106.5, 107),   # b4  idx3  (K starts here)
        ohlc(107, 107.0, 105.5, 106),   # b5  idx4
        ohlc(106, 106.0, 104.5, 105),   # b6  idx5
        ohlc(105, 105.0, 103.5, 104),   # b7  idx6  (last closed before fill)
        ohlc(104, 104.5, 103.0, 103),   # FILL idx7
    ]
    return frame(rows)


# ---------------------------------------------------------------------------
# 1. Column contract
# ---------------------------------------------------------------------------

def test_column_contract_keys():
    df = descending_frame()
    out = aq.features_at_fill(df, df.index[7], "bullish", 1.0)
    assert set(out.keys()) == set(aq.APPROACH_FEATURE_COLUMNS)
    assert len(out) == 3


def test_all_none_on_empty_frame():
    out = aq.features_at_fill(pd.DataFrame(), pd.Timestamp("2019-01-01"), "bullish", 1.0)
    assert out == aq.features_none()


def test_all_none_on_fill_ts_none():
    df = descending_frame()
    out = aq.features_at_fill(df, None, "bullish", 1.0)
    assert out == aq.features_none()


def test_all_none_on_fewer_than_7_prior_bars():
    df = descending_frame()
    # Fill at idx 6 -> only 6 prior bars (idx 0..5) -> all None.
    out = aq.features_at_fill(df, df.index[6], "bullish", 1.0)
    assert out == aq.features_none()
    # Fill at idx 7 -> exactly 7 prior bars -> real values.
    out7 = aq.features_at_fill(df, df.index[7], "bullish", 1.0)
    assert out7["approach_speed_atr_at_fill"] is not None


# ---------------------------------------------------------------------------
# 2. Math, speed (sign convention)
# ---------------------------------------------------------------------------

def test_speed_bullish_positive_when_falling_into_zone():
    df = descending_frame()
    # K = idx3..6, closes 107,106,105,104. C[b4]-C[b7] = 107-104 = 3. atr=1.5.
    out = aq.features_at_fill(df, df.index[7], "bullish", 1.5)
    assert out["approach_speed_atr_at_fill"] == round(3 / 1.5, 3)  # 2.0


def test_speed_bearish_is_opposite_sign():
    df = descending_frame()
    # bearish over the SAME falling path: C[b7]-C[b4] = 104-107 = -3.
    out = aq.features_at_fill(df, df.index[7], "bearish", 1.5)
    assert out["approach_speed_atr_at_fill"] == round(-3 / 1.5, 3)  # -2.0


def test_speed_none_on_bad_direction():
    df = descending_frame()
    out = aq.features_at_fill(df, df.index[7], "sideways", 1.5)
    assert out["approach_speed_atr_at_fill"] is None
    # body/ER independent -> still present
    assert out["approach_body_ratio_at_fill"] is not None
    assert out["approach_er_at_fill"] is not None


# ---------------------------------------------------------------------------
# 3. Math, body ratio
# ---------------------------------------------------------------------------

def test_body_ratio_known_value():
    # 7 approach bars; only the K bars (last 4) matter. Build known bodies.
    # Each K bar: |C-O|/(H-L). Make all four = 0.5 exactly.
    base = [ohlc(100, 100.4, 99.6, 100)] * 3  # b1..b3 don't affect body
    k = [
        ohlc(100, 101, 99, 101),   # body |101-100|=1, range 2 -> 0.5
        ohlc(100, 101, 99, 101),   # 0.5
        ohlc(100, 101, 99, 101),   # 0.5
        ohlc(100, 101, 99, 101),   # 0.5
    ]
    df = frame(base + k + [ohlc(101, 101, 100, 100)])  # + fill bar
    out = aq.features_at_fill(df, df.index[7], "bullish", 1.0)
    assert out["approach_body_ratio_at_fill"] == 0.5


def test_body_ratio_skips_zero_range_bar():
    base = [ohlc(100, 100.4, 99.6, 100)] * 3
    k = [
        ohlc(100, 101, 99, 101),   # 0.5
        ohlc(100, 100, 100, 100),  # ZERO range -> skipped
        ohlc(100, 101, 99, 101),   # 0.5
        ohlc(100, 101, 99, 101),   # 0.5
    ]
    df = frame(base + k + [ohlc(101, 101, 100, 100)])
    out = aq.features_at_fill(df, df.index[7], "bullish", 1.0)
    # mean over the 3 non-skipped bars = 0.5
    assert out["approach_body_ratio_at_fill"] == 0.5


def test_body_ratio_none_when_all_zero_range():
    base = [ohlc(100, 100.4, 99.6, 100)] * 3
    k = [ohlc(100, 100, 100, 100)] * 4  # all zero range
    df = frame(base + k + [ohlc(100, 100, 100, 100)])
    out = aq.features_at_fill(df, df.index[7], "bullish", 1.0)
    assert out["approach_body_ratio_at_fill"] is None


# ---------------------------------------------------------------------------
# 4. Math, ER
# ---------------------------------------------------------------------------

def test_er_perfect_one_way_path():
    # 7 closes strictly monotonic by +1 each step: 100..106. Net = 6, sum|Δ| = 6
    # -> ER = 1.0.
    rows = [ohlc(c, c + 0.1, c - 0.1, c) for c in (100, 101, 102, 103, 104, 105, 106)]
    df = frame(rows + [ohlc(106, 106, 105, 105)])
    out = aq.features_at_fill(df, df.index[7], "bullish", 1.0)
    assert out["approach_er_at_fill"] == 1.0


def test_er_choppy_path_below_one():
    # closes: 100,101,100,101,100,101,100 -> net = 0? net |100-100| = 0 -> ER 0.
    # Instead: 100,102,101,103,102,104,103 -> net |103-100|=3, sum|Δ|=2+1+2+1+2+1=9
    rows = [ohlc(c, c + 0.1, c - 0.1, c) for c in (100, 102, 101, 103, 102, 104, 103)]
    df = frame(rows + [ohlc(103, 103, 102, 102)])
    out = aq.features_at_fill(df, df.index[7], "bullish", 1.0)
    assert out["approach_er_at_fill"] == round(3 / 9, 3)


def test_er_none_on_flat_closes():
    # all closes equal -> denominator 0 -> None, never a fake 0.0.
    rows = [ohlc(100, 100.5, 99.5, 100)] * 7
    df = frame(rows + [ohlc(100, 100.5, 99.5, 100)])
    out = aq.features_at_fill(df, df.index[7], "bullish", 1.0)
    assert out["approach_er_at_fill"] is None


# ---------------------------------------------------------------------------
# 5. No look-ahead (bite-proven)
# ---------------------------------------------------------------------------

def test_no_look_ahead_future_bars_ignored():
    df = descending_frame()
    clean = aq.features_at_fill(df, df.index[7], "bullish", 1.5)
    # Mutate the FILL bar and every later bar to garbage.
    poisoned = df.copy()
    poisoned.iloc[7] = [9999, 9999, -9999, 9999]
    # Append 20 more garbage bars after the fill.
    extra_idx = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=20, freq="h", tz="UTC")
    extra = pd.DataFrame(
        [[1e6, 1e6, -1e6, 1e6]] * 20, columns=df.columns, index=extra_idx
    )
    poisoned = pd.concat([poisoned, extra])
    dirty = aq.features_at_fill(poisoned, df.index[7], "bullish", 1.5)
    assert clean == dirty


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    df = descending_frame()
    a = aq.features_at_fill(df, df.index[7], "bullish", 1.234)
    b = aq.features_at_fill(df, df.index[7], "bullish", 1.234)
    assert a == b


# ---------------------------------------------------------------------------
# 7. ATR scaling (Area B one-denominator law)
# ---------------------------------------------------------------------------

def test_atr_scaling_speed_halves_body_er_unchanged():
    df = descending_frame()
    a = aq.features_at_fill(df, df.index[7], "bullish", 1.0)
    b = aq.features_at_fill(df, df.index[7], "bullish", 2.0)
    assert b["approach_speed_atr_at_fill"] == round(a["approach_speed_atr_at_fill"] / 2, 3)
    assert b["approach_body_ratio_at_fill"] == a["approach_body_ratio_at_fill"]
    assert b["approach_er_at_fill"] == a["approach_er_at_fill"]


# ---------------------------------------------------------------------------
# 8. Independent None-ability
# ---------------------------------------------------------------------------

def test_atr_none_speed_none_body_er_real():
    df = descending_frame()
    out = aq.features_at_fill(df, df.index[7], "bullish", None)
    assert out["approach_speed_atr_at_fill"] is None
    assert out["approach_body_ratio_at_fill"] is not None
    assert out["approach_er_at_fill"] is not None


def test_atr_nonpositive_or_nonfinite_speed_none():
    df = descending_frame()
    for bad in (0, -1.0, float("nan"), float("inf")):
        out = aq.features_at_fill(df, df.index[7], "bullish", bad)
        assert out["approach_speed_atr_at_fill"] is None
        assert out["approach_body_ratio_at_fill"] is not None


def test_direction_none_speed_none_only():
    df = descending_frame()
    out = aq.features_at_fill(df, df.index[7], None, 1.5)
    assert out["approach_speed_atr_at_fill"] is None
    assert out["approach_body_ratio_at_fill"] is not None
    assert out["approach_er_at_fill"] is not None


# ---------------------------------------------------------------------------
# 9. Live/backtest parity (one implementation)
# ---------------------------------------------------------------------------

def test_live_backtest_parity():
    df = descending_frame()
    # Backtest: full frame, fill at idx 7.
    bt = aq.features_at_fill(df, df.index[7], "bullish", 1.5)
    # Live: frame truncated to just the closed bars before the fill (idx 0..6).
    live = aq.features_now(df.iloc[:7], "bullish", 1.5)
    assert bt == live


# ---------------------------------------------------------------------------
# 10. Never raises
# ---------------------------------------------------------------------------

def test_never_raises_on_garbage():
    df = descending_frame()
    # NaT fill_ts
    assert aq.features_at_fill(df, pd.NaT, "bullish", 1.5) == aq.features_none()
    # String frame
    assert aq.features_at_fill("not a frame", df.index[7], "bullish", 1.5) == aq.features_none()
    # None frame
    assert aq.features_at_fill(None, df.index[7], "bullish", 1.5) == aq.features_none()
    # Frame with wrong columns
    bad = pd.DataFrame({"foo": range(10)}, index=pd.date_range("2019-01-01", periods=10, freq="h", tz="UTC"))
    assert aq.features_at_fill(bad, bad.index[8], "bullish", 1.5) == aq.features_none()
    # features_now garbage
    assert aq.features_now("nope", "bullish", 1.5) == aq.features_none()
    assert aq.features_now(None, "bullish", 1.5) == aq.features_none()


def test_never_raises_on_nan_bar_in_window():
    df = descending_frame()
    df.iloc[2] = [np.nan, np.nan, np.nan, np.nan]  # broken bar inside window
    out = aq.features_at_fill(df, df.index[7], "bullish", 1.5)
    assert out == aq.features_none()


# ---------------------------------------------------------------------------
# 11. Simulator integration (RETRACE_QUALITY_SPEC §7.11)
#
# Drive the REAL walk + simulate_h1_only_dual on the warm parquet cache for one
# short window. Assert: the 3 approach columns exist on every emitted row;
# filled rows have them populated (at least one real value), never_filled rows
# have all three None. Proves the spread is wired end-to-end, not just the unit.
# Offline + deterministic (cache only). Skips cleanly if the cache is absent.
# ---------------------------------------------------------------------------

import json  # noqa: E402
from datetime import datetime, timezone, timedelta  # noqa: E402


def _load_eurusd_conf():
    cfg = json.load(open(_ROOT / "config.json"))
    p = next(p for p in cfg["pairs"] if p["name"] == "EURUSD")
    p["atr_multiplier"] = {"forex": 3.0, "index": 3.5, "commodity": 3.5}.get(
        p.get("pair_type"), p.get("atr_multiplier"))
    return p


def test_simulator_integration_populates_approach_columns():
    try:
        from backtest import data_loader, replay_engine, h1_only_simulator
    except Exception as e:  # pragma: no cover
        pytest.skip(f"backtest package import failed: {e}")

    pair_conf = _load_eurusd_conf()
    start = datetime(2019, 3, 1, tzinfo=timezone.utc)
    end = datetime(2019, 5, 1, tzinfo=timezone.utc)
    df = data_loader.load_bars(pair_conf["symbol"], "1h", start - timedelta(days=35), end)
    if df is None or df.empty:
        pytest.skip("no cached EURUSD 1h data for 2019 window")

    state = replay_engine.ReplayState()
    rows, seen = [], set()
    for ev in replay_engine.replay_pair(
        pair_conf, df, state=state,
        walk_start_ts=pd.Timestamp(start), walk_end_ts=pd.Timestamp(end)):
        if ev.get("kind") != "alert":
            continue
        key = ((ev.get("ob") or {}).get("ob_timestamp"),
               (ev.get("ob") or {}).get("direction"))
        if key in seen:
            continue
        seen.add(key)
        rows.extend(h1_only_simulator.simulate_h1_only_dual(ev, pair_conf, df, risk_usd=250.0))

    if not rows:
        pytest.skip("no trades produced in the 2019 window (nothing to assert)")

    cols = aq.APPROACH_FEATURE_COLUMNS
    # Every row carries the 3 columns as keys (spread wired into the row dict).
    for r in rows:
        for c in cols:
            assert c in r, f"row missing approach column {c}"

    filled = [r for r in rows if r.get("exit_reason") != "never_filled"]
    never = [r for r in rows if r.get("exit_reason") == "never_filled"]

    # never_filled -> all three None (population = filled rows only).
    for r in never:
        assert all(r[c] is None for c in cols), \
            f"never_filled row has a non-None approach value: {[r[c] for c in cols]}"

    # At least one filled row should carry at least one real approach value
    # (the cache window has >7 bars of history before fills).
    if filled:
        any_real = any(
            any(r[c] is not None for c in cols) for r in filled
        )
        assert any_real, "no filled row carried any populated approach value"
