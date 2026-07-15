"""EQH/EQL equal-level clusters — structural guards (truth-ledger gate).

Guard classes for the new EQ columns (same B5 discipline as the pool tests):
  1) NO-LOOK-AHEAD — features at time t identical whether or not future bars
     exist in the frame (a future sweep must not leak back).
  2) DETERMINISM — same frame twice -> identical output.
  3) Cluster formation — pair forms, third member joins, beyond-tolerance
     stays out, beyond-gap-cap splits, unconfirmed newest swing excluded.
  4) Lifecycle — wick+close-back = swept, close+hold = broken (delegated to
     pool_builder.pool_status; asserted through THIS layer).
  5) Stop-vs-pool geometry — eq_sl_gap_atr sign and eq_sl_at_risk band in
     BOTH directions (the instant-death hypothesis column).
  6) Thin history / no ATR -> all-None dict, never an exception.

Run:  python -m pytest tests/test_eq_pools.py -q
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

import eq_pools as eq  # noqa: E402

ATR = 0.0010  # H1 ATR for a 5dp FX pair; tol = EQ_TOL_ATR * ATR (0.20 -> 2 pips)


def frame(rows):
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame(
        {"Open": [r[1] for r in rows], "High": [r[2] for r in rows],
         "Low": [r[3] for r in rows], "Close": [r[4] for r in rows]},
        index=idx,
    )


def flat_bars(start_ts, n, mid=1.1000, amp=0.00001):
    """n quiet H1 bars that can never form a swing or pierce anything."""
    rows = []
    ts = pd.Timestamp(start_ts)
    for i in range(n):
        rows.append((ts + pd.Timedelta(hours=i),
                     mid, mid + amp, mid - amp, mid))
    return rows


def spike_high(ts, level, mid=1.1000):
    """One bar whose high prints `level` (a swing-high candidate when its
    neighbours are quiet)."""
    return [(pd.Timestamp(ts), mid, level, mid - 0.00001, mid)]


def spike_low(ts, level, mid=1.1000):
    return [(pd.Timestamp(ts), mid, mid + 0.00001, level, mid)]


def build_eqh_frame(levels_and_gaps, tail_quiet=6, start="2026-06-01 00:00"):
    """Frame with spike highs at the given levels, separated by quiet bars.

    levels_and_gaps: list of (level, quiet_bars_before_spike).
    Ends with `tail_quiet` quiet bars so the last spike is a CONFIRMED swing.
    Returns (df, positions_of_spikes).
    """
    rows, spikes = [], []
    ts = pd.Timestamp(start)
    for level, gap in levels_and_gaps:
        rows += flat_bars(ts, gap)
        ts += pd.Timedelta(hours=gap)
        spikes.append(len(rows))
        rows += spike_high(ts, level)
        ts += pd.Timedelta(hours=1)
    rows += flat_bars(ts, tail_quiet)
    return frame(rows), spikes


def feats(df, entry=1.1000, sl=None, direction="bullish", asof=None):
    asof = asof if asof is not None else df.index[-1] + pd.Timedelta(hours=1)
    return eq.features_at_alert(df, asof, direction=direction,
                                entry=entry, sl=sl, atr=ATR)


# ---------------------------------------------------------------------------
# 3) Cluster formation
# ---------------------------------------------------------------------------

def test_pair_of_equal_highs_forms_cluster():
    df, _ = build_eqh_frame([(1.1050, 6), (1.10505, 6)])
    f = feats(df)
    assert f["eqh_above_size"] == 2
    assert f["eqh_above_dist_atr"] == 5.05  # (1.10505 - 1.1000) / 0.0010
    assert f["eq_intact_above_count"] == 1


def test_third_equal_high_joins_cluster():
    df, _ = build_eqh_frame([(1.1050, 6), (1.10505, 6), (1.10498, 6)])
    f = feats(df)
    assert f["eqh_above_size"] == 3


def test_beyond_tolerance_high_does_not_join():
    # 0.0005 above = 0.5 ATR > tol (0.1 ATR): second spike seeds its own
    # single-member candidate — no cluster at all.
    df, _ = build_eqh_frame([(1.1050, 6), (1.1055, 6)])
    f = feats(df)
    assert f["eqh_above_size"] is None
    assert f["eq_intact_above_count"] == 0


def test_gap_cap_splits_clusters():
    # Two equal highs but EQ_MAX_MEMBER_GAP_BARS+ quiet bars apart: no cluster.
    df, _ = build_eqh_frame([(1.1050, 6),
                             (1.10502, eq.EQ_MAX_MEMBER_GAP_BARS + 5)])
    f = feats(df)
    assert f["eqh_above_size"] is None


def test_unconfirmed_swing_excluded():
    # Second equal high with only 2 bars after it: not yet a confirmed swing
    # (lookback 3), so only 1 member -> no cluster.
    df, _ = build_eqh_frame([(1.1050, 6), (1.10505, 6)], tail_quiet=2)
    f = feats(df)
    assert f["eqh_above_size"] is None


# ---------------------------------------------------------------------------
# 4) Lifecycle through pool_status
# ---------------------------------------------------------------------------

def _with_tail(df, rows):
    add = frame(rows)
    return pd.concat([df, add])


def test_wick_through_close_back_marks_swept():
    df, _ = build_eqh_frame([(1.1050, 6), (1.10505, 6)])
    ts = df.index[-1] + pd.Timedelta(hours=1)
    # Wick 3 pips through the cluster extreme, close back below.
    df2 = _with_tail(df, [(ts, 1.1000, 1.10535, 1.0999, 1.1000)]
                     + flat_bars(ts + pd.Timedelta(hours=1), 2))
    f = feats(df2)
    assert f["eqh_above_size"] is None          # spent — no longer intact
    assert f["eq_intact_above_count"] == 0
    assert f["eq_last_sweep_side"] == "high"
    assert f["eq_last_sweep_age_h1"] == 2       # 2 closed bars since the sweep bar


def test_close_through_and_hold_marks_broken_not_swept():
    df, _ = build_eqh_frame([(1.1050, 6), (1.10505, 6)])
    ts = df.index[-1] + pd.Timedelta(hours=1)
    df2 = _with_tail(df, [
        (ts, 1.1000, 1.1060, 1.0999, 1.1058),                          # close beyond
        (ts + pd.Timedelta(hours=1), 1.1058, 1.1061, 1.1055, 1.1059),  # holds (N=1)
    ])
    f = feats(df2)
    assert f["eqh_above_size"] is None
    assert f["eq_last_sweep_side"] is None      # broken, never swept


# ---------------------------------------------------------------------------
# 1) + 2) No-look-ahead and determinism
# ---------------------------------------------------------------------------

def test_no_look_ahead_future_sweep_does_not_leak():
    df, _ = build_eqh_frame([(1.1050, 6), (1.10505, 6)])
    asof = df.index[-1] + pd.Timedelta(hours=1)
    ts = asof
    df_future = _with_tail(df, [(ts, 1.1000, 1.10535, 1.0999, 1.1000)])
    base = feats(df, asof=asof)
    with_future = feats(df_future, asof=asof)
    assert base == with_future
    assert base["eqh_above_size"] == 2          # still intact as of asof


def test_determinism():
    df, _ = build_eqh_frame([(1.1050, 6), (1.10505, 6)])
    assert feats(df) == feats(df.copy())


# ---------------------------------------------------------------------------
# 5) Stop-vs-pool geometry (both directions)
# ---------------------------------------------------------------------------

def _eql_frame():
    """Equal lows at ~1.0950 under price 1.1000."""
    rows, ts = [], pd.Timestamp("2026-06-01 00:00")
    for level in (1.0950, 1.09505):
        rows += flat_bars(ts, 6)
        ts += pd.Timedelta(hours=6)
        rows += spike_low(ts, level)
        ts += pd.Timedelta(hours=1)
    rows += flat_bars(ts, 6)
    return frame(rows)


def test_sl_at_risk_bullish_stop_in_front_of_eql():
    df = _eql_frame()
    # Stop ABOVE the pool (1.0970 > 1.0950): a run to the pool takes us out.
    f = feats(df, entry=1.1000, sl=1.0970, direction="bullish")
    assert f["eql_below_size"] == 2
    assert f["eq_sl_gap_atr"] == 2.0            # (1.0970 - 1.0950) / 0.0010
    assert f["eq_sl_at_risk"] is False          # gap > 1 ATR band
    f2 = feats(df, entry=1.1000, sl=1.09550, direction="bullish")
    assert f2["eq_sl_gap_atr"] == 0.5
    assert f2["eq_sl_at_risk"] is True          # inside the bait band
    f3 = feats(df, entry=1.1000, sl=1.0940, direction="bullish")
    assert f3["eq_sl_gap_atr"] == -1.0          # tucked BEHIND the pool
    assert f3["eq_sl_at_risk"] is False


def test_sl_at_risk_bearish_mirror():
    df, _ = build_eqh_frame([(1.1050, 6), (1.10505, 6)])
    f = feats(df, entry=1.1000, sl=1.10455, direction="bearish")
    assert f["eq_sl_gap_atr"] == 0.5            # (1.10505 - 1.10455) / 0.0010
    assert f["eq_sl_at_risk"] is True
    f2 = feats(df, entry=1.1000, sl=1.1060, direction="bearish")
    assert f2["eq_sl_gap_atr"] == -0.95
    assert f2["eq_sl_at_risk"] is False


def test_trade_toward_eq():
    df, _ = build_eqh_frame([(1.1050, 6), (1.10505, 6)])
    assert feats(df, direction="bullish")["eq_trade_toward"] is True
    assert feats(df, direction="bearish")["eq_trade_toward"] is False


# ---------------------------------------------------------------------------
# 6) Degraded inputs
# ---------------------------------------------------------------------------

def test_thin_history_and_missing_atr_give_all_none():
    df = frame(flat_bars("2026-06-01 00:00", 3))
    f = eq.features_at_alert(df, df.index[-1], direction="bullish",
                             entry=1.1000, sl=1.0950, atr=ATR)
    assert f == eq.features_none()
    df2, _ = build_eqh_frame([(1.1050, 6), (1.10505, 6)])
    f2 = eq.features_at_alert(df2, df2.index[-1] + pd.Timedelta(hours=1),
                              direction="bullish", entry=1.1000, sl=None,
                              atr=None)
    assert f2 == eq.features_none()


def test_column_contract():
    """Every emitted key is in EQ_FEATURE_COLUMNS and vice versa — the CSV
    writer, the ledger and the None-fallback all key off this one tuple."""
    df, _ = build_eqh_frame([(1.1050, 6), (1.10505, 6)])
    f = feats(df, sl=1.0950)
    assert set(f.keys()) == set(eq.EQ_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Live line formatting (plain English, no crash)
# ---------------------------------------------------------------------------

def test_format_eq_line_and_warning():
    df, _ = build_eqh_frame([(1.1050, 6), (1.10505, 6)])
    ctx = eq.live_eq_context(df, ATR,
                             now_utc=df.index[-1] + pd.Timedelta(hours=2))
    assert ctx is not None
    line = eq.format_eq_line(ctx, ref_price=1.1000, atr=ATR)
    assert line is not None and "stops above: equal highs" in line
    warn = eq.format_eq_sl_warning({"eq_sl_at_risk": True,
                                    "eq_sl_gap_atr": 0.5})
    assert warn is not None and "FRONT" in warn
    assert eq.format_eq_sl_warning({"eq_sl_at_risk": False}) is None
