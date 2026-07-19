"""setup_liq.py — structural guards (truth-ledger gate).

Guard classes for the setup_liq_* columns (same discipline as the sweep2 / EQ
tests). Reuses the PROVEN build_scenario generator from test_liquidity_sweep
(real lb-3+1.5-ATR swings — hand-rolled toy frames were too short/tight for the
3-bar confirmation + ATR filter to keep any pivot):

  build_scenario() yields (verified): swing HIGH i5 = 1.1001, swing HIGH i10 =
  1.1050, swing LOW i30 = 1.0925. We anchor the SL / TP bands on these real
  swings.

  1) READ 1 stop-side — active swing low (LONG) in the SL band -> present +
     SIGNED offset (negative below SL, positive inside risk); band width bites.
  2) READ 2 tp-side magnet — active swing high (LONG) in the TP band -> present
     + signed offset; a 1:1-fallback TP reads absent by construction.
  3) ACTIVE-only — a broken swing in the band is NOT counted (dead liquidity).
  4) BAND width — a swing beyond 0.5 ATR of the anchor is out of range.
  5) READ 3.2 leg-extreme — a leg whose extreme swept an active swing -> True;
     a plain-displacement extreme -> False.
  6) DETERMINISM — same frame twice -> identical reads.
  7) DEGRADED inputs -> all-None / None reads, never a raise.

Run:  python -m pytest tests/test_setup_liq.py -q
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

import setup_liq as sl  # noqa: E402
from tests.test_liquidity_sweep import build_scenario, frame, bar, quiet, T0, BASE  # noqa: E402

ATR = 0.0010          # H1 ATR for a 5dp FX pair (matches build_scenario's ATR)

# Real swings build_scenario yields (asserted below so a generator change is caught).
SWING_LOW = 1.0925    # i30
SWING_HIGH_NEAR = 1.1050  # i10 (the higher, later swing high)


def test_generator_swings_are_where_we_think():
    """Guard: if build_scenario's geometry drifts, every anchored test below is
    meaningless. Pin the three swings this file relies on."""
    import dealing_range
    df, *_ = build_scenario()
    sw = {(s["type"], s["idx"]): round(s["price"], 5)
          for s in dealing_range.detect_swings(df, lookback=3)}
    assert sw.get(("low", 30)) == SWING_LOW
    assert sw.get(("high", 10)) == SWING_HIGH_NEAR


# ── 1) READ 1 stop-side signed offset ────────────────────────────────────────

def test_stop_side_present_offset_negative_below_sl():
    """LONG: the active swing low 1.0925 sits BELOW an SL of 1.0928 -> present,
    offset negative (a sweep must blow through the stop then recover)."""
    df, *_ = build_scenario()
    r = sl.reads_stop_and_tp(df, "bullish", sl=1.0928, tp1=1.1050, atr=ATR,
                             pair_type="forex")
    assert r["setup_liq_stop_present"] is True
    # (1.0925 - 1.0928)/0.0010 = -0.3
    assert abs(r["setup_liq_stop_offset_atr"] - (-0.3)) < 1e-6
    assert r["setup_liq_stop_tier"] in ("bare", "PD", "PW", "EQ")


def test_stop_side_offset_positive_inside_risk():
    """LONG: the swing low 1.0925 sits ABOVE (inside risk of) an SL of 1.0922 ->
    positive offset (survive-the-hunt: sweep+reverse before the stop)."""
    df, *_ = build_scenario()
    r = sl.reads_stop_and_tp(df, "bullish", sl=1.0922, tp1=1.1050, atr=ATR,
                             pair_type="forex")
    assert r["setup_liq_stop_present"] is True
    # (1.0925 - 1.0922)/0.0010 = +0.3
    assert abs(r["setup_liq_stop_offset_atr"] - 0.3) < 1e-6


# ── 2) READ 2 tp-side magnet ─────────────────────────────────────────────────

def _flat(ts, n, mid):
    """n flat bars at `mid` with a tiny 0.2-pip range (wide enough that the
    lb-3 geometry treats a spike as a true pivot; narrow enough it makes no
    swing of its own). Verified to yield an isolated active high below."""
    return [(ts + pd.Timedelta(hours=i), mid, mid + 0.00002, mid - 0.00002, mid)
            for i in range(n)]


def _scenario_unbroken_high_above():
    """A LONG setup with a VERIFIED isolated, UNBROKEN swing high 1.1080 above
    the current price (a real TP magnet). build_scenario's leg breaks every high
    it makes, so a magnet needs a high the leg has NOT traded through. Geometry
    pinned in test_generator_swings_are_where_we_think-style below: swing high
    i7 = 1.1080, price then settles ~1.1050 (never re-breaks 1.1080)."""
    rows = []
    ts = T0
    rows += _flat(ts, 5, 1.1030); ts += pd.Timedelta(hours=5)                     # i0..4 flat
    rows += bar(ts, 1.1030, 1.1050, 1.1029, 1.1049); ts += pd.Timedelta(hours=1)  # i5 up
    rows += bar(ts, 1.1049, 1.1065, 1.1048, 1.1064); ts += pd.Timedelta(hours=1)  # i6 up
    rows += bar(ts, 1.1064, 1.1080, 1.1063, 1.1055); ts += pd.Timedelta(hours=1)  # i7 SWING HIGH 1.1080
    rows += bar(ts, 1.1055, 1.1056, 1.1040, 1.1041); ts += pd.Timedelta(hours=1)  # i8 down
    rows += bar(ts, 1.1041, 1.1042, 1.1030, 1.1031); ts += pd.Timedelta(hours=1)  # i9 down (confirms i7)
    rows += _flat(ts, 5, 1.1050)                                                  # i10..14 settle below 1.1080
    return frame(rows)


def test_tp_side_magnet_present():
    """LONG: an UNBROKEN swing high 1.1080 sits in a TP band of 1.1080 -> magnet
    present, offset ~0."""
    df = _scenario_unbroken_high_above()
    r = sl.reads_stop_and_tp(df, "bullish", sl=1.1030, tp1=1.1080, atr=ATR,
                             pair_type="forex")
    assert r["setup_liq_tp_present"] is True
    assert abs(r["setup_liq_tp_offset_atr"] - 0.0) < 1e-6


def test_tp_fallback_reads_no_magnet():
    """A 1:1-fallback TP1 sits on no pool -> magnet absent by construction, even
    with a swing near the price."""
    df, *_ = build_scenario()
    r = sl.reads_stop_and_tp(df, "bullish", sl=1.0928, tp1=1.1050, atr=ATR,
                             pair_type="forex", tp1_is_fallback=True)
    assert r["setup_liq_tp_present"] is False
    assert r["setup_liq_tp_offset_atr"] is None


# ── 4) band width ────────────────────────────────────────────────────────────

def test_swing_beyond_band_not_counted():
    """The swing high 1.1050 is 20 pips from a TP of 1.1070 (> 0.5 ATR = 5 pips)
    -> out of band, not the magnet."""
    df, *_ = build_scenario()
    r = sl.reads_stop_and_tp(df, "bullish", sl=1.0928, tp1=1.1070, atr=ATR,
                             pair_type="forex")
    assert r["setup_liq_tp_present"] is False


# ── 3) active-only ───────────────────────────────────────────────────────────

def test_broken_swing_in_band_not_counted():
    """A swing low later CLOSED through (broken) holds no liquidity -> even in
    the SL band it is not counted. build_scenario's leg already trades UP away
    from the 1.0925 low; append a bar that closes BELOW it to break it."""
    df, *_ = build_scenario()
    rows = [(df.index[i], df["Open"].iloc[i], df["High"].iloc[i],
             df["Low"].iloc[i], df["Close"].iloc[i]) for i in range(len(df))]
    ts = df.index[-1] + pd.Timedelta(hours=1)
    rows += bar(ts, 1.1070, 1.1071, 1.0910, 1.0912)   # closes below 1.0925 -> broken
    rows += quiet(ts + pd.Timedelta(hours=1), 3, mid=1.0912)
    df2 = frame(rows)
    r = sl.reads_stop_and_tp(df2, "bullish", sl=1.0928, tp1=1.1050, atr=ATR,
                             pair_type="forex")
    assert r["setup_liq_stop_present"] is False


# ── 6) determinism ───────────────────────────────────────────────────────────

def test_determinism():
    df, *_ = build_scenario()
    a = sl.reads_stop_and_tp(df, "bullish", sl=1.0928, tp1=1.1050, atr=ATR,
                             pair_type="forex")
    b = sl.reads_stop_and_tp(df, "bullish", sl=1.0928, tp1=1.1050, atr=ATR,
                             pair_type="forex")
    assert a == b


# ── 7) degraded inputs ───────────────────────────────────────────────────────

def test_degraded_inputs_all_none_never_raise():
    df, *_ = build_scenario()
    assert all(v is None for v in sl.reads_stop_and_tp(
        None, "bullish", 1.0, 1.1, ATR, "forex").values())
    assert all(v is None for v in sl.reads_stop_and_tp(
        df, "sideways", 1.0, 1.1, ATR, "forex").values())
    assert all(v is None for v in sl.reads_stop_and_tp(
        df, "bullish", 1.0, 1.1, 0.0, "forex").values())
    assert all(v is None for v in sl.features_none().values())


# ── 5) leg-extreme-was-a-sweep (Read 3.2) ────────────────────────────────────

def test_legextreme_swept_true_when_extreme_takes_active_swing():
    """build_scenario's bullish leg extreme is the break bar i34 (high 1.1075).
    Insert an active swing HIGH just below it that the extreme wicks past +
    closes back below -> the extreme itself swept it -> True.

    We reuse the generator up to the OB, then hand-place a swing high at 1.1065
    and make the break bar wick to 1.1075 but close at 1.1060 (back below 1.1065
    -> a sweep of that high)."""
    # Build a bespoke bullish leg where the terminal HIGH sweeps an active high.
    rows = []
    ts = T0
    rows += quiet(ts, 3, mid=BASE); ts += pd.Timedelta(hours=3)                 # i0..2
    rows += bar(ts, BASE, 1.1065, BASE - 0.0001, 1.1010); ts += pd.Timedelta(hours=1)  # i3 SWING HIGH 1.1065
    rows += bar(ts, 1.1010, 1.1012, 1.0995, 1.0996); ts += pd.Timedelta(hours=1)       # i4 down
    rows += bar(ts, 1.0996, 1.0998, 1.0980, 1.0981); ts += pd.Timedelta(hours=1)       # i5 down (confirms i3 high)
    rows += bar(ts, 1.0981, 1.0999, 1.0980, 1.0998); ts += pd.Timedelta(hours=1)       # i6 up
    rows += bar(ts, 1.0998, 1.1030, 1.0997, 1.1029); ts += pd.Timedelta(hours=1)       # i7 up
    # i8 = leg extreme: wick to 1.1075 (ABOVE the 1.1065 swing high) then close
    # BACK below (1.1060) -> the extreme swept the resting high.
    rows += bar(ts, 1.1029, 1.1075, 1.1028, 1.1060); ts += pd.Timedelta(hours=1)       # i8 extreme sweep
    rows += quiet(ts, 4, mid=1.1060)
    df = frame(rows)
    swept = sl.read_legextreme_swept(df, leg_extreme=1.1075, extreme_end_idx=8,
                                     direction="bullish", pair_type="forex",
                                     atr=ATR)
    assert swept is True


def test_legextreme_swept_false_plain_displacement():
    """A leg whose extreme took no resting swing (clean push, nothing above it)
    -> False. build_scenario's i34 break high (1.1075) tops the whole frame with
    no active swing high above it to sweep."""
    df, *_ = build_scenario()
    swept = sl.read_legextreme_swept(df, leg_extreme=1.1075, extreme_end_idx=34,
                                     direction="bullish", pair_type="forex",
                                     atr=ATR)
    assert swept is False


def test_legextreme_degraded_returns_none():
    df, *_ = build_scenario()
    assert sl.read_legextreme_swept(None, 1.1, 5, "bullish", "forex", ATR) is None
    assert sl.read_legextreme_swept(df, None, 5, "bullish", "forex", ATR) is None
    assert sl.read_legextreme_swept(df, 1.1, 5, "bullish", "forex", 0.0) is None


# ── features assembly ────────────────────────────────────────────────────────

def test_features_from_reads_assembles_all_six():
    df, *_ = build_scenario()
    reads = sl.reads_stop_and_tp(df, "bullish", sl=1.0928, tp1=1.1050, atr=ATR,
                                 pair_type="forex")
    feats = sl.features_from_reads(reads, legextreme_swept=False)
    assert set(feats.keys()) == set(sl.SETUP_LIQ_FEATURE_COLUMNS)
    assert feats["setup_liq_legextreme_swept"] is False
    assert feats["setup_liq_stop_present"] is True
