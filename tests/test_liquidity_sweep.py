"""Sweep v2 (liquidity_sweep.py) — structural guards (truth-ledger gate).

Guard classes for the sweep2_* columns (same B5 discipline as the EQ tests):
  1) NO-LOOK-AHEAD — snapshot at OB build identical whether or not future
     bars exist in the frame (a future dive must not leak back).
  2) DETERMINISM — same frame twice -> identical snapshot.
  3) Target discipline — bare minor swings are NOT sweeps (the old noise
     source); pre-drained pools excluded (intact-at-leg-start required);
     close-and-hold (broken) excluded.
  4) Both directions — PD low raid (bullish) and PD high raid (bearish).
  5) Ranking — PD outranks EQ; RN alignment outranks tier; pools_swept
     counts the raid width.
  6) Round numbers — the FEED-BUFFERED tolerance (8 pips) aligns levels the
     legacy 5-pip tag tolerance would miss.
  7) Tier provability — a frame that cannot prove the full prior day drops
     the PD tier from tiers_checked instead of silently mis-measuring.
  8) Degraded inputs -> snapshot_failed / all-None columns, never a raise.
  9) Live-frame shape — reset-index (Datetime column) frame gives the same
     answer as the DatetimeIndex frame.

Run:  python -m pytest tests/test_liquidity_sweep.py -q
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

import liquidity_sweep as ls  # noqa: E402

ATR = 0.0010          # H1 ATR for a 5dp FX pair
BASE = 1.1000
T0 = pd.Timestamp("2026-06-01 21:00")  # server-day 2026-06-02 (Tue) begins


def frame(rows):
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame(
        {"Open": [r[1] for r in rows], "High": [r[2] for r in rows],
         "Low": [r[3] for r in rows], "Close": [r[4] for r in rows]},
        index=idx,
    )


def quiet(ts, n, mid=BASE):
    """n dead-flat-ish bars that never pierce or pivot below/above anything."""
    return [(ts + pd.Timedelta(hours=i), mid, mid + 0.00001, mid - 0.00001, mid)
            for i in range(n)]


def bar(ts, o, h, l, c):
    return [(ts, o, h, l, c)]


def build_scenario(pdl=1.0932, sweep_low=1.0925, eq_lows=None,
                   pre_window_dip=None, window_closes_below=False,
                   tail=2):
    """One bullish PD-raid timeline. Returns (df, ob_idx, impulse_idx,
    break_idx).

    Layout (hourly from T0):
      i0..23   server-day 06-02: quiet, one dip bar (low=pdl) at i5, one high
               bar (1.1050) at i10, optional EQ spike lows at i14/i18.
      i24..29  server-day 06-03 pre-window: quiet (or a pre-window dip).
      i30      sweep bar: wick to sweep_low, close back at 1.1002.
      i31      up bar.
      i32      OB candle.
      i33      up bar.
      i34      break bar (high 1.1075).
      i35..    quiet tail.
    """
    rows = []
    ts = T0
    rows += quiet(ts, 5); ts += pd.Timedelta(hours=5)
    rows += bar(ts, BASE, BASE + 0.0001, pdl, BASE); ts += pd.Timedelta(hours=1)      # i5 PDL
    rows += quiet(ts, 4); ts += pd.Timedelta(hours=4)
    rows += bar(ts, BASE, 1.1050, BASE - 0.0001, BASE); ts += pd.Timedelta(hours=1)   # i10 PDH
    if eq_lows is None:
        rows += quiet(ts, 13); ts += pd.Timedelta(hours=13)                            # i11..23
    else:
        lo1, lo2 = eq_lows
        rows += quiet(ts, 3); ts += pd.Timedelta(hours=3)                              # i11..13
        rows += bar(ts, BASE, BASE + 0.00001, lo1, BASE); ts += pd.Timedelta(hours=1)  # i14
        rows += quiet(ts, 3); ts += pd.Timedelta(hours=3)                              # i15..17
        rows += bar(ts, BASE, BASE + 0.00001, lo2, BASE); ts += pd.Timedelta(hours=1)  # i18
        rows += quiet(ts, 5); ts += pd.Timedelta(hours=5)                              # i19..23
    rows += quiet(ts, 2); ts += pd.Timedelta(hours=2)                                  # i24..25
    if pre_window_dip is not None:
        rows += bar(ts, BASE, BASE + 0.00001, pre_window_dip, BASE)                    # i26
    else:
        rows += quiet(ts, 1)
    ts += pd.Timedelta(hours=1)
    rows += quiet(ts, 3); ts += pd.Timedelta(hours=3)                                  # i27..29
    if window_closes_below:
        # Close-and-hold through the level: broken, never a sweep.
        rows += bar(ts, BASE, BASE + 0.0001, pdl - 0.0010, pdl - 0.0008)               # i30
        ts += pd.Timedelta(hours=1)
        rows += bar(ts, pdl - 0.0008, pdl - 0.0006, pdl - 0.0012, pdl - 0.0010)        # i31
        ts += pd.Timedelta(hours=1)
        rows += bar(ts, pdl - 0.0010, pdl - 0.0005, pdl - 0.0012, pdl - 0.0009)        # i32 (OB)
        ts += pd.Timedelta(hours=1)
    else:
        rows += bar(ts, BASE, 1.1003, sweep_low, 1.1002); ts += pd.Timedelta(hours=1)  # i30 sweep
        rows += bar(ts, 1.1002, 1.1021, 1.1001, 1.1020); ts += pd.Timedelta(hours=1)   # i31
        rows += bar(ts, 1.1020, 1.1022, 1.1014, 1.1015); ts += pd.Timedelta(hours=1)   # i32 OB
    rows += bar(ts, 1.1015, 1.1046, 1.1014, 1.1045); ts += pd.Timedelta(hours=1)       # i33
    rows += bar(ts, 1.1045, 1.1075, 1.1044, 1.1070); ts += pd.Timedelta(hours=1)       # i34 break
    rows += quiet(ts, tail, mid=1.1070)
    return frame(rows), 32, 30, 34  # ob_idx, impulse_start_idx, break_idx


def observe(df, ob=32, imp=30, brk=34, direction="bullish", atr=ATR,
            pair_name="EURUSD", pair_type="forex", prior=None):
    return ls.observe_pool_sweep(df, ob, imp, direction, atr, pair_type,
                                 pair_name, prior_event_idx=prior,
                                 break_idx=brk)


def _stripped(snap):
    s = dict(snap)
    s.pop("observed_at", None)
    return s


# ── 4) both directions ──────────────────────────────────────────────────────

def test_pd_low_sweep_detected_bullish():
    df, ob, imp, brk = build_scenario()
    snap = observe(df)
    assert snap["exists"] is True
    assert snap["tier"] == "PD"
    assert snap["side"] == "low"
    assert abs(snap["level"] - 1.0932) < 1e-9
    assert abs(snap["pierce_atr"] - 0.7) < 1e-6          # (1.0932-1.0925)/ATR
    # sweep bar i30: body |1.1002-1.1000|=0.0002, lower wick 1.1000-1.0925
    assert abs(snap["rejection_ratio"] - round(0.0075 / 0.0002, 3)) < 1e-6
    # follow-through: max high i31..i34 = 1.1075 -> (1.1075-1.0932)/ATR
    assert abs(snap["follow_atr"] - 14.3) < 1e-6
    assert snap["pools_swept"] == 1
    assert snap["rn_aligned"] is False                    # 1.0932 is 18p off grid
    assert "pd" in snap["tiers_checked"] and "eq" in snap["tiers_checked"]
    assert "pw" not in snap["tiers_checked"]              # no full prior week in frame
    assert snap["sweep_ts"] == df.index[30].isoformat()


def test_pd_high_sweep_detected_bearish():
    # Mirror: prior-day high raided, leg drives down into the break.
    rows = []
    ts = T0
    rows += quiet(ts, 5); ts += pd.Timedelta(hours=5)
    rows += bar(ts, BASE, 1.1068, BASE - 0.0001, BASE); ts += pd.Timedelta(hours=1)  # i5 PDH
    rows += quiet(ts, 18); ts += pd.Timedelta(hours=18)                              # i6..23
    rows += quiet(ts, 6); ts += pd.Timedelta(hours=6)                                # i24..29
    rows += bar(ts, BASE, 1.1074, 1.0997, 1.0998); ts += pd.Timedelta(hours=1)       # i30 sweep
    rows += bar(ts, 1.0998, 1.0999, 1.0979, 1.0980); ts += pd.Timedelta(hours=1)     # i31
    rows += bar(ts, 1.0980, 1.0986, 1.0978, 1.0985); ts += pd.Timedelta(hours=1)     # i32 OB
    rows += bar(ts, 1.0985, 1.0986, 1.0954, 1.0955); ts += pd.Timedelta(hours=1)     # i33
    rows += bar(ts, 1.0955, 1.0956, 1.0925, 1.0930); ts += pd.Timedelta(hours=1)     # i34 break
    rows += quiet(ts, 2, mid=1.0930)
    df = frame(rows)
    snap = observe(df, direction="bearish")
    assert snap["exists"] is True
    assert snap["tier"] == "PD" and snap["side"] == "high"
    assert abs(snap["level"] - 1.1068) < 1e-9
    assert abs(snap["pierce_atr"] - 0.6) < 1e-6           # (1.1074-1.1068)/ATR
    # follow-through: level - min low i31..i34 = 1.1068 - 1.0925
    assert abs(snap["follow_atr"] - 14.3) < 1e-6
    assert snap["rn_aligned"] is False


# ── 3) target discipline ────────────────────────────────────────────────────

def test_bare_swing_sweep_now_tier_sw():
    """A lone minor swing low (no ranked pool) gets wicked. The sweep-v2 rebuild
    REJECTED this (bare swings were noise). 2026-07-20 owner decision: fold the
    normal-swing fuel read back in as tier SW — but only lb-3+1.5-ATR swings
    resting at leg start, ranked LAST, observe-only until the full run judges
    whether the ATR filter rescued the old bare-swing signal.

    Here PDL is far below (1.0900, never touched); the lone spike lows at 1.0968
    / 1.0990 are lb-3+1.5-ATR swings; the sweep bar wicks 1.0960 (below the
    1.0968 spike). No ranked pool is raided, so the winner is tier SW."""
    df, *_ = build_scenario(pdl=1.0900, sweep_low=1.0960,
                            eq_lows=(1.0968, 1.0990))  # 2nd "low" far -> no EQ cluster
    snap = observe(df)
    assert snap["exists"] is True
    assert snap["tier"] == "SW"            # bare swing now qualifies, ranked last
    assert snap["eq_size"] is None         # SW carries no shelf size
    assert "sw" in snap["tiers_checked"]
    assert "pd" in snap["tiers_checked"]   # PD was provable — just not raided


def test_drained_ranked_pool_still_excluded_sw_separate():
    """PDL already wicked BEFORE the leg window — its liquidity is spent, so the
    leg's re-pierce is NOT a fresh PD raid (drained-pool discipline unchanged).
    A bare swing swept in the same leg is a SEPARATE tier-SW catch: the SW
    resting-at-leg-start rule is enforced on the swing itself, not on the PDL."""
    df, *_ = build_scenario(pre_window_dip=1.0930)  # i26 wicks below PDL 1.0932
    snap = observe(df)
    # PD must NOT be the winner (drained pre-leg); if anything wins it is SW.
    assert snap["tier"] != "PD"
    if snap["exists"]:
        assert snap["tier"] == "SW"


def test_broken_pool_excluded():
    """Close-beyond + hold through the level is a BREAK, not a sweep."""
    df, *_ = build_scenario(window_closes_below=True)
    snap = observe(df)
    # A close-and-hold break is not a sweep for a RANKED pool. (A bare swing
    # made by the same collapse could register as SW, but the PD/PW/EQ break
    # discipline is what this test guards — the winner is never a broken PD.)
    assert snap["tier"] != "PD"


def test_sw_tier_requires_swing_resting_at_leg_start():
    """Tier-SW discipline (2026-07-20): a bare swing already SWEPT before the
    leg window is drained liquidity — the leg's re-pierce is not fresh fuel, so
    it must NOT count as SW. Mirror of the ranked-pool drained rule, enforced by
    is_swing_active(before_idx=lo_pos) inside _sw_candidates.

    Build a lone swing low, drain it with a pre-window dip BELOW it, then let the
    leg wick it again. No ranked pool is in play, so a wrongly-kept SW would be
    the only catch — the correct answer is 'no fresh raid'."""
    # PDL parked far below (never touched); a lone swing low at 1.0968 (i14) is
    # drained by a pre-window dip to 1.0955 (i26, below the swing), then the
    # sweep bar re-wicks to 1.0960. The swing was not resting at leg start.
    df, *_ = build_scenario(pdl=1.0900, sweep_low=1.0960,
                            eq_lows=(1.0968, 1.0990), pre_window_dip=1.0955)
    snap = observe(df)
    # The 1.0968 swing was drained pre-leg -> not SW-eligible. The 1.0990 swing
    # sits above the sweep wick (1.0960), never pierced -> not swept. So no
    # fresh raid on any resting swing.
    assert snap["tier"] != "PD"            # PDL never touched
    if snap["exists"]:
        # If anything survives it must be a genuinely resting swing, never the
        # drained 1.0968 one.
        assert abs(snap["level"] - 1.0968) > 1e-9


# ── 5) ranking ──────────────────────────────────────────────────────────────

def test_eq_shelf_sweep():
    """Equal-lows shelf (2 touches) raided while the deeper PDL stays intact."""
    df, *_ = build_scenario(pdl=1.0900, sweep_low=1.0960,
                            eq_lows=(1.0968, 1.0967))
    snap = observe(df)
    assert snap["exists"] is True
    assert snap["tier"] == "EQ"            # EQ still outranks the SW bare swings
    assert snap["eq_size"] == 2
    assert abs(snap["level"] - 1.0967) < 1e-9
    # EQ shelf + the bare-swing SW catches on the same leg (EQ wins the rank).
    assert snap["pools_swept"] == 2


def test_tier_ranking_pd_over_eq_over_sw():
    """Leg raids the PDL, the EQ shelf, AND a bare swing — PD (biggest pool)
    wins. SW ranks below both and only pads pools_swept; it never displaces a
    ranked winner (the whole point of ranking SW last)."""
    df, *_ = build_scenario(pdl=1.0930, sweep_low=1.0925,
                            eq_lows=(1.0968, 1.0967))
    snap = observe(df)
    assert snap["exists"] is True
    assert snap["pools_swept"] == 3       # PD + EQ + SW
    assert snap["tier"] == "PD"
    assert abs(snap["level"] - 1.0930) < 1e-9


def test_rn_alignment_does_not_change_the_winner():
    """Round-number alignment is a LOGGED FACT, never a ranking thumb
    (2026-07-19, owner call — the RN-outranks-tier hunch was not supported by
    a 2016-17 sample). Even when the EQ shelf is RN-aligned and the PD pool is
    not, the bigger pool (PD) still wins; rn_aligned is reported on the winner
    as-is (here the PD winner is NOT on the grid)."""
    df, *_ = build_scenario(pdl=1.0930, sweep_low=1.0925,
                            eq_lows=(1.0947, 1.0946))  # 4p off 1.0950 grid line
    snap = observe(df)
    assert snap["exists"] is True
    assert snap["pools_swept"] == 3        # PD + EQ + SW (SW never changes the winner)
    assert snap["tier"] == "PD"            # bigger pool wins, NOT the RN shelf
    assert snap["rn_aligned"] is False     # 1.0930 is 20p off the grid


# ── 6) round numbers (feed-buffered tolerance) ──────────────────────────────

def test_rn_alignment_buffered():
    """7 pips off the 50-pip grid: MISSED by the legacy 5-pip tag tolerance,
    caught by the feed-buffered 8-pip tolerance (the MT5-vs-TD p95 gap)."""
    df, *_ = build_scenario(pdl=1.0943, sweep_low=1.0936)
    snap = observe(df)
    assert snap["exists"] is True and snap["tier"] == "PD"
    assert snap["rn_aligned"] is True
    assert abs(snap["rn_dist_atr"] - (-0.7)) < 1e-6       # (1.0943-1.0950)/ATR
    import smc_detector
    legacy_tol = smc_detector.ROUND_NUMBER_TOLERANCE["forex"]
    assert abs(1.0943 - 1.0950) > legacy_tol  # proves the buffer changed the answer


# ── 7) tier provability ─────────────────────────────────────────────────────

def test_tier_provability():
    """A frame that starts mid-prior-day cannot prove the full prior day —
    the PD tier must drop out of tiers_checked (never a silently-wrong
    partial-day level), so the raid is honestly not evaluable."""
    df, *_ = build_scenario()
    short = df.iloc[3:]  # frame now starts 3 bars into the prior server day
    snap = observe(short, ob=32 - 3, imp=30 - 3, brk=34 - 3)
    assert "pd" not in (snap["tiers_checked"] or "")
    assert snap["exists"] is False  # the PDL raid can no longer be proven


# ── 1) no-look-ahead + 2) determinism ───────────────────────────────────────

def test_no_look_ahead():
    df, *_ = build_scenario()
    snap1 = observe(df)
    # Append a violent future: a dive through every level in the book.
    future = []
    ts = df.index[-1] + pd.Timedelta(hours=1)
    for i in range(24):
        future.append((ts + pd.Timedelta(hours=i), 1.0800, 1.0801,
                       1.0700, 1.0750))
    df2 = frame([(t, r.Open, r.High, r.Low, r.Close)
                 for t, r in df.iterrows()] + future)
    snap2 = observe(df2)
    assert _stripped(snap1) == _stripped(snap2)


def test_determinism():
    df, *_ = build_scenario()
    assert _stripped(observe(df)) == _stripped(observe(df))


def test_prior_event_floor_respected():
    """A prior structural event INSIDE the lookback clips the window: a raid
    printed before the floor is invisible to this OB."""
    df, *_ = build_scenario()
    # Floor the window at bar 30 (prior event at 30 -> window starts at 31):
    # the sweep bar 30 is outside the window, so no raid is found.
    snap = observe(df, prior=30)
    assert snap["exists"] is False


def test_empty_window_is_a_real_negative():
    """Prior event at/after the OB candle -> the leg has no pre-window at all
    (~5% of real OBs). That is 'ran, found none' (pools_swept=0,
    tiers_checked=''), NEVER a layer failure — the legacy detector's
    empty-observation semantics."""
    df, *_ = build_scenario()
    snap = observe(df, prior=32)  # floor -> lo=33 > ob_idx=32
    assert snap["exists"] is False
    assert snap["pools_swept"] == 0
    assert snap["tiers_checked"] == ""
    out = ls.features_from_snapshot(snap, df, df.index[35])
    assert out["sweep2_present"] is False  # counted as a real negative


# ── 9) live-frame shape parity ──────────────────────────────────────────────

def test_reset_index_frame_parity():
    """The live Phase-1 engine passes a reset-index frame (Datetime column,
    integer index) — the answer must match the DatetimeIndex frame."""
    df, *_ = build_scenario()
    live_shape = df.reset_index().rename(columns={"index": "Datetime"})
    assert _stripped(observe(df)) == _stripped(observe(live_shape))


# ── 8) degraded inputs ──────────────────────────────────────────────────────

def test_degraded_inputs_never_raise():
    df, *_ = build_scenario()
    cases = [
        dict(df=None), dict(atr=None), dict(atr=0.0),
        dict(direction="sideways"),
        dict(ob=None), dict(imp=None),
        dict(ob=5, imp=10),          # impulse after OB
        dict(ob=len(df) + 5),        # out of range
    ]
    for kw in cases:
        args = dict(df=df, ob=32, imp=30, brk=34, direction="bullish", atr=ATR)
        args.update(kw)
        snap = observe(args.pop("df"), ob=args["ob"], imp=args["imp"],
                       brk=args["brk"], direction=args["direction"],
                       atr=args["atr"])
        assert snap["exists"] is False
        assert snap["pools_swept"] is None, f"case {kw} not marked as failed"


# ── row-build column mapping ────────────────────────────────────────────────

def test_features_from_snapshot_and_age():
    df, *_ = build_scenario()
    snap = observe(df)
    alert_ts = df.index[35]  # alert fires on the bar stamped i35
    out = ls.features_from_snapshot(snap, df, alert_ts)
    assert out["sweep2_present"] is True
    assert out["sweep2_tier"] == "PD"
    assert abs(out["sweep2_level"] - 1.0932) < 1e-9
    assert out["sweep2_pools_swept"] == 1
    assert out["sweep2_tiers_checked"] == snap["tiers_checked"]
    # closed bars strictly before i35 = 35; last closed = i34; sweep bar i30
    assert out["sweep2_age_at_alert_h1"] == 4


def test_features_none_on_legacy_or_failed():
    df, *_ = build_scenario()
    for snap in (None, {}, ls.snapshot_failed()):
        out = ls.features_from_snapshot(snap, df, df.index[-1])
        assert out == ls.features_none()
        assert all(v is None for v in out.values())


def test_snapshot_ran_clean_maps_to_present_false():
    """Ran-and-found-nothing must map to present=False (not None) so the
    column separates 'no raid' from 'not evaluable'."""
    df, *_ = build_scenario(pdl=1.0900, sweep_low=1.0995)  # nothing pierced
    snap = observe(df)
    assert snap["exists"] is False and snap["pools_swept"] == 0
    out = ls.features_from_snapshot(snap, df, df.index[-1])
    assert out["sweep2_present"] is False
    assert out["sweep2_pools_swept"] == 0


def test_column_contract():
    """features dict keys == SWEEP2_FEATURE_COLUMNS, exactly (ledger gate)."""
    df, *_ = build_scenario()
    out = ls.features_from_snapshot(observe(df), df, df.index[-1])
    assert tuple(out.keys()) == ls.SWEEP2_FEATURE_COLUMNS
