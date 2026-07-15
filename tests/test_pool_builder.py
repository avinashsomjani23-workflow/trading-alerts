"""PD/PW liquidity pools — structural guards (DAILY_BIAS_V4_SPEC §1.3 gate).

Covers the B5 guard class the spec demands for the new pool columns:
  1) NO-LOOK-AHEAD — the feature dict at time t is identical whether or not
     future bars exist in the frame (synthetic series where the future would
     change the answer if it leaked).
  2) DETERMINISM — same frame twice -> identical output.
  3) Status machine — every transition the spec names: wick sweep, clean
     N=1-confirmed break, failed break (close-through that reverses), broken
     overrides swept, unconfirmed close-beyond reports the prior status.
  4) Boundary correctness — PD levels roll exactly at the 21:00 UTC server
     midnight; PW levels roll at the week boundary; weekly = Mon–Fri roll-up.
  5) Feature derivation — distances/tiers to nearest UNSPENT pool,
     trade_toward_pool, day_state vocabulary, thin-history all-None.

Run:  python -m pytest tests/test_pool_builder.py -q
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

import pool_builder as pb  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic H1 builder. Server day D = UTC [D-1 21:00, D 20:00].
# 2026-06-01 is a Monday.
# ---------------------------------------------------------------------------

def quiet_day(server_day, hi, lo):
    """24 quiet H1 bars for one server day. Bar 0 prints the day's full
    high/low range; the rest sit at mid so nothing else pierces anything."""
    day = pd.Timestamp(server_day)
    start = day - pd.Timedelta(hours=pb.SERVER_UTC_OFFSET_HOURS)
    mid = (hi + lo) / 2.0
    rows = []
    for h in range(24):
        ts = start + pd.Timedelta(hours=h)
        if h == 0:
            rows.append((ts, mid, hi, lo, mid))
        else:
            rows.append((ts, mid, mid + 1e-6, mid - 1e-6, mid))
    return rows


def frame(rows):
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame(
        {"Open": [r[1] for r in rows], "High": [r[2] for r in rows],
         "Low": [r[3] for r in rows], "Close": [r[4] for r in rows]},
        index=idx,
    )


def patch_bar(df, ts, o=None, h=None, l=None, c=None):
    ts = pd.Timestamp(ts)
    for col, v in (("Open", o), ("High", h), ("Low", l), ("Close", c)):
        if v is not None:
            df.loc[ts, col] = v
    return df


def two_weeks(base_hi=1.2000, base_lo=1.1900):
    """Week 1 (Jun 1–5) + week 2 (Jun 8–12), all quiet inside one range."""
    rows = []
    for d in ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04",
              "2026-06-05", "2026-06-08", "2026-06-09", "2026-06-10",
              "2026-06-11", "2026-06-12"):
        rows += quiet_day(d, base_hi, base_lo)
    return frame(rows)


# Server day 2026-06-09 begins at UTC 2026-06-08 21:00.
TUE_BAR = "2026-06-09 05:00"     # a mid-Tuesday (server) bar, UTC-stamped
TUE_NEXT = "2026-06-09 06:00"
ASOF = pd.Timestamp("2026-06-09 12:00")


def snap_at(df, asof=ASOF):
    closed = df.loc[df.index < asof]
    return pb.snapshot(closed, asof_ts=asof)


# ---------------------------------------------------------------------------
# Levels + boundaries
# ---------------------------------------------------------------------------

def test_levels_prev_day_and_prev_week():
    df = two_weeks()
    # Make Monday Jun 8 distinct so PDH/PDL provably come from THAT day.
    patch_bar(df, "2026-06-07 21:00", h=1.2050, l=1.1880)  # Jun 8 server day
    # Make Wednesday Jun 3 print week 1's extreme.
    patch_bar(df, "2026-06-02 21:00", h=1.2100, l=1.1850)  # Jun 3 server day
    lv = pb.levels_at(df, ASOF)
    assert lv["prev_day"] == "2026-06-08"
    assert lv["pdh"] == 1.2050 and lv["pdl"] == 1.1880
    assert lv["prev_week"] == "2026-06-01"
    assert lv["pwh"] == 1.2100 and lv["pwl"] == 1.1850


def test_pd_levels_roll_exactly_at_2100_utc():
    df = two_weeks()
    patch_bar(df, "2026-06-08 21:00", h=1.2070)  # Jun 9's own high
    before_roll = pb.levels_at(df, pd.Timestamp("2026-06-09 20:59"))
    after_roll = pb.levels_at(df, pd.Timestamp("2026-06-09 21:00"))
    assert before_roll["prev_day"] == "2026-06-08"
    assert after_roll["prev_day"] == "2026-06-09"
    assert after_roll["pdh"] == 1.2070


def test_thin_history_is_all_none():
    # One day of bars only: no completed prior day at its own start.
    df = frame(quiet_day("2026-06-01", 1.2, 1.19))
    feats = pb.features_at_alert(df, pd.Timestamp("2026-06-01 02:00"),
                                 direction="bullish", ref_price=1.195, atr=0.001)
    assert feats == pb.features_none()
    assert set(feats.keys()) == set(pb.POOL_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Status machine
# ---------------------------------------------------------------------------

def test_intact_when_untouched():
    snap = snap_at(two_weeks())
    assert all(snap["pools"][k]["status"] == "intact" for k in pb.POOL_KEYS)
    assert snap["day_state"] == "INSIDE"


def test_wick_sweep_of_pdh():
    df = two_weeks()
    patch_bar(df, TUE_BAR, h=1.2005, c=1.1950)  # pierce 1.2000, close back
    snap = snap_at(df)
    assert snap["pools"]["pdh"]["status"] == "swept"
    assert snap["day_state"] == "SWEPT_HIGH"


def test_clean_break_needs_next_close_confirm():
    df = two_weeks()
    patch_bar(df, TUE_BAR, h=1.2020, c=1.2010)   # close beyond PDH
    patch_bar(df, TUE_NEXT, h=1.2030, c=1.2015)  # next close holds
    snap = snap_at(df)
    assert snap["pools"]["pdh"]["status"] == "broken"
    assert snap["day_state"] == "EXPANSION_UP"


def test_failed_break_is_swept():
    df = two_weeks()
    patch_bar(df, TUE_BAR, h=1.2020, c=1.2010)   # close beyond PDH
    patch_bar(df, TUE_NEXT, c=1.1950)            # next close back inside
    snap = snap_at(df)
    assert snap["pools"]["pdh"]["status"] == "swept"


def test_unconfirmed_close_beyond_reports_prior_status():
    df = two_weeks()
    patch_bar(df, TUE_BAR, h=1.2020, c=1.2010)   # close beyond PDH ...
    asof = pd.Timestamp("2026-06-09 06:00")      # ... and no later closed bar
    snap = snap_at(df, asof)
    assert snap["pools"]["pdh"]["status"] == "intact"


def test_break_overrides_earlier_sweep():
    df = two_weeks()
    patch_bar(df, "2026-06-09 03:00", h=1.2005, c=1.1950)  # sweep first
    patch_bar(df, TUE_BAR, h=1.2020, c=1.2010)             # then break
    patch_bar(df, TUE_NEXT, c=1.2015)                      # confirmed
    snap = snap_at(df)
    assert snap["pools"]["pdh"]["status"] == "broken"


def test_pdl_side_and_both_sides_state():
    df = two_weeks()
    patch_bar(df, "2026-06-09 03:00", l=1.1895, c=1.1950)  # sweep PDL
    patch_bar(df, TUE_BAR, h=1.2020, c=1.2010)             # break PDH
    patch_bar(df, TUE_NEXT, c=1.2015)
    snap = snap_at(df)
    assert snap["pools"]["pdl"]["status"] == "swept"
    assert snap["pools"]["pdh"]["status"] == "broken"
    assert snap["day_state"] == "BOTH_SIDES"


def test_weekly_pool_evaluated_over_whole_week():
    df = two_weeks()
    patch_bar(df, "2026-06-02 21:00", h=1.2100)            # PWH = 1.2100 (Jun 3)
    patch_bar(df, "2026-06-08 05:00", h=1.2105, c=1.1950)  # Monday sweeps PWH
    snap = snap_at(df)  # asof Tuesday — Monday's sweep must still count
    assert snap["pools"]["pwh"]["status"] == "swept"
    # ... but the DAY pools were reborn on Tuesday, so pdh is intact again.
    assert snap["pools"]["pdh"]["status"] == "intact"


# ---------------------------------------------------------------------------
# No-look-ahead + determinism
# ---------------------------------------------------------------------------

def test_no_look_ahead():
    """The future must not leak: a huge break AFTER the alert cannot change
    the features stamped AT the alert."""
    base = two_weeks()
    with_future = base.copy()
    patch_bar(with_future, "2026-06-09 15:00", h=1.2500, c=1.2400)
    patch_bar(with_future, "2026-06-09 16:00", c=1.2450)
    f_base = pb.features_at_alert(base, ASOF, "bullish", 1.1950, 0.0010)
    pb._FRAME_CACHE.clear()  # frames share id-space; force a clean build
    f_future = pb.features_at_alert(with_future, ASOF, "bullish", 1.1950, 0.0010)
    assert f_base == f_future
    assert f_base["pdh_status_at_alert"] == "intact"


def test_determinism():
    df = two_weeks()
    patch_bar(df, TUE_BAR, h=1.2005, c=1.1950)
    a = pb.features_at_alert(df, ASOF, "bullish", 1.1950, 0.0010)
    pb._FRAME_CACHE.clear()
    b = pb.features_at_alert(df.copy(), ASOF, "bullish", 1.1950, 0.0010)
    assert a == b


# ---------------------------------------------------------------------------
# Trade features
# ---------------------------------------------------------------------------

def test_distances_tiers_and_toward_pool():
    df = two_weeks()
    patch_bar(df, "2026-06-02 21:00", h=1.2100, l=1.1850)  # week1 extremes
    # ref 1.1950, ATR 0.0010. Intact pools: PDH 1.2000 (50 pips above),
    # PWH 1.2100 (150 above), PDL 1.1900 (50 below), PWL 1.1850 (100 below).
    f = pb.features_at_alert(df, ASOF, "bullish", 1.1950, 0.0010)
    assert f["dist_next_pool_above_atr"] == 5.0   # 50 pips / 10-pip ATR
    assert f["next_pool_above_tier"] == "PD"
    assert f["dist_next_pool_below_atr"] == 5.0
    assert f["next_pool_below_tier"] == "PD"
    # Equidistant above/below -> nearest resolves to 'above' (<=), so a
    # bullish trade points at it.
    assert f["trade_toward_pool"] is True
    f_short = pb.features_at_alert(df, ASOF, "bearish", 1.1950, 0.0010)
    assert f_short["trade_toward_pool"] is False


def test_spent_pool_leaves_distance_slot():
    df = two_weeks()
    patch_bar(df, "2026-06-02 21:00", h=1.2100, l=1.1850)
    patch_bar(df, TUE_BAR, h=1.2005, c=1.1950)  # sweep PDH -> spent
    f = pb.features_at_alert(df, ASOF, "bullish", 1.1950, 0.0010)
    # Nearest UNSPENT above is now the weekly high, 150 pips away.
    assert f["next_pool_above_tier"] == "PW"
    assert f["dist_next_pool_above_atr"] == 15.0
    assert f["pdh_status_at_alert"] == "swept"
    assert f["last_sweep_tier"] == "PD"
    assert f["last_sweep_age_h1"] is not None and f["last_sweep_age_h1"] >= 0


def test_day_state_vocabulary_is_closed():
    assert pb.day_state("intact", "intact") == "INSIDE"
    assert pb.day_state("broken", "intact") == "EXPANSION_UP"
    assert pb.day_state("intact", "broken") == "EXPANSION_DOWN"
    assert pb.day_state("swept", "intact") == "SWEPT_HIGH"
    assert pb.day_state("intact", "swept") == "SWEPT_LOW"
    assert pb.day_state("swept", "broken") == "BOTH_SIDES"
    assert pb.day_state("broken", "broken") == "BOTH_SIDES"
    assert pb.day_state(None, "intact") is None


# ---------------------------------------------------------------------------
# Live-path helpers (pure parts only — no network)
# ---------------------------------------------------------------------------

def test_drop_forming():
    df = two_weeks()
    last = df.index[-1]
    assert len(pb.drop_forming(df, last + pd.Timedelta(minutes=30))) == len(df) - 1
    assert len(pb.drop_forming(df, last + pd.Timedelta(hours=1))) == len(df)


def test_strip_for_pools_server_weekend():
    rows = quiet_day("2026-06-05", 1.2, 1.19)  # Friday (server)
    # Friday 21:00–23:00 UTC = SATURDAY server date -> must be stripped.
    for h in (21, 22, 23):
        ts = pd.Timestamp(f"2026-06-05 {h:02d}:00")
        rows.append((ts, 1.195, 1.196, 1.194, 1.195))
    df = frame(rows)
    out = pb.strip_for_pools(df)
    assert len(out) == 24
    assert out.index.max() == pd.Timestamp("2026-06-05 20:00")


def test_format_pool_line():
    df = two_weeks()
    patch_bar(df, TUE_BAR, h=1.2005, c=1.1950)
    snap = snap_at(df)
    line = pb.format_pool_line(snap, 5)
    # Compact glyph form (2026-07-14 de-bloat): PDH swept, PDL untouched,
    # day-state as a short tag. Long meaning lives once in POOL_LEGEND.
    assert "PDH swept" in line          # swept glyph on the prior-day high
    assert "PDL ✓" in line              # untouched prior-day low
    assert "high swept" in line         # short day-state tag, not a sentence
    assert "SWEPT (dipped past" not in line  # verbose phrasing is gone


def test_format_liquidity_inference_names_exact_pool():
    """P2 inference line must name the EXACT pool the trade points at (not a
    vague 'level') and read toward/away correctly. Built from feature dicts
    directly — no frame needed."""
    # LONG toward a near prior-day HIGH -> names PDH, calls it a target.
    f = pb.features_none()
    f.update({"trade_toward_pool": True,
              "next_pool_above_tier": "PD", "dist_next_pool_above_atr": 1.2})
    line = pb.format_liquidity_inference(f, "LONG")
    assert "yesterday's high (PDH)" in line
    assert "target" in line.lower()

    # SHORT toward a near prior-week LOW -> names PWL.
    f = pb.features_none()
    f.update({"trade_toward_pool": True,
              "next_pool_below_tier": "PW", "dist_next_pool_below_atr": 0.8})
    line = pb.format_liquidity_inference(f, "SHORT")
    assert "last week's low (PWL)" in line

    # LONG running AWAY -> names the above pool, warns weaker follow-through.
    f = pb.features_none()
    f.update({"trade_toward_pool": False,
              "next_pool_above_tier": "PW", "dist_next_pool_above_atr": 2.0})
    line = pb.format_liquidity_inference(f, "LONG")
    assert "AWAY" in line and "last week's high (PWH)" in line

    # Far pool (>=3 ATR) toward -> tempered wording, still names the pool.
    f = pb.features_none()
    f.update({"trade_toward_pool": True,
              "next_pool_above_tier": "PD", "dist_next_pool_above_atr": 4.5})
    line = pb.format_liquidity_inference(f, "LONG")
    assert "far" in line.lower() and "yesterday's high (PDH)" in line

    # No trade_toward signal -> None (nothing to say).
    assert pb.format_liquidity_inference(pb.features_none(), "LONG") is None


def test_naive_utc_index_accepts_live_reset_index_frame():
    """REGRESSION (2026-07-14): the live engine passes a reset-index frame with
    the timestamp in a `Datetime` column (smc_radar.fetch_data), not the index.
    _naive_utc_index must restore the DatetimeIndex instead of crashing on
    `df.index.tz` ('RangeIndex' object has no attribute 'tz'), which silently
    killed the pool layer on every live scan.

    Failure mode: pool_context_fail on all pairs every scan -> blank pool line.
    Guard lives out of the live path (offline test). Trading impact if
    unguarded: pool observation context (never gates/scores) stays None.
    """
    df = two_weeks()                       # DatetimeIndex, tz-naive
    live_shape = df.copy()
    live_shape.index.name = "Datetime"
    live_shape = live_shape.reset_index()  # RangeIndex + 'Datetime' column
    assert not isinstance(live_shape.index, pd.DatetimeIndex)

    out = pb._naive_utc_index(live_shape)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is None
    # Same values, same order as the DatetimeIndexed source.
    assert list(out.index) == list(df.index)

    # tz-aware column variant also lands as tz-naive UTC.
    aware = df.copy()
    aware.index = aware.index.tz_localize("UTC")
    aware.index.name = "Datetime"
    aware = aware.reset_index()
    out_aware = pb._naive_utc_index(aware)
    assert isinstance(out_aware.index, pd.DatetimeIndex)
    assert out_aware.index.tz is None
