"""Weekly PD zone — structural guards (observe-only, 2026-07-15).

Covers the B5 guard class for the new weekly-PD columns:
  1) NO-LOOK-AHEAD — the feature dict at time t is identical whether or not
     future bars exist in the frame. THE bug this kills: the weekly range at
     alert time must never see a bar that closed after the alert. A future
     week that would move PWH/PWL is appended and must NOT change the answer.
  2) DETERMINISM — same frame twice -> identical output.
  3) FIXED BOUNDARY / break-through — the boundaries do NOT re-anchor when
     price closes beyond them; the % runs > 1.0 above last week's high and
     < 0.0 below last week's low (the break signal, by owner decision).
  4) AGREEMENT — both_premium / both_discount / mixed at the 0.5 split,
     including the broken-level (>1 / <0) cases still resolving.
  5) COLUMN CONTRACT — exactly WEEKLY_PD_FEATURE_COLUMNS, all-None on thin
     history / degenerate range.

Run:  python -m pytest tests/test_weekly_pd.py -q
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

import pool_builder as pb  # noqa: E402
import weekly_pd as wpd  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic H1 builder. Server day D = UTC [D-1 21:00, D 20:00].
# 2026-06-01 is a Monday. One "quiet day" prints its full high/low in bar 0
# and sits at mid otherwise, so day/week extremes are exactly (hi, lo).
# ---------------------------------------------------------------------------

def quiet_day(server_day, hi, lo):
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


def week_of(hi, lo, monday="2026-06-01"):
    """Five quiet weekday bars for one week; the week high/low = (hi, lo)
    printed on Monday, the rest quiet. Returns the row list."""
    rows = []
    day = pd.Timestamp(monday)
    for d in range(5):  # Mon-Fri
        sd = day + pd.Timedelta(days=d)
        rows += quiet_day(sd, hi if d == 0 else (hi + lo) / 2.0 + 1e-6,
                          lo if d == 0 else (hi + lo) / 2.0 - 1e-6)
    return rows


# ---------------------------------------------------------------------------
# 1) NO-LOOK-AHEAD — the load-bearing guard
# ---------------------------------------------------------------------------

def test_no_look_ahead():
    """The weekly range at an alert in week 2 must be week 1's completed
    high/low — and must NOT change when a (higher-high / lower-low) FUTURE
    week 3 is appended to the frame. If the future leaked, the answer would
    move; it must not."""
    # Week 1 (completed): high 1.20, low 1.10 -> PWH/PWL for a week-2 alert.
    w1 = week_of(1.20, 1.10, monday="2026-06-01")
    # Week 2: alert lands here. Price sits at 1.15 (mid of week 1 range = 0.5).
    w2 = week_of(1.16, 1.14, monday="2026-06-08")
    base = frame(w1 + w2)

    # Alert timestamp: Tuesday of week 2, 12:00 UTC.
    alert_ts = pd.Timestamp("2026-06-09 12:00")

    got = wpd.features_at_alert(base, alert_ts, ref_price=1.15,
                                h4_pd_position=0.60)

    # Append a FUTURE week 3 whose extremes (1.50 / 0.90) would move any
    # look-ahead-leaking weekly range.
    w3 = week_of(1.50, 0.90, monday="2026-06-15")
    future = frame(w1 + w2 + w3)
    got_future = wpd.features_at_alert(future, alert_ts, ref_price=1.15,
                                       h4_pd_position=0.60)

    assert got == got_future, "future bars leaked into the weekly range"
    # And the range is week 1's completed extremes, not week 2's / week 3's.
    assert got["weekly_range_high_at_alert"] == 1.20
    assert got["weekly_range_low_at_alert"] == 1.10
    # 1.15 sits at the mid of [1.10, 1.20] -> 0.5 -> premium.
    assert got["weekly_pd_position_at_alert"] == 0.5
    assert got["weekly_pd_zone_at_alert"] == "premium"


def test_determinism():
    w1 = week_of(1.20, 1.10, monday="2026-06-01")
    w2 = week_of(1.16, 1.14, monday="2026-06-08")
    base = frame(w1 + w2)
    alert_ts = pd.Timestamp("2026-06-09 12:00")
    a = wpd.features_at_alert(base, alert_ts, ref_price=1.15, h4_pd_position=0.6)
    b = wpd.features_at_alert(base, alert_ts, ref_price=1.15, h4_pd_position=0.6)
    assert a == b


# ---------------------------------------------------------------------------
# 3) FIXED BOUNDARY — the % runs past its ends, boundaries never move
# ---------------------------------------------------------------------------

def test_broken_up_runs_above_one():
    """Price above last week's high -> weekly_pd_position > 1.0, boundary
    unchanged (owner: no re-anchor). The % itself is the break signal."""
    r = wpd.weekly_pd_read(price=1.30, weekly_high=1.20, weekly_low=1.10)
    assert r["weekly_range_high"] == 1.20  # boundary NOT moved to 1.30
    assert r["weekly_pd_position"] == 2.0  # (1.30-1.10)/(1.20-1.10)
    assert r["weekly_pd_zone"] == "premium"


def test_broken_down_runs_negative():
    r = wpd.weekly_pd_read(price=1.05, weekly_high=1.20, weekly_low=1.10)
    assert r["weekly_range_low"] == 1.10   # boundary NOT moved to 1.05
    assert r["weekly_pd_position"] == -0.5  # (1.05-1.10)/0.10
    assert r["weekly_pd_zone"] == "discount"


def test_split_at_exactly_half():
    """0.5 is premium (>=), just under is discount (owner anchor)."""
    assert wpd.weekly_pd_read(1.15, 1.20, 1.10)["weekly_pd_zone"] == "premium"
    assert wpd.weekly_pd_read(1.1499, 1.20, 1.10)["weekly_pd_zone"] == "discount"


# ---------------------------------------------------------------------------
# 4) AGREEMENT — 0.5 split on both sides, broken cases still resolve
# ---------------------------------------------------------------------------

def test_agreement_both_premium():
    assert wpd.pd_zone_agreement(0.70, 0.80) == "both_premium"


def test_agreement_both_discount():
    assert wpd.pd_zone_agreement(0.30, 0.20) == "both_discount"


def test_agreement_mixed():
    assert wpd.pd_zone_agreement(0.70, 0.30) == "mixed"
    assert wpd.pd_zone_agreement(0.40, 0.60) == "mixed"


def test_agreement_broken_weekly_still_premium():
    """A broken-up weekly (>1.0) is still premium -> agrees with a premium H4."""
    assert wpd.pd_zone_agreement(0.60, 1.40) == "both_premium"


def test_agreement_broken_weekly_still_discount():
    assert wpd.pd_zone_agreement(0.30, -0.20) == "both_discount"


def test_agreement_none_when_a_read_missing():
    assert wpd.pd_zone_agreement(None, 0.5) is None
    assert wpd.pd_zone_agreement(0.5, None) is None


# ---------------------------------------------------------------------------
# 5) COLUMN CONTRACT + thin/degenerate all-None
# ---------------------------------------------------------------------------

def test_column_contract():
    got = wpd.features_at_alert(
        frame(week_of(1.20, 1.10) + week_of(1.16, 1.14, "2026-06-08")),
        pd.Timestamp("2026-06-09 12:00"), ref_price=1.15, h4_pd_position=0.6)
    assert set(got.keys()) == set(wpd.WEEKLY_PD_FEATURE_COLUMNS)


def test_thin_history_all_none():
    """No completed prior week in frame -> every column None."""
    only_week1 = frame(week_of(1.20, 1.10, monday="2026-06-01"))
    # Alert on Tuesday of the SAME (first) week — no completed prior week.
    got = wpd.features_at_alert(only_week1, pd.Timestamp("2026-06-02 12:00"),
                                ref_price=1.15, h4_pd_position=0.6)
    assert got == wpd.features_none()


def test_degenerate_range_all_none():
    r = wpd.weekly_pd_read(price=1.15, weekly_high=1.20, weekly_low=1.20)
    assert r["weekly_pd_position"] is None


def test_missing_h4_gives_none_agreement_but_keeps_weekly():
    """When the H4 read is missing, agreement is None but the weekly read
    (position / zone / boundaries) is still stamped — the two are independent."""
    got = wpd.build_features(price=1.15, weekly_high=1.20, weekly_low=1.10,
                             h4_pd_position=None)
    assert got["weekly_pd_position_at_alert"] == 0.5
    assert got["weekly_pd_zone_at_alert"] == "premium"
    assert got["pd_zone_agreement_at_alert"] is None


# ---------------------------------------------------------------------------
# Email line — predefined sentences name BOTH timeframes + BOTH percentages
# ---------------------------------------------------------------------------

def test_email_line_names_both_percentages():
    feats = wpd.build_features(price=1.184, weekly_high=1.20, weekly_low=1.10,
                              h4_pd_position=0.61)  # weekly pos 0.84
    line = wpd.format_agreement_line(feats, h4_pd_position=0.61, bias="LONG")
    assert line is not None
    assert "84%" in line and "61%" in line
    assert "EXPENSIVE" in line and "cautious" in line.lower()


def test_email_line_none_when_agreement_missing():
    feats = wpd.build_features(price=1.15, weekly_high=1.20, weekly_low=1.10,
                              h4_pd_position=None)
    assert wpd.format_agreement_line(feats, None, "LONG") is None
