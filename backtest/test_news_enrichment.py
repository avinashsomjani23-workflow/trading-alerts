"""Guard tests for backtest/news_enrichment.py (offline analysis tooling).

Synthetic events + trades; no network, no live path. Every boundary the
+/-1h-from-candle rule promises is pinned here so a regression can't ship
silently.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_enrichment import enrich, load_events  # noqa: E402


def _load_from_frame(frame):
    import tempfile
    fd, p = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    frame.to_csv(p, index=False)
    try:
        from news_enrichment import load_events as le
        return le(p)
    finally:
        os.unlink(p)


def _trades(rows):
    """rows: (pair, fill_iso|None, exit_iso|None)"""
    return pd.DataFrame(
        [{"pair": p, "fill_ts": f, "exit_ts": x} for p, f, x in rows])


# Baseline calendar: enough low-impact noise in every month touched by the
# tests that no coverage hole fires unless a test wants one.
def _noise(ccys=("USD", "EUR", "GBP"), months=("2020-01", "2020-02")):
    out = []
    for c in ccys:
        for m in months:
            out.append((f"{m}-15T10:00:00Z", c, "Minor Print", "low", "6:00am"))
    return out


def test_fill_window_inclusive_edges():
    # fill bar open 12:00 -> window [11:00, 14:00] inclusive
    ev = _mk([("2020-01-10 11:00", "USD", "At Lower Edge", "high", "6:00am"),
              ("2020-01-10 14:00", "USD", "At Upper Edge", "high", "9:00am")])
    tr = _trades([("EURUSD", "2020-01-10T12:00:00+00:00",
                   "2020-01-10T15:00:00+00:00")])
    res, _ = enrich(tr, ev)
    assert res.loc[0, "news_fill"] == 1
    assert res.loc[0, "news_fill_event"] in ("At Lower Edge", "At Upper Edge")


def _with_noise(frame, months=("2019-12", "2020-01", "2020-02")):
    noise = pd.DataFrame({
        "dateline": [pd.Timestamp(f"{m}-15T10:00", tz="UTC").value // 10**9
                     for m in months for _ in ("USD", "EUR")],
        "currency": ["USD", "EUR"] * len(months),
        "name": ["Minor Print"] * (2 * len(months)),
        "impactName": ["low"] * (2 * len(months)),
        "timeLabel": ["6:00am"] * (2 * len(months)),
    })
    return pd.concat([frame, noise], ignore_index=True)


def _mk(rows, months=("2019-12", "2020-01", "2020-02")):
    """rows: (iso, ccy, name, impact, label) -> loaded events frame + noise."""
    f = pd.DataFrame({
        "dateline": [pd.Timestamp(r[0], tz="UTC").value // 10**9 for r in rows],
        "currency": [r[1] for r in rows],
        "name": [r[2] for r in rows],
        "impactName": [r[3] for r in rows],
        "timeLabel": [r[4] for r in rows],
    })
    return _load_from_frame(_with_noise(f, months))


FILL = "2020-01-10T12:00:00+00:00"
EXIT = "2020-01-12T15:00:00+00:00"


def test_outside_window_not_flagged():
    ev = _mk([("2020-01-10 10:59", "USD", "Too Early", "high", "5:59am"),
              ("2020-01-10 14:01", "USD", "Too Late", "high", "9:01am")])
    res, _ = enrich(_trades([("EURUSD", FILL, EXIT)]), ev)
    assert res.loc[0, "news_fill"] == 0


def test_wrong_currency_leg_not_flagged():
    ev = _mk([("2020-01-10 12:30", "GBP", "UK CPI", "high", "7:30am")])
    res, _ = enrich(_trades([("EURUSD", FILL, EXIT)]), ev)
    assert res.loc[0, "news_fill"] == 0


def test_either_leg_flags():
    ev = _mk([("2020-01-10 12:30", "EUR", "German CPI", "high", "7:30am")])
    res, _ = enrich(_trades([("EURUSD", FILL, EXIT)]), ev)
    assert res.loc[0, "news_fill"] == 1
    assert res.loc[0, "news_fill_ccy"] == "EUR"


def test_medium_impact_not_flagged():
    ev = _mk([("2020-01-10 12:30", "USD", "Some Print", "medium", "7:30am")])
    res, _ = enrich(_trades([("EURUSD", FILL, EXIT)]), ev)
    assert res.loc[0, "news_fill"] == 0


def test_speech_excluded_press_conference_kept():
    ev = _mk([("2020-01-10 12:30", "USD", "Fed Chair Powell Speaks",
               "high", "7:30am")])
    res, s = enrich(_trades([("EURUSD", FILL, EXIT)]), ev)
    assert res.loc[0, "news_fill"] == 0
    assert s["speech_high_dropped"] == 1
    ev2 = _mk([("2020-01-10 12:30", "EUR", "ECB Press Conference",
                "high", "7:30am")])
    res2, _ = enrich(_trades([("EURUSD", FILL, EXIT)]), ev2)
    assert res2.loc[0, "news_fill"] == 1


def test_untimed_excluded_but_counted():
    ev = _mk([("2020-01-10 12:30", "USD", "WEF Annual Meetings",
               "high", "All Day"),
              ("2020-01-10 12:45", "USD", "OPEC Meetings",
               "high", "Tentative")])
    res, s = enrich(_trades([("EURUSD", FILL, EXIT)]), ev)
    assert res.loc[0, "news_fill"] == 0
    assert s["untimed_high_dropped"] == 2


def test_news_open_hit_and_miss():
    # event 2 days into the trade, far outside the fill window
    ev = _mk([("2020-01-12 10:00", "USD", "Mid-Trade CPI", "high", "5:00am")])
    res, _ = enrich(_trades([("EURUSD", FILL, EXIT)]), ev)
    assert res.loc[0, "news_fill"] == 0
    assert res.loc[0, "news_open"] == 1
    assert res.loc[0, "news_open_event"] == "Mid-Trade CPI"
    # event after exit bar close -> clean
    ev2 = _mk([("2020-01-12 16:01", "USD", "Post-Exit CPI", "high", "11:01am")])
    res2, _ = enrich(_trades([("EURUSD", FILL, EXIT)]), ev2)
    assert res2.loc[0, "news_open"] == 0


def test_never_filled_is_none():
    ev = _mk([("2020-01-10 12:30", "USD", "CPI", "high", "7:30am")])
    res, _ = enrich(_trades([("EURUSD", None, None)]), ev)
    assert pd.isna(res.loc[0, "news_fill"])
    assert pd.isna(res.loc[0, "news_open"])


def test_coverage_hole_is_none_not_zero():
    # trade in 2020-03: noise exists only for 2020-01/02 -> hole for both legs
    ev = _mk([("2020-01-10 12:30", "USD", "CPI", "high", "7:30am"),
              ("2020-03-25 12:30", "USD", "Anchor keeps range wide",
               "low", "7:30am")])
    res, s = enrich(_trades([("EURUSD", "2020-03-10T12:00:00+00:00",
                              "2020-03-10T15:00:00+00:00")]), ev)
    assert pd.isna(res.loc[0, "news_fill"])
    assert s["hole_hits"] == 1


def test_outside_events_range_is_none():
    ev = _mk([("2020-01-10 12:30", "USD", "CPI", "high", "7:30am")])
    res, _ = enrich(_trades([("EURUSD", "2019-06-10T12:00:00+00:00",
                              "2019-06-10T15:00:00+00:00")]), ev)
    assert pd.isna(res.loc[0, "news_fill"])


def test_idempotent_rerun():
    ev = _mk([("2020-01-10 12:30", "USD", "CPI", "high", "7:30am")])
    tr = _trades([("EURUSD", FILL, EXIT)])
    once, _ = enrich(tr, ev)
    twice, _ = enrich(once, ev)
    assert list(once.columns) == list(twice.columns)
    assert once.loc[0, "news_fill"] == twice.loc[0, "news_fill"] == 1


# The pure era-table unit test moved to test_mt5_clock.py (Part A, 2026-07-16).
# The two tests below exercise enrich()'s clock CONTRACT AFTER Part C: candles
# arrive as TRUE UTC (import_mt5 corrects at source), so enrich no longer shifts
# fill_ts/exit_ts — it only NULLs the 2014 flip window.


def test_fill_ts_treated_as_true_utc_no_shift():
    # Post-Part-C: fill_ts IS true UTC. A 12:00 fill -> window [11:00, 14:00]
    # inclusive, with NO era shift applied by enrich. Event at 13:30 (era-B
    # summer date) MUST flag now — the old code would have shifted the window to
    # [10:00, 13:00] and missed it. This pins that the double-correction is gone.
    ev = _mk([("2016-07-10 13:30", "USD", "In Window", "high", "9:30am")],
             months=("2016-06", "2016-07", "2016-08"))
    tr = _trades([("EURUSD", "2016-07-10T12:00:00+00:00",
                   "2016-07-10T15:00:00+00:00")])
    res, _ = enrich(tr, ev)
    assert res.loc[0, "news_fill"] == 1
    # An event past the true-UTC window edge (14:01) must NOT flag.
    ev2 = _mk([("2016-07-10 14:01", "USD", "Past Edge", "high", "10:01am")],
              months=("2016-06", "2016-07", "2016-08"))
    res2, _ = enrich(tr, ev2)
    assert res2.loc[0, "news_fill"] == 0


def test_ambiguous_clock_window_is_none():
    # Flip-window fills keep their provisional label (in the flip date range),
    # so enrich still NULLs them via is_flip_window.
    ev = _mk([("2014-11-20 13:30", "USD", "CPI", "high", "8:30am")],
             months=("2014-10", "2014-11", "2014-12"))
    tr = _trades([("EURUSD", "2014-11-20T13:00:00+00:00",
                   "2014-11-20T16:00:00+00:00")])
    res, s = enrich(tr, ev)
    assert pd.isna(res.loc[0, "news_fill"])
    assert s["clock_ambiguous"] == 1


def test_unmapped_pair_raises():
    ev = _mk([("2020-01-10 12:30", "USD", "CPI", "high", "7:30am")])
    with pytest.raises(ValueError, match="no currency mapping"):
        enrich(_trades([("EURNOK", FILL, EXIT)]), ev)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
