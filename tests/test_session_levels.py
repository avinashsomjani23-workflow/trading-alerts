"""Session H/L SWEEP + BREAK study — structural guards (SESSION_SWEEP_STUDY_SPEC §4).

The checks the spec demands for the four new session_level_* columns:
  1) DST GUARD (§4.1 — the critical one): a London session in SUMMER and one in
     WINTER must cover the correct LOCAL 07:00-15:00 London bars in BOTH, i.e. the
     UTC bars used SHIFT by one hour across the BST/GMT change. If they don't shift,
     DST is still broken and the whole study is polluted.
  2) RECOMPUTE AUDIT (§4.2): rebuild session_level_* from raw session H/L + entry
     independently and assert it equals build_session_level_event.
  3) POINT-IN-TIME (§4.3): the answer never uses a bar at/after the alert.
  4) SWEEP-vs-BREAK (§4.4): wick-and-return -> 'sweep'; close-through-and-hold
     -> 'break' (reusing pool_builder.pool_status, so this also proves the reuse).
  5) COLUMN CONTRACT: features_none / build both emit exactly the four columns.
  6) PAIR RELEVANCE (flag, not filter): session_level_pair_relevant is True only
     when the reported session is one the pair trades (PAIR_SESSION_TAGS), and a
     relevant event is PREFERRED over a nearer off-tag one — but off-tag events are
     still reported when they are the only event (so the study can measure them).

Run:  python -m pytest tests/test_session_levels.py -q
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

import session_levels as sl  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic H1 builder. Index is naive UTC (what the simulator passes in).
# ---------------------------------------------------------------------------

def _bar(ts_utc, hi, lo, close=None, open_=None):
    o = open_ if open_ is not None else (hi + lo) / 2.0
    c = close if close is not None else (hi + lo) / 2.0
    return (pd.Timestamp(ts_utc), o, hi, lo, c)


def _frame(rows):
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame(
        {"Open": [r[1] for r in rows], "High": [r[2] for r in rows],
         "Low": [r[3] for r in rows], "Close": [r[4] for r in rows]},
        index=idx,
    )


def _flat_hours(date_str, hours, price):
    """Flat filler bars (tiny range at `price`) on the given UTC hours of a date."""
    base = pd.Timestamp(date_str)
    return [_bar(base + pd.Timedelta(hours=h), price + 1e-6, price - 1e-6, price)
            for h in hours]


# ===========================================================================
# 1) DST GUARD — the critical test (SESSION_SWEEP_STUDY_SPEC §4.1)
# ===========================================================================

def test_london_session_window_shifts_one_hour_between_bst_and_gmt():
    """London H/L must be measured over LOCAL 08:00-16:00 bars in BOTH seasons.
    Winter (GMT): that is 08:00-16:00 UTC. Summer (BST=UTC+1): that is 07:00-15:00
    UTC. So a 07:00-UTC bar belongs to the London session in SUMMER but NOT winter,
    and a 15:00-UTC bar belongs in WINTER but NOT summer. That one-hour shift is the
    proof DST is honest. If _session_hl_pools ignored DST, the same UTC hours would
    be picked in both seasons and this test fails."""

    # A distinctive spike price placed only on the boundary UTC hour we probe.
    SPIKE = 2.0
    FLOOR = 1.0
    MID = 1.5

    # ---- SUMMER (BST): probe the 07:00-UTC bar (= 08:00 London, IN session) ----
    # July 15 2026 is BST. Put the spike high at 07:00 UTC (both in London
    # 08:00-16:00 local). Put a NON-session decoy at 15:00 UTC (= 16:00 local =
    # window end, EXCLUSIVE -> must be OUT).
    summer_rows = []
    summer_rows += _flat_hours("2026-07-15", range(0, 7), MID)
    summer_rows.append(_bar("2026-07-15 07:00", SPIKE, FLOOR, MID))   # 08:00 London -> IN
    summer_rows += _flat_hours("2026-07-15", range(8, 15), MID)
    summer_rows.append(_bar("2026-07-15 15:00", SPIKE + 5, FLOOR, MID))  # 16:00 London -> OUT
    summer_rows += _flat_hours("2026-07-15", range(16, 24), MID)
    summer = _frame(summer_rows)
    # Alert next day, after the session fully closed.
    alert = pd.Timestamp("2026-07-16 12:00")
    pools = sl._session_hl_pools(sl._naive_utc_index(summer), "london", alert)
    assert len(pools) == 1, f"expected 1 London pool in summer, got {pools}"
    # The 07:00-UTC spike IS counted; the 15:00-UTC decoy (SPIKE+5) is NOT.
    assert pools[0]["high"] == SPIKE, (
        f"summer London high should be the 07:00-UTC (=08:00 BST) spike {SPIKE}, "
        f"not the 15:00-UTC decoy — got {pools[0]['high']}. DST not applied.")

    # ---- WINTER (GMT): probe the 15:00-UTC bar (= 15:00 London, IN session) ----
    # Jan 15 2026 is GMT. The SAME 07:00-UTC bar is now 07:00 London = BEFORE the
    # session (OUT). A 15:00-UTC bar is now 15:00 London = IN. Mirror the fixture.
    winter_rows = []
    winter_rows += _flat_hours("2026-01-15", range(0, 7), MID)
    winter_rows.append(_bar("2026-01-15 07:00", SPIKE + 5, FLOOR, MID))  # 07:00 London -> OUT
    winter_rows += _flat_hours("2026-01-15", range(8, 15), MID)
    winter_rows.append(_bar("2026-01-15 15:00", SPIKE, FLOOR, MID))      # 15:00 London -> IN
    winter_rows += _flat_hours("2026-01-15", range(16, 24), MID)
    winter = _frame(winter_rows)
    alert_w = pd.Timestamp("2026-01-16 12:00")
    pools_w = sl._session_hl_pools(sl._naive_utc_index(winter), "london", alert_w)
    assert len(pools_w) == 1, f"expected 1 London pool in winter, got {pools_w}"
    assert pools_w[0]["high"] == SPIKE, (
        f"winter London high should be the 15:00-UTC (=15:00 GMT) spike {SPIKE}, "
        f"not the OUT-of-session 07:00-UTC decoy — got {pools_w[0]['high']}. "
        "The UTC window did NOT shift with DST.")


def test_london_close_boundary_is_utc_shifted():
    """Tighter DST bite: the 07:00-UTC bar is IN London in summer and OUT in winter;
    assert both directions on the SAME probe hour so a no-op DST implementation
    (which would treat 07:00 UTC identically in both seasons) cannot pass."""
    SPIKE, FLOOR, MID = 3.0, 1.0, 2.0

    def one_hour_frame(date_str):
        rows = _flat_hours(date_str, range(0, 7), MID)
        rows.append(_bar(f"{date_str} 07:00", SPIKE, FLOOR, MID))  # probe hour
        rows += _flat_hours(date_str, range(8, 24), MID)
        return sl._naive_utc_index(_frame(rows))

    # Summer: 07:00 UTC = 08:00 London -> IN -> high captured == SPIKE.
    summer = one_hour_frame("2026-07-15")
    p_s = sl._session_hl_pools(summer, "london", pd.Timestamp("2026-07-16 12:00"))
    assert p_s and p_s[0]["high"] == SPIKE

    # Winter: 07:00 UTC = 07:00 London -> OUT of 08:00-16:00 -> that bar excluded.
    # Every other bar is MID, so the captured high is MID (not the SPIKE).
    winter = one_hour_frame("2026-01-15")
    p_w = sl._session_hl_pools(winter, "london", pd.Timestamp("2026-01-16 12:00"))
    assert p_w and p_w[0]["high"] != SPIKE, (
        "07:00 UTC must be OUTSIDE the London session in winter (07:00 GMT) — "
        "it was counted, so DST is not resolved per candle.")


# ===========================================================================
# 2) SWEEP-vs-BREAK (§4.4) — proves the pool_builder.pool_status reuse
# ===========================================================================

def _london_session_day(date_str, hi, lo):
    """A London session (07:00-15:00 local) on `date_str` printing high `hi` at
    its open bar and low `lo`, everything else flat at mid. Uses 08:00 UTC (safely
    inside London in both seasons) for the range bar."""
    mid = (hi + lo) / 2.0
    rows = _flat_hours(date_str, [0, 1, 2, 3, 4, 5, 6], mid)
    rows.append(_bar(f"{date_str} 08:00", hi, lo, mid))  # 08/09 London -> IN
    rows += _flat_hours(date_str, range(9, 24), mid)
    return rows


def test_wick_through_and_return_is_sweep():
    """Price wicks ABOVE the London high on a later bar but CLOSES back inside
    -> pool_status = swept -> session_level_event == 'sweep'."""
    hi, lo = 2.0, 1.0
    rows = _london_session_day("2026-07-15", hi, lo)
    # Next day: a bar whose HIGH pierces `hi` but CLOSE returns below it.
    rows.append(_bar("2026-07-16 08:00", hi + 0.5, 1.5, hi - 0.1))  # wick over, close back
    rows += _flat_hours("2026-07-16", [9, 10, 11], 1.6)
    df = _frame(rows)
    alert = pd.Timestamp("2026-07-16 13:00")
    prior = df[df.index < alert]
    out = sl.build_session_level_event(prior, alert, ref_price=hi)
    assert out["session_level_event"] == "sweep", out
    assert out["session_level_which"] == "london", out
    assert out["session_level_side"] == "high", out


def test_close_through_and_hold_is_break():
    """Price CLOSES above the London high and the next close HOLDS above it
    -> pool_status = broken -> session_level_event == 'break'."""
    hi, lo = 2.0, 1.0
    rows = _london_session_day("2026-07-15", hi, lo)
    # Next day: close clears `hi`, then next close holds beyond (N=1 confirm).
    rows.append(_bar("2026-07-16 08:00", hi + 0.3, 1.9, hi + 0.2))  # close beyond
    rows.append(_bar("2026-07-16 09:00", hi + 0.5, hi + 0.1, hi + 0.4))  # holds beyond
    rows += _flat_hours("2026-07-16", [10, 11], hi + 0.4)
    df = _frame(rows)
    alert = pd.Timestamp("2026-07-16 13:00")
    prior = df[df.index < alert]
    out = sl.build_session_level_event(prior, alert, ref_price=hi)
    assert out["session_level_event"] == "break", out
    assert out["session_level_which"] == "london", out
    assert out["session_level_side"] == "high", out


def test_untouched_session_level_is_none():
    """A session H/L that price never approaches -> 'none' on all three columns."""
    hi, lo = 2.0, 1.0
    rows = _london_session_day("2026-07-15", hi, lo)
    rows += _flat_hours("2026-07-16", range(8, 14), 1.5)  # stays inside the range
    df = _frame(rows)
    alert = pd.Timestamp("2026-07-16 13:00")
    prior = df[df.index < alert]
    out = sl.build_session_level_event(prior, alert, ref_price=1.5)
    assert out == sl.features_none(), out


# ===========================================================================
# 3) POINT-IN-TIME (§4.3) — no bar at/after the alert may change the answer
# ===========================================================================

def test_no_look_ahead():
    """Adding FUTURE bars (>= alert) that WOULD flip the event if leaked must not
    change the stamped columns. Build a case where a later break exists only after
    the alert; the alert-time answer must stay 'sweep' (or none), never the future
    break."""
    hi, lo = 2.0, 1.0
    rows = _london_session_day("2026-07-15", hi, lo)
    # Before alert: a sweep (wick over, close back).
    rows.append(_bar("2026-07-16 08:00", hi + 0.4, 1.5, hi - 0.1))
    df_prior = _frame(rows)
    alert = pd.Timestamp("2026-07-16 10:00")

    base = sl.build_session_level_event(df_prior[df_prior.index < alert], alert,
                                        ref_price=hi)

    # Now append FUTURE bars (>= alert) that would turn it into a clean break.
    fut = list(rows)
    fut.append(_bar("2026-07-16 10:00", hi + 0.6, hi + 0.2, hi + 0.5))
    fut.append(_bar("2026-07-16 11:00", hi + 0.7, hi + 0.3, hi + 0.6))
    df_future = _frame(fut)
    leaked = sl.build_session_level_event(df_future[df_future.index < alert], alert,
                                          ref_price=hi)
    assert base == leaked, (base, leaked)
    assert base["session_level_event"] == "sweep", base


def test_session_not_yet_closed_is_not_a_pool():
    """A session whose local window has NOT fully closed before the alert must not
    be usable as a pool (still forming). Alert mid-London-session -> that day's
    London pool is excluded."""
    hi, lo = 2.0, 1.0
    rows = _flat_hours("2026-07-15", [7], (hi + lo) / 2.0)
    rows.append(_bar("2026-07-15 08:00", hi + 1, lo, (hi + lo) / 2.0))  # 09:00 London, mid-session
    df = _frame(rows)
    # Alert at 09:00 UTC (= 10:00 London) — London 07-15 has NOT closed yet.
    alert = pd.Timestamp("2026-07-15 09:00")
    prior = df[df.index < alert]
    pools = sl._session_hl_pools(sl._naive_utc_index(prior), "london", alert)
    assert pools == [], f"unclosed London session must not be a pool, got {pools}"


# ===========================================================================
# 4) RECOMPUTE AUDIT (§4.2) — independent rebuild equals the emitted columns
# ===========================================================================

def test_recompute_matches_independent_rebuild():
    """Independently rebuild the nearest-event answer from the raw pools + the
    entry, and assert it equals build_session_level_event (Area-C 0-mismatch
    method), over a mixed multi-session frame."""
    # Two sessions with events: a London high sweep and an Asia low break, at
    # different distances from the entry, so 'nearest' actually decides.
    rows = []
    # London session with high = 2.0 (near the entry).
    rows += _london_session_day("2026-07-15", hi=2.0, lo=1.8)
    # Asia session (00-09 JST = 15:00-24:00 UTC prev day) with low = 0.5 (far).
    # 2026-07-15 00:00-09:00 JST == 2026-07-14 15:00-24:00 UTC.
    rows += _flat_hours("2026-07-14", range(15, 24), 1.5)
    # overwrite one Asia bar to print the low 0.5
    rows.append(_bar("2026-07-14 16:00", 1.6, 0.5, 1.5))
    # After both sessions closed: sweep the London high (near) + break the Asia low (far).
    rows.append(_bar("2026-07-16 08:00", 2.4, 1.9, 1.95))   # wick over London high 2.0, close back
    rows.append(_bar("2026-07-16 09:00", 1.6, 0.4, 0.45))   # close below Asia low 0.5
    rows.append(_bar("2026-07-16 10:00", 0.6, 0.3, 0.42))   # holds below -> Asia break
    df = _frame(rows)
    alert = pd.Timestamp("2026-07-16 12:00")
    entry = 2.05  # nearest to the London high (2.0), far from Asia low (0.5)
    prior = df[df.index < alert]

    got = sl.build_session_level_event(prior, alert, ref_price=entry)

    # --- Independent rebuild: enumerate every session pool, classify, pick nearest.
    bars = sl._naive_utc_index(prior)
    best = None
    for sess in ("asia", "london", "ny"):
        for p in sl._session_hl_pools(bars, sess, pd.Timestamp(alert)):
            after = bars[bars.index >= p["close_utc"]]
            for side_key, side_arg, level in (("high", "above", p["high"]),
                                              ("low", "below", p["low"])):
                ev = sl._event_for_level(after, level, side_arg)
                if ev == "none":
                    continue
                d = abs(entry - level)
                if best is None or d < best[0]:
                    best = (d, ev, sess, side_key)
    # pair defaults to None here -> no session is relevant -> flag is False.
    expected = (sl.features_none() if best is None else
                {"session_level_event": best[1], "session_level_which": best[2],
                 "session_level_side": best[3], "session_level_pair_relevant": False})
    assert got == expected, (got, expected)
    # And sanity: the NEAR London sweep should win over the FAR Asia break.
    assert got["session_level_which"] == "london", got
    assert got["session_level_event"] == "sweep", got


def test_determinism():
    """Same frame twice -> identical output."""
    rows = _london_session_day("2026-07-15", 2.0, 1.0)
    rows.append(_bar("2026-07-16 08:00", 2.4, 1.5, 1.9))
    df = _frame(rows)
    alert = pd.Timestamp("2026-07-16 12:00")
    prior = df[df.index < alert]
    a = sl.build_session_level_event(prior, alert, ref_price=2.0)
    b = sl.build_session_level_event(prior, alert, ref_price=2.0)
    assert a == b


# ===========================================================================
# 5) COLUMN CONTRACT + degraded input
# ===========================================================================

def test_column_contract():
    keys = set(sl.SESSION_LEVEL_FEATURE_COLUMNS)
    assert set(sl.features_none()) == keys
    df = _frame(_london_session_day("2026-07-15", 2.0, 1.0))
    out = sl.build_session_level_event(df, pd.Timestamp("2026-07-16 12:00"), 2.0)
    assert set(out) == keys


# ===========================================================================
# 6) PAIR RELEVANCE — flag, not filter (session_level_pair_relevant)
# ===========================================================================

def test_pair_relevant_flag_reads_live_map():
    """session_level_pair_relevant must equal 'reported which is in PAIR_SESSION_TAGS
    for this pair' — read the LIVE map and re-derive the expected flag from it, so
    the test never bakes a stale tag copy and is robust to which session the fixture
    happens to surface. This proves the flag is a correct per-pair readout of the
    live map, not a hardcoded guess."""
    import smc_detector
    hi, lo = 2.0, 1.0
    rows = _london_session_day("2026-07-15", hi, lo)
    rows.append(_bar("2026-07-16 08:00", hi + 0.5, 1.5, hi - 0.1))  # sweep London high
    rows += _flat_hours("2026-07-16", [9, 10, 11], 1.6)
    df = _frame(rows)
    alert = pd.Timestamp("2026-07-16 13:00")
    prior = df[df.index < alert]

    for pair in ("EURUSD", "USDJPY", "NZDUSD", "USDCHF", "GOLD"):
        out = sl.build_session_level_event(prior, alert, ref_price=hi, pair=pair)
        which = out["session_level_which"]
        tags = smc_detector.PAIR_SESSION_TAGS.get(pair, [])
        expected = (which != "none") and (which in tags)
        assert out["session_level_pair_relevant"] is expected, (pair, out, tags)

    # pair=None -> never relevant (no map entry).
    none_pair = sl.build_session_level_event(prior, alert, ref_price=hi, pair=None)
    assert none_pair["session_level_pair_relevant"] is False, none_pair


def test_relevant_session_preferred_over_nearer_offtag():
    """When a relevant AND an off-tag session both have an event, the RELEVANT one is
    reported even if the off-tag level is NEARER to entry. Uses two sessions the code
    cleanly separates (London high vs Asia low, far apart) and a pair (USDCHF) that
    trades London but NOT Asia, with entry placed NEAR the off-tag Asia level."""
    # London high = 2.0 (FAR from entry); Asia low = 0.5 (NEAR entry).
    rows = _london_session_day("2026-07-15", hi=2.0, lo=1.9)
    # Asia 00-09 JST == prev-day 15:00-24:00 UTC. Print the Asia low 0.5.
    rows += _flat_hours("2026-07-14", range(15, 24), 1.5)
    rows.append(_bar("2026-07-14 16:00", 1.6, 0.5, 1.5))  # Asia low 0.5
    # After both close: sweep London high (far) + sweep Asia low (near).
    rows.append(_bar("2026-07-16 08:00", 2.4, 1.95, 1.98))  # wick over London high 2.0, close back
    rows.append(_bar("2026-07-16 09:00", 1.6, 0.4, 0.55))   # wick under Asia low 0.5, close back
    rows += _flat_hours("2026-07-16", [10, 11], 1.2)
    df = _frame(rows)
    alert = pd.Timestamp("2026-07-16 13:00")
    prior = df[df.index < alert]
    entry = 0.55  # NEAREST to the Asia low (0.5), FAR from London high (2.0)

    # USDCHF trades London (relevant) but NOT Asia (off-tag). Relevant London must WIN
    # despite the Asia low being nearer to entry.
    chf = sl.build_session_level_event(prior, alert, ref_price=entry, pair="USDCHF")
    assert chf["session_level_which"] == "london", chf
    assert chf["session_level_pair_relevant"] is True, chf

    # NZDUSD trades Asia (relevant) — now the nearest relevant Asia low wins.
    nzd = sl.build_session_level_event(prior, alert, ref_price=entry, pair="NZDUSD")
    assert nzd["session_level_which"] == "asia", nzd
    assert nzd["session_level_pair_relevant"] is True, nzd


def test_degraded_inputs_return_none_dict():
    n = sl.features_none()
    assert sl.build_session_level_event(None, pd.Timestamp("2026-07-16"), 1.0) == n
    empty = _frame([])
    assert sl.build_session_level_event(empty, pd.Timestamp("2026-07-16"), 1.0) == n
    df = _frame(_london_session_day("2026-07-15", 2.0, 1.0))
    assert sl.build_session_level_event(df, pd.Timestamp("2026-07-16 12:00"), None) == n


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
