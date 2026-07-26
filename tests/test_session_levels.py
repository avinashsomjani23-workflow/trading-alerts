"""Session H/L SWEEP + BREAK study — structural guards (SESSION_SWEEP_STUDY_SPEC §4).

The checks the spec demands for the four new session_level_* columns:
  1) DST GUARD (§4.1 — the critical one): a London session in SUMMER and one in
     WINTER must cover the correct LOCAL 07:00-15:00 London bars in BOTH, i.e. the
     UTC bars used SHIFT by one hour across the BST/GMT change. If they don't shift,
     DST is still broken and the whole study is polluted.
  2) RECOMPUTE AUDIT (§4.2): rebuild session_level_* from raw session H/L + entry
     independently (RECENCY pick) and assert it equals build_session_level_event.
  3) POINT-IN-TIME (§4.3): the answer never uses a bar at/after the alert.
  4) SWEEP-vs-BREAK (§4.4): wick-and-return -> 'sweep'; close-through-and-hold
     -> 'break' (reusing pool_builder.pool_status, so this also proves the reuse).
  5) COLUMN CONTRACT: features_none / build both emit exactly the four columns.
  6) PAIR RELEVANCE (flag, not filter): session_level_pair_relevant is True only
     when the reported session is one the pair trades (PAIR_SESSION_TAGS). It is a
     recorded FLAG — it does NOT reorder the pick (the reported session is always the
     most-recently-closed one, on-tag or not, so the study can measure off-tag too).

  RECENCY PICK (2026-07-27, the design fix): the reported session is the ONE whose
  window closed LATEST before the alert — never an older session, never nearest-in-
  price across history (the removed bug). If that session had no event -> 'none' (no
  fall-back to an older session). If both its high and low fired -> the side nearer
  to entry.

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
    """A London session printing high `hi` / low `lo`, filler flat at mid.

    Emits bars ONLY on UTC hours 08:00-11:00 — inside the London window AND clear of
    the NY and Asia windows in BOTH seasons, so London is the sole session this pool
    creates:
      - London: BST 08-16 local = 07-15 UTC / GMT = 08-16 UTC  -> 08-11 UTC always IN.
      - NY:     EDT 08-17 local = 12-21 UTC / EST = 13-22 UTC  -> 08-11 UTC always OUT.
      - Asia:   JST 00-09        = 15-24 UTC (prev day)         -> 08-11 UTC always OUT.
    London and NY genuinely OVERLAP at midday (12-16 local), so any filler at/after
    12:00 UTC would create a stray NY pool that closes LATER than London (21:00 UTC)
    and steal the recency pick — that is exactly the pollution the old all-24-hour
    helper hid behind the retired nearest-in-price pick. Keeping filler to 08-11 UTC
    makes the recency pick unambiguous."""
    mid = (hi + lo) / 2.0
    rows = [_bar(f"{date_str} 08:00", hi, lo, mid)]      # 09:00 London -> IN
    rows += _flat_hours(date_str, [9, 10, 11], mid)      # London-only filler
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
    """Independently rebuild the RECENCY answer from the raw pools + the entry, and
    assert it equals build_session_level_event (Area-C 0-mismatch method), over a
    mixed multi-session frame where recency (not price) must decide."""
    # Two sessions with events, closing at DIFFERENT times so recency decides:
    #   Asia 2026-07-15 (00-09 JST == 2026-07-14 15:00-24:00 UTC) closes EARLIER;
    #   London 2026-07-15 (08-16 BST == 07:00-15:00 UTC) closes LATER (15:00 UTC).
    # London is the MOST-RECENTLY-CLOSED completed session -> it must be reported,
    # even though the Asia low is NEARER to entry (proves price no longer decides).
    rows = []
    # London session: high = 2.0 (FAR from entry), low = 0.1 (parked below the whole
    # price path so London's LOW is never touched — only its HIGH gets swept).
    rows += _london_session_day("2026-07-15", hi=2.0, lo=0.1)
    # Asia session with low = 0.5 (NEAR the entry).
    rows += _flat_hours("2026-07-14", range(15, 24), 1.5)
    rows.append(_bar("2026-07-14 16:00", 1.6, 0.5, 1.5))  # Asia low 0.5
    # After both sessions closed (all bars on London-only UTC hours 08-11):
    #   sweep the London HIGH (far from entry) + break the Asia LOW (near entry),
    #   the down-move staying between London low 0.1 and Asia low 0.5.
    rows.append(_bar("2026-07-16 08:00", 2.4, 1.9, 1.95))   # wick over London high 2.0, close back -> London high sweep
    rows.append(_bar("2026-07-16 09:00", 1.6, 0.4, 0.45))   # close below Asia low 0.5
    rows.append(_bar("2026-07-16 10:00", 0.6, 0.3, 0.42))   # holds below -> Asia break
    df = _frame(rows)
    alert = pd.Timestamp("2026-07-16 12:00")
    entry = 0.55  # NEAREST to the Asia low (0.5), FAR from London high (2.0)
    prior = df[df.index < alert]

    got = sl.build_session_level_event(prior, alert, ref_price=entry)

    # --- Independent rebuild: find the most-recently-closed session, then classify
    # ONLY its two sides; both-fired tiebreak = nearest to entry. Nothing older.
    bars = sl._naive_utc_index(prior)
    newest = None  # (close_utc, sess, high, low)
    for sess in ("asia", "london", "ny"):
        for p in sl._session_hl_pools(bars, sess, pd.Timestamp(alert)):
            if newest is None or p["close_utc"] > newest[0]:
                newest = (p["close_utc"], sess, p["high"], p["low"])
    if newest is None:
        expected = sl.features_none()
    else:
        close_utc, sess, hi, lo = newest
        after = bars[bars.index >= close_utc]
        fired = []
        for side_key, side_arg, level in (("high", "above", hi),
                                          ("low", "below", lo)):
            ev = sl._event_for_level(after, level, side_arg)
            if ev != "none":
                fired.append((side_key, ev, abs(entry - level)))
        if not fired:
            expected = sl.features_none()
        else:
            b = min(fired, key=lambda f: f[2])
            # pair defaults to None -> no session relevant -> flag False.
            expected = {"session_level_event": b[1], "session_level_which": sess,
                        "session_level_side": b[0],
                        "session_level_pair_relevant": False}
    assert got == expected, (got, expected)
    # Sanity: the MOST-RECENT (London) session wins on recency, NOT the nearer Asia low.
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


def test_older_session_event_is_not_reported_over_more_recent_quiet_session():
    """RECENCY, not event-hunting: an OLDER session with an event must NOT be reported
    when a MORE-RECENTLY-CLOSED session exists — even if the recent session had no
    event. The pick is the last session that closed, full stop; we never reach back to
    find something interesting (the removed disease)."""
    # Asia 2026-07-15 (00-09 JST == 2026-07-14 15-24 UTC) prints a low that gets BROKEN.
    rows = _flat_hours("2026-07-14", range(15, 24), 1.5)
    rows.append(_bar("2026-07-14 16:00", 1.6, 0.5, 1.5))  # Asia low 0.5
    # London 2026-07-15 (closes 2026-07-15 15:00 UTC — LATER) prints high 2.0, low 0.1.
    # London's low is placed WELL BELOW the Asia-break path (0.3-0.4) so the break
    # bars pierce the Asia low without ever touching London's range — London stays quiet.
    rows += _london_session_day("2026-07-15", hi=2.0, lo=0.1)
    # After both close: BREAK the old Asia low (0.5), staying above London's low 0.1.
    rows.append(_bar("2026-07-16 08:00", 0.6, 0.4, 0.45))  # close below Asia low, above London low
    rows.append(_bar("2026-07-16 09:00", 0.55, 0.3, 0.42))  # holds below -> Asia break
    rows += _flat_hours("2026-07-16", [10, 11], 0.45)      # London range never pierced
    df = _frame(rows)
    alert = pd.Timestamp("2026-07-16 13:00")
    prior = df[df.index < alert]

    # London is the most-recent completed session and had NO event -> 'none'. The
    # older Asia break must NOT surface.
    out = sl.build_session_level_event(prior, alert, ref_price=0.45)
    assert out == sl.features_none(), (
        "recency pick must report the most-recent (London, quiet) session as 'none', "
        f"never reach back to the older Asia break — got {out}")


def test_both_sides_fired_reports_side_nearer_to_entry():
    """When the most-recent session has an event on BOTH its high and its low, the
    side NEARER to entry is reported (the ONLY place ref_price is used — a pure
    within-session tiebreak, never a cross-history search)."""
    # One London session with high 2.0 and low 1.0; both get swept after it closes.
    rows = _london_session_day("2026-07-15", hi=2.0, lo=1.0)
    rows.append(_bar("2026-07-16 08:00", 2.3, 1.6, 1.9))   # wick over high 2.0, close back -> sweep high
    rows.append(_bar("2026-07-16 09:00", 1.4, 0.8, 1.1))   # wick under low 1.0, close back -> sweep low
    rows += _flat_hours("2026-07-16", [10, 11], 1.5)
    df = _frame(rows)
    alert = pd.Timestamp("2026-07-16 13:00")
    prior = df[df.index < alert]

    # Entry near the HIGH -> high reported.
    near_high = sl.build_session_level_event(prior, alert, ref_price=1.95)
    assert near_high["session_level_side"] == "high", near_high
    assert near_high["session_level_event"] == "sweep", near_high

    # Entry near the LOW -> low reported (same frame, tiebreak flips).
    near_low = sl.build_session_level_event(prior, alert, ref_price=1.05)
    assert near_low["session_level_side"] == "low", near_low
    assert near_low["session_level_event"] == "sweep", near_low


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
