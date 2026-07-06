"""Regression guard: Phase-1 sweep/FVG timestamps must resolve to real dates,
never integer row numbers.

Bug class (fixed 2026-07-06): smc_radar builds its H1 frame with
`reset_index()` WITHOUT drop=True, moving the DatetimeIndex into a 'Datetime'
column and leaving an integer RangeIndex. Detection code that read
`df.index[k]` expecting a timestamp instead stamped the ROW NUMBER (e.g. 122)
onto the sweep / swept-swing / FVG ISO fields. Phase 2 then failed to resolve
those overlays onto its own separately-fetched chart df — sweep/FVG markers
dropped or landed on the wrong candle in the trade email.

This test drives the real Phase-1 code path (observe_phase1_sweep on a
reset-index df) and asserts every emitted timestamp is an ISO date string, not
a bare integer. If someone reintroduces a raw `df.index[k]` timestamp read on
the Phase-1 frame, this goes red.
"""

import pandas as pd
import pytest

import smc_detector as d


def _phase1_style_df(rows):
    """Build a df shaped exactly like smc_radar's: UTC 'Datetime' COLUMN +
    integer RangeIndex (the reset_index(no drop) shape)."""
    idx = pd.date_range("2026-07-05 18:00", periods=len(rows), freq="h", tz="UTC")
    df = pd.DataFrame(rows, index=idx,
                      columns=["Open", "High", "Low", "Close"])
    df["Volume"] = 0.0
    df.index.name = "Datetime"
    return df.reset_index()  # -> 'Datetime' column, integer index (Phase 1 shape)


def test_sweep_timestamps_are_dates_not_row_numbers():
    # A pivot LOW (idx7, 1.0990), a recovery, then a SWEEP candle (idx13) that
    # dips just below the pivot and closes back above, then displacement up to
    # an OB anchor (idx15). impulse_start=8 keeps both the pivot and the sweep
    # inside observe_phase1_sweep's leg-anchored search window. Verified to fire
    # a real sweep — so the timestamp guard below is actually exercised.
    rows = [
        (1.1080, 1.1090, 1.1070, 1.1075),  # 0
        (1.1075, 1.1080, 1.1060, 1.1065),  # 1
        (1.1065, 1.1070, 1.1050, 1.1055),  # 2
        (1.1055, 1.1060, 1.1040, 1.1045),  # 3
        (1.1045, 1.1050, 1.1020, 1.1025),  # 4
        (1.1025, 1.1030, 1.1000, 1.1005),  # 5
        (1.1005, 1.1010, 1.0995, 1.1000),  # 6
        (1.1000, 1.1005, 1.0990, 1.0995),  # 7  PIVOT LOW (1.0990)
        (1.0995, 1.1015, 1.0993, 1.1010),  # 8  up (impulse_start passed here)
        (1.1010, 1.1030, 1.1008, 1.1025),  # 9  recovery
        (1.1025, 1.1030, 1.1015, 1.1020),  # 10
        (1.1020, 1.1025, 1.1005, 1.1010),  # 11 back down
        (1.1010, 1.1012, 1.1000, 1.1005),  # 12
        (1.1005, 1.1008, 1.0985, 1.1006),  # 13 SWEEP: 1.0985 < 1.0990, closes above
        (1.1006, 1.1050, 1.1004, 1.1045),  # 14 impulse
        (1.1045, 1.1090, 1.1040, 1.1085),  # 15 OB anchor / break
        (1.1085, 1.1110, 1.1080, 1.1105),  # 16
    ]
    df = _phase1_style_df(rows)
    atr = 0.0025

    obs = d.observe_phase1_sweep(
        df, ob_idx=15, impulse_start_idx=8, direction="bullish",
        tf_atr=atr, pair_type="forex", pair_name="EURUSD",
        tf_label="H1", event_type="BOS", prior_event_idx=0,
    )

    # The core guard: whatever timestamps are emitted, none may be an integer
    # row number. Skip the assertion body only if no sweep was observed at all
    # (that would be a fixture problem, not the bug we guard) — but a resolved
    # sweep MUST carry ISO dates.
    if not obs.get("exists"):
        pytest.skip("fixture produced no sweep; timestamp guard not exercised")

    for field in ("timestamp", "swept_swing_ts"):
        val = obs.get(field)
        if val is None:
            continue
        assert not _looks_like_row_number(val), (
            f"{field}={val!r} is a bare row number — reset_index poisoning "
            f"regressed (df.index[k] read on a Phase-1 frame)."
        )
        # Positive proof it's a real date, not just 'not an int'.
        assert str(val).startswith("2026-07-05") or str(val).startswith("2026-07-06"), (
            f"{field}={val!r} is not the expected ISO date."
        )


def test_iso_and_ts_helpers_read_datetime_column():
    """Unit-level guard on the two shared resolvers directly."""
    df = _phase1_style_df([
        (1.10, 1.11, 1.09, 1.105),
        (1.105, 1.12, 1.10, 1.115),
        (1.115, 1.13, 1.11, 1.125),
        (1.125, 1.14, 1.12, 1.135),
    ])
    # Raw index is integers — the trap.
    assert df.index[2] == 2
    # Helpers must return the real timestamp instead.
    assert d._iso_for_idx(df, 2) == "2026-07-05T20:00:00+00:00"
    assert str(d._ts_for_idx(df, 2)) == "2026-07-05 20:00:00+00:00"


def _looks_like_row_number(val):
    """True if val is (or stringifies to) a small bare integer like 3 / '3'."""
    if isinstance(val, bool):
        return False
    if isinstance(val, int):
        return True
    s = str(val).strip()
    return s.isdigit()  # '122', '3' — a date string never passes .isdigit()
