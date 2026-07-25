"""Regression guard: the shared timestamp resolvers must read the 'Datetime'
COLUMN, never the integer RangeIndex.

Bug class (fixed 2026-07-06): smc_radar builds its H1 frame with
`reset_index()` WITHOUT drop=True, moving the DatetimeIndex into a 'Datetime'
column and leaving an integer RangeIndex. Detection code that read
`df.index[k]` expecting a timestamp instead stamped the ROW NUMBER (e.g. 122)
onto emitted ISO fields. Phase 2 then failed to resolve those overlays onto its
own separately-fetched chart df — markers dropped or landed on the wrong candle.

The v1 sweep detector (observe_phase1_sweep) that carried the original sweep
half of this guard was retired 2026-07-24 — sweep v2 (liquidity_sweep.py) is
structurally immune to the bug: it rebuilds a DatetimeIndex via _naive_utc_index
BEFORE reading any position, and returns snapshot_failed() if the index is not a
DatetimeIndex, so it can never stamp a row number. What survives here is the
unit-level guard on the two shared resolvers Phase 2 / chart rendering still use.
"""

import pandas as pd

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
