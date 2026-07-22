"""Guard: the LIVE Phase-1 session-H/L badge (smc_detector._session_hl_until) is
DST-honest.

Failure mode this kills: the helper used to key off a frozen-UTC window
(SESSION_WINDOWS_UTC London (7,12)), so for ~half the year it measured the London
session H/L over the WRONG bars — the badge said 'london_high' / 'london_low' about
a level that was not actually the London extreme. Fixed 2026-07-22 to resolve each
candle into the session's OWN timezone per candle date (SESSION_WINDOWS_LOCAL), so
the UTC edge shifts with BST/GMT.

This test pins the SAME local-clock spike in summer and winter and asserts the
helper picks the DST-shifted UTC bar in each — a frozen-UTC regression picks the
same UTC bar in both seasons and goes red here.

Out-of-band by design (offline test, never in the live alert path) per CLAUDE.md.
"""

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import smc_detector as sd  # noqa: E402


def _frame(rows):
    """rows = [(utc_ts, high, low)]. _session_hl_until reads via _row_timestamps,
    which prefers a 'Datetime' column (Phase-1 reset_index shape) then the index."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    return pd.DataFrame(
        {"Open": [(r[1] + r[2]) / 2 for r in rows],
         "High": [r[1] for r in rows],
         "Low": [r[2] for r in rows],
         "Close": [(r[1] + r[2]) / 2 for r in rows]},
        index=idx,
    )


def _london_day(date_str, spike_utc_hour, spike_hi, spike_lo, base=1.5):
    """A full UTC day of flat bars at `base`, with one spike bar at
    `spike_utc_hour`. Anchor is set late in the day so the whole session is visible."""
    rows = []
    for h in range(0, 24):
        if h == spike_utc_hour:
            rows.append((f"{date_str} {h:02d}:00", spike_hi, spike_lo))
        else:
            rows.append((f"{date_str} {h:02d}:00", base + 1e-6, base - 1e-6))
    return rows


def test_london_hl_uses_dst_shifted_utc_bar():
    """London is 08:00-16:00 LOCAL. Summer (BST=UTC+1): 08:00 local = 07:00 UTC.
    Winter (GMT): 08:00 local = 08:00 UTC. Put the day's high on the 07:00-UTC bar:
    in summer that bar IS the London open (counted -> high == spike); in winter that
    bar is 07:00 London = BEFORE the session (excluded -> high != spike)."""
    SPIKE_HI, SPIKE_LO = 9.0, 0.5

    # SUMMER — 07:00 UTC == 08:00 BST -> inside London -> spike counted.
    summer = _frame(_london_day("2026-07-15", 7, SPIKE_HI, SPIKE_LO))
    anchor_s = pd.Timestamp("2026-07-15 16:00")  # after London close (15:00 UTC summer)
    hi_s, lo_s = sd._session_hl_until(summer, anchor_s, "london")
    assert hi_s == SPIKE_HI, f"summer: 07:00 UTC must be in London (08:00 BST), got hi={hi_s}"

    # WINTER — 07:00 UTC == 07:00 GMT -> BEFORE London 08:00 -> spike excluded.
    winter = _frame(_london_day("2026-01-15", 7, SPIKE_HI, SPIKE_LO))
    anchor_w = pd.Timestamp("2026-01-15 17:00")  # after London close (16:00 UTC winter)
    hi_w, lo_w = sd._session_hl_until(winter, anchor_w, "london")
    assert hi_w is not None
    assert hi_w != SPIKE_HI, (
        f"winter: 07:00 UTC must be OUTSIDE London (07:00 GMT, before 08:00 open), "
        f"but the spike was counted (hi={hi_w}) — DST not resolved per candle.")


def test_london_close_edge_shifts_with_dst():
    """Mirror at the close. London closes 16:00 LOCAL: 15:00 UTC summer, 16:00 UTC
    winter. Put the high on the 15:00-UTC bar: summer 15:00 UTC = 16:00 BST = window
    end (EXCLUSIVE) -> excluded; winter 15:00 UTC = 15:00 GMT = inside -> counted."""
    SPIKE_HI, SPIKE_LO = 9.0, 0.5

    summer = _frame(_london_day("2026-07-15", 15, SPIKE_HI, SPIKE_LO))
    hi_s, _ = sd._session_hl_until(summer, pd.Timestamp("2026-07-15 20:00"), "london")
    assert hi_s != SPIKE_HI, "summer: 15:00 UTC = 16:00 BST = window end (exclusive), must be OUT"

    winter = _frame(_london_day("2026-01-15", 15, SPIKE_HI, SPIKE_LO))
    hi_w, _ = sd._session_hl_until(winter, pd.Timestamp("2026-01-15 20:00"), "london")
    assert hi_w == SPIKE_HI, "winter: 15:00 UTC = 15:00 GMT = inside London, must be counted"


def test_no_future_leak():
    """The helper clips to the anchor: a spike AFTER the anchor is never counted."""
    SPIKE_HI, SPIKE_LO = 9.0, 0.5
    # London bar at 10:00 UTC (inside session both seasons) but anchor is 09:00 UTC.
    rows = _london_day("2026-07-15", 10, SPIKE_HI, SPIKE_LO)
    df = _frame(rows)
    hi, _ = sd._session_hl_until(df, pd.Timestamp("2026-07-15 09:00"), "london")
    # 09:00 UTC = 10:00 BST: London is open, but the 10:00-UTC spike is in the future.
    assert hi != SPIKE_HI, "a bar after the anchor must not be counted (future leak)"


def test_pair_session_tags_cover_all_config_pairs():
    """Every instrument in config.json must appear in PAIR_SESSION_TAGS, else the
    backtest relevance flag silently marks that pair's sessions all-irrelevant."""
    import json
    cfg = json.load(open(_ROOT / "config.json"))
    pairs = [p["name"] for p in cfg["pairs"]]
    missing = [p for p in pairs if p not in sd.PAIR_SESSION_TAGS]
    assert not missing, f"pairs absent from PAIR_SESSION_TAGS: {missing}"
    # Every tagged session must be a real session key.
    valid = set(sd.SESSION_WINDOWS_LOCAL)
    for pair, sessions in sd.PAIR_SESSION_TAGS.items():
        bad = [s for s in sessions if s not in valid]
        assert not bad, f"{pair} has unknown session(s) {bad}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
