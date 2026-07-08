"""Guard: session labels (session / ob_session / fill_session) are DST-aware.

Failure mode this kills: the session columns were bucketed on fixed UTC hours,
so a trade at a London/NY session boundary got the WRONG label for half the year
(the EDT/EST clock shift moves the real session edge +/-1h vs UTC). This test
pins two timestamps that are the SAME UTC hour but fall in DIFFERENT NY-local
sessions across the DST change; a fixed-UTC regression labels them identically
and goes red here.

Out-of-band by design (offline test, never in the live alert path) per the
CLAUDE.md guard rules.
"""

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest import h1_only_simulator as sim


def test_ny_boundary_flips_with_dst():
    """13:00 UTC sits at the London->NY NY-local boundary. In summer (EDT,
    UTC-4) 13:00 UTC = 09:00 NY = NY session. In winter (EST, UTC-5) 13:00 UTC
    = 08:00 NY = still NY (08:00 is the NY open). Use 12:00 UTC to catch the
    flip: summer 12:00 UTC = 08:00 NY = NY; winter 12:00 UTC = 07:00 NY =
    London. Fixed-UTC bucketing would call BOTH the same."""
    summer = sim._ts_hour_ny("2020-07-01T12:00:00+00:00")  # EDT: 08:00 NY
    winter = sim._ts_hour_ny("2020-01-01T12:00:00+00:00")  # EST: 07:00 NY
    assert summer == 8
    assert winter == 7
    assert sim._session_from_ny_hour(summer) == "NY"
    assert sim._session_from_ny_hour(winter) == "London"


def test_session_buckets_cover_the_clock():
    """Every NY-local hour maps to a real label; no gaps, Asia wraps midnight."""
    labels = {sim._session_from_ny_hour(h) for h in range(24)}
    assert labels == {"Asia", "London", "NY", "Other"}
    # Asia wrap: 23:00 and 00:00 NY are both Asia.
    assert sim._session_from_ny_hour(23) == "Asia"
    assert sim._session_from_ny_hour(0) == "Asia"
    assert sim._session_from_ny_hour(1) == "Asia"


def test_ob_and_fill_session_use_ny_dst():
    """The OB/fill helpers route through the same DST-aware path, not raw UTC."""
    ob = {"ob_timestamp": "2020-07-01T12:00:00+00:00"}  # 08:00 NY summer
    assert sim._ob_session(ob) == "NY"
    # winter, same UTC hour -> London
    ob_w = {"ob_timestamp": "2020-01-01T12:00:00+00:00"}  # 07:00 NY winter
    assert sim._ob_session(ob_w) == "London"


def test_missing_ts_is_unknown_not_crash():
    assert sim._ob_session({}) == "unknown"
    assert sim._fill_session(None, None) == "unknown"


def test_naive_ts_treated_as_utc():
    """A naive (tz-less) timestamp is read as UTC then converted to NY."""
    h = sim._ts_hour_ny(pd.Timestamp("2020-07-01T12:00:00"))
    assert h == 8  # 12:00 UTC -> 08:00 EDT
