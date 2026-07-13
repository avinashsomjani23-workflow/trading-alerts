"""FVG single-source guards (2026-07-13 duplication collapse).

Two structures used to be hand-copied and are now defined ONCE:

1. The missing-fvg fallback dict — was inlined in BOTH live Phase 2 and the
   backtest simulator. If those two literals ever drift apart, backtest/live
   parity breaks silently (the exact bug class zone.py exists to kill).
   Now: smc_detector.fvg_missing() is the only definition.

2. detect_fvg_in_zone's live/ghost result dicts — were hand-copied in the
   LONG and SHORT branches (4 copy sites). Now: _live_result/_ghost_result
   built once, so both branches emit the identical shape by construction.

These tests pin the exact shapes (keys AND order — the dicts are serialized
into active_obs.json, so key order is part of the byte-identity contract)
and grep the two consumer files so the inline literal cannot quietly return.
"""
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import smc_detector  # noqa: E402


# --- 1. fvg_missing: exact shape, key order, fresh object per call ----------

def test_fvg_missing_exact_shape():
    d = smc_detector.fvg_missing()
    assert d == {"exists": False, "was_detected": False, "mitigation": "none"}
    assert list(d.keys()) == ["exists", "was_detected", "mitigation"], \
        "key order changed — serialized state files would change bytes"


def test_fvg_missing_returns_fresh_dict():
    a = smc_detector.fvg_missing()
    b = smc_detector.fvg_missing()
    assert a is not b, \
        "shared dict — one caller mutating it would poison the other"


# --- 2. consumers actually use the single source -----------------------------

def _src(rel):
    return (_ROOT / rel).read_text(encoding="utf-8")


def test_no_inline_fallback_literal_in_consumers():
    for rel in ("Phase2_Alert_Engine.py", "backtest/h1_only_simulator.py"):
        src = _src(rel)
        assert '"exists": False, "was_detected": False' not in src, \
            f"{rel}: inline missing-fvg literal is back — parity drift risk"
        assert "smc_detector.fvg_missing()" in src, \
            f"{rel}: no longer uses the single-source fallback"


# --- 3. detect_fvg_in_zone result shapes (LONG vs SHORT parity) --------------

_LIVE_KEYS = ["exists", "fvg_top", "fvg_bottom", "c1_idx", "c3_idx",
              "c1_timestamp", "mitigation", "was_detected"]
_GHOST_KEYS = ["exists", "fvg_top", "fvg_bottom", "was_detected",
               "mitigation", "ghost_top", "ghost_bottom", "ghost_c1_idx",
               "ghost_c3_idx", "ghost_c1_timestamp", "mitigated_at_idx",
               "mitigated_at_iso"]


def _frame(rows):
    idx = pd.date_range("2024-03-04 00:00", periods=len(rows), freq="h",
                        tz="UTC")
    return pd.DataFrame(rows, index=idx,
                        columns=["Open", "High", "Low", "Close"])


def _long_frame(ghost=False):
    # c1 = bar2 (H=1.0010), displacement bar3, c3 = bar4 (L=1.0030)
    # -> LONG FVG ft=1.0030 fb=1.0010. ghost=True dips bar6 through fb.
    rows = [
        [1.0000, 1.0005, 0.9995, 1.0002],
        [1.0002, 1.0008, 0.9998, 1.0005],
        [1.0005, 1.0010, 1.0000, 1.0008],
        [1.0008, 1.0040, 1.0006, 1.0038],
        [1.0038, 1.0060, 1.0030, 1.0055],
        [1.0055, 1.0070, 1.0045, 1.0065],
        [1.0065, 1.0080, 1.0005 if ghost else 1.0050, 1.0075],
        [1.0075, 1.0085, 1.0055, 1.0080],
    ]
    return _frame(rows)


def _short_frame(ghost=False):
    # Mirror: c1 = bar2 (L=1.0010), c3 = bar4 (H=0.9990)
    # -> SHORT FVG ft=1.0010 fb=0.9990. ghost=True spikes bar6 through ft.
    rows = [
        [1.0020, 1.0025, 1.0015, 1.0018],
        [1.0018, 1.0022, 1.0012, 1.0015],
        [1.0015, 1.0020, 1.0010, 1.0012],
        [1.0012, 1.0014, 0.9980, 0.9982],
        [0.9982, 0.9990, 0.9960, 0.9965],
        [0.9965, 0.9975, 0.9950, 0.9955],
        [0.9955, 1.0015 if ghost else 0.9970, 0.9945, 0.9950],
        [0.9950, 0.9965, 0.9940, 0.9945],
    ]
    return _frame(rows)


def _detect(df, bias):
    return smc_detector.detect_fvg_in_zone(
        df, bias, zone_top=1.2, zone_bottom=0.9, atr_floor=0.001,
        leg_start_idx=0, leg_end_idx=7, pair_type="forex")


def test_live_result_shape_long_and_short():
    for bias, df in (("LONG", _long_frame()), ("SHORT", _short_frame())):
        r = _detect(df, bias)
        assert list(r.keys()) == _LIVE_KEYS, f"{bias} live shape drifted"
        assert r["exists"] is True and r["was_detected"] is True
        assert r["mitigation"] == "pristine"
        assert (r["c1_idx"], r["c3_idx"]) == (2, 4)
    # exact values pin the geometry once per side
    r_long = _detect(_long_frame(), "LONG")
    assert (r_long["fvg_top"], r_long["fvg_bottom"]) == (1.0030, 1.0010)
    r_short = _detect(_short_frame(), "SHORT")
    assert (r_short["fvg_top"], r_short["fvg_bottom"]) == (1.0010, 0.9990)


def test_ghost_result_shape_long_and_short():
    for bias, df in (("LONG", _long_frame(ghost=True)),
                     ("SHORT", _short_frame(ghost=True))):
        r = _detect(df, bias)
        assert list(r.keys()) == _GHOST_KEYS, f"{bias} ghost shape drifted"
        assert r["exists"] is False and r["was_detected"] is True
        assert r["mitigation"] == "full"
        assert r["mitigated_at_idx"] == 6
        assert (r["ghost_c1_idx"], r["ghost_c3_idx"]) == (2, 4)
