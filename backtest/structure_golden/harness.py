"""Core of the compute_structure golden harness (Wave 2 item 2A).

One job: given a fixture (a real H1 OHLC window + its pair), run the LIVE
structure path exactly as Phase 1 does and produce a CANONICAL, deterministic
serialization of compute_structure's full return — so the same input always
hashes to the same bytes, and any behaviour drift shows up as a readable diff.

Live path mirrored (verified against smc_radar.compute_pair_walls,
smc_radar.py:1073-1083):
    h4 = h4_range.compute_h4_range(window)
    out = dealing_range.compute_structure(window, h4)

Determinism guards:
- compute_atr is memoised on an OHLC fingerprint (smc_detector._ATR_CACHE).
  We clear it before every fixture so no cache state can leak between windows.
- ATR-derived floats carry float noise across runs. We round EVERY float to the
  fixture pair's decimal_places before serializing. Rounding is decided ONCE,
  here, and is the only lossy step. Documented so a future reader knows the
  golden bytes are rounded, not raw.

This module has NO knowledge of where fixtures come from (cache vs hand-crafted)
— it only takes an OHLC list of rows. That keeps the CI test offline: it reads
committed JSON fixtures, never the gitignored backtest/cache/ parquet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

import h4_range  # noqa: E402
import dealing_range  # noqa: E402
import smc_detector  # noqa: E402

# Fixtures live next to this module, committed to the repo.
FIXTURE_DIR = _HERE / "fixtures"

# Pair -> decimal_places, used to round float noise before hashing. Mirrors
# config.json (verified 2026-06-16). Kept local so the harness has zero runtime
# dependency on config loading or the live config path.
PAIR_DECIMALS: Dict[str, int] = {
    "EURUSD": 5,
    "USDJPY": 3,
    "NZDUSD": 5,
    "USDCHF": 5,
    "NAS100": 2,
    "GOLD": 2,
}

# The OHLC columns compute_structure / compute_h4_range read. Index is the bar
# timestamp (UTC ISO string in the fixture; restored to a tz-aware DatetimeIndex
# when the frame is rebuilt).
OHLC_COLS = ["Open", "High", "Low", "Close", "Volume"]


def window_to_rows(df: "pd.DataFrame") -> List[Dict[str, Any]]:
    """Serialize an H1 OHLC window to a plain list of row dicts for a fixture.

    Each row: {"ts": iso, "Open":..., "High":..., "Low":..., "Close":...,
    "Volume":...}. Floats are kept FULL precision here (the input must round-trip
    exactly so compute_structure sees identical numbers it saw at record time).
    Only the OUTPUT is rounded, never the input.
    """
    rows: List[Dict[str, Any]] = []
    for ts, r in df.iterrows():
        row: Dict[str, Any] = {"ts": pd.Timestamp(ts).isoformat()}
        for c in OHLC_COLS:
            v = r[c]
            row[c] = None if pd.isna(v) else float(v)
        rows.append(row)
    return rows


def rows_to_window(rows: List[Dict[str, Any]]) -> "pd.DataFrame":
    """Rebuild the exact tz-aware OHLC frame compute_structure expects."""
    idx = pd.DatetimeIndex([pd.Timestamp(r["ts"]) for r in rows], name="Datetime")
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    data = {c: [r.get(c) for r in rows] for c in OHLC_COLS}
    df = pd.DataFrame(data, index=idx)
    return df


def run_structure(df: "pd.DataFrame") -> Dict[str, Any]:
    """Run the LIVE structure path on a window, with the ATR cache cleared first.

    Mirrors compute_pair_walls exactly (h4_range then compute_structure). Returns
    compute_structure's full raw dict (un-rounded).
    """
    # Clear the memoised ATR cache so no prior fixture's fingerprint can leak in.
    smc_detector._ATR_CACHE.clear()
    h4 = h4_range.compute_h4_range(df)
    return dealing_range.compute_structure(df, h4)


def _round_floats(obj: Any, dp: int) -> Any:
    """Recursively round every float to `dp` decimals; coerce timestamps to ISO.

    This is the ONLY lossy step. It kills ATR-derived float noise so the golden
    bytes are stable run-to-run. Bools are left alone (bool is a subclass of int,
    not float). Everything else passes through json-friendly.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, dp)
    if isinstance(obj, dict):
        return {k: _round_floats(v, dp) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, dp) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def canonicalize(out: Dict[str, Any], pair: str) -> Dict[str, Any]:
    """Round + normalize compute_structure's output into a stable, hashable dict.

    Float rounding is keyed off the pair's decimal_places. The structure of the
    dict is preserved 1:1 — every field compute_structure returns is captured;
    nothing is dropped or reordered here (json.dumps with sort_keys does the
    ordering at serialize time).
    """
    if pair not in PAIR_DECIMALS:
        raise KeyError(f"unknown pair {pair!r}; add it to PAIR_DECIMALS")
    return _round_floats(out, PAIR_DECIMALS[pair])


def serialize(canonical: Dict[str, Any]) -> str:
    """Stable JSON: sorted keys, no whitespace drift. The golden bytes."""
    return json.dumps(canonical, sort_keys=True, ensure_ascii=True, indent=2)


def compute_golden(rows: List[Dict[str, Any]], pair: str) -> Dict[str, Any]:
    """Fixture rows -> canonical output dict (the value stored as the golden)."""
    df = rows_to_window(rows)
    raw = run_structure(df)
    return canonicalize(raw, pair)


# --- readable diff ----------------------------------------------------------

def diff_canonical(expected: Dict[str, Any], actual: Dict[str, Any]) -> List[str]:
    """Human-readable list of what changed between two canonical outputs.

    Walks both dicts and reports field-by-field. The event ring and swing list
    are compared element-wise so a drift names the exact event/swing index and
    field that moved — not just "events differ".
    """
    msgs: List[str] = []
    _diff_node(expected, actual, "", msgs)
    return msgs


def _diff_node(exp: Any, act: Any, path: str, msgs: List[str]) -> None:
    if isinstance(exp, dict) and isinstance(act, dict):
        for k in sorted(set(exp) | set(act)):
            p = f"{path}.{k}" if path else k
            if k not in exp:
                msgs.append(f"+ {p}: (new) {act[k]!r}")
            elif k not in act:
                msgs.append(f"- {p}: (missing) was {exp[k]!r}")
            else:
                _diff_node(exp[k], act[k], p, msgs)
        return
    if isinstance(exp, list) and isinstance(act, list):
        if len(exp) != len(act):
            msgs.append(f"~ {path}: length {len(exp)} -> {len(act)}")
        for i in range(min(len(exp), len(act))):
            _diff_node(exp[i], act[i], f"{path}[{i}]", msgs)
        return
    if exp != act:
        msgs.append(f"~ {path}: {exp!r} -> {act!r}")
