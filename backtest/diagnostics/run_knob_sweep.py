"""Knob-sweep orchestrator — one knob, one calendar month, one email.

This is what the GitHub workflow calls. It does NOT re-implement detection: it
drives backtest.diagnostics.h1_knob_sweep.sweep(), which drives the LIVE system
through backtest.diagnostics.driver. Zero duplicated trading logic.

Flow (manifest-first, gated):
  1. month + year  -> REAL calendar dates (July=31) via sweep_logging.month_bounds
  2. open SweepRun  -> writes manifest.json BEFORE any compute
  3. run sweep()    -> with a walk_sink that streams the bar-by-bar causal story
                       into walk.jsonl.gz, and return_rows so we get the metrics
  4. write results.jsonl (decision metrics only) + finalize run_health.json
  5. email ONLY on PASS (no recon failure, no scope violation, rows present)

CLI (what the workflow runs):
  python -m backtest.diagnostics.run_knob_sweep \
      --knob BOS_ATR_MULT --year 2024 --month 7 \
      --pairs EURUSD,USDJPY,NAS100,GOLD --email
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.diagnostics import h1_knob_sweep as hk
from backtest.diagnostics import sweep_logging as sl
from backtest.diagnostics import driver

# ---------------------------------------------------------------------------
# Baked-in grids — FROZEN per knob so two years of monthly runs pool cleanly.
# The grid is NOT a workflow input: if it could change per run, rows wouldn't
# line up across months and the aggregator could not pool them. Each grid
# brackets the live default (which sweep() also auto-inserts as the baseline
# row). Values are multipliers for dict knobs, absolutes for scalar knobs,
# matching h1_knob_sweep._resolve_value semantics.
# ---------------------------------------------------------------------------
KNOB_GRID: Dict[str, List[float]] = {
    "MIN_LEG_ATR_MULT":               [1.0, 1.25, 1.5, 1.75, 2.0],
    "BOS_ATR_MULT":                   [0.2, 0.3, 0.4, 0.5, 0.7],
    "STRUCTURE_CHOCH_ATR_MULT":       [0.5, 0.75, 1.0, 1.25, 1.5],
    "STRUCTURE_LOCK_ATR_MULT":        [1.0, 1.25, 1.5, 1.75, 2.0],
    "OB_MAX_RANGE_ATR_MULT":          [1.5, 1.75, 2.0, 2.5, 3.0],
    "MIN_OB_RANGE_ATR_MULT":          [0.0, 0.1, 0.2, 0.3],
    # dict knobs -> grid is a MULTIPLIER applied to the live per-type defaults.
    "FVG_NOISE_FLOOR_MULT":           [0.5, 0.75, 1.0, 1.25, 1.5],
    "SWEEP_EQUAL_LEVEL_TOLERANCE_ATR": [0.5, 0.75, 1.0, 1.25, 1.5],
    "SWEEP_WICK_PIERCE_MIN_ATR":      [0.5, 0.75, 1.0, 1.25, 1.5],
}

DEFAULT_PAIRS = ["EURUSD", "USDJPY", "NAS100", "GOLD"]


def _grid_mode_for(knob: str) -> str:
    """Dict knobs are swept as multipliers of their per-type defaults; scalar
    knobs as absolute values. Mirrors h1_knob_sweep._resolve_value."""
    return "multiplier" if hk.KNOB_SCOPE.get(knob, {}).get("is_dict") else "absolute"


def _live_knob_snapshot() -> Dict[str, Any]:
    """Record every swept knob's live default at run time, straight from the
    engine's LIVE_DEFAULT table (which mirrors the live module constants). This
    is what makes a run interpretable two years later — you know the baseline
    the grid was measured against."""
    return dict(hk.LIVE_DEFAULT)


def _pairs_served_manifest(pairs: List[str], start: str, end: str) -> List[Dict[str, Any]]:
    """Load each pair's window once, fingerprint it, and record served vs
    requested. Done before the sweep so the manifest pins exactly what data the
    run saw (the determinism anchor). The sweep reloads internally; this small
    duplicate load is the price of a self-contained manifest."""
    cfg = json.load(open(_REPO_ROOT / "config.json"))
    confs = [p for p in cfg["pairs"] if p["name"] in pairs]
    served = []
    for pc in confs:
        df = driver.load_window(pc, start, end)
        if isinstance(df, driver.WindowUnserveable):
            served.append({"name": pc["name"], "symbol": pc.get("symbol"),
                           "requested_start": start, "requested_end": end,
                           "served": False, "reason": str(df)})
            continue
        in_win = df.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")]
        served.append({
            "name": pc["name"], "symbol": pc.get("symbol"),
            "requested_start": start, "requested_end": end,
            "served": True,
            "served_start": str(in_win.index.min()) if not in_win.empty else None,
            "served_end": str(in_win.index.max()) if not in_win.empty else None,
            "n_bars": int(len(in_win)),
            "fingerprint": sl.frame_fingerprint(df),
            "prox_cap_atr": pc.get("atr_multiplier"),
        })
    return served


def run(knob: str, year: int, month: int, pairs: List[str],
        *, risk_usd: float = 250.0, send_email: bool = False,
        root: Path = None) -> Dict[str, Any]:
    knob = knob.upper()
    if knob not in KNOB_GRID:
        raise SystemExit(f"unknown/ungridd knob {knob!r}; have {sorted(KNOB_GRID)}")
    grid = KNOB_GRID[knob]
    grid_mode = _grid_mode_for(knob)
    start, end = sl.month_bounds(year, month)

    print(f"[sweep] {knob} {year}-{month:02d} ({start}..{end}) "
          f"grid={grid} mode={grid_mode} pairs={pairs}", flush=True)

    pairs_served = _pairs_served_manifest(pairs, start, end)

    # `with` is load-bearing: if the sweep aborts (exception, or a catchable
    # signal), SweepRun.__exit__ still writes a FAIL run_health.json so the dir
    # is never left with a manifest but no health — that ambiguity is the exact
    # silent-failure mode this whole design guards against. (A hard SIGKILL,
    # e.g. an external `timeout`, cannot be caught by anything — but then the
    # absence of BOTH health and a clean exit is itself the detectable signal.)
    with sl.SweepRun.begin(
            knob, year, month, grid=grid, grid_mode=grid_mode, pairs=pairs,
            live_knob_snapshot=_live_knob_snapshot(),
            pairs_served=pairs_served, risk_usd=risk_usd, root=root) as run_obj:

        # walk_sink: stream the causal walk story per (pair x grid_value) into
        # the gzipped walk log. No extra compute — these records come from the
        # walk the sweep already ran.
        def walk_sink(pair: str, grid_value: float, records: List[Dict[str, Any]]):
            for rec in (records or []):
                run_obj.write_walk_record({"pair": pair, "grid_value": grid_value, **rec})

        md_path, rows, scope_violations = hk.sweep(
            knob, grid, pairs, start, end, run_obj.run_dir / "harness_md",
            risk_usd=risk_usd, grid_mode=grid_mode,
            walk_sink=walk_sink, return_rows=True)
        for r in rows:
            run_obj.write_result(r)
        run_obj.add_scope_violations(scope_violations)
        health = run_obj.finalize()

    print(f"[sweep] health: {health['overall']} "
          f"(rows={health['n_result_rows']}, walk={health['n_walk_records']}, "
          f"recon_ok={health['recon_ok']}, scope_ok={health['scope_ok']})", flush=True)

    if send_email:
        if health["overall"] == "PASS":
            from backtest.diagnostics import sweep_email
            sweep_email.send_for_run(run_obj.run_dir)
        else:
            from backtest.diagnostics import sweep_email
            sweep_email.send_failure_notice(run_obj.run_dir, health)

    return health


def _parse_periods(spec: str) -> List[tuple[int, int]]:
    """Parse a comma list of 'YYYY-MM' into [(year, month), ...].

    One knob run is always ONE calendar month (so months pool cleanly in the
    aggregator). Selecting several months across years just means several runs,
    each its own immutable dir + email. Order preserved, exact dupes dropped."""
    out: List[tuple[int, int]] = []
    seen = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            y_s, m_s = tok.split("-")
            y, m = int(y_s), int(m_s)
        except ValueError:
            raise SystemExit(f"bad period {tok!r}; want 'YYYY-MM' (e.g. 2024-07)")
        if not (1 <= m <= 12):
            raise SystemExit(f"bad month in period {tok!r}; month must be 1..12")
        if (y, m) not in seen:
            seen.add((y, m))
            out.append((y, m))
    if not out:
        raise SystemExit("no valid periods parsed")
    return out


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", required=True)
    ap.add_argument("--year", type=int)
    ap.add_argument("--month", type=int)
    ap.add_argument("--periods", help="comma list of YYYY-MM (e.g. 2024-07,2025-01); "
                                      "runs one sweep per month. Overrides --year/--month.")
    ap.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    ap.add_argument("--risk-usd", type=float, default=250.0)
    ap.add_argument("--email", action="store_true")
    args = ap.parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    if args.periods:
        periods = _parse_periods(args.periods)
    elif args.year and args.month:
        periods = [(args.year, args.month)]
    else:
        raise SystemExit("give either --periods 'YYYY-MM,...' or both --year and --month")

    # Each month is an independent run. One FAIL must not abort the rest, but the
    # process exits non-zero if ANY month failed so the caller surfaces it loudly.
    all_pass = True
    for (y, m) in periods:
        health = run(args.knob, y, m, pairs,
                     risk_usd=args.risk_usd, send_email=args.email)
        all_pass = all_pass and (health.get("overall") == "PASS")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
