"""Harness 1 â ATR-knob sweep.

For ONE knob, holding everything else at live defaults, sweep a grid of values
and show how structure counts, alert counts, and r_realised P&L move. Slice-mode
B throughout (alerts and P&L only make sense bar-by-bar).

The knob-scope registry (FABLE_REFERENCE Â§1a/Â§6) is baked in as BOTH annotation
(invariant columns get an "IDENTICAL BY DESIGN" note so they read as truth, not
a broken sweep) AND assertion (if a supposedly-invariant column moves, the sweep
refuses the pretty table and escalates to Harness 3).

Run:
  python -m backtest.diagnostics.h1_knob_sweep --knob MIN_LEG_ATR_MULT --grid 1.0,1.5,2.0 --pairs EURUSD --start 2026-03-01 --end 2026-03-31
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.diagnostics import driver

# Knob name -> (KnobOverrides kwarg, is_dict, scope). scope: columns that may
# move vs columns asserted invariant. Columns: swings, obs, choch, bos,
# alerts, trades, pnl, score.
ALL_COLS = ["n_swings", "n_obs", "n_choch", "n_bos",
            "n_alerts_total", "n_trades_filled", "sum_r_realised", "avg_logged_score"]

KNOB_SCOPE: Dict[str, Dict[str, Any]] = {
    "MIN_LEG_ATR_MULT": dict(kw="min_leg_atr_mult", is_dict=False,
                             invariant=[]),
    "BOS_ATR_MULT": dict(kw="bos_atr_mult", is_dict=False,
                         invariant=["n_swings"]),
    "STRUCTURE_CHOCH_ATR_MULT": dict(kw="structure_choch_atr_mult", is_dict=False,
                                     invariant=["n_swings"], soft=["n_bos"]),
    "STRUCTURE_LOCK_ATR_MULT": dict(kw="structure_lock_atr_mult", is_dict=False,
                                    invariant=["n_swings"]),
    "OB_MAX_RANGE_ATR_MULT": dict(kw="ob_max_range_atr_mult", is_dict=False,
                                  invariant=["n_swings", "n_choch", "n_bos"]),
    "FVG_NOISE_FLOOR_MULT": dict(kw="fvg_noise_floor_mult", is_dict=True,
                                 invariant=["n_swings", "n_obs", "n_choch", "n_bos",
                                            "n_alerts_total", "n_trades_filled",
                                            "sum_r_realised"]),
    "SWEEP_EQUAL_LEVEL_TOLERANCE_ATR": dict(kw="sweep_equal_level_tolerance_atr", is_dict=True,
                                            invariant=["n_swings", "n_obs", "n_choch", "n_bos",
                                                       "n_alerts_total", "n_trades_filled",
                                                       "sum_r_realised"]),
    "SWEEP_WICK_PIERCE_MIN_ATR": dict(kw="sweep_wick_pierce_min_atr", is_dict=True,
                                      invariant=["n_swings", "n_obs", "n_choch", "n_bos",
                                                 "n_alerts_total", "n_trades_filled",
                                                 "sum_r_realised"]),
    "PROXIMITY_CAP": dict(kw="proximity_cap", is_dict=True,
                          invariant=["n_swings", "n_obs", "n_choch", "n_bos"]),
}

# Current live default for each knob (for baseline auto-insert).
LIVE_DEFAULT: Dict[str, Any] = {
    "MIN_LEG_ATR_MULT": 1.5, "BOS_ATR_MULT": 0.4,
    "STRUCTURE_CHOCH_ATR_MULT": 1.0, "STRUCTURE_LOCK_ATR_MULT": 1.5,
    "OB_MAX_RANGE_ATR_MULT": 2.0,
    "FVG_NOISE_FLOOR_MULT": {"forex": 0.08, "index": 0.15, "commodity": 0.12},
    "SWEEP_EQUAL_LEVEL_TOLERANCE_ATR": {"forex": 0.30, "index": 0.40, "commodity": 0.40},
    "SWEEP_WICK_PIERCE_MIN_ATR": {"forex": 0.05, "index": 0.08, "commodity": 0.08},
}

REFUSED = {"REARM_EXTRA_ATR"}


def _ts(v):
    if v is None or v == "":
        return None
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def _structure_counts(pair_conf, df, start, end, *, stride: int) -> Dict[str, int]:
    """Union BOS/CHoCH events across a stride detection walk (the ring holds
    only the last 20, so we must walk to count them all). Auto-densify a gap if
    it shows >=18 new events (ring overflow risk)."""
    events: Set[Tuple[str, str]] = set()
    obs: Set[Tuple[Any, Any]] = set()
    last_n = 0
    last_snap_swings = 0
    prev_count = 0
    snaps = list(driver.walk_detection(pair_conf, df, start, end, stride=stride))
    for i, snap in enumerate(snaps):
        new_here = 0
        for e in snap.events_tail:
            key = (e.type, e.candle_ts)
            if key not in events:
                events.add(key)
                new_here += 1
        for ob in snap.active_zones:
            obs.add((ob.ob_timestamp, ob.direction))
        if new_here >= 18 and stride > 1:
            # Ring may have overflowed between snapshots; densify this gap.
            lo = snaps[i - 1].wall_clock_ts if i > 0 else snap.wall_clock_ts
            for s2 in driver.walk_detection(pair_conf, df, lo, snap.wall_clock_ts, stride=1):
                for e in s2.events_tail:
                    events.add((e.type, e.candle_ts))
        last_snap_swings = len(snap.swings) if snap.swings else last_snap_swings
    n_bos = sum(1 for t, _ in events if t == "BOS")
    n_choch = sum(1 for t, _ in events if t == "CHoCH")
    return {"n_swings": last_snap_swings, "n_bos": n_bos,
            "n_choch": n_choch, "n_obs": len(obs)}


def _alert_pnl(pair_conf, df, start, end, risk_usd, overrides) -> Dict[str, Any]:
    res = driver.walk_alerts(pair_conf, df, start, end, risk_usd=risk_usd,
                             overrides=overrides)
    filled = [r for r in res.trade_rows
              if r.get("exit_reason") not in (None, "never_filled")]
    sum_r = round(sum(float(r.get("r_realised") or 0) for r in filled), 4)
    sum_pnl = round(sum(float(r.get("pnl_usd") or 0) for r in filled), 2)
    wins = sum(1 for r in filled if (r.get("r_realised") or 0) > 0)
    wr = round(wins / len(filled), 4) if filled else 0.0
    exp = round(sum_r / len(filled), 4) if filled else 0.0
    scores = [float(r.get("score") or 0) for r in res.trade_rows]
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    # trade-set identity (for score-knob invariance assertion)
    tset = frozenset((r.get("pair"), r.get("ob_timestamp"), r.get("direction"),
                      r.get("alert_ts")) for r in res.trade_rows)
    return {"n_alerts_total": res.counters["alerts_total"],
            "n_alerts_simulated": res.counters["alerts_simulated"],
            "n_trades_filled": len(filled),
            "sum_r_realised": sum_r, "sum_pnl_usd": sum_pnl,
            "win_rate": wr, "expectancy_r": exp, "avg_logged_score": avg_score,
            "trade_set": tset, "recon_ok": abs(sum_pnl - sum_r * risk_usd) <= 0.01}


def sweep(knob: str, grid: List[Any], pairs: List[str], start, end, out_dir: Path,
          *, risk_usd: float = 250.0, grid_mode: str = "multiplier",
          stride: int = 6) -> Path:
    if knob in REFUSED:
        raise SystemExit(
            f"{knob} (knob #9) cannot be overridden without editing live code. "
            "Proposed (NOT applied) fix: hoist REARM_EXTRA_ATR to a module "
            "constant in replay_engine read at call time. See the Harness-3 report.")
    if knob not in KNOB_SCOPE:
        raise SystemExit(f"unknown knob {knob!r}; choose one of {sorted(KNOB_SCOPE)}")
    scope = KNOB_SCOPE[knob]
    is_dict = scope["is_dict"]
    invariant = set(scope["invariant"])

    cfg = json.load(open(_REPO_ROOT / "config.json"))
    confs = [p for p in cfg["pairs"] if (pairs == ["all"] or p["name"] in pairs)]

    # Ensure the baseline (live default) value is in the grid for non-dict knobs.
    baseline_val = LIVE_DEFAULT.get(knob)
    if not is_dict and baseline_val is not None and baseline_val not in grid:
        grid = sorted(set(grid) | {baseline_val})

    rows: List[Dict[str, Any]] = []
    scope_violations: List[str] = []

    for pc in confs:
        name = pc["name"]
        pt = pc.get("pair_type", "forex")
        print(f"\n=== {name} | sweeping {knob} over {grid} ===", flush=True)
        df = driver.load_window(pc, start, end)
        if isinstance(df, driver.WindowUnserveable):
            print(f"  [SKIP] {df}", flush=True)
            continue

        per_value: Dict[Any, Dict[str, Any]] = {}
        for gv in grid:
            ov_val = _resolve_value(knob, gv, pt, is_dict, grid_mode)
            kwargs = {scope["kw"]: ov_val}
            print(f"  value={gv} ...", flush=True)
            with driver.KnobOverrides(**kwargs) as ovr:
                sc = _structure_counts(pc, df, start, end, stride=stride)
                ap = _alert_pnl(pc, df, start, end, risk_usd, ovr)
            is_base = (not is_dict and baseline_val is not None
                       and abs(float(gv) - float(baseline_val)) < 1e-12)
            rec = {"knob": knob, "grid_value": gv, "baseline": is_base,
                   "grid_mode": grid_mode, "pair": name, **sc, **ap}
            if not ap["recon_ok"]:
                scope_violations.append(f"{name} v={gv}: P&L reconciliation failed")
            per_value[gv] = rec
            rows.append(rec)

        # Invariance assertion across grid for this pair.
        base_row = next((r for r in per_value.values() if r["baseline"]),
                        next(iter(per_value.values()), None))
        if base_row:
            for col in invariant:
                vals = {r[col] for r in per_value.values()}
                if len(vals) > 1:
                    scope_violations.append(
                        f"SCOPE VIOLATION: {name} column {col} moved under {knob}: {vals} "
                        f"(either scope classification is wrong or there is leakage â escalate to Harness 3)")
            # Score knobs: trade-set must be identical across grid.
            if invariant >= {"n_alerts_total"}:
                tsets = {r["trade_set"] for r in per_value.values()}
                if len(tsets) > 1:
                    scope_violations.append(
                        f"SCOPE VIOLATION: {name} trade-set changed under score knob {knob}")

    return _write(knob, grid, rows, scope_violations, invariant, scope, out_dir,
                  meta={"pairs": ",".join(p["name"] for p in confs),
                        "start": str(pd.Timestamp(start).date()),
                        "end": str(pd.Timestamp(end).date()),
                        "grid_mode": grid_mode, "risk_usd": risk_usd})


def _resolve_value(knob, gv, pair_type, is_dict, grid_mode):
    if not is_dict:
        if knob == "PROXIMITY_CAP":
            return {pair_type: float(gv)}
        return float(gv)
    base = LIVE_DEFAULT[knob]
    if grid_mode == "absolute":
        return {k: float(gv) for k in base}
    return {k: float(v) * float(gv) for k, v in base.items()}  # multiplier


def _write(knob, grid, rows, violations, invariant, scope, out_dir, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    runstamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
    csv_path = out_dir / f"h1_sweep_{knob}_{runstamp}.csv"
    md_path = out_dir / f"h1_sweep_{knob}_{runstamp}.md"

    csv_cols = ["knob", "grid_value", "baseline", "grid_mode", "pair",
                "n_swings", "n_obs", "n_choch", "n_bos",
                "n_alerts_total", "n_alerts_simulated", "n_trades_filled",
                "sum_r_realised", "sum_pnl_usd", "win_rate", "expectancy_r",
                "avg_logged_score", "recon_ok"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = [f"# Harness 1 â knob sweep: {knob}", "",
             f"- pairs: {meta['pairs']} | window: {meta['start']}..{meta['end']} "
             f"| grid_mode: {meta['grid_mode']} | slice_mode: **B**",
             f"- non-swept knobs frozen at live defaults; risk_usd={meta['risk_usd']}", ""]
    if invariant:
        lines.append(f"Columns IDENTICAL BY DESIGN under this knob "
                     f"(see FABLE_REFERENCE Â§1a/Â§6): **{', '.join(sorted(invariant))}**.")
    if scope.get("soft"):
        lines.append(f"Soft expectation (informational, not asserted): "
                     f"{', '.join(scope['soft'])}.")
    if knob in ("FVG_NOISE_FLOOR_MULT", "SWEEP_EQUAL_LEVEL_TOLERANCE_ATR",
                "SWEEP_WICK_PIERCE_MIN_ATR"):
        lines.append("> Score knobs are **P&L-neutral** in the current no-gate "
                     "H1-only backtest â only `avg_logged_score` may move. They "
                     "would bite only if a score gate were enabled.")
    lines.append("")
    if violations:
        lines.append("## âš ï¸ SCOPE / RECON VIOLATIONS")
        for v in violations:
            lines.append(f"- {v}")
        lines.append("")
    # one table per pair
    pairs_seen = []
    for r in rows:
        if r["pair"] not in pairs_seen:
            pairs_seen.append(r["pair"])
    for p in pairs_seen:
        lines.append(f"### {p}")
        lines.append("| value | base | swings | OBs | CHoCH | BOS | alerts | filled | sumR | $PnL | WR | exp_R | avg_score |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in [x for x in rows if x["pair"] == p]:
            lines.append(f"| {r['grid_value']} | {'âœ“' if r['baseline'] else ''} | "
                         f"{r['n_swings']} | {r['n_obs']} | {r['n_choch']} | {r['n_bos']} | "
                         f"{r['n_alerts_total']} | {r['n_trades_filled']} | {r['sum_r_realised']} | "
                         f"{r['sum_pnl_usd']} | {r['win_rate']} | {r['expectancy_r']} | {r['avg_logged_score']} |")
        lines.append("")
    lines.append("## Honest weaknesses")
    lines.append("- One knob at a time: interaction effects are NOT explored; a best "
                 "value here does not compose into a best joint config.")
    lines.append("- Conditional on the ~720-day yfinance window â one regime sample. "
                 "In-sample; diagnostic, not tuning truth.")
    lines.append("- `n_swings` is a window-end census of survivors, not a per-bar experience.")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[h1] csv -> {csv_path}\n[h1] md  -> {md_path}", flush=True)
    if violations:
        print(f"[h1] {len(violations)} SCOPE/RECON VIOLATION(S) â see report", flush=True)
    return md_path


def self_check() -> bool:
    """Baseline-identity: a sweep at the knob's live value equals a no-override
    run. Verified inside driver.self_check already; here we confirm the registry
    is internally consistent."""
    for k, sc in KNOB_SCOPE.items():
        assert set(sc["invariant"]).issubset(set(ALL_COLS)), f"bad invariant for {k}"
    return True


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", required=True)
    ap.add_argument("--grid", required=True, help="comma list of floats")
    ap.add_argument("--pairs", default="EURUSD")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--grid-mode", default="multiplier", choices=["multiplier", "absolute"])
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--risk-usd", type=float, default=250.0)
    ap.add_argument("--stride", type=int, default=6,
                    help="detection-walk stride for structure counts")
    args = ap.parse_args()
    self_check()
    grid = [float(x) for x in args.grid.split(",") if x.strip()]
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    sweep(args.knob.upper(), grid, pairs, args.start, args.end, Path(args.out),
          risk_usd=args.risk_usd, grid_mode=args.grid_mode, stride=args.stride)


if __name__ == "__main__":
    main()
