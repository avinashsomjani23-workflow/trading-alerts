"""Two-year sweep aggregator — turns a corpus of monthly runs into a verdict.

The per-knob email is deliberately humble: ONE month, in-sample, no action. The
POWER lives here. This reads every PASSed run for a knob across all months and
asks the questions a single month cannot answer:

  * Distribution, not a point: "value X beats the baseline in 14 of 22 months",
    median Δexpectancy, sign-consistency — not "+0.18R in March".
  * Regime split: does the edge hold in BAU and WAR, or only one? (regime is
    looked up per run via the existing WAR_REGIME_WEEKS detector.)
  * Out-of-sample discipline: an odd/even month split so a value that wins
    in-sample can be checked on held-out months.

It reads ONLY results.jsonl + manifest + run_health (one parser, the same source
of truth as the email). It recomputes NOTHING about detection. Built now, even
though it has little to chew on until months accumulate — the schema and the
pooling contract must exist from day one so the corpus is mined, not stranded.

CLI:
  python -m backtest.diagnostics.aggregate_sweeps --knob BOS_ATR_MULT
  python -m backtest.diagnostics.aggregate_sweeps --knob BOS_ATR_MULT --regime bau
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.diagnostics import sweep_logging as sl


def _regime_for(start: str, end: str) -> str:
    """Tag a run's month as war/bau using the SAME detector the backtest uses,
    so the corpus splits consistently with everything else."""
    try:
        from backtest import regime_detector
        regime, _label = regime_detector.detect_regime(start, end)
        return regime or "unknown"
    except Exception:
        return "unknown"


def _month_expectancy_by_value(rows: List[Dict[str, Any]]) -> Dict[Any, Dict[str, float]]:
    """Pool one month's rows across pairs, per grid value. Filled-weighted
    expectancy (same rule as the email) plus the filled count behind it."""
    agg: Dict[Any, Dict[str, float]] = {}
    for r in rows:
        v = r.get("grid_value")
        a = agg.setdefault(v, {"filled": 0, "sum_r": 0.0, "baseline": False})
        a["filled"] += int(r.get("n_trades_filled") or 0)
        a["sum_r"] += float(r.get("sum_r_realised") or 0)
        a["baseline"] = a["baseline"] or bool(r.get("baseline"))
    for a in agg.values():
        a["expectancy_r"] = (a["sum_r"] / a["filled"]) if a["filled"] else 0.0
    return agg


def aggregate(knob: str, *, regime: Optional[str] = None,
              root: Optional[Path] = None) -> Dict[str, Any]:
    """Pool every PASSed run for `knob`. Optionally restrict to one regime.

    Returns a structured verdict: per-value month-by-month Δexpectancy vs that
    month's baseline, win-fraction, median Δ, and a held-out (even-month) check.
    """
    knob = knob.upper()
    months: List[Dict[str, Any]] = []
    for run_dir in sl.iter_run_dirs(knob, root=root):
        health = sl.read_health(run_dir)
        if health.get("overall") != "PASS":
            continue  # never aggregate an untrustworthy run
        manifest = sl.read_manifest(run_dir)
        start, end = manifest.get("resolved_start"), manifest.get("resolved_end")
        run_regime = _regime_for(start, end)
        if regime and run_regime != regime:
            continue
        rows = sl.read_results(run_dir)
        by_v = _month_expectancy_by_value(rows)
        base_v = next((v for v, a in by_v.items() if a["baseline"]), None)
        base_exp = by_v.get(base_v, {}).get("expectancy_r", 0.0)
        months.append({
            "run_id": manifest.get("run_id"),
            "year": manifest.get("year"), "month": manifest.get("month"),
            "regime": run_regime, "base_v": base_v,
            "delta": {v: round(a["expectancy_r"] - base_exp, 4) for v, a in by_v.items()},
            "filled": {v: a["filled"] for v, a in by_v.items()},
        })

    # Per-value distribution across months.
    per_value: Dict[Any, List[float]] = defaultdict(list)
    per_value_oos: Dict[Any, List[float]] = defaultdict(list)  # even-month holdout
    for m in months:
        for v, d in m["delta"].items():
            per_value[v].append(d)
            if int(m.get("month") or 0) % 2 == 0:
                per_value_oos[v].append(d)

    verdict = {}
    for v, deltas in per_value.items():
        positive = sum(1 for d in deltas if d > 0)
        verdict[v] = {
            "n_months": len(deltas),
            "win_fraction": round(positive / len(deltas), 3) if deltas else None,
            "median_delta_r": round(statistics.median(deltas), 4) if deltas else None,
            "mean_delta_r": round(statistics.fmean(deltas), 4) if deltas else None,
            "oos_median_delta_r": (round(statistics.median(per_value_oos[v]), 4)
                                   if per_value_oos.get(v) else None),
            "oos_n_months": len(per_value_oos.get(v, [])),
        }

    return {"knob": knob, "regime": regime or "all", "n_months": len(months),
            "months": months, "per_value": verdict}


def render(agg: Dict[str, Any]) -> str:
    lines = [f"# Two-year aggregate — {agg['knob']} (regime={agg['regime']})",
             f"- months pooled: {agg['n_months']}", ""]
    if agg["n_months"] == 0:
        lines.append("_No PASSed runs yet for this knob/regime. "
                     "The corpus is empty — run monthly sweeps to fill it._")
        return "\n".join(lines)
    lines.append("| value | months | win_frac | median Δexp | mean Δexp | "
                 "OOS median Δ (even mo.) | OOS n |")
    lines.append("|---|---|---|---|---|---|---|")
    for v in sorted(agg["per_value"], key=lambda x: (isinstance(x, str), x)):
        s = agg["per_value"][v]
        lines.append(f"| {v} | {s['n_months']} | {s['win_fraction']} | "
                     f"{s['median_delta_r']} | {s['mean_delta_r']} | "
                     f"{s['oos_median_delta_r']} | {s['oos_n_months']} |")
    lines += ["",
              "**How to read this (cautious by design):**",
              "- `win_frac` = share of months where the value beat THAT month's "
              "baseline. A real edge is consistent (≳0.6 across many months), not "
              "a single fat month.",
              "- `median Δexp` resists one outlier month; prefer it to the mean.",
              "- `OOS median` is the even-month holdout. If a value looks great "
              "overall but its OOS column collapses, it is in-sample luck.",
              "- Still one knob at a time; joint tuning is a separate exercise."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", required=True)
    ap.add_argument("--regime", choices=["war", "bau"], default=None)
    args = ap.parse_args()
    agg = aggregate(args.knob, regime=args.regime)
    print(render(agg))


if __name__ == "__main__":
    main()
