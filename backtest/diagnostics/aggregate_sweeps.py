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


# Pair -> bucket. Read once from config so a results.jsonl row (which carries
# only the pair NAME, never pair_type — the schema is frozen) can be bucketed
# without touching the frozen result schema. forex pairs pool together; the lone
# index (NAS100) and commodity (GOLD) each stand alone but are reported under
# their own asset-class heading, which is the conclusion the trader asked for.
_BUCKET_OF_TYPE = {"forex": "Forex", "index": "Index", "commodity": "Commodity"}


def _pair_buckets(root: Optional[Path] = None) -> Dict[str, str]:
    """Map pair name -> asset-class bucket from config.json. Unknown pairs fall
    back to 'Other' so a config drift surfaces in the table rather than silently
    dropping rows."""
    cfg_path = (root or _REPO_ROOT) / "config.json"
    try:
        import json
        cfg = json.load(open(cfg_path))
    except Exception:
        return {}
    return {p["name"]: _BUCKET_OF_TYPE.get(p.get("pair_type", "forex"), "Other")
            for p in cfg.get("pairs", [])}


def _regime_for(start: str, end: str) -> str:
    """Tag a run's month as war/bau using the SAME detector the backtest uses,
    so the corpus splits consistently with everything else."""
    try:
        from backtest import regime_detector
        regime, _label = regime_detector.detect_regime(start, end)
        return regime or "unknown"
    except Exception:
        return "unknown"


def _month_expectancy_by_value(rows: List[Dict[str, Any]],
                               keep_pairs: Optional[set] = None
                               ) -> Dict[Any, Dict[str, float]]:
    """Pool one month's rows across pairs, per grid value. Filled-weighted
    expectancy (same rule as the email) plus the filled count behind it. When
    `keep_pairs` is given, only those pairs are pooled — this is how the
    per-bucket (Forex/Index/Commodity) view reuses the SAME pooling rule as the
    headline, no second implementation."""
    agg: Dict[Any, Dict[str, float]] = {}
    for r in rows:
        if keep_pairs is not None and r.get("pair") not in keep_pairs:
            continue
        v = r.get("grid_value")
        a = agg.setdefault(v, {"filled": 0, "sum_r": 0.0, "baseline": False})
        a["filled"] += int(r.get("n_trades_filled") or 0)
        a["sum_r"] += float(r.get("sum_r_realised") or 0)
        a["baseline"] = a["baseline"] or bool(r.get("baseline"))
    for a in agg.values():
        a["expectancy_r"] = (a["sum_r"] / a["filled"]) if a["filled"] else 0.0
    return agg


def _verdict_from_month_deltas(months: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """Collapse a list of per-month {delta: {value: Δexp}} into the per-value
    distribution verdict (win-fraction, median/mean Δ, even-month OOS holdout).
    Factored out so the headline and every bucket share ONE definition."""
    per_value: Dict[Any, List[float]] = defaultdict(list)
    per_value_oos: Dict[Any, List[float]] = defaultdict(list)
    for m in months:
        for v, d in m["delta"].items():
            per_value[v].append(d)
            if int(m.get("month") or 0) % 2 == 0:
                per_value_oos[v].append(d)
    verdict: Dict[Any, Dict[str, Any]] = {}
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
    return verdict


def aggregate(knob: str, *, regime: Optional[str] = None,
              root: Optional[Path] = None) -> Dict[str, Any]:
    """Pool every PASSed run for `knob`. Optionally restrict to one regime.

    Returns a structured verdict: per-value month-by-month Δexpectancy vs that
    month's baseline, win-fraction, median Δ, and a held-out (even-month) check.
    """
    knob = knob.upper()
    buckets = _pair_buckets(root)              # pair name -> Forex/Index/Commodity
    bucket_names = sorted(set(buckets.values()))
    # pairs grouped by bucket, for the per-bucket pooling pass
    pairs_in_bucket = {b: {p for p, bk in buckets.items() if bk == b}
                       for b in bucket_names}

    months: List[Dict[str, Any]] = []
    bucket_months: Dict[str, List[Dict[str, Any]]] = {b: [] for b in bucket_names}
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
        # Same month, re-pooled within each bucket. Each bucket gets its OWN
        # baseline expectancy (the bucket at the live value), so a bucket's Δ is
        # honest about that asset class, not the all-pairs baseline.
        for b in bucket_names:
            bv = _month_expectancy_by_value(rows, keep_pairs=pairs_in_bucket[b])
            if not bv:
                continue
            b_base_v = next((v for v, a in bv.items() if a["baseline"]), base_v)
            b_base_exp = bv.get(b_base_v, {}).get("expectancy_r", 0.0)
            bucket_months[b].append({
                "year": manifest.get("year"), "month": manifest.get("month"),
                "delta": {v: round(a["expectancy_r"] - b_base_exp, 4) for v, a in bv.items()},
                "filled": {v: a["filled"] for v, a in bv.items()},
            })

    verdict = _verdict_from_month_deltas(months)
    by_bucket = {b: _verdict_from_month_deltas(bucket_months[b])
                 for b in bucket_names if bucket_months[b]}

    return {"knob": knob, "regime": regime or "all", "n_months": len(months),
            "months": months, "per_value": verdict, "by_bucket": by_bucket}


def _verdict_table(lines: List[str], per_value: Dict[Any, Dict[str, Any]]) -> None:
    """Append the standard per-value verdict table to `lines`. One definition,
    used by the headline and every bucket so they always read the same way."""
    lines.append("| value | months | win_frac | median Δexp | mean Δexp | "
                 "OOS median Δ (even mo.) | OOS n |")
    lines.append("|---|---|---|---|---|---|---|")
    for v in sorted(per_value, key=lambda x: (isinstance(x, str), x)):
        s = per_value[v]
        lines.append(f"| {v} | {s['n_months']} | {s['win_fraction']} | "
                     f"{s['median_delta_r']} | {s['mean_delta_r']} | "
                     f"{s['oos_median_delta_r']} | {s['oos_n_months']} |")


def render(agg: Dict[str, Any]) -> str:
    lines = [f"# Two-year aggregate — {agg['knob']} (regime={agg['regime']})",
             f"- months pooled: {agg['n_months']}", ""]
    if agg["n_months"] == 0:
        lines.append("_No PASSed runs yet for this knob/regime. "
                     "The corpus is empty — run monthly sweeps to fill it._")
        return "\n".join(lines)
    _verdict_table(lines, agg["per_value"])

    # Per-bucket break-out: the asset-class conclusion. The best MIN_LEG value
    # for noisy indices/gold can differ from forex — this is where that shows.
    by_bucket = agg.get("by_bucket") or {}
    if by_bucket:
        lines += ["", "## By asset class",
                  "_Same months, re-pooled within each class against that class's "
                  "own current setting. The best value can differ by class — forex "
                  "is calmer than gold or the index._", ""]
        for b in sorted(by_bucket):
            lines.append(f"### {b}")
            _verdict_table(lines, by_bucket[b])
            lines.append("")
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


def _md_to_html(md: str) -> str:
    """Tiny markdown->HTML for the roll-up email: headings, GitHub-style pipe
    tables, list items, italics. Deliberately minimal — no dependency, and the
    aggregate text is the only thing it ever has to render."""
    out: List[str] = []
    in_table = False

    def close_table():
        nonlocal in_table
        if in_table:
            out.append("</table>")
            in_table = False

    cell = "padding:5px 9px;border:1px solid #ddd;text-align:right;"
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("|") and set(line.replace("|", "").strip()) <= {"-", ":", " "}:
            continue  # the |---|---| separator row
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not in_table:
                out.append("<table style='border-collapse:collapse;"
                           "font-family:Arial,sans-serif;font-size:13px;margin:6px 0;'>")
                in_table = True
                tag, style = "th", cell + "background:#f0f0f0;"
            else:
                tag, style = "td", cell
            out.append("<tr>" + "".join(
                f"<{tag} style='{style}'>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        close_table()
        if line.startswith("### "):
            out.append(f"<h3 style='margin:12px 0 4px;'>{line[4:]}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2 style='margin:16px 0 4px;'>{line[3:]}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1 style='font-size:20px;margin:0 0 6px;'>{line[2:]}</h1>")
        elif line.startswith("- "):
            out.append(f"<div style='margin:2px 0 2px 10px;'>• {line[2:].replace('**','')}</div>")
        elif line.startswith("_") and line.endswith("_") and len(line) > 1:
            out.append(f"<div style='color:#555;font-size:13px;margin:4px 0;'>"
                       f"{line.strip('_')}</div>")
        elif line.startswith("**") and line.endswith("**"):
            out.append(f"<div style='font-weight:bold;margin:8px 0 2px;'>{line.strip('*')}</div>")
        elif line:
            out.append(f"<div style='margin:2px 0;'>{line.replace('**','')}</div>")
    close_table()
    return ("<div style='font-family:Arial,sans-serif;max-width:820px;color:#222;'>"
            + "".join(out) + "</div>")


def send_aggregate_email(knob: str, *, regime: Optional[str] = None,
                         root: Optional[Path] = None) -> bool:
    """Build the cross-month verdict for `knob` and send ONE roll-up email — what
    the workflow's final job calls after all monthly cells land, instead of 11
    per-month emails. Reuses sweep_email's SMTP sender (one send path)."""
    knob = knob.upper()
    agg = aggregate(knob, regime=regime, root=root)
    html = _md_to_html(render(agg))
    from backtest.diagnostics.sweep_email import _smtp_send
    subject = f"Knob Sweep — {knob} — {agg['n_months']}-month aggregate"
    return _smtp_send(subject, html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", required=True)
    ap.add_argument("--regime", choices=["war", "bau"], default=None)
    ap.add_argument("--email", action="store_true",
                    help="send ONE roll-up email instead of printing markdown")
    args = ap.parse_args()
    if args.email:
        ok = send_aggregate_email(args.knob, regime=args.regime)
        print(f"[aggregate-email] {'sent' if ok else 'NOT sent'}")
        return
    agg = aggregate(args.knob, regime=args.regime)
    print(render(agg))


if __name__ == "__main__":
    main()
