"""Pool all backtest run folders into one combined dataset and produce a verdict.

Usage:
  python backtest/aggregate_runs.py                    # all completed runs
  python backtest/aggregate_runs.py --groups 1,2       # specific groups only
  python backtest/aggregate_runs.py --entry proximal   # proximal (only zone)

Outputs (written to backtest/results/combined/):
  all_trades.csv      combined raw trades
  insights.json       all computed metrics
  VERDICT.md          plain-English verdict — the document to read before trading
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
RESULTS_DIR  = _HERE / "results"
REGISTRY_PATH = _HERE / "registry.json"
COMBINED_DIR = RESULTS_DIR / "combined"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest import insights as ins


def _load_registry() -> List[Dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return []
    with open(REGISTRY_PATH) as f:
        return json.load(f).get("runs", [])


def _load_trades(run_dir: Path, run_meta: Dict) -> Optional[pd.DataFrame]:
    """Load trades.csv from a run folder. Returns None if not found."""
    for name in ("trades.csv",):
        p = run_dir / name
        if p.exists():
            try:
                df = pd.read_csv(p)
                # Attach cross-run metadata so insights can group across runs.
                df["run_id"]   = run_meta["run_id"]
                df["group"]    = run_meta.get("group", 0)
                df["regime"]   = run_meta.get("regime", "unspecified")
                df["condition"] = run_meta.get("condition", "")
                return df
            except Exception as e:
                print(f"  [skip] {run_dir.name}: could not read trades — {e}")
                return None
    # Fallback: try forex_trades.xlsx (legacy Phase 2 runs)
    for name in ("forex_trades.xlsx", "nas_xau_trades.xlsx"):
        p = run_dir / name
        if p.exists():
            try:
                df = pd.read_excel(p, sheet_name="Trades")
                df["run_id"]    = run_meta["run_id"]
                df["group"]     = run_meta.get("group", 0)
                df["regime"]    = run_meta.get("regime", "unspecified")
                df["condition"] = run_meta.get("condition", "")
                return df
            except Exception:
                continue
    return None


def _build_verdict_md(
    verdict: Dict,
    overall: Dict,
    instr_v: Dict,
    pair_sess: List,
    score_v: Dict,
    group_comp: Dict,
    entry_comp: Dict,
    freshness: List,
    confluence: Dict,
    regime_v: Dict,
    n_runs: int,
    groups_included: List[int],
    r_col: str,
) -> str:

    def _fmt(v, unit="R"):
        if v is None:
            return "n/a"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.2f}{unit}"

    def _wr(v):
        # Win rate is None when no trade resolved (all breakevens). Show an
        # em-dash, never 0% — a breakeven-only group has no accuracy to report.
        return "—" if v is None else f"{v:.1f}%"

    lines = [
        "# Backtest Verdict",
        "",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        f"*Runs included: {n_runs} | Groups: {groups_included} | Entry view: {r_col}*",
        "",
    ]

    # --- Top-line verdict ---
    v_colour = {"GREEN": "✅", "YELLOW": "⚠️", "RED": "❌"}.get(verdict["overall"], "?")
    lines += [
        f"## Overall verdict: {v_colour} {verdict['overall']}",
        "",
    ]

    if verdict["issues"]:
        lines += ["**Issues to resolve before trading:**", ""]
        for issue in verdict["issues"]:
            lines.append(f"- {issue}")
        lines.append("")
    else:
        lines += ["No blocking issues. See live-eligible cells below.", ""]

    if verdict["live_cells"]:
        lines += ["**Live-eligible pair × session cells:**", ""]
        for cell in verdict["live_cells"]:
            lines.append(f"- {cell}")
        lines.append("")
    else:
        lines += ["No cells have 20+ trades, positive expectancy, and CI excluding zero yet.", ""]

    lines += ["---", ""]

    # --- Overall stats ---
    n = overall.get("n", 0)
    ci_lo = overall.get("ci_lo_95")
    ci_hi = overall.get("ci_hi_95")
    ci_str = f"[{_fmt(ci_lo)}, {_fmt(ci_hi)}]" if ci_lo is not None else "n/a (< 5 trades)"
    lines += [
        "## Overall statistics",
        "",
        f"| Metric | Value | Threshold (Forex) |",
        f"|--------|-------|------------------|",
        f"| Filled trades | {n} | Target: 200+ |",
        f"| Win rate | {_wr(overall.get('win_rate_pct'))} | ≥ 45% |",
        f"| Expectancy | {_fmt(overall.get('expectancy_r'))} | ≥ +0.3R |",
        f"| 95% CI on expectancy | {ci_str} | Lower bound > 0 |",
        f"| CI excludes zero? | {'Yes' if overall.get('ci_excludes_zero') else 'No'} | Must be Yes |",
        f"| Sharpe | {overall.get('sharpe', 0):.2f} | > 0.5 |",
        f"| Max drawdown | {_fmt(overall.get('max_dd_r'))} | ≤ 6R |",
        f"| Longest losing streak | {overall.get('longest_losing_streak', 0)} trades | ≤ 7 |",
        f"| Win capture | {overall.get('win_capture_pct', 0):.0f}% of available move | ≥ 75% |",
        "",
    ]

    # --- Instrument-specific ---
    if instr_v:
        lines += ["## Instrument breakdown (separate thresholds)", ""]
        for instr, iv in instr_v.items():
            t = iv["thresholds"]
            v_tag = {"GREEN": "✅", "YELLOW": "⚠️", "RED": "❌"}.get(iv["verdict"], "?")
            lines += [
                f"### {instr.upper()} {v_tag}",
                f"*{iv['note']}*",
                "",
                f"| Metric | Value | Threshold |",
                f"|--------|-------|-----------|",
                f"| Trades | {iv.get('n', 0)} | — |",
                f"| Win rate | {_wr(iv.get('win_rate_pct'))} | ≥ {t['green']['win_rate_pct']}% |",
                f"| Expectancy | {_fmt(iv.get('expectancy_r'))} | ≥ {_fmt(t['green']['expectancy_r'])} |",
                f"| CI excludes zero? | {'Yes' if iv.get('ci_excludes_zero') else 'No'} | Must be Yes |",
                "",
            ]

    # --- Group consistency ---
    if group_comp:
        group_labels = {1: "Group 1 — Study", 2: "Group 2 — Out-of-sample", 3: "Group 3 — Live-era"}
        lines += [
            "## Performance by group",
            "",
            "| Group | Trades | Win rate | Expectancy | CI | Max DD |",
            "|-------|--------|----------|------------|-----|--------|",
        ]
        for g, s in sorted(group_comp.items()):
            ci = f"[{_fmt(s.get('ci_lo_95'))}, {_fmt(s.get('ci_hi_95'))}]" if s.get("ci_lo_95") is not None else "n/a"
            label = group_labels.get(g, f"Group {g}")
            lines.append(
                f"| {label} | {s['n']} | {_wr(s['win_rate_pct'])} "
                f"| {_fmt(s['expectancy_r'])} | {ci} | {_fmt(s['max_dd_r'])} |"
            )
        lines += ["", "*All three groups should be positive. Decline Group 1 → 3 means era-dependency.*", ""]

    # --- Pair × session ---
    if pair_sess:
        lines += [
            "## Pair × session heatmap",
            "",
            "🟢 = 20+ trades (statistically meaningful) | 🟡 = 10–19 | 🔴 = < 10",
            "",
            "| Pair | Session | Trades | Win rate | Expectancy | CI | Live eligible? |",
            "|------|---------|--------|----------|------------|-----|----------------|",
        ]
        for row in pair_sess:
            badge = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(row["confidence"], "")
            ci = f"[{_fmt(row.get('ci_lo_95'))}, {_fmt(row.get('ci_hi_95'))}]" if row.get("ci_lo_95") is not None else "n/a"
            eligible = "✅ Yes" if row.get("live_eligible") else "No"
            lines.append(
                f"| {row['pair']} | {row['session']} | {badge} {row['n']} "
                f"| {_wr(row['win_rate_pct'])} | {_fmt(row['expectancy_r'])} | {ci} | {eligible} |"
            )
        lines.append("")

    # --- Score validation ---
    sv = score_v
    lines += [
        "## Score validation",
        "",
        f"**Verdict: {sv.get('verdict', 'n/a')}**",
        "",
    ]
    if sv.get("spearman_r") is not None:
        lines.append(f"Spearman correlation between score and outcome: {sv['spearman_r']:.3f} (closer to +1 = stronger relationship)")
    lines.append("")
    if sv.get("buckets"):
        lines += [
            "| Score bucket | Trades | Win rate | Expectancy |",
            "|-------------|--------|----------|------------|",
        ]
        for b in sv["buckets"]:
            lines.append(
                f"| {b['bucket']} | {b['n']} | {_wr(b['win_rate_pct'])} | {_fmt(b['expectancy_r'])} |"
            )
        lines.append("")

    # --- Confluence attribution ---
    if confluence:
        lines += [
            "## Confluence attribution",
            "",
            "Does each confluence actually improve trade outcomes?",
            "",
            "| Confluence | With (n) | Without (n) | Expectancy with | Expectancy without | Uplift | Verdict |",
            "|------------|----------|-------------|-----------------|-------------------|--------|---------|",
        ]
        for name, c in confluence.items():
            lines.append(
                f"| {name} | {c['n_with']} | {c['n_without']} "
                f"| {_fmt(c['exp_with'])} | {_fmt(c['exp_without'])} "
                f"| {_fmt(c.get('uplift_r'))} | {c['verdict']} |"
            )
        lines.append("")

    # --- Entry zone comparison ---
    if entry_comp:
        lines += [
            "## Entry zone (proximal only)",
            "",
            "| Entry zone | Fills | Fill rate | Win rate | Expectancy | CI |",
            "|------------|-------|-----------|----------|------------|-----|",
        ]
        for zone, ez in entry_comp.items():
            ci = f"[{_fmt(ez.get('ci_lo_95'))}, {_fmt(ez.get('ci_hi_95'))}]" if ez.get("ci_lo_95") is not None else "n/a"
            fill = f"{ez.get('fill_rate_pct', 0):.1f}%" if ez.get("fill_rate_pct") is not None else "n/a"
            lines.append(
                f"| {zone} | {ez['n']} | {fill} "
                f"| {_wr(ez['win_rate_pct'])} | {_fmt(ez['expectancy_r'])} | {ci} |"
            )
        lines.append("")

    # --- OB freshness by touch count ---
    if freshness:
        lines += [
            "## OB freshness (win/loss by touch count)",
            "",
            "Touch number of the OB at fire time. Fresh = 1st approach. The engine "
            "kills a zone on the 3rd touch, so 3 is the deepest a trade can fire on. "
            "Empty buckets mean the system does not trade re-touched OBs.",
            "",
            "| Touch | Trades | Wins | Losses | BE | Win rate | Expectancy |",
            "|-------|--------|------|--------|----|----------|------------|",
        ]
        for fr in freshness:
            lines.append(
                f"| {fr['label']} | {fr['n']} | {fr['wins']} | {fr['losses']} "
                f"| {fr['breakevens']} | {_wr(fr['win_rate_pct'])} "
                f"| {_fmt(fr['expectancy_r'])} |"
            )
        lines.append("")

    # --- Regime verification ---
    flagged = {rid: rv for rid, rv in regime_v.items() if rv.get("regime_flag")}
    if flagged:
        lines += [
            "## Regime label warnings",
            "",
            "The following runs showed behaviour inconsistent with their pre-assigned BAU label.",
            "Verify against actual ATR data before drawing conclusions from those weeks.",
            "",
        ]
        for rid, rv in flagged.items():
            lines.append(f"- **{rid}:** {rv['regime_flag']}")
        lines.append("")

    # --- Known limitations ---
    lines += [
        "---",
        "",
        "## Known limitations",
        "",
        "1. No spread, slippage, or swap cost modelled. Real P&L ~5–10% lower.",
        "2. Same-bar SL+TP collision resolves SL-first (pessimistic assumption).",
        "3. H1 bar resolution only — entry and exit are approximate.",
        "4. MT5/FundingPips bars may differ from your broker's bars.",
        "5. Proximal entry only (the live model); the 50% leg is retired.",
        "6. Regime labels (BAU/war) are based on known calendar events, not verified ATR data.",
        "",
    ]

    return "\n".join(lines)


def run(
    groups: Optional[List[int]] = None,
    entry_zone: str = "proximal",
    r_col: str = "r_realised",
) -> None:
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    registry = _load_registry()
    if not registry:
        print("No runs in registry. Run update_registry.py first.")
        return

    frames: List[pd.DataFrame] = []
    included_runs = 0

    for entry in registry:
        run_id = entry["run_id"]
        g = entry.get("group", 0)
        if groups and g not in groups:
            continue
        run_dir = RESULTS_DIR / run_id
        if not run_dir.exists():
            print(f"  [missing dir] {run_id} — skipping")
            continue
        df = _load_trades(run_dir, entry)
        if df is None or df.empty:
            print(f"  [no trades] {run_id} — skipping")
            continue
        frames.append(df)
        included_runs += 1
        print(f"  [loaded] {run_id} — {len(df)} rows (group {g})")

    if not frames:
        print("No trade data found. Run the backtest schedule first.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(COMBINED_DIR / "all_trades.csv", index=False)
    print(f"\n  Combined: {len(combined)} total rows from {included_runs} runs")

    # Filter to the requested entry zone for the primary analysis view.
    if "entry_zone" in combined.columns:
        primary = combined[combined["entry_zone"] == entry_zone].copy()
        print(f"  Primary view: entry_zone={entry_zone}, {len(primary)} rows after filter")
    else:
        primary = combined.copy()

    filled = ins._filled(primary)
    print(f"  Filled trades: {len(filled)}")

    groups_included = sorted(combined["group"].unique().tolist()) if "group" in combined.columns else []

    # Compute all metrics.
    overall     = ins.compute_overall(filled, r_col)
    instr_v     = ins.instrument_verdicts(filled, r_col)
    pair_sess   = ins.pair_session_matrix(filled, r_col)
    score_v     = ins.score_validation(filled, r_col)
    group_comp  = ins.group_comparison(filled, r_col)
    entry_comp  = ins.entry_zone_comparison(primary, r_col)  # uses full primary (incl never_filled for fill rate)
    freshness   = ins.ob_freshness_comparison(filled, r_col)
    confluence  = ins.confluence_attribution(filled, r_col)
    regime_v    = ins.regime_verification(filled)
    verdict     = ins.generate_verdict(overall, instr_v, score_v, pair_sess, group_comp)

    # Write machine-readable output.
    all_metrics = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "runs_included":     included_runs,
        "groups_included":   groups_included,
        "entry_zone_filter": entry_zone,
        "r_col":             r_col,
        "verdict":           verdict,
        "overall":           overall,
        "instrument":        instr_v,
        "pair_session":      pair_sess,
        "score":             score_v,
        "groups":            group_comp,
        "entry_zones":       entry_comp,
        "ob_freshness":      freshness,
        "confluence":        confluence,
        "regime_checks":     regime_v,
    }
    with open(COMBINED_DIR / "insights.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Write human-readable verdict.
    verdict_md = _build_verdict_md(
        verdict, overall, instr_v, pair_sess, score_v,
        group_comp, entry_comp, freshness, confluence, regime_v,
        included_runs, groups_included, r_col,
    )
    (COMBINED_DIR / "VERDICT.md").write_text(verdict_md, encoding="utf-8")

    print(f"\n  ✅ Verdict: {verdict['overall']}")
    if verdict["issues"]:
        print("  Issues:")
        for issue in verdict["issues"]:
            print(f"    - {issue}")
    print(f"\n  Files written to {COMBINED_DIR}/")
    print("    all_trades.csv")
    print("    insights.json")
    print("    VERDICT.md  ← read this")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=None,
                    help="Comma-separated group numbers to include, e.g. 1,2. Default: all.")
    ap.add_argument("--entry", default="proximal", choices=["proximal"],
                    help="Which entry zone to use as the primary analysis view. "
                         "Proximal is the only zone (50% mean entry removed 2026-07).")
    ap.add_argument("--r-col", default="r_realised",
                    choices=["r_realised", "r_if_exit_tp1", "r_if_exit_tp2"],
                    help="Which R column to use as the outcome measure.")
    args = ap.parse_args()

    groups = [int(g.strip()) for g in args.groups.split(",")] if args.groups else None
    run(groups=groups, entry_zone=args.entry, r_col=args.r_col)


if __name__ == "__main__":
    main()
