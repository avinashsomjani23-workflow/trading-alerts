"""Backtest registry builder.

Scans all result folders, extracts key metrics from summary.json and trades.csv,
and writes two files:

  backtest/registry.json   — machine-readable, used by aggregate_runs.py
  BACKTEST_LOG.md          — human-readable log for cross-run discussion

Run after every backtest:
    python backtest/update_registry.py

Or target a single run:
    python backtest/update_registry.py --run-id h1only_20240610_20240614
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.insights import win_rate_pct as _win_rate

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
RESULTS_DIR = _HERE / "results"
REGISTRY_PATH = _HERE / "registry.json"
LOG_PATH = _REPO_ROOT / "BACKTEST_LOG.md"

# Maps run prefixes or date ranges to their plan group and market context.
# Extend this as new runs are added to BACKTEST_PLAN.md.
RUN_METADATA: Dict[str, Dict[str, Any]] = {
    "20240610_20240614": {"group": 1, "regime": "bau",  "condition": "BAU moderate — ECB first rate cut"},
    "20240714_20240718": {"group": 1, "regime": "bau",  "condition": "BAU trending — no major events, mid-summer"},
    "20240819_20240823": {"group": 1, "regime": "bau",  "condition": "BAU choppy — post-Yen-carry-unwind settle"},
    "20241104_20241108": {"group": 1, "regime": "war",  "condition": "Extreme vol — US election 2024"},
    "20250210_20250214": {"group": 1, "regime": "bau",  "condition": "BAU 2025 normal — no major events"},
    "20250407_20250411": {"group": 1, "regime": "war",  "condition": "Shock — tariff Liberation Day aftermath"},
    "20250512_20250516": {"group": 2, "regime": "bau",  "condition": "BAU — post-tariff settle"},
    "20250707_20250711": {"group": 2, "regime": "bau",  "condition": "Mid-summer 2025"},
    "20250908_20250912": {"group": 2, "regime": "bau",  "condition": "Autumn start 2025"},
    "20251027_20251031": {"group": 2, "regime": "bau",  "condition": "Pre-Fed October 2025"},
    "20251117_20251121": {"group": 2, "regime": "bau",  "condition": "Late autumn 2025"},
    "20260112_20260116": {"group": 3, "regime": "bau",  "condition": "Early 2026"},
    "20260309_20260313": {"group": 3, "regime": "bau",  "condition": "Q1 2026"},
    "20260413_20260417": {"group": 3, "regime": "bau",  "condition": "Recent — April 2026"},
    "20260505_20260509": {"group": 3, "regime": "war",  "condition": "Recent active — May 2026"},
}


def _date_key(run_id: str) -> str:
    """Extract the YYYYMMDD_YYYYMMDD portion of a run_id."""
    parts = run_id.split("_")
    # run_id format: h1only_20240610_20240614 or bau_20240610_20240614
    if len(parts) >= 3:
        return f"{parts[-2]}_{parts[-1]}"
    return ""


def _load_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "summary.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _load_trades(run_dir: Path) -> Optional[pd.DataFrame]:
    for name in ("trades.csv", "forex_trades.xlsx", "nas_xau_trades.xlsx"):
        p = run_dir / name
        if p.exists():
            try:
                if name.endswith(".csv"):
                    return pd.read_csv(p)
                return pd.read_excel(p, sheet_name="Trades")
            except Exception:
                continue
    return None


def _session_breakdown(df: pd.DataFrame, r_col: str = "r_realised") -> Dict[str, Any]:
    if df is None or r_col not in df.columns:
        return {}
    filled = df[df.get("exit_reason", pd.Series(dtype=str)) != "never_filled"] if "exit_reason" in df.columns else df
    if "session" not in filled.columns or filled.empty:
        return {}
    out = {}
    for sess, grp in filled.groupby("session"):
        out[sess] = {
            "trades": int(len(grp)),
            "win_rate_pct": _win_rate(grp, r_col),  # wins/(wins+losses); None if all BE
            "expectancy_r": round(float(grp[r_col].mean()), 3) if len(grp) else 0,
        }
    return out


def _max_drawdown(df: pd.DataFrame, r_col: str = "r_realised") -> float:
    """Max peak-to-trough drawdown in R from the equity curve."""
    if df is None or r_col not in df.columns or df.empty:
        return 0.0
    filled = df[df.get("exit_reason", pd.Series(dtype=str)) != "never_filled"] if "exit_reason" in df.columns else df
    if filled.empty:
        return 0.0
    equity = filled[r_col].cumsum().values
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 3)


def _longest_losing_streak(df: pd.DataFrame, r_col: str = "r_realised") -> int:
    if df is None or r_col not in df.columns or df.empty:
        return 0
    filled = df[df.get("exit_reason", pd.Series(dtype=str)) != "never_filled"] if "exit_reason" in df.columns else df
    streak = max_streak = 0
    for v in filled[r_col]:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _score_verdict(score_buckets: List[Dict]) -> str:
    """Simple monotonicity check across score buckets."""
    if not score_buckets or len(score_buckets) < 2:
        return "insufficient data"
    exp_values = [b.get("expectancy_r", 0) for b in score_buckets]
    # Count how many consecutive bucket pairs are improving.
    rises = sum(1 for a, b in zip(exp_values, exp_values[1:]) if b > a)
    total = len(exp_values) - 1
    if total == 0:
        return "insufficient data"
    ratio = rises / total
    if ratio >= 0.7:
        return "WORKS — higher score leads to better trades"
    if ratio >= 0.4:
        return "WEAK — partial relationship"
    return "BROKEN — score does not predict outcome"


def _extract_metrics(run_dir: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Pull all cross-run-relevant metrics from one run folder."""
    trades_df = _load_trades(run_dir)

    # Headline view: proximal r_realised (default policy = TP2-ride with
    # SL-to-BE after TP1). Same column the email headline uses, so the
    # registry's numbers reconcile with what the user saw in their inbox.
    # The exit_tp1 / exit_tp2 hypotheticals are kept in summary.json for
    # the head-to-head policy comparison only -- never as the headline.
    boards = summary.get("scoreboards", {})
    primary = boards.get("proximal_realised", {})
    if not primary or primary.get("trades", 0) == 0:
        # Legacy summaries (pre-2026-05) only had the tp1/tp2 hypotheticals;
        # fall back so old runs still register.
        primary = boards.get("proximal_exit_tp2", {}) or boards.get("proximal_exit_tp1", {})

    total_trade_rows = summary.get("total_trade_rows", 0)
    filled_trades = primary.get("trades", 0)
    win_rate = primary.get("win_rate_pct", 0)
    expectancy_r = primary.get("expectancy_r", 0)
    fill_prox = summary.get("fill_rate_proximal", {})

    # Breakdown keys match the writer in h1_only_reporting.write_h1_only_report
    # (per_pair_proximal_realised, score_buckets_proximal_realised).
    per_pair = (summary.get("per_pair_proximal_realised")
                or summary.get("per_pair_proximal_tp2")  # legacy fallback
                or [])
    score_buckets = (summary.get("score_buckets_proximal_realised")
                     or summary.get("score_buckets_tp2")  # legacy fallback
                     or [])

    # Registry headline metrics (filled_trades, win_rate, expectancy) all come
    # from the proximal scoreboard. Drawdown, streak, and session breakdown must
    # use the SAME population — otherwise we mix two entry zones in one record
    # and the numbers contradict each other.
    if trades_df is not None and "entry_zone" in trades_df.columns:
        prox_df = trades_df[trades_df["entry_zone"] == "proximal"]
    else:
        prox_df = trades_df
    max_dd = _max_drawdown(prox_df)
    streak = _longest_losing_streak(prox_df)
    session_data = _session_breakdown(prox_df)

    return {
        "total_rows": total_trade_rows,
        "filled_trades": filled_trades,
        "win_rate_pct": win_rate,
        "expectancy_r": expectancy_r,
        "max_dd_r": max_dd,
        "longest_losing_streak": streak,
        "fill_rate_proximal_pct": fill_prox.get("fill_rate_pct", 0),
        "per_pair": per_pair,
        "by_session": session_data,
        "score_buckets": score_buckets,
        "score_verdict": _score_verdict(score_buckets),
    }


def _build_entry(run_dir: Path) -> Optional[Dict[str, Any]]:
    run_id = run_dir.name
    summary = _load_summary(run_dir)
    if summary is None:
        return None

    date_key = _date_key(run_id)
    plan_meta = RUN_METADATA.get(date_key, {})

    meta = summary.get("meta", {})
    start_date = meta.get("start", date_key[:8] if date_key else "")
    end_date = meta.get("end", date_key[9:] if len(date_key) > 8 else "")

    metrics = _extract_metrics(run_dir, summary)

    return {
        "run_id": run_id,
        "start": start_date,
        "end": end_date,
        "group": plan_meta.get("group", 0),
        "regime": plan_meta.get("regime", summary.get("meta", {}).get("regime", "unspecified")),
        "condition": plan_meta.get("condition", ""),
        "risk_usd": summary.get("risk_per_trade_usd", 250),
        "metrics": metrics,
        "notes": "",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def _fmt_pct(v, suffix="%") -> str:
    # None = win rate undefined (no resolved trade, all breakevens). Em-dash.
    if v is None:
        return "—"
    return f"{v:.1f}{suffix}"


def _fmt_r(v) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}R"


def _write_markdown(entries: List[Dict[str, Any]]) -> None:
    lines = [
        "# Backtest Registry",
        "",
        f"*Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "Each row is one backtest run. Use this to spot patterns across runs — not just within one.",
        "",
    ]

    # Cross-run summary table
    filled_entries = [e for e in entries if e["metrics"].get("filled_trades", 0) > 0]
    if filled_entries:
        total_trades = sum(e["metrics"]["filled_trades"] for e in filled_entries)
        all_exp = [e["metrics"]["expectancy_r"] for e in filled_entries if e["metrics"].get("expectancy_r") is not None]
        avg_exp = round(sum(all_exp) / len(all_exp), 3) if all_exp else 0
        lines += [
            "## Cross-run snapshot",
            "",
            f"- Runs completed: **{len(filled_entries)}** of {len(entries)}",
            f"- Total filled trades across all runs: **{total_trades}**",
            f"- Average expectancy across runs: **{_fmt_r(avg_exp)}**",
            "",
        ]

        # Group breakdown
        for g in [1, 2, 3]:
            g_entries = [e for e in filled_entries if e.get("group") == g]
            if not g_entries:
                continue
            g_trades = sum(e["metrics"]["filled_trades"] for e in g_entries)
            g_exp = [e["metrics"]["expectancy_r"] for e in g_entries]
            g_avg = round(sum(g_exp) / len(g_exp), 3) if g_exp else 0
            label = {1: "Study", 2: "Out-of-sample", 3: "Live-era"}.get(g, f"Group {g}")
            lines.append(f"- **Group {g} ({label}):** {len(g_entries)} runs, {g_trades} trades, avg expectancy {_fmt_r(g_avg)}")
        lines.append("")

    lines += ["---", "", "## Individual run log", ""]

    for entry in sorted(entries, key=lambda e: (e.get("group", 9), e.get("start", ""))):
        m = entry["metrics"]
        group_label = {1: "Group 1 — Study", 2: "Group 2 — OOS", 3: "Group 3 — Live-era"}.get(entry.get("group", 0), "Unclassified")
        header = f"### {entry['run_id']}"
        lines += [header, ""]
        lines.append(f"**{group_label}** | {entry.get('start')} to {entry.get('end')} | Regime: `{entry.get('regime', '?')}`")
        if entry.get("condition"):
            lines.append(f"Market: *{entry['condition']}*")
        lines.append("")

        if m.get("filled_trades", 0) == 0:
            lines.append("*No filled trades — run may have failed or week was too quiet.*")
            lines.append("")
            continue

        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Filled trades | {m['filled_trades']} (of {m['total_rows']} rows) |",
            f"| Win rate | {_fmt_pct(m.get('win_rate_pct', 0))} |",
            f"| Expectancy | {_fmt_r(m.get('expectancy_r', 0))} |",
            f"| Max drawdown | {_fmt_r(m.get('max_dd_r', 0))} |",
            f"| Longest losing streak | {m.get('longest_losing_streak', 0)} trades |",
            f"| Proximal fill rate | {_fmt_pct(m.get('fill_rate_proximal_pct', 0))} |",
            f"| Score verdict | {m.get('score_verdict', '?')} |",
            "",
        ]

        # Per-pair
        per_pair = m.get("per_pair", [])
        if per_pair:
            lines += ["**By pair (proximal entry, TP2 exit):**", ""]
            lines += ["| Pair | Trades | Win rate | Expectancy |", "|------|--------|----------|------------|"]
            for row in per_pair:
                lines.append(
                    f"| {row.get('pair','?')} | {row.get('trades','?')} "
                    f"| {_fmt_pct(row.get('win_rate_pct', 0))} "
                    f"| {_fmt_r(row.get('expectancy_r', 0))} |"
                )
            lines.append("")

        # Session
        by_sess = m.get("by_session", {})
        if by_sess:
            lines += ["**By session:**", ""]
            lines += ["| Session | Trades | Win rate | Expectancy |", "|---------|--------|----------|------------|"]
            for sess, sd in sorted(by_sess.items()):
                lines.append(
                    f"| {sess} | {sd.get('trades','?')} "
                    f"| {_fmt_pct(sd.get('win_rate_pct', 0))} "
                    f"| {_fmt_r(sd.get('expectancy_r', 0))} |"
                )
            lines.append("")

        # Score buckets
        score_buckets = m.get("score_buckets", [])
        if score_buckets:
            lines += ["**Score vs outcome:**", ""]
            lines += ["| Score bucket | Trades | Win rate | Expectancy |", "|-------------|--------|----------|------------|"]
            for b in score_buckets:
                lines.append(
                    f"| {b.get('score_bucket','?')} | {b.get('trades','?')} "
                    f"| {_fmt_pct(b.get('win_rate_pct', 0))} "
                    f"| {_fmt_r(b.get('expectancy_r', 0))} |"
                )
            lines.append("")

        if entry.get("notes"):
            lines += [f"**Notes:** {entry['notes']}", ""]

        lines += ["---", ""]

    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [registry] BACKTEST_LOG.md written ({len(entries)} runs)")


def build_registry(target_run_id: Optional[str] = None) -> None:
    # Load existing registry so we don't lose manually-added notes.
    existing: Dict[str, Dict] = {}
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH) as f:
                for e in json.load(f).get("runs", []):
                    existing[e["run_id"]] = e
        except Exception:
            pass

    run_dirs = sorted(RESULTS_DIR.iterdir()) if RESULTS_DIR.exists() else []
    entries: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        if target_run_id and run_dir.name != target_run_id:
            # Keep existing entry, don't re-process.
            if run_dir.name in existing:
                entries.append(existing[run_dir.name])
            continue

        entry = _build_entry(run_dir)
        if entry is None:
            continue
        # Preserve manually-added notes from existing registry.
        if run_dir.name in existing:
            entry["notes"] = existing[run_dir.name].get("notes", "")
        entries.append(entry)
        print(f"  [registry] processed {run_dir.name} — "
              f"{entry['metrics'].get('filled_trades', 0)} filled trades, "
              f"expectancy {_fmt_r(entry['metrics'].get('expectancy_r', 0))}")

    with open(REGISTRY_PATH, "w") as f:
        json.dump({"updated_at": datetime.now(timezone.utc).isoformat(), "runs": entries}, f, indent=2, default=str)
    print(f"  [registry] registry.json written ({len(entries)} runs)")

    _write_markdown(entries)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None, help="Update only this run ID. Omit to rebuild all.")
    args = ap.parse_args()
    build_registry(target_run_id=args.run_id)


if __name__ == "__main__":
    main()
