"""Fast email-report preview harness.

    python -m backtest.render_report <run_id> [--out preview] [--no-excel]

Reads an EXISTING run dir (backtest/results/<run_id>/) and re-renders the two
per-group HTML emails in seconds — no 25-minute simulation. Design-iteration
tool ONLY: it never commits, pushes, emails, or overwrites the committed report.
Output goes to backtest/results/<run_id>/preview/.

Why this exists (EMAIL_REBUILD_SPEC §5): rebuilding the email body was a
25-min-per-change loop because the only way to see it was a full backtest. This
loads the run's on-disk artifacts back into the exact in-memory shapes
write_h1_only_report expects, so a body change renders instantly.

THE TYPE-COERCION TRAP (§5): pd.read_csv turns bool columns into the STRINGS
"True"/"False", and "False" is truthy — so `if t.get("news_blocked")` would fire
on every row. Every bool column is coerced back to a real bool below, NaN → None
everywhere, numerics stay numeric. Get this wrong and the preview silently
diverges from the pipeline.

THE EXIT-LAB SINK (§5, corrected): the spec assumed exit_lab_trades.csv matches
the sink row keys. It does NOT — that CSV is written by a SEPARATE diagnostic
(diagnostics/exit_lab.py) with a different schema (no entry_zone, no
ob_timestamp, recipe_exit_reason ≠ the LIVE exit_reason the recipe table filters
on). The faithful source is exit_lab_sink.jsonl, persisted by run_backtest.py
for every post-fix run. When that file is absent (runs older than the sink-dump
change), we fall back to a CLEARLY-LABELLED reconstruction from
exit_lab_trades.csv + trades.csv — good enough for LAYOUT work, but its recipe
numbers are approximate (the join key is non-unique for a handful of trades), so
the harness prints a loud warning and the parity gate will not pass on it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest import h1_only_reporting as R


# Bool columns pd.read_csv mangles into "True"/"False" strings. Coerce to real
# bools so truthiness checks (`if t.get("news_blocked")`) behave like the
# pipeline's live dicts. Verified against trades.csv header on 2026-07-02.
_BOOL_COLS = [
    "fvg_present", "sweep_present", "news_blocked", "ist_blocked",
    "weekend_blocked", "killzone_blocked", "ob_in_killzone", "fill_in_killzone",
    "sl_collision", "eligible_for_headline", "reversed_from_extreme",
    "fvg_mitigation", "sl_bar_was_sweep", "sl_swept_then_tp1",
]


def _coerce_bool(v: Any) -> Optional[bool]:
    """Coerce a CSV cell to a real bool, or None when blank/NaN."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0", "yes"):
        return True
    if s in ("false", "0", "0.0", "no"):
        return False
    if s in ("", "nan", "none"):
        return None
    return bool(v)


def _load_trades(run_dir: Path) -> List[Dict[str, Any]]:
    """trades.csv → List[Dict] in the shape _build_group_html consumes.

    Bool columns → real bools; every NaN → None; numerics stay numeric.
    """
    df = pd.read_csv(run_dir / "trades.csv")
    rows: List[Dict[str, Any]] = []
    bool_cols = [c for c in _BOOL_COLS if c in df.columns]
    for rec in df.to_dict("records"):
        out: Dict[str, Any] = {}
        for k, v in rec.items():
            # NaN → None everywhere (pd uses float NaN for every missing cell).
            if isinstance(v, float) and math.isnan(v):
                out[k] = None
            else:
                out[k] = v
        for c in bool_cols:
            out[c] = _coerce_bool(rec.get(c))
        rows.append(out)
    return rows


def _load_exit_lab_sink(run_dir: Path) -> tuple:
    """The exit-lab sink in PIPELINE shape. Prefers the faithful jsonl dump;
    falls back to a labelled reconstruction from the diagnostic CSV + trades.csv.

    Returns (rows, exact): `exact` is True only when the faithful jsonl dump was
    read. On the reconstructed fallback `exact` is False, which tells the report
    builder to skip the exit-table count invariant (a reconstructed sink cannot
    satisfy it exactly — the numbers are layout-only).

    Sink row keys the email reads: pair, alert_ts, ob_timestamp, direction,
    entry_zone, committed_r, exit_reason, config, r.
    """
    jsonl = run_dir / "exit_lab_sink.jsonl"
    if jsonl.exists():
        rows: List[Dict[str, Any]] = []
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows, True

    # --- Fallback: reconstruct from exit_lab_trades.csv + trades.csv ----------
    csv = run_dir / "exit_lab_trades.csv"
    if not csv.exists():
        print("  [render warn] no exit_lab_sink.jsonl and no exit_lab_trades.csv "
              "— recipe table will render empty.")
        return [], False
    print("  [render WARN] exit_lab_sink.jsonl absent (pre-sink-dump run). "
          "Reconstructing the sink from exit_lab_trades.csv — recipe numbers are "
          "APPROXIMATE (non-unique join key). Use for LAYOUT ONLY; re-render on a "
          "post-fix run before trusting recipe values. Parity gate will NOT pass.")
    el = pd.read_csv(csv)
    tr = pd.read_csv(run_dir / "trades.csv")
    # Best-available join: (pair, alert_ts, direction) → the LIVE exit_reason and
    # ob_timestamp. Non-unique for ~a handful of collision trades (two OBs, same
    # pair+ts+dir); those rows may carry a wrong exit_reason. Hence "layout only".
    key_cols = ["pair", "alert_ts", "direction"]
    look: Dict[tuple, Dict[str, Any]] = {}
    for rec in tr.to_dict("records"):
        k = (rec.get("pair"), str(rec.get("alert_ts")), rec.get("direction"))
        # First write wins (stable); collisions are the known imperfect case.
        look.setdefault(k, {"exit_reason": rec.get("exit_reason"),
                            "ob_timestamp": rec.get("ob_timestamp")})
    rows = []
    for rec in el.to_dict("records"):
        k = (rec.get("pair"), str(rec.get("alert_ts")), rec.get("direction"))
        joined = look.get(k, {})
        rows.append({
            "pair": rec.get("pair"),
            "alert_ts": str(rec.get("alert_ts")),
            "ob_timestamp": joined.get("ob_timestamp"),
            "direction": rec.get("direction"),
            "entry_zone": "proximal",  # trades.csv is proximal-only (verified)
            "committed_r": rec.get("committed_r"),
            "exit_reason": joined.get("exit_reason"),  # LIVE reason, not recipe's
            "config": rec.get("config"),
            "r": rec.get("r"),
        })
    return rows, False


def _load_meta(run_dir: Path) -> Dict[str, Any]:
    """summary.json → the `meta` dict write_h1_only_report expects."""
    p = run_dir / "summary.json"
    if not p.exists():
        return {}
    s = json.loads(p.read_text(encoding="utf-8"))
    return dict(s.get("meta") or {})


def _load_raw_alerts(run_dir: Path) -> List[Dict[str, Any]]:
    """raw_alerts.jsonl → list of alert dicts. Only `pair` is read downstream
    (the Act 6 funnel counts alerts per book), but we load full rows so the
    per-group split matches the pipeline."""
    p = run_dir / "raw_alerts.jsonl"
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def render(run_id: str, write_excel: bool = False) -> Path:
    base = Path(__file__).parent / "results"
    run_dir = base / run_id
    if not run_dir.exists():
        sys.exit(f"run dir not found: {run_dir}")

    trades = _load_trades(run_dir)
    sink, sink_exact = _load_exit_lab_sink(run_dir)
    meta = _load_meta(run_dir)
    raw_alerts = _load_raw_alerts(run_dir)

    out_dir = R.write_h1_only_report(
        run_id, trades, raw_alerts, meta,
        risk_usd=float(meta.get("risk_per_trade_usd") or 250.0),
        exit_lab_sink=sink, preview=True, write_excel=write_excel,
        sink_exact=sink_exact,
    )
    preview_dir = out_dir  # write_h1_only_report already appended /preview
    print(f"\n[render] preview written to {preview_dir}")
    for name in ("report_forex.html", "report_gold_nas.html"):
        fp = preview_dir / name
        if fp.exists():
            kb = len(fp.read_bytes()) / 1024
            print(f"    {name}  ({kb:.0f} KB)")
    return preview_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast email-report preview harness.")
    ap.add_argument("run_id", help="run dir under backtest/results/")
    ap.add_argument("--out", default="preview",
                    help="(informational) preview subdir name; always 'preview'")
    ap.add_argument("--excel", action="store_true",
                    help="also rebuild the xlsx (slower; off by default)")
    args = ap.parse_args()
    render(args.run_id, write_excel=args.excel)


if __name__ == "__main__":
    main()
