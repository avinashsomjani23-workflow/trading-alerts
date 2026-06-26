"""
EXIT LAB — compare exit recipes from ONE fresh, self-consistent backtest pass.

Why a fresh run (not a replay over committed trades)
----------------------------------------------------
Exit configs only change what happens after a fill, so in principle we could
replay them over the committed trades. We tried — it FAILS: the yfinance cache has
drifted from when those trades were generated (GC=F/NQ=F futures are back-adjusted
on every rollover; even =X spot drifts 1-3 pips). Reconstructed bars are not the
bars the trades were born from, so a replay is unfaithful (the baseline self-check
caught this).

The faithful method: run the backtest ONCE, and at each fill let the simulator's
side-channel (h1_only_simulator.EXIT_LAB_*) replay every recipe over the SAME
in-memory post-fill bars. Entry/SL/TP1/exits all come from one consistent dataset.

Self-consistency check: the BASELINE recipe (single liquidity-TP + BE@+1R = the
live policy) must reproduce each trade's committed r_realised within this run.

NOTE: numbers use CURRENT yfinance bars, so they will differ from the committed
audit (especially GOLD). They are PROVISIONAL until re-run on MT5 data — but the
RELATIVE ranking of exit recipes (the research question) is valid within the run.

Run:
  python -m backtest.diagnostics.exit_lab
  python -m backtest.diagnostics.exit_lab --pairs EURUSD,NZDUSD,USDJPY,USDCHF,GOLD \
        --start 2024-07-01 --end 2025-06-30
"""
import argparse
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import backtest.h1_only_simulator as sim
from backtest.insights import bootstrap_ci

# ── The locked experiment set (HANDOFF / RECOMMENDATIONS) ───────────────────
# target spec: float = R-multiple, "tp1" = liquidity TP. be_trigger_r None = no BE.
CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline_liqTP_be1.0":  {"legs": [(1.0, "tp1")], "be_trigger_r": 1.0, "be_to_r": 0.0},
    # Group B — break-even sweep (no partial, liquidity TP).
    "B_be0.5":               {"legs": [(1.0, "tp1")], "be_trigger_r": 0.5, "be_to_r": 0.0},
    "B_be0.7":               {"legs": [(1.0, "tp1")], "be_trigger_r": 0.7, "be_to_r": 0.0},
    # Group C — set-and-forget fixed full TP, no partial, no BE.
    "C_fullTP_0.5R":         {"legs": [(1.0, 0.5)], "be_trigger_r": None},
    "C_fullTP_1.0R":         {"legs": [(1.0, 1.0)], "be_trigger_r": None},
    "C_fullTP_1.5R":         {"legs": [(1.0, 1.5)], "be_trigger_r": None},
    "C_fullTP_2.0R":         {"legs": [(1.0, 2.0)], "be_trigger_r": None},
    # Group D — partial 50% @ +1R, runner to liquidity, BE entry after partial.
    "D_partial50_1R_runLiq": {"legs": [(0.5, 1.0), (0.5, "tp1")], "be_trigger_r": 1.0, "be_to_r": 0.0},
}


def _quarter(ts_str: str) -> str:
    ts = pd.to_datetime(ts_str, utc=True)
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


def _summary(sub: pd.DataFrame, r_col: str = "r") -> Dict[str, Any]:
    vals = sub[r_col].tolist()
    n = len(vals)
    wins = int((sub[r_col] > 0).sum())
    losses = int((sub[r_col] < 0).sum())
    resolved = wins + losses
    wr = round(100 * wins / resolved, 1) if resolved else None
    lo, hi = bootstrap_ci(vals)
    pq = sub.groupby("quarter")[r_col].mean().round(2).to_dict()
    return {"n": n, "totR": round(float(np.sum(vals)), 1),
            "expR": round(float(np.mean(vals)), 3) if n else None,
            "WR": wr, "CI": (lo, hi), "pq": pq}


def _print_block(rep: pd.DataFrame, label: str) -> None:
    print(f"\n================  {label}  ================")
    print(f"{'config':24} {'N':>4} {'totR':>7} {'expR':>7} {'WR%':>5} "
          f"{'95% CI':>18}  per-quarter expR")
    for name in CONFIGS:
        sub = rep[rep["config"] == name]
        if sub.empty:
            continue
        s = _summary(sub)
        ci = s["CI"]
        ci_s = f"[{ci[0]}, {ci[1]}]" if ci[0] is not None else "[--]"
        pq_s = " ".join(f"{v:+.2f}" for _, v in sorted(s["pq"].items()))
        flag = "  <= CI>0" if (ci[0] is not None and ci[0] > 0) else ""
        wr = s["WR"] if s["WR"] is not None else "--"
        print(f"{name:24} {s['n']:>4} {s['totR']:>7} {s['expR']:>7} {wr:>5} "
              f"{ci_s:>18}  {pq_s}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="EURUSD,NZDUSD,USDJPY,USDCHF,GOLD",
                    help="ex-NAS by default (NAS100 is being dropped)")
    ap.add_argument("--start", default="2024-07-01")
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--keep-run", action="store_true",
                    help="keep the throwaway results dir this run creates")
    args = ap.parse_args()

    # Neutralise persistence — this is a diagnostic, it must NOT touch git/registry.
    import backtest.update_registry as _ur
    import backtest.commit_logs as _cl
    _ur.build_registry = lambda *a, **k: None
    _cl.commit_run_logs = lambda *a, **k: "diag-nocommit"

    # Arm the side-channel.
    sink: List[Dict[str, Any]] = []
    sim.EXIT_LAB_CONFIGS = CONFIGS
    sim.EXIT_LAB_SINK = sink

    import backtest.run_backtest as rb
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]

    print(f"Fresh backtest {args.start}..{args.end}  pairs={pairs}")
    print("(persistence disabled; exit-lab side-channel armed)")
    out_dir = rb.run(start, end, pairs, regime="bau", risk_usd=250.0, send_email=False)

    sim.EXIT_LAB_CONFIGS = None
    sim.EXIT_LAB_SINK = None

    rep = pd.DataFrame(sink)
    rep = rep[rep["entry_zone"] == "proximal"].copy()  # the live model
    rep["quarter"] = rep["alert_ts"].map(_quarter)
    n_trades = rep["alert_ts"].nunique()
    print(f"\nCaptured {n_trades} proximal trades x {len(CONFIGS)} configs")

    # ── Self-check: baseline replay vs committed r_realised (same run) ───────
    base = rep[rep["config"] == "baseline_liqTP_be1.0"]
    c_exp = round(base["committed_r"].mean(), 3)
    r_exp = round(base["r"].mean(), 3)
    diff = round(abs(c_exp - r_exp), 4)
    print("\n---- ENGINE SELF-CHECK (baseline vs committed, same bars) ----")
    print(f"  committed expR={c_exp}  baseline-replay expR={r_exp}  |diff|={diff}")
    print("  PASS" if diff <= 0.01 else "  *** FAIL — investigate before trusting ***")

    _print_block(rep, "EX-NAS book")

    if not args.keep_run and out_dir is not None:
        try:
            shutil.rmtree(out_dir)
            print(f"\n[cleaned throwaway run dir {out_dir}]")
        except Exception as e:
            print(f"\n[could not remove {out_dir}: {e}]")


if __name__ == "__main__":
    main()
