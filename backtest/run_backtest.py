"""CLI entry point for backtest runs.

Usage:
    python backtest/run_backtest.py --start 2025-09-15 --end 2025-09-19 --regime war
    python backtest/run_backtest.py --start 2025-09-15 --end 2025-09-19 --pairs EURUSD,GOLD
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from backtest import data_loader, replay_engine, trade_simulator, reporting
from backtest import reporting_email

PHASE3_PAIRS = {"NAS100", "GOLD"}


def _load_config() -> dict:
    with open(_REPO_ROOT / "config.json") as f:
        return json.load(f)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def run(start: datetime, end: datetime, pair_names: list,
        regime: str = "unspecified", risk_gbp: float = 250.0,
        send_email: bool = False) -> Path:
    cfg = _load_config()

    # Need ~30d warmup before start for dealing_range cold start.
    fetch_start = start - timedelta(days=35)

    pairs_to_run = [p for p in cfg["pairs"] if p["name"] in pair_names]
    if not pairs_to_run:
        print(f"  [abort] no matching pairs for {pair_names}")
        return None

    state = replay_engine.ReplayState()
    all_alerts = []
    all_trades = []

    walk_start_ts = pd.Timestamp(start)
    walk_end_ts = pd.Timestamp(end)

    for pair_conf in pairs_to_run:
        name = pair_conf["name"]
        symbol = pair_conf["symbol"]
        print(f"\n=== {name} ({symbol}) ===")

        df_h1 = data_loader.load_bars(symbol, "1h", fetch_start, end)
        df_m15 = data_loader.load_bars(symbol, "15m", fetch_start, end)
        df_m5 = data_loader.load_bars(symbol, "5m", fetch_start, end)

        if df_h1 is None or df_m15 is None:
            print(f"  [skip {name}] H1 or M15 unavailable for window")
            continue

        # Choose trigger TF: M5 if available + recent enough, else M15.
        # For backtesting, treat NAS/XAU as M15 if M5 doesn't cover the window.
        trigger = df_m15
        used_phase3 = False
        if name in PHASE3_PAIRS and df_m5 is not None and not df_m5.empty:
            if df_m5.index.min() <= walk_start_ts:
                used_phase3 = True
                # trigger stays M15 — Phase 3 uses its own M5 walk internally
        print(f"  trigger TF: {'M5' if trigger is df_m5 else 'M15'}"
              f" {'(Phase 3 model)' if used_phase3 else '(Phase 2 model)'}")

        alerts_for_pair = []
        for event in replay_engine.replay_pair(
            pair_conf, df_h1, df_m15, df_m5, state, walk_start_ts, walk_end_ts
        ):
            if event["kind"] == "alert":
                alerts_for_pair.append(event)
                all_alerts.append({
                    "pair": event["pair"],
                    "ts": str(event["ts"]),
                    "ob_timestamp": event["ob"].get("ob_timestamp"),
                    "direction": event["ob"].get("direction"),
                    "bos_tag": event["ob"].get("bos_tag"),
                    "bos_tier": event["ob"].get("bos_tier"),
                })

        print(f"  {name}: {len(alerts_for_pair)} would-be alerts")

        for alert in alerts_for_pair:
            score, breakdown = trade_simulator.score_ob_confluences(
                alert["ob"], pair_conf, alert["current_price"],
                alert["h1_atr"], alert["walls"]
            )
            if score < pair_conf.get("min_confidence", 6.0):
                continue

            if used_phase3 and df_m5 is not None and not df_m5.empty:
                # Pass H1 frame so replay_phase3_watch can re-check OB liveness
                # at M5 trigger time (mirrors live Fix 1+12).
                alert["_df_h1"] = df_h1
                # Phase 3 path: walk M5 for tap + CHoCH trigger, then simulate.
                p3_trigger = replay_engine.replay_phase3_watch(
                    alert, pair_conf, df_m5, walk_end_ts
                )
                if p3_trigger:
                    trade = trade_simulator.simulate_phase3_trade(
                        p3_trigger, pair_conf, df_h1, df_m15, df_m5,
                        risk_gbp=risk_gbp
                    )
                    if trade:
                        trade["score"] = score
                        trade["score_breakdown"] = breakdown
                        all_trades.append(trade)
            else:
                # Phase 2 path: limit-order on M15.
                trade = trade_simulator.simulate_trade(
                    alert, pair_conf, df_h1, trigger, risk_gbp=risk_gbp
                )
                if trade:
                    trade["score"] = score
                    trade["score_breakdown"] = breakdown
                    all_trades.append(trade)

        print(f"  {name}: {sum(1 for t in all_trades if t['pair'] == name)} simulated trades")

    run_id = f"{regime}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    meta = {
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "regime": regime,
        "pairs": pair_names,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = reporting.write_report(run_id, all_trades, all_alerts, meta, risk_gbp=risk_gbp)
    print(f"\nReport written to {out_dir}")

    if send_email:
        reporting_email.send_report(out_dir, subject_suffix=f"({regime})")

    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--pairs", default="EURUSD,NZDUSD,USDJPY,USDCHF,NAS100,GOLD",
                    help="Comma-separated pair names")
    ap.add_argument("--regime", default="unspecified", choices=["war", "bau", "unspecified"])
    ap.add_argument("--risk-gbp", type=float, default=250.0)
    ap.add_argument("--email", action="store_true", help="Send report email")
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    run(start, end, pairs, regime=args.regime, risk_gbp=args.risk_gbp,
        send_email=args.email)


if __name__ == "__main__":
    main()
