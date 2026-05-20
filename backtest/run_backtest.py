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
from backtest.run_logger import RunLogger, log_event

PHASE3_PAIRS = {"NAS100", "GOLD"}
RESULTS_ROOT = _REPO_ROOT / "backtest" / "results"


def _load_config() -> dict:
    with open(_REPO_ROOT / "config.json") as f:
        return json.load(f)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def run(start: datetime, end: datetime, pair_names: list,
        regime: str = "unspecified", risk_usd: float = 250.0,
        send_email: bool = False) -> Path:
    cfg = _load_config()

    # Initialise per-run logger as the first action. console.log + run_log.jsonl
    # land in the results folder so they ride along with the artifact upload.
    run_id = f"{regime}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    out_dir = RESULTS_ROOT / run_id
    logger = RunLogger.init(out_dir)
    logger.event("run_start", regime=regime, start=start.strftime("%Y-%m-%d"),
                 end=end.strftime("%Y-%m-%d"), pairs=pair_names,
                 risk_usd=risk_usd, send_email=send_email)

    try:
        return _run_inner(cfg, start, end, pair_names, regime, risk_usd,
                          send_email, out_dir, run_id)
    except Exception as e:
        log_event("run_fatal", level="error", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        log_event("run_end")
        logger.close()


def _run_inner(cfg, start, end, pair_names, regime, risk_usd, send_email,
               out_dir, run_id):
    # Need ~30d warmup before start for dealing_range cold start.
    fetch_start = start - timedelta(days=35)

    pairs_to_run = [p for p in cfg["pairs"] if p["name"] in pair_names]
    if not pairs_to_run:
        log_event("abort_no_pairs", level="error", requested=pair_names)
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

        if df_h1 is None:
            log_event("pair_skip", level="warn", pair=name,
                      reason="h1_unavailable")
            continue
        log_event("pair_data_loaded", pair=name,
                  h1_rows=len(df_h1),
                  m15_rows=(0 if df_m15 is None else len(df_m15)),
                  m5_rows=(0 if df_m5 is None else len(df_m5)),
                  h1_first=str(df_h1.index.min()),
                  h1_last=str(df_h1.index.max()))

        # Determine simulation mode.
        # M15 and M5 are limited to ~60 days by Yahoo Finance.
        # For older weeks only H1 is available — fall back to H1-only mode.
        m15_covers_window = (
            df_m15 is not None and not df_m15.empty
            and df_m15.index.min() <= walk_start_ts
        )
        used_phase3 = False
        h1_only_mode = not m15_covers_window

        if h1_only_mode:
            trigger = df_h1
            print(f"  [H1-ONLY MODE] M15 unavailable for this period "
                  f"(yfinance 60d limit). Simulating on H1 bars. "
                  f"Results are less granular — read in context.")
        else:
            trigger = df_m15
            if name in PHASE3_PAIRS and df_m5 is not None and not df_m5.empty:
                if df_m5.index.min() <= walk_start_ts:
                    used_phase3 = True
            print(f"  trigger TF: M15"
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

        log_event("pair_alerts_scored", pair=name,
                  alerts=len(alerts_for_pair), mode=("h1_only" if h1_only_mode else ("phase3" if used_phase3 else "phase2")))
        print(f"  {name}: {len(alerts_for_pair)} would-be alerts")

        for alert in alerts_for_pair:
            score, breakdown = trade_simulator.score_ob_confluences(
                alert["ob"], pair_conf, alert["current_price"],
                alert["h1_atr"], alert["walls"]
            )
            if score < pair_conf.get("min_confidence", 6.0):
                continue

            if h1_only_mode:
                # H1-only fallback: M15 not available for this period.
                trade = trade_simulator.simulate_trade_h1only(
                    alert, pair_conf, df_h1, risk_usd=risk_usd
                )
            elif used_phase3 and df_m5 is not None and not df_m5.empty:
                alert["_df_h1"] = df_h1
                p3_trigger = replay_engine.replay_phase3_watch(
                    alert, pair_conf, df_m5, walk_end_ts
                )
                if p3_trigger:
                    trade = trade_simulator.simulate_phase3_trade(
                        p3_trigger, pair_conf, df_h1, df_m15, df_m5,
                        risk_usd=risk_usd
                    )
                else:
                    trade = None
            else:
                trade = trade_simulator.simulate_trade(
                    alert, pair_conf, df_h1, trigger, risk_usd=risk_usd
                )
            if trade:
                trade["score"] = score
                trade["score_breakdown"] = breakdown
                all_trades.append(trade)

        n_trades_pair = sum(1 for t in all_trades if t['pair'] == name)
        log_event("pair_trades_simulated", pair=name, trades=n_trades_pair)
        print(f"  {name}: {n_trades_pair} simulated trades")

    meta = {
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "regime": regime,
        "pairs": pair_names,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_dir = reporting.write_report(run_id, all_trades, all_alerts, meta, risk_usd=risk_usd)
    log_event("report_written", path=str(report_dir),
              total_alerts=len(all_alerts), total_trades=len(all_trades))
    print(f"\nReport written to {report_dir}")

    if send_email:
        try:
            reporting_email.send_report(report_dir, subject_suffix=f"({regime})")
            log_event("email_sent", path=str(report_dir))
        except Exception as e:
            log_event("email_failed", level="error",
                      error=f"{type(e).__name__}: {e}")

    return report_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--pairs", default="EURUSD,NZDUSD,USDJPY,USDCHF,NAS100,GOLD",
                    help="Comma-separated pair names")
    ap.add_argument("--regime", default="unspecified", choices=["war", "bau", "unspecified"])
    ap.add_argument("--risk-usd", type=float, default=250.0)
    ap.add_argument("--email", action="store_true", help="Send report email")
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    run(start, end, pairs, regime=args.regime, risk_usd=args.risk_usd,
        send_email=args.email)


if __name__ == "__main__":
    main()
