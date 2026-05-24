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

from backtest import data_loader, replay_engine
from backtest import reporting_email
from backtest import h1_only_simulator
from backtest import h1_only_reporting
from backtest import ist_window
from backtest import killzone as killzone_filter
from backtest.run_logger import RunLogger, log_event
import news_filter

RESULTS_ROOT = _REPO_ROOT / "backtest" / "results"


def _load_config() -> dict:
    with open(_REPO_ROOT / "config.json") as f:
        return json.load(f)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _regime_label_for(regime: str, start: datetime, end: datetime):
    """Look up the curated WAR-range label so the email subject can name the
    specific event. Returns None if regime is not 'war' or no range matched."""
    if regime != "war":
        return None
    try:
        from backtest import regime_detector
        _r, label = regime_detector.detect_regime(
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        return label
    except Exception:
        return None


def run(start: datetime, end: datetime, pair_names: list,
        regime: str = "auto", risk_usd: float = 250.0,
        send_email: bool = False) -> Path:
    cfg = _load_config()

    # Regime: 'auto' (default) consults WAR_REGIME_WEEKS.json. Explicit
    # 'war'/'bau' overrides the file. We resolve once here so every downstream
    # tag (run_id prefix, log fields, meta, email subject) sees the same value.
    regime_label = None
    if regime == "auto":
        from backtest import regime_detector
        regime, regime_label = regime_detector.detect_regime(
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    # Initialise per-run logger as the first action. console.log + run_log.jsonl
    # land in the results folder so they ride along with the artifact upload.
    run_id = f"h1only_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    out_dir = RESULTS_ROOT / run_id
    logger = RunLogger.init(out_dir)
    logger.event("run_start", regime=regime, regime_label=regime_label,
                 mode="h1_only",
                 start=start.strftime("%Y-%m-%d"),
                 end=end.strftime("%Y-%m-%d"), pairs=pair_names,
                 risk_usd=risk_usd, send_email=send_email)

    try:
        return _run_h1_only(cfg, start, end, pair_names, regime,
                            risk_usd, send_email, out_dir, run_id)
    except Exception as e:
        log_event("run_fatal", level="error", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        log_event("run_end")
        logger.close()


def _run_h1_only(cfg, start, end, pair_names, regime, risk_usd, send_email,
                 out_dir, run_id):
    """H1-only backtest run.

      - Skips M15 + M5 data fetches entirely (faster, no yfinance 60d issue).
      - No scoring gate (every OB-touch fires).
      - Fires TWO trade rows per OB-touch (proximal + 50% mean) via
        h1_only_simulator.simulate_h1_only_dual.
      - Uses h1_only_reporting for the side-by-side TP1/TP2 scoreboard.
    """
    fetch_start = start - timedelta(days=35)
    pairs_to_run = [p for p in cfg["pairs"] if p["name"] in pair_names]
    if not pairs_to_run:
        log_event("abort_no_pairs", level="error", requested=pair_names)
        return None

    # Backtest-only proximity caps. Live config.json stays at 4.0/4.5; the
    # backtest uses tighter caps because Phase 2 in backtest fires hourly on
    # closed-bar wicks (no microstructure noise), so the live padding isn't
    # needed. Trader-set: 3.0 ATR for FX, 3.5 ATR for index/commodity.
    BACKTEST_ATR_MULT = {"forex": 3.0, "index": 3.5, "commodity": 3.5}
    for p in pairs_to_run:
        live_mult = p.get("atr_multiplier")
        bt_mult = BACKTEST_ATR_MULT.get(p.get("pair_type"), live_mult)
        p["atr_multiplier"] = bt_mult
        log_event("backtest_atr_override", pair=p["name"],
                  pair_type=p.get("pair_type"),
                  live=live_mult, backtest=bt_mult)

    state = replay_engine.ReplayState()
    all_alerts: list = []
    all_trades: list = []
    # Per-pair tally of alerts dropped by the killzone hard filter. Reported
    # in summary.json and the email so the user can see how many setups the
    # filter rejected and why (config_windows in the audit block).
    killzone_drops_by_pair: dict = {}
    walk_start_ts = pd.Timestamp(start)
    walk_end_ts = pd.Timestamp(end)

    print(f"\n[H1-ONLY MODE] start={start.date()} end={end.date()} "
          f"pairs={pair_names} risk_per_trade=${risk_usd:.0f}")
    print(f"  No M15/M5 fetches. No scoring gate. Dual entry per OB.")

    # News blackout filter. Fetch FF (scheduled high-impact events) for the
    # full backtest range plus a 1-day pad on each side (events at the
    # range edges still need their Â±30min window evaluated).
    # `start`/`end` from argparse via _parse_date are already tz-aware UTC,
    # so localize only when naive (defensive â€” never double-localize).
    def _to_utc(ts):
        t = pd.Timestamp(ts)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    news_start = _to_utc(start) - timedelta(days=1)
    news_end   = _to_utc(end)   + timedelta(days=1)
    news_data = news_filter.fetch_events(
        news_start.to_pydatetime(), news_end.to_pydatetime(),
        sources=("ff",),
    )
    news_events = news_data["events"]
    news_coverage = news_data["coverage"]
    print(f"  News: {len(news_events)} High-impact events fetched "
          f"(coverage: {news_coverage})")
    log_event("news_fetched", level="info",
              events=len(news_events), coverage=news_coverage,
              range=f"{news_start.isoformat()}..{news_end.isoformat()}")

    for pair_conf in pairs_to_run:
        name = pair_conf["name"]
        symbol = pair_conf["symbol"]
        print(f"\n=== {name} ({symbol}) ===")

        df_h1 = data_loader.load_bars(symbol, "1h", fetch_start, end)
        if df_h1 is None or df_h1.empty:
            log_event("pair_skip", level="warn", pair=name,
                      reason="h1_unavailable")
            print(f"  [SKIP] H1 unavailable for {name}")
            continue
        log_event("pair_data_loaded", pair=name, h1_rows=len(df_h1),
                  h1_first=str(df_h1.index.min()),
                  h1_last=str(df_h1.index.max()))

        # Walk H1 bars and collect OB-touch alerts.
        alerts_for_pair = []
        for event in replay_engine.replay_pair(
            pair_conf, df_h1,
            state=state, walk_start_ts=walk_start_ts, walk_end_ts=walk_end_ts,
        ):
            if event["kind"] == "alert":
                alerts_for_pair.append(event)
                all_alerts.append({
                    "pair": event["pair"],
                    "ts": str(event["ts"]),
                    "alert_bar_ts": (str(event["alert_bar_ts"])
                                     if event.get("alert_bar_ts") is not None
                                     else None),
                    "alert_seq": int(event.get("alert_seq", 1)),
                    "ob_timestamp": (event.get("ob") or {}).get("ob_timestamp"),
                    "bos_timestamp": (event.get("ob") or {}).get("bos_timestamp"),
                    "direction": (event.get("ob") or {}).get("direction"),
                    "bos_tag": (event.get("ob") or {}).get("bos_tag"),
                    "bos_tier": (event.get("ob") or {}).get("bos_tier"),
                })

        log_event("pair_alerts_collected", pair=name,
                  alerts=len(alerts_for_pair), mode="h1_only")
        print(f"  {name}: {len(alerts_for_pair)} OB-touch alerts")

        n_trades_for_pair = 0
        n_blocked_for_pair = 0
        n_ist_blocked_for_pair = 0
        n_killzone_dropped_for_pair = 0
        pair_type = pair_conf.get("pair_type", "forex")
        for alert in alerts_for_pair:
            # Killzone gate -- alerts outside the pair's configured killzone
            # are SIMULATED for audit (so we can show "what you would have
            # made if you had traded outside the killzone") but tagged
            # `killzone_blocked=True` and EXCLUDED from every aggregate
            # metric in the report. This mirrors the IST/news gates so the
            # user can verify per-pair killzone filtering is actually saving
            # R, not just dropping winners.
            alert_ts_raw = alert["ts"]
            if not isinstance(alert_ts_raw, pd.Timestamp):
                alert_ts_raw = pd.Timestamp(alert_ts_raw)
            if alert_ts_raw.tzinfo is None:
                alert_ts_raw = alert_ts_raw.tz_localize("UTC")
            killzone_blocked = not killzone_filter.in_pair_killzone(
                alert_ts_raw, pair_conf
            )
            if killzone_blocked:
                n_killzone_dropped_for_pair += 1
                log_event("killzone_drop", pair=name,
                          alert_ts=str(alert["ts"]),
                          utc_hour=int(alert_ts_raw.hour),
                          utc_minute=int(alert_ts_raw.minute),
                          windows=killzone_filter.windows_label(pair_conf))

            rows = h1_only_simulator.simulate_h1_only_dual(
                alert, pair_conf, df_h1, risk_usd=risk_usd,
            )
            # Blackout tagging. Option B: simulate every alert; tag
            # blocked rows so they appear in Excel for audit but are
            # excluded from every aggregate metric (handled in reporting).
            #
            # The blackout check uses the alert timestamp, not the fill
            # timestamp, because that is the moment we would (live) decide
            # whether to place the limit order.
            alert_ts = alert_ts_raw
            blocked, src_event = news_filter.is_news_blackout(
                alert_ts.to_pydatetime(), name, news_events,
                window_minutes=30,
            )
            # IST trading-window gate. Live system suppresses everything
            # outside the user's IST window -- backtest must mirror this.
            # Alerts outside the window are simulated for audit but excluded
            # from every aggregate metric. Killzone-blocked rows are also
            # excluded by the same mechanism (handled in reporting).
            ist_blocked = not ist_window.in_user_trading_window(
                alert_ts, pair_type
            )
            for row in rows:
                row["news_blocked"]        = bool(blocked)
                row["news_event_title"]    = src_event["title"]    if blocked else ""
                row["news_event_currency"] = src_event["currency"] if blocked else ""
                row["news_event_source"]   = src_event["source"]   if blocked else ""
                row["news_event_ts"]       = src_event["ts_utc"].isoformat() if blocked else ""
                row["ist_blocked"]         = bool(ist_blocked)
                row["killzone_blocked"]    = bool(killzone_blocked)
                row["killzone_windows"]    = killzone_filter.windows_label(pair_conf)
                row["alert_utc_hour"]      = int(alert_ts.hour)
                all_trades.append(row)
                n_trades_for_pair += 1
                if blocked:
                    n_blocked_for_pair += 1
                if ist_blocked:
                    n_ist_blocked_for_pair += 1
                log_event("trade_simulated", pair=name,
                          alert_ts=str(alert["ts"]),
                          entry_zone=row.get("entry_zone"),
                          score=row.get("score"),
                          model="h1_only",
                          exit_reason=row.get("exit_reason"),
                          r_realised=row.get("r_realised"),
                          r_if_exit_tp1=row.get("r_if_exit_tp1"),
                          r_if_exit_tp2=row.get("r_if_exit_tp2"),
                          pnl_usd=row.get("pnl_usd"),
                          news_blocked=bool(blocked),
                          ist_blocked=bool(ist_blocked),
                          news_event_title=row.get("news_event_title"),
                          news_event_source=row.get("news_event_source"))

        log_event("pair_trades_simulated", pair=name,
                  trades=n_trades_for_pair,
                  news_blocked=n_blocked_for_pair,
                  ist_blocked=n_ist_blocked_for_pair,
                  killzone_dropped_alerts=n_killzone_dropped_for_pair)
        print(f"  {name}: {n_trades_for_pair} simulated trade rows "
              f"(2 per qualified OB-touch; "
              f"{n_blocked_for_pair} news-blocked, "
              f"{n_ist_blocked_for_pair} IST-blocked, "
              f"{n_killzone_dropped_for_pair} killzone-dropped alerts)")
        # Aggregate killzone drops onto a run-level dict for the meta block.
        killzone_drops_by_pair[name] = n_killzone_dropped_for_pair

    n_blocked_total = sum(1 for t in all_trades if t.get("news_blocked"))
    n_ist_blocked_total = sum(1 for t in all_trades if t.get("ist_blocked"))
    n_killzone_dropped_total = sum(killzone_drops_by_pair.values())
    # Per-pair killzone windows for the audit section. Keyed by pair name so
    # the report can show "EURUSD: 07:00-16:30 UTC" alongside the drop count.
    killzone_windows_by_pair = {
        p["name"]: killzone_filter.windows_label(p)
        for p in pairs_to_run
    }
    meta = {
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "regime": regime,
        "regime_label": _regime_label_for(regime, start, end),
        "mode": "h1_only",
        "pairs": pair_names,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        # News filter metadata. Coverage is per-source. Blocked trades are
        # excluded from every aggregate metric in the report but kept in
        # Excel for audit.
        "news_coverage":        news_coverage,
        "news_events_fetched":  len(news_events),
        "news_blocked_rows":    n_blocked_total,
        "news_window_minutes":  30,
        # IST trading-window gate. Alerts outside the user's IST window
        # are excluded from aggregates; rows kept in Excel for audit so
        # the user can decide whether to shift their trading hours.
        "ist_blocked_rows":     n_ist_blocked_total,
        "ist_window_forex":     ist_window.window_label("forex"),
        "ist_window_index":     ist_window.window_label("index"),
        # Killzone hard filter. Alerts outside the pair's configured
        # killzone never enter the simulator -- they are not in `all_trades`,
        # not in trades.xlsx, not in any aggregate. These two fields are
        # the only place the count is preserved.
        "killzone_dropped_alerts":   n_killzone_dropped_total,
        "killzone_drops_by_pair":    killzone_drops_by_pair,
        "killzone_windows_by_pair":  killzone_windows_by_pair,
    }
    report_dir = h1_only_reporting.write_h1_only_report(
        run_id, all_trades, all_alerts, meta, risk_usd=risk_usd,
    )
    log_event("report_written", path=str(report_dir),
              total_alerts=len(all_alerts), total_trades=len(all_trades))
    print(f"\nH1-only report written to {report_dir}")

    # Update cross-run registry BEFORE commit so registry.json + BACKTEST_LOG.md
    # land in the same commit as the run's log files. If registry update runs
    # AFTER commit, its output files are unstaged when the push retries, which
    # causes `git rebase` to fail with "cannot rebase: You have unstaged
    # changes" and the run logs never reach GitHub (see May 2026 incident).
    try:
        from backtest.update_registry import build_registry
        build_registry(target_run_id=report_dir.name)
    except Exception as e:
        print(f"  [registry update skipped: {e}]")
        log_event("registry_update_failed", level="warn", error=str(e))

    # Push logs to GitHub BEFORE the email goes out. The email is the user's
    # signal that a run completed; if they get the email but the logs aren't
    # on GitHub, the run is unrecoverable (see March 2026 incident). Order
    # matters: persistence first, notification second. If push fails the
    # backtest exits non-zero and no email is sent -- user re-runs.
    from backtest.commit_logs import commit_run_logs, LogCommitError
    try:
        sha = commit_run_logs(report_dir, _REPO_ROOT, push=True)
        print(f"  [log commit OK] {report_dir.name} -> {sha}")
        log_event("logs_pushed_to_github", run_id=report_dir.name, sha=sha)
    except LogCommitError as e:
        log_event("logs_push_failed", level="error",
                  run_id=report_dir.name, error=str(e))
        print(f"\n!!! LOG COMMIT FAILED for {report_dir.name} !!!")
        print(f"    Reason: {e}")
        print(f"    Email NOT sent. Fix the push problem and re-run.")
        raise

    if send_email:
        try:
            reporting_email.send_report(report_dir, subject_suffix="(h1_only)")
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
    ap.add_argument("--regime", default="auto", choices=["auto", "war", "bau"],
                    help="auto reads backtest/WAR_REGIME_WEEKS.json; explicit war/bau overrides")
    ap.add_argument("--risk-usd", type=float, default=250.0)
    ap.add_argument("--email", action="store_true", help="Send report email")
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    out_dir = run(start, end, pairs, regime=args.regime, risk_usd=args.risk_usd,
                  send_email=args.email)

    # Registry update + log persistence to GitHub happen INSIDE _run_h1_only,
    # before the email is sent. If we reached this point, both succeeded
    # (otherwise _run_h1_only would have raised and main() would have exited
    # non-zero). Order: registry, persist, email. See March/May 2026 incidents.
    if out_dir is None:
        raise RuntimeError(
            "Backtest produced no output directory -- nothing to persist. "
            "This is a bug in the run() path; investigate before re-running."
        )


if __name__ == "__main__":
    main()
