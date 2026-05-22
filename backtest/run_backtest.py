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
from backtest import h1_only_simulator
from backtest import h1_only_reporting
from backtest.run_logger import RunLogger, log_event
import news_filter

PHASE3_PAIRS = {"NAS100", "GOLD"}
RESULTS_ROOT = _REPO_ROOT / "backtest" / "results"

# Modes:
#   auto      -- existing behaviour: Phase 2 / Phase 3 if M15/M5 available,
#                else falls back to the legacy h1_only single-entry simulator.
#   h1_only   -- new H1-only mode: NO scoring gate, fires dual entries
#                (proximal + 50% mean) per OB, reuses live OB detection and
#                liquidity-based TP logic, skips M15/M5 fetches entirely.
VALID_MODES = ("auto", "h1_only")


def _load_config() -> dict:
    with open(_REPO_ROOT / "config.json") as f:
        return json.load(f)


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def run(start: datetime, end: datetime, pair_names: list,
        regime: str = "unspecified", risk_usd: float = 250.0,
        send_email: bool = False, mode: str = "auto") -> Path:
    cfg = _load_config()

    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    # Initialise per-run logger as the first action. console.log + run_log.jsonl
    # land in the results folder so they ride along with the artifact upload.
    prefix = "h1only" if mode == "h1_only" else regime
    run_id = f"{prefix}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    out_dir = RESULTS_ROOT / run_id
    logger = RunLogger.init(out_dir)
    logger.event("run_start", regime=regime, mode=mode,
                 start=start.strftime("%Y-%m-%d"),
                 end=end.strftime("%Y-%m-%d"), pairs=pair_names,
                 risk_usd=risk_usd, send_email=send_email)

    try:
        if mode == "h1_only":
            return _run_h1_only(cfg, start, end, pair_names, regime,
                                risk_usd, send_email, out_dir, run_id)
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
    # Zone register: every OB ever seen during the walk, indexed by
    # (pair, ob_timestamp). Tracks the full lifecycle of each zone:
    # detected -> approached -> alerted -> scored -> traded/failed/mitigated.
    zone_register = {}

    walk_start_ts = pd.Timestamp(start)
    walk_end_ts = pd.Timestamp(end)

    # Up-front H1-only decision: if the walk window starts more than the
    # yfinance intraday limit (58 days) ago, M15/M5 cannot cover it. Force
    # H1-only mode for the entire run, for every pair, and skip the wasted
    # M15/M5 fetches. The per-pair m15_covers_window check below still runs
    # as a belt-and-braces guard for the boundary case (~58d window).
    YF_INTRADAY_LIMIT_DAYS = 58
    days_back_to_start = (datetime.now(timezone.utc) - start).days
    force_h1_only_for_run = days_back_to_start > YF_INTRADAY_LIMIT_DAYS
    if force_h1_only_for_run:
        log_event("h1_only_mode_forced", level="warn",
                  reason="window_start_beyond_yfinance_intraday_limit",
                  days_back=days_back_to_start,
                  limit_days=YF_INTRADAY_LIMIT_DAYS,
                  routed_to="_run_h1_only")
        print(f"\n[H1-ONLY MODE for ENTIRE RUN] start is {days_back_to_start}d ago "
              f"(yfinance intraday limit is {YF_INTRADAY_LIMIT_DAYS}d). "
              f"Routing to dual-entry H1-only simulator "
              f"(proximal + 50pct rows, entry_zone column).")
        # Whole window is beyond yfinance's M15 horizon. Delegate to the
        # dual-entry H1-only path so the report carries both entry zones and
        # the entry_zone column — instead of silently using the legacy
        # single-entry h1-only fallback further down.
        return _run_h1_only(cfg, start, end, pair_names, regime, risk_usd,
                            send_email, out_dir, run_id)

    for pair_conf in pairs_to_run:
        name = pair_conf["name"]
        symbol = pair_conf["symbol"]
        print(f"\n=== {name} ({symbol}) ===")

        df_h1 = data_loader.load_bars(symbol, "1h", fetch_start, end)
        if force_h1_only_for_run:
            df_m15 = None
            df_m5 = None
        else:
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
            log_event("h1_only_mode", level="warn", pair=name,
                      reason="m15_unavailable_for_walk_start",
                      walk_start=str(walk_start_ts),
                      m15_first=(str(df_m15.index.min()) if df_m15 is not None and not df_m15.empty else None))
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
            ob = event.get("ob", {}) or {}
            zone_id = (event["pair"], ob.get("ob_timestamp"))
            kind = event["kind"]
            if kind == "ob_seen":
                # First time this zone enters the active list. Register it.
                zone_register.setdefault(zone_id, {
                    "pair": event["pair"],
                    "ob_timestamp": ob.get("ob_timestamp"),
                    "direction": ob.get("direction"),
                    "bos_tag": ob.get("bos_tag"),
                    "bos_tier": ob.get("bos_tier"),
                    "proximal": ob.get("proximal_line"),
                    "distal": ob.get("distal_line"),
                    "first_seen_ts": str(event["ts"]),
                    "first_seen_price": event.get("current_price"),
                    "alerted": False,
                    "alert_ts": None,
                    "score": None,
                    "score_passed": None,
                    "breakdown": None,
                    "outcome": "detected_only",
                    "outcome_reason": "in_active_list_no_proximity",
                    "pnl_usd": None,
                    "r_realised": None,
                })
            elif kind == "ob_mitigated":
                z = zone_register.get(zone_id)
                if z is not None and z["outcome"] in ("detected_only", "approached"):
                    z["outcome"] = "mitigated"
                    z["outcome_reason"] = event.get("reason", "mitigated")
                    z["mitigated_ts"] = str(event["ts"])
            elif kind == "alert":
                alerts_for_pair.append(event)
                # Mark the zone as approached (proximity check passed).
                z = zone_register.setdefault(zone_id, {
                    "pair": event["pair"],
                    "ob_timestamp": ob.get("ob_timestamp"),
                    "direction": ob.get("direction"),
                    "bos_tag": ob.get("bos_tag"),
                    "bos_tier": ob.get("bos_tier"),
                    "proximal": ob.get("proximal_line"),
                    "distal": ob.get("distal_line"),
                    "first_seen_ts": str(event["ts"]),
                    "first_seen_price": event.get("current_price"),
                    "alerted": False,
                    "alert_ts": None,
                    "score": None,
                    "score_passed": None,
                    "breakdown": None,
                    "outcome": "detected_only",
                    "outcome_reason": "in_active_list_no_proximity",
                    "pnl_usd": None,
                    "r_realised": None,
                })
                z["approached"] = True
                z["approach_ts"] = str(event["ts"])
                z["approach_price"] = event.get("current_price")
                if z["outcome"] == "detected_only":
                    z["outcome"] = "approached"
                    z["outcome_reason"] = "score_pending"
                all_alerts.append({
                    "pair": event["pair"],
                    "ts": str(event["ts"]),
                    "ob_timestamp": ob.get("ob_timestamp"),
                    "direction": ob.get("direction"),
                    "bos_tag": ob.get("bos_tag"),
                    "bos_tier": ob.get("bos_tier"),
                })

        log_event("pair_alerts_scored", pair=name,
                  alerts=len(alerts_for_pair), mode=("h1_only" if h1_only_mode else ("phase3" if used_phase3 else "phase2")))
        print(f"  {name}: {len(alerts_for_pair)} would-be alerts")

        min_conf = pair_conf.get("min_confidence", 6.0)
        # Backtest calibration: gate disabled so every scored alert is simulated.
        # `passed` is still recorded per-zone so reports can bucket by would-have-passed.
        gate_enabled = False
        for alert in alerts_for_pair:
            # Use live smc_detector.run_scorecard (same path Phase2 uses) so
            # backtest scoring stays in lockstep with live as scoring evolves.
            score, breakdown = trade_simulator.score_alert_via_live(
                alert, pair_conf, df_h1, df_m15
            )
            passed = score >= min_conf
            alert_proceeds = passed if gate_enabled else True
            log_event("alert_scored", pair=name,
                      ts=str(alert["ts"]),
                      ob_ts=alert["ob"].get("ob_timestamp"),
                      direction=alert["ob"].get("direction"),
                      bos_tag=alert["ob"].get("bos_tag"),
                      bos_tier=alert["ob"].get("bos_tier"),
                      score=round(score, 2),
                      min_conf=min_conf,
                      passed=passed,
                      gate_enabled=gate_enabled,
                      proceeds=alert_proceeds,
                      breakdown={k: round(float(v), 2) for k, v in breakdown.items()})

            zone_id = (alert["pair"], alert["ob"].get("ob_timestamp"))
            zr = zone_register.get(zone_id)
            if zr is not None:
                zr["score"] = round(score, 2)
                zr["score_passed"] = passed
                zr["min_conf"] = min_conf
                zr["gate_enabled"] = gate_enabled
                zr["breakdown"] = {k: round(float(v), 2) for k, v in breakdown.items()}
                zr["alerted"] = alert_proceeds
                zr["alert_ts"] = str(alert["ts"]) if alert_proceeds else None

            if not alert_proceeds:
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
                    if trade is None:
                        log_event("trade_sim_none", level="warn", pair=name,
                                  alert_ts=str(alert["ts"]), model="phase3",
                                  reason="phase3_sim_returned_none")
                else:
                    trade = None
                    log_event("trade_sim_none", level="warn", pair=name,
                              alert_ts=str(alert["ts"]), model="phase3",
                              reason="m5_choch_not_fired_in_window")
            else:
                trade = trade_simulator.simulate_trade(
                    alert, pair_conf, df_h1, trigger, risk_usd=risk_usd
                )
                if trade is None:
                    log_event("trade_sim_none", level="warn", pair=name,
                              alert_ts=str(alert["ts"]), model="phase2",
                              reason="phase2_sim_returned_none")
            if trade:
                trade["score"] = score
                trade["score_breakdown"] = breakdown
                log_event("trade_simulated", pair=name,
                          alert_ts=str(alert["ts"]),
                          score=round(score, 2),
                          model=trade.get("model"),
                          exit_reason=trade.get("exit_reason"),
                          r_realised=trade.get("r_realised"),
                          pnl_usd=trade.get("pnl_usd"))
                all_trades.append(trade)
                if zr is not None:
                    zr["outcome"] = f"traded_{trade.get('exit_reason', 'unknown')}"
                    zr["outcome_reason"] = trade.get("exit_reason", "")
                    zr["pnl_usd"] = trade.get("pnl_usd")
                    zr["r_realised"] = trade.get("r_realised")
                    zr["model"] = trade.get("model")
                    zr["fill_ts"] = trade.get("fill_ts")
                    zr["exit_ts"] = trade.get("exit_ts")
            else:
                # Sim returned None — the granular reason was already logged
                # inside simulate_trade / simulate_phase3_trade via sim_none_detail.
                # Mark the zone as alerted but not traded.
                if zr is not None:
                    if h1_only_mode:
                        zr["outcome"] = "alerted_sim_none_h1only"
                    elif used_phase3 and df_m5 is not None and not df_m5.empty:
                        zr["outcome"] = "alerted_sim_none_phase3"
                    else:
                        zr["outcome"] = "alerted_sim_none_phase2"
                    # The specific reason is in run_log.jsonl under sim_none_detail.
                    zr["outcome_reason"] = "see_sim_none_detail_in_run_log"

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
    zones_list = list(zone_register.values())
    report_dir = reporting.write_report(
        run_id, all_trades, all_alerts, meta,
        risk_usd=risk_usd, zones=zones_list,
    )
    log_event("report_written", path=str(report_dir),
              total_alerts=len(all_alerts), total_trades=len(all_trades),
              total_zones_registered=len(zones_list))
    print(f"\nReport written to {report_dir}")

    if send_email:
        try:
            reporting_email.send_report(report_dir, subject_suffix=f"({regime})")
            log_event("email_sent", path=str(report_dir))
        except Exception as e:
            log_event("email_failed", level="error",
                      error=f"{type(e).__name__}: {e}")

    return report_dir


def _run_h1_only(cfg, start, end, pair_names, regime, risk_usd, send_email,
                 out_dir, run_id):
    """H1-only backtest run.

    Differences from _run_inner:
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

    state = replay_engine.ReplayState()
    all_alerts: list = []
    all_trades: list = []
    walk_start_ts = pd.Timestamp(start)
    walk_end_ts = pd.Timestamp(end)

    print(f"\n[H1-ONLY MODE] start={start.date()} end={end.date()} "
          f"pairs={pair_names} risk_per_trade=${risk_usd:.0f}")
    print(f"  No M15/M5 fetches. No scoring gate. Dual entry per OB.")

    # News blackout filter. Fetch FF (scheduled high-impact events) for the
    # full backtest range plus a 1-day pad on each side (events at the
    # range edges still need their ±30min window evaluated).
    # `start`/`end` from argparse via _parse_date are already tz-aware UTC,
    # so localize only when naive (defensive — never double-localize).
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
                  m15_rows=0, m5_rows=0,
                  h1_first=str(df_h1.index.min()),
                  h1_last=str(df_h1.index.max()))

        # Walk H1 bars and collect OB-touch alerts.
        alerts_for_pair = []
        for event in replay_engine.replay_pair(
            pair_conf, df_h1, df_m15=None, df_m5=None,
            state=state, walk_start_ts=walk_start_ts, walk_end_ts=walk_end_ts,
        ):
            if event["kind"] == "alert":
                alerts_for_pair.append(event)
                all_alerts.append({
                    "pair": event["pair"],
                    "ts": str(event["ts"]),
                    "ob_timestamp": (event.get("ob") or {}).get("ob_timestamp"),
                    "direction": (event.get("ob") or {}).get("direction"),
                    "bos_tag": (event.get("ob") or {}).get("bos_tag"),
                    "bos_tier": (event.get("ob") or {}).get("bos_tier"),
                })

        log_event("pair_alerts_collected", pair=name,
                  alerts=len(alerts_for_pair), mode="h1_only")
        print(f"  {name}: {len(alerts_for_pair)} OB-touch alerts")

        n_trades_for_pair = 0
        n_blocked_for_pair = 0
        for alert in alerts_for_pair:
            rows = h1_only_simulator.simulate_h1_only_dual(
                alert, pair_conf, df_h1, risk_usd=risk_usd,
            )
            # News blackout tagging. Option B: simulate every alert; tag
            # blocked rows so they appear in Excel for audit but are
            # excluded from every aggregate metric (handled in reporting).
            #
            # The blackout check uses the alert timestamp, not the fill
            # timestamp, because that is the moment we would (live) decide
            # whether to place the limit order.
            alert_ts = alert["ts"]
            if not isinstance(alert_ts, pd.Timestamp):
                alert_ts = pd.Timestamp(alert_ts)
            if alert_ts.tzinfo is None:
                alert_ts = alert_ts.tz_localize("UTC")
            blocked, src_event = news_filter.is_news_blackout(
                alert_ts.to_pydatetime(), name, news_events,
                window_minutes=30,
            )
            for row in rows:
                row["news_blocked"]        = bool(blocked)
                row["news_event_title"]    = src_event["title"]    if blocked else ""
                row["news_event_currency"] = src_event["currency"] if blocked else ""
                row["news_event_source"]   = src_event["source"]   if blocked else ""
                row["news_event_ts"]       = src_event["ts_utc"].isoformat() if blocked else ""
                all_trades.append(row)
                n_trades_for_pair += 1
                if blocked:
                    n_blocked_for_pair += 1
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
                          news_event_title=row.get("news_event_title"),
                          news_event_source=row.get("news_event_source"))

        log_event("pair_trades_simulated", pair=name,
                  trades=n_trades_for_pair,
                  news_blocked=n_blocked_for_pair)
        print(f"  {name}: {n_trades_for_pair} simulated trade rows "
              f"(2 per qualified OB-touch; {n_blocked_for_pair} news-blocked)")

    n_blocked_total = sum(1 for t in all_trades if t.get("news_blocked"))
    meta = {
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "regime": regime,
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
    }
    report_dir = h1_only_reporting.write_h1_only_report(
        run_id, all_trades, all_alerts, meta, risk_usd=risk_usd,
    )
    log_event("report_written", path=str(report_dir),
              total_alerts=len(all_alerts), total_trades=len(all_trades))
    print(f"\nH1-only report written to {report_dir}")

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
    ap.add_argument("--regime", default="unspecified", choices=["war", "bau", "unspecified"])
    ap.add_argument("--mode", default="h1_only", choices=list(VALID_MODES),
                    help="h1_only = H1-only dual-entry, no scoring gate (default); "
                         "auto = Phase 2/3 if M15/M5 available (legacy)")
    ap.add_argument("--risk-usd", type=float, default=250.0)
    ap.add_argument("--email", action="store_true", help="Send report email")
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    out_dir = run(start, end, pairs, regime=args.regime, risk_usd=args.risk_usd,
                  send_email=args.email, mode=args.mode)

    # Auto-update cross-run registry after every run.
    try:
        from backtest.update_registry import build_registry
        if out_dir is not None:
            build_registry(target_run_id=out_dir.name)
    except Exception as e:
        print(f"  [registry update skipped: {e}]")


if __name__ == "__main__":
    main()
