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
from backtest.scanlog import emitter as scanlog_emitter
from backtest.scanlog import gates as scanlog_gates
import news_filter

RESULTS_ROOT = _REPO_ROOT / "backtest" / "results"
SCANLOG_ROOT = _REPO_ROOT / "backtest" / "out" / "scanlog"


def _git_sha() -> str:
    """Short git SHA of the repo, for the manifest. Empty string on failure."""
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


def _build_scanlog(run_id, start, end, pairs_to_run, dfs, risk_usd, fetch_pad_days):
    """Open the per-run ScanLog with a complete manifest (SPEC Â§2.1).

    Manifest is written FIRST, before any scanning. `dfs` maps pair name ->
    served H1 frame (already loaded). The proximity cap recorded per pair is the
    backtest atr_multiplier in force AFTER the override - the permanent record of
    the live-vs-backtest divergence.
    """
    pairs_served = []
    knobs = {}
    for p in pairs_to_run:
        name = p["name"]
        df = dfs.get(name)
        served = {
            "name": name,
            "symbol": p["symbol"],
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "served_start": (df.index.min().isoformat() if df is not None and not df.empty else None),
            "served_end": (df.index.max().isoformat() if df is not None and not df.empty else None),
            "n_bars": (int(len(df)) if df is not None else 0),
            "fingerprint": scanlog_emitter.fingerprint(df) if df is not None else "none",
            "prox_cap_atr": p.get("atr_multiplier"),
        }
        pairs_served.append(served)
        # Every Â§3 knob that can drift, read live per pair.
        knobs[f"{name}.atr_multiplier"] = p.get("atr_multiplier")
        knobs[f"{name}.distal_invalidation_mode"] = p.get("distal_invalidation_mode")
        knobs[f"{name}.spread_pips"] = p.get("spread_pips")
    manifest = scanlog_emitter.build_manifest(
        run_id=run_id, git_sha=_git_sha(), risk_usd=risk_usd,
        min_warmup_bars=50, pairs_served=pairs_served, knobs=knobs,
        fetch_pad_days=fetch_pad_days,
    )
    out_dir = SCANLOG_ROOT / run_id
    return scanlog_emitter.ScanLog.begin(out_dir, manifest), knobs


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
        send_email: bool = False,
        min_ob_range_atr: float = None) -> Path:
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
                            risk_usd, send_email, out_dir, run_id,
                            min_ob_range_atr=min_ob_range_atr)
    except Exception as e:
        log_event("run_fatal", level="error", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        log_event("run_end")
        logger.close()


def _run_h1_only(cfg, start, end, pair_names, regime, risk_usd, send_email,
                 out_dir, run_id, min_ob_range_atr=None):
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

    # Proximity cap = the LIVE config.json value (identical to live). The
    # previous BACKTEST_ATR_MULT override (3.0/3.5) made the backtest fire
    # alerts at a WIDER proximity than live (2.5/3.0), so the backtest saw more
    # / earlier alerts than live ever would. Per trader decision (2026-06-12)
    # the two must be identical. We now leave p["atr_multiplier"] at the config
    # value untouched, so replay_engine reads the same cap live does. Self-
    # syncing: change config.json and both move together.
    for p in pairs_to_run:
        live_mult = p.get("atr_multiplier")
        log_event("backtest_atr_cap", pair=p["name"],
                  pair_type=p.get("pair_type"),
                  live=live_mult, backtest=live_mult)

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

    # --- Pre-load every pair's H1 data BEFORE scanning so the scanlog manifest
    # can be written first (SPEC Â§2.1: an unrecorded run never executes). Data
    # is parquet-cached, so this pass is cheap. The manifest then captures the
    # served window + fingerprint per pair before a single bar is walked.
    dfs: dict = {}
    for pair_conf in pairs_to_run:
        dfs[pair_conf["name"]] = data_loader.load_bars(
            pair_conf["symbol"], "1h", fetch_start, end)
    scanlog, manifest_knobs = _build_scanlog(
        run_id, start, end, pairs_to_run, dfs, risk_usd,
        fetch_pad_days=35)
    print(f"  [scanlog] manifest written -> {scanlog.run_dir}")
    # YF_CLAMP: served history starts later than requested (yfinance 720d cap).
    # Recorded as a WARN condition so the gate table shows it, never hidden.
    for pair_conf in pairs_to_run:
        _df = dfs.get(pair_conf["name"])
        if _df is not None and not _df.empty and _df.index.min() > walk_start_ts:
            scanlog.condition("YF_CLAMP", pair=pair_conf["name"],
                              requested_start=str(walk_start_ts),
                              served_start=str(_df.index.min()))

    for pair_conf in pairs_to_run:
        name = pair_conf["name"]
        symbol = pair_conf["symbol"]
        print(f"\n=== {name} ({symbol}) ===")

        df_h1 = dfs.get(name)
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
            min_ob_range_atr=min_ob_range_atr,
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
        # Dedup: only the FIRST alert per (OB, entry_zone) is simulated.
        # The replay engine emits one alert per re-armed approach (state
        # machine), but the 2026-03 run showed the same OB firing 5-60
        # times and producing identical loser clones. Treat each OB as
        # one trade attempt per backtest -- once it fires, it's done.
        # See RCA #1, #7, #8.
        seen_obs: set = set()
        for alert in alerts_for_pair:
            ob_key = (
                (alert.get("ob") or {}).get("ob_timestamp"),
                (alert.get("ob") or {}).get("direction"),
            )
            if ob_key in seen_obs:
                log_event("ob_dedup_skip", pair=name,
                          alert_ts=str(alert["ts"]),
                          ob_ts=ob_key[0], direction=ob_key[1],
                          alert_seq=int(alert.get("alert_seq", 1)))
                scanlog.condition("DEDUP_SUPPRESSED", pair=name,
                                  ob_timestamp=ob_key[0], direction=ob_key[1])
                scanlog.event("alert_suppressed_dedup", pair=name,
                              ob_timestamp=ob_key[0], direction=ob_key[1])
                continue
            seen_obs.add(ob_key)
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
                # scanlog: fill + exit events with the full causality chain so
                # G1 (P&L reconciliation) and G3 (causality) can judge each
                # trade. r_if_exit_* are carried but tagged hypothetical (G6).
                if row.get("fill_ts"):
                    scanlog.event("trade_fill", pair=name,
                                  alert_ts=row.get("alert_ts"),
                                  fill_ts=row.get("fill_ts"),
                                  entry=row.get("entry"),
                                  entry_zone=row.get("entry_zone"))
                scanlog.event("trade_exit", pair=name,
                              alert_ts=row.get("alert_ts"),
                              ob_timestamp=row.get("ob_timestamp"),
                              bos_timestamp=row.get("bos_timestamp"),
                              fill_ts=row.get("fill_ts"),
                              exit_ts=row.get("exit_ts"),
                              exit_reason=row.get("exit_reason"),
                              r_realised=row.get("r_realised"),
                              pnl_usd=row.get("pnl_usd"),
                              r_if_exit_tp1=row.get("r_if_exit_tp1"),
                              r_if_exit_tp2=row.get("r_if_exit_tp2"),
                              hypothetical=True)
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

    # --- SCAN-LOG HARD GATE (SPEC Â§4) -------------------------------------
    # Evaluate every gate against the records this run produced, then decide
    # PASS/FAIL. The reporting headline (realised P&L) is reconciled against a
    # figure recomputed independently from r_realised (G1). A fresh live read
    # of the knobs detects mid-run drift (G5). On FAIL we STILL email - the
    # user asked to be told exactly what broke, not to fail silently - then we
    # raise so the process exits non-zero and the run is not trusted.
    health = _finalize_scanlog(
        scanlog, all_trades, risk_usd, report_dir, pairs_to_run,
        manifest_knobs)
    print(scanlog_gates.render_table(health, scanlog))
    if not health.passed:
        _email_scanlog_failure(report_dir, scanlog, health, start, end, regime)
        scanlog.close()
        _zip_scanlog(scanlog.run_dir)
        raise SystemExit(
            f"SCAN-LOG GATE FAILED for {run_id} - see run_health.json. "
            f"Failure email sent. Run NOT trusted (exit 1).")

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

    # PASS path: close + zip the scanlog (fast write during run, small on disk).
    scanlog.close()
    _zip_scanlog(scanlog.run_dir)

    return report_dir


def _read_realised_headline(report_dir) -> float:
    """The reporting layer's realised headline, summed across both scoreboards.
    G1 requires our independent r_realised figure to equal this. Returns 0.0 if
    summary.json is absent (the gate then reconciles only the per-row sum)."""
    p = report_dir / "summary.json"
    if not p.exists():
        return None
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
        boards = s.get("scoreboards", {})
        total = 0.0
        for key in ("proximal_realised", "fifty_pct_realised"):
            total += float(boards.get(key, {}).get("total_pnl_usd", 0.0))
        return round(total, 6)
    except Exception:
        return None


def _finalize_scanlog(scanlog, all_trades, risk_usd, report_dir,
                      pairs_to_run, manifest_knobs):
    """Re-read knobs live (G5 drift check), then evaluate every gate."""
    recheck = {}
    for p in pairs_to_run:
        recheck[f"{p['name']}.atr_multiplier"] = p.get("atr_multiplier")
        recheck[f"{p['name']}.distal_invalidation_mode"] = p.get("distal_invalidation_mode")
        recheck[f"{p['name']}.spread_pips"] = p.get("spread_pips")
    return scanlog_gates.evaluate(
        scanlog=scanlog, trades=all_trades, risk_usd=risk_usd,
        reported_headline_usd=_read_realised_headline(report_dir),
        manifest_recheck_knobs=recheck,
    )


def _zip_scanlog(run_dir) -> None:
    """Compress the scan log artifacts at run end (SPEC Â§8: compress, never
    sample). The .jsonl files zip ~10x; the report CLI reads either form."""
    import zipfile
    try:
        targets = [run_dir / "scan_log.jsonl", run_dir / "events.jsonl"]
        for t in targets:
            if not t.exists():
                continue
            zp = t.with_suffix(t.suffix + ".gz.zip")
            with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(t, arcname=t.name)
            t.unlink()  # keep only the compressed copy
        print(f"  [scanlog] compressed artifacts in {run_dir}")
    except Exception as e:
        print(f"  [scanlog zip skipped: {e}]")


def _email_scanlog_failure(report_dir, scanlog, health, start, end, regime):
    """Send a plain failure email naming exactly what broke (user decision:
    a hard stop must still tell me what needs repairing, not fail silently)."""
    try:
        import os, smtplib
        from email.mime.text import MIMEText
        sender = os.environ.get("GMAIL_ADDRESS")
        password = os.environ.get("GMAIL_APP_PASSWORD")
        to = (os.environ.get("BACKTEST_EMAIL") or sender
              or "avinash.somjani98@gmail.com")
        failed = [g for g in health.gates if g.verdict == "FAIL"]
        lines = [
            f"BACKTEST SCAN-LOG GATE FAILED  ({start.date()} -> {end.date()}, {regime})",
            f"run: {report_dir.name}",
            "",
            "What broke (each line is a failed safety gate):",
        ]
        for g in failed:
            lines.append(f"  [{g.id}] {g.description}")
            lines.append(f"        observed: {g.observed}")
        nz = {k: v for k, v in scanlog.condition_counts.items()
              if v and k in ("ALERT_LOOKAHEAD_BLOCKED", "FILL_BEFORE_ALERT",
                             "TREND_CONTRADICTION", "ZONE_STATE_CONTRADICTION",
                             "CONFIG_DRIFT", "PNL_MISMATCH", "HEARTBEAT_GAP",
                             "TS_NOT_BOUNDARY", "TZ_NAIVE", "UNCLASSIFIED_CONDITION")}
        if nz:
            lines.append("")
            lines.append("Red-flag conditions hit:")
            for code, n in sorted(nz.items()):
                lines.append(f"  {code}: {n}")
        lines += ["", "The run was NOT trusted and NOT pushed. Fix and re-run.",
                  f"Full detail: {report_dir / 'run_health.json'}"]
        body = "\n".join(lines)
        if not sender or not password:
            print("  [scanlog failure email skipped: SMTP env not set]")
            print(body)
            return
        msg = MIMEText(body)
        msg["Subject"] = f"[BACKTEST FAILED] {report_dir.name} - scan-log gate"
        msg["From"] = sender
        msg["To"] = to
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, password)
            s.send_message(msg)
        print(f"  [scanlog failure email sent] -> {to}")
    except Exception as e:
        print(f"  [scanlog failure email error: {e}]")


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
    ap.add_argument("--min-ob-range-atr", type=float, default=None,
                    help="OB candidate range floor in ATR units. Omit to use the "
                         "live default (smc_detector.OB_MIN_RANGE_ATR_MULT). Set "
                         "e.g. 0.3/0.4/0.5 to retune the OB-size gate per run.")
    args = ap.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    out_dir = run(start, end, pairs, regime=args.regime, risk_usd=args.risk_usd,
                  send_email=args.email, min_ob_range_atr=args.min_ob_range_atr)

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
