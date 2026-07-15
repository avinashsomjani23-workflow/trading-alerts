"""CLI entry point for backtest runs.

Usage:
    python backtest/run_backtest.py --start 2025-09-15 --end 2025-09-19 --regime war
    python backtest/run_backtest.py --start 2025-09-15 --end 2025-09-19 --pairs EURUSD,GOLD
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from backtest.run_logger import RunLogger, log_event, STALL_GAP_S
from backtest.scanlog import emitter as scanlog_emitter
from backtest.scanlog import gates as scanlog_gates
import news_filter

RESULTS_ROOT = _REPO_ROOT / "backtest" / "results"
SCANLOG_ROOT = _REPO_ROOT / "backtest" / "out" / "scanlog"


def _git_sha() -> str:
    """Short git SHA of the repo at run time, with a '-dirty' suffix when the
    working tree has uncommitted changes. The dirty flag matters for provenance:
    a clean SHA on a dirty tree is a FALSE alibi — it would say the run used
    committed code when it actually ran with edits on top (the exact ambiguity
    that made the pre-gate-removal 18-yr run un-attributable). Empty string on
    failure. Stamped into BOTH the scanlog manifest and summary.json (single
    source — one _git_sha() call feeds meta['code_sha'])."""
    import subprocess
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if not sha:
            return ""
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
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
        # Clamped replay skips any bar with < LIVE_DETECTION_BARS history
        # (replay_engine._min_warmup = detection_bars). The manifest must record
        # that real floor, not the legacy unclamped 50-bar value.
        min_warmup_bars=replay_engine.smc_radar.LIVE_DETECTION_BARS,
        pairs_served=pairs_served, knobs=knobs,
        fetch_pad_days=fetch_pad_days,
    )
    out_dir = SCANLOG_ROOT / run_id
    return scanlog_emitter.ScanLog.begin(out_dir, manifest), knobs


def _process_pair(pair_conf, df_h1, walk_start_ts, walk_end_ts,
                  news_events, risk_usd, exit_lab_configs=None,
                  scanlog_worker_dir=None) -> dict:
    """Run one pair's full replay + simulation in an isolated worker process.

    Called by ProcessPoolExecutor — must be a top-level function so Python can
    pickle it. Each worker process gets its own memory: the global scanlog and
    run_logger state are isolated, so concurrent writes never collide.

    `exit_lab_configs` (when passed) arms the simulator's exit-lab side-channel
    INSIDE this worker. The pool uses spawn (Windows + py3.14), so a sink set in
    the parent is never seen here — the worker must arm its OWN sink and RETURN
    it. The side-channel stays pure (it never touches r_realised or the trade
    row); it only replays alternative exit recipes over the same post-fill bars.

    Returns a dict with alerts, trades, per-pair tallies, and (when armed) the
    worker's exit-lab sink rows for the main process to merge.
    """
    # Worker-local imports (each process re-imports these; no shared state).
    import pandas as _pd
    from pathlib import Path as _Path
    from backtest import replay_engine as _re, h1_only_simulator as _sim
    from backtest import ist_window as _ist
    from backtest import killzone as _kz
    from backtest.run_logger import RunLogger as _RL, BufferRunLogger as _BRL
    from backtest.scanlog import emitter as _sle
    import news_filter as _nf

    name = pair_conf["name"]
    pair_type = pair_conf.get("pair_type", "forex")

    # Arm worker-local observability (2026-07-02 fix). A spawned worker has no
    # RunLogger and no active ScanLog, so every log_event / scanlog emit from
    # the replay + simulator was silently dropped — the committed run_log had
    # no funnel/skip events and the scan-log gates judged zero records. The
    # buffer logger collects events for the parent to replay; the worker
    # ScanLog writes real records into its own subdir for the parent to merge.
    _buf = _BRL(pair=name)
    _RL._instance = _buf
    _wsl = None
    if scanlog_worker_dir:
        _wsl = _sle.ScanLog.begin_worker(_Path(scanlog_worker_dir))

    # Arm the exit-lab side-channel in THIS worker (spawn => parent globals don't
    # reach us). Pure observe: replays each recipe over the same post-fill bars,
    # appends per-recipe R to the local sink, never touches r_realised.
    exit_lab_sink: list = []
    if exit_lab_configs:
        _sim.EXIT_LAB_CONFIGS = exit_lab_configs
        _sim.EXIT_LAB_SINK = exit_lab_sink

    pair_alerts = []
    pair_trades = []
    n_trades = 0
    n_blocked = 0
    n_ist_blocked = 0
    n_kz_dropped = 0

    state = _re.ReplayState()
    for event in _re.replay_pair(
        pair_conf, df_h1,
        state=state, walk_start_ts=walk_start_ts, walk_end_ts=walk_end_ts,
    ):
        if event["kind"] == "alert":
            pair_alerts.append(event)

    print(f"  {name}: {len(pair_alerts)} OB-touch alerts")

    # 2026-07-15: seen_obs first-touch dedupe REMOVED. Every re-armed re-touch is
    # a real, spaced re-approach (re-arm hysteresis, replay_engine.py:519 — price
    # must clear (prox_cap + REARM_EXTRA_ATR)xATR and return), NOT a same-bar clone
    # (proven 0 clones on 2019H1/2016H1 samples). A mitigated OB is already dropped
    # UPSTREAM before it can fire (replay_engine.py:350-363: 3rd proximal touch OR
    # close beyond distal), so re-fires can only occur while the zone is alive. The
    # backtest now trades every touch until mitigation; touches_at_alert (frozen at
    # the yield) is the row-slice lever for later capping how many touches to trade.
    # The old 2026-03 "cloned-fill" RCA is guarded by the re-arm state machine, not
    # by this dedupe — see tests/test_retouch_trading.py.
    for alert in pair_alerts:
        alert_ts_raw = alert["ts"]
        if not isinstance(alert_ts_raw, _pd.Timestamp):
            alert_ts_raw = _pd.Timestamp(alert_ts_raw)
        if alert_ts_raw.tzinfo is None:
            alert_ts_raw = alert_ts_raw.tz_localize("UTC")

        killzone_blocked = not _kz.in_pair_killzone(alert_ts_raw, pair_conf)
        if killzone_blocked:
            n_kz_dropped += 1

        # Remember where this alert's exit-lab rows START so we can stamp the
        # hard-block flags onto exactly the rows this alert produced (the sink is
        # a shared module list appended to across every alert). Without this the
        # sink keeps IST/weekend-blocked fills the headline throws out, so the
        # exit table's N over-counts vs the headline (the 826-vs-668 RCA).
        _sink_mark = len(exit_lab_sink)
        rows = _sim.simulate_h1_only_dual(alert, pair_conf, df_h1, risk_usd=risk_usd)

        alert_ts = alert_ts_raw
        blocked, src_event = _nf.is_news_blackout(
            alert_ts.to_pydatetime(), name, news_events, window_minutes=30,
        )
        ist_blocked = not _ist.in_user_trading_window(alert_ts, pair_type)

        # Stamp the same hard-block flags on this alert's exit-lab sink rows so the
        # exit table can drop exactly what the headline drops. weekend_blocked is
        # already on the trade row (per entry_zone); reuse it by (entry_zone) match.
        _wk_by_zone = {r.get("entry_zone"): bool(r.get("weekend_blocked"))
                       for r in rows}
        for _sr in exit_lab_sink[_sink_mark:]:
            _sr["ist_blocked"] = bool(ist_blocked)
            _sr["weekend_blocked"] = _wk_by_zone.get(_sr.get("entry_zone"), False)

        for row in rows:
            row["news_blocked"]        = bool(blocked)
            row["news_event_title"]    = src_event["title"]    if blocked else ""
            row["news_event_currency"] = src_event["currency"] if blocked else ""
            row["news_event_source"]   = src_event["source"]   if blocked else ""
            row["news_event_ts"]       = src_event["ts_utc"].isoformat() if blocked else ""
            row["ist_blocked"]         = bool(ist_blocked)
            row["killzone_blocked"]    = bool(killzone_blocked)
            row["killzone_windows"]    = _kz.windows_label(pair_conf)
            row["alert_utc_hour"]      = int(alert_ts.hour)
            pair_trades.append(row)
            n_trades += 1
            if blocked:
                n_blocked += 1
            if ist_blocked:
                n_ist_blocked += 1

    print(f"  {name}: {n_trades} simulated trade rows "
          f"({n_blocked} news-blocked, {n_ist_blocked} IST-blocked, "
          f"{n_kz_dropped} killzone-dropped alerts)")

    # Return plain dicts only — no pandas objects, no live-module references.
    # ProcessPoolExecutor pickles the return value back to the main process.
    alert_rows = []
    for event in pair_alerts:
        alert_rows.append({
            "pair": event["pair"],
            "ts": str(event["ts"]),
            "alert_bar_ts": (str(event["alert_bar_ts"])
                             if event.get("alert_bar_ts") is not None else None),
            "alert_seq": int(event.get("alert_seq", 1)),
            "ob_timestamp": (event.get("ob") or {}).get("ob_timestamp"),
            "bos_timestamp": (event.get("ob") or {}).get("bos_timestamp"),
            "direction": (event.get("ob") or {}).get("direction"),
            "bos_tag": (event.get("ob") or {}).get("bos_tag"),
            "bos_tier": (event.get("ob") or {}).get("bos_tier"),
        })

    # Disarm + harvest the side-channel so the sink rides home in the pickled
    # return (the parent's global was never the one we appended to).
    if exit_lab_configs:
        _sim.EXIT_LAB_CONFIGS = None
        _sim.EXIT_LAB_SINK = None

    # Close + summarise worker observability for the parent merge.
    _scanlog_summary = None
    if _wsl is not None:
        _wsl.close()
        _scanlog_summary = _wsl.worker_summary()

    return {
        "name": name,
        "alerts": alert_rows,
        "trades": pair_trades,
        "n_trades": n_trades,
        "n_blocked": n_blocked,
        "n_ist_blocked": n_ist_blocked,
        "n_kz_dropped": n_kz_dropped,
        "exit_lab_sink": exit_lab_sink,
        "scanlog_summary": _scanlog_summary,
        "log_records": _buf.records,
    }


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
        exit_lab_sink_out: list = None) -> Path:
    """Run an H1-only backtest. If `exit_lab_sink_out` is a list, the merged
    exit-lab side-channel rows (every recipe replayed per fill) are appended to
    it for a diagnostic caller (exit_lab.py) -- the email already gets the same
    data internally, so normal callers leave this None."""
    cfg = _load_config()

    # NAS100 is excluded ENTIRELY from the backtest (2026-06-30 trader decision).
    # Hard-drop it here so NO caller -- CLI default, cron, or any list passed in
    # -- can generate a NAS100 trade row. Removing it run-level (not just at the
    # reporting split) guarantees it never reaches a CSV, Excel, summary, or
    # email. Other diagnostics that still want it pass their own list to a
    # different entry point; the live backtest does not trade it.
    pair_names = [p for p in pair_names if p != "NAS100"]

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
                            exit_lab_sink_out=exit_lab_sink_out)
    except Exception as e:
        log_event("run_fatal", level="error", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        # Honest timing: wall clock vs active compute. A big idle_s/max_gap_s
        # means the machine was frozen (laptop asleep, CI paused) -- that
        # inflates wall time but never touches the data. Budget future runs on
        # active_min, NOT wall_min (which lies when the box slept mid-run).
        try:
            _t = logger.timing_summary()
            log_event("run_end", **_t)
            if _t["idle_s"] > STALL_GAP_S:
                print(f"  [timing] WALL {_t['wall_min']}min but ACTIVE only "
                      f"{_t['active_min']}min -- machine was FROZEN "
                      f"{round(_t['idle_s']/60,1)}min (longest freeze "
                      f"{round(_t['max_gap_s']/60,1)}min). Data unaffected; "
                      f"budget on active_min.")
        except Exception:
            log_event("run_end")
        logger.close()


def _run_h1_only(cfg, start, end, pair_names, regime, risk_usd, send_email,
                 out_dir, run_id, exit_lab_sink_out=None):
    """H1-only backtest run.

      - H1 data only (MT5/FundingPips parquet cache). No M15/M5 fetches.
      - No scoring gate at detection (every OB-touch fires); the live score
        floor is applied later by the reporting/headline layer.
      - Fires ONE trade row per OB-touch: the proximal entry (the live limit),
        via h1_only_simulator.simulate_h1_only_dual. Proximal is the only entry
        zone (the 50% mean leg was removed 2026-07).
      - Uses h1_only_reporting for the TP1/TP2 scoreboard.
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
    # News is INFORMATION-ONLY here (it never gates an alert), and the FairEconomy
    # historical calendar URL now 404s on old weeks -- fetching it on a multi-year
    # run just burns time. Set BACKTEST_SKIP_NEWS=1 to skip the fetch and run with
    # zero events (used for long break-quality / exit-lab runs).
    import os as _os
    if _os.environ.get("BACKTEST_SKIP_NEWS", "").strip().lower() in ("1", "true", "yes"):
        news_data = {"events": [], "coverage": {"ff": False}}
        print("  News: SKIPPED (BACKTEST_SKIP_NEWS set)")
    else:
        news_data = news_filter.fetch_events(
            news_start.to_pydatetime(), news_end.to_pydatetime(),
            sources=("ff",),
        )
        print(f"  News: {len(news_data['events'])} High-impact events fetched "
              f"(coverage: {news_data['coverage']})")
    news_events = news_data["events"]
    news_coverage = news_data["coverage"]
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
    # YF_CLAMP (legacy condition name): served history starts later than
    # requested -- now because the MT5/FundingPips cache doesn't reach that far
    # back, not the old yfinance 720d cap. Recorded as a WARN condition so the
    # gate table shows it, never hidden. (Identifier kept for scanlog schema
    # stability; the cause is the cache extent.)
    for pair_conf in pairs_to_run:
        _df = dfs.get(pair_conf["name"])
        if _df is not None and not _df.empty and _df.index.min() > walk_start_ts:
            scanlog.condition("YF_CLAMP", pair=pair_conf["name"],
                              requested_start=str(walk_start_ts),
                              served_start=str(_df.index.min()))

    # Log data-load stats for every pair before launching workers.
    for pair_conf in pairs_to_run:
        name = pair_conf["name"]
        df_h1 = dfs.get(name)
        if df_h1 is None or df_h1.empty:
            log_event("pair_skip", level="warn", pair=name, reason="h1_unavailable")
            print(f"  [SKIP] H1 unavailable for {name}")
        else:
            log_event("pair_data_loaded", pair=name, h1_rows=len(df_h1),
                      h1_first=str(df_h1.index.min()),
                      h1_last=str(df_h1.index.max()))

    # Run all pairs concurrently. Each pair is independent: ReplayState is
    # per-pair, scanlog/run_logger globals live in isolated worker memory
    # (ProcessPoolExecutor forks separate processes, not threads). Wall time
    # drops from sum(pairs) to max(slowest_pair). CI runner is 2-core so we
    # cap workers at min(6, cpu_count) to avoid over-subscribing.
    import os as _os
    n_workers = min(len(pairs_to_run), (_os.cpu_count() or 2))
    valid_pairs = [(p, dfs[p["name"]]) for p in pairs_to_run
                   if dfs.get(p["name"]) is not None and not dfs[p["name"]].empty]

    print(f"\n  [parallel] launching {len(valid_pairs)} pairs "
          f"across {n_workers} workers")

    # Exit-lab recipe set, armed in every worker so the email's exit section has
    # the full per-recipe data (the same recipes the standalone exit_lab studies).
    # One canonical source -- imported, not redefined here. The side-channel is
    # pure observe; it adds replay work per fill but never changes r_realised.
    from backtest.diagnostics.exit_lab import CONFIGS as _EXIT_LAB_CONFIGS

    pair_results: dict = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _process_pair,
                pair_conf, df_h1, walk_start_ts, walk_end_ts,
                news_events, risk_usd, _EXIT_LAB_CONFIGS,
                # Per-pair worker scanlog dir: the worker writes real scan/event
                # records there; the parent merges them below (2026-07-02 fix —
                # workers previously emitted into the NullScanLog and the gate
                # layer judged zero records).
                str(scanlog.run_dir / "workers" / pair_conf["name"]),
            ): pair_conf["name"]
            for pair_conf, df_h1 in valid_pairs
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                pair_results[name] = fut.result()
            except Exception as e:
                log_event("pair_worker_error", level="error", pair=name,
                          error=f"{type(e).__name__}: {e}")
                print(f"  [ERROR] {name} worker failed: {e}")

    # Merge results back into main-process accumulators in config order so
    # report output is deterministic regardless of which pair finished first.
    exit_lab_sink: list = []
    for pair_conf in pairs_to_run:
        name = pair_conf["name"]
        res = pair_results.get(name)
        if res is None:
            killzone_drops_by_pair[name] = 0
            continue

        all_alerts.extend(res["alerts"])
        all_trades.extend(res["trades"])
        exit_lab_sink.extend(res.get("exit_lab_sink") or [])
        killzone_drops_by_pair[name] = res["n_kz_dropped"]

        # Fold the worker's observability back in (config-pair order, so the
        # merged artifacts and the G7 stamp are deterministic): scanlog records
        # + counters feed G2/G4/G8; buffered log events land in run_log.jsonl.
        if res.get("scanlog_summary"):
            scanlog.merge_worker(res["scanlog_summary"])
        _lg = RunLogger.get()
        if _lg is not None:
            for _rec in (res.get("log_records") or []):
                _lg.write_raw(_rec)

        log_event("pair_alerts_collected", pair=name,
                  alerts=len(res["alerts"]), mode="h1_only")
        log_event("pair_trades_simulated", pair=name,
                  trades=res["n_trades"],
                  news_blocked=res["n_blocked"],
                  ist_blocked=res["n_ist_blocked"],
                  killzone_dropped_alerts=res["n_kz_dropped"])

        # Replay scanlog events from worker results back into the main scanlog
        # so the audit trail (trade_fill, trade_exit) is complete.
        for row in res["trades"]:
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

    # Hand the merged exit-lab sink to a diagnostic caller (exit_lab.py) if one
    # asked for it. The email gets the same rows via write_h1_only_report below.
    if exit_lab_sink_out is not None:
        exit_lab_sink_out.extend(exit_lab_sink)

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
        # Code provenance: short HEAD SHA (+ '-dirty' if the tree had uncommitted
        # edits at run time). Flows into summary.json so any results file can be
        # traced to the exact code that produced it WITHOUT reverse-engineering it
        # from row data. Single source: same _git_sha() the scanlog manifest uses.
        "code_sha": _git_sha(),
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
        # Killzone gate. Alerts outside the pair's configured killzone ARE
        # simulated and kept in `all_trades` tagged killzone_blocked=True (so
        # the audit section can show their would-have R), but are EXCLUDED from
        # every aggregate metric by the reporting layer -- same pattern as the
        # news and IST gates. (Earlier comment here wrongly said they never
        # enter the simulator.)
        "killzone_dropped_alerts":   n_killzone_dropped_total,
        "killzone_drops_by_pair":    killzone_drops_by_pair,
        "killzone_windows_by_pair":  killzone_windows_by_pair,
    }
    report_dir = h1_only_reporting.write_h1_only_report(
        run_id, all_trades, all_alerts, meta, risk_usd=risk_usd,
        exit_lab_sink=exit_lab_sink,
    )
    log_event("report_written", path=str(report_dir),
              total_alerts=len(all_alerts), total_trades=len(all_trades))
    print(f"\nH1-only report written to {report_dir}")

    # Persist the EXACT in-memory exit-lab sink the email consumed, one JSON
    # object per line. This is the ONLY faithful on-disk copy: exit_lab_trades.csv
    # is a SEPARATE diagnostic (diagnostics/exit_lab.py) with a different schema
    # and no ob_timestamp/entry_zone, so it cannot reconstruct the sink's join
    # key. The email preview harness (render_report.py) reads THIS file to render
    # byte-identical recipe numbers without re-running the 25-min simulation.
    try:
        _sink_path = report_dir / "exit_lab_sink.jsonl"
        with open(_sink_path, "w", encoding="utf-8") as _sf:
            for _row in exit_lab_sink:
                _sf.write(json.dumps(_row, default=str) + "\n")
        log_event("exit_lab_sink_persisted", path=str(_sink_path),
                  rows=len(exit_lab_sink))
    except Exception as _e:
        log_event("exit_lab_sink_persist_failed", level="warn",
                  error=f"{type(_e).__name__}: {_e}")

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

    # Replace the Act-6 <!--GATES--> token in each report HTML with a one-line-
    # per-gate PASS/FAIL table. Gates read the report's own headline (G1), so
    # they can only run AFTER the report is written — hence the token + this
    # post-gate patch. Runs on BOTH the FAIL and PASS paths so the emailed report
    # (we email even on FAIL) always carries the real gate verdicts.
    _inject_gates_html(report_dir, health)
    if not health.passed:
        _email_scanlog_failure(report_dir, scanlog, health, start, end, regime)
        scanlog.close()
        _zip_scanlog(scanlog.run_dir)
        raise SystemExit(
            f"SCAN-LOG GATE FAILED for {run_id} - see run_health.json. "
            f"Failure email sent. Run NOT trusted (exit 1).")

    # PASS path: finalize the scan log NOW (close + zip) so its audit artifacts
    # -- run_health.json, manifest.json, and the zipped scan_log/events -- are
    # on disk and STABLE before commit_run_logs runs. commit_run_logs commits
    # them alongside the result logs so every emailed run keeps its audit trail.
    # Previously the scan log was zipped only AFTER commit + email, so it was
    # never persisted and the next run overwrote it (the reason no emailed run
    # had a fetchable scan log). Zipping first also leaves the tracked
    # out/scanlog tree clean for the push rebase.
    scanlog.close()
    _zip_scanlog(scanlog.run_dir)

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
    # Auto-PUSH only in GitHub Actions, where it is the ONLY way a CI run's
    # results persist. On a local machine we COMMIT but do NOT push: the local
    # repo lives inside OneDrive, and git's rapid file churn during a push-rebase
    # collides with OneDrive's sync (a held file lock), which left the repo in a
    # broken half-rebase state (2026-06 incident). The result logs are still
    # committed locally every run (audit trail intact); the user pushes when
    # ready. Set BACKTEST_PUSH=1 to force a push from a local run anyway.
    import os as _os
    _in_ci = _os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    _force_push = _os.environ.get("BACKTEST_PUSH", "").strip() in ("1", "true", "yes")
    _do_push = _in_ci or _force_push

    from backtest.commit_logs import commit_run_logs, LogCommitError
    try:
        sha = commit_run_logs(report_dir, _REPO_ROOT, push=_do_push)
        if _do_push:
            print(f"  [log commit + push OK] {report_dir.name} -> {sha}")
            log_event("logs_pushed_to_github", run_id=report_dir.name, sha=sha)
        else:
            print(f"  [log commit OK — LOCAL ONLY, not pushed] {report_dir.name} -> {sha}")
            print(f"    To publish to GitHub when ready: git push origin main")
            log_event("logs_committed_local_only", run_id=report_dir.name, sha=sha)
    except LogCommitError as e:
        log_event("logs_push_failed", level="error",
                  run_id=report_dir.name, error=str(e))
        print(f"\n!!! LOG COMMIT FAILED for {report_dir.name} !!!")
        print(f"    Reason: {e}")
        _what = "push" if _do_push else "local commit"
        print(f"    Email NOT sent. Fix the {_what} problem and re-run.")
        raise

    if send_email:
        try:
            reporting_email.send_report(report_dir, subject_suffix="(h1_only)")
            log_event("email_sent", path=str(report_dir))
        except Exception as e:
            log_event("email_failed", level="error",
                      error=f"{type(e).__name__}: {e}")

    # Scan log was already closed + zipped above (before commit) so its audit
    # artifacts ride in the same commit as the result logs.
    return report_dir


def _inject_gates_html(report_dir, health) -> None:
    """Replace the <!--GATES--> placeholder in each report HTML with a compact
    PASS/FAIL gate table built from the in-process HealthResult. Idempotent and
    best-effort: a missing token or missing file is a silent no-op (never break
    the run over a cosmetic patch)."""
    try:
        overall_ok = health.overall == scanlog_gates.PASS
        total = len(health.gates)
        # On a clean run, nobody reads a wall of green — one line is enough.
        # Only surface the gates that actually broke (FAIL or WARN).
        if overall_ok:
            table = (
                "<p style='font-size:13px;margin-bottom:6px;color:#0ca30c;'>"
                "<b>&#10003; All %d integrity gates passed.</b></p>" % total)
        else:
            failed = [g for g in health.gates
                      if str(g.verdict).upper() not in ("PASS", "OK")]
            rows = ""
            for g in failed:
                vcolor = ("#fab219" if str(g.verdict).upper() == "WARN"
                          else "#d03b3b")
                rows += (
                    "<tr><td style='padding:3px 8px;border-bottom:1px solid #e1e0d9;'>"
                    "<b>%s</b></td>"
                    "<td style='padding:3px 8px;border-bottom:1px solid #e1e0d9;'>%s</td>"
                    "<td style='padding:3px 8px;border-bottom:1px solid #e1e0d9;"
                    "color:%s;font-weight:700;'>%s</td></tr>" % (
                        g.id, g.description, vcolor, g.verdict))
            head = ("<span style='font-weight:700;color:#d03b3b;'>"
                    "GATE FAILURE — run not trusted (%d of %d gates)</span>" % (
                        len(failed), total))
            table = (
                "<p style='font-size:13px;margin-bottom:6px;'>%s</p>"
                "<table style='width:100%%;border-collapse:collapse;font-size:12px;'>"
                "<thead><tr>"
                "<th style='text-align:left;padding:3px 8px;'>Gate</th>"
                "<th style='text-align:left;padding:3px 8px;'>Check</th>"
                "<th style='text-align:left;padding:3px 8px;'>Result</th>"
                "</tr></thead><tbody>%s</tbody></table>" % (head, rows))
        for name in ("report_forex.html", "report_gold_nas.html"):
            fp = report_dir / name
            if not fp.exists():
                continue
            html = fp.read_text(encoding="utf-8")
            if "<!--GATES-->" in html:
                fp.write_text(html.replace("<!--GATES-->", table), encoding="utf-8")
    except Exception as e:
        log_event("gates_html_inject_failed", level="warn",
                  error=f"{type(e).__name__}: {e}")


def _read_realised_headline(report_dir) -> float:
    """The reporting layer's realised headline. PROXIMAL ONLY (2026-06-30): the
    report is now the proximal/live book alone — 50% mean entry is dead — so G1
    reconciles the proximal scoreboard against the proximal r_realised sum.
    Returns None if summary.json is absent (gate then reconciles per-row only)."""
    p = report_dir / "summary.json"
    if not p.exists():
        return None
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
        boards = s.get("scoreboards", {})
        return round(float(boards.get("proximal_realised", {})
                           .get("total_pnl_usd", 0.0)), 6)
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
                             "PHYS_IMPOSSIBLE_METRIC",
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
    ap.add_argument("--pairs",
                    default="EURUSD,NZDUSD,USDJPY,USDCHF,GOLD,"
                            "GBPUSD,AUDUSD,USDCAD,EURJPY,BTCUSD",
                    help="Comma-separated pair names (NAS100 is dropped run-level)")
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
