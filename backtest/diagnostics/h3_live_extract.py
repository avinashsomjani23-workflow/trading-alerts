"""Tier-B live-record extractor (Harness 3 addendum Â§A).

Reconstructs the history of LIVE Phase-2 alerts from git history, read-only,
without ever touching the working tree. Live runs commit `phase2_scan_log.jsonl`
on every scan ("P2 Update [skip ci]" commits), so the union of that file across
commits is the live alert trail.

Output:
  out/live_alerts.csv   one row per live in-proximity zone observation
  prints a coverage report (first/last ts, per-pair counts, overlap caveats)

Honesty:
  - Trusts PAYLOAD timestamps (ts_ist), not commit timestamps (commit time =
    when CI pushed; payload time = when the scan computed).
  - Records committed BEFORE the parity field-add (distal/ob_timestamp/
    fired_levels) lack those fields â they come through as None. Tier-B on that
    older history is proximity-only; trade-level comparison needs post-add data.
  - Deterministic: two runs over the same git range produce identical CSV.

Run:
  python -m backtest.diagnostics.h3_live_extract --out backtest/diagnostics/out
  python -m backtest.diagnostics.h3_live_extract --match --pairs EURUSD --start 2026-03-01 --end 2026-03-31
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCAN_LOG = "phase2_scan_log.jsonl"
IST = "Asia/Kolkata"

# Results that mean "price was within proximity" (an alert-worthy approach),
# i.e. NOT dropped for being too far. Everything else is a live approach we can
# line up against a backtest alert.
_DROPPED_FAR = {"dropped_proximity"}


def _git(args: List[str]) -> str:
    return subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout


def _commits_touching(path: str) -> List[Tuple[str, str]]:
    """List (sha, committer_iso) for commits that touched `path`, oldest first."""
    out = _git(["log", "--follow", "--format=%H|%cI", "--", path])
    rows = []
    for line in out.splitlines():
        if "|" in line:
            sha, ciso = line.split("|", 1)
            rows.append((sha.strip(), ciso.strip()))
    return list(reversed(rows))  # oldest first


def _show(sha: str, path: str) -> Optional[str]:
    try:
        return _git(["show", f"{sha}:{path}"])
    except subprocess.CalledProcessError:
        return None  # file didn't exist at that commit


def _ist_to_utc(ts_ist: str) -> Optional[pd.Timestamp]:
    if not ts_ist:
        return None
    try:
        t = pd.Timestamp(ts_ist)
        if t.tzinfo is None:
            t = t.tz_localize(IST)
        return t.tz_convert("UTC")
    except Exception:
        return None


def extract(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    commits = _commits_touching(SCAN_LOG)
    print(f"[tierB] {len(commits)} commits touched {SCAN_LOG}", flush=True)

    seen: Set[Tuple] = set()
    rows: List[Dict[str, Any]] = []
    for sha, _ciso in commits:
        content = _show(sha, SCAN_LOG)
        if not content:
            continue
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            pair = rec.get("pair")
            ts_ist = rec.get("ts_ist")
            alert_ts = _ist_to_utc(ts_ist)
            fired = rec.get("fired_levels", {}) or {}
            for zo in rec.get("zone_outcomes", []) or []:
                result = zo.get("result")
                if result in _DROPPED_FAR:
                    continue
                direction = zo.get("direction")
                proximal = zo.get("proximal")
                ob_ts = zo.get("ob_timestamp")
                key = (pair, str(alert_ts), direction,
                       round(float(proximal), 8) if proximal is not None else None,
                       ob_ts)
                if key in seen:
                    continue
                seen.add(key)
                lv = {}
                if ob_ts and fired:
                    # fired_levels is keyed by zone_id, not ob_ts; best-effort
                    # match by scanning values whose context matches this zone.
                    lv = _match_fired(fired, direction)
                rows.append({
                    "pair": pair,
                    "scan_ts_utc": str(alert_ts) if alert_ts is not None else "",
                    "alert_ts_utc": str(alert_ts) if alert_ts is not None else "",
                    "ob_timestamp": ob_ts,
                    "direction": direction,
                    "proximal": proximal,
                    "distal": zo.get("distal"),
                    "entry": lv.get("entry"),
                    "sl": lv.get("sl"),
                    "tp1": lv.get("tp1"),
                    "tp2": lv.get("tp2"),
                    "result": result,
                    "source_sha": sha[:10],
                })

    rows.sort(key=lambda r: (r["pair"], r["alert_ts_utc"]))
    csv_path = out_dir / "live_alerts.csv"
    cols = ["pair", "scan_ts_utc", "alert_ts_utc", "ob_timestamp", "direction",
            "proximal", "distal", "entry", "sl", "tp1", "tp2", "result", "source_sha"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    _coverage(rows)
    print(f"[tierB] live_alerts.csv -> {csv_path} ({len(rows)} rows)", flush=True)
    return csv_path


def _match_fired(fired: Dict[str, Any], direction: str) -> Dict[str, Any]:
    """Best-effort pick of a fired_levels entry. fired_levels is keyed by
    zone_id which encodes bias; pick the first whose key matches the bias."""
    want = "LONG" if direction == "bullish" else "SHORT"
    for zid, lv in fired.items():
        if f"_{want}_" in zid:
            return lv
    # fall back to a sole entry
    if len(fired) == 1:
        return next(iter(fired.values()))
    return {}


def _coverage(rows: List[Dict[str, Any]]):
    if not rows:
        print("[tierB] coverage: NO live records found.", flush=True)
        return
    ts = [r["alert_ts_utc"] for r in rows if r["alert_ts_utc"]]
    by_pair: Dict[str, int] = {}
    with_levels = 0
    for r in rows:
        by_pair[r["pair"]] = by_pair.get(r["pair"], 0) + 1
        if r["entry"] is not None:
            with_levels += 1
    print(f"[tierB] coverage: {len(rows)} approaches | "
          f"first={min(ts) if ts else '?'} last={max(ts) if ts else '?'}", flush=True)
    print(f"[tierB] per-pair: {by_pair}", flush=True)
    print(f"[tierB] {with_levels} rows carry trade levels (post field-add); "
          f"{len(rows) - with_levels} are proximity-only (older history).", flush=True)
    if len(rows) < 30:
        print("[tierB] WARNING: <30 live records â any parity verdict is "
              "'held on N=...', never 'proven' (small-N rule).", flush=True)


def match(pairs: List[str], start, end, out_dir: Path) -> Path:
    """Basic MATCHED / LIVE_ONLY / BACKTEST_ONLY split over the overlap window,
    replaying at the LIVE proximity cap. Deep mismatch drill-down
    (data->config->warmup->logic) is the documented next layer."""
    from backtest.diagnostics import driver
    live_csv = out_dir / "live_alerts.csv"
    if not live_csv.exists():
        extract(out_dir)
    live = list(csv.DictReader(open(live_csv, encoding="utf-8")))
    cfg = json.load(open(_REPO_ROOT / "config.json"))
    confs = [p for p in cfg["pairs"] if (pairs == ["all"] or p["name"] in pairs)]
    start = pd.Timestamp(start, tz="UTC"); end = pd.Timestamp(end, tz="UTC")

    report = [out_dir / "tierB_match.md"]
    lines = ["# Tier-B â live vs backtest (live proximity cap)", ""]
    for pc in confs:
        name = pc["name"]; pt = pc.get("pair_type", "forex")
        live_pair = [r for r in live if r["pair"] == name
                     and r["alert_ts_utc"]
                     and start <= pd.Timestamp(r["alert_ts_utc"]) <= end]
        df = driver.load_window(pc, start, end)
        if isinstance(df, driver.WindowUnserveable):
            lines.append(f"## {name}: window unserveable ({df.reason})"); continue
        live_cap = float(pc["atr_multiplier"])
        res = driver.walk_alerts(
            pc, df, start, end,
            overrides=driver.KnobOverrides(proximity_cap={pt: live_cap}))

        # Match on OB PROXIMAL with a tolerance, not ob_timestamp: older live
        # history predates the ob_timestamp field, but proximal is present and
        # both sides run the same detection, so the OB proximal lines up. One
        # OB is approached many times live -> dedup to distinct (direction,
        # proximal) on both sides. Tolerance = 1 pip (forex) / wider for index.
        dp = int(pc.get("decimal_places", 5))
        tol = (0.0001 if dp >= 4 else 0.01) * 2  # ~2 pips slack for feed drift
        if pt == "index":
            tol = 5.0
        elif pt == "commodity":
            tol = 1.0

        def _distinct(items):  # list of (direction, proximal_float)
            out = []
            for d, px in items:
                if px is None:
                    continue
                if not any(d == d2 and abs(px - px2) <= tol for d2, px2 in out):
                    out.append((d, px))
            return out

        bt_ds = _distinct([((a.get("ob") or {}).get("direction"),
                            (a.get("ob") or {}).get("proximal_line"))
                           for a in res.alerts])
        live_ds = _distinct([(r["direction"],
                             float(r["proximal"]) if r["proximal"] else None)
                            for r in live_pair])

        def _matched(a_list, b_list):
            return [a for a in a_list
                    if any(a[0] == b[0] and abs(a[1] - b[1]) <= tol for b in b_list)]

        matched = _matched(live_ds, bt_ds)
        live_only = [a for a in live_ds if a not in matched]
        bt_only = [b for b in bt_ds
                   if not any(b[0] == m[0] and abs(b[1] - m[1]) <= tol for m in matched)]
        lines.append(f"## {name}")
        lines.append(f"- live in-proximity (window): {len(live_pair)} approaches "
                     f"-> {len(live_ds)} distinct OB proximals")
        lines.append(f"- backtest alerts (live cap {live_cap}): {len(res.alerts)} "
                     f"-> {len(bt_ds)} distinct OB proximals")
        lines.append(f"- MATCHED={len(matched)} BACKTEST_ONLY={len(bt_only)} "
                     f"LIVE_ONLY={len(live_only)} (proximal tol={tol})")
        if live_only:
            lines.append(f"  - LIVE_ONLY proximals: {[round(x[1],dp) for x in live_only][:10]}")
        if bt_only:
            lines.append(f"  - BACKTEST_ONLY proximals: {[round(x[1],dp) for x in bt_only][:10]}")
        if len(matched) < 30:
            lines.append(f"- âš ï¸ small-N: {len(matched)} matches â verdict is "
                         f"'held on N={len(matched)}', not proven.")
        lines.append("- NOTE: matched on OB proximal (Â±tol) because older live "
                     "history predates ob_timestamp. Mismatch drill-down "
                     "(dataâ†’configâ†’warmupâ†’logic) is the documented next layer.")
        lines.append("")
    report[0].write_text("\n".join(lines), encoding="utf-8")
    print(f"[tierB] match report -> {report[0]}", flush=True)
    return report[0]


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--match", action="store_true")
    ap.add_argument("--pairs", default="EURUSD")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out)
    if args.match:
        if not (args.start and args.end):
            raise SystemExit("--match requires --start and --end")
        pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
        match(pairs, args.start, args.end, out_dir)
    else:
        extract(out_dir)


if __name__ == "__main__":
    main()
