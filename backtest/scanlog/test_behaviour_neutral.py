"""Behaviour-neutrality proof (SPEC Â§7.8, mandatory before merge).

Runs the REAL walk + simulator on one pair / one window TWICE:
  A) with NO active scanlog   (instrumentation is a no-op via NullScanLog)
  B) with an active scanlog   (every emit line fires)

Then asserts the two produce IDENTICAL trade rows and IDENTICAL Sum r_realised.
If the emit lines changed any branch, value, or ordering, this fails.

Uses the warm parquet cache (no network), so it is deterministic and offline.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from backtest import data_loader, replay_engine, h1_only_simulator
from backtest.scanlog import emitter as scanlog_emitter


PAIR = "EURUSD"
START = datetime(2025, 1, 6, tzinfo=timezone.utc)
END = datetime(2025, 2, 6, tzinfo=timezone.utc)
RISK = 250.0


def _load_pair_conf():
    cfg = json.load(open(_REPO_ROOT / "config.json"))
    p = next(p for p in cfg["pairs"] if p["name"] == PAIR)
    # mirror the backtest atr override so the walk matches a real run
    p["atr_multiplier"] = {"forex": 3.0, "index": 3.5, "commodity": 3.5}.get(
        p.get("pair_type"), p.get("atr_multiplier"))
    return p


def _walk_and_sim(pair_conf, df, with_scanlog):
    """Run one full walk + dual simulate, return the list of trade rows."""
    sl = None
    if with_scanlog:
        td = tempfile.mkdtemp()
        manifest = scanlog_emitter.build_manifest(
            run_id="neutral", git_sha="x", risk_usd=RISK, min_warmup_bars=50,
            pairs_served=[{"name": PAIR, "symbol": pair_conf["symbol"],
                           "requested_start": START.isoformat(),
                           "requested_end": END.isoformat(),
                           "served_start": str(df.index.min()),
                           "served_end": str(df.index.max()),
                           "n_bars": len(df),
                           "fingerprint": scanlog_emitter.fingerprint(df),
                           "prox_cap_atr": pair_conf["atr_multiplier"]}],
            knobs={f"{PAIR}.atr_multiplier": pair_conf["atr_multiplier"]},
            fetch_pad_days=35)
        sl = scanlog_emitter.ScanLog.begin(Path(td), manifest)
    else:
        scanlog_emitter.set_active(None)

    state = replay_engine.ReplayState()
    rows = []
    seen = set()
    for ev in replay_engine.replay_pair(
        pair_conf, df, state=state,
        walk_start_ts=pd.Timestamp(START), walk_end_ts=pd.Timestamp(END)):
        if ev["kind"] != "alert":
            continue
        key = ((ev.get("ob") or {}).get("ob_timestamp"),
               (ev.get("ob") or {}).get("direction"))
        if key in seen:
            continue
        seen.add(key)
        rows.extend(h1_only_simulator.simulate_h1_only_dual(
            ev, pair_conf, df, risk_usd=RISK))
    if sl is not None:
        sl.close()
    return rows


def _canonical(rows):
    """Strip to the fields that define trade identity + outcome."""
    keys = ("pair", "alert_ts", "entry_zone", "entry", "sl_initial", "tp1",
            "tp2", "exit_reason", "exit_price", "r_realised", "pnl_usd")
    return [tuple(r.get(k) for k in keys) for r in rows]


def main() -> int:
    pair_conf = _load_pair_conf()
    fetch_start = START - timedelta(days=35)
    df = data_loader.load_bars(pair_conf["symbol"], "1h", fetch_start, END)
    if df is None or df.empty:
        print("FAIL: no cached data for", PAIR)
        return 2

    rows_a = _walk_and_sim(pair_conf, df, with_scanlog=False)
    rows_b = _walk_and_sim(pair_conf, df, with_scanlog=True)

    print(f"  rows without scanlog: {len(rows_a)}")
    print(f"  rows with scanlog   : {len(rows_b)}")

    ca, cb = _canonical(rows_a), _canonical(rows_b)
    sum_a = round(sum(float(r.get("r_realised", 0.0)) for r in rows_a), 9)
    sum_b = round(sum(float(r.get("r_realised", 0.0)) for r in rows_b), 9)
    print(f"  Sum r_realised A: {sum_a}")
    print(f"  Sum r_realised B: {sum_b}")

    ok = True
    if len(ca) != len(cb):
        print(f"  FAIL: row count differs ({len(ca)} vs {len(cb)})")
        ok = False
    elif ca != cb:
        for i, (a, b) in enumerate(zip(ca, cb)):
            if a != b:
                print(f"  FAIL: row {i} differs:\n    A={a}\n    B={b}")
                ok = False
                break
    if sum_a != sum_b:
        print(f"  FAIL: Sum r_realised differs")
        ok = False

    print("=" * 50)
    if ok:
        print("BEHAVIOUR-NEUTRAL: instrumentation changed nothing")
        return 0
    print("BEHAVIOUR CHANGED - instrumentation is NOT a no-op")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
