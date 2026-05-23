"""Throwaway diagnostic: where is CHoCH lost?

At each H1 bar:
  - Tally NEW dealing_range events (only the one whose candle_ts == this bar)
  - Tally OBs returned from smc_radar by (bos_tag, bos_tier)
  - Also collect drop_gate counts from ob_build_diagnostics

Run:  python backtest/_diag_choch.py
"""
from __future__ import annotations
import sys, json, contextlib, io, time
from pathlib import Path
from collections import Counter

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd
from backtest import data_loader
import dealing_range, smc_radar

# Same window as the 12-alert run.
START = "2026-04-01"
END = "2026-04-08"

with open(_REPO / "config.json") as f:
    cfg = json.load(f)

start_ts = pd.Timestamp(START, tz="UTC")
end_ts   = pd.Timestamp(END,   tz="UTC") + pd.Timedelta(days=1)

# Limit to 2 pairs to get evidence faster.
LIMIT_PAIRS = {"EURUSD", "NZDUSD"}

total_new_events = Counter()
total_radar_obs  = Counter()
total_drops      = Counter()

for pair_conf in cfg["pairs"]:
    pair_name = pair_conf["name"]
    if pair_name not in LIMIT_PAIRS:
        continue

    print(f"\n=== {pair_name} ===", flush=True)
    t0 = time.time()

    df_h1 = data_loader.load_bars(
        pair_conf["symbol"], "1h",
        (start_ts - pd.Timedelta(days=30)).to_pydatetime(),
        end_ts.to_pydatetime(),
    )
    if df_h1 is None or df_h1.empty:
        print(f"  no H1 data", flush=True)
        continue
    print(f"  loaded H1 rows={len(df_h1)} range=[{df_h1.index[0]} .. {df_h1.index[-1]}]", flush=True)

    h1_in_window = df_h1.loc[start_ts:end_ts]
    print(f"  walk bars in window: {len(h1_in_window)}", flush=True)

    dr_state = None
    pair_events = Counter()
    pair_radar = Counter()
    pair_drops = Counter()
    seen_event_keys = set()  # de-dupe events seen across bars

    for k, h1_ts in enumerate(h1_in_window.index):
        h1_slice = df_h1.loc[:h1_ts]
        if len(h1_slice) < 50:
            continue

        dr_state = dealing_range.update_pair(h1_slice, dr_state, pair_conf)
        walls = dr_state or {}
        events = walls.get("events", [])

        # Identify events whose candle_ts == this bar (the ones that fired NOW).
        for ev in events:
            key = (ev.get("candle_ts"), ev.get("type"), ev.get("tier"), ev.get("direction"))
            if key in seen_event_keys:
                continue
            seen_event_keys.add(key)
            try:
                if pd.Timestamp(ev.get("candle_ts")) == h1_ts:
                    pair_events[(ev.get("type"), ev.get("tier"))] += 1
            except Exception:
                pass

        with contextlib.redirect_stdout(io.StringIO()):
            res = smc_radar.detect_smc_radar(
                h1_slice, pair_type=pair_conf["pair_type"],
                events=events, walls=walls, pair_name=pair_name,
            )
        obs = (res or {}).get("active_unmitigated_obs") or []
        for ob in obs:
            pair_radar[(ob.get("bos_tag"), ob.get("bos_tier"))] += 1
        diags = (res or {}).get("ob_build_diagnostics") or []
        for d in diags:
            if d.get("outcome") == "dropped":
                pair_drops[(d.get("event_type"), d.get("event_tier"), d.get("drop_gate"))] += 1

        if (k + 1) % 20 == 0:
            print(f"  ... bar {k+1}/{len(h1_in_window)} t={time.time()-t0:.1f}s", flush=True)

    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    print(f"  NEW dealing_range events (type, tier): {dict(pair_events)}", flush=True)
    print(f"  smc_radar OBs by (bos_tag, bos_tier):  {dict(pair_radar)}", flush=True)
    print(f"  OB drops by (ev_type, ev_tier, gate):  {dict(pair_drops)}", flush=True)

    # Sample last 15 events in the final ring.
    if dr_state and dr_state.get("events"):
        sample = [
            (e.get("type"), e.get("tier"), e.get("direction"), e.get("candle_ts"))
            for e in dr_state["events"][-15:]
        ]
        print(f"  last 15 events in ring: {sample}", flush=True)

    total_new_events.update(pair_events)
    total_radar_obs.update(pair_radar)
    total_drops.update(pair_drops)

print("\n=== TOTALS ===", flush=True)
print(f"dealing_range new events: {dict(total_new_events)}", flush=True)
print(f"smc_radar OBs:            {dict(total_radar_obs)}", flush=True)
print(f"OB drops:                 {dict(total_drops)}", flush=True)
