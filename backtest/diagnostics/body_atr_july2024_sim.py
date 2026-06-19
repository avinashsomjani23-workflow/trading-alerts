"""ONE-OFF DIAGNOSTIC — body-ATR gate vs current distance gate, July 2024.

Apples-to-apples: same swings, same fire candle, same data. The ONLY difference is
whether we additionally require the break candle (or the candle right after) to have
a body >= MULT * ATR. We do NOT re-run the whole Phase2/Phase3 pipeline — we isolate
the structure-detection layer (where the proposed gate lives) and measure each event's
forward outcome with a fixed, identical resolution rule for both methods.

Resolution rule (identical for both arms, so the comparison is fair):
  - Entry  = break candle close.
  - Stop   = the broken swing price (structure invalidation).
  - Target = 2R (2x the entry-to-stop distance) in the break direction.
  - Walk forward up to HORIZON bars; whichever of target/stop is hit first = win/loss.
  - Neither hit in horizon -> 'open' (excluded from W/L, reported separately).

This is a RELATIVE comparison harness, not the production P&L engine. Its only job is
to show how many events each gate keeps and how those events resolve under one rule.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
import dealing_range as dr
import smc_detector as sd

PAIRS = {
    'EURUSD': 'EURUSD_X_1h.parquet',
    'NZDUSD': 'NZDUSD_X_1h.parquet',
    'USDJPY': 'JPY_X_1h.parquet',
    'USDCHF': 'CHF_X_1h.parquet',
    'XAUUSD': 'GC_F_1h.parquet',
    'NAS100': 'NQ_F_1h.parquet',
}
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cache'))

MONTH_START = pd.Timestamp('2024-07-01', tz='UTC')
MONTH_END   = pd.Timestamp('2024-08-01', tz='UTC')
HORIZON = 72            # H1 bars to resolve a trade (~3 days)
RR = 2.0               # target = 2R
WARMUP = 200           # bars of history before we start emitting (swings+ATR need history)

# Body-gate multipliers to test (the 5 simulations requested), applied to BOTH BOS+CHoCH
# at first, then a split is reported too.
SIM_MULTS = [0.5, 0.8, 1.0, 1.3, 1.5]


def collect_events(df):
    """Single-pass: detection is CAUSAL (each event fires on its own candle's close,
    independent of future bars), so one full-series run with an unbounded event ring
    yields exactly the same events the walk-forward would, far faster. We raise the
    module ring cap so nothing is trimmed, then map each event's candle_ts to its idx."""
    dr.EVENT_RING_MAX = 10_000_000
    out = dr.compute_structure(df, None)
    ts_to_idx = {df.index[i]: i for i in range(len(df))}
    events = []
    for ev in (out.get('events') or []):
        ts = ev.get('candle_ts')
        if ts is None:
            continue
        idx = ts_to_idx.get(pd.Timestamp(ts))
        if idx is None or idx < WARMUP:
            continue
        ev = dict(ev)
        ev['_idx'] = idx
        events.append(ev)
    return events


def body_atr_at(df, idx, atr):
    if atr is None or atr <= 0 or idx is None or idx < 0 or idx >= len(df):
        return 0.0
    o = float(df['Open'].iloc[idx]); c = float(df['Close'].iloc[idx])
    return abs(c - o) / atr


def resolve(df, idx, direction, broken_price):
    """Return 'win'/'loss'/'open' under the fixed 2R rule."""
    if idx is None or idx >= len(df):
        return 'open'
    entry = float(df['Close'].iloc[idx])
    stop = float(broken_price)
    risk = abs(entry - stop)
    if risk <= 0:
        return 'open'
    if direction in ('bullish', 'bull', 'up'):
        target = entry + RR * risk
        for j in range(idx + 1, min(idx + 1 + HORIZON, len(df))):
            hi = float(df['High'].iloc[j]); lo = float(df['Low'].iloc[j])
            hit_t = hi >= target
            hit_s = lo <= stop
            if hit_s and hit_t:
                return 'loss'  # conservative: stop first on same bar
            if hit_s:
                return 'loss'
            if hit_t:
                return 'win'
    else:
        target = entry - RR * risk
        for j in range(idx + 1, min(idx + 1 + HORIZON, len(df))):
            hi = float(df['High'].iloc[j]); lo = float(df['Low'].iloc[j])
            hit_t = lo <= target
            hit_s = hi >= stop
            if hit_s and hit_t:
                return 'loss'
            if hit_s:
                return 'loss'
            if hit_t:
                return 'win'
    return 'open'


def tally(results):
    w = sum(1 for r in results if r == 'win')
    l = sum(1 for r in results if r == 'loss')
    o = sum(1 for r in results if r == 'open')
    wr = (100.0 * w / (w + l)) if (w + l) else float('nan')
    return w, l, o, wr


def main():
    print(f"July 2024 — current distance gate vs body-ATR gate\n")
    print(f"Horizon={HORIZON} H1 bars, RR={RR}, gate floor unchanged (0.4 BOS / 1.0 CHoCH distance)\n")

    # accumulate across all pairs
    all_events = []  # (pair, ev, body_break, body_window)
    for pair, fn in PAIRS.items():
        path = os.path.join(CACHE, fn)
        if not os.path.exists(path):
            print(f"  [skip] {pair}: no data")
            continue
        df = pd.read_parquet(path)
        evs = collect_events(df)
        for ev in evs:
            ts = pd.Timestamp(ev['candle_ts'])
            if not (MONTH_START <= ts < MONTH_END):
                continue
            idx = ev['_idx']
            atr = sd.compute_atr(df.iloc[:idx+1])
            bb = body_atr_at(df, idx, atr)              # break candle body
            bw = max(bb, body_atr_at(df, idx + 1, atr)) # window [idx, idx+1]
            res = resolve(df, idx, ev['direction'], ev['broken_swing_price'])
            all_events.append({
                'pair': pair, 'type': ev['type'], 'dir': ev['direction'],
                'ts': ts, 'idx': idx, 'body_break': bb, 'body_window': bw,
                'result': res,
            })

    if not all_events:
        print("NO EVENTS in July 2024 window.")
        return

    # CURRENT method = all events pass (distance gate already applied by engine)
    cur = [e['result'] for e in all_events]
    w, l, o, wr = tally(cur)
    print(f"CURRENT (distance only): trades={w+l+o}  W={w}  L={l}  open={o}  WR={wr:.1f}%")
    print(f"  BOS={sum(1 for e in all_events if e['type']=='BOS')}  CHoCH={sum(1 for e in all_events if e['type']=='CHoCH')}\n")

    print("BODY-ATR GATE (window = break candle OR next candle, max body):")
    print(f"{'mult':>5} | {'trades':>6} {'W':>3} {'L':>3} {'open':>4} {'WR%':>6} | {'cut':>4}")
    for m in SIM_MULTS:
        kept = [e for e in all_events if e['body_window'] >= m]
        res = [e['result'] for e in kept]
        w, l, o, wr = tally(res)
        cut = len(all_events) - len(kept)
        print(f"{m:>5} | {w+l+o:>6} {w:>3} {l:>3} {o:>4} {wr:>6.1f} | {cut:>4}")

    print("\nBODY-ATR GATE (break candle ONLY, no window):")
    print(f"{'mult':>5} | {'trades':>6} {'W':>3} {'L':>3} {'open':>4} {'WR%':>6} | {'cut':>4}")
    for m in SIM_MULTS:
        kept = [e for e in all_events if e['body_break'] >= m]
        res = [e['result'] for e in kept]
        w, l, o, wr = tally(res)
        cut = len(all_events) - len(kept)
        print(f"{m:>5} | {w+l+o:>6} {w:>3} {l:>3} {o:>4} {wr:>6.1f} | {cut:>4}")

    # how many events would the window save vs break-only?
    saved = sum(1 for e in all_events if e['body_break'] < 1.0 <= e['body_window'])
    print(f"\nImpulse-on-next-candle cases (break body <1.0 but window >=1.0): {saved}")

    # dump per-event detail
    print("\nPer-event detail:")
    print(f"{'pair':>7} {'type':>5} {'dir':>8} {'body_brk':>9} {'body_win':>9} {'result':>6}")
    for e in sorted(all_events, key=lambda x: x['ts']):
        print(f"{e['pair']:>7} {e['type']:>5} {e['dir']:>8} {e['body_break']:>9.2f} {e['body_window']:>9.2f} {e['result']:>6}")


if __name__ == '__main__':
    main()
