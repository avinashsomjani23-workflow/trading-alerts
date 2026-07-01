"""
Scratch: D1 daily-bias flip study (EURUSD, NZDUSD, XAUUSD).

Answers, with data, the open questions from the 2026-06-29 daily-bias chat:

 Q1  Flip rule: RAW formation flip (sticky HH+HL / LH+LL) vs the engine's
     CHoCH + Confirmation-BOS flip. Which has less LAG, which has fewer FAKEOUTS?
 Q2  ATR buffer on D1: engine buffer ON (choch 1.0 ATR + body gates + BOS 0.4 ATR)
     vs OFF (pure close-beyond). Does removing the buffer blow up fakeouts?
 Q3  lookback 2 vs 3 on D1.
 Q4  Ranging flag = the stall counter (STRUCTURE_RANGING_STALE). At N=1/2/3,
     how much trend-time is flagged ranging, and do those bars actually move less
     (validation that the flag means real ranging, not lag)?

Reuses the LIVE engine (dealing_range.compute_structure) for the confirmation
rule via its _trace hook, and detect_swings for the raw-formation rule. The H1
ATR leg filter is disabled on D1 (_min_leg_atr_mult=None) per the chat decision.

Run:  python backtest/diagnostics/scratch_d1_bias_flip.py
Scratch / uncommitted. Re-runnable.
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import dealing_range as dr  # noqa: E402

INSTRUMENTS = ["EURUSD", "NZDUSD", "XAUUSD"]
DATA = os.path.join(_ROOT, "backtest", "mt5_data")
FWD_W = 20          # forward window (D1 bars) to resolve a flip correct/fakeout
RANGE_FWD = 10      # forward window to validate the ranging flag (displacement)


def load(inst):
    df = pd.read_csv(os.path.join(DATA, f"{inst}_D1.csv"))
    df = df.rename(columns={"time_server": "Datetime", "open": "Open",
                            "high": "High", "low": "Low", "close": "Close"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    return df.reset_index(drop=True)


# ---- Rule A: raw sticky formation flip, point-in-time -----------------------
def rule_a_flips(df, lookback):
    """Sticky HH+HL / LH+LL over confirmed swings. A swing at idx i is only
    visible at bar i+lookback (no look-ahead). Returns [(flip_bar_idx, newdir)]."""
    swings = dr.detect_swings(df, lookback=lookback, min_leg_atr_mult=None)
    highs, lows = [], []
    state = None
    flips = []
    for s in swings:
        (highs if s["type"] == "high" else lows).append(s)
        vis_bar = s["idx"] + lookback
        if vis_bar >= len(df):
            continue
        new = state
        if len(highs) >= 2 and len(lows) >= 2:
            HH = highs[-1]["price"] > highs[-2]["price"]
            HL = lows[-1]["price"] > lows[-2]["price"]
            LH = highs[-1]["price"] < highs[-2]["price"]
            LL = lows[-1]["price"] < lows[-2]["price"]
            if HH and HL:
                new = "up"
            elif LH and LL:
                new = "down"
        if new is not None and new != state:
            if state is not None:
                flips.append((vis_bar, new))
            state = new
    return flips


# ---- Rule B: engine CHoCH + Confirmation BOS, via _trace --------------------
def rule_b_flips(df, lookback, buffer_on):
    saved = (dr.BOS_ATR_MULT, dr.BOS_BODY_ATR_MULT, dr.STRUCTURE_CHOCH_BODY_ATR_MULT)
    if not buffer_on:
        dr.BOS_ATR_MULT = 0.0
        dr.BOS_BODY_ATR_MULT = 0.0
        dr.STRUCTURE_CHOCH_BODY_ATR_MULT = 0.0
    trace = []
    try:
        dr.compute_structure(
            df, None, lookback=lookback, _min_leg_atr_mult=None,
            choch_atr_mult=(1.0 if buffer_on else 0.0), _trace=trace)
    finally:
        dr.BOS_ATR_MULT, dr.BOS_BODY_ATR_MULT, dr.STRUCTURE_CHOCH_BODY_ATR_MULT = saved
    flips = []
    prev = None
    for i, st in enumerate(trace):
        if st in ("up", "down"):
            if prev is not None and st != prev:
                flips.append((i, st))
            prev = st
    return flips


# ---- shared: lag + correctness for a flip list ------------------------------
def analyze(df, flips):
    H = df["High"].values
    L = df["Low"].values
    C = df["Close"].values
    n = len(df)
    lags, verdicts = [], []          # verdict: True=held, False=fakeout, None=unresolved
    for k, (fbar, newdir) in enumerate(flips):
        prev_bar = flips[k - 1][0] if k > 0 else 0
        seg = slice(prev_bar, fbar + 1)
        if fbar <= prev_bar:
            continue
        if newdir == "down":
            ext_idx = prev_bar + int(np.argmax(H[seg]))   # old up-trend high
            ext_price = H[ext_idx]
            trig = L[fbar]
            lag = fbar - ext_idx
            end = min(fbar + FWD_W, n - 1)
            v = None
            for j in range(fbar + 1, end + 1):
                if C[j] > ext_price:          # reclaimed old high -> fakeout
                    v = False
                    break
                if L[j] < trig:               # extended lower -> held
                    v = True
                    break
        else:
            ext_idx = prev_bar + int(np.argmin(L[seg]))   # old down-trend low
            ext_price = L[ext_idx]
            trig = H[fbar]
            lag = fbar - ext_idx
            end = min(fbar + FWD_W, n - 1)
            v = None
            for j in range(fbar + 1, end + 1):
                if C[j] < ext_price:
                    v = False
                    break
                if H[j] > trig:
                    v = True
                    break
        lags.append(lag)
        verdicts.append(v)
    return lags, verdicts


def summarize(lags, verdicts):
    resolved = [v for v in verdicts if v is not None]
    held = sum(1 for v in resolved if v)
    n_flips = len(verdicts)
    pct_correct = (100.0 * held / len(resolved)) if resolved else float("nan")
    med_lag = float(np.median(lags)) if lags else float("nan")
    return n_flips, len(resolved), pct_correct, med_lag


# ---- Q4: ranging flag occupancy + validation --------------------------------
def ranging_study(df, lookback):
    """Mirror the engine's per-bar stall counter (trend_dir_swings_since_extend)
    on the raw-formation state, then report occupancy + forward displacement for
    N = 1,2,3. ranging bars should move LESS than expansion bars if the flag is real."""
    swings = dr.detect_swings(df, lookback=lookback, min_leg_atr_mult=None)
    H = df["High"].values
    L = df["Low"].values
    C = df["Close"].values
    atr = dr._compute_atr(df) or np.nan
    n = len(df)
    # build per-bar (state, counter) by walking visible swings
    ev = []  # (vis_bar, swing)
    for s in swings:
        ev.append((s["idx"] + lookback, s))
    ev.sort(key=lambda x: x[0])
    state = None
    run_hi = run_lo = None
    counter = 0
    highs, lows = [], []
    per_bar = []  # counter value applicable from this vis_bar
    ei = 0
    for bar in range(n):
        while ei < len(ev) and ev[ei][0] == bar:
            s = ev[ei][1]
            (highs if s["type"] == "high" else lows).append(s)
            # sticky state
            if len(highs) >= 2 and len(lows) >= 2:
                HH = highs[-1]["price"] > highs[-2]["price"]
                HL = lows[-1]["price"] > lows[-2]["price"]
                LH = highs[-1]["price"] < highs[-2]["price"]
                LL = lows[-1]["price"] < lows[-2]["price"]
                ns = state
                if HH and HL:
                    ns = "up"
                elif LH and LL:
                    ns = "down"
                if ns != state:
                    state = ns
                    counter = 0
                    run_hi = s["price"] if state == "up" else run_hi
                    run_lo = s["price"] if state == "down" else run_lo
            # stall counter (engine logic): extend resets, opposite-type +1
            if state == "up":
                if s["type"] == "high" and (run_hi is None or s["price"] > run_hi):
                    run_hi = s["price"]
                    counter = 0
                elif s["type"] == "low":
                    counter += 1
            elif state == "down":
                if s["type"] == "low" and (run_lo is None or s["price"] < run_lo):
                    run_lo = s["price"]
                    counter = 0
                elif s["type"] == "high":
                    counter += 1
            ei += 1
        per_bar.append((state, counter))

    out = {}
    for N in (1, 2, 3):
        rng_moves, exp_moves = [], []
        rng_bars = 0
        trend_bars = 0
        for bar in range(n - RANGE_FWD):
            st, cnt = per_bar[bar]
            if st not in ("up", "down"):
                continue
            trend_bars += 1
            fwd = abs(C[bar + RANGE_FWD] - C[bar]) / atr if atr == atr else np.nan
            if cnt >= N:
                rng_bars += 1
                rng_moves.append(fwd)
            else:
                exp_moves.append(fwd)
        out[N] = {
            "occupancy_pct": 100.0 * rng_bars / trend_bars if trend_bars else float("nan"),
            "rng_fwd_atr": float(np.nanmean(rng_moves)) if rng_moves else float("nan"),
            "exp_fwd_atr": float(np.nanmean(exp_moves)) if exp_moves else float("nan"),
        }
    return out


def main():
    print(f"\nD1 BIAS FLIP STUDY  |  fwd window={FWD_W} bars  |  instruments={INSTRUMENTS}")
    print("=" * 92)

    configs = [
        ("A formation", "A", None, 3),
        ("A formation", "A", None, 2),
        ("B CHoCH+BOS buf ON", "B", True, 3),
        ("B CHoCH+BOS buf OFF", "B", False, 3),
        ("B CHoCH+BOS buf ON", "B", True, 2),
        ("B CHoCH+BOS buf OFF", "B", False, 2),
    ]

    # pooled accumulators
    for label, rule, buf, lb in configs:
        pooled_lags, pooled_verdicts = [], []
        rows = []
        for inst in INSTRUMENTS:
            df = load(inst)
            if rule == "A":
                flips = rule_a_flips(df, lb)
            else:
                flips = rule_b_flips(df, lb, buf)
            lags, verdicts = analyze(df, flips)
            pooled_lags += lags
            pooled_verdicts += verdicts
            rows.append((inst, summarize(lags, verdicts)))
        nf, nr, pc, ml = summarize(pooled_lags, pooled_verdicts)
        print(f"\n[{label}  lb={lb}]")
        print(f"  {'inst':8} {'flips':>6} {'resolved':>9} {'%correct':>9} {'med_lag':>8}")
        for inst, (a, b, c, d) in rows:
            print(f"  {inst:8} {a:6d} {b:9d} {c:9.1f} {d:8.1f}")
        print(f"  {'POOLED':8} {nf:6d} {nr:9d} {pc:9.1f} {ml:8.1f}")

    print("\n" + "=" * 92)
    print("RANGING FLAG (stall counter = STRUCTURE_RANGING_STALE) validation, lb=3")
    print(f"  occupancy = % of trend-bars flagged ranging; fwd move = mean |close move| "
          f"over {RANGE_FWD} bars in ATR")
    for inst in INSTRUMENTS:
        df = load(inst)
        rs = ranging_study(df, 3)
        print(f"\n  {inst}")
        print(f"    {'N':>3} {'occupancy%':>11} {'ranging_fwd':>12} {'expansion_fwd':>14}")
        for N in (1, 2, 3):
            r = rs[N]
            print(f"    {N:3d} {r['occupancy_pct']:11.1f} {r['rng_fwd_atr']:12.2f} "
                  f"{r['exp_fwd_atr']:14.2f}")


if __name__ == "__main__":
    main()
