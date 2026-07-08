"""Guard for the EVENT-CANDLE FIX (2026-07-09).

Root bug (fixed): BOS/CHoCH events stamped `candle_ts` on the CONFIRMATION candle
(where the displacement buffer AND the body gate both finally passed), not on the
TRUE break candle (the first candle whose close cleared the displacement buffer).
Every downstream anchor — the OB especially — inherited the wrong candle.

The fix (dealing_range.py `_true_break_idx`, called in all 6 event branches):
keep both gates as the fire/don't-fire decision, but once fired, stamp the FIRST
candle of the contiguous run of past-buffer closes that ends at the confirmation
candle. Logged per event as `event_candle_delta` (= confirm_idx − true_break_idx).

Silent-failure mode this guards: someone re-welds the candle choice back onto the
confirmation candle (the original bug), or drops the walk-back on one branch. That
would corrupt the OB on ~15% of events (34% of CHoCHs) with NO error — a wrong
alert, not a crash. This test lives OUT of the live path (offline, on frozen
cached data) so it can go red without touching a real alert.

Two guards:
  1. BEHAVIORAL — on a frozen real USDCHF window that is known to contain a
     multi-bar-confirmed CHoCH, assert the emitted event_candle_delta > 0 and that
     candle_ts is EXACTLY the first close past the displacement buffer (and the bar
     before it is NOT past — i.e. the true start of the run).
  2. SOURCE TRIPWIRE — `_true_break_idx` is defined and called from every one of
     the six event branches. A revert that inlines `ts_now = _ts_iso(df, ci)` back
     into a branch turns this red.

Run:  python tests/test_event_candle_fix.py
Exit 0 iff every guard passes. No pytest dependency (mirrors test_structure_signals).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import dealing_range as dr  # noqa: E402
import smc_radar  # noqa: E402
import smc_detector  # noqa: E402

_FAILS = []


def _ok(m): print(f"  OK:   {m}")
def _bad(m):
    print(f"  FAIL: {m}")
    _FAILS.append(m)


# Frozen cached window: the last 150 closed bars up to 2026-05-28 14:00 UTC on
# USDCHF. Verified (2026-07-09) to emit a CHoCH-bearish whose true break is
# 2026-05-24 21:00 and whose confirmation is 15 bars later — a textbook grind
# where the body gate lagged the actual break. The cache is immutable history
# (backtest/cache/*.parquet, sliced never fetched), so this window is stable.
_CACHE = _ROOT / "backtest" / "cache" / "CHF_X_1h.parquet"
_WINDOW_END = "2026-05-28 14:00"
_PAIR = "USDCHF"
_STRUCTURE_CHOCH_ATR_MULT = 1.5  # dealing_range.STRUCTURE_CHOCH_ATR_MULT


def _behavioral():
    if not _CACHE.exists():
        _bad(f"cache missing: {_CACHE} (cannot run behavioral guard)")
        return
    df = pd.read_parquet(_CACHE)
    w = df.loc[:_WINDOW_END].tail(smc_radar.LIVE_DETECTION_BARS)
    if len(w) < smc_radar.LIVE_DETECTION_BARS:
        _bad(f"frozen window short: {len(w)} bars")
        return

    walls = smc_radar.compute_pair_walls(w, _PAIR)
    events = walls.get("events", []) or []

    # Every real structural event must carry the delta field (int or 0). None is
    # only legal on CHoCH_FAILED (not a break) — which never appears in the ring.
    for ev in events:
        if ev.get("type") in ("BOS", "CHoCH"):
            if "event_candle_delta" not in ev:
                _bad(f"event {ev.get('type')} @ {ev.get('candle_ts')} missing event_candle_delta")
    _ok("every BOS/CHoCH event carries event_candle_delta")

    # Find the multi-bar CHoCH (delta > 0) — the whole point of the fix.
    multi = [e for e in events
             if e.get("type") == "CHoCH" and (e.get("event_candle_delta") or 0) > 0]
    if not multi:
        _bad("frozen window emitted NO multi-bar CHoCH — fixture stale or fix reverted "
             "(pre-fix code stamps the confirmation candle so delta is always 0)")
        return
    ev = max(multi, key=lambda e: e["event_candle_delta"])
    _ok(f"multi-bar CHoCH present: delta={ev['event_candle_delta']} bars "
        f"(candle_ts={ev['candle_ts'][:16]})")

    # Reconstruct the buffer and prove candle_ts is the FIRST close past it.
    atr = smc_detector.compute_atr(w, 14)
    broken = ev["broken_swing_price"]
    bearish = ev["direction"] == "bearish"
    thr = (broken - _STRUCTURE_CHOCH_ATR_MULT * atr) if bearish \
        else (broken + _STRUCTURE_CHOCH_ATR_MULT * atr)

    i = w.index.get_indexer([pd.Timestamp(ev["candle_ts"])])[0]
    if i <= 0:
        _bad(f"candle_ts {ev['candle_ts']} not locatable in window (idx={i})")
        return
    c_break = float(w["Close"].iloc[i])
    c_prev = float(w["Close"].iloc[i - 1])
    past_break = (c_break < thr) if bearish else (c_break > thr)
    past_prev = (c_prev < thr) if bearish else (c_prev > thr)

    if past_break and not past_prev:
        _ok("candle_ts IS the first close past the displacement buffer "
            "(the bar before it closed inside — true start of the run)")
    else:
        _bad(f"candle_ts is NOT the run start: close@break={c_break} past={past_break}, "
             f"close@prev={c_prev} past_prev={past_prev}, thr={thr:.5f}")

    # The whole run [true_break .. confirmation] must be unbroken past-buffer.
    conf_i = i + int(ev["event_candle_delta"])
    run = w["Close"].iloc[i:conf_i + 1]
    all_past = ((run < thr).all()) if bearish else ((run > thr).all())
    if all_past:
        _ok(f"contiguous run true_break->confirmation all past buffer ({len(run)} bars)")
    else:
        _bad("run true_break->confirmation is NOT all past buffer — bound wrong")

    # DISPLACEMENT CONVICTION must be graded on the CONFIRMATION candle, not the
    # (possibly weak) true-break candle. The event carries confirm_ts for exactly
    # this; break_body_atr must equal the confirmation-candle window body, and the
    # break_quality excess invariant (>=1.0) must hold — feeding the true-break
    # candle here was the bug that silently mislabeled decisive breaks as marginal.
    if not ev.get("confirm_ts"):
        _bad("multi-bar event missing confirm_ts — break_quality would grade the wrong candle")
    else:
        cf = w.index.get_indexer([pd.Timestamp(ev["confirm_ts"])])[0]
        bq = smc_detector.compute_break_quality(
            w, cf, broken, ev["direction"], atr, event_type="CHoCH")
        if bq["excess"] >= 1.0:
            _ok(f"break_quality graded on confirmation candle: excess={bq['excess']} (>=1.0 invariant holds)")
        else:
            _bad(f"break_quality excess={bq['excess']} < 1.0 — graded on the weak true-break candle (bug regressed)")
        # break_body_atr on the event must reflect the confirmation window, which
        # on a grind is strictly stronger than the true-break candle's own body.
        tb_body = abs(float(w["Close"].iloc[i]) - float(w["Open"].iloc[i])) / atr
        if ev.get("break_body_atr") is not None and ev["break_body_atr"] > tb_body:
            _ok(f"break_body_atr={ev['break_body_atr']} reflects confirmation window "
                f"(> true-break candle body {tb_body:.2f}) — conviction not weakened by the anchor move")
        else:
            _bad(f"break_body_atr={ev.get('break_body_atr')} <= true-break body {tb_body:.2f} "
                 "— displacement graded on the wrong candle")


# ── SOURCE TRIPWIRE: _true_break_idx must exist and be called from each of the
# six event branches. Count the call sites in the producer; a revert that inlines
# the old confirmation-candle stamp on any branch drops the count. -------------
def _source_tripwire():
    src = (_ROOT / "dealing_range.py").read_text(encoding="utf-8")
    if "def _true_break_idx(" not in src:
        _bad("_true_break_idx helper is GONE from dealing_range.py — fix reverted")
        return
    _ok("_true_break_idx helper present")

    calls = len(re.findall(r"_true_break_idx\(", src))
    # 1 definition + 6 call sites (2 confirmation-BOS, 2 CHoCH-arm, 2 continuation
    # BOS). Birth intentionally does NOT call it (raw close-through, delta=0).
    if calls >= 7:
        _ok(f"_true_break_idx referenced {calls}x (1 def + >=6 branch calls)")
    else:
        _bad(f"_true_break_idx referenced only {calls}x — expected >=7 "
             "(1 def + 6 branches); a branch reverted to stamping the confirmation candle")

    # confirm_idx must reach _push_event so the delta is logged.
    if "confirm_idx=ci" in src:
        _ok("confirm_idx=ci threaded into _push_event (delta logged)")
    else:
        _bad("confirm_idx=ci not passed to _push_event — event_candle_delta will be None")


def main():
    print("EVENT-CANDLE FIX guards")
    print("- behavioral (frozen USDCHF window):")
    _behavioral()
    print("- source tripwire:")
    _source_tripwire()
    if _FAILS:
        print(f"\n{len(_FAILS)} FAIL(s)")
        sys.exit(1)
    print("\nall event-candle-fix guards: OK")


if __name__ == "__main__":
    main()
