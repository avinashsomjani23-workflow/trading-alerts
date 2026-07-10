"""Guard for the GATES-REMOVED break model (2026-07-10).

History:
- 2026-07-09: the "event-candle fix" split the fire candle from the true-break
  candle. A distance buffer + body gate decided WHETHER an event fired; once
  fired, `_true_break_idx` walked back to the first candle of the past-buffer run.
- 2026-07-10: break-quality (body/distance) was never a proven outcome predictor,
  so BOTH gates were REMOVED. A BOS / CHoCH / Confirmation BOS now fires on a bare
  close through the level (any amount past, on close not wick). `break_body_atr`
  and `event_candle_delta` are still LOGGED as pure observations (the raw signal a
  future data-derived gate would read) — they gate nothing.

New invariant this guards (the thing that must not silently regress):
  1. Every BOS/CHoCH fires on close-through with NO body/distance floor — a break
     whose body is tiny must STILL appear (the old bug silently dropped it).
  2. `break_body_atr` is logged on every BOS/CHoCH event (observation intact).
  3. `event_candle_delta` == 0 for a CONTINUOUS break (fire candle == true-break
     candle). It may be > 0 ONLY when a SESSION GAP separates the true break from
     the gated fire (weekend-open close-through delayed by `_traded_through`); in
     that case candle_ts must be a close already past the BARE level.
  4. SOURCE TRIPWIRE: no `_body_gate_ok` anywhere, no `± *_disp` distance buffer in
     any break condition, and `_true_break_idx` still wired into all six branches
     (kept for the gap walk-back and for re-arming a data-derived gate later).

Silent-failure mode guarded: someone re-adds a body/distance floor (dropping real
small-body breaks with no error — a MISSING alert, not a crash), or drops the
break_body_atr/delta logging (blinding the future data-gate study). Lives OUT of
the live path (offline, frozen cached data) so it can go red without touching a
real alert.

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
# USDCHF. This window contains (verified 2026-07-10) a bearish BOS whose true
# break is the Sunday-open bar 2026-05-24 21:00 and whose gated fire is 3 bars
# later (a weekend-gap walk-back — delta 3), plus several continuous breaks with
# delta 0 and small bodies (e.g. body 0.65-0.68 ATR, well under the OLD 1.0 gate)
# that the removed gate would have dropped. The cache is immutable history
# (backtest/cache/*.parquet, sliced never fetched), so this window is stable.
_CACHE = _ROOT / "backtest" / "cache" / "CHF_X_1h.parquet"
_WINDOW_END = "2026-05-28 14:00"
_PAIR = "USDCHF"
_BAR_HOURS = 1.0
_GAP_HOURS = 1.5 * _BAR_HOURS  # > this between bars == a session gap


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
    breaks = [e for e in events if e.get("type") in ("BOS", "CHoCH")]
    if not breaks:
        _bad("frozen window emitted NO BOS/CHoCH — fixture stale or engine broken")
        return
    _ok(f"{len(breaks)} BOS/CHoCH events emitted on the frozen window")

    # 1. break_body_atr logged on every break event (observation intact).
    missing_body = [e for e in breaks if e.get("break_body_atr") is None]
    if missing_body:
        _bad(f"{len(missing_body)} break events missing break_body_atr (observation dropped)")
    else:
        _ok("break_body_atr logged on every BOS/CHoCH event")

    # 2. GATE REMOVED — at least one kept break has a body BELOW the old floors
    #    (BOS 1.0 / CHoCH 1.5 ATR). If the smallest body is still >= the old
    #    floor, a gate may have crept back in and be silently dropping small breaks.
    small = [e for e in breaks
             if e.get("break_body_atr") is not None
             and e["break_body_atr"] < (dr.STRUCTURE_CHOCH_BODY_ATR_MULT
                                        if e["type"] == "CHoCH" else dr.BOS_BODY_ATR_MULT)]
    if small:
        _ok(f"{len(small)} kept break(s) have body below the OLD gate floor "
            f"(min={min(e['break_body_atr'] for e in small)}) — gate is truly OFF")
    else:
        _bad("NO kept break is below the old body floor — a body gate may have "
             "regressed and be silently dropping small-body breaks")

    # 3. event_candle_delta discipline: 0 for continuous breaks; > 0 ONLY across a
    #    session gap, and then candle_ts must be a close past the BARE level.
    ts_list = list(w.index)
    for e in breaks:
        d = e.get("event_candle_delta")
        if d is None:
            _bad(f"{e['type']} @ {e.get('candle_ts')} has event_candle_delta None (should be int)")
            continue
        if d == 0:
            # continuous break: fire candle == true break. confirm_ts must equal candle_ts.
            if e.get("confirm_ts") != e.get("candle_ts"):
                _bad(f"delta 0 but confirm_ts != candle_ts ({e['type']} @ {e['candle_ts']})")
            continue
        if d < 0:
            _bad(f"{e['type']} @ {e['candle_ts']} has NEGATIVE delta {d}")
            continue
        # d > 0 — the gate that delayed the fire is `_traded_through`, which only
        # bites on a SESSION-GAP candle. That gap sits at (or just before) the
        # true-break candle, so check the interval into the true-break bar. The
        # true-break close must also be past the bare level.
        tb = pd.Timestamp(e["candle_ts"])
        pos = ts_list.index(tb)
        gap_in = ((ts_list[pos] - ts_list[pos - 1]).total_seconds() / 3600.0
                  if pos > 0 else 0.0)
        has_gap = gap_in > _GAP_HOURS
        bp = e["broken_swing_price"]
        up = e["direction"] in ("bullish", "up")
        c_tb = float(w.loc[tb, "Close"])
        past = (c_tb < bp) if not up else (c_tb > bp)
        if has_gap and past:
            _ok(f"{e['type']} delta={d} justified by a {gap_in:.0f}h session gap "
                f"into the true break; close past the bare level "
                f"(candle_ts={str(tb)[:16]})")
        else:
            _bad(f"{e['type']} delta={d} but gap-into-true-break={gap_in:.1f}h "
                 f"(<= {_GAP_HOURS}h) — walk-back should be 0 on a continuous "
                 "break once the distance buffer is removed")


# ── SOURCE TRIPWIRE ──────────────────────────────────────────────────────────
def _source_tripwire():
    src = (_ROOT / "dealing_range.py").read_text(encoding="utf-8")

    # (a) The body gate helper and every call must be GONE.
    if "_body_gate_ok" in src:
        _bad("_body_gate_ok still present in dealing_range.py — body gate regressed")
    else:
        _ok("_body_gate_ok fully removed (no body gate)")

    # (b) No break condition may carry a distance buffer term. The buffers are now
    #     the literal 0.0; a re-added `- bos_disp` / `+ choch_disp` etc. in a
    #     comparison would re-arm the distance gate.
    buffer_terms = re.findall(r"[<>]\s*[^\n]*?[+\-]\s*(?:bos_disp|choch_disp)\b", src)
    if buffer_terms:
        _bad(f"distance buffer re-added to a break condition: {buffer_terms}")
    else:
        _ok("no ± bos_disp / choch_disp term in any break condition (distance gate off)")

    # (c) The buffers must still be defined as 0.0 (kept for _true_break_idx + a
    #     future re-armed gate), not deleted.
    if "bos_disp = 0.0" in src and "choch_disp = 0.0" in src:
        _ok("bos_disp / choch_disp defined as 0.0 (gate off, walk-back symbol intact)")
    else:
        _bad("bos_disp / choch_disp are not both defined as 0.0 — buffer plumbing changed")

    # (d) _true_break_idx kept and wired into all six break branches (gap walk-back
    #     + re-arm readiness). 1 def + >=6 calls.
    if "def _true_break_idx(" not in src:
        _bad("_true_break_idx helper is GONE — gap walk-back / re-arm path lost")
    else:
        calls = len(re.findall(r"_true_break_idx\(", src))
        if calls >= 7:
            _ok(f"_true_break_idx referenced {calls}x (1 def + >=6 branch calls)")
        else:
            _bad(f"_true_break_idx referenced only {calls}x — expected >=7 (1 def + 6 branches)")

    # (e) confirm_idx still threaded so the delta is logged.
    if "confirm_idx=ci" in src:
        _ok("confirm_idx=ci threaded into _push_event (delta + break_body_atr logged)")
    else:
        _bad("confirm_idx=ci not passed to _push_event — event_candle_delta will be None")


def main():
    print("GATES-REMOVED break-model guards")
    print("- behavioral (frozen USDCHF window):")
    _behavioral()
    print("- source tripwire:")
    _source_tripwire()
    if _FAILS:
        print(f"\n{len(_FAILS)} FAIL(s)")
        sys.exit(1)
    print("\nall gates-removed guards: OK")


if __name__ == "__main__":
    main()
