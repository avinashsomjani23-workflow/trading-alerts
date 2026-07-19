"""Unit tests for the ATR-scaled same-leg OB dedupe (DETECTION_FIXES_SPEC Fix 2).

Run:  python tests/test_dedupe.py
Exit 0 iff every case passes.

Covers:
  - the threshold DERIVATION rule (DEDUPE_PROXIMAL_ATR_MULT x H1 ATR, with the
    0.00030 forex fallback when ATR is None/0) — this is the live/backtest edit
    at smc_radar.py, computed just before the _dedupe_same_leg_impl call.
  - the dedupe DECISION (_dedupe_same_leg_impl): same-direction OBs within the
    supplied threshold merge to one; outside it, both survive.

Why direct-call: the production code computes `thresh` inline and passes it to
the module-level _dedupe_same_leg_impl. We test the same arithmetic and the same
function the production path uses — no re-implementation of the merge ladder.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import smc_radar  # noqa: E402

_FAILS = []


def _ok(m): print(f"  OK:   {m}")
def _bad(m):
    # RAISE, don't just collect: CI runs these via `pytest tests/ -q`, which
    # never calls main(). A print-and-append _bad is invisible to pytest -> the
    # guard is green even when the code is broken (Deep Value A, 2026-07-10).
    print(f"  FAIL: {m}")
    _FAILS.append(m)
    raise AssertionError(m)


def _thresh(atr):
    """Mirror of the production threshold derivation (smc_radar detect_smc_radar):
    ATR-scaled when ATR is usable, else the 0.00030 forex fallback."""
    return (smc_radar.DEDUPE_PROXIMAL_ATR_MULT * atr
            if atr and atr > 0 else 0.00030)


def _ob(direction, proximal, bos_idx, touches=0, fvg_exists=False):
    """Minimal OB dict with only the fields the dedupe ladder reads."""
    return {
        "direction": direction,
        "proximal_line": float(proximal),
        "bos_idx": int(bos_idx),
        "touches": int(touches),
        "fvg": {"exists": bool(fvg_exists)},
    }


# --- 1) JPY scale: proximals 0.02 apart, ATR 0.5 -> MUST merge --------------

def test_jpy_merge():
    atr = 0.5
    thr = _thresh(atr)                       # 0.25 * 0.5 = 0.125
    a = _ob("bullish", 150.00, bos_idx=10)
    b = _ob("bullish", 150.02, bos_idx=12)   # 0.02 apart << 0.125
    kept = smc_radar._dedupe_same_leg_impl([a, b], thr)
    if len(kept) == 1:
        _ok(f"JPY: 0.02 apart, ATR 0.5 (thr={thr:g}) -> merged to 1")
    else:
        _bad(f"JPY: expected merge to 1, got {len(kept)} (thr={thr:g})")


# --- 2) EURUSD scale: merge below thr, no-merge above ----------------------

def test_eurusd_merge():
    atr = 0.0012
    thr = _thresh(atr)                       # 1.0 * 0.0012 = 0.0012
    a = _ob("bearish", 1.10000, bos_idx=10)
    b = _ob("bearish", 1.10020, bos_idx=12)  # 0.0002 apart (0.17 ATR) << thr
    kept = smc_radar._dedupe_same_leg_impl([a, b], thr)
    if len(kept) == 1:
        _ok(f"EURUSD: 0.0002 apart, ATR 0.0012 (thr={thr:g}) -> merged to 1")
    else:
        _bad(f"EURUSD merge: expected 1, got {len(kept)} (thr={thr:g})")


def test_eurusd_merge_under_one_atr():
    # 2026-07-17: at the widened 1.0-ATR threshold, a 0.83-ATR gap that used to
    # survive (old 0.25-ATR rule) now correctly merges — this is the USDCHF fix.
    atr = 0.0012
    thr = _thresh(atr)                       # 0.0012
    a = _ob("bearish", 1.10000, bos_idx=10)
    b = _ob("bearish", 1.10100, bos_idx=12)  # 0.0010 apart (0.83 ATR) < thr
    kept = smc_radar._dedupe_same_leg_impl([a, b], thr)
    if len(kept) == 1:
        _ok(f"EURUSD: 0.83 ATR apart, ATR 0.0012 (thr={thr:g}) -> merged to 1")
    else:
        _bad(f"EURUSD under-1-ATR merge: expected 1, got {len(kept)} (thr={thr:g})")


def test_eurusd_no_merge():
    # Gap > 1.0 ATR -> genuinely separate zones, both survive.
    atr = 0.0012
    thr = _thresh(atr)                       # 0.0012
    a = _ob("bearish", 1.10000, bos_idx=10)
    b = _ob("bearish", 1.10200, bos_idx=12)  # 0.0020 apart (1.67 ATR) > thr
    kept = smc_radar._dedupe_same_leg_impl([a, b], thr)
    if len(kept) == 2:
        _ok(f"EURUSD: 1.67 ATR apart, ATR 0.0012 (thr={thr:g}) -> both survive")
    else:
        _bad(f"EURUSD no-merge: expected 2, got {len(kept)} (thr={thr:g})")


def test_span_cap_prevents_chaining():
    # THREE same-direction OBs, each 0.8 ATR from the next (proximals 0, 0.8,
    # 1.6 ATR). Naive "merge into nearest survivor" would chain all three into
    # one zone spanning 1.6 ATR. The span cap must keep the merged zone <= 1.0
    # ATR: OB1+OB2 merge (0.8 <= 1.0); OB3 is 1.6 ATR from OB1 -> cannot join,
    # survives on its own. Expect 2 zones, not 1.
    atr = 0.0012
    thr = _thresh(atr)                       # 0.0012 (1.0 ATR)
    step = 0.8 * atr                         # 0.00096
    a = _ob("bullish", 1.10000, bos_idx=10)
    b = _ob("bullish", 1.10000 + step, bos_idx=11)      # 0.8 ATR from a
    c = _ob("bullish", 1.10000 + 2 * step, bos_idx=12)  # 1.6 ATR from a
    kept = smc_radar._dedupe_same_leg_impl([a, b, c], thr)
    if len(kept) == 2:
        _ok("span cap: 3 OBs at 0.8-ATR steps -> 2 zones (no >1 ATR chaining)")
    else:
        _bad(f"span cap: expected 2 zones, got {len(kept)} "
             f"(proximals {[round(o['proximal_line'],5) for o in kept]})")


# --- 3) ATR None/0 -> falls back to 0.0003 behaviour -----------------------

def test_atr_fallback():
    for atr in (None, 0, 0.0):
        thr = _thresh(atr)
        if thr != 0.00030:
            _bad(f"ATR {atr!r}: expected fallback 0.00030, got {thr}")
            continue
        # At the forex fallback, 0.0002 apart merges; 0.0010 apart does not.
        merged = smc_radar._dedupe_same_leg_impl(
            [_ob("bullish", 1.10000, 10), _ob("bullish", 1.10020, 12)], thr)
        split = smc_radar._dedupe_same_leg_impl(
            [_ob("bullish", 1.10000, 10), _ob("bullish", 1.10100, 12)], thr)
        if len(merged) == 1 and len(split) == 2:
            _ok(f"ATR {atr!r}: fallback 0.00030 behaviour preserved")
        else:
            _bad(f"ATR {atr!r}: fallback behaviour wrong "
                 f"(merge={len(merged)}, split={len(split)})")


# --- 4) guards: opposite directions never merge; ladder picks pristine ------

def test_opposite_direction_never_merges():
    thr = _thresh(0.5)
    a = _ob("bullish", 150.00, 10)
    b = _ob("bearish", 150.00, 12)   # identical proximal, opposite direction
    kept = smc_radar._dedupe_same_leg_impl([a, b], thr)
    if len(kept) == 2:
        _ok("opposite-direction OBs at same proximal -> both survive")
    else:
        _bad(f"opposite direction: expected 2, got {len(kept)}")


def test_pristine_wins_on_merge():
    thr = _thresh(0.0012)
    tested = _ob("bullish", 1.10000, bos_idx=20, touches=2)
    pristine = _ob("bullish", 1.10010, bos_idx=10, touches=0)  # within thr
    kept = smc_radar._dedupe_same_leg_impl([tested, pristine], thr)
    if len(kept) == 1 and kept[0]["touches"] == 0:
        _ok("merge keeps the Pristine OB over the Tested one (Test 1)")
    else:
        _bad(f"pristine-wins: got {[o['touches'] for o in kept]}")


def main():
    print("== threshold + merge (JPY / EURUSD scale) ==")
    test_jpy_merge()
    test_eurusd_merge()
    test_eurusd_merge_under_one_atr()
    test_eurusd_no_merge()
    print("\n== span cap (anti-chaining) ==")
    test_span_cap_prevents_chaining()
    print("\n== ATR-unavailable fallback ==")
    test_atr_fallback()
    print("\n== ladder guards ==")
    test_opposite_direction_never_merges()
    test_pristine_wins_on_merge()
    print()
    if _FAILS:
        print(f"FAILED: {len(_FAILS)} problem(s)")
        return 1
    print("PASSED: ATR-scaled dedupe merges on-scale and falls back correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
