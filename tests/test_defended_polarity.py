"""Guard: `defended` tracks the LAST confirmed counter-trend swing — BOTH
polarities (2026-07-16 fix).

The bug this guards against (proven live, USDCHF 2026-07-16):
  compute_structure sec. 4 moved `defended` only on made_lh (a LOWER high in a
  downtrend) / made_hl (a HIGHER low in an uptrend). A counter-polarity swing —
  a HIGHER confirmed high in a downtrend, a LOWER confirmed low in an uptrend —
  left `defended` stuck on the stale older swing. A close BETWEEN the stale
  level and the true last swing then fired a CHoCH that never actually broke
  the last confirmed swing (USDCHF: defended stuck at 0.80635, true swing high
  0.80649, close 0.80643 -> false CHoCH-up -> false bullish OB / Phase-2 zone).

Invariants guarded (all offline, synthetic candles — never the live path):
  1. DOWNTREND + higher confirmed high: `defended` moves UP to the new high; a
     close between old and new high fires NO CHoCH; a close above the new high
     fires the CHoCH at the NEW level.
  2. UPTREND + lower confirmed low (exact mirror): `defended` moves DOWN; no
     false CHoCH-down between the two lows; the real CHoCH breaks the new low.
  3. Normal path unchanged: a plain LH in a downtrend still becomes `defended`
     and its break still fires the CHoCH (no overcorrection).

Silent-failure mode guarded: the one-polarity update regresses (someone
restores `if made_lh:` around the defended update) — alerts would silently
fire CHoCHs off stale swings again, with no crash.

Run:  python tests/test_defended_polarity.py
Exit 0 iff every guard passes. No pytest dependency (house style).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import dealing_range as dr  # noqa: E402

_FAILS = []


def _ok(m): print(f"  OK:   {m}")


def _bad(m):
    # RAISE, don't just collect: CI runs these via `pytest tests/ -q`, which
    # never calls main(). See test_event_candle_fix for the history.
    print(f"  FAIL: {m}")
    _FAILS.append(m)
    raise AssertionError(m)


# ── Synthetic H1 bars ────────────────────────────────────────────────────────
# (High, Low, Close) per bar; Open = previous Close. Hourly, mid-week, no
# session gaps (so _traded_through is always True and gap logic never bites).
#
# Path (bearish scenario — the USDCHF shape):
#   idx 4   swing high A = 1.1050 (confirmed idx 7)
#   idx 10  swing low  L1 = 1.0945 (confirmed idx 13)
#   idx 14  BIRTH DOWN (close 1.0938 < L1) — defended = A
#   idx 17  swing low  L2 = 1.0895 (confirmed idx 20)
#   idx 23  swing high B = 1.0990  LH (confirmed idx 26) — defended = B
#   idx 28  continuation BOS down (close 1.0892 < L2) — expected, benign
#   idx 30  swing low  L3 = 1.0875 (confirmed idx 33)
#   idx 36  swing high C = 1.1000  HIGHER high, wick only — every close < B
#           (confirmed idx 39) -> defended must move to C
#   idx 40  PROBE: close 1.0995, between B (1.0990) and C (1.1000)
#           -> the buggy engine fired a false CHoCH here; fixed engine: nothing
#   idx 41  BREAK: close 1.1010 > C -> the real CHoCH, level = C
_BARS = [
    (1.1005, 1.0990, 1.1000),   # 0
    (1.1015, 1.0995, 1.1010),   # 1
    (1.1030, 1.1005, 1.1025),   # 2
    (1.1042, 1.1020, 1.1038),   # 3
    (1.1050, 1.1032, 1.1044),   # 4  A pivot
    (1.1046, 1.1020, 1.1025),   # 5
    (1.1035, 1.1005, 1.1010),   # 6
    (1.1020, 1.0988, 1.0995),   # 7  A confirmed
    (1.1000, 1.0968, 1.0975),   # 8
    (1.0985, 1.0952, 1.0958),   # 9
    (1.0962, 1.0945, 1.0952),   # 10 L1 pivot
    (1.0968, 1.0950, 1.0962),   # 11
    (1.0975, 1.0955, 1.0970),   # 12
    (1.0980, 1.0958, 1.0965),   # 13 L1 confirmed
    (1.0966, 1.0930, 1.0938),   # 14 birth DOWN
    (1.0945, 1.0915, 1.0920),   # 15
    (1.0928, 1.0898, 1.0902),   # 16
    (1.0912, 1.0895, 1.0900),   # 17 L2 pivot
    (1.0918, 1.0900, 1.0912),   # 18
    (1.0930, 1.0908, 1.0925),   # 19
    (1.0944, 1.0920, 1.0940),   # 20 L2 confirmed
    (1.0962, 1.0935, 1.0955),   # 21
    (1.0978, 1.0950, 1.0972),   # 22
    (1.0990, 1.0968, 1.0980),   # 23 B pivot (LH)
    (1.0982, 1.0955, 1.0960),   # 24
    (1.0970, 1.0940, 1.0945),   # 25
    (1.0955, 1.0925, 1.0930),   # 26 B confirmed -> defended = 1.0990
    (1.0935, 1.0905, 1.0910),   # 27
    (1.0920, 1.0885, 1.0892),   # 28 continuation BOS down (< L2)
    (1.0900, 1.0878, 1.0882),   # 29
    (1.0890, 1.0875, 1.0885),   # 30 L3 pivot
    (1.0908, 1.0880, 1.0902),   # 31
    (1.0925, 1.0898, 1.0920),   # 32
    (1.0942, 1.0915, 1.0938),   # 33 L3 confirmed
    (1.0960, 1.0932, 1.0955),   # 34
    (1.0975, 1.0948, 1.0970),   # 35
    (1.1000, 1.0965, 1.0985),   # 36 C pivot — HIGHER high, close < B
    (1.0988, 1.0958, 1.0968),   # 37
    (1.0980, 1.0952, 1.0960),   # 38
    (1.0972, 1.0945, 1.0955),   # 39 C confirmed -> defended must = 1.1000
    (1.0998, 1.0970, 1.0995),   # 40 PROBE: between B and C — must NOT CHoCH
    (1.1015, 1.0985, 1.1010),   # 41 BREAK: above C — real CHoCH at 1.1000
]

_A, _B, _C = 1.1050, 1.0990, 1.1000
_PROBE_IDX, _BREAK_IDX = 40, 41

# Mirror constant: price' = _K - price flips the whole path vertically, turning
# the bearish scenario into its exact bullish mirror (uptrend + LOWER low).
_K = 2.2000


def _mk_df(bars, mirror=False):
    rows = []
    prev_c = None
    for (h, l, c) in bars:
        if mirror:
            h, l, c = _K - l, _K - h, _K - c
        o = prev_c if prev_c is not None else c
        rows.append((o, h, l, c, 0.0))
        prev_c = c
    idx = pd.date_range("2026-07-14 00:00", periods=len(rows),
                        freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close",
                                       "Volume"], index=idx)


def _run(df):
    return dr.compute_structure(df, None)


def _chochs(out):
    return [e for e in (out.get("events") or []) if e.get("type") == "CHoCH"]


# ── 1. Downtrend + HIGHER confirmed high (the proven live bug) ──────────────
def _bearish_scenario():
    df = _mk_df(_BARS)

    # At the probe bar: defended must already be C, and NO CHoCH may fire.
    out = _run(df.iloc[:_PROBE_IDX + 1])
    if out["state"] != "down":
        _bad(f"S1 probe: state={out['state']!r}, expected 'down'")
    if _chochs(out):
        _bad(f"S1 probe: false CHoCH fired between stale LH ({_B}) and the "
             f"true higher high ({_C}) — defended is stuck on the stale swing")
    if out["defended"] != _C:
        _bad(f"S1 probe: defended={out['defended']} — must track the LAST "
             f"confirmed swing high {_C}, not the stale {_B}")
    if out["flip_unconfirmed"]:
        _bad("S1 probe: flip_unconfirmed True — no reversal should be pending")
    _ok(f"S1: higher high in a downtrend moved defended {_B} -> {_C}; close "
        f"between the two fired NOTHING")

    # At the break bar: the real CHoCH fires, at the NEW level.
    out = _run(df.iloc[:_BREAK_IDX + 1])
    ch = _chochs(out)
    if len(ch) != 1 or ch[-1]["direction"] != "bullish":
        _bad(f"S1 break: expected exactly one bullish CHoCH, got {ch}")
    if out["choch_level"] != _C:
        _bad(f"S1 break: choch_level={out['choch_level']} — the CHoCH must "
             f"break the true last swing high {_C}")
    if not out["flip_unconfirmed"]:
        _bad("S1 break: CHoCH armed but flip_unconfirmed is False")
    _ok(f"S1: close above {_C} fired the real CHoCH at the new level")


# ── 2. Uptrend + LOWER confirmed low (exact mirror) ──────────────────────────
def _bullish_scenario():
    df = _mk_df(_BARS, mirror=True)
    b_m, c_m = _K - _B, _K - _C   # mirrored HL / lower-low levels

    out = _run(df.iloc[:_PROBE_IDX + 1])
    if out["state"] != "up":
        _bad(f"S2 probe: state={out['state']!r}, expected 'up'")
    if _chochs(out):
        _bad(f"S2 probe: false CHoCH-down fired between stale HL ({b_m}) and "
             f"the true lower low ({c_m})")
    if round(out["defended"], 6) != round(c_m, 6):
        _bad(f"S2 probe: defended={out['defended']} — must track the LAST "
             f"confirmed swing low {c_m}, not the stale {b_m}")
    _ok(f"S2: lower low in an uptrend moved defended {b_m:.4f} -> {c_m:.4f}; "
        f"close between the two fired NOTHING")

    out = _run(df.iloc[:_BREAK_IDX + 1])
    ch = _chochs(out)
    if len(ch) != 1 or ch[-1]["direction"] != "bearish":
        _bad(f"S2 break: expected exactly one bearish CHoCH, got {ch}")
    if round(out["choch_level"], 6) != round(c_m, 6):
        _bad(f"S2 break: choch_level={out['choch_level']} != {c_m}")
    _ok(f"S2: close below {c_m:.4f} fired the real CHoCH at the new level")


# ── 3. Normal LH path unchanged (no overcorrection) ──────────────────────────
def _normal_lh_scenario():
    # Truncate before the higher high ever forms; break the plain LH (B).
    bars = _BARS[:27] + [(1.1008, 1.0975, 1.1000)]  # close 1.1000 > B
    df = _mk_df(bars)
    out = _run(df)
    ch = _chochs(out)
    if len(ch) != 1 or ch[-1]["direction"] != "bullish":
        _bad(f"S3: plain LH break: expected one bullish CHoCH, got {ch}")
    if out["choch_level"] != _B:
        _bad(f"S3: choch_level={out['choch_level']} != LH {_B} — normal "
             f"made_lh path regressed")
    _ok(f"S3: plain LH ({_B}) still defends and its break still fires the "
        f"CHoCH — normal path intact")


def main():
    print("defended-polarity guard (counter-trend swing tracking):")
    _bearish_scenario()
    _bullish_scenario()
    _normal_lh_scenario()
    print("ALL GUARDS PASSED" if not _FAILS else f"{len(_FAILS)} FAILURES")
    return 1 if _FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
