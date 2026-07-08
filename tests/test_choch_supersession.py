"""Regression guard for CHoCH-supersession retirement (smc_radar.is_choch_superseded).

Run:  python tests/test_choch_supersession.py
Exit 0 iff every case passes.

WHY THIS GUARD EXISTS (silent failure class):
  A CHoCH-born OB whose flip has been CONFIRMED and CONTINUED (>= 2 later
  same-direction BOS) is dead history — price already ran and left it behind.
  Before 2026-07-08 the live slate had no rule to retire it: such a zone sits
  far from current structure, so structure_supplanted / out_of_proximity never
  fire, and it kept re-surfacing as a live alert (the 2026-07-08 NZDUSD 8.0
  LONG on a June-30 CHoCH). The two silent ways this can regress:
    1) rule stops firing  -> dead zones re-alert (money-losing counter-trend).
    2) rule fires too eagerly (>=1 BOS, or ignores direction) -> it KILLS a
       live, confirmed setup (the freshest valid OB after a single BOS).
  Both are silent — no crash, just a wrong alert. This test pins the exact
  boundary (CHoCH + exactly-2 same-direction BOS) against the live data shape.

Direct-call: exercises the SAME function the live reconcile loop calls
(smc_radar.is_choch_superseded), against event rings shaped like the live
structure ring (type / direction / candle_ts).
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
    print(f"  FAIL: {m}")
    _FAILS.append(m)


def _ev(t, d, ts):
    return {"type": t, "direction": d, "candle_ts": ts}


def _check(name, zone, events, expected):
    got = smc_radar.is_choch_superseded(zone, events)
    (_ok if got == expected else _bad)(f"{name}: got {got}, expected {expected}")


# The live 2026-07-08 NZDUSD ring: 3 bullish BOS, no CHoCH left in the ring.
NZD_RING = [
    _ev("BOS", "bullish", "2026-07-02T08:00:00+00:00"),
    _ev("BOS", "bullish", "2026-07-02T12:00:00+00:00"),
    _ev("BOS", "bullish", "2026-07-08T07:00:00+00:00"),
]


def _zone(bos_tag, direction, bos_ts):
    return {"bos_tag": bos_tag, "direction": direction, "bos_timestamp": bos_ts}


def main():
    print("CHoCH-supersession retirement guard")

    # --- THE TARGET CASE: June-30 bullish CHoCH, 3 later bullish BOS -> retire.
    _check("nzd34 dead choch (3 later bull BOS)",
           _zone("CHoCH", "bullish", "2026-06-30T14:00:00+00:00"),
           NZD_RING, True)

    # --- BOUNDARY: exactly 2 same-direction BOS -> retire (the approved rule).
    ring2 = [
        _ev("BOS", "bullish", "2026-07-02T08:00:00+00:00"),
        _ev("BOS", "bullish", "2026-07-02T12:00:00+00:00"),
    ]
    _check("exactly 2 same-dir BOS -> retire",
           _zone("CHoCH", "bullish", "2026-06-30T14:00:00+00:00"),
           ring2, True)

    # --- BOUNDARY: only 1 same-direction BOS -> KEEP (confirmed, not continued).
    # This is the freshest valid OB; retiring here would kill a live setup.
    ring1 = [_ev("BOS", "bullish", "2026-07-02T08:00:00+00:00")]
    _check("only 1 same-dir BOS -> keep",
           _zone("CHoCH", "bullish", "2026-06-30T14:00:00+00:00"),
           ring1, False)

    # --- DIRECTION GUARD: 2 OPPOSITE-direction BOS -> KEEP (different lifecycle).
    ring_opp = [
        _ev("BOS", "bearish", "2026-07-02T08:00:00+00:00"),
        _ev("BOS", "bearish", "2026-07-02T12:00:00+00:00"),
    ]
    _check("2 opposite-dir BOS -> keep",
           _zone("CHoCH", "bullish", "2026-06-30T14:00:00+00:00"),
           ring_opp, False)

    # --- TYPE GUARD: only CHoCH-born zones are subject to this rule.
    _check("BOS-born zone -> keep (not a CHoCH)",
           _zone("BOS", "bullish", "2026-07-02T12:00:00+00:00"),
           NZD_RING, False)

    # --- ORDERING GUARD: same-dir BOS that are EARLIER than the CHoCH do not count.
    ring_before = [
        _ev("BOS", "bullish", "2026-06-29T08:00:00+00:00"),
        _ev("BOS", "bullish", "2026-06-29T12:00:00+00:00"),
    ]
    _check("2 same-dir BOS BEFORE the CHoCH -> keep",
           _zone("CHoCH", "bullish", "2026-06-30T14:00:00+00:00"),
           ring_before, False)

    # --- CHoCH events in the ring are NOT BOS -> do not count toward supersession.
    ring_choch = [
        _ev("CHoCH", "bullish", "2026-07-02T08:00:00+00:00"),
        _ev("CHoCH", "bullish", "2026-07-02T12:00:00+00:00"),
    ]
    _check("2 later same-dir CHoCH (not BOS) -> keep",
           _zone("CHoCH", "bullish", "2026-06-30T14:00:00+00:00"),
           ring_choch, False)

    # --- LIVE PENDING CHoCH: today's flip, 0 later BOS -> keep.
    _check("fresh pending CHoCH (0 later BOS) -> keep",
           _zone("CHoCH", "bullish", "2026-07-08T07:00:00+00:00"),
           NZD_RING, False)

    # --- SAFETY: missing / malformed fields never retire (never kill on a bug).
    _check("empty dict -> keep", {}, NZD_RING, False)
    _check("None events -> keep",
           _zone("CHoCH", "bullish", "2026-06-30T14:00:00+00:00"), None, False)
    _check("unparseable bos_timestamp -> keep",
           _zone("CHoCH", "bullish", "garbage"), NZD_RING, False)
    _check("missing direction -> keep",
           {"bos_tag": "CHoCH", "bos_timestamp": "2026-06-30T14:00:00+00:00"},
           NZD_RING, False)

    print()
    if _FAILS:
        print(f"FAILED ({len(_FAILS)} case(s))")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
