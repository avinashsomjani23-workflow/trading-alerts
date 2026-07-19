"""Guards for the displacement-leg extreme + Kaufman ER (DISPLACEMENT_LEG_BUILD_SPEC).

Run:  python tests/test_displacement_leg.py
Exit 0 iff every guard passes.

The ONE rule these tests defend: every value is measured on THE EXACT leg that
caused the break -- from the OB candle, through the break, to the leg's structural
top. A later, unrelated move must NEVER pollute the extreme or the ER.

Cases (spec §8):
  1. Unrelated-later-high -- a higher high after the structural top must NOT change
     the extreme (the core one-rule guard).
  2. Span starts at the OB, not the deeper leg origin -- a low beneath the OB must
     not enter the ER path or move the extreme.
  3. Break candle included -- an extreme that IS the break candle's own high counts.
  4. Breather does not stop the leg -- a single non-extending bar mid-run does not
     end it; a later in-direction extreme (before any opposing swing) is included.
  5. Running-extreme fallback -- a still-running leg with no confirmed opposing
     swing yet -> extreme = running high/low to the last closed bar (not None).
  6. Look-ahead -- an unconfirmed opposing swing (confirmation bar > wall) is not
     used as the leg-end.
  7. ER math -- straight leg ~= 1.0; known choppy value; flat closes -> None; <3 -> None.
  8. Freeze/refresh -- values hold between alerts, refresh on re-arm (payload scalar
     behaviour, proven at the replay layer in test_structure_signals.py).
  9. Clipped / old-OB -- OB candle outside the frame -> (None, None, None).

Style mirrors tests/test_structure_signals.py: a plain assertion harness, no pytest
dependency, _bad RAISES so `pytest tests/ -q` (which never calls main) still goes red.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import displacement_leg  # noqa: E402
import dealing_range as dr  # noqa: E402

_FAILS = []
_RB = dr.SWING_LOOKBACK  # confirmation lag (default 3) -- never hardcode


def _ok(m): print(f"  OK:   {m}")
def _bad(m):
    print(f"  FAIL: {m}")
    _FAILS.append(m)
    raise AssertionError(m)


# ── frame builder ─────────────────────────────────────────────────────────────
# One UTC-hourly OHLC frame from a list of (open, high, low, close) tuples. The
# index is what the core resolves ob_ts / bos_ts / swing ts against.
_T0 = pd.Timestamp("2020-01-01T00:00:00", tz="UTC")


def _frame(bars):
    idx = pd.date_range(_T0, periods=len(bars), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "Open":  [b[0] for b in bars],
            "High":  [b[1] for b in bars],
            "Low":   [b[2] for b in bars],
            "Close": [b[3] for b in bars],
        },
        index=idx,
    )


def _ts(df, i):
    return df.index[i].isoformat()


def _swing(df, i, typ):
    return {"type": typ, "idx": i, "price": float(df["High"].iloc[i]), "ts": _ts(df, i)}


# ── Case 1: unrelated-later-high (THE one-rule guard) ─────────────────────────
def test_unrelated_later_high_ignored():
    # Bullish leg: OB at bar 1, break at bar 2, structural top at bar 4 (a
    # confirmed swing HIGH). A MUCH higher high prints at bar 9 -- well after the
    # leg structurally topped and turned. It must NOT enter the extreme.
    # bar 4 is the swing high; it needs _RB bars to its right to be confirmed, so
    # the wall (last bar) is placed comfortably past bar 4 + _RB.
    bars = [
        (10.0, 10.2, 9.8, 10.0),   # 0  pre-OB
        (10.0, 10.1, 9.9, 9.95),   # 1  OB (small down candle)
        (9.95, 10.6, 9.95, 10.55), # 2  BREAK (up impulse)
        (10.55, 10.8, 10.5, 10.75),# 3
        (10.75, 11.0, 10.7, 10.72),# 4  structural TOP (swing high) = 11.0
        (10.72, 10.8, 10.4, 10.45),# 5  turn down (confirms bar 4)
        (10.45, 10.6, 10.3, 10.35),# 6
        (10.35, 10.5, 10.2, 10.3), # 7
        (10.3, 10.6, 10.2, 10.4),  # 8
        (10.4, 13.0, 10.4, 12.9),  # 9  UNRELATED much higher high (must be ignored)
    ]
    df = _frame(bars)
    swings = [_swing(df, 4, "high")]  # confirmed: 4 + _RB = 7 <= wall (9)
    extreme, er, end_idx = displacement_leg.compute_leg_extreme_er(
        df, _ts(df, 1), _ts(df, 2), "bullish", swings, right_lookback=_RB)
    if extreme == 11.0 and end_idx == 4:
        _ok("Case1: extreme = leg's structural top (11.0), unrelated later high ignored")
    else:
        _bad(f"Case1: later high polluted extreme -- got {extreme} end_idx={end_idx} (want 11.0 @ 4)")


# ── Case 2: span starts at OB, not the deeper leg origin ──────────────────────
def test_span_starts_at_ob_not_origin():
    # A deep low sits BELOW the OB (the leg origin), and price dips between origin
    # and OB. The span must start at the OB (bar 3), so neither the origin low nor
    # the dip enters the ER path or moves the extreme.
    bars = [
        (10.0, 10.1, 8.0, 8.1),    # 0  deep leg ORIGIN low = 8.0 (below OB)
        (8.1, 8.3, 8.0, 8.2),      # 1  still low
        (8.2, 8.5, 8.1, 8.3),      # 2
        (9.90, 10.0, 9.85, 9.88),  # 3  OB (down candle, ABOVE the origin)
        (9.88, 10.8, 9.88, 10.7),  # 4  BREAK up
        (10.7, 11.2, 10.6, 11.0),  # 5  structural TOP = 11.2
        (11.0, 11.1, 10.5, 10.6),  # 6  turn (confirms bar 5)
        (10.6, 10.7, 10.3, 10.4),  # 7
        (10.4, 10.6, 10.2, 10.3),  # 8
    ]
    df = _frame(bars)
    swings = [_swing(df, 5, "high")]  # 5 + _RB = 8 <= wall (8)
    extreme, er, end_idx = displacement_leg.compute_leg_extreme_er(
        df, _ts(df, 3), _ts(df, 4), "bullish", swings, right_lookback=_RB)
    # Extreme is the top over [3,5] = 11.2, NOT influenced by the 8.0 origin.
    # ER first close = OB close (9.88), last = top-window close (11.0); the deep
    # low's steps are excluded (they are before ob_idx).
    if extreme == 11.2 and end_idx == 5 and er is not None and er > 0.5:
        _ok(f"Case2: span starts at OB -- extreme 11.2, origin low excluded (ER={er})")
    else:
        _bad(f"Case2: origin leaked -- extreme={extreme} end_idx={end_idx} er={er} (want 11.2 @ 5)")


# ── Case 3: break candle included ─────────────────────────────────────────────
def test_break_candle_included():
    # The extreme IS the break candle's own high. Span is inclusive of bos_idx, so
    # the extreme equals that high (not skipped as bos_idx+1).
    bars = [
        (10.0, 10.1, 9.9, 10.0),   # 0
        (10.0, 10.05, 9.9, 9.95),  # 1  OB
        (9.95, 12.0, 9.95, 10.5),  # 2  BREAK -- highest high of the whole leg = 12.0
        (10.5, 10.7, 10.3, 10.4),  # 3  structural top (lower) -- swing high @ 3
        (10.4, 10.5, 10.1, 10.2),  # 4  turn (confirms bar 3)
        (10.2, 10.4, 10.0, 10.1),  # 5
        (10.1, 10.3, 9.9, 10.0),   # 6
    ]
    df = _frame(bars)
    swings = [_swing(df, 3, "high")]  # 3 + _RB = 6 <= wall (6)
    extreme, er, end_idx = displacement_leg.compute_leg_extreme_er(
        df, _ts(df, 1), _ts(df, 2), "bullish", swings, right_lookback=_RB)
    if extreme == 12.0:
        _ok("Case3: break candle's own high (12.0) is the extreme (span includes bos_idx)")
    else:
        _bad(f"Case3: break candle excluded -- got {extreme} (want 12.0)")


# ── Case 4: breather does not stop the leg ────────────────────────────────────
def test_breather_does_not_stop_leg():
    # A single non-extending bar mid-run (a breather) is NOT a confirmed opposing
    # swing, so it must not end the leg. A NEW in-direction extreme after it,
    # before any opposing swing confirms, must be included.
    bars = [
        (10.0, 10.1, 9.9, 10.0),   # 0
        (10.0, 10.05, 9.9, 9.95),  # 1  OB
        (9.95, 10.6, 9.95, 10.55), # 2  BREAK
        (10.55, 10.8, 10.5, 10.75),# 3  extends
        (10.75, 10.78, 10.6, 10.65),# 4 BREATHER (does not extend, not a swing)
        (10.65, 11.5, 10.6, 11.4), # 5  NEW higher extreme = 11.5 (post-breather)
        (11.4, 11.6, 11.3, 11.45), # 6  structural top = 11.6 (swing high @ 6)
        (11.45, 11.5, 11.0, 11.1), # 7  turn (confirms bar 6)
        (11.1, 11.2, 10.9, 11.0),  # 8
        (11.0, 11.1, 10.8, 10.9),  # 9
    ]
    df = _frame(bars)
    swings = [_swing(df, 6, "high")]  # 6 + _RB = 9 <= wall (9)
    extreme, er, end_idx = displacement_leg.compute_leg_extreme_er(
        df, _ts(df, 1), _ts(df, 2), "bullish", swings, right_lookback=_RB)
    if extreme == 11.6 and end_idx == 6:
        _ok("Case4: breather did not stop the leg -- extreme includes post-breather 11.6")
    else:
        _bad(f"Case4: breather stopped the leg -- got {extreme} end_idx={end_idx} (want 11.6 @ 6)")


# ── Case 5: running-extreme fallback (Rule B) ─────────────────────────────────
def test_running_extreme_fallback():
    # Still-running leg: NO opposing swing has confirmed yet. extreme = running
    # high up to the last closed bar (the wall), not None.
    bars = [
        (10.0, 10.1, 9.9, 10.0),   # 0
        (10.0, 10.05, 9.9, 9.95),  # 1  OB
        (9.95, 10.6, 9.95, 10.55), # 2  BREAK
        (10.55, 10.9, 10.5, 10.85),# 3
        (10.85, 11.3, 10.8, 11.25),# 4  running high = 11.3 (wall)
    ]
    df = _frame(bars)
    extreme, er, end_idx = displacement_leg.compute_leg_extreme_er(
        df, _ts(df, 1), _ts(df, 2), "bullish", [], right_lookback=_RB)
    if extreme == 11.3 and end_idx == len(df) - 1:
        _ok("Case5: no confirmed opposing swing -> running extreme (11.3) to the wall")
    else:
        _bad(f"Case5: fallback wrong -- got {extreme} end_idx={end_idx} (want 11.3 @ {len(df)-1})")


# ── Case 6: look-ahead -- unconfirmed opposing swing is not the leg-end ────────
def test_lookahead_unconfirmed_swing_not_used():
    # A swing high sits at bar 4 but the wall (last bar) is only bar 5 -- the swing
    # needs _RB bars to its right to be confirmed (4 + _RB = 7 > 5). It is NOT yet
    # confirmed, so it can't be the leg-end -> Rule B running extreme to the wall.
    bars = [
        (10.0, 10.1, 9.9, 10.0),   # 0
        (10.0, 10.05, 9.9, 9.95),  # 1  OB
        (9.95, 10.6, 9.95, 10.55), # 2  BREAK
        (10.55, 10.9, 10.5, 10.85),# 3
        (10.85, 11.4, 10.8, 11.0), # 4  would-be swing high = 11.4, NOT confirmed yet
        (11.0, 11.6, 10.95, 11.5), # 5  wall -- running high = 11.6
    ]
    df = _frame(bars)
    swings = [_swing(df, 4, "high")]  # 4 + _RB = 7 > wall (5) -> not usable
    extreme, er, end_idx = displacement_leg.compute_leg_extreme_er(
        df, _ts(df, 1), _ts(df, 2), "bullish", swings, right_lookback=_RB)
    if extreme == 11.6 and end_idx == 5:
        _ok("Case6: unconfirmed opposing swing not used -> running extreme to the wall (11.6)")
    else:
        _bad(f"Case6: peeked at unconfirmed swing -- got {extreme} end_idx={end_idx} (want 11.6 @ 5)")


# ── Case 6b: swing between OB and break is NOT the leg top ─────────────────────
def test_swing_before_break_skipped():
    # There is always a swing high at the level the break broke (between OB and
    # break). It must be skipped -- the leg-end is the first confirmed opposing
    # swing STRICTLY AFTER bos_idx.
    bars = [
        (10.0, 10.3, 9.9, 10.0),   # 0  swing high @ 0 (the level the break clears)
        (10.0, 10.05, 9.6, 9.7),   # 1
        (9.7, 9.9, 9.6, 9.85),     # 2  OB (up? no) -- use bearish? keep bullish, OB is a down candle
        (9.85, 10.6, 9.85, 10.55), # 3  BREAK (clears the 10.3 high)
        (10.55, 11.0, 10.5, 10.9), # 4  top = 11.0 (swing high @ 4)
        (10.9, 10.95, 10.4, 10.5), # 5  turn (confirms bar 4)
        (10.5, 10.6, 10.2, 10.3),  # 6
        (10.3, 10.4, 10.1, 10.2),  # 7
    ]
    df = _frame(bars)
    # bar 2 is the OB (last down-ish candle before the break); bar 0 is a swing
    # high BEFORE bos_idx (3). Include it -- it must be skipped.
    swings = [_swing(df, 0, "high"), _swing(df, 4, "high")]  # 4 + _RB = 7 <= wall (7)
    extreme, er, end_idx = displacement_leg.compute_leg_extreme_er(
        df, _ts(df, 2), _ts(df, 3), "bullish", swings, right_lookback=_RB)
    if end_idx == 4 and extreme == 11.0:
        _ok("Case6b: pre-break swing skipped; leg-end is the first swing after bos (11.0 @ 4)")
    else:
        _bad(f"Case6b: wrong leg-end -- extreme={extreme} end_idx={end_idx} (want 11.0 @ 4)")


# ── Case 7: ER math ───────────────────────────────────────────────────────────
def test_er_math():
    # Straight leg: closes strictly monotone up -> ER == 1.0 exactly (net == path).
    straight = _frame([
        (1.0, 1.0, 1.0, 1.00),   # 0  OB
        (1.0, 1.0, 1.0, 1.10),   # 1  BREAK
        (1.0, 1.0, 1.0, 1.20),   # 2
        (1.0, 1.0, 1.0, 1.30),   # 3  top (swing high)
        (1.0, 1.0, 1.0, 1.25),   # 4  turn (confirms bar 3)
        (1.0, 1.0, 1.0, 1.20),   # 5
        (1.0, 1.0, 1.0, 1.15),   # 6
    ])
    sw = [{"type": "high", "idx": 3, "ts": _ts(straight, 3)}]  # 3 + _RB = 6 <= 6
    _, er, _ = displacement_leg.compute_leg_extreme_er(
        straight, _ts(straight, 0), _ts(straight, 1), "bullish", sw, right_lookback=_RB)
    if er == 1.0:
        _ok("Case7: straight monotone leg -> ER = 1.0")
    else:
        _bad(f"Case7: straight leg ER wrong -- got {er} (want 1.0)")

    # Known choppy value: closes 1.0, 1.2, 1.1, 1.3 over the span [0..3].
    #   net = |1.3 - 1.0| = 0.3 ; path = 0.2 + 0.1 + 0.2 = 0.5 ; ER = 0.6
    choppy = _frame([
        (1.0, 1.0, 1.0, 1.0),    # 0  OB
        (1.0, 1.0, 1.0, 1.2),    # 1  BREAK
        (1.0, 1.0, 1.0, 1.1),    # 2  pullback close
        (1.0, 1.0, 1.0, 1.3),    # 3  top (swing high)
        (1.0, 1.0, 1.0, 1.25),   # 4  turn (confirms bar 3)
        (1.0, 1.0, 1.0, 1.2),    # 5
        (1.0, 1.0, 1.0, 1.15),   # 6
    ])
    sw2 = [{"type": "high", "idx": 3, "ts": _ts(choppy, 3)}]
    _, er2, _ = displacement_leg.compute_leg_extreme_er(
        choppy, _ts(choppy, 0), _ts(choppy, 1), "bullish", sw2, right_lookback=_RB)
    if er2 == 0.6:
        _ok("Case7: known choppy leg -> ER = 0.6 (net 0.3 / path 0.5)")
    else:
        _bad(f"Case7: choppy leg ER wrong -- got {er2} (want 0.6)")

    # Flat closes over the span -> denom 0 -> ER None (never a faked 0.0).
    flat = _frame([
        (1.0, 1.5, 0.5, 1.0),    # 0  OB
        (1.0, 1.5, 0.5, 1.0),    # 1  BREAK (flat close)
        (1.0, 1.5, 0.5, 1.0),    # 2
        (1.0, 1.5, 0.5, 1.0),    # 3  swing high (equal highs OK; ER path is closes)
        (1.0, 1.5, 0.5, 1.0),    # 4
        (1.0, 1.5, 0.5, 1.0),    # 5
        (1.0, 1.5, 0.5, 1.0),    # 6
    ])
    sw3 = [{"type": "high", "idx": 3, "ts": _ts(flat, 3)}]
    _, er3, _ = displacement_leg.compute_leg_extreme_er(
        flat, _ts(flat, 0), _ts(flat, 1), "bullish", sw3, right_lookback=_RB)
    if er3 is None:
        _ok("Case7: flat-close leg -> ER None (denom 0, never a faked 0.0)")
    else:
        _bad(f"Case7: flat leg ER should be None -- got {er3}")

    # < 3 bars -> ER None. OB and break adjacent, no room; running fallback with a
    # 2-bar span.
    tiny = _frame([
        (1.0, 1.1, 0.9, 1.0),    # 0  OB
        (1.0, 1.3, 0.95, 1.2),   # 1  BREAK = wall (span [0,1] = 2 bars)
    ])
    _, er4, _ = displacement_leg.compute_leg_extreme_er(
        tiny, _ts(tiny, 0), _ts(tiny, 1), "bullish", [], right_lookback=_RB)
    if er4 is None:
        _ok("Case7: 2-bar leg -> ER None (< 3 bars)")
    else:
        _bad(f"Case7: 2-bar leg ER should be None -- got {er4}")


# ── Case 9: clipped / old-OB and other degenerates -> None triple ──────────────
def test_degenerate_returns_none_triple():
    df = _frame([
        (10.0, 10.1, 9.9, 10.0),
        (10.0, 10.6, 9.95, 10.55),
        (10.55, 10.9, 10.5, 10.8),
        (10.8, 11.0, 10.7, 10.9),
    ])
    # OB ts not in the frame (predates the slice) -> None triple, never a guess.
    old_ob_ts = (df.index[0] - pd.Timedelta(hours=5)).isoformat()
    r = displacement_leg.compute_leg_extreme_er(
        df, old_ob_ts, _ts(df, 1), "bullish", [], right_lookback=_RB)
    if r == (None, None, None):
        _ok("Case9: OB predates the frame -> (None, None, None), never a clipped guess")
    else:
        _bad(f"Case9: old-OB should be None triple -- got {r}")

    # ob_idx > bos_idx (degenerate) -> None triple.
    r2 = displacement_leg.compute_leg_extreme_er(
        df, _ts(df, 2), _ts(df, 1), "bullish", [], right_lookback=_RB)
    if r2 == (None, None, None):
        _ok("Case9: ob_idx > bos_idx -> (None, None, None)")
    else:
        _bad(f"Case9: ob>bos should be None triple -- got {r2}")

    # bos ts unresolvable -> None triple.
    r3 = displacement_leg.compute_leg_extreme_er(
        df, _ts(df, 0), "not-a-timestamp-xyz", "bullish", [], right_lookback=_RB)
    if r3 == (None, None, None):
        _ok("Case9: unresolvable bos ts -> (None, None, None)")
    else:
        _bad(f"Case9: bad bos ts should be None triple -- got {r3}")

    # empty frame -> None triple, never raises.
    r4 = displacement_leg.compute_leg_extreme_er(
        _frame([]), _ts(df, 0), _ts(df, 1), "bullish", [], right_lookback=_RB)
    if r4 == (None, None, None):
        _ok("Case9: empty frame -> (None, None, None), no raise")
    else:
        _bad(f"Case9: empty frame should be None triple -- got {r4}")


# ── Bearish mirror: extreme is the lowest Low ─────────────────────────────────
def test_bearish_extreme_is_lowest_low():
    bars = [
        (10.0, 10.1, 9.9, 10.0),   # 0
        (10.0, 10.1, 9.95, 10.05), # 1  OB (small up candle before down impulse)
        (10.05, 10.05, 9.4, 9.45), # 2  BREAK down
        (9.45, 9.5, 9.0, 9.1),     # 3  structural bottom = 9.0 (swing low @ 3)
        (9.1, 9.4, 9.05, 9.35),    # 4  turn up (confirms bar 3)
        (9.35, 9.6, 9.3, 9.5),     # 5
        (9.5, 9.7, 9.4, 9.6),      # 6
        (9.6, 12.0, 9.55, 11.9),   # 7  unrelated later action -- must not matter
    ]
    df = _frame(bars)
    swings = [{"type": "low", "idx": 3, "ts": _ts(df, 3)}]  # 3 + _RB = 6 <= wall (7)
    extreme, er, end_idx = displacement_leg.compute_leg_extreme_er(
        df, _ts(df, 1), _ts(df, 2), "bearish", swings, right_lookback=_RB)
    if extreme == 9.0 and end_idx == 3:
        _ok("Bearish: extreme = lowest Low at the structural bottom (9.0 @ 3)")
    else:
        _bad(f"Bearish: wrong -- extreme={extreme} end_idx={end_idx} (want 9.0 @ 3)")


def main():
    print("== Case 1: unrelated later high ignored ==")
    test_unrelated_later_high_ignored()
    print("\n== Case 2: span starts at OB, not origin ==")
    test_span_starts_at_ob_not_origin()
    print("\n== Case 3: break candle included ==")
    test_break_candle_included()
    print("\n== Case 4: breather does not stop the leg ==")
    test_breather_does_not_stop_leg()
    print("\n== Case 5: running-extreme fallback ==")
    test_running_extreme_fallback()
    print("\n== Case 6: look-ahead / unconfirmed swing ==")
    test_lookahead_unconfirmed_swing_not_used()
    test_swing_before_break_skipped()
    print("\n== Case 7: ER math ==")
    test_er_math()
    print("\n== Case 9: degenerate -> None triple ==")
    test_degenerate_returns_none_triple()
    print("\n== Bearish mirror ==")
    test_bearish_extreme_is_lowest_low()
    print("\nPASSED: displacement-leg extreme + ER guards green "
          "(DISPLACEMENT_LEG_BUILD_SPEC §8)")


if __name__ == "__main__":
    main()
    if _FAILS:
        sys.exit(1)
