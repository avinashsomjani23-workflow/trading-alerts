"""Displacement-leg extreme (structural) + Kaufman Efficiency Ratio.

ONE shared core for backtest AND live (DISPLACEMENT_LEG_BUILD_SPEC.md). Grades
the displacement of the order block: from the OB candle, through the break, to
the leg's STRUCTURAL top — never a later, unrelated move.

Two observe-only values (gate/score/filter NOTHING):
  * leg_extreme_at_alert — highest High (bullish) / lowest Low (bearish) over the
    span [ob_idx, extreme_end_idx], inclusive of the break candle.
  * leg_er_at_alert — Kaufman Efficiency Ratio over the SAME span's closes:
    |C[last]-C[first]| / Σ|ΔC|. 1 = one straight displacement; near 0 = a grind.

The one rule above all: every value is measured on THE EXACT leg that caused the
break. If a value could include price action from after the leg structurally
ended, the implementation is WRONG.

Interface is TIMESTAMPS-IN, values out (spec §1.3 — the #1 trap): ob_idx/bos_idx
and swing idx are detection-frame integers, valid only in the frame they were
resolved in. Backtest rebuilds them on the same slice (idx valid); live loads
them from persisted Phase-1 state (idx STALE). So the core takes df + timestamps
and re-resolves every ts->idx against the passed df. One code path, both callers
correct. Never trusts an incoming integer index.

NEVER raises: the whole body is wrapped so a failure returns (None, None, None).
Safe to call inside the live alert path (CLAUDE.md live-path guard rule).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import smc_detector      # ts_to_utc_instant (tz-agnostic instant matching)
import dealing_range     # SWING_LOOKBACK (confirmation lag) — do not hardcode 3

_MIN_LEG_BARS = 3        # ER needs >= 3 bars (2 close-to-close steps) to mean anything


def _resolve_idx(df, raw_ts):
    """ts -> integer position in df, or None. Instant-not-string matching.

    Mirrors the chart X-marker resolution: normalise to a UTC instant
    (smc_detector.ts_to_utc_instant) then get_indexer, so a tz-variant
    serialization of the SAME moment still lands on the right candle. A raw
    string `==` silently misses '...-04:00' vs '...+00:00'. Returns None when
    the ts is unresolvable or falls outside the frame."""
    inst = smc_detector.ts_to_utc_instant(raw_ts)
    if inst is None:
        return None
    idx = df.index
    # Align the lookup instant to the frame's own tz-awareness before indexing.
    if getattr(idx, "tz", None) is not None:
        lookup = inst.tz_convert(idx.tz)
    else:
        lookup = inst.tz_convert("UTC").tz_localize(None)
    pos = idx.get_indexer([lookup])
    p = int(pos[0])
    return p if p >= 0 else None


def _extreme_end_idx(df, bos_idx, direction, swings, right_lookback, wall):
    """Find the leg's structural top (spec §2, Rule A + Rule B).

    Rule A — first CONFIRMED opposing swing strictly AFTER bos_idx:
      bullish -> first swing HIGH; bearish -> first swing LOW. "Confirmed" means
      the swing is in the list AND its confirmation bar
      (resolved_idx + right_lookback) is <= wall (the last closed bar). A swing
      high sitting between the OB and the break is NOT the leg top (it is the
      level the break broke) — hence the search is strictly (bos_idx, wall].
      A one-bar breather is not a confirmed opposing swing, so it does NOT end
      the leg (we ask "did price reverse?", not "did one bar fail to extend?").

    Rule B — running extreme (fallback): no opposing swing has confirmed yet
      (the leg is still young / still running) -> extreme_end_idx = wall (the
      last closed bar). Honest: no structural top exists yet, so the running
      extreme is the best truth. Refreshed on re-arm.
    """
    want = "high" if direction == "bullish" else "low"
    best = None
    for s in (swings or []):
        if s.get("type") != want:
            continue
        s_idx = _resolve_idx(df, s.get("ts"))
        if s_idx is None:
            continue
        if s_idx <= bos_idx:          # must be strictly after the break
            continue
        if s_idx + right_lookback > wall:  # not yet confirmed at the wall -> can't be leg-end
            continue
        if s_idx > wall:              # pivot bar itself past the wall -> never peek
            continue
        # First confirmed opposing swing (swings are sorted by idx; the earliest
        # qualifying one is the leg top price turns from).
        if best is None or s_idx < best:
            best = s_idx
    if best is not None:
        return best                   # Rule A: window right edge = the pivot bar
    return wall                       # Rule B: still running -> last closed bar


def compute_leg_extreme_er(
    df,
    ob_ts,
    bos_ts,
    direction,
    swings,
    right_lookback=dealing_range.SWING_LOOKBACK,
):
    """(leg_extreme, leg_er, extreme_end_idx) for the displacement leg, or the
    None triple.

    Args:
      df: closed-bars-only OHLC frame to measure in. The caller passes a frame
        whose LAST row is the wall (last closed bar before the alert) — backtest
        point-in-time slice is already closed-only; live drops the forming bar.
      ob_ts: span START = the OB candle timestamp (ob['ob_timestamp']).
      bos_ts: the break candle timestamp (ob['bos_timestamp']); INCLUDED in span.
      direction: 'bullish' | 'bearish'.
      swings: list of {ts, type, ...}; each ts re-resolved to idx in df here.
      right_lookback: swing confirmation lag (default dealing_range.SWING_LOOKBACK).

    Span = [ob_idx, extreme_end_idx], inclusive of bos_idx. Opposing-swing search
    is strictly (bos_idx, wall]. Returns (None, None, None) on any degenerate or
    unmeasurable input, and NEVER raises (live-path safe).
    """
    try:
        if df is None or len(df) == 0 or direction not in ("bullish", "bearish"):
            return (None, None, None)

        ob_idx = _resolve_idx(df, ob_ts)
        bos_idx = _resolve_idx(df, bos_ts)
        # Degenerate / unmeasurable -> None (never a clipped guess, spec §2.2):
        #  * OB candle predates the slice (can't resolve) -> honest None.
        #  * break can't resolve, or OB is not before the break.
        if ob_idx is None or bos_idx is None or ob_idx > bos_idx:
            return (None, None, None)

        wall = len(df) - 1            # last closed bar in the passed frame
        if bos_idx > wall:            # break itself past the wall -> never peek
            return (None, None, None)

        end_idx = _extreme_end_idx(df, bos_idx, direction, swings,
                                   int(right_lookback), wall)
        if end_idx < ob_idx:          # defensive: span must be forward
            return (None, None, None)

        # --- leg extreme over [ob_idx, end_idx], inclusive of bos_idx ---------
        leg = df.iloc[ob_idx:end_idx + 1]
        if len(leg) == 0:
            return (None, None, None)
        if direction == "bullish":
            leg_extreme = float(leg["High"].max())
        else:
            leg_extreme = float(leg["Low"].min())

        # --- Kaufman ER over the SAME span's closes --------------------------
        # Mirror approach_quality.py:140-144 exactly (one concept, one impl).
        # None (never a faked 0.0) when < 3 bars, any close non-finite, or a
        # flat-close leg (denom 0). A real 0.0 is genuinely computed.
        leg_er = None
        C = leg["Close"].to_numpy(dtype=float)
        if len(C) >= _MIN_LEG_BARS and np.isfinite(C).all():
            steps = np.abs(np.diff(C))
            denom = float(steps.sum())
            if denom > 0:
                leg_er = round(abs(C[-1] - C[0]) / denom, 3)

        return (leg_extreme, leg_er, int(end_idx))
    except Exception:
        # Live-path guard: any failure is a silent None triple, never a raise.
        return (None, None, None)
