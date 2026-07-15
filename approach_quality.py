"""Approach quality — how price travelled INTO the zone (observe-only).

The "journey into the fill" read: over the closed H1 bars immediately before a
fill (backtest) / scan (live), how fast did price move toward the zone, how
clean were the candles, and how one-way was the path. Motivating evidence
(RETRACE_QUALITY_SPEC §0, 2026-07-15 probe): gold shows a V-spike tell (price
drifts away then one violent move in) and a fast-return tell; FX is null. All
must re-prove on the fresh canonical run before any rule ships.

OBSERVATION ONLY. Nothing here gates, scores, or filters a trade (same
discipline as pool_builder / eq_pools / weekly_pd). These are FILL-time (live:
scan-time) features — NOT alert-time. At alert the approach has not happened
yet, so they may NEVER enter the alert-time entry screen (look-ahead wall,
RETRACE_QUALITY_SPEC §1.3). Their only legal action class is order_rule
(cancel-pending-limit) if validated.

TIMING CLASS — read twice:
  - Backtest: window = the closed bars strictly BEFORE the fill bar.
  - Live: window ends at the LAST CLOSED bar (the "if price fills next bar, this
    is what the approach looked like" read). The caller strips any forming bar.
  Every input bar is CLOSED before the fill/decision moment, so this is legal to
  compute live — no look-ahead.

SINGLE IMPLEMENTATION (one concept, one implementation): both the backtest row
build and live Phase 2 call ONE core (_features_before). The two public wrappers
differ only in how end_pos is located (searchsorted the fill_ts vs len(df)).

NEVER RAISES: this code sits inside the live alert path. Any failure returns the
all-None dict (CLAUDE.md guard rule: no fail-loud in live alert generation).
"""

import math

import numpy as np
import pandas as pd

# The trades.csv column set this module owns. One list — the backtest row build,
# the reporting front_cols and the None-fallback all key off it. (The live twin
# renames these to *_at_email at the Phase 2 stamp site — RETRACE_QUALITY_SPEC
# §4; the math is identical.)
APPROACH_FEATURE_COLUMNS = (
    "approach_speed_atr_at_fill",   # directed net close move toward the zone / ATR
    "approach_body_ratio_at_fill",  # mean |C-O|/(H-L) over the approach bars
    "approach_er_at_fill",          # Kaufman efficiency ratio over the window
)

# Window sizes (frozen — RETRACE_QUALITY_SPEC §1.2). The probe's "3-bar"/"6-bar"
# labels meant 3/6 close-to-close STEPS; the bar windows are 4 and 7.
_WINDOW_BARS = 7   # full window (b1..b7); ER runs over its 6 steps
_K_BARS = 4        # last 4 bars (b4..b7); speed = 3 steps, body = mean of 4


def features_none():
    """All-None approach feature dict — the honest value when history is too
    thin (<7 closed bars before end_pos), the frame is broken, or the layer
    errored. Each value is also independently None-able (see §1.2)."""
    return {col: None for col in APPROACH_FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# Pure core: the three reads over the window ending just before end_pos
# ---------------------------------------------------------------------------

def _valid_atr(atr):
    """True only for a real positive finite ATR (Area B one-denominator law)."""
    if atr is None:
        return False
    try:
        a = float(atr)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and a > 0


def _features_before(df_h1, end_pos, direction, atr):
    """Core: approach features over the 7 closed bars strictly before end_pos.

    end_pos = positional index of the fill bar (backtest) / len(df) (live).
    Window = df.iloc[end_pos-7:end_pos] (b1..b7, b7 = last closed bar before the
    fill/scan moment). Bars may span weekends/holidays — positional by design;
    live sees the same gap. Fewer than 7 available -> all three None.

    Each value is independently None-able:
      - speed None when direction invalid or atr invalid (body/ER still real)
      - body  None when all 4 K-bars are zero-range
      - ER    None when the 6-step denominator is 0 (flat closes; never fake 0.0)
    Never raises — the wrappers catch; this core is defensive too.
    """
    out = features_none()
    if df_h1 is None or end_pos is None:
        return out
    try:
        start = int(end_pos) - _WINDOW_BARS
    except (TypeError, ValueError):
        return out
    if start < 0:
        return out  # <7 closed bars available -> all None

    win = df_h1.iloc[start:int(end_pos)]
    if len(win) < _WINDOW_BARS:
        return out

    # Pull the OHLC columns as floats. Column names match the live/backtest H1
    # frame convention (title-case OHLC, as used everywhere in the simulator).
    try:
        C = win["Close"].to_numpy(dtype=float)
        O = win["Open"].to_numpy(dtype=float)
        H = win["High"].to_numpy(dtype=float)
        L = win["Low"].to_numpy(dtype=float)
    except (KeyError, ValueError, TypeError):
        return out
    if len(C) < _WINDOW_BARS:
        return out
    if not np.isfinite(C).all():
        return out  # a broken bar in the window poisons every read

    # b1..b7 are win[0..6]; K = last 4 bars = win[3..6].
    kC, kO, kH, kL = C[-_K_BARS:], O[-_K_BARS:], H[-_K_BARS:], L[-_K_BARS:]

    # ── speed: directed net close move toward the zone over K (3 steps), in ATR
    if direction in ("bullish", "bearish") and _valid_atr(atr):
        a = float(atr)
        # bullish trade: zone below, healthy approach FALLS -> C[b4]-C[b7] > 0
        # bearish trade: zone above, healthy approach RISES -> C[b7]-C[b4] > 0
        if direction == "bullish":
            net = kC[0] - kC[-1]
        else:
            net = kC[-1] - kC[0]
        out["approach_speed_atr_at_fill"] = round(net / a, 3)

    # ── body ratio: mean |C-O|/(H-L) over the 4 K-bars, zero-range skipped
    rng = kH - kL
    body_ratios = []
    for i in range(_K_BARS):
        if rng[i] > 0:
            body_ratios.append(abs(kC[i] - kO[i]) / rng[i])
    if body_ratios:
        out["approach_body_ratio_at_fill"] = round(float(np.mean(body_ratios)), 3)

    # ── Kaufman ER over all 7 bars (6 steps): |C[b7]-C[b1]| / Σ|ΔC|
    steps = np.abs(np.diff(C))
    denom = float(steps.sum())
    if denom > 0:  # a real 0.0 is never faked (efficiency_ratio_at_alert rule)
        out["approach_er_at_fill"] = round(abs(C[-1] - C[0]) / denom, 3)

    return out


# ---------------------------------------------------------------------------
# Public wrappers — one core, two ways to locate end_pos. Both never raise.
# ---------------------------------------------------------------------------

def features_at_fill(df_h1, fill_ts, direction, atr):
    """Backtest entry: approach features over the closed bars BEFORE the fill.

    fill_ts is the fill bar's timestamp from the walk
    (h1_only_simulator.py:750 `fill_ts = ts`). Window = bars strictly before the
    fill bar. fill_ts None (never_filled) -> all-None dict. Never raises.
    """
    try:
        if fill_ts is None or df_h1 is None or len(df_h1) == 0:
            return features_none()
        ts = pd.Timestamp(fill_ts)
        if ts is pd.NaT or pd.isna(ts):  # NaT fill_ts -> no locatable fill bar
            return features_none()
        # searchsorted with the frame's own tz-awareness: normalise both sides.
        idx = df_h1.index
        if getattr(idx, "tz", None) is not None:
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts = ts.tz_convert(idx.tz)
        else:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
        # fill_ts IS an exact bar timestamp -> searchsorted gives the fill bar's
        # position; the window is the bars strictly before it.
        end_pos = idx.searchsorted(ts)
        return _features_before(df_h1, end_pos, direction, atr)
    except Exception:  # never let the approach layer kill a backtest row / scan
        return features_none()


def features_now(df_h1_closed, direction, atr):
    """Live entry: same math, window ends at the LAST CLOSED bar.

    df_h1_closed must contain closed bars only — the caller strips any forming
    bar the same way the P2 proximity read does. end_pos = len(df) so the window
    is the last 7 closed bars ("if price filled next bar, this is the approach").
    Never raises.
    """
    try:
        if df_h1_closed is None or len(df_h1_closed) == 0:
            return features_none()
        return _features_before(df_h1_closed, len(df_h1_closed), direction, atr)
    except Exception:
        return features_none()
