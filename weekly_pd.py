"""Weekly PD zone — the higher-timeframe premium/discount read.

"Is price expensive on the big picture?" measured against the most recent
COMPLETED weekly high/low. Pairs with the H4 PD zone
(dealing_range.compute_pd_position) to produce a plain-English agreement read:
when the big picture AND the local H4 both say the same thing, a continuation
trade is buying into (or selling into) a reversion zone — worth a caution.

OBSERVATION ONLY. Nothing here gates, scores, or filters a trade (same
discipline as pool_builder / eq_pools, owner-approved 2026-07-15). Live callers
surface one line in the P2 trade email; the backtest stamps the
WEEKLY_PD_FEATURE_COLUMNS onto trades.csv for the edge engine to judge.

DESIGN DECISIONS (owner, 2026-07-15):
  - Daily PD is deliberately SKIPPED — it is redundant with the H4 zone, and
    the daily high/low is already tracked as PD pools (pool_builder).
  - The weekly boundaries are FIXED at the last completed week's high/low.
    They do NOT move when price closes beyond them (no adaptive re-anchor).
    Reason (owner): if the boundary moved, we would lose sight of where last
    week's high/low actually was. Instead the % is allowed to run past its
    ends — a weekly_pd_position ABOVE 1.0 means price has closed above last
    week's high (broken up); BELOW 0.0 means below last week's low (broken
    down). The raw % itself carries the break signal, so the owner reads it
    directly and can override manually.
  - Premium vs discount splits at exactly 0.5 for BOTH timeframes (owner:
    "0.5 is a good anchor", no dead-band). >= 0.5 = premium (expensive),
    < 0.5 = discount (cheap). A broken-up weekly (> 1.0) is still clearly
    premium; a broken-down weekly (< 0.0) is still clearly discount, so the
    agreement always resolves.

SINGLE-IMPLEMENTATION REUSE (one concept, one implementation):
  - Weekly range: pool_builder.levels_at → the SAME PWH/PWL the PD/PW pools
    use (server_days → server_weeks resample of our own H1, proven vs MT5's
    own W1 files). We do NOT re-derive the weekly high/low — we read it from
    the pool layer's levels dict so live and backtest can never diverge.
  - Frame plumbing: pool_builder._naive_utc_index / drop_forming.
  - No extra feed pull: the backtest reuses the cached day/week resample; the
    live path reuses the same wide H1 frame the pool layer already fetched.
"""

from datetime import datetime, timezone

import pandas as pd

from pool_builder import (
    _naive_utc_index, drop_forming, levels_at, _cached_days_weeks,
)

# The trades.csv column set this module owns. One list, one implementation —
# the backtest row build, the reporting front_cols and the None-fallback all
# key off it.
WEEKLY_PD_FEATURE_COLUMNS = (
    "weekly_pd_position_at_alert",   # (price - PWL) / (PWH - PWL); may be <0 / >1
    "weekly_range_high_at_alert",    # last completed week's high (PWH)
    "weekly_range_low_at_alert",     # last completed week's low  (PWL)
    "weekly_pd_zone_at_alert",       # "premium" / "discount" (split at 0.5)
    "pd_zone_agreement_at_alert",    # both_premium / both_discount / mixed
)


def features_none():
    """All-None weekly-PD feature dict — the honest value when history is too
    thin (no completed prior week in frame) or the layer errored."""
    return {col: None for col in WEEKLY_PD_FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# Pure core: the weekly PD read and the H4-vs-weekly agreement
# ---------------------------------------------------------------------------

def _zone(pd_position):
    """premium/discount from a PD position, split at 0.5 (owner anchor).
    A broken level (>1 or <0) still resolves cleanly: >1 is premium (price
    closed above the high), <0 is discount. None passes through."""
    if pd_position is None:
        return None
    return "premium" if pd_position >= 0.5 else "discount"


def weekly_pd_read(price, weekly_high, weekly_low):
    """The weekly PD position + zone from a price and last week's high/low.

    Returns dict(weekly_pd_position, weekly_range_high, weekly_range_low,
    weekly_pd_zone). Boundaries are FIXED (never re-anchored): the position is
    allowed to run below 0.0 (closed below last week's low) or above 1.0
    (closed above last week's high) — that is the break signal by design.
    All None when a level is missing or the range is degenerate.
    """
    out = {"weekly_pd_position": None, "weekly_range_high": None,
           "weekly_range_low": None, "weekly_pd_zone": None}
    if price is None or weekly_high is None or weekly_low is None:
        return out
    hi, lo = float(weekly_high), float(weekly_low)
    width = hi - lo
    if width <= 0:  # degenerate / inverted range — cannot position
        return out
    # Round FIRST, then derive the zone from the rounded value, so the zone
    # the owner reads always matches the % shown (float noise like
    # 1.15-1.10 = 0.04999… must not print "50%" yet resolve as discount).
    pos = round((float(price) - lo) / width, 4)
    out["weekly_range_high"] = hi
    out["weekly_range_low"] = lo
    out["weekly_pd_position"] = pos
    out["weekly_pd_zone"] = _zone(pos)
    return out


def pd_zone_agreement(h4_pd_position, weekly_pd_position):
    """both_premium / both_discount / mixed from the two PD positions.

    Each side is split at 0.5 (>= 0.5 premium, < 0.5 discount). None when
    either position is missing — agreement is undefined without both reads.
    """
    if h4_pd_position is None or weekly_pd_position is None:
        return None
    h4_zone = _zone(h4_pd_position)
    wk_zone = _zone(weekly_pd_position)
    if h4_zone == wk_zone:
        return "both_premium" if h4_zone == "premium" else "both_discount"
    return "mixed"


def build_features(price, weekly_high, weekly_low, h4_pd_position):
    """WEEKLY_PD_FEATURE_COLUMNS dict from a price, last week's high/low and
    the already-computed H4 PD position. Pure — no frame, no I/O. Shared by
    the backtest row build and the live email path so both stamp identically.
    """
    out = features_none()
    read = weekly_pd_read(price, weekly_high, weekly_low)
    out["weekly_pd_position_at_alert"] = read["weekly_pd_position"]
    out["weekly_range_high_at_alert"] = read["weekly_range_high"]
    out["weekly_range_low_at_alert"] = read["weekly_range_low"]
    out["weekly_pd_zone_at_alert"] = read["weekly_pd_zone"]
    out["pd_zone_agreement_at_alert"] = pd_zone_agreement(
        h4_pd_position, read["weekly_pd_position"])
    return out


# ---------------------------------------------------------------------------
# Backtest entry point (mirror of pool_builder.features_at_alert)
# ---------------------------------------------------------------------------

def features_at_alert(df_h1, alert_ts, ref_price, h4_pd_position):
    """Backtest row-build entry point: the weekly-PD columns at one alert.

    The weekly high/low come from pool_builder.levels_at using ONLY completed
    weeks strictly before the alert's week — the SAME PWH/PWL the PD/PW pools
    stamp (no second weekly derivation). Completed weeks closed entirely in
    the past, so this cannot look ahead. Never raises — returns the all-None
    dict on any internal failure so a weekly-PD bug can never kill a run row.

    ref_price       — the placed entry (backtest).
    h4_pd_position  — the H4 PD position at this alert (0-1 or None), read off
                      the frozen dealing-range snapshot by the caller.
    """
    try:
        cached = _cached_days_weeks(df_h1)   # shared per-frame day/week cache
        h1 = cached["h1"]
        ts = pd.Timestamp(alert_ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        pos = h1.index.searchsorted(ts)  # bars strictly before ts
        if pos == 0:
            return features_none()
        lv = levels_at(h1, ts, days=cached["days"], weeks=cached["weeks"])
        return build_features(ref_price, lv.get("pwh"), lv.get("pwl"),
                              h4_pd_position)
    except Exception as e:  # never let the weekly-PD layer kill a backtest row
        print(f"  [WEEKLY_PD WARN] features_at_alert failed at {alert_ts}: "
              f"{type(e).__name__}: {e}")
        return features_none()


# ---------------------------------------------------------------------------
# Live entry point + plain-English P2 email line
# ---------------------------------------------------------------------------

def live_features(df_h1, price, h4_pd_position, weekly_levels=None,
                  now_utc=None):
    """Weekly-PD columns on the live engine frame (forming bar dropped here).

    weekly_levels — the pool layer's per-day cached levels dict (carries
    'pwh'/'pwl'); passed in so the live path reuses the SAME weekly high/low
    the pool banner shows (no second fetch, no divergence). When None, the
    weekly high/low are resampled from df_h1 as a fallback. Returns the
    all-None dict on thin/broken input — never raises.
    """
    try:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        if weekly_levels and weekly_levels.get("pwh") is not None:
            pwh, pwl = weekly_levels.get("pwh"), weekly_levels.get("pwl")
        else:
            closed = drop_forming(_naive_utc_index(df_h1), now_utc)
            if closed is None or len(closed) == 0:
                return features_none()
            lv = levels_at(closed, asof_ts=pd.Timestamp(now_utc))
            pwh, pwl = lv.get("pwh"), lv.get("pwl")
        return build_features(price, pwh, pwl, h4_pd_position)
    except Exception as e:
        print(f"  [WEEKLY_PD WARN] live_features failed: "
              f"{type(e).__name__}: {e}")
        return features_none()


# Plain-English percentage of a PD position (0.61 -> "61%"). The owner reads
# the % directly, so every sentence names it. A broken level is shown as its
# true >100% / negative value on purpose — that IS the break signal.
def _pct(pd_position):
    return None if pd_position is None else f"{round(pd_position * 100)}%"


def format_agreement_line(features, h4_pd_position, bias):
    """One predefined plain-English P2 line from the weekly-vs-H4 agreement.

    Names BOTH timeframes and BOTH percentages (owner requirement) so the
    trader always sees the numbers and can override manually. `bias` is
    "LONG"/"SHORT". Information only — never gates or scores. None when the
    agreement could not be computed (a timeframe read is missing).
    """
    if not features:
        return None
    agree = features.get("pd_zone_agreement_at_alert")
    if agree is None:
        return None
    wk_pct = _pct(features.get("weekly_pd_position_at_alert"))
    h4_pct = _pct(h4_pd_position)
    if wk_pct is None or h4_pct is None:
        return None
    nums = f"(weekly PD {wk_pct}, H4 PD {h4_pct})"

    if agree == "both_premium":
        # Continuation LONG into a premium zone = buying expensive.
        if bias == "LONG":
            return (f"Big picture and H4 both say price is EXPENSIVE {nums} — "
                    f"a continuation long here buys into a reversion zone. "
                    f"Be cautious.")
        return (f"Big picture and H4 both say price is EXPENSIVE {nums} — a "
                f"short here sells into that expensive zone, which is the "
                f"with-the-read side.")
    if agree == "both_discount":
        # Continuation SHORT into a discount zone = selling cheap.
        if bias == "SHORT":
            return (f"Big picture and H4 both say price is CHEAP {nums} — a "
                    f"continuation short here sells into a reversion zone. "
                    f"Be cautious.")
        return (f"Big picture and H4 both say price is CHEAP {nums} — a long "
                f"here buys into that cheap zone, which is the with-the-read "
                f"side.")
    # mixed
    return (f"Big picture and H4 disagree on expensive vs cheap {nums} — the "
            f"premium/discount edge is neutral here.")
