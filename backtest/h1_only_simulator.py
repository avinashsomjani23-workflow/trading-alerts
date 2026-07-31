"""H1-only trade simulator (proximal entry — the live model).

Tests the SMC system using ONLY H1 data — H1 finds the OB, entry happens at
the OB, SL/TP are sized off the H1 OB and H1 swing liquidity. No M15, no M5.

For every H1 OB-touch alert, this simulator fires ONE trade row: the proximal
entry (fills when price touches the OB proximal edge = the live limit). SL is
the OB distal +/- spread; TP price levels are the liquidity-based opposing H1
swings, reused from live compute_phase2_levels.

No scoring gate — every OB-touch is simulated regardless of confluence score.
Score is logged for post-run analysis (discover the optimal threshold empirically).

Exit policy (2026-07-31, FIXED_2R_BASELINE_SPEC): a single FIXED 2R bracket —
full position exits at +2R (`exit_reason == "tp"`) or the stop at -1R
(`exit_reason == "sl"`), no break-even, no trail, no liquidity-pool target. The
committed run speaks ONE exit language: fixed 2R. Liquidity-pool TP columns and
the BE@1R policy are retired from this run (the code that computes pool levels
stays for LIVE Phase 2; the backtest just stops logging it).

Hard rule (matches live simulator): same-bar SL+TP collision resolves SL-first.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import timedelta
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import smc_detector  # live, read-only

from backtest.run_logger import log_event


# H1 trade hold limit. 48 H1 bars = 2 trading days. Long enough for a swing
# setup to play out; short enough to release capital and avoid degenerate
# trades that never resolve.
MAX_HOLD_H1_BARS = 48
DEFAULT_RISK_USD = 250.0

# MSS (Market Structure Shift) displacement threshold, in RAW ATR body units.
# is_mss = True on a CHoCH row when break_body_atr (the raw ATR body of the break
# candle, smc_detector.compute_break_quality) >= this. DESCRIPTIVE ONLY — gates
# nothing, scores nothing.
#
# DERIVED 2026-07-21 from the canonical CSV (backtest/results/CANONICAL.md,
# h1only_20080102_20251231, 113 cols) as the MEDIAN raw break_body_atr of
# eligible live-pair CHoCH rows (EURUSD/USDJPY/NZDUSD/USDCHF/GOLD, n=3493) =
# 1.70 ATR — the empirical "typical vs above-typical displacement" split (~50/50).
# NOT the old 1.5 gate constant and NOT keyed off break_excess (which divides by
# a per-event reference — see MSS_AND_ATRFILL_HANDOFF.md A3).
#
# MEASURE-FIRST RESULT (do not treat as a proven edge): higher displacement did
# NOT predict better reversals. At every candidate cut the high-body group had
# WORSE expectancy than the low-body group (T=1.5 delta_exp=-0.163R, bootstrap
# 95% CI [-0.260,-0.068] excludes 0 on the WRONG side; only 39% of 59 quarters
# had high>low). So is_mss is logged descriptively; it is NOT yet a separator and
# must NOT be wired into the score. Re-derive on the next canonical baseline (this
# CSV predates the 2026-07-10 break-gate removal — detection columns are stale).
MSS_BODY_ATR_MULT = 1.70

# SL-sweep lookahead: bars after an SL exit to check whether a swept stop later
# recovered to the +2R or +1R target (only when the stop bar itself was a sweep).
# Matches the hold horizon so a late reversal is still caught. Diagnostic only.
SL_SWEEP_LOOKBACK_BARS = MAX_HOLD_H1_BARS

# TP1 RR floor. As of 2026-07-15 LIVE also floors at 0.5R (compute_phase2_levels
# default tp1_min_rr=0.5), so this is NOT a backtest-only relaxation anymore —
# live and backtest share the 0.5R floor and surface the same population. TP1
# selection takes the NEAREST unbroken opposing pool whose PLACED zone-edge RR
# lands in [0.5R, TP1_MAX_RR(4.0)]; if none qualifies the trade still fires on a
# mechanical 1:1 fallback (no pool). See compute_phase2_levels(tp1_min_rr=...) /
# _pick_tp1.
BACKTEST_TP1_MIN_RR = 0.5

# Weekend-flat (user rule, 2026-06-21): never hold a position into the FX
# weekend. Any OPEN trade is force-closed at the first Friday bar at/after
# WEEKEND_FLAT_HOUR_UTC, at that bar's open. Set WEEKEND_FLAT=False to disable.
# NOTE: this is RISK management, not a P&L improver -- a 4-quarter re-sim showed
# it costs ~3R/yr vs letting trades run (weekend-spanning trades were ~neutral).
# Cutoff = end of the user's Friday IST trading session: IST midnight (24:00 IST)
# = 18:30 UTC. On the hourly grid we flatten at the first Friday bar with
# hour >= 18 UTC (opens 18:00 UTC = 23:30 IST), i.e. before the weekend.
WEEKEND_FLAT = True
WEEKEND_FLAT_HOUR_UTC = 18

# ── Exit-lab side-channel (diagnostic only; OFF by default) ──────────────────
# When EXIT_LAB_SINK is a list AND EXIT_LAB_CONFIGS is a {name: config} dict, the
# simulator ALSO replays each alternative exit recipe over the SAME in-memory
# post-fill bars via exit_engine.walk_multileg, and appends per-config R to the
# sink. This is a PURE side-channel: r_realised, the trade row, and live parity
# are never touched. It is the only faithful way to study exits: every recipe sees
# the EXACT in-memory post-fill bars the trade was born from, so entry/SL/TP1/exits
# all share one consistent dataset (a replay over separately-reloaded bars would
# drift). Driven by backtest/diagnostics/exit_lab.py. Never set in a normal or
# live run.
EXIT_LAB_CONFIGS = None
EXIT_LAB_SINK = None

# Session windows in NY-LOCAL time (America/New_York), DST-resolved per candle
# date — the SAME tz the killzones use (config.json killzones all key off
# America/New_York, resolved via smc_detector.ts_in_killzone). Fixed-UTC buckets
# were WRONG half the year: London/NY session edges shift 1h across the EDT/EST
# change, so a boundary trade got the wrong session label. NY-local bucketing
# self-corrects because the zone conversion carries the DST offset.
#
# NY-local equivalents of the old UTC intent (Asia 0-7, London 7-13, NY 13-21
# UTC ~= EST): Asia 19:00->02:00, London 02:00->08:00, NY 08:00->16:00, else
# Other. These are the SAME session blocks, now DST-honest.
_NY_TZ = "America/New_York"


def _session_from_ny_hour(h: int) -> str:
    """Map NY-LOCAL hour -> trading session label. DST is already baked into `h`
    by the caller's tz conversion, so the boundaries are constant in NY-local
    time and correct in both EDT and EST."""
    if 2 <= h < 8:
        return "London"
    if 8 <= h < 16:
        return "NY"
    # Asia wraps past NY-midnight: 19:00 -> 02:00 (next day).
    if h >= 19 or h < 2:
        return "Asia"
    return "Other"


def _ts_hour_ny(ts_val) -> Optional[int]:
    """Coerce ts (str / pd.Timestamp / None) to America/New_York local hour,
    DST-resolved for that timestamp's date, or None if unparseable. Naive
    timestamps are treated as UTC (matches the rest of this module)."""
    if ts_val is None or ts_val == "":
        return None
    try:
        ts = pd.Timestamp(ts_val)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return int(ts.tz_convert(_NY_TZ).hour)
    except Exception:
        return None


def _ob_session(ob: Dict[str, Any]) -> str:
    """Session label for the OB candle itself (when the institutional move
    that created the zone happened). 'unknown' if ob_timestamp missing."""
    h = _ts_hour_ny(ob.get("ob_timestamp"))
    return _session_from_ny_hour(h) if h is not None else "unknown"


def _fill_session(fill_ts, alert_ts) -> str:
    """Session at fill (when capital was at work). Falls back to alert hour
    for never_filled rows so the column is never empty."""
    h = _ts_hour_ny(fill_ts) if fill_ts is not None else None
    if h is None:
        h = _ts_hour_ny(alert_ts)
    return _session_from_ny_hour(h) if h is not None else "unknown"


def _ts_in_killzone(ts_val, pair_conf: Dict[str, Any]) -> bool:
    """DST-aware killzone membership for a full timestamp. Routes through the
    shared smc_detector engine so the backtest resolves the SAME UTC windows
    the live engine does, per candle date. The full date matters: the same UTC
    hour can be in/out of a killzone depending on the EDT/EST season."""
    if ts_val is None or ts_val == "":
        return False
    killzones = pair_conf.get("killzones")
    if not killzones:
        return False
    try:
        ts = pd.Timestamp(ts_val)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return smc_detector.ts_in_killzone(ts.isoformat(), killzones)
    except Exception:
        return False


def _in_weekend_block(fill_ts, pair_conf: Dict[str, Any]) -> bool:
    """True iff `fill_ts` falls inside the pair's configured weekend no-trade
    window. Currently used for crypto (BTC): we do not trade Sat 00:00 -> Mon
    09:00 IST. Defined in config as `weekend_block` (tz Asia/Kolkata). Returns
    False when the pair has no weekend_block (all non-crypto pairs).

    Rule (BTC): block from Sat 00:00 IST through Mon 09:00 IST. In UTC that is
    Fri 18:30 -> Mon 03:30 (IST = UTC+5:30). Friday daytime trades are KEPT.
    We compute in IST directly (robust to any future window change) rather than
    hardcoding the UTC equivalents."""
    if fill_ts is None or fill_ts == "":
        return False
    wb = pair_conf.get("weekend_block")
    if not wb:
        return False
    try:
        ts = pd.Timestamp(fill_ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        ist = ts.tz_convert("Asia/Kolkata")
        dow = ist.dayofweek           # Mon=0 .. Sun=6
        mins = ist.hour * 60 + ist.minute
        # Saturday (5) and Sunday (6): always blocked.
        if dow in (5, 6):
            return True
        # Monday (0): blocked until 09:00 IST.
        if dow == 0 and mins < 9 * 60:
            return True
        # Friday (4): the window starts Sat 00:00 IST, so Friday is NOT blocked.
        return False
    except Exception:
        return False


def _ob_in_killzone(ob: Dict[str, Any], pair_conf: Dict[str, Any]) -> bool:
    return _ts_in_killzone(ob.get("ob_timestamp"), pair_conf)


def _fill_in_killzone(fill_ts, pair_conf: Dict[str, Any]) -> bool:
    return _ts_in_killzone(fill_ts, pair_conf)


def _killzone_alignment(ob: Dict[str, Any], fill_ts, alert_ts,
                        pair_conf: Dict[str, Any]) -> str:
    """4-way bucket for the SMC veteran hypothesis test:
       - 'Both'    : OB candle AND fill candle both fell in a killzone window
       - 'OB only' : OB in killzone, fill outside
       - 'Fill only': fill in killzone, OB outside
       - 'Neither' : both outside
       - 'never_filled': fill_ts is None (no fill happened)
    """
    if fill_ts is None:
        return "never_filled"
    ob_kz = _ob_in_killzone(ob, pair_conf)
    fl_kz = _fill_in_killzone(fill_ts, pair_conf)
    if ob_kz and fl_kz:
        return "Both"
    if ob_kz:
        return "OB only"
    if fl_kz:
        return "Fill only"
    return "Neither"


def _pd_zone_from_dr(price: float, dr: Optional[Dict[str, Any]]) -> str:
    """Where in the dealing range is `price`?
       discount = lower half, premium = upper half. Plain 0.5 split to match
       the scorecard (smc_detector.classify_setup) -- one PD threshold across
       the whole system, no dead-band. Returns 'unknown' if dealing range data
       is missing/invalid.
    """
    if not isinstance(dr, dict) or not dr.get("valid"):
        return "unknown"
    try:
        rng_low = float(dr["range_low"])
        rng_high = float(dr["range_high"])
    except (KeyError, TypeError, ValueError):
        return "unknown"
    width = rng_high - rng_low
    if width <= 0:
        return "unknown"
    pos = (price - rng_low) / width  # 0.0 at low, 1.0 at high
    return "discount" if pos <= 0.5 else "premium"


def _pd_alignment(bias: str, pd_zone: str) -> str:
    """Direction-aware PD-array read. Raw discount/premium is meaningless
    without the trade direction: SMC wants LONGS in discount and SHORTS in
    premium. The opposite (long in premium / short in discount) is a red flag,
    not a confluence -- the old pd_zone column could not tell them apart.

       aligned  = with the draw on liquidity (long+discount / short+premium)
       counter  = against it (long+premium / short+discount)
       unknown  = no valid dealing range

    No equilibrium/neutral bucket: the PD split is a plain 0.5 line
    (_pd_zone_from_dr), so every valid zone is either discount or premium.
    """
    if pd_zone in (None, "unknown"):
        return "unknown"
    if bias == "LONG":
        return "aligned" if pd_zone == "discount" else "counter"
    return "aligned" if pd_zone == "premium" else "counter"


def _confluences_present(breakdown: Dict[str, float]) -> str:
    """Comma-separated list of confluences that scored > 0 on this OB.
    Killzone removed 2026-05-25 (no longer a scoring input)."""
    names = []
    if breakdown.get("structure", 0) > 0:
        names.append("structure")
    if breakdown.get("sweep", 0) > 0:
        names.append("sweep")
    if breakdown.get("fvg", 0) > 0:
        names.append("fvg")
    if breakdown.get("freshness", 0) > 0:
        names.append("freshness")
    return ",".join(names) if names else "none"


def _event_label(bos_tag: Optional[str], bos_tier: Optional[str]) -> str:
    """One-column event label for the trade row.

    tier 'Confirm' = a Confirmation BOS (the first BOS in a CHoCH's direction
    that confirms the reversal — see dealing_range.py CONFIRMATION-BOS model).
    Labelled distinctly so it is tracked separately from a plain/Range BOS and
    from the CHoCH itself.
    """
    tag = bos_tag or "BOS"
    tier = bos_tier or "Major"
    if tag == "BOS" and tier == "Confirm":
        return "Confirmation BOS"
    return f"{tier} {tag}"


# FVG re-arm distance for fresh-vs-stale classification. Mirrors
# REARM_EXTRA_ATR in replay_engine.py (=1.0); that one is defined inside a
# function so it can't be imported cleanly. Same number, anchored to the FVG
# band here instead of the OB proximal. If the replay constant changes, change
# this too.
_FVG_REARM_ATR = 1.0


def _fvg_state(ob: Dict[str, Any], df_h1: pd.DataFrame,
               alert_ts: pd.Timestamp) -> str:
    """Classify the FVG at trigger time: 'fresh' | 'stale' | 'no_fvg'.

    no_fvg : no FVG ever formed in this zone -> excluded from the headline.
    fresh  : FVG still live at trigger (incl. partial), OR it was filled during
             THIS approach to the zone. First-approach pass-through is fresh:
             price must cross the FVG to reach the OB, so a same-visit fill is
             healthy, not stale.
    stale  : FVG was fully filled, price then LEFT the FVG band (cleared it by
             the re-arm distance) and RETURNED to trigger. The imbalance was
             already discharged on an earlier trip.

    Anchored to the FVG band (ghost_top/ghost_bottom), NOT the OB proximal, so a
    fill-then-reverse-before-the-OB-then-return is correctly stale. Uses
    mitigated_at_iso plumbed from smc_detector. Never raises -> defaults 'fresh'
    (the non-penalising bucket) on any missing data."""
    fvg = ob.get("fvg") or {}
    if not fvg.get("was_detected"):
        return "no_fvg"
    if fvg.get("exists"):
        return "fresh"                      # live at trigger (incl. partial)
    fill_iso = fvg.get("mitigated_at_iso")
    top, bot = fvg.get("ghost_top"), fvg.get("ghost_bottom")
    if not fill_iso or top is None or bot is None:
        return "fresh"                      # filled but no timing/band -> don't penalise
    try:
        fill_ts = pd.Timestamp(fill_iso)
        if fill_ts.tzinfo is None:
            fill_ts = fill_ts.tz_localize("UTC")
        rearm = _FVG_REARM_ATR * float(ob.get("h1_atr") or 0.0)
        # Bars from the FVG-fill up to (but excluding) the still-forming alert
        # bar. Excluding alert_ts keeps this consistent with the closed-only
        # slice the score + levels use (no forming-bar lookahead).
        win = df_h1.loc[fill_ts:alert_ts]
        win = win[win.index < alert_ts]
        if win.empty:
            return "fresh"
        # Did price pull clear of the FVG band by the re-arm distance after
        # filling it? Above the top or below the bottom counts as "left".
        left = ((win["Low"] > top + rearm) | (win["High"] < bot - rearm)).any()
        return "stale" if bool(left) else "fresh"
    except Exception:
        return "fresh"


def _ob_age_h1_bars(ob: Dict[str, Any], df_h1: pd.DataFrame,
                    alert_ts: pd.Timestamp) -> int:
    """How many H1 bars old is this OB at the alert moment?"""
    ob_ts_iso = ob.get("ob_timestamp")
    if not ob_ts_iso:
        return -1
    try:
        ob_ts = pd.Timestamp(ob_ts_iso)
        if ob_ts.tzinfo is None:
            ob_ts = ob_ts.tz_localize("UTC")
        # Use df_h1 index to count actual H1 bars between ob_ts and alert_ts,
        # not wall-clock hours (avoids weekend / data-gap inflation).
        in_window = df_h1.loc[ob_ts:alert_ts]
        # subtract 1 so age=0 means "alert on the OB-formation bar itself"
        return max(0, len(in_window) - 1)
    except Exception:
        return -1


def _closed_bars_at_alert(df_h1: pd.DataFrame,
                          alert_ts: pd.Timestamp) -> pd.DataFrame:
    """Live-parity input frame: the last LIVE_P2_H1_BARS bars CLOSED before
    alert_ts — exactly what live P2 hands run_scorecard/compute_phase2_levels
    (feed_adapter.fetch_h1 outputsize=200). tail() cannot add future bars, so
    the lookahead guarantee is unchanged (TRUTH_FIXES_SPEC_2 T5). Replaces the
    two separate unbounded closed-bar slices (scoring + levels) — one concept,
    one implementation.
    """
    s = df_h1.loc[df_h1.index < alert_ts]
    if s.empty:
        s = df_h1.loc[:alert_ts]  # degenerate guard, never empty in practice
    s = s.tail(smc_detector.LIVE_P2_H1_BARS)
    # FIX 1 pattern — cheap, loud runtime tripwire that the clamp holds.
    assert len(s) <= smc_detector.LIVE_P2_H1_BARS
    return s


def _score_h1_only(alert: Dict[str, Any], pair_conf: Dict[str, Any],
                   df_h1: pd.DataFrame, alert_ts: pd.Timestamp
                   ) -> Tuple[float, Dict[str, float]]:
    """Score the OB using live run_scorecard (H1-only since 2026-05-26).
    Returns (total, breakdown). Never raises — bad scores log and return 0.
    """
    ob = alert["ob"]
    bias = "LONG" if ob.get("direction") == "bullish" else "SHORT"
    # Lookahead + live-parity guard: score from ONLY the last LIVE_P2_H1_BARS
    # bars a live trader could see at the alert -- bars that had already CLOSED.
    # The alert fires at alert_ts (the bar opening then is still forming), so
    # closed bars are those indexed strictly before alert_ts; the 200-bar tail
    # matches live P2's fetch window (TRUTH_FIXES_SPEC_2 T5). Previously this fed
    # run_scorecard UNBOUNDED history (up to 15 yrs), so depth-sensitive score
    # inputs drifted with run start date instead of matching live.
    h1_slice = _closed_bars_at_alert(df_h1, alert_ts)
    # Live-parity fallback — ONE definition with live P2; never inline it.
    fvg_h1 = ob.get("fvg", smc_detector.fvg_missing())
    fvg_data = {"h1": fvg_h1}
    try:
        score_res = smc_detector.run_scorecard(
            bias, h1_slice, ob, fvg_data, alert["current_price"],
            pair_conf,
        )
    except Exception as e:
        log_event("h1only_scorecard_error", level="warn",
                  pair=alert.get("pair"),
                  error=f"{type(e).__name__}: {e}")
        return 0.0, {}
    breakdown = dict(score_res.get("breakdown", {}))
    # Killzone IS scored (2026-06-18) on the OB-FORMATION candle. The hard
    # filter gates the entry/alert time, NOT the OB candle, so this score is
    # independent of the filter and must flow through to the backtest total.
    total = round(sum(float(v) for v in breakdown.values()), 1)
    return total, breakdown


def _simulate_single_entry(
    alert: Dict[str, Any],
    pair_conf: Dict[str, Any],
    df_h1: pd.DataFrame,
    entry_zone: str,
    score: float,
    breakdown: Dict[str, float],
    risk_usd: float,
) -> Optional[Dict[str, Any]]:
    """Simulate one proximal trade for one OB-touch alert.

    Returns a row dict or None if the trade is invalid (entry would chase price,
    zero risk, or an OB thinner than the spread leaves no profit room — the same
    validity rules as live; a below-0.5R pool is NOT invalid, it routes to the 1:1
    fallback). Returns a "never_filled" row when the limit is not touched within
    the hold window, so we can count the miss.
    """
    ob = alert["ob"]
    pair = alert["pair"]
    bias = "LONG" if ob.get("direction") == "bullish" else "SHORT"
    alert_ts = alert["ts"]
    if not isinstance(alert_ts, pd.Timestamp):
        alert_ts = pd.Timestamp(alert_ts)
    if alert_ts.tzinfo is None:
        alert_ts = alert_ts.tz_localize("UTC")
    # Fill walk starts on the bar that OPENS at alert_ts (the bar still
    # forming when the alert fires). The just-closed bar that triggered the
    # alert is NOT a fill candidate — at the moment its wick was making the
    # move that triggered proximity, the limit order didn't exist yet (the
    # alert hadn't fired). The earliest a live broker could fill the limit
    # is during the bar that starts at alert_ts. Same-bar fills (within
    # this opening bar) ARE allowed and common — market momentum that
    # carries straight into the zone.
    current_price = alert["current_price"]

    # Lookahead guard (2026-06): TP/SL levels must be computed from ONLY the
    # bars a live trader could see at the alert -- bars that had already CLOSED.
    # The alert fires at alert_ts (the bar opening then is still forming), so
    # closed bars are those indexed strictly before alert_ts, clamped to live
    # P2's 200-bar fetch window (TRUTH_FIXES_SPEC_2 T5). Passing the full df_h1
    # let compute_phase2_levels.get_swing_points pick opposing swings that formed
    # AFTER the alert (future liquidity), biasing both TP selection and the 0.5R
    # TP1-floor grade optimistically; passing UNBOUNDED past history made TP
    # selection depend on run start date instead of matching live. The forward
    # fill-walk below intentionally keeps the FULL df_h1 -- it must see the
    # future to simulate how the trade plays out.
    df_h1_at_alert = _closed_bars_at_alert(df_h1, alert_ts)

    try:
        levels = smc_detector.compute_phase2_levels(
            pair_conf, bias, ob, current_price, df_h1_at_alert,
            entry_zone=entry_zone, tp1_min_rr=BACKTEST_TP1_MIN_RR,
            tp_targets="triple",
        )
    except Exception as e:
        log_event("h1only_levels_error", level="error", pair=pair,
                  entry_zone=entry_zone, alert_ts=str(alert_ts),
                  error=f"{type(e).__name__}: {e}")
        return None

    if not levels or not levels.get("valid", False):
        log_event("h1only_sim_skip", level="info", pair=pair,
                  entry_zone=entry_zone, alert_ts=str(alert_ts),
                  reason=levels.get("reason", "levels_invalid")
                         if isinstance(levels, dict) else "levels_none")
        return None
    # entry = the RAW OB execution price (2026-07-30 raw convention — no spread on
    # entry) — used for R-distance, MFE/MAE anchor, the FILL TRIGGER, and all exit
    # math. It IS the raw OB edge: bars are BID so the fill fires when the chart
    # reaches this line. (The entry_raw twin was dropped 2026-07-31 — under the raw
    # model it equalled entry, so it was a redundant column.)
    entry  = float(levels["entry"])
    sl     = float(levels["sl"])
    # FIXED_2R_BASELINE (2026-07-31): the liquidity-pool TP ladder
    # (tp1/tp2/tp_wick/tp_nextpool + all their *_raw/*_rr/zone_source twins) is
    # RETIRED from this run. compute_phase2_levels still computes it (LIVE Phase 2
    # consumes it) — the backtest simply stops reading it into the row. The one
    # committed exit is a fixed 2R bracket computed below from entry & sl.

    # SPREAD MODEL (2026-07-30, "raw" convention): the ONE spread in the system is
    # applied ONCE, upstream in compute_phase2_levels, which returns `sl` already
    # widened one spread past the OB distal (LONG below / SHORT above). Entry comes
    # back RAW. The simulator must NOT re-widen the stop — doing so double-counted
    # the spread and grew every 1R by ~8% (the pre-2026-07-30 bug). So `sl` is used
    # as-is. Slippage and swap are NOT modelled (user decision).

    r_distance = abs(entry - sl)
    if r_distance <= 0:
        log_event("h1only_sim_skip", level="warn", pair=pair,
                  entry_zone=entry_zone, alert_ts=str(alert_ts),
                  reason="zero_r_distance")
        return None

    # ── FIXED 2R target ───────────────────────────────────────────────────────
    # The single committed exit level: +2R above (LONG) / below (SHORT) entry,
    # where 1R = r_distance = |entry - sl| (sl already one-spread-widened upstream).
    tp_2r = (entry + 2 * r_distance) if bias == "LONG" else (entry - 2 * r_distance)

    # ── SETUP-LIQ Reads 1 & 2 (setup_liq / SWING_SWEEP_SPEC) ──────────────────
    # Read 1 (stop-side liquidity) + Read 2 (tp-side magnet) anchor on the FINAL
    # trade SL (post-spread) and the fixed 2R target. SETUP features (where liquidity
    # rests near the stop / the 2R target), NOT exit columns. Same closed-bar frame
    # the levels used (df_h1_at_alert -> look-ahead-safe); ATR = ob['h1_atr'] (the
    # shared *_atr denominator). Fixed 2R always has a target (no 1:1 fallback), so
    # the tp-side magnet is always evaluated. Observation only; never raises (all-None
    # on failure). Read 3.2 (leg-extreme) is a SEPARATE payload scalar from the replay
    # yield (leg_extreme_swept) — it anchors on leg geometry, not SL/TP.
    import setup_liq
    _setup_liq_reads = setup_liq.reads_stop_and_tp(
        df_h1_at_alert, ob.get("direction"), sl, tp_2r,
        ob.get("h1_atr"), pair_conf.get("pair_type", "forex"),
    )

    # Fill walk starts on the ALERT candle itself (alert_ts).
    #
    # Timeline (the trader's real clock, MT5 feed): the candle that triggered the
    # proximity alert is `alert_bar_ts` (the last CLOSED candle — "candle A"). It
    # closes, and the alert publishes at `alert_ts` = candle A's close = the OPEN
    # of the next candle ("candle B"). In real life the trader reads the email a
    # few minutes after candle A closes and places the limit a few minutes into
    # candle B — so candle B is the FIRST candle the order can fill on. We fill
    # from candle B (alert_ts), not candle C (alert_ts + 1h). The old +1h skipped
    # candle B entirely and filled a whole candle late (the "18%" of fills where
    # price reached entry on candle B were pushed to candle C or lost). The only
    # unmodelled sliver is candle B's first ~5 min before the order was placed —
    # negligible on H1. Same-bar fill+SL is still resolved SL-first by the
    # fill-bar rule below, so a candle-B fill never fabricates an unearned win.
    #
    # NOTE the earlier "cloned-fill" RCA (2026-03): a zone that re-alerted while its
    # trade was still open booked a second independent fill for one position. That is
    # guarded in run_backtest by the ONE-TRADE-PER-ZONE gate (`filled_obs`: once a
    # zone produces a filled trade, later alerts from that zone are dropped) — NOT by
    # the +1h fill offset, which was an over-correction on top. (This gate replaced
    # the 2026-07-15 "trade every re-touch" experiment; it is fill-based, not the old
    # first-alert `seen_obs` dedupe.) Filling on candle B is safe because a zone can
    # only be filled once, so no identical re-fire row can be created.
    #
    # Two separate clocks:
    #   - Pre-fill:  limit pends at most MAX_HOLD_H1_BARS candles from candle B.
    #                If price never reaches entry in that window -> never_filled.
    #   - Post-fill: once filled, trade runs at most MAX_HOLD_H1_BARS candles
    #                before forced timeout. Independent of the pre-fill wait.
    fill_walk_start = alert_ts
    future = df_h1.loc[fill_walk_start:]
    if future.empty:
        return None

    # OB mitigation / distal invalidation is decided UPSTREAM by the engine
    # (replay_engine._is_ob_mitigated_replay -> is_ob_mitigated_phase1), anchored
    # on the BOS/CHoCH event candle and using the per-instrument distal mode from
    # config -- the SAME rule live Phase 1/2 applies. A mitigated zone is dropped
    # before it can alert, so any alert reaching this simulator is, by live's own
    # rules, a valid un-killed zone. The simulator therefore does NOT run a second
    # distal kill: doing so (it previously anchored on the OB candle with a raw
    # wick) double-counted the impulse leg and diverged from live. One concept,
    # one implementation -- the engine owns mitigation.

    # ── Alert-candle distal-touch gate (2026-06-19; bar fixed 2026-07-02) ───
    # Live, the alert email is sent only AFTER the alert bar closes. A trader
    # reading that email re-checks the setup against the just-closed candle: if
    # that candle traded into the OB's DISTAL (far) line, the zone is spent /
    # violated and no trade is placed. The just-closed candle is candle A
    # (alert_bar_ts = alert_ts - 1h), whose high/low the replay engine stamps on
    # the alert as alert_bar_high / alert_bar_low.
    #
    # LOOKAHEAD FIX (2026-07-02): this gate previously read df_h1.loc[alert_ts]
    # — the bar that OPENS at alert_ts, i.e. candle B, which is STILL FORMING
    # when the alert publishes. Using candle B's final high/low dropped setups
    # based on where price went AFTER the decision moment. Since a candle-B
    # distal touch usually means fill-then-stop, the gate was deleting mostly
    # losers with future knowledge and inflating the headline. Now it checks
    # candle A only — the candle a live trader can actually see.
    #
    # Rule (just-closed candle ONLY): drop the setup if its wick TOUCHES the
    # distal line. Touch, not close -- a wick into the far edge is enough.
    #   SHORT (bearish OB, distal above): drop if alert_bar_high >= distal
    #   LONG  (bullish OB, distal below): drop if alert_bar_low  <= distal
    # Later bars (candle B onward) are handled by the normal fill/SL walk.
    distal_line = ob.get("distal_line")
    ab_hi = alert.get("alert_bar_high")
    ab_lo = alert.get("alert_bar_low")
    if distal_line is not None and ab_hi is not None and ab_lo is not None:
        distal_line = float(distal_line)
        distal_touched = (
            float(ab_hi) >= distal_line if bias == "SHORT"
            else float(ab_lo) <= distal_line
        )
        if distal_touched:
            log_event("h1only_sim_skip", level="info", pair=pair,
                      entry_zone=entry_zone, alert_ts=str(alert_ts),
                      reason="alert_candle_touched_distal",
                      distal=distal_line, bias=bias)
            return None

    filled = False
    fill_ts: Optional[pd.Timestamp] = None
    fill_bar_idx = -1
    # The proximal entry is a pre-placed pending limit sitting at the OB proximal
    # edge for the OB's lifetime. Fill when price first crosses the entry level
    # (long fills on bar low <= entry; short on bar high >= entry). This handles
    # three cases uniformly:
    #   - alert bar exactly touched the level   -> fills on alert bar
    #   - alert bar approaching but not yet at  -> fills on subsequent bar
    #   - alert bar overshot past the level     -> fills when price pulls back

    exit_ts: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None
    mfe_price = entry
    mae_price = entry
    # OUTCOME-side descriptors (2026-07-26): H1 bars from fill (bar 0) to the bar
    # that SET the running MFE/MAE extreme. Tracked O(1) inside the SAME walk that
    # finds mfe_r/mae_r — no second pass. First bar to reach an extreme wins ties
    # (strict >/< below keeps the earliest). Anchor bar is the fill bar, so both
    # default to 0 (mfe/mae anchor at `entry` on the fill bar). NEVER a model/entry
    # feature — pure look-ahead. See bars_to_mfe/bars_to_mae in TRUTH_LEDGER.md.
    mfe_bar_idx = 0
    mae_bar_idx = 0
    sl_collision = False
    bars_walked_post_fill = 0
    # Bars from fill (bar 0) to the EXIT bar. Captured when the exit LATCHES — the
    # walk keeps running afterwards for window-MFE (A3), so bars_walked_post_fill
    # would otherwise inflate bars_to_exit to the window end (2026-07-31 fix).
    exit_bar_offset: Optional[int] = None

    # ── FIXED 2R policy state (2026-07-31, FIXED_2R_BASELINE_SPEC) ────────────
    # The committed exit is a single fixed 2R bracket: full position exits at the
    # +2R target (tp_2r) or the stop (sl) — no break-even, no trail. `cur_sl` is
    # just `sl` (never mutated). `r_realised` and `pnl_usd` follow THIS policy and
    # nothing else.
    #
    # WINDOW-MFE DECOUPLE (A3): the walk does NOT break at the exit. Once the exit
    # is latched, it keeps walking bars for MFE/MAE ONLY until the window ends
    # (48-bar hold cap / friday / data end). So mfe_r tracks the FULL post-fill
    # window excursion, independent of where the 2R exit fired — mfe_r can legally
    # EXCEED r_realised (exit +2R, price later ran +3.5R -> mfe_r = 3.5).
    cur_sl = sl

    for i, (ts, bar) in enumerate(future.iterrows()):
        bar_hi = float(bar["High"])
        bar_lo = float(bar["Low"])

        is_fill_bar_this_iter = False
        if not filled:
            # Weekend-flat fill guard (2026-07-02 fix): a limit may NOT fill on a
            # Friday bar >= WEEKEND_FLAT_HOUR_UTC. Filling there would open a
            # position that immediately rides the weekend gap -- the weekend-flat
            # check above only force-closes an ALREADY-open position on a later
            # bar, so it never catches a position opened on the Friday-evening
            # bar itself (is_fill_bar_this_iter was True on the fill bar, which
            # the flat check explicitly skips). The order stays pending into
            # Monday rather than being killed -- same as any other no-touch bar.
            friday_evening = (WEEKEND_FLAT and ts.dayofweek == 4
                              and ts.hour >= WEEKEND_FLAT_HOUR_UTC)
            # Pending limit fill: the FILL TRIGGER is the OB line `entry` (bars are
            # BID; the chart must reach the limit). Long triggers when bar.low <=
            # entry, short when bar.high >= entry. Under the raw model there is no
            # spread on entry, so this same line is also the fill price and the
            # mfe/mae anchor set below.
            if not friday_evening and (
                    (bias == "LONG" and bar_lo <= entry) or
                    (bias == "SHORT" and bar_hi >= entry)):
                filled = True
                fill_ts = ts
                fill_bar_idx = i
                mfe_price = entry
                mae_price = entry
                is_fill_bar_this_iter = True
            else:
                # Pre-fill cap: limit pends at most MAX_HOLD_H1_BARS bars.
                # i is 0-indexed bars-since-alert, so >= cap - 1 means we've
                # already waited the full window without a touch -> give up.
                if i >= MAX_HOLD_H1_BARS - 1:
                    break
                continue

        bars_walked_post_fill = i - fill_bar_idx
        # Hold cap ends BOTH the trade and (post-exit) the MFE-only walk. If the
        # 2R exit already latched, this just stops the window; only stamp a
        # `timeout` exit when nothing resolved (still open).
        if bars_walked_post_fill > MAX_HOLD_H1_BARS:
            if exit_reason is None:
                exit_ts = ts
                exit_reason = "timeout"
                exit_price = float(bar["Close"])
                exit_bar_offset = bars_walked_post_fill
            break

        # Weekend-flat: force-close an OPEN position before the FX weekend. Also
        # bounds the MFE-only walk after a 2R exit — a swept-past-Friday excursion
        # is not something the trade could hold, so the window ends here too.
        if (WEEKEND_FLAT and not is_fill_bar_this_iter
                and ts.dayofweek == 4 and ts.hour >= WEEKEND_FLAT_HOUR_UTC):
            if exit_reason is None:
                exit_ts = ts
                exit_reason = "friday_flat"
                exit_price = float(bar["Open"])
                exit_bar_offset = bars_walked_post_fill
            break

        if bias == "LONG":
            sl_hit_in_bar = bar_lo <= cur_sl
            tp_hit_in_bar = bar_hi >= tp_2r
            # WINDOW-MFE/MAE (A3): track the FULL post-fill window excursion off
            # the RAW bar extremes — NO cap at the 2R target (that cap belonged to
            # the retired TP1 exit). mfe_r can legitimately exceed r_realised.
            # Two bars still cannot contribute their raw extremes (intrabar order
            # unknowable — same pessimism as before):
            #   - SL bar: the wick that touched SL also printed the bar high, so
            #     crediting that high fakes a positive excursion on the stop bar.
            #   - FILL bar: a LONG limit fills on the bar LOW; that bar's HIGH
            #     happened BEFORE the fill (price fell through entry), so it is
            #     pre-fill price, not favourable excursion.
            if not sl_hit_in_bar and not is_fill_bar_this_iter:
                # Compare-then-record (not max/min) so bars_to_mfe/mae capture the
                # bar index of the extreme with strict >/< (first bar wins ties).
                if bar_hi > mfe_price:
                    mfe_price = bar_hi
                    mfe_bar_idx = i - fill_bar_idx
                if bar_lo < mae_price:
                    mae_price = bar_lo
                    mae_bar_idx = i - fill_bar_idx
        else:
            sl_hit_in_bar = bar_hi >= cur_sl
            tp_hit_in_bar = bar_lo <= tp_2r
            if not sl_hit_in_bar and not is_fill_bar_this_iter:
                # SHORT: favourable = lower price (MFE), adverse = higher (MAE).
                if bar_lo < mfe_price:
                    mfe_price = bar_lo
                    mfe_bar_idx = i - fill_bar_idx
                if bar_hi > mae_price:
                    mae_price = bar_hi
                    mae_bar_idx = i - fill_bar_idx

        # Once the 2R exit is latched, there is no more SL/TP resolution to do —
        # post-exit bars ONLY update the window MFE/MAE above and honor the window
        # cap. Skip the resolution block.
        if exit_reason is not None:
            continue

        # Fill-bar rule (2026-05-25):
        # On the bar where the limit just filled, we cannot infer intra-bar
        # sequence of fill -> TP vs fill -> SL. SL-side: if the bar pierced SL,
        # price had to travel through entry first (limit fills, then SL), so SL is
        # the honest outcome. TP-side: bar high reaching the 2R target could mean
        # (a) price ticked up to it before pulling down to fill, OR (b) filled then
        # rallied. Can't tell. Conservative: do NOT credit TP on the fill bar.
        if is_fill_bar_this_iter:
            tp_hit_in_bar = False

        # ── Realised exit: fixed 2R bracket ─────────────────────────────────
        # Priority within a bar:
        #   1. SL+TP same bar -> SL wins (unprovable order; keep pessimism).
        #   2. SL hit -> loss (-1R), terminal.
        #   3. 2R target hit -> win (+2R), terminal. Exit reason "tp" (one target,
        #      so NOT "tp1"/"tp2r" — avoids TP1/TP2 confusion).
        # After latching, the loop keeps running for MFE/MAE only (the `continue`
        # above); it does NOT break here — that is the window-MFE decouple.
        if sl_hit_in_bar and tp_hit_in_bar:
            sl_collision = True
            exit_ts = ts
            exit_reason = "sl"
            exit_price = cur_sl
            exit_bar_offset = bars_walked_post_fill
            continue
        if sl_hit_in_bar:
            exit_ts = ts
            exit_reason = "sl"
            exit_price = cur_sl
            exit_bar_offset = bars_walked_post_fill
            continue
        if tp_hit_in_bar:
            exit_ts = ts
            exit_reason = "tp"
            exit_price = tp_2r
            exit_bar_offset = bars_walked_post_fill
            continue

    if not filled:
        # Limit never touched within the hold window. Emit a "never_filled" row
        # so the report can count "would-have-missed" trades.
        return _build_row(
            alert=alert, pair_conf=pair_conf, ob=ob,
            entry_zone=entry_zone, entry=entry,
            sl=sl,
            setup_liq_reads=_setup_liq_reads,
            score=score, breakdown=breakdown,
            df_h1=df_h1, alert_ts=alert_ts,
            fill_ts=None, exit_ts=None, exit_reason="never_filled",
            exit_price=None,
            r_realised=0.0,
            mfe_r=0.0, mae_r=0.0, bars_to_mfe=None, bars_to_mae=None,
            bars_to_exit=0,
            sl_collision=False, risk_usd=risk_usd,
            sl_bar_was_sweep=None,
            sl_swept_then_2r=None, sl_swept_then_1r=None,
            ob_to_fill_hours=None,
            bars_break_to_pullback=None,
        )

    if exit_reason is None:
        # Window exhausted with position open and no SL/2R hit.
        last = future.iloc[-1]
        exit_ts = future.index[-1]
        exit_reason = "window_end"
        exit_price = float(last["Close"])
        exit_bar_offset = bars_walked_post_fill

    # ── r_realised: the FIXED 2R policy. This is the one true outcome — pnl_usd
    # and every report headline derive from it. A clean win books +2R, a clean
    # loss -1R; timeout/friday_flat/window_end book the partial close-price R.
    # mfe_r/mae_r are the FULL post-fill WINDOW excursion (A3 decouple), so
    # mfe_r can legitimately exceed r_realised (invariant: mfe_r >= r_realised).
    if bias == "LONG":
        r_realised = (exit_price - entry) / r_distance
        mfe_r = (mfe_price - entry) / r_distance
        mae_r = -(entry - mae_price) / r_distance
    else:
        r_realised = (entry - exit_price) / r_distance
        mfe_r = (entry - mfe_price) / r_distance
        mae_r = -(mae_price - entry) / r_distance

    # bars_to_mfe / bars_to_mae: bar index (fill = 0) of the running extreme,
    # captured O(1) in the walk above. If a filled trade had NO post-fill bar
    # (filled on the last window bar -> window_end with nothing walked), the
    # extremes never moved off the fill anchor and the index is meaningless ->
    # emit NULL, never 0 (spec 2026-07-26). bars_walked_post_fill is the last
    # i - fill_bar_idx reached; 0 means no forward bar contributed.
    if bars_walked_post_fill > 0:
        bars_to_mfe = mfe_bar_idx
        bars_to_mae = mae_bar_idx
    else:
        bars_to_mfe = None
        bars_to_mae = None

    # bars_to_exit is the bars-to-EXIT captured when the exit latched (A3: the walk
    # continues past the exit for window-MFE, so bars_walked_post_fill overshoots to
    # the window end). Fall back to the walked count only if never captured.
    bars_to_exit = max(0, exit_bar_offset if exit_bar_offset is not None
                       else bars_walked_post_fill)

    # ── Sweep diagnostics: was the STOP a liquidity grab, and did it reverse? ──
    # SMC definition of a sweep: the candle WICKS through the level but CLOSES BACK
    # on our side (grab-then-reject). A candle that CLOSES THROUGH the stop is a
    # genuine break, not a sweep — and a wider stop would just lose more.
    #
    #   sl_bar_was_sweep   : the STOP CANDLE itself was a sweep of the stop that
    #                        fired (cur_sl — the fixed SL; BE is retired, so cur_sl
    #                        is always the initial SL).
    #                        Long : Low <= cur_sl AND Close > cur_sl.
    #                        Short: High >= cur_sl AND Close < cur_sl.
    #   sl_swept_then_2r   : STRICT. sl_bar_was_sweep is True AND, walking the
    #                        post-stop bars in order within SL_SWEEP_LOOKBACK_BARS,
    #                        price reached the +2R target BEFORE it ever traded back
    #                        to the fired stop cur_sl. Swept once, HELD, then ran the
    #                        full 2R. A bar re-hitting cur_sl first (or spanning both
    #                        cur_sl and +2R) = False.
    #   sl_swept_then_1r   : same STRICT test, but the target is +1R (breakeven-plus
    #                        = entry ± r_distance). Answers the narrower "came back
    #                        to just +1R" question. Both None when the stop bar was
    #                        not a sweep.
    # Rationale (FIXED_2R_BASELINE_SPEC A5): keeping BOTH readings shows which
    # stopped-out trades came back only to +1R vs ran all the way to the 2R target
    # — the wider-stop question.
    #
    # HONEST CAVEAT (peak-vs-fill law): these are PATH-aware (they require the stop
    # to HOLD until the target), but the target leg is still a TOUCH check on the
    # winning bar, not a real-order replay — that touch could be its own spike-and-
    # fade. Read them as strong HINTS ("would a wider stop have saved us"), never as
    # bankable "free money". All None for non-SL / non-sweep exits.
    sl_bar_was_sweep = None
    sl_swept_then_2r = None
    sl_swept_then_1r = None
    # sl_wick_depth_atr (2026-07-08): how far the STOP CANDLE's wick pierced BEYOND
    # the fired stop (cur_sl), normalised by the OB-formation ATR (ob['h1_atr'], the
    # same denominator every *_atr feature uses). The missing input for sizing a
    # wider stop: a sweep tells us the wick crossed the stop, this tells us HOW FAR,
    # so a "distal + X·ATR" replay grid can be chosen from data instead of guessed.
    #   LONG  (stop below): depth = (cur_sl - sl_bar_low)  / h1_atr   [>=0]
    #   SHORT (stop above): depth = (sl_bar_high - cur_sl) / h1_atr   [>=0]
    # 0.0 = the wick closed exactly at the stop (no overshoot). None for non-SL exits
    # or when h1_atr is unavailable (legacy zone). NOT a gate — logging only.
    sl_wick_depth_atr = None
    # ── Exit-track outcome-time columns (EXIT TRACK ONLY — leakage as entry
    # features). All describe what happened AFTER the stop fired, so a wider-stop
    # replay can be designed and sanity-checked from data. None for non-SL exits.
    # TOUCH-based HINTS, never bankable — only a real-order replay counts. The
    # sweep-conditioned ones (max_adverse, recovered) are only set when the stop bar
    # was a sweep (a clean close-through has no "would a wider stop have survived"
    # question to answer).
    #   sl_max_adverse_after_sweep_atr : furthest price ran AGAINST us BEYOND the
    #     fired stop, over SL_SWEEP_LOOKBACK_BARS after the stop bar, in OB-formation
    #     ATR. RAW context measure. 0.0 = never past the stop. None = non-sweep/no ATR.
    #   bars_sl_to_2r_touch : STRICT. Defined ONLY for the clean swept-then-HELD-then
    #     -2R case: 1-indexed H1 bars from the stop bar to the bar that reached +2R,
    #     when +2R was reached BEFORE price traded back to the fired stop cur_sl.
    #     None whenever sl_swept_then_2r is not True.
    #   bars_sl_to_1r_touch : same, for the +1R target (sl_swept_then_1r True).
    #   sl_recovered_to_entry : after a sweep, did price trade back to ENTRY
    #     (breakeven) within the lookback, even if no target was reached? Catches
    #     the "a wider stop would have SCRATCHED, not won" middle case (BE-sweep).
    sl_max_adverse_after_sweep_atr = None
    bars_sl_to_2r_touch = None
    bars_sl_to_1r_touch = None
    sl_recovered_to_entry = None
    # The +1R target (breakeven-plus) for the SL-anatomy 1R reading. +2R is tp_2r.
    _target_1r = (entry + r_distance) if bias == "LONG" else (entry - r_distance)
    if exit_reason == "sl" and exit_ts is not None:
        try:
            sl_bar = future.loc[exit_ts]
            sl_bar_hi = float(sl_bar["High"])
            sl_bar_lo = float(sl_bar["Low"])
            sl_bar_cl = float(sl_bar["Close"])
            if bias == "LONG":
                sl_bar_was_sweep = bool(sl_bar_lo <= cur_sl and sl_bar_cl > cur_sl)
            else:
                sl_bar_was_sweep = bool(sl_bar_hi >= cur_sl and sl_bar_cl < cur_sl)

            _h1_atr_sl = ob.get("h1_atr")
            if _h1_atr_sl:
                if bias == "LONG":
                    _overshoot = cur_sl - sl_bar_lo
                else:
                    _overshoot = sl_bar_hi - cur_sl
                sl_wick_depth_atr = round(max(0.0, _overshoot) / _h1_atr_sl, 3)

            post_sl = future.loc[future.index > exit_ts]
            horizon = post_sl.iloc[:SL_SWEEP_LOOKBACK_BARS]

            # sl_swept_then_2r / _1r (STRICT) + bars_sl_to_2r/1r_touch are BOTH
            # defined ONLY for a swept stop bar. A non-sweep (clean close-through)
            # has no "held then ran to target" story to tell, so all stay None.
            # Walk the post-stop bars IN ORDER once; per bar, per target (+2R, +1R),
            # test in this priority:
            #   FAIL first: price traded back to the fired stop cur_sl (LONG Low <=
            #     cur_sl / SHORT High >= cur_sl) -> stop hit again -> False, bars
            #     None. Applies to BOTH readings (a re-hit kills both).
            #   WIN  first: price reached the target in our direction (LONG High >=
            #     target / SHORT Low <= target) -> True, bars = 1-indexed from the
            #     stop bar. Recorded independently for +2R and +1R (a bar can reach
            #     +1R without reaching +2R). Whichever fires on the EARLIER bar wins.
            # Edge case: a single bar that both breaches cur_sl AND touches a target
            # (wick spans both) is a FAIL — the stop level was hit, intrabar order
            # unknowable, conservative read. Neither in the lookback -> False / None.
            if sl_bar_was_sweep:
                sl_swept_then_2r = False   # swept but never cleanly ran the 2R
                sl_swept_then_1r = False   # swept but never cleanly ran the 1R
                _done_2r = False
                _done_1r = False
                for _i in range(len(horizon)):
                    _bar = horizon.iloc[_i]
                    _bhi = float(_bar["High"])
                    _blo = float(_bar["Low"])
                    if bias == "LONG":
                        _hit_stop = _blo <= cur_sl
                        _hit_2r = _bhi >= tp_2r
                        _hit_1r = _bhi >= _target_1r
                    else:
                        _hit_stop = _bhi >= cur_sl
                        _hit_2r = _blo <= tp_2r
                        _hit_1r = _blo <= _target_1r
                    if _hit_stop:                       # fail wins ties, kills both
                        break
                    if _hit_2r and not _done_2r:
                        sl_swept_then_2r = True
                        bars_sl_to_2r_touch = _i + 1    # 1-indexed from stop bar
                        _done_2r = True
                    if _hit_1r and not _done_1r:
                        sl_swept_then_1r = True
                        bars_sl_to_1r_touch = _i + 1
                        _done_1r = True
                    if _done_2r and _done_1r:
                        break

                # Max adverse excursion BEYOND the fired stop, after the stop
                # bar (how much further it ran against us). RAW context measure.
                # LONG stop is below, so "against" = further DOWN (lower Low);
                # SHORT = further UP.
                if len(horizon) and _h1_atr_sl:
                    if bias == "LONG":
                        _worst = float(horizon["Low"].min())
                        _adverse = cur_sl - _worst
                    else:
                        _worst = float(horizon["High"].max())
                        _adverse = _worst - cur_sl
                    sl_max_adverse_after_sweep_atr = round(
                        max(0.0, _adverse) / _h1_atr_sl, 3)

                # Did price return to ENTRY (breakeven) within the lookback?
                if len(horizon):
                    if bias == "LONG":
                        sl_recovered_to_entry = bool((horizon["High"] >= entry).any())
                    else:
                        sl_recovered_to_entry = bool((horizon["Low"] <= entry).any())
                else:
                    sl_recovered_to_entry = False
        except Exception:
            sl_bar_was_sweep = None
            sl_swept_then_2r = None
            sl_swept_then_1r = None
            sl_wick_depth_atr = None
            sl_max_adverse_after_sweep_atr = None
            bars_sl_to_2r_touch = None
            bars_sl_to_1r_touch = None
            sl_recovered_to_entry = None

    # ── ob_to_fill_hours: OB formation -> fill gap (diagnostic; NOT a gate) ──
    # corr with r ~0 both years, not monotonic — logged only for the edge engine
    # to slice. See EDGE_ENGINE_HANDOFF 9b.
    ob_to_fill_hours = None
    try:
        _ob_ts = pd.to_datetime(ob.get("ob_timestamp"), utc=True)
        if _ob_ts is not None and fill_ts is not None:
            ob_to_fill_hours = round(
                (pd.to_datetime(fill_ts, utc=True) - _ob_ts).total_seconds() / 3600.0, 2)
    except Exception:
        ob_to_fill_hours = None

    # ── bars_break_to_pullback: H1 bars from the break candle to the first bar
    # that traded back to the OB proximal (the pullback that fills us). Flags the
    # "strong break + very fast snapback" bucket (BS1) — thin + news-confounded,
    # validate over 18yr before gating. See EDGE_ENGINE_HANDOFF 9b.
    bars_break_to_pullback = None
    try:
        _bos_ts = pd.to_datetime(ob.get("bos_timestamp"), utc=True)
        if _bos_ts is not None and fill_ts is not None:
            _post_break = df_h1.loc[df_h1.index > _bos_ts]
            _fill_dt = pd.to_datetime(fill_ts, utc=True)
            _hit = _post_break.index <= _fill_dt
            bars_break_to_pullback = int(_hit.sum())
    except Exception:
        bars_break_to_pullback = None

    # ── Exit-lab side-channel (diagnostic; no effect on r_realised / the row) ──
    # FIXED_2R_BASELINE (2026-07-31): the liquidity-TP comparison recipes ("tp1",
    # "tp_wick", "tp_nextpool" specs) are RETIRED — this run studies only fixed-R
    # exits (float R-multiple legs), which need no structural target. Any recipe
    # still carrying a structural string spec is skipped rather than raising.
    if EXIT_LAB_CONFIGS and EXIT_LAB_SINK is not None and fill_bar_idx >= 0:
        from backtest.exit_engine import walk_multileg
        _post = future.iloc[fill_bar_idx:]
        for _name, _cfg in EXIT_LAB_CONFIGS.items():
            _specs = {s for _, s in _cfg["legs"] if isinstance(s, str)}
            # Structural / liquidity-TP specs are no longer computed in this run.
            if _specs & {"tp1", "tp_wick", "tp_nextpool"}:
                continue
            try:
                _res = walk_multileg(
                    _post, bias, entry, sl, r_distance, None, _cfg,
                    weekend_flat=WEEKEND_FLAT,
                    weekend_hour_utc=WEEKEND_FLAT_HOUR_UTC,
                    max_hold=MAX_HOLD_H1_BARS,
                )
                EXIT_LAB_SINK.append({
                    "pair": pair, "alert_ts": str(alert_ts),
                    # ob_timestamp + direction make the row uniquely joinable back
                    # to its trade: two different OBs can alert on the same pair at
                    # the same timestamp, so (pair, alert_ts) alone is NOT unique.
                    "ob_timestamp": ob.get("ob_timestamp"),
                    "direction": ob.get("direction"),
                    "entry_zone": entry_zone, "committed_r": round(r_realised, 4),
                    # Realised exit reason of the LIVE walk — lets the exit report
                    # score the exit study over the SAME population the headline
                    # counts (drop never_filled/timeout/window_end unresolved rows).
                    "exit_reason": exit_reason,
                    "config": _name, "r": _res["r_realised"],
                })
            except Exception as _e:  # never let a diagnostic break a run
                log_event("exit_lab_error", level="warn", pair=pair,
                          config=_name, error=f"{type(_e).__name__}: {_e}")

    return _build_row(
        alert=alert, pair_conf=pair_conf, ob=ob,
        entry_zone=entry_zone, entry=entry,
        sl=sl,
        setup_liq_reads=_setup_liq_reads,
        score=score, breakdown=breakdown,
        df_h1=df_h1, alert_ts=alert_ts,
        fill_ts=fill_ts, exit_ts=exit_ts, exit_reason=exit_reason,
        exit_price=exit_price,
        r_realised=round(r_realised, 3),
        mfe_r=round(mfe_r, 3), mae_r=round(mae_r, 3),
        bars_to_mfe=bars_to_mfe, bars_to_mae=bars_to_mae,
        bars_to_exit=bars_to_exit,
        sl_collision=sl_collision, risk_usd=risk_usd,
        sl_bar_was_sweep=sl_bar_was_sweep,
        sl_swept_then_2r=sl_swept_then_2r,
        sl_swept_then_1r=sl_swept_then_1r,
        sl_wick_depth_atr=sl_wick_depth_atr,
        sl_max_adverse_after_sweep_atr=sl_max_adverse_after_sweep_atr,
        bars_sl_to_2r_touch=bars_sl_to_2r_touch,
        bars_sl_to_1r_touch=bars_sl_to_1r_touch,
        sl_recovered_to_entry=sl_recovered_to_entry,
        ob_to_fill_hours=ob_to_fill_hours,
        bars_break_to_pullback=bars_break_to_pullback,
    )


# FIX 3e — OB-field classification for the trade row (closes the "mutable state
# logged from a frozen snapshot" bug class). Any NEW ob field logged in a trade
# row MUST fall into one of these buckets:
#   IMMUTABLE EVENT FACTS (freezing correct): bos_timestamp, ob_timestamp,
#     direction, bos_tag, bos_tier, bos_swing_price, impulse_start_price, high,
#     low, proximal_line, distal_line, median_leg_body, ob_body, h1_atr
#     (formation ATR, frozen by design), reversal_pct, broken_was_wall,
#     bos_sequence_count, last_choch_idx, event_candle_delta.
#   FROZEN-BY-DESIGN, LIVE DOES THE SAME: dealing_range (incl. S4
#     dr_ceiling_broken_at_ob / dr_floor_broken_at_ob, read off the frozen
#     snapshot), sweep_v2 (v1 retired 2026-07-24).
#   STAMPED AT ALERT (correct source): bos_verdict, touches_at_alert +
#     fvg_at_alert, h1_trend / trend_alignment / alert_bar_*, and the S2/S3
#     structure signals (flip_pending_at_alert,
#     flip_pending_dir_at_alert,
#     leg_extreme_at_alert, leg_extreme_clipped — all payload scalars from the
#     replay yield).
#   MUTABLE STATE, fixed by this spec: touches/status (3a), break_quality (3b),
#     fvg (3c/3d).
# RULE: mutable state is stamped `*_at_alert` at the replay yield and read from
# that snapshot here — NEVER read live off the ob dict at row-build time (the
# replay keeps mutating it after the alert).
#
# The two functions below are the SINGLE implementation of the freeze contract
# (CLAUDE.md "one concept, one implementation"). The live path calls them and
# tests/test_ob_alert_freeze.py imports and drives the SAME functions — so the
# freeze test can never pass on a stale copy while the live read rots.


def build_alert_ob_view(alert: Dict[str, Any]) -> Dict[str, Any]:
    """T1+T4: the alert-time view of the OB.

    The replay mutates the shared OB dict after this alert fired (re-fires
    re-stamp bos_verdict AND the touches_at_alert/fvg_at_alert dict keys; the
    per-bar loop updates touches/status/fvg), and rows are built after the whole
    walk. The alert PAYLOAD is the one source — bos_verdict (T1), touches_at_alert
    / fvg_at_alert (T4) travel as payload scalars snapshotted at the yield. The
    dict stamps remain only as a legacy fallback for old alerts.

    TRAP: dict(_ob_live) copies the (possibly re-stamped) *_at_alert KEYS into the
    view, and _build_row PREFERS those keys — so BOTH key spellings (touches /
    touches_at_alert, fvg / fvg_at_alert) must be overwritten. One view, built
    once — never patch individual fields inline downstream.
    """
    _ob_live = alert["ob"]
    ob_view = dict(_ob_live)
    if alert.get("bos_verdict") is not None:
        ob_view["bos_verdict"] = alert["bos_verdict"]
    _touches = alert.get("touches_at_alert", _ob_live.get("touches_at_alert"))
    if _touches is not None:
        ob_view["touches"] = _touches
        ob_view["touches_at_alert"] = _touches
    _fvg = alert.get("fvg_at_alert") or _ob_live.get("fvg_at_alert")
    if _fvg is not None:
        ob_view["fvg"] = _fvg
        ob_view["fvg_at_alert"] = _fvg
    return ob_view


def read_frozen_ob_fields(ob: Dict[str, Any]) -> Dict[str, Any]:
    """FIX 3d: read the alert-time-frozen mutable OB fields for the trade row.

    `touches`/`fvg` on the live ob keep changing after the alert as the per-bar
    loop walks on; the frozen values live under touches_at_alert / fvg_at_alert
    (stamped at the replay yield). This is the ONE place those reads live.

    Returns the scalars the row dict emits plus `ob_at_alert` — a shallow view
    carrying the alert-time fvg so the fvg_state helper (which reads ob["fvg"]
    internally) classifies at the alert moment. Legacy OBs with no *_at_alert
    snapshot fall back to the live read (no crash).
    """
    _touches_at_alert = ob.get("touches_at_alert", ob.get("touches"))
    _fvg_at_alert = ob.get("fvg_at_alert")
    if _fvg_at_alert is not None:
        _ob_at_alert = dict(ob)
        _ob_at_alert["fvg"] = _fvg_at_alert
    else:
        _ob_at_alert = ob  # legacy OB with no alert-time snapshot -> live read
    _fvg_view = _fvg_at_alert or ob.get("fvg") or {}
    return {
        "ob_touches": _touches_at_alert,
        "fvg_present": bool(_fvg_view.get("exists")),
        "fvg_mitigation": _fvg_view.get("mitigation"),
        "ob_at_alert": _ob_at_alert,
    }


def read_s4_broken_flags(dr):
    """S4 (STRUCTURE_SIGNALS_SPEC): the dealing-range ceiling/floor broken flags,
    read off the FROZEN ob["dealing_range"] snapshot. Returns
    (dr_ceiling_broken_at_ob, dr_floor_broken_at_ob); both None when the snapshot
    is invalid / legacy (the flag was never resolvable). Single implementation —
    the row build and tests/test_structure_signals.py both call this."""
    if isinstance(dr, dict) and dr.get("valid"):
        _cb = dr.get("ceiling_broken")
        _fb = dr.get("floor_broken")
        return (
            bool(_cb) if _cb is not None else None,
            bool(_fb) if _fb is not None else None,
        )
    return (None, None)


def _build_row(*, alert, pair_conf, ob, entry_zone, entry, sl,
               score, breakdown, df_h1, alert_ts,
               fill_ts, exit_ts, exit_reason, exit_price,
               r_realised,
               mfe_r, mae_r, bars_to_exit,
               bars_to_mfe=None, bars_to_mae=None,
               sl_collision, risk_usd,
               sl_bar_was_sweep=None,
               sl_swept_then_2r=None, sl_swept_then_1r=None,
               sl_wick_depth_atr=None, sl_max_adverse_after_sweep_atr=None,
               bars_sl_to_2r_touch=None, bars_sl_to_1r_touch=None,
               sl_recovered_to_entry=None,
               ob_to_fill_hours=None,
               bars_break_to_pullback=None,
               setup_liq_reads=None) -> Dict[str, Any]:
    """Assemble the final trade row dict in stable column order."""
    # Field freeze/live/re-read classification (which ob[...] reads are frozen at
    # birth vs live vs re-read): see the "OB FIELD FREEZE / LIVE / RE-READ
    # CLASSIFICATION" section in TRUTH_LEDGER.md.
    direction = ob.get("direction", "?")
    # FIX 3d: mutable OB state (touches, fvg) is frozen at the replay yield into
    # touches_at_alert / fvg_at_alert. read_frozen_ob_fields is the ONE reader —
    # it returns the row scalars plus `ob_at_alert` (a shallow view carrying the
    # alert-time fvg so the fvg_state helper classifies at the alert moment).
    _frozen = read_frozen_ob_fields(ob)
    _touches_at_alert = _frozen["ob_touches"]
    _ob_at_alert = _frozen["ob_at_alert"]
    bos_tag = ob.get("bos_tag", "BOS")
    bos_tier = ob.get("bos_tier", "Major")
    # Break quality of the BOS/CHoCH candle — computed ONCE by smc_radar at
    # detection (smc_detector.compute_break_quality) and carried on the OB. Never
    # recomputed here; we only surface the frozen numbers so the backtest can
    # benchmark what break ATR multiple actually wins, per event type.
    #   break_close_atr = raw ATR multiple the close cleared the broken level by
    #   break_excess    = break body / body reference (BOS 1.0/CHoCH 1.5 ATR); NOT a gate (removed 2026-07-10)
    _bq = ob.get("break_quality") or {}
    # MSS (Market Structure Shift) label — a CHoCH on a STRONG displacement candle
    # (confirmed reversal) vs a CHoCH on a weak one (soft warning). The ONLY
    # difference is displacement, so this keys off the RAW ATR break body
    # (_bq['body_atr'] == the break_body_atr column), NEVER break_excess (which
    # divides by a per-event reference — see MSS_AND_ATRFILL_HANDOFF.md A3). True
    # only on CHoCH rows whose body >= MSS_BODY_ATR_MULT; None on non-CHoCH (a BOS
    # has no reversal to displace). Descriptive — gates/scores nothing.
    _break_body_atr = _bq.get("body_atr")
    if bos_tag == "CHoCH" and _break_body_atr is not None:
        is_mss = bool(_break_body_atr >= MSS_BODY_ATR_MULT)
    else:
        is_mss = None
    dr = ob.get("dealing_range")
    pd_zone = _pd_zone_from_dr(entry, dr)
    pd_alignment = _pd_alignment("LONG" if direction == "bullish" else "SHORT",
                                 pd_zone)
    # % position within the dealing range: 0% = range low, 100% = range high.
    # Gives an exact read on where the entry sits in the PD array.
    if isinstance(dr, dict) and dr.get("valid"):
        try:
            _rng_low = float(dr["range_low"])
            _rng_high = float(dr["range_high"])
            _width = _rng_high - _rng_low
            pd_pct = round((entry - _rng_low) / _width * 100, 1) if _width > 0 else None
        except (KeyError, TypeError, ValueError):
            pd_pct = None
    else:
        pd_pct = None
    # ── S4 (STRUCTURE_SIGNALS_SPEC): broken-wall PD flags at OB formation ───────
    # Was the dealing-range ceiling / floor riding the LIVE extreme (broken, not a
    # confirmed swing) when this OB formed? Read straight off the FROZEN
    # ob["dealing_range"] snapshot (immutable after OB build, same bucket as the
    # existing dealing_range fields — no *_at_alert needed). get_dealing_range now
    # carries these additive keys on its valid branch; None when the snapshot is
    # invalid / legacy (the flag was never resolvable). read_s4_broken_flags is
    # the ONE reader (same fn tests/test_structure_signals.py drives).
    dr_ceiling_broken_at_ob, dr_floor_broken_at_ob = read_s4_broken_flags(dr)
    # reversal_pct: the CHoCH-origin-in-extreme flag, computed ONCE in
    # dealing_range.compute_structure (_reversed_from_premium / _discount) and
    # carried on the OB by smc_radar. 1.0 = the swing the CHoCH reversed FROM sat
    # in the frozen confirmed range's extreme (top 25% for a down CHoCH, bottom
    # 25% for an up CHoCH); 0.0 = it did not; None = not stamped. Surfaced here
    # raw, NEVER recomputed — this is the exact origin-based field, not the
    # entry-position proxy (pd_pct). Only a CHoCH carries it meaningfully (a BOS
    # is always 0.0/None — there is no reversal origin to test).
    reversal_pct = ob.get("reversal_pct")
    # reversed_from_extreme: plain-English derived flag for the CHoCH/reversal
    # book. True only when this is a CHoCH AND its origin sat in the extreme.
    # None when it is not a CHoCH, or when reversal_pct was never stamped.
    # CAVEAT (documented, not hidden): reversal_pct is 0.0 both when the origin
    # was genuinely mid-range AND when the confirmed-range gate was invalid
    # (no fully-confirmed H4 range yet, so the extreme could not be tested).
    # Those two cases are not distinguishable from this field alone; treat 0.0
    # as "not confirmed from the extreme", not as "proven mid-range".
    _is_choch = "CHoCH" in str(bos_tag)
    if not _is_choch or reversal_pct is None:
        reversed_from_extreme = None
    else:
        reversed_from_extreme = bool(float(reversal_pct) >= 1.0)

    # ── Setup-geometry features (observe-only; feed the edge-discovery engine) ──
    # All ATR-normalized against the OB-formation ATR (ob['h1_atr'], frozen at
    # detection) so a single bucket boundary works across instruments. Every value
    # here is read from fields the detector already froze on the OB — nothing is
    # recomputed, so all are point-in-time clean (no look-ahead). None when the
    # input is missing (legacy zone) or the ATR is unavailable (avoids div-by-zero).
    _h1_atr = ob.get("h1_atr")

    def _atr_norm(v):
        return round(v / _h1_atr, 3) if (_h1_atr and v is not None) else None

    # OB candle range in ATR. NOTE: with SL at the distal and entry at the
    # proximal, this ~= the stop distance — the NAS failure axis (stop < one
    # candle's range). Same quantity as "zone thickness"; logged once.
    ob_range_atr = _atr_norm(abs(float(ob.get("high", 0.0)) - float(ob.get("low", 0.0)))) \
        if ob.get("high") is not None and ob.get("low") is not None else None

    # Walk-back geometry (A3, DECISION_GUARDRAILS.md — logging only, no gate
    # yet). Frozen at OB formation (smc_radar.py), read as-is, never recomputed.
    ob_body_ratio = ob.get("body_ratio")
    ob_walkback_depth = ob.get("walkback_depth")
    # Choppiness Index — daily trend-vs-range regime on the alert's server day
    # (compute_choppiness_index). alert_ts is candle B's OPEN (still forming), so
    # anchor on the last CLOSED bar strictly before it — no look-ahead.
    _closed_pre = df_h1.loc[df_h1.index < alert_ts]
    chop_at_alert = smc_detector.compute_choppiness_index(df_h1, _closed_pre.index[-1] if len(_closed_pre) else None)

    # FVG size in ATR — the displacement gap's magnitude (present/absent throws
    # this gradient away). FVG-mitigation-agnostic: measures the gap as detected.
    _fvg = ob.get("fvg") or {}
    if _fvg.get("exists") and _fvg.get("fvg_top") is not None and _fvg.get("fvg_bottom") is not None:
        fvg_size_atr = _atr_norm(abs(float(_fvg["fvg_top"]) - float(_fvg["fvg_bottom"])))
    else:
        fvg_size_atr = None

    # (impulse_leg_to_extreme_atr — the leg-size-in-ATR column — is computed
    # below in the S3 block, where its source leg_extreme_at_alert is read. It
    # REPLACED the frozen impulse_leg_atr, which measured to the broken level and
    # anchored on the untrustworthy impulse-start walk-back candle.)

    # Raw OB-formation ATR (price units) — volatility context. NOT cross-instrument
    # comparable raw; the engine normalizes it within-pair (vs the pair's typical
    # ATR) for a regime read. Logged because it's free and frozen.
    atr_at_ob = round(float(_h1_atr), 6) if _h1_atr else None

    # Raw ATR at the FILL bar (price units) — the volatility we actually TRADE
    # INTO, computed FRESH (NOT ob['h1_atr'], which is formation-time). An OB can
    # form quiet and fill into a spike; atr_at_ob vs atr_at_fill is that regime
    # comparison (no third column needed). POINT-IN-TIME: the slice ENDS at the
    # fill bar (df_h1.loc[:fill_ts]) — a single look-ahead bar would poison the
    # read. None when never_filled (fill_ts is None) or the slice is too short for
    # ATR (compute_atr returns None for < period+1 bars). Rounded to 6dp to match
    # atr_at_ob. Observe-only — gates/scores nothing.
    if fill_ts is not None and df_h1 is not None:
        _fill_slice = df_h1.loc[:fill_ts]
        _atr_fill = smc_detector.compute_atr(_fill_slice)
        atr_at_fill = round(float(_atr_fill), 6) if _atr_fill else None
    else:
        atr_at_fill = None

    # Era-stable volatility-regime read at the FILL bar (2026-07-25). fill-bar
    # ATR(14) vs its OWN trailing-90-day distribution: percentile rank (0-100,
    # robust) + ratio-to-mean (magnitude). CAUSAL (bars strictly BEFORE fill,
    # mirrors exit_lab._atr_at_fill). Both None on never_filled / <30 calendar
    # days of prior history. Precomputed series -> O(1) lookup. See the
    # _atr_regime_at_fill section for the full contract. Observe-only.
    _atr_regime = _atr_regime_at_fill(df_h1, fill_ts)

    # ── Derived columns (2026-07-08): encoded in CODE (were previously pasted into
    # the CSV from a sheet and got column-shift corrupted). All three are computed
    # from real, frozen source columns so every run reproduces them deterministically.
    #
    #   sl_distance_atr : |entry - sl_initial| / OB-formation ATR. Risk width in
    #     ATR. Uses sl_initial = the traded stop (OB distal -/+ 1 spread), the one
    #     true stop distance. This is ~1 for most trades (the "one H1 bar" instant-
    #     death axis). Point-in-time clean: entry + sl + ATR are all known at fill.
    sl_distance_atr = round(abs(entry - sl) / _h1_atr, 3) \
        if (_h1_atr and entry is not None and sl is not None) else None

    #   sl_dist_atr_at_alert / tp_dist_atr_at_alert : how big is this trade's stop /
    #     target vs NORMAL recent movement, at the ALERT moment. DELIBERATELY unlike
    #     sl_distance_atr above (which anchors ENTRY and divides by OB-formation
    #     _h1_atr, stale by alert). Here:
    #       - anchor = OB PROXIMAL line (the live system's reference; no fill exists
    #         at alert), matching live Phase2_Alert_Engine.
    #       - target = the fixed +2R level (entry ± 2·r_distance) — the ONE committed
    #         target now (liquidity-pool TP1 retired, FIXED_2R_BASELINE_SPEC).
    #       - ruler = a FRESH ATR(14) on the last 14 CLOSED H1 candles as of the
    #         alert (_closed_bars_at_alert already drops the forming bar), NOT the
    #         formation ATR. Backtest/live identical (same anchor, same closed-bar
    #         fresh ATR, same period).
    #     Point-in-time clean: proximal + SL + entry + alert-time bars all known at
    #     alert. Observe-only. None when the ATR slice is too short or a level missing.
    _prox_alert = ob.get("proximal_line")
    _bias_row = "LONG" if direction == "bullish" else "SHORT"
    _r_dist_row = abs(entry - sl) if (entry is not None and sl is not None) else None
    _tp_2r_row = (
        ((entry + 2 * _r_dist_row) if _bias_row == "LONG" else (entry - 2 * _r_dist_row))
        if _r_dist_row is not None else None
    )
    _alert_slice = _closed_bars_at_alert(df_h1, alert_ts) if df_h1 is not None else None
    _atr_alert = smc_detector.compute_atr(_alert_slice, period=14) if _alert_slice is not None else None
    if _atr_alert and _atr_alert > 0 and _prox_alert is not None:
        sl_dist_atr_at_alert = round(abs(_prox_alert - sl) / _atr_alert, 3) \
            if sl is not None else None
        tp_dist_atr_at_alert = round(abs(_prox_alert - _tp_2r_row) / _atr_alert, 3) \
            if _tp_2r_row is not None else None
    else:
        sl_dist_atr_at_alert = None
        tp_dist_atr_at_alert = None

    #   r_capture_ratio : r_realised / mfe_r. How much of the best favorable move
    #     we actually kept — now against the FULL-WINDOW MFE (A3 decouple), so this
    #     is "2R capture of the full-window excursion". 1.0 = kept all the way to the
    #     window's best; <1 = the window ran further after we exited; can be negative
    #     on a loser that had a favorable poke first. None when mfe_r <= 0 (no
    #     favorable move to capture — ratio undefined, never 0/0). OUTCOME-time (uses
    #     r_realised) → exit/description only, NEVER an entry feature.
    r_capture_ratio = round(r_realised / mfe_r, 3) \
        if (mfe_r is not None and mfe_r > 0 and r_realised is not None) else None

    #   trend_pd_agree : do the two directional confluences agree — is the trade
    #     WITH the H1 trend AND PD-aligned? True only when both point the same way.
    #     h1_trend is absolute (bullish/bearish); pd_alignment is already relative
    #     to direction (aligned/counter). Point-in-time clean (both frozen at alert).
    #     None when either input is missing (legacy/degraded row).
    _h1_trend_val = alert.get("h1_trend")
    if _h1_trend_val is None or pd_alignment is None:
        trend_pd_agree = None
    else:
        _with_trend = (
            (direction == "bullish" and _h1_trend_val == "bullish")
            or (direction == "bearish" and _h1_trend_val == "bearish")
        )
        trend_pd_agree = bool(_with_trend and pd_alignment == "aligned")

    # ── S3 (DISPLACEMENT_LEG_BUILD_SPEC): displacement-leg extreme + ER ────────
    # `leg_extreme_at_alert` (structural leg extreme, span [ob_idx, extreme_end_idx])
    # and `leg_er_at_alert` (Kaufman ER over the same span) are payload scalars
    # frozen at the replay yield by the shared displacement_leg core.
    # `leg_extreme_clipped` (payload) flags an OB older than the point-in-time slice
    # (extreme is None there, honest). (leg_retrace_pct_at_alert was removed 2026-07-19:
    # retracement quality is uninformative for an order-block-limit system — our
    # entry sits at one fixed depth by construction, and the shallow-retrace cases
    # never reach the limit, so there is no depth variation to measure.)
    leg_extreme_at_alert = alert.get("leg_extreme_at_alert")
    leg_er_at_alert = alert.get("leg_er_at_alert")
    leg_extreme_clipped = alert.get("leg_extreme_clipped")

    # impulse_leg_to_extreme_atr — the displacement leg's size in ATR, measured
    # from the OB to the leg's ACTUAL structural extreme. REPLACES the frozen
    # impulse_leg_atr, which (a) measured only to the broken level (bos_swing),
    # not the leg's real extreme, and (b) anchored on the untrustworthy
    # impulse_start walk-back candle.
    #   anchor = ob['proximal_line'] (OB high for bullish / OB low for bearish,
    #     smc_radar.py:1287). CONSISTENT with leg_extreme_at_alert: that extreme
    #     is the max High / min Low over the span [ob_idx, extreme_end_idx] which
    #     STARTS at the OB candle, so the OB's own proximal edge is inside the
    #     span and the extreme is guaranteed on the same side -> the distance is
    #     always >= 0 (the far end pushed at least to the OB's near edge).
    #   far end = leg_extreme_at_alert (the ONE structural extreme; never
    #     recomputed a second way — one source).
    # ALERT-TIME + MUTABLE: leg_extreme_at_alert is a payload scalar re-stamped on
    # every re-fire (replay_engine.py:696), so this rides the LATEST alert's
    # extreme — never read off the shared ob dict. None when leg_extreme_at_alert
    # is None (unmeasurable leg), proximal missing, or ATR missing (never 0.0).
    # Observe-only — gates/scores/filters NOTHING.
    _leg_prox = ob.get("proximal_line")
    impulse_leg_to_extreme_atr = _atr_norm(abs(float(leg_extreme_at_alert) - float(_leg_prox))) \
        if (leg_extreme_at_alert is not None and _leg_prox is not None) else None

    pnl_usd = round(r_realised * risk_usd, 2)

    # Setup badge (Phase 2 email banner) — same classifier live fires
    # (Phase2_Alert_Engine.py:2542), same inputs, so the backtest can finally
    # check whether "A+ Reversal at the Wall" / "A First Pullback" /
    # "Caution: Late-Trend Chase" actually correlate with r_realised.
    # classify_setup wants pd_position on a 0-1 scale; pd_pct here is 0-100.
    _pd_position_01 = (pd_pct / 100.0) if pd_pct is not None else None
    setup_badge, _setup_note, setup_kind = smc_detector.classify_setup(
        ob, _pd_position_01, alert.get("trend_alignment")
    )

    return {
        "pair":          alert["pair"],
        "alert_ts":      alert_ts.isoformat() if hasattr(alert_ts, "isoformat") else str(alert_ts),
        "alert_bar_ts":  (alert.get("alert_bar_ts").isoformat()
                          if hasattr(alert.get("alert_bar_ts"), "isoformat")
                          else str(alert.get("alert_bar_ts")) if alert.get("alert_bar_ts") is not None
                          else None),
        "alert_seq":     int(alert.get("alert_seq", 1)),
        "bos_timestamp": ob.get("bos_timestamp"),
        "fill_ts":       fill_ts.isoformat() if (fill_ts is not None and hasattr(fill_ts, "isoformat")) else None,
        "exit_ts":       exit_ts.isoformat() if (exit_ts is not None and hasattr(exit_ts, "isoformat")) else None,
        "direction":     direction,
        "bias":          "LONG" if direction == "bullish" else "SHORT",
        "model":         "h1_only",
        "event":         _event_label(bos_tag, bos_tier),
        "entry_zone":    entry_zone,
        # entry is the RAW OB execution price (2026-07-30 raw convention — no spread
        # shift) and the fill-trigger line. None-safe. (entry_raw twin dropped
        # 2026-07-31 — it equalled entry under the raw model.)
        "entry":         entry,
        # sl_initial = the traded stop = OB distal -/+ 1 spread (the single spread,
        # applied once upstream). The one stop column (sl_raw twin dropped
        # 2026-07-31 — it equalled this under the raw model).
        "sl_initial":    sl,
        # tp_2r = the FIXED 2R target (entry ± 2·r_distance), the ONE committed exit
        # level (FIXED_2R_BASELINE_SPEC 2026-07-31). Liquidity-pool TP1/TP2 and the
        # 3-target ladder are retired from this run. Known at fill; audit-friendly.
        "tp_2r":         _tp_2r_row,
        "exit_price":    exit_price,
        # exit_reason "tp" == the fixed 2R target was hit (+2R); "sl" == stop (-1R).
        # One target, so NOT "tp1"/"tp2" — avoids TP1/TP2 confusion.
        "exit_reason":   exit_reason,
        "r_realised":    r_realised,
        "pnl_usd":       pnl_usd,
        "mfe_r":         mfe_r,
        "mae_r":         mae_r,
        # OUTCOME-side descriptors (2026-07-26; NEVER entry/model features — pure
        # look-ahead). H1 bars from fill (bar 0) to the bar that SET mfe_r / mae_r,
        # captured O(1) inside the same walk. First bar to reach the extreme wins
        # ties. Loser autopsy: separates "died in 3 bars" from "bled for 40" and
        # powers the time-stop question. NULL (not 0) when a filled trade had no
        # post-fill bar. never_filled rows carry None.
        "bars_to_mfe":   bars_to_mfe,
        "bars_to_mae":   bars_to_mae,
        # Sweep diagnostics (SL exits only; None otherwise).
        #   sl_bar_was_sweep  : stop candle wicked the stop but closed back on our
        #                       side (SMC grab-then-reject) vs a clean close-through.
        #   sl_swept_then_2r  : STRICT — sweep bar AND price reached the +2R target
        #                       BEFORE re-hitting the fired stop cur_sl (swept, held,
        #                       ran the full 2R). None on non-sweep. HINT, not a replay.
        #   sl_swept_then_1r  : same, but the target is +1R (breakeven-plus). Shows
        #                       which stopped-out trades came back only to +1R vs ran
        #                       all the way to +2R (the wider-stop question).
        #   sl_wick_depth_atr : how far the stop candle's wick pierced BEYOND the
        #                       fired stop, in OB-formation ATR. The missing input
        #                       for sizing a "distal + X·ATR" wider-stop replay.
        #                       0.0 = closed at the stop; None = non-SL / no ATR.
        "sl_bar_was_sweep":  sl_bar_was_sweep,
        "sl_swept_then_2r":  sl_swept_then_2r,
        "sl_swept_then_1r":  sl_swept_then_1r,
        "sl_wick_depth_atr": sl_wick_depth_atr,
        # Outcome-time exit-track columns (NEVER entry features).
        #   sl_max_adverse_after_sweep_atr : furthest run against us BEYOND the
        #     stop after a sweep, in ATR — RAW context.
        #   bars_sl_to_2r_touch : STRICT — 1-indexed bars from stop bar to +2R,
        #     ONLY when swept-then-held-then-2R (sl_swept_then_2r True); else None.
        #   bars_sl_to_1r_touch : same, for +1R (sl_swept_then_1r True).
        #   sl_recovered_to_entry : after a sweep, did price return to entry (BE)?
        "sl_max_adverse_after_sweep_atr": sl_max_adverse_after_sweep_atr,
        "bars_sl_to_2r_touch":            bars_sl_to_2r_touch,
        "bars_sl_to_1r_touch":            bars_sl_to_1r_touch,
        "sl_recovered_to_entry":          sl_recovered_to_entry,
        # OB-formation -> fill gap (hours). Diagnostic only, corr with r ~0.
        "ob_to_fill_hours": ob_to_fill_hours,
        # H1 bars from break candle to the pullback that fills us (BS1 flag).
        "bars_break_to_pullback": bars_break_to_pullback,
        "bars_to_exit":  bars_to_exit,
        "ob_age_h1_bars": _ob_age_h1_bars(ob, df_h1, alert_ts),
        "ob_timestamp":  ob.get("ob_timestamp"),
        # Event-candle delta (2026-07-09): bars the true break candle sits before
        # the confirmation candle. 0 = clean single-candle break. Frozen event
        # fact, carried through from the zone (never recomputed here). Audits the
        # candle shift from the event-candle fix.
        "event_candle_delta": ob.get("event_candle_delta"),
        "pd_zone":       pd_zone,
        "pd_alignment":  pd_alignment,
        "pd_pct":        pd_pct,
        # CHoCH-origin-in-extreme flag (raw 1.0/0.0/None) + plain-English derived
        # boolean. Exact origin-based field for the reversal book — see the build
        # comment above. BOS rows carry None/False (no reversal origin).
        "reversal_pct":          reversal_pct,
        "reversed_from_extreme": reversed_from_extreme,
        "score":         round(float(score), 2),
        "structure_pts": round(float(breakdown.get("structure", 0.0)), 2),
        "sweep_pts":     round(float(breakdown.get("sweep", 0.0)), 2),
        "fvg_pts":       round(float(breakdown.get("fvg", 0.0)), 2),
        "freshness_pts": round(float(breakdown.get("freshness", 0.0)), 2),
        "killzone_pts":  round(float(breakdown.get("killzone", 0.0)), 2),
        "confluences_present": _confluences_present(breakdown),
        "session":       _fill_session(alert_ts, alert_ts),
        # Crypto weekend no-trade window (BTC: Sat 00:00 -> Mon 09:00 IST). When
        # the FILL lands in that window the trade is audit-only — the reporting
        # layer's _headline_exclusion drops it from P&L. False/absent for every
        # non-crypto pair (no weekend_block in config).
        "weekend_blocked": _in_weekend_block(fill_ts, pair_conf),
        "sl_collision":  sl_collision,
        "bos_tag":       bos_tag,
        "bos_tier":      bos_tier,
        # Continuation-drive verdict (holding / fading) AT ALERT TIME — carried
        # as a yield-payload scalar and applied via the alert-time OB view (T1),
        # so a multi-fire zone's traded row never logs a later fire's verdict.
        # 'fading' = the leg's recent break bodies decayed vs its start. Pair
        # with bos_sequence_count to see whether deep AND fading legs lose.
        "bos_verdict":   ob.get("bos_verdict", "holding"),
        # Continuation depth: # of BOS since the last CHoCH (CHoCH resets to 0,
        # each continuation BOS +1). Stamped on the OB by detect_smc_radar
        # (smc_radar.py). Surfaced here so the backtest can benchmark whether
        # the structure-score exhaustion penalty (late BOS -> low score) is
        # justified. CHoCH/Range rows carry the count at their event too.
        "bos_sequence_count": ob.get("bos_sequence_count"),
        "break_tier":        _bq.get("tier"),
        "break_close_atr":   _bq.get("close_beyond_atr"),
        "break_excess":      _bq.get("excess"),
        "break_body_atr":    _bq.get("body_atr"),
        # MSS label (2026-07-21): CHoCH break_body_atr >= MSS_BODY_ATR_MULT.
        # Descriptive only (NOT a proven separator — see the constant's comment
        # and the ledger). None on non-CHoCH rows.
        "is_mss":            is_mss,
        # Setup-geometry features (ATR-normalized; observe-only edge-engine inputs).
        "ob_range_atr":      ob_range_atr,
        "fvg_size_atr":      fvg_size_atr,
        # Displacement leg size in ATR, OB proximal -> the leg's structural
        # extreme (leg_extreme_at_alert). Alert-time, re-stamped on every re-fire.
        # Replaced the frozen impulse_leg_atr. Observe-only.
        "impulse_leg_to_extreme_atr": impulse_leg_to_extreme_atr,
        "atr_at_ob":         atr_at_ob,
        # ATR at the fill bar (fresh, point-in-time) — entry-regime vol vs the
        # formation-vol atr_at_ob. None on never_filled / short slice.
        "atr_at_fill":       atr_at_fill,
        # Era-stable fill-bar volatility regime (2026-07-25): fill-bar ATR(14)
        # ranked (pct) and ratio'd against its own trailing-90-day distribution
        # (bars strictly before fill). None on never_filled / <30d history.
        "atr_regime_pct_at_fill":   _atr_regime["atr_regime_pct_at_fill"],
        "atr_regime_ratio_at_fill": _atr_regime["atr_regime_ratio_at_fill"],
        # Derived-in-code columns (2026-07-08). Replaces the previously PASTED
        # sheet columns (sl_distance_atr / r_capture_ratio / trend_pd_agree) that
        # were CSV-corrupted. r_capture_ratio is OUTCOME-time (exit track only).
        "sl_distance_atr":   sl_distance_atr,
        # SL/TP distance vs NORMAL recent movement, at alert: proximal-anchored,
        # divided by a FRESH closed-bar ATR(14) (not formation ATR). See derivation.
        "sl_dist_atr_at_alert": sl_dist_atr_at_alert,
        "tp_dist_atr_at_alert": tp_dist_atr_at_alert,
        "r_capture_ratio":   r_capture_ratio,
        "trend_pd_agree":    trend_pd_agree,
        # Walk-back geometry (A3) — None for legacy zones built before this change.
        "ob_body_ratio":     ob_body_ratio,
        "ob_walkback_depth": ob_walkback_depth,
        # Choppiness Index on the alert's server trading day — daily trend-vs-
        # range regime at the alert bar. None when un-measurable. Observe-only.
        "chop_at_alert":     chop_at_alert,
        "fvg_present":   _frozen["fvg_present"],
        # fresh / stale / no_fvg — was the FVG already discharged on an earlier
        # approach before this trigger? Feeds the FVG-staleness breakdown.
        "fvg_state":     _fvg_state(_ob_at_alert, df_h1, alert_ts),
        # FVG mitigation label (none / pristine / partial / full) — the raw
        # discharge state of the gap, frozen at OB detection. Point-in-time clean.
        # Complements fvg_state (which is approach-relative) + fvg_size_atr (size).
        "fvg_mitigation": _frozen["fvg_mitigation"],
        # Proximal touch count AS OF THE ALERT (0 = pristine). Frozen at the
        # replay yield (touches_at_alert); the live ob["touches"] keeps updating
        # for the rest of the walk, so it must never be read here (Fix 3d).
        "ob_touches":    _touches_at_alert,
        # sweep_present now reads ob['sweep_v2'] (v1 retired 2026-07-24) — it is
        # the SAME value as sweep2_present below; the legacy column name is kept
        # so the diagnostics/reporting that reference it keep working, now on v2.
        "sweep_present": bool((ob.get("sweep_v2") or {}).get("exists")),
        # Session breakdown — OB formation vs fill, plus killzone alignment.
        # Fill session is the more honest label (when capital was actually
        # at work). OB session captures setup quality (institutional vs not).
        # Alignment buckets: Both / OB only / Fill only / Neither -- used by
        # email and Excel reporting to test the SMC veteran hypothesis that
        # both-in-killzone trades have a higher win rate.
        "ob_session":          _ob_session(ob),
        "fill_session":        _fill_session(fill_ts, alert_ts),
        "ob_in_killzone":      _ob_in_killzone(ob, pair_conf),
        "fill_in_killzone":    _fill_in_killzone(fill_ts, pair_conf),
        "killzone_alignment":  _killzone_alignment(ob, fill_ts, alert_ts, pair_conf),
        "h1_trend":            alert.get("h1_trend"),
        "trend_alignment":     alert.get("trend_alignment"),
        # ── STRUCTURE SIGNALS (STRUCTURE_SIGNALS_SPEC) ─────────────────────────
        # S2: v2 structure state at THIS alert (payload scalars, frozen at the
        # replay yield — never re-read off the shared ob dict). None only when
        # structure_v2 was missing (degraded walls).
        "flip_pending_at_alert":        alert.get("flip_pending_at_alert"),
        "flip_pending_dir_at_alert":    alert.get("flip_pending_dir_at_alert"),
        # S3 (DISPLACEMENT_LEG_BUILD_SPEC): structural displacement-leg extreme +
        # Kaufman ER over the span [ob_idx, extreme_end_idx] (through the break
        # candle to the leg's structural top). Both payload scalars, stamped at
        # the replay yield, sharing the exact same span. leg_extreme_clipped stays
        # informational (True only when the OB predates the slice — extreme None).
        # (leg_retrace_pct_at_alert removed 2026-07-19 — see comment at the S3
        # computation.)
        "leg_extreme_at_alert":         leg_extreme_at_alert,
        "leg_er_at_alert":              leg_er_at_alert,
        "leg_extreme_clipped":          leg_extreme_clipped,
        # S4: broken-wall PD flags read off the FROZEN ob["dealing_range"]
        # snapshot (immutable after OB build). None when the snapshot is
        # invalid / legacy.
        "dr_ceiling_broken_at_ob":      dr_ceiling_broken_at_ob,
        "dr_floor_broken_at_ob":        dr_floor_broken_at_ob,
        # Setup badge (email banner) — see build comment above. None = no
        # named pattern matched. kind is 'premium' or 'caution'; None otherwise.
        "setup_badge":         setup_badge,
        "setup_badge_kind":    setup_kind,
        # ── PD/PW LIQUIDITY POOLS (DAILY_BIAS_V4_SPEC §1.3) ────────────────────
        # 12 columns spread from ONE helper (day_state / pdh|pdl|pwh|pwl status /
        # nearest-unspent-pool distances+tiers / trade_toward_pool / last sweep
        # age+tier), anchored at the FILL bar: derived from H1 bars strictly
        # BEFORE fill_ts. The fill candle is the latest info a real trade holds;
        # alert-time pool status can be stale by the time the limit fills, so an
        # alert-anchored status is not what the trade actually saw (owner call
        # 2026-07-16). Observation only, no gate/score consumer, so fill-anchoring
        # crosses no look-ahead wall. never_filled rows (fill_ts=None) -> all-None
        # dict via the shim guard. The helper is defined BELOW simulate_h1_only_dual
        # on purpose: any line added above this return dict would shift every
        # ledger line-ref (tests/test_truth_ledger.py).
        # Column list: pool_builder.POOL_FEATURE_COLUMNS.
        **_pool_features_at_fill(df_h1, fill_ts, ob, entry),
        # ── EQH/EQL EQUAL-LEVEL CLUSTERS (2026-07-14) ──────────────────────────
        # 11 columns spread from ONE helper (nearest intact equal-highs /
        # equal-lows shelf distance+size / trade-toward / stop-vs-pool gap +
        # at-risk flag / last EQ sweep age+side / intact counts), anchored at the
        # FILL bar: derived from H1 bars strictly BEFORE fill_ts — same fill
        # anchor and rationale as the pool columns above. EQ is H1-only (built
        # from H1 swings, no resample). Observation only, no gate.
        # never_filled rows (fill_ts=None) -> all-None dict via the shim guard.
        # Column list: eq_pools.EQ_FEATURE_COLUMNS.
        **_eq_features_at_fill(df_h1, fill_ts, ob, entry, sl),
        # ── WEEKLY PD ZONE (higher-timeframe premium/discount, 2026-07-15) ─────
        # 5 columns spread from ONE helper: the weekly PD position (price vs
        # last COMPLETED week's high/low — may run <0 / >1 when price closed
        # beyond, which IS the break signal by owner decision), the weekly
        # range high/low, the premium/discount zone (split at 0.5), and the
        # H4-vs-weekly agreement (both_premium / both_discount / mixed). Weekly
        # levels are the SAME PWH/PWL the pool spread above uses (one weekly
        # derivation); the H4 read is the frozen pd_pct computed above. All at
        # alert from bars strictly BEFORE alert_ts. Observation only, no gate.
        # Column list: weekly_pd.WEEKLY_PD_FEATURE_COLUMNS.
        **_weekly_pd_features_at_alert(df_h1, alert_ts, entry, _pd_position_01),
        # ── APPROACH QUALITY (fill-time entry mechanics, RETRACE_QUALITY_SPEC) ─
        # 3 columns from ONE helper: how price travelled into the zone over the
        # closed bars strictly BEFORE the fill bar (speed toward zone in
        # formation-ATR, candle body share, Kaufman ER). FILL-time, NOT
        # alert-time — never an alert-time screen input (look-ahead wall).
        # All None when never_filled / thin history. Observation only, no gate.
        # Column list: approach_quality.APPROACH_FEATURE_COLUMNS.
        **_approach_features_at_fill(df_h1, fill_ts, ob),
        # ── SWEEP V2 (rebuilt pool-anchored sweep, 2026-07-18) ─────────────────
        # 12 columns spread from ONE helper, re-labelled off the FROZEN
        # ob['sweep_v2'] snapshot stamped at OB build inside detect_smc_radar
        # (the replay drives the same function — nothing is re-detected here).
        # Only sweep2_age_at_fill_h1 is derived: arithmetic on the frozen
        # sweep_ts against the FILL bar (fill-anchored to match the pool/eq/
        # approach columns; renamed from *_at_alert 2026-07-25). The 5
        # sweep2_sw_* cols are the best-SW always-on metrics off the same frozen
        # snapshot. Legacy zones / failed layer -> all-None dict. Observation only.
        # NOTE (2026-07-24): sweep v1 is retired — sweep_present above, sweep_pts
        # (the score leg) and every sweep2_* column now ALL read the one frozen
        # ob['sweep_v2'] snapshot. sweep_present == sweep2_present by construction.
        # Column list: liquidity_sweep.SWEEP2_FEATURE_COLUMNS.
        **_sweep2_features(ob, df_h1, fill_ts),
        # ── SETUP-LIQ (this trade's own stop/target vs swing liquidity) ────────
        # 6 columns from ONE helper. Reads 1 & 2 (stop-side / tp-side magnet)
        # were computed WITH the trade levels in _simulate_single_entry
        # (setup_liq_reads) and anchor on the FINAL SL / TP1 — NOT frozen at OB
        # build, because the anchor (SL/TP) is born from compute_phase2_levels.
        # Read 3.2 (leg-extreme-was-a-sweep) is a payload scalar from the replay
        # yield (leg_extreme_swept), anchored on leg geometry. All from bars
        # strictly at/before the alert (look-ahead-safe). Observation only, no
        # gate. Column list: setup_liq.SETUP_LIQ_FEATURE_COLUMNS.
        **_setup_liq_features(setup_liq_reads, alert.get("leg_extreme_swept")),
        # ── SESSION H/L SWEEP + BREAK (SESSION_SWEEP_STUDY_SPEC, 2026-07-21) ────
        # 3 columns from ONE helper: did price sweep or break the nearest prior
        # Asia/London/NY session high/low before this trade filled. DST-honest
        # session windows resolved PER CANDLE (session_levels, NOT the DST-broken
        # smc_detector._session_hl_until); sweep-vs-break decided by REUSING
        # pool_builder.pool_status (one implementation). ALERT-time, from bars
        # strictly BEFORE alert_ts (look-ahead-safe). Pair-specific study — never
        # pooled across pairs. Observation only, no gate/score/live consumer.
        # Column list: session_levels.SESSION_LEVEL_FEATURE_COLUMNS.
        **_session_level_features_at_alert(df_h1, alert_ts, entry, alert.get("pair")),
    }


def simulate_h1_only_dual(
    alert: Dict[str, Any],
    pair_conf: Dict[str, Any],
    df_h1: pd.DataFrame,
    risk_usd: float = DEFAULT_RISK_USD,
) -> List[Dict[str, Any]]:
    """Public entry point: simulate the proximal entry for one OB-touch alert.

    Returns [] if proximal levels are invalid (entry would chase price, zero risk,
    or the OB is thinner than the spread — a below-0.5R pool instead routes to the
    1:1 fallback, still a valid trade), else the one proximal trade row. (`_dual`
    is a historical name from the removed 50%
    A/B leg; it now yields a single proximal row.)
    """
    alert_ts = alert["ts"]
    if not isinstance(alert_ts, pd.Timestamp):
        alert_ts = pd.Timestamp(alert_ts)
    if alert_ts.tzinfo is None:
        alert_ts = alert_ts.tz_localize("UTC")

    # T1 + T4 (TRUTH_FIXES_SPEC / _2): ALERT-TIME view of the OB, built by the
    # single shared helper build_alert_ob_view (same fn the freeze test drives).
    # It swaps the alert-time bos_verdict / touches / fvg (both key spellings)
    # onto a copy of the OB, so scoring/badge/row read the alert moment, never the
    # post-alert-mutated live dict. One view, built once — never patch inline.
    ob_view = build_alert_ob_view(alert)
    alert = dict(alert)
    alert["ob"] = ob_view

    score, breakdown = _score_h1_only(alert, pair_conf, df_h1, alert_ts)

    # Proximal is the only live model. (The 50% mean-entry A/B leg was removed
    # 2026-07: it never traded live and its rows leaked into the exit-lab sink.)
    prox_row = _simulate_single_entry(
        alert, pair_conf, df_h1, "proximal", score, breakdown, risk_usd,
    )
    if prox_row is None:
        return []
    return [prox_row]


def _pool_features_at_fill(df_h1, fill_ts, ob, entry):
    """PD/PW pool columns for one row (pool_builder.POOL_FEATURE_COLUMNS).

    Thin shim over pool_builder.features_at_alert (a generic asof helper — the
    name is historical), anchored at the FILL bar: bars strictly before fill_ts
    only, per-frame day/week resample cached inside pool_builder. The fill candle
    is the latest info the trade holds; alert-anchored status can be stale by
    fill. Observation only (no gate/score consumer), so fill-anchoring crosses no
    look-ahead wall. ATR denominator = ob['h1_atr'] (frozen OB-formation ATR),
    matching every other *_atr feature column. ref_price = the placed entry.

    never_filled rows pass fill_ts=None -> all-None dict (no fill happened, so no
    fill-anchored status exists), mirroring _approach_features_at_fill.

    DEFINED AFTER _build_row / simulate_h1_only_dual on purpose: a top-of-file
    import or any code line inserted above _build_row's return dict would
    shift the ledger's row-build line-refs (tests/test_truth_ledger.py guards
    them). Python resolves this name at call time, so placement is safe.
    Never raises (pool_builder guarantees the all-None dict on failure).
    """
    import pool_builder
    if fill_ts is None:
        return dict.fromkeys(pool_builder.POOL_FEATURE_COLUMNS)
    return pool_builder.features_at_alert(
        df_h1, fill_ts,
        direction=ob.get("direction"),
        ref_price=entry,
        atr=ob.get("h1_atr"),
    )


def _eq_features_at_fill(df_h1, fill_ts, ob, entry, sl):
    """EQH/EQL cluster columns for one row (eq_pools.EQ_FEATURE_COLUMNS).

    Thin shim over eq_pools.features_at_alert (generic asof helper — name is
    historical), anchored at the FILL bar: bars strictly before fill_ts only,
    per-frame raw-swing pool cached inside eq_pools. EQ is H1-only (built from H1
    swings, no resample). Same fill anchor and rationale as _pool_features_at_fill
    above. ATR denominator = ob['h1_atr'] (frozen OB-formation ATR), matching
    every other *_atr column. ref_price = the placed entry; sl = the traded stop
    (sl_initial), feeding the eq_sl_gap_atr / eq_sl_at_risk geometry.

    never_filled rows pass fill_ts=None -> all-None dict, mirroring
    _approach_features_at_fill.

    Same deliberate placement as _pool_features_at_fill above (defined after
    _build_row so the ledger's row-build line-refs never shift). Never raises
    (eq_pools guarantees the all-None dict on failure).
    """
    import eq_pools
    if fill_ts is None:
        return dict.fromkeys(eq_pools.EQ_FEATURE_COLUMNS)
    return eq_pools.features_at_alert(
        df_h1, fill_ts,
        direction=ob.get("direction"),
        entry=entry,
        sl=sl,
        atr=ob.get("h1_atr"),
    )


def _sweep2_features(ob, df_h1, fill_ts):
    """Sweep-v2 columns for one row (liquidity_sweep.SWEEP2_FEATURE_COLUMNS).

    Shim over liquidity_sweep.features_from_snapshot. WS2 (2026-07-25) split:
      - SW + EQ blocks: re-labelled straight off the BIRTH-FROZEN ob['sweep_v2']
        snapshot (stamped once at OB build by detect_smc_radar; the replay runs
        the same function so the snapshot is point-in-time clean; the zone merge
        refreshes only fvg). No re-detection for these.
      - PW + PD blocks: RE-JUDGED AT THE FILL BAR (pw_pd_at_fill) on the OB's
        frozen fuel window, reading only closed bars strictly before fill_ts —
        because "yesterday's / last week's" H/L roll as time passes. This is the
        ONLY re-detection, and it is fill-anchored + look-ahead-clean by design.
    sweep2_age_at_fill_h1 = closed H1 bars from the earliest raid across present
    tiers to the FILL bar. Live parity: the frozen snapshot (and the LIVE winner
    read off it) is untouched — see the liquidity_sweep docstring parity note.

    Legacy zones (no snapshot) / failed layer -> all-None dict.

    Same deliberate placement as the pool / eq shims above (defined after
    _build_row so the ledger's row-build line-refs never shift). Never raises
    (liquidity_sweep guarantees the all-None dict on failure).
    """
    import liquidity_sweep
    return liquidity_sweep.features_from_snapshot(
        ob.get("sweep_v2"), df_h1, fill_ts)


def _setup_liq_features(setup_liq_reads, leg_extreme_swept):
    """Setup-liquidity columns for one row (setup_liq.SETUP_LIQ_FEATURE_COLUMNS).

    Pure assembly from the pre-computed Read 1/2 dict (setup_liq_reads, built in
    _simulate_single_entry when the trade levels were computed) and the Read 3.2
    payload scalar (leg_extreme_swept, from the replay yield). Nothing is
    re-detected here. A None reads dict (legacy path / degraded) -> all-None
    columns via the module contract.

    Same deliberate placement as the pool / eq / sweep2 shims above (defined
    after _build_row so the ledger's row-build line-refs never shift). Never
    raises (setup_liq guarantees the all-None dict on failure).
    """
    import setup_liq
    return setup_liq.features_from_reads(setup_liq_reads, leg_extreme_swept)


def _weekly_pd_features_at_alert(df_h1, alert_ts, entry, h4_pd_position):
    """Weekly-PD columns for one row (weekly_pd.WEEKLY_PD_FEATURE_COLUMNS).

    Thin shim over weekly_pd.features_at_alert — bars strictly before alert_ts
    only, weekly high/low from the SAME pool-layer resample (PWH/PWL), the H4
    read passed straight through as the already-computed frozen pd_pct/100
    (0-1). ref_price = the placed entry.

    Same deliberate placement as the pool / eq shims above (defined after
    _build_row so the ledger's row-build line-refs never shift). Never raises
    (weekly_pd guarantees the all-None dict on failure).
    """
    import weekly_pd
    return weekly_pd.features_at_alert(
        df_h1, alert_ts,
        ref_price=entry,
        h4_pd_position=h4_pd_position,
    )


def _session_level_features_at_alert(df_h1, alert_ts, entry, pair):
    """Session H/L sweep/break columns for one row
    (session_levels.SESSION_LEVEL_FEATURE_COLUMNS).

    Thin shim over session_levels.build_session_level_event, anchored at ALERT
    time: bars strictly BEFORE alert_ts only (SESSION_SWEEP_STUDY_SPEC §3c/§4.3 —
    frozen at alert, point-in-time, no future leak). The reported session is the
    MOST-RECENTLY-CLOSED one (recency, bounded — never an older nearest-in-price
    level); ref_price = the placed entry is only a within-session tiebreak when both
    the session high and low fired. `pair` drives the pair-relevance FLAG only
    (PAIR_SESSION_TAGS) — it never filters which sessions are scanned. DST-honest session windows are resolved per candle inside
    session_levels (NOT the DST-broken smc_detector._session_hl_until). Observation
    only — no gate/score consumer.

    Same deliberate placement as the pool / eq / sweep2 / weekly shims above
    (defined after _build_row so the ledger's row-build line-refs never shift).
    Never raises (session_levels guarantees the all-'none' dict on failure).
    """
    import session_levels
    # PERF (§3.9 Stage B, 2026-07-26): df_h1.index is sorted, so searchsorted+iloc
    # selects the SAME "strictly before alert_ts" rows as the old
    # `df_h1[df_h1.index < alert_ts]` boolean mask without copying the whole frame
    # per alert. side='left' -> first row with index >= alert_ts, so iloc[:pos] is
    # exactly the rows with index < alert_ts. Point-in-time wall unchanged.
    _pos = df_h1.index.searchsorted(pd.Timestamp(alert_ts), side="left")
    prior = df_h1.iloc[:_pos]
    return session_levels.build_session_level_event(prior, alert_ts, entry, pair)


def _approach_features_at_fill(df_h1, fill_ts, ob):
    """Approach-quality columns for one row (approach_quality.APPROACH_FEATURE_COLUMNS).

    Thin shim over approach_quality.features_at_fill — the 7 closed H1 bars
    strictly BEFORE the fill bar only (fill_ts is an exact bar timestamp from
    the walk, :750). FILL-time, NOT alert-time (look-ahead wall,
    RETRACE_QUALITY_SPEC §1.3). ATR denominator = ob['h1_atr'] (frozen
    OB-formation ATR), matching every other *_atr column. never_filled rows
    pass fill_ts=None (:878) -> all-None dict via the module contract.

    Same deliberate placement as the pool / eq / weekly-PD shims above (defined
    after _build_row so the ledger's row-build line-refs never shift). Never
    raises (approach_quality guarantees the all-None dict on failure).
    """
    import approach_quality
    return approach_quality.features_at_fill(
        df_h1, fill_ts,
        direction=ob.get("direction"),
        atr=ob.get("h1_atr"),
    )


# ── ATR VOLATILITY-REGIME AT FILL (2026-07-25) ──────────────────────────────
# atr_regime_pct_at_fill / atr_regime_ratio_at_fill: an ERA-STABLE volatility
# read. Raw ATR is era-dependent (a "normal" 2019 ATR dwarfs a "normal" 2008
# one), so a raw number means different things in different years. These two
# express the fill-bar ATR RELATIVE to that pair's own recent behaviour, so the
# read means the same thing in 2008 and 2019:
#   pct   = percentile rank (0-100) of the fill-bar ATR(14) within the SAME
#           ATR(14) sampled over the trailing 90 CALENDAR days before the fill.
#           Robust to freak spikes — the PRIMARY read.
#   ratio = fill-bar ATR(14) / mean of that 90-day ATR(14) distribution. Keeps
#           magnitude; logged alongside so analysis can pick which separates
#           winners better.
#
# CAUSAL — the fill bar is still FORMING when a live order fills, so only bars
# STRICTLY BEFORE fill_ts are closed and known. Both the fill-bar ATR and the
# 90-day distribution obey `< fill_ts` (strict), mirroring
# backtest/diagnostics/exit_lab.py _atr_at_fill (the same look-ahead ban).
#
# ATR = smc_detector.compute_atr's definition (simple mean of the last `period`
# true ranges, TR = max(H-L, |H-prevClose|, |L-prevClose|)) — ONE ATR
# definition system-wide. Here it is vectorized into a full rolling series so
# the read is O(1) per trade, not recomputed from scratch each row.
#
# BASELINE COMPOSITION: the 90-day baseline is EVERY H1 candle the feed has in
# the trailing 90 calendar days (continuous rolling window, no time-of-day
# filtering, no resampling). Weekend gaps are simply absent, not filled. This is
# the simple whole-window average — NOT bucketed by session or hour-of-day.
#
# NULL POLICY: fewer than 30 calendar days of H1 history strictly before the
# fill -> BOTH None (never a faked value). Also None when fill_ts is None
# (never_filled) or the fill-bar ATR itself is undefined (<15 prior bars).
#
# PERFORMANCE: the rolling ATR series is built ONCE per pair frame and memoized
# on the frame's content fingerprint (first_ts, last_ts, len) — the same
# frame object is reused across every alert on that pair within a run, so the
# series is computed once per pair per run and each trade does one lookup.
_ATR_REGIME_MIN_DAYS = 30           # min calendar-days of prior history -> else null
_ATR_REGIME_WINDOW_DAYS = 90        # trailing distribution window
_ATR_REGIME_SERIES_CACHE: Dict[Any, pd.Series] = {}


def _atr_regime_series(df_h1: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
    """CAUSAL rolling ATR(period) series aligned to bar timestamps.

    The value at bar t is the ATR computed on the bars UP TO AND INCLUDING t
    (mean of the 14 true ranges ending at t) — identical to what
    smc_detector.compute_atr(df.loc[:t]) returns, but vectorized. Bars before
    enough history (< period+1 bars, i.e. < period true ranges) are NaN.

    Memoized on the frame's content fingerprint so it is built once per pair
    per run. Returns None if the frame is too short for any ATR at all.
    """
    if df_h1 is None or len(df_h1) < period + 1:
        return None
    key = (df_h1.index[0], df_h1.index[-1], len(df_h1), period)
    cached = _ATR_REGIME_SERIES_CACHE.get(key)
    if cached is not None:
        return cached
    high = df_h1["High"].astype(float)
    low = df_h1["Low"].astype(float)
    prev_close = df_h1["Close"].astype(float).shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # tr.iloc[0] has no prev close -> NaN, matching compute_atr skipping bar 0.
    # min_periods=period so the first `period` closed TRs are required (== the
    # period+1 bars compute_atr demands); earlier bars stay NaN (undefined ATR).
    atr = tr.rolling(window=period, min_periods=period).mean()
    if len(_ATR_REGIME_SERIES_CACHE) > 64:   # a run touches <=~11 pair frames
        _ATR_REGIME_SERIES_CACHE.clear()
    _ATR_REGIME_SERIES_CACHE[key] = atr
    return atr


def _atr_regime_at_fill(df_h1: pd.DataFrame, fill_ts, period: int = 14) -> Dict[str, Any]:
    """{'atr_regime_pct_at_fill', 'atr_regime_ratio_at_fill'} for one trade.

    O(1)-ish: one lookup into the pre-built causal ATR series. See the section
    header above for the full contract (causality, ATR definition, null policy,
    performance). Both None on: never_filled, <period+1 prior bars (fill-bar ATR
    undefined), or < _ATR_REGIME_MIN_DAYS calendar-days of history before fill.
    """
    none = {"atr_regime_pct_at_fill": None, "atr_regime_ratio_at_fill": None}
    if fill_ts is None or df_h1 is None:
        return none
    series = _atr_regime_series(df_h1, period)
    if series is None:
        return none
    fill_ts = pd.Timestamp(fill_ts)
    if fill_ts.tzinfo is None and series.index.tz is not None:
        fill_ts = fill_ts.tz_localize("UTC")
    # CAUSAL cutoff: only bars STRICTLY BEFORE the still-forming fill bar.
    prior = series.loc[series.index < fill_ts].dropna()
    if prior.empty:
        return none
    # Need >= _ATR_REGIME_MIN_DAYS calendar-days of history before the fill,
    # measured on the ATR series' own defined span (first non-NaN .. last prior
    # bar). Too thin a history -> null, never a faked regime read.
    span_days = (prior.index[-1] - prior.index[0]).total_seconds() / 86400.0
    if span_days < _ATR_REGIME_MIN_DAYS:
        return none
    fill_atr = float(prior.iloc[-1])          # ATR known at the fill bar
    window_start = fill_ts - pd.Timedelta(days=_ATR_REGIME_WINDOW_DAYS)
    dist = prior.loc[prior.index >= window_start]
    if dist.empty:
        return none
    pct = round(float((dist <= fill_atr).sum()) / len(dist) * 100.0, 2)
    mean = float(dist.mean())
    ratio = round(fill_atr / mean, 4) if mean > 0 else None
    return {"atr_regime_pct_at_fill": pct, "atr_regime_ratio_at_fill": ratio}
