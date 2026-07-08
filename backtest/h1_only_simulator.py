"""H1-only trade simulator (proximal entry — the live model).

Tests the SMC system using ONLY H1 data — H1 finds the OB, entry happens at
the OB, SL/TP are sized off the H1 OB and H1 swing liquidity. No M15, no M5.

For every H1 OB-touch alert, this simulator fires ONE trade row: the proximal
entry (fills when price touches the OB proximal edge = the live limit). SL is
the OB distal +/- spread; TP price levels are the liquidity-based opposing H1
swings, reused from live compute_phase2_levels.

No scoring gate — every OB-touch is simulated regardless of confluence score.
Score is logged for post-run analysis (discover the optimal threshold empirically).

Per-trade outputs cover both exit policies (TP1 vs TP2) so the user can
see what their real-life TP1-only behaviour would produce vs the system's
default TP2 target.

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

# sl_swept_then_tp1 lookahead: bars after an SL exit to check for a reversal to
# TP1 (only when the stop bar itself was a sweep). Matches the hold horizon so a
# late reversal is still caught. Diagnostic only.
SL_SWEEP_LOOKBACK_BARS = MAX_HOLD_H1_BARS

# Backtest trade-existence floor (2026-07). Live rejects any setup whose best
# target clears < 1.5R; the backtest relaxes that to 0.5R so we can study the
# sub-1.5R population live never sees. This ONLY adds previously-rejected trades
# — TP1 selection still prefers a >= 1.5R target when one exists, so every trade
# that lives today keeps its exact TP1 (winners are never cut). See
# compute_phase2_levels(tp1_min_rr=...).
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

def _session_from_utc_hour(h: int) -> str:
    """Map UTC hour -> trading session label. Matches reporting._classify_session_utc."""
    if 0 <= h < 7:
        return "Asia"
    if 7 <= h < 13:
        return "London"
    if 13 <= h < 21:
        return "NY"
    return "Other"


def _ts_hour_utc(ts_val) -> Optional[int]:
    """Coerce ts (str / pd.Timestamp / None) to UTC hour, or None if unparseable."""
    if ts_val is None or ts_val == "":
        return None
    try:
        ts = pd.Timestamp(ts_val)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.hour)
    except Exception:
        return None


def _ob_session(ob: Dict[str, Any]) -> str:
    """Session label for the OB candle itself (when the institutional move
    that created the zone happened). 'unknown' if ob_timestamp missing."""
    h = _ts_hour_utc(ob.get("ob_timestamp"))
    return _session_from_utc_hour(h) if h is not None else "unknown"


def _fill_session(fill_ts, alert_ts) -> str:
    """Session at fill (when capital was at work). Falls back to alert hour
    for never_filled rows so the column is never empty."""
    h = _ts_hour_utc(fill_ts) if fill_ts is not None else None
    if h is None:
        h = _ts_hour_utc(alert_ts)
    return _session_from_utc_hour(h) if h is not None else "unknown"


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
    fvg_h1 = ob.get("fvg", {"exists": False, "was_detected": False,
                            "mitigation": "none"})
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


def _reference_touch_indices(future, bias, entry, sl, tp1, tp2, fill_bar_idx):
    """Replay the legacy 'ride to TP2 on original SL' policy over `future`
    (post-fill bars) purely to populate the REFERENCE columns r_if_exit_tp1 /
    r_if_exit_tp2. Independent of the live TP1+BE@1R walk so those study
    columns keep meaning exactly what they meant before the 2026-06-18 change:
    "did price touch TP1 / TP2 before the ORIGINAL stop would have hit."

    Returns (tp1_hit_bar_idx, tp2_hit_bar_idx, ref_exit_r_unscaled_price).
    The third value is the exit price under the legacy default policy (TP1->BE
    ->TP2), used as the fallback R when neither TP was touched.
    """
    sl_after_tp1 = sl
    tp1_idx = -1
    tp2_idx = -1
    ref_exit_price = None
    for i, (ts, bar) in enumerate(future.iterrows()):
        if i < fill_bar_idx:
            continue
        is_fill_bar = (i == fill_bar_idx)
        bar_hi = float(bar["High"]); bar_lo = float(bar["Low"])
        bars_post = i - fill_bar_idx
        if bars_post > MAX_HOLD_H1_BARS and ref_exit_price is None:
            ref_exit_price = float(bar["Close"])
            break
        if bias == "LONG":
            sl_hit = bar_lo <= sl_after_tp1
            tp1_hit = bar_hi >= tp1
            tp2_hit = (tp2 is not None) and (bar_hi >= tp2)
        else:
            sl_hit = bar_hi >= sl_after_tp1
            tp1_hit = bar_lo <= tp1
            tp2_hit = (tp2 is not None) and (bar_lo <= tp2)
        if is_fill_bar:
            tp1_hit = False; tp2_hit = False
        # SL-first on ANY collision bar (2026-07-02 fix): when the same bar hits
        # both the stop and a TP, the intra-bar order is unprovable, so the TP
        # touch must NOT be recorded. Previously tp1_idx/tp2_idx were stamped
        # BEFORE the SL check, so r_if_exit_tp1/r_if_exit_tp2 booked the TP as a
        # win on the very bar the walk itself resolved as SL — the reference
        # columns were optimistic exactly where the realised walk is pessimistic.
        if sl_hit:
            ref_exit_price = sl_after_tp1; break
        if tp1_hit and tp1_idx == -1:
            tp1_idx = bars_post
        if tp2_hit and tp2_idx == -1:
            tp2_idx = bars_post
        if tp2_hit:
            ref_exit_price = tp2; break
        if tp1_hit and sl_after_tp1 != entry:
            sl_after_tp1 = entry
    if ref_exit_price is None:
        # Window exhausted with position open under legacy policy.
        ref_exit_price = float(future.iloc[-1]["Close"]) if len(future) else entry
    return tp1_idx, tp2_idx, ref_exit_price


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

    Returns a row dict or None if the trade is invalid (e.g. no TP1 clearing
    1.5R — same gate as live). Returns a "never_filled" row when the limit is
    not touched within the hold window, so we can count the miss.
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
    # AFTER the alert (future liquidity), biasing both TP selection and the 1.5R
    # validity gate optimistically; passing UNBOUNDED past history made TP
    # selection depend on run start date instead of matching live. The forward
    # fill-walk below intentionally keeps the FULL df_h1 -- it must see the
    # future to simulate how the trade plays out.
    df_h1_at_alert = _closed_bars_at_alert(df_h1, alert_ts)

    try:
        levels = smc_detector.compute_phase2_levels(
            pair_conf, bias, ob, current_price, df_h1_at_alert,
            entry_zone=entry_zone, tp1_min_rr=BACKTEST_TP1_MIN_RR,
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
    entry  = float(levels["entry"])
    sl     = float(levels["sl"])
    tp1    = float(levels["tp1"])
    tp2    = float(levels["tp2"]) if levels.get("tp2") is not None else None
    tp1_rr = float(levels.get("rr", 0.0))
    tp2_rr = float(levels.get("tp2_rr", 0.0)) if tp2 is not None else 0.0

    # Apply pair spread to widen SL (worst-case execution). spread_pips is
    # the pair's typical broker spread. pip_size derived from decimal_places:
    # 4-5dp instruments (EURUSD, NZDUSD, USDCHF) -> pip = 0.0001
    # 2-3dp instruments (USDJPY, GOLD, NAS100)   -> pip = 0.01
    # For a LONG, SL sits below entry; spread pushes SL further down (worse).
    # For a SHORT, SL sits above entry; spread pushes SL further up (worse).
    # TP levels are NOT widened -- pessimistic, matches what the user gets
    # at the bid/ask after entering. Slippage and swap are NOT modelled
    # (user decision; revisit when needed). RCA #9.
    #
    # CRYPTO EXCEPTION: BTC is quoted in dollars and its spread is stated in
    # dollars (~$20), not in 0.01 "pips". Using pip_size=0.01 would shrink a $20
    # spread to $0.20 (100x too small) and flatter RR. For crypto, spread_pips is
    # read as a DOLLAR spread directly (pip_size = 1.0).
    spread_pips = float(pair_conf.get("spread_pips", 0.0))
    decimal_places = int(pair_conf.get("decimal_places", 5))
    if pair_conf.get("pair_type") == "crypto":
        pip_size = 1.0
    else:
        pip_size = 0.01 if decimal_places <= 3 else 0.0001
    spread_price = spread_pips * pip_size
    # Pre-spread stop (the raw OB distal boundary), logged so a spread audit is a
    # clean two-column diff (sl_raw vs sl_initial) instead of a reconstruction.
    sl_raw = sl
    if spread_price > 0:
        if bias == "LONG":
            sl = sl - spread_price
        else:
            sl = sl + spread_price

    r_distance = abs(entry - sl)
    if r_distance <= 0:
        log_event("h1only_sim_skip", level="warn", pair=pair,
                  entry_zone=entry_zone, alert_ts=str(alert_ts),
                  reason="zero_r_distance")
        return None

    # Defense in depth: drop the trade if TP2 is on the wrong side of TP1.
    # compute_phase2_levels already filters this; this guard catches any
    # future regression or forced-TP path where the upstream check is bypassed.
    if tp2 is not None:
        bad = (bias == "LONG" and tp2 <= tp1) or (bias == "SHORT" and tp2 >= tp1)
        if bad:
            log_event("h1only_sim_skip", level="error", pair=pair,
                      entry_zone=entry_zone, alert_ts=str(alert_ts),
                      reason="tp_order_invalid",
                      tp1=tp1, tp2=tp2, bias=bias)
            return None

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
    # NOTE the earlier "cloned-fill" RCA (2026-03): that was fixed by the OB dedup
    # in run_backtest (first alert per OB only) — NOT by the +1h, which was an
    # over-correction on top. Filling on candle B is safe because dedup already
    # removed the identical re-fire rows.
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
    tp1_hit_bar_idx = -1
    tp2_hit_bar_idx = -1
    mfe_price = entry
    mae_price = entry
    sl_collision = False
    bars_walked_post_fill = 0
    bars_to_tp1 = -1
    bars_to_tp2 = -1

    # ── LIVE policy state (2026-06-18): TP1 + break-even at +1R ──────────────
    # The realised policy is now: fill -> if price reaches +1R move SL to entry
    # -> exit at TP1. TP2 is never traded; it survives only as a reference
    # column (r_if_exit_tp2) for the MFE/TP2 study. `r_realised` and `pnl_usd`
    # follow THIS policy and nothing else.
    #
    # `cur_sl` is the live stop: starts at the initial SL, jumps to entry once
    # the +1R break-even arms. The walk BREAKS at the realised exit, so mfe_r /
    # mae_r measure IN-TRADE excursion only (fill -> exit), never the post-exit
    # path. The full-window touch diagnostics (bars_to_tp1 / bars_to_tp2) and
    # the r_if_exit_* reference columns come from the SEPARATE legacy walk in
    # _reference_touch_indices, which ignores the realised exit.
    cur_sl = sl
    be_armed = False
    be_trigger = (entry + r_distance) if bias == "LONG" else (entry - r_distance)
    # FP-boundary tolerance (2026-07-03): be_trigger = entry +/- r_distance carries
    # accumulated float error (e.g. 1.0151000000000001 vs a bar high of 1.0151), so
    # a bar that touches EXACTLY +1R can fail `bar_hi >= be_trigger` by ~2e-16 while
    # MFE still credits +1R off the raw high. That split produced a physically
    # impossible row (exit sl at -1R with mfe_r rounded to +1.0) — G10 rule (b). Arm
    # break-even when price reaches +1R within this tolerance so the BE trigger and
    # the MFE recorder agree on the exact-touch bar. Scaled to r_distance so it is
    # instrument-agnostic and far below any real tick.
    be_eps = r_distance * 1e-6
    # Measurement only (2026-07-02, no behavior change): on the bar that arms
    # break-even, did that SAME bar's wick also trade back to entry? If so the
    # intra-bar order of "+1R first, arm, THEN pull back to entry" vs "pull back
    # to entry first, THEN +1R" is unprovable -- we arm at bar CLOSE regardless
    # and let the trade ride, which is optimistic if the pullback actually came
    # first (a live break-even stop would have been tagged for 0R that bar).
    # Logged so the ambiguous population size can be measured before deciding
    # whether it's worth a rule.
    be_arm_bar_touched_entry: Optional[bool] = None

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
            # Pending limit fill: long fills when bar.low <= entry,
            # short fills when bar.high >= entry (pre-placed limit semantics).
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
        if bars_walked_post_fill > MAX_HOLD_H1_BARS and exit_reason is None:
            exit_ts = ts
            exit_reason = "timeout"
            exit_price = float(bar["Close"])
            break

        # Weekend-flat: force-close an OPEN position before the FX weekend.
        # Realised walk only -- the reference TP1/TP2 study columns ignore it.
        if (WEEKEND_FLAT and not is_fill_bar_this_iter and exit_reason is None
                and ts.dayofweek == 4 and ts.hour >= WEEKEND_FLAT_HOUR_UTC):
            exit_ts = ts
            exit_reason = "friday_flat"
            exit_price = float(bar["Open"])
            break

        if bias == "LONG":
            sl_hit_in_bar = bar_lo <= cur_sl
            tp1_hit_in_bar = bar_hi >= tp1
            tp2_hit_in_bar = (tp2 is not None) and (bar_hi >= tp2)
            be_reached_in_bar = bar_hi >= be_trigger - be_eps
            # MFE/MAE track the trade's IN-TRADE excursion only (fill -> exit).
            # Three bars cannot contribute their raw extremes:
            #   - SL bar: the wick that touched SL also printed the bar high, so
            #     crediting that high fakes a positive excursion on the very bar
            #     that stopped us out.
            #   - FILL bar: a LONG limit fills on the bar LOW; that bar's HIGH
            #     happened BEFORE the fill (price fell through entry), so it is
            #     pre-fill price, not favourable excursion. Crediting it fakes a
            #     large MFE on a bar where price was actually falling to SL.
            #   - TP1-EXIT bar (2026-07-02): the realised exit fills AT the first
            #     TP1 touch, so any price beyond TP1 printed at-or-after the exit
            #     (post-exit, not ours); MFE is capped at TP1. The bar's LOW is
            #     order-ambiguous (before or after the touch) -> no MAE update.
            # None of these bars lets us infer the intra-bar sequence.
            if not sl_hit_in_bar and not is_fill_bar_this_iter:
                if tp1_hit_in_bar:
                    mfe_price = max(mfe_price, tp1)
                else:
                    mfe_price = max(mfe_price, bar_hi)
                    mae_price = min(mae_price, bar_lo)
        else:
            sl_hit_in_bar = bar_hi >= cur_sl
            tp1_hit_in_bar = bar_lo <= tp1
            tp2_hit_in_bar = (tp2 is not None) and (bar_lo <= tp2)
            be_reached_in_bar = bar_lo <= be_trigger + be_eps
            if not sl_hit_in_bar and not is_fill_bar_this_iter:
                if tp1_hit_in_bar:
                    mfe_price = min(mfe_price, tp1)
                else:
                    mfe_price = min(mfe_price, bar_lo)
                    mae_price = max(mae_price, bar_hi)

        # Fill-bar rule (2026-05-25):
        # On the bar where the limit just filled, we cannot infer intra-bar
        # sequence of fill -> TP vs fill -> SL. SL-side: if the bar pierced
        # SL, price had to travel through entry first (limit fills, then SL),
        # so SL is the honest outcome. TP-side: bar high reaching TP could
        # mean (a) price ticked up to TP before pulling down to fill, OR (b)
        # filled then rallied to TP. Can't tell. Conservative call: do NOT
        # credit TP (or arm break-even) on the fill bar. Walk forward.
        if is_fill_bar_this_iter:
            tp1_hit_in_bar = False
            tp2_hit_in_bar = False
            be_reached_in_bar = False

        # Record first-touch bar indices for diagnostic columns.
        if tp1_hit_in_bar and tp1_hit_bar_idx == -1:
            tp1_hit_bar_idx = bars_walked_post_fill
        if tp2_hit_in_bar and tp2_hit_bar_idx == -1:
            tp2_hit_bar_idx = bars_walked_post_fill

        # ── Realised exit: TP1 + break-even at +1R ──────────────────────────
        # Priority within a bar:
        #   1. SL+TP1 collision -> SL wins (conservative, matches legacy).
        #   2. SL hit (at cur_sl: initial SL, or entry once BE armed).
        #   3. TP1 hit -> win, terminal (no TP2 ride).
        #   4. Else arm break-even if +1R reached this bar.
        # +1R-and-SL in the same bar (pre-arm) falls into case 1/2: SL wins,
        # because we cannot prove +1R printed before the stop.
        if sl_hit_in_bar and tp1_hit_in_bar:
            sl_collision = True
            exit_ts = ts
            exit_reason = "sl"
            exit_price = cur_sl
            break
        if sl_hit_in_bar:
            exit_ts = ts
            exit_reason = "sl"
            exit_price = cur_sl
            break
        if tp1_hit_in_bar:
            exit_ts = ts
            exit_reason = "tp1"
            exit_price = tp1
            break
        if be_reached_in_bar and not be_armed:
            # Price reached +1R without hitting SL/TP1 this bar -> move the
            # stop to entry. A later pullback to entry now books 0R.
            be_armed = True
            cur_sl = entry
            be_arm_bar_touched_entry = bool(
                bar_lo <= entry if bias == "LONG" else bar_hi >= entry
            )

    if not filled:
        # Limit never touched within the hold window. Emit a "never_filled" row
        # so the report can count "would-have-missed" trades.
        bars_to_exit = bars_walked_post_fill
        return _build_row(
            alert=alert, pair_conf=pair_conf, ob=ob,
            entry_zone=entry_zone, entry=entry, sl=sl, sl_raw=sl_raw, tp1=tp1, tp2=tp2,
            tp1_rr=tp1_rr, tp2_rr=tp2_rr,
            score=score, breakdown=breakdown,
            df_h1=df_h1, alert_ts=alert_ts,
            fill_ts=None, exit_ts=None, exit_reason="never_filled",
            exit_price=None,
            r_realised=0.0, r_if_exit_tp1=0.0, r_if_exit_tp2=0.0,
            mfe_r=0.0, mae_r=0.0, bars_to_exit=0,
            bars_to_tp1=-1, bars_to_tp2=-1,
            sl_collision=False, risk_usd=risk_usd,
            sl_bar_was_sweep=None, sl_swept_then_tp1=None, ob_to_fill_hours=None,
            bars_break_to_pullback=None,
        )

    if exit_reason is None:
        # Window exhausted with position open and no SL/TP hit.
        last = future.iloc[-1]
        exit_ts = future.index[-1]
        exit_reason = "window_end"
        exit_price = float(last["Close"])

    # ── r_realised: the LIVE policy (TP1 + break-even at +1R). This is the
    # one true outcome — pnl_usd and every report headline derive from it.
    if bias == "LONG":
        r_realised = (exit_price - entry) / r_distance
        mfe_r = (mfe_price - entry) / r_distance
        mae_r = -(entry - mae_price) / r_distance
    else:
        r_realised = (entry - exit_price) / r_distance
        mfe_r = (entry - mfe_price) / r_distance
        mae_r = -(mae_price - entry) / r_distance

    # ── Reference columns (study only — never traded). Computed from an
    # independent legacy walk (original SL, TP1->BE->TP2 ride) so they answer
    # "what would TP1-only / TP2-ride have produced" regardless of where the
    # live TP1+BE walk exited.
    ref_tp1_idx, ref_tp2_idx, ref_exit_price = _reference_touch_indices(
        future, bias, entry, sl, tp1, tp2, fill_bar_idx
    )
    if bias == "LONG":
        ref_realised = (ref_exit_price - entry) / r_distance
    else:
        ref_realised = (entry - ref_exit_price) / r_distance

    # Overwrite the diagnostic touch indices with the reference-walk values so
    # bars_to_tp1 / bars_to_tp2 describe the full-window touches (the live walk
    # breaks early at TP1 and would under-report them).
    tp1_hit_bar_idx = ref_tp1_idx
    tp2_hit_bar_idx = ref_tp2_idx

    # r_if_exit_tp1: TP1 touched (on original stop) -> book TP1, else legacy R.
    if ref_tp1_idx >= 0:
        r_if_exit_tp1 = round(tp1_rr, 3)
    else:
        r_if_exit_tp1 = round(ref_realised, 3)

    # r_if_exit_tp2: TP2 touched -> book TP2, else the legacy ride outcome.
    if ref_tp2_idx >= 0:
        r_if_exit_tp2 = round(tp2_rr, 3) if tp2 is not None else round(ref_realised, 3)
    else:
        r_if_exit_tp2 = round(ref_realised, 3)

    bars_to_exit = max(0, bars_walked_post_fill)

    # ── Sweep diagnostics: was the STOP a liquidity grab, and did it reverse? ──
    # SMC definition of a sweep: the candle WICKS through the level but CLOSES BACK
    # on our side (grab-then-reject). A candle that CLOSES THROUGH the stop is a
    # genuine break, not a sweep — and a wider stop would just lose more.
    #
    #   sl_bar_was_sweep   : the STOP CANDLE itself was a sweep of the stop that
    #                        fired (cur_sl — the initial SL, or entry once BE armed).
    #                        Long : Low <= cur_sl AND Close > cur_sl.
    #                        Short: High >= cur_sl AND Close < cur_sl.
    #   sl_swept_then_tp1  : sl_bar_was_sweep is True AND, within
    #                        SL_SWEEP_LOOKBACK_BARS bars AFTER the stop bar, price
    #                        reached TP1 in our direction. This is the honest
    #                        "a wider stop plausibly wins" signal.
    #
    # HONEST CAVEAT (peak-vs-fill law): sl_swept_then_tp1 still ends on a TOUCH
    # check of a later bar, not a real-order replay — that later TP1 tag could be
    # its own spike-and-fade. Read it as a strong HINT ("would a wider stop have
    # saved us"), never as bankable "free money". Both are None for non-SL exits.
    # cur_sl at this point is the level that actually stopped the trade (BE-aware).
    sl_bar_was_sweep = None
    sl_swept_then_tp1 = None
    # sl_wick_depth_atr (2026-07-08): how far the STOP CANDLE's wick pierced BEYOND
    # the stop that fired (cur_sl — initial SL, or entry once BE armed), normalised
    # by the OB-formation ATR (ob['h1_atr'], the same denominator every *_atr
    # feature uses). This is the missing input for sizing a wider stop: a sweep
    # tells us the wick crossed the stop, this tells us HOW FAR, so a "distal + X·ATR"
    # replay grid can be chosen from data instead of guessed.
    #   LONG  (stop below): depth = (cur_sl - sl_bar_low)  / h1_atr   [>=0]
    #   SHORT (stop above): depth = (sl_bar_high - cur_sl) / h1_atr   [>=0]
    # 0.0 = the wick closed exactly at the stop (no overshoot). None for non-SL exits
    # or when h1_atr is unavailable (legacy zone). NOT a gate — logging only.
    sl_wick_depth_atr = None
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

            # Later-reversal check only matters when the stop bar was a sweep.
            if sl_bar_was_sweep:
                post_sl = future.loc[future.index > exit_ts]
                horizon = post_sl.iloc[:SL_SWEEP_LOOKBACK_BARS]
                if len(horizon):
                    if bias == "LONG":
                        sl_swept_then_tp1 = bool((horizon["High"] >= tp1).any())
                    else:
                        sl_swept_then_tp1 = bool((horizon["Low"] <= tp1).any())
                else:
                    sl_swept_then_tp1 = False
            else:
                sl_swept_then_tp1 = False
        except Exception:
            sl_bar_was_sweep = None
            sl_swept_then_tp1 = None
            sl_wick_depth_atr = None

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
    if EXIT_LAB_CONFIGS and EXIT_LAB_SINK is not None and fill_bar_idx >= 0:
        from backtest.exit_engine import walk_multileg
        _post = future.iloc[fill_bar_idx:]
        for _name, _cfg in EXIT_LAB_CONFIGS.items():
            try:
                _res = walk_multileg(
                    _post, bias, entry, sl, r_distance, tp1, _cfg,
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
        entry_zone=entry_zone, entry=entry, sl=sl, sl_raw=sl_raw, tp1=tp1, tp2=tp2,
        tp1_rr=tp1_rr, tp2_rr=tp2_rr,
        score=score, breakdown=breakdown,
        df_h1=df_h1, alert_ts=alert_ts,
        fill_ts=fill_ts, exit_ts=exit_ts, exit_reason=exit_reason,
        exit_price=exit_price,
        r_realised=round(r_realised, 3),
        r_if_exit_tp1=r_if_exit_tp1,
        r_if_exit_tp2=r_if_exit_tp2,
        mfe_r=round(mfe_r, 3), mae_r=round(mae_r, 3),
        bars_to_exit=bars_to_exit,
        bars_to_tp1=tp1_hit_bar_idx,
        bars_to_tp2=tp2_hit_bar_idx,
        sl_collision=sl_collision, risk_usd=risk_usd,
        sl_bar_was_sweep=sl_bar_was_sweep,
        sl_swept_then_tp1=sl_swept_then_tp1,
        sl_wick_depth_atr=sl_wick_depth_atr,
        ob_to_fill_hours=ob_to_fill_hours,
        bars_break_to_pullback=bars_break_to_pullback,
        be_arm_bar_touched_entry=be_arm_bar_touched_entry,
    )


# FIX 3e — OB-field classification for the trade row (closes the "mutable state
# logged from a frozen snapshot" bug class). Any NEW ob field logged in a trade
# row MUST fall into one of these buckets:
#   IMMUTABLE EVENT FACTS (freezing correct): bos_timestamp, ob_timestamp,
#     direction, bos_tag, bos_tier, bos_swing_price, impulse_start_price, high,
#     low, proximal_line, distal_line, median_leg_body, ob_body, h1_atr
#     (formation ATR, frozen by design), reversal_pct, broken_was_wall,
#     bos_sequence_count, last_choch_idx.
#   FROZEN-BY-DESIGN, LIVE DOES THE SAME: dealing_range (incl. S4
#     dr_ceiling_broken_at_ob / dr_floor_broken_at_ob, read off the frozen
#     snapshot), sweep_observed.
#   STAMPED AT ALERT (correct source): bos_verdict, touches_at_alert +
#     fvg_at_alert, h1_trend / trend_alignment / alert_bar_*, and the S2/S3
#     structure signals (structure_ranging_at_alert, flip_pending_at_alert,
#     flip_pending_dir_at_alert, leg_extreme_at_alert, leg_extreme_clipped —
#     all payload scalars from the replay yield; leg_retrace_pct_at_alert is
#     derived here from leg_extreme_at_alert + the placed entry).
#   MUTABLE STATE, fixed by this spec: touches/status (3a), break_quality (3b),
#     fvg (3c/3d).
# RULE: mutable state is stamped `*_at_alert` at the replay yield and read from
# that snapshot here — NEVER read live off the ob dict at row-build time (the
# replay keeps mutating it after the alert).
def _build_row(*, alert, pair_conf, ob, entry_zone, entry, sl, tp1, tp2,
               tp1_rr, tp2_rr, score, breakdown, df_h1, alert_ts,
               fill_ts, exit_ts, exit_reason, exit_price,
               r_realised, r_if_exit_tp1, r_if_exit_tp2,
               mfe_r, mae_r, bars_to_exit, bars_to_tp1, bars_to_tp2,
               sl_collision, risk_usd, sl_raw=None,
               sl_bar_was_sweep=None, sl_swept_then_tp1=None,
               sl_wick_depth_atr=None, ob_to_fill_hours=None,
               bars_break_to_pullback=None,
               be_arm_bar_touched_entry=None) -> Dict[str, Any]:
    """Assemble the final trade row dict in stable column order."""
    direction = ob.get("direction", "?")
    # FIX 3d: mutable OB state (touches, fvg) is frozen at the replay yield into
    # touches_at_alert / fvg_at_alert. Read those here — the live ob["touches"]/
    # ob["fvg"] keep changing after the alert as the per-bar loop walks on.
    # `_ob_at_alert` is a shallow view carrying the alert-time fvg so the fvg
    # helpers (which read ob["fvg"] internally) classify at the alert moment.
    _touches_at_alert = ob.get("touches_at_alert", ob.get("touches"))
    _fvg_at_alert = ob.get("fvg_at_alert")
    if _fvg_at_alert is not None:
        _ob_at_alert = dict(ob)
        _ob_at_alert["fvg"] = _fvg_at_alert
    else:
        _ob_at_alert = ob  # legacy OB with no alert-time snapshot -> live read
    bos_tag = ob.get("bos_tag", "BOS")
    bos_tier = ob.get("bos_tier", "Major")
    # Break quality of the BOS/CHoCH candle — computed ONCE by smc_radar at
    # detection (smc_detector.compute_break_quality) and carried on the OB. Never
    # recomputed here; we only surface the frozen numbers so the backtest can
    # benchmark what break ATR multiple actually wins, per event type.
    #   break_close_atr = raw ATR multiple the close cleared the broken level by
    #   break_excess    = times over that event's required ATR floor (BOS 0.4 / CHoCH 1.0)
    _bq = ob.get("break_quality") or {}
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
    # invalid / legacy (the flag was never resolvable).
    if isinstance(dr, dict) and dr.get("valid"):
        _dr_ceiling_broken = dr.get("ceiling_broken")
        _dr_floor_broken = dr.get("floor_broken")
        dr_ceiling_broken_at_ob = bool(_dr_ceiling_broken) if _dr_ceiling_broken is not None else None
        dr_floor_broken_at_ob = bool(_dr_floor_broken) if _dr_floor_broken is not None else None
    else:
        dr_ceiling_broken_at_ob = None
        dr_floor_broken_at_ob = None
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

    # FVG size in ATR — the displacement gap's magnitude (present/absent throws
    # this gradient away). FVG-mitigation-agnostic: measures the gap as detected.
    _fvg = ob.get("fvg") or {}
    if _fvg.get("exists") and _fvg.get("fvg_top") is not None and _fvg.get("fvg_bottom") is not None:
        fvg_size_atr = _atr_norm(abs(float(_fvg["fvg_top"]) - float(_fvg["fvg_bottom"])))
    else:
        fvg_size_atr = None

    # Impulse-leg size in ATR — the displacement that broke structure and left
    # the OB behind: origin (impulse_start) -> the swing it broke (bos_swing).
    # FVG-INDEPENDENT by construction (defined on the swing structure, not the
    # gap), per the SMC definition. Measures the leg up to the break level; a
    # full-extreme variant (origin -> leg high/low) is a future refinement but
    # needs the detection-frame bars, so it can't be done cross-fetch here.
    _isp = ob.get("impulse_start_price")
    _bsp = ob.get("bos_swing_price")
    impulse_leg_atr = _atr_norm(abs(float(_bsp) - float(_isp))) \
        if (_isp is not None and _bsp is not None) else None

    # Raw OB-formation ATR (price units) — volatility context. NOT cross-instrument
    # comparable raw; the engine normalizes it within-pair (vs the pair's typical
    # ATR) for a regime read. Logged because it's free and frozen.
    atr_at_ob = round(float(_h1_atr), 6) if _h1_atr else None

    # ── S3 (STRUCTURE_SIGNALS_SPEC): leg retracement at alert ──────────────────
    # How much of the displacement leg price has given back by the time the OB
    # fills. Stamped at alert: `leg_extreme_at_alert` (the a-priori extreme since
    # OB formation) is a payload scalar frozen at the replay yield; `entry` is the
    # limit the simulator placed. impulse_start_price is the leg origin (frozen at
    # OB build). retrace_pct is derived HERE because only _build_row knows `entry`.
    #   bullish OB: (leg_extreme - entry) / (leg_extreme - impulse_start) * 100
    #   bearish OB: (entry - leg_extreme) / (impulse_start - leg_extreme) * 100
    # None (never a crash) when: extreme not stamped, impulse_start missing, or a
    # degenerate denominator (extreme not beyond the leg origin). >100 is VALID
    # (price ran past the leg origin) and is NOT clamped. `leg_extreme_clipped`
    # (payload) flags an OB older than the point-in-time slice — extreme may be
    # understated; the flag keeps the column honest.
    leg_extreme_at_alert = alert.get("leg_extreme_at_alert")
    leg_extreme_clipped = alert.get("leg_extreme_clipped")
    leg_retrace_pct_at_alert = None
    if leg_extreme_at_alert is not None and _isp is not None:
        try:
            _lex = float(leg_extreme_at_alert)
            _leg_origin = float(_isp)
            if direction == "bullish":
                _denom = _lex - _leg_origin
                if _denom > 0:
                    leg_retrace_pct_at_alert = round((_lex - entry) / _denom * 100, 1)
            else:
                _denom = _leg_origin - _lex
                if _denom > 0:
                    leg_retrace_pct_at_alert = round((entry - _lex) / _denom * 100, 1)
        except (TypeError, ValueError):
            leg_retrace_pct_at_alert = None

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
        "entry":         entry,
        # sl_raw = the raw OB distal (pre-spread) stop. sl_initial = sl_raw widened
        # by one spread (the stop actually traded). Logging both makes a spread
        # audit a clean diff instead of reconstructing sl_raw from sl_initial.
        "sl_raw":        sl_raw,
        "sl_initial":    sl,
        "tp1":           tp1,
        "tp2":           tp2,
        "tp1_rr":        round(tp1_rr, 3),
        "tp2_rr":        round(tp2_rr, 3) if tp2 is not None else None,
        "exit_price":    exit_price,
        "exit_reason":   exit_reason,
        "r_realised":    r_realised,
        "r_if_exit_tp1": r_if_exit_tp1,
        "r_if_exit_tp2": r_if_exit_tp2,
        "pnl_usd":       pnl_usd,
        "mfe_r":         mfe_r,
        "mae_r":         mae_r,
        # Sweep diagnostics (SL exits only; None otherwise).
        #   sl_bar_was_sweep  : stop candle wicked the stop but closed back on our
        #                       side (SMC grab-then-reject) vs a clean close-through.
        #   sl_swept_then_tp1 : sweep bar AND price later reached TP1 — the honest
        #                       "wider stop plausibly wins" HINT (still a touch
        #                       check, not a real-order replay; never free money).
        #   sl_wick_depth_atr : how far the stop candle's wick pierced BEYOND the
        #                       fired stop, in OB-formation ATR. The missing input
        #                       for sizing a "distal + X·ATR" wider-stop replay.
        #                       0.0 = closed at the stop; None = non-SL / no ATR.
        "sl_bar_was_sweep":  sl_bar_was_sweep,
        "sl_swept_then_tp1": sl_swept_then_tp1,
        "sl_wick_depth_atr": sl_wick_depth_atr,
        # Measurement only, no behavior change (2026-07-02). True when the bar
        # that armed break-even (+1R touch) ALSO traded back to entry within the
        # same bar -- the intra-bar order (arm-then-pullback vs pullback-then-arm)
        # is unprovable, so we arm at bar close and let the trade ride either way.
        # None when BE never armed (SL/TP1 hit first, or trade never reached +1R).
        "be_arm_bar_touched_entry": be_arm_bar_touched_entry,
        # OB-formation -> fill gap (hours). Diagnostic only, corr with r ~0.
        "ob_to_fill_hours": ob_to_fill_hours,
        # H1 bars from break candle to the pullback that fills us (BS1 flag).
        "bars_break_to_pullback": bars_break_to_pullback,
        "bars_to_exit":  bars_to_exit,
        "bars_to_tp1":   bars_to_tp1,
        "bars_to_tp2":   bars_to_tp2,
        "ob_age_h1_bars": _ob_age_h1_bars(ob, df_h1, alert_ts),
        "ob_timestamp":  ob.get("ob_timestamp"),
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
        "session":       _session_from_utc_hour(alert_ts.hour),
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
        # Setup-geometry features (ATR-normalized; observe-only edge-engine inputs).
        "ob_range_atr":      ob_range_atr,
        "fvg_size_atr":      fvg_size_atr,
        "impulse_leg_atr":   impulse_leg_atr,
        "atr_at_ob":         atr_at_ob,
        # Walk-back geometry (A3) — None for legacy zones built before this change.
        "ob_body_ratio":     ob_body_ratio,
        "ob_walkback_depth": ob_walkback_depth,
        "fvg_present":   bool((_fvg_at_alert or ob.get("fvg") or {}).get("exists")),
        # fresh / stale / no_fvg — was the FVG already discharged on an earlier
        # approach before this trigger? Feeds the FVG-staleness breakdown.
        "fvg_state":     _fvg_state(_ob_at_alert, df_h1, alert_ts),
        # FVG mitigation label (none / pristine / partial / full) — the raw
        # discharge state of the gap, frozen at OB detection. Point-in-time clean.
        # Complements fvg_state (which is approach-relative) + fvg_size_atr (size).
        "fvg_mitigation": (_fvg_at_alert or ob.get("fvg") or {}).get("mitigation"),
        # Proximal touch count AS OF THE ALERT (0 = pristine). Frozen at the
        # replay yield (touches_at_alert); the live ob["touches"] keeps updating
        # for the rest of the walk, so it must never be read here (Fix 3d).
        "ob_touches":    _touches_at_alert,
        "sweep_present": bool((ob.get("sweep_observed") or {}).get("exists")),
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
        "structure_ranging_at_alert":   alert.get("structure_ranging_at_alert"),
        "flip_pending_at_alert":        alert.get("flip_pending_at_alert"),
        "flip_pending_dir_at_alert":    alert.get("flip_pending_dir_at_alert"),
        # S3: leg retracement at alert. leg_extreme_at_alert + leg_extreme_clipped
        # are payload scalars (a-priori extreme since OB formation); retrace_pct is
        # derived above from those + the placed entry. Support columns (extreme /
        # clipped) exist for audit + derivation; only retrace_pct is a screen
        # candidate. None per the edge cases documented at the computation.
        "leg_extreme_at_alert":         leg_extreme_at_alert,
        "leg_retrace_pct_at_alert":     leg_retrace_pct_at_alert,
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
    }


def simulate_h1_only_dual(
    alert: Dict[str, Any],
    pair_conf: Dict[str, Any],
    df_h1: pd.DataFrame,
    risk_usd: float = DEFAULT_RISK_USD,
) -> List[Dict[str, Any]]:
    """Public entry point: simulate the proximal entry for one OB-touch alert.

    Returns [] if proximal levels are invalid (e.g. no TP1 >= 1.5R), else the
    one proximal trade row. (`_dual` is a historical name from the removed 50%
    A/B leg; it now yields a single proximal row.)
    """
    alert_ts = alert["ts"]
    if not isinstance(alert_ts, pd.Timestamp):
        alert_ts = pd.Timestamp(alert_ts)
    if alert_ts.tzinfo is None:
        alert_ts = alert_ts.tz_localize("UTC")

    # T1 + T4 (TRUTH_FIXES_SPEC / _2): ALERT-TIME view of the OB. The replay
    # mutates the shared OB dict after this alert fired (re-fires re-stamp
    # bos_verdict AND the touches_at_alert/fvg_at_alert dict keys; the per-bar
    # loop updates touches/status/fvg), and rows are built after the whole walk.
    # The ALERT PAYLOAD is the one source — bos_verdict (T1), touches_at_alert /
    # fvg_at_alert (T4) all travel as payload scalars snapshotted at the yield.
    # The dict stamps remain only as a legacy fallback for old alerts.
    # TRAP: dict(_ob_live) copies the (possibly re-stamped) *_at_alert KEYS into
    # the view, and _build_row PREFERS those keys — so both key spellings
    # (touches / touches_at_alert, fvg / fvg_at_alert) must be overwritten.
    # One view, built once — never patch individual fields inline downstream.
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
