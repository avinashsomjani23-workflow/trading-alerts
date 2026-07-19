"""Bar-by-bar H1 replay through live SMC detection modules.

Walks H1 candles one at a time. At each H1 close, slices H1 data up to that
bar and calls the live `smc_radar.compute_pair_walls` and
`smc_radar.detect_smc_radar` functions to get OBs that would have existed at
that moment.

Hard rule: every slice asserted to end at or before the replay timestamp.
Lookahead = bug.

State (structure walls, event ring) is recomputed per pair from each H1 slice
(compute_structure is stateless) and kept in-memory — never written to live JSON.
"""

from __future__ import annotations

import sys
import os
import io
import contextlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Iterator, Tuple

import pandas as pd

# Add repo root so we can import live modules without altering them.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import smc_radar              # live module — read-only use (compute_pair_walls + detect_smc_radar)
import smc_detector           # live module — read-only use
import displacement_leg       # live module — shared displacement-leg extreme + ER core
import setup_liq              # live module — setup-liquidity reads (Read 3.2 leg-extreme-was-a-sweep)

from backtest.run_logger import log_event
from backtest.scanlog import get_active as _scanlog


class ReplayState:
    """Per-pair in-memory state across the walk.

    Mirrors what the live system stores in `state/structure_state.json` and
    `active_obs.json`, but stays in RAM. Never touches live JSON files.
    """
    def __init__(self):
        # pair_name -> dealing_range state dict
        self.dr_state: Dict[str, Dict[str, Any]] = {}
        # pair_name -> list of currently-active OBs (matches live active_obs)
        self.active_obs: Dict[str, List[Dict[str, Any]]] = {}
        # pair_name -> {zone_id -> per-OB state dict}
        # Per-OB state shape:
        #   {"state": "armed" | "cooling",
        #    "fire_count": int,             # how many times this OB has fired
        #    "last_fire_ts": pd.Timestamp}  # timestamp of most recent fire
        # State machine:
        #   armed   -> fires when wick within prox_cap × ATR of proximal
        #              -> moves to cooling, emits ONE alert (= one limit
        #              order with proximal + 50% scenarios)
        #   cooling -> re-arms when wick clears (prox_cap + 1) × ATR from
        #              proximal (price has meaningfully moved away)
        # Mitigation (handled separately): distal wick-touch OR 3rd proximal
        # touch kills the OB permanently. No "exhausted after fill" flag —
        # a vet re-trades clean un-mitigated zones on legitimate re-approach.
        self.ob_alert_state: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _assert_no_lookahead(slice_df: pd.DataFrame, replay_ts: pd.Timestamp,
                         tag: str) -> None:
    """Hard guard: slice must not contain bars after replay_ts."""
    if slice_df is None or slice_df.empty:
        return
    last_ts = slice_df.index[-1]
    if last_ts > replay_ts:
        raise AssertionError(
            f"LOOKAHEAD VIOLATION [{tag}]: slice last_ts={last_ts} > replay_ts={replay_ts}"
        )


def _slice_closed_before(df: pd.DataFrame, wall_clock_ts: pd.Timestamp
                         ) -> pd.DataFrame:
    """Return df rows whose bar OPENED strictly before wall_clock_ts.

    H1 bars (MT5/FundingPips feed) are open-timestamped: a bar indexed `12:00`
    covers 12:00→13:00 and is only KNOWN at 13:00 when it closes. So at
    wall-clock moment T, the bars that have actually closed are those with
    index < T.

    Example: wall_clock_ts=13:00 -> includes 12:00 bar (closed at 13:00),
    excludes 13:00 bar (still forming, won't close until 14:00).
    """
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df.index < wall_clock_ts]


def _is_ob_mitigated_replay(direction: str, distal: float, proximal: float,
                            df_h1_slice: pd.DataFrame,
                            anchor_ts: Optional[str],
                            distal_mode: str = "close") -> Tuple[bool, str, int]:
    """Wrap live mitigation logic; returns (mitigated, reason, touches).

    Window = from the candle AFTER the structural-event (BOS/CHoCH) candle,
    matching live Phase 1 (smc_radar mitigation + determine_drop_reason) and
    Phase 2's gate. `anchor_ts` MUST be the event candle ts (bos_timestamp),
    NOT the OB ts: candles between the OB and the break are the impulse leg
    that BUILT the zone, not tests of it. Starting at ob_idx+1 let that leg
    count phantom touches / distal hits and silently disagree with live.

    `distal_mode` flows from the per-instrument config (resolve_distal_mode) so
    the backtest applies the SAME distal rule as live Phase 1/2.

    `touches` is passed through (was discarded) so the replay can stamp the
    stored OB's touch count / status every bar — see Fix 3a.
    """
    if not anchor_ts:
        return False, "", 0
    try:
        anchor_idx, found = smc_detector.locate_ob_candle_idx(df_h1_slice, anchor_ts)
        if not found:
            # FIX 1 companion: with the 150-bar clamp a stored OB can outlive its
            # BOS candle (OB age cap 15d >> ~6 trading days of slice). If the
            # anchor is EARLIER than the slice's first bar, every visible bar is
            # post-event -> scan the whole slice (start_idx = -1 + 1 = 0). Any
            # other unresolved anchor is a data problem -> keep the safe
            # "not mitigated" answer so we never false-drop a live zone.
            try:
                _a = pd.Timestamp(anchor_ts)
                if _a.tzinfo is None:
                    _a = _a.tz_localize("UTC")
                if len(df_h1_slice) and _a < df_h1_slice.index[0]:
                    anchor_idx, found = -1, True
            except Exception:
                pass
        if not found:
            return False, "", 0
        mitigated, reason, touches = smc_detector.is_ob_mitigated_phase1(
            direction, distal, proximal, df_h1_slice, start_idx=anchor_idx + 1,
            distal_mode=distal_mode,
        )
        return bool(mitigated), reason or "", int(touches or 0)
    except Exception as e:
        return False, f"mitigation_check_error: {e}", 0


def replay_pair(
    pair_conf: Dict[str, Any],
    df_h1: pd.DataFrame,
    state: ReplayState,
    walk_start_ts: pd.Timestamp,
    walk_end_ts: pd.Timestamp,
    detection_bars: Optional[int] = smc_radar.LIVE_DETECTION_BARS,
) -> Iterator[Dict[str, Any]]:
    """Walk H1 bars in [walk_start_ts, walk_end_ts] and yield events.

    Yields dicts of two shapes:
      {"kind": "ob_seen", "pair": ..., "ts": ..., "ob": {...}}      — diagnostic
      {"kind": "alert",   "pair": ..., "ts": ..., "ob": {...}}      — would-be alert

    The caller (run_backtest) decides whether to feed an "alert" to the trade
    simulator.

    `detection_bars` (FIX 1): clamp each detection slice to the last N closed
    H1 bars — live's exact window (smc_radar.LIVE_DETECTION_BARS). Default =
    live parity. `None` = legacy UNCLAMPED behaviour (full-history slice, old
    50-bar warmup, no slice-length assert); the runner NEVER sets this in normal
    runs — it exists solely for the validation A/B (arm A). See DETECTION_FIXES_SPEC.
    """
    pair_name = pair_conf["name"]
    pair_type = pair_conf["pair_type"]

    if df_h1 is None or df_h1.empty:
        return

    _clamped = detection_bars is not None

    # Walk only H1 bars in the requested window. Each h1_ts in this index is
    # the OPEN time of a bar; we treat it as the wall-clock moment when the
    # hourly P1+P2 cron would fire. At that moment, the most recently CLOSED
    # bar is the one indexed (h1_ts - 1h), because the bar indexed h1_ts has
    # only just begun.
    h1_in_window = df_h1.loc[walk_start_ts:walk_end_ts]
    if h1_in_window.empty:
        return

    state.dr_state.setdefault(pair_name, None)
    state.active_obs.setdefault(pair_name, [])
    state.ob_alert_state.setdefault(pair_name, {})

    # Warmup rule (FIX 1). Clamped mode: live ALWAYS runs on a full
    # LIVE_DETECTION_BARS frame, so a bar with fewer bars of history cannot
    # reproduce live and is SKIPPED, not approximated. Legacy (unclamped) mode
    # keeps the old 50-bar floor so arm A of the A/B behaves as before.
    _min_warmup = detection_bars if _clamped else 50

    # Re-arm threshold: price wick/close must clear (prox_cap + REARM_EXTRA_ATR)
    # × ATR from proximal before a fired OB can fire again. Stops one OB from
    # spamming alerts every hour while price wiggles inside the proximity band.
    REARM_EXTRA_ATR = 1.0

    # Hard age cap mirrors live OB_MAX_AGE_DAYS (smc_radar.py:100). Drop
    # stale OBs from the slate even if not formally mitigated.
    OB_MAX_AGE_DAYS = 15

    # Diagnostic counters — printed at end of walk to show where the funnel collapses.
    diag = {
        "bars_walked": 0,
        "warmup_skipped": 0,
        "dr_errors": 0,
        "radar_errors": 0,
        "bars_with_events": 0,
        "bars_with_obs_returned": 0,
        "new_obs_added": 0,
        "alerts_emitted": 0,
        "alerts_resuppressed": 0,   # blocked because BOS was on just-closed bar
        "alerts_rearmed": 0,
        "prox_checks": 0,
        "obs_aged_out": 0,
        "closest_dist_atr_seen": None,  # min (distance / h1_atr) ever observed
    }

    # Heartbeat contract (SPEC Â§2.2 / G2): declare how many bars the walk will
    # visit so the health layer can prove every one produced a scan record.
    _scanlog().declare_walk(pair_name, len(h1_in_window.index))

    for h1_ts in h1_in_window.index:
        diag["bars_walked"] += 1
        # P1 sees only CLOSED bars. The bar opened at h1_ts is in progress.
        h1_slice = _slice_closed_before(df_h1, h1_ts)
        # FIX 1: clamp to live's window (the last detection_bars closed bars).
        # tail() cannot introduce future bars, so the lookahead guard stays valid.
        if _clamped:
            h1_slice = h1_slice.tail(detection_bars)
        _assert_no_lookahead(h1_slice, h1_ts, "H1")
        if len(h1_slice) < _min_warmup:
            diag["warmup_skipped"] += 1
            _scanlog().scan(pair=pair_name, ts=h1_ts, index=df_h1.index,
                            outcome="WARMUP_SKIP", n_bars_in_slice=len(h1_slice),
                            bt_conditions=["WARMUP_SKIP"])
            _scanlog().condition("WARMUP_SKIP", pair=pair_name)
            continue
        # FIX 1 regression guard (clamped mode only): every scanned bar MUST run
        # on exactly the live window. A mismatch = silent divergence -> abort loud.
        if _clamped and len(h1_slice) != detection_bars:
            _scanlog().condition("SLICE_WINDOW_MISMATCH", pair=pair_name,
                                 ts=str(h1_ts), n_bars=len(h1_slice),
                                 expected=detection_bars)
            raise AssertionError(
                f"slice/window mismatch: {len(h1_slice)} != {detection_bars} "
                f"at {pair_name} {h1_ts}")

        # The "just-closed" bar: the most recent bar whose open is < h1_ts.
        # All P2 proximity decisions read this bar's high/low/close.
        just_closed = h1_slice.iloc[-1]
        just_closed_ts = h1_slice.index[-1]

        # --- Step 1: build structure + dealing-range state -------------------
        # Mirror live Phase 1 exactly: smc_radar.compute_pair_walls runs the H4
        # range + v2 structure engine and returns the same `walls` dict the live
        # scan persists. compute_structure is pure (recomputes from the whole
        # slice each bar), so there is no incremental state to thread — dr_state
        # just holds the latest snapshot for parity with live.
        try:
            new_state = smc_radar.compute_pair_walls(h1_slice, pair_name)
            state.dr_state[pair_name] = new_state
        except Exception as e:
            # Detection errors shouldn't abort the whole walk — log and skip bar.
            diag["dr_errors"] += 1
            log_event("dr_error", level="error", echo=(diag["dr_errors"] <= 3),
                      pair=pair_name, ts=str(h1_ts),
                      error=f"{type(e).__name__}: {e}")
            _scanlog().scan(pair=pair_name, ts=h1_ts, index=df_h1.index,
                            outcome="DEGENERATE_SKIP",
                            bt_conditions=["DEGENERATE_BAR"])
            _scanlog().condition("DEGENERATE_BAR", pair=pair_name, ts=str(h1_ts),
                                 where="compute_pair_walls",
                                 error=f"{type(e).__name__}: {e}")
            continue

        # `walls` carries ceiling_price, floor_price, trend, events, swings, ...
        # Live smc_radar passes the whole dict as `walls=` and reads the event
        # ring from walls['events'].
        walls = new_state or {}
        events = walls.get("events", [])
        if events:
            diag["bars_with_events"] += 1

        # --- Step 2: detect OBs ----------------------------------------------
        # Suppress detect_smc_radar's per-event [OB-DROP] stdout chatter during
        # backtest replay. Live keeps these prints; backtest gets the same diag
        # info in the returned `ob_build_diagnostics` dict (currently unused
        # downstream — no consumer reads them today). Pure I/O reduction, no
        # logic change to the live function.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                obs_result = smc_radar.detect_smc_radar(
                    h1_slice,
                    pair_type=pair_type,
                    events=events,
                    walls=walls,
                    pair_name=pair_name,
                    cap_zones=False,   # backtest sees EVERY OB (no 2-OB live cap)
                )
        except Exception as e:
            diag["radar_errors"] += 1
            log_event("radar_error", level="error", echo=(diag["radar_errors"] <= 3),
                      pair=pair_name, ts=str(h1_ts),
                      error=f"{type(e).__name__}: {e}")
            _scanlog().scan(pair=pair_name, ts=h1_ts, index=df_h1.index,
                            outcome="DEGENERATE_SKIP",
                            bt_conditions=["DEGENERATE_BAR"])
            _scanlog().condition("DEGENERATE_BAR", pair=pair_name, ts=str(h1_ts),
                                 where="detect_smc_radar",
                                 error=f"{type(e).__name__}: {e}")
            continue

        if not obs_result:
            _scanlog().scan(pair=pair_name, ts=h1_ts, index=df_h1.index,
                            outcome="NO_ZONE", n_active_zones=0)
            continue
        # detect_smc_radar can return a dict, list, or tuple depending on version
        # Normalise to list of OB dicts.
        if not isinstance(obs_result, dict):
            _scanlog().condition("NORMALIZE_NONDICT", pair=pair_name,
                                 ts=str(h1_ts), shape=type(obs_result).__name__)
        obs = _normalize_obs_result(obs_result)
        if not obs:
            _scanlog().scan(pair=pair_name, ts=h1_ts, index=df_h1.index,
                            outcome="NO_ZONE", n_active_zones=0)
            continue
        diag["bars_with_obs_returned"] += 1

        current_price = float(just_closed["Close"])
        just_closed_high = float(just_closed["High"])
        just_closed_low = float(just_closed["Low"])

        # Drop mitigated OBs from active list. Distal rule is per-instrument
        # via distal_invalidation_mode (default 'close'; 'wick' tunable) — same
        # knob the live engines read, so backtest P&L reflects live behaviour.
        # Yield a diagnostic event so the zone register knows which OBs died.
        _distal_mode = smc_detector.resolve_distal_mode(pair_conf)
        kept = []
        for ob in state.active_obs[pair_name]:
            # Anchor on the BOS/CHoCH candle (live parity); fall back to the OB
            # candle only for legacy OBs missing bos_timestamp.
            _anchor_ts = ob.get("bos_timestamp") or ob.get("ob_timestamp")
            mitigated, mit_reason, touches = _is_ob_mitigated_replay(
                ob.get("direction"), float(ob["distal_line"]),
                float(ob["proximal_line"]), h1_slice, _anchor_ts,
                distal_mode=_distal_mode,
            )
            if mitigated:
                zone_id = ob.get("ob_timestamp") or f"{ob.get('direction')}_{ob.get('proximal_line')}"
                state.ob_alert_state[pair_name].pop(zone_id, None)
                _scanlog().event("ob_mitigated", pair=pair_name, ts=str(h1_ts),
                                 ob_timestamp=ob.get("ob_timestamp"),
                                 reason=mit_reason or "mitigated")
                yield {
                    "kind": "ob_mitigated",
                    "pair": pair_name,
                    "ts": h1_ts,
                    "ob": ob,
                    "reason": mit_reason or "mitigated",
                }
                continue
            # FIX 3a: refresh mutable touch state every bar (live refreshes zone
            # state every scan). Label format mirrors live smc_radar.py exactly.
            ob["touches"] = int(touches)
            ob["status"] = "Pristine" if touches == 0 else f"Tested ({touches}x proximal)"
            # FIX 3b: one-time break_quality re-grade once the option-B window
            # (break candle + next candle) is complete. The two candle bodies
            # never change afterwards, so grading ONCE when the window exists is
            # the faithful measure (the OB may have been built on the break's
            # edge bar, before the next candle existed).
            if not ob.get("_bq_regraded"):
                _a_idx, _a_found = smc_detector.locate_ob_candle_idx(
                    h1_slice, ob.get("bos_timestamp") or "")
                if _a_found and _a_idx + 1 < len(h1_slice):
                    ob["break_quality"] = smc_detector.compute_break_quality(
                        h1_slice, _a_idx, ob.get("bos_swing_price"),
                        ob.get("direction"), smc_detector.compute_atr(h1_slice),
                        event_type=ob.get("bos_tag", "BOS"))
                    ob["_bq_regraded"] = True
            kept.append(ob)
        state.active_obs[pair_name] = kept

        # Age cap — drop OBs older than OB_MAX_AGE_DAYS (mirrors live slate).
        kept_after_age = []
        for ob in state.active_obs[pair_name]:
            ob_ts_iso = ob.get("ob_timestamp")
            if ob_ts_iso:
                try:
                    ob_ts = pd.Timestamp(ob_ts_iso)
                    if ob_ts.tzinfo is None:
                        ob_ts = ob_ts.tz_localize("UTC")
                    age_days = (h1_ts - ob_ts).total_seconds() / 86400.0
                    if age_days > OB_MAX_AGE_DAYS:
                        diag["obs_aged_out"] += 1
                        zone_id = ob_ts_iso
                        state.ob_alert_state[pair_name].pop(zone_id, None)
                        _scanlog().event("ob_aged_out", pair=pair_name,
                                         ts=str(h1_ts), ob_timestamp=ob_ts_iso,
                                         age_days=round(age_days, 2))
                        yield {
                            "kind": "ob_mitigated",
                            "pair": pair_name,
                            "ts": h1_ts,
                            "ob": ob,
                            "reason": f"aged_out_{age_days:.1f}d",
                        }
                        continue
                except Exception:
                    pass
            kept_after_age.append(ob)
        state.active_obs[pair_name] = kept_after_age

        # Merge newly-detected OBs (by ob_timestamp identity).
        # FIX 3c: a re-surfaced OB is NOT skipped outright — its FVG dict is
        # refreshed from the fresh (point-in-time clean) detection, mirroring
        # live's per-scan zone refresh. Only `fvg` is touched here; touches/
        # status are owned by 3a, break_quality by 3b (one concept, one impl).
        existing_by_ts = {o.get("ob_timestamp"): o for o in state.active_obs[pair_name]}
        for ob in obs:
            match = existing_by_ts.get(ob.get("ob_timestamp"))
            if match is not None:
                match["fvg"] = ob.get("fvg", match.get("fvg"))
                continue
            state.active_obs[pair_name].append(ob)
            zone_id = ob.get("ob_timestamp") or f"{ob.get('direction')}_{ob.get('proximal_line')}"
            # New OB starts armed, ready to fire on a future approach.
            state.ob_alert_state[pair_name][zone_id] = {
                "state": "armed",
                "fire_count": 0,
                "last_fire_ts": None,
            }
            diag["new_obs_added"] += 1
            _scanlog().event("ob_seen", pair=pair_name, ts=str(h1_ts),
                             ob_timestamp=ob.get("ob_timestamp"),
                             bos_timestamp=ob.get("bos_timestamp"),
                             direction=ob.get("direction"),
                             proximal=ob.get("proximal_line"),
                             distal=ob.get("distal_line"))
            yield {
                "kind": "ob_seen",
                "pair": pair_name,
                "ts": h1_ts,
                "current_price": current_price,
                "ob": ob,
            }

        # --- Step 3: P2 alert step. Per OB, run state machine -------------
        # Rules:
        #   * Alert candle (h1_ts) MUST be strictly after the BOS event
        #     candle. If BOS happened on the just-closed bar, no alert this
        #     cycle (wait one more hour). This kills the "trade the BOS
        #     candle" bug from the previous backtest engine.
        #   * Proximity uses wick OR close: short OB checks just-closed bar
        #     high vs proximal; long OB checks just-closed bar low vs
        #     proximal. (For short, "wick reaching up toward proximal" =
        #     high near proximal. For long, "wick reaching down toward
        #     proximal" = low near proximal.)
        #   * State machine: armed -> fires -> cooling; cools back to
        #     armed only after price clears (cap + REARM_EXTRA_ATR) × ATR.
        #     Each fire emits exactly one alert event => one limit-order
        #     pair (proximal + 50% midpoint). No double-counting.
        #   * A fill does NOT exhaust the OB. A clean, un-mitigated zone
        #     re-arms and can fire again on a legitimate re-approach, so one
        #     OB yields multiple trades at rising touch counts. The OB dies
        #     only on mitigation (distal wick-touch OR 3rd proximal touch);
        #     see OBReplayState. There is no exhausted-after-fill flag.
        h1_atr = smc_detector.compute_atr(h1_slice)
        _scanlog().note_post_warmup_bar(pair_name, atr_is_nan=not h1_atr)
        if not h1_atr:
            _scanlog().scan(pair=pair_name, ts=h1_ts, index=df_h1.index,
                            outcome="NAN_ATR_SKIP", atr=None,
                            n_active_zones=len(state.active_obs[pair_name]),
                            bt_conditions=["NAN_ATR_SKIP"])
            _scanlog().condition("NAN_ATR_SKIP", pair=pair_name, ts=str(h1_ts))
            continue
        prox_cap = pair_conf["atr_multiplier"] * h1_atr
        rearm_cap = prox_cap + REARM_EXTRA_ATR * h1_atr

        # --- scanlog per-bar accumulators (behaviour-neutral; SPEC Â§2.2) -----
        # These only OBSERVE the loop below to produce exactly one heartbeat
        # record for this bar after it finishes. They change no branch, value,
        # or yield. Grep `_bt_` to remove all instrumentation in this block.
        _bt_n_zones = len(state.active_obs[pair_name])
        _bt_fired = False
        _bt_any_cooling = False
        _bt_nearest = None        # (dist_atr, ob_ts, direction)
        _bt_trend = (walls or {}).get("trend")

        for ob in state.active_obs[pair_name]:
            zone_id = ob.get("ob_timestamp") or f"{ob.get('direction')}_{ob.get('proximal_line')}"
            ob_state = state.ob_alert_state[pair_name].setdefault(zone_id, {
                "state": "armed", "fire_count": 0, "last_fire_ts": None,
            })

            proximal = float(ob["proximal_line"])
            direction = ob.get("direction")

            # Wick-based distance: for SHORT OB the threat candle is one
            # whose HIGH approaches proximal from BELOW; for LONG OB it's
            # the candle whose LOW approaches proximal from ABOVE.
            if direction == "bullish":  # LONG OB
                wick_distance = max(0.0, just_closed_low - proximal)
            else:                        # SHORT OB
                wick_distance = max(0.0, proximal - just_closed_high)

            diag["prox_checks"] += 1
            if h1_atr > 0:
                dist_in_atr = wick_distance / h1_atr
                if (diag["closest_dist_atr_seen"] is None
                        or dist_in_atr < diag["closest_dist_atr_seen"]):
                    diag["closest_dist_atr_seen"] = dist_in_atr
                # scanlog: track nearest zone this bar (observe-only).
                if _bt_nearest is None or dist_in_atr < _bt_nearest[0]:
                    _bt_nearest = (dist_in_atr, ob.get("ob_timestamp"), direction)

            # Cooling -> armed transition (price has cleared re-arm band)
            if ob_state["state"] == "cooling" and wick_distance > rearm_cap:
                ob_state["state"] = "armed"
                diag["alerts_rearmed"] += 1
                _scanlog().event("re_arm", pair=pair_name, ts=str(h1_ts),
                                 ob_timestamp=ob.get("ob_timestamp"))

            if ob_state["state"] != "armed":
                _bt_any_cooling = True   # scanlog: a zone is waiting to re-arm
                continue
            if wick_distance > prox_cap:
                continue

            # Safety assertion: under the new closed-bar-only detection,
            # bos_ts is always <= just_closed_ts (BOS is in the history P1
            # sees), and alert fires at h1_ts > just_closed_ts. So
            # alert_ts > bos_ts is structurally guaranteed. If this ever
            # fails, detection has leaked a future bar — abort loudly so we
            # don't write wrong data.
            bos_ts_iso = ob.get("bos_timestamp")
            if bos_ts_iso:
                try:
                    bos_ts = pd.Timestamp(bos_ts_iso)
                    if bos_ts.tzinfo is None:
                        bos_ts = bos_ts.tz_localize("UTC")
                    if h1_ts <= bos_ts:
                        diag["alerts_resuppressed"] += 1
                        log_event("alert_lookahead_blocked", level="error",
                                  echo=True, pair=pair_name,
                                  h1_ts=str(h1_ts), bos_ts=str(bos_ts),
                                  ob_ts=ob.get("ob_timestamp"))
                        # scanlog: guard #3 - a future-stamped OB tried to
                        # alert. FAIL condition (auto-promotes to a finding).
                        _scanlog().condition("ALERT_LOOKAHEAD_BLOCKED",
                                             pair=pair_name, h1_ts=str(h1_ts),
                                             bos_ts=str(bos_ts),
                                             ob_ts=ob.get("ob_timestamp"))
                        _scanlog().event("alert_lookahead_blocked",
                                         pair=pair_name, h1_ts=str(h1_ts),
                                         bos_ts=str(bos_ts),
                                         ob_ts=ob.get("ob_timestamp"))
                        continue
                except Exception:
                    pass

            # Fire!
            # Stamp the continuation-drive verdict (holding/fading) at alert time
            # from the in-memory event leg — the SAME shared read live Phase 2 uses
            # (smc_detector.bos_leg_read). Without this the OB carries the default
            # 'holding' and the backtest structure score silently diverges from live
            # for a fading late-continuation. Point-in-time clean: events come from
            # the closed-bar walls slice.
            ob["bos_verdict"] = smc_detector.bos_leg_read(events).get("verdict", "holding")
            # FIX 3d: freeze the alert-time mutable state as scalars. run_backtest
            # and the per-bar loop keep mutating ob["touches"]/ob["fvg"] AFTER the
            # alert; if the trade row is built at close it would log post-alert
            # values. Snapshot here so the row logs state AS OF THE ALERT.
            ob["touches_at_alert"] = int(ob.get("touches") or 0)
            ob["fvg_at_alert"] = dict(ob.get("fvg") or {})
            diag["alerts_emitted"] += 1
            ob_state["state"] = "cooling"
            ob_state["fire_count"] += 1
            ob_state["last_fire_ts"] = h1_ts
            # S2 (STRUCTURE_SIGNALS_SPEC) — snapshot the v2 structure state AS OF
            # THIS ALERT as payload scalars. structure_v2 rides `walls`; read it
            # here at fire time (the OB dict is shared + re-stamped on re-fire, so
            # these must NOT live on `ob`). None only when structure_v2 is missing
            # (degraded walls). Pending dir up/down -> bullish/bearish (same map as
            # smc_detector.py:695). Read BEFORE _bt_align because the alignment
            # derivation now consumes the flip-pending state (Task 1 parity fix).
            _sv2 = walls.get("structure_v2") or {}
            _pend_dir_raw = _sv2.get("choch_pending_dir")
            _pend_dir_map = {"up": "bullish", "down": "bearish"}
            if _sv2:
                _structure_ranging_at_alert = bool(_sv2.get("ranging"))
                _flip_pending_at_alert = bool(_sv2.get("flip_unconfirmed"))
                _flip_pending_dir_at_alert = _pend_dir_map.get(_pend_dir_raw)
            else:
                _structure_ranging_at_alert = None
                _flip_pending_at_alert = None
                _flip_pending_dir_at_alert = None

            # scanlog: record the alert with its causality chain + trend context.
            # trend_alignment comes from the SHARED helper (smc_detector.
            # derive_trend_alignment) — the SAME implementation live Phase 2 uses,
            # so the edge-engine feature means the same thing in both paths. It now
            # DEMOTES a with-trend read to counter_trend/ambiguous while a CHoCH
            # flip is pending (raw `trend` still reads the old direction then). The
            # backtest fires counter-trend by design, so a with/against label is
            # NOT a contradiction. A real contradiction would be an unparseable
            # trend value or a missing direction while an alignment is asserted —
            # those raise TREND_CONTRADICTION (guards the raw trend, not the label).
            _bt_align = smc_detector.derive_trend_alignment(
                direction, _bt_trend,
                bool(_flip_pending_at_alert), _flip_pending_dir_at_alert)
            if _bt_trend not in (None, "bullish", "bearish"):
                _scanlog().condition("TREND_CONTRADICTION", pair=pair_name,
                                     ts=str(h1_ts), trend=str(_bt_trend),
                                     direction=direction)
            _scanlog().event("alert", pair=pair_name, alert_ts=str(h1_ts),
                             ob_timestamp=ob.get("ob_timestamp"),
                             bos_timestamp=ob.get("bos_timestamp"),
                             direction=direction, alert_seq=ob_state["fire_count"],
                             trend=_bt_trend, trend_alignment=_bt_align)
            _bt_fired = True

            # S3 (STRUCTURE_SIGNALS_SPEC / DISPLACEMENT_LEG_BUILD_SPEC) — the
            # displacement leg that gave birth to this OB, graded by the ONE
            # shared core (displacement_leg.compute_leg_extreme_er). The span is
            # [ob_idx, extreme_end_idx] — from the OB candle, THROUGH the break
            # candle, to the leg's STRUCTURAL top (first confirmed opposing swing
            # after the break, or the running extreme if none has confirmed yet).
            # This REPLACES the old unbounded .max()/.min() to the end of the
            # slice, which folded in later unrelated moves (the bug it fixed).
            #   leg_extreme_at_alert = highest High / lowest Low over that span.
            #   leg_er_at_alert      = Kaufman ER over that span's closes.
            # Both are PAYLOAD SCALARS (T1: never stamped on the shared ob dict,
            # which the next re-fire would overwrite). The core takes TIMESTAMPS
            # (ob_timestamp/bos_timestamp) + walls['swings'] and re-resolves
            # ts->idx internally; h1_slice is already closed-bars-only. It never
            # raises. `leg_extreme_clipped` stays informational: True only when
            # the OB predates the slice (extreme unmeasurable -> extreme is None,
            # so the flag never accompanies a clipped guess).
            _ob_ts_raw = ob.get("ob_timestamp")
            _leg_extreme_clipped = None
            if _ob_ts_raw is not None and len(h1_slice) > 0:
                try:
                    _ob_ts_norm = smc_detector.ts_to_utc_instant(_ob_ts_raw)
                    _slice_start = smc_detector.ts_to_utc_instant(h1_slice.index[0])
                    if _ob_ts_norm is not None and _slice_start is not None:
                        _leg_extreme_clipped = bool(_ob_ts_norm < _slice_start)
                except Exception:
                    _leg_extreme_clipped = None
            _leg_extreme_at_alert, _leg_er_at_alert, _leg_extreme_end_idx = \
                displacement_leg.compute_leg_extreme_er(
                    h1_slice,
                    ob.get("ob_timestamp"),
                    ob.get("bos_timestamp"),
                    direction,
                    walls.get("swings"),
                )

            # Read 3.2 (setup_liq / SWING_SWEEP_SPEC): was the leg's OWN terminal
            # extreme itself a sweep of an active swing (the leg ENDED on a
            # stop-run, then reversed — institutional-intent tell)? Computed on
            # the SAME closed-bar h1_slice + the extreme_end_idx the shared core
            # just resolved, so it shares the leg core's point-in-time frame and
            # never re-detects. Payload scalar (T1 re-fire discipline). None when
            # the extreme was unmeasurable. Never raises.
            _leg_extreme_swept = setup_liq.read_legextreme_swept(
                h1_slice, _leg_extreme_at_alert, _leg_extreme_end_idx,
                direction, pair_type, h1_atr)

            yield {
                "kind": "alert",
                "pair": pair_name,
                "ts": h1_ts,
                "current_price": current_price,
                "h1_atr": h1_atr,
                "ob": ob,
                "walls": walls,
                "alert_seq": ob_state["fire_count"],
                "zone_id": zone_id,
                "h1_trend": _bt_trend,
                "trend_alignment": _bt_align,
                # S2 (STRUCTURE_SIGNALS_SPEC): v2 structure state snapshotted at
                # THIS fire as payload scalars (T1 pattern — never stamped on the
                # shared ob dict, which the next re-fire would overwrite).
                "structure_ranging_at_alert": _structure_ranging_at_alert,
                "flip_pending_at_alert": _flip_pending_at_alert,
                "flip_pending_dir_at_alert": _flip_pending_dir_at_alert,
                # S3 (STRUCTURE_SIGNALS_SPEC / DISPLACEMENT_LEG_BUILD_SPEC):
                # structural displacement-leg extreme + Kaufman ER over the span
                # [ob_idx, extreme_end_idx], payload scalars (same re-fire
                # rationale). Both share the exact same span in the same stamp.
                "leg_extreme_at_alert": _leg_extreme_at_alert,
                "leg_er_at_alert": _leg_er_at_alert,
                "leg_extreme_clipped": _leg_extreme_clipped,
                # Read 3.2 (setup_liq): leg-extreme-was-a-sweep, payload scalar
                # (same re-fire rationale — never on the shared ob dict).
                "leg_extreme_swept": _leg_extreme_swept,
                # T1 (TRUTH_FIXES_SPEC): the OB dict is shared and re-stamped on
                # every re-fire; rows are built after the whole walk. Carry the
                # ALERT-TIME verdict as a payload scalar so a multi-fire zone's
                # traded row never logs a later fire's verdict.
                "bos_verdict": ob["bos_verdict"],
                # T4 (TRUTH_FIXES_SPEC_2): same rationale as T1's payload verdict —
                # the dict stamps at :572-573 are overwritten by every re-fire; rows
                # are built post-walk. Carry alert-time touches/fvg as payload
                # scalars. Fresh dict() copy is mandatory — the per-bar loop mutates
                # ob["fvg"] in place.
                "touches_at_alert": int(ob.get("touches") or 0),
                "fvg_at_alert": dict(ob.get("fvg") or {}),
                # Just-closed bar's high/low — simulator uses these to do the
                # SAME-BAR fill check (alert and limit order are
                # instantaneous; if the bar that triggered the alert already
                # tagged proximal/midpoint, the limit fills immediately).
                "alert_bar_high": just_closed_high,
                "alert_bar_low": just_closed_low,
                "alert_bar_ts": just_closed_ts,
            }

        # scanlog: exactly one heartbeat for this fully-processed bar. Outcome
        # priority ALERT > RE_ARM_WAIT > OUT_OF_RANGE > NO_ZONE. This is the
        # record G2 counts; every non-skipped bar lands here.
        if _bt_fired:
            _bt_outcome = "ALERT"
        elif _bt_n_zones == 0:
            _bt_outcome = "NO_ZONE"
        elif _bt_nearest is not None and _bt_nearest[0] <= pair_conf["atr_multiplier"]:
            _bt_outcome = "RE_ARM_WAIT" if _bt_any_cooling else "OUT_OF_RANGE"
        elif _bt_any_cooling:
            _bt_outcome = "RE_ARM_WAIT"
        else:
            _bt_outcome = "OUT_OF_RANGE"
        _scanlog().scan(
            pair=pair_name, ts=h1_ts, index=df_h1.index, outcome=_bt_outcome,
            just_closed_ts=str(just_closed_ts), n_bars_in_slice=len(h1_slice),
            atr=h1_atr, trend=_bt_trend, n_active_zones=_bt_n_zones,
            nearest_zone=({"ob_timestamp": _bt_nearest[1],
                           "direction": _bt_nearest[2],
                           "distance_atr": round(_bt_nearest[0], 4)}
                          if _bt_nearest else None),
            prox_cap_atr=pair_conf["atr_multiplier"],
        )

    closest = diag["closest_dist_atr_seen"]
    closest_str = f"{closest:.2f}×ATR" if closest is not None else "n/a"
    atr_mult = pair_conf.get("atr_multiplier", "?")
    log_event(
        "pair_funnel",
        pair=pair_name,
        bars_walked=diag["bars_walked"],
        warmup_skipped=diag["warmup_skipped"],
        dr_errors=diag["dr_errors"],
        radar_errors=diag["radar_errors"],
        bars_with_events=diag["bars_with_events"],
        bars_with_obs=diag["bars_with_obs_returned"],
        new_obs=diag["new_obs_added"],
        prox_checks=diag["prox_checks"],
        closest_dist_atr=(round(closest, 3) if closest is not None else None),
        atr_multiplier_cap=atr_mult,
        alerts=diag["alerts_emitted"],
        alerts_resuppressed=diag["alerts_resuppressed"],
        alerts_rearmed=diag["alerts_rearmed"],
        obs_aged_out=diag["obs_aged_out"],
    )
    print(
        f"  [DIAG {pair_name}] walked={diag['bars_walked']} "
        f"warmup_skip={diag['warmup_skipped']} "
        f"dr_err={diag['dr_errors']} radar_err={diag['radar_errors']} "
        f"bars_with_events={diag['bars_with_events']} "
        f"bars_with_obs={diag['bars_with_obs_returned']} "
        f"new_obs={diag['new_obs_added']} "
        f"prox_checks={diag['prox_checks']} "
        f"closest={closest_str} (cap={atr_mult}×ATR) "
        f"alerts={diag['alerts_emitted']} "
        f"resuppressed={diag['alerts_resuppressed']} "
        f"rearmed={diag['alerts_rearmed']} "
        f"aged_out={diag['obs_aged_out']}"
    )


def _normalize_obs_result(result: Any) -> List[Dict[str, Any]]:
    """detect_smc_radar's return signature — normalise to OB list.

    Live return shape (confirmed from source):
      {"current_price": float,
       "active_zones": [ob, ...],
       "ob_build_diagnostics": [...]}
    """
    if result is None:
        return []
    if isinstance(result, dict):
        # Primary shape used by live smc_radar.detect_smc_radar
        if "active_zones" in result:
            obs = result["active_zones"]
            if isinstance(obs, list):
                return [o for o in obs if isinstance(o, dict) and o.get("proximal_line")]
        # Fallback shapes (future-proofing)
        for key in ("obs", "active_obs"):
            if key in result and isinstance(result[key], list):
                return [o for o in result[key]
                        if isinstance(o, dict) and o.get("proximal_line")]
        for k in ("primary", "alternative", "ob1", "ob2"):
            v = result.get(k)
            if isinstance(v, dict) and v.get("proximal_line"):
                pass  # collected below
        obs = [result[k] for k in ("primary", "alternative", "ob1", "ob2")
               if isinstance(result.get(k), dict) and result[k].get("proximal_line")]
        if obs:
            return obs
        if result.get("proximal_line"):
            return [result]
        return []
    if isinstance(result, list):
        return [o for o in result if isinstance(o, dict) and o.get("proximal_line")]
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, list):
                return [o for o in item if isinstance(o, dict) and o.get("proximal_line")]
    return []
