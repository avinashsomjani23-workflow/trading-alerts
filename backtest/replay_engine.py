"""Bar-by-bar H1 replay through live SMC detection modules.

Walks H1 candles one at a time. At each H1 close, slices H1 data up to that
bar and calls the live `dealing_range.update_pair` and
`smc_radar.detect_smc_radar` functions to get OBs that would have existed at
that moment.

Hard rule: every slice asserted to end at or before the replay timestamp.
Lookahead = bug.

State (dealing_range walls, event ring) is kept in-memory per pair and
threaded through update_pair calls — never written to live JSON.
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

import dealing_range          # live module — read-only use
import smc_radar              # live module — read-only use
import smc_detector           # live module — read-only use

from backtest.run_logger import log_event


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
        # pair_name -> set of zone IDs we've already emitted an alert for
        # (prevents the same OB triggering an alert on every H1 close)
        self.alerted_zones: Dict[str, set] = {}


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


def _slice_up_to(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """Return df rows with index <= ts. Empty df if none."""
    if df is None or df.empty:
        return pd.DataFrame()
    return df.loc[:ts]


def _is_ob_mitigated_replay(direction: str, distal: float, proximal: float,
                            df_h1_slice: pd.DataFrame,
                            ob_ts: Optional[str]) -> Tuple[bool, str]:
    """Wrap live mitigation logic; returns (mitigated, reason)."""
    if not ob_ts:
        return False, ""
    try:
        ob_idx, found = smc_detector.locate_ob_candle_idx(df_h1_slice, ob_ts)
        if not found:
            return False, ""
        mitigated, reason, _ = smc_detector.is_ob_mitigated_phase1(
            direction, distal, proximal, df_h1_slice, start_idx=ob_idx + 1
        )
        return bool(mitigated), reason or ""
    except Exception as e:
        return False, f"mitigation_check_error: {e}"


def replay_pair(
    pair_conf: Dict[str, Any],
    df_h1: pd.DataFrame,
    state: ReplayState,
    walk_start_ts: pd.Timestamp,
    walk_end_ts: pd.Timestamp,
) -> Iterator[Dict[str, Any]]:
    """Walk H1 bars in [walk_start_ts, walk_end_ts] and yield events.

    Yields dicts of two shapes:
      {"kind": "ob_seen", "pair": ..., "ts": ..., "ob": {...}}      — diagnostic
      {"kind": "alert",   "pair": ..., "ts": ..., "ob": {...}}      — would-be alert

    The caller (run_backtest) decides whether to feed an "alert" to the trade
    simulator.
    """
    pair_name = pair_conf["name"]
    pair_type = pair_conf["pair_type"]

    if df_h1 is None or df_h1.empty:
        return

    # Walk only H1 bars in the requested window.
    h1_in_window = df_h1.loc[walk_start_ts:walk_end_ts]
    if h1_in_window.empty:
        return

    state.dr_state.setdefault(pair_name, None)
    state.active_obs.setdefault(pair_name, [])
    state.alerted_zones.setdefault(pair_name, set())

    # Need >= 50 H1 bars of history before the walk start for reliable structure.
    MIN_WARMUP = 50

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
        "prox_checks": 0,
        "closest_dist_atr_seen": None,  # min (distance / h1_atr) ever observed
    }

    for h1_ts in h1_in_window.index:
        diag["bars_walked"] += 1
        h1_slice = _slice_up_to(df_h1, h1_ts)
        _assert_no_lookahead(h1_slice, h1_ts, "H1")
        if len(h1_slice) < MIN_WARMUP:
            diag["warmup_skipped"] += 1
            continue

        # --- Step 1: update dealing_range walls + event ring -----------------
        try:
            new_state = dealing_range.update_pair(
                h1_slice, state.dr_state.get(pair_name), pair_conf
            )
            state.dr_state[pair_name] = new_state
        except Exception as e:
            # Detection errors shouldn't abort the whole walk — log and skip bar.
            diag["dr_errors"] += 1
            log_event("dr_error", level="error", echo=(diag["dr_errors"] <= 3),
                      pair=pair_name, ts=str(h1_ts),
                      error=f"{type(e).__name__}: {e}")
            continue

        # dealing_range.update_pair returns the walls state dict at the top
        # level (ceiling_price, floor_price, trend, events, ...). Live smc_radar
        # passes the whole state as `walls=` and reads events from state['events'].
        # Earlier code looked for non-existent sub-keys 'walls' and 'event_ring'
        # which silently passed empty events → 0 OBs → 0 alerts on every run.
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
                )
        except Exception as e:
            diag["radar_errors"] += 1
            log_event("radar_error", level="error", echo=(diag["radar_errors"] <= 3),
                      pair=pair_name, ts=str(h1_ts),
                      error=f"{type(e).__name__}: {e}")
            continue

        if not obs_result:
            continue
        # detect_smc_radar can return a dict, list, or tuple depending on version
        # Normalise to list of OB dicts.
        obs = _normalize_obs_result(obs_result)
        if not obs:
            continue
        diag["bars_with_obs_returned"] += 1

        current_price = float(h1_slice["Close"].iloc[-1])

        # Drop mitigated OBs from active list. Yield a diagnostic event so the
        # zone register knows which OBs died and why (vs being alerted/traded).
        kept = []
        for ob in state.active_obs[pair_name]:
            mitigated, mit_reason = _is_ob_mitigated_replay(
                ob.get("direction"), float(ob["distal_line"]),
                float(ob["proximal_line"]), h1_slice, ob.get("ob_timestamp")
            )
            if mitigated:
                yield {
                    "kind": "ob_mitigated",
                    "pair": pair_name,
                    "ts": h1_ts,
                    "ob": ob,
                    "reason": mit_reason or "mitigated",
                }
                continue
            kept.append(ob)
        state.active_obs[pair_name] = kept

        # Merge newly-detected OBs (by ob_timestamp identity).
        existing_ts = {o.get("ob_timestamp") for o in state.active_obs[pair_name]}
        for ob in obs:
            if ob.get("ob_timestamp") in existing_ts:
                continue
            state.active_obs[pair_name].append(ob)
            diag["new_obs_added"] += 1
            yield {
                "kind": "ob_seen",
                "pair": pair_name,
                "ts": h1_ts,
                "current_price": current_price,
                "ob": ob,
            }

        # --- Step 3: emit alerts for any active OB within proximity ----------
        h1_atr = smc_detector.compute_atr(h1_slice)
        if not h1_atr:
            continue
        prox_cap = pair_conf["atr_multiplier"] * h1_atr

        for ob in state.active_obs[pair_name]:
            zone_id = ob.get("ob_timestamp") or f"{ob.get('direction')}_{ob.get('proximal_line')}"
            if zone_id in state.alerted_zones[pair_name]:
                continue
            proximal = float(ob["proximal_line"])
            distance = abs(current_price - proximal)
            diag["prox_checks"] += 1
            if h1_atr > 0:
                dist_in_atr = distance / h1_atr
                if (diag["closest_dist_atr_seen"] is None
                        or dist_in_atr < diag["closest_dist_atr_seen"]):
                    diag["closest_dist_atr_seen"] = dist_in_atr
            if distance > prox_cap:
                continue
            diag["alerts_emitted"] += 1
            yield {
                "kind": "alert",
                "pair": pair_name,
                "ts": h1_ts,
                "current_price": current_price,
                "h1_atr": h1_atr,
                "ob": ob,
                "walls": walls,
            }
            state.alerted_zones[pair_name].add(zone_id)

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
        f"alerts={diag['alerts_emitted']}"
    )


def _normalize_obs_result(result: Any) -> List[Dict[str, Any]]:
    """detect_smc_radar's return signature — normalise to OB list.

    Live return shape (confirmed from source):
      {"current_price": float,
       "active_unmitigated_obs": [ob, ...],
       "ob_build_diagnostics": [...]}
    """
    if result is None:
        return []
    if isinstance(result, dict):
        # Primary shape used by live smc_radar.detect_smc_radar
        if "active_unmitigated_obs" in result:
            obs = result["active_unmitigated_obs"]
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
