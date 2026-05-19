"""Bar-by-bar replay through live SMC detection modules.

Walks H1 candles one at a time. At each H1 close, slices H1+M15(+M5) data up to
that bar and calls the live `dealing_range.update_pair` and
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
    df_m15: pd.DataFrame,
    df_m5: Optional[pd.DataFrame],
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

    for h1_ts in h1_in_window.index:
        h1_slice = _slice_up_to(df_h1, h1_ts)
        _assert_no_lookahead(h1_slice, h1_ts, "H1")
        if len(h1_slice) < MIN_WARMUP:
            continue

        m15_slice = _slice_up_to(df_m15, h1_ts) if df_m15 is not None else None
        if m15_slice is not None:
            _assert_no_lookahead(m15_slice, h1_ts, "M15")

        m5_slice = _slice_up_to(df_m5, h1_ts) if df_m5 is not None else None
        if m5_slice is not None:
            _assert_no_lookahead(m5_slice, h1_ts, "M5")

        # --- Step 1: update dealing_range walls + event ring -----------------
        try:
            new_state = dealing_range.update_pair(
                h1_slice, state.dr_state.get(pair_name), pair_conf
            )
            state.dr_state[pair_name] = new_state
        except Exception as e:
            # Detection errors shouldn't abort the whole walk — log and skip bar.
            print(f"  [DR ERROR] {pair_name} @ {h1_ts}: {e}")
            continue

        walls = (new_state or {}).get("walls", {}) if new_state else {}
        events = (new_state or {}).get("event_ring", []) if new_state else []

        # --- Step 2: detect OBs ----------------------------------------------
        try:
            obs_result = smc_radar.detect_smc_radar(
                h1_slice,
                pair_type=pair_type,
                events=events,
                walls=walls,
                pair_name=pair_name,
            )
        except Exception as e:
            print(f"  [RADAR ERROR] {pair_name} @ {h1_ts}: {e}")
            continue

        if not obs_result:
            continue
        # detect_smc_radar can return a dict, list, or tuple depending on version
        # Normalise to list of OB dicts.
        obs = _normalize_obs_result(obs_result)
        if not obs:
            continue

        current_price = float(h1_slice["Close"].iloc[-1])

        # Drop mitigated OBs from active list.
        kept = []
        for ob in state.active_obs[pair_name]:
            mitigated, reason = _is_ob_mitigated_replay(
                ob.get("direction"), float(ob["distal_line"]),
                float(ob["proximal_line"]), h1_slice, ob.get("ob_timestamp")
            )
            if mitigated:
                continue
            kept.append(ob)
        state.active_obs[pair_name] = kept

        # Merge newly-detected OBs (by ob_timestamp identity).
        existing_ts = {o.get("ob_timestamp") for o in state.active_obs[pair_name]}
        for ob in obs:
            if ob.get("ob_timestamp") in existing_ts:
                continue
            state.active_obs[pair_name].append(ob)
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
            if distance > prox_cap:
                continue
            yield {
                "kind": "alert",
                "pair": pair_name,
                "ts": h1_ts,
                "current_price": current_price,
                "h1_atr": h1_atr,
                "ob": ob,
                "walls": walls,
                "m15_slice_end": m15_slice.index[-1] if m15_slice is not None and not m15_slice.empty else None,
            }
            state.alerted_zones[pair_name].add(zone_id)


def _normalize_obs_result(result: Any) -> List[Dict[str, Any]]:
    """detect_smc_radar's return signature varies. Normalise to OB list."""
    if result is None:
        return []
    if isinstance(result, dict):
        # Common shapes: {"obs": [...]} or {"primary": ob, "alternative": ob}
        if "obs" in result and isinstance(result["obs"], list):
            return result["obs"]
        obs = []
        for k in ("primary", "alternative", "ob1", "ob2"):
            v = result.get(k)
            if isinstance(v, dict) and v.get("proximal_line"):
                obs.append(v)
        if obs:
            return obs
        # Single OB dict?
        if result.get("proximal_line"):
            return [result]
        return []
    if isinstance(result, list):
        return [o for o in result if isinstance(o, dict) and o.get("proximal_line")]
    if isinstance(result, tuple):
        # e.g. (obs_list, diagnostics)
        for item in result:
            if isinstance(item, list):
                return [o for o in item if isinstance(o, dict) and o.get("proximal_line")]
    return []
