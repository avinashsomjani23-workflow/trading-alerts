"""
Review detector — scans Phase 1 state after every H1 close, queues cases
that need vet review.

Strict isolation rules:
  - Reads `state/structure_state.json` (Phase 1's output) only.
  - Writes only to `debug/queue/review_queue.json`.
  - Never imports from dealing_range or smc_radar.
  - Any failure logs and returns; never raises into Phase 1.

Conditions checked (per pair, in order). First match wins for that pair
on this scan — we surface ONE case per pair per H1 close so the queue
stays signal, not noise:

  1. RUNAWAY      — price has run > runaway_dr_mult * width from DR mid
                    with no internal CHoCH in the current leg.
  2. STALE_ANCHOR — the trend-side OR opposite confirmed wall has not been
                    challenged for stale_anchor_bars H1 bars
                    (challenge = wick within stale_anchor_atr_mult * ATR).
  3. ORPHAN_BOS   — last event is BOS Major, fired > orphan_bos_bars
                    H1 bars ago, AND no newer event has appeared.
  4. PLACEHOLDER  — either DR side is still placeholder after
                    placeholder_bars H1 bars since the last event.

Dedupe: if the most recent queue entry for this pair has the SAME
condition AND is still `pending`, no new entry is appended. Operator
seeing the same anchor stale for 60 bars doesn't need 12 entries.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfg
from . import queue_io

logger = logging.getLogger(__name__)

# Read-only path to Phase 1 state. Hard-coded (matches dealing_range.STATE_PATH)
# rather than imported, so this module never reaches back into Phase 1.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE1_STATE_PATH = os.path.join(_PROJECT_ROOT, "state", "structure_state.json")


def _load_phase1_state() -> Dict[str, Any]:
    try:
        with open(PHASE1_STATE_PATH, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("review_detector: cannot load Phase 1 state: %s", e)
        return {}


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # fromisoformat handles "+00:00" suffix on 3.7+
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _bars_between(a_iso: Optional[str], b_iso: Optional[str]) -> Optional[int]:
    """Approximate H1 bar count between two ISO timestamps. Returns int hours.

    H1 bars are not strictly 1h apart on weekends/gaps, but for staleness
    detection ~hour-count is fine — vet would think in trading days, and
    weekend gaps inside the count actually make the metric more conservative
    (a 48-bar stale check over a weekend is really ~36 trading bars).
    """
    a = _parse_iso(a_iso)
    b = _parse_iso(b_iso)
    if a is None or b is None:
        return None
    delta = b - a
    return max(0, int(delta.total_seconds() // 3600))


def _has_internal_choch_in_leg(events: List[Dict[str, Any]],
                                last_major_event_ts: Optional[str]) -> bool:
    """Return True if any CHoCH (any tier) appears after the most recent
    Major BOS in the events ring. Used by RUNAWAY: if price has run far
    but a CHoCH has formed since the BOS, the system is still tracking.
    """
    if not events or not last_major_event_ts:
        return False
    anchor = _parse_iso(last_major_event_ts)
    if anchor is None:
        return False
    for ev in events:
        if ev.get("type") != "CHoCH":
            continue
        et = _parse_iso(ev.get("candle_ts"))
        if et is None:
            continue
        if et > anchor:
            return True
    return False


def _check_pair(pair: str, state: Dict[str, Any],
                current_price: Optional[float], h1_atr: Optional[float],
                scan_now_iso: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return (condition, details) tuple for first matching condition, else None."""
    thresholds = cfg.get_thresholds(pair)

    ceiling = state.get("ceiling_price")
    floor   = state.get("floor_price")
    ceiling_ts = state.get("ceiling_ts")
    floor_ts   = state.get("floor_ts")
    ceiling_ph = bool(state.get("ceiling_is_placeholder", True))
    floor_ph   = bool(state.get("floor_is_placeholder", True))
    last_event_ts   = state.get("last_event_ts")
    last_event_type = state.get("last_event_type")
    last_event_tier = state.get("last_event_tier")
    last_scanned_ts = state.get("last_scanned_ts") or scan_now_iso
    events          = state.get("events") or []

    if ceiling is None or floor is None or ceiling <= floor:
        return None  # Degenerate state — Phase 1 will rebuild; nothing to flag.

    width = float(ceiling) - float(floor)
    mid   = (float(ceiling) + float(floor)) / 2.0

    # --- 1. RUNAWAY -------------------------------------------------------
    # Price far outside DR with no internal CHoCH in the current leg.
    if current_price is not None and width > 0:
        distance = abs(float(current_price) - mid)
        if distance > thresholds["runaway_dr_mult"] * width:
            anchor_ts = state.get("trend_start_ts") or last_event_ts
            if not _has_internal_choch_in_leg(events, anchor_ts):
                return ("RUNAWAY", {
                    "current_price": float(current_price),
                    "dr_mid":        mid,
                    "dr_width":      width,
                    "distance_mult": distance / width if width else None,
                    "threshold":     thresholds["runaway_dr_mult"],
                })

    # --- 2. STALE_ANCHOR --------------------------------------------------
    # A confirmed wall hasn't been tested for too long. ATR-relative.
    # We can only assess "tested" precisely if we re-read price history,
    # which would couple us to Phase 1. Proxy: time since wall ts vs.
    # last_scanned_ts. If the wall was set N bars ago and is still confirmed
    # (not placeholder) and nothing has moved it, it's been untouched for
    # at least the entire incremental scan window.
    for side, price, ts, is_ph in (
        ("ceiling", ceiling, ceiling_ts, ceiling_ph),
        ("floor",   floor,   floor_ts,   floor_ph),
    ):
        if is_ph:
            continue  # Placeholder is checked by PLACEHOLDER condition.
        bars = _bars_between(ts, last_scanned_ts)
        if bars is None:
            continue
        if bars >= thresholds["stale_anchor_bars"]:
            return ("STALE_ANCHOR", {
                "side":             side,
                "wall_price":       float(price),
                "wall_ts":          ts,
                "bars_since_set":   bars,
                "threshold_bars":   thresholds["stale_anchor_bars"],
                "atr":              h1_atr,
            })

    # --- 3. ORPHAN_BOS ----------------------------------------------------
    if last_event_type == "BOS" and last_event_tier == "Major":
        bars = _bars_between(last_event_ts, last_scanned_ts)
        if bars is not None and bars >= thresholds["orphan_bos_bars"]:
            # No newer event = events ring's latest matches last_event_ts.
            latest_in_ring = events[-1].get("candle_ts") if events else None
            if latest_in_ring == last_event_ts or len(events) == 0:
                return ("ORPHAN_BOS", {
                    "last_event_ts":  last_event_ts,
                    "bars_since":     bars,
                    "threshold_bars": thresholds["orphan_bos_bars"],
                })

    # --- 4. PLACEHOLDER ---------------------------------------------------
    if ceiling_ph or floor_ph:
        ref_ts = last_event_ts or state.get("trend_start_ts")
        bars = _bars_between(ref_ts, last_scanned_ts)
        if bars is not None and bars >= thresholds["placeholder_bars"]:
            return ("PLACEHOLDER", {
                "ceiling_is_placeholder": ceiling_ph,
                "floor_is_placeholder":   floor_ph,
                "bars_since_event":       bars,
                "threshold_bars":         thresholds["placeholder_bars"],
            })

    return None


def _is_duplicate(queue: List[Dict[str, Any]], pair: str, condition: str) -> bool:
    """Dedupe rule: skip if the latest pending entry for this pair has the
    same condition. Resolved/dismissed entries do NOT block re-queueing
    (state can recur after operator decision).
    """
    for entry in reversed(queue):
        if entry.get("pair") != pair:
            continue
        if entry.get("status") != "pending":
            return False  # latest entry for this pair is resolved; allow re-queue
        return entry.get("condition") == condition
    return False


def run(pair_prices: Optional[Dict[str, float]] = None,
        pair_atrs: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Scan all pairs in Phase 1 state. Append new cases to the queue.

    Args:
      pair_prices: optional {pair_name: current_price}. If absent, RUNAWAY
                   check is skipped for that pair (degraded gracefully).
      pair_atrs:   optional {pair_name: h1_atr}. Used by STALE_ANCHOR
                   detail block only — not in the condition itself.

    Returns the list of NEW entries appended this call (may be empty).

    Safe to call from anywhere. Never raises.
    """
    try:
        state_all = _load_phase1_state()
        if not state_all:
            return []

        scan_now = datetime.now(timezone.utc).isoformat()
        queue = queue_io.load_queue()
        new_entries: List[Dict[str, Any]] = []

        for pair, pair_state in state_all.items():
            if not isinstance(pair_state, dict):
                continue
            try:
                cp  = (pair_prices or {}).get(pair)
                atr = (pair_atrs or {}).get(pair)
                hit = _check_pair(pair, pair_state, cp, atr, scan_now)
                if not hit:
                    continue
                condition, details = hit
                if _is_duplicate(queue, pair, condition):
                    continue
                entry = {
                    "id":             f"{pair}-{condition}-{int(datetime.now(timezone.utc).timestamp())}",
                    "pair":           pair,
                    "condition":      condition,
                    "queued_at":      scan_now,
                    "status":         "pending",
                    "details":        details,
                    "phase1_snapshot": {
                        "ceiling_price": pair_state.get("ceiling_price"),
                        "ceiling_ts":    pair_state.get("ceiling_ts"),
                        "ceiling_is_placeholder": pair_state.get("ceiling_is_placeholder"),
                        "floor_price":   pair_state.get("floor_price"),
                        "floor_ts":      pair_state.get("floor_ts"),
                        "floor_is_placeholder":   pair_state.get("floor_is_placeholder"),
                        "trend":              pair_state.get("trend"),
                        "last_event_type":    pair_state.get("last_event_type"),
                        "last_event_tier":    pair_state.get("last_event_tier"),
                        "last_event_ts":      pair_state.get("last_event_ts"),
                        "last_scanned_ts":    pair_state.get("last_scanned_ts"),
                    },
                }
                queue.append(entry)
                new_entries.append(entry)
            except Exception as inner:
                logger.warning("review_detector: pair %s failed: %s", pair, inner)
                continue

        if new_entries:
            queue_io.save_queue(queue)
        return new_entries
    except Exception as e:
        logger.warning("review_detector.run failed: %s", e)
        return []
