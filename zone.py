"""Typed slate Zone (Wave 2 item 2B).

Why this exists
---------------
A slate zone (one Order Block persisted in active_obs.json) was hand-copied in
TWO places in smc_radar.py — `fresh_to_slate_zone` (create) and
`refresh_slate_zone` (update). The field list lived in both, by hand, and had
to be kept in sync manually. Fields HAVE died in that hand-copy before: it
caused the BOS-sequence-count bug and the mitigation-window bug. A single field
definition with ONE serialization path retires that entire bug class.

This module is that single definition. `Zone` enumerates every slate field
exactly once. The two copy sites become:
    fresh_to_slate_zone   -> Zone.from_fresh(...).to_dict()
    refresh_slate_zone    -> Zone.from_dict(existing).refresh(fresh, ...).to_dict()

Contract (LOCKED, behaviour-neutral):
- The on-disk dict is BYTE-IDENTICAL to the old hand-built dict, including KEY
  ORDER (active_obs.json is saved with json.dump(indent=2) and NO sort_keys, so
  insertion order is the on-disk order). `to_dict()` emits fields in the exact
  order `fresh_to_slate_zone` used. A round-trip test
  (test_zone_roundtrip.py) loads the live active_obs.json and asserts
  from_dict -> to_dict is byte-identical for every zone.
- Phase 2, Phase 3 and the backtest read zones as PLAIN DICTS straight from
  JSON. They never see a Zone object. Only smc_radar (the sole slate writer)
  uses this class. So this change is contained to one module.
- Nested snapshots (fvg, sweep_observed, dealing_range, break_quality) are
  carried THROUGH unchanged — Zone does not reshape them; it owns the top-level
  field list and the create/refresh rules only.

NOT in scope (needs separate owner sign-off): any death-line / SL-buffer
behaviour. This class is field-plumbing only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _pip_unit(dp: int) -> float:
    """Pip unit by decimal places — mirrors the inline rule in both copy sites."""
    return 0.0001 if dp == 5 else (0.01 if dp == 3 else 1.0)


# Default nested snapshots — the exact literals the old copy sites used as
# fallbacks, kept here so behaviour is identical when a fresh dict omits them.
def _default_break_quality() -> Dict[str, Any]:
    return {"tier": "marginal"}


def _default_sweep() -> Dict[str, Any]:
    return {"exists": False}


def _default_dealing_range() -> Dict[str, Any]:
    return {"valid": False}


# The on-disk field order (verified against fresh_to_slate_zone, smc_radar.py).
# to_dict() emits in THIS order so active_obs.json stays byte-identical.
_FIELD_ORDER = [
    "zone_id", "status", "drop_reason",
    "first_seen_iso", "first_seen_label", "last_seen_iso", "last_seen_label",
    "is_new_this_scan", "ob_timestamp", "direction", "bos_tag", "bos_tier",
    "broken_was_wall", "reversal_pct", "bos_timestamp",
    "proximal_line", "distal_line", "high", "low", "ob_body", "median_leg_body",
    "body_ratio", "walkback_depth",
    "bos_idx", "ob_idx", "impulse_start_idx", "impulse_start_price",
    "bos_swing_price", "bos_sequence_count", "break_quality", "touches",
    "status_label", "h1_atr", "current_price_at_scan",
    "distance_to_proximal_pips", "fvg", "sweep_observed", "dealing_range",
    "role",
]


@dataclass
class Zone:
    """One slate Order Block. Single source of truth for the slate field list."""

    # identity / lifecycle
    zone_id: str
    status: str = "active"
    drop_reason: Optional[str] = None
    first_seen_iso: Optional[str] = None
    first_seen_label: Optional[str] = None
    last_seen_iso: Optional[str] = None
    last_seen_label: Optional[str] = None
    is_new_this_scan: bool = True

    # structural identity (event / direction)
    ob_timestamp: Optional[str] = None
    direction: str = ""
    bos_tag: str = "BOS"
    bos_tier: str = "BOS"
    broken_was_wall: bool = False
    reversal_pct: Optional[float] = None
    bos_timestamp: Optional[str] = None

    # geometry
    proximal_line: float = 0.0
    distal_line: float = 0.0
    high: float = 0.0
    low: float = 0.0
    ob_body: float = 0.0
    median_leg_body: float = 0.0

    # OB-candle setup geometry (DECISION_GUARDRAILS.md A3, observe-only).
    # Frozen at OB formation, NEVER re-stamped on refresh — the OB candle's
    # identity does not change once the zone exists. body_ratio = chosen OB
    # candle's body/range; walkback_depth = opposing candles skipped (doji /
    # oversized / undersized) before this one was accepted (0 = first opposing
    # candle took). Nullable for zones created before this logging shipped.
    body_ratio: Optional[float] = None
    walkback_depth: Optional[int] = None

    # df-frame indices (roll each scan; refreshed together)
    bos_idx: Optional[int] = None
    ob_idx: Optional[int] = None
    impulse_start_idx: Optional[int] = None
    impulse_start_price: float = 0.0
    bos_swing_price: float = 0.0

    # scoring / display
    bos_sequence_count: int = 1
    break_quality: Dict[str, Any] = field(default_factory=_default_break_quality)
    touches: int = 0
    status_label: str = "Pristine"
    h1_atr: float = 0.0
    current_price_at_scan: float = 0.0
    distance_to_proximal_pips: float = 0.0

    # frozen Phase-1 snapshots (carried through unchanged)
    fvg: Dict[str, Any] = field(default_factory=dict)
    sweep_observed: Dict[str, Any] = field(default_factory=_default_sweep)
    dealing_range: Dict[str, Any] = field(default_factory=_default_dealing_range)

    # two-OB role
    role: str = "primary"

    # ---- constructors -----------------------------------------------------

    @classmethod
    def from_fresh(cls, fresh: Dict[str, Any], zone_id: str, ist_now,
                   current_price: float, dp: int) -> "Zone":
        """Build a brand-new slate zone from a fresh-detection dict.

        Mirrors fresh_to_slate_zone EXACTLY (same defaults, same computed
        first/last-seen, same distance pips).
        """
        iso = ist_now.isoformat()
        label = ist_now.strftime("%H:%M IST")
        dist_pips = round(
            abs(current_price - fresh["proximal_line"]) / _pip_unit(dp), 1)
        return cls(
            zone_id=zone_id,
            status="active",
            drop_reason=None,
            first_seen_iso=iso,
            first_seen_label=label,
            last_seen_iso=iso,
            last_seen_label=label,
            is_new_this_scan=True,
            ob_timestamp=fresh.get("ob_timestamp"),
            direction=fresh["direction"],
            bos_tag=fresh["bos_tag"],
            bos_tier=fresh.get("bos_tier", "BOS"),
            broken_was_wall=fresh.get("broken_was_wall", False),
            reversal_pct=fresh.get("reversal_pct"),
            bos_timestamp=fresh.get("bos_timestamp"),
            proximal_line=fresh["proximal_line"],
            distal_line=fresh["distal_line"],
            high=fresh["high"],
            low=fresh["low"],
            ob_body=fresh["ob_body"],
            median_leg_body=fresh["median_leg_body"],
            # Formation-time setup geometry — stamped once here, frozen (A3).
            body_ratio=fresh.get("body_ratio"),
            walkback_depth=fresh.get("walkback_depth"),
            bos_idx=fresh["bos_idx"],
            ob_idx=fresh["ob_idx"],
            impulse_start_idx=fresh["impulse_start_idx"],
            impulse_start_price=fresh["impulse_start_price"],
            bos_swing_price=fresh["bos_swing_price"],
            bos_sequence_count=fresh.get("bos_sequence_count", 1),
            break_quality=fresh.get("break_quality", _default_break_quality()),
            touches=fresh.get("touches", 0),
            status_label=fresh.get("status", "Pristine"),
            h1_atr=fresh.get("h1_atr", 0.0),
            current_price_at_scan=current_price,
            distance_to_proximal_pips=dist_pips,
            fvg=fresh["fvg"],
            sweep_observed=fresh.get("sweep_observed", _default_sweep()),
            dealing_range=fresh.get("dealing_range", _default_dealing_range()),
            role=fresh.get("role", "primary"),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Zone":
        """Load an existing slate-zone dict into a Zone (lossless for known fields).

        Unknown/extra keys are preserved via `_extra` so a round-trip of a slate
        written by an older/newer code version never silently drops data.

        KEY-ORDER PRESERVATION: a zone first written by an older code version can
        carry a DIFFERENT on-disk key order (e.g. bos_sequence_count/break_quality
        appended at the end because they were added to the schema later, after the
        zone was created). To keep the on-disk file byte-identical across a
        no-op refresh, we record the incoming key order and replay it in to_dict.
        Only brand-new zones (from_fresh) use the canonical _FIELD_ORDER.
        """
        known = {k: d[k] for k in _FIELD_ORDER if k in d}
        z = cls(**known)  # type: ignore[arg-type]
        z._extra = {k: v for k, v in d.items() if k not in _FIELD_ORDER}
        z._key_order = list(d.keys())  # preserve the original on-disk order
        return z

    # ---- refresh ----------------------------------------------------------

    def refresh(self, fresh: Dict[str, Any], ist_now, current_price: float,
                dp: int) -> "Zone":
        """Update this zone from a fresh scan, preserving identity.

        Mirrors refresh_slate_zone EXACTLY: identity (zone_id, first_seen)
        untouched; mutable structural + snapshot fields refreshed; the same
        get-with-self-fallback semantics for fields that may be absent on a
        fresh dict.
        """
        self.last_seen_iso = ist_now.isoformat()
        self.last_seen_label = ist_now.strftime("%H:%M IST")
        self.is_new_this_scan = False

        self.proximal_line = fresh["proximal_line"]
        self.distal_line = fresh["distal_line"]
        self.high = fresh["high"]
        self.low = fresh["low"]
        self.ob_body = fresh["ob_body"]
        self.median_leg_body = fresh["median_leg_body"]

        # Setup geometry is FORMATION-FROZEN (A3): keep the value stamped when
        # the zone was created; never overwrite from a re-scan. The only time we
        # adopt from fresh is back-fill — a zone created before this logging
        # shipped carries None, so a later scan may fill it in once.
        if self.body_ratio is None:
            self.body_ratio = fresh.get("body_ratio")
        if self.walkback_depth is None:
            self.walkback_depth = fresh.get("walkback_depth")

        # idx-bearing fields refreshed together (same df frame).
        self.bos_idx = fresh["bos_idx"]
        self.ob_idx = fresh["ob_idx"]
        self.impulse_start_idx = fresh["impulse_start_idx"]
        self.impulse_start_price = fresh["impulse_start_price"]
        self.bos_swing_price = fresh["bos_swing_price"]

        self.bos_sequence_count = fresh.get("bos_sequence_count",
                                            self.bos_sequence_count)
        self.break_quality = fresh.get("break_quality", self.break_quality)
        self.touches = fresh.get("touches", 0)
        self.status_label = fresh.get("status", "Pristine")
        self.h1_atr = fresh.get("h1_atr", 0.0)
        self.current_price_at_scan = current_price
        self.distance_to_proximal_pips = round(
            abs(current_price - fresh["proximal_line"]) / _pip_unit(dp), 1)
        self.fvg = fresh["fvg"]
        self.sweep_observed = fresh.get("sweep_observed", self.sweep_observed)
        self.dealing_range = fresh.get("dealing_range", self.dealing_range)
        self.bos_tier = fresh.get("bos_tier", self.bos_tier)
        self.broken_was_wall = fresh.get("broken_was_wall", self.broken_was_wall)
        self.reversal_pct = fresh.get("reversal_pct", self.reversal_pct)
        self.bos_timestamp = fresh.get("bos_timestamp", self.bos_timestamp)
        self.role = fresh.get("role", self.role)
        return self

    # ---- serialization ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Emit the on-disk dict.

        Order policy:
          - A zone loaded via from_dict replays its ORIGINAL on-disk key order
            (so a no-op load->save is byte-identical even for legacy zones whose
            order differs from the canonical one). Any known field that was
            absent on disk is appended after, then any extra/unknown keys.
          - A brand-new zone (from_fresh, no recorded order) uses the canonical
            _FIELD_ORDER.
        """
        extra = getattr(self, "_extra", None) or {}
        known_vals = {name: getattr(self, name) for name in _FIELD_ORDER}

        key_order = getattr(self, "_key_order", None)
        if not key_order:
            out = dict(known_vals)
            for k, v in extra.items():
                out[k] = v
            return out

        # Replay original order; append any known field missing from it, then
        # any extras not already placed.
        out: Dict[str, Any] = {}
        seen = set()
        for k in key_order:
            if k in known_vals:
                out[k] = known_vals[k]
            elif k in extra:
                out[k] = extra[k]
            else:
                continue
            seen.add(k)
        for name in _FIELD_ORDER:
            if name not in seen:
                out[name] = known_vals[name]
                seen.add(name)
        for k, v in extra.items():
            if k not in seen:
                out[k] = v
        return out


# from_dict() sets these; declared here so the dataclass doesn't manage them.
Zone._extra = {}        # type: ignore[attr-defined]
Zone._key_order = None  # type: ignore[attr-defined]
