"""
One-shot overrides for Phase 1's dealing-range builder.

Flow:
  1. Operator runs `python -m debug.cli decide <id> force_new_range`.
  2. CLI writes `debug/overrides/<PAIR>_override.json` with a
     `force_new_range_from` ISO timestamp.
  3. Next Phase 1 scan calls `dealing_range.update_pair()`. At the top
     of that function, a single call to `overrides.consume_override(name)`
     returns the override payload (or None) AND deletes the file.
  4. update_pair() rebuilds the range from that timestamp forward by
     trimming df and forcing a cold-start.

Contract (locked):
  - File is deleted IMMEDIATELY on read. Idempotent — if the consumer
    crashes before applying, the override is lost. That's intentional:
    a half-applied override is worse than no override. Operator can
    re-issue it.
  - This module never imports from dealing_range or smc_radar. One-way
    dependency only.
  - If this module raises for any reason, the caller swallows and
    treats it as no-op. Phase 1 must never break because of debug code.

Override file schema:
  {
    "pair":                 "NZDUSD",
    "force_new_range_from": "2026-05-15T00:00:00+00:00",
    "reason":               "stale anchor, vet review 2026-05-18",
    "issued_at":            "2026-05-18T11:00:00+00:00"
  }
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_DIR = os.path.dirname(os.path.abspath(__file__))
OVERRIDES_DIR = os.path.join(_DIR, "overrides")


def _ensure_dir() -> None:
    if not os.path.isdir(OVERRIDES_DIR):
        os.makedirs(OVERRIDES_DIR, exist_ok=True)


def _path_for(pair: str) -> str:
    return os.path.join(OVERRIDES_DIR, f"{pair}_override.json")


def write_override(pair: str, force_from_iso: str, reason: str) -> str:
    """Write a one-shot override file for `pair`. Returns the file path.

    Overwrites any existing override for the same pair — last decision wins,
    operator is in control.
    """
    _ensure_dir()
    payload = {
        "pair":                 pair,
        "force_new_range_from": force_from_iso,
        "reason":               reason,
        "issued_at":            datetime.now(timezone.utc).isoformat(),
    }
    path = _path_for(pair)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def consume_override(pair: str) -> Optional[Dict[str, Any]]:
    """Read-and-delete the override file for `pair`. Returns payload or None.

    Designed to be safe to call from Phase 1: any failure returns None
    rather than raising. Deletion is best-effort — if the file can't be
    deleted, the override is still returned ONCE (the next call will
    re-return it; not catastrophic, just unwanted re-rebuild — but in
    practice os.remove on a writable file does not fail).
    """
    path = _path_for(pair)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception:
        # Corrupt override — delete and ignore.
        try:
            os.remove(path)
        except Exception:
            pass
        return None
    try:
        os.remove(path)
    except Exception:
        pass
    if not isinstance(payload, dict):
        return None
    if not payload.get("force_new_range_from"):
        return None
    return payload


def list_overrides() -> Dict[str, Dict[str, Any]]:
    """Return all currently-pending overrides keyed by pair. Read-only."""
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(OVERRIDES_DIR):
        return out
    for fn in os.listdir(OVERRIDES_DIR):
        if not fn.endswith("_override.json"):
            continue
        path = os.path.join(OVERRIDES_DIR, fn)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                out[data.get("pair", fn)] = data
        except Exception:
            continue
    return out
