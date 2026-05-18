"""
Atomic JSON I/O for the review queue and decisions log.

Same atomic-write pattern as dealing_range.save_state(): write to a .tmp
sibling then os.replace() — avoids torn writes if the process dies
mid-write. Reads are tolerant: any failure returns the empty default
rather than raising, so a corrupt queue never breaks Phase 1.
"""

import json
import os
from typing import Any, Dict, List

_DIR = os.path.dirname(os.path.abspath(__file__))
QUEUE_DIR     = os.path.join(_DIR, "queue")
QUEUE_PATH    = os.path.join(QUEUE_DIR, "review_queue.json")
DECISIONS_PATH = os.path.join(QUEUE_DIR, "review_decisions.json")


def _ensure_dir() -> None:
    if not os.path.isdir(QUEUE_DIR):
        os.makedirs(QUEUE_DIR, exist_ok=True)


def _atomic_write(path: str, payload: Any) -> None:
    _ensure_dir()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def load_queue() -> List[Dict[str, Any]]:
    try:
        with open(QUEUE_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_queue(queue: List[Dict[str, Any]]) -> None:
    _atomic_write(QUEUE_PATH, queue)


def load_decisions() -> List[Dict[str, Any]]:
    try:
        with open(DECISIONS_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def append_decision(decision: Dict[str, Any]) -> None:
    existing = load_decisions()
    existing.append(decision)
    _atomic_write(DECISIONS_PATH, existing)
