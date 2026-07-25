"""Fix D anti-stale guard: the canonical OB freeze/live/re-read classification in
TRUTH_LEDGER.md must match the ACTUAL freeze rules in Zone.refresh (zone.py).

Run:  python tests/test_zone_freeze_classification.py
Exit 0 iff the machine-checkable split derived from Zone.refresh's own source
matches the split this test pins (which mirrors the TRUTH_LEDGER list). If someone
changes how a field is frozen/refreshed in zone.py without updating the canonical
list, THIS TEST goes red — the list cannot drift silently (CLAUDE.md forbids a
hand-maintained MD that quietly lies).

WHAT IS MACHINE-CHECKED (the contract the code actually encodes):
  FROZEN  = a field NEVER unconditionally assigned in refresh. Either not assigned
            at all (identity), or assigned ONLY inside an `if self.X is None` /
            `if not self.X` one-time back-fill guard (birth fact).
  MUTABLE = a field assigned UNCONDITIONALLY every refresh call (the LIVE + RE-READ
            + RE-STAMPED buckets — all re-written each scan; the doc splits those
            three by MEANING, which no test can read, so they are pinned as one set).
The frozen-vs-mutable boundary is the safety-critical line (a birth fact silently
un-freezing is the drift bug class), and it IS derivable from the source. This test
derives it and compares to the pinned expectation.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import zone  # noqa: E402  (path insert above)


# ── The pinned classification (mirrors the TRUTH_LEDGER canonical list) ────────
# If zone.py changes a field's freeze behaviour, update BOTH this set and the
# TRUTH_LEDGER "OB FIELD FREEZE / LIVE / RE-READ CLASSIFICATION" section together.
FROZEN_IDENTITY = {
    "zone_id", "first_seen_iso", "first_seen_label",
    "ob_timestamp", "direction", "bos_tag", "bos_tier",
}
FROZEN_BIRTH = {
    "body_ratio", "walkback_depth", "h1_atr", "sweep_v2", "bos_timestamp",
}
# LIVE + RE-READ + RE-STAMPED — all unconditionally re-written each refresh.
MUTABLE = {
    "last_seen_iso", "last_seen_label", "is_new_this_scan",
    "bos_idx", "ob_idx", "impulse_start_idx", "impulse_start_price",
    "bos_swing_price", "bos_sequence_count", "break_quality", "touches",
    "status_label", "current_price_at_scan", "distance_to_proximal_pips",
    "proximal_line", "distal_line", "high", "low", "ob_body", "median_leg_body",
    "broken_was_wall", "reversal_pct",
    "fvg", "dealing_range", "role",
}
# `status` and `drop_reason` are LIVE but set OUTSIDE refresh (mitigation / drop
# path), so refresh never assigns them — the derivation classifies them frozen-here.
# Pin them as "not touched by refresh" so the test does not demand refresh set them.
SET_OUTSIDE_REFRESH = {"status", "drop_reason"}


def _derive_from_refresh_source():
    """Parse Zone.refresh's OWN source into {field: 'frozen'|'mutable'}.

    frozen  = self.X assigned ONLY inside an `if self.X is None`/`if not self.X`
              back-fill guard, OR never assigned in refresh at all.
    mutable = self.X assigned unconditionally at least once in the method body.
    """
    src = inspect.getsource(zone.Zone.refresh)
    tree = ast.parse(_dedent(src))
    fn = tree.body[0]

    unconditional: set[str] = set()
    guarded: set[str] = set()

    def _self_targets(node):
        """self.<attr> names assigned by an assignment node."""
        names = []
        for tgt in getattr(node, "targets", []):
            if (isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"):
                names.append(tgt.attr)
        return names

    # top-level statements = unconditional; statements inside an `if` = guarded.
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign):
            unconditional.update(_self_targets(stmt))
        elif isinstance(stmt, ast.If):
            for inner in ast.walk(stmt):
                if isinstance(inner, ast.Assign):
                    guarded.update(_self_targets(inner))

    out = {}
    for name in unconditional:
        out[name] = "mutable"
    for name in guarded:
        out.setdefault(name, "frozen")  # guarded-only = back-fill = frozen
    return out


def _dedent(src: str) -> str:
    lines = src.splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[indent:] if len(line) >= indent else line
                     for line in lines)


def test_freeze_classification_matches_code():
    derived = _derive_from_refresh_source()

    errors = []

    # Every FROZEN_BIRTH field must be derived as frozen (guarded back-fill only).
    for f in FROZEN_BIRTH:
        if derived.get(f) != "frozen":
            errors.append(
                f"{f}: listed FROZEN-birth but Zone.refresh assigns it "
                f"unconditionally (derived={derived.get(f)!r}). A birth fact is "
                f"silently un-freezing — this is the drift bug class.")

    # Every MUTABLE field must be derived as mutable (unconditional each scan).
    for f in MUTABLE:
        if derived.get(f) != "mutable":
            errors.append(
                f"{f}: listed MUTABLE (live/re-read/re-stamped) but Zone.refresh "
                f"does NOT re-assign it unconditionally (derived={derived.get(f)!r})."
                f" A live field may be going stale.")

    # FROZEN_IDENTITY + SET_OUTSIDE_REFRESH must NOT be touched by refresh at all.
    for f in (FROZEN_IDENTITY | SET_OUTSIDE_REFRESH):
        if f in derived:
            errors.append(
                f"{f}: listed FROZEN-identity / set-outside-refresh but "
                f"Zone.refresh assigns it (derived={derived.get(f)!r}).")

    # No field the code touches may be unclassified (catches a NEW refresh field).
    classified = FROZEN_IDENTITY | FROZEN_BIRTH | MUTABLE | SET_OUTSIDE_REFRESH
    for f in derived:
        if f not in classified:
            errors.append(
                f"{f}: Zone.refresh assigns it but it is in NO bucket of the "
                f"canonical list. Add it to TRUTH_LEDGER + this test.")

    assert not errors, "Freeze classification drifted from zone.py:\n  " + \
        "\n  ".join(errors)


if __name__ == "__main__":
    test_freeze_classification_matches_code()
    print("OK test_zone_freeze_classification: list matches Zone.refresh")
