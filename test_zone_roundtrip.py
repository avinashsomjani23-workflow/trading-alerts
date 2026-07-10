"""Round-trip + self-tests for the slate Zone dataclass (Wave 2 item 2B).

Run:  python test_zone_roundtrip.py
Exit 0 iff:
  - every zone in the live active_obs.json survives from_dict -> to_dict
    BYTE-IDENTICAL (no field dropped, no key reordered), AND
  - the planted-failure self-tests prove the check actually bites.

This is the gate that makes the dataclass migration safe: if a field ever falls
out of the Zone definition, this test fails loudly instead of the field
silently dying on disk (the exact bug class 2B exists to kill).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from zone import Zone, _FIELD_ORDER  # noqa: E402

SLATE = _HERE / "active_obs.json"

_FAILS = []


def _ok(m): print(f"  OK:   {m}")
def _bad(m):
    print(f"  FAIL: {m}")
    _FAILS.append(m)


def _canon(d):
    """Stable JSON for byte comparison (preserve order; do not sort)."""
    return json.dumps(d, indent=2, ensure_ascii=True)


def _iter_live_zones():
    if not SLATE.exists():
        return
    raw = json.load(open(SLATE, encoding="utf-8"))
    for pair, block in (raw.get("pairs") or {}).items():
        for z in (block.get("zones") or []):
            yield pair, z


# --- 1) live round-trip ----------------------------------------------------

# Fields added to the schema AFTER some on-disk zones were written. A legacy
# zone that predates one of these will gain the key (value null) on its next
# write — an intended, additive, one-time migration, NOT the silent-drop bug
# this test guards. Round-trip may differ ONLY by these keys appearing with a
# null value; any other diff (a dropped field, reorder, or changed value) still
# fails loudly.
_ADDITIVE_MIGRATION_FIELDS = {"body_ratio", "walkback_depth"}


def _diff_is_only_additive_nulls(original: dict, roundtripped: dict) -> bool:
    """True iff roundtripped == original plus one-or-more allowed migration keys,
    each carrying null, and nothing else changed."""
    added = set(roundtripped) - set(original)
    if not added or not added.issubset(_ADDITIVE_MIGRATION_FIELDS):
        return False
    if any(roundtripped[k] is not None for k in added):
        return False  # a migration key that came back non-null is a real change
    # Every shared key must be byte-identical in value AND relative order.
    if {k: original[k] for k in original} != {k: roundtripped[k] for k in original}:
        return False
    return True


def test_live_roundtrip():
    n = 0
    for pair, zdict in _iter_live_zones():
        n += 1
        rt = Zone.from_dict(zdict).to_dict()
        if _canon(rt) == _canon(zdict):
            _ok(f"{pair} {zdict.get('zone_id')} round-trips byte-identical")
        elif _diff_is_only_additive_nulls(zdict, rt):
            added = sorted(set(rt) - set(zdict))
            _ok(f"{pair} {zdict.get('zone_id')} gains additive null field(s) "
                f"{added} — intended schema migration, not a drop")
        else:
            _bad(f"{pair} {zdict.get('zone_id')} DIFFERS after round-trip")
            a = _canon(zdict).splitlines()
            b = _canon(rt).splitlines()
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"        line {i}: {x!r} -> {y!r}")
            if len(a) != len(b):
                print(f"        length {len(a)} -> {len(b)}")
    if n == 0:
        _ok("no live zones present (empty slate) — round-trip vacuously holds")
    else:
        _ok(f"checked {n} live zone(s)")


# --- 2) field-order matches the dataclass ----------------------------------

def test_field_order_complete():
    """_FIELD_ORDER must list every dataclass field exactly once (no field can
    be silently missing from to_dict)."""
    from dataclasses import fields
    dc = [f.name for f in fields(Zone)]
    missing = [f for f in dc if f not in _FIELD_ORDER]
    extra = [f for f in _FIELD_ORDER if f not in dc]
    if missing:
        _bad(f"dataclass fields not in _FIELD_ORDER (would be dropped): {missing}")
    elif extra:
        _bad(f"_FIELD_ORDER names non-fields: {extra}")
    else:
        _ok(f"_FIELD_ORDER covers all {len(dc)} dataclass fields, no extras")


# --- 3) self-test: a dropped field IS caught -------------------------------

def test_dropped_field_caught():
    """Simulate the bug 2B prevents: a zone with an extra legacy key must NOT
    be silently dropped — from_dict preserves it via _extra and to_dict re-emits."""
    base = {k: None for k in _FIELD_ORDER}
    base["zone_id"] = "TST01"
    base["legacy_ghost_field"] = 42  # a key not in the dataclass
    rt = Zone.from_dict(base).to_dict()
    if rt.get("legacy_ghost_field") == 42:
        _ok("unknown legacy field preserved through round-trip (not dropped)")
    else:
        _bad("unknown legacy field was DROPPED — silent data loss")


# --- 4) self-test: from_fresh == old fresh_to_slate_zone shape --------------

def test_from_fresh_shape():
    """from_fresh must produce exactly the _FIELD_ORDER keys (the create site)."""
    fresh = {
        "ob_timestamp": "2025-01-01T00:00:00+00:00", "direction": "bullish",
        "bos_tag": "BOS", "proximal_line": 1.10, "distal_line": 1.09,
        "high": 1.105, "low": 1.085, "ob_body": 0.01, "median_leg_body": 0.02,
        "bos_idx": 10, "ob_idx": 5, "impulse_start_idx": 3,
        "impulse_start_price": 1.08, "bos_swing_price": 1.10, "fvg": {},
    }
    ist = datetime(2025, 1, 1, 12, 0, 0)
    d = Zone.from_fresh(fresh, "EUR01", ist, 1.0999, 5).to_dict()
    keys = list(d.keys())
    if keys == _FIELD_ORDER:
        _ok("from_fresh emits exactly the locked field order")
    else:
        _bad(f"from_fresh key set/order drift: {keys}")


# --- 5) walk-back setup geometry survives create + stays frozen on refresh ---
# Bug class (fixed 2026-07-06): smc_radar computes body_ratio / walkback_depth
# at OB formation, but zone.py's _FIELD_ORDER + from_fresh did NOT copy them, so
# they were silently dropped before reaching active_obs.json (null on disk).
# Backtest read them from the raw OB dict and was fine; LIVE was blind.

def test_walkback_fields_persist_and_freeze():
    fresh = {
        "ob_timestamp": "2026-07-05T21:00:00+00:00", "direction": "bearish",
        "bos_tag": "BOS", "proximal_line": 0.571, "distal_line": 0.5713,
        "high": 0.5713, "low": 0.571, "ob_body": 0.00011, "median_leg_body": 0.00001,
        "bos_idx": 138, "ob_idx": 135, "impulse_start_idx": 114,
        "impulse_start_price": 0.571, "bos_swing_price": 0.571, "fvg": {},
        "body_ratio": 0.407, "walkback_depth": 2,
    }
    ist = datetime(2026, 7, 6, 22, 35, 0)

    # Create: both fields must land on disk with their formation values.
    z = Zone.from_fresh(fresh, "NZD99", ist, 0.571, 4)
    d = z.to_dict()
    ok = True
    if d.get("body_ratio") != 0.407:
        _bad(f"body_ratio dropped/wrong on create: {d.get('body_ratio')}"); ok = False
    if d.get("walkback_depth") != 2:
        _bad(f"walkback_depth dropped/wrong on create: {d.get('walkback_depth')}"); ok = False

    # Refresh with DRIFTED values (a re-scan): the frozen formation values must
    # NOT change (DECISION_GUARDRAILS A3 — stamped once, never re-stamped).
    drift = dict(fresh, body_ratio=0.999, walkback_depth=9)
    z.refresh(drift, datetime(2026, 7, 6, 23, 35, 0), 0.571, 4)
    d2 = z.to_dict()
    if d2.get("body_ratio") != 0.407:
        _bad(f"body_ratio re-stamped on refresh (A3 violation): {d2.get('body_ratio')}"); ok = False
    if d2.get("walkback_depth") != 2:
        _bad(f"walkback_depth re-stamped on refresh (A3 violation): {d2.get('walkback_depth')}"); ok = False

    # Back-fill: a legacy zone (fields absent) should adopt them ONCE on refresh.
    legacy = {k: v for k, v in fresh.items()
              if k not in ("body_ratio", "walkback_depth")}
    lz = Zone.from_fresh(legacy, "NZD98", ist, 0.571, 4)
    if lz.to_dict().get("walkback_depth") is not None:
        _bad("legacy zone should start with null walkback_depth"); ok = False
    lz.refresh(fresh, datetime(2026, 7, 6, 23, 35, 0), 0.571, 4)
    if lz.to_dict().get("walkback_depth") != 2:
        _bad("legacy zone did not back-fill walkback_depth on refresh"); ok = False

    if ok:
        _ok("walk-back fields persist on create, freeze on refresh, back-fill once")


# --- 6) OB event-identity fields: which freeze vs re-stamp on refresh --------
# Deep Value Pass Area A (2026-07-10). The 6 OB-build "frozen-by-design" fields
# split into TWO groups in the LIVE persistence layer (zone.py refresh):
#
#   TRULY FROZEN (refresh never touches them):  ob_timestamp, direction, bos_tag
#   RE-STAMPED on refresh from the fresh scan:  bos_timestamp, bos_tier, h1_atr
#
# This is a REAL divergence from TRUTH_LEDGER.md, which calls bos_timestamp /
# bos_tier / atr_at_ob "immutable event fact / frozen BY DESIGN". On LIVE a
# proximity-fallback match (find_matching_slate_zone branch b, smc_radar.py:3328
# — same price, DIFFERENT break) runs refresh() and overwrites those three
# (zone.py:261/268/271). The canonical BACKTEST CSV is UNAFFECTED (its replay
# merge keys on exact ob_timestamp and refreshes only fvg) — this is a LIVE-only
# read-meaning drift, flagged as a DISCUSSION POINT, not fixed here.
#
# This test LOCKS the real behaviour so any future change in EITHER direction
# (a frozen field starting to move, or a re-stamped one being frozen) fails
# loudly. It is the guard the ledger note points at.

def test_ob_identity_fields_freeze_split_on_refresh():
    fresh = {
        "ob_timestamp": "2026-07-01T00:00:00+00:00", "direction": "bullish",
        "bos_tag": "BOS", "bos_tier": "BOS",
        "bos_timestamp": "2026-07-01T00:00:00+00:00", "h1_atr": 0.0010,
        "proximal_line": 1.10, "distal_line": 1.09, "high": 1.105, "low": 1.085,
        "ob_body": 0.01, "median_leg_body": 0.02, "bos_idx": 10, "ob_idx": 5,
        "impulse_start_idx": 3, "impulse_start_price": 1.08,
        "bos_swing_price": 1.10, "fvg": {},
    }
    ist = datetime(2026, 7, 1, 12, 0, 0)
    z = Zone.from_fresh(fresh, "EUR01", ist, 1.0999, 5)

    # A proximity-matched re-fire: SAME zone identity kept, but a LATER break was
    # re-detected carrying drifted event fields.
    drift = dict(fresh, bos_timestamp="2026-07-05T09:00:00+00:00",
                 bos_tier="Range", h1_atr=0.0025)
    z.refresh(drift, datetime(2026, 7, 5, 10, 0, 0), 1.0999, 5)
    d = z.to_dict()

    # Plain asserts (NOT _bad): this file's _bad only prints+appends, so a bare
    # _bad is fake-green under `pytest test_zone_roundtrip.py`. asserts bite under
    # BOTH the script path (main) AND pytest.
    # Group 1 — TRULY FROZEN. These MUST keep the formation value.
    assert d["ob_timestamp"] == "2026-07-01T00:00:00+00:00", \
        f"ob_timestamp moved on refresh (must be frozen): {d['ob_timestamp']}"
    assert d["direction"] == "bullish", \
        f"direction moved on refresh (must be frozen): {d['direction']}"
    assert d["bos_tag"] == "BOS", \
        f"bos_tag moved on refresh (must be frozen): {d['bos_tag']}"
    # Group 2 — RE-STAMPED (documented live behaviour, diverges from the ledger's
    # "immutable" note). If any starts freezing, the ledger claim would become
    # TRUE and this must be revisited — fail loudly to force that conversation.
    assert d["bos_timestamp"] == "2026-07-05T09:00:00+00:00", \
        f"bos_timestamp did NOT re-stamp on refresh — live behaviour changed: {d['bos_timestamp']}"
    assert d["bos_tier"] == "Range", \
        f"bos_tier did NOT re-stamp on refresh — behaviour changed: {d['bos_tier']}"
    assert d["h1_atr"] == 0.0025, \
        f"h1_atr did NOT re-stamp on refresh — behaviour changed: {d['h1_atr']}"
    _ok("refresh freezes ob_timestamp/direction/bos_tag; re-stamps "
        "bos_timestamp/bos_tier/h1_atr (live divergence from ledger, locked)")


def main():
    print("== live round-trip ==")
    test_live_roundtrip()
    print("\n== field-order completeness ==")
    test_field_order_complete()
    print("\n== self-tests (the check bites) ==")
    test_dropped_field_caught()
    test_from_fresh_shape()
    print("\n== walk-back geometry persistence ==")
    test_walkback_fields_persist_and_freeze()
    print("\n== OB identity fields: freeze vs re-stamp split on refresh ==")
    test_ob_identity_fields_freeze_split_on_refresh()
    print()
    if _FAILS:
        print(f"FAILED: {len(_FAILS)} problem(s)")
        return 1
    print("PASSED: Zone round-trip is byte-identical + checks bite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
