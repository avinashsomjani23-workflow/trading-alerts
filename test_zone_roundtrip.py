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
# NOTE: efficiency_ratio was removed 2026-07-15 (replaced by chop_at_alert, which
# is alert-anchored and NOT frozen on the zone). Legacy zones on disk may still
# carry it as an inert `_extra` key; that round-trips unchanged, so it is not a
# migration field here.
_ADDITIVE_MIGRATION_FIELDS = {"body_ratio", "walkback_depth", "sweep_v2"}


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

    # Create: all three fields must land on disk with their formation values.
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


# --- 6) OB event-identity fields: ALL frozen on refresh ---------------------
# Deep Value Pass Area A (2026-07-10). The 6 OB-build "frozen-by-design" fields
# (ob_timestamp, direction, bos_tag, bos_timestamp, bos_tier, h1_atr/atr_at_ob)
# are immutable event facts stamped once at OB formation. They MUST all survive
# a refresh unchanged.
#
# BUG FOUND + FIXED 2026-07-10: a proximity-fallback match
# (find_matching_slate_zone branch b, smc_radar.py:3328 — same price, DIFFERENT
# break) ran refresh() and USED TO overwrite bos_timestamp/bos_tier/h1_atr with
# the newer break's values (zone.py:261/268/271), so a refreshed zone no longer
# described the break that formed it and its atr_at_ob (→ every LIVE *_atr
# metric) silently drifted with today's volatility. Owner confirmed this is a
# leak, not intended. Fixed: refresh now keeps the formation value for all three
# (back-fill-if-unset only, mirroring body_ratio/walkback_depth). The backtest
# never had this bug (its replay merges by EXACT ob_timestamp, refreshes only
# fvg); this brings live to parity. This test proves the fix and bites if any of
# the 6 ever starts moving on refresh again.

def test_ob_identity_fields_all_frozen_on_refresh():
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
    # re-detected carrying DRIFTED event fields. None of them may leak in.
    drift = dict(fresh, bos_timestamp="2026-07-05T09:00:00+00:00",
                 bos_tier="Range", h1_atr=0.0025)
    z.refresh(drift, datetime(2026, 7, 5, 10, 0, 0), 1.0999, 5)
    d = z.to_dict()

    # Plain asserts (NOT _bad): this file's _bad only prints+appends, so a bare
    # _bad is fake-green under `pytest test_zone_roundtrip.py`. asserts bite under
    # BOTH the script path (main) AND pytest.
    frozen_expect = {
        "ob_timestamp": "2026-07-01T00:00:00+00:00",
        "direction": "bullish",
        "bos_tag": "BOS",
        "bos_timestamp": "2026-07-01T00:00:00+00:00",   # was leaking to 07-05
        "bos_tier": "BOS",                              # was leaking to Range
        "h1_atr": 0.0010,                               # was leaking to 0.0025
    }
    for k, want in frozen_expect.items():
        assert d[k] == want, (
            f"{k} changed on refresh (must stay frozen at formation): "
            f"got {d[k]!r}, want {want!r} — the proximity-match re-stamp leak is back")
    _ok("all 6 OB event-identity fields stay frozen through a proximity-match refresh")


def test_ob_identity_fields_backfill_once_when_legacy_unset():
    """A legacy zone that predates a field (h1_atr==0.0 / bos_timestamp None)
    adopts it ONCE on refresh, then it freezes — same one-time back-fill contract
    as body_ratio/walkback_depth. Guards the fix from over-freezing legacy zones
    into a permanently-null field."""
    fresh = {
        "ob_timestamp": "2026-07-01T00:00:00+00:00", "direction": "bullish",
        "bos_tag": "BOS", "bos_tier": "BOS", "bos_timestamp": None, "h1_atr": 0.0,
        "proximal_line": 1.10, "distal_line": 1.09, "high": 1.105, "low": 1.085,
        "ob_body": 0.01, "median_leg_body": 0.02, "bos_idx": 10, "ob_idx": 5,
        "impulse_start_idx": 3, "impulse_start_price": 1.08,
        "bos_swing_price": 1.10, "fvg": {},
    }
    ist = datetime(2026, 7, 1, 12, 0, 0)
    z = Zone.from_fresh(fresh, "LEG01", ist, 1.0999, 5)
    d0 = z.to_dict()
    assert not d0["h1_atr"] and d0["bos_timestamp"] is None, \
        "legacy zone should start with unset h1_atr / bos_timestamp"

    # First refresh carries the real values -> back-fill once.
    z.refresh(dict(fresh, h1_atr=0.0018, bos_timestamp="2026-07-01T00:00:00+00:00"),
              datetime(2026, 7, 2, 10, 0, 0), 1.0999, 5)
    d1 = z.to_dict()
    assert d1["h1_atr"] == 0.0018, f"h1_atr did not back-fill: {d1['h1_atr']}"
    assert d1["bos_timestamp"] == "2026-07-01T00:00:00+00:00", \
        f"bos_timestamp did not back-fill: {d1['bos_timestamp']}"

    # A later refresh with DRIFTED values must NOT overwrite the back-filled ones.
    z.refresh(dict(fresh, h1_atr=0.0099, bos_timestamp="2026-07-09T00:00:00+00:00"),
              datetime(2026, 7, 3, 10, 0, 0), 1.0999, 5)
    d2 = z.to_dict()
    assert d2["h1_atr"] == 0.0018, f"h1_atr re-stamped after back-fill: {d2['h1_atr']}"
    assert d2["bos_timestamp"] == "2026-07-01T00:00:00+00:00", \
        f"bos_timestamp re-stamped after back-fill: {d2['bos_timestamp']}"
    _ok("legacy zone back-fills h1_atr/bos_timestamp once, then freezes")


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
    print("\n== OB identity fields: all frozen on refresh (leak fixed) ==")
    test_ob_identity_fields_all_frozen_on_refresh()
    test_ob_identity_fields_backfill_once_when_legacy_unset()
    print()
    if _FAILS:
        print(f"FAILED: {len(_FAILS)} problem(s)")
        return 1
    print("PASSED: Zone round-trip is byte-identical + checks bite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
