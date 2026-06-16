"""Tests for schema_version stamp + tolerant/fail-loud reader (Wave 1 item 1C).

Run:  python test_schema_version.py
Exit 0 iff the version contract holds:
  - stamp writes the current version,
  - a v1 file passes,
  - a MISSING version is treated as v1 (deploy-safe — pre-1C files keep working),
  - a MISMATCHED version FAILS LOUD (raises SchemaVersionError),
  - the real load_state / load_slate paths honour all of the above.

The mismatch case is the whole point of the fail-loud half: prove the net bites.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import schema  # noqa: E402

_FAILS = []


def _ok(m): print(f"  OK:   {m}")
def _bad(m):
    print(f"  FAIL: {m}")
    _FAILS.append(m)


# --- unit: stamp + check ----------------------------------------------------

def test_stamp():
    d = schema.stamp({"a": 1})
    if d.get(schema.SCHEMA_KEY) == schema.CURRENT_VERSION:
        _ok("stamp sets current version")
    else:
        _bad(f"stamp did not set version: {d}")


def test_check_matching():
    d = {"a": 1, schema.SCHEMA_KEY: schema.CURRENT_VERSION}
    try:
        schema.check(d, name="t")
        _ok("matching version passes")
    except schema.SchemaVersionError:
        _bad("matching version wrongly raised")


def test_check_missing_is_v1():
    d = {"a": 1}  # no schema_version
    try:
        out = schema.check(d, name="t")
        if out is d:
            _ok("missing version treated as v1 (deploy-safe, no raise)")
        else:
            _bad("missing version mutated the dict")
    except schema.SchemaVersionError:
        _bad("missing version wrongly FAILED LOUD (would break first deploy)")


def test_check_mismatch_fails_loud():
    d = {"a": 1, schema.SCHEMA_KEY: 999}
    try:
        schema.check(d, name="t")
        _bad("mismatched version did NOT fail loud — silent mis-parse risk")
    except schema.SchemaVersionError:
        _ok("mismatched version FAILS LOUD (raises)")


def test_check_non_dict_passes():
    # logs are lists; check must not choke on them (defence-in-depth).
    out = schema.check([1, 2, 3], name="t")
    if out == [1, 2, 3]:
        _ok("non-dict (list) passes through untouched")
    else:
        _bad("non-dict was altered")


# --- integration: real load_state / load_slate round-trip -------------------

def test_dealing_range_state_roundtrip(tmp):
    import dealing_range as dr
    orig = dr.STATE_PATH
    try:
        dr.STATE_PATH = os.path.join(tmp, "structure_state.json")
        dr.save_state({"EURUSD": {"trend": "bullish"}})
        # file got stamped
        raw = json.load(open(dr.STATE_PATH, encoding="utf-8"))
        assert raw.get(schema.SCHEMA_KEY) == schema.CURRENT_VERSION
        # load returns it fine
        loaded = dr.load_state()
        if loaded.get("EURUSD", {}).get("trend") == "bullish":
            _ok("dealing_range save->load round-trips with version stamp")
        else:
            _bad("dealing_range round-trip lost data")
        # now corrupt the version -> load_state must fail loud
        raw[schema.SCHEMA_KEY] = 999
        json.dump(raw, open(dr.STATE_PATH, "w", encoding="utf-8"))
        try:
            dr.load_state()
            _bad("dealing_range load_state did NOT fail loud on bad version")
        except schema.SchemaVersionError:
            _ok("dealing_range load_state fails loud on bad version")
        # missing version (pre-1C file) -> loads as v1
        del raw[schema.SCHEMA_KEY]
        json.dump(raw, open(dr.STATE_PATH, "w", encoding="utf-8"))
        try:
            dr.load_state()
            _ok("dealing_range load_state accepts pre-1C (no-version) file")
        except schema.SchemaVersionError:
            _bad("dealing_range load_state wrongly rejected a pre-1C file")
    finally:
        dr.STATE_PATH = orig


def test_slate_roundtrip(tmp):
    import smc_radar as sr
    orig = sr.SLATE_FILE
    try:
        sr.SLATE_FILE = os.path.join(tmp, "active_obs.json")
        sr.save_slate({"slate_date": "x", "slate_started_iso": "y", "pairs": {}})
        raw = json.load(open(sr.SLATE_FILE, encoding="utf-8"))
        assert raw.get(schema.SCHEMA_KEY) == schema.CURRENT_VERSION
        loaded = sr.load_slate()
        if "pairs" in loaded:
            _ok("slate save->load round-trips with version stamp")
        else:
            _bad("slate round-trip lost structure")
        raw[schema.SCHEMA_KEY] = 999
        json.dump(raw, open(sr.SLATE_FILE, "w", encoding="utf-8"))
        try:
            sr.load_slate()
            _bad("slate load_slate did NOT fail loud on bad version")
        except schema.SchemaVersionError:
            _ok("slate load_slate fails loud on bad version")
    finally:
        sr.SLATE_FILE = orig


def main():
    print("== unit: stamp/check ==")
    test_stamp()
    test_check_matching()
    test_check_missing_is_v1()
    test_check_mismatch_fails_loud()
    test_check_non_dict_passes()

    print("\n== integration: real load/save paths ==")
    with tempfile.TemporaryDirectory() as tmp:
        test_dealing_range_state_roundtrip(tmp)
        test_slate_roundtrip(tmp)

    print()
    if _FAILS:
        print(f"FAILED: {len(_FAILS)} problem(s)")
        return 1
    print("PASSED: schema_version stamp + tolerant/fail-loud reader verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
