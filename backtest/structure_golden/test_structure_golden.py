"""Golden-file regression test for dealing_range.compute_structure (Wave 2 item 2A).

Run:  python -m backtest.structure_golden.test_structure_golden
Exit code 0 iff every committed fixture's compute_structure output is
byte-identical to the recorded golden. Any drift prints a readable field-level
diff and fails.

This is the gate: no Wave-2 item that touches structure (2B slate, 2F rewrite)
may proceed unless this is green. It is fully OFFLINE — it reads the committed
JSON fixtures in ./fixtures/, never the gitignored parquet cache.

Self-checks (prove the net isn't a rubber stamp):
  - determinism: re-running the SAME fixture twice must produce identical bytes
    (guards the ATR-cache-leak failure mode).
  - tamper:      a deliberately corrupted golden MUST be caught by the diff.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.structure_golden import harness as H  # noqa: E402

_FAILS: List[str] = []


def _ok(msg: str) -> None:
    print(f"  OK:   {msg}")


def _bad(msg: str) -> None:
    print(f"  FAIL: {msg}")
    _FAILS.append(msg)


def _load_fixtures() -> List[Dict[str, Any]]:
    paths = sorted(H.FIXTURE_DIR.glob("*.json"))
    fixtures = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            fx = json.load(fh)
        fx["_path"] = p.name
        fixtures.append(fx)
    return fixtures


def _check_fixture(fx: Dict[str, Any]) -> None:
    pair = fx["pair"]
    rows = fx["input_rows"]
    expected = fx["golden_output"]

    actual = H.compute_golden(rows, pair)
    if H.serialize(actual) == H.serialize(expected):
        _ok(f"{fx['_path']:36s} [{fx.get('case','?')}] matches golden")
        return

    _bad(f"{fx['_path']} [{fx.get('case','?')}] DRIFTED from golden")
    for line in H.diff_canonical(expected, actual):
        print(f"        {line}")


def _check_determinism(fx: Dict[str, Any]) -> None:
    """Same fixture, two runs -> identical bytes (ATR-cache-leak guard)."""
    a = H.serialize(H.compute_golden(fx["input_rows"], fx["pair"]))
    b = H.serialize(H.compute_golden(fx["input_rows"], fx["pair"]))
    if a == b:
        _ok(f"{fx['_path']:36s} deterministic across runs")
    else:
        _bad(f"{fx['_path']} NON-deterministic (cache leak?)")


def _check_tamper(fx: Dict[str, Any]) -> None:
    """Corrupt a golden in memory; the diff MUST report a change."""
    good = json.loads(json.dumps(fx["golden_output"]))
    tampered = json.loads(json.dumps(good))
    tampered["state"] = "TAMPERED"
    msgs = H.diff_canonical(tampered, good)
    if msgs:
        _ok(f"{fx['_path']:36s} tamper caught ({len(msgs)} diff line(s))")
    else:
        _bad(f"{fx['_path']} tamper NOT caught — diff is blind")


def main() -> int:
    fixtures = _load_fixtures()
    if not fixtures:
        print("FAIL: no fixtures found in", H.FIXTURE_DIR)
        print("  Generate them first: python -m backtest.structure_golden.gen_fixtures")
        return 1

    print(f"Loaded {len(fixtures)} fixtures from {H.FIXTURE_DIR}\n")

    print("== golden match ==")
    for fx in fixtures:
        _check_fixture(fx)

    print("\n== determinism (cache-leak guard) ==")
    for fx in fixtures:
        _check_determinism(fx)

    print("\n== tamper (diff is not a rubber stamp) ==")
    # one representative per pair is enough to prove the diff bites
    seen = set()
    for fx in fixtures:
        if fx["pair"] in seen:
            continue
        seen.add(fx["pair"])
        _check_tamper(fx)

    print()
    if _FAILS:
        print(f"FAILED: {len(_FAILS)} problem(s)")
        return 1
    print(f"PASSED: {len(fixtures)} fixtures, all golden + deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
