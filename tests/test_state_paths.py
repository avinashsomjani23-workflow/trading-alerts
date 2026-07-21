"""Guard for the state-file path choke-point (prevents the root-clutter mess).

Run:  python tests/test_state_paths.py       (exit 0 iff all pass)
Also collected by pytest tests/ in CI.

WHY THIS EXISTS
---------------
State files (heartbeat, dedup memory, failure logs, ...) used to be saved by a
BARE filename, so they landed at repo root wherever the program ran. That is the
clutter we cleaned up. To stop it rebuilding, both engines route every JSON
load/save through smc_radar.resolve_state_path(), which rewrites a bare state
filename to state/<name> (allowlisting the two files that live at root on
purpose: active_obs.json, config.json).

This test has two jobs:
  1. LOCK THE ROUTING RULE  — resolve_state_path() behaves exactly as specified
     (bare -> state/, allowlist stays root, explicit folder untouched, idempotent).
  2. CATCH A NEW OFFENDER   — scan the two live engines for any bare ".json"/
     ".jsonl" string handed to open()/load_json/save_json/load_json_safe/
     save_json_atomic WITHOUT going through resolve_state_path. If someone adds
     a new root-writer later, this goes RED in CI and names the line. That is the
     automatic enforcement — no convention to remember, no doc to read.

If this test fails on a new write, the fix is: call the load/save helper (they
route automatically) or wrap the raw path in resolve_state_path(...).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import smc_radar  # noqa: E402

resolve = smc_radar.resolve_state_path

# Failures accumulate here so BOTH runners fail correctly:
#   - standalone main() prints the full list and returns exit 1
#   - each pytest test_* function asserts the list is empty at its end (below),
#     so `pytest tests/` in CI goes red — not a silent pass.
_FAILURES: list[str] = []


def _check(cond, msg):
    if not cond:
        _FAILURES.append(msg)


def _assert_clean():
    """Fail the current pytest test if anything was recorded so far."""
    assert not _FAILURES, "state-path guard failures:\n  - " + "\n  - ".join(_FAILURES)


# ---------------------------------------------------------------------------
# JOB 1 — lock the routing rule
# ---------------------------------------------------------------------------
def test_routing_rule():
    j = os.path.join  # OS-correct separator so this passes on win + posix

    # bare state filename -> state/<name>
    _check(resolve("heartbeat_state.json") == j("state", "heartbeat_state.json"),
           "bare .json must route into state/")
    _check(resolve("zone_audit_log.jsonl") == j("state", "zone_audit_log.jsonl"),
           "bare .jsonl must route into state/")

    # allowlist: the two files that live at root ON PURPOSE stay bare
    _check(resolve("active_obs.json") == "active_obs.json",
           "active_obs.json must stay at root (daily slate)")
    _check(resolve("config.json") == "config.json",
           "config.json must stay at root")

    # explicit folder / absolute path -> left exactly as given (idempotent)
    already = j("state", "heartbeat_state.json")
    _check(resolve(already) == already, "an already-state/ path must be unchanged")
    _check(resolve(j("backtest", "results", "x.json")) == j("backtest", "results", "x.json"),
           "a non-state folder path must be untouched")
    _check(resolve("C:\\tmp\\x.json") == "C:\\tmp\\x.json",
           "an absolute windows path must be untouched")

    # non-json strings are not our business
    _check(resolve("something.txt") == "something.txt",
           "non-json filenames must pass through unchanged")
    _check(resolve("plainname") == "plainname",
           "extensionless names must pass through unchanged")

    # non-str input must not crash (defensive)
    _check(resolve(None) is None, "None must pass through, not raise")
    _assert_clean()


def test_helpers_actually_route():
    """The scan trusts that load_json/save_json route internally. PROVE it here:
    save a BARE filename via the helpers, then assert the bytes landed in state/
    and NOT at repo root. If a future edit removes routing from a helper, the
    scan can't see it — but this test will go red."""
    import tempfile
    import shutil

    # Import the heavy Phase-2 module WHILE STILL at repo root (it reads
    # config.json at import time), before we chdir into the scratch dir.
    import Phase2_Alert_Engine as p2

    workdir = tempfile.mkdtemp(prefix="state_route_")
    cwd0 = os.getcwd()
    try:
        os.chdir(workdir)
        os.makedirs("state", exist_ok=True)
        name = "guard_probe_state.json"

        # smc_radar pair
        smc_radar.save_json_atomic(name, {"k": 1})
        _check(os.path.exists(os.path.join("state", name)),
               "save_json_atomic must write into state/")
        _check(not os.path.exists(name),
               "save_json_atomic must NOT write a bare file at root")
        _check(smc_radar.load_json_safe(name, None) == {"k": 1},
               "load_json_safe must read the state/ copy from a bare name")

        # Phase 2 pair
        name2 = "guard_probe_state2.json"
        p2.save_json(name2, {"k": 2})
        _check(os.path.exists(os.path.join("state", name2)),
               "P2 save_json must write into state/")
        _check(not os.path.exists(name2),
               "P2 save_json must NOT write a bare file at root")
        _check(p2.load_json(name2, None) == {"k": 2},
               "P2 load_json must read the state/ copy from a bare name")
    finally:
        os.chdir(cwd0)
        shutil.rmtree(workdir, ignore_errors=True)
    _assert_clean()


# ---------------------------------------------------------------------------
# JOB 2 — catch a NEW bare root-writer sneaking in
# ---------------------------------------------------------------------------
# The live engines we police. (Backtest tooling writes elsewhere and is out of
# scope for the live state dir.)
_ENGINE_FILES = ["smc_radar.py", "Phase2_Alert_Engine.py"]

# Files ALLOWED bare at root — imported from the engine so the two allowlists
# can never drift apart (single source of truth: smc_radar._ROOT_ALLOWLIST).
_ROOT_OK = set(smc_radar._ROOT_ALLOWLIST)

# A bare filename literal: quotes, no slash/backslash inside, ends .json/.jsonl.
_BARE_LITERAL = re.compile(r"""['"]([A-Za-z0-9_\-.]+\.(?:json|jsonl))['"]""")

# The helpers that ROUTE INTERNALLY. A bare filename handed to one of these is
# safe — resolve_state_path() runs inside them. (Their own defs live in the
# engines too; the def lines contain no bare literal, so they never self-flag.)
_ROUTING_HELPERS = re.compile(
    r"(?:load_json|save_json|load_json_safe|save_json_atomic)\s*\("
)

# raw open( ... ) does NOT route — a bare literal here lands at root. This is the
# dangerous pattern (e.g. an append-mode jsonl log written with open()).
_RAW_OPEN = re.compile(r"(?<![.\w])open\s*\(")

# A line is already safe when it routes the path explicitly:
#   resolve_state_path(...) | os.path.join("state", ...) | "state/…"/'state\…'
_ALREADY_ROUTED = re.compile(
    r"""resolve_state_path\s*\(|os\.path\.join\s*\(\s*['"]state['"]|['"]state[\\/]"""
)


def test_no_new_bare_root_writer():
    for fname in _ENGINE_FILES:
        path = _ROOT / fname
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]  # ignore trailing comments
            # A routing helper handles the bare name itself — safe, skip.
            if _ROUTING_HELPERS.search(code):
                continue
            # Only raw open() with a bare literal is dangerous.
            if not _RAW_OPEN.search(code):
                continue
            if _ALREADY_ROUTED.search(code):
                continue  # explicitly routed into state/ — good
            for m in _BARE_LITERAL.finditer(code):
                name = m.group(1)
                if name in _ROOT_OK:
                    continue
                _FAILURES.append(
                    f"{fname}:{i}: bare state file '{name}' passed to raw open() "
                    f"without routing — it will land at repo root. Wrap the path "
                    f"in resolve_state_path(...). Line: {code.strip()}"
                )
    _assert_clean()


def main():
    test_routing_rule()
    test_helpers_actually_route()
    test_no_new_bare_root_writer()
    if _FAILURES:
        print("FAIL — state-path guard:")
        for f in _FAILURES:
            print("  -", f)
        return 1
    print("PASS — state-path choke-point routing + no new bare root-writer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
