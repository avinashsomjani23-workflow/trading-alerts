"""Run the FULL CI gate locally, in the same order, before any push.

Usage:  python preflight.py            (all steps; exits non-zero on first fail)
        python preflight.py --fast     (skip the slow pytest suite)

WHY THIS EXISTS
---------------
CI kept going red RIGHT AFTER a ship — not because the change was wrong, but
because a guard was pushed before the pre-existing mess it catches was swept
(e.g. the file-placement net failing on two stray .md files already at root).
The fix is to run the SAME checks CI runs, locally, before pushing — so nothing
red-on-arrival can ever leave this machine.

This mirrors .github/workflows/tests.yml step-for-step. If a step is added or
removed there, mirror it here (tests/test_repo_paths.py keeps the repo tidy;
this keeps the gate honest). Keep the two in sync — they are the same gate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# (label, argv) — mirrors the CI job's steps, in order. Fast checks first so a
# common failure surfaces in seconds, not after the slow suite.
STEPS = [
    ("Structure golden-file regression",
     [sys.executable, "-m", "backtest.structure_golden.test_structure_golden"]),
    ("Scan-log self-test",
     [sys.executable, "-m", "backtest.scanlog.test_scanlog_self"]),
    ("Slate Zone round-trip",
     [sys.executable, "test_zone_roundtrip.py"]),
    ("Schema version stamp + fail-loud reader",
     [sys.executable, "test_schema_version.py"]),
    ("Live feed strips synthetic weekend bars",
     [sys.executable, "tests/test_feed_weekend_strip.py"]),
    ("Event-candle fix guard",
     [sys.executable, "tests/test_event_candle_fix.py"]),
    ("Defended-swing polarity guard",
     [sys.executable, "tests/test_defended_polarity.py"]),
    ("File-placement net (no misplaced file at repo root)",
     [sys.executable, "tests/test_repo_paths.py"]),
    ("State-path choke-point guard",
     [sys.executable, "tests/test_state_paths.py"]),
]

# The slow one — the full offline pytest suite. Skipped by --fast.
PYTEST_STEP = ("Offline pytest suite",
               [sys.executable, "-m", "pytest", "tests/", "-q"])


def _run(label, argv):
    print(f"\n>>> {label}")
    r = subprocess.run(argv, cwd=str(_ROOT))
    ok = r.returncode == 0
    print(f"    {'OK' if ok else 'FAIL'}  ({' '.join(argv[1:])})")
    return ok


def main():
    fast = "--fast" in sys.argv
    steps = list(STEPS)
    if not fast:
        steps.append(PYTEST_STEP)

    for label, argv in steps:
        if not _run(label, argv):
            print(f"\nPREFLIGHT FAILED at: {label}")
            print("Fix this before pushing — CI will fail on the same step.")
            return 1

    tail = " (fast — pytest suite skipped)" if fast else ""
    print(f"\nPREFLIGHT GREEN{tail} — safe to push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
