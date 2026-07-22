"""Net that catches a misplaced file before it ships (paths.py enforcement).

Run:  python tests/test_repo_paths.py     (exit 0 iff clean)
Also collected by pytest tests/ in CI.

WHY THIS EXISTS
---------------
paths.resolve_repo_path() decides WHERE each kind of file belongs (state/ docs/
tests/ or root). Anything the project creates routes through it, so new files land
correctly with no watcher and no human rule. The ONE gap is a file created BY HAND
outside any caller — it could land at root wrong. This test is the net for that:
it scans the TRACKED files at repo root and fails if any of them is one that
resolve_repo_path would have placed in a folder. A stray .md or a stray test_*.py
committed at root turns CI red, naming the file and where it should go — caught
before it ships, without moving anything on anyone's machine.

Only TRACKED files are scanned (git ls-files), so work-in-progress that hasn't
been committed yet is invisible here — you are never blocked by your own drafts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: E402


def _tracked_root_files():
    """Bare filenames tracked at repo ROOT (no directory component)."""
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=str(_ROOT), capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return [p for p in out if "/" not in p]


def test_no_misplaced_root_file():
    failures = []
    for name in _tracked_root_files():
        want = paths.resolve_repo_path(name)
        # resolve_repo_path returns the SAME bare name for files that belong at
        # root (engine modules, config, ROOT_KEEP). If it returns a folder path,
        # this file is sitting at root but belongs elsewhere.
        if want != name:
            failures.append(f"{name}  ->  should live at  {want}")
    assert not failures, (
        "Misplaced files tracked at repo root (move them, or add to "
        "paths.ROOT_KEEP with a reason if root is intentional):\n  - "
        + "\n  - ".join(failures)
    )


def test_root_keep_files_are_bare():
    """Every ROOT_KEEP entry must be a bare filename (no folder) — otherwise the
    allowlist would silently fail to match a root file."""
    bad = [n for n in paths.ROOT_KEEP if os.path.dirname(n)]
    assert not bad, f"ROOT_KEEP entries must be bare filenames, got: {bad}"


def main():
    failures = []
    for fn in (test_no_misplaced_root_file, test_root_keep_files_are_bare):
        try:
            fn()
        except AssertionError as e:
            failures.append(str(e))
    if failures:
        print("FAIL — repo-path net:")
        for f in failures:
            print("  " + f)
        return 1
    print("PASS — no misplaced tracked file at repo root")
    return 0


if __name__ == "__main__":
    sys.exit(main())
