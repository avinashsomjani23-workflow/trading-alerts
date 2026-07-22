"""Single source of truth for WHERE every kind of file belongs.

WHY THIS EXISTS
---------------
Files used to land wherever they were created (root), so the repo cluttered up.
This module is the ONE place that knows the folder for each kind of file, and a
single function — resolve_repo_path(name) — that returns the correct location.

Every file-creation path in the project routes through here, so a new file lands
in the right folder automatically, with NO background process, NO watcher, and
NO rule for a human to remember. It is a pure function: it only computes a path,
it never runs on its own, so it cannot break the live engine or fight OneDrive.

The state-file router (smc_radar.resolve_state_path) delegates to this, so state
routing and doc/code routing share ONE brain — never two that can drift.

WHAT IT DOES NOT DO
-------------------
It does not move files that already exist, and it does not watch the filesystem.
It answers "given a bare filename, what is its correct repo-relative path?" —
callers use that when they create the file. A file created by hand OUTSIDE any
caller is the only gap; tests/test_repo_paths.py scans the tree and fails CI if
such a stray is found at root, so a slip is caught before it ships (the net,
not a mover).

RULES (add a new kind here, in ONE place)
-----------------------------------------
  *.json / *.jsonl   -> state/          (runtime state + logs)
  *.md               -> docs/           (handoffs, specs, studies)
  test_*.py          -> tests/          (offline test modules)
  everything else     -> repo root       (engine modules, config, etc.)

ROOT_KEEP overrides the rules above: files that live at repo root ON PURPOSE
because code or cron reads them by their root name. Adding a file here is a
deliberate "this stays at root" decision — keep the reason next to it.
"""

from __future__ import annotations

import os

STATE_DIR = "state"
DOCS_DIR = "docs"
TESTS_DIR = "tests"

# Files that MUST stay at repo root — code/cron reads them by root name.
# Each entry carries the reason so a future edit doesn't "tidy" it into a folder
# and silently break the reader.
ROOT_KEEP = {
    # --- runtime state read at root by code/cron ---
    "config.json",            # instrument config, hand-edited, read at root
    "active_obs.json",        # daily slate, committed at root by the Phase-1 job
    "phase2_scan_log.jsonl",  # read at root name across git history by
                              # backtest/diagnostics/h3_live_extract.py
    # --- docs that are code-wired / conventional at root ---
    "README.md",              # repo landing doc
    "CLAUDE.md",              # project guide, loaded from root every session
    "DECISION_GUARDRAILS.md", # frozen rules, referenced at root by CLAUDE.md
    "Benchmarking.md",        # code-wired doc (stays root per FOLDER_MAP)
    "TRUTH_LEDGER.md",        # the column truth-ledger, root by convention + CI
    "SWING_SWEEP_SPEC.md",    # referenced by the swing-sweep tooling
    "BACKTEST_LOG.md",        # written at root by backtest/update_registry.py
    # --- test scripts wired to run FROM ROOT by .github/workflows/tests.yml ---
    "test_schema_version.py", # CI: `python test_schema_version.py`
    "test_zone_roundtrip.py", # CI: `python test_zone_roundtrip.py`
}


def resolve_repo_path(name):
    """Return the correct repo-relative path for a bare filename.

    Idempotent and allowlist-aware:
      - a path that already has a folder (state/x, docs/y, C:\\abs) is returned
        unchanged — only a BARE filename gets a folder assigned;
      - a ROOT_KEEP file is returned bare (stays at root on purpose);
      - otherwise the extension decides the folder (see module docstring).
    """
    if not isinstance(name, str):
        return name
    # Already has a directory component, or is an absolute/drive path → leave it.
    if os.path.dirname(name) or ":" in name:
        return name
    if name in ROOT_KEEP:
        return name
    if name.endswith(".json") or name.endswith(".jsonl"):
        return os.path.join(STATE_DIR, name)
    if name.endswith(".md"):
        return os.path.join(DOCS_DIR, name)
    if name.startswith("test_") and name.endswith(".py"):
        return os.path.join(TESTS_DIR, name)
    # Engine modules, config, and anything unclassified stay at root.
    return name
