"""Commit backtest log artifacts to the repo, regardless of trigger source.

Every backtest run -- local or GitHub Actions -- must leave a permanent log
in the repo at backtest/results/<run_id>/. Previously this lived only in
.github/workflows/backtest.yml, so local runs (the more common case) never
persisted. This module makes the commit happen at the end of every run.

Files committed (text, small):
  run_log.jsonl, console.log, summary.json, trades.csv, zone_register.json

Files NOT committed (binary or large):
  trades.xlsx (binary), raw_alerts.jsonl (can be large)

Idempotency: if there is nothing new to add (e.g. workflow already pushed
the same files), the function exits cleanly.

Safety: only stages the specific log files. Never `git add -A`. Live-system
state JSONs in the workspace are not touched.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

# Artifacts safe to keep in git: text, small, useful for cross-run debugging.
_LOG_FILES = [
    "run_log.jsonl",
    "console.log",
    "summary.json",
    "trades.csv",
    "zone_register.json",
]


def _run(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _git_available(repo_root: Path) -> bool:
    r = _run(["git", "rev-parse", "--is-inside-work-tree"], repo_root)
    return r.returncode == 0 and "true" in r.stdout.lower()


def _has_remote(repo_root: Path) -> bool:
    r = _run(["git", "remote"], repo_root)
    return r.returncode == 0 and bool(r.stdout.strip())


def commit_run_logs(
    run_dir: Path,
    repo_root: Path,
    push: bool = True,
    max_push_attempts: int = 5,
) -> Optional[str]:
    """Stage and commit the run's log files to the current branch.

    Returns the commit SHA on success, None if nothing was committed.
    Never raises on git failure -- prints the error and returns None.
    The backtest run itself is already complete; failing to commit logs
    must not poison the calling process.
    """
    if not run_dir.exists():
        print(f"[commit_logs] run_dir does not exist: {run_dir}")
        return None

    if not _git_available(repo_root):
        print(f"[commit_logs] not a git repo at {repo_root} -- skipping")
        return None

    # Files that actually exist in this run.
    present = [f for f in _LOG_FILES if (run_dir / f).exists()]
    if not present:
        print(f"[commit_logs] no log files present in {run_dir} -- skipping")
        return None

    # Stash the log files in a temp location so we can reset/clean the
    # workspace without losing them. The live system (phases 1/2/3) may have
    # touched state JSONs in the workspace; those must NOT be carried.
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="bt_logs_"))
    try:
        for fname in present:
            shutil.copy2(run_dir / fname, tmp / fname)

        # Force-stage only the specific log files (results/ is gitignored,
        # so -f is required).
        rel_dir = run_dir.relative_to(repo_root)
        targets = [str(rel_dir / f) for f in present]

        # Make sure the files are physically in the working tree at the
        # expected path (they should be -- this is defensive).
        for fname in present:
            dst = run_dir / fname
            if not dst.exists():
                shutil.copy2(tmp / fname, dst)

        add = _run(["git", "add", "-f", *targets], repo_root)
        if add.returncode != 0:
            print(f"[commit_logs] git add failed: {add.stderr.strip()}")
            return None

        # Anything actually staged?
        diff = _run(["git", "diff", "--cached", "--quiet", "--", *targets],
                    repo_root)
        if diff.returncode == 0:
            # No staged changes (already committed in a previous run, e.g.
            # by the GHA workflow). That's fine -- exit clean.
            print(f"[commit_logs] no changes to commit for {run_dir.name}")
            return None

        msg = f"Backtest logs: {run_dir.name} [skip ci]"
        commit = _run(["git", "commit", "-m", msg], repo_root)
        if commit.returncode != 0:
            print(f"[commit_logs] git commit failed: {commit.stderr.strip()}")
            return None

        sha_proc = _run(["git", "rev-parse", "HEAD"], repo_root)
        sha = sha_proc.stdout.strip()[:8] if sha_proc.returncode == 0 else "?"
        print(f"[commit_logs] committed {sha} -- {len(present)} files")

        if not push:
            return sha
        if not _has_remote(repo_root):
            print(f"[commit_logs] no remote configured -- commit only, no push")
            return sha

        # Push with retry: live phases push state JSONs constantly, so a
        # non-fast-forward is expected and harmless. Rebase + retry.
        for attempt in range(1, max_push_attempts + 1):
            push_proc = _run(["git", "push", "origin", "HEAD:main"], repo_root)
            if push_proc.returncode == 0:
                print(f"[commit_logs] push succeeded on attempt {attempt}")
                return sha
            print(f"[commit_logs] push attempt {attempt} failed -- "
                  f"{push_proc.stderr.strip()[:200]}")
            # Rebase onto remote and retry.
            _run(["git", "fetch", "origin", "main"], repo_root)
            rebase = _run(["git", "rebase", "origin/main"], repo_root)
            if rebase.returncode != 0:
                _run(["git", "rebase", "--abort"], repo_root)
                print(f"[commit_logs] rebase failed -- giving up")
                return sha  # commit is local; user can push manually
        print(f"[commit_logs] push failed after {max_push_attempts} attempts "
              f"-- commit is local at {sha}")
        return sha

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
