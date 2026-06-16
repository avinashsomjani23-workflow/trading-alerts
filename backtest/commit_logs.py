"""Commit backtest log artifacts to the repo.

Every backtest run -- local or GitHub Actions -- must leave a permanent log
in the repo at backtest/results/<run_id>/. This module is the SINGLE source
of truth for that persistence. The previous bash equivalent in
.github/workflows/backtest.yml has been removed because it raced this code
and discarded its commits via `git reset --hard`.

Files committed (text, small):
  run_log.jsonl, console.log, summary.json, trades.csv, zone_register.json

Files NOT committed (binary or large):
  trades.xlsx (binary), raw_alerts.jsonl (can be large)

Failure policy: this module RAISES on every failure path. A backtest that
cannot persist its log is a failed backtest -- silent skip is forbidden.
The caller (run_backtest.main) must propagate the failure to the user.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List

_LOG_FILES = [
    "run_log.jsonl",
    "console.log",
    "summary.json",
    "trades.csv",
    "zone_register.json",
]


class LogCommitError(RuntimeError):
    """Raised when backtest logs cannot be committed/pushed. The backtest
    run is considered failed -- the user sees this immediately, not weeks
    later when they try to fetch a run that was never persisted."""


def _run(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _git_available(repo_root: Path) -> bool:
    r = _run(["git", "rev-parse", "--is-inside-work-tree"], repo_root)
    return r.returncode == 0 and "true" in r.stdout.lower()


def _has_remote(repo_root: Path) -> bool:
    r = _run(["git", "remote"], repo_root)
    return r.returncode == 0 and bool(r.stdout.strip())


def _verify_committed(repo_root: Path, run_id: str) -> str:
    """After commit+push, prove the commit exists by grepping the log.
    Returns the SHA. Raises if no commit with this run_id is found."""
    r = _run(["git", "log", "-1", "--format=%H",
              f"--grep=Backtest logs: {run_id}"], repo_root)
    if r.returncode != 0 or not r.stdout.strip():
        raise LogCommitError(
            f"Verification failed: no commit matching "
            f"'Backtest logs: {run_id}' found in git log. "
            f"git log stderr: {r.stderr.strip()[:300]}"
        )
    return r.stdout.strip()[:8]


def commit_run_logs(
    run_dir: Path,
    repo_root: Path,
    push: bool = True,
    max_push_attempts: int = 5,
) -> str:
    """Stage, commit, push, and VERIFY the run's log files.

    Returns the commit SHA on success.
    Raises LogCommitError on any failure (run_dir missing, no log files
    written, git add/commit/push fails, or post-push verification fails).
    """
    if not run_dir.exists():
        raise LogCommitError(f"run_dir does not exist: {run_dir}")

    if not _git_available(repo_root):
        raise LogCommitError(f"not a git repo at {repo_root}")

    present = [f for f in _LOG_FILES if (run_dir / f).exists()]
    if not present:
        raise LogCommitError(
            f"no log files present in {run_dir} -- the run wrote nothing "
            f"to persist. Check that RunLogger.init() and the reporting "
            f"step ran successfully."
        )

    rel_dir = run_dir.relative_to(repo_root)
    targets = [str(rel_dir / f) for f in present]

    # Cross-run rollup files written by update_registry. Staging them in the
    # same commit is mandatory: if they sit unstaged, `git rebase` during the
    # push-retry loop fails with "cannot rebase: You have unstaged changes"
    # and the run logs never reach GitHub (see May 2026 incident). Add only
    # if present -- update_registry may have been skipped (caller logs that).
    for extra in ("backtest/registry.json", "BACKTEST_LOG.md"):
        if (repo_root / extra).exists():
            targets.append(extra)

    add = _run(["git", "add", "-f", *targets], repo_root)
    if add.returncode != 0:
        raise LogCommitError(f"git add failed: {add.stderr.strip()}")

    # Anything actually staged? If the files are byte-identical to a prior
    # commit (same run_id, same content), there's nothing to commit -- but
    # in that case the commit ALREADY EXISTS, so verification will still
    # succeed. We treat "nothing to stage" as success only if verification
    # then finds the prior commit.
    diff = _run(["git", "diff", "--cached", "--quiet", "--", *targets],
                repo_root)
    nothing_staged = (diff.returncode == 0)

    if not nothing_staged:
        msg = f"Backtest logs: {run_dir.name} [skip ci]"
        commit = _run(["git", "commit", "-m", msg], repo_root)
        if commit.returncode != 0:
            raise LogCommitError(f"git commit failed: {commit.stderr.strip()}")

        sha_proc = _run(["git", "rev-parse", "HEAD"], repo_root)
        sha = sha_proc.stdout.strip()[:8] if sha_proc.returncode == 0 else "?"
        print(f"[commit_logs] committed {sha} -- {len(present)} files for {run_dir.name}")
    else:
        print(f"[commit_logs] no new content for {run_dir.name} -- "
              f"verifying prior commit exists")

    if not push or not _has_remote(repo_root):
        # Local-only: verify and return.
        return _verify_committed(repo_root, run_dir.name)

    # Push with rebase-retry. Live phases push state JSONs constantly, so
    # non-fast-forward is expected. After max attempts, RAISE.
    last_err = ""
    for attempt in range(1, max_push_attempts + 1):
        push_proc = _run(["git", "push", "origin", "HEAD:main"], repo_root)
        if push_proc.returncode == 0:
            print(f"[commit_logs] push succeeded on attempt {attempt}")
            return _verify_committed(repo_root, run_dir.name)

        last_err = push_proc.stderr.strip()[:300]
        print(f"[commit_logs] push attempt {attempt} failed -- {last_err}")
        _run(["git", "fetch", "origin", "main"], repo_root)

        # A backtest run mutates many tracked files (state JSONs, source,
        # engine artifacts) beyond the log files we staged. Those leftovers
        # sit unstaged, and `git rebase` refuses to run with a dirty tree --
        # "cannot rebase: You have unstaged changes". Hardcoding extra files
        # into the staged set (the May 2026 fix for registry.json /
        # BACKTEST_LOG.md) only patched the two files known at the time; any
        # new unstaged file re-broke the push. Stash EVERYTHING unstaged
        # before the rebase and restore it after, so the rebase is robust to
        # any tracked-file churn the run produces. Our commit is already made,
        # so it is unaffected by the stash.
        stash = _run(["git", "stash", "push", "--include-untracked",
                      "-m", "commit_logs-rebase-guard"], repo_root)
        stashed = (stash.returncode == 0
                   and "No local changes" not in stash.stdout)

        rebase = _run(["git", "rebase", "origin/main"], repo_root)
        if rebase.returncode != 0:
            rebase_err = rebase.stderr.strip()[:300]
            _run(["git", "rebase", "--abort"], repo_root)
            if stashed:
                _run(["git", "stash", "pop"], repo_root)
            raise LogCommitError(
                f"rebase onto origin/main failed during push retry: "
                f"{rebase_err}. Commit IS local but is not on GitHub. "
                f"Resolve manually with: git fetch origin && "
                f"git rebase origin/main && git push origin main"
            )

        if stashed:
            # Restore the run's working-tree changes. A pop conflict here does
            # NOT fail the push -- our commit rebased cleanly and will push on
            # the next loop. The leftover changes are run scratch, not the logs.
            pop = _run(["git", "stash", "pop"], repo_root)
            if pop.returncode != 0:
                print(f"[commit_logs] stash pop conflict (non-fatal, "
                      f"working-tree scratch only): {pop.stderr.strip()[:200]}")
                _run(["git", "checkout", "--", "."], repo_root)
                _run(["git", "stash", "drop"], repo_root)

    raise LogCommitError(
        f"push failed after {max_push_attempts} attempts. Last error: "
        f"{last_err}. Commit IS local (run: {run_dir.name}) but is not on "
        f"GitHub. Push manually with: git push origin main"
    )
