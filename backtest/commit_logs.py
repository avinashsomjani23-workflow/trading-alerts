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


def _restore_stash(repo_root: Path) -> bool:
    """Pop the rebase-guard stash. Returns True if the working tree was
    restored cleanly, False if a conflict left the changes IN the stash.

    A False return is NOT a failure to act on by discarding: the changes stay
    safely in `git stash` for manual recovery. This function NEVER runs
    `git checkout -- .` or `git stash drop` -- the old code did, and it
    silently destroyed uncommitted source edits (2026-06 incident)."""
    pop = _run(["git", "stash", "pop"], repo_root)
    return pop.returncode == 0


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

    # Audit trail: the scan log (gate verdict + determinism hash + per-bar
    # heartbeat) is the only proof a run is honest. Persist it WITH the run so
    # an emailed result is always auditable later -- previously it lived only in
    # a local out/scanlog dir that the next run overwrote, so the audit trail
    # for emailed runs was lost. The run zips scan_log/events before this call,
    # so we commit the small .gz.zip plus run_health.json + manifest.json.
    scanlog_dir = repo_root / "backtest" / "out" / "scanlog" / run_dir.name
    if scanlog_dir.is_dir():
        for f in ("run_health.json", "manifest.json",
                  "scan_log.jsonl.gz.zip", "events.jsonl.gz.zip"):
            p = scanlog_dir / f
            if p.exists():
                targets.append(str(p.relative_to(repo_root)))

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
        # "cannot rebase: You have unstaged changes". We stash everything
        # unstaged before the rebase and restore it after. Our commit is
        # already made, so it is unaffected by the stash.
        #
        # SAFETY (2026-06 rewrite): this path used to `git checkout -- .` and
        # `git stash drop` on a pop conflict, which silently DESTROYED
        # uncommitted source edits and trapped work in dangling stashes. It no
        # longer does. Working changes are stashed, restored, or -- worst case
        # -- left safely in `git stash` for manual recovery. Nothing is ever
        # discarded.
        stash = _run(["git", "stash", "push", "--include-untracked",
                      "-m", "commit_logs-rebase-guard"], repo_root)
        # A FAILED stash (e.g. a OneDrive/file-lock error) leaves the tree
        # dirty. Rebasing now would hit "cannot rebase: You have unstaged
        # changes" and risk a half-applied state. Bail safely instead -- the
        # commit is local and recoverable; nothing is touched.
        if stash.returncode != 0:
            raise LogCommitError(
                f"could not stash the working tree before rebase: "
                f"{stash.stderr.strip()[:200]}. The log commit IS local but "
                f"is not on GitHub, and your working tree is untouched. This "
                f"is usually a transient file lock (OneDrive sync). Retry with: "
                f"git stash --include-untracked && git rebase origin/main && "
                f"git push origin main && git stash pop"
            )
        stashed = ("No local changes" not in stash.stdout)

        rebase = _run(["git", "rebase", "origin/main"], repo_root)
        if rebase.returncode != 0:
            rebase_err = rebase.stderr.strip()[:300]
            _run(["git", "rebase", "--abort"], repo_root)
            restored = _restore_stash(repo_root) if stashed else True
            stash_note = ("" if restored else
                          "Your working-tree changes are SAFE in `git stash` "
                          "(message: commit_logs-rebase-guard) -- recover with "
                          "`git stash pop` after resolving. ")
            raise LogCommitError(
                f"rebase onto origin/main failed during push retry: "
                f"{rebase_err}. Commit IS local but is not on GitHub. "
                f"{stash_note}Resolve manually with: git fetch origin && "
                f"git rebase origin/main && git push origin main"
            )

        # Rebase succeeded -> retry the push on the next loop iteration. Restore
        # the working tree first. A pop conflict here is NON-fatal and NON-
        # destructive: the changes stay in the stash, we warn, and continue.
        if stashed and not _restore_stash(repo_root):
            print("[commit_logs] stash pop hit a conflict -- your working-tree "
                  "changes are PRESERVED in `git stash` (commit_logs-rebase-"
                  "guard). Run `git stash pop` after the backtest. Nothing was "
                  "discarded.")

    raise LogCommitError(
        f"push failed after {max_push_attempts} attempts. Last error: "
        f"{last_err}. Commit IS local (run: {run_dir.name}) but is not on "
        f"GitHub. Push manually with: git push origin main"
    )
