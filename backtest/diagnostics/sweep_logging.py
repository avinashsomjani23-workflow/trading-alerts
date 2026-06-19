"""Knob-sweep run logging — the durable substrate for a two-year tuning corpus.

ONE sweep run = ONE immutable directory under
`backtest/diagnostics/sweeps/<run_id>/` holding four artifacts:

    manifest.json        - written FIRST, before any compute. git_sha, knob,
                           grid, month/year, RESOLVED calendar dates, pairs,
                           live-config knob snapshot, schema_version, lib
                           versions, per-pair data fingerprint. A run with a
                           manifest but no run_health is a DETECTABLE abort, not
                           a silent failure.
    results.jsonl        - one row per (pair x grid_value). DECISION metrics
                           only (see SWEEP_SCHEMA_VERSION notes). Append-only.
                           This is the ONLY file the aggregator reads.
    walk.jsonl.gz        - per (pair x grid_value x bar) heartbeat, gzipped.
                           Lets us discuss nuances later WITHOUT re-running the
                           bar-by-bar walk. Gzip because it compresses ~95%.
    run_health.json      - PASS/FAIL gate verdict + recon + scope-violation +
                           content_hash. An email is sent ONLY on PASS.

Design rules (mirrors backtest/scanlog, the proven pattern):
  * Manifest written first; an incomplete manifest refuses to run.
  * Schema is VERSIONED and frozen. Any field change bumps the version so the
    aggregator can read old vs new across two years. This is the failure mode
    that bit logging before — guard it at the schema, not in ad-hoc code.
  * Append-only JSONL. No overwrite, no `git reset --hard` race.
  * Content hash over results+walk for determinism (same inputs -> same hash).
  * This module NEVER touches live state or trading logic. It only records.
"""

from __future__ import annotations

import calendar
import gzip
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Bump on ANY change to the manifest / results / walk / health field set. The
# aggregator keys its parsing on this. Two years of data survive a code change
# only because old runs carry the version they were written under.
SWEEP_SCHEMA_VERSION = "1.0.0"

_SWEEPS_ROOT = Path(__file__).resolve().parent / "sweeps"


# ---------------------------------------------------------------------------
# Calendar — the "July has 31 days" guarantee
# ---------------------------------------------------------------------------
def month_bounds(year: int, month: int) -> tuple[str, str]:
    """Return (first_day, last_day) as 'YYYY-MM-DD' for a calendar month.

    Uses calendar.monthrange so the last day is the REAL last day: 31 for July,
    28/29 for February. No hardcoded 30. This is the only place month->date
    conversion happens; the workflow passes month+year, never dates.
    """
    if not (1 <= month <= 12):
        raise ValueError(f"month out of range: {month}")
    last = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last:02d}"


def run_id_for(knob: str, year: int, month: int, stamp: Optional[str] = None) -> str:
    """sweep_<KNOB>_<YYYYMM>_<UTCstamp>. One knob per run (one email per knob)."""
    if stamp is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"sweep_{knob}_{year:04d}{month:02d}_{stamp}"


def sweeps_root() -> Path:
    return _SWEEPS_ROOT


# ---------------------------------------------------------------------------
# Fingerprint — same family as the ATR cache / scanlog (proves data identity)
# ---------------------------------------------------------------------------
def frame_fingerprint(df: Optional[pd.DataFrame]) -> str:
    if df is None or df.empty:
        return "empty"
    h = hashlib.sha256()
    idx = df.index.view("int64") if hasattr(df.index, "view") else df.index.astype("int64")
    h.update(pd.Series(idx).to_numpy().tobytes())
    for c in ("Open", "High", "Low", "Close"):
        if c in df.columns:
            h.update(df[c].to_numpy(dtype="float64").tobytes())
    return h.hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Decision-metric whitelist — only what drives a tuning call gets logged.
# n_swings is DELIBERATELY excluded: the harness itself calls it a window-end
# census of survivors, "not a per-bar experience". Logging it would let a
# future pooler average a number the code admits is misleading. sum_pnl_usd is
# excluded too: it is sum_r * risk by construction (recon checks it once); we
# keep R, the regime-independent unit.
# ---------------------------------------------------------------------------
RESULT_METRIC_FIELDS = [
    "knob", "grid_value", "baseline", "grid_mode", "pair",
    "n_obs",            # the structural knob's direct effect (what we even see)
    "n_alerts_total",   # separates "more trades" from "better trades"
    "n_trades_filled",  # SAMPLE SIZE — nothing is trusted without it
    "sum_r_realised",   # the realised edge
    "expectancy_r",     # the headline: edge per trade
    "win_rate",         # texture: frequency vs magnitude
    "avg_logged_score", # only meaningful for score knobs; carried, not featured
    "recon_ok",         # P&L reconciliation flag (per row)
]


class SweepRun:
    """Owns one sweep run directory. manifest -> results/walk -> health."""

    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.schema_version = SWEEP_SCHEMA_VERSION
        self._results_f = open(run_dir / "results.jsonl", "w", encoding="utf-8")
        self._walk_f = gzip.open(run_dir / "walk.jsonl.gz", "wt", encoding="utf-8")
        self._content_hash = hashlib.sha256()
        self.n_results = 0
        self.n_walk = 0
        self.recon_failures: List[str] = []
        self.scope_violations: List[str] = []
        self.manifest: Dict[str, Any] = {}
        self._closed = False

    # -- construction -------------------------------------------------------
    @classmethod
    def begin(cls, knob: str, year: int, month: int, *, grid: List[Any],
              grid_mode: str, pairs: List[str], live_knob_snapshot: Dict[str, Any],
              pairs_served: List[Dict[str, Any]], risk_usd: float,
              run_id: Optional[str] = None,
              root: Optional[Path] = None) -> "SweepRun":
        """Write the manifest FIRST, then return a live writer. An incomplete
        manifest aborts before any compute (the run is unrecorded -> it never
        executes). Returns a SweepRun bound to its immutable directory."""
        start, end = month_bounds(year, month)
        rid = run_id or run_id_for(knob, year, month)
        root = root or _SWEEPS_ROOT
        manifest = {
            "run_id": rid,
            "schema_version": SWEEP_SCHEMA_VERSION,
            "generated_utc": _now_iso(),
            "git_sha": _git_sha(),
            "knob": knob,
            "grid": grid,
            "grid_mode": grid_mode,
            "year": year,
            "month": month,
            "resolved_start": start,   # REAL calendar dates (July=31)
            "resolved_end": end,
            "pairs_requested": pairs,
            "risk_usd": risk_usd,
            "live_knob_snapshot": live_knob_snapshot,  # config value at run time
            "pairs_served": pairs_served,              # per-pair served + fingerprint
            "versions": _versions(),
        }
        required = {"run_id", "schema_version", "knob", "grid", "resolved_start",
                    "resolved_end", "pairs_requested", "live_knob_snapshot"}
        missing = required - set(manifest)
        if missing:
            raise RuntimeError(
                f"sweep manifest incomplete, refusing to run: missing {sorted(missing)}")
        run_dir = root / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        sr = cls(run_dir)
        sr.manifest = manifest
        return sr

    # -- writers ------------------------------------------------------------
    def write_result(self, row: Dict[str, Any]) -> None:
        """One (pair x grid_value) decision row. Filtered to the metric
        whitelist so the corpus never carries misleading columns."""
        rec = {k: row.get(k) for k in RESULT_METRIC_FIELDS}
        if not rec.get("recon_ok", True):
            self.recon_failures.append(
                f"{rec.get('pair')} v={rec.get('grid_value')}: recon FAIL")
        self._results_f.write(json.dumps(rec, default=str) + "\n")
        self._bump_hash(rec)
        self.n_results += 1

    def write_walk_record(self, rec: Dict[str, Any]) -> None:
        """One (pair x grid_value x bar) heartbeat into the gzipped walk. Keeps
        the bar-by-bar story so nuances can be discussed without re-running."""
        self._walk_f.write(json.dumps(rec, default=str) + "\n")
        self._bump_hash(rec)
        self.n_walk += 1

    def add_scope_violations(self, violations: List[str]) -> None:
        self.scope_violations.extend(violations or [])

    def _bump_hash(self, rec: Dict[str, Any]) -> None:
        self._content_hash.update(
            json.dumps(rec, sort_keys=True, default=str).encode("utf-8"))

    # -- finalisation -------------------------------------------------------
    def finalize(self) -> Dict[str, Any]:
        """Write run_health.json and return it. PASS only if no recon failure
        and no scope violation. The orchestrator emails ONLY on PASS."""
        self._results_f.flush()
        self._walk_f.flush()
        recon_ok = not self.recon_failures
        scope_ok = not self.scope_violations
        overall = "PASS" if (recon_ok and scope_ok and self.n_results > 0) else "FAIL"
        health = {
            "run_id": self.manifest.get("run_id"),
            "schema_version": SWEEP_SCHEMA_VERSION,
            "overall": overall,
            "n_result_rows": self.n_results,
            "n_walk_records": self.n_walk,
            "recon_ok": recon_ok,
            "recon_failures": self.recon_failures,
            "scope_ok": scope_ok,
            "scope_violations": self.scope_violations,
            "content_hash": self._content_hash.hexdigest(),
            "finalized_utc": _now_iso(),
        }
        with open(self.run_dir / "run_health.json", "w", encoding="utf-8") as f:
            json.dump(health, f, indent=2, default=str)
        return health

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._results_f.close()
            self._walk_f.close()
        finally:
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # If the caller exits without an explicit finalize() (e.g. an exception
        # mid-run), still write a FAIL health file so the dir is never left with
        # a manifest but no health — that ambiguity is the silent-failure mode.
        try:
            if not (self.run_dir / "run_health.json").exists():
                if exc_type is not None:
                    self.scope_violations.append(
                        f"run aborted: {exc_type.__name__}: {exc}")
                self.finalize()
        finally:
            self.close()
        return False


def _versions() -> Dict[str, str]:
    try:
        pandas_v = pd.__version__
    except Exception:
        pandas_v = "unknown"
    try:
        import yfinance as _yf
        yf_v = getattr(_yf, "__version__", "unknown")
    except Exception:
        yf_v = "unknown"
    return {"python": platform.python_version(), "pandas": pandas_v, "yfinance": yf_v}


# ---------------------------------------------------------------------------
# Reader side — used by the email builder and the two-year aggregator. ONE
# parser for the corpus; nothing reads the markdown, nothing hand-rolls JSON.
# ---------------------------------------------------------------------------
def read_results(run_dir: Path) -> List[Dict[str, Any]]:
    p = run_dir / "results.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_manifest(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "manifest.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def read_health(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "run_health.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_run_dirs(knob: Optional[str] = None,
                  root: Optional[Path] = None):
    """Yield sweep run dirs, optionally filtered to one knob. Glob on the
    immutable run_id naming so the aggregator never depends on a side index."""
    root = root or _SWEEPS_ROOT
    if not root.exists():
        return
    pattern = f"sweep_{knob}_*" if knob else "sweep_*"
    for d in sorted(root.glob(pattern)):
        if d.is_dir() and (d / "manifest.json").exists():
            yield d
