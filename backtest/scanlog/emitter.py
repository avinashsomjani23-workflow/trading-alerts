"""The per-run scan-log emitter (SPEC Â§2).

One ScanLog instance per backtest run. It owns the run directory and writes
four artifacts:

    manifest.json     - written FIRST, before any scanning (Â§2.1). If it cannot
                        be fully written, the run aborts before scanning.
    scan_log.jsonl    - one record per (pair, bar) scanned - the heartbeat.
    events.jsonl      - alerts / OBs / trades with full causality timestamps.
    run_health.json   - written by gates.finalize(), not here.

Hard rules honoured here:
  * Every emit validates timestamps are tz-aware (TZ_NAIVE is a FAIL condition)
    and, for scan records, that the bar ts is an element of the pair's index
    (TS_NOT_BOUNDARY). Validation RECORDS a condition; it never silently fixes.
  * NaN ATR is written as JSON null, never a fake number (Â§2.2).
  * No sampling, no rate-limiting (Â§8). JSONL streams append-only.

This module does NOT decide PASS/FAIL. It accumulates counters and the raw
records; gates.py reads them at run end and judges.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd

from backtest.scanlog import conditions as cond


_SCHEMA_VERSION = "1.0.0"

# Active-emitter registry. The instrumented modules (replay_engine,
# run_backtest) fetch the active ScanLog via get_active() so the emit calls are
# single grep-able lines with no plumbing. When no run is active (unit tests,
# scripts) get_active() returns a NullScanLog that swallows every call - the
# instrumentation is then a true no-op.
_ACTIVE: Optional["ScanLog"] = None


def get_active() -> "ScanLog":
    return _ACTIVE if _ACTIVE is not None else _NULL


def set_active(sl: Optional["ScanLog"]) -> None:
    global _ACTIVE
    _ACTIVE = sl


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_tz_aware(ts: Any) -> bool:
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return False
    return t.tzinfo is not None


def _frame_fingerprint(df: pd.DataFrame) -> str:
    """Hash of the OHLC frame - same family the ATR cache uses (Â§2.1).

    Deterministic over content: index (ns) + OHLC values. Used to prove two
    runs saw identical data (G7 determinism) and to pin the served window.
    """
    if df is None or df.empty:
        return "empty"
    h = hashlib.sha256()
    idx = df.index.view("int64") if hasattr(df.index, "view") else df.index.astype("int64")
    h.update(pd.Series(idx).to_numpy().tobytes())
    for c in ("Open", "High", "Low", "Close"):
        if c in df.columns:
            h.update(df[c].to_numpy(dtype="float64").tobytes())
    return h.hexdigest()[:16]


class ScanLog:
    """Owns one run directory and streams the three append-only artifacts.

    Construct with ScanLog.begin(out_dir, manifest_inputs); that writes the
    manifest first and refuses to proceed if it is incomplete.
    """

    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.schema_version = _SCHEMA_VERSION
        self._scan_f = open(run_dir / "scan_log.jsonl", "w", encoding="utf-8")
        self._events_f = open(run_dir / "events.jsonl", "w", encoding="utf-8")
        # counters: condition codes + event kinds + scan outcomes.
        self.condition_counts: Counter = Counter()
        self.event_counts: Counter = Counter()
        self.outcome_counts: Counter = Counter()
        # expected vs actual heartbeat, keyed by pair, for G2.
        self.expected_scan_records: Dict[str, int] = {}
        self.actual_scan_records: Counter = Counter()
        # post-warmup bar count per pair, for the G8 NaN budget.
        self.post_warmup_bars: Counter = Counter()
        self.nan_atr_skips: Counter = Counter()
        # manifest snapshot kept in memory for the G5 end-of-run re-read.
        self.manifest: Dict[str, Any] = {}
        # rolling content hash of every scan + event record, for G7.
        self._content_hash = hashlib.sha256()
        self._closed = False
        # Worker-merge support (2026-07-02): per-worker content hashes folded
        # into G7 order-independently. Empty for a plain single-process run.
        self._worker_hashes: List[str] = []
        self._is_worker = False

    # -- construction --------------------------------------------------------

    @classmethod
    def begin(cls, out_dir: Path, manifest: Dict[str, Any]) -> "ScanLog":
        """Write the manifest FIRST, then return a live emitter (SPEC Â§2.1).

        Raises RuntimeError if the manifest is missing any required key - an
        unrecorded run never executes.
        """
        required = {
            "run_id", "schema_version", "git_sha", "risk_usd",
            "min_warmup_bars", "pairs", "knobs", "versions",
        }
        missing = required - set(manifest)
        if missing:
            raise RuntimeError(
                f"manifest incomplete, refusing to run: missing {sorted(missing)}"
            )
        sl = cls(out_dir)
        sl.manifest = dict(manifest)
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        set_active(sl)
        return sl

    @classmethod
    def begin_worker(cls, out_dir: Path) -> "ScanLog":
        """Start a WORKER emitter (2026-07-02 fix). ProcessPoolExecutor workers
        get a fresh interpreter where no ScanLog is active, so every scan /
        event / condition emitted by the replay walk was silently swallowed by
        the NullScanLog — G2 and G8 passed vacuously (0 == 0) and worker-raised
        FAIL conditions (e.g. ALERT_LOOKAHEAD_BLOCKED) never reached G4. A
        worker emitter writes real records into its own subdir (unique per
        pair, so no file contention) and the parent folds them back in via
        merge_worker(). No manifest is written — the parent owns it."""
        sl = cls(out_dir)
        sl._is_worker = True
        set_active(sl)
        return sl

    def worker_summary(self) -> Dict[str, Any]:
        """Counters + artifact location for the parent merge. Call after
        close() so the jsonl files are flushed and readable."""
        return {
            "run_dir": str(self.run_dir),
            "condition_counts": dict(self.condition_counts),
            "event_counts": dict(self.event_counts),
            "outcome_counts": dict(self.outcome_counts),
            "expected_scan_records": dict(self.expected_scan_records),
            "actual_scan_records": dict(self.actual_scan_records),
            "post_warmup_bars": dict(self.post_warmup_bars),
            "nan_atr_skips": dict(self.nan_atr_skips),
            "content_hash": self._content_hash.hexdigest(),
        }

    def merge_worker(self, summary: Dict[str, Any]) -> None:
        """Fold one worker's output into this parent emitter: counters are
        ADDED, the worker's scan/event lines are appended to the parent files
        (each record was already validated at write time in the worker), and
        the worker's content hash is folded into G7 order-independently (see
        content_hash). Worker files are deleted after the merge so the run dir
        keeps the canonical single scan_log.jsonl / events.jsonl pair."""
        for attr in ("condition_counts", "event_counts", "outcome_counts",
                     "actual_scan_records", "post_warmup_bars",
                     "nan_atr_skips"):
            getattr(self, attr).update(Counter(summary.get(attr) or {}))
        for pair, n in (summary.get("expected_scan_records") or {}).items():
            self.expected_scan_records[pair] = (
                self.expected_scan_records.get(pair, 0) + int(n))
        wdir = Path(summary["run_dir"])
        for name, fh in (("scan_log.jsonl", self._scan_f),
                         ("events.jsonl", self._events_f)):
            src = wdir / name
            if src.exists():
                with open(src, "r", encoding="utf-8") as f:
                    for line in f:
                        fh.write(line)
                src.unlink()
        try:
            wdir.rmdir()
        except OSError:
            pass  # non-empty or locked (OneDrive) — harmless leftovers
        self._worker_hashes.append(str(summary.get("content_hash") or ""))

    # -- helpers -------------------------------------------------------------

    def _bump_content_hash(self, rec: Dict[str, Any]) -> None:
        # Sort keys so the hash is order-independent and reproducible (G7).
        self._content_hash.update(
            json.dumps(rec, sort_keys=True, default=str).encode("utf-8")
        )

    def condition(self, code: str, **ctx: Any) -> None:
        """Register one occurrence of a named condition (SPEC Â§3).

        An unregistered code is itself an anomaly: it is recorded AND a
        companion UNCLASSIFIED_CONDITION is raised so the run fails loud.
        """
        if not cond.is_registered(code):
            self.condition_counts["UNCLASSIFIED_CONDITION"] += 1
            self._emit_event("unclassified_condition", offending_code=code, **ctx)
        self.condition_counts[code] += 1
        if ctx:
            # Conditions with context are also event rows so the causality and
            # offending values are recoverable, not just counted.
            self._emit_event(f"condition:{code}", **ctx)

    # -- the heartbeat (Â§2.2) ------------------------------------------------

    def declare_walk(self, pair: str, n_bars: int) -> None:
        """Record how many bars the walk will visit for `pair`. G2 compares
        this against the number of scan records actually written."""
        self.expected_scan_records[pair] = (
            self.expected_scan_records.get(pair, 0) + n_bars
        )

    def scan(self, *, pair: str, ts: Any, index: Optional[pd.DatetimeIndex],
             outcome: str, **fields: Any) -> None:
        """Write exactly one heartbeat record for one (pair, bar).

        Validation performed here (records, never fixes):
          * ts must be tz-aware            -> TZ_NAIVE (FAIL)
          * ts must be in the pair's index -> TS_NOT_BOUNDARY (FAIL)
          * outcome must be a known enum   -> UNCLASSIFIED via condition()
        """
        if not _is_tz_aware(ts):
            self.condition("TZ_NAIVE", where="scan", pair=pair, ts=str(ts))
        if index is not None and pd.Timestamp(ts) not in index:
            self.condition("TS_NOT_BOUNDARY", pair=pair, ts=str(ts))
        if outcome not in cond.OUTCOMES:
            self.condition("UNCLASSIFIED_CONDITION", where="scan_outcome",
                           outcome=outcome, pair=pair)

        rec = {
            "ts": str(ts),
            "pair": pair,
            "outcome": outcome,
            "bt_slice_mode": "B",
        }
        rec.update(fields)
        self.outcome_counts[outcome] += 1
        self.actual_scan_records[pair] += 1
        self._scan_f.write(json.dumps(rec, default=str) + "\n")
        self._bump_content_hash(rec)

    def note_post_warmup_bar(self, pair: str, atr_is_nan: bool) -> None:
        """Track the G8 NaN-ATR budget: count post-warmup bars and how many had
        no ATR. Called once per post-warmup bar by the instrumented walk."""
        self.post_warmup_bars[pair] += 1
        if atr_is_nan:
            self.nan_atr_skips[pair] += 1

    # -- events (Â§2.3) -------------------------------------------------------

    def event(self, kind: str, **fields: Any) -> None:
        """Public event emit. Validates any *_ts field is tz-aware."""
        for k, v in fields.items():
            if k.endswith("_ts") and v is not None and not _is_tz_aware(v):
                self.condition("TZ_NAIVE", where=f"event:{kind}", field=k, value=str(v))
        self.event_counts[kind] += 1
        self._emit_event(kind, **fields)

    def _emit_event(self, kind: str, **fields: Any) -> None:
        rec = {"kind": kind, "emit_ts": _now_iso()}
        rec.update(fields)
        self._events_f.write(json.dumps(rec, default=str) + "\n")
        # Determinism stamp (G7) hashes CONTENT, not wall-clock. emit_ts is the
        # only non-deterministic field on an event record, so exclude it; two
        # runs over identical data then produce identical hashes.
        hashed = {k: v for k, v in rec.items() if k != "emit_ts"}
        self._bump_content_hash(hashed)

    @contextmanager
    def span(self, name: str, **ctx: Any) -> Iterator[None]:
        """Behaviour-neutral span marker. Emits a start/end event pair."""
        self._emit_event(f"span_start:{name}", **ctx)
        try:
            yield
        finally:
            self._emit_event(f"span_end:{name}", **ctx)

    # -- finalisation --------------------------------------------------------

    def content_hash(self) -> str:
        """G7 determinism stamp. When workers were merged, their per-worker
        hashes are folded in SORTED order so the combined hash is independent
        of worker completion order — two runs over identical data produce
        identical stamps regardless of scheduling."""
        h = self._content_hash.hexdigest()
        if not self._worker_hashes:
            return h
        comb = hashlib.sha256(h.encode("utf-8"))
        for wh in sorted(self._worker_hashes):
            comb.update(wh.encode("utf-8"))
        return comb.hexdigest()

    def flush(self) -> None:
        self._scan_f.flush()
        self._events_f.flush()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._scan_f.close()
            self._events_f.close()
        finally:
            self._closed = True
            if get_active() is self:
                set_active(None)


class NullScanLog:
    """No-op emitter used when no run is active. Every method swallows.

    This is what makes the instrumentation a true no-op in unit tests and
    standalone scripts: emit lines stay in the shared backtest code, but with
    no active run they do nothing and cost nothing.
    """

    def __getattr__(self, _name: str):
        def _noop(*_a: Any, **_k: Any) -> None:
            return None
        return _noop

    @contextmanager
    def span(self, *_a: Any, **_k: Any) -> Iterator[None]:
        yield


_NULL = NullScanLog()


def build_manifest(*, run_id: str, git_sha: str, risk_usd: float,
                   min_warmup_bars: int, pairs_served: List[Dict[str, Any]],
                   knobs: Dict[str, Any], fetch_pad_days: int) -> Dict[str, Any]:
    """Assemble the manifest dict (SPEC Â§2.1).

    `pairs_served` is a list of per-pair dicts already carrying requested vs
    served windows, bar counts, fingerprint, and the proximity cap actually in
    force. `knobs` is the live-read snapshot of every Â§3 knob. Versions are
    captured here so the manifest is self-contained.
    """
    try:
        import pandas as _pd
        pandas_v = _pd.__version__
    except Exception:
        pandas_v = "unknown"
    try:
        import yfinance as _yf
        yf_v = getattr(_yf, "__version__", "unknown")
    except Exception:
        yf_v = "unknown"
    return {
        "run_id": run_id,
        "schema_version": _SCHEMA_VERSION,
        "generated_utc": _now_iso(),
        "git_sha": git_sha,
        "risk_usd": risk_usd,
        "min_warmup_bars": min_warmup_bars,
        "fetch_pad_days": fetch_pad_days,
        "pairs": pairs_served,
        "knobs": knobs,
        "versions": {
            "python": platform.python_version(),
            "pandas": pandas_v,
            "yfinance": yf_v,
        },
    }


def fingerprint(df: pd.DataFrame) -> str:
    """Public wrapper so callers building the manifest can fingerprint frames
    without reaching into a private helper."""
    return _frame_fingerprint(df)
