"""Per-run logger for backtest. Two outputs, one results folder:

  results/<run_id>/console.log    — raw stdout + stderr (everything print()s,
                                    including live smc_radar / dealing_range
                                    diagnostic lines). Read this when you want
                                    the same stream the GitHub Actions console
                                    showed, without needing admin auth.

  results/<run_id>/run_log.jsonl  — structured events emitted explicitly by the
                                    backtest harness. One JSON object per line:
                                    {"ts","level","event", ...fields}. Read
                                    this when you want to grep / diff / plot.

Both files are produced for every run regardless of email or report success.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# A gap between two logged events longer than this is treated as the machine
# being FROZEN (laptop asleep, CI paused, process suspended), not real work.
# Real backtest work emits events far more often than this. Time inside such a
# gap is "idle" -- it inflates wall-clock but does NOT touch the data (a paused
# process resumes at the same instruction; nothing is computed while frozen).
STALL_GAP_S = 120.0


class _Tee:
    """Write to multiple text streams. Used to mirror stdout/stderr to a file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False


class RunLogger:
    """Singleton logger for the active backtest run. Initialise once at the
    top of run_backtest.run() and close() in a finally block.
    """
    _instance: Optional["RunLogger"] = None

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        # Timing (honest wall vs idle split). _wall_start is real clock;
        # _mono_start / _last_mono use a monotonic clock so a system clock jump
        # can't corrupt the elapsed math. _idle_s sums the FROZEN gaps (> STALL);
        # active_s = wall - idle is the real compute time to budget against.
        self._wall_start = time.time()
        self._mono_start = time.monotonic()
        self._last_mono = self._mono_start
        self._idle_s = 0.0
        self._max_gap_s = 0.0
        self.jsonl_path = out_dir / "run_log.jsonl"
        self.console_path = out_dir / "console.log"
        self._jsonl_f = open(self.jsonl_path, "w", encoding="utf-8")
        self._console_f = open(self.console_path, "w", encoding="utf-8")
        # Mirror stdout + stderr into console.log so live-module prints
        # (smc_radar, dealing_range, smc_detector) are captured too.
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._console_f)
        sys.stderr = _Tee(self._orig_stderr, self._console_f)

    def event(self, event: str, level: str = "info", echo: bool = True, **fields: Any) -> None:
        """Append a structured event to run_log.jsonl. Echoes a compact
        one-liner to stdout (which goes to console.log too) unless echo=False.
        """
        # Track the gap since the previous event. A gap > STALL_GAP_S means the
        # process was frozen (not working) -- accumulate it as idle so the final
        # timing split can subtract sleep time from the wall clock.
        now_mono = time.monotonic()
        gap = now_mono - self._last_mono
        self._last_mono = now_mono
        if gap > self._max_gap_s:
            self._max_gap_s = gap
        if gap > STALL_GAP_S:
            self._idle_s += gap
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
        }
        rec.update(fields)
        try:
            self._jsonl_f.write(json.dumps(rec, default=str) + "\n")
            self._jsonl_f.flush()
        except Exception as e:
            # Logging must never crash the run.
            try:
                self._orig_stderr.write(f"[run_logger write fail] {e}\n")
            except Exception:
                pass
        if echo:
            kv = " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"[{level.upper()}] {event}{(' ' + kv) if kv else ''}")

    def timing_summary(self) -> dict:
        """Honest wall vs active split for the run so far.

        wall_s   = real clock elapsed (what a stopwatch shows -- includes any
                   time the machine was asleep/frozen).
        idle_s   = summed frozen gaps (> STALL_GAP_S between events): sleep,
                   suspend, CI pause. NOT compute -- does not touch the data.
        active_s = wall_s - idle_s: the real work time to budget future runs on.
        max_gap_s= the single longest freeze (a big value => the machine slept).
        """
        wall = time.monotonic() - self._mono_start
        active = max(0.0, wall - self._idle_s)
        return {
            "wall_s": round(wall, 1),
            "active_s": round(active, 1),
            "idle_s": round(self._idle_s, 1),
            "max_gap_s": round(self._max_gap_s, 1),
            "wall_min": round(wall / 60, 2),
            "active_min": round(active / 60, 2),
        }

    def close(self) -> None:
        try:
            self._jsonl_f.close()
        except Exception:
            pass
        try:
            self._console_f.close()
        except Exception:
            pass
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    def write_raw(self, rec: dict) -> None:
        """Append an already-built record (a worker's buffered event) verbatim,
        preserving its original ts/level. Used by the parent to replay worker
        events into run_log.jsonl."""
        try:
            self._jsonl_f.write(json.dumps(rec, default=str) + "\n")
            self._jsonl_f.flush()
        except Exception:
            pass

    @classmethod
    def init(cls, out_dir: Path) -> "RunLogger":
        if cls._instance is not None:
            try:
                cls._instance.close()
            except Exception:
                pass
        cls._instance = cls(out_dir)
        return cls._instance

    @classmethod
    def get(cls) -> Optional["RunLogger"]:
        return cls._instance


class BufferRunLogger:
    """In-memory logger for ProcessPoolExecutor workers (2026-07-02 fix).

    Workers get a fresh interpreter where RunLogger._instance is None, so every
    log_event() there — sim skips (levels_invalid, distal-touch drops), the
    pair_funnel diagnostics, scorecard/levels errors — was a silent no-op and
    the committed run_log.jsonl had NO record of the alert->trade funnel.
    Install one of these as RunLogger._instance inside the worker, return
    `.records` in the worker result, and the parent replays them into the real
    run_log.jsonl via RunLogger.write_raw()."""

    def __init__(self, pair: str = ""):
        self.pair = pair
        self.records: list = []

    def event(self, event: str, level: str = "info", echo: bool = True,
              **fields: Any) -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
        }
        rec.update(fields)
        self.records.append(rec)

    def close(self) -> None:
        pass


def log_event(event: str, level: str = "info", echo: bool = True, **fields: Any) -> None:
    """Module-level convenience wrapper. Safe no-op if no logger initialised
    (the harness can be imported and used in scripts that don't want logging).
    """
    lg = RunLogger.get()
    if lg is None:
        return
    lg.event(event, level=level, echo=echo, **fields)
