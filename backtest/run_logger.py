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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


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


def log_event(event: str, level: str = "info", echo: bool = True, **fields: Any) -> None:
    """Module-level convenience wrapper. Safe no-op if no logger initialised
    (the harness can be imported and used in scripts that don't want logging).
    """
    lg = RunLogger.get()
    if lg is None:
        return
    lg.event(event, level=level, echo=echo, **fields)
