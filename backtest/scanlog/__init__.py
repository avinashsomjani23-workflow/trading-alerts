"""Backtest Scan Log & Run-Health system ("zero silent failures").

See SPEC_backtest_scanlog.md. This package is ADDITIVE instrumentation only:
it observes the existing backtest and writes structured records to disk, then a
hard gate decides PASS/FAIL with a non-zero process exit code on FAIL.

Public surface:
    from backtest.scanlog import ScanLog        # the per-run emitter
    from backtest.scanlog import gates          # gate evaluation
    SCHEMA_VERSION                              # bump on any field change

Design rule (SPEC Â§8): emit and move on. No new branches, defaults, retries,
or "graceful" fallbacks in the instrumented code paths. The gates do the
judging; the emitter only records.
"""

from __future__ import annotations

SCHEMA_VERSION = "1.0.0"

from backtest.scanlog.emitter import ScanLog, NullScanLog, get_active, set_active

__all__ = [
    "SCHEMA_VERSION",
    "ScanLog",
    "NullScanLog",
    "get_active",
    "set_active",
]
