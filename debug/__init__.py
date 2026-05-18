"""
Debug / review-queue helper module.

This package is a SIDECAR. It reads Phase 1 state but never writes to it.
The only point of contact with the main system is `consume_override()`,
called once near the top of dealing_range.update_pair(). If this package
is deleted, the main system runs identically — the override call resolves
to a no-op when the package or override file is absent.

Layout:
  config.py        - pair-aware thresholds (ATR/bar based, vet-calibrated)
  review_detector.py - scans Phase 1 state, writes pending cases
  queue_io.py      - atomic JSON read/write for queue + decisions
  overrides.py     - read/consume one-shot override files (called by Phase 1)
  cli.py           - operator interface: list / show / decide / clear
  queue/           - review_queue.json, review_decisions.json
  overrides/       - one-shot override files (consumed-then-deleted)
"""
