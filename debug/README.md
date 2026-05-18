# debug/ — review queue sidecar

A side-car helper for manually reviewing edge cases in Phase 1 dealing-range
state. The main system (Phase 1 / 2 / 3) is unaware of this package except
for a single read-point inside `dealing_range.update_pair()` that checks for
operator-issued overrides.

## Architecture rule (non-negotiable)

```
phase1 ─────────► debug/   (debug reads phase1 state)
phase1 ◄─── one read-point ─── debug/overrides/   (operator → phase1)
```

- `debug/` imports from Phase 1 are NOT allowed.
- Phase 1 has exactly ONE import from `debug/`: `overrides.consume_override`,
  wrapped in try/except. Delete `debug/` and Phase 1 runs identically.

## Detector conditions (per pair, vet-calibrated)

| Condition | Trigger |
|---|---|
| `RUNAWAY` | `\|price − DR_mid\| > 1.5 × DR_width` AND no internal CHoCH in leg |
| `STALE_ANCHOR` | Confirmed wall untouched for ≥ N H1 bars (48 fx / 30 gold,nas) |
| `ORPHAN_BOS` | Last event is BOS Major, ≥ N bars old with no newer event |
| `PLACEHOLDER` | Either DR side still placeholder ≥ N bars after last event |

First match per pair per scan wins. Dedupe: same condition + pending = no new entry.

## Operator CLI

```powershell
python -m debug.cli scan                                   # run detector
python -m debug.cli list                                   # show pending
python -m debug.cli show <id>                              # full detail
python -m debug.cli decide <id> force_new_range --reason "stale anchor"
python -m debug.cli decide <id> keep
python -m debug.cli decide <id> invalidate
python -m debug.cli decide <id> ignore
python -m debug.cli overrides                              # pending overrides
```

Actions:
- `keep` — no system change; decision logged for later pattern review.
- `force_new_range` — drops a one-shot override; next Phase 1 scan rebuilds
  the pair's dealing range from the anchor timestamp (cold-start trimmed df).
- `invalidate` — same logging as keep; reserved for future invalidation hook.
- `ignore` — close as noise; decision logged.

## Files (runtime, gitignored)

- `queue/review_queue.json` — append-only queue. Each entry has status `pending`
  or `resolved`.
- `queue/review_decisions.json` — append-only decision log (training data
  for future rule codification).
- `overrides/<PAIR>_override.json` — one-shot, consumed-and-deleted by the
  next Phase 1 scan of that pair.
