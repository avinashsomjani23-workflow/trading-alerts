"""
Operator CLI for the review queue.

Usage (PowerShell or any shell):
  python -m debug.cli scan          # run detector against current Phase 1 state
  python -m debug.cli list          # show pending entries
  python -m debug.cli list --all    # show all (incl. resolved)
  python -m debug.cli show <id>     # full detail for one entry
  python -m debug.cli decide <id> <action> [--reason "text"]
        actions: keep | force_new_range | invalidate | ignore
  python -m debug.cli overrides     # list pending overrides
  python -m debug.cli clear <id>    # remove an entry entirely (rare)

Decisions are logged to debug/queue/review_decisions.json. force_new_range
additionally writes a one-shot override file under debug/overrides/.

No external dependencies; works without pandas/network access. Phase 1
remains untouched — this is a side-car.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import overrides
from . import queue_io
from . import review_detector


# --- helpers -----------------------------------------------------------------

def _find(queue: List[Dict[str, Any]], entry_id: str) -> Optional[Dict[str, Any]]:
    for e in queue:
        if e.get("id") == entry_id:
            return e
    return None


def _short(entry: Dict[str, Any]) -> str:
    return (f"{entry.get('id')}  {entry.get('pair'):<8}"
            f"{entry.get('condition'):<14}{entry.get('status'):<10}"
            f"queued={entry.get('queued_at')}")


# --- commands ----------------------------------------------------------------

def cmd_scan(_args) -> int:
    new = review_detector.run()
    print(f"Scan complete. {len(new)} new case(s) queued.")
    for e in new:
        print("  + " + _short(e))
    return 0


def cmd_list(args) -> int:
    queue = queue_io.load_queue()
    if not args.all:
        queue = [e for e in queue if e.get("status") == "pending"]
    if not queue:
        print("(no entries)")
        return 0
    for e in queue:
        print(_short(e))
    return 0


def cmd_show(args) -> int:
    entry = _find(queue_io.load_queue(), args.id)
    if entry is None:
        print(f"No entry with id {args.id}", file=sys.stderr)
        return 1
    print(json.dumps(entry, indent=2, default=str))
    return 0


VALID_ACTIONS = {"keep", "force_new_range", "invalidate", "ignore"}


def cmd_decide(args) -> int:
    if args.action not in VALID_ACTIONS:
        print(f"Invalid action {args.action!r}. Use one of: {sorted(VALID_ACTIONS)}",
              file=sys.stderr)
        return 2

    queue = queue_io.load_queue()
    entry = _find(queue, args.id)
    if entry is None:
        print(f"No entry with id {args.id}", file=sys.stderr)
        return 1
    if entry.get("status") != "pending":
        print(f"Entry {args.id} is already {entry.get('status')}. Refusing to change.",
              file=sys.stderr)
        return 1

    now_iso = datetime.now(timezone.utc).isoformat()
    entry["status"]      = "resolved"
    entry["resolved_at"] = now_iso
    entry["decision"]    = args.action
    entry["reason"]      = args.reason or ""

    # If forcing a new range, drop the override file.
    override_path = None
    if args.action == "force_new_range":
        pair = entry["pair"]
        # Anchor: the wall_ts (for STALE_ANCHOR) or last_event_ts otherwise.
        # Operator can supply --from to override.
        anchor = args.force_from
        if not anchor:
            details = entry.get("details", {})
            anchor = (details.get("wall_ts")
                      or details.get("last_event_ts")
                      or entry.get("phase1_snapshot", {}).get("last_event_ts"))
        if not anchor:
            print("Cannot force_new_range: no anchor ts found in entry; "
                  "pass --from <ISO> explicitly.", file=sys.stderr)
            return 1
        override_path = overrides.write_override(
            pair=pair,
            force_from_iso=anchor,
            reason=args.reason or f"force_new_range from {anchor} (entry {entry['id']})",
        )
        entry["override_file"] = override_path

    queue_io.save_queue(queue)
    queue_io.append_decision({
        "id":          entry["id"],
        "pair":        entry["pair"],
        "condition":   entry["condition"],
        "action":      args.action,
        "reason":      args.reason or "",
        "resolved_at": now_iso,
    })

    print(f"Decision recorded: {entry['id']} -> {args.action}")
    if override_path:
        print(f"  override written: {override_path}")
    return 0


def cmd_overrides(_args) -> int:
    pending = overrides.list_overrides()
    if not pending:
        print("(no pending overrides)")
        return 0
    for pair, payload in pending.items():
        print(f"{pair:<8} from={payload.get('force_new_range_from')}  "
              f"reason={payload.get('reason')}")
    return 0


def cmd_clear(args) -> int:
    queue = queue_io.load_queue()
    before = len(queue)
    queue = [e for e in queue if e.get("id") != args.id]
    if len(queue) == before:
        print(f"No entry with id {args.id}", file=sys.stderr)
        return 1
    queue_io.save_queue(queue)
    print(f"Removed entry {args.id}.")
    return 0


# --- main --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="debug.cli",
                                description="Trading-alerts review queue CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("scan", help="Run the review detector once.")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("list", help="List queue entries.")
    sp.add_argument("--all", action="store_true",
                    help="Include resolved entries (default: pending only).")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("show", help="Show one entry in full.")
    sp.add_argument("id")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("decide", help="Record a decision for an entry.")
    sp.add_argument("id")
    sp.add_argument("action", choices=sorted(VALID_ACTIONS))
    sp.add_argument("--reason", default="")
    sp.add_argument("--from", dest="force_from", default=None,
                    help="ISO timestamp for force_new_range anchor "
                         "(overrides auto-detected anchor).")
    sp.set_defaults(func=cmd_decide)

    sp = sub.add_parser("overrides", help="List pending overrides.")
    sp.set_defaults(func=cmd_overrides)

    sp = sub.add_parser("clear", help="Remove an entry by id.")
    sp.add_argument("id")
    sp.set_defaults(func=cmd_clear)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
