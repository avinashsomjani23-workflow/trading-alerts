"""Re-render run health, human-readable, from a finished run (SPEC Â§6).

    python -m backtest.scanlog.report --run <run_id>
    python -m backtest.scanlog.report --run <run_id> --verify-determinism

--verify-determinism recomputes the content hash from the stored scan_log +
events and compares it to the hash recorded in run_health.json (gate G7). It
works on the SAME stored artifacts - it proves the recorded hash matches the
data on disk. A true re-run determinism check (run twice, same warm cache,
identical hashes) is exercised by the self-test suite, which has a writable
fixture; here we verify the on-disk stamp is internally consistent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Iterator, Dict, Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCANLOG_ROOT = _REPO_ROOT / "backtest" / "out" / "scanlog"


def _open_jsonl(run_dir: Path, name: str) -> Iterator[Dict[str, Any]]:
    """Yield records from <name>.jsonl, transparently handling the zipped
    form (<name>.jsonl.gz.zip) the run writes at the end."""
    plain = run_dir / name
    zipped = run_dir / (name + ".gz.zip")
    if plain.exists():
        with open(plain, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif zipped.exists():
        with zipfile.ZipFile(zipped) as zf:
            with zf.open(name) as f:
                for raw in f:
                    line = raw.decode("utf-8").strip()
                    if line:
                        yield json.loads(line)
    else:
        raise FileNotFoundError(f"{name}(.gz.zip) not found in {run_dir}")


def _recompute_hash(run_dir: Path) -> str:
    """Mirror emitter._bump_content_hash: hash scan records then event records,
    each json.dumps(sort_keys=True). Order is scan-file order then events-file
    order - identical to how the emitter accumulated them."""
    h = hashlib.sha256()
    for rec in _open_jsonl(run_dir, "scan_log.jsonl"):
        h.update(json.dumps(rec, sort_keys=True, default=str).encode("utf-8"))
    for rec in _open_jsonl(run_dir, "events.jsonl"):
        # emit_ts is wall-clock and excluded from the determinism stamp - it
        # must be stripped here too so the recompute matches the emitter (G7).
        rec = {k: v for k, v in rec.items() if k != "emit_ts"}
        h.update(json.dumps(rec, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_id (folder under out/scanlog)")
    ap.add_argument("--verify-determinism", action="store_true",
                    help="recompute the content hash and compare to run_health.json")
    args = ap.parse_args()

    run_dir = SCANLOG_ROOT / args.run
    if not run_dir.exists():
        print(f"run not found: {run_dir}")
        return 2

    health = json.loads((run_dir / "run_health.json").read_text(encoding="utf-8"))

    print("=" * 64)
    print(f"RUN HEALTH  -  {args.run}")
    print("=" * 64)
    print(f"{'GATE':<5} {'VERDICT':<7} DESCRIPTION")
    for g in health.get("gates", []):
        print(f"{g['id']:<5} {g['verdict']:<7} {g['description']}")
    print("-" * 64)
    nz = {k: v for k, v in health.get("condition_counts", {}).items() if v}
    if nz:
        print("CONDITION COUNTERS (nonzero):")
        for code, n in sorted(nz.items()):
            print(f"  {code:<26} {n}")
    else:
        print("CONDITION COUNTERS: none")
    print("-" * 64)
    print(f"headline (from r_realised): ${health.get('headline_pnl_usd', 0):,.2f}")
    if health.get("warnings_present"):
        print("warnings_present: TRUE")
    print(f"OVERALL: {health.get('overall')} (exit code {health.get('exit_code')})")
    print("=" * 64)

    if args.verify_determinism:
        recorded = health.get("content_hash", "")
        recomputed = _recompute_hash(run_dir)
        match = recorded == recomputed
        print(f"\ndeterminism check:")
        print(f"  recorded   : {recorded}")
        print(f"  recomputed : {recomputed}")
        print(f"  {'MATCH (G7 OK)' if match else 'MISMATCH (G7 FAIL)'}")
        if not match:
            return 1

    return int(health.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
