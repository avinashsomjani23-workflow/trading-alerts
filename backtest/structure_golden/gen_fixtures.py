"""Generate the committed golden fixtures for compute_structure (Wave 2 item 2A).

Run OFFLINE, by hand, ONLY when you intend to (re)record the golden behaviour:

    python -m backtest.structure_golden.gen_fixtures

This reads the gitignored backtest/cache/*.parquet (real H1 history) and writes
self-contained JSON fixtures to backtest/structure_golden/fixtures/. Each fixture
embeds BOTH the input OHLC window AND the canonical golden output, so the CI test
(test_structure_golden.py) is fully offline — it never touches the parquet cache.

Regenerating is the FEATURE: it forces you to eyeball every behaviour change.
After regenerating, diff the fixtures dir in git and read every change before you
commit. A green test after a code change means "no drift"; a changed fixture means
"behaviour moved — confirm that was intended."

Per pair we record windows that COLLECTIVELY cover the nasty cases the handoff
names (verified present in the live data):
  - birth / cold-start   : a small slice from series start (undefined -> first BOS)
  - CHoCH-in-flight      : a window whose tail leaves flip_unconfirmed True
                           (failure-window live -> the `continue` block exercised)
  - Range BOS            : a window containing >=1 tier=='Range' event (H4-wall break)
  - ranging              : a window that ends with the ranging flag set
  - plain trend          : a clean trending window (baseline)
  - weekend gap          : a window straddling a >5h bar gap (gap-aware H4 resample)

Selection is data-driven (we scan windows and pick the first that satisfies each
tag) so it self-heals if the cache is refreshed, and every fixture is tagged with
what it actually exercises for auditability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from backtest.structure_golden import harness as H  # noqa: E402
import h4_range  # noqa: E402
import dealing_range  # noqa: E402
import smc_detector  # noqa: E402

CACHE_DIR = _REPO_ROOT / "backtest" / "cache"

# pair -> cached parquet stem (symbol with '=' and '.' sanitized, per data_loader)
PAIR_SYMBOL_STEM = {
    "EURUSD": "EURUSD_X",
    "USDJPY": "JPY_X",
    "NZDUSD": "NZDUSD_X",
    "USDCHF": "CHF_X",
    "NAS100": "NQ_F",
    "GOLD": "GC_F",
}

WINDOW = 600          # bars per non-coldstart fixture window
COLD_START = 60       # bars for the birth/cold-start window (just past birth)
STEP = 200            # scan stride when hunting for a tagged window


def _load(pair: str) -> "pd.DataFrame":
    p = CACHE_DIR / f"{PAIR_SYMBOL_STEM[pair]}_1h.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"cache parquet missing for {pair}: {p}. Run the backtest data "
            f"loader first to populate backtest/cache/."
        )
    df = pd.read_parquet(p)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _summary(out: Dict[str, Any]) -> Dict[str, Any]:
    """Quick coverage tags for a structure output (for fixture metadata + hunting)."""
    ev = out.get("events", [])
    return {
        "state": out.get("state"),
        "flip_unconfirmed": bool(out.get("flip_unconfirmed")),
        "ranging": bool(out.get("ranging")),
        "n_events": len(ev),
        "n_choch": sum(1 for e in ev if e.get("type") == "CHoCH"),
        "n_range_bos": sum(1 for e in ev if e.get("tier") == "Range"),
        "n_plain_bos": sum(1 for e in ev if e.get("type") == "BOS"
                           and e.get("tier") == "BOS"),
        # Confirmation BOS (post-CHoCH reversal confirmation) — tracked so the
        # golden set always includes at least one window exercising the flip path.
        "n_confirm_bos": sum(1 for e in ev if e.get("type") == "BOS"
                             and e.get("tier") == "Confirm"),
    }


def _has_weekend_gap(df: "pd.DataFrame") -> bool:
    idx = df.index
    for i in range(1, len(idx)):
        if (idx[i] - idx[i - 1]).total_seconds() > 3600 * 5:
            return True
    return False


def _run(df: "pd.DataFrame") -> Dict[str, Any]:
    smc_detector._ATR_CACHE.clear()
    return dealing_range.compute_structure(df, h4_range.compute_h4_range(df))


def _select_windows(pair: str, df: "pd.DataFrame") -> List[Dict[str, Any]]:
    """Pick fixture windows for a pair, each tagged with the case it covers.

    Returns a list of {case, start, size} specs. First-match per case so output
    is stable for a given cache. A window may satisfy several cases; we keep the
    first window that newly covers an uncovered case to keep the fixture set small
    and each fixture's PRIMARY case clear.
    """
    specs: List[Dict[str, Any]] = []
    chosen_starts: set = set()

    # 1) birth / cold-start (always: slice from the very start)
    specs.append({"case": "cold_start", "start": 0, "size": COLD_START})

    # cases to satisfy by scanning, in priority order
    want = ["choch_in_flight", "confirmation_bos", "range_bos", "ranging",
            "plain_trend", "weekend_gap"]
    covered: set = set()

    n = len(df)
    for start in range(0, n - WINDOW, STEP):
        if len(covered) == len(want):
            break
        if start in chosen_starts:
            continue
        w = df.iloc[start:start + WINDOW]
        s = _summary(_run(w))
        gap = _has_weekend_gap(w)

        def take(case: str) -> None:
            if case in want and case not in covered:
                covered.add(case)
                chosen_starts.add(start)
                specs.append({"case": case, "start": start, "size": WINDOW})

        if s["flip_unconfirmed"]:
            take("choch_in_flight")
        elif s["n_confirm_bos"] >= 1:
            take("confirmation_bos")
        elif s["n_range_bos"] >= 1:
            take("range_bos")
        elif s["ranging"]:
            take("ranging")
        elif gap and s["n_events"] >= 5:
            take("weekend_gap")
        elif s["n_choch"] >= 1 and s["n_plain_bos"] >= 1:
            take("plain_trend")

    missing = [c for c in want if c not in covered]
    if missing:
        print(f"  [warn] {pair}: no window covered {missing} "
              f"(data may not contain that case in this cache)")
    return specs


def build_fixture(pair: str, df: "pd.DataFrame", spec: Dict[str, Any]) -> Dict[str, Any]:
    start, size = spec["start"], spec["size"]
    w = df.iloc[start:start + size]
    rows = H.window_to_rows(w)
    golden = H.compute_golden(rows, pair)  # canonical (rounded) output
    return {
        "schema": "structure_golden/v1",
        "pair": pair,
        "case": spec["case"],
        "window": {
            "start_ts": rows[0]["ts"],
            "end_ts": rows[-1]["ts"],
            "n_bars": len(rows),
        },
        "coverage": _summary(H.run_structure(H.rows_to_window(rows))),
        "decimals": H.PAIR_DECIMALS[pair],
        "input_rows": rows,
        "golden_output": golden,
    }


def main() -> int:
    out_dir = H.FIXTURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    # Self-cleaning: wipe every existing fixture before writing the fresh set.
    # The fixture set is DATA-DERIVED (_select_windows picks the first cache
    # window matching each case), so when the cache changes the set of cases a
    # pair yields can change too. Without this wipe, a case that the new cache no
    # longer produces leaves an ORPHAN fixture on disk that regen never
    # overwrites — and the test (which loads EVERY *.json) then drifts on it
    # forever, un-fixable by regen. Wiping first makes the on-disk set always
    # exactly equal to what regen produced, so orphans are impossible.
    removed = 0
    for old in out_dir.glob("*.json"):
        old.unlink()
        removed += 1
    if removed:
        print(f"  cleaned {removed} existing fixture(s) before regen")
    total = 0
    for pair in PAIR_SYMBOL_STEM:
        print(f"[{pair}]")
        df = _load(pair)
        specs = _select_windows(pair, df)
        for spec in specs:
            fx = build_fixture(pair, df, spec)
            fname = f"{pair}__{spec['case']}.json"
            path = out_dir / fname
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(fx, fh, indent=2, ensure_ascii=True)
            cov = fx["coverage"]
            print(f"  wrote {fname:32s} bars={fx['window']['n_bars']:4d} "
                  f"state={cov['state']:9s} flip={int(cov['flip_unconfirmed'])} "
                  f"rng={int(cov['ranging'])} choch={cov['n_choch']} "
                  f"rangeBOS={cov['n_range_bos']}")
            total += 1
    print(f"\nWrote {total} fixtures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
