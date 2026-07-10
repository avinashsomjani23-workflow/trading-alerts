"""Regression: ob_freshness_comparison buckets on ob_touches, NOT alert_seq.

Run:  python tests/test_ob_freshness_column.py
Exit 0 iff the freshness insight splits trades by the alert-time proximal touch
count (ob_touches: 0/1/2), independent of the re-fire counter (alert_seq).

Why this shape / the bug class it kills:
  run_backtest.py dedupes to the FIRST alert per OB, so alert_seq is ALWAYS 1 in
  trades.csv. The freshness table was keyed on alert_seq, which made every trade
  land in one bucket and left the other two empty by construction — the insight
  was structurally inert while looking populated (TRUTH_LEDGER.md).

  This guard builds a frame where ob_touches and alert_seq DISAGREE on purpose:
  ob_touches spreads across 0/1/2, but alert_seq is pinned to 1 (the live
  reality). If the function ever regresses to keying on alert_seq (or any
  constant column), all rows collapse into one bucket and the >=2 populated
  buckets assertion fails. That is the tripwire.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from backtest import insights as ins

_FAILS = []


def _ok(m): print(f"  OK:   {m}")
def _bad(m):
    # RAISE, don't just collect: CI runs these via `pytest tests/ -q`, which
    # never calls main(). A print-and-append _bad is invisible to pytest -> the
    # guard is green even when the code is broken (Deep Value A, 2026-07-10).
    print(f"  FAIL: {m}")
    _FAILS.append(m)
    raise AssertionError(m)


def _frame():
    """Filled+resolved trades spread across ob_touches 0/1/2, alert_seq pinned to 1.

    ob_touches carries the real signal; alert_seq mirrors the live first-fire
    dedup (always 1). r_realised is set so each bucket has a distinct, non-empty
    win/loss mix — a bucket that silently merges with another would change these.
    """
    rows = []
    # touch 0: 3 wins, 1 loss
    for r in (1.0, 1.0, 1.0, -1.0):
        rows.append({"ob_touches": 0, "alert_seq": 1, "exit_reason": "tp1", "r_realised": r})
    # touch 1: 1 win, 2 losses
    for r in (1.0, -1.0, -1.0):
        rows.append({"ob_touches": 1, "alert_seq": 1, "exit_reason": "sl", "r_realised": r})
    # touch 2: 1 win, 1 loss
    for r in (1.0, -1.0):
        rows.append({"ob_touches": 2, "alert_seq": 1, "exit_reason": "sl", "r_realised": r})
    # a never_filled row that _filled must drop (never counted in any bucket)
    rows.append({"ob_touches": 0, "alert_seq": 1, "exit_reason": "never_filled", "r_realised": 0.0})
    return pd.DataFrame(rows)


def main():
    df = _frame()
    out = ins.ob_freshness_comparison(df)

    by_touch = {r["touch"]: r for r in out}

    # 1) All three buckets present, keyed 0/1/2 (the ob_touches domain, not 1/2/3).
    if set(by_touch) == {0, 1, 2}:
        _ok("buckets keyed on ob_touches domain {0,1,2}")
    else:
        _bad(f"expected touch keys {{0,1,2}}, got {sorted(by_touch)} "
             f"(regressed to alert_seq/1-based?)")

    # 2) At least two buckets are populated — the class-kill. If the function keys
    #    on the constant alert_seq column, everything lands in one bucket.
    populated = [t for t, r in by_touch.items() if r["n"] > 0]
    if len(populated) >= 2:
        _ok(f"{len(populated)} buckets populated (split is real, not collapsed)")
    else:
        _bad(f"only {len(populated)} bucket populated — column is constant "
             f"(alert_seq regression)")

    # 3) Exact per-bucket counts (never_filled dropped, wins/losses correct).
    expect = {0: (4, 3, 1), 1: (3, 1, 2), 2: (2, 1, 1)}  # n, wins, losses
    for t, (n, w, l) in expect.items():
        r = by_touch.get(t, {})
        if (r.get("n"), r.get("wins"), r.get("losses")) == (n, w, l):
            _ok(f"touch={t}: n={n} wins={w} losses={l}")
        else:
            _bad(f"touch={t}: expected n/w/l={n}/{w}/{l}, "
                 f"got {r.get('n')}/{r.get('wins')}/{r.get('losses')}")

    print()
    if _FAILS:
        print(f"FAILED ({len(_FAILS)} check(s))")
        sys.exit(1)
    print("PASSED")


if __name__ == "__main__":
    main()
