"""TRUTH LEDGER GATE (TRUTH_FIXES_SPEC.md T3).

Every trades.csv column must have a row in TRUTH_LEDGER.md (source file:line,
when stamped, population, status). This test makes the CLAUDE.md rule
mechanical: a new column added to the CSV writer without a ledger entry turns
CI red instead of shipping unaudited.

Rides `front_cols` in backtest/h1_only_reporting.py — the writer's own
canonical column order. Columns outside front_cols land via the `rest`
catch-all; those are governed by the alert-time-view classification comment in
h1_only_simulator (accepted residual, documented in the ledger).
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_every_csv_column_has_a_ledger_row():
    ledger = (_ROOT / "TRUTH_LEDGER.md").read_text(encoding="utf-8")
    src = (_ROOT / "backtest" / "h1_only_reporting.py").read_text(encoding="utf-8")

    m = re.search(r"front_cols = \[(.*?)\]", src, re.S)
    assert m, "front_cols list not found in h1_only_reporting.py — writer moved; update this test"

    cols = re.findall(r'"([a-z0-9_]+)"', m.group(1))
    assert len(cols) > 40, f"front_cols parse looks wrong (only {len(cols)} names)"

    missing = [c for c in cols if c not in ledger]
    assert not missing, (
        "trades.csv columns missing a TRUTH_LEDGER.md row: "
        f"{missing}. Rule (CLAUDE.md, Logging): no new metric ships without a "
        "ledger entry + a class-killing guard."
    )


if __name__ == "__main__":
    test_every_csv_column_has_a_ledger_row()
    print("truth-ledger gate: OK")
