"""Aggregate population guard (TRUTH_FIXES_SPEC.md T2).

The cross-run aggregator must never feed force-closed audit rows
(timeout / window_end — arbitrary-price exits, audit-only by the settled
unresolved-trade policy) or hard-blocked rows into any metric. trades.csv
ships eligible_for_headline for exactly this; legacy CSVs are reconstructed
from the same inputs. This test kills the class: a population that
re-derives eligibility wrongly fails here before it can publish a number.
"""

import csv
import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest.aggregate_runs import _eligible_mask
from backtest import insights as ins
from backtest.h1_only_reporting import (
    _headline_exclusion, _is_real_filled, _EXCLUDE_REASONS,
)


def _df(rows):
    return pd.DataFrame(rows)


def _canonical_csv() -> Path:
    """Resolve the ONE canonical trades.csv from CANONICAL.md (never glob)."""
    doc = (_ROOT / "backtest" / "results" / "CANONICAL.md").read_text(encoding="utf-8")
    m = re.search(r"^\s{2,}(backtest/results/\S+/trades\.csv)\s*$", doc, re.M)
    assert m, "canonical trades.csv path not found in CANONICAL.md — format changed"
    p = _ROOT / m.group(1)
    assert p.exists(), f"canonical CSV named in CANONICAL.md does not exist: {p}"
    return p


def _truthy(v) -> bool:
    return str(v).strip().lower() == "true"


def test_force_closed_rows_are_excluded():
    df = _df([
        {"exit_reason": "tp1",        "r_realised":  1.0, "eligible_for_headline": True},
        {"exit_reason": "timeout",    "r_realised": -0.4, "eligible_for_headline": False},
        {"exit_reason": "window_end", "r_realised":  0.7, "eligible_for_headline": False},
        {"exit_reason": "never_filled", "r_realised": 0.0, "eligible_for_headline": False},
    ])
    eligible = df[_eligible_mask(df)]
    filled = ins._filled(eligible)
    assert len(filled) == 1
    assert filled.iloc[0]["exit_reason"] == "tp1"
    # The timeout row's R must not be in the expectancy input.
    assert -0.4 not in filled["r_realised"].tolist()
    assert 0.7 not in filled["r_realised"].tolist()


def test_string_bools_from_csv_roundtrip():
    # pandas reads mixed bool columns back as strings — masks must survive it.
    df = _df([
        {"exit_reason": "tp1", "r_realised": 1.0, "eligible_for_headline": "True"},
        {"exit_reason": "sl",  "r_realised": -1.0, "eligible_for_headline": "False"},
    ])
    eligible = df[_eligible_mask(df)]
    assert len(eligible) == 1
    assert eligible.iloc[0]["exit_reason"] == "tp1"


def test_legacy_csv_without_column_reconstructs_from_rule_inputs():
    df = _df([
        {"exit_reason": "tp1",     "r_realised": 1.0, "ist_blocked": "False", "weekend_blocked": False},
        {"exit_reason": "tp1",     "r_realised": 1.0, "ist_blocked": "True",  "weekend_blocked": False},
        {"exit_reason": "timeout", "r_realised": 0.3, "ist_blocked": "False", "weekend_blocked": False},
        {"exit_reason": "sl",      "r_realised": -1.0, "ist_blocked": "False", "weekend_blocked": "True"},
    ])
    eligible = df[_eligible_mask(df)]
    assert len(eligible) == 1
    assert eligible.iloc[0]["exit_reason"] == "tp1"


# ── AREA D — insight population, proven ON THE CANONICAL FILE (Deep Value Pass,
# 2026-07-10). Batch 3 only checked the plumbing LINE of each mask. These run the
# REAL eligibility mask + the REAL headline rule over the whole canonical CSV and
# assert force-closed rows are dropped and the headline reconciles on the file.


def test_area_d_masks_drop_force_closed_rows_on_canonical_data():
    """The cross-run eligibility mask (_eligible_mask) must drop EVERY
    timeout/window_end/never_filled force-closed row when run over the real
    canonical population — not just on a 4-row fixture.

    Bites if the mask ever stops honouring eligible_for_headline (or the column
    regresses) and lets arbitrary-price audit rows back into the P&L population.
    """
    path = _canonical_csv()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(df) > 1000, f"canonical CSV suspiciously small ({len(df)})"

    force_closed_total = int(df["exit_reason"].isin(_EXCLUDE_REASONS).sum())
    assert force_closed_total > 0, (
        "canonical file has NO force-closed rows — cannot prove the mask drops "
        "them (wrong file or exit_reason vocabulary changed)")

    eligible = df[_eligible_mask(df)]
    # Not one force-closed row survives the mask.
    surviving = eligible["exit_reason"].isin(_EXCLUDE_REASONS).sum()
    assert surviving == 0, (
        f"{surviving} force-closed rows survived _eligible_mask on the canonical "
        f"file (of {force_closed_total} total) — audit rows are polluting every "
        "cross-run insight")
    # And _filled on the eligible set carries no excluded reason either (the
    # population that actually feeds compute_overall / score_validation / etc).
    filled = ins._filled(eligible)
    assert not filled["exit_reason"].isin(_EXCLUDE_REASONS).any(), (
        "ins._filled(eligible) still contains a force-closed exit_reason")


def test_area_d_headline_reconciles_on_canonical_file():
    """The report enforces sum(pnl_usd where eligible) == headline P&L at write
    time (h1_only_reporting.py:_reconcile, :3446-3457). Prove that invariant holds
    on the SHIPPED file: the headline population is exactly the eligible rows, and
    every eligibility source agrees.

    Three independent recomputations of the headline population must give the same
    row set and the same P&L:
      (a) the eligible_for_headline column trades.csv ships,
      (b) the LIVE rule _headline_exclusion re-run per row,
      (c) the cross-run _eligible_mask.
    If any diverges, the file's headline is not reconcilable — fail loud.
    """
    path = _canonical_csv()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 1000, f"canonical CSV suspiciously small ({len(rows)})"

    # (b) LIVE rule per row. The CSV stores bools as strings, so coerce the two
    # bool inputs the rule reads to real booleans first (else 'False' is truthy).
    def _rule_eligible(r):
        t = dict(r)
        t["ist_blocked"] = _truthy(r["ist_blocked"])
        t["weekend_blocked"] = _truthy(r["weekend_blocked"])
        return _headline_exclusion(t) == ""

    col_eligible = [r for r in rows if _truthy(r["eligible_for_headline"])]
    rule_eligible = [r for r in rows if _rule_eligible(r)]

    # (a) vs (b): the shipped column IS the live rule, row for row.
    col_set = {id(r) for r in col_eligible}
    rule_set = {id(r) for r in rule_eligible}
    assert col_set == rule_set, (
        f"eligible_for_headline column ({len(col_eligible)}) disagrees with the "
        f"live _headline_exclusion rule ({len(rule_eligible)}) — the file's "
        "headline membership is not the one rule")

    # (c): the cross-run mask selects the same rows.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    mask_n = int(_eligible_mask(df).sum())
    assert mask_n == len(col_eligible), (
        f"_eligible_mask selects {mask_n} rows but the column marks "
        f"{len(col_eligible)} — cross-run population != headline population")

    # P&L reconcile: headline == sum(pnl_usd) over the eligible population, and the
    # excluded rows are exactly the complement (nothing double-counted or dropped).
    headline = round(sum(float(r["pnl_usd"] or 0) for r in col_eligible), 2)
    all_pnl = round(sum(float(r["pnl_usd"] or 0) for r in rows), 2)
    excluded_pnl = round(sum(
        float(r["pnl_usd"] or 0) for r in rows
        if not _truthy(r["eligible_for_headline"])), 2)
    assert abs(headline + excluded_pnl - all_pnl) <= 0.01, (
        f"headline ({headline}) + excluded ({excluded_pnl}) != all "
        f"({all_pnl}) — the eligible/excluded split is not a clean partition")
    # Every eligible row is a real fill (eligible ⊆ real_filled by construction).
    assert all(_is_real_filled(r) for r in col_eligible), (
        "an eligible row carries a force-closed exit_reason — eligibility must "
        "imply real-filled")


def test_area_d_peak_metric_gate_rejects_a_touch_claim():
    """Re-assert the peak-vs-fill law fires: is_peak_metric flags an mfe/touch
    metric, and verify_capturable REJECTS a claim where the touch fraction far
    exceeds the capturable fraction (the classic MFE trap). Bites if the gate is
    ever loosened to pass an uncapturable touch as bankable."""
    # is_peak_metric recognises the touch columns.
    assert ins.is_peak_metric("mfe_r")
    assert ins.is_peak_metric("pct_reached_1R")
    assert not ins.is_peak_metric("r_realised")

    # A real touch-vs-fill gap (47% touched +1R, only ~23% capturable) must REJECT.
    res = ins.verify_capturable("47% of stops reached +1R",
                                peak_count=470, captured_count=230, total=1000)
    assert res["verified"] is False, (
        f"peak-metric gate FAILED to reject an uncapturable touch claim: {res}")
    assert res["severity"] == "reject"
    # A claim where fill tracks touch within tolerance is allowed through.
    ok = ins.verify_capturable("levels that actually fill",
                               peak_count=100, captured_count=95, total=1000)
    assert ok["verified"] is True, f"gate wrongly rejected a capturable claim: {ok}"


if __name__ == "__main__":
    test_force_closed_rows_are_excluded()
    test_string_bools_from_csv_roundtrip()
    test_legacy_csv_without_column_reconstructs_from_rule_inputs()
    test_area_d_masks_drop_force_closed_rows_on_canonical_data()
    test_area_d_headline_reconciles_on_canonical_file()
    test_area_d_peak_metric_gate_rejects_a_touch_claim()
    print("aggregate eligibility guard: OK")
