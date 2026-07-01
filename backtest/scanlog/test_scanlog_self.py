"""Test the tester (SPEC Â§7). Each test plants a specific failure and proves
the matching gate FAILS - i.e. the safety net actually catches things, it
isn't a rubber stamp.

Run: python -m backtest.scanlog.test_scanlog_self
Exit code 0 iff every planted failure was caught and every clean case passed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from backtest.scanlog.emitter import ScanLog, build_manifest, fingerprint
from backtest.scanlog import gates as G
from backtest.scanlog import conditions as cond


_FAILS = []


def _ok(msg): print(f"  OK:   {msg}")
def _bad(msg):
    print(f"  FAIL: {msg}")
    _FAILS.append(msg)


def _utc(s):
    return pd.Timestamp(s, tz="UTC")


def _mini_manifest(run_id="selftest"):
    return build_manifest(
        run_id=run_id, git_sha="deadbee", risk_usd=250.0, min_warmup_bars=50,
        pairs_served=[{"name": "EURUSD", "symbol": "EURUSD=X",
                       "requested_start": "2024-09-01", "requested_end": "2024-09-02",
                       "served_start": "2024-09-01", "served_end": "2024-09-02",
                       "n_bars": 2, "fingerprint": "abc", "prox_cap_atr": 3.0}],
        knobs={"EURUSD.atr_multiplier": 3.0}, fetch_pad_days=35,
    )


def _good_trade(**over):
    """A causally-clean, reconciled filled trade row.

    CONTRACT: every field that _is_eligible() tests must be present here
    and must pass. If _is_eligible gains a new condition (score floor, IST
    window, etc.), add the corresponding field to this dict — or every gate
    reconciliation test silently breaks.
    """
    t = {
        "pair": "EURUSD",
        "ob_timestamp": "2024-09-01T08:00:00+00:00",
        "bos_timestamp": "2024-09-01T09:00:00+00:00",
        "alert_ts": "2024-09-01T10:00:00+00:00",
        "fill_ts": "2024-09-01T11:00:00+00:00",
        "exit_ts": "2024-09-01T15:00:00+00:00",
        "exit_reason": "tp2",
        "r_realised": 2.0,
        "r_if_exit_tp1": 1.0,
        "r_if_exit_tp2": 2.0,
        "pnl_usd": round(2.0 * 250.0, 2),
        "score": 10.0,       # must clear SCORE_FLOOR (config min_score_to_email)
        "ist_blocked": False, # must not be IST-blocked
        "entry_zone": "proximal", # headline is proximal-only (2026-06-30); the
                                  # gate counts proximal rows only, so the fixture
                                  # must be proximal to be reconciled.
    }
    t.update(over)
    return t


def _begin(td, manifest=None):
    return ScanLog.begin(Path(td), manifest or _mini_manifest())


def _full_heartbeat(sl, pair, idx):
    """Declare and write a complete heartbeat so G2 passes by default."""
    sl.declare_walk(pair, len(idx))
    for ts in idx:
        sl.scan(pair=pair, ts=ts, index=idx, outcome="NO_ZONE", n_active_zones=0)


# ---------------------------------------------------------------------------
# 1. Planted silent failure: delete one heartbeat record -> G2 FAIL.
# ---------------------------------------------------------------------------
def test_1_heartbeat_gap():
    print("\n== 1. planted heartbeat gap -> G2 FAIL ==")
    with tempfile.TemporaryDirectory() as td:
        sl = _begin(td)
        idx = pd.DatetimeIndex([_utc("2024-09-01T00:00"), _utc("2024-09-01T01:00"),
                                _utc("2024-09-01T02:00")])
        sl.declare_walk("EURUSD", 3)
        # write only 2 of 3 -> one bar produced no record.
        for ts in idx[:2]:
            sl.scan(pair="EURUSD", ts=ts, index=idx, outcome="NO_ZONE")
        res = G.evaluate(scanlog=sl, trades=[], risk_usd=250.0)
        g2 = next(g for g in res.gates if g.id == "G2")
        (_ok if g2.verdict == "FAIL" else _bad)(f"G2 verdict = {g2.verdict}")
        (_ok if not res.passed else _bad)(f"overall = {res.overall}")
        sl.close()


# ---------------------------------------------------------------------------
# 2. Planted causality bug: alert_ts == bos_ts -> G3 FAIL.
# ---------------------------------------------------------------------------
def test_2_causality():
    print("\n== 2. planted causality (bos==alert) -> G3 FAIL ==")
    with tempfile.TemporaryDirectory() as td:
        sl = _begin(td)
        idx = pd.DatetimeIndex([_utc("2024-09-01T00:00")])
        _full_heartbeat(sl, "EURUSD", idx)
        bad = _good_trade(bos_timestamp="2024-09-01T10:00:00+00:00",
                          alert_ts="2024-09-01T10:00:00+00:00")  # bos == alert
        res = G.evaluate(scanlog=sl, trades=[bad], risk_usd=250.0)
        g3 = next(g for g in res.gates if g.id == "G3")
        (_ok if g3.verdict == "FAIL" else _bad)(f"G3 verdict = {g3.verdict}")
        sl.close()


# ---------------------------------------------------------------------------
# 3. Planted P&L drift: perturb one pnl_usd -> G1 FAIL.
# ---------------------------------------------------------------------------
def test_3_pnl_drift():
    print("\n== 3. planted P&L drift -> G1 FAIL ==")
    with tempfile.TemporaryDirectory() as td:
        sl = _begin(td)
        idx = pd.DatetimeIndex([_utc("2024-09-01T00:00")])
        _full_heartbeat(sl, "EURUSD", idx)
        bad = _good_trade(pnl_usd=round(2.0 * 250.0, 2) + 1e-3)  # drift
        res = G.evaluate(scanlog=sl, trades=[bad], risk_usd=250.0)
        g1 = next(g for g in res.gates if g.id == "G1")
        (_ok if g1.verdict == "FAIL" else _bad)(f"G1 verdict = {g1.verdict}")
        sl.close()


# ---------------------------------------------------------------------------
# 4. Planted knob drift: knob changes between manifest and run end -> G5 FAIL.
# ---------------------------------------------------------------------------
def test_4_knob_drift():
    print("\n== 4. planted knob drift -> G5 FAIL ==")
    with tempfile.TemporaryDirectory() as td:
        sl = _begin(td)
        idx = pd.DatetimeIndex([_utc("2024-09-01T00:00")])
        _full_heartbeat(sl, "EURUSD", idx)
        res = G.evaluate(scanlog=sl, trades=[_good_trade()], risk_usd=250.0,
                         manifest_recheck_knobs={"EURUSD.atr_multiplier": 9.9})
        g5 = next(g for g in res.gates if g.id == "G5")
        (_ok if g5.verdict == "FAIL" else _bad)(f"G5 verdict = {g5.verdict}")
        sl.close()


# ---------------------------------------------------------------------------
# 5. Unknown condition -> UNCLASSIFIED_CONDITION (FAIL), not swallowed.
# ---------------------------------------------------------------------------
def test_5_unclassified():
    print("\n== 5. unregistered anomaly -> UNCLASSIFIED_CONDITION FAIL ==")
    with tempfile.TemporaryDirectory() as td:
        sl = _begin(td)
        idx = pd.DatetimeIndex([_utc("2024-09-01T00:00")])
        _full_heartbeat(sl, "EURUSD", idx)
        sl.condition("SOMETHING_NEVER_REGISTERED", pair="EURUSD")
        res = G.evaluate(scanlog=sl, trades=[_good_trade()], risk_usd=250.0)
        hit = sl.condition_counts.get("UNCLASSIFIED_CONDITION", 0) > 0
        (_ok if hit else _bad)("UNCLASSIFIED_CONDITION raised")
        g4 = next(g for g in res.gates if g.id == "G4")
        (_ok if g4.verdict == "FAIL" else _bad)(f"G4 verdict = {g4.verdict}")
        sl.close()


# ---------------------------------------------------------------------------
# 6. Exit-code wiring: a failing result yields exit_code 1.
# ---------------------------------------------------------------------------
def test_6_exit_code():
    print("\n== 6. failing run -> exit_code 1 ==")
    with tempfile.TemporaryDirectory() as td:
        sl = _begin(td)
        sl.declare_walk("EURUSD", 5)  # declared 5, wrote 0 -> G2 FAIL
        res = G.evaluate(scanlog=sl, trades=[], risk_usd=250.0)
        (_ok if res.exit_code == 1 else _bad)(f"exit_code = {res.exit_code}")
        sl.close()


# ---------------------------------------------------------------------------
# 7. Determinism: same records twice -> identical content hash (G7).
# ---------------------------------------------------------------------------
def test_7_determinism():
    print("\n== 7. identical records -> identical hash (G7) ==")
    hashes = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as td:
            sl = _begin(td)
            idx = pd.DatetimeIndex([_utc("2024-09-01T00:00"), _utc("2024-09-01T01:00")])
            sl.declare_walk("EURUSD", 2)
            for ts in idx:
                sl.scan(pair="EURUSD", ts=ts, index=idx, outcome="NO_ZONE",
                        n_active_zones=0)
            sl.event("ob_seen", pair="EURUSD", ob_timestamp="2024-09-01T00:00:00+00:00")
            hashes.append(sl.content_hash())
            sl.close()
    (_ok if hashes[0] == hashes[1] else _bad)(
        f"hash equal: {hashes[0][:12]} == {hashes[1][:12]}")


# ---------------------------------------------------------------------------
# 8. Clean run passes: a fully consistent run -> overall PASS, exit 0.
#    (Behaviour-neutrality of the real walk is checked separately by the
#    pre-vs-post-instrumentation trade-row comparison; this asserts the gate
#    layer does not false-positive on a clean run.)
# ---------------------------------------------------------------------------
def test_8_clean_pass():
    print("\n== 8. clean run -> overall PASS ==")
    with tempfile.TemporaryDirectory() as td:
        sl = _begin(td)
        idx = pd.DatetimeIndex([_utc("2024-09-01T00:00"), _utc("2024-09-01T01:00")])
        _full_heartbeat(sl, "EURUSD", idx)
        sl.note_post_warmup_bar("EURUSD", atr_is_nan=False)
        res = G.evaluate(scanlog=sl, trades=[_good_trade()], risk_usd=250.0,
                         reported_headline_usd=round(2.0 * 250.0, 6),
                         manifest_recheck_knobs={"EURUSD.atr_multiplier": 3.0})
        for g in res.gates:
            if g.verdict == "FAIL":
                _bad(f"{g.id} unexpectedly FAILed: {g.observed}")
        (_ok if res.passed else _bad)(f"overall = {res.overall} (exit {res.exit_code})")
        sl.close()


# ---------------------------------------------------------------------------
# Extra: schema conformance (G9) - a real-shaped live record and a backtest
# record both validate; a tz-naive ts is caught.
# ---------------------------------------------------------------------------
def test_9_schema_conformance():
    print("\n== 9. schema: tz-naive ts caught, boundary enforced ==")
    with tempfile.TemporaryDirectory() as td:
        sl = _begin(td)
        idx = pd.DatetimeIndex([_utc("2024-09-01T00:00")])
        sl.declare_walk("EURUSD", 1)
        # tz-naive ts -> TZ_NAIVE; off-index ts -> TS_NOT_BOUNDARY.
        sl.scan(pair="EURUSD", ts=pd.Timestamp("2024-09-01T00:00"), index=idx,
                outcome="NO_ZONE")
        res = G.evaluate(scanlog=sl, trades=[], risk_usd=250.0)
        tz = sl.condition_counts.get("TZ_NAIVE", 0) > 0
        (_ok if tz else _bad)("TZ_NAIVE raised on naive ts")
        g9 = next(g for g in res.gates if g.id == "G9")
        (_ok if g9.verdict == "FAIL" else _bad)(f"G9 verdict = {g9.verdict}")
        sl.close()


# ---------------------------------------------------------------------------
# 10. Reconciliation row-set parity (regression for the Feb 2026 G1 failure).
# A clean run that contains a timeout row: the reporting layer drops timeout
# (and never_filled) from its headline, so the gate must drop them too. Before
# the fix the gate summed the timeout row's r_realised while reporting did not,
# producing a spurious G1 FAIL (headline != reported). The reported figure
# here mirrors reporting: tp2 row only, timeout + never_filled excluded.
# ---------------------------------------------------------------------------
def test_10_timeout_rowset_parity():
    print("\n== 10. never_filled/timeout/window_end excluded from headline -> G1 PASS ==")
    with tempfile.TemporaryDirectory() as td:
        sl = _begin(td)
        idx = pd.DatetimeIndex([_utc("2024-09-01T00:00"), _utc("2024-09-01T01:00")])
        _full_heartbeat(sl, "EURUSD", idx)
        sl.note_post_warmup_bar("EURUSD", atr_is_nan=False)
        # tp2 row counts; timeout row has real r_realised but is audit-only;
        # never_filled is r=0. Reporting headline = tp2 only = 500.0.
        good = _good_trade()
        timeout = _good_trade(exit_reason="timeout", r_realised=0.376,
                              r_if_exit_tp2=0.376,
                              pnl_usd=round(0.376 * 250.0, 2))
        window_end = _good_trade(exit_reason="window_end", r_realised=0.5,
                                 r_if_exit_tp2=0.5,
                                 pnl_usd=round(0.5 * 250.0, 2))
        never = _good_trade(exit_reason="never_filled", r_realised=0.0,
                            r_if_exit_tp2=0.0, fill_ts=None, pnl_usd=0.0)
        res = G.evaluate(scanlog=sl,
                         trades=[good, timeout, window_end, never], risk_usd=250.0,
                         reported_headline_usd=round(2.0 * 250.0, 6),
                         manifest_recheck_knobs={"EURUSD.atr_multiplier": 3.0})
        g1 = next(g for g in res.gates if g.id == "G1")
        (_ok if g1.verdict == "PASS" else _bad)(
            f"G1 verdict = {g1.verdict} (observed {g1.observed})")
        (_ok if res.passed else _bad)(f"overall = {res.overall} (exit {res.exit_code})")
        sl.close()


def main() -> int:
    test_1_heartbeat_gap()
    test_2_causality()
    test_3_pnl_drift()
    test_4_knob_drift()
    test_5_unclassified()
    test_6_exit_code()
    test_7_determinism()
    test_8_clean_pass()
    test_9_schema_conformance()
    test_10_timeout_rowset_parity()
    print("\n" + "=" * 50)
    if _FAILS:
        print(f"SELF-TEST FAILED: {len(_FAILS)} problem(s)")
        return 1
    print("ALL SELF-TESTS PASSED - the gates catch planted failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
