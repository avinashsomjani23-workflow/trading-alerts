"""Guards for the four structure-signal columns (STRUCTURE_SIGNALS_SPEC).

Run:  python tests/test_structure_signals.py
Exit 0 iff every guard passes.

Covers:
  S2  structure state snapshotted at alert as PAYLOAD scalars (re-fire freeze +
      source tripwire) — kills the last-fire-stamp bug class for these columns.
  S3  leg-retracement math (long/short/degenerate/missing/clipped/>100) + the
      extreme-at-alert re-fire freeze.
  S4  broken-wall PD flags land in the frozen ob["dealing_range"] snapshot and
      survive to the row (broken vs intact).

Style mirrors tests/test_ob_alert_freeze.py: a plain assertion harness, no
pytest dependency, source tripwires that turn a silent revert red.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest.h1_only_simulator import (  # noqa: E402
    read_s4_broken_flags,
)

_FAILS = []


def _ok(m): print(f"  OK:   {m}")
def _bad(m):
    # RAISE, don't just collect: CI runs these via `pytest tests/ -q`, which
    # never calls main(). A print-and-append _bad is invisible to pytest -> the
    # guard is green even when the code is broken (Deep Value A, 2026-07-10).
    print(f"  FAIL: {m}")
    _FAILS.append(m)
    raise AssertionError(m)


# ── S2/S3 re-fire freeze — mirror the S2/S3 payload-scalar logic and prove the
# FIRST fire's snapshot wins when the shared ob dict / walls change on a re-fire.
# _build_row reads these strictly from alert.get(...) (payload scalars). ----------

def _row_structure_signals(alert):
    """Reads S2/S3/S4 the way _build_row does. The S2/S3 payload passthroughs are
    the live expression verbatim (alert.get, no logic to rot); the S4 flags go
    through the LIVE read_s4_broken_flags helper — so an S4-logic drift trips this
    test, not a private copy."""
    ob = alert["ob"]
    cb, fb = read_s4_broken_flags(ob.get("dealing_range"))
    return {
        "flip_pending_at_alert": alert.get("flip_pending_at_alert"),
        "flip_pending_dir_at_alert": alert.get("flip_pending_dir_at_alert"),
        "leg_extreme_at_alert": alert.get("leg_extreme_at_alert"),
        "leg_extreme_clipped": alert.get("leg_extreme_clipped"),
        "dr_ceiling_broken_at_ob": cb,
        "dr_floor_broken_at_ob": fb,
    }


def test_s2_state_frozen_from_first_fire_payload():
    # First fire: no pending flip. A LATER fire flipped the shared walls/ob to
    # pending — but the traded row is built from the FIRST fire's payload
    # scalars, which must win.
    alert = {
        "flip_pending_at_alert": False,
        "flip_pending_dir_at_alert": None,
        "leg_extreme_at_alert": 1.2345,
        "leg_extreme_clipped": False,
        # shared ob dict re-stamped by a later fire (must be ignored):
        "ob": {"flip_pending_at_alert": True,
               "leg_extreme_at_alert": 9.9999,
               "dealing_range": {"valid": True, "ceiling_broken": True,
                                 "floor_broken": False}},
    }
    r = _row_structure_signals(alert)
    if (r["flip_pending_at_alert"] is False
            and r["leg_extreme_at_alert"] == 1.2345):
        _ok("S2/S3: row reads the FIRST fire's payload, not the re-stamped dict")
    else:
        _bad(f"re-fire poisoned the row: {r}")


def test_s3_extreme_higher_on_refire_but_first_wins():
    # The spec's freeze case: second fire sees a higher extreme; first row keeps
    # the first-fire value (payload scalar).
    first = {"leg_extreme_at_alert": 1.2000, "ob": {"leg_extreme_at_alert": 1.5000}}
    if _row_structure_signals(first)["leg_extreme_at_alert"] == 1.2000:
        _ok("S3: leg_extreme_at_alert holds the first-fire value under re-fire")
    else:
        _bad("S3: leg_extreme_at_alert drifted to a later fire's higher extreme")


# ── S4 snapshot — the frozen dr snapshot carries the broken flags; row reads
# them; intact range => both False; invalid/legacy => None.

def test_s4_broken_flags_from_snapshot():
    broken = {"ob": {"dealing_range": {"valid": True, "ceiling_broken": True,
                                       "floor_broken": False}}}
    r = _row_structure_signals(broken)
    if r["dr_ceiling_broken_at_ob"] is True and r["dr_floor_broken_at_ob"] is False:
        _ok("S4: broken ceiling / intact floor read off the frozen snapshot")
    else:
        _bad(f"S4 broken-flag read wrong: {r}")
    intact = {"ob": {"dealing_range": {"valid": True, "ceiling_broken": False,
                                       "floor_broken": False}}}
    r2 = _row_structure_signals(intact)
    if r2["dr_ceiling_broken_at_ob"] is False and r2["dr_floor_broken_at_ob"] is False:
        _ok("S4: intact range -> both flags False")
    else:
        _bad(f"S4 intact-range read wrong: {r2}")
    legacy = {"ob": {"dealing_range": {"valid": False}}}
    r3 = _row_structure_signals(legacy)
    if r3["dr_ceiling_broken_at_ob"] is None and r3["dr_floor_broken_at_ob"] is None:
        _ok("S4: invalid/legacy snapshot -> both flags None")
    else:
        _bad(f"S4 legacy read wrong: {r3}")


def test_s4_source_chain_emits_flags():
    """Source guard: get_dealing_range's valid branch emits the additive broken
    flags from compute_pd_position's *_is_placeholder. Reverting re-drops them."""
    src = (_ROOT / "smc_detector.py").read_text(encoding="utf-8")
    if ('"ceiling_broken": bool(pd_info.get("ceiling_is_placeholder"' in src
            and '"floor_broken":   bool(pd_info.get("floor_is_placeholder"' in src):
        _ok("S4: get_dealing_range emits ceiling_broken/floor_broken (additive)")
    else:
        _bad("S4: broken-flag emission missing from get_dealing_range")


# ── Tripwires: the replay yield carries the S2/S3 payload scalars; the row build
# reads them from alert.get(...). Reverting either re-opens the last-fire class.

def test_source_yield_carries_payload_scalars():
    yield_src = (_ROOT / "backtest" / "replay_engine.py").read_text(encoding="utf-8")
    row_src = (_ROOT / "backtest" / "h1_only_simulator.py").read_text(encoding="utf-8")
    y_ok = all(k in yield_src for k in (
        '"flip_pending_at_alert": _flip_pending_at_alert',
        '"flip_pending_dir_at_alert": _flip_pending_dir_at_alert',
        '"leg_extreme_at_alert": _leg_extreme_at_alert',
        '"leg_extreme_clipped": _leg_extreme_clipped',
    ))
    r_ok = all(k in row_src for k in (
        'alert.get("flip_pending_at_alert")',
        'alert.get("flip_pending_dir_at_alert")',
        'alert.get("leg_extreme_at_alert")',
        'alert.get("leg_extreme_clipped")',
    ))
    if y_ok and r_ok:
        _ok("replay yields S2/S3 payload scalars; row build reads them from payload")
    else:
        _bad(f"S2/S3 payload plumbing missing — yield_ok={y_ok}, row_ok={r_ok}")


# ── TREND-ALIGNMENT PARITY (LIVE_BUGS_FIX_SPEC Task 1) ─────────────────────────
# derive_trend_alignment is the ONE implementation both the live Phase 2 engine
# and the backtest replay call. Before it existed the two paths used different
# branch logic AND a different vocabulary, so the edge engine trained on a label
# live would never emit. These guards pin the branch table and forbid a silent
# re-fork of the logic in either caller.

import smc_detector as _smc  # noqa: E402


def test_trend_alignment_branch_table():
    B, R = "bullish", "bearish"
    # (zone_dir, trend, flip_pending, flip_pending_dir) -> expected
    cases = [
        # 1. flip pending, zone opposes the pending flip -> counter_trend
        #    (the EURUSD 2026-06-29 bug: SHORT zone, bullish CHoCH pending)
        ((R, B, True, B), "counter_trend"),
        ((B, R, True, R), "counter_trend"),
        # 2. flip pending, zone matches the pending flip -> ambiguous (unconfirmed,
        #    no with-trend credit even though raw trend still reads the old dir)
        ((B, R, True, B), "ambiguous"),
        ((R, B, True, R), "ambiguous"),
        # flip pending but dir unknown (None) still demotes: zone != None -> counter
        ((B, B, True, None), "counter_trend"),
        # 3. no flip pending, trend unknown -> ambiguous
        ((B, None, False, None), "ambiguous"),
        ((R, None, False, None), "ambiguous"),
        # 4. no flip pending, trend == zone -> with_trend
        ((B, B, False, None), "with_trend"),
        ((R, R, False, None), "with_trend"),
        # 5. no flip pending, trend != zone -> counter_trend
        ((B, R, False, None), "counter_trend"),
        ((R, B, False, None), "counter_trend"),
    ]
    bad = []
    for (zone_dir, trend, fp, fpd), want in cases:
        got = _smc.derive_trend_alignment(zone_dir, trend, fp, fpd)
        if got != want:
            bad.append(f"({zone_dir},{trend},fp={fp},{fpd}) -> {got!r} != {want!r}")
    if bad:
        _bad("derive_trend_alignment branch table: " + "; ".join(bad))
    else:
        _ok(f"derive_trend_alignment branch table ({len(cases)} cases, incl. both "
            "flip-pending demotions)")


def test_trend_alignment_matches_live_phase2_branches():
    # Independent re-implementation of the live Phase2 branch ORDER (the exact
    # if/elif chain at Phase2_Alert_Engine.py) — asserts the helper reproduces it
    # for every combination, so a future edit to either can't drift apart silently.
    def live_rule(zone_dir, trend, fp, fpd):
        if fp and zone_dir != fpd:
            return "counter_trend"
        if fp:
            return "ambiguous"
        if trend is None:
            return "ambiguous"
        if trend == zone_dir:
            return "with_trend"
        return "counter_trend"
    dirs = ["bullish", "bearish"]
    mism = []
    for zd in dirs:
        for tr in dirs + [None]:
            for fp in (True, False):
                for fpd in dirs + [None]:
                    a = _smc.derive_trend_alignment(zd, tr, fp, fpd)
                    b = live_rule(zd, tr, fp, fpd)
                    if a != b:
                        mism.append(f"({zd},{tr},{fp},{fpd}): {a}!={b}")
    if mism:
        _bad("helper diverges from live branch order: " + "; ".join(mism))
    else:
        _ok("helper == live Phase2 branch order across all combinations")


def test_single_implementation_of_trend_alignment():
    # SINGLE SOURCE OF TRUTH guard. Both callers must call the shared helper, and
    # neither may re-derive the label locally (the old backtest fork emitted
    # against_trend/no_trend — those value strings must be gone from code).
    p2  = (_ROOT / "Phase2_Alert_Engine.py").read_text(encoding="utf-8")
    rep = (_ROOT / "backtest" / "replay_engine.py").read_text(encoding="utf-8")
    calls_ok = ("smc_detector.derive_trend_alignment(" in p2
                and "smc_detector.derive_trend_alignment(" in rep)
    # The old backtest vocabulary must not appear as emitted values anywhere in
    # code (comments/markdown are out of the .py surface these tests read).
    no_old_vocab = ('"against_trend"' not in rep and '"no_trend"' not in rep
                    and "'against_trend'" not in rep and "'no_trend'" not in rep)
    if calls_ok and no_old_vocab:
        _ok("both callers use derive_trend_alignment; old backtest vocabulary gone")
    else:
        _bad(f"single-impl guard — calls_ok={calls_ok}, no_old_vocab={no_old_vocab}")


def test_no_spent_swing_break_in_golden_fixtures():
    """Spent-swing guard (dealing_range._already_broken): NO CHoCH/BOS in any
    committed golden fixture may (a) break a swing already broken by an earlier
    event [re-fire] or (b) break a level a prior candle already BODY-closed
    through [spent]. Locks out the bug proven live on EURUSD 2026-08-10 (two
    CHoCHs fired on the same 1.15558 low). Data-driven over real windows, so a
    regen with buggy code re-introduces a violation and turns this red."""
    import glob
    import json as _json
    fdir = _ROOT / "backtest" / "structure_golden" / "fixtures"
    refires = spent = 0
    for p in sorted(glob.glob(str(fdir / "*.json"))):
        with open(p, encoding="utf-8") as fh:
            fx = _json.load(fh)
        rows = fx["input_rows"]
        idx = {r["ts"]: i for i, r in enumerate(rows)}
        C = [r["Close"] for r in rows]
        seen = {}
        for e in fx["golden_output"].get("events", []):
            if e.get("type") not in ("CHoCH", "BOS"):
                continue
            st = e.get("broken_swing_ts")
            seen[st] = seen.get(st, 0) + 1
            P = e.get("broken_swing_price")
            bi = idx.get(e.get("candle_ts"))
            si = idx.get(st)
            if P is None or bi is None or si is None or si + 1 >= bi:
                continue
            seg = C[si + 1:bi]
            if e["direction"] == "bearish" and any(x < P for x in seg):
                spent += 1
            elif e["direction"] == "bullish" and any(x > P for x in seg):
                spent += 1
        refires += sum(1 for k, v in seen.items() if k and v > 1)
    if refires == 0 and spent == 0:
        _ok("no re-fire / no spent break across all golden fixtures")
    else:
        _bad(f"spent-swing guard breached: {refires} re-fire(s), {spent} "
             f"spent-break(s) — a break fired on an already-broken swing")


def main():
    print("== S2/S3: alert-time payload freeze ==")
    test_s2_state_frozen_from_first_fire_payload()
    test_s3_extreme_higher_on_refire_but_first_wins()
    print("\n== S4: broken-wall PD flags ==")
    test_s4_broken_flags_from_snapshot()
    test_s4_source_chain_emits_flags()
    print("\n== source tripwires ==")
    test_source_yield_carries_payload_scalars()
    print("\n== trend-alignment parity (LIVE_BUGS_FIX_SPEC Task 1) ==")
    test_trend_alignment_branch_table()
    test_trend_alignment_matches_live_phase2_branches()
    test_single_implementation_of_trend_alignment()
    print("\n== spent-swing guard (no break on an already-broken swing) ==")
    test_no_spent_swing_break_in_golden_fixtures()
    print()
    if _FAILS:
        print(f"FAILED: {len(_FAILS)} problem(s)")
        return 1
    print("PASSED: structure signals logged + frozen per STRUCTURE_SIGNALS_SPEC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
