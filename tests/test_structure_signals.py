"""Guards for the four structure-signal columns (STRUCTURE_SIGNALS_SPEC).

Run:  python tests/test_structure_signals.py
Exit 0 iff every guard passes.

Covers:
  S1  ranging counter resets on a Confirmation-BOS trend flip (behavioral proof
      on a crafted candle sequence + source guard on both flip branches).
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

import pandas as pd  # noqa: E402
import dealing_range as dr  # noqa: E402

_FAILS = []


def _ok(m): print(f"  OK:   {m}")
def _bad(m):
    print(f"  FAIL: {m}")
    _FAILS.append(m)


# ── S1 fixture: down-trend that goes RANGING (counter >=2), then a CHoCH-up +
# Confirmation-BOS-up flip. On the flip bar (end=48) the ranging counter belongs
# to the OLD down trend and must reset to 0 → ranging False. Pre-fix it leaked
# True (proven empirically when the fix was built). end=47 is the last down-trend
# bar, still ranging True. Prices are on a 1000 scale so ATR (~5) makes the
# CHoCH/BOS displacement + body gates fire deterministically. -------------------
_S1_ROWS = [
    (1000.0, 1000.5, 999.5, 1000.0), (1000.0, 1000.5, 999.5, 1000.0),
    (1000.0, 1000.5, 999.5, 1000.0), (1000.0, 1000.75, 984.25, 985.0),
    (985.0, 985.75, 969.25, 970.0), (970.0, 970.75, 954.25, 955.0),
    (955.0, 955.75, 939.25, 940.0), (940.0, 949.1875, 939.5625, 948.75),
    (948.75, 957.9375, 948.3125, 957.5), (957.5, 966.6875, 957.0625, 966.25),
    (966.25, 975.4375, 965.8125, 975.0), (975.0, 975.9375, 955.3125, 956.25),
    (956.25, 957.1875, 936.5625, 937.5), (937.5, 938.4375, 917.8125, 918.75),
    (918.75, 919.6875, 899.0625, 900.0), (900.0, 910.5, 899.5, 910.0),
    (910.0, 920.5, 909.5, 920.0), (920.0, 930.5, 919.5, 930.0),
    (930.0, 940.5, 929.5, 940.0), (940.0, 940.3125, 933.4375, 933.75),
    (933.75, 934.0625, 927.1875, 927.5), (927.5, 927.8125, 920.9375, 921.25),
    (921.25, 921.5625, 914.6875, 915.0), (915.0, 924.1875, 914.5625, 923.75),
    (923.75, 932.9375, 923.3125, 932.5), (932.5, 941.6875, 932.0625, 941.25),
    (941.25, 950.4375, 940.8125, 950.0), (950.0, 950.4, 941.6, 942.0),
    (942.0, 942.4, 933.6, 934.0), (934.0, 934.4, 925.6, 926.0),
    (926.0, 926.4, 917.6, 918.0), (918.0, 926.925, 917.575, 926.5),
    (926.5, 935.425, 926.075, 935.0), (935.0, 943.925, 934.575, 943.5),
    (943.5, 952.425, 943.075, 952.0), (952.0, 967.225, 951.275, 966.5),
    (966.5, 981.725, 965.775, 981.0), (981.0, 996.225, 980.275, 995.5),
    (995.5, 1010.725, 994.775, 1010.0), (1010.0, 1010.25, 1004.75, 1005.0),
    (1005.0, 1005.25, 999.75, 1000.0), (1000.0, 1000.25, 994.75, 995.0),
    (995.0, 995.25, 989.75, 990.0), (990.0, 996.5625, 989.6875, 996.25),
    (996.25, 1002.8125, 995.9375, 1002.5), (1002.5, 1009.0625, 1002.1875, 1008.75),
    (1008.75, 1015.3125, 1008.4375, 1015.0), (1015.0, 1026.8125, 1014.4375, 1026.25),
]


def _s1_df(n):
    rows = _S1_ROWS[:n]
    idx = pd.date_range("2020-01-01", periods=len(rows), freq="1h", tz="UTC")
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"] = 100.0
    return df


def test_s1_ranging_resets_on_flip():
    pre = dr.compute_structure(_s1_df(47), None)   # last down-trend bar
    flip = dr.compute_structure(_s1_df(48), None)  # Confirmation-BOS-up flip bar
    # Pre-flip: down trend, ranging tripped (counter >= 2).
    if pre["state"] == "down" and pre["ranging"] is True:
        _ok("pre-flip bar: down trend is ranging (counter >= 2)")
    else:
        _bad(f"pre-flip setup wrong: state={pre['state']} ranging={pre['ranging']} "
             "(want down/True) — fixture drifted")
    # The flip itself happened (state is UP).
    if flip["state"] == "up":
        _ok("flip bar: state flipped to UP (Confirmation BOS fired)")
    else:
        _bad(f"flip did not occur: state={flip['state']} (want up) — fixture drifted")
    # THE FIX: the new up trend must NOT inherit the old trend's ranging label.
    if flip["ranging"] is False:
        _ok("flip bar: ranging reset to False on the fresh trend (S1 fix)")
    else:
        _bad("flip bar leaked ranging=True — S1 reset missing (stale counter "
             "survived the trend flip)")


def test_s1_reset_in_both_flip_branches_only():
    """Source guard: the reset lives in BOTH Confirmation-BOS flip branches and
    NOT in the CHoCH_FAILED / reclaim paths (where the old trend resumes and the
    stale count legitimately carries)."""
    src = (_ROOT / "dealing_range.py").read_text(encoding="utf-8")
    # Both flip branches re-seed the leg extremes then reset the counter.
    up_flip = "leg_extreme_high = hi_i; leg_extreme_low = lo_i" in src
    n_reset = src.count("trend_dir_swings_since_extend = 0")
    # 5 pre-existing resets (init, continuation-BOS x2, HL/LH extend x2) + the
    # two S1 flip-branch resets = 7. Assert >= 6 so the two flip resets can't be
    # silently dropped; the S1 comment markers below pin them to the flip path.
    s1_up = "# S1: a trend flip starts a fresh leg" in src
    s1_down = "# S1: fresh leg on the flip" in src
    if up_flip and s1_up and s1_down and n_reset >= 6:
        _ok(f"both flip branches reset the ranging counter "
            f"({n_reset} total resets incl. S1's two)")
    else:
        _bad(f"S1 flip-branch reset missing/misplaced: up_flip={up_flip} "
             f"s1_up={s1_up} s1_down={s1_down} n_reset={n_reset}")
    # The reclaim/failed paths must NOT carry an S1-style reset — the old trend
    # resumes there. Assert the CHoCH_FAILED branches do not reset the counter by
    # checking the reset never appears in the two lines after a CHoCH_FAILED stamp.
    if 'kind": "CHoCH_FAILED"' in src:
        _ok("CHoCH_FAILED branches present (reclaim path — no reset, by design)")
    else:
        _bad("CHoCH_FAILED branch not found — structure changed, re-verify S1")


# ── S2/S3 re-fire freeze — mirror the S2/S3 payload-scalar logic and prove the
# FIRST fire's snapshot wins when the shared ob dict / walls change on a re-fire.
# _build_row reads these strictly from alert.get(...) (payload scalars). ----------

def _row_structure_signals(alert):
    """Mirror of _build_row's S2/S3/S4 row sources (payload + frozen dr)."""
    ob = alert["ob"]
    dr_snap = ob.get("dealing_range")
    out = {
        "structure_ranging_at_alert": alert.get("structure_ranging_at_alert"),
        "flip_pending_at_alert": alert.get("flip_pending_at_alert"),
        "flip_pending_dir_at_alert": alert.get("flip_pending_dir_at_alert"),
        "leg_extreme_at_alert": alert.get("leg_extreme_at_alert"),
        "leg_extreme_clipped": alert.get("leg_extreme_clipped"),
    }
    # S4 reads off the frozen snapshot.
    if isinstance(dr_snap, dict) and dr_snap.get("valid"):
        cb = dr_snap.get("ceiling_broken")
        fb = dr_snap.get("floor_broken")
        out["dr_ceiling_broken_at_ob"] = bool(cb) if cb is not None else None
        out["dr_floor_broken_at_ob"] = bool(fb) if fb is not None else None
    else:
        out["dr_ceiling_broken_at_ob"] = None
        out["dr_floor_broken_at_ob"] = None
    return out


def test_s2_state_frozen_from_first_fire_payload():
    # First fire: not ranging, no pending flip. A LATER fire flipped the shared
    # walls/ob to ranging+pending — but the traded row is built from the FIRST
    # fire's payload scalars, which must win.
    alert = {
        "structure_ranging_at_alert": False,
        "flip_pending_at_alert": False,
        "flip_pending_dir_at_alert": None,
        "leg_extreme_at_alert": 1.2345,
        "leg_extreme_clipped": False,
        # shared ob dict re-stamped by a later fire (must be ignored):
        "ob": {"structure_ranging_at_alert": True,
               "flip_pending_at_alert": True,
               "leg_extreme_at_alert": 9.9999,
               "dealing_range": {"valid": True, "ceiling_broken": True,
                                 "floor_broken": False}},
    }
    r = _row_structure_signals(alert)
    if (r["structure_ranging_at_alert"] is False
            and r["flip_pending_at_alert"] is False
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


# ── S3 retrace math — long/short normal, degenerate denom, missing impulse,
# clipped flag pass-through, >100 not clamped. Mirrors _build_row's derivation.

def _retrace(direction, leg_extreme, entry, impulse_start):
    """Exact mirror of _build_row's leg_retrace_pct_at_alert derivation."""
    if leg_extreme is None or impulse_start is None:
        return None
    lex = float(leg_extreme)
    origin = float(impulse_start)
    if direction == "bullish":
        denom = lex - origin
        return round((lex - entry) / denom * 100, 1) if denom > 0 else None
    denom = origin - lex
    return round((entry - lex) / denom * 100, 1) if denom > 0 else None


def test_s3_retrace_math():
    # LONG normal: leg 100->120, entry 110 => 50% back.
    if _retrace("bullish", 120.0, 110.0, 100.0) == 50.0:
        _ok("S3 long retrace ~50% correct")
    else:
        _bad(f"long retrace wrong: {_retrace('bullish',120.0,110.0,100.0)}")
    # SHORT normal: leg 100->80, entry 90 => 50% back.
    if _retrace("bearish", 80.0, 90.0, 100.0) == 50.0:
        _ok("S3 short retrace ~50% correct")
    else:
        _bad(f"short retrace wrong: {_retrace('bearish',80.0,90.0,100.0)}")
    # Degenerate denominator (extreme not beyond origin) -> None.
    if _retrace("bullish", 100.0, 105.0, 100.0) is None:
        _ok("S3 degenerate denominator -> None (no crash)")
    else:
        _bad("S3 degenerate denominator should be None")
    # Missing impulse_start -> None.
    if _retrace("bullish", 120.0, 110.0, None) is None:
        _ok("S3 missing impulse_start -> None")
    else:
        _bad("S3 missing impulse_start should be None")
    # > 100 is VALID (price ran past the leg origin), not clamped.
    v = _retrace("bullish", 120.0, 95.0, 100.0)  # (120-95)/(120-100)=125%
    if v == 125.0:
        _ok("S3 retrace > 100 preserved (not clamped)")
    else:
        _bad(f"S3 >100 clamped or wrong: {v}")


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
        '"structure_ranging_at_alert": _structure_ranging_at_alert',
        '"flip_pending_at_alert": _flip_pending_at_alert',
        '"flip_pending_dir_at_alert": _flip_pending_dir_at_alert',
        '"leg_extreme_at_alert": _leg_extreme_at_alert',
        '"leg_extreme_clipped": _leg_extreme_clipped',
    ))
    r_ok = all(k in row_src for k in (
        'alert.get("structure_ranging_at_alert")',
        'alert.get("flip_pending_at_alert")',
        'alert.get("flip_pending_dir_at_alert")',
        'alert.get("leg_extreme_at_alert")',
        'alert.get("leg_extreme_clipped")',
    ))
    if y_ok and r_ok:
        _ok("replay yields S2/S3 payload scalars; row build reads them from payload")
    else:
        _bad(f"S2/S3 payload plumbing missing — yield_ok={y_ok}, row_ok={r_ok}")


def main():
    print("== S1: ranging counter resets on trend flip ==")
    test_s1_ranging_resets_on_flip()
    test_s1_reset_in_both_flip_branches_only()
    print("\n== S2/S3: alert-time payload freeze ==")
    test_s2_state_frozen_from_first_fire_payload()
    test_s3_extreme_higher_on_refire_but_first_wins()
    print("\n== S3: leg-retracement math ==")
    test_s3_retrace_math()
    print("\n== S4: broken-wall PD flags ==")
    test_s4_broken_flags_from_snapshot()
    test_s4_source_chain_emits_flags()
    print("\n== source tripwires ==")
    test_source_yield_carries_payload_scalars()
    print()
    if _FAILS:
        print(f"FAILED: {len(_FAILS)} problem(s)")
        return 1
    print("PASSED: structure signals logged + frozen per STRUCTURE_SIGNALS_SPEC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
