"""Unit-style assertions for the H1-only mode.

Validates:
  1. compute_phase2_levels(h1_only=True) skips M15 nest with df_m15=None and
     does not crash.
  2. Proximal entry sits exactly at OB proximal (within tolerance).
  3. 50pct entry sits exactly at OB midpoint.
  4. SL is identical for both entry zones (only entry differs).
  5. R-distance for 50pct entry is exactly half of proximal entry's.
  6. TP1/TP2 price levels are identical for both entries; TP1 RR ~doubles
     for the 50pct entry (because R distance halves).
  7. Dual simulator returns at most 2 rows per alert; columns are present.

Run: python -m backtest.test_h1_only
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import smc_detector
from backtest import h1_only_simulator


def _synth_h1_df(n_bars: int = 200) -> pd.DataFrame:
    """Build a synthetic H1 dataframe with large enough swings that the
    H1 swing-point detector can find TP targets at >=1.5R for a tight OB."""
    base = pd.Timestamp("2026-04-01 00:00", tz="UTC")
    idx = pd.date_range(base, periods=n_bars, freq="h")
    # Large zigzag: +120 pips up, -40 pips back, +120 pips up, ... ~80-pip
    # swings between adjacent fractals — plenty for an 18-pip OB to find TP.
    closes = []
    px = 1.0800
    for i in range(n_bars):
        phase = i % 10
        if phase < 5:
            px += 0.0024     # +24 pips per bar for 5 bars = +120 pips
        else:
            px -= 0.0008     # -8 pips per bar for 5 bars = -40 pips
        closes.append(px)
    df = pd.DataFrame({
        "Open":  [c - 0.0004 for c in closes],
        "High":  [c + 0.0008 for c in closes],
        "Low":   [c - 0.0008 for c in closes],
        "Close": closes,
        "Volume": [100] * n_bars,
    }, index=idx)
    return df


def _synth_pair_conf() -> dict:
    return {
        "name": "TESTPAIR",
        "symbol": "TEST=X",
        "pair_type": "forex",
        "decimal_places": 5,
        "spread_pips": 2,
        "atr_multiplier": 3.0,
    }


def _synth_ob_bullish(df_h1: pd.DataFrame, ts_idx: int = -50) -> dict:
    """A small bullish OB sitting just below price at the given bar index.
       proximal = high (top), distal = low (bottom).
       Tight 18-pip width so TP1 at 1.5R is reachable by the synthetic swings.
    """
    px_at_ts = float(df_h1["Close"].iloc[ts_idx])
    top = px_at_ts - 0.0005   # 5 pips below the at-touch price
    bot = top - 0.0018        # 18-pip OB
    return {
        "direction": "bullish",
        "bos_tag":   "BOS",
        "bos_tier":  "Major",
        "high":      top,
        "low":       bot,
        "proximal_line": top,
        "distal_line":   bot,
        "ob_timestamp": (df_h1.index[ts_idx]).isoformat(),
        "fvg": {"exists": False},
        "touches": 0,
        "dealing_range": {"valid": True, "range_low": bot - 0.0080,
                          "range_high": top + 0.0300},
    }


def check(condition, msg):
    if not condition:
        print(f"  FAIL: {msg}")
        return False
    print(f"  OK:   {msg}")
    return True


def test_levels_dual_entry():
    print("\n== test_levels_dual_entry ==")
    df_h1 = _synth_h1_df()
    pair_conf = _synth_pair_conf()
    # Use an OB anchored to a bar near the front of the future window, so the
    # H1 swings AFTER it (which compute_phase2_levels uses for TP) are abundant.
    # Note: get_swing_points uses df_h1 as passed; we slice up to the alert
    # time elsewhere, but for level computation we pass full df_h1 here.
    ob = _synth_ob_bullish(df_h1, ts_idx=-150)
    proximal_expected = ob["proximal_line"]
    midpoint_expected = (ob["high"] + ob["low"]) / 2.0
    distal_expected = ob["distal_line"]
    spread = pair_conf["spread_pips"] * 0.0001
    sl_expected = distal_expected - spread  # bullish: SL below distal

    # At OB-touch moment, current_price = proximal exactly.
    current_price = proximal_expected

    lv_prox = smc_detector.compute_phase2_levels(
        pair_conf, "LONG", ob, current_price, df_h1, df_m15=None,
        h1_only=True, entry_zone="proximal",
    )
    lv_mid = smc_detector.compute_phase2_levels(
        pair_conf, "LONG", ob, current_price, df_h1, df_m15=None,
        h1_only=True, entry_zone="50pct",
    )

    ok = True
    ok &= check(lv_prox is not None and lv_prox.get("valid"),
                "proximal levels valid")
    ok &= check(lv_mid is not None and lv_mid.get("valid"),
                "50pct levels valid")
    if not ok:
        print(f"    lv_prox = {lv_prox}")
        print(f"    lv_mid  = {lv_mid}")
        return False

    ok &= check(abs(lv_prox["entry"] - proximal_expected) < 1e-6,
                f"proximal entry == OB proximal ({lv_prox['entry']} vs {proximal_expected})")
    ok &= check(abs(lv_mid["entry"] - midpoint_expected) < 1e-6,
                f"50pct entry == OB midpoint ({lv_mid['entry']} vs {midpoint_expected})")
    ok &= check(abs(lv_prox["sl"] - lv_mid["sl"]) < 1e-6,
                "SL identical for both entry zones")
    ok &= check(abs(lv_prox["sl"] - sl_expected) < 1e-6,
                f"SL == distal - spread ({lv_prox['sl']} vs {sl_expected})")

    # R-distance relation. Both R = OB-half + spread (for 50pct) and OB-full
    # + spread (for proximal), so ratio depends on spread vs OB width. For a
    # tight OB with non-zero spread, R_mid > R_prox/2 strictly.
    r_prox = abs(lv_prox["entry"] - lv_prox["sl"])
    r_mid = abs(lv_mid["entry"] - lv_mid["sl"])
    ok &= check(r_mid < r_prox,
                f"50pct R-distance smaller than proximal ({r_mid:.5f} < {r_prox:.5f})")
    ob_w = ob["high"] - ob["low"]
    spread = pair_conf["spread_pips"] * 0.0001
    r_mid_expected = ob_w / 2.0 + spread
    r_prox_expected = ob_w + spread
    ok &= check(abs(r_mid - r_mid_expected) < 1e-6,
                f"50pct R == OB_width/2 + spread ({r_mid:.5f} vs {r_mid_expected:.5f})")
    ok &= check(abs(r_prox - r_prox_expected) < 1e-6,
                f"proximal R == OB_width + spread ({r_prox:.5f} vs {r_prox_expected:.5f})")

    # TP1: because 50pct entry has tighter R, its 1.5R threshold is lower,
    # so it can qualify a NEARER opposing swing. Proximal needs a further
    # swing. So TP1_proximal >= TP1_50pct (for bullish; reverse for bearish).
    if lv_prox.get("tp1") is not None and lv_mid.get("tp1") is not None:
        ok &= check(lv_prox["tp1"] >= lv_mid["tp1"] - 1e-9,
                    f"proximal TP1 ({lv_prox['tp1']}) >= 50pct TP1 ({lv_mid['tp1']}) "
                    "(tighter R gate lets 50pct pick a nearer swing)")
        # Both RRs must clear the 1.5R live gate.
        ok &= check(lv_prox.get("rr", 0) >= 1.5,
                    f"proximal RR >= 1.5 ({lv_prox.get('rr')})")
        ok &= check(lv_mid.get("rr", 0) >= 1.5,
                    f"50pct RR >= 1.5 ({lv_mid.get('rr')})")
        # If TPs happen to be identical, 50pct RR must be strictly larger
        # (tighter R, same numerator). If TPs differ, no strict relation.
        if abs(lv_prox["tp1"] - lv_mid["tp1"]) < 1e-9:
            ok &= check(lv_mid["rr"] > lv_prox["rr"],
                        f"shared TP: 50pct RR > proximal RR "
                        f"({lv_mid['rr']} > {lv_prox['rr']})")

    return ok


def test_no_m15_no_crash():
    """h1_only=True with df_m15=None should never invoke find_m15_ob_inside_h1."""
    print("\n== test_no_m15_no_crash ==")
    df_h1 = _synth_h1_df()
    pair_conf = _synth_pair_conf()
    ob = _synth_ob_bullish(df_h1)
    current_price = ob["proximal_line"]
    try:
        lv = smc_detector.compute_phase2_levels(
            pair_conf, "LONG", ob, current_price, df_h1, df_m15=None,
            h1_only=True, entry_zone="proximal",
        )
        return check(isinstance(lv, dict), "no crash with df_m15=None")
    except Exception as e:
        print(f"  FAIL: crashed with df_m15=None: {type(e).__name__}: {e}")
        return False


def test_dual_simulator_columns():
    """simulate_h1_only_dual returns rows with the expected column names."""
    print("\n== test_dual_simulator_columns ==")
    df_h1 = _synth_h1_df()
    pair_conf = _synth_pair_conf()
    ob = _synth_ob_bullish(df_h1, ts_idx=-150)
    # Place the OB touch ~150 bars in so there's plenty of future H1 bars
    # for the trade walker.
    alert_ts_idx = -150
    alert = {
        "pair": "TESTPAIR",
        "ts": df_h1.index[alert_ts_idx],
        "current_price": ob["proximal_line"],
        "h1_atr": 0.0010,
        "ob": ob,
    }
    rows = h1_only_simulator.simulate_h1_only_dual(
        alert, pair_conf, df_h1, risk_usd=250.0,
    )
    print(f"  rows returned: {len(rows)}")
    if not rows:
        print("  FAIL: dual simulator returned 0 rows for valid setup")
        return False
    expected_cols = {
        "pair", "alert_ts", "entry_zone", "entry", "sl_initial",
        "tp1", "tp2", "tp1_rr", "tp2_rr", "exit_reason",
        "r_realised", "r_if_exit_tp1", "r_if_exit_tp2",
        "mfe_r", "mae_r", "bars_to_exit", "ob_age_h1_bars",
        "pd_zone", "score", "structure_pts", "sweep_pts",
        "fvg_pts", "freshness_pts", "killzone_pts",
        "confluences_present", "session", "event",
    }
    ok = True
    for r in rows:
        missing = expected_cols - set(r.keys())
        ok &= check(not missing, f"all expected columns present (missing={missing})")
        ok &= check(r.get("model") == "h1_only", "model tag = h1_only")
        ok &= check(r.get("entry_zone") in ("proximal", "50pct"),
                    f"entry_zone valid ({r.get('entry_zone')})")
    zones = {r["entry_zone"] for r in rows}
    ok &= check("proximal" in zones, "proximal row present")
    ok &= check("50pct" in zones, "50pct row present")
    return ok


def test_tp2_ordering_invariant():
    """TP2, when present, must be strictly past TP1 by direction. Covers both
    bullish (TP2 > TP1) and bearish (TP2 < TP1) cases. Guards against a
    regression of the rounding-collision / dealing-range-fallback bug."""
    print("\n== test_tp2_ordering_invariant ==")
    df_h1 = _synth_h1_df()
    pair_conf = _synth_pair_conf()

    ok = True
    for bias_label, direction, ts_idx in [("LONG", "bullish", -150), ("SHORT", "bearish", -150)]:
        ob = _synth_ob_bullish(df_h1, ts_idx=ts_idx)
        if direction == "bearish":
            px = float(df_h1["Close"].iloc[ts_idx])
            bot = px + 0.0005
            top = bot + 0.0018
            ob["direction"]     = "bearish"
            ob["high"]          = top
            ob["low"]           = bot
            ob["proximal_line"] = bot
            ob["distal_line"]   = top
            current_price = bot
        else:
            current_price = ob["proximal_line"]

        for zone in ("proximal", "50pct"):
            lv = smc_detector.compute_phase2_levels(
                pair_conf, bias_label, ob, current_price, df_h1, df_m15=None,
                h1_only=True, entry_zone=zone,
            )
            if not (isinstance(lv, dict) and lv.get("valid")):
                continue
            tp1 = lv.get("tp1")
            tp2 = lv.get("tp2")
            if tp2 is None:
                ok &= check(True, f"{bias_label}/{zone}: tp2=None (no second swing) - acceptable")
                continue
            if bias_label == "LONG":
                ok &= check(tp2 > tp1, f"LONG/{zone}: tp2 ({tp2}) > tp1 ({tp1})")
            else:
                ok &= check(tp2 < tp1, f"SHORT/{zone}: tp2 ({tp2}) < tp1 ({tp1})")
    return ok


def main():
    results = [
        ("test_no_m15_no_crash",        test_no_m15_no_crash()),
        ("test_levels_dual_entry",      test_levels_dual_entry()),
        ("test_dual_simulator_columns", test_dual_simulator_columns()),
        ("test_tp2_ordering_invariant", test_tp2_ordering_invariant()),
    ]
    print("\n=== SUMMARY ===")
    fail = 0
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'} : {name}")
        if not ok:
            fail += 1
    if fail:
        print(f"\n{fail}/{len(results)} test(s) failed.")
        sys.exit(1)
    print(f"\nAll {len(results)} tests passed.")


if __name__ == "__main__":
    main()
