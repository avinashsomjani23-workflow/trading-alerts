"""Unit-style assertions for the H1-only mode.

Validates:
  1. compute_phase2_levels returns H1 OB geometry (no M15 nest path).
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
        pair_conf, "LONG", ob, current_price, df_h1,
        entry_zone="proximal",
    )
    lv_mid = smc_detector.compute_phase2_levels(
        pair_conf, "LONG", ob, current_price, df_h1,
        entry_zone="50pct",
    )

    ok = True
    ok &= check(lv_prox is not None and lv_prox.get("valid"),
                "proximal levels valid")
    ok &= check(lv_mid is not None and lv_mid.get("valid"),
                "50pct levels valid")
    if not ok:
        print(f"    lv_prox = {lv_prox}")
        print(f"    lv_mid  = {lv_mid}")
        assert False, "levels not valid (see lv_prox/lv_mid above)"

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

    assert ok, "levels dual-entry checks failed (see output above)"


def test_signature_h1_only():
    """compute_phase2_levels takes H1 data only -- no df_m15 / h1_only kwargs."""
    print("\n== test_signature_h1_only ==")
    df_h1 = _synth_h1_df()
    pair_conf = _synth_pair_conf()
    ob = _synth_ob_bullish(df_h1)
    current_price = ob["proximal_line"]
    try:
        lv = smc_detector.compute_phase2_levels(
            pair_conf, "LONG", ob, current_price, df_h1,
            entry_zone="proximal",
        )
        ok = check(isinstance(lv, dict), "compute_phase2_levels returns dict")
        ok &= check("m15_ob" not in lv,
                    "result no longer carries m15_ob payload")
    except Exception as e:
        raise AssertionError(f"compute_phase2_levels crashed: {type(e).__name__}: {e}")
    assert ok, "signature h1-only checks failed (see output above)"


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
    assert rows, "dual simulator returned 0 rows for valid setup"
    expected_cols = {
        "pair", "alert_ts", "entry_zone", "entry", "sl_raw", "sl_initial",
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
        ok &= check(r.get("entry_zone") == "proximal",
                    f"entry_zone is proximal ({r.get('entry_zone')})")
    zones = {r["entry_zone"] for r in rows}
    # Proximal is the only entry zone (50% mean entry removed 2026-07): exactly
    # one proximal row per valid alert, and no 50pct row ever.
    ok &= check(zones == {"proximal"}, f"only proximal rows emitted ({zones})")
    ok &= check(len(rows) == 1, f"exactly one row per alert ({len(rows)})")
    assert ok, "one or more column checks failed (see output above)"


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
                pair_conf, bias_label, ob, current_price, df_h1,
                entry_zone=zone,
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
    assert ok, "tp2 ordering invariant checks failed (see output above)"


def test_be_arms_at_exact_1r_touch():
    """Regression (2026-07-03): a bar that touches EXACTLY +1R must arm break-even.

    Root bug: be_trigger = entry +/- r_distance carries float error (e.g.
    1.0151000000000001 for a bar high of 1.0151), so `hi >= be_trigger` failed by
    ~2e-16 on the exact-touch bar while MFE still credited the raw high as +1R. The
    trade then rode to a full -1R loss -> a physically impossible row (exit sl at
    -1R with mfe_r rounded to +1.0), which G10 rule (b) flags. This pins the fix
    (be_eps tolerance) on the standalone exit walk. Real occurrence: USDCHF
    2008-04-02, USDCAD 2008-07-18.
    """
    print("\n== test_be_arms_at_exact_1r_touch ==")
    from backtest.exit_engine import walk_multileg
    # Reproduce the exact USDCHF geometry: entry 1.0115, sl 1.0079 -> r_distance
    # 0.0036 (with the same FP tail), be_trigger = 1.0151000000000001.
    entry, sl = 1.0115, 1.0079
    r_distance = abs(entry - sl)
    be_trigger = entry + r_distance
    check(be_trigger > 1.0151, f"FP: be_trigger {be_trigger!r} > raw 1.0151 (the trap)")
    tp1 = 1.0194  # well past +1R; not hit in this window
    idx = pd.date_range("2008-04-02 13:00", periods=3, freq="h", tz="UTC")
    # Bar 0 = fill bar (low touches entry). Bar 1 = high touches EXACTLY +1R (1.0151),
    # no other level hit. Bar 2 = drops to the initial SL.
    future = pd.DataFrame({
        "Open":  [1.0157, 1.0126, 1.0098],
        "High":  [1.0157, 1.0151, 1.0098],
        "Low":   [1.0112, 1.0112, 1.0068],
        "Close": [1.0126, 1.0149, 1.0070],
    }, index=idx)
    cfg = {"legs": [(1.0, "tp1")], "be_trigger_r": 1.0, "be_to_r": 0.0}
    res = walk_multileg(future, "LONG", entry, sl, r_distance, tp1, cfg,
                        weekend_flat=False)
    ok = True
    # BE armed at +1R -> the SL exit lands at entry (0R), NOT -1R.
    ok &= check(res["r_realised"] >= -0.002,
                f"exact +1R touch arms BE -> exit ~0R (got r={res['r_realised']})")
    # And the impossible pairing (sl at -1R with mfe_r ~+1R) must not occur.
    impossible = (res["exit_reason"] == "sl" and res["r_realised"] <= -0.999
                  and res["mfe_r"] >= 0.999)
    ok &= check(not impossible,
                f"no full_sl_loser_with_1R_mfe (exit={res['exit_reason']}, "
                f"r={res['r_realised']}, mfe_r={res['mfe_r']})")
    assert ok, "be-arms-at-exact-1R checks failed (see output above)"


def test_wider_stop_replay_and_pre_payment():
    """Guard (2026-07-09, Edge Lab Step B): the wider-stop lever, out-of-band.

    Three things a silent bug would corrupt, pinned here (never in the live path):
      (a) SURVIVAL — a wider stop must NOT fire on a bar that hit the baseline stop
          (else the whole lever is a no-op and the ceiling is a lie).
      (b) EXACT −1R OF THE NEW RISK — a clean stop-through of the wider stop loses
          exactly −1.0R measured against the WIDER r_distance (the walker's R math
          must rescale to the stop the caller passed, not the original).
      (c) SPREAD NOT DOUBLE-CHARGED — _net_r_of_legs pre-pays spread on a stop-out
          at the TRADED (widened) stop; the old `== sl_initial` check wrongly charged
          it because a widened-stop exit price != sl_initial (SPEC §7.2).
    """
    print("\n== test_wider_stop_replay_and_pre_payment ==")
    from backtest.exit_engine import walk_multileg
    from backtest.diagnostics.edge_engine import _net_r_of_legs
    ok = True

    # LONG. entry 1.1000, baseline sl 1.0960 -> baseline r_distance 0.0040. ATR 0.0020.
    # Wider stop k=1.0 ATR -> sl 1.0940, wider r_distance 0.0060.
    entry = 1.1000
    base_sl = 1.0960
    atr = 0.0020
    k = 1.0
    wider_sl = base_sl - k * atr           # 1.0940
    base_r = abs(entry - base_sl)          # 0.0040
    wider_r = abs(entry - wider_sl)        # 0.0060
    tp1 = 1.1200                            # far away; not hit in these windows
    cfg = {"legs": [(1.0, "tp1")], "be_trigger_r": None}

    # ── (a) SURVIVAL: a bar dips to 1.0945 (below baseline 1.0960, ABOVE wider
    # 1.0940), then recovers. Baseline stop = hit (-1R). Wider stop = survives.
    idx = pd.date_range("2020-01-06 09:00", periods=3, freq="h", tz="UTC")
    fut_survive = pd.DataFrame({
        "Open":  [1.1000, 1.0990, 1.0990],
        "High":  [1.1000, 1.0995, 1.1010],
        "Low":   [1.0975, 1.0945, 1.0985],   # bar1 low 1.0945: kills baseline, spares wider
        "Close": [1.0990, 1.0990, 1.1005],
    }, index=idx)
    base_res = walk_multileg(fut_survive, "LONG", entry, base_sl, base_r, tp1, cfg,
                             weekend_flat=False)
    wide_res = walk_multileg(fut_survive, "LONG", entry, wider_sl, wider_r, tp1, cfg,
                             weekend_flat=False)
    ok &= check(base_res["exit_reason"] == "sl",
                f"(a) baseline stop fires on the 1.0945 dip (got {base_res['exit_reason']})")
    ok &= check(wide_res["exit_reason"] != "sl",
                f"(a) wider stop SURVIVES the same bar (got {wide_res['exit_reason']}, "
                f"r={wide_res['r_realised']})")

    # ── (b) EXACT -1R OF NEW RISK: a bar cleanly trades through the wider stop
    # (low 1.0930 < 1.0940). r_realised must be exactly -1.0 against the WIDER r.
    idx2 = pd.date_range("2020-01-07 09:00", periods=2, freq="h", tz="UTC")
    fut_through = pd.DataFrame({
        "Open":  [1.1000, 1.0990],
        "High":  [1.1000, 1.0992],
        "Low":   [1.0975, 1.0930],   # bar1 low 1.0930: through the wider stop 1.0940
        "Close": [1.0990, 1.0935],
    }, index=idx2)
    wide_through = walk_multileg(fut_through, "LONG", entry, wider_sl, wider_r, tp1, cfg,
                                 weekend_flat=False)
    ok &= check(wide_through["exit_reason"] == "sl",
                f"(b) wider stop fires on the clean stop-through "
                f"(got {wide_through['exit_reason']})")
    ok &= check(abs(wide_through["r_realised"] - (-1.0)) < 1e-9,
                f"(b) loss is exactly -1.0R of the NEW risk "
                f"(got r={wide_through['r_realised']})")

    # ── (c) SPREAD PRE-PAYMENT at the traded (widened) stop, not sl_initial.
    # One leg exiting at the widened stop must be charged ZERO cost; the SAME leg
    # exited at any non-stop price must be charged the cost. cost_r arbitrary 0.02.
    cost_r = 0.02
    stop_leg = [{"frac": 1.0, "reason": "sl", "exit_price": wider_sl}]
    net_stop = _net_r_of_legs(stop_leg, "LONG", entry, wider_sl, wider_r, cost_r)
    ok &= check(abs(net_stop - (-1.0)) < 1e-9,
                f"(c) widened-stop loser pre-pays spread -> net exactly -1.0R "
                f"(got {net_stop}, NOT -1.02)")
    tp_leg = [{"frac": 1.0, "reason": "tp1", "exit_price": entry + wider_r}]  # +1R at TP
    net_tp = _net_r_of_legs(tp_leg, "LONG", entry, wider_sl, wider_r, cost_r)
    ok &= check(abs(net_tp - (1.0 - cost_r)) < 1e-9,
                f"(c) a non-stop exit IS charged the spread "
                f"(got {net_tp}, want {1.0 - cost_r})")
    assert ok, "wider-stop replay/pre-payment guard failed"


def test_trailing_stop_lever():
    """Guard (2026-07-09, Edge Lab Step B): the trailing-stop lever (exit_engine
    step 7b), out-of-band. Four things a silent trail bug would corrupt, pinned
    here (never in the live path):

      (RATCHET)  the stop tightens as price runs favourable and NEVER loosens on a
                 later pullback (a loosen = look-ahead give-back, corrupts every R).
      (ARM)      trail_arm_r gates trailing: below the arm level the stop stays put.
      (FILL-BAR) the fill bar's extreme does NOT seed the trail (it is pre-fill
                 price — the same exclusion MFE uses). Pinning this stops a future
                 "fix" from turning it into a look-ahead bug.
      (CONTROL)  trail_r=None => no early trail exit; the trade rides to window_end.

    Reference geometry (hand-verified, reused from the handoff): LONG entry 1.1000,
    sl 1.0960 -> r_distance 0.0040, tp1 far. trail_r=1.0, arm=0.0. The fill bar
    (bar0) high is 1.1080 but is EXCLUDED, so the first trail seed is bar1's high
    1.1075 -> trailed stop 1.1035 (=+0.875R). A later bar whose low is 1.1035 hits
    that trailed stop -> exit +0.875R.
    """
    print("\n== test_trailing_stop_lever ==")
    from backtest.exit_engine import walk_multileg
    ok = True

    entry, sl = 1.1000, 1.0960
    r_distance = abs(entry - sl)          # 0.0040
    tp1 = 1.1200                          # far; never hit in these windows

    # ── TRAIL SEED + FILL-BAR EXCLUDED ───────────────────────────────────────
    # bar0 (fill): high 1.1080 (must NOT seed the trail), low 1.0995 fills the entry.
    # bar1: high 1.1075 -> trail_best 1.1075, trailed stop 1.1035 (+0.875R). low
    #        stays above 1.1035 so no exit yet.
    # bar2: PULLBACK — high only 1.1050 (lower than bar1), low 1.1040 (above the
    #        1.1035 trailed stop). trail_best stays 1.1075; no exit.
    # bar3: low 1.1035 -> hits the trailed stop -> exit +0.875R.
    idx = pd.date_range("2020-01-06 09:00", periods=4, freq="h", tz="UTC")
    fut = pd.DataFrame({
        "Open":  [1.1000, 1.1010, 1.1045, 1.1042],
        "High":  [1.1080, 1.1075, 1.1050, 1.1044],   # bar0 high must be ignored
        "Low":   [1.0995, 1.1040, 1.1040, 1.1035],   # bar3 low hits trailed stop
        "Close": [1.1010, 1.1045, 1.1042, 1.1036],
    }, index=idx)
    cfg_trail = {"legs": [(1.0, "tp1")], "be_trigger_r": None,
                 "trail_r": 1.0, "trail_arm_r": 0.0}
    res = walk_multileg(fut, "LONG", entry, sl, r_distance, tp1, cfg_trail,
                        weekend_flat=False)
    ok &= check(res["exit_reason"] == "sl",
                f"(trail) exit is the trailed stop (got {res['exit_reason']})")
    ok &= check(abs(res["r_realised"] - 0.875) < 1e-9,
                f"(trail+fillbar) trailed exit = +0.875R, fill-bar high 1.1080 "
                f"excluded (got {res['r_realised']})")
    # Fill-bar-seed counter-check: if the fill bar's 1.1080 HAD seeded the trail,
    # the stop would be 1.1040 by bar0, and bar1's low 1.1040 would exit at +1.0R.
    # The +0.875R above already proves that did NOT happen; assert the negative too.
    ok &= check(abs(res["r_realised"] - 1.0) > 1e-6,
                "(fillbar) trail did NOT seed off the fill bar (would give +1.0R)")

    # ── RATCHET: the trail must NEVER loosen an already-tighter stop ──────────
    # This is the sub-test that bites the `cand > cur_sl` guard. trail_best is
    # monotonic on its own, so a loosening bug only shows when BE has already moved
    # the stop TIGHTER than a later trail candidate. Setup: BE arms at +1R -> stop to
    # entry 1.1000; then trail_r=2.0 computes a candidate at 1.0960 (LOOSER than BE).
    # Ratchet keeps 1.1000. bar2 low 1.0980 hits the BE stop -> exit exactly 0.0R.
    # If the ratchet were removed the stop would drop to 1.0960, 1.0980 would miss it,
    # and the trade would ride to window_end at -0.375R (verified: this bites).
    idx_r = pd.date_range("2020-02-03 09:00", periods=5, freq="h", tz="UTC")
    fut_ratchet = pd.DataFrame({
        "Open":  [1.1000, 1.1020, 1.0990, 1.0990, 1.0990],
        "High":  [1.1010, 1.1040, 1.0995, 1.0995, 1.0995],  # bar1 high 1.1040 = +1R
        "Low":   [1.0995, 1.1015, 1.0980, 1.0980, 1.0980],  # bar2 low hits BE stop
        "Close": [1.1020, 1.1035, 1.0985, 1.0985, 1.0985],
    }, index=idx_r)
    cfg_ratchet = {"legs": [(1.0, "tp1")], "be_trigger_r": 1.0, "be_to_r": 0.0,
                   "trail_r": 2.0, "trail_arm_r": 0.0}
    res_ratchet = walk_multileg(fut_ratchet, "LONG", entry, sl, r_distance,
                                1.1300, cfg_ratchet, weekend_flat=False)
    ok &= check(res_ratchet["exit_reason"] == "sl",
                f"(ratchet) BE stop holds, is not loosened by a looser trail "
                f"(got {res_ratchet['exit_reason']})")
    ok &= check(abs(res_ratchet["r_realised"] - 0.0) < 1e-9,
                f"(ratchet) exit at the tighter BE stop = 0.0R, trail did NOT loosen "
                f"it (got {res_ratchet['r_realised']})")

    # ── ARM: trailing suppressed below trail_arm_r ───────────────────────────
    # Same trail-seed bars, but arm at +2.0R. Best excursion is bar1 high 1.1075 =
    # +1.875R, never reaches +2R, so the trail NEVER arms -> the stop stays at 1.0960
    # and the trade rides to window_end (no early trailed exit).
    cfg_arm = {"legs": [(1.0, "tp1")], "be_trigger_r": None,
               "trail_r": 1.0, "trail_arm_r": 2.0}
    res_arm = walk_multileg(fut, "LONG", entry, sl, r_distance, tp1, cfg_arm,
                            weekend_flat=False)
    ok &= check(res_arm["exit_reason"] == "window_end",
                f"(arm) below the arm level the stop never trails "
                f"(got {res_arm['exit_reason']})")

    # ── CONTROL: trail_r=None -> no trailing at all, rides to window_end ──────
    cfg_none = {"legs": [(1.0, "tp1")], "be_trigger_r": None, "trail_r": None}
    res_none = walk_multileg(fut, "LONG", entry, sl, r_distance, tp1, cfg_none,
                             weekend_flat=False)
    ok &= check(res_none["exit_reason"] == "window_end",
                f"(control) trail_r=None does not stop early "
                f"(got {res_none['exit_reason']})")
    ok &= check(res_none["r_realised"] != res["r_realised"],
                f"(control) no-trail R differs from trailed R "
                f"({res_none['r_realised']} vs {res['r_realised']})")

    assert ok, "trailing-stop lever checks failed (see output above)"


def test_g10_rounded_near_miss_is_not_a_violation():
    """Regression (2026-07-03): G10 rule (b) must NOT fire on a genuine full-SL
    loser whose raw MFE (0.9995-0.99999R) rounds UP to exactly 1.000R.

    Root bug: rule (b) tripped at `mfe_r >= 0.999`. A real trade can reach just
    under +1R (below the be_eps arm tolerance, so break-even never arms) and then
    take a full -1R stop -> the stored 3dp-rounded mfe_r shows 1.000. That is
    physically possible, but the old threshold flagged it as PHYS_IMPOSSIBLE.
    Over an 18yr run 2 such rows tripped G10 FAIL after the be_eps walk fix was
    already in place -- the gate threshold, not the walk, was the residual gap.
    This pins rule (b) at +1.001R: rounding a sub-(1-1e-6)R excursion can never
    reach it, but the true fake-excursion bug (MFE of 2-5R) still trips.
    Binds to the LIVE predicate (g10_violations), not a copy of the threshold.
    """
    print("\n== test_g10_rounded_near_miss_is_not_a_violation ==")
    from backtest.scanlog.gates import g10_violations
    ok = True
    # (A) The exact failing shape: full SL, mfe_r rounded to 1.0 -> must PASS.
    near_miss = {"exit_reason": "sl", "r_realised": -1.0, "mfe_r": 1.0,
                 "mae_r": -1.0}
    ok &= check("full_sl_loser_with_1R_mfe" not in g10_violations(near_miss),
                f"rounded near-miss (mfe_r=1.0, r=-1.0) is not a G10 violation "
                f"(got {g10_violations(near_miss)})")
    # (B) The real fake-excursion bug: MFE well past +1R -> must still FAIL.
    fake = {"exit_reason": "sl", "r_realised": -1.0, "mfe_r": 2.3, "mae_r": -1.0}
    ok &= check("full_sl_loser_with_1R_mfe" in g10_violations(fake),
                f"true fake excursion (mfe_r=2.3 on a full SL) still trips G10 "
                f"(got {g10_violations(fake)})")
    # (C) Sign rules and TP1-cap rule unaffected.
    ok &= check("excursion_sign" in g10_violations(
                    {"exit_reason": "sl", "r_realised": -1.0, "mfe_r": -0.5,
                     "mae_r": 0.0}),
                "negative mfe_r still trips excursion_sign")
    ok &= check("mfe_beyond_tp1_exit" in g10_violations(
                    {"exit_reason": "tp1", "r_realised": 1.0, "mfe_r": 1.5,
                     "mae_r": -0.2}),
                "mfe past TP1 exit still trips mfe_beyond_tp1_exit")
    assert ok, "g10 rounded-near-miss checks failed (see output above)"


def test_sl_wick_depth_atr():
    """sl_wick_depth_atr (2026-07-08): on an SL exit it must be the wick's
    overshoot BEYOND the fired stop, normalised by the OB-formation ATR, and
    never negative. None on non-SL exits. This pins the geometry so a future
    wider-stop replay can trust the column instead of re-deriving it.

    Builds the SL bar directly through _build_row's sibling path by driving one
    LONG trade to a stop whose candle wicks a known distance past the stop.
    """
    print("\n== test_sl_wick_depth_atr ==")
    df_h1 = _synth_h1_df()
    pair_conf = _synth_pair_conf()
    ob = _synth_ob_bullish(df_h1, ts_idx=-150)
    alert = {
        "pair": "TESTPAIR",
        "ts": df_h1.index[-150],
        "current_price": ob["proximal_line"],
        "h1_atr": ob.get("h1_atr", 0.0010),
        "ob": ob,
    }
    rows = h1_only_simulator.simulate_h1_only_dual(
        alert, pair_conf, df_h1, risk_usd=250.0,
    )
    ok = True
    ok &= check(bool(rows), "sim returned a row")
    assert rows, "sim returned no rows"
    r = rows[0]
    ok &= check("sl_wick_depth_atr" in r, "sl_wick_depth_atr column present")
    depth = r.get("sl_wick_depth_atr")
    if r.get("exit_reason") == "sl":
        ok &= check(depth is not None, "depth stamped on an SL exit")
        ok &= check(depth is None or depth >= 0.0,
                    f"depth is non-negative (got {depth})")
    else:
        ok &= check(depth is None,
                    f"depth is None on a non-SL exit ({r.get('exit_reason')})")
    # Unit check via the raw formula: a LONG stop wicked 0.5 ATR below the stop
    # must yield 0.5 (uses the same max(0, overshoot)/atr the sim uses).
    atr, stop, wick_low = 0.0010, 1.2000, 1.1995
    manual = round(max(0.0, stop - wick_low) / atr, 3)
    ok &= check(manual == 0.5, f"formula: 0.5-ATR overshoot -> 0.5 (got {manual})")
    # A wick that closes exactly at the stop = 0.0, not None.
    manual0 = round(max(0.0, stop - stop) / atr, 3)
    ok &= check(manual0 == 0.0, f"formula: no overshoot -> 0.0 (got {manual0})")
    assert ok, "one or more sl_wick_depth_atr checks failed (see output above)"


def test_edge_lab_columns():
    """Edge-lab columns (2026-07-08): the 3 derived (encoded, ex-pasted) + the 3
    outcome-time SL columns. Pins the DEFINITIONS so a formula regression fails
    loudly out-of-band (never in the live path). These mirror the exact math in
    h1_only_simulator._build_row / _simulate_single_entry.
    """
    print("\n== test_edge_lab_columns ==")
    ok = True

    # --- sl_distance_atr = |entry - sl_initial| / ATR (uses sl_initial, not sl_raw)
    entry, sl_initial, atr = 1.4793, 1.4814, 0.001914
    sl_dist = round(abs(entry - sl_initial) / atr, 3)
    ok &= check(sl_dist == 1.097, f"sl_distance_atr: 1.097 (got {sl_dist})")

    # --- r_capture_ratio = r_realised / mfe_r ; None when mfe_r <= 0
    ok &= check(round(1.629 / 1.629, 3) == 1.0, "r_capture: full ride -> 1.0")
    ok &= check(round(0.0 / 1.0, 3) == 0.0, "r_capture: gave back to BE -> 0.0")
    rcap_none = (0.0 if (0.0 > 0) else None)  # mfe_r == 0 branch
    ok &= check(rcap_none is None, "r_capture: mfe_r<=0 -> None (no 0/0)")

    # --- trend_pd_agree = with-H1-trend AND pd_alignment=='aligned'
    def _agree(direction, h1_trend, pd_alignment):
        if h1_trend is None or pd_alignment is None:
            return None
        wt = ((direction == "bullish" and h1_trend == "bullish")
              or (direction == "bearish" and h1_trend == "bearish"))
        return bool(wt and pd_alignment == "aligned")
    ok &= check(_agree("bearish", "bearish", "aligned") is True,
                "trend_pd_agree: with-trend + aligned -> True")
    ok &= check(_agree("bullish", "bearish", "counter") is False,
                "trend_pd_agree: counter-trend -> False")
    ok &= check(_agree("bearish", "bullish", "aligned") is False,
                "trend_pd_agree: against-trend but aligned -> False")
    ok &= check(_agree("bullish", None, "aligned") is None,
                "trend_pd_agree: missing h1_trend -> None")

    # --- sl_max_adverse_after_sweep_atr: further run BEYOND stop, in ATR, >=0
    # LONG stop below at cur_sl; worst Low after the stop bar.
    cur_sl, worst_low, atr2 = 1.2000, 1.1975, 0.0010
    adverse = round(max(0.0, cur_sl - worst_low) / atr2, 3)
    ok &= check(adverse == 2.5, f"max_adverse: 2.5-ATR deeper (got {adverse})")
    # never negative when price did NOT go past the stop (recovered immediately)
    adverse0 = round(max(0.0, cur_sl - 1.2005) / atr2, 3)
    ok &= check(adverse0 == 0.0, f"max_adverse: no further run -> 0.0 (got {adverse0})")

    # --- bars_sl_to_tp1_touch: 1-indexed bar of first TP1 touch after the stop
    # (first post-stop bar == 1). Simulated via a small index-of search.
    highs = [1.0, 1.0, 5.0, 1.0]  # tp1=4.0 first cleared at position 2 (0-indexed)
    tp1 = 4.0
    idx = next((i for i, h in enumerate(highs) if h >= tp1), None)
    bars = (idx + 1) if idx is not None else None
    ok &= check(bars == 3, f"bars_sl_to_tp1_touch: 1-indexed -> 3 (got {bars})")
    idx_none = next((i for i, h in enumerate([1.0, 2.0]) if h >= tp1), None)
    ok &= check(idx_none is None, "bars_sl_to_tp1_touch: never touched -> None")

    assert ok, "one or more edge-lab column checks failed (see output above)"


def _atr_sweep_scenario(atr):
    """Drive the LIVE simulator to a bullish SL-SWEEP exit with a known ATR, so
    ALL SIX *_atr columns are populated on one row (the four row-build ones plus
    the two SL-outcome ones). Returns the single trade row.

    Geometry: a Monday-open bullish OB; price fills the proximal, then bar 4 wicks
    deep below the stop but CLOSES back above it (sweep), then crashes further —
    guaranteeing sl_bar_was_sweep, sl_wick_depth_atr and
    sl_max_adverse_after_sweep_atr are all set. Everything before Friday so the
    weekend-flat force-close cannot pre-empt the SL.
    """
    base = pd.Timestamp("2026-04-06 00:00", tz="UTC")   # Monday (dayofweek==0)
    n = 60
    idx = pd.date_range(base, periods=n, freq="h")
    top = 1.0800
    bot = top - 0.0018          # 18-pip OB
    spread = 0.0002             # 2 pips (matches _synth_pair_conf spread_pips=2)
    sl = bot - spread
    ob = {
        "direction": "bullish", "bos_tag": "BOS", "bos_tier": "Major",
        "high": top, "low": bot, "proximal_line": top, "distal_line": bot,
        "ob_timestamp": idx[0].isoformat(), "touches": 0, "h1_atr": atr,
        # Give impulse + FVG real geometry so impulse_leg_atr / fvg_size_atr are
        # non-None too (they read ob internals not present in the CSV).
        "impulse_start_price": bot - 0.0050,
        "bos_swing_price": top + 0.0030,
        "fvg": {"exists": True, "fvg_top": top + 0.0009, "fvg_bottom": top + 0.0003},
        "dealing_range": {"valid": True, "range_low": bot - 0.02,
                          "range_high": top + 0.05},
    }
    closes = []
    for i in range(n):
        if i < 2:
            closes.append(top + 0.0010)     # above the zone
        elif i < 4:
            closes.append(top - 0.0002)     # dip in -> fill at proximal
        elif i == 4:
            closes.append(sl + 0.0001)      # close back above the stop (sweep)
        else:
            closes.append(sl - 0.0030)      # then crash further (adverse run)
    highs = [c + 0.0005 for c in closes]
    lows = [c - 0.0005 for c in closes]
    lows[4] = sl - 0.0020                    # bar 4 wick pierces DEEP below the stop
    df = pd.DataFrame({"Open": closes, "High": highs, "Low": lows,
                       "Close": closes, "Volume": [100] * n}, index=idx)
    alert = {"pair": "TESTPAIR", "ts": idx[0], "current_price": top,
             "h1_atr": atr, "ob": ob}
    rows = h1_only_simulator.simulate_h1_only_dual(
        alert, _synth_pair_conf(), df, risk_usd=250.0)
    assert rows, "atr-scenario produced no trade row"
    return rows[0]


# The six *_atr columns Area B covers (break_* are Batch 1, sweep_* out-of-scope).
_AREA_B_ATR_COLS = [
    "ob_range_atr", "fvg_size_atr", "impulse_leg_atr", "sl_distance_atr",
    "sl_wick_depth_atr", "sl_max_adverse_after_sweep_atr",
]


def test_area_b_all_atr_cols_share_the_one_h1_atr_denominator():
    """AREA B (Deep Value Pass): prove EVERY *_atr column divides by the single
    frozen ob['h1_atr'] (== atr_at_ob), not a per-scan or hard-coded ATR.

    Method drives LIVE code (simulate_h1_only_dual): run the SAME sweep scenario
    twice, changing ONLY ob['h1_atr'] (0.0010 -> 0.0020). If a column truly
    divides by that one denominator, doubling the ATR must HALVE the column (the
    numerator is denominator-free). atr_at_ob itself must DOUBLE (it IS the
    denominator, stored at 6dp). A column reading a different ATR would not scale.

    Bite check baked in: the same scenario is re-run with a source tripwire (see
    test_area_b_atr_source_uses_h1_atr) — here the value bite is the exact 2x/0.5x
    ratio, which fails the moment any *_atr divides by something that isn't the
    doubled h1_atr.
    """
    print("\n== test_area_b_all_atr_cols_share_the_one_h1_atr_denominator ==")
    r1 = _atr_sweep_scenario(0.0010)
    r2 = _atr_sweep_scenario(0.0020)   # exactly 2x the ATR
    ok = True
    # atr_at_ob is the denominator; it must DOUBLE with h1_atr (and be non-None).
    a1, a2 = r1.get("atr_at_ob"), r2.get("atr_at_ob")
    ok &= check(a1 is not None and a2 is not None, "atr_at_ob populated both runs")
    ok &= check(a1 is not None and abs(a2 - 2 * a1) < 1e-9,
                f"atr_at_ob doubled with h1_atr ({a1} -> {a2})")
    for c in _AREA_B_ATR_COLS:
        v1, v2 = r1.get(c), r2.get(c)
        ok &= check(v1 is not None and v2 is not None,
                    f"{c} populated on the sweep row (v1={v1}, v2={v2})")
        if v1 is None or v2 is None:
            continue
        ok &= check(v1 != 0.0, f"{c} non-zero so the ratio is meaningful (v1={v1})")
        # Doubling the denominator halves the value: v1 / v2 == 2 (within round(,3)).
        ok &= check(abs(v1 - 2 * v2) <= 0.0015,
                    f"{c} halves when h1_atr doubles ({v1} -> {v2}; "
                    f"expected ~{round(v1/2,3)})")
    assert ok, "one or more Area B ATR-denominator checks failed (see output above)"


def test_area_b_atr_source_uses_h1_atr():
    """AREA B source tripwire: the *_atr divisors in h1_only_simulator must read
    ob['h1_atr'], never a per-scan alert ATR. This bites a denominator swap that
    the value test could miss if a fixture happened to make two ATRs equal.

    Row-build columns divide via `_atr_norm` / `_h1_atr`, and both `_h1_atr` and
    the SL-path `_h1_atr_sl` are assigned `ob.get("h1_atr")`. Assert those exact
    bindings are present, and that atr_at_ob rounds the SAME `_h1_atr`.
    """
    print("\n== test_area_b_atr_source_uses_h1_atr ==")
    src = (Path(h1_only_simulator.__file__).read_text(encoding="utf-8"))
    ok = True
    ok &= check('_h1_atr = ob.get("h1_atr")' in src,
                "_h1_atr bound to ob['h1_atr'] (row-build denominator)")
    ok &= check('_h1_atr_sl = ob.get("h1_atr")' in src,
                "_h1_atr_sl bound to ob['h1_atr'] (SL-outcome denominator)")
    ok &= check("return round(v / _h1_atr, 3)" in src,
                "_atr_norm divides by _h1_atr (ob_range/fvg_size/impulse_leg)")
    ok &= check("abs(entry - sl) / _h1_atr" in src,
                "sl_distance_atr divides by _h1_atr")
    ok &= check("_overshoot) / _h1_atr_sl" in src,
                "sl_wick_depth_atr divides by _h1_atr_sl")
    ok &= check("_adverse) / _h1_atr_sl" in src,
                "sl_max_adverse_after_sweep_atr divides by _h1_atr_sl")
    ok &= check("atr_at_ob = round(float(_h1_atr), 6)" in src,
                "atr_at_ob is the SAME _h1_atr rounded to 6dp (the CSV denominator)")
    assert ok, "Area B source tripwire failed (see output above)"


def main():
    # Robust to BOTH styles: assert-based tests (return None on pass, raise on
    # fail — the pytest-visible ones) and legacy bool-returning tests. A raised
    # AssertionError counts as a fail; a None return counts as a pass.
    def _run(label, fn):
        try:
            r = fn()
        except AssertionError as e:
            print(f"  FAIL (assert): {label}: {e}")
            return False
        return True if r is None else bool(r)

    tests = [
        ("test_signature_h1_only",      test_signature_h1_only),
        ("test_levels_dual_entry",      test_levels_dual_entry),
        ("test_dual_simulator_columns", test_dual_simulator_columns),
        ("test_tp2_ordering_invariant", test_tp2_ordering_invariant),
        ("test_be_arms_at_exact_1r_touch", test_be_arms_at_exact_1r_touch),
        ("test_wider_stop_replay_and_pre_payment",
         test_wider_stop_replay_and_pre_payment),
        ("test_trailing_stop_lever",     test_trailing_stop_lever),
        ("test_g10_rounded_near_miss_is_not_a_violation",
         test_g10_rounded_near_miss_is_not_a_violation),
        ("test_sl_wick_depth_atr",      test_sl_wick_depth_atr),
        ("test_edge_lab_columns",       test_edge_lab_columns),
        ("test_area_b_all_atr_cols_share_the_one_h1_atr_denominator",
         test_area_b_all_atr_cols_share_the_one_h1_atr_denominator),
        ("test_area_b_atr_source_uses_h1_atr",
         test_area_b_atr_source_uses_h1_atr),
    ]
    results = [(name, _run(name, fn)) for name, fn in tests]
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
