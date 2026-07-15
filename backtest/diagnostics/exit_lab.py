"""
EXIT LAB — compare exit recipes on a COMPLETED backtest (post-processing step).

Model (what the user wants): run the backtest first; it writes trades on the frozen
MT5 cache. THEN run exit-lab — it reads those trades + the SAME frozen cache, replays
each exit recipe over each trade's post-fill bars, and writes an Excel workbook.

Why reading-after is faithful now: the cache is committed MT5 data that never changes,
so the bars a trade was born from are the exact bars exit-lab re-reads. (This failed
before only because the old trades were yfinance while the cache had moved to MT5.)
The SELF-CHECK proves it: the baseline recipe must reproduce each trade's r_realised.

Recipes (the locked set): baseline (live: liquidity TP + BE@1R), BE-sweep, fixed-TP
{0.5/1/1.5/2}, partial 50%@1R + runner. One exit implementation: exit_engine.walk_multileg.

Run:
  # 1) run the backtest (writes backtest/results/h1only_<start>_<end>/trades.csv)
  python backtest/run_backtest.py --start 2024-07-01 --end 2025-06-30 --pairs EURUSD,NZDUSD,USDJPY,USDCHF,GOLD
  # 2) then exit-lab on it
  python -m backtest.diagnostics.exit_lab --run-dir backtest/results/h1only_20240701_20250630
  #    (or: --start 2024-07-01 --end 2025-06-30  -> derives the run dir)

Outputs (next to trades.csv): exit_lab.xlsx + exit_lab_recipes.csv + exit_lab_trades.csv.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest import data_loader, insights
from backtest.exit_engine import walk_multileg
from backtest import h1_only_simulator as sim
import smc_detector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "backtest", "results")

# ── The locked experiment set (HANDOFF / RECOMMENDATIONS) ───────────────────
# target spec: float = R-multiple, "tp1" = liquidity TP. be_trigger_r None = no BE.
CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline_liqTP_be1.0":  {"legs": [(1.0, "tp1")], "be_trigger_r": 1.0, "be_to_r": 0.0},
    "B_be0.5":               {"legs": [(1.0, "tp1")], "be_trigger_r": 0.5, "be_to_r": 0.0},
    "B_be0.7":               {"legs": [(1.0, "tp1")], "be_trigger_r": 0.7, "be_to_r": 0.0},
    "C_fullTP_0.5R":         {"legs": [(1.0, 0.5)], "be_trigger_r": None},
    "C_fullTP_1.0R":         {"legs": [(1.0, 1.0)], "be_trigger_r": None},
    "C_fullTP_1.5R":         {"legs": [(1.0, 1.5)], "be_trigger_r": None},
    "C_fullTP_2.0R":         {"legs": [(1.0, 2.0)], "be_trigger_r": None},
    "D_partial50_1R_runLiq": {"legs": [(0.5, 1.0), (0.5, "tp1")], "be_trigger_r": 1.0, "be_to_r": 0.0},
    # ATR-anchored mechanical exit (the KVignesh MT5-bot exit, for a head-to-head
    # vs our structural stop). SL = atr_sl_mult x ATR, TP = atr_tp_mult x ATR, both
    # measured from the fill price — NOT the structural r_distance. `atr_sl_mult`
    # present => _replay REBUILDS sl / r_distance / target from ATR-14 computed on
    # the H1 bars STRICTLY BEFORE the fill candle (causal — the fill bar is still
    # forming when a live order fills, so its range is unknown; and NOT atr_at_ob,
    # which is frozen at OB formation and is stale when the OB sat unfilled for many
    # bars). Single target, no BE, no trail — his exact mechanical exit. Trades with
    # <15 prior bars are skipped for this recipe (real ATR unavailable; never faked).
    "E_atr_sl1.5_tp2.5":     {"legs": [(1.0, "atr_tp")], "be_trigger_r": None,
                              "atr_sl_mult": 1.5, "atr_tp_mult": 2.5, "atr_period": 14},
}


def _quarter(ts) -> str:
    ts = pd.to_datetime(ts, utc=True)
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


def _atr_at_fill(pb: pd.DataFrame, fill_ts, period: int = 14):
    """ATR(period) as it would be known the instant a live order fills.

    CAUSAL — no look-ahead. A live ATR stop is set the moment the limit order
    fills; the fill bar is still FORMING at that instant, so only bars STRICTLY
    BEFORE the fill are closed and known. We therefore compute ATR on
    `pb[pb.index < fill_ts]` (< not <=). Including the fill bar would leak that
    bar's own range into the stop — a one-bar look-ahead the trader flagged.

    Uses the engine's OWN ATR (smc_detector.compute_atr = simple mean of the last
    `period` true ranges) — NOT a bespoke Wilder ATR — so this recipe's volatility
    matches `atr_at_ob` and every other *_atr in the system (one ATR definition).

    This is deliberately NOT `atr_at_ob` (frozen at OB formation): an OB can sit
    unfilled for many bars, so its formation ATR is stale by fill time. We want the
    volatility at the fill, which is what a live ATR stop would see.

    Returns None when <period+1 prior bars exist (compute_atr returns None) — the
    ATR is undefined and is NEVER fabricated.
    """
    prior = pb.loc[pb.index < fill_ts]
    atr = smc_detector.compute_atr(prior, period=period)
    return atr if (atr and atr > 0) else None


def _symbol_map() -> Dict[str, str]:
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    return {p["name"]: p["symbol"] for p in cfg["pairs"]}


def _load_bars(symbols: Dict[str, str], pairs_needed,
               start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    """Load each needed pair's H1 bars from the frozen MT5 cache, once.

    The window is derived from the trades' own fill-time range (not hardcoded) so
    a multi-year / 18-year run reconstructs every trade — never silently drops the
    pre-window ones. `end` is padded by the post-fill hold so the last trade's full
    exit window is present.
    """
    out = {}
    for pair in pairs_needed:
        sym = symbols.get(pair)
        if not sym:
            continue
        df = data_loader.load_bars(sym, "1h", start, end)
        if df is not None and not df.empty:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            out[pair] = df
    return out


def _replay(trades: pd.DataFrame, bars: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (trade, config). Reconstructs post-fill bars from the cache."""
    max_hold = sim.MAX_HOLD_H1_BARS
    wk_flat = sim.WEEKEND_FLAT
    wk_hour = sim.WEEKEND_FLAT_HOUR_UTC
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for _, t in trades.iterrows():
        pair = t["pair"]
        pb = bars.get(pair)
        if pb is None:
            skipped += 1
            continue
        fill_ts = pd.to_datetime(t["fill_ts"], utc=True)
        future = pb.loc[pb.index >= fill_ts]
        if future.empty:
            skipped += 1
            continue
        future = future.iloc[: max_hold + 2]
        bias = t["bias"] if t.get("bias") in ("LONG", "SHORT") else (
            "LONG" if t["direction"] == "bullish" else "SHORT")
        entry = float(t["entry"]); sl = float(t["sl_initial"]); tp1 = float(t["tp1"])
        r_distance = abs(entry - sl)
        if r_distance <= 0:
            skipped += 1
            continue
        base = {
            "pair": pair, "quarter": _quarter(t["alert_ts"]),
            "alert_ts": t["alert_ts"], "fill_ts": t["fill_ts"],
            "direction": t.get("direction"), "entry": entry, "sl_initial": sl, "tp1": tp1,
            "committed_r": float(t["r_realised"]),
            "committed_mfe_r": float(t.get("mfe_r", np.nan)),
            "committed_mae_r": float(t.get("mae_r", np.nan)),
            "break_close_atr": t.get("break_close_atr"),
            "break_body_atr": t.get("break_body_atr"),
            "pd_zone": t.get("pd_zone"), "event": t.get("event"),
        }
        # ATR at the fill candle (fresh, period-14) — only computed if an ATR
        # recipe needs it. None when <15 pre-fill bars exist (ATR undefined).
        _atr_fill_cache: Dict[int, Optional[float]] = {}
        for name, cfg in CONFIGS.items():
            r_sl, r_rd, r_tp1, r_legs = sl, r_distance, tp1, None
            atr_mult = cfg.get("atr_sl_mult")
            if atr_mult is not None:
                period = int(cfg.get("atr_period", 14))
                if period not in _atr_fill_cache:
                    _atr_fill_cache[period] = _atr_at_fill(pb, fill_ts, period)
                atr_fill = _atr_fill_cache[period]
                if not atr_fill:
                    # No real ATR at fill -> this recipe cannot run for this trade.
                    rows.append({**base, "config": name, "r": np.nan,
                                 "recipe_exit_reason": "no_atr",
                                 "recipe_mfe_r": np.nan, "recipe_mae_r": np.nan})
                    continue
                # Rebuild SL / risk / target from the FILL-bar ATR (not r_distance).
                r_rd = float(atr_mult) * atr_fill
                r_sl = (entry - r_rd) if bias == "LONG" else (entry + r_rd)
                # Resolve each leg's "atr_tp" spec to an R-multiple of the ATR risk:
                #   TP at atr_tp_mult x ATR  ==  (atr_tp_mult / atr_sl_mult) R.
                tp_R = float(cfg["atr_tp_mult"]) / float(atr_mult)
                r_legs = [(frac, tp_R if spec == "atr_tp" else spec)
                          for frac, spec in cfg["legs"]]
            run_cfg = dict(cfg)
            if r_legs is not None:
                run_cfg["legs"] = r_legs
            res = walk_multileg(future, bias, entry, r_sl, r_rd, r_tp1, run_cfg,
                                weekend_flat=wk_flat, weekend_hour_utc=wk_hour,
                                max_hold=max_hold)
            rows.append({**base, "config": name, "r": res["r_realised"],
                         "recipe_exit_reason": res["exit_reason"],
                         "recipe_mfe_r": res["mfe_r"], "recipe_mae_r": res["mae_r"]})
    if skipped:
        print(f"  [note] {skipped} trades skipped (no bars / bad slice)")
    return pd.DataFrame(rows)


def _recipe_summary(rep: pd.DataFrame, r_col: str = "r") -> pd.DataFrame:
    out = []
    for name in CONFIGS:
        sub = rep[rep["config"] == name]
        # ATR recipes emit NaN r for trades with no fill-bar ATR (<15 pre-fill
        # bars). Drop them so totR/expR/CI/quarters are computed on real exits only.
        sub = sub[sub[r_col].notna()]
        if sub.empty:
            continue
        vals = sub[r_col].tolist()
        wins = sub[sub[r_col] > 0][r_col]
        losses = sub[sub[r_col] < 0][r_col]
        n_res = len(wins) + len(losses)
        avg_win = round(float(wins.mean()), 3) if len(wins) else 0.0
        avg_loss = round(float(losses.mean()), 3) if len(losses) else 0.0
        be_wr = round(1.0 / (1.0 + avg_win) * 100, 1) if avg_win > 0 else None
        lo, hi = insights.bootstrap_ci(vals)
        pq = sub.groupby("quarter")[r_col].mean().round(3).to_dict()
        pos_q = sum(1 for v in pq.values() if v > 0)
        out.append({
            "config": name,
            "N": len(vals),
            "totR": round(float(np.sum(vals)), 2),
            "expR": round(float(np.mean(vals)), 4),
            "WR_pct": round(100 * len(wins) / n_res, 1) if n_res else None,
            "avg_win_R": avg_win,
            "avg_loss_R": avg_loss,
            "breakeven_WR_pct": be_wr,
            "ci_lo": lo, "ci_hi": hi,
            "ci_excludes_0": (lo is not None and (lo > 0 or hi < 0)),
            "pos_quarters": f"{pos_q}/{len(pq)}",
            "max_dd_R": insights.max_drawdown_r(vals),
            "sharpe": insights.sharpe(vals),
            "per_quarter": " ".join(f"{q}:{v:+.2f}" for q, v in sorted(pq.items())),
        })
    return pd.DataFrame(out)


def _per_pair(rep: pd.DataFrame) -> pd.DataFrame:
    piv = rep.pivot_table(index="config", columns="pair", values="r",
                          aggfunc="mean").round(3)
    piv = piv.reindex([c for c in CONFIGS if c in piv.index])
    return piv.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="path to a completed backtest results dir (with trades.csv)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        if not (args.start and args.end):
            ap.error("give --run-dir, or --start and --end")
        rid = f"h1only_{args.start.replace('-','')}_{args.end.replace('-','')}"
        run_dir = os.path.join(RESULTS, rid)
    trades_p = os.path.join(run_dir, "trades.csv")
    if not os.path.exists(trades_p):
        ap.error(f"no trades.csv at {trades_p} — run the backtest first")

    print(f"Reading completed backtest: {trades_p}")
    trades = pd.read_csv(trades_p)
    trades = trades[trades["exit_reason"] != "never_filled"]
    trades = trades[trades["fill_ts"].notna()]
    if "entry_zone" in trades.columns:
        trades = trades[trades["entry_zone"] == "proximal"]  # the live model
    trades = trades.reset_index(drop=True)
    print(f"  {len(trades)} filled proximal trades")

    # Bar window = the trades' own fill-time span (+ pads), so any run length —
    # including the full 18-year history — reconstructs every trade.
    fts = pd.to_datetime(trades["fill_ts"], utc=True)
    bar_start = (fts.min() - pd.Timedelta(days=5)).to_pydatetime()
    bar_end = (fts.max() + pd.Timedelta(days=sim.MAX_HOLD_H1_BARS // 24 + 5)).to_pydatetime()
    symbols = _symbol_map()
    bars = _load_bars(symbols, sorted(trades["pair"].unique()), bar_start, bar_end)
    print(f"  bars (MT5 cache) {bar_start.date()}..{bar_end.date()}: {', '.join(bars.keys())}")

    rep = _replay(trades, bars)
    if rep.empty:
        print("  no replayable trades — abort")
        return

    # ── SELF-CHECK: baseline recipe must reproduce committed r_realised ──────
    base = rep[rep["config"] == "baseline_liqTP_be1.0"]
    c_exp = round(base["committed_r"].mean(), 4)
    r_exp = round(base["r"].mean(), 4)
    diff = round(abs(c_exp - r_exp), 4)
    ok = diff <= 0.01
    print("\n---- SELF-CHECK (baseline recipe vs committed r_realised) ----")
    print(f"  committed expR={c_exp}  replay expR={r_exp}  |diff|={diff}  "
          f"{'PASS' if ok else '*** FAIL — numbers NOT trustworthy ***'}")

    recipes = _recipe_summary(rep)
    per_pair = _per_pair(rep)

    print("\n================  EXIT RECIPES  ================")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(recipes.to_string(index=False))

    # ── Write outputs next to trades.csv ────────────────────────────────────
    xlsx = os.path.join(run_dir, "exit_lab.xlsx")
    recipes.to_csv(os.path.join(run_dir, "exit_lab_recipes.csv"), index=False)
    rep.to_csv(os.path.join(run_dir, "exit_lab_trades.csv"), index=False)
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        pd.DataFrame([
            {"key": "run_dir", "value": run_dir},
            {"key": "trades", "value": len(trades)},
            {"key": "self_check_committed_expR", "value": c_exp},
            {"key": "self_check_replay_expR", "value": r_exp},
            {"key": "self_check_diff", "value": diff},
            {"key": "self_check", "value": "PASS" if ok else "FAIL"},
            {"key": "generated_utc", "value": datetime.now(timezone.utc).isoformat()},
        ]).to_excel(xw, sheet_name="meta", index=False)
        recipes.to_excel(xw, sheet_name="recipes", index=False)
        per_pair.to_excel(xw, sheet_name="per_pair_expR", index=False)
        rep.to_excel(xw, sheet_name="per_trade", index=False)
    print(f"\nWrote: {xlsx}")
    print(f"       {os.path.join(run_dir, 'exit_lab_recipes.csv')}")
    print(f"       {os.path.join(run_dir, 'exit_lab_trades.csv')}")
    if not ok:
        raise SystemExit("SELF-CHECK FAILED — investigate before trusting the table")


if __name__ == "__main__":
    main()