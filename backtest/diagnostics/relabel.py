"""
RELABEL (Playbook Stage 2) — assign every Discovery trade a loss/BE/win label
under a FROZEN exit, on the completed backtest. No re-run.

Three label sets, kept physically separate so the next session cannot confuse them
or contaminate one with another:

  - PRIMARY   = baseline exit (live: liquidity TP + BE@1R). Column `label_baseline`.
  - SHADOW    = mechanical ATR exit (E_atr_sl1.5_tp2.5). Column `label_atr`.
  - FIXED-1R  = full position exits at exactly +1.0R, no BE (C_fullTP_1.0R).
                Column `label_fulltp1r`.

All come from exit_lab_trades.csv, which holds one row per (trade, config). The
`config` column is the firewall: label_baseline is read ONLY from
config==baseline_liqTP_be1.0 rows, label_atr ONLY from config==E_atr_sl1.5_tp2.5
rows, label_fulltp1r ONLY from config==C_fullTP_1.0R rows. No cell is shared;
no label is ever derived from another's R.

Non-contamination rules (hard):
  - The ATR recipe emits NaN r on trades with <15 pre-fill bars (ATR undefined).
    Those trades get label_atr = NA. We NEVER borrow the baseline label to fill an
    ATR gap. A missing ATR label stays missing (r_atr also NA). The same no-borrow
    rule applies to label_fulltp1r (though the fixed-1R recipe always resolves).
  - label_baseline exists for every filled proximal trade (baseline always resolves).
  - Per-trade self-check: baseline r must equal the trade's committed r_realised;
    any mismatch aborts (the label set is not trustworthy).

Label rule: win if r > 0, loss if r < 0, breakeven (BE) if r == 0. (Matches the
WR convention: breakevens are their own class, excluded from win rate.)

Run:
  python -m backtest.diagnostics.relabel --run-dir backtest/results/h1only_20080102_20161231
Output (next to trades.csv): relabel.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

BASELINE_CFG = "baseline_liqTP_be1.0"
ATR_CFG = "E_atr_sl1.5_tp2.5"
FULLTP1R_CFG = "C_fullTP_1.0R"  # full position exits at exactly +1.0R, no BE

# Unique per trade in the replay file. alert_ts+fill_ts alone is NOT unique — two
# different order blocks can fire on the same bar and fill on the same bar (26 such
# collisions here), so entry/sl/tp1 (which differ per OB) are part of the key.
KEY = ["pair", "alert_ts", "fill_ts", "entry", "sl_initial", "tp1"]


def _label(r):
    if pd.isna(r):
        return pd.NA
    if r > 0:
        return "win"
    if r < 0:
        return "loss"
    return "BE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    rep_p = os.path.join(args.run_dir, "exit_lab_trades.csv")
    if not os.path.exists(rep_p):
        ap.error(f"no exit_lab_trades.csv at {rep_p} — run exit_lab first")
    rep = pd.read_csv(rep_p)

    base = rep[rep["config"] == BASELINE_CFG].copy()
    atr = rep[rep["config"] == ATR_CFG].copy()
    fulltp1r = rep[rep["config"] == FULLTP1R_CFG].copy()
    if base.empty:
        ap.error(f"no {BASELINE_CFG} rows in the replay file")

    # Key must be unique per config, or a merge would fan out and cross-contaminate.
    for nm, df in (("baseline", base), ("atr", atr), ("fulltp1r", fulltp1r)):
        dup = df.duplicated(KEY).sum()
        if dup:
            ap.error(f"{dup} duplicate keys in {nm} rows — key not unique, abort")

    out = base[KEY + ["quarter", "direction", "committed_r"]].copy()
    out = out.rename(columns={"committed_r": "committed_r_realised"})

    # PRIMARY: baseline. r_baseline is baseline's own replayed R (== committed).
    out["r_baseline"] = base["r"].values
    out["label_baseline"] = [_label(r) for r in out["r_baseline"]]

    # SHADOW: ATR. Left-merge so trades with NO atr row (never happens — every
    # trade gets a row, NaN-r where ATR is undefined) or NaN r stay NA. We
    # explicitly do NOT fill from baseline.
    atr_slim = atr[KEY + ["r"]].rename(columns={"r": "r_atr"})
    out = out.merge(atr_slim, on=KEY, how="left")
    out["label_atr"] = [_label(r) for r in out["r_atr"]]

    # ── CONTAMINATION GUARD: r_atr NaN => label_atr must be NA (never borrowed) ──
    borrowed = ((out["r_atr"].isna()) & (out["label_atr"].notna())).sum()
    if borrowed:
        raise SystemExit(f"CONTAMINATION: {borrowed} ATR labels present without an "
                         f"ATR r — aborting")

    # FIXED-1R: full position exits at exactly +1.0R (C_fullTP_1.0R). Same left-merge
    # + no-borrow discipline as the ATR shadow. This recipe always resolves, so the
    # column is normally fully populated, but the guard below stays structurally
    # identical so a label can never be borrowed from baseline.
    fulltp1r_slim = fulltp1r[KEY + ["r"]].rename(columns={"r": "r_fulltp1r"})
    out = out.merge(fulltp1r_slim, on=KEY, how="left")
    out["label_fulltp1r"] = [_label(r) for r in out["r_fulltp1r"]]

    # ── CONTAMINATION GUARD: r_fulltp1r NaN => label_fulltp1r must be NA ──
    borrowed_ft = ((out["r_fulltp1r"].isna()) & (out["label_fulltp1r"].notna())).sum()
    if borrowed_ft:
        raise SystemExit(f"CONTAMINATION: {borrowed_ft} fixed-1R labels present "
                         f"without a fixed-1R r — aborting")

    # ── SELF-CHECK: baseline r must match committed r_realised, per trade ──
    # committed_r_realised is stored to 3 dp in trades.csv; the replay is full
    # precision. So allow up to the stored rounding (1.1e-3) on the NUMBER, but
    # require ZERO disagreement on the LABEL (a win/loss/BE flip is the real bug;
    # a 0.0005 rounding gap is not).
    d = (out["r_baseline"] - out["committed_r_realised"]).abs()
    num_bad = int((d > 1.1e-3).sum())
    if num_bad:
        raise SystemExit(f"SELF-CHECK FAILED: {num_bad} trades where baseline r "
                         f"differs from committed r_realised by >1e-3 — labels NOT "
                         f"trustworthy")
    committed_label = [_label(r) for r in out["committed_r_realised"]]
    label_bad = int((pd.Series(committed_label) != out["label_baseline"]).sum())
    if label_bad:
        raise SystemExit(f"SELF-CHECK FAILED: {label_bad} trades where the baseline "
                         f"label disagrees with the committed r_realised sign — abort")

    out_p = os.path.join(args.run_dir, "relabel.csv")
    out.to_csv(out_p, index=False)

    n = len(out)
    n_atr = int(out["label_atr"].notna().sum())
    n_ft = int(out["label_fulltp1r"].notna().sum())
    print(f"Relabel written: {out_p}")
    print(f"  {n} trades  (baseline labels: {n}/{n}, ATR labels: {n_atr}/{n}; "
          f"{n - n_atr} ATR-undefined kept NA — NOT borrowed; "
          f"fixed-1R labels: {n_ft}/{n})")
    print("  baseline self-check vs committed r_realised: PASS")
    print("\n  label_baseline:")
    print(out["label_baseline"].value_counts(dropna=False).to_string())
    print("  label_atr:")
    print(out["label_atr"].value_counts(dropna=False).to_string())
    print("  label_fulltp1r:")
    print(out["label_fulltp1r"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
