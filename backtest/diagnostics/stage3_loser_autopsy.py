"""
STAGE 3 — LOSER AUTOPSY (Playbook Stage 3). READ, DON'T DECIDE.

Splits the LOSING trades into two deaths and shows how each feature's losers are
distributed across that split. No filtering, no scoring — Stage 3 is a look.

  - died-fast    = a loss whose max favourable excursion never reached 0.5R
                   (mfe_r < 0.5). Price basically went straight to the stop.
  - gave-it-back = a loss that DID reach >=0.5R in our favour, then reversed and
                   stopped out.

Per user decision (2026-07-27): the cut lives at mfe_r = 0.5. But we do NOT reduce
losers to a binary — for every feature bucket we print the FULL mfe_r distribution
of its losers (min/p25/median/p75/max + died-fast share), so the 0.5 line is a
MARKER on a visible shape, never a filter that throws data away. A distribution
can't be gamed by a threshold choice.

Exit policy is a REQUIRED argument. This file crowns nothing — Stage 1 is still
deferred (see ANALYSIS_POINTERS.md CAVEAT). We run it once per policy and compare
by eye; a died-fast loss under baseline can be a different death under ATR.

  --exit baseline  -> label_baseline / r_baseline
  --exit atr       -> label_atr      / r_atr   (NA on ATR-undefined trades: skipped)

Data (no re-run): joins relabel.csv (labels, both exits) to trades.csv (features +
mfe_r) on the 6-col trade key. mfe_r is the trade's own replayed excursion — same
number both exits share, so the died-fast cut is defined on real order-replay, not
raw candle MFE.

Run:
  python -m backtest.diagnostics.stage3_loser_autopsy \
      --run-dir backtest/results/h1only_20080102_20161231 --exit baseline

Output: printed tables only (read-only stage). Nothing written to disk.
"""
import argparse
import os

import numpy as np
import pandas as pd

KEY = ["pair", "alert_ts", "fill_ts", "entry", "sl_initial", "tp1"]
DIED_FAST_CUT = 0.5  # mfe_r < this = died-fast (user decision 2026-07-27)

# Categorical features to autopsy (all confirmed present + clean in the canonical
# run header). Each becomes one bucket table. Numeric features get their own
# quantile-bucketed pass below.
CAT_FEATURES = [
    "session", "pd_zone", "break_tier", "sweep_present", "sweep2_present",
    "trend_pd_agree", "pd_alignment", "trade_toward_pool",
    "ob_in_killzone", "fill_in_killzone", "killzone_alignment",
    "weekly_pd_zone_at_alert", "pd_zone_agreement_at_alert",
]

# Numeric features -> bucketed into quantile bins on the LOSER population.
NUM_FEATURES = [
    "break_close_atr", "break_body_atr", "ob_age_h1_bars",
    "pd_pct", "tp1_rr",
]


def _fmt_pct(x):
    return "  n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:5.1f}"


def _dist_row(sub):
    """Full mfe_r distribution + died-fast share for one bucket of losers."""
    m = sub["mfe_r"]
    n = len(sub)
    df_share = 100.0 * (m < DIED_FAST_CUT).mean() if n else float("nan")
    return {
        "N": n,
        "died_fast_pct": df_share,
        "mfe_min": m.min() if n else float("nan"),
        "mfe_p25": m.quantile(0.25) if n else float("nan"),
        "mfe_med": m.median() if n else float("nan"),
        "mfe_p75": m.quantile(0.75) if n else float("nan"),
        "mfe_max": m.max() if n else float("nan"),
    }


def _print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'bucket':<22}{'N':>6}{'died-fast%':>12}"
          f"{'mfe_min':>9}{'mfe_p25':>9}{'mfe_med':>9}{'mfe_p75':>9}{'mfe_max':>9}")
    for label, r in rows:
        print(f"  {str(label):<22}{r['N']:>6}{_fmt_pct(r['died_fast_pct']):>12}"
              f"{r['mfe_min']:>9.2f}{r['mfe_p25']:>9.2f}{r['mfe_med']:>9.2f}"
              f"{r['mfe_p75']:>9.2f}{r['mfe_max']:>9.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--exit", required=True, choices=["baseline", "atr"],
                    help="which frozen exit's labels to autopsy")
    args = ap.parse_args()

    lbl_col = f"label_{args.exit}"
    r_col = f"r_{args.exit}"

    rel_p = os.path.join(args.run_dir, "relabel.csv")
    trd_p = os.path.join(args.run_dir, "trades.csv")
    for p in (rel_p, trd_p):
        if not os.path.exists(p):
            ap.error(f"missing {p}")

    rel = pd.read_csv(rel_p)
    trd = pd.read_csv(trd_p, low_memory=False)

    # Join labels onto features. relabel.csv is the label authority; trades.csv the
    # feature authority. Inner join on the unique 6-col key.
    keep_feats = [f for f in (CAT_FEATURES + NUM_FEATURES) if f in trd.columns]
    df = rel[KEY + [lbl_col, r_col, "quarter"]].merge(
        trd[KEY + ["mfe_r"] + keep_feats], on=KEY, how="inner", validate="one_to_one")

    n_labelled = df[lbl_col].notna().sum()
    losers = df[df[lbl_col] == "loss"].copy()
    n_loss = len(losers)

    print("=" * 92)
    print(f"STAGE 3 LOSER AUTOPSY  |  exit policy = {args.exit.upper()}  "
          f"({lbl_col})  |  run = {os.path.basename(args.run_dir)}")
    print("=" * 92)
    print(f"trades joined:        {len(df)}")
    print(f"labelled (this exit): {n_labelled}"
          + ("" if args.exit == "baseline"
             else f"  ({len(df) - n_labelled} ATR-undefined skipped)"))
    print(f"LOSERS:               {n_loss}")
    if n_loss == 0:
        print("no losers under this exit — nothing to autopsy.")
        return

    overall = _dist_row(losers)
    print(f"\nALL LOSERS  mfe_r:  died-fast(<{DIED_FAST_CUT})={_fmt_pct(overall['died_fast_pct'])}%  "
          f"min={overall['mfe_min']:.2f} p25={overall['mfe_p25']:.2f} "
          f"med={overall['mfe_med']:.2f} p75={overall['mfe_p75']:.2f} "
          f"max={overall['mfe_max']:.2f}")
    print(f"(died-fast = went <0.5R in favour before the stop; "
          f"gave-it-back = reached >=0.5R then reversed)")

    # ── Categorical features ──
    for f in CAT_FEATURES:
        if f not in losers.columns:
            continue
        rows = []
        col = losers[f]
        for val in sorted(col.dropna().unique(), key=lambda x: str(x)):
            rows.append((val, _dist_row(losers[col == val])))
        n_na = col.isna().sum()
        if n_na:
            rows.append(("<NA>", _dist_row(losers[col.isna()])))
        _print_table(f"[{f}]", rows)

    # ── Numeric features: quantile buckets on the loser population ──
    for f in NUM_FEATURES:
        if f not in losers.columns:
            continue
        col = pd.to_numeric(losers[f], errors="coerce")
        valid = col.dropna()
        if valid.nunique() < 4:
            continue
        try:
            bins = pd.qcut(valid, 4, duplicates="drop")
        except ValueError:
            continue
        rows = []
        for interval in bins.cat.categories:
            mask = (col >= interval.left) & (col <= interval.right)
            rows.append((f"{interval.left:.2f}..{interval.right:.2f}",
                         _dist_row(losers[mask])))
        n_na = col.isna().sum()
        if n_na:
            rows.append(("<NA>", _dist_row(losers[col.isna()])))
        _print_table(f"[{f}] (quartiles)", rows)

    print("\n" + "=" * 92)
    print("READ-ONLY. No filter, no score applied. Stage 3 forms hypotheses only.")
    print("=" * 92)


if __name__ == "__main__":
    main()
