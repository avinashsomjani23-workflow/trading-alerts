"""
BASELINE FREEZE — the frozen FIXED-2R baseline for the current canonical run.

Regenerates the exact baseline table from the canonical CSV and ASSERTS the frozen
numbers, so the baseline can never silently drift. If any number moves (detection
re-run, population-rule change, wrong file), this SCREAMS instead of quietly lying.

Populations (see docs/LOSER_AUTOPSY_PLAYBOOK.md §1 + CLAUDE.md standing rules):
  - Resolved only: exit_reason in {tp, sl}. never_filled / friday_flat / timeout /
    window_end are AUDIT-ONLY, excluded always (they corrupt WR / meanR).
  - LOOSE  = news_fill==0            -> the FROZEN baseline / tradeable universe / EV.
             News is informational, never a live gate (h1_only_reporting.py), so
             news-open trades are real trades the system really takes -> kept.
  - STRICT = news_fill==0 & news_open==0 -> the loser-autopsy / feature-screen pop.

The LOOSE row is THE baseline. Frozen 2026-08-02 against canonical
`h1only_20080102_20161231` (EURUSD Discovery 2008-2016, FIXED 2R, 180 x 3322).
(Shape corrected 2026-08-02 from a stale 182: the `ranging delete` dropped
`structure_ranging_at_alert` and `entry_raw` was retired, leaving 180. The frozen
tier numbers below derive from r_realised + news columns, which the column-count
change does not touch, so they stand.)

Run (from repo root):
  python -m backtest.diagnostics.baseline_freeze
Exit code 0 = numbers match the freeze. Non-zero = drift; investigate before trusting
any downstream analysis.
"""
import sys

import pandas as pd

# --- what the canonical run must be (repoint here + re-freeze on a new run) ---
CANONICAL_CSV = "backtest/results/h1only_20080102_20161231/trades.csv"
EXPECT_SHAPE = (3322, 180)

# --- the FROZEN baseline (LOOSE) + the tiers around it. re-derive on any re-run. ---
# Each: (N, wins, losses, WR%, meanR, sumR) rounded as printed below.
FROZEN = {
    "all_resolved": dict(N=2153, W=641, L=1512, wr=29.8, meanR=-0.107, sumR=-230.0),
    "LOOSE":        dict(N=1477, W=445, L=1032, wr=30.1, meanR=-0.096, sumR=-142.0),
    "STRICT":       dict(N=1254, W=349, L=905,  wr=27.8, meanR=-0.165, sumR=-207.0),
    "news_open_seg":dict(N=223,  W=96,  L=127,  wr=43.0, meanR=+0.291, sumR=+65.0),
}
# meanR/sumR tolerance for float noise; counts + WR must be exact.
TOL_MEANR = 0.001
TOL_SUMR = 0.5


def _tier(d):
    w = int((d.exit_reason == "tp").sum())
    l = int((d.exit_reason == "sl").sum())
    wr = round(100.0 * w / (w + l), 1) if (w + l) else float("nan")
    return dict(N=len(d), W=w, L=l, wr=wr,
                meanR=round(d.r_realised.mean(), 3),
                sumR=round(d.r_realised.sum(), 1))


def _row(name, got):
    return (f"  {name:<22} N={got['N']:<5} W={got['W']:<4} L={got['L']:<5} "
            f"WR={got['wr']:5.1f}%  meanR={got['meanR']:+.3f}  sumR={got['sumR']:+8.1f}")


def _cmp(name, got, exp):
    """Return list of human-readable mismatches (empty = match)."""
    bad = []
    for k in ("N", "W", "L", "wr"):
        if got[k] != exp[k]:
            bad.append(f"{name}.{k}: got {got[k]} != frozen {exp[k]}")
    if abs(got["meanR"] - exp["meanR"]) > TOL_MEANR:
        bad.append(f"{name}.meanR: got {got['meanR']:+.3f} != frozen {exp['meanR']:+.3f}")
    if abs(got["sumR"] - exp["sumR"]) > TOL_SUMR:
        bad.append(f"{name}.sumR: got {got['sumR']:+.1f} != frozen {exp['sumR']:+.1f}")
    return bad


def main():
    df = pd.read_csv(CANONICAL_CSV, low_memory=False)
    if df.shape != EXPECT_SHAPE:
        print(f"STOP — canonical shape {df.shape} != frozen {EXPECT_SHAPE}. "
              f"Wrong file or a re-run happened. Re-freeze before trusting numbers.")
        return 2

    res = df[df.exit_reason.isin(["tp", "sl"])].copy()
    loose = res[res.news_fill == 0]
    got = {
        "all_resolved": _tier(res),
        "LOOSE":        _tier(loose),
        "STRICT":       _tier(res[(res.news_fill == 0) & (res.news_open == 0)]),
        "news_open_seg":_tier(loose[loose.news_open == 1]),
    }

    print("=" * 78)
    print(f"BASELINE FREEZE  |  canonical {CANONICAL_CSV.split('/')[-2]}  |  shape {df.shape}")
    print("=" * 78)
    print(_row("all resolved", got["all_resolved"]))
    print(_row("LOOSE  <-- THE BASELINE", got["LOOSE"]))
    print(_row("STRICT (autopsy pop)", got["STRICT"]))
    print(_row("  +65 news_open segment", got["news_open_seg"]))

    problems = []
    for name in FROZEN:
        problems += _cmp(name, got[name], FROZEN[name])

    print("-" * 78)
    if problems:
        print("DRIFT DETECTED — the baseline is NOT what was frozen:")
        for p in problems:
            print("  ! " + p)
        print("Do NOT trust downstream analysis until this is explained + re-frozen.")
        return 1

    print("FROZEN BASELINE VERIFIED — all numbers match. WR 30.1%, meanR -0.096, "
          "sumR -142 (LOOSE).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
