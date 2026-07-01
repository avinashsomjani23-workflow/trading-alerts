"""Step 1 of the sweep rebuild (SWEEP_REBUILD_HANDOFF §2): reproduce the
claimed INVERSE correlation between sweep presence and trade results from
r_realised — the single P&L source of truth.

Read-only. No redesign. Pools non-overlapping quarter runs, dedupes, filters to
headline-eligible filled trades, then slices by sweep present/absent, by
sweep_pts bucket (tier proxy), and by pair / asset class. Prints expectancy + WR.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Non-overlapping windows that each still carry a raw trades.csv (the big
# multi-month runs were slimmed on commit). Covers Jul-2024 -> mid-2026.
RUNS = [
    "h1only_20240701_20240930",
    "h1only_20241001_20241231",
    "h1only_20250101_20250331",
    "h1only_20250401_20250630",
    "h1only_20250701_20251231",
    "h1only_20260222_20260522",
    "h1only_20260601_20260619",
]

FX_NON_JPY = {"EURUSD", "NZDUSD", "USDCHF"}
GRADED = {"USDJPY", "GOLD", "XAUUSD", "NAS100"}


def _load() -> pd.DataFrame:
    frames = []
    for r in RUNS:
        p = RESULTS / r / "trades.csv"
        if not p.exists():
            print(f"  [skip] {r} (no trades.csv)", file=sys.stderr)
            continue
        df = pd.read_csv(p)
        df["__run"] = r
        frames.append(df)
    if not frames:
        sys.exit("no trades.csv found")
    df = pd.concat(frames, ignore_index=True)
    # Dedupe across overlapping windows on the trade identity.
    key = [c for c in ("pair", "alert_ts", "entry_zone", "entry") if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=key, keep="first")
    print(f"loaded {before} rows -> {len(df)} after dedupe on {key}\n")
    return df


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0", "yes"])


def _stats(sub: pd.DataFrame) -> str:
    n = len(sub)
    if n == 0:
        return f"{'':>6} n=0"
    r = sub["r_realised"].astype(float)
    exp = r.mean()
    wins = (r > 0).sum()
    losses = (r < 0).sum()
    wr = wins / (wins + losses) * 100 if (wins + losses) else float("nan")
    total = r.sum()
    return (f"n={n:<4} exp={exp:+.3f}R  WR={wr:4.1f}%  "
            f"totalR={total:+7.1f}  (W{wins}/L{losses}/BE{n - wins - losses})")


def _block(title: str, df: pd.DataFrame, mask) -> None:
    print(f"== {title} ==")
    print(f"  sweep PRESENT : {_stats(df[mask])}")
    print(f"  sweep ABSENT  : {_stats(df[~mask])}")
    d_exp = df[mask]["r_realised"].mean() - df[~mask]["r_realised"].mean()
    print(f"  delta (present - absent) exp = {d_exp:+.3f}R\n")


def main() -> None:
    df = _load()
    elig = _truthy(df["eligible_for_headline"])
    df = df[elig].copy()
    print(f"headline-eligible filled trades: {len(df)}\n")

    df["sweep_present_b"] = _truthy(df["sweep_present"])
    df["sweep_pts"] = pd.to_numeric(df["sweep_pts"], errors="coerce").fillna(0.0)
    present = df["sweep_present_b"]

    _block("ALL INSTRUMENTS", df, present)

    print("-- by asset class --")
    fx = df[df["pair"].isin(FX_NON_JPY)]
    _block("FX non-JPY (presence-only consumption)", fx, fx["sweep_present_b"])
    gr = df[df["pair"].isin(GRADED)]
    _block("JPY / Gold / NAS (graded consumption)", gr, gr["sweep_present_b"])

    print("-- by pair --")
    for pair in sorted(df["pair"].unique()):
        sub = df[df["pair"] == pair]
        m = sub["sweep_present_b"]
        print(f"  {pair:<8} present {_stats(sub[m])}")
        print(f"  {'':<8} absent  {_stats(sub[~m])}")
    print()

    print("-- by sweep_pts bucket (graded pairs only; tier proxy) --")
    for lo, hi, lbl in [(0.01, 0.5, "weak"), (0.5, 1.0, "decent"),
                        (1.0, 99, "strong")]:
        sub = gr[(gr["sweep_pts"] >= lo) & (gr["sweep_pts"] < hi)]
        print(f"  {lbl:<8}[{lo:.2f},{hi:.2f}) {_stats(sub)}")
    print(f"  none    pts=0       {_stats(gr[gr['sweep_pts'] == 0])}")


if __name__ == "__main__":
    main()
