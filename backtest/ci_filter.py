"""
CI-ON-FILTERS  (analysis tool — reads results only, never touches trading code)
Answers: "What is the 95% bootstrap CI on the best filter we have?"

Pipeline (single source of truth = the live headline book):
  - entry_zone == 'proximal'  AND  eligible_for_headline == True
  - BE+0.5 re-sim: any SL exit whose trade first ran >= +0.5R becomes a 0R scratch
    (BE moves stop to entry once +0.5R is touched; the price had to pass back through
    entry to hit SL, so the worst case is breakeven, not -1R).
  - Then layer the candidate gates and bootstrap the per-trade R.

Run:  python backtest/ci_filter.py
"""
import csv, random
from collections import defaultdict

QFILES = [
    ("Q3-24", "backtest/results/h1only_20240701_20240930"),
    ("Q4-24", "backtest/results/h1only_20241001_20241231"),
    ("Q1-25", "backtest/results/h1only_20250101_20250331"),
    ("Q2-25", "backtest/results/h1only_20250401_20250630"),
]
random.seed(42)


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def load():
    rows = []
    for q, d in QFILES:
        for r in csv.DictReader(open(f"{d}/trades.csv", newline="", encoding="utf-8")):
            if r["entry_zone"] == "proximal" and r["eligible_for_headline"].strip().lower() == "true":
                r["_q"] = q
                rows.append(r)
    return rows


def r_live(r):
    return f(r["r_realised"])


def r_be05(r):
    """BE moved to +0.5R instead of the committed +1.0R."""
    if r["exit_reason"] == "sl" and (f(r["mfe_r"]) or -9) >= 0.5:
        return 0.0
    return f(r["r_realised"])


def boot(vals, n=10000):
    if len(vals) < 2:
        return (None, None)
    means = []
    k = len(vals)
    for _ in range(n):
        s = sum(vals[random.randrange(k)] for _ in range(k)) / k
        means.append(s)
    means.sort()
    return (means[int(0.025 * n)], means[int(0.975 * n)])


def report(name, rows, rfn):
    vals = [rfn(r) for r in rows]
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        print(f"{name}: empty")
        return
    tot = sum(vals)
    exp = tot / n
    w = sum(1 for r in rows if (rfn(r) or 0) > 0)
    l = sum(1 for r in rows if (rfn(r) or 0) < 0)
    wr = 100 * w / (w + l) if (w + l) else 0
    lo, hi = boot(vals)
    # per-quarter expectancy sign
    qd = defaultdict(list)
    for r in rows:
        v = rfn(r)
        if v is not None:
            qd[r["_q"]].append(v)
    qsign = {q: round(sum(v) / len(v), 3) for q, v in sorted(qd.items())}
    pos_q = sum(1 for v in qsign.values() if v > 0)
    verdict = "CI excludes 0 (real)" if (lo is not None and (lo > 0 or hi < 0)) else "CI crosses 0 (UNPROVEN)"
    print(f"{name}")
    print(f"   N={n}  totR={tot:+.1f}  expR={exp:+.3f}  WR={wr:.0f}%  "
          f"95%CI=[{lo:+.3f}, {hi:+.3f}]  pos_qtrs={pos_q}/4  -> {verdict}")
    print(f"   per-qtr expR: {qsign}")


def main():
    rows = load()
    exNAS = [r for r in rows if r["pair"] != "NAS100"]
    fresh = [r for r in exNAS if r["fvg_state"] == "fresh"]
    aligned = [r for r in fresh if r["pd_alignment"] == "aligned"]

    print("=" * 78)
    print("SANITY — pooled live (BE+1R) should match handoff -53.5R / -0.157 / CI[-0.28,-0.03]")
    print("=" * 78)
    report("pooled LIVE (BE+1R, all pairs)", rows, r_live)

    print("\n" + "=" * 78)
    print("LAYERING THE GATES on COMMITTED exits (BE+1R) — fully reproducible")
    print("=" * 78)
    report("LIVE + ex-NAS", exNAS, r_live)
    report("LIVE + ex-NAS + fresh-FVG", fresh, r_live)
    report("LIVE + ex-NAS + fresh-FVG + PD-aligned  <-- BEST FILTER (on live exits)", aligned, r_live)

    print("\nNOTE: BE+0.5 NOT reconstructable from trades.csv (same-bar +0.5-then-SL sweeps")
    print("      never arm BE; no per-bar MFE column). The handoff's BE+0.5 numbers come")
    print("      from this session's bar-level re-sim, which must be re-run for that CI.")


if __name__ == "__main__":
    main()
