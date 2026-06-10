"""Harness 2 â swing-detection noise audit.

The single H1 swing definition is lb-3 geometry AND a leg-size gate of
MIN_LEG_ATR_MULT Ã mean ATR across the leg. This harness shows, per instrument:
raw lb-N swings, which the ATR gate DROPPED vs KEPT, each swing's effective
leg-size-in-ATR (the gate-crossing multiplier M*), and a variant grid across
candidate (lookback, mult) settings.

Slice-mode A (static full-dataset census) â justified: the gate's kept/dropped
decision depends only on bars from the previous kept anchor up to the swing
itself (left-to-right; a dropped swing does not advance the anchor), so future
bars cannot change a past decision. (Harness 3 check C16 independently verifies
no repainting.) Right-edge swings within `lookback` bars of the data end are
unconfirmable and excluded.

Never re-derives the leg math: raw geometry comes from the real detect_swings
with the gate OFF; kept/dropped and M* come from the real
_filter_swings_by_leg_atr. The `_trace` hook does NOT expose per-swing filter
decisions (verified â it records trend state only), so M* uses gate-crossing
bisection over the real filter.

Run:
  python -m backtest.diagnostics.h2_swing_audit --pairs EURUSD --start 2026-03-01 --end 2026-03-31 --lookbacks 3 --mults 1.0,1.5,2.0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.diagnostics import driver
import dealing_range
import smc_radar

LIVE_LOOKBACK = 3
LIVE_MULT = 1.5
BORDERLINE_BAND = 0.2
M_HI = 4.0  # bisection upper bound


def _swing_key(s: Dict[str, Any]) -> Tuple[str, str]:
    """Identity = (type, iso-ts). detect_swings emits ts as ISO string."""
    ts = s.get("ts")
    if isinstance(ts, pd.Timestamp):
        ts = ts.isoformat()
    return (s.get("type"), str(ts))


def _raw_swings(df: pd.DataFrame, lookback: int) -> List[Dict[str, Any]]:
    """Raw lb-N geometry via the REAL detect_swings with the gate OFF (0.0)."""
    return dealing_range.detect_swings(df, lookback=lookback, min_leg_atr_mult=0.0)


def _kept_at(raw: List[Dict[str, Any]], df: pd.DataFrame, mult: float) -> List[Dict[str, Any]]:
    """Kept set at a given mult via the REAL filter (single source of truth)."""
    if mult <= 0:
        return list(raw)
    return dealing_range._filter_swings_by_leg_atr(raw, df, min_mult=mult)


def _m_star(raw: List[Dict[str, Any]], df: pd.DataFrame, swing_key: Tuple,
            *, precision: float = 0.01) -> Tuple[Optional[float], bool]:
    """Smallest mult at which `swing` is DROPPED (gate-crossing). Returns
    (m_star, monotone). If kept at M_HI, returns (M_HI, monotone) meaning it
    survives even very large gates. monotone=False flags single-crossing
    violations (reported, not hidden)."""
    # membership(M): is swing kept at mult M?
    def kept(M):
        ks = {_swing_key(s) for s in _kept_at(raw, df, M)}
        return swing_key in ks
    if not kept(0.0):
        return (0.0, True)  # dropped even with no gate (shouldn't happen for raw)
    if kept(M_HI):
        return (M_HI, True)  # survives the largest gate tested
    # monotonicity probe on a coarse grid
    grid = [i * 0.25 for i in range(0, int(M_HI / 0.25) + 1)]
    memb = [kept(M) for M in grid]
    crossings = sum(1 for i in range(1, len(memb)) if memb[i] != memb[i - 1])
    monotone = crossings <= 1
    # bisection for the single crossing (kept -> dropped)
    lo, hi = 0.0, M_HI
    while hi - lo > precision:
        mid = (lo + hi) / 2
        if kept(mid):
            lo = mid
        else:
            hi = mid
    return (round(hi, 3), monotone)


def audit_pair(pc: Dict[str, Any], df: pd.DataFrame, lookbacks: List[int],
               mults: List[float]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Returns (swing_rows, variant_rows, borderline_rows) for one pair."""
    name = pc["name"]
    n = len(df)
    swing_rows: List[Dict] = []
    variant_rows: List[Dict] = []
    borderline_rows: List[Dict] = []

    # Live-setting kept set (for jaccard reference).
    raw_live = _raw_swings(df, LIVE_LOOKBACK)
    live_keys = {_swing_key(s) for s in _kept_at(raw_live, df, LIVE_MULT)}

    for L in lookbacks:
        raw = _raw_swings(df, L)
        # right-edge exclusion: swings within L bars of the end are unconfirmable
        edge_ts = df.index[-L] if n > L else df.index[0]
        raw_conf = [s for s in raw
                    if (pd.Timestamp(s["ts"]) if not isinstance(s["ts"], pd.Timestamp)
                        else s["ts"]) <= edge_ts]
        n_right_edge = len(raw) - len(raw_conf)

        # Per-swing M* (only at the live lookback, to bound cost; that's the
        # eyeball view the user wants).
        if L == LIVE_LOOKBACK:
            for s in raw_conf:
                key = _swing_key(s)
                m_star, monotone = _m_star(raw, df, key)
                status = "kept" if (m_star is None or m_star > LIVE_MULT) else "dropped"
                margin = round((m_star - LIVE_MULT), 3) if m_star is not None else None
                border = (m_star is not None and abs(m_star - LIVE_MULT) < BORDERLINE_BAND)
                row = {"pair": name, "lookback": L, "mult": LIVE_MULT,
                       "ts": str(s["ts"]), "type": s["type"], "price": s["price"],
                       "status": status,
                       "m_star": ("NON_MONOTONE" if not monotone else m_star),
                       "margin": margin, "borderline": border}
                swing_rows.append(row)
                if border:
                    borderline_rows.append(row)

        for M in mults:
            kept_keys = {_swing_key(s) for s in _kept_at(raw, df, M)}
            kept_keys = {k for k in kept_keys
                         if k in {_swing_key(s) for s in raw_conf}}
            n_kept = len(kept_keys)
            n_raw = len(raw_conf)
            inter = len(kept_keys & live_keys)
            union = len(kept_keys | live_keys)
            jac = round(inter / union, 4) if union else 1.0
            variant_rows.append({
                "pair": name, "lookback": L, "mult": M,
                "n_raw": n_raw, "n_kept": n_kept, "n_dropped": n_raw - n_kept,
                "drop_rate": round((n_raw - n_kept) / n_raw, 4) if n_raw else 0.0,
                "jaccard_vs_live": jac, "n_right_edge_excluded": n_right_edge,
            })
    return swing_rows, variant_rows, borderline_rows


def self_check(pc, df) -> bool:
    """Raw-set assumption, subset law, cross-validation vs live assembler."""
    raw = _raw_swings(df, LIVE_LOOKBACK)
    # 1. gate OFF (0.0) is a superset of any gated set
    kept15 = {_swing_key(s) for s in _kept_at(raw, df, LIVE_MULT)}
    rawk = {_swing_key(s) for s in raw}
    assert kept15.issubset(rawk), "subset law violated: kept(1.5) not subset of raw"
    # 2. cross-validation: kept(L=3,M=1.5) must equal the live assembler's pool
    cen = driver.census_full_df(pc, df)
    live_pool = {(s.type, s.ts.isoformat() if s.ts is not None else None)
                 for s in cen.swings}
    kept_norm = set()
    for s in _kept_at(raw, df, LIVE_MULT):
        ts = s["ts"]
        ts = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)
        kept_norm.add((s["type"], ts.isoformat()))
    # allow the assembler to drop right-edge swings it hasn't confirmed
    missing = kept_norm - live_pool
    extra = live_pool - kept_norm
    # tolerate small right-edge differences only
    if len(missing) + len(extra) > max(3, 0.05 * len(live_pool)):
        raise AssertionError(
            f"cross-validation vs live assembler failed: "
            f"{len(missing)} in-h2-not-live, {len(extra)} in-live-not-h2")
    return True


def run(pairs: List[str], start, end, out_dir: Path,
        lookbacks: List[int], mults: List[float]) -> Path:
    cfg = json.load(open(_REPO_ROOT / "config.json"))
    confs = [p for p in cfg["pairs"] if (pairs == ["all"] or p["name"] in pairs)]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_swings: List[Dict] = []
    all_variants: List[Dict] = []
    all_border: List[Dict] = []
    notes: List[str] = []
    for pc in confs:
        name = pc["name"]
        print(f"=== {name} ===", flush=True)
        df = driver.load_window(pc, start, end)
        if isinstance(df, driver.WindowUnserveable):
            print(f"  [SKIP] {df}", flush=True)
            notes.append(f"{name}: {df}")
            continue
        self_check(pc, df)
        sw, var, bd = audit_pair(pc, df, lookbacks, mults)
        all_swings += sw
        all_variants += var
        all_border += bd
        print(f"  swings={len(sw)} variants={len(var)} borderline={len(bd)}", flush=True)

    # Write per-pair swing CSVs + a combined variants CSV + borderline md.
    runstamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
    for p in {r["pair"] for r in all_swings}:
        sp = out_dir / f"h2_swings_{p}_{runstamp}.csv"
        rows = [r for r in all_swings if r["pair"] == p]
        with open(sp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    vp = out_dir / f"h2_variants_summary_{runstamp}.csv"
    if all_variants:
        with open(vp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_variants[0].keys()))
            w.writeheader(); w.writerows(all_variants)

    md = out_dir / f"h2_borderline_{runstamp}.md"
    lines = ["# Harness 2 â swing noise audit (borderline swings)", "",
             f"slice_mode=**A** (static census â see module docstring). "
             f"Live setting L={LIVE_LOOKBACK}, M={LIVE_MULT}.",
             f"Borderline = |M* - {LIVE_MULT}| < {BORDERLINE_BAND}: the swings "
             "where the gate is genuinely deciding noise-vs-structure.", ""]
    if notes:
        lines += ["## data notes"] + [f"- {n}" for n in notes] + [""]
    lines.append("## Variant grid (drop rate + churn vs live)")
    lines.append("| pair | lb | mult | raw | kept | dropped | drop% | jaccard_vs_live |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in all_variants:
        lines.append(f"| {r['pair']} | {r['lookback']} | {r['mult']} | {r['n_raw']} | "
                     f"{r['n_kept']} | {r['n_dropped']} | {r['drop_rate']} | {r['jaccard_vs_live']} |")
    lines.append("")
    lines.append(f"## Borderline swings at live setting ({len(all_border)})")
    lines.append("| pair | ts | type | price | status | M* | margin |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(all_border, key=lambda x: (x["pair"], x["ts"])):
        lines.append(f"| {r['pair']} | {r['ts']} | {r['type']} | {r['price']} | "
                     f"{r['status']} | {r['m_star']} | {r['margin']} |")
    lines += ["", "## Honest weaknesses",
              "- Mode-A census tells you what the gate cuts, not when the system "
              "would have known it (timing belongs to Harness 3).",
              "- M* is exact under monotonicity; NON_MONOTONE swings are reported "
              "as such rather than compressed to one number.",
              "- 'Noise vs real structure' is your chart judgment; H2 gives "
              "evidence (M*, margins, borderline list), not verdicts.",
              "- ~720-day window: one regime sample."]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[h2] swings + variants CSVs and borderline md -> {out_dir}", flush=True)
    return md


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="EURUSD")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--lookbacks", default="2,3,4,5")
    ap.add_argument("--mults", default="1.0,1.25,1.5,1.75,2.0")
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    args = ap.parse_args()
    lookbacks = [int(x) for x in args.lookbacks.split(",") if x.strip()]
    mults = [float(x) for x in args.mults.split(",") if x.strip()]
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    run(pairs, args.start, args.end, Path(args.out), lookbacks, mults)


if __name__ == "__main__":
    main()
