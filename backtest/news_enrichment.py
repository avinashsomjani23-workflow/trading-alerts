"""Offline news-flag enrichment for backtest trades.csv.

Adds five columns to an existing per-trade CSV (in place, atomic, idempotent):

    news_fill        1 / 0 / blank(None)  -- a HIGH-impact scheduled event on
                     either currency leg landed within +/-1h of the fill
                     candle. The fill timestamp is the H1 bar OPEN; the true
                     fill is anywhere inside that hour, so the check window is
                     [fill_open - 1h, fill_open + 2h] inclusive: no true +/-1h
                     violation can slip through (some 1-2h-away events
                     over-flag; that errs toward removing contamination).
    news_fill_event  title of the nearest qualifying event (audit receipt)
    news_fill_ccy    currency of that event
    news_open        1 / 0 / blank(None)  -- a HIGH-impact event landed while
                     the trade was open: [fill_open, exit_bar_open + 1h]
                     inclusive. Live (prop-firm news rule) this trade would
                     have been force-closed before the event; the backtest let
                     it ride, so its P&L is optimistic vs live.
    news_open_event  title of the first such event (audit receipt)

None (empty cell) means "cannot know", never "safe":
    - never_filled rows (no fill to anchor)
    - the trade window falls outside events-data coverage, or in a detected
      coverage hole (a calendar month with zero events of ANY impact for a
      required currency)
    - fill lands in the 2014-11-01..2014-12-07 broker-clock regime-flip
      window (mt5_clock.is_flip_window — those candle labels cannot be
      resolved to true UTC, so they stay provisional and get NULL flags)
    - news_open only: filled but exit_ts missing

Clock: fill_ts / exit_ts are candle labels. As of 2026-07-16 (Part B) the MT5
importer corrects the broker-clock label error AT SOURCE (mt5_clock era table),
so those timestamps arrive as TRUE UTC and this module does NOT correct them
again — it only NULLs the 2014 flip-window rows (is_flip_window). This is correct
ONLY on a run built with the fixed importer; do not re-run it over the pre-fix
canonical CSV (whose labels are still provisional). See mt5_clock.py.

Event source: CSV produced from ForexFactory month pages. Timestamps are the
page-embedded `dateline` epochs (absolute UTC seconds) -- no timezone
conversion happens anywhere in this pipeline. Scheduled events only; FF does
not backfill unscheduled shocks, matching the spec.

Exclusions (per owner decision 2026-07-16):
    - speeches/testimony (title matches _SPEECH_RE) -- start time != the
      market-moving moment; press conferences are KEPT (scheduled,
      recorded start, e.g. ECB presser)
    - untimed events (FF "All Day" / "Tentative" / date-range rows) -- no
      clock time exists; counted and reported, never flagged on

This is analysis tooling. It is NOT in the live alert path and NOT in the
simulator; it runs after a backtest over the finished CSV. Distinct concept
from the live alert-anchored `news_blocked`/`news_event_*` columns
(run_backtest.py:205-209), which use the current-week FairEconomy feed.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from typing import Dict, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from news_filter import currencies_for_pair  # single source for pair->legs
# Era table lives in ONE place (Part A, 2026-07-16). Candles are now corrected at
# import (Part B), so enrich() no longer calls true_utc; it only needs
# is_flip_window to keep the 2014 regime-flip rows NULL.
from mt5_clock import is_flip_window  # noqa: E402

# +/-1h rule, inclusive (owner spec). The fill anchor is the H1 bar open, so
# the true fill is in [open, open+1h): window = [open - W, open + 1h + W].
WINDOW_MINS = 60

_SPEECH_RE = re.compile(r"\bSpeaks\b|\bSpeech\b|Testif", re.IGNORECASE)
_TIMED_RE = re.compile(r"^\d{1,2}:\d{2}(am|pm)$", re.IGNORECASE)

NEW_COLS = ["news_fill", "news_fill_event", "news_fill_ccy",
            "news_open", "news_open_event"]


def load_events(path: str) -> pd.DataFrame:
    """Events CSV -> DataFrame with utc / currency / name / high / timed / speech."""
    ev = pd.read_csv(path)
    need = {"dateline", "currency", "name", "impactName", "timeLabel"}
    missing = need - set(ev.columns)
    if missing:
        raise ValueError(f"events file missing columns: {missing}")
    ev["utc"] = pd.to_datetime(ev["dateline"], unit="s", utc=True)
    ev["timed"] = ev["timeLabel"].astype(str).str.match(_TIMED_RE)
    ev["speech"] = ev["name"].astype(str).str.contains(_SPEECH_RE)
    ev["high"] = ev["impactName"].eq("high")
    return ev


def coverage_holes(events: pd.DataFrame, currencies) -> set:
    """(currency, 'YYYY-MM') months with ZERO events of ANY impact.

    The FF calendar always carries minor events for an active currency, so an
    empty month = missing data, not a quiet market. Trades touching a hole get
    None, never 0.
    """
    holes = set()
    per = events.assign(ym=events["utc"].dt.strftime("%Y-%m"))
    months = sorted(per["ym"].unique())
    for ccy in currencies:
        have = set(per.loc[per["currency"] == ccy, "ym"])
        holes.update((ccy, m) for m in months if m not in have)
    return holes


def _months_touched(start: pd.Timestamp, end: pd.Timestamp):
    return {p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")}


def enrich(trades: pd.DataFrame, events: pd.DataFrame,
           window_mins: int = WINDOW_MINS) -> Tuple[pd.DataFrame, Dict]:
    """Return (trades with the 5 columns replaced, stats dict)."""
    hi = events[events["high"] & events["timed"] & ~events["speech"]]
    cov_lo, cov_hi = events["utc"].min(), events["utc"].max()

    all_ccys = set()
    for p in trades["pair"].dropna().unique():
        all_ccys |= set(currencies_for_pair(p))
    unmapped = sorted(p for p in trades["pair"].dropna().unique()
                      if not currencies_for_pair(p))
    if unmapped:
        raise ValueError(f"pairs with no currency mapping: {unmapped}")
    holes = coverage_holes(events, all_ccys)

    by_ccy = {c: g.sort_values("utc") for c, g in hi.groupby("currency")}
    w = pd.Timedelta(minutes=window_mins)
    h1 = pd.Timedelta(hours=1)

    fill_lbl = pd.to_datetime(trades["fill_ts"], utc=True, format="mixed",
                              errors="coerce")
    exit_lbl = pd.to_datetime(trades["exit_ts"], utc=True, format="mixed",
                              errors="coerce")

    out = {c: [] for c in NEW_COLS}
    stats = {"flagged_fill": 0, "flagged_open": 0, "null_fill": 0,
             "null_open": 0, "hole_hits": 0, "clock_ambiguous": 0}

    for i in trades.index:
        # Candles are ALREADY true UTC (import_mt5 applies the era correction at
        # source, 2026-07-16 Part C). So DO NOT correct again — read fill_ts /
        # exit_ts as true UTC. The ONLY residual is the 2014 regime-flip window:
        # those bars kept their provisional label, so they still fall in the flip
        # date range and is_flip_window re-derives the ambiguity -> None.
        # (Guard for the OLD canonical: on a pre-Part-B CSV the labels are still
        #  provisional and this UNDER-corrects. Do NOT re-enrich the old CSV —
        #  the next fresh run is the first true-UTC pairing. See spec Part C.)
        f = fill_lbl.loc[i]
        x = exit_lbl.loc[i]
        if pd.notna(f) and is_flip_window(f):
            stats["clock_ambiguous"] += 1
            f = pd.NaT
        if pd.notna(x) and is_flip_window(x):
            x = pd.NaT
        legs = currencies_for_pair(trades.at[i, "pair"])
        if pd.isna(f):
            out["news_fill"].append(None); out["news_fill_event"].append(None)
            out["news_fill_ccy"].append(None)
            out["news_open"].append(None); out["news_open_event"].append(None)
            stats["null_fill"] += 1; stats["null_open"] += 1
            continue

        span_end = (x + h1) if pd.notna(x) else (f + h1)
        in_hole = any((c, m) in holes for c in legs
                      for m in _months_touched(f - w, span_end + w))
        covered = (f - w) >= cov_lo and (span_end + w) <= cov_hi and not in_hole
        if in_hole:
            stats["hole_hits"] += 1
        if not covered:
            out["news_fill"].append(None); out["news_fill_event"].append(None)
            out["news_fill_ccy"].append(None)
            out["news_open"].append(None); out["news_open_event"].append(None)
            stats["null_fill"] += 1; stats["null_open"] += 1
            continue

        lo, hi_edge = f - w, f + h1 + w          # inclusive fill window
        best: Optional[Tuple[pd.Timedelta, str, str]] = None
        open_hit: Optional[Tuple[pd.Timestamp, str]] = None
        for c in legs:
            g = by_ccy.get(c)
            if g is None:
                continue
            m = g[(g["utc"] >= lo) & (g["utc"] <= hi_edge)]
            for _, r in m.iterrows():
                d = abs(r["utc"] - (f + pd.Timedelta(minutes=30)))
                if best is None or d < best[0]:
                    best = (d, r["name"], c)
            if pd.notna(x):
                mo = g[(g["utc"] >= f) & (g["utc"] <= x + h1)]
                if len(mo) and (open_hit is None or
                                mo.iloc[0]["utc"] < open_hit[0]):
                    open_hit = (mo.iloc[0]["utc"], mo.iloc[0]["name"])

        out["news_fill"].append(1 if best else 0)
        out["news_fill_event"].append(best[1] if best else None)
        out["news_fill_ccy"].append(best[2] if best else None)
        if best:
            stats["flagged_fill"] += 1
        if pd.isna(x):
            out["news_open"].append(None); out["news_open_event"].append(None)
            stats["null_open"] += 1
        else:
            out["news_open"].append(1 if open_hit else 0)
            out["news_open_event"].append(open_hit[1] if open_hit else None)
            if open_hit:
                stats["flagged_open"] += 1

    res = trades.copy()
    for c in NEW_COLS:  # replace, never duplicate -- idempotent re-runs
        res[c] = out[c]
    stats["untimed_high_dropped"] = int(
        (events["high"] & ~events["timed"]).sum())
    stats["speech_high_dropped"] = int(
        (events["high"] & events["timed"] & events["speech"]).sum())
    stats["coverage_holes"] = sorted(holes)
    return res, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--window-mins", type=int, default=WINDOW_MINS)
    args = ap.parse_args()

    trades = pd.read_csv(args.trades, low_memory=False)
    events = load_events(args.events)
    res, stats = enrich(trades, events, args.window_mins)

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(args.trades)),
                               suffix=".csv")
    os.close(fd)
    res.to_csv(tmp, index=False)
    os.replace(tmp, args.trades)

    print(f"rows: {len(res)}")
    for k in ("flagged_fill", "flagged_open", "null_fill", "null_open",
              "hole_hits", "clock_ambiguous", "untimed_high_dropped",
              "speech_high_dropped"):
        print(f"{k}: {stats[k]}")
    print(f"coverage_holes: {len(stats['coverage_holes'])}")
    for h in stats["coverage_holes"][:20]:
        print("  HOLE:", h)


if __name__ == "__main__":
    main()
