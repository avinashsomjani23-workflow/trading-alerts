"""Historical OHLC loader for backtest. yfinance source, parquet cache.

Public API:
    load_bars(symbol, interval, start, end) -> pd.DataFrame
    validate_bars(df, interval) -> dict  (gap / duplicate / staleness report)

Hard rules:
- Cache hits never expire by date (history is immutable). They expire only
  if requested range falls outside the cached range.
- Returned dataframe is UTC-indexed, columns: Open, High, Low, Close, Volume.
- No live-system side effects. Cache lives in backtest/cache/.
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import yfinance as yf


CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# yfinance interval limits (from yfinance docs, conservative).
INTERVAL_MAX_DAYS = {
    "5m":  58,    # ~60d max, we use 58 to be safe
    "15m": 720,   # ~730d, same as 1h
    "1h":  720,   # 730d advertised
}

INTERVAL_TIMEDELTA = {
    "5m":  timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h":  timedelta(hours=1),
}


def _cache_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace("=", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}_{interval}.parquet"


def _meta_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace("=", "_").replace("/", "_")
    return CACHE_DIR / f"{safe}_{interval}.meta.json"


def _normalize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    cols = ["Open", "High", "Low", "Close"]
    for c in cols:
        if c not in df.columns:
            return None
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=cols)


def _fetch_chunk(symbol: str, interval: str, start: datetime, end: datetime,
                 retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch one chunk from yfinance with retries."""
    last_err = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                symbol,
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                timeout=30,
                auto_adjust=False,
            )
            df = _normalize_df(raw)
            if df is not None and not df.empty:
                return df
            last_err = "empty response"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    print(f"  [fetch fail] {symbol} {interval} {start.date()}..{end.date()}: {last_err}")
    return None


def _chunked_fetch(symbol: str, interval: str, start: datetime,
                   end: datetime) -> Optional[pd.DataFrame]:
    """Walk start->end in interval-sized chunks (yfinance caps 5m/15m at ~60d)."""
    max_days = INTERVAL_MAX_DAYS.get(interval, 60)
    chunks = []
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=max_days - 1), end)
        df = _fetch_chunk(symbol, interval, cur, chunk_end)
        if df is not None:
            chunks.append(df)
        cur = chunk_end + timedelta(days=1)
    if not chunks:
        return None
    merged = pd.concat(chunks).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def load_bars(symbol: str, interval: str, start: datetime,
              end: datetime, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Load bars for [start, end]. Uses parquet cache if range is covered.

    Returns UTC-indexed DataFrame or None on persistent failure.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    cache_p = _cache_path(symbol, interval)
    meta_p = _meta_path(symbol, interval)

    cached = None
    if cache_p.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(cache_p)
            if cached.index.tz is None:
                cached.index = cached.index.tz_localize("UTC")
        except Exception as e:
            print(f"  [cache read fail] {cache_p}: {e}")
            cached = None

    needed_start = start
    needed_end = end

    if cached is not None and not cached.empty:
        cache_start = cached.index.min()
        cache_end = cached.index.max()
        # If cache covers the whole requested range, just slice.
        if cache_start <= start and cache_end >= end:
            sliced = cached.loc[start:end]
            return sliced if not sliced.empty else None
        # Otherwise refetch the full requested range (simpler than gap-merging
        # for v1; we can optimise later if cache hit rate matters).
        needed_start = min(start, cache_start) if cache_start else start
        needed_end = max(end, cache_end) if cache_end else end

    # yfinance hard limit on intraday history (~730d for 1h, ~60d for 15m/5m).
    # Don't request earlier than that.
    max_lookback_days = INTERVAL_MAX_DAYS.get(interval, 60)
    earliest_allowed = datetime.now(timezone.utc) - timedelta(days=max_lookback_days)
    if needed_start < earliest_allowed:
        needed_start = earliest_allowed
        print(f"  [clamp] {symbol} {interval} start clamped to "
              f"{needed_start.date()} (yfinance {max_lookback_days}d limit)")

    if needed_start >= needed_end:
        print(f"  [skip] {symbol} {interval}: requested range entirely outside yfinance window")
        return None

    df = _chunked_fetch(symbol, interval, needed_start, needed_end)
    if df is None:
        return cached.loc[start:end] if cached is not None else None

    # Merge with cache if exists.
    if cached is not None and not cached.empty:
        df = pd.concat([cached, df]).sort_index()
        df = df[~df.index.duplicated(keep="last")]

    # Persist.
    try:
        df.to_parquet(cache_p)
        with open(meta_p, "w") as f:
            json.dump({
                "symbol": symbol,
                "interval": interval,
                "rows": len(df),
                "start": df.index.min().isoformat(),
                "end": df.index.max().isoformat(),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
    except Exception as e:
        print(f"  [cache write fail] {cache_p}: {e}")

    sliced = df.loc[start:end]
    return sliced if not sliced.empty else None


def validate_bars(df: pd.DataFrame, interval: str) -> Dict[str, Any]:
    """Report gaps, duplicates, weekend bars."""
    if df is None or df.empty:
        return {"valid": False, "reason": "empty"}
    step = INTERVAL_TIMEDELTA.get(interval)
    if step is None:
        return {"valid": True, "reason": "no step rule"}
    deltas = df.index.to_series().diff().dropna()
    # Gaps > 3x expected step (allows weekend gap for forex).
    big_gaps = deltas[deltas > step * 3]
    weekend_bars = df[df.index.dayofweek >= 5]  # Sat / Sun
    return {
        "valid": True,
        "rows": len(df),
        "start": df.index.min().isoformat(),
        "end": df.index.max().isoformat(),
        "gap_count": int(len(big_gaps)),
        "largest_gap_hours": float(big_gaps.max().total_seconds() / 3600) if len(big_gaps) else 0.0,
        "weekend_bar_count": int(len(weekend_bars)),
    }
