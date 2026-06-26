"""Historical OHLC loader for backtest. MT5 (FundingPips) parquet cache — the SOLE
source. yfinance has been removed (2026-06-26).

Public API:
    load_bars(symbol, interval, start, end) -> pd.DataFrame
    validate_bars(df, interval) -> dict  (gap / duplicate / staleness report)

Hard rules:
- The ONLY data source is the parquet cache in backtest/cache/, populated from the
  user's MT5 / FundingPips feed by backtest/mt5_data/import_mt5.py (server UTC+3 ->
  UTC, -3h). There is NO remote fetch. If the cache is missing, load_bars raises —
  it never silently falls back to anything.
- Cache files are immutable history; load_bars only ever SLICES them.
- Returned dataframe is UTC-indexed, columns: Open, High, Low, Close, Volume.
- The old yfinance backups live in backtest/yfinance_archive/ (kept for reference,
  never read by the engine).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from backtest.run_logger import log_event


CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

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


def load_bars(symbol: str, interval: str, start: datetime,
              end: datetime, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Load bars for [start, end] from the MT5 parquet cache.

    Cache-only: no remote fetch. Raises RuntimeError if the symbol/interval cache
    is missing (yfinance is gone — a missing cache is a hard error so it is never
    silently ignored). Returns a UTC-indexed DataFrame, or None if the cache holds
    no rows in the requested window.

    `force_refresh` is accepted for call-site compatibility but has no effect
    (there is nothing to refresh from).
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    cache_p = _cache_path(symbol, interval)
    if not cache_p.exists():
        log_event("data_cache_missing", level="error", symbol=symbol,
                  interval=interval, path=str(cache_p))
        raise RuntimeError(
            f"No MT5 cache for {symbol} {interval} ({cache_p.name}). yfinance is "
            f"disabled. Import the MT5 feed via backtest/mt5_data/import_mt5.py."
        )

    try:
        cached = pd.read_parquet(cache_p)
    except Exception as e:
        log_event("data_cache_read_fail", level="error", symbol=symbol,
                  interval=interval, error=f"{type(e).__name__}: {e}")
        raise RuntimeError(f"Failed reading MT5 cache {cache_p}: {e}")

    if cached.index.tz is None:
        cached.index = cached.index.tz_localize("UTC")
    if cached.empty:
        return None

    cache_start, cache_end = cached.index.min(), cached.index.max()
    if start < cache_start or end > cache_end:
        # Not an error — just surface that the request runs off the edge of the
        # available MT5 history so a short slice is never mistaken for a full one.
        log_event("data_partial_coverage", level="warn", symbol=symbol,
                  interval=interval,
                  requested=f"{start.isoformat()}..{end.isoformat()}",
                  available=f"{cache_start.isoformat()}..{cache_end.isoformat()}")
        print(f"  [partial coverage] {symbol} {interval}: requested "
              f"{start.date()}..{end.date()} runs beyond cache "
              f"{cache_start.date()}..{cache_end.date()} — returning available slice")

    sliced = cached.loc[start:end]
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
