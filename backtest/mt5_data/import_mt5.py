"""Wire MT5 H1 CSVs as the SOLE backtest data source.

CLOCK: the flat -3h is only PROVISIONAL — it is TRUE UTC for the most recent
broker era only. Correcting audit (2026-07-16, MT5_CANDLE_CLOCK_AUDIT.md):
spike-aligning 13,839 exact-UTC ForexFactory events against the cached H1
candles proves the broker's SERVER CLOCK CHANGED TWICE over 18 years, so the
label - true_utc offset is era- AND season-dependent:
    era A  ..2014-10-31        0 in EU-DST summer, -1 in winter  (EET+DST)
    flip   2014-11-01..12-07    regime flip -> unfixable, label left provisional
    era B  2014-12-08..2024-10-26  +1 in EU-DST summer, 0 in winter (UTC+3+DST)
    era C  2024-10-27..         0 year-round (fixed UTC+3 -> the -3h below is exact)
  So for H1 we first apply the flat -3h (PROVISIONAL, = era C), then map through
  the empirical era table in backtest/mt5_clock.py to TRUE UTC (load_one below).
  The earlier "-3h, NO DST, timeframe-independent" pin (2026-06-23/24/26) was
  TRUE FOR ERA C ONLY: the week-open proof sampled recent behaviour and could
  not see the two historical regime changes. Prices/OHLC are untouched — only
  the hour LABEL was wrong, by +/-1h, ~5 months/year in eras A/B.

  D1/W1 are session-aggregated (no hour-of-day meaning) so they KEEP the flat
  -3h; only H1 gets the era correction.

What this does:
  - reads backtest/mt5_data/<SYM>_H1.csv (cols: time_server, open..close, tick_volume)
  - H1: server -> PROVISIONAL UTC (-3h) -> TRUE UTC (mt5_clock era table);
    D1/W1: server -> UTC (-3h). Drops weekend/dup rows, renames to OHLCV
  - backs up the existing yfinance parquet to <name>.yf.parquet (once)
  - writes UTC-indexed parquet to backtest/cache/<cache_name>.parquet + meta.json
  - cache spans 2008+, so load_bars() slices it and NEVER re-fetches yfinance.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Shared era table (Part A, 2026-07-16). is_flip_window flags the 2014 regime
# flip; mt5_label_error_hours maps the PROVISIONAL -3h label to true UTC.
sys.path.insert(0, str(Path(__file__).parent.parent))
from mt5_clock import mt5_label_error_hours  # noqa: E402

SERVER_TO_UTC_HOURS = 3  # MT5 server is UTC+3; subtract to PROVISIONAL UTC (era C).

HERE = Path(__file__).parent
CACHE = HERE.parent / "cache"

# MT5 csv stem -> cache symbol-key the engine asks load_bars for. The cache stem
# is built as <key>_<interval> (data_loader._cache_path: symbol with '=' -> '_').
# Each MT5 stem now caches THREE timeframes (H1/D1/W1) -> three parquet files
# (<key>_1h / <key>_1d / <key>_1wk) so the higher-timeframe narrative layer reads
# D1/W1 from the broker's true closes with NO re-import.
#
# Old 6 pairs keep their historical yfinance-style keys (so existing caches/tooling
# are untouched). The 5 NEW evaluation pairs use plain-symbol keys matching their
# config.json `symbol` (e.g. "GBPUSD=X" -> "GBPUSD_X").
MAPPING = {
    "EURUSD": "EURUSD_X",
    "USDJPY": "JPY_X",
    "NZDUSD": "NZDUSD_X",
    "USDCHF": "CHF_X",
    "NAS100": "NQ_F",
    "XAUUSD": "GC_F",
    # --- new evaluation pairs (backtest-only) ---
    "GBPUSD": "GBPUSD_X",
    "AUDUSD": "AUDUSD_X",
    "USDCAD": "USDCAD_X",
    "EURJPY": "EURJPY_X",
    "BTCUSD": "BTCUSD_X",
}

# meta "symbol" field (keep stable for tooling). Keyed by cache base-key above.
CACHE_SYMBOL = {
    "EURUSD_X": "EURUSD=X",
    "JPY_X": "JPY=X",
    "NZDUSD_X": "NZDUSD=X",
    "CHF_X": "CHF=X",
    "NQ_F": "NQ=F",
    "GC_F": "GC=F",
    "GBPUSD_X": "GBPUSD=X",
    "AUDUSD_X": "AUDUSD=X",
    "USDCAD_X": "USDCAD=X",
    "EURJPY_X": "EURJPY=X",
    "BTCUSD_X": "BTCUSD=X",
}

# Timeframes to import per symbol. csv suffix -> (cache interval suffix, meta label).
# The csv suffix matches mt5_pull.py's output (<SYM>_H1/_D1/_W1.csv); the cache
# interval suffix matches data_loader's interval strings (1h/1d/1wk).
TIMEFRAMES = [
    ("H1", "1h"),
    ("D1", "1d"),
    ("W1", "1wk"),
]


def _provisional_to_true_utc(provisional: pd.DatetimeIndex):
    """Map the PROVISIONAL -3h UTC labels to TRUE UTC via the era table.

    The parquet labels we produce today ARE the provisional -3h values, and the
    era table (mt5_clock) was calibrated against exactly those labels, so it
    composes: true = provisional - mt5_label_error_hours(provisional). Applied
    per row because the correction is era- AND season-dependent.

    Returns the TRUE-UTC index. Flip-window rows (2014-11-01..12-07) are
    UNFIXABLE to the hour: their shift is 0 so the label stays at the provisional
    value. No flag column is added — because those rows keep the provisional
    label, they still fall in the flip date range, so a consumer re-derives the
    ambiguity with is_flip_window(true_label) and needs no schema change.
    """
    err = [mt5_label_error_hours(ts) for ts in provisional]
    # None (flip window) -> 0h shift (keep provisional label; re-derivable later).
    shift = pd.Series(err).fillna(0).astype("int64").to_numpy()
    return pd.DatetimeIndex(provisional - pd.to_timedelta(shift, unit="h"))


def load_one(csv_path: Path, interval: str = "1h", drop_saturday: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["time_server"])
    # server (naive, UTC+3) -> PROVISIONAL UTC (-3h; correct for era C only)
    provisional = pd.DatetimeIndex(
        df["time_server"] - pd.Timedelta(hours=SERVER_TO_UTC_HOURS)
    ).tz_localize("UTC")
    df = df.drop(columns=["time_server"])
    # PROVISIONAL -> TRUE UTC via the empirical era table (mt5_clock). Fixes the
    # +/-1h seasonal label error in eras A/B; era C is a no-op. Flip-window rows
    # keep the provisional label (unfixable to the hour) and stay re-derivable
    # via is_flip_window(true_label) — no schema/column change.
    # H1 ONLY: the bug is an hour-of-day LABEL error. D1/W1 bars are session-
    # aggregated (no hour-of-day meaning); shifting their labels ±1h would move a
    # daily/weekly open off its true session boundary. So D1/W1 keep the flat -3h.
    if interval == "1h":
        df.index = _provisional_to_true_utc(provisional)
    else:
        df.index = provisional
    df.index.name = "Datetime"
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    for c in ("Open", "High", "Low", "Close"):
        df[c] = df[c].astype("float64")
    df["Volume"] = df["Volume"].astype("int64")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # Drop SATURDAY only (the Sat 00:00 server bar -> Fri 21:00 UTC is the week
    # close tail; in UTC it lands on Friday, so this mainly removes stray Sat
    # stamps). KEEP Sunday: the FX week opens Sun ~22:00 UTC, matching the
    # yfinance baseline which carried Sun 23:00 open bars. Never drop Sunday.
    # ONLY applies to H1 FX/index/commodity: D1/W1 bars are session-aggregated
    # (a daily/weekly bar is not "a Saturday"), and crypto (drop_saturday=False)
    # trades 24/7 so its weekend bars are real. Filtering those would corrupt the
    # series. The BTC weekend *trading* block is enforced in the simulator, not here.
    if drop_saturday and interval == "1h":
        df = df[df.index.dayofweek != 5]
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def main():
    for mt5_stem, base_key in MAPPING.items():
        # crypto trades 24/7 -> never drop weekend bars (real liquidity).
        is_crypto = mt5_stem == "BTCUSD"
        for tf_suffix, cache_interval in TIMEFRAMES:
            csv_path = HERE / f"{mt5_stem}_{tf_suffix}.csv"
            if not csv_path.exists():
                # H1 is mandatory; D1/W1 may legitimately be missing if a pull
                # only fetched H1. Warn and skip rather than abort the whole run.
                if tf_suffix == "H1":
                    raise SystemExit(f"missing {csv_path}")
                print(f"  [skip] {mt5_stem} {tf_suffix}: no csv ({csv_path.name})")
                continue
            df = load_one(csv_path, interval=cache_interval,
                          drop_saturday=not is_crypto)

            cache_stem = f"{base_key}_{cache_interval}"
            parquet = CACHE / f"{cache_stem}.parquet"
            meta = CACHE / f"{cache_stem}.meta.json"
            backup = CACHE / f"{cache_stem}.yf.parquet"

            # back up an existing (yfinance) parquet exactly once (never clobber).
            if parquet.exists() and not backup.exists():
                backup.write_bytes(parquet.read_bytes())
                print(f"  backed up {parquet.name} -> {backup.name}")

            df.to_parquet(parquet)
            with open(meta, "w") as f:
                json.dump({
                    "symbol": CACHE_SYMBOL[base_key],
                    "interval": cache_interval,
                    "rows": len(df),
                    "start": df.index.min().isoformat(),
                    "end": df.index.max().isoformat(),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "source": (
                        "MT5 FundingPips2-SIM; H1 server -> provisional UTC (-3h) "
                        "-> true UTC via mt5_clock era table (2 broker-clock eras, "
                        "see MT5_CANDLE_CLOCK_AUDIT.md); D1/W1 flat -3h"
                        if cache_interval == "1h" else
                        "MT5 FundingPips2-SIM, server UTC+3 -> UTC (-3h); "
                        "D1/W1 keep flat -3h (era correction is H1-only)"
                    ),
                }, f, indent=2)
            print(f"{mt5_stem:7s} {tf_suffix:2s} -> {cache_stem:14s}  rows={len(df):7d}  "
                  f"{df.index.min().date()}..{df.index.max().date()}")


if __name__ == "__main__":
    main()
