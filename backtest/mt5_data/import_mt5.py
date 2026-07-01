"""Wire MT5 H1 CSVs as the SOLE backtest data source.

Offset PINNED empirically (2026-06-23, terminal open, FundingPips2-SIM).
Re-verified 2026-06-24 against the full 18y history of all 6 symbols:
  MT5 server clock = UTC+3 (fixed, NO DST). Proof:
    1. Monday week-open server hour is CONSTANT 01:00 across all 12 months
       (115k bars x 5 FX symbols). A DST broker would flip 00:00/01:00 by
       season; it never does -> server runs no daylight saving.
    2. Apply -3h to every week-open bar -> 5 FX symbols land Sun 22:00 UTC
       (XAUUSD Sun 23:00, its true session open is 1h later than spot FX).
       867/934 EURUSD weeks hit Sun 22:00 UTC exactly; 100% land on Sunday.
       1h-wrong offset would land them on 21:00 or 23:00 as the mode. It does
       not -> -3h is correct.
    3. NAS100 after -3h shows the daily index break thinning at 21:00 UTC
       (5pm ET), matching US index session structure.
  All 6 symbols share ONE FundingPips account = ONE server clock, so the
  -3h proven on FX applies identically to Gold and NAS.
  => convert server -> UTC by SUBTRACTING 3 hours.

  Re-verified 2026-06-26 for the new evaluation pairs (GBPUSD, AUDUSD, USDCAD,
  EURJPY, BTCUSD): the Monday week-open SERVER hour is the SAME constant 01:00
  across all 12 months on every new FX pair (identical DST-free signature to
  EURUSD) -> same server clock -> the -3h is correct for them too. BTC's calendar
  week-open sits at server 00:00 because it trades 24/7 (no session open), but it
  is the SAME account/clock, so -3h still yields correct UTC. Same offset applies
  to D1/W1 (a clock conversion is timeframe-independent).

What this does:
  - reads backtest/mt5_data/<SYM>_H1.csv (cols: time_server, open..close, tick_volume)
  - server -> UTC (-3h), drops weekend/dup rows, renames to Open/High/Low/Close/Volume
  - backs up the existing yfinance parquet to <name>.yf.parquet (once)
  - writes UTC-indexed parquet to backtest/cache/<cache_name>.parquet + meta.json
  - cache spans 2008+, so load_bars() slices it and NEVER re-fetches yfinance.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SERVER_TO_UTC_HOURS = 3  # MT5 server is UTC+3; subtract to get UTC.

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


def load_one(csv_path: Path, interval: str = "1h", drop_saturday: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["time_server"])
    # server (naive, UTC+3) -> UTC
    idx = df["time_server"] - pd.Timedelta(hours=SERVER_TO_UTC_HOURS)
    df = df.drop(columns=["time_server"])
    df.index = pd.DatetimeIndex(idx).tz_localize("UTC")
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
                    "source": "MT5 FundingPips2-SIM, server UTC+3 -> UTC (-3h)",
                }, f, indent=2)
            print(f"{mt5_stem:7s} {tf_suffix:2s} -> {cache_stem:14s}  rows={len(df):7d}  "
                  f"{df.index.min().date()}..{df.index.max().date()}")


if __name__ == "__main__":
    main()
