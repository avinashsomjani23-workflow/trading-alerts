"""Live price feed — Twelve Data (the SOLE live source; yfinance is gone).

One adapter, one symbol map. Both live engines (smc_radar Phase 1, Phase2_Alert_Engine)
fetch H1 through fetch_h1() here so there is exactly one network path to reason about.

Shape contract: fetch_h1() returns a UTC-DatetimeIndexed DataFrame with columns
Open/High/Low/Close/Volume (newest row last), or None on persistent failure. That is the
same shape clean_df() used to hand back from yfinance, so callers are unchanged downstream.

Auth: key read from TWELVEDATA_API_KEY (GitHub secret). Never hardcoded.

Symbol map: config still carries yfinance-style symbols (EURUSD=X, GC=F, ...). We translate
to Twelve Data symbols here. NAS100 (NQ=F) has NO free Twelve Data equivalent and is not
mapped — it has been removed from the live slate (config backtest_only), so it never reaches
here. A symbol with no mapping raises, loudly, rather than silently fetching the wrong thing.
"""

import os
import time

import pandas as pd
import requests

API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
BASE_URL = "https://api.twelvedata.com/time_series"

# yfinance-style config symbol  ->  Twelve Data symbol. Forex + gold only; these are the
# live instruments verified available on the Twelve Data free tier (2026-06-29).
SYMBOL_MAP = {
    "EURUSD=X": "EUR/USD",
    "JPY=X":    "USD/JPY",
    "NZDUSD=X": "NZD/USD",
    "CHF=X":    "USD/CHF",
    "GC=F":     "XAU/USD",
    # backtest_only pairs, mapped so a future promotion to live just works:
    "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "EURJPY=X": "EUR/JPY",
    "BTCUSD=X": "BTC/USD",
}

# Twelve Data interval strings keyed by the engine's interval labels.
INTERVAL_MAP = {"1h": "1h", "15m": "15min", "5m": "5min"}

# Twelve Data free tier: 8 API credits/minute. On a 429-style credit error we wait out
# the minute once and retry rather than failing the scan.
RATE_LIMIT_WAIT_S = 62


def to_td_symbol(config_symbol):
    """Translate a config (yfinance-style) symbol to a Twelve Data symbol.

    Raises KeyError for an unmapped symbol so a misconfiguration surfaces immediately
    instead of silently skipping or fetching the wrong instrument.
    """
    try:
        return SYMBOL_MAP[config_symbol]
    except KeyError:
        raise KeyError(
            f"No Twelve Data mapping for '{config_symbol}'. Live instruments must be "
            f"Twelve-Data-available; NAS100 (NQ=F) is not and must stay backtest_only."
        )


def fetch_h1(config_symbol, outputsize=200, retries=2):
    """Fetch H1 candles for one instrument from Twelve Data.

    Returns a UTC-DatetimeIndexed OHLCV DataFrame (newest last), or None on persistent
    failure. outputsize is the number of most-recent hourly bars (max 5000 on free tier);
    200 comfortably covers the ~30-day H1 window the live engines need.

    The live caller (Phase2_Alert_Engine.fetch_with_retry) passes
    smc_detector.LIVE_P2_H1_BARS (=200) explicitly so the backtest can clamp its
    scorecard/levels input to the SAME window — see TRUTH_FIXES_SPEC_2 T5. This
    literal default is a fallback only; the parity source of truth is that constant.
    """
    return _fetch(config_symbol, "1h", outputsize, retries)


def _fetch(config_symbol, interval, outputsize, retries):
    if not API_KEY:
        print("  [FEED ERR] TWELVEDATA_API_KEY not set — cannot fetch live data.")
        return None

    td_symbol = to_td_symbol(config_symbol)
    td_interval = INTERVAL_MAP.get(interval, interval)
    params = {
        "symbol": td_symbol,
        "interval": td_interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "format": "JSON",
        "order": "ASC",
        "timezone": "UTC",
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            data = resp.json()

            message = str(data.get("message", ""))
            if "API credits" in message:  # rate limited — wait out the minute, retry
                print(f"  [RATE LIMIT] {td_symbol} {td_interval}: waiting {RATE_LIMIT_WAIT_S}s")
                time.sleep(RATE_LIMIT_WAIT_S)
                continue

            if data.get("status") == "ok" and data.get("values"):
                return _to_dataframe(data["values"])

            last_error = message or data.get("status", "unknown error")
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < retries:
            wait = 2 ** attempt
            print(f"  [RETRY {attempt + 1}/{retries}] {td_symbol} {td_interval}: "
                  f"{last_error}. Waiting {wait}s.")
            time.sleep(wait)

    print(f"  [SKIP] {td_symbol} {td_interval}: {last_error}")
    return None


def _to_dataframe(values):
    """Twelve Data 'values' list -> UTC-indexed OHLCV DataFrame, newest row last."""
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()

    rename = {"open": "Open", "high": "High", "low": "Low",
              "close": "Close", "volume": "Volume"}
    df = df.rename(columns=rename)
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col].astype(float)
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    else:
        df["Volume"] = 0.0

    return df[["Open", "High", "Low", "Close", "Volume"]]
