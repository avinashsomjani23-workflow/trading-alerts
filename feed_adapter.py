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


def fetch_h1_unstripped(config_symbol, outputsize=450, retries=2):
    """H1 candles WITHOUT the closed-market strip — for the PD/PW pool
    resampler (pool_builder), which needs the Sunday-evening UTC bars that
    fetch_h1 deliberately drops: they are the first bars of MONDAY'S MT5
    server day (server = UTC+3), so stripping them would clip Monday's true
    daily high/low. pool_builder applies its own SERVER-time strip instead.
    Engines keep using fetch_h1 — chart/positional parity is unchanged.
    """
    return _fetch(config_symbol, "1h", outputsize, retries, strip=False)


def _fetch(config_symbol, interval, outputsize, retries, strip=True):
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
                return _to_dataframe(data["values"], td_symbol=td_symbol,
                                     strip=strip)

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


def _to_dataframe(values, td_symbol="", strip=True):
    """Twelve Data 'values' list -> UTC-indexed OHLCV DataFrame, newest row last.

    td_symbol (Twelve Data symbol, e.g. 'XAU/USD') selects the closed-market rule
    in _strip_market_closed — Gold has a daily maintenance break FX does not.
    strip=False skips that strip entirely (fetch_h1_unstripped / pool levels).
    """
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

    if strip:
        df = _strip_market_closed(df, td_symbol=td_symbol)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _strip_market_closed(df, td_symbol=""):
    """Drop synthetic closed-market bars so the live H1 frame matches MT5's gapped shape.

    WHY: Twelve Data's free tier pads FX-closed hours with filler bars. The whole
    system was built on MT5 data, which OMITS closed-market hours (real gaps).
    Those filler bars break two things at once:
      1) charts render a band of dead micro-candles that MT5 never had, and —
         because both renderers plot on POSITIONAL x (bar index, not timestamp) —
         every candle after a gap is shoved sideways vs the real MT5 shape;
      2) resync_slate_zone_indices (smc_radar) shifts every leg sub-index by ONE
         uniform delta, which is only correct when bar spacing is constant —
         filler bars inserted mid-leg push FVG/BOS/OB boxes off their candles.
    Stripping here (the single network path) fixes both for Phase 1 and Phase 2.

    RULE (DST-proof, no server-offset math). Verified against MT5 directly (1 full
    year, all 5 live pairs), reading weekday/hour straight off the UTC index
    (Twelve Data returns UTC; timezone=UTC in the request):

    - WEEKEND (all instruments): the broker week is strictly Monday 00:00 UTC ->
      Friday 23:00 UTC. MT5 prints ZERO Saturday and ZERO Sunday bars — INCLUDING
      Sunday >= 21:00 UTC. (A prior version kept Sunday >= 21:00 on a generic-FX
      reopen assumption; our broker's MT5 does not print until Monday 00:00, so it
      leaked 3 phantom Sunday-evening bars every weekend.)

    - GOLD DAILY BREAK (XAU/USD only): MT5 Gold has a daily maintenance gap — the
      00:00 UTC hour is ABSENT on every weekday (missing 264/264 weekdays over the
      year; FX prints all 24 hours). Twelve Data pads that hour, so Gold needs its
      weekday 00:00 UTC bar stripped too, else Gold drifts one bar per day vs MT5.

    - HOLIDAY EARLY-CLOSE (all instruments), the general safety net so this class
      of bug can never recur without a code change: markets close early on some
      holidays (e.g. Gold on the pre-Juneteenth / pre-July-4 Friday closes ~20:00
      UTC). MT5 simply omits those hours; Twelve Data pads them with FLAT filler
      candles (high == low). We can't hardcode holiday dates (they move yearly), so
      we strip any RUN of >= 2 consecutive flat bars. Verified against MT5 (2yr,
      all 5 pairs): real flat bars never occur consecutively — the longest real
      flat run is 1 (a lone one-tick New-Year blip on NZDUSD). Requiring a run of
      >= 2 therefore drops holiday filler blocks while NEVER dropping a real bar.

    Friday (<= 23:00) and Monday (00:00+) FX bars are legit and untouched.
    """
    idx = df.index
    dow = idx.dayofweek   # Mon=0 .. Sat=5, Sun=6
    closed = (dow == 5) | (dow == 6)
    if _is_gold(td_symbol):
        # Gold's daily maintenance break: no 00:00 UTC bar on trading days.
        closed = closed | ((dow < 5) & (idx.hour == 0))
    closed = closed | _flat_filler_run_mask(df)
    return df[~closed]


def _flat_filler_run_mask(df):
    """Boolean mask (index-aligned) for synthetic FLAT filler bars that pad
    holiday early-closes. A bar is flat when high == low. Real MT5 flat bars are
    always isolated (verified 2yr/5 pairs: longest real flat run == 1), so we only
    strip a flat bar when it belongs to a run of >= 2 consecutive flat bars —
    never dropping a genuine one-tick blip.
    """
    flat = (df["High"] == df["Low"])
    if not flat.any():
        return flat  # all-False, index-aligned
    # A flat bar is filler iff its neighbour (prev OR next) is also flat.
    prev_flat = flat.shift(1, fill_value=False)
    next_flat = flat.shift(-1, fill_value=False)
    return flat & (prev_flat | next_flat)


def _is_gold(td_symbol):
    """True for the Gold instrument (Twelve Data 'XAU/USD'). Case-insensitive."""
    return "XAU" in (td_symbol or "").upper()
