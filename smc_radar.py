import yfinance as yf
import pandas as pd
import numpy as np
import json
import smtplib
import logging
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="smc_radar.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMAIL CONFIGURATION (Fetching from GitHub Secrets)
# ─────────────────────────────────────────────────────────────────────────────
EMAIL_CONFIG = {
    "sender":      ["avinash.somjani23@gmail.com"],
    "recipient":   ["avinash.somjani23@gmail.com", "fernandesbrezhnev@gmail.com"],
    "smtp_server": "smtp.gmail.com",
    "smtp_port":   587,
    "password":    os.environ.get("GMAIL_APP_PASSWORD")
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. CORE CONFIGURATION
#    — Forex pairs : H1 only
#    — Gold & NAS100 : H1 + H4
#    — NZDUSD lookback raised to 5 (thinner liquidity, noisier swings)
#    — XAUUSD H4 lookback set to 5
#    — NAS100 H4 lookback set to 6 (matches H1 volatility profile)
# ─────────────────────────────────────────────────────────────────────────────
PAIRS_CONFIG = {
    "EURUSD=X": {
        "name": "EURUSD",
        "type": "forex",
        "timeframes": {
            "H1": {"lookback": 4, "period": "15d", "interval": "1h"}
        }
    },
    "JPY=X": {
        "name": "USDJPY",
        "type": "forex",
        "timeframes": {
            "H1": {"lookback": 4, "period": "15d", "interval": "1h"}
        }
    },
    "NZDUSD=X": {
        "name": "NZDUSD",
        "type": "forex",
        "timeframes": {
            "H1": {"lookback": 5, "period": "15d", "interval": "1h"}
        }
    },
    "CHF=X": {
        "name": "USDCHF",
        "type": "forex",
        "timeframes": {
            "H1": {"lookback": 4, "period": "15d", "interval": "1h"}
        }
    },
    "GC=F": {
        "name": "XAUUSD",
        "type": "commodity",
        "timeframes": {
            "H1": {"lookback": 5, "period": "15d", "interval": "1h"},
            "H4": {"lookback": 5, "period": "60d", "interval": "4h"}
        }
    },
    "NQ=F": {
        "name": "NAS100",
        "type": "index",
        "timeframes": {
            "H1": {"lookback": 6, "period": "15d", "interval": "1h"},
            "H4": {"lookback": 6, "period": "60d", "interval": "4h"}
        }
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. ALGORITHMIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data(ticker, interval, period):
    """Generic fetcher — works for any interval and period."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df.tail(150).copy().reset_index()


def is_valid_ob_candle(open_p, close_p, high_p, low_p):
    """
    Doji Filter — body must exceed 10% of total candle range.
    Prevents wicks and indecision candles from being marked as OBs.
    """
    body = abs(open_p - close_p)
    rng  = high_p - low_p
    if rng == 0:
        return False
    return body > (rng * 0.15)


def detect_smc_radar(df, config):
    lookback = config["lookback"]
    n = len(df)

    O = df['Open'].values
    C = df['Close'].values
    H = df['High'].values
    L = df['Low'].values

    swings = []

    # ── RULE A: Swing Mapping ─────────────────────────────────────────────────
    # FIX: elif prevents the same candle being tagged as both a swing high
    # and a swing low simultaneously, which is structurally nonsensical.
    for i in range(lookback, n - lookback):
        window_highs = H[i - lookback: i + lookback + 1]
        window_lows  = L[i - lookback: i + lookback + 1]

        if H[i] == max(window_highs):
            swings.append({'type': 'high', 'idx': i, 'price': float(H[i])})
        elif L[i] == min(window_lows):
            swings.append({'type': 'low',  'idx': i, 'price': float(L[i])})

    swings = sorted(swings, key=lambda x: x['idx'])

    active_obs  = []
    bos_events  = []
    trend_state = None   # Tracks current structural trend: 'bullish' or 'bearish'

    for i in range(lookback + 1, n):

        past_swings = [s for s in swings if s['idx'] < i]
        if len(past_swings) < 2:
            continue

        latest_high = [s for s in past_swings if s['type'] == 'high']
        latest_low  = [s for s in past_swings if s['type'] == 'low']

        if not latest_high or not latest_low:
            continue

        sh = latest_high[-1]
        sl = latest_low[-1]

        bos_detected = False
        bos_type     = None

        # ── RULE C: Break of Structure ────────────────────────────────────────
        if C[i] > sh['price'] and C[i - 1] <= sh['price']:
            bos_detected = True
            bos_type     = 'bullish'
        elif C[i] < sl['price'] and C[i - 1] >= sl['price']:
            bos_detected = True
            bos_type     = 'bearish'

        if bos_detected:

            # ── CHoCH vs Continuation BOS ─────────────────────────────────────
            # FIX: Every BOS is now tagged. A BOS against the prevailing trend
            # is a CHoCH (trend shift signal). A BOS with the trend is a
            # continuation. OBs from a CHoCH carry higher structural significance.
            if trend_state is None or trend_state != bos_type:
                bos_tag = 'CHoCH+BOS'
            else:
                bos_tag = 'continuation_BOS'

            trend_state = bos_type
            bos_events.append({
                'type': bos_type,
                'tag':  bos_tag,
                'idx':  i,
                'price': C[i]
            })

            # ── RULE D: OB Identification — Walk Backward Along Impulse Leg ──
            ob_idx = -1
            impulse_start_idx = sl['idx'] if bos_type == 'bullish' else sh['idx']

            for j in range(i - 1, impulse_start_idx - 1, -1):
                is_bearish = C[j] < O[j]
                is_bullish = C[j] > O[j]

                if (bos_type == 'bullish' and is_bearish) or \
                   (bos_type == 'bearish' and is_bullish):

                    if is_valid_ob_candle(O[j], C[j], H[j], L[j]):
                        ob_idx = j
                        break
                    # FIX: Removed blind fallback to j-1. If this candle fails
                    # the doji filter, the loop continues searching for the next
                    # valid opposing candle rather than picking an unvalidated one.

            if ob_idx == -1:
                continue

            # ── RULE E: FVG Validation (5-Candle Window) ─────────────────────
            fvg_valid  = False
            window_end = min(ob_idx + 6, n)

            for k in range(ob_idx, window_end - 2):
                if bos_type == 'bullish':
                    if H[k] < L[k + 2]:       # Bullish imbalance
                        fvg_valid = True
                        break
                elif bos_type == 'bearish':
                    if L[k] > H[k + 2]:       # Bearish imbalance
                        fvg_valid = True
                        break

            if not fvg_valid:
                continue

            ob_high = float(H[ob_idx])
            ob_low  = float(L[ob_idx])

            active_obs.append({
                'bos_idx':      i,
                'ob_idx':       ob_idx,
                'direction':    bos_type,
                'bos_tag':      bos_tag,
                'high':         ob_high,
                'low':          ob_low,
                'mean':         float((ob_high + ob_low) / 2),
                'proximal_line': ob_high if bos_type == 'bullish' else ob_low,
                'distal_line':   ob_low  if bos_type == 'bullish' else ob_high
            })

    # ── RULE F: Mitigation Check ──────────────────────────────────────────────
    # FIX: Mitigation is now defined as a candle CLOSING beyond the DISTAL line
    # (the far edge of the OB). The previous logic used a wick touch of the
    # proximal line, which is literally the entry point — it was invalidating
    # zones the moment price entered them, which is incorrect SMC behaviour.
    pristine_obs  = []
    current_price = float(C[-1])

    for ob in active_obs:
        mitigated = False

        for m in range(ob['ob_idx'] + 2, n):
            if ob['direction'] == 'bullish':
                if C[m] < ob['distal_line']:   # Close below OB bottom
                    mitigated = True
                    break
            else:
                if C[m] > ob['distal_line']:   # Close above OB top
                    mitigated = True
                    break

        if not mitigated:
            dist = abs(current_price - ob['proximal_line'])
            pristine_obs.append({
                "direction":     "Bullish (Demand)" if ob['direction'] == 'bullish' else "Bearish (Supply)",
                "bos_tag":       ob['bos_tag'],
                "proximal_line": round(ob['proximal_line'], 5),
                "distal_line":   round(ob['distal_line'],   5),
                "dist_to_price": round(dist, 5),
                "ob_high":       round(ob['high'],  5),
                "ob_low":        round(ob['low'],   5),
                "ob_mean":       round(ob['mean'],  5),
                "ob_time_utc":   str(df['Datetime'].iloc[ob['ob_idx']])
                                 if 'Datetime' in df.columns
                                 else str(df.index[ob['ob_idx']])
            })

    # ── RULE B: External Range Mapping ────────────────────────────────────────
    ext_range_high = None
    ext_range_low  = None

    if len(bos_events) >= 2:
        last_bos = bos_events[-1]['idx']
        prev_bos = bos_events[-2]['idx']
        ext_range_high = float(max(H[prev_bos: last_bos + 1]))
        ext_range_low  = float(min(L[prev_bos: last_bos + 1]))

    return {
        "current_price": round(current_price, 5),
        "external_range": {
            "high": round(ext_range_high, 5) if ext_range_high else None,
            "low":  round(ext_range_low,  5) if ext_range_low  else None
        },
        "active_unmitigated_obs": pristine_obs
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. EMAIL FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def send_email(payload):
    """Sends the radar payload as a plain-text email."""
    try:
        subject = f"SMC Radar Report — {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        body    = json.dumps(payload, indent=4)

        msg              = MIMEMultipart()
        msg['From']      = EMAIL_CONFIG['sender']
        msg['To']        = ", ".join(EMAIL_CONFIG['recipient'])
        msg['Subject']   = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
            server.sendmail(EMAIL_CONFIG['sender'], EMAIL_CONFIG['recipient'], msg.as_string())

        logging.info("Email dispatched successfully.")

    except Exception as e:
        logging.error(f"Email dispatch failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXECUTION ENGINE
#    — Runs every hour (scheduled externally via cron or task scheduler)
#    — Logs every run
#    — Sends email every 2 hours (when UTC hour is even)
# ─────────────────────────────────────────────────────────────────────────────

def run_radar():
    radar_payload = {}
    run_time      = datetime.utcnow()

    for ticker, config in PAIRS_CONFIG.items():
        name                 = config["name"]
        radar_payload[name]  = {}

        for tf_label, tf_config in config["timeframes"].items():
            try:
                df = fetch_data(ticker, tf_config["interval"], tf_config["period"])
                if df is not None:
                    result = detect_smc_radar(df, tf_config)
                    radar_payload[name][tf_label] = result
                else:
                    radar_payload[name][tf_label] = {"error": "Failed to fetch data"}
            except Exception as e:
                radar_payload[name][tf_label] = {"error": str(e)}

    # Hourly log — every run is recorded
    logging.info(f"Radar scan complete at {run_time.strftime('%Y-%m-%d %H:%M')} UTC — "
                 f"{json.dumps(radar_payload)}")

    # 2-hour email cadence — fires when UTC hour is even (00, 02, 04 ... 22)
    if run_time.hour % 2 == 0:
        send_email(radar_payload)

    print(json.dumps(radar_payload, indent=4))


if __name__ == "__main__":
    run_radar()
