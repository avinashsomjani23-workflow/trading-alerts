import feed_adapter
import pandas as pd
import numpy as np
import json
import os
import smtplib
import requests
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import base64
from io import BytesIO
import xml.etree.ElementTree as ET
import smc_detector
import news_filter
import charts  # shared H1 chart style engine (Wave 2 item 2C)
import pool_builder  # PD/PW liquidity pools — informational only, never gates
import eq_pools      # EQH/EQL equal-level clusters — informational only, never gates
import liquidity_sweep  # sweep v2 (pool-anchored) — informational only, never gates
import weekly_pd     # Weekly PD zone (HTF premium/discount) — informational only
import approach_quality  # approach-into-zone shape — observe-only, never gates
import displacement_leg  # displacement-leg extreme + Kaufman ER — observe-only, never gates
import setup_liq        # setup-liquidity reads (stop-side / tp magnet / leg-extreme) — observe-only, never gates
# Direction-aware structural-event label. One implementation lives in smc_radar
# (chart uses it too); the email reuses it so every CHoCH/BOS mention shows
# bullish/bearish — the trader must never verify direction off the candles.
from smc_radar import _event_label_dir
# Single choke-point: bare state filenames auto-route to state/ (one impl,
# defined in smc_radar). See its docstring + tests/test_state_paths.py.
from smc_radar import resolve_state_path

with open("config.json") as f:
    config = json.load(f)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "dummy")
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "avinash.somjani23@gmail.com")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD", "dummy")

# Minimum raw score (out of 10) a zone must reach to email a TRADE READY alert.
# Set to 4 on 2026-06-18 — the only quality gate in Phase 2. Below this, the
# zone is logged as below_score_floor and silently skipped.
MIN_SCORE_TO_EMAIL = float(config.get("scoring", {}).get("min_score_to_email", 4))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def get_day_id_ist(now=None):
    """Trading-day identifier. Day starts at 09:00 IST. Mirrors Phase 1.
    Pre-09:00 IST belongs to the previous day's slate."""
    if now is None:
        now = get_ist_now()
    if now.hour >= 9:
        return now.strftime("%Y-%m-%d")
    return (now - timedelta(days=1)).strftime("%Y-%m-%d")


def score_to_int(score):
    """Integer floor of score. Stabilises 7.4 and 7.9 to the same bucket (7).
    Used to decide whether a re-emerging zone has materially changed."""
    try:
        return int(float(score))
    except Exception:
        return -1


def scorecard_real_max(pair_conf):
    """Real (internal) maximum for a pair's confluence scorecard. Both pair
    classes total 10 (2026-06-18 rebalance), so the max is a flat 10.

    Components (mirrors smc_detector.run_scorecard):
      Non-JPY forex: Structure 4 | Sweep 1 | FVG 2 | Freshness 2 | Killzone 1.
      JPY/Gold/NAS:  Structure 4 | Sweep 2 | FVG 2 | Freshness 1 | Killzone 1.
    The freshness/sweep split keeps both at 10 (see run_scorecard for why).
    """
    return 10


def normalized_score(raw, pair_conf):
    """Both pair classes now top out at 10, so the raw score is already on the
    /10 scale. Kept as a thin pass-through (rounded) so callers and the subject
    line don't need to change, and a future max change has one place to edit.
    """
    real_max = scorecard_real_max(pair_conf)
    if real_max <= 0:
        return 0.0
    return round(float(raw) / real_max * 10.0, 1)


# Re-entry flicker guard. A zone that leaves proximity and returns re-emails
# (plain TRADE READY — price has come back). But a zone hovering on the
# proximity boundary must NOT spam an email every hour it jitters across the
# line. We only arm the re-entry trigger once price has pulled beyond
# REENTRY_EXIT_MULT x the proximity cap since the last email — i.e. a genuine
# departure, not boundary noise.
REENTRY_EXIT_MULT = 1.5


def _ob_in_killzone_label(ob, pair_conf):
    """Label for whether the OB candle landed in a killzone window. DST-aware
    via the shared smc_detector engine (single source of truth for windows).

    Returns:
      - "in killzone (<window>)" (green) naming the window the OB candle landed
        in (e.g. "London Open") if it overlaps any window for its date
      - "outside killzone" (amber) if it doesn't
      - "unknown" if either input is missing
    """
    try:
        ob_ts_iso = (ob or {}).get("ob_timestamp")
        killzones = (pair_conf or {}).get("killzones")
        if not ob_ts_iso or not killzones:
            return "<span style='color:#888;'>unknown</span>"
        label = smc_detector.killzone_label_for_ts(ob_ts_iso, killzones)
        if label:
            return f"<b style='color:#27ae60;'>in killzone ({label})</b>"
        return "<b style='color:#e67e22;'>outside killzone</b>"
    except Exception:
        return "<span style='color:#888;'>unknown</span>"


def _entry_killzone_forecast_label(data, pair_conf):
    """Lean one-phrase forecast of whether the limit will FILL inside a killzone,
    plus the IST cut-off. Info only — never gates. ETA = distance / ATR (an
    estimate); the cut-off is exact clock math on the DST-resolved windows,
    shown in IST. Returns '' if inputs are missing (renders nothing rather than
    a misleading guess)."""
    killzones = (pair_conf or {}).get("killzones")
    if not killzones:
        return "<span style='color:#888;'>n/a</span>"
    distance = data.get("distance_to_proximal")
    atr = data.get("h1_atr")
    now_utc = get_ist_now() - timedelta(hours=5, minutes=30)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    fc = smc_detector.killzone_entry_forecast(now_utc, distance, atr, killzones)

    def _to_ist(dt_utc):
        return (dt_utc.astimezone(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%H:%M IST")

    deadline = fc.get("deadline_utc")
    deadline_str = _to_ist(deadline) if deadline else None
    deadline_label = fc.get("deadline_label")
    # Name the window the deadline belongs to, e.g. "London Open until 14:30 IST".
    window_phrase = f"{deadline_label} until {deadline_str}" if (deadline_label and deadline_str) \
        else (f"in killzone until {deadline_str}" if deadline_str else None)

    if fc.get("eta_hours") is None:
        # No usable ETA — still show the cut-off if a window is upcoming.
        if window_phrase:
            return f"<b style='color:#aaa;'>{window_phrase}</b> <i>(info)</i>"
        return "<span style='color:#888;'>n/a</span>"

    if fc.get("eta_in_kz"):
        tail = f" &middot; enter by {deadline_str}" if deadline_str else ""
        in_kz = f"likely in killzone ({deadline_label})" if deadline_label else "likely in killzone"
        return f"<b style='color:#27ae60;'>{in_kz}</b>{tail} <i>(est.)</i>"

    if window_phrase:
        return (f"<b style='color:#e67e22;'>likely outside</b> "
                f"&middot; {window_phrase} <i>(est.)</i>")
    return "<b style='color:#e67e22;'>likely outside killzone</b> <i>(est.)</i>"


def load_json(path, default):
    try:
        with open(resolve_state_path(path)) as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    """Atomic save: write to temp file then rename."""
    path = resolve_state_path(path)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    
def load_slate_as_pair_map(path="active_obs.json"):
    """
    Backward-compatible reader. Phase 1 daily-slate schema is:
      {"slate_date": "...", "slate_started_iso": "...",
       "pairs": {"EURUSD": {"next_id_counter": N, "zones": [...]}, ...}}

    Legacy flat schema:
      {"EURUSD": [...], "USDJPY": [...], ...}

    Phase 2's runtime expects flat {pair_name: [zones]}. This adapter
    detects the schema version and returns the flat map regardless.

    For new schema: only zones with status="active" are returned.
    Dropped zones are excluded — Phase 2 must never alert on a dropped zone.
    """
    raw = load_json(path, {})
    if not isinstance(raw, dict):
        return {}
    # New schema detection: presence of "pairs" key with dict value
    if "pairs" in raw and isinstance(raw.get("pairs"), dict):
        flat = {}
        for pair_name, pblock in raw["pairs"].items():
            if not isinstance(pblock, dict):
                continue
            zones = pblock.get("zones", [])
            if not isinstance(zones, list):
                continue
            # Filter to active only — dropped zones must not feed Phase 2
            flat[pair_name] = [z for z in zones if z.get("status") == "active"]
        return flat
    # Legacy schema: assume already flat
    return raw
    
def append_scan_log(entry):
    """Append one JSON line per pair per scan to phase2_scan_log.jsonl."""
    try:
        with open("phase2_scan_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"  [SCANLOG ERR] {e}")


# Permanent archive for zones that were ALERTED in Phase 2 and have since died.
# A Phase-2 dedup entry exists iff that zone was emailed to the user; when it is
# GC-evicted (Phase 1 dropped the zone => stop hit / invalidated), we keep the
# full record forever for study + the weekly review. Append-only, monthly jsonl
# (mirrors the Phase 1 scan-log convention). One line per zone per lifetime.
# Path is fixed and predictable: state/phase2_history/<YYYY-MM>.jsonl
PHASE2_HISTORY_DIR = os.path.join("state", "phase2_history")


def archive_phase2_zone(zone_id, entry, ist_now):
    """Append one dead Phase-2-alerted zone's full record to the monthly archive.

    `entry` is the phase2_sent dedup entry, which carries the full `ob_snapshot`
    captured at alert time (geometry + FVG band + sweep + dealing range). Never
    raises — archival must never block a scan or the GC it runs inside.
    """
    try:
        os.makedirs(PHASE2_HISTORY_DIR, exist_ok=True)
        path = os.path.join(PHASE2_HISTORY_DIR, f"{ist_now.strftime('%Y-%m')}.jsonl")
        record = dict(entry)
        record["zone_id"] = zone_id
        record["archived_iso"] = ist_now.isoformat()
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        print(f"  [P2 ARCHIVE ERR] {zone_id}: {e}")


SCAN_LOG_RETENTION_DAYS = 14


def rotate_scan_log(ist_now):
    """Trim phase2_scan_log.jsonl to entries within retention window.

    Called once at P2 start. Filters lines by their 'ts_ist' field; entries
    older than SCAN_LOG_RETENTION_DAYS are dropped. Unparseable lines are
    kept (we don't silently delete data we can't read).
    """
    path = "phase2_scan_log.jsonl"
    if not os.path.exists(path):
        return
    try:
        cutoff = ist_now - timedelta(days=SCAN_LOG_RETENTION_DAYS)
        kept = []
        dropped = 0
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts_str = entry.get("ts_ist")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str)
                        if ts < cutoff:
                            dropped += 1
                            continue
                except Exception:
                    pass  # unparseable — keep it
                kept.append(line)
        if dropped == 0:
            return
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for line in kept:
                f.write(line + "\n")
        os.replace(tmp, path)
        print(f"  [SCANLOG ROT] Dropped {dropped} entries older than {SCAN_LOG_RETENTION_DAYS}d.")
    except Exception as e:
        print(f"  [SCANLOG ROT ERR] {e}")

def pip_size(pair_conf):
    dp = pair_conf.get("decimal_places", 5)
    if dp == 5:
        return 0.0001
    if dp == 3:
        return 0.01
    return 1.0


def pip_unit_label(pair_conf):
    return "pips" if pair_conf.get("pair_type") == "forex" else "points"


def atr_distance_label(distance, atr, tf_label="H1"):
    if atr is None or atr == 0:
        return ""
    ratio = distance / atr
    if ratio < 0.5:
        return f"Price is at the zone now (under 0.5x {tf_label} ATR)."
    if ratio < 1.5:
        return f"Price is approx one candle away at {tf_label} TF."
    if ratio < 3:
        return f"Price is {ratio:.1f}x {tf_label} ATR away (2-3 candles)."
    return f"Price is {ratio:.1f}x {tf_label} ATR away — still distant."


# ---------------------------------------------------------------------------
# Fetch with retry + staleness check (B3, B5)
# ---------------------------------------------------------------------------

# Staleness thresholds in hours. If latest candle is older than this, data is
# considered stale and fetch is skipped (after retries).
STALENESS_HOURS = {
    "1h": 2.0,
    "15m": 0.75,
    "5m": 0.30
}


def _check_staleness(df, interval):
    """Return (is_stale, age_hours). age_hours is None if df is empty."""
    if df is None or df.empty:
        return True, None
    try:
        last_ts = df.index[-1]
        if hasattr(last_ts, 'tz_convert') and last_ts.tzinfo is not None:
            last_utc = last_ts.tz_convert('UTC').tz_localize(None).to_pydatetime()
        elif hasattr(last_ts, 'to_pydatetime'):
            last_utc = last_ts.to_pydatetime()
            if last_utc.tzinfo is not None:
                last_utc = last_utc.replace(tzinfo=None)
        else:
            last_utc = last_ts

        age_hours = (datetime.utcnow() - last_utc).total_seconds() / 3600
        max_age = STALENESS_HOURS.get(interval, 2.0)
        return age_hours > max_age, age_hours
    except Exception:
        return False, None  # Can't determine — don't block


def fetch_with_retry(symbol, period, interval, retries=2):
    """
    Fetch H1 from Twelve Data (via feed_adapter) with a staleness check.
    Returns the df on success, None on persistent failure/staleness.
    `period` is accepted for call-site compatibility; the adapter fetches a fixed
    most-recent window (200 H1 bars ≈ 30 days, the live window the engine needs).
    Logs staleness skips to yfinance_stale_log.json (kept name) for weekly review.
    """
    df = feed_adapter.fetch_h1(symbol, outputsize=smc_detector.LIVE_P2_H1_BARS,
                               retries=retries)
    if df is None:
        # Adapter already retried and logged the cause; record the skip for review.
        _log_stale_skip(symbol, interval, "feed fetch failed", None)
        return None

    is_stale, age_hours = _check_staleness(df, interval)
    if not is_stale:
        return df

    last_error = (f"stale data (age {age_hours:.2f}h, "
                  f"limit {STALENESS_HOURS.get(interval, 2.0)}h)")
    print(f"  [SKIP] {symbol} {interval}: {last_error}")
    _log_stale_skip(symbol, interval, last_error, age_hours)
    return None


def _log_stale_skip(symbol, interval, reason, age_hours):
    """Log fetch failures for weekly review."""
    try:
        log = load_json("yfinance_stale_log.json", [])
        log.append({
            "ts": get_ist_now().isoformat(),
            "symbol": symbol,
            "interval": interval,
            "reason": reason,
            "age_hours": round(age_hours, 2) if age_hours is not None else None
        })
        # Keep only last 200 entries
        log = log[-200:]
        save_json("yfinance_stale_log.json", log)
    except Exception as e:
        print(f"  [LOG ERR] stale log: {e}")


def _log_gemini_failure(pair, reason):
    """Log Gemini API failures for weekly review (B1)."""
    try:
        log = load_json("gemini_failure_log.json", [])
        log.append({
            "ts": get_ist_now().isoformat(),
            "pair": pair,
            "reason": reason
        })
        log = log[-200:]
        save_json("gemini_failure_log.json", log)
    except Exception as e:
        print(f"  [LOG ERR] gemini log: {e}")


def _log_chart_failure(pair, chart_type):
    """Log chart render failures for weekly review (B8)."""
    try:
        log = load_json("chart_failure_log.json", [])
        log.append({
            "ts": get_ist_now().isoformat(),
            "pair": pair,
            "chart_type": chart_type
        })
        log = log[-200:]
        save_json("chart_failure_log.json", log)
    except Exception as e:
        print(f"  [LOG ERR] chart log: {e}")


def _log_smtp_failure(recipient, reason):
    """Log SMTP send failures so heartbeat can surface them. Without this,
    a bad app password or transient SMTP outage produces a silent system."""
    try:
        log = load_json("smtp_failure_log.json", [])
        log.append({
            "ts": get_ist_now().isoformat(),
            "recipient": recipient,
            "reason": reason
        })
        log = log[-200:]
        save_json("smtp_failure_log.json", log)
    except Exception as e:
        print(f"  [LOG ERR] smtp log: {e}")


# ---------------------------------------------------------------------------
# Macro news + Gemini
# ---------------------------------------------------------------------------

# News blackout window (hours) from config.json scoring block. This IS the only
# window we flag on: an event is relevant only while now sits inside
# [event - before, event + after]. We deliberately do NOT surface events further
# out (e.g. an ECB print 7h away) — that flagged on every event and was noise.
# Bump to 3/2 in config if a wider avoidance window is wanted.
_NEWS_CFG = config.get("scoring", {})
NEWS_BLACKOUT_BEFORE_H = float(_NEWS_CFG.get("news_blackout_hours_before", 2))
NEWS_BLACKOUT_AFTER_H  = float(_NEWS_CFG.get("news_blackout_hours_after", 1))


def fetch_scheduled_news(now_utc=None):
    """Fetch the ForexFactory scheduled High/Medium-impact calendar ONCE per scan.

    Returns (events, coverage_ok). This is the SAME single-source calendar the
    backtest uses (news_filter.py) — wired into live Phase 2 here so the alert
    surfaces the events that actually move these pairs (NFP, CPI, FOMC, ECB,
    BoE, etc.), instead of the old generic-RSS scrape that rarely contained the
    pair-relevant event. Fetched once and shared across all pairs in the scan.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    # Fetch the FULL IST trading day so the email can show a whole-day calendar
    # (e.g. "FOMC tonight 03:30 IST" on the morning scan), not just events near
    # `now`. The trading day starts 09:00 IST = 03:30 UTC (mirrors trading_day()).
    # The active 2h-before / 1h-after marker is still computed from this same set
    # in get_pair_news_context — wider fetch, identical marker logic.
    ist_now = now_utc + timedelta(hours=5, minutes=30)
    day_start_ist = ist_now.replace(hour=9, minute=0, second=0, microsecond=0)
    if ist_now.hour < 9:  # pre-09:00 IST belongs to the previous trading day
        day_start_ist -= timedelta(days=1)
    start = day_start_ist - timedelta(hours=5, minutes=30)  # back to UTC
    end   = start + timedelta(days=1)
    try:
        res = news_filter.fetch_events(start, end, sources=("ff",))
        cov = res.get("coverage", {})
        coverage_ok = all(cov.values()) if cov else False
        return res.get("events", []), coverage_ok
    except Exception as e:
        print(f"  [NEWS] calendar fetch failed: {type(e).__name__}: {e}")
        return [], False


def get_pair_news_context(pair_name, events, coverage_ok, now_utc=None):
    """Build the per-pair scheduled-news context from the shared calendar.

    INFORMATION ONLY. Nothing here gates, filters or suppresses a trade setup.
    The caller renders two display pieces:
      1. A whole-day list of the pair's High and Medium-impact events (in IST).
      2. An "active now" marker when now sits inside [event-2h, event+1h] —
         this is the heads-up window the user asked for (alert from 2h before a
         release until 1h after). It is a label, not a filter.

    Returns a dict the email renders directly:
      active_event:    dict|None  (event whose 2h-before/1h-after window is live)
      active_now:      bool       (now is inside an event's window)
      day_events:      list       (all pair-relevant events in the IST day)
      coverage_ok:     bool
      headlines_text:  str        (fed to Gemini so its summary is about real events)
      # legacy aliases (kept so existing renderer keys don't break):
      blackout / blackout_event -> active_now / active_event
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    ccys = news_filter.currencies_for_pair(pair_name)
    day_events = sorted(
        (e for e in events if e.get("currency") in ccys),
        key=lambda e: e["ts_utc"],
    )

    # Heads-up window (config: before=2h, after=1h). Pure label — never a gate.
    before = timedelta(hours=NEWS_BLACKOUT_BEFORE_H)
    after  = timedelta(hours=NEWS_BLACKOUT_AFTER_H)
    active_now = False
    active_event = None
    for e in day_events:
        if (e["ts_utc"] - before) <= now_utc <= (e["ts_utc"] + after):
            active_now = True
            active_event = e
            break

    if day_events:
        headlines_text = "\n".join(
            f"- {(e['ts_utc'] + timedelta(hours=5, minutes=30)).strftime('%a %H:%M IST')} "
            f"{e['currency']} {e.get('impact', '').upper()}: {e.get('title', '')}"
            for e in day_events
        )
    else:
        headlines_text = (
            f"No scheduled High/Medium-impact events for {sorted(ccys)} today."
        )

    return {
        "active_now":     active_now,
        "active_event":   active_event,
        "day_events":     day_events,
        "coverage_ok":    coverage_ok,
        "headlines_text": headlines_text,
        # Legacy aliases — keep old renderer/Gemini keys working.
        "blackout":       active_now,
        "blackout_event": active_event,
        "relevant":       day_events,
    }


def fetch_macro_news(pair_name, news_ctx=None):
    """Headlines string handed to Gemini. Now sourced from the scheduled FF
    calendar (via get_pair_news_context) so the AI summary is about real,
    pair-relevant events instead of a generic global feed. `news_ctx` is the
    precomputed per-pair context; falls back to a self-contained fetch if a
    caller doesn't pass it (defensive — keeps the old signature working).
    """
    if news_ctx is None:
        events, cov = fetch_scheduled_news()
        news_ctx = get_pair_news_context(pair_name, events, cov)
    return news_ctx.get("headlines_text", "Could not fetch latest news.")


_GEMINI_CACHE_PATH = os.path.join("state", "gemini_cache.json")


def _gemini_cache_key(pair, date_ist):
    # Keyed on pair+day only — NOT bias. Bias can flip intraday as structure
    # updates live; that's real trading logic and must keep moving freely.
    # The macro cache is a separate concern (Gemini call-budget control) and
    # must never gate on or react to bias.
    return f"{pair}|{date_ist}"


def _gemini_cache_load():
    return load_json(_GEMINI_CACHE_PATH, {})


def _gemini_cache_save(cache, today_date_ist):
    # GC: drop any key not from today before writing.
    clean = {k: v for k, v in cache.items() if k.endswith(f"|{today_date_ist}")}
    try:
        os.makedirs("state", exist_ok=True)
        save_json(_GEMINI_CACHE_PATH, clean)
    except Exception as e:
        print(f"  [GEMINI CACHE] write failed: {e}")


def _gemini_cache_get(pair, date_ist):
    cache = _gemini_cache_load()
    key = _gemini_cache_key(pair, date_ist)
    entry = cache.get(key)
    if entry and (entry.get("macro_summary") or entry.get("macro_unavailable")):
        print(f"  [GEMINI] {pair}: cache hit for {date_ist}")
        return entry
    return None


def _gemini_cache_set(pair, date_ist, result):
    if not result or (not result.get("macro_summary") and not result.get("macro_unavailable")):
        return
    cache = _gemini_cache_load()
    key = _gemini_cache_key(pair, date_ist)
    cache[key] = result
    _gemini_cache_save(cache, date_ist)


def call_gemini_flash(pair, bias, news_headlines):
    date_ist = get_day_id_ist()

    # Cache hit: same pair+day already summarised — skip API call. Keyed on
    # pair+day only (not bias) so a live bias flip never triggers a re-call;
    # the cached briefing is reused as-is for the rest of the day.
    cached = _gemini_cache_get(pair, date_ist)
    if cached is not None:
        return cached

    prompt = f"""
    You are a macro risk briefer for a forex/commodities trader. Be direct and specific.
    PAIR: {pair} | DIRECTION: {bias}
    TODAY'S SCHEDULED HIGH-IMPACT EVENTS:
    {news_headlines}

    Write a macro briefing the trader reads before deciding whether to take this setup.
    Rules:
    - Name each event explicitly (e.g. "US CPI at 18:30 IST").
    - For each event, say in one plain sentence what outcome would HELP this trade and what would HURT it.
    - If the event is already past, say so and state what the actual result was if it is in the data.
    - If there are no high-impact events today for {pair}, say so clearly in one sentence.
    - Do NOT use filler like "traders should monitor" or "volatility may increase". Be specific.
    - Total length: 2-4 sentences maximum.

    OUTPUT FORMAT (Strict JSON):
    {{
        "macro_summary": "Your 2-4 sentence briefing here."
    }}
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.15,
            "thinkingConfig": {"thinkingBudget": 0}
        }
    }
    last_err = "unknown"
    for attempt in range(3):
        try:
            resp = requests.post(url, json=body, timeout=20)
            r = resp.json()
            if "candidates" in r:
                result = json.loads(r["candidates"][0]["content"]["parts"][0]["text"].strip())
                _gemini_cache_set(pair, date_ist, result)
                return result

            # No candidates -> the body always says why. Capture the real reason
            # instead of discarding it (a blind "no candidates field" string is
            # what made every past failure undiagnosable).
            if "error" in r:
                code = r["error"].get("code", resp.status_code)
                msg  = str(r["error"].get("message", ""))[:120]
                last_err = f"API error {code}: {msg}"
                # Distinguish the two kinds of 429:
                #   - quota / credits exhausted  -> NOT transient. Retrying 3x
                #     just floods the failure log and burns nothing useful (the
                #     account is out). Fail once.
                #   - bare per-minute rate limit -> transient. One short back-off
                #     can clear it.
                # 503 (overloaded) is transient too.
                quota_dead = code == 429 and any(
                    k in msg.lower()
                    for k in ("quota", "credit", "billing", "plan")
                )
                transient = (code == 503) or (code == 429 and not quota_dead)
                if transient and attempt < 2:
                    wait = 10 * (attempt + 1)  # 10s, then 20s
                    print(f"  [GEMINI] {pair}: {code}, backing off {wait}s "
                          f"(attempt {attempt + 1}/3)")
                    time.sleep(wait)
                    continue
                break
            elif r.get("promptFeedback", {}).get("blockReason"):
                last_err = f"blocked: {r['promptFeedback']['blockReason']}"
                break  # safety block is deterministic — same prompt, same result
            else:
                last_err = f"no candidates, unknown body: {str(r)[:120]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:80]}"
            if attempt < 2:
                time.sleep(3)

    # All attempts failed — log and surface the failure HONESTLY to the
    # trader. Previously this returned a fake "safe" summary that read like
    # "no Tier-1 events" — misleading. macro_summary=None + macro_unavailable
    # lets the email render a distinct "unavailable" banner.
    _log_gemini_failure(pair, last_err)
    failure = {"macro_summary": None, "macro_unavailable": True}
    _gemini_cache_set(pair, date_ist, failure)
    return failure


# ---------------------------------------------------------------------------
# Chart generators — H1 context + H1 zoomed entry
# ---------------------------------------------------------------------------

def _base_canvas():
    # Phase 2 context chart keeps its fixed (shorter) figure height; canvas
    # theming comes from the shared style engine (charts.py).
    return charts.base_canvas(fig_height=charts.FIG_HEIGHT_P2)


def _draw_candles(ax, df_plot):
    # Canonical candle look (shared with Phase 1): thin body / fat wick. This is
    # the 2C drift fix — Phase 2 candles previously rendered squatter (0.8/1.2).
    charts.draw_candles(
        ax,
        df_plot['Open'].to_numpy(dtype=float),
        df_plot['High'].to_numpy(dtype=float),
        df_plot['Low'].to_numpy(dtype=float),
        df_plot['Close'].to_numpy(dtype=float),
    )


def _p2_swing_markers(ax, df_h1, h1_pos_to_local, n, pair_conf, y_min, y_max, ob=None):
    """Render swing triangles + current-setup broken-swing X on a Phase 2 chart.

    SINGLE SOURCE: reads the persisted lb-3+ATR swing pool from dealing_range
    state (walls['swings']) — the exact swings Phase 1 renders. Phase 2 detects
    nothing itself. Each swing is positioned by ts using locate_ob_candle_idx
    (same df_h1 index frame), then mapped to its local plot position via
    h1_pos_to_local (a dict built from the actual df_plot rows after dropna+tail).
    Using index arithmetic (abs_i - window_start) is wrong when dropna removes
    any rows in the tail window — the map is the correct approach. Any failure
    is swallowed so a chart never breaks on marker rendering. The X marks ONLY
    the swing broken by THIS OB's defining event — resolved per-OB via
    ob_broken_swing_ts() so the X is always co-located with the OB."""
    try:
        import dealing_range as _dr
        pair_name = (pair_conf or {}).get('name')
        if not pair_name:
            return
        state = _dr.load_state() or {}
        walls = state.get(pair_name) or {}
        swings = smc_detector.swings_for_chart(walls)
        if not swings:
            return
        SWING_COLOR = '#d4a017'
        SETUP_BREAK_COLOR = '#ffffff'  # max contrast on dark bg + red/green candles; bold X marker, distinct from the thin white price line
        offset = (y_max - y_min) * 0.012
        ob_bst = smc_detector.ob_broken_swing_ts(ob, walls) if ob else None
        # Compare instants, not raw strings: the swing pool and event ring can
        # serialize the same moment in different tz forms (see ts_to_utc_instant),
        # which a string == would miss — silently dropping the break X.
        ob_bst_inst = smc_detector.ts_to_utc_instant(ob_bst) if ob_bst else None
        for s in swings:
            ts = s.get('ts')
            if not ts:
                continue
            abs_i, found = smc_detector.locate_ob_candle_idx(df_h1, ts)
            if not found:
                continue
            xi = h1_pos_to_local.get(abs_i, -1)
            if not (0 <= xi < n):
                continue
            # Anchor to the candle's ACTUAL extreme at abs_i (high -> High,
            # low -> Low), not the price stored in state — yfinance revises
            # recent intraday bars, so the stored price can float the marker
            # off its candle. Mirrors the Phase 1 radar fix.
            is_high = (s['type'] == 'high')
            try:
                candle_price = float(df_h1['High'].iloc[abs_i]) if is_high \
                    else float(df_h1['Low'].iloc[abs_i])
            except Exception:
                continue
            if ob_bst_inst is not None and \
                    smc_detector.ts_to_utc_instant(ts) == ob_bst_inst:
                ax.scatter([xi], [candle_price], marker='x', s=70,
                           color=SETUP_BREAK_COLOR, linewidths=2.0, zorder=8)
            elif is_high:
                ax.scatter([xi], [candle_price + offset], marker='v', s=42,
                           color=SWING_COLOR, edgecolors=SWING_COLOR,
                           linewidths=1.0, zorder=6)
            else:
                ax.scatter([xi], [candle_price - offset], marker='^', s=42,
                           color=SWING_COLOR, edgecolors=SWING_COLOR,
                           linewidths=1.0, zorder=6)
    except Exception:
        return


def _fig_to_b64(fig):
    # One save path, shared with Phase 1 (charts.py).
    return charts.fig_to_b64(fig)


def generate_h1_zoomed_chart(df_h1, ob, pair_conf, title, levels=None):
    """H1 zoomed entry chart. Replaces the legacy M15 approach chart.

    Visual choices (per trader preference 2026-05-26, window 30 -> 60 on
    2026-06-04, then -> 45 on 2026-06-16 because 60 candles squeezed into the
    fixed-width figure made bodies unreadably small, especially on a strong
    trend like NAS100 where the y-axis also stretches to fit far TP levels):
      - 45 H1 candles, focused on the OB and approach to it (long candles).
      - Wider figsize relative to candle count -> visibly larger bodies.
      - Same colour palette as the wide H1 context chart for consistency.
      - Renders OB band, entry/SL/TP1/TP2 lines, current price line,
        FVG box (if present), OB candle outline, swing triangles + broken-swing
        X. No dealing-range band (intentional -- this chart is about the entry,
        not the macro view).
    """
    try:
        dp = pair_conf.get("decimal_places", 5)
        tail_n = 45
        df_clean = df_h1.dropna(subset=['Open', 'High', 'Low', 'Close'])
        df_plot_raw = df_clean.tail(tail_n)
        # Build a map: df_h1 integer position -> df_plot local position.
        # Using index arithmetic (abs_i - window_start) is wrong whenever dropna
        # removes any rows inside the tail window — every marker shifts by the
        # number of dropped rows before it. The map is the only correct approach.
        # Guard: get_loc returns a non-int (slice/array) on duplicate index values
        # (yfinance occasionally emits duplicate timestamps). Skip those rows so
        # the map only holds clean int -> int entries.
        h1_pos_to_local = {}
        for local_i, idx in enumerate(df_plot_raw.index):
            loc = df_h1.index.get_loc(idx)
            if isinstance(loc, int):
                h1_pos_to_local[loc] = local_i
        df_plot = df_plot_raw.copy().reset_index(drop=True)
        n = len(df_plot)
        if n < 5:
            return None

        fig, ax = charts.base_canvas(fig_height=charts.FIG_HEIGHT_ZOOM,
                                     fig_width=charts.FIG_WIDTH_ZOOM)

        # Draw candles -- INTENTIONALLY wider bodies for the zoomed entry view
        # (close-up; not the drift the 2C unify fixed). Colours are the shared
        # palette so they can never drift from the context/scout charts.
        charts.draw_candles(
            ax,
            df_plot['Open'].to_numpy(dtype=float),
            df_plot['High'].to_numpy(dtype=float),
            df_plot['Low'].to_numpy(dtype=float),
            df_plot['Close'].to_numpy(dtype=float),
            body_w=charts.BODY_W_ZOOM, wick_w=charts.WICK_W_ZOOM,
            body_alpha=charts.BODY_ALPHA_ZOOM, butt_cap=False,
        )

        proximal = float(ob.get('proximal_line', 0))
        distal   = float(ob.get('distal_line', 0))
        zone_hi, zone_lo = max(proximal, distal), min(proximal, distal)

        if zone_hi > 0 and zone_lo > 0:
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                facecolor='#9b59b6', alpha=0.18, zorder=1
            ))
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                fill=False, edgecolor='#bb8fce', linestyle=':', linewidth=1.5, zorder=2
            ))

        # FVG box (H1 only).
        fvg = ob.get('fvg', {}) or {}
        if fvg.get('exists'):
            ft, fb = float(fvg.get('fvg_top', 0)), float(fvg.get('fvg_bottom', 0))
            c1_ts = fvg.get('c1_timestamp')
            c1_resolved = None
            if c1_ts:
                idx_c1, found_c1 = smc_detector.locate_ob_candle_idx(df_h1, c1_ts)
                if found_c1:
                    c1_resolved = idx_c1
            if ft > 0 and fb > 0 and c1_resolved is not None:
                # FVG box sits on C2 (one candle after C1). Resolve via map.
                mid_local = h1_pos_to_local.get(c1_resolved + 1, -1)
                if 0 <= mid_local < n:
                    mit = fvg.get('mitigation', 'pristine')
                    face_col, edge_col = ('#f4d03f', '#f1c40f') if mit == 'partial' else ('#27ae60', '#2ecc71')
                    ax.add_patch(patches.Rectangle(
                        (mid_local - 0.6, fb), 3.0, ft - fb,
                        facecolor=face_col, alpha=0.18, zorder=1
                    ))
                    ax.add_patch(patches.Rectangle(
                        (mid_local - 0.6, fb), 3.0, ft - fb,
                        fill=False, edgecolor=edge_col, linestyle='--', linewidth=1.2, zorder=2
                    ))

        # Entry / SL / TP lines.
        entry_p = float(levels.get('entry', 0)) if isinstance(levels, dict) else 0
        sl_p    = float(levels.get('sl', 0))    if isinstance(levels, dict) else 0
        tp1_p   = float(levels.get('tp1', 0))   if isinstance(levels, dict) else 0
        tp2_p   = float(levels.get('tp2', 0))   if isinstance(levels, dict) else 0
        if entry_p > 0:
            ax.axhline(y=entry_p, color='#e67e22', linewidth=1.2, linestyle='--', alpha=0.9, zorder=3)
        if sl_p > 0:
            ax.axhline(y=sl_p, color='#e74c3c', linewidth=1.2, linestyle='--', alpha=0.9, zorder=3)
        if tp1_p > 0:
            ax.axhline(y=tp1_p, color='#27ae60', linewidth=1.0, linestyle=':', alpha=0.85, zorder=3)
        if tp2_p > 0:
            ax.axhline(y=tp2_p, color='#1e8449', linewidth=1.0, linestyle=':', alpha=0.85, zorder=3)

        current = float(df_plot['Close'].iloc[-1])
        ax.axhline(y=current, color='#ffffff', linewidth=0.9, linestyle='-', alpha=0.55, zorder=2)

        # OB candle outline (white).
        ob_ts_iso = ob.get('ob_timestamp')
        if ob_ts_iso:
            abs_idx, found = smc_detector.locate_ob_candle_idx(df_h1, ob_ts_iso)
            if found:
                local_idx = h1_pos_to_local.get(abs_idx, -1)
                if 0 <= local_idx < n:
                    ob_c_h = float(df_plot['High'].iloc[local_idx])
                    ob_c_l = float(df_plot['Low'].iloc[local_idx])
                    ax.add_patch(patches.Rectangle(
                        (local_idx - 0.45, ob_c_l), 0.9, ob_c_h - ob_c_l,
                        fill=False, edgecolor='#ffffff', linewidth=1.7, zorder=5
                    ))

        # Right-edge price labels (entry/SL/TP1/TP2, current price).
        right_labels = []
        if entry_p > 0: right_labels.append((entry_p, f" {entry_p:.{dp}f}", '#e67e22'))
        if sl_p > 0:    right_labels.append((sl_p,    f" {sl_p:.{dp}f}",    '#e74c3c'))
        if tp1_p > 0:   right_labels.append((tp1_p,   f" {tp1_p:.{dp}f}",   '#27ae60'))
        if tp2_p > 0:   right_labels.append((tp2_p,   f" {tp2_p:.{dp}f}",   '#1e8449'))
        right_labels.append((current, f" {current:.{dp}f}", '#ffffff'))
        right_stacked = smc_detector.stack_labels(right_labels, pair_conf)
        for adj_price, text, color in right_stacked:
            ax.text(n + 0.6, adj_price, text, color=color, fontsize=10, va='center',
                    fontweight='bold', zorder=5)

        # Left-edge OB proximal / distal markers.
        left_labels = []
        if zone_hi > 0:
            left_labels.append((proximal, f"{proximal:.{dp}f}", '#bb8fce'))
            left_labels.append((distal,   f"{distal:.{dp}f}",   '#bb8fce'))
        left_stacked = smc_detector.stack_labels(left_labels, pair_conf)
        for adj_price, text, color in left_stacked:
            ax.text(-0.7, adj_price, text, color=color, fontsize=9, va='center',
                    ha='left', fontweight='bold', zorder=5,
                    bbox=dict(facecolor='#131722', edgecolor='none', pad=1.5, alpha=0.75))

        # Swing triangles + broken-swing X (single source: dealing_range state).
        _p2_swing_markers(ax, df_h1, h1_pos_to_local, n, pair_conf,
                          float(df_plot['Low'].min()), float(df_plot['High'].max()), ob=ob)

        # Y-axis (2026-06-16): anchor on the candles + the ENTRY-relevant levels
        # (zone, entry, SL). TP1/TP2 are NOT force-included — on a strong trend
        # they sit far from price and, if included, compress every candle into a
        # flat line (the NAS100 "tiny candles" complaint). A far TP is allowed to
        # fall off-chart; its right-edge price label still shows. We only let a
        # TP stretch the axis if it's within ~50% beyond the candle range, so a
        # near TP still frames naturally.
        cl, ch = float(df_plot['Low'].min()), float(df_plot['High'].max())
        y_min, y_max = cl, ch
        for val in (zone_lo, zone_hi, entry_p, sl_p):
            if val > 0:
                y_min = min(y_min, val)
                y_max = max(y_max, val)
        candle_span = max(ch - cl, 1e-9)
        tp_limit_hi = y_max + candle_span * 0.5
        tp_limit_lo = y_min - candle_span * 0.5
        for val in (tp1_p, tp2_p):
            if val > 0 and tp_limit_lo <= val <= tp_limit_hi:
                y_min = min(y_min, val)
                y_max = max(y_max, val)
        pad = (y_max - y_min) * 0.10
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlim(-1, n + 10)
        ax.set_title(title, color='#dddddd', fontsize=11, pad=8, loc='left')
        ax.tick_params(colors='#888', labelsize=9)
        ax.yaxis.tick_right()
        ax.set_xticks([])
        plt.tight_layout(pad=0.5)
        return _fig_to_b64(fig)
    except Exception as e:
        print(f"H1 zoomed chart error: {e}")
        plt.close('all')
        return None


def generate_h1_chart(df_h1, ob, pair_conf, title, levels=None, dealing_range=None,
                      pools=None, eq_ctx=None):
    try:
        dp = pair_conf.get("decimal_places", 5)
        # Window widened 80 -> 130 to match the Phase 1 context chart. Same
        # figsize as before, so 130 candles render at Phase 1 proportions
        # (not stretched / thinned). Phase 2 functionality is unchanged.
        # window_start uses raw df_h1 length (same index frame as
        # locate_ob_candle_idx, which indexes df_h1) to preserve the existing
        # sweep / FVG / marker positioning contract.
        tail_n = 130
        df_plot = df_h1.dropna(subset=['Open', 'High', 'Low', 'Close']).tail(tail_n).copy().reset_index(drop=True)
        n = len(df_plot)
        if n < 5:
            return None

        fig, ax = _base_canvas()
        _draw_candles(ax, df_plot)

        full_n = len(df_h1)
        window_start = max(0, full_n - tail_n)

        proximal = float(ob.get('proximal_line', 0))
        distal = float(ob.get('distal_line', 0))
        zone_hi, zone_lo = max(proximal, distal), min(proximal, distal)

        # PD/PW pool levels: accept the full snapshot ({"pools": {...}}) or the
        # pools sub-dict itself. None / empty = no lines (feed degrade).
        pool_lines = None
        if pools:
            pool_lines = pools.get("pools", pools)
        # Proximity gate (mirrors Phase 1's _wall_in_view): a pool level is
        # drawn only when it sits within max(2.5*ATR, 1.5*candle-range) of the
        # candle range, so a stale far level can't stretch the axis or leave a
        # stray line. Same band logic as the DR walls in the scout chart.
        _candle_lo = float(df_plot['Low'].min())
        _candle_hi = float(df_plot['High'].max())
        _candle_span = max(_candle_hi - _candle_lo, 1e-9)
        _p2_atr = float(ob.get('h1_atr', 0.0) or 0.0)
        if _p2_atr > 0:
            _pool_prox_max = max(2.5 * _p2_atr, 1.5 * _candle_span)
        else:
            _pool_prox_max = 1.5 * _candle_span

        def _pool_in_view(price):
            return (price is not None
                    and _candle_lo - _pool_prox_max <= price <= _candle_hi + _pool_prox_max)

        # --- Zone band ---
        if zone_hi > 0 and zone_lo > 0:
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                facecolor='#9b59b6', alpha=0.15, zorder=1
            ))
            ax.add_patch(patches.Rectangle(
                (0, zone_lo), n + 5, zone_hi - zone_lo,
                fill=False, edgecolor='#bb8fce', linestyle=':', linewidth=1.5, zorder=2
            ))

        # --- BOS / CHoCH horizontal line ---
        # Palette (matches smc_radar chart + _chart_legend_html; v2 has no
        # Major/Minor — only BOS / Range BOS / CHoCH):
        #   BOS        -> magenta #e91e63 (internal swing break)
        #   Range BOS  -> teal  #00897b  (H4 dealing-range wall break)
        #   CHoCH      -> orange #ff9800 (trend flip)
        bos_price = float(ob.get('bos_swing_price', 0))
        bos_tag = ob.get('bos_tag', 'BOS')
        bos_tier = ob.get('bos_tier', 'BOS')
        if bos_tag == 'CHoCH':
            bos_color = '#ff9800'
        elif bos_tier == 'Range':
            bos_color = '#00897b'
        else:
            bos_color = '#e91e63'
        if bos_price > 0:
            ax.axhline(y=bos_price, color=bos_color, linewidth=0.8, linestyle='--', alpha=0.7, zorder=2)

        # --- Liquidity sweep overlay — SWEEP V2 (pool-anchored, 2026-07-19) ---
        # One source: the frozen ob['sweep_v2'] snapshot (the same read the
        # emails narrate — one definition on every surface). Draws:
        #   - dotted rectangle around the raid wick of the sweep candle,
        #   - dotted line at the raided pool level (6-bar stub into the candle),
        #   - star + 'Sweep' label at the wick tip.
        # The old any-swing overlay (sweep_observed star on the swept swing) is
        # retired with the old narration. Resolve by TIMESTAMP — P1 indices are
        # not portable to P2's fresh dataframe.
        sw = ob.get('sweep_v2')
        if isinstance(sw, dict) and sw.get('exists') and sw.get('sweep_ts'):
            sw_color = '#00e5ff'
            sw_level = sw.get('level')
            ob_dir = ob.get('direction')
            sw_abs_idx, sw_found = smc_detector.locate_ob_candle_idx(df_h1, sw['sweep_ts'])
            if sw_found and sw_abs_idx >= window_start:
                sw_local = sw_abs_idx - window_start
                if 0 <= sw_local < n:
                    sw_o = float(df_plot['Open'].iloc[sw_local])
                    sw_h = float(df_plot['High'].iloc[sw_local])
                    sw_l = float(df_plot['Low'].iloc[sw_local])
                    sw_c = float(df_plot['Close'].iloc[sw_local])
                    body_top = max(sw_o, sw_c)
                    body_bot = min(sw_o, sw_c)
                    # LONG raid -> lower wick took the pool; SHORT -> upper wick.
                    if ob_dir == 'bullish':
                        wick_lo, wick_hi = sw_l, body_bot
                        wick_tip = sw_l
                    else:
                        wick_lo, wick_hi = body_top, sw_h
                        wick_tip = sw_h
                    if wick_hi > wick_lo:
                        ax.add_patch(patches.Rectangle(
                            (sw_local - 0.45, wick_lo), 0.9, wick_hi - wick_lo,
                            fill=False, edgecolor=sw_color, linestyle=':',
                            linewidth=1.5, zorder=5
                        ))
                    if sw_level is not None:
                        x_lo = max(0, sw_local - 6)
                        if x_lo < sw_local:
                            ax.plot([x_lo, sw_local], [sw_level, sw_level],
                                    color=sw_color, linewidth=1.0,
                                    linestyle=(0, (3, 2)), alpha=0.8, zorder=4)
                    ax.scatter([sw_local], [wick_tip], marker='*', s=140,
                               color=sw_color, edgecolors='#001f24',
                               linewidths=0.8, zorder=8)
                    label_dy = -14 if ob_dir == 'bullish' else 14
                    label_va = 'top' if ob_dir == 'bullish' else 'bottom'
                    ax.annotate('Sweep', xy=(sw_local, wick_tip),
                                xytext=(0, label_dy), textcoords='offset points',
                                color=sw_color, fontsize=8, fontweight='bold',
                                ha='center', va=label_va, zorder=8)

        # --- FVG: outline middle (displacement) candle only, slightly wider for mitigation visibility ---
        # Cross-phase safety: resolve c1 candle position via timestamp (c1_idx
        # from Phase 1 is NOT portable to Phase 2's fresh dataframe).
        fvg = ob.get('fvg', {}) or {}
        if fvg.get('exists'):
            ft, fb = float(fvg.get('fvg_top', 0)), float(fvg.get('fvg_bottom', 0))
            c1_ts = fvg.get('c1_timestamp')
            c1_resolved = None
            if c1_ts:
                idx_c1, found_c1 = smc_detector.locate_ob_candle_idx(df_h1, c1_ts)
                if found_c1:
                    c1_resolved = idx_c1
            if c1_resolved is None:
                # Legacy fallback (only valid same-run; Phase 2 with no timestamp
                # means the ob predates this fix — outline likely off but harmless).
                c1_idx_legacy = fvg.get('c1_idx')
                if c1_idx_legacy is not None:
                    try:
                        c1_resolved = int(c1_idx_legacy)
                    except Exception:
                        c1_resolved = None

            if ft > 0 and fb > 0 and c1_resolved is not None:
                mid_abs = c1_resolved + 1  # displacement candle (middle of c1-c2-c3)
                mid_local = mid_abs - window_start
                if 0 <= mid_local < n:
                    mit = fvg.get('mitigation', 'pristine')
                    if mit == 'partial':
                        # Amber — partial mitigation (caution).
                        face_col, edge_col = '#f4d03f', '#f1c40f'
                    else:
                        face_col, edge_col = '#27ae60', '#2ecc71'
                    fvg_x_start = mid_local - 0.6
                    fvg_width = 1.8 + 1.2  # 1.8 candle widths + extra right-side mitigation visibility
                    ax.add_patch(patches.Rectangle(
                        (fvg_x_start, fb), fvg_width, ft - fb,
                        facecolor=face_col, alpha=0.15, zorder=1
                    ))
                    ax.add_patch(patches.Rectangle(
                        (fvg_x_start, fb), fvg_width, ft - fb,
                        fill=False, edgecolor=edge_col, linestyle='--', linewidth=1.0, zorder=2
                    ))
        # --- Dealing range band + equilibrium ---
        dr_eq = None
        if dealing_range and dealing_range.get('valid'):
            dr_hi = float(dealing_range['range_high'])
            dr_lo = float(dealing_range['range_low'])
            dr_eq = float(dealing_range['equilibrium'])
            y_min_candle = float(df_plot['Low'].min())
            y_max_candle = float(df_plot['High'].max())
            candle_range = y_max_candle - y_min_candle
            if candle_range > 0 and (dr_hi - dr_lo) < candle_range * 3:
                ax.add_patch(patches.Rectangle(
                    (0, dr_lo), n + 5, dr_hi - dr_lo,
                    facecolor='#3498db', alpha=0.06, zorder=0
                ))
                ax.add_patch(patches.Rectangle(
                    (0, dr_lo), n + 5, dr_hi - dr_lo,
                    fill=False, edgecolor='#5dade2', linestyle='-.', linewidth=0.8, zorder=1
                ))
            ax.axhline(y=dr_eq, color='#5dade2', linewidth=0.9, linestyle='-.', alpha=0.6, zorder=2)

        # --- Entry / SL / TP horizontal lines ---
        entry_p = 0
        sl_p = 0
        tp1_p = 0
        tp2_p = 0
        if levels and levels.get('valid', True):
            entry_p = float(levels.get('entry', 0))
            sl_p    = float(levels.get('sl', 0))
            tp1_p   = float(levels.get('tp1', 0))
            tp2_p   = float(levels.get('tp2', 0))
            if entry_p > 0:
                ax.axhline(y=entry_p, color='#e67e22', linewidth=1.0, linestyle='--', alpha=0.8, zorder=3)
            if sl_p > 0:
                ax.axhline(y=sl_p, color='#e74c3c', linewidth=1.0, linestyle='--', alpha=0.8, zorder=3)
            if tp1_p > 0:
                ax.axhline(y=tp1_p, color='#27ae60', linewidth=1.0, linestyle=':', alpha=0.85, zorder=3)
            if tp2_p > 0:
                ax.axhline(y=tp2_p, color='#1e8449', linewidth=1.0, linestyle=':', alpha=0.85, zorder=3)

        # PD/PW + EQ liquidity lines are drawn AFTER the y-axis range is set
        # (below), so off-screen edge arrows land on the final padded bounds.

        # --- Current price line ---
        current = float(df_plot['Close'].iloc[-1])
        ax.axhline(y=current, color='#ffffff', linewidth=0.8, linestyle='-', alpha=0.5, zorder=2)

        # --- OB candle outline (white) ---
        ob_ts_iso = ob.get('ob_timestamp')
        if ob_ts_iso:
            abs_idx, found = smc_detector.locate_ob_candle_idx(df_h1, ob_ts_iso)
            if found and abs_idx >= window_start:
                local_idx = abs_idx - window_start
                if 0 <= local_idx < n:
                    ob_c_h = float(df_plot['High'].iloc[local_idx])
                    ob_c_l = float(df_plot['Low'].iloc[local_idx])
                    ax.add_patch(patches.Rectangle(
                        (local_idx - 0.5, ob_c_l), 1.0, ob_c_h - ob_c_l,
                        fill=False, edgecolor='#ffffff', linewidth=1.5, zorder=5
                    ))

        # --- BOS/CHoCH break candle outline (cyan or orange) ---
        br_start, br_end = smc_detector.compute_h1_break_candle_span(df_h1, ob, None)
        if br_start is not None and br_end is not None:
            for abs_i in range(br_start, br_end + 1):
                if abs_i < window_start:
                    continue
                local_i = abs_i - window_start
                if 0 <= local_i < n:
                    c_h = float(df_plot['High'].iloc[local_i])
                    c_l = float(df_plot['Low'].iloc[local_i])
                    ax.add_patch(patches.Rectangle(
                        (local_i - 0.5, c_l), 1.0, c_h - c_l,
                        fill=False, edgecolor=bos_color, linewidth=1.5, zorder=5
                    ))

        # --- Right-edge tags: ENTRY, SL, TP1, TP2 (numbers only, colour-matched) ---
        right_labels = []
        if entry_p > 0:
            right_labels.append((entry_p, f" {entry_p:.{dp}f}", '#e67e22'))
        if sl_p > 0:
            right_labels.append((sl_p, f" {sl_p:.{dp}f}", '#e74c3c'))
        if tp1_p > 0:
            right_labels.append((tp1_p, f" {tp1_p:.{dp}f}", '#27ae60'))
        if tp2_p > 0:
            right_labels.append((tp2_p, f" {tp2_p:.{dp}f}", '#1e8449'))
        right_stacked = smc_detector.stack_labels(right_labels, pair_conf)
        for adj_price, text, color in right_stacked:
            ax.text(n + 1, adj_price, text, color=color, fontsize=10, va='center',
                    fontweight='bold', zorder=5)

        # --- Left-edge tags: proximal, distal, BOS/CHoCH ---
        left_labels = []
        if zone_hi > 0:
            left_labels.append((proximal, f"{proximal:.{dp}f}", '#bb8fce'))
            left_labels.append((distal, f"{distal:.{dp}f}", '#bb8fce'))
        if bos_price > 0:
            left_labels.append((bos_price, f"{bos_price:.{dp}f}", bos_color))
        left_stacked = smc_detector.stack_labels(left_labels, pair_conf)
        for adj_price, text, color in left_stacked:
            ax.text(-1, adj_price, text, color=color, fontsize=8, va='center',
                    ha='left', fontweight='bold', zorder=5,
                    bbox=dict(facecolor='#131722', edgecolor='none', pad=1.5, alpha=0.75))

        # --- Mid-chart tags: current price, EQ ---
        mid_x = n / 2.0
        center_labels = []
        center_labels.append((current, f"{current:.{dp}f}", '#ffffff'))
        if dr_eq is not None:
            center_labels.append((dr_eq, f"{dr_eq:.{dp}f}", '#5dade2'))
        center_stacked = smc_detector.stack_labels(center_labels, pair_conf)
        for adj_price, text, color in center_stacked:
            ax.text(mid_x, adj_price, text, color=color, fontsize=8, va='center',
                    ha='center', fontweight='bold', zorder=5,
                    bbox=dict(facecolor='#131722', edgecolor='none', pad=1.5, alpha=0.75))

        # --- Swing markers (triangles + broken-swing X) -----------------------
        # SINGLE SOURCE: same persisted lb-3+ATR swing pool from dealing_range
        # state that Phase 1 renders. Phase 2 does NOT detect swings itself.
        h1_pos_to_local_ctx = {}
        for local_i, idx in enumerate(df_h1.dropna(subset=['Open','High','Low','Close']).tail(tail_n).index):
            loc = df_h1.index.get_loc(idx)
            if isinstance(loc, int):
                h1_pos_to_local_ctx[loc] = local_i
        _p2_swing_markers(ax, df_h1, h1_pos_to_local_ctx, n, pair_conf,
                          float(df_plot['Low'].min()), float(df_plot['High'].max()), ob=ob)

        # --- Y-axis range ---
        y_min, y_max = float(df_plot['Low'].min()), float(df_plot['High'].max())
        for val in [zone_lo, zone_hi]:
            if val > 0:
                y_min = min(y_min, val)
                y_max = max(y_max, val)
        if sl_p > 0:
            y_min = min(y_min, sl_p)
        if entry_p > 0:
            y_max = max(y_max, entry_p)
        # In-view PD/PW pool levels: keep them inside the axis so a drawn line
        # isn't clipped. Far levels were already gated out of drawing above.
        if pool_lines:
            for _pk in ("pdh", "pdl", "pwh", "pwl"):
                _p = pool_lines.get(_pk)
                _lv = _p.get("level") if _p else None
                if _lv is None:
                    continue
                try:
                    _lv_f = float(_lv)
                except (TypeError, ValueError):
                    continue
                if _pool_in_view(_lv_f):
                    y_min = min(y_min, _lv_f)
                    y_max = max(y_max, _lv_f)
        pad = (y_max - y_min) * 0.08
        _ylim_lo, _ylim_hi = y_min - pad, y_max + pad
        ax.set_ylim(_ylim_lo, _ylim_hi)
        ax.set_xlim(-1, n + 8)

        # --- PD/PW pool levels (PDH/PDL/PWH/PWL) + EQ shelves ---
        # Same shared draw + style as the Phase 1 scout chart, so the two emails
        # can't drift. In-view levels draw a full dotted line; off-view levels
        # draw an edge arrow (↑/↓) on the final padded bounds (#6). EQ shelves
        # (intact only) draw as teal dotted lines (#5). Observation only.
        if pool_lines:
            charts.draw_pool_lines(ax, pool_lines, x_right=n + 5,
                                   in_view=_pool_in_view, zorder=2,
                                   y_min=_ylim_lo, y_max=_ylim_hi,
                                   x_center=n / 2.0)
        if eq_ctx:
            # EQ shelves as dotted SEGMENTS between their touch bars (matches
            # Phase 1). Map a member ts -> local plot-x via the SAME contract the
            # swing markers use here (locate_ob_candle_idx -> h1_pos_to_local_ctx),
            # so shelf endpoints land on the exact touch candles even when dropna
            # removed rows inside the window.
            def _eq_ts_to_x_p2(ts_iso):
                _abs, _found = smc_detector.locate_ob_candle_idx(df_h1, ts_iso)
                if not _found:
                    return None
                _xi = h1_pos_to_local_ctx.get(_abs, -1)
                return _xi if 0 <= _xi < n else None

            charts.draw_eq_lines(ax, eq_ctx, in_view=_pool_in_view, zorder=2,
                                 y_min=_ylim_lo, y_max=_ylim_hi,
                                 x_center=n / 2.0,
                                 ts_to_x=_eq_ts_to_x_p2,
                                 x_right=n - 1)

        ax.set_title(title, color='#dddddd', fontsize=11, pad=8, loc='left')
        ax.tick_params(colors='#888', labelsize=9)
        ax.yaxis.tick_right()
        ax.set_xticks([])
        plt.tight_layout(pad=0.5)
        return _fig_to_b64(fig)
    except Exception as e:
        print(f"H1 chart error: {e}")
        plt.close('all')
        return None
        


# ---------------------------------------------------------------------------
# Email assembly
# ---------------------------------------------------------------------------

def build_scorecard_html(rows, total, total_max=10.0, display_total=None):
    """Per-row math and the header `total/total_max` are the REAL score
    (8 for non-JPY forex, 10 otherwise). `display_total`, when given, is the
    /10-normalized score shown as the headline so alerts compare across
    instruments; the real math is kept beside it for trust.
    """
    body = ""
    for label, score, max_score, status, expl in rows:
        if status == "info":
            # Display-only row (e.g. PD post-removal). No score column shown.
            body += f"""
            <tr>
              <td style="padding:5px 8px;color:#888;font-weight:bold;font-size:13px;white-space:nowrap;vertical-align:top;">&bull;</td>
              <td style="padding:5px 8px;color:#eee;font-size:12px;white-space:nowrap;vertical-align:top;">{label}</td>
              <td style="padding:5px 8px;color:#888;font-size:11px;font-family:monospace;white-space:nowrap;vertical-align:top;">info</td>
              <td style="padding:5px 8px;color:#bbb;font-size:12px;line-height:1.45;">{expl}</td>
            </tr>"""
            continue
        icon = {"ok": "&#10003;", "warn": "!", "fail": "&#10007;"}.get(status, "&bull;")
        color = {"ok": "#27ae60", "warn": "#e67e22", "fail": "#e74c3c"}.get(status, "#888")
        body += f"""
        <tr>
          <td style="padding:5px 8px;color:{color};font-weight:bold;font-size:13px;white-space:nowrap;vertical-align:top;">{icon}</td>
          <td style="padding:5px 8px;color:#eee;font-size:12px;white-space:nowrap;vertical-align:top;">{label}</td>
          <td style="padding:5px 8px;color:#aaa;font-size:11px;font-family:monospace;white-space:nowrap;vertical-align:top;">{score}/{max_score}</td>
          <td style="padding:5px 8px;color:#bbb;font-size:12px;line-height:1.45;">{expl}</td>
        </tr>"""

    if display_total is not None and float(total_max) != 10.0:
        # Headline is the /10-normalized score; real math shown beside it.
        headline = (
            f'<span style="color:#eee;font-size:14px;font-weight:bold;">{display_total}/10</span>'
            f'<span style="color:#888;font-size:11px;font-weight:normal;">'
            f' &nbsp;(real {total}/{total_max})</span>'
        )
    else:
        headline = f'<span style="color:#eee;font-size:14px;font-weight:bold;">{total}/{total_max}</span>'
    return f"""
    <div style="margin-bottom:14px;">
      <div style="color:#aaa;font-size:11px;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase;">
        Confluence Scorecard &mdash; {headline}
      </div>
      <table style="width:100%;border-collapse:collapse;background:#1a1a2e;border-radius:6px;">
        <tbody>{body}</tbody>
      </table>
    </div>"""

def _chart_legend_html(bos_tag="BOS", bos_tier="BOS"):
    """Colour-code legend rendered below each chart. Cosmetic only.

    Args kept for backwards-compat call sites but ignored — like the Phase 1
    legend, this surfaces ALL structure-event colours at once instead of
    switching on a single zone's event.

    The v2 engine has ONE structural tier — there is no Major/Minor. The only
    event types are BOS (internal swing break), Range BOS (H4 dealing-range
    wall break) and CHoCH (trend flip). Colours below match the chart exactly
    (see smc_radar chart rendering): BOS #e91e63, Range BOS #00897b,
    CHoCH #ff9800.
    """
    items = [
        ('#bb8fce', 'Zone band (proximal/distal)'),
        ('#2ecc71', 'FVG pristine (displacement)'),
        ('#f1c40f', 'FVG partial (proximal touched)'),
        ('#e91e63', 'BOS break candle / level (internal swing break)'),
        ('#00897b', 'Range BOS break candle / level (H4 dealing-range wall break)'),
        ('#ff9800', 'CHoCH break candle / level (trend flip)'),
        ('#ffffff', 'OB candle / current price'),
        ('#e67e22', 'Entry'),
        ('#e74c3c', 'Stop loss'),
        ('#27ae60', 'TP1'),
        ('#1e8449', 'TP2'),
        ('#5dade2', 'Equilibrium (50% of range)'),
        ('#00e5ff', 'Liquidity sweep — raid of a real pool (wick highlight + level line)'),
    ]
    rows = "".join(
        f'<span style="display:inline-block;margin:2px 10px 2px 0;font-size:11px;color:#bbb;">'
        f'<span style="display:inline-block;width:10px;height:10px;background:{c};'
        f'border-radius:2px;vertical-align:middle;margin-right:5px;"></span>{txt}</span>'
        for c, txt in items
    )
    return (
        f'<div style="margin:4px 0 12px 0;padding:8px 10px;background:#0d0d1a;'
        f'border-radius:4px;line-height:1.8;">{rows}</div>'
    )
def build_trade_email(data, pair, pair_conf, state_msg, scorecard_rows, total_score,
                      atr_label, distance_str, dollar_risk_str, scan_start_ist,
                      h1_chart_ok=True, m15_chart_ok=True):
    dp = pair_conf.get("decimal_places", 5)
    bias = data.get("bias", "-")
    sent_ist = get_ist_now().strftime('%H:%M IST')
    scan_ist = scan_start_ist.strftime('%H:%M IST')
    ist_time = f"Scanned {scan_ist} · Sent {sent_ist}"
    ob = data.get('ob', {})
    levels = data.get('levels', {})
    # Freshness display handled by scorecard row alone; no separate context line.
    bos_tag = ob.get('bos_tag', 'BOS')
    bos_tier = ob.get('bos_tier', 'BOS')
    # H1-only migration (2026-05-26): every pair routes through the limit
    # branch. The legacy ltf_choch (M5 CHoCH) approach block is retired.

    # Setup badge — the mentor's verdict, rendered FIRST so it frames everything
    # below it. Green star for a high-conviction named pattern; red warning for a
    # caution pattern. Nothing rendered when no named pattern matched (most
    # alerts) — the badge means something precisely because it is rare.
    _badge = data.get("setup_badge")
    if _badge and _badge.get("name"):
        if _badge.get("kind") == "caution":
            _bg, _border, _ic, _title_col = "#2d1a1a", "#e74c3c", "&#9888;", "#ff8a80"
        else:
            _bg, _border, _ic, _title_col = "#10241a", "#2ecc71", "&#11088;", "#7fe3a0"
        setup_badge_html = (
            f'<div style="background:{_bg};border:1px solid {_border};border-left:5px solid '
            f'{_border};border-radius:10px;padding:12px 16px;margin-bottom:14px;">'
            f'<div style="color:{_title_col};font-size:15px;font-weight:bold;margin-bottom:4px;">'
            f'{_ic} {_badge["name"]}</div>'
            f'<div style="color:#dcdcdc;font-size:12px;line-height:1.5;">{_badge.get("note","")}</div>'
            f'</div>'
        )
    else:
        setup_badge_html = ""

    # Entry/SL/TP banner removed 2026-06-16 (trader reads levels off the charts
    # + platform, not this text block; the numbers can differ slightly from the
    # broker feed). Levels still drive the charts and subject line.

    # Body shows the REAL math. Both pair classes total 10 (2026-06-18 rebalance):
    # non-JPY forex = Structure 4 | Sweep 1 | FVG 2 | Freshness 2 | Killzone 1;
    # JPY/Gold/NAS = Structure 4 | Sweep 2 | FVG 2 | Freshness 1 | Killzone 1.
    total_max_for_card = scorecard_real_max(pair_conf)
    display_total = normalized_score(total_score, pair_conf)
    scorecard_html = build_scorecard_html(
        scorecard_rows, total_score, total_max_for_card, display_total=display_total
    )

    distance_html = f"""
    <div style="margin-bottom:12px;padding:8px 12px;background:#0d0d1a;border-left:3px solid #00bcd4;border-radius:4px;font-size:12px;color:#bbb;">
        <b style="color:#eee;">Distance:</b> {distance_str} &nbsp;&middot;&nbsp; {atr_label}
    </div>"""

    # Killzone annotation. OB-side is SCORED (zone formed in institutional
    # hours). The entry side is INFO ONLY — an ETA-based forecast of whether the
    # limit will FILL inside a killzone, plus the IST cut-off time to still be in
    # one. ETA = distance / ATR (estimate); the cut-off is exact clock math on
    # the DST-resolved windows.
    ob_in_kz_label = _ob_in_killzone_label(ob, pair_conf)
    entry_kz_label = _entry_killzone_forecast_label(data, pair_conf)

    context_html = f"""
    <div style="margin-bottom:12px;font-size:11px;color:#888;">
        <b style="color:#aaa;">Zone:</b> {_event_label_dir(bos_tag, bos_tier, ob.get('direction'))}
        &nbsp;&middot;&nbsp; Proximal {ob.get('proximal_line', 0):.{dp}f}
        / Distal {ob.get('distal_line', 0):.{dp}f}
        &nbsp;&middot;&nbsp; <b style="color:#aaa;">OB candle:</b> {ob_in_kz_label}
        &nbsp;&middot;&nbsp; <b style="color:#aaa;">Entry:</b> {entry_kz_label}
    </div>"""

    # Break quality — how far PAST its required minimum the structural break
    # cleared (event-aware: a CHoCH must clear more than a BOS to qualify, so we
    # grade times-over-the-floor, not a fixed ATR bar). Info only. The number
    # IS the reason for the verdict, so they stay consistent.
    _bq = ob.get('break_quality') or {}
    _bq_tier = _bq.get('tier')
    _bq_excess = _bq.get('excess')
    if _bq_tier and _bq_excess:
        _bq_col = {'strong': '#27ae60', 'solid': '#f1c40f',
                   'marginal': '#e67e22'}.get(_bq_tier, '#888')
        _bq_word = {'strong': 'Strong', 'solid': 'Solid',
                    'marginal': 'Marginal'}.get(_bq_tier, _bq_tier)
        break_quality_html = (
            '<div style="margin-bottom:12px;font-size:11px;color:#888;">'
            '<b style="color:#aaa;">Break:</b> '
            f'<b style="color:{_bq_col};">{_bq_word}</b> &middot; cleared '
            f'{_bq_excess}&times; the required displacement</div>'
        )
    else:
        break_quality_html = ""

    # Market regime — daily Choppiness Index on the alert's server trading day.
    # Plain word + raw number, no verdict/colour/score — observe only. Low chop =
    # the day trended; high chop = it ranged. The "choppy = worse" idea is an
    # UNPROVEN hypothesis; this line asserts nothing about trade quality.
    # `—` when unmeasurable (None). (see compute_choppiness_index)
    _chop = data.get('chop_at_alert')
    if _chop is not None:
        if _chop <= 38:
            _regime_word = "Trending"
        elif _chop >= 62:
            _regime_word = "Ranging"
        else:
            _regime_word = "Neutral"
        _regime_detail = f"{_regime_word} ({_chop:.0f})"
    else:
        _regime_detail = "&mdash;"
    trend_quality_html = (
        '<div style="margin-bottom:12px;font-size:11px;color:#888;">'
        f'<b style="color:#aaa;">Trend vs range (choppiness):</b> {_regime_detail}</div>'
    )

    # Trend banner (information only — trader decides whether to take counter-trend)
    trend_alignment = data.get("trend_alignment", "ambiguous")
    trend_label = data.get("trend_label", "H1 trend unavailable")
    if trend_alignment == "with_trend":
        trend_bg, trend_border, trend_icon = "#1a3a1a", "#27ae60", "&#9989;"  # green ✅
    elif trend_alignment == "counter_trend":
        trend_bg, trend_border, trend_icon = "#3a1a1a", "#e74c3c", "&#9888;"  # red ⚠️
    else:
        trend_bg, trend_border, trend_icon = "#2a2a1a", "#f39c12", "&#10067;"  # amber ❓
    trend_banner_html = f"""
    <div style="margin-bottom:12px;padding:10px 14px;background:{trend_bg};border-left:4px solid {trend_border};border-radius:4px;font-size:13px;color:#eee;font-weight:bold;">
        {trend_icon} Trend Context: {trend_label}
    </div>"""

    # LIQUIDITY MAP — one consolidated INFORMATION-ONLY block (was two stacked
    # banners for PD/PW pools and EQH/EQL clusters). Fixed shape so the reader
    # always gets the verdict on the last line:
    #   1. the pool status line (PDH/PDL/PWH/PWL glyphs)
    #   2. equal-level stop clusters around price (when present)
    #   3. "→ What it means" — a predefined inference naming the EXACT pool
    #      this trade points at (fixes the old vague "level")
    # The stop-placement warning (stop resting in FRONT of an equal pool) is
    # promoted to the TOP in red — it is the single most actionable line here.
    pools = data.get("pools") or {}
    eq = data.get("eq") or {}
    _pf = pools.get("features") or {}

    _liq_rows = []
    _warn = eq.get("warning")
    if _warn:
        _liq_rows.append(
            f'<div style="margin-bottom:8px;padding:6px 8px;background:#2d1a1a;'
            f'border-left:3px solid #e74c3c;border-radius:3px;color:#f0ad4e;'
            f'font-weight:bold;">{_warn}</div>')
    if pools.get("line"):
        _liq_rows.append(
            '<b style="color:#eee;">Prev day / prev week:</b> '
            f'{pools["line"]}')
    if eq.get("line"):
        _liq_rows.append(f'{eq["line"]}')
    # The "what it means for THIS trade" read — three labelled bullets
    # (Fact / Means / For this trade) so it reads as a list, not a paragraph.
    # format_liquidity_inference returns a list of (label, text) tuples.
    _infer = pool_builder.format_liquidity_inference(_pf, bias)
    if _infer:
        _bullets = "".join(
            f'<div style="margin-top:4px;color:#eee;">'
            f'<span style="color:#5dade2;">&bull;</span> '
            f'<b style="color:#8fc9ea;">{lbl}:</b> {txt}</div>'
            for lbl, txt in _infer)
        _liq_rows.append(
            f'<div style="margin-top:6px;">{_bullets}</div>')

    # Weekly PD zone — HTF premium/discount vs the H4 read. Predefined
    # sentence names both timeframes + both percentages so the trader always
    # sees the numbers and can override manually. Information only.
    _wk = data.get("weekly_pd") or {}
    if _wk.get("line"):
        _liq_rows.append(
            f'<div style="margin-top:6px;color:#eee;">{_wk["line"]}</div>')

    pools_banner_html = ""
    if _liq_rows:
        pools_banner_html = (
            '<div style="margin-bottom:12px;padding:10px 12px;background:#1a1a2e;'
            'border-left:3px solid #5dade2;border-radius:4px;font-size:12px;'
            'color:#bbb;line-height:1.5;">'
            '<div style="color:#aaa;font-size:10px;letter-spacing:0.6px;'
            'text-transform:uppercase;margin-bottom:6px;">Liquidity</div>'
            + '<br>'.join(_liq_rows)
            + '</div>'
        )

    # Approach-into-zone shape (approach_quality.py, aka RETRACE_QUALITY) REMOVED
    # from the email 2026-07-17 (owner call). The three reads (speed / body share
    # / Kaufman ER over a FIXED 4-/7-bar window before fill) were the retracement-
    # quality probe: FX = NULL, gold-only signal never re-proven, fill-time class
    # so it can never inform an alert decision (only cancel a pending limit). They
    # added no decision value and three verdict-less numbers only confused the
    # read. The trades.csv columns stay logged for research — only the email line
    # is gone. See project_retrace_quality memory.

    # Chart blocks with fallback banner (B8)
    if h1_chart_ok:
        h1_chart_block = '<img src="cid:chart_h1" style="width:100%;border-radius:6px;margin-bottom:12px;" />'
    else:
        h1_chart_block = '<div style="padding:10px 12px;background:#2d1a1a;border-left:3px solid #e74c3c;border-radius:4px;color:#e74c3c;font-size:12px;margin-bottom:12px;">&#9888; H1 chart failed to render for this alert. Check GitHub Actions logs.</div>'

    if m15_chart_ok:
        # The "chart_m15" CID is preserved for backward compat with the email
        # MIME pipeline; the bytes are the H1 zoomed entry chart.
        m15_chart_block = '<img src="cid:chart_m15" style="width:100%;border-radius:6px;margin-bottom:12px;" />'
    else:
        m15_chart_block = '<div style="padding:10px 12px;background:#2d1a1a;border-left:3px solid #e74c3c;border-radius:4px;color:#e74c3c;font-size:12px;margin-bottom:12px;">&#9888; H1 zoomed chart failed to render for this alert. Check GitHub Actions logs.</div>'

    sweep_breakdown_html = build_sweep_breakdown_html(data, dp)
    setup_liq_html = build_setup_liq_html(data)

    # Scheduled-news block — INFORMATION ONLY. Never gates, filters or suppresses
    # a setup. Two pieces:
    #   (4B) whole-day list of the pair's HIGH-impact events, in IST.
    #   (4A) a neutral "active now" marker when now is inside an event's
    #        2h-before / 1h-after heads-up window.
    # A distinct note covers a failed fetch (so "no events" is never confused
    # with "couldn't check"). No red, no "avoid entries" — purely a heads-up.
    news_ctx = data.get('news_ctx') or {}

    def _ist(ev):
        return (ev['ts_utc'] + timedelta(hours=5, minutes=30)).strftime('%a %H:%M IST')

    if not news_ctx:
        news_banner_html = ""
    elif not news_ctx.get('coverage_ok'):
        news_banner_html = (
            '<div style="margin-top:12px;padding:10px 12px;background:#1a1a2e;'
            'border-left:3px solid #888;border-radius:4px;font-size:12px;'
            'color:#bbb;line-height:1.5;">'
            '<b style="color:#eee;">News calendar:</b> the ForexFactory fetch '
            'failed for this scan — could not load today\'s events.</div>'
        )
    else:
        day_events = news_ctx.get('day_events') or []
        # Active heads-up marker (neutral — never "avoid"). Yellow for High,
        # orange for Medium so the email visually distinguishes severity.
        if news_ctx.get('active_now') and news_ctx.get('active_event'):
            ev = news_ctx['active_event']
            is_high = ev.get('impact', '').lower() == 'high'
            marker_color = '#f1c40f' if is_high else '#e67e22'
            marker = (
                f'<div style="margin-bottom:8px;color:{marker_color};">'
                f"&#9201; Heads-up: {ev['currency']} {ev.get('impact','').upper()}-impact — "
                f"{ev.get('title','')} at {_ist(ev)} "
                f"(within {NEWS_BLACKOUT_BEFORE_H:g}h before / "
                f"{NEWS_BLACKOUT_AFTER_H:g}h after the release).</div>"
            )
        else:
            marker = ""
        # Trading-day calendar list (09:00 IST -> 09:00 IST next day), NOT a
        # calendar day. Each row tagged with its own impact level; Medium rows
        # rendered in orange, High rows in the default grey. A divider is
        # inserted before the first row that rolls into the next calendar day
        # so it can't be misread as part of "today."
        if day_events:
            first_day = (day_events[0]['ts_utc'] + timedelta(hours=5, minutes=30)).date()
            rows_parts = []
            crossed = False
            for e in day_events:
                ist_date = (e['ts_utc'] + timedelta(hours=5, minutes=30)).date()
                if not crossed and ist_date != first_day:
                    rows_parts.append(
                        '<div style="border-top:1px solid #333;margin:6px 0;'
                        'padding-top:4px;color:#888;font-size:11px;">'
                        '&#8629; rolls into next day (IST)</div>'
                    )
                    crossed = True
                rows_parts.append(
                    '<div style="color:{0};">&#8226; {1} &middot; '
                    '{2} &middot; <b>{3}</b> &middot; {4}</div>'.format(
                        '#bbb' if e.get('impact', '').lower() == 'high' else '#e67e22',
                        _ist(e), e['currency'], e.get('impact', '').upper(),
                        e.get('title', ''),
                    )
                )
            rows = "".join(rows_parts)
            day_block = (
                '<div style="color:#9ad29a;margin-bottom:4px;">Trading-day '
                'High/Medium-impact, 09:00 IST&ndash;09:00 IST (IST):</div>' + rows
            )
        else:
            day_block = (
                '<div style="color:#bbb;">No scheduled High/Medium-impact '
                'events for this pair today.</div>'
            )
        news_banner_html = (
            '<div style="margin-top:12px;padding:10px 12px;background:#0d0d1a;'
            'border-left:3px solid #888;border-radius:4px;font-size:12px;'
            'line-height:1.5;">'
            '<b style="color:#eee;">News (info only):</b>'
            f'<div style="margin-top:6px;">{marker}{day_block}</div></div>'
        )

    # Macro context block (SECONDARY — freeform AI colour on the real events
    # above). If Gemini failed, render a distinct unavailable banner so the
    # trader knows to manually check — NOT a fake "no events" summary.
    if data.get('macro_unavailable'):
        macro_html = (
            '<div style="margin-top:12px;padding:10px 12px;background:#2d1a1a;'
            'border-left:3px solid #e74c3c;border-radius:4px;font-size:12px;'
            'color:#ffb3b3;line-height:1.5;">'
            '<b style="color:#e74c3c;">&#9888; Macro Context Unavailable:</b> '
            'Gemini API failed for this scan. Check macro news manually before '
            'entering.</div>'
        )
    else:
        _ms = data.get('macro_summary') or 'N/A'
        macro_html = (
            f'<div style="margin-top:12px;padding:10px 12px;background:#0d0d1a;'
            f'border-left:3px solid #888;border-radius:4px;font-size:12px;'
            f'color:#bbb;line-height:1.5;">'
            f'<b style="color:#eee;">Macro colour (AI):</b> {_ms}</div>'
        )

    return f"""<html><body style="font-family:Arial,sans-serif;background:#0d0d1a;padding:12px;margin:0;">
    <div style="max-width:650px;margin:auto;background:#13131f;border-radius:14px;overflow:hidden;">
        <div style="background:#1a1a2e;padding:14px 18px;">
            <h2 style="color:#eee;margin:0;font-size:16px;">{state_msg}: {pair} &middot; {bias}</h2>
            <p style="color:#888;margin:4px 0 0;font-size:11px;">{ist_time}</p>
        </div>
        <div style="padding:14px 18px;">
            {setup_badge_html}
            {trend_banner_html}
            {pools_banner_html}
            {distance_html}
            {context_html}
            {break_quality_html}
            {trend_quality_html}
            {scorecard_html}
            <div style="margin:14px 0 6px 0;color:#aaa;font-size:11px;letter-spacing:1px;text-transform:uppercase;">H1 Context</div>
            {h1_chart_block}
            <div style="margin:14px 0 6px 0;color:#aaa;font-size:11px;letter-spacing:1px;text-transform:uppercase;">H1 Zoomed - Entry Zone</div>
            {m15_chart_block}
            {_chart_legend_html(bos_tag, bos_tier)}
            {sweep_breakdown_html}
            {setup_liq_html}
            {news_banner_html}
            {macro_html}
        </div>
    </div></body></html>"""


def build_sweep_breakdown_html(data, dp):
    """
    Render the Liquidity Sweep banner — SWEEP V2 (2026-07-18, pool-anchored).

    Reads the FORMATION-FROZEN ob['sweep_v2'] snapshot (liquidity_sweep.py):
    a sweep exists only when this zone's own leg took out a REAL resting-
    liquidity level (yesterday's / last week's high-low, or an equal-level
    shelf) that was untouched when the leg began. The old any-minor-swing
    grade (textbook/decent/weak, verified noise-to-inverse on the canonical
    run) is no longer narrated; its score leg is unchanged in the scorecard
    until the edge engine rules on the new columns.

    Legacy zones with no snapshot render nothing (they age out in days).
    Plain English, short — replaces the old sweep text, never adds beside it.
    """
    ob = data.get('ob') or {}
    sw = ob.get('sweep_v2')
    if not isinstance(sw, dict) or sw.get('pools_swept') is None:
        return ""  # legacy zone / layer couldn't run — nothing honest to say

    if not sw.get('exists'):
        return """
    <div style="margin-top:12px;padding:10px 12px;background:#0d0d1a;border-left:3px solid #555;border-radius:4px;font-size:12px;color:#bbb;line-height:1.6;">
        <b style="color:#eee;">Liquidity sweep:</b> <span style="color:#888;">none —
        the move behind this zone did not take out any real resting-liquidity level
        (yesterday's / last week's high-low or an equal-level shelf). No stop-run fuel.</span>
    </div>"""

    pool_word = liquidity_sweep.describe_pool(sw) or "a mapped pool"
    level = sw.get('level')
    level_str = f"{level:.{dp}f}" if level is not None else "n/a"

    # Rejection read — RAW wick:body of the sweep candle, folded into plain
    # words (same thresholds a vet eyeballs: 3x+ textbook, 2x+ strong).
    ratio = sw.get('rejection_ratio')
    if ratio is None:
        rej_label = "rejection unreadable"
    elif ratio >= 3.0:
        rej_label = f"textbook rejection — wick {ratio:.1f}x the candle body"
    elif ratio >= 2.0:
        rej_label = f"strong rejection — wick {ratio:.1f}x the candle body"
    elif ratio >= 1.0:
        rej_label = f"modest rejection — wick {ratio:.1f}x the candle body"
    else:
        rej_label = f"weak rejection — wick only {ratio:.1f}x the candle body (body dominates)"

    # Follow-through: how far price drove beyond the raided level into the
    # break that built this zone. The displacement is the proof the raid fed
    # real orders, not just a wick.
    ft = sw.get('follow_atr')
    ft_html = (f" &middot; drove {ft:.1f}&times; ATR beyond it into the break"
               if isinstance(ft, (int, float)) else "")

    rn_html = ""
    if sw.get('rn_aligned'):
        rn_html = ('<div>&#128205; <b style="color:#eee;">Round number:</b> '
                   'the raided level sits on a round number — the strongest '
                   'sweep location.</div>')

    extra = ""
    n_pools = sw.get('pools_swept') or 0
    if n_pools > 1:
        extra = (f'<div>This leg cleared {n_pools} resting-liquidity levels '
                 f'in one run — a wide stop hunt.</div>')

    return f"""
    <div style="margin-top:12px;padding:10px 12px;background:#0d0d1a;border-left:3px solid #00bcd4;border-radius:4px;font-size:12px;color:#bbb;line-height:1.6;">
        <b style="color:#eee;">Liquidity sweep:</b> the move behind this zone took out
        <b style="color:#00e5ff;">{pool_word}</b> at {level_str} &middot; {rej_label}{ft_html}
        {rn_html}{extra}
    </div>"""


def build_setup_liq_html(data):
    """Render the Setup-Liquidity banner — Reads 1 (stop-side), 2 (tp magnet),
    3.2 (leg-extreme-was-a-sweep) — from the setup_liq reads computed at alert.

    Observe-only narration: this trade's own stop and target vs resting swing
    liquidity, plus whether the OB leg ended on a sweep. Renders nothing when the
    reads are absent (legacy path / degraded). Plain English, short — one line
    per read, only the lines that have something honest to say.
    """
    reads = data.get('setup_liq_reads') or {}
    swept = data.get('leg_extreme_swept')
    lines = []
    stop_line = setup_liq.describe_stop(
        reads.get('setup_liq_stop_present'),
        reads.get('setup_liq_stop_offset_atr'),
        reads.get('setup_liq_stop_tier'))
    if stop_line:
        lines.append(f'<div>&#128737; <b style="color:#eee;">Stop:</b> {stop_line}</div>')
    tp_line = setup_liq.describe_tp(
        reads.get('setup_liq_tp_present'),
        reads.get('setup_liq_tp_offset_atr'))
    if tp_line:
        lines.append(f'<div>&#127919; <b style="color:#eee;">Target:</b> {tp_line}</div>')
    leg_line = setup_liq.describe_legextreme(swept)
    if leg_line:
        lines.append(f'<div>&#9889; <b style="color:#eee;">Leg end:</b> {leg_line}</div>')
    if not lines:
        return ""
    body = "".join(lines)
    return f"""
    <div style="margin-top:12px;padding:10px 12px;background:#0d0d1a;border-left:3px solid #7e57c2;border-radius:4px;font-size:12px;color:#bbb;line-height:1.6;">
        <b style="color:#eee;">Setup liquidity:</b>
        {body}
    </div>"""
def send_email(subject, html_body, h1_chart_b64, m15_chart_b64):
    for recipient in config["account"].get("alert_emails", []):
        msg = MIMEMultipart("related")
        msg["Subject"], msg["From"], msg["To"] = subject, GMAIL_ADDRESS, recipient
        msg.attach(MIMEText(html_body, "html"))
        if h1_chart_b64:
            img = MIMEImage(base64.b64decode(h1_chart_b64))
            img.add_header("Content-ID", "<chart_h1>")
            img.add_header("Content-Disposition", "inline", filename="chart_h1.png")
            msg.attach(img)
        if m15_chart_b64:
            img = MIMEImage(base64.b64decode(m15_chart_b64))
            img.add_header("Content-ID", "<chart_m15>")
            img.add_header("Content-Disposition", "inline", filename="chart_m15.png")
            msg.attach(img)
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(GMAIL_ADDRESS, GMAIL_PASS)
                server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
        except Exception as e:
            reason = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"Email failed: {reason}")
            _log_smtp_failure(recipient, reason)


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

HEARTBEAT_INTERVAL_HOURS = 3
HEARTBEAT_WINDOW_HOURS = 3  # Window to count recent failures for rules 2, 3, 6


def _count_recent_log_entries(log_path, window_hours, ist_now):
    """Count entries in a log file whose 'ts' is within the last `window_hours`."""
    try:
        entries = load_json(log_path, [])
        if not isinstance(entries, list):
            return 0
        cutoff = ist_now - timedelta(hours=window_hours)
        count = 0
        for e in entries:
            ts_str = e.get("ts") if isinstance(e, dict) else None
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts >= cutoff:
                    count += 1
            except Exception:
                continue
        return count
    except Exception:
        return 0


def _count_recent_by_kind(log_path, window_hours, ist_now):
    """Like _count_recent_log_entries but returns {kind: count} for entries in
    the window. Used for the P1-degrade heartbeat rule, where one log file holds
    several degrade kinds (p1_email_fail / walls_h4_error / walls_structure_error)
    and the action text names which ones fired."""
    out = {}
    try:
        entries = load_json(log_path, [])
        if not isinstance(entries, list):
            return out
        cutoff = ist_now - timedelta(hours=window_hours)
        for e in entries:
            if not isinstance(e, dict):
                continue
            ts_str = e.get("ts")
            if not ts_str:
                continue
            try:
                if datetime.fromisoformat(ts_str) >= cutoff:
                    k = e.get("kind", "unknown")
                    out[k] = out.get(k, 0) + 1
            except Exception:
                continue
        return out
    except Exception:
        return out


def _get_active_obs_mtime_hours(ist_now):
    """Return hours since active_obs.json was last modified. None if missing."""
    try:
        if not os.path.exists("active_obs.json"):
            return None
        mtime_utc = datetime.utcfromtimestamp(os.path.getmtime("active_obs.json"))
        mtime_ist = mtime_utc + timedelta(hours=5, minutes=30)
        return (ist_now - mtime_ist).total_seconds() / 3600
    except Exception:
        return None


def _is_weekday_market_hours(ist_now):
    """Rough check: weekday and not deep weekend. FX runs Mon-Fri IST."""
    # Monday=0 ... Sunday=6. Forex is closed Sat 03:30 IST through Mon 09:00 IST.
    wd = ist_now.weekday()
    if wd == 0 and ist_now.hour < 9:  # Monday before 09:00 IST — market not yet open
        return False
    if wd < 5:  # Mon-Fri
        return True
    if wd == 6 and ist_now.hour >= 4:  # Sunday after ~04:00 IST markets reopen
        return True
    return False


# Phase 1 freshness gate threshold. P1 runs hourly via cron-job.org. We give
# 75 minutes of slack — one P1 cycle (60m) + 15m grace for slow runs / commit
# lag. If active_obs.json hasn't been touched in this window during market
# hours, P2 refuses to scan and emails a one-shot "P1 stale" warning.
P1_FRESHNESS_MAX_AGE_HOURS = 1.25


def check_p1_freshness(ist_now):
    """
    Return (is_fresh, reason). Fresh means BOTH conditions hold:
      (a) active_obs.json was modified within P1_FRESHNESS_MAX_AGE_HOURS, AND
      (b) the slate's slate_date matches today's trading day.

    Either fails => stale. Phase 2 will skip scanning + emit one alert. Off
    on weekends / outside market hours so we don't spam during legitimate
    quiet periods.
    """
    if not _is_weekday_market_hours(ist_now):
        return True, "off_hours"

    age_hrs = _get_active_obs_mtime_hours(ist_now)
    if age_hrs is None:
        return False, "active_obs_missing"
    if age_hrs > P1_FRESHNESS_MAX_AGE_HOURS:
        return False, f"active_obs_age_{age_hrs:.1f}h_over_{P1_FRESHNESS_MAX_AGE_HOURS:.2f}h"

    # Slate-date check: the file may have been touched recently (e.g. by a
    # P1 run that crashed before writing the new slate) but still hold a
    # stale slate_date. Compare to today's trading day.
    try:
        raw = load_json("active_obs.json", {})
        slate_date = raw.get("slate_date") if isinstance(raw, dict) else None
    except Exception:
        slate_date = None
    today_str = get_day_id_ist(ist_now)
    if slate_date and slate_date != today_str:
        return False, f"slate_date_{slate_date}_not_{today_str}"

    return True, "ok"


def emit_p1_stale_alert(ist_now, reason):
    """
    Send a one-shot 'P1 stale' email and persist a flag so subsequent P2
    runs don't spam. Flag clears automatically on the next fresh scan.
    """
    state = load_json(os.path.join("state", "p1_stale_alert_state.json"), {})
    if not isinstance(state, dict):
        state = {}
    if state.get("alerted"):
        # Already notified for the current stale streak. Stay silent.
        print(f"  [STALE] P1 still stale ({reason}); already notified — silent.")
        return

    ts_str = ist_now.strftime("%H:%M IST, %d %b")
    subject = f"P2 PAUSED | Phase 1 data is stale | {ts_str}"
    html = f"""<html><body style="background:#131722;font-family:Arial,sans-serif;padding:20px;">
        <div style="max-width:640px;margin:auto;background:#1e222d;padding:20px;border-radius:8px;">
            <div style="color:#eee;font-size:16px;font-weight:bold;margin-bottom:14px;">
                Phase 2 paused — {ts_str}
            </div>
            <div style="padding:14px;background:#3a1b1b;border-left:4px solid #ef5350;border-radius:4px;color:#eee;font-size:14px;line-height:1.6;">
                <b style="color:#ef5350;">&#9888; Phase 1 data is stale.</b><br>
                Phase 2 refused to scan. No trade alerts will be sent until P1 recovers.<br><br>
                <b>Reason:</b> {reason}<br>
                <b>Action:</b> Check P1 Actions tab on GitHub. Investigate why the hourly run isn't producing fresh active_obs.json.
            </div>
            <div style="padding:10px 12px;margin-top:12px;background:#0d0d1a;border-left:3px solid #555;border-radius:4px;font-size:12px;color:#aaa;line-height:1.5;">
                You will receive ONE alert per stale streak. The next P2 run that sees fresh data will resume scanning silently.
            </div>
        </div>
    </body></html>"""
    try:
        send_email(subject, html, None, None)
    except Exception as e:
        print(f"  [STALE ALERT ERR] {e}")
    state["alerted"] = True
    state["since_ist"] = ist_now.isoformat()
    state["reason"] = reason
    save_json(os.path.join("state", "p1_stale_alert_state.json"), state)
    print(f"  [STALE] One-shot P1-stale alert sent: {reason}")


def clear_p1_stale_flag():
    """Called once P1 is fresh again. Resets the one-shot alert flag."""
    state = load_json(os.path.join("state", "p1_stale_alert_state.json"), {})
    if isinstance(state, dict) and state.get("alerted"):
        save_json(os.path.join("state", "p1_stale_alert_state.json"), {"alerted": False})
        print("  [STALE] P1 fresh again — stale-alert flag cleared.")


def collect_heartbeat_diagnostics(ist_now, active_obs):
    """Return dict with issues list and context fields."""
    issues = []

    # Rule 1: P1 stale during market hours
    ob_age_hrs = _get_active_obs_mtime_hours(ist_now)
    if _is_weekday_market_hours(ist_now):
        if ob_age_hrs is None:
            issues.append({
                "title": "active_obs.json is missing",
                "action": "Check P1 Actions tab for last successful run."
            })
        elif ob_age_hrs > 3:
            issues.append({
                "title": f"P1 has not updated active_obs.json in {ob_age_hrs:.1f} hours",
                "action": "Check P1 Actions tab for last successful run."
            })

    # Rule 2: Gemini failures in window
    gemini_fails = _count_recent_log_entries(
        "gemini_failure_log.json", HEARTBEAT_WINDOW_HOURS, ist_now
    )
    if gemini_fails >= 3:
        issues.append({
            "title": f"{gemini_fails} Gemini API failures in last {HEARTBEAT_WINDOW_HOURS}h",
            "action": "Check gemini_failure_log.json — likely rate limit or key issue."
        })

    # Rule 3: feed stale skips in window (log filename kept for history continuity)
    yf_stale = _count_recent_log_entries(
        "yfinance_stale_log.json", HEARTBEAT_WINDOW_HOURS, ist_now
    )
    if yf_stale >= 3:
        issues.append({
            "title": f"{yf_stale} stale/failed data fetches in last {HEARTBEAT_WINDOW_HOURS}h",
            "action": "Check yfinance_stale_log.json. If persistent, the Twelve Data feed may be rate-limiting or down."
        })

    # Rule 5: active_obs empty
    ob_count = 0
    if isinstance(active_obs, dict):
        for pair_list in active_obs.values():
            if isinstance(pair_list, list):
                ob_count += len(pair_list)
    if ob_count == 0:
        issues.append({
            "title": "active_obs.json has zero zones across all pairs",
            "action": "Verify P1 is detecting structure. Check smc_radar.log in last P1 run."
        })

    # Rule 6: Chart failures in window
    chart_fails = _count_recent_log_entries(
        "chart_failure_log.json", HEARTBEAT_WINDOW_HOURS, ist_now
    )
    if chart_fails > 5:
        issues.append({
            "title": f"{chart_fails} chart render failures in last {HEARTBEAT_WINDOW_HOURS}h",
            "action": "Check chart_failure_log.json. Charts failing silently; emails still send with banner."
        })

    # Rule 7: SMTP failures in window. Even one means an alert may have been
    # dropped silently. The heartbeat itself rides the same SMTP path, so the
    # log file is the only reliable signal of a degraded send path.
    smtp_fails = _count_recent_log_entries(
        "smtp_failure_log.json", HEARTBEAT_WINDOW_HOURS, ist_now
    )
    if smtp_fails >= 1:
        issues.append({
            "title": f"{smtp_fails} SMTP send failures in last {HEARTBEAT_WINDOW_HOURS}h",
            "action": "Check smtp_failure_log.json. Likely bad GMAIL_APP_PASSWORD or Gmail outage. Alerts may have been lost."
        })

    # Rule 8 (1B): silent Phase-1 degrades. p1_degrade_log.json is P1-owned;
    # read it the same way Rule 1 reads active_obs.json mtime. Any occurrence
    # matters — an email-send fail means a digest was lost; a walls compute
    # exception means a pair degraded to a placeholder with no structure.
    # 'dealing_range_legacy_fallback' is a diagnostic counter (Wave-1 1E), not a
    # failure — exclude it so it never triggers a false heartbeat issue.
    _P1_DIAGNOSTIC_KINDS = {"dealing_range_legacy_fallback"}
    p1_degrades = {
        k: v for k, v in _count_recent_by_kind(
            os.path.join("state", "p1_degrade_log.json"), HEARTBEAT_WINDOW_HOURS, ist_now
        ).items()
        if k not in _P1_DIAGNOSTIC_KINDS
    }
    p1_degrade_total = sum(p1_degrades.values())
    if p1_degrade_total >= 1:
        breakdown = ", ".join(f"{k}×{v}" for k, v in sorted(p1_degrades.items()))
        issues.append({
            "title": f"{p1_degrade_total} silent Phase-1 degrade(s) in last {HEARTBEAT_WINDOW_HOURS}h ({breakdown})",
            "action": "Check p1_degrade_log.json + P1 Actions tab. p1_email_fail=a digest was lost; walls_*_error=a pair degraded to a placeholder (no structure/OBs)."
        })

    return {
        "issues": issues,
        "ob_count": ob_count,
        "ob_age_hrs": ob_age_hrs,
        "gemini_fails": gemini_fails,
        "yf_stale": yf_stale,
        "chart_fails": chart_fails,
        "smtp_fails": smtp_fails,
        "p1_degrade_total": p1_degrade_total,
    }


def build_heartbeat_email_html(diag, ist_now):
    """Return (subject, html_body)."""
    issues = diag["issues"]
    ts_str = ist_now.strftime("%H:%M IST, %d %b")
    # Guard formatting against None ob_age_hrs (active_obs.json missing).
    # Previous code formatted '.1f' unconditionally and crashed the heartbeat
    # in the exact scenario where it was most needed.
    ob_age_str = (f"{diag['ob_age_hrs']:.1f}h" if diag['ob_age_hrs'] is not None
                  else "unknown")
    smtp_fails = diag.get('smtp_fails', 0)

    if not issues:
        subject = f"P2 HEARTBEAT | {ts_str} | ✅ ALL CLEAR"
        body_inner = f"""
            <div style="padding:14px;background:#1b3a1b;border-left:4px solid #26a69a;border-radius:4px;color:#eee;font-size:14px;line-height:1.6;">
                <b style="color:#26a69a;font-size:15px;">✅ All systems green.</b><br>
                Phase 2 is scanning on schedule. No errors in last {HEARTBEAT_WINDOW_HOURS}h.<br>
                Active zones across all pairs: <b>{diag['ob_count']}</b>.
            </div>
            <div style="padding:10px 12px;margin-top:12px;background:#0d0d1a;border-left:3px solid #555;border-radius:4px;font-size:12px;color:#aaa;line-height:1.5;">
                <b style="color:#ccc;">FYI:</b>
                P1 last updated active_obs.json {ob_age_str} ago.
                Gemini failures: {diag['gemini_fails']}. yfinance stale: {diag['yf_stale']}. Chart failures: {diag['chart_fails']}. SMTP failures: {smtp_fails}.
            </div>
        """
    else:
        n = len(issues)
        subject = f"P2 HEARTBEAT | {ts_str} | ⚠️ {n} ISSUE{'S' if n > 1 else ''}"
        issue_blocks = ""
        for i, iss in enumerate(issues, 1):
            issue_blocks += f"""
            <div style="padding:12px 14px;margin-bottom:10px;background:#3a1b1b;border-left:4px solid #ef5350;border-radius:4px;color:#eee;font-size:14px;line-height:1.6;">
                <b style="color:#ef5350;">⚠️ ISSUE {i}:</b> {iss['title']}<br>
                <span style="color:#ffb74d;">→ ACTION:</span> {iss['action']}
            </div>
            """
        body_inner = issue_blocks + f"""
            <div style="padding:10px 12px;margin-top:12px;background:#0d0d1a;border-left:3px solid #555;border-radius:4px;font-size:12px;color:#aaa;line-height:1.5;">
                <b style="color:#ccc;">Context:</b>
                Active zones: {diag['ob_count']}.
                P1 last update: {ob_age_str} ago.
                Gemini fails (3h): {diag['gemini_fails']}. yfinance stale (3h): {diag['yf_stale']}. Chart fails (3h): {diag['chart_fails']}. SMTP fails (3h): {smtp_fails}.
            </div>
        """

    html = f"""<html><body style="background:#131722;font-family:Arial,sans-serif;padding:20px;">
        <div style="max-width:640px;margin:auto;background:#1e222d;padding:20px;border-radius:8px;">
            <div style="color:#eee;font-size:16px;font-weight:bold;margin-bottom:14px;letter-spacing:0.5px;">
                Phase 2 Heartbeat — {ts_str}
            </div>
            {body_inner}
        </div>
    </body></html>"""
    return subject, html


def send_heartbeat_if_due(ist_now, active_obs):
    """Check timestamp gate, run diagnostics, send email if due. Never raises."""
    try:
        hb_state = load_json(os.path.join("state", "heartbeat_state.json"), {})
        last_iso = hb_state.get("last_sent_ist")
        if last_iso:
            try:
                last_dt = datetime.fromisoformat(last_iso)
                hrs_since = (ist_now - last_dt).total_seconds() / 3600
                if hrs_since < HEARTBEAT_INTERVAL_HOURS:
                    return
            except Exception:
                pass  # Corrupt timestamp -> treat as due

        diag = collect_heartbeat_diagnostics(ist_now, active_obs)
        email_sent = False
        if diag["issues"]:
            subject, html = build_heartbeat_email_html(diag, ist_now)
            send_email(subject, html, None, None)
            email_sent = True

        hb_state["last_sent_ist"] = ist_now.isoformat()
        save_json(os.path.join("state", "heartbeat_state.json"), hb_state)

        # --- Rolling diagnostic log: append snapshot of this heartbeat ---
        try:
            ob_per_pair = {}
            if isinstance(active_obs, dict):
                for pair_name, pair_list in active_obs.items():
                    ob_per_pair[pair_name] = len(pair_list) if isinstance(pair_list, list) else 0

            log_entry = {
                "ts": ist_now.isoformat(),
                "ts_human": ist_now.strftime("%H:%M IST, %d %b %Y"),
                "ob_count_total": diag["ob_count"],
                "ob_count_per_pair": ob_per_pair,
                "ob_age_hrs": diag["ob_age_hrs"],
                "gemini_fails_3h": diag["gemini_fails"],
                "yfinance_stale_3h": diag["yf_stale"],
                "chart_fails_3h": diag["chart_fails"],
                "smtp_fails_3h": diag.get("smtp_fails", 0),
                "issue_count": len(diag["issues"]),
                "issues": [{"title": i["title"], "action": i["action"]} for i in diag["issues"]],
            }
            hb_log = load_json(os.path.join("state", "heartbeat_log.json"), [])
            if not isinstance(hb_log, list):
                hb_log = []
            hb_log.append(log_entry)
            hb_log = hb_log[-50:]  # keep last 50 heartbeats (~6 days at 3h interval)
            save_json(os.path.join("state", "heartbeat_log.json"), hb_log)
        except Exception as e:
            print(f"  [HEARTBEAT LOG ERR] {e}")

        status = "Sent" if email_sent else "Silent (all clear)"
        print(f"  [HEARTBEAT] {status}. Issues: {len(diag['issues'])}. OB count: {diag['ob_count']}.")
    except Exception as e:
        print(f"  [HEARTBEAT ERR] {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# NEW
if __name__ == "__main__":
    ist_now = get_ist_now()
    scan_start_ts = ist_now  # Captured at scan start; separate from per-alert send time.
    print(f"Phase 2 Engine started {ist_now.strftime('%H:%M')} IST")

    # --- Phase 1 freshness gate -----------------------------------------
    # Phase 2 must NEVER alert on stale Phase 1 data. If P1 hasn't refreshed
    # active_obs.json within the freshness window during market hours, we
    # send one warning email and exit. The flag self-clears on recovery.
    is_fresh, fresh_reason = check_p1_freshness(ist_now)
    if not is_fresh:
        print(f"  [STALE GATE] Refusing to scan — {fresh_reason}")
        emit_p1_stale_alert(ist_now, fresh_reason)
        import sys
        sys.exit(0)
    else:
        clear_p1_stale_flag()

    # Rotate scan log first so a long-running system doesn't accumulate
    # unbounded JSONL. Keeps last 14 days.
    rotate_scan_log(ist_now)

    active_obs = load_slate_as_pair_map("active_obs.json")
    watch_state = load_json(os.path.join("state", "active_watch_state.json"), {})

    # --- active_watch_state stale-entry GC ---
    # Evict watches whose first_seen_ist (or legacy alert_ist) is older than
    # WATCH_STATE_RETENTION_DAYS. A 15-day-old zone is no longer a respectable
    # SMC setup; if price returns, P2 will re-register it fresh on next scan.
    # Anchor on first_seen rather than last_seen so a re-emailed wobbling zone
    # eventually expires regardless of refresh activity.
    WATCH_STATE_RETENTION_DAYS = 15
    watch_cutoff = ist_now - timedelta(days=WATCH_STATE_RETENTION_DAYS)
    watch_kept = {}
    watch_evicted = 0
    for k, v in (watch_state or {}).items():
        if not isinstance(v, dict):
            watch_evicted += 1
            continue
        anchor_iso = v.get("first_seen_ist") or v.get("alert_ist")
        if not anchor_iso:
            watch_kept[k] = v
            continue
        try:
            anchor_dt = datetime.fromisoformat(anchor_iso)
        except Exception:
            watch_kept[k] = v
            continue
        if anchor_dt < watch_cutoff:
            watch_evicted += 1
            continue
        watch_kept[k] = v
    if watch_evicted:
        print(f"  [WATCH GC] Evicted {watch_evicted} watches older than {WATCH_STATE_RETENTION_DAYS}d.")
        # Persist eviction immediately so an orphaned key can't sneak back via
        # the end-of-scan concurrency-safe merge.
        save_json(os.path.join("state", "active_watch_state.json"), watch_kept)
    watch_state = watch_kept
    # --- Phase 2 dedup state — lifetime model with daily re-send ---
    # Structure: { "day_id": "YYYY-MM-DD",  # informational only; no longer wipes
    #              "zones": { zone_id: {"score_int": ..., "score_raw": ...,
    #                                    "alert_ist": ..., "last_seen_ist": ...,
    #                                    "last_email_day": "YYYY-MM-DD",
    #                                    "still_valid_count": int,
    #                                    "reentry_armed": bool,
    #                                    "max_exit_distance": float,
    #                                    "breakdown": ..., ...} } }
    #
    # A zone's dedup entry lives as long as the zone itself lives in P1's
    # active_obs slate. Re-emails fire on FOUR triggers (see the scoring loop):
    #   - fresh:       first ever sighting
    #   - still_valid: new trading day, still in proximity, fires on
    #                  ALTERNATE trading days only (day 2, 4, 6... of the
    #                  zone's life — zone-relative, not calendar-relative;
    #                  the in-between day goes silent but still tracks state)
    #   - reentry:     price left proximity (>1.5x cap) and returned, any day
    #   - updated:     same-day score crosses +0.7 / -0.5
    # Same-day, in-proximity, in-band re-sightings stay silent.
    #
    # Stale-entry garbage collection: any entry whose last_seen_ist is older
    # than DEDUP_STALE_DAYS days is evicted at load time. last_seen_ist is now
    # refreshed even when a zone is OUT of proximity (as long as P1 still hands
    # it to P2), so a live-but-distant zone is never GC'd — only a zone P1 has
    # actually dropped ages out. Conservative window so a transient yfinance
    # hiccup that skips one P2 scan never evicts a live zone's dedup state.
    DEDUP_STALE_DAYS = 7
    phase2_state = load_json(os.path.join("state", "phase2_sent.json"), {"day_id": None, "zones": {}})

    # Defensive: handle legacy schema (flat zone_id -> iso) gracefully.
    if not isinstance(phase2_state, dict) or "zones" not in phase2_state:
        phase2_state = {"day_id": None, "zones": {}}

    # Stale-entry GC. An entry's freshness anchor is last_seen_ist; older
    # entries fall back to alert_ist (entries written before this field
    # existed).
    stale_cutoff = ist_now - timedelta(days=DEDUP_STALE_DAYS)
    zones_in = phase2_state.get("zones") or {}
    zones_kept = {}
    evicted = 0
    for zid, entry in zones_in.items():
        if not isinstance(entry, dict):
            evicted += 1
            continue
        anchor_iso = entry.get("last_seen_ist") or entry.get("alert_ist")
        if not anchor_iso:
            # No anchor — keep (cannot prove stale). Will get one on next match.
            zones_kept[zid] = entry
            continue
        try:
            anchor_dt = datetime.fromisoformat(anchor_iso)
        except Exception:
            zones_kept[zid] = entry
            continue
        if anchor_dt < stale_cutoff:
            # Zone died (Phase 1 dropped it; P2 stopped seeing it 7d ago).
            # Archive the full record before discarding the dedup entry.
            archive_phase2_zone(zid, entry, ist_now)
            evicted += 1
            continue
        zones_kept[zid] = entry
    if evicted:
        print(f"  [DEDUP GC] Evicted {evicted} dedup entries older than {DEDUP_STALE_DAYS}d.")
    phase2_state["zones"] = zones_kept
    phase2_state["day_id"] = get_day_id_ist(ist_now)  # informational only; no longer triggers wipe

    phase2_zones = phase2_state["zones"]

    # CONCURRENCY: track watch_state writes; merge on save so P3 deletions stick.
    watch_writes = {}
    balance = config["account"]["balance"]
    risk_pct = config["account"]["risk_percent"]
    dollar_risk = balance * (risk_pct / 100.0)
    dollar_risk_str = f"${dollar_risk:,.0f}"

    # Scheduled high-impact news calendar — fetched ONCE for the whole scan and
    # shared across all pairs (the FF feed is per-week, not per-pair). Replaces
    # the old per-pair generic-RSS scrape. Per-pair relevance is sliced from
    # this shared list in the scoring loop via get_pair_news_context.
    news_now_utc = datetime.now(timezone.utc)
    news_events, news_coverage_ok = fetch_scheduled_news(news_now_utc)
    print(f"  [NEWS] FF calendar: {len(news_events)} high-impact events in window "
          f"(coverage_ok={news_coverage_ok})")

    for pair_conf in config["pairs"]:
        # backtest_only pairs are config-present for the backtest harness but
        # MUST NOT alert live until they have earned promotion. Skip them here.
        if pair_conf.get("backtest_only"):
            continue
        symbol = pair_conf["symbol"]
        name = pair_conf["name"]
        dp = pair_conf.get("decimal_places", 5)
        entry_model = pair_conf.get("entry_model", "limit")
        pair_obs = active_obs.get(name, [])

        scan_record = {
            "ts_ist": ist_now.isoformat(),
            "pair": name,
            "zones_in_active_obs": len(pair_obs),
            "current_price": None,
            "h1_trend": None,
            "zone_outcomes": [],
            "final_action": None
        }

        if not pair_obs:
            scan_record["final_action"] = "no_zones_in_active_obs"
            append_scan_log(scan_record)
            continue

        # 30d H1 mirrors Phase 1's MAX_OB_AGE_DAYS window. If a zone is
        # still active in P1's slate, its OB candle must be locatable on
        # P2's H1 frame too — no silent fallback to "latest candle" inside
        # the scorer.
        df_h1 = fetch_with_retry(symbol, "30d", "1h")
        if df_h1 is None:
            print(f"  [SKIP] {name}: H1 data unavailable after retries")
            continue

        # Current price = last H1 close (the forming bar's most recent print
        # from yfinance — already reflects intra-bar movement). Wick-aware
        # proximity below uses the forming bar's high/low so an OB-touching
        # wick mid-hour still fires the proximity gate.
        current_price = float(df_h1['Close'].iloc[-1])
        h1_bar_high   = float(df_h1['High'].iloc[-1])
        h1_bar_low    = float(df_h1['Low'].iloc[-1])
        h1_atr = smc_detector.compute_atr(df_h1)
        if not h1_atr:
            continue

        # BOS sequence count is read from dealing_range state (single source of
        # truth). Counter resets on CHoCH (v2 has no Major/Minor); any BOS
        # (plain or Range) increments it.
        bos_counter = smc_detector.compute_bos_sequence_count(name)

        scan_record["current_price"] = current_price

        # --- PD/PW LIQUIDITY POOLS (informational, DAILY_BIAS_V4_SPEC) ---
        # Same layer Phase 1 runs: previous-day / previous-week high & low with
        # sweep/break status on this fetch's CLOSED bars (live_pool_context
        # drops the forming bar itself). Levels come from the shared per-day
        # cache (state/pool_levels.json), so this normally costs zero extra API
        # credits. Feeds the alert email + scan log ONLY — no score, no gate.
        # mark_transitions stays False: P1 owns the NEW-event marker state.
        try:
            pool_ctx = pool_builder.live_pool_context(
                name, symbol, df_h1,
                is_gold=(pair_conf.get("pair_type") == "commodity"),
            )
        except Exception as _pool_err:
            print(f"  [POOL WARN] {name}: pool context failed: {_pool_err}")
            pool_ctx = None
        scan_record["pools"] = pool_ctx

        # --- EQH/EQL equal-level clusters (informational) --- same family as
        # the pool layer: detected on this fetch's CLOSED bars, surfaces in the
        # email + scan log only. No score, no gate. live_eq_context drops the
        # forming bar itself and never raises past its own guard.
        try:
            eq_ctx = eq_pools.live_eq_context(df_h1, h1_atr)
        except Exception as _eq_err:
            print(f"  [EQ WARN] {name}: eq context failed: {_eq_err}")
            eq_ctx = None
        scan_record["eq_clusters"] = (
            [{k: c.get(k) for k in ("side", "level", "size", "status")}
             for c in eq_ctx["clusters"]] if eq_ctx else None)

        surviving_obs = []
        for ob in pair_obs:
            proximal = float(ob['proximal_line'])
            distal = float(ob['distal_line'])
            bias = "LONG" if ob['direction'] == 'bullish' else "SHORT"
            # Wick-aware proximity. Use the closest point on the forming H1
            # bar to the OB proximal:
            #   LONG  (bullish OB, price coming down): closest = bar Low
            #   SHORT (bearish OB, price coming up):   closest = bar High
            # Fallback to current_price for the other side. This catches a
            # wick that pierced toward the OB and reversed within the same
            # bar — an event the close alone would hide.
            # Clamp proximity (uniform with the backtest replay engine,
            # 2026-06-18). distance is 0 once price has REACHED or PASSED the
            # proximal -- exactly when a limit resting there would fill. The
            # old abs() inflated distance on a deep wick THROUGH the zone and
            # could suppress an alert that should fire. For a LONG the closest
            # point is the bar low (price comes down to the OB); for a SHORT it
            # is the bar high. current_price is irrelevant: it is always above
            # the low (LONG) / below the high (SHORT), so it never sets the min.
            if bias == "LONG":
                closest_to_ob = h1_bar_low
                distance = max(0.0, h1_bar_low - proximal)
            else:
                closest_to_ob = h1_bar_high
                distance = max(0.0, proximal - h1_bar_high)
            prox_cap = pair_conf["atr_multiplier"] * h1_atr

            # Structural dedup key built HERE (before the proximity gate) so the
            # out-of-proximity branch can keep the zone's dedup entry alive and
            # arm the re-entry trigger. Stashed on the ob for the scoring loop.
            # Key uses BOS swing price (stable across scans) + ob_timestamp
            # hour-bucket; see the scoring loop's original derivation.
            bos_swing_px = float(ob.get('bos_swing_price', proximal))
            bos_tag = ob.get('bos_tag', 'BOS')
            key_dp = max(0, dp - 1)
            ob_ts_iso = ob.get('ob_timestamp') or ''
            ts_bucket = ob_ts_iso[:13] if len(ob_ts_iso) >= 13 else ob_ts_iso
            zone_id = f"{name}_{bias}_{bos_tag}_{round(bos_swing_px, key_dp)}_{ts_bucket}"
            ob['_zone_id'] = zone_id

            zone_outcome = {
                "direction": ob['direction'],
                "proximal": proximal,
                # `distal` added for backtest/live parity (Tier-B OB identity).
                # Data already in scope (line ~1920); additive log field only.
                "distal": round(distal, dp),
                "ob_timestamp": ob.get('ob_timestamp'),
                "closest_price": round(closest_to_ob, dp),
                "current_price": round(current_price, dp),
                "distance": round(distance, dp),
                "proximity_cap": round(prox_cap, dp),
                "result": None
            }

            # 1. Proximity gate (wick-aware)
            if distance > prox_cap:
                zone_outcome["result"] = "dropped_proximity"
                scan_record["zone_outcomes"].append(zone_outcome)
                # Out-of-proximity does NOT delete the zone (Phase 1 owns
                # deletion). Keep the dedup entry alive (refresh last_seen so
                # GC never evicts a still-live zone) and track the furthest
                # departure. Once price pulls beyond REENTRY_EXIT_MULT x the
                # cap, arm the re-entry trigger so a genuine return re-emails.
                prior_oop = phase2_zones.get(zone_id)
                if prior_oop is not None:
                    prior_oop["last_seen_ist"] = ist_now.isoformat()
                    prior_oop["max_exit_distance"] = max(
                        float(prior_oop.get("max_exit_distance", 0.0)), distance
                    )
                    if prior_oop["max_exit_distance"] >= REENTRY_EXIT_MULT * prox_cap:
                        prior_oop["reentry_armed"] = True
                continue

            # 2. OB still-active gate. Phase 1 owns the canonical drop decision,
            # but P1 only runs hourly. Between P1 cycles, price can invalidate
            # the zone. We replay candles on P2's H1 frame using the SAME
            # mitigation rule AND the SAME window as Phase 1.
            #
            # Window = from the candle AFTER the structural-event (BOS/CHoCH)
            # candle, NOT after the OB candle. The OB becomes a live zone only
            # once displacement breaks structure; the candles between the OB and
            # the break are the impulse leg that BUILT the zone, not tests of it.
            # Phase 1 (detect_smc_radar mitigation + determine_drop_reason) both
            # start at event-candle + 1; starting at OB + 1 here let the impulse
            # leg count phantom touches / distal hits and silently disagree with
            # Phase 1. `bos_timestamp` holds the event candle ts (BOS or CHoCH).
            ob_ts_iso_gate  = ob.get('ob_timestamp')
            bos_ts_iso_gate = ob.get('bos_timestamp')
            # Prefer the event (BOS/CHoCH) candle for the window start; fall back
            # to the OB candle only if the event ts is missing (legacy zone).
            gate_anchor_ts = bos_ts_iso_gate or ob_ts_iso_gate
            if gate_anchor_ts:
                anchor_idx_gate, on_chart_gate = smc_detector.locate_ob_candle_idx(
                    df_h1, gate_anchor_ts
                )
                if on_chart_gate:
                    mitigated, mit_reason, _touches = smc_detector.is_ob_mitigated_phase1(
                        ob['direction'], distal, proximal, df_h1,
                        start_idx=anchor_idx_gate + 1,
                        distal_mode=smc_detector.resolve_distal_mode(pair_conf),
                        atr=h1_atr,
                    )
                    if mitigated:
                        zone_outcome["result"] = f"dropped_invalidated_{mit_reason}"
                        scan_record["zone_outcomes"].append(zone_outcome)
                        continue
                else:
                    # Event candle older than 30d H1 fetch — should be vanishingly
                    # rare given P1's MAX_OB_AGE_DAYS guard. Skip the alert rather
                    # than score against a zone we can't locate.
                    zone_outcome["result"] = "dropped_ob_off_chart"
                    scan_record["zone_outcomes"].append(zone_outcome)
                    continue

            zone_outcome["result"] = "passed_to_trend_gate"
            scan_record["zone_outcomes"].append(zone_outcome)
            surviving_obs.append(ob)
        if not surviving_obs:
            scan_record["final_action"] = "no_zones_passed_proximity"
            append_scan_log(scan_record)
            continue

        # H1 trend: information-only banner (no gating). Trader decides
        # with-trend vs counter-trend at execution time.
        current_trend = bos_counter.get('trend')  # 'bullish' | 'bearish' | None
        scan_record["h1_trend"] = current_trend

        # All surviving zones get their own alert (H1-only migration, 2026-05-26).
        # Pre-migration code picked nearest-only to dampen M15-noise spam; with
        # H1-only entries, every legitimate zone deserves visibility. Score gate
        # also removed — hysteresis dedup still suppresses redundant re-emails.
        for ob in surviving_obs:
            # BOS sequence count is PER-OB. Phase 1 stamps each OB with the BOS
            # count AT ITS OWN EVENT and persists it on the slate. OB2 belongs to
            # an EARLIER event than OB1, so it must keep its own (lower) count —
            # NOT today's whole-ring total. We therefore READ the persisted value
            # and only fall back to the live ring count for legacy slate zones
            # written before this field existed (they age out within ~15 days).
            if ob.get('bos_sequence_count') is None:
                ob['bos_sequence_count'] = bos_counter['count']  # legacy fallback
            ob['bos_count_maxed'] = bool(bos_counter.get('count_maxed', False))
            # Continuation-drive verdict ('holding' | 'fading') describes the
            # CURRENT trend leg's displacement decay — a property of live
            # structure, not of the OB's own (possibly older) event. It applies to
            # every surviving OB: when price returns to OB2, what matters is
            # whether the trend is still driving NOW. Drives the structure quality
            # score (count does not). Defaults 'holding' for legacy slate zones.
            ob['bos_verdict'] = bos_counter.get('verdict', 'holding')

            zone_dir = ob.get('direction')
            # Pending-reversal context (Fix A, 2026-06-30). A CHoCH ARMS a flip but
            # does NOT flip `current_trend` until a Confirmation BOS prints, so the
            # raw `trend` field still reads the OLD direction while the reversal is
            # contested. Without this branch a zone in the OLD direction was tagged
            # "WITH H1 trend" off that stale field (the EURUSD 2026-06-29 bug: a
            # SHORT labelled with-trend while a bullish CHoCH was pending). The
            # banner is information-only; the trader decides.
            flip_pending     = bos_counter.get('flip_pending', False)
            flip_pending_dir = bos_counter.get('flip_pending_dir')  # 'bullish'|'bearish'|None
            _choch_ts        = bos_counter.get('choch_ts')
            _choch_day = ""
            if _choch_ts:
                try:
                    _choch_day = " (" + str(_choch_ts)[:10] + ")"
                except Exception:
                    _choch_day = ""
            # Alignment VALUE comes from the shared helper (one implementation,
            # backtest calls the same one). The label TEXT is presentation and
            # stays local to each branch below.
            trend_alignment = smc_detector.derive_trend_alignment(
                zone_dir, current_trend, flip_pending, flip_pending_dir)
            if flip_pending and zone_dir != flip_pending_dir:
                # Zone is in the OLD trend direction while a reversal AGAINST it is
                # pending. This is NOT with-trend — it is counter to the forming
                # reversal. Drop the with-trend badge and warn.
                trend_label = (f"AGAINST pending H1 reversal — {flip_pending_dir} "
                               f"CHoCH printed{_choch_day}, Confirmation BOS pending "
                               f"(zone is {zone_dir})")
            elif flip_pending:
                # Zone direction matches the pending reversal. The trend is contested
                # and UNCONFIRMED, so do not credit it as with-trend (no score
                # inflation on an unconfirmed flip) — surface it as forming.
                trend_label = (f"Pending H1 reversal in zone's favour — {flip_pending_dir} "
                               f"CHoCH printed{_choch_day}, Confirmation BOS pending")
            elif current_trend is None:
                trend_label = "H1 trend ambiguous — no clear BOS sequence"
            elif current_trend == zone_dir:
                trend_label = f"WITH H1 trend (H1 is {current_trend})"
            else:
                trend_label = f"AGAINST H1 trend (H1 is {current_trend}, zone is {zone_dir})"

            scan_record["trend_alignment"] = trend_alignment

            proximal = float(ob['proximal_line'])
            bias = "LONG" if ob['direction'] == 'bullish' else "SHORT"

            # FVG — H1 only. Inherited from Phase 1's frozen snapshot
            # (radar detected with the OB→OB+7 H1 window). P2 does NOT redetect.
            # Fallback lives in ONE place (fvg_missing) — parity with backtest.
            fvg_h1 = ob.get("fvg", smc_detector.fvg_missing())
            fvg_data = {"h1": fvg_h1}
            fvg_source = "H1" if fvg_h1.get('exists') else None

            # Score FIRST. Gemini macro is display-only — don't pay for the
            # API call until we know the zone passes level validity.
            score_res = smc_detector.run_scorecard(
                bias, df_h1, ob, fvg_data, current_price, pair_conf
            )

            levels = smc_detector.compute_phase2_levels(pair_conf, bias, ob, current_price, df_h1)
            if not levels['valid']:
                scan_record["final_action"] = "levels_invalid"
                append_scan_log(scan_record)
                continue

            # SETUP-LIQ Reads 1 & 2 (setup_liq / SWING_SWEEP_SPEC) — this trade's
            # own STOP-side liquidity and TP-side magnet vs resting swings.
            # Anchored on the SL / TP1 just computed (levels['sl'] already carries
            # the OB-distal +/- spread — the live traded stop). Same df_h1 the
            # levels used (compute_phase2_levels drops the forming bar internally
            # -> look-ahead-safe); ATR = ob['h1_atr'] (shared *_atr denominator).
            # A 1:1-fallback TP1 sits on no pool -> magnet absent by construction.
            # Observe-only; never raises (all-None on failure). Backtest computes
            # the identical reads at the level-calc step (h1_only_simulator).
            setup_liq_reads = setup_liq.reads_stop_and_tp(
                df_h1, ob.get("direction"),
                levels.get("sl"), levels.get("tp1"), ob.get("h1_atr"),
                pair_conf.get("pair_type", "forex"),
                tp1_is_fallback=bool(levels.get("no_liquidity_pool_fallback")),
            )

            # Setup classifier — recognise named textbook patterns (badge + note).
            # Reads only already-computed fields; pd_position from the scorecard.
            setup_name, setup_note, setup_kind = smc_detector.classify_setup(
                ob, score_res.get('pd_position'), trend_alignment
            )
            # Log the badge so a future "badged vs unbadged outcome" study has the
            # label from day one (Fable's process note).
            scan_record["setup_badge"] = setup_name

            # Record the fired-alert trade levels for backtest/live parity
            # (Tier-B). Keyed by zone_id so the extractor can join to the
            # zone_outcome above. Additive log only — no decision/branch change.
            scan_record.setdefault("fired_levels", {})[ob.get('_zone_id', zone_id)] = {
                "entry": levels.get("entry"),
                "sl":    levels.get("sl"),
                "tp1":   levels.get("tp1"),
                "tp2":   levels.get("tp2"),
                "rr":    levels.get("rr"),
            }

            # Score + levels passed. NOW build news context — only spent on
            # zones that will actually email. The deterministic scheduled-event
            # context (blackout + next event) comes from the shared FF calendar;
            # Gemini gets those same real events to summarise (not generic RSS).
            news_ctx = get_pair_news_context(
                name, news_events, news_coverage_ok, news_now_utc
            )
            gemini_risk = call_gemini_flash(name, bias, fetch_macro_news(name, news_ctx))

            # B2: Pass fvg_source, dealing_range, pd_position, plus new sweep
            # tier + components + age into scorecard rows for richer narration.
            scorecard_rows = smc_detector.generate_scorecard_rows(
                bias, score_res['breakdown'], ob,
                score_res.get('sweep_price'), score_res.get('sweep_tf', 'H1'), pair_conf,
                score_res.get('dealing_range'), fvg_source, score_res.get('pd_position'),
                sweep_tier=score_res.get('sweep_tier'),
                sweep_components=score_res.get('sweep_components'),
                fvg=fvg_data
            )
            # (Legacy ob['sweep_tf']/ob['sweep_timestamp'] chart injection
            # removed 2026-07-19 — the chart overlay now reads the frozen
            # ob['sweep_v2'] snapshot directly, same source as the emails.)

            distance = abs(current_price - proximal)
            pip = pip_size(pair_conf)
            distance_pips_num = round(distance / pip, 1)
            distance_str = f"{distance_pips_num} {pip_unit_label(pair_conf)}"
            atr_label = atr_distance_label(distance, h1_atr, "H1")

            # PD/PW pool context for THIS trade's direction (informational).
            # Snapshot line + trade-relative features (nearest unspent pool,
            # toward/against). None-safe: a dead pool layer just drops the line.
            pool_email = None
            if pool_ctx:
                try:
                    pool_email = {
                        "line": pool_builder.format_pool_line(pool_ctx, dp),
                        "features": pool_builder.trade_features(
                            pool_ctx, ref_price=current_price, atr=h1_atr,
                            direction=ob.get("direction"),
                            h1_index=pool_builder.drop_forming(df_h1).index),
                    }
                except Exception as _pf_err:
                    print(f"  [POOL WARN] {name}: trade features failed: {_pf_err}")

            # EQH/EQL context for THIS trade (informational). One digest line
            # plus the stop-placement warning when the trade's stop sits in
            # front of a resting equal-level pool (eq_sl_at_risk geometry).
            # None-safe: a dead EQ layer just drops the line.
            eq_email = None
            if eq_ctx:
                try:
                    _eq_feats = eq_pools.features_from_clusters(
                        eq_ctx["clusters"], ref_price=current_price,
                        sl=levels.get("sl"), direction=ob.get("direction"),
                        atr=h1_atr, asof_pos=eq_ctx["asof_pos"])
                    eq_email = {
                        "line": eq_pools.format_eq_line(
                            eq_ctx, current_price, h1_atr),
                        "warning": eq_pools.format_eq_sl_warning(_eq_feats),
                        "features": _eq_feats,
                    }
                except Exception as _eqf_err:
                    print(f"  [EQ WARN] {name}: eq trade features failed: {_eqf_err}")

            # WEEKLY PD ZONE (higher-timeframe premium/discount, informational).
            # Compares the weekly PD read (price vs last completed week's high/
            # low, from the SAME PWH/PWL the pool banner shows — pool_ctx carries
            # the cached levels) against the H4 PD position the scorecard already
            # computed. Produces one plain-English agreement line naming BOTH
            # timeframes and BOTH percentages. None-safe: a dead layer / missing
            # read just drops the line. No score, no gate.
            weekly_pd_email = None
            try:
                _h4_pd = score_res.get('pd_position')  # H4 read, 0-1 scale
                _wk_levels = (pool_ctx.get("levels") if pool_ctx else None)
                _wk_feats = weekly_pd.live_features(
                    df_h1, current_price, _h4_pd, weekly_levels=_wk_levels)
                _wk_line = weekly_pd.format_agreement_line(
                    _wk_feats, _h4_pd, bias)
                if _wk_line:
                    weekly_pd_email = {"line": _wk_line, "features": _wk_feats}
            except Exception as _wk_err:
                print(f"  [WEEKLY_PD WARN] {name}: weekly PD read failed: {_wk_err}")

            # ── APPROACH QUALITY (scan-time entry mechanics, RETRACE_QUALITY_SPEC)
            # The live twin of the backtest's fill-time approach columns: as of
            # now, how did price travel into the zone over the last closed H1
            # bars (speed toward zone in formation-ATR, candle body share, ER).
            # Closed bars only (drop the forming bar, same convention the pool /
            # chop reads above use). ATR = ob['h1_atr'] (the frozen OB-formation
            # ATR, matching every backtest *_atr column). Observe-only: NO gate,
            # NO score, NO verdict — one plain email line. Never raises; an
            # all-None dict renders em-dashes. Computed ONCE here, shared by both
            # the email payload and the phase2_zones stamp below.
            _approach_closed = pool_builder.drop_forming(df_h1)
            approach = approach_quality.features_now(
                _approach_closed,
                direction=ob.get("direction"),
                atr=ob.get("h1_atr"),
            )
            approach_asof_utc = (
                _approach_closed.index[-1].isoformat()
                if _approach_closed is not None and len(_approach_closed) else None
            )

            # DISPLACEMENT_LEG_BUILD_SPEC — structural displacement-leg extreme +
            # Kaufman ER for THIS OB, via the SAME shared core the backtest uses
            # (displacement_leg.compute_leg_extreme_er). Parity is real for the
            # first time: live previously did not stamp leg_extreme at all.
            # Observe-only: NO gate, NO score, NO verdict. The core takes
            # TIMESTAMPS (ob_timestamp/bos_timestamp) + the persisted swing list
            # and re-resolves ts->idx internally, because the OB/swing indices
            # loaded from Phase-1 state are STALE against the live H1 frame
            # (spec §1.3). Closed bars only (drop the forming bar, same as the
            # approach read). Never raises; returns None on any failure.
            #   *_at_alert here matches the backtest column name; the phase2_zones
            #   re-stamp below carries the *_at_email twin (same value, re-anchored
            #   on re-arm — trader-approved for these two columns only, spec §5).
            _leg_swings = []
            try:
                import dealing_range as _dr_leg
                _leg_state = _dr_leg.load_state() or {}
                _leg_swings = smc_detector.swings_for_chart(_leg_state.get(name) or {})
            except Exception:
                _leg_swings = []
            leg_extreme_at_alert, leg_er_at_alert, _leg_extreme_end_idx = \
                displacement_leg.compute_leg_extreme_er(
                    _approach_closed,               # closed bars only (forming dropped)
                    ob.get("ob_timestamp"),
                    ob.get("bos_timestamp"),
                    ob.get("direction"),
                    _leg_swings,
                )

            # Read 3.2 (setup_liq): was the leg's OWN terminal extreme itself a
            # sweep of an active swing (the leg ENDED on a stop-run, then
            # reversed)? Same closed frame + the extreme_end_idx the shared core
            # just resolved -> shares the leg core's point-in-time picture, no
            # re-detection. Observe-only; None when un-measurable. Backtest
            # computes the identical value at the replay yield (replay_engine).
            leg_extreme_swept = setup_liq.read_legextreme_swept(
                _approach_closed, leg_extreme_at_alert, _leg_extreme_end_idx,
                ob.get("direction"), pair_conf.get("pair_type", "forex"),
                ob.get("h1_atr"),
            )

            trade_data = {
                "pair": name,
                "bias": bias,
                "score": score_res['total'],
                "breakdown": score_res['breakdown'],
                # Legacy sweep record fields — SCORE-side facts only (the sweep
                # leg still feeds the score, so what it saw stays logged).
                # sweep_hours_before_ob dropped 2026-07-19 with the old banner
                # (write-only after the sweep-v2 narration replaced it).
                "sweep_price": score_res.get('sweep_price'),
                "sweep_tf": score_res.get('sweep_tf', 'H1'),
                "sweep_tier": score_res.get('sweep_tier'),
                "sweep_components": score_res.get('sweep_components'),
                "macro_summary": gemini_risk.get("macro_summary"),
                "macro_unavailable": bool(gemini_risk.get("macro_unavailable", False)),
                "news_ctx": news_ctx,
                "setup_badge": ({"name": setup_name, "note": setup_note, "kind": setup_kind}
                                if setup_name else None),
                "levels": levels,
                "ob": ob,
                # DISPLACEMENT_LEG_BUILD_SPEC — displacement-leg extreme + Kaufman
                # ER at this alert (observe-only). *_at_alert name matches the
                # backtest column; the phase2_zones re-stamp carries the *_at_email
                # twin. None when un-measurable (OB predates frame / degenerate).
                "leg_extreme_at_alert": leg_extreme_at_alert,
                "leg_er_at_alert": leg_er_at_alert,
                # SETUP-LIQ (setup_liq / SWING_SWEEP_SPEC) — Reads 1 (stop-side),
                # 2 (tp magnet), 3.2 (leg-extreme-was-a-sweep). Observe-only; the
                # email narrates these (describe_stop/describe_tp/describe_legextreme).
                # Backtest logs the identical values as setup_liq_* columns.
                "setup_liq_reads": setup_liq_reads,
                "leg_extreme_swept": leg_extreme_swept,
                # Choppiness Index on the alert's server trading day — daily
                # trend-vs-range regime as of THIS alert. Anchored at the last
                # CLOSED H1 bar (drop the forming bar), grouped by the same
                # server day PDH/PDL use. Observe-only, gates nothing; None when
                # un-measurable. Backtest computes the identical value at
                # alert_ts (h1_only_simulator). See compute_choppiness_index.
                "chop_at_alert": smc_detector.compute_choppiness_index(
                    df_h1, pool_builder.drop_forming(df_h1).index[-1]
                    if len(pool_builder.drop_forming(df_h1)) else None),
                "current_price": current_price,
                "distance_to_proximal": distance,
                "h1_atr": h1_atr,
                "alert_ist": ist_now.isoformat(),
                "scorecard_version": "v2",
                "trend_alignment": trend_alignment,
                "trend_label": trend_label,
                "h1_trend": current_trend,
                # PD/PW pools — email banner + dedup-state breadcrumb only.
                "pools": pool_email,
                # EQH/EQL clusters — email banner line + stop warning only.
                "eq": eq_email,
                # Weekly PD zone — HTF premium/discount agreement line only.
                "weekly_pd": weekly_pd_email,
                # Approach-into-zone shape (RETRACE_QUALITY_SPEC) — one info line,
                # no verdict. The 3 values + asof travel to the email builder here.
                "approach": approach,
                "approach_asof_utc": approach_asof_utc,
            }
            dr = score_res.get('dealing_range')

            # Unified H1-limit alert path (post-2026-05-26 migration). Every
            # pair is `entry_model == "limit"`; the legacy `ltf_choch` branch
            # (Gold/NAS M5 CHoCH route) is retired.
            #
            # Structural dedup key was built (and stashed on the ob) in the
            # proximity loop above so the out-of-proximity branch could share it.
            zone_id = ob['_zone_id']

            # Re-email triggers (in priority order):
            #   1. fresh       — never emailed → TRADE READY
            #   2. reentry     — left proximity (>1.5x cap) and returned, any
            #                    day → TRADE READY (plain; price has come back)
            #   3. still_valid — new trading day, still in proximity, ONLY on
            #                    the 11:35 IST scan, and ONLY on alternate
            #                    trading days (zone-relative) → TRADE READY
            #                    (STILL VALID)
            #   - silent       — same trading day, already emailed (intraday
            #                    score moves NEVER re-email — one email per day)
            current_score_raw = float(score_res['total'])
            current_breakdown = score_res.get('breakdown', {})

            # Score floor (2026-06-18). Zones scoring below MIN_SCORE_TO_EMAIL
            # never email in ANY path (fresh, still_valid, reentry).
            # This is the only quality gate in Phase 2; raw 3.8 floors to "3"
            # in the UI so it must be blocked — gate on raw < floor.
            if current_score_raw < MIN_SCORE_TO_EMAIL:
                scan_record["final_action"] = (
                    f"below_score_floor "
                    f"({current_score_raw:.1f} < {MIN_SCORE_TO_EMAIL:.1f})"
                )
                append_scan_log(scan_record)
                continue

            prior = phase2_zones.get(zone_id)
            today_id = get_day_id_ist(ist_now)
            email_kind = "fresh"            # fresh | reentry | still_valid
            prior_score_raw = None
            prior_breakdown = None
            prior_alert_ist = None
            hi_water = current_score_raw
            lo_water = current_score_raw
            if prior is not None:
                prior_score_raw = float(prior.get("score_raw", 0.0))
                prior_breakdown = prior.get("breakdown")
                prior_alert_ist = prior.get("alert_ist")
                prior_hi = float(prior.get("score_high_water", prior_score_raw))
                prior_lo = float(prior.get("score_low_water",  prior_score_raw))
                hi_water = max(prior_hi, current_score_raw)
                lo_water = min(prior_lo, current_score_raw)

                if prior.get("reentry_armed"):
                    # Price genuinely left the zone and came back — fresh approach.
                    email_kind = "reentry"
                elif prior.get("last_email_day") != today_id:
                    # New trading day. Compute how many days have elapsed since
                    # the last email. Gap of exactly 1 day = still_valid reminder.
                    # Gap of 2+ days = treat as fresh (wipe prior context so the
                    # zone re-emails with no score band / watermark history).
                    try:
                        last_day_dt = datetime.strptime(prior["last_email_day"], "%Y-%m-%d")
                        today_dt    = datetime.strptime(today_id, "%Y-%m-%d")
                        day_gap = (today_dt - last_day_dt).days
                    except Exception:
                        day_gap = 1  # parse failure → treat as 1-day gap (safe default)

                    if day_gap >= 2:
                        # Zone went quiet for 2+ days — treat as a brand-new sighting.
                        # Wipe prior context: score band, watermarks, first-seen time.
                        prior = None
                        prior_score_raw = None
                        prior_breakdown = None
                        prior_alert_ist = None
                        hi_water = current_score_raw
                        lo_water = current_score_raw
                        email_kind = "fresh"
                    else:
                        # Exactly 1 new trading day — still valid reminder.
                        # Gate (2026-07-02): still-valid reminders fire ONLY on
                        # the 11:35 IST scan, never the 09:35 scan. The engine
                        # runs 3x/day (09:35, 11:35, 19:35 IST); the 11:35 run
                        # lands in the 11:xx hour. If this is any other scan, go
                        # silent WITHOUT stamping last_email_day, so the 11:35
                        # scan still re-detects and emails the zone.
                        if ist_now.hour != 11:
                            prior["last_seen_ist"] = ist_now.isoformat()
                            save_json(os.path.join("state", "phase2_sent.json"), phase2_state)
                            scan_record["final_action"] = (
                                f"still_valid_deferred_to_1135 "
                                f"(scan {ist_now.strftime('%H:%M')} IST)"
                            )
                            append_scan_log(scan_record)
                            continue

                        # Alternate-day throttle (zone-relative, not calendar).
                        # still_valid_count tracks how many new trading days
                        # have passed since the zone was last emailed (fresh
                        # or still_valid), incrementing every day regardless
                        # of whether this day's email actually sends. Odd
                        # count (1, 3, 5...) = send; even count = stay silent
                        # but persist the day so tomorrow's count is correct.
                        prior_count = int(prior.get("still_valid_count", 0))
                        this_count = prior_count + 1
                        if this_count % 2 == 0:
                            prior["still_valid_count"] = this_count
                            prior["last_email_day"] = today_id
                            prior["last_seen_ist"] = ist_now.isoformat()
                            save_json(os.path.join("state", "phase2_sent.json"), phase2_state)
                            scan_record["final_action"] = (
                                f"still_valid_alternate_day_silent "
                                f"(day {this_count} of zone life)"
                            )
                            append_scan_log(scan_record)
                            continue

                        email_kind = "still_valid"
                else:
                    # Same trading day, already emailed → always silent.
                    # Score-change ("UPDATED") re-emails are retired (2026-07-02):
                    # a zone is emailed ONCE per day; intraday score moves no
                    # longer re-email. Watermarks are still persisted for
                    # diagnostics so the score history survives.
                    prior["score_high_water"] = hi_water
                    prior["score_low_water"]  = lo_water
                    # Refresh last_seen_ist so stale-entry GC doesn't evict
                    # a live silent zone.
                    prior["last_seen_ist"] = ist_now.isoformat()
                    save_json(os.path.join("state", "phase2_sent.json"), phase2_state)
                    scan_record["final_action"] = (
                        f"dedup_same_day_silent "
                        f"({prior_score_raw:.1f}->{current_score_raw:.1f})"
                    )
                    append_scan_log(scan_record)
                    continue

            # Register zone state in-memory. Persisted AFTER send_email so
            # a crash mid-send re-attempts the alert next scan (duplicate
            # is recoverable; silent loss is not). Stores breakdown so
            # future updates can show driver deltas. Watermarks reset to
            # the current score at email time.
            phase2_zones[zone_id] = {
                "score_int": score_to_int(current_score_raw),
                "score_raw": current_score_raw,
                "score_high_water": current_score_raw,
                "score_low_water":  current_score_raw,
                "breakdown": current_breakdown,
                # First-alert time is preserved across re-emails so STILL VALID
                # carryovers retain when the zone was originally flagged.
                "alert_ist": prior_alert_ist or ist_now.isoformat(),
                "last_seen_ist": ist_now.isoformat(),
                # Daily-reset + re-entry tracking. Stamping last_email_day=today
                # suppresses a second daily reminder; clearing the re-entry flag
                # and exit watermark re-arms the flicker guard from zero.
                "last_email_day": today_id,
                # Zone-relative day counter driving the still_valid alternate-
                # day throttle. Carries forward on a still_valid send; resets
                # to 0 on fresh/reentry since those start a new approach.
                "still_valid_count": this_count if email_kind == "still_valid" else 0,
                "reentry_armed": False,
                "max_exit_distance": 0.0,
                "bias": bias,
                "pair": name,
                # Full OB snapshot captured AT ALERT TIME (geometry + FVG band +
                # sweep + dealing range). Phase 1 deletes dropped zones same-day,
                # so this is the only place the full record survives long enough
                # to be archived when the zone later dies (see GC eviction).
                "ob_snapshot": ob,
                # Approach-into-zone shape at this email (RETRACE_QUALITY_SPEC §4).
                # MUTABLE BY DESIGN — re-stamped on every email (fresh/reentry/
                # still_valid) because it is a rolling scan-time observation, not
                # frozen OB state; approach_asof_utc disambiguates. archive_
                # phase2_zone carries this dict to phase2_history when the zone
                # dies, so backtest-comparable history comes free.
                "approach_speed_atr_at_email": approach["approach_speed_atr_at_fill"],
                "approach_body_ratio_at_email": approach["approach_body_ratio_at_fill"],
                "approach_er_at_email": approach["approach_er_at_fill"],
                # DISPLACEMENT_LEG_BUILD_SPEC — displacement-leg extreme + ER
                # re-stamped on every email (fresh/reentry/still_valid). This is
                # the trader-approved re-anchor for THESE TWO COLUMNS ONLY (spec
                # §5): a leg still running at first alert, or a top that has since
                # confirmed, gets its true extreme now. Same value shape as the
                # *_at_alert payload above, different stamp name (the *_at_email
                # convention the approach twins use). approach_asof_utc dates it.
                "leg_extreme_at_email": leg_extreme_at_alert,
                "leg_er_at_email": leg_er_at_alert,
                "approach_asof_utc": approach_asof_utc,
            }
            # H1 wide context + H1 zoomed entry. The "m15_chart" variable name
            # is preserved through the email plumbing for now (CID = chart_m15,
            # MIME slot 2) -- the bytes it carries are the zoomed H1 chart.
            # Renaming the CID would invalidate any cached email templates.
            h1_chart = generate_h1_chart(df_h1, ob, pair_conf,
                                         f"{name} H1 - {bias} zone context", levels, dr,
                                         pools=pool_ctx, eq_ctx=eq_ctx)
            m15_chart = generate_h1_zoomed_chart(
                df_h1, ob, pair_conf,
                f"{name} H1 zoomed - entry zone", levels
            )
            h1_ok = h1_chart is not None
            m15_ok = m15_chart is not None
            if not h1_ok:
                _log_chart_failure(name, "h1_phase2_limit")
            if not m15_ok:
                _log_chart_failure(name, "h1_zoomed_phase2_limit")

            if email_kind == "still_valid":
                # Next-day reminder for a zone still sitting in proximity.
                subject_prefix = "TRADE READY (STILL VALID)"
                email_label = "TRADE READY — still valid"
                log_action = "alert_sent_TRADE_READY_STILL_VALID"
                print_label = f"TRADE READY STILL VALID: {name}"
            elif email_kind == "reentry":
                # Price left proximity and came back — plain TRADE READY.
                subject_prefix = "TRADE READY"
                email_label = "TRADE READY"
                log_action = "alert_sent_TRADE_READY_REENTRY"
                print_label = f"TRADE READY RE-ENTRY: {name}"
            else:  # fresh
                subject_prefix = "TRADE READY"
                email_label = "TRADE READY"
                log_action = "alert_sent_TRADE_READY"
                print_label = f"TRADE READY: {name}"

            html = build_trade_email(
                trade_data, name, pair_conf, email_label,
                scorecard_rows, score_res['total'],
                atr_label, distance_str, dollar_risk_str, scan_start_ts,
                h1_chart_ok=h1_ok, m15_chart_ok=m15_ok
            )
            # Subject shows the /10-normalized score so forex and gold/NAS/JPY
            # alerts are comparable at a glance. Real math lives in the body.
            # A named setup (if any) goes right after the prefix so the verdict
            # is visible in the inbox without opening the email; the name is
            # self-describing, so no symbols are needed.
            _subj_score = normalized_score(score_res['total'], pair_conf)
            _setup_subj = trade_data.get("setup_badge")
            _badge_subj = (f"{_setup_subj['name']} | "
                           if _setup_subj and _setup_subj.get("name") else "")
            send_email(
                f"{subject_prefix} | {_badge_subj}{name} | {bias} | "
                f"Score {_subj_score:.1f}/10 | {ist_now.strftime('%H:%M IST')}",
                html, h1_chart, m15_chart
            )
            # Persist dedup state AFTER send. See comment above.
            save_json(os.path.join("state", "phase2_sent.json"), phase2_state)
            print(f"  [OK] {print_label}")
            scan_record["final_action"] = log_action
            append_scan_log(scan_record)

    save_json(os.path.join("state", "phase2_sent.json"), phase2_state)

    # CONCURRENCY-SAFE SAVE: re-read latest disk state, apply only our upserts.
    # If P3 deleted keys mid-run, those deletions are preserved.
    #
    # Sticky fields preserved across upserts: a small set of fields that
    # Phase 3 owns and Phase 2 must never wipe. Currently: 'tapped',
    # 'tapped_ist'. Without this carry-over, every P2 hourly upsert would
    # reset Phase 3's sticky tap flag, defeating fix #5.
    STICKY_FROM_P3 = ("tapped", "tapped_ist")
    fresh_disk = load_json(os.path.join("state", "active_watch_state.json"), {})
    for k, v in watch_writes.items():
        prior = fresh_disk.get(k)
        if isinstance(prior, dict) and isinstance(v, dict):
            for sf in STICKY_FROM_P3:
                if sf in prior and sf not in v:
                    v[sf] = prior[sf]
        fresh_disk[k] = v
    save_json(os.path.join("state", "active_watch_state.json"), fresh_disk)
    print(f"Phase 2 complete. Watch upserts: {len(watch_writes)}")

    # Heartbeat — runs after main scan is fully saved. Wrapped in try/except.
    send_heartbeat_if_due(ist_now, active_obs)
