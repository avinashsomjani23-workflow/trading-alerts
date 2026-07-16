"""News blackout filter.

Single source of truth for "is this timestamp inside a high/medium-impact
news window for this pair." Used by backtest (FF only) and reserved for live
(FF + Reuters; not wired here since Phase 2/3 are paused).

Public surface:
    fetch_events(start_utc, end_utc, sources=("ff",))
        -> list of NewsEvent records
    is_news_blackout(ts_utc, pair, events, window_minutes=30)
        -> (blocked: bool, source_event: dict | None)

Design guarantees:
    1. Zero hallucination. If a feed returns nothing, no events are
       synthesised. The filter only fires on records the feeds actually
       emit. Every block returns the source record for audit.
    2. Fail-open on fetch error. If a feed is unreachable, that feed
       contributes zero events and the filter logs the gap rather than
       silently treating "no events" as "safe." A `coverage` field in the
       returned dict makes the gap visible to the caller.
    3. No metric should ever count a blocked trade. The filter only marks;
       exclusion happens in the report layer.

Coverage caveats (documented in KNOWN_LIMITATIONS.md):
    - FF: >99% of scheduled releases (NFP, CPI, FOMC, ECB, BoE, etc.).
    - Unscheduled geopolitical shocks are NOT covered. We accept this gap
      rather than inject noise from low-signal article feeds.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Currency-to-pair mapping
# ---------------------------------------------------------------------------

# Which currencies are "in" each pair. A High-impact event on any listed
# currency triggers blackout for that pair.
#
# XAUUSD / NAS100 / GOLD: USD-only is industry standard. Gold and indices
# also react to global risk-off events, but FF tags those as USD-impact too
# (NFP, FOMC, geopolitical shocks affecting USD), so USD coverage catches them.
_PAIR_CURRENCIES: Dict[str, frozenset] = {
    "EURUSD": frozenset({"USD", "EUR"}),
    "NZDUSD": frozenset({"USD", "NZD"}),
    "USDJPY": frozenset({"USD", "JPY"}),
    "USDCHF": frozenset({"USD", "CHF"}),
    "XAUUSD": frozenset({"USD"}),
    "GOLD":   frozenset({"USD"}),
    "NAS100": frozenset({"USD"}),
    # Backtest-only pairs (config.json backtest_only). Live never queries
    # these; listed so offline analysis (backtest/news_enrichment.py) shares
    # this one map instead of duplicating it.
    "GBPUSD": frozenset({"USD", "GBP"}),
    "AUDUSD": frozenset({"USD", "AUD"}),
    "USDCAD": frozenset({"USD", "CAD"}),
    "EURJPY": frozenset({"EUR", "JPY"}),
    "BTCUSD": frozenset({"USD"}),
}


def currencies_for_pair(pair: str) -> frozenset:
    return _PAIR_CURRENCIES.get(pair.upper(), frozenset())


# ---------------------------------------------------------------------------
# ForexFactory / FairEconomy scheduled-events fetcher
# ---------------------------------------------------------------------------

# FairEconomy publishes the FF calendar at this URL. `thisweek` returns
# the current ISO week (Monday-Sunday). For historical weeks they expose
# per-week URLs of the form `ff_calendar_YYYY-MM-DD.xml` (date = a Monday).
_FF_THISWEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
_FF_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_{date}.xml"

# Per-week XML cache. FF rate-limits (429) and drops older weekly files, so a
# live fetch can fail even when the data existed minutes ago. We persist each
# week's raw XML on success and serve it when a later fetch fails. Past weeks
# are immutable history; only the current week changes intraday, so a slightly
# stale current-week copy is still far better than an empty calendar.
_FF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "state", "ff_calendar_cache")


def _ff_cache_path(monday: str) -> str:
    return os.path.join(_FF_CACHE_DIR, f"ff_calendar_{monday}.xml")


def _ff_cache_write(monday: str, body: bytes) -> None:
    """Persist a week's raw XML. Best-effort: a cache write must never break
    the fetch path."""
    try:
        os.makedirs(_FF_CACHE_DIR, exist_ok=True)
        path = _ff_cache_path(monday)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, path)  # atomic — no half-written cache on crash
    except OSError as e:
        logger.warning("ff_cache_write_failed monday=%s err=%s", monday, e)


def _ff_cache_read(monday: str) -> Optional[bytes]:
    """Return cached week XML, or None if absent/unreadable."""
    try:
        with open(_ff_cache_path(monday), "rb") as f:
            return f.read()
    except OSError:
        return None


def _ff_monday_for(d: datetime) -> str:
    """The Monday of d's ISO week, formatted YYYY-MM-DD."""
    monday = d - timedelta(days=d.weekday())
    return monday.strftime("%Y-%m-%d")


def _parse_ff_xml(xml_bytes: bytes) -> List[Dict[str, Any]]:
    """Parse a FairEconomy weekly XML. Returns High and Medium-impact events.
    Each event has: ts_utc (datetime), currency (str), impact (str),
    title (str), source ('ff')."""
    out: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        logger.error("ff_xml_parse_error: %s", e)
        return out

    for event in root.findall(".//event"):
        impact = (event.findtext("impact") or "").strip()
        if impact.lower() not in ("high", "medium"):
            continue
        date_str = (event.findtext("date") or "").strip()       # "MM-DD-YYYY"
        time_str = (event.findtext("time") or "").strip()       # "1:30pm"
        currency = (event.findtext("country") or "").strip()    # "USD"
        title    = (event.findtext("title") or "").strip()
        if not (date_str and time_str and currency):
            continue
        # FF publishes in US Eastern. The XML feed marks the timezone
        # implicitly; the FairEconomy feed honors US/Eastern (EST/EDT).
        # We convert to UTC.
        try:
            local = datetime.strptime(f"{date_str} {time_str}",
                                       "%m-%d-%Y %I:%M%p")
        except ValueError:
            # "All Day" / "Tentative" / "Day 1" entries have no precise
            # time. Skip — we can't apply a ±30min window without one.
            continue
        # The FairEconomy weekly XML publishes <time> in UTC (NOT US Eastern).
        # Verified against release anchors: ISM 2:00pm = 14:00 UTC = 10am EDT;
        # Spanish Flash CPI 7:00am = 07:00 UTC = 09:00 CEST; JP data 11:30pm =
        # 23:30 UTC = 08:30 JST. The old code treated it as America/New_York
        # and shifted +4h (EDT), pushing every US event 4h late (a 7:30pm IST
        # release rendered as 11:30pm IST). Stamp UTC directly — no shift.
        ts_utc = local.replace(tzinfo=timezone.utc)
        out.append({
            "ts_utc":   ts_utc,
            "currency": currency.upper(),
            "impact":   impact,
            "title":    title,
            "source":   "ff",
        })
    return out


def _http_get(url: str, timeout: float = 20.0, retries: int = 2) -> Optional[bytes]:
    """GET with a short retry on transient failures (429 / 5xx / network).
    Returns bytes on success, None after all attempts fail. The None vs bytes
    distinction lets the caller fall back to cache and log the gap explicitly."""
    import time
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "trading-alerts/news_filter"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            # Retry only transient conditions. A 404 (week file removed) is
            # permanent for this URL — don't waste attempts, let the caller
            # fall back to the thisweek alias / cache.
            code = getattr(e, "code", None)
            transient = code in (429, 500, 502, 503, 504) or code is None
            if transient and attempt < retries:
                wait = 2 * (attempt + 1)  # 2s, then 4s
                logger.warning("http_get_retry url=%s err=%s wait=%ss", url, e, wait)
                time.sleep(wait)
                continue
            logger.warning("http_get_failed url=%s err=%s", url, e)
            return None
    return None


def fetch_ff_events(start_utc: datetime, end_utc: datetime) -> Tuple[List[Dict[str, Any]], bool]:
    """Fetch FF High and Medium-impact events covering [start_utc, end_utc].

    Returns (events, coverage_complete). coverage_complete=False means at
    least one weekly fetch failed — the caller should propagate that to
    the run metadata so the report can flag partial coverage.
    """
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError("fetch_ff_events requires tz-aware UTC datetimes")

    events: List[Dict[str, Any]] = []
    weeks_attempted = 0
    weeks_succeeded = 0

    # Walk Monday-to-Monday across the date range.
    current = start_utc - timedelta(days=start_utc.weekday())
    while current.date() <= end_utc.date():
        weeks_attempted += 1
        monday = _ff_monday_for(current)
        url = _FF_WEEK_URL.format(date=monday)
        body = _http_get(url)
        if body is None:
            # Try the thisweek alias as a fallback only if the date is
            # the current ISO week. Otherwise the per-week URL is the only
            # live source.
            now_utc = datetime.now(timezone.utc)
            if monday == _ff_monday_for(now_utc):
                body = _http_get(_FF_THISWEEK_URL)

        if body is not None:
            # Live fetch worked — refresh the cache and count full coverage.
            _ff_cache_write(monday, body)
            weeks_succeeded += 1
        else:
            # Live fetch failed (429 / removed file / network). Serve the last
            # good copy of THIS week so the calendar is never silently empty.
            body = _ff_cache_read(monday)
            if body is not None:
                logger.warning("ff_week_from_cache date=%s", monday)
                weeks_succeeded += 1  # we still have the data, just not live
            else:
                logger.warning("ff_week_missing date=%s (no cache)", monday)
                current += timedelta(days=7)
                continue

        events.extend(_parse_ff_xml(body))
        current += timedelta(days=7)

    # Filter to the requested range.
    events = [e for e in events
              if start_utc <= e["ts_utc"] <= end_utc]
    coverage_complete = (weeks_succeeded == weeks_attempted)
    logger.info("ff_fetch weeks_ok=%d/%d events=%d",
                weeks_succeeded, weeks_attempted, len(events))
    return events, coverage_complete


# ---------------------------------------------------------------------------
# Reuters RSS scaffold (live; not wired since Phase 2/3 are paused)
# ---------------------------------------------------------------------------

def fetch_reuters_events(start_utc: datetime, end_utc: datetime) -> Tuple[List[Dict[str, Any]], bool]:
    """Reuters RSS scaffold for live trading. Not called from the backtest
    pipeline. Will be wired into Phase 2 when live trading resumes.

    Returns (events, coverage_complete). For now, returns ([], True) so
    callers can compose it into a multi-source fetcher without changes.
    """
    # Intentionally a stub. Live wiring will replace this body with a
    # poll of e.g. http://feeds.reuters.com/reuters/businessNews keyword-
    # filtered for war/strike/sanctions/ceasefire/emergency.
    return [], True


# ---------------------------------------------------------------------------
# Public API: fetch_events + is_news_blackout
# ---------------------------------------------------------------------------

def fetch_events(
    start_utc: datetime,
    end_utc: datetime,
    sources: Iterable[str] = ("ff",),
) -> Dict[str, Any]:
    """Fetch all news events for [start_utc, end_utc] from the requested
    sources. Returns a dict with:
        events: List[event_dict]
        coverage: dict mapping source -> bool (True = complete)
    """
    all_events: List[Dict[str, Any]] = []
    coverage: Dict[str, bool] = {}
    for src in sources:
        if src == "ff":
            ev, ok = fetch_ff_events(start_utc, end_utc)
        elif src == "reuters":
            ev, ok = fetch_reuters_events(start_utc, end_utc)
        else:
            logger.error("unknown_news_source: %s", src)
            continue
        all_events.extend(ev)
        coverage[src] = ok
    return {"events": all_events, "coverage": coverage}


def is_news_blackout(
    ts_utc: datetime,
    pair: str,
    events: List[Dict[str, Any]],
    window_minutes: int = 30,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """True if `ts_utc` falls within ±window_minutes of any High-impact
    event affecting any currency in `pair`.

    Zero-hallucination guarantee: returns False unless an actual event
    record from `events` is matched. The matched event is returned as the
    second tuple element for audit.

    Window semantics: inclusive at both ends. ts at exactly event - window
    blocks; ts at event + window also blocks; ts at event + window + 1
    second does NOT block.
    """
    if ts_utc.tzinfo is None:
        raise ValueError("is_news_blackout requires tz-aware ts_utc")
    pair_ccys = currencies_for_pair(pair)
    if not pair_ccys:
        return False, None  # unknown pair -> never block (visible bug, not silent)
    window = timedelta(minutes=window_minutes)
    for ev in events:
        if ev["currency"] not in pair_ccys:
            continue
        delta = ts_utc - ev["ts_utc"]
        if -window <= delta <= window:
            return True, ev
    return False, None


# ---------------------------------------------------------------------------
# Cache (file-backed, opt-in)
# ---------------------------------------------------------------------------


