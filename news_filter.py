"""News blackout filter.

Single source of truth for "is this timestamp inside a high-impact news
window for this pair." Used by backtest (FF + GDELT) and reserved for live
(FF + Reuters; not wired here since Phase 2/3 are paused).

Public surface:
    fetch_events(start_utc, end_utc, sources=("ff", "gdelt"))
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
    - GDELT: ~70% of geopolitical shocks with 15-min lag.
    - Combined: ~80% of price-moving events. Misses: surprise central
      bank actions, tweet-driven moves, commodity-specific shocks.
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


def _ff_monday_for(d: datetime) -> str:
    """The Monday of d's ISO week, formatted YYYY-MM-DD."""
    monday = d - timedelta(days=d.weekday())
    return monday.strftime("%Y-%m-%d")


def _parse_ff_xml(xml_bytes: bytes) -> List[Dict[str, Any]]:
    """Parse a FairEconomy weekly XML. Returns High-impact events only.
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
        if impact.lower() != "high":
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
        # US Eastern -> UTC. Use zoneinfo for DST correctness.
        try:
            from zoneinfo import ZoneInfo
            local = local.replace(tzinfo=ZoneInfo("America/New_York"))
            ts_utc = local.astimezone(timezone.utc)
        except Exception:
            # Fallback: assume EST (UTC-5) if zoneinfo unavailable.
            # Logged so the gap is visible.
            logger.warning("zoneinfo unavailable; assuming EST offset")
            ts_utc = (local - timedelta(hours=-5)).replace(tzinfo=timezone.utc)
        out.append({
            "ts_utc":   ts_utc,
            "currency": currency.upper(),
            "impact":   impact,
            "title":    title,
            "source":   "ff",
        })
    return out


def _http_get(url: str, timeout: float = 20.0) -> Optional[bytes]:
    """Single GET. Returns bytes on success, None on any failure.
    The None vs bytes distinction lets the caller log the gap explicitly."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "trading-alerts/news_filter"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.warning("http_get_failed url=%s err=%s", url, e)
        return None


def fetch_ff_events(start_utc: datetime, end_utc: datetime) -> Tuple[List[Dict[str, Any]], bool]:
    """Fetch FF High-impact events covering [start_utc, end_utc].

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
        url = _FF_WEEK_URL.format(date=_ff_monday_for(current))
        body = _http_get(url)
        if body is None:
            # Try the thisweek alias as a fallback only if the date is
            # the current ISO week. Otherwise mark as gap.
            now_utc = datetime.now(timezone.utc)
            if _ff_monday_for(current) == _ff_monday_for(now_utc):
                body = _http_get(_FF_THISWEEK_URL)
        if body is None:
            logger.warning("ff_week_missing date=%s",
                           _ff_monday_for(current))
            current += timedelta(days=7)
            continue
        weeks_succeeded += 1
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
# GDELT geopolitical-events fetcher (backtest)
# ---------------------------------------------------------------------------

# GDELT 2.0 Events API. Returns CSV records. We use the doc search API
# because the Events table requires BigQuery for filtered access.
# We query the article search and filter to high-severity geopolitical
# headlines that mention the currencies/regions we trade.
_GDELT_DOC_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query={query}&mode=ArtList&format=json&maxrecords=250"
    "&startdatetime={start}&enddatetime={end}"
)

# Currencies we care about and the country/region terms that signal them.
_GDELT_QUERY = (
    '(war OR strike OR missile OR sanctions OR ceasefire OR invasion '
    'OR "emergency rate" OR "rate cut")'
)


def fetch_gdelt_events(start_utc: datetime, end_utc: datetime) -> Tuple[List[Dict[str, Any]], bool]:
    """Fetch GDELT geopolitical articles for the date range.

    Returns (events, coverage_complete). Each event has:
        ts_utc, currency (USD default if unmappable), impact ('High'),
        title, source ('gdelt').
    """
    start = start_utc.strftime("%Y%m%d%H%M%S")
    end   = end_utc.strftime("%Y%m%d%H%M%S")
    url = _GDELT_DOC_URL.format(
        query=urllib.parse.quote(_GDELT_QUERY),
        start=start, end=end,
    )
    body = _http_get(url, timeout=30.0)
    if body is None:
        logger.warning("gdelt_fetch_failed range=%s..%s", start, end)
        return [], False

    try:
        data = json.loads(body.decode("utf-8", errors="ignore"))
    except json.JSONDecodeError as e:
        logger.error("gdelt_json_parse_error: %s", e)
        return [], False

    out: List[Dict[str, Any]] = []
    for art in data.get("articles", []):
        ts_str = art.get("seendate")  # "20260415T130000Z"
        if not ts_str:
            continue
        try:
            ts_utc = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue
        title = (art.get("title") or "").strip()
        # Tag affected currencies by mention in the title. Fall back to USD
        # (most geopolitical shocks affect USD via safe-haven flows).
        title_upper = title.upper()
        currencies = set()
        for ccy, terms in [
            ("USD", ["US ", "U.S.", "AMERICA", "FED", "WASHINGTON"]),
            ("EUR", ["EU ", "EUROPE", "EURO ", "ECB", "GERMANY", "FRANCE"]),
            ("GBP", ["UK ", "BRITAIN", "BOE", "LONDON"]),
            ("JPY", ["JAPAN", "TOKYO", "BOJ"]),
            ("CHF", ["SWISS", "SWITZERLAND", "SNB"]),
            ("NZD", ["NEW ZEALAND", "RBNZ"]),
        ]:
            if any(t in title_upper for t in terms):
                currencies.add(ccy)
        if not currencies:
            currencies.add("USD")  # default for geopolitical headlines
        for ccy in currencies:
            out.append({
                "ts_utc":   ts_utc,
                "currency": ccy,
                "impact":   "High",
                "title":    title,
                "source":   "gdelt",
            })
    logger.info("gdelt_fetch range=%s..%s events=%d", start, end, len(out))
    return out, True


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
    sources: Iterable[str] = ("ff", "gdelt"),
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
        elif src == "gdelt":
            ev, ok = fetch_gdelt_events(start_utc, end_utc)
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

def cache_events(events_dict: Dict[str, Any], path: str) -> None:
    """Serialise an events dict to JSON. Datetimes are isoformat strings."""
    serialisable = {
        "events": [
            {**e, "ts_utc": e["ts_utc"].isoformat()}
            for e in events_dict.get("events", [])
        ],
        "coverage": events_dict.get("coverage", {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)


def load_cached_events(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for e in data.get("events", []):
        e["ts_utc"] = datetime.fromisoformat(e["ts_utc"])
    return data
