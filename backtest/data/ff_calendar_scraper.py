"""Scrape ForexFactory calendar month pages -> per-month JSON cache -> combined CSV.

Primary source for the news-flag build. Uses the embedded `dateline` epoch
(absolute UTC seconds) so no timezone conversion is ever applied to source data.
Resumable: months already cached on disk are skipped.
"""
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).parent / "ff_months"
OUT.mkdir(exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

EVENT_RE = re.compile(r'\{"id":\d+,"ebaseId":.*?"dateline":\d+.*?\}(?=,\{"id"|\])')
KEEP = ("id", "ebaseId", "name", "country", "currency", "dateline",
        "impactName", "impactTitle", "timeLabel", "timeMasked",
        "actual", "forecast", "previous")


def month_list():
    out = []
    for y in range(2007, 2027):
        for mi, m in enumerate(MONTHS, 1):
            if (y, mi) < (2007, 12):
                continue
            if (y, mi) > (2026, 1):
                continue
            out.append((y, mi, f"{m}.{y}"))
    return out


def fetch(tag: str) -> str:
    url = f"https://www.forexfactory.com/calendar?month={tag}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("windows-1252", errors="replace")


def parse(html: str):
    rows, seen = [], set()
    for raw in EVENT_RE.findall(html):
        try:
            e = json.loads(raw)
        except json.JSONDecodeError:
            continue
        key = (e.get("id"), e.get("dateline"))
        if key in seen:
            continue
        seen.add(key)
        rows.append({k: e.get(k) for k in KEEP})
    return rows


def main():
    todo = month_list()
    print(f"{len(todo)} months", flush=True)
    fails = []
    for y, mi, tag in todo:
        f = OUT / f"{y}-{mi:02d}.json"
        if f.exists():
            continue
        ok = False
        for attempt in range(4):
            try:
                html = fetch(tag)
                rows = parse(html)
                # a real month page has hundreds of events; < 50 = blocked/empty page
                if len(rows) < 50:
                    raise ValueError(f"only {len(rows)} events")
                f.write_text(json.dumps(rows), encoding="utf-8")
                print(f"{tag}: {len(rows)}", flush=True)
                ok = True
                break
            except Exception as ex:  # noqa: BLE001
                wait = 5 * (attempt + 1)
                print(f"{tag} attempt {attempt+1} failed: {ex}; retry in {wait}s",
                      flush=True)
                time.sleep(wait)
        if not ok:
            fails.append(tag)
        time.sleep(1.5)
    print("DONE. failures:", fails, flush=True)
    if fails:
        sys.exit(2)


if __name__ == "__main__":
    main()
