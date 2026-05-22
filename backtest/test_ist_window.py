"""Sanity tests for the IST trading-window gate.

The gate exists because the live system suppresses everything outside
the user's IST trading window -- the backtest must mirror this. Anything
that lets blocked rows leak into aggregate metrics is a critical bug.

Window definitions (single source of truth = backtest/ist_window.py):
  forex / commodity : UTC 03:30 .. 18:30  (=  IST 09:00 .. 24:00)
  index             : UTC 13:00 .. 20:00  (=  IST 18:30 .. 01:30)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest import ist_window


def check(condition: bool, label: str) -> bool:
    status = "OK  " if condition else "FAIL"
    print(f"  {status}: {label}")
    return condition


def _ts(hour: int, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(f"2026-05-07 {hour:02d}:{minute:02d}:00", tz="UTC")


def test_forex_window():
    print("\n== test_forex_window ==")
    ok = True
    # Inside.
    ok &= check(ist_window.in_user_trading_window(_ts(3, 30), "forex"),
                "03:30 UTC is inside (= 09:00 IST, open)")
    ok &= check(ist_window.in_user_trading_window(_ts(12, 0), "forex"),
                "12:00 UTC is inside (mid-day)")
    ok &= check(ist_window.in_user_trading_window(_ts(18, 0), "forex"),
                "18:00 UTC is inside (= 23:30 IST)")
    # Outside -- close is exclusive.
    ok &= check(not ist_window.in_user_trading_window(_ts(18, 30), "forex"),
                "18:30 UTC is outside (= 24:00 IST, closed)")
    ok &= check(not ist_window.in_user_trading_window(_ts(3, 0), "forex"),
                "03:00 UTC is outside (= 08:30 IST, too early)")
    ok &= check(not ist_window.in_user_trading_window(_ts(0, 0), "forex"),
                "00:00 UTC is outside")
    ok &= check(not ist_window.in_user_trading_window(_ts(20, 0), "forex"),
                "20:00 UTC is outside")
    return ok


def test_commodity_window_same_as_forex():
    print("\n== test_commodity_window_same_as_forex ==")
    ok = True
    # Commodity uses the same window as forex (handled by default branch).
    ok &= check(ist_window.in_user_trading_window(_ts(10, 0), "commodity"),
                "10:00 UTC commodity inside")
    ok &= check(not ist_window.in_user_trading_window(_ts(2, 0), "commodity"),
                "02:00 UTC commodity outside")
    return ok


def test_index_window():
    print("\n== test_index_window ==")
    ok = True
    ok &= check(ist_window.in_user_trading_window(_ts(13, 0), "index"),
                "13:00 UTC is inside (= 18:30 IST, open)")
    ok &= check(ist_window.in_user_trading_window(_ts(19, 59), "index"),
                "19:59 UTC is inside (just before close)")
    ok &= check(not ist_window.in_user_trading_window(_ts(20, 0), "index"),
                "20:00 UTC is outside (= 01:30 IST, closed)")
    ok &= check(not ist_window.in_user_trading_window(_ts(12, 0), "index"),
                "12:00 UTC is outside (= 17:30 IST, too early)")
    ok &= check(not ist_window.in_user_trading_window(_ts(8, 0), "index"),
                "08:00 UTC is outside (Asia-time, no trading)")
    return ok


def test_unknown_pair_type_defaults_to_forex():
    print("\n== test_unknown_pair_type_defaults_to_forex ==")
    ok = True
    # Defensive: unknown pair_type should not crash; treat as forex/commodity.
    ok &= check(ist_window.in_user_trading_window(_ts(10, 0), "unknown_type"),
                "unknown pair_type uses forex/commodity bounds")
    ok &= check(not ist_window.in_user_trading_window(_ts(2, 0), "unknown_type"),
                "unknown pair_type outside-of-window correctly blocked")
    return ok


def test_naive_timestamp_rejected():
    print("\n== test_naive_timestamp_rejected ==")
    naive = pd.Timestamp("2026-05-07 10:00:00")
    try:
        ist_window.in_user_trading_window(naive, "forex")
        return check(False, "naive timestamp must raise ValueError")
    except ValueError:
        return check(True, "naive timestamp raises ValueError")


def test_report_excludes_ist_blocked_from_aggregates():
    """End-to-end: ist_blocked=True trades must not enter any aggregate.

    This is the same metric-exclusion invariant the news-blackout test
    enforces, applied to the IST gate.
    """
    print("\n== test_report_excludes_ist_blocked_from_aggregates ==")
    import json, tempfile
    from datetime import datetime
    from backtest import h1_only_reporting
    from backtest.test_news_filter import _baseline_trades, _synth_trade

    def _build(trades):
        run_id = f"_test_ist_{datetime.utcnow().strftime('%H%M%S%f')}"
        meta = {
            "start": "2026-05-07", "end": "2026-05-08",
            "regime": "test", "mode": "h1_only",
            "pairs": ["EURUSD"], "generated_utc": "2026-05-08T00:00:00+00:00",
            "news_coverage": {}, "news_events_fetched": 0,
            "news_blocked_rows": 0, "news_window_minutes": 30,
            "ist_window_forex": "UTC 03:30-18:30",
            "ist_window_index": "UTC 13:00-20:00",
        }
        out_dir = h1_only_reporting.write_h1_only_report(
            run_id, trades, raw_alerts=[], meta=meta, risk_usd=250.0,
        )
        with open(out_dir / "summary.json") as f:
            summary = json.load(f)
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        return summary

    baseline = _baseline_trades()
    summary_a = _build(baseline)

    poison = [
        _synth_trade("EURUSD", "2026-05-07 01:00:00+00:00", "proximal", "tp2",
                     +10.0, ist_blocked=True, alert_utc_hour=1,
                     session="Asia"),
        _synth_trade("EURUSD", "2026-05-07 01:00:00+00:00", "50pct", "tp2",
                     +10.0, ist_blocked=True, alert_utc_hour=1,
                     session="Asia"),
    ]
    summary_b = _build(baseline + poison)

    ok = True
    sb_a = summary_a["scoreboards"]["proximal_realised"]
    sb_b = summary_b["scoreboards"]["proximal_realised"]
    ok &= check(sb_a == sb_b,
                "proximal_realised scoreboard unchanged after IST-blocked poison")
    ok &= check(summary_b["ist_blocked_trade_rows"] == 2,
                "ist_blocked_trade_rows = 2 (1 alert x 2 zones)")
    ok &= check(summary_a["ist_blocked_trade_rows"] == 0,
                "baseline ist_blocked_trade_rows = 0")
    return ok


def main():
    tests = [
        ("test_forex_window",                          test_forex_window),
        ("test_commodity_window_same_as_forex",        test_commodity_window_same_as_forex),
        ("test_index_window",                          test_index_window),
        ("test_unknown_pair_type_defaults_to_forex",   test_unknown_pair_type_defaults_to_forex),
        ("test_naive_timestamp_rejected",              test_naive_timestamp_rejected),
        ("test_report_excludes_ist_blocked_from_aggregates",
         test_report_excludes_ist_blocked_from_aggregates),
    ]
    results = [(name, fn()) for name, fn in tests]
    print("\n=== SUMMARY ===")
    failed = 0
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'} : {name}")
        if not passed:
            failed += 1
    if failed:
        print(f"\n{failed} of {len(results)} test(s) FAILED.")
        sys.exit(1)
    print(f"\nAll {len(results)} tests passed.")


if __name__ == "__main__":
    main()
