"""Sanity tests for the news blackout filter and its integration with the
backtest report.

Two layers:
  1. news_filter unit tests (no network): currency map, window edges,
     zero-hallucination, audit-event return.
  2. Metric-exclusion invariant: render the report twice -- once with N
     trades, once with N + K extreme-outcome news-blocked trades -- and
     prove every aggregate metric is identical. If any metric changes,
     the blocked trade leaked into a calculation.

Run: python -m backtest.test_news_filter
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import news_filter as nf
from backtest import h1_only_reporting


def check(cond: bool, msg: str) -> bool:
    if cond:
        print(f"  OK:   {msg}")
        return True
    print(f"  FAIL: {msg}")
    return False


# ---------------------------------------------------------------------------
# Layer 1: news_filter unit tests
# ---------------------------------------------------------------------------

def test_currency_map():
    print("\n== test_currency_map ==")
    ok = True
    ok &= check(nf.currencies_for_pair("EURUSD") == frozenset({"USD", "EUR"}),
                "EURUSD -> {USD, EUR}")
    ok &= check(nf.currencies_for_pair("XAUUSD") == frozenset({"USD"}),
                "XAUUSD -> {USD} only")
    ok &= check(nf.currencies_for_pair("NAS100") == frozenset({"USD"}),
                "NAS100 -> {USD} only")
    ok &= check(nf.currencies_for_pair("UNKNOWN_PAIR") == frozenset(),
                "unknown pair -> empty (never blocks)")
    return ok


def test_zero_hallucination():
    print("\n== test_zero_hallucination ==")
    ts = datetime(2026, 5, 7, 13, 30, tzinfo=timezone.utc)
    blocked, src = nf.is_news_blackout(ts, "EURUSD", [], window_minutes=30)
    ok = check(blocked is False and src is None,
               "empty events list returns (False, None)")
    return ok


def test_window_edges():
    print("\n== test_window_edges ==")
    ev = {"ts_utc":   datetime(2026, 5, 7, 13, 30, tzinfo=timezone.utc),
          "currency": "USD", "impact": "High",
          "title":    "NFP", "source": "ff"}
    cases = [
        (datetime(2026, 5, 7, 12, 59, tzinfo=timezone.utc), False, "before window (-31m)"),
        (datetime(2026, 5, 7, 13,  0, tzinfo=timezone.utc), True,  "window start (-30m exact)"),
        (datetime(2026, 5, 7, 13, 30, tzinfo=timezone.utc), True,  "event exact"),
        (datetime(2026, 5, 7, 14,  0, tzinfo=timezone.utc), True,  "window end (+30m exact)"),
        (datetime(2026, 5, 7, 14,  1, tzinfo=timezone.utc), False, "after window (+31m)"),
    ]
    ok = True
    for ts, expected, label in cases:
        got, _ = nf.is_news_blackout(ts, "EURUSD", [ev], window_minutes=30)
        ok &= check(got == expected, label)
    return ok


def test_currency_filter():
    print("\n== test_currency_filter ==")
    ts = datetime(2026, 5, 7, 13, 30, tzinfo=timezone.utc)
    usd_ev = {"ts_utc": ts, "currency": "USD", "impact": "High",
              "title": "NFP", "source": "ff"}
    eur_ev = {"ts_utc": ts, "currency": "EUR", "impact": "High",
              "title": "ECB Rate", "source": "ff"}

    ok = True
    # USD event blocks USDJPY (USD in pair)
    b, _ = nf.is_news_blackout(ts, "USDJPY", [usd_ev])
    ok &= check(b, "USD event blocks USDJPY")
    # EUR event does NOT block NZDUSD (EUR not in pair)
    b, _ = nf.is_news_blackout(ts, "NZDUSD", [eur_ev])
    ok &= check(not b, "EUR event does not block NZDUSD")
    # EUR event blocks EURUSD
    b, _ = nf.is_news_blackout(ts, "EURUSD", [eur_ev])
    ok &= check(b, "EUR event blocks EURUSD")
    return ok


def test_audit_event_returned():
    print("\n== test_audit_event_returned ==")
    ev = {"ts_utc": datetime(2026, 5, 7, 13, 30, tzinfo=timezone.utc),
          "currency": "USD", "impact": "High",
          "title": "NFP", "source": "ff"}
    b, src = nf.is_news_blackout(ev["ts_utc"], "EURUSD", [ev])
    ok = True
    ok &= check(b, "block fires")
    ok &= check(src is ev, "source event returned by identity for audit")
    return ok


def test_naive_datetime_rejected():
    print("\n== test_naive_datetime_rejected ==")
    ev = {"ts_utc": datetime(2026, 5, 7, 13, 30, tzinfo=timezone.utc),
          "currency": "USD", "impact": "High",
          "title": "NFP", "source": "ff"}
    try:
        nf.is_news_blackout(datetime(2026, 5, 7, 13, 30),
                             "EURUSD", [ev])
    except ValueError:
        return check(True, "naive datetime raises ValueError")
    return check(False, "naive datetime accepted (should raise)")


# ---------------------------------------------------------------------------
# Layer 2: metric-exclusion invariant
# ---------------------------------------------------------------------------

def _synth_trade(pair, alert_ts, entry_zone, exit_reason, r,
                 *, killzone=True, news_blocked=False,
                 news_event_title="", session="London", bos_tag="BOS",
                 bos_tier="Major"):
    """Builds a minimal trade row with all keys the report needs."""
    return {
        "pair":              pair,
        "alert_ts":          alert_ts,
        "entry_zone":        entry_zone,
        "exit_reason":       exit_reason,
        "r_realised":        r,
        "r_if_exit_tp1":     min(r, 1.5),
        "r_if_exit_tp2":     r,
        "pnl_usd":           r * 250,
        "mfe_r":             max(r, 0),
        "mae_r":             min(r, 0),
        "bars_to_exit":      10,
        "bars_to_tp1":       5 if r > 0 else -1,
        "bars_to_tp2":       8 if r > 0 else -1,
        "tp1_rr":            1.5,
        "tp2_rr":            3.0,
        "entry":             1.1000,
        "sl_initial":        1.0980,
        "tp1":               1.1030,
        "tp2":               1.1060,
        "exit_price":        1.1060 if r > 0 else 1.0980,
        "ob_age_h1_bars":    10,
        "pd_zone":           "discount",
        "score":             4.5,
        "structure_pts":     1.5,
        "sweep_pts":         0.0,
        "fvg_pts":           1.0,
        "freshness_pts":     1.5,
        "killzone_pts":      0.5 if killzone else 0.0,
        "confluences_present": "structure,fvg,freshness",
        "sl_collision":      False,
        "model":             "h1_only",
        "ob_timestamp":      "2026-05-07T12:00:00+00:00",
        "bos_tag":           bos_tag,
        "bos_tier":          bos_tier,
        "fvg_present":       True,
        "sweep_present":     False,
        "session":           session,
        "direction":         "bullish",
        "news_blocked":         news_blocked,
        "news_event_title":     news_event_title,
        "news_event_currency":  "USD" if news_blocked else "",
        "news_event_source":    "ff" if news_blocked else "",
        "news_event_ts":        "2026-05-07T13:30:00+00:00" if news_blocked else "",
    }


def _baseline_trades():
    """Six clean (non-blocked) trades: 3 wins, 3 losses, varied pair/session."""
    return [
        _synth_trade("EURUSD", "2026-05-07 08:00:00+00:00", "proximal", "tp2", 2.0),
        _synth_trade("EURUSD", "2026-05-07 08:00:00+00:00", "50pct",    "tp2", 3.0),
        _synth_trade("EURUSD", "2026-05-08 09:00:00+00:00", "proximal", "sl", -1.0),
        _synth_trade("EURUSD", "2026-05-08 09:00:00+00:00", "50pct",    "sl", -1.0),
        _synth_trade("USDCHF", "2026-05-08 16:00:00+00:00", "proximal", "tp2", 1.8,
                     session="NY"),
        _synth_trade("USDCHF", "2026-05-08 16:00:00+00:00", "50pct",    "tp2", 2.5,
                     session="NY"),
        _synth_trade("NZDUSD", "2026-05-08 02:00:00+00:00", "proximal", "sl", -1.0,
                     session="Asia", killzone=False),
        _synth_trade("NZDUSD", "2026-05-08 02:00:00+00:00", "50pct",    "sl", -1.0,
                     session="Asia", killzone=False),
    ]


def _build_summary(trades):
    """Render the report into a temp dir and return summary.json."""
    with tempfile.TemporaryDirectory() as td:
        # The report writes to backtest/results/<run_id>; we let it. Then
        # we read the summary out and clean up.
        run_id = f"_test_news_{datetime.utcnow().strftime('%H%M%S%f')}"
        meta = {
            "start": "2026-05-07", "end": "2026-05-08",
            "regime": "test", "mode": "h1_only",
            "pairs": ["EURUSD", "USDCHF", "NZDUSD"],
            "generated_utc": "2026-05-08T00:00:00+00:00",
            "news_coverage": {"ff": True, "gdelt": True},
            "news_events_fetched": 0,
            "news_blocked_rows": sum(1 for t in trades if t.get("news_blocked")),
            "news_window_minutes": 30,
        }
        out_dir = h1_only_reporting.write_h1_only_report(
            run_id, trades, raw_alerts=[], meta=meta, risk_usd=250.0,
        )
        with open(out_dir / "summary.json", "r") as f:
            summary = json.load(f)
        # Cleanup the temp run dir.
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        return summary


def _extract_metrics(summary):
    """Pull every aggregate metric we care about into a flat dict."""
    sb_prox_tp2 = summary["scoreboards"]["proximal_exit_tp2"]
    sb_mid_tp2  = summary["scoreboards"]["fifty_pct_exit_tp2"]
    sb_prox_tp1 = summary["scoreboards"]["proximal_exit_tp1"]
    sb_mid_tp1  = summary["scoreboards"]["fifty_pct_exit_tp1"]
    return {
        "total_trade_rows":              summary["total_trade_rows"],
        "fill_rate_proximal":            summary["fill_rate_proximal"],
        "fill_rate_50pct":               summary["fill_rate_50pct"],
        "exit_reason_counts_proximal":   summary["exit_reason_counts_proximal"],
        "exit_reason_counts_50pct":      summary["exit_reason_counts_50pct"],
        "prox_tp2_pnl":                  sb_prox_tp2.get("total_pnl_usd"),
        "prox_tp2_wr":                   sb_prox_tp2.get("win_rate_pct"),
        "prox_tp2_exp":                  sb_prox_tp2.get("expectancy_r"),
        "prox_tp2_trades":               sb_prox_tp2.get("trades"),
        "prox_tp2_wins":                 sb_prox_tp2.get("wins"),
        "prox_tp2_losses":               sb_prox_tp2.get("losses"),
        "prox_tp1_pnl":                  sb_prox_tp1.get("total_pnl_usd"),
        "prox_tp1_wr":                   sb_prox_tp1.get("win_rate_pct"),
        "mid_tp2_pnl":                   sb_mid_tp2.get("total_pnl_usd"),
        "mid_tp2_wr":                    sb_mid_tp2.get("win_rate_pct"),
        "mid_tp1_pnl":                   sb_mid_tp1.get("total_pnl_usd"),
        "per_pair_proximal_tp2":         summary["per_pair_proximal_tp2"],
        "per_pair_50pct_tp2":            summary["per_pair_50pct_tp2"],
        "score_buckets_tp1":             summary["score_buckets_tp1"],
        "score_buckets_tp2":             summary["score_buckets_tp2"],
    }


def test_metric_exclusion_invariant():
    """The killer test: adding news-blocked trades must not change any metric."""
    print("\n== test_metric_exclusion_invariant ==")

    baseline = _baseline_trades()
    summary_a = _build_summary(baseline)
    metrics_a = _extract_metrics(summary_a)

    # Add extreme-outcome blocked trades that WOULD wildly distort metrics
    # if they leaked in: a +10R win and a -5R loss.
    poison = [
        _synth_trade("EURUSD", "2026-05-07 13:30:00+00:00", "proximal", "tp2",
                     +10.0, news_blocked=True,
                     news_event_title="NFP +10R poison"),
        _synth_trade("EURUSD", "2026-05-07 13:30:00+00:00", "50pct",    "tp2",
                     +10.0, news_blocked=True,
                     news_event_title="NFP +10R poison"),
        _synth_trade("USDJPY", "2026-05-07 14:30:00+00:00", "proximal", "sl",
                     -5.0, news_blocked=True,
                     news_event_title="FOMC -5R poison"),
        _synth_trade("USDJPY", "2026-05-07 14:30:00+00:00", "50pct",    "sl",
                     -5.0, news_blocked=True,
                     news_event_title="FOMC -5R poison"),
    ]
    poisoned = baseline + poison
    summary_b = _build_summary(poisoned)
    metrics_b = _extract_metrics(summary_b)

    ok = True
    # Every metric must match exactly.
    for k in metrics_a:
        ok &= check(metrics_a[k] == metrics_b[k],
                    f"{k} unchanged after adding {len(poison)} blocked trades")

    # Audit fields MUST differ -- the blocked trades should be visible.
    ok &= check(summary_b["news_blocked_trade_rows"] == 4,
                "news_blocked_trade_rows = 4 (= 2 alerts x 2 zones)")
    ok &= check(summary_a["news_blocked_trade_rows"] == 0,
                "baseline news_blocked_trade_rows = 0")
    ok &= check(len(summary_b["news_blocked_audit"]) == 4,
                "news_blocked_audit lists all 4 blocked rows")
    ok &= check(
        any("NFP +10R poison" in e["event_title"]
            for e in summary_b["news_blocked_audit"]),
        "audit list mentions the +10R poison event"
    )

    return ok


def test_tz_pad_helper_accepts_both():
    """_run_h1_only pads the news fetch range by 1 day on each side. The pad
    builder must accept both tz-aware and naive inputs because the start/end
    parameters come from argparse via _parse_date in one branch and from
    pd.Timestamp construction in another. A regression broke this in CI:
    `Timestamp.tz_localize` raises on already-aware values."""
    print("\n== test_tz_pad_helper_accepts_both ==")
    import pandas as pd
    from datetime import datetime, timezone

    # The exact helper defined in _run_h1_only (kept in sync; if the
    # function changes, mirror it here).
    def _to_utc(ts):
        t = pd.Timestamp(ts)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

    ok = True
    # tz-aware datetime (the CI failure case)
    aware = datetime(2024, 11, 4, tzinfo=timezone.utc)
    try:
        out = _to_utc(aware)
        ok &= check(out.tzinfo is not None, "aware -> still aware")
    except Exception as e:
        ok &= check(False, f"aware input crashed: {type(e).__name__}: {e}")

    # naive datetime
    naive = datetime(2024, 11, 4)
    try:
        out = _to_utc(naive)
        ok &= check(out.tzinfo is not None, "naive -> localized to UTC")
    except Exception as e:
        ok &= check(False, f"naive input crashed: {type(e).__name__}: {e}")

    # string
    try:
        out = _to_utc("2024-11-04")
        ok &= check(out.tzinfo is not None, "string -> localized to UTC")
    except Exception as e:
        ok &= check(False, f"string input crashed: {type(e).__name__}: {e}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = [
        ("test_currency_map",              test_currency_map()),
        ("test_zero_hallucination",        test_zero_hallucination()),
        ("test_window_edges",              test_window_edges()),
        ("test_currency_filter",           test_currency_filter()),
        ("test_audit_event_returned",      test_audit_event_returned()),
        ("test_naive_datetime_rejected",   test_naive_datetime_rejected()),
        ("test_tz_pad_helper_accepts_both", test_tz_pad_helper_accepts_both()),
        ("test_metric_exclusion_invariant", test_metric_exclusion_invariant()),
    ]
    print("\n=== SUMMARY ===")
    fail = 0
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'} : {name}")
        if not ok:
            fail += 1
    if fail:
        print(f"\n{fail}/{len(results)} test(s) failed.")
        sys.exit(1)
    print(f"\nAll {len(results)} tests passed.")


if __name__ == "__main__":
    main()
