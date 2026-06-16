"""P&L reconciliation tests.

Guards against the bug class where different sections of the same email
report show different P&L for the same population of trades. Three things
that previously diverged:

  1. Section A/B headline ("realised") vs head-to-head table ("tp2 hypothetical")
  2. Section headline vs Excel "Dollar P&L" column
  3. Email headline vs registry.json headline

Each test pins one of those reconciliation points so a future change that
re-introduces the divergence fails CI instead of reaching the inbox.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest import h1_only_reporting
from backtest.update_registry import _extract_metrics


def _mk_trade(pair, entry_zone, r_realised, r_tp1, r_tp2, risk=250.0,
              exit_reason="tp2", session="NY", direction="bullish"):
    """One simulator-shaped trade row. Mirrors h1_only_simulator output."""
    return {
        "pair": pair,
        "entry_zone": entry_zone,
        "alert_ts":   "2024-08-19T13:00:00Z",
        "ob_timestamp": "2024-08-19T10:00:00Z",
        "direction":  direction,
        "session":    session,
        "entry":      1.1000, "sl_initial": 1.0950,
        "tp1": 1.1050, "tp2": 1.1100,
        "tp1_rr": 1.0, "tp2_rr": 2.0,
        "exit_reason":  exit_reason,
        "exit_price":   1.1100,
        "r_realised":   r_realised,
        "r_if_exit_tp1": r_tp1,
        "r_if_exit_tp2": r_tp2,
        "pnl_usd":      round(r_realised * risk, 2),
        "mfe_r": max(r_realised, 0.5), "mae_r": min(r_realised, -0.2),
        "bars_to_exit": 5, "bars_to_tp1": 2, "bars_to_tp2": 5,
        "ob_age_h1_bars": 3, "pd_zone": "discount",
        "score": 4, "structure_pts": 1, "sweep_pts": 1, "fvg_pts": 0,
        "freshness_pts": 1, "killzone_pts": 1,
        "confluences_present": "kill,sweep,struct",
        "sl_collision": False, "model": "h1_only",
        "bos_tag": "BOS", "bos_tier": "Major",
        "fvg_present": False, "sweep_present": True,
        "news_blocked": False, "news_event_title": None,
        "news_event_currency": None, "news_event_source": None,
        "news_event_ts": None,
        "ist_blocked": False, "alert_utc_hour": 13,
    }


def _passed(msg): print(f"  OK:   {msg}")
def _failed(msg):
    print(f"  FAIL: {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Test 1: headline P&L matches the sum of per-trade pnl_usd.
# The same assertion is hard-coded in write_h1_only_report; this test pins
# the contract so removing the assertion would still fail CI.
# ---------------------------------------------------------------------------
def test_headline_matches_per_trade_sum():
    print("\n== test_headline_matches_per_trade_sum ==")

    # Headline = realised P&L of RESOLVED trades only. never_filled, timeout
    # and window_end are unresolved (force-closed at an arbitrary price) and
    # are excluded from every aggregate -- audit-only (_EXCLUDE_REASONS).
    #
    # Proximal: +2R (tp2), -1R (sl), and a +0.5R window_end that MUST be
    #   dropped. Resolved headline = +2R -1R = +1R * 250 = $250.
    #   If window_end were counted (the pre-2026-06-16 bug) it would be $375.
    # 50pct: +1R (tp1), -1R (sl). Resolved headline = 0R * 250 = $0.
    trades = [
        _mk_trade("EURUSD", "proximal", 2.0,  1.0, 2.0, exit_reason="tp2"),
        _mk_trade("EURUSD", "proximal", -1.0, -1.0, -1.0, exit_reason="sl"),
        _mk_trade("NAS100", "proximal", 0.5, 0.5, 0.5,   exit_reason="window_end"),
        _mk_trade("EURUSD", "50pct",    1.0, 1.0, 1.0,   exit_reason="tp1"),
        _mk_trade("NAS100", "50pct",   -1.0, -1.0, -1.0, exit_reason="sl"),
    ]

    with tempfile.TemporaryDirectory() as td:
        meta = {"start": "2024-08-19", "end": "2024-08-23", "regime": "bau",
                "pairs": ["EURUSD", "NAS100"]}
        # write_h1_only_report contains the reconciliation assertion. If it
        # raises, the test fails. If it succeeds, the headline matches.
        h1_only_reporting.write_h1_only_report(
            "test_run", trades, [], meta, risk_usd=250.0, out_root=Path(td),
        )
        summary_path = Path(td) / "test_run" / "summary.json"
        s = json.loads(summary_path.read_text())

        sb_prox = s["scoreboards"]["proximal_realised"]
        sb_mid  = s["scoreboards"]["fifty_pct_realised"]

        # $250, not $375 -- the window_end row is excluded from the headline.
        if sb_prox["total_pnl_usd"] != 250.0:
            _failed(f"prox headline expected $250 (window_end excluded), "
                    f"got ${sb_prox['total_pnl_usd']}")
        _passed(f"proximal headline = ${sb_prox['total_pnl_usd']} (window_end excluded)")

        # window_end must NOT inflate the trade count either: 2 resolved, not 3.
        if sb_prox["trades"] != 2:
            _failed(f"prox trade count expected 2 (resolved only), "
                    f"got {sb_prox['trades']}")
        _passed(f"proximal trade count = {sb_prox['trades']} (resolved only)")

        if sb_mid["total_pnl_usd"] != 0.0:
            _failed(f"50pct headline expected $0, got ${sb_mid['total_pnl_usd']}")
        _passed(f"50pct headline = ${sb_mid['total_pnl_usd']}")

        # window_end stays VISIBLE in the exit-reason counts -- audit-only,
        # not hidden. The veteran's requirement: trader must see how many
        # trades didn't resolve before trusting the win rate.
        ec = s["exit_reason_counts_proximal"]
        if ec.get("window_end", 0) != 1:
            _failed(f"window_end must remain counted for audit, got {ec}")
        _passed(f"window_end still visible in exit-reason counts ({ec.get('window_end')})")


# ---------------------------------------------------------------------------
# Test 2: registry primary headline matches summary.json proximal_realised.
# Catches the bug where update_registry pointed at the tp2 hypothetical.
# ---------------------------------------------------------------------------
def test_registry_uses_realised_not_tp2():
    print("\n== test_registry_uses_realised_not_tp2 ==")

    # Construct a summary where realised and tp2 deliberately differ. If the
    # extractor reads the wrong key, the win_rate/expectancy will mismatch.
    summary = {
        "total_trade_rows": 4,
        "fill_rate_proximal": {"alerts": 4, "filled": 2, "fill_rate_pct": 50.0},
        "fill_rate_50pct":    {"alerts": 4, "filled": 1, "fill_rate_pct": 25.0},
        "scoreboards": {
            "proximal_realised":  {"trades": 2, "win_rate_pct": 50.0, "expectancy_r": 0.5},
            "proximal_exit_tp1":  {"trades": 2, "win_rate_pct": 100.0, "expectancy_r": 1.0},
            "proximal_exit_tp2":  {"trades": 2, "win_rate_pct": 100.0, "expectancy_r": 2.0},
            "fifty_pct_realised": {"trades": 1, "win_rate_pct": 0.0,  "expectancy_r": -1.0},
        },
        "per_pair_proximal_realised":     [{"pair": "EURUSD", "trades": 2}],
        "score_buckets_proximal_realised":[{"score_bucket": "3-4", "trades": 2}],
    }

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "h1only_20240819_20240823"
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(json.dumps(summary))
        # No trades.csv -- _extract_metrics tolerates that.

        m = _extract_metrics(run_dir, summary)
        # MUST be the realised values (0.5, 50%), NOT the tp2 ones (2.0, 100%).
        if m["expectancy_r"] != 0.5:
            _failed(f"registry expectancy expected 0.5 (realised), got "
                    f"{m['expectancy_r']} -- registry is reading the wrong board")
        _passed(f"registry expectancy = {m['expectancy_r']} (realised)")
        if m["win_rate_pct"] != 50.0:
            _failed(f"registry WR expected 50% (realised), got {m['win_rate_pct']}")
        _passed(f"registry WR = {m['win_rate_pct']}% (realised)")
        if len(m["per_pair"]) != 1:
            _failed(f"per_pair empty -- registry reading wrong key ({m['per_pair']})")
        _passed(f"per_pair populated from realised key ({len(m['per_pair'])} rows)")
        if len(m["score_buckets"]) != 1:
            _failed(f"score_buckets empty -- registry reading wrong key")
        _passed(f"score_buckets populated from realised key")


# ---------------------------------------------------------------------------
# Test 3: Excel Dollar P&L equals r_realised * risk_usd, not r * pnl_usd.
# ---------------------------------------------------------------------------
def test_excel_dollar_pnl_formula():
    print("\n== test_excel_dollar_pnl_formula ==")

    # +6.88R at $250 risk MUST render as $1,720. Previous bug rendered
    # r * pnl_usd = 6.88 * 1720 = $11,833.
    trades = [
        _mk_trade("EURUSD", "proximal", 6.88, 1.0, 6.88, exit_reason="tp2"),
        _mk_trade("EURUSD", "50pct",   -1.0, -1.0, -1.0, exit_reason="sl"),
    ]
    pivot_df = h1_only_reporting._build_zone_register_df(trades)
    if pivot_df.empty:
        _failed("pivoted excel df empty")
    row = pivot_df.iloc[0]
    expected = round(6.88 * 250, 0)
    actual = row["Proximal Dollar P&L"]
    if actual != expected:
        _failed(f"Proximal Dollar P&L: expected ${expected}, got ${actual} "
                f"(would be ${round(6.88 * 6.88 * 250)} under the old bug)")
    _passed(f"Proximal Dollar P&L = ${actual} (correct, not the bug value)")

    mid_actual = row["50% Dollar P&L"]
    mid_expected = round(-1.0 * 250, 0)
    if mid_actual != mid_expected:
        _failed(f"50% Dollar P&L: expected ${mid_expected}, got ${mid_actual}")
    _passed(f"50% Dollar P&L = ${mid_actual}")


if __name__ == "__main__":
    test_headline_matches_per_trade_sum()
    test_registry_uses_realised_not_tp2()
    test_excel_dollar_pnl_formula()
    print("\n=== All P&L reconciliation tests passed ===")
