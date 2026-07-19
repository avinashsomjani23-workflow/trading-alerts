"""Guard tests for zone selection + the ONE-TRADE-PER-ZONE gate:

  1. _split_primary_alternative is MODE-SPLIT:
       cap=False (BACKTEST) -> every OB surfaces, no 'role'.
       cap=True  (LIVE, default) -> at most 2 OBs (primary + alternative),
                 'role' stamped. The default MUST be cap=True so live never
                 silently uncaps. replay_engine must call cap_zones=False.
  2. The backtest run paths (run_backtest.py, diagnostics/driver.py) carry a
     ONE-TRADE-PER-ZONE gate (2026-07-19): once a (ob_timestamp, direction) zone
     has produced a FILLED trade, later alerts from that zone are dropped, so one
     open position can never be booked twice. The gate is FILL-based (`filled_obs`),
     NOT the pre-2026-07-15 first-alert `seen_obs` dedupe — a non-filling alert must
     never mark the zone spent (else the real, later-filling trade is lost). This
     guard fails if either the gate is missing OR the old alert-based `seen_obs`
     dedupe is reintroduced.

These are OUT of the live trade path (offline asserts), per the guard rule:
a guard must never break the thing it protects.

Run:  python -m pytest tests/test_retouch_trading.py -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import smc_radar  # noqa: E402


def _ob(direction, proximal, bos_idx, touches=0, atr=1.0):
    """Minimal OB dict with the fields _split_primary_alternative reads."""
    return {
        "direction": direction,
        "proximal_line": float(proximal),
        "distal_line": float(proximal) + (0.5 if direction == "bearish" else -0.5),
        "bos_idx": int(bos_idx),
        "touches": int(touches),
        "h1_atr": float(atr),
    }


# --- 1) BACKTEST (cap=False): every OB survives, no cap -------------------
def test_backtest_uncapped_all_obs_surface():
    cur = 100.0
    obs = [
        _ob("bullish", 99.5, bos_idx=10),
        _ob("bullish", 99.0, bos_idx=11),
        _ob("bearish", 101.0, bos_idx=12),
        _ob("bearish", 102.0, bos_idx=13),
        _ob("bullish", 98.0, bos_idx=14),
    ]
    kept = smc_radar._split_primary_alternative(obs, cur, trend="bullish", cap=False)
    assert len(kept) == len(obs), (
        f"backtest must be uncapped: {len(obs)} in, only {len(kept)} out — "
        f"every un-mitigated OB must surface when cap=False"
    )


# --- 2) BACKTEST (cap=False): no 'role' stamped ---------------------------
def test_backtest_uncapped_no_role_stamped():
    cur = 100.0
    obs = [_ob("bullish", 99.5, 10), _ob("bearish", 101.0, 11)]
    kept = smc_radar._split_primary_alternative(obs, cur, trend="bullish", cap=False)
    for o in kept:
        assert "role" not in o, (
            "cap=False must not stamp 'role' (no primary/alternative labelling "
            "in the uncapped backtest view)"
        )


# --- 3) LIVE (cap=True): at most 2 OBs, role stamped ----------------------
def test_live_capped_at_two_with_role():
    cur = 100.0
    obs = [
        _ob("bullish", 99.5, bos_idx=10),
        _ob("bullish", 99.0, bos_idx=11),
        _ob("bearish", 101.0, bos_idx=12),
        _ob("bullish", 98.0, bos_idx=14),
    ]
    kept = smc_radar._split_primary_alternative(obs, cur, trend="bullish", cap=True)
    assert len(kept) <= 2, (
        f"live cap regression: expected <=2 zones, got {len(kept)}"
    )
    roles = {o.get("role") for o in kept}
    assert "primary" in roles, "live cap must stamp a 'primary' zone"
    # OB1 = last-event (highest bos_idx)
    primary = next(o for o in kept if o.get("role") == "primary")
    assert primary["bos_idx"] == 14, (
        f"primary must be last-event (highest bos_idx); got {primary['bos_idx']}"
    )


# --- 4) cap=True is the DEFAULT (live safety) -----------------------------
def test_cap_true_is_default():
    cur = 100.0
    obs = [_ob("bullish", 99.5, i) for i in range(10, 16)]  # 6 OBs
    kept = smc_radar._split_primary_alternative(obs, cur)  # no cap arg
    assert len(kept) <= 2, (
        "default (no cap arg) must behave as live cap=True (<=2 zones) — a "
        "missing default would silently uncap live"
    )


# --- 5) Distance still stamped (both modes) -------------------------------
def test_distance_still_stamped():
    cur = 100.0
    obs = [_ob("bullish", 99.0, 10, atr=2.0)]  # 1.0 away, atr 2.0 -> 0.5 ATR
    kept = smc_radar._split_primary_alternative(obs, cur, cap=False)
    assert kept[0]["_distance_atr"] == 0.5, (
        f"_distance_atr must still be stamped for display/logging; got "
        f"{kept[0].get('_distance_atr')}"
    )


# --- 6) Empty input is safe -----------------------------------------------
def test_empty_input():
    assert smc_radar._split_primary_alternative([], 100.0) == []


# --- 7) Backtest detection call passes cap_zones=False --------------------
def test_backtest_calls_uncapped():
    src = _read("backtest/replay_engine.py")
    assert "cap_zones=False" in src, (
        "replay_engine must call detect_smc_radar(cap_zones=False) so the "
        "backtest sees every OB"
    )


# --- Source guards: ONE-TRADE-PER-ZONE gate present, fill-based ------------
def _read(rel):
    return (_ROOT / rel).read_text(encoding="utf-8-sig")


def _assert_fill_based_zone_gate(src, path):
    """The run path must (a) carry a fill-based `filled_obs` zone gate and
    (b) NOT reintroduce the old alert-based `seen_obs` dedupe."""
    assert re.search(r"filled_obs\s*:\s*set\s*=", src), (
        f"{path}: ONE-TRADE-PER-ZONE gate missing — `filled_obs` set not "
        f"constructed. Without it a re-alert double-books an open position."
    )
    assert "if ob_key in filled_obs" in src, (
        f"{path}: `filled_obs` membership skip missing — later alerts on a "
        f"filled zone are not dropped."
    )
    # The gate must mark the zone only on a real fill (fill_ts), never on the
    # first alert — the whole point of gating on fill, not alert.
    assert "fill_ts" in src, (
        f"{path}: zone must be marked spent on fill_ts, not on first alert."
    )
    # The old alert-based dedupe must not come back (it loses a non-filling
    # alert #1's zone -> the later filling alert #2 would be wrongly skipped).
    assert not re.search(r"seen_obs\s*:\s*set\s*=", src), (
        f"{path}: old alert-based `seen_obs` dedupe reintroduced — it marks a "
        f"zone spent on the first alert even when it never fills, losing the "
        f"real later trade. Use the fill-based `filled_obs` gate instead."
    )


def test_run_backtest_has_fill_based_zone_gate():
    _assert_fill_based_zone_gate(_read("backtest/run_backtest.py"),
                                 "run_backtest.py")


def test_driver_has_fill_based_zone_gate():
    _assert_fill_based_zone_gate(_read("backtest/diagnostics/driver.py"),
                                 "diagnostics/driver.py")


if __name__ == "__main__":
    test_backtest_uncapped_all_obs_surface()
    test_backtest_uncapped_no_role_stamped()
    test_live_capped_at_two_with_role()
    test_cap_true_is_default()
    test_distance_still_stamped()
    test_empty_input()
    test_backtest_calls_uncapped()
    test_run_backtest_has_fill_based_zone_gate()
    test_driver_has_fill_based_zone_gate()
    print("all retouch-trading guards passed")
