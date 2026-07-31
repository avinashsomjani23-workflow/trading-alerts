"""
COLUMN BUCKETS generator — the sure-shot "no column missed" machine.

Reads the ONE canonical trades.csv header (via CANONICAL.md, never a glob) and
writes COLUMN_BUCKETS.md: every column, tagged by WHEN its value is knowable —

    alert    : known when the alert fires (a live scorer could compute it).
    fill     : known only if/when the limit fills.
    outcome  : known only AFTER the trade runs (look-ahead if used to filter entries).

The doc is two lists:
    SAFE FOR AN ENTRY FILTER  = alert + fill columns.
    OUTCOME / LOOK-AHEAD       = outcome columns.

Outcome columns are NOT off-limits: they are the best tool for finding patterns in
losers (autopsy). The ONE rule is: they may describe/group losers freely, but must
never drive a live entry filter — live, you do not know them yet.

WHY GENERATED, NOT HAND-WRITTEN: a hand list rots. This reads the live header, so a
new column cannot silently go missing. If a column is not in any explicit set below,
this RAISES — an unclassified column is a hard error, never a silent "assume safe".
That default is deliberate: silence is exactly the contamination we are killing.

Run:  python -m backtest.gen_column_buckets          # rewrite the doc
      python -m backtest.gen_column_buckets --check   # CI: fail if doc is stale
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DOC = _ROOT / "COLUMN_BUCKETS.md"


def _canonical_csv() -> Path:
    """Resolve the ONE canonical trades.csv from CANONICAL.md (never glob)."""
    doc = (_ROOT / "backtest" / "results" / "CANONICAL.md").read_text(encoding="utf-8")
    m = re.search(r"^\s{2,}(backtest/results/\S+/trades\.csv)\s*$", doc, re.M)
    if not m:
        raise SystemExit("canonical trades.csv path not found in CANONICAL.md — format changed")
    p = _ROOT / m.group(1)
    if not p.exists():
        raise SystemExit(f"canonical CSV named in CANONICAL.md does not exist: {p}")
    return p


def _header() -> list[str]:
    with _canonical_csv().open(newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


# ── The explicit timing sets. These are the ONLY judgment in the system. ─────
# Everything not named here that is not caught by a rule below is a HARD ERROR
# (see classify), so no column can slip through unclassified.

# OUTCOME = knowable only after the trade runs. The look-ahead set. These may
# describe losers but must never drive a live entry filter.
OUTCOME = {
    # realised result + P&L (exit_ts is outcome-time: the exit clock is only known
    # once the trade ends, same as exit_price/exit_reason). FIXED_2R_BASELINE
    # 2026-07-31: the r_if_exit_tp1/tp2 hypothetical columns are retired.
    "exit_ts", "exit_reason", "exit_price", "r_realised", "pnl_usd",
    # excursions and how the trade ran (mfe_r/mae_r are now FULL-WINDOW, A3 decouple)
    "mfe_r", "mae_r", "r_capture_ratio",
    "bars_to_exit", "bars_to_mfe", "bars_to_mae",
    # stop anatomy — all measured during/after the trade. Re-anchored to the fixed
    # 2R / 1R targets (FIXED_2R_BASELINE_SPEC A5); both readings kept.
    "sl_bar_was_sweep", "sl_swept_then_2r", "sl_swept_then_1r", "sl_wick_depth_atr",
    "sl_max_adverse_after_sweep_atr", "bars_sl_to_2r_touch", "bars_sl_to_1r_touch",
    "sl_recovered_to_entry", "sl_collision",
}

# FILL = knowable only if/when the limit fills (fill-anchored). Usable for entry
# analysis (the alert/fill gate is flexible), but MUST be labelled fill — a live
# alert-time scorer cannot see these.
FILL = {
    "fill_ts", "fill_session", "fill_in_killzone", "killzone_alignment",
    "ob_to_fill_hours", "bars_break_to_pullback", "sl_distance_atr",
    "atr_at_fill", "atr_regime_pct_at_fill", "atr_regime_ratio_at_fill",
    "weekend_blocked",
    # PD/PW pool status + nearest-pool + draw-on-liquidity (fill-anchored 2026-07-16)
    "day_state_at_fill", "pdh_status_at_fill", "pdl_status_at_fill",
    "pwh_status_at_fill", "pwl_status_at_fill",
    "dist_next_pool_above_atr", "dist_next_pool_below_atr",
    "next_pool_above_tier", "next_pool_below_tier", "trade_toward_pool",
    "last_sweep_age_h1", "last_sweep_tier",
    # EQ clusters (fill-anchored) — draw-toward + instant-death stop-in-pool test
    "eq_trade_toward", "eq_sl_gap_atr", "eq_sl_at_risk",
    "eq_intact_above_count", "eq_intact_below_count",
    # approach quality — the journey into the zone over bars before the fill
    "approach_speed_atr_at_fill", "approach_body_ratio_at_fill", "approach_er_at_fill",
    # sweep2 PD/PW legs roll with the calendar → judged at the fill candle
    "sweep2_pw_present", "sweep2_pw_pierce_atr", "sweep2_pw_rejection_ratio",
    "sweep2_pw_follow_atr", "sweep2_pw_rn_aligned",
    "sweep2_pd_present", "sweep2_pd_pierce_atr", "sweep2_pd_rejection_ratio",
    "sweep2_pd_follow_atr", "sweep2_pd_rn_aligned",
    "sweep2_age_at_fill_h1",
    # post-run news enrichment anchored on fill/exit ts
    "news_fill", "news_fill_event", "news_fill_ccy", "news_open", "news_open_event",
}

# EXECUTION / BOOKKEEPING — identity, raw timestamps, spread-placed price LEVELS,
# eligibility flags, config labels. Knowable at alert (so entry-safe), but they are
# not signals; still listed so nothing is missed. Suffix rules below also route the
# honest *_at_alert / *_at_ob names to alert.
ALERT_BOOKKEEPING = {
    "setup_id", "pair", "model", "alert_ts", "alert_bar_ts", "alert_seq",
    "ob_timestamp", "bos_timestamp", "direction", "bias",
    "entry_zone", "entry", "entry_raw", "sl_initial",
    # tp_2r = the fixed 2R target (FIXED_2R_BASELINE_SPEC 2026-07-31). Computed from
    # entry & sl (both alert-known), so it is an alert-time level like the retired tp1.
    "tp_2r",
    "eligible_for_headline", "headline_exclusion", "killzone_windows",
}


def classify(col: str) -> str:
    """alert | fill | outcome. Unknown column -> hard error (never silent-safe)."""
    if col in OUTCOME:
        return "outcome"
    if col in FILL:
        return "fill"
    if col in ALERT_BOOKKEEPING:
        return "alert"
    # Honest timing suffixes route deterministically. Audited 2026-07-30: every CSV
    # column carrying one of these suffixes is stamped consistent with its name.
    if col.endswith("_at_fill"):
        return "fill"
    if col.endswith(("_at_alert", "_at_ob")):
        return "alert"
    # Everything remaining is an alert-time detection/structure/score column. But we
    # do NOT blanket-assume: the ledger-verified alert set is enumerated so a genuinely
    # new, unclassified column trips the error below instead of hiding as "alert".
    if col in _ALERT_KNOWN:
        return "alert"
    raise KeyError(
        f"column {col!r} is not in any timing set — classify it in "
        "backtest/gen_column_buckets.py before it can enter analysis. "
        "(An unclassified column is a hard error, never a silent 'safe'.)"
    )


# The enumerated alert-time columns (detection / structure / score / levels-RR /
# alert-anchored context). Kept explicit so a new column errors rather than defaults.
_ALERT_KNOWN = {
    "session", "event", "reversal_pct", "reversed_from_extreme", "pd_zone",
    "pd_alignment", "pd_pct", "score", "structure_pts", "sweep_pts", "fvg_pts",
    "freshness_pts", "killzone_pts", "confluences_present", "bos_tag", "bos_tier",
    "event_candle_delta", "bos_verdict", "bos_sequence_count", "fvg_present",
    "fvg_mitigation", "fvg_state", "sweep_present", "ob_touches", "break_close_atr",
    "break_body_atr", "break_excess", "break_tier", "is_mss", "ob_range_atr",
    "fvg_size_atr", "impulse_leg_to_extreme_atr", "atr_at_ob", "ob_body_ratio",
    "ob_walkback_depth", "ob_age_h1_bars", "alert_utc_hour", "h1_trend",
    "trend_alignment", "trend_pd_agree", "ob_session", "ob_in_killzone",
    "setup_badge", "setup_badge_kind", "leg_extreme_clipped",
    # (tp1_rr / tp2_rr / *_wick_rr / tp_nextpool_rr retired 2026-07-31 — the
    # liquidity-pool RR columns are gone under the fixed 2R baseline.)
    # sweep2 SW + EQ legs freeze at OB birth -> alert-time
    "sweep2_present", "sweep2_pools_swept", "sweep2_tiers_checked",
    "sweep2_sw_present", "sweep2_sw_pierce_atr", "sweep2_sw_rejection_ratio",
    "sweep2_sw_follow_atr", "sweep2_sw_rn_aligned", "sweep2_eq_present",
    "sweep2_eq_pierce_atr", "sweep2_eq_rejection_ratio", "sweep2_eq_follow_atr",
    "sweep2_eq_rn_aligned", "sweep2_eq_size",
    # setup-liquidity level checks, computed with the alert-time levels
    "setup_liq_stop_present", "setup_liq_stop_offset_atr", "setup_liq_stop_tier",
    "setup_liq_tp_present", "setup_liq_tp_offset_atr", "setup_liq_legextreme_swept",
    # session-level sweep read (alert-anchored)
    "session_level_event", "session_level_which", "session_level_side",
    "session_level_pair_relevant",
    # news / ist / killzone blocks — computed from the frozen alert_ts
    "news_blocked", "news_event_title", "news_event_currency", "news_event_source",
    "news_event_ts", "ist_blocked", "killzone_blocked",
    # eq last-sweep read on closed bars before ALERT
    "eq_last_sweep_age_h1", "eq_last_sweep_side",
    # EQH/EQL shelf distances + sizes — computed from H1 bars strictly BEFORE
    # alert_ts (ledger: immutable price history, no yield freeze) -> alert-time
    "eqh_above_dist_atr", "eqh_above_size", "eql_below_dist_atr", "eql_below_size",
    # weekly PD — bars strictly before alert_ts -> alert-time
    "weekly_pd_position_at_alert", "weekly_range_high_at_alert",
    "weekly_range_low_at_alert", "weekly_pd_zone_at_alert", "pd_zone_agreement_at_alert",
    # structure-state + displacement-leg snapshots at the alert yield
    "flip_pending_at_alert", "flip_pending_dir_at_alert",
    "leg_extreme_at_alert", "leg_er_at_alert",
    # dealing-range break flags frozen at OB build; distance-to-level reads at alert
    "dr_ceiling_broken_at_ob", "dr_floor_broken_at_ob",
    "sl_dist_atr_at_alert", "tp_dist_atr_at_alert", "chop_at_alert",
}


_TIMING_NOTE = {
    "alert": "known when the alert fires — safe for the live entry filter",
    "fill": "known only once the limit fills — usable for entry analysis, but label it FILL",
    "outcome": "known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter",
}


def build_doc() -> str:
    header = _header()
    rows = [(c, classify(c)) for c in header]
    safe = [(c, t) for c, t in rows if t in ("alert", "fill")]
    outcome = [c for c, t in rows if t == "outcome"]

    n_alert = sum(t == "alert" for _, t in rows)
    n_fill = sum(t == "fill" for _, t in rows)

    L: list[str] = []
    ap = L.append
    ap("# COLUMN BUCKETS — read before ANY analysis")
    ap("")
    ap("> **GENERATED — do not hand-edit.** Rebuild: `python -m backtest.gen_column_buckets`.")
    ap("> To reclassify a column, edit the sets in `backtest/gen_column_buckets.py`, not this file.")
    ap("")
    ap("## The one rule")
    ap("")
    ap("- Running a loser / entry analysis? Use **every** column below — none is skipped.")
    ap("- Each column is tagged by **when its value is knowable**: `alert`, `fill`, `outcome`.")
    ap("- **`outcome` columns are look-ahead.** They are your best tool to *find patterns in "
       "losers* (autopsy — \"losers die in 3 bars\"), so use them freely to DESCRIBE / group losers. "
       "They must **never** drive a *live entry filter* — live, you do not know them yet.")
    ap("- **`fill` columns** are usable for entry analysis, but ALWAYS say \"this is fill-time\" — "
       "a live alert-time scorer cannot see them.")
    ap("- No column may enter analysis unclassified. A new column not in a set makes the generator "
       "RAISE and CI go red — that is the \"nothing missed, nothing contaminated\" guarantee.")
    ap("")
    ap(f"Canonical: `{_canonical_csv().relative_to(_ROOT).as_posix()}` — "
       f"**{len(header)} columns** ({n_alert} alert, {n_fill} fill, {len(outcome)} outcome).")
    ap("")
    ap("---")
    ap("")
    ap(f"## ✅ SAFE FOR AN ENTRY FILTER — {len(safe)} columns (alert + fill)")
    ap("")
    ap("A live scorer could act on these. `fill` ones are only known once the limit fills — "
       "flag that whenever you use one.")
    ap("")
    ap("| column | timing | meaning |")
    ap("|---|---|---|")
    for c, t in safe:
        ap(f"| `{c}` | {t} | {_TIMING_NOTE[t]} |")
    ap("")
    ap(f"## 🚩 OUTCOME / LOOK-AHEAD — {len(outcome)} columns")
    ap("")
    ap("Known only after the trade runs. **Describe losers with these; never filter entries on them.**")
    ap("")
    ap("| column | timing | meaning |")
    ap("|---|---|---|")
    for c in outcome:
        ap(f"| `{c}` | outcome | {_TIMING_NOTE['outcome']} |")
    ap("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="CI mode: exit 1 if COLUMN_BUCKETS.md is missing or stale")
    args = ap.parse_args()

    want = build_doc()  # also validates: every column classifies or this raises
    if args.check:
        have = _DOC.read_text(encoding="utf-8") if _DOC.exists() else ""
        if have != want:
            print("COLUMN_BUCKETS.md is STALE — run: python -m backtest.gen_column_buckets",
                  file=sys.stderr)
            return 1
        print("COLUMN_BUCKETS.md up to date.")
        return 0
    _DOC.write_text(want, encoding="utf-8")
    print(f"wrote {_DOC.relative_to(_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
