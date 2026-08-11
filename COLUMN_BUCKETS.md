# COLUMN BUCKETS — read before ANY analysis

> **GENERATED — do not hand-edit.** Rebuild: `python -m backtest.gen_column_buckets`.
> To reclassify a column, edit the sets in `backtest/gen_column_buckets.py`, not this file.

## The one rule

- Running a loser / entry analysis? Use **every** column below — none is skipped.
- Each column is tagged by **when its value is knowable**: `alert`, `fill`, `outcome`.
- **`outcome` columns are look-ahead.** They are your best tool to *find patterns in losers* (autopsy — "losers die in 3 bars"), so use them freely to DESCRIBE / group losers. They must **never** drive a *live entry filter* — live, you do not know them yet.
- **`fill` columns** are usable for entry analysis, but ALWAYS say "this is fill-time" — a live alert-time scorer cannot see them.
- No column may enter analysis unclassified. A new column not in a set makes the generator RAISE and CI go red — that is the "nothing missed, nothing contaminated" guarantee.

Canonical: `backtest/results/h1only_20080102_20161231/trades.csv` — **184 columns** (113 alert, 47 fill, 24 outcome).

---

## ✅ SAFE FOR AN ENTRY FILTER — 160 columns (alert + fill)

A live scorer could act on these. `fill` ones are only known once the limit fills — flag that whenever you use one.

| column | timing | meaning |
|---|---|---|
| `setup_id` | alert | known when the alert fires — safe for the live entry filter |
| `pair` | alert | known when the alert fires — safe for the live entry filter |
| `alert_ts` | alert | known when the alert fires — safe for the live entry filter |
| `fill_ts` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `session` | alert | known when the alert fires — safe for the live entry filter |
| `direction` | alert | known when the alert fires — safe for the live entry filter |
| `event` | alert | known when the alert fires — safe for the live entry filter |
| `entry_zone` | alert | known when the alert fires — safe for the live entry filter |
| `entry` | alert | known when the alert fires — safe for the live entry filter |
| `sl_initial` | alert | known when the alert fires — safe for the live entry filter |
| `tp_2r` | alert | known when the alert fires — safe for the live entry filter |
| `eligible_for_headline` | alert | known when the alert fires — safe for the live entry filter |
| `headline_exclusion` | alert | known when the alert fires — safe for the live entry filter |
| `sl_distance_atr` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `sl_dist_atr_at_alert` | alert | known when the alert fires — safe for the live entry filter |
| `tp_dist_atr_at_alert` | alert | known when the alert fires — safe for the live entry filter |
| `ob_to_fill_hours` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `bars_break_to_pullback` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `ob_age_h1_bars` | alert | known when the alert fires — safe for the live entry filter |
| `pd_zone` | alert | known when the alert fires — safe for the live entry filter |
| `reversal_pct` | alert | known when the alert fires — safe for the live entry filter |
| `reversed_from_extreme` | alert | known when the alert fires — safe for the live entry filter |
| `score` | alert | known when the alert fires — safe for the live entry filter |
| `structure_pts` | alert | known when the alert fires — safe for the live entry filter |
| `sweep_pts` | alert | known when the alert fires — safe for the live entry filter |
| `fvg_pts` | alert | known when the alert fires — safe for the live entry filter |
| `freshness_pts` | alert | known when the alert fires — safe for the live entry filter |
| `killzone_pts` | alert | known when the alert fires — safe for the live entry filter |
| `confluences_present` | alert | known when the alert fires — safe for the live entry filter |
| `model` | alert | known when the alert fires — safe for the live entry filter |
| `ob_timestamp` | alert | known when the alert fires — safe for the live entry filter |
| `bos_tag` | alert | known when the alert fires — safe for the live entry filter |
| `bos_tier` | alert | known when the alert fires — safe for the live entry filter |
| `event_candle_delta` | alert | known when the alert fires — safe for the live entry filter |
| `bos_verdict` | alert | known when the alert fires — safe for the live entry filter |
| `fvg_present` | alert | known when the alert fires — safe for the live entry filter |
| `fvg_mitigation` | alert | known when the alert fires — safe for the live entry filter |
| `sweep_present` | alert | known when the alert fires — safe for the live entry filter |
| `ob_touches` | alert | known when the alert fires — safe for the live entry filter |
| `break_close_atr` | alert | known when the alert fires — safe for the live entry filter |
| `break_body_atr` | alert | known when the alert fires — safe for the live entry filter |
| `break_excess` | alert | known when the alert fires — safe for the live entry filter |
| `break_tier` | alert | known when the alert fires — safe for the live entry filter |
| `is_mss` | alert | known when the alert fires — safe for the live entry filter |
| `ob_range_atr` | alert | known when the alert fires — safe for the live entry filter |
| `fvg_size_atr` | alert | known when the alert fires — safe for the live entry filter |
| `impulse_leg_to_extreme_atr` | alert | known when the alert fires — safe for the live entry filter |
| `atr_at_ob` | alert | known when the alert fires — safe for the live entry filter |
| `atr_at_fill` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `ob_body_ratio` | alert | known when the alert fires — safe for the live entry filter |
| `ob_walkback_depth` | alert | known when the alert fires — safe for the live entry filter |
| `chop_at_alert` | alert | known when the alert fires — safe for the live entry filter |
| `news_blocked` | alert | known when the alert fires — safe for the live entry filter |
| `news_event_title` | alert | known when the alert fires — safe for the live entry filter |
| `news_event_currency` | alert | known when the alert fires — safe for the live entry filter |
| `news_event_source` | alert | known when the alert fires — safe for the live entry filter |
| `news_event_ts` | alert | known when the alert fires — safe for the live entry filter |
| `ist_blocked` | alert | known when the alert fires — safe for the live entry filter |
| `alert_utc_hour` | alert | known when the alert fires — safe for the live entry filter |
| `h1_trend` | alert | known when the alert fires — safe for the live entry filter |
| `trend_alignment` | alert | known when the alert fires — safe for the live entry filter |
| `trend_pd_agree` | alert | known when the alert fires — safe for the live entry filter |
| `flip_pending_at_alert` | alert | known when the alert fires — safe for the live entry filter |
| `flip_pending_dir_at_alert` | alert | known when the alert fires — safe for the live entry filter |
| `leg_extreme_at_alert` | alert | known when the alert fires — safe for the live entry filter |
| `leg_extreme_clipped` | alert | known when the alert fires — safe for the live entry filter |
| `dr_ceiling_broken_at_ob` | alert | known when the alert fires — safe for the live entry filter |
| `dr_floor_broken_at_ob` | alert | known when the alert fires — safe for the live entry filter |
| `day_state_at_fill` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `pdh_status_at_fill` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `pdl_status_at_fill` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `pwh_status_at_fill` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `pwl_status_at_fill` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `dist_next_pool_above_atr` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `dist_next_pool_below_atr` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `next_pool_above_tier` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `next_pool_below_tier` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `trade_toward_pool` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `last_sweep_age_h1` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `last_sweep_tier` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `eqh_above_dist_atr` | alert | known when the alert fires — safe for the live entry filter |
| `eqh_above_size` | alert | known when the alert fires — safe for the live entry filter |
| `eql_below_dist_atr` | alert | known when the alert fires — safe for the live entry filter |
| `eql_below_size` | alert | known when the alert fires — safe for the live entry filter |
| `eq_trade_toward` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `eq_sl_gap_atr` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `eq_sl_at_risk` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `eq_last_sweep_age_h1` | alert | known when the alert fires — safe for the live entry filter |
| `eq_last_sweep_side` | alert | known when the alert fires — safe for the live entry filter |
| `eq_intact_above_count` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `eq_intact_below_count` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `sweep2_present` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_pools_swept` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_age_at_fill_h1` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `sweep2_tiers_checked` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_sw_present` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_sw_pierce_atr` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_sw_rejection_ratio` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_sw_follow_atr` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_sw_rn_aligned` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_eq_present` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_eq_pierce_atr` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_eq_rejection_ratio` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_eq_follow_atr` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_eq_rn_aligned` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_eq_size` | alert | known when the alert fires — safe for the live entry filter |
| `sweep2_pw_present` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `sweep2_pw_pierce_atr` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `sweep2_pw_rejection_ratio` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `sweep2_pw_follow_atr` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `sweep2_pw_rn_aligned` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `sweep2_pd_present` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `sweep2_pd_pierce_atr` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `sweep2_pd_rejection_ratio` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `sweep2_pd_follow_atr` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `sweep2_pd_rn_aligned` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `setup_liq_stop_present` | alert | known when the alert fires — safe for the live entry filter |
| `setup_liq_stop_offset_atr` | alert | known when the alert fires — safe for the live entry filter |
| `setup_liq_stop_tier` | alert | known when the alert fires — safe for the live entry filter |
| `setup_liq_tp_present` | alert | known when the alert fires — safe for the live entry filter |
| `setup_liq_tp_offset_atr` | alert | known when the alert fires — safe for the live entry filter |
| `setup_liq_legextreme_swept` | alert | known when the alert fires — safe for the live entry filter |
| `session_level_event` | alert | known when the alert fires — safe for the live entry filter |
| `session_level_which` | alert | known when the alert fires — safe for the live entry filter |
| `session_level_side` | alert | known when the alert fires — safe for the live entry filter |
| `session_level_pair_relevant` | alert | known when the alert fires — safe for the live entry filter |
| `alert_bar_ts` | alert | known when the alert fires — safe for the live entry filter |
| `alert_seq` | alert | known when the alert fires — safe for the live entry filter |
| `bos_timestamp` | alert | known when the alert fires — safe for the live entry filter |
| `bias` | alert | known when the alert fires — safe for the live entry filter |
| `pd_alignment` | alert | known when the alert fires — safe for the live entry filter |
| `pd_pct` | alert | known when the alert fires — safe for the live entry filter |
| `weekend_blocked` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `bos_sequence_count` | alert | known when the alert fires — safe for the live entry filter |
| `atr_regime_pct_at_fill` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `atr_regime_ratio_at_fill` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `fvg_state` | alert | known when the alert fires — safe for the live entry filter |
| `ob_session` | alert | known when the alert fires — safe for the live entry filter |
| `fill_session` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `ob_in_killzone` | alert | known when the alert fires — safe for the live entry filter |
| `fill_in_killzone` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `killzone_alignment` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `leg_er_at_alert` | alert | known when the alert fires — safe for the live entry filter |
| `setup_badge` | alert | known when the alert fires — safe for the live entry filter |
| `setup_badge_kind` | alert | known when the alert fires — safe for the live entry filter |
| `weekly_pd_position_at_alert` | alert | known when the alert fires — safe for the live entry filter |
| `weekly_range_high_at_alert` | alert | known when the alert fires — safe for the live entry filter |
| `weekly_range_low_at_alert` | alert | known when the alert fires — safe for the live entry filter |
| `weekly_pd_zone_at_alert` | alert | known when the alert fires — safe for the live entry filter |
| `pd_zone_agreement_at_alert` | alert | known when the alert fires — safe for the live entry filter |
| `approach_speed_atr_at_fill` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `approach_body_ratio_at_fill` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `approach_er_at_fill` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `killzone_blocked` | alert | known when the alert fires — safe for the live entry filter |
| `killzone_windows` | alert | known when the alert fires — safe for the live entry filter |
| `news_fill` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `news_fill_event` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `news_fill_ccy` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `news_open` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |
| `news_open_event` | fill | known only once the limit fills — usable for entry analysis, but label it FILL |

## 🚩 OUTCOME / LOOK-AHEAD — 24 columns

Known only after the trade runs. **Describe losers with these; never filter entries on them.**

| column | timing | meaning |
|---|---|---|
| `exit_ts` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `exit_reason` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `exit_price` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `r_realised` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `pnl_usd` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `mfe_r` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `mae_r` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `r_capture_ratio` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `sl_bar_was_sweep` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `sl_swept_then_2r` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `sl_swept_then_1r` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `sl_wick_depth_atr` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `sl_max_adverse_after_sweep_atr` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `bars_sl_to_2r_touch` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `bars_sl_to_1r_touch` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `sl_recovered_to_entry` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `bars_to_exit` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `bars_to_mfe` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `bars_to_mae` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `sl_collision` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `mfe_intrade_r` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `sl_bar_best_favor_r` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `sl_bar_reached_1r_ambiguous` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |
| `ob_penetration_depth` | outcome | known only after the trade runs — LOOK-AHEAD: describe losers freely, never a live entry filter |

