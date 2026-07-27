# Edge Lab v2 — Step 2: Univariate Pooled Discovery Scorecard

- **Scope:** pooled_fx_gold (Book A + Book B; BTC deferred to Step 3)
- **Window:** 2008-01-02..2016-12-31 (DISCOVERY split; C5 holdout sealed)
- **Target:** `r_realised` (per-trade R, source of truth)
- **Seed:** 42 · **Generated:** 2026-07-09T09:05:34.162736+00:00
- **N (gates-off):** 11363 · **N (live-gated, score≥4):** 10748
- **Effect floor (informational):** 0.05R · **MI reliability floor:** subpop≥500

**Layer 1 = permissive discovery — ranks everything, kills nothing. Disposition is a triage into two visible buckets; the strict SHIP GATE and the holdout are LATER steps. Nothing here changes live trading.**

Test per type (spec §3): continuous → Spearman + MI + decile curve; ordinal → **Spearman (not Kruskal)**; nominal → Kruskal-Wallis; binary → bootstrap diff-CI. Effect = top-vs-bottom bucket ΔR. FDR & effect-floor columns are informational, not gates.

### View: gates_off

| # | feature | kind | timing | N | effect R | eff CI | sig | MI | test stat | p | FDR | consistency | disposition |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ob_walkback_depth | ordinal | alert_time | 11363 | -0.1933 | [-0.3395, -0.0394] | ✅ | n/a | ρ=-0.0195 | 0.0373 | — | — | 🟢 QUEUE |
| 2 | trend_alignment | nominal | alert_time | 11363 | 0.1207 | [0.0211, 0.2198] | ✅ | n/a | H=4.9603 | 0.0837 | — | 8/14 | · not proven |
| 3 | ob_to_fill_hours | continuous | fill_time | 11363 | 0.1132 | [0.0120, 0.2144] | ✅ | 0.00094 | ρ=-0.0015 | 0.8719 | — | 6/19 | · not proven |
| 4 | pair | nominal | alert_time | 11363 | 0.1017 | [0.0060, 0.1950] | ✅ | n/a | H=6.1256 | 0.6332 | — | 16/35 | · not proven |
| 5 | alert_utc_hour | continuous | alert_time | 11363 | 0.1050 | [-0.0045, 0.1827] | — | 0.00000 | ρ=0.0292 | 0.0018 | rej | 17/27 | · not proven |
| 6 | setup_badge | nominal | alert_time | 967 | 0.0963 | [-0.0641, 0.2553] | — | n/a | H=1.2248 | 0.5420 | — | 0/0 | · not proven |
| 7 | impulse_leg_atr | continuous | alert_time | 11363 | -0.0900 | [-0.1897, 0.0124] | — | 0.00414 | ρ=-0.0048 | 0.6064 | — | 11/23 | · not proven |
| 8 | leg_retrace_pct_at_alert | continuous | alert_time | 11058 | 0.0894 | [-0.0068, 0.1915] | — | 0.00115 | ρ=-0.0052 | 0.5857 | — | 11/20 | · not proven |
| 9 | ob_age_h1_bars | continuous | alert_time | 11363 | 0.0804 | [-0.0245, 0.1801] | — | 0.00043 | ρ=-0.0023 | 0.8061 | — | 7/19 | · not proven |
| 10 | fvg_size_atr | continuous | alert_time | 4424 | 0.0766 | [-0.0960, 0.2478] | — | 0.00000 | ρ=0.0054 | 0.7193 | — | 0/0 | · not proven |
| 11 | fill_session | nominal | fill_time | 11363 | 0.0712 | [-0.0600, 0.2059] | — | n/a | H=1.5630 | 0.6678 | — | 17/32 | · not proven |
| 12 | killzone_alignment | nominal | fill_time | 11363 | 0.0568 | [-0.0057, 0.1192] | — | n/a | H=11.4557 | 0.0095 | rej | 15/36 | · not proven |
| 13 | break_close_atr | continuous | alert_time | 11363 | -0.0545 | [-0.1644, 0.0388] | — | 0.00000 | ρ=-0.0284 | 0.0025 | rej | 4/20 | · not proven |
| 14 | bars_break_to_pullback | continuous | fill_time | 11363 | 0.0519 | [-0.0287, 0.1653] | — | 0.00296 | ρ=0.0062 | 0.5106 | — | 8/21 | · not proven |
| 15 | bos_tier | nominal | alert_time | 11363 | 0.0504 | [-0.0015, 0.1030] | — | n/a | H=8.6574 | 0.0342 | — | 11/36 | · not proven |
| 16 | event | nominal | alert_time | 11363 | 0.0504 | [-0.0015, 0.1030] | — | n/a | H=8.6574 | 0.0342 | — | 11/36 | · not proven |
| 17 | break_body_atr | continuous | alert_time | 11363 | -0.0482 | [-0.1400, 0.0702] | — | 0.00569 | ρ=-0.0214 | 0.0226 | — | 8/19 | · not proven |
| 18 | reversal_pct | binary | alert_time | 11363 | -0.0435 | [-0.0990, 0.0112] | — | n/a | bootstrap_diff_ci(binary) | — | — | 9/36 | · not proven |
| 19 | bos_tag | binary | alert_time | 11363 | -0.0404 | [-0.0900, 0.0107] | — | n/a | bootstrap_diff_ci(binary) | — | — | 10/36 | · not proven |
| 20 | fill_in_killzone | binary | fill_time | 11363 | -0.0388 | [-0.0855, 0.0072] | — | n/a | bootstrap_diff_ci(binary) | — | — | 15/36 | · not proven |
| 21 | trend_pd_agree | binary | alert_time | 11363 | 0.0385 | [-0.0105, 0.0881] | — | n/a | bootstrap_diff_ci(binary) | — | — | 13/36 | · not proven |
| 22 | pd_alignment | binary | alert_time | 11363 | -0.0367 | [-0.0827, 0.0098] | — | n/a | bootstrap_diff_ci(binary) | — | — | 12/36 | · not proven |
| 23 | score | continuous | alert_time | 11363 | -0.0363 | [-0.1116, 0.0548] | — | 0.00684 | ρ=-0.0085 | 0.3677 | — | 14/36 | · not proven |
| 24 | ob_session | nominal | alert_time | 11363 | 0.0343 | [-0.0762, 0.1459] | — | n/a | H=10.7190 | 0.0133 | rej | 0/0 | · not proven |
| 25 | fvg_state | nominal | alert_time | 11363 | 0.0333 | [-0.0651, 0.1304] | — | n/a | H=3.2715 | 0.1948 | — | 11/36 | · not proven |
| 26 | session | nominal | alert_time | 11363 | 0.0322 | [-0.0170, 0.0824] | — | n/a | H=5.2301 | 0.0732 | — | 13/36 | · not proven |
| 27 | flip_pending_at_alert | binary | alert_time | 11363 | -0.0260 | [-0.0777, 0.0248] | — | n/a | bootstrap_diff_ci(binary) | — | — | 9/36 | · not proven |
| 28 | fvg_mitigation | nominal | alert_time | 11363 | 0.0260 | [-0.0669, 0.1168] | — | n/a | H=3.2146 | 0.3597 | — | 11/31 | · not proven |
| 29 | structure_ranging_at_alert | binary | alert_time | 11363 | -0.0260 | [-0.1152, 0.0678] | — | n/a | bootstrap_diff_ci(binary) | — | — | 7/36 | · not proven |
| 30 | reversed_from_extreme | binary | alert_time | 3264 | -0.0206 | [-0.1140, 0.0747] | — | n/a | bootstrap_diff_ci(binary) | — | — | 4/7 | · not proven |
| 31 | pd_pct | continuous | alert_time | 11363 | -0.0196 | [-0.1244, 0.0776] | — | 0.00339 | ρ=-0.0016 | 0.8624 | — | 8/21 | · not proven |
| 32 | ob_in_killzone | binary | alert_time | 11363 | 0.0190 | [-0.0276, 0.0648] | — | n/a | bootstrap_diff_ci(binary) | — | — | 13/36 | · not proven |
| 33 | ob_range_atr | continuous | alert_time | 11363 | -0.0185 | [-0.1241, 0.0808] | — | 0.00000 | ρ=0.0380 | 0.0001 | rej | 6/22 | · not proven |
| 34 | ob_touches | ordinal | alert_time | 11363 | -0.0181 | [-0.1330, 0.0958] | — | n/a | ρ=0.0108 | 0.2513 | — | — | · not proven |
| 35 | dr_floor_broken_at_ob | binary | alert_time | 11241 | -0.0157 | [-0.0726, 0.0423] | — | n/a | bootstrap_diff_ci(binary) | — | — | 6/36 | · not proven |
| 36 | ob_body_ratio | continuous | alert_time | 11363 | 0.0156 | [-0.0867, 0.1193] | — | 0.00405 | ρ=0.0085 | 0.3661 | — | 10/20 | · not proven |
| 37 | dr_ceiling_broken_at_ob | binary | alert_time | 11241 | 0.0127 | [-0.0446, 0.0684] | — | n/a | bootstrap_diff_ci(binary) | — | — | 13/36 | · not proven |
| 38 | bias | binary | alert_time | 11363 | -0.0084 | [-0.0540, 0.0383] | — | n/a | bootstrap_diff_ci(binary) | — | — | 10/36 | · not proven |
| 39 | pd_zone | binary | alert_time | 11363 | -0.0077 | [-0.0533, 0.0378] | — | n/a | bootstrap_diff_ci(binary) | — | — | 8/36 | · not proven |
| 40 | atr_at_ob | continuous | alert_time | 11363 | -0.0075 | [-0.1001, 0.1047] | — | 0.00435 | ρ=-0.0034 | 0.7202 | — | 10/21 | · not proven |
| 41 | flip_pending_dir_at_alert | binary | alert_time | 3109 | -0.0056 | [-0.0946, 0.0819] | — | n/a | bootstrap_diff_ci(binary) | — | — | 12/36 | · not proven |
| 42 | fvg_present | binary | alert_time | 11363 | 0.0048 | [-0.0419, 0.0522] | — | n/a | bootstrap_diff_ci(binary) | — | — | 11/36 | · not proven |
| 43 | bos_sequence_count | ordinal | alert_time | 11363 | 0.0029 | [-0.1545, 0.1681] | — | n/a | ρ=0.0194 | 0.0383 | — | — | · not proven |
| 44 | bos_verdict | binary | alert_time | 11363 | 0.0024 | [-0.0984, 0.0971] | — | n/a | bootstrap_diff_ci(binary) | — | — | 9/36 | · not proven |

### View: live_gated

| # | feature | kind | timing | N | effect R | eff CI | sig | MI | test stat | p | FDR | consistency | disposition |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ob_walkback_depth | ordinal | alert_time | 10748 | -0.1856 | [-0.3393, -0.0262] | ✅ | n/a | ρ=-0.0211 | 0.0284 | rej | — | 🟢 QUEUE |
| 2 | trend_alignment | nominal | alert_time | 10748 | 0.1201 | [0.0206, 0.2193] | ✅ | n/a | H=4.7712 | 0.0920 | — | 6/12 | · not proven |
| 3 | pair | nominal | alert_time | 10748 | 0.0990 | [0.0050, 0.1952] | ✅ | n/a | H=5.9234 | 0.6558 | — | 20/35 | · not proven |
| 4 | ob_to_fill_hours | continuous | fill_time | 10748 | 0.0867 | [0.0023, 0.2110] | ✅ | 0.00311 | ρ=-0.0030 | 0.7548 | — | 7/19 | · not proven |
| 5 | alert_utc_hour | continuous | alert_time | 10748 | 0.0768 | [0.0021, 0.1983] | ✅ | 0.00018 | ρ=0.0352 | 0.0003 | rej | 16/26 | 🟢 QUEUE |
| 6 | ob_age_h1_bars | continuous | alert_time | 10748 | 0.0925 | [-0.0250, 0.1838] | — | 0.00640 | ρ=-0.0038 | 0.6958 | — | 6/16 | · not proven |
| 7 | setup_badge | nominal | alert_time | 735 | 0.0897 | [-0.1181, 0.2862] | — | n/a | H=0.8357 | 0.6585 | — | 0/0 | · not proven |
| 8 | impulse_leg_atr | continuous | alert_time | 10748 | -0.0876 | [-0.1923, 0.0143] | — | 0.00358 | ρ=-0.0085 | 0.3773 | — | 6/17 | · not proven |
| 9 | leg_retrace_pct_at_alert | continuous | alert_time | 10454 | 0.0792 | [-0.0254, 0.1809] | — | 0.00765 | ρ=-0.0064 | 0.5112 | — | 6/16 | · not proven |
| 10 | killzone_alignment | nominal | fill_time | 10748 | 0.0634 | [-0.0011, 0.1285] | — | n/a | H=11.8200 | 0.0080 | rej | 16/36 | · not proven |
| 11 | fill_session | nominal | fill_time | 10748 | 0.0628 | [-0.0743, 0.1970] | — | n/a | H=0.8038 | 0.8486 | — | 13/30 | · not proven |
| 12 | fvg_size_atr | continuous | alert_time | 4401 | 0.0624 | [-0.1118, 0.2355] | — | 0.00143 | ρ=0.0040 | 0.7911 | — | 0/0 | · not proven |
| 13 | break_body_atr | continuous | alert_time | 10748 | -0.0605 | [-0.1665, 0.0550] | — | 0.00032 | ρ=-0.0263 | 0.0063 | rej | 7/17 | · not proven |
| 14 | fvg_state | nominal | alert_time | 10748 | 0.0588 | [-0.0452, 0.1588] | — | n/a | H=4.7323 | 0.0938 | — | 10/36 | · not proven |
| 15 | ob_session | nominal | alert_time | 10748 | 0.0514 | [-0.0657, 0.1721] | — | n/a | H=12.5844 | 0.0056 | rej | 0/0 | · not proven |
| 16 | fvg_mitigation | nominal | alert_time | 10748 | 0.0512 | [-0.0456, 0.1475] | — | n/a | H=4.5861 | 0.2047 | — | 12/28 | · not proven |
| 17 | bars_break_to_pullback | continuous | fill_time | 10748 | 0.0509 | [-0.0436, 0.1558] | — | 0.00503 | ρ=0.0047 | 0.6260 | — | 9/18 | · not proven |
| 18 | break_close_atr | continuous | alert_time | 10748 | -0.0497 | [-0.1562, 0.0559] | — | 0.00000 | ρ=-0.0295 | 0.0022 | rej | 5/18 | · not proven |
| 19 | bos_tier | nominal | alert_time | 10748 | 0.0474 | [-0.0052, 0.1007] | — | n/a | H=8.3031 | 0.0401 | — | 15/36 | · not proven |
| 20 | event | nominal | alert_time | 10748 | 0.0474 | [-0.0052, 0.1007] | — | n/a | H=8.3031 | 0.0401 | — | 15/36 | · not proven |
| 21 | reversal_pct | binary | alert_time | 10748 | -0.0423 | [-0.0972, 0.0137] | — | n/a | bootstrap_diff_ci(binary) | — | — | 11/36 | · not proven |
| 22 | session | nominal | alert_time | 10748 | 0.0414 | [-0.0091, 0.0911] | — | n/a | H=7.5133 | 0.0234 | rej | 14/36 | · not proven |
| 23 | ob_range_atr | continuous | alert_time | 10748 | -0.0386 | [-0.1430, 0.0657] | — | 0.00000 | ρ=0.0357 | 0.0002 | rej | 5/20 | · not proven |
| 24 | fill_in_killzone | binary | fill_time | 10748 | -0.0372 | [-0.0852, 0.0109] | — | n/a | bootstrap_diff_ci(binary) | — | — | 14/36 | · not proven |
| 25 | bos_tag | binary | alert_time | 10748 | -0.0369 | [-0.0885, 0.0153] | — | n/a | bootstrap_diff_ci(binary) | — | — | 13/36 | · not proven |
| 26 | ob_touches | ordinal | alert_time | 10748 | -0.0351 | [-0.1698, 0.1020] | — | n/a | ρ=0.0111 | 0.2515 | — | — | · not proven |
| 27 | trend_pd_agree | binary | alert_time | 10748 | 0.0329 | [-0.0187, 0.0846] | — | n/a | bootstrap_diff_ci(binary) | — | — | 13/36 | · not proven |
| 28 | pd_alignment | binary | alert_time | 10748 | -0.0311 | [-0.0785, 0.0156] | — | n/a | bootstrap_diff_ci(binary) | — | — | 9/36 | · not proven |
| 29 | atr_at_ob | continuous | alert_time | 10748 | -0.0297 | [-0.1214, 0.0851] | — | 0.00509 | ρ=-0.0046 | 0.6356 | — | 10/20 | · not proven |
| 30 | score | continuous | alert_time | 10748 | -0.0289 | [-0.1164, 0.0703] | — | 0.00638 | ρ=-0.0089 | 0.3568 | — | 14/36 | · not proven |
| 31 | reversed_from_extreme | binary | alert_time | 3220 | -0.0272 | [-0.1230, 0.0693] | — | n/a | bootstrap_diff_ci(binary) | — | — | 2/4 | · not proven |
| 32 | flip_pending_at_alert | binary | alert_time | 10748 | -0.0255 | [-0.0769, 0.0271] | — | n/a | bootstrap_diff_ci(binary) | — | — | 11/36 | · not proven |
| 33 | pd_pct | continuous | alert_time | 10748 | -0.0236 | [-0.1324, 0.0743] | — | 0.00142 | ρ=-0.0034 | 0.7211 | — | 8/19 | · not proven |
| 34 | ob_in_killzone | binary | alert_time | 10748 | 0.0205 | [-0.0280, 0.0678] | — | n/a | bootstrap_diff_ci(binary) | — | — | 12/36 | · not proven |
| 35 | ob_body_ratio | continuous | alert_time | 10748 | 0.0198 | [-0.0890, 0.1220] | — | 0.00206 | ρ=0.0084 | 0.3815 | — | 9/19 | · not proven |
| 36 | structure_ranging_at_alert | binary | alert_time | 10748 | -0.0192 | [-0.1137, 0.0765] | — | n/a | bootstrap_diff_ci(binary) | — | — | 6/36 | · not proven |
| 37 | dr_ceiling_broken_at_ob | binary | alert_time | 10634 | 0.0158 | [-0.0422, 0.0744] | — | n/a | bootstrap_diff_ci(binary) | — | — | 13/36 | · not proven |
| 38 | fvg_present | binary | alert_time | 10748 | 0.0100 | [-0.0380, 0.0595] | — | n/a | bootstrap_diff_ci(binary) | — | — | 12/36 | · not proven |
| 39 | pd_zone | binary | alert_time | 10748 | -0.0095 | [-0.0564, 0.0381] | — | n/a | bootstrap_diff_ci(binary) | — | — | 8/36 | · not proven |
| 40 | bos_sequence_count | ordinal | alert_time | 10748 | 0.0093 | [-0.1584, 0.1853] | — | n/a | ρ=0.0208 | 0.0308 | rej | — | · not proven |
| 41 | bias | binary | alert_time | 10748 | -0.0087 | [-0.0558, 0.0380] | — | n/a | bootstrap_diff_ci(binary) | — | — | 7/36 | · not proven |
| 42 | dr_floor_broken_at_ob | binary | alert_time | 10634 | -0.0086 | [-0.0643, 0.0498] | — | n/a | bootstrap_diff_ci(binary) | — | — | 7/36 | · not proven |
| 43 | flip_pending_dir_at_alert | binary | alert_time | 3043 | -0.0044 | [-0.0936, 0.0843] | — | n/a | bootstrap_diff_ci(binary) | — | — | 14/36 | · not proven |
| 44 | bos_verdict | binary | alert_time | 10748 | 0.0014 | [-0.1374, 0.1384] | — | n/a | bootstrap_diff_ci(binary) | — | — | 9/36 | · not proven |
