# DISCOVERY REPORT (engine phase A) — h1only_20080102_20251231

> CANDIDATE ONLY — discovery split only. Luck is NOT ruled out. A candidate is confirmed only if it REPEATS on validation years it has never seen.

- **run_id:** h1only_20080102_20251231
- **window:** 2008-01-10..2016-12-30
- **N (discovery split):** 11329
- **split census:** DISCOVERY=11333  VALIDATION=6765  HOLDOUT=5221  WAR=0  OUT=0
- **scope:** verdict
- **generated (UTC):** 2026-07-05T07:54:49.726318+00:00

**APPROVAL TOKEN (same 4-line block as the email):**

```
APPROVAL TOKEN: 20ddd2883cbc
To confirm on validation (ONE shot):  press CONFIRM in the Action with this token,
or locally:  python -m backtest.diagnostics.edge_engine --approve 20ddd2883cbc
             python -m backtest.diagnostics.edge_engine --phase confirm
```

### How to read this report

- **candidate** = passed all four discovery criteria (FDR-reject, CI excludes 0,
  both extreme buckets N≥150, effect ≥0.10R). It is NOT confirmed — luck is not
  ruled out until it repeats on validation years it has never seen.
- **candidate_thin** = a real discovery signal but a thin bucket or a sub-0.10R
  effect. Kept visible, never shipped from here.
- **noise** = did not clear the discovery signal bar. **thin** = fewer than two
  testable buckets to compare at all.
- **Δdisc** = top-vs-bottom expR gap on the discovery split (categorical: best vs
  worst level). Its 95% bootstrap CI is shown beside it.
- **Why discovery numbers can still be luck:** these are one split, gates off, all
  scores. A gap here is a hypothesis to test, not a result to trade.

## 1. Verdict table (all features, |Δdisc| desc)

| feature | type | timing | verdict | Δdisc | CI | N(top/bot) | screen p | fdr | failed criteria |
|---|---|---|---|---|---|---|---|---|---|
| ob_walkback_depth | categorical | alert_time | noise | 0.194 | [+0.0456, +0.3380] | 7802/239 | 0.03016 | False | fail:fdr_reject |
| fill_session | categorical | fill_time | noise | 0.1505 | [+0.0230, +0.2747] | 1782/467 | 0.08165 | False | fail:fdr_reject |
| alert_utc_hour | continuous | alert_time | candidate | 0.102 | [+0.0226, +0.1799] | 1330/3016 | 0.0002876 | True | PASS |
| trend_alignment | categorical | alert_time | noise | 0.0998 | [-0.0013, +0.2000] | 930/2645 | 0.1659 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| pair | categorical | alert_time | noise | 0.0966 | [+0.0032, +0.1911] | 1442/990 | 0.9486 | False | fail:fdr_reject,substance_effect |
| setup_badge | categorical | alert_time | noise | 0.0851 | [-0.0695, +0.2417] | 471/438 | 0.5339 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| ob_touches | categorical | alert_time | noise | 0.08 | [-0.0485, +0.2082] | 2189/430 | 0.1817 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| fvg_size_atr | continuous | alert_time | noise | 0.0793 | [-0.0428, +0.2006] | 887/889 | 0.3765 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| fvg_mitigation | categorical | alert_time | noise | 0.0775 | [-0.0169, +0.1714] | 1295/1496 | 0.1777 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| bos_sequence_count | continuous | alert_time | noise | 0.0666 | [-0.0023, +0.1355] | 1515/6839 | 0.03222 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| killzone_alignment | categorical | fill_time | noise | 0.061 | [-0.0033, +0.1248] | 2580/4047 | 0.02263 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| reversed_from_extreme | categorical | alert_time | noise | 0.0604 | [-0.0397, +0.1633] | 854/2312 | 0.2798 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| impulse_leg_atr | continuous | alert_time | noise | -0.054 | [-0.1265, +0.0183] | 2264/2267 | 0.2023 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| break_close_atr | continuous | alert_time | noise | -0.0526 | [-0.1253, +0.0215] | 2241/2326 | 0.03318 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| session | categorical | alert_time | noise | 0.0472 | [-0.0198, +0.1150] | 4191/2047 | 0.02138 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| bos_tier | categorical | alert_time | noise | 0.0448 | [-0.0271, +0.1138] | 6560/1550 | 0.05645 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| event | categorical | alert_time | noise | 0.0448 | [-0.0271, +0.1138] | 6560/1550 | 0.05645 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| fill_in_killzone | categorical | fill_time | noise | 0.0427 | [-0.0036, +0.0891] | 4656/6673 | 0.05671 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| fvg_state | categorical | alert_time | noise | 0.0409 | [-0.0558, +0.1345] | 5241/694 | 0.1618 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| structure_ranging_at_alert | categorical | alert_time | noise | 0.0409 | [-0.0598, +0.1414] | 713/10616 | 0.7702 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| ob_session | categorical | alert_time | noise | 0.0334 | [-0.0355, +0.1014] | 4620/1497 | 0.1429 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| bars_break_to_pullback | continuous | fill_time | noise | 0.0322 | [-0.0381, +0.1041] | 2251/2535 | 0.7304 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| bos_tag | categorical | alert_time | noise | 0.0257 | [-0.0249, +0.0769] | 8163/3166 | 0.2017 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| pd_zone | categorical | alert_time | noise | 0.0255 | [-0.0200, +0.0719] | 5399/5930 | 0.4679 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| leg_retrace_pct_at_alert | continuous | alert_time | noise | -0.0247 | [-0.0964, +0.0473] | 2202/2218 | 0.1702 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| pd_alignment | categorical | alert_time | noise | 0.0218 | [-0.0245, +0.0661] | 5456/5873 | 0.6954 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| score | continuous | alert_time | noise | -0.0201 | [-0.0880, +0.0510] | 1721/4508 | 0.3116 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| flip_pending_dir_at_alert | categorical | alert_time | noise | 0.0193 | [-0.0470, +0.0867] | 8201/1527 | 0.5385 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| fvg_present | categorical | alert_time | noise | 0.019 | [-0.0277, +0.0670] | 4439/6890 | 0.4194 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| break_body_atr | continuous | alert_time | noise | -0.018 | [-0.0919, +0.0551] | 2233/2356 | 0.2423 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| pd_pct | continuous | alert_time | noise | -0.0179 | [-0.0887, +0.0536] | 2255/2276 | 0.7553 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| ob_body_ratio | continuous | alert_time | noise | 0.0175 | [-0.0542, +0.0896] | 2261/2266 | 0.3972 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| ob_to_fill_hours | continuous | fill_time | noise | 0.0172 | [-0.0529, +0.0874] | 2238/2560 | 0.6179 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| ob_in_killzone | categorical | alert_time | noise | 0.0122 | [-0.0338, +0.0577] | 4702/6627 | 0.02657 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| atr_at_ob | continuous | alert_time | noise | -0.012 | [-0.0861, +0.0612] | 2266/2267 | 0.7708 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| flip_pending_at_alert | categorical | alert_time | noise | 0.0115 | [-0.0409, +0.0640] | 8201/3128 | 0.2945 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| ob_age_h1_bars | continuous | alert_time | noise | -0.0095 | [-0.0775, +0.0600] | 2174/3027 | 0.4611 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| bos_verdict | categorical | alert_time | noise | 0.0069 | [-0.0940, +0.1079] | 10705/624 | 0.7889 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| dr_ceiling_broken_at_ob | categorical | alert_time | noise | 0.0053 | [-0.0503, +0.0627] | 2456/8757 | 0.8529 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| ob_range_atr | continuous | alert_time | noise | -0.005 | [-0.0784, +0.0686] | 2259/2270 | 0.0001815 | True | fail:ci_excludes_0,substance_effect |
| dr_floor_broken_at_ob | categorical | alert_time | noise | 0.0043 | [-0.0534, +0.0600] | 8863/2350 | 0.9547 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| bias | categorical | alert_time | noise | 0.004 | [-0.0417, +0.0495] | 5677/5652 | 0.9625 | False | fail:fdr_reject,ci_excludes_0,substance_effect |
| reversal_pct | continuous | alert_time | thin | — | [—, —] | —/— | — | False | — |

## 2. Candidate deep-dives

### `alert_utc_hour`  (continuous, alert_time)

- Δdisc **0.102** CI [+0.0226, +0.1799], N(top/bot) 1330/3016, screen p 0.0002876
- favoured bucket: 4
- CANDIDATE — luck not ruled out. The confirm phase re-computes this same Δ on validation years and requires the same sign + ≥60% positive quarters.

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 4.0 | 7.0 | 3016 | -0.0858 | [-0.1310, -0.0400] | 32.8 | -258.64 | 8/36 |
| 1.0 | 7.0 | 9.0 | 2101 | -0.0643 | [-0.1190, -0.0090] | 33.2 | -135.17 | 10/36 |
| 2.0 | 9.0 | 12.0 | 2021 | -0.0536 | [-0.1070, -0.0010] | 35.4 | -108.24 | 13/36 |
| 3.0 | 12.0 | 15.0 | 2861 | -0.0541 | [-0.0990, -0.0080] | 34.9 | -154.76 | 13/36 |
| 4.0 | 15.0 | 18.0 | 1330 | 0.0163 | [-0.0490, +0.0800] | 39.7 | 21.67 | 16/28 |

## 3. Near-misses

**NOT candidates. Shown for transparency (C4). No action, no threshold renegotiation (F).**

| feature | verdict | Δdisc | CI | N(top/bot) | the one failed criterion |
|---|---|---|---|---|---|
| ob_walkback_depth | noise | 0.194 | [+0.0456, +0.3380] | 7802/239 | fdr_reject |
| fill_session | noise | 0.1505 | [+0.0230, +0.2747] | 1782/467 | fdr_reject |

## 4. Baseline context — how did pairs / sessions do

**Caveats (read before any number below):** (a) this is the gates-off, all-scores discovery population — NOT what live (score≥4, filtered) trading would produce; (b) Book B pairs (GBPUSD, AUDUSD, USDCAD, EURJPY) are pooled per SPEC §3.3 but are NOT in live trade scope.

### Overall discovery-split
| — | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| ALL | 11329 | -0.0561 | [-0.0790, -0.0340] | 34.7 | -635.13 | 9/36 |

### Per pair
| pair | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| AUDUSD | 1017 | -0.0437 | [-0.1210, +0.0340] | 35.1 | -44.43 | 2/10 |
| EURJPY | 1150 | -0.0618 | [-0.1350, +0.0100] | 33.9 | -71.03 | 10/24 |
| EURUSD | 1433 | -0.0722 | [-0.1350, -0.0070] | 34.2 | -103.39 | 15/34 |
| GBPUSD | 1450 | -0.0697 | [-0.1330, -0.0050] | 33.3 | -101.05 | 9/36 |
| GOLD | 1370 | -0.0497 | [-0.1190, +0.0190] | 32.6 | -68.12 | 11/33 |
| NZDUSD | 990 | -0.1 | [-0.1660, -0.0330] | 37.2 | -98.96 | 4/15 |
| USDCAD | 1442 | -0.0034 | [-0.0690, +0.0630] | 36.7 | -4.95 | 18/34 |
| USDCHF | 1344 | -0.0519 | [-0.1180, +0.0160] | 35.1 | -69.74 | 9/31 |
| USDJPY | 1133 | -0.0648 | [-0.1380, +0.0100] | 34.5 | -73.46 | 9/25 |

### Per session (alert session)
| session | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| Asia | 2047 | -0.079 | [-0.1340, -0.0240] | 32.9 | -161.69 | 9/36 |
| London | 5091 | -0.0669 | [-0.1010, -0.0320] | 33.9 | -340.35 | 9/36 |
| NY | 4191 | -0.0318 | [-0.0690, +0.0060] | 36.4 | -133.09 | 15/36 |

## 5. Sub-screens

### Snapback bins (bars_break_to_pullback)

| bin | n | expR | CI | wr% | totR | pos_q | caveat |
|---|---|---|---|---|---|---|---|
| 1-2 | 3751 | -0.0984 | [-0.1350, -0.0610] | 35.9 | -368.93 | 21/71 | 1-2 bin under-sampled: backtest alerts ~1 bar late vs live forming-bar proximity; trust a POSITIVE effect, distrust a null-on-thin |
| 3-5 | 3471 | -0.0579 | [-0.0980, -0.0170] | 35.4 | -201.02 | 28/72 |  |
| 6-12 | 3646 | -0.0586 | [-0.0990, -0.0180] | 35.1 | -213.56 | 24/72 |  |
| 13-inf | 11537 | -0.0244 | [-0.0470, -0.0010] | 35.5 | -281.58 | 30/72 |  |

### SL-anatomy (clean-break rate by bucket, on eligible SL exits)

| feature | hi | lo | clean_rate hi | clean_rate lo | rate_diff | CI | n hi/lo | robust |
|---|---|---|---|---|---|---|---|---|
| break_close_atr | q4 | q0 | 0.457 | 0.465 | -0.0082 | [-0.0425, +0.0267] | 1586/1616 | False |
| break_body_atr | q4 | q0 | 0.476 | 0.469 | 0.0074 | [-0.0267, +0.0416] | 1575/1619 | False |
| impulse_leg_atr | q4 | q0 | 0.435 | 0.49 | -0.0552 | [-0.0899, -0.0205] | 1586/1587 | True |
| fvg_size_atr | q4 | q0 | 0.461 | 0.455 | 0.0062 | [-0.0490, +0.0615] | 633/635 | False |
| ob_range_atr | q4 | q0 | 0.476 | 0.471 | 0.0052 | [-0.0301, +0.0406] | 1583/1592 | False |
| atr_at_ob | q4 | q0 | 0.471 | 0.464 | 0.0077 | [-0.0276, +0.0429] | 1583/1592 | False |
| pd_pct | q4 | q0 | 0.464 | 0.465 | -0.0005 | [-0.0357, +0.0341] | 1585/1594 | False |
| ob_age_h1_bars | q4 | q0 | 0.456 | 0.484 | -0.028 | [-0.0608, +0.0046] | 1577/2079 | False |
| ob_to_fill_hours | q4 | q0 | 0.464 | 0.492 | -0.0275 | [-0.0610, +0.0072] | 1556/1766 | False |
| bars_break_to_pullback | q4 | q0 | 0.473 | 0.492 | -0.0188 | [-0.0538, +0.0155] | 1567/1751 | False |
| bos_sequence_count | q2 | q0 | 0.475 | 0.47 | 0.0048 | [-0.0292, +0.0393] | 1001/4905 | False |
| score | q4 | q0 | 0.464 | 0.476 | -0.012 | [-0.0446, +0.0205] | 1234/3129 | False |
| alert_utc_hour | q4 | q0 | 0.441 | 0.474 | -0.0339 | [-0.0665, -0.0014] | 1539/2169 | True |
| ob_body_ratio | q4 | q0 | 0.453 | 0.488 | -0.0358 | [-0.0709, -0.0013] | 1584/1597 | True |
| leg_retrace_pct_at_alert | q4 | q0 | 0.488 | 0.469 | 0.0186 | [-0.0163, +0.0535] | 1533/1551 | False |
| bos_tag | CHoCH | BOS | 0.481 | 0.464 | 0.0171 | [-0.0068, +0.0416] | 2295/5636 | False |
| bos_tier | CHoCH | Confirm | 0.481 | 0.445 | 0.0363 | [+0.0009, +0.0719] | 2295/1121 | True |
| bos_verdict | holding | fading | 0.47 | 0.456 | 0.0138 | [-0.0348, +0.0618] | 7508/423 | False |
| event | CHoCH CHoCH | Confirmation BOS | 0.481 | 0.445 | 0.0363 | [+0.0009, +0.0719] | 2295/1121 | True |
| reversed_from_extreme | False | other | 0.501 | 0.464 | 0.0365 | [-0.0059, +0.0787] | 609/5636 | False |
| fvg_present | False | True | 0.479 | 0.454 | 0.0253 | [+0.0032, +0.0479] | 4760/3171 | True |
| fvg_mitigation | none | partial | 0.482 | 0.454 | 0.0284 | [-0.0080, +0.0648] | 3693/906 | False |
| fvg_state | no_fvg | fresh | 0.482 | 0.455 | 0.0267 | [+0.0041, +0.0492] | 3693/3734 | True |
| pd_zone | premium | discount | 0.47 | 0.469 | 0.0008 | [-0.0211, +0.0224] | 4186/3745 | False |
| pd_alignment | aligned | counter | 0.47 | 0.469 | 0.0013 | [-0.0206, +0.0233] | 3860/4071 | False |
| session | Asia | NY | 0.478 | 0.456 | 0.0223 | [-0.0086, +0.0543] | 1475/2851 | False |
| ob_session | NY | London | 0.483 | 0.454 | 0.0285 | [-0.0065, +0.0632] | 1036/3173 | False |
| fill_session | Asia | Other | 0.477 | 0.431 | 0.0457 | [-0.0126, +0.1047] | 1247/350 | False |
| killzone_alignment | Fill only | Both | 0.477 | 0.458 | 0.0185 | [-0.0109, +0.0481] | 2894/1778 | False |
| ob_in_killzone | False | True | 0.471 | 0.466 | 0.0048 | [-0.0179, +0.0268] | 4728/3203 | False |
| fill_in_killzone | True | False | 0.47 | 0.469 | 0.0013 | [-0.0206, +0.0243] | 4672/3259 | False |
| trend_alignment | ambiguous | counter_trend | 0.485 | 0.453 | 0.0313 | [-0.0121, +0.0751] | 1917/664 | False |
| setup_badge | other | Caution: Late-Trend Chase | 0.47 | 0.46 | 0.01 | [-0.0470, +0.0697] | 7302/298 | False |
| ob_touches | 1 | 2 | 0.475 | 0.447 | 0.0274 | [-0.0339, +0.0889] | 1504/313 | False |
| bias | LONG | SHORT | 0.47 | 0.468 | 0.0018 | [-0.0196, +0.0237] | 3960/3971 | False |
| pair | USDCAD | NZDUSD | 0.485 | 0.439 | 0.0459 | [-0.0031, +0.0940] | 987/660 | False |
| ob_walkback_depth | 2 | 3 | 0.48 | 0.432 | 0.0481 | [-0.0357, +0.1302] | 544/183 | False |
| structure_ranging_at_alert | True | False | 0.477 | 0.469 | 0.0085 | [-0.0348, +0.0532] | 505/7426 | False |
| flip_pending_at_alert | True | False | 0.482 | 0.464 | 0.0179 | [-0.0065, +0.0419] | 2263/5668 | False |
| flip_pending_dir_at_alert | bearish | other | 0.487 | 0.464 | 0.0228 | [-0.0082, +0.0552] | 1150/5668 | False |
| dr_ceiling_broken_at_ob | True | False | 0.478 | 0.467 | 0.0116 | [-0.0155, +0.0382] | 1706/6145 | False |
| dr_floor_broken_at_ob | False | True | 0.472 | 0.461 | 0.011 | [-0.0162, +0.0376] | 6218/1633 | False |

_Promotions are a validation concept — deferred to the confirm phase._

### News confounder (context only, never a gate)

_news unusable_

_Interactions: deferred to the confirm phase._

## 6. Appendix — every feature's bucket/level table

### `break_close_atr`  (continuous, alert_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.0 | 0.65 | 2326 | -0.0313 | [-0.0790, +0.0190] | 37.6 | -72.77 | 14/36 |
| 1.0 | 0.65 | 1.0 | 2249 | -0.0476 | [-0.0980, +0.0050] | 35.2 | -107.06 | 13/36 |
| 2.0 | 1.0 | 1.34 | 2239 | -0.0762 | [-0.1260, -0.0260] | 34.3 | -170.55 | 10/36 |
| 3.0 | 1.34 | 1.9 | 2274 | -0.0425 | [-0.0960, +0.0100] | 34.7 | -96.71 | 14/36 |
| 4.0 | 1.9 | 12.25 | 2241 | -0.0839 | [-0.1370, -0.0290] | 31.4 | -188.04 | 12/36 |

### `break_body_atr`  (continuous, alert_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.02 | 1.16 | 2356 | -0.0433 | [-0.0940, +0.0090] | 35.2 | -102.13 | 14/36 |
| 1.0 | 1.16 | 1.46 | 2257 | -0.014 | [-0.0670, +0.0380] | 37.4 | -31.69 | 16/36 |
| 2.0 | 1.46 | 1.7 | 2230 | -0.0955 | [-0.1440, -0.0450] | 33.2 | -213.0 | 10/36 |
| 3.0 | 1.7 | 2.14 | 2253 | -0.0671 | [-0.1160, -0.0170] | 34.1 | -151.24 | 12/36 |
| 4.0 | 2.14 | 10.41 | 2233 | -0.0614 | [-0.1120, -0.0070] | 33.2 | -137.07 | 13/36 |

### `impulse_leg_atr`  (continuous, alert_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.0 | 0.986 | 2267 | -0.0432 | [-0.0960, +0.0110] | 34.2 | -98.0 | 14/36 |
| 1.0 | 0.986 | 1.606 | 2266 | -0.059 | [-0.1100, -0.0070] | 34.4 | -133.8 | 14/36 |
| 2.0 | 1.606 | 2.312 | 2266 | -0.041 | [-0.0910, +0.0090] | 35.9 | -92.94 | 17/36 |
| 3.0 | 2.312 | 3.874 | 2266 | -0.0399 | [-0.0910, +0.0140] | 35.7 | -90.37 | 14/36 |
| 4.0 | 3.874 | 50.711 | 2264 | -0.0972 | [-0.1470, -0.0450] | 33.1 | -220.03 | 11/36 |

### `fvg_size_atr`  (continuous, alert_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.059 | 0.241 | 889 | -0.0888 | [-0.1720, -0.0020] | 31.8 | -78.9 | 2/7 |
| 1.0 | 0.241 | 0.434 | 889 | -0.0797 | [-0.1610, +0.0060] | 32.1 | -70.81 | 4/8 |
| 2.0 | 0.434 | 0.658 | 887 | 0.0197 | [-0.0660, +0.1100] | 35.9 | 17.43 | 4/5 |
| 3.0 | 0.658 | 1.06 | 887 | -0.0644 | [-0.1500, +0.0240] | 31.8 | -57.1 | 2/9 |
| 4.0 | 1.06 | 9.522 | 887 | -0.0094 | [-0.0980, +0.0820] | 34.7 | -8.35 | 5/9 |

### `ob_range_atr`  (continuous, alert_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.011 | 0.667 | 2270 | -0.0897 | [-0.1460, -0.0300] | 29.2 | -203.67 | 10/36 |
| 1.0 | 0.667 | 0.857 | 2269 | -0.0245 | [-0.0760, +0.0300] | 34.5 | -55.54 | 16/36 |
| 2.0 | 0.857 | 1.057 | 2261 | -0.0453 | [-0.0950, +0.0050] | 35.6 | -102.38 | 13/36 |
| 3.0 | 1.057 | 1.349 | 2270 | -0.0262 | [-0.0740, +0.0230] | 37.9 | -59.52 | 16/36 |
| 4.0 | 1.349 | 2.0 | 2259 | -0.0947 | [-0.1400, -0.0480] | 36.2 | -214.04 | 12/36 |

### `atr_at_ob`  (continuous, alert_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.000339 | 0.001358 | 2267 | -0.0458 | [-0.0980, +0.0060] | 35.6 | -103.82 | 8/22 |
| 1.0 | 0.001358 | 0.0018962 | 2265 | -0.07 | [-0.1190, -0.0200] | 34.8 | -158.59 | 10/31 |
| 2.0 | 0.0018962 | 0.002993 | 2266 | -0.0431 | [-0.0950, +0.0090] | 35.2 | -97.65 | 13/28 |
| 3.0 | 0.002993 | 0.2158858 | 2265 | -0.0636 | [-0.1150, -0.0130] | 34.6 | -144.06 | 7/33 |
| 4.0 | 0.2158858 | 19.525714 | 2266 | -0.0578 | [-0.1090, -0.0050] | 33.0 | -131.01 | 12/36 |

### `pd_pct`  (continuous, alert_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | -155.7 | 25.5 | 2276 | -0.0527 | [-0.1030, -0.0010] | 35.7 | -119.95 | 13/36 |
| 1.0 | 25.5 | 42.8 | 2257 | -0.0395 | [-0.0930, +0.0140] | 34.6 | -89.16 | 14/36 |
| 2.0 | 42.8 | 61.3 | 2275 | -0.1068 | [-0.1560, -0.0560] | 32.0 | -242.93 | 7/36 |
| 3.0 | 61.3 | 77.3 | 2266 | -0.0106 | [-0.0630, +0.0410] | 36.4 | -23.97 | 18/36 |
| 4.0 | 77.3 | 323.5 | 2255 | -0.0706 | [-0.1210, -0.0200] | 34.6 | -159.12 | 12/36 |

### `reversal_pct`  (continuous, alert_time) — verdict: thin

_no bucket rows in the CSV for this feature (thin / insufficient distinct values)._

### `ob_age_h1_bars`  (continuous, alert_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 2.0 | 3.0 | 3027 | -0.0648 | [-0.1070, -0.0220] | 35.9 | -196.09 | 12/36 |
| 1.0 | 3.0 | 4.0 | 1616 | -0.0512 | [-0.1120, +0.0120] | 35.5 | -82.75 | 14/35 |
| 2.0 | 4.0 | 7.0 | 2447 | -0.0598 | [-0.1090, -0.0100] | 34.3 | -146.35 | 12/36 |
| 3.0 | 7.0 | 22.0 | 2065 | -0.0234 | [-0.0790, +0.0310] | 35.6 | -48.38 | 14/36 |
| 4.0 | 22.0 | 257.0 | 2174 | -0.0743 | [-0.1270, -0.0190] | 31.8 | -161.56 | 12/36 |

### `ob_to_fill_hours`  (continuous, fill_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 2.0 | 6.0 | 2560 | -0.0991 | [-0.1450, -0.0530] | 34.9 | -253.75 | 11/36 |
| 1.0 | 6.0 | 11.0 | 1975 | -0.0451 | [-0.1020, +0.0110] | 35.0 | -89.16 | 14/36 |
| 2.0 | 11.0 | 26.0 | 2285 | -0.0196 | [-0.0690, +0.0320] | 36.7 | -44.88 | 16/36 |
| 3.0 | 26.0 | 66.0 | 2271 | -0.0282 | [-0.0810, +0.0240] | 35.5 | -64.06 | 11/36 |
| 4.0 | 66.0 | 432.0 | 2238 | -0.0819 | [-0.1350, -0.0270] | 31.0 | -183.29 | 11/36 |

### `bars_break_to_pullback`  (continuous, fill_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.0 | 3.0 | 2535 | -0.1005 | [-0.1470, -0.0540] | 34.5 | -254.88 | 10/36 |
| 1.0 | 3.0 | 8.0 | 2195 | -0.052 | [-0.1050, +0.0000] | 34.9 | -114.05 | 11/36 |
| 2.0 | 8.0 | 20.0 | 2233 | -0.0299 | [-0.0800, +0.0210] | 36.2 | -66.78 | 14/36 |
| 3.0 | 20.0 | 34.0 | 2115 | -0.0215 | [-0.0750, +0.0310] | 36.1 | -45.58 | 15/36 |
| 4.0 | 34.0 | 283.0 | 2251 | -0.0683 | [-0.1210, -0.0140] | 31.6 | -153.84 | 14/36 |

### `bos_sequence_count`  (continuous, alert_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.0 | 1.0 | 6839 | -0.0687 | [-0.0980, -0.0380] | 33.2 | -470.11 | 11/36 |
| 1.0 | 1.0 | 3.0 | 2975 | -0.0544 | [-0.0970, -0.0100] | 35.7 | -161.85 | 8/36 |
| 2.0 | 3.0 | 10.0 | 1515 | -0.0021 | [-0.0630, +0.0600] | 38.9 | -3.18 | 14/35 |

### `score`  (continuous, alert_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.0 | 5.0 | 4508 | -0.0553 | [-0.0910, -0.0180] | 35.2 | -249.19 | 11/36 |
| 1.0 | 5.0 | 5.2 | 33 | 0.1789 | [-0.2660, +0.6370] | 44.4 | 5.91 | 0/0 |
| 2.0 | 5.2 | 6.0 | 2763 | -0.0386 | [-0.0860, +0.0080] | 35.8 | -106.7 | 14/36 |
| 3.0 | 6.0 | 7.0 | 2304 | -0.0675 | [-0.1180, -0.0170] | 33.4 | -155.42 | 11/36 |
| 4.0 | 7.0 | 10.0 | 1721 | -0.0754 | [-0.1350, -0.0140] | 32.7 | -129.72 | 14/36 |

### `alert_utc_hour`  (continuous, alert_time) — verdict: candidate

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 4.0 | 7.0 | 3016 | -0.0858 | [-0.1310, -0.0400] | 32.8 | -258.64 | 8/36 |
| 1.0 | 7.0 | 9.0 | 2101 | -0.0643 | [-0.1190, -0.0090] | 33.2 | -135.17 | 10/36 |
| 2.0 | 9.0 | 12.0 | 2021 | -0.0536 | [-0.1070, -0.0010] | 35.4 | -108.24 | 13/36 |
| 3.0 | 12.0 | 15.0 | 2861 | -0.0541 | [-0.0990, -0.0080] | 34.9 | -154.76 | 13/36 |
| 4.0 | 15.0 | 18.0 | 1330 | 0.0163 | [-0.0490, +0.0800] | 39.7 | 21.67 | 16/28 |

### `ob_body_ratio`  (continuous, alert_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.2 | 0.3076 | 2266 | -0.0694 | [-0.1200, -0.0180] | 34.2 | -157.24 | 9/36 |
| 1.0 | 0.3076 | 0.417 | 2287 | -0.0643 | [-0.1140, -0.0140] | 34.2 | -146.96 | 12/36 |
| 2.0 | 0.417 | 0.537 | 2253 | -0.0406 | [-0.0950, +0.0140] | 34.4 | -91.4 | 13/36 |
| 3.0 | 0.537 | 0.683 | 2262 | -0.0541 | [-0.1050, -0.0040] | 35.3 | -122.33 | 12/36 |
| 4.0 | 0.683 | 1.0 | 2261 | -0.0518 | [-0.1020, -0.0020] | 35.2 | -117.21 | 12/36 |

### `leg_retrace_pct_at_alert`  (continuous, alert_time) — verdict: noise

| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.6 | 35.6 | 2218 | -0.0179 | [-0.0690, +0.0320] | 39.7 | -39.75 | 17/36 |
| 1.0 | 35.6 | 52.5 | 2196 | -0.0382 | [-0.0900, +0.0160] | 35.6 | -83.95 | 12/36 |
| 2.0 | 52.5 | 69.5 | 2207 | -0.1359 | [-0.1850, -0.0850] | 30.6 | -299.91 | 7/36 |
| 3.0 | 69.5 | 87.8 | 2201 | -0.0488 | [-0.1030, +0.0060] | 33.5 | -107.33 | 15/36 |
| 4.0 | 87.8 | 10131.6 | 2202 | -0.0426 | [-0.0950, +0.0100] | 33.8 | -93.88 | 12/36 |

### `bos_tag`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| BOS | 8163 | -0.0489 | [-0.0760, -0.0210] | 35.3 | -398.91 | 9/36 |
| CHoCH | 3166 | -0.0746 | [-0.1170, -0.0300] | 32.8 | -236.23 | 14/36 |

### `bos_tier`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| BOS | 6560 | -0.0384 | [-0.0680, -0.0080] | 36.0 | -251.78 | 11/36 |
| CHoCH | 3166 | -0.0746 | [-0.1170, -0.0300] | 32.8 | -236.23 | 14/36 |
| Confirm | 1550 | -0.0832 | [-0.1490, -0.0170] | 32.6 | -129.03 | 12/36 |
| other | 53 | -0.3415 | [-0.5660, -0.1080] | 32.6 | -18.1 | 0/0 |

### `bos_verdict`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| fading | 624 | -0.0626 | [-0.1590, +0.0390] | 36.5 | -39.06 | 1/2 |
| holding | 10705 | -0.0557 | [-0.0790, -0.0320] | 34.5 | -596.08 | 10/36 |

### `event`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| BOS BOS | 6560 | -0.0384 | [-0.0680, -0.0080] | 36.0 | -251.78 | 11/36 |
| CHoCH CHoCH | 3166 | -0.0746 | [-0.1170, -0.0300] | 32.8 | -236.23 | 14/36 |
| Confirmation BOS | 1550 | -0.0832 | [-0.1490, -0.0170] | 32.6 | -129.03 | 12/36 |
| other | 53 | -0.3415 | [-0.5660, -0.1080] | 32.6 | -18.1 | 0/0 |

### `reversed_from_extreme`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| False | 854 | -0.0305 | [-0.1170, +0.0590] | 34.8 | -26.04 | 2/6 |
| True | 2312 | -0.0909 | [-0.1410, -0.0390] | 32.1 | -210.19 | 11/36 |
| other | 8163 | -0.0489 | [-0.0760, -0.0210] | 35.3 | -398.91 | 9/36 |

### `fvg_present`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| False | 6890 | -0.0635 | [-0.0920, -0.0350] | 35.5 | -437.4 | 10/36 |
| True | 4439 | -0.0445 | [-0.0830, -0.0060] | 33.3 | -197.73 | 11/36 |

### `fvg_mitigation`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| full | 1496 | -0.0927 | [-0.1540, -0.0300] | 32.6 | -138.66 | 10/34 |
| none | 5394 | -0.0554 | [-0.0870, -0.0230] | 36.3 | -298.74 | 13/36 |
| partial | 1295 | -0.0152 | [-0.0850, +0.0570] | 34.5 | -19.68 | 12/25 |
| pristine | 3144 | -0.0566 | [-0.1030, -0.0110] | 32.8 | -178.06 | 11/36 |

### `fvg_state`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| fresh | 5241 | -0.0519 | [-0.0870, -0.0170] | 33.3 | -272.0 | 14/36 |
| no_fvg | 5394 | -0.0554 | [-0.0870, -0.0230] | 36.3 | -298.74 | 13/36 |
| stale | 694 | -0.0928 | [-0.1830, -0.0000] | 31.9 | -64.4 | 0/0 |

### `pd_zone`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| discount | 5399 | -0.0427 | [-0.0760, -0.0080] | 35.1 | -230.49 | 10/36 |
| premium | 5930 | -0.0682 | [-0.0990, -0.0370] | 34.2 | -404.64 | 10/36 |

### `pd_alignment`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| aligned | 5456 | -0.0448 | [-0.0790, -0.0110] | 33.9 | -244.26 | 9/36 |
| counter | 5873 | -0.0666 | [-0.0980, -0.0340] | 35.3 | -390.87 | 10/36 |

### `session`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| Asia | 2047 | -0.079 | [-0.1340, -0.0240] | 32.9 | -161.69 | 9/36 |
| London | 5091 | -0.0669 | [-0.1010, -0.0320] | 33.9 | -340.35 | 9/36 |
| NY | 4191 | -0.0318 | [-0.0690, +0.0060] | 36.4 | -133.09 | 15/36 |

### `ob_session`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| Asia | 4564 | -0.0586 | [-0.0960, -0.0210] | 33.4 | -267.39 | 11/36 |
| London | 4620 | -0.0467 | [-0.0810, -0.0120] | 35.9 | -215.58 | 14/36 |
| NY | 1497 | -0.0801 | [-0.1390, -0.0200] | 35.5 | -119.94 | 10/34 |
| Other | 648 | -0.0497 | [-0.1540, +0.0580] | 32.4 | -32.22 | 1/1 |

### `fill_session`  (categorical, fill_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| Asia | 1782 | 0.0031 | [-0.0560, +0.0630] | 36.4 | 5.55 | 18/36 |
| London | 4502 | -0.0439 | [-0.0810, -0.0060] | 34.5 | -197.64 | 11/36 |
| NY | 4578 | -0.0817 | [-0.1160, -0.0470] | 34.6 | -374.23 | 8/36 |
| Other | 467 | -0.1474 | [-0.2530, -0.0370] | 30.0 | -68.82 | 0/0 |

### `killzone_alignment`  (categorical, fill_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| Both | 2626 | -0.0563 | [-0.1010, -0.0110] | 36.3 | -147.95 | 10/36 |
| Fill only | 4047 | -0.0849 | [-0.1240, -0.0460] | 32.9 | -343.39 | 6/36 |
| Neither | 2580 | -0.0239 | [-0.0750, +0.0270] | 34.6 | -61.79 | 16/36 |
| OB only | 2076 | -0.0395 | [-0.0940, +0.0140] | 36.2 | -82.01 | 17/36 |

### `ob_in_killzone`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| False | 6627 | -0.0611 | [-0.0920, -0.0310] | 33.5 | -405.18 | 8/36 |
| True | 4702 | -0.0489 | [-0.0840, -0.0140] | 36.2 | -229.96 | 14/36 |

### `fill_in_killzone`  (categorical, fill_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| False | 4656 | -0.0309 | [-0.0670, +0.0050] | 35.3 | -143.8 | 13/36 |
| True | 6673 | -0.0736 | [-0.1030, -0.0440] | 34.2 | -491.34 | 8/36 |

### `trend_alignment`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| ambiguous | 2645 | -0.0827 | [-0.1310, -0.0350] | 32.6 | -218.69 | 10/36 |
| counter_trend | 930 | 0.0171 | [-0.0730, +0.1060] | 34.4 | 15.92 | 4/7 |
| with_trend | 7754 | -0.0558 | [-0.0830, -0.0290] | 35.4 | -432.36 | 8/36 |

### `setup_badge`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| A First Pullback | 471 | -0.0438 | [-0.1550, +0.0690] | 33.8 | -20.64 | 0/0 |
| Caution: Late-Trend Chase | 438 | -0.1289 | [-0.2360, -0.0190] | 35.4 | -56.44 | 0/0 |
| other | 10420 | -0.0536 | [-0.0770, -0.0290] | 34.7 | -558.06 | 11/36 |

### `ob_touches`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| 0 | 8710 | -0.0633 | [-0.0890, -0.0370] | 34.3 | -551.15 | 12/36 |
| 1 | 2189 | -0.0189 | [-0.0720, +0.0340] | 36.7 | -41.44 | 15/36 |
| 2 | 430 | -0.0989 | [-0.2140, +0.0260] | 32.1 | -42.55 | 0/0 |

### `bias`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| LONG | 5652 | -0.0581 | [-0.0910, -0.0260] | 34.6 | -328.23 | 8/36 |
| SHORT | 5677 | -0.0541 | [-0.0870, -0.0210] | 34.7 | -306.91 | 12/36 |

### `pair`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| AUDUSD | 1017 | -0.0437 | [-0.1210, +0.0340] | 35.1 | -44.43 | 2/10 |
| EURJPY | 1150 | -0.0618 | [-0.1350, +0.0100] | 33.9 | -71.03 | 10/24 |
| EURUSD | 1433 | -0.0722 | [-0.1350, -0.0070] | 34.2 | -103.39 | 15/34 |
| GBPUSD | 1450 | -0.0697 | [-0.1330, -0.0050] | 33.3 | -101.05 | 9/36 |
| GOLD | 1370 | -0.0497 | [-0.1190, +0.0190] | 32.6 | -68.12 | 11/33 |
| NZDUSD | 990 | -0.1 | [-0.1660, -0.0330] | 37.2 | -98.96 | 4/15 |
| USDCAD | 1442 | -0.0034 | [-0.0690, +0.0630] | 36.7 | -4.95 | 18/34 |
| USDCHF | 1344 | -0.0519 | [-0.1180, +0.0160] | 35.1 | -69.74 | 9/31 |
| USDJPY | 1133 | -0.0648 | [-0.1380, +0.0100] | 34.5 | -73.46 | 9/25 |

### `ob_walkback_depth`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| 0 | 7802 | -0.0439 | [-0.0710, -0.0150] | 35.5 | -342.43 | 12/36 |
| 1 | 2422 | -0.0678 | [-0.1170, -0.0180] | 33.8 | -164.22 | 11/36 |
| 2 | 748 | -0.1162 | [-0.2020, -0.0250] | 30.8 | -86.93 | 0/0 |
| 3 | 239 | -0.2379 | [-0.3790, -0.0880] | 26.0 | -56.85 | 0/0 |
| other | 118 | 0.1297 | [-0.1290, +0.4160] | 36.2 | 15.3 | 0/0 |

### `structure_ranging_at_alert`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| False | 10616 | -0.0586 | [-0.0830, -0.0350] | 34.7 | -622.52 | 8/36 |
| True | 713 | -0.0177 | [-0.1160, +0.0810] | 34.4 | -12.61 | 1/2 |

### `flip_pending_at_alert`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| False | 8201 | -0.0529 | [-0.0790, -0.0260] | 35.3 | -433.76 | 8/36 |
| True | 3128 | -0.0644 | [-0.1080, -0.0200] | 33.0 | -201.37 | 10/36 |

### `flip_pending_dir_at_alert`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| bearish | 1601 | -0.0569 | [-0.1210, +0.0090] | 33.1 | -91.13 | 15/36 |
| bullish | 1527 | -0.0722 | [-0.1350, -0.0100] | 32.8 | -110.24 | 12/34 |
| other | 8201 | -0.0529 | [-0.0790, -0.0260] | 35.3 | -433.76 | 8/36 |

### `dr_ceiling_broken_at_ob`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| False | 8757 | -0.0573 | [-0.0830, -0.0310] | 34.3 | -502.04 | 10/36 |
| True | 2456 | -0.052 | [-0.1010, -0.0020] | 35.7 | -127.82 | 13/36 |
| other | 116 | -0.0455 | [-0.2550, +0.1760] | 36.8 | -5.28 | 0/0 |

### `dr_floor_broken_at_ob`  (categorical, alert_time) — verdict: noise

| level | n | expR | CI | wr% | totR | pos_q |
|---|---|---|---|---|---|---|
| False | 8863 | -0.0553 | [-0.0810, -0.0300] | 34.4 | -489.75 | 5/36 |
| True | 2350 | -0.0596 | [-0.1110, -0.0090] | 35.3 | -140.11 | 10/36 |
| other | 116 | -0.0455 | [-0.2550, +0.1760] | 36.8 | -5.28 | 0/0 |

## 7. Observations (observation-only, no action)

- `event` and `bos_tier` produced identical screen stats (Δdisc 0.0448, p 0.05645) — likely a 1:1 level mapping in this population. Observation only; neither feature is dropped (manifest is frozen, B6).

