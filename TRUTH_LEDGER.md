# TRUTH LEDGER

One line per trades.csv column and per emitted insight. A number is trustworthy only
when its row here says **verified** or **fixed** — anything else is unproven.

- **verified** — live code path traced end-to-end with file:line evidence; stamped on the right bar, frozen correctly, population known.
- **fixed** — defect found, fix shipped WITH a structural guard (test/gate/whitelist).
- **voided** — computed on a corrupted-era column; invalid until re-derived from a post-fix baseline run.
- **pending** — not yet audited. Assume nothing.
- **pending-fix** — defect known (spec written), fix NOT yet in code.
- **out-of-scope** — owned by another workstream (sweep rebuild).

Status date: 2026-07-03. DETECTION_FIXES_SPEC.md Fixes 1/2/3 **LANDED** (markers
verified in smc_radar.py / replay_engine.py / h1_only_simulator.py; guards
tests/test_dedupe.py + tests/test_ob_alert_freeze.py PASS 2026-07-03).
TRUTH_FIXES_SPEC.md: T2 + T3 SHIPPED 2026-07-03 (this session, guards green);
T1 remains for Opus. Voided insights stay VOID until the first post-fix baseline run.

## Layer map (where numbers are born)

1. **Detection** — smc_radar.py / smc_detector.py / dealing_range.py (shared with live).
2. **Replay** — backtest/replay_engine.py walks bars, holds OB state, yields alerts.
3. **Simulation + row build** — backtest/h1_only_simulator.py; row dict at h1_only_simulator.py:1141-1274.
4. **Insights** — backtest/insights.py (stats) + backtest/h1_only_reporting.py (sections, Excel tabs, trades.csv writer at :1208).
5. **Email/report** — backtest/reporting_email.py + backtest/render_report.py.

## Population caveats

- **RESOLVED 2026-07-03:** Fix 1 (150-bar live-window clamp) and Fix 2 (ATR-scaled dedupe) are in code. All pre-fix runs remain non-comparable; the first post-fix full run is the new baseline.

## trades.csv columns

Row build: h1_only_simulator.py:1141-1274. "src" = the line the value is read/written in the row dict.

| column | src (h1_only_simulator.py) | stamped when | status | note |
|---|---|---|---|---|
| pair | :1142 | alert | verified | yield payload scalar (replay_engine.py:528) |
| alert_ts | :1143 | alert | verified | = h1_ts of the closed-bar scan (replay_engine.py:529); alert>bos guard at replay_engine.py:461-491 |
| alert_bar_ts | :1144 | alert | verified | yield payload scalar (replay_engine.py:544) — just-closed bar's ts |
| alert_seq | :1148 | alert | verified | yield payload scalar from fire_count (replay_engine.py:534); traded row is always first fire per zone (dedupe run_backtest.py:157-165) |
| bos_timestamp | :1149 | OB build (immutable fact) | verified | 2026-07-02 detection audit 3e classification; alert>bos guard replay_engine.py:461-491 |
| fill_ts | :1150 | fill | verified | 2026-07-02 truth-chain audit |
| exit_ts | :1151 | exit | verified | 2026-07-02 truth-chain audit |
| direction | :1152 | OB build (immutable) | verified | detection audit 3e: immutable event fact |
| bias | :1153 | derived from direction | verified | deterministic at :1153 |
| model | :1154 | constant | verified | literal "h1_only" |
| event | :1155 | OB build | verified | deterministic label from immutable bos_tag/bos_tier |
| entry_zone | :1156 | alert | verified | proximal only (run_backtest.py:345); truth-chain audit (fills) |
| entry | :1157 | alert | verified | truth-chain audit (fills) |
| sl_raw | :1161 | alert | verified | 2026-07-02 truth-chain audit |
| sl_initial | :1162 | alert | verified | 2026-07-02 truth-chain audit |
| tp1 / tp2 | :1163-1164 | alert (levels on 200-bar window) | **fixed** | T5: compute_phase2_levels now fed the last LIVE_P2_H1_BARS=200 closed bars via _closed_bars_at_alert (h1_only_simulator.py), matching live P2's fetch depth; was unbounded history (TP selection drifted with run start). Guard tests/test_p2_window_clamp.py PASS |
| tp1_rr / tp2_rr | :1165-1166 | alert (levels on 200-bar window) | **fixed** | T5: same 200-bar clamp as tp1/tp2 (RR derived from the clamped-window TP pick) |
| exit_price | :1167 | exit | verified | 2026-07-02 truth-chain audit |
| exit_reason | :1168 | exit | verified | 2026-07-02 truth-chain audit |
| r_realised | :1169 | exit | verified | 2026-07-02 truth-chain audit; P&L source of truth |
| r_if_exit_tp1 / r_if_exit_tp2 | :1170-1171 | exit | verified | 2026-07-02 truth-chain audit |
| pnl_usd | :1172 (calc :1129) | exit | verified | r_realised × risk_usd |
| mfe_r / mae_r | :1173-1174 | exit | verified | 2026-07-02 truth-chain audit (excursions) |
| sl_bar_was_sweep | :1181 | SL exit | verified | 2026-07-02 truth-chain audit |
| sl_swept_then_tp1 | :1182 | SL exit | verified | 2026-07-02 truth-chain audit; hint only, peak-metric gate |
| be_arm_bar_touched_entry | :1188 | BE arm | verified | 2026-07-02 truth-chain audit |
| ob_to_fill_hours | :1190 (calc :951-958) | fill | verified | deterministic from immutable ob_timestamp + fill_ts; None if never filled |
| bars_break_to_pullback | :1192 (calc :964-973) | fill | verified | bar count bos_ts→fill on df index; deterministic from immutable inputs |
| bars_to_exit / bars_to_tp1 / bars_to_tp2 | :1193-1195 | exit | verified | 2026-07-02 truth-chain audit |
| ob_age_h1_bars | :1196 (calc :330-346) | alert | verified | bar count on df index ob_ts→alert_ts; deterministic, weekend-gap-safe |
| ob_timestamp | :1197 | OB build (immutable) | verified | detection audit 3e: immutable event fact |
| pd_zone | :1198 (calc :205-223) | row build, entry vs frozen DR | verified | 0.5 split, deterministic; dealing_range frozen-by-design (live same, smc_radar.py:961-985) |
| pd_alignment | :1199 (calc :226-243) | row build | verified | deterministic from bias + pd_zone |
| pd_pct | :1200 (calc :1055-1064) | row build, entry vs frozen DR | verified | deterministic; None when DR invalid |
| reversal_pct | :1204 (src :1073) | structure event (frozen) | verified | detection audit 3e: immutable; epoch-drift PARKED per spec (near-zero after Fix 1) |
| reversed_from_extreme | :1205 (calc :1082-1086) | row build, derived | verified | deterministic; 0.0 ambiguity documented at :1077-1081 |
| score | :1206 (calc :349-385) | simulate time, on alert-time ob_view (payload) | **fixed** | T1+T4 view feeds run_scorecard alert-time bos_verdict + payload touches_at_alert/fvg_at_alert; T5 clamps the scoring slice to LIVE_P2_H1_BARS=200 via _closed_bars_at_alert. Guards test_ob_alert_freeze.py + test_p2_window_clamp.py PASS |
| structure_pts | :1207 | simulate time, on alert-time ob_view | **fixed** | smc_detector.py:2090 now reads the alert-time verdict via ob_view (T1) |
| sweep_pts | :1208 | alert | out-of-scope | sweep rebuild workstream |
| fvg_pts | :1209 | simulate time, on alert-time ob_view (payload) | **fixed** | Fix 3c/3d + T4: fvg_at_alert now travels in the alert PAYLOAD (not the re-stamped dict), routed into scoring via ob_view. Guard test_ob_alert_freeze.py T4 cases PASS |
| freshness_pts | :1210 | simulate time, on alert-time ob_view (payload) | **fixed** | Fix 3a/3d + T4: touches_at_alert now travels in the alert PAYLOAD, routed into scoring via ob_view. Guard test_ob_alert_freeze.py T4 cases PASS |
| killzone_pts | :1211 | simulate time | verified | scored on immutable ob_timestamp via shared DST-aware ts_in_killzone (smc_detector.py:2238-2239); max 1 |
| confluences_present | :1212 (calc :246-257) | simulate time | **fixed** | derived from breakdown, now computed on alert-time ob_view (T1 + Fix 3) |
| session | :1213 (calc :82-90) | alert | verified | fixed UTC-hour buckets. CAVEAT: UTC-fixed, NOT DST-aware — session edges shift 1h half the year (killzone columns ARE DST-aware; known asymmetry) |
| weekend_blocked | :1218 (calc :142-172) | fill | verified | deterministic IST calc from fill_ts + config; False for non-crypto |
| sl_collision | :1219 (stamp :659, :804-809) | exit | verified | SL-first collision rule; truth-chain audit (exits) |
| bos_tag / bos_tier | :1220-1221 | OB build (immutable) | verified | detection audit 3e: immutable event facts |
| bos_verdict | :1227 | alert (payload scalar via ob_view) | **fixed** | T1 SHIPPED 2026-07-03; guard tests/test_ob_alert_freeze.py T1 cases + source tripwire PASS |
| bos_sequence_count | :1233 | OB build (immutable) | verified | detection audit 3e: immutable event fact |
| break_tier / break_close_atr / break_excess / break_body_atr | :1234-1237 | one-time re-grade when window complete | **fixed** | Fix 3b LANDED (_bq_regraded in replay_engine.py); verify distribution shift at baseline review |
| ob_range_atr | :1239 (calc :1102) | row build from frozen high/low | verified | deterministic from immutable high/low + frozen h1_atr |
| fvg_size_atr | :1240 (calc :1107-1111) | row build from refreshed fvg | **fixed** | Fix 3c LANDED (fvg refresh on re-surface); alert-time snapshot guard PASS |
| impulse_leg_atr | :1241 (calc :1119-1122) | row build from frozen prices | verified | deterministic from immutable impulse_start_price + bos_swing_price |
| atr_at_ob | :1242 (calc :1127) | OB build (frozen by design) | verified | formation ATR, frozen BY DESIGN (detection audit 3e) |
| ob_body_ratio | h1_only_simulator.py:1144 (read) / :1286 (emit), stamped smc_radar.py:1117 | OB build (frozen by design) | verified | chosen OB candle's own body/range, walk-back loop smc_radar.py:883-901; range>0 guarded (is_valid_ob_candle already rejects range==0); None for legacy zones (pre-change); observe-only per DECISION_GUARDRAILS.md A3, no gate |
| ob_walkback_depth | h1_only_simulator.py:1145 (read) / :1287 (emit), stamped smc_radar.py:1124 | OB build (frozen by design) | verified | oversized_count + undersized_count + doji_count at acceptance (smc_radar.py:878-901); 0 = first candidate passed; sums ALL three skip reasons so knob-sweep (MIN_OB_RANGE_ATR_MULT>0) is counted; None for legacy zones; observe-only per DECISION_GUARDRAILS.md A3, no gate |
| fvg_present | :1243 | alert (fvg_at_alert, payload) | **fixed** | Fix 3c/3d + T4: stamped at the yield as a PAYLOAD scalar (survives re-fire re-stamping of the shared dict); ob_view overwrites both fvg + fvg_at_alert keys. Guard test_ob_alert_freeze.py T4 cases PASS |
| fvg_state | :1246 (helper) | row build (payload fvg via ob_view) | **fixed** | Fix 3d + T4: helper classifies against the payload fvg_at_alert carried on ob_view; guard test_ob_alert_freeze.py PASS |
| fvg_mitigation | :1250 | alert (fvg_at_alert, payload) | **fixed** | Fix 3c/3d + T4: payload scalar via ob_view. Guard test_ob_alert_freeze.py T4 cases PASS |
| ob_touches | :1255 | alert (touches_at_alert, payload) | **fixed** | Fix 3a/3d + T4: stamped at the yield as a PAYLOAD scalar (survives re-fire re-stamping); ob_view overwrites both touches + touches_at_alert keys. Guard test_ob_alert_freeze.py T4 cases PASS. Meaning: touches between structural event and ALERT |
| sweep_present | :1256 | frozen snapshot | out-of-scope | sweep rebuild workstream |
| ob_session | :1263 (calc :108-112) | OB build ts | verified | deterministic from immutable ob_timestamp; UTC-fixed caveat as session |
| fill_session | :1264 (calc :115-121) | fill | verified | CAVEAT: never_filled rows fall back to ALERT hour (:118-120) — population mixes fill-time and alert-time labels; never_filled rows are audit-only |
| ob_in_killzone | :1265 (calc :124-139, :175-176) | OB build ts | verified | DST-aware via shared live engine smc_detector.ts_in_killzone (:137) |
| fill_in_killzone | :1266 (calc :179-180) | fill | verified | same shared DST-aware engine |
| killzone_alignment | :1267 (calc :183-202) | derived | verified | deterministic 4-way bucket + never_filled |
| h1_trend | :1268 | alert | verified | yield payload scalar (replay_engine.py:536) — immune to post-alert OB mutation |
| trend_alignment | :1269 | alert | verified | derived at fire (replay_engine.py:511-515), passed as payload scalar (:537) |
| setup_badge / setup_badge_kind | :1272-1273 (calc via ob_view) | row build, live classifier on alert-time view | **fixed** | classify_setup (smc_detector.py:2459-2466) now receives alert-time touches/fvg via ob_view (T1); sweep input stays sweep-workstream-owned |
| news_blocked / news_event_title / news_event_currency / news_event_source / news_event_ts | run_backtest.py:201-205 | post-walk, from frozen alert_ts | verified (stamping) | class-safe (deterministic fn of frozen alert_ts). INFORMATIONAL — does NOT gate headline (h1_only_reporting.py:162-173). Window fn internals not line-audited (gate flag, not a metric) |
| ist_blocked | run_backtest.py:206 (calc :189) | post-walk, from frozen alert_ts | verified (stamping) | HARD gate — rows dropped from everything except audit (:3235-3237, :191-192) |
| killzone_blocked | run_backtest.py:207 (calc :173) | post-walk, from frozen alert_ts | verified (stamping) | informational; does not gate |
| killzone_windows | run_backtest.py:208 | config label | verified | static label |
| setup_id | h1_only_reporting.py:3246-3249 | report build, pre-CSV | verified | atomic counter; same ID in CSV/Excel/HTML/email |
| eligible_for_headline / headline_exclusion | h1_only_reporting.py:1215-1223 | CSV write, from THE one rule (:142-173) | verified | self-describing headline membership; sum(pnl_usd where eligible) == headline, enforced by reconcile assert :3354-3365 |
| alert_utc_hour | run_backtest.py:209 | from frozen alert_ts | verified | int(alert_ts.hour), deterministic |

## Defects found by THIS audit (not in DETECTION_FIXES_SPEC.md)

### D1 — bos_verdict logs the LAST fire's verdict, not the traded alert's (2026-07-02)

- Evidence chain:
  - run_backtest.py:148-153 — the replay generator is FULLY drained into `pair_alerts` before any simulation.
  - run_backtest.py:158-183 — rows are built AFTER the whole walk, but the yield passed `ob` by reference (replay_engine.py:532).
  - replay_engine.py:500 — `ob["bos_verdict"]` is re-assigned on EVERY fire of the same zone.
  - run_backtest.py:157-165 — only the FIRST fire per zone is traded; its row (h1_only_simulator.py:1227) reads whatever the LAST fire stamped.
- Impact: multi-fire zones can log a verdict from hours/days after the traded alert (holding↔fading flip). Single-fire zones unaffected.
- Class: identical to spec Fix 3d's class — mutable OB state read at row-build time. Correct fix: carry `bos_verdict` in the yield payload (like h1_trend) or stamp `bos_verdict_at_alert`; guard = extend the 3e regression test to mutate `bos_verdict` post-alert and assert the row keeps the alert-time value.
- **FIXED 2026-07-03 (T1):** verdict carried as yield-payload scalar (replay_engine.py fire block) + alert-time ob_view in simulate_h1_only_dual feeds scoring/badge/row. Guard: tests/test_ob_alert_freeze.py T1 cases incl. source tripwire — PASS. Pre-fix bos_verdict-keyed slices remain suspect for multi-fire zones until the baseline rerun.

### D2 — stale comment claims news gates aggregates — FIXED 2026-07-03

- Was: h1_only_reporting.py:1248-1250 claimed news_blocked rows were excluded from aggregates. Code truth: news never gates (:162-173). Comment corrected to point at the one rule.

### D3 — cross-run aggregator ignores headline eligibility (2026-07-03)

- Evidence chain:
  - aggregate_runs.py:44-73 — `_load_trades` reads trades.csv raw, no eligibility filter.
  - aggregate_runs.py:404 — `filled = ins._filled(primary)` drops ONLY never_filled (insights.py:32-36).
  - trades.csv contains timeout/window_end rows with force-closed r_realised (audit-only per the settled unresolved-trade policy) and ships `eligible_for_headline` exactly so consumers can filter (h1_only_reporting.py:1216-1223) — aggregate_runs never reads it.
- Impact: EVERY cross-run insight (compute_overall, instrument_verdicts, score_validation, pair_session_matrix, group_comparison, freshness, regime, generate_verdict) included arbitrary-price force-closed rows. Per-run email/Excel numbers were clean; the cross-run layer over-counted.
- **FIXED 2026-07-03 (T2):** `_eligible_mask` + hard assert in aggregate_runs.py; entry_zone fill-rate keeps never_filled only. Guard: tests/test_aggregate_eligibility.py (3 cases incl. string-bool round-trip + legacy CSV) — PASS. All prior combined/VERDICT outputs are stale until re-aggregated.

## Insights (backtest/insights.py + reporting sections)

Two consumers, two populations — this matters:
- **Per-run (Excel tabs / email)**: population = `trades_all` (IST+weekend hard-dropped at h1_only_reporting.py:3237) filtered to real fills. Matches headline. CLEAN.
- **Cross-run (aggregate_runs.py)**: loads raw trades.csv with NO eligibility filter (aggregate_runs.py:44-73, :404) → timeout/window_end force-closed rows FEED every aggregate insight = defect D3.

| insight | src | population | status | note |
|---|---|---|---|---|
| win_rate_pct | insights.py:43-64 | resolved only (BE excluded num+den), None when none resolved | verified | matches settled convention, line-read |
| bootstrap_ci / sharpe / max_drawdown_r / longest_losing_streak | insights.py:71-110 | caller's | verified | standard formulas, seeded rng(42); CI None under n=5 |
| capture_pct | insights.py:113-121 | wins only | verified | avg booked / avg MFE — peak-adjacent by design, guarded by is_peak_metric law |
| compute_overall | insights.py:128-156 | _filled(eligible) | **fixed** | T2 landed: eligibility mask + hard assert; guard tests/test_aggregate_eligibility.py PASS |
| pair_session_matrix | insights.py:163-194 | _filled(eligible) | **fixed** | T2; per-run tab was already clean; uses UTC-fixed `session` (parked caveat) |
| instrument_verdicts | insights.py:216-255 | _filled(eligible) | **fixed** | T2. PREMIUM_PAIRS still lists NAS100 (dropped pair — harmless, stale) |
| confluence_attribution | insights.py:262-323 | _filled(caller) | voided | fvg/freshness legs read corrupted-era columns (Fix 3); sweep leg out-of-scope; also D3 in aggregate |
| score_validation | insights.py:330-378 | _filled(caller) | voided | score embeds fvg/freshness pts (corrupted) + D1 structure taint; also D3 |
| setup_badge_validation | insights.py:387-454 | _filled(caller) | voided | badge from frozen touches/fvg/sweep (smc_detector.py:2459-2466); also D3 |
| group_comparison | insights.py:456-476 | _filled(eligible) | **fixed** | T2 |
| entry_zone_comparison | insights.py:483-509 | eligible + never_filled (fill-rate denominator) | **fixed** | T2; single proximal bucket now (degenerate, harmless) |
| ob_freshness_comparison | insights.py:527-552 | _filled, keyed on alert_seq | verified | CORRECTION: already re-keyed to alert_seq (verified column), NOT ob_touches. But structurally INERT: traded rows are always first-fire (run_backtest.py:157-165) so buckets 2/3 are empty by construction — zero rows is a population fact, not a finding about re-touched OBs. D3 in aggregate |
| regime_verification | insights.py:559-598 | _filled(eligible) per run_id | **fixed** | T2 |
| generate_verdict | insights.py:605-672 | upstream dicts | **fixed** (population) | T2; its score-issue line inherits voided score_v until baseline re-derive |
| verify_capturable / is_peak_metric | insights.py:708-716 | gate | verified | 2026-07-02 truth-chain audit (peak-vs-fill gate); law block at :675-705 |
| G1-G10 gates | reporting layer | headline | verified | 2026-07-02 truth-chain audit |
| headline exclusions (_headline_exclusion) | h1_only_reporting.py:142 | audit-only rows out of P&L | verified | 2026-07-02 truth-chain audit + G1 blocked-rows fix |
| killzone alignment table | h1_only_reporting.py:537 | prox_trades (verified population, :3256) | verified | aggregates verified killzone_alignment column over headline population |
| driver buckets / two-way | h1_only_reporting.py:806, :861 | prox_trades | voided (partial) | fvg/freshness/badge/bos_verdict dims read corrupted columns; clean dims (session/killzone/pd) fine — re-derive whole section post-baseline |
| counterfactual exits | h1_only_reporting.py:619 | prox_trades | verified | population verified; exit replay = truth-chain audit scope |
| Act 1-6 sections | h1_only_reporting.py:1925-2967 | prox_trades (verified) | verified (population) | aggregation of audited columns over headline population; sections quoting score/badge/fvg slices are void until baseline rerun |
| recipe ranking | h1_only_reporting.py:2722-2882 | exit-lab sink (block-stamped, run_backtest.py:196-198) | verified (population) | sink rows carry ist/weekend flags so the table drops what the headline drops (826-vs-668 RCA fix) |
| zone register | h1_only_reporting.py:1120 | trades_all | verified (population) | |
| Excel tabs (confluences/badges/pair-session/break-ladder) | h1_only_reporting.py:1388-1480 | filled(trades_all) = headline population (:1489, :3237) | voided (partial) | population CLEAN; confluence/badge/break-ladder tabs read corrupted-era columns → void until baseline rerun; pair-session tab verified |

## Rules

- No new trades.csv column or emitted insight ships without: (1) a row here, (2) a guard (test/gate) that kills its bug CLASS. (CLAUDE.md rule: ADDED 2026-07-02, Logging section.)
- Insights on pending-fix columns are automatically VOID until the first post-fix baseline run.
- Sweep detector columns/insights: owned by the sweep-rebuild chat. Not audited here.
