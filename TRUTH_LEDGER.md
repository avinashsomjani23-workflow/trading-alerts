# TRUTH LEDGER

One line per trades.csv column and per emitted insight. A number is trustworthy only
when its row here says **verified** or **fixed** — anything else is unproven.

- **verified** — live code path traced end-to-end with file:line evidence; stamped on the right bar, frozen correctly, population known.
- **fixed** — defect found, fix shipped WITH a structural guard (test/gate/whitelist).
- **voided** — computed on a corrupted-era column; invalid until re-derived from a post-fix baseline run.
- **pending** — not yet audited. Assume nothing.
- **pending-fix** — defect known (spec written), fix NOT yet in code.
- **out-of-scope** — owned by another workstream (sweep rebuild).

Status date: 2026-07-04 (line refs refreshed to HEAD after T4/T5 landed).
DETECTION_FIXES_SPEC.md Fixes 1/2/3 **LANDED**; TRUTH_FIXES_SPEC.md T1/T2/T3
**SHIPPED**; TRUTH_FIXES_SPEC_2.md T4/T5 **SHIPPED**. Guards green
(tests/test_dedupe.py, test_ob_alert_freeze.py, test_aggregate_eligibility.py,
test_truth_ledger.py, test_p2_window_clamp.py, test_structure_signals.py — full
suite 77 passed 2026-07-04). Voided insights stay VOID until the first post-fix
baseline run.

## Layer map (where numbers are born)

1. **Detection** — smc_radar.py / smc_detector.py / dealing_range.py (shared with live).
2. **Replay** — backtest/replay_engine.py walks bars, holds OB state, yields alerts.
3. **Simulation + row build** — backtest/h1_only_simulator.py; `_build_row` at :1075, row dict at :1250-1405.
4. **Insights** — backtest/insights.py (stats) + backtest/h1_only_reporting.py (sections, Excel tabs, trades.csv writer `_trades_csv` at :1217).
5. **Email/report** — backtest/reporting_email.py + backtest/render_report.py.

## Population caveats

- **RESOLVED 2026-07-03:** Fix 1 (150-bar live-window clamp) and Fix 2 (ATR-scaled dedupe) are in code. All pre-fix runs remain non-comparable; the first post-fix full run is the new baseline.

## trades.csv columns

Row build: h1_only_simulator.py `_build_row` at :1075; row dict at :1250-1405. "src" = the line the value is read/written in the row dict.

| column | src (h1_only_simulator.py) | stamped when | status | note |
|---|---|---|---|---|
| pair | :1251 | alert | verified | yield payload scalar (replay_engine.py:648) |
| alert_ts | :1252 | alert | verified | = h1_ts of the closed-bar scan (replay_engine.py:649); alert>bos guard at replay_engine.py:529-554 |
| alert_bar_ts | :1253 | alert | verified | yield payload scalar (replay_engine.py:687) — just-closed bar's ts |
| alert_seq | :1257 | alert | verified | yield payload scalar from fire_count (replay_engine.py:654); traded row is always first fire per zone (dedupe run_backtest.py:161-169) |
| bos_timestamp | :1258 | OB build (immutable fact) | verified | 2026-07-02 detection audit 3e classification; alert>bos guard replay_engine.py:529-554 |
| fill_ts | :1259 | fill | verified | 2026-07-02 truth-chain audit |
| exit_ts | :1260 | exit | verified | 2026-07-02 truth-chain audit |
| direction | :1261 | OB build (immutable) | verified | detection audit 3e: immutable event fact |
| bias | :1262 | derived from direction | verified | deterministic at :1262 |
| model | :1263 | constant | verified | literal "h1_only" |
| event | :1264 | OB build | verified | deterministic label from immutable bos_tag/bos_tier |
| entry_zone | :1265 | alert | verified | proximal only (run_backtest.py:345); truth-chain audit (fills) |
| entry | :1266 | alert | verified | truth-chain audit (fills) |
| sl_raw | :1270 | alert | verified | 2026-07-02 truth-chain audit |
| sl_initial | :1271 | alert | verified | 2026-07-02 truth-chain audit |
| tp1 / tp2 | :1272-1273 | alert (levels on 200-bar window) | **fixed** | T5: compute_phase2_levels now fed the last LIVE_P2_H1_BARS=200 closed bars via _closed_bars_at_alert (h1_only_simulator.py:349, used at :504), matching live P2's fetch depth; was unbounded history (TP selection drifted with run start). Guard tests/test_p2_window_clamp.py PASS |
| tp1_rr / tp2_rr | :1274-1275 | alert (levels on 200-bar window) | **fixed** | T5: same 200-bar clamp as tp1/tp2 (RR derived from the clamped-window TP pick) |
| exit_price | :1276 | exit | verified | 2026-07-02 truth-chain audit |
| exit_reason | :1277 | exit | verified | 2026-07-02 truth-chain audit |
| r_realised | :1278 | exit | verified | 2026-07-02 truth-chain audit; P&L source of truth |
| r_if_exit_tp1 / r_if_exit_tp2 | :1279-1280 | exit | verified | 2026-07-02 truth-chain audit |
| pnl_usd | :1281 | exit | verified | r_realised × risk_usd |
| mfe_r / mae_r | :1282-1283 | exit | verified | 2026-07-02 truth-chain audit (excursions) |
| sl_bar_was_sweep | :1290 | SL exit | verified | 2026-07-02 truth-chain audit |
| sl_swept_then_tp1 | :1291 | SL exit | verified | 2026-07-02 truth-chain audit; hint only, peak-metric gate |
| be_arm_bar_touched_entry | :1297 | BE arm | verified | 2026-07-02 truth-chain audit; be_eps FP-boundary tolerance at :670 (2026-07-03 fix, G10 rule b) |
| ob_to_fill_hours | :1299 (calc :951-958) | fill | verified | deterministic from immutable ob_timestamp + fill_ts; None if never filled |
| bars_break_to_pullback | :1301 (calc :964-973) | fill | verified | bar count bos_ts→fill on df index; deterministic from immutable inputs |
| bars_to_exit / bars_to_tp1 / bars_to_tp2 | :1302-1304 | exit | verified | 2026-07-02 truth-chain audit |
| ob_age_h1_bars | :1305 (calc :330-346) | alert | verified | bar count on df index ob_ts→alert_ts; deterministic, weekend-gap-safe |
| ob_timestamp | :1306 | OB build (immutable) | verified | detection audit 3e: immutable event fact |
| pd_zone | :1307 (calc :205-223) | row build, entry vs frozen DR | verified | 0.5 split, deterministic; dealing_range frozen-by-design (live same, smc_radar.py:1042-1052) |
| pd_alignment | :1308 (calc :226-243) | row build | verified | deterministic from bias + pd_zone |
| pd_pct | :1309 (calc :1055-1064) | row build, entry vs frozen DR | verified | deterministic; None when DR invalid |
| reversal_pct | :1313 (src :1073) | structure event (frozen) | verified | detection audit 3e: immutable; epoch-drift PARKED per spec (near-zero after Fix 1) |
| reversed_from_extreme | :1314 (calc :1082-1086) | row build, derived | verified | deterministic; 0.0 ambiguity documented at :1077-1081 |
| score | :1315 (calc :367-385, slice :382) | simulate time, on alert-time ob_view (payload) | **fixed** | T1+T4 view feeds run_scorecard alert-time bos_verdict + payload touches_at_alert/fvg_at_alert; T5 clamps the scoring slice to LIVE_P2_H1_BARS=200 via _closed_bars_at_alert (:382). Guards test_ob_alert_freeze.py + test_p2_window_clamp.py PASS |
| structure_pts | :1316 | simulate time, on alert-time ob_view | **fixed** | smc_detector.py:2110 now reads the alert-time verdict via ob_view (T1) |
| sweep_pts | :1317 | alert | out-of-scope | sweep rebuild workstream |
| fvg_pts | :1318 | simulate time, on alert-time ob_view (payload) | **fixed** | Fix 3c/3d + T4: fvg_at_alert now travels in the alert PAYLOAD (not the re-stamped dict), routed into scoring via ob_view. Guard test_ob_alert_freeze.py T4 cases PASS |
| freshness_pts | :1319 | simulate time, on alert-time ob_view (payload) | **fixed** | Fix 3a/3d + T4: touches_at_alert now travels in the alert PAYLOAD, routed into scoring via ob_view. Guard test_ob_alert_freeze.py T4 cases PASS |
| killzone_pts | :1320 | simulate time | verified | scored on immutable ob_timestamp via shared DST-aware ts_in_killzone (smc_detector.py:1937); max 1 |
| confluences_present | :1321 (calc :246-257) | simulate time | **fixed** | derived from breakdown, now computed on alert-time ob_view (T1 + Fix 3) |
| session | :1322 (calc :82-90) | alert | verified | fixed UTC-hour buckets. CAVEAT: UTC-fixed, NOT DST-aware — session edges shift 1h half the year (killzone columns ARE DST-aware; known asymmetry) |
| weekend_blocked | :1327 (calc :142-172) | fill | verified | deterministic IST calc from fill_ts + config; False for non-crypto |
| sl_collision | :1328 (stamp :674, :829) | exit | verified | SL-first collision rule; truth-chain audit (exits) |
| bos_tag / bos_tier | :1329-1330 | OB build (immutable) | verified | detection audit 3e: immutable event facts |
| bos_verdict | :1336 | alert (payload scalar via ob_view) | **fixed** | T1 SHIPPED 2026-07-03; payload scalar at replay_engine.py:673; guard tests/test_ob_alert_freeze.py T1 cases + source tripwire PASS |
| bos_sequence_count | :1342 | OB build (immutable) | verified | detection audit 3e: immutable event fact |
| break_tier / break_close_atr / break_excess / break_body_atr | :1343-1346 | one-time re-grade when window complete | **fixed** | Fix 3b LANDED (_bq_regraded in replay_engine.py:373-381); verify distribution shift at baseline review |
| ob_range_atr | :1348 (calc :1102) | row build from frozen high/low | verified | deterministic from immutable high/low + frozen h1_atr |
| fvg_size_atr | :1349 (calc :1107-1111) | row build from refreshed fvg | **fixed** | Fix 3c LANDED (fvg refresh on re-surface, replay_engine.py:424); alert-time snapshot guard PASS |
| impulse_leg_atr | :1350 (calc :1119-1122) | row build from frozen prices | verified | deterministic from immutable impulse_start_price + bos_swing_price |
| atr_at_ob | :1351 (calc :1127) | OB build (frozen by design) | verified | formation ATR, frozen BY DESIGN (detection audit 3e) |
| ob_body_ratio | h1_only_simulator.py:1180 (read) / :1353 (emit), stamped smc_radar.py:1117; LIVE persist zone.py `body_ratio` (_FIELD_ORDER:67, from_fresh:166, frozen in refresh) | OB build (frozen by design) | fixed | chosen OB candle's own body/range, walk-back loop smc_radar.py:880-901; range>0 guarded (is_valid_ob_candle already rejects range==0); observe-only per DECISION_GUARDRAILS.md A3, no gate. FIX 2026-07-06: zone.py `_FIELD_ORDER`+`from_fresh` did NOT carry it → dropped to null on active_obs.json (backtest read raw OB dict, was fine; LIVE was blind). Now persisted + frozen on refresh (never re-stamped) + back-filled once for legacy zones. Guard: test_zone_roundtrip.py test_walkback_fields_persist_and_freeze. None only for zones written before this fix, until they re-fire. |
| ob_walkback_depth | h1_only_simulator.py:1181 (read) / :1354 (emit), stamped smc_radar.py:1124; LIVE persist zone.py `walkback_depth` (_FIELD_ORDER:67, from_fresh:167, frozen in refresh) | OB build (frozen by design) | fixed | oversized_count + undersized_count + doji_count at acceptance (smc_radar.py:880-901); 0 = first candidate passed; sums ALL three skip reasons so knob-sweep (MIN_OB_RANGE_ATR_MULT>0) is counted; observe-only per DECISION_GUARDRAILS.md A3, no gate. FIX 2026-07-06: same zone.py drop bug as ob_body_ratio → was null on active_obs.json live. Now persisted + frozen + back-filled once. Guard: test_zone_roundtrip.py test_walkback_fields_persist_and_freeze. |
| fvg_present | :1355 | alert (fvg_at_alert, payload) | **fixed** | Fix 3c/3d + T4: stamped at the yield as a PAYLOAD scalar (replay_engine.py:680, survives re-fire re-stamping of the shared dict); ob_view overwrites both fvg + fvg_at_alert keys. Guard test_ob_alert_freeze.py T4 cases PASS |
| fvg_state | :1358 (helper) | row build (payload fvg via ob_view) | **fixed** | Fix 3d + T4: helper classifies against the payload fvg_at_alert carried on ob_view; guard test_ob_alert_freeze.py PASS |
| fvg_mitigation | :1362 | alert (fvg_at_alert, payload) | **fixed** | Fix 3c/3d + T4: payload scalar via ob_view. Guard test_ob_alert_freeze.py T4 cases PASS |
| ob_touches | :1366 | alert (touches_at_alert, payload) | **fixed** | Fix 3a/3d + T4: stamped at the yield as a PAYLOAD scalar (replay_engine.py:679, survives re-fire re-stamping); ob_view overwrites both touches + touches_at_alert keys. Guard test_ob_alert_freeze.py T4 cases PASS. Meaning: touches between structural event and ALERT |
| sweep_present | :1367 | frozen snapshot | out-of-scope | sweep rebuild workstream |
| ob_session | :1374 (calc :108-112) | OB build ts | verified | deterministic from immutable ob_timestamp; UTC-fixed caveat as session |
| fill_session | :1375 (calc :115-121) | fill | verified | CAVEAT: never_filled rows fall back to ALERT hour (:118-120) — population mixes fill-time and alert-time labels; never_filled rows are audit-only |
| ob_in_killzone | :1376 (calc :124-139, :175-176) | OB build ts | verified | DST-aware via shared live engine smc_detector.ts_in_killzone (:137) |
| fill_in_killzone | :1377 (calc :179-180) | fill | verified | same shared DST-aware engine |
| killzone_alignment | :1378 (calc :183-202) | derived | verified | deterministic 4-way bucket + never_filled |
| h1_trend | :1379 | alert | verified | yield payload scalar (replay_engine.py:656) — immune to post-alert OB mutation |
| trend_alignment | :1380 | alert | **fixed** | 2026-07-05 parity fix: now derived at fire via the SHARED helper smc_detector.derive_trend_alignment (replay_engine.py:607) — SAME implementation live Phase 2 uses (Phase2_Alert_Engine.py:2492), fed the flip-pending state so a with-trend read is demoted to counter_trend/ambiguous while a CHoCH flip is pending. Vocabulary unified on live's values (with_trend/counter_trend/ambiguous); the old backtest dialect (against_trend/no_trend) is GONE. Passed as payload scalar (:660). Guards: tests/test_structure_signals.py (branch table + live-parity + single-implementation source-assert). Pre-fix runs' values non-comparable |
| structure_ranging_at_alert | :1385 (payload) | alert (payload scalar) | verified | S2: `walls["structure_v2"]["ranging"]` snapshotted at the replay yield (replay_engine.py:661, `_sv2.get("ranging")` at :610); every row, None only if structure_v2 missing (degraded walls). Ranging counter now flip-reset (S1). Guard test_structure_signals.py + freeze case |
| flip_pending_at_alert | :1386 (payload) | alert (payload scalar) | verified | S2: `structure_v2["flip_unconfirmed"]` at yield (replay_engine.py:662, :611); a CHoCH armed but not yet Confirmation-BOS. Every row, None if structure_v2 missing. Guard test_structure_signals.py + freeze case |
| flip_pending_dir_at_alert | :1387 (payload) | alert (payload scalar) | verified | S2: `structure_v2["choch_pending_dir"]` mapped up→bullish/down→bearish (replay_engine.py:663, :607-612); None when no pending flip OR structure_v2 missing. Guard test_structure_signals.py |
| leg_extreme_at_alert | :1393 (payload) | alert (payload scalar) | verified | S3 SUPPORT (not screened): a-priori leg extreme (max High / min Low) over closed slice bars ts>=ob_timestamp, computed in replay fire block (has h1_slice) + carried as payload scalar (replay_engine.py:667). None if ob_timestamp missing or slice empty. Guard test_structure_signals.py + freeze case |
| leg_retrace_pct_at_alert | :1394 (calc in _build_row) | alert-derived (payload extreme + placed entry) | verified | S3: (leg_extreme−entry)/(leg_extreme−impulse_start)×100 for bullish, mirrored bearish; from leg_extreme_at_alert (payload) + entry (placed limit) + frozen impulse_start_price. None when extreme unstamped / impulse_start missing / denominator≤0. >100 VALID, not clamped. Guard test_structure_signals.py math cases + freeze case |
| leg_extreme_clipped | :1395 (payload) | alert (payload scalar) | verified | S3 SUPPORT (not screened): True when ob_timestamp predates the point-in-time slice start (extreme may be understated); keeps leg_retrace honest (replay_engine.py:668). None on measurement failure. Guard test_structure_signals.py |
| dr_ceiling_broken_at_ob | :1399 (read from frozen dr) | OB build (frozen snapshot) | verified | S4: read off frozen ob["dealing_range"]["ceiling_broken"], sourced get_dealing_range valid branch (smc_detector.py) ← compute_pd_position ceiling_is_placeholder ← h4_range ceiling_broken (h4_range.py:233). None when snapshot invalid/legacy. Frozen at OB build (no re-stamp). Guard test_structure_signals.py snapshot case |
| dr_floor_broken_at_ob | :1400 (read from frozen dr) | OB build (frozen snapshot) | verified | S4: read off frozen ob["dealing_range"]["floor_broken"], same chain as dr_ceiling_broken_at_ob (h4_range floor_broken, h4_range.py:234). None when snapshot invalid/legacy. Guard test_structure_signals.py snapshot case |
| setup_badge / setup_badge_kind | :1403-1404 (calc via ob_view) | row build, live classifier on alert-time view | **fixed** | classify_setup (smc_detector.py:2462, reads touches/fvg :2479-2484) now receives alert-time touches/fvg via ob_view (T1); sweep input stays sweep-workstream-owned. 2026-07-05: its trend_alignment input (:2510 keys off `== 'with_trend'`) now comes from the shared derive_trend_alignment, so the badge no longer diverges from live when a CHoCH flip is pending. Guard: tests/test_structure_signals.py trend-alignment parity. Pre-fix runs' values non-comparable |
| news_blocked / news_event_title / news_event_currency / news_event_source / news_event_ts | run_backtest.py:205-209 | post-walk, from frozen alert_ts | verified (stamping) | class-safe (deterministic fn of frozen alert_ts). INFORMATIONAL — does NOT gate headline (h1_only_reporting.py:142-176). **Event `ts_utc` was +4h wrong pre-2026-07-06 (D4): FF feed time is UTC, was mis-stamped as US Eastern — news_filter.py:152-158 now stamps UTC directly, guard test_ff_xml_time_is_utc. Pre-fix news_blocked is on wrong rows.** |
| ist_blocked | run_backtest.py:210 (calc :193) | post-walk, from frozen alert_ts | verified (stamping) | HARD gate — rows dropped from everything except audit (:201, h1_only_reporting.py _headline_exclusion :142-176) |
| killzone_blocked | run_backtest.py:211 (calc :177) | post-walk, from frozen alert_ts | verified (stamping) | informational; does not gate |
| killzone_windows | run_backtest.py:212 | config label | verified | static label |
| setup_id | h1_only_reporting.py:3323-3333 | report build, pre-CSV | verified | atomic counter (lock file, :33); same ID in CSV/Excel/HTML/email |
| eligible_for_headline / headline_exclusion | h1_only_reporting.py:1230-1231 | CSV write, from THE one rule _headline_exclusion (:142-176) | verified | self-describing headline membership; sum(pnl_usd where eligible) == headline, enforced by reconcile assert :3435-3449 |
| alert_utc_hour | run_backtest.py:213 | from frozen alert_ts | verified | int(alert_ts.hour), deterministic |

## Defects found by THIS audit (not in DETECTION_FIXES_SPEC.md)

### D1 — bos_verdict logs the LAST fire's verdict, not the traded alert's (2026-07-02)

- Evidence chain:
  - run_backtest.py:144-159 — the replay generator is FULLY drained into `pair_alerts` before any simulation.
  - run_backtest.py:162+ — rows are built AFTER the whole walk, but the yield passed `ob` by reference (replay_engine.py:652).
  - replay_engine.py:567 — `ob["bos_verdict"]` is re-assigned on EVERY fire of the same zone.
  - run_backtest.py:161-169 — only the FIRST fire per zone is traded; its row (h1_only_simulator.py:1336) reads whatever the LAST fire stamped.
- Impact: multi-fire zones can log a verdict from hours/days after the traded alert (holding↔fading flip). Single-fire zones unaffected.
- Class: identical to spec Fix 3d's class — mutable OB state read at row-build time. Correct fix: carry `bos_verdict` in the yield payload (like h1_trend) or stamp `bos_verdict_at_alert`; guard = extend the 3e regression test to mutate `bos_verdict` post-alert and assert the row keeps the alert-time value.
- **FIXED 2026-07-03 (T1):** verdict carried as yield-payload scalar (replay_engine.py fire block) + alert-time ob_view in simulate_h1_only_dual feeds scoring/badge/row. Guard: tests/test_ob_alert_freeze.py T1 cases incl. source tripwire — PASS. Pre-fix bos_verdict-keyed slices remain suspect for multi-fire zones until the baseline rerun.

### D2 — stale comment claims news gates aggregates — FIXED 2026-07-03

- Was: h1_only_reporting.py claimed news_blocked rows were excluded from aggregates. Code truth: news never gates (the one rule is _headline_exclusion :142-176). Comment corrected to point at the one rule (now at :1260-1261 and :3414).

### D3 — cross-run aggregator ignores headline eligibility (2026-07-03)

- Evidence chain (pre-fix line refs; the raw-load bug is what T2 closed):
  - aggregate_runs.py:73 — `_load_trades` reads trades.csv raw, no eligibility filter.
  - trades.csv contains timeout/window_end rows with force-closed r_realised (audit-only per the settled unresolved-trade policy) and ships `eligible_for_headline` exactly so consumers can filter (h1_only_reporting.py:1230-1231) — pre-fix aggregate_runs never read it.
- Impact: EVERY cross-run insight (compute_overall, instrument_verdicts, score_validation, pair_session_matrix, group_comparison, freshness, regime, generate_verdict) included arbitrary-price force-closed rows. Per-run email/Excel numbers were clean; the cross-run layer over-counted.
- **FIXED 2026-07-03 (T2):** `_eligible_mask` + hard assert in aggregate_runs.py; entry_zone fill-rate keeps never_filled only. Guard: tests/test_aggregate_eligibility.py (3 cases incl. string-bool round-trip + legacy CSV) — PASS. All prior combined/VERDICT outputs are stale until re-aggregated.

### D4 — news event ts_utc was shifted +4h (feed time treated as US Eastern, is actually UTC) (2026-07-06)

- Evidence chain:
  - news_filter.py:145-147 — FairEconomy weekly XML `<time>` parsed naive (`"%m-%d-%Y %I:%M%p"`).
  - news_filter.py:155-156 (pre-fix) — stamped `tzinfo=America/New_York` then `.astimezone(UTC)`, i.e. ADDED the ET offset (+4h EDT / +5h EST). Fallback :161 shared the same wrong assumption.
  - The feed `<time>` is already UTC — proven against 3 independent release anchors in state/ff_calendar_cache/ff_calendar_2026-06-29.xml: ISM `2:00pm` = 14:00 UTC = 10:00 ET ✓; Spanish Flash CPI `7:00am` = 07:00 UTC = 09:00 CEST ✓; Japan data `11:30pm` = 23:30 UTC = 08:30 JST ✓.
- Impact: EVERY event's `ts_utc` was 4h late (summer). Two blast zones: (1) live Phase 2 email — a real ISM Services PMI at 19:30 IST (7:30pm) rendered as 23:30 IST (11:30pm); (2) `is_news_blackout()` + the backtest news filter gated the ±window 4h late, so `news_blocked` marked the wrong rows. news_blocked is INFORMATIONAL (never gates the headline, per D2), so P&L is unaffected — but any news-attribution slice pre-fix is on wrong rows.
- Class: naive-parse-then-wrong-tz. Correct fix: stamp `local.replace(tzinfo=UTC)` directly — no shift.
- **FIXED 2026-07-06:** news_filter.py:152-158 stamps UTC directly (Eastern conversion + buggy fallback deleted). Guard: backtest/test_news_filter.py `test_ff_xml_time_is_utc` — asserts ISM `2:00pm` → 14:00 UTC → 19:30 IST and Spanish CPI `7:00am` → 07:00 UTC — PASS. Any pre-fix news_blocked-keyed slice is on wrong rows until re-run.

### D5 — Phase-1 sweep/FVG/context timestamps read integer row numbers (reset_index poisoning) (2026-07-06)

- Evidence chain:
  - smc_radar.py:453 — Phase-1 H1 frame built with `tailed.reset_index()` (NO drop=True): moves the UTC DatetimeIndex into a `Datetime` COLUMN and leaves a plain integer RangeIndex.
  - smc_detector.py (pre-fix) read `df.index[i]` expecting a timestamp in THREE spots — sweep stamp (:1338), swept-swing stamp (:1344), FVG `_idx_to_iso` (:1483) — each returned the ROW NUMBER (e.g. 122), poisoning the ISO fields.
  - Same integer-index trap silently broke TWO context-tag helpers: `_prior_trading_day_hl` (:1033) and `_session_hl_until` (:1079) iterated `df.index` → every `.date()` threw → caught → returned (None, None), so prior-day/session tags NEVER fired for Phase 1.
- Impact: LIVE Phase-2 trade email only. Sweep/FVG overlays were stamped `"122"/"105"/"129"` (row numbers); Phase 2's `locate_ob_candle_idx` fails `datetime.fromisoformat("122")` → overlays dropped or landed on the wrong candle (the NZDUSD email the owner flagged). Prior-day/session context badges silently absent. NO effect on P&L, scoring, or the backtest (backtest builds its own DatetimeIndex-real df).
- Class: reset_index-drops-datetime-into-column; index read where a timestamp was assumed.
- **FIXED 2026-07-06:** two shared resolvers in smc_detector.py — `_iso_for_idx` / `_ts_for_idx` (read `Datetime`/`Date` column first, index fallback) — wired into all three stamps; `_row_timestamps` used by both context-tag loops. Mirrors smc_radar `_ts_for_idx`. Guard: tests/test_sweep_timestamp_resolution.py (drives observe_phase1_sweep on a reset-index df; asserts no ISO field is a bare row number — proven to go RED when the bug is reintroduced). Full suite 47 passed.

## Insights (backtest/insights.py + reporting sections)

Two consumers, two populations — this matters:
- **Per-run (Excel tabs / email)**: population = `trades_all` (IST+weekend hard-dropped at h1_only_reporting.py:3237) filtered to real fills. Matches headline. CLEAN.
- **Cross-run (aggregate_runs.py)**: FIXED by T2 — `_eligible_mask` (:48) filters the population before insights (`eligible = primary[_eligible_mask(primary)]` :435, `filled = ins._filled(eligible)` + hard assert :439). Pre-fix it loaded raw trades.csv with NO eligibility filter and timeout/window_end force-closed rows fed every aggregate insight (D3).

| insight | src | population | status | note |
|---|---|---|---|---|
| win_rate_pct | insights.py:43-64 | resolved only (BE excluded num+den), None when none resolved | verified | matches settled convention, line-read |
| bootstrap_ci / sharpe / max_drawdown_r / longest_losing_streak | insights.py:71-110 | caller's | verified | standard formulas, seeded rng(42); CI None under n=5 |
| capture_pct | insights.py:113-121 | wins only | verified | avg booked / avg MFE — peak-adjacent by design, guarded by is_peak_metric law |
| compute_overall | insights.py:128-156 | _filled(eligible) | **fixed** | T2 landed: eligibility mask + hard assert; guard tests/test_aggregate_eligibility.py PASS |
| pair_session_matrix | insights.py:163-194 | _filled(eligible) | **fixed** | T2; per-run tab was already clean; uses UTC-fixed `session` (parked caveat) |
| instrument_verdicts | insights.py:216-255 | _filled(eligible) | **fixed** | T2. PREMIUM_PAIRS still lists NAS100 (dropped pair — harmless, stale) |
| confluence_attribution | insights.py:286-323 | _filled(caller) | voided | fvg/freshness legs read corrupted-era columns (Fix 3); sweep leg out-of-scope; also D3 in aggregate |
| score_validation | insights.py:330-378 | _filled(caller) | voided | score embeds fvg/freshness pts (corrupted) + D1 structure taint; also D3 |
| setup_badge_validation | insights.py:387-454 | _filled(caller) | voided | badge from frozen touches/fvg/sweep (smc_detector.py:2459-2466); also D3 |
| group_comparison | insights.py:456-476 | _filled(eligible) | **fixed** | T2 |
| entry_zone_comparison | insights.py:483-509 | eligible + never_filled (fill-rate denominator) | **fixed** | T2; single proximal bucket now (degenerate, harmless) |
| ob_freshness_comparison | insights.py:527-559 | _filled, keyed on ob_touches | **fixed** | 2026-07-07: re-keyed from the DEAD alert_seq (pinned to 1 by the first-fire dedup, run_backtest.py:161-169) onto ob_touches — the alert-time proximal touch count (touches_at_alert, real 0/1/2 variance). Buckets are now 0/1/2 prior touches and populate for real. No re-run needed: ob_touches already in trades.csv. Guard: tests/test_ob_freshness_column.py (fails if it ever re-keys on a constant column) |
| regime_verification | insights.py:559-598 | _filled(eligible) per run_id | **fixed** | T2 |
| generate_verdict | insights.py:605-672 | upstream dicts | **fixed** (population) | T2; its score-issue line inherits voided score_v until baseline re-derive |
| verify_capturable / is_peak_metric | insights.py:708-716 | gate | verified | 2026-07-02 truth-chain audit (peak-vs-fill gate); law block at :675-705 |
| G1-G10 gates | backtest/scanlog/gates.py (evaluate) + g10_violations :105 | headline | verified | 2026-07-02 truth-chain audit; G10 rule set extracted to pure g10_violations() predicate 2026-07-03 (gate + test bind to one threshold; be_eps FP fix, rule b at +1.001R) |
| headline exclusions (_headline_exclusion) | h1_only_reporting.py:142-176 | audit-only rows out of P&L | verified | 2026-07-02 truth-chain audit + G1 blocked-rows fix |
| killzone alignment table | h1_only_reporting.py:546 (_killzone_alignment_table) | prox_trades (verified population) | verified | aggregates verified killzone_alignment column over headline population |
| driver buckets / two-way | h1_only_reporting.py:815 (_driver_buckets), :870 (_driver_two_way) | prox_trades | voided (partial) | fvg/freshness/badge/bos_verdict dims read corrupted columns; clean dims (session/killzone/pd) fine — re-derive whole section post-baseline |
| counterfactual exits | h1_only_reporting.py:628 (_counterfactual_dataframe) | prox_trades | verified | population verified; exit replay = truth-chain audit scope |
| Act 1-6 sections | h1_only_reporting.py (report body) | prox_trades (verified) | verified (population) | aggregation of audited columns over headline population; sections quoting score/badge/fvg slices are void until baseline rerun |
| recipe ranking | h1_only_reporting.py (recipe table) | exit-lab sink (block-stamped, run_backtest.py) | verified (population) | sink rows carry ist/weekend flags so the table drops what the headline drops (826-vs-668 RCA fix) |
| zone register | h1_only_reporting.py:1126 | trades_all | verified (population) | |
| Excel tabs (confluences/badges/pair-session/break-ladder) | h1_only_reporting.py:1403 (reference tabs), :1468 (break ladder) | filled(trades_all) = headline population (:3458 write) | voided (partial) | population CLEAN; confluence/badge/break-ladder tabs read corrupted-era columns → void until baseline rerun; pair-session tab verified |

## Rules

- No new trades.csv column or emitted insight ships without: (1) a row here, (2) a guard (test/gate) that kills its bug CLASS. (CLAUDE.md rule: ADDED 2026-07-02, Logging section.)
- Insights on pending-fix columns are automatically VOID until the first post-fix baseline run.
- Sweep detector columns/insights: owned by the sweep-rebuild chat. Not audited here.
