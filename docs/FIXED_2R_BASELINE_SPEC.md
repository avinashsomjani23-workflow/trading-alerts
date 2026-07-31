# SPEC — Switch the backtest baseline to FIXED 2R (execution handoff)

**Status:** APPROVED scope, NOT yet executed. Written 2026-07-31.
**Purpose:** Change the committed backtest exit from the liquidity-pool TP1 + break-even@1R
policy to a **fixed 2R** exit, so entries can be studied against a constant, unambiguous
ruler. Every column in the new run must speak ONE language: fixed 2R. Nothing may whisper
"TP1", "liquidity pool", "wick", or "next pool" in any EXIT/OUTCOME column.

> Read this whole doc before editing. Then execute top-to-bottom, pausing after Section A
> (the simulator) for a diff review before touching the report. Verify (Section H) before
> declaring done. This is a critical change — meticulous over fast.

---

## 0. WHY (the decision, so the executor doesn't re-litigate it)

- The current baseline (liquidity-pool TP1 + BE@1R) has been shown unsound in prior work.
  To improve ENTRIES we need the EXIT held constant. A fixed 2R bracket is the cleanest
  possible ruler: hit +2R = +2R, hit stop = −1R, nothing in between depends on an exit
  decision. Every entry is judged on the same yardstick.
- We are NOT choosing 2R because it performs best (it doesn't — on the current run the ATR
  exit and 1.5R score similar/better). We choose it because it is a clean, fixed benchmark.
  The benchmark run WILL be negative expectancy. That is expected — entry work must lift it.
- Liquidity-pool TP logic is being retired from THIS run (not deleted from the codebase — see
  §D). Ranked pools / sweeps / EQ / PD-PW remain as SETUP descriptors (entry features); only
  the liquidity-TP EXIT machinery leaves the run's output.

### Decisions locked by the trader (do not change without asking)
1. Exit = **fixed 2R**, full position, no break-even, no trail. Exit reason `"tp"` (one target,
   so NOT "tp1"/"tp2r" — just `"tp"`, to avoid TP1/TP2 confusion).
2. **Window-MFE replaces MFE.** MFE/MAE must track the best/worst excursion across the FULL
   post-fill window (max 48 bars), independent of where the 2R exit fired. It is a look-ahead
   / OUTCOME column — usable to describe & study, NEVER to drive a live entry filter.
3. **Break-even code: delete from the live walk.** Re-adding it later is a few lines; a
   pointer to the removed logic goes in this doc (§A4) so revival is trivial. Do NOT keep dead
   BE branches in the hot path.
4. **Meaningless twin columns removed:** `entry_raw`, `sl_raw`, `tp1_raw`, `tp2_raw`. Under the
   2026-07-30 raw spread convention each EQUALS its base (`entry`, `sl`, `tp1`, `tp2`); the one
   spread lives on the stop only. They are pure duplicates → drop them.
5. **SL-anatomy keeps BOTH readings, re-anchored to the NEW targets:** swept-then-**2R** and
   swept-then-**1R** (breakeven-plus). Same for bar-counts. Every comment says "2R"/"1R" in
   plain words — NO "TP1", NO "pool". Rationale: shows which stopped-out trades came back to
   just +1R vs ran all the way to the 2R target — the wider-stop question. bars-to-ENTRY is
   NOT added (trader decision).
6. **Pool-TP "archived" = the COLUMNS stop being written to this run's CSV/Excel.** The CODE
   that computes them (in `smc_detector.py`, used by LIVE Phase 2) stays put. We stop
   CONSUMING it in the backtest. Live must remain byte-identical (§D).

---

## 1. TERMINOLOGY the executor must keep straight

- `tp1` (the column/level) = the liquidity-pool zone-edge TP. Being retired from the run.
- `tp1_raw` = duplicate of `tp1` (raw==emitted now). Drop.
- `r_if_exit_tp1 / r_if_exit_tp2` = "what liquidity-TP would have paid" reference columns,
  produced by the SEPARATE `_reference_touch_indices` walk. Pure liquidity language → drop.
- `tp_wick` / `tp_nextpool` = same-pool buffered wick & next-pool runner (triple-mode). Drop.
- The NEW fixed target is computed as `entry ± 2·r_distance` where `r_distance = abs(entry−sl)`
  and `sl` is the already-one-spread-widened stop. Call the internal variable `tp_2r`.

---

## A. SIMULATOR — `backtest/h1_only_simulator.py`  (the heart; review diff after this)

Current commit path read & confirmed at these lines (may shift slightly on edit — re-locate
by content, not line number):

- `_reference_touch_indices` — lines ~441–494. **DELETE the whole function.** Its only job is
  `r_if_exit_tp1 / r_if_exit_tp2` + `bars_to_tp1 / bars_to_tp2`, all being dropped.
- Level setup — lines ~568–620. Keep `entry`, `sl`, `r_distance`. **Remove** the reads/locals
  for `tp1_raw, tp2_raw, entry_raw, sl_raw` and the whole `tp_wick/tp_nextpool/tp1_wick/
  tp1_zone_source/tp2_*/tp_targets/tp2_collapsed_to_tp1` block. Keep `tp1`/`tp2` locals ONLY
  if still needed transiently; goal is they no longer reach the row.
- The walk — lines ~803–1002. This is the surgery:

### A1. New target
Compute once before the loop:
```
tp_2r = entry + 2*r_distance if bias == "LONG" else entry - 2*r_distance
```

### A2. Fixed-2R exit, no BE
Inside the loop, per bar (keeping the EXISTING pessimism rules):
- Stop check first (`cur_sl` is just `sl` now — no BE mutation). SL CAN fire on fill bar.
- TP check: `tp_hit = bar_hi >= tp_2r` (LONG) / `bar_lo <= tp_2r` (SHORT). Suppressed on fill bar.
- SL+TP same bar → **SL wins** (unprovable order; keep pessimism).
- On first resolution: latch `exit_reason ("sl" or "tp")`, `exit_price (cur_sl or tp_2r)`,
  `exit_ts`. **DO NOT `break`.** (See A3.)
- `timeout` / `friday_flat` / `window_end` / `never_filled` logic UNCHANGED.

### A3. Window-MFE (decouple excursion from exit) — CRITICAL
- Today the loop `break`s at exit and MFE stops there. NEW: once the exit is latched, keep
  walking bars for MFE/MAE only, until the window ends (48-bar hold cap / friday / data end).
- MFE/MAE stay a running max/min (a pullback never lowers MFE). KEEP excluding the **fill bar**
  and the **stop bar** from the excursion update (intrabar order unknowable — same pessimism).
- After the exit is latched, there is no more SL/TP resolution to do — only excursion tracking
  and the timeout/friday/window-end window bound. Structure the loop so post-exit bars ONLY
  update mfe/mae and honor the window cap; they do not change the latched exit.
- Result: `mfe_r` can now EXCEED `r_realised` legitimately (e.g. exit +2R, price later ran
  +3.5R → mfe_r = 3.5). This is correct and desired. **Invariant: `mfe_r >= r_realised` always.**

### A4. Break-even removal (pointer for future revival)
Delete: `be_armed`, `be_trigger`, `be_eps`, `be_arm_bar_touched_entry`, the `cur_sl = entry`
arming block (~948–955), and the `be_reached_in_bar` computations (~861/891).
> TO RE-ADD BE LATER: reintroduce `be_trigger = entry ± r_distance`, arm on reach (with the
> `be_eps = r_distance*1e-6` float-boundary tolerance), set `cur_sl = entry` once armed. The
> `exit_engine.walk_multileg` recipe path already supports BE via config `be_trigger_r`, so BE
> as a *comparison* study needs no simulator change — only reviving it as the COMMITTED policy
> would touch the simulator.

### A5. SL-anatomy re-anchor (lines ~1049–1198)
Rename + re-anchor. The block currently measures "after the stop was swept, did price reach
**TP1**". Change every TP reference to the fixed targets and KEEP BOTH readings:
- `sl_swept_then_tp1` → `sl_swept_then_2r` (reached the fixed 2R target after the sweep).
- ADD `sl_swept_then_1r` (reached +1R / breakeven-plus after the sweep).
- `bars_sl_to_tp1_touch` → `bars_sl_to_2r_touch`; ADD `bars_sl_to_1r_touch`.
- `sl_max_adverse_after_sweep_atr`, `sl_recovered_to_entry`, `sl_wick_depth_atr`,
  `sl_bar_was_sweep` — KEEP (already TP-agnostic; recovery-to-entry is fine). Re-read their
  comments and strip any "TP1"/"pool" wording.
- The internal "reached TP" test uses `tp_2r` (for the 2R reading) and `entry ± r_distance`
  (for the 1R reading). No liquidity price is used. Comments say "+2R"/"+1R" explicitly.

### A6. Row build (`_build_row` call ~1268 + the never_filled early return ~961)
Remove every dropped kwarg: `tp1_raw, tp2_raw, entry_raw, sl_raw, tp1_rr, tp2_rr, tp1_wick,
tp1_wick_rr, tp1_zone_source, tp2_wick, tp2_zone_source, tp_wick, tp_wick_rr, tp_nextpool,
tp_nextpool_rr, tp_nextpool_zone_source, tp2_collapsed_to_tp1, tp_targets, r_if_exit_tp1,
r_if_exit_tp2, bars_to_tp1, bars_to_tp2`. Update the SL-anatomy kwargs to the new names and
add the two new 1R ones. Update `_build_row`'s SIGNATURE + its internal dict identically.
- `r_capture_ratio` (~1618) = `r_realised / mfe_r` — KEEP (still valid, now "2R capture of the
  full-window MFE"). Re-read its comment.
- `exit_reason` value `"tp1"` no longer emitted; the walk emits `"tp"`. Grep the file for the
  string `"tp1"` / `"tp2"` used as exit reasons and fix.

### A7. Exit-lab side-channel (~1227–1266) — KEEP but re-verify
`walk_multileg` is called with `tp1` for recipes that target `"tp1"`. Since the committed run
no longer uses `tp1`, decide: EITHER pass a synthetic `tp1 = tp_2r` so the "baseline" recipe
== fixed 2R, OR keep computing `tp1` transiently ONLY to feed the exit-lab comparison recipes
(so BE@1R etc. remain comparable). RECOMMENDED: keep `tp1` computed locally (not logged) and
feed exit-lab, so BE-vs-2R stays a study. Confirm with trader if unsure. If kept, `tp1` is a
LOCAL only — it must NOT appear in any row/column.

---

## B. COLUMN COUNT / CANONICAL

- Current canonical: 203 columns (`h1only_20080102_20161231/trades.csv`).
- After drops (~22 removed) + 2 SL-anatomy additions, the count CHANGES. Compute the exact
  new count from the fresh run's header — do NOT hardcode a guess.
- `backtest/results/CANONICAL.md`: repoint to the NEW run (path + new column count) IN THE
  SAME COMMIT that produces it; archive the old run per its own rules.

---

## C. REPORT / EXCEL — `backtest/h1_only_reporting.py`  (71 hits; the bulk of the work)

- CSV header block (~1291–1349): remove all dropped columns; rename SL-anatomy + add the two
  1R columns; re-read the block's comments (the "entry/tp1/tp2 raw twins" comment ~1295 is now
  false — fix it).
- Excel label map (~1461–1520): remove labels for dropped columns; add labels for
  `sl_swept_then_2r/1r`, `bars_sl_to_2r_touch/1r_touch`; the exit-reason display map
  (~1143, ~1516) `"tp1":"TP1 Hit"` → `"tp":"TP Hit (2R)"`; drop `"tp2"`.
- `_runner_r` / `_attach_runner_r` / `r_if_runner` (~496–520, ~1665–1673): DELETE — pure
  TP1+runner reference. Remove `pnl_usd_tp1` / `pnl_usd_runner` too.
- Counterfactual (`_counterfactual_dataframe` ~627): DELETE the `tp1_rr` and `tp2_rr` filter
  sections (~660–673). Keep score/confluence/killzone/session/dow/pd sections.
- `_DRIVER_CONTINUOUS` (~805): remove `("tp1_rr", "TP1 distance (R)")`.
- Exit-recipe labels + baseline key (~770–785): `_EXIT_BASELINE_KEY` should point at the
  fixed-2R recipe (`C_fullTP_2.0R`) and its label read **"Fixed 2R (baseline)"**. Remove the
  "TP1 + BE@1R (LIVE)" wording. Update every `_EXIT_RECIPE_LABELS` string that says TP1/zone/
  wick/pool to plain names (they still exist as COMPARISON recipes — keep the rows, fix the
  words so the baseline column isn't mislabeled).
- Scoreboards using `r_if_exit_tp1/tp2` (~3240, ~3497, ~3554): DELETE those aggregate blocks
  and their HTML/summary consumers.
- Narrative helpers referencing TP1 (`_same_bar_resolution_html` ~977, the tp1_touched leak
  ~985, the "left money"/vet-review ~330): rewrite to speak 2R. `_flag_vet_review` logic is
  fine (uses mfe vs r) — just check wording.
- The level-sanity check (~1079–1106): it validates `tp1>entry` etc. Replace with a `tp_2r`
  ordering check, or remove (2R is computed, can't be mis-ordered) — RECOMMENDED remove.

---

## D. DETECTOR — `smc_detector.py`  (98 hits) — MINIMIZE / IDEALLY ZERO CHANGES

- The pool-TP computation here also serves LIVE Phase 2. **Goal: change nothing in this file.**
- The backtest simply stops READING `tp1/tp_wick/tp_nextpool/...` from the levels dict into the
  row. `compute_phase2_levels` can keep returning them; the simulator just doesn't log them.
- BEFORE finalizing: confirm live Phase 2 still consumes these (it does — live alerts show the
  pool TP). If any backtest-only helper in the detector exists solely for the dropped columns,
  only THEN consider archiving it. Default expectation: `smc_detector.py` is UNTOUCHED.
- Verify live parity: no live email/alert output changes. (If detector is untouched, this holds
  by construction.)

---

## E. SMALLER CONSUMERS

- `backtest/run_backtest.py` (~610–611): remove the `r_if_exit_tp1/tp2` kwargs passed through.
- `backtest/diagnostics/exit_report_gate.py`: its whole purpose (G6) is proving the headline
  never leaks from `tp1/tp2` columns. Once those columns are gone the leak is structurally
  impossible — simplify/retire the `r_if_exit_tp2` sum check; keep the "headline from
  r_realised only" assertion. Update comments that name tp1/tp2.
- `backtest/scanlog/gates.py` (~223): `SL_ANATOMY_COLS = ["sl_bar_was_sweep",
  "sl_swept_then_tp1"]` → update to the new names. Also ~217 column list references `tp1`.
- `backtest/diagnostics/edge_engine.py` / `edge_lab.py`: update any `r_if_exit_*` / `tp1_rr` /
  `bars_to_tp1` reads. `walk_multileg` calls that pass `tp1` for `"tp1"`-spec recipes: if those
  recipes are kept as comparisons, feed them `tp_2r` or a locally-computed `tp1`; if not, prune.
- `backtest/insights.py`, `render_report.py`, `aggregate_runs.py` (1 hit each): fix the single
  reference in each.

---

## F. COLUMN-BUCKET GENERATOR (the CI safety net) — `backtest/gen_column_buckets.py`

- Three lists reference the dropped/renamed columns (~65–72 outcome, ~112–115 setup, ~159 RR).
  Remove dropped names; add `sl_swept_then_2r`, `sl_swept_then_1r`, `bars_sl_to_2r_touch`,
  `bars_sl_to_1r_touch` in the correct bucket (OUTCOME — they are look-ahead, computed from
  post-fill bars).
- The generator RAISES if a live CSV column is undocumented → run it after the fresh run; a
  green run PROVES no column was missed. This is the guarantee, not a nicety.
- `COLUMN_BUCKETS.md` is regenerated by `python -m backtest.gen_column_buckets`.

---

## G. DOCS & TESTS

- `TRUTH_LEDGER.md`: remove rows for dropped columns; add/rename rows for the 2R exit, the two
  SL-anatomy 1R columns, and window-MFE (its meaning CHANGED: now full-window, not in-trade).
  Every row: source `file:line`, when stamped, population, timing class.
- Tests referencing dropped columns: `tests/test_truth_ledger.py`, `tests/test_spread_
  placement.py`, `backtest/test_h1_only.py`, `backtest/test_pnl_reconciliation.py`,
  `backtest/test_news_filter.py`, `tests/test_edge_lab_scaffold.py`, `backtest/scanlog/
  test_scanlog_self.py`, `backtest/diagnostics/exit_report_gate.py` tests. Update expectations
  to the new column set / new exit reason `"tp"`.
- `CLAUDE.md` / `ANALYSIS_POINTERS.md`: note the baseline is now fixed 2R (one line).

---

## H. VERIFY (MANDATORY before declaring done — trader's explicit requirement)

Run on a SMALL cached window (a few months / one pair) — NOT the full 18yr.

1. **Hand-trace 3–5 real trades** pulled from the cache:
   - a clean 2R winner → `r_realised == +2.0`, `exit_reason == "tp"`.
   - a clean −1R loser → `r_realised == -1.0`, `exit_reason == "sl"`.
   - one that pops +1R then reverses to SL → `r_realised == -1.0` AND `mfe_r >= 1.0`
     (window-MFE recorded the +1R peak even though it lost). This proves the decouple.
   Manually compute each from the OHLC bars and confirm the sim matches.
2. **Invariants over the whole small run:**
   - every filled row: `r_realised ∈ {+2.0, −1.0}` OR reason ∈ {timeout, friday_flat,
     window_end} (partial R allowed only for those).
   - `mfe_r >= r_realised` for EVERY row (no exceptions).
   - `pnl_usd == r_realised * risk_usd`.
3. **Language grep on the FINAL CSV header:** grep for
   `tp1|tp2|liq|pool|wick|nextpool|zone_source|be_|breakeven|runner`. Must return NOTHING in
   any EXIT/OUTCOME column. (Setup columns like `pd_zone`, sweep_*, eq_* are fine — they are
   not exit language.) If any exit/outcome column matches → STOP, not done.
4. **Column-bucket generator green** (§F) — proves no column undocumented.
5. **Live parity:** confirm `smc_detector.py` untouched (§D) → live alerts byte-identical.

---

## I. EXECUTION ORDER

1. Simulator (§A). **PAUSE — show trader the diff.**
2. Report/Excel (§C).
3. Small consumers (§E) + detector confirm (§D).
4. Docs/tests (§G) + column-bucket generator (§F).
5. Fresh small run → VERIFY (§H).
6. Repoint CANONICAL (§B). Commit local-only (OneDrive policy); push in CI on "ship it".

## J. OPEN QUESTIONS for the executor to confirm with the trader if hit
- §A7 / §E: keep the liquidity-TP comparison recipes (BE@1R, wick, etc.) in exit-lab by
  feeding them a locally-computed `tp1`? (Trader wants BE@1R comparable → likely YES, but the
  `tp1` value must never reach a logged column.)
- Whether `sl` (the committed stop) also has a now-redundant twin beyond `sl_raw`/`sl_initial`
  worth pruning — check `sl_initial` vs `sl` before removing anything named `sl_*`.
