# TRUTH FIXES SPEC — 2026-07-03 data-truth audit

> **STATUS 2026-07-03 (final): ALL THREE FIXES SHIPPED.** T1 (alert-time
> ob_view + payload verdict), T2 (aggregate eligibility mask + hard assert),
> T3 (truth-ledger gate test) are in the working tree with guards green:
> tests/test_ob_alert_freeze.py (incl. T1 cases + source tripwire),
> tests/test_aggregate_eligibility.py, tests/test_truth_ledger.py. D2 comment
> fixed. DETECTION_FIXES_SPEC.md confirmed landed. **Executor: nothing left to
> implement — this document is now the audit record.** Remaining process step:
> the full baseline run + re-derivation of the VOID insights.

Executor: implement EXACTLY what is written here. No scope creep. Every claim was
verified against live code on 2026-07-03; if the code has moved, re-verify the cited
lines before editing — code is truth, this document is not.

## Relationship to DETECTION_FIXES_SPEC.md (READ FIRST)

- DETECTION_FIXES_SPEC.md (Fixes 1-3) is being executed in a SEPARATE session.
  **This spec lands strictly AFTER it.** Fix T1 below touches the same fire block
  as its Fix 3d and the same row-source lines — rebase on the landed code, re-verify
  every cited line, and reuse its `*_at_alert` machinery instead of duplicating it.
- The 3e classification comment + regression test that spec adds above `_build_row`
  are the home for this spec's guards. Extend them; do not create a parallel system.
- Coverage map: TRUTH_LEDGER.md at repo root. Every column/insight this spec fixes
  must be flipped to **fixed** there in the same commit ("one line per truth").

## Ground rules

- Design decisions are LOCKED (trader-approved 2026-07-03). Do not re-litigate.
- Do NOT touch: sweep detector (`observe_phase1_sweep`, sweep columns), fill/exit
  simulation, excursion metrics, G1-G10 logic, `_headline_exclusion` semantics.
- No behaviour change to which alerts fire, fill, or their P&L. T1-T3 are
  measurement/logging/aggregation truth only.
- After all fixes: run the dedupe + touches + truth tests, structure-golden gate,
  then a SHORT window run (~3 months) — G1-G10 green, no new scanlog FAIL
  conditions. Commit per the OneDrive policy (local-only on dev; CI pushes).

## Verified facts this spec relies on (all re-checkable by line)

- `run_backtest.py:148-153` drains the ENTIRE replay generator into `pair_alerts`
  before simulating anything; rows are built afterwards (:158-183).
- The alert yield passes the OB dict BY REFERENCE (`"ob": ob`,
  backtest/replay_engine.py:532).
- `ob["bos_verdict"]` is (re-)stamped on EVERY fire of the same zone
  (backtest/replay_engine.py:500).
- Only the FIRST fire per (ob_timestamp, direction) is traded
  (run_backtest.py:157-165, `seen_obs`).
- The row sources `ob.get("bos_verdict", "holding")`
  (backtest/h1_only_simulator.py:1227); the scorecard reads
  `ob.get('bos_verdict', 'holding')` at smc_detector.py:2090; scoring happens at
  simulate time via `_score_h1_only` (h1_only_simulator.py:349-385, called at :1295).
- `classify_setup` reads `touches` / `fvg` / `sweep_observed` off the OB dict
  (smc_detector.py:2459-2466); the row calls it at h1_only_simulator.py:1137.
- trades.csv ships `eligible_for_headline` / `headline_exclusion`, stamped from THE
  single eligibility rule (h1_only_reporting.py:1215-1223, rule at :142-173).
- `aggregate_runs.py` loads trades.csv raw (:44-73) and filters ONLY never_filled
  via `insights._filled` (:404; insights.py:32-36) — timeout/window_end force-closed
  rows feed every cross-run metric. Per the settled policy those rows are audit-only.
- h1_only_reporting.py:1248-1250 comment claims news-blocked rows are excluded from
  aggregates; code truth is news NEVER gates (:162-173, :3330-3332).

---

## FIX T1 — Alert-time OB view: bos_verdict (and badge inputs) must be the
## traded alert's values, not the last fire's

### Why (one line)
Rows are built after the whole walk; the OB dict is shared and re-stamped on every
re-fire, so a multi-fire zone logs the LAST fire's `bos_verdict` — and the scorecard
and setup badge read the same drifted dict. Same bug class Fix 3d closes for
touches/fvg; this closes the remaining holes.

### Changes

1. **backtest/replay_engine.py — freeze the verdict into the yield payload.**
   At the fire block, immediately after the existing stamp at line 500
   (`ob["bos_verdict"] = ...`), add the scalar to the yield dict (next to
   `"h1_trend"` / `"trend_alignment"` around lines 536-537):
   ```python
   "bos_verdict": ob["bos_verdict"],
   ```
   (If Fix 3d landed `touches_at_alert` / `fvg_at_alert` stamps here, keep them —
   this is an addition, not a replacement.)

2. **backtest/h1_only_simulator.py — ONE alert-time view, used everywhere.**
   In `simulate_h1_only_dual` (line ~1277), BEFORE `_score_h1_only` is called
   (line 1295), build the view and use it for the whole simulate/row path:
   ```python
   # Alert-time view of the OB. The replay mutates the shared OB dict after
   # this alert fired (re-fires re-stamp bos_verdict; per-bar loops update
   # touches/status/fvg). Everything row-facing must read the ALERT-TIME
   # values: scalars frozen into the yield payload / *_at_alert stamps.
   # One concept, one implementation — never patch individual fields inline.
   ob = alert["ob"]
   ob_view = dict(ob)
   if alert.get("bos_verdict") is not None:
       ob_view["bos_verdict"] = alert["bos_verdict"]
   if ob.get("touches_at_alert") is not None:      # Fix 3d stamp
       ob_view["touches"] = ob["touches_at_alert"]
       ob_view["status"] = None  # derived label; not row-sourced — do not fake it
   if ob.get("fvg_at_alert") is not None:          # Fix 3d stamp
       ob_view["fvg"] = ob["fvg_at_alert"]
   ```
   Then pass `ob_view` (not `ob`) into:
   - `_score_h1_only` (so smc_detector.py:2090 and the freshness/fvg branches of
     `run_scorecard` see alert-time state), and
   - the alert dict consumed by `_build_row` / `classify_setup`
     (h1_only_simulator.py:1137), and
   - the row source for `"bos_verdict"` (h1_only_simulator.py:1227).
   Implementation freedom: either swap `alert["ob"]` for `ob_view` in a shallow
   copy of `alert`, or thread `ob_view` explicitly — but there must be exactly ONE
   view construction, at the top of `simulate_h1_only_dual`.
   Do NOT modify `smc_detector.run_scorecard` or `classify_setup` signatures —
   they are live-shared; the backtest feeds them a corrected dict instead.

3. **Doc note in the row build:** update the `bos_verdict` comment
   (h1_only_simulator.py:1222-1227) to say the value is the ALERT-TIME verdict
   carried in the yield payload.

### Explicitly accepted effects
- `bos_verdict`, `structure_pts`, `score`, `setup_badge` can change ONLY on
  multi-fire zones (first-fire zones already read the correct value). No fill/P&L
  change (score does not gate — trader decision 2026-06-30).

### Regression guard (kills the class — extend Fix 3e's test, same file)
- Build an alert with payload `bos_verdict="holding"`, `touches_at_alert=1`, an
  alert-time fvg snapshot; then mutate `ob["bos_verdict"]="fading"`,
  `ob["touches"]=3`, `ob["fvg"]` BEFORE building the row; assert the row logs
  `bos_verdict == "holding"`, `ob_touches == 1`, alert-time fvg — AND that the
  scorecard call inside the simulate path received the alert-time verdict (assert
  via the resulting `structure_pts` on a case where holding=3 vs fading=1).
- Rule line to add to the 3e classification block: **any field the scorecard,
  badge classifier, or row reads off the OB MUST come through the alert-time
  view; reading `alert["ob"]` directly in the row path is a defect.**

---

## FIX T2 — Cross-run aggregator must respect headline eligibility

### Why (one line)
`aggregate_runs.py` feeds timeout/window_end force-closed rows (audit-only by the
settled unresolved-trade policy) into every cross-run insight, ignoring the
`eligible_for_headline` column trades.csv ships precisely for this.

### Changes (all in backtest/aggregate_runs.py)

1. **Eligibility mask helper** (near `_load_trades`):
   ```python
   _EXCLUDE_REASONS = {"never_filled", "timeout", "window_end"}

   def _eligible_mask(df):
       """Headline-eligible rows, per the ONE rule exported by the reporting
       layer (eligible_for_headline). Legacy CSVs that predate the column are
       reconstructed from the same inputs the rule uses."""
       if "eligible_for_headline" in df.columns:
           s = df["eligible_for_headline"]
           return s.astype(str).str.strip().str.lower().eq("true") | s.eq(True)
       m = ~df.get("exit_reason", pd.Series(index=df.index)).isin(_EXCLUDE_REASONS)
       for col in ("ist_blocked", "weekend_blocked"):
           if col in df.columns:
               m &= ~df[col].astype(str).str.strip().str.lower().eq("true") & ~df[col].eq(True)
       return m
   ```
   (CSV round-trips can turn bools into `"True"`/`"False"` strings — normalise, as
   written above.)

2. **Apply it** after the `primary` slice (line ~399-404):
   ```python
   eligible = primary[_eligible_mask(primary)].copy()
   filled = ins._filled(eligible)   # unchanged helper; belt over braces
   ```
   Every insight currently fed `filled` keeps `filled`. `entry_zone_comparison`
   (line 416) needs never_filled rows for its fill-rate denominator — feed it
   `primary[_eligible_mask(primary) | primary["exit_reason"].eq("never_filled")]`.

3. **Hard guard (fail loud, not silently wrong):** immediately after `filled` is
   built:
   ```python
   assert not filled["exit_reason"].isin(("timeout", "window_end")).any(), \
       "aggregate population contains force-closed audit rows — eligibility filter broken"
   ```

4. `all_trades.csv` (line ~394) stays FULL — it is the audit artifact. Only the
   metric population is filtered.

### Regression guard (kills the class)
- New test (tests/ or backtest/test_*, same home as test_pnl_reconciliation.py):
  build a tiny in-memory df with one tp1 row, one timeout row (nonzero r_realised),
  one never_filled row, one row with eligible_for_headline="False"; run it through
  `_eligible_mask` + `ins._filled` and assert: n == 1, the timeout row's R is NOT
  in the expectancy input, and the string-bool normalisation held.

---

## FIX T3 — Truth-ledger structural gate (new metric cannot ship unregistered)

### Why (one line)
The CLAUDE.md rule ("no new metric without a TRUTH_LEDGER.md row + guard") is
prose; this makes it mechanical, per the 3e pattern — silent reintroduction
impossible.

### Changes

1. **New test `tests/test_truth_ledger.py`** (or the existing test home):
   ```python
   import re
   from pathlib import Path

   def test_every_csv_column_has_a_ledger_row():
       root = Path(__file__).resolve().parents[1]
       ledger = (root / "TRUTH_LEDGER.md").read_text(encoding="utf-8")
       src = (root / "backtest" / "h1_only_reporting.py").read_text(encoding="utf-8")
       m = re.search(r"front_cols = \[(.*?)\]", src, re.S)
       assert m, "front_cols list not found — writer moved; update this test"
       cols = re.findall(r'"([a-z0-9_]+)"', m.group(1))
       missing = [c for c in cols if c not in ledger]
       assert not missing, f"trades.csv columns missing a TRUTH_LEDGER.md row: {missing}"
   ```
   This rides `front_cols` (h1_only_reporting.py:1224-1257) — the writer's own
   canonical column order. A new column added there without a ledger row turns CI
   red. (Columns outside front_cols land via the `rest` catch-all at :1260; the
   3e classification comment governs those — acceptable residual risk, noted.)

2. **Comment fix (defect D2):** correct h1_only_reporting.py:1248-1250 — news
   columns are INFORMATIONAL (news never gates; the one rule is
   `_headline_exclusion` at :142-173). Point the comment at the rule instead of
   restating it.

### Regression guard
- The test IS the guard. Run it in the same CI lane as the structure-golden gate.

---

## VOID / RE-DERIVE LIST (process, not code — do with the post-fix baseline run)

After DETECTION_FIXES_SPEC.md + this spec land, the first full run is the new
baseline. Until re-derived on it, these are VOID (already marked in
TRUTH_LEDGER.md): confluence_attribution (fvg/freshness legs), score_validation,
setup_badge_validation, driver-bucket slices keyed on fvg/freshness/badge/
bos_verdict, Excel confluence/badge/break-ladder tabs. Flag the run manifest:
"first run with alert-time OB truth (T1) + eligible aggregates (T2)".

## NO-OP / PARKED — do NOT implement

- **`session` column is UTC-fixed** (h1_only_simulator.py:82-90) while killzone
  columns are DST-aware — a known asymmetry, ±1h on session edges half the year.
  Trader call: PARKED (killzone columns are the scored/decision fields; session is
  a coarse bucket). Documented in the ledger; do not "fix".
- **`fill_session` falls back to alert hour for never_filled rows**
  (h1_only_simulator.py:118-120) — those rows are audit-only; label mixing is
  documented, harmless. Leave.
- **`ob_freshness_comparison` is structurally inert** (all traded rows are
  first-fire ⇒ buckets 2/3 empty; insights.py:527-552 + run_backtest.py:157-165).
  Correct as built; do not delete, do not "fix" — it self-activates if multi-alert
  trading ever returns.
- **`instrument_verdicts.PREMIUM_PAIRS` still lists NAS100** (insights.py:202) —
  stale but harmless (NAS100 produces no rows). Optional one-word cleanup ONLY if
  already editing that file.
- **Sweep columns/insights** — separate rebuild workstream. Do not touch.

## Execution order & acceptance

1. Confirm DETECTION_FIXES_SPEC.md is fully landed (grep `LIVE_DETECTION_BARS`,
   `DEDUPE_PROXIMAL_ATR_MULT`, `touches_at_alert` in CODE, not just the spec).
   If not landed: STOP, report, do not start.
2. T1 (alert-time view) → extended 3e regression test green.
3. T2 (aggregate eligibility) → new population test green + hard assert in place.
4. T3 (ledger gate test + D2 comment fix) → test green against current ledger.
5. Structure-golden gate green (no re-baseline expected — no detection change).
6. Short window run (~3 months): G1-G10 green; diff `bos_verdict` distribution vs
   a pre-T1 run — differences ONLY on multi-fire zones (spot-check 2-3 by
   ob_timestamp in the scanlog).
7. Flip every touched row in TRUTH_LEDGER.md to **fixed** (with guard name) in the
   same commit. Update the run manifest wording per the VOID list above.
