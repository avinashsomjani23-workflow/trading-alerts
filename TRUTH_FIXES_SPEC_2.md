# TRUTH FIXES SPEC 2 — 2026-07-04 verification audit (T4, T5)

> **STATUS: APPROVED FOR IMPLEMENTATION, NOT STARTED.** Found and proven by the
> 2026-07-04 baseline-verification session (determinism / prefix / start-shift
> runs against the 2008 / 2009 / 2010-2025 baselines). Evidence runs live in
> `backtest/results/h1only_20080102_20080415` (A/B) and
> `backtest/results/h1only_20080215_20080430` (C).

Executor: implement EXACTLY what is written here. No scope creep. Every claim was
verified against live code on 2026-07-04 at the current HEAD; if the code has moved,
re-verify the cited lines before editing — code is truth, this document is not.

## Relationship to earlier specs (READ FIRST)

- TRUTH_FIXES_SPEC.md (T1-T3, SHIPPED) is the parent. **T4 is T1's bug class on two
  fields T1 left behind.** Reuse T1's machinery: the payload-scalar pattern in the
  replay yield and the single `ob_view` in the simulator. Extend
  `tests/test_ob_alert_freeze.py`; do not create a parallel guard system.
- DETECTION_FIXES_SPEC.md Fix 3d created the `*_at_alert` dict stamps that T4
  replaces as row source. Keep the dict stamps (live-parity mitigation logic and
  legacy consumers may read them); the ROW source moves to the payload.
- DECISION_GUARDRAILS.md applies. Relevant: B1 (clean tree before baseline rerun),
  B2 (plumbing on a short run first), B5 (ledger row + structural guard per
  column), B6 (feature freeze NOT yet triggered — edge_engine has only
  stage0_gate.json as of 2026-07-04; these fixes are legal and must land BEFORE
  Stage 1 runs).
- TRUTH_LEDGER.md: every column whose source changes must be updated in the same
  commit (one line per truth).

## Ground rules

- T4 changes NOTHING about which alerts fire, which trades fill, or any P&L number.
  Proven: pre/post trade streams must be byte-identical on
  entry/sl/tp/fill/exit/r_realised (acceptance test below enforces this).
- T5 IS a deliberate behaviour change to TP/score inputs (live-parity correction).
  tp1/tp2, level validity (trade population), and score can move. That is the fix
  working, not a regression — but it must be QUANTIFIED in the acceptance A/B.
- Do NOT touch: detection (150-bar clamp), sweep detector, fill/exit walker,
  eligibility rules, G1-G10, aggregation.
- Do NOT try to emulate live's forming bar. Closed-bar-only is the accepted parity
  convention (backtest alerts ~1 bar later than live; settled). T5's `tail(200)`
  of closed bars vs live's 200-with-forming-bar is an accepted off-by-one of the
  same class.

## Verified facts this spec relies on (re-check each line before editing)

- Rows are built AFTER the whole walk: `run_backtest.py` `_process_pair` drains the
  entire replay generator into `pair_alerts` (:152-157), then simulates (:162+).
  Only the FIRST fire per (ob_timestamp, direction) is traded (`seen_obs`, :161-169).
- The alert yield passes the OB dict BY REFERENCE (`"ob": ob`,
  backtest/replay_engine.py:605).
- Fire block stamps onto that SHARED dict: `ob["bos_verdict"]` (:567),
  `ob["touches_at_alert"]` (:572), `ob["fvg_at_alert"]` (:573). Every re-fire of the
  same zone OVERWRITES all three. T1 already carries `bos_verdict` as a payload
  scalar (:615) so its row value is safe; touches/fvg are NOT in the payload — that
  is bug T4.
- The simulator's single alert-time view: `h1_only_simulator.py:1339-1354` builds
  `ob_view = dict(_ob_live)` and patches `bos_verdict` from the payload but
  `touches`/`fvg` from the DICT stamps (:1349-1352). NOTE THE TRAP: `dict(_ob_live)`
  also copies the (possibly re-stamped) `touches_at_alert` / `fvg_at_alert` KEYS
  into the view, and `_build_row` PREFERS those keys
  (`ob.get("touches_at_alert", ob.get("touches"))` at :1070,
  `ob.get("fvg_at_alert")` at :1071). Fixing only `ob_view["touches"]/["fvg"]` is
  NOT enough — the `*_at_alert` keys inside the view must be overwritten too.
- Row fields fed by these: `ob_touches` (:1299), `fvg_present` (:1288),
  `fvg_state` (:1291), `fvg_mitigation` (:1295), plus `score`/`breakdown` — scoring
  runs at simulate time via `_score_h1_only` (:349-385, called at :1356) and reads
  `ob["fvg"]` (:367) and (inside `run_scorecard`) the view's touches; `alert["ob"]`
  is already the `ob_view` by then (:1354), so fixing the view fixes the score.
- **T4 empirical proof:** GBPUSD zone ob_timestamp 2008-04-08T02:00 fired 3× in the
  full-2008 baseline (Apr 10 seq1 traded, Apr 14, Apr 17) but 2× in a run ending
  Apr 15. Same detection stream, same P&L; the traded row's
  score/fvg_present/ob_touches differed between the two runs (7/True vs 5/False).
  ~40% of zones re-fire (2008: 39.8%, 2009: 41.1% of 2558/1452 zones) → ~40% of
  rows carry last-fire stamps.
- Live Phase 2 fetches **200 H1 bars** and feeds them to BOTH the scorecard and the
  levels engine: `feed_adapter.fetch_h1(symbol, outputsize=200)`
  (feed_adapter.py:66-73, called from Phase2_Alert_Engine.py:360 via
  `fetch_with_retry`, frame used at :2530 `run_scorecard` and :2534
  `compute_phase2_levels`).
- The backtest feeds them UNBOUNDED history instead: `h1_only_simulator.py:364`
  (`_score_h1_only` slice) and :487 (`df_h1_at_alert` for
  `compute_phase2_levels`, :492-495) — all closed bars since fetch_start
  (run start − 35d pad; up to 15 YEARS inside the 2010-2025 run). That is bug T5.
- **T5 empirical proof:** start-shift run C (2008-02-15→04-30) vs the full-2008
  baseline, converged window Mar 3-31: 139/139 trades identical on
  entry/sl/fill/exit/R, but 4 tp1/tp2 mismatches (e.g. USDCAD 2008-03-12 09:00 has
  a tp2 in C, NaN in baseline) — TP selection depends on fetch depth, which neither
  matches live (200) nor is internally stable across runs.

---

## FIX T4 — touches_at_alert / fvg_at_alert must travel in the alert payload

### Why (one line)
Rows are built after the walk; re-fires re-stamp the shared dict, so a traded first
fire logs the LAST fire's touches/fvg (and the fvg/freshness points inside score) —
T1's bug on the two fields T1 didn't move to the payload.

### Change 1 — replay yield (backtest/replay_engine.py, fire block ~:599-623)
Add to the yielded alert dict, next to the existing `"bos_verdict"` payload scalar:

```python
# T4 (TRUTH_FIXES_SPEC_2): same rationale as T1's payload verdict — the dict
# stamps below are overwritten by every re-fire; rows are built post-walk.
"touches_at_alert": int(ob.get("touches") or 0),
"fvg_at_alert": dict(ob.get("fvg") or {}),
```

Fresh `dict(...)` copy is mandatory (the per-bar loop mutates `ob["fvg"]` in place).
Keep the existing dict stamps at :572-573 unchanged.

### Change 2 — simulator view (backtest/h1_only_simulator.py:1349-1352)
Payload first, dict stamp as legacy fallback, and overwrite BOTH key spellings in
the view (see the trap in Verified facts):

```python
_touches = alert.get("touches_at_alert",
                     _ob_live.get("touches_at_alert"))
if _touches is not None:
    ob_view["touches"] = _touches
    ob_view["touches_at_alert"] = _touches
_fvg = alert.get("fvg_at_alert") or _ob_live.get("fvg_at_alert")
if _fvg is not None:
    ob_view["fvg"] = _fvg
    ob_view["fvg_at_alert"] = _fvg
```

Update the T1 comment block (:1339-1344) to name the payload as the one source.
No other read site changes: `_build_row` and `_score_h1_only` already consume the
view / `alert["ob"]`.

### Guards (B5 — kill the class, not the instance)
- Extend `tests/test_ob_alert_freeze.py` (T1's home): scenario — fire a zone, mutate
  `ob["touches"]` / `ob["fvg"]["exists"]` / re-stamp the `*_at_alert` dict keys (as
  a second fire would), THEN simulate the first alert. Assert the row's
  `ob_touches`, `fvg_present`, and the fvg/freshness score points equal the
  FIRST-fire values. Follow the file's existing source-tripwire pattern so a future
  revert of the payload read fails the test.
- TRUTH_LEDGER.md: update source file:line + "stamped at yield (payload)" for
  `ob_touches`, `fvg_present`, `fvg_state`, `fvg_mitigation`, `score`,
  `fvg_pts`, `freshness_pts`.

### Acceptance (run, don't argue)
1. All tests green (`test_ob_alert_freeze.py`, dedupe/touches/truth suites,
   structure-golden gate).
2. **P&L-invariance:** run 2008-01-02→2008-04-15 (10 baseline pairs, `--regime bau`,
   `BACKTEST_SKIP_NEWS=1`) at the fixed code; diff against the pre-fix run of the
   same window (archived: scratchpad `runA_trades.csv`, or re-run at the pre-fix
   commit): entry/sl/tp/fill_ts/exit_ts/exit_reason/r_realised IDENTICAL on every
   row. Only stamp columns (`ob_touches`, `fvg_*`, `score`, breakdown) may differ.
3. **Cutoff-independence (the structural proof):** at the fixed code, run
   2008-01-02→2008-04-15 AND 2008-01-02→2008-06-30. Every column of every shared
   row — INCLUDING the Apr 1-14 tail zone — must be identical (the 2026-07-04
   session measured 29 stamp mismatches there pre-fix; post-fix the count is 0).

---

## FIX T5 — clamp the scorecard + levels input to live's 200-bar window

### Why (one line)
Live P2 computes score and TP levels from 200 bars; the backtest feeds the same
functions unbounded history, so TP selection (and any depth-sensitive score input)
depends on run start date instead of matching live.

### Change 1 — one shared constant
In `smc_detector.py` (imported by BOTH live Phase 2 and the simulator already):

```python
# Live Phase 2 fetches this many H1 bars (feed_adapter.fetch_h1 outputsize) and
# computes run_scorecard + compute_phase2_levels from that frame. The backtest
# MUST feed the same functions the same window — parity breaks silently otherwise.
LIVE_P2_H1_BARS = 200
```

In `Phase2_Alert_Engine.py:360` replace the literal:
`feed_adapter.fetch_h1(symbol, outputsize=smc_detector.LIVE_P2_H1_BARS, retries=retries)`
(verify `smc_detector` is imported there — it is, it's called at :2530/:2534).
Leave `feed_adapter.py`'s own default alone; add a docstring cross-reference.

### Change 2 — one shared slice helper (kills the duplicate too)
`h1_only_simulator.py` computes the closed-bars slice TWICE (:364 and :487) — one
concept, two implementations. Replace both with a module-level helper:

```python
def _closed_bars_at_alert(df_h1, alert_ts):
    """Live-parity input frame: the last LIVE_P2_H1_BARS bars CLOSED before
    alert_ts — exactly what live P2 hands run_scorecard/compute_phase2_levels.
    tail() cannot add future bars, so the lookahead guarantee is unchanged."""
    s = df_h1.loc[df_h1.index < alert_ts]
    if s.empty:
        s = df_h1.loc[:alert_ts]  # degenerate guard, never empty in practice
    return s.tail(smc_detector.LIVE_P2_H1_BARS)
```

Use it at both call sites. The forward fill-walk keeps the FULL df_h1 (it must see
the future to simulate the trade) — do not touch it.

### Guards
- Runtime tripwire (FIX 1 pattern): in the helper, `assert len(s) <= LIVE_P2_H1_BARS`
  after tail (cheap, loud).
- Unit test: synthetic H1 frame where the nearest opposing swing beyond the TP floor
  sits 300 bars back and a different one 50 bars back; assert
  `compute_phase2_levels` via the helper picks the 50-bar swing (200-window) and
  that passing the unclamped frame picks the other (proves the clamp is live).
- TRUTH_LEDGER.md: update `tp1`, `tp2`, `tp1_rr`, `tp2_rr`, `score` input-window
  note.

### Acceptance
1. **Fetch-depth independence (the structural proof):** at the fixed code, re-run
   the start-shift pair — 2008-02-15→2008-04-30 vs 2008-01-02→2008-06-30 — and
   compare window Mar 3-31: ZERO mismatches on ALL columns including tp1/tp2
   (pre-fix: 4). Both runs now see the identical 200-bar tail at every alert.
2. **Impact quantification (report, don't hide):** diff the fixed short run vs the
   same window pre-fix: report % rows with tp1/tp2 changed, % rows
   added/dropped (level-validity flips), % r_realised changed, and the headline
   WR/expectancy delta. Put the numbers in the commit message.

---

## Order of operations (after both fixes reviewed)

1. Land T4 + T5 in one commit each (or one bundle — pre-declared here, D1
   satisfied), tests + ledger in the same commits. Clean tree (B1).
2. Short plumbing run (~3 months, B2): G1-G10 green, no new scanlog FAIL
   conditions, acceptance diffs above.
3. Re-run the three baseline chunks in CI (same commands, same 10 pairs):
   2008-01-02→2008-12-31, 2009-01-01→2009-12-31, 2010-01-01→2025-12-31.
   These become the new baseline; the 2026-07-03 runs' feature columns and TP
   levels are VOID for engine use (their P&L remains historically valid but is
   superseded).
4. Stage 0 on the new 2010-2025 run, then Stage 1. Nothing before this point has
   fed the engine, so B6 is intact.

## Explicitly out of scope

- The ~15-day OB-inventory warmup seam at chunk starts (measured, negligible: two
  seams in 18 years) — inherent to chunked runs, not a bug.
- The 2008 chunk's first 150 skipped bars (data inception; unavoidable).
- Forming-bar / +1h fill-offset parity diffs (settled decisions).
- Sweep detector (A5), detection window (A6), NAS100 (A4).
