# DETECTION FIXES SPEC — 2026-07-02 audit

Executor: implement EXACTLY what is written here. No scope creep. Every claim below was
verified against live code on 2026-07-02; if the code has moved, re-verify the cited
lines before editing — code is truth, this document is not.

## Ground rules for the executor

- Design decisions are LOCKED (trader-approved). Do not re-litigate thresholds or semantics.
- Do NOT touch: `observe_phase1_sweep` (sweep rebuild is a separate workstream),
  `compute_structure` internals, fill/exit simulation, excursion metrics, G1–G10 logic.
- After all fixes: run the structure-golden gate (re-baseline via `gen_fixtures` if red —
  policy in memory "Structure Golden Gate"), run a SHORT backtest window (~3 months) and
  check G1–G10 pass, then run the full re-baseline run. Commit per the OneDrive policy
  (backtests commit local-only on dev; CI pushes).
- Expected and accepted: headline numbers WILL differ from all previous runs (Fix 1
  corrects a live-parity defect). Old runs are not comparable. Do not "fix" the diff.

## Verified facts this spec relies on

- Live Phase 1 detection window = the LAST 150 CLOSED H1 BARS, not 15 days. `fetch_data`
  ignores its `period` argument and tails the adapter frame to 150 rows
  (smc_radar.py:421-446, tail at :440). The `"15d"` string at smc_radar.py:3710 is dead.
- The backtest replay feeds detectors an UNBOUNDED slice: `_slice_closed_before` returns
  every bar before the walk timestamp with no tail clamp (backtest/replay_engine.py:198).
- ATR everywhere is the plain mean of the last 14 true ranges
  (smc_detector.py:84-94; dealing_range._compute_atr delegates to it). Therefore
  clamping the slice does NOT change ATR at a given bar, and no displacement gate
  (choch_disp, bos_disp, prox_cap) moves because of Fix 1.
- The dedupe threshold is a hardcoded default: `thresh = cand.get('_dedupe_thresh', 0.00030)`
  (smc_radar.py:1144). `_dedupe_thresh` is NEVER written anywhere in the repo (grep
  verified: only the read at :1144 and a dead `pop` at :1169).
- Replay freezes OB dicts at first sight: merge skips any OB whose `ob_timestamp` already
  exists (backtest/replay_engine.py:355-358). Live refreshes zone state every scan.
- `_is_ob_mitigated_replay` discards the `touches` return value of
  `is_ob_mitigated_phase1` (backtest/replay_engine.py:96-124).
- trades.csv sources `"ob_touches": ob.get("touches")` (backtest/h1_only_simulator.py:1255).

---

## FIX 1 — Clamp the replay detection slice to live's 150-bar window

### Why (one line)
Backtest detects on years of history; live detects on the last 150 closed H1 bars.
Structure state is path-dependent, so backtest and live can disagree at the same
timestamp. Clamp = parity + a large perf win (slices drop from ~100k bars to 150).

### Changes

1. **smc_radar.py — single source of truth for the window.**
   - Add near the other module constants (around line 78):
     ```python
     # Live Phase 1 detection window: the H1 frame every detector sees is the
     # last N CLOSED bars (fetch_data tails the adapter frame to this). The
     # backtest replay clamps its slices to the SAME window for parity.
     LIVE_DETECTION_BARS = 150
     ```
   - Replace the literal in `fetch_data`: `df.tail(150)` → `df.tail(LIVE_DETECTION_BARS)`
     (smc_radar.py:440). No other live behaviour change.

2. **backtest/replay_engine.py — clamp the slice.**
   - In `replay_pair`, immediately after `h1_slice = _slice_closed_before(df_h1, h1_ts)`
     (line 198) and BEFORE the lookahead assert:
     ```python
     h1_slice = h1_slice.tail(smc_radar.LIVE_DETECTION_BARS)
     ```
     Keep `_assert_no_lookahead(h1_slice, h1_ts, "H1")` right after (tail cannot add
     future bars; the assert stays as the guard).
   - Replace the warmup rule. Delete `MIN_WARMUP = 50` (line ~163) and change the check
     (line ~200) to:
     ```python
     if len(h1_slice) < smc_radar.LIVE_DETECTION_BARS:
     ```
     Keep the existing `WARMUP_SKIP` scanlog outcome/condition unchanged. Rationale:
     live ALWAYS runs with a full 150-bar frame; a backtest bar with fewer bars of
     history cannot reproduce live and must be skipped, not approximated. (This also
     retires the parked "MIN_WARMUP=50 < H4's 80-bar floor" issue: 150 ≥ 80 always.)

3. **backtest/replay_engine.py — mitigation-anchor fallback (MANDATORY companion).**
   Problem the clamp creates: a stored OB can be up to 15 days old (`OB_MAX_AGE_DAYS`),
   but its BOS candle falls off the 150-bar slice after ~6 trading days. Today
   `_is_ob_mitigated_replay` returns `(False, "")` when `locate_ob_candle_idx` does not
   find the anchor — with the clamp, old OBs would silently become immortal
   (never mitigated, still able to fire alerts).
   - Edit `_is_ob_mitigated_replay` (lines 96-124): when `found` is False, check whether
     the anchor timestamp is EARLIER than the slice's first bar; if so, every visible bar
     is post-event, so scan the whole slice:
     ```python
     anchor_idx, found = smc_detector.locate_ob_candle_idx(df_h1_slice, anchor_ts)
     if not found:
         # Anchor older than the clamped window => every visible bar is after
         # the structural event; scan the full slice. Any other unresolved
         # anchor is a data problem -> keep the safe "not mitigated" answer.
         try:
             _a = pd.Timestamp(anchor_ts)
             if _a.tzinfo is None:
                 _a = _a.tz_localize("UTC")
             if len(df_h1_slice) and _a < df_h1_slice.index[0]:
                 anchor_idx, found = -1, True   # start_idx = anchor_idx + 1 = 0
         except Exception:
             pass
     if not found:
         return False, ""
     ```
     (`start_idx=anchor_idx + 1` is the existing call; `-1 + 1 = 0` scans from the slice
     start. Touch counts are then window-local — same limitation live has; the hourly
     cadence catches distal breaks on the bar they happen, so nothing is missed.)
   - `OB_MAX_AGE_DAYS = 15` in the replay stays UNCHANGED (mirrors the live slate, which
     persists zones beyond the detection window via state file).

### Explicitly accepted effects
- Detection memory becomes ~150 bars (~6 trading days), exactly live's. Swings, event
  ring, H4 range, trend state all now derive from the same window live sees.
- All headline metrics change vs previous runs. The first clamped full run is the new
  baseline; note this in the run manifest / RECOMMENDATIONS.
- Runtime drops sharply (no more full-history recompute per bar).

4. **A/B override (diagnostics only).** `replay_pair` gets an optional
   `detection_bars` parameter, default `smc_radar.LIVE_DETECTION_BARS`; `None` =
   legacy unclamped behaviour (warmup threshold falls back to the old 50 in that
   mode, and the slice-length assert is skipped). The runner NEVER sets it in
   normal runs — it exists solely for the validation A/B below.

### Regression guard
- In `replay_pair`, after the warmup check (clamped mode only), add a hard guard:
  ```python
  assert len(h1_slice) == smc_radar.LIVE_DETECTION_BARS, "slice/window mismatch"
  ```
- Add a scanlog FAIL condition if the assert would fire (mirror the existing
  `ALERT_LOOKAHEAD_BLOCKED` pattern) OR let the assert abort loudly — either is fine,
  silent divergence is not.

### Is 150 the RIGHT window? (sizing rationale + the one open empirical check)
150 is live's current value, but parity alone does not prove it correct. Component
needs, verified in code:
- ATR(14): 15 bars (smc_detector.py:92-94) — identical for any window ≥ 15.
- Swing geometry: local (±3 bars). OB walk-back / FVG / sweep: bounded inside the
  event's own impulse leg — always recent.
- Zones older than the window: PERSIST via `active_obs` (backtest) / slate (live) —
  the detection window does not need to cover the 15-day OB age cap.
- Trader's own relevance prior: dealing-range lookback is set at ~5 trading days for
  forex (smc_detector.py:133). 150 bars ≈ 6.25 trading days ≥ that.
- Trend/defended state: where a long walk and a 150-bar walk disagree, the long walk
  is carrying months-old H1 state that current swings never re-confirmed. SMC defines
  trend by CURRENT confirmed structure; stale carried state is memory, not structure.
  Where they differ, the short window is the methodologically defensible answer.
- Second-order window effect (be precise): the swing leg-ATR filter
  (`_filter_swings_by_leg_atr`, dealing_range.py:277-323) is path-dependent from the
  slice START — the kept/dropped chain can differ near the window's left edge before
  it re-anchors (each kept swing resets the reference). So "only trend changes" is
  NOT claimed; window size can also perturb which left-edge swings survive. This is
  bounded and decays within a few swings; it is captured empirically by the A/B's
  alert-match and trend-agreement metrics, not argued away.
- The ONLY component that can legitimately want more than 150 bars: the H4 range
  (needs a confirmed H4 swing pair; in a strong one-way trend the counter-side swing
  can age out of ~37 H4 candles → `no_confirmed_pair`, walls invalid).
- Regardless of all of the above: the backtest window MUST equal live's window. The
  enumeration matters only for deciding whether LIVE's window should grow — and that
  is decided by the H4-starvation counter below, not by intuition.

Therefore the validation A/B MUST measure H4-range starvation (step 3 below). Decision
rule: if arm B's H4-invalid rate is within ~2 percentage points of arm A's, 150 is
CONFIRMED as the shared window. If arm B is materially more starved, the remedy is to
RAISE the SHARED constant for live AND backtest together (e.g. 240 ≈ 10 trading days;
note live's adapter fetch is `outputsize=200` — bump it alongside), never to let the
backtest silently use a bigger window than live.

### VALIDATION RUN (mandatory before the full baseline — trader reviews the diff)
Budget: 5–10 minutes total. Do NOT ship Fix 1 on inspection alone.
1. Walk window ~2 months (e.g. 2024-10-01 → 2024-11-30) on 3 pairs: one sub-1 FX
   (EURUSD), USDJPY, XAUUSD. NAS100 excluded (dropped from the system, trader
   decision 2026-07-02).
2. Run A = `detection_bars=None` (legacy). Run B = default (150). Same config otherwise.
3. Diff report (throwaway script is fine):
   - alerts per pair per arm + match rate keyed `(pair, alert_ts, ob_timestamp)`
   - per-bar trend agreement % between arms (scanlog `trend` field)
   - **per-arm % of scanned bars with h4_range invalid, split by source
     (`no_confirmed_pair` vs other)** — this decides the window question above
   - per-arm % of bars with trend undefined
   - headline expectancy per arm (informational only — window too thin to judge edge)
   - zombie check: arm B must show `ob_mitigated` events for OBs whose
     `bos_timestamp` predates the 150-bar slice (proves the anchor fallback works)
4. HAND THE DIFF TO THE TRADER. Shipping is his call, not an auto-pass. Expected
   picture: trend agreement high (~85%+), alert overlap well above half, H4-invalid
   rates within ~2pp of each other, gates green, no crashes. Trend agreement below
   ~80% or H4 starvation above the rule = stop, report, decide window size with the
   trader before the full run.

---

## FIX 2 — ATR-scaled same-leg dedupe threshold

### Why (one line)
The dedupe threshold is 0.0003 absolute price units for every instrument, so same-leg
dedupe is inert on USDJPY (~150), XAUUSD (~2600), NAS100 (~18000+): duplicate same-zone
OBs survive, can occupy both OB1 and OB2, and double-count alerts/trades on one zone.

### Changes

1. **smc_radar.py — constant** (near `OB_PROXIMITY_ATR`, line 78):
   ```python
   # Same-leg dedupe: two same-direction OBs whose proximal lines are within
   # this fraction of H1 ATR are the same visual zone. 0.25 x a typical EURUSD
   # H1 ATR (~0.0012) ~= 0.0003, the old hardcoded forex threshold -- sub-1
   # pair behaviour is preserved; JPY/Gold/NAS dedupe starts working.
   DEDUPE_PROXIMAL_ATR_MULT = 0.25
   ```

2. **smc_radar.py — `_dedupe_same_leg`** (nested in `detect_smc_radar`; edit line 1144):
   ```python
   thresh = (DEDUPE_PROXIMAL_ATR_MULT * h1_atr_for_leg
             if h1_atr_for_leg and h1_atr_for_leg > 0 else 0.00030)
   ```
   `h1_atr_for_leg` is already in scope (computed at smc_radar.py:579). The 0.00030
   fallback keeps today's behaviour when ATR is unavailable (never worse than now).

3. **Cleanup:** delete the dead `o.pop('_dedupe_thresh', None)` (line 1169) and correct
   the docstring "pair-aware threshold" wording (lines ~1094-1096) to describe the
   ATR-scaled rule.

### Notes
- This function is shared: live Phase 1 digest dedupe is fixed by the same edit. Intended.
- Trading-logic change, trader-approved 2026-07-02.

### Regression guard (unit tests, new file `tests/test_dedupe.py` or the existing test home)
- JPY scale: two same-direction OBs, proximals 0.02 apart, H1 ATR 0.5 → MUST merge to one.
- EURUSD scale: proximals 0.0002 apart, ATR 0.0012 → merge; proximals 0.0010 apart → do NOT merge.
- ATR None/0 → falls back to 0.0003 behaviour.

---

## FIX 3 — Refresh mutable OB state in the replay (stale `ob_touches` etc.)

### Why (one line)
The replay freezes each OB at first detection; live refreshes zone state every scan —
so trades.csv logs touch counts / FVG state / break quality as of zone BIRTH, not as of
the alert, and the freshness-by-touch insight measures the wrong thing.

### Changes (all in backtest/replay_engine.py unless stated)

**3a — touches + status, updated every bar.**
- Change `_is_ob_mitigated_replay` to pass through the third return value:
  return `(mitigated, reason, touches)`; on the early-exit paths return `(False, "", 0)`
  and `(False, f"mitigation_check_error: {e}", 0)`.
- In the mitigation loop (lines ~298-321), on the NOT-mitigated path stamp the stored OB:
  ```python
  ob["touches"] = int(touches)
  ob["status"] = "Pristine" if touches == 0 else f"Tested ({touches}x proximal)"
  ```
  (Label format mirrors live smc_radar.py:1063 exactly.)

**3b — one-time `break_quality` re-grade.**
Context: `break_quality` is graded at OB build with an option-B window (break candle +
next candle). When the OB is built on the break's edge bar, the next candle does not
exist yet, and the replay never re-grades. The two candle bodies never change afterwards,
so grading ONCE when the window is complete is the faithful measure.
- In the same mitigation loop, after 3a, add:
  ```python
  if not ob.get("_bq_regraded"):
      _a_idx, _a_found = smc_detector.locate_ob_candle_idx(
          h1_slice, ob.get("bos_timestamp") or "")
      if _a_found and _a_idx + 1 < len(h1_slice):
          ob["break_quality"] = smc_detector.compute_break_quality(
              h1_slice, _a_idx, ob.get("bos_swing_price"),
              ob.get("direction"), smc_detector.compute_atr(h1_slice),
              event_type=ob.get("bos_tag", "BOS"))
          ob["_bq_regraded"] = True
  ```
  `compute_atr` is memoised (ATR cache) — no perf concern. Signature verified:
  `compute_break_quality(df, bos_idx, broken_price, direction, atr, event_type)`
  (smc_detector.py:2689).

**3c — FVG state refresh on re-surface (mirrors live's per-scan zone refresh).**
- In the merge block (lines ~354-359), instead of skipping a matched OB outright,
  refresh its FVG dict from the fresh detection:
  ```python
  existing_by_ts = {o.get("ob_timestamp"): o for o in state.active_obs[pair_name]}
  for ob in obs:
      match = existing_by_ts.get(ob.get("ob_timestamp"))
      if match is not None:
          match["fvg"] = ob.get("fvg", match.get("fvg"))
          continue
      ...  # existing new-OB path unchanged
  ```
  The fresh copy is computed from the closed slice — point-in-time clean. Only `fvg` is
  refreshed here; touches/status are owned by 3a and break_quality by 3b (one concept,
  one implementation).

**3d — freeze the alert-time value (collision guard).**
`run_backtest` mutates the SAME OB dict after the alert (e.g. exhausted flag), and the
per-bar loop keeps updating `touches`; if the row is built when the trade closes, the
logged count would include post-alert touches. Freeze a scalar at fire time:
- At the alert fire block (just before the `yield` at line ~526, next to the
  `bos_verdict` stamp at line 500):
  ```python
  ob["touches_at_alert"] = int(ob.get("touches") or 0)
  ob["fvg_at_alert"] = dict(ob.get("fvg") or {})
  ```
- **backtest/h1_only_simulator.py:1255** — change the row source:
  ```python
  "ob_touches": ob.get("touches_at_alert", ob.get("touches")),
  ```
- If `_build_row` (or the FVG fill-freshness logic near h1_only_simulator.py:301-333)
  reads `ob["fvg"]` AFTER the fill for classification, point it at `fvg_at_alert` the
  same way. If it already snapshots at alert handling time, leave it — verify before
  editing, do not blind-edit.

**3e — OB-sourced column classification + class-killing regression test.**
Purpose: make "mutable state logged from a frozen snapshot" a CLOSED bug class, not
whack-a-mole. Classification of every field the row-build path reads off the OB dict
(verified against the OB build at smc_radar.py:1005-1039 and the row build at
backtest/h1_only_simulator.py:1039-1267):

- IMMUTABLE EVENT FACTS — freezing is correct: `bos_timestamp`, `ob_timestamp`,
  `direction`, `bos_tag`, `bos_tier`, `bos_swing_price`, `impulse_start_price`,
  `high`, `low`, `proximal_line`, `distal_line`, `median_leg_body`, `ob_body`,
  `h1_atr` (formation ATR, frozen BY DESIGN — the row comments say so),
  `reversal_pct`, `broken_was_wall`, `bos_sequence_count`, `last_choch_idx`.
- FROZEN-BY-DESIGN AND LIVE DOES THE SAME: `dealing_range` (live computes once at OB
  build, Phase 2 never recomputes — smc_radar.py:961-985), `sweep_observed`
  (snapshot semantics; owned by the sweep-rebuild workstream — do not touch).
- STAMPED AT ALERT (correct source): `bos_verdict` (replay_engine.py:500),
  `touches_at_alert` + `fvg_at_alert` (new, 3d), `h1_trend` / `trend_alignment` /
  `alert_bar_*` (yield payload).
- MUTABLE STATE, FIXED BY THIS SPEC: `touches`/`status` (3a), `break_quality` (3b),
  `fvg` (3c/3d).

Add this classification as a comment block above `_build_row` with the rule: any NEW
ob field logged in a trade row MUST be placed in one of these buckets; mutable state
must be stamped `*_at_alert` at the yield — never read live at row-build time.

Regression test (kills the class, not just the instance): build an alert whose OB has
`touches_at_alert=1` and an alert-time fvg snapshot; then mutate `ob["touches"]=3` and
`ob["fvg"]` BEFORE calling `_build_row` (simulating post-alert replay mutation); assert
the row logs 1 and the alert-time fvg values.

### Explicitly accepted effects
- `ob_touches` changes meaning: touches accrued between the structural event and the
  ALERT (was: touches at the moment the replay FIRST SAW the zone — 0 for most OB1s,
  nonzero for late-surfacing OB2s, meaningful for neither). Any existing
  "freshness/touch" insight computed on the old column is INVALID and must be
  re-derived from the first post-fix run. Flag this in the run manifest.
- `break_quality` distribution shifts slightly for OBs built on the break's edge bar.
- No change to which alerts fire, fills, or P&L — this is measurement/logging only.

### Regression guard
- Unit/fixture test: synthetic frame where price wick-touches proximal once, pulls back
  beyond the re-arm band, then approaches again to fire the alert → row must log
  `ob_touches == 1` (not 0, not 2).

---

## PARITY STATEMENT — what is and is not identical to live after this spec

Identical (same shared functions, same 150-closed-bar frame, both systems):
- `compute_pair_walls` / `compute_structure` (BOS/CHoCH/trend) / H4 range / ATR /
  swing definition / OB build + OB1-OB2 selection / mitigation + touch rule / dedupe.

Known remaining divergences — accepted or out of scope, do NOT "fix" them here:
1. **Alert timing:** live proximity reads the FORMING bar; the backtest reads the
   closed bar (~1 bar pessimistic on entries). Accepted, documented decision.
2. **Data feed:** backtest = MT5 parquet; live = Twelve Data (adopted 2026-06-29).
   Different quote sources → occasional level differences (TD-vs-MT5 eval: FX p50
   ~1 pip, p95 ~5-12 pips on daily rebuilds). Feed gap, not logic gap — no code
   change in this spec can close it.
3. **NAS100 is OUT** (trader decision 2026-07-02): it has no live Twelve Data symbol
   and is dropped from backtest runs too. Exclude it from the validation run and the
   baseline run (remove it from the backtest run config / pair list; leave historical
   data untouched).
4. **Slate persistence details** (re-alert cadence, drop narration) live in
   `run_radar`; the replay's armed/cooling machine implements the same rules in
   separate code.

Claimed: detection-LOGIC parity on identical input. NOT claimed: 100% parity of
alert streams (items 1-3 above are real, known, and accepted).

## NO-OP / PARKED — do NOT implement

- **Swing-ingest ordering (sections 1-3 run before section 4 in `compute_structure`):**
  NOT a defect. Proof: a swing low at idx i confirms on bar i+3 only if L[i] is STRICTLY
  below every low in the window — so the confirming bar's low (hence its close) is above
  the pivot and can never simultaneously break it downward (symmetric for highs). The
  order is provably safe. Leave it.
- **`reversal_pct` / `choch_from_zone` epoch drift** (premium/discount lines computed
  once per slice at dealing_range.py:733-737, applied to historical candles): PARKED.
  Residual error is near zero once Fix 1 lands (events are captured at the slice edge
  when fresh); a per-event H4 recompute would need a dealing_range→h4_range import,
  which is circular. Live behaves identically, so parity holds. Do not touch.
- **Sweep detector (`observe_phase1_sweep`):** owned by the separate sweep-rebuild
  workstream. Do not touch.

## Execution order & acceptance

1. Fix 2 (isolated, unit-testable) → run dedupe unit tests.
2. Fix 3 (replay + one simulator line) → run the touches fixture test.
3. Fix 1 (clamp + warmup + anchor fallback) → assert guard in place.
4. Structure-golden gate green (re-baseline if red per policy).
5. Short window run (~3 months): G1–G10 green, scanlog shows no
   ALERT_LOOKAHEAD_BLOCKED / slice-mismatch conditions, per-pair funnel diag sane.
6. Full re-baseline run. Mark manifest: "first run with live-window parity
   (LIVE_DETECTION_BARS=150) + ATR dedupe + alert-time OB state".
