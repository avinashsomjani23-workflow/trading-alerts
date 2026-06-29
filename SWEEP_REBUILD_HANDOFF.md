# Sweep Detection Rebuild — Build Handoff

**For the next chat. Read the mandate first — it is not optional.**

---

## 0. Mandate (the tone this work was built in)

This is a **data-first, brutally honest** project. The next chat must be the same:
- **Verify against the code and the data before claiming anything.** No assertions from
  memory. Every number must come from running real data — re-run if in doubt.
- **No sycophancy. Disagree when the user is wrong.**
- **When data and SMC conflict, it is a DISCUSSION POINT, not a conclusion.** Name the
  likely cause; do not silently score on it.
- **Gate nothing first.** Any new sweep score is a SCORE input, backtested, before it can
  ever block a trade.
- Plain English, bullets, short. The user hates verbosity.

---

## 1. What we are building

A sweep detector that scores **what liquidity was swept and whether the sweep actually
worked** — not just "was there a long wick." Two missing references get added:
1. **Liquidity tier** — was the swept level a daily/weekly pool (PDH/PWH/PDL/PWL) or just a
   random internal wiggle? Today they score the same. They are not the same.
2. **Follow-through** — did price actually displace away after the sweep, or did it wick and
   die? A rejecting wick is not a successful liquidity grab.

---

## 2. The problem (start here — do NOT skip the verification step)

- **User's report:** there is an **inverse correlation between sweep presence and trade
  results** — setups WITH a detected sweep underperform.
- **This is UNVERIFIED by the handoff author.** It came from a prior chat.
- **STEP 1 OF THIS BUILD, before any redesign:** reproduce that correlation from
  `r_realised` (the single P&L source of truth). Slice filled trades by sweep
  present/absent and by sweep tier, compare expectancy + WR. Confirm the inverse
  correlation is real and find WHERE it concentrates (which pairs, which tiers, presence-
  only vs graded). If it does NOT reproduce, stop and report — do not rebuild on a ghost.

---

## 3. What the current system does (verified in code — read these before touching anything)

**Detection — `observe_phase1_sweep` (`smc_detector.py:1033`):**
- Phase 1 takes an **H1 snapshot at OB formation**. Phase 2 **consumes the frozen snapshot,
  never re-grades** (`smc_detector.py:2048`).
- Search window = the structural leg `[prior_event_idx, ob_idx]` (CHoCH/BOS anchored), with
  a candle-count fallback when no anchor exists.
- **Targets ACTIVE internal pivots only** — unbroken AND unswept swings in the window
  (`get_swing_points`, lookback=3). Drained/broken swings correctly don't qualify. This part
  is SMC-faithful.
- A qualifying sweep: wick pierces a prior same-type pivot by `SWEEP_WICK_PIERCE_MIN_ATR`
  **and closes back** past it. Deepest active pierce wins; tie-break = more recent.

**Scoring (max 3.0, then scaled):**
- `base` + `equal_levels` (cluster / equal-highs-lows proxy) + `rejection`
  (`_rejection_score:829` = wick/body ratio, 4 tiers) → `_sweep_tier:873`
  (textbook/decent/weak/none).
- **Consumption (`smc_detector.py:2089`):**
  - **Non-JPY forex = PRESENCE-ONLY.** `bd["sweep"] = 1.0 if tier != none else 0.0`. The
    entire quality grade is **discarded** — rationale on file: spot forex has no centralized
    stop pool, so quality grading was judged noise.
  - **JPY / Gold / NAS = graded.** 0-3 score scaled to a 0-2 budget.

---

## 4. Suspected weaknesses (ranked — verify each against the Step-1 data)

1. **No HTF liquidity reference. THE big one.** Targets internal pivots only. Sweeping a
   random internal wiggle scores identically to sweeping PDH/PWH/PDL/PWL. ICT says the
   pool's rank IS the signal. **This is the hard dependency on the daily-bias build** (it
   provides the ranked pools — see §6).
2. **No follow-through / outcome check.** A wick that closes back counts as a "sweep" even if
   price never reversed. Rejection geometry ≠ a liquidity grab that led anywhere. Need:
   after the sweep candle, did price **displace away** from the swept level within N candles
   (reuse H1 FVG / body-dominant close)? No displacement = failed sweep = no credit.
3. **Presence-only on forex may be the culprit — or the only useful part.** If the quality
   grade is noise (as suspected) and forex already ignores it, then for forex the inverse
   correlation is driven by **sweep PRESENCE itself**. Test directly: is presence alone
   inversely correlated on forex? If yes, the fix is not better grading — it's that a bare
   sweep with no tier/follow-through is a weak or contrary signal.
4. **Frozen snapshot at OB formation.** The sweep is judged once and never revisited. If the
   OB forms before the real stop-run, the true sweep is missed.
5. **Leg-anchored window** can catch in-leg noise or miss the actual stop-run outside it.

---

## 5. What to KEEP vs BUILD

**Keep (sound mechanics — do not reinvent):**
- The wick/body rejection geometry (`_rejection_score`).
- Active-swing targeting (only unbroken/unswept pivots are valid liquidity).
- Equal-levels clustering (a decent *internal* proxy for a stronger pool).

**Build:**
- **Tier the swept level** by the liquidity hierarchy: internal < PDH/PDL < PWH/PWL. The
  tier becomes the dominant term in the new score.
- **Follow-through gate:** sweep only earns credit if price displaces away from the swept
  level within N candles (binary close-back / FVG test — keep it simple, no graded buckets).
- **Re-decide forex:** does forex stay presence-only, or does a *tiered* sweep (PWL swept,
  not a wiggle) finally earn a graded signal on forex? Let the Step-1 data answer.

---

## 6. Dependencies (respect the order or this fails)

- **HARD dependency: the liquidity hierarchy from `DAILY_BIAS_BUILD_HANDOFF.md` must land
  first.** It computes the ranked pools (PDH/PDL from real D1, PWH/PWL from real W1 — no
  resampling) that this rebuild tiers against. Without it, weakness #1 cannot be fixed.
- **Data:** MT5 D1/W1 CSVs (backtest, already pulled) + yfinance D1/W1 (live). H1 from the
  existing feed. `r_realised` for the correlation test.
- **Sequence:** liquidity hierarchy → reproduce inverse correlation (§2) → redesign sweep
  with tier + follow-through → backtest as a SCORE input → only then consider any gate.

---

## 7. Where it is consumed (don't break these)

- **Phase 1** (`smc_radar.py`) calls `observe_phase1_sweep`, writes `ob['sweep_observed']`.
- **Phase 2** (`Phase2_Alert_Engine.py` → `run_scorecard` / `generate_scorecard_rows`)
  reads the snapshot into `bd["sweep"]`, the confidence-score line, and renders the sweep
  breakdown in the email.
- Schema is shared (`components`, `hours_before_anchor`, `tier`, `score`). Any new field must
  carry through the canonical empty shape so missing snapshots degrade cleanly.

---

## 8. Validation (before anything goes live)

- Reproduce the inverse correlation FIRST (§2). No redesign until it's confirmed and located.
- Any new sweep score is a **SCORE input only**. Re-run the backtest; confirm the new
  tier+follow-through score is positively correlated with `r_realised` out-of-sample, not
  just in one slice.
- Promote to a gate ONLY if it earns it. Gate nothing first.

---

## 9. Honesty caveats to carry forward

- The inverse correlation is the user's prior finding — **treat it as a hypothesis to
  reproduce, not a fact.**
- The current sweep is more sophisticated than "long wick = sweep" (active targeting +
  clustering). Do not strawman it. The real gaps are the HTF tier and follow-through, plus
  the forex presence-only question.
- Better entries still bleed if exits leak — exit work is a separate chat. Sequence P&L
  levers honestly; don't claim this rebuild fixes more than it does.
