# EDGE LAB — MASTER RESEARCH HANDOFF (Steps 1 → E)

**Stamped:** 2026-07-09. **Author:** Opus, from a working session with the trader.
**Purpose:** the single map for the whole edge-research phase. Every per-step chat reads
THIS file first, does ONE step, writes its result back. You never hold more than one step
in your head.

**This file consolidates and supersedes the scattered plan.** It does NOT replace:
- `SMC_EDGE_LAB_SPEC.md` — the frozen method bible (stats, splits, gates). Still law.
- `DECISION_GUARDRAILS.md` — the frozen decision rules (flag violations by ID first).
It DOES tie together the step handoffs and the old loser-mechanics files below.

---

## 0. HOW TO USE THIS FILE (read once)

- **One step = one chat.** Never carry two steps in one conversation.
- **Read order every chat:** this file → the frozen `SMC_EDGE_LAB_SPEC.md` → the prior
  step's stamped output. Then do only your step.
- **Nothing ships to live trading here.** Every step PROPOSES; the trader DISPOSES.
- **Two buckets for every finding:** (a) ship-gate queue, (b) "interesting, not proven."
  Nothing floats between them.
- **THE RESEARCH FLAGS DEAD-ENDS EXPLICITLY (trader cannot spot them alone).** When a lever
  is not worth the trader's time, SAY SO in plain words with the evidence: "proven dead —
  here is the test that killed it" or "unproven but low-odds — spend effort elsewhere." Do
  not let the trader burn time on a settled null. Equally: when something is genuinely
  unproven (not closed), say that too — never overclaim a null as settled (see §2).
- **The research enumerates a full family — the trader is not expected to.** If a signal
  belongs to a family (geometry levers, OB-quality features…), test EVERY member, not the
  one or two that happened to come up in conversation.
- **The holdout (2022–2025) is opened ONCE, at the very end.** Never peek early (C5).

---

## 1. THE METHODS, IN PLAIN ENGLISH (so you can read any table)

You do not need maths. You need to know what each word means when you see it.

**The raw ingredients**
- **Trade** — one alert that filled and resolved. ~23,000 filled across 18 years.
- **R** — profit/loss of a trade measured in units of its own risk. +1R = made one risk;
  −1R = lost the full risk. This is the source of truth (`r_realised`).
- **Expectancy** — the average R across a group of trades. Positive = that group makes
  money on average; negative = it bleeds. **This is the number we are trying to turn
  positive.**
- **Win rate** — % of trades that won. Useful but a trap on its own: a high win rate with
  tiny wins and huge losses still loses money. Expectancy is king, win rate is a passenger.

**How we look at a feature (e.g. OB size, FVG size)**
- **Bucket** — we split a feature into groups (e.g. small / medium / large OB). Then we
  look at each group's expectancy, win rate, and death rate. A "bucket curve" is those
  numbers laid out group by group.
- **Straight-to-SL / death rate** — % of a group that lost immediately (hit stop, ~−1R,
  almost no favourable move first). Your small-OB concern lives here.
- **Cliff** — one group (usually the worst) fails badly while the rest look fine. A single
  average hides a cliff. **New rule (spec §3 / guardrail F-BUCKET): we now always print
  every bucket, so a cliff can never hide again.**

**How we tell a real signal from luck**
- **Effect size** — how big the gap is between the good group and the bad group, in R.
  Small effect = not worth the trading cost.
- **Bootstrap CI (confidence interval)** — we reshuffle the trades 1,000 times to ask "is
  this gap solid, or could random luck produce it?" If the interval crosses zero, it might
  be luck. If it stays on one side of zero, it's more real.
- **Spearman** — asks "as the feature goes up, does the result move steadily up (or down)?"
  A trend test for ordered things.
- **Mutual Information (MI)** — asks "does knowing this feature tell you ANYTHING about the
  outcome, even a bendy U-shape a trend test misses?" MI = 0 means truly no information.
  (This is why `alert_utc_hour` was rejected: MI = 0 → it's noise dressed as significance.)
- **Consistency (per-quarter / per-year)** — does the signal show up again and again across
  time, or only in one lucky stretch? A real edge repeats.
- **AUC** — how well a score separates winners from losers. 0.50 = coin flip (useless),
  1.00 = perfect, **0.64 = weak but real** (we can smell death, not cleanly enough to
  trade on).

**How we test overfitting (fooling ourselves)**
- **Discovery / Validation / Holdout** — three time periods. We hunt on Discovery
  (2008–2016), confirm on Validation (2017–2021), and open Holdout (2022–2025) exactly
  once at the end. A signal that only works on the years we hunted is memorised, not real.
- **Purged CV** — when testing across time, drop trades whose life overlaps the test
  window so the model can't "peek." Stops a subtle leak.
- **PBO / DSR** — scorekeepers for the WHOLE search: "given how many things we tried, how
  likely is our best pick just luck?" Used for the interaction/model hunt.

**How we test exits (the survival engine)**
- **The walker / real-order replay** — re-runs a trade bar-by-bar with a *different* rule
  (e.g. a wider stop) and reports the REAL result. This is the only honest way to test an
  exit idea.
- **Touch vs exit (C6)** — "price touched TP1" is NOT "we banked TP1." A touch is a hint.
  Only a walker replay counts as a result. Every "prize" number in the old files is a
  touch-based ceiling — never bankable.

---

## 2. WHERE WE STAND (the honest status, corrected)

- **The system loses on raw fills.** Discovery expectancy ≈ −0.05R/trade. The whole 18-yr
  run is negative. This is the problem we are solving.
- **Most losers die instantly** (~66% of true losers hit stop with almost no favourable
  move). The mechanism is 1-bar geometry: limit sits at the zone edge, the touching bar
  runs ~1 ATR through the zone AND the stop in one move.
- **Entry SELECTION status — NOT closed (corrected 2026-07-09).** Simple selection on the
  columns we currently log has not found an edge that turns the kept set positive. BUT:
  per-pair discovery, interactions/XGBoost/SHAP, and any better-engineered feature are
  **not yet done on clean data.** Do NOT state "selection has no edge" as settled. The
  correct claim: "the easy single-column win isn't there; the thorough hunt is unfinished."
- **The barbell** — the trades that predictably die (e.g. small OBs) are ALSO the best
  winners (small-OB win = 2.06R vs large-OB 1.59R). So you cannot just delete them; you'd
  cut the champions too. This is why raw size floors fail — and why the lever may be
  changing geometry (exits), not deleting trades.
- **Exits are the top open lever by ceiling.** ~53% of losers' stop candles are wicks
  (swept then closed back) — BUT whether those recovered or kept losing is **not yet
  measured** (see red flag below). The new columns exist to answer it.

---

## 3. THE PLAN — SIMPLE TABLE FIRST

Read this table top to bottom. Details for each step follow in §4.

| Step | What we do | Why (in one line) | Main method | Runs when |
|---|---|---|---|---|
| **1. Correctness gate** | Verify detector, walker, simulator, discovery module, CSV provenance are bug-free & logical | Never analyse on broken plumbing — bugs mid-analysis have burned us | Code read + targeted checks; STOP & fix if anything is wrong | FIRST — blocks everything |
| **A. Bucket-cliff diagnostic** | Print the full bucket curve for EVERY feature (win rate, mean R, straight-to-SL rate, N; best & worst bucket named) | Find hidden cliffs like small-OB death that a single number buried | Bucket tables + the new F-BUCKET guardrail | After Step 1 passes |
| **B. Exit lever (all trades)** | Wider / sweep-aware stops on the WHOLE population; measure if swept losers recovered | Changing geometry may save losers WITHOUT deleting any trade | Walker real-order replay, sized from wick-depth columns | Parallel with C |
| **C. Entry discovery (deep)** | Per-pair + interactions + XGBoost/SHAP on clean data | The thorough hunt the "null" never actually finished | Spec §4b/§5 methods + PBO/DSR | Parallel with B |
| **D. Freeze filter, re-tune exits** | Lock any entry filter, then tune exit rules on the trades that survive | A filter changes which trades exist → exits must be tuned on the real survivors | Walker replay on survivors | After B & C decided |
| **E. Ship gate + holdout** | Apply the 5-point gate; open 2022–2025 ONCE | The final, one-shot exam | Spec §2 gate + holdout | Last, once |

**Sequence logic (why this order):**
- Correctness before everything — a bug invalidates every number after it.
- Diagnostic early — cheap, tests a concrete belief, exercises the new guardrail.
- **B and C run in PARALLEL and INDEPENDENTLY** (trader's call, correct): both are just
  different reads of the same frozen CSV; neither needs the other's result. Entry discovery
  is NOT dependent on the exit result.
- Exits keep ALL trades (they change management, they don't delete trades) — so we lose no
  volume and no SMC knowledge while testing them. Filtering, if any, comes LAST and only on
  the residue that stays negative under the best exit.
- Holdout dead last — spent once, protects the whole verdict.

---

## 4. EACH STEP IN FULL

### STEP 1 — CORRECTNESS GATE (do this first, always)
- **Entry:** none. This blocks all analysis.
- **Do:** verify the machinery is correct and logical, not just present:
  - Detector path (`smc_detector.py`, `smc_radar.py`, `dealing_range.py`) — does the live
    code produce the OBs/structure it claims? Code is truth, not comments.
  - The walker / replay (`edge_engine.py:_replay_recipe` @689, `_ensure_bars` @661,
    `walk_multileg`) — does a replay reproduce `r_realised` exactly (baseline self-check)?
  - The simulator (`h1_only_simulator.py`) — the new columns (`sl_wick_depth_atr`,
    `sl_max_adverse_after_sweep_atr`, `bars_sl_to_tp1_touch`, `sl_recovered_to_entry`,
    `sl_distance_atr`, `r_capture_ratio`, `trend_pd_agree`) computed correctly, stamped at
    the right time, matching their TRUTH_LEDGER rows.
  - The discovery module (`edge_lab.py`, `edge_lab_step2.py`) — schema guard actually bites;
    tests pass; zero look-ahead leak.
  - **CSV PROVENANCE — RESOLVED 2026-07-09.** The current baseline is `trades.csv`
    committed at **`1feb2db` (2026-07-09 00:05 UTC), git-clean/reproducible**, 108 cols,
    33,838 rows, **11,366 discovery-eligible**, all 7 new columns populated. The **Step-2
    scorecard (07-08 13:14 UTC, N=11,329) is STALE** — it ran on an intermediate CSV since
    replaced. Three drifting discovery counts confirm the churn: 11,352 (old corrupted) →
    11,329 (Step-2's CSV) → 11,366 (current committed). **Action for this step:** run the
    scaffold guard + Step-2 scorecard fresh on `1feb2db`; discard the old scorecard numbers.
    Remaining Step-1 work (walker self-check, detector audit, new-column stamp-time check)
    still to do before analysis.
- **Exit:** a written PASS/FAIL. If ANY check fails → STOP. Fix the bug, re-run whatever it
  touched, then resume. No analysis on suspect plumbing.
- **Guard note:** these checks live out-of-band (tests / standalone scripts), never in the
  live alert path.

### STEP A — BUCKET-CLIFF DIAGNOSTIC (your small-OB finding)
- **Entry:** Step 1 PASS.
- **Do:** for the FULL entry-legal feature list (all 44+, not a few), print the complete
  bucket curve per the new F-BUCKET rule: each bucket's N, win rate, mean R, and
  **straight-to-SL rate**, with the best and worst bucket named. Implement the code+test
  enforcement of F-BUCKET here (the guardrail's teeth).
- **First target:** `ob_range_atr`, `ob_body_ratio`, `fvg_size_atr` — does the smallest
  bucket carry a death cliff, and how much of the total loser pile does it own?
- **Legality:** the per-bucket SL rate crosses an entry feature against loser behaviour —
  a DESCRIPTIVE lens (legal on all trades), never an entry input. Wall stands.
- **Exit:** one bucket-curve report; any cliff surfaced as a candidate for the ship gate
  (NOT auto-shipped). Write findings back to this file's ledger.

### STEP B — EXIT LEVER, ON ALL TRADES (the geometry fix)
- **Entry:** Step 1 PASS. Runs parallel to C. Descriptive on all trades (allowed pre-filter).
- **Do:**
  - Measure, on swept losers, whether they RECOVERED or kept losing — using
    `sl_max_adverse_after_sweep_atr`, `sl_recovered_to_entry`, `bars_sl_to_tp1_touch`. This
    answers the "and then what?" that the bare 53%-swept number cannot.
  - Real-order replay a wider / sweep-aware stop (SL = distal + k·ATR grid), k sized from
    the `sl_wick_depth_atr` distribution. Report net-R delta vs baseline, per split.
  - Break-even sweep track: test delaying/offsetting/dropping BE arming, same walker.
- **THE FULL GEOMETRY FAMILY (test all — do not stop at wider stops).** "Change the
  geometry, don't delete the trade" has many levers; enumerate and replay each:
  1. Wider / sweep-aware stop (SL = distal + k·ATR) — top open lever.
  2. Stop sized beyond one bar's range (deaths are ~1.1 ATR = one H1 bar).
  3. Break-even arming rule (delay / offset by wick depth / drop).
  4. Partial / scale-out before the sweep zone.
  5. Trailing rule (let winners run vs fixed TP).
  6. Time-stop (exit if no progress in N bars).
  7. Target placement (TP1 distance vs sweep-and-reverse behaviour).
  8. Deeper entry (50%-zone) — already REJECTED (adverse selection), listed so it is not
     re-tried without new evidence.
  The trader found the small-OB cliff by chance; it is the RESEARCH's job (not the
  trader's) to enumerate every bucket of a family and test it. Never present two levers as
  "the family" when more exist.
- **The point:** keep every trade, change the stop. If a wider stop lifts the population
  materially, that's the geometry win — no trade deleted.
- **C6 discipline:** no touch-based prize is banked; only walker replay counts. Every old
  "+X R" ceiling shrinks as the stop widens — prove the real number, never quote the ceiling.
- **Exit:** exit-rule recommendations with real-order-replay net-R deltas per split.

### STEP C — ENTRY DISCOVERY, DEEP (the hunt the null never finished)
- **Entry:** Step 1 PASS. Runs parallel to B, independently.
- **Do (spec §4b, §5):**
  - Per-pair discovery across all 10 instruments (a signal real for one pair, dead for
    another, is invisible when pooled). Thin pairs flagged loudly.
  - Interaction discovery: shallow trees (≤3 conditions, ≥150 disc / ≥75 val per leaf),
    XGBoost as a signal-finder (never live), SHAP to read it, with purged CV + PBO/DSR.
  - Better features are allowed as NEW candidates (logged, then screened) — this is where a
    signal we never logged can finally appear.
- **Why it runs regardless of B:** it's an independent read of the same CSV. A null is still
  a result; and this stack has NOT run on clean data with the new columns.
- **Exit:** per-pair tables + interaction candidates with stability + PBO/DSR verdicts, each
  queued for the ship gate. Any pair-elimination candidate is a DISCUSSION POINT needing
  data AND an SMC reason (never data alone).

### STEP D — FREEZE FILTER, RE-TUNE EXITS ON SURVIVORS
- **Entry:** B and C decided; any entry filter frozen.
- **Do:** re-tune exit rules (stop, BE, partial, trail) on the trades that actually survive
  the frozen filter — because a filter changes the excursion profile.
- **Exit:** final exit-rule set with per-split replay deltas on the surviving distribution.

### STEP E — SHIP GATE + ONE-SHOT HOLDOUT
- **Entry:** everything above frozen.
- **Do:** apply the 5-point ship gate (SMC sense, real in discovery, repeats in validation,
  big enough, survives holdout). Open 2022–2025 exactly once.
- **Exit:** shortlist of eligible signals + holdout results, handed to the trader. Trader
  decides ship / don't-ship — never in the same sitting as reading the report (E6).

---

## 5. INHERITED FINDINGS — USE AS HYPOTHESES ONLY (RED FLAG)

These come from the **OLD / corrupted-column run**, before the clean rebuild. They are
**questions to test, not answers to trust.** Every number must be re-verified on the clean
run before it counts. Kept because they point the hunt; deleted the moment the clean run
contradicts them.

- `DISCOVERY_FINDINGS.md` — instant-death mechanics, OB-size barbell, killzone-death,
  50%-entry rejection, the 53%-sweep number. **All pre-fix. Re-verify.**
- `INSTANT_DEATH_DECISION_TREE.md` — the death-geometry decision tree. Hypothesis.
- `BREAKEVEN_SWEEP_HANDOFF.md` — BE-stop sweep track. Hypothesis; needs `sl_wick_depth_atr`
  on clean data.
- `EDGE_LAB_STEP1_HANDOFF.md` — scaffold/timing-wall/schema-guard. **Still structurally
  valid — keep.** Confirm its exit contract was actually signed on the clean CSV (Step 1).
- `EDGE_LAB_STEP2_HANDOFF.md` — pooled univariate "null." **Keep, but its CSV provenance is
  in question** (Step 1 resolves it) and its "selection null" is explicitly NOT settled (§2).

**Rule for all of the above:** a pre-fix number may seed a QUESTION for Steps A–C. It may
never seed a CONCLUSION. Data-vs-SMC and C3 (thin ≠ decision) still apply.

---

## 6. GUARDRAILS THIS PLAN HONORS (so nothing contradicts the frozen rules)

- **C5 (sacred):** all of Steps 1–D run on 2008–2021 only. Holdout opened once, in Step E.
- **Look-ahead wall:** outcome-time columns (SL anatomy) are a descriptive lens and exit
  input only — NEVER an entry selector.
- **C6:** no touch-based prize banked; only walker replay counts.
- **A4/A5:** NAS out; `sweep_present` decreed out of every entry screen.
- **A7:** no OB minimum-size floor re-added without sweep evidence — Step A DESCRIBES the
  cliff, it does not auto-ship a floor.
- **Data-vs-SMC:** any pair elimination or cliff-based cut needs data AND an SMC reason;
  data alone is a discussion point.
- **F-BUCKET (new 2026-07-09):** full per-bucket curve + named best/worst + per-bucket SL
  rate mandatory, enforced in code+test (spec §3, guardrail §F).

---

## 7. STATUS OF THIS HANDOFF

- **Step 0 (guardrail) — DONE 2026-07-09:** spec §3 "Bucket-reporting rule" + DECISION_
  GUARDRAILS §F-BUCKET + change log. Code/test enforcement lands in Step A.
- **CSV-provenance crack — RESOLVED 2026-07-09:** current baseline = `trades.csv` @ commit
  `1feb2db` (git-clean, 11,366 discovery-eligible, all 7 new cols populated); Step-2
  scorecard marked STALE. See §4 Step 1.
- **Next chat:** Step 1 (remaining correctness gate — walker self-check, detector audit,
  new-column stamp-time check, scaffold PASS on `1feb2db`). Then Step A. Nothing analytical
  starts until the gate passes.
- **Not committed** (OneDrive commit-local policy; push only on "ship it").
