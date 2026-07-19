# SWING-SWEEP SUITE — Build Spec

Single source of truth for the swing-based sweep build. Written 2026-07-19.
Supersedes and REPLACES the three deleted sweep handoffs (SWEEP_V2_DESIGN.md,
SWEEP_SUITE_HANDOFF.md, SWEEP_REBUILD_HANDOFF.md) — those are gone on purpose so
there is no stale trap. Read this file + the live code it points at. Code is
truth (CLAUDE.md rule 1): quote `file:line` before stating any behaviour.

---

## 0. Owner intent in one paragraph

The system already tracks sweeps of the OB's *fuel leg* against ranked pools
(sweep v2: PD / PW / EQ). What it does NOT track is the liquidity picture around
THIS trade's own **stop** and **target**, built from **swing points** (not tied
to PD/PW/EQ). The owner wants three observe-only reads, frozen at OB build, that
answer: (1) is there sweepable liquidity where my STOP sits? (2) is my TARGET
sitting on unspent liquidity that will pull price toward it? (3) was the OB
leg's own extreme itself a sweep of an active swing? All swing reads obey ONE
non-negotiable rule: **only ACTIVE (unbroken AND unspent) swings count** — a
swing already wicked or broken holds no liquidity.

---

## 1. What already exists — do NOT rebuild (with dates, verified via git)

| Thing | File | Built | What it does |
|---|---|---|---|
| PD/PW levels + status | `pool_builder.py` | 2026-07-13 | yesterday/last-week H-L, `pool_status` intact/swept |
| EQ shelves + status | `eq_pools.py` | 2026-07-15 | equal-highs/lows clusters, intact/swept |
| **Sweep v2 (fuel-side pool raid)** | `liquidity_sweep.py` | **2026-07-19 (Fable)** | did the OB LEG raid a PD/PW/EQ pool → `sweep2_*` cols, P2 banner, chart |
| Legacy swing sweep | `smc_detector.observe_phase1_sweep` | oldest | sweeps of lookback-3 swings; **feeds the score** (see §6) |
| Active-swing filter | `smc_detector.is_swing_active` | — | unbroken AND unswept check — THE tool we reuse for the unspent rule |
| Swing source | `dealing_range.detect_swings` | — | one swing definition; already computes leg-ATR internally (§5 H4 proxy) |
| One sweep judge | `pool_builder.pool_status` | — | wick+close-back = swept; the only sweep definition, reused everywhere |
| SL/TP at alert | `Phase2_Alert_Engine.py:900,1190` | — | `levels.get('sl')` / tp known at ALERT (not fill) — confirmed |
| Outcome-time SL sweep | `backtest/h1_only_simulator.py` (`sl_bar_was_sweep`…) | — | AFTER stop hit; the outcome side, NOT rebuilt |

**Reuse, never reinvent:** `pool_status` (one judge), `detect_swings` (one swing
source), `is_swing_active` (the unspent filter). No new detector, no second
sweep definition. (Owner approved reuse.)

---

## 2. Timing / anchoring — LOCKED (owner confirmed)

- Snapshot computed **ONCE at OB build** inside `detect_smc_radar`, stamped on
  the zone, **formation-frozen** — never re-graded (mirror `sweep_v2` exactly:
  `liquidity_sweep.py:34-41`; live `Zone.refresh` back-fills once, never
  re-stamps; backtest replay merge refreshes only `fvg`).
- The **SL and TP are known at ALERT** (`levels.get('sl')`,
  `Phase2_Alert_Engine.py:900,1190`) — NOT fill. So the stop-band / tp-band
  reads have their anchor available. (Owner: "SL is calculated during alert,
  not fill" — confirmed in code.)
- Live vs backtest differ ONLY in when the frozen snapshot is READ into a row:
  live reads at alert, backtest reads at fill. The sweep JUDGMENT itself is the
  same frozen OB-build value in both. (This is the point that caused confusion
  earlier — stated plainly here so it never recurs.)

---

## 3. The three reads

All entry-time, frozen at OB build, observe-only. Long-side wording; short-side
is the mirror (swing lows↔highs, buy-side↔sell-side, above↔below flip).

### Read 1 — STOP-side liquidity (the strongest lead)

*Is there an ACTIVE swing that could be swept in the region of my stop?*

- Side: for a LONG, we hunt **sell-side liquidity** = active swing **lows**
  (mirror: shorts → active swing highs).
- **Band = SL ± 0.5 ATR (H1)** — symmetric, NOT "below only" (rationale §4).
- A swing qualifies only if `is_swing_active` at OB build (unbroken + unspent).
- Log the **signed offset** of the swing from the SL. This is the core signal:
  - offset **negative** (swing below SL): a sweep must blow through your stop,
    then recover — the "stopped out, then it ran without me" case.
  - offset **≈ 0** (swing at SL): exact stop-hunt hit.
  - offset **positive** (swing between entry and SL, inside risk): sweep +
    reverse happens BEFORE your stop is hit — the **survive-the-hunt** case,
    the one that makes money.

### Read 2 — TP-side magnet (Draw on Liquidity)

*Is my target sitting on UNSPENT liquidity that will pull price toward it?*

- Polarity (owner-confirmed, both statements true, same coin):
  - TP on **unspent** liquidity = **positive** (fresh fuel, price is drawn to
    it — Draw on Liquidity).
  - TP on **spent** liquidity = **negative** (money gone, no magnet, price may
    turn against us).
- Side: for a LONG, the target sits above → look for active swing **highs**
  (unspent buy-side liquidity) near the TP.
- **Band = TP ± 0.5 ATR (H1)** — REUSES the stop-side band logic exactly (owner
  agreed: one helper, two anchors). Signed offset logged.
- Web basis: Draw on Liquidity — price gravitates to the nearest meaningful
  UNSPENT pool (old highs/lows, equal highs/lows). Sources:
  OpoFinance blog "ICT Draw on Liquidity"; TradingFinder "Draw on Liquidity
  (DOL) in ICT". FX included. (No prior-run insight used here — owner's call.)

### Read 3 — Leg-extreme sweep

*Was the OB leg's OWN terminal extreme built by sweeping an active swing?*

- The move that made this OB — did it END by grabbing fresh liquidity at an
  active swing, then reverse? Institutional-intent tell on the setup itself.
- Reuses the exact leg window sweep v2 / legacy already lock
  (`[impulse_start-3, ob_idx]`, floored at prior structural event).
- `is_swing_active` mandatory here too.

---

## 4. The SL/TP band — rationale (feed-variance-aware) — LOCKED

**Band = anchor ± 0.5 ATR (H1), symmetric, signed offset logged.**

Why a **band**, not a line:
- A swing rarely sits exactly on the SL/TP price.
- The SL price differs slightly between MT5 (backtest) and Twelve Data (live):
  logged gap p50 ~1 pip, **p95 5–12 pips, unpredictable** (memory:
  project_oanda_twelvedata_eval). A zero-width check flags a swing on MT5 but
  misses the same swing live. Same reason sweep v2 buffered its RN tolerance.

Why **symmetric** (not below-only): all three offset positions are signal
(below = stopped-then-runs, at = exact hit, above = survive-the-hunt). Owner
corrected an earlier "below only" framing — a symmetric band is required or we
blind ourselves to the survive-the-hunt case (the profitable one).

Why **0.5 ATR** specifically:
- **Floor:** must comfortably exceed the p95 feed gap (5–12 pips). On EURUSD
  0.5 ATR ≈ 25–40 pips — safely above 12 pips, so feed variance never flips a
  flag.
- **Ceiling:** must stay LOCAL to the anchor. Wider than ~0.5 ATR starts
  catching swings that belong to the other end of the trade (double-counting).
  0.5 ATR keeps each read in its own neighbourhood.
- ATR = the frozen formation `ob['h1_atr']` (the shared `*_atr` denominator).

---

## 5. The "H4 sweep" question — an H1 BREAK that is really an H4 SWEEP

**BUILDER MUST DO THIS FIRST: explain this idea back to the owner in the chat in
the very simplest English, using the picture below, BEFORE writing any code.
Confirm the owner agrees. Then record the A/B decision at the bottom of this
section.**

### The insight (owner's, corrected from an earlier wrong "leg-size" framing)

The earlier draft of this spec proposed grading swings by LEG SIZE (big move =
"H4-grade"). **That was wrong and is discarded** — it answers a different
question and misses the real case. The real case is:

> **An H1 BREAK can be an H4 SWEEP.**
>
> On H1, price CLOSES through a swing low → H1 calls it a genuine break.
> But an H4 candle is just 4 H1 candles glued together. If a LATER H1 candle in
> that same H4 block pulls price back ABOVE the level, the merged H4 candle only
> **wicked** below and **closed back above** — on H4 that same move is a
> **sweep (stop-run), not a break.**
>
> Example:
> ```
> H1:  price closes below the swing low 1.0950   → H1 says "broken"
> H4:  the 4 H1 bars merge into ONE candle that dipped to 1.0948 but
>      CLOSED at 1.0970 (back above 1.0950)       → H4 says "swept, not broken"
> ```
> What H1 calls a break, H4 calls a stop-run. The close-through gets "erased"
> when the candles merge because price was reclaimed shortly after.

### Why this matters for us

Our active-swing filter (`is_swing_active`, `smc_detector.py:948`) currently
KILLS a swing the moment an H1 candle CLOSES through it (treats it as broken →
dead liquidity). But if that close-through is **reclaimed shortly after**, it
was never a real break — it was a **sweep**, and the level is still live,
grabbable liquidity. Today we wrongly discard it.

### The cheap fix — NO H4 candles, NO resample, NO clock-bug risk

An H4 candle only tells us one thing that matters here: *"where did price close a
few H1 bars later?"* We can ask that directly on the H1 data we already have:

> After an H1 close-through of the level, did price **close back on the original
> side within the next few H1 bars**?
> - YES → the break was reclaimed → it was a **SWEEP**, keep the level alive.
> - NO  → it stayed through → a **real break**, level is dead (current behaviour).

That reclaim check reproduces the H4 close verdict with pure H1 bars. No second
timeframe is built.

### Where it lands in the build

- Extend the unspent/active judgement so a **reclaimed** close-through counts as
  a SWEEP of a still-active swing, not a break that kills it. This sharpens
  Read 1 (stop-side), Read 2 (tp-side), AND Read 3 (leg-extreme) — all of them
  rely on "is this swing still live."
- Reuse `pool_status` semantics where possible (it already treats
  wick+close-back as swept); the NEW piece is the *reclaim-window* check applied
  to a close-through inside `is_swing_active`'s logic (or a wrapper) — do NOT
  fork a second sweep judge.

### OPEN DECISION for owner (record before Stage 2) — the reclaim window

Two ways to define "shortly after":
- **(A)** Tie it to the true H4 boundary — reclaim counts only if price closes
  back before the current H4 candle closes. Exactly matches a real H4 chart, but
  ties us to H4 clock boundaries → mild MT5 clock-era exposure at the boundary.
- **(B)** A fixed **next-N-H1-bars** reclaim window (e.g. next 3 H1 bars), no H4
  boundary at all — zero clock exposure, simpler; not pinned to a real H4 grid
  edge, but "did the break hold or get reclaimed" is the true signal, not the
  grid line.

Builder's lean (and owner leaned the same in chat): **B** — captures the insight
with no clock risk. Confirm N (default 3) and A-vs-B here. → __________

---

## 6. Columns — COMPACT (6, owner asked to avoid bloat) + LEDGER

Prefix `setup_sweep_` so nothing collides with `sweep2_*` (fuel) or `sl_*`
(outcome). One struct per anchor; six columns carry every read.

| column | meaning |
|---|---|
| `setup_sweep_stop_present` | active swing (unspent) exists in the SL ± 0.5 ATR band |
| `setup_sweep_stop_offset_atr` | signed dist of that swing from SL (− below / 0 at / + inside-risk) — Read 1 core signal |
| `setup_sweep_stop_grade` | pool coincidence of the swing: bare / EQ / PD / PW (bigger pool = more meaningful stop-hunt). NOTE: the "H4 sweep" idea is NOT a grade here — it is the reclaim-window fix to the active-swing judgement in §5, applied to all three reads, not a separate column |
| `setup_sweep_tp_present` | active unspent swing in the TP ± 0.5 ATR band (the magnet) |
| `setup_sweep_tp_offset_atr` | signed dist of that swing from TP |
| `setup_sweep_legextreme_swept` | the OB leg's own extreme was an active-swing sweep (Read 3) |

- Depth/age can ride INSIDE the frozen snapshot struct if needed for narration;
  these six are the analysis levers. Do not spawn per-metric columns unless a
  reviewer shows a lever is lost.
- **Truth-ledger gate (CLAUDE.md):** every column needs a `TRUTH_LEDGER.md` row
  (source `file:line`, when stamped = OB build, population) before any run feeds
  it. `EDGE_ENGINE_SPEC.md §12` disposition row too.

---

## 7. Legacy replacement — LOAD-BEARING, owner said "connect it" (READ CAREFULLY)

Owner instruction: *"Whatever is load-bearing should be connected to this new
sweep. No dead code."*

**What is load-bearing (verified):** `run_scorecard` turns `ob['sweep_observed']`
(the legacy swing sweep) into ACTUAL score points —
`smc_detector.py:2648-2651`: non-JPY FX gets `bd["sweep"] = 1.0` if a sweep
exists else 0.0; JPY/Gold get a scaled quality grade. That score feeds the
confidence score that GATES live alerts.

**Therefore, rewiring the score to the new sweep is a LIVE-BEHAVIOUR CHANGE, not
a cleanup:** the new sweep judges differently → FX confidence scores shift by up
to a point → some trades that fire today won't, and vice-versa. This MUST be
flagged and approved with eyes open, not done silently (CLAUDE.md: no silent
P&L moves).

**Plan (Stage 2):**
1. Rewire the scoring input at `smc_detector.py:2648-2651` to read the NEW swing
   sweep snapshot instead of `ob['sweep_observed']`. Preserve the same shape the
   scorecard expects (presence for non-JPY FX; a quality grade for JPY/Gold).
2. Remove the legacy `observe_phase1_sweep` + `is_swing_active` ONLY if the new
   file fully absorbs their role; otherwise KEEP `is_swing_active` (we reuse it)
   and retire only the legacy observer.
3. Remove genuinely-dead orphans (sweep v2 already listed some:
   `_render_sweep_observation_html`, dead params). Confirm zero callers via grep
   before each deletion.
4. Full test suite + structure-golden must stay green (structure changes flip CI
   red by design — re-baseline via the hook, memory: project_structure_golden_gate).

**OPEN DECISION for owner (record before Stage 2):** rewire scoring to new sweep
NOW (accept the live-alert shift) — or keep legacy scoring parallel and ship the
new sweep observe-only first, migrate scoring later? Owner leaned "connect it";
confirm they accept the live shift. → __________

---

## 8. Stages (owner's model — Stage 3 dropped)

- **Stage 0** — Smart-sampled measurement to confirm the effect is worth
  building (cached parquet, NO 18-yr run). **OPEN: owner may skip this and judge
  on their own 18-yr run — confirm.** → __________
- **Stage 1** — This spec. Done on approval.
- **Stage 2** — Build `swing_sweep.py` (mirror `liquidity_sweep.py` structure:
  all-None fallback, never raises, does NOT evict the eq_pools frame cache,
  `pool_status` the one judge, `is_swing_active` unspent filter). Rewire scoring
  (§7). Spread the 6 columns LAST in `_build_row` after the sweep2 spread.
  TRUTH_LEDGER rows + EDGE_ENGINE §12. Tests mirror
  `tests/test_liquidity_sweep.py` (no-look-ahead, determinism, both directions,
  degraded inputs, unspent-swing-excluded, stop-side / tp-side / leg-extreme).
  Real-data audit on random cached-parquet windows (reuse the sweep v2 harness:
  drive `detect_smc_radar`, re-audit every claim independently).
- **Stage 3 — DROPPED.** Owner runs the cross-table + luck test themselves
  across pairs over 18 years and decides on any gate.

---

## 9. Hard rules (CLAUDE.md governs)

- Code is truth — quote `file:line`; no code change without approval.
- Observe-only this phase EXCEPT the approved §7 scoring rewire.
- No 18-yr run on the builder's judgement — smart-sample cached parquet only.
- Truth-ledger gate: column → ledger row + class-killing guard before any run.
- Guards live OUT of the live alert path (offline tests / CI), never a raise
  inside OB build or the row build — the sweep layer must never kill a build
  (mirror `liquidity_sweep.py`'s try/except → `snapshot_failed()`).
- Bullets not paragraphs, plain English, no sycophancy.

---

## 10. Things the builder must NOT assume (uncertainty flagged honestly)

This chat was long; these are the points to VERIFY in code before relying on
them, not carry from memory:

1. Exact call site + signature of `observe_pool_sweep` inside `detect_smc_radar`
   — mirror it for the new observer's call site (read it, don't assume).
2. Exact shape `run_scorecard` expects from the sweep snapshot before rewiring
   (`smc_detector.py:2607-2651`) — preserve every field it reads.
3. Whether removing `observe_phase1_sweep` orphans anything beyond the scorecard
   (charts, P2 narration, zone-map digest) — grep all callers first.
4. The reclaim-window fix (§5): read `is_swing_active`
   (`smc_detector.py:948-997`) — it kills a swing on ANY close-through. The new
   behaviour lets a close-through that is RECLAIMED within N H1 bars count as a
   SWEEP (level stays active), not a break. Confirm whether to extend
   `is_swing_active` in place or wrap it; do NOT fork a second sweep judge, and
   do NOT evict the eq_pools per-frame cache with a second swing pass.
5. Front-cols / `_build_row` spread location — put the 6 columns AFTER the
   sweep2 spread, same pattern (`backtest/h1_only_simulator.py`).

---

## 11. Files to read first (in order)

1. `liquidity_sweep.py` (direct template — structure, freeze, fallback, cache
   discipline).
2. `smc_detector.py`: `is_swing_active` (:948), `observe_phase1_sweep` (:1273),
   `run_scorecard` sweep block (:2607-2651).
3. `pool_builder.py` `pool_status` (:297); `dealing_range.py` `detect_swings`
   (:339); `eq_pools.py` (EQ/`eq_sl_at_risk` stop-vs-liquidity precedent).
4. `backtest/h1_only_simulator.py`: `_build_row` (:1309), `_sweep2_features`
   (:1866), the `sl_bar_was_sweep` family (outcome side — do NOT rebuild).
5. `backtest/results/CANONICAL.md`, `DECISION_GUARDRAILS.md`, `TRUTH_LEDGER.md`.
