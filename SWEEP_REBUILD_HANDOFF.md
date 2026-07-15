# SWEEP REBUILD — Handoff Prompt

Paste this file into a NEW chat (Fable recommended). It is self-contained:
what exists, what is suspected wrong, what just landed that changes the
design, and the rules the rebuild must respect.

---

## Prompt

You are rebuilding the SWEEP detector of an SMC trading-alert system.
Read this whole file, then read the live code before proposing anything —
code is truth, this document is only the map.

### Rules that bind you (non-negotiable)

1. CLAUDE.md governs: code is truth (quote file:line for every claim),
   no code changes without explicit approval, bullets not paragraphs.
2. DECISION_GUARDRAILS.md governs backtest/engine decisions — flag rule IDs
   before helping if a step would break one.
3. Truth-ledger gate: every new trades.csv column needs a TRUTH_LEDGER.md
   row + a guard test that kills its bug class. Mutable state must be
   stamped `*_at_alert` at the yield, never read at row-build time.
4. Observe first, gate never (this phase): the rebuilt sweep LOGS columns;
   no score/filter changes until the edge engine judges them on a canonical
   run.
5. No 18-yr backtest on your own judgment. Validate on frozen cached
   windows (`backtest/cache/*.parquet`) and smart samples first.

### What exists today (verified 2026-07-14)

- Detector: `smc_detector.observe_phase1_sweep` (smc_detector.py:1158) —
  snapshot at OB formation, consumed by Phase 2; it does NOT re-grade.
- Targets: ACTIVE lookback-3 swings — unbroken AND unswept via
  `is_swing_active` (smc_detector.py:833). Any qualifying swing counts;
  there is NO ranking of targets by liquidity importance.
- Window: `[impulse_start − 3, ob_idx]`, hard-floored at the prior
  structural event (smc_detector.py:1286-1304; constants :312).
- Score (max 3.0, constants smc_detector.py:361-363):
  presence 1.5 + equal-levels 0/0.25/0.5 (`_equal_levels_score` :885,
  tolerance 0.30 ATR — note: WIDER than the new EQ layer's 0.10) +
  rejection wick:body 0/0.33/0.66/1.0 (`_rejection_score` :951).
- Context tags (`_compute_context_tags` :1113): round numbers, prior-day
  H/L, session H/L — tags only, they do NOT drive target selection.
- Survivorship: a candidate is rejected if any later candle in the leg
  wicks strictly deeper (smc_detector.py:1351-1374).
- trades.csv: `sweep_pts` / `sweep_present` are marked out-of-scope in
  TRUTH_LEDGER.md, owned by THIS rebuild.

### Why it is suspected wrong (hypotheses to verify, not facts)

- Prior data pass suggested sweep score may be INVERSE-correlated with
  outcome (unverified — re-verify on the current canonical CSV named in
  backtest/results/CANONICAL.md before believing it).
- The detector rewards any wick+close-back off ANY minor lookback-3 swing.
  SMC-wise most of those swings hold no meaningful liquidity — the signal
  the score was meant to capture (a real stop-run) is diluted by noise
  "sweeps" of irrelevant levels.
- No follow-through requirement: nothing checks that displacement AWAY
  from the sweep actually followed.
- Measurements that already exist for the exit side (sl_bar_was_sweep,
  sl_wick_depth_atr, sl_max_adverse_after_sweep_atr) are OUTCOME-time;
  do not confuse them with the entry-time sweep feature.

### What just landed that changes the design (2026-07-14)

`eq_pools.py` — EQH/EQL equal-level clusters with lifecycle:
- Cluster = ≥2 confirmed raw-geometry lookback-3 swings within
  0.20 ATR of the running extreme (raw swing pool via
  `dealing_range.detect_swings(min_leg_atr_mult=None)`; approved for the
  sweep/EQ use-case ONLY; 0.20 chosen for live-vs-backtest feed-noise
  robustness — see eq_pools.py EQ_TOL_ATR comment).
- Level = the member extreme (where the stops actually sit).
- Lifecycle = pool_builder.pool_status — the SAME status machine as the
  PD/PW pools (intact / swept / broken, N=1 confirm, failed break = swept).
- Guards: tests/test_eq_pools.py (no-look-ahead, determinism, lifecycle).
- 11 trades.csv columns already wired (see TRUTH_LEDGER.md EQ rows).

Together with pool_builder.py (PDH/PDL/PWH/PWL) the system now has a
RANKED LIQUIDITY MAP the old sweep never had:
  PWH/PWL > PDH/PDL > EQ clusters (≥3 touches > pairs) > bare lookback-3
  swings (lowest tier — arguably not liquidity at all).

### The rebuild thesis to design against

A sweep is only meaningful when it takes REAL liquidity and price REJECTS
it with follow-through. Proposed shape (challenge it if the data disagrees):
1. TARGET: the swept level must be a ranked pool (PW / PD / EQ cluster),
   not a bare minor swing. Tier of the swept pool is a logged feature.
2. EVENT: wick through the pool level + close back (the pool_status
   'swept' event IS this — one implementation, reuse it; do not write a
   second sweep judge).
3. REJECTION: keep the wick:body grading (it is sound) but log raw, don't
   cap at tiers.
4. FOLLOW-THROUGH: displacement away from the sweep within N bars
   (define N from data; log, don't gate).
5. RELEVANCE TO THE TRADE: sweep side must be the trade's fuel side
   (bullish trade ← low-side sweep), and recency in bars is a feature.

### Deliverables (complete build and fix — BOTH paths, owner requirement)

1. A verification pass FIRST: on the canonical CSV, test the
   inverse-correlation suspicion of the CURRENT sweep_pts. If it is noise,
   say so; if inverse, quantify.
2. Design doc (short) mapping new columns → ledger rows → guards.
3. Implementation behind the observe-only discipline, columns stamped
   at-alert, never re-graded post-alert.
4. BACKTEST logging: new sweep columns spread into the _build_row row dict
   (follow the _eq_features_at_alert precedent at the end of
   backtest/h1_only_simulator.py) + h1_only_reporting.py front_cols +
   TRUTH_LEDGER.md rows. The old sweep_pts / sweep_present rows in the
   ledger are yours to supersede — mark them, don't silently drop them.
5. LIVE logging: Phase 1 scan record (smc_radar._build_phase1_scan_record —
   'pools' / 'eq_clusters' keys are the precedent) + the P1 sweep badge and
   P2 email narration updated to the rebuilt definition (plain English,
   short; the owner's emails are already crowded — replace the old sweep
   text, don't add alongside it). Phase 2 consumes the P1 snapshot; keep
   that snapshot-not-regrade contract.
6. Tests: no-look-ahead, determinism, both directions, degraded inputs —
   mirror tests/test_eq_pools.py's structure. Full suite must stay green.
7. Sample-window validation on cached parquet BEFORE asking the owner
   about any full run.

### Round-number levels — fold into the pool RANKING, add a feed buffer

Round numbers are ALREADY half-built — do not build a new module. They exist
today only as a boolean context TAG, not a ranked target:
- Grid + tolerance constants: smc_detector.py:356 (ROUND_NUMBER_GRID) and
  :363 (ROUND_NUMBER_TOLERANCE). FX grid = 0.0050 (50 pips), FX tol = 0.0005
  (5 pips); JPY 0.50 / 0.05; gold 5.0 / 0.50; index 50 / 5.
- Tagged in _compute_context_tags (smc_detector.py:1164-1170): if the swept
  price is within tol of the nearest grid line, append the 'round_number' tag.
  That is the whole current treatment — a flag, nothing ranks on it.

What the rebuild must do with them:
1. PROMOTE round numbers to a first-class ranked pool tier, sitting near the
   TOP of the target hierarchy. Practitioner consensus (and the owner's
   research read) is that on FX a sweep only reliably holds when the swept
   level IS or SITS ON a round number — bare equal highs mid-range are noise.
   So a swept pool that is ALSO a round number should rank ABOVE a plain
   PD/PW pool of the same type. Proposed order:
     round-number-aligned pool > PWH/PWL > PDH/PDL > EQ cluster > bare swing.
   Log the alignment as a feature (e.g. rn_aligned bool + rn_dist_atr), do
   NOT gate on it — the run judges whether "round number holds, others don't"
   is real in OUR data.

2. ADD A FEED BUFFER to the round-number tolerance. The 5-pip FX tolerance
   was set for MT5 backtest data. Live runs on Twelve Data, and the logged
   MT5-vs-TD gap is ~1 pip p50 but 5-12 pips at p95 (memory:
   project_oanda_twelvedata_eval / feed_hole_diagnosis). At 5 pips a genuine
   round-number touch on the live feed can land 6-10 pips off the MT5 grid
   line and be MISSED — the level never gets detected, so the room to detect
   it is the problem, not the sweep logic. Widen the tolerance so a
   round-number level has room to register across BOTH feeds. Suggested
   starting point (challenge with data): FX 5 -> 8-10 pips, JPY/gold/index
   scaled to match the same feed-noise fraction. Keep it a NAMED constant
   with a comment citing the feed-gap reason, mirroring EQ_TOL_ATR=0.20's
   rationale, so the next reader sees WHY it is looser than a chart eyeball
   would suggest. Same buffer logic the EQ layer used when it went 0.10->0.20.

3. Do NOT touch the round-number constants in THIS (EQ) build — this is a
   note for the sweep rebuild ONLY. The owner asked for it parked here, added
   smartly, not done now.

### Files to read first (in order)

1. TRUTH_LEDGER.md (EQ rows + sweep out-of-scope markers)
2. eq_pools.py, pool_builder.py (the liquidity map + THE status machine)
3. smc_detector.py:833-1450 (current sweep, scorers, context tags)
4. backtest/results/CANONICAL.md (the one CSV)
5. DECISION_GUARDRAILS.md
