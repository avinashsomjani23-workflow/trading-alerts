# Sweep v2 — Logging fixes + Live-pool / freeze rework (HANDOFF)

**Written:** 2026-07-25. **Status:** agreed in discussion, NOTHING coded yet.
**Read `CLAUDE.md` first.** Code is truth — quote `file:line` before stating any
behaviour. This doc is background, not truth; if it disagrees with live code,
code wins.

## ⚠️ Working-tree state (verify before touching anything)
- Legacy sweep (`observe_phase1_sweep`) is **being removed** in the current
  working tree: `smc_detector.py` ~−642 lines, the `observe_phase1_sweep` call
  + `sweep_observed` stamp deleted from `smc_radar.py`, and
  `backtest/h1_only_simulator.py:1848` now reads `ob['sweep_v2']` for
  `sweep_present`. `_rejection_score` / `is_swing_active` SURVIVE (v2 uses them).
  → **Confirm this is committed/coherent before building on it.** Everything
  below lives in the v2 detector (`liquidity_sweep.py`); the "SW tier" IS the
  retired legacy sweep, folded in.

## ⛔ Discipline for whoever picks this up
- **Do NOT go on a rampage run.** No 18-yr backtest. No full baseline "to see".
  Validate on a SMALL cached slice (a few pairs / a few months) — CLAUDE.md
  rule 5. Frozen `backtest/cache/*.parquet` first, no feed pull.
- Be mindful and token-smart: read the exact `file:line` anchors here, don't
  re-scan the whole repo. Accuracy over volume.
- Nothing here GATES/scores/ranks trades — all observation-only (guardrail A5).
- Trading-logic / architecture changes need explicit owner approval before code.

---

## Anchors (read these, not the whole file)
- Detector: `liquidity_sweep.observe_pool_sweep` — `liquidity_sweep.py:353`.
- Winner rank `PW>PD>EQ>SW`, tie-break deepest pierce — `liquidity_sweep.py:465-468`.
- Per-candidate pierce + rn stamped — `liquidity_sweep.py:442-460`.
- Rejection (winner only today) — `smc_detector._rejection_score:1006`.
- Follow-through (winner only today) — `liquidity_sweep.py:488-494`.
- Age (alert-anchored today) — `liquidity_sweep.py:560-573`.
- Column list — `liquidity_sweep.SWEEP2_FEATURE_COLUMNS:116`.
- Row-build entry — `features_from_snapshot:529`; wired at
  `backtest/h1_only_simulator.py:1941` (`_sweep2_features`).
- Freeze contract (WHY the snapshot is formation-frozen) —
  `liquidity_sweep.py:40-58` + docstring lines 44-47 (kills the re-grade bug).
- SW always provable; PW/PD have a provability gate — `liquidity_sweep.py:49-58, 427`.
- TRUTH_LEDGER gate for any new column — `CLAUDE.md` "Logging" + `TRUTH_LEDGER.md`.

---

## Data grounding (already measured this chat — don't re-run to re-confirm)
- 6,099 trade rows across runs with sweep2 cols. Of 822 real sweeps:
  1 pool 72%, 2 pools 22%, 3 pools 5%, 4-5 <1% → **28% hit 2+ levels** (a deep
  sweep runs stacked levels). Multi-tier is normal, not an edge case.
- Winner tier: PD 597, SW 105, EQ 98, PW 22. SW wins only 13% (ranked last, so
  only when it's the SOLE sweep) → SW is often **present but hidden**. This is
  why SW gets its own always-on columns.
- Legacy pierce floor `0.05×ATR` ≈ 0.8 pips on EURUSD (real MT5 ATR). Owner
  decision: **KEEP the floor as-is. A 1-pip sweep is still a sweep. No floor
  change.** (Do not reopen.)

---

## WORKSTREAM 1 — Logging (small, self-contained, do FIRST)

**1a. SW-tier always-on columns (+5).** Compute pierce / rejection / follow /
rn for the best SW candidate **even when a ranked pool wins**, freeze on the
snapshot. New columns:
- `sweep2_sw_present`, `sweep2_sw_pierce_atr`, `sweep2_sw_rejection_ratio`,
  `sweep2_sw_follow_atr`, `sweep2_sw_rn_aligned`.
- SW-collapse when >1 SW swept: use **earliest sweep** (chronological fact,
  matches the "first touch = the sweep" rule; NOT a quality judgment).
- Winner columns already exist (`sweep2_pierce_atr` etc.) → winner gets NO new
  columns. When SW itself wins, SW cols == winner cols (equal, honest).
- rn stays a pure **flag**, never enters ranking/score (already removed from
  `_rank_key`, `liquidity_sweep.py:105-111`) — keep it inert.

**1b. Age → fill-anchored.** Rename `sweep2_age_at_alert_h1` →
`sweep2_age_at_fill_h1`. Arithmetic on the FROZEN `sweep_ts` vs the FILL bar
(sweep_ts stays frozen — only the measuring anchor moves). `never_filled` →
None. Precedent: pool / EQ / approach columns are all fill-anchored. Owner
explicitly approved the re-anchor (clears CLAUDE.md rule 5b).

**1c. Plumbing.** Update `SWEEP2_FEATURE_COLUMNS`, expand `features_from_snapshot`,
add TRUTH_LEDGER rows for all 6 changed/new columns (source file:line, when
stamped, population), add ONE guard test in `tests/test_liquidity_sweep.py`
(SW metrics present when a pool wins; fill-age arithmetic correct). Not a
detection change → winner/score/email byte-identical → rides next run.

**Rejected (do NOT build):** full 20-col per-tier set. 15 cols empty 72% of the
time = bloat. Winner+SW covers the real signal. Only loss: two ranked pools
(e.g. PD+PW) swept in one short leg — rare (PW=22 total). Revisit only if data
demands it.

---

## WORKSTREAM 2 — Live pools + closed-candle freeze (BIGGER — new chat, own approval)

This is an ARCHITECTURE change to the freeze contract. It re-opens the exact
re-grade bug class the freeze was built to kill (`liquidity_sweep.py:44-47`) —
handle with care, not in the same sitting as WS1.

**The frozen-vs-live cut the owner wants:**
- **SW — frozen** (OB's own leg liquidity; frozen truth; never changes).
- **EQ — frozen** (owner: "EQ can be stamped").
- **PW / PD — detected LIVE every scan**, NOT stamped. Reason: "yesterday's low"
  / "last week's low" roll as days/weeks pass; a formation-frozen PW/PD is stale
  by alert time.

**The bigger problem than rolling (owner, this chat):**
- **A sweep can UN-happen.** A wick-through-and-reject at 11:00 is a sweep; if
  price returns and closes back through by 14:00, the read has changed. Freezing
  at formation locks a picture later price action can invalidate. Live pools must
  re-judge validity as price evolves — a validity-over-time problem, separate
  from (and on top of) the rolling-level problem.

**Closed-candle rule (applies to the frozen tiers too):**
- SW and EQ must be stamped **only after a candle CLOSES**, never off a forming
  bar. A mid-bar wick/close can still move; a "sweep" seen on a forming candle
  may not survive the close. → verify the detection anchor is a closed bar in
  both live scan and replay; fix if it stamps a forming bar.

**Open design questions for WS2 (decide in the new chat, with code loaded):**
- How to recompute PW/PD live without the 150-bar rolled frame silently
  mis-measuring (the honesty-bounds/provability gate, `liquidity_sweep.py:49-58`).
- Snapshot split: which fields freeze (SW/EQ) vs recompute-per-scan (PW/PD), and
  how the winner/`sweep2_tier`/score leg read a half-frozen/half-live snapshot
  without double-counting or look-ahead.
- Does the winner get "decided fresh" each scan once PW/PD are live? Consequence
  for `score_inputs` (`liquidity_sweep.py:606`) and email narration.
- Backtest parity: replay must see the same closed-bar, same live-pool picture
  live would.

---

## Suggested order
1. WS1 (logging) — land clean, own run, low risk.
2. WS2 (live pools + closed-candle) — fresh chat, fresh approval, design-first.
Do NOT bundle them.
