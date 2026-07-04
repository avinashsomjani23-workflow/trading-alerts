# SIGNAL CANDIDATES — Structure-Derived Strength/Weakness Tells

**Created 2026-07-04.** A holding pen for mechanical signals the structure engine can
already see (or almost see) that may predict trade quality.

## Rules for this file

- **Nothing here is a shipped signal.** Every item is a hypothesis until Stage 1
  discovery + validation confirm it (DECISION_GUARDRAILS.md: C3, D2, F).
- **No email verdict text ships from this list.** Emails may only state a signal as
  a judgment ("weak leg — caution") after it passes validation. Raw numbers may be
  displayed earlier, judgments may not.
- **Logging first.** A candidate becomes testable the day its measurement lands in
  trades.csv (with a TRUTH_LEDGER.md row + structural guard, per B5). Log now,
  judge later.
- Evidence notes below marked *thin* come from a 2-year in-sample read (2008–09,
  ~2,450 trades). Direction-finding only. Never a basis for a rule.

---

## 1. Leg retracement depth at entry — NEEDS NEW COLUMN

- **Definition:** how much of the displacement leg price has given back when it
  fills the OB. `retrace_pct = (leg_extreme_at_alert − entry) / (leg_extreme_at_alert
  − impulse_start)` for longs; mirrored for shorts. Stamp at alert (leg extreme
  keeps moving after).
- **Vet logic:** strong displacement gets a shallow pullback (~25–50%) and continues.
  A leg retraced ~100% back to its origin is failing — the OB there is the last
  defense, not a spring-board.
- **Evidence so far:** *thin.* 2008–09: "deep discount" longs (≈ deep retrace)
  −0.21R; "expensive" longs (≈ shallow retrace) +0.07R, the only positive bucket.
  Shorts murkier. Matches vet logic for longs; unproven.
- **Status:** not logged. `impulse_start_price` exists on the OB; leg extreme at
  alert needs stamping. Candidate column: `leg_retrace_pct_at_alert`.

## 2. Displacement decay — ALREADY LOGGED

- **Definition:** `bos_verdict` = holding / fading. Each continuation break's
  displacement vs the leg's earlier breaks (smc_detector.bos_leg_read).
- **Vet logic:** shrinking displacement late in a leg = distribution risk.
- **Status:** logged (`bos_verdict`), already shown in the email structure row.
  Validation still pending like everything else — Stage 1 will grade it.

## 3. Leg age (break count) — ALREADY LOGGED

- **Definition:** `bos_sequence_count` = how many plain BOS deep the current leg is.
- **Vet logic:** first pullback of a new leg ≠ fifth. Label-only since the 2026-06-24
  displacement-decay overhaul.
- **Status:** logged. Test in Stage 1 as an interaction with #2, not alone.

## 4. Ranging structure flag — COMPUTED, NOT LOGGED

- **Definition:** structure engine sets `ranging` when the trend is intact but no
  trend-direction swing has extended for STRUCTURE_RANGING_STALE swings
  (dealing_range.py structure v2).
- **Vet logic:** trend labels are stale inside a box; continuation entries in chop
  are the classic bleed. Daily-bias evidence (in-sample) said ranging days were the
  biggest money-loser.
- **Status:** computed every bar, invisible in trades.csv. Candidate column:
  `structure_ranging_at_alert`.

## 5. Pending flip (CHoCH armed, unconfirmed) — COMPUTED, NOT LOGGED

- **Definition:** a CHoCH has fired against the trend but no Confirmation BOS yet
  (`flip_pending` / `choch_pending_dir` in structure v2 state).
- **Vet logic:** two-sided tape. A "with trend" continuation taken while a reversal
  is armed against it is a different trade than one on a clean one-sided trend.
- **Status:** computed, not in trades.csv. Candidate column: `flip_pending_at_alert`
  (+ direction). Wiring into the replay alert payload to verify at implementation.

## 6. Broken-wall PD reading — NEEDS NEW COLUMN

- **Definition:** at OB formation, was the dealing-range floor/ceiling broken and
  riding the live extreme (h4_range `floor_broken` / `ceiling_broken`)?
- **Vet logic:** when a wall rides the live low, every long near the low reads
  "deep discount" by construction — the PD label is unreliable exactly when price
  is running. Separates "PD in a real range" from "PD against a moving wall".
- **Status:** flags exist in the live h4_range dict; whether the frozen
  `ob["dealing_range"]` copy carries them — verify at implementation. Candidate
  column: `dr_wall_broken_at_ob`.

## 7. Break quality — ALREADY LOGGED

- **Definition:** `break_close_atr` / `break_excess` — how hard the BOS/CHoCH candle
  closed through the level vs its ATR floor.
- **Vet logic:** conviction of the break = quality of the zone it left behind.
- **Status:** logged, frozen at detection. Stage 1 grades it.

## 8. Speed of return to the zone — ALREADY LOGGED

- **Definition:** `bars_break_to_pullback` — bars between the break and price
  returning to the OB.
- **Vet logic:** ambiguous by itself (instant return can be a failed break; slow
  return can be a fresh untouched zone or a stale one). Interaction candidate with
  freshness/touches, not a standalone signal.
- **Status:** logged. No hypothesis strong enough to pre-register alone.

## 9. Sweep quality — BLOCKED (A5)

- Suspected inverted; excluded from every screen until the sweep rebuild lands.
  Listed here only so nobody re-adds it by accident.

---

## Change log

- 2026-07-04 — file created after the trend/dealing-range investigation (with_trend
  label ≈ continuation-vs-reversal; local H4 range makes PD ≈ retracement depth).
- 2026-07-04 — items 1, 4, 5, 6 spec'd for execution in `STRUCTURE_SIGNALS_SPEC.md`
  (verification done: trend/flip/leg-verdict sound; ranging counter had a
  flip-carryover defect, fix = spec item S1).
