# PROMPT 2 — Architecture / Maintainability Review

> Paste this into Fable AFTER uploading `FABLE_REFERENCE_LIVE.md`. Plain English, bullet
> points and headers only in your reply. No praise, no preamble.

---

You are reviewing the **software architecture** of a live H1 SMC trading-alert system. The
attached file `FABLE_REFERENCE_LIVE.md` is verified ground truth, read line-by-line from the
real code. **Reason against that file, not against assumptions.** Where it flags "DOC-DRIFT",
the code wins over the comments.

## Who you're reviewing for
A solo operator running this as two cron jobs (Phase 1 hourly, Phase 2 on its own schedule),
communicating only through JSON files on disk. He wants a system that is **robust, observable,
lean, and not bloated** — and he wants to know if dots are left unconnected that, once
connected, would make tuning easier. Be brutally honest.

## Hard rules
- **Propose nothing as final code.** Output an architecture critique, ranked refactor
  **options with trade-offs**, and failure-mode analysis — a SPEC the engineer brings back.
- The system is **H1-only**. Ignore M5/M15, Phase 3, and the backtest entirely.
- Think like a senior Python architect: "Is this clean, observable, maintainable, lean?"
- If you need a fact the reference doesn't give, **list it as a question** at the end. Do not
  guess. The engineer verifies against code and answers.

## What to evaluate

**1. Module boundaries + single-source-of-truth (reference §2, §4, §5)**
- `dealing_range.py` (structure), `h4_range.py` (walls), `smc_detector.py` (toolbox + Phase 2
  math), `smc_radar.py` (Phase 1 orchestration), `Phase2_Alert_Engine.py` (alerts),
  `news_filter.py`. Are responsibilities cleanly split, or do they leak? Is `smc_detector.py`
  a coherent module or a junk drawer?
- Are the "single source of truth" claims real? (Swing definition, ATR, mitigation, dealing
  range.) Flag every place the same concept is computed in two places.

**2. Duplication / drift risk (reference §3, §5.6, §6)**
- Two ATR implementations (cached `compute_atr` vs uncached `_compute_atr`). Worth unifying?
- Two H1 chart renderers (smc_radar vs Phase2). Consolidate or leave?
- BOS sequence count computed two ways. Risk?
- The DOC-DRIFT cases (mitigation docs vs code). How should the codebase prevent doc/code
  divergence going forward?

**3. The structure engine as a maintainability object (reference §4.4)**
- `compute_structure` is one ~500-line stateful loop recomputed from the full df every scan,
  with load-bearing block ordering. Is "pure recompute from scratch each call" the right design
  (simple, no state corruption) or a liability (slow, untestable, fragile)? Options for making
  it testable WITHOUT changing its output.

**4. State + persistence layer (reference §2, §4.11, §5.1)**
- Communication between Phase 1 and Phase 2 is JSON files (`structure_state.json`,
  `active_obs.json`, `phase2_sent.json`, watch state, email gate). Is this robust? Failure
  modes: partial writes, stale reads, concurrency between the two cron jobs, GC eviction of a
  live zone, schema drift between writer and reader.
- The slate reconcile / drop-reason / "hold if no concrete reason" design. Sound, or a place
  ghosts accumulate?
- The mitigation start-index inconsistency (Phase 1 BOS+1 vs Phase 2 OB+1, reference §5.1) —
  is this an architecture smell (two callers of one function disagreeing on inputs) and how
  should it be structurally prevented?

**5. Observability + failure handling (reference §4.11, §5.5–5.6)**
- Scan logs, heartbeat, stale-data gates, failure logs. Is the system observable enough that a
  silent failure is caught? Where could it fail silently and email nothing / email garbage?
- Many functions "never raise / degrade to a default." Does swallowing errors hide real
  problems? Where should it fail loud instead?

**6. Leanness / bloat (the owner explicitly wants this)**
- What is dead weight? (e.g. PD + killzone computed then scored 0; declared-but-maybe-unwired
  config like `zone_fatigue_threshold` and the news blackout; "TEMP DIAG" scaffolding left in
  `detect_smc_radar`; legacy fallbacks for "old engine" that may no longer be reachable.)
- What can be deleted with zero behaviour change? What's worth keeping but consolidating?

**7. Unconnected dots that would help tuning (the owner explicitly wants this)**
- Where is data computed but thrown away that, if persisted/surfaced, would make the system
  tunable? (e.g. sweep components, drop reasons, score breakdowns, the OB-build diagnostics.)
- What single observability addition would most improve the owner's ability to tune knobs
  with evidence instead of guesswork?

## Output format (strict)
For EACH of the 7 areas:
- **Verdict:** clean / smell / problem — one line.
- **Why:** the reasoning.
- **Options:** 2–3 concrete refactor/cleanup options, each with trade-offs and a rough effort
  estimate (small / medium / large). Rank them.
- **Risk if untouched:** what breaks or rots if left as-is.

Then:
- **Top 5 architecture changes, ranked by (impact ÷ effort)** — best ROI first. Mark which are
  pure cleanup (zero behaviour change) vs which touch behaviour (need owner sign-off).
- **Lean list:** everything safe to delete with zero behaviour change.
- **Questions for the engineer** — facts you need confirmed against code. Be specific.

Brutal honesty. If a module is actually clean, say so — don't manufacture problems.
