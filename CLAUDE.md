# Trading Alerts System — Project Guide for Claude

Automated SMC alert system. Goal: replicate veteran SMC judgment, not generate noise.

---

## NON-NEGOTIABLE — READ EVERY TURN

Four rules. Violating any one means the response is wrong — rewrite before sending.

**1. Code is truth.**
- Before stating ANY column meaning, detection behaviour, or trading logic, quote the live `file:line` in the SAME response. No code quote = do not state it.
- Column meanings: read the `file:line` pointer in `TRUTH_LEDGER.md`, then read that code. `TRUTH_LEDGER.md` is the ONLY doc trusted for columns.
- Every other `.md` (handoffs, specs, findings) is background, not truth. If a doc and the code disagree, code wins and the doc is a stale trap — do not repeat it. (Docs have lied before: "dual entry" / "yfinance" long after both were removed.)
- If you cannot quote the code, say "I haven't verified this" — never imply you did.
- Changing code = update its comments/docstrings in the SAME edit. A comment that now lies is a bug (stale comments have burned hours). If behaviour moves, the words next to it move too.
- Session memories and recalled numbers/insights are HYPOTHESES, not facts. State one only after re-verifying against live code or an approved analysis `.md` (TRUTH_LEDGER.md, backtest/RECOMMENDATIONS.md) whose date is AFTER the last detection change. Detection changed = every older derived stat (%, WR, counts) is stale until recomputed. (The "93% with-trend" figure was quoted 2 weeks stale — this rule exists because of that.)

**2. One data file.**
- The only analysis CSV is the one named in `backtest/results/CANONICAL.md`. Never `glob` for `trades.csv` and use whatever turns up — dozens of stale versions with different schemas have burned hours.
- Confirm the column count matches CANONICAL before using it. Wrong count = wrong file = STOP.

**3. No sycophancy.** Softening a real disagreement or praising before thinking causes wrong logic and lost money.
- State disagreements. Delete "great question" / "you're right" — show reasoning instead.
- Don't flip position under pushback without new evidence — hold and cite, or admit error with reasoning.
- Don't call something "the big one" before proving it. Don't claim "verified" on a skim.

**4. Response format.** Bullets and headers only, no paragraphs. Every word earns its place. Plain English, simple sentences.

**5. No 18-year backtest without an explicit ask.**
- NEVER launch the full 18-yr run (2008→now) on your own judgment. It is slow and expensive. Run it ONLY when the user says so in that turn.
- Default to SMART SAMPLING: a small, representative slice (a few months / a few pairs / a targeted window) proves or kills a hypothesis. Validate on a sample first; escalate to more data only when the sample is genuinely ambiguous, and say why.
- Frozen cached windows (`backtest/cache/*.parquet`) and single-window replays are the first tool — no feed pull, no full run.

---

## NON-NEGOTIABLE — DECISION GUARDRAILS

- `DECISION_GUARDRAILS.md` holds the frozen edge-engine-phase rules.
- Before any backtest/engine/detection decision, check the action against it. If it breaks a rule, FLAG THE RULE ID FIRST (e.g. "this breaks C5"), then help.
- Those rules change only per that file's own change-log procedure — never in the same sitting as the decision they block.

---

## What The System Actually Does (verified against code)

**Live-traded instruments (5):** EURUSD, USDJPY, NZDUSD, USDCHF, GOLD (XAUUSD).
- NAS100 + GBPUSD, AUDUSD, USDCAD, EURJPY, BTCUSD are `backtest_only: true` in `config.json` — never judge or tune live behaviour on them.

**Timeframe:** H1 only. M15/M5 entry is retired (`Phase2_Alert_Engine.py:1400, 2635`). Phase 3 (`phase3_engine.py`) is dormant dead code on the old yfinance feed. Do not propose M5/M15 changes.

**Live feed:** Twelve Data (`feed_adapter.py`). yfinance removed from live. Backtest data is MT5 2008+.

**How a trade is found:**
1. Dealing Range defines what we trade within.
2. CHoCH or BOS inside the range identifies the relevant Order Block.
3. When price approaches the OB, confluences are scored (FVG, liquidity sweep, kill zone, macro news, PD array alignment, OB/FVG freshness).

**Phases:**
- **Phase 1 — Scout** (`smc_detector.py`, `smc_radar.py`, `dealing_range.py`): dealing range, structure, OBs.
- **Phase 2 — Trade Readiness** (`Phase2_Alert_Engine.py`): tradeability, limit orders, confidence score.
- **Weekly Review** (`weekly_review.py`).

---

## Rules

**Communication**
- Plain English, no jargon unless necessary. Match length to the question — default short.
- Recommend thinking-mode upfront for non-trivial methodology / architecture / scoring questions.

**Reading vs changing**
- Reading is NEVER gated — read code, data, logs, config, state immediately without asking.
- Approval gates apply ONLY to writing (edits, commits, pushes).

**Code changes**
- Never touch code without explicit approval. Trading-logic changes always need confirmation.
- Flag architectural changes before acting; small obvious fixes after announcing.
- One concept, one implementation. Duplicate logic is a bug, not design.

**Data vs SMC methodology — never conclude on data alone**
- Map every data finding against verified SMC methodology before it becomes a conclusion.
- Data + SMC agree → conclude and act. Data + SMC disagree → DISCUSSION POINT, not a conclusion: surface it, name the likely cause (detector bug, thin sample, timeframe mismatch), brainstorm — do NOT score/filter on it yet.
- A thin sample never overrides sound SMC logic, and weak data does not disprove a sound SMC signal.

**Git workflow**
- Approving an edit ≠ approving a push. Commit + push only on "ship it" / "push" / "publish".
- On ship: stage only relevant files (never blanket `git add -A`), clear message, push `origin main`. Skip `.claude/settings.local.json` unless asked.
- Repo lives in OneDrive — backtests commit local-only on dev, push only in CI (lock collisions). Pull only when the remote may have moved.

**Quality**
- Sanity check after every change. Flag design / logic / system problems proactively.
- Anticipate edge cases. Present options when input is needed; never surface a problem without solutions.
- Prefer low/no-maintenance solutions; flag and justify any medium/high-maintenance option.

**Logging (log everything measurable)**
- If a value can be measured, LOG IT — into the per-trade row (trades.csv) at minimum, and the email breakdown when it's a win-rate lever.
- Always FLAG what you logged and where. If a new metric is NOT logged, say so and why.
- **Truth-ledger gate:** no new trades.csv column or emitted insight ships without a row in `TRUTH_LEDGER.md` (source file:line, when stamped, population). Mutable OB state must be stamped `*_at_alert` at the yield, never read live at row-build time.

**Dual perspective**
- Trading logic: think like a vet who has placed thousands of trades — "would a vet respect this signal?"
- Architecture: think like a senior Python architect — "is this clean, observable, maintainable?"
- Surface where the two disagree. Methodology is open to evolution — raise better detection / scoring approaches.

---

## Defensive Coding & Regression Guards — Judgment, Not Reflex

- Add a guard only when a change can **silently** fail, be skipped, or regress in a way that corrupts alerts or P&L unnoticed. If the failure mode isn't real and silent, don't — a needless guard is dead weight.
- **A guard must never break the thing it protects.** Guards live OUT of the live trade path: offline tests, CI gates, standalone scripts.
- Do NOT put a new assertion / fail-loud raise INSIDE live alert generation or the row build unless the alternative is a silent WRONG alert. When in doubt, log-and-continue and catch it in a test.
- Cheapest guard that bites is usually one regression test. When a guard IS warranted, state briefly: failure mode → the check (confirm out-of-band) → trading impact if unguarded.
