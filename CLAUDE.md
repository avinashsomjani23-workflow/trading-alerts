# Trading Alerts System — Project Guide for Claude

Automated SMC alert system. Goal: replicate veteran SMC judgment, not generate noise.

**We are in the ANALYSIS / TESTING phase.** The system is built. The job now is to measure
it on the backtest, keep the signals that filter out losing trades and amplify winning ones,
and drop the rest. Not to keep building detection.

---

## NON-NEGOTIABLE — READ EVERY TURN

Seven rules. Break one and the response is wrong — rewrite before sending.

**1. Code is truth.**
- Quote the live `file:line` in the SAME response before stating any column meaning,
  detection behaviour, or trading logic. No code quote = do not state it.
- Column meanings come ONLY from `TRUTH_LEDGER.md` (its pointer, read against live code).
  Every other `.md` is background — if a doc disagrees with code, code wins, doc is stale.
- Can't quote the code? Say "I haven't verified this." Never imply you did.
- **Stale-comment check:** after any edit, re-read the comments on the lines you changed;
  if the edit changed a name, number, path, condition, or behaviour they describe, fix them
  in the SAME edit. A comment that now lies is a bug.

**2. Canonical = the active analysis run.**
- The ONE truth CSV is whatever `backtest/results/CANONICAL.md` names. Read it EVERY time —
  never `glob` for `trades.csv`. Confirm the column count before use; wrong count = STOP.
- New run → repoint CANONICAL.md (path + shape) in the SAME commit and archive the old run.

**3. Validation & Holdout are SEALED.**
- Discovery is the ONLY block open. NEVER read or compute on Validation/Holdout years
  without explicit permission that turn — peeking burns the one-shot test.
- Log any method change made after seeing a sealed block in `ANALYSIS_POINTERS.md`.

**4. Every pre-change number is stale.**
- Detection changed = every %, win-rate, count, CI, and feature verdict from before is VOID
  until re-derived from the new run. A remembered number is a HYPOTHESIS — re-verify against
  the current canonical + live code before stating it.

**5. Question me — this system filters losers and amplifies winners, it does not flatter me.**
- Every bias, hypothesis, and assumption I bring gets questioned and tested for relevance and
  accuracy — including in the turn I bring it. Agreeing to please me loses money.
- State disagreements plainly. Delete "great question" / "you're right" — show reasoning.
- Don't flip under pushback without new evidence — hold and cite, or admit error with reasoning.

**6. No full 18-yr run without an explicit ask.**
- Never launch the full 2008→now run on your own judgment — slow and expensive.
- Default to SMART SAMPLING: a few months / pairs / a targeted window proves or kills a
  hypothesis. Cached windows (`backtest/cache/*.parquet`) and single-window replays first.

**7. Read `COLUMN_BUCKETS.md` before ANY analysis — no column missed, no look-ahead.**
- Before running a loser / entry analysis, read `COLUMN_BUCKETS.md`. It lists EVERY canonical
  column, tagged by WHEN its value is knowable: `alert` / `fill` / `outcome`. It is generated
  from the live CSV header (`python -m backtest.gen_column_buckets`), so nothing is ever missing.
- **Use every column** in the analysis population — none is silently skipped.
- **`outcome` columns are look-ahead** (r_realised, mfe/mae, bars_to_*, exit_*, sl_* anatomy).
  Use them FREELY to describe / group / find patterns in losers (autopsy). NEVER let one drive
  a live entry filter — live, you don't know them yet.
- **`fill` columns** are usable for entry analysis, but ALWAYS say out loud "this is fill-time"
  (a live alert-time scorer can't see them). The alert/fill gate is flexible; the label is not.
- Never guess a column's class from its NAME — the doc/generator is truth (a name like
  `pdh_status_at_fill` is fill-anchored; older code mis-named it `_at_alert`). If a column isn't
  in the doc, the generator RAISES and CI goes red — that is the guarantee, not a suggestion.

---

## THE ANALYSIS ANSWER CONTRACT — every finding, fixed shape

Six beats, plain English, in order. Never dump raw stats.

1. **The idea** — what am I testing and why would a vet care.
2. **What the data is** — pair, block (Discovery unless permitted), row count, news-clean.
   In words, not just a filename.
3. **The result** — the numbers; each number followed by ONE plain sentence of what it means.
4. **Significant or not** — show the CI AND NAME THE METHOD (bootstrap / Wilson / whatever),
   or the sample size N. Then say plainly: "real" or "thin — don't act." A number with no CI
   and no N is not a finding, it's noise. Name any other stat used (FDR, DSR, calibration).
5. **SMC cross-check** — does the mechanism agree? Agree → conclude. Disagree → DISCUSSION
   POINT, not a filter: name the likely cause (detector bug, thin sample, TF mismatch). A
   thin sample never overrides sound SMC; weak data never disproves a sound signal.
6. **So what** — act, park, or kill. One line.

**Standing rules for every finding:**
- **News-clean only.** News-mixed rows are filtered OUT of the population. Always.
- **A signal is one hammer, not the wall.** Judge it on whether it correctly filters bad
  trades and/or amplifies good ones — NEVER dismiss it because it doesn't move the whole book.
  Many small correct signals stacked is the goal.
- **Barely-insignificant → LOG it, don't bin it.** A near-miss (CI just crossing, small
  effect) gets logged to `ANALYSIS_POINTERS.md` with its numbers. Nine years of one pair is
  not the verdict — it may hold on Validation, Holdout, or another pair.
- **Plain English always.** Explain idea, data, and result as to someone smart who doesn't
  speak SMC. Define any term you must use.
- The full 10-step method lives in `ANALYSIS_POINTERS.md` — followed there, not copied here.

---

## STILL BINDING (from the build phase — a change now must not leave these behind)

- **Log everything measurable** — into `trades.csv` at minimum, email breakdown when it's a
  WR lever. Flag what you logged and where; if a metric is NOT logged, say so and why.
- **Truth-ledger gate** — no new column or emitted insight ships without a row in
  `TRUTH_LEDGER.md` (source `file:line`, when stamped, population). Mutable OB state stamped
  `*_at_alert` at the yield, never read live at row-build.
- **Edits need approval, then the stale-comment check.** No code touched without your OK;
  trading-logic changes always confirmed. One concept, one implementation.

---

## REFERENCE — stable facts + pointers

- **System:** 5 live pairs (EURUSD, USDJPY, NZDUSD, USDCHF, GOLD); others `backtest_only` in
  `config.json` — never tune live on them. H1 only (M15/M5 retired, Phase 3 dormant). Live
  feed Twelve Data (`feed_adapter.py`); backtest data MT5 2008+. Trade = Dealing Range →
  CHoCH/BOS finds the Order Block → confluences scored as price approaches the OB.
- **Reading is never gated** (code, data, logs, config, state — read immediately). Approval
  gates apply ONLY to writing. Bullets/headers, not paragraphs. Default short.
- **Git:** edit-approval ≠ push-approval; push only on "ship it". Stage relevant files only.
  OneDrive repo — backtests commit local-only on dev, push in CI.
- **Decision guardrails:** `DECISION_GUARDRAILS.md` is frozen. Breaking a rule → FLAG THE
  RULE ID FIRST (e.g. "this breaks C5"), then help.
- **Pointers:** playbook + parked ideas → `ANALYSIS_POINTERS.md` · columns → `TRUTH_LEDGER.md`
  · active data → `backtest/results/CANONICAL.md` · reliability method → `backtest/RECOMMENDATIONS.md`
