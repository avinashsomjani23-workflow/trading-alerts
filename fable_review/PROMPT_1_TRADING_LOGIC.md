# PROMPT 1 — Trading Logic / Methodology Review

> Paste this into Fable AFTER uploading `FABLE_REFERENCE_LIVE.md`. Plain English, bullet
> points and headers only in your reply. No praise, no preamble.

---

You are reviewing the **trading methodology** of a live H1 Smart Money Concepts (SMC)
alert system. The attached file `FABLE_REFERENCE_LIVE.md` is verified ground truth, read
line-by-line from the real code. **Design against that file, not against assumptions.**
Where it flags "DOC-DRIFT", the code wins over the comments.

## Who you're reviewing for
A discretionary-trained SMC trader who has placed thousands of trades. He wants a veteran to
**respect every signal**. The end goal is a robust algorithm, not a noisy one. Be brutally
honest. If something is weak, say it plainly. Do not soften to be agreeable.

## Hard rules
- **Propose nothing as final code.** Output methodology critique, ranked improvement
  **options with trade-offs**, and edge cases — a SPEC the engineer brings back.
- The system is **H1-only**. Ignore M5/M15, Phase 3, and the backtest entirely.
- Do not relitigate H1-only. Everything else (including the structure constants) is open.
- If you need a fact that the reference doesn't give you, **list it as a question** at the end
  — do not guess. The engineer will verify against code and answer.

## What to evaluate — ask "would a vet respect this?" at every step

**1. Swing + structure definition (reference §4.1, §4.4)**
- Is lb-3 geometry + a 1.5×ATR leg filter a sound swing definition for H1 across forex, gold,
  and an index? Where does it over- or under-detect?
- The trend / BOS / CHoCH engine: is firing BOS and CHoCH **on close** with displacement
  thresholds (BOS 0.4·ATR, CHoCH 1.0·ATR) and a failure window (lock 1.5·ATR / reclaim) a
  faithful model of how a vet reads structure? Where does it diverge from discretionary reading?
- The block ORDER (failure-window pre-empts CHoCH/BOS each bar). Any sequence that produces a
  wrong or whipsaw-prone result?

**2. Order Block construction (reference §4.6)**
- "First opposing candle walking back from the break, skipping oversized (>2·ATR) and doji
  (<20% body) candles" — is that the right OB? When does it pick the wrong candle?
- OB1 (most recent event) bypasses the proximity gate and always surfaces. Signal or noise?
- Only 2 OBs ever surface per pair (OB1 + closest). Does capping at 2 drop setups a vet wants?

**3. The invalidation rule (reference §4.9, §6 items 2–3) — look here hard**
- A bullish OB dies when a single **wick** touches the OB low (distal), and 3 wick-touches at
  proximal also kill it. Is wick-based distal invalidation correct, or too aggressive — and is
  it inconsistent that FVG mitigation is close-based for gold/NAS but OB distal is wick-based
  for all pairs?

**4. Sweep + FVG confluence (reference §4.7, §4.8)**
- Is the sweep model (active-target-only, survivorship check, base/equal-levels/rejection
  scoring) how a vet judges a liquidity sweep? What's missing or over-weighted?
- Is collapsing sweep to **presence-only on non-JPY forex** (discarding quality) defensible, or
  is it discarding real edge on EURUSD/NZDUSD/USDCHF?
- FVG: 3-candle gap, pair-aware mitigation (touch for forex, close for gold/NAS). Sound?

**5. Scoring construction (reference §5.3)**
- Structure 4 / Sweep 1-or-3 / FVG 2 / Freshness 1. Are the weights and the forex/non-forex
  asymmetry justified, or arbitrary? What does a vet actually weight most?
- PD and killzone are computed but scored 0. Right call, or should one of them carry weight?
- There is currently **no score gate** (score is advisory). Don't assume one. Instead:
  (a) critique whether this scorecard is even measuring the right things, and
  (b) tell me **exactly what to measure** to decide whether score predicts trade outcome and,
  if it does, what gate threshold and shape you'd test first.

**6. Entry / SL / TP (reference §5.2)**
- Entry at OB proximal, SL at distal ± 1 spread, TP1 = nearest opposing swing clearing 1.5R,
  no-trade if nothing clears 1.5R. Is that vet-grade trade construction? Where does it leave
  money on the table or take bad risk?
- TP depends entirely on H1 swing availability. Is "no qualifying swing → no trade" correct,
  or a coverage hole that silently kills good setups?

**7. Alert cadence (reference §5.4)**
- Four re-email triggers + hysteresis. Does this match how a vet wants to be pinged, or will it
  over- or under-alert? Any trigger that would annoy or mislead a real trader?

## Output format (strict)
For EACH of the 7 areas:
- **Verdict:** sound / weak / broken — one line.
- **Why:** the reasoning, vet's-eye.
- **Options:** 2–3 concrete improvement options, each with trade-offs. Rank them.
- **Edge cases:** where the current logic silently does the wrong thing.

Then:
- **Top 5 changes, ranked by expected impact on signal quality** (most bang first).
- **Questions for the engineer** — anything you couldn't decide without a fact you don't have.
  Be specific (name the function / behaviour you need confirmed).

Brutal honesty. If an area is actually fine, say "fine" and move on — don't invent problems.
