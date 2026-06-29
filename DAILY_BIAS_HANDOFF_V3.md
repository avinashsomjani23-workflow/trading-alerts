# Daily Bias — Master Handoff V3

**This is the SINGLE source for the daily-bias layer. It absorbs and REPLACES the old
`DAILY_BIAS_BUILD_HANDOFF.md` (v2) and `DAILY_BIAS_FINDINGS_HANDOFF.md` (both deleted).**

Everything below is a **CLAIM to re-verify, not gospel.** The work so far was theory-first and
produced four fragile versions. Your job is to CHECK this critically and then SOLVE it — not
inherit it on faith. Re-run every script before trusting a number.

---

## 0. HOW TO WORK ON THIS (inherit this operating style — it is why the good chat worked)

This problem is not blocked by ideas; it is blocked by sloppy method and unverified assumptions.
Work like this, unprompted — do not wait to be asked:

- **Verify the detection method by READING the code before claiming anything about it.** Most of
  the wasted time came from describing behaviour from memory. Open `dealing_range.py`, read the
  function, then speak. Cite `file:line`.
- **Web-search external facts the moment they matter** — broker daily-candle boundaries, ICT/SMC
  methodology, Twelve Data API behaviour, DST windows. Do not reason from stale memory about the
  outside world. The good chat searched the instant a fact about the world (not the code) was load-
  bearing.
- **Data-first, not rule-first.** Do not invent a detection rule and then test it. Take real data,
  find the pattern, let the pattern define the rule, then validate it on UNSEEN data. (See §9.)
- **Brutal honesty, no sycophancy.** If the user's premise is wrong, say so with evidence. If a
  finding kills the layer, say "kill it." A small/thin sample never overrides sound SMC, and sound
  data never gets softened to be agreeable.
- **Data vs SMC:** when they agree → conclude. When they disagree → it is a DISCUSSION POINT (name
  the likely cause: detector bug, thin sample, timeframe mismatch); do NOT silently score on it.
- **Be proactive about edge cases and blind spots.** Anticipate them, surface them with a solution
  attached. Never present a problem without options.
- **The user is technical but hates verbosity and jargon.** Bullets, plain English, short. Match
  length to the question. H1 is the ONLY execution clock (M5/M15/Phase-3 dormant) — never re-open
  scope on that. P&L truth is `r_realised`, always. NAS100 excluded.

---

## 1. THE ONE DECISION (read before anything else)

**Stop inventing detection rules and testing them. Invert the approach:**
1. Take a large slice of the committed H1 backtest results (`r_realised` per trade).
2. For each trade, compute the **daily structure that preceded it** (point-in-time, no look-ahead).
3. Find which daily conditions actually separate winners from losers.
4. **Let the surviving pattern define the bias rule** — then validate it on a DIFFERENT slice (OOS).

We are on version 4 because we kept guessing rules. The detector is only a means to better trade
SELECTION. If a big-slice study shows NO selection value → **kill the layer and save the time.**
That outcome is a win, not a failure.

---

## 2. What we are building, and why

A **daily CONTEXT layer** on top of H1. It never enters a trade. It answers: *is this H1 setup
running WITH or AGAINST the higher-timeframe story, and is there a liquidity draw ahead?* Intended
value = **trade selection / conviction**, NOT direction prediction (that is a proven coin-flip, §6).

Three deterministic components (NO LLM — bias is geometry):
1. **Daily bias** — sticky direction (from the structure engine, §4).
2. **Liquidity hierarchy** — ranked external pools: `PWH/PWL > PDH/PDL > internal`.
3. **Draw-on-liquidity (DOL)** — nearest unswept pool in bias direction (the magnet) + the
   recently-swept opposite pool that fuels a reversal.

---

## 3. The journey — what died and why (do not rebuild these)

- **v1:** stateless daily HH/HL recompute → 38% neutral, no stickiness. Dead.
- **v2:** sticky bias + a separate "ranging mode" stall counter + pools + DOL. Looked complete;
  died on contact with data (this chat).
- **Findings chat:** proved sticky bias is a **directional coin-flip (50.4–50.6%** over 18y).
  SETTLED: **daily bias does NOT predict direction. Do not try to forecast the regime.**
- **This chat (V3):** found the system already HAS a fractal structure engine; tested it on D1;
  **killed the "ranging" stall flag** (§5); confirmed the H1 ATR buffer is wrong for D1.

**Dropped number — do NOT cite:** the old "with-bias +0.153R vs neutral −0.228R" came from an
H1-trend-state proxy, not a real D1 bias. **There is currently NO valid selection evidence.** The
layer's worth is UNPROVEN. §1/§9 exist to prove or kill it.

---

## 4. THE DETECTION METHOD (verify by reading the code, then attack it)

**Core realization — verify first, it changes the whole design:** the system already has a
**fractal, timeframe-agnostic structure engine**, `compute_structure(df, h4_range, ...)` at
`dealing_range.py:481`. It is what sets the H1 trend today. **Feed it D1 bars → D1 bias in the
SAME language as H1.** Do NOT invent a separate D1 method: a different method makes D1 and H1
disagree as an ARTIFACT, not a signal, which poisons every with/counter slice.

Engine flip model ("CONFIRMATION-BOS", locked with the trader 2026-06-23, `dealing_range.py:702-945`):
- Trend is sticky: `HH+HL → up`, `LH+LL → down`; mixed structure HOLDS (never neutral).
- A **CHoCH** (close beyond the defended swing by `choch_atr_mult * ATR` + a body gate) only
  **ARMS** a reversal — it does NOT flip the trend.
- The flip fires on a **Confirmation BOS** (a break in the CHoCH direction).
- A pending CHoCH **cancels on reclaim** (close back past origin) → `CHoCH_FAILED`.
- **`ranging`** (`dealing_range.py:1187`): `state set AND trend_dir_swings_since_extend >= STRUCTURE_RANGING_STALE`
  (default **N=2**, `:473`). The counter = confirmed swings since the trend extreme last extended,
  +1 on a non-extending swing, reset on extension (`:1162-1182`). **This IS the "stall counter
  N=1/2/3" v2 wanted to build — same mechanism, already coded.**

Reuse confirmed (verify): `detect_swings(df, lookback, min_leg_atr_mult)` at `:274` (geometry
lb=3 + optional ATR leg filter); `_compute_atr` at `:109`; `compute_pd_position` reads
`walls["h4_range"]` for premium/discount; `compute_h4_range` broken-wall tracking is mirror-able
to D1. The H1 ATR leg filter `MIN_LEG_ATR_MULT=1.5` (`:74`) is H1-tuned (its docstring says so) —
disable on D1 with `_min_leg_atr_mult=None`.

**Two flip rules were compared in this chat:**
- **Rule A — raw formation:** flip the instant `HH+HL`/`LH+LL` forms on confirmed swings, no
  confirmation. Earliest, whippiest.
- **Rule B — engine CHoCH+Confirmation BOS**, with the H1 ATR/body buffers ON or OFF (OFF = pure
  close-beyond).

**The test harness:** `backtest/diagnostics/scratch_d1_bias_flip.py` (uncommitted, re-runnable).
Loads D1 CSVs, runs both rules point-in-time (Rule B via the engine's `_trace` hook; buffer OFF =
monkey-patch `BOS_ATR_MULT`/`BOS_BODY_ATR_MULT`/`STRUCTURE_CHOCH_BODY_ATR_MULT` to 0 and
`choch_atr_mult=0`), and measures per flip:
- **lag** = D1 bars between the *actual* price extreme and the bar the flip fired
  (`lag = flip_bar − extreme_bar`; i.e. how many days AFTER the true top/bottom we called it).
- **correct vs fakeout** = within 20 fwd bars, did price extend in the flip direction (held) or
  reclaim the old extreme first (fakeout)?
- **ranging validation** = on ranging-flagged bars, mean |10-bar net close move| in ATR vs
  expansion bars (if the flag is real, ranging should move LESS).

Tested on **EURUSD, NZDUSD, XAUUSD** only. Data: `backtest/mt5_data/<INST>_D1.csv` (MT5, 2008+,
~5,400 D1 bars each).

---

## 5. Data collected THIS chat (reproduce with the harness before trusting)

**Flip study, pooled EUR/NZD/XAU:**

| Rule | flips | %correct | median lag (D1 bars) |
|---|---|---|---|
| A — raw formation, lb=3 | 562 | 84% | 14 |
| A — raw formation, lb=2 | 775 | 85% | 9 |
| B — CHoCH+BOS, buffer OFF, lb=3 | 159 | 98% | 31 |
| B — CHoCH+BOS, buffer ON, lb=3 | 51 | 98% | 45 |
| B — CHoCH+BOS, buffer OFF, lb=2 | 226 | 95% | 20 |

**Caveat on the 98%:** Rule B flips fire LATER, after price already moved — so "extended before
reclaiming" is partly tautological. The 84%-vs-98% gap is partly a measurement artifact of lag,
not pure edge. Read it as *earlier-but-whippy (A)* vs *later-but-self-confirming (B)*.

**Settled by this data:**
- **The H1 ATR/body buffer is wrong for D1.** Buffer ON → XAUUSD got **0 flips in 18 years** (the
  body gate, body ≥ 1.5×ATR, never passes on a single daily candle). Buffer OFF → correctness
  unharmed (98% at lb3), lag lower. **On D1 use pure close-beyond.** (User confirmed.)
- **lb=2 is competitive on D1** (v2's "lb=3 stays" was an H1 finding). Lower lag, similar
  correctness. Keep open.

**Ranging stall flag — KILLED (the headline finding):**

| inst | N=1 rng/exp fwd-ATR | N=2 | N=3 |
|---|---|---|---|
| EURUSD | 2.47 / 2.48 | 2.49 / 2.47 | 3.78 / 2.44 |
| NZDUSD | 2.50 / 2.46 | 2.46 / 2.48 | 3.42 / 2.46 |
| XAUUSD | 0.35 / 0.33 | 0.39 / 0.33 | 0.65 / 0.33 |

Bars flagged "ranging" move the SAME or MORE than expansion bars. **The stall counter does NOT
detect sideways/low-movement. `stall ≠ ranging`. Do not use it to lower trust.** Real ranging =
price CONTAINED between two levels (low net travel) — a different computation that falls out of the
liquidity-pool layer, not the swing counter. (Caveat: one metric, single-ATR normalizer; the
relative ranging-vs-expansion gap is robust across all 3 instruments — re-run if you doubt it.)

---

## 6. Inherited facts from the findings chat (re-verify; scripts may need re-creating)

- **Sticky bias directional accuracy:** pooled H5 50.6 / H10 50.4 / H20 50.6% (6 instruments, 18y)
  — a coin flip. Proof: `backtest/diagnostics/scratch_bias_reliability.py`.
- **First-break fakeout rate = 27%** (at the first close beyond a prior-day level, 27% close back
  inside next bar; consistent EURUSD/XAU/BTC). → N=1 confirm is justified; one bar of latency is
  fine for a daily layer.
- **W1 from D1 is lossless:** weekly H/L = max/min of that week's daily H/L matches MT5 native W1
  100% exact. (Proved within ONE feed — not a cross-feed claim.)
- **Predictive ranging / stale-count / width / compression all measured BELOW base-rate** as
  forward predictors. No method forecasts trend-vs-range >90%. Stop chasing forward prediction.

---

## 7. Feed, boundary, DST trap, BTC (settle only when shipping pools LIVE — not before)

- **Backtest = MT5.** D1 rolls at **21:00 UTC** (00:00 server, fixed GMT+3 = 5pm NY; MT5 H1
  reproduces MT5 D1 100% at server-midnight). **Live feed is being switched to Twelve Data**
  (OANDA geo-blocks India). Prior parity test (memory `project_oanda_twelvedata_eval`): Twelve Data
  ~90% self-consistent, FX p95 ~5–10 pips vs MT5, beats yfinance but missed a strict gate — re-test
  before adopting for exact pools.
- **What is feed-robust vs not:** bias DIRECTION is cross-feed robust (98.8% — it is a sequence).
  Exact POOL prices (PDH/PDL) are NOT — absolute price lines differ by feed/boundary. So the §1/§9
  SELECTION study (direction-based) is feed-safe TODAY on MT5; only live exact-pool alerts wait on
  the feed.
- **DST parity trap:** MT5 is fixed GMT+3 → 21:00 UTC year-round = 5pm NY summer / 4pm NY winter.
  Any live builder must mirror the **fixed 21:00 UTC**, NOT chase the shifting true-NY-close, or
  live silently desyncs from the backtest. **Confirm Twelve Data's daily boundary by web search +
  a fetch test before pools go live.**
- **BTC:** crypto-canonical day = **00:00 UTC** (where retail stops sit); MT5 rolls BTC at 21:00
  UTC (non-canonical). Build BTC pools at 00:00 UTC from H1. Venue divergence large (~$40 median
  high gap) → bucket BTC separately.
- **Pools must be built on ONE fixed boundary** — "real D1 from both feeds" guarantees mismatch
  because the feeds use different native day boundaries. Do NOT use ATR-fuzzy zones to paper over
  this (it corrupts the sweep-vs-break distinction). Fix = one clean consistent feed.

---

## 8. SETTLED vs OPEN

**Settled (re-verify, then build on):**
- Reuse `compute_structure` on D1 — fractal, cohesive with H1. No separate D1 method.
- D1 uses **pure close-beyond** (no H1 ATR/body buffer).
- The `ranging` stall flag is dead for trust-scoring. Park it; define real ranging via containment
  only if the data later demands it.
- Daily bias does NOT predict direction (50%). Value = selection only.

**Open (answer with §9, data-first):**
- **Does daily bias have ANY selection value?** The make-or-break question. No valid evidence
  exists. Answer it FIRST; kill the layer if the answer is no.
- Flip rule: formation (timely, 84%, lag 14) vs confirmation (reliable, 98%*, lag 31) — decide by
  selection value on `r_realised`, not by lag/correctness in isolation.
- lb 2 vs 3 on D1.
- Should H1 itself use formation vs confirmation? (`scratch_d1_bias_flip.py` can point at H1 — do
  NOT assume H1 is optimal; read + test it.)
- Liquidity tier value (feeds the separate sweep rebuild — `SWEEP_REBUILD_HANDOFF.md`).

---

## 9. RECOMMENDED FIRST MOVE (concrete, do this in the new chat)

1. Pull committed H1 results (`backtest/results/h1only_*`). Split: one large DISCOVERY slice, a
   separate held-back VALIDATION slice (different years — true OOS).
2. For each trade, compute D1 bias with `compute_structure` (buffer OFF, lb=3), point-in-time as of
   the trade's H1 bar (persist/replay the state — it is path-dependent; a recompute-from-now
   silently diverges live vs backtest).
3. Slice `r_realised` by: with-bias vs counter-bias; bars-since-flip; freshness (position in the
   live D1 range). **Look for a separation that SURVIVES into the validation slice.**
4. If nothing survives OOS → report honestly and STOP (layer not worth building). If something does
   → that pattern is the rule to wire, observe-first: log the daily context next to `r_realised`,
   change NO behaviour, score nothing until OOS-proven. Prove P&L identical to a pre-build baseline
   (any drift = you contaminated selection).

---

## 10. Files, pointers, glossary

- **Engine** `dealing_range.py`: `compute_structure` (481), `detect_swings` (274), `_compute_atr`
  (109), ranging (1187), `STRUCTURE_RANGING_STALE` (473), `MIN_LEG_ATR_MULT` (74).
- **This chat's study** `backtest/diagnostics/scratch_d1_bias_flip.py`. **Directional proof**
  `backtest/diagnostics/scratch_bias_reliability.py`.
- **D1 data** `backtest/mt5_data/<INST>_D1.csv`. **P&L truth** `r_realised`. **NAS100 excluded.**
- **Related (separate tasks):** `SWEEP_REBUILD_HANDOFF.md` (depends on this layer's liquidity tier);
  exit-leak / MFE give-back is a separate chat and may be a bigger P&L lever than this whole layer —
  sequence honestly.
- **Glossary:** Buyside = above price (PWH/PDH/forming-high); Sellside = below (PWL/PDL/forming-low).
  Sweep = wick through + close back (fuel). Break/displacement = close beyond + hold (N=1). DOL =
  nearest unswept pool in bias direction. A pool dies once swept. Tier: `PWH/PWL > PDH/PDL > internal`.
```
