# News-Clean Discovery Log — EURUSD, Discovery block (2008-2016)

**The one home for loser-autopsy / feature-screen findings.** Method pointers live in
`docs/ANALYSIS_POINTERS.md`; column meanings in `TRUTH_LEDGER.md`; active data in
`backtest/results/CANONICAL.md`. Do NOT put finding tables in those files — put them here.

---

## Scope of everything below

- **Data:** EURUSD only, Discovery block 2008-2016 (`h1only_20080102_20161231/trades.csv`, 180 cols).
  ONE pair, ONE nine-year block. Validation + Holdout SEALED — untouched.
- **Population:** STRICT news-clean (`news_fill==0 & news_open==0`), resolved only (`exit_reason ∈ {tp,sl}`).
  Baselines: **STRICT WR 27.8%** (autopsy pop) · **LOOSE WR 30.1%** (`news_fill==0`, for EV).
- **Exit:** fixed 2R — win = +2R, loss = −1R. So **meanR = 3·WR − 1** (meanR is just the win rate as money).
- **Method (uniform, every feature):** quantile buckets or True/False levels → per bucket report
  **N · WR + Wilson 95% CI · meanR + bootstrap 95% CI · sumR · p vs the rest.** Raw continuous
  columns only (never tiers/buckets like `bos_tier`). Outcome columns describe losers, never drive a filter.
- **Statistics policy — DISCOVERY STAGE:** raw numbers only, **NO multiple-testing (FDR/BH) correction.**
  Rationale: at discovery a false alarm is cheap (it dies in the next test), a missed real signal is
  expensive (we lose the edge). FDR trades misses to cut false alarms — backwards for us here.
  **Confirmation = out-of-sample (Validation + Holdout + other pairs), not a p-value gate.** Nothing is
  discarded on statistics; everything is logged with its raw numbers.
- **Reminder on the univariate null:** finding little on single features is EXPECTED in noisy price data,
  not a red flag. Edges live in combinations/context → that is the multivariate step, not a failure here.

**How to read a p-value here:** it assumes the feature has ZERO effect, then asks "how often would pure
luck alone show a gap this big?" Small p = hard to explain by luck. It is NOT "chance this is luck."
p=0.84 → luck routinely makes gaps this big → learned nothing. p=0.03 → luck rarely does → worth a look.

---

## AMPLIFIERS (win rate above the 27.8% baseline)

| signal | N | WR (Wilson 95%) | meanR (boot 95%) | sumR | p vs rest | note |
|---|--:|---|---|--:|--:|---|
| **FVG present** (any size) | 457 | 31.5% [27,36] | −0.05 | −24 | — | +5.8 WR vs no-FVG (25.7%); **holds all 3 year-blocks** (+9/+4/+5). Presence, not size. |
| **swing sweep fuel** `sweep2_sw_present` | 206 | 34.0% [28,41] | +0.02 | +4 | 0.031 | H1-swing liquidity swept by the leg that built the OB. |
| **patient pullback** `bars_break_to_pullback` ≥48 | 254 | 34.3% [29,40] | +0.03 | +7 | 0.011 | Very slow break→fill wins. **Fill-time** (live scorer can't see it — flag). |
| **equal-level sweep FUEL** `sweep2_eq_present` | 25 | 48.0% [30,67] | +0.44 | +11 | 0.023 | **N=25 — VERY THIN, CI huge. Watch only.** See "two equal-level ideas" below. |

## FILTERS / LOSS-LEANING (win rate below baseline) — hypotheses, not yet acted

| signal | N | WR (Wilson 95%) | meanR (boot 95%) | sumR | p vs rest | note |
|---|--:|---|---|--:|--:|---|
| **biggest break candle** `break_body_atr` top tercile | ~400 | 24.0% | −0.28 | −large | 0.03 | −3.8 WR vs rest; **consistent across all 3 year-blocks**. Weak, but stable. |
| **leg-extreme was a sweep** `setup_liq_legextreme_swept`=True | 114 | 22.8% [16,31] | −0.32 | −36 | 0.256 | Buying INTO a grab = wrong side. Same for long (23.3%) & short (22.2%). **N=114 thin — carry to multivariate + Validation + other pairs.** |
| **late / internal LONG-in-bullish** `bos_sequence_count`≥3 | 81 | 18.5% [11,29] | −0.44 | −large | 0.053 | Deep-sequence (late/internal-structure) longs in bullish H1 trend. See LONG deep-dive. |

## FLAT IN THIS SLICE — keep, do NOT delete from the system

- **Prior-day / prior-week sweep fuel** (`sweep2_pd_present` 27.7% p=0.98, N=101; `sweep2_pw_present`
  28.6% p=0.95, N=14): dead on baseline HERE. That is "no evidence in EURUSD Discovery," NOT "useless."
  Small N, one pair, one block. **Keep for other pairs, multivariate, and as LEVELS (not sweep-fuel).**
- **FVG size** (`fvg_size_atr`): no clean size gradient — every size bucket 28-37% WR, overlapping CIs.
  The signal is FVG **presence**, not size. Tiny FVGs are not worse than big ones. (Answers "should we
  bucket FVG by size" → no size lever found; presence lever kept above.)

---

## LONG-in-bullish-H1-trend bleed — deep dive

- **The cell:** LONG trade + bullish H1 trend → N 388, **WR 24.2%**, sumR −106 (worst cell in the book).
- **Cause = LATE / INTERNAL-STRUCTURE entry, not premium:**
  - By `bos_sequence_count` (how many breaks deep = how late/internal): seq 0/1 → ~28.8% (fine);
    seq 2 → 24.8%; **seq 3 → 18.5%** (p=0.053). Deeper = worse.
  - Within the cell: early (seq ≤1) 26.2% vs **late (seq ≥2) 21.2%** (dies fast 51% of the time).
  - Premium tested and REJECTED as the driver: within the cell, premium 24.5% vs discount 23.7% —
    both bad, no gap. Book-wide premium hurts (−3.8) but it is NOT what drives this cell.
- **Direction context, `pd_pct` (0=range low/cheap, 100=range high/expensive):** for LONGS, buying LOW
  in range 32.7% vs buying HIGH 23.2% (p=0.027) — textbook "don't chase up into premium." For SHORTS
  the mirror is weak. So pd_pct is mainly a long-side "don't buy the high" signal.
- **LOG STATUS:** stored for further analysis (multivariate + Validation). Mechanism: late internal
  longs in an up-trend = chasing, buying used-up structure.

---

## Two DIFFERENT "equal-level sweep" ideas — do not conflate

1. **FUEL sweep — `sweep2_eq_present`** ([liquidity_sweep.py:358-376](../../liquidity_sweep.py#L358)):
   the leg that BUILT the OB swept an equal-highs/lows shelf behind it. A **confluence/amplifier**
   (grabbed stops, then reversed into our OB). This is the 48% (N=25, thin) row above.
2. **ENTRY-LEVEL sweep — the OB's own extreme sitting ON an equal shelf / the extreme itself swept**
   (`setup_liq_legextreme_swept`): the level we ENTER from is itself resting liquidity / was itself
   swept → **warning** (we're a target, or on the wrong side). This is the 22.8% (N=114) row.
   **Owner is testing this cleanly in the stop-loss-swept chat** — do not pre-conclude here.

Mechanically OPPOSITE: (1) sweep FEEDS the OB from behind (good); (2) sweep is AT the entry (bad).

---

## Open, deliberately NOT concluded here (going to their own fresh chats)

- **Instant-death pile** (loss with `bars_to_exit≤1`): being tested as its own question in a fresh chat.
  **No verdict logged** on purpose (would bias that chat). Screen every feature there, open-minded.
- **Stop-hunt pile** (`sl_swept_then_1r`): likewise its own fresh chat; includes the entry-level
  equal-sweep idea (#2 above) and stop-placement geometry.

---

## Next: multivariate (in-chat, step by step — no edge_engine)

Single features barely separate (expected). Next step is combinations/context via a decision tree /
gradient-boosted trees (XGBoost) / L1-logistic, cross-validated, features read and SMC-checked, trained
ONLY on Discovery (Validation/Holdout stay sealed). Carry into it: FVG-present, swing-sweep, patient
pullback, biggest-break, leg-extreme-swept, late-internal-long, pd_pct-by-direction.
