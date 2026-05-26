# Benchmarking — External SMC References

## Purpose

Catalog of ideas studied from external SMC repos and indicators. Tracks what was adopted, what's pending verification, and what was rejected (with reasons, so we don't re-evaluate).

**This is not a source of truth.** Every item here was written by someone outside this project. Every adoption must be verified against vet methodology, our existing code, and live testing.

Calibration questions (test X vs Y on historical data) live in BACKLOG.md, not here.

## Sources

- **joshyattridge/smart-money-concepts** — Python lib, ~1100 stars. Cloned to `..\smc-research\joshyattridge-smc\`.
- **LuxAlgo SMC indicator** — TradingView Pine, free. Community Enhanced fork at `..\smc-research\luxalgo\luxalgo-smc-enhanced.pine` (1129 lines).
- **smtlab/smartmoneyconcepts** — Python lib. Rejected wholesale (incorrect OB definition, deprecated numpy).

---

## 1. Implemented

### OB candidate range cap — 2x ATR maximum
- **Source:** LuxAlgo `ob_coord` function.
- **Rule:** OB candidate candle must have `(high - low) <= 2 × H1 ATR`. Otherwise skipped.
- **Purpose:** rejects volatility spikes (news bars) from the OB candidate pool.
- **Where:** `smc_detector.OB_MAX_RANGE_ATR_MULT = 2.0`; applied in `smc_radar.py` OB-walk loop.
- **Different from:** removed 1.5x median minimum filter (that one rejected small candles, this rejects oversized).

### PD-array gate on OBs (premium / discount)
- **Source:** vet methodology (also referenced in LuxAlgo's PD bands, though they use narrow 5% bands — see section 7).
- **Rule:** bullish OBs valid only if proximal sits in the discount half (≤ equilibrium) of the dealing range. Bearish OBs valid only in the premium half (≥ equilibrium). Applied to all OBs (BOS and CHoCH alike).
- **Failure mode:** fails open when wall geometry is unavailable (`compute_pd_position` returns `valid=False`) OR cold-start fallback range is active (`fallback_active=True`). We cannot trust the gate without real walls.
- **Where:** `smc_radar.detect_smc_radar` build loop, after `ob_proximal` is computed and before the proximity gate. Uses `dealing_range.compute_pd_position`.
- **Open question (BACKLOG):** should BOS-OBs in late-trend impulse legs be treated more leniently? Currently they get the same gate as CHoCH-OBs.

### OB tier labelling (BOS / Major CHoCH / Minor CHoCH)
- **Reasoning:** BOS detection is single-tier (always Major). CHoCH detection has a Major / Minor split: Major = opposite-wall break (trend flips); Minor = internal lookback=3 break after a wall touch within MINOR_CHOCH_WALL_TOUCH_ATR * ATR (trend weakening, does not flip). Both tiers use the same lookback=3 swing pool — geometry distinguishes them. The OB inherits this from its source event.
- **Where:** `smc_radar._event_label` helper. Used by Phase 1 chart title, Gemini prompts, and fallback narratives. Existing `_phase1_chart_legend_html` already applied the same labelling for chart legends.
- **Why it matters:** downstream display / scoring can distinguish a Minor-CHoCH OB (early-warning, trend not yet flipped) from a Major-CHoCH OB (genuine reversal anchor) without re-deriving from `bos_tag` + `bos_tier`.

---

## 2. Detection methodology comparison

### BOS — LuxAlgo
- Tracks most recent confirmed swing high (`top_y`) and low (`btm_y`) plus a trend integer.
- Fires structural event when a candle's CLOSE crosses above `top_y` (or below `btm_y`).
- Event type: BOS if trend already matched (continuation), CHoCH if trend opposed (flip).
- Runs on two tiers in parallel: `length=50` swings (major) and `length=5` internal.
- **No leg-displacement validation.** Any crossover fires.

### BOS — joshyattridge
- Requires a 4-swing alternating pattern (low-high-low-high) with valid level ordering before BOS can fire.
- Misses continuation breaks in strong trends (no return to make a clear higher low). **Rejected as methodology.**

### BOS — ours
- Crossover-based detection (same as LuxAlgo). ✓
- Adds 0.4× ATR leg-displacement gate past the broken wall (LuxAlgo lacks this). ✓
- Distinguishes range-wall breaks from internal-swing breaks (LuxAlgo's tiers fire structural events on internal too, just labeled). ✓
- Chop flag: CHoCH within 5 candles of prior event tagged for review. LuxAlgo has none. ✓

### CHoCH — handling of internal swings (the micro-swing scenario)

**LuxAlgo:** if a small swing (e.g. 130 in the example) qualifies the internal tier (length=5), internal CHoCH fires when it's broken. Major tier stays silent until a major wall breaks. Two tiers running in parallel handle this case.

**joshyattridge:** single lookback (50 default). The 130 swing does not qualify. They miss the break entirely.

**Ours:** single lookback=3 pool. Major CHoCH = opposite-wall break (trend flips). Minor CHoCH = internal lb-3 break gated by a wall-touch precondition (price tested the trend-direction wall within MINOR_CHOCH_WALL_TOUCH_ATR * ATR in the current leg); informational only — trend does not flip, walls do not move. Mid-range lb-3 breaks without a wall touch are noise and produce no event.

**Status:** implemented in `dealing_range._pick_choch_pivot`.

### Sweep — LuxAlgo (EQH/EQL)
- Pivot high/low with small lookback (3 each side).
- Compare current pivot to PREVIOUS pivot (single comparison).
- If `|current - previous| < ATR × 0.1` → equal levels.
- No explicit sweep-event tracking. When price exceeds, the level just deletes.
- **Simple. ATR-scaled. Two-pivot comparison only.**

### Sweep — joshyattridge
- For each swing high, scan forward for other swing highs within `range_percent × total_range`.
- Group multiple highs into a "liquidity zone" with averaged level.
- Track the candle index that exceeded `range_high` as the sweep candle.
- **More than two-level grouping.** But uses `%` of total range, which drifts with dataset size — wrong scaling.

### Sweep — ours
- ATR-scaled tolerance, **pair-aware** (forex 0.15, index/commodity 0.25 × ATR).
- "Two prior swings out of last 3 same-type swings" — multi-prior confirmation.
- Two recency windows: 10h grading, 72h observation.
- 3-component score: base presence + equal-level count + rejection wick:body ratio.
- M15-vs-H1 priority buffer (1.10).

**Honest read on complexity:** the ATR scaling, pair-awareness, and two-tier recency are all justified. The 3-component score is the most "extra" piece — could collapse to 2 components without losing core signal. If simplification is wanted, start there. Not stupid as-is.

---

## 3. Confirmed in our code — no action needed

### OB freshness scoring
- Both repos mark OBs as mitigated on first touch but keep tradeable until full close-through.
- Ours rejects on full distal touch (zero score), warns on partial proximal touch (0.25), full score only when pristine. Stricter than both repos. Methodology choice.
- Located: `smc_detector.py` lines 966-976.

### Last-opposing-candle rule for OB selection
- Both repos use lowest-low / highest-high. Wrong from vet methodology.
- Ours: most recent opposing candle. Correct. Located: `smc_radar.py` ~line 388.

### ATR-scaled equal-level tolerance for sweeps
- LuxAlgo: 0.1 × ATR. joshyattridge: % of total range (wrong scale).
- Ours: pair-aware 0.15 / 0.25 × ATR. Correct.

### CHoCH explicit trend state
- LuxAlgo tracks `trend` and `itrend` as integers. Cleaner than implicit state.
- Ours: explicit `trend` variable in `dealing_range.py` line 498-504. Matches LuxAlgo's pattern. Correct.

### FVG noise floor
- LuxAlgo: cumulative-average-body %. Ours: ATR-based.
- Both filter junk FVGs. Different methods, both valid. No change.

### Cross-TF cascade vs same-TF dual-tier structure
- LuxAlgo runs internal (5) and swing (50) in parallel on the same TF.
- Ours: H1 OB → M15 sub-OB nested inside the H1 OB zone (ICT-aligned cascade).
- Cascade is more correct for trading. Keep ours.

### Sweep recency windows (10h grading + 72h observation)
- Vet view: 10h grading is sharp. 72h observation is on the looser end of defensible. Two-tier design is correct.

---

## 4. To verify against our code

### Two-stage mitigation (tap vs full break)
- **LuxAlgo:** tap = mark mitigated, keep visible. Close-through = remove. Two distinct states.
- **Action:** confirm our OB lifecycle treats tap and full-break as separate states. (Likely yes given freshness scoring exists, but verify.)

### Strong vs Weak High/Low classification
- **LuxAlgo:** high made during downtrend = "Strong" (held). High made during uptrend = "Weak" (likely to break). Mirror for lows.
- **Status:** approved 2026-05-08, scheduled for build 2026-05-09. Tracked in BACKLOG. Will be a Phase 2 confidence input, not a Phase 1 hard gate.


---

## 5. Kill zones — current state

### User trading window
Weekdays 09:00–24:00 IST = 03:30–18:30 UTC. Hours outside this window are blackout (no trades fire).

### Our hours (UTC, post-tightening 2026-05-25)
| Pair | Scoring windows (`_killzone_hit`) | Hard-filter windows (`killzones_utc`) | IST equivalent |
|---|---|---|---|
| EURUSD, USDCHF | 07–10, 12–17 | 07–10, 12–17 | 12:30–15:30, 17:30–22:30 IST |
| USDJPY, NZDUSD | 07–10, 12–17 | **03:30–07:00**, 07–10, 12–17 | 09:00–12:30, 12:30–15:30, 17:30–22:30 IST |
| Index (NAS100) | 13–21 | 13–21 | 18:30–02:30 IST |
| Commodity (Gold) | 07–10, 12–21 | 07–10, 12–21 | 12:30–15:30, 17:30–02:30 IST |

Dead-hour cuts: London lunch (10–12 UTC) and forex post-NY drift (17–21 UTC) excluded from forex. NAS/Gold retain NY through cash close because they are NY-driven products.

JPY/NZD Asia-tail (03:30–07:00 UTC): kept as a hard-filter window only because Tokyo session drives JPY/NZD flow and overlaps the user's IST trading window (03:30 UTC = 09:00 IST). Scoring side is bucket-level and doesn't credit it — Asia-tail JPY/NZD alerts fire but lose the +0.5 killzone confluence point.

### Compared to standard SMC kill zones (UTC)
- London open (07:00–10:00) — covered ✓
- New York (12:00–15:00) — covered ✓
- London close (15:00–17:00) — covered ✓
- Asian (00:00–04:00) — excluded by user's trading window. Deliberate.

### Removed in conversion
- `ist_hour == 0` clause (forex/commodity) — was UTC 18:30–19:30. Outside user's trading window. Confirmed blackout, removed.
- `ist_hour <= 1` clause (index) — included hour 0 (UTC 18:30–19:30) and hour 1 (UTC 19:30–20:30). Both blackout. Removed.

### Deferred suggestions (not implemented)
- **Gold-specific Asian session inclusion (UTC 00:00–04:00).** Gold trades aggressively in Asia. Currently only relevant if user expands trading window to include Asia. Park.
- **Half-hour-precision boundaries.** Original IST code mapped to UTC 06:30/12:30/18:30 boundaries. Current UTC code is whole-hour. Marginal; widens kill zone by ~30 min on each side. If precision is wanted, switch from `utc_hour` integer to a minute-aware check.

---

## 6. Future enhancements (deferred)

### FVG join_consecutive
- joshyattridge. Merge back-to-back FVGs into one zone.
- Relevance: Gold/NAS impulsive moves stack 2-3 FVGs in one displacement.
- Defer until base FVG handling is reviewed end-to-end.

### Volume-weighted OB strength (NAS100, Gold only)
- joshyattridge. Imbalance % between absorption side and breakout side.
- Useless on forex (tick volume = noise). Potential confidence-score bonus.
- Defer until base scoring is stable.

---

## 7. Rejected — do not re-evaluate

### LuxAlgo same-TF two-tier (internal + swing on one chart)
- We cascade H1 → M15 (nested), more correct for trading. Same-TF parallel layers are a charting compromise.

### LuxAlgo one-sided forward swing detection
- Academic difference vs centered approach under strict equality. No measurable benefit.

### LuxAlgo Premium/Discount as top/bottom 5% bands
- Vet treats anything above 50% as premium, below as discount. LuxAlgo's narrow bands miss the actionable area. Our dealing-range PD is whole-range.

### LuxAlgo "Confluence Filter" on internal breaks (candle wick shape)
- Filter requires upper-wick > lower-wick for bullish breaks. Logic questionable / possibly inverted. Don't adopt.

### joshyattridge swing_length = 50 default
- Too coarse. Misses internal structure. Our cross-TF cascade handles granularity properly.
- **Why they chose 50 (researched 2026-05-26):** single-TF indicators (LuxAlgo, joshyattridge) lack a dealing range or cross-TF cascade. Lookback=50 on H1 ≈ ~2 days of bars ≈ approximates daily/H4 swings. They use wide lookback as a poor-man's higher-TF anchor. Also reduces Pine repaint on TradingView. Not relevant to us — we have a dealing range (currently H1-derived; D1 planned) + H1→M15 cascade. Our lookback=3 + ATR leg-size filter (see section 8) preserves intraday granularity while filtering noise.

### joshyattridge OB selection by lowest-low
- Doesn't enforce opposing-candle rule. We pick most-recent opposing candle. Correct.

### joshyattridge BOS 4-swing pattern requirement
- Misses continuation breaks. Crossover-based detection (ours and LuxAlgo's) is the correct approach.

### joshyattridge sweep tolerance as % of total range
- Drifts with dataset size. ATR-scaled (ours, LuxAlgo's EQH/EQL) is correct.

### smtlab/smartmoneyconcepts (entire repo)
- OB defined as "last candle before any FVG" — methodologically wrong. Deprecated numpy.

### DavidCico/Enhanced-Event-Driven-Backtester
- Event-queue framework for systems routing real orders. Overkill for "log would-be alerts to CSV." `backtesting.py` is the chosen harness (see BACKLOG.md).

---

## 8. ATR thresholds in external SMC libs — catalog

Research note (2026-05-26): documenting where public SMC implementations use ATR-based thresholds. Source-of-truth catalog so we don't re-search.

### LuxAlgo
| Use case | Threshold | Notes |
|---|---|---|
| Equal high / equal low tolerance (EQH/EQL) | `0.1 × ATR` | Two-pivot comparison. If `|current_pivot - previous_pivot| < 0.1 × ATR`, levels are equal. |
| OB candidate range cap | `2 × ATR` | OB candle's `(high - low)` must be ≤ 2× ATR. Rejects news/volatility-spike candles. **Already adopted — section 1.** |
| FVG noise filter | Cumulative-average-body % (NOT ATR) | LuxAlgo uses body-size % instead of ATR for FVG minimum size. |
| Zone overlap prevention | ATR-based threshold | Supply/demand zones around swing points use ATR-based spacing to avoid overlap. Exact multiplier not documented in the public Pine source. |
| Swing detection | **No ATR threshold.** Lookback only. |

### joshyattridge
| Use case | Threshold | Notes |
|---|---|---|
| Equal levels / liquidity zones | `% of total range` (NOT ATR) | Drifts with dataset size — rejected, see section 7. |
| Swing detection | **No ATR threshold.** Lookback only (default 50). |
| FVG / OB sizing | **No ATR threshold** in public API. |

### Vacuum confirmed
- **No public SMC implementation filters swing magnitude in ATR units.** Lookback is the only swing-validation mechanism in both major libs. Our planned `MIN_LEG_ATR_MULT` (see BACKLOG) puts us ahead of public standards on this dimension, not behind. Calibration is on us — no external default to copy.

### ICT trader practice (not code)
- Stop loss placement: 10–20 pips beyond a swing on FX (multiple ICT tutorials).
- Implied "real swing size" on EURUSD H1: 15–25 pips ≈ 1.5–2.5× H1 ATR (with current ATR ≈ 9.7 pips).
- This is loose evidence, not measurable code, but it's the only external anchor for what discretionary SMC traders treat as a tradeable swing.
