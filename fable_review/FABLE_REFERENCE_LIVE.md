# FABLE REFERENCE — Live H1 System (Phase 1 + Phase 2)

This file is the **ground truth** about a live Smart Money Concepts (SMC) trading-alert
system. You (Fable) cannot see the repo and cannot run code. Everything below was read
line-by-line from the actual source by an engineer with repo access (Claude Code) and
verified. **Design and critique against THIS, not against assumptions.** Where a code
comment and the code itself disagree, the CODE wins — and those disagreements are called
out explicitly (search "DOC-DRIFT").

## Your job
Two separate reviews will be run against this same file: one on **trading logic /
methodology**, one on **architecture / maintainability**. Each review has its own prompt.
Read this whole file first either way. You produce **critique + ranked improvement options
+ edge cases**, as a SPEC the engineer brings back — **NOT final code**. Change zero
trading logic yourself; you are reasoning, not implementing.

## Scope — what is IN and OUT
- **IN:** Phase 1 (H1 structure + Order Block detection) and Phase 2 (H1 alert / scoring /
  level computation), plus the surrounding state, orchestration, chart and email layer.
- **OUT — do not analyse or mention:** M5/M15 timeframes, "Phase 3", and the backtest
  engine. The system is **H1-only**. Any M15/M5 parameter you see referenced
  (e.g. `FVG_WINDOW_M15_CANDLES`) is dead in the live H1 path — ignore it.

---

## Changes already applied (2026-06-10) — review the FIXED system, not these bugs
These were found in the first pass and already fixed in code. They are listed so you don't
re-flag them; you MAY still critique the fixed design.
- **Invalidation window aligned.** Phase 2's still-alive gate now starts mitigation at the
  structural-event candle + 1 (BOS/CHoCH + 1), matching Phase 1. It no longer scans the
  impulse leg (OB+1). All three call sites now use event-candle+1. (§5.1)
- **DOC-DRIFT removed.** OB mitigation is WICK-based (a wick to the distal line kills the
  zone; 3 proximal wick-touches = exhausted). Docstrings/comments that said "close beyond
  distal" were corrected to match the code. The wick-based rule is intentional and kept. (§4.9)
- **Email touch clarity.** Drop/invalidation emails now name the line explicitly ("wick hit the
  DISTAL line", "PROXIMAL line wicked 3×") and the active-zone status reads
  "Tested (Nx proximal)". (§4.11, §5)
- **ATR de-duplicated.** `dealing_range._compute_atr` now delegates to the single cached
  `smc_detector.compute_atr` (lazy import) with a raw fallback. One ATR implementation. (§3)
- **News filter rewired (see §5.5).** Live Phase 2 now uses the real ForexFactory scheduled
  high-impact calendar (`news_filter.py`), fetched once per scan, mapped to each pair's
  currencies, surfaced as a deterministic blackout/next-event banner. The old generic-RSS
  scrape is gone; Gemini now summarises the real scheduled events (demoted to secondary colour).

Still OPEN (fair game, NOT yet fixed): BOS-sequence-count computed two ways (§6), two H1 chart
renderers (§6), no score gate (§6), non-JPY-forex sweep collapse (§6), and all methodology
questions.

---

## 0. System in one paragraph

Automated SMC alert system. **H1 timeframe only.** 6 instruments: EURUSD, NZDUSD, USDJPY,
USDCHF, GOLD (XAUUSD), NAS100. Two cron jobs:
- **Phase 1 (`smc_radar.run_radar`)** runs hourly: fetches H1 data, computes H4 dealing-range
  walls + H1 swing structure (trend, BOS, CHoCH), builds Order Blocks (OBs) from the
  structural-event ring, persists them to a "slate" (`active_obs.json`), and emails a scout
  digest.
- **Phase 2 (`Phase2_Alert_Engine.py`)** runs on its own schedule: reads the slate, checks
  whether price is approaching an OB (proximity gate), checks the OB is still alive, scores
  confluences, computes entry / SL / TP, and emails a "TRADE READY" alert.

The goal is to **replicate veteran SMC judgment, not generate noise.** The owner is a
discretionary-trained SMC trader; he wants a vet to "respect the signal." Anti-sycophancy is
a hard rule on this project — be brutally honest, flag weak spots, do not soften.

---

## 1. Instruments and per-pair config (`config.json`)

Account: balance $50,000, risk 0.5% per trade ($250). Firm: Funding Pips.

| Name | yf symbol | pair_type | atr_multiplier | decimals | spread_pips | killzones (UTC) |
|------|-----------|-----------|----------------|----------|-------------|-----------------|
| EURUSD | EURUSD=X | forex | 2.5 | 5 | 2.0 | 07–10, 12–17 |
| USDJPY | JPY=X | forex | 2.5 | 3 | 2.0 | 03:30–07, 07–10, 12–17 |
| NZDUSD | NZDUSD=X | forex | 2.5 | 5 | 4.0 | 03:30–07, 07–10, 12–17 |
| USDCHF | CHF=X | forex | 2.5 | 5 | 2.0 | 07–10, 12–17 |
| NAS100 | NQ=F | index | 3.0 | 2 | 2.0 | 13–21 |
| GOLD | GC=F | commodity | 3.0 | 2 | 0.50 | 07–10, 12–21 |

- `atr_multiplier` is the **Phase 2 proximity cap** (how close, in ATR units, price must come
  to an OB's proximal line before an alert fires). It is NOT a swing/structure knob.
- `entry_model` is `"limit"` for all pairs.
- Scoring config: `hard_gates: ["rr_minimum_1.5"]`, `news_blackout_hours_before: 2`,
  `news_blackout_hours_after: 1`, `zone_fatigue_threshold: 5`. (Note: only the 1.5R gate is
  actually enforced in the live H1 path — see §6.3. Treat the others as declared-not-wired
  until proven otherwise; flag if so.)

---

## 2. Data flow and state files

```
Phase 1 (hourly):
  fetch H1 (15d window)  →  compute_pair_walls(df)               [smc_radar.py:1009]
       → compute_h4_range(df)         H4 walls                   [h4_range.py:183]
       → compute_structure(df, h4)    trend + BOS/CHoCH ring     [dealing_range.py:440]
     writes →  state/structure_state.json   (per-pair walls + event ring + swings)
  detect_smc_radar(df, events, walls) builds OBs                 [smc_radar.py:501]
     reconcile fresh OBs with slate (match / add / drop)
     writes →  active_obs.json   (the "slate" — surviving OBs, persisted across days)
  email scout digest (gated: ≥100 min since last email)

Phase 2 (its own schedule):
  read active_obs.json (slate)
  per OB: proximity gate → still-alive gate → score → levels → re-email decision
     writes →  phase2_sent.json        (per-zone dedup / re-email state)
              active_watch_state.json  (watch state, GC'd at 15d)
  email TRADE READY
```

**State files (JSON, on disk):**
- `state/structure_state.json` — per pair: trend, H4 walls, the **event ring** (last 20
  BOS/CHoCH events), confirmed swings. Written by Phase 1 only; read by Phase 2 (for trend +
  BOS count) and by the OB builder.
- `active_obs.json` — the **slate**: surviving OBs per pair, with `zone_id`, status, touches,
  FVG/sweep snapshots, dealing-range snapshot. Persists across trading days.
- `phase2_sent.json` — per-zone re-email state (score watermarks, last_email_day,
  reentry_armed, max_exit_distance). GC at 7 days idle.
- `active_watch_state.json` — Phase 2 watch state. GC at 15 days from first_seen.
- `email_gate.json` — last digest email timestamp (Phase 1's 100-min email throttle).
- Several `*_log.json` / `*.jsonl` — scan logs, heartbeat, failure logs (forensics).

**Key consequence:** Phase 1 and Phase 2 are **decoupled processes** that communicate only
through JSON files. Phase 1 owns detection and the canonical "is this OB dead?" decision;
Phase 2 re-derives some of that between Phase 1 cycles (see §6.2 still-alive gate) and that
re-derivation is NOT identical to Phase 1's (see DOC-DRIFT and §8).

---

## 3. ATR and the full knob inventory (verified file:line + value)

### ATR — single implementation (de-duplicated 2026-06-10)
- `smc_detector.compute_atr(df, period=14)` [smc_detector.py:96] — mean of True Range over
  last `period` bars. **Memoised** on an OHLC fingerprint (`_ATR_CACHE`). The single source.
- `dealing_range._compute_atr(df, period=14)` [dealing_range.py:95] — now a thin delegator to
  `smc_detector.compute_atr` (lazy import to avoid the load-time circular import), with a raw
  fallback so it still never raises. Verified to return identical values. One concept, one body.

### Knob table (constant, file:line, value, what it gates, what it can move)

| # | Constant | file:line | Value | Gates | What it moves |
|---|----------|-----------|-------|-------|---------------|
| 1 | `SWING_LOOKBACK` | dealing_range.py:59 | 3 | swing pivot geometry (bars each side) | everything downstream |
| 2 | `MIN_LEG_ATR_MULT` | dealing_range.py:70 | 1.5 | min swing-leg size in ATR | which swings survive → structure, OBs, alerts |
| 3 | `BOS_ATR_MULT` | dealing_range.py:75 | 0.4 | BOS displacement past broken swing; also Range-BOS wall proximity | BOS count, OBs |
| 4 | `STRUCTURE_CHOCH_ATR_MULT` | dealing_range.py:422 | 1.0 | CHoCH displacement past defended swing | CHoCH count, trend flips, OBs |
| 5 | `STRUCTURE_LOCK_ATR_MULT` | dealing_range.py:428 | 1.5 | CHoCH failure-window lock distance | CHoCH confirm vs revert |
| 6 | `STRUCTURE_RANGING_STALE` | dealing_range.py:432 | 2 | "ranging" flag (informational) | label only |
| 7 | `PREMIUM_PCT` / `DISCOUNT_PCT` | dealing_range.py:80–81 | 0.75 / 0.25 | premium/discount zone lines for CHoCH-from-zone tag | `reversal_pct` tag only |
| 8 | `EVENT_RING_MAX` | dealing_range.py:86 | 20 | max events kept in ring | how far back OBs can be built |
| 9 | `OB_MAX_RANGE_ATR_MULT` | smc_detector.py:177 | 2.0 | reject oversized OB candle (news bar) | which candle becomes the OB |
| 10 | `FVG_NOISE_FLOOR_MULT` | smc_detector.py:149 | forex .08 / index .15 / commodity .12 | min FVG size (× ATR) | FVG presence → FVG score |
| 11 | `FVG_WINDOW_H1_CANDLES` | smc_detector.py:169 | 10 | soft cap on FVG search window past OB | FVG presence |
| 12 | `SWEEP_EQUAL_LEVEL_TOLERANCE_ATR` | smc_detector.py:199 | forex .30 / idx .40 / comm .40 | "equal highs/lows" tolerance + context tags | sweep equal-levels sub-score |
| 13 | `SWEEP_WICK_PIERCE_MIN_ATR` | smc_detector.py:209 | forex .05 / idx .08 / comm .08 | min wick pierce past level to count as a sweep | sweep detection |
| 14 | `ROUND_NUMBER_GRID` / `_TOLERANCE` | smc_detector.py:219 / 225 | see source | round-number context tag | sweep narration tag only |
| 15 | `SWEEP_SCORE_BASE_MAX` / `_EQUAL_LEVEL_MAX` / `_REJECTION_MAX` | smc_detector.py:260–262 | 1.5 / 0.5 / 1.0 | sweep score caps (sum 3.0) | sweep score |
| 16 | `OB_PROXIMITY_ATR` | smc_radar.py:68 | 5.0 | Phase 1 OB surfacing window (× ATR) | which OBs surface to the slate |
| 17 | `OB_MAX_KEEP` | smc_radar.py:69 | 2 | max OBs surfaced per pair | always ≤2 OBs reach Phase 2 |
| 18 | `OB_MAX_AGE_DAYS` | smc_radar.py:96 | 15 | hard age cap on slate OBs | when a stale OB is retired |
| 19 | `EMAIL_GATE_MINUTES` | smc_radar.py:83 | 100 | min minutes between Phase 1 digests | email cadence only |
| 20 | `ZONE_PROXIMITY_THRESHOLDS` | smc_radar.py:53 | fx .0003 / idx 30 / comm 3.0 | slate match + dedupe "same zone" distance | zone identity / dedupe |
| 21 | `STALENESS_HOURS` | smc_radar.py:74 | 1h: 2.0 | data-staleness skip threshold | skips scan if data old |
| 22 | `atr_multiplier` (per pair) | config.json | fx 2.5 / idx+comm 3.0 | **Phase 2 proximity cap** | which OB-touches fire an alert |
| 23 | Phase 2 re-email | Phase2_Alert_Engine.py:86–95 | up +0.7 / down −0.5 / reentry 1.5× | re-email hysteresis + re-entry arm | email cadence, not signal |

Nothing is off-limits to challenge, including the structure constants (#1–5). H1-only is locked.

---

## 4. PHASE 1 — detection, function by function

### 4.1 Swing detection — the single swing definition
`detect_swings(df, lookback=3, min_leg_atr_mult=1.5, right_lookback=None)` [dealing_range.py:238]
- A candle `i` is a **swing high** if its High is **strictly greater** than every High in
  `[i-lookback, i+right_lookback]`; **swing low** is the strict-min mirror. Equal highs/lows
  do NOT register (flat tops produce no swing).
- `right_lookback` defaults to `lookback`. A smaller right_lookback lets the newest pivot be
  confirmed sooner (used by H4 walls — see §4.3). H1 uses the symmetric default.
- After geometry, the **ATR leg-size filter** is applied:
  `_filter_swings_by_leg_atr` [dealing_range.py:189]. A swing is kept only if the leg from the
  previous kept opposite-type swing is `>= 1.5 × mean_TR_across_that_leg`. The first swing
  passes free; consecutive same-type swings bypass the test; a dropped swing does NOT advance
  the reference anchor. **This is the single source of truth for the swing definition.**
- `smc_detector.get_swing_points` [smc_detector.py:423] is a **thin wrapper** that calls
  `detect_swings` (it does NOT re-implement geometry — verified).
- A swing becomes "known"/confirmed only `lookback` bars after it forms
  (`_structure_confirm_idx` = `idx + lookback`) [dealing_range.py:435].

### 4.2 ATR
See §3. `compute_atr` (cached) vs `_compute_atr` (uncached duplicate inside the structure engine).

### 4.3 H4 dealing range — `compute_h4_range(df)` [h4_range.py:183]
- `build_h4(df)` [h4_range.py:101] resamples H1→H4 on a 4h grid, **gap-aware**: a new H4 candle
  starts on a bucket change OR a >1h gap (so no H4 candle spans a weekend/session break).
- Detects swings on the H4 bars (same lb-3 + ATR-leg definition).
- Two passes:
  - **symmetric** pass (lb both sides) → drives the **frozen `confirmed_*` range** that the
    CHoCH premium/discount gate reads. Stays stable, does not jitter.
  - **early** pass (`right_lookback=1`, [h4_range.py:248]) → lets the **live wall** advance to a
    more-recent pivot sooner, without confirmation lag.
- **Live wall vs confirmed wall:** when price trades beyond the most-recent confirmed wall,
  the live ceiling/floor "rides" the live extreme (`ceiling_broken`/`floor_broken` = True,
  ts = None). The `confirmed_*` block never rides — it only moves on fully-confirmed swings.
- Returns both: `ceiling/floor` (live) and `confirmed_ceiling/floor` (frozen). The CHoCH gate
  uses `compute_pd_confirmed` [dealing_range.py:937] which reads ONLY the confirmed block
  (premium line = floor + 0.75·width, discount line = floor + 0.25·width).

### 4.4 The structure engine — `compute_structure(df, h4_range)` [dealing_range.py:440]
**This is the heart of the system. ~500 lines, one function, one forward pass over every bar.
PURE: recomputes from the whole df each call, no incremental state.** Read this section
carefully — most trading-logic risk lives here.

Locked spec (decided with the trader, from the source comments):
- One trend state: `up | down | undefined` (undefined only before "birth").
- CHoCH flips the trend on its own candle. No transition state.
- Premium/discount is a **quality tag only**, never a flip gate.
- CHoCH failure = price closes back past the origin extreme (reclaim) OR runs one full
  structural leg (1.5 ATR) past the broken level (lock).
- Re-arm guard: an invalidated CHoCH direction cannot re-fire until a fresh confirmed swing
  forms in the reverted direction (`rearm_block_dir`).

The per-bar loop, in order (the ORDER matters — earlier blocks `continue` and pre-empt later):
1. **Update leg extremes** (`leg_extreme_high/low`) — running max/min since last reset.
2. **Failure window** [dealing_range.py:642]: if a CHoCH is unconfirmed (`prior_trend` set),
   either (a) price runs `lock_dist = 1.5·ATR` past the CHoCH level → CHoCH **locks**
   (confirmed, window closes), or (b) price closes back past `choch_origin` → CHoCH **reverts**
   (trend flips back, sets `rearm_block_dir`, emits a `CHoCH_FAILED` marker, `continue`).
3. **CHoCH** [dealing_range.py:681]: in an uptrend, a close `< defended − choch_disp`
   (`choch_disp = 1.0·ATR`) flips to down (and mirror). Records the broken swing **by exact
   ts** (`broken_swing_ts`), sets `prior_trend`, `choch_origin = leg_extreme`, increments
   `choch_flip_count`, pushes a CHoCH event. Gated by `rearm_block_dir`.
4. **BOS (continuation), fires ON-CLOSE** [dealing_range.py:739]: in a downtrend, a close
   `< bos_break_low.price − bos_disp` (`bos_disp = 0.4·ATR`) fires a BOS on that candle
   (mirror for uptrend). The target is the **most-recent confirmed** trend-direction swing;
   after firing it is cleared (`bos_break_low = None`) so the same swing can't re-fire, and
   re-seeds on the next confirmed swing. This is "Option B" — fire on close, no swing-confirm
   lag. (Earlier code reported a stale `lows[-2]` level; this was fixed.) A BOS whose broken
   swing is within `bos_disp` of the H4 wall is tagged `tier='Range'` (Range BOS).
5. **Birth (cold start)** [dealing_range.py:779]: when `state == undefined`, the first close
   beyond `recent_high`/`recent_low` sets the initial trend and emits a `BOS` (BOS_BIRTH).
6. **Ingest confirmed swings** [dealing_range.py:817]: for swings becoming known at this bar,
   append to highs/lows. In an uptrend a new confirmed swing high becomes the next BOS-up
   target; a new Higher-Low (`made_hl`) becomes the new `defended` swing for the CHoCH check
   and resets `rearm_block_dir`. Mirror for downtrend.

**Event ring** (`events_ring`, trimmed to 20) — the output every downstream consumer reads.
Each event [dealing_range.py:593]:
```
type:               'BOS' | 'CHoCH'
tier:               'BOS' | 'Range'  (for BOS); 'CHoCH' for CHoCH. NO Major/Minor in v2.
direction:          'bullish' | 'bearish'
candle_ts:          iso — bar the event fired on
impulse_start_ts:   iso — start of the impulse leg (for OB walk-back)
broken_swing_price: float — level broken
broken_swing_ts:    iso of the EXACT broken swing object, or None if a raw leg extreme
broken_was_wall:    True if tier == 'Range'
reversal_pct:       1.0 if reversed from premium/discount zone else 0.0
trend_after:        'bullish' | 'bearish'
```
Returns also: `state`, `trend` (bullish/bearish/None for Phase 2 compat), `choch_flip_count`,
`defended`, `flip_unconfirmed`, `ranging`, `swings` (confirmed pool, one tagged
`is_setup_break` for the chart).

### 4.5 `compute_pair_walls(df, pair_name)` [smc_radar.py:1009]
Assembler. Calls `compute_h4_range` then `compute_structure` (with **no knob args** → all
structure knobs use their def-time defaults). Packs trend, H4 walls (ceiling/floor +
placeholder flags), `last_event_*` (sourced ONLY from the ring's last entry, never from the
transient `last_bos` scratchpad — that fix is documented at [smc_radar.py:1035]), event ring,
swings, and the raw `h4_range`/`structure_v2` blocks. Never raises (degrades to cold-start dict).

### 4.6 OB building — `detect_smc_radar(df, pair_type, events, walls, pair_name)` [smc_radar.py:501]
**No detection here — the event ring is the source of truth.** For each event in the ring:
1. Resolve `bos_idx` and `impulse_start_idx` from absolute timestamps via `_idx_from_ts`
   [smc_radar.py:569] — a **linear scan of the whole df per lookup** (O(events × bars)). Events
   referencing candles outside the current df window are dropped (`event_outside_window`).
2. **Walk back** from `bos_idx-1` to `impulse_start_idx` to find the OB candle [smc_radar.py:701]:
   the **first opposing candle** from the break (last down-candle before a bullish impulse;
   last up-candle before a bearish impulse). Two rejections while walking:
   - **oversized guard:** candle range `> OB_MAX_RANGE_ATR_MULT(2.0)·ATR` → skip (news bar).
   - **doji/body guard** (`is_valid_ob_candle`, body ≥20% of range [smc_radar.py:420]) → skip.
   If none qualifies → `no_qualifying_ob_candle`.
3. OB geometry: `proximal_line` = OB high (bullish) / low (bearish); `distal_line` = the other.
4. **Proximity gate** [smc_radar.py:742]: drop if `|price − proximal| > OB_PROXIMITY_ATR(5.0)·ATR`
   — **EXCEPT the last-event OB (OB1)**, which surfaces regardless of distance (trader decision).
5. Attach **FVG** (`detect_fvg_in_zone`, window `[ob_idx, bos_idx+1]` soft-capped at +10), a
   **sweep snapshot** (`observe_phase1_sweep`, window = the OB's own impulse leg), and a
   **dealing-range snapshot** (`get_dealing_range`). All three are frozen on the OB; Phase 2
   consumes them and does NOT recompute.
6. **Mitigation + touches** (`is_ob_mitigated_phase1`, from `bos_idx+1`) [smc_radar.py:887]:
   sets `touches` and `status` (Pristine / Tested Nx); mitigated OBs are dropped.
7. **Same-leg dedupe** (`_dedupe_same_leg` [smc_radar.py:922]): OBs in the same direction with
   proximal lines within `ZONE_PROXIMITY_THRESHOLDS` are one zone; pick winner by ladder
   **Pristine > FVG-holder > Freshest (higher bos_idx) > first**.
8. **Split to two OBs** (`_split_primary_alternative` [smc_radar.py:1107]):
   - **OB1 = primary** = the OB from the most recent event (highest `bos_idx`). **Ungated** by
     proximity — always surfaces while it exists.
   - **OB2 = alternative** = the **closest** OB to current price across ALL directions
     (buy/sell compete), proximity-gated at 5·ATR, pristine breaks ties, OB1 excluded.
   So **at most 2 OBs per pair reach the slate / Phase 2.**

### 4.7 Sweep observation — `observe_phase1_sweep(...)` [smc_detector.py:916]
Snapshot of the validating liquidity sweep, frozen on the OB. Window = **the OB's own impulse
leg** `[impulse_start_idx, ob_idx]` (locked 2026-06 after testing on USDJPY — the prior
"reach back to the prior opposing event" rule grabbed unrelated old liquidity). Logic:
- A qualifying sweep candle: bullish OB → low pierces a prior **active** swing low by
  `≥ SWEEP_WICK_PIERCE_MIN_ATR·ATR` AND closes back above it (mirror for bearish).
- **Target eligibility:** only **ACTIVE** swings qualify — `is_swing_active` [smc_detector.py:596]
  = unbroken (no close beyond) AND unswept (no wick pierce ≥ pierce_min) between the swing and
  the candidate. Drained levels carry no liquidity → not a valid target.
- **Survivorship check** [smc_detector.py:1099]: the sweep candle's extreme must remain the
  leg's extreme through to the OB candle; if a later candle wicks strictly deeper, this
  "sweep" was just a wick on the way to a real extreme → rejected.
- **Selection:** highest-scored candidate wins (tie → more recent). The OB candle itself may
  be the sweep candle (engulfing / rejection-block).
- **Score (max 3.0)** = `base 1.5` (presence) + `equal_levels` (0 / .25 / .5 for 0/1/2 active
  same-type levels within tolerance; `_equal_levels_score` [smc_detector.py:648]) + `rejection`
  (0 / .33 / .66 / 1.0 by wick:body ratio <1 / 1–2 / 2–3 / >3; `_rejection_score`
  [smc_detector.py:714]). Tier: textbook ≥2.4 / decent ≥1.8 / weak >0 / none.
- **Context tags** (`_compute_context_tags` [smc_detector.py:871]): round-number, prior-day
  H/L, session H/L — narration only, not scored.

### 4.8 FVG detection — `detect_fvg_in_zone(...)` [smc_detector.py:1224]
3-candle fair-value gap inside the displacement leg. Scan **oldest-first**, return the FIRST
**live** (pristine/partial) FVG (closest to the OB on retrace); if none, fall back to the most
recent **ghost** (fully-mitigated) for chart context.
- Gap must be `≥ FVG_NOISE_FLOOR_MULT·ATR`.
- **Session-gap guard** [smc_detector.py:1313]: rejects 3-candle patterns where the bar spacing
  exceeds 1.5× the median bar size (weekend/holiday gaps are not real FVGs).
- Mitigation states: `pristine` (untouched) / `partial` (proximal touched, distal not — still
  full score) / `full` (dead). **Full mitigation is pair-aware**: forex = touch-based (any wick
  through distal kills it); index/commodity = close-based (only a close past distal kills it,
  because gold/NAS wick through on news). Partial is touch-based for all.

### 4.9 OB mitigation — `is_ob_mitigated_phase1(...)` [smc_detector.py:2086]
Single source of truth for "is this OB dead?". Replays candles in `[start_idx, end_idx)`:
- **WICK-based (intentional, docs now corrected).** A bullish OB dies when `L[m] <= distal`
  (a wick to/below the OB low — NOT a close); bearish mirror. The trader accepts this
  aggressive rule: a wick reaching the line has filled the resting orders. Docstrings that
  used to say "close beyond distal" were fixed to match the code.
- A wick into `proximal` counts as a touch; **3 touches → mitigated** (`mitigated_three_touches`).
  Touches are PROXIMAL only (a distal hit is terminal). Emails now name the line hit.
- Methodology still open (§6): wick-distal kill is aggressive, and it's worth asking whether
  gold/NAS (which wick through levels on news, and get close-based FVG mitigation) should also
  get close-based OB-distal mitigation rather than wick-based.

### 4.10 `get_dealing_range(...)` [smc_detector.py:303]
Wrapper that returns the dealing range for display/snapshot. Prefers the H4 live range from
`structure_state.json`; falls back to a legacy fixed-lookback window high/low
(`DEALING_RANGE_LOOKBACK_H1` = forex 120h / index 72h / commodity 120h) when state is missing.

### 4.11 Phase 1 orchestration — `run_radar()` [smc_radar.py:3496]
- Blackout before 09:00 IST.
- Loads the slate; carries zones across trading days; marks all `is_new_this_scan=False`.
- Email gate: only sends a digest if ≥`EMAIL_GATE_MINUTES`(100) since last.
- Per pair: `fetch_data` (15d H1) → `compute_pair_walls` → **save `structure_state.json`** →
  `detect_smc_radar` → reconcile fresh OBs with slate:
  - match (`find_matching_slate_zone`) → `refresh_slate_zone`; else add (`fresh_to_slate_zone`).
  - unmatched active slate zones → `determine_drop_reason` [smc_radar.py:3293]. Drop reasons:
    `aged_out_of_window` (15d), `out_of_proximity` (>5·ATR), `mitigated_distal_break` /
    `mitigated_three_touches`, `structure_supplanted` (a fresher same-direction OB took over).
    **If no concrete reason fires, the zone is KEPT** and logged (`log_unverified_drop_attempt`)
    + indices re-synced (`resync_slate_zone_indices`) — deliberately no silent "unknown" drops.
- If fetch fails for a pair, slate zones are **held** (not dropped) and tagged.
- Builds and emails the master digest (summary table + active/new/dropped/invalid cards).

---

## 5. PHASE 2 — alert engine, function by function (`Phase2_Alert_Engine.py`)

Entry point at [Phase2_Alert_Engine.py:1739]. Per scan:

### 5.1 Gates before scoring
1. **Phase-1 freshness gate** [line 1748]: if `active_obs.json` hasn't refreshed within the
   freshness window during market hours, send ONE stale-warning email and exit (never alert on
   stale P1 data). Self-clears on recovery.
2. Load slate, watch state, dedup state; run GC on each (watch 15d, dedup 7d).
3. Per pair: fetch **30d H1**, current price = last H1 close; also grab the forming bar's
   high/low for wick-aware proximity. Compute `h1_atr`. `bos_counter =
   compute_bos_sequence_count(name)` [smc_detector.py:504] reads the event ring from
   `structure_state.json`.
4. **Proximity gate (wick-aware)** [line 1959]: distance = `|closest_point − proximal|` where
   closest_point uses the forming bar's low (LONG) / high (SHORT) so an intrabar wick toward
   the OB still fires. `prox_cap = atr_multiplier · h1_atr` (2.5 fx / 3.0 idx+comm). If
   `distance > prox_cap` → not dropped from slate (Phase 1 owns deletion); instead the dedup
   entry is kept alive, `max_exit_distance` tracked, and **re-entry armed** once price pulls
   beyond `1.5 × cap`.
5. **Still-alive gate** [line ~1985]: locate the structural-event candle by `bos_timestamp`
   on P2's frame, then `is_ob_mitigated_phase1` from **event-candle + 1** (BOS/CHoCH + 1). If
   mitigated → drop without alerting. **(Fixed 2026-06-10 — was OB+1.)** This now matches Phase
   1 (`detect_smc_radar` mitigation from bos_idx+1, `determine_drop_reason` after-BOS), so the
   two processes agree on whether a zone is alive. Falls back to OB ts only for a legacy zone
   missing `bos_timestamp`.

### 5.2 Levels — `compute_phase2_levels(...)` [smc_detector.py:1453]
- Entry: OB **proximal** edge (live). `entry_zone="50pct"` (OB midpoint) exists but is
  backtest-only — ignore.
- SL: OB **distal** ± 1× spread (spread = `spread_pips · pip_unit`, pip_unit by decimals).
- **Limit-chase guard:** if the proximal entry sits on the wrong side of current price (LONG
  entry above price / SHORT below), the alert is invalid (the limit would chase). Tolerance
  0.5× spread.
- **TP1** = nearest opposing H1 swing (lb-3) past entry that clears **1.5R**. **If none clears
  1.5R → NO TRADE** (the only hard gate in the live path). **TP2** = next opposing swing past
  TP1 (no RR gate); if none, ride TP1 + break-even policy. Post-rounding collision check drops
  TP2 if it collapses onto TP1.

### 5.3 Scoring — `run_scorecard(...)` [smc_detector.py:1601]
Confluence score, **computed and emailed but NEVER used as a gate** (no score threshold blocks
an alert in the live path). Components:
- **Structure (max 4):** CHoCH → 4; Range BOS → 4; plain BOS #1–2 since CHoCH → 3; mid-trend →
  2; BOS ≥ caution threshold (forex 3 / index 5 / commodity 4) → 1. (BOS sequence count read
  from the event ring.)
- **Sweep:** consumed from Phase 1's frozen snapshot (`ob['sweep_observed']`), NOT re-detected.
  **Asymmetric by pair** [line 1682]: **non-JPY forex collapses sweep to presence-only**
  (1.0 if a qualifying sweep exists else 0.0 — quality discarded); JPY / Gold / NAS keep the
  full graded 0–3.0. Rationale in source: spot forex has no centralized stop pool, so sweep
  quality is noise there.
- **FVG (max 2):** pristine 2 / partial 1 / mitigated-or-none 0 (ghost = 0). H1 only.
- **Freshness (max 1):** 0 touches → 1; 1+ touches → 0 (binary).
- **PD = 0, Killzone = 0** [lines 1745–1752]: both REMOVED from scoring but still computed and
  rendered as display-only. (Killzone removed because the hard killzone filter means 100% of
  alerts would have scored it — "a confluence that fires for every alert is noise.")
- **Totals:** non-JPY forex max **8**; JPY/Gold/NAS max **10**. Subject line shows a /10-
  normalized score for cross-pair comparability.

### 5.4 Re-email model (hysteresis) [lines 86–95, 2128–2256]
Four triggers, in priority order: **fresh** (first ever) → **reentry** (left proximity >1.5×
cap and returned) → **still_valid** (new trading day, still in proximity, any score) →
**updated** (same day, score crosses +0.7 up / −0.5 down). Same-day in-band re-sightings stay
silent. Score watermarks (`score_high_water`/`low_water`) persisted; `last_seen_ist` refreshed
even out-of-proximity so a live-but-distant zone is never GC'd.

### 5.5 News (rewired 2026-06-10 — now the real scheduled calendar)
- **Source:** `news_filter.py` — ForexFactory high-impact economic calendar (NFP, CPI, FOMC,
  ECB, BoE, etc.), parsed to UTC, per-currency. `currencies_for_pair` maps each pair to its
  currencies (e.g. EURUSD → {USD, EUR}; GOLD/NAS100 → {USD}). This module was previously wired
  only into the backtest; it is now wired into live Phase 2.
- **Fetch once per scan** (`fetch_scheduled_news`) over only the flag window
  `[now − (after+0.5h), now + (before+0.5h)]`; shared across all 6 pairs (feed is per-week).
- **Per-pair context** (`get_pair_news_context`): slices the shared list to the pair's
  currencies and computes the **asymmetric blackout** flag (config: 2h before / 1h after).
  The ONLY thing flagged is "now is inside [event−before, event+after]". Events outside that
  window are deliberately NOT surfaced (an earlier 24h "next event in Xh" banner flagged on
  every event — e.g. an ECB print 7h out — and was removed as noise 2026-06-10).
- **Email banner** (deterministic, primary): RED "NEWS BLACKOUT" inside the window; GREEN
  "clear — no high-impact event within the window" otherwise; RED "calendar incomplete — check
  manually" if the fetch failed (clear ≠ unchecked).
- **Gemini** (`call_gemini_flash`) is kept but **demoted to secondary "Macro colour (AI)"** and
  now receives the REAL scheduled events as input (via `fetch_macro_news(name, news_ctx)`)
  instead of a generic global RSS — so its summary is finally about pair-relevant events.
- **The old path is gone:** the generic `forexlive.com/feed/news` scrape (unfiltered, last-10
  headlines, no pair/impact/time filter) was the reason "not much relevant news" showed up.
- **Still open to critique:** blackout does NOT block the alert (informational banner only) —
  is that the right call for a discretionary trader, or should a blackout suppress/flag harder?
  And FF covers scheduled releases only (no unscheduled geopolitical shocks).

### 5.6 Charts + email (cosmetic, but large)
- **Two H1 chart renderers exist:** `smc_radar.generate_h1_chart` [smc_radar.py:1170] (Phase 1
  digest) and `Phase2_Alert_Engine.generate_h1_chart` [Phase2_Alert_Engine.py:814] +
  `generate_h1_zoomed_chart` [line 651]. They must be kept in visual sync by hand.
- Email: Phase 1 `send_master_digest_v2` [smc_radar.py:2827]; Phase 2 `build_trade_email`
  [Phase2_Alert_Engine.py:1169] + `send_email` (SMTP via Gmail app password).
- A **heartbeat** subsystem [Phase2_Alert_Engine.py:1395–1735] emails system health every few
  hours and on failure patterns.
- The Phase 2 trade email renders, in order: action block → trend banner → distance → zone
  context → scorecard → H1 context chart → H1 zoomed chart → sweep breakdown → **news banner
  (deterministic FF calendar, §5.5)** → macro colour (AI, secondary). `build_trade_email`
  [Phase2_Alert_Engine.py:1169]. (These builders have been fully read — no blind spots.)

---

## 6. Honest list of suspected weak / risky / duplicated spots (starting points, not gospel)

Verify each against the file:line before trusting. These are leads for both reviews. Items
marked FIXED are done — listed only so you understand the current state.

1. ~~Mitigation start-index inconsistency~~ **FIXED** — Phase 2 now uses event-candle+1 (§5.1).
2. ~~DOC-DRIFT on distal mitigation~~ **FIXED** — docs match the wick-based code (§4.9).
3. **Wick-based distal kill, methodology question (OPEN).** A single wick to the distal kills
   the zone for ALL pairs, yet FVG mitigation is close-based for gold/NAS (they wick through
   on news). Is wick-distal too aggressive for gold/NAS specifically? Should OB-distal
   mitigation also be close-based there?
4. **`compute_structure` is one ~500-line stateful loop, recomputed from full df every scan.**
   Hard to test in isolation, high bug-surface; the block ORDER (failure-window `continue`
   pre-empts BOS/CHoCH) is load-bearing and subtle.
5. ~~Two ATR implementations~~ **FIXED** — single implementation via delegation (§3).
6. **Two H1 chart renderers** (smc_radar vs Phase2) — duplication, sync-by-hand. OPEN.
7. **`_idx_from_ts` linear scan per event in the OB builder** (O(events×bars)). OPEN.
8. **OB1 bypasses the proximity gate** — a far-away OB1 always surfaces. Signal or noise?
9. **No score gate anywhere** — the whole scorecard is advisory. (Owner's plan: add a gate
   after measuring whether score predicts outcome. Critique scoring's *construction* and say
   what to measure to decide a gate; don't assume one exists.)
10. **Non-JPY forex sweep collapse to presence-only** — deliberate; does discarding sweep
    quality on EURUSD/NZDUSD/USDCHF throw away edge?
11. **BOS sequence count has two computation paths (OPEN).** `detect_smc_radar` stamps a
    per-event count on each OB; Phase 2 then OVERWRITES both surviving OBs with the latest
    whole-ring count (`compute_bos_sequence_count`) before scoring. For OB2 (an earlier event)
    that overwrite is arguably wrong — its structure score uses the latest count, not its own.
    Worth deciding which source is correct per-OB.
12. **TP depends entirely on H1 swing availability** — thin-swing regimes silently produce
    "no trade." How often, and is that correct or a coverage hole?
13. **PD + killzone computed but scored 0** — dead weight in the scorer; wire meaningfully or
    stop computing.
14. ~~News filter context-only / generic-RSS~~ **FIXED** — real FF calendar wired (§5.5). The
    remaining open question is whether a blackout should merely warn or actively suppress.
15. **Declared-but-maybe-unwired config** (`zone_fatigue_threshold`) — verify it's consumed
    anywhere; if not, delete it.

---

## 7. Constraints for everything you propose
- **Change zero trading logic.** Output specs, options, trade-offs, edge cases — not code.
- H1-only is locked. Do not propose M5/M15/Phase-3/backtest changes.
- Data reality: yfinance H1, ~2 years max history; Phase 1 fetches 15d, Phase 2 fetches 30d.
- Every claim you make about behaviour must be checkable against a file:line in this doc; if
  this doc doesn't state something you need, say "I need X verified" rather than assume.
- Treat code comments with suspicion where this doc flags DOC-DRIFT.
- Be brutally honest. If a chunk isn't worth reviewing, say so. If you need a fact to reason,
  ask — the engineer will verify and answer.
