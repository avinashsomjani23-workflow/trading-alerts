> ⚠️ **STALE — DO NOT TRUST THE NUMBERS BELOW (flagged 2026-07-09).** This scorecard was
> generated 2026-07-08 13:14 UTC on an INTERMEDIATE `trades.csv` that the 2026-07-09 run
> (committed `1feb2db`, 00:05 UTC) has since REPLACED. Its discovery N=11,329 no longer
> exists; the current committed baseline is **11,366 discovery-eligible**. Step 2 MUST be
> re-run on the current CSV before its "entry null" counts. The METHOD here is still valid;
> only the numbers are stale. See `EDGE_LAB_MASTER_HANDOFF.md` §4 Step 1.

# Edge Lab v2 — Step 2 (Univariate Pooled Discovery) HANDOFF

**Read this whole file before Step 3 (or before re-running Step 2).** Stamped exit file
for Step 2 (spec §11). **Stamped:** 2026-07-08. **Spec:** `SMC_EDGE_LAB_SPEC.md` (frozen).

---

## TL;DR (read this first)

- **Step 2 is DONE and its code is bug-fixed + test-guarded.** Do NOT re-run the
  backtest — Step 2 only READS the frozen `trades.csv`, never writes it.
- **The real Step-2 result is a NULL: no single pre-entry feature saves the system.**
  44 features screened; only 2 clear the triage, and both are weak/fragile.
- **`alert_utc_hour` is NOT a real edge** — effect 0.094R but **MI = 0.00000** (zero
  win/loss information). It ranks only because a tiny effect goes "significant" on 11k
  trades. It is a proxy for *when limit orders sit at zone edges*; change proximity
  rules and it dies. Trader flagged this correctly. Do not treat it as a find.
- **The trader's real signal — 60%+ of losers die straight to SL, many SL-swept — is
  REAL but was STRUCTURALLY INVISIBLE to Step 2** because those columns are
  outcome-time (look-ahead wall). That signal belongs to the **EXIT TRACK (Step 6)**,
  not entry discovery. **Recommendation: jump the SL-sweep / wider-stop replay forward.**

---

## What Step 2 built

- **`backtest/diagnostics/edge_lab_step2.py`** — the univariate pooled discovery
  scorecard (Track A, spec §4/§2/§3/§9). Imports the v1 spine + the Step-1 scaffold;
  never copies them.
- **`tests/test_edge_lab_step2.py`** — 12 hand-computed verification tests, ALL PASSING.
  Out-of-band (never touches the live path). Guards the exact bug classes hit below.
- **Outputs:** `backtest/results/h1only_20080102_20251231/edge_lab/step2_scorecard.{json,md}`
  — one ranked table, every feature a row, richest on top, two-bucket disposition.

### Methodology (what each number IS — so the next chat trusts it)
- **Population:** pooled FX + Gold (Book A + B), **DISCOVERY split 2008–2016 only**
  (C5 holdout sealed). N = 11,329 (gates-off) / 10,721 (live-gated, score≥4). BTC
  excluded (standalone, starts 2017 — deferred to Step 3 per trader).
- **Target:** `r_realised` (per-trade R, source of truth). Seed 42, deterministic.
- **Test per feature type (spec §3, correct-test rule):**
  - continuous → **Spearman** trend + **Mutual Information** (U-shape catcher) + decile curve
  - ordinal → **Spearman (NOT Kruskal)** — the exact v1 bug §0.3 flags
  - nominal → **Kruskal–Wallis** across levels
  - binary → **bootstrap diff-CI**
- **Effect size:** top-vs-bottom bucket ΔR, over levels clearing MIN_BUCKET_N (150).
- **Consistency:** per-quarter pos-fraction (a quarter counts only at N≥30).
- **Bayesian shrinkage:** thin buckets pulled toward pooled mean (can't masquerade).
- **FDR + effect-floor (0.05R):** INFORMATIONAL columns, NOT gates (spec §8/§10).
- **Two views side by side (spec §9):** gates-off vs live-gated — they AGREE here (the
  current score filter is not hiding a better subset).

---

## Bugs found IN THE SCORECARD CODE and fixed (backtest untouched)

All three were in the analysis layer, found by eyeballing output, now test-guarded:

1. **`bos_tier` mis-typed as ordinal.** Its levels are BOS/CHoCH/Confirm/Range —
   *unordered structure types*, not ranked tiers. My hardcoded tier map → all NaN →
   N=0. **Fix:** reclassified nominal (Kruskal). Guard: `test_bos_tier_is_nominal_not_ordinal`.
2. **Fake −0.9254R effect on `bos_sequence_count`.** Pulled from an N=1 extreme bucket
   (level 10, one trade). **Fix:** effect computed only over levels ≥ MIN_BUCKET_N; walk
   inward, else report no effect (Spearman still carries the trend). Guard:
   `test_ordinal_effect_ignores_thin_extreme_bucket`.
3. **Same thin-bucket leak on the nominal path** (`setup_badge` etc.). **Fix:** best/worst
   effect over fat levels only. Guard: `test_nominal_effect_ignores_thin_level`.

**Verification:** 12/12 tests pass (type classification, thin-bucket guards, Spearman
sign, binary diff, disposition triage, Bayesian shrinkage). Module compiles, zero unused
imports.

---

## The ranked table — what it actually says

Full table: `edge_lab/step2_scorecard.md`. The two features that clear the triage:

| # | feature | kind | effect R | eff CI | MI | p | consistency | verdict |
|---|---|---|---|---|---|---|---|---|
| queue | `alert_utc_hour` | continuous | +0.094 | [0.0002, 0.189] | **0.00000** | 0.0003 | 16/28 q | **fragile — not a real edge** |
| queue | `ob_walkback_depth` | ordinal | −0.194 | [−0.338, −0.043] | — | 0.034 | (0→3 monotonic) | **spec pre-warns it dies in fuller sample (§0.3/§15)** |

**Everything else (42 features) = "interesting, not proven."** Break quality, FVG size,
OB age, killzone, PD alignment, trend agreement, score itself — none carry a reliable
single-feature edge on entry.

### Interpretation (the honest conclusion)
- **This is the SAME null the instant-death / OB-age / trend-alignment investigations
  already reached** (memory + `DISCOVERY_FINDINGS.md`): **selection is the wrong lever.**
  No pre-entry column turns the kept set positive.
- `alert_utc_hour` is the v1 lone candidate resurfacing. **MI=0 proves it's noise-rank,
  not signal.** Do not build on it.
- `ob_walkback_depth`'s monotonic decline is real IN discovery but pre-registered as
  non-persistent. Step 2 surfaces it honestly; the ship gate (later) kills it if noise.

---

## The DISAGREEMENT that redirects the work (trader was right)

- **Trader's point:** "60%+ of losers go straight to SL; a large chunk had SL swept —
  THAT is the signal. Where are we tracking it? alert_utc_hour is worthless."
- **Correct.** Those columns exist: `sl_bar_was_sweep`, `sl_swept_then_tp1`,
  `sl_wick_depth_atr`, `sl_max_adverse_after_sweep_atr`, `bars_sl_to_tp1_touch`,
  `sl_recovered_to_entry`. **Step 2 did NOT screen them — by design:** they are
  **outcome-time** (known only after the trade dies) → blocked by the look-ahead wall.
- **So Step 2 was structurally blind to the trader's best signal.** Not a bug — but it
  means entry discovery is the WRONG place to chase it. It lives in the **exit track**:
  don't *filter* these trades, *survive* them (wider / sweep-aware stops).

### Established, carried in (do NOT re-discover)
- Instant death = 1-bar geometry; 92.9% die ≤1 bar of fill; SL ≈ one H1 bar's range.
- 53% of true losers' stop candles are sweeps (wick through, close back); 30% swept then
  touched TP1. Whether a wider stop wins depends ENTIRELY on wick-pierce depth.
- BE stops (16.7% of filled): 53.2% of BE stop-outs are sweeps, 34.1% swept then TP1.
- Every "prize" number there is a touch-based ceiling — **never bankable; only a real
  -order replay counts (C6).**

---

## RECOMMENDED NEXT STEP (my call — trader disposes)

**Do NOT proceed to Step 3 (pair-level entry discovery) next.** It will re-confirm the
same entry null per pair. Instead:

- **Jump to the EXIT TRACK wider-stop / SL-sweep replay (spec §7, Step 6 Pass 1).**
  The six columns to SIZE it are now in the clean CSV. This is the one lever with a real
  mechanism behind it and it's exactly what the trader has been pointing at.
- Pass 1 is descriptive on ALL trades (allowed before the entry filter is frozen), so it
  can start now. The wider-stop win is a REAL-ORDER replay via `_replay_recipe`, sized
  from the `sl_wick_depth_atr` distribution — never the touch-based ceiling.

If the trader still wants Step 3 first, it's cheap and valid — just expect a null.

---

## Entry contract status for Step 3 (if taken next)
| requirement | status |
|---|---|
| Step 2 done, scorecard written | ✅ |
| Step 2 code bug-fixed + test-guarded | ✅ 12/12 |
| pooled discovery table exists (JSON + md) | ✅ |

## Guardrails honored
- **C5 sacred:** DISCOVERY 2008–2016 only; holdout untouched.
- **Look-ahead wall:** outcome-time SL columns NOT screened as entry features (correct).
- **Data-vs-SMC:** `alert_utc_hour` flagged as fragile/mechanism-less despite
  significance — a discussion point, never shipped on data alone.
- **C6:** no SL-sweep "prize" number banked; only real-order replay will count (Step 6).
- **Guards out-of-band:** all 12 tests run offline, never in the live alert path.

## Files this chat touched
- `backtest/diagnostics/edge_lab_step2.py` (new)
- `tests/test_edge_lab_step2.py` (new)
- `backtest/results/h1only_20080102_20251231/edge_lab/step2_scorecard.{json,md}` (new outputs)
- `EDGE_LAB_STEP2_HANDOFF.md` (this file)
- **None of these are committed yet** (OneDrive commit-local policy; push only on "ship it").
