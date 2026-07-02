# THE EDGE ENGINE — Master Handoff

**The single source for the edge-discovery engine. Build the script in a fresh chat from this file.**

This engine decides whether this system has a real, repeatable edge — in entries, exits, pairs, or
nowhere — and if it does, emits the exact recipe to trade it. It is the most important artifact in
the project. The numbers it produces dictate how the trader spends the next decade. Treat every
claim below as a CLAIM TO RE-VERIFY, not gospel. Read the code, run the data, then speak.

---

## 0. HOW TO WORK ON THIS (inherit this operating style — it is why the good chats worked)

- **Data-first, not rule-first.** Never invent a filter and then test it. Take 18 years of real
  setups + outcomes, find what actually separates winners from losers, let the pattern define the
  rule, validate it on UNSEEN years. A rule that is not OOS-validated does not exist.
- **Bootstrap CI + per-quarter sign on EVERYTHING.** A pooled average is a trap. An edge is real
  only if its 95% CI clears 0 AND its sign holds across most quarters. A signal that flips sign
  quarter-to-quarter is noise even if the pooled number looks great.
- **Verify by reading the code before claiming.** Cite `file:line`. Most wasted time in this
  project came from describing behaviour from memory.
- **Brutal honesty, no sycophancy.** If the engine finds no edge, say "no edge" — that is a WIN,
  not a failure. Killing a dead system early saves years. A thin/small sample never overrides
  sound SMC; sound data never gets softened to be agreeable.
- **Data vs SMC:** agree → conclude. Disagree → DISCUSSION POINT (name the cause: detector bug,
  thin N, censored data, regime); do not silently score on it. ICT picks the candidate features;
  the 18-yr bootstrap picks the survivors. (Sweep is the cautionary tale: ICT-positive, −0.440R in
  our data.)
- **Plain English, bullets, short.** The trader is technical but hates verbosity and jargon. H1 is
  the ONLY execution clock. P&L truth is `r_realised`. NAS100 excluded. Proximal entry only.

---

## 1. THE MISSION (read before anything else)

The system today does **not** break even. Every prior attempt to fix it tuned **guessed filters**
on **pre-filtered (censored) data** — and failed. This engine inverts that:

1. Strip every guessed gate. Run the full, unfiltered population of setups over 18 years.
2. Log every candidate feature + every outcome per setup.
3. Learn — from data, validated OOS — which setups, which exits, and which pairs carry an edge.
4. Emit a small **recipe spec**, or an honest **"no edge"** verdict.

> **A score concentrates an edge; it cannot create one.** If the whole population loses and no
> sub-group is *robustly* positive (CI > 0, consistent across quarters), then no threshold turns it
> positive — you can only lose less by trading less. Fix the edge (entry / exit / pairs), not the
> points. — `backtest/RECOMMENDATIONS.md`

The trader has TWO independent shots at an edge — **entries** (an EV-score) and **exits** (already
showing promise: losses are given-back winners, BE@0.5 recovers ~14R) — plus a third in reserve
(the parked daily-narrative layer). The engine must test all of them and report honestly which, if
any, pays.

---

## 2. THE INPUT — one 18-year run, everything logged

- **One MT5 backtest pass, full history (~2008→present).** MT5 parquet cache is the sole data
  source (`backtest/data_loader.py`; yfinance removed 2026-06-26). Do not run it twice — this run
  feeds BOTH the exit study and the feature study.
- **Two books, kept separate** (matches the two emails): (A) old FX + Gold [EURUSD, NZDUSD, USDJPY,
  USDCHF, XAUUSD], (B) new FX + BTC [GBPUSD, AUDUSD, USDCAD, EURJPY, BTCUSD]. **NAS100 excluded
  entirely.** BTC bucketed separately (venue/boundary noise).
- **Proximal entry only.** 50% mean entry is dead — do not compute it.
- **NO score gate.** The confidence score is logged as a number, never filters. Study the full
  range; let the EV-score re-derive any real threshold from data.
- **War regime (2026) bucketed separately** from BAU (<2026). The daily-bias study proved the war
  year flips conclusions — never let it pool silently.
- **Every candidate feature logged per setup** (see §3). This logging is built in the PRIOR chat
  (observe-only, parity-proven) before the run — confirm the columns exist before trusting Stage 1.

---

## 3. THE FEATURES (the levers — what the engine screens)

All ATR-normalized (cross-instrument comparable). Status: ✓ already logged · + being added · ✗ dropped.

**Structure / event quality**
- ✓ `break_close_atr` — how far past the level the break candle CLOSED. *Your "how good is the
  break."* Primary lever.
- ✓ `break_body_atr` — break-candle body size (displacement). Score the RAW value, never the
  `break_excess` ratio (the ratio re-maps when the floor moves).
- ✓ `bos_tag` (BOS/CHoCH), `bos_tier` (Major/Minor/Range/Confirm), `bos_verdict` (holding/fading),
  `reversal_pct` / `reversed_from_extreme` (CHoCH-from-extreme).
- + `impulse_leg_atr` — size of the displacement leg that broke structure and left the OB behind.
  **Measured FVG-independently** (an FVG must NOT gate it). Definition: OB origin → the swing
  extreme the displacement created. **SMC-verify the definition (web search + vet) before trusting.**

**Imbalance**
- + `fvg_size_atr` — `(fvg_top − fvg_bottom) / ATR_at_OB_formation`. *Your "how big is the FVG."*
  Replaces the present/absent boolean's lost gradient.
- ✓ `fvg_present`, FVG `mitigation` (pristine/partial/full).

**Location (the one PROVEN lever)**
- ✓ `pd_pct` / `pd_zone` / `pd_alignment` — premium/discount position. **Counter-PD CHoCH is the
  only robustly-proven signal in the whole audit (CI excludes 0, −0.433R) — and the live scorer
  THROWS IT AWAY (PD removed from scoring).** Re-introduce as a gate/penalty, not additive points.

**Zone / risk geometry**
- + `ob_range_atr` — OB candle high-to-low in ATR. NOTE: this ≈ the **stop distance** (SL = distal,
  entry = proximal). Log once; do not also log "zone thickness" (same thing — confirmed
  `smc_detector.py:1647-1665`). Connects directly to the NAS failure mode (stop < one candle's range).
- + `atr_at_entry` — raw volatility context. **Not an SMC concept — a quant feature.** Standalone
  it tested weak (no clean regime); keep it only as an INTERACTION term (with stop-width) in the
  regression, low expectations.
- ✓ `ob_age_h1_bars`, freshness/`touches`.

**Context / time**
- ✓ `session`, `ob_session`, `fill_session`, `killzone_alignment`, `h1_trend`, `trend_alignment`.
  (Trend alignment is now meaningful — the label flips on confirmation BOS, not on CHoCH, so
  "against-trend" is a real populated bucket; `RECOMMENDATIONS.md` 2026-06-26.)

**Sweep**
- ✗ `sweep_present` is logged but **drop it from any score** — detector inverts SMC (−0.440R).
  Keep the column for audit; do not feature it until the detector is rebuilt.

**Stop-out anatomy (SL exits only — the MOST IMPORTANT split to bucket over 18 yr)**
- ✓ `sl_bar_was_sweep` / `sl_swept_then_tp1` (added 2026-07-02; replaced the old touch-only
  `sl_was_swept`). Definitions in `h1_only_simulator.py` (~line 850). **Bucket every stop-out into
  two populations and study them for OPPOSITE reasons — they are not the same failure:**
    - **`sl_bar_was_sweep = True` → an EXIT / stop-placement problem.** The setup was right, the
      market grabbed liquidity and rejected. Lever = wider stop / better stop placement / exit
      policy. `sl_swept_then_tp1 = True` sub-bucket = the stop was swept AND price then reached TP1
      → the strongest "a wider stop would have won" candidate. (March-2025 real slice: 37/63
      stop-outs were sweeps, 22 of those later tagged TP1 — real signal, collect over 18 yr.)
    - **`sl_bar_was_sweep = False` → an ENTRY / SETUP problem.** Price CLOSED CLEAN THROUGH the
      stop. That is not a stop-hunt — it is the market rejecting the setup outright (wrong OB, wrong
      bias, broken structure). A wider stop only loses MORE here. If a feature slice (event type, PD
      zone, pair, session) over-produces clean-break stop-outs, the fault is UPSTREAM of the exit —
      screen those setups out, do not re-place the stop. **This bucket is the entry-quality
      tripwire the exit study alone would miss.**
  - CAVEAT (small, ignore during discovery, respect only at recipe stage): `sl_swept_then_tp1` ends
    on a TOUCH of a later bar, so it can slightly over-count (that TP1 tag could be its own
    spike-fade). It does NOT hide the pattern across 18 yr — only matters when a pattern is turned
    into a real rule, at which point replay it as a real wider-stop order (`exit_lab`).

**News (already logged — do NOT web-search per trade; it is token-heavy and unreliable)**
- ✓ `news_blocked`, `news_event_title`, `news_event_currency`, `news_event_source`, `news_event_ts`
  are ALREADY columns in trades.csv (from `ci_filter.py`). Timestamp→event mapping is free from the
  news feed — no web search. Population depends on the run's news-feed config; **verify these
  populate on the 18-yr run before slicing on them** (they were empty on the March-2025 slice
  because no row was near a flagged event). Use to check whether clean-break stop-outs cluster on
  news bars (a confounder, not a setup fault).

**Setup badges (Phase 2 email banners)**
- + `setup_badge` — `smc_detector.classify_setup(ob, pd_position, trend_alignment)`
  (`smc_detector.py:2442`) output: "A+ Reversal at the Wall" / "A First Pullback" /
  "Caution: Late-Trend Chase" / none. **Fires live in the email
  (`Phase2_Alert_Engine.py:2542`) but is NOT called in the backtest simulator** — no
  column exists to check whether these banners actually correlate with win rate.
  Must be added to the simulator (same inputs it already computes) before Stage 1 can
  screen it. Treat as UNVALIDATED until it survives Stage 1 like any other feature —
  a banner name is not evidence.

**Outcomes (the targets)**
- ✓ `r_realised` (P&L truth), `mfe_r`, `mae_r`, `bars_to_exit`, `exit_reason`, `tp1_rr`.
- ✓ `sl_bar_was_sweep` / `sl_swept_then_tp1` — outcome-side flags on SL exits; the stop-out-anatomy
  split above. Screen features against BOTH (which setups produce clean-break vs sweep stop-outs).

---

## 4. THE ENGINE — four stages

### Stage 1 — Univariate feature screen (kill the noise)

**Standing rule — ANY new candidate insight goes through this before it is trusted.** Any metric,
banner, badge, or signal we can measure and bucket (e.g. `setup_badge` vs win rate, insight quality
vs P&L) MUST be run through this screen — bucket it, compute **Spearman's rank correlation**
(`backtest/insights.py` already has `_spearmanr` — reuse it, don't reimplement) between the
feature and `r_realised` (or WR per bucket), and read the p-value/CI before it graduates from
"looks right in email copy" to "actually predicts outcome." A banner name, a star rating, or a
plain-English label is a CLAIM, not a signal, until it clears this bar. No exceptions — this is how
`sweep_present` was caught inverted (−0.440R) instead of shipped on vibes.

For each feature in §3:
- Bucket it (quantiles for continuous; categories as-is).
- Per bucket: N, expR, **bootstrap 95% CI**, **per-quarter sign**, WR (wins/(wins+losses)).
- **Spearman's r** (feature rank vs outcome rank) + p-value — statistical soundness check, not just
  eyeballed bucket differences.
- Split DISCOVERY (e.g. ≤2018) vs OOS (later years) — a feature survives ONLY if its separation
  holds in OOS.
- **Output:** a ranked table of features that *robustly* separate outcomes (survivors) vs noise
  (dropped). This is the dynamic win/loss driver analysis run over the full book — also test the
  top 2-way interactions (e.g. pair×session, PD×event) for patterns a univariate screen can't see.

### Stage 2 — The EV-score (multivariate; concentrate the edge)
- Feed the Stage-1 survivors into a regression: **logistic** (→ win probability) and/or **linear /
  gradient** (→ expected R). The regression handles COLLINEARITY a point-stack cannot — e.g.
  `break_body_atr`, `bos_tier`, and `impulse_leg_atr` partly measure the same displacement; the
  model won't triple-count them.
- **Output: one number per setup = its expected R (the EV-score).** Calibrate it (predicted vs
  realised R per decile). Keep the model SIMPLE and interpretable — a black box you can't explain to
  a vet is a liability, not an edge.
- PD gate goes in regardless (already proven).

### Stage 3 — Setup-conditional exit optimization (the trader's lightbulb, done right)
- **The insight:** TP should be conditioned on the setup, because MFE distributions differ by setup
  type. The naive version (set TP = average MFE) is WRONG — the mean is not the expected-R-maximizing
  level and errs in both directions under skew. (Proof: a 50/50 mix of +3R-MFE and 0R-MFE has mean
  MFE 1.5R, but TP=3R yields +1.0R E[R] vs +0.25R at TP=1.5R.)
- **The method:** for each setup cluster (by EV-score decile or by the surviving features), **sweep
  TP / BE / partial recipes via `backtest/exit_engine.walk_multileg` and pick the
  expected-R-maximizing recipe per cluster** — using the empirical reach-before-stop frequencies
  from the full MFE distribution, **net of spread**, discounting uncapturable same-bar intrabar
  spikes (a loser's MFE often spikes on the fill bar where BE cannot arm — `RECOMMENDATIONS.md`).
- **The prize:** the engine emits, per cluster, BOTH the EV (take/skip/size) AND its optimal exit.
  Entry and exit co-optimized. This is the frontier — most of the system's recoverable money is
  here (losses are given-back winners).

### Stage 4 — Recipe synthesis + OOS validation (the verdict)
- Combine: **EV top-slice × per-cluster exit × pair set.** Validate the COMBINATION on held-back
  years (true OOS), bootstrap CI + per-quarter sign, **net of spread**.
- **Output — the recipe spec:** `{ features that gate/size, the EV formula, the exit recipe(s) per
  cluster, the pair set }` + an HONEST verdict:
  - **Edge in entries** → ship the EV-score gate/sizing.
  - **Edge only in exits** → ship the exit recipe; entries are a coin-flip, stop pretending.
  - **No edge survives OOS** → say so plainly. Trade less or rebuild the entry basis (daily
    narrative / different TF). Killing it is the win.

---

## 5. METHODOLOGY GUARDRAILS (the engine is worthless if these are skipped)

- **Multiple comparisons.** Screening many features manufactures false positives. Guard: OOS
  confirmation + per-quarter sign + a minimum N per bucket. A pattern that fails the guard is shown
  as "directional, thin" — never promoted to a rule.
- **Censored data.** Every logged trade already passed the OLD gates, so you can only ever test
  TIGHTENING, never loosening. Stripping the score gate (§2) is what un-censors the population —
  the run MUST be done with gates off, or Stage 1 is blind to half the answer.
- **Spread / costs.** The backtest models no spread/slippage. The FINAL recipe must be re-checked
  net of realistic cost — tight-TP recipes are the most cost-fragile.
- **PEAK-vs-FILL — the trap that has bitten twice (READ THIS).** A touch/extremum is NOT an
  achievable outcome. `mfe_r`, `mae_r`, "price reached X% N times", "reversed from level L" — all
  record where price *touched* on the path, not where you could have *exited*. Banking a level needs
  a resting order that fills BEFORE the stop under the sim's pessimistic same-bar rule (SL checked
  first). Two false claims came from ignoring this: "47% of stop-outs reached +1R" and "93% reversed
  from +0.5R" — both collapsed to ~1-2% when replayed with a REAL fixed exit. **THE LAW: any claim
  whose number comes from a peak/touch column is UNVERIFIED until re-run as a real order
  (`exit_engine.walk_multileg` / `exit_lab`) and reconciled to `r_realised`.** Enforced in code:
  `backtest/insights.verify_capturable(...)` (hard gate — pass touch-count AND replayed capturable-
  count; it rejects the claim if they diverge) and `insights.is_peak_metric(name)` (flags such
  columns). Call the gate before quoting any "reached a level" stat. Never treat raw MFE as profit.
- **r_realised is truth.** Every P&L number reconciles to it. Reject any stage whose baseline
  doesn't reproduce committed `r_realised` (the `exit_lab` self-check pattern).
- **War regime.** Keep 2026 bucketed separate from BAU end-to-end.

---

## 6. WHAT ALREADY EXISTS (orchestrate — do NOT rebuild)

- `backtest/exit_engine.walk_multileg` — the single multi-leg exit walker. Stage 3 calls it.
- `backtest/diagnostics/exit_lab.py` — the side-channel pattern: one fresh run, replay N recipes
  over the SAME in-memory post-fill bars, self-check baseline vs committed `r_realised`. The engine
  EXTENDS this pattern to features + EV + per-cluster exits.
- `backtest/insights.bootstrap_ci`, `win_rate_pct` — the stats primitives.
- The 18-yr MT5 parquet cache — already populated.
- The per-trade feature logging (§3) — built in the prior chat, observe-only.
- `backtest/RECOMMENDATIONS.md` — the durable findings + methods. READ IT before judging any run.

The engine is an ORCHESTRATOR over these, plus a regression. It is a diagnostic — it must NOT touch
git/registry/live (neutralise persistence exactly as `exit_lab.py:109-114` does).

---

## 7. BUILD IT AS A SCRIPT, INTERPRET IT IN CHAT

- The screen + regression run over 18 years × thousands of trades — that is COMPUTE, it must be a
  script (`backtest/diagnostics/edge_engine.py`), not chat work.
- The script emits a clean, dense **table + JSON** (Stage-1 ranked features, Stage-2 EV calibration,
  Stage-3 per-cluster exit table, Stage-4 OOS recipe + verdict).
- The JUDGMENT — reading that output, deciding the recipe, sanity-checking against the vet's eye —
  happens in a chat with the output pasted in. Script crunches; human + Claude decide.

---

## 8. SEQUENCING (where this sits)

1. **Logging additions** (prior chat) — `fvg_size_atr`, `impulse_leg_atr` (SMC-verified),
   `ob_range_atr`, `atr_at_entry`. Observe-only, parity-proven. **The long pole — everything here
   needs the columns first.**
2. **Exit-readiness fixes** (prior chat) — stale yfinance docstrings → MT5; XAUUSD pair name.
3. **The one 18-yr run** — gates off, all features logged, two books, war bucketed.
4. **This engine** — Stages 1→4 on that run's output.
5. **The email** — the readable report surface for what the engine finds (separate handoff).
6. **Daily narrative** — a future feature that plugs into Stage 2 once in-system levers are settled.

---

## 9. FIRST MOVE (concrete, do this in the new chat)

1. Confirm the §3 feature columns exist in `trades.csv` from a small run. If any are missing, the
   logging step was not finished — fix that FIRST, the engine is blind without them.
2. Build Stage 1 (univariate screen + bootstrap + per-quarter + OOS split) and run it on the 18-yr
   output. Read the ranked survivors honestly. If NOTHING survives OOS, stop and report — do not
   force Stage 2 on noise.
3. Only if survivors exist: Stage 2 (EV-score), then Stage 3 (per-cluster exits), then Stage 4
   (OOS recipe + verdict). Prove each stage's baseline reproduces committed `r_realised`.

---

## 9b. INVESTIGATION LOG (BOS-quality dig, 2026-07-02 — collect more data before concluding)

Two runs analysed: 2008 full year + 2024-07..2025-06. BOS-family filled trades ~905.
None of these are conclusions yet — flagged for the 18-year run to confirm/kill.

- **FVG-proximal entry for strong breaks — TESTED, does NOT help. Parked.**
  Hypothesis: on a strong break the OB proximal is too far, so enter at the FVG near-edge
  instead. Reconstructed the real 3-candle FVG from MT5 for all 52 strong+FVG breaks and
  checked which line price reverses from first. Result: FVG near-edge and OB proximal are
  ~28 bps apart (median) — effectively the same price. Combined: 34/52 reached OB proximal,
  10/52 reversed in the thin gap, 8/52 reversed short of both. FVG-proximal entry would not
  save these. Strong breaks fail because the pullback often doesn't return to EITHER line, not
  because of line placement. Collect more (18-yr) before final kill, but no edge in sight.

- **ob_to_fill_hours × WR — log as diagnostic, NOT a gate.** Gap from OB formation to fill.
  corr(gap, r_realised) ≈ 0 both years (2008 −0.007, 2024-25 +0.053). Not monotonic: 2008
  fast fills worst, 2024-25 fast fills best (opposite). Mild tilt: middle buckets (10-60h)
  best, extremes (<6h and 60h+) worst — weak, not tradeable. Symptom, not edge. Log the column
  (`ob_to_fill_hours`) for the engine to slice; do NOT pre-gate on it.

- **Strong-break early-pullback flag (BS1) — promising, thin, confounded by news.** Strong break
  + pullback within 3h = 11% WR / −0.69R (n=9); ≤6h = 21% WR (n=14); slower = 26%. A strong
  break that snaps back fast is the worst bucket. BUT samples tiny and the likely cause is a
  news spike reverting, not structure quality. Log `bars_break_to_pullback`; validate over 18-yr
  AND tag news bars before it can gate anything.

- **Fading verdict — FIXED & SHIPPED (2026-07).** Old rule needed >=3 plain BOS and compared
  last-2-vs-first-2 body mean; fired only ~12/900 (1.3%). New rule (`smc_detector.bos_leg_read`):
  fire from the 2nd plain BOS — tag 'fading' when the LATEST break body < 0.60 × the leg's FIRST
  break body. It is DISPLACEMENT-decay, NOT leg-depth: data showed depth alone is not exhaustion
  (seq 3-4 lose 26-31% WR but seq 5-9 WIN 40-67%), so a flat "seq>=3 = fading" was rejected — we
  read the body, never the count. Verified: fading now fires ~10% of BOS and correctly splits
  losers (fading 0% WR / −0.75R vs holding 50% / +0.27R on a Q1-2025 slice). Scorer already maps
  fading -> structure=1 (vs holding=3), so the penalty now reaches the trades it should.

- **"Given-back winner" / SL-sweep (BS2) — my mfe≥1R read was a TRAP; corrected.** Raw
  `mfe_r` on baseline SL-losers said ~41% touched ≥1R favourable before stopping (the loose "47%"
  once quoted). But applying a REAL fixed-1R exit (`exit_lab C_fullTP_1.0R`, 2024-07..2025-06)
  converts **23.5%** of live-stopped trades to a booked +1R (92/391) — vs 41.4% that merely
  touched it. The 18pp gap = same-candle +1R-and-SL collisions a real 1R limit cannot bank (SL
  resolves first, pessimistic — the Stage-3 intrabar trap). So the honest capturable figure is
  ~23%, NOT the ~1-2% once written here, and NOT the touch-based 47%. Fixed-1R still LOSES money
  net as a blanket policy (2024-25 −15R, 2008 −60R, CI excludes 0) — the 23% that flip do not pay
  for the ones the earlier TP costs.

- **Sweep columns replaced `sl_was_swept` (2026-07-02).** `sl_was_swept` (later-bars-only, no
  sweep gate) is GONE from new runs. Two honest columns replace it, both `None` unless `exit_reason
  == 'sl'`:
    - `sl_bar_was_sweep` — the STOP CANDLE itself was a sweep: it wicked through the stop that
      fired (`cur_sl`, break-even-aware) but CLOSED BACK on our side. Long: `Low<=cur_sl AND
      Close>cur_sl`. Short: `High>=cur_sl AND Close<cur_sl`. Strict close-back (a close exactly AT
      the stop is NOT a sweep). This is the SMC grab-then-reject shape vs a clean close-through
      (genuine break — a wider stop just loses more).
    - `sl_swept_then_tp1` — `sl_bar_was_sweep` is True AND, within `SL_SWEEP_LOOKBACK_BARS` bars
      AFTER the stop bar, price reached TP1 in our direction. The honest "a wider stop plausibly
      wins" signal. CAVEAT: still ends on a TOUCH check of a later bar, not a real-order replay —
      read as a HINT, never bankable free money (peak-vs-fill law).

## 10. GLOSSARY

- **EV-score** — for one setup, the model's estimate of average R if you took setups like it many
  times. Replaces "how many confluences stacked" with "expected R per setup."
- **MFE / MAE** — max favorable / adverse excursion in R (how far the trade went your way / against
  you before exit).
- **Bootstrap CI** — resample trades with replacement 10k×, take the middle 95% of the averages.
  CI > 0 = real edge; crosses 0 = unproven; < 0 = real loser.
- **Per-quarter sign** — the sign (+/−) of expR each calendar quarter; a real edge holds sign, noise
  flips it.
- **OOS** — out-of-sample: years held back from discovery, used to confirm a pattern is real.
- **Censored data** — outcomes only exist for setups the old gates let through, so loosening a gate
  is untestable from this data; only tightening is.
</content>
</invoke>
