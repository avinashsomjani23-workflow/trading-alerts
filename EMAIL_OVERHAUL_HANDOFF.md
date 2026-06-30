# EMAIL OVERHAUL — Handoff

**Goal: rebuild the backtest email from a 24-section kitchen-sink audit into a tight, intuitive
report where every data point drives a decision and the loss/win analysis discovers patterns
instead of checking a frozen list.** Build in a fresh chat from this file.

Operating style: bullets, plain English, no verbosity. Verify by reading code, cite `file:line`.
Brutal honesty. P&L truth is `r_realised`. Proximal entry only. NAS100 excluded.

---

## 1. THE SHAPE (locked with the trader)

- **Two emails, no combined report.** Kill `report.html`; keep only the two group emails.
  - **Book A** = old FX + Gold: EURUSD, NZDUSD, USDJPY, USDCHF, GOLD (data = XAUUSD).
  - **Book B** = new FX + BTC: GBPUSD, AUDUSD, USDCAD, EURJPY, BTCUSD.
  - **Drop NAS100 entirely** from `INDEX_COMMODITY_PAIRS` (`h1_only_reporting.py:3242`).
- **Drop the 50% mean entry everywhere** — Section B and the proximal-vs-50% head-to-head. Proximal
  is the only live model. (50% in Excel only, if at all.)
- **Net effect before any other change:** ~halves every email (the 13-section block was rendered
  twice per email, once per entry zone — `_zone_block_html` at `:3051`, called `:3420/:3426`).

---

## 2. THE TOPOLOGY TODAY (what you're cutting into)

- `write_h1_only_report` (`:3507`) builds the combined `report.html` (calls `_zone_block_html` ×2)
  AND two group emails via `_build_group_html` (`:3262`, called `:3949/:3954`), each ALSO ×2.
- `_zone_block_html` is the 13-section workhorse. Rendered 6× per run today. **This is the bloat.**

---

## 3. SECTION-BY-SECTION VERDICTS (audited; trader-corrected)

KEEP = stays, may trim · REBUILD = replace the logic · CUT = remove · APPENDIX/EXCEL = out of body.

| Section (`_zone_block_html`) | Verdict | Note |
|---|---|---|
| Signal vs noise (`_signal_vs_noise_html`) | **COLLAPSE → 1 line** | Score is settled noise (Spearman 0.05). State it; don't re-prove it every run. |
| Confidence score table | **MERGE into the line above** | Same settled question. |
| Confluences (uplift + per-pair) | **APPENDIX** | Settled non-predictive; two full tables every run is core bloat. |
| Where the edge leaked (`_edge_leak_html`) | **REBUILD (dynamic)** | 6 hardcoded checks → driver engine. See §4. |
| Backtest fidelity / same-bar | **KEEP** | Honesty about H1 resolution limit. Small. |
| Structure event performance | **KEEP, trim** | Keep the counter-PD CHoCH finding; cut the rest. |
| Trades worth a second look (vet review) | **EXCEL** | Row-level; belongs in the attachment. |
| By pair | **KEEP** | Concise, decision-relevant. |
| By session | **APPENDIX** | Granularity, rarely actioned. |
| Pair × session matrix | **APPENDIX** | Sparse, thin-N. |
| **Trend alignment** | **KEEP** | Trader override (correct): the old "93% one value" was the pre-fix CHoCH-flip behaviour. The label now flips on confirmation BOS (`RECOMMENDATIONS.md` 2026-06-26), so against-trend is a real, populated, decision-relevant bucket. |
| **Killzone alignment + losses-by-bucket** | **KEEP** (may reconstruct) | Trader wants the OB/fill-session buckets in the body. Keep them. |
| What-if counterfactual | **EXCEL only** | Heavy; trader moved it to the attachment. |
| Exit policy comparison | **PROMOTE TO TOP** | This is the point of the email now. Currently buried last. |
| Break quality (BOS/CHoCH ATR benchmark) | **KEEP** | Trader override (correct): don't demote before measuring — test on the 18-yr run, then conclude. Stays in the body. |
| Head-to-head proximal vs 50% | **CUT** | Dies with the 50% zone. |
| Killzone/News/IST/below-floor audits | **COLLAPSE → one line each** | "Filters dropped N alerts (killzone X, news Y, IST Z)." Expand only when non-zero AND changed. |
| Validation check | **KEEP** | The "did anything break" guard. |

---

## 4. THE DYNAMIC DRIVER ENGINE (replaces `_edge_leak_html`)

**Problem with today's `_edge_leak_html` (`:1276`):** six hardcoded `if` checks (same-bar spike,
tight TP1, OB age, time-in-trade, CHoCH share, Minor-structure share) with frozen thresholds. It can
only ever report the six things it was pre-told to look for. If the real loss driver this period is
"USDCHF + London + CHoCH" or "TP1 > 3R", it prints "no dominant pattern" while the pattern sits in
the data.

**Build ONE engine, surface BOTH tails (loss drivers AND win drivers):**
- Over the population, for **every available dimension** (pair, session, day-of-week, `bos_tag`,
  `bos_tier`, `fvg_state`, `killzone_alignment`, OB-age bucket, TP1-distance bucket, score bucket,
  break-strength bucket, and the new ATR features `ob_range_atr`/`fvg_size_atr`/`impulse_leg_atr`),
  compute each value's expR + loss-rate vs the system base-rate.
- **Auto-surface** only buckets whose divergence from base-rate clears the guard; rank by
  lift × support. New logged dimensions → new candidate buckets automatically. Test the top 2-way
  combos (pair×session, PD×event) for patterns a univariate pass can't see.
- Output two short ranked lists: "Losses concentrate in …" and "Winners concentrate in …", each row:
  bucket, loss/win rate vs base, N, expR, one-line SMC read.
- **THE GUARD (non-negotiable — the trader agreed):** automated mining over many dimensions
  manufactures phantom patterns on small samples. Gate every surfaced bucket behind **min-N +
  base-rate margin + per-quarter sign**. A bucket that fails the guard shows as "directional, thin"
  or not at all. Without this, the engine cries "new pattern!" on noise every run — the rigid
  problem in reverse.
- One engine, two outputs (loss + win are the same computation read in opposite directions —
  don't write two functions).

---

## 5. THE HEADLINE GAP (add this — it's the most decision-relevant number)

- Today's headline (`:3104-3110`) shows P&L / WR / avg only. **No bootstrap CI, no per-quarter
  sign** — yet `RECOMMENDATIONS.md:21` mandates both on every "is it good" call. The email preaches
  the method and doesn't use it.
- **Add bootstrap CI + per-quarter sign to the headline**, plus the one-line verdict ("real edge /
  unproven / real loser"). Reuse `backtest/insights.bootstrap_ci` and the per-quarter helper the
  exit_lab already uses — no new stats code.

---

## 6. PROPOSED EMAIL SHAPE (per book)

1. **Headline** — P&L, expR, **CI, per-quarter sign**, WR + plain verdict.
2. **Exit policy ranking** — recipes, expR + CI + per-quarter, winner flagged. (The point.)
3. **Where losses / wins concentrate** — the §4 driver engine.
4. **Backtest fidelity** — same-bar / H1-resolution honesty.
5. **By pair** — concise.
6. **Structure events** (trimmed) · **Trend alignment** · **Killzone alignment + losses** ·
   **Break quality**.
7. **Filters + validation** — one-line filter counters + the break guard.
8. **Appendix / Excel** — confluences, by-session, pair×session, counterfactual, vet rows.
- **Result: ~70% smaller; every body number drives a decision.**

---

## 7. REUSE / DON'T REBUILD

- Section builders already exist for everything kept — `_by_pair_html`, `_trend_alignment_html`,
  `_killzone_alignment_html` + `_killzone_alignment_losses_html`, `_break_benchmark_html`,
  `_same_bar_resolution_html`, `_structure_event_breakdown_html`, `_exit_policy_html`,
  `_validation_html`. The work is RE-ORDERING + DELETING, not re-writing these.
- `_edge_leak_html` is the one genuine rebuild (§4).
- `insights.bootstrap_ci` / win-rate helpers for §5.

---

## 8. PARITY / GOTCHAS

- The combined `report.html` removal + section deletions change reporting only — no `r_realised`
  impact. But confirm `summary.json` / registry consumers don't read a deleted section's output.
- The repo lives in OneDrive; backtests commit-local-only on dev, push only in CI
  (`project_onedrive_git_pushpolicy`). Structure-golden gate is unaffected (reporting isn't structure).
- Two-email split already half-built: `FOREX_PAIRS` / `INDEX_COMMODITY_PAIRS` sets at `:3241-3242`
  are where the book membership lives.

---

## 9. SEQUENCING

- This is downstream of: the logging additions (done — the new ATR features must appear in the
  driver engine) and ideally the 18-yr run + edge engine (so the exit-ranking + driver sections are
  designed against real output). Build the skeleton anytime; wire the exit-ranking section to the
  edge-engine output once it exists. See `EDGE_ENGINE_HANDOFF.md`.
</content>
