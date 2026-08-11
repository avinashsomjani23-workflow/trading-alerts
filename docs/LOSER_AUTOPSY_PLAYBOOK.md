# LOSER AUTOPSY PLAYBOOK

**Pair- and timeframe-agnostic method** for dissecting losing trades on ANY backtest run.
Re-used unchanged for every new pair / timeframe so every run is analysed the same robust way.

> **METHOD ONLY — no run-specific numbers here.** Per-run measurements live in a
> per-run findings doc (e.g. `docs/MFE_FIX_PLAN.md` snapshot, or a run findings file).
> This file is the recipe; the numbers are re-derived each run against live code
> (CLAUDE.md Rule 4 — a remembered number is only a hypothesis).

**Prereq:** Stage 0 health sign-off passed for the run; `CANONICAL.md` repointed;
column count re-confirmed against the live header.

---

## 1. Population selection — do this IDENTICALLY every run

**a. Resolved trades only.** The analysis population is trades whose
`exit_reason ∈ {tp, sl}` — a clean win (+2R under fixed 2R) or a clean loss (−1R).

**b. Excluded — AUDIT-ONLY, never in the population:**
`never_filled` (the limit never triggered — the trade never happened),
`friday_flat`, `timeout`, `window_end` (booked a partial close-price R — not a clean
win or loss). Folding any of these in corrupts WR and meanR. This exclusion is the
one most likely to be forgotten in a fresh chat — it is pinned in CLAUDE.md.

**c. News filter — TWO populations, TWO purposes (do not conflate):**

| use | filter | why |
|---|---|---|
| **Baseline / tradeable-universe / EV** | **LOOSE** — `news_fill==0` | Keeps trades where news landed *while open*. Our system does **not** close for news (news is informational, never a gate — h1_only_reporting.py:1294). Those outcomes are real and live-reproducible; deleting them hides real P&L. |
| **Loser autopsy / feature screen** | **STRICT** — `news_fill==0` AND `news_open==0` | Also removes trades where a high-impact event hit *while the trade was open*. A news-driven win/loss is exogenous (the shock, not the setup); including it makes the feature screen credit whatever SMC feature the trade happened to carry. |

- `news_fill` / `news_open` **blank = "cannot know" = NEVER treated as clean.**
- **Always log the removed news-open segment separately.** Removal ≠ junk — that
  segment can carry its own (often positive) signal worth its own study.
- Definitions (TRUTH_LEDGER.md, `news_enrichment.py`): `news_fill` = high-impact
  scheduled event within ±1h of the fill candle; `news_open` = such an event while
  the trade was open `[fill, exit close]`.

**Frozen baseline check.** The current run's LOOSE baseline is pinned in code:
`python -m backtest.diagnostics.baseline_freeze` regenerates the tier table from the
canonical CSV and ASSERTS the frozen numbers (exit 0 = match, non-zero = drift). Any
detection re-run or population-rule change repoints + re-freezes that script in the
same commit. Current freeze (2026-08-02, `h1only_20080102_20161231`): LOOSE
N=1,477 · WR 30.1% · meanR −0.096 · sumR −142.

---

## 2. Loser taxonomy — split losers by HOW they died

Different death mechanisms need **opposite** fixes, so never lump losers into one pile.

| # | bucket | definition | what it tells us |
|---|---|---|---|
| 1 | **Instant death** | in-trade MFE ≈ 0, died in ~1 bar | Never had a chance. Entry was wrong/mistimed → **entry-filter target.** |
| 2 | **Slow bleed** | in-trade MFE ≈ 0 but lived several bars, never traded above entry | Dead zone — price sat under entry then stopped. Entry-quality signal. |
| 3 | **Weak reaction** | small in-trade MFE (0–1R), a poke in our favour then folded | Thin OB / spent fuel — got a reaction, couldn't sustain. |
| 4 | **Real give-back** | in-trade MFE ≥ 1R with the peak **before** the stop | Had real profit, lost it. Under a frozen exit there is little to fix here — and these setups overlap the ones that run to +2R, so filtering them kills winners too. Note & park. |
| 5 | **Stop-hunt** | stopped first, **then** ran to target (`sl_swept_then_1r/2r`) | Shaken out — entry/stop-placement fault, entry-fixable. |

**Read, don't decide.** This taxonomy is a MAP of where the losses are, not a filter.
Filters ship later, with CIs and a garbage-first cut, on the feature screen.

### ⚠️ WHICH MFE COLUMN — never mix these up
Two favourable-excursion measures exist. Picking the wrong one silently corrupts the autopsy.

| column | measures | USE IT FOR | NEVER use it for |
|---|---|---|---|
| **`mfe_intrade_r`** (in-trade; SHIPPED 2026-08-02 in commit `c555fa71`, canonical col 150) | best favourable move **fill → stop only** (post-stop bars truncated) | the **loser autopsy** — "did this loser ever look good before it died?" (buckets 1–4) | — |
| **`mfe_r`** (full-window) | best favourable move over the **whole 48-bar window**, incl. *after* the trade already exited | the **winner** study — "how far past +2R does a winner run?" (target-distance / exit research) | **the loser autopsy** — it keeps running after the stop, so ~67% of losers "peak" after they are already dead. Splitting losers on `mfe_r` mislabels dead-on-arrival trades as "had profit." |

**Rule of thumb:** loser question → `mfe_intrade_r`. Winner run-past-target question → `mfe_r`.
`mfe_intrade_r` now EXISTS in the canonical run, so the MFE-dependent buckets (3/4) are
answerable — never substitute `mfe_r` for it.

### ⚠️ MFE dependency — current status
Buckets 1–4 need a clean **in-trade** MFE (favourable move *before* the stop). The raw
`mfe_r` column is a FULL-WINDOW measure that keeps running *after* the stop, so it is
**contaminated for losers** (a large fraction peak after death). The in-trade MFE fix has
**SHIPPED** (2026-08-02, commit `c555fa71`, `docs/MFE_FIX_PLAN.md`) — `mfe_intrade_r` is
canonical col 150 in the 184-column run:
- **Trustworthy now:** *Instant death* / *Slow bleed* via `bars_to_exit`; *Stop-hunt*
  via strict `sl_swept_then_1r` / `sl_swept_then_2r` + `sl_bar_was_sweep`.
- **Now also answerable:** *Weak reaction* vs *Real give-back* — split them on
  `mfe_intrade_r` (fill→stop favourable move), never on the contaminated `mfe_r`.

---

## NEXT — to be added (discussed, not yet frozen)
- Per-feature screening method (Stage 4): bucketing rules (quantile bins, min-N floor,
  edge reporting, sensitivity across bin counts, threshold sweep with CI).
- Metrics per bucket: N, WR (Wilson CI), meanR (bootstrap CI; note meanR = 3·WR−1 under
  fixed 2R), sumR (total bleed → prioritisation).
- Winner-side anatomy; interaction / confounding checks (Stage 5).
