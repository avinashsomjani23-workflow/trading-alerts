# MFE FIX PLAN — in-trade MFE for loser autopsy

**Goal:** give the loser autopsy a favourable-excursion measure that reflects only what
happened *inside the trade* (fill → stop), so buckets 3/4 (Weak reaction vs Real
give-back) can be separated cleanly. Also stores the current-run loser snapshot.

> Take this file to a NEW chat to implement the fix. Nothing here is built yet.
> Verify every line against live code before acting (CLAUDE.md Rule 1).

---

## The problem (two distinct issues)

**Problem A — post-stop wander (FIXABLE with certainty).**
`mfe_r` is the FULL post-fill window excursion; the simulator walk does **not** break at
the exit — it keeps updating `mfe_price` for up to 48 bars *after* the stop latches
(`h1_only_simulator.py:675-679, 744-776`; window-MFE "A3 decouple"). So a loser's
`mfe_r` counts favourable moves that happened *after it was already dead*. **Evidence:**
on the canonical run, **66.9% of strict clean losers have their MFE peak AFTER the stop**
(median ~29 bars later); a loser that objectively died in 1 bar can still report a large
positive `mfe_r`. This makes `mfe_r` unusable as an in-trade death descriptor.

**Problem B — the stop bar's intrabar order (NOT resolvable from H1).**
The walk excludes both the fill bar and the stop bar from favourable tracking
(`:750-756`). Fill-bar exclusion is correct (its high printed before the fill).
Stop-bar exclusion is a **pessimistic floor**: for a long stop, the bar's low ≤ stop
(why it stopped) and its high is known, but H1 OHLC **cannot** tell us whether the high
(favourable) came before or after the stop. Crediting the high would *assume* the
favourable move came first — unprovable. So a bar-1 death mechanically reports MFE=0 even
if price poked up first. This is a genuine information loss on exactly one bar, and it is
**fundamentally unknowable from H1 data.**

---

## The fix

### Which column to use once both exist (bind this into any analysis)
- **Loser autopsy → `mfe_intrade_r`.** "Did this loser ever look good *before it died*?"
  Fill → stop only. This is the ONLY correct MFE for classifying losers.
- **Winner study → `mfe_r`.** "How far past +2R does a winner run?" Full 48-bar window,
  incl. after exit — which is the whole point *for winners* (target-distance research).
- **Never** use `mfe_r` on losers: it keeps running after the stop (~67% peak after death),
  so it mislabels dead-on-arrival losers as "had profit." See LOSER_AUTOPSY_PLAYBOOK §2 table.

**1. New column `mfe_intrade_r` (ADD — do NOT modify `mfe_r`).**
Max favourable excursion over bars `[fill+1 … exit_bar − 1]` only — fill bar excluded,
**post-stop bars excluded by truncating the walk at the latched exit bar.** Everything
after the exit is not part of the trade, so there is nothing to guess — we control
Problem A **confidently**. `mfe_r` stays as-is (it still serves the *winner*
target-distance study, where full-window run-past-2R is the point).
- For losers `mfe_intrade_r ∈ [0, 2)` by construction (reaching +2R = a `tp` win).
- Why an ADD, not a change: detection and the exit stay frozen, so **every existing
  column stays byte-identical** after the re-run — no prior analysis on existing columns
  is voided. Only new columns appear.

**2. Stop-bar handling — bound it, don't fake it.**
Add, for SL exits, the stop bar's favourable extreme and a flag:
- `sl_bar_best_favor_r` = the stop bar's favourable move in R (long: `(bar_high−entry)/r_dist`).
- `sl_bar_reached_1r_ambiguous` = True when that extreme ≥ +1R AND the stop was hit in
  the same bar (both +1R and −1R touched in one bar → order unknowable). Report the COUNT
  of these so we know exactly how much uncertainty the taxonomy carries.
- When `sl_bar_best_favor_r < 1R`: the loser *unambiguously* never reached the half-target,
  regardless of intrabar order — a firm classification.

**3. Batch with the OB-penetration column (see below) — ONE Discovery re-run, not two.**

---

## Batch rider — OB penetration depth (hypothesis)
When price returns to the OB, how far past the proximal (entry) line does it poke before
reversing? If setups on average only poke ~X% into the zone, entries could sit deeper for
a tighter stop / better RR. Approximable now from `mae_r` + OB geometry (`ob_range_atr`,
entry vs `sl`), but cleaner as a dedicated `ob_penetration_depth` column computed in the
same walk. **Include it in the batch re-run** so we don't pay for two runs. Park until then.

---

## Implementation checklist (for the new chat)
1. Add `mfe_intrade_r` (+ `sl_bar_best_favor_r`, `sl_bar_reached_1r_ambiguous`,
   `ob_penetration_depth`) in `h1_only_simulator.py`, computed inside the existing walk.
2. TRUTH_LEDGER.md row for each (source `file:line`, timing class, population). New column
   → `gen_column_buckets` classify (else CI goes red).
3. Guard test: on a hand-built loser, `mfe_intrade_r` ignores post-stop bars; on a
   fill-bar death it is 0; ambiguous-stop-bar flag fires only when both levels in one bar.
4. Re-run **EURUSD Discovery only** (2008–2016; cached windows) — NOT the full 18yr.
5. Repoint `CANONICAL.md` (path + new shape) in the SAME commit; archive the superseded run.
6. Re-derive the taxonomy buckets 3/4 on the new column; update the snapshot below.

**Acceptance:** existing columns byte-identical (diff proves ADD-only); `mfe_intrade_r ≥ 0`,
`< 2` for all losers; `mfe_intrade_r ≤ mfe_r` always; ambiguous count reported.

---

## CURRENT RUN SNAPSHOT — canonical `h1only_20080102_20161231`, EURUSD Discovery
> Provisional per Rule 4; re-derive after any re-run. Generated by the autopsy scripts
> (population = resolved `tp|sl`). **CLEAN facts are trustworthy now; PROVISIONAL facts
> depend on the contaminated `mfe_r` and must be re-derived after the fix.**

**Baseline tiers**
| tier | N | W | L | WR | meanR | sumR |
|---|---|---|---|---|---|---|
| all resolved (no news filter) | 2,153 | 641 | 1,512 | 29.8% | −0.107 | −230 |
| LOOSE `news_fill==0` (frozen baseline) | 1,477 | 445 | 1,032 | 30.1% | −0.096 | −142 |
| STRICT (also `news_open==0`, autopsy) | 1,254 | 349 | 905 | 27.8% | −0.165 | −207 |
| news-open segment removed by strict | 223 | 96/127 | — | 43.0% | +0.291 | +65 |

**Strict clean losers (N=905) — CLEAN (trustworthy):**
- died in ≤1 bar: **515 (56.9%)**
- stop bar was a sweep (`sl_bar_was_sweep`): 476 (52.6%)
- swept then ran +1R: 106 (11.7%) · swept then ran +2R: 74 (8.2%)
- MFE peak AFTER stop (the contamination): 605 (66.9%)

**Strict losers — PROVISIONAL (re-derive after MFE fix):**
- instant death (mfe0 & ≤1 bar): 138 · slow bleed (mfe0 & >1 bar): 11
- weak reaction (0–1R, peak pre-stop): 85 · real give-back (≥1R, peak pre-stop): 66
