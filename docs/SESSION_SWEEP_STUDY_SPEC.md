# SESSION HIGH/LOW SWEEP + BREAK — BACKTEST STUDY (Spec)

**Status:** APPROVED to build on BACKTEST ONLY. Measure-first. Live only if it earns it.
**Date:** 2026-07-21
**This is a SPEC. It defines what to build — it is not built. Read it, verify every `file:line` against live code, then build in a fresh chat.**

---

## 0. RULES (read first)

- **Code is truth.** Before stating any column meaning/behaviour, quote the live `file:line` in the same response. Every `.md` (incl. this one) is background, not truth. If this doc and the code disagree, code wins — flag it.
- This is a **DETECTION/LOGGING** addition → rides the NEXT canonical baseline; does not retro-fix old CSVs.
- **Measure-first.** Adds columns + a per-pair study. Does NOT filter/gate/score/touch live. Any live change is a SEPARATE, later decision that requires the luck test to pass first.
- **Truth-ledger gate:** no new `trades.csv` column ships without a row in `TRUTH_LEDGER.md` (source `file:line`, when stamped, population). Alert-time snapshots frozen at the yield, never read live at row-build.
- **No sycophancy, plain English, bullets not paragraphs.**
- **Scope:** live-traded pairs only — EURUSD, USDJPY, NZDUSD, USDCHF, GOLD. NAS100/others are `backtest_only` — never judge live behaviour on them.

---

## 1. WHY (plain words + what the vets proved)

- A "session sweep" = price grabs the HIGH or LOW of a prior session (Asia / London / NY) — a known pool of stop orders — then reverses. A "session break" = price closes cleanly THROUGH that level and holds (continuation), not a wick-and-reverse.
- Web research (respectable ICT sources) says this is a **real, pair-specific edge — NOT uniform:**
  - **EURUSD (and GBP):** London open sweeps the **Asian range** high/low, then reverses into the daily trend. London sets EURUSD/GBP daily direction on ~70-80% of days. **Strongest here.**
  - **USDJPY:** the opposite — real directional moves happen DURING Asia. One cited backtest: **~63% win rate on USDJPY vs ~22% on GBPUSD** for the SAME rule. Same rule, wildly different by pair.
- The whole point of this study: **it is pair-specific.** If a pair shows the OPPOSITE effect, that is a KILL signal for that pair, not a reason to build. Per-pair slice is the deliverable, not a global number.

---

## 2. WHAT WE ALREADY HAVE (verified) — and what is BROKEN in it

- Session H/L sweep detection EXISTS but only as a **Phase-1 email badge**, NOT a backtest column:
  - `_session_hl_until` (`smc_detector.py:1180-1225`) — session high/low up to the anchor bar (clipped to anchor — no future bars).
  - `SESSION_WINDOWS_UTC` (`smc_detector.py:449-453`): Asia `(22,7)`, London `(7,12)`, NY `(12,17)`.
  - Per-pair session map `PAIR_SESSION_TAGS` (`smc_detector.py:456-463`).
  - Tag emission `{sess}_high`/`{sess}_low` (`smc_detector.py:1258-1266`).
- **It is NOT in `trades.csv`** — grep of `h1_only_simulator.py` for session-sweep columns = zero hits. So the backtest cannot currently answer "did sweeping the London low matter for r_realised."

- **🚨 THE BLOCKING DEFECT — `_session_hl_until` IS DST-BROKEN. DO NOT REUSE IT AS-IS.**
  - `SESSION_WINDOWS_UTC` is a HARDCODED UTC window (`smc_detector.py:449-453`) and `_session_hl_until` applies those fixed hours regardless of season (`smc_detector.py:1197-1207`).
  - London/NY sessions are defined by LOCAL clocks that shift with DST. London open = 08:00 local = **07:00 UTC in winter but 06:00 UTC in summer (BST)**. NY shifts too. A fixed-UTC window is off by ~1h for roughly half of every year → the session H/L is measured over the WRONG bars for half of history → the whole study is polluted.
  - **The system ALREADY has the correct pattern — mirror it, don't invent one:** `_ts_hour_ny` (`h1_only_simulator.py:109-111`) converts each timestamp to `America/New_York` LOCAL hour with DST resolved PER CANDLE via ZoneInfo. The comment at `h1_only_simulator.py:82-92` states outright that fixed-UTC session buckets are wrong and that zone-conversion "self-corrects because the zone conversion carries the DST offset." The `session`/`ob_session`/`fill_session` columns already use this DST-honest path (`TRUTH_LEDGER.md:71-73`, `:207`).

---

## 3. THE STUDY (backtest-first, measure only — NO gate)

### 3a. Build DST-correct session windows (the real work)
- Define each session by its LOCAL open/close in its OWN timezone, resolved PER CANDLE via ZoneInfo — the `_ts_hour_ny` pattern (`h1_only_simulator.py:109-111`), NOT `SESSION_WINDOWS_UTC`:
  - **Asia** — Tokyo (`Asia/Tokyo`); note: Japan does NOT observe DST, so Asia is stable, but resolve it through the same local-tz path for consistency.
  - **London** — `Europe/London` (observes BST — this is where the ±1h error lives).
  - **New York** — `America/New_York` (observes EDT).
- Pick the local session hours deliberately and STATE them in the spec you implement (e.g. Asia 00:00–09:00 JST, London 08:00–16:00 London, NY 08:00–17:00 NY) — align them to how the vets define the killzones, and cross-check against the existing killzone windows the system already trusts so we do not invent a fourth definition of "London".
- Compute each session's H/L per day from candles whose LOCAL time falls in that session's LOCAL window. Point-in-time: only bars ≤ the alert bar. No future leak.
- **One implementation.** Either fix `_session_hl_until` to be DST-honest (preferred — kills the live badge bug at the same time) OR write the backtest version against the shared `_ts_hour_ny`-style helper. Do NOT create a second, parallel, still-UTC session definition. Flag which path you took and why.

### 3b. Detect BOTH sweep AND break (user requirement — do both)
- Reuse the SAME mechanism the daily/weekly pools use for sweep vs break — do not invent new geometry (one concept, one implementation). Find where PD/PW pool sweep-vs-break is decided (`pool_builder.py` + the sweep/break classifier) and feed the session levels into it.
- **Sweep** = wick takes the session level, close returns back inside (liquidity grab, reversal-flavoured).
- **Break** = close clears the session level and holds (continuation-flavoured).
- Log both — they are different signals and the study must separate them.

### 3c. Columns to add to `trades.csv` (frozen at ALERT time, point-in-time)

| Column | Meaning |
|---|---|
| `session_level_event` | `sweep` / `break` / `none` — what happened at the nearest session level before entry |
| `session_level_which` | `asia` / `london` / `ny` / `none` — which session's pool |
| `session_level_side` | `high` / `low` / `none` |

- (Adjust names to match existing naming conventions in the row — check neighbours like `swept_*` / `pool_*` before finalising.)

### 3d. Slice, PER PAIR (this is the whole point — never pool pairs)
- `r_realised` and win-rate bucketed by (`session_level_event` × `session_level_which` × pair).
- Test the two vet claims explicitly: EURUSD Asia-sweep-at-London; USDJPY Asia-session behaviour.
- Sweep vs break split out separately — they may point opposite ways.
- Bootstrap-CI + per-quarter-consistency (`backtest/RECOMMENDATIONS.md`). Session buckets get thin fast per-pair — the luck test is mandatory, not optional.

### 3e. Decision rule
- A pair's bucket with expectancy CI clearly above the rest AND holding across quarters → real per-pair lever → THEN (separate approval) propose adding to live.
- Opposite effect on a pair → KILL for that pair. Flat/in-noise → shelve; it was cheap.
- Per CLAUDE.md: data + SMC must AGREE. The vet mechanism (Asia→London sweep) is sound, so a positive EURUSD/USDJPY result is a *conclusion*; a positive result on a pair with no mechanism is a *discussion point* (possible overfit), not a gate.

---

## 4. VERIFY (no blind spots)

1. **DST guard (the critical one):** take a London-session trade in SUMMER and one in WINTER; assert the session-H/L window covers the correct LOCAL 08:00–16:00 bars in BOTH — i.e. the UTC bars used SHIFT by an hour between the two. If they don't shift, DST is still broken. This test is the whole reason the study is trustworthy.
2. **Recompute audit:** independently rebuild `session_level_*` from raw session H/L + the entry price, assert equals emitted columns row-for-row (the Area-C 0-mismatch method).
3. **Point-in-time guard:** assert session H/L never uses a bar after the alert.
4. **Sweep-vs-break guard:** a fixture where price wicks-and-returns must label `sweep`; a fixture where price closes-through-and-holds must label `break`.
5. Ledger rows + `tests/test_truth_ledger.py` green for every new column.

---

## 5. LANDMINES (verified)

- **`_session_hl_until` is DST-broken** — §2. This is the one that silently corrupts half of history. Fix it or replace it; never reuse the UTC windows as-is.
- **MT5 clock-era problem is RESOLVED (2026-07-20)** — commits `f783ab04` + `608ec58e`; caches rebuilt to true-UTC labels; guards `test_mt5_clock` / `test_mt5_clock_import` green. Candle labels are now correct UTC, so once §3a resolves sessions by local-tz-per-candle, the underlying clock is sound. Do NOT re-raise the old ±1h era concern — it's dead.
- **No chart work.** This is backtest measurement only; there are no charts to draw. (Any future live/chart step is out of scope for this spec.)
- These session badge helpers were silently DEAD for a period (index-poisoning → `(None,None)`), fixed as D5 (`TRUTH_LEDGER.md:299`). Any read of the OLD Phase-1 tags pre-fix is void — build the backtest column fresh, don't trust historical tag output.
- Session tolerance for "price sits on the level" is ATR-scaled in the existing tag logic — confirm the reused sweep/break mechanism keeps that (so Gold's larger ATR doesn't distort vs FX).

---

## 6. SOURCES (web, 2026-07-21)

- London open sweeps Asian range, best on EURUSD/GBP: https://arongroups.co/technical-analyze/ict-london-open-strategy/
- Asian range sweep strategy, EUR/GBP best: https://fxnx.com/en/blog/master-the-ict-asian-range-forex-strategy
- Per-pair win-rate divergence (USDJPY ~63% vs GBPUSD ~22%): https://fxglory.com/learn/forex-strategies/123-forex-trading-strategy/
- JPY moves during Asia (exception to mark-don't-trade): innercircletrader / arongroups killzone guides.
