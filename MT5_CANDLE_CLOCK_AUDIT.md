# MT5 Candle Clock Audit — 2026-07-16

**Finding: the cached MT5 H1 candles are NOT uniformly UTC. The `-3h` fixed
conversion in `backtest/mt5_data/import_mt5.py` is only correct for part of the
history.** Discovered while building the news-fill column: 13,839 high-impact
ForexFactory events (page-embedded epoch timestamps, absolute UTC) were
spike-aligned against the cached candles.

## The empirical eras (label − true UTC, hours)

| Era | Range (label time) | EU-DST summer | EU winter | Server clock implied |
|-----|--------------------|---------------|-----------|----------------------|
| A | ≤ 2014-10-31 | 0 | **−1** | EET (UTC+2/+3, DST) |
| flip | 2014-11-01 → 2014-12-07 | — ambiguous, per-event votes conflict — | | |
| B | 2014-12-08 → 2024-10-26 | **+1** | 0 | UTC+3/+4 with DST |
| C | ≥ 2024-10-27 | 0 | 0 | fixed UTC+3 (the pinned era) |

- Seasons follow **EU DST boundaries** (Europe/Athens), proven by the 2–3
  week/year US-on/EU-off mismatch windows siding with winter every year.
- The 2026-06-23 "-3h, NO DST" pin (import_mt5.py:3-18) is TRUE for era C
  only. The week-open proof sampled mostly-recent behaviour and could not see
  the historical regime changes.
- Era C start = exactly the 2024 EU DST end; no ambiguity (both rules agree
  through winter, summer 2025 proves fixed).
- Proof: after applying the correction, all 36 pooled year-season cells
  2008–2025 peak at offset 0 (n=228–666 events each, ~16.6k event-bar checks).
  Uncorrected: era-A winters peaked −1, era-B summers +1, every year.

## What this does and does NOT affect

- Prices/OHLC are untouched — only the hour LABEL is wrong by ±1h in the
  affected seasons. Structure detection (BOS/CHoCH/OB geometry) is unaffected.
- **Affected: every clock-of-day derived backtest column for 2008–2024** —
  killzone flags (`ob_in_killzone`, `fill_in_killzone`, `killzone_alignment`),
  session buckets, IST/weekend gates, `alert_utc_hour` — all seasonally
  mislabeled by ±1h in era-A winters and era-B summers (~5 months/year each).
- Live is unaffected (Twelve Data feed, separate clock).
- The news columns (`news_fill`/`news_open`) already apply the correction —
  `backtest/news_enrichment.py::mt5_label_error_hours` is the single
  implementation of the era table.

## "Isn't it more likely the news was just delayed?" — no, and here is the proof

The natural objection: a wrong release time is a smaller claim than "my candle
feed is mis-clocked." But a delayed/mis-timed release and a clock error leave
DIFFERENT fingerprints, and what we see is unambiguously the clock:

- **Seasonal, not random.** A late release scatters (some +1, some 0, some +2,
  tied to nothing). What we see is a clean 0-in-summer / −1-in-winter split that
  is identical EVERY year within an era and flips on the exact clock-change
  weekend. A release cannot know the DST date; a clock does.
- **All 8 currencies move together on the same day.** A delay hits one release;
  the observed shift moves USD, CAD, EUR, GBP… by the identical amount at once.
  That is a property of the shared candle clock, not of any one publisher.
- **The boundaries are EU (Europe/Athens) DST dates, not US.** If the news
  schedule were the cause, errors would track the publisher's country clock.
  They track the European broker's server clock — unrelated to when NFP prints.
- **Three independent sources agree on the news, disagree with the candles.**
  FF epochs, a second FF-derived dataset, and official publishers (BLS, ABS,
  Stats NZ) all give the same release times. Only the candle label dissents —
  so the candle is the outlier, not the news.

A delayed-release world is incompatible with all four facts simultaneously.
Only a server-clock offset produces this exact signature.

## Decision (owner, 2026-07-16): FIX AT SOURCE — RESOLVED (code shipped, uncommitted)

Fixed at source per `CLOCK_AND_NEWS_FIX_SPEC.md`, 2026-07-16:
- Era table extracted to `backtest/mt5_clock.py` (single implementation; Part A).
- `import_mt5.load_one` now maps the provisional −3h H1 label through the era
  table to TRUE UTC (Part B). D1/W1 keep the flat −3h (no hour-of-day meaning).
- Guard `backtest/test_mt5_clock_import.py` re-runs the spike alignment on the
  REBUILT candles: era-A winter and era-B summer cells peak at offset 0 after
  the fix, −1/+1 without it (bite proven). Flip-window rows keep the provisional
  label and stay re-derivable via `is_flip_window`.
- News enrichment stopped double-correcting (Part C) and became run-produced
  (Part D). Docs updated (Part E).

STATUS: code committed to the working tree, **not yet baselined**. The parquet
cache still holds the OLD −3h labels until `import_mt5.main()` is re-run; the
fix takes effect on the next fresh 18-year baseline (owner's explicit word
required, CLAUDE.md rule 5). Commit hash: pending (uncommitted at time of write).

Past time-of-day conclusions (killzone / alert-hour edge tests) inherited the
wrong hour and must be re-run on the fixed baseline before being cited again
(spec Part F — flag only, not re-opened this sitting).
