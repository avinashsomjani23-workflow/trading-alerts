# Known Limitations

Honest list of what this harness cannot do precisely. Read this before
trusting any number it produces.

## Data quality (yfinance)

- **M5 history:** ~60 days reliable. Phase 3 backtest only valid for the
  recent window (~last 60 days from run date).
- **M15 history:** ~2 years.
- **H1 history:** ~2 years.
- **No bid/ask spread.** Modelled as a fixed `spread_pips` per pair (from
  `config.json`). Real spread widens during news and Asian session.
- **No slippage.** All fills assumed at the modelled price.
- **No swap / overnight financing.** Multi-day trades will be slightly
  overstated in P&L.
- **Weekend gaps:** forex Sunday opens model badly. Trades held across
  weekends are flagged in the report; expect ±5% noise.
- **Time zone:** yfinance returns UTC. The harness keeps everything in UTC
  internally and converts to IST only for display.

## Scoring model

- **Phase 2 scoring is mirrored, not called.** The live scoring logic lives
  inline in `Phase2_Alert_Engine.py` main block. The harness reimplements
  it in `trade_simulator.py::score_ob_confluences()`. Any change to the
  live scoring rules will silently drift this harness until the mirror is
  updated. Cross-check by comparing harness scores against
  `phase2_scan_log.jsonl` for the same week.
- **No macro news input.** The live system calls Gemini for macro context.
  The harness skips this — backtests don't simulate news flow. News-blackout
  rules are also skipped; trades that fired during NFP/CPI/FOMC are flagged
  in the report but not auto-rejected.
- **No Gemini summary.** Reports do not include AI commentary.

## Phase 3 substitution

- NAS100 and Gold normally use Phase 3 (M5 CHoCH trigger).
- For weeks older than ~60 days, M5 data isn't reliable.
- Backtest falls back to Phase 2 limit-order model for older NAS/Gold weeks.
- **Report banners this clearly per pair per week.** A Phase 2-substitute
  result for NAS/Gold pre-2026-03 tells you about a hypothetical M15 system,
  NOT the live Phase 3 system. Do not tune Phase 3 based on this.

## Trade simulation

- **Same-bar SL + TP collision:** SL hits first (pessimistic). Count of
  collisions is reported so the drag is visible.
- **Limit order fill:** Assumed instant fill when M15 low/high touches
  the limit price (no requote, no partial fill).
- **No re-entry logic.** One trade per OB per direction per zone-life.
- **Time stop:** Trades not closed within `MAX_HOLD_HOURS` (default 72)
  are time-stopped at the last bar's close. Configurable.

## What this CAN'T tell you

- **Whether a vet would have taken the trade.** Mechanical grading only.
- **System stability under real news flow.** Bars don't carry sentiment.
- **Performance under different account sizes.** £-P&L is computed from
  a fixed risk amount (default £250/trade), not from a balance curve with
  compounding.
- **Slippage in fast markets.** Real fills can be 5-15 pips off the
  modelled price during news.

## Accuracy expectation

- **Signal generation:** ±5% vs what the live system would have produced
  (the detection code is the live code, modulo Phase 2 scoring mirror).
- **Realised P&L:** ±10-15% vs what would have actually filled in a live
  account.

If you need tighter, the path is Dukascopy tick data and a broker-spread
model. Out of scope per current decision.
