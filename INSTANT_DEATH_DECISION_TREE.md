# Instant-Death Investigation — Decision Tree

**Mission:** Eliminate the ~64% of true losers that die instantly (MFE ≈ 0, straight to −1R SL).
**Data:** `backtest/results/h1only_20080102_20251231/trades.csv`, 23,369 filled trades, 2008–2025, 10 pairs.
**Status:** LIVE investigation. Do not score/filter anything until a node is marked SHIP.

---

## Verified facts (do not re-litigate)

- **Instant death** = `exit_reason=="sl"` AND `mfe_r <= 0.05`. All are exactly −1.0R. No breakeven contamination. VERIFIED against raw entry/SL/exit prices.
- **True losers** = `r_realised == -1.0` → **12,228** (NOT 16,076; the SL bucket also holds 3,848 breakeven-stop 0.0R exits).
- **Instant deaths = 7,867 = 64.3% of true losers.** This is THE number.
- MFE curve is flat past 0.05 → deaths cluster at *exactly zero* favorable move, not "small MFE." Hard instant reversal.
- Dollars (`pnl_usd`) = fixed ~$250/trade risk, cumulative 18yr×10pairs. NOT an account balance. **R is truth.**
- Overall system expectancy: **−0.0437R/trade.** Negative. Whole system has no edge on raw fills.

---

## Finding 1 — OB size predicts DEATH, not WINS (structural loss, lucky wins)

- OB size (`ob_range_atr`) moves instant-death rate by **29 pts** across quintiles, but win rate by only **7 pts**.
- → Losing is STRUCTURAL (predictable). Winning is ~LUCK (barely predictable).
- **Strategy consequence:** we cannot engineer wins. We CAN avoid structural deaths. That is the whole game.
- Small OB (<0.658 ATR) is a barbell: worst losers AND best wins (small-OB win pays 1.99R vs 1.48R large). → A pure size gate does NOT help (all 13 floors tested stay negative). Cutting size removes fat wins too.

## Finding 2 — Fill-in-killzone is worse (contradicts ICT doctrine → DISCUSSION POINT)

- `fill_in_killzone` numbers below are **instant-death rate = deaths / all filled in that group**:
  - Fill INSIDE killzone: **38.0%** death, −0.065R
  - Fill OUTSIDE killzone: **27.8%** death, −0.014R
- Independent of OB size (OB range identical in both groups; effect survives the 2×2). NOT a confound.
- Stable: outside-KZ beats inside-KZ in **14/18 years**.
- Mechanism: **UNKNOWN.** Volatility theory TESTED and REJECTED (N1): inside-KZ is not higher vol (z −0.035 vs +0.050), and killzone still hurts equally in both vol buckets. Fills come slightly faster inside KZ (4.0h vs 5.5h, London-leaning) but every other feature (trend/direction/sweep/reversal) is identical. **Real effect, no visible cause in logged columns.** → NOT scored; effect stands, cause open.

## The 2×2 that matters (all filled trades)

| OB size | fill in KZ | n | exp_R | death% | win% | total R |
|---|---|---|---|---|---|---|
| large | outside | 7866 | **−0.011** | 24.3 | 28.2 | −84 |
| large | inside | 10837 | −0.057 | 34.1 | 26.8 | −614 |
| small | outside | 1994 | −0.028 | 41.4 | 25.7 | −55 |
| **small** | **inside** | **2672** | **−0.101** | **53.6** | 24.5 | **−268** |

- **Worst cell = small OB + inside killzone: −0.101R, 53.6% instant death.** User's hypothesis: "small OBs in killzone are to be killed." Data agrees.
- Cutting that one cell: remove 2,672 trades (11.4%) → system exp −0.0437 → **−0.0364R.** Helps but still negative. It trims the worst bleed but is not the whole fix.

---

## Decision nodes

- [x] **N1 — Verify volatility mechanism. REJECTED.** Inside-KZ is not higher-vol; killzone hurts equally in both vol buckets. Cause of killzone effect remains unknown. Effect itself stands (14/18 yrs).
- [ ] **N2 — Confirmation-entry re-sim.** Enter only after a close back into the zone (not on touch). Only honest way to test if we keep small-OB wins while dropping instant deaths. BLOCKER: not answerable from CSV — needs an 18yr backtest re-run with the rule coded. STATUS: explained, awaiting go.
- [ ] **N3 — Decide the cut.** Options once N1/N2 land: (a) kill small-OB-in-killzone cell, (b) fill-time vol gate, (c) confirmation entry. Ship only what survives out-of-sample + per-year stability.

---

## Guardrails for this investigation

- Every % must state its denominator (of losers / of true losers / of all filled). No bare percentages.
- Data-vs-SMC: any finding that fights methodology (e.g. killzone) stays a DISCUSSION POINT until mechanism is proven.
- No gate ships without per-year + out-of-sample stability and expectancy (not just win rate).
- We are hunting to ELIMINATE structural deaths, not to manufacture wins (wins are ~luck).
