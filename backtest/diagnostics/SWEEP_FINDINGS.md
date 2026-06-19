# Knob-Sweep Findings (persistent ledger)

Auto-appended after every sweep. Metrics are PROXIMAL-ONLY, score-gated (>=4). Each block = one sweep run.

> **CORRECTION (2026-06-19).** The FIRST TWO sweeps below — BOS_ATR_MULT and
> MIN_LEG_ATR_MULT, both the 2024-07-01..2024-07-31 (BAU) AND 2026-03-01..2026-03-31
> (War) blocks — were produced with a BUGGED filter that only excluded
> `never_filled`. It wrongly counted `distal_killed` rows (no position ever held)
> as "filled" with r=0, so their **filled counts are inflated and expectancy is
> deflated**. Worst hit: EURUSD-BAU cells reading "6 filled / 0.0" were really
> ZERO real trades. The bug is FIXED (now imports `_EXCLUDE_REASONS` from
> h1_only_reporting — one source of truth). These four blocks are DIRECTIONAL
> ONLY; the genuinely-filled cells (USDJPY/NAS100/GOLD/EURUSD-War) still indicate
> direction. Re-run on the full pooled dataset before trusting any number.
> All blocks AFTER this note use the corrected filter.

---
## BOS_ATR_MULT — 2024-07-01..2024-07-31 (grid_mode multiplier, risk $250)
_run 2026-06-18 19:56 UTC; pairs EURUSD,USDJPY,NAS100,GOLD_

| pair | value | base | filled | sumR | WR | exp_R | avg_score | OBs | recon |
|---|---|---|---|---|---|---|---|---|---|
| EURUSD | 0.4 | ✓ | 6 | 0.0 | 0.0 | 0.0 | 5.455 | 14 | ok |
| EURUSD | 0.5 |  | 5 | 0.0 | 0.0 | 0.0 | 5.7 | 12 | ok |
| EURUSD | 0.7 |  | 5 | 0.0 | 0.0 | 0.0 | 5.9 | 12 | ok |
| EURUSD | 1.0 |  | 6 | -1.0 | 0.0 | -0.1667 | 6.3 | 10 | ok |
| EURUSD | 1.2 |  | 6 | -1.0 | 0.0 | -0.1667 | 6.4 | 9 | ok |
| USDJPY | 0.4 | ✓ | 7 | 4.118 | 0.2857 | 0.5883 | 5.818 | 17 | ok |
| USDJPY | 0.5 |  | 5 | 4.118 | 0.4 | 0.8236 | 5.889 | 16 | ok |
| USDJPY | 0.7 |  | 6 | 2.431 | 0.1667 | 0.4052 | 5.2 | 12 | ok |
| USDJPY | 1.0 |  | 5 | 2.431 | 0.2 | 0.4862 | 5.333 | 10 | ok |
| USDJPY | 1.2 |  | 5 | 2.431 | 0.2 | 0.4862 | 5.333 | 9 | ok |
| NAS100 | 0.4 | ✓ | 4 | 0.0 | 0.0 | 0.0 | 4.9 | 18 | ok |
| NAS100 | 0.5 |  | 6 | 0.0 | 0.0 | 0.0 | 4.791 | 14 | ok |
| NAS100 | 0.7 |  | 6 | 0.0 | 0.0 | 0.0 | 4.855 | 14 | ok |
| NAS100 | 1.0 |  | 8 | -1.0 | 0.0 | -0.125 | 4.725 | 10 | ok |
| NAS100 | 1.2 |  | 5 | 0.0 | 0.0 | 0.0 | 4.875 | 9 | ok |
| GOLD | 0.4 | ✓ | 4 | 0.0 | 0.0 | 0.0 | 5.125 | 11 | ok |
| GOLD | 0.5 |  | 5 | 0.0 | 0.0 | 0.0 | 4.9 | 11 | ok |
| GOLD | 0.7 |  | 5 | 0.0 | 0.0 | 0.0 | 5.25 | 9 | ok |
| GOLD | 1.0 |  | 5 | 0.0 | 0.0 | 0.0 | 5.3 | 9 | ok |
| GOLD | 1.2 |  | 4 | 0.0 | 0.0 | 0.0 | 5.375 | 7 | ok |

---
## MIN_LEG_ATR_MULT — 2024-07-01..2024-07-31 (grid_mode multiplier, risk $250)
_run 2026-06-18 19:57 UTC; pairs EURUSD,USDJPY,NAS100,GOLD_

| pair | value | base | filled | sumR | WR | exp_R | avg_score | OBs | recon |
|---|---|---|---|---|---|---|---|---|---|
| EURUSD | 1.25 |  | 6 | 0.0 | 0.0 | 0.0 | 5.455 | 14 | ok |
| EURUSD | 1.5 | ✓ | 6 | 0.0 | 0.0 | 0.0 | 5.455 | 14 | ok |
| EURUSD | 1.75 |  | 6 | 0.0 | 0.0 | 0.0 | 5.455 | 14 | ok |
| EURUSD | 2.0 |  | 6 | 0.0 | 0.0 | 0.0 | 5.636 | 14 | ok |
| USDJPY | 1.25 |  | 7 | 4.118 | 0.2857 | 0.5883 | 5.818 | 17 | ok |
| USDJPY | 1.5 | ✓ | 7 | 4.118 | 0.2857 | 0.5883 | 5.818 | 17 | ok |
| USDJPY | 1.75 |  | 5 | 4.118 | 0.4 | 0.8236 | 6.25 | 17 | ok |
| USDJPY | 2.0 |  | 6 | 2.431 | 0.1667 | 0.4052 | 5.333 | 13 | ok |
| NAS100 | 1.25 |  | 4 | 0.0 | 0.0 | 0.0 | 5.0 | 18 | ok |
| NAS100 | 1.5 | ✓ | 4 | 0.0 | 0.0 | 0.0 | 4.9 | 18 | ok |
| NAS100 | 1.75 |  | 5 | 0.0 | 0.0 | 0.0 | 4.818 | 19 | ok |
| NAS100 | 2.0 |  | 7 | 0.0 | 0.0 | 0.0 | 4.746 | 20 | ok |
| GOLD | 1.25 |  | 7 | 0.0 | 0.0 | 0.0 | 5.143 | 12 | ok |
| GOLD | 1.5 | ✓ | 4 | 0.0 | 0.0 | 0.0 | 5.125 | 11 | ok |
| GOLD | 1.75 |  | 4 | 0.0 | 0.0 | 0.0 | 5.125 | 12 | ok |
| GOLD | 2.0 |  | 9 | 1.902 | 0.1111 | 0.2113 | 5.25 | 15 | ok |

---
## BOS_ATR_MULT — 2026-03-01..2026-03-31 (grid_mode multiplier, risk $250)
_run 2026-06-18 21:21 UTC; pairs EURUSD,USDJPY,NAS100,GOLD_

| pair | value | base | filled | sumR | WR | exp_R | avg_score | OBs | recon |
|---|---|---|---|---|---|---|---|---|---|
| EURUSD | 0.4 | ✓ | 7 | 1.393 | 0.1429 | 0.199 | 5.417 | 14 | ok |
| EURUSD | 0.5 |  | 5 | 1.393 | 0.2 | 0.2786 | 5.6 | 12 | ok |
| EURUSD | 0.7 |  | 3 | 1.393 | 0.3333 | 0.4643 | 5.375 | 10 | ok |
| EURUSD | 1.0 |  | 2 | 0.0 | 0.0 | 0.0 | 5.714 | 10 | ok |
| EURUSD | 1.2 |  | 1 | 0.0 | 0.0 | 0.0 | 5.833 | 8 | ok |
| USDJPY | 0.4 | ✓ | 9 | -1.0 | 0.0 | -0.1111 | 5.389 | 11 | ok |
| USDJPY | 0.5 |  | 9 | -1.0 | 0.0 | -0.1111 | 5.389 | 11 | ok |
| USDJPY | 0.7 |  | 7 | -1.0 | 0.0 | -0.1429 | 5.5 | 10 | ok |
| USDJPY | 1.0 |  | 7 | -1.0 | 0.0 | -0.1429 | 5.5 | 8 | ok |
| USDJPY | 1.2 |  | 7 | -1.0 | 0.0 | -0.1429 | 5.5 | 7 | ok |
| NAS100 | 0.4 | ✓ | 6 | 1.505 | 0.1667 | 0.2508 | 5.267 | 16 | ok |
| NAS100 | 0.5 |  | 7 | 3.219 | 0.2857 | 0.4599 | 5.36 | 14 | ok |
| NAS100 | 0.7 |  | 7 | 3.219 | 0.2857 | 0.4599 | 5.46 | 12 | ok |
| NAS100 | 1.0 |  | 7 | 0.714 | 0.1429 | 0.102 | 5.678 | 10 | ok |
| NAS100 | 1.2 |  | 6 | 1.714 | 0.1667 | 0.2857 | 5.825 | 9 | ok |
| GOLD | 0.4 | ✓ | 5 | 1.293 | 0.2 | 0.2586 | 5.544 | 16 | ok |
| GOLD | 0.5 |  | 4 | 1.293 | 0.25 | 0.3232 | 5.5 | 13 | ok |
| GOLD | 0.7 |  | 5 | 1.293 | 0.2 | 0.2586 | 5.571 | 9 | ok |
| GOLD | 1.0 |  | 4 | 1.293 | 0.25 | 0.3232 | 5.9 | 8 | ok |
| GOLD | 1.2 |  | 4 | 1.293 | 0.25 | 0.3232 | 5.9 | 7 | ok |

---
## MIN_LEG_ATR_MULT — 2026-03-01..2026-03-31 (grid_mode multiplier, risk $250)
_run 2026-06-18 21:27 UTC; pairs EURUSD,USDJPY,NAS100,GOLD_

| pair | value | base | filled | sumR | WR | exp_R | avg_score | OBs | recon |
|---|---|---|---|---|---|---|---|---|---|
| EURUSD | 1.25 |  | 8 | 0.393 | 0.125 | 0.0491 | 5.583 | 15 | ok |
| EURUSD | 1.5 | ✓ | 7 | 1.393 | 0.1429 | 0.199 | 5.417 | 14 | ok |
| EURUSD | 1.75 |  | 6 | 2.393 | 0.1667 | 0.3988 | 5.455 | 12 | ok |
| EURUSD | 2.0 |  | 8 | 1.393 | 0.125 | 0.1741 | 5.364 | 11 | ok |
| USDJPY | 1.25 |  | 10 | -1.0 | 0.0 | -0.1 | 5.55 | 12 | ok |
| USDJPY | 1.5 | ✓ | 9 | -1.0 | 0.0 | -0.1111 | 5.389 | 11 | ok |
| USDJPY | 1.75 |  | 10 | -1.0 | 0.0 | -0.1 | 5.7 | 12 | ok |
| USDJPY | 2.0 |  | 10 | -1.0 | 0.0 | -0.1 | 5.7 | 11 | ok |
| NAS100 | 1.25 |  | 6 | 1.505 | 0.1667 | 0.2508 | 5.267 | 15 | ok |
| NAS100 | 1.5 | ✓ | 6 | 1.505 | 0.1667 | 0.2508 | 5.267 | 16 | ok |
| NAS100 | 1.75 |  | 6 | 1.505 | 0.1667 | 0.2508 | 5.425 | 15 | ok |
| NAS100 | 2.0 |  | 7 | 1.505 | 0.1429 | 0.215 | 5.267 | 14 | ok |
| GOLD | 1.25 |  | 4 | 1.293 | 0.25 | 0.3232 | 5.278 | 16 | ok |
| GOLD | 1.5 | ✓ | 5 | 1.293 | 0.2 | 0.2586 | 5.544 | 16 | ok |
| GOLD | 1.75 |  | 5 | 2.745 | 0.2 | 0.549 | 5.544 | 16 | ok |
| GOLD | 2.0 |  | 4 | 2.745 | 0.25 | 0.6863 | 5.278 | 16 | ok |
