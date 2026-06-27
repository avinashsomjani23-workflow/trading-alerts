# Harness 1 â knob sweep: MIN_LEG_ATR_MULT

- pairs: EURUSD,USDJPY,NZDUSD,USDCHF,NAS100,GOLD | window: 2008-10-01..2008-10-31 | grid_mode: absolute | slice_mode: **B**
- non-swept knobs frozen at live defaults; risk_usd=250.0


### EURUSD
| value | base | swings | OBs | CHoCH | BOS | alerts | filled | sumR | $PnL | WR | exp_R | avg_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 |  | 194 | 18 | 21 | 55 | 37 | 14 | 0.458 | 114.5 | 0.2727 | 0.0327 | 6.15 |
| 1.5 | âœ“ | 180 | 16 | 21 | 54 | 36 | 14 | 0.458 | 114.5 | 0.2727 | 0.0327 | 6.15 |
| 2.0 |  | 150 | 15 | 20 | 55 | 30 | 14 | -0.034 | -8.5 | 0.25 | -0.0024 | 6.105 |
| 2.5 |  | 135 | 15 | 21 | 53 | 31 | 14 | -0.034 | -8.5 | 0.25 | -0.0024 | 6.158 |

### USDJPY
| value | base | swings | OBs | CHoCH | BOS | alerts | filled | sumR | $PnL | WR | exp_R | avg_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 |  | 198 | 18 | 34 | 56 | 33 | 11 | 6.422 | 1605.5 | 0.7143 | 0.5838 | 5.893 |
| 1.5 | âœ“ | 187 | 19 | 30 | 59 | 33 | 11 | 6.422 | 1605.5 | 0.7143 | 0.5838 | 5.893 |
| 2.0 |  | 154 | 19 | 35 | 51 | 34 | 10 | 4.047 | 1011.75 | 0.5714 | 0.4047 | 5.615 |
| 2.5 |  | 135 | 13 | 22 | 45 | 29 | 8 | 5.944 | 1486.0 | 0.8 | 0.743 | 5.227 |

### NZDUSD
| value | base | swings | OBs | CHoCH | BOS | alerts | filled | sumR | $PnL | WR | exp_R | avg_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 |  | 191 | 21 | 16 | 49 | 44 | 15 | -0.897 | -224.25 | 0.3333 | -0.0598 | 6.0 |
| 1.5 | âœ“ | 180 | 22 | 17 | 45 | 46 | 17 | 3.801 | 950.25 | 0.4 | 0.2236 | 6.136 |
| 2.0 |  | 156 | 19 | 17 | 42 | 44 | 14 | 4.653 | 1163.25 | 0.4545 | 0.3324 | 6.316 |
| 2.5 |  | 129 | 17 | 14 | 30 | 40 | 11 | 3.797 | 949.25 | 0.5714 | 0.3452 | 6.467 |

### USDCHF
| value | base | swings | OBs | CHoCH | BOS | alerts | filled | sumR | $PnL | WR | exp_R | avg_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 |  | 195 | 14 | 24 | 53 | 47 | 11 | -2.674 | -668.5 | 0.25 | -0.2431 | 5.8 |
| 1.5 | âœ“ | 182 | 13 | 23 | 52 | 22 | 10 | -4.111 | -1027.75 | 0.1429 | -0.4111 | 5.769 |
| 2.0 |  | 154 | 10 | 25 | 42 | 19 | 8 | -2.193 | -548.25 | 0.2 | -0.2741 | 5.667 |
| 2.5 |  | 132 | 12 | 25 | 43 | 22 | 8 | -1.193 | -298.25 | 0.25 | -0.1491 | 5.417 |

### GOLD
| value | base | swings | OBs | CHoCH | BOS | alerts | filled | sumR | $PnL | WR | exp_R | avg_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 |  | 202 | 21 | 29 | 61 | 98 | 12 | 8.993 | 2248.25 | 0.5455 | 0.7494 | 5.955 |
| 1.5 | âœ“ | 186 | 20 | 25 | 55 | 103 | 11 | 9.993 | 2498.25 | 0.6 | 0.9085 | 5.916 |
| 2.0 |  | 156 | 21 | 26 | 52 | 105 | 10 | 8.041 | 2010.25 | 0.625 | 0.8041 | 5.958 |
| 2.5 |  | 136 | 20 | 23 | 47 | 107 | 9 | 9.492 | 2373.0 | 0.625 | 1.0547 | 5.661 |

## Honest weaknesses
- One knob at a time: interaction effects are NOT explored; a best value here does not compose into a best joint config.
- Conditional on the ~720-day yfinance window â one regime sample. In-sample; diagnostic, not tuning truth.
- `n_swings` is a window-end census of survivors, not a per-bar experience.