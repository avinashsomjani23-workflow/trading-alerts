# Harness 1 â knob sweep: MIN_LEG_ATR_MULT

- pairs: EURUSD,USDJPY,NZDUSD,USDCHF,NAS100,GOLD | window: 2009-04-01..2009-04-30 | grid_mode: absolute | slice_mode: **B**
- non-swept knobs frozen at live defaults; risk_usd=250.0


### EURUSD
| value | base | swings | OBs | CHoCH | BOS | alerts | filled | sumR | $PnL | WR | exp_R | avg_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 |  | 209 | 25 | 36 | 58 | 95 | 19 | 15.307 | 3826.75 | 0.625 | 0.8056 | 6.433 |
| 1.5 | âœ“ | 198 | 25 | 32 | 54 | 95 | 19 | 15.307 | 3826.75 | 0.625 | 0.8056 | 6.433 |
| 2.0 |  | 167 | 25 | 31 | 48 | 90 | 21 | 10.439 | 2609.75 | 0.5 | 0.4971 | 6.107 |
| 2.5 |  | 139 | 24 | 28 | 42 | 84 | 18 | 3.816 | 954.0 | 0.4667 | 0.212 | 5.96 |

### USDJPY
| value | base | swings | OBs | CHoCH | BOS | alerts | filled | sumR | $PnL | WR | exp_R | avg_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 |  | 222 | 21 | 27 | 59 | 125 | 13 | 4.642 | 1160.5 | 0.625 | 0.3571 | 5.317 |
| 1.5 | âœ“ | 210 | 20 | 26 | 56 | 125 | 12 | 4.8 | 1200.0 | 0.625 | 0.4 | 5.188 |
| 2.0 |  | 171 | 19 | 24 | 53 | 121 | 12 | 3.043 | 760.75 | 0.5714 | 0.2536 | 5.247 |
| 2.5 |  | 140 | 19 | 25 | 47 | 120 | 14 | 3.866 | 966.5 | 0.5556 | 0.2761 | 4.885 |

### NZDUSD
| value | base | swings | OBs | CHoCH | BOS | alerts | filled | sumR | $PnL | WR | exp_R | avg_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 |  | 198 | 24 | 27 | 54 | 79 | 12 | 0.585 | 146.25 | 0.4444 | 0.0487 | 6.077 |
| 1.5 | âœ“ | 185 | 23 | 26 | 54 | 78 | 12 | 0.585 | 146.25 | 0.4444 | 0.0487 | 6.12 |
| 2.0 |  | 145 | 21 | 20 | 52 | 68 | 15 | 5.592 | 1398.0 | 0.5833 | 0.3728 | 6.0 |
| 2.5 |  | 128 | 21 | 22 | 51 | 68 | 15 | 6.047 | 1511.75 | 0.5833 | 0.4031 | 6.0 |

### USDCHF
| value | base | swings | OBs | CHoCH | BOS | alerts | filled | sumR | $PnL | WR | exp_R | avg_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 |  | 214 | 23 | 36 | 65 | 81 | 15 | -2.649 | -662.25 | 0.3333 | -0.1766 | 7.0 |
| 1.5 | âœ“ | 200 | 23 | 33 | 64 | 81 | 15 | -4.228 | -1057.0 | 0.2857 | -0.2819 | 6.962 |
| 2.0 |  | 165 | 19 | 33 | 62 | 77 | 12 | -0.466 | -116.5 | 0.3636 | -0.0388 | 7.318 |
| 2.5 |  | 142 | 19 | 26 | 53 | 79 | 12 | -0.371 | -92.75 | 0.3636 | -0.0309 | 7.318 |

### GOLD
| value | base | swings | OBs | CHoCH | BOS | alerts | filled | sumR | $PnL | WR | exp_R | avg_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 |  | 195 | 26 | 34 | 55 | 92 | 17 | 1.39 | 347.5 | 0.3636 | 0.0818 | 5.307 |
| 1.5 | âœ“ | 180 | 26 | 33 | 53 | 88 | 16 | -0.131 | -32.75 | 0.3 | -0.0082 | 5.319 |
| 2.0 |  | 156 | 20 | 32 | 54 | 80 | 12 | 2.1 | 525.0 | 0.4444 | 0.175 | 5.271 |
| 2.5 |  | 136 | 20 | 29 | 53 | 82 | 12 | 0.93 | 232.5 | 0.3333 | 0.0775 | 5.16 |

## Honest weaknesses
- One knob at a time: interaction effects are NOT explored; a best value here does not compose into a best joint config.
- Conditional on the ~720-day yfinance window â one regime sample. In-sample; diagnostic, not tuning truth.
- `n_swings` is a window-end census of survivors, not a per-bar experience.