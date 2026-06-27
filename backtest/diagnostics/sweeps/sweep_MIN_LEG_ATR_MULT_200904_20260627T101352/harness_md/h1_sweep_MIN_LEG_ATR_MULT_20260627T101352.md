# Harness 1 â knob sweep: MIN_LEG_ATR_MULT

- pairs:  | window: 2009-04-01..2009-04-30 | grid_mode: absolute | slice_mode: **B**
- non-swept knobs frozen at live defaults; risk_usd=250.0


## Honest weaknesses
- One knob at a time: interaction effects are NOT explored; a best value here does not compose into a best joint config.
- Conditional on the ~720-day yfinance window â one regime sample. In-sample; diagnostic, not tuning truth.
- `n_swings` is a window-end census of survivors, not a per-bar experience.