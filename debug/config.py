"""
Pair-aware thresholds for the review detector.

Numbers below are vet-calibrated, not arbitrary. Reasoning summary:

- A vet thinks in *time on the chart* and *ATR proximity*, not in raw bars
  or raw pips. So thresholds are expressed in H1 bars (time) and ATR
  multiples (price), never raw pips.

- Forex pairs trade ~24 H1 bars/day across overlapping sessions, but only
  ~12-14 of those carry institutional flow (London + NY overlap). Two
  trading days of stagnation = ~48 H1 bars. Beyond that, a dealing-range
  anchor is referencing flow that the market has already discounted.

- Gold and NAS100 move faster (single trending session + reaction) and
  decay reference levels sooner. ~30 H1 bars ≈ 1.5 trading days for these.

- "Untouched" is ATR-relative because raw pip distances are meaningless
  across instruments (a 20-pip wick on USDJPY is noise; on EURUSD it's
  significant). 0.25 * ATR is a vet's "didn't even look at it" threshold.

- Orphan BOS: a Major BOS without a follow-up swing within 15 H1 bars
  (Forex) / 10 (Gold/NAS) is the system tracking a stale break. Internal
  swings normally form within 5-10 bars on H1.

- Persistent placeholder: walls that haven't been promoted to confirmed
  within 30 bars indicate price has run too far or sideways for proper
  structure to form on the placeholder side.

- Runaway: |price - DR mid| > 1.5 * DR width with no internal CHoCH
  inside the new leg means the system is anchored to a range price has
  already abandoned.

These are intentionally LOOSE — better to over-flag and review, than to
miss a stale anchor and produce dead alerts. Tighten with usage data.
"""

# Default thresholds applied to any pair not explicitly listed below.
DEFAULTS = {
    "stale_anchor_bars":       48,    # H1 bars without a touch on the side
    "stale_anchor_atr_mult":   0.25,  # what counts as "touched"
    "orphan_bos_bars":         15,    # bars after Major BOS with no new swing
    "placeholder_bars":        30,    # placeholder DR side persists this long
    "runaway_dr_mult":         1.5,   # |price - mid| > N * range_width
}

# Pair-specific overrides. Keys MUST match the `name` field in config.json
# (the canonical instrument identifier used everywhere in Phase 1 state).
PAIR_THRESHOLDS = {
    "EURUSD": DEFAULTS,
    "USDJPY": DEFAULTS,
    "NZDUSD": DEFAULTS,
    "USDCHF": DEFAULTS,
    # Gold + NAS: faster TFs (entry on M5), structure decays sooner.
    "GOLD": {
        "stale_anchor_bars":       30,
        "stale_anchor_atr_mult":   0.25,
        "orphan_bos_bars":         10,
        "placeholder_bars":        24,
        "runaway_dr_mult":         1.5,
    },
    "NAS100": {
        "stale_anchor_bars":       30,
        "stale_anchor_atr_mult":   0.25,
        "orphan_bos_bars":         10,
        "placeholder_bars":        24,
        "runaway_dr_mult":         1.5,
    },
}


def get_thresholds(pair_name: str) -> dict:
    """Return thresholds for the given pair, falling back to DEFAULTS."""
    return PAIR_THRESHOLDS.get(pair_name, DEFAULTS)
