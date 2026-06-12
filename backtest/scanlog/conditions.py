"""Condition taxonomy (SPEC Â§3). Every edge case gets a NAME and a SEVERITY.

Severity meaning:
    INFO  - expected, counted, never affects PASS/FAIL.
    WARN  - legal but degrades confidence; sets warnings_present, never FAILs.
    FAIL  - the run is wrong; the owning gate flips overall to FAIL (exit 1).

Rule (SPEC Â§3): encountering an UNREGISTERED anomaly raises
UNCLASSIFIED_CONDITION (FAIL). The system fails loud on the unknown rather
than absorbing it. Extend this registry whenever a new condition is
discovered; never handle one without registering it.
"""

from __future__ import annotations

INFO = "INFO"
WARN = "WARN"
FAIL = "FAIL"

# code -> (severity, human meaning). NAN_ATR_SKIP is INFO at the record level
# and promoted to WARN/FAIL by the G8 budget gate, not per-record.
CONDITIONS: dict[str, tuple[str, str]] = {
    "YF_CLAMP":                 (WARN, "served window < requested (record both)"),
    "WARMUP_SKIP":              (INFO, "slice < MIN_WARMUP_BARS"),
    "NAN_ATR_SKIP":             (INFO, "ATR unavailable for this bar"),
    "DEGENERATE_BAR":           (WARN, "high==low or non-finite OHLC encountered"),
    "DEGENERATE_OB":            (WARN, "proximal==distal zone seen"),
    "ZERO_R_TRADE":             (WARN, "entry==SL; excluded from rate math, listed"),
    "GAP_IN_INDEX":             (WARN, "non-calendar gap between consecutive H1 bars"),
    "ALERT_LOOKAHEAD_BLOCKED":  (FAIL, "guard #3 tripped - future-stamped OB produced an alert"),
    "ASSERT_NO_LOOKAHEAD_TRIP": (FAIL, "guard #1 tripped - slice contained a future bar"),
    "RING_OVERFLOW_RISK":       (WARN, ">=18 new structure events between observation points"),
    "DEDUP_SUPPRESSED":         (INFO, "alert suppressed by OB-dedup"),
    "NORMALIZE_NONDICT":        (WARN, "radar returned non-dict shape"),
    "CONFIG_DRIFT":             (FAIL, "knob value observed mid-run != manifest value"),
    "PNL_MISMATCH":             (FAIL, "pnl_usd != r_realised * risk_usd on a row"),
    "HEARTBEAT_GAP":            (FAIL, "a (pair, bar) in the walk produced no scan record"),
    "TS_NOT_BOUNDARY":          (FAIL, "a wall-clock ts is not an element of df.index"),
    "TZ_NAIVE":                 (FAIL, "a naive (tz-unaware) timestamp appeared"),
    # Contradictory-rationale family (user-requested: catch misbehaviour, not
    # just arithmetic). Each is a logical impossibility, not a weak signal.
    "TREND_CONTRADICTION":      (FAIL, "alert tagged with-trend while trend reading is opposite"),
    "ZONE_STATE_CONTRADICTION": (FAIL, "OB reported active and mitigated in the same bar"),
    "FILL_BEFORE_ALERT":        (FAIL, "trade fill ts <= its own alert ts"),
    "UNCLASSIFIED_CONDITION":   (FAIL, "an anomaly with no registered code"),
}

# Outcome enum for the scan_log heartbeat (SPEC Â§2.2). One per pair-bar.
OUTCOMES = {
    "NO_ZONE",         # no active zones this bar
    "OUT_OF_RANGE",    # zones present but none within prox cap
    "ALERT",           # at least one alert fired
    "RE_ARM_WAIT",     # nearest zone in cooling, waiting to re-arm
    "WARMUP_SKIP",     # slice < MIN_WARMUP
    "NAN_ATR_SKIP",    # ATR NaN/None
    "DEGENERATE_SKIP", # degenerate bar/detection error skipped the bar
}


def severity_of(code: str) -> str:
    """Severity for a code, or FAIL if unregistered (fail loud on the unknown)."""
    entry = CONDITIONS.get(code)
    return entry[0] if entry else FAIL


def is_registered(code: str) -> bool:
    return code in CONDITIONS
