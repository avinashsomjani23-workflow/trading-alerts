"""EXIT-LEVER HONESTY GATE — report-layer check for exit-rule results.

Trader-approved 2026-07-09 (STEP_B_EXIT_TRACK_HANDOFF.md §7). Out-of-band ONLY:
this module is imported by diagnostic / research scripts that emit exit-lever
results. It is NEVER imported by live alert generation or the row build — a
guard must never be able to break the thing it protects.

Why it exists (all three burned us in the 2026-07-09 Step-B session):
  (a) a wider-stop "win" was reported without avg R:R — hiding that R:R had
      collapsed 2.08 -> 0.80;
  (b) a sweep-conditional widen was reported as tradeable although it
      conditions on an OUTCOME-time column (sl_bar_was_sweep) — you can't act
      on a column that only exists after the stop fired;
  (c) break-even scratches (r_realised == 0) were counted as losers, poisoning
      every stopout/sweep percentage.

Contract: every emitted exit-lever result is a dict. Call `check_exit_result`
per result (returns violations) or `assert_exit_report` over the batch (raises
on any violation). A result dict must carry:

  net_expR       : float — the headline number.
  avg_RR         : float — MUST accompany net_expR (rule a).
  conditions_on  : list[str] — every column the rule keys on to decide WHICH
                   trades it applies to ([] for a blanket rule). If any is
                   outcome-time (sl_*, exit_*, r_*, mfe/mae family), the result
                   MUST carry non_tradeable=True (rule b).
  loser_def      : str — MUST be exactly "r<0". Any definition that lets
                   r == 0 scratches into the loser denominator fails (rule c).
"""
from typing import Any, Dict, Iterable, List

# Outcome-time column families (SMC_EDGE_LAB_SPEC.md §12 timing classifier).
OUTCOME_TIME_PREFIXES = ("sl_", "exit_", "r_")
OUTCOME_TIME_EXACT = {"mfe_r", "mae_r", "bars_to_exit", "bars_sl_to_tp1_touch"}
# Entry-legal exceptions that carry a matching prefix but are known BEFORE the
# outcome (sl_distance_atr is fixed at fill; r_distance is the risk, not a result).
ENTRY_LEGAL_EXCEPTIONS = {"sl_distance_atr", "r_distance"}

LOSER_DEF_REQUIRED = "r<0"


def is_outcome_time(col: str) -> bool:
    if col in ENTRY_LEGAL_EXCEPTIONS:
        return False
    return col in OUTCOME_TIME_EXACT or col.startswith(OUTCOME_TIME_PREFIXES)


def check_exit_result(result: Dict[str, Any]) -> List[str]:
    """Return the list of honesty violations for ONE exit-lever result dict.
    Empty list = clean."""
    v: List[str] = []
    name = result.get("recipe", result.get("name", "<unnamed>"))

    # (a) no net expR without avg R:R beside it.
    if "net_expR" in result and result.get("avg_RR") is None:
        v.append(f"{name}: net_expR reported without avg_RR (rule a — an expR "
                 "'win' that collapses R:R must be visible)")

    # (b) outcome-conditioned rules must be flagged non-tradeable.
    conds = result.get("conditions_on")
    if conds is None:
        v.append(f"{name}: missing conditions_on ([] for a blanket rule) — "
                 "cannot verify the outcome-conditioning rule (b)")
    else:
        leaks = [c for c in conds if is_outcome_time(c)]
        if leaks and result.get("non_tradeable") is not True:
            v.append(f"{name}: conditions on outcome-time column(s) {leaks} "
                     "without non_tradeable=True (rule b)")

    # (c) the loser denominator is r<0 only — never r==0 scratches.
    if result.get("loser_def") != LOSER_DEF_REQUIRED:
        v.append(f"{name}: loser_def={result.get('loser_def')!r} — must be "
                 f"{LOSER_DEF_REQUIRED!r}; BE scratches (r==0) are never losers "
                 "(rule c)")
    return v


def assert_exit_report(results: Iterable[Dict[str, Any]]) -> None:
    """Raise ValueError listing every violation in a batch of exit results.
    Call this before an exit report is written/printed/emailed."""
    violations: List[str] = []
    for res in results:
        violations.extend(check_exit_result(res))
    if violations:
        raise ValueError(
            "EXIT-LEVER HONESTY GATE failed:\n  " + "\n  ".join(violations))
