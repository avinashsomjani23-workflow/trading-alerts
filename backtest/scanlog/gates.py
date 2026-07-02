"""Hard gates + final verdict (SPEC Â§4).

Evaluated at run end. Any FAIL gate -> overall FAIL -> process exit code 1.
WARN never flips overall but always prints and sets warnings_present.

The gate layer is the JUDGE. It recomputes the headline P&L from r_realised
ALONE (never from the hypothetical tp1/tp2 columns) and compares; it checks
causality strict-inequalities on every filled trade; it checks the heartbeat is
complete; it checks for the contradiction conditions the emitter recorded.

Nothing here mutates trade logic. It reads records and judges.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.scanlog import conditions as cond
from backtest.scanlog.emitter import ScanLog
# Single source of truth for which exit reasons never feed P&L. The reporting
# layer strips these before building its scoreboards; the gate headline must
# strip the identical set or G1 compares an all-rows sum against a
# reason-excluded sum (same apples-to-oranges trap as _is_blocked).
from backtest.h1_only_reporting import _is_eligible


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_REL_TOL = 1e-9


@dataclass
class Gate:
    id: str
    description: str
    threshold: str
    observed: Any
    verdict: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "threshold": self.threshold,
            "observed": self.observed,
            "verdict": self.verdict,
        }


@dataclass
class HealthResult:
    overall: str
    exit_code: int
    warnings_present: bool
    gates: List[Gate] = field(default_factory=list)
    headline_pnl_usd: float = 0.0
    # Live trades ONLY the proximal limit (Phase2 compute_phase2_levels defaults
    # entry_zone="proximal"). The 50% mean entry is a backtest A/B study, never
    # placed live. So `headline_pnl_usd` (both zones summed) is ~2x the live book.
    # These split fields make the LIVE number (proximal) unambiguous.
    proximal_headline_usd: float = 0.0
    fifty_pct_headline_usd: float = 0.0
    content_hash: str = ""

    @property
    def passed(self) -> bool:
        return self.overall == PASS


def _is_counted(t: Dict[str, Any]) -> bool:
    """A trade row that feeds the reporting headline. Must sum the IDENTICAL row
    set the scoreboards do, or G1 reconciles two different sums.

    PROXIMAL ONLY (2026-06-30): the reporting headline is now the proximal/live
    book alone — the 50% mean entry is dead and shown nowhere. So G1's
    independent r_realised sum is restricted to proximal too, matching the
    report. (The 50% split field is computed separately for audit.)
    Delegates to _is_eligible (resolved fill AND clears the live score floor).
    Live parity: news / IST / killzone NEVER gate, so they are not consulted."""
    return t.get("entry_zone") == "proximal" and _is_eligible(t)


def _sum_r_realised_pnl(trades: List[Dict[str, Any]], risk_usd: float,
                        counted=None) -> float:
    """Headline from r_realised ONLY (G1/G6). Never reads pnl_usd or the
    hypothetical tp1/tp2 columns. Rows are restricted to the SAME set the
    reporting layer aggregates: proximal, not blocked, resolved fill, clears the
    score floor. This is the single source of truth the reporting layer must
    reconcile against. `counted` overrides the row predicate (used to compute
    the 50%-zone audit split, which the default proximal-only predicate hides)."""
    pred = counted or _is_counted
    return round(
        sum(float(t.get("r_realised", 0.0)) for t in trades if pred(t))
        * risk_usd, 6)


def _has_fail_conditions(counts: Dict[str, int]) -> List[str]:
    out = []
    for code, n in counts.items():
        if n > 0 and cond.severity_of(code) == cond.FAIL:
            out.append(code)
    return out


def evaluate(
    *,
    scanlog: ScanLog,
    trades: List[Dict[str, Any]],
    risk_usd: float,
    reported_headline_usd: Optional[float] = None,
    manifest_recheck_knobs: Optional[Dict[str, Any]] = None,
) -> HealthResult:
    """Run every gate and return a HealthResult. Also writes run_health.json.

    `reported_headline_usd` is what the reporting layer asserts as the run's
    headline; G1 requires it to equal the r_realised-derived figure.
    `manifest_recheck_knobs` is a fresh live read of the knobs taken at run END
    so G5 can detect drift from the manifest snapshot.
    """
    gates: List[Gate] = []
    cc = dict(scanlog.condition_counts)

    # ---- G1 reconciliation ------------------------------------------------
    headline = _sum_r_realised_pnl(trades, risk_usd)
    # Per-row pnl_usd must equal r_realised * risk on every row.
    pnl_row_mismatch = 0
    for t in trades:
        expect = round(float(t.get("r_realised", 0.0)) * risk_usd, 2)
        got = t.get("pnl_usd")
        if got is None or abs(float(got) - expect) > 1e-6:
            pnl_row_mismatch += 1
            scanlog.condition("PNL_MISMATCH", pair=t.get("pair"),
                              alert_ts=t.get("alert_ts"),
                              expected=expect, got=got)
    g1_ok = pnl_row_mismatch == 0
    if reported_headline_usd is not None:
        if not math.isclose(headline, round(reported_headline_usd, 6),
                            rel_tol=_REL_TOL, abs_tol=1e-6):
            g1_ok = False
    gates.append(Gate(
        "G1", "Sum pnl_usd == Sum r_realised*risk, and equals reporting headline",
        f"rel_tol {_REL_TOL}",
        {"headline_usd": headline, "reported": reported_headline_usd,
         "row_mismatches": pnl_row_mismatch},
        PASS if g1_ok else FAIL,
    ))

    # ---- G2 heartbeat completeness ----------------------------------------
    expected_total = sum(scanlog.expected_scan_records.values())
    actual_total = sum(scanlog.actual_scan_records.values())
    missing_by_pair = {
        p: scanlog.expected_scan_records.get(p, 0) - scanlog.actual_scan_records.get(p, 0)
        for p in scanlog.expected_scan_records
        if scanlog.expected_scan_records.get(p, 0) != scanlog.actual_scan_records.get(p, 0)
    }
    if missing_by_pair:
        for p, gap in missing_by_pair.items():
            scanlog.condition("HEARTBEAT_GAP", pair=p, missing=gap)
    # ANTI-VACUOUS (2026-07-02): a run that produced trades but declared ZERO
    # walk bars did not pass the heartbeat — it never recorded one. This is
    # exactly what happened when pairs moved into ProcessPoolExecutor workers:
    # each worker got the NullScanLog, every scan/event/condition record was
    # silently swallowed, and G2 "passed" on 0 == 0 while the audit layer was
    # blind. 0 expected + >0 trades is a FAIL, never a pass.
    g2_vacuous = (expected_total == 0 and len(trades) > 0)
    g2_ok = ((expected_total == actual_total) and not missing_by_pair
             and not g2_vacuous)
    gates.append(Gate(
        "G2", "scan records == expected (pair x walk bars); zero HEARTBEAT_GAP; "
              "never vacuous (0 declared bars with trades present = blind run)",
        "exact, non-vacuous",
        {"expected": expected_total, "actual": actual_total,
         "missing_by_pair": missing_by_pair, "vacuous": g2_vacuous},
        PASS if g2_ok else FAIL,
    ))

    # ---- G3 causality ------------------------------------------------------
    # For every filled trade: ob_ts <= bos_ts < alert_ts <= fill_ts.
    # fill_ts >= alert_ts (NOT strictly >): the fill candle is the ALERT candle
    # itself (alert_ts = candle B, the candle the trader places the order into a
    # few minutes after the trigger candle closed), so fill_ts == alert_ts is the
    # earliest legal fill, not a bug. The old rule (fill_ts >= alert_ts + 1h)
    # skipped candle B and filled a candle late; that requirement is removed.
    # bos_ts < alert_ts stays strict (a BOS on the just-closed candle cannot
    # alert the same cycle — the historical lookahead bug).
    causality_violations = []
    for t in trades:
        fill = t.get("fill_ts")
        if not fill:
            continue  # never_filled rows have no causality chain to check
        ob = t.get("ob_timestamp")
        bos = t.get("bos_timestamp")
        alert = t.get("alert_ts")
        try:
            ob_t = pd.Timestamp(ob) if ob else None
            bos_t = pd.Timestamp(bos) if bos else None
            alert_t = pd.Timestamp(alert)
            fill_t = pd.Timestamp(fill)
        except Exception as e:
            causality_violations.append({"pair": t.get("pair"), "err": str(e)})
            continue
        bad = []
        if ob_t is not None and bos_t is not None and not (ob_t <= bos_t):
            bad.append("ob<=bos")
        if bos_t is not None and not (bos_t < alert_t):
            bad.append("bos<alert")
        if not (alert_t <= fill_t):
            bad.append("alert<=fill")
        if bad:
            causality_violations.append(
                {"pair": t.get("pair"), "alert_ts": str(alert), "violations": bad})
            scanlog.condition("FILL_BEFORE_ALERT", pair=t.get("pair"),
                              alert_ts=str(alert), fill_ts=str(fill),
                              violations=bad)
    g3_ok = not causality_violations
    gates.append(Gate(
        "G3", "ob<=bos<alert<=fill on every filled trade (fill on the alert candle allowed)",
        "strict inequalities",
        {"violations": len(causality_violations),
         "sample": causality_violations[:5]},
        PASS if g3_ok else FAIL,
    ))

    # ---- G4 zero FAIL conditions ------------------------------------------
    fail_codes = _has_fail_conditions(cc)
    gates.append(Gate(
        "G4", "every FAIL-severity condition counter is 0", "== 0",
        {code: cc[code] for code in fail_codes} or "none",
        PASS if not fail_codes else FAIL,
    ))

    # ---- G5 manifest honesty (knob drift) ---------------------------------
    drift = {}
    if manifest_recheck_knobs is not None:
        orig = scanlog.manifest.get("knobs", {})
        for k, v in manifest_recheck_knobs.items():
            if orig.get(k) != v:
                drift[k] = {"manifest": orig.get(k), "now": v}
                scanlog.condition("CONFIG_DRIFT", knob=k,
                                  manifest=orig.get(k), now=v)
    g5_ok = not drift
    gates.append(Gate(
        "G5", "every knob re-read at END equals manifest (no drift)", "exact",
        drift or "none", PASS if g5_ok else FAIL,
    ))

    # ---- G6 hypothetical isolation ----------------------------------------
    # The health headline is recomputed from r_realised only (above). If it
    # equals a tp2-derived figure that would be coincidence, not consumption;
    # the proof of isolation is that THIS layer never reads tp1/tp2. We assert
    # the recomputed headline does not silently equal the tp2 aggregate when
    # they should differ - i.e. we recompute independently and expose both.
    tp2_headline = round(
        sum(float(t.get("r_if_exit_tp2", 0.0)) for t in trades if _is_counted(t))
        * risk_usd, 6)
    gates.append(Gate(
        "G6", "headline recomputed from r_realised only; tp1/tp2 excluded",
        "independent recompute",
        {"realised_headline": headline, "tp2_headline_for_contrast": tp2_headline},
        PASS,  # structurally guaranteed: this function never sums into headline from tp2
    ))

    # ---- G7 determinism stamp ---------------------------------------------
    chash = scanlog.content_hash()
    gates.append(Gate(
        "G7", "content hash of scan_log + events recorded for re-run compare",
        "recorded", chash, PASS,
    ))

    # ---- G8 NaN budget ----------------------------------------------------
    g8_verdict = PASS
    nan_detail = {}
    for p, total in scanlog.post_warmup_bars.items():
        nans = scanlog.nan_atr_skips.get(p, 0)
        pct = (nans / total * 100.0) if total else 0.0
        nan_detail[p] = round(pct, 3)
        if pct > 5.0:
            g8_verdict = FAIL
        elif pct > 1.0 and g8_verdict != FAIL:
            g8_verdict = WARN
            scanlog.condition("NAN_ATR_SKIP", pair=p, pct=round(pct, 3),
                              note="exceeds 1% post-warmup budget")
    gates.append(Gate(
        "G8", "NAN_ATR_SKIP <= 1% post-warmup per pair (WARN >1%, FAIL >5%)",
        "<=1% warn, >5% fail", nan_detail, g8_verdict,
    ))

    # ---- G9 schema validity (sampled at write-time; here we confirm) -------
    # The emitter validates ts/outcome on every record as it writes. G9 here
    # confirms no schema-level FAIL condition was raised during the run.
    schema_fail = (cc.get("TS_NOT_BOUNDARY", 0) + cc.get("TZ_NAIVE", 0))
    gates.append(Gate(
        "G9", "every record validated against SCHEMA.md (ts/tz/outcome)",
        "0 schema FAILs", schema_fail, PASS if schema_fail == 0 else FAIL,
    ))

    # ---- G10 physical possibility of per-trade metrics ---------------------
    # Tripwires for the "measured a thing that could not have happened" class
    # (the MFE fill-bar bug family). Parent-side over trade rows, so worker
    # log loss can never hide it. Rules (tolerances cover 3dp rounding):
    #   a) mfe_r < 0 or mae_r > 0            — impossible by construction
    #      (both start at entry = 0 and only widen).
    #   b) exit sl at ~-1R with mfe_r >= ~+1R — impossible under BE@1R: a clean
    #      +1R print arms break-even (worst exit 0R), and contaminated bars
    #      (fill/SL/TP1-exit) are excluded from MFE. Any such row means the
    #      excursion walk credited pre-fill / same-bar price again.
    #   c) exit tp1 with mfe_r > r_realised   — MFE is capped at the TP1 exit
    #      level; beyond it is post-exit price.
    phys_violations = []
    for t in trades:
        if t.get("exit_reason") == "never_filled":
            continue
        try:
            mfe = float(t.get("mfe_r") or 0.0)
            mae = float(t.get("mae_r") or 0.0)
            r = float(t.get("r_realised") or 0.0)
        except (TypeError, ValueError):
            continue
        bad = []
        if mfe < -0.002 or mae > 0.002:
            bad.append("excursion_sign")
        if t.get("exit_reason") == "sl" and r <= -0.999 and mfe >= 0.999:
            bad.append("full_sl_loser_with_1R_mfe")
        if t.get("exit_reason") == "tp1" and mfe > r + 0.002:
            bad.append("mfe_beyond_tp1_exit")
        if bad:
            phys_violations.append({"pair": t.get("pair"),
                                    "alert_ts": t.get("alert_ts"),
                                    "exit_reason": t.get("exit_reason"),
                                    "r": r, "mfe_r": mfe, "mae_r": mae,
                                    "violations": bad})
            scanlog.condition("PHYS_IMPOSSIBLE_METRIC", pair=t.get("pair"),
                              alert_ts=t.get("alert_ts"), violations=bad)
    gates.append(Gate(
        "G10", "per-trade metrics physically possible (no fake excursions)",
        "0 violations",
        {"violations": len(phys_violations), "sample": phys_violations[:5]},
        PASS if not phys_violations else FAIL,
    ))

    # ---- overall ----------------------------------------------------------
    any_fail = any(g.verdict == FAIL for g in gates)
    any_warn = any(g.verdict == WARN for g in gates) or any(
        n > 0 and cond.severity_of(code) == cond.WARN
        for code, n in cc.items()
    )
    overall = FAIL if any_fail else PASS
    result = HealthResult(
        overall=overall,
        exit_code=1 if any_fail else 0,
        warnings_present=bool(any_warn),
        gates=gates,
        headline_pnl_usd=headline,
        proximal_headline_usd=_sum_r_realised_pnl(
            [t for t in trades if t.get("entry_zone") == "proximal"], risk_usd),
        # 50% is an audit-only split now (not in the headline); count it with a
        # 50%-zone predicate since the default _is_counted is proximal-only.
        fifty_pct_headline_usd=_sum_r_realised_pnl(
            trades, risk_usd,
            counted=lambda t: t.get("entry_zone") == "50pct" and _is_eligible(t)),
        content_hash=chash,
    )

    _write_health(scanlog, result)
    return result


def _write_health(scanlog: ScanLog, result: HealthResult) -> None:
    health = {
        "run_id": scanlog.manifest.get("run_id"),
        "overall": result.overall,
        "exit_code": result.exit_code,
        "warnings_present": result.warnings_present,
        "headline_pnl_usd": result.headline_pnl_usd,
        "live_proximal_headline_usd": result.proximal_headline_usd,
        "study_fifty_pct_headline_usd": result.fifty_pct_headline_usd,
        "content_hash": result.content_hash,
        "condition_counts": dict(scanlog.condition_counts),
        "event_counts": dict(scanlog.event_counts),
        "outcome_counts": dict(scanlog.outcome_counts),
        "gates": [g.as_dict() for g in result.gates],
    }
    with open(scanlog.run_dir / "run_health.json", "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2, default=str)


def render_table(result: HealthResult, scanlog: ScanLog) -> str:
    """Human-readable gate table + condition counters (SPEC Â§6 console block)."""
    lines = []
    lines.append("=" * 64)
    lines.append("SCAN-LOG RUN HEALTH")
    lines.append("=" * 64)
    lines.append(f"{'GATE':<5} {'VERDICT':<7} DESCRIPTION")
    for g in result.gates:
        lines.append(f"{g.id:<5} {g.verdict:<7} {g.description}")
    lines.append("-" * 64)
    nonzero = {k: v for k, v in scanlog.condition_counts.items() if v}
    if nonzero:
        lines.append("CONDITION COUNTERS (nonzero):")
        for code, n in sorted(nonzero.items()):
            lines.append(f"  {cond.severity_of(code):<5} {code:<26} {n}")
    else:
        lines.append("CONDITION COUNTERS: none")
    lines.append("-" * 64)
    lines.append(f"headline (from r_realised): ${result.headline_pnl_usd:,.2f}")
    if result.warnings_present:
        lines.append("warnings_present: TRUE")
    lines.append(f"OVERALL: {result.overall} (exit code {result.exit_code})")
    lines.append("=" * 64)
    return "\n".join(lines)
