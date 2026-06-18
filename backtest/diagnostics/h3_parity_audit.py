"""Harness 3 â Backtest parity / look-ahead audit.

Proves (or refutes) on REAL data that the backtest does not see the future and
does not reproduce the historical "event-is-the-trade" bug. Trusts no comment;
every guard is checked empirically.

Three pillars (FABLE_REFERENCE Â§H3-1):
  Pillar 1  Causality ledger    â for every filled trade, the ordering chain
                                  ob <= bos < alert < fill must hold (alert>bos
                                  STRICT; equality is the historical bug).
  Pillar 2  Truncation oracle    â delete every bar at/after an alert; the alert
                                  must still fire identically. Catches any
                                  hidden use of future bars without knowing where
                                  it hides.
  Pillar 3  Read-only verifier   â re-check trade rows against raw bars
                                  (h3_verifier.py).
Plus Tier-A proximity A/B (the one confirmed config divergence) and a
"test-the-tester" pre-flight that must catch a planted event-is-the-trade bug
before the audit is allowed to run.

Outputs (under --out, default backtest/diagnostics/out):
  h3_report.md      ranked human report
  h3_findings.csv   machine-readable findings (one row per check, clean or not)

Run:
  python -m backtest.diagnostics.h3_parity_audit --pairs EURUSD --start 2026-03-01 --end 2026-03-31
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtest.diagnostics import driver
from backtest.diagnostics import h3_verifier

CATEGORY_ORDER = ["LOOKAHEAD", "BEHAVIOUR_DIVERGENCE", "CONFIG_DIVERGENCE",
                  "PESSIMISTIC_BIAS", "HARNESS_TRAP", "DATA_LIMIT", "INFO"]

# Backtest proximity cap. RESOLVED 2026-06-12: run_backtest no longer overrides
# the config value (it reads pair_conf["atr_multiplier"] untouched), so the
# backtest cap now EQUALS the live config cap. This map is therefore empty; the
# C19 finding falls back to live_cap per pair-type and should report zero
# divergence. The old override was {forex 3.0, index 3.5, commodity 3.5}; kept
# in this comment only as the historical record of what was removed.
BACKTEST_ATR_MULT: dict = {}


@dataclass
class Finding:
    finding_id: str
    category: str
    severity: str            # high | med | low | info
    guard_ref: str
    file_line: str
    title: str
    verdict: str             # CONFIRMED | REFUTED | NOT_TESTABLE(reason)
    n_occurrences: int = 0
    impact_metric: str = ""
    impact_value_r: float = 0.0
    demonstration: str = ""
    proposed_fix_option: str = ""
    status: str = "NOT_APPLIED"


# ---------------------------------------------------------------------------
# Pillar 1 â causality ledger
# ---------------------------------------------------------------------------
def _ts(v):
    if v is None or v == "":
        return None
    t = pd.Timestamp(v)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t


def causality_ledger(alerts: List[Dict[str, Any]],
                     trade_rows: List[Dict[str, Any]]) -> List[Finding]:
    """C03/C15: ob <= bos < alert < fill; alert > bos STRICT."""
    findings = []
    # Alert-level: alert_ts strictly after bos_ts.
    n_alert = 0
    eq_or_before = []
    for a in alerts:
        ob = a.get("ob") or {}
        alert_ts = _ts(a.get("ts"))
        bos_ts = _ts(ob.get("bos_timestamp"))
        ob_ts = _ts(ob.get("ob_timestamp"))
        if alert_ts is None:
            continue
        n_alert += 1
        if bos_ts is not None and not (alert_ts > bos_ts):
            eq_or_before.append((a.get("pair"), str(alert_ts), str(bos_ts)))
        if ob_ts is not None and bos_ts is not None and ob_ts > bos_ts:
            eq_or_before.append(("OB_AFTER_BOS", str(ob_ts), str(bos_ts)))

    findings.append(Finding(
        finding_id="C03", category="LOOKAHEAD" if eq_or_before else "INFO",
        severity="high" if eq_or_before else "info",
        guard_ref="guard 3 (alert strictly after BOS)",
        file_line="replay_engine.py:386-400",
        title="Alert fires strictly after the BOS candle",
        verdict="CONFIRMED" if eq_or_before else "REFUTED",
        n_occurrences=len(eq_or_before),
        impact_metric="alerts with alert_ts<=bos_ts",
        demonstration=(f"{len(eq_or_before)} alert(s) violated strict ordering; "
                       f"first: {eq_or_before[0]}" if eq_or_before
                       else f"held on N={n_alert} alerts: every alert_ts > bos_ts"),
        proposed_fix_option=("Investigate the upstream OB that carried a "
                             "future/equal bos_timestamp" if eq_or_before else ""),
    ))

    # Trade-level: ob <= bos < alert < fill_bar_open, fill >= alert+1h.
    n_filled = 0
    violations = []
    for r in trade_rows:
        if r.get("exit_reason") == "never_filled":
            continue
        n_filled += 1
        alert_ts = _ts(r.get("alert_ts"))
        bos_ts = _ts(r.get("bos_timestamp"))
        fill_ts = _ts(r.get("fill_ts"))
        ob_ts = _ts(r.get("ob_timestamp"))
        if bos_ts is not None and alert_ts is not None and not (alert_ts > bos_ts):
            violations.append(("alert<=bos", r.get("pair"), str(alert_ts), str(bos_ts)))
        if fill_ts is not None and alert_ts is not None and \
                fill_ts < alert_ts + pd.Timedelta(hours=1):
            violations.append(("fill<alert+1h", r.get("pair"), str(fill_ts), str(alert_ts)))
        if ob_ts is not None and bos_ts is not None and ob_ts > bos_ts:
            violations.append(("ob>bos", r.get("pair"), str(ob_ts), str(bos_ts)))

    findings.append(Finding(
        finding_id="C15", category="LOOKAHEAD" if violations else "INFO",
        severity="high" if violations else "info",
        guard_ref="guards 3+7 (causality chain on filled trades)",
        file_line="h1_only_simulator.py:357-377",
        title="Filled trades obey ob<=bos<alert<fill (the anti event-is-the-trade chain)",
        verdict="CONFIRMED" if violations else "REFUTED",
        n_occurrences=len(violations),
        impact_metric="filled trades violating causality",
        demonstration=(f"{len(violations)} violation(s); first: {violations[0]}"
                       if violations
                       else f"held on N={n_filled} filled trades"),
    ))
    return findings


# ---------------------------------------------------------------------------
# Pillar 2 â truncation oracle
# ---------------------------------------------------------------------------
def _alert_fingerprint(a: Dict[str, Any]) -> tuple:
    ob = a.get("ob") or {}
    return (ob.get("ob_timestamp"), ob.get("direction"),
            round(float(ob.get("proximal_line") or 0), 8),
            round(float(ob.get("distal_line") or 0), 8))


def truncation_oracle(pair_conf, df, alerts, *, sample_n: int = 30,
                      seed: int = 0) -> Finding:
    """C13: for sampled alerts, deleting all bars >= alert_ts must not change
    the alert produced at alert_ts. Each test is a single-bar fresh-state walk
    (cheap), comparing full-df vs truncated-df detection at the same T."""
    import random
    rng = random.Random(seed)
    uniq = {}
    for a in alerts:
        t = _ts(a.get("ts"))
        if t is not None:
            uniq.setdefault(t, a)
    sample_ts = sorted(uniq.keys())
    if len(sample_ts) > sample_n:
        sample_ts = sorted(rng.sample(sample_ts, sample_n))

    tested = 0
    mismatches = []
    for T in sample_ts:
        full = driver.walk_alerts(pair_conf, df, T, T)
        df_trunc = df[df.index <= T]
        trunc = driver.walk_alerts(pair_conf, df_trunc, T, T)
        full_fp = {_alert_fingerprint(x) for x in full.alerts if _ts(x.get("ts")) == T}
        trunc_fp = {_alert_fingerprint(x) for x in trunc.alerts if _ts(x.get("ts")) == T}
        tested += 1
        if full_fp != trunc_fp:
            mismatches.append((str(T), sorted(full_fp), sorted(trunc_fp)))

    n = tested
    return Finding(
        finding_id="C13", category="LOOKAHEAD" if mismatches else "INFO",
        severity="high" if mismatches else "info",
        guard_ref="full-series indicator / future-bar leakage",
        file_line="(whole detection path)",
        title="Deleting bars after an alert does not change the alert (truncation invariance)",
        verdict="CONFIRMED" if mismatches else ("REFUTED" if n else "NOT_TESTABLE(no alerts)"),
        n_occurrences=len(mismatches),
        impact_metric="sampled alerts that changed when the future was deleted",
        demonstration=(f"{len(mismatches)} of {n} sampled alerts changed; first: {mismatches[0]}"
                       if mismatches else
                       (f"held on N={n} sampled alerts: identical with future deleted"
                        if n else "no alerts to sample")),
    )


# ---------------------------------------------------------------------------
# Pillar 3 â verifier wrapper
# ---------------------------------------------------------------------------
def verifier_findings(df, trade_rows, risk_usd) -> List[Finding]:
    vr = h3_verifier.verify_trade_rows(df, trade_rows, risk_usd=risk_usd)
    findings = []
    by_check: Dict[str, int] = {}
    for v in vr.violations:
        by_check[v["check"]] = by_check.get(v["check"], 0) + 1

    findings.append(Finding(
        finding_id="C07_08", category="LOOKAHEAD" if (
            by_check.get("guard7_fill_before_alert_plus_1h") or
            by_check.get("guard8_not_first_fill") or
            by_check.get("guard8_no_qualifying_bar") or
            by_check.get("never_filled_but_touchable")) else "INFO",
        severity="high",
        guard_ref="guards 7+8 (fill walk start + first-fill rule)",
        file_line="h1_only_simulator.py:376-429",
        title="Fills occur on the first qualifying bar at/after alert_ts+1h",
        verdict="CONFIRMED" if (by_check.get("guard7_fill_before_alert_plus_1h") or
                                by_check.get("guard8_not_first_fill") or
                                by_check.get("guard8_no_qualifying_bar") or
                                by_check.get("never_filled_but_touchable")) else "REFUTED",
        n_occurrences=(by_check.get("guard7_fill_before_alert_plus_1h", 0) +
                       by_check.get("guard8_not_first_fill", 0) +
                       by_check.get("guard8_no_qualifying_bar", 0) +
                       by_check.get("never_filled_but_touchable", 0)),
        impact_metric="fill-rule violations",
        demonstration=(f"verifier flagged {sum(by_check.values())} issue(s): {by_check}"
                       if by_check else
                       f"held on N={vr.n_filled} filled + {vr.n_never_filled} never-filled rows"),
    ))

    findings.append(Finding(
        finding_id="RECON", category="LOOKAHEAD" if vr.recon_violations else "INFO",
        severity="high",
        guard_ref="P&L source of truth (pnl_usd == r_realised*risk)",
        file_line="h1_only_reporting.py:2593",
        title="Every trade's pnl_usd reconciles to r_realised",
        verdict="CONFIRMED" if vr.recon_violations else "REFUTED",
        n_occurrences=vr.recon_violations,
        impact_metric="reconciliation violations",
        demonstration=(f"{vr.recon_violations} rows fail pnl==r*risk"
                       if vr.recon_violations else
                       f"held on N={vr.n_rows} rows"),
    ))

    # Pessimism statistics (documented design, not bugs).
    findings.append(Finding(
        finding_id="C09_10", category="PESSIMISTIC_BIAS", severity="low",
        guard_ref="guards 9+10 (fill-bar TP suppressed; same-bar SL wins)",
        file_line="h1_only_simulator.py:451-477",
        title="Fill-bar TP suppression and same-bar SL-first cost (pessimism, not look-ahead)",
        verdict="REFUTED",  # not a leak; quantified
        n_occurrences=vr.fill_bar_tp_suppressed,
        impact_metric="fill bars that also reached TP (TP not credited) / same-bar SL+TP",
        demonstration=(f"{vr.fill_bar_tp_suppressed} fill bars also reached TP1 "
                       f"(credit suppressed); {vr.same_bar_sl_tp_collision} same-bar "
                       f"SL+TP collisions resolved SL-first. Both are intentional "
                       f"pessimism; listed so the cost is visible."),
    ))
    return findings, vr


# ---------------------------------------------------------------------------
# Tier A â proximity A/B (C19)
# ---------------------------------------------------------------------------
def tier_a_proximity(pair_conf, df, start, end, bt_result) -> Finding:
    """Reuses the already-computed backtest-cap walk (bt_result) and runs only
    ONE additional live-cap walk, then diffs. Two walks total per pair, not
    three."""
    pt = pair_conf.get("pair_type", "forex")
    live_cap = float(pair_conf["atr_multiplier"])          # config.json value
    bt_cap = BACKTEST_ATR_MULT.get(pt, live_cap)
    live_res = driver.walk_alerts(
        pair_conf, df, start, end,
        overrides=driver.KnobOverrides(proximity_cap={pt: live_cap}))
    live_fp = {_alert_fingerprint(a) for a in live_res.alerts}
    bt_fp = {_alert_fingerprint(a) for a in bt_result.alerts}
    bt_only = bt_fp - live_fp
    live_only = live_fp - bt_fp
    superset = live_fp.issubset(bt_fp)
    return Finding(
        finding_id="C19", category="CONFIG_DIVERGENCE", severity="med",
        guard_ref="proximity cap: live config vs BACKTEST_ATR_MULT",
        file_line="run_backtest.py:114 / config.json",
        title="Backtest proximity cap is wider than live (fires more/earlier alerts)",
        verdict="CONFIRMED",
        n_occurrences=len(bt_only),
        impact_metric=f"alerts: live({live_cap})={len(live_fp)} vs backtest({bt_cap})={len(bt_fp)}",
        demonstration=(f"backtest cap {bt_cap} produced {len(bt_fp)} alerts vs "
                       f"{len(live_fp)} at live cap {live_cap}; "
                       f"{len(bt_only)} backtest-only, {len(live_only)} live-only; "
                       f"live âŠ† backtest = {superset}."),
        proposed_fix_option=("Decide which cap is canonical. If live is truth, "
                             "set BACKTEST_ATR_MULT = config values "
                             "(run_backtest.py:114). NOT applied."),
    )


# ---------------------------------------------------------------------------
# C22 â radar return shape
# ---------------------------------------------------------------------------
def radar_shape(pair_conf, df, start, end, *, max_snaps: int = 10) -> Finding:
    types = set()
    n = 0
    nbars = len(df.loc[start:end])
    stride = max(24, nbars // max(1, max_snaps))
    for snap in driver.walk_detection(pair_conf, df, start, end, stride=stride):
        types.add(snap.raw_radar_return_type)
        n += 1
    bad = types - {"dict_active_zones"}
    return Finding(
        finding_id="C22", category="INFO" if not bad else "BEHAVIOUR_DIVERGENCE",
        severity="info" if not bad else "med",
        guard_ref="detect_smc_radar return shape",
        file_line="replay_engine.py:463-501 (_normalize_obs_result)",
        title="Live radar always returns the dict/active_zones shape",
        verdict="REFUTED" if not bad else "CONFIRMED",
        n_occurrences=len(bad),
        impact_metric="non-canonical radar return types",
        demonstration=(f"held on N={n} snapshots: only 'dict_active_zones'"
                       if not bad else f"non-canonical shapes seen: {bad}"),
    )


# ---------------------------------------------------------------------------
# Pre-flight â test the tester (must catch a planted bug)
# ---------------------------------------------------------------------------
def preflight() -> bool:
    # Planted event-is-the-trade: alert_ts == bos_ts, fill on alert bar.
    bad_alert = {"pair": "TEST", "ts": "2026-01-05T10:00:00+00:00",
                 "ob": {"ob_timestamp": "2026-01-05T08:00:00+00:00",
                        "bos_timestamp": "2026-01-05T10:00:00+00:00",  # == alert
                        "direction": "bullish",
                        "proximal_line": 1.0, "distal_line": 0.99}}
    bad_row = {"pair": "TEST", "alert_ts": "2026-01-05T10:00:00+00:00",
               "bos_timestamp": "2026-01-05T10:00:00+00:00",
               "ob_timestamp": "2026-01-05T08:00:00+00:00",
               "fill_ts": "2026-01-05T10:00:00+00:00",  # same bar as alert
               "exit_reason": "tp2", "bias": "LONG",
               "entry": 1.0, "sl_initial": 0.99, "tp1": 1.02, "tp2": 1.04,
               "exit_ts": "2026-01-05T12:00:00+00:00", "exit_price": 1.04,
               "r_realised": 4.0, "pnl_usd": 1000.0}
    cf = causality_ledger([bad_alert], [bad_row])
    if not any(f.verdict == "CONFIRMED" and f.category == "LOOKAHEAD" for f in cf):
        raise AssertionError("PRE-FLIGHT FAIL: ledger did not catch event-is-the-trade")
    # Verifier must flag fill before alert+1h.
    df = driver._synthetic_swingy_df(60)
    vr = h3_verifier.verify_trade_rows(df, [bad_row])
    if not any(v["check"].startswith("guard7") for v in vr.violations):
        raise AssertionError("PRE-FLIGHT FAIL: verifier did not catch same-bar fill")
    print("[h3.preflight] planted event-is-the-trade caught by ledger + verifier OK")
    return True


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def _rank(findings: List[Finding]) -> List[Finding]:
    def key(f):
        cat = CATEGORY_ORDER.index(f.category) if f.category in CATEGORY_ORDER else 99
        return (cat, -abs(f.impact_value_r), -f.n_occurrences)
    return sorted(findings, key=key)


def write_report(findings: List[Finding], meta: Dict[str, Any], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = _rank(findings)
    # CSV
    csv_path = out_dir / "h3_findings.csv"
    cols = ["rank"] + list(asdict(ranked[0]).keys()) if ranked else ["rank"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, fd in enumerate(ranked, 1):
            row = {"rank": i, **asdict(fd)}
            w.writerow(row)
    # Markdown
    md_path = out_dir / "h3_report.md"
    lines = ["# Harness 3 â Parity / Look-Ahead Audit", "",
             f"- window: **{meta['start']} â {meta['end']}** (post-clamp: {meta.get('data_range','?')})",
             f"- pairs: {meta['pairs']}",
             f"- slice_mode: **B** (Tier-A and verifier are bar-by-bar)",
             f"- generated: {meta['generated']}", "",
             "Every guard gets a row even when clean â absence of a row is never "
             "evidence. `status` is `NOT_APPLIED` on every row (nothing was changed).",
             ""]
    lookahead = [f for f in ranked if f.category == "LOOKAHEAD" and f.verdict == "CONFIRMED"]
    lines.append(f"## Verdict: {'âš ï¸ LOOKAHEAD FOUND' if lookahead else 'âœ… no look-ahead confirmed'}")
    lines.append("")
    lines.append("| rank | id | category | verdict | N | title | demonstration |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, fd in enumerate(ranked, 1):
        lines.append(f"| {i} | {fd.finding_id} | {fd.category} | {fd.verdict} | "
                     f"{fd.n_occurrences} | {fd.title} | {fd.demonstration} |")
    lines.append("")
    lines.append("## Proposed fixes (NOT applied)")
    for fd in ranked:
        if fd.proposed_fix_option:
            lines.append(f"- **{fd.finding_id}** ({fd.file_line}): {fd.proposed_fix_option}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, csv_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_audit(pairs: List[str], start, end, out_dir: Path, *,
              risk_usd: float = 250.0, sample_n: int = 30, fast: bool = False):
    preflight()
    if not driver.self_check(verbose=False):
        raise RuntimeError("driver self-check failed")
    print("[h3] driver self-check OK", flush=True)

    cfg = json.load(open(_REPO_ROOT / "config.json"))
    confs = [p for p in cfg["pairs"] if (pairs == ["all"] or p["name"] in pairs)]
    start = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
    end = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)

    all_findings: List[Finding] = []
    data_ranges = []
    for pc in confs:
        name = pc["name"]
        print(f"\n=== {name} ===")
        df = driver.load_window(pc, start, end)
        if isinstance(df, driver.WindowUnserveable):
            all_findings.append(Finding(
                finding_id=f"DATA_{name}", category="DATA_LIMIT", severity="med",
                guard_ref="data_loader window coverage", file_line="data_loader.py",
                title=f"{name}: window unserveable",
                verdict=f"NOT_TESTABLE({df.reason})",
                demonstration=df.detail))
            continue
        data_ranges.append(f"{name}:{df.index.min().date()}..{df.index.max().date()}")

        # Main walk at the BACKTEST cap (what the real backtest actually runs).
        pt = pc.get("pair_type", "forex")
        bt_cap = BACKTEST_ATR_MULT.get(pt, float(pc["atr_multiplier"]))
        print(f"  [walk 1/{'1' if fast else '2'}] main walk @ backtest cap {bt_cap}...", flush=True)
        res = driver.walk_alerts(
            pc, df, start, end, risk_usd=risk_usd,
            overrides=driver.KnobOverrides(proximity_cap={pt: bt_cap}))
        print(f"  alerts_total={res.counters['alerts_total']} "
              f"simulated={res.counters['alerts_simulated']} "
              f"trades={len(res.trade_rows)}", flush=True)

        all_findings += causality_ledger(res.alerts, res.trade_rows)
        vfind, _vr = verifier_findings(df, res.trade_rows, risk_usd)
        all_findings += vfind
        print("  [truncation oracle]...", flush=True)
        all_findings.append(truncation_oracle(pc, df, res.alerts, sample_n=sample_n))
        all_findings.append(radar_shape(pc, df, start, end))
        if not fast:
            print(f"  [walk 2/2] Tier-A live-cap walk...", flush=True)
            all_findings.append(tier_a_proximity(pc, df, start, end, res))
        else:
            all_findings.append(Finding(
                finding_id="C19", category="CONFIG_DIVERGENCE", severity="med",
                guard_ref="proximity cap", file_line="run_backtest.py:114",
                title="Tier-A proximity A/B", verdict="NOT_TESTABLE(--fast skipped)",
                demonstration="re-run without --fast to quantify the cap divergence"))
        # tag findings with pair via title prefix for multi-pair runs
        for f in all_findings[-7:]:
            if not f.title.startswith(name):
                f.title = f"{name}: {f.title}"

    meta = {"start": str(start.date()), "end": str(end.date()),
            "pairs": ",".join(p["name"] for p in confs),
            "data_range": "; ".join(data_ranges),
            "generated": pd.Timestamp.utcnow().isoformat()}
    md, csvp = write_report(all_findings, meta, out_dir)
    print(f"\n[h3] report -> {md}\n[h3] findings -> {csvp}")
    return md, csvp


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="EURUSD")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--risk-usd", type=float, default=250.0)
    ap.add_argument("--max-truncation-samples", type=int, default=30)
    ap.add_argument("--fast", action="store_true",
                    help="skip the Tier-A second walk (smoke tests)")
    args = ap.parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    run_audit(pairs, args.start, args.end, Path(args.out),
              risk_usd=args.risk_usd, sample_n=args.max_truncation_samples,
              fast=args.fast)


if __name__ == "__main__":
    main()
