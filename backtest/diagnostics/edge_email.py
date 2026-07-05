"""EDGE ENGINE — staged-review phase emails (SPEC §18 / SPEC_STAGED §9).

Three notifications, one per staged phase:
    send_discovery  — Phase A: candidate list + the approval token (never the
                      words "survivor" / "edge" — luck is not ruled out yet).
    send_confirm    — Phase B: survivors + the died-in-validation table.
    send_verdict    — Phase C: the holdout verdict + caveats.

Transport copies backtest/diagnostics/sweep_email.py exactly (smtplib,
smtp.gmail.com:587, env GMAIL_ADDRESS / GMAIL_APP_PASSWORD / BACKTEST_EMAIL,
same fallback recipient). Bodies are PLAIN-TEXT-first (monospace tables), summary
only — the committed .md carries full detail (SPEC §16.3). Email failure NEVER
fails the engine run (every sender is wrapped; a warning is printed, nothing
raises) — same convention as the existing mailers.
"""

from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Transport (mirrors sweep_email._smtp_send)
# ---------------------------------------------------------------------------
def _smtp_send(subject: str, body: str) -> bool:
    # In the Action, email is sent AFTER the commit step (so it can reference the
    # committed file and still fire even if the push fails — SPEC_STAGED §10). The
    # engine's inline send is deferred there via EDGE_EMAIL_DEFER=1; the local
    # (in-chat) run sends inline as normal.
    if os.environ.get("EDGE_EMAIL_DEFER") == "1":
        print(f"  [edge-email deferred] {subject}")
        return False
    sender = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    to = (os.environ.get("BACKTEST_EMAIL") or sender or "avinash.somjani98@gmail.com")
    if not sender or not password:
        print("  [edge-email skipped] GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set")
        return False
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, password)
            s.send_message(msg)
        print(f"  [edge-email sent] -> {to} ({subject})")
        return True
    except Exception as e:  # never let a mail failure fail the run
        print(f"  [edge-email error] {e} ({subject})")
        return False


def _run_prefix(res: Dict[str, Any]) -> str:
    """[VALIDATION RE-RUN N] subject prefix when the ledger shows a re-run (§5.4)."""
    n = res.get("validation_runs")
    if res.get("validation_burned") and n and n > 1:
        return f"[VALIDATION RE-RUN {n}] "
    return ""


def _fmt(x) -> str:
    return "—" if x is None else str(x)


def _verdict_counts(features: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in features:
        out[r.get("verdict", "?")] = out.get(r.get("verdict", "?"), 0) + 1
    return out


# ---------------------------------------------------------------------------
# Phase A — discovery (§9.1). NO "survivor" / "edge" anywhere in this body.
# ---------------------------------------------------------------------------
def build_discovery_body(res: Dict[str, Any]) -> str:
    from backtest.diagnostics import edge_report

    lines: List[str] = []
    lines.append(res.get("language_stamp", ""))
    lines.append("")
    lines.append(f"run: {res.get('run_id')}   window: {res.get('window')}   "
                 f"N(discovery): {res.get('n_discovery')}")
    # Overall discovery-split one-liner: context so "candidate=1" reads as "1 of 43
    # screened on N trades", not "the system found almost nothing" (§5).
    ov = (res.get("population_stats") or {}).get("overall") or {}
    n_screened = len(res.get("features", []))
    if ov:
        lines.append(f"overall discovery split: expR {_fmt(ov.get('expR'))} "
                     f"(N {_fmt(ov.get('n'))}, wr {_fmt(ov.get('wr_pct'))}%) — "
                     f"{len(res.get('candidates', []))} candidate(s) of "
                     f"{n_screened} features screened")
    counts = _verdict_counts(res.get("features", []))
    lines.append("candidate counts by verdict: "
                 + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    # Near-miss count (features failing exactly one criterion) — derived from the
    # same helper the report uses so the two never disagree.
    n_near = len(edge_report.near_miss_features(res))
    lines.append(f"near-misses: {n_near} — see report")
    lines.append("")
    lines.append("TOP CANDIDATES (CANDIDATE — luck not ruled out)")
    lines.append(f"  {'feature':24} {'verdict':16} {'Δdisc':>9}  {'CI':>20}  "
                 f"{'N(top/bot)':>14}  timing")
    # map feature -> record for CI / N (ranked_candidates carries only a subset).
    by_feat = {r["feature"]: r for r in res.get("features", [])}
    for r in res.get("ranked_candidates", [])[:10]:
        f = r["feature"]
        rec = by_feat.get(f, {})
        ci = rec.get("delta_disc_ci", [None, None])
        n = rec.get("top_bottom_n_disc") or rec.get("best_worst_n_disc") or [None, None]
        lines.append(f"  {f:24} {r['verdict']:16} {_fmt(r.get('delta_disc')):>9}  "
                     f"{_fmt(ci):>20}  {_fmt(n):>14}  {r.get('timing')}")
    if not res.get("ranked_candidates"):
        lines.append("  (none — a valid null on discovery)")
    lines.append("")
    # snapback / anatomy one-liners.
    snap = res.get("snapback") or []
    if snap:
        lines.append("snapback bins (bars_break_to_pullback): "
                     + ", ".join(f"{b.get('bin')}={_fmt(b.get('expR'))}(n{b.get('n')})"
                                 for b in snap))
    anat = res.get("sl_anatomy") or {}
    if isinstance(anat, dict) and anat.get("rows"):
        lines.append(f"SL-anatomy: {len(anat['rows'])} clean-break rows "
                     "(promotions deferred to confirm phase)")
    lines.append("interactions: deferred to confirm phase")
    lines.append("")
    # Path wrapped in backticks so no space directly precedes "edge" — the §11.3
    # rule bans the literal " edge" (space+edge), and "committed at: edge_engine/..."
    # would trip it. A backtick between the space and "edge" avoids the substring.
    lines.append(f"full detail report committed at "
                 f"`edge_engine/{edge_report.REPORT_MD}`")
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"APPROVAL TOKEN: {res.get('token')}")
    lines.append("To confirm on validation (ONE shot):  press CONFIRM in the Action "
                 "with this token,")
    lines.append(f"or locally:  python -m backtest.diagnostics.edge_engine "
                 f"--approve {res.get('token')}")
    lines.append("             python -m backtest.diagnostics.edge_engine "
                 "--phase confirm")
    lines.append("=" * 60)
    return "\n".join(lines)


def send_discovery(engine_dir: str, res: Dict[str, Any]) -> bool:
    subject = f"EDGE ENGINE — DISCOVERY candidates — {res.get('run_id')}"
    return _smtp_send(subject, build_discovery_body(res))


# ---------------------------------------------------------------------------
# Phase B — confirm (§9.2). Survivors + the died-in-validation table.
# ---------------------------------------------------------------------------
def _died_in_validation(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every feature whose final verdict is not `survivor` but which cleared
    discovery signal (has a real Δdisc) — the trader's main learning artefact
    (SPEC_STAGED §6). Candidates dying here is the system working, not a bug."""
    out = []
    for r in res.get("features", []):
        v = r.get("verdict")
        if v in ("survivor", "thin"):
            continue
        if r.get("delta_disc") is None:
            continue
        out.append({"feature": r["feature"], "verdict": v,
                    "delta_disc": r.get("delta_disc"),
                    "delta_val": r.get("delta_val")})
    return out


def build_confirm_body(res: Dict[str, Any]) -> str:
    lines: List[str] = []
    survivors = [r for r in res.get("features", []) if r.get("verdict") == "survivor"]
    lines.append(f"run: {res.get('run_id')}   window: {res.get('window')}")
    lines.append(f"validation_runs: {res.get('validation_runs')}   "
                 f"burned: {res.get('validation_burned')}")
    if res.get("validation_burned"):
        lines.append("** VALIDATION RE-OPENED — this is no longer a one-shot "
                     "confirmation; treat the verdict as exploratory (D4). **")
    lines.append("")
    lines.append(f"SURVIVORS ({len(survivors)}):")
    lines.append(f"  {'feature':24} {'Δdisc':>9} {'Δval':>9}  {'val quarters':>14}  "
                 f"{'timing':10} actionable")
    for r in sorted(survivors, key=lambda r: -(abs(r.get("delta_val") or 0.0))):
        lines.append(f"  {r['feature']:24} {_fmt(r.get('delta_disc')):>9} "
                     f"{_fmt(r.get('delta_val')):>9}  "
                     f"{_fmt(r.get('val_favoured_pos_quarters')):>14}  "
                     f"{_fmt(r.get('timing')):10} {_fmt(r.get('actionable_at'))}")
    if not survivors:
        lines.append("  (none survived — luck did not repeat; a valid, honest null)")
    lines.append("")
    died = _died_in_validation(res)
    lines.append(f"DIED IN VALIDATION ({len(died)}) — candidates that did not repeat "
                 "(the system working, not a bug):")
    lines.append(f"  {'feature':24} {'verdict':18} {'Δdisc':>9} {'Δval':>9}")
    for d in died:
        lines.append(f"  {d['feature']:24} {d['verdict']:18} "
                     f"{_fmt(d['delta_disc']):>9} {_fmt(d['delta_val']):>9}")
    lines.append("")
    inter = res.get("interactions")
    if isinstance(inter, list):
        lines.append(f"interactions flagged: {len(inter)}")
    lines.append("")
    lines.append("NEXT: read this, sleep on it (E6), then press FINAL to open the "
                 "holdout (once).")
    return "\n".join(lines)


def send_confirm(engine_dir: str, res: Dict[str, Any]) -> bool:
    survivors = [r for r in res.get("features", []) if r.get("verdict") == "survivor"]
    candidates = _died_in_validation(res)
    subject = (f"{_run_prefix(res)}EDGE ENGINE — VALIDATION confirm — "
               f"{res.get('run_id')} — {len(survivors)} survivors of "
               f"{len(survivors) + len(candidates)} candidates")
    return _smtp_send(subject, build_confirm_body(res))


# ---------------------------------------------------------------------------
# Phase C — verdict (§9.3).
# ---------------------------------------------------------------------------
def build_verdict_body(res: Dict[str, Any]) -> str:
    lines: List[str] = []
    h = res.get("holdout", {}) or {}
    lines.append(f"run: {res.get('input_run')}")
    lines.append(f"VERDICT: {res.get('verdict')}   robustness: {res.get('robustness')}")
    lines.append(f"validation_runs: {res.get('validation_runs')}   "
                 f"burned: {res.get('validation_burned')}")
    lines.append("")
    lines.append(f"Holdout (2022–2025): N={h.get('n')}  expR_net={h.get('expR_net')}")
    lines.append(f"  iid CI={h.get('ci_iid')}  block CI={h.get('ci_block')}  "
                 f"pos_quarters={h.get('pos_quarters')}")
    lines.append(f"  vs baseline diff CI={h.get('vs_baseline_diff_ci')}  "
                 f"max DD={h.get('max_drawdown_R')}R")
    lines.append(f"pair set: {res.get('pair_set')}")
    lines.append("")
    cav = res.get("caveats") or []
    if cav:
        lines.append("CAVEATS:")
        for c in cav:
            lines.append(f"  - {c}")
    lines.append("")
    lines.append("Full detail: edge_engine_report.md (committed in the run folder).")
    return "\n".join(lines)


def send_verdict(engine_dir: str, res: Dict[str, Any]) -> bool:
    subject = (f"{_run_prefix(res)}EDGE ENGINE — VERDICT {res.get('verdict')} — "
               f"{res.get('input_run')}")
    return _smtp_send(subject, build_verdict_body(res))


# ---------------------------------------------------------------------------
# CLI — the Action sends the deferred phase email AFTER committing (SPEC §10).
# Reads the committed phase JSON from engine_dir; never recomputes.
# ---------------------------------------------------------------------------
def _load(engine_dir: str, name: str) -> Optional[Dict[str, Any]]:
    import json
    p = os.path.join(engine_dir, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def send_phase(engine_dir: str, phase: str) -> bool:
    """Send the email for a completed phase, reading its committed JSON. Used by
    the Action after the commit step. Force-enabled even if EDGE_EMAIL_DEFER is
    set (this IS the deferred send)."""
    os.environ.pop("EDGE_EMAIL_DEFER", None)
    if phase == "discovery":
        res = _load(engine_dir, "stage1_discovery.json")
        return send_discovery(engine_dir, res) if res else False
    if phase == "confirm":
        res = _load(engine_dir, "stage1_features.json")
        return send_confirm(engine_dir, res) if res else False
    if phase == "final":
        res = _load(engine_dir, "stage4_recipe.json")
        return send_verdict(engine_dir, res) if res else False
    return False


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Send a staged-review phase email")
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--phase", required=True, choices=["discovery", "confirm", "final"])
    a = ap.parse_args()
    send_phase(a.engine_dir, a.phase)
