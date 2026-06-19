"""Per-knob sweep email — a cautious decision AID, never a decision.

Design principle (agreed with the trader): on ONE month of in-sample data the
email must not talk you into overfitting. It shows the SHAPE of the curve and
where it is fragile, states a candidate only under guardrails, and always names
its own weakness. A recommendation is "promote-worthy" ONLY if:
    filled >= MIN_FILLED  AND  it is a plateau (not a cliff)
    AND pairs agree        AND recon ok / no scope violation.
Otherwise the verdict is HOLD / LOW-N / INERT — no matter how pretty the peak.

Reads ONLY results.jsonl + manifest + run_health (the single source of truth);
it never recomputes and never parses the markdown. One email per knob.

Sender reuses the backtest SMTP env-var convention (GMAIL_ADDRESS /
GMAIL_APP_PASSWORD / BACKTEST_EMAIL).
"""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backtest.diagnostics import sweep_logging as sl

# Guardrail thresholds. A candidate value is only ever "promote-worthy" if it
# clears ALL of these. One month is in-sample; these stop a lucky peak on a
# handful of trades from reading as a real edge.
MIN_FILLED_POOLED = 12      # pooled filled trades behind the headline value
MIN_FILLED_PER_PAIR = 4     # a pair's own number is anecdote below this
PLATEAU_TOL_R = 0.05        # neighbours within this of the peak = a plateau
AGREE_FRACTION = 0.5        # >= this share of pairs must favour the value


# ---------------------------------------------------------------------------
# Analysis — pure functions over result rows. No I/O, fully testable.
# ---------------------------------------------------------------------------
def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _pool_by_value(rows: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """Pool rows across pairs, per grid value. Pooled expectancy is the
    filled-WEIGHTED mean (sum_r / filled), NOT a mean of per-pair expectancies
    — a pair with 1 trade must not swing the headline like a pair with 10."""
    pooled: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        v = r.get("grid_value")
        p = pooled.setdefault(v, {"grid_value": v, "filled": 0, "sum_r": 0.0,
                                  "alerts": 0, "obs": 0, "wins": 0, "resolved": 0,
                                  "baseline": bool(r.get("baseline"))})
        p["filled"] += int(r.get("n_trades_filled") or 0)
        p["sum_r"] += _f(r.get("sum_r_realised"))
        p["alerts"] += int(r.get("n_alerts_total") or 0)
        p["obs"] += int(r.get("n_obs") or 0)
        p["baseline"] = p["baseline"] or bool(r.get("baseline"))
    for p in pooled.values():
        p["expectancy_r"] = round(p["sum_r"] / p["filled"], 4) if p["filled"] else 0.0
    return pooled


def _baseline_value(pooled: Dict[Any, Dict[str, Any]]):
    for v, p in pooled.items():
        if p["baseline"]:
            return v
    return None


def _best_value(pooled: Dict[Any, Dict[str, Any]]):
    """Best = highest pooled expectancy among values with ANY filled trades.
    Ties broken toward more filled trades (more evidence)."""
    cand = [(v, p) for v, p in pooled.items() if p["filled"] > 0]
    if not cand:
        return None
    cand.sort(key=lambda vp: (vp[1]["expectancy_r"], vp[1]["filled"]), reverse=True)
    return cand[0][0]


def _shape(pooled: Dict[Any, Dict[str, Any]]) -> str:
    """Classify the expectancy curve over the sorted numeric grid."""
    try:
        items = sorted(((float(v), p["expectancy_r"]) for v, p in pooled.items()),
                       key=lambda t: t[0])
    except (TypeError, ValueError):
        return "unordered"
    exps = [e for _, e in items]
    if len({round(e, 4) for e in exps}) == 1:
        return "flat"
    n = len(exps)
    peak_i = max(range(n), key=lambda i: exps[i])
    rising = all(exps[i] <= exps[i + 1] + 1e-9 for i in range(n - 1))
    falling = all(exps[i] >= exps[i + 1] - 1e-9 for i in range(n - 1))
    if rising:
        return "monotonic-up"
    if falling:
        return "monotonic-down"
    # interior peak: plateau if a neighbour is within tol, else a cliff.
    near = []
    if peak_i > 0:
        near.append(exps[peak_i - 1])
    if peak_i < n - 1:
        near.append(exps[peak_i + 1])
    if near and (exps[peak_i] - max(near)) <= PLATEAU_TOL_R:
        return "plateau"
    return "cliff"


def _pairs_agreeing(rows: List[Dict[str, Any]], value) -> Tuple[int, int]:
    """Of pairs with >= MIN_FILLED_PER_PAIR trades at `value`, how many have
    that value as their own best (or tied-best) expectancy. Returns (agree, total)."""
    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_pair.setdefault(r.get("pair"), []).append(r)
    agree = total = 0
    for pair, prows in by_pair.items():
        at_v = next((r for r in prows if r.get("grid_value") == value), None)
        if not at_v or int(at_v.get("n_trades_filled") or 0) < MIN_FILLED_PER_PAIR:
            continue
        total += 1
        best_here = max(prows, key=lambda r: _f(r.get("expectancy_r")))
        if _f(at_v.get("expectancy_r")) >= _f(best_here.get("expectancy_r")) - 1e-9:
            agree += 1
    return agree, total


def _alerts_vs_quality(pooled, best_v, base_v) -> str:
    """The vet's question: more trades, or better trades?"""
    if best_v is None or base_v is None or best_v == base_v:
        return ""
    b, base = pooled.get(best_v), pooled.get(base_v)
    if not b or not base:
        return ""
    more_alerts = b["alerts"] > base["alerts"]
    better_exp = b["expectancy_r"] > base["expectancy_r"]
    if more_alerts and not better_exp:
        return "more trades but NOT better — classic noise-adding direction."
    if not more_alerts and better_exp:
        return "fewer but better trades — selectivity is paying."
    if more_alerts and better_exp:
        return "more AND better — rare; treat with suspicion on one month."
    return "fewer trades and no better — strictly worse."


def analyse(rows: List[Dict[str, Any]], health: Dict[str, Any]) -> Dict[str, Any]:
    """Produce the full analysis dict the renderer turns into HTML/subject."""
    pooled = _pool_by_value(rows)
    base_v = _baseline_value(pooled)
    best_v = _best_value(pooled)
    shape = _shape(pooled)
    integrity_ok = bool(health.get("recon_ok", True)) and bool(health.get("scope_ok", True))

    agree = total = 0
    promote = False
    reason = ""
    if best_v is not None:
        agree, total = _pairs_agreeing(rows, best_v)
        best_filled = pooled[best_v]["filled"]
        gain = (pooled[best_v]["expectancy_r"] -
                (pooled[base_v]["expectancy_r"] if base_v in pooled else 0.0))
        # Guardrails — ALL must hold to call a value promote-worthy.
        checks = {
            "integrity": integrity_ok,
            "sample": best_filled >= MIN_FILLED_POOLED,
            "shape": shape in ("plateau", "monotonic-up", "monotonic-down"),
            "agreement": (total > 0 and agree / total >= AGREE_FRACTION),
            "beats_baseline": (best_v != base_v and gain > PLATEAU_TOL_R),
        }
        promote = all(checks.values())
        failed = [k for k, ok in checks.items() if not ok]
        reason = ("all guardrails clear" if promote
                  else "blocked by: " + ", ".join(failed))

    # Verdict tag for the subject line.
    if not integrity_ok:
        verdict = "INTEGRITY ✗"
    elif shape == "flat":
        verdict = "INERT ○"
    elif best_v is not None and pooled[best_v]["filled"] < MIN_FILLED_POOLED:
        verdict = "LOW-N ⚠"
    elif promote:
        verdict = "MOVE ▲"
    else:
        verdict = "HOLD ●"

    return {
        "pooled": pooled, "base_v": base_v, "best_v": best_v, "shape": shape,
        "integrity_ok": integrity_ok, "agree": agree, "agree_total": total,
        "promote": promote, "promote_reason": reason, "verdict": verdict,
        "quality_note": _alerts_vs_quality(pooled, best_v, base_v),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _td(x, bold=False, color=None):
    style = "padding:4px 8px;border:1px solid #ddd;"
    if color:
        style += f"color:{color};"
    if bold:
        style += "font-weight:bold;"
    return f"<td style='{style}'>{x}</td>"


def _pooled_table(a: Dict[str, Any]) -> str:
    pooled = a["pooled"]
    base_v = a["base_v"]
    best_exp_base = pooled.get(base_v, {}).get("expectancy_r", 0.0)
    head = ("<tr style='background:#f0f0f0;'>"
            "<th style='padding:4px 8px;border:1px solid #ddd;'>value</th>"
            "<th style='padding:4px 8px;border:1px solid #ddd;'>base</th>"
            "<th style='padding:4px 8px;border:1px solid #ddd;'>filled</th>"
            "<th style='padding:4px 8px;border:1px solid #ddd;'>sumR</th>"
            "<th style='padding:4px 8px;border:1px solid #ddd;'>exp_R</th>"
            "<th style='padding:4px 8px;border:1px solid #ddd;'>Δexp vs base</th>"
            "<th style='padding:4px 8px;border:1px solid #ddd;'>alerts</th>"
            "<th style='padding:4px 8px;border:1px solid #ddd;'>OBs</th></tr>")
    body = []
    for v in sorted(pooled, key=lambda x: (isinstance(x, str), x)):
        p = pooled[v]
        d = round(p["expectancy_r"] - best_exp_base, 4)
        dcol = "#0a7d28" if d > 0 else ("#b00" if d < 0 else "#555")
        is_best = (v == a["best_v"])
        rowstyle = "background:#fffbe6;" if is_best else ""
        body.append(
            f"<tr style='{rowstyle}'>"
            + _td(v, bold=is_best) + _td("✓" if p["baseline"] else "")
            + _td(p["filled"]) + _td(round(p["sum_r"], 3))
            + _td(p["expectancy_r"], bold=is_best)
            + _td(f"{d:+.3f}", color=dcol) + _td(p["alerts"]) + _td(p["obs"])
            + "</tr>")
    return ("<table style='border-collapse:collapse;font-family:monospace;"
            f"font-size:13px;'>{head}{''.join(body)}</table>")


def _per_pair_table(rows: List[Dict[str, Any]]) -> str:
    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_pair.setdefault(r.get("pair"), []).append(r)
    out = []
    for pair in sorted(by_pair):
        prows = sorted(by_pair[pair],
                       key=lambda r: (isinstance(r.get("grid_value"), str),
                                      r.get("grid_value")))
        head = ("<tr style='background:#f0f0f0;'>"
                "<th style='padding:3px 7px;border:1px solid #eee;'>value</th>"
                "<th style='padding:3px 7px;border:1px solid #eee;'>filled</th>"
                "<th style='padding:3px 7px;border:1px solid #eee;'>exp_R</th>"
                "<th style='padding:3px 7px;border:1px solid #eee;'>WR</th>"
                "<th style='padding:3px 7px;border:1px solid #eee;'>OBs</th></tr>")
        best = max(prows, key=lambda r: _f(r.get("expectancy_r"))) if prows else None
        body = []
        for r in prows:
            is_best = best is not None and r is best and _f(r.get("expectancy_r")) != 0
            rowstyle = "background:#fffbe6;" if is_best else ""
            wr = r.get("win_rate")
            wr_s = "—" if wr is None else f"{_f(wr):.2f}"
            body.append(
                f"<tr style='{rowstyle}'>"
                + _td(r.get("grid_value"), bold=is_best)
                + _td(r.get("n_trades_filled"))
                + _td(r.get("expectancy_r"), bold=is_best)
                + _td(wr_s) + _td(r.get("n_obs")) + "</tr>")
        out.append(f"<h4 style='margin:14px 0 4px;'>{pair}</h4>"
                   "<table style='border-collapse:collapse;font-family:monospace;"
                   f"font-size:12px;'>{head}{''.join(body)}</table>")
    return "".join(out)


def render_html(manifest: Dict[str, Any], rows: List[Dict[str, Any]],
                health: Dict[str, Any], a: Dict[str, Any]) -> str:
    knob = manifest.get("knob", "?")
    win = f"{manifest.get('resolved_start')} → {manifest.get('resolved_end')}"
    base_v, best_v = a["base_v"], a["best_v"]
    pooled = a["pooled"]

    # Integrity banner first, in red, if anything is off — recommendation is
    # suppressed in that case.
    integ = ""
    if not a["integrity_ok"]:
        integ = ("<div style='background:#ffdddd;border:2px solid #b00;padding:10px;"
                 "margin:0 0 12px;'><b>INTEGRITY FAILURE — recommendation suppressed.</b>"
                 f"<br>recon_failures: {health.get('recon_failures')}<br>"
                 f"scope_violations: {health.get('scope_violations')}</div>")

    # TL;DR — three lines, the whole point of the email if you read nothing else.
    base_exp = pooled.get(base_v, {}).get("expectancy_r")
    best_exp = pooled.get(best_v, {}).get("expectancy_r")
    best_fill = pooled.get(best_v, {}).get("filled")
    tldr = (
        f"<div style='background:#eef4ff;border:1px solid #aac;padding:12px;"
        f"margin:0 0 14px;font-size:14px;'>"
        f"<b>{knob} — {a['verdict']}</b><br>"
        f"Live baseline = <b>{base_v}</b> (pooled exp {base_exp}). "
        f"Best pooled value = <b>{best_v}</b> (exp {best_exp} on {best_fill} filled). "
        f"Curve shape: <b>{a['shape']}</b>. "
        f"Cross-pair agreement: <b>{a['agree']}/{a['agree_total']}</b>.<br>"
        f"<b>Verdict: {a['verdict']}</b> — {a['promote_reason']}."
        + (f"<br><i>{a['quality_note']}</i>" if a['quality_note'] else "")
        + "</div>")

    # Recommendation block — guarded, never "apply this".
    if not a["integrity_ok"]:
        rec = ""
    elif a["promote"]:
        rec = (f"<div style='background:#e7f7ea;border:1px solid #8c8;padding:10px;"
               f"margin:12px 0;'><b>Candidate for confirmation: {best_v}</b> "
               f"(baseline {base_v}).<br>It clears every guardrail on THIS month: "
               f"sample ≥ {MIN_FILLED_POOLED} filled, {a['shape']} (not a cliff), "
               f"pairs agree {a['agree']}/{a['agree_total']}, recon clean.<br>"
               f"<b>Do NOT change live on one month.</b> Re-run this knob on a second, "
               f"out-of-sample month and confirm the same value before touching config.</div>")
    else:
        rec = (f"<div style='background:#fff4e5;border:1px solid #e0b070;padding:10px;"
               f"margin:12px 0;'><b>Hold the baseline ({base_v}).</b><br>"
               f"The peak at {best_v} is not promote-worthy: {a['promote_reason']}. "
               f"On one in-sample month this is inside the noise — collecting it into "
               f"the two-year corpus is the right move, acting on it is not.</div>")

    weaknesses = (
        "<div style='color:#666;font-size:12px;margin-top:16px;'>"
        "<b>Honest weaknesses (always true):</b><ul>"
        "<li>ONE calendar month, in-sample. A best value here is a data point, "
        "not a decision. The recommendation engine pools across two years.</li>"
        "<li>One knob at a time — interaction effects are not explored; a best "
        "value here does not compose into a best joint config.</li>"
        "<li>Pooled expectancy is filled-weighted; a thin pair can still skew it. "
        "Read the per-pair table, not just the headline.</li>"
        "</ul></div>")

    versions = manifest.get("versions", {})
    footer = (f"<div style='color:#999;font-size:11px;margin-top:12px;'>"
              f"run_id {manifest.get('run_id')} · git {manifest.get('git_sha')} · "
              f"grid {manifest.get('grid')} ({manifest.get('grid_mode')}) · "
              f"py {versions.get('python')} pandas {versions.get('pandas')} · "
              f"content_hash {health.get('content_hash','')[:12]}</div>")

    return (
        f"<div style='font-family:Arial,sans-serif;max-width:820px;'>"
        f"<h2 style='margin-bottom:2px;'>Knob Sweep — {knob}</h2>"
        f"<div style='color:#555;margin-bottom:10px;'>{win} · "
        f"pairs {', '.join(manifest.get('pairs_requested', []))}</div>"
        f"{integ}{tldr}{rec}"
        f"<h3>Pooled across pairs</h3>{_pooled_table(a)}"
        f"<h3 style='margin-top:18px;'>Per pair (is one instrument carrying it?)</h3>"
        f"{_per_pair_table(rows)}"
        f"{weaknesses}{footer}</div>")


def subject_for(manifest: Dict[str, Any], a: Dict[str, Any]) -> str:
    return (f"Knob Sweep — {manifest.get('knob')} — "
            f"{manifest.get('year')}-{int(manifest.get('month')):02d} — {a['verdict']}")


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------
def _smtp_send(subject: str, html: str) -> bool:
    sender = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    to = (os.environ.get("BACKTEST_EMAIL") or sender or "avinash.somjani98@gmail.com")
    if not sender or not password:
        print("  [sweep-email skipped] GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set")
        return False
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, password)
            s.send_message(msg)
        print(f"  [sweep-email sent] -> {to} ({subject})")
        return True
    except Exception as e:
        print(f"  [sweep-email error] {e} ({subject})")
        return False


def send_for_run(run_dir: Path) -> bool:
    """Build and send the per-knob email from a PASSed run directory."""
    manifest = sl.read_manifest(run_dir)
    rows = sl.read_results(run_dir)
    health = sl.read_health(run_dir)
    a = analyse(rows, health)
    html = render_html(manifest, rows, health, a)
    # Always drop the HTML next to the run for the record / debugging.
    (run_dir / "email.html").write_text(html, encoding="utf-8")
    return _smtp_send(subject_for(manifest, a), html)


def send_failure_notice(run_dir: Path, health: Dict[str, Any]) -> bool:
    """A run that FAILed the gate still gets a short, loud email — silence is the
    failure mode we are designing against."""
    manifest = sl.read_manifest(run_dir)
    subject = (f"Knob Sweep — {manifest.get('knob')} — "
               f"{manifest.get('year')}-{int(manifest.get('month')):02d} — FAILED ✗")
    html = (f"<div style='font-family:Arial;max-width:700px;'>"
            f"<h2 style='color:#b00;'>Sweep run FAILED its health gate</h2>"
            f"<p>run_id <code>{manifest.get('run_id')}</code> — no recommendation "
            f"was produced; the data is not trustworthy.</p>"
            f"<ul><li>recon_ok: {health.get('recon_ok')}</li>"
            f"<li>recon_failures: {health.get('recon_failures')}</li>"
            f"<li>scope_ok: {health.get('scope_ok')}</li>"
            f"<li>scope_violations: {health.get('scope_violations')}</li>"
            f"<li>result rows: {health.get('n_result_rows')}</li></ul>"
            f"<p>Inspect the run directory before re-running.</p></div>")
    (run_dir / "email.html").write_text(html, encoding="utf-8")
    return _smtp_send(subject, html)
