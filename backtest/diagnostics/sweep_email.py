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
    # Robustness of the peak = is the BEST neighbour also close to the peak?
    # A peak whose neighbour holds up is "robust"; one that falls off a ledge is
    # "fragile". This works at a grid EDGE too: an edge peak has only one
    # neighbour, and we judge robustness off that one (the old code wrongly
    # called every edge peak a "cliff" because it had no left-neighbour).
    near = []
    if peak_i > 0:
        near.append(exps[peak_i - 1])
    if peak_i < n - 1:
        near.append(exps[peak_i + 1])
    if near and (exps[peak_i] - max(near)) <= PLATEAU_TOL_R:
        return "robust-peak"
    return "fragile-peak"


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


# Plain-English label for each guardrail that failed — no internal jargon ever
# reaches the email. Keyed by the check name in `analyse`.
_PLAIN_BLOCK = {
    "sample": "not enough trades to trust it",
    "shape": "the result is fragile — the next setting drops off sharply",
    "agreement": "the pairs do not agree",
    "beats_baseline": "it is not clearly better than the current setting",
    "integrity": "the run failed its data checks",
}


def analyse(rows: List[Dict[str, Any]], health: Dict[str, Any],
            risk_usd: float = 250.0) -> Dict[str, Any]:
    """Produce the full analysis dict the renderer turns into HTML/subject.
    Carries plain-English strings so the renderer never has to translate."""
    pooled = _pool_by_value(rows)
    base_v = _baseline_value(pooled)
    best_v = _best_value(pooled)
    shape = _shape(pooled)
    integrity_ok = bool(health.get("recon_ok", True)) and bool(health.get("scope_ok", True))

    agree = total = 0
    promote = False
    blocks: List[str] = []
    gain_r = 0.0
    if best_v is not None:
        agree, total = _pairs_agreeing(rows, best_v)
        best_filled = pooled[best_v]["filled"]
        gain_r = round(pooled[best_v]["expectancy_r"] -
                       (pooled[base_v]["expectancy_r"] if base_v in pooled else 0.0), 4)
        # Guardrails — ALL must hold to call a value promote-worthy. "Robust"
        # now includes a holding-up edge peak; only a genuine fragile-peak fails.
        checks = {
            "integrity": integrity_ok,
            "sample": best_filled >= MIN_FILLED_POOLED,
            "shape": shape in ("robust-peak", "monotonic-up", "monotonic-down"),
            "agreement": (total > 0 and agree / total >= AGREE_FRACTION),
            "beats_baseline": (best_v != base_v and gain_r > PLATEAU_TOL_R),
        }
        promote = all(checks.values())
        blocks = [_PLAIN_BLOCK[k] for k, ok in checks.items() if not ok]

    # Verdict tag for the subject line.
    if not integrity_ok:
        verdict = "INTEGRITY ✗"
    elif shape == "flat":
        verdict = "NO EFFECT ○"
    elif best_v is not None and pooled[best_v]["filled"] < MIN_FILLED_POOLED:
        verdict = "TOO FEW TRADES ⚠"
    elif promote:
        verdict = "WORTH A LOOK ▲"
    else:
        verdict = "KEEP AS-IS ●"

    return {
        "pooled": pooled, "base_v": base_v, "best_v": best_v, "shape": shape,
        "integrity_ok": integrity_ok, "agree": agree, "agree_total": total,
        "promote": promote, "blocks": blocks, "gain_r": gain_r,
        "gain_usd": round(gain_r * risk_usd, 2), "risk_usd": risk_usd,
        "verdict": verdict,
        "quality_note": _alerts_vs_quality(pooled, best_v, base_v),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_TH = "padding:5px 9px;border:1px solid #ddd;text-align:right;"
_TD = "padding:5px 9px;border:1px solid #ddd;text-align:right;"


def _money(r: float, risk: float) -> str:
    """R -> a plain dollar string, signed."""
    d = r * risk
    return f"${d:,.0f}" if d >= 0 else f"-${abs(d):,.0f}"


def _knob_meaning(knob: str) -> str:
    """One plain sentence: which direction is looser vs stricter. Helps read the
    table without knowing the internals."""
    looser = {
        "BOS_ATR_MULT": "Lower = a smaller move counts as a break (more setups). "
                        "Higher = needs a bigger move (fewer, cleaner setups).",
        "MIN_LEG_ATR_MULT": "Lower = accepts smaller swings (more structure). "
                            "Higher = only big swings count (less, stronger).",
        "OB_MAX_RANGE_ATR_MULT": "Higher = allows bigger order blocks. "
                                 "Lower = only tight order blocks.",
        "MIN_OB_RANGE_ATR_MULT": "Higher = throws out the smallest order blocks.",
    }
    return looser.get(knob, "Higher = stricter, fewer setups. Lower = looser, more setups.")


def _pooled_table(a: Dict[str, Any]) -> str:
    pooled = a["pooled"]
    risk = a["risk_usd"]
    base_exp = pooled.get(a["base_v"], {}).get("expectancy_r", 0.0)
    head = ("<tr style='background:#f0f0f0;'>"
            f"<th style='{_TH}text-align:left;'>setting</th>"
            f"<th style='{_TH}'>trades</th>"
            f"<th style='{_TH}'>total</th>"
            f"<th style='{_TH}'>per&nbsp;trade</th>"
            f"<th style='{_TH}'>vs&nbsp;now</th></tr>")
    body = []
    for v in sorted(pooled, key=lambda x: (isinstance(x, str), x)):
        p = pooled[v]
        per_r = p["expectancy_r"]
        diff_r = round(per_r - base_exp, 4)
        dcol = "#0a7d28" if diff_r > 0 else ("#b00" if diff_r < 0 else "#777")
        is_best = (v == a["best_v"])
        is_base = p["baseline"]
        label = f"{v}" + (" (now)" if is_base else "")
        rowstyle = "background:#fffbe6;" if is_best else ""
        namecell = (f"<td style='{_TD}text-align:left;"
                    f"{'font-weight:bold;' if is_best else ''}'>{label}</td>")
        body.append(
            f"<tr style='{rowstyle}'>" + namecell
            + f"<td style='{_TD}'>{p['filled']}</td>"
            + f"<td style='{_TD}'>{_money(p['sum_r'], risk)}<br>"
              f"<span style='color:#999;font-size:11px;'>{round(p['sum_r'],2)}R</span></td>"
            + f"<td style='{_TD}{'font-weight:bold;' if is_best else ''}'>"
              f"{_money(per_r, risk)}<br>"
              f"<span style='color:#999;font-size:11px;'>{round(per_r,2)}R</span></td>"
            + f"<td style='{_TD}color:{dcol};'>{('+' if diff_r>=0 else '')}{_money(diff_r,risk)}"
              f"<br><span style='font-size:11px;'>{diff_r:+.2f}R</span></td>"
            + "</tr>")
    return ("<table style='border-collapse:collapse;font-family:Arial,sans-serif;"
            f"font-size:13px;'>{head}{''.join(body)}</table>"
            "<div style='color:#888;font-size:11px;margin-top:4px;'>"
            f"\"per trade\" is the average result of one trade at that setting. "
            f"\"vs now\" compares it to your current setting. Dollars use "
            f"${risk:,.0f} risk per trade; the small grey number is the same in R.</div>")


def _per_pair_table(rows: List[Dict[str, Any]], risk: float) -> str:
    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_pair.setdefault(r.get("pair"), []).append(r)
    out = []
    for pair in sorted(by_pair):
        prows = sorted(by_pair[pair],
                       key=lambda r: (isinstance(r.get("grid_value"), str),
                                      r.get("grid_value")))
        head = ("<tr style='background:#f5f5f5;'>"
                f"<th style='{_TH}text-align:left;'>setting</th>"
                f"<th style='{_TH}'>trades</th>"
                f"<th style='{_TH}'>per&nbsp;trade</th>"
                f"<th style='{_TH}'>win&nbsp;%</th></tr>")
        best = max(prows, key=lambda r: _f(r.get("expectancy_r"))) if prows else None
        body = []
        for r in prows:
            is_best = best is not None and r is best and _f(r.get("expectancy_r")) != 0
            rowstyle = "background:#fffbe6;" if is_best else ""
            wr = r.get("win_rate")
            wr_s = "—" if wr is None else f"{round(_f(wr)*100)}%"
            per_r = _f(r.get("expectancy_r"))
            body.append(
                f"<tr style='{rowstyle}'>"
                f"<td style='{_TD}text-align:left;{'font-weight:bold;' if is_best else ''}'>"
                f"{r.get('grid_value')}</td>"
                f"<td style='{_TD}'>{r.get('n_trades_filled')}</td>"
                f"<td style='{_TD}{'font-weight:bold;' if is_best else ''}'>"
                f"{_money(per_r, risk)} <span style='color:#999;font-size:11px;'>"
                f"{round(per_r,2)}R</span></td>"
                f"<td style='{_TD}'>{wr_s}</td></tr>")
        out.append(f"<h4 style='margin:14px 0 4px;'>{pair}</h4>"
                   "<table style='border-collapse:collapse;font-family:Arial,sans-serif;"
                   f"font-size:12px;'>{head}{''.join(body)}</table>")
    return "".join(out)


def render_html(manifest: Dict[str, Any], rows: List[Dict[str, Any]],
                health: Dict[str, Any], a: Dict[str, Any]) -> str:
    knob = manifest.get("knob", "?")
    win = f"{manifest.get('resolved_start')} to {manifest.get('resolved_end')}"
    risk = a["risk_usd"]
    base_v, best_v = a["base_v"], a["best_v"]
    pooled = a["pooled"]
    best_fill = pooled.get(best_v, {}).get("filled", 0)

    # Red banner first if the data failed its checks — no advice in that case.
    integ = ""
    if not a["integrity_ok"]:
        integ = ("<div style='background:#ffdddd;border:2px solid #b00;padding:12px;"
                 "margin:0 0 12px;'><b>This run failed its data checks.</b> "
                 "No recommendation is given — the numbers are not trustworthy. "
                 f"Details: {health.get('recon_failures')} {health.get('scope_violations')}</div>")

    # THE BOTTOM LINE — plain, dollars first.
    same = (best_v == base_v)
    if same:
        bottom = (f"The best setting this month was your current one "
                  f"(<b>{base_v}</b>). Nothing to change.")
    else:
        sign = "more" if a["gain_r"] > 0 else "less"
        bottom = (f"The best setting this month was <b>{best_v}</b> "
                  f"(you use <b>{base_v}</b> now). At {best_v}, the average trade made "
                  f"<b>{_money(a['gain_r'], risk)} {sign}</b> than now "
                  f"(<span style='color:#999;'>{a['gain_r']:+.2f}R</span>), "
                  f"over {best_fill} trades.")

    tldr = (f"<div style='background:#eef4ff;border:1px solid #aac;padding:14px;"
            f"margin:0 0 14px;font-size:15px;'>"
            f"<div style='font-size:17px;margin-bottom:6px;'><b>{a['verdict']}</b></div>"
            f"{bottom}</div>")

    # THE ADVICE — two plain sentences, never "apply this".
    if not a["integrity_ok"]:
        rec = ""
    elif a["promote"]:
        rec = (f"<div style='background:#e7f7ea;border:1px solid #8c8;padding:12px;"
               f"margin:12px 0;font-size:14px;'>"
               f"<b>Worth a closer look: try {best_v}.</b> It passed every check this "
               f"month — enough trades, the nearby settings also held up, and most "
               f"pairs agreed.<br><b>But do not change anything on one month.</b> "
               f"Run this same knob on another month first. If {best_v} wins again, "
               f"then it is real.</div>")
    else:
        why = a["blocks"][0] if a["blocks"] else "it is inside normal noise"
        rec = (f"<div style='background:#fff4e5;border:1px solid #e0b070;padding:12px;"
               f"margin:12px 0;font-size:14px;'>"
               f"<b>Keep your current setting ({base_v}) for now.</b> "
               f"The best-looking setting ({best_v}) is not a safe bet yet — {why}. "
               f"One month is too little to act on. This run is saved, and the picture "
               f"gets clearer as more months are added.</div>")

    # A concern line if the pooled headline and the pairs disagree.
    concern = ""
    if a["quality_note"] or (a["agree_total"] and a["agree"] < a["agree_total"]):
        bits = []
        if a["agree_total"]:
            bits.append(f"{a['agree']} of {a['agree_total']} pairs liked the best "
                        f"setting — check the per-pair tables to see who did not.")
        if a["quality_note"]:
            bits.append(a["quality_note"])
        concern = ("<div style='color:#555;font-size:13px;margin:6px 0 14px;'>"
                   "<b>Worth noticing:</b> " + " ".join(bits) + "</div>")

    meaning = (f"<div style='color:#555;font-size:13px;margin:4px 0 12px;'>"
               f"<b>What this knob does:</b> {_knob_meaning(knob)}</div>")

    weaknesses = (
        "<div style='color:#666;font-size:12px;margin-top:18px;'>"
        "<b>Keep in mind:</b><ul style='margin-top:4px;'>"
        "<li>This is one month only. A good number here is a clue, not a decision. "
        "Real answers come from many months pooled together.</li>"
        "<li>Only one knob was changed. The best setting here may not be best once "
        "other knobs move too.</li>"
        "<li>The combined number can be pulled around by one pair. Always glance at "
        "the per-pair tables below.</li></ul></div>")

    versions = manifest.get("versions", {})
    footer = (f"<div style='color:#aaa;font-size:11px;margin-top:12px;'>"
              f"run {manifest.get('run_id')} · code {manifest.get('git_sha')} · "
              f"settings tested {manifest.get('grid')} · "
              f"check {health.get('content_hash','')[:12]}</div>")

    return (
        f"<div style='font-family:Arial,sans-serif;max-width:760px;color:#222;'>"
        f"<h2 style='margin-bottom:2px;'>Knob test — {knob}</h2>"
        f"<div style='color:#555;margin-bottom:12px;'>{win} &nbsp;·&nbsp; "
        f"pairs: {', '.join(manifest.get('pairs_requested', []))}</div>"
        f"{integ}{tldr}{rec}{concern}{meaning}"
        f"<h3 style='margin-top:18px;'>What each setting did (all pairs together)</h3>"
        f"{_pooled_table(a)}"
        f"<h3 style='margin-top:20px;'>Each pair on its own "
        f"<span style='font-weight:normal;font-size:13px;color:#777;'>"
        f"(is one pair carrying the result?)</span></h3>"
        f"{_per_pair_table(rows, risk)}"
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
    a = analyse(rows, health, risk_usd=float(manifest.get("risk_usd", 250.0)))
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
