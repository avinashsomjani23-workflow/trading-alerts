"""H1-only backtest report writer.

Produces:
  results/<run_id>/trades.csv        machine-readable, full column set
  results/<run_id>/trades.xlsx       human-readable, plain-English column names
  results/<run_id>/report.html       email body (3-5 min read)
  results/<run_id>/raw_alerts.jsonl  OB-touch alerts before simulation
  results/<run_id>/summary.json      all metrics, machine-readable
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Aggregation helpers (used by summary.json + HTML)
# ---------------------------------------------------------------------------

# Trades to exclude from every aggregate. never_filled = limit never hit.
# window_end / timeout = walk ran out of bars; trade unresolved, treat as
# audit-only (RCA #5). These rows remain in CSV/Excel but never feed P&L,
# win rate, expectancy, or any reported metric.
_EXCLUDE_REASONS = {"never_filled", "window_end", "timeout"}


def _is_real_filled(t: Dict[str, Any]) -> bool:
    return t.get("exit_reason") not in _EXCLUDE_REASONS


def _aggregate_for_exit(trades: List[Dict[str, Any]], r_col: str,
                        risk_usd: float) -> Dict[str, Any]:
    """Aggregate filled trades under a hypothetical exit policy."""
    filled = [t for t in trades if _is_real_filled(t)]
    if not filled:
        return {"trades": 0, "filled": 0}
    df = pd.DataFrame(filled)
    if r_col not in df.columns:
        return {"trades": 0, "filled": 0}
    wins   = df[df[r_col] > 0]
    losses = df[df[r_col] < 0]
    bes    = df[df[r_col] == 0]
    exp    = float(df[r_col].mean())
    avg_mfe = float(wins["mfe_r"].mean()) if (len(wins) and "mfe_r" in wins.columns) else 0
    avg_mae = float(losses["mae_r"].mean()) if (len(losses) and "mae_r" in losses.columns) else 0
    capture = round(float(wins[r_col].mean()) / avg_mfe * 100, 0) if (avg_mfe > 0 and len(wins)) else 0
    return {
        "trades":         int(len(df)),
        "wins":           int(len(wins)),
        "losses":         int(len(losses)),
        "breakevens":     int(len(bes)),
        "win_rate_pct":   round(len(wins) / len(df) * 100, 1) if len(df) else 0,
        "expectancy_r":   round(exp, 3),
        "expectancy_usd": round(exp * risk_usd, 2),
        "total_r":        round(float(df[r_col].sum()), 3),
        "total_pnl_usd":  round(float(df[r_col].sum()) * risk_usd, 2),
        "avg_win_r":      round(float(wins[r_col].mean()), 3) if len(wins) else 0.0,
        "avg_loss_r":     round(float(losses[r_col].mean()), 3) if len(losses) else 0.0,
        "avg_win_usd":    round(float(wins[r_col].mean()) * risk_usd, 0) if len(wins) else 0.0,
        "avg_loss_usd":   round(float(losses[r_col].mean()) * risk_usd, 0) if len(losses) else 0.0,
        "avg_mfe_r":      round(avg_mfe, 3),
        "avg_mae_r":      round(avg_mae, 3),
        "win_capture_pct": capture,
    }


def _per_pair_breakdown(trades: List[Dict[str, Any]], r_col: str,
                        risk_usd: float) -> List[Dict[str, Any]]:
    filled = [t for t in trades if _is_real_filled(t)]
    if not filled:
        return []
    df = pd.DataFrame(filled)
    out = []
    for pair, sub in df.groupby("pair"):
        wins = sub[sub[r_col] > 0]
        out.append({
            "pair":          pair,
            "trades":        int(len(sub)),
            "win_rate_pct":  round(len(wins) / len(sub) * 100, 1) if len(sub) else 0,
            "expectancy_r":  round(float(sub[r_col].mean()), 3),
            "total_pnl_usd": round(float(sub[r_col].sum()) * risk_usd, 2),
        })
    out.sort(key=lambda r: r["total_pnl_usd"], reverse=True)
    return out


def _per_session_breakdown(trades: List[Dict[str, Any]], r_col: str,
                           risk_usd: float) -> List[Dict[str, Any]]:
    filled = [t for t in trades if _is_real_filled(t)]
    if not filled:
        return []
    df = pd.DataFrame(filled)
    if "session" not in df.columns:
        return []
    out = []
    for sess, sub in df.groupby("session"):
        wins = sub[sub[r_col] > 0]
        out.append({
            "session":       sess,
            "trades":        int(len(sub)),
            "win_rate_pct":  round(len(wins) / len(sub) * 100, 1) if len(sub) else 0,
            "expectancy_r":  round(float(sub[r_col].mean()), 3),
            "total_pnl_usd": round(float(sub[r_col].sum()) * risk_usd, 2),
        })
    session_order = {"Asia": 0, "London": 1, "NY": 2, "Other": 3}
    out.sort(key=lambda r: session_order.get(r["session"], 9))
    return out


def _fill_rate(trades: List[Dict[str, Any]], entry_zone: str) -> Dict[str, Any]:
    zone_rows = [t for t in trades if t.get("entry_zone") == entry_zone]
    if not zone_rows:
        return {"alerts": 0, "filled": 0, "fill_rate_pct": 0.0}
    filled = [t for t in zone_rows if _is_real_filled(t)]
    return {
        "alerts":        len(zone_rows),
        "filled":        len(filled),
        "fill_rate_pct": round(len(filled) / len(zone_rows) * 100, 1) if zone_rows else 0.0,
    }


def _score_buckets(trades: List[Dict[str, Any]], r_col: str) -> List[Dict[str, Any]]:
    filled = [t for t in trades if _is_real_filled(t)]
    if not filled:
        return []
    df = pd.DataFrame(filled)
    if "score" not in df.columns or r_col not in df.columns:
        return []
    edges  = [0, 2, 3, 4, 5, 6, 7, 99]
    labels = ["0-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7+"]
    df["bucket"] = pd.cut(df["score"], bins=edges, labels=labels,
                          right=False, include_lowest=True)
    out = []
    for label in labels:
        sub = df[df["bucket"] == label]
        if sub.empty:
            continue
        wins = sub[sub[r_col] > 0]
        out.append({
            "score_bucket":   label,
            "trades":         int(len(sub)),
            "win_rate_pct":   round(len(wins) / len(sub) * 100, 1) if len(sub) else 0,
            "expectancy_r":   round(float(sub[r_col].mean()), 3),
        })
    return out


def _exit_reason_counts(trades: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        counts[r] = counts.get(r, 0) + 1
    return counts


def _flag_vet_review(t: Dict[str, Any]) -> Tuple[bool, str]:
    r    = t.get("r_realised", 0)
    mfe  = t.get("mfe_r", 0)
    score = t.get("score", 0)
    if r > 0 and mfe > r * 2 and mfe > 1.5:
        return True, f"Left money on table — price reached {mfe:.1f}R, you booked {r:.1f}R"
    if r < 0 and mfe > 0.5:
        return True, f"Nearly worked — price reached +{mfe:.1f}R before reversing to SL"
    if score >= 4 and r < 0:
        return True, f"High-score setup ({score:.1f}) that lost — worth understanding why"
    return False, ""


# ---------------------------------------------------------------------------
# Sample-size discipline + exit-policy math
# ---------------------------------------------------------------------------

# Below this trade count, comparative verdicts are demoted to
# "insufficient sample" rather than asserted. Reader still sees the numbers.
MIN_N_FOR_VERDICT = 5


def _runner_r(t: Dict[str, Any]) -> float:
    """R under TP1+runner policy: 50% of position closes at TP1 when TP1 hit,
    50% rides to whatever r_realised did (SL is moved to entry after TP1,
    so the runner can BE-stop, win to TP2, or end at window close).

    When TP1 never hits, the full position is treated under default policy
    (so the runner branch == r_realised).
    """
    if t.get("exit_reason") == "never_filled":
        return 0.0
    r_realised = float(t.get("r_realised") or 0.0)
    bars_to_tp1 = t.get("bars_to_tp1")
    tp1_rr = float(t.get("tp1_rr") or 0.0)
    if bars_to_tp1 is not None and bars_to_tp1 >= 0:
        return 0.5 * tp1_rr + 0.5 * r_realised
    return r_realised


def _attach_runner_r(trades: List[Dict[str, Any]]) -> None:
    """Mutate trades in place to add `r_if_runner` (TP1+runner policy)."""
    for t in trades:
        if "r_if_runner" not in t:
            t["r_if_runner"] = round(_runner_r(t), 3)


def _exit_policy_table(trades: List[Dict[str, Any]],
                       risk_usd: float) -> List[Dict[str, Any]]:
    """Aggregate the three exit policies side by side: TP1-only, TP1+runner,
    TP2-ride (current). Aggregates use `_aggregate_for_exit` for consistency
    with the per-zone scoreboards.
    """
    _attach_runner_r(trades)
    out = []
    for label, col in [
        ("TP1-only",     "r_if_exit_tp1"),
        ("TP1 + runner", "r_if_runner"),
        ("TP2-ride (current)", "r_realised"),
    ]:
        sb = _aggregate_for_exit(trades, col, risk_usd)
        out.append({"policy": label, "col": col, **sb})
    return out


def _best_policy_by_dim(trades: List[Dict[str, Any]],
                        dim: str,
                        risk_usd: float) -> List[Dict[str, Any]]:
    """For each value of `dim` (pair or session), find which exit policy
    produced the highest total R. Returns rows sorted by trade count desc.
    """
    _attach_runner_r(trades)
    filled = [t for t in trades if _is_real_filled(t)]
    if not filled:
        return []
    df = pd.DataFrame(filled)
    if dim not in df.columns:
        return []
    out = []
    for val, sub in df.groupby(dim):
        n = int(len(sub))
        totals = {
            "TP1-only":            float(sub["r_if_exit_tp1"].sum()),
            "TP1 + runner":        float(sub["r_if_runner"].sum()),
            "TP2-ride (current)":  float(sub["r_realised"].sum()),
        }
        best_label = max(totals, key=totals.get)
        out.append({
            dim:        val,
            "trades":   n,
            "tp1_r":    round(totals["TP1-only"], 2),
            "runner_r": round(totals["TP1 + runner"], 2),
            "tp2_r":    round(totals["TP2-ride (current)"], 2),
            "best":     best_label,
            "edge_r":   round(totals[best_label] - totals["TP2-ride (current)"], 2),
        })
    out.sort(key=lambda r: r["trades"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Findings panel — auto-derived one-liners that lead each zone block.
# Output is a list of (severity, text) pairs. Severity drives the bullet color.
# Severities: "good" (green), "warn" (amber), "bad" (red), "info" (grey).
# ---------------------------------------------------------------------------

def _findings_panel(trades: List[Dict[str, Any]],
                    sb: Dict[str, Any],
                    risk_usd: float) -> List[Tuple[str, str]]:
    findings: List[Tuple[str, str]] = []
    filled = [t for t in trades if _is_real_filled(t)]
    n = len(filled)
    if n == 0:
        return [("info", "No filled trades this period.")]
    df = pd.DataFrame(filled)

    # 1. Headline result.
    pnl = sb.get("total_pnl_usd", 0)
    exp = sb.get("expectancy_r", 0)
    wr  = sb.get("win_rate_pct", 0)
    if pnl >= 0 and exp >= 0.3:
        findings.append(("good", f"Net <b>{_m(pnl)}</b> across {n} trades "
                                 f"(avg {_r(exp)}, {wr:.0f}% WR). Edge present."))
    elif pnl >= 0:
        findings.append(("info", f"Net <b>{_m(pnl)}</b> across {n} trades "
                                 f"(avg {_r(exp)}, {wr:.0f}% WR). Marginal — verify before sizing up."))
    else:
        findings.append(("bad", f"Net <b>{_m(pnl)}</b> across {n} trades "
                                f"(avg {_r(exp)}, {wr:.0f}% WR). Edge missing this period."))

    # 2. Best exit policy.
    _attach_runner_r(trades)
    pol = {
        "TP1-only":           float(df["r_if_exit_tp1"].sum()),
        "TP1 + runner":       float(df["r_if_runner"].sum()),
        "TP2-ride (current)": float(df["r_realised"].sum()),
    }
    best_pol = max(pol, key=pol.get)
    delta = pol[best_pol] - pol["TP2-ride (current)"]
    if best_pol != "TP2-ride (current)" and delta >= 1.0:
        findings.append(("warn",
            f"Best exit policy was <b>{best_pol}</b> "
            f"({pol[best_pol]:+.1f}R vs current TP2-ride {pol['TP2-ride (current)']:+.1f}R, "
            f"delta {delta:+.1f}R). Worth simulating live."))
    else:
        findings.append(("info",
            f"Current TP2-ride policy is best of the three "
            f"(TP1-only {pol['TP1-only']:+.1f}R · runner {pol['TP1 + runner']:+.1f}R · "
            f"TP2 {pol['TP2-ride (current)']:+.1f}R)."))

    # 3. Best and worst (pair, session) — only show with n >= 3 in the cell.
    if "pair" in df.columns and "session" in df.columns:
        cells = []
        for (p, s), sub in df.groupby(["pair", "session"]):
            if len(sub) >= 3:
                cells.append((p, s, len(sub), float(sub["r_realised"].mean()),
                              (sub["r_realised"] > 0).sum() / len(sub) * 100))
        if cells:
            cells.sort(key=lambda x: x[3], reverse=True)
            p, s, nn, e, w = cells[0]
            if e > 0.3:
                findings.append(("good",
                    f"Best edge: <b>{p} / {s}</b> ({nn} trades, "
                    f"{w:.0f}% WR, avg {_r(e)})."))
            p, s, nn, e, w = cells[-1]
            if e < -0.3 and len(cells) > 1:
                findings.append(("bad",
                    f"Weakest cell: <b>{p} / {s}</b> ({nn} trades, "
                    f"{w:.0f}% WR, avg {_r(e)}). Consider gating off."))

    # 4. MFE leak: losers that ran significantly in your favour first.
    losers = df[df["r_realised"] < 0]
    if len(losers) >= 3:
        leakers = losers[losers["mfe_r"] >= 1.0]
        if len(leakers) >= max(2, int(0.3 * len(losers))):
            findings.append(("warn",
                f"<b>{len(leakers)} of {len(losers)}</b> losers ran ≥+1.0R before reversing "
                f"(avg peak {leakers['mfe_r'].mean():.1f}R). Trailing stop study warranted."))

    # 5. Win capture — too tight vs too loose.
    cap = sb.get("win_capture_pct", 0)
    avg_mfe = sb.get("avg_mfe_r", 0)
    if sb.get("wins", 0) >= MIN_N_FOR_VERDICT and avg_mfe > 0:
        if cap < 60:
            findings.append(("warn",
                f"Winners captured only {cap:.0f}% of peak (avg peak {_r(avg_mfe)}) — "
                f"TP placement is leaving runner R on the table."))
        elif cap >= 85:
            findings.append(("info",
                f"Winners captured {cap:.0f}% of peak — TP placement is tight to peak; "
                f"little headroom left."))

    return findings


def _findings_panel_html(findings: List[Tuple[str, str]]) -> str:
    color_map = {
        "good": ("#27ae60", "✓"),
        "warn": ("#f39c12", "!"),
        "bad":  ("#e74c3c", "✗"),
        "info": ("#6c757d", "•"),
    }
    items = []
    for sev, text in findings:
        col, sym = color_map.get(sev, color_map["info"])
        items.append(
            f"<li style='list-style:none;padding:6px 0 6px 18px;"
            f"border-left:3px solid {col};margin-bottom:4px;'>"
            f"<span style='color:{col};font-weight:700;margin-right:6px;'>{sym}</span>"
            f"{text}</li>"
        )
    return ("<ul style='padding-left:0;margin:0;font-size:13px;line-height:1.5;'>"
            + "".join(items) + "</ul>")


# ---------------------------------------------------------------------------
# Exit-policy comparison: three policies + per-pair + per-session.
# ---------------------------------------------------------------------------

def _exit_policy_html(trades: List[Dict[str, Any]], risk_usd: float) -> str:
    rows_data = _exit_policy_table(trades, risk_usd)
    if not rows_data or rows_data[0].get("trades", 0) == 0:
        return "<p style='color:#888;'>No filled trades to compare policies on.</p>"

    n = rows_data[0]["trades"]
    header = _table_row(
        ["Exit policy", "Win rate", "Avg R", "Total R", "Total P&L"],
        header=True,
    )
    body = ""
    for r in rows_data:
        is_current = r["policy"].startswith("TP2-ride")
        pnl = r.get("total_pnl_usd", 0)
        color = "#eafaf1" if pnl >= 0 else "#fdf2f2"
        label = (f"<b>{r['policy']}</b>" if not is_current
                 else f"<b>{r['policy']}</b>")
        body += _table_row([
            label,
            f"{r.get('win_rate_pct', 0):.0f}%",
            _r(r.get("expectancy_r", 0)),
            f"{r.get('total_r', 0):+.1f}R",
            f"<b style='color:{'#27ae60' if pnl >= 0 else '#e74c3c'};'>{_m(pnl)}</b>",
        ], color=color)

    note = (f"<p style='font-size:12px;color:#666;margin-top:6px;'>"
            f"All three policies use the same fills ({n} trades). "
            f"TP1-only: close full position at TP1. "
            f"TP1+runner: half off at TP1, rest rides with SL-to-BE. "
            f"TP2-ride: full position rides to TP2 (current policy).</p>")
    return f"<table><thead>{header}</thead><tbody>{body}</tbody></table>{note}"


def _exit_policy_by_dim_html(trades: List[Dict[str, Any]],
                             risk_usd: float, dim: str, dim_label: str) -> str:
    rows_data = _best_policy_by_dim(trades, dim, risk_usd)
    if not rows_data:
        return ""
    header = _table_row(
        [dim_label, "Trades", "TP1-only", "TP1+runner", "TP2-ride", "Best", "Edge vs TP2"],
        header=True,
    )
    body = ""
    for r in rows_data:
        edge_color = "#27ae60" if r["edge_r"] > 0 else "#888"
        body += _table_row([
            f"<b>{r[dim]}</b>",
            str(r["trades"]),
            f"{r['tp1_r']:+.1f}R",
            f"{r['runner_r']:+.1f}R",
            f"{r['tp2_r']:+.1f}R",
            f"<b>{r['best']}</b>",
            f"<span style='color:{edge_color};'>{r['edge_r']:+.1f}R</span>",
        ])
    return f"<table><thead>{header}</thead><tbody>{body}</tbody></table>"


# ---------------------------------------------------------------------------
# Where the edge leaked — replaces the old "losing trades" bullet block.
# Same data as before plus MFE-leak, OB-age distribution, time-in-trade.
# ---------------------------------------------------------------------------

def _edge_leak_html(trades: List[Dict[str, Any]]) -> str:
    filled = [t for t in trades if _is_real_filled(t)]
    losers = [t for t in filled if (t.get("r_realised") or 0) < 0]
    winners = [t for t in filled if (t.get("r_realised") or 0) > 0]
    n_loss = len(losers)
    if n_loss == 0:
        return "<p style='color:#27ae60;'>No losing trades this period.</p>"

    findings: List[str] = []

    # 1. MFE-leak among losers.
    leak_1r = sum(1 for t in losers if (t.get("mfe_r") or 0) >= 1.0)
    leak_2r = sum(1 for t in losers if (t.get("mfe_r") or 0) >= 2.0)
    if leak_1r >= 2:
        avg_peak = sum((t.get("mfe_r") or 0) for t in losers
                       if (t.get("mfe_r") or 0) >= 1.0) / leak_1r
        findings.append(
            f"<b>{leak_1r} of {n_loss}</b> losers ran ≥+1.0R before reversing "
            f"(avg peak {avg_peak:.1f}R; {leak_2r} reached ≥+2.0R). "
            f"Trailing stop or partial-at-1R study warranted."
        )

    # 2. Tight TP1 — already-shipped finding, keep.
    tight_tp = sum(1 for t in losers if 1.5 <= (t.get("tp1_rr") or 0) < 2.0)
    if tight_tp >= 2:
        findings.append(
            f"<b>{tight_tp} of {n_loss}</b> losers had TP1 between 1.5R and 2.0R "
            f"(barely cleared the minimum gate). Opposing-swing lookback may be "
            f"finding nearby wicks instead of real liquidity."
        )

    # 3. OB age distribution.
    ages = [(t.get("ob_age_h1_bars") or 0) for t in losers]
    if ages:
        old = sum(1 for a in ages if a > 48)
        fresh = sum(1 for a in ages if a <= 12)
        if old >= max(2, int(0.3 * n_loss)):
            findings.append(
                f"<b>{old} of {n_loss}</b> losers used an OB older than 48 H1 bars (2 days). "
                f"Old OBs are more likely to be mitigated. Consider a freshness cutoff."
            )
        elif fresh >= max(2, int(0.5 * n_loss)) and fresh > old:
            findings.append(
                f"<b>{fresh} of {n_loss}</b> losers used a fresh OB (≤12 bars old). "
                f"Freshness alone isn't a guarantee — other confluences should carry the setup."
            )

    # 4. Time-in-trade for winners vs losers.
    win_hrs = [(t.get("bars_to_exit") or 0) for t in winners if (t.get("bars_to_exit") or 0) > 0]
    loss_hrs = [(t.get("bars_to_exit") or 0) for t in losers if (t.get("bars_to_exit") or 0) > 0]
    if len(win_hrs) >= MIN_N_FOR_VERDICT and len(loss_hrs) >= MIN_N_FOR_VERDICT:
        avg_w = sum(win_hrs) / len(win_hrs)
        avg_l = sum(loss_hrs) / len(loss_hrs)
        if avg_l < avg_w * 0.6:
            findings.append(
                f"Losers stopped out fast (avg {avg_l:.0f}h held vs {avg_w:.0f}h for winners). "
                f"Fast losses suggest the bias was wrong from entry — not late management failures."
            )
        elif avg_l > avg_w * 1.5:
            findings.append(
                f"Losers held long (avg {avg_l:.0f}h vs {avg_w:.0f}h for winners). "
                f"Slow losses suggest TP1 is too far — trade idea was right but target was unreachable."
            )

    # 5. CHoCH — kept, but only flagged when meaningful share.
    choch = sum(1 for t in losers if str(t.get("bos_tag") or "").upper() == "CHOCH")
    if choch >= 2 and choch / n_loss >= 0.25:
        findings.append(
            f"<b>{choch} of {n_loss}</b> losers were CHoCH setups (counter-trend). "
            f"CHoCH carries higher failure risk than BOS."
        )

    # 6. Minor structure — kept, threshold tightened.
    minor = sum(1 for t in losers if str(t.get("bos_tier") or "").lower() == "minor")
    if minor >= 2 and minor / n_loss >= 0.5:
        findings.append(
            f"<b>{minor} of {n_loss}</b> losers were on Minor structure events. "
            f"If this repeats across runs, consider restricting entries to Major events only."
        )

    if not findings:
        return (f"<p>{n_loss} losing trades this period. No dominant loss pattern detected — "
                f"losses appear distributed normally across conditions.</p>")

    items = "".join(f"<li style='margin-bottom:8px;'>{f}</li>" for f in findings)
    return f"<ul style='padding-left:18px;font-size:13px;line-height:1.6;'>{items}</ul>"


# ---------------------------------------------------------------------------
# By-pair and by-session flat tables (replaces the sparse 2D grid)
# ---------------------------------------------------------------------------

_SESSION_ORDER = ["Asia", "London", "NY", "Other"]


def _flat_breakdown_row(label: str, sub: pd.DataFrame, r_col: str) -> str:
    """One row in a flat breakdown table. Empty groups still appear (they
    happened in the data) but with neutral styling. Sample size is shown as
    a trade count -- the reader judges weight, no automatic fading."""
    n = len(sub)
    if n == 0:
        return (f"<tr style='color:#bbb;'>"
                f"<td><b>{label}</b></td><td>0</td><td>—</td><td>—</td></tr>")
    wins = (sub[r_col] > 0).sum()
    wr   = wins / n * 100
    exp  = sub[r_col].mean()
    if wr >= 50:
        bg = "#eafaf1"
    elif wr >= 40:
        bg = "#fef9e7"
    else:
        bg = "#fdf2f2"
    sign = "+" if exp >= 0 else ""
    return (f"<tr style='background:{bg};'>"
            f"<td><b>{label}</b></td>"
            f"<td>{n}</td>"
            f"<td>{wr:.0f}%</td>"
            f"<td>{sign}{exp:.2f}R</td></tr>")


def _by_pair_html(trades: List[Dict[str, Any]], r_col: str) -> str:
    """Win rate / avg R / trade count, by pair (all sessions combined)."""
    filled = [t for t in trades if _is_real_filled(t)]
    if not filled:
        return "<p style='color:#888;'>No filled trades.</p>"
    df = pd.DataFrame(filled)
    if "pair" not in df.columns or r_col not in df.columns:
        return "<p style='color:#888;'>Pair data missing.</p>"

    pairs = sorted(df["pair"].unique())
    header = "<tr><th>Pair</th><th>Trades</th><th>Win rate</th><th>Avg R</th></tr>"
    rows = "".join(_flat_breakdown_row(p, df[df["pair"] == p], r_col)
                   for p in pairs)
    return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>"


def _by_session_html(trades: List[Dict[str, Any]], r_col: str) -> str:
    """Win rate / avg R / trade count, by trading session (all pairs combined)."""
    filled = [t for t in trades if _is_real_filled(t)]
    if not filled:
        return "<p style='color:#888;'>No filled trades.</p>"
    df = pd.DataFrame(filled)
    if "session" not in df.columns or r_col not in df.columns:
        return "<p style='color:#888;'>Session data missing.</p>"

    header = "<tr><th>Session</th><th>Trades</th><th>Win rate</th><th>Avg R</th></tr>"
    rows = "".join(_flat_breakdown_row(s, df[df["session"] == s], r_col)
                   for s in _SESSION_ORDER if s in df["session"].unique())
    return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>"


def _pair_session_matrix_html(trades: List[Dict[str, Any]], r_col: str) -> str:
    """Pair x Session cross-tab. Pairs are rows, sessions are columns.
    Each cell shows trade count on top, WR and avg R below.

    Color encodes WR (green >=50, amber 40-50, red <40). Every non-empty cell
    is shown -- no n-threshold suppression. The user reads the sample size
    from the cell's trade count and makes the call themselves.
    Right-most column and bottom row show pair-totals and session-totals.
    """
    filled = [t for t in trades if _is_real_filled(t)]
    if not filled:
        return "<p style='color:#888;'>No filled trades.</p>"
    df = pd.DataFrame(filled)
    if "pair" not in df.columns or "session" not in df.columns \
            or r_col not in df.columns:
        return "<p style='color:#888;'>Pair/session data missing.</p>"

    pairs = sorted(df["pair"].unique())
    sessions = [s for s in _SESSION_ORDER if s in df["session"].unique()]
    if not sessions:
        return "<p style='color:#888;'>No session data.</p>"

    def _cell(sub: pd.DataFrame) -> str:
        n = len(sub)
        if n == 0:
            return ("<td style='text-align:center;color:#ccc;font-size:11px;'>"
                    "&mdash;</td>")
        wins = (sub[r_col] > 0).sum()
        wr = wins / n * 100
        exp = sub[r_col].mean()
        if wr >= 50:
            bg = "#eafaf1"
        elif wr >= 40:
            bg = "#fef9e7"
        else:
            bg = "#fdf2f2"
        sign = "+" if exp >= 0 else ""
        return (f"<td style='background:{bg};text-align:center;'>"
                f"<div style='font-size:11px;color:#888;'>{n}t</div>"
                f"<div style='font-weight:600;'>{wr:.0f}%</div>"
                f"<div style='font-size:11px;'>{sign}{exp:.2f}R</div>"
                f"</td>")

    # Header row.
    header = "<tr><th>Pair</th>"
    for s in sessions:
        header += f"<th style='text-align:center;'>{s}</th>"
    header += "<th style='text-align:center;background:#34495e;'>All</th></tr>"

    rows_html = ""
    for pair in pairs:
        pair_df = df[df["pair"] == pair]
        row = f"<tr><td><b>{pair}</b></td>"
        for s in sessions:
            row += _cell(pair_df[pair_df["session"] == s])
        row += _cell(pair_df).replace("background:#eafaf1",
                                       "background:#eafaf1;border-left:2px solid #34495e") \
                              .replace("background:#fef9e7",
                                       "background:#fef9e7;border-left:2px solid #34495e") \
                              .replace("background:#fdf2f2",
                                       "background:#fdf2f2;border-left:2px solid #34495e")
        row += "</tr>"
        rows_html += row

    # Bottom totals row -- session totals across all pairs.
    totals_row = "<tr><td style='background:#34495e;color:#fff;'><b>All</b></td>"
    for s in sessions:
        sess_df = df[df["session"] == s]
        cell = _cell(sess_df)
        # Mark the totals row with a darker top border.
        cell = cell.replace("<td style='",
                            "<td style='border-top:2px solid #34495e;")
        totals_row += cell
    grand = _cell(df).replace("<td style='",
                              "<td style='border-top:2px solid #34495e;"
                              "border-left:2px solid #34495e;")
    totals_row += grand + "</tr>"

    return (f"<table><thead>{header}</thead>"
            f"<tbody>{rows_html}{totals_row}</tbody></table>"
            f"<p style='font-size:11px;color:#888;margin-top:6px;'>"
            f"Each cell: trade count, win rate, average R. "
            f"Right column and bottom row are roll-ups.</p>")


_DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _pair_dow_matrix_html(trades: List[Dict[str, Any]], r_col: str) -> str:
    """Pair x Day-of-Week cross-tab. Same format as the session matrix.

    Lets the user spot per-pair day effects (e.g. EURUSD weak on Friday,
    GOLD strong Monday). Every non-empty cell is shown -- no n suppression.
    """
    filled = [t for t in trades if _is_real_filled(t)]
    if not filled:
        return "<p style='color:#888;'>No filled trades.</p>"
    df = pd.DataFrame(filled)
    if "pair" not in df.columns or "alert_ts" not in df.columns \
            or r_col not in df.columns:
        return "<p style='color:#888;'>Pair/alert_ts data missing.</p>"

    df["_dow"] = df["alert_ts"].apply(_day_of_week)
    pairs = sorted(df["pair"].unique())
    days_present = [d for d in _DOW_ORDER if d in df["_dow"].unique()]
    if not days_present:
        return "<p style='color:#888;'>No day-of-week data.</p>"

    def _cell(sub: pd.DataFrame) -> str:
        n = len(sub)
        if n == 0:
            return ("<td style='text-align:center;color:#ccc;font-size:11px;'>"
                    "&mdash;</td>")
        wins = (sub[r_col] > 0).sum()
        wr = wins / n * 100
        exp = sub[r_col].mean()
        if wr >= 50:
            bg = "#eafaf1"
        elif wr >= 40:
            bg = "#fef9e7"
        else:
            bg = "#fdf2f2"
        sign = "+" if exp >= 0 else ""
        return (f"<td style='background:{bg};text-align:center;'>"
                f"<div style='font-size:11px;color:#888;'>{n}t</div>"
                f"<div style='font-weight:600;'>{wr:.0f}%</div>"
                f"<div style='font-size:11px;'>{sign}{exp:.2f}R</div>"
                f"</td>")

    header = "<tr><th>Pair</th>"
    for d in days_present:
        header += f"<th style='text-align:center;'>{d}</th>"
    header += "<th style='text-align:center;background:#34495e;'>All</th></tr>"

    rows_html = ""
    for pair in pairs:
        pair_df = df[df["pair"] == pair]
        row = f"<tr><td><b>{pair}</b></td>"
        for d in days_present:
            row += _cell(pair_df[pair_df["_dow"] == d])
        row += _cell(pair_df).replace("background:#eafaf1",
                                       "background:#eafaf1;border-left:2px solid #34495e") \
                              .replace("background:#fef9e7",
                                       "background:#fef9e7;border-left:2px solid #34495e") \
                              .replace("background:#fdf2f2",
                                       "background:#fdf2f2;border-left:2px solid #34495e")
        row += "</tr>"
        rows_html += row

    totals_row = "<tr><td style='background:#34495e;color:#fff;'><b>All</b></td>"
    for d in days_present:
        day_df = df[df["_dow"] == d]
        cell = _cell(day_df)
        cell = cell.replace("<td style='",
                            "<td style='border-top:2px solid #34495e;")
        totals_row += cell
    grand = _cell(df).replace("<td style='",
                              "<td style='border-top:2px solid #34495e;"
                              "border-left:2px solid #34495e;")
    totals_row += grand + "</tr>"

    return (f"<table><thead>{header}</thead>"
            f"<tbody>{rows_html}{totals_row}</tbody></table>"
            f"<p style='font-size:11px;color:#888;margin-top:6px;'>"
            f"Each cell: trade count, win rate, average R. "
            f"Use this to spot per-pair day-of-week effects.</p>")


def _killzone_audit_html(kz_blocked_trades: List[Dict[str, Any]],
                         meta: Dict[str, Any]) -> str:
    """Audit section for the per-pair killzone gate.

    Killzone-blocked alerts are now SIMULATED for audit (same pattern as IST
    and news) but excluded from every aggregate metric above. This lets the
    user see the per-pair would-have R: positive = killzone filter is costing
    you trades; negative = killzone filter is saving you R.

    Block shows:
      - per-pair count + would-have R + WR
      - the configured killzone windows per pair (from meta)
      - a per-UTC-hour distribution so the user can see if the misses cluster
        in a particular hour (and consider extending the window)
    """
    windows = meta.get("killzone_windows_by_pair") or {}
    drops_meta = meta.get("killzone_drops_by_pair") or {}
    total_drops = int(meta.get("killzone_dropped_alerts") or 0)

    # Filter to one row per alert (proximal) to avoid double-counting.
    rows = [t for t in kz_blocked_trades
            if t.get("entry_zone") == "proximal"]
    n = len(rows)

    header = (
        "<p style='font-size:12px;color:#666;margin-bottom:8px;'>"
        "Alerts whose timestamp fell outside the pair's killzone window. "
        "These rows are <b>excluded from every aggregate metric above</b>; "
        "they are simulated only so we can show what would have happened "
        "if the filter were off. Positive would-have R = filter is costing "
        "you trades; negative = filter is saving you R."
        "</p>"
    )

    if n == 0 and not windows:
        return header + ("<p style='color:#888;'>"
                         "No killzone filter active for this run.</p>")
    if n == 0:
        # Filter was configured but produced no out-of-window alerts.
        pair_names_sorted = sorted(windows.keys())
        win_rows = ""
        for pair in pair_names_sorted:
            w = windows.get(pair, "no killzone configured")
            d = int(drops_meta.get(pair, 0))
            win_rows += (
                f"<tr><td><b>{pair}</b></td>"
                f"<td style='font-family:monospace;font-size:12px;'>{w}</td>"
                f"<td style='text-align:right;'>{d}</td></tr>"
            )
        return header + (
            f"<p style='color:#27ae60;'>No alerts fell outside the killzone "
            f"this run across {len(pair_names_sorted)} configured pair(s).</p>"
            f"<table><thead>"
            f"<tr><th>Pair</th><th>Killzone window(s) UTC</th><th>Dropped</th></tr>"
            f"</thead><tbody>{win_rows}</tbody></table>"
        )

    df = pd.DataFrame(rows)

    # 1. By pair: alerts + filled + WR + would-have R + the configured window.
    by_pair_rows = ""
    if "pair" in df.columns and "r_realised" in df.columns:
        for pair, sub in df.groupby("pair"):
            n_alerts = len(sub)
            filled_sub = sub[~sub["exit_reason"].isin(["never_filled","window_end","timeout"])]
            n_filled = len(filled_sub)
            total_r = float(filled_sub["r_realised"].sum()) if n_filled else 0.0
            wins = int((filled_sub["r_realised"] > 0).sum()) if n_filled else 0
            wr = (wins / n_filled * 100) if n_filled else None
            sign = "+" if total_r >= 0 else ""
            wr_str = f"{wr:.0f}%" if wr is not None else "&mdash;"
            r_color = "#27ae60" if total_r > 0 else ("#e74c3c" if total_r < 0 else "#888")
            w = windows.get(pair, "&mdash;")
            verdict = ("filter cost R" if total_r > 0.5
                       else "filter saved R" if total_r < -0.5
                       else "neutral")
            verdict_color = ("#e74c3c" if total_r > 0.5
                             else "#27ae60" if total_r < -0.5
                             else "#888")
            by_pair_rows += (
                f"<tr><td><b>{pair}</b></td>"
                f"<td style='font-family:monospace;font-size:11px;'>{w}</td>"
                f"<td>{n_alerts}</td>"
                f"<td>{n_filled}</td>"
                f"<td>{wr_str}</td>"
                f"<td style='color:{r_color};'>{sign}{total_r:.2f}R</td>"
                f"<td style='color:{verdict_color};font-size:12px;'>{verdict}</td>"
                f"</tr>"
            )

    by_pair_table = ""
    if by_pair_rows:
        by_pair_table = (
            f"<h4>Per-pair would-have R</h4>"
            f"<table><thead>"
            f"<tr><th>Pair</th><th>Killzone window(s) UTC</th>"
            f"<th>Alerts</th><th>Filled</th><th>WR</th>"
            f"<th>Would-have R</th><th>Verdict</th></tr>"
            f"</thead><tbody>{by_pair_rows}</tbody></table>"
            f"<p style='font-size:11px;color:#888;margin-top:6px;'>"
            f"Verdict thresholds: |R| ≥ 0.5R = filter has effect; otherwise "
            f"neutral. Repeat across multiple runs before drawing conclusions.</p>"
        )

    # 2. By UTC hour: helps spot whether the misses cluster in one hour
    # (e.g. London-open extension might recover edge), the same way the
    # IST gate's hour table does.
    by_hour_rows = ""
    if "alert_utc_hour" in df.columns and "r_realised" in df.columns:
        grouped = df.groupby("alert_utc_hour")
        hour_data = []
        for hour, sub in grouped:
            n_alerts = len(sub)
            filled_sub = sub[~sub["exit_reason"].isin(["never_filled","window_end","timeout"])]
            n_filled = len(filled_sub)
            total_r = float(filled_sub["r_realised"].sum()) if n_filled else 0.0
            wins = int((filled_sub["r_realised"] > 0).sum()) if n_filled else 0
            wr = (wins / n_filled * 100) if n_filled else None
            hour_data.append((int(hour), n_alerts, n_filled, wins, wr, total_r))
        hour_data.sort()
        for hour, n_alerts, n_filled, wins, wr, total_r in hour_data:
            sign = "+" if total_r >= 0 else ""
            wr_str = f"{wr:.0f}%" if wr is not None else "&mdash;"
            r_color = "#27ae60" if total_r > 0 else ("#e74c3c" if total_r < 0 else "#888")
            by_hour_rows += (
                f"<tr><td>{hour:02d}:00 UTC</td>"
                f"<td>{n_alerts}</td>"
                f"<td>{n_filled}</td>"
                f"<td>{wr_str}</td>"
                f"<td style='color:{r_color};'>{sign}{total_r:.2f}R</td></tr>"
            )

    by_hour_table = ""
    if by_hour_rows:
        by_hour_table = (
            f"<h4>Out-of-window alerts by UTC hour</h4>"
            f"<table><thead>"
            f"<tr><th>UTC hour</th><th>Alerts</th><th>Filled</th>"
            f"<th>WR</th><th>Would-have R</th></tr>"
            f"</thead><tbody>{by_hour_rows}</tbody></table>"
        )

    return (
        header
        + f"<p style='font-size:13px;margin-bottom:8px;'>"
          f"<b>{n} alert(s)</b> outside the configured killzone "
          f"(proximal rows; each alert may have a paired 50% row too). "
          f"meta-counter records {total_drops} dropped.</p>"
        + by_pair_table + by_hour_table
    )


def _ist_blackout_html(ist_blocked_trades: List[Dict[str, Any]],
                       meta: Dict[str, Any]) -> str:
    """Audit section for alerts filtered by the IST trading-window gate.

    Shows: count of dropped alerts, distribution by pair and UTC hour, and
    the R outcomes those alerts *would* have produced if traded. This is
    the data the user needs to decide whether to shift sleep hours.
    """
    # Filter to one row per alert (proximal trade is canonical -- it always
    # gets a row, 50pct sometimes does not). Avoids double-counting alerts.
    rows = [t for t in ist_blocked_trades
            if t.get("entry_zone") == "proximal"]
    n = len(rows)

    forex_window = meta.get("ist_window_forex", "UTC 03:30-18:30")
    index_window = meta.get("ist_window_index", "UTC 13:00-20:00")

    header = (
        f"<p style='font-size:12px;color:#666;margin-bottom:8px;'>"
        f"Alerts whose timestamp fell outside the user's IST trading window. "
        f"Live system suppresses these -- backtest mirrors that. These rows "
        f"are <b>excluded from every aggregate metric above</b>. They are "
        f"shown here so you can decide whether shifting your sleep schedule "
        f"would unlock real edge.</p>"
        f"<p style='font-size:11px;color:#888;margin-bottom:10px;'>"
        f"Window: forex/commodity {forex_window} &middot; index {index_window}."
        f"</p>"
    )

    if n == 0:
        return header + (
            "<p style='color:#27ae60;'>"
            "No alerts fell outside the IST window this run."
            "</p>"
        )

    df = pd.DataFrame(rows)

    # 1. By UTC hour: count of alerts + would-have R sum.
    by_hour_rows = ""
    if "alert_utc_hour" in df.columns and "r_realised" in df.columns:
        grouped = df.groupby("alert_utc_hour")
        hour_data = []
        for hour, sub in grouped:
            n_alerts = len(sub)
            filled_sub = sub[~sub["exit_reason"].isin(["never_filled","window_end","timeout"])]
            n_filled = len(filled_sub)
            total_r = float(filled_sub["r_realised"].sum()) if n_filled else 0.0
            wins = int((filled_sub["r_realised"] > 0).sum()) if n_filled else 0
            wr = (wins / n_filled * 100) if n_filled else None
            hour_data.append((int(hour), n_alerts, n_filled, wins, wr, total_r))
        hour_data.sort()
        for hour, n_alerts, n_filled, wins, wr, total_r in hour_data:
            sign = "+" if total_r >= 0 else ""
            wr_str = f"{wr:.0f}%" if wr is not None else "&mdash;"
            ist_hour = (hour + 5) % 24  # rough IST (+5:30 truncated to +5)
            ist_minute = 30
            r_color = "#27ae60" if total_r > 0 else ("#e74c3c" if total_r < 0 else "#888")
            by_hour_rows += (
                f"<tr><td>{hour:02d}:00 UTC</td>"
                f"<td style='color:#888;'>~{ist_hour:02d}:{ist_minute:02d} IST</td>"
                f"<td>{n_alerts}</td>"
                f"<td>{n_filled}</td>"
                f"<td>{wr_str}</td>"
                f"<td style='color:{r_color};'>{sign}{total_r:.2f}R</td></tr>"
            )

    by_hour_table = ""
    if by_hour_rows:
        by_hour_table = (
            f"<h4>Alerts by UTC hour (with hypothetical outcome)</h4>"
            f"<table><thead>"
            f"<tr><th>UTC hour</th><th>IST (approx)</th>"
            f"<th>Alerts</th><th>Filled</th><th>WR</th>"
            f"<th>Would-have R</th></tr>"
            f"</thead><tbody>{by_hour_rows}</tbody></table>"
            f"<p style='font-size:11px;color:#888;margin-top:6px;'>"
            f"Would-have R = sum of r_realised across these dropped alerts "
            f"if you had been awake to take them. "
            f"Positive = sleeping cost you R; negative = blackout saved you R.</p>"
        )

    # 2. By pair: same view, sliced differently.
    by_pair_rows = ""
    if "pair" in df.columns and "r_realised" in df.columns:
        for pair, sub in df.groupby("pair"):
            n_alerts = len(sub)
            filled_sub = sub[~sub["exit_reason"].isin(["never_filled","window_end","timeout"])]
            n_filled = len(filled_sub)
            total_r = float(filled_sub["r_realised"].sum()) if n_filled else 0.0
            wins = int((filled_sub["r_realised"] > 0).sum()) if n_filled else 0
            wr = (wins / n_filled * 100) if n_filled else None
            sign = "+" if total_r >= 0 else ""
            wr_str = f"{wr:.0f}%" if wr is not None else "&mdash;"
            r_color = "#27ae60" if total_r > 0 else ("#e74c3c" if total_r < 0 else "#888")
            by_pair_rows += (
                f"<tr><td><b>{pair}</b></td>"
                f"<td>{n_alerts}</td>"
                f"<td>{n_filled}</td>"
                f"<td>{wr_str}</td>"
                f"<td style='color:{r_color};'>{sign}{total_r:.2f}R</td></tr>"
            )

    by_pair_table = ""
    if by_pair_rows:
        by_pair_table = (
            f"<h4>Alerts by pair</h4>"
            f"<table><thead>"
            f"<tr><th>Pair</th><th>Alerts</th><th>Filled</th>"
            f"<th>WR</th><th>Would-have R</th></tr>"
            f"</thead><tbody>{by_pair_rows}</tbody></table>"
        )

    return (
        header
        + f"<p style='font-size:13px;margin-bottom:8px;'>"
          f"<b>{n} alert(s)</b> dropped by IST gate "
          f"(proximal rows; each alert may have a paired 50% row too).</p>"
        + by_hour_table + by_pair_table
    )


# ---------------------------------------------------------------------------
# Confluence per pair
# ---------------------------------------------------------------------------

_CONFLUENCE_COLS = {
    "FVG":          ("fvg_present",   lambda s: s == True),
    "Sweep":        ("sweep_present", lambda s: s == True),
    "Kill zone":    ("killzone_pts",  lambda s: s > 0),
    "Freshness":    ("freshness_pts", lambda s: s > 0),
    "Structure":    ("structure_pts", lambda s: s > 0),
}

def _confluence_per_pair_html(trades: List[Dict[str, Any]], r_col: str) -> str:
    filled = [t for t in trades if _is_real_filled(t)]
    if not filled:
        return "<p style='color:#888;'>No data.</p>"
    df = pd.DataFrame(filled)
    if "pair" not in df.columns or r_col not in df.columns:
        return "<p style='color:#888;'>Data missing.</p>"

    pairs  = sorted(df["pair"].unique())
    confs  = list(_CONFLUENCE_COLS.keys())

    def _mask(df: pd.DataFrame, name: str) -> pd.Series:
        col, fn = _CONFLUENCE_COLS[name]
        if col not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        try:
            return fn(df[col])
        except Exception:
            return pd.Series([False] * len(df), index=df.index)

    header = "<tr><th>Pair</th>" + "".join(f"<th>{c}</th>" for c in confs) + "</tr>"
    rows   = ""
    for pair in pairs:
        sub = df[df["pair"] == pair]
        row = f"<tr><td><b>{pair}</b></td>"
        for c in confs:
            mask    = _mask(sub, c)
            with_c  = sub[mask]
            wout_c  = sub[~mask]
            if len(with_c) == 0:
                # No trades had this confluence -- there's nothing to display.
                # This is different from "few trades": no data at all.
                row += "<td style='color:#bbb;text-align:center;'>—</td>"
                continue
            wr_with  = (with_c[r_col] > 0).mean() * 100
            wr_wout  = (wout_c[r_col] > 0).mean() * 100 if len(wout_c) > 0 else None
            uplift   = (wr_with - wr_wout) if wr_wout is not None else None
            bg = "#eafaf1" if (uplift is not None and uplift > 5) else \
                 "#fdf2f2" if (uplift is not None and uplift < -5) else "#f9f9f9"
            arrow = " ↑" if (uplift and uplift > 5) else " ↓" if (uplift and uplift < -5) else ""
            row += (f"<td style='background:{bg};text-align:center;'>"
                    f"{wr_with:.0f}%{arrow}<br>"
                    f"<small>{len(with_c)}t</small></td>")
        row += "</tr>"
        rows += row

    note = ("<p style='font-size:11px;color:#888;margin-top:6px;'>"
            "Win rate when that confluence was present. ↑ = helped vs without it, "
            "↓ = hurt. — = the confluence was never present for this pair. "
            "Small number under each cell is the trade count — read sample "
            "size from there, no minimum threshold applied.</p>")
    return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>{note}"


# ---------------------------------------------------------------------------
# News blackout report section
# ---------------------------------------------------------------------------

def _news_blackout_html(blocked_trades: List[Dict[str, Any]],
                        meta: Dict[str, Any]) -> str:
    """Renders the news-blackout audit section: how many rows were dropped,
    which events caused each, coverage of the upstream feeds.

    These trades do NOT appear in any aggregate metric above this section --
    they are listed here only so the user can verify the filter is acting
    correctly. Every entry shows the source event so blocks are auditable.
    """
    n = len(blocked_trades)
    coverage = meta.get("news_coverage", {}) or {}
    coverage_str = ", ".join(
        f"{src}: {'ok' if ok else 'PARTIAL — fetch failed'}"
        for src, ok in coverage.items()
    ) if coverage else "not run"
    n_events = meta.get("news_events_fetched", 0)
    window   = meta.get("news_window_minutes", 30)

    header = (
        f"<p style='font-size:12px;color:#666;margin-bottom:8px;'>"
        f"Trades whose alert timestamp fell within &plusmn;{window} min of a "
        f"High-impact news event for any currency in the pair. These rows "
        f"are <b>excluded from every aggregate metric above</b> (P&amp;L, WR, "
        f"RR, expectancy, by-pair, by-session, killzone, structure, score). "
        f"They appear in <code>trades.xlsx</code> with column <code>news_blocked=True</code> "
        f"for audit only."
        f"</p>"
        f"<p style='font-size:11px;color:#888;margin-bottom:10px;'>"
        f"Feed coverage: {coverage_str}. Events fetched: {n_events}."
        f"</p>"
    )

    if n == 0:
        return header + (
            "<p style='color:#27ae60;'>"
            "No trades were filtered this week."
            "</p>"
        )

    # Deduplicate by (pair, alert_ts, event_ts) so we show one entry per
    # alert (proximal+50pct share the same news block).
    seen = set()
    unique_rows = []
    for t in blocked_trades:
        key = (t.get("pair"), t.get("alert_ts"), t.get("news_event_ts"))
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(t)

    rows_html = ""
    for t in unique_rows:
        rows_html += (
            f"<tr>"
            f"<td>{t.get('pair', '?')}</td>"
            f"<td>{t.get('alert_ts', '?')}</td>"
            f"<td>{t.get('news_event_currency', '?')}</td>"
            f"<td>{t.get('news_event_title', '?')}</td>"
            f"<td>{t.get('news_event_source', '?')}</td>"
            f"<td style='font-family:monospace;font-size:11px;'>{t.get('news_event_ts', '?')}</td>"
            f"</tr>"
        )

    return header + (
        f"<p style='font-size:13px;margin-bottom:8px;'>"
        f"<b>{n} trade row(s) filtered</b> across {len(unique_rows)} unique alert(s)."
        f"</p>"
        f"<table><thead>"
        f"<tr><th>Pair</th><th>Alert ts (UTC)</th><th>Ccy</th>"
        f"<th>Event</th><th>Source</th><th>Event ts (UTC)</th></tr>"
        f"</thead><tbody>{rows_html}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Loss pattern analysis
# ---------------------------------------------------------------------------

def _loss_analysis_html(trades: List[Dict[str, Any]]) -> str:
    # By the time this function runs, news-blocked and IST-blocked rows have
    # already been stripped upstream in write_h1_only_report. So every trade
    # we see here is one we actually counted. The bullets below describe
    # conditions of the *counted* losses; they must not re-introduce a "news
    # window" bullet, because no row that reaches this function is news-flagged.
    losses = [t for t in trades
              if _is_real_filled(t) and t.get("r_realised", 0) < 0]
    if not losses:
        return "<p style='color:#27ae60;'>No losing trades this week.</p>"

    n = len(losses)
    findings = []

    # 2. Tight TP (barely cleared the 1.5R minimum gate)
    tight_tp = sum(1 for t in losses if 1.5 <= (t.get("tp1_rr") or 0) < 2.0)
    if tight_tp > 0:
        findings.append(
            f"<b>{tight_tp} of {n}</b> losing trades had TP1 between 1.5R and 2.0R — "
            f"barely cleared the minimum gate. The opposing swing used as TP1 may have been "
            f"too close (lookback=3 finding nearby wicks rather than real liquidity). "
            f"These trades needed a significant move but the target wasn't far enough."
        )

    # 3. Old OB
    old_ob = sum(1 for t in losses if (t.get("ob_age_h1_bars") or 0) > 48)
    if old_ob > 0:
        findings.append(
            f"<b>{old_ob} of {n}</b> losing trades used an OB older than 48 H1 bars (2 days). "
            f"Old OBs are more likely to be mitigated or no longer holding institutional interest. "
            f"Consider a freshness cutoff."
        )

    # 4. (Removed) Outside-killzone bullet. With the per-pair killzone hard
    # filter in place, every counted loss is already in-killzone -- this
    # bullet would either always read "0 of N" or contradict the audit
    # block by mixing the old pair_type window with the new per-pair gate.

    # 5. CHoCH (counter-trend risk)
    choch = sum(1 for t in losses if t.get("bos_tag", "").upper() == "CHOCH")
    if choch > 0:
        findings.append(
            f"<b>{choch} of {n}</b> losing trades were CHoCH setups. CHoCH means the system "
            f"traded against the prior trend. These carry higher failure risk than BOS setups."
        )

    # 6. Minor structure (smaller swing — less institutional weight)
    minor = sum(1 for t in losses if str(t.get("bos_tier", "")).lower() == "minor")
    if minor > 0 and minor / n >= 0.4:
        findings.append(
            f"<b>{minor} of {n}</b> losing trades were on Minor structure events. "
            f"Minor events are smaller swings detected at a lower lookback and carry "
            f"less institutional commitment than Major events. If this pattern repeats "
            f"across runs, consider restricting entries to Major events only."
        )

    if not findings:
        return (f"<p>{n} losing trades this week. No dominant loss pattern detected — "
                f"losses appear to be distributed normally across conditions.</p>")

    items = "".join(f"<li style='margin-bottom:8px;'>{f}</li>" for f in findings)
    return f"<ul style='padding-left:18px;font-size:13px;line-height:1.6;'>{items}</ul>"


# ---------------------------------------------------------------------------
# Structure event performance breakdown (Major vs Minor, BOS vs CHoCH)
# ---------------------------------------------------------------------------

def _structure_event_breakdown_html(trades: List[Dict[str, Any]], r_col: str) -> str:
    """Show trades grouped by structure event type. Helps verify Major vs Minor
    detection reliability before live trading."""
    filled = [t for t in trades if _is_real_filled(t)]
    if not filled:
        return "<p style='color:#888;'>No filled trades.</p>"
    df = pd.DataFrame(filled)
    if "bos_tier" not in df.columns or "bos_tag" not in df.columns or r_col not in df.columns:
        return "<p style='color:#888;'>Structure event data missing.</p>"

    rows = "<tr><th>Event Type</th><th>Trades</th><th>Win rate</th><th>Avg R</th><th>Total P&L</th></tr>"
    seen = []
    for (tier, tag), sub in df.groupby(["bos_tier", "bos_tag"]):
        n = len(sub)
        wins = (sub[r_col] > 0).sum()
        wr = wins / n * 100 if n else 0
        exp = float(sub[r_col].mean()) if n else 0
        total = float(sub[r_col].sum()) * 250  # default 1R = $250 for display
        bg = "#eafaf1" if exp >= 0 else "#fdf2f2"
        seen.append((tier, tag, n, wr, exp))
        rows += (f"<tr style='background:{bg};'>"
                 f"<td><b>{tier} {tag}</b></td>"
                 f"<td>{n}</td>"
                 f"<td>{wr:.0f}%</td>"
                 f"<td>{_r(exp)}</td>"
                 f"<td>{_m(total)}</td></tr>")

    # Headline verdict comparing Major vs Minor.
    major_n = sum(n for tier, _, n, _, _ in seen if tier == "Major")
    minor_n = sum(n for tier, _, n, _, _ in seen if tier == "Minor")
    major_exp = (sum(n * e for tier, _, n, _, e in seen if tier == "Major") / major_n) if major_n else None
    minor_exp = (sum(n * e for tier, _, n, _, e in seen if tier == "Minor") / minor_n) if minor_n else None

    # Sample-size discipline: a verdict is only meaningful when BOTH tiers
    # have a non-trivial sample. Below the threshold, the diff can flip
    # period to period (and was flipping between proximal/50% in the same
    # email). Suppress the verdict in that case rather than asserting it.
    note = ""
    if major_exp is not None and minor_exp is not None:
        diff = major_exp - minor_exp
        if major_n < MIN_N_FOR_VERDICT or minor_n < MIN_N_FOR_VERDICT:
            note = (f"<p style='font-size:12px;color:#888;margin-top:8px;'>"
                    f"Sample too small for a Major-vs-Minor verdict "
                    f"(Major: {major_n}t, Minor: {minor_n}t; need ≥{MIN_N_FOR_VERDICT} each). "
                    f"Diff this period: {diff:+.2f}R — re-check across runs.</p>")
        elif diff > 0.15:
            note = (f"<p style='font-size:13px;color:#27ae60;margin-top:8px;'>"
                    f"✓ Major events outperformed Minor by {diff:.2f}R per trade. "
                    f"This is the expected behaviour — Major detection is working as designed.</p>")
        elif diff < -0.15:
            note = (f"<p style='font-size:13px;color:#e74c3c;margin-top:8px;'>"
                    f"⚠ Minor events outperformed Major by {abs(diff):.2f}R — unexpected. "
                    f"Major event detection may be misclassifying setups. Verify before trading live.</p>")
        else:
            note = (f"<p style='font-size:13px;color:#888;margin-top:8px;'>"
                    f"Major and Minor performance is similar (difference: {diff:+.2f}R).</p>")
    elif major_n == 0 and minor_n == 0:
        note = ("<p style='font-size:12px;color:#888;margin-top:8px;'>"
                "No structure-event data this period.</p>")
    else:
        note = (f"<p style='font-size:12px;color:#888;margin-top:8px;'>"
                f"Only one tier present this period "
                f"(Major: {major_n}, Minor: {minor_n}). No comparison possible.</p>")

    legend = ("<p style='font-size:11px;color:#aaa;margin-top:4px;'>"
              "<b>BOS</b> = Break of Structure (trend continuation). "
              "<b>CHoCH</b> = Change of Character (trend reversal). "
              "<b>Major</b> = larger swing, more institutional weight. "
              "<b>Minor</b> = smaller internal swing, less weight.</p>")
    return f"<table>{rows}</table>{note}{legend}"


# ---------------------------------------------------------------------------
# Trade validation
# ---------------------------------------------------------------------------

def _validate_trades(trades: List[Dict[str, Any]]) -> List[str]:
    """Check structural invariants. Returns a list of violation strings (empty = clean)."""
    violations = []
    filled = [t for t in trades if _is_real_filled(t)]
    for t in filled:
        pair      = t.get("pair", "?")
        ts        = t.get("alert_ts", "?")
        direction = t.get("direction", "")
        entry     = t.get("entry")
        sl        = t.get("sl_initial")
        tp1       = t.get("tp1")
        tp2       = t.get("tp2")
        r         = t.get("r_realised", 0)
        reason    = t.get("exit_reason", "")

        if entry is None or sl is None or tp1 is None:
            violations.append(f"{pair} @ {ts}: missing entry/SL/TP1")
            continue

        if direction == "bullish":
            if sl >= entry:
                violations.append(f"{pair} @ {ts}: SL ({sl}) ≥ entry ({entry}) for LONG")
            if tp1 <= entry:
                violations.append(f"{pair} @ {ts}: TP1 ({tp1}) ≤ entry ({entry}) for LONG")
            if tp2 is not None and tp2 <= tp1:
                violations.append(f"{pair} @ {ts}: TP2 ({tp2}) ≤ TP1 ({tp1}) for LONG")
        elif direction == "bearish":
            if sl <= entry:
                violations.append(f"{pair} @ {ts}: SL ({sl}) ≤ entry ({entry}) for SHORT")
            if tp1 >= entry:
                violations.append(f"{pair} @ {ts}: TP1 ({tp1}) ≥ entry ({entry}) for SHORT")
            if tp2 is not None and tp2 >= tp1:
                violations.append(f"{pair} @ {ts}: TP2 ({tp2}) ≥ TP1 ({tp1}) for SHORT")

        # TP/SL exit should have correct sign
        if reason == "sl" and r > 0:
            violations.append(f"{pair} @ {ts}: exit=SL but r_realised={r:.2f} (positive — unexpected)")
        if reason in ("tp1", "tp2") and r < 0:
            violations.append(f"{pair} @ {ts}: exit={reason} but r_realised={r:.2f} (negative — unexpected)")

    return violations


def _validation_html(trades: List[Dict[str, Any]]) -> str:
    violations = _validate_trades(trades)
    if not violations:
        # Validation runs over BOTH entry zones (it's a sim-integrity check),
        # so the count here is rows across proximal + 50pct, NOT the headline
        # filled-trade count (proximal only). Make that explicit so the number
        # doesn't appear to contradict the headline.
        n_prox = len([t for t in trades
                      if _is_real_filled(t)
                      and t.get("entry_zone") == "proximal"])
        n_mid  = len([t for t in trades
                      if _is_real_filled(t)
                      and t.get("entry_zone") == "50pct"])
        return (f"<p style='color:#27ae60;'>✓ All filled trade rows passed validation "
                f"({n_prox} proximal + {n_mid} 50pct) — "
                f"entry/SL/TP levels are correctly ordered, and exit outcomes match exit reasons.</p>")
    items = "".join(f"<li style='color:#e74c3c;font-family:monospace;font-size:12px;'>{v}</li>"
                    for v in violations)
    return (f"<p style='color:#e74c3c;'><b>⚠ {len(violations)} validation issue(s) found:</b></p>"
            f"<ul>{items}</ul>"
            f"<p style='font-size:12px;color:#888;'>These indicate calculation errors in the simulator. "
            f"Do not draw trading conclusions until they are resolved.</p>")


# ---------------------------------------------------------------------------
# Zone register (for Excel tab)
# ---------------------------------------------------------------------------

def _build_zone_register_df(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per OB-touch event, showing both entry zone outcomes side by side."""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    # Key = unique OB touch event: pair + alert_ts (each alert produces 2 rows)
    if "alert_ts" not in df.columns or "pair" not in df.columns:
        return pd.DataFrame()

    _exit_labels = {
        "sl": "Stop Loss Hit", "tp1": "TP1 Hit", "tp2": "TP2 Hit",
        "timeout": "Time Limit (48h)", "window_end": "End of Window",
        "sl_collision": "SL+TP Same Bar", "never_filled": "Never Filled",
    }

    rows = []
    for (pair, alert_ts), grp in df.groupby(["pair", "alert_ts"]):
        prox = grp[grp["entry_zone"] == "proximal"].iloc[0].to_dict() if not grp[grp["entry_zone"] == "proximal"].empty else {}
        mid  = grp[grp["entry_zone"] == "50pct"].iloc[0].to_dict()   if not grp[grp["entry_zone"] == "50pct"].empty else {}

        def _v(d, k, default=""):
            return d.get(k, default)

        # News flag is set at the alert level, so prox and 50pct always
        # share the same value. Take from prox first; fall back to mid.
        news_blocked = bool(_v(prox, "news_blocked") or _v(mid, "news_blocked"))
        news_event_title    = _v(prox, "news_event_title")    or _v(mid, "news_event_title")
        news_event_currency = _v(prox, "news_event_currency") or _v(mid, "news_event_currency")
        news_event_source   = _v(prox, "news_event_source")   or _v(mid, "news_event_source")
        news_event_ts       = _v(prox, "news_event_ts")       or _v(mid, "news_event_ts")
        # Killzone flag is alert-level; prox and 50pct share the same value.
        kz_blocked = bool(_v(prox, "killzone_blocked") or _v(mid, "killzone_blocked"))
        ist_blocked_zr = bool(_v(prox, "ist_blocked") or _v(mid, "ist_blocked"))

        rows.append({
            "Pair":                    pair,
            "OB Formed (UTC)":         _v(prox, "ob_timestamp"),
            "Alert Time (UTC)":        alert_ts,
            "Direction":               "Long" if _v(prox, "direction") == "bullish" else "Short",
            "Structure Event":         _v(prox, "bos_tag"),
            "Structure Tier":          _v(prox, "bos_tier"),
            "OB Age (H1 bars)":        _v(prox, "ob_age_h1_bars"),
            "Setup Score":             _v(prox, "score"),
            "Trading Session":         _v(prox, "session"),
            "Day of Week":             _day_of_week(alert_ts),
            "Entry Price (Proximal)":  _v(prox, "entry"),
            "Entry Price (50% Mid)":   _v(mid,  "entry"),
            "Stop Loss":               _v(prox, "sl_initial"),
            "Take Profit 1":           _v(prox, "tp1"),
            "Take Profit 2":           _v(prox, "tp2"),
            "TP1 Reward:Risk":         _v(prox, "tp1_rr"),
            "TP2 Reward:Risk":         _v(prox, "tp2_rr"),
            "Proximal Filled?":        "Yes" if _v(prox, "exit_reason") != "never_filled" and prox else "No",
            "Proximal Outcome":        _exit_labels.get(_v(prox, "exit_reason"), _v(prox, "exit_reason")),
            "Proximal R":              _v(prox, "r_realised"),
            "Proximal Dollar P&L":     round(float(_v(prox, "pnl_usd", 0) or 0), 0) if prox else "",
            "50% Filled?":             "Yes" if _v(mid, "exit_reason") != "never_filled" and mid else "No",
            "50% Outcome":             _exit_labels.get(_v(mid, "exit_reason"), _v(mid, "exit_reason")),
            "50% R":                   _v(mid, "r_realised"),
            "50% Dollar P&L":          round(float(_v(mid, "pnl_usd", 0) or 0), 0) if mid else "",
            "FVG Present":             "Yes" if _v(prox, "fvg_present") else "No",
            "Sweep Present":           "Yes" if _v(prox, "sweep_present") else "No",
            "Confluences Active":      _v(prox, "confluences_present"),
            # News blackout audit columns. If "Yes" in News Blocked, this
            # row was excluded from every aggregate metric in the email.
            "News Blocked":            "Yes" if news_blocked else "No",
            "News Event":              news_event_title,
            "News Currency":           news_event_currency,
            "News Source":             news_event_source,
            "News Event Time (UTC)":   news_event_ts,
            "Killzone Blocked":        "Yes" if kz_blocked else "No",
            "IST Blocked":             "Yes" if ist_blocked_zr else "No",
        })

    return pd.DataFrame(rows)


def _day_of_week(ts_str: str) -> str:
    try:
        return pd.Timestamp(ts_str).day_name()[:3]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# CSV (machine-readable, column names unchanged — used by aggregate_runs.py)
# ---------------------------------------------------------------------------

def _trades_csv(trades: List[Dict[str, Any]], path: Path) -> None:
    if not trades:
        pd.DataFrame([{"info": "no trades this run"}]).to_csv(path, index=False)
        return
    front_cols = [
        "pair", "alert_ts", "fill_ts", "exit_ts", "session",
        "direction", "event", "entry_zone",
        "entry", "sl_initial", "tp1", "tp2", "tp1_rr", "tp2_rr",
        "exit_reason", "exit_price",
        "r_realised", "r_if_exit_tp1", "r_if_exit_tp2", "pnl_usd",
        "mfe_r", "mae_r", "bars_to_exit", "bars_to_tp1", "bars_to_tp2",
        "ob_age_h1_bars", "pd_zone",
        "score", "structure_pts", "sweep_pts", "fvg_pts",
        "freshness_pts", "killzone_pts", "confluences_present",
        "sl_collision", "model", "ob_timestamp", "bos_tag", "bos_tier",
        "fvg_present", "sweep_present",
        # News blackout audit columns. news_blocked=True means this row
        # was excluded from every aggregate metric in summary.json.
        "news_blocked", "news_event_title", "news_event_currency",
        "news_event_source", "news_event_ts",
        # IST blackout audit columns. ist_blocked=True means this alert
        # fell outside the user's IST trading window and was excluded
        # from aggregates (live system would have suppressed it).
        "ist_blocked", "alert_utc_hour",
    ]
    df = pd.DataFrame(trades)
    cols_present = [c for c in front_cols if c in df.columns]
    rest = [c for c in df.columns if c not in cols_present]
    df[cols_present + rest].to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Excel (human-readable — plain English column names, formatted)
# ---------------------------------------------------------------------------

_EXCEL_COL_NAMES = {
    "pair":              "Currency Pair",
    "direction":         "Direction",
    "session":           "Trading Session",
    "entry_zone":        "Entry Type",
    "entry":             "Entry Price",
    "sl_initial":        "Stop Loss",
    "tp1":               "Take Profit 1",
    "tp2":               "Take Profit 2",
    "tp1_rr":            "TP1 Reward:Risk",
    "tp2_rr":            "TP2 Reward:Risk",
    "exit_reason":       "How Trade Closed",
    "exit_price":        "Exit Price",
    "r_realised":        "R Achieved (TP2-ride)",
    "pnl_usd":           "Dollar P&L (TP2-ride)",
    "r_if_exit_tp1":     "R if Closed at TP1",
    "r_if_runner":       "R if TP1 + Runner",
    "mfe_r":             "Best Price Reached (R)",
    "mae_r":             "Worst Price Reached (R)",
    "bars_to_exit":      "Hours Held",
    "bars_to_tp1":       "Hours to TP1 (-1 if never)",
    "score":             "Setup Score (0–8)",
    "confluences_present": "Confluences Active",
    "fvg_present":       "FVG Present",
    "sweep_present":     "Liquidity Sweep Present",
    "bos_tag":           "Structure Event (BOS / CHoCH)",
    "bos_tier":          "Structure Tier (Major / Minor)",
    "vet_review":        "Worth Reviewing",
    "vet_review_reason": "Why Worth Reviewing",
    "alert_ts":          "Alert Time (UTC)",
    "exit_ts":           "Trade Closed (UTC)",
    "ob_time_ist":       "OB Formed (IST)",
    "fill_time_ist":     "Entry Fill (IST)",
    "sl_hit_time_ist":   "SL Hit (IST)",
    "tp_fill_time_ist":  "TP Fill (IST)",
    "news_blocked":         "News Blocked",
    "news_event_title":     "News Event",
    "news_event_currency":  "News Currency",
    "news_event_source":    "News Source",
    "news_event_ts":        "News Event Time (UTC)",
    "ist_blocked":          "IST Window Blocked",
    "killzone_blocked":     "Killzone Blocked",
    "killzone_windows":     "Killzone Window(s)",
    "alert_utc_hour":       "Alert Hour (UTC)",
}

_EXIT_LABELS = {
    "sl":           "Stop Loss Hit",
    "tp1":          "TP1 Hit",
    "tp2":          "TP2 Hit",
    "timeout":      "Time Limit Reached (48h)",
    "window_end":   "End of Test Window",
    "sl_collision": "SL and TP Same Bar — SL Taken",
    "never_filled": "Order Never Filled",
}

_ENTRY_LABELS = {
    "proximal": "Proximal (OB edge)",
    "50pct":    "50% Midpoint",
}

_DIR_LABELS = {
    "bullish": "Long",
    "bearish": "Short",
}


def _to_ist_str(ts_val: Any) -> str:
    """Convert a UTC ISO timestamp string (or pd.Timestamp) to IST 'YYYY-MM-DD HH:MM'.

    All stored timestamps in the backtest pipeline are UTC (data_loader pins
    every H1 frame to UTC and the simulator localises naïve timestamps to UTC
    before emitting). So a bare conversion to Asia/Kolkata is correct — no
    silent assumption about the source zone.
    """
    if ts_val is None or ts_val == "" or (isinstance(ts_val, float) and pd.isna(ts_val)):
        return ""
    try:
        ts = pd.Timestamp(ts_val)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _try_excel(trades: List[Dict[str, Any]], path: Path) -> Optional[Path]:
    """Write human-readable Excel. Returns path or None on failure."""
    # Only filled trades in the Excel — never_filled are counted in fill rate
    # but are not trade outcomes and would confuse the spreadsheet.
    filled = [t for t in trades if _is_real_filled(t)]
    if not filled:
        return None
    try:
        # Make sure r_if_runner is populated for every row so the new Excel
        # column has data. _attach_runner_r is idempotent.
        _attach_runner_r(filled)
        df = pd.DataFrame(filled)

        # Add day of week (tracked for cross-run aggregate analysis).
        if "alert_ts" in df.columns:
            df["day_of_week"] = df["alert_ts"].apply(_day_of_week)

        # IST timestamp columns (for TradingView verification). Source columns
        # are UTC ISO strings. SL/TP IST cells split exit_ts by exit_reason so
        # a row only carries the IST timestamp for the outcome that actually
        # happened — empty otherwise.
        df["ob_time_ist"]    = df["ob_timestamp"].apply(_to_ist_str) if "ob_timestamp" in df.columns else ""
        df["fill_time_ist"]  = df["fill_ts"].apply(_to_ist_str)      if "fill_ts"      in df.columns else ""

        def _sl_ist(row):
            return _to_ist_str(row.get("exit_ts")) if row.get("exit_reason") in ("sl", "sl_collision") else ""

        def _tp_ist(row):
            return _to_ist_str(row.get("exit_ts")) if row.get("exit_reason") in ("tp1", "tp2") else ""

        df["sl_hit_time_ist"] = df.apply(_sl_ist, axis=1) if "exit_ts" in df.columns else ""
        df["tp_fill_time_ist"] = df.apply(_tp_ist, axis=1) if "exit_ts" in df.columns else ""

        # Compute vet_review for each trade.
        reviews = [_flag_vet_review(t) for t in filled]
        df["vet_review"]        = ["Yes" if r[0] else "No" for r in reviews]
        df["vet_review_reason"] = [r[1] for r in reviews]

        # Human-friendly value mapping.
        if "direction" in df.columns:
            df["direction"] = df["direction"].map(_DIR_LABELS).fillna(df["direction"])
        if "entry_zone" in df.columns:
            df["entry_zone"] = df["entry_zone"].map(_ENTRY_LABELS).fillna(df["entry_zone"])
        if "exit_reason" in df.columns:
            df["exit_reason"] = df["exit_reason"].map(_EXIT_LABELS).fillna(df["exit_reason"])
        if "fvg_present" in df.columns:
            df["fvg_present"] = df["fvg_present"].map({True: "Yes", False: "No"}).fillna("")
        if "sweep_present" in df.columns:
            df["sweep_present"] = df["sweep_present"].map({True: "Yes", False: "No"}).fillna("")

        # Select and rename columns — add day_of_week right after session.
        _col_names_with_dow = dict(_EXCEL_COL_NAMES)
        _col_names_with_dow["day_of_week"] = "Day of Week"
        desired = [c for c in list(_EXCEL_COL_NAMES.keys())[:4]  # pair, direction, session
                   + ["day_of_week"]
                   + list(_EXCEL_COL_NAMES.keys())[4:]
                   if c in df.columns]
        out_df = df[desired].rename(columns=_col_names_with_dow)

        # Zone register (one row per OB — both entry zones side by side).
        zone_df = _build_zone_register_df(trades)

        # Split trades into proximal and 50% tabs.
        prox_df = out_df[out_df["Entry Type"] == _ENTRY_LABELS["proximal"]] if "Entry Type" in out_df.columns else out_df
        mid_df  = out_df[out_df["Entry Type"] == _ENTRY_LABELS["50pct"]]   if "Entry Type" in out_df.columns else pd.DataFrame()

        col_widths = {
            "Currency Pair": 14, "Direction": 10, "Trading Session": 12,
            "Entry Type": 18, "Entry Price": 12, "Stop Loss": 12,
            "Take Profit 1": 13, "Take Profit 2": 13,
            "TP1 Reward:Risk": 14, "TP2 Reward:Risk": 14,
            "How Trade Closed": 24, "Exit Price": 12,
            "R Achieved (TP2-ride)": 18, "Dollar P&L (TP2-ride)": 20,
            "R if Closed at TP1": 18, "R if TP1 + Runner": 18,
            "Proximal R": 12, "Proximal Dollar P&L": 18,
            "50% R": 12, "50% Dollar P&L": 18,
            "Best Price Reached (R)": 20, "Worst Price Reached (R)": 20,
            "Hours Held": 10, "Hours to TP1 (-1 if never)": 22,
            "Setup Score (0–8)": 14,
            "Confluences Active": 22,
            "FVG Present": 12, "Liquidity Sweep Present": 22,
            "Structure Event (BOS / CHoCH)": 24,
            "Structure Tier (Major / Minor)": 24,
            "Worth Reviewing": 15, "Why Worth Reviewing": 40,
            "Alert Time (UTC)": 20, "Trade Closed (UTC)": 20,
            "OB Formed (IST)": 18, "Entry Fill (IST)": 18,
            "SL Hit (IST)": 18, "TP Fill (IST)": 18,
        }

        def _style_trades_sheet(ws, from_openpyxl):
            PatternFill, Font, Alignment = (
                from_openpyxl["PatternFill"],
                from_openpyxl["Font"],
                from_openpyxl["Alignment"],
            )
            green_fill = PatternFill("solid", fgColor="C6EFCE")
            red_fill   = PatternFill("solid", fgColor="FFC7CE")
            grey_fill  = PatternFill("solid", fgColor="F2F2F2")

            for cell in ws[1]:
                cell.font      = Font(bold=True, color="FFFFFF")
                cell.fill      = PatternFill("solid", fgColor="2C3E50")
                cell.alignment = Alignment(wrap_text=True)

            headers = [cell.value for cell in ws[1]]
            pnl_cols = [headers.index(h) + 1 for h in
                        ("Dollar P&L (TP2-ride)", "Dollar P&L",
                         "Proximal Dollar P&L", "50% Dollar P&L")
                        if h in headers]
            rev_col = headers.index("Worth Reviewing") + 1 if "Worth Reviewing" in headers else None
            nb_col  = headers.index("News Blocked") + 1   if "News Blocked" in headers else None

            for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=2):
                base_fill = grey_fill if row_idx % 2 == 0 else None
                for cell in row:
                    if base_fill:
                        cell.fill = base_fill
                if pnl_cols:
                    net = sum(
                        ws.cell(row=row_idx, column=c).value or 0
                        for c in pnl_cols
                        if isinstance(ws.cell(row=row_idx, column=c).value, (int, float))
                    )
                    fill = green_fill if net > 0 else (red_fill if net < 0 else None)
                    if fill:
                        for cell in row:
                            cell.fill = fill
                if rev_col and ws.cell(row=row_idx, column=rev_col).value == "Yes":
                    for cell in row:
                        cell.fill = PatternFill("solid", fgColor="FFEB9C")

            if nb_col:
                news_fill = PatternFill("solid", fgColor="E8DAEF")
                for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=2):
                    if ws.cell(row=row_idx, column=nb_col).value in (True, "True", "Yes", 1, "1"):
                        for cell in row:
                            cell.fill = news_fill

            for i, col in enumerate(ws.columns, start=1):
                header = ws.cell(row=1, column=i).value
                ws.column_dimensions[col[0].column_letter].width = col_widths.get(header, 14)
            ws.freeze_panes = "A2"

        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            prox_df.to_excel(xw, sheet_name="Proximal", index=False)
            if not mid_df.empty:
                mid_df.to_excel(xw, sheet_name="50% Entry", index=False)
            if not zone_df.empty:
                zone_df.to_excel(xw, sheet_name="Zone Register", index=False)

            try:
                from openpyxl.styles import PatternFill, Font, Alignment
                _opx = {"PatternFill": PatternFill, "Font": Font, "Alignment": Alignment}

                _style_trades_sheet(xw.sheets["Proximal"], _opx)
                if "50% Entry" in xw.sheets:
                    _style_trades_sheet(xw.sheets["50% Entry"], _opx)

                if "Zone Register" in xw.sheets:
                    zws = xw.sheets["Zone Register"]
                    for cell in zws[1]:
                        cell.font = Font(bold=True, color="FFFFFF")
                        cell.fill = PatternFill("solid", fgColor="1A5490")
                    for col in zws.columns:
                        zws.column_dimensions[col[0].column_letter].width = 18
                    zws.freeze_panes = "A2"

            except ImportError:
                pass  # openpyxl styles not available — plain Excel still written

        return path
    except Exception as e:
        print(f"  [excel write skipped: {e}]")
        return None


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _m(amount: float) -> str:
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):,.0f}"


def _r(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}R"


def _week_headline(sb: Dict[str, Any]) -> str:
    n   = sb.get("trades", 0)
    exp = sb.get("expectancy_r", 0)
    wr  = sb.get("win_rate_pct", 0)
    if n == 0:
        return "No filled trades this period."
    # No n-threshold suppression. The headline is shown for any non-zero
    # sample size; the user reads the trade count from the strip below
    # and decides whether to trust the verdict.
    if exp >= 0.3 and wr >= 50:
        return "System performed well this period."
    if exp >= 0 and wr >= 45:
        return "Modest positive period — edge is there but modest."
    if exp >= 0 and wr < 45:
        return "Profitable but fewer wins than expected — check if losses are stopping cleanly."
    if exp < 0 and wr >= 50:
        return "Many wins but losses were too large — take profit levels may be too tight."
    return "Difficult period — below threshold on both win rate and expectancy."


def _exit_narrative(sb: Dict[str, Any], risk_usd: float) -> str:
    wins   = sb.get("wins", 0)
    losses = sb.get("losses", 0)
    bes    = sb.get("breakevens", 0)
    aw     = sb.get("avg_win_r", 0)
    al     = sb.get("avg_loss_r", 0)
    mfe    = sb.get("avg_mfe_r", 0)
    mae    = sb.get("avg_mae_r", 0)
    cap    = sb.get("win_capture_pct", 0)

    parts = []

    # Wins
    if wins > 0:
        parts.append(
            f"<b>{wins} {'trade' if wins == 1 else 'trades'} won</b>, averaging "
            f"<b>{_r(aw)}</b> ({_m(aw * risk_usd)}) each."
        )
        if mfe > 0:
            if cap >= 80:
                parts.append(
                    f"Winners ran to an average best of <b>{_r(mfe)}</b> before reversing. "
                    f"You captured <b>{cap:.0f}%</b> of that move — take profits are well-placed."
                )
            elif cap >= 60:
                parts.append(
                    f"Winners ran to an average best of <b>{_r(mfe)}</b> before reversing. "
                    f"You captured <b>{cap:.0f}%</b> — take profits slightly conservative."
                )
            else:
                parts.append(
                    f"Winners ran to an average best of <b>{_r(mfe)}</b> before reversing. "
                    f"You only captured <b>{cap:.0f}%</b> — take profits are too tight."
                )

    # Losses
    if losses > 0:
        clean = abs(mae) >= 0.75
        parts.append(
            f"<b>{losses} {'trade' if losses == 1 else 'trades'} lost</b>, averaging "
            f"<b>{_r(al)}</b> ({_m(al * risk_usd)}) each. "
            f"Before stopping out, price moved an average of <b>{_r(abs(mae))}</b> against the position. "
            + ("Stops hit cleanly — not getting wicked out." if clean
               else "Stops hit shallow — position was stopped out before the full adverse move. Stops may be too tight.")
        )

    # Breakevens
    if bes > 0:
        parts.append(f"<b>{bes} {'trade' if bes == 1 else 'trades'} broke even</b> (hit TP1, SL moved to entry, then reversed).")

    return " ".join(f"<p>{p}</p>" for p in parts) if parts else "<p>No filled trades this week.</p>"


def _table_row(cells: List[str], header: bool = False, color: str = "") -> str:
    tag = "th" if header else "td"
    style = f" style='background:{color};'" if color else ""
    return "<tr>" + "".join(f"<{tag}{style}>{c}</{tag}>" for c in cells) + "</tr>"


def _pair_section_html(pp: List[Dict], ss: List[Dict], r_col_label: str) -> str:
    if not pp:
        return "<p style='color:#888;'>No filled trades.</p>"

    # Pair table
    pair_rows = _table_row(["Pair", "Trades", "Win rate", "Expectancy", "Total P&L"], header=True)
    for row in pp:
        color = "#eafaf1" if row["total_pnl_usd"] >= 0 else "#fdf2f2"
        pair_rows += _table_row([
            f"<b>{row['pair']}</b>",
            str(row["trades"]),
            f"{row['win_rate_pct']:.0f}%",
            _r(row["expectancy_r"]),
            f"<b style='color:{'#27ae60' if row['total_pnl_usd'] >= 0 else '#e74c3c'};'>"
            f"{_m(row['total_pnl_usd'])}</b>",
        ], color=color)

    # Session table
    sess_rows = ""
    if ss:
        sess_rows = "<h4 style='margin-top:16px;color:#555;'>By session</h4>"
        sess_rows += "<table><thead>" + _table_row(["Session", "Trades", "Win rate", "Expectancy", "Total P&L"], header=True) + "</thead><tbody>"
        for row in ss:
            color = "#eafaf1" if row["total_pnl_usd"] >= 0 else "#fdf2f2"
            sess_rows += _table_row([
                row["session"],
                str(row["trades"]),
                f"{row['win_rate_pct']:.0f}%",
                _r(row["expectancy_r"]),
                f"<b style='color:{'#27ae60' if row['total_pnl_usd'] >= 0 else '#e74c3c'};'>"
                f"{_m(row['total_pnl_usd'])}</b>",
            ], color=color)
        sess_rows += "</tbody></table>"

    return f"<table><thead>{pair_rows}</thead><tbody>" + \
           "".join(_table_row([
               f"<b>{r['pair']}</b>", str(r["trades"]),
               f"{r['win_rate_pct']:.0f}%", _r(r["expectancy_r"]),
               f"<b style='color:{'#27ae60' if r['total_pnl_usd']>=0 else '#e74c3c'};'>{_m(r['total_pnl_usd'])}</b>",
           ], color="#eafaf1" if r["total_pnl_usd"] >= 0 else "#fdf2f2") for r in pp) + \
           "</tbody></table>" + sess_rows


def _score_verdict_text(buckets: List[Dict]) -> str:
    if not buckets:
        return "No score data this period."
    # Sample-size discipline: buckets with very small n distort the trend check.
    # Use only buckets with n >= 3 for the verdict, but keep all in the table.
    robust = [b for b in buckets if b["trades"] >= 3]
    if len(robust) < 2:
        total_n = sum(b["trades"] for b in buckets)
        return (f"Insufficient sample to call score-vs-outcome trend "
                f"({total_n} trades across {len(buckets)} buckets; "
                f"need ≥2 buckets with ≥3 trades each).")
    exp_vals = [b["expectancy_r"] for b in robust]
    rises = sum(1 for a, b in zip(exp_vals, exp_vals[1:]) if b > a)
    total = len(exp_vals) - 1
    ratio = rises / total if total > 0 else 0
    if ratio >= 0.7:
        return "✓ Yes — higher score setups consistently produced better outcomes this period."
    if ratio >= 0.4:
        return "~ Partial — some relationship, not consistent across all score levels."
    return "✗ No — score did not predict outcomes this period."


def _score_table_html(buckets: List[Dict]) -> str:
    if not buckets:
        return "<p style='color:#888;'>No score data.</p>"
    rows = _table_row(["Score", "Trades", "Win rate", "Avg R per trade"], header=True)
    for b in buckets:
        color = "#eafaf1" if b["expectancy_r"] >= 0 else "#fdf2f2"
        rows += _table_row([
            b["score_bucket"], str(b["trades"]),
            f"{b['win_rate_pct']:.0f}%", _r(b["expectancy_r"]),
        ], color=color)
    return f"<table><thead>{rows}</thead></table>"  # already has header in rows


def _entry_comparison_html(sb_prox: Dict, sb_mid: Dict,
                           fill_prox: Dict, fill_mid: Dict,
                           prox_trades: List[Dict[str, Any]],
                           mid_trades: List[Dict[str, Any]],
                           risk_usd: float) -> str:
    """Head-to-head of proximal vs 50% entry, broken out for each exit policy
    (TP1-only, TP1+runner, TP2-ride). Lets the reader see whether one entry
    zone dominates universally or only under a specific exit policy."""
    def _row(label: str, prox_val: str, mid_val: str) -> str:
        return f"<tr><td><b>{label}</b></td><td>{prox_val}</td><td>{mid_val}</td></tr>"

    # Per-policy aggregates.
    _attach_runner_r(prox_trades)
    _attach_runner_r(mid_trades)
    sb_prox_tp1 = _aggregate_for_exit(prox_trades, "r_if_exit_tp1", risk_usd)
    sb_prox_run = _aggregate_for_exit(prox_trades, "r_if_runner",   risk_usd)
    sb_mid_tp1  = _aggregate_for_exit(mid_trades,  "r_if_exit_tp1", risk_usd)
    sb_mid_run  = _aggregate_for_exit(mid_trades,  "r_if_runner",   risk_usd)

    rows = "".join([
        _table_row(["", "Proximal entry", "50% midpoint entry"], header=True),
        _row("Alerts triggered",
             str(fill_prox["alerts"]), str(fill_mid["alerts"])),
        _row("Orders filled",
             f"{fill_prox['filled']} ({fill_prox['fill_rate_pct']:.0f}%)",
             f"{fill_mid['filled']} ({fill_mid['fill_rate_pct']:.0f}%)"),
        _row("Win rate (TP2-ride)",
             f"{sb_prox.get('win_rate_pct', 0):.0f}%",
             f"{sb_mid.get('win_rate_pct', 0):.0f}%"),
        _row("Avg R (TP2-ride, current)",
             _r(sb_prox.get("expectancy_r", 0)),
             _r(sb_mid.get("expectancy_r", 0))),
        _row("Total P&L &mdash; TP1-only",
             _m(sb_prox_tp1.get("total_pnl_usd", 0)),
             _m(sb_mid_tp1.get("total_pnl_usd", 0))),
        _row("Total P&L &mdash; TP1 + runner",
             _m(sb_prox_run.get("total_pnl_usd", 0)),
             _m(sb_mid_run.get("total_pnl_usd", 0))),
        _row("Total P&L &mdash; TP2-ride (current)",
             f"<b>{_m(sb_prox.get('total_pnl_usd', 0))}</b>",
             f"<b>{_m(sb_mid.get('total_pnl_usd', 0))}</b>"),
    ])

    # Best (entry, policy) call-out.
    grid = {
        ("Proximal", "TP1-only"):           sb_prox_tp1.get("total_pnl_usd", 0),
        ("Proximal", "TP1+runner"):         sb_prox_run.get("total_pnl_usd", 0),
        ("Proximal", "TP2-ride (current)"): sb_prox.get("total_pnl_usd", 0),
        ("50% entry", "TP1-only"):           sb_mid_tp1.get("total_pnl_usd", 0),
        ("50% entry", "TP1+runner"):         sb_mid_run.get("total_pnl_usd", 0),
        ("50% entry", "TP2-ride (current)"): sb_mid.get("total_pnl_usd", 0),
    }
    best_combo = max(grid, key=grid.get)
    current = grid[("Proximal", "TP2-ride (current)")] + grid[("50% entry", "TP2-ride (current)")]
    best_pnl = grid[best_combo]

    if best_combo == ("Proximal", "TP2-ride (current)") or best_combo == ("50% entry", "TP2-ride (current)"):
        verdict = ("<p style='font-size:13px;color:#27ae60;margin-top:10px;'>"
                   f"<b>Current policy is best on the single dominant entry/exit pair.</b> "
                   f"Best single combo: {best_combo[0]} + {best_combo[1]} ({_m(best_pnl)}).</p>")
    else:
        verdict = ("<p style='font-size:13px;color:#f39c12;margin-top:10px;'>"
                   f"<b>Best single combo this period: {best_combo[0]} + {best_combo[1]} "
                   f"({_m(best_pnl)})</b> &mdash; "
                   f"vs current TP2-ride total {_m(current)}. Worth simulating live.</p>")

    note = ("<p style='font-size:12px;color:#666;margin-top:4px;'>"
            "All policies target the same TP1/TP2 levels. TP1-only closes at TP1; "
            "TP1+runner takes half at TP1 and rides the rest under a BE stop; "
            "TP2-ride is the current default (full position rides to TP2).</p>")
    return f"<table>{rows}</table>{verdict}{note}"


def _vet_review_html(trades: List[Dict]) -> str:
    flagged = [(t, *_flag_vet_review(t)) for t in trades
               if _is_real_filled(t)]
    flagged = [(t, flag, reason) for t, flag, reason in flagged if flag]
    if not flagged:
        return "<p>Nothing flagged this week. All wins and losses behaved as expected.</p>"

    rows = _table_row(["Pair", "Direction", "Session", "R Achieved", "What to look at"], header=True)
    for t, _, reason in flagged:
        direction = "Long" if t.get("direction") == "bullish" else "Short"
        rows += _table_row([
            t.get("pair", "?"), direction,
            t.get("session", "?"), _r(t.get("r_realised", 0)),
            reason,
        ])
    return f"<table><thead></thead><tbody>{rows}</tbody></table>"


# ---------------------------------------------------------------------------
# Per-entry-zone block: emits the full per-zone analysis stack.
# ---------------------------------------------------------------------------

def _zone_block_html(
    zone_label: str,
    zone_trades: List[Dict[str, Any]],
    sb_realised: Dict[str, Any],
    sb_tp1: Dict[str, Any],
    sb_tp2: Dict[str, Any],
    risk_usd: float,
    band_color: str,
) -> str:
    """One self-contained section per entry zone. New layout:
      findings panel -> compact header -> by-pair -> by-session ->
      pair x session matrix -> pair x DOW matrix -> exit-policy comparison ->
      score buckets -> confluences -> structure events ->
      where the edge leaked -> trades worth a second look.

    Every aggregate inside this block uses r_realised so the headline and
    the breakdowns reconcile to the same total. Exit-policy section uses
    r_if_exit_tp1, r_if_runner and r_realised side by side.
    """
    # Headline figures.
    n         = sb_realised.get("trades", 0)
    total_pnl = sb_realised.get("total_pnl_usd", 0)
    exp_r     = sb_realised.get("expectancy_r", 0)
    wr        = sb_realised.get("win_rate_pct", 0)
    wins      = sb_realised.get("wins", 0)
    losses    = sb_realised.get("losses", 0)
    bes       = sb_realised.get("breakevens", 0)
    pnl_color = "#27ae60" if total_pnl >= 0 else "#e74c3c"

    # Auto-derived findings — the lead.
    findings = _findings_panel(zone_trades, sb_realised, risk_usd)
    findings_block = _findings_panel_html(findings)

    # Score buckets and verdict (this zone's data only).
    score_buckets = _score_buckets(zone_trades, "r_realised")
    score_verdict = _score_verdict_text(score_buckets)

    return f"""
<!-- ZONE BAND -->
<div style="background:{band_color};color:#fff;padding:14px 28px;
            font-size:14px;font-weight:700;letter-spacing:0.04em;
            text-transform:uppercase;">
  {zone_label}
</div>

<!-- COMPACT HEADER + FINDINGS PANEL -->
<div class="section">
  <div style="display:flex;justify-content:space-between;align-items:baseline;
              flex-wrap:wrap;gap:14px;margin-bottom:14px;">
    <div style="font-size:24px;font-weight:700;color:{pnl_color};">
      {_m(total_pnl)}
    </div>
    <div style="font-size:13px;color:#666;">
      <b>{n}</b> filled &middot; <b>{wr:.0f}%</b> WR &middot; <b>{_r(exp_r)}</b> avg &middot;
      {wins}W / {losses}L / {bes}BE
    </div>
  </div>
  {findings_block}
</div>

<!-- BY PAIR -->
<div class="section">
  <h2>By pair</h2>
  {_by_pair_html(zone_trades, "r_realised")}
</div>

<!-- BY SESSION -->
<div class="section">
  <h2>By session</h2>
  {_by_session_html(zone_trades, "r_realised")}
</div>

<!-- PAIR x SESSION MATRIX -->
<div class="section">
  <h2>Pair &times; session</h2>
  {_pair_session_matrix_html(zone_trades, "r_realised")}
</div>

<!-- PAIR x DAY-OF-WEEK MATRIX -->
<div class="section">
  <h2>Pair &times; day of week</h2>
  {_pair_dow_matrix_html(zone_trades, "r_realised")}
</div>

<!-- EXIT POLICY COMPARISON -->
<div class="section">
  <h2>Exit policy comparison</h2>
  {_exit_policy_html(zone_trades, risk_usd)}
  <h4 style="margin-top:16px;">Best policy by pair</h4>
  {_exit_policy_by_dim_html(zone_trades, risk_usd, "pair", "Pair")}
  <h4 style="margin-top:16px;">Best policy by session</h4>
  {_exit_policy_by_dim_html(zone_trades, risk_usd, "session", "Session")}
</div>

<!-- SCORE -->
<div class="section">
  <h2>Confidence score &mdash; did it predict outcomes?</h2>
  <p><b>{score_verdict}</b></p>
  {_score_table_html(score_buckets)}
</div>

<!-- CONFLUENCES -->
<div class="section">
  <h2>Confluences by pair</h2>
  {_confluence_per_pair_html(zone_trades, "r_realised")}
</div>

<!-- STRUCTURE -->
<div class="section">
  <h2>Structure event performance</h2>
  {_structure_event_breakdown_html(zone_trades, "r_realised")}
</div>

<!-- EDGE LEAK -->
<div class="section">
  <h2>Where the edge leaked</h2>
  {_edge_leak_html(zone_trades)}
</div>

<!-- VET REVIEW -->
<div class="section">
  <h2>Trades worth a second look</h2>
  {_vet_review_html(zone_trades)}
</div>
"""


# ---------------------------------------------------------------------------
# Pair groups for split per-email reports
# ---------------------------------------------------------------------------

FOREX_PAIRS = {"EURUSD", "NZDUSD", "USDJPY", "USDCHF"}
INDEX_COMMODITY_PAIRS = {"NAS100", "GOLD", "XAUUSD"}


def _filter_meta_by_pairs(meta: Dict[str, Any], allowed: set) -> Dict[str, Any]:
    """Return a copy of meta with per-pair fields scoped to `allowed`.

    The audit sections in the per-group emails should only show counters
    for pairs in that group. We touch the per-pair dicts (killzone drops,
    killzone windows) and the pair list; other meta fields are unchanged.
    """
    out = dict(meta)
    out["pairs"] = [p for p in meta.get("pairs", []) if p in allowed]
    drops = meta.get("killzone_drops_by_pair") or {}
    out["killzone_drops_by_pair"] = {k: v for k, v in drops.items() if k in allowed}
    windows = meta.get("killzone_windows_by_pair") or {}
    out["killzone_windows_by_pair"] = {k: v for k, v in windows.items() if k in allowed}
    out["killzone_dropped_alerts"] = int(sum(out["killzone_drops_by_pair"].values()))
    return out


def _build_group_html(
    group_label: str,
    group_trades_all: List[Dict[str, Any]],
    group_meta: Dict[str, Any],
    risk_usd: float,
    out_dir: Path,
    html_filename: str,
    excel_filename: str,
) -> Dict[str, Any]:
    """Build one HTML report + one Excel for a single pair group.

    Mirrors the section layout of the combined report, but every aggregate
    is recomputed against `group_trades_all` so the headline, Sections A/B,
    head-to-head, and every audit section are internally consistent.
    Returns the group's summary dict so the caller can fold it into the
    combined summary.json under `by_group`.
    """
    trades = [t for t in group_trades_all
              if not t.get("news_blocked")
              and not t.get("ist_blocked")
              and not t.get("killzone_blocked")]
    blocked_trades = [t for t in group_trades_all if t.get("news_blocked")]
    ist_blocked_trades = [t for t in group_trades_all if t.get("ist_blocked")]
    kz_blocked_trades = [t for t in group_trades_all if t.get("killzone_blocked")]

    prox_trades = [t for t in trades if t.get("entry_zone") == "proximal"]
    mid_trades  = [t for t in trades if t.get("entry_zone") == "50pct"]

    sb_prox     = _aggregate_for_exit(prox_trades, "r_realised",     risk_usd)
    sb_mid      = _aggregate_for_exit(mid_trades,  "r_realised",     risk_usd)
    sb_prox_tp1 = _aggregate_for_exit(prox_trades, "r_if_exit_tp1",  risk_usd)
    sb_prox_tp2 = _aggregate_for_exit(prox_trades, "r_if_exit_tp2",  risk_usd)
    sb_mid_tp1  = _aggregate_for_exit(mid_trades,  "r_if_exit_tp1",  risk_usd)
    sb_mid_tp2  = _aggregate_for_exit(mid_trades,  "r_if_exit_tp2",  risk_usd)
    fill_prox   = _fill_rate(trades, "proximal")
    fill_mid    = _fill_rate(trades, "50pct")

    # Reconciliation invariant within this group, same as the combined path.
    def _reconcile(zone_label, scoreboard, zone_trades):
        from_scoreboard = round(float(scoreboard.get("total_pnl_usd", 0)), 2)
        filled = [t for t in zone_trades if _is_real_filled(t)]
        from_trades = round(sum(float(t.get("pnl_usd") or 0) for t in filled), 2)
        if abs(from_scoreboard - from_trades) > 0.01:
            raise AssertionError(
                f"P&L reconciliation failed for {group_label}/{zone_label}: "
                f"scoreboard={from_scoreboard} vs per-trade-sum={from_trades}.")
    _reconcile("proximal", sb_prox, prox_trades)
    _reconcile("50pct",    sb_mid,  mid_trades)

    # Excel: this group's filled+blocked rows only (matches what trades.xlsx
    # does for the combined report -- audit rows preserved).
    excel_ok = _try_excel(group_trades_all, out_dir / excel_filename) is not None

    total_pnl_prox = sb_prox.get("total_pnl_usd", 0)
    total_pnl_mid  = sb_mid.get("total_pnl_usd", 0)
    n_prox_filled  = sb_prox.get("trades", 0)
    n_mid_filled   = sb_mid.get("trades", 0)
    pnl_color_prox = "#27ae60" if total_pnl_prox >= 0 else "#e74c3c"
    pnl_color_mid  = "#27ae60" if total_pnl_mid  >= 0 else "#e74c3c"
    pairs_str  = ", ".join(group_meta.get("pairs", []))
    regime_str = group_meta.get("regime", "")

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: #f8f9fa; color: #212529; font-size: 14px; line-height: 1.6; }}
  .wrap {{ max-width: 680px; margin: 0 auto; background: #fff; }}
  .top-band {{ background: #2c3e50; color: #fff; padding: 20px 28px; }}
  .top-band h1 {{ font-size: 18px; font-weight: 700; margin-bottom: 4px; }}
  .top-band .meta {{ font-size: 12px; color: #bdc3c7; }}
  .headline {{ padding: 24px 28px; border-bottom: 1px solid #eee; }}
  .summary-strip {{ display: flex; gap: 12px; margin-top: 16px; }}
  .summary-card {{ flex: 1; padding: 14px; border-radius: 6px; border: 1px solid #eee; }}
  .summary-card .label {{ font-size: 11px; color: #888;
                          text-transform: uppercase; letter-spacing: 0.06em; }}
  .summary-card .val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .summary-card .meta {{ font-size: 11px; color: #888; margin-top: 4px; }}
  .section {{ padding: 22px 28px; border-bottom: 1px solid #eee; }}
  .section h2 {{ font-size: 13px; font-weight: 700; text-transform: uppercase;
                 letter-spacing: 0.06em; color: #888; margin-bottom: 14px; }}
  .section p {{ margin-bottom: 10px; font-size: 14px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 4px; }}
  th {{ background: #2c3e50; color: #fff; padding: 8px 10px; text-align: left;
        font-weight: 600; font-size: 12px; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #eee; }}
  tr:last-child td {{ border-bottom: none; }}
  h4 {{ font-size: 12px; font-weight: 700; color: #666; margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .stats-strip {{ display: flex; gap: 0; border: 1px solid #eee; border-radius: 6px;
                  overflow: hidden; margin-bottom: 16px; }}
  .stat {{ flex: 1; padding: 14px 12px; text-align: center; border-right: 1px solid #eee; }}
  .stat:last-child {{ border-right: none; }}
  .stat .val {{ font-size: 20px; font-weight: 700; }}
  .stat .lbl {{ font-size: 11px; color: #888; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .caveat {{ background: #fffbea; border-left: 3px solid #f59e0b; padding: 10px 14px;
             border-radius: 0 4px 4px 0; font-size: 12px; color: #555; margin-top: 8px; }}
  .footer {{ padding: 16px 28px; background: #f8f9fa; font-size: 11px; color: #999; }}
</style>
</head>
<body>
<div class="wrap">

<div class="top-band">
  <h1>{group_label} &mdash; {group_meta.get('start')} to {group_meta.get('end')}</h1>
  <div class="meta">
    {pairs_str} &nbsp;&middot;&nbsp; 1R = ${risk_usd:.0f} &nbsp;&middot;&nbsp;
    Regime: {regime_str} &nbsp;&middot;&nbsp; H1 bars only, no spread or slippage modelled
  </div>
</div>

<div class="headline">
  <div style="font-size:13px;color:#888;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:8px;">
    Period summary &mdash; both entry zones ({group_label})
  </div>
  <div class="summary-strip">
    <div class="summary-card" style="border-left:4px solid #2c3e50;">
      <div class="label">Proximal entry</div>
      <div class="val" style="color:{pnl_color_prox};">{_m(total_pnl_prox)}</div>
      <div class="meta">{n_prox_filled} filled &middot;
        {sb_prox.get('win_rate_pct', 0):.0f}% won &middot;
        avg {_r(sb_prox.get('expectancy_r', 0))}</div>
    </div>
    <div class="summary-card" style="border-left:4px solid #34495e;">
      <div class="label">50% mean entry</div>
      <div class="val" style="color:{pnl_color_mid};">{_m(total_pnl_mid)}</div>
      <div class="meta">{n_mid_filled} filled &middot;
        {sb_mid.get('win_rate_pct', 0):.0f}% won &middot;
        avg {_r(sb_mid.get('expectancy_r', 0))}</div>
    </div>
  </div>
  <p style="font-size:12px;color:#888;margin-top:14px;">
    Aggregates above and in every section below are scoped to
    {group_label.lower()} pairs only. Numbers reconcile within this email.
  </p>
</div>

{_zone_block_html(
    "Section A &mdash; Proximal entry",
    prox_trades, sb_prox, sb_prox_tp1, sb_prox_tp2, risk_usd,
    "#2c3e50",
)}

{_zone_block_html(
    "Section B &mdash; 50% mean entry",
    mid_trades, sb_mid, sb_mid_tp1, sb_mid_tp2, risk_usd,
    "#34495e",
)}

<div class="section">
  <h2>Proximal entry vs 50% midpoint entry &mdash; head-to-head</h2>
  {_entry_comparison_html(sb_prox, sb_mid, fill_prox, fill_mid,
                          prox_trades, mid_trades, risk_usd)}
</div>

<div class="section">
  <h2>Killzone filter &mdash; alerts dropped</h2>
  {_killzone_audit_html(kz_blocked_trades, group_meta)}
</div>

<div class="section">
  <h2>IST trading-window gate &mdash; alerts dropped</h2>
  {_ist_blackout_html(ist_blocked_trades, group_meta)}
</div>

<div class="section">
  <h2>News blackout &mdash; trades filtered</h2>
  {_news_blackout_html(blocked_trades, group_meta)}
</div>

<div class="section">
  <h2>What's attached</h2>
  <ul style="padding-left:18px;font-size:13px;">
    <li><b>{excel_filename} — Trades tab:</b>
      {"every filled trade for this group, plain-English headers." if excel_ok else "<span style='color:#e74c3c;'>FAILED — openpyxl not installed.</span>"}</li>
    <li><b>{excel_filename} — Zone Register tab:</b> one row per OB, both entry zones side by side.</li>
  </ul>
</div>

<div class="section">
  <h2>System validation check</h2>
  {_validation_html(prox_trades + mid_trades)}
</div>

<div class="footer">
  <b>Limitations:</b>
  No spread, slippage, or swap costs modelled. Exits simulated at H1 bar boundaries.
  Same-bar SL+TP collision resolves SL-first (pessimistic).
  <br><br>
  <b>Run log:</b> <code>backtest/results/{out_dir.name}/</code> &middot;
  verify with <code>git log --grep="Backtest logs: {out_dir.name}"</code>
</div>

</div></body></html>"""

    (out_dir / html_filename).write_text(html, encoding="utf-8")

    return {
        "label": group_label,
        "pairs": group_meta.get("pairs", []),
        "scoreboards": {
            "proximal_realised":  sb_prox,
            "proximal_exit_tp1":  sb_prox_tp1,
            "proximal_exit_tp2":  sb_prox_tp2,
            "fifty_pct_realised": sb_mid,
            "fifty_pct_exit_tp1": sb_mid_tp1,
            "fifty_pct_exit_tp2": sb_mid_tp2,
        },
        "fill_rate_proximal": fill_prox,
        "fill_rate_50pct":    fill_mid,
        "news_blocked_trade_rows": len(blocked_trades),
        "ist_blocked_trade_rows":  len(ist_blocked_trades),
        "killzone_dropped_alerts": int(group_meta.get("killzone_dropped_alerts") or 0),
        "html_file":  html_filename,
        "excel_file": excel_filename,
    }


# ---------------------------------------------------------------------------
# Main report writer
# ---------------------------------------------------------------------------

def write_h1_only_report(
    run_id: str,
    trades: List[Dict[str, Any]],
    raw_alerts: List[Dict[str, Any]],
    meta: Dict[str, Any],
    risk_usd: float = 250.0,
    out_root: Path = None,
) -> Path:
    base = out_root if out_root is not None else (Path(__file__).parent / "results")
    out_dir = Path(base) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # News-blocked AND IST-blocked trades are kept in `trades_all` for
    # Excel/CSV audit (so the user can see what was filtered and why) but
    # are STRIPPED from `trades` -- the variable every metric path uses
    # below. Single point of exclusion: no aggregate calculation should
    # ever see a blocked row.
    #
    # News blackout = +/-30 min around high-impact economic event.
    # IST blackout  = outside user's IST trading window (live mirror).
    trades_all = list(trades)
    trades         = [t for t in trades_all
                      if not t.get("news_blocked")
                      and not t.get("ist_blocked")
                      and not t.get("killzone_blocked")]
    blocked_trades = [t for t in trades_all if t.get("news_blocked")]
    ist_blocked_trades = [t for t in trades_all if t.get("ist_blocked")]
    kz_blocked_trades = [t for t in trades_all if t.get("killzone_blocked")]

    prox_trades = [t for t in trades if t.get("entry_zone") == "proximal"]
    mid_trades  = [t for t in trades if t.get("entry_zone") == "50pct"]

    # Primary view: r_realised under the default policy (ride to TP2 with
    # SL-to-BE after TP1 hit). Every per-pair / per-session / score / killzone
    # / structure / loss table now uses r_realised, so the headline and the
    # breakdowns reconcile. The TP1-only hypothetical column (r_if_exit_tp1)
    # and pure TP2-ride column (r_if_exit_tp2) are only used in the explicit
    # "Proximal vs 50%" comparison table where the exit policy is the variable
    # under test.
    sb_prox    = _aggregate_for_exit(prox_trades, "r_realised",     risk_usd)
    sb_mid     = _aggregate_for_exit(mid_trades,  "r_realised",     risk_usd)
    sb_prox_tp2 = _aggregate_for_exit(prox_trades, "r_if_exit_tp2", risk_usd)
    sb_mid_tp2  = _aggregate_for_exit(mid_trades,  "r_if_exit_tp2", risk_usd)
    sb_prox_tp1 = _aggregate_for_exit(prox_trades, "r_if_exit_tp1", risk_usd)
    sb_mid_tp1  = _aggregate_for_exit(mid_trades,  "r_if_exit_tp1", risk_usd)

    pp_prox   = _per_pair_breakdown(prox_trades,  "r_realised", risk_usd)
    pp_mid    = _per_pair_breakdown(mid_trades,   "r_realised", risk_usd)
    ss_prox   = _per_session_breakdown(prox_trades, "r_realised", risk_usd)
    ss_mid    = _per_session_breakdown(mid_trades,  "r_realised", risk_usd)
    fill_prox = _fill_rate(trades, "proximal")
    fill_mid  = _fill_rate(trades, "50pct")

    score_buckets      = _score_buckets(prox_trades, "r_realised")
    score_buckets_mid  = _score_buckets(mid_trades,  "r_realised")
    # Exit counts must be scoped per entry-zone — the headline counts proximal
    # trades only, so mixing zones here makes the "How trades closed" line
    # contradict the headline (e.g. 21 filled but 29 SL hits across both zones).
    exit_counts_prox = _exit_reason_counts(prox_trades)
    exit_counts_mid  = _exit_reason_counts(mid_trades)

    # News-blackout audit. Every blocked row is listed (without the trade
    # outcome — that would imply we counted it; we did not). The list is
    # intentionally redundant with the Excel sheet so the user can grep
    # summary.json for a specific event without opening the workbook.
    blocked_audit = [
        {
            "pair":     t.get("pair"),
            "alert_ts": t.get("alert_ts"),
            "entry_zone": t.get("entry_zone"),
            "event_title":    t.get("news_event_title"),
            "event_currency": t.get("news_event_currency"),
            "event_source":   t.get("news_event_source"),
            "event_ts":       t.get("news_event_ts"),
        }
        for t in blocked_trades
    ]

    summary = {
        "run_id":              run_id,
        "meta":                meta,
        "risk_per_trade_usd":  risk_usd,
        "total_trade_rows":    len(trades),
        "fill_rate_proximal":  fill_prox,
        "fill_rate_50pct":     fill_mid,
        "exit_reason_counts_proximal": exit_counts_prox,
        "exit_reason_counts_50pct":    exit_counts_mid,
        # Scoreboards under three exit policies. r_realised is the default
        # (TP2-ride with SL-to-BE after TP1); the tp1/tp2 columns are pure
        # hypotheticals and named accordingly.
        "scoreboards": {
            "proximal_realised":  sb_prox,
            "proximal_exit_tp1":  sb_prox_tp1,
            "proximal_exit_tp2":  sb_prox_tp2,
            "fifty_pct_realised": sb_mid,
            "fifty_pct_exit_tp1": sb_mid_tp1,
            "fifty_pct_exit_tp2": sb_mid_tp2,
        },
        # Per-pair / per-session breakdowns use r_realised so they reconcile
        # to the headline total. Keys carry the column name explicitly.
        "per_pair_proximal_realised": pp_prox,
        "per_pair_50pct_realised":    pp_mid,
        "per_session_proximal_realised": ss_prox,
        "per_session_50pct_realised":    ss_mid,
        "score_buckets_proximal_realised": score_buckets,
        "score_buckets_50pct_realised":    score_buckets_mid,
        # Blackout exclusion counters. Each is the count of trade rows
        # (proximal + 50pct, up to 2x distinct alerts) dropped from every
        # aggregate metric calculation above.
        "news_blocked_trade_rows": len(blocked_trades),
        "news_blocked_audit":      blocked_audit,
        "ist_blocked_trade_rows":  len(ist_blocked_trades),
        # Killzone hard filter: alerts dropped before simulation. Counts
        # come from run_backtest meta -- the dropped alerts never produced
        # trade rows, so we have to pass the totals through, not derive
        # them from `trades`.
        "killzone_dropped_alerts":  int(meta.get("killzone_dropped_alerts") or 0),
        "killzone_drops_by_pair":   dict(meta.get("killzone_drops_by_pair") or {}),
        "killzone_windows_by_pair": dict(meta.get("killzone_windows_by_pair") or {}),
    }

    # Reconciliation invariant. The headline P&L (sb_prox / sb_mid) MUST
    # equal the sum of per-trade pnl_usd for the same population. If this
    # fails, the email is publishing inconsistent numbers across sections
    # -- the kind of bug that wastes hours to debug downstream. Fail loud.
    def _reconcile(zone_label, scoreboard, zone_trades):
        from_scoreboard = round(float(scoreboard.get("total_pnl_usd", 0)), 2)
        filled = [t for t in zone_trades if _is_real_filled(t)]
        from_trades = round(sum(float(t.get("pnl_usd") or 0) for t in filled), 2)
        if abs(from_scoreboard - from_trades) > 0.01:
            raise AssertionError(
                f"P&L reconciliation failed for {zone_label}: "
                f"scoreboard={from_scoreboard} vs per-trade-sum={from_trades}. "
                f"This means the email headline contradicts the trade rows. "
                f"Fix the aggregator before shipping the report."
            )
    _reconcile("proximal", sb_prox, prox_trades)
    _reconcile("50pct",    sb_mid,  mid_trades)

    # Files. Use trades_all for CSV and Excel so blocked rows appear in
    # the audit outputs (column news_blocked + event metadata). Metrics
    # were computed above on the filtered `trades`, so summary stats are
    # unaffected by this.
    _trades_csv(trades_all, out_dir / "trades.csv")
    excel_ok = _try_excel(trades_all, out_dir / "trades.xlsx") is not None
    with open(out_dir / "raw_alerts.jsonl", "w") as f:
        for a in raw_alerts:
            f.write(json.dumps(a, default=str) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # --- HTML ---
    # Top-of-email headline mirrors the Proximal zone's r_realised (the
    # default-policy column). This is the same column the zone breakdowns
    # use, so headline + breakdowns reconcile to the same total.
    total_pnl_prox = sb_prox.get("total_pnl_usd", 0)
    total_pnl_mid  = sb_mid.get("total_pnl_usd", 0)
    n_prox_filled  = sb_prox.get("trades", 0)
    n_mid_filled   = sb_mid.get("trades", 0)
    pnl_color_prox = "#27ae60" if total_pnl_prox >= 0 else "#e74c3c"
    pnl_color_mid  = "#27ae60" if total_pnl_mid  >= 0 else "#e74c3c"
    pairs_str      = ", ".join(meta.get("pairs", []))
    regime_str     = meta.get("regime", "")

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: #f8f9fa; color: #212529; font-size: 14px; line-height: 1.6; }}
  .wrap {{ max-width: 680px; margin: 0 auto; background: #fff; }}

  /* Top band */
  .top-band {{ background: #2c3e50; color: #fff; padding: 20px 28px; }}
  .top-band h1 {{ font-size: 18px; font-weight: 700; margin-bottom: 4px; }}
  .top-band .meta {{ font-size: 12px; color: #bdc3c7; }}

  /* Headline number */
  .headline {{ padding: 24px 28px; border-bottom: 1px solid #eee; }}
  .headline .big {{ font-size: 32px; font-weight: 700; }}
  .headline .sub {{ font-size: 14px; color: #555; margin-top: 4px; }}
  .headline .verdict {{ font-size: 13px; color: #444; margin-top: 10px;
                        background: #f8f9fa;
                        padding: 8px 12px; border-radius: 0 4px 4px 0; }}
  .summary-strip {{ display: flex; gap: 12px; margin-top: 16px; }}
  .summary-card {{ flex: 1; padding: 14px; border-radius: 6px; border: 1px solid #eee; }}
  .summary-card .label {{ font-size: 11px; color: #888;
                          text-transform: uppercase; letter-spacing: 0.06em; }}
  .summary-card .val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .summary-card .meta {{ font-size: 11px; color: #888; margin-top: 4px; }}

  /* Sections */
  .section {{ padding: 22px 28px; border-bottom: 1px solid #eee; }}
  .section h2 {{ font-size: 13px; font-weight: 700; text-transform: uppercase;
                 letter-spacing: 0.06em; color: #888; margin-bottom: 14px; }}
  .section p {{ margin-bottom: 10px; font-size: 14px; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 4px; }}
  th {{ background: #2c3e50; color: #fff; padding: 8px 10px; text-align: left;
        font-weight: 600; font-size: 12px; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #eee; }}
  tr:last-child td {{ border-bottom: none; }}
  h4 {{ font-size: 12px; font-weight: 700; color: #666; margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 0.04em; }}

  /* Stats strip */
  .stats-strip {{ display: flex; gap: 0; border: 1px solid #eee; border-radius: 6px;
                  overflow: hidden; margin-bottom: 16px; }}
  .stat {{ flex: 1; padding: 14px 12px; text-align: center; border-right: 1px solid #eee; }}
  .stat:last-child {{ border-right: none; }}
  .stat .val {{ font-size: 20px; font-weight: 700; }}
  .stat .lbl {{ font-size: 11px; color: #888; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.04em; }}

  /* Caveat */
  .caveat {{ background: #fffbea; border-left: 3px solid #f59e0b; padding: 10px 14px;
             border-radius: 0 4px 4px 0; font-size: 12px; color: #555; margin-top: 8px; }}

  /* Footer */
  .footer {{ padding: 16px 28px; background: #f8f9fa; font-size: 11px; color: #999; }}
</style>
</head>
<body>
<div class="wrap">

<!-- TOP BAND -->
<div class="top-band">
  <h1>H1-Only Backtest &mdash; {meta.get('start')} to {meta.get('end')}</h1>
  <div class="meta">
    {pairs_str} &nbsp;&middot;&nbsp; 1R = ${risk_usd:.0f} &nbsp;&middot;&nbsp;
    Regime: {regime_str} &nbsp;&middot;&nbsp; H1 bars only, no spread or slippage modelled
  </div>
</div>

<!-- HEADLINE: side-by-side summary of both entry zones -->
<div class="headline">
  <div style="font-size:13px;color:#888;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:8px;">
    Week summary &mdash; both entry zones
  </div>
  <div class="summary-strip">
    <div class="summary-card" style="border-left:4px solid #2c3e50;">
      <div class="label">Proximal entry</div>
      <div class="val" style="color:{pnl_color_prox};">{_m(total_pnl_prox)}</div>
      <div class="meta">{n_prox_filled} filled &middot;
        {sb_prox.get('win_rate_pct', 0):.0f}% won &middot;
        avg {_r(sb_prox.get('expectancy_r', 0))}</div>
    </div>
    <div class="summary-card" style="border-left:4px solid #34495e;">
      <div class="label">50% mean entry</div>
      <div class="val" style="color:{pnl_color_mid};">{_m(total_pnl_mid)}</div>
      <div class="meta">{n_mid_filled} filled &middot;
        {sb_mid.get('win_rate_pct', 0):.0f}% won &middot;
        avg {_r(sb_mid.get('expectancy_r', 0))}</div>
    </div>
  </div>
  <p style="font-size:12px;color:#888;margin-top:14px;">
    Each section below is a self-contained analysis of one entry zone.
    Numbers within a section reconcile to that zone's headline.
  </p>
</div>

<!-- ============================================================ -->
<!-- SECTION A: PROXIMAL ENTRY -->
<!-- ============================================================ -->
{_zone_block_html(
    "Section A &mdash; Proximal entry",
    prox_trades, sb_prox, sb_prox_tp1, sb_prox_tp2, risk_usd,
    "#2c3e50",
)}

<!-- ============================================================ -->
<!-- SECTION B: 50% MEAN ENTRY -->
<!-- ============================================================ -->
{_zone_block_html(
    "Section B &mdash; 50% mean entry",
    mid_trades, sb_mid, sb_mid_tp1, sb_mid_tp2, risk_usd,
    "#34495e",
)}

<!-- ============================================================ -->
<!-- SECTION C: ENTRY-MODE COMPARISON (single table, cross-zone) -->
<!-- ============================================================ -->
<div class="section">
  <h2>Proximal entry vs 50% midpoint entry &mdash; head-to-head</h2>
  {_entry_comparison_html(sb_prox, sb_mid, fill_prox, fill_mid,
                          prox_trades, mid_trades, risk_usd)}
</div>

<!-- ============================================================ -->
<!-- SECTION D0: KILLZONE FILTER AUDIT -->
<!-- ============================================================ -->
<div class="section">
  <h2>Killzone filter &mdash; alerts dropped</h2>
  {_killzone_audit_html(kz_blocked_trades, meta)}
</div>

<!-- ============================================================ -->
<!-- SECTION D: IST BLACKOUT AUDIT -->
<!-- ============================================================ -->
<div class="section">
  <h2>IST trading-window gate &mdash; alerts dropped</h2>
  {_ist_blackout_html(ist_blocked_trades, meta)}
</div>

<!-- ============================================================ -->
<!-- SECTION E: NEWS BLACKOUT AUDIT -->
<!-- ============================================================ -->
<div class="section">
  <h2>News blackout &mdash; trades filtered</h2>
  {_news_blackout_html(blocked_trades, meta)}
</div>

<!-- SECTION 8: FILES -->
<div class="section">
  <h2>What's attached</h2>
  <ul style="padding-left:18px;font-size:13px;">
    <li><b>trades.xlsx — Trades tab:</b> {"every filled trade, plain-English headers, day of week, color-coded P&L, amber highlights on flagged trades. News-blocked rows are highlighted and marked in column News Blocked." if excel_ok else "<span style='color:#e74c3c;'>FAILED — openpyxl not installed. Use trades.csv instead.</span>"}</li>
    <li><b>trades.xlsx — Zone Register tab:</b> one row per OB, both entry zones side by side — use this to verify entry/SL/TP levels and fill logic. Includes News Blocked + event details columns.</li>
    <li><b>trades.csv</b> — machine-readable column names, used by aggregate_runs.py</li>
    <li><b>summary.json</b> — all metrics in structured format</li>
    <li><b>run_log.jsonl + console.log</b> — full diagnostic log</li>
  </ul>
</div>

<!-- SECTION 9: VALIDATION -->
<div class="section">
  <h2>System validation check</h2>
  <p style="font-size:12px;color:#666;margin-bottom:8px;">
    Verifies that entry prices are correctly positioned relative to SL and TP,
    and that exit outcomes match exit types (e.g. SL exit has negative R).
  </p>
  {_validation_html(prox_trades + mid_trades)}
</div>

<!-- FOOTER -->
<div class="footer">
  <b>Limitations:</b>
  No spread, slippage, or swap costs modelled — real P&amp;L ~5–10% lower.
  Exits simulated at H1 bar boundaries. Same-bar SL+TP collision resolves SL-first (pessimistic).
  yfinance bars may differ slightly from broker bars.
  <br><br>
  <b>Run log:</b> <code>backtest/results/{run_id}/</code> &middot;
  verify with <code>git log --grep="Backtest logs: {run_id}"</code>
</div>

</div><!-- /wrap -->
</body></html>"""

    (out_dir / "report.html").write_text(html, encoding="utf-8")

    # ---- Per-group split reports (Forex vs Gold/NAS) ----
    # Same Section A/B layout, same thresholds, but every aggregate and audit
    # section is recomputed against the group's pair subset. The combined
    # report above is unchanged so update_registry.py / aggregate_runs.py keep
    # reading trades.xlsx + summary.json as today.
    forex_trades_all = [t for t in trades_all if t.get("pair") in FOREX_PAIRS]
    indcom_trades_all = [t for t in trades_all if t.get("pair") in INDEX_COMMODITY_PAIRS]

    forex_meta  = _filter_meta_by_pairs(meta, FOREX_PAIRS)
    indcom_meta = _filter_meta_by_pairs(meta, INDEX_COMMODITY_PAIRS)

    by_group = {}
    if forex_trades_all:
        by_group["forex"] = _build_group_html(
            "Forex", forex_trades_all, forex_meta, risk_usd,
            out_dir, "report_forex.html", "forex_trades.xlsx",
        )
    if indcom_trades_all:
        by_group["gold_nas"] = _build_group_html(
            "Gold + NAS100", indcom_trades_all, indcom_meta, risk_usd,
            out_dir, "report_gold_nas.html", "nas_xau_trades.xlsx",
        )

    if by_group:
        # Fold per-group summaries into the combined summary.json so future
        # tooling can read the partitioned numbers without re-running anything.
        summary["by_group"] = by_group
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    return out_dir
