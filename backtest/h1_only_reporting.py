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

def _aggregate_for_exit(trades: List[Dict[str, Any]], r_col: str,
                        risk_usd: float) -> Dict[str, Any]:
    """Aggregate filled trades under a hypothetical exit policy."""
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
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
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
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
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
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
    filled = [t for t in zone_rows if t.get("exit_reason") != "never_filled"]
    return {
        "alerts":        len(zone_rows),
        "filled":        len(filled),
        "fill_rate_pct": round(len(filled) / len(zone_rows) * 100, 1) if zone_rows else 0.0,
    }


def _score_buckets(trades: List[Dict[str, Any]], r_col: str) -> List[Dict[str, Any]]:
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
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
# By-pair and by-session flat tables (replaces the sparse 2D grid)
# ---------------------------------------------------------------------------

_SESSION_ORDER = ["Asia", "London", "NY", "Other"]


def _flat_breakdown_row(label: str, sub: pd.DataFrame, r_col: str) -> str:
    """One row in a flat breakdown table. Low-n cells are faded so the eye
    sees the number is thin without losing the data point."""
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
    # Low-n rows render faded -- number is shown but de-emphasised.
    opacity = "opacity:0.55;" if n < 3 else ""
    return (f"<tr style='background:{bg};{opacity}'>"
            f"<td><b>{label}</b></td>"
            f"<td>{n}</td>"
            f"<td>{wr:.0f}%</td>"
            f"<td>{sign}{exp:.2f}R</td></tr>")


def _by_pair_html(trades: List[Dict[str, Any]], r_col: str) -> str:
    """Win rate / avg R / trade count, by pair (all sessions combined)."""
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
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
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
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

    Color encodes WR (green >=50, amber 40-50, red <40). Low-n cells (n<3)
    are faded so the eye sees the thin sample without losing the data point.
    Right-most column and bottom row show pair-totals and session-totals so
    weeks with only one trade per (pair, session) still produce a readable
    pair-level row.
    """
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
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
        opacity = "opacity:0.55;" if n < 3 else ""
        sign = "+" if exp >= 0 else ""
        return (f"<td style='background:{bg};{opacity}text-align:center;'>"
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
            f"Faded cells have fewer than 3 trades -- read with caution. "
            f"Right column and bottom row are roll-ups.</p>")


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
            filled_sub = sub[sub["exit_reason"] != "never_filled"]
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
            filled_sub = sub[sub["exit_reason"] != "never_filled"]
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
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
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
            if len(with_c) < 3:
                row += "<td style='color:#bbb;text-align:center;'>—</td>"
                continue
            wr_with  = (with_c[r_col] > 0).mean() * 100
            wr_wout  = (wout_c[r_col] > 0).mean() * 100 if len(wout_c) >= 3 else None
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
            "↓ = hurt. — = fewer than 3 trades with this confluence.</p>")
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
# Killzone diagnostic: are wins concentrated inside the killzone vs losses?
# ---------------------------------------------------------------------------

def _killzone_split_html(trades: List[Dict[str, Any]]) -> str:
    """Two-line diagnostic: % of wins in killzone vs % of losses in killzone.

    Killzone label comes from killzone_pts in the score breakdown (>0 = in).
    A useful killzone should show wins clustering in-killzone and losses
    clustering outside. If wins% and losses% are both ~50/50, the killzone
    isn't separating signal from noise.
    """
    filled = [t for t in trades
              if t.get("exit_reason") not in ("never_filled",)
              and t.get("entry_zone") == "proximal"]
    if not filled:
        return "<p style='color:#888;'>No filled trades.</p>"

    wins  = [t for t in filled if (t.get("r_realised") or 0) > 0]
    losses = [t for t in filled if (t.get("r_realised") or 0) <= 0]

    def _in_kz(t):
        return (t.get("killzone_pts") or 0) > 0

    n_w, n_l = len(wins), len(losses)
    w_in = sum(1 for t in wins   if _in_kz(t))
    l_in = sum(1 for t in losses if _in_kz(t))

    def _pct(num, denom):
        return f"{num / denom * 100:.0f}%" if denom else "—"

    # Expectancy split (uses r_realised under default policy)
    in_kz  = [t for t in filled if _in_kz(t)]
    out_kz = [t for t in filled if not _in_kz(t)]
    def _exp(rows):
        return sum((t.get("r_realised") or 0) for t in rows) / len(rows) if rows else None
    def _wr(rows):
        return sum(1 for t in rows if (t.get("r_realised") or 0) > 0) / len(rows) * 100 if rows else None

    e_in, e_out = _exp(in_kz), _exp(out_kz)
    wr_in, wr_out = _wr(in_kz), _wr(out_kz)

    rows = (
        f"<tr><td><b>Wins</b></td><td>{n_w}</td>"
        f"<td>{w_in} ({_pct(w_in, n_w)})</td>"
        f"<td>{n_w - w_in} ({_pct(n_w - w_in, n_w)})</td></tr>"
        f"<tr><td><b>Losses</b></td><td>{n_l}</td>"
        f"<td>{l_in} ({_pct(l_in, n_l)})</td>"
        f"<td>{n_l - l_in} ({_pct(n_l - l_in, n_l)})</td></tr>"
    )

    def _fmt_r(v):
        if v is None: return "—"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.2f}R"

    expectancy_line = (
        f"<p style='font-size:13px;color:#555;margin-top:8px;'>"
        f"<b>In-killzone</b>: n={len(in_kz)}, "
        f"win rate {wr_in:.0f}%, expectancy {_fmt_r(e_in)} &nbsp;|&nbsp; "
        f"<b>Outside killzone</b>: n={len(out_kz)}, "
        f"win rate {wr_out:.0f}%, expectancy {_fmt_r(e_out)}"
        f"</p>" if (in_kz and out_kz)
        else f"<p style='font-size:13px;color:#888;margin-top:8px;'>"
        f"Single-bucket dataset — need trades in both buckets for split.</p>"
    )

    return (
        f"<p style='font-size:12px;color:#666;margin-bottom:10px;'>"
        f"Where are the wins coming from — inside or outside killzone hours? "
        f"If wins cluster in-killzone and losses cluster outside, the killzone "
        f"is doing real work. If both split ~50/50, killzone isn't separating signal."
        f"</p>"
        f"<table><thead>"
        f"<tr><th>Outcome</th><th>Total</th><th>In killzone</th><th>Outside killzone</th></tr>"
        f"</thead><tbody>{rows}</tbody></table>"
        f"{expectancy_line}"
        f"<p style='font-size:11px;color:#aaa;margin-top:4px;'>"
        f"Killzone = UTC 06:00–18:00 (forex/commodity) or 13:00–20:00 (NAS100)."
        f"</p>"
    )


# ---------------------------------------------------------------------------
# Loss pattern analysis
# ---------------------------------------------------------------------------

# UTC hours that overlap with high-impact economic releases.
_NEWS_HOURS = {7, 8, 9, 13, 14}  # EUR/UK data 07-09, US data 13-14 UTC

def _loss_analysis_html(trades: List[Dict[str, Any]]) -> str:
    losses = [t for t in trades
              if t.get("exit_reason") not in ("never_filled",) and t.get("r_realised", 0) < 0]
    if not losses:
        return "<p style='color:#27ae60;'>No losing trades this week.</p>"

    n = len(losses)
    findings = []

    # 1. News timing
    news_hits = 0
    for t in losses:
        ts = t.get("fill_ts") or t.get("alert_ts", "")
        try:
            h = pd.Timestamp(ts).hour
            if h in _NEWS_HOURS:
                news_hits += 1
        except Exception:
            pass
    if news_hits > 0:
        findings.append(
            f"<b>{news_hits} of {n}</b> losing trades were filled during high-impact news windows "
            f"(07–09 UTC or 13–14 UTC). These hours have unpredictable price spikes that "
            f"can invalidate OB setups. Consider filtering these hours out."
        )

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

    # 4. Outside kill zone
    no_kz = sum(1 for t in losses if (t.get("killzone_pts") or 0) == 0)
    if no_kz > 0:
        findings.append(
            f"<b>{no_kz} of {n}</b> losing trades were taken outside of kill zone hours. "
            f"Entries during off-peak hours see lower institutional participation and "
            f"weaker follow-through."
        )

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
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
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

    note = ""
    if major_exp is not None and minor_exp is not None and major_n >= 3 and minor_n >= 3:
        diff = major_exp - minor_exp
        if diff > 0.15:
            note = (f"<p style='font-size:13px;color:#27ae60;margin-top:8px;'>"
                    f"✓ Major events outperformed Minor by {diff:.2f}R per trade. "
                    f"This is the expected behaviour — Major detection is working as designed.</p>")
        elif diff < -0.15:
            note = (f"<p style='font-size:13px;color:#e74c3c;margin-top:8px;'>"
                    f"⚠ Minor events outperformed Major by {abs(diff):.2f}R — unexpected. "
                    f"Major event detection may be misclassifying setups. Verify before trading live.</p>")
        else:
            note = (f"<p style='font-size:13px;color:#888;margin-top:8px;'>"
                    f"Major and Minor performance is similar this week (difference: {diff:+.2f}R). "
                    f"Need more data to draw conclusions.</p>")
    elif major_n < 3 or minor_n < 3:
        note = (f"<p style='font-size:12px;color:#888;margin-top:8px;'>"
                f"Major: {major_n} trades · Minor: {minor_n} trades. "
                f"Need at least 3 of each before comparing.</p>")

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
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
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
                      if t.get("exit_reason") != "never_filled"
                      and t.get("entry_zone") == "proximal"])
        n_mid  = len([t for t in trades
                      if t.get("exit_reason") != "never_filled"
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
            "Proximal Dollar P&L":     round(float(_v(prox, "r_realised", 0)) * float(_v(prox, "pnl_usd", 0) or 0 or 1), 0)
                                       if prox else "",
            "50% Filled?":             "Yes" if _v(mid, "exit_reason") != "never_filled" and mid else "No",
            "50% Outcome":             _exit_labels.get(_v(mid, "exit_reason"), _v(mid, "exit_reason")),
            "50% R":                   _v(mid, "r_realised"),
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
        })

    return pd.DataFrame(rows)


def _day_of_week(ts_str: str) -> str:
    try:
        return pd.Timestamp(ts_str).day_name()
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
    "r_realised":        "R Achieved",
    "pnl_usd":           "Dollar P&L",
    "mfe_r":             "Best Price Reached (R)",
    "mae_r":             "Worst Price Reached (R)",
    "bars_to_exit":      "Hours Held",
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
    "news_blocked":         "News Blocked",
    "news_event_title":     "News Event",
    "news_event_currency":  "News Currency",
    "news_event_source":    "News Source",
    "news_event_ts":        "News Event Time (UTC)",
    "ist_blocked":          "IST Window Blocked",
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


def _try_excel(trades: List[Dict[str, Any]], path: Path) -> Optional[Path]:
    """Write human-readable Excel. Returns path or None on failure."""
    # Only filled trades in the Excel — never_filled are counted in fill rate
    # but are not trade outcomes and would confuse the spreadsheet.
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
    if not filled:
        return None
    try:
        df = pd.DataFrame(filled)

        # Add day of week (tracked for cross-run aggregate analysis).
        if "alert_ts" in df.columns:
            df["day_of_week"] = df["alert_ts"].apply(_day_of_week)

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

        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            out_df.to_excel(xw, sheet_name="Trades", index=False)
            if not zone_df.empty:
                zone_df.to_excel(xw, sheet_name="Zone Register", index=False)
            ws = xw.sheets["Trades"]

            # Apply color to Dollar P&L column and R Achieved column.
            try:
                from openpyxl.styles import PatternFill, Font, Alignment
                green_fill = PatternFill("solid", fgColor="C6EFCE")
                red_fill   = PatternFill("solid", fgColor="FFC7CE")
                grey_fill  = PatternFill("solid", fgColor="F2F2F2")

                # Header row styling.
                for cell in ws[1]:
                    cell.font      = Font(bold=True, color="FFFFFF")
                    cell.fill      = PatternFill("solid", fgColor="2C3E50")
                    cell.alignment = Alignment(wrap_text=True)

                # Find P&L and R columns.
                headers = [cell.value for cell in ws[1]]
                pnl_col = headers.index("Dollar P&L") + 1 if "Dollar P&L" in headers else None
                r_col   = headers.index("R Achieved") + 1 if "R Achieved" in headers else None
                rev_col = headers.index("Worth Reviewing") + 1 if "Worth Reviewing" in headers else None

                for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=2):
                    # Alternate row background.
                    base_fill = grey_fill if row_idx % 2 == 0 else None
                    for cell in row:
                        if base_fill:
                            cell.fill = base_fill

                    # P&L color.
                    if pnl_col:
                        val = ws.cell(row=row_idx, column=pnl_col).value
                        if val is not None:
                            fill = green_fill if val > 0 else (red_fill if val < 0 else None)
                            if fill:
                                for cell in row:
                                    cell.fill = fill

                    # Highlight "Worth Reviewing = Yes" in amber.
                    if rev_col:
                        rev_val = ws.cell(row=row_idx, column=rev_col).value
                        if rev_val == "Yes":
                            for cell in row:
                                cell.fill = PatternFill("solid", fgColor="FFEB9C")

                # News-blocked rows: lavender highlight applied AFTER P&L /
                # reviewing colors, so the eye sees "this row was excluded"
                # regardless of its hypothetical outcome. Search by header.
                nb_col = headers.index("News Blocked") + 1 if "News Blocked" in headers else None
                if nb_col:
                    news_fill = PatternFill("solid", fgColor="E8DAEF")  # light purple
                    for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=2):
                        val = ws.cell(row=row_idx, column=nb_col).value
                        if val in (True, "True", "Yes", 1, "1"):
                            for cell in row:
                                cell.fill = news_fill

                # Column widths.
                col_widths = {
                    "Currency Pair": 14, "Direction": 10, "Trading Session": 12,
                    "Entry Type": 18, "Entry Price": 12, "Stop Loss": 12,
                    "Take Profit 1": 13, "Take Profit 2": 13,
                    "TP1 Reward:Risk": 14, "TP2 Reward:Risk": 14,
                    "How Trade Closed": 24, "Exit Price": 12,
                    "R Achieved": 12, "Dollar P&L": 12,
                    "Best Price Reached (R)": 20, "Worst Price Reached (R)": 20,
                    "Hours Held": 10, "Setup Score (0–8)": 14,
                    "Confluences Active": 22,
                    "FVG Present": 12, "Liquidity Sweep Present": 22,
                    "Structure Event (BOS / CHoCH)": 24,
                    "Structure Tier (Major / Minor)": 24,
                    "Worth Reviewing": 15, "Why Worth Reviewing": 40,
                    "Alert Time (UTC)": 20, "Trade Closed (UTC)": 20,
                }
                for i, col in enumerate(ws.columns, start=1):
                    header = ws.cell(row=1, column=i).value
                    ws.column_dimensions[col[0].column_letter].width = col_widths.get(header, 14)

                ws.freeze_panes = "A2"

                # Style Zone Register tab if it exists.
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
    if n < 5:
        return "Not enough trades this week to draw a conclusion (fewer than 5 filled)."
    if exp >= 0.3 and wr >= 50:
        return "System performed well this week."
    if exp >= 0 and wr >= 45:
        return "Modest positive week — edge is there but modest."
    if exp >= 0 and wr < 45:
        return "Profitable but fewer wins than expected — check if losses are stopping cleanly."
    if exp < 0 and wr >= 50:
        return "Many wins but losses were too large — take profit levels may be too tight."
    return "Difficult week — below threshold on both win rate and expectancy."


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
    if len(buckets) < 2:
        return "Not enough score diversity to draw a conclusion."
    exp_vals = [b["expectancy_r"] for b in buckets]
    rises = sum(1 for a, b in zip(exp_vals, exp_vals[1:]) if b > a)
    total = len(exp_vals) - 1
    ratio = rises / total if total > 0 else 0
    if ratio >= 0.7:
        return "✓ Yes — higher score setups consistently produced better outcomes this week."
    if ratio >= 0.4:
        return "~ Partial — some relationship, not consistent across all score levels."
    return "✗ No — score did not predict outcomes this week. One week is not enough to conclude — check across all runs."


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


def _entry_comparison_html(sb_prox: Dict, sb_mid: Dict, fill_prox: Dict, fill_mid: Dict) -> str:
    def _row(label: str, prox_val: str, mid_val: str) -> str:
        return f"<tr><td><b>{label}</b></td><td>{prox_val}</td><td>{mid_val}</td></tr>"

    rows = "".join([
        _table_row(["", "Proximal entry", "50% midpoint entry"], header=True),
        _row("Alerts triggered",
             str(fill_prox["alerts"]), str(fill_mid["alerts"])),
        _row("Orders filled",
             f"{fill_prox['filled']} ({fill_prox['fill_rate_pct']:.0f}%)",
             f"{fill_mid['filled']} ({fill_mid['fill_rate_pct']:.0f}%)"),
        _row("Win rate",
             f"{sb_prox.get('win_rate_pct', 0):.0f}%",
             f"{sb_mid.get('win_rate_pct', 0):.0f}%"),
        _row("Avg R per trade",
             _r(sb_prox.get("expectancy_r", 0)),
             _r(sb_mid.get("expectancy_r", 0))),
        _row("Total P&L",
             f"<b>{_m(sb_prox.get('total_pnl_usd', 0))}</b>",
             f"<b>{_m(sb_mid.get('total_pnl_usd', 0))}</b>"),
    ])
    note = ("<p style='font-size:12px;color:#666;margin-top:8px;'>"
            "Both entries target the same take profit levels. The 50% entry fills less often "
            "but has a tighter stop-to-target ratio when it does fill.</p>")
    return f"<table>{rows}</table>{note}"


def _vet_review_html(trades: List[Dict]) -> str:
    flagged = [(t, *_flag_vet_review(t)) for t in trades
               if t.get("exit_reason") != "never_filled"]
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
    """One self-contained section per entry zone: headline -> by-pair ->
    by-session -> pair x session matrix -> killzone split -> score buckets
    -> confluences -> structure events -> loss analysis -> TP1 vs TP2.

    Every aggregate inside this block uses r_realised so the headline and
    the breakdowns reconcile to the same total.
    """
    # Headline figures (mirror the top-of-email shape but scoped to this zone).
    n         = sb_realised.get("trades", 0)
    total_pnl = sb_realised.get("total_pnl_usd", 0)
    exp_r     = sb_realised.get("expectancy_r", 0)
    wr        = sb_realised.get("win_rate_pct", 0)
    pnl_color = "#27ae60" if total_pnl >= 0 else "#e74c3c"
    headline  = _week_headline(sb_realised)

    # Exit-reason narrative scoped to this zone.
    exit_counts = _exit_reason_counts(zone_trades)
    exit_reason_plain = {
        "sl": "Stop loss", "tp1": "TP1 hit", "tp2": "TP2 hit",
        "timeout": "Time limit", "window_end": "End of window",
        "sl_collision": "SL+TP same bar", "never_filled": "Never filled",
    }
    exit_breakdown = " &middot; ".join(
        f"<b>{exit_reason_plain.get(k, k)}: {v}</b>"
        for k, v in sorted(exit_counts.items())
    )

    # TP1-vs-TP2 hypothetical line.
    _filled = [t for t in zone_trades if t.get("exit_reason") != "never_filled"]
    if _filled:
        sum_tp2 = sum((t.get("r_if_exit_tp2") or 0) for t in _filled)
        sum_tp1 = sum((t.get("r_if_exit_tp1") or 0) for t in _filled)
        def _f(v): return f"{'+' if v >= 0 else ''}{v:.1f}R"
        tp1_vs_tp2_line = (
            f"<p style='font-size:12px;color:#888;margin-top:2px;'>"
            f"If we'd banked at TP1 instead of riding to TP2: "
            f"total {_f(sum_tp1)} across {len(_filled)} trades "
            f"(vs current TP2-ride policy: {_f(sum_tp2)}).</p>"
        )
    else:
        tp1_vs_tp2_line = ""

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

<!-- ZONE HEADLINE -->
<div class="section">
  <div style="font-size:26px;font-weight:700;color:{pnl_color};">
    {_m(total_pnl)}
  </div>
  <div style="font-size:14px;color:#555;margin-top:4px;">
    {n} filled trades &nbsp;&middot;&nbsp;
    {wr:.0f}% won &nbsp;&middot;&nbsp;
    avg {_r(exp_r)} per trade
  </div>
  <div style="font-size:13px;color:#444;margin-top:10px;
              background:#f8f9fa;border-left:3px solid {pnl_color};
              padding:8px 12px;border-radius:0 4px 4px 0;">
    {headline}
  </div>

  <div class="stats-strip" style="margin-top:14px;">
    <div class="stat">
      <div class="val" style="color:#27ae60;">{sb_realised.get('wins', 0)}</div>
      <div class="lbl">Wins</div>
    </div>
    <div class="stat">
      <div class="val" style="color:#e74c3c;">{sb_realised.get('losses', 0)}</div>
      <div class="lbl">Losses</div>
    </div>
    <div class="stat">
      <div class="val" style="color:#888;">{sb_realised.get('breakevens', 0)}</div>
      <div class="lbl">Break-evens</div>
    </div>
    <div class="stat">
      <div class="val">{_r(sb_realised.get('avg_win_r', 0))}</div>
      <div class="lbl">Avg win</div>
    </div>
    <div class="stat">
      <div class="val">{_r(sb_realised.get('avg_loss_r', 0))}</div>
      <div class="lbl">Avg loss</div>
    </div>
  </div>
  {_exit_narrative(sb_realised, risk_usd)}
  <p style="font-size:12px;color:#888;margin-top:4px;">
    How trades closed: {exit_breakdown}
  </p>
  {tp1_vs_tp2_line}
</div>

<!-- BY PAIR -->
<div class="section">
  <h2>Where it worked &mdash; by pair</h2>
  <p style="font-size:12px;color:#666;margin-bottom:10px;">
    Win rate, average R and trade count per pair. Faded rows have fewer
    than 3 trades &mdash; read with caution.
  </p>
  {_by_pair_html(zone_trades, "r_realised")}
</div>

<!-- BY SESSION -->
<div class="section">
  <h2>Where it worked &mdash; by session</h2>
  <p style="font-size:12px;color:#666;margin-bottom:10px;">
    Same view, sliced by trading session.
  </p>
  {_by_session_html(zone_trades, "r_realised")}
</div>

<!-- PAIR x SESSION MATRIX -->
<div class="section">
  <h2>Pair &times; session matrix</h2>
  <p style="font-size:12px;color:#666;margin-bottom:10px;">
    The view that actually tells you where edge lives: each (pair, session)
    intersection in one table. Roll-ups on the right and bottom.
  </p>
  {_pair_session_matrix_html(zone_trades, "r_realised")}
</div>

<!-- KILLZONE -->
<div class="section">
  <h2>Killzone &mdash; wins vs losses</h2>
  {_killzone_split_html(zone_trades)}
</div>

<!-- SCORE -->
<div class="section">
  <h2>Did the confidence score predict better trades?</h2>
  <p><b>{score_verdict}</b></p>
  {_score_table_html(score_buckets)}
  <p style="font-size:12px;color:#888;margin-top:8px;">
    Score rises with each confluence present: FVG, liquidity sweep, kill zone,
    PD alignment, OB freshness, structure tier.
  </p>
</div>

<!-- CONFLUENCES -->
<div class="section">
  <h2>Which confluences actually helped &mdash; by pair</h2>
  <p style="font-size:12px;color:#666;margin-bottom:10px;">
    Win rate when each confluence was present. &uarr; = improved results vs without it.
    &darr; = hurt results. &mdash; = fewer than 3 trades with this confluence for this pair.
  </p>
  {_confluence_per_pair_html(zone_trades, "r_realised")}
</div>

<!-- STRUCTURE -->
<div class="section">
  <h2>Performance by structure event type</h2>
  <p style="font-size:12px;color:#666;margin-bottom:10px;">
    Verify that Major events outperform Minor &mdash; they should, if detection is working.
  </p>
  {_structure_event_breakdown_html(zone_trades, "r_realised")}
</div>

<!-- LOSS ANALYSIS -->
<div class="section">
  <h2>What was different about the losing trades</h2>
  {_loss_analysis_html(zone_trades)}
</div>

<!-- VET REVIEW -->
<div class="section">
  <h2>Trades worth a second look</h2>
  {_vet_review_html(zone_trades)}
</div>
"""


# ---------------------------------------------------------------------------
# Main report writer
# ---------------------------------------------------------------------------

def write_h1_only_report(
    run_id: str,
    trades: List[Dict[str, Any]],
    raw_alerts: List[Dict[str, Any]],
    meta: Dict[str, Any],
    risk_usd: float = 250.0,
) -> Path:
    out_dir = Path(__file__).parent / "results" / run_id
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
                      and not t.get("ist_blocked")]
    blocked_trades = [t for t in trades_all if t.get("news_blocked")]
    ist_blocked_trades = [t for t in trades_all if t.get("ist_blocked")]

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
    }

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
  {_entry_comparison_html(sb_prox_tp2, sb_mid_tp2, fill_prox, fill_mid)}
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
</div>

</div><!-- /wrap -->
</body></html>"""

    (out_dir / "report.html").write_text(html, encoding="utf-8")
    return out_dir
