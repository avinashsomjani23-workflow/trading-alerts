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
# Pair × Session 2D grid
# ---------------------------------------------------------------------------

_SESSION_ORDER = ["Asia", "London", "NY", "Other"]

def _pair_session_grid_html(trades: List[Dict[str, Any]], r_col: str) -> str:
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
    if not filled:
        return "<p style='color:#888;'>No filled trades.</p>"
    df = pd.DataFrame(filled)
    if "pair" not in df.columns or "session" not in df.columns or r_col not in df.columns:
        return "<p style='color:#888;'>Session or pair data missing.</p>"

    pairs    = sorted(df["pair"].unique())
    sessions = [s for s in _SESSION_ORDER if s in df["session"].unique()]

    def _cell(sub: pd.DataFrame) -> str:
        n = len(sub)
        if n < 3:
            return f"<td style='color:#bbb;text-align:center;'>—<br><small>{n}t</small></td>"
        wr  = (sub[r_col] > 0).mean() * 100
        exp = sub[r_col].mean()
        if wr >= 50:
            bg = "#eafaf1"
        elif wr >= 40:
            bg = "#fef9e7"
        else:
            bg = "#fdf2f2"
        sign = "+" if exp >= 0 else ""
        return (f"<td style='background:{bg};text-align:center;'>"
                f"<b>{wr:.0f}%</b><br>"
                f"<small>{sign}{exp:.2f}R &middot; {n}t</small></td>")

    header = "<tr><th>Pair</th>" + "".join(f"<th>{s}</th>" for s in sessions) + "</tr>"
    rows   = ""
    for pair in pairs:
        pair_df = df[df["pair"] == pair]
        row = f"<tr><td><b>{pair}</b></td>"
        for sess in sessions:
            sub = pair_df[pair_df["session"] == sess]
            row += _cell(sub)
        row += "</tr>"
        rows += row

    legend = ("<p style='font-size:11px;color:#888;margin-top:6px;'>"
              "🟢 ≥ 50% win rate &nbsp; 🟡 40–49% &nbsp; 🔴 &lt; 40% &nbsp;"
              "— = fewer than 3 trades (not enough to conclude)</p>")
    return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>{legend}"


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

    prox_trades = [t for t in trades if t.get("entry_zone") == "proximal"]
    mid_trades  = [t for t in trades if t.get("entry_zone") == "50pct"]

    # Primary view: proximal entry, TP2 default policy.
    sb_prox_tp2 = _aggregate_for_exit(prox_trades, "r_if_exit_tp2", risk_usd)
    sb_mid_tp2  = _aggregate_for_exit(mid_trades,  "r_if_exit_tp2", risk_usd)
    sb_prox_tp1 = _aggregate_for_exit(prox_trades, "r_if_exit_tp1", risk_usd)
    sb_mid_tp1  = _aggregate_for_exit(mid_trades,  "r_if_exit_tp1", risk_usd)

    pp_prox   = _per_pair_breakdown(prox_trades,  "r_if_exit_tp2", risk_usd)
    ss_prox   = _per_session_breakdown(prox_trades, "r_if_exit_tp2", risk_usd)
    fill_prox = _fill_rate(trades, "proximal")
    fill_mid  = _fill_rate(trades, "50pct")

    score_buckets = _score_buckets(prox_trades, "r_if_exit_tp2")
    # Exit counts must be scoped per entry-zone — the headline counts proximal
    # trades only, so mixing zones here makes the "How trades closed" line
    # contradict the headline (e.g. 21 filled but 29 SL hits across both zones).
    exit_counts_prox = _exit_reason_counts(prox_trades)
    exit_counts_mid  = _exit_reason_counts(mid_trades)

    summary = {
        "run_id":              run_id,
        "meta":                meta,
        "risk_per_trade_usd":  risk_usd,
        "total_trade_rows":    len(trades),
        "fill_rate_proximal":  fill_prox,
        "fill_rate_50pct":     fill_mid,
        "exit_reason_counts_proximal": exit_counts_prox,
        "exit_reason_counts_50pct":    exit_counts_mid,
        "scoreboards": {
            "proximal_exit_tp1":  sb_prox_tp1,
            "proximal_exit_tp2":  sb_prox_tp2,
            "fifty_pct_exit_tp1": sb_mid_tp1,
            "fifty_pct_exit_tp2": sb_mid_tp2,
        },
        "per_pair_proximal_tp2": pp_prox,
        "per_pair_50pct_tp2":    _per_pair_breakdown(mid_trades, "r_if_exit_tp2", risk_usd),
        "score_buckets_tp1":     _score_buckets(prox_trades, "r_if_exit_tp1"),
        "score_buckets_tp2":     score_buckets,
    }

    # Files.
    _trades_csv(trades, out_dir / "trades.csv")
    excel_ok = _try_excel(trades, out_dir / "trades.xlsx") is not None
    with open(out_dir / "raw_alerts.jsonl", "w") as f:
        for a in raw_alerts:
            f.write(json.dumps(a, default=str) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # --- HTML ---
    n          = sb_prox_tp2.get("trades", 0)
    total_pnl  = sb_prox_tp2.get("total_pnl_usd", 0)
    exp_r      = sb_prox_tp2.get("expectancy_r", 0)
    wr         = sb_prox_tp2.get("win_rate_pct", 0)
    headline   = _week_headline(sb_prox_tp2)
    pnl_color  = "#27ae60" if total_pnl >= 0 else "#e74c3c"
    pairs_str  = ", ".join(meta.get("pairs", []))
    regime_str = meta.get("regime", "")

    exit_reason_plain = {
        "sl": "Stop loss", "tp1": "TP1 hit", "tp2": "TP2 hit",
        "timeout": "Time limit", "window_end": "End of window",
        "sl_collision": "SL+TP same bar", "never_filled": "Never filled",
    }
    # Email shows the proximal breakdown only — same population as the
    # headline (n filled trades, wins, losses, narrative). The 50pct breakdown
    # lives in summary.json under exit_reason_counts_50pct for offline review.
    exit_breakdown = " &middot; ".join(
        f"<b>{exit_reason_plain.get(k, k)}: {v}</b>"
        for k, v in sorted(exit_counts_prox.items())
    )

    score_verdict = _score_verdict_text(score_buckets)

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
  .headline .big {{ font-size: 32px; font-weight: 700; color: {pnl_color}; }}
  .headline .sub {{ font-size: 14px; color: #555; margin-top: 4px; }}
  .headline .verdict {{ font-size: 13px; color: #444; margin-top: 10px;
                        background: #f8f9fa; border-left: 3px solid {pnl_color};
                        padding: 8px 12px; border-radius: 0 4px 4px 0; }}

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

<!-- HEADLINE -->
<div class="headline">
  <div class="big">{_m(total_pnl)}</div>
  <div class="sub">
    {n} filled trades &nbsp;&middot;&nbsp;
    {wr:.0f}% won &nbsp;&middot;&nbsp;
    avg {_r(exp_r)} per trade
  </div>
  <div class="verdict">{headline}</div>
</div>

<!-- SECTION 1: WHAT HAPPENED -->
<div class="section">
  <h2>What happened this week</h2>
  <div class="stats-strip">
    <div class="stat">
      <div class="val" style="color:#27ae60;">{sb_prox_tp2.get('wins', 0)}</div>
      <div class="lbl">Wins</div>
    </div>
    <div class="stat">
      <div class="val" style="color:#e74c3c;">{sb_prox_tp2.get('losses', 0)}</div>
      <div class="lbl">Losses</div>
    </div>
    <div class="stat">
      <div class="val" style="color:#888;">{sb_prox_tp2.get('breakevens', 0)}</div>
      <div class="lbl">Break-evens</div>
    </div>
    <div class="stat">
      <div class="val">{_r(sb_prox_tp2.get('avg_win_r', 0))}</div>
      <div class="lbl">Avg win</div>
    </div>
    <div class="stat">
      <div class="val">{_r(sb_prox_tp2.get('avg_loss_r', 0))}</div>
      <div class="lbl">Avg loss</div>
    </div>
  </div>
  {_exit_narrative(sb_prox_tp2, risk_usd)}
  <p style="font-size:12px;color:#888;margin-top:4px;">
    How trades closed: {exit_breakdown}
  </p>
</div>

<!-- SECTION 2: WHERE IT WORKED — pair × session grid -->
<div class="section">
  <h2>Where it worked — pair × session (proximal entry)</h2>
  <p style="font-size:12px;color:#666;margin-bottom:10px;">
    Each cell shows win rate / avg R / trade count for that pair + session combination.
    Cells with fewer than 3 trades are shown as — (not enough data).
  </p>
  {_pair_session_grid_html(prox_trades, "r_if_exit_tp2")}
  <p style="font-size:11px;color:#aaa;margin-top:10px;">
    Day of week is tracked in the Excel (Zone Register tab) and will be analysed
    in the aggregate report after enough runs accumulate.
  </p>
</div>

<!-- SECTION 3: SCORE -->
<div class="section">
  <h2>Did the confidence score predict better trades?</h2>
  <p><b>{score_verdict}</b></p>
  {_score_table_html(score_buckets)}
  <p style="font-size:12px;color:#888;margin-top:8px;">
    Score rises with each confluence present: FVG, liquidity sweep, kill zone,
    PD alignment, OB freshness, structure tier.
  </p>
</div>

<!-- SECTION 4: CONFLUENCE PER PAIR -->
<div class="section">
  <h2>Which confluences actually helped — by pair</h2>
  <p style="font-size:12px;color:#666;margin-bottom:10px;">
    Win rate when each confluence was present. ↑ = improved results vs without it.
    ↓ = hurt results. — = fewer than 3 trades with this confluence for this pair.
  </p>
  {_confluence_per_pair_html(prox_trades, "r_if_exit_tp2")}
</div>

<!-- SECTION 5: STRUCTURE EVENT PERFORMANCE -->
<div class="section">
  <h2>Performance by structure event type</h2>
  <p style="font-size:12px;color:#666;margin-bottom:10px;">
    The system detects four kinds of structural events. Each carries different reliability.
    Use this table to verify that Major events outperform Minor — they should, if detection is working.
  </p>
  {_structure_event_breakdown_html(prox_trades, "r_if_exit_tp2")}
</div>

<!-- SECTION 6: WHY WE LOST -->
<div class="section">
  <h2>What was different about the losing trades</h2>
  {_loss_analysis_html(prox_trades)}
</div>

<!-- SECTION 6: PROXIMAL vs 50% -->
<div class="section">
  <h2>Proximal entry vs 50% midpoint entry</h2>
  {_entry_comparison_html(sb_prox_tp2, sb_mid_tp2, fill_prox, fill_mid)}
</div>

<!-- SECTION 7: VET REVIEW -->
<div class="section">
  <h2>Trades worth a second look</h2>
  {_vet_review_html(prox_trades)}
</div>

<!-- SECTION 8: FILES -->
<div class="section">
  <h2>What's attached</h2>
  <ul style="padding-left:18px;font-size:13px;">
    <li><b>trades.xlsx — Trades tab:</b> {"every filled trade, plain-English headers, day of week, color-coded P&L, amber highlights on flagged trades" if excel_ok else "<span style='color:#e74c3c;'>FAILED — openpyxl not installed. Use trades.csv instead.</span>"}</li>
    <li><b>trades.xlsx — Zone Register tab:</b> one row per OB, both entry zones side by side — use this to verify entry/SL/TP levels and fill logic</li>
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
