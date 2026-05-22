"""H1-only backtest report writer.

Produces a focused report for the H1-only experiment:
  - Side-by-side scoreboards: TP1-exit vs TP2-exit, proximal vs 50% entry.
  - Per-pair breakdown so structural bias (e.g. USDCHF anomaly) is visible.
  - Score-vs-winrate bucket table so the user can discover the optimal H1
    confidence threshold empirically (the whole point of running gateless).
  - Trades CSV with the full v1 column set.
  - JSON summary for machine consumption.

Lives separately from reporting.py because the H1-only report has different
shape (dual entries, TP1/TP2 split, no Phase 2 review flags) and the existing
reporting was getting cluttered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def _aggregate_for_exit(trades: List[Dict[str, Any]], r_col: str,
                        risk_usd: float) -> Dict[str, Any]:
    """Aggregate by hypothetical exit policy (r_col = r_if_exit_tp1 / r_if_exit_tp2 / r_realised).

    Only counts trades that filled (excludes never_filled rows).
    """
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
    if not filled:
        return {"trades": 0, "filled": 0}
    df = pd.DataFrame(filled)
    if r_col not in df.columns:
        return {"trades": 0, "filled": 0, "error": f"missing column {r_col}"}
    wins = df[df[r_col] > 0]
    losses = df[df[r_col] < 0]
    bes = df[df[r_col] == 0]
    expectancy = float(df[r_col].mean())
    return {
        "trades":         int(len(df)),
        "wins":           int(len(wins)),
        "losses":         int(len(losses)),
        "breakevens":     int(len(bes)),
        "win_rate_pct":   round(len(wins) / len(df) * 100, 1) if len(df) else 0,
        "expectancy_r":   round(expectancy, 3),
        "expectancy_usd": round(expectancy * risk_usd, 2),
        "total_r":        round(float(df[r_col].sum()), 3),
        "total_pnl_usd":  round(float(df[r_col].sum()) * risk_usd, 2),
        "avg_win_r":      round(float(wins[r_col].mean()), 3) if len(wins) else 0.0,
        "avg_loss_r":     round(float(losses[r_col].mean()), 3) if len(losses) else 0.0,
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


def _fill_rate(trades: List[Dict[str, Any]], entry_zone: str) -> Dict[str, Any]:
    """For 50pct entries especially: how often did the limit actually fill?"""
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
    """Bucket trades by score and compute win rate + expectancy per bucket.

    The whole purpose of running gateless: see at what score the system stops
    being profitable. Buckets: [0-2), [2-3), [3-4), [4-5), [5-6), [6-7), [7+].
    """
    filled = [t for t in trades if t.get("exit_reason") != "never_filled"]
    if not filled:
        return []
    df = pd.DataFrame(filled)
    if "score" not in df.columns or r_col not in df.columns:
        return []
    edges = [0, 2, 3, 4, 5, 6, 7, 99]
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


def _trades_csv(trades: List[Dict[str, Any]], path: Path) -> None:
    if not trades:
        pd.DataFrame([{"info": "no trades this run"}]).to_csv(path, index=False)
        return
    # Stable column order — what the user will see when they open the CSV.
    front_cols = [
        "pair", "alert_ts", "fill_ts", "exit_ts", "session",
        "direction", "event", "entry_zone",
        "entry", "sl_initial", "tp1", "tp2",
        "tp1_rr", "tp2_rr",
        "exit_reason", "exit_price",
        "r_realised", "r_if_exit_tp1", "r_if_exit_tp2", "pnl_usd",
        "mfe_r", "mae_r",
        "bars_to_exit", "bars_to_tp1", "bars_to_tp2",
        "ob_age_h1_bars", "pd_zone",
        "score", "structure_pts", "sweep_pts", "fvg_pts",
        "freshness_pts", "killzone_pts",
        "confluences_present",
        "sl_collision", "model",
        "ob_timestamp", "bos_tag", "bos_tier",
        "fvg_present", "sweep_present",
    ]
    df = pd.DataFrame(trades)
    cols_present = [c for c in front_cols if c in df.columns]
    rest = [c for c in df.columns if c not in cols_present]
    df = df[cols_present + rest]
    df.to_csv(path, index=False)


def _try_excel(trades: List[Dict[str, Any]], path: Path,
               summary: Dict[str, Any]) -> Optional[Path]:
    """Try to write an Excel mirror of the CSV. Returns path or None on failure."""
    if not trades:
        return None
    try:
        df = pd.DataFrame(trades)
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="Trades", index=False)
            pd.DataFrame([summary]).T.reset_index().to_excel(
                xw, sheet_name="Summary", index=False, header=["metric", "value"]
            )
        return path
    except Exception as e:
        print(f"  [excel write skipped: {e}]")
        return None


def _fmt_money(amount: float) -> str:
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):,.0f}"


def _scoreboard_html(name: str, agg: Dict[str, Any], risk_usd: float) -> str:
    if agg.get("trades", 0) == 0:
        return f"<h3>{name}</h3><p style='color:#888;'>No trades.</p>"
    return f"""
    <h3>{name}</h3>
    <p>
      <b>{agg['trades']} trades</b> &middot;
      <b>{agg['win_rate_pct']}% win rate</b> &middot;
      Expectancy: <b>{agg['expectancy_r']:+.2f}R</b>
      ({_fmt_money(agg['expectancy_usd'])} / trade)<br>
      Total: <b>{_fmt_money(agg['total_pnl_usd'])}</b>
      ({agg['wins']} wins, {agg['losses']} losses, {agg['breakevens']} BE)<br>
      Avg win: {agg['avg_win_r']:+.2f}R &middot;
      Avg loss: {agg['avg_loss_r']:+.2f}R
    </p>
    """


def _per_pair_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    body = "".join(
        f"<tr><td>{r['pair']}</td><td>{r['trades']}</td>"
        f"<td>{r['win_rate_pct']:.1f}%</td>"
        f"<td>{r['expectancy_r']:+.2f}R</td>"
        f"<td style='color:{'#27ae60' if r['total_pnl_usd']>=0 else '#e74c3c'};'>"
        f"{_fmt_money(r['total_pnl_usd'])}</td></tr>"
        for r in rows
    )
    return f"""
    <table>
      <thead><tr><th>Pair</th><th>Trades</th><th>Win rate</th>
        <th>Expectancy</th><th>Total P&amp;L</th></tr></thead>
      <tbody>{body}</tbody>
    </table>
    """


def _score_bucket_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<p style='color:#888;'>No trades to bucket.</p>"
    body = "".join(
        f"<tr><td>{r['score_bucket']}</td><td>{r['trades']}</td>"
        f"<td>{r['win_rate_pct']:.1f}%</td>"
        f"<td>{r['expectancy_r']:+.2f}R</td></tr>"
        for r in rows
    )
    return f"""
    <table>
      <thead><tr><th>Score</th><th>Trades</th><th>Win rate</th>
        <th>Expectancy</th></tr></thead>
      <tbody>{body}</tbody>
    </table>
    """


def write_h1_only_report(
    run_id: str,
    trades: List[Dict[str, Any]],
    raw_alerts: List[Dict[str, Any]],
    meta: Dict[str, Any],
    risk_usd: float = 250.0,
) -> Path:
    out_dir = Path(__file__).parent / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split trades by entry zone for the side-by-side comparison.
    prox_trades = [t for t in trades if t.get("entry_zone") == "proximal"]
    mid_trades  = [t for t in trades if t.get("entry_zone") == "50pct"]

    # 4 scoreboards: proximal-TP1, proximal-TP2, 50%-TP1, 50%-TP2.
    sb_prox_tp1 = _aggregate_for_exit(prox_trades, "r_if_exit_tp1", risk_usd)
    sb_prox_tp2 = _aggregate_for_exit(prox_trades, "r_if_exit_tp2", risk_usd)
    sb_mid_tp1  = _aggregate_for_exit(mid_trades,  "r_if_exit_tp1", risk_usd)
    sb_mid_tp2  = _aggregate_for_exit(mid_trades,  "r_if_exit_tp2", risk_usd)

    # Per-pair under default policy (TP2).
    pp_prox = _per_pair_breakdown(prox_trades, "r_if_exit_tp2", risk_usd)
    pp_mid  = _per_pair_breakdown(mid_trades,  "r_if_exit_tp2", risk_usd)

    # Score buckets — combined across both entry zones, default TP2 policy.
    score_buckets_tp1 = _score_buckets(trades, "r_if_exit_tp1")
    score_buckets_tp2 = _score_buckets(trades, "r_if_exit_tp2")

    fill_prox = _fill_rate(trades, "proximal")
    fill_mid  = _fill_rate(trades, "50pct")
    exit_counts = _exit_reason_counts(trades)

    summary = {
        "run_id":          run_id,
        "meta":            meta,
        "risk_per_trade_usd": risk_usd,
        "total_trade_rows":   len(trades),
        "fill_rate_proximal": fill_prox,
        "fill_rate_50pct":    fill_mid,
        "exit_reason_counts": exit_counts,
        "scoreboards": {
            "proximal_exit_tp1": sb_prox_tp1,
            "proximal_exit_tp2": sb_prox_tp2,
            "fifty_pct_exit_tp1": sb_mid_tp1,
            "fifty_pct_exit_tp2": sb_mid_tp2,
        },
        "per_pair_proximal_tp2": pp_prox,
        "per_pair_50pct_tp2":    pp_mid,
        "score_buckets_tp1":     score_buckets_tp1,
        "score_buckets_tp2":     score_buckets_tp2,
    }

    # CSV + Excel mirror + raw alerts.
    _trades_csv(trades, out_dir / "trades.csv")
    _try_excel(trades, out_dir / "trades.xlsx", summary)
    with open(out_dir / "raw_alerts.jsonl", "w") as f:
        for a in raw_alerts:
            f.write(json.dumps(a, default=str) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # HTML report.
    html = f"""<html><head><meta charset="utf-8"><style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            max-width: 900px; margin: 24px auto; color: #1a1a1a; padding: 0 16px; }}
    h1 {{ font-size: 22px; border-bottom: 2px solid #1a1a1a; padding-bottom: 8px; }}
    h2 {{ color: #1a5490; font-size: 18px; margin-top: 32px;
          border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
    h3 {{ font-size: 14px; color: #444; margin-top: 18px; margin-bottom: 6px; }}
    p {{ line-height: 1.55; font-size: 14px; }}
    table {{ font-size: 13px; margin-top: 6px; border-collapse: collapse; }}
    th, td {{ padding: 4px 12px; text-align: left; border-bottom: 1px solid #eee; }}
    th {{ background: #f4f4f8; }}
    .meta {{ background: #f4f4f8; padding: 10px 14px; border-radius: 6px;
             font-size: 13px; margin-bottom: 16px; }}
    .caveat {{ background: #fff8e6; border-left: 3px solid #f1c40f;
               padding: 10px 14px; border-radius: 4px; font-size: 13px; }}
    .twocol {{ display: flex; gap: 24px; }}
    .twocol > div {{ flex: 1; }}
    </style></head><body>

    <h1>H1-Only Backtest &mdash; {meta.get('start')} to {meta.get('end')}</h1>
    <div class="meta">
      <b>Mode:</b> H1-only (dual entry, no scoring gate) &middot;
      <b>1R = ${risk_usd:.0f}</b> &middot;
      <b>Pairs:</b> {', '.join(meta.get('pairs', []))} &middot;
      <b>Trade rows:</b> {len(trades)}
    </div>

    <div class="caveat">
      <b>What you're looking at:</b> Every H1 OB-touch fired a trade,
      regardless of confluence score. Each OB produced TWO rows: one with
      entry at the OB proximal edge, one with entry at the OB 50% mean.
      Both share the same SL (OB distal) and TP price levels (opposing H1
      liquidity), so the 50% entry has a tighter R-distance and naturally
      higher RR. <b>Use the score-vs-winrate table at the bottom</b> to
      figure out where the H1 system stops being profitable &mdash; that's
      the empirical min-confidence threshold. No spread, no slippage, no
      swap modelled. Same-bar SL+TP collision resolves SL-first.
    </div>

    <h2>Side-by-side scoreboards</h2>
    <p>Four cells: entry zone (proximal vs 50%) crossed with exit policy
       (TP1-only vs TP2 default). Compare them directly.</p>
    <div class="twocol">
      <div>
        <h3 style="color:#1a5490;">Entry: Proximal edge</h3>
        {_scoreboard_html('If you exited at TP1', sb_prox_tp1, risk_usd)}
        {_scoreboard_html('If you exited at TP2', sb_prox_tp2, risk_usd)}
      </div>
      <div>
        <h3 style="color:#1a5490;">Entry: 50% mean</h3>
        {_scoreboard_html('If you exited at TP1', sb_mid_tp1, risk_usd)}
        {_scoreboard_html('If you exited at TP2', sb_mid_tp2, risk_usd)}
      </div>
    </div>

    <h2>Fill rate</h2>
    <p>
      <b>Proximal:</b> {fill_prox['filled']}/{fill_prox['alerts']} alerts filled
      ({fill_prox['fill_rate_pct']:.1f}%) &mdash; should always be 100% by construction.<br>
      <b>50% mean:</b> {fill_mid['filled']}/{fill_mid['alerts']} alerts filled
      ({fill_mid['fill_rate_pct']:.1f}%) &mdash; misses are setups where price
      reversed before reaching the OB midpoint.
    </p>

    <h2>By pair &mdash; default TP2 exit policy</h2>
    <div class="twocol">
      <div>
        <h3>Proximal entry</h3>
        {_per_pair_table(pp_prox) or "<p style='color:#888;'>No trades.</p>"}
      </div>
      <div>
        <h3>50% mean entry</h3>
        {_per_pair_table(pp_mid) or "<p style='color:#888;'>No trades.</p>"}
      </div>
    </div>

    <h2>Score &times; win rate &mdash; the threshold finder</h2>
    <p>This is the headline diagnostic. Win rate and expectancy should rise
       monotonically with score. Where they stop rising or turn negative,
       that's a candidate min-confidence threshold for an H1-only live system.
       (Combines both entry zones.)</p>
    <div class="twocol">
      <div>
        <h3>If you exited at TP1</h3>
        {_score_bucket_table(score_buckets_tp1)}
      </div>
      <div>
        <h3>If you exited at TP2</h3>
        {_score_bucket_table(score_buckets_tp2)}
      </div>
    </div>

    <h2>Exit-reason breakdown</h2>
    <ul>
      {''.join(f'<li><b>{k}:</b> {v}</li>' for k, v in sorted(exit_counts.items()))}
    </ul>

    <h2>Files</h2>
    <ul>
      <li><b>trades.csv</b> &mdash; every trade row, full column set.</li>
      <li><b>trades.xlsx</b> &mdash; same as CSV but Excel-formatted (if available).</li>
      <li><b>summary.json</b> &mdash; all the numbers above, machine-readable.</li>
      <li><b>raw_alerts.jsonl</b> &mdash; OB-touch alerts before simulation.</li>
      <li><b>run_log.jsonl</b> + <b>console.log</b> &mdash; full run diagnostics.</li>
    </ul>

    </body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    return out_dir
