"""Excel + HTML report assembly. Two reports per run: Forex and NAS/XAU.

Format reviewed with vet: money first, plain English, actionable review list
near the top, caveats visible.

Produces:
  results/<run_id>/forex_trades.xlsx
  results/<run_id>/nas_xau_trades.xlsx
  results/<run_id>/report.html
  results/<run_id>/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd


FOREX_PAIRS = {"EURUSD", "NZDUSD", "USDJPY", "USDCHF"}
NAS_XAU_PAIRS = {"NAS100", "GOLD"}

# Vet review thresholds
MFE_OVER_BOOKED_X = 2.0          # winner left money if MFE > 2× booked R
NEAR_WINNER_MAE_R = 0.5          # loser was nearly a winner if reached > +0.5R


def _classify_session_utc(ts_iso: str) -> str:
    """Rough session classification from UTC timestamp.

    Asia: 00-07 UTC, London: 07-13 UTC, NY: 13-21 UTC, Other: 21-24 UTC.
    """
    try:
        h = pd.Timestamp(ts_iso).hour
    except Exception:
        return "Unknown"
    if 0 <= h < 7:
        return "Asia"
    if 7 <= h < 13:
        return "London"
    if 13 <= h < 21:
        return "NY"
    return "Other"


def _flag_vet_review(trade: Dict[str, Any]) -> Tuple[bool, str]:
    """Tag a trade if a vet would want to eyeball it."""
    r = trade.get("r_realised", 0)
    mfe = trade.get("mfe_r", 0)
    mae = trade.get("mae_r", 0)
    if r > 0 and mfe > r * MFE_OVER_BOOKED_X and mfe > 1.5:
        return True, f"left_money_MFE_{mfe:.1f}R_vs_booked_{r:.1f}R"
    if r < 0 and abs(mae) > 0 and mae > NEAR_WINNER_MAE_R:
        # mae stored as negative-magnitude in some conventions; we treat
        # the "reached above entry" magnitude as the positive MFE-side
        # while position is open and losing — best-effort using mfe field.
        if mfe > NEAR_WINNER_MAE_R:
            return True, f"nearly_winner_reached_{mfe:.1f}R_then_lost"
    return False, ""


def _aggregate(trades: List[Dict[str, Any]], risk_gbp: float) -> Dict[str, Any]:
    if not trades:
        return {"trades": 0}
    df = pd.DataFrame(trades)
    wins = df[df["r_realised"] > 0]
    losses = df[df["r_realised"] < 0]
    bes = df[df["r_realised"] == 0]
    expectancy = float(df["r_realised"].mean())
    total_pnl = float(df["pnl_gbp"].sum())
    avg_win_r = float(wins["r_realised"].mean()) if len(wins) else 0.0
    avg_loss_r = float(losses["r_realised"].mean()) if len(losses) else 0.0
    avg_win_mfe_r = float(wins["mfe_r"].mean()) if len(wins) else 0.0
    avg_loss_mae_r = float(losses["mae_r"].mean()) if len(losses) else 0.0
    capture_pct = round(avg_win_r / avg_win_mfe_r * 100, 0) if avg_win_mfe_r > 0 else 0
    return {
        "trades": int(len(df)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "breakevens": int(len(bes)),
        "win_rate": round(len(wins) / len(df) * 100, 1) if len(df) else 0,
        "expectancy_r": round(expectancy, 3),
        "expectancy_gbp": round(expectancy * risk_gbp, 2),
        "pnl_gbp": round(total_pnl, 2),
        "avg_win_r": round(avg_win_r, 3),
        "avg_loss_r": round(avg_loss_r, 3),
        "avg_win_gbp": round(avg_win_r * risk_gbp, 2),
        "avg_loss_gbp": round(avg_loss_r * risk_gbp, 2),
        "avg_win_mfe_r": round(avg_win_mfe_r, 3),
        "avg_win_mfe_gbp": round(avg_win_mfe_r * risk_gbp, 2),
        "avg_loss_mae_r": round(avg_loss_mae_r, 3),
        "tp_capture_pct": int(capture_pct),
        "sl_collisions": int(df["sl_collision"].sum()),
    }


def _per_pair(trades: List[Dict[str, Any]], risk_gbp: float) -> List[Dict[str, Any]]:
    if not trades:
        return []
    df = pd.DataFrame(trades)
    out = []
    for pair, sub in df.groupby("pair"):
        wins = sub[sub["r_realised"] > 0]
        out.append({
            "pair": pair,
            "trades": int(len(sub)),
            "pnl_gbp": round(float(sub["pnl_gbp"].sum()), 2),
            "win_rate": round(len(wins) / len(sub) * 100, 0) if len(sub) else 0,
        })
    out.sort(key=lambda r: r["pnl_gbp"], reverse=True)
    return out


def _per_session(trades: List[Dict[str, Any]], risk_gbp: float) -> List[Dict[str, Any]]:
    if not trades:
        return []
    df = pd.DataFrame(trades)
    df["session"] = df["alert_ts"].apply(_classify_session_utc)
    out = []
    for session, sub in df.groupby("session"):
        wins = sub[sub["r_realised"] > 0]
        out.append({
            "session": session,
            "trades": int(len(sub)),
            "pnl_gbp": round(float(sub["pnl_gbp"].sum()), 2),
            "win_rate": round(len(wins) / len(sub) * 100, 0) if len(sub) else 0,
        })
    order = {"London": 0, "NY": 1, "Asia": 2, "Other": 3, "Unknown": 4}
    out.sort(key=lambda r: order.get(r["session"], 9))
    return out


def _review_list(trades: List[Dict[str, Any]], risk_gbp: float) -> Dict[str, List[Dict[str, Any]]]:
    left_money = []
    nearly_won = []
    for t in trades:
        flag, reason = _flag_vet_review(t)
        if not flag:
            continue
        item = {
            "pair": t["pair"],
            "alert_ts": t["alert_ts"],
            "direction": t["direction"],
            "r_booked": t["r_realised"],
            "gbp_booked": t["pnl_gbp"],
            "mfe_r": t["mfe_r"],
            "mfe_gbp": round(t["mfe_r"] * risk_gbp, 0),
            "reason": reason,
        }
        if "left_money" in reason:
            left_money.append(item)
        elif "nearly_winner" in reason:
            nearly_won.append(item)
    return {"left_money": left_money, "nearly_won": nearly_won}


def _excel(trades: List[Dict[str, Any]], path: Path, summary: Dict[str, Any]) -> None:
    if not trades:
        pd.DataFrame([{"info": "no trades this run"}]).to_excel(path, index=False)
        return
    # Annotate trades with session + vet_review flag.
    for t in trades:
        t["session"] = _classify_session_utc(t.get("alert_ts", ""))
        flag, reason = _flag_vet_review(t)
        t["vet_review"] = flag
        t["vet_review_reason"] = reason
    df = pd.DataFrame(trades)
    # Reorder to put most useful columns left.
    front = [c for c in [
        "pair", "alert_ts", "session", "direction", "bias",
        "r_realised", "pnl_gbp", "mfe_r", "mae_r",
        "exit_reason", "tp1_hit", "vet_review", "vet_review_reason",
        "score", "model", "hold_minutes",
        "entry", "sl_initial", "tp1", "tp2", "exit_price",
        "bos_tag", "bos_tier", "fvg_present", "sweep_present",
    ] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="Trades", index=False)
            pd.DataFrame([summary]).T.reset_index().to_excel(
                xw, sheet_name="Summary", index=False, header=["metric", "value"]
            )
    except Exception as e:
        print(f"  [excel fail, writing csv]: {e}")
        df.to_csv(path.with_suffix(".csv"), index=False)


def _fmt_money(amount: float) -> str:
    sign = "+" if amount >= 0 else "-"
    return f"{sign}£{abs(amount):,.0f}"


def _section(name: str, trades: List[Dict[str, Any]], summary: Dict[str, Any],
             risk_gbp: float) -> str:
    if summary.get("trades", 0) == 0:
        return f"""
        <h2>{name}</h2>
        <p style="color:#888;">No trades this run.</p>
        """

    pp = _per_pair(trades, risk_gbp)
    ps = _per_session(trades, risk_gbp)

    # Plain-English exit analysis
    exit_block = ""
    if summary["wins"] > 0:
        verdict = ""
        cap = summary["tp_capture_pct"]
        if cap >= 80:
            verdict = "→ TPs look well-placed."
        elif cap >= 60:
            verdict = "→ TPs slightly conservative. See WHAT TO REVIEW below."
        else:
            verdict = "→ TPs too tight. Significant money left on the table."
        exit_block += f"""
        <p><b>When you won:</b> price kept going on average to
           <b>{_fmt_money(summary['avg_win_mfe_gbp'])}</b> before reversing.
           You booked <b>{_fmt_money(summary['avg_win_gbp'])}</b>.
           You captured roughly <b>{summary['tp_capture_pct']}%</b> of the available move.
           {verdict}</p>
        """
    if summary["losses"] > 0:
        mae = summary["avg_loss_mae_r"]
        if abs(mae) >= 0.85:
            mae_verdict = "→ Stops hit cleanly — not hunted by wicks."
        elif abs(mae) >= 0.5:
            mae_verdict = "→ Some stops hit close to full SL. Acceptable."
        else:
            mae_verdict = "→ Many trades stopped out shallow. SL may be too tight."
        exit_block += f"""
        <p><b>When you lost:</b> price pushed an average
           <b>{_fmt_money(summary['avg_loss_mae_r'] * risk_gbp)}</b> against you
           before stopping out at <b>{_fmt_money(summary['avg_loss_gbp'])}</b>.
           {mae_verdict}</p>
        """

    # Per-pair table
    pair_rows = "".join(
        f"<tr><td>{r['pair']}</td><td>{r['trades']}</td>"
        f"<td style='color:{'#27ae60' if r['pnl_gbp']>=0 else '#e74c3c'};'>{_fmt_money(r['pnl_gbp'])}</td>"
        f"<td>{r['win_rate']:.0f}%</td></tr>"
        for r in pp
    )
    pair_table = f"""
        <h3>By pair</h3>
        <table style="border-collapse:collapse;">
          <thead><tr><th>Pair</th><th>Trades</th><th>P&amp;L</th><th>Win rate</th></tr></thead>
          <tbody>{pair_rows}</tbody>
        </table>
    """

    # Per-session table
    sess_rows = "".join(
        f"<tr><td>{r['session']}</td><td>{r['trades']}</td>"
        f"<td style='color:{'#27ae60' if r['pnl_gbp']>=0 else '#e74c3c'};'>{_fmt_money(r['pnl_gbp'])}</td>"
        f"<td>{r['win_rate']:.0f}%</td></tr>"
        for r in ps
    )
    sess_table = f"""
        <h3>By session</h3>
        <table style="border-collapse:collapse;">
          <thead><tr><th>Session</th><th>Trades</th><th>P&amp;L</th><th>Win rate</th></tr></thead>
          <tbody>{sess_rows}</tbody>
        </table>
    """

    return f"""
    <h2>{name}</h2>
    <p style="font-size:15px;">
      <b>Money:</b> {_fmt_money(summary['pnl_gbp'])} over {summary['trades']} trades<br>
      <b>Per trade:</b> {_fmt_money(summary['expectancy_gbp'])} average
         ({summary['expectancy_r']:+.2f}R)<br>
      <b>Won:</b> {summary['wins']} trades · avg win {_fmt_money(summary['avg_win_gbp'])}<br>
      <b>Lost:</b> {summary['losses']} trades · avg loss {_fmt_money(summary['avg_loss_gbp'])}
    </p>

    <h3>Were exits the right size?</h3>
    {exit_block}

    {pair_table}
    {sess_table}
    """


def _review_section(forex_trades, nas_trades, risk_gbp: float) -> str:
    all_t = forex_trades + nas_trades
    review = _review_list(all_t, risk_gbp)
    if not review["left_money"] and not review["nearly_won"]:
        return """
        <h2>What to review</h2>
        <p>Nothing flagged this week. Wins took proper R, losses stopped at proper R.</p>
        """
    blocks = []
    if review["left_money"]:
        items = "".join(
            f"<li>{r['pair']} {r['direction']} {r['alert_ts'][:16]} — "
            f"booked {_fmt_money(r['gbp_booked'])} ({r['r_booked']:+.1f}R), "
            f"price went to {_fmt_money(r['mfe_gbp'])} ({r['mfe_r']:+.1f}R)</li>"
            for r in review["left_money"]
        )
        blocks.append(f"""
            <p><b>{len(review['left_money'])} trades where you left money on the table</b>
               (MFE was more than 2× what you booked):</p>
            <ul>{items}</ul>
        """)
    if review["nearly_won"]:
        items = "".join(
            f"<li>{r['pair']} {r['direction']} {r['alert_ts'][:16]} — "
            f"reached {r['mfe_r']:+.1f}R, then reversed to full SL</li>"
            for r in review["nearly_won"]
        )
        blocks.append(f"""
            <p><b>{len(review['nearly_won'])} trades that were nearly winners</b>
               then stopped out:</p>
            <ul>{items}</ul>
        """)
    return f"<h2>What to review</h2>{''.join(blocks)}"


def write_report(
    run_id: str,
    trades: List[Dict[str, Any]],
    raw_alerts: List[Dict[str, Any]],
    meta: Dict[str, Any],
    risk_gbp: float = 250.0,
) -> Path:
    out_dir = Path(__file__).parent / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    forex_trades = [t for t in trades if t["pair"] in FOREX_PAIRS]
    nas_trades = [t for t in trades if t["pair"] in NAS_XAU_PAIRS]

    fx_sum = _aggregate(forex_trades, risk_gbp)
    nx_sum = _aggregate(nas_trades, risk_gbp)

    _excel(forex_trades, out_dir / "forex_trades.xlsx", fx_sum)
    _excel(nas_trades, out_dir / "nas_xau_trades.xlsx", nx_sum)

    with open(out_dir / "raw_alerts.jsonl", "w") as f:
        for a in raw_alerts:
            f.write(json.dumps(a, default=str) + "\n")

    summary_full = {
        "run_id": run_id,
        "meta": meta,
        "forex": fx_sum,
        "nas_xau": nx_sum,
        "risk_per_trade_gbp": risk_gbp,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_full, f, indent=2, default=str)

    fx_section = _section("Forex", forex_trades, fx_sum, risk_gbp)
    nx_section = _section("NAS100 + Gold", nas_trades, nx_sum, risk_gbp)
    review_section = _review_section(forex_trades, nas_trades, risk_gbp)

    html = f"""<html><head><meta charset="utf-8"><style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            max-width: 820px; margin: 24px auto; color: #1a1a1a; padding: 0 16px; }}
    h1 {{ font-size: 22px; border-bottom: 2px solid #1a1a1a; padding-bottom: 8px; margin-bottom: 4px; }}
    h2 {{ color: #1a5490; font-size: 18px; margin-top: 32px;
          border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
    h3 {{ font-size: 14px; color: #444; margin-top: 18px; margin-bottom: 6px; }}
    p {{ line-height: 1.55; font-size: 14px; }}
    ul {{ font-size: 14px; line-height: 1.55; }}
    table {{ font-size: 13px; margin-top: 6px; }}
    th, td {{ padding: 4px 12px; text-align: left; border-bottom: 1px solid #eee; }}
    th {{ background: #f4f4f8; }}
    .meta {{ background: #f4f4f8; padding: 10px 14px; border-radius: 6px;
             font-size: 13px; margin-bottom: 16px; }}
    .caveat {{ background: #fff8e6; border-left: 3px solid #f1c40f;
               padding: 10px 14px; border-radius: 4px; font-size: 13px; }}
    </style></head><body>

    <h1>Backtest Result &mdash; {meta.get('start')} to {meta.get('end')}</h1>
    <div class="meta">
      <b>Regime:</b> {meta.get('regime', 'unspecified')} &middot;
      <b>1R = £{risk_gbp:.0f}</b> &middot;
      <b>Pairs:</b> {', '.join(meta.get('pairs', []))}
    </div>

    <div class="caveat">
      <b>Before you read the numbers:</b>
      No spread, no slippage, no swap modelled. Real P&amp;L will likely be
      5-10% lower than shown. Same-bar SL+TP collisions resolve as SL hit
      first (pessimistic). Data is yfinance — broker bars may differ slightly.
    </div>

    {fx_section}
    {nx_section}
    {review_section}

    <h2>Files attached</h2>
    <ul>
      <li><b>forex_trades.xlsx</b> &mdash; every Forex trade, one row each.
          The <code>vet_review</code> column flags ones worth eyeballing.</li>
      <li><b>nas_xau_trades.xlsx</b> &mdash; same for NAS100 + Gold.</li>
      <li><b>summary.json</b> &mdash; raw metrics for further analysis.</li>
    </ul>

    </body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    return out_dir
