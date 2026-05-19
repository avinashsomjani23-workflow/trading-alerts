"""Excel + HTML report assembly. Two reports per run: Forex and NAS/XAU.

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
from typing import List, Dict, Any

import pandas as pd


FOREX_PAIRS = {"EURUSD", "NZDUSD", "USDJPY", "USDCHF"}
NAS_XAU_PAIRS = {"NAS100", "GOLD"}


def _aggregate(trades: List[Dict[str, Any]], risk_gbp: float) -> Dict[str, Any]:
    if not trades:
        return {"trades": 0, "win_rate": 0, "expectancy_r": 0, "pnl_gbp": 0}
    df = pd.DataFrame(trades)
    wins = df[df["r_realised"] > 0]
    losses = df[df["r_realised"] < 0]
    bes = df[df["r_realised"] == 0]
    expectancy = df["r_realised"].mean()
    total_pnl = df["pnl_gbp"].sum()
    return {
        "trades": int(len(df)),
        "win_rate": round(len(wins) / len(df) * 100, 1) if len(df) else 0,
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "breakevens": int(len(bes)),
        "expectancy_r": round(float(expectancy), 3),
        "expectancy_gbp": round(float(expectancy) * risk_gbp, 2),
        "pnl_gbp": round(float(total_pnl), 2),
        "avg_win_r": round(float(wins["r_realised"].mean()), 3) if len(wins) else 0,
        "avg_loss_r": round(float(losses["r_realised"].mean()), 3) if len(losses) else 0,
        "avg_mfe_r": round(float(df["mfe_r"].mean()), 3),
        "avg_mae_r": round(float(df["mae_r"].mean()), 3),
        "sl_collisions": int(df["sl_collision"].sum()),
        "tp_capture_pct": round(
            float(wins["r_realised"].mean() / wins["mfe_r"].mean() * 100), 1
        ) if len(wins) and wins["mfe_r"].mean() > 0 else 0,
    }


def _excel(trades: List[Dict[str, Any]], path: Path, summary: Dict[str, Any]) -> None:
    if not trades:
        # Still write empty file so artifact upload doesn't fail.
        pd.DataFrame([{"info": "no trades"}]).to_excel(path, index=False)
        return
    df = pd.DataFrame(trades)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="Trades", index=False)
            pd.DataFrame([summary]).T.reset_index().to_excel(
                xw, sheet_name="Summary", index=False, header=["metric", "value"]
            )
    except Exception as e:
        # Fallback to CSV if openpyxl missing.
        print(f"  [excel fail, writing csv]: {e}")
        df.to_csv(path.with_suffix(".csv"), index=False)


def _html_block(name: str, summary: Dict[str, Any], risk_gbp: float) -> str:
    if summary["trades"] == 0:
        return f"<h2>{name}</h2><p>No trades.</p>"
    return f"""
    <h2>{name}</h2>
    <p><b>Headline:</b> {summary['trades']} trades · expectancy
       {summary['expectancy_r']:+.2f}R (£{summary['expectancy_gbp']:+,.0f}) ·
       win rate {summary['win_rate']}% ·
       total P&amp;L £{summary['pnl_gbp']:+,.0f} (1R = £{risk_gbp:.0f})</p>
    <p><b>TP/SL diagnostics (plain English):</b><br>
       Winners on average reached {summary['avg_mfe_r']:.2f}R before reversing;
       you captured {summary['tp_capture_pct']:.0f}% of the available move.
       Average loser MAE: {summary['avg_mae_r']:.2f}R
       (closer to -1.0 = clean stops; closer to 0 = SL hunted).</p>
    <p><b>Other:</b> avg win {summary['avg_win_r']:.2f}R · avg loss
       {summary['avg_loss_r']:.2f}R · SL/TP same-bar collisions: {summary['sl_collisions']}</p>
    """


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

    html = f"""<html><head><style>
    body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 20px auto; color: #222; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
    h2 {{ color: #1a5490; margin-top: 28px; }}
    p {{ line-height: 1.55; }}
    .meta {{ background: #f4f4f8; padding: 10px 14px; border-radius: 6px; font-size: 13px; }}
    </style></head><body>
    <h1>Backtest Report — {run_id}</h1>
    <div class="meta">
      Window: {meta.get('start')} → {meta.get('end')} ·
      Regime: {meta.get('regime', 'unspecified')} ·
      Pairs: {', '.join(meta.get('pairs', []))} ·
      Risk per trade: £{risk_gbp:.0f}
    </div>
    {_html_block("Forex", fx_sum, risk_gbp)}
    {_html_block("NAS100 + Gold", nx_sum, risk_gbp)}
    <h2>Caveats</h2>
    <p>Mechanical grading only. yfinance data has no spread / slippage /
       weekend gap modelling. See KNOWN_LIMITATIONS.md.</p>
    </body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    return out_dir
