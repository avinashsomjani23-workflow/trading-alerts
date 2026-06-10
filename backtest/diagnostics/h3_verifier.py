"""Read-only fill verifier for Harness 3 (APPROVED by user).

Re-checks every simulated trade row against RAW OHLC bars, independently of the
simulator. Imports no trading-path code, produces no trades, places nothing.
Its only job is to confirm the simulator's claimed fills/exits are consistent
with the bars and free of look-ahead (FABLE_REFERENCE Â§6 guards 7-12).

Why this exists: you cannot prove the simulator's fill rules by calling the
simulator on its own output (code grading its own homework). This module reads
the bars directly and asserts the documented rules held.

All timestamps in trade rows are ISO strings; bars are a UTC-indexed df.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

RECON_TOL = 1e-6  # absolute USD tolerance for pnl reconciliation


@dataclass
class VerifyResult:
    n_rows: int = 0
    n_filled: int = 0
    n_never_filled: int = 0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    # Pessimism / behaviour statistics (NOT violations â documented design).
    fill_bar_tp_suppressed: int = 0      # guard 9 bit
    same_bar_sl_tp_collision: int = 0    # guard 10 bit
    recon_violations: int = 0

    def add(self, row, check, detail):
        self.violations.append({
            "pair": row.get("pair"),
            "alert_ts": row.get("alert_ts"),
            "entry_zone": row.get("entry_zone"),
            "ob_timestamp": row.get("ob_timestamp"),
            "check": check,
            "detail": detail,
        })


def _ts(v) -> Optional[pd.Timestamp]:
    if v is None or v == "":
        return None
    t = pd.Timestamp(v)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t


def verify_trade_rows(df: pd.DataFrame, rows: List[Dict[str, Any]],
                      risk_usd: float = 250.0) -> VerifyResult:
    """Verify every trade row against raw bars. Returns a VerifyResult whose
    `violations` list is empty when all guards held."""
    res = VerifyResult()
    for row in rows:
        res.n_rows += 1
        exit_reason = row.get("exit_reason")
        bias = row.get("bias")
        alert_ts = _ts(row.get("alert_ts"))
        entry = row.get("entry")
        sl = row.get("sl_initial")
        tp1 = row.get("tp1")
        tp2 = row.get("tp2")

        # --- Reconciliation (every row) -----------------------------------
        r = row.get("r_realised")
        pnl = row.get("pnl_usd")
        if r is not None and pnl is not None:
            if abs(pnl - r * risk_usd) > max(RECON_TOL, abs(r * risk_usd) * 1e-9):
                res.recon_violations += 1
                res.add(row, "reconciliation",
                        f"pnl_usd={pnl} != r_realised*risk={r * risk_usd}")

        # --- never_filled rows --------------------------------------------
        if exit_reason == "never_filled":
            res.n_never_filled += 1
            if row.get("fill_ts"):
                res.add(row, "never_filled_has_fill", f"fill_ts={row.get('fill_ts')}")
            if r not in (0, 0.0, None):
                res.add(row, "never_filled_nonzero_r", f"r_realised={r}")
            # Assert no bar in the pend window actually touched entry.
            if alert_ts is not None and entry is not None:
                pend = df.loc[alert_ts + pd.Timedelta(hours=1):]
                touched = _first_fill_bar(pend, bias, entry)
                if touched is not None:
                    res.add(row, "never_filled_but_touchable",
                            f"entry {entry} was touched at {touched} within data")
            continue

        # --- filled rows ---------------------------------------------------
        res.n_filled += 1
        fill_ts = _ts(row.get("fill_ts"))
        exit_ts = _ts(row.get("exit_ts"))
        exit_price = row.get("exit_price")

        # Guard 7: fill bar opens at or after alert_ts + 1h (alert bar excluded).
        if fill_ts is None:
            res.add(row, "filled_missing_fill_ts", "exit_reason filled but fill_ts None")
            continue
        if alert_ts is not None and fill_ts < alert_ts + pd.Timedelta(hours=1):
            res.add(row, "guard7_fill_before_alert_plus_1h",
                    f"fill_ts={fill_ts} < alert_ts+1h={alert_ts + pd.Timedelta(hours=1)}")

        # Guard 8: fill bar satisfies the limit condition; no EARLIER bar did.
        if alert_ts is not None and entry is not None:
            pend = df.loc[alert_ts + pd.Timedelta(hours=1):]
            first = _first_fill_bar(pend, bias, entry)
            if first is None:
                res.add(row, "guard8_no_qualifying_bar",
                        f"claimed fill {fill_ts} but no bar satisfies entry {entry}")
            elif first != fill_ts:
                res.add(row, "guard8_not_first_fill",
                        f"claimed fill {fill_ts}, earliest qualifying bar {first}")

        # Ordering: exit at or after fill.
        if exit_ts is not None and fill_ts is not None and exit_ts < fill_ts:
            res.add(row, "exit_before_fill", f"exit_ts={exit_ts} < fill_ts={fill_ts}")

        # Exit price legality: within the exit bar's [low, high] (small tol).
        if exit_ts is not None and exit_price is not None and exit_ts in df.index:
            bar = df.loc[exit_ts]
            lo, hi = float(bar["Low"]), float(bar["High"])
            tol = (hi - lo) * 1e-6 + 1e-9
            if not (lo - tol <= exit_price <= hi + tol):
                res.add(row, "exit_price_out_of_bar",
                        f"exit_price={exit_price} not in [{lo},{hi}] at {exit_ts}")

        # Guard 9/10 statistics on the fill bar (documented pessimism, not bugs).
        if fill_ts is not None and fill_ts in df.index:
            fb = df.loc[fill_ts]
            fhi, flo = float(fb["High"]), float(fb["Low"])
            if bias == "LONG":
                tp_on_fill = (tp1 is not None and fhi >= tp1)
                sl_on_fill = (sl is not None and flo <= sl)
            else:
                tp_on_fill = (tp1 is not None and flo <= tp1)
                sl_on_fill = (sl is not None and fhi >= sl)
            if tp_on_fill:
                res.fill_bar_tp_suppressed += 1
            if tp_on_fill and sl_on_fill:
                res.same_bar_sl_tp_collision += 1

        # Sign sanity: SL exits should not be strongly positive; tp2 positive.
        if exit_reason == "sl" and r is not None and r > 0.05:
            res.add(row, "sl_exit_positive_r", f"exit_reason=sl but r_realised={r}")
        if exit_reason == "tp2" and r is not None and r <= 0:
            res.add(row, "tp2_exit_nonpositive_r", f"exit_reason=tp2 but r_realised={r}")

    return res


def _first_fill_bar(pend: pd.DataFrame, bias: str,
                    entry: float) -> Optional[pd.Timestamp]:
    """First bar in `pend` whose range crosses the entry (long: low<=entry;
    short: high>=entry). Returns the bar's timestamp or None."""
    if pend is None or pend.empty or entry is None:
        return None
    if bias == "LONG":
        hit = pend.index[pend["Low"].to_numpy() <= entry]
    else:
        hit = pend.index[pend["High"].to_numpy() >= entry]
    return hit[0] if len(hit) else None
