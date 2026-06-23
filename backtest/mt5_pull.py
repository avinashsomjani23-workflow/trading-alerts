"""
MT5 PULL  (run on YOUR Windows machine, MetaTrader5 terminal OPEN + logged into
Funding Pips). Read-only: downloads H1 history for our 6 instruments and saves raw
CSVs into backtest/mt5_data/. It does NOT trade. Claude reads the CSVs afterward and
handles timezone conversion + backtest formatting.

Run the .bat (double-click), or:  py backtest/mt5_pull.py

Login: by default this attaches to the already-running, logged-in terminal.
If that fails with "Authorization failed", create backtest/mt5_login.txt
(copy mt5_login.example.txt) with login / password / server / path filled in.
Use the read-only INVESTOR password, never the master password.
"""
import os
import csv
from datetime import datetime, timezone

try:
    import MetaTrader5 as mt5
except Exception as e:
    raise SystemExit(f"MetaTrader5 not installed ({e}). Run:  py -m pip install MetaTrader5")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "mt5_data")
os.makedirs(OUTDIR, exist_ok=True)
LOGIN_FILE = os.path.join(HERE, "mt5_login.txt")


def read_login():
    """Optional key=value file: login, password, server, path. Returns kwargs for initialize()."""
    if not os.path.exists(LOGIN_FILE):
        return {}
    cfg = {}
    for line in open(LOGIN_FILE, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip().lower()] = v.strip()
    kw = {}
    if cfg.get("path"):
        kw["path"] = cfg["path"]
    if cfg.get("login"):
        kw["login"] = int(cfg["login"])
    if cfg.get("password"):
        kw["password"] = cfg["password"]
    if cfg.get("server"):
        kw["server"] = cfg["server"]
    return kw


def connect():
    kw = read_login()
    if kw:
        print(f"Using mt5_login.txt (login={kw.get('login','?')}, server={kw.get('server','?')})")
    ok = mt5.initialize(**kw)
    if not ok:
        err = mt5.last_error()
        hint = ("\n  -> Terminal not authorized. Either: (a) open MT5, File -> Login to Trade "
                "Account, log into Funding Pips until you see live prices, then re-run; or "
                "(b) fill backtest/mt5_login.txt (copy mt5_login.example.txt).")
        raise SystemExit(f"initialize() failed: {err}{hint}")

# canonical name -> candidate broker names (first that returns data wins)
TARGETS = {
    "EURUSD": ["EURUSD"],
    "USDJPY": ["USDJPY"],
    "NZDUSD": ["NZDUSD"],
    "USDCHF": ["USDCHF"],
    "XAUUSD": ["XAUUSD", "GOLD"],
    "NAS100": ["NAS100", "US100", "USTEC", "USTECH", "NDX100", "NDX", "USTech100"],
}
SCAN = ["EUR", "JPY", "NZD", "CHF", "XAU", "GOLD", "NAS", "US100", "USTEC", "NDX", "TECH"]


def main():
    connect()

    ti, ai = mt5.terminal_info(), mt5.account_info()
    print("Connected:", bool(ti and ti.connected),
          "| server:", getattr(ai, "server", "?"),
          "| company:", getattr(ai, "company", "?"))

    all_syms = [s.name for s in (mt5.symbols_get() or [])]
    upper = {n.upper(): n for n in all_syms}
    print(f"Total symbols on account: {len(all_syms)}")
    print("Matching symbols (for reference, exact names):")
    for n in all_syms:
        if any(m in n.upper() for m in SCAN):
            print("   ", n)

    def resolve(cands):
        for c in cands:
            if c.upper() in upper:
                return upper[c.upper()]
        hits = [n for n in all_syms if any(n.upper().startswith(c.upper()) for c in cands)]
        if not hits:
            hits = [n for n in all_syms if any(c.upper() in n.upper() for c in cands)]
        return min(hits, key=len) if hits else None

    start = datetime(2008, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    print("\nPulling H1 (times = BROKER SERVER clock; Claude converts to true UTC):")
    summary = []
    for canon, cands in TARGETS.items():
        name = resolve(cands)
        if not name:
            print(f"   {canon:8} -> SYMBOL NOT FOUND (check matching list above)")
            summary.append((canon, "NOT FOUND", 0, "", ""))
            continue
        mt5.symbol_select(name, True)
        rates = mt5.copy_rates_range(name, mt5.TIMEFRAME_H1, start, end)
        if rates is None or len(rates) == 0:
            rates = mt5.copy_rates_from_pos(name, mt5.TIMEFRAME_H1, 0, 200000)
        if rates is None or len(rates) == 0:
            print(f"   {canon:8} ({name}) -> NO DATA ({mt5.last_error()}); "
                  f"open its H1 chart, press Home, scroll left, re-run")
            summary.append((canon, name, 0, "", ""))
            continue
        path = os.path.join(OUTDIR, f"{canon}_H1.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time_server", "open", "high", "low", "close", "tick_volume"])
            for r in rates:
                ts = datetime.fromtimestamp(int(r["time"]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                w.writerow([ts, r["open"], r["high"], r["low"], r["close"], int(r["tick_volume"])])
        first = datetime.fromtimestamp(int(rates[0]["time"]), tz=timezone.utc)
        last = datetime.fromtimestamp(int(rates[-1]["time"]), tz=timezone.utc)
        print(f"   {canon:8} ({name:12}) -> {len(rates):>7} bars | "
              f"{first:%Y-%m-%d} .. {last:%Y-%m-%d} | saved {os.path.basename(path)}")
        summary.append((canon, name, len(rates), f"{first:%Y-%m-%d}", f"{last:%Y-%m-%d}"))

    mt5.shutdown()
    print("\n===== SUMMARY (paste this to Claude) =====")
    for canon, name, n, fr, la in summary:
        print(f"  {canon:8} {name:12} bars={n:>7}  {fr} .. {la}")
    print(f"\nFiles saved in: {OUTDIR}")


if __name__ == "__main__":
    main()
