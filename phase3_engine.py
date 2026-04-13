import yfinance as yf
import pandas as pd
import json, os, smtplib
from datetime import datetime
import smc_detector

with open("config.json") as f: config = json.load(f)
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "dummy@gmail.com")
GMAIL_PASS    = os.environ.get("GMAIL_APP_PASSWORD", "dummy")

def send_simple_email(subject, text_body):
    try:
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        recipients = config["account"].get("alert_emails", [])
        for recipient in recipients:
            msg = MIMEMultipart()
            msg["Subject"], msg["From"], msg["To"] = subject, GMAIL_ADDRESS, recipient
            msg.attach(MIMEText(text_body, "plain"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(GMAIL_ADDRESS, GMAIL_PASS)
                server.sendmail(GMAIL_ADDRESS, recipient, msg.as_string())
    except Exception as e:
        print(f"Email failed: {e}")

def run_phase3():
    print(f"Phase 3 (M5 Trigger) started at {datetime.utcnow().strftime('%H:%M')} UTC")
    try:
        with open("active_watch_state.json") as f: watch_state = json.load(f)
    except Exception:
        print("No active watch state found. Exiting.")
        return

    if not watch_state:
        print("Watch state empty. No pairs approaching LTF triggers.")
        return

    keys_to_delete = []

    for key, data in watch_state.items():
        pair_name, bias, ob = data["pair"], data["bias"], data.get("ob", {})
        distal, proximal = float(ob.get("distal_line", 0)), float(ob.get("proximal_line", 0))

        pair_conf = next((p for p in config["pairs"] if p["name"] == pair_name), None)
        if not pair_conf: continue

        df_m5 = yf.download(pair_conf["symbol"], period="3d", interval="5m", progress=False)
        if df_m5 is None or df_m5.empty: continue
        if isinstance(df_m5.columns, pd.MultiIndex): df_m5.columns = [col[0] for col in df_m5.columns]
        
        current_close = float(df_m5['Close'].iloc[-1])
        invalidated = (bias == "LONG" and current_close < distal) or (bias == "SHORT" and current_close > distal)

        if invalidated:
            print(f"  [X] {pair_name} INVALIDATED: M5 closed beyond distal line ({distal}).")
            send_simple_email(f"INVALIDATED | {pair_name}", f"Price breached distal line at {distal}. Setup canceled.")
            keys_to_delete.append(key)
            continue

        tapped = (bias == "LONG" and df_m5['Low'].min() <= proximal) or (bias == "SHORT" and df_m5['High'].max() >= proximal)
        if not tapped:
            print(f"  [-] {pair_name}: Waiting for price to tap proximal line ({proximal}).")
            continue

        m5_struct = smc_detector.detect_bos_choch(df_m5, pair_conf, atr_value=0, lookback=4, scan_window=15)
        
        if m5_struct["confirmed"] and m5_struct["direction"] == ("bullish" if bias == "LONG" else "bearish"):
            print(f"  [✓] LTF TRIGGER FIRED: {pair_name} M5 CHoCH detected!")
            levels = data.get("levels", {})
            body = (f"LTF CHoCH Trigger Confirmed on M5.\n\nPair: {pair_name}\n"
                    f"Action: {'BUY' if bias == 'LONG' else 'SELL'} MARKET / LIMIT\n"
                    f"Entry Zone: {levels.get('entry', proximal)}\n"
                    f"Stop Loss: {levels.get('sl', distal)}\n"
                    f"Take Profit 1: {levels.get('tp1', 'TBD')}\n\nPhase 3 Engine.")
            send_simple_email(f"TRADE READY (LTF) | {pair_name} | {bias}", body)
            keys_to_delete.append(key)

    for k in keys_to_delete: del watch_state[k]
    with open("active_watch_state.json", "w") as f: json.dump(watch_state, f, indent=2)

if __name__ == "__main__": run_phase3()
