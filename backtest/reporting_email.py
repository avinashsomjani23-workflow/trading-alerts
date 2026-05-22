"""Backtest-only email sender. Own SMTP, no live-system reuse.

Sends report.html + Excel attachments. Skips silently if env vars missing.
"""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path


def send_report(run_dir: Path, recipient: str = None, subject_suffix: str = "") -> bool:
    sender = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    to = recipient or os.environ.get("BACKTEST_EMAIL") or os.environ.get("GMAIL_ADDRESS", "avinash.somjani98@gmail.com")
    if not sender or not password:
        print("  [email skipped] GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set")
        return False

    html_path = run_dir / "report.html"
    if not html_path.exists():
        print(f"  [email skipped] {html_path} missing")
        return False

    msg = MIMEMultipart()
    msg["Subject"] = f"Backtest Report — {run_dir.name}{(' ' + subject_suffix) if subject_suffix else ''}"
    msg["From"] = sender
    msg["To"] = to
    msg.attach(MIMEText(html_path.read_text(encoding="utf-8"), "html"))

    for xl in ("forex_trades.xlsx", "nas_xau_trades.xlsx", "zone_register.xlsx",
               "trades.xlsx", "raw_alerts.jsonl",
               "summary.json", "run_log.jsonl", "console.log"):
        p = run_dir / xl
        if not p.exists():
            continue
        with open(p, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{xl}"')
        msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, password)
            s.send_message(msg)
        print(f"  [email sent] -> {to}")
        return True
    except Exception as e:
        print(f"  [email error] {e}")
        return False
