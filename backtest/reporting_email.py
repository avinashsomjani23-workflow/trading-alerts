"""Backtest-only email sender. Own SMTP, no live-system reuse.

Sends two emails per backtest run:
  1. Original (FX majors + Gold)   (report_forex.html    + forex_trades.xlsx)
  2. New (new FX + BTC + NAS100)   (report_gold_nas.html + nas_xau_trades.xlsx)

Subject lines carry the date window and the auto-detected regime
(WAR vs BAU). Only the per-group Excel rides along as an attachment --
no JSON/log files in mail.

Skips silently if env vars missing.
"""

from __future__ import annotations

import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Optional, Tuple


def _read_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [email warn] could not parse summary.json: {e}")
        return {}


def _regime_phrase(regime: str, regime_label: Optional[str]) -> str:
    """Human-readable regime fragment for the subject line."""
    if regime == "war":
        if regime_label:
            return f"WAR regime ({regime_label})"
        return "WAR regime"
    if regime == "bau":
        return "BAU regime"
    return regime or "regime unspecified"


def _health_banner() -> Tuple[str, list]:
    """Returns (html_banner, subject_tags). Surfaces perf-optimisation
    failures so a broken cache doesn't go unnoticed across runs."""
    failure_tags: list = []
    body_banner = ""
    try:
        import smc_detector  # type: ignore
        atr_err = smc_detector._atr_cache_status()
        if atr_err:
            failure_tags.append("ATR-CACHE-FAIL")
            body_banner += (
                f"<div style='background:#ffdddd;border:2px solid #b00;"
                f"padding:10px;margin:10px 0;font-family:monospace;'>"
                f"<b>WARNING — ATR memoization failed during this run.</b><br>"
                f"Error: {atr_err}<br>"
                f"Results are still valid (fell back to raw compute), but the "
                f"runtime optimisation is broken."
                f"</div>"
            )
    except Exception as e:
        failure_tags.append("HEALTH-CHECK-FAIL")
        body_banner += (
            f"<div style='background:#ffdddd;padding:10px;'>"
            f"Health-check raised: {type(e).__name__}: {e}</div>"
        )
    return body_banner, failure_tags


def _send_one(
    sender: str,
    password: str,
    to: str,
    subject: str,
    html_path: Path,
    attachment_path: Optional[Path],
    banner: str,
) -> bool:
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    html_body = html_path.read_text(encoding="utf-8")
    if banner:
        html_body = banner + html_body
    msg.attach(MIMEText(html_body, "html"))

    if attachment_path and attachment_path.exists():
        with open(attachment_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{attachment_path.name}"',
        )
        msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, password)
            s.send_message(msg)
        print(f"  [email sent] -> {to}  ({subject})")
        return True
    except Exception as e:
        print(f"  [email error] {e}  ({subject})")
        return False


def send_report(run_dir: Path, recipient: str = None, subject_suffix: str = "") -> bool:
    """Send the per-group reports.

    `subject_suffix` is kept in the signature for backwards compatibility with
    run_backtest.py but is not appended to the subjects -- the new subject
    layout is the source of truth for what the user sees.
    """
    sender = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    to = (recipient
          or os.environ.get("BACKTEST_EMAIL")
          or os.environ.get("GMAIL_ADDRESS")
          or "avinash.somjani98@gmail.com")
    if not sender or not password:
        print("  [email skipped] GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set")
        return False

    summary = _read_summary(run_dir)
    meta = (summary.get("meta") or {})
    start = meta.get("start") or "?"
    end   = meta.get("end") or "?"
    regime = meta.get("regime") or "auto"
    regime_label = meta.get("regime_label")
    regime_phrase = _regime_phrase(regime, regime_label)
    date_window = f"{start} → {end}"

    banner, failure_tags = _health_banner()
    tag_str = (" [" + " ".join(failure_tags) + "]") if failure_tags else ""

    groups = [
        ("Original backtest (FX majors + Gold)", "report_forex.html",    "forex_trades.xlsx"),
        ("New backtest (new FX + BTC + NAS100)", "report_gold_nas.html", "nas_xau_trades.xlsx"),
    ]

    any_sent = False
    for label, html_name, xl_name in groups:
        html_path = run_dir / html_name
        if not html_path.exists():
            print(f"  [email skipped] {html_name} not produced "
                  f"(likely no trades for this group)")
            continue
        subject = f"{label} · {date_window} · {regime_phrase}{tag_str}"
        xl_path = run_dir / xl_name
        ok = _send_one(sender, password, to, subject, html_path,
                       xl_path if xl_path.exists() else None, banner)
        any_sent = any_sent or ok

    return any_sent
