# email_alert.py
# Handles email notifications and alert throttling logic.
# Sends failure/stale alerts and prevents repeated email spam.

import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone

from config import (
    USE_FIRESTORE, STATION_ID_DEFAULT,
    ALERT_EMAIL_TO, ALERT_EMAIL_FROM,
    SMTP_RELAY_HOST, SMTP_RELAY_PORT, SMTP_RELAY_USERNAME, SMTP_RELAY_PASSWORD,
    ALERT_THROTTLE_MINUTES, TORONTO_TZ, logger
)
from helper import now_toronto_str
from firestore import get_db

# Send an email using configured SMTP settings.
def send_email_smtp(subject: str, body: str) -> None:
    missing = []
    if not ALERT_EMAIL_TO: missing.append("ALERT_EMAIL_TO")
    if not ALERT_EMAIL_FROM: missing.append("ALERT_EMAIL_FROM")
    if not SMTP_RELAY_USERNAME: missing.append("SMTP_RELAY_USERNAME")
    if not SMTP_RELAY_PASSWORD: missing.append("SMTP_RELAY_PASSWORD")
    if missing:
        raise RuntimeError(f"SMTP not configured, missing: {', '.join(missing)}")

    msg = EmailMessage()
    msg["To"] = ALERT_EMAIL_TO
    msg["From"] = ALERT_EMAIL_FROM
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(SMTP_RELAY_HOST, SMTP_RELAY_PORT, timeout=20) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SMTP_RELAY_USERNAME, SMTP_RELAY_PASSWORD)
        server.send_message(msg)

# Determine whether an alert should be sent (throttled).
def should_send_alert(alert_key: str, last_doc_mem: dict | None) -> tuple[bool, dict | None]:
    throttle_seconds = ALERT_THROTTLE_MINUTES * 60
    mem_key = f"_last_alert_{alert_key}"
    
    # In-memory throttling
    if not USE_FIRESTORE:
        last = (last_doc_mem or {}).get(mem_key)
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                if (datetime.now(timezone.utc) - last_dt).total_seconds() < throttle_seconds:
                    return False, last_doc_mem
            except Exception:
                pass

        if last_doc_mem is None:
            last_doc_mem = {}
        last_doc_mem[mem_key] = datetime.now(timezone.utc).isoformat()
        return True, last_doc_mem

    db = get_db()
    if db is None:
        return True, last_doc_mem

    # Firestore-based throttling
    doc_ref = db.collection("alerts").document(f"{STATION_ID_DEFAULT}_{alert_key}")
    snap = doc_ref.get()
    if snap.exists:
        d = snap.to_dict() or {}
        last_sent = d.get("last_sent_utc")
        if last_sent:
            try:
                if last_sent.endswith("Z"):
                    last_sent = last_sent.replace("Z", "+00:00")
                last_dt = datetime.fromisoformat(last_sent)
                if (datetime.now(timezone.utc) - last_dt).total_seconds() < throttle_seconds:
                    return False, last_doc_mem
            except Exception:
                pass

    # Update alert timestamp
    doc_ref.set(
        {
            "station_id": STATION_ID_DEFAULT,
            "alert_key": alert_key,
            "last_sent_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "last_sent_toronto": now_toronto_str(TORONTO_TZ),
        },
        merge=True,
    )
    return True, last_doc_mem
