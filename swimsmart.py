# swimsmart.py
# Send forecast objects (FOBJ) to SwimSmart devices through Digi Remote Manager.

import base64
import json
from datetime import datetime, timezone
import requests

from config import (
    DIGI_URL,
    DIGI_USERNAME,
    DIGI_PASSWORD,
    DIGI_ACTOR,
    DIGI_DEVICE_IDS,
    SWIMSMART_ENABLED,
    TORONTO_TZ,
    logger,
)

# Format timestamp required by the SwimSmart API (e.g., "637 am edt thu jun 6 2024")
def _fmt_update_time(dt: datetime) -> str:
    local_dt = dt.astimezone(TORONTO_TZ)
    s = local_dt.strftime("%I%M %p %Z %a %b %d %Y").lower()
    if s.startswith("0"):
        s = s[1:]
    parts = s.split()
    if len(parts) >= 6 and parts[4].startswith("0"):
        parts[4] = parts[4][1:]
        s = " ".join(parts)
    return s

# Convert model risk output to SwimSmart format: low, moderate, high
def risk_to_swimsmart(risk_level: str) -> str:
    val = (risk_level or "").strip().lower()
    if val not in {"low", "moderate", "high"}:
        raise ValueError(f"Unsupported SwimSmart risk level: {risk_level}")
    return val

# Build the forecast object payload that SwimSmart expects in the Digi API request
def build_fobj_payload(doc: dict) -> dict:
    recorded_at_utc = datetime.strptime(
        doc["recorded_at_utc"], "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=timezone.utc)
    return {
        "update time": _fmt_update_time(recorded_at_utc),
        "issuance": "today",
        "risk": risk_to_swimsmart(doc["risk_level"]),
    }

# Construct the XML body required by Digi Remote Manager to send a forecast object to a device
def build_sci_body(device_id: str, payload: dict) -> str:
    fobj_json = json.dumps(payload)
    return f"""<sci_request version="1.0">
<data_service>
  <targets>
    <device id="{device_id}" />
  </targets>
  <requests>
    <device_request target_name="server">
      fobj$ {fobj_json}
    </device_request>
  </requests>
</data_service>
</sci_request>"""

# Create HTTP Basic Auth header required by Digi API
def _auth_header(username: str, password: str) -> str:
    creds = f"{username}:{password}"
    encoded = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
    return f"Basic {encoded}"

# Send a forecast object to a single SwimSmart device and return the response information
def send_fobj_to_device(device_id: str, payload: dict) -> dict:
    if not DIGI_USERNAME or not DIGI_PASSWORD or not DIGI_ACTOR:
        raise RuntimeError(
            "Missing Digi credentials. Required: DIGI_USERNAME, DIGI_PASSWORD, DIGI_ACTOR"
        )
    body = build_sci_body(device_id, payload)
    headers = {
        "Content-Type": "application/xml",
        "actor": DIGI_ACTOR,
        "Authorization": _auth_header(DIGI_USERNAME, DIGI_PASSWORD),
    }
    resp = requests.post(
        DIGI_URL,
        data=body.encode("utf-8"),
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return {
        "device_id": device_id,
        "status_code": resp.status_code,
        "response_text": resp.text,
        "payload": payload,
    }

# Send prediction results to all configured SwimSmart devices and return the list of responses
def send_prediction_to_swimsmart(doc: dict) -> list[dict]:
    if not SWIMSMART_ENABLED:
        logger.info("SwimSmart send skipped: SWIMSMART_ENABLED is false")
        return []
    if not DIGI_DEVICE_IDS:
        raise RuntimeError("SWIMSMART_ENABLED is true but DIGI_DEVICE_IDS is empty")
    payload = build_fobj_payload(doc)
    results = []
    for device_id in DIGI_DEVICE_IDS:
        result = send_fobj_to_device(device_id, payload)
        results.append(result)
        logger.info("SwimSmart send succeeded for device %s", device_id)
    return results