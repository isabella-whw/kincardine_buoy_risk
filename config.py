import os
import logging
from zoneinfo import ZoneInfo

logger = logging.getLogger("uvicorn.error")

STATION_ID_DEFAULT = os.getenv("STATION_ID", "41049")
PICKLE_DIR = os.getenv("PICKLE_DIR", "./pickle")
USE_FIRESTORE = os.getenv("USE_FIRESTORE", "true").lower() == "true"

TORONTO_TZ = ZoneInfo("America/Toronto")
ONSHORE_DEG = float(os.getenv("ONSHORE_DEG", "315"))

MODEL_FILES = {
    "wave_height_m": "WaveHeight.pkl",
    "wave_period_s": "WavePeriod.pkl",
    "wind_speed_ms": "WindSpeed.pkl",
    "wave_dir_deg": "WaveDirection.pkl",
    "wind_dir_deg": "WindDirection.pkl",
}

DEFAULT_PREDICTORS = [
    "hour_decimal", "WDIRs", "WDIRc", "WSPD", "GST", "WVHT", "DPD",
    "APD", "MWDs", "MWDc", "PRES", "ATMP", "WTMP", "DEWP"
]

# Email alerts
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")

SMTP_RELAY_HOST = os.getenv("SMTP_RELAY_HOST", "smtp-relay.gmail.com")
SMTP_RELAY_PORT = int(os.getenv("SMTP_RELAY_PORT", "587"))
SMTP_RELAY_USERNAME = os.getenv("SMTP_RELAY_USERNAME", "")
SMTP_RELAY_PASSWORD = os.getenv("SMTP_RELAY_PASSWORD", "")

ALERT_STALE_MINUTES = int(os.getenv("ALERT_STALE_MINUTES", "90"))
ALERT_THROTTLE_MINUTES = int(os.getenv("ALERT_THROTTLE_MINUTES", "60"))
