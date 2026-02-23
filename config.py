# config.py
# Central configuration for the hazard prediction system.
# Defines environment variables, model file mappings, and alert settings.

import os
import logging
from zoneinfo import ZoneInfo

# Use uvicorn logger so logs appear in Cloud Run output
logger = logging.getLogger("uvicorn.error")

# Default buoy station ID
STATION_ID_DEFAULT = os.getenv("STATION_ID", "41049")

# Directory containing trained ML model pickle files
PICKLE_DIR = os.getenv("PICKLE_DIR", "./pickle")

# Toggle Firestore usage
USE_FIRESTORE = os.getenv("USE_FIRESTORE", "true").lower() == "true"

# Timezone configuration
TORONTO_TZ = ZoneInfo("America/Toronto")

# Reference shoreline orientation (degrees)
ONSHORE_DEG = float(os.getenv("ONSHORE_DEG", "315"))

# Mapping of prediction output names to corresponding model filenames
MODEL_FILES = {
    "wave_height_m": "WaveHeight.pkl",
    "wave_period_s": "WavePeriod.pkl",
    "wind_speed_ms": "WindSpeed.pkl",
    "wave_dir_deg": "WaveDirection.pkl",
    "wind_dir_deg": "WindDirection.pkl",
}

# Default feature list expected by trained ML models
DEFAULT_PREDICTORS = [
    "hour_decimal", "WDIRs", "WDIRc", "WSPD", "GST", "WVHT", "DPD",
    "APD", "MWDs", "MWDc", "PRES", "ATMP", "WTMP", "DEWP"
]


# Email recipient and sender
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")

# SMTP relay configuration for sending alert emails
SMTP_RELAY_HOST = os.getenv("SMTP_RELAY_HOST", "smtp-relay.gmail.com")
SMTP_RELAY_PORT = int(os.getenv("SMTP_RELAY_PORT", "587"))
SMTP_RELAY_USERNAME = os.getenv("SMTP_RELAY_USERNAME", "")
SMTP_RELAY_PASSWORD = os.getenv("SMTP_RELAY_PASSWORD", "")

# Time threshold (minutes) to consider buoy data stale
ALERT_STALE_MINUTES = int(os.getenv("ALERT_STALE_MINUTES", "90"))

# Minimum interval (minutes) between repeated alert emails
ALERT_THROTTLE_MINUTES = int(os.getenv("ALERT_THROTTLE_MINUTES", "60"))

# Tobermory water level config
TOBERMORY_STATION_ID = os.getenv("TOBERMORY_STATION_ID", "5cebf1e43d0f4a073c4bc39d")

# Normal lake level baseline (meters)
TOBERMORY_NORMAL_LEVEL_M = float(os.getenv("TOBERMORY_NORMAL_LEVEL_M", "0"))

# Hours of history to request (must be >= 2)
TOBERMORY_LOOKBACK_HOURS = int(os.getenv("TOBERMORY_LOOKBACK_HOURS", "2"))