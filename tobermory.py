# tobermory.py
# Real-time Tobermory water level utilities

from datetime import datetime, timedelta, timezone
import requests
import pandas as pd


def fetch_tobermory_df(station_id: str, start_utc: datetime, end_utc: datetime):
    url = f"https://api-iwls.dfo-mpo.gc.ca/api/v1/stations/{station_id}/data"

    params = {
        "time-series-code": "wlo",
        "from": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()

    df = pd.DataFrame(r.json())

    if df.empty:
        raise RuntimeError("No Tobermory water level data returned")

    df["eventDate"] = pd.to_datetime(df["eventDate"], utc=True)
    df.rename(columns={"value": "water_level_m"}, inplace=True)

    df = df.set_index("eventDate").sort_index()
    return df[["water_level_m"]]


def last_completed_hour_tobermory_min(
    station_id: str,
    lookback_hours: int = 2,
):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=lookback_hours)

    df = fetch_tobermory_df(station_id, start, now)

    hourly = df["water_level_m"].resample("1h").agg(["mean", "min"])

    if len(hourly) < 2:
        raise RuntimeError("Not enough Tobermory data for completed hour")

    last_completed = hourly.iloc[-2]

    return float(last_completed["min"])