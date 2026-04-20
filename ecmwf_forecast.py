# ecmwf_forecast.py
# Fetches ECMWF weather and marine forecasts using Open-Meteo APIs.
# Combines datasets, formats timestamps, and prepares hourly/3-hourly outputs.

from __future__ import annotations

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from config import ONSHORE_DEG, TORONTO_TZ
from helper import circ_diff_deg, fmt
from risk_predict import pred_haz

# Config
LAT = 44.1834
LON = -81.6331

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
    "sunshine_duration",
    "wind_gusts_10m",
    "wind_speed_10m",
    "wind_direction_10m",
]

MARINE_VARS = [
    "wave_height",
    "wave_direction",
    "wave_peak_period",
    "wave_period",
]

LOCAL_TZ = "America/Toronto"

# Fetch JSON response from API.
def _fetch_json(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# Convert API time array to pandas datetime.
def _hourly_time_index(hourly: dict) -> pd.Series:
    return pd.to_datetime(hourly["time"])

# Convert local timestamps to UTC.
def _local_to_utc(series: pd.Series) -> pd.Series:
    return (
        series.dt.tz_localize(LOCAL_TZ)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

# Fetch ECMWF weather forecast data.
def fetch_weather_df(
    latitude: float = LAT,
    longitude: float = LON,
    forecast_days: int = 10,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(WEATHER_VARS),
        "models": "ecmwf_ifs",
        "timezone": "auto",
        "forecast_days": forecast_days,
        "wind_speed_unit": "ms",
    }
    data = _fetch_json(WEATHER_URL, params)
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise RuntimeError("Weather response missing hourly data")
    df = pd.DataFrame(
        {
            "time_local": _hourly_time_index(hourly),
            "temperature_2m_C": hourly["temperature_2m"],
            "relative_humidity_2m_pct": hourly["relative_humidity_2m"],
            "dew_point_2m_C": hourly["dew_point_2m"],
            "precipitation_mm": hourly["precipitation"],
            "pressure_msl_hPa": hourly["pressure_msl"],
            "cloud_cover_pct": hourly["cloud_cover"],
            "sunshine_duration_s": hourly["sunshine_duration"],
            "wind_gusts_10m_ms": hourly["wind_gusts_10m"],
            "wind_speed_10m_ms": hourly["wind_speed_10m"],
            "wind_direction_10m_deg": hourly["wind_direction_10m"],
        }
    )
    df["time_utc"] = _local_to_utc(df["time_local"])
    return df

# Fetch ECMWF marine forecast data.
def fetch_marine_df(
    latitude: float = LAT,
    longitude: float = LON,
    forecast_days: int = 10,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(MARINE_VARS),
        "models": "ecmwf_wam",
        "timezone": "auto",
        "forecast_days": forecast_days,
    }
    data = _fetch_json(MARINE_URL, params)
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise RuntimeError("Marine response missing hourly data")
    df = pd.DataFrame(
        {
            "time_local": _hourly_time_index(hourly),
            "wave_height_m": hourly["wave_height"],
            "wave_direction_deg": hourly["wave_direction"],
            "wave_peak_period_s": hourly["wave_peak_period"],
            "wave_period_s": hourly["wave_period"],
        }
    )
    df["time_utc"] = _local_to_utc(df["time_local"])
    return df

# Merge weather and marine forecasts into a single dataframe.
def fetch_ecmwf_forecast_df(
    latitude: float = LAT,
    longitude: float = LON,
    forecast_days: int = 10,
) -> pd.DataFrame:
    weather_df = fetch_weather_df(latitude, longitude, forecast_days)
    marine_df = fetch_marine_df(latitude, longitude, forecast_days)
    forecast_df = (
        pd.merge(
            weather_df.drop(columns=["time_local"]),
            marine_df.drop(columns=["time_local"]),
            on="time_utc",
            how="outer",
        )
        .sort_values("time_utc")
        .reset_index(drop=True)
    )
    retrieved_at_utc = pd.Timestamp.now(tz="UTC").floor("s").tz_localize(None)
    forecast_df = (
        forecast_df[forecast_df["time_utc"] >= retrieved_at_utc.floor("h")]
        .sort_values("time_utc")
        .head(240)
        .reset_index(drop=True)
    )
    forecast_df["retrieved_at_utc"] = retrieved_at_utc
    forecast_df = add_ecmwf_hazard(forecast_df)
    return forecast_df

def add_ecmwf_hazard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    haz_in = pd.DataFrame(
        {
            "wave_height_m": out["wave_height_m"].astype(float),
            "wave_dir_deg": np.abs(circ_diff_deg(ONSHORE_DEG, out["wave_direction_deg"].astype(float))),
            "wave_period_s": out["wave_period_s"].astype(float),
            "wind_speed_ms": out["wind_speed_10m_ms"].astype(float),
            "wind_dir_deg": np.abs(circ_diff_deg(ONSHORE_DEG, out["wind_direction_10m_deg"].astype(float))),
        }
    )
    haz_in["max_wave_height_12h_m"] = (
        haz_in["wave_height_m"]
        .rolling(12, min_periods=1)
        .max()
    )
    haz_out = pred_haz(haz_in)
    out["wave_dir_deg"] = haz_in["wave_dir_deg"]
    out["wind_speed_ms"] = haz_in["wind_speed_ms"]
    out["wind_dir_deg"] = haz_in["wind_dir_deg"]
    out["max_wave_height_12h_m"] = haz_in["max_wave_height_12h_m"]
    out["wave_factor"] = haz_out["wave_factor"]
    out["max_wave_factor"] = haz_out["max_wave_factor"]
    out["period_factor"] = haz_out["period_factor"]
    out["wind_factor"] = haz_out["wind_factor"]
    out["total_score"] = haz_out["total_score"]
    out["risk_level"] = haz_out["risk_level"]

    return out

# Convert dataframe to hourly record format.
def make_hourly_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    out["time_utc"] = out["time_utc"].dt.strftime("%Y-%m-%d %H:%M")
    out["retrieved_at_utc"] = out["retrieved_at_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_dict(orient="records")

# Convert dataframe to 3-hourly record format.
def make_three_hourly_records(df: pd.DataFrame) -> list[dict]:
    out = df.iloc[::3].copy().reset_index(drop=True)
    out["time_utc"] = out["time_utc"].dt.strftime("%Y-%m-%d %H:%M")
    out["retrieved_at_utc"] = out["retrieved_at_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_dict(orient="records")

# Build forecast snapshot for storage.
def build_forecast_snapshot(
    latitude: float = LAT,
    longitude: float = LON,
    forecast_days: int = 10,
) -> dict:
    df = fetch_ecmwf_forecast_df(latitude, longitude, forecast_days)
    if df.empty:
        raise RuntimeError("ECMWF forecast dataframe is empty after filtering")
    retrieved_at_utc = df["retrieved_at_utc"].iloc[0]
    retrieved_at_toronto = retrieved_at_utc.tz_localize("UTC").tz_convert(TORONTO_TZ)
    snapshot = {
        "retrieved_at_utc": retrieved_at_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "retrieved_at_toronto": retrieved_at_toronto.strftime("%Y-%m-%d %H:%M:%S"),
        "recorded_at_utc": fmt(datetime.now(timezone.utc)),
        "recorded_at_toronto": fmt(datetime.now(timezone.utc).astimezone(TORONTO_TZ)),
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": forecast_days,
        "hourly_count": len(df),
        "three_hourly_count": len(df.iloc[::3]),
        "hourly_records": make_hourly_records(df),
        "three_hourly_records": make_three_hourly_records(df),
    }
    return snapshot

def build_latest_ecmwf_doc_from_df(df: pd.DataFrame) -> dict:
    if df.empty:
        raise RuntimeError("ECMWF forecast dataframe is empty after filtering")

    row = df.iloc[0]
    ing_utc = datetime.now(timezone.utc)
    ing_tor = ing_utc.astimezone(TORONTO_TZ)

    return {
        "recorded_at_utc": fmt(ing_utc),
        "recorded_at_toronto": fmt(ing_tor),
        "timestamp_utc": pd.to_datetime(row["time_utc"]).strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_toronto": pd.to_datetime(row["time_utc"], utc=True).tz_convert(TORONTO_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "station_id": "ecmwf",
        "wave_height_m": float(row["wave_height_m"]),
        "wave_dir_deg": float(row["wave_dir_deg"]),
        "wave_period_s": float(row["wave_period_s"]),
        "wind_speed_ms": float(row["wind_speed_ms"]),
        "wind_dir_deg": float(row["wind_dir_deg"]),
        "max_wave_height_12h_m": float(row["max_wave_height_12h_m"]),
        "wave_factor": float(row["wave_factor"]),
        "max_wave_factor": float(row["max_wave_factor"]),
        "period_factor": float(row["period_factor"]),
        "wind_factor": float(row["wind_factor"]),
        "total_score": float(row["total_score"]),
        "risk_level": str(row["risk_level"]),
    }

def build_forecast_products(
    latitude: float = LAT,
    longitude: float = LON,
    forecast_days: int = 10,
) -> tuple[dict, dict]:
    df = fetch_ecmwf_forecast_df(latitude, longitude, forecast_days)
    if df.empty:
        raise RuntimeError("ECMWF forecast dataframe is empty after filtering")

    retrieved_at_utc = df["retrieved_at_utc"].iloc[0]
    retrieved_at_toronto = retrieved_at_utc.tz_localize("UTC").tz_convert(TORONTO_TZ)

    snapshot = {
        "retrieved_at_utc": retrieved_at_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "retrieved_at_toronto": retrieved_at_toronto.strftime("%Y-%m-%d %H:%M:%S"),
        "recorded_at_utc": fmt(datetime.now(timezone.utc)),
        "recorded_at_toronto": fmt(datetime.now(timezone.utc).astimezone(TORONTO_TZ)),
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": forecast_days,
        "hourly_count": len(df),
        "three_hourly_count": len(df.iloc[::3]),
        "hourly_records": make_hourly_records(df),
        "three_hourly_records": make_three_hourly_records(df),
    }

    latest_doc = build_latest_ecmwf_doc_from_df(df)
    return snapshot, latest_doc

# Run forecast retrieval and save outputs to CSV.
def main() -> None:
    snapshot = build_forecast_snapshot()

    hourly_df = pd.DataFrame(snapshot["hourly_records"])
    three_hourly_df = pd.DataFrame(snapshot["three_hourly_records"])

    hourly_df.to_csv("ecmwf_forecast_hourly.csv", index=False)
    three_hourly_df.to_csv("ecmwf_forecast_3hourly.csv", index=False)

    print("Saved hourly and 3-hourly ECMWF forecast snapshots")
    print(f"Retrieved at UTC: {snapshot['retrieved_at_utc']}")
    print(hourly_df.head())


if __name__ == "__main__":
    main()