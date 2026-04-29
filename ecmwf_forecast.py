# ecmwf_forecast.py
# Fetches ECMWF weather and marine data using Open-Meteo APIs.
# Combines datasets and prepares the current-hour input for the hazard model.

from __future__ import annotations

import requests
import pandas as pd
import numpy as np
from config import LAT, LON, WEATHER_URL, MARINE_URL, LOCAL_TZ

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

# Fetch JSON response from API.
def _fetch_json(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# Convert API time array to pandas datetime.
def _hourly_time_index(hourly: dict) -> pd.Series:
    return pd.to_datetime(hourly["time"])

# Convert API local timestamps to UTC (required for alignment with NOAA format).
def _local_to_utc(series: pd.Series) -> pd.Series:
    return (
        series.dt.tz_localize(LOCAL_TZ)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

# Fetch ECMWF weather data (Open-Meteo API).
def fetch_weather_df(latitude: float = LAT, longitude: float = LON) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(WEATHER_VARS),
        "models": "ecmwf_ifs",
        "timezone": "auto",
        "forecast_days": 10,
        "wind_speed_unit": "ms",
    }
    data = _fetch_json(WEATHER_URL, params)
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise RuntimeError("Weather response missing hourly data")
    df = pd.DataFrame({
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
    })
    df["time_utc"] = _local_to_utc(df["time_local"])
    return df

# Fetch ECMWF marine wave data (Open-Meteo API).
def fetch_marine_df(latitude: float = LAT, longitude: float = LON) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(MARINE_VARS),
        "models": "ecmwf_wam",
        "timezone": "auto",
        "forecast_days": 10,
    }
    data = _fetch_json(MARINE_URL, params)
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise RuntimeError("Marine response missing hourly data")
    df = pd.DataFrame({
        "time_local": _hourly_time_index(hourly),
        "wave_height_m": hourly["wave_height"],
        "wave_direction_deg": hourly["wave_direction"],
        "wave_peak_period_s": hourly["wave_peak_period"],
        "wave_period_s": hourly["wave_period"],
    })
    df["time_utc"] = _local_to_utc(df["time_local"])
    return df

# Merge ECMWF weather and marine data into a single dataframe (future hours only).
def fetch_ecmwf_forecast_df(latitude: float = LAT, longitude: float = LON) -> pd.DataFrame:
    weather_df = fetch_weather_df(latitude, longitude)
    marine_df = fetch_marine_df(latitude, longitude)
    df = pd.merge(
        weather_df.drop(columns=["time_local"]),
        marine_df.drop(columns=["time_local"]),
        on="time_utc",
        how="outer",
    ).sort_values("time_utc").reset_index(drop=True)
    now_utc = pd.Timestamp.now(tz="UTC").floor("h").tz_localize(None)
    df = df[df["time_utc"] >= now_utc].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("ECMWF forecast dataframe is empty after filtering")
    return df

# Build a single-row ML input using the current ECMWF hour.
def build_ecmwf_ml_input_row(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if df.empty:
        raise RuntimeError("ECMWF forecast dataframe is empty")
    row = df.iloc[0].copy()
    ts = pd.to_datetime(row["time_utc"], utc=True)
    max_wave_height_12h_m = float(df["wave_height_m"].astype(float).head(12).max())
    ml = pd.DataFrame([{
        "datetime": ts,
        "month": int(ts.month),
        "hour_decimal": float(ts.hour + ts.minute / 100.0),
        "WDIR": float(row["wind_direction_10m_deg"]),
        "WSPD": float(row["wind_speed_10m_ms"]),
        "GST": float(row["wind_gusts_10m_ms"]),
        "WVHT": float(row["wave_height_m"]),
        "DPD": float(row["wave_peak_period_s"]),
        "APD": float(row["wave_period_s"]),
        "MWD": float(row["wave_direction_deg"]),
        "PRES": float(row["pressure_msl_hPa"]),
        "ATMP": float(row["temperature_2m_C"]),
        "DEWP": float(row["dew_point_2m_C"]),
    }])
    ml["WDIRs"] = np.sin(np.deg2rad(ml["WDIR"]))
    ml["WDIRc"] = np.cos(np.deg2rad(ml["WDIR"]))
    ml["MWDs"] = np.sin(np.deg2rad(ml["MWD"]))
    ml["MWDc"] = np.cos(np.deg2rad(ml["MWD"]))
    return ml, max_wave_height_12h_m

def build_ecmwf_ml_input_row_with_wtmp(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    ml, max_wave_height_12h_m = build_ecmwf_ml_input_row(df)
    ml["WTMP"] = ml["ATMP"]
    return ml, max_wave_height_12h_m
