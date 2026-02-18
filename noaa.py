# noaa.py
# Retrieves and preprocesses real-time NOAA NDBC buoy observations.
# Converts raw text data into structured pandas DataFrame format.

import requests
import pandas as pd
import numpy as np
from io import StringIO

# Fetch the latest observation row from NOAA NDBC station.
def fetch_ndbc_latest_df(station_id: str) -> pd.DataFrame:
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    text = r.text

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        raise RuntimeError("NDBC response too short / malformed")

    header = lines[0].lstrip("#").strip().split()
    data_lines = lines[2:]
    df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        sep=r"\s+",
        names=header,
        na_values=["MM"],
    )
    if df.empty:
        raise RuntimeError("No data rows in NDBC response")
    return df.iloc[[0]].copy()

# Construct UTC datetime column from NDBC date fields.
def build_datetime_utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mm" not in df.columns:
        df["mm"] = 0
    dt = pd.to_datetime(
        dict(
            year=df["YY"].astype(int),
            month=df["MM"].astype(int),
            day=df["DD"].astype(int),
            hour=df["hh"].astype(int),
            minute=df["mm"].astype(int),
        ),
        errors="coerce",
        utc=True,
    )
    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Failed to parse datetime from NDBC fields")
    return df

# Add month and decimal-hour features for ML predictors.
def add_month_hour_decimal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["datetime"].dt.month.astype(int)
    df["hour_decimal"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 100.0
    return df

# Convert directional degrees into sine and cosine components.
def ensure_dir_sin_cos(df: pd.DataFrame, deg_col: str, s_col: str, c_col: str) -> pd.DataFrame:
    df = df.copy()
    rad = np.deg2rad(df[deg_col].astype(float))
    df[s_col] = np.sin(rad)
    df[c_col] = np.cos(rad)
    return df
