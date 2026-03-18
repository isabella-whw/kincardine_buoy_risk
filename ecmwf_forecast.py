from __future__ import annotations

import requests
import pandas as pd

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
]

MARINE_VARS = [
    "wave_height",
    "wave_direction",
    "wave_peak_period",
    "wave_period",
]

LOCAL_TZ = "America/Toronto"


def _fetch_json(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _hourly_time_index(hourly: dict) -> pd.Series:
    return pd.to_datetime(hourly["time"])


def _local_to_utc(series: pd.Series) -> pd.Series:
    return (
        series.dt.tz_localize(LOCAL_TZ)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )


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
        }
    )
    df["time_utc"] = _local_to_utc(df["time_local"])
    return df


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
    forecast_df["forecast_horizon_hours"] = (
        (forecast_df["time_utc"] - retrieved_at_utc).dt.total_seconds() / 3600.0
    )

    return forecast_df


def make_hourly_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    out["time_utc"] = out["time_utc"].dt.strftime("%Y-%m-%d %H:%M")
    out["retrieved_at_utc"] = out["retrieved_at_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_dict(orient="records")


def make_three_hourly_records(df: pd.DataFrame) -> list[dict]:
    out = df.iloc[::3].copy().reset_index(drop=True)
    out["time_utc"] = out["time_utc"].dt.strftime("%Y-%m-%d %H:%M")
    out["retrieved_at_utc"] = out["retrieved_at_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.to_dict(orient="records")


def build_forecast_snapshot(
    latitude: float = LAT,
    longitude: float = LON,
    forecast_days: int = 10,
) -> dict:
    df = fetch_ecmwf_forecast_df(latitude, longitude, forecast_days)

    if df.empty:
        raise RuntimeError("ECMWF forecast dataframe is empty after filtering")

    retrieved_at_utc = df["retrieved_at_utc"].iloc[0]

    snapshot = {
        "retrieved_at_utc": retrieved_at_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": forecast_days,
        "hourly_count": len(df),
        "three_hourly_count": len(df.iloc[::3]),
        "hourly_records": make_hourly_records(df),
        "three_hourly_records": make_three_hourly_records(df),
    }
    return snapshot


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