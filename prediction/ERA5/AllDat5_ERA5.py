import pandas as pd
import numpy as np

INPUT_CSV = "era5_timeseries.csv"
ALLDAT_CSV = "ALLDat5.csv"
OUTPUT_CSV = "AllDat5_ERA5.csv"

df = pd.read_csv(INPUT_CSV)

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"]).copy()

out = pd.DataFrame()

out["Time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M")
out["match_time"] = df["time"].dt.floor("h")
out["month"] = df["time"].dt.month
out["hour_decimal"] = df["time"].dt.hour + df["time"].dt.minute / 100.0

wind_dir_rad = np.deg2rad(df["wind_direction"].astype(float))
out["WDIRs"] = np.sin(wind_dir_rad)
out["WDIRc"] = np.cos(wind_dir_rad)

out["WSPD"] = df["wind_speed"].astype(float)
out["GST"] = df["instantaneous_10m_wind_gust"].astype(float)
out["WVHT"] = df["significant_height_of_combined_wind_waves_and_swell"].astype(float)
out["DPD"] = df["peak_wave_period"].astype(float)
out["APD"] = df["mean_wave_period"].astype(float)

wave_dir_rad = np.deg2rad(df["mean_wave_direction"].astype(float))
out["MWDs"] = np.sin(wave_dir_rad)
out["MWDc"] = np.cos(wave_dir_rad)

out["PRES"] = df["mean_sea_level_pressure"].astype(float) / 100.0
out["ATMP"] = df["2m_temperature"].astype(float) - 273.15
out["WTMP"] = df["sea_surface_temperature"].astype(float) - 273.15
out["DEWP"] = df["2m_dewpoint_temperature"].astype(float) - 273.15

out = out.dropna().reset_index(drop=True)

alldat = pd.read_csv(ALLDAT_CSV)

alldat["Time"] = pd.to_datetime(alldat["Time"], errors="coerce")
alldat = alldat.dropna(subset=["Time"]).copy()
alldat["match_time"] = alldat["Time"].dt.floor("h")

alldat = alldat[
    [
        "Time",
        "match_time",
        "T",
        "H",
        "Ds",
        "Dc",
        "WDs",
        "WDc",
        "WS",
    ]
].copy()

final = pd.merge(
    out,
    alldat,
    on="match_time",
    how="inner"
)

final = final[
    [
        "Time_x",
        "month",
        "hour_decimal",
        "WDIRs",
        "WDIRc",
        "WSPD",
        "GST",
        "WVHT",
        "DPD",
        "APD",
        "MWDs",
        "MWDc",
        "PRES",
        "ATMP",
        "WTMP",
        "DEWP",
        "T",
        "H",
        "Ds",
        "Dc",
        "WDs",
        "WDc",
        "WS",
    ]
].copy()

final = final.rename(columns={"Time_x": "Time"})
final = final.dropna().reset_index(drop=True)

final.to_csv(OUTPUT_CSV, index=False)
