import pandas as pd

era5 = pd.read_csv("era5_noaa_format_prediction6.csv")
noaa = pd.read_csv("../NOAA_2019to2024_predictions6.csv")

era5["datetime"] = pd.to_datetime(era5["datetime"])
noaa["datetime"] = pd.to_datetime(noaa["datetime"])

era5["match_time"] = era5["datetime"]
noaa["match_time"] = noaa["datetime"].dt.floor("h")

merged = pd.merge(
    era5,
    noaa,
    on="match_time",
    how="inner",
    suffixes=("_era5", "_noaa")
)

merged = merged.sort_values("match_time").reset_index(drop=True)
merged.to_csv("era5_noaa_matched_predictions6.csv", index=False)
