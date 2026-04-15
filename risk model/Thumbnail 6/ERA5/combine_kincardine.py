import pandas as pd

era5 = pd.read_csv("era5_noaa_format_prediction6.csv")
kincardine = pd.read_csv("Kincardine_clean_prediction6.csv")

era5["datetime"] = pd.to_datetime(era5["datetime"])
kincardine["datetime"] = pd.to_datetime(kincardine["datetime"])

era5["match_time"] = era5["datetime"]
kincardine["match_time"] = kincardine["datetime"].dt.floor("h")

merged = pd.merge(
    era5,
    kincardine,
    on="match_time",
    how="inner",
    suffixes=("_era5", "_kincardine")
)

merged = merged.sort_values("match_time").reset_index(drop=True)
merged.to_csv("era5_kincardine_matched_predictions6.csv", index=False)
