import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(".."))
from Cstats import circ_diff_deg
from Thumbnail6 import pred_haz

PREDICTIONS_CSV = "era5_noaa_format.csv"
OUTPUT_CSV = "era5_noaa_format_prediction6.csv"

df = pd.read_csv(PREDICTIONS_CSV)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df[df["datetime"].dt.year.isin([2019, 2020, 2021, 2022, 2023, 2024])].copy()

df["wave_dir_deg"] = abs(circ_diff_deg(315, df["wave_dir_deg"]))
df["wind_dir_deg"] = abs(circ_diff_deg(315, df["wind_dir_deg"]))

results = pred_haz(df[
    [
        "wave_height_m",
        "wave_dir_deg",
        "wave_period_s",
        "max_wave_height_12h_m",
        "wind_speed_ms",
        "wind_dir_deg",
    ]
])

df_full = pd.concat([df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)

class_order = ["Low", "Moderate", "High", "Extreme"]
df_full["risk_level"] = pd.Categorical(
    df_full["risk_level"],
    categories=class_order,
    ordered=True
)

df_full.to_csv(OUTPUT_CSV, index=False)
