import numpy as np
import pandas as pd

from Cstats import circ_diff_deg
from Thumbnail4 import pred_haz

PREDICTIONS_CSV = "NOAA_2019to2024_predictions_12h.csv"
OUTPUT_CSV = "NOAA_2019to2024_predictions4.csv"

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
        "wind_speed_ms",
        "wind_dir_deg",
    ]
])

df_out = df[
    [
        "datetime",
        "wave_height_m",
        "wave_dir_deg",
        "wave_period_s",
        "wind_speed_ms",
        "wind_dir_deg",
        "max_wave_height_12h_m",
    ]
].copy()

# keep only the weighted contributions
df_out["wave_height_factor"] = results["wave_height_factor"].values
df_out["wave_period_factor"] = results["wave_period_factor"].values
df_out["wave_direction_factor"] = results["wave_direction_factor"].values
df_out["wind_speed_factor"] = results["wind_speed_factor"].values
df_out["wind_direction_factor"] = results["wind_direction_factor"].values

df_out["total_score"] = results["total_score"].values
df_out["risk_level"] = results["risk_level"].values

class_order = ["Low", "Moderate", "High"]
df_out["risk_level"] = pd.Categorical(
    df_out["risk_level"],
    categories=class_order,
    ordered=True
)

df_out.to_csv(OUTPUT_CSV, index=False)
