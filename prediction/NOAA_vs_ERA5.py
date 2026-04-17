import pandas as pd
import sys

sys.path.append(r"..\risk model\Thumbnail 6")
from Thumbnail6 import pred_haz

def process_file(input_csv):
    df = pd.read_csv(input_csv)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").copy()
    df = df.set_index("datetime")
    df["max_wave_height_12h_m"] = (
        df["wave_height_m"]
        .rolling("12h", min_periods=1)
        .max()
    )
    df = df.reset_index()
    haz_in = df[[
        "wave_height_m",
        "wave_dir_deg",
        "wave_period_s",
        "wind_speed_ms",
        "wind_dir_deg",
        "max_wave_height_12h_m"
    ]].copy()
    haz_out = pred_haz(haz_in)
    return pd.concat([df, haz_out], axis=1)

noaa = process_file("NOAA_2019to2024_predictions.csv")
era5 = process_file(r"ERA5\ERA5_2019to2024_predictions.csv")

noaa.to_csv("NOAA_2019to2024_predictions6.csv", index=False)
era5.to_csv(r"ERA5\ERA5_2019to2024_predictions6.csv", index=False)

noaa = noaa.sort_values("datetime").copy()
era5 = era5.sort_values("datetime").copy()

merged = pd.merge_asof(
    noaa[["datetime", "total_score", "risk_level"]],
    era5[["datetime", "total_score", "risk_level"]],
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta("1h"),
    suffixes=("_noaa", "_era5")
)

merged = merged.dropna(subset=["total_score_era5", "risk_level_era5"]).copy()

order = ["Low", "Moderate", "High", "Extreme"]

merged["risk_level_noaa"] = pd.Categorical(
    merged["risk_level_noaa"], categories=order, ordered=True
)
merged["risk_level_era5"] = pd.Categorical(
    merged["risk_level_era5"], categories=order, ordered=True
)

agreement = (merged["risk_level_noaa"] == merged["risk_level_era5"]).mean() * 100
score_diff = (merged["total_score_noaa"] - merged["total_score_era5"]).abs().mean()

confusion = pd.crosstab(
    merged["risk_level_noaa"],
    merged["risk_level_era5"],
    rownames=["NOAA"],
    colnames=["ERA5"]
)

confusion = confusion.reindex(index=order, columns=order, fill_value=0)

print(f"Risk level agreement: {agreement:.4f}%")
print(f"Mean score difference: {score_diff:.4f}")
print("Confusion matrix:")
print(confusion)

merged.to_csv("NOAA_vs_ERA5_thumbnail6.csv", index=False)
