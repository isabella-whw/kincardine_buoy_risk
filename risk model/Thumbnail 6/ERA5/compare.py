import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))
from Cstats import circ_diff_deg
from Thumbnail6 import pred_haz

NOAA_CSV = "../NOAA_2019to2024_predictions_12h.csv"
ERA5_CSV = "era5_noaa_format_prediction6.csv"
OUTPUT_CSV = "era5_vs_noaa_hazard_comparison.csv"
ONSHORE_DEG = 315.0

def prepare_hazard_input(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["wave_height_m"] = df[f"wave_height_m{suffix}"].astype(float)
    out["wave_dir_deg"] = np.abs(circ_diff_deg(ONSHORE_DEG, df[f"wave_dir_deg{suffix}"].astype(float)))
    out["wave_period_s"] = df[f"wave_period_s{suffix}"].astype(float)
    out["wind_speed_ms"] = df[f"wind_speed_ms{suffix}"].astype(float)
    out["wind_dir_deg"] = np.abs(circ_diff_deg(ONSHORE_DEG, df[f"wind_dir_deg{suffix}"].astype(float)))
    if f"max_wave_height_12h_m{suffix}" in df.columns:
        out["max_wave_height_12h_m"] = df[f"max_wave_height_12h_m{suffix}"].astype(float)
    return out

def main():
    noaa = pd.read_csv(NOAA_CSV)
    era5 = pd.read_csv(ERA5_CSV)

    noaa["datetime"] = pd.to_datetime(noaa["datetime"])
    era5["datetime"] = pd.to_datetime(era5["datetime"])

    noaa = noaa[noaa["datetime"].dt.year.isin([2022, 2023])].copy()
    era5 = era5[era5["datetime"].dt.year.isin([2022, 2023])].copy()

    noaa = noaa.sort_values("datetime").drop_duplicates(subset=["datetime"])
    era5 = era5.sort_values("datetime").drop_duplicates(subset=["datetime"])

    noaa["datetime_hour"] = noaa["datetime"].dt.floor("h")
    era5["datetime_hour"] = era5["datetime"].dt.floor("h")

    noaa = noaa.sort_values("datetime").drop_duplicates(subset=["datetime_hour"], keep="last")
    era5 = era5.sort_values("datetime").drop_duplicates(subset=["datetime_hour"], keep="last")

    merged = pd.merge(
        noaa,
        era5,
        on="datetime_hour",
        how="inner",
        suffixes=("_noaa", "_era5")
    )

    haz_in_noaa = prepare_hazard_input(merged, "_noaa")
    haz_in_era5 = prepare_hazard_input(merged, "_era5")

    haz_noaa = pred_haz(haz_in_noaa).add_suffix("_noaa")
    haz_era5 = pred_haz(haz_in_era5).add_suffix("_era5")

    out = pd.concat([merged, haz_noaa, haz_era5], axis=1)

    out["same_risk_level"] = out["risk_level_noaa"] == out["risk_level_era5"]
    out["score_diff_era5_minus_noaa"] = out["total_score_era5"] - out["total_score_noaa"]

    risk_order = ["Low", "Moderate", "High", "Extreme"]
    out["risk_level_noaa"] = pd.Categorical(out["risk_level_noaa"], categories=risk_order, ordered=True)
    out["risk_level_era5"] = pd.Categorical(out["risk_level_era5"], categories=risk_order, ordered=True)

    out.to_csv(OUTPUT_CSV, index=False)

    agreement = out["same_risk_level"].mean()
    confusion = pd.crosstab(
        out["risk_level_noaa"],
        out["risk_level_era5"],
        rownames=["NOAA"],
        colnames=["ERA5"],
        dropna=False
    )

    print(f"Risk level agreement: {agreement:.4%}")
    print("Confusion matrix:")
    print(confusion)

    mae_score = (out["score_diff_era5_minus_noaa"].abs()).mean()
    corr_score = out[["total_score_noaa", "total_score_era5"]].corr().iloc[0, 1]

    print(f"Mean score difference: {mae_score:.4f}")
    print(f"Score correlation: {corr_score:.4f}")

    mismatches = out.loc[~out["same_risk_level"], [
        "datetime_noaa",
        "datetime_era5",
        "datetime_hour",
        "wave_height_m_noaa", "wave_dir_deg_noaa", "wave_period_s_noaa", "wind_speed_ms_noaa", "wind_dir_deg_noaa",
        "wave_height_m_era5", "wave_dir_deg_era5", "wave_period_s_era5", "wind_speed_ms_era5", "wind_dir_deg_era5",
        "total_score_noaa", "risk_level_noaa",
        "total_score_era5", "risk_level_era5",
        "score_diff_era5_minus_noaa"
    ]]


if __name__ == "__main__":
    main()