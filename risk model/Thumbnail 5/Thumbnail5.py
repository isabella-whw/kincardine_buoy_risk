
# --------------------------------------------------------------------------------------
#pred_haz_level is a tool to predict the potential =risks for swimmers at Kincardine, 
#Ontario, including high wave conditions and possible rip current development. Inputs
#include wave height, wave direction, wave period, wind speed, and wind direction. Each
#input is assigned a numerical factor, and the recommendation for low (<4), modereate
#(4 - 7), or high (>7) risks levels are determined by the summation of all factors. This
#tool uses a modified version of the Meadows (2011) Great Lakes Rip Current Checklist
#(GLRCC). Future work could use alternative methods to identify rip currents and other
#hazards on-site to further modify these values.

# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
#Import model requirements

import pandas as pd
import numpy as np

# --------------------------------------------------------------------------------------
#Vectorized hazard level predictions. (Note: Input is a pandas data frame, and includes
#wave height (m), wave direction (o), wave period (s), wind speed (m/s), and wind
#direction (o)

def pred_haz(df):

    out = pd.DataFrame(index=df.index)

    #Wave direction classes for on-shore and oblique
    wave_dir_0_30 = df["wave_dir_deg"].between(0, 30, inclusive="both")
    wave_dir_30_60 = (df["wave_dir_deg"] > 30) & (df["wave_dir_deg"] <= 60)
    wave_dir_60_90 = (df["wave_dir_deg"] > 60) & (df["wave_dir_deg"] <= 90)

    #Wave height bins, depending on direction
    h = df["wave_height_m"].astype(float)
    wave_factor = pd.Series(0.0, index=df.index)
    wave_height_bins = [
        ((h >= 0.0)  & (h < 0.25), 0.0, 0.0, 0.0),
        ((h >= 0.25) & (h < 0.5),  1.0, 0.5, 0.0),
        ((h >= 0.5)  & (h < 0.75), 2.0, 1.5, 1.0),
        ((h >= 0.75) & (h < 1.0),  3.0, 2.5, 2.0),
        ((h >= 1.0)  & (h < 1.25), 4.0, 3.5, 3.0),
        ((h >= 1.25) & (h < 1.5),  5.0, 4.5, 4.0),
        ((h >= 1.5)  & (h < 2.0),  6.0, 5.5, 5.0),
        (h >= 2.0,                 7.0, 7.5, 6.0),
    ]
    
    #Computes the wave height factor for a given direction
    for mask, val_0_30, val_30_60, val_60_90 in wave_height_bins:
        wave_factor = wave_factor.where(
            ~mask,
            np.where(
                wave_dir_0_30, val_0_30,
                np.where(
                    wave_dir_30_60, val_30_60,
                    np.where(wave_dir_60_90, val_60_90, 0.0)
                )
            )
        )
    out["wave_factor"] = wave_factor

    # Max wave height over past 12 hours factor
    mh = df["max_wave_height_12h_m"].astype(float)
    max_wave_dir_0_30 = df["wave_dir_deg"].between(0, 30, inclusive="both")
    max_wave_dir_over_30 = df["wave_dir_deg"] > 30
    out["max_wave_factor"] = np.select(
        [
            (mh >= 0.0) & (mh < 1.25) & max_wave_dir_0_30,
            (mh >= 0.0) & (mh < 1.25) & max_wave_dir_over_30,
            (mh >= 1.25) & (mh < 1.5) & max_wave_dir_0_30,
            (mh >= 1.25) & (mh < 1.5) & max_wave_dir_over_30,
            (mh >= 1.5) & max_wave_dir_0_30,
            (mh >= 1.5) & max_wave_dir_over_30,
        ],
        [0.0, 0.0, 1.0, 0.5, 2.0, 1.5],
        default=0.0,
    ).astype(float)

    #Defines wave period bins, depending on height
    wp = df["wave_period_s"].astype(float)
    small_wave = h <= 1.25

    #Computes the wave period factor for a given height
    out["period_factor"] = np.select(
        [
            (wp >= 4.5) & (wp < 5.5) & small_wave,
            (wp >= 4.5) & (wp < 5.5) & ~small_wave,
            (wp >= 5.5) & (wp <= 6.5) & small_wave,
            (wp >= 5.5) & (wp <= 6.5) & ~small_wave,
            (wp > 6.5) & small_wave,
            (wp > 6.5) & ~small_wave,
        ],
        [0.5, 1.0, 1.5, 2.0, 2.0, 3.0],
        default=0.0,
    ).astype(float)

    #Wind speed factor
    ws = df["wind_speed_ms"].astype(float)
    out["wind_factor"] = np.select(
        [
            (ws >= 0.0) & (ws < 3.0),
            (ws >= 3.0) & (ws < 6.0),
            (ws >= 6.0) & (ws < 9.0),
            (ws >= 9.0) & (ws < 12.0),
            (ws >= 12.0),
        ],
        [0.0, 1.0, 2.0, 3.0, 4.0],
        default=0.0,
    ).astype(float)
    
    #Make a prediction based on the factor summation
    out["total_score"] = (
        out["wave_factor"]
        + out["max_wave_factor"]
        + out["period_factor"]
        + out["wind_factor"]
    )

    out["risk_level"] = np.select(
        [
            out["total_score"] < 3,
            (out["total_score"] >= 3) & (out["total_score"] < 7),
            (out["total_score"] >= 7) & (out["total_score"] <= 11),
            out["total_score"] > 11,
        ],
        ["Low", "Moderate", "High", "Extreme"],
        default="Low",
    )

    return out



