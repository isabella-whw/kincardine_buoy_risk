
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
    wave_dir_0_29 = df["wave_dir_deg"].between(0, 29)
    wave_dir_30_59 = df["wave_dir_deg"].between(30, 59)
    wave_dir_60_90 = df["wave_dir_deg"].between(60, 90)

    #Wave height bins, depending on direction
    h = df["wave_height_m"].astype(float)
    wave_factor = pd.Series(0.0, index=df.index)
    wave_height_bins = [
        (h.between(0.0, 0.57912),    1.0, 0.5, 0.0),   # 0.0-1.9 ft
        (h.between(0.57912, 0.73152), 2.0, 1.5, 1.0),  # 2.0-2.4 ft
        (h.between(0.73152, 0.88392), 3.0, 2.5, 2.0),  # 2.5-2.9 ft
        (h.between(0.88392, 1.03632), 4.5, 4.0, 3.5),  # 3.0-3.4 ft
        (h.between(1.03632, 1.18872), 6.5, 6.0, 5.0),  # 3.5-3.9 ft
        (h.between(1.18872, 1.34112), 8.0, 7.5, 7.0),  # 4.0-4.4 ft
        (h.between(1.34112, 2.10312), 10.0, 9.0, 8.0), # 4.5-6.9 ft
        (h >= 2.13360,                13.0, 12.0, 11.0), # >= 7.0 ft
    ]

    #Computes the wave height factor for a given direction
    for mask, val_0_29, val_30_59, val_60_90 in wave_height_bins:
        wave_factor = wave_factor.where(
            ~mask,
            np.where(
                wave_dir_0_29, val_0_29,
                np.where(
                    wave_dir_30_59, val_30_59,
                    np.where(wave_dir_60_90, val_60_90, 0.0)
                )
            )
        )
    out["wave_factor"] = wave_factor

    # Max wave height over past 12 hours factor
    mh = df["max_wave_height_12h_m"].astype(float)
    out["max_wave_factor"] = np.select(
        [
            mh.between(0.9144, 1.18872),   # 3.0-3.9 ft
            mh.between(1.2192, 1.49352),   # 4.0-4.9 ft
            mh >= 1.524,                   # >= 5.0 ft
        ],
        [0.5, 1.0, 2.0],
        default=0.0,
    ).astype(float)

    #Defines wave period bins, depending on height
    wp = df["wave_period_s"].astype(float)
    small_wave = h < 0.9144

    #Computes the wave period factor for a given height
    out["period_factor"] = np.select(
        [
            wp.between(4.5, 5.9) & small_wave,
            wp.between(4.5, 5.9) & ~small_wave,
            wp.between(6.0, 7.0) & small_wave,
            wp.between(6.0, 7.0) & ~small_wave,
            (wp > 7.0) & small_wave,
            (wp > 7.0) & ~small_wave,
        ],
        [0.5, 1.0, 1.5, 2.0, 2.0, 3.0],
        default=0.0,
    ).astype(float)

    #Make a prediction based on the factor summation
    out["total_score"] = (
        out["wave_factor"]
        + out["max_wave_factor"]
        + out["period_factor"]
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



