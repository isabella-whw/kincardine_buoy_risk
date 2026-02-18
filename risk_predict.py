
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
    wave_dir_0_60 = df["wave_dir_deg"].between(0, 60)
    wave_dir_60_90 = df["wave_dir_deg"].between(61, 90)

    #Wave height bins, depending on direction
    h = df["wave_height_m"]
    wave_factor = pd.Series(0.0, index=df.index)
    wave_height_bins = [
        (h.between(0, 0.60), 1, 0.5),
        (h.between(0.60, 0.76), 2.0, 1.5),
        (h.between(0.76, 0.91), 3.0, 2.5),
        (h.between(0.91, 1.06), 4.5, 4.0),
        (h.between(1.06, 1.21), 6.5, 6),
        (h.between(1.21, 1.37), 8, 7.5),
        (h.between(1.37, 2.13), 10, 9),
        (h > 2.13, 13, 12)
    ]

    #Computes the wave height factor for a given direction
    for mask, val_0_60, val_60_90 in wave_height_bins:
        wave_factor = wave_factor.where(
            ~mask,
            np.where(wave_dir_0_60, val_0_60,
            np.where(wave_dir_60_90, val_60_90,
            0.0))
        )
    out["wave_factor"] = wave_factor

    #Defines wave period bins, depending on height
    wp = df["wave_period_s"]
    bins = [4.5, 6, 7, np.inf]
    idx = np.digitize(wp, bins) - 1
    labels_small_waves = np.array([0.5, 1.5, 2.0])
    labels_large_waves = np.array([1, 2, 3])

    #Computes the wave period factor for a given height
    out["period_factor"] = np.where(
    h < 0.91,
    labels_small_waves[idx],
    labels_large_waves[idx]
    )
        
    #Wind direction classes for on-shore and offshore
    wind_dir_0_90 = df["wind_dir_deg"].between(0, 90)
    wind_dir_90_180 = df["wind_dir_deg"].between(91, 180)

    #Wind speed bins, depending on direction
    ws = df["wind_speed_ms"]
    wind_factor = pd.Series(0.0, index=df.index)
    wind_speed_bins = [
        (ws.between(0, 3.08), 0.5, 0.0),
        (ws.between(3.08, 4.63), 0.75, 0.5),
        (ws.between(4.63, 6.17), 1.0, 0.75),
        (ws.between(6.17, 7.71), 1.25, 1),
        (ws.between(7.71, 9.25), 1.50, 1.25),
        (ws.between(9.25, 10.28), 1.75, 1.5),
        (ws > 10.28, 2.0, 1.75)
    ]

    #Computes the wind speed factor for a given direction
    for mask, val_on, val_off in wind_speed_bins:
        wind_factor = wind_factor.where(
            ~mask,
            np.where(wind_dir_0_90, val_on,
            np.where(wind_dir_90_180, val_off,
            0.0))
        )
    out["wind_factor"] = wind_factor

    #Make a prediction based on the factor summation
    out["total_score"] = (
        out["wave_factor"]
        + out["period_factor"]
        + out["wind_factor"]
    )

    out["risk_level"] = np.where(
        out["total_score"] < 4, "Low",
        np.where(out["total_score"] <= 7, "Moderate", "High")
    )

    return out

    

