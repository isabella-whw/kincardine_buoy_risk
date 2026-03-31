
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
    wave_dir_0_30 = df["wave_dir_deg"].between(0, 30)
    wave_dir_30_70 = df["wave_dir_deg"].between(31, 70)

    #Wave height bins, depending on direction
    h = df["wave_height_m"]
    wave_factor = pd.Series(0.0, index=df.index)
    wave_height_bins = [
        (h.between(0.0, 0.3048), 0.0, 0.0),
        (h.between(0.3048, 0.4572), 0.5, 0.0),
        (h.between(0.4572, 0.762), 1.0, 0.5),
        (h.between(0.762, 1.3716), 2.0, 1.0),
        (h.between(1.3716, 2.286), 3.0, 1.5),
        (h.between(2.286, 3.2004), 4.0, 2.0),
        (h > 3.2004, 5.0, 3)
    ]

    #Computes the wave height factor for a given direction
    for mask, val_0_30, val_30_70 in wave_height_bins:
        wave_factor = wave_factor.where(
            ~mask,
            np.where(wave_dir_0_30, val_0_30,
            np.where(wave_dir_30_70, val_30_70,
            0.0))
        )
    out["wave_factor"] = wave_factor

    #Defines wave period bins, depending on height
    wp = df["wave_period_s"]
    period_factor = pd.Series(0.0, index=df.index)
    wave_period_bins = [
        (wp.between(0.0, 4.5), 0.0),
        (wp.between(4.5, 6.5), 0.5),
        (wp.between(6.5, 8.5), 1.0),
        (wp.between(8.5, 10.5), 2.0),
        (wp > 10.5, 3)
    ]

    #Computes the wave period factor for a given height
    for mask, value in wave_period_bins:
        period_factor.loc[mask] = value
    
    out["period_factor"] = period_factor
    

    #Wind direction classes for on-shore and offshore
    wind_dir_0_90 = df["wind_dir_deg"].between(0, 90)
    wind_dir_90_180 = df["wind_dir_deg"].between(91, 180)

    #Wind speed bins, depending on direction
    ws = df["wind_speed_ms"]
    wind_factor = pd.Series(0.0, index=df.index)
    wind_speed_bins = [
        (ws.between(0, 2.829442), 0.5, 0.5),
        (ws.between(2.829442, 4.372774), 1.0, 1.0),
        (ws.between(4.372774, 5.916106), 1.5, 1.5),
        (ws.between(5.916106, 7.459438), 2.0, 2.0),
        (ws.between(7.459438, 9.00277), 3.0, 3.0),
        (ws.between(9.00277, 10.546102), 4.0, 4.0),
        (ws > 10.546102, 3.5, 4.0)
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



