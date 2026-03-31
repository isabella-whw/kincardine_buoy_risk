
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

    # Wave height score
    h = df["wave_height_m"].astype(float)
    out["wave_height_score"] = np.select(
        [
            (h >= 0.0) & (h < 0.2),
            (h >= 0.2) & (h < 0.9),
            (h >= 0.9) & (h < 1.4),
            (h >= 1.4),
        ],
        [0.0, 1.0, 2.0, 3.0],
        default=0.0,
    ).astype(float)

    # Wave period score
    wp = df["wave_period_s"].astype(float)
    out["wave_period_score"] = np.select(
        [
            (wp < 4.5),
            (wp >= 4.5) & (wp < 5.5),
            (wp >= 5.5) & (wp <= 6.5),
            (wp > 6.5),
        ],
        [0.0, 1.0, 2.0, 3.0],
        default=0.0,
    ).astype(float)

    # Wave direction score
    wad = df["wave_dir_deg"].astype(float)
    out["wave_direction_score"] = np.select(
        [
            (wad > 60) & (wad <= 90),
            (wad > 30) & (wad <= 60),
            (wad >= 0) & (wad <= 30),
        ],
        [1.0, 2.0, 3.0],
        default=0.0,
    ).astype(float)

    # Wind speed score
    ws = df["wind_speed_ms"].astype(float)
    out["wind_speed_score"] = np.select(
        [
            (ws >= 0.0) & (ws < 1.0),
            (ws >= 1.0) & (ws < 6.0),
            (ws >= 6.0) & (ws <= 12.0),
            (ws > 12.0),
        ],
        [0.0, 1.0, 2.0, 3.0],
        default=0.0,
    ).astype(float)

    # Wind direction score
    wid = df["wind_dir_deg"].astype(float)
    out["wind_direction_score"] = np.select(
        [
            (wid > 60) & (wid <= 150),
            (wid > 150) & (wid <= 180),
            (wid >= 0) & (wid <= 60),
        ],
        [1.0, 2.0, 3.0],
        default=0.0,
    ).astype(float)

    # Weighted factors
    out["wave_height_factor"] = out["wave_height_score"] * 0.437956
    out["wave_period_factor"] = out["wave_period_score"] * 0.145985
    out["wave_direction_factor"] = out["wave_direction_score"] * 0.218978
    out["wind_speed_factor"] = out["wind_speed_score"] * 0.109489
    out["wind_direction_factor"] = out["wind_direction_score"] * 0.087591

    # Make a prediction based on the factor summation
    out["total_score"] = (
        out["wave_height_factor"]
        + out["wave_period_factor"]
        + out["wave_direction_factor"]
        + out["wind_speed_factor"]
        + out["wind_direction_factor"]
    )

    out["risk_level"] = np.select(
        [
            out["total_score"] <= 1.5,
            (out["total_score"] > 1.5) & (out["total_score"] < 2.5),
            out["total_score"] >= 2.5,
        ],
        ["Low", "Moderate", "High"],
        default="Low",
    )

    return out



