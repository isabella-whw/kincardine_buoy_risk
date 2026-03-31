# survey_analysis.py
# Processes Qualtrics survey responses and matched photo metadata.
# Computes descriptive statistics, forecast summaries, and photo-based class summaries.

from pathlib import Path
import numpy as np
import pandas as pd

# Config
INPUT_CSV = "Great Lakes Surf Zone Hazards_March 30, 2026_12.50.csv"
MATCHED_PHOTO_FILE = "kincardine_timestamps.csv"
OUTPUT_DIR = Path("survey_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RISK_MAP = {"Low": 1, "Moderate": 2, "High": 3}
YN_MAP = {"Yes": 1, "No": 0}

DATE_COL = "RecordedDate"
CUTOFF_DATE = pd.Timestamp("2026-03-17")

RANK_COLS = [f"Q11_{i}" for i in range(1, 6)]
WAVE_HEIGHT_COLS = [f"Q10_{i}" for i in range(1, 8)]
WAVE_PERIOD_COLS = [f"Q13_{i}" for i in range(1, 6)]
WIND_SPEED_COLS = [f"Q14_{i}" for i in range(1, 7)]
PHOTO_COLS = ["Q16", "Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23", "Q26", "Q27", "Q28", "Q29", "Q30"]
YES_NO_COLS = ["Q39"]

FORECAST_VALUE_MAPS = {
    "wave_height_m": {
        "Q10_1": 0.15,
        "Q10_2": 0.45,
        "Q10_3": 0.75,
        "Q10_4": 1.05,
        "Q10_5": 1.35,
        "Q10_6": 1.65,
        "Q10_7": 1.95,
    },
    "wave_period_s": {
        "Q13_1": 1.0,
        "Q13_2": 3.0,
        "Q13_3": 5.0,
        "Q13_4": 7.0,
        "Q13_5": 9.0,
    },
    "wind_speed_m_s_1": {
        "Q14_1": 5.0 / 3.6,
        "Q14_2": 15.0 / 3.6,
        "Q14_3": 25.0 / 3.6,
        "Q14_4": 35.0 / 3.6,
        "Q14_5": 45.0 / 3.6,
        "Q14_6": 55.0 / 3.6,
    },
}

# Assign Low / Moderate / High class based on mean score.
def classify_mean(x):
    if pd.isna(x):
        return np.nan
    if x < 1.5:
        return "Low"
    if x < 2.5:
        return "Moderate"
    return "High"

# Load Qualtrics CSV and extract question labels.
def load_qualtrics_csv(path):
    raw = pd.read_csv(path, dtype=str)
    question_text = raw.iloc[0].to_dict()
    df = raw.iloc[2:].copy().reset_index(drop=True)
    labels = {}
    for col in df.columns:
        label = question_text.get(col, col)
        labels[col] = str(label).strip() if not pd.isna(label) else col

    return df, labels

# Convert selected columns to numeric values using an optional mapping.
def map_vals(d, cols, mapping=None):
    out = d[[c for c in cols if c in d.columns]].copy()
    for c in out.columns:
        if mapping is not None:
            out[c] = out[c].replace(mapping)
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# Compute descriptive statistics for a group of survey columns.
def descriptive_stats(df, labels, group_name):
    rows = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "group": group_name,
            "column_id": col,
            "question": labels.get(col, col),
            "n": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "class (mean)": classify_mean(s.mean()),
        })
    return pd.DataFrame(rows)

# Compute circular mean for directional variables in degrees.
def circular_mean_deg(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    rad = np.deg2rad(s)
    sin_mean = np.mean(np.sin(rad))
    cos_mean = np.mean(np.cos(rad))
    return (np.degrees(np.arctan2(sin_mean, cos_mean)) + 360) % 360

# Compute standard descriptive statistics for a numeric series.
def compute_stats(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return {
        "n": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "min": s.min(),
        "max": s.max(),
    }

# Compute descriptive statistics for directional variables.
def compute_circular_stats(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return {
        "n": len(s),
        "mean": circular_mean_deg(s),
        "median": np.nan,
        "std": np.nan,
        "min": s.min(),
        "max": s.max(),
    }

# Reshape forecast-response questions into long format.
def build_forecast_long(df, labels):
    frames = []
    for variable, value_map in FORECAST_VALUE_MAPS.items():
        cols_present = [c for c in value_map if c in df.columns]
        if not cols_present:
            continue
        tmp = df[cols_present].copy()
        tmp["response_id"] = df.index.astype(str)
        tmp = tmp.melt(
            id_vars="response_id",
            value_vars=cols_present,
            var_name="forecast_col",
            value_name="forecast_risk"
        )
        tmp["forecast_question"] = tmp["forecast_col"].map(lambda c: labels.get(c, c))
        tmp["forecast_risk_num"] = tmp["forecast_risk"].replace(RISK_MAP)
        tmp["forecast_risk_num"] = pd.to_numeric(tmp["forecast_risk_num"], errors="coerce")
        tmp["variable"] = variable
        tmp["variable_value"] = tmp["forecast_col"].map(value_map)
        tmp = tmp.dropna(subset=["forecast_risk_num", "variable_value"]).reset_index(drop=True)
        tmp["class"] = tmp["forecast_risk"]
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=[
            "response_id",
            "forecast_col",
            "forecast_risk",
            "forecast_question",
            "forecast_risk_num",
            "variable",
            "variable_value",
            "class",
        ])
    return pd.concat(frames, ignore_index=True)

# Compute class-based summary statistics for forecast responses.
def forecast_class_stats(forecast_long):
    rows = []
    for (cls, var), g in forecast_long.groupby(["class", "variable"]):
        vals = pd.to_numeric(g["variable_value"], errors="coerce").dropna()
        if vals.empty:
            continue
        if "direction" in var:
            stats = compute_circular_stats(vals)
        else:
            stats = compute_stats(vals)
        if stats is None:
            continue
        rows.append({
            "dataset": "forecast_responses",
            "class": cls,
            "variable": var,
            **stats,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        class_order = {"Low": 1, "Moderate": 2, "High": 3}
        out["class_order"] = out["class"].map(class_order)
        out = (
            out.sort_values(["class_order", "variable"])
            .drop(columns="class_order")
            .reset_index(drop=True)
        )
    return out


# Reshape photo-response questions into long format.
def build_photo_long(df, labels):
    photo_cols_present = [c for c in PHOTO_COLS if c in df.columns]
    photo_position_map = {col: i + 1 for i, col in enumerate(photo_cols_present)}
    long_df = df[photo_cols_present].copy()
    long_df["response_id"] = df.index.astype(str)
    long_df = long_df.melt(
        id_vars="response_id",
        value_vars=photo_cols_present,
        var_name="photo_col",
        value_name="photo_risk"
    )
    long_df["photo_position"] = long_df["photo_col"].map(photo_position_map)
    long_df["photo_question"] = long_df["photo_col"].map(lambda c: labels.get(c, c))
    long_df["photo_risk_num"] = long_df["photo_risk"].replace(RISK_MAP)
    long_df["photo_risk_num"] = pd.to_numeric(long_df["photo_risk_num"], errors="coerce")
    long_df = long_df.dropna(subset=["photo_risk_num"]).reset_index(drop=True)
    long_df["class"] = long_df["photo_risk"]
    return long_df

# Load matched photo metadata and standardize required columns.
def load_photo_matches(path):
    meta = pd.read_csv(path).copy()
    required = {
        "target_time",
        "Time",
        "sea_surface_wave_mean_period_s",
        "sea_surface_wave_significant_height_m",
        "sea_surface_wave_from_direction_degree",
        "wind_from_direction_degree",
        "wind_speed_m_s_1",
    }
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Matched photo file is missing required columns: {sorted(missing)}")
    meta["target_time"] = pd.to_datetime(meta["target_time"], errors="coerce")
    meta["Time"] = pd.to_datetime(meta["Time"], errors="coerce")
    numeric_cols = [
        "sea_surface_wave_mean_period_s",
        "sea_surface_wave_significant_height_m",
        "sea_surface_wave_from_direction_degree",
        "wind_from_direction_degree",
        "wind_speed_m_s_1",
    ]
    for c in numeric_cols:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    meta = meta.dropna(subset=[
        "Time",
        "sea_surface_wave_mean_period_s",
        "sea_surface_wave_significant_height_m",
        "sea_surface_wave_from_direction_degree",
        "wind_from_direction_degree",
        "wind_speed_m_s_1",
    ]).reset_index(drop=True)
    meta["photo_position"] = np.arange(1, len(meta) + 1)
    meta = meta.rename(columns={
        "Time": "timestamp",
        "sea_surface_wave_significant_height_m": "wave_height_m",
        "sea_surface_wave_mean_period_s": "wave_period_s",
        "wind_from_direction_degree": "wind_direction_deg",
        "sea_surface_wave_from_direction_degree": "wave_direction_deg",
    })
    return meta[[
        "photo_position",
        "timestamp",
        "wave_height_m",
        "wave_period_s",
        "wind_speed_m_s_1",
        "wind_direction_deg",
        "wave_direction_deg",
    ]].reset_index(drop=True)

# Compute photo-response statistics for each matched variable value.
def photo_variable_stats(photo_long, photo_meta):
    merged = photo_long.merge(photo_meta, on="photo_position", how="left")
    variable_cols = [
        "wave_height_m",
        "wave_period_s",
        "wind_speed_m_s_1",
        "wind_direction_deg",
        "wave_direction_deg",
    ]
    rows = []
    for var in variable_cols:
        tmp = merged.dropna(subset=[var, "photo_risk_num"]).copy()
        if tmp.empty:
            continue
        for value, g in tmp.groupby(var, dropna=True):
            s = g["photo_risk_num"].dropna()
            if s.empty:
                continue
            rows.append({
                "variable": var,
                "variable_value": value,
                "n": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "max": s.max(),
                "class (mean)": classify_mean(s.mean()),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["variable", "variable_value"]).reset_index(drop=True)
    return out

# Compute class-based summary statistics for matched photo responses.
def photo_class_stats(photo_long, photo_meta):
    merged = photo_long.merge(photo_meta, on="photo_position", how="left")
    variable_cols = [
        "wave_height_m",
        "wave_period_s",
        "wind_speed_m_s_1",
        "wind_direction_deg",
        "wave_direction_deg",
    ]
    rows = []
    for cls, g in merged.groupby("class"):
        for var in variable_cols:
            tmp = g.dropna(subset=[var])
            if tmp.empty:
                continue
            if "direction" in var:
                stats = compute_circular_stats(tmp[var])
            else:
                stats = compute_stats(tmp[var])
            if stats is None:
                continue
            rows.append({
                "dataset": "photo_responses",
                "class": cls,
                "variable": var,
                **stats,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        class_order = {"Low": 1, "Moderate": 2, "High": 3}
        out["class_order"] = out["class"].map(class_order)
        out = (
            out.sort_values(["class_order", "variable"])
            .drop(columns="class_order")
            .reset_index(drop=True)
        )
    return out

# Run the full survey analysis pipeline and save outputs.
def main():
    df, labels = load_qualtrics_csv(INPUT_CSV)
    if "Finished" in df.columns:
        df = df[df["Finished"].str.lower() == "true"].copy()
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df[df[DATE_COL] >= CUTOFF_DATE].copy()
    df.to_csv(OUTPUT_DIR / "cleaned_responses.csv", index=False)
    all_stats = []
    all_stats.append(descriptive_stats(map_vals(df, YES_NO_COLS, YN_MAP), labels, "yes_no"))
    all_stats.append(descriptive_stats(map_vals(df, RANK_COLS), labels, "ranking"))
    all_stats.append(descriptive_stats(map_vals(df, WAVE_HEIGHT_COLS, RISK_MAP), labels, "wave_height"))
    all_stats.append(descriptive_stats(map_vals(df, WAVE_PERIOD_COLS, RISK_MAP), labels, "wave_period"))
    all_stats.append(descriptive_stats(map_vals(df, WIND_SPEED_COLS, RISK_MAP), labels, "wind_speed"))
    all_stats.append(descriptive_stats(map_vals(df, PHOTO_COLS, RISK_MAP), labels, "photo"))
    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df.to_csv(OUTPUT_DIR / "quantitative_stats.csv", index=False)
    forecast_long = build_forecast_long(df, labels)
    forecast_class = forecast_class_stats(forecast_long)
    forecast_class.to_csv(OUTPUT_DIR / "forecast_class_stats.csv", index=False)
    text_cols = ["Q12", "Q15", "Q40", "Q37", "Q35", "Q36"]
    text_cols = [c for c in text_cols if c in df.columns]
    if text_cols:
        text_df = df[text_cols].rename(columns=lambda c: labels.get(c, c))
        text_df.to_csv(OUTPUT_DIR / "text_responses.csv", index=False)
    photo_long = build_photo_long(df, labels)
    photo_meta = load_photo_matches(MATCHED_PHOTO_FILE)
    photo_stats = photo_variable_stats(photo_long, photo_meta)
    photo_stats.to_csv(OUTPUT_DIR / "photo_stats.csv", index=False)
    photo_class = photo_class_stats(photo_long, photo_meta)
    photo_class.to_csv(OUTPUT_DIR / "photo_class_stats.csv", index=False)
    print("Done. Files created:")
    print("- cleaned_responses.csv")
    print("- quantitative_stats.csv")
    print("- forecast_class_stats.csv")
    print("- text_responses.csv")
    print("- photo_stats.csv")
    print("- photo_class_stats.csv")

if __name__ == "__main__":
    main()
