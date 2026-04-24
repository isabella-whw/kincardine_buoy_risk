import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from risk_predict import pred_haz

NOAA_CSV = "era5_timeseries.csv"
PICKLE_DIR = "pickle"
OUT_CSV = "ERA5_predictions.csv"
OUT_CSV_RISK = "ERA5_predictions6.csv"

MODEL_FILES = {
    "wave_height_m": "WaveHeight.pkl",
    "wave_period_s": "WavePeriod.pkl",
    "wind_speed_ms": "WindSpeed.pkl",
    "wave_dir_deg": "WaveDirection.pkl",
    "wind_dir_deg": "WindDirection.pkl",
}

def build_datetime_utc(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["YY", "MM", "DD", "hh", "mm"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing time columns {miss}. Found: {list(df.columns)[:30]}")
    dt = pd.to_datetime(
        dict(
            year=df["YY"].astype(int),
            month=df["MM"].astype(int),
            day=df["DD"].astype(int),
            hour=df["hh"].astype(int),
            minute=df["mm"].astype(int),
        ),
        errors="coerce",
        utc=True,
    )
    df = df.copy()
    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df

def add_month_hour_decimal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["datetime"].dt.month.astype(int)
    df["hour_decimal"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 100.0
    return df

def ensure_dir_sin_cos(df: pd.DataFrame, deg_col: str, s_col: str, c_col: str) -> pd.DataFrame:
    df = df.copy()
    if s_col in df.columns and c_col in df.columns:
        return df
    if deg_col not in df.columns:
        raise ValueError(f"Missing {deg_col} needed to create {s_col}/{c_col}")
    rad = np.deg2rad(df[deg_col].astype(float))
    df[s_col] = np.sin(rad)
    df[c_col] = np.cos(rad)
    return df

def reconstruct_angle(sin_vals, cos_vals):
    ang = np.degrees(np.arctan2(sin_vals, cos_vals))
    return (ang + 360) % 360

def load_pickle(path: str):
    return joblib.load(path)

def predict_from_bundle(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    model_sin = bundle.get("model_sin")
    model_cos = bundle.get("model_cos")
    predictors = bundle.get("predictors")
    if model_sin is None or model_cos is None or predictors is None:
        raise ValueError(f"Bundle missing keys. Found keys: {list(bundle.keys())}")
    missing = [c for c in predictors if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing predictors required by bundle: {missing}")
    X = df[predictors].to_numpy(dtype=float, copy=False)
    pred_s = model_sin.predict(X)
    pred_c = model_cos.predict(X)
    return reconstruct_angle(pred_s, pred_c)

def predict_scalar_model(model, df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "feature_names_in_"):
        feats = list(model.feature_names_in_)
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for model: {missing}")
        X = df[feats]
        return model.predict(X)
    default_predictors = [
        "hour_decimal", "WDIRs", "WDIRc", "WSPD", "GST", "WVHT", "DPD",
        "APD", "MWDs", "MWDc", "PRES", "ATMP", "DEWP"
    ]
    if all(c in df.columns for c in default_predictors) and hasattr(model, "n_features_in_"):
        if model.n_features_in_ == len(default_predictors):
            X = df[default_predictors].to_numpy(dtype=float, copy=False)
            return model.predict(X)
    raise ValueError(
        "Scalar model was fit without feature names, and I can't infer the exact predictor set safely.\n"
        "Fix: retrain saving predictors, or save sklearn models with feature names (train with DataFrame),\n"
        "or tell me the exact predictor column list used for this scalar model."
    )

def main():
    df = pd.read_csv(NOAA_CSV)
    df = build_datetime_utc(df)
    df = add_month_hour_decimal(df)
    df = ensure_dir_sin_cos(df, "WDIR", "WDIRs", "WDIRc")
    df = ensure_dir_sin_cos(df, "MWD", "MWDs", "MWDc")
    preds = {}
    for out_col, fname in MODEL_FILES.items():
        path = os.path.join(PICKLE_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing pickle: {path}")
        obj = load_pickle(path)
        if isinstance(obj, dict) and ("model_sin" in obj and "model_cos" in obj):
            preds[out_col] = predict_from_bundle(obj, df)
        else:
            preds[out_col] = predict_scalar_model(obj, df)
    df_out = pd.DataFrame({
        "datetime": df["datetime"],
        "wave_height_m": preds["wave_height_m"],
        "wave_dir_deg": preds["wave_dir_deg"],
        "wave_period_s": preds["wave_period_s"],
        "wind_speed_ms": preds["wind_speed_ms"],
        "wind_dir_deg": preds["wind_dir_deg"],
    })
    df_save = df_out.copy()
    df_save["datetime"] = df_save["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_save.to_csv(OUT_CSV, index=False)
    df_risk = df_out.sort_values("datetime").reset_index(drop=True).copy()
    df_risk = df_risk.set_index("datetime")
    df_risk["max_wave_height_12h_m"] = df_risk["wave_height_m"].rolling("12h", min_periods=1).max()
    df_risk = df_risk.reset_index()
    haz_in = df_risk[[
        "wave_height_m",
        "wave_dir_deg",
        "wave_period_s",
        "wind_speed_ms",
        "wind_dir_deg",
        "max_wave_height_12h_m",
    ]].copy()
    haz_out = pred_haz(haz_in)
    df_final = pd.concat([df_risk, haz_out], axis=1)
    df_final["datetime"] = df_final["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_final.to_csv(OUT_CSV_RISK, index=False)

if __name__ == "__main__":
    main()
