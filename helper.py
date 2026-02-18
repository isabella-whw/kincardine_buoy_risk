import os
import gc
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from config import DEFAULT_PREDICTORS, logger


def fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def now_toronto_str(tz: ZoneInfo) -> str:
    return fmt(datetime.now(timezone.utc).astimezone(tz))

def circ_diff_deg(a: float, b: float) -> float:
    diff = (a - b + 180) % 360 - 180
    return diff

def reconstruct_angle(sin_vals, cos_vals):
    ang = np.degrees(np.arctan2(sin_vals, cos_vals))
    return (ang + 360) % 360

def predict_from_bundle(bundle: dict, df: pd.DataFrame) -> float:
    X = df[bundle["predictors"]].to_numpy(dtype=float, copy=False)
    pred_s = bundle["model_sin"].predict(X)
    pred_c = bundle["model_cos"].predict(X)
    return float(reconstruct_angle(pred_s, pred_c)[0])

def predict_scalar_model(model, df: pd.DataFrame) -> float:
    feats = list(getattr(model, "feature_names_in_", DEFAULT_PREDICTORS))
    X = df[feats].to_numpy(dtype=float, copy=False)
    return float(model.predict(X)[0])

def load_model(path: str):
    if not os.path.exists(path):
        logger.error(f"MODEL FILE NOT FOUND: {path}")
        raise RuntimeError(f"Model file missing in container: {path}")
    return joblib.load(path)

def safe_del(obj) -> None:
    try:
        del obj
    except Exception:
        pass
    gc.collect()
