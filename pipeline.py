# pipeline.py
# Core prediction pipeline: fetch NOAA data, run ML models, compute hazard score,
# and return a Firestore-ready prediction document.

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from config import (
    STATION_ID_DEFAULT, PICKLE_DIR, MODEL_FILES,
    TORONTO_TZ, ONSHORE_DEG,
    ALERT_STALE_MINUTES, logger
)
from helper import fmt, now_toronto_str, circ_diff_deg, load_model, predict_from_bundle, predict_scalar_model, safe_del
from risk_predict import pred_haz
from noaa import fetch_ndbc_latest_df, build_datetime_utc, add_month_hour_decimal, ensure_dir_sin_cos 
from email_alert import send_email_smtp, should_send_alert

# Run one end-to-end prediction cycle and return output document.
def make_prediction(station_id: str, last_doc_mem: dict | None) -> tuple[dict, dict | None]:
    # Fetch latest NOAA observation and build UTC datetime
    df_raw = fetch_ndbc_latest_df(station_id)
    df = build_datetime_utc(df_raw)

    # Check if observation is stale
    obs_utc_peek = df.loc[0, "datetime"]
    age_minutes = (datetime.now(timezone.utc) - obs_utc_peek).total_seconds() / 60.0
    if age_minutes > ALERT_STALE_MINUTES:
        ok, last_doc_mem = should_send_alert("stale_data", last_doc_mem)
        if ok:
            subject = f"[Kincardine Buoy] STALE DATA: station {station_id}"
            body = (
                f"Buoy station {station_id} appears to be reporting stale data.\n\n"
                f"Latest observation (UTC): {fmt(obs_utc_peek)}\n"
                f"Age (minutes): {age_minutes:.1f}\n"
                f"Threshold (minutes): {ALERT_STALE_MINUTES}\n"
                f"Recorded at (Toronto): {now_toronto_str(TORONTO_TZ)}\n"
            )
            try:
                send_email_smtp(subject, body)
            except Exception:
                logger.exception("Failed to send stale data alert email")

    # Add time features and direction sin/cos features for ML predictors
    df = add_month_hour_decimal(df)
    df = ensure_dir_sin_cos(df, "WDIR", "WDIRs", "WDIRc")
    df = ensure_dir_sin_cos(df, "MWD", "MWDs", "MWDc")

    # Load models and generate predictions
    preds: dict[str, float] = {}
    for out_col, fname in MODEL_FILES.items():
        path = os.path.join(PICKLE_DIR, fname)
        model = load_model(path)
        if isinstance(model, dict) and ("model_sin" in model) and ("model_cos" in model):
            preds[out_col] = predict_from_bundle(model, df)
        else:
            preds[out_col] = predict_scalar_model(model, df)
        safe_del(model)

    # Prepare inputs for hazard scoring
    haz_in = pd.DataFrame([{
        "wave_height_m": preds["wave_height_m"],
        "wave_dir_deg": preds["wave_dir_deg"],
        "wave_period_s": preds["wave_period_s"],
        "wind_speed_ms": preds["wind_speed_ms"],
        "wind_dir_deg": preds["wind_dir_deg"],
    }])
    haz_in["wave_dir_deg"] = np.abs(circ_diff_deg(ONSHORE_DEG, haz_in["wave_dir_deg"]))
    haz_in["wind_dir_deg"] = np.abs(circ_diff_deg(ONSHORE_DEG, haz_in["wind_dir_deg"]))

    # Compute hazard factors and risk level
    haz_out = pred_haz(haz_in).iloc[0].to_dict()

    # Build output document
    obs_utc = df.loc[0, "datetime"]
    obs_tor = obs_utc.astimezone(TORONTO_TZ)
    ing_utc = datetime.now(timezone.utc)
    ing_tor = ing_utc.astimezone(TORONTO_TZ)
    doc = {
        "recorded_at_utc": fmt(ing_utc),
        "recorded_at_toronto": fmt(ing_tor),
        "timestamp_utc": fmt(obs_utc),
        "timestamp_toronto": fmt(obs_tor),
        "station_id": station_id,
        "wave_height_m": float(preds["wave_height_m"]),
        "wave_dir_deg": float(haz_in.loc[0, "wave_dir_deg"]),
        "wave_period_s": float(preds["wave_period_s"]),
        "wind_speed_ms": float(preds["wind_speed_ms"]),
        "wind_dir_deg": float(haz_in.loc[0, "wind_dir_deg"]),
        "wave_factor": float(haz_out["wave_factor"]),
        "period_factor": float(haz_out["period_factor"]),
        "wind_factor": float(haz_out["wind_factor"]),
        "total_score": float(haz_out["total_score"]),
        "risk_level": str(haz_out["risk_level"]),
    }
    return doc, last_doc_mem
