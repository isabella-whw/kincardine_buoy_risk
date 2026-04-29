# pipeline.py
# Core prediction pipeline: runs NOAA and ECMWF hazard predictions,
# applies ML models, computes hazard score, and returns Firestore-ready documents.

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from config import (
    SWIMSMART_SOURCE,
    STATION_ID_DEFAULT,
    PICKLE_DIR,
    PICKLE_DIR_NO_WTMP,
    MODEL_FILES,
    DEFAULT_PREDICTORS,
    DEFAULT_PREDICTORS_NO_WTMP,
    TORONTO_TZ,
    ONSHORE_DEG,
    ALERT_STALE_MINUTES,
    logger,
)
from helper import (
    fmt,
    now_toronto_str,
    circ_diff_deg,
    load_model,
    predict_from_bundle,
    predict_scalar_model,
    safe_del,
)
from risk_predict import pred_haz
from noaa import (
    fetch_ndbc_recent_df,
    build_datetime_utc,
    add_month_hour_decimal,
    ensure_dir_sin_cos,
)
from email_alert import send_email_smtp, should_send_alert
from swimsmart import send_prediction_to_swimsmart
from ecmwf_forecast import fetch_ecmwf_forecast_df, build_ecmwf_ml_input_row, build_ecmwf_ml_input_row_with_wtmp

# Run all trained ML models for a single input row and return predictions.
def _run_models(
    feature_row: pd.DataFrame,
    pickle_dir: str,
    model_files: dict[str, str],
    default_predictors: list[str],
) -> dict[str, float]:
    preds: dict[str, float] = {}
    for out_col, fname in model_files.items():
        path = os.path.join(pickle_dir, fname)
        model = load_model(path)
        if isinstance(model, dict) and ("model_sin" in model) and ("model_cos" in model):
            preds[out_col] = predict_from_bundle(model, feature_row)
        else:
            preds[out_col] = predict_scalar_model(model, feature_row, default_predictors)
        safe_del(model)
    return preds

# Build a standardized prediction document from model outputs.
# Converts directions relative to shoreline, computes hazard score, and handles SwimSmart sending logic.
def _build_doc_from_prediction(
    timestamp_utc,
    station_id: str,
    preds: dict[str, float],
    max_wave_height_12h_m: float,
) -> dict:
    haz_in = pd.DataFrame([{
        "wave_height_m": preds["wave_height_m"],
        "wave_dir_deg": preds["wave_dir_deg"],
        "wave_period_s": preds["wave_period_s"],
        "wind_speed_ms": preds["wind_speed_ms"],
        "wind_dir_deg": preds["wind_dir_deg"],
        "max_wave_height_12h_m": float(max_wave_height_12h_m),
    }])
    haz_in["wave_dir_deg"] = np.abs(circ_diff_deg(ONSHORE_DEG, haz_in["wave_dir_deg"]))
    haz_in["wind_dir_deg"] = np.abs(circ_diff_deg(ONSHORE_DEG, haz_in["wind_dir_deg"]))
    haz_out = pred_haz(haz_in).iloc[0].to_dict()
    obs_utc = pd.to_datetime(timestamp_utc, utc=True)
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
        "max_wave_height_12h_m": float(haz_in.loc[0, "max_wave_height_12h_m"]),
        "wave_factor": float(haz_out["wave_factor"]),
        "max_wave_factor": float(haz_out["max_wave_factor"]),
        "period_factor": float(haz_out["period_factor"]),
        "wind_factor": float(haz_out["wind_factor"]),
        "total_score": float(haz_out["total_score"]),
        "risk_level": str(haz_out["risk_level"]),
    }

    # Determine if this prediction is the active source for SwimSmart
    # Only one source (NOAA or ECMWF) is allowed to send at a time
    active = (
        (SWIMSMART_SOURCE == "noaa" and station_id == STATION_ID_DEFAULT) or
        (SWIMSMART_SOURCE == "ecmwf" and station_id == "ecmwf") or
        (SWIMSMART_SOURCE == "ecmwf_wtmp" and station_id == "ecmwf_wtmp")
    )
    doc["swimsmart_active_source"] = SWIMSMART_SOURCE
    doc["sent_to_swimsmart"] = active
    if active:
        try:
            swimsmart_results = send_prediction_to_swimsmart(doc)
            doc["swimsmart_sent"] = bool(swimsmart_results)
            doc["swimsmart_results"] = swimsmart_results
        except Exception as e:
            logger.exception("SwimSmart send failed")
            doc["swimsmart_sent"] = False
            doc["swimsmart_error"] = repr(e)
    else:
        doc["swimsmart_sent"] = False
    return doc

# Run ECMWF-based hazard prediction using forecast data.
def make_ecmwf_prediction(last_doc_mem: dict | None) -> tuple[dict, dict | None]:
    df_ecmwf = fetch_ecmwf_forecast_df()
    feature_row, max_wave_height_12h_m = build_ecmwf_ml_input_row(df_ecmwf)
    preds = _run_models(
        feature_row,
        PICKLE_DIR_NO_WTMP,
        MODEL_FILES,
        DEFAULT_PREDICTORS_NO_WTMP,
    )
    timestamp_utc = feature_row.loc[0, "datetime"]
    doc = _build_doc_from_prediction(
        timestamp_utc,
        "ecmwf",
        preds,
        max_wave_height_12h_m,
    )
    return doc, last_doc_mem

def make_ecmwf_prediction_with_wtmp(last_doc_mem: dict | None) -> tuple[dict, dict | None]:
    df_ecmwf = fetch_ecmwf_forecast_df()
    feature_row, max_wave_height_12h_m = build_ecmwf_ml_input_row_with_wtmp(df_ecmwf)
    preds = _run_models(
        feature_row,
        PICKLE_DIR,
        MODEL_FILES,
        DEFAULT_PREDICTORS,
    )
    timestamp_utc = feature_row.loc[0, "datetime"]
    doc = _build_doc_from_prediction(
        timestamp_utc,
        "ecmwf_wtmp",
        preds,
        max_wave_height_12h_m,
    )
    return doc, last_doc_mem

# Run NOAA-based hazard prediction using buoy data. Includes stale data detection and alerting.
def make_prediction(station_id: str, last_doc_mem: dict | None) -> tuple[dict, dict | None]:
    df_raw = fetch_ndbc_recent_df(station_id)
    df = build_datetime_utc(df_raw)
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")
    df["max_wave_height_12h_m"] = df["WVHT"].astype(float).rolling("12h", min_periods=1).max()
    df = df.reset_index()
    latest_row = df.iloc[[-1]].reset_index(drop=True)
    obs_utc_peek = latest_row.loc[0, "datetime"]
    age_minutes = (datetime.now(timezone.utc) - obs_utc_peek).total_seconds() / 60.0
    if age_minutes > ALERT_STALE_MINUTES:
        ok, last_doc_mem = should_send_alert("stale_data", last_doc_mem)
        if ok:
            subject = f"[Kincardine Buoy] STALE DATA: station {station_id}"
            body = (
                f"NOAA buoy station {station_id} appears to be reporting stale data.\n\n"
                f"Latest observation (UTC): {fmt(obs_utc_peek)}\n"
                f"Age (minutes): {age_minutes:.1f}\n"
                f"Threshold (minutes): {ALERT_STALE_MINUTES}\n"
                f"Recorded at (Toronto): {now_toronto_str(TORONTO_TZ)}\n"
            )
            try:
                send_email_smtp(subject, body)
            except Exception:
                logger.exception("Failed to send stale data alert email")
    latest_row = add_month_hour_decimal(latest_row)
    latest_row = ensure_dir_sin_cos(latest_row, "WDIR", "WDIRs", "WDIRc")
    latest_row = ensure_dir_sin_cos(latest_row, "MWD", "MWDs", "MWDc")
    preds = _run_models(
        latest_row,
        PICKLE_DIR,
        MODEL_FILES,
        DEFAULT_PREDICTORS,
    )
    doc = _build_doc_from_prediction(
        latest_row.loc[0, "datetime"],
        station_id,
        preds,
        float(latest_row.loc[0, "max_wave_height_12h_m"]),
    )
    return doc, last_doc_mem
