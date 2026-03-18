# firestore.py
# Handles Firestore database interactions for storing and retrieving predictions.
# Supports latest prediction storage, historical query operations, and forecast storage.

import pandas as pd

from functools import lru_cache
from google.cloud import firestore
from datetime import date

from config import USE_FIRESTORE

@lru_cache(maxsize=1)
def get_db():
    if not USE_FIRESTORE:
        return None
    return firestore.Client()

def history_doc_id(doc: dict) -> str:
    return doc["recorded_at_toronto"].replace(" ", "_").replace(":", "-")

def write_latest(doc: dict) -> None:
    if not USE_FIRESTORE:
        return
    db = get_db()
    if db is None:
        return
    station_id = doc["station_id"]
    db.collection("predictions").document(station_id).set(doc)
    hist_id = history_doc_id(doc)
    db.collection("predictions") \
        .document(station_id) \
        .collection("history") \
        .document(hist_id) \
        .set(doc)

def read_latest(station_id: str, last_doc_mem: dict | None) -> dict | None:
    if not USE_FIRESTORE:
        if last_doc_mem and last_doc_mem.get("station_id") == station_id:
            return last_doc_mem
        return None
    db = get_db()
    if db is None:
        return None
    snap = db.collection("predictions").document(station_id).get()
    return snap.to_dict() if snap.exists else None

def date_to_range_strings(start: date, end: date) -> tuple[str, str]:
    start_s = f"{start.isoformat()} 00:00:00"
    end_s = f"{end.isoformat()} 23:59:59"
    return start_s, end_s

def read_history_range(
    station_id: str,
    start_dt_s: str,
    end_dt_s: str,
    limit: int = 500,
) -> list[dict]:
    if not USE_FIRESTORE:
        return []
    db = get_db()
    if db is None:
        return []
    q = (
        db.collection("predictions")
        .document(station_id)
        .collection("history")
        .where("recorded_at_toronto", ">=", start_dt_s)
        .where("recorded_at_toronto", "<=", end_dt_s)
        .order_by("recorded_at_toronto")
        .limit(limit)
    )
    return [d.to_dict() for d in q.stream()]

def _forecast_row_doc_id(row: dict) -> str:
    return str(row["time_utc"]).replace(" ", "_").replace(":", "-")

def write_forecast(snapshot: dict) -> None:
    if not USE_FIRESTORE:
        return

    db = get_db()
    if db is None:
        return

    retrieved_at_toronto = (
        pd.to_datetime(snapshot["retrieved_at_utc"], utc=True)
        .tz_convert("America/Toronto")
        .strftime("%Y-%m-%d_%H-%M-%S")
    )
    run_id = retrieved_at_toronto

    meta = {
        "retrieved_at_utc": snapshot["retrieved_at_utc"],
        "latitude": snapshot["latitude"],
        "longitude": snapshot["longitude"],
        "hourly_count": snapshot["hourly_count"],
        "three_hourly_count": snapshot["three_hourly_count"],
    }

    run_ref = db.collection("forecast").document(run_id)
    run_ref.set(meta)

    batch = db.batch()
    ops = 0

    for row in snapshot.get("hourly_records", []):
        doc_id = _forecast_row_doc_id(row)
        batch.set(run_ref.collection("hourly").document(doc_id), row)
        ops += 1
        if ops == 400:
            batch.commit()
            batch = db.batch()
            ops = 0

    for row in snapshot.get("three_hourly_records", []):
        doc_id = _forecast_row_doc_id(row)
        batch.set(run_ref.collection("three_hourly").document(doc_id), row)
        ops += 1
        if ops == 400:
            batch.commit()
            batch = db.batch()
            ops = 0

    if ops > 0:
        batch.commit()