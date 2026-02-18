from functools import lru_cache
from google.cloud import firestore
from datetime import date

from config import USE_FIRESTORE
from helper import fmt

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

def read_history_range(station_id: str, start_dt_s: str, end_dt_s: str, limit: int = 500) -> list[dict]:
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
