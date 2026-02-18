from fastapi import FastAPI, HTTPException, Query
from datetime import datetime, date
from zoneinfo import ZoneInfo

from config import STATION_ID_DEFAULT, TORONTO_TZ, logger
from pipeline import make_prediction
from firestore import write_latest, read_latest, date_to_range_strings, read_history_range
from email_alert import should_send_alert, send_email_smtp
from helper import now_toronto_str

def build_app() -> FastAPI:
    app = FastAPI(title="Kincardine Buoy Prediction API", version="1.0")
    app.state.LAST_DOC = None

    @app.get("/health", include_in_schema=False)
    def health():
        return {"ok": True}

    @app.post("/run_once", include_in_schema=False)
    def run_once():
        try:
            doc, new_mem = make_prediction(STATION_ID_DEFAULT, app.state.LAST_DOC)
            app.state.LAST_DOC = new_mem if new_mem is not None else doc
            write_latest(doc)
            return {"ok": True, "stored": doc}
        except Exception as e:
            logger.exception("run_once failed")

            ok, new_mem = should_send_alert("run_once_failed", app.state.LAST_DOC)
            app.state.LAST_DOC = new_mem if new_mem is not None else app.state.LAST_DOC

            if ok:
                subject = f"[Kincardine Buoy] RUN FAILED: station {STATION_ID_DEFAULT}"
                body = (
                    f"run_once failed for station {STATION_ID_DEFAULT}.\n\n"
                    f"Recorded at (Toronto): {now_toronto_str(TORONTO_TZ)}\n"
                    f"Error: {repr(e)}\n"
                )
                try:
                    send_email_smtp(subject, body)
                except Exception:
                    logger.exception("Failed to send failure alert email")

            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/latest")
    def latest():
        doc = read_latest(STATION_ID_DEFAULT, app.state.LAST_DOC)
        if not doc:
            raise HTTPException(status_code=404, detail="No prediction stored yet")
        return doc

    @app.get("/predictions")
    def predictions(
        startDate: date = Query(..., description="YYYY-MM-DD (required, Toronto local date)"),
        endDate: date | None = Query(None, description="YYYY-MM-DD (optional; defaults to today Toronto)"),
    ):
        if endDate is None:
            endDate = datetime.now(TORONTO_TZ).date()
        if endDate < startDate:
            raise HTTPException(status_code=400, detail="endDate must be >= startDate")

        start_s, end_s = date_to_range_strings(startDate, endDate)
        items = read_history_range(STATION_ID_DEFAULT, start_s, end_s)

        return {
            "station_id": STATION_ID_DEFAULT,
            "startDate": startDate.isoformat(),
            "endDate": endDate.isoformat(),
            "count": len(items),
            "items": items,
        }

    return app
