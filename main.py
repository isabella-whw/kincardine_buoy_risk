# main.py
# Application entry point for Cloud Run deployment.
# Exposes the FastAPI app instance.

from api import build_app

app = build_app()
