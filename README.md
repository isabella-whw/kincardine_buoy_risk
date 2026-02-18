# Kincardine Buoy Hazard Prediction System

This repository contains the complete implementation of the real-time hazard prediction system for Kincardine Beach (Lake Huron, Ontario).
The system retrieves real-time offshore buoy observations from NOAA NDBC Station 45008, applies trained machine learning models to predict nearshore conditions, computes a quantitative hazard score using a rule-based scoring framework, and stores results in Google Firestore. 
The system is deployed using Google Cloud Run and is automatically triggered hourly via Cloud Scheduler.


# System Overview

The operational pipeline performs the following steps:
1. Fetch real-time NOAA offshore buoy data
2. Construct predictor variables and directional features
3. Apply trained machine learning models to predict:
   - Significant wave height
   - Dominant wave period
   - Wind speed
   - Wave direction
   - Wind direction
4. Convert wave and wind directions relative to the onshore reference angle (315°)
5. Compute a hazard score and risk classification
6. Store predictions in Firestore
7. Trigger automated email alerts for:
   - System failure
   - Stale buoy data


# Repository Structure

- api.py — FastAPI application and endpoints
- pipeline.py — Core prediction and scoring workflow
- noaa.py — NOAA data retrieval and preprocessing
- risk_predict.py — Hazard scoring implementation
- firestore.py — Firestore read/write logic
- email_alert.py — Email notification logic
- config.py — Environment configuration
- helper.py — Utility functions
- main.py — Application entry point
- requirements.txt — Python dependencies


# Deployment Architecture

- Google Cloud Run: Serverless API deployment
- Firestore database: Prediction storage
- Cloud Scheduler: Hourly automated execution
- SMTP email alert integration: Email alert notifications


# API Endpoints
### POST /run_once
Triggers one prediction cycle manually.

### GET /latest
Returns the most recent prediction stored in Firestore.

### GET /predictions
Returns historical predictions within a specified date range:

/predictions?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD


## For detailed methodology, see [METHODS.md](METHODS.md).
