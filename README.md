# Kincardine Buoy Risk Prediction System

This project is a real-time hazard prediction system for Kincardine Beach (Lake Huron, Ontario).

It uses offshore buoy observations and forecast data to:
- Predict nearshore wave and wind conditions using machine learning models
- Compute a swimmer hazard score
- Classify risk levels (Low / Moderate / High / Extreme)
- Store results and optionally send outputs to external systems (e.g., SwimSmart)

---

## System Overview

There are two main pipelines:

### 1. Real-Time Pipeline (NOAA Buoy)
- Uses NOAA NDBC buoy data (station 41049)
- Runs hourly
- Produces current hazard conditions

### 2. Forecast Pipeline (ECMWF)
- Uses Open-Meteo ECMWF weather and marine APIs
- Produces hourly and 3-hourly forecasts
- Computes hazard for future conditions

Both pipelines use the same hazard model.

---

## Project Structure


kincardine_buoy_risk/
│
├── main.py
├── api.py
├── pipeline.py
├── risk_predict.py
├── noaa.py
├── ecmwf_forecast.py
├── firestore.py
├── swimsmart.py
├── email_alert.py
├── helper.py
├── config.py
│
├── prediction/
├── risk model/
├── survey/


---

## Setup

### Clone the repository

git clone https://github.com/isabella-whw/kincardine_buoy_risk.git

cd kincardine_buoy_risk


### Install dependencies

pip install -r requirements.txt


---

## Model Files

### Default Location

prediction/pickle/


Defined in `config.py`:

PICKLE_DIR = "./prediction/pickle"


### Required Files

WaveHeight.pkl
WavePeriod.pkl
WindSpeed.pkl
WaveDirection.pkl
WindDirection.pkl


---

## Adding Model Files

### Option 1 — Copy Existing Models (Recommended)

Place files into:

prediction/pickle/


---

### Option 2 — Train Models

Run:

python Mtrain.py
python Ctrain.py


Datasets:
- `AllDat5.csv` → wave height, period, directions
- `AllDat5_2023.csv` → wind speed

Wind speed requires:

df = pd.read_csv(os.path.join(BASE_DIR, "AllDat5_2023.csv"))


### Notes on Model Files and Git

Model `.pkl` files are not tracked in Git by default.

Add the following to `.gitignore`:

prediction/pickle/
prediction/ERA5/pickle/
*.pkl


This prevents large model files from being uploaded to GitHub.

#### Important

- Each user must provide their own `.pkl` files locally (by copying or training)
- The system will not run without these files

#### When Deploying

If model files are required for deployment:

1. Temporarily remove or comment out the `.gitignore` rules
2. Deploy to Cloud Run
3. Restore `.gitignore` afterward to avoid tracking models permanently


---

## Cloud Deployment and Execution

### 1. Deploy to Cloud Run

After modifying code or models:


gcloud run deploy kincardine-test
--source .
--region us-central1
--allow-unauthenticated


This updates the live service.

---

### 2. Manually Run Jobs


gcloud scheduler jobs run kincardine-hourly --location=us-central1
gcloud scheduler jobs run ecmwf-forecast --location=us-central1
gcloud scheduler jobs run ecmwf-hourly --location=us-central1


---

### 3. What Each Job Does

- `kincardine-hourly`  
  Runs NOAA real-time prediction → updates `predictions`

- `ecmwf-forecast`  
  Fetches forecast data → updates `forecast`

- `ecmwf-hourly`  
  Runs forecast hazard model → updates `predictions_ecmwf`

---

## Verification and Testing

### Check Cloud Scheduler

https://console.cloud.google.com/cloudscheduler?project=kincardine-buoy-test

Verify:
- Job status = Success
- Last run updated

---

### Check Cloud Run Logs

https://console.cloud.google.com/run/detail/us-central1/kincardine-test/logs?project=kincardine-buoy-test

Look for:
- Successful `/run_once` execution
- No errors

---

### Check Firestore

https://console.cloud.google.com/firestore/databases/-default-/data/panel?project=kincardine-buoy-test

#### Collections

##### predictions

predictions/{station_id}/history/{timestamp}


##### predictions_ecmwf

predictions_ecmwf/ecmwf/history/{timestamp}


##### forecast

forecast/{run_timestamp}/hourly
forecast/{run_timestamp}/three_hourly


##### alerts

alerts/{alert_id}


---

### Verify Data

Check that:
- New timestamps appear
- Fields update:
  - wave_height_m
  - total_score
  - risk_level

---

### Test via API

https://kincardine-test-448213829784.us-central1.run.app/docs

Run:

POST /run_once
GET /latest


---

## Typical Workflow

1. Modify code or models  
2. Deploy to Cloud Run  
3. Run scheduler jobs manually  
4. Check:
   - Scheduler → success
   - Logs → no errors
   - Firestore → data updated  
5. Validate results via API  

---

## Hazard Model


total_score = wave_factor + max_wave_factor + period_factor + wind_factor


Risk Levels:

| Score | Risk |
|------|-----|
| < 3 | Low |
| 3–7 | Moderate |
| 7–11 | High |
| > 11 | Extreme |

---

## Notes

- Directions are converted relative to shoreline (315°)
- Rolling 12-hour wave height captures sustained conditions
- Models trained using NOAA and ERA5 datasets
- Firestore stores all outputs and history
