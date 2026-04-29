# Kincardine Buoy Risk Prediction System

This project is a real-time hazard prediction system for Kincardine Beach (Lake Huron, Ontario).

It uses offshore buoy observations and ECMWF forecast data to:
- Predict nearshore wave and wind conditions using machine learning models
- Compute a swimmer hazard score
- Classify risk levels (Low / Moderate / High)
- Store results and optionally send outputs to external systems (e.g., SwimSmart)

The system supports manual switching between NOAA and ECMWF for SwimSmart output.

---

## System Overview

There are two main pipelines:

### 1. Real-Time Pipeline (NOAA Buoy)
- Uses NOAA NDBC buoy data (station 41049)
- Runs hourly
- Produces current hazard conditions
- Writes results to `predictions`


### 2. Backup / Parallel Pipelines (ECMWF)

Two ECMWF-based pipelines run in parallel:

#### (a) ECMWF (No WTMP)
- Uses Open-Meteo ECMWF weather and marine APIs
- Runs hourly
- Uses the current ECMWF hour as model input
- Computes hazard conditions using the same hazard model
- Does NOT include water temperature
- Writes results to `predictions_ecmwf`

#### (b) ECMWF (With WTMP)
- Same as above, but includes WTMP as a model input
- WTMP is approximated using air temperature (no true water temperature available)
- Writes results to `predictions_ecmwf_wtmp`

All pipelines can run at the same time.

Only one source is sent to SwimSmart at a time, controlled manually by `SWIMSMART_SOURCE` in `config.py`.

---

## Manual Source Switching for SwimSmart

The active SwimSmart source is controlled by:

`SWIMSMART_SOURCE = "noaa"` or `SWIMSMART_SOURCE = "ecmwf"` or `SWIMSMART_SOURCE = "ecmwf_wtmp"`

Defined in `config.py`.

### NOAA active
```python
SWIMSMART_SOURCE = "noaa"
```
NOAA predictions are still generated
ECMWF predictions are still generated
Only NOAA is sent to SwimSmart

---

## Project Structure

```text
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
```

---

## Setup

### Clone the repository

```bash
git clone https://github.com/isabella-whw/kincardine_buoy_risk.git

cd kincardine_buoy_risk
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Model Files

### Default Location

```text
prediction/pickle/                 # WITH WTMP
prediction/No WTMP/pickle/         # WITHOUT WTMP
prediction/ERA5/pickle/            # ERA5 (optional)
```

Defined in `config.py`:

```text
PICKLE_DIR = "./prediction/pickle"
PICKLE_DIR_NO_WTMP = "./prediction/No WTMP/pickle"
```


### Required Files

Each model folder should contain the same five model filenames:
```text
WaveHeight.pkl
WavePeriod.pkl
WindSpeed.pkl
WaveDirection.pkl
WindDirection.pkl
```

Locations:
```text
prediction/pickle/                  # WITH WTMP models
prediction/No WTMP/pickle/          # WITHOUT WTMP models
prediction/ERA5/pickle/             # ERA5 models, if used
```

---

## Adding Model Files

### Option 1 — Copy Existing Models (Recommended)

Place files into:
```text
prediction/pickle/                 # WITH WTMP
prediction/No WTMP/pickle/         # WITHOUT WTMP
```

---

### Option 2 — Train Models

Run:

```bash
python Mtrain.py
python Ctrain.py
```

Datasets:
- `AllDat5.csv` → wave height, period, directions
- `AllDat5_2023.csv` → wind speed

Wind speed requires:

df = pd.read_csv(os.path.join(BASE_DIR, "AllDat5_2023.csv"))

Note: Model training may take approximately 10 minutes depending on hardware.


### Notes on Model Files and Git

Model `.pkl` files are not tracked in Git by default.

Add the following to `.gitignore`:

```text
*.pkl

!prediction/pickle/.gitkeep
!prediction/No WTMP/pickle/.gitkeep
!prediction/ERA5/pickle/.gitkeep
```

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

```bash
gcloud run deploy kincardine-test --source . --region us-central1 --allow-unauthenticated
```

This updates the live service.

---

### 2. Manually Run Jobs

```bash
gcloud scheduler jobs run kincardine-hourly --location=us-central1
gcloud scheduler jobs run ecmwf-hourly --location=us-central1
gcloud scheduler jobs run ecmwf-wtmp-hourly --location=us-central1
```

---

### 3. What Each Job Does

- `kincardine-hourly`  
  Runs NOAA real-time prediction → updates `predictions`

- `ecmwf-hourly`  
  Runs ECMWF prediction without WTMP → updates `predictions_ecmwf`

- `ecmwf-wtmp-hourly`  
  Runs ECMWF prediction with WTMP → updates `predictions_ecmwf_wtmp`

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
- Successfulessful `/run_ecmwf_once` execution
- Successful `/run_ecmwf_wtmp_once` execution
- No errors

---

### Check Firestore

https://console.cloud.google.com/firestore/databases/-default-/data/panel?project=kincardine-buoy-test

#### Collections

##### predictions

predictions/{station_id}/history/{timestamp}


##### predictions_ecmwf

predictions_ecmwf/ecmwf/history/{timestamp}


##### predictions_ecmwf_wtmp

predictions_ecmwf_wtmp/ecmwf_wtmp/history/{timestamp}


##### alerts

alerts/{alert_id}


---

### Verify Data

Check that new timestamps appear and fields update:
- wave_height_m
- max_wave_height_12h_m
- wave_factor
- max_wave_factor
- period_factor
- wind_factor
- total_score
- risk_level

---

### Test via API

https://kincardine-test-448213829784.us-central1.run.app/docs

Run:

POST /run_once
GET  /latest

POST /run_ecmwf_once
GET  /latest_ecmwf

POST /run_ecmwf_wtmp_once
GET  /latest_ecmwf_wtmp


---

## Typical Workflow

1. Modify code or models  
2. Deploy to Cloud Run  
3. Run scheduler jobs:
   - `kincardine-hourly`
   - `ecmwf-hourly`
   - `ecmwf-wtmp-hourly`
4. Check:
   - Scheduler → success
   - Logs → no errors
   - Firestore → data updated  
5. Validate results via API
6. Set SWIMSMART_SOURCE manually:
   - noaa
   - ecmwf
   - ecmwf_wtmp

---

## GitHub Update Workflow

After confirming everything works, update GitHub:

```bash
git status
git add .
git commit -m "your commit message"
git pull --rebase origin master
git push
```

If Git reports no changes, you can skip commit and push.

---
## Hazard Model


total_score = wave_factor + max_wave_factor + period_factor + wind_factor


Risk Levels:

| Score | Risk |
|------|-----|
| < 3 | Low |
| 3–7 | Moderate |
| >= 7 | High |

---

## Notes

- Directions are converted relative to shoreline (315°)
- Rolling 12-hour wave height captures sustained conditions
- Models trained using NOAA and ERA5 datasets
- NOAA, ECMWF (no WTMP), and ECMWF (with WTMP) pipelines run in parallel
- SwimSmart output is controlled manually using `SWIMSMART_SOURCE`
- Firestore stores both latest outputs and history
- Model .pkl files are not tracked by Git and must exist locally
