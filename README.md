# Kincardine Buoy Hazard Prediction System

This repository contains the full implementation of the **Kincardine Beach hazard prediction system** for Lake Huron, Ontario.

The project combines:
- real-time environmental observations
- machine learning prediction models
- rule-based hazard scoring
- forecast generation
- historical model testing
- survey-based comparison workflows

The system was developed to estimate swimmer hazard conditions and support beach safety communication.

---

## Project Overview

The operational system uses offshore buoy observations and forecast data to estimate nearshore environmental conditions and classify swimmer hazard levels.

The primary environmental variables used in the hazard framework are:

- wave height
- wave direction
- wave period
- wind speed
- wind direction

Several hazard scoring systems were tested during development, including:
- the original **Great Lakes Rip Current Checklist (GLRCC)**
- multiple **thumbnail-based scoring versions**
- re-scaled scoring variants
- versions including **12-hour maximum wave height**

The active operational code is stored in the root repository, while historical model runs are archived separately.

---

## Main Components

### 1. Real-Time Prediction Pipeline
The real-time system:

1. Retrieves the latest offshore buoy observation from **NOAA NDBC**
2. Preprocesses predictor variables
3. Applies trained machine learning models to estimate nearshore conditions
4. Converts directional predictions relative to the shoreline orientation
5. Computes a swimmer hazard score
6. Assigns a hazard classification
7. Stores outputs in **Google Firestore**
8. Optionally pushes outputs to **SwimSmart**
9. Sends automated email alerts when needed

### 2. Forecast Pipeline
The repository also includes a forecast workflow that retrieves **ECMWF weather and marine forecasts** and stores forecast snapshots for later use.

### 3. Historical Model Testing
The repository includes archived scoring versions and historical reruns used to compare different hazard scoring approaches.

### 4. Survey Analysis
The repository also includes a survey-analysis workflow used to compare:
- public hazard perception
- forecast-based conditions
- matched environmental / photo conditions

---

## Repository Structure

```text
kincardine_buoy_risk/
│
├── prediction/
│   ├── README.md
│   └── pickle/
│       ├── WaveHeight.pkl
│       ├── WavePeriod.pkl
│       ├── WindSpeed.pkl
│       ├── WaveDirection.pkl
│       └── WindDirection.pkl
│
├── risk model/
│   ├── README.md
│   ├── GLRCC/
│   ├── Thumbnail 1/
│   ├── Thumbnail 2/
│   ├── Thumbnail 3/
│   ├── Thumbnail 3 rescaled/
│   ├── Thumbnail 3 rescaled 12h/
│   └── Thumbnail 4/
│
├── survey/
│   ├── README.md
│   ├── survey_outputs/
│   ├── Great Lakes Surf Zone Hazards_...
│   ├── kincardine_timestamps.csv
│   └── survey.py
│
├── api.py
├── config.py
├── ecmwf_forecast.py
├── email_alert.py
├── firestore.py
├── helper.py
├── main.py
├── METHODS.md
├── noaa.py
├── pipeline.py
├── README.md
├── requirements.txt
├── risk_predict.py
├── swimsmart.py
└── tobermory.py
