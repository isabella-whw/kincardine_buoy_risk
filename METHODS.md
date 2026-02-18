# Methods

This document describes the data processing, machine learning prediction, and hazard scoring methodology implemented in this repository.


# 1. Data Source

Real-time offshore buoy observations are retrieved from NOAA NDBC Station 45008:

https://www.ndbc.noaa.gov/data/realtime2/

Variables used include:

- WVHT — Significant wave height (m)
- DPD — Dominant wave period (s)
- WSPD — Wind speed (m/s)
- WDIR — Wind direction (degrees)
- MWD — Mean wave direction (degrees)
- Additional environmental predictors (if required by trained models)

All timestamps are parsed as UTC and converted to Toronto local time for storage and reporting.



# 2. Feature Engineering

## 2.1 Timestamp Construction

UTC timestamps are constructed from buoy YY, MM, DD, hh, and mm fields.

Local Toronto time is computed for logging and database storage.


## 2.2 Directional Encoding

Wave and wind directions are transformed into sine and cosine components:

  sin(θ), cos(θ)

This avoids angular discontinuity issues near 0°/360°.

Predicted sin/cos components are later reconstructed into directional angles using:

  θ = arctan2(sin, cos)


## 2.3 Onshore Direction Transformation

To quantify onshore forcing, wave and wind directions are transformed relative to a fixed shoreline reference angle:

  onshore_reference = 315°

The relative forcing angle is computed as:

  relative_angle = | (direction − 315°) |

Smaller values indicate stronger onshore forcing.



# 3. Machine Learning Prediction

Five trained models are applied:

- WaveHeight.pkl
- WavePeriod.pkl
- WindSpeed.pkl
- WaveDirection.pkl
- WindDirection.pkl

Directional models are implemented using sine/cosine decomposition.

Models were trained using historical offshore NOAA buoy data aligned with nearshore conditions at Kincardine.

Each hourly execution produces predicted nearshore conditions.



# 4. Hazard Risk Scoring

Predicted environmental conditions are converted into a quantitative hazard score using a rule-based scoring system.

The following variables are used:

- Predicted significant wave height (m)
- Predicted dominant wave period (s)
- Predicted wind speed (m/s)
- Wave direction relative to onshore (degrees)
- Wind direction relative to onshore (degrees)


## 4.1 Wave Height Contribution

Wave height contributes increasing hazard points as magnitude increases.

Larger waves correspond to greater nearshore energy and increased rip current likelihood.


## 4.2 Wave Period Contribution

Longer wave periods increase hazard contribution due to higher wave energy and enhanced nearshore circulation.

Short-period waves contribute less to total hazard.


## 4.3 Wind Contribution

Wind contribution depends on:

- Wind speed magnitude
- Wind direction relative to onshore

Stronger onshore winds increase setup and circulation hazard.

Offshore winds reduce hazard contribution.



## 4.4 Total Hazard Score

The total hazard score is computed as:

  total_score = wave_factor + period_factor + wind_factor

Risk classification thresholds:

  - Low Risk: total_score < 4
  - Moderate Risk: 4 ≤ total_score ≤ 7
  - High Risk: total_score > 7

These thresholds are selected to provide stable, interpretable public-facing hazard categories.



# 5. Automation and Monitoring

The system runs automatically every hour using Google Cloud Scheduler.

Each prediction is stored in Firestore under:
  
  predictions/{station\_id}/history

Two automated alert mechanisms are implemented:

1. Stale Data Alert
   Triggered when buoy observations exceed a defined age threshold.

2. System Failure Alert  
  Triggered when the prediction pipeline raises an exception.

Alerts are sent via SMTP email.
