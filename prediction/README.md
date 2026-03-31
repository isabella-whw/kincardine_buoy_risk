# Prediction Models

This folder contains the trained machine learning models used by the operational Kincardine hazard prediction pipeline.

---

## Contents

### `pickle/`
Contains the five production model files:

- `WaveHeight.pkl`
- `WavePeriod.pkl`
- `WindSpeed.pkl`
- `WaveDirection.pkl`
- `WindDirection.pkl`

---

## Model Purpose

These models are used to predict nearshore environmental conditions from offshore buoy observations.

Predicted outputs include:
- wave height
- wave period
- wind speed
- wave direction
- wind direction

These predictions are then passed into the hazard scoring framework in the operational pipeline.

---

## Notes

- These are the production model files used by the real-time system.
- Training scripts and historical model development files are stored elsewhere in the repository.
