# Risk Model Module

This folder contains different versions of the hazard scoring model.

---

## Structure

Each subfolder represents a different model version:

- `GLRCC/` — baseline model based on Great Lakes Rip Current Checklist
- `Thumbnail 1–6/` — experimental hazard models
- `Thumbnail 3 rescaled`, `6h`, `12h` — variations of scoring methods

---

## Contents per Folder

Typical files:
- `Hazards*.py` — hazard calculation logic
- `Cstats.py` — circular statistics
- `NOAA_*predictions*.csv` — input data
- `photo_hazard_table.csv` — survey comparison
- `*_output.pdf` — model evaluation reports

---

## Purpose

- Develop and compare different hazard scoring approaches
- Tune thresholds and factor weights
- Validate model performance against observed data and survey responses

---

## Notes

- Not all models are used in production
- The current production hazard model is based on **Thumbnail 6**
- The logic from `Thumbnail 6` has been implemented in `risk_predict.py`
