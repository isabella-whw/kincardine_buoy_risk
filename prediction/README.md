# Prediction Module

This folder contains machine learning models, training scripts, and datasets used to predict nearshore conditions.

---

## Structure

### Root (`prediction/`)
- `Mtrain.py` — trains scalar models (wave height, wave period, wind speed)
- `Ctrain.py` — trains circular models (wave direction, wind direction)
- `Cstats.py` — helper functions for circular statistics
- `AllDat5.csv` — main dataset
- `AllDat5_2023.csv` — dataset used for wind speed model
- `report_*.pdf` — training evaluation outputs

---

## Model Output

Trained models are saved as `.pkl` files:


prediction/pickle/


Required:

WaveHeight.pkl
WavePeriod.pkl
WindSpeed.pkl
WaveDirection.pkl
WindDirection.pkl


---

## ERA5 Subfolder

The `ERA5/` folder contains models and experiments using ERA5 reanalysis data.

---

## Usage

To train models:


python Mtrain.py
python Ctrain.py


Ensure datasets are available before running.

---

## Notes

- Scalar models predict numeric values (height, period, speed)
- Circular models predict direction using sine/cosine transformation
- `.pkl` files are required by the main pipeline
