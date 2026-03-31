# Methods

This document summarizes the main data sources, preprocessing steps, prediction workflow, hazard scoring framework, forecast workflow, and survey-analysis methods used in the Kincardine Buoy Hazard Prediction System.

---

## 1. Data Sources

### 1.1 NOAA NDBC Buoy Data
The real-time prediction workflow uses offshore buoy observations retrieved from NOAA NDBC station data.

These observations provide the primary environmental inputs used for nearshore prediction.

Typical variables include:

- wind direction
- wind speed
- gust speed
- wave height
- dominant period
- average period
- mean wave direction
- pressure
- air temperature
- water temperature
- dew point

### 1.2 ECMWF Forecast Data
Forecast weather and marine conditions are retrieved using Open-Meteo APIs with ECMWF-backed forecast products.

These include:

#### Weather variables
- air temperature
- relative humidity
- dew point
- precipitation
- pressure
- cloud cover
- sunshine duration
- wind gusts
- wind speed

#### Marine variables
- wave height
- wave direction
- wave peak period
- wave period

### 1.3 Water Level Data
Water level data are retrieved separately for water-level-based hazard adjustment and related testing.

### 1.4 Survey Data
Survey-response data are analyzed separately to compare perceived swimmer hazard with matched environmental conditions.

---

## 2. Preprocessing and Feature Engineering

### 2.1 Time Construction
NOAA observations are parsed into structured tabular format and converted into UTC datetime values.

### 2.2 Derived Features
Additional predictor variables are derived before prediction.

These include:

- month
- decimal hour
- directional sine/cosine components for:
  - wind direction
  - wave direction

### 2.3 Directional Variables
Circular variables are represented using sine/cosine transformations for modeling and reconstructed later into directional angles.

---

## 3. Nearshore Prediction Workflow

The operational workflow uses trained machine learning models to estimate nearshore environmental conditions from offshore buoy observations.

The predicted variables are:

- wave height
- wave period
- wind speed
- wave direction
- wind direction

These predicted values are then passed into the hazard scoring framework.

---

## 4. Direction Handling

Wave and wind directions are converted relative to shoreline orientation before hazard scoring.

The system uses an onshore reference angle of **315°**.

Angular differences are computed using circular difference methods so that directional scoring reflects approach relative to shore rather than raw compass direction.

---

## 5. Hazard Scoring Framework

Hazard classification is based on a rule-based scoring framework applied to the predicted environmental conditions.

The main operational inputs are:

- wave height
- wave direction relative to shore
- wave period
- wind speed
- wind direction relative to shore

Historical testing versions also include additional scoring variants, such as:

- modified wave-angle scoring
- re-scaled score bins
- weighted scoring systems
- maximum wave height over the previous 12 hours

Different hazard scoring systems tested during development are archived separately in the repository.

---

## 6. Forecast Workflow

A separate forecast workflow retrieves weather and marine forecast data and combines them into forecast snapshots.

This workflow includes:

1. weather forecast retrieval
2. marine forecast retrieval
3. UTC timestamp alignment
4. merged forecast generation
5. hourly and 3-hourly forecast formatting

These forecast outputs are stored for later retrieval and comparison.

---

## 7. Survey Analysis Workflow

A separate survey-analysis workflow is used to compare survey responses with forecast and matched environmental conditions.

This workflow includes:

1. loading Qualtrics survey responses
2. filtering and cleaning responses
3. converting selected response categories to numeric values
4. computing descriptive statistics
5. computing class-based forecast summaries
6. matching photo responses to environmental timestamps
7. computing photo-level and class-based summaries

The survey workflow is used to compare model-based hazard conditions with perceived hazard from respondents.

---

## 8. Historical Model Testing

Multiple hazard scoring systems were tested during development.

These include:

- the original Great Lakes Rip Current Checklist (GLRCC)
- thumbnail-based scoring versions
- re-scaled scoring systems
- versions including 12-hour wave-height context
- weighted scoring systems

These versions were used for historical reruns and comparison of hazard outputs.

---

## 9. Output Categories

The repository produces several categories of outputs, including:

- real-time hazard predictions
- historical prediction reruns
- forecast snapshots
- PDF summaries and plots
- survey summary outputs
- matched-condition comparison tables

---