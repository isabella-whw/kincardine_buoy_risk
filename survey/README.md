# Survey Analysis

This folder contains the latest survey analysis workflow used to evaluate public hazard perception and compare responses with environmental conditions.

---

## Contents

### `survey.py`
Main analysis script for processing survey responses.

### `Great Lakes Surf Zone Hazards_...csv`
Raw Qualtrics survey export.

### `kincardine_timestamps.csv`
Matched environmental/photo timestamp file used to connect survey photo responses to corresponding environmental conditions.

### `survey_outputs/`
Generated output files from the survey analysis.

---

## What the Script Does

The survey workflow performs the following steps:

1. Loads the Qualtrics survey export
2. Cleans and filters completed responses
3. Converts selected response categories to numeric values
4. Computes descriptive statistics for:
   - ranking questions
   - wave-height questions
   - wave-period questions
   - wind-speed questions
   - photo-response questions
5. Builds class-based summaries for forecast-response questions
6. Matches photo-response questions to environmental conditions
7. Computes:
   - photo-level summaries
   - class-based photo summaries
8. Exports all outputs to `survey_outputs/`

---

## Output Files

The `survey_outputs/` folder contains:

- `cleaned_responses.csv`
- `quantitative_stats.csv`
- `forecast_class_stats.csv`
- `photo_stats.csv`
- `photo_class_stats.csv`
- `text_responses.csv`

---

## Notes

- This folder currently contains the latest survey analysis version only.
- Earlier survey-analysis variants are not archived here.
