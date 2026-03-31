# Risk Model Runs

This folder contains the historical hazard scoring systems and model-run outputs tested during development of the Kincardine swimmer hazard prediction system.

Each subfolder represents a different hazard scoring framework or scoring version used during testing.

---

## Overview

The files in this folder document multiple scoring systems used to classify swimmer hazard conditions from predicted environmental variables.

These folders include versions that differ in one or more of the following:

- scoring structure
- wave height bins
- wave direction / angle bins
- wave period bins
- wind speed bins
- inclusion of additional variables
- use of weighted scoring
- hazard classification thresholds

Each model folder typically contains:

- a `.png` file showing the scoring scale used
- one or more Python scripts used to generate predictions
- CSV prediction outputs
- PDF summaries or figures
- in some cases, survey comparison outputs

---

## Folder Descriptions

### `GLRCC/`
Contains the original **Great Lakes Rip Current Checklist (GLRCC)** scoring framework.

This is a standalone checklist-based system and is structurally different from the thumbnail models.

Key characteristics:

- Includes the following factors:
  - wave height and angle of approach
  - wave period
  - wind speed
  - lake level relative to normal
- Scoring is based on adding individual factor scores
- Risk classification:
  - Low: < 4  
  - Moderate: 4 – 7  
  - High: > 7  

This folder includes the GLRCC reference scale and related outputs.

---

### `Thumbnail 1/`
Contains the first thumbnail-based hazard scoring version.

This version is distinct from GLRCC and uses a different simplified scoring structure.

It includes:

- wave height
- wave period
- wave direction
- wind speed
- wind direction

This version also includes a modified treatment of wave angle / direction relative to shore.

---

### `Thumbnail 2/`
Contains a later thumbnail-based scoring version.

This folder includes an updated scoring scale relative to Thumbnail 1 and the corresponding prediction outputs.

---

### `Thumbnail 3/`
Contains another thumbnail-based scoring version used for historical prediction outputs and additional comparison files.

This folder also includes files related to survey-linked hazard analysis.

---

### `Thumbnail 3 rescaled/`
Contains a re-scaled version of the Thumbnail 3 scoring framework.

This version uses updated score values relative to Thumbnail 3.

---

### `Thumbnail 3 rescaled 12h/`
Contains a re-scaled version of Thumbnail 3 that also includes **maximum wave height over the previous 12 hours** as an additional scoring variable.

---

### `Thumbnail 4/`
Contains a later scoring version with a different scoring scale and different hazard classification thresholds than the earlier thumbnail versions.

This folder includes the associated scoring image, scripts, and outputs.
-  - Risk classification:
  - Low: <= 1.5
  - Moderate: 1-5 - 2.5 
  - High: >= 2.5 

---

## Notes

- The `.png` files in each folder define the scoring scale used for that version.
- `GLRCC/` is a separate checklist-based framework and is structurally different from the thumbnail-based scoring versions.
- The thumbnail folders document later scoring systems tested during model development.
- These folders are retained to preserve the different scoring versions used during testing.
- The operational production code is stored in the root repository and is not run directly from these archived folders.
