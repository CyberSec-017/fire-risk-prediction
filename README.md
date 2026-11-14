# 🔥 FIREWATCH AI — Wildfire Risk Prediction & Impact Assessment

**An academic data-mining project for wildfire risk prediction, geospatial monitoring, and environmental impact assessment in Algeria.**

FIREWATCH AI combines satellite-based fire observations, meteorological data, machine learning, and geospatial analysis to identify wildfire-prone areas and assess the potential exposure of nearby populations and infrastructure.

The project integrates data from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) and [NASA POWER](https://power.larc.nasa.gov/), with additional geospatial datasets used by the impact-assessment module.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Data Sources](#data-sources)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Machine Learning Model](#machine-learning-model)
- [Application Modules](#application-modules)
- [Prediction-First Landing Page](#prediction-first-landing-page)
- [Wildfire Impact Assessment](#wildfire-impact-assessment)
- [Model Evaluation Notebook](#model-evaluation-notebook)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Testing and Validation](#testing-and-validation)
- [Current Limitations](#current-limitations)
- [Development Workflow](#development-workflow)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

## Overview

Wildfires represent a major environmental risk, particularly in regions exposed to high temperatures, prolonged dry periods, and changing climatic conditions.

This project aims to develop a data-driven system capable of estimating wildfire occurrence risk using historical fire observations and meteorological variables.

Rather than relying exclusively on historical fire detections, the project constructs a complete spatial-temporal dataset by combining:

- Meteorological observations from NASA POWER.
- Satellite fire detections from NASA FIRMS.
- Weather-derived temporal features.
- Machine learning-based risk estimation.
- Geospatial population and infrastructure information.

The resulting system provides an interactive dashboard for exploring historical fire activity, monitoring geographical patterns, estimating wildfire risk, and examining potential exposure to surrounding populations and infrastructure.

**Study area:** Northern Algeria.

**Temporal coverage:** 2018–2024 for the configured meteorological data collection pipeline.

---

## Objectives

The project is organized around the following objectives:

1. **Data acquisition:** Collect historical meteorological and satellite fire-observation data.
2. **Data preprocessing:** Clean fire detections and align them with a common geographical grid.
3. **Feature engineering:** Extract weather-related indicators and temporal patterns associated with wildfire occurrence.
4. **Predictive modeling:** Train and evaluate a machine learning classifier using chronological validation.
5. **Geospatial monitoring:** Visualize fire activity, weather conditions, and geographical risk distribution.
6. **Impact assessment:** Estimate the exposure of populations and infrastructure to wildfire-prone areas.
7. **Interactive visualization:** Provide an accessible interface for exploring the results.

---

## System Architecture

The project follows a modular architecture separating data processing, model development, application logic, visualization, and impact assessment.

```text
                  EXTERNAL DATA SOURCES
                 ┌─────────────────────┐
                 │                     │
             NASA FIRMS           NASA POWER
             Fire detections      Weather data
                 │                     │
                 └──────────┬──────────┘
                            │
                            ▼
                   DATA PREPROCESSING
                            │
                            ▼
                   DATASET CONSTRUCTION
                            │
                            ▼
                    FEATURE ENGINEERING
                            │
                            ▼
                   MACHINE LEARNING MODEL
                            │
                            ▼
                    PREDICTION ENGINE
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
       STREAMLIT DASHBOARD          IMPACT ASSESSMENT
             │                             │
             ▼                             ▼
       Risk visualization          Population exposure
       Historical analysis         Infrastructure exposure
       Geospatial monitoring       Impact indicators
```

The impact-assessment module is intentionally separated from the predictive model. It consumes geographical and population-related information without modifying the trained model or its feature contract.

---

## Data Sources

### 1. NASA FIRMS — Fire Information for Resource Management System

[Official website](https://firms.modaps.eosdis.nasa.gov/)

NASA FIRMS provides satellite-based active-fire observations.

The project uses fire detection records to identify geographical cells and dates associated with observed fire activity.

Relevant information includes:

- Geographic coordinates.
- Acquisition date and time.
- Fire radiative power (FRP).
- Detection confidence.
- Detection type.

The observations are filtered, spatially aligned, and aggregated to produce daily fire labels.

### 2. NASA POWER — Prediction Of Worldwide Energy Resources

[Official website](https://power.larc.nasa.gov/)

NASA POWER provides meteorological observations on a regular geographical grid.

The project currently uses the following variables:

| Variable | Description |
|---|---|
| T2M | Mean temperature at 2 metres |
| T2M_MAX | Maximum temperature at 2 metres |
| RH2M | Relative humidity at 2 metres |
| WS2M | Wind speed at 2 metres |
| PRECTOTCORR | Corrected precipitation |
| GWETROOT | Root-zone soil wetness |

The geographical study area is configured in `src/config.py`.

### 3. Additional geospatial datasets

The impact-assessment module uses additional geographical information:

| Dataset | Purpose |
|---|---|
| WorldPop | Population distribution |
| OpenStreetMap | Roads, buildings, settlements and facilities |
| ESA WorldCover | Land-cover classification and fuel-related indicators |
| GHSL | Built-up surface comparison |
| OCHA COD-AB | Administrative boundaries |

Some of these datasets are optional and must be downloaded manually. Their availability is documented in the impact manifest.

---

## Data Processing Pipeline

The main data pipeline is implemented in `src/`.

### Step 1 — Meteorological data collection

`src/fetch_power.py`

Downloads daily meteorological observations from NASA POWER.

The study area is divided into smaller geographical tiles to respect the API's regional request limitations.

Downloaded files are cached locally to allow interrupted downloads to resume without retrieving previously collected data.

### Step 2 — Fire detection preprocessing

`src/clean_firms.py`

Processes the manually downloaded FIRMS CSV file.

Main operations:

- Detection quality filtering.
- Removal of irrelevant detection types.
- Spatial alignment with the NASA POWER grid.
- Aggregation of detections by geographical cell and date.
- Identification of persistent heat-source cells.

Output:

`data/interim/fires_daily.parquet`

### Step 3 — Dataset construction

`src/build_panel.py`

Combines meteorological observations and fire labels.

NASA POWER provides the complete geographical and temporal grid, while FIRMS supplies the observed fire labels.

This approach allows the dataset to contain both positive observations (fire detected) and negative observations (no fire detected).

The resulting dataset contains one record per geographical cell and date.

Output:

`data/processed/fire_panel.parquet`

### Step 4 — Feature engineering

The dataset is enriched with historical weather indicators.

Rolling-window features are calculated over 7, 14, and 30 days:

- Cumulative precipitation.
- Mean maximum temperature.
- Minimum relative humidity.
- Consecutive dry days.
- Seasonal indicators using sine and cosine transformations.

All weather-derived features use historical observations rather than future information.

### Step 5 — Model training and evaluation

`src/train_baseline.py`

Trains a HistGradientBoostingClassifier and calibrates its predicted probabilities.

Evaluation uses chronological data splitting rather than random splitting to reduce temporal leakage.

The evaluation includes:

- Precision-Recall AUC.
- Brier score.
- Brier skill score.
- Calibration analysis.
- Risk-level evaluation.
- Comparison against a historical climatology baseline.

---

## Machine Learning Model

### Prediction target

The model estimates whether a geographical cell will experience an observed fire event within the next three days.

The target is constructed using the historical fire-label sequence for each geographical cell.

### Feature set

The current deployed model uses 20 features:

| Feature group | Variables |
|---|---|
| Location | `lat_cell`, `lon_cell` |
| Daily weather | `T2M`, `T2M_MAX`, `RH2M`, `WS2M`, `PRECTOTCORR`, `GWETROOT` |
| 7-day history | `rain_7d`, `tmax_7d`, `rhmin_7d` |
| 14-day history | `rain_14d`, `tmax_14d`, `rhmin_14d` |
| 30-day history | `rain_30d`, `tmax_30d`, `rhmin_30d` |
| Dryness | `dry_days` |
| Seasonality | `doy_sin`, `doy_cos` |

The authoritative feature order and model metadata are stored in:

- `models/model_card.json`
- `models/fire_model.joblib`

### Current model evaluation

The values recorded in the shipped model card (`v0.1.0`) are:

| Metric | Result |
|---|---:|
| PR-AUC | 0.1582 |
| Base fire rate | 2.37% |
| Brier score | 0.02234 |
| Brier skill score | 0.03456 |
| Top-decile fire share | 52.57% |

A fresh chronological re-evaluation of the same artifact on the current panel
(train < 2021, validation = 2021, test ≥ 2022) gives:

| Metric | Result |
|---|---:|
| PR-AUC | 0.2441 |
| Base fire rate (test) | 4.32% |
| ROC-AUC | 0.8812 |
| Brier score | 0.03618 |
| Brier skill score | +0.1253 |
| Expected calibration error (ECE) | 0.00506 |
| Top-decile fire share | 50.50% |
| Climatology PR-AUC (baseline) | 0.2066 |

The two sets of numbers come from different panel snapshots: the model card was
written when the stored panel had a lower base rate. The card is kept unchanged
because `tests/test_impact_integrity.py` pins its SHA256; the reproducible,
up-to-date numbers live in `report/evaluation_summary.json`. Every metric is
regenerated by `notebook.py`.

The top-decile fire share measures the proportion of observed positive events captured within the highest-scored 10% of predictions.

These results should be interpreted in the context of a highly imbalanced classification problem.

**Important:** Model scores are estimates based on historical observations, not guarantees of future fire occurrence.

---

## Application Modules

The interactive application is built using Streamlit.

`app.py` is the **prediction-first landing page**: it opens directly on a
"enter conditions → get a risk estimate" workflow. The analytical modules are
kept as separate pages in `pages/`, reachable from the sidebar and from the
landing page itself.

### Landing page (`app.py`)

Fire-risk prediction with a short, guided input flow:

1. Location (latitude / longitude) and date, snapped to the model grid.
2. A scenario preset (summer heatwave, dry spring, mild autumn, winter wet) or
   manual fire-weather inputs: max temperature, humidity, wind, rainfall,
   root-zone soil wetness and consecutive dry days.
3. Optional advanced overrides for the 7/14/30-day lag features.
4. Optional live fetch of the latest NASA POWER weather for the selected point.

The result panel shows a probability gauge, the risk band, a plain-language
interpretation, a one-feature-at-a-time sensitivity chart explaining the score,
and actionable guidance. Model metrics and data provenance stay folded in an
expander so the prediction remains the focus.

### 1. Overview

Provides an overview of the platform, its data, and the available analytical modules.

### 2. Geospatial Monitoring

Explores the geographical distribution of fire activity and related spatial information.

### 3. Data Analytics

Provides analytical views of historical observations, weather variables, and fire activity.

### 4. Risk Prediction

The full risk cockpit: batch scoring over a date, score distribution, per-cell
inspection and comparison with observed detections.

### 5. About

Provides information about the project and its methodology.

### 6. Impact Assessment

Examines population and infrastructure exposure using the supplementary geospatial datasets.

---

## Prediction-First Landing Page

The landing page is implemented in `app.py` and delegates all model-facing
logic to `dashboard/predictor.py`, a Streamlit-free module that is unit-tested
in `tests/test_predictor.py`.

`dashboard/predictor.py` provides:

| Function | Role |
|---|---|
| `build_feature_row` | Turns a handful of human inputs into the exact 20-feature row |
| `score_conditions` | Scores conditions with the calibrated model |
| `feature_sensitivity` | One-feature-at-a-time explanation of the score |
| `condition_summary` | Human-readable summary of the entered conditions |
| `guidance_for` / `risk_narrative` | Non-alarmist, band-specific guidance |
| `baseline_comparison` | Compares the score with the climatological base rate |

Design choices:

- **Grid snapping is explicit.** The UI shows the grid cell actually used
  (`lat_cell`, `lon_cell`), so the user is never misled about resolution.
- **Every input is traceable.** Lag features are either derived from the
  entered weather or overridden manually — nothing is silently imputed.
- **The explanation is honest.** Sensitivity is a transparent local
  perturbation, not a claimed causal attribution.
- **Uncertainty is stated.** The score is a probability of a fire detection in
  the next 3 days for a coarse grid cell, not a forecast of a specific fire.

---

## Wildfire Impact Assessment

The impact-assessment module is implemented independently under `impact/`.

Its objective is to complement wildfire risk estimation with information about the potential exposure of surrounding populations and infrastructure.

The module processes geographical datasets to construct indicators such as:

- Population within a geographical cell.
- Population within surrounding distance rings.
- Road-network density.
- Building counts and building footprints.
- Critical infrastructure and facilities.
- Settlement distribution.

The module computes an Impact Exposure Index (IEI) using available exposure components.

Population and infrastructure contributions are weighted and normalized according to the components available in the generated dataset.

**Important distinction:** The IEI represents relative exposure within the study area. It is not a probability of fire damage, a prediction of burned area, or a validated loss estimate.

### Data preparation

Additional geospatial inputs must be downloaded manually.

Detailed instructions, expected filenames, checksums, and processing requirements are documented in:

`DATA_DOWNLOAD.md`

The impact module uses separate build dependencies:

```bash
python -m pip install -r requirements-impact.txt
```

Build the derived layers:

```bash
python scripts/build_impact_layers.py
```

Validate the generated artifacts:

```bash
python scripts/validate_impact_layers.py --write-report
```

The validation report is generated at:

`data/impact/validation_report.json`

The generated manifest records input provenance, available components, exclusions, and processing metadata.

---

## Model Evaluation Notebook

`notebook.py` is a runnable evaluation script (not a `.ipynb` file) that
reproduces the published metrics end to end and writes the evidence used in
the project report.

```bash
python notebook.py
```

It performs, in order:

1. Loads the model artifact and the fire panel, and prints the split sizes.
2. Rebuilds the chronological split (train < 2021, validation = 2021, test ≥ 2022).
3. Scores the test period and computes PR-AUC, ROC-AUC, Brier score, Brier
   skill score, ECE, calibration slope/intercept, threshold analysis and the
   top-decile fire share.
4. Compares against a per-cell, per-month climatology baseline.
5. Evaluates the four risk bands and per-year performance.
6. Computes sampled permutation importance for the model features.
7. Prints dataset statistics and writes the figures and summary files.

Outputs:

| Path | Content |
|---|---|
| `report/EVALUATION.md` | Human-readable evaluation report |
| `report/evaluation_summary.json` | Machine-readable metrics, bands, yearly results |
| `report/figures/01_fire_seasonality.png` | Fire seasonality |
| `report/figures/02_precision_recall_capture.png` | Precision-recall and capture curve |
| `report/figures/03_roc_curve.png` | ROC curve |
| `report/figures/04_calibration.png` | Reliability / calibration plot |
| `report/figures/05_risk_bands.png` | Risk-band calibration |
| `report/figures/06_yearly_performance.png` | Yearly metrics |
| `report/figures/07_feature_importance.png` | Permutation importance |
| `report/figures/08_confusion_matrix.png` | Confusion matrix at the 0.5 threshold |
| `report/figures/09_hotspots.png` | Spatial hotspot map |
| `report/figures/10_data_overview.png` | Dataset overview |

The same results are summarised in `report/FIREWATCH_AI_Report.pdf`, generated
by `report/build_report.py`.

---

## Project Structure

```text
fire-risk-prediction/
│
├── app.py                       # prediction-first landing page
├── app_core.py                  # model loading, scoring, risk bands
├── notebook.py                  # reproducible evaluation script
│
├── src/
│   ├── config.py
│   ├── fetch_power.py
│   ├── clean_firms.py
│   ├── build_panel.py
│   ├── train_baseline.py
│   ├── evaluate_spatial.py
│   └── evaluate_rolling.py
│
├── dashboard/
│   ├── config.py
│   ├── components.py
│   ├── charts.py
│   ├── maps.py
│   ├── geo.py
│   ├── data_loader.py
│   ├── model_loader.py
│   └── predictor.py             # Streamlit-free prediction helpers
│
├── pages/
│   ├── 1_📊_Overview.py
│   ├── 2_🗺️_Geospatial_Monitoring.py
│   ├── 3_📈_Data_Analytics.py
│   ├── 4_🎯_Risk_Prediction.py
│   ├── 5_ℹ️_About.py
│   └── 6_Impact_Assessment.py
│
├── impact/
│   ├── config.py
│   ├── data.py
│   ├── exposure.py
│   ├── severity.py
│   ├── methodology.py
│   ├── maps.py
│   ├── charts.py
│   └── report.py
│
├── scripts/
│   ├── build_impact_layers.py
│   └── validate_impact_layers.py
│
├── report/
│   ├── build_report.py
│   ├── FIREWATCH_AI_Report.pdf
│   ├── EVALUATION.md
│   ├── evaluation_summary.json
│   └── figures/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── impact/
│
├── models/
│   ├── fire_model.joblib
│   └── model_card.json
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
├── requirements-impact.txt
├── requirements-report.txt
├── DATA_DOWNLOAD.md
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10 or later (use a version compatible with the pinned dependencies).
- Git.
- pip.
- Internet connection for external data acquisition.

### 1. Clone the repository

```bash
git clone https://github.com/CyberSec-017/fire-risk-prediction.git

cd fire-risk-prediction
```

### 2. Create a virtual environment

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

For geospatial impact-layer construction:

```bash
python -m pip install -r requirements-impact.txt
```

For the PDF report generator:

```bash
python -m pip install -r requirements-report.txt
```

---

## Running the Project

### Launch the dashboard

From the repository root:

```bash
streamlit run app.py
```

Streamlit will provide a local URL for accessing the application. The landing
page opens on the risk predictor; the analytical pages are available from the
sidebar.

### Reproduce the model evaluation

```bash
python notebook.py
```

This rewrites `report/EVALUATION.md`, `report/evaluation_summary.json` and the
figures in `report/figures/`.

### Rebuild the PDF report

```bash
python report/build_report.py
```

### Execute the data pipeline

Download the FIRMS observations manually and place the CSV file at:

`data/raw/firms_raw.csv`

Then execute the pipeline in order:

```bash
python src/fetch_power.py

python src/clean_firms.py

python src/build_panel.py

python src/train_baseline.py
```

Each step generates data consumed by subsequent stages.

NASA POWER downloads are cached, so the collection script can be rerun to fill missing data.

### Run impact assessment preparation

Refer to `DATA_DOWNLOAD.md` for the required external datasets.

Then execute:

```bash
python scripts/build_impact_layers.py

python scripts/validate_impact_layers.py --write-report
```

---

## Testing and Validation

The project includes automated tests to protect important functionality and data integrity.

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The tests include checks for:

- Model artifact integrity.
- Feature-contract consistency.
- Separation between the impact module and the predictive model.
- Data integrity and missing-data handling.
- Geospatial processing assumptions.
- The prediction helpers and the landing-page render.

The impact validation script also produces a separate report describing errors, warnings, and successful checks.

---

## Current Limitations

The project is an ongoing academic and technical development effort. Its current limitations include:

1. **Spatial resolution:** NASA POWER uses a relatively coarse meteorological grid, limiting the ability to capture local terrain and microclimate variations.
2. **Satellite observations:** FIRMS detects active fires, but some events may be missed due to cloud cover, satellite revisit frequency, or detection limitations.
3. **Class imbalance:** Fire observations represent a small proportion of the dataset, making evaluation more challenging.
4. **Geographical scope:** The current implementation focuses on northern Algeria and should not be assumed to generalize to other geographical regions.
5. **Impact data availability:** Some optional geospatial datasets are not yet ingested, so certain exposure indicators remain unavailable.
6. **Impact interpretation:** Exposure scores do not represent validated estimates of actual wildfire damage.
7. **Prediction uncertainty:** The model relies on historical observations and meteorological indicators; its outputs should not be interpreted as official emergency forecasts.
8. **Location dominance:** Permutation importance shows that the grid-cell coordinates carry most of the signal. The model is therefore strongest at ranking *where* fires happen and weaker at explaining a single day from weather alone.
9. **Model-card drift:** The metrics inside `models/model_card.json` were recorded on an earlier panel snapshot and differ from the current reproducible evaluation. See `report/EVALUATION.md`.

---

## Development Workflow

The project uses Git and GitHub for collaborative development.

The `main` branch is intended to contain stable and reviewed code.

Contributors should work on individual branches and submit pull requests rather than pushing directly to `main`.

Recommended branch naming:

```text
feature/<feature-name>
fix/<bug-name>
refactor/<module-name>
docs/<documentation-change>
test/<test-name>
```

Example:

```bash
git switch main

git pull origin main

git switch -c feature/new-risk-analysis

# Implement changes

git add .

git commit -m "feat: add new risk analysis"

git push -u origin feature/new-risk-analysis
```

Submit a pull request targeting `main` for review.

---

## Future Improvements

Potential directions for future development include:

- Incorporating additional environmental variables.
- Improving model calibration and predictive performance.
- Evaluating generalization across different geographical regions.
- Integrating additional land-cover and vegetation datasets.
- Incorporating independent burned-area products.
- Improving geospatial resolution.
- Developing more advanced spatial and temporal validation strategies.
- Expanding automated testing and continuous integration.
- Improving model interpretability and uncertainty analysis.
- Regenerating `models/model_card.json` from the current panel so the stored metrics match `report/` (requires updating the pinned hash in the integrity test).

---

## Acknowledgements

This project relies on publicly available scientific and geospatial datasets.

We acknowledge the organizations and communities responsible for providing these resources:

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)
- [NASA POWER](https://power.larc.nasa.gov/)
- [WorldPop](https://www.worldpop.org/)
- [OpenStreetMap contributors](https://www.openstreetmap.org/)
- [ESA WorldCover](https://esa-worldcover.org/)
- [Copernicus Global Human Settlement Layer](https://human-settlement.emergency.copernicus.eu/)
- [Humanitarian Data Exchange (OCHA)](https://data.humdata.org/)

---

**FIREWATCH AI — Data-driven wildfire risk analysis and geospatial impact assessment.**
