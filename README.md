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
- [Experimental Feature Engineering V2](#experimental-feature-engineering-v2)
- [Application Modules](#application-modules)
- [Prediction-First Landing Page](#prediction-first-landing-page)
- [Next-Generation Interface — FireWatch AI UI V2](#next-generation-interface--firewatch-ai-ui-v2)
- [Batch CSV Inference](#batch-csv-inference)
- [Wildfire Impact Assessment](#wildfire-impact-assessment)
- [Model Evaluation Notebook](#model-evaluation-notebook)
- [Scientific Project Report](#scientific-project-report)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Docker and Reproducibility](#docker-and-reproducibility)
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

The project follows a modular architecture separating data processing, feature engineering, model development, application logic, visualization, and impact assessment.

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
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
    Production Baseline            Experimental Features V2
    (20 core features)             (12 new: VPD, Anomalies, Indices, Neighbors)
            │                               │
            ▼                               ▼
    Calibrated Production Model     Offline Benchmarking & Analysis
    (HistGradientBoostingClassifier) (src/compare_features_v2.py)
            │
            ▼
     PREDICTION ENGINE (app_core.py)
            │
    ┌───────┴───────────────────────┬────────────────────────┐
    ▼                               ▼                        ▼
SINGLE ENTRY POINT (app.py)    ANALYTICAL SECTIONS          IMPACT ASSESSMENT
- Level 1: executive landing   - EDA and data quality       - Population exposure
- Level 2: satellite cockpit   - Statistical validation     - Infrastructure exposure
- Batch CSV inference          - Benchmark and explainability - Impact indicators
```

The impact-assessment module is intentionally separated from the predictive model. It consumes geographical and population-related information without modifying the trained model or its feature contract. Similarly, the experimental Feature Engineering V2 framework and the next-generation UI V2 interface operate alongside the core production pipelines without altering the deployed 20-feature contract or legacy application flows.
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
regenerated by `report/notebook.py`. The landing page's model-card expander displays
both columns side by side, so the drift is visible in the app rather than only
in the documentation.

The top-decile fire share measures the proportion of observed positive events captured within the highest-scored 10% of predictions.

These results should be interpreted in the context of a highly imbalanced classification problem.

**Important:** Model scores are estimates based on historical observations, not guarantees of future fire occurrence.

---

## Experimental Feature Engineering V2

To investigate whether enhanced physical, climatological, and spatial representations can improve wildfire risk estimation, an advanced experimental feature engineering framework is implemented in `src/features_v2.py`.

The system retains the authoritative 20 baseline features of the production contract and introduces **12 new candidate features** grouped into four scientific families.

### Feature Families

```text
                               12 NEW EXPERIMENTAL FEATURES
                                             │
      ┌──────────────────────┬───────────────┴──────────────┬─────────────────────────┐
      ▼                      ▼                              ▼                         ▼
Vapor Pressure Deficit  Temperature Anomalies         Fire-Weather Indices      Neighbor Fire History
(Magnus-Tetens)         (Causal & Climatological)     (Empirical Fire Danger)   (NASA FIRMS Spatial Lag)
- vpd                   - tmax_anomaly_30d            - angstrom_proxy          - neighbor_fires_7d
- vpd_tmax_proxy        - t2m_anomaly_hist            - cbi                     - neighbor_fires_14d
                        - t2m_zscore_hist             - hdw_proxy               - neighbor_fires_30d
                                                      - nesterov_proxy
```

#### A. Vapor Pressure Deficit (VPD)

Vapor Pressure Deficit (VPD) quantifies the drying power of the atmosphere by measuring the difference between saturation vapor pressure $e_s(T)$ and actual vapor pressure $e_a(T, \text{RH})$. Formulations use the Magnus-Tetens approximation (Murray 1967; Allen et al. 1998, FAO-56):

- `vpd`: Daily mean Vapor Pressure Deficit (in kPa) derived from daily mean temperature (`T2M`) and relative humidity (`RH2M`).
- `vpd_tmax_proxy`: Estimated daytime peak VPD combining daily maximum temperature (`T2M_MAX`) with daily mean relative humidity (`RH2M`). **Note:** This variable serves as an empirical proxy for peak daytime atmospheric dryness; it is not an instantaneous physical measurement captured at the exact moment of daily maximum temperature.

#### B. Temperature Anomalies

These features capture both acute short-term heat stress and long-term departure from local climatological baselines:

- `tmax_anomaly_30d`: Daily maximum temperature (`T2M_MAX`) minus the antecedent 30-day rolling mean of `T2M_MAX` (computed with a 1-day shift to strictly exclude day $t$). It identifies acute heatwave spikes relative to recent local weather $[t-30, t-1]$.
- `t2m_anomaly_hist`: Difference between daily mean temperature (`T2M`) and the historical monthly mean for that grid cell and calendar month, computed strictly across prior years ($Y' < Y$). Evaluates to `NaN` when no prior reference years exist (e.g., year 2018).
- `t2m_zscore_hist`: Standardized temperature anomaly relative to historical years ($Y' < Y$). Requires at least two prior reference years ($n \ge 2$) and strictly positive standard deviation ($\sigma > 0$); evaluates to `NaN` otherwise.

#### C. Fire-Weather Indices

Four empirical fire danger indicators adapted to daily gridded reanalysis:

- `angstrom_proxy`: Daily adaptation of the Swedish Ångström fire risk index:
  $$I = \frac{\text{RH2M}}{20.0} + \frac{27.0 - \text{T2M}}{10.0}$$
  Lower values denote higher danger ($I < 2.0$ signifies very high fire hazard). **Note:** Adapted to daily aggregated means rather than standard 13:00 local time readings.
- `cbi`: Chandler Burning Index metric daily adaptation based on Chandler et al. (1983):
  $$\text{CBI} = \frac{(104.5 - 1.373 \cdot \text{RH2M} + 0.54 \cdot \text{T2M}) \cdot 124 \cdot 10^{-0.0142 \cdot \text{RH2M}}}{60}$$
  Floored at $0.0$ to ensure non-negative fire potential.
- `hdw_proxy`: Surface Hot-Dry-Windy proxy calculated as $\text{VPD} \times \text{WS2M}$ (in $\text{kPa} \cdot \text{m/s}$). **Note:** Evaluated using 2-meter surface wind observations, whereas the original atmospheric HDW index (Srock et al., 2018) integrates wind and moisture across the 500-meter planetary boundary layer.
- `nesterov_proxy`: Cumulative Nesterov drought-combustion index (Nesterov, 1949). Daily increments $\Delta NI = \max(\text{T2M} \cdot (\text{T2M} - T_{\text{dew}}), 0)$ (for $\text{T2M} > 0$) accumulate sequentially and reset to $0.0$ on days with significant precipitation ($\text{PRECTOTCORR} \ge 3.0\text{ mm}$). Prior to the first valid $\ge 3.0\text{ mm}$ rain reset in a cell's sequence, the value evaluates to `NaN`.

#### D. Spatial-Temporal Fire History

Wildfire propagation and regional fire activity exhibit strong spatial autocorrelation. These variables capture satellite-confirmed active fire detections in the neighborhood:

- `neighbor_fires_7d`: Cumulative NASA FIRMS active fire counts across the 8 spatial Moore neighbors of the grid cell over the preceding 7 days $[t-7, t-1]$.
- `neighbor_fires_14d`: Cumulative FIRMS active fire counts across the 8 neighbors over the preceding 14 days $[t-14, t-1]$.
- `neighbor_fires_30d`: Cumulative FIRMS active fire counts across the 8 neighbors over the preceding 30 days $[t-30, t-1]$.

### Methodological Guarantees

The implementation in `src/features_v2.py` adheres to strict mathematical and causal validation guarantees:

- **Rolling-window leakage prevention:** Rolling windows (such as `tmax_anomaly_30d` and neighbor fire history counts) apply `shift(1)` to strictly exclude the prediction day $t$, ensuring features consume only antecedent observations $[t-k, t-1]$.
- **Historical climatology leakage prevention:** Historical monthly means and standard deviations (`t2m_anomaly_hist`, `t2m_zscore_hist`) are calculated exclusively from strictly preceding years ($Y' < Y$) for the corresponding cell and calendar month. Prior-year climatological baselines do not use `shift(1)`; they enforce temporal separation by filtering out the current year and all future years.
- **Explicit missingness:** Incomplete historical windows (e.g., initial observation year 2018 where no prior years exist, or periods before precipitation resets) produce explicit `NaN` values, preserving genuine missingness for tree models.
- **Target independence:** The feature engineering routines never inspect the target variable (`target` or fire labels), guaranteeing total causal purity.
- **Data immutability:** All transformation functions operate on defensive dataframe copies, preserving row order, original indices, and exact row counts.

---

### Comparative Experimental Study

An extensive comparative benchmarking experiment was executed via `src/compare_features_v2.py` using `HistGradientBoostingClassifier` with tuned hyperparameters (`learning_rate=0.03`, `max_depth=8`, `max_iter=150`, `min_samples_leaf=100`, `l2_regularization=0.1`).

Six distinct feature configurations were evaluated:

| Configuration | Features | Description |
|---|---:|---|
| A — Baseline | 20 | Original production feature set |
| B — Baseline + VPD | 22 | Baseline plus two VPD features |
| C — Baseline + Anomalies | 23 | Baseline plus three temperature anomaly features |
| D — Baseline + Indices | 24 | Baseline plus four fire-weather indicators |
| E — Baseline + Neighbors | 23 | Baseline plus three neighboring-fire history features |
| F — All Families | 32 | Baseline plus all 12 experimental features |

#### Experimental Protocol

The evaluation adheres to a rigorous chronological validation protocol:

| Partition | Period | Rows |
|---|---|---:|
| Training | 2018-03-02 to 2021-12-28 | 292,182 |
| Validation | 2022-01-01 to 2022-12-28 | 75,658 |
| Test | 2023–2024 | Excluded from this experiment |

Key protocol rules:

- **Strictly identical partitions:** Training and validation splits are identical in size and temporal bounds across all six configurations.
- **Chronological purge gap:** A 3-day purge gap is enforced between time partitions to reflect the 3-day prediction horizon and eliminate boundary leakage.
- **Continuous pre-test panel:** In `src/compare_features_v2.py`, `load_and_prepare_continuous_panel()` loads the panel dataset and filters out the test period ($date \le 2022\text{-}12\text{-}28$) prior to feature computation. Feature transformations are computed continuously across this pre-test panel to preserve rolling-lag continuity across the training-validation purge boundary.
- **Out-of-Fold threshold selection:** Decision thresholds are selected exclusively on the Training partition using 5-fold chronological expanding-window splits with a 3-day purge gap to maximize F1-score.
- **Frozen threshold evaluation:** Selected thresholds are frozen and evaluated out-of-sample on the 2022 Validation partition.
- **Test partition exclusion:** Observations from 2023–2024 are strictly excluded from experimental feature construction, model training, Out-of-Fold threshold selection, and validation evaluation.
- **Evaluation metrics:** Precision-Recall AUC (PR-AUC, primary metric for imbalanced fire classification), ROC-AUC, Brier Score, Precision, Recall, and F1-score.

#### Experimental Results

| Configuration | PR-AUC | ROC-AUC | Brier Score | OOF Threshold | F1 |
|---|---:|---:|---:|---:|---:|
| A — Baseline | 0.3103 | 0.8945 | 0.0396 | 0.2094 | 0.3446 |
| B — VPD | 0.3114 | 0.8944 | 0.0395 | 0.1929 | 0.3486 |
| C — Anomalies | 0.3118 | 0.8929 | 0.0393 | 0.2244 | 0.3474 |
| D — Indices | 0.3090 | 0.8946 | 0.0396 | 0.2174 | 0.3492 |
| E — Neighbors | 0.3146 | 0.8934 | 0.0388 | 0.1984 | 0.3403 |
| F — All Families | **0.3159** | 0.8920 | **0.0386** | 0.2057 | 0.3436 |

#### Scientific Analysis & Production Safeguards

1. **Ranking & calibration gains:** Configuration F (incorporating all 12 experimental features) achieves the highest PR-AUC (**0.3159** vs. 0.3103 for the baseline), representing a **+1.81% relative improvement**. It also achieves the best probability calibration with the lowest Brier Score (**0.0386** vs. 0.0396).
2. **Metric trade-offs:** Configuration D achieves the highest F1-score (**0.3492** vs. 0.3446), while Configuration A retains a slightly higher ROC-AUC (0.8945 vs. 0.8920). These nuances illustrate that individual evaluation metrics do not all improve simultaneously.
3. **Mandatory scientific caution:** These findings represent **offline validation results** on the 2022 partition. They are not final test results, nor do they represent an immediate change to the deployed model.
4. **Production stability:** The deployed production model strictly preserves its 20-feature contract (`models/model_card.json`) until an official validation, integrity verification, and deployment approval cycle is completed.

**Experiment deliverables:**
- Code implementation: `src/features_v2.py`
- Evaluation pipeline: `src/compare_features_v2.py`
- Comparative benchmark metrics: `results/feature_engineering_v2_comparison.csv`
- Experiment metadata and parameters: `results/feature_engineering_v2_metadata.json`

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

Four supporting features keep the page useful beyond a single score:

- **Scenario comparison.** A collapsed table and bar chart score all four
  weather presets on the selected grid cell and date, showing the model's
  response surface next to the historical base rate.
- **Stable results.** The result panel renders from a snapshot taken when
  *Predict* is pressed. Editing an input afterwards raises a warning naming the
  changed drivers instead of silently changing the displayed probability.
- **Shareable links.** Location, date and scenario are written to the URL as
  `?lat=&lon=&date=&scenario=&run=1`; a link with `run=1` re-scores on arrival.
- **JSON export.** The same panel offers a downloadable record of the model
  version, grid cell, inputs, score, band and drivers for reports or archiving.

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
| `compare_presets` | Scores all four presets on one grid cell and date |
| `changed_features` / `feature_label` | Detects edited inputs against the snapshot |
| `share_query` / `share_url` / `read_share_params` | Builds and parses shareable URLs |
| `prediction_payload` | JSON-serialisable record of one prediction |

Design choices:

- **Grid snapping is explicit.** The UI shows the grid cell actually used
  (`lat_cell`, `lon_cell`), so the user is never misled about resolution.
- **Every input is traceable.** Lag features are either derived from the
  entered weather or overridden manually — nothing is silently imputed.
- **The explanation is honest.** Sensitivity is a transparent local
  perturbation, not a claimed causal attribution.
- **Uncertainty is stated.** The score is a probability of a fire detection in
  the next 3 days for a coarse grid cell, not a forecast of a specific fire.
- **Displayed results do not drift.** A prediction is immutable until the user
  asks for a new one, and edited inputs are flagged explicitly.
- **State is linkable.** A prediction can be reproduced from its URL, which
  makes results easy to discuss, bookmark or cite.

### Sharing a prediction

A prediction can be reopened from a URL:

```text
http://localhost:8501/?lat=36.5&lon=3.125&date=2023-08-15&scenario=Summer+heatwave&run=1
```

| Parameter | Meaning |
|---|---|
| `lat`, `lon` | Coordinates in the study area; clamped to the valid range |
| `date` | ISO date used for the seasonal features |
| `scenario` | One of the four preset names (ignored if unknown) |
| `run` | `1` scores once on arrival; omit it to open the form ready to edit |

Only the location, date and scenario are encoded, because the manual weather
inputs stay fully editable in the form.

### Accessibility

The shared design system (`dashboard/components.py`) provides visible
`:focus-visible` rings on light and dark surfaces, a screen-reader-only helper,
readable disabled controls, callouts announced through `role="alert"` /
`role="status"`, and a `prefers-reduced-motion` fallback.

---

## Next-Generation Interface — FireWatch AI UI V2

The platform uses **FireWatch AI UI V2**, an analytical interface designed for operational monitoring, interactive geospatial exploration, and decision support. The original prediction-first landing experience is preserved as Level 1 of this interface.

UI V2 is implemented under `ui_v2/` with its single master entry point in `app.py`. All analytical sections share the same model bundle and navigation shell.

```text
                               FIREWATCH AI UI V2 ARCHITECTURE
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
            LEVEL 1: LANDING PAGE                           LEVEL 2: DASHBOARD
           "Understand. Monitor. Predict."           (Interactive Geographical Cockpit)
           ui_v2/landing.py                          ui_v2/dashboard_view.py
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
          COMPONENTS & STATE                                ANALYTICAL VIEWS
          - Satellite map (satellite_map.py)                - Overview (overview.py)
          - 209-cell context (point_context.py)             - Historical (historical.py)
          - Prediction modal (prediction_modal.py)          - Weather (weather.py)
          - Geoaether theme (styles/theme.css)              - Risk Map (risk_map.py)
          - Master navigation (navbar.py & sidebar.py)      - Predictions Cockpit (fire_predictions.py)
                                                            - About & Attributions (about.py)
```

### Two-Level Architecture

UI V2 organizes the user journey into two distinct operational levels:

#### Level 1 — Professional Welcome & Landing Page

- **File:** `ui_v2/landing.py`
- **Core concept:** *"Understand. Monitor. Predict."*
- Serves as the executive gateway to the platform, articulating the platform's mission, scientific foundations, and active coverage domain.
- Provides high-level feature highlights and a direct call-to-action button to enter the interactive dashboard.

#### Level 2 — Interactive Geographical Dashboard

- **File:** `ui_v2/dashboard_view.py`
- An interactive operational environment centered on Northern Algeria:
  - **209 active geographic grid cells:** Defined and indexed via `ui_v2/components/point_context.py`. Filters out non-computational maritime areas in the Mediterranean and deep unpopulated Sahara cells to focus strictly on fire-vulnerable land mass.
  - **Dual-basemap interactive map:** Powered by Folium with seamless switching between Esri World Imagery (high-resolution satellite) and OpenStreetMap (cartographic/street basemap). Bottom technical controls (attribution/scale clutter) are cleanly suppressed for interface elegance, while full legal attributions are preserved in the About page.
  - **Synchronized location context:** Clicking on any grid cell or entering coordinates instantly snaps to the model grid and updates local contextual metadata (Wilaya name, elevation, historical fire statistics, and risk band).
  - **Interactive fire prediction dialog:** A dedicated modal dialog (`ui_v2/components/prediction_modal.py`) supporting both manual feature simulation and live meteorological fetching from NASA POWER.
  - **Immediate history synchronization:** Predictions executed inside the modal are immediately written to `st.session_state["prediction_history"]` and rendered instantly in the prediction log and map without requiring manual page reloads.

#### Dedicated Analytical Pages

The cockpit provides six specialized analytical views accessible via the horizontal navbar (`ui_v2/navbar.py`) and dark enterprise sidebar (`ui_v2/sidebar.py`):

1. **Executive Overview (`ui_v2/pages/overview.py`):** Macro KPIs, seasonal fire distribution, and overall risk summary.
2. **Historical Detections (`ui_v2/pages/historical.py`):** Multi-year NASA FIRMS fire detections, temporal trends, and fire radiative power analysis.
3. **Meteorological Monitoring (`ui_v2/pages/weather.py`):** Weather variable distributions, drought tracking, and climate anomalies.
4. **Spatial Risk Distribution (`ui_v2/pages/risk_map.py`):** Gridded risk heatmaps, regional exposure indices, and territorial fire risk mapping.
5. **Fire Predictions Cockpit (`ui_v2/pages/fire_predictions.py`):** Standalone satellite fire risk prediction cockpit for point inference and scenario comparison.
6. **Methodology & About (`ui_v2/pages/about.py`):** Comprehensive model card transparency, mathematical explanations, and legal cartographic attributions.

The same navigation shell also exposes three cross-cutting sections documented below: interactive EDA and data quality (`ui_v2/pages/eda.py`), statistical validation and benchmark (`ui_v2/pages/statistics.py`), and CSV batch inference (`ui_v2/pages/batch.py`).

### Technical Architecture & State Management

- **Master Router (`app.py`):** Routes between the landing page, satellite cockpit, EDA, statistical validation and batch inference based on `st.session_state["app_level"]`.
- **Sub-Page Navigation:** Within the cockpit, pages are activated via `st.session_state["current_page"]` without hard page reloads.
- **Design System (`ui_v2/styles/theme.css`):** Custom *Geoaether* dark aesthetic with high-contrast typography, refined translucent cards, accessible interactive states, and consistent risk-band color accents.
- **Modular Components (`ui_v2/components/`):**
  - `point_context.py`: 209-cell spatial polygon indexing, coordinate snapping, geographic bounding boxes, administrative Wilaya lookup, and pre-test data isolation.
  - `prediction_modal.py`: Prediction dialog with input validation, scenario overrides, real-time model inference, and instant history synchronization.
  - `satellite_map.py`: Folium satellite/street map component with responsive click capture and selected marker positioning.

### Single Application Entry Point

UI V2 is non-destructive: it preserves the original prediction-first landing experience as Level 1 while sharing the deployed model bundle (`app_core.py`) and preprocessed panel datasets with every analytical section. There is no second Streamlit entry point.

```bash
python -m streamlit run app.py
```

The app is served at `http://localhost:8501` by default.

---

## Batch CSV Inference

The **📄 Inférence CSV** section scores many cell-day records in one pass using the same production model bundle as the map. The pure validation and export logic lives in `src/batch_inference.py`; `dashboard/batch_view.py` only renders the Streamlit workflow.

### Input contract

- One row describes one geographical cell and one day through the **20 features** in `models/model_card.json`. A header-only template is downloadable from the UI.
- Column order in the CSV is flexible, but names must match exactly. Extra columns are preserved in the export and ignored by the model.
- Every required value must be numeric, finite and inside a documented physical range. Coordinates must also lie on the model grid (latitude step `0.5°`, longitude step `0.625°`).
- Missing, blank, non-numeric, out-of-range or off-grid values reject that row. **No value is imputed** and no DEMO fallback is used for a batch.

### Output contract

The downloaded UTF-8 CSV (with BOM for Excel) contains:

- all original input columns,
- `row_number`, `validation_status`, `predicted_probability`, `risk_category`,
- `horizon_days` fixed at `3`, and
- explicit `rejection_reasons` for every rejected row.

Rejected rows receive neither a probability nor a risk category, so they can be corrected and re-uploaded. Risk categories come from the model-card thresholds `[0, 0.005, 0.02, 0.06, 1.0]` labeled `LOW`, `MODERATE`, `HIGH` and `VERY HIGH`.

### Explainability in the batch view

The explainability tab documents why SHAP is not a hard dependency: `TreeExplainer` does not support the deployed isotonic-calibrated wrapper, and applying it to the inner tree would explain pre-calibration margins rather than the displayed probability. The view instead offers a local one-feature sensitivity for a selected valid row and the existing global permutation-importance figure. Both are explicitly labeled **non-causal**; they describe model behavior, not meteorological causes of a fire.

**Important:** a predicted risk is a model estimate based on historical detections and weather. It is not an observed fire, a confirmed wildfire occurrence, a burned-area measurement, or an official alert.

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
python -m pip install -r requirements/requirements-impact.txt
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

`report/notebook.py` is a runnable evaluation script (not a `.ipynb` file) that
reproduces the published metrics end to end and writes the evidence used in
the project report.

```bash
python report/notebook.py
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

The compact summary is `report/FIREWATCH_AI_Report.pdf`, generated by
`report/build_report.py`.

## Scientific Project Report

The expanded scientific report is `docs/Rapport_Projet.pdf`. It is generated by
`report/build_rapport_projet.py` from the existing evaluation JSON, statistical
tests, benchmark artifacts and figures. The builder writes only that path and
asserts that the result has 15–30 pages; it does not overwrite the compact report.

```bash
python -m pip install -r requirements/requirements-report.txt
python report/build_rapport_projet.py
```

The report covers the problem statement, data sources, preprocessing, EDA,
statistical tests, model benchmark, evaluation protocol, production model,
results, architecture, explainability decision, limitations, reproducibility
and conclusion. It contains no invented results: every number and figure comes
from artifacts already in the repository.

---

## Project Structure

```text
fire-risk-prediction/
│
├── app.py                       # single Streamlit master entry point
├── app_core.py                  # model loading, scoring, risk bands
├── Dockerfile                   # reproducible Streamlit runtime image
├── .dockerignore
│
├── src/
│   ├── config.py
│   ├── fetch_power.py
│   ├── clean_firms.py
│   ├── build_panel.py
│   ├── train_baseline.py
│   ├── compare_classifiers.py   # multi-model cross-validation benchmark
│   ├── tune_classifiers.py      # hyperparameter tuning pipeline
│   ├── evaluate_spatial.py
│   ├── evaluate_rolling.py
│   ├── features_v2.py           # experimental feature engineering V2
│   ├── compare_features_v2.py   # comparative feature evaluation pipeline
│   └── batch_inference.py       # CSV validation, production scoring, export
│
├── ui_v2/                       # next-generation UI V2 application
│   ├── __init__.py
│   ├── landing.py               # Level 1: executive welcome & landing page
│   ├── dashboard_view.py        # Level 2: interactive geographical dashboard
│   ├── navbar.py                # horizontal navigation bar
│   ├── sidebar.py               # enterprise dark sidebar
│   ├── components/
│   │   ├── __init__.py
│   │   ├── point_context.py     # 209-cell spatial domain & inference context
│   │   ├── prediction_modal.py  # interactive prediction modal & history sync
│   │   └── satellite_map.py     # dual-basemap satellite & street map
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── overview.py          # macro KPIs & seasonal summaries
│   │   ├── historical.py        # FIRMS historical fire detections
│   │   ├── weather.py           # meteorological distributions
│   │   ├── risk_map.py          # spatial risk distribution heatmaps
│   │   ├── fire_predictions.py  # standalone satellite prediction cockpit
│   │   ├── eda.py               # interactive data quality and EDA
│   │   ├── statistics.py        # statistical tests and benchmark
│   │   ├── batch.py             # batch CSV inference page
│   │   └── about.py             # model card & cartographic attributions
│   └── styles/
│       └── theme.css            # Geoaether dark theme stylesheet
│
├── dashboard/
│   ├── config.py
│   ├── components.py
│   ├── charts.py
│   ├── maps.py
│   ├── geo.py
│   ├── data_loader.py
│   ├── model_loader.py
│   ├── predictor.py             # Streamlit-free prediction helpers
│   └── batch_view.py            # batch CSV workflow and explanations
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
├── results/                     # experimental benchmark deliverables
│   ├── feature_engineering_v2_comparison.csv
│   ├── feature_engineering_v2_metadata.json
│   ├── classifier_comparison.csv
│   ├── tuned_classifier_comparison.csv
│   ├── best_hyperparameters.json
│   ├── statistical_validation.json
│   ├── benchmark_protocol.json
│   └── model_benchmark.csv
│
├── scripts/
│   ├── build_impact_layers.py
│   └── validate_impact_layers.py
│
├── report/
│   ├── notebook.py
│   ├── build_report.py
│   ├── build_rapport_projet.py
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
├── docs/
│   └── Rapport_Projet.pdf       # 15–30 page scientific report
│
├── notebooks/
│
├── tests/
│   ├── test_app_core.py
│   ├── test_features_v2.py
│   ├── test_compare_features_v2.py
│   ├── test_ui_v2_app.py
│   ├── test_ui_v2_map_sync.py
│   ├── test_ui_v2_history_sync.py
│   ├── test_ui_v2_point_context.py
│   ├── test_predictor.py
│   ├── test_dashboard_risk.py
│   ├── test_components_a11y.py
│   ├── test_compare_classifiers.py
│   ├── test_tune_classifiers.py
│   ├── test_batch_inference.py
│   ├── test_batch_view.py
│   ├── test_project_docs.py
│   └── test_impact_*.py
│
├── requirements/
│   ├── requirements.txt
│   ├── requirements-impact.txt
│   └── requirements-report.txt
├── pyproject.toml                 # project metadata + pytest defaults
├── DATA_DOWNLOAD.md
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11 (the Docker image and pinned dependencies are validated for this version).
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

python -m pip install -r requirements/requirements.txt
```

For geospatial impact-layer construction:

```bash
python -m pip install -r requirements/requirements-impact.txt
```

For the PDF report generator:

```bash
python -m pip install -r requirements/requirements-report.txt
```

---

## Running the Project

### Launch the Streamlit application

There is one entry point for both the Level 1 landing experience and the Level 2 satellite cockpit:

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501`. Use the landing-page calls-to-action or the cockpit navigation to reach EDA, statistical validation, batch CSV inference and the analytical cockpit views. `STREAMLIT_SERVER_PORT` and `STREAMLIT_SERVER_ADDRESS` can override the default binding.

### Run batch CSV inference

1. Open the app and choose **📄 Inférence CSV**.
2. Download the header-only template to see the 20 required feature names.
3. Fill one row per cell-day, save as CSV and upload it.
4. Review accepted and rejected rows, then download `firewatch_batch_predictions.csv`.

Rejected rows include explicit reasons and are never scored or imputed.

### Run Feature Engineering V2 comparison

Reproduce the 6-configuration experimental feature benchmark:

```bash
python src/compare_features_v2.py
```

This evaluates Configurations A through F across the 2018–2021 training and 2022 validation partitions, updating `results/feature_engineering_v2_comparison.csv` and `results/feature_engineering_v2_metadata.json`.

### Reproduce the model evaluation

```bash
python report/notebook.py
```

This rewrites `report/EVALUATION.md`, `report/evaluation_summary.json` and the
figures in `report/figures/`.

### Rebuild the PDF reports

```bash
python report/build_report.py             # compact report/FIREWATCH_AI_Report.pdf
python report/build_rapport_projet.py     # 15–30 page docs/Rapport_Projet.pdf
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

## Docker and Reproducibility

`Dockerfile` builds a Streamlit image from `python:3.11-slim`, installs `libgomp1`
for scikit-learn and uses the pinned `requirements/requirements.txt`. The
`.dockerignore` keeps caches, Git metadata, `data/impact/raw/` and
`data/impact_test/` out of the image context.

Required local artifacts (already tracked or generated by the commands above):

- `models/fire_model.joblib` and `models/model_card.json`,
- `data/processed/fire_panel.parquet` and required interim layers,
- `report/figures/`, `report/evaluation_summary.json`,
- `results/statistical_validation.json`, `results/benchmark_protocol.json` and
  `results/model_benchmark.csv`.

No API key or secret is needed at runtime. Optional environment variables are
`STREAMLIT_SERVER_ADDRESS` and `STREAMLIT_SERVER_PORT` (default `0.0.0.0` and
`8501` in the image); Python buffering is set through `PYTHONUNBUFFERED` and
`PYTHONDONTWRITEBYTECODE`.

```bash
docker build -t firewatch-ai .
docker run --rm -p 8501:8501 firewatch-ai
```

Open `http://localhost:8501`. To change the port, pass
`-e STREAMLIT_SERVER_PORT=8502 -p 8502:8502`. The Docker CLI was not available
in the development environment used for this phase, so the image build is
documented but was **not verified locally**.

---

## Testing and Validation

The project includes an extensive automated test suite protecting functionality, mathematical correctness, data integrity, and interface behavior.

Run the entire test suite from the repository root:

```bash
python -m pytest -q
```

`pyproject.toml` points a bare `pytest` at `tests/`. The validated environment
reports **443 passed, 1 warning** with `python -m pytest -q`. The only
warning is a pre-existing pandas `dayofyear` deprecation in `src/build_panel.py`.

- **Batch inference (10 tests):** CSV schema validation, missing/invalid/out-of-range rejection, production scoring, export contents, AppTest upload/download and navigation.
- **Documentation and reproducibility (5 tests):** PDF page count, Docker contents, current README paths, required artifacts and pinned production hashes.
- **Core & UI V2 suite:** Core model routines, probability calibration, Feature Engineering V2, UI V2 workflows, classifier cross-validation, dashboard components and accessibility checks.
- **Impact suite:** Geospatial boundary integrity and layer validation; `tests/test_impact_build_script.py` additionally needs the optional impact dependencies.

On a clean environment, install the optional geospatial dependencies before
running the complete suite:

```bash
python -m pip install -r requirements/requirements-impact.txt
python -m pytest -q
```

Coverage areas across the test suite include:

- **Core & Model Integrity:** Artifact checksums, 20-feature contract validation, model loading, and probability calibration (`test_app_core.py`).
- **Feature Engineering V2 (`test_features_v2.py`, `test_compare_features_v2.py` — 30 tests):** Causal purity, strict non-leakage verification, explicit NaN propagation, historical climatology bounds, and reproducible benchmark pipelines.
- **Next-Generation UI V2 (`test_ui_v2_*.py` — 41 tests):** Two-level navigation lifecycle (`test_ui_v2_app.py`), Folium click extraction and basemap persistence (`test_ui_v2_map_sync.py`), prediction dialog session history synchronization (`test_ui_v2_history_sync.py`), and 209-cell spatial indexing and context resolution (`test_ui_v2_point_context.py`).
- **Legacy Prediction Helpers & UI:** Prediction helper contracts (`test_predictor.py`), sensitivity perturbation, risk-band assignment, and AppTest landing-page integration.
- **Classifier Benchmarking & Hyperparameter Tuning:** Cross-validation splitting with purge gaps, subsampling controls, and fault-tolerant serialization (`test_compare_classifiers.py`, `test_tune_classifiers.py`).
- **Impact Assessment:** Geospatial boundary integrity, layer building, population exposure weighting, and manifest validation (`test_impact_*.py`).
- **Accessibility & Design System:** Visible `:focus-visible` styling, screen-reader status announcements, and `prefers-reduced-motion` compliance (`test_components_a11y.py`).
- **Batch Inference & Documentation:** Strict CSV validation, transparent exports, non-causal explanation UI, Docker/README paths and PDF page-count checks (`test_batch_*.py`, `test_project_docs.py`).

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
10. **Risk versus occurrence:** A predicted risk is a model estimate, not an observed or confirmed wildfire, a burned-area measurement, or an official alert.
11. **Batch inputs:** CSV rows with missing, invalid or off-grid values are rejected rather than imputed; the export horizon is fixed at `H = 3` days.
12. **Explainability:** SHAP is not a hard dependency because the deployed calibrated wrapper would make naive tree attributions misleading. Sensitivity and permutation-importance views are non-causal and should not be read as meteorological causes.
13. **Container verification:** The Dockerfile is provided and documented, but the image build could not be executed in the development environment because Docker was unavailable.

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
- Validating and potentially deploying high-performing candidate features from Feature Engineering V2 (e.g., Configuration F with VPD, temperature anomalies, and neighbor fire history) after multi-year test evaluation.
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
