# Fire Risk Prediction from NASA FIRMS and NASA POWER

Academic Data Mining project that aims to analyze and predict wildfire risk using satellite fire detections and meteorological data.

The project combines:
- **NASA FIRMS:** Satellite-based fire detection data.
- **NASA POWER:** Daily meteorological data.

The goal is to build a dataset linking weather conditions to fire occurrences and use it to train and evaluate machine learning models.

## Project Overview

NASA POWER provides daily weather information for geographic grid cells, while NASA FIRMS provides fire detections at specific locations and dates.

Both datasets are combined using geographic cells and dates to create a dataset where each row represents one cell on one day.

The target variable is `burned`:
- `1`: At least one fire detection was recorded.
- `0`: No fire detection was recorded.

Weather features are then used for analysis and prediction.

## Folder Structure

```text
fire-risk-prediction/
├── data/
│   ├── raw/
│   │   ├── firms_raw.csv
│   │   └── power_cache/
│   ├── interim/
│   │   └── fires_daily.parquet
│   └── processed/
│       └── fire_panel.parquet
├── src/
│   ├── config.py
│   ├── fetch_power.py
│   ├── clean_firms.py
│   ├── build_panel.py
│   └── train_baseline.py
├── notebooks/
│   └── exploration.ipynb
├── models/
│   ├── fire_model.joblib
│   └── model_card.json
├── report/
│   └── figures/
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and install the required libraries:

```bash
git clone https://github.com/CyberSec-017/fire-risk-prediction.git
cd fire-risk-prediction

pip install -r requirements.txt
```

## How to Run the Project

### 1. Collect FIRMS Data

Download fire detection data from:

[NASA FIRMS Download](https://firms.modaps.eosdis.nasa.gov/download/)

Select the study area, date range and VIIRS S-NPP sensor. Place the downloaded CSV files in `data/raw/`.

Merge the collected files into:

`data/raw/firms_raw.csv`

### 2. Configure the Project

Edit `src/config.py` to define:
- Geographic boundaries.
- Start and end years.
- NASA POWER meteorological variables.
- Data storage paths.

### 3. Run the Data Pipeline

Execute the scripts from the project root in the following order:

```bash
python src/fetch_power.py
python src/clean_firms.py
python src/build_panel.py
```

The scripts download weather data, clean fire detections and combine both datasets.

### 4. Analysis and Modeling

After building the final dataset:
- Perform Exploratory Data Analysis (EDA).
- Analyze relationships between weather conditions and fire occurrence.
- Train and compare machine learning models.
- Evaluate model performance.
- Develop a Streamlit application.

## Data Sources and Collection

### 1. NASA FIRMS — Fire Detection Data

NASA FIRMS (Fire Information for Resource Management System) provides satellite-based fire detection information.

- Website: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)
- Satellite product: Suomi-NPP VIIRS Collection 2.
- Format: CSV.
- Study period: 2024–2026.
- Raw records collected: Approximately 86,884.

Important attributes include:
- Latitude and longitude.
- Acquisition date.
- Detection type.
- Confidence level.
- Fire Radiative Power (FRP).

Raw data are stored in `data/raw/`.

### 2. NASA POWER — Meteorological Data

NASA POWER (Prediction of Worldwide Energy Resources) provides daily meteorological information.

- Website: [NASA POWER](https://power.larc.nasa.gov/)
- Data frequency: Daily.
- Format: CSV.
- Study period: 2024–2026.
- Number of variables: 7.
- Geographic tiles: 6.
- Expected files: 126.

The project uses the following variables:

| Variable | Description |
|---|---|
| T2M | Mean air temperature at 2 meters |
| T2M_MAX | Maximum air temperature at 2 meters |
| RH2M | Relative humidity at 2 meters |
| WS2M | Wind speed at 2 meters |
| PRECTOTCORR | Corrected precipitation |
| ALLSKY_SFC_SW_DWN | Surface solar radiation |
| GWETROOT | Root zone soil wetness |

The files are stored in `data/raw/power_cache/`.

**Note:** NASA POWER data for 2026 are partial because the year is still in progress and recent observations may not yet be available.

### 3. Study Area

The project focuses on northern Algeria and the surrounding region.

| Parameter | Value |
|---|---:|
| Minimum latitude | 32° |
| Maximum latitude | 37.5° |
| Minimum longitude | -2° |
| Maximum longitude | 9° |

The region is divided into six geographic tiles to facilitate NASA POWER data collection.

## Data Preprocessing and Feature Engineering

The preprocessing stage prepares the raw datasets for analysis and machine learning.

### FIRMS Processing

The `clean_firms.py` script performs:
- Data type filtering.
- Confidence filtering.
- Geographic grid assignment.
- Daily aggregation by grid cell.
- Calculation of fire detection counts and total FRP.

Output:

`data/interim/fires_daily.parquet`

### Weather Data and Fusion

The `build_panel.py` script:
- Loads and combines NASA POWER variables.
- Merges weather and fire data using grid cells and dates.
- Creates the binary target `burned`.
- Generates additional weather and temporal features.

### Feature Engineering

The generated features include:
- Rolling precipitation over 7, 14 and 30 observations.
- Rolling maximum temperature averages.
- Rolling minimum humidity.
- Consecutive dry-day counts.
- Seasonal features using sine and cosine transformations.

Final dataset:

`data/processed/fire_panel.parquet`

## Fire Risk Model

The trained model is stored in:

- `models/fire_model.joblib` — calibrated scikit-learn classifier, supplied as a
  packaged artifact: `{"model": ..., "features": [...], "horizon_days": 3,
  "risk_bins": [...], "risk_labels": [...]}`.
- `models/model_card.json` — model version, feature order, target horizon, risk
  bands and calibration metadata. The app cross-checks the artifact's packaged
  metadata against this card and refuses to score if they disagree.

The model expects **20 features**, in model-card order:

`lat_cell`, `lon_cell`, `T2M`, `T2M_MAX`, `RH2M`, `WS2M`, `PRECTOTCORR`, `GWETROOT`,
`rain_7d`, `tmax_7d`, `rhmin_7d`, `rain_14d`, `tmax_14d`, `rhmin_14d`,
`rain_30d`, `tmax_30d`, `rhmin_30d`, `dry_days`, `doy_sin`, `doy_cos`.

Solar radiation (`ALLSKY_SFC_SW_DWN`) is collected for analysis but is not a model feature.

The model covers **northern Algeria only** (32.0°–37.5°N, -2.0°–9.0°E on the
0.5° × 0.625° NASA POWER grid). Cells outside that grid are not scored.

The target is **at least one fire in the grid cell within the next 3 days**
(`horizon_days = 3`). This differs from the panel's `burned` column, which is a
same-day detection label used for the observed overlay and retrospective analysis.

Probabilities are **isotonic-calibrated on the 2021 validation year** (training
2018–2020, test 2022–2024) and are mapped to these risk bands:

| Band | Probability |
|---|---|
| LOW | below 1% |
| MODERATE | 1% – 5% |
| HIGH | 5% – 15% |
| VERY HIGH | 15% or more |

The dashboard panel covers 2024–2026, which is newer than the model's test
period. The app labels those dates as new data; read the probabilities as
relative risk screening, not as calibrated absolute probabilities for 2024–2026.

The supplied artifact does not include evaluation metrics, so the model card
records `"metrics": null` rather than reporting numbers from a different model.

Run the test suite from the project root:

```bash
python -m pytest -q
```

The model was trained with **scikit-learn 1.9.1**, which is pinned in
`requirements.txt`; loading the artifact with another version can fail.
Install all dependencies (app, notebooks and tests) with:

```bash
pip install -r requirements.txt
```

## Important Considerations

### Class Imbalance

Fire occurrences are much less frequent than non-fire observations.

Therefore, accuracy alone is not sufficient to evaluate the models.

The project will also consider:
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- PR-AUC.

### Data Leakage

Variables directly derived from fire observations, such as `n_detections` and `total_frp`, must not be used as predictors when predicting the `burned` target.

Data preprocessing and transformations must also be handled carefully to avoid using information from the test dataset during training.

### Time-Based Evaluation

A chronological train/validation/test split is preferred.

Training data should come from earlier dates, while validation and test data should come from later dates.

### Limitations

- NASA POWER has a relatively coarse spatial resolution.
- Satellite detections may miss some fire events.
- The 2026 dataset is incomplete.
- Fire detections do not necessarily represent every real fire.
- Model predictions are estimates and cannot guarantee whether a fire will occur.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- NASA FIRMS
- NASA POWER
- Git and GitHub

## Project Workflow

1. Data collection from NASA FIRMS and NASA POWER.
2. Data cleaning and preprocessing.
3. Feature engineering and dataset construction.
4. Exploratory Data Analysis (EDA).
5. Statistical analysis.
6. Machine learning classification and clustering.
7. Hyperparameter optimization using GridSearchCV with at least 5 folds.
8. Model evaluation and comparison.
9. Streamlit application development.
10. Scientific report preparation.

Deep Learning and Large Language Models are not used as machine learning models, in accordance with the course requirements.

## References

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)
- [NASA FIRMS Download](https://firms.modaps.eosdis.nasa.gov/download/)
- [NASA POWER](https://power.larc.nasa.gov/)
- [NASA POWER Documentation](https://power.larc.nasa.gov/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)