# Fire Risk Prediction from NASA FIRMS + NASA POWER

Academic Data Mining project: predicts wildfire risk for a location from user-supplied weather inputs (temperature, humidity, coordinates, etc.), trained on a merged NASA FIRMS (fire detections) + NASA POWER (weather) dataset.

## Idea in One Picture

```text
POWER (weather grid, complete)       FIRMS (fire points, sparse)
every cell x every day                only where/when a fire was seen
        │                                      │
        └──────────── left join on ────────────┘
                    (grid cell, date)
                         │
                         ▼
        one row per cell per day, label = burned 0/1
                         │
                         ▼
        + rolling rain/temp/humidity features (7/14/30-day)
                         │
                         ▼
              data/processed/fire_panel.parquet
                         │
                         ▼
                  train / evaluate model
```

FIRMS alone has no negative examples ("no fire here today" is never recorded), so the panel is built from POWER's complete grid and stamped with fire labels wherever a FIRMS detection lands.

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
├── report/
│   └── figures/
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

## How to Run It

### 0. Setup

```bash
pip install -r requirements.txt
```

### 1. Download FIRMS Data (Manual, One-Time)

Go to:

https://firms.modaps.eosdis.nasa.gov/download/

Select the study area, date range and VIIRS S-NPP sensor. Download the CSV files and place them in `data/raw/`.

The project uses `merge_firms.py` to combine the collected FIRMS files into:

`data/raw/firms_raw.csv`

### 2. Set Your Study Area

Edit `src/config.py`:

- `LAT_MIN`
- `LAT_MAX`
- `LON_MIN`
- `LON_MAX`
- `START_YEAR`
- `END_YEAR`

The study area covers northern Algeria and the surrounding region.

### 3. Run the Pipeline in Order

From the project root:

```bash
python src/fetch_power.py
python merge_firms.py
python src/clean_firms.py
python src/build_panel.py
python src/train_baseline.py
```

Each step reads the previous step's output from `data/`, so you can rerun individual steps without repeating earlier ones.

### 4. Explore and Write Up

Open `notebooks/exploration.ipynb` for plots such as:

- Fire counts over time.
- Geographic distribution of fire detections.
- Weather feature distributions.
- Relationships between meteorological variables and fire occurrence.

These visualizations can be used in the scientific report.

## Data Sources and Collection Details

### 1. NASA FIRMS — Fire Detection Data

The fire detection data are obtained from NASA's Fire Information for Resource Management System (FIRMS).

- Official website: https://firms.modaps.eosdis.nasa.gov/
- Satellite product: Suomi-NPP VIIRS Collection 2.
- Data format: CSV.
- Study period: 2024–2026.
- Approximate number of raw records collected: 86,884.

The FIRMS dataset contains information about detected fire events, including:

- Latitude and longitude.
- Acquisition date.
- Detection type.
- Confidence level.
- Fire Radiative Power (FRP).

The collected files are merged into:

`data/raw/firms_raw.csv`

### 2. NASA POWER — Weather Data

Meteorological data are obtained from NASA POWER (Prediction Of Worldwide Energy Resources).

- Official website: https://power.larc.nasa.gov/
- Data type: Daily meteorological data.
- Data format: CSV.
- Study period: 2024–2026.
- Number of meteorological variables: 7.
- Geographic organization: 6 spatial tiles.
- Total collected files: 126.

The following meteorological variables are used:

| Variable | Description |
|---|---|
| T2M | Mean air temperature at 2 meters |
| T2M_MAX | Maximum air temperature at 2 meters |
| RH2M | Relative humidity at 2 meters |
| WS2M | Wind speed at 2 meters |
| PRECTOTCORR | Corrected precipitation |
| ALLSKY_SFC_SW_DWN | All-sky surface shortwave downward irradiance |
| GWETROOT | Root zone soil wetness |

The downloaded files are stored in:

`data/raw/power_cache/`

### 3. Study Area

The project focuses on a geographic area defined by the following coordinates:

| Parameter | Value |
|---|---:|
| Minimum latitude | 32° |
| Maximum latitude | 37.5° |
| Minimum longitude | -2° |
| Maximum longitude | 9° |

The study area is divided into six geographic tiles to organize NASA POWER data collection.

### 4. Study Period

The project covers the following years:

- 2024
- 2025
- 2026

NASA POWER data for 2026 are partial because the latest available observations depend on the data publication delay.

### 5. Data Collection Summary

| Source | Collected data |
|---|---|
| NASA FIRMS | Approximately 86,884 raw records |
| NASA POWER | 126 CSV files |
| Weather variables | 7 |
| Study years | 3 |
| Geographic tiles | 6 |

These datasets provide the input for the preprocessing, exploratory analysis, statistical testing and machine learning stages.

## Preprocessing and Feature Engineering

The preprocessing stage transforms the raw data into a structured dataset suitable for machine learning.

### FIRMS Processing

The FIRMS processing script performs:

- Data type filtering.
- Confidence filtering.
- Geographic grid assignment.
- Grouping fire detections by grid cell and day.
- Calculation of daily detection counts and total FRP.

Output:

`data/interim/fires_daily.parquet`

### Weather Processing and Data Fusion

The panel construction script:

- Loads NASA POWER meteorological data.
- Combines the weather variables.
- Joins fire observations with the weather grid using geographic cells and dates.
- Creates the binary target variable `burned`.
- Generates temporal features.

The temporal features include:

- Rolling precipitation features over 7, 14 and 30 days.
- Rolling temperature features.
- Rolling humidity features.
- Consecutive dry-day counts.
- Seasonal sine and cosine features.

Final dataset:

`data/processed/fire_panel.parquet`

## Notes for the Report

### Class Imbalance

Fires are rare events, representing approximately 1–3% of the dataset rows in the current panel.

Accuracy alone is therefore insufficient to evaluate the model. Metrics such as Precision, Recall, F1-score, PR-AUC and ROC-AUC should also be considered.

### Time-Based Split

The model should be evaluated using a chronological split rather than a random split.

Training data must come from earlier dates, while validation and test data must come from later dates. This avoids using future information to predict past observations.

### Coordinates and Predictive Features

Raw latitude and longitude require careful consideration when selecting model features. They can allow a model to memorize geographic patterns instead of learning meteorological relationships.

Their inclusion or exclusion should be justified according to the prediction objective.

### Data Leakage

Variables derived from fire observations, such as `n_detections` and `total_frp`, must not be used as predictors when they are also used to construct the target variable `burned`.

All transformations that learn parameters from the data, such as imputation and normalization, must be fitted using training data only.

### Known Limitations

- NASA POWER's approximately 50 km native grid may not capture local terrain and microclimate effects.
- FIRMS may miss some small or short-lived fires, including detections affected by cloud cover.
- Satellite fire detections do not necessarily represent every fire event.
- The 2026 meteorological data are incomplete.
- The resulting model estimates risk from available data and does not guarantee that a fire will or will not occur.

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
5. Statistical hypothesis testing.
6. Classical Machine Learning classification and clustering.
7. Hyperparameter optimization using GridSearchCV with at least 5 folds.
8. Model evaluation and comparison.
9. Development of a Streamlit web application.
10. Scientific report preparation.

Deep Learning and Large Language Models are not used as machine learning models in this project, in accordance with the course requirements.

## References

- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/
- NASA FIRMS Download: https://firms.modaps.eosdis.nasa.gov/download/
- NASA POWER: https://power.larc.nasa.gov/
- NASA POWER Documentation: https://power.larc.nasa.gov/docs/
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- Streamlit Documentation: https://docs.streamlit.io/