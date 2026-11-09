# Fire Risk Prediction from NASA FIRMS + NASA POWER

Academic data mining project: predicts wildfire risk for a location from
user-supplied weather inputs (temperature, humidity, coordinates, etc.),
trained on a merged NASA FIRMS (fire detections) + NASA POWER (weather)
dataset.

## Idea in one picture

```
POWER  (weather grid, complete)        FIRMS  (fire points, sparse)
every cell x every day                 only where/when a fire was seen
        │                                        │
        └──────────── left join on ──────────────┘
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

FIRMS alone has no negative examples ("no fire here today" is never
recorded), so the panel is built from POWER's complete grid and stamped
with fire labels wherever a FIRMS detection lands.

## Folder structure

```
fire-risk-project/
├── data/
│   ├── raw/
│   │   ├── firms_raw.csv        <- you download this manually (see below)
│   │   └── power_cache/         <- fetch_power.py fills this in
│   ├── interim/
│   │   └── fires_daily.parquet  <- clean_firms.py output
│   └── processed/
│       ├── fire_panel.parquet   <- build_panel.py output (final dataset)
│       ├── fire_model.joblib    <- train_baseline.py output (trained model)
│       └── model_card.json      <- feature contract + test metrics
├── src/
│   ├── config.py                <- all settings: study area, years, parameters
│   ├── fetch_power.py           <- Step 1: download weather data
│   ├── clean_firms.py           <- Step 2: clean + snap fire points to grid
│   ├── build_panel.py           <- Step 3+4: join + rolling features
│   └── train_baseline.py        <- Step 5: model, evaluation, export
├── notebooks/
│   └── exploration.ipynb        <- plots and sanity checks for the report
├── tests/
│   ├── test_pipeline.py         <- pipeline regression tests (pytest)
│   ├── test_app_core.py         <- dashboard logic tests (pytest)
│   └── test_app_smoke.py        <- Streamlit startup smoke test
├── app.py                       <- Streamlit dashboard (UI)
├── app_core.py                  <- dashboard logic: model loading + features
├── .streamlit/
│   └── config.toml              <- dark dashboard theme
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## How to run it

### 0. Setup

```bash
pip install -r requirements.txt
```

### 1. Download FIRMS data (manual, one-time)

Go to https://firms.modaps.eosdis.nasa.gov/download/, draw your study
area, pick a date range and the VIIRS S-NPP sensor, download the CSV,
and save it as `data/raw/firms_raw.csv`.

### 2. Set your study area

Edit `src/config.py`: `LAT_MIN/MAX`, `LON_MIN/MAX`, `START_YEAR`,
`END_YEAR`. Defaults are set to northern Algeria.

### 3. Run the pipeline in order

```bash
cd src
python fetch_power.py        # downloads weather data (slow, cached, resumable)
python clean_firms.py        # cleans + grids the fire detections
python build_panel.py        # builds the final labeled + featured dataset
python train_baseline.py     # trains and evaluates a baseline model
```

Each step reads the previous step's output from `data/`, so you can
re-run any single step without repeating earlier ones.

### 4. Run the dashboard

```bash
streamlit run app.py
```

The app is a single-page prediction dashboard. The map, weather inputs and
risk result are always on screen:

- **Sidebar** — model status badge, version/training metadata, the
  manual-vs-live prediction-mode toggle, and a **Reload model** button.
- **Location & weather** — choose a preset or custom coordinates and a date,
  then enter manual weather or fetch 120 days of NASA POWER history.
- **Risk assessment** — the calibrated risk card, the selected grid cell on
  the map, the feature vector used, and an optional sensitivity sweep.

Manual mode re-scores instantly as the sliders move. Live mode fetches NASA
POWER only when you press **Fetch & predict**, so the page never blocks on the
network during unrelated changes.

The status pill makes the model state explicit:

| State | Meaning |
| --- | --- |
| `REAL MODEL` | `fire_model.joblib` and `model_card.json` exist and agree. |
| `DEMO MODE` | The model artifact is missing; a transparent weather heuristic is used. |
| `MODEL ERROR` | The artifact/card is corrupt or the contract does not match; predictions are disabled. |

The app reads the authoritative feature contract from
`data/processed/model_card.json`, so it stays aligned with the training
pipeline without duplicating the feature list.

### 5. Explore and write up

Open `notebooks/exploration.ipynb` for plots (fire count over time, a
risk map, feature distributions) — these become your report's figures.

## Notes for the report

- **Class imbalance**: fires are rare (measured at ~2.7% of panel rows on the
  current data). Accuracy is meaningless here — the model reports PR-AUC and
  Brier score instead.
- **Time-based split**: training uses years before `VAL_YEAR`, that year fits
  the probability calibrator, and every later year is the test set. A random
  split would leak information because adjacent days/cells are correlated.
- **Coordinates**: `USE_COORDS = True` includes `lat_cell`/`lon_cell` as
  features. Set it to `False` in `src/train_baseline.py` if you want the model
  to learn from weather only; then re-run training.
- **Model contract**: `train_baseline.py` writes `data/processed/model_card.json`
  with the exact feature names, their order, a feature hash and the test
  metrics. Any serving layer should check these before loading the model.
- **Serving contract**: the dashboard reads `model_card.json` and builds the
  20-feature vector in the card's order, including `lat_cell`/`lon_cell`. It
  validates the feature hash and the model's `feature_names_in_` before
  enabling real predictions, so a mismatched artifact pair fails loudly
  instead of silently producing bad scores.
- **Known limitations** (worth a paragraph in your report): POWER's
  ~50km grid can't capture local terrain/microclimate effects; FIRMS
  can miss small or short-lived fires under cloud cover.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
