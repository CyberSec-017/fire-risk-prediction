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
│       └── fire_panel.parquet   <- build_panel.py output (final dataset)
├── src/
│   ├── config.py                <- all settings: study area, years, parameters
│   ├── fetch_power.py           <- Step 1: download weather data
│   ├── clean_firms.py           <- Step 2: clean + snap fire points to grid
│   ├── build_panel.py           <- Step 3+4: join + rolling features
│   └── train_baseline.py        <- Step 5: model + evaluation
├── notebooks/
│   └── exploration.ipynb        <- plots and sanity checks for the report
├── report/
│   └── figures/                 <- exported charts go here
├── requirements.txt
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

### 4. Explore and write up

Open `notebooks/exploration.ipynb` for plots (fire count over time, a
risk map, feature distributions) — these become your report's figures.

## Notes for the report

- **Class imbalance**: fires are rare (~1-3% of rows). Accuracy is
  meaningless here — the model reports PR-AUC and Brier score instead.
- **Time-based split**: the model is evaluated on the most recent year
  only, never a random split, because adjacent days/cells are strongly
  correlated and a random split would leak information.
- **Coordinates excluded from features**: including raw lat/lon lets
  the model memorize "this region burns" instead of learning from
  weather, which would defeat the project's goal.
- **Known limitations** (worth a paragraph in your report): POWER's
  ~50km grid can't capture local terrain/microclimate effects; FIRMS
  can miss small or short-lived fires under cloud cover.
