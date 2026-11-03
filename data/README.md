# Raw Data Documentation

This document describes the raw datasets collected for the Fire Risk Prediction project, including their sources, contents, formats and organization.

## 1. Data Sources

The project uses two main data sources:

| Source | Data | Organization |
|---|---|---|
| **NASA FIRMS** | Fire detections | Merged CSV file |
| **NASA POWER** | Temperature, humidity, wind, precipitation, solar radiation and soil moisture | CSV files organized by variable, year and geographic tile |

## 2. NASA FIRMS — Fire Detection Data

**Source:** NASA Fire Information for Resource Management System (FIRMS)

Website: https://firms.modaps.eosdis.nasa.gov/

FIRMS provides satellite-based fire and thermal anomaly detections.

- **Satellite product:** Suomi-NPP VIIRS Collection 2.
- **Format:** CSV.
- **Period:** 2024–2026.
- **Geographic coverage:** Latitude 32° to 37.5°, longitude -2° to 9°.
- **Number of raw records:** Approximately 86,884 lines.

The dataset contains information such as:

- Latitude and longitude.
- Acquisition date and time.
- Confidence level.
- Detection type.
- Fire Radiative Power (FRP).

### Organization

The downloaded FIRMS CSV files are stored in `data/raw/` and combined into:

`data/raw/firms_raw.csv`

This file contains the raw fire detections used by the preprocessing pipeline.

## 3. NASA POWER — Meteorological Data

**Source:** NASA Prediction Of Worldwide Energy Resources (POWER)

Website: https://power.larc.nasa.gov/

NASA POWER provides daily meteorological and solar data.

- **Format:** CSV.
- **Period:** 2024–2026.
- **Geographic coverage:** Latitude 32° to 37.5°, longitude -2° to 9°.
- **Expected number of files:** 126 CSV files.
- **Number of variables:** 7.
- **Geographic organization:** 6 tiles.

### 3.1 Meteorological Variables

The following table presents the variables collected from NASA POWER:

| Variable | Meaning |
|---|---|
| `T2M` | Mean temperature at 2 meters |
| `T2M_MAX` | Maximum temperature at 2 meters |
| `RH2M` | Relative humidity at 2 meters |
| `WS2M` | Wind speed at 2 meters |
| `PRECTOTCORR` | Corrected precipitation |
| `ALLSKY_SFC_SW_DWN` | Incoming solar radiation under all-sky conditions |
| `GWETROOT` | Root zone soil wetness |

These variables describe weather and soil conditions that may influence wildfire occurrence. They will be used for preprocessing, exploratory analysis and machine learning.

### 3.2 Organization

The NASA POWER CSV files are stored in:

`data/raw/power_cache/`

Files are organized by:

- Meteorological variable.
- Year.
- Geographic tile.

Each file contains daily observations for one variable and one geographic tile.

### 3.3 Missing Values

NASA POWER uses the value `-999` to indicate missing data.

During preprocessing, this value must be converted to a missing value (NaN) and must not be treated as a real measurement.

## 4. Geographic and Temporal Coverage

| Parameter | Value |
|---|---|
| Minimum latitude | 32° |
| Maximum latitude | 37.5° |
| Minimum longitude | -2° |
| Maximum longitude | 9° |
| Start year | 2024 |
| End year | 2026 |

**Note:** The 2026 data are partial because the year is still in progress. NASA POWER data may also be published with a delay.

## 5. Raw Data Summary

| Dataset | Organization | Quantity |
|---|---|---:|
| FIRMS | Merged CSV | Approximately 86,884 raw records |
| POWER | CSV files by variable, year and tile | 126 expected files |
| POWER variables | Meteorological variables | 7 |
| POWER geographic tiles | Spatial divisions | 6 |

## 6. Data Usage

The raw datasets are used as input for the preprocessing pipeline:

1. Clean and filter FIRMS fire detections.
2. Process NASA POWER meteorological data.
3. Handle missing values.
4. Combine fire detections with meteorological data.
5. Generate the final dataset for exploratory analysis and machine learning.

**Important:** Raw data are stored separately from intermediate and processed datasets. This helps preserve the original data and makes the processing steps easier to track.

## 7. FIRMS Cleaning Results

The script `src/clean_firms.py` cleans the raw FIRMS data, applies quality filters, maps fire detections to the NASA POWER geographic grid and groups detections by cell and day.

### 7.1 Cleaning Results

The following results were obtained after running the script:

| Step | Number of detections |
|---|---:|
| Raw FIRMS data | 86,884 |
| After type filter (`type = 0`) | 15,295 |
| After confidence filter | 13,601 |
| After FRP validation | 13,601 |
| After aggregation by cell and day | 5,230 |

### 7.2 Final Output

The cleaned dataset is saved in:

`data/interim/fires_daily.parquet`

Dataset characteristics:

- **Number of rows:** 5,230
- **Number of columns:** 5
- **First date:** 2024-01-01
- **Last date:** 2026-07-31
- **Missing values in final columns:** 0
- **Duplicates on `(lat_cell, lon_cell, date)`:** 0

The data are grouped to obtain one row per geographic cell and per day.

**Note:** The available period in this dataset ends on July 31, 2026. It does not cover the entire year 2026.

## 8. Column Dictionary

The file `data/interim/fires_daily.parquet` contains the following columns:

| Column | Type | Description |
|---|---|---|
| `lat_cell` | Float | Latitude of the NASA POWER geographic cell center. |

| `lon_cell` | Float | Longitude of the NASA POWER geographic cell center. |
| `date` | DateTime | Date when fire detections were recorded. |
| `n_detections` | Integer | Number of FIRMS detections recorded in the same cell during the day. |
| `total_frp` | Float | Sum of the Fire Radiative Power values detected in the cell during the day. |

These columns provide the fire data that will be used in the next processing step and combined with NASA POWER meteorological data.