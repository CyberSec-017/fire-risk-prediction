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

The data contain information such as:

- Latitude and longitude.
- Acquisition date and time.
- Confidence level.
- Detection type.
- Fire Radiative Power (FRP).

### Organization

The downloaded FIRMS CSV files are stored in `data/raw/` and merged into:

`data/raw/firms_raw.csv`

This merged file contains the raw fire detection records used by the preprocessing pipeline.

## 3. NASA POWER — Meteorological Data

**Source:** NASA Prediction Of Worldwide Energy Resources (POWER)

Website: https://power.larc.nasa.gov/

NASA POWER provides daily meteorological and solar data.

- **Format:** CSV.
- **Period:** 2024–2026.
- **Geographic coverage:** Latitude 32° to 37.5°, longitude -2° to 9°.
- **Number of files:** 126 CSV files.
- **Number of variables:** 7.
- **Geographic organization:** 6 tiles.

### Meteorological Variables

| Variable | Description |
|---|---|
| `T2M` | Mean temperature at 2 meters |
| `T2M_MAX` | Maximum temperature at 2 meters |
| `RH2M` | Relative humidity at 2 meters |
| `WS2M` | Wind speed at 2 meters |
| `PRECTOTCORR` | Corrected precipitation |
| `ALLSKY_SFC_SW_DWN` | All-sky surface shortwave downward radiation |
| `GWETROOT` | Root zone soil wetness |

### Organization

The NASA POWER CSV files are stored in:

`data/raw/power_cache/`

Files are organized according to:

- Meteorological variable.
- Year.
- Geographic tile.

Each CSV contains daily observations for the corresponding variable and geographic tile.

### Missing Values

NASA POWER uses the value `-999` to represent missing data.

This value must be treated as missing during preprocessing and must not be interpreted as a real meteorological measurement.

## 4. Geographic and Temporal Coverage

| Parameter | Value |
|---|---|
| Minimum latitude | 32° |
| Maximum latitude | 37.5° |
| Minimum longitude | -2° |
| Maximum longitude | 9° |
| Start year | 2024 |
| End year | 2026 |

**Note:** The 2026 data may be incomplete because the year is still in progress and NASA POWER data publication can have a delay.

## 5. Raw Data Summary

| Dataset | Organization | Quantity |
|---|---|---:|
| FIRMS | Merged CSV | Approximately 86,884 raw lines |
| POWER | CSV files by variable, year and tile | 126 files |
| POWER variables | Meteorological variables | 7 |
| POWER geographic tiles | Spatial divisions | 6 |

## 6. Data Usage

The raw datasets are the input of the preprocessing pipeline:

1. Clean and filter FIRMS fire detections.
2. Process NASA POWER meteorological variables.
3. Handle missing values.
4. Combine fire detections with meteorological data.
5. Generate the final dataset for exploratory analysis and machine learning.

**Important:** Raw data are preserved separately from intermediate and processed datasets to maintain data traceability.