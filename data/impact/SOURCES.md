# Impact data sources, licences and attribution

This document records the provenance of the exposure layers used by the
independent Wildfire Impact Assessment module. It is the human-readable
companion to the machine-readable `data/impact/manifest.json` written by
`scripts/build_impact_layers.py`.

The raw downloads live in `data/impact/raw/` (git-ignored). Only small derived
parquet artifacts are committed.

## Required sources

### ESA WorldCover 2021 v200 (land cover)

- **Product:** ESA WorldCover 10 m 2021 v200
- **Role:** Land-cover composition, flammable fuel area and non-fuel area.
- **Licence:** CC BY 4.0
- **Attribution:** © ESA WorldCover project 2021 / Contains modified Copernicus
  Sentinel data (2021) processed by ESA WorldCover consortium
- **Citation:** Zanaga, D., Van De Kerckhove, R., Daems, D., De Keersmaecker, W.,
  Brockmann, C., Kirches, G., Wevers, J., Cartus, O., Santoro, M., Fritz, S.,
  Lesiv, M., Herold, M., Tsendbazar, N.E., Xu, P., Ramoino, F., Arino, O., 2022.
  ESA WorldCover 10 m 2021 v200. https://doi.org/10.5281/zenodo.7254221
- **Product page:** https://esa-worldcover.org/en/data-access
- **Tiles used:** twelve 3° × 3° COGs covering 32.0–37.5 °N and
  1.875 °W–8.75 °E (see DATA_DOWNLOAD.md).

### WorldPop 2020 constrained, UN adjusted (population)

- **Product:** WorldPop 2020 constrained individual country, Algeria, 100 m,
  UN adjusted (`dza_ppp_2020_UNadj_constrained.tif`).
- **Role:** Gridded population counts inside cells and 5/10/25 km rings, and
  settlement population estimates.
- **Licence:** CC BY 4.0
- **Attribution:** WorldPop (www.worldpop.org), licensed under CC BY 4.0
- **Citation:** Bondarenko M., Kerr D., Sorichetta A., and Tatem, A.J. 2020.
  Census/projection-disaggregated gridded population datasets for 189 countries
  in 2020 using Built-Settlement Growth Model (BSGM) outputs. WorldPop,
  University of Southampton, UK. doi:10.5258/SOTON/WP00684
- **Dataset page:** https://hub.worldpop.org/geodata/summary?id=49737

### OpenStreetMap — Geofabrik Algeria extract

- **Product:** Geofabrik `algeria-latest-free.shp.zip` snapshot.
- **Role:** Mapped roads, buildings, settlements/places and critical-facility
  points.
- **Licence:** ODbL 1.0
- **Attribution:** © OpenStreetMap contributors, available under the Open
  Database License (ODbL 1.0). https://www.openstreetmap.org/copyright
- **Download:** https://download.geofabrik.de/africa/algeria-latest-free.shp.zip
- **Note:** OSM completeness varies by region and category. Counts are lower
  bounds, and a missing feature in OSM does not mean the feature does not exist.

## Optional / diagnostic sources

### Global Human Settlement Layer (GHSL) GHS-BUILT-S R2023A

- **Role:** Built-up surface cross-check for the OSM-vs-GHSL diagnostic in
  `scripts/validate_impact_layers.py`. It is not used to replace OSM counts.
- **Licence:** CC BY 4.0 (European Commission reuse notice)
- **Attribution:** European Commission Joint Research Centre, Global Human
  Settlement Layer (GHSL), GHS-BUILT-S R2023A.
- **Citation:** Pesaresi, M.; Politis, P. (2023): GHS-BUILT-S R2023A - GHS
  built-up surface grid, derived from Sentinel2 composite and Landsat,
  multitemporal (1975-2030). European Commission, JRC.
  doi:10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA
- **Portal:** https://human-settlement.emergency.copernicus.eu/download.php?ds=bu

### OCHA COD-AB Algeria (administrative boundaries)

- **Role:** Administrative assignment for regional comparison only.
- **Licence:** CC BY 3.0 IGO (verify the licence shown on each release).
- **Attribution:** UN OCHA / Humanitarian Data Exchange (HDX), COD-AB Algeria.
- **Dataset page:** https://data.humdata.org/dataset/cod-ab-dza
- **Note:** Assignment uses the cell centre and can misallocate boundary cells.

## Existing, unchanged sources

- **NASA FIRMS VIIRS SV-C2** — observed active-fire detections already present
  in `data/processed/fire_panel.parquet`. Open, attribution requested.
- **NASA POWER** — meteorological predictors already present in the panel.
  Open, attribution requested.

## Deliberately excluded

- **GADM** — redistribution restrictions; use OCHA COD-AB or Natural Earth.
- **WDPA / Protected Planet** — non-commercial licence.

## Licence summary

| Source | Licence | Redistributable derived artifact |
| --- | --- | --- |
| ESA WorldCover 2021 v200 | CC BY 4.0 | Yes, with attribution |
| WorldPop 2020 constrained | CC BY 4.0 | Yes, with attribution |
| OpenStreetMap / Geofabrik | ODbL 1.0 | Yes, with attribution and share-alike for derived databases |
| GHSL R2023A | CC BY 4.0 | Yes, with attribution |
| OCHA COD-AB | CC BY 3.0 IGO | Yes, with attribution |

When publishing maps or reports built from these artifacts, include the
attribution strings above. The application's report builder must reproduce them
verbatim.
