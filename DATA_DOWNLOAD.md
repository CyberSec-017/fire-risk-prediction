# Manual data download for the Wildfire Impact Assessment

The impact build is **offline and manual by design**. No script downloads data
for you, and nothing in `data/impact/raw/` is committed. Follow the steps below
once, then run the build and validation scripts.

All paths are relative to the repository root:

```text
data/impact/raw/
├── ESA_WorldCover_10m_2021_v200_N30W003_Map.tif
├── ... (eleven more WorldCover tiles)
├── dza_ppp_2020_UNadj_constrained.tif
├── algeria-latest-free.shp/
│   ├── gis_osm_buildings_a_free_1.shp
│   ├── gis_osm_places_free_1.shp
│   ├── gis_osm_pois_free_1.shp
│   └── gis_osm_roads_free_1.shp
├── dza_adm1.shp              (optional, administrative comparison)
├── dza_adm2.shp              (optional)
├── GHSL_BUILT_S_2020_4326.tif (optional, OSM-vs-GHSL diagnostic)
└── checksums.json            (created in step 5)
```

The model grid uses cell centres 32.0–37.5 °N and 1.875 °W–8.75 °E (the panel
contains 209 of the 216 full-grid cells). The downloads below cover that whole
box with a margin.

---

## 1. ESA WorldCover 2021 v200 (12 tiles, ~2–4 GB total)

Product page: <https://esa-worldcover.org/en/data-access>

Direct COG URLs follow this pattern:

```text
https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_<TILE>_Map.tif
```

The study box is covered by the twelve 3° × 3° tiles below. Each tile is named
by its lower-left corner.

| Latitude band | Longitude bands | Tile names |
| --- | --- | --- |
| 30–33 °N | 3 °W–0°, 0–3 °E, 3–6 °E, 6–9 °E | `N30W003`, `N30E000`, `N30E003`, `N30E006` |
| 33–36 °N | 3 °W–0°, 0–3 °E, 3–6 °E, 6–9 °E | `N33W003`, `N33E000`, `N33E003`, `N33E006` |
| 36–39 °N | 3 °W–0°, 0–3 °E, 3–6 °E, 6–9 °E | `N36W003`, `N36E000`, `N36E003`, `N36E006` |

PowerShell (run from the repository root):

```powershell
$base = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
$tiles = @(
  "N30W003","N30E000","N30E003","N30E006",
  "N33W003","N33E000","N33E003","N33E006",
  "N36W003","N36E000","N36E003","N36E006"
)
New-Item -ItemType Directory -Force data/impact/raw | Out-Null
foreach ($tile in $tiles) {
  $name = "ESA_WorldCover_10m_2021_v200_${tile}_Map.tif"
  Write-Host "Downloading $name"
  Invoke-WebRequest -Uri "$base/$name" -OutFile "data/impact/raw/$name"
}
```

Bash equivalent:

```bash
mkdir -p data/impact/raw
for tile in N30W003 N30E000 N30E003 N30E006 \
            N33W003 N33E000 N33E003 N33E006 \
            N36W003 N36E000 N36E003 N36E006; do
  name="ESA_WorldCover_10m_2021_v200_${tile}_Map.tif"
  curl -fL --retry 3 \
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/$name" \
    -o "data/impact/raw/$name"
done
```

> Keep only the `_Map.tif` layer. The `_InputQuality.tif` layers are not used.

Expected file names (all twelve must exist):

```text
ESA_WorldCover_10m_2021_v200_N30W003_Map.tif
ESA_WorldCover_10m_2021_v200_N30E000_Map.tif
ESA_WorldCover_10m_2021_v200_N30E003_Map.tif
ESA_WorldCover_10m_2021_v200_N30E006_Map.tif
ESA_WorldCover_10m_2021_v200_N33W003_Map.tif
ESA_WorldCover_10m_2021_v200_N33E000_Map.tif
ESA_WorldCover_10m_2021_v200_N33E003_Map.tif
ESA_WorldCover_10m_2021_v200_N33E006_Map.tif
ESA_WorldCover_10m_2021_v200_N36W003_Map.tif
ESA_WorldCover_10m_2021_v200_N36E000_Map.tif
ESA_WorldCover_10m_2021_v200_N36E003_Map.tif
ESA_WorldCover_10m_2021_v200_N36E006_Map.tif
```

---

## 2. WorldPop 2020 constrained, UN adjusted (1 file, ~9 MB)

Dataset page: <https://hub.worldpop.org/geodata/summary?id=49737>

Direct URL:

```text
https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/DZA/dza_ppp_2020_UNadj_constrained.tif
```

PowerShell:

```powershell
Invoke-WebRequest `
  -Uri "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/DZA/dza_ppp_2020_UNadj_constrained.tif" `
  -OutFile "data/impact/raw/dza_ppp_2020_UNadj_constrained.tif"
```

Bash:

```bash
curl -fL --retry 3 \
  "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/DZA/dza_ppp_2020_UNadj_constrained.tif" \
  -o data/impact/raw/dza_ppp_2020_UNadj_constrained.tif
```

Expected file name: `dza_ppp_2020_UNadj_constrained.tif`.

Use the **UN adjusted constrained** file so the national total matches UN
estimates. Do not substitute the 1 km or unconstrained files; the build expects
the 100 m product and reports a checksum mismatch otherwise.

---

## 3. OpenStreetMap — Geofabrik Algeria shapefile extract (~100–300 MB)

Download page: <https://download.geofabrik.de/africa/algeria.html>

Direct URL:

```text
https://download.geofabrik.de/africa/algeria-latest-free.shp.zip
```

PowerShell:

```powershell
Invoke-WebRequest `
  -Uri "https://download.geofabrik.de/africa/algeria-latest-free.shp.zip" `
  -OutFile "data/impact/raw/algeria-latest-free.shp.zip"
Expand-Archive `
  -Path "data/impact/raw/algeria-latest-free.shp.zip" `
  -DestinationPath "data/impact/raw/algeria-latest-free.shp" `
  -Force
```

Bash:

```bash
curl -fL --retry 3 \
  "https://download.geofabrik.de/africa/algeria-latest-free.shp.zip" \
  -o data/impact/raw/algeria-latest-free.shp.zip
unzip -o data/impact/raw/algeria-latest-free.shp.zip \
  -d data/impact/raw/algeria-latest-free.shp
```

The build needs these four shapefiles (with their `.dbf`, `.shx`, `.prj`
sidecars), which the Geofabrik archive places inside
`data/impact/raw/algeria-latest-free.shp/`:

```text
gis_osm_roads_free_1.shp
gis_osm_buildings_a_free_1.shp
gis_osm_pois_free_1.shp
gis_osm_places_free_1.shp
```

The build searches recursively, so a single extra nesting level is fine. You may
delete the archive after extracting.

---

## 4. Optional diagnostic layers

### GHSL built-up surface (OSM-vs-GHSL diagnostic)

Portal: <https://human-settlement.emergency.copernicus.eu/download.php?ds=bu>

Choose **GHS-BUILT-S**, epoch **2020**, resolution **3 arcsec**, coordinate
system **WGS84**, classification **Total RES+NRES**. Download the tile(s)
covering 32–37.5 °N / 2 °W–9 °E, mosaic/crop them if needed, and save the result
as:

```text
data/impact/raw/GHSL_BUILT_S_2020_4326.tif
```

Values must be square metres of built-up surface per pixel. If this file is
absent, the build still succeeds and the validator reports the OSM-vs-GHSL
diagnostic as unavailable.

### OCHA COD-AB Algeria (administrative comparison)

Dataset page: <https://data.humdata.org/dataset/cod-ab-dza>

Download the ADM1 and ADM2 shapefiles and place them at:

```text
data/impact/raw/dza_adm1.shp
data/impact/raw/dza_adm2.shp
```

Keep the `.dbf`, `.shx` and `.prj` sidecars. These files are optional; without
them `data/impact/admin_units.parquet` is not written and the UI reports it as
unavailable.

---

## 5. Record and verify SHA-256 checksums

The build never trusts unverified inputs. The first time, record the checksums
of the files **you just downloaded**:

```bash
python scripts/build_impact_layers.py --record-checksums
```

This writes `data/impact/raw/checksums.json`. Commit that small file so later
builds verify the exact same downloads. If you re-download a file, verify its
hash against the publisher or re-record it deliberately; a changed hash makes
the build stop with a clear error.

Then run the build:

```bash
python scripts/build_impact_layers.py
```

The script fails loudly, before writing any artifact, if a required file is
missing, a checksum is absent, or a checksum does not match. It refuses to
substitute zeros for missing data.

---

## 6. Validate the derived artifacts

```bash
python scripts/validate_impact_layers.py --write-report
```

This checks cell-area sums, land-cover fractions, population plausibility and
the OSM-vs-GHSL diagnostic. It writes `data/impact/validation_report.json` and
exits non-zero if any **error** is found. Warnings do not fail the run.

---

## Troubleshooting

- **`MissingSourceError`** — one or more required files are absent. Re-read the
  list in the error message and the expected names above.
- **`ChecksumError`** — a file changed or the checksum entry is missing. Confirm
  you downloaded the exact product/version, then re-record deliberately if
  needed.
- **`BuildDependencyError`** — install the build-only stack:
  `python -m pip install -r requirements-impact.txt`.
- **WorldPop all-NaN cells** — the constrained raster only covers mapped
  settled areas. Cells with no valid pixels remain explicit missing values;
  they are never filled with zero.
- **Large runtime** — WorldCover is 10 m data. The build reads it block by
  block, but expect tens of minutes on a laptop. Use `--block-size` to trade
  memory for speed.
