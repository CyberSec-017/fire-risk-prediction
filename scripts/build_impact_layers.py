"""Build the small impact-exposure parquet artifacts from raw downloads.

This is an **offline build script**.  It reads manually downloaded files from
``data/impact/raw/`` and writes the canonical artifacts described in
``impact/config.py`` plus a provenance manifest.

Design rules
------------
* Geospatial libraries are imported lazily, so the pure helper functions in
  this module can be imported and unit-tested without geopandas/rasterio.
* No network access and no silent downloads.
* Every required input is checksum-verified before any artifact is written.
* Missing inputs fail loudly; missing values are never replaced with zeros.
* The script never imports Streamlit, ``app_core`` or the dashboard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from impact import config, exposure, severity  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORLDCOVER_VERSION = "2021 v200"
WORLDCOVER_YEAR = 2021

WORLDPOP_FILENAME = "dza_ppp_2020_UNadj_constrained.tif"
OSM_FOLDER = "algeria-latest-free.shp"
OSM_REQUIRED = (
    "gis_osm_roads_free_1.shp",
    "gis_osm_buildings_a_free_1.shp",
    "gis_osm_pois_free_1.shp",
    "gis_osm_places_free_1.shp",
)

CHECKSUM_FILENAME = "checksums.json"
MANIFEST_SCHEMA_VERSION = "1.0"

# Download extent = extent of the model-cell centres (not cell edges).
STUDY_LAT_MIN = 32.0
STUDY_LAT_MAX = 37.5
STUDY_LON_MIN = -1.875
STUDY_LON_MAX = 8.75

# ESA WorldCover class code -> canonical land-cover key.  Codes without their
# own canonical column are folded into "other" (never silently dropped).
WORLDCOVER_TO_LC = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "crop",
    50: "built",
    60: "bare",
    70: "other",
    80: "water",
    90: "other",
    95: "other",
    100: "other",
}

ROAD_CLASS_COLUMNS = {
    "primary": "road_km_primary",
    "secondary": "road_km_secondary",
    "tertiary": "road_km_tertiary",
    "track": "road_km_track",
}

CRITICAL_CLASS_COLUMNS = {
    "hospital": "critical_hospital_count",
    "school": "critical_school_count",
    "power": "critical_power_count",
    "airport": "critical_airport_count",
    "water": "critical_water_count",
    "emergency": "critical_emergency_count",
}

ROAD_PRIMARY = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
}
ROAD_SECONDARY = {"secondary", "secondary_link"}
ROAD_TERTIARY = {"tertiary", "tertiary_link"}
ROAD_TRACK = {"track", "track_grade1", "track_grade2", "track_grade3"}

SOURCE_METADATA: dict[str, dict[str, str]] = {
    "worldcover": {
        "name": "ESA WorldCover 2021 v200",
        "version": WORLDCOVER_VERSION,
        "license": "CC BY 4.0",
        "attribution": (
            "© ESA WorldCover project 2021 / Contains modified Copernicus "
            "Sentinel data (2021) processed by ESA WorldCover consortium"
        ),
        "role": "Land-cover composition and flammable fuel area",
    },
    "worldpop": {
        "name": "WorldPop 2020 constrained UN adjusted (Algeria)",
        "version": "2020 / BSGM / UNadj constrained / 100 m",
        "license": "CC BY 4.0",
        "attribution": "WorldPop (www.worldpop.org), licensed under CC BY 4.0",
        "role": "Gridded population, ring population and settlement population",
    },
    "osm": {
        "name": "OpenStreetMap — Geofabrik Algeria extract",
        "version": "algeria-latest-free.shp snapshot",
        "license": "ODbL 1.0",
        "attribution": (
            "© OpenStreetMap contributors, available under the Open Database "
            "License (ODbL 1.0)"
        ),
        "role": "Mapped roads, buildings, places and critical facilities",
    },
    "ghsl": {
        "name": "Global Human Settlement Layer GHS-BUILT-S R2023A",
        "version": "R2023A / 2020 / WGS84 3 arcsec",
        "license": "CC BY 4.0",
        "attribution": (
            "European Commission Joint Research Centre, Global Human "
            "Settlement Layer (GHSL), GHS-BUILT-S R2023A"
        ),
        "role": "Built-up surface cross-check for OSM completeness",
    },
    "ocha": {
        "name": "OCHA COD-AB Algeria",
        "version": "COD-AB snapshot",
        "license": "CC BY 3.0 IGO (verify per release)",
        "attribution": "UN OCHA / Humanitarian Data Exchange (HDX)",
        "role": "Administrative assignment for regional comparison",
    },
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class BuildError(RuntimeError):
    """Base class for explicit, user-facing build failures."""


class MissingSourceError(BuildError):
    """Raised when one or more required raw inputs are absent."""

    def __init__(self, missing: Iterable[str], raw_dir: Path):
        self.missing = tuple(missing)
        message = (
            f"Missing {len(self.missing)} required raw file(s) under {raw_dir}:\n  - "
            + "\n  - ".join(self.missing)
            + "\nDownload them manually as described in DATA_DOWNLOAD.md. "
            "The build will not substitute zeros or download anything."
        )
        super().__init__(message)


class ChecksumError(BuildError):
    """Raised when checksums are absent or do not match."""


class BuildDependencyError(BuildError):
    """Raised when the optional build-only geospatial stack is absent."""


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def _grid_values(start: float, stop: float, step: float) -> list[float]:
    first = math.ceil(start / step - 1e-9) * step
    last = math.floor(stop / step + 1e-9) * step
    count = int(round((last - first) / step)) + 1
    return [round(first + index * step, 10) for index in range(count)]


def worldcover_tile_names(
    lat_min: float = STUDY_LAT_MIN,
    lat_max: float = STUDY_LAT_MAX,
    lon_min: float = STUDY_LON_MIN,
    lon_max: float = STUDY_LON_MAX,
) -> list[str]:
    """Return the WorldCover tile names covering a bounding box."""
    lat_origins = _grid_values(math.floor(lat_min / 3) * 3, lat_max, 3)
    lon_origins = _grid_values(math.floor(lon_min / 3) * 3, lon_max, 3)
    names: list[str] = []
    for lat in lat_origins:
        lat_label = f"N{int(lat):02d}" if lat >= 0 else f"S{abs(int(lat)):02d}"
        for lon in lon_origins:
            lon_label = f"E{int(lon):03d}" if lon >= 0 else f"W{abs(int(lon)):03d}"
            names.append(
                f"ESA_WorldCover_10m_{WORLDCOVER_YEAR}_v200_"
                f"{lat_label}{lon_label}_Map.tif"
            )
    return names


def expected_worldcover_files() -> list[str]:
    return worldcover_tile_names()


def grid_cells() -> pd.DataFrame:
    """Return every cell centre on the model grid."""
    lat_values = _grid_values(config.MODEL_LAT_MIN, config.MODEL_LAT_MAX, config.MODEL_LAT_STEP)
    lon_values = _grid_values(config.MODEL_LON_MIN, config.MODEL_LON_MAX, config.MODEL_LON_STEP)
    rows = []
    for lat in lat_values:
        for lon in lon_values:
            rows.append(
                {
                    "cell_id": cell_id_for(lat, lon),
                    "lat_cell": lat,
                    "lon_cell": lon,
                    "cell_area_km2": exposure.cell_area_km2(lat, lon),
                }
            )
    return pd.DataFrame(rows)


def cell_id_for(lat: float, lon: float) -> str:
    """Stable identifier for one model-grid cell."""
    return f"lat{float(lat):.3f}_lon{float(lon):.3f}"


def cells_from_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the unique cell grid actually present in the fire panel."""
    missing = [column for column in ("lat_cell", "lon_cell") if column not in frame.columns]
    if missing:
        raise BuildError(f"Panel is missing coordinate column(s): {missing}.")
    coordinates = frame[["lat_cell", "lon_cell"]].copy()
    coordinates["lat_cell"] = pd.to_numeric(coordinates["lat_cell"], errors="coerce")
    coordinates["lon_cell"] = pd.to_numeric(coordinates["lon_cell"], errors="coerce")
    coordinates = coordinates.dropna().drop_duplicates()
    if coordinates.empty:
        raise BuildError("Panel contains no valid cell coordinates.")
    coordinates = coordinates.sort_values(["lat_cell", "lon_cell"]).reset_index(drop=True)
    bad = [
        (float(row.lat_cell), float(row.lon_cell))
        for row in coordinates.itertuples()
        if not exposure.is_on_model_grid(row.lat_cell, row.lon_cell)
    ]
    if bad:
        raise BuildError(f"Panel contains off-grid cell(s): {bad[:5]}.")
    coordinates["cell_id"] = [
        cell_id_for(lat, lon)
        for lat, lon in zip(coordinates["lat_cell"], coordinates["lon_cell"])
    ]
    coordinates["cell_area_km2"] = [
        exposure.cell_area_km2(lat, lon)
        for lat, lon in zip(coordinates["lat_cell"], coordinates["lon_cell"])
    ]
    return coordinates[
        ["cell_id", "lat_cell", "lon_cell", "cell_area_km2"]
    ].reset_index(drop=True)


def cells_for_build(panel_path: Path | None = None, use_panel: bool = True) -> pd.DataFrame:
    """Use the panel's 209 cells when available, otherwise the full grid."""
    if use_panel:
        path = Path(panel_path) if panel_path is not None else config.FIRE_PANEL_PATH
        if path.exists():
            panel = pd.read_parquet(path, columns=["lat_cell", "lon_cell"])
            return cells_from_panel(panel)
    return grid_cells()


def haversine_km(
    lat1: Any,
    lon1: Any,
    lat2: Any,
    lon2: Any,
) -> Any:
    """Great-circle distance in kilometres (vectorised)."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * config.EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _snap_coordinate(values: Any, step: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.floor(array / step + 0.5) * step


def nearest_cell_coordinates(lat: Any, lon: Any) -> tuple[Any, Any]:
    """Snap arbitrary coordinates to the nearest model-cell centre."""
    lat_cell = np.round(_snap_coordinate(lat, config.MODEL_LAT_STEP), 10)
    lon_cell = np.round(_snap_coordinate(lon, config.MODEL_LON_STEP), 10)
    return lat_cell, lon_cell


def assign_points_to_cells(
    points: pd.DataFrame,
    cells: pd.DataFrame,
    lat_column: str = "lat",
    lon_column: str = "lon",
) -> pd.DataFrame:
    """Attach ``lat_cell``/``lon_cell``/``cell_id`` to arbitrary points.

    Points outside the supplied cell list receive a null ``cell_id``; they are
    not snapped onto a nearby cell and are never counted as inside.
    """
    missing = [column for column in (lat_column, lon_column) if column not in points.columns]
    if missing:
        raise BuildError(f"Point table is missing column(s): {missing}.")
    out = points.copy()
    lat_cell, lon_cell = nearest_cell_coordinates(out[lat_column], out[lon_column])
    out["lat_cell"] = lat_cell
    out["lon_cell"] = lon_cell
    lookup = cells[["cell_id", "lat_cell", "lon_cell"]].drop_duplicates()
    return out.merge(lookup, on=["lat_cell", "lon_cell"], how="left", validate="many_to_one")


def points_within_radius_sum(
    points: pd.DataFrame,
    cells: pd.DataFrame,
    radius_km: float,
    value_column: str = "value",
) -> pd.DataFrame:
    """Sum point values within ``radius_km`` of every cell centre.

    Pure helper used by tests.  Ring values overlap between neighbouring cells
    and must never be added across cells.
    """
    missing = [
        column
        for column in ("lat", "lon", value_column)
        if column not in points.columns
    ]
    if missing:
        raise BuildError(f"Point table is missing column(s): {missing}.")
    if radius_km <= 0:
        raise BuildError("radius_km must be strictly positive.")
    values = pd.to_numeric(points[value_column], errors="coerce").to_numpy(dtype=float)
    rows = []
    for cell in cells.itertuples():
        distances = haversine_km(
            cell.lat_cell, cell.lon_cell, points["lat"], points["lon"]
        )
        inside = (distances <= radius_km) & np.isfinite(values)
        rows.append(
            {
                "lat_cell": float(cell.lat_cell),
                "lon_cell": float(cell.lon_cell),
                "ring_km": float(radius_km),
                "value": float(values[inside].sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Land-cover helpers
# ---------------------------------------------------------------------------
def worldcover_lc_key(code: int) -> str:
    """Map a WorldCover class code to a canonical land-cover key."""
    try:
        return WORLDCOVER_TO_LC[int(code)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unknown WorldCover class code: {code!r}.") from exc


def landcover_areas_from_counts(
    counts: Mapping[int, float],
    pixel_area_km2: float,
) -> dict[str, float]:
    """Convert per-class pixel counts into canonical land-cover areas."""
    pixel_area = float(pixel_area_km2)
    if not np.isfinite(pixel_area) or pixel_area <= 0:
        raise ValueError("pixel_area_km2 must be finite and strictly positive.")
    areas = {key: 0.0 for key in config.LC_AREA_COLUMNS}
    for raw_code, raw_count in counts.items():
        count = float(raw_count)
        if not np.isfinite(count) or count < 0:
            raise ValueError(f"Invalid pixel count {count!r} for class {raw_code!r}.")
        code = int(raw_code)
        if code == 0:
            continue  # nodata
        if code not in WORLDCOVER_TO_LC:
            raise ValueError(f"Unknown WorldCover class code: {code}.")
        areas[WORLDCOVER_TO_LC[code]] += count * pixel_area
    return areas


def landcover_frame_from_class_areas(class_areas: pd.DataFrame) -> pd.DataFrame:
    """Fold a wide per-cell class-area table into canonical LC columns.

    ``class_areas`` must contain ``cell_id`` and one column per WorldCover
    class code (integers or strings).  Unknown nonzero codes raise instead of
    being silently dropped.
    """
    if "cell_id" not in class_areas.columns:
        raise ValueError("class_areas must contain a 'cell_id' column.")
    out = pd.DataFrame({"cell_id": class_areas["cell_id"].to_numpy()})
    totals = {key: np.zeros(len(class_areas), dtype=float) for key in config.LC_AREA_COLUMNS}
    for column in class_areas.columns:
        if column == "cell_id":
            continue
        code = int(column)
        if code == 0:
            continue
        key = worldcover_lc_key(code)
        values = pd.to_numeric(class_areas[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid area values for WorldCover class {code}.")
        totals[key] += values
    for key, column in config.LC_AREA_COLUMNS.items():
        out[column] = totals[key]
    return out


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checksums(text: str) -> dict[str, str]:
    """Parse either a JSON checksum map or ``sha256sum`` text lines."""
    stripped = text.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ChecksumError("checksums.json must contain a JSON object.")
        result: dict[str, str] = {}
        for key, value in payload.items():
            if not isinstance(value, str) or len(value) != 64:
                raise ChecksumError(f"Invalid SHA-256 value for {key!r}.")
            result[str(key)] = value.lower()
        return result
    result = {}
    for line in stripped.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace("*", " ").split()
        if len(parts) < 2 or len(parts[0]) != 64:
            raise ChecksumError(f"Could not parse checksum line: {line!r}")
        result[" ".join(parts[1:])] = parts[0].lower()
    return result


def load_checksums(path: Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        raise ChecksumError(
            f"Checksum file {path} does not exist. Run "
            "`python scripts/build_impact_layers.py --record-checksums` once "
            "after downloading the raw inputs, then commit checksums.json."
        )
    return parse_checksums(path.read_text(encoding="utf-8"))


def _checksum_key(raw_dir: Path, path: Path, checksums: Mapping[str, str]) -> str | None:
    candidates: list[str] = []
    try:
        candidates.append(path.resolve().relative_to(raw_dir.resolve()).as_posix())
    except ValueError:
        pass
    candidates.append(path.name)
    for candidate in candidates:
        if candidate in checksums:
            return candidate
    lowered = {key.lower(): key for key in checksums}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def verify_checksums(
    raw_dir: Path,
    files: Iterable[Path],
    checksums: Mapping[str, str],
) -> dict[str, str]:
    """Verify every file and return ``relative path -> sha256``.

    Raises :class:`ChecksumError` with every problem listed, so the operator
    sees all missing entries and mismatches at once.
    """
    raw_dir = Path(raw_dir)
    verified: dict[str, str] = {}
    problems: list[str] = []
    for path in files:
        path = Path(path)
        key = _checksum_key(raw_dir, path, checksums)
        if key is None:
            problems.append(f"no checksum recorded for {path.name}")
            continue
        actual = sha256_file(path)
        if actual != checksums[key]:
            problems.append(
                f"checksum mismatch for {path.name}: expected {checksums[key]}, got {actual}"
            )
            continue
        try:
            relative = path.resolve().relative_to(raw_dir.resolve()).as_posix()
        except ValueError:
            relative = path.name
        verified[relative] = actual
    if problems:
        raise ChecksumError(
            "Raw input verification failed:\n  - " + "\n  - ".join(problems)
            + "\nRe-download the affected files or re-record checksums deliberately."
        )
    return verified


def write_checksums(raw_dir: Path, files: Iterable[Path], path: Path) -> dict[str, str]:
    raw_dir = Path(raw_dir)
    payload: dict[str, str] = {}
    for file in sorted(Path(item) for item in files):
        try:
            relative = file.resolve().relative_to(raw_dir.resolve()).as_posix()
        except ValueError:
            relative = file.name
        payload[relative] = sha256_file(file)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------
# Raw source resolution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedSources:
    """Paths to every raw input the build can use."""

    raw_dir: Path
    worldcover: tuple[Path, ...]
    worldpop: Path
    osm_roads: Path
    osm_buildings: Path
    osm_pois: Path
    osm_places: Path
    admin_adm1: Path | None = None
    admin_adm2: Path | None = None
    ghsl: Path | None = None

    def required_files(self) -> tuple[Path, ...]:
        return (
            *self.worldcover,
            self.worldpop,
            self.osm_roads,
            self.osm_buildings,
            self.osm_pois,
            self.osm_places,
        )

    def all_files(self) -> tuple[Path, ...]:
        optional = tuple(
            path
            for path in (self.admin_adm1, self.admin_adm2, self.ghsl)
            if path is not None
        )
        return (*self.required_files(), *optional)


def _find_file(raw_dir: Path, filename: str) -> Path | None:
    direct = raw_dir / filename
    if direct.is_file():
        return direct
    matches = sorted(path for path in raw_dir.rglob(filename) if path.is_file())
    return matches[0] if matches else None


def _find_optional(raw_dir: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        direct = raw_dir / pattern
        if direct.is_file():
            return direct
        matches = sorted(path for path in raw_dir.rglob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def resolve_required_sources(raw_dir: Path) -> ResolvedSources:
    """Locate required raw inputs or fail with one complete missing list."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise MissingSourceError(
            ["data/impact/raw/ (directory does not exist)"], raw_dir
        )

    missing: list[str] = []
    worldcover: list[Path] = []
    for name in expected_worldcover_files():
        found = _find_file(raw_dir, name)
        if found is None:
            missing.append(name)
        else:
            worldcover.append(found)

    worldpop_candidates = (
        WORLDPOP_FILENAME,
        "dza_ppp_2020*constrained*.tif",
    )
    worldpop = _find_optional(raw_dir, worldpop_candidates)
    if worldpop is None:
        missing.append(WORLDPOP_FILENAME)

    osm_paths: dict[str, Path] = {}
    for name in OSM_REQUIRED:
        found = _find_file(raw_dir, name)
        if found is None:
            missing.append(f"{OSM_FOLDER}/{name}")
        else:
            osm_paths[name] = found

    if missing:
        raise MissingSourceError(missing, raw_dir)

    return ResolvedSources(
        raw_dir=raw_dir,
        worldcover=tuple(sorted(worldcover)),
        worldpop=worldpop,
        osm_roads=osm_paths["gis_osm_roads_free_1.shp"],
        osm_buildings=osm_paths["gis_osm_buildings_a_free_1.shp"],
        osm_pois=osm_paths["gis_osm_pois_free_1.shp"],
        osm_places=osm_paths["gis_osm_places_free_1.shp"],
        admin_adm1=_find_optional(raw_dir, ("dza_adm1.shp", "*adm1*.shp")),
        admin_adm2=_find_optional(raw_dir, ("dza_adm2.shp", "*adm2*.shp")),
        ghsl=_find_optional(
            raw_dir, ("GHSL_BUILT_S*.tif", "GHS_BUILT_S*.tif", "*GHSL*BUILT*.tif")
        ),
    )


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------
def classify_road_class(value: object) -> str:
    """Normalize an OSM road class into primary/secondary/tertiary/track/other."""
    text = "" if value is None else str(value).strip().lower()
    if text in ROAD_PRIMARY:
        return "primary"
    if text in ROAD_SECONDARY:
        return "secondary"
    if text in ROAD_TERTIARY:
        return "tertiary"
    if text in ROAD_TRACK:
        return "track"
    return "other"


def classify_critical_facility(value: object) -> str | None:
    """Map an OSM POI class to a critical-facility category or ``None``."""
    text = "" if value is None else str(value).strip().lower()
    if text in {"hospital", "clinic", "doctors", "health_centre", "health_center"}:
        return "hospital"
    if text in {"school", "university", "college", "kindergarten", "college_university"}:
        return "school"
    if text in {
        "power_plant",
        "power_substation",
        "substation",
        "generator",
        "power_station",
    }:
        return "power"
    if text in {"airport", "aerodrome", "helipad", "airfield"}:
        return "airport"
    if text in {
        "water_treatment_plant",
        "water_works",
        "reservoir",
        "wastewater_plant",
        "water_tower",
    }:
        return "water"
    if text in {
        "fire_station",
        "police",
        "ambulance_station",
        "ranger_station",
        "rescue_station",
    }:
        return "emergency"
    return None


# ---------------------------------------------------------------------------
# Frozen severity thresholds and manifest
# ---------------------------------------------------------------------------
def freeze_tier_thresholds(
    values: Sequence[float] | pd.Series,
    quantiles: Sequence[float] = (0.20, 0.40, 0.60, 0.80),
) -> tuple[float, ...]:
    """Freeze four IEI thresholds from the study-area distribution.

    Fixed quantiles make the five tiers balanced by construction.  Degenerate
    distributions (for example many identical IEI values) are nudged into a
    strictly increasing sequence inside (0, 1) rather than producing invalid
    thresholds.
    """
    quantiles = tuple(float(value) for value in quantiles)
    if len(quantiles) != len(severity.TIER_LABELS) - 1:
        raise ValueError("Exactly four quantiles are required.")
    if any(not (0.0 < value < 1.0) for value in quantiles):
        raise ValueError("Quantiles must lie strictly between 0 and 1.")
    if list(quantiles) != sorted(quantiles) or len(set(quantiles)) != len(quantiles):
        raise ValueError("Quantiles must be strictly increasing.")

    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        raise ValueError("Cannot freeze thresholds from an empty IEI distribution.")
    numeric = np.clip(numeric, 0.0, 1.0)
    raw = np.quantile(numeric, quantiles)

    epsilon = 1e-6
    frozen: list[float] = []
    for value in raw:
        candidate = min(max(float(value), epsilon), 1.0 - epsilon)
        if frozen and candidate <= frozen[-1]:
            candidate = frozen[-1] + epsilon
        if candidate >= 1.0:
            raise ValueError(
                "Cannot freeze five distinct tiers: the IEI distribution is "
                "too concentrated near 1."
            )
        frozen.append(round(candidate, 8))
    if len(set(frozen)) != len(frozen):
        raise ValueError("Frozen thresholds are not distinct.")
    return tuple(frozen)


def _combined_checksum(file_checksums: Mapping[str, str]) -> str:
    text = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(file_checksums.items())
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_record(
    key: str,
    raw_dir: Path,
    files: Iterable[Path],
    verified_checksums: Mapping[str, str],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one manifest source entry with per-file and combined checksums."""
    metadata = SOURCE_METADATA[key]
    file_hashes: dict[str, str] = {}
    names: list[str] = []
    for path in sorted(Path(item) for item in files):
        try:
            relative = path.resolve().relative_to(Path(raw_dir).resolve()).as_posix()
        except ValueError:
            relative = path.name
        digest = verified_checksums.get(relative)
        if digest is None:
            digest = sha256_file(path)
        file_hashes[path.name] = digest
        names.append(relative)
    return {
        "name": metadata["name"],
        "version": metadata["version"],
        "license": metadata["license"],
        "attribution": metadata["attribution"],
        "role": metadata["role"],
        "coverage": dict(coverage),
        "files": names,
        "checksums": file_hashes,
        "checksum": _combined_checksum(file_hashes),
    }


def build_manifest(
    sources: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    artifacts: Mapping[str, Mapping[str, Any]],
    study_area: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the JSON-serialisable provenance manifest."""
    frozen = tuple(float(value) for value in thresholds)
    if len(frozen) != 4:
        raise ValueError("Exactly four severity thresholds are required.")
    if list(frozen) != sorted(frozen) or len(set(frozen)) != len(frozen):
        raise ValueError("Severity thresholds must be strictly increasing.")
    if any(not (0.0 < value < 1.0) for value in frozen):
        raise ValueError("Severity thresholds must lie strictly between 0 and 1.")

    timestamp = generated_at or (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": timestamp,
        "generator": "scripts/build_impact_layers.py",
        "study_area": dict(study_area),
        "grid": {
            "lat_min": config.MODEL_LAT_MIN,
            "lat_max": config.MODEL_LAT_MAX,
            "lon_min": config.MODEL_LON_MIN,
            "lon_max": config.MODEL_LON_MAX,
            "lat_step": config.MODEL_LAT_STEP,
            "lon_step": config.MODEL_LON_STEP,
        },
        "sources": list(sources),
        "severity": {
            "tier_labels": list(severity.TIER_LABELS),
            "tier_thresholds": list(frozen),
            "method": (
                "Fixed quantiles (0.20, 0.40, 0.60, 0.80) of the study-area "
                "Impact Exposure Index distribution; frozen at build time."
            ),
            "quantiles": [0.20, 0.40, 0.60, 0.80],
            "iei_weights": dict(severity.IEI_WEIGHTS),
            "infra_subweights": dict(severity.INFRA_SUBWEIGHTS),
            "note": (
                "Thresholds describe this study area only. They are not "
                "validated damage probabilities."
            ),
        },
        "artifacts": {key: dict(value) for key, value in artifacts.items()},
        "phase_2": {
            "burned_area": (
                "Not built. Confirmed burned area must come from an independent "
                "product (MCD64A1/VNP64A1, ESA CCI Fire or Copernicus EMS)."
            )
        },
    }


# ---------------------------------------------------------------------------
# Lazy geospatial dependencies
# ---------------------------------------------------------------------------
def _geopandas():
    try:
        import geopandas as gpd  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise BuildDependencyError(
            "geopandas is required for the offline build. Install the build-only "
            "stack with `python -m pip install -r requirements-impact.txt`."
        ) from exc
    return gpd


def _rasterio():
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise BuildDependencyError(
            "rasterio is required for the offline build. Install the build-only "
            "stack with `python -m pip install -r requirements-impact.txt`."
        ) from exc
    return rasterio


def _geod():
    try:
        from pyproj import Geod  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise BuildDependencyError(
            "pyproj is required for the offline build. Install the build-only "
            "stack with `python -m pip install -r requirements-impact.txt`."
        ) from exc
    return Geod(ellps="WGS84")


def _iter_windows(dataset, block_size: int):
    rasterio = _rasterio()
    from rasterio.windows import Window  # noqa: PLC0415

    block_size = max(int(block_size), 256)
    for row in range(0, dataset.height, block_size):
        for column in range(0, dataset.width, block_size):
            yield Window(
                column,
                row,
                min(block_size, dataset.width - column),
                min(block_size, dataset.height - row),
            )


def _nearest_grid_positions(values: Any, grid_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest position in a sorted grid plus an in-range mask."""
    values = np.asarray(values, dtype=float)
    grid_values = np.asarray(grid_values, dtype=float)
    positions = np.searchsorted(grid_values, values)
    positions = np.clip(positions, 1, max(len(grid_values) - 1, 1))
    left = grid_values[positions - 1]
    right = grid_values[np.minimum(positions, len(grid_values) - 1)]
    nearest = np.where(np.abs(values - left) <= np.abs(values - right), positions - 1, positions)
    nearest = np.clip(nearest, 0, len(grid_values) - 1)
    spacing = np.diff(grid_values)
    tolerance = (float(np.median(spacing)) if spacing.size else 0.5) * 0.51
    within = np.abs(grid_values[nearest] - values) <= tolerance
    return nearest, within


def _block_cell_indices(dataset, window, cells: pd.DataFrame):
    """Return 2-D cell index (-1 outside) and pixel-area arrays for a window."""
    lat_values = np.array(sorted(cells["lat_cell"].unique()), dtype=float)
    lon_values = np.array(sorted(cells["lon_cell"].unique()), dtype=float)
    cell_index = np.full((len(lat_values), len(lon_values)), -1, dtype=int)
    for index, row in cells.reset_index(drop=True).iterrows():
        lat_pos = int(np.argmin(np.abs(lat_values - row["lat_cell"])))
        lon_pos = int(np.argmin(np.abs(lon_values - row["lon_cell"])))
        cell_index[lat_pos, lon_pos] = index

    transform = dataset.transform
    rows = np.arange(window.row_off, window.row_off + window.height)
    columns = np.arange(window.col_off, window.col_off + window.width)
    lat_centres = transform.f + (rows + 0.5) * transform.e
    lon_centres = transform.c + (columns + 0.5) * transform.a

    lat_idx, lat_ok = _nearest_grid_positions(lat_centres, lat_values)
    lon_idx, lon_ok = _nearest_grid_positions(lon_centres, lon_values)

    indices = cell_index[lat_idx[:, None], lon_idx[None, :]]
    indices = np.where(lat_ok[:, None] & lon_ok[None, :], indices, -1)

    lat_top = transform.f + rows * transform.e
    lat_bottom = lat_top + transform.e
    row_area = (
        config.EARTH_RADIUS_KM**2
        * abs(math.radians(transform.a))
        * (np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bottom)))
    )
    areas = np.broadcast_to(row_area[:, None], indices.shape)
    return indices, areas


def _read_vector(path: Path):
    gpd = _geopandas()
    frame = gpd.read_file(path)
    if frame.crs is None:
        frame = frame.set_crs(4326)
    elif frame.crs.to_epsg() != 4326:
        frame = frame.to_crs(4326)
    return frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()


def _first_existing(frame, names: Sequence[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _representative_points(frame) -> pd.DataFrame:
    points = frame.geometry.representative_point()
    return pd.DataFrame(
        {"lat": points.y.to_numpy(), "lon": points.x.to_numpy()},
        index=frame.index,
    )


# ---------------------------------------------------------------------------
# Raster readers
# ---------------------------------------------------------------------------
def _require_epsg4326(dataset, label: str | None = None) -> None:
    """Raise unless the raster is in plain WGS84 longitude/latitude."""
    name = label or getattr(dataset, "name", "raster")
    if dataset.crs is None or dataset.crs.to_epsg() != 4326:
        raise BuildError(
            f"{Path(name).name} is not in EPSG:4326; the impact build expects "
            "the official WGS84 longitude/latitude product."
        )


def read_worldcover_areas(
    files: Sequence[Path],
    cells: pd.DataFrame,
    block_size: int = 2048,
) -> pd.DataFrame:
    """Aggregate WorldCover class areas (km²) per model cell."""
    rasterio = _rasterio()
    codes = tuple(sorted(WORLDCOVER_TO_LC))
    code_to_column = {code: index for index, code in enumerate(codes)}
    areas = np.zeros((len(cells), len(codes)), dtype=float)

    for path in files:
        with rasterio.open(path) as dataset:
            _require_epsg4326(dataset, path.name)
            for window in _iter_windows(dataset, block_size):
                band = dataset.read(1, window=window, masked=True)
                indices, pixel_areas = _block_cell_indices(dataset, window, cells)
                data = np.asarray(band.data)
                valid = (
                    ~np.ma.getmaskarray(band)
                    & np.isfinite(data)
                    & (indices >= 0)
                )
                values = data.astype(np.int64, copy=False)
                known = np.isin(values, codes)
                unknown = np.unique(values[valid & ~known & (values != 0)])
                if unknown.size:
                    raise BuildError(
                        f"{path.name} contains unknown WorldCover class code(s): "
                        f"{unknown.tolist()}."
                    )
                flat_cells = indices[valid & known]
                flat_codes = values[valid & known]
                flat_areas = pixel_areas[valid & known]
                for code in np.unique(flat_codes):
                    selector = flat_codes == code
                    np.add.at(
                        areas[:, code_to_column[int(code)]],
                        flat_cells[selector],
                        flat_areas[selector],
                    )

    class_frame = pd.DataFrame(areas, columns=list(codes))
    class_frame.insert(0, "cell_id", cells["cell_id"].to_numpy())
    return landcover_frame_from_class_areas(class_frame)


def read_worldpop_cell_totals(
    path: Path,
    cells: pd.DataFrame,
    block_size: int = 2048,
) -> pd.DataFrame:
    """Sum WorldPop population inside each model cell."""
    rasterio = _rasterio()
    totals = np.zeros(len(cells), dtype=float)
    valid_pixels = np.zeros(len(cells), dtype=np.int64)

    with rasterio.open(path) as dataset:
        _require_epsg4326(dataset, Path(path).name)
        for window in _iter_windows(dataset, block_size):
            band = dataset.read(1, window=window, masked=True)
            indices, _ = _block_cell_indices(dataset, window, cells)
            data = np.asarray(band.data, dtype=float)
            valid = (
                ~np.ma.getmaskarray(band)
                & np.isfinite(data)
                & (data >= 0)
                & (indices >= 0)
            )
            flat_cells = indices[valid]
            flat_values = data[valid]
            np.add.at(totals, flat_cells, flat_values)
            np.add.at(valid_pixels, flat_cells, 1)

    result = pd.DataFrame(
        {
            "cell_id": cells["cell_id"].to_numpy(),
            "lat_cell": cells["lat_cell"].to_numpy(),
            "lon_cell": cells["lon_cell"].to_numpy(),
            "pop_cell": totals,
        }
    )
    # A cell with no valid raster pixel is explicitly missing, never zero.
    result.loc[valid_pixels == 0, "pop_cell"] = np.nan
    return result


def read_worldpop_ring_totals(
    path: Path,
    cells: pd.DataFrame,
    distances: Sequence[int] = config.RING_DISTANCES_KM,
) -> pd.DataFrame:
    """Sum WorldPop population inside 5/10/25 km rings around each cell."""
    rasterio = _rasterio()
    from rasterio.windows import Window  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    with rasterio.open(path) as dataset:
        _require_epsg4326(dataset, Path(path).name)
        for cell in cells.itertuples():
            for radius in distances:
                lat_margin = radius / 111.0
                lon_margin = radius / (111.0 * max(math.cos(math.radians(cell.lat_cell)), 0.05))
                row_min, col_min = rasterio.transform.rowcol(
                    dataset.transform,
                    cell.lon_cell - lon_margin,
                    cell.lat_cell + lat_margin,
                )
                row_max, col_max = rasterio.transform.rowcol(
                    dataset.transform,
                    cell.lon_cell + lon_margin,
                    cell.lat_cell - lat_margin,
                )
                row_min = max(int(row_min) - 1, 0)
                col_min = max(int(col_min) - 1, 0)
                row_max = min(int(row_max) + 1, dataset.height - 1)
                col_max = min(int(col_max) + 1, dataset.width - 1)
                if row_max < row_min or col_max < col_min:
                    rows.append(
                        {
                            "lat_cell": cell.lat_cell,
                            "lon_cell": cell.lon_cell,
                            "ring_km": float(radius),
                            "population": np.nan,
                        }
                    )
                    continue
                window = Window(
                    col_min, row_min, col_max - col_min + 1, row_max - row_min + 1
                )
                band = dataset.read(1, window=window, masked=True)
                data = np.asarray(band.data, dtype=float)
                pixel_rows = np.arange(row_min, row_max + 1)
                pixel_cols = np.arange(col_min, col_max + 1)
                lats = dataset.transform.f + (pixel_rows + 0.5) * dataset.transform.e
                lons = dataset.transform.c + (pixel_cols + 0.5) * dataset.transform.a
                distances_km = haversine_km(
                    cell.lat_cell, cell.lon_cell, lats[:, None], lons[None, :]
                )
                valid = (
                    ~np.ma.getmaskarray(band)
                    & np.isfinite(data)
                    & (data >= 0)
                    & (distances_km <= radius)
                )
                rows.append(
                    {
                        "lat_cell": cell.lat_cell,
                        "lon_cell": cell.lon_cell,
                        "ring_km": float(radius),
                        "population": (
                            float(data[valid].sum()) if valid.any() else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def read_ghsl_built_surface(path: Path, cells: pd.DataFrame) -> pd.DataFrame:
    """Sum GHSL built-up surface (m²/pixel) into km² per cell."""
    rasterio = _rasterio()
    totals = np.zeros(len(cells), dtype=float)
    valid_pixels = np.zeros(len(cells), dtype=np.int64)
    with rasterio.open(path) as dataset:
        _require_epsg4326(dataset, Path(path).name)
        for window in _iter_windows(dataset, 2048):
            band = dataset.read(1, window=window, masked=True)
            indices, _ = _block_cell_indices(dataset, window, cells)
            data = np.asarray(band.data, dtype=float)
            valid = (
                ~np.ma.getmaskarray(band)
                & np.isfinite(data)
                & (data >= 0)
                & (indices >= 0)
            )
            np.add.at(totals, indices[valid], data[valid] / 1_000_000.0)
            np.add.at(valid_pixels, indices[valid], 1)
    result = pd.DataFrame(
        {
            "cell_id": cells["cell_id"].to_numpy(),
            "built_surface_km2": totals,
        }
    )
    result.loc[valid_pixels == 0, "built_surface_km2"] = np.nan
    return result


# ---------------------------------------------------------------------------
# Vector readers
# ---------------------------------------------------------------------------
def read_osm_roads(path: Path, cells: pd.DataFrame) -> pd.DataFrame:
    """Road length (km) per cell and per class."""
    frame = _read_vector(path)
    class_column = _first_existing(frame, ("fclass", "type", "class", "highway"))
    points = _representative_points(frame)
    points["fclass"] = (
        frame[class_column].astype(str).to_numpy()
        if class_column
        else "other"
    )
    geod = _geod()
    points["length_km"] = [
        abs(geod.geometry_length(geometry)) / 1000.0
        for geometry in frame.geometry.to_numpy()
    ]
    assigned = assign_points_to_cells(points, cells).dropna(subset=["cell_id"])
    assigned["road_class"] = assigned["fclass"].map(classify_road_class)

    result = cells[["cell_id"]].copy()
    result["road_km_total"] = 0.0
    for column in ROAD_CLASS_COLUMNS.values():
        result[column] = 0.0
    grouped = assigned.groupby(["cell_id", "road_class"])["length_km"].sum()
    for (cell, road_class), value in grouped.items():
        mask = result["cell_id"] == cell
        result.loc[mask, "road_km_total"] += value
        column = ROAD_CLASS_COLUMNS.get(road_class)
        if column:
            result.loc[mask, column] += value
    return result


def read_osm_buildings(path: Path, cells: pd.DataFrame) -> pd.DataFrame:
    """Building count and footprint area (km²) per cell."""
    frame = _read_vector(path)
    points = _representative_points(frame)
    geod = _geod()
    points["footprint_km2"] = [
        abs(geod.geometry_area_perimeter(geometry)[0]) / 1_000_000.0
        for geometry in frame.geometry.to_numpy()
    ]
    assigned = assign_points_to_cells(points, cells).dropna(subset=["cell_id"])

    result = cells[["cell_id"]].copy()
    result["building_count"] = 0.0
    result["building_footprint_km2"] = 0.0
    grouped = assigned.groupby("cell_id").agg(
        building_count=("footprint_km2", "size"),
        building_footprint_km2=("footprint_km2", "sum"),
    )
    result = result.merge(grouped, on="cell_id", how="left")
    result[["building_count", "building_footprint_km2"]] = result[
        ["building_count", "building_footprint_km2"]
    ].fillna(0.0)
    return result


def read_osm_pois(
    path: Path, cells: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Critical-facility points and per-cell counts."""
    frame = _read_vector(path)
    class_column = _first_existing(frame, ("fclass", "type", "class", "amenity"))
    name_column = _first_existing(frame, ("name", "name_en"))
    categories = (
        frame[class_column].map(classify_critical_facility)
        if class_column
        else pd.Series([None] * len(frame), index=frame.index)
    )
    keep = categories.notna()
    points = _representative_points(frame.loc[keep])
    points["category"] = categories.loc[keep].to_numpy()
    points["asset_name"] = (
        frame.loc[keep, name_column].fillna("").astype(str).to_numpy()
        if name_column
        else [""] * int(keep.sum())
    )
    assigned = assign_points_to_cells(points, cells).dropna(subset=["cell_id"])
    assets = assigned[
        ["lat_cell", "lon_cell", "category", "asset_name", "cell_id"]
    ].copy()

    result = cells[["cell_id"]].copy()
    result["critical_count_total"] = 0.0
    for column in CRITICAL_CLASS_COLUMNS.values():
        result[column] = 0.0
    if not assets.empty:
        grouped = assets.groupby(["cell_id", "category"]).size()
        for (cell, category), count in grouped.items():
            mask = result["cell_id"] == cell
            result.loc[mask, "critical_count_total"] += float(count)
            column = CRITICAL_CLASS_COLUMNS.get(str(category))
            if column:
                result.loc[mask, column] += float(count)
    return assets, result


def _sample_raster(path: Path, lats: Sequence[float], lons: Sequence[float]) -> np.ndarray:
    rasterio = _rasterio()
    values: list[float] = []
    with rasterio.open(path) as dataset:
        _require_epsg4326(dataset, Path(path).name)
        for sample in dataset.sample(zip(lons, lats), masked=True):
            value = float(np.ma.filled(sample[0], np.nan))
            values.append(value if np.isfinite(value) and value >= 0 else np.nan)
    return np.array(values, dtype=float)


def summarize_settlements(
    settlements: pd.DataFrame | None,
    cells: pd.DataFrame,
) -> pd.DataFrame:
    """Per-cell settlement counts and summed population.

    ``settlement_count`` is a known zero when the places source is present but
    no settlement falls in a cell.  ``settlement_population`` stays NaN when
    every sampled population in the cell is missing: an absent WorldPop sample
    is not "zero inhabitants".
    """
    result = cells[["cell_id"]].copy()
    result["settlement_count"] = 0.0
    result["settlement_population"] = np.nan
    if settlements is None or settlements.empty:
        return result
    grouped = settlements.groupby("cell_id").agg(
        settlement_count=("settlement_name", "size"),
        settlement_population=("population", "sum"),
        settlement_population_known=("population", lambda values: values.notna().sum()),
    )
    grouped.loc[
        grouped["settlement_population_known"] == 0, "settlement_population"
    ] = np.nan
    grouped = grouped.drop(columns=["settlement_population_known"])
    result = result.drop(
        columns=["settlement_count", "settlement_population"]
    ).merge(grouped, on="cell_id", how="left")
    result["settlement_count"] = result["settlement_count"].fillna(0.0)
    return result


def read_osm_settlements(
    places_path: Path,
    worldpop_path: Path,
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Settlement points with sampled population and per-cell totals."""
    frame = _read_vector(places_path)
    place_column = _first_existing(frame, ("fclass", "type", "class", "place"))
    name_column = _first_existing(frame, ("name", "name_en"))
    if place_column:
        wanted = {"city", "town", "village", "hamlet", "suburb", "locality"}
        frame = frame[frame[place_column].astype(str).str.lower().isin(wanted)]
    frame = frame.copy()
    points = _representative_points(frame)
    points["settlement_name"] = (
        frame[name_column].fillna("").astype(str).to_numpy()
        if name_column
        else [""] * len(frame)
    )
    assigned = assign_points_to_cells(points, cells).dropna(subset=["cell_id"])
    assigned["population"] = _sample_raster(
        worldpop_path, assigned["lat"].to_numpy(), assigned["lon"].to_numpy()
    )
    assigned["distance_km"] = haversine_km(
        assigned["lat_cell"],
        assigned["lon_cell"],
        assigned["lat"],
        assigned["lon"],
    )
    settlements = assigned[
        [
            "lat_cell",
            "lon_cell",
            "settlement_name",
            "population",
            "distance_km",
            "cell_id",
        ]
    ].copy()

    return settlements, summarize_settlements(settlements, cells)


def read_admin_units(path: Path, cells: pd.DataFrame) -> pd.DataFrame:
    """Assign each cell centre to an administrative polygon."""
    gpd = _geopandas()
    frame = _read_vector(path)
    code_column = _first_existing(frame, ("ADM2_PCODE", "ADM1_PCODE", "admin_code", "code"))
    name_column = _first_existing(frame, ("ADM2_EN", "ADM1_EN", "admin_name", "name"))
    if code_column is None or name_column is None:
        raise BuildError(
            f"{Path(path).name} must contain an admin code and name column."
        )
    points = gpd.GeoDataFrame(
        cells[["cell_id", "lat_cell", "lon_cell"]].copy(),
        geometry=gpd.points_from_xy(cells["lon_cell"], cells["lat_cell"]),
        crs=4326,
    )
    joined = gpd.sjoin(points, frame[[code_column, name_column, "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return pd.DataFrame(
        {
            "admin_code": joined[code_column].fillna("").astype(str).to_numpy(),
            "admin_name": joined[name_column].fillna("").astype(str).to_numpy(),
            "lat_cell": joined["lat_cell"].to_numpy(),
            "lon_cell": joined["lon_cell"].to_numpy(),
        }
    )


def _cells_geoframe(cells: pd.DataFrame):
    gpd = _geopandas()
    return gpd.GeoDataFrame(
        cells[["cell_id", "lat_cell", "lon_cell"]].copy(),
        geometry=gpd.points_from_xy(cells["lon_cell"], cells["lat_cell"]),
        crs=4326,
    )


def summarize_vector_rings(
    frame,
    cells: pd.DataFrame,
    distances: Sequence[int],
    kind: str,
) -> pd.DataFrame:
    """Aggregate OSM features inside 5/10/25 km buffers around cell centres.

    ``kind`` is ``road``, ``building`` or ``critical``.  Buffers overlap
    between neighbouring cells; the result must not be summed across cells.
    """
    gpd = _geopandas()
    projected = frame.to_crs(32631)
    cell_points = _cells_geoframe(cells).to_crs(32631)
    rows: list[dict[str, Any]] = []
    for radius in distances:
        buffers = cell_points.copy()
        buffers["geometry"] = buffers.geometry.buffer(float(radius) * 1000.0)
        buffers = buffers.rename(columns={"cell_id": "buffer_cell_id"})
        joined = gpd.sjoin(
            projected,
            buffers[["buffer_cell_id", "geometry"]],
            how="inner",
            predicate="intersects",
        )
        if kind == "road":
            values = joined.geometry.length / 1000.0
            grouped = values.groupby(joined["buffer_cell_id"]).sum()
            for cell, value in grouped.items():
                rows.append(
                    {"cell_id": cell, "ring_km": float(radius), "road_km": float(value)}
                )
        elif kind == "building":
            areas = joined.geometry.area / 1_000_000.0
            grouped = pd.DataFrame(
                {"cell_id": joined["buffer_cell_id"].to_numpy(), "area": areas.to_numpy()}
            ).groupby("cell_id").agg(
                building_count=("area", "size"), building_footprint_km2=("area", "sum")
            )
            for cell, row in grouped.iterrows():
                rows.append(
                    {
                        "cell_id": cell,
                        "ring_km": float(radius),
                        "building_count": float(row.building_count),
                        "building_footprint_km2": float(row.building_footprint_km2),
                    }
                )
        elif kind == "critical":
            grouped = joined.groupby("buffer_cell_id").size()
            for cell, value in grouped.items():
                rows.append(
                    {"cell_id": cell, "ring_km": float(radius), "critical_count": float(value)}
                )
        else:  # pragma: no cover - programming error
            raise ValueError(f"Unknown ring kind: {kind!r}")
    return pd.DataFrame(rows)


def build_ring_exposure(
    pop_rings: pd.DataFrame,
    road_rings: pd.DataFrame,
    building_rings: pd.DataFrame,
    critical_rings: pd.DataFrame,
    cells: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the canonical non-additive ring table."""
    base = []
    for cell in cells.itertuples():
        for radius in config.RING_DISTANCES_KM:
            base.append(
                {
                    "cell_id": cell.cell_id,
                    "lat_cell": cell.lat_cell,
                    "lon_cell": cell.lon_cell,
                    "ring_km": float(radius),
                    "cell_area_km2": cell.cell_area_km2,
                }
            )
    ring = pd.DataFrame(base)
    if not pop_rings.empty:
        population = pop_rings.rename(columns={"value": "population"})
        ring = ring.merge(
            population[["lat_cell", "lon_cell", "ring_km", "population"]],
            on=["lat_cell", "lon_cell", "ring_km"],
            how="left",
            validate="one_to_one",
        )
    else:
        ring["population"] = np.nan
    for frame, columns in (
        (road_rings, ["road_km"]),
        (building_rings, ["building_count", "building_footprint_km2"]),
        (critical_rings, ["critical_count"]),
    ):
        if frame is None or frame.empty:
            for column in columns:
                ring[column] = 0.0
            continue
        ring = ring.merge(
            frame[["cell_id", "ring_km", *columns]],
            on=["cell_id", "ring_km"],
            how="left",
            validate="one_to_one",
        )
    for column in (
        "road_km",
        "building_count",
        "building_footprint_km2",
        "critical_count",
    ):
        if column in ring.columns:
            ring[column] = ring[column].fillna(0.0)
    ordered = [
        "cell_id",
        "lat_cell",
        "lon_cell",
        "ring_km",
        "cell_area_km2",
        "population",
        "road_km",
        "building_count",
        "building_footprint_km2",
        "critical_count",
    ]
    for column in ordered:
        if column not in ring.columns:
            ring[column] = np.nan
    return ring[ordered].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def assemble_cell_exposure(
    cells: pd.DataFrame,
    landcover: pd.DataFrame,
    population: pd.DataFrame,
    infrastructure: pd.DataFrame,
    settlements: pd.DataFrame | None = None,
    ghsl: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join per-cell dimensions into the canonical exposure table.

    Required land-cover and population values may not be missing.  OSM counts
    may legitimately be zero when no feature is mapped in a cell, because the
    OSM snapshot itself is present.
    """
    required = [
        "cell_id",
        "lat_cell",
        "lon_cell",
        config.CELL_AREA_COLUMN,
    ]
    out = cells[required].copy()

    out = out.merge(landcover, on="cell_id", how="left", validate="one_to_one")
    lc_columns = list(config.LC_AREA_COLUMNS.values())
    missing_rows = out[lc_columns].isna().any(axis=1)
    if missing_rows.any():
        raise BuildError(
            "WorldCover did not produce values for cell(s): "
            f"{out.loc[missing_rows, 'cell_id'].head().tolist()}. "
            "Missing land-cover data is never replaced with zero."
        )

    population = population.copy()
    out = out.merge(population, on="cell_id", how="left", validate="one_to_one")
    for column in ("pop_cell", "pop_5km", "pop_10km", "pop_25km"):
        if column not in out.columns:
            raise BuildError(f"Population table is missing {column}.")
    if out["pop_cell"].isna().any():
        raise BuildError(
            "WorldPop produced no valid population for cell(s): "
            f"{out.loc[out['pop_cell'].isna(), 'cell_id'].head().tolist()}. "
            "Missing population is never replaced with zero."
        )

    out = out.merge(infrastructure, on="cell_id", how="left", validate="one_to_one")
    for column in config.INFRA_REQUIRED_COLUMNS:
        if column not in out.columns:
            raise BuildError(f"Infrastructure table is missing {column}.")
        if column != config.CELL_AREA_COLUMN:
            out[column] = out[column].fillna(0.0)

    if settlements is not None:
        out = out.merge(settlements, on="cell_id", how="left", validate="one_to_one")
        # A zero settlement count is a known zero (the places source is present
        # and no settlement was mapped).  A missing settlement population stays
        # NaN: it means WorldPop had no valid sample, not that nobody lives there.
        if "settlement_count" in out.columns:
            out["settlement_count"] = out["settlement_count"].fillna(0.0)

    if ghsl is not None:
        out = out.merge(ghsl, on="cell_id", how="left", validate="one_to_one")

    return out.reset_index(drop=True)


def build_cell_exposure(
    cells: pd.DataFrame,
    landcover: pd.DataFrame,
    pop_cell: pd.DataFrame,
    pop_rings: pd.DataFrame,
    roads: pd.DataFrame,
    buildings: pd.DataFrame,
    critical_counts: pd.DataFrame,
    settlements: pd.DataFrame,
    ghsl: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine the per-cell dimensions into the canonical frame."""
    keep = [
        column
        for column in ("cell_id", "lat_cell", "lon_cell", "pop_cell")
        if column in pop_cell.columns
    ]
    population = pop_cell[keep].copy()
    for radius in config.RING_DISTANCES_KM:
        column = f"pop_{radius}km"
        population = population.drop(columns=[column], errors="ignore")
        subset = pop_rings[pop_rings["ring_km"] == float(radius)]
        renamed = subset[
            ["lat_cell", "lon_cell", "population"]
        ].rename(columns={"population": column})
        population = population.merge(
            renamed, on=["lat_cell", "lon_cell"], how="left", validate="one_to_one"
        )
        if renamed.empty:
            population[column] = np.nan

    infrastructure = roads.merge(buildings, on="cell_id", how="outer", validate="one_to_one")
    infrastructure = infrastructure.merge(
        critical_counts, on="cell_id", how="outer", validate="one_to_one"
    )
    infrastructure = cells[["cell_id"]].merge(
        infrastructure, on="cell_id", how="left", validate="one_to_one"
    )
    for column in config.INFRA_REQUIRED_COLUMNS:
        if column == config.CELL_AREA_COLUMN:
            continue
        infrastructure[column] = infrastructure[column].fillna(0.0)

    population = population.drop(columns=["lat_cell", "lon_cell"], errors="ignore")
    return assemble_cell_exposure(
        cells,
        landcover,
        population,
        infrastructure,
        settlements=settlements,
        ghsl=ghsl,
    )


# ---------------------------------------------------------------------------
# Artifact writer
# ---------------------------------------------------------------------------
def write_parquet(frame: pd.DataFrame, path: Path, required: Iterable[str]) -> dict[str, Any]:
    """Validate required columns and write one parquet artifact."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise BuildError(f"Cannot write {path.name}: missing column(s) {missing}.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {
        "path": path.name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": sha256_file(path),
    }


# ---------------------------------------------------------------------------
# Main build flow
# ---------------------------------------------------------------------------
def _source_coverage() -> dict[str, Any]:
    return {
        "lat_min": STUDY_LAT_MIN,
        "lat_max": STUDY_LAT_MAX,
        "lon_min": STUDY_LON_MIN,
        "lon_max": STUDY_LON_MAX,
        "description": "Model-cell-centre extent, clipped to the study area",
    }


def run_build(
    raw_dir: Path,
    out_dir: Path,
    checksums_path: Path,
    record_checksums: bool = False,
    block_size: int = 2048,
    use_panel: bool = True,
    with_rings: bool = True,
) -> dict[str, Any]:
    """Run the complete offline build and return the manifest."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    sources = resolve_required_sources(raw_dir)

    if record_checksums:
        write_checksums(raw_dir, sources.all_files(), checksums_path)
    checksums = load_checksums(checksums_path)
    verified = verify_checksums(raw_dir, sources.all_files(), checksums)

    cells = cells_for_build(use_panel=use_panel)
    landcover = read_worldcover_areas(sources.worldcover, cells, block_size=block_size)
    pop_cell = read_worldpop_cell_totals(sources.worldpop, cells, block_size=block_size)
    pop_rings_long = read_worldpop_ring_totals(sources.worldpop, cells)

    roads = read_osm_roads(sources.osm_roads, cells)
    buildings = read_osm_buildings(sources.osm_buildings, cells)
    critical_assets, critical_counts = read_osm_pois(sources.osm_pois, cells)
    settlements, settlement_counts = read_osm_settlements(
        sources.osm_places, sources.worldpop, cells
    )
    ghsl = read_ghsl_built_surface(sources.ghsl, cells) if sources.ghsl else None

    cell_exposure = build_cell_exposure(
        cells,
        landcover,
        pop_cell,
        pop_rings_long,
        roads,
        buildings,
        critical_counts,
        settlement_counts,
        ghsl,
    )

    prepared = exposure.prepare_exposure(cell_exposure)
    if not prepared.available:
        raise BuildError(
            "Derived exposure failed the Stage 1 availability check: "
            f"{prepared.message}"
        )
    indexed = severity.compute_index(prepared.frame)
    thresholds = freeze_tier_thresholds(indexed["iei"])

    artifacts: dict[str, dict[str, Any]] = {}
    artifacts["cell_exposure"] = write_parquet(
        cell_exposure,
        out_dir / config.CELL_EXPOSURE_PATH.name,
        required=(
            (config.CELL_ID_COLUMN, "lat_cell", "lon_cell")
            + config.ENV_REQUIRED_COLUMNS
            + config.POP_REQUIRED_COLUMNS
            + config.INFRA_REQUIRED_COLUMNS
        ),
    )
    artifacts["settlements"] = write_parquet(
        settlements.drop(columns=["cell_id"]),
        out_dir / config.SETTLEMENTS_PATH.name,
        required=config.SETTLEMENTS_REQUIRED_COLUMNS,
    )
    artifacts["infrastructure"] = write_parquet(
        critical_assets.drop(columns=["cell_id"]),
        out_dir / config.INFRASTRUCTURE_PATH.name,
        required=config.INFRASTRUCTURE_REQUIRED_COLUMNS,
    )

    if with_rings:
        road_frame = _read_vector(sources.osm_roads)
        building_frame = _read_vector(sources.osm_buildings)
        poi_frame = _read_vector(sources.osm_pois)
        class_column = _first_existing(poi_frame, ("fclass", "type", "class", "amenity"))
        category = (
            poi_frame[class_column].map(classify_critical_facility)
            if class_column
            else pd.Series([None] * len(poi_frame), index=poi_frame.index)
        )
        poi_frame = poi_frame[category.notna()].copy()

        road_rings = summarize_vector_rings(
            road_frame, cells, config.RING_DISTANCES_KM, "road"
        )
        building_rings = summarize_vector_rings(
            building_frame, cells, config.RING_DISTANCES_KM, "building"
        )
        critical_rings = summarize_vector_rings(
            poi_frame, cells, config.RING_DISTANCES_KM, "critical"
        )
        ring_exposure = build_ring_exposure(
            pop_rings_long, road_rings, building_rings, critical_rings, cells
        )
        artifacts["ring_exposure"] = write_parquet(
            ring_exposure,
            out_dir / config.RING_EXPOSURE_PATH.name,
            required=config.RING_REQUIRED_COLUMNS,
        )

    if sources.admin_adm2 or sources.admin_adm1:
        admin_path = sources.admin_adm2 or sources.admin_adm1
        admin_units = read_admin_units(admin_path, cells)
        artifacts["admin_units"] = write_parquet(
            admin_units,
            out_dir / config.ADMIN_UNITS_PATH.name,
            required=config.ADMIN_UNITS_REQUIRED_COLUMNS,
        )

    manifest_sources = [
        source_record(
            "worldcover", raw_dir, sources.worldcover, verified, _source_coverage()
        ),
        source_record("worldpop", raw_dir, [sources.worldpop], verified, _source_coverage()),
        source_record(
            "osm",
            raw_dir,
            [
                sources.osm_roads,
                sources.osm_buildings,
                sources.osm_pois,
                sources.osm_places,
            ],
            verified,
            _source_coverage(),
        ),
    ]
    if sources.ghsl:
        manifest_sources.append(
            source_record("ghsl", raw_dir, [sources.ghsl], verified, _source_coverage())
        )
    if sources.admin_adm1 or sources.admin_adm2:
        admin_files = [
            path
            for path in (sources.admin_adm1, sources.admin_adm2)
            if path is not None
        ]
        manifest_sources.append(
            source_record("ocha", raw_dir, admin_files, verified, _source_coverage())
        )

    manifest = build_manifest(
        manifest_sources,
        thresholds,
        artifacts,
        study_area={
            "lat_min": STUDY_LAT_MIN,
            "lat_max": STUDY_LAT_MAX,
            "lon_min": STUDY_LON_MIN,
            "lon_max": STUDY_LON_MAX,
            "cell_count": int(len(cells)),
            "cell_area_total_km2": float(cells["cell_area_km2"].sum()),
            "cell_source": "fire_panel" if use_panel else "full_model_grid",
        },
    )
    manifest_path = out_dir / config.MANIFEST_PATH.name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=config.IMPACT_RAW_DIR,
        help="Directory containing manual raw downloads.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.IMPACT_DATA_DIR,
        help="Directory for derived parquet artifacts and manifest.json.",
    )
    parser.add_argument(
        "--checksums",
        type=Path,
        default=None,
        help=f"Checksum file (default: <raw-dir>/{CHECKSUM_FILENAME}).",
    )
    parser.add_argument(
        "--record-checksums",
        action="store_true",
        help="Record SHA-256 hashes for the downloaded files before building.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=2048,
        help="Raster block size in pixels (default: 2048).",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Build all 216 grid cells instead of the panel's cells.",
    )
    parser.add_argument(
        "--skip-rings",
        action="store_true",
        help="Do not build the 5/10/25 km ring table (faster).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checksums_path = args.checksums or (args.raw_dir / CHECKSUM_FILENAME)
    try:
        manifest = run_build(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            checksums_path=checksums_path,
            record_checksums=args.record_checksums,
            block_size=args.block_size,
            use_panel=not args.full_grid,
            with_rings=not args.skip_rings,
        )
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - offline operator-facing script
        print(f"ERROR: unexpected build failure: {exc}", file=sys.stderr)
        return 1

    print("Impact layers built successfully.")
    print(f"  cells: {manifest['study_area']['cell_count']}")
    print(f"  thresholds: {manifest['severity']['tier_thresholds']}")
    print(f"  artifacts: {', '.join(sorted(manifest['artifacts']))}")
    print(f"  manifest: {args.out_dir / config.MANIFEST_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
