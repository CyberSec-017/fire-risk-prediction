"""Unit tests for the pure helpers in ``scripts/build_impact_layers.py``.

The tests only use synthetic frames and temporary files.  They never touch the
network, the real raw downloads or large raster/vector data, and they never
require geopandas, rasterio, shapely or pyproj to be installed: those imports
are lazy inside the script.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from impact import config, exposure, severity  # noqa: E402

BUILD_PATH = ROOT / "scripts" / "build_impact_layers.py"
DOWNLOAD_DOC = ROOT / "DATA_DOWNLOAD.md"


@pytest.fixture(scope="module")
def build():
    spec = importlib.util.spec_from_file_location("impact_build_script", BUILD_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Synthetic frames
# ---------------------------------------------------------------------------
def _cells() -> pd.DataFrame:
    coordinates = [(32.0, -1.875), (32.0, -1.25), (32.5, -1.875)]
    return pd.DataFrame(
        {
            "cell_id": [
                f"lat{lat:.3f}_lon{lon:.3f}" for lat, lon in coordinates
            ],
            "lat_cell": [lat for lat, _ in coordinates],
            "lon_cell": [lon for _, lon in coordinates],
            "cell_area_km2": [
                exposure.cell_area_km2(lat, lon) for lat, lon in coordinates
            ],
        }
    )


def _landcover(cells: pd.DataFrame) -> pd.DataFrame:
    fractions = {
        "lc_tree_km2": 0.40,
        "lc_shrub_km2": 0.20,
        "lc_grass_km2": 0.15,
        "lc_crop_km2": 0.10,
        "lc_bare_km2": 0.07,
        "lc_built_km2": 0.04,
        "lc_water_km2": 0.03,
        "lc_other_km2": 0.01,
    }
    frame = pd.DataFrame({"cell_id": cells["cell_id"].to_numpy()})
    for column, fraction in fractions.items():
        frame[column] = cells["cell_area_km2"].to_numpy() * fraction
    return frame


def _population(cells: pd.DataFrame) -> pd.DataFrame:
    scale = np.arange(1, len(cells) + 1, dtype=float)
    return pd.DataFrame(
        {
            "cell_id": cells["cell_id"].to_numpy(),
            "pop_cell": 1_000.0 * scale,
            "pop_5km": 5_000.0 * scale,
            "pop_10km": 15_000.0 * scale,
            "pop_25km": 80_000.0 * scale,
            "pop_status": "ok",
        }
    )


def _infrastructure(cells: pd.DataFrame) -> pd.DataFrame:
    scale = np.arange(1, len(cells) + 1, dtype=float)
    return pd.DataFrame(
        {
            "cell_id": cells["cell_id"].to_numpy(),
            "road_km_total": 10.0 * scale,
            "building_count": 100.0 * scale,
            "building_footprint_km2": 0.4 * scale,
            "critical_count_total": 1.0 + scale,
            "pop_status": "ok",
        }
    )


# ---------------------------------------------------------------------------
# Grid and tile helpers
# ---------------------------------------------------------------------------
def test_worldcover_tiles_cover_study_area(build):
    names = build.expected_worldcover_files()
    assert len(names) == 12
    assert len(set(names)) == 12
    assert all(name.endswith("_Map.tif") for name in names)
    for required in ("N30W003", "N30E000", "N33E003", "N36E006"):
        assert any(required in name for name in names)


def test_expected_tiles_match_download_documentation(build):
    text = DOWNLOAD_DOC.read_text(encoding="utf-8")
    for name in build.expected_worldcover_files():
        assert name in text, f"{name} is missing from DATA_DOWNLOAD.md"


def test_geospatial_imports_are_lazy(build):
    tree = ast.parse(BUILD_PATH.read_text(encoding="utf-8"))
    top_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    forbidden = {"geopandas", "rasterio", "shapely", "pyproj", "pyogrio"}
    assert not (top_level & forbidden)
    # Importing the module must not pull the geospatial stack either.
    assert "geopandas" not in sys.modules
    assert "rasterio" not in sys.modules


def test_grid_cells_cover_full_model_grid(build):
    frame = build.grid_cells()
    # The panel's grid is 12 latitude rows x 18 longitude columns.
    assert len(frame) == 216
    assert frame["cell_id"].is_unique
    assert frame[[ "lat_cell", "lon_cell"]].drop_duplicates().shape[0] == 216
    assert (frame["cell_area_km2"] > 0).all()


def test_cells_from_panel_uses_panel_cells(build):
    panel = pd.DataFrame(
        {
            "lat_cell": [32.0, 32.0, 37.5],
            "lon_cell": [-1.875, -1.875, 8.75],
        }
    )
    cells = build.cells_from_panel(panel)
    assert len(cells) == 2
    assert cells["cell_id"].is_unique
    assert cells["lat_cell"].max() == 37.5


def test_cells_from_panel_rejects_off_grid_coordinates(build):
    panel = pd.DataFrame({"lat_cell": [32.1], "lon_cell": [-1.875]})
    with pytest.raises(build.BuildError, match="off-grid"):
        build.cells_from_panel(panel)


def test_cells_for_build_falls_back_to_full_grid(build, tmp_path):
    missing = tmp_path / "does_not_exist.parquet"
    cells = build.cells_for_build(panel_path=missing, use_panel=True)
    assert len(cells) == 216


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------
def test_parse_checksums_accepts_json_and_sha256sum_text(build):
    digest = "a" * 64
    assert build.parse_checksums(json.dumps({"tile.tif": digest})) == {
        "tile.tif": digest
    }
    text = f"{digest} *tile.tif\n"
    assert build.parse_checksums(text) == {"tile.tif": digest}
    assert build.parse_checksums("") == {}


@pytest.mark.parametrize(
    "text",
    [
        json.dumps({"tile.tif": "not-a-hash"}),
        "deadbeef  tile.tif",
        json.dumps(["not", "an", "object"]),
    ],
)
def test_parse_checksums_rejects_invalid_input(build, text):
    with pytest.raises(build.ChecksumError):
        build.parse_checksums(text)


def test_verify_checksums_detects_mismatch_and_missing_entry(build, tmp_path):
    good = tmp_path / "good.bin"
    good.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()

    assert build.verify_checksums(tmp_path, [good], {"good.bin": digest}) == {
        "good.bin": digest
    }

    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"tampered")
    with pytest.raises(build.ChecksumError, match="checksum mismatch for bad.bin"):
        build.verify_checksums(tmp_path, [bad], {"bad.bin": digest})

    with pytest.raises(build.ChecksumError, match="no checksum recorded"):
        build.verify_checksums(tmp_path, [good], {})


def test_checksum_roundtrip(build, tmp_path):
    file = tmp_path / "raw" / "tile.bin"
    file.parent.mkdir()
    file.write_bytes(b"payload")
    checksum_path = tmp_path / "checksums.json"

    written = build.write_checksums(tmp_path / "raw", [file], checksum_path)
    assert written == {"tile.bin": hashlib.sha256(b"payload").hexdigest()}
    assert build.load_checksums(checksum_path) == written


def test_load_checksums_reports_missing_file(build, tmp_path):
    with pytest.raises(build.ChecksumError, match="does not exist"):
        build.load_checksums(tmp_path / "absent.json")


# ---------------------------------------------------------------------------
# Land-cover helpers
# ---------------------------------------------------------------------------
def test_worldcover_codes_map_to_canonical_keys(build):
    assert build.worldcover_lc_key(10) == "tree"
    assert build.worldcover_lc_key(50) == "built"
    assert build.worldcover_lc_key(95) == "other"
    assert build.worldcover_lc_key(100) == "other"
    with pytest.raises(ValueError, match="Unknown WorldCover class code"):
        build.worldcover_lc_key(999)


def test_landcover_areas_from_counts(build):
    areas = build.landcover_areas_from_counts({10: 4, 50: 2, 0: 99}, 0.5)
    assert set(areas) == set(config.LC_AREA_COLUMNS)
    assert areas["tree"] == pytest.approx(2.0)
    assert areas["built"] == pytest.approx(1.0)
    assert areas["water"] == pytest.approx(0.0)


def test_landcover_areas_reject_bad_input(build):
    with pytest.raises(ValueError, match="pixel_area_km2"):
        build.landcover_areas_from_counts({10: 1}, 0)
    with pytest.raises(ValueError, match="Unknown WorldCover class code"):
        build.landcover_areas_from_counts({999: 1}, 0.5)
    with pytest.raises(ValueError, match="Invalid pixel count"):
        build.landcover_areas_from_counts({10: -1}, 0.5)


def test_landcover_frame_folds_class_codes(build):
    class_areas = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            10: [1.0, 2.0],
            20: [0.5, 0.5],
            90: [0.25, 0.25],
            95: [0.25, 0.25],
            0: [9.0, 9.0],
        }
    )
    folded = build.landcover_frame_from_class_areas(class_areas)
    assert list(folded["cell_id"]) == ["a", "b"]
    assert folded.loc[0, "lc_tree_km2"] == pytest.approx(1.0)
    assert folded.loc[1, "lc_tree_km2"] == pytest.approx(2.0)
    assert folded.loc[0, "lc_other_km2"] == pytest.approx(0.5)
    assert folded.loc[0, "lc_water_km2"] == pytest.approx(0.0)
    assert set(config.LC_AREA_COLUMNS.values()) <= set(folded.columns)


def test_landcover_frame_rejects_unknown_and_negative(build):
    with pytest.raises(ValueError, match="Unknown WorldCover class code"):
        build.landcover_frame_from_class_areas(
            pd.DataFrame({"cell_id": ["a"], 999: [1.0]})
        )
    with pytest.raises(ValueError, match="Invalid area values"):
        build.landcover_frame_from_class_areas(
            pd.DataFrame({"cell_id": ["a"], 10: [-1.0]})
        )


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("primary", "primary"),
        ("motorway_link", "primary"),
        ("secondary", "secondary"),
        ("tertiary_link", "tertiary"),
        ("track_grade2", "track"),
        ("residential", "other"),
        (None, "other"),
    ],
)
def test_classify_road_class(build, value, expected):
    assert build.classify_road_class(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("hospital", "hospital"),
        ("clinic", "hospital"),
        ("school", "school"),
        ("substation", "power"),
        ("aerodrome", "airport"),
        ("water_works", "water"),
        ("fire_station", "emergency"),
        ("restaurant", None),
        (None, None),
    ],
)
def test_classify_critical_facility(build, value, expected):
    assert build.classify_critical_facility(value) == expected


# ---------------------------------------------------------------------------
# Thresholds and manifest
# ---------------------------------------------------------------------------
def test_freeze_tier_thresholds_are_strictly_increasing(build):
    values = np.linspace(0.0, 1.0, 101)
    thresholds = build.freeze_tier_thresholds(values)
    assert len(thresholds) == 4
    assert list(thresholds) == sorted(thresholds)
    assert len(set(thresholds)) == 4
    assert all(0.0 < value < 1.0 for value in thresholds)


def test_freeze_tier_thresholds_survives_concentrated_distribution(build):
    thresholds = build.freeze_tier_thresholds([0.5] * 50)
    assert len(set(thresholds)) == 4
    assert all(value == pytest.approx(0.5, abs=1e-4) for value in thresholds)


def test_freeze_tier_thresholds_rejects_invalid_input(build):
    with pytest.raises(ValueError, match="empty"):
        build.freeze_tier_thresholds([])
    with pytest.raises(ValueError, match="Exactly four quantiles"):
        build.freeze_tier_thresholds([0.5], quantiles=(0.2, 0.5))
    with pytest.raises(ValueError, match="Quantiles must lie strictly"):
        build.freeze_tier_thresholds([0.5], quantiles=(0.0, 0.4, 0.6, 0.8))
    with pytest.raises(ValueError):
        build.freeze_tier_thresholds([1.0] * 20)


def test_build_manifest_is_json_serialisable_and_validated(build):
    manifest = build.build_manifest(
        sources=[{"name": "synthetic", "checksum": "a" * 64}],
        thresholds=(0.1, 0.2, 0.3, 0.4),
        artifacts={"cell_exposure": {"path": "cell_exposure.parquet", "rows": 3}},
        study_area={"cell_count": 3},
        generated_at="2026-01-01T00:00:00Z",
    )
    assert manifest["schema_version"] == build.MANIFEST_SCHEMA_VERSION
    assert manifest["severity"]["tier_thresholds"] == [0.1, 0.2, 0.3, 0.4]
    assert manifest["severity"]["tier_labels"] == list(severity.TIER_LABELS)
    assert manifest["grid"]["lat_step"] == config.MODEL_LAT_STEP
    assert manifest["generated_at"] == "2026-01-01T00:00:00Z"
    assert "burned_area" in manifest["phase_2"]
    json.dumps(manifest)


def test_build_manifest_rejects_bad_thresholds(build):
    arguments = dict(
        sources=[],
        artifacts={},
        study_area={},
    )
    with pytest.raises(ValueError, match="Exactly four"):
        build.build_manifest(thresholds=(0.1, 0.2, 0.3), **arguments)
    with pytest.raises(ValueError, match="strictly increasing"):
        build.build_manifest(thresholds=(0.2, 0.1, 0.3, 0.4), **arguments)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        build.build_manifest(thresholds=(0.0, 0.2, 0.3, 0.4), **arguments)


def test_source_record_checksums_change_with_content(build, tmp_path):
    file = tmp_path / "tile.bin"
    file.write_bytes(b"first")
    verified = {"tile.bin": hashlib.sha256(b"first").hexdigest()}
    first = build.source_record(
        "worldcover", tmp_path, [file], verified, {"lat_min": 32.0}
    )
    assert first["name"] == build.SOURCE_METADATA["worldcover"]["name"]
    assert first["license"] == "CC BY 4.0"
    assert first["files"] == ["tile.bin"]
    assert first["checksum"] == build._combined_checksum(first["checksums"])

    file.write_bytes(b"second")
    second = build.source_record(
        "worldcover", tmp_path, [file], {}, {"lat_min": 32.0}
    )
    assert second["checksum"] != first["checksum"]
    assert second["checksums"]["tile.bin"] == hashlib.sha256(b"second").hexdigest()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def test_haversine_known_distance(build):
    assert build.haversine_km(32.0, 0.0, 33.0, 0.0) == pytest.approx(111.2, abs=0.5)
    assert build.haversine_km(32.0, 0.0, 32.0, 0.0) == pytest.approx(0.0)


def test_nearest_cell_coordinates_snap_to_grid(build):
    lat, lon = build.nearest_cell_coordinates([32.2, 40.0], [-1.8, 20.0])
    assert lat[0] == pytest.approx(32.0)
    assert lon[0] == pytest.approx(-1.875)
    assert lat[1] == pytest.approx(40.0)
    assert lon[1] == pytest.approx(20.0)


def test_assign_points_to_cells_marks_outside_as_null(build):
    cells = _cells()
    points = pd.DataFrame({"lat": [32.05, 40.0], "lon": [-1.80, 20.0]})
    assigned = build.assign_points_to_cells(points, cells)
    assert assigned.loc[0, "cell_id"] == cells.loc[0, "cell_id"]
    assert pd.isna(assigned.loc[1, "cell_id"])


def test_points_within_radius_sum_keeps_rings_separate(build):
    cells = _cells().iloc[[0]]
    points = pd.DataFrame(
        {
            "lat": [32.0, 32.4, 32.0],
            "lon": [-1.875, -1.875, -1.875],
            "value": [10.0, 5.0, np.nan],
        }
    )
    near = build.points_within_radius_sum(points, cells, 5, "value")
    far = build.points_within_radius_sum(points, cells, 100, "value")
    assert near.loc[0, "value"] == pytest.approx(10.0)
    assert near.loc[0, "ring_km"] == pytest.approx(5.0)
    assert far.loc[0, "value"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Source resolution and assembly
# ---------------------------------------------------------------------------
def test_resolve_required_sources_lists_every_missing_file(build, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(build.MissingSourceError) as info:
        build.resolve_required_sources(raw)
    message = str(info.value)
    # WorldCover is optional, so WorldPop and OSM are listed as missing
    assert build.WORLDPOP_FILENAME in message
    assert "gis_osm_roads_free_1.shp" in message
    assert "DATA_DOWNLOAD.md" in message


def test_resolve_required_sources_rejects_partial_worldcover(build, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # Write only 1 tile out of 12 -> must fail loudly
    (raw / "ESA_WorldCover_10m_2021_v200_N30W003_Map.tif").write_bytes(b"tile")
    with pytest.raises(build.BuildError, match="Partial WorldCover coverage detected"):
        build.resolve_required_sources(raw)


def test_resolve_required_sources_without_worldcover(build, tmp_path):
    raw = tmp_path / "raw"
    osm = raw / build.OSM_FOLDER
    osm.mkdir(parents=True)
    (raw / build.WORLDPOP_FILENAME).write_bytes(b"pop")
    for name in build.OSM_REQUIRED:
        (osm / name).write_bytes(b"osm")

    sources = build.resolve_required_sources(raw)
    assert len(sources.worldcover) == 0
    assert sources.worldpop.name == build.WORLDPOP_FILENAME
    assert sources.osm_roads.name == "gis_osm_roads_free_1.shp"
    assert len(sources.required_files()) == 5


def test_resolve_required_sources_accepts_dated_geofabrik_folder(build, tmp_path):
    raw = tmp_path / "raw"
    osm = raw / "algeria-261110-free.shp"
    osm.mkdir(parents=True)
    (raw / build.WORLDPOP_FILENAME).write_bytes(b"pop")
    for name in build.OSM_REQUIRED:
        (osm / name).write_bytes(b"osm")

    sources = build.resolve_required_sources(raw)
    assert sources.osm_folder_name == "algeria-261110-free.shp"
    assert sources.osm_roads.name == "gis_osm_roads_free_1.shp"


def test_resolve_required_sources_with_synthetic_files(build, tmp_path):
    raw = tmp_path / "raw"
    osm = raw / build.OSM_FOLDER
    osm.mkdir(parents=True)
    for name in build.expected_worldcover_files():
        (raw / name).write_bytes(b"tile")
    (raw / build.WORLDPOP_FILENAME).write_bytes(b"pop")
    for name in build.OSM_REQUIRED:
        (osm / name).write_bytes(b"osm")

    sources = build.resolve_required_sources(raw)
    assert len(sources.worldcover) == 12
    assert sources.worldpop.name == build.WORLDPOP_FILENAME
    assert sources.osm_roads.name == "gis_osm_roads_free_1.shp"
    assert sources.ghsl is None
    assert len(sources.required_files()) == 17
    assert len(sources.all_files()) == 17


def test_resolve_required_sources_reports_missing_directory(build, tmp_path):
    with pytest.raises(build.MissingSourceError, match="directory does not exist"):
        build.resolve_required_sources(tmp_path / "absent")


def test_assemble_cell_exposure_builds_canonical_columns(build):
    cells = _cells()
    frame = build.assemble_cell_exposure(
        cells, _landcover(cells), _population(cells), _infrastructure(cells)
    )
    required = (
        set(config.ENV_REQUIRED_COLUMNS)
        | set(config.POP_REQUIRED_COLUMNS)
        | set(config.INFRA_REQUIRED_COLUMNS)
    )
    assert required <= set(frame.columns)
    assert len(frame) == len(cells)
    assert (frame["pop_cell"] > 0).all()


def test_assemble_cell_exposure_never_zero_fills_missing_landcover(build):
    cells = _cells()
    landcover = _landcover(cells).drop(index=1)
    with pytest.raises(build.BuildError, match="never replaced with zero"):
        build.assemble_cell_exposure(
            cells, landcover, _population(cells), _infrastructure(cells)
        )


def test_assemble_cell_exposure_never_zero_fills_missing_population(build):
    cells = _cells()
    population = _population(cells)
    population.loc[1, "pop_cell"] = np.nan
    with pytest.raises(build.BuildError, match="never replaced with zero"):
        build.assemble_cell_exposure(
            cells, _landcover(cells), population, _infrastructure(cells)
        )


def test_assemble_cell_exposure_allows_zero_osm_counts(build):
    cells = _cells()
    infrastructure = _infrastructure(cells)
    infrastructure.loc[:, ["road_km_total", "building_count", "critical_count_total"]] = 0.0
    frame = build.assemble_cell_exposure(
        cells, _landcover(cells), _population(cells), infrastructure
    )
    assert (frame["road_km_total"] == 0).all()
    assert frame["pop_cell"].notna().all()


def test_summarize_settlements_keeps_missing_population_explicit(build):
    cells = _cells()
    settlements = pd.DataFrame(
        {
            "cell_id": [
                cells.loc[0, "cell_id"],
                cells.loc[0, "cell_id"],
                cells.loc[2, "cell_id"],
            ],
            "settlement_name": ["A", "B", "C"],
            "population": [100.0, 200.0, np.nan],
        }
    )
    totals = build.summarize_settlements(settlements, cells)

    assert totals.loc[0, "settlement_count"] == 2
    assert totals.loc[0, "settlement_population"] == pytest.approx(300.0)
    assert totals.loc[1, "settlement_count"] == 0
    assert pd.isna(totals.loc[1, "settlement_population"])
    assert totals.loc[2, "settlement_count"] == 1
    assert pd.isna(totals.loc[2, "settlement_population"])

    empty = build.summarize_settlements(None, cells)
    assert (empty["settlement_count"] == 0).all()
    assert empty["settlement_population"].isna().all()


# ---------------------------------------------------------------------------
# Writer and CLI
# ---------------------------------------------------------------------------
def test_write_parquet_validates_columns_and_hashes(build, tmp_path):
    frame = pd.DataFrame({"cell_id": ["a", "b"], "value": [1.0, 2.0]})
    path = tmp_path / "out" / "example.parquet"
    record = build.write_parquet(frame, path, required=["cell_id", "value"])
    assert path.exists()
    assert record["rows"] == 2
    assert record["sha256"] == build.sha256_file(path)
    assert pd.read_parquet(path).equals(frame)

    with pytest.raises(build.BuildError, match="missing column"):
        build.write_parquet(frame, tmp_path / "bad.parquet", required=["absent"])


def test_main_returns_two_when_sources_are_missing(build, tmp_path, capsys):
    code = build.main(
        ["--raw-dir", str(tmp_path / "raw"), "--out-dir", str(tmp_path / "out")]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert "Missing" in captured.err
    assert "DATA_DOWNLOAD.md" in captured.err
    assert not (tmp_path / "out").exists()


def test_run_build_assembles_artifacts_with_injected_readers(build, tmp_path, monkeypatch):
    """Exercise the full build flow without any geospatial dependency.

    All raster/vector readers are replaced by synthetic frames, so this test
    checks the assembly, manifest and writer wiring only.
    """
    raw = tmp_path / "raw"
    osm = raw / build.OSM_FOLDER
    osm.mkdir(parents=True)
    for name in build.expected_worldcover_files():
        (raw / name).write_bytes(name.encode("utf-8"))
    (raw / build.WORLDPOP_FILENAME).write_bytes(b"population")
    for name in build.OSM_REQUIRED:
        (osm / name).write_bytes(name.encode("utf-8"))

    checksums_path = raw / build.CHECKSUM_FILENAME
    sources = build.resolve_required_sources(raw)
    build.write_checksums(raw, sources.all_files(), checksums_path)

    cells = build.cells_from_panel(
        pd.DataFrame({"lat_cell": [32.0, 32.0], "lon_cell": [-1.875, -1.25]})
    )

    def _with_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.merge(cells, on="cell_id", how="left", validate="one_to_one")

    population = _with_coordinates(_population(cells))
    ring_rows = []
    for cell in cells.itertuples():
        for radius in config.RING_DISTANCES_KM:
            ring_rows.append(
                {
                    "lat_cell": cell.lat_cell,
                    "lon_cell": cell.lon_cell,
                    "ring_km": float(radius),
                    "population": 1_000.0 * radius,
                }
            )
    population_rings = pd.DataFrame(ring_rows)

    roads = _infrastructure(cells)[["cell_id", "road_km_total"]].copy()
    for column in build.ROAD_CLASS_COLUMNS.values():
        roads[column] = 1.0
    buildings = _infrastructure(cells)[
        ["cell_id", "building_count", "building_footprint_km2"]
    ].copy()
    critical_counts = _infrastructure(cells)[["cell_id", "critical_count_total"]].copy()
    for column in build.CRITICAL_CLASS_COLUMNS.values():
        critical_counts[column] = 1.0

    settlements = pd.DataFrame(
        {
            "lat_cell": cells["lat_cell"].to_numpy(),
            "lon_cell": cells["lon_cell"].to_numpy(),
            "settlement_name": ["A", "B"],
            "population": [100.0, 200.0],
            "distance_km": [0.0, 1.0],
            "cell_id": cells["cell_id"].to_numpy(),
        }
    )
    settlement_counts = pd.DataFrame(
        {
            "cell_id": cells["cell_id"].to_numpy(),
            "settlement_count": [1.0, 1.0],
            "settlement_population": [100.0, 200.0],
        }
    )
    assets = pd.DataFrame(
        {
            "lat_cell": cells["lat_cell"].to_numpy(),
            "lon_cell": cells["lon_cell"].to_numpy(),
            "category": ["hospital", "school"],
            "asset_name": ["Clinic A", "School B"],
            "cell_id": cells["cell_id"].to_numpy(),
        }
    )

    monkeypatch.setattr(build, "cells_for_build", lambda **_: cells.copy())
    monkeypatch.setattr(
        build, "read_worldcover_areas", lambda *_, **__: _landcover(cells)
    )
    monkeypatch.setattr(
        build, "read_worldpop_cell_totals", lambda *_, **__: population
    )
    monkeypatch.setattr(
        build, "read_worldpop_ring_totals", lambda *_, **__: population_rings
    )
    monkeypatch.setattr(build, "read_osm_roads", lambda *_, **__: roads)
    monkeypatch.setattr(build, "read_osm_buildings", lambda *_, **__: (buildings, None))
    monkeypatch.setattr(
        build, "read_osm_pois", lambda *_, **__: (assets, critical_counts)
    )
    monkeypatch.setattr(
        build,
        "read_osm_settlements",
        lambda *_, **__: (settlements, settlement_counts),
    )
    monkeypatch.setattr(
        build,
        "_read_vector",
        lambda *_, **__: pd.DataFrame({"fclass": pd.Series([], dtype=object)}),
    )
    monkeypatch.setattr(
        build,
        "summarize_vector_rings",
        lambda frame, cells, distances, kind, **__: pd.DataFrame(),
    )

    out = tmp_path / "out"
    manifest = build.run_build(
        raw_dir=raw,
        out_dir=out,
        checksums_path=checksums_path,
        with_rings=True,
    )

    assert manifest["study_area"]["cell_count"] == 2
    assert len(manifest["severity"]["tier_thresholds"]) == 4
    assert set(manifest["artifacts"]) == {
        "cell_exposure",
        "settlements",
        "infrastructure",
        "ring_exposure",
    }
    for name in (
        config.CELL_EXPOSURE_PATH.name,
        config.SETTLEMENTS_PATH.name,
        config.INFRASTRUCTURE_PATH.name,
        config.RING_EXPOSURE_PATH.name,
        config.MANIFEST_PATH.name,
    ):
        assert (out / name).exists()

    rings = pd.read_parquet(out / config.RING_EXPOSURE_PATH.name)
    assert len(rings) == 2 * len(config.RING_DISTANCES_KM)
    cell_exposure = pd.read_parquet(out / config.CELL_EXPOSURE_PATH.name)
    assert cell_exposure["pop_cell"].notna().all()
    assert (cell_exposure["lc_tree_km2"] > 0).all()


def test_run_build_rejects_checksum_mismatch(build, tmp_path):
    raw = tmp_path / "raw"
    osm = raw / build.OSM_FOLDER
    osm.mkdir(parents=True)
    for name in build.expected_worldcover_files():
        (raw / name).write_bytes(name.encode("utf-8"))
    (raw / build.WORLDPOP_FILENAME).write_bytes(b"population")
    for name in build.OSM_REQUIRED:
        (osm / name).write_bytes(name.encode("utf-8"))

    wrong = {
        name: "0" * 64
        for name in build.expected_worldcover_files()
    }
    checksums_path = raw / build.CHECKSUM_FILENAME
    checksums_path.write_text(json.dumps(wrong), encoding="utf-8")

    with pytest.raises(build.ChecksumError, match="checksum mismatch"):
        build.run_build(
            raw_dir=raw,
            out_dir=tmp_path / "out",
            checksums_path=checksums_path,
            with_rings=False,
        )
    assert not (tmp_path / "out").exists()


def test_run_build_without_worldcover(build, tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    osm = raw / "algeria-261110-free.shp"
    osm.mkdir(parents=True)
    (raw / build.WORLDPOP_FILENAME).write_bytes(b"population")
    for name in build.OSM_REQUIRED:
        (osm / name).write_bytes(name.encode("utf-8"))

    checksums_path = raw / build.CHECKSUM_FILENAME
    sources = build.resolve_required_sources(raw)
    build.write_checksums(raw, sources.all_files(), checksums_path)

    cells = build.cells_from_panel(
        pd.DataFrame({"lat_cell": [32.0, 32.0], "lon_cell": [-1.875, -1.25]})
    )

    def _with_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.merge(cells, on="cell_id", how="left", validate="one_to_one")

    population = _with_coordinates(_population(cells))
    ring_rows = []
    for cell in cells.itertuples():
        for radius in config.RING_DISTANCES_KM:
            ring_rows.append(
                {
                    "lat_cell": cell.lat_cell,
                    "lon_cell": cell.lon_cell,
                    "ring_km": float(radius),
                    "population": 1_000.0 * radius,
                }
            )
    population_rings = pd.DataFrame(ring_rows)

    roads = _infrastructure(cells)[["cell_id", "road_km_total"]].copy()
    for column in build.ROAD_CLASS_COLUMNS.values():
        roads[column] = 1.0
    buildings = _infrastructure(cells)[
        ["cell_id", "building_count", "building_footprint_km2"]
    ].copy()
    critical_counts = _infrastructure(cells)[["cell_id", "critical_count_total"]].copy()
    for column in build.CRITICAL_CLASS_COLUMNS.values():
        critical_counts[column] = 1.0

    settlements = pd.DataFrame(
        {
            "lat_cell": cells["lat_cell"].to_numpy(),
            "lon_cell": cells["lon_cell"].to_numpy(),
            "settlement_name": ["A", "B"],
            "population": [100.0, 200.0],
            "distance_km": [0.0, 1.0],
            "cell_id": cells["cell_id"].to_numpy(),
        }
    )
    settlement_counts = pd.DataFrame(
        {
            "cell_id": cells["cell_id"].to_numpy(),
            "settlement_count": [1.0, 1.0],
            "settlement_population": [100.0, 200.0],
        }
    )
    assets = pd.DataFrame(
        {
            "lat_cell": cells["lat_cell"].to_numpy(),
            "lon_cell": cells["lon_cell"].to_numpy(),
            "category": ["hospital", "school"],
            "asset_name": ["Clinic A", "School B"],
            "cell_id": cells["cell_id"].to_numpy(),
        }
    )

    monkeypatch.setattr(build, "cells_for_build", lambda **_: cells.copy())
    monkeypatch.setattr(
        build, "read_worldpop_cell_totals", lambda *_, **__: population
    )
    monkeypatch.setattr(
        build, "read_worldpop_ring_totals", lambda *_, **__: population_rings
    )
    monkeypatch.setattr(build, "read_osm_roads", lambda *_, **__: roads)
    monkeypatch.setattr(build, "read_osm_buildings", lambda *_, **__: (buildings, None))
    monkeypatch.setattr(
        build, "read_osm_pois", lambda *_, **__: (assets, critical_counts)
    )
    monkeypatch.setattr(
        build,
        "read_osm_settlements",
        lambda *_, **__: (settlements, settlement_counts),
    )

    out = tmp_path / "out"
    manifest = build.run_build(
        raw_dir=raw,
        out_dir=out,
        checksums_path=checksums_path,
        with_rings=False,
    )

    # Verify manifest records WorldCover as not ingested
    wc_source = [s for s in manifest["sources"] if s["name"] == build.SOURCE_METADATA["worldcover"]["name"]][0]
    assert wc_source["status"] == "not ingested"
    assert "not downloaded" in wc_source["reason"]

    # Verify severity was computed with pop+infra and renormalized weights
    assert manifest["severity"]["components_used"] == ["pop", "infra"]
    assert "0.35/0.65" in manifest["severity"]["method"]

    # Verify cell_exposure has NaN landcover and NaN fuel_area_km2
    cell_exposure = pd.read_parquet(out / config.CELL_EXPOSURE_PATH.name)
    assert cell_exposure["lc_tree_km2"].isna().all()
    assert cell_exposure["fuel_area_km2"].isna().all()
    assert cell_exposure["pop_cell"].notna().all()


# ---------------------------------------------------------------------------
# New tests: chunking, bbox, vectorised areas, --max-chunks protection
# ---------------------------------------------------------------------------

def test_osm_bbox_constant_matches_study_area(build):
    """OSM_BBOX must fully contain the study area and follow (xmin,ymin,xmax,ymax)."""
    xmin, ymin, xmax, ymax = build.OSM_BBOX
    assert xmin <= build.STUDY_LON_MIN
    assert xmax >= build.STUDY_LON_MAX
    assert ymin <= build.STUDY_LAT_MIN
    assert ymax >= build.STUDY_LAT_MAX


def test_equal_area_crs_constant_is_6933(build):
    """EQUAL_AREA_CRS must be 6933 (EPSG:6933 — EASE-Grid 2.0 World)."""
    assert build.EQUAL_AREA_CRS == 6933


def test_osm_chunk_features_constant_is_set(build):
    """OSM_CHUNK_FEATURES must be a positive integer."""
    assert isinstance(build.OSM_CHUNK_FEATURES, int)
    assert build.OSM_CHUNK_FEATURES >= 1_000


def test_read_vector_chunks_honours_max_chunks(build, tmp_path, monkeypatch):
    """_read_vector_chunks must stop after max_chunks iterations."""
    import types

    calls: list[int] = []

    def _fake_pyogrio():
        mod = types.SimpleNamespace()

        def read_info(path):
            return {"features": 1_000_000, "fields": []}

        def read_dataframe(path, bbox=None, columns=None, skip_features=0,
                           max_features=None, use_arrow=False):
            import geopandas as gpd
            from shapely.geometry import Point

            n = min(max_features or 10, max(0, 1_000_000 - skip_features))
            calls.append(skip_features)
            return gpd.GeoDataFrame(
                {"geometry": [Point(0, 32)] * n},
                crs=4326,
            )

        mod.read_info = read_info
        mod.read_dataframe = read_dataframe
        return mod

    monkeypatch.setattr(build, "_pyogrio", _fake_pyogrio)

    chunks = list(
        build._read_vector_chunks(
            tmp_path / "fake.shp",
            chunk_size=100_000,
            max_chunks=3,
        )
    )
    assert len(chunks) == 3
    assert len(calls) == 3
    # skip_features must advance by chunk_size each call
    assert calls[0] == 0
    assert calls[1] == 100_000
    assert calls[2] == 200_000


def test_read_vector_chunks_passes_bbox(build, tmp_path, monkeypatch):
    """_read_vector_chunks must forward OSM_BBOX to pyogrio.read_dataframe."""
    import types

    received_bbox: list = []

    def _fake_pyogrio():
        mod = types.SimpleNamespace()
        mod.read_info = lambda path: {"features": 10, "fields": []}

        def read_dataframe(path, bbox=None, **kwargs):
            received_bbox.append(bbox)
            import geopandas as gpd
            from shapely.geometry import Point
            return gpd.GeoDataFrame({"geometry": [Point(0, 32)]}, crs=4326)

        mod.read_dataframe = read_dataframe
        return mod

    monkeypatch.setattr(build, "_pyogrio", _fake_pyogrio)

    list(build._read_vector_chunks(tmp_path / "fake.shp", chunk_size=100, max_chunks=1))
    assert received_bbox[0] == build.OSM_BBOX

def test_vectorised_area_uses_equal_area_crs(build, monkeypatch):
    pytest.importorskip("geopandas")
    pytest.importorskip("shapely")
    import geopandas as gpd
    from shapely.geometry import box

    cells = _cells()
    lat0, lon0 = float(cells.iloc[0]["lat_cell"]), float(cells.iloc[0]["lon_cell"])
    # ~1 km x 1 km square at the first cell centre
    d = 0.0045
    geom = box(lon0 - d, lat0 - d, lon0 + d, lat0 + d)

    def _fake_pyogrio():
        import types
        mod = types.SimpleNamespace()
        mod.read_info = lambda path: {"features": 1, "fields": []}
        mod.read_dataframe = lambda path, **kw: gpd.GeoDataFrame(
            {"geometry": [geom]}, crs=4326)
        return mod

    monkeypatch.setattr(build, "_pyogrio", _fake_pyogrio)
    df, _ = build.read_osm_buildings(Path("fake.shp"), cells, chunk_size=10, max_chunks=1)
    area = float(df["building_footprint_km2"].sum())
    assert 0.5 < area < 1.5   # about 1 km2, not square degrees



def test_max_chunks_protection_refuses_canonical_dir(build, tmp_path, monkeypatch):
    """run_build with max_chunks must refuse to write to the canonical impact dir."""
    import impact.config as _cfg

    canonical = Path(_cfg.IMPACT_DATA_DIR).resolve()
    # We only test the guard; no real sources needed
    with pytest.raises(build.BuildError, match="--max-chunks"):
        build.run_build(
            raw_dir=tmp_path / "raw",
            out_dir=canonical,
            checksums_path=tmp_path / "c.json",
            max_chunks=2,
        )


def test_max_chunks_protection_allows_alternate_dir(build, tmp_path, monkeypatch):
    """run_build with max_chunks must NOT raise if out_dir differs from canonical."""
    # The build will fail at resolve_required_sources (no files), not at the guard.
    with pytest.raises((build.BuildError, build.MissingSourceError)) as exc_info:
        build.run_build(
            raw_dir=tmp_path / "raw",
            out_dir=tmp_path / "output",   # not the canonical dir
            checksums_path=tmp_path / "c.json",
            max_chunks=2,
        )
    # Must NOT be the max-chunks guard error
    assert "--max-chunks" not in str(exc_info.value)


def test_split_excluded_cells_drops_pop_status_and_records_reason(build):
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "lat_cell": [32.0, 32.0, 32.5],
            "lon_cell": [-1.875, -1.25, -1.875],
            "pop_cell": [1.0, 2.0, 3.0],
            "pop_status": ["ok", "partial_coverage", "not_covered"],
        }
    )
    kept, excluded = build.split_excluded_cells(frame)

    assert list(kept["cell_id"]) == ["a"]
    assert "pop_status" not in kept.columns
    assert set(excluded["cell_id"]) == {"b", "c"}
    assert list(excluded.columns) == [
        "cell_id",
        "lat_cell",
        "lon_cell",
        "pop_status",
        "reason",
    ]
    assert set(excluded["reason"]) == {build.EXCLUDED_CELLS_REASON}


def test_split_excluded_cells_requires_pop_status(build):
    with pytest.raises(build.BuildError, match="pop_status"):
        build.split_excluded_cells(
            pd.DataFrame({"cell_id": ["a"], "pop_cell": [1.0]})
        )


def test_filter_cells_and_coordinates_remove_only_excluded_rows(build):
    cells = _cells()
    excluded = pd.DataFrame(
        {
            "cell_id": [cells.loc[1, "cell_id"]],
            "lat_cell": [cells.loc[1, "lat_cell"]],
            "lon_cell": [cells.loc[1, "lon_cell"]],
            "pop_status": ["not_covered"],
            "reason": [build.EXCLUDED_CELLS_REASON],
        }
    )
    ids = build.excluded_cell_ids(excluded)

    by_id = build.filter_cells(_population(cells), ids)
    assert len(by_id) == len(cells) - 1
    assert cells.loc[1, "cell_id"] not in set(by_id["cell_id"])

    admin = cells[["lat_cell", "lon_cell"]].copy()
    by_coords = build.filter_cells_by_coordinates(admin, excluded)
    assert len(by_coords) == len(cells) - 1
    assert not (
        (by_coords["lat_cell"] == cells.loc[1, "lat_cell"])
        & (by_coords["lon_cell"] == cells.loc[1, "lon_cell"])
    ).any()

    assert build.filter_cells(None, ids) is None
    assert build.excluded_cell_ids(None) == set()


def test_run_build_excludes_cells_outside_worldpop_coverage(
    build, tmp_path, monkeypatch
):
    """Uncovered/partly covered cells leave every artifact and the thresholds."""
    raw = tmp_path / "raw"
    osm = raw / build.OSM_FOLDER
    osm.mkdir(parents=True)
    for name in build.expected_worldcover_files():
        (raw / name).write_bytes(name.encode("utf-8"))
    (raw / build.WORLDPOP_FILENAME).write_bytes(b"population")
    for name in build.OSM_REQUIRED:
        (osm / name).write_bytes(name.encode("utf-8"))

    checksums_path = raw / build.CHECKSUM_FILENAME
    sources = build.resolve_required_sources(raw)
    build.write_checksums(raw, sources.all_files(), checksums_path)

    cells = build.cells_from_panel(
        pd.DataFrame(
            {
                "lat_cell": [32.0, 32.0, 32.5],
                "lon_cell": [-1.875, -1.25, -1.875],
            }
        )
    )
    # cells_from_panel sorts by latitude then longitude.
    kept_id = cells.loc[0, "cell_id"]
    excluded_ids = set(cells.loc[1:, "cell_id"])
    statuses = ["ok", "partial_coverage", "not_covered"]

    population = cells.copy()
    population["pop_cell"] = [1_000.0, 500.0, np.nan]
    population["pop_5km"] = [5_000.0, 2_000.0, np.nan]
    population["pop_10km"] = [15_000.0, 6_000.0, np.nan]
    population["pop_25km"] = [80_000.0, 20_000.0, np.nan]
    population["pop_status"] = statuses
    population.attrs["raster_bounds"] = [-2.0, 31.0, 1.0, 37.095]

    ring_rows = []
    for cell in cells.itertuples():
        for radius in config.RING_DISTANCES_KM:
            ring_rows.append(
                {
                    "lat_cell": cell.lat_cell,
                    "lon_cell": cell.lon_cell,
                    "ring_km": float(radius),
                    "population": 1_000.0 * radius,
                }
            )
    population_rings = pd.DataFrame(ring_rows)

    roads = _infrastructure(cells)[["cell_id", "road_km_total"]].copy()
    for column in build.ROAD_CLASS_COLUMNS.values():
        roads[column] = 1.0
    buildings = _infrastructure(cells)[
        ["cell_id", "building_count", "building_footprint_km2"]
    ].copy()
    critical_counts = _infrastructure(cells)[["cell_id", "critical_count_total"]].copy()
    for column in build.CRITICAL_CLASS_COLUMNS.values():
        critical_counts[column] = 1.0

    settlements = pd.DataFrame(
        {
            "lat_cell": cells["lat_cell"].to_numpy(),
            "lon_cell": cells["lon_cell"].to_numpy(),
            "settlement_name": ["A", "B", "C"],
            "population": [100.0, 200.0, 300.0],
            "distance_km": [0.0, 1.0, 2.0],
            "cell_id": cells["cell_id"].to_numpy(),
        }
    )
    settlement_counts = pd.DataFrame(
        {
            "cell_id": cells["cell_id"].to_numpy(),
            "settlement_count": [1.0, 1.0, 1.0],
            "settlement_population": [100.0, 200.0, 300.0],
        }
    )
    assets = pd.DataFrame(
        {
            "lat_cell": cells["lat_cell"].to_numpy(),
            "lon_cell": cells["lon_cell"].to_numpy(),
            "category": ["hospital", "school", "power"],
            "asset_name": ["Clinic A", "School B", "Plant C"],
            "cell_id": cells["cell_id"].to_numpy(),
        }
    )

    monkeypatch.setattr(build, "cells_for_build", lambda **_: cells.copy())
    monkeypatch.setattr(
        build, "read_worldcover_areas", lambda *_, **__: _landcover(cells)
    )
    monkeypatch.setattr(
        build, "read_worldpop_cell_totals", lambda *_, **__: population
    )
    monkeypatch.setattr(
        build, "read_worldpop_ring_totals", lambda *_, **__: population_rings
    )
    monkeypatch.setattr(build, "read_osm_roads", lambda *_, **__: roads)
    monkeypatch.setattr(build, "read_osm_buildings", lambda *_, **__: (buildings, None))
    monkeypatch.setattr(
        build, "read_osm_pois", lambda *_, **__: (assets, critical_counts)
    )
    monkeypatch.setattr(
        build,
        "read_osm_settlements",
        lambda *_, **__: (settlements, settlement_counts),
    )
    monkeypatch.setattr(
        build,
        "_read_vector",
        lambda *_, **__: pd.DataFrame({"fclass": pd.Series([], dtype=object)}),
    )
    monkeypatch.setattr(
        build,
        "summarize_vector_rings",
        lambda frame, cells, distances, kind, **__: pd.DataFrame(),
    )

    captured: dict[str, int] = {}
    real_freeze = build.freeze_tier_thresholds

    def _spy(values, quantiles=(0.20, 0.40, 0.60, 0.80)):
        captured["count"] = len(list(values))
        return real_freeze(values, quantiles)

    monkeypatch.setattr(build, "freeze_tier_thresholds", _spy)

    out = tmp_path / "out"
    manifest = build.run_build(
        raw_dir=raw,
        out_dir=out,
        checksums_path=checksums_path,
        with_rings=True,
    )

    excluded_info = manifest["excluded_cells"]
    assert excluded_info["count"] == 2
    assert set(excluded_info["cells"]) == excluded_ids
    assert excluded_info["reason"] == build.EXCLUDED_CELLS_REASON
    assert excluded_info["raster_bounds"] == [-2.0, 31.0, 1.0, 37.095]
    assert excluded_info["nodata_rule"] == build.WORLDPOP_NODATA_RULE

    study_area = manifest["study_area"]
    assert study_area["cell_count"] == 1
    assert study_area["total_cell_count"] == 3
    assert study_area["excluded_cell_count"] == 2

    # Thresholds are frozen from the one remaining cell only.
    assert captured["count"] == 1

    excluded_frame = pd.read_parquet(out / build.EXCLUDED_CELLS_FILENAME)
    assert len(excluded_frame) == 2
    assert set(excluded_frame["cell_id"]) == excluded_ids
    assert set(excluded_frame["pop_status"]) == {
        "partial_coverage",
        "not_covered",
    }
    assert set(excluded_frame["reason"]) == {build.EXCLUDED_CELLS_REASON}

    cell_exposure = pd.read_parquet(out / config.CELL_EXPOSURE_PATH.name)
    assert len(cell_exposure) == 1
    assert cell_exposure.loc[0, "cell_id"] == kept_id
    assert "pop_status" not in cell_exposure.columns

    rings = pd.read_parquet(out / config.RING_EXPOSURE_PATH.name)
    assert len(rings) == len(config.RING_DISTANCES_KM)
    assert set(rings["cell_id"]) == {kept_id}

    settlement_artifact = pd.read_parquet(out / config.SETTLEMENTS_PATH.name)
    assert len(settlement_artifact) == 1
    assert list(settlement_artifact["settlement_name"]) == ["A"]

    infrastructure_artifact = pd.read_parquet(
        out / config.INFRASTRUCTURE_PATH.name
    )
    assert len(infrastructure_artifact) == 1
    assert list(infrastructure_artifact["asset_name"]) == ["Clinic A"]


def test_read_osm_roads_accumulates_chunks(build, monkeypatch, tmp_path):
    """read_osm_roads must accumulate counts across multiple chunks."""
    import types
    from shapely.geometry import LineString
    import geopandas as gpd

    cells = _cells()
    # Two chunks of 1 road each landing in the first cell
    chunk_calls = [0]

    def _fake_pyogrio():
        mod = types.SimpleNamespace()
        mod.read_info = lambda path: {"features": 2, "fields": ["fclass"]}

        def read_dataframe(path, skip_features=0, max_features=None, **kwargs):
            # Each chunk: one road near cell centre (32.0, -1.875)
            line = LineString([(-1.876, 32.0), (-1.874, 32.0)])  # ~200 m
            return gpd.GeoDataFrame({"fclass": ["primary"], "geometry": [line]}, crs=4326)

        mod.read_dataframe = read_dataframe
        return mod

    monkeypatch.setattr(build, "_pyogrio", _fake_pyogrio)

    result = build.read_osm_roads(
        tmp_path / "roads.shp", cells, chunk_size=1, max_chunks=2
    )
    # Both chunks should have been accumulated into road_km_primary
    assert result["road_km_primary"].sum() > 0.0, "chunks not accumulated"
    assert result["road_km_total"].sum() > 0.0
