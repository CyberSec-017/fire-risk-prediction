"""Unit tests for the pure helpers in ``scripts/validate_impact_layers.py``.

Everything here is synthetic: small frames and temporary parquet files.  The
validator reads only derived artifacts, so no geospatial dependency is needed.
"""

from __future__ import annotations

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

from impact import config, exposure  # noqa: E402

VALIDATE_PATH = ROOT / "scripts" / "validate_impact_layers.py"


@pytest.fixture(scope="module")
def validate():
    spec = importlib.util.spec_from_file_location(
        "impact_validate_script", VALIDATE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Synthetic artifact builders
# ---------------------------------------------------------------------------
COORDINATES = [(32.0, -1.875), (32.0, -1.25), (32.5, -1.875)]

LC_FRACTIONS = {
    "lc_tree_km2": 0.40,
    "lc_shrub_km2": 0.20,
    "lc_grass_km2": 0.15,
    "lc_crop_km2": 0.10,
    "lc_bare_km2": 0.07,
    "lc_built_km2": 0.04,
    "lc_water_km2": 0.03,
    "lc_other_km2": 0.01,
}


def _cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [
                f"lat{lat:.3f}_lon{lon:.3f}" for lat, lon in COORDINATES
            ],
            "lat_cell": [lat for lat, _ in COORDINATES],
            "lon_cell": [lon for _, lon in COORDINATES],
            "cell_area_km2": [
                exposure.cell_area_km2(lat, lon) for lat, lon in COORDINATES
            ],
        }
    )


def _exposure_frame(with_ghsl: bool = False) -> pd.DataFrame:
    cells = _cells()
    frame = cells.copy()
    for column, fraction in LC_FRACTIONS.items():
        frame[column] = cells["cell_area_km2"].to_numpy() * fraction
    frame["pop_cell"] = [1_000.0, 4_000.0, 2_500.0]
    frame["pop_5km"] = [5_000.0, 20_000.0, 12_000.0]
    frame["pop_10km"] = [15_000.0, 60_000.0, 30_000.0]
    frame["pop_25km"] = [80_000.0, 250_000.0, 150_000.0]
    frame["road_km_total"] = [10.0, 40.0, 25.0]
    frame["building_count"] = [100.0, 800.0, 300.0]
    frame["building_footprint_km2"] = [0.4, 2.5, 1.1]
    frame["critical_count_total"] = [2.0, 9.0, 4.0]
    if with_ghsl:
        frame["built_surface_km2"] = [10.0, 1.0, 1.1]
    return frame


def _ring_frame() -> pd.DataFrame:
    cells = _cells()
    rows = []
    for cell in cells.itertuples():
        for radius in config.RING_DISTANCES_KM:
            rows.append(
                {
                    "lat_cell": cell.lat_cell,
                    "lon_cell": cell.lon_cell,
                    "ring_km": float(radius),
                    "cell_area_km2": cell.cell_area_km2,
                    "population": 1_000.0 * radius,
                    "road_km": 10.0,
                    "building_count": 100.0,
                    "building_footprint_km2": 0.4,
                    "critical_count": 2.0,
                }
            )
    return pd.DataFrame(rows)


def _settlement_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat_cell": [lat for lat, _ in COORDINATES],
            "lon_cell": [lon for _, lon in COORDINATES],
            "settlement_name": ["A", "B", "C"],
            "population": [100.0, 200.0, 300.0],
            "distance_km": [0.0, 0.5, 1.0],
        }
    )


def _asset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat_cell": [lat for lat, _ in COORDINATES],
            "lon_cell": [lon for _, lon in COORDINATES],
            "category": ["hospital", "school", "power"],
            "asset_name": ["Clinic A", "School B", "Substation C"],
        }
    )


def _write_artifacts(directory: Path, with_ghsl: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _exposure_frame(with_ghsl=with_ghsl).to_parquet(
        directory / config.CELL_EXPOSURE_PATH.name, index=False
    )
    _ring_frame().to_parquet(directory / config.RING_EXPOSURE_PATH.name, index=False)
    _settlement_frame().to_parquet(
        directory / config.SETTLEMENTS_PATH.name, index=False
    )
    _asset_frame().to_parquet(
        directory / config.INFRASTRUCTURE_PATH.name, index=False
    )


def _codes(issues) -> list[str]:
    return [issue.code for issue in issues]


# ---------------------------------------------------------------------------
# Cell areas
# ---------------------------------------------------------------------------
def test_validate_cell_areas_accepts_consistent_frame(validate):
    frame = _exposure_frame()
    issues, summary = validate.validate_cell_areas(frame)
    assert issues == []
    assert summary["cells"] == len(frame)
    assert summary["total_area_km2"] == pytest.approx(
        summary["expected_total_area_km2"]
    )
    assert summary["max_abs_error_km2"] == pytest.approx(0.0)


def test_validate_cell_areas_detects_tampering(validate):
    frame = _exposure_frame()
    frame.loc[1, config.CELL_AREA_COLUMN] = (
        frame.loc[1, config.CELL_AREA_COLUMN] + 5.0
    )
    issues, summary = validate.validate_cell_areas(frame)
    assert issues and issues[0].code == "cell_area_mismatch"
    assert issues[0].cell_id == frame.loc[1, "cell_id"]
    assert summary["max_abs_error_km2"] > 1.0


def test_validate_cell_areas_reports_missing_columns(validate):
    frame = _exposure_frame().drop(columns=[config.CELL_AREA_COLUMN])
    issues, summary = validate.validate_cell_areas(frame)
    assert _codes(issues) == ["missing_columns"]
    assert summary == {}


# ---------------------------------------------------------------------------
# Land-cover fractions
# ---------------------------------------------------------------------------
def test_landcover_fractions_sum_to_one(validate):
    issues, summary = validate.validate_landcover_fractions(_exposure_frame())
    assert issues == []
    assert summary["min_fraction_sum"] == pytest.approx(1.0, abs=1e-6)
    assert summary["max_fraction_sum"] == pytest.approx(1.0, abs=1e-6)


def test_landcover_fractions_detect_incomplete_coverage(validate):
    frame = _exposure_frame()
    for column in LC_FRACTIONS:
        frame[column] = frame[column] * 0.5
    issues, summary = validate.validate_landcover_fractions(frame)
    assert _codes(issues).count("landcover_fraction_below_one") == len(frame)
    assert summary["min_fraction_sum"] == pytest.approx(0.5, abs=1e-6)


def test_landcover_fractions_detect_over_coverage(validate):
    frame = _exposure_frame()
    for column in LC_FRACTIONS:
        frame[column] = frame[column] * 1.5
    issues, _ = validate.validate_landcover_fractions(frame)
    assert _codes(issues).count("landcover_fraction_above_one") == len(frame)

    single = _exposure_frame()
    single.loc[0, "lc_tree_km2"] = single.loc[0, config.CELL_AREA_COLUMN] * 1.2
    issues, _ = validate.validate_landcover_fractions(single)
    assert "landcover_class_above_one" in _codes(issues)


def test_landcover_rejects_negative_areas(validate):
    frame = _exposure_frame()
    frame.loc[0, "lc_tree_km2"] = -1.0
    issues, _ = validate.validate_landcover_fractions(frame)
    assert "negative_landcover_area" in _codes(issues)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
def test_population_plausibility_accepts_normal_values(validate):
    issues, summary = validate.validate_population(_exposure_frame())
    assert issues == []
    assert summary["cells"] == 3
    assert summary["non_monotonic_cells"] == 0
    assert summary["implausible_density_cells"] == 0


def test_population_rejects_negative_and_missing(validate):
    frame = _exposure_frame()
    frame.loc[0, "pop_cell"] = -10.0
    issues, _ = validate.validate_population(frame)
    assert "negative_population" in _codes(issues)

    frame = _exposure_frame()
    frame.loc[1, "pop_cell"] = np.nan
    issues, summary = validate.validate_population(frame)
    assert _codes(issues) == ["invalid_population_values"]
    assert summary == {}


def test_population_flags_implausible_density(validate):
    frame = _exposure_frame()
    frame.loc[0, "pop_cell"] = 1e12
    issues, summary = validate.validate_population(frame)
    assert "implausible_population_density" in _codes(issues)
    assert summary["implausible_density_cells"] == 1


def test_population_flags_non_monotonic_rings(validate):
    frame = _exposure_frame()
    frame.loc[0, "pop_10km"] = 1.0
    issues, summary = validate.validate_population(frame)
    assert "non_monotonic_rings" in _codes(issues)
    assert summary["non_monotonic_cells"] == 1


# ---------------------------------------------------------------------------
# OSM vs GHSL diagnostic
# ---------------------------------------------------------------------------
def test_ghsl_diagnostic_reports_unavailable(validate):
    issues, diagnostic, summary = validate.ghsl_osm_diagnostics(_exposure_frame())
    assert _codes(issues) == ["ghsl_unavailable"]
    assert diagnostic.empty
    assert summary == {"available": False}


def test_ghsl_diagnostic_flags_coverage_problems(validate):
    frame = _exposure_frame(with_ghsl=True)
    issues, diagnostic, summary = validate.ghsl_osm_diagnostics(frame)
    assert summary["available"] is True
    assert summary["low_coverage_cells"] == 1
    assert summary["above_ghsl_cells"] == 1
    assert diagnostic.loc[0, "diagnostic"] == "osm_low_coverage"
    assert diagnostic.loc[1, "diagnostic"] == "osm_above_ghsl"
    assert diagnostic.loc[2, "diagnostic"] == "ok"
    assert "osm_low_coverage" in _codes(issues)
    assert "osm_above_ghsl" in _codes(issues)
    assert diagnostic["osm_to_ghsl_ratio"].notna().all()


def test_ghsl_diagnostic_flags_zero_ghsl_with_osm(validate):
    frame = _exposure_frame(with_ghsl=True)
    frame.loc[0, "built_surface_km2"] = 0.0
    issues, diagnostic, _ = validate.ghsl_osm_diagnostics(frame)
    assert diagnostic.loc[0, "diagnostic"] == "ghsl_zero_osm_positive"
    assert "ghsl_zero_osm_positive" in _codes(issues)


# ---------------------------------------------------------------------------
# Cross-file checks
# ---------------------------------------------------------------------------
def test_cross_file_accepts_consistent_artifacts(validate):
    issues, summary = validate.validate_cross_file(
        _exposure_frame(), _ring_frame(), _settlement_frame(), _asset_frame()
    )
    assert issues == []
    assert summary["rows"] == len(_ring_frame())
    assert summary["distances_km"] == sorted(config.RING_DISTANCES_KM)


def test_cross_file_rejects_points_outside_study_area(validate):
    settlements = _settlement_frame()
    settlements.loc[0, "lat_cell"] = 45.0
    issues, _ = validate.validate_cross_file(_exposure_frame(), None, settlements)
    assert "points_outside_study_area" in _codes(issues)


def test_cross_file_rejects_points_not_on_known_cells(validate):
    assets = _asset_frame()
    assets.loc[0, "lat_cell"] = 33.0
    issues, _ = validate.validate_cross_file(_exposure_frame(), None, None, assets)
    assert "points_not_on_cells" in _codes(issues)


def test_cross_file_rejects_invalid_ring_rows(validate):
    rings = _ring_frame()
    rings.loc[0, "ring_km"] = 7.0
    rings.loc[1, "road_km"] = -1.0
    issues, _ = validate.validate_cross_file(_exposure_frame(), rings)
    assert "invalid_ring_distance" in _codes(issues)
    assert "negative_ring_values" in _codes(issues)


def test_cross_file_rejects_negative_settlement_distance(validate):
    settlements = _settlement_frame()
    settlements.loc[0, "distance_km"] = -2.0
    issues, _ = validate.validate_cross_file(_exposure_frame(), None, settlements)
    assert "invalid_settlement_distance" in _codes(issues)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def test_clean_json_converts_non_finite_values(validate):
    cleaned = validate._clean_json(
        {
            "nan": np.nan,
            "float": np.float64(1.5),
            "items": [np.int64(2), np.nan],
            "flag": np.True_,
            "text": "ok",
        }
    )
    assert cleaned == {"nan": None, "float": 1.5, "items": [2, None], "flag": True, "text": "ok"}


def test_run_validation_reports_missing_artifacts(validate, tmp_path):
    report = validate.run_validation(tmp_path, write_report=False)
    assert report["ok"] is False
    codes = [issue["code"] for issue in report["issues"]]
    assert codes.count("artifact_missing") >= 1
    assert "admin_units_unavailable" in codes
    assert not (tmp_path / validate.REPORT_FILENAME).exists()


def test_run_validation_passes_clean_artifacts(validate, tmp_path):
    _write_artifacts(tmp_path)
    report = validate.run_validation(tmp_path, write_report=True)
    assert report["ok"] is True
    assert report["counts"]["error"] == 0
    assert report["summaries"]["cell_areas"]["cells"] == 3
    assert report["summaries"]["landcover"]["max_fraction_sum"] == pytest.approx(
        1.0, abs=1e-6
    )
    assert report["summaries"]["cross_file"]["rows"] == len(_ring_frame())
    report_path = tmp_path / validate.REPORT_FILENAME
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["schema_version"] == "1.0"


def test_run_validation_detects_inconsistent_artifact(validate, tmp_path):
    _write_artifacts(tmp_path)
    frame = _exposure_frame()
    frame.loc[0, config.CELL_AREA_COLUMN] = frame.loc[0, config.CELL_AREA_COLUMN] * 2
    frame.to_parquet(tmp_path / config.CELL_EXPOSURE_PATH.name, index=False)
    report = validate.run_validation(tmp_path)
    assert report["ok"] is False
    assert any(
        issue["code"] == "cell_area_mismatch" for issue in report["issues"]
    )


def test_cli_exit_codes(validate, tmp_path, capsys):
    assert validate.main(["--impact-dir", str(tmp_path)]) == 1

    clean = tmp_path / "clean"
    _write_artifacts(clean)
    assert validate.main(["--impact-dir", str(clean)]) == 0
    # The missing optional GHSL column is a warning, not an error.
    assert validate.main(["--impact-dir", str(clean), "--fail-on-warning"]) == 1
    captured = capsys.readouterr()
    assert "error(s)" in captured.out
