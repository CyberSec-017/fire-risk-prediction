"""Integrity tests for the impact module.

These tests protect four things:

1. the trained model artifacts and 20-feature contract are unchanged;
2. the impact module has no dependency on the model/UI code (and vice versa);
3. its grid and paths match the rest of the project;
4. missing data is reported explicitly, never silently filled with zeros.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from impact import config, data  # noqa: E402

IMPACT_DIR = ROOT / "impact"

# v0.1.0 baseline.  A change here means the model artifacts were modified.
MODEL_SHA256 = "d816d2e211cefe06c89512a0bba7dcf4215ab11885f8a84dcd3a3becdf01b2ab"
MODEL_CARD_SHA256 = "8b549faae5ab93717680278c13c7288adbaebf7b94444fca655724c7884b7852"
MODEL_FEATURE_HASH = "63e5895979fa4521f7dd650ea1e82aff8e553265b9c89ffeeceb8178629a56da"
MODEL_FEATURES = (
    "lat_cell",
    "lon_cell",
    "T2M",
    "T2M_MAX",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
    "GWETROOT",
    "rain_7d",
    "tmax_7d",
    "rhmin_7d",
    "rain_14d",
    "tmax_14d",
    "rhmin_14d",
    "rain_30d",
    "tmax_30d",
    "rhmin_30d",
    "dry_days",
    "doy_sin",
    "doy_cos",
)

FORBIDDEN_IMPACT_IMPORTS = ("app_core", "dashboard", "streamlit")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def imported_modules(path: Path) -> set[str]:
    """Return every imported module (and imported name) in a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                modules.add(module)
            for alias in node.names:
                modules.add(f"{module}.{alias.name}" if module else alias.name)
    return modules


# ---------------------------------------------------------------------------
# Model artifacts and contract
# ---------------------------------------------------------------------------
def test_model_artifact_hashes_unchanged():
    assert sha256(ROOT / "models" / "fire_model.joblib") == MODEL_SHA256
    assert sha256(ROOT / "models" / "model_card.json") == MODEL_CARD_SHA256


def test_model_card_contract_unchanged():
    card = json.loads((ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))
    assert card["feature_hash"] == MODEL_FEATURE_HASH
    assert card["n_features"] == 20
    assert tuple(card["feature_names"]) == MODEL_FEATURES
    assert card["risk_bins"] == [0, 0.005, 0.02, 0.06, 1.0]


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------
def test_impact_modules_do_not_import_model_or_ui_code():
    for path in sorted(IMPACT_DIR.glob("*.py")):
        modules = imported_modules(path)
        for module in modules:
            root = module.split(".")[0]
            assert root not in FORBIDDEN_IMPACT_IMPORTS, (
                f"{path.name} imports {module!r}; the impact module must stay "
                "independent of app_core, dashboard and Streamlit."
            )


def test_impact_modules_do_not_reference_dashboard_config_years():
    for path in sorted(IMPACT_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "START_YEAR" not in text, path.name
        assert "END_YEAR" not in text, path.name


def test_model_and_loader_do_not_import_impact():
    for relative in ("app_core.py", "dashboard/model_loader.py"):
        modules = imported_modules(ROOT / relative)
        offenders = [
            module
            for module in modules
            if module.split(".")[0] == "impact"
        ]
        assert not offenders, f"{relative} imports {offenders}; the model path must stay independent."


# ---------------------------------------------------------------------------
# Grid / path alignment with the rest of the project
# ---------------------------------------------------------------------------
def test_grid_constants_match_dashboard_and_model():
    import app_core
    from dashboard import config as dashboard_config

    assert config.MODEL_LAT_MIN == app_core.MODEL_LAT_MIN == dashboard_config.LAT_MIN
    assert config.MODEL_LAT_MAX == app_core.MODEL_LAT_MAX == dashboard_config.LAT_MAX
    assert config.MODEL_LON_MIN == app_core.MODEL_LON_MIN == dashboard_config.LON_MIN
    assert config.MODEL_LON_MAX == app_core.MODEL_LON_MAX == dashboard_config.LON_MAX
    assert config.MODEL_LAT_STEP == app_core.MODEL_LAT_STEP == dashboard_config.LAT_STEP
    assert config.MODEL_LON_STEP == app_core.MODEL_LON_STEP == dashboard_config.LON_STEP


def test_panel_path_matches_dashboard_config():
    from dashboard import config as dashboard_config

    assert config.FIRE_PANEL_PATH == dashboard_config.FIRE_PANEL_PATH
    assert config.IMPACT_DATA_DIR == ROOT / "data" / "impact"


# ---------------------------------------------------------------------------
# Date range comes from the panel, not from dashboard configuration
# ---------------------------------------------------------------------------
def test_panel_date_range_uses_panel_dates(tmp_path):
    path = tmp_path / "panel.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-05-01", "2020-09-30", "2019-05-01"]),
            "lat_cell": [36.0, 36.5, 36.0],
            "lon_cell": [3.0, 3.625, 3.0],
        }
    ).to_parquet(path, index=False)

    result = data.load_panel_date_range(path)
    assert result.available
    assert result.start == pd.Timestamp("2019-05-01")
    assert result.end == pd.Timestamp("2020-09-30")
    assert len(result.dates) == 2
    # dashboard.config.START_YEAR/END_YEAR are 2024/2026; the data wins.
    assert (result.start.year, result.end.year) == (2019, 2020)


def test_panel_date_range_missing_file_is_explicit(tmp_path):
    result = data.load_panel_date_range(tmp_path / "absent.parquet")
    assert result.status == config.STATUS_MISSING
    assert result.start is None and result.end is None
    assert "does not exist" in result.message


def test_date_range_from_panel_handles_missing_dates():
    assert data.date_range_from_panel(None) is None
    assert data.date_range_from_panel(pd.DataFrame({"lat_cell": [1.0]})) is None
    assert data.date_range_from_panel(pd.DataFrame({"date": [None]})) is None


# ---------------------------------------------------------------------------
# Missing / invalid derived data
# ---------------------------------------------------------------------------
def test_missing_exposure_returns_missing_not_empty_frame(tmp_path):
    result = data.load_cell_exposure(tmp_path / "absent.parquet")
    assert result.status == config.STATUS_MISSING
    assert result.frame is None
    assert result.unavailable


def test_corrupt_parquet_returns_invalid(tmp_path):
    path = tmp_path / "broken.parquet"
    path.write_text("this is not parquet", encoding="utf-8")
    result = data.load_cell_exposure(path)
    assert result.status == config.STATUS_INVALID
    assert result.frame is None


def test_schema_with_missing_columns_is_invalid(tmp_path):
    path = tmp_path / "partial.parquet"
    pd.DataFrame({"lat_cell": [36.0], "lon_cell": [3.0]}).to_parquet(
        path, index=False
    )
    result = data.load_cell_exposure(path)
    assert result.status == config.STATUS_INVALID
    assert result.frame is None
    assert "missing required column" in result.message


def test_burned_area_is_explicitly_unavailable_in_stage_1(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BURNED_AREA_PATH", tmp_path / "burned.parquet")
    result = data.load_burned_area()
    assert result.status == config.STATUS_UNAVAILABLE
    assert result.frame is None
    assert "phase 2" in result.message
    assert "never" in result.message


def test_manifest_missing_invalid_and_valid(tmp_path):
    missing = data.load_manifest(tmp_path / "missing.json")
    assert missing.status == config.STATUS_MISSING
    assert missing.manifest is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    assert data.load_manifest(invalid).status == config.STATUS_INVALID

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"schema_version": "1"}), encoding="utf-8")
    assert data.load_manifest(incomplete).status == config.STATUS_INVALID

    valid = tmp_path / "manifest.json"
    valid.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generated_at": "2026-01-01T00:00:00Z",
                "sources": [
                    {
                        "name": "ESA WorldCover",
                        "version": "2021 v200",
                        "license": "CC BY 4.0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert data.load_manifest(valid).status == config.STATUS_AVAILABLE


def test_availability_report_is_explicit_when_everything_is_missing(
    tmp_path, monkeypatch
):
    for attribute in (
        "CELL_EXPOSURE_PATH",
        "RING_EXPOSURE_PATH",
        "SETTLEMENTS_PATH",
        "INFRASTRUCTURE_PATH",
        "ADMIN_UNITS_PATH",
        "BURNED_AREA_PATH",
        "MANIFEST_PATH",
    ):
        monkeypatch.setattr(config, attribute, tmp_path / f"{attribute.lower()}.parquet")

    rows = data.availability_report()
    assert {row["key"] for row in rows} >= {
        "cell_exposure",
        "burned_area",
        "manifest",
    }
    assert all(not row["available"] for row in rows)
    assert all(
        row["status"] in {config.STATUS_MISSING, config.STATUS_UNAVAILABLE}
        for row in rows
    )


def test_no_silent_zero_fill_in_impact_source():
    for path in sorted(IMPACT_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "fillna(0" not in text, f"{path.name} silently fills missing values with zero"
