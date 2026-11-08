"""Regression tests for the data pipeline.

Run with:
    python -m pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The pipeline scripts use ``from config import ...`` and are run as
# ``python src/<script>.py``, so make src/ importable for the tests too.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import build_panel  # noqa: E402
import clean_firms  # noqa: E402
import config  # noqa: E402


# ---------------------------------------------------------------------------
# clean_firms.py
# ---------------------------------------------------------------------------

def test_filter_quality_keeps_viirs_nominal_and_high():
    df = pd.DataFrame({
        "type": [0, 0, 0, 2],
        "confidence": ["n", "h", "l", "n"],
    })

    out = clean_firms.filter_quality(df)

    assert len(out) == 2
    assert list(out["confidence"]) == ["n", "h"]


def test_filter_quality_keeps_numeric_confidence_at_least_50():
    df = pd.DataFrame({
        "type": [0, 0, 0],
        "confidence": [80, 20, 55],
    })

    out = clean_firms.filter_quality(df)

    assert sorted(out["confidence"]) == [55, 80]


def test_filter_quality_handles_text_typed_type_column():
    df = pd.DataFrame({
        "type": ["0", "2", "0"],
        "confidence": ["n", "n", "h"],
    })

    out = clean_firms.filter_quality(df)

    assert list(out["type"]) == ["0", "0"]


def test_snap_to_grid_uses_power_cell_size():
    df = pd.DataFrame({
        "latitude": [32.1],
        "longitude": [-1.9],
        "acq_date": ["2020-01-01"],
    })

    out = clean_firms.snap_to_grid(df)

    assert out.loc[0, "lat_cell"] == pytest.approx(32.0)
    assert out.loc[0, "lon_cell"] == pytest.approx(-1.875)


def test_filter_study_area_drops_outside_box_and_years():
    df = pd.DataFrame({
        "latitude": [33.0, 50.0, 33.0],
        "longitude": [0.0, 0.0, 0.0],
        "acq_date": ["2020-01-01", "2020-01-01", "2030-01-01"],
    })

    out = clean_firms.filter_study_area(df)

    assert len(out) == 1
    assert out.iloc[0]["acq_date"] == "2020-01-01"


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

def test_cache_path_is_shared_by_fetch_and_build():
    path = config.power_cache_path("T2M", 2020, 1)

    assert path.name == "T2M_2020_tile1.csv"
    assert path.parent == config.POWER_CACHE


def test_generate_tiles_covers_the_study_box():
    tiles = config.generate_tiles()

    assert tiles
    for lat_min, lat_max, lon_min, lon_max in tiles:
        assert lat_max - lat_min <= config.MAX_TILE_DEG + 1e-9
        assert lon_max - lon_min <= config.MAX_TILE_DEG + 1e-9

    assert min(t[0] for t in tiles) == pytest.approx(config.LAT_MIN)
    assert max(t[1] for t in tiles) == pytest.approx(config.LAT_MAX)
    assert min(t[2] for t in tiles) == pytest.approx(config.LON_MIN)
    assert max(t[3] for t in tiles) == pytest.approx(config.LON_MAX)


# ---------------------------------------------------------------------------
# build_panel.py
# ---------------------------------------------------------------------------

def test_parse_power_csv_rejects_a_file_without_a_power_header(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("LAT,LON\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="END HEADER"):
        build_panel._parse_power_csv(bad, "T2M")


def test_add_features_uses_backward_windows_only():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "lat_cell": [32.0] * 10,
        "lon_cell": [0.0] * 10,
        "date": dates,
        "PRECTOTCORR": [0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
        "T2M_MAX": list(range(10)),
        "RH2M": [50, 40, 30, 20, 10, 20, 30, 40, 50, 60],
    })

    out = build_panel.add_features(df)

    # 7-day windows ending on index 2 only see indices 0..2.
    assert out.loc[2, "rain_7d"] == 5
    assert out.loc[2, "tmax_7d"] == pytest.approx(1.0)
    assert out.loc[2, "rhmin_7d"] == 30

    # dry_days = days since the last rain > 1 mm (0 on the wet day).
    assert out.loc[2, "dry_days"] == 0
    assert out.loc[3, "dry_days"] == 1
    assert out.loc[9, "dry_days"] == 7

    expected_sin = np.sin(2 * np.pi * 1 / 365.25)
    assert out.loc[0, "doy_sin"] == pytest.approx(expected_sin)
