"""Pure-logic tests for the interactive EDA transformations and chart inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import eda  # noqa: E402


@pytest.fixture
def panel() -> pd.DataFrame:
    return pd.DataFrame({
        "lat_cell": [36.0, 36.5, 36.0, 36.5, 36.0],
        "lon_cell": [3.0, 3.0, 4.0, 4.0, 3.0],
        "date": pd.to_datetime([
            "2021-01-01", "2021-01-01", "2021-02-03", "2021-02-03", "2021-03-05",
        ]),
        "T2M_MAX": [10.0, 12.0, 30.0, 32.0, np.nan],
        "RH2M": [80.0, 75.0, 20.0, 25.0, 30.0],
        "WS2M": [2.0, 3.0, 8.0, 9.0, 4.0],
        "PRECTOTCORR": [5.0, 4.0, 0.0, 0.0, 1.0],
        "GWETROOT": [0.7, 0.6, 0.2, 0.1, 0.5],
        "burned": [0, 0, 1, 1, 0],
    })


@pytest.fixture
def fires() -> pd.DataFrame:
    return pd.DataFrame({
        "lat_cell": [36.0, 36.5, 36.0, 36.5],
        "lon_cell": [3.0, 3.0, 4.0, 4.0],
        "date": pd.to_datetime([
            "2021-01-01", "2021-02-03", "2021-02-03", "2021-07-14",
        ]),
        "n_detections": [1, 3, 2, 5],
        "total_frp": [0.5, 4.0, 2.0, 10.0],
    })


@pytest.fixture
def raw_firms() -> pd.DataFrame:
    return pd.DataFrame({
        "latitude": ["36.10", "36.70", "36.00", "40.00", None],
        "longitude": [3.0, 4.0, 4.0, 3.0, None],
        "acq_date": ["2021-01-01", "2021-02-03", "2021-02-03", "2021-08-01", None],
        "confidence": ["n", "h", "l", "n", None],
        "frp": [0.5, 12.0, 2.0, 1.0, None],
        "brightness": [300.0, 340.0, 310.0, 305.0, None],
    })


def test_prepare_firms_drops_incomplete_rows_and_labels_confidence(raw_firms):
    clean, stats = eda.prepare_firms(raw_firms)

    assert stats["input_rows"] == 5
    assert stats["valid_rows"] == 4
    assert stats["dropped_rows"] == 1
    assert list(clean["confidence_label"]) == [
        "Nominale (n)", "Élevée (h)", "Faible (l)", "Nominale (n)"
    ]
    assert clean["is_high_confidence"].tolist() == [False, True, False, False]
    assert pd.api.types.is_datetime64_any_dtype(clean["date"])
    assert pd.api.types.is_float_dtype(clean["latitude"])


def test_prepare_firms_reports_missing_values_without_dropping_partial_rows():
    raw = pd.DataFrame({
        "latitude": [36.0, 36.5], "longitude": [3.0, 3.5],
        "date": ["2021-01-01", "2021-01-02"],
        "frp": [1.0, None], "confidence": [None, "h"],
    })
    clean, stats = eda.prepare_firms(raw)

    assert stats["valid_rows"] == 2
    assert stats["missing_by_column"]["frp"] == 1
    assert clean["confidence_label"].tolist() == ["Non renseignée", "Élevée (h)"]


def test_date_and_geographic_coverage(fires):
    dates = eda.date_coverage(fires)
    geo = eda.geographic_coverage(fires)

    assert dates["start"] == pd.Timestamp("2021-01-01")
    assert dates["end"] == pd.Timestamp("2021-07-14")
    assert dates["days"] == 3
    assert dates["years"] == 1
    assert dates["invalid_dates"] == 0
    assert geo["unique_cells"] == 4
    assert (geo["min_lat"], geo["max_lat"]) == (36.0, 36.5)
    assert geo["valid_rows"] == 4
    assert geo["outside_study_area"] == 0


def test_coverage_flags_invalid_dates_and_outside_domain():
    frame = pd.DataFrame({
        "lat_cell": [36.0, 60.0, None],
        "lon_cell": [3.0, 20.0, 3.0],
        "date": ["2021-01-01", "bad", None],
    })
    dates = eda.date_coverage(frame)
    geo = eda.geographic_coverage(frame)

    assert dates["invalid_dates"] == 2
    assert geo["invalid_rows"] == 1
    assert geo["outside_study_area"] == 1
    assert geo["valid_rows"] == 2


@pytest.mark.parametrize(
    ("start", "end", "expected_start", "expected_end"),
    [
        ("2021-01-01", "2021-12-31", pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")),
        ("2021-06-01", "2021-01-01", None, None),
        (None, "2021-01-01", None, pd.Timestamp("2021-01-01")),
    ],
)
def test_normalize_date_range(start, end, expected_start, expected_end):
    resolved_start, resolved_end, note = eda.normalize_date_range(start, end)

    assert resolved_start == expected_start
    assert resolved_end == expected_end
    if expected_start is None and start is not None:
        assert note and "invalide" in note


@pytest.mark.parametrize(
    ("bounds", "expected", "expects_note"),
    [
        ((35.0, 36.0), (35.0, 36.0), False),
        ((30.0, 40.0), eda.STUDY_AREA["latitude"], True),
        ((37.0, 35.0), eda.STUDY_AREA["latitude"], True),
        (None, eda.STUDY_AREA["latitude"], False),
    ],
)
def test_normalize_geo_range(bounds, expected, expects_note):
    resolved, note = eda.normalize_geo_range(bounds, "latitude")

    assert resolved == expected
    assert bool(note) is expects_note


def test_apply_filters_date_and_geography(panel, fires, raw_firms):
    firms, _ = eda.prepare_firms(raw_firms)
    filters = eda.EdaFilters(
        start=pd.Timestamp("2021-02-03"), end=pd.Timestamp("2021-02-03"),
        lat_range=(36.0, 36.0), lon_range=(4.0, 4.0),
    )
    result = eda.apply_filters(panel, fires, firms, filters)

    assert len(result.panel) == 1
    assert len(result.fires) == 1
    assert result.fires.iloc[0]["n_detections"] == 2
    assert len(result.firms) == 1
    assert result.firms.iloc[0]["confidence_label"] == "Faible (l)"
    assert result.filters == filters


def test_apply_filters_invalid_bounds_are_ignored_not_swapped(panel, fires, raw_firms):
    firms, _ = eda.prepare_firms(raw_firms)
    start, end, date_note = eda.normalize_date_range("2021-12-31", "2021-01-01")
    lat_bounds, _ = eda.normalize_geo_range((37.0, 35.0), "latitude")
    lon_bounds, _ = eda.normalize_geo_range((9.0, -2.0), "longitude")
    filters = eda.EdaFilters(
        start=start, end=end, lat_range=lat_bounds, lon_range=lon_bounds,
        notes=(date_note,) if date_note else (),
    )

    assert filters.start is None and filters.end is None
    assert filters.notes
    assert len(eda.apply_filters(panel, fires, firms, filters).fires) == len(fires)
    # One raw row is outside the model domain (lat 40).
    assert len(eda.apply_filters(panel, fires, firms, filters).firms) == 3


def test_empty_frames_produce_well_shaped_chart_inputs():
    empty = pd.DataFrame()
    builders = {
        "monthly_detections": lambda: eda.monthly_detections(empty),
        "seasonal_detections": lambda: eda.seasonal_detections(empty),
        "top_cells": lambda: eda.top_detection_cells(empty),
        "geographic": lambda: eda.detection_scatter(empty),
        "bands": lambda: eda.latitude_band_detections(empty),
        "confidence": lambda: eda.confidence_distribution(empty),
        "frp": lambda: eda.frp_distribution(empty),
        "histogram": lambda: eda.weather_histogram(empty, "T2M_MAX"),
        "summary": lambda: eda.weather_summary(empty),
        "correlation": lambda: eda.correlation_matrix(empty),
        "trend": lambda: eda.monthly_weather_trend(empty, "T2M_MAX"),
        "classes": lambda: eda.detection_rate_by_weather_class(empty, "T2M_MAX"),
        "coverage": lambda: eda.coverage_frame(empty, empty, empty),
    }
    for name, builder in builders.items():
        result = builder()
        assert isinstance(result, pd.DataFrame), name
        # An empty panel has no numeric columns, so no correlation matrix exists.
        assert len(result.columns) > 0 or name == "correlation", name
        assert len(result) == 0 or name in {"bands", "frp", "summary", "coverage"}, name


def test_weather_histogram_ignores_missing_values_and_keeps_bins(panel):
    frame = eda.weather_histogram(panel, "T2M_MAX", bins=5)

    assert len(frame) == 5
    assert frame["without_detection"].sum() + frame["with_detection"].sum() == 4
    assert np.isfinite(frame["bin_center"]).all()
    assert frame["with_detection"].sum() == 2


def test_chart_inputs_summarise_large_dataset_without_exploding():
    n = 200_000
    dates = pd.date_range("2018-01-01", periods=730, freq="D")
    large = pd.DataFrame({
        "lat_cell": np.full(n, 36.5),
        "lon_cell": np.full(n, 4.0),
        "date": np.random.default_rng(0).choice(dates, size=n),
        "n_detections": np.ones(n, dtype=int),
        "T2M_MAX": np.random.default_rng(1).normal(28, 6, n),
        "burned": np.random.default_rng(2).binomial(1, 0.02, n),
    })

    monthly = eda.monthly_detections(large)
    histogram = eda.weather_histogram(large, "T2M_MAX", bins=24)

    assert len(monthly) <= 24
    assert len(histogram) == 24
    assert monthly["detections"].sum() == n
    assert histogram["without_detection"].sum() + histogram["with_detection"].sum() == n


def test_deterministic_sample_is_bounded_and_reproducible(panel):
    sample = eda.deterministic_sample(panel, 3)

    assert len(sample) == 3
    assert sample.equals(eda.deterministic_sample(panel, 3))
    assert eda.deterministic_sample(panel, 50).equals(panel)


def test_detection_rate_by_weather_class_uses_filtered_quantiles(panel):
    frame = eda.detection_rate_by_weather_class(panel, "T2M_MAX", bins=3)

    assert list(frame.columns) == ["weather_class", "fire_rate_pct", "cell_days"]
    assert frame["cell_days"].sum() == 4  # one missing T2M_MAX is excluded
    assert frame["fire_rate_pct"].between(0, 100).all()
    assert not frame["weather_class"].duplicated().any()


def test_correlation_matrix_keeps_fire_proxy_and_finite_values(panel):
    matrix = eda.correlation_matrix(panel)

    assert set(matrix.columns) <= set(panel.select_dtypes(include="number").columns)
    assert "burned" in matrix.columns
    assert matrix.shape[0] == matrix.shape[1]
    assert np.isfinite(matrix.to_numpy()).all()


def test_visualization_registry_has_unique_ids_and_known_sources():
    ids = eda.VISUALIZATION_IDS

    assert len(ids) >= 10
    assert len(ids) == len(set(ids))
    assert {v.source for v in eda.VISUALIZATIONS} <= {"fires", "firms", "panel"}
    assert eda.VISUALIZATION_TITLES["correlation_matrix"] == "Matrice de corrélation"
