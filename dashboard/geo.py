"""Study-area geometry and click-to-predict helpers for the GIS page.

The model's spatial contract is a 0.5° × 0.625° grid inside the northern
Algeria bounding box.  This module adds the missing *polygon* layer: the real
Algeria boundary (Natural Earth 1:50m) intersected with that existing bounding
box, so clicks on the sea, Morocco or Tunisia are rejected without expanding
the model domain.

Everything here is deterministic and offline.  A click is snapped to the
nearest model grid cell, and that cell's existing panel feature row is used
unchanged.  Missing features are reported, never imputed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd

import app_core
from dashboard.config import STUDY_POLYGON_PATH

_EARTH_RADIUS_KM = 6371.0088


class OutsideStudyAreaError(ValueError):
    """The clicked point is outside the model/study-area polygon."""


class FeatureDataUnavailable(ValueError):
    """The selected model cell has no complete feature row for the date."""


@dataclass(frozen=True)
class ResolvedPoint:
    """A clicked point with its nearest model cell and geometry status."""

    lat: float
    lon: float
    lat_cell: float
    lon_cell: float
    distance_km: float
    in_area: bool
    reason: str | None
    in_bbox: bool
    in_polygon: bool


@dataclass(frozen=True)
class FeatureAvailability:
    """Whether a model cell/date has a complete 20-feature row."""

    ok: bool
    missing: tuple[str, ...]
    reason: str | None
    row: pd.Series | None = None


@dataclass(frozen=True)
class PointPrediction:
    """A single selected-point prediction from the real model."""

    probability: float
    risk: str
    lat: float
    lon: float
    lat_cell: float
    lon_cell: float
    distance_km: float
    date: pd.Timestamp
    horizon_days: int


@lru_cache(maxsize=1)
def study_area_polygon() -> tuple[tuple[float, float], ...]:
    """Return the Algeria boundary as an outer ring of ``(lon, lat)`` tuples.

    The GeoJSON is stored in the repository so validation never needs a network
    request.  Only the largest ring is used; Algeria's Natural Earth geometry
    is a single polygon without holes.
    """
    if not STUDY_POLYGON_PATH.exists():
        raise FileNotFoundError(
            f"Study-area polygon is missing: {STUDY_POLYGON_PATH}"
        )

    payload = json.loads(STUDY_POLYGON_PATH.read_text(encoding="utf-8"))
    if payload.get("type") == "Feature":
        payload = payload["geometry"]

    coordinates = payload.get("coordinates") or []
    if not coordinates:
        raise ValueError(f"Study-area polygon has no coordinates: {STUDY_POLYGON_PATH}")

    if payload.get("type") == "MultiPolygon":
        rings = [ring for polygon in coordinates for ring in polygon]
    else:
        rings = list(coordinates)
    ring = max(rings, key=len)
    if len(ring) < 3:
        raise ValueError(f"Study-area polygon is degenerate: {STUDY_POLYGON_PATH}")

    return tuple((float(lon), float(lat)) for lon, lat in ring)


def point_in_polygon(
    lat: float,
    lon: float,
    ring: Sequence[tuple[float, float]] | None = None,
) -> bool:
    """Ray-casting point-in-polygon test.

    ``ring`` contains ``(lon, lat)`` pairs.  The classic even-odd rule is used;
    it works for convex and concave rings and needs no external dependency.
    """
    if ring is None:
        ring = study_area_polygon()
    if len(ring) < 3:
        return False

    x = float(lon)
    y = float(lat)
    inside = False
    previous = len(ring) - 1

    for current in range(len(ring)):
        x1, y1 = ring[previous]
        x2, y2 = ring[current]
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current

    return inside


def point_in_model_bbox(lat: float, lon: float) -> bool:
    """True inside the existing model-domain bounding box."""
    return bool(
        app_core.MODEL_LAT_MIN <= float(lat) <= app_core.MODEL_LAT_MAX
        and app_core.MODEL_LON_MIN <= float(lon) <= app_core.MODEL_LON_MAX
    )


def point_in_study_area(
    lat: float,
    lon: float,
    ring: Sequence[tuple[float, float]] | None = None,
) -> bool:
    """True only on Algerian land inside the existing model domain."""
    if not point_in_model_bbox(lat, lon):
        return False
    return point_in_polygon(lat, lon, ring)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    a = min(max(a, 0.0), 1.0)
    return float(2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a)))


def is_model_cell(lat_cell: float, lon_cell: float) -> bool:
    """True when a cell centre is on the model's training grid and in bounds."""
    lat = float(lat_cell)
    lon = float(lon_cell)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return False

    lat_grid = round(lat / app_core.MODEL_LAT_STEP) * app_core.MODEL_LAT_STEP
    lon_grid = round(lon / app_core.MODEL_LON_STEP) * app_core.MODEL_LON_STEP
    on_grid = math.isclose(lat, lat_grid, abs_tol=1e-6) and math.isclose(
        lon, lon_grid, abs_tol=1e-6
    )
    return bool(on_grid and point_in_model_bbox(lat, lon))


def resolve_point(
    lat: float,
    lon: float,
    ring: Sequence[tuple[float, float]] | None = None,
) -> ResolvedPoint:
    """Snap a click to the nearest model cell and classify its location."""
    lat = float(lat)
    lon = float(lon)

    if not (math.isfinite(lat) and math.isfinite(lon)):
        return ResolvedPoint(
            lat=lat,
            lon=lon,
            lat_cell=float("nan"),
            lon_cell=float("nan"),
            distance_km=float("nan"),
            in_area=False,
            reason="The selected coordinates are not valid numbers.",
            in_bbox=False,
            in_polygon=False,
        )

    lat_cell, lon_cell = app_core.snap_to_grid(lat, lon)
    in_bbox = point_in_model_bbox(lat, lon)
    in_polygon = point_in_polygon(lat, lon, ring)
    in_area = in_bbox and in_polygon

    reason = None
    if not in_bbox:
        reason = (
            "Outside the model domain "
            f"({app_core.MODEL_LAT_MIN}–{app_core.MODEL_LAT_MAX}°N, "
            f"{app_core.MODEL_LON_MIN}–{app_core.MODEL_LON_MAX}°E)."
        )
    elif not in_polygon:
        reason = (
            "Outside Algeria. The study area is restricted to Algerian land "
            "inside the model domain; this point is in the sea or a "
            "neighbouring country."
        )

    return ResolvedPoint(
        lat=lat,
        lon=lon,
        lat_cell=float(lat_cell),
        lon_cell=float(lon_cell),
        distance_km=haversine_km(lat, lon, lat_cell, lon_cell),
        in_area=bool(in_area),
        reason=reason,
        in_bbox=bool(in_bbox),
        in_polygon=bool(in_polygon),
    )


def feature_row_for(
    panel: pd.DataFrame | None,
    date: str | pd.Timestamp,
    lat_cell: float,
    lon_cell: float,
    feature_names: Sequence[str],
) -> FeatureAvailability:
    """Return the complete feature row for one model cell/date.

    A row is accepted only when every card feature is present and finite.
    Nothing is defaulted or imputed.  Missing values are reported with the
    feature names so the UI can explain why prediction is unavailable.
    """
    names = [str(name) for name in feature_names]
    if panel is None or panel.empty:
        return FeatureAvailability(
            ok=False,
            missing=tuple(names),
            reason="The fire panel is unavailable, so feature data cannot be checked.",
        )

    if not is_model_cell(lat_cell, lon_cell):
        return FeatureAvailability(
            ok=False,
            missing=(),
            reason=(
                f"{float(lat_cell):.4f}°N, {float(lon_cell):.4f}°E is not on "
                "the model's 0.5° × 0.625° grid."
            ),
        )

    target = pd.Timestamp(date).normalize()
    frame = panel
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame = frame[dates == target]
    if frame.empty:
        return FeatureAvailability(
            ok=False,
            missing=(),
            reason=(
                f"No panel row exists for grid cell "
                f"{float(lat_cell):.4f}°N, {float(lon_cell):.4f}°E on "
                f"{target:%Y-%m-%d}."
            ),
        )

    missing_columns = [name for name in names if name not in frame.columns]
    if missing_columns:
        return FeatureAvailability(
            ok=False,
            missing=tuple(missing_columns),
            reason=(
                "The fire panel is missing model feature column(s): "
                f"{', '.join(missing_columns)}."
            ),
        )

    lat_ok = np.isclose(
        pd.to_numeric(frame["lat_cell"], errors="coerce"),
        float(lat_cell),
        atol=1e-6,
    )
    lon_ok = np.isclose(
        pd.to_numeric(frame["lon_cell"], errors="coerce"),
        float(lon_cell),
        atol=1e-6,
    )
    matches = frame[lat_ok & lon_ok]
    if matches.empty:
        return FeatureAvailability(
            ok=False,
            missing=(),
            reason=(
                f"No panel row exists for grid cell "
                f"{float(lat_cell):.4f}°N, {float(lon_cell):.4f}°E on "
                f"{target:%Y-%m-%d}."
            ),
        )

    row = matches.iloc[0]
    missing_values = [name for name in names if pd.isna(row[name])]
    if missing_values:
        return FeatureAvailability(
            ok=False,
            missing=tuple(missing_values),
            reason=(
                f"The feature row for grid cell {float(lat_cell):.4f}°N, "
                f"{float(lon_cell):.4f}°E on {target:%Y-%m-%d} is missing "
                f"{', '.join(missing_values)}. Prediction is blocked rather "
                "than filling in missing values. Choose a nearby inland "
                "location or a different anchor date."
            ),
            row=row,
        )

    return FeatureAvailability(ok=True, missing=(), reason=None, row=row)


def predict_point(
    bundle: app_core.ModelBundle,
    panel: pd.DataFrame | None,
    lat: float,
    lon: float,
    date: str | pd.Timestamp,
    *,
    feature_names: Sequence[str] | None = None,
    ring: Sequence[tuple[float, float]] | None = None,
) -> PointPrediction:
    """Predict fire risk for the model cell nearest to a clicked point.

    The click is validated against the Algeria polygon *and* the existing model
    bounding box.  The cell's existing panel features are passed to the same
    ``app_core`` prediction function used elsewhere in the app.
    """
    point = resolve_point(lat, lon, ring=ring)
    if not point.in_area:
        raise OutsideStudyAreaError(point.reason or "Outside the study area.")

    if bundle.status != "REAL" or bundle.model is None:
        raise app_core.ModelNotReadyError(
            bundle.error or "The trained model is not available."
        )

    if feature_names is None:
        if bundle.card is None or "feature_names" not in bundle.card:
            raise app_core.FeatureContractError(
                "The model card does not declare feature names."
            )
        names = [str(name) for name in bundle.card["feature_names"]]
    else:
        names = [str(name) for name in feature_names]

    availability = feature_row_for(
        panel, date, point.lat_cell, point.lon_cell, names
    )
    if not availability.ok or availability.row is None:
        raise FeatureDataUnavailable(
            availability.reason or "Feature data is unavailable for this cell."
        )

    frame = availability.row[names].to_frame().T.astype(float)
    probability = app_core.predict_probability(bundle, frame)
    risk = app_core.risk_level(probability, bundle.card)
    horizon = int(bundle.card.get("horizon_days", 1)) if bundle.card else 1

    return PointPrediction(
        probability=float(probability),
        risk=str(risk).upper(),
        lat=point.lat,
        lon=point.lon,
        lat_cell=point.lat_cell,
        lon_cell=point.lon_cell,
        distance_km=point.distance_km,
        date=pd.Timestamp(date).normalize(),
        horizon_days=horizon,
    )
