"""Core logic for point inspection and predictive context (NO Streamlit dependency).

Provides pure, testable functions for:
  - Grid cell snapping and boundary verification across the 209 NASA POWER cells.
  - Regional Wilaya resolution and topography estimation for Northern Algeria.
  - Historical cell fire frequency and climatological medians (pre-test 2018–2022).
  - Feature row construction via app_core.build_manual_features (Simulation mode)
    and app_core.build_live_features (NASA POWER observed mode).
  - Model scoring and risk categorization via app_core.score_features.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
import sys
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_core

# Lat/lon resolution constants
LAT_STEP = app_core.MODEL_LAT_STEP    # 0.5 deg
LON_STEP = app_core.MODEL_LON_STEP    # 0.625 deg
HALF_LAT = LAT_STEP / 2.0             # 0.25 deg
HALF_LON = LON_STEP / 2.0             # 0.3125 deg

# Cache of 209 active cells
_ACTIVE_209_CELLS: set[tuple[float, float]] | None = None

# Reference coordinates for Northern Algeria Wilayas and Forest Massifs
ALGERIA_WILAYAS = [
    ("Alger", 36.75, 3.05, 35),
    ("Blida / Atlas de Chréa", 36.47, 2.83, 750),
    ("Tipaza", 36.59, 2.44, 120),
    ("Boumerdès", 36.76, 3.47, 60),
    ("Tizi Ouzou / Massif du Djurdjura", 36.72, 4.05, 820),
    ("Bouira", 36.38, 3.90, 520),
    ("Béjaïa / Massif des Babors", 36.75, 5.08, 480),
    ("Jijel / Forêt de Guerrouche", 36.80, 5.76, 450),
    ("Skikda / Massif de Collo", 36.88, 6.90, 280),
    ("Annaba / Monts de l'Edough", 36.90, 7.76, 310),
    ("El Tarf / Parc National d'El Kala", 36.77, 8.31, 140),
    ("Guelma", 36.46, 7.43, 290),
    ("Souk Ahras", 36.28, 7.95, 650),
    ("Constantine", 36.36, 6.61, 620),
    ("Mila", 36.45, 6.26, 460),
    ("Sétif / Hauts Plateaux", 36.19, 5.41, 1080),
    ("Bordj Bou Arréridj", 36.07, 4.76, 920),
    ("Médéa", 36.26, 2.75, 880),
    ("Aïn Defla", 36.26, 1.97, 280),
    ("Chlef / Dahra", 36.16, 1.33, 140),
    ("Mascara / Monts des Beni Chougrane", 35.40, 0.14, 580),
    ("Relizane", 35.74, 0.55, 120),
    ("Mostaganem", 35.93, 0.09, 80),
    ("Oran / Monts de Tessala", 35.70, -0.63, 110),
    ("Aïn Témouchent", 35.30, -1.14, 150),
    ("Tlemcen / Monts de Tlemcen", 34.88, -1.31, 800),
    ("Sidi Bel Abbès", 35.19, -0.64, 470),
    ("Batna / Massif de l'Aurès", 35.55, 6.17, 1040),
    ("Khenchela / Aurès oriental", 35.43, 7.14, 1120),
    ("Tébessa", 35.40, 8.12, 850),
    ("Tiaret / Sersou", 35.37, 1.32, 980),
    ("Djelfa / Atlas Saharien", 34.67, 3.25, 1150),
    ("M'Sila / Hodna", 35.70, 4.54, 480),
    ("Biskra / Ziban", 34.85, 5.73, 120),
    ("Laghouat", 33.80, 2.87, 750),
]


def get_active_209_cells(panel_path: Path | str | None = None) -> set[tuple[float, float]]:
    """Return the exact set of 209 active NASA POWER grid cells from fire_panel.parquet."""
    global _ACTIVE_209_CELLS
    if _ACTIVE_209_CELLS is not None:
        return _ACTIVE_209_CELLS

    path = Path(panel_path) if panel_path is not None else app_core.FIRE_PANEL_PATH
    if path.exists():
        df = pd.read_parquet(path, columns=["lat_cell", "lon_cell"])
        pairs = set(zip(df["lat_cell"].round(4), df["lon_cell"].round(4)))
        if len(pairs) == 209:
            _ACTIVE_209_CELLS = pairs
            return _ACTIVE_209_CELLS

    # Fallback to coordinate grid bounded by model training box (excluding the 7 missing cells)
    excluded_cells = {
        (33.0, 3.125), (33.5, 8.75), (36.0, 0.0), (36.5, 3.125),
        (36.5, 6.25), (37.0, 6.875), (37.0, 7.5),
    }
    lat_vals = np.arange(app_core.MODEL_LAT_MIN, app_core.MODEL_LAT_MAX + 1e-5, LAT_STEP)
    lon_vals = np.arange(app_core.MODEL_LON_MIN, app_core.MODEL_LON_MAX + 1e-5, LON_STEP)
    generated = set()
    for la in lat_vals:
        for lo in lon_vals:
            cell_pair = (round(float(la), 4), round(float(lo), 4))
            if cell_pair in excluded_cells:
                continue
            generated.add(cell_pair)

    _ACTIVE_209_CELLS = generated
    return _ACTIVE_209_CELLS


def snap_to_grid(lat: float, lon: float) -> tuple[float, float]:
    """Snap geographic coordinates to NASA POWER grid resolution."""
    return app_core.snap_to_grid(lat, lon)


def get_cell_bounding_box(lat_cell: float, lon_cell: float) -> list[list[float]]:
    """Return the [[south, west], [north, east]] bounds for a NASA POWER cell polygon."""
    south = round(lat_cell - HALF_LAT, 4)
    north = round(lat_cell + HALF_LAT, 4)
    west = round(lon_cell - HALF_LON, 4)
    east = round(lon_cell + HALF_LON, 4)
    return [[south, west], [north, east]]


def is_point_in_active_cells(
    lat: float,
    lon: float,
    active_cells: set[tuple[float, float]] | None = None,
) -> tuple[bool, float, float]:
    """Verify if a point falls strictly inside one of the 209 active NASA POWER cells.

    Returns:
        (is_covered, lat_cell, lon_cell)
    """
    lat_cell, lon_cell = snap_to_grid(lat, lon)
    cells = active_cells if active_cells is not None else get_active_209_cells()

    if (lat_cell, lon_cell) not in cells:
        return False, lat_cell, lon_cell

    # Verify that the coordinate is within the half-step bounding box
    tol = 1e-4
    in_lat = (lat_cell - HALF_LAT - tol) <= lat <= (lat_cell + HALF_LAT + tol)
    in_lon = (lon_cell - HALF_LON - tol) <= lon <= (lon_cell + HALF_LON + tol)
    is_covered = bool(in_lat and in_lon)
    return is_covered, lat_cell, lon_cell


def get_wilaya_and_region(lat: float, lon: float) -> str:
    """Resolve coordinates to the nearest Northern Algeria Wilaya or major geographical massif."""
    best_name = "Nord algérien"
    best_dist = float("inf")
    for name, w_lat, w_lon, _ in ALGERIA_WILAYAS:
        d = math.hypot(lat - w_lat, lon - w_lon)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def get_elevation_estimate(lat: float, lon: float) -> int:
    """Estimate ground elevation (meters above sea level) from topography reference."""
    best_alt = 350
    best_dist = float("inf")
    for _, w_lat, w_lon, alt in ALGERIA_WILAYAS:
        d = math.hypot(lat - w_lat, lon - w_lon)
        if d < best_dist:
            best_dist = d
            best_alt = alt
    # Small interpolation variation
    return int(best_alt)


def get_cell_statistics(
    lat_cell: float,
    lon_cell: float,
    panel_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute pre-test historical fire stats and weather medians for a specific cell.

    Strictly uses data before 2023-01-01 (pre-test 2018–2022) to avoid test set contamination.
    """
    default_stats: dict[str, Any] = {
        "lat_cell": lat_cell,
        "lon_cell": lon_cell,
        "is_active_cell": True,
        "fire_days": 0,
        "total_days": 0,
        "fire_rate": 0.0,
        "t2m_median": 25.0,
        "t2m_max_median": 32.0,
        "rh2m_median": 45.0,
        "ws2m_median": 4.5,
        "precip_median": 0.0,
        "gwetroot_median": 0.25,
        "dry_days_median": 10.0,
    }

    if panel_df is None or panel_df.empty:
        return default_stats

    # Filter pre-test dates strictly
    mask = (
        (panel_df["lat_cell"].round(4) == round(lat_cell, 4))
        & (panel_df["lon_cell"].round(4) == round(lon_cell, 4))
        & (panel_df["date"] < "2023-01-01")
    )
    cell_data = panel_df[mask]

    if cell_data.empty:
        return default_stats

    fire_days = int(cell_data["burned"].sum()) if "burned" in cell_data else 0
    total_days = int(len(cell_data))
    fire_rate = float(fire_days / total_days) if total_days > 0 else 0.0

    return {
        "lat_cell": lat_cell,
        "lon_cell": lon_cell,
        "is_active_cell": True,
        "fire_days": fire_days,
        "total_days": total_days,
        "fire_rate": fire_rate,
        "t2m_median": float(cell_data["T2M"].median()) if "T2M" in cell_data else 25.0,
        "t2m_max_median": float(cell_data["T2M_MAX"].median()) if "T2M_MAX" in cell_data else 32.0,
        "rh2m_median": float(cell_data["RH2M"].median()) if "RH2M" in cell_data else 45.0,
        "ws2m_median": float(cell_data["WS2M"].median()) if "WS2M" in cell_data else 4.5,
        "precip_median": float(cell_data["PRECTOTCORR"].median()) if "PRECTOTCORR" in cell_data else 0.0,
        "gwetroot_median": float(cell_data["GWETROOT"].median()) if "GWETROOT" in cell_data else 0.25,
        "dry_days_median": float(cell_data["dry_days"].median()) if "dry_days" in cell_data else 10.0,
    }


def evaluate_point_prediction(
    bundle: app_core.ModelBundle,
    lat: float,
    lon: float,
    when: str | datetime.date | datetime.datetime | pd.Timestamp,
    weather_inputs: Mapping[str, float],
) -> dict[str, Any]:
    """Execute prediction pipeline on manual simulation point weather inputs."""
    t2m = float(weather_inputs.get("T2M", 25.0))
    t2m_max = float(weather_inputs.get("T2M_MAX", max(t2m, 30.0)))
    rh2m = float(np.clip(weather_inputs.get("RH2M", 40.0), 1.0, 100.0))
    ws2m = float(max(weather_inputs.get("WS2M", 3.0), 0.0))
    precip = float(max(weather_inputs.get("PRECTOTCORR", 0.0), 0.0))
    gwetroot = float(np.clip(weather_inputs.get("GWETROOT", 0.25), 0.0, 1.0))
    dry_days = float(max(weather_inputs.get("dry_days", 10.0), 0.0))

    clean_weather_inputs = {
        "T2M": t2m,
        "T2M_MAX": t2m_max,
        "RH2M": rh2m,
        "WS2M": ws2m,
        "PRECTOTCORR": precip,
        "GWETROOT": gwetroot,
        "dry_days": dry_days,
    }

    feature_row = app_core.build_manual_features(
        lat=lat,
        lon=lon,
        when=when,
        t2m=t2m,
        t2m_max=t2m_max,
        rh2m=rh2m,
        ws2m=ws2m,
        precip=precip,
        gwetroot=gwetroot,
        dry_days=dry_days,
        card=bundle.card,
    )

    probability, score_source = app_core.score_features(bundle, feature_row)
    level = app_core.risk_level(probability, bundle.card)
    disp = app_core.risk_display(level)
    color = app_core.risk_color(level)
    prob_str = app_core.format_probability(probability)

    lat_cell, lon_cell = snap_to_grid(lat, lon)
    wilaya = get_wilaya_and_region(lat, lon)
    pred_id = uuid.uuid4().hex[:12]
    now_dt = datetime.datetime.now()

    return {
        "prediction_id": pred_id,
        "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "created_at": now_dt.isoformat(),
        "lat": lat,
        "lon": lon,
        "lat_cell": lat_cell,
        "lon_cell": lon_cell,
        "wilaya": wilaya,
        "date": str(pd.Timestamp(when).date()),
        "horizon": "H = 3 jours",
        "weather_inputs": clean_weather_inputs,
        "probability": probability,
        "probability_formatted": prob_str,
        "risk_level": level,
        "risk_display": disp,
        "risk_color": color,
        "score_source": score_source,
        "feature_row": feature_row,
        "mode": "MANUAL_SIMULATION",
        "persistence_note": (
            "Hypothèse de persistance (build_manual_features) : Les conditions météorologiques "
            "saisies pour ce jour sont considérées représentatives des 30 jours antérieurs "
            "(cumuls de pluie 7j/14j/30j et températures maximales)."
        ),
    }


def evaluate_live_point_prediction(
    bundle: app_core.ModelBundle,
    lat: float,
    lon: float,
    when: str | datetime.date | datetime.datetime | pd.Timestamp,
    history_df: pd.DataFrame,
) -> dict[str, Any]:
    """Execute prediction using real fetched NASA POWER history."""
    feature_row = app_core.build_live_features(
        history=history_df,
        lat=lat,
        lon=lon,
        when=when,
        card=bundle.card,
    )
    probability, score_source = app_core.score_features(bundle, feature_row)
    level = app_core.risk_level(probability, bundle.card)
    disp = app_core.risk_display(level)
    color = app_core.risk_color(level)
    prob_str = app_core.format_probability(probability)
    lat_cell, lon_cell = snap_to_grid(lat, lon)
    wilaya = get_wilaya_and_region(lat, lon)
    pred_id = uuid.uuid4().hex[:12]
    now_dt = datetime.datetime.now()

    # Extract observed weather snapshot from history_df if available
    live_weather_inputs: dict[str, float] = {}
    if not history_df.empty:
        last_row = history_df.iloc[-1]
        for col_name in ("T2M", "T2M_MAX", "RH2M", "WS2M", "PRECTOTCORR", "GWETROOT"):
            if col_name in last_row:
                live_weather_inputs[col_name] = float(last_row[col_name])
        if "dry_days" in feature_row.columns:
            live_weather_inputs["dry_days"] = float(feature_row["dry_days"].iloc[0])

    return {
        "prediction_id": pred_id,
        "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "created_at": now_dt.isoformat(),
        "lat": lat,
        "lon": lon,
        "lat_cell": lat_cell,
        "lon_cell": lon_cell,
        "wilaya": wilaya,
        "date": str(pd.Timestamp(when).date()),
        "horizon": "H = 3 jours",
        "weather_inputs": live_weather_inputs,
        "probability": probability,
        "probability_formatted": prob_str,
        "risk_level": level,
        "risk_display": disp,
        "risk_color": color,
        "score_source": score_source,
        "feature_row": feature_row,
        "mode": "AUTOMATIC_NASA_POWER",
        "persistence_note": (
            "Données réelles observées (NASA POWER) : Les caractéristiques de mémoire "
            "hydrique et thermique (7j, 14j, 30j) sont calculées directement à partir "
            "de l'historique satellitaire authentique."
        ),
    }
