"""
FIREWATCH AI — Shared Configuration, Paths, and Professional Design System.

Centralizes data paths, study area boundaries, meteorological constants,
and design system specifications (colors, typography, risk categories).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FIRE_PANEL_PATH = PROCESSED_DIR / "fire_panel.parquet"
FIRES_DAILY_PATH = INTERIM_DIR / "fires_daily.parquet"
FIRMS_RAW_PATH = RAW_DIR / "firms_raw.csv"
MODEL_PATH = MODELS_DIR / "fire_model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

# ---------------------------------------------------------------------------
# Study Area & Spatial Resolution
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 32.0, 37.5
LON_MIN, LON_MAX = -2.0, 9.0
START_YEAR, END_YEAR = 2024, 2026

LAT_STEP = 0.5
LON_STEP = 0.625

MAP_CENTER_LAT = (LAT_MIN + LAT_MAX) / 2
MAP_CENTER_LON = (LON_MIN + LON_MAX) / 2
MAP_ZOOM = 6

# ---------------------------------------------------------------------------
# Meteorological & Feature Definitions
# ---------------------------------------------------------------------------
WEATHER_VARIABLES = {
    "T2M": {
        "name": "Mean Air Temperature",
        "unit": "°C",
        "description": "Daily average temperature at 2 meters above ground",
    },
    "T2M_MAX": {
        "name": "Maximum Air Temperature",
        "unit": "°C",
        "description": "Daily maximum temperature at 2 meters above ground",
    },
    "RH2M": {
        "name": "Relative Humidity",
        "unit": "%",
        "description": "Daily mean relative humidity at 2 meters",
    },
    "WS2M": {
        "name": "Wind Speed",
        "unit": "m/s",
        "description": "Daily average wind speed at 2 meters",
    },
    "PRECTOTCORR": {
        "name": "Precipitation",
        "unit": "mm/day",
        "description": "Corrected daily total precipitation",
    },
    "ALLSKY_SFC_SW_DWN": {
        "name": "Solar Radiation",
        "unit": "MJ/m²/day",
        "description": "Total downward surface solar irradiance under all-sky conditions",
    },
    "GWETROOT": {
        "name": "Root Zone Soil Wetness",
        "unit": "0–1",
        "description": "Normalized soil moisture in root zone layer",
    },
}

ENGINEERED_FEATURES = [
    "rain_7d", "rain_14d", "rain_30d",
    "tmax_7d", "tmax_14d", "tmax_30d",
    "rhmin_7d", "rhmin_14d", "rhmin_30d",
    "dry_days", "doy_sin", "doy_cos",
]

# Exact model-card feature order.  This is the model input contract, not the
# list of weather variables available for display (which still includes solar).
MODEL_FEATURES = [
    "lat_cell", "lon_cell",
    "T2M", "T2M_MAX", "RH2M", "WS2M", "PRECTOTCORR", "GWETROOT",
    "rain_7d", "tmax_7d", "rhmin_7d",
    "rain_14d", "tmax_14d", "rhmin_14d",
    "rain_30d", "tmax_30d", "rhmin_30d",
    "dry_days", "doy_sin", "doy_cos",
]

EXCLUDE_FROM_PREDICTION = {
    "lat_cell", "lon_cell", "date",
    "burned", "n_detections", "total_frp",
}

# ---------------------------------------------------------------------------
# Professional Design System Palette
# ---------------------------------------------------------------------------
# Designed for high-readability environmental intelligence software:
# Dark charcoal sidebar + crisp off-white workspace + deep forest green brand
COLORS = {
    # Brand Identity (Deep Forest Green)
    "brand_primary": "#0F5132",
    "brand_dark": "#0B3D26",
    "brand_accent": "#15803D",
    "brand_subtle": "#E8F5E9",
    "brand_border": "#C8E6C9",

    # Sidebar (Dark Charcoal / Deep Slate)
    "sidebar_bg": "#0F172A",
    "sidebar_surface": "#1E293B",
    "sidebar_border": "#334155",
    "sidebar_text": "#F8FAFC",
    "sidebar_muted": "#94A3B8",
    "sidebar_active": "#10B981",

    # Main Workspace (Off-White & Crisp Surfaces)
    "workspace_bg": "#F8FAFC",
    "surface_white": "#FFFFFF",
    "surface_subtle": "#F1F5F9",
    "surface_hover": "#F8FAFC",

    # Borders & Dividers
    "border_light": "#E2E8F0",
    "border_subtle": "#CBD5E1",
    "border_focus": "#0F5132",

    # Typography
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "text_muted": "#64748B",
    "text_subtle": "#94A3B8",

    # Semantic Risk Levels (Calibrated for universal display)
    "risk_low": "#15803D",
    "risk_low_bg": "#DCFCE7",
    "risk_moderate": "#D97706",
    "risk_moderate_bg": "#FEF3C7",
    "risk_high": "#EA580C",
    "risk_high_bg": "#FFEDD5",
    "risk_very_high": "#DC2626",
    "risk_very_high_bg": "#FEE2E2",
}

RISK_LEVELS = {
    "Low": {
        "color": COLORS["risk_low"],
        "bg": COLORS["risk_low_bg"],
        "label": "Low Risk",
        "description": "Environmental conditions do not favor active fire ignition or spread.",
    },
    "Moderate": {
        "color": COLORS["risk_moderate"],
        "bg": COLORS["risk_moderate_bg"],
        "label": "Moderate Risk",
        "description": "Mild drying and moderate temperatures indicate localized vigilance required.",
    },
    "High": {
        "color": COLORS["risk_high"],
        "bg": COLORS["risk_high_bg"],
        "label": "High Risk",
        "description": "Elevated thermal stress, low humidity, and fuel aridity create high ignition susceptibility.",
    },
    "Very High": {
        "color": COLORS["risk_very_high"],
        "bg": COLORS["risk_very_high_bg"],
        "label": "Very High Risk",
        "description": "Critical fire weather: severe drought, high winds, and high temperatures present acute danger.",
    },
}

# Plotly Theme Settings
PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_BG = "rgba(0,0,0,0)"
PLOTLY_PAPER_BG = "rgba(0,0,0,0)"
PLOTLY_GRID_COLOR = "#F1F5F9"
PLOTLY_FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
