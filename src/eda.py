"""Pure data transformations and chart inputs for the interactive EDA section.

The Streamlit layer (``dashboard/eda_view.py``) owns all widgets and Plotly
figures. This module deliberately depends only on pandas/numpy so every data
transformation and every chart's input table can be unit-tested without
running Streamlit.

Data-provenance rules enforced here and in the view:

* ``fires_daily`` / ``firms_raw`` are **historical NASA FIRMS observations**;
* ``fire_panel`` weather columns are NASA POWER reanalysis and ``burned`` is a
  *cell-day with at least one FIRMS detection* (a proxy, not a confirmed
  burned area and not damage);
* no predicted risk is produced, stored or displayed anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

# ---------------------------------------------------------------------------
# Shared vocabulary and chart registry
# ---------------------------------------------------------------------------

STUDY_AREA: dict[str, tuple[float, float]] = {
    "latitude": (LAT_MIN, LAT_MAX),
    "longitude": (LON_MIN, LON_MAX),
}

WEATHER_VARIABLES: dict[str, dict[str, str]] = {
    "T2M": {"label": "Température moyenne à 2 m", "unit": "°C"},
    "T2M_MAX": {"label": "Température maximale à 2 m", "unit": "°C"},
    "RH2M": {"label": "Humidité relative à 2 m", "unit": "%"},
    "WS2M": {"label": "Vent à 2 m", "unit": "m/s"},
    "PRECTOTCORR": {"label": "Précipitations corrigées", "unit": "mm/j"},
    "GWETROOT": {"label": "Humidité racinaire du sol", "unit": "0–1"},
    "rain_30d": {"label": "Cumul de pluie sur 30 jours", "unit": "mm"},
    "dry_days": {"label": "Jours consécutifs sans pluie", "unit": "j"},
}

CONFIDENCE_LABELS = {"n": "Nominale (n)", "l": "Faible (l)", "h": "Élevée (h)"}

SEASON_ORDER = ("DJF (hiver)", "MAM (printemps)", "JJA (été)", "SON (automne)")
_MONTH_TO_SEASON = {
    12: "DJF (hiver)", 1: "DJF (hiver)", 2: "DJF (hiver)",
    3: "MAM (printemps)", 4: "MAM (printemps)", 5: "MAM (printemps)",
    6: "JJA (été)", 7: "JJA (été)", 8: "JJA (été)",
    9: "SON (automne)", 10: "SON (automne)", 11: "SON (automne)",
}
# Public alias reused by the statistical-validation module so season labels
# stay identical between the EDA charts and the hypothesis tests.
MONTH_TO_SEASON = dict(_MONTH_TO_SEASON)

LATITUDE_BANDS = (
    ("Nord côtier (< 34°N)", -np.inf, 34.0),
    ("Tell central (34–35.75°N)", 34.0, 35.75),
    ("Hauts plateaux (> 35.75°N)", 35.75, np.inf),
)

FRP_BINS_EDGES = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, np.inf)
FRP_BIN_LABELS = (
    "0–1", "1–2", "2–5", "5–10", "10–20", "20–50", "50–100", "100–200", "200–500", "≥ 500"
)


@dataclass(frozen=True)
class EdaVisualization:
    """Metadata for one chart rendered by the EDA view."""

    id: str
    title: str
    source: str
    description: str


VISUALIZATIONS: tuple[EdaVisualization, ...] = (
    EdaVisualization("monthly_detections", "Détections FIRMS par mois", "fires",
                     "Volume mensuel de détections et nombre de cellules actives."),
    EdaVisualization("seasonal_detections", "Distribution saisonnière", "fires",
                     "Détections regroupées par saison météorologique."),
    EdaVisualization("top_cells", "Cellules les plus détectées", "fires",
                     "Top 15 des cellules NASA POWER par nombre de détections."),
    EdaVisualization("geographic_detections", "Répartition géographique", "fires",
                     "Carte de dispersion des centres de cellules avec détections."),
    EdaVisualization("latitude_bands", "Observations par bande latitudinale", "fires",
                     "Détections regroupées par larges bandes de latitude (non administratives)."),
    EdaVisualization("confidence_distribution", "Confiance des détections FIRMS", "firms",
                     "Répartition des niveaux de confiance des capteurs MODIS/VIIRS."),
    EdaVisualization("frp_distribution", "Puissance radiative du feu (FRP)", "firms",
                     "Distribution de la puissance radiative en MW, échelle logarithmique visuelle."),
    EdaVisualization("temperature_distribution", "Distribution des températures", "panel",
                     "Température maximale quotidienne selon la présence d'une détection."),
    EdaVisualization("humidity_distribution", "Distribution de l'humidité relative", "panel",
                     "Humidité relative quotidienne selon la présence d'une détection."),
    EdaVisualization("wind_distribution", "Distribution du vent", "panel",
                     "Vitesse du vent à 2 m selon la présence d'une détection."),
    EdaVisualization("precipitation_distribution", "Distribution des précipitations", "panel",
                     "Précipitations quotidiennes selon la présence d'une détection."),
    EdaVisualization("weather_comparison", "Conditions météo et détections", "panel",
                     "Taux de cellules-jours détectées par classe de la variable choisie."),
    EdaVisualization("correlation_matrix", "Matrice de corrélation", "panel",
                     "Corrélations de Pearson entre variables numériques et détection."),
    EdaVisualization("monthly_weather_trend", "Météo et détections au fil des mois", "panel",
                     "Moyenne mensuelle d'une variable et taux de cellules-jours détectées."),
    EdaVisualization("selected_variable", "Distribution d'une variable au choix", "panel",
                     "Histogramme de la variable sélectionnée dans le panneau de filtres."),
)
VISUALIZATION_IDS: tuple[str, ...] = tuple(v.id for v in VISUALIZATIONS)
VISUALIZATION_TITLES: dict[str, str] = {v.id: v.title for v in VISUALIZATIONS}


# ---------------------------------------------------------------------------
# Filters and coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdaFilters:
    """Validated, user-facing filter values."""

    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    lat_range: tuple[float, float] = STUDY_AREA["latitude"]
    lon_range: tuple[float, float] = STUDY_AREA["longitude"]
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FilteredEdaData:
    """Filtered copies of the three EDA frames plus the resolved filters."""

    panel: pd.DataFrame
    fires: pd.DataFrame
    firms: pd.DataFrame
    filters: EdaFilters


def normalize_date_range(
    start: Any,
    end: Any,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str | None]:
    """Return usable inclusive date bounds and an optional explanation.

    An incomplete or inverted range is *ignored* rather than silently swapped,
    so a chart can never show a period the user did not ask for.
    """
    start_ts = pd.to_datetime(start, errors="coerce") if start is not None else None
    end_ts = pd.to_datetime(end, errors="coerce") if end is not None else None
    if isinstance(start_ts, pd.Series):
        start_ts = None
    if isinstance(end_ts, pd.Series):
        end_ts = None
    start_ts = None if pd.isna(start_ts) else pd.Timestamp(start_ts).normalize()
    end_ts = None if pd.isna(end_ts) else pd.Timestamp(end_ts).normalize()

    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        return None, None, "Période invalide (début > fin) : le filtre de dates est ignoré."
    if (start is not None and start_ts is None) or (end is not None and end_ts is None):
        return start_ts, end_ts, "Date de début ou de fin illisible : filtre partiel appliqué."
    return start_ts, end_ts, None


def normalize_geo_range(
    bounds: Sequence[Any] | None,
    axis: str,
) -> tuple[tuple[float, float], str | None]:
    """Clamp a geographic range to the study area, or ignore invalid bounds."""
    limits = STUDY_AREA[axis]
    if bounds is None:
        return limits, None
    try:
        low, high = (float(bounds[0]), float(bounds[1]))
    except (TypeError, ValueError, IndexError):
        return limits, f"Borne {axis} illisible : filtre géographique ignoré."
    if not np.isfinite(low) or not np.isfinite(high):
        return limits, f"Borne {axis} non finie : filtre géographique ignoré."
    if low > high:
        return limits, f"Borne {axis} inversée : filtre géographique ignoré."

    clamped = (max(low, limits[0]), min(high, limits[1]))
    if clamped[0] > clamped[1]:
        return limits, f"Borne {axis} hors zone d'étude : filtre géographique ignoré."
    note = None
    if clamped != (low, high):
        note = f"Borne {axis} limitée à la zone d'étude ({limits[0]} à {limits[1]})."
    return clamped, note


def _valid_dates(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return pd.to_datetime(df["date"], errors="coerce")


def _series_or_default(df: pd.DataFrame, column: str, default: Any) -> pd.Series:
    """Return a numeric/string Series even when an input column is absent."""
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def apply_bounds(
    df: pd.DataFrame,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    """Return a filtered copy; missing/non-numeric values never match a bound."""
    if df.empty:
        return df.copy()
    mask = pd.Series(True, index=df.index)
    dates = _valid_dates(df)
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    for column, bounds in ((lat_col, lat_range), (lon_col, lon_range)):
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        mask &= values.between(bounds[0], bounds[1], inclusive="both").fillna(False)
    return df.loc[mask].copy()


def apply_filters(
    panel: pd.DataFrame,
    fires: pd.DataFrame,
    firms: pd.DataFrame,
    filters: EdaFilters,
) -> FilteredEdaData:
    """Apply validated filters consistently to all three EDA datasets."""
    bounds: dict[str, tuple[float, float]] = {
        "latitude": filters.lat_range,
        "longitude": filters.lon_range,
    }
    return FilteredEdaData(
        panel=apply_bounds(panel, start=filters.start, end=filters.end,
                           lat_range=bounds["latitude"], lon_range=bounds["longitude"],
                           lat_col="lat_cell", lon_col="lon_cell"),
        fires=apply_bounds(fires, start=filters.start, end=filters.end,
                           lat_range=bounds["latitude"], lon_range=bounds["longitude"],
                           lat_col="lat_cell", lon_col="lon_cell"),
        firms=apply_bounds(firms, start=filters.start, end=filters.end,
                           lat_range=bounds["latitude"], lon_range=bounds["longitude"],
                           lat_col="latitude", lon_col="longitude"),
        filters=filters,
    )


def date_coverage(df: pd.DataFrame, column: str = "date") -> dict[str, Any]:
    """Summarise the temporal extent of a dataset without modifying it."""
    empty = {"start": None, "end": None, "days": 0, "years": 0, "invalid_dates": 0}
    if df.empty or column not in df.columns:
        return empty
    dates = pd.to_datetime(df[column], errors="coerce").dropna()
    if dates.empty:
        empty["invalid_dates"] = len(df)
        return empty
    return {
        "start": dates.min(),
        "end": dates.max(),
        "days": int(dates.dt.normalize().nunique()),
        "years": int(dates.dt.year.nunique()),
        "invalid_dates": int(len(df) - len(dates)),
    }


def geographic_coverage(
    df: pd.DataFrame,
    lat_col: str = "lat_cell",
    lon_col: str = "lon_cell",
) -> dict[str, Any]:
    """Summarise spatial extent and flag rows outside the model domain."""
    empty = {
        "min_lat": None, "max_lat": None, "min_lon": None, "max_lon": None,
        "unique_cells": 0, "valid_rows": 0, "invalid_rows": 0, "outside_study_area": 0,
    }
    if df.empty or lat_col not in df.columns or lon_col not in df.columns:
        return empty
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    valid = lat.notna() & lon.notna()
    outside = (
        valid
        & (
            ~lat.between(LAT_MIN, LAT_MAX, inclusive="both")
            | ~lon.between(LON_MIN, LON_MAX, inclusive="both")
        )
    )
    if not valid.any():
        empty["invalid_rows"] = int(len(df))
        return empty
    return {
        "min_lat": float(lat[valid].min()),
        "max_lat": float(lat[valid].max()),
        "min_lon": float(lon[valid].min()),
        "max_lon": float(lon[valid].max()),
        "unique_cells": int(pd.DataFrame({"lat": lat[valid], "lon": lon[valid]}).drop_duplicates().shape[0]),
        "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "outside_study_area": int(outside.sum()),
    }


# ---------------------------------------------------------------------------
# Raw FIRMS preparation
# ---------------------------------------------------------------------------

_NUMERIC_FIRMS_COLUMNS = (
    "latitude", "longitude", "brightness", "bright_t31", "frp", "scan", "track",
)


def prepare_firms(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Coerce FIRMS types and remove structurally empty/invalid rows.

    Blank tail lines in the shipped CSV have no date or coordinates; they are
    counted in the returned diagnostics instead of being plotted. Rows with
    valid coordinates and date but a missing FRP/brightness are kept so the
    missing-value report remains honest.
    """
    work = raw.copy()
    for column in _NUMERIC_FIRMS_COLUMNS:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    if "date" not in work.columns and "acq_date" in work.columns:
        work["date"] = pd.to_datetime(work["acq_date"], errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        work["date"] = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")

    required = [c for c in ("latitude", "longitude", "date") if c in work.columns]
    usable = work[required].notna().all(axis=1) if required else pd.Series(False, index=work.index)
    clean = work.loc[usable].copy()
    clean["confidence_label"] = (
        clean["confidence"].map(CONFIDENCE_LABELS).fillna("Non renseignée")
        if "confidence" in clean.columns else "Non renseignée"
    )
    clean["is_high_confidence"] = (
        clean["confidence"].eq("h") if "confidence" in clean.columns else False
    )

    stats = {
        "input_rows": int(len(work)),
        "valid_rows": int(len(clean)),
        "dropped_rows": int(len(work) - len(clean)),
        "missing_by_column": {
            str(column): int(count) for column, count in work.isna().sum().items() if count
        },
    }
    return clean, stats


# ---------------------------------------------------------------------------
# Chart inputs: all return tidy DataFrames with stable column names
# ---------------------------------------------------------------------------


def monthly_detections(fires: pd.DataFrame) -> pd.DataFrame:
    """Monthly detection volume and active-cell count from FIRMS daily data."""
    columns = ["month", "detections", "active_cells", "active_days"]
    if fires.empty or "date" not in fires.columns:
        return pd.DataFrame(columns=columns)
    frame = fires.copy()
    dates = _valid_dates(frame)
    month = dates.dt.to_period("M").astype(str)
    work = pd.DataFrame({
        "month": month,
        "n_detections": pd.to_numeric(_series_or_default(frame, "n_detections", 0), errors="coerce").fillna(0),
        "cell": (_series_or_default(frame, "lat_cell", 0).astype(str)
                 + "_" + _series_or_default(frame, "lon_cell", 0).astype(str)),
        "date": dates,
    })
    work = work[work["month"] != "NaT"]
    grouped = work.groupby("month").agg(
        detections=("n_detections", "sum"),
        active_cells=("cell", "nunique"),
        active_days=("date", "nunique"),
    ).reset_index().sort_values("month")
    for column in ("detections", "active_cells", "active_days"):
        grouped[column] = grouped[column].astype(int)
    return grouped[columns]


def seasonal_detections(fires: pd.DataFrame) -> pd.DataFrame:
    """Detection totals by meteorological season (ordered DJF → SON)."""
    columns = ["season", "detections"]
    if fires.empty or "date" not in fires.columns:
        return pd.DataFrame(columns=columns)
    dates = _valid_dates(fires)
    seasons = dates.dt.month.map(_MONTH_TO_SEASON)
    work = pd.DataFrame({
        "season": seasons,
        "n_detections": pd.to_numeric(_series_or_default(fires, "n_detections", 0), errors="coerce").fillna(0),
    })
    grouped = work.dropna(subset=["season"]).groupby("season")["n_detections"].sum()
    return pd.DataFrame({
        "season": list(SEASON_ORDER),
        "detections": [int(grouped.get(season, 0)) for season in SEASON_ORDER],
    })


def top_detection_cells(fires: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Most frequently detected grid cells with their centre coordinates."""
    columns = ["cell", "latitude", "longitude", "detections", "active_days"]
    if fires.empty:
        return pd.DataFrame(columns=columns)
    work = pd.DataFrame({
        "cell": (_series_or_default(fires, "lat_cell", 0).astype(str)
                 + "_" + _series_or_default(fires, "lon_cell", 0).astype(str)),
        "latitude": pd.to_numeric(_series_or_default(fires, "lat_cell", np.nan), errors="coerce"),
        "longitude": pd.to_numeric(_series_or_default(fires, "lon_cell", np.nan), errors="coerce"),
        "detections": pd.to_numeric(_series_or_default(fires, "n_detections", 0), errors="coerce").fillna(0),
        "date": _valid_dates(fires),
    })
    grouped = work.groupby("cell", dropna=False).agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        detections=("detections", "sum"),
        active_days=("date", "nunique"),
    ).reset_index().sort_values("detections", ascending=False).head(top_n)
    return grouped[columns].reset_index(drop=True)


def detection_scatter(fires: pd.DataFrame) -> pd.DataFrame:
    """Coordinates and detection counts for the geographic scatter chart."""
    columns = ["latitude", "longitude", "detections"]
    if fires.empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame({
        "latitude": pd.to_numeric(_series_or_default(fires, "lat_cell", np.nan), errors="coerce"),
        "longitude": pd.to_numeric(_series_or_default(fires, "lon_cell", np.nan), errors="coerce"),
        "detections": pd.to_numeric(_series_or_default(fires, "n_detections", 0), errors="coerce").fillna(0),
    })[columns].dropna(subset=["latitude", "longitude"])


def latitude_band_detections(fires: pd.DataFrame) -> pd.DataFrame:
    """Detection volume over broad, explicitly non-administrative latitude bands."""
    columns = ["band", "detections", "active_cells"]
    if fires.empty:
        return pd.DataFrame({columns[0]: [b[0] for b in LATITUDE_BANDS],
                             columns[1]: [0, 0, 0], columns[2]: [0, 0, 0]})
    lat = pd.to_numeric(_series_or_default(fires, "lat_cell", np.nan), errors="coerce")
    detections = pd.to_numeric(_series_or_default(fires, "n_detections", 0), errors="coerce").fillna(0)
    rows = []
    for label, low, high in LATITUDE_BANDS:
        mask = lat.between(low, high, inclusive="neither" if np.isinf(low) or np.isinf(high) else "both")
        cells = pd.DataFrame({
            "cell": fires.loc[mask, "lat_cell"].astype(str) + "_" + fires.loc[mask, "lon_cell"].astype(str)
        }) if {"lat_cell", "lon_cell"}.issubset(fires.columns) else pd.DataFrame({"cell": []})
        rows.append({
            "band": label,
            "detections": int(detections[mask].sum()),
            "active_cells": int(cells["cell"].nunique()),
        })
    return pd.DataFrame(rows, columns=columns)


def confidence_distribution(firms: pd.DataFrame) -> pd.DataFrame:
    """Counts and percentages per FIRMS confidence level."""
    columns = ["confidence", "detections", "percent"]
    if firms.empty or "confidence_label" not in firms.columns:
        return pd.DataFrame(columns=columns)
    counts = firms["confidence_label"].value_counts()
    total = int(counts.sum())
    order = [label for label in CONFIDENCE_LABELS.values() if label in counts.index]
    order += [label for label in counts.index if label not in order]
    return pd.DataFrame({
        "confidence": order,
        "detections": [int(counts[label]) for label in order],
        "percent": [round(100.0 * int(counts[label]) / total, 2) if total else 0.0 for label in order],
    })[columns]


def frp_distribution(firms: pd.DataFrame) -> pd.DataFrame:
    """Binned fire radiative power counts (MW) using fixed, readable classes."""
    columns = ["frp_bin", "detections"]
    if firms.empty or "frp" not in firms.columns:
        return pd.DataFrame({"frp_bin": list(FRP_BIN_LABELS), "detections": [0] * len(FRP_BIN_LABELS)})
    values = pd.to_numeric(firms["frp"], errors="coerce").dropna()
    counts, _ = np.histogram(values.to_numpy(), bins=np.asarray(FRP_BINS_EDGES, dtype=float))
    return pd.DataFrame({"frp_bin": list(FRP_BIN_LABELS), "detections": counts.astype(int)})


def weather_histogram(panel: pd.DataFrame, variable: str, bins: int = 24) -> pd.DataFrame:
    """Histogram counts of a weather variable, split by FIRMS detection proxy."""
    columns = ["bin_center", "bin_label", "without_detection", "with_detection"]
    if panel.empty or variable not in panel.columns:
        return pd.DataFrame(columns=columns)
    values = pd.to_numeric(panel[variable], errors="coerce")
    burned = pd.to_numeric(_series_or_default(panel, "burned", 0), errors="coerce").fillna(0).gt(0)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.DataFrame(columns=columns)
    counts, edges = np.histogram(finite, bins=bins)
    no_fire, _ = np.histogram(values[~burned].dropna(), bins=edges)
    fire, _ = np.histogram(values[burned].dropna(), bins=edges)
    return pd.DataFrame({
        "bin_center": (edges[:-1] + edges[1:]) / 2.0,
        "bin_label": [f"{a:.1f}–{b:.1f}" for a, b in zip(edges[:-1], edges[1:])],
        "without_detection": no_fire.astype(int),
        "with_detection": fire.astype(int),
    })[columns]


def weather_summary(
    panel: pd.DataFrame,
    variables: Sequence[str] = ("T2M_MAX", "RH2M", "WS2M", "PRECTOTCORR", "GWETROOT"),
) -> pd.DataFrame:
    """Mean values of selected variables with and without a detection proxy."""
    columns = ["variable", "label", "without_detection", "with_detection", "difference"]
    burned = (
        pd.to_numeric(_series_or_default(panel, "burned", 0), errors="coerce").fillna(0).gt(0)
        if not panel.empty else pd.Series(dtype=bool)
    )
    rows = []
    for variable in variables:
        if variable not in panel.columns or panel.empty:
            values = pd.Series(dtype=float)
            no_fire, fire = float("nan"), float("nan")
        else:
            values = pd.to_numeric(panel[variable], errors="coerce")
            no_fire = values[~burned].mean()
            fire = values[burned].mean()
        difference = fire - no_fire if pd.notna(fire) and pd.notna(no_fire) else float("nan")
        rows.append({
            "variable": variable,
            "label": WEATHER_VARIABLES.get(variable, {}).get("label", variable),
            "without_detection": no_fire,
            "with_detection": fire,
            "difference": difference,
        })
    return pd.DataFrame(rows, columns=columns)


def detection_rate_by_weather_class(
    panel: pd.DataFrame,
    variable: str,
    bins: int = 8,
) -> pd.DataFrame:
    """Detection-proxy rate by quantile class of one weather variable.

    Classes are built from the filtered data itself, so the x-axis always
    reflects the period and region currently selected by the user.
    """
    columns = ["weather_class", "fire_rate_pct", "cell_days"]
    if panel.empty or variable not in panel.columns:
        return pd.DataFrame(columns=columns)
    values = pd.to_numeric(panel[variable], errors="coerce")
    burned = pd.to_numeric(_series_or_default(panel, "burned", 0), errors="coerce").fillna(0).gt(0)
    valid = values.notna()
    if not valid.any():
        return pd.DataFrame(columns=columns)
    try:
        classes, edges = pd.qcut(values[valid], q=bins, duplicates="drop", retbins=True)
        labels = [f"{low:.1f}–{high:.1f}" for low, high in zip(edges[:-1], edges[1:])]
        codes = pd.Series(classes.cat.codes.to_numpy(), index=values[valid].index)
        work = pd.DataFrame({
            "code": codes,
            "weather_class": codes.map(lambda code: labels[code] if 0 <= code < len(labels) else "n/a"),
            "burned": burned[valid],
        })
    except ValueError:  # constant series cannot be split into quantiles
        work = pd.DataFrame({"weather_class": ["valeur unique"], "burned": burned[valid]})
    grouped = work.groupby(["weather_class"], sort=False, observed=True).agg(
        fire_rate_pct=("burned", "mean"), cell_days=("burned", "size")
    ).reset_index()
    grouped["fire_rate_pct"] *= 100.0
    return grouped[columns]


def correlation_matrix(
    panel: pd.DataFrame,
    variables: Sequence[str] = ("T2M_MAX", "RH2M", "WS2M", "PRECTOTCORR", "GWETROOT", "burned"),
) -> pd.DataFrame:
    """Pearson correlation matrix on selected numeric columns."""
    columns = [var for var in variables if var in panel.columns]
    if not columns or panel.empty:
        return pd.DataFrame()
    return panel[columns].apply(pd.to_numeric, errors="coerce").corr()


def monthly_weather_trend(panel: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Monthly mean weather and share of cell-days with a FIRMS detection."""
    columns = ["month", "mean_weather", "fire_rate_pct"]
    if panel.empty or "date" not in panel.columns or variable not in panel.columns:
        return pd.DataFrame(columns=columns)
    dates = _valid_dates(panel)
    work = pd.DataFrame({
        "month": dates.dt.to_period("M").astype(str),
        "weather": pd.to_numeric(panel[variable], errors="coerce"),
        "burned": pd.to_numeric(_series_or_default(panel, "burned", 0), errors="coerce").fillna(0),
    })
    work = work[work["month"] != "NaT"]
    grouped = work.groupby("month").agg(
        mean_weather=("weather", "mean"), fire_rate_pct=("burned", "mean")
    ).reset_index().sort_values("month")
    grouped["fire_rate_pct"] *= 100.0
    return grouped[columns]


def deterministic_sample(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """Return at most ``max_rows`` rows, reproducibly, without mutating input."""
    if max_rows <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=seed).sort_index().copy()


def coverage_frame(
    panel: pd.DataFrame,
    fires: pd.DataFrame,
    firms: pd.DataFrame,
) -> pd.DataFrame:
    """One-row-per-dataset table of temporal and geographic coverage."""
    rows = []
    for name, frame, lat_col, lon_col in (
        ("fire_panel", panel, "lat_cell", "lon_cell"),
        ("fires_daily", fires, "lat_cell", "lon_cell"),
        ("firms_raw (valides)", firms, "latitude", "longitude"),
    ):
        dates = date_coverage(frame)
        geo = geographic_coverage(frame, lat_col, lon_col)
        rows.append({
            "dataset": name,
            "rows": len(frame),
            "start": dates["start"],
            "end": dates["end"],
            "days": dates["days"],
            "years": dates["years"],
            "unique_cells": geo["unique_cells"],
            "min_lat": geo["min_lat"],
            "max_lat": geo["max_lat"],
            "min_lon": geo["min_lon"],
            "max_lon": geo["max_lon"],
            "invalid_rows": geo["invalid_rows"] + dates["invalid_dates"],
            "outside_study_area": geo["outside_study_area"],
        })
    return pd.DataFrame(rows)
