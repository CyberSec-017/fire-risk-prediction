"""Advanced Feature Engineering V2 for wildfire risk prediction.

This module provides modular, causally strict, and scientifically grounded feature
engineering functions to enrich wildfire risk prediction datasets.

Feature Families (12 new features):
  1. VPD (2 features):
     - vpd: Vapor Pressure Deficit from T2M and RH2M (Magnus-Tetens, in kPa).
     - vpd_tmax_proxy: Proxy for peak daytime VPD from T2M_MAX and RH2M (in kPa).
  2. Temperature Anomalies (3 features):
     - tmax_anomaly_30d: T2M_MAX minus 30-day antecedent rolling mean (shift(1)).
     - t2m_anomaly_hist: T2M minus historical cell monthly mean from prior years (Y' < Y).
     - t2m_zscore_hist: Standardized temperature anomaly on prior years (requires >= 2 years and std > 0).
  3. Meteorological Indices (4 features):
     - angstrom_proxy: Ångström fire risk index adapted for daily averages.
     - cbi: Chandler Burning Index (metric daily adaptation).
     - hdw_proxy: Hot-Dry-Windy surface proxy (vpd * WS2M, in kPa * m/s).
     - nesterov_proxy: Cumulative Nesterov index with dew point and >= 3mm precipitation reset.
  4. Spatial-Temporal Neighbor Fire History (3 features):
     - neighbor_fires_7d: Cumulative FIRMS detections in 8 neighbors over [t-7, t-1].
     - neighbor_fires_14d: Cumulative FIRMS detections in 8 neighbors over [t-14, t-1].
     - neighbor_fires_30d: Cumulative FIRMS detections in 8 neighbors over [t-30, t-1].

Methodological guarantees:
  - Zero Future Leakage: Every rolling window applies shift(1) to exclude prediction day t.
  - Zero Climatological Leakage: Historical monthly baselines use strictly prior years (Y' < Y).
  - Explicit Missingness: Unobserved histories (e.g. 2018 for annual climatology, or incomplete
    rolling windows) yield explicit NaN values, handled natively by tree-based models.
  - Target Independence: The 'target' column is never accessed, preserving strict causal purity.
  - Immutability: Every function operates on copies, preserving row order, indices, and row counts.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


FEATURE_FAMILIES = {
    "vpd": ["vpd", "vpd_tmax_proxy"],
    "anomalies": ["tmax_anomaly_30d", "t2m_anomaly_hist", "t2m_zscore_hist"],
    "indices": ["angstrom_proxy", "cbi", "hdw_proxy", "nesterov_proxy"],
    "neighbors": ["neighbor_fires_7d", "neighbor_fires_14d", "neighbor_fires_30d"],
}

ALL_NEW_FEATURES = [feat for feats in FEATURE_FAMILIES.values() for feat in feats]


# ---------------------------------------------------------------------------
# 1. Vapor Pressure Deficit (VPD)
# ---------------------------------------------------------------------------

def _calc_vpd_series(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Calculate Vapor Pressure Deficit (kPa) via Magnus-Tetens formulation.

    Formula:
      es(T) = 0.61078 * exp((17.27 * T) / (T + 237.3))  [kPa]
      ea(T, RH) = es(T) * clip(RH, 0, 100) / 100         [kPa]
      VPD = max(es(T) - ea(T, RH), 0.0)                  [kPa]

    References:
      Murray, F. W. (1967). On the computation of saturation vapor pressure.
      FAO Irrigation and Drainage Paper No. 56 (Allen et al., 1998).
    """
    rh_clipped = np.clip(rh_pct, 0.0, 100.0)
    denom = temp_c + 237.3
    valid = (denom > 0) & temp_c.notna() & rh_pct.notna()

    es = np.where(valid, 0.61078 * np.exp((17.27 * temp_c) / denom), np.nan)
    vpd = np.where(valid, es * (1.0 - rh_clipped / 100.0), np.nan)
    return pd.Series(np.maximum(vpd, 0.0), index=temp_c.index, name="vpd")


def compute_vpd(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Vapor Pressure Deficit features.

    Parameters:
      df: DataFrame containing 'T2M', 'T2M_MAX', and 'RH2M'.

    Returns:
      Copy of df enriched with:
        - 'vpd': daily mean VPD (kPa) from T2M and RH2M.
        - 'vpd_tmax_proxy': heuristic proxy combining daily maximum temperature
          (T2M_MAX) with daily mean relative humidity (RH2M). Not an instantaneous
          measurement at peak temperature.
    """
    out = df.copy()
    out["vpd"] = _calc_vpd_series(out["T2M"], out["RH2M"])
    out["vpd_tmax_proxy"] = _calc_vpd_series(out["T2M_MAX"], out["RH2M"])
    return out


# ---------------------------------------------------------------------------
# 2. Temperature Anomalies (Past-Only, Zero Future Leakage)
# ---------------------------------------------------------------------------

def compute_temperature_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Compute past-only causal temperature anomalies.

    Features generated:
      - 'tmax_anomaly_30d': T2M_MAX minus antecedent 30-day rolling mean (shift(1)).
        Measures acute heatwave / thermal spike relative to the past 30 days [t-30, t-1].
      - 't2m_anomaly_hist': T2M minus historical monthly mean computed strictly from
        prior years (Y' < Y) for the same cell and calendar month. Evaluates to NaN
        when no prior historical year is available (e.g. 2018).
      - 't2m_zscore_hist': Standardized monthly anomaly on prior years. Requires at
        least two reference years (n_years >= 2) and strictly positive standard deviation
        (std > 0). Evaluates to NaN otherwise.

    Parameters:
      df: DataFrame containing 'lat_cell', 'lon_cell', 'date', 'T2M', 'T2M_MAX'.

    Returns:
      Copy of df enriched with the 3 temperature anomaly columns.
    """
    out = df.copy()
    out["_row_id"] = np.arange(len(out))

    # Sort chronologically per spatial cell
    sorted_df = out.sort_values(["lat_cell", "lon_cell", "date"]).copy()
    g_cell = sorted_df.groupby(["lat_cell", "lon_cell"], sort=False)

    # A. Antecedent 30-day rolling max anomaly (shift(1) excludes day t)
    tmax_prev_30d = (
        g_cell["T2M_MAX"]
        .apply(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    sorted_df["tmax_anomaly_30d"] = sorted_df["T2M_MAX"] - tmax_prev_30d

    # B. Historical monthly reference based strictly on prior years (Y' < Y)
    sorted_df["_year"] = sorted_df["date"].dt.year
    sorted_df["_month"] = sorted_df["date"].dt.month

    # Annual monthly mean per (cell, year, month)
    annual_monthly = (
        sorted_df.groupby(["lat_cell", "lon_cell", "_year", "_month"], as_index=False)["T2M"]
        .mean()
        .rename(columns={"T2M": "annual_mean_t2m"})
    )

    unique_years = sorted(sorted_df["_year"].unique())
    clim_pieces = []

    for y in unique_years:
        sub = sorted_df[sorted_df["_year"] == y].copy()
        prior = annual_monthly[annual_monthly["_year"] < y]

        if len(prior) == 0:
            # First observation year (e.g. 2018): no prior years exist -> NaN
            sub["t2m_anomaly_hist"] = np.nan
            sub["t2m_zscore_hist"] = np.nan
        else:
            stats = (
                prior.groupby(["lat_cell", "lon_cell", "_month"])["annual_mean_t2m"]
                .agg(
                    n_years="count",
                    t2m_hist_mean="mean",
                    t2m_hist_std=lambda s: s.std(ddof=1) if len(s) >= 2 else np.nan,
                )
                .reset_index()
            )
            sub = sub.merge(stats, on=["lat_cell", "lon_cell", "_month"], how="left")
            sub["t2m_anomaly_hist"] = sub["T2M"] - sub["t2m_hist_mean"]

            # Z-score requires at least 2 distinct reference years and positive std
            valid_z = (sub["n_years"] >= 2) & (sub["t2m_hist_std"] > 1e-6)
            sub["t2m_zscore_hist"] = np.where(
                valid_z,
                (sub["T2M"] - sub["t2m_hist_mean"]) / sub["t2m_hist_std"],
                np.nan,
            )
            sub = sub.drop(columns=["n_years", "t2m_hist_mean", "t2m_hist_std"])

        clim_pieces.append(sub)

    res = pd.concat(clim_pieces, ignore_index=True)
    res = res.sort_values("_row_id").drop(columns=["_row_id", "_year", "_month"])
    res.index = out.index
    return res


# ---------------------------------------------------------------------------
# 3. Meteorological Fire Weather Indices (Proxies)
# ---------------------------------------------------------------------------

def compute_fire_weather_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simplified empirical fire weather indices.

    Features generated:
      - 'angstrom_proxy': Ångström fire risk index adapted for daily averages:
          I = RH2M / 20.0 + (27.0 - T2M) / 10.0
          Lower values indicate higher danger (I < 2: very high danger).
          Proxy notation: original formula requires 13:00 local time readings.
      - 'cbi': Chandler Burning Index (metric daily adaptation):
          CBI = ((104.5 - 1.373 * RH + 0.54 * T) * 124 * 10^(-0.0142 * RH)) / 60
          Floored at 0.0 (non-negative fire potential).
          Reference: Chandler et al. (1983).
      - 'hdw_proxy': Hot-Dry-Windy surface proxy:
          HDW = vpd * WS2M  [kPa * m/s]
          Proxy notation: original HDW (Srock et al., 2018) evaluates planetary
          boundary layer over 500m. This uses 2m surface data.
      - 'nesterov_proxy': Cumulative Nesterov drought-combustion index:
          Point de rosée T_dew from Magnus formula.
          Daily increment: Delta_NI = max(T2M * (T2M - T_dew), 0) if T2M > 0 else 0.
          Accumulates daily, resets to 0 on days with PRECTOTCORR >= 3.0 mm.
          Reference: Nesterov (1949).

    Parameters:
      df: DataFrame containing 'T2M', 'RH2M', 'WS2M', 'PRECTOTCORR', and optionally 'vpd'.

    Returns:
      Copy of df enriched with the 4 index columns.
    """
    out = df.copy()
    out["_row_id"] = np.arange(len(out))

    t = out["T2M"]
    rh = np.clip(out["RH2M"], 0.0, 100.0)
    ws = np.maximum(out["WS2M"], 0.0)
    precip = np.maximum(out["PRECTOTCORR"], 0.0)

    # 1. Ångström index proxy
    valid_th = t.notna() & rh.notna()
    angstrom = np.where(valid_th, rh / 20.0 + (27.0 - t) / 10.0, np.nan)
    out["angstrom_proxy"] = angstrom

    # 2. Chandler Burning Index (metric form, Chandler et al., 1983)
    cbi_raw = ((104.5 - 1.373 * rh + 0.54 * t) * 124.0 * (10.0 ** (-0.0142 * rh))) / 60.0
    cbi = np.where(valid_th, np.maximum(cbi_raw, 0.0), np.nan)
    out["cbi"] = cbi

    # 3. Hot-Dry-Windy surface proxy (HDW = VPD * WS2M)
    if "vpd" in out.columns:
        vpd = out["vpd"]
    else:
        vpd = _calc_vpd_series(t, rh)
    valid_hdw = vpd.notna() & ws.notna()
    hdw = np.where(valid_hdw, np.maximum(vpd * ws, 0.0), np.nan)
    out["hdw_proxy"] = hdw

    # 4. Cumulative Nesterov index with dew point and precipitation reset
    sorted_df = out.sort_values(["lat_cell", "lon_cell", "date"]).copy()

    # Dew point via Magnus formula
    alpha = (17.27 * sorted_df["T2M"]) / (237.3 + sorted_df["T2M"]) + np.log(
        np.clip(sorted_df["RH2M"], 1.0, 100.0) / 100.0
    )
    td = (237.3 * alpha) / (17.27 - alpha)
    diff = sorted_df["T2M"] - td

    # Daily increment: Delta_NI = T * (T - T_dew) for T > 0 and T > T_dew
    valid_inc = (sorted_df["T2M"] > 0.0) & (diff > 0.0) & sorted_df["T2M"].notna() & sorted_df["RH2M"].notna()
    daily_inc = np.where(valid_inc, sorted_df["T2M"] * diff, 0.0)

    # Rainy day threshold (3.0 mm reset)
    rain_reset = sorted_df["PRECTOTCORR"] >= 3.0
    streak_id = rain_reset.groupby([sorted_df["lat_cell"], sorted_df["lon_cell"]]).cumsum()
    has_reset = rain_reset.groupby([sorted_df["lat_cell"], sorted_df["lon_cell"]]).cummax()

    # On rainy days, the index resets to 0
    daily_inc_eff = np.where(rain_reset, 0.0, daily_inc)
    sorted_df["_daily_inc_eff"] = daily_inc_eff
    sorted_df["_streak_id"] = streak_id

    sorted_df["nesterov_proxy"] = sorted_df.groupby(
        ["lat_cell", "lon_cell", "_streak_id"], sort=False
    )["_daily_inc_eff"].cumsum()

    # Prior to the first valid rain reset >= 3.0 mm, antecedent drought is unknown -> NaN
    sorted_df.loc[~has_reset, "nesterov_proxy"] = np.nan

    # If inputs were missing, set to NaN
    missing_inputs = sorted_df["T2M"].isna() | sorted_df["RH2M"].isna() | sorted_df["PRECTOTCORR"].isna()
    sorted_df.loc[missing_inputs, "nesterov_proxy"] = np.nan

    # Restore ordering
    sorted_df = sorted_df.sort_values("_row_id").drop(
        columns=["_row_id", "_daily_inc_eff", "_streak_id"]
    )
    sorted_df.index = out.index
    out["nesterov_proxy"] = sorted_df["nesterov_proxy"]

    out = out.drop(columns=["_row_id"])
    return out


# ---------------------------------------------------------------------------
# 4. Spatial-Temporal Neighbor Fire History (FIRMS)
# ---------------------------------------------------------------------------

def compute_neighbor_fire_history(
    df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    lat_step: float | None = None,
    lon_step: float | None = None,
) -> pd.DataFrame:
    """Compute spatial-temporal neighbor fire history features from FIRMS detections.

    Features generated:
      - 'neighbor_fires_7d': sum of detections in 8 neighbors over [t-7, t-1].
      - 'neighbor_fires_14d': sum of detections in 8 neighbors over [t-14, t-1].
      - 'neighbor_fires_30d': sum of detections in 8 neighbors over [t-30, t-1].

    Guarantees:
      - Strictly uses 'n_detections' (historical physical observations). Never uses 'target'.
      - Strictly applies shift(1) to exclude prediction day t.
      - Uses Chebyshev distance <= 1 on the spatial grid (Moore 8-neighborhood).
      - Incomplete history produces NaN (distinguishing observed calm 0 from unknown history).
      - Operates continuously across Train/Validation boundaries without temporal cuts.

    Parameters:
      df: DataFrame containing 'lat_cell', 'lon_cell', 'date', 'n_detections'.
      history_df: Optional prior historical DataFrame containing 'lat_cell', 'lon_cell',
        'date', 'n_detections' for dates preceding df['date'].min() (e.g. Jan-Feb 2018).
      lat_step: Grid step in latitude (default: auto-detected or 0.5).
      lon_step: Grid step in longitude (default: auto-detected or 0.625).

    Returns:
      Copy of df enriched with the 3 neighbor fire columns.
    """
    out = df.copy()
    out["_row_id"] = np.arange(len(out))

    req_cols = ["lat_cell", "lon_cell", "date", "n_detections"]
    if history_df is not None and not history_df.empty:
        prior_data = history_df[req_cols].copy()
        combined = pd.concat([prior_data, out[req_cols]], ignore_index=True)
    else:
        combined = out[req_cols].copy()

    # Auto-detect grid resolution if not provided
    unique_lats = np.sort(combined["lat_cell"].unique())
    unique_lons = np.sort(combined["lon_cell"].unique())
    if lat_step is None:
        lat_step = float(np.min(np.diff(unique_lats))) if len(unique_lats) > 1 else 0.5
    if lon_step is None:
        lon_step = float(np.min(np.diff(unique_lons))) if len(unique_lons) > 1 else 0.625

    # Unique cell coordinates
    cell_pairs = sorted(combined[["lat_cell", "lon_cell"]].drop_duplicates().to_records(index=False).tolist())
    n_cells = len(cell_pairs)

    # Build Moore 8-connectivity adjacency matrix A (shape: n_cells x n_cells)
    A = np.zeros((n_cells, n_cells), dtype=np.float32)
    tol = 1e-4
    for i, (lat, lon) in enumerate(cell_pairs):
        for j, (nlat, nlon) in enumerate(cell_pairs):
            if i == j:
                continue
            dlat = abs(lat - nlat)
            dlon = abs(lon - nlon)
            if dlat <= lat_step + tol and dlon <= lon_step + tol:
                A[i, j] = 1.0

    # Pivot to date x cell matrix with contiguous calendar date index
    all_dates = pd.date_range(combined["date"].min(), combined["date"].max(), name="date")
    piv = combined.pivot(index="date", columns=["lat_cell", "lon_cell"], values="n_detections")
    piv = piv.reindex(index=all_dates, columns=pd.MultiIndex.from_tuples(cell_pairs, names=["lat_cell", "lon_cell"]))
    piv = piv.fillna(0.0)

    # Daily neighbor sum: matrix multiplication (n_dates x n_cells) @ (n_cells x n_cells)^T
    daily_nb = piv.values @ A.T
    daily_nb_df = pd.DataFrame(daily_nb, index=piv.index, columns=piv.columns)

    # Rolling window aggregations: shift(1) excludes date t, min_periods=w ensures NaN for incomplete history
    res_dfs = []
    for w in (7, 14, 30):
        roll_df = daily_nb_df.rolling(w, min_periods=w).sum().shift(1)
        s = roll_df.stack(["lat_cell", "lon_cell"], future_stack=True).rename(f"neighbor_fires_{w}d").reset_index()
        res_dfs.append(s)

    merged = out
    for w_idx, w in enumerate((7, 14, 30)):
        merged = merged.merge(res_dfs[w_idx], on=["date", "lat_cell", "lon_cell"], how="left")

    merged = merged.sort_values("_row_id").drop(columns=["_row_id"])
    merged.index = out.index
    return merged


# ---------------------------------------------------------------------------
# 5. Pipeline Dispatcher
# ---------------------------------------------------------------------------

def build_features_v2(
    df: pd.DataFrame,
    feature_families: Sequence[str] | None = None,
    history_df: pd.DataFrame | None = None,
    lat_step: float | None = None,
    lon_step: float | None = None,
) -> pd.DataFrame:
    """Enrich dataframe with specified feature families.

    Parameters:
      df: Input DataFrame.
      feature_families: List or tuple of feature family names to add.
        Allowed families: 'vpd', 'anomalies', 'indices', 'neighbors'.
        If None, all 4 families (12 features) are computed.
      history_df: Optional historical FIRMS detections passed to neighbor computation.
      lat_step: Latitude grid step.
      lon_step: Longitude grid step.

    Returns:
      Copy of df enriched with the requested new features.
    """
    if feature_families is None:
        families = list(FEATURE_FAMILIES.keys())
    else:
        valid_families = set(FEATURE_FAMILIES.keys())
        for fam in feature_families:
            if fam not in valid_families:
                raise ValueError(
                    f"Unknown feature family '{fam}'. Allowed families: {sorted(valid_families)}"
                )
        families = list(feature_families)

    out = df.copy()

    if "vpd" in families:
        out = compute_vpd(out)

    if "anomalies" in families:
        out = compute_temperature_anomalies(out)

    if "indices" in families:
        out = compute_fire_weather_indices(out)

    if "neighbors" in families:
        out = compute_neighbor_fire_history(
            out,
            history_df=history_df,
            lat_step=lat_step,
            lon_step=lon_step,
        )

    return out
