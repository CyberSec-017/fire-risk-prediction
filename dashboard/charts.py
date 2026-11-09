"""
FIREWATCH AI — Scientific Data Visualization Module (Plotly).

Constructs publication-quality, interactive charts adhering to the
clean environmental intelligence design system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.config import (
    COLORS,
    PLOTLY_TEMPLATE,
    PLOTLY_BG,
    PLOTLY_PAPER_BG,
    PLOTLY_GRID_COLOR,
    PLOTLY_FONT_FAMILY,
    WEATHER_VARIABLES,
)


def _apply_theme(
    fig: go.Figure,
    title: str = "",
    subtitle: str = "",
    height: int = 380,
) -> go.Figure:
    """Apply the unified scientific design system to any Plotly chart."""
    header_text = title
    if subtitle:
        header_text = f"<b>{title}</b><br><span style='font-size:11px; font-weight:normal; color:{COLORS['text_muted']};'>{subtitle}</span>"

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text=header_text,
            font=dict(
                size=13,
                color=COLORS["text_primary"],
                family=PLOTLY_FONT_FAMILY,
            ),
            x=0,
            xanchor="left",
            y=0.96,
        ) if title else None,
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG,
        font=dict(
            family=PLOTLY_FONT_FAMILY,
            color=COLORS["text_secondary"],
            size=11,
        ),
        height=height,
        margin=dict(l=46, r=18, t=52 if title else 24, b=42),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=COLORS["text_secondary"]),
            bgcolor="rgba(255,255,255,0.7)",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=PLOTLY_GRID_COLOR,
            zeroline=False,
            linecolor=COLORS["border_light"],
            tickfont=dict(size=10, color=COLORS["text_muted"]),
            title_font=dict(size=11, color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=PLOTLY_GRID_COLOR,
            zeroline=False,
            linecolor=COLORS["border_light"],
            tickfont=dict(size=10, color=COLORS["text_muted"]),
            title_font=dict(size=11, color=COLORS["text_secondary"]),
        ),
        hoverlabel=dict(
            bgcolor=COLORS["surface_white"],
            bordercolor=COLORS["border_light"],
            font_size=11,
            font_family=PLOTLY_FONT_FAMILY,
            font_color=COLORS["text_primary"],
        ),
    )
    return fig


# ──────────────────────────────────────────────
# Temporal Fire Trends
# ──────────────────────────────────────────────

def fire_detections_timeline(
    df: pd.DataFrame,
    freq: str = "W",
) -> go.Figure:
    """Aggregated timeline of fire detection occurrences."""
    time_grouped = (
        df.groupby(pd.Grouper(key="date", freq=freq))["burned"]
        .sum()
        .reset_index()
        .rename(columns={"burned": "fire_cells"})
    )

    freq_name = {"D": "Daily", "W": "Weekly", "ME": "Monthly"}.get(freq, "Period")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_grouped["date"],
        y=time_grouped["fire_cells"],
        mode="lines",
        line=dict(color=COLORS["risk_very_high"], width=1.75),
        fill="tozeroy",
        fillcolor="rgba(220, 38, 38, 0.08)",
        name=f"{freq_name} Fire Cells",
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Active Cells:</b> %{y:,}<extra></extra>",
    ))

    # Add rolling moving average if daily or weekly
    if freq in ["D", "W"] and len(time_grouped) > 8:
        window = 30 if freq == "D" else 4
        rolling_series = time_grouped["fire_cells"].rolling(window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=time_grouped["date"],
            y=rolling_series,
            mode="lines",
            line=dict(color=COLORS["brand_primary"], width=1.5, dash="dash"),
            name=f"Trend ({window}{freq.lower()} avg)",
            hovertemplate="<b>Trend:</b> %{y:.1f}<extra></extra>",
        ))

    fig.update_xaxes(title_text="Observation Date")
    fig.update_yaxes(title_text="Active Grid Cells")
    return _apply_theme(
        fig,
        title=f"{freq_name} Fire Detection Activity",
        subtitle="Count of 0.5° × 0.625° cells with ≥1 confirmed satellite detection",
    )


def monthly_fire_distribution(df: pd.DataFrame) -> go.Figure:
    """Monthly seasonal distribution of fire cells."""
    df_copy = df.copy()
    df_copy["month_num"] = df_copy["date"].dt.month
    df_copy["month_name"] = df_copy["date"].dt.strftime("%b")

    monthly = (
        df_copy[df_copy["burned"] == 1]
        .groupby(["month_num", "month_name"])
        .size()
        .reset_index(name="fire_cells")
        .sort_values("month_num")
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["month_name"],
        y=monthly["fire_cells"],
        marker=dict(
            color=COLORS["risk_high"],
            line=dict(color=COLORS["risk_very_high"], width=0.5),
        ),
        hovertemplate="<b>Month:</b> %{x}<br><b>Fire Cells:</b> %{y:,}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Total Fire Cells")
    return _apply_theme(
        fig,
        title="Seasonal Fire Concentration",
        subtitle="Cumulative fire cell detections aggregated by calendar month",
    )


def yearly_fire_breakdown(df: pd.DataFrame) -> go.Figure:
    """Multi-year comparison grouped by month."""
    df_copy = df.copy()
    df_copy["year"] = df_copy["date"].dt.year
    df_copy["month_num"] = df_copy["date"].dt.month
    df_copy["month_name"] = df_copy["date"].dt.strftime("%b")

    grouped = (
        df_copy[df_copy["burned"] == 1]
        .groupby(["year", "month_num", "month_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["year", "month_num"])
    )

    year_colors = {
        2024: COLORS["text_secondary"],
        2025: COLORS["brand_primary"],
        2026: COLORS["risk_very_high"],
    }

    fig = go.Figure()
    for year in sorted(grouped["year"].unique()):
        sub = grouped[grouped["year"] == year]
        color = year_colors.get(year, COLORS["text_muted"])
        fig.add_trace(go.Bar(
            x=sub["month_name"],
            y=sub["count"],
            name=str(year),
            marker_color=color,
            hovertemplate=f"<b>Year:</b> {year}<br><b>Month:</b> %{{x}}<br><b>Cells:</b> %{{y:,}}<extra></extra>",
        ))

    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Fire Cells")
    return _apply_theme(
        fig,
        title="Year-over-Year Seasonal Comparison",
        subtitle="Monthly fire volume compared across 2024, 2025, and 2026",
    )


# ──────────────────────────────────────────────
# Spatial Distributions
# ──────────────────────────────────────────────

def fire_by_latitude_profile(df: pd.DataFrame) -> go.Figure:
    """Zonal profile of fire detections across latitude bands."""
    fire_records = df[df["burned"] == 1]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=fire_records["lat_cell"],
        nbinsx=18,
        marker=dict(
            color=COLORS["brand_primary"],
            line=dict(color=COLORS["surface_white"], width=1),
        ),
        hovertemplate="<b>Lat Band:</b> %{x:.2f}°N<br><b>Fire Cells:</b> %{y:,}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Latitude (°N)")
    fig.update_yaxes(title_text="Fire Cell Frequency")
    return _apply_theme(
        fig,
        title="Latitudinal Zonal Profile",
        subtitle="Distribution of fire cells from inland Tell Atlas to Mediterranean coast",
    )


def fire_by_longitude_profile(df: pd.DataFrame) -> go.Figure:
    """Meridional profile of fire detections across longitude bands."""
    fire_records = df[df["burned"] == 1]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=fire_records["lon_cell"],
        nbinsx=22,
        marker=dict(
            color=COLORS["risk_moderate"],
            line=dict(color=COLORS["surface_white"], width=1),
        ),
        hovertemplate="<b>Lon Band:</b> %{x:.2f}°E<br><b>Fire Cells:</b> %{y:,}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Longitude (°E)")
    fig.update_yaxes(title_text="Fire Cell Frequency")
    return _apply_theme(
        fig,
        title="Longitudinal Meridional Profile",
        subtitle="West-to-East spatial fire gradient (-2.0°W to 9.0°E)",
    )


# ──────────────────────────────────────────────
# Meteorological Analysis
# ──────────────────────────────────────────────

def weather_distribution_comparison(
    df: pd.DataFrame,
    var_key: str = "T2M_MAX",
) -> go.Figure:
    """Boxplot comparing environmental distribution on fire vs unburned days."""
    var_meta = WEATHER_VARIABLES.get(var_key, {"name": var_key, "unit": ""})
    sub = df[["burned", var_key]].dropna()

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=sub[sub["burned"] == 0][var_key],
        name="Unburned Cell-Days",
        marker_color=COLORS["text_secondary"],
        boxmean=True,
    ))
    fig.add_trace(go.Box(
        y=sub[sub["burned"] == 1][var_key],
        name="Active Fire Cell-Days",
        marker_color=COLORS["risk_very_high"],
        boxmean=True,
    ))

    fig.update_yaxes(title_text=f"{var_meta['name']} ({var_meta['unit']})")
    return _apply_theme(
        fig,
        title=f"{var_meta['name']} Separation",
        subtitle=f"Distribution comparison between fire incidents and non-fire days ({var_meta['unit']})",
    )


def weather_time_series_trend(
    df: pd.DataFrame,
    var_key: str = "T2M_MAX",
    freq: str = "W",
) -> go.Figure:
    """Weekly aggregated regional average of a weather variable."""
    var_meta = WEATHER_VARIABLES.get(var_key, {"name": var_key, "unit": ""})
    series = (
        df.groupby(pd.Grouper(key="date", freq=freq))[var_key]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    fig = go.Figure()
    # Shaded band for min-max envelope
    fig.add_trace(go.Scatter(
        x=series["date"],
        y=series["max"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=series["date"],
        y=series["min"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(15, 81, 50, 0.08)",
        name="Min–Max Range",
        hoverinfo="skip",
    ))
    # Mean line
    fig.add_trace(go.Scatter(
        x=series["date"],
        y=series["mean"],
        mode="lines",
        line=dict(color=COLORS["brand_primary"], width=1.75),
        name="Regional Mean",
        hovertemplate=f"<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Mean:</b> %{{y:.2f}} {var_meta['unit']}<extra></extra>",
    ))

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=f"{var_meta['name']} ({var_meta['unit']})")
    return _apply_theme(
        fig,
        title=f"{var_meta['name']} Regional Dynamics",
        subtitle=f"Temporal trajectory with regional range envelope ({var_meta['unit']})",
    )


def weather_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation matrix between weather parameters and the fire target."""
    keys = [k for k in WEATHER_VARIABLES if k in df.columns]
    cols = keys + (["burned"] if "burned" in df.columns else [])
    corr = df[cols].corr()

    labels = [WEATHER_VARIABLES.get(c, {}).get("name", c) for c in cols]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#1E3A8A"],
            [0.5, "#F8FAFC"],
            [1.0, "#DC2626"],
        ],
        zmin=-1.0,
        zmax=1.0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, family=PLOTLY_FONT_FAMILY),
        hoverongaps=False,
        colorbar=dict(
            title="Pearson r",
            tickfont=dict(size=10),
            len=0.8,
        ),
    ))

    fig.update_layout(height=460)
    return _apply_theme(
        fig,
        title="Environmental Correlation Matrix",
        subtitle="Pearson linear correlation coefficients between meteorological drivers and fire occurrence",
        height=460,
    )


def bivariate_weather_scatter(
    df: pd.DataFrame,
    var_x: str = "T2M_MAX",
    var_y: str = "RH2M",
    sample_size: int = 4000,
) -> go.Figure:
    """Bivariate scatter plot separating fire and non-fire days."""
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    meta_x = WEATHER_VARIABLES.get(var_x, {"name": var_x, "unit": ""})
    meta_y = WEATHER_VARIABLES.get(var_y, {"name": var_y, "unit": ""})

    fig = go.Figure()
    # Non-fire background points
    sub_0 = sample[sample["burned"] == 0]
    fig.add_trace(go.Scattergl(
        x=sub_0[var_x],
        y=sub_0[var_y],
        mode="markers",
        name="Unburned",
        marker=dict(
            size=3.5,
            color=COLORS["text_muted"],
            opacity=0.3,
        ),
        hovertemplate=f"{meta_x['name']}: %{{x:.1f}}<br>{meta_y['name']}: %{{y:.1f}}<extra></extra>",
    ))
    # Fire active points
    sub_1 = sample[sample["burned"] == 1]
    fig.add_trace(go.Scattergl(
        x=sub_1[var_x],
        y=sub_1[var_y],
        mode="markers",
        name="Confirmed Fire",
        marker=dict(
            size=6,
            color=COLORS["risk_very_high"],
            opacity=0.85,
            line=dict(color="#FFFFFF", width=0.75),
        ),
        hovertemplate=f"<b>FIRE</b><br>{meta_x['name']}: %{{x:.1f}}<br>{meta_y['name']}: %{{y:.1f}}<extra></extra>",
    ))

    fig.update_xaxes(title_text=f"{meta_x['name']} ({meta_x['unit']})")
    fig.update_yaxes(title_text=f"{meta_y['name']} ({meta_y['unit']})")
    return _apply_theme(
        fig,
        title=f"{meta_x['name']} vs. {meta_y['name']}",
        subtitle="Bivariate fire clustering in environmental feature space",
    )
