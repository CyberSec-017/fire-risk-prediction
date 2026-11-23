"""Streamlit view for interactive data-quality and exploratory analysis.

All data preparation lives in the pure ``src/eda.py`` module; this file only
owns widgets, Plotly figures and narrative text. It is rendered inside the
existing UI V2 navigation and never replaces the satellite cockpit.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
for candidate in (str(SRC_DIR), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import eda as eda_logic  # noqa: E402
from dashboard.config import COLORS, FIRMS_RAW_PATH  # noqa: E402
from dashboard.data_loader import load_fire_panel, load_fires_daily  # noqa: E402
from dashboard.data_quality_view import (  # noqa: E402
    build_pipeline_quality_reports,
    render_data_quality_summary,
)
from data_quality import column_profile, invalid_value_counts  # noqa: E402

_FIRE_COLOR = COLORS["risk_very_high"]
_WEATHER_COLOR = COLORS["brand_primary"]
_NEUTRAL_COLOR = COLORS["text_muted"]

_SOURCE_NOTES = {
    "fires": "Source : NASA FIRMS — détections historiques agrégées par cellule et par jour.",
    "firms": "Source : NASA FIRMS — lignes de détection brutes ; les lignes incomplètes sont exclues et comptées.",
    "panel": (
        "Source : NASA POWER (réanalyse météo) et proxy FIRMS « burned » (cellule-jour avec détection). "
        "Ce proxy n'est ni une surface brûlée confirmée ni un dégât."
    ),
}


@st.cache_data(show_spinner="Lecture des détections FIRMS brutes…")
def _load_firms_raw() -> pd.DataFrame | None:
    if not FIRMS_RAW_PATH.exists():
        return None
    return pd.read_csv(FIRMS_RAW_PATH, low_memory=False)


@st.cache_data(show_spinner="Chargement des jeux d'analyse…")
def get_eda_datasets() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Load the three source frames once per Streamlit cache."""
    return load_fire_panel(), load_fires_daily(), _load_firms_raw()


@st.cache_data(show_spinner=False)
def _cached_quality_reports():
    return build_pipeline_quality_reports()


def _style_figure(
    fig: go.Figure,
    *,
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 340,
) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", font=dict(size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=48, r=28, t=48, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color=COLORS["text_secondary"], size=11),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=COLORS["border_light"]),
    )
    if x_title:
        fig.update_xaxes(title_text=x_title, gridcolor=COLORS["surface_subtle"], zeroline=False)
    if y_title:
        fig.update_yaxes(title_text=y_title, gridcolor=COLORS["surface_subtle"], zeroline=False)
    return fig


def _empty_notice(chart_id: str) -> None:
    st.info(f"Aucune donnée disponible pour « {eda_logic.VISUALIZATION_TITLES[chart_id]} » avec ces filtres.")


def _render_chart(chart_id: str, data: eda_logic.FilteredEdaData, variable: str) -> None:
    """Render one chart from a pure chart-input table."""
    meta = next(v for v in eda_logic.VISUALIZATIONS if v.id == chart_id)
    st.markdown(f"##### {meta.title}")
    fires, firms, panel = data.fires, data.firms, data.panel

    if chart_id == "monthly_detections":
        frame = eda_logic.monthly_detections(fires)
        if frame.empty:
            return _empty_notice(chart_id)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=frame["month"], y=frame["detections"], name="Détections FIRMS",
            marker_color=_FIRE_COLOR,
            hovertemplate="%{x}<br>Détections : %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=frame["month"], y=frame["active_cells"], name="Cellules actives",
            mode="lines+markers", line=dict(color=_WEATHER_COLOR, width=2), yaxis="y2",
            hovertemplate="%{x}<br>Cellules actives : %{y:,}<extra></extra>",
        ))
        fig.update_layout(
            yaxis2=dict(title="Cellules actives", overlaying="y", side="right", showgrid=False),
            barmode="group",
        )
        _style_figure(fig, title="Volume mensuel de détections et cellules actives",
                      x_title="Mois", y_title="Nombre de détections")

    elif chart_id == "seasonal_detections":
        frame = eda_logic.seasonal_detections(fires)
        if frame.empty or frame["detections"].sum() == 0:
            return _empty_notice(chart_id)
        fig = px.bar(frame, x="season", y="detections", text="detections",
                     color="detections", color_continuous_scale="Oranges")
        fig.update_traces(textposition="outside", marker_line_color="#C2410C")
        fig.update_layout(coloraxis_showscale=False)
        _style_figure(fig, title="Détections par saison météorologique",
                      x_title="Saison", y_title="Nombre de détections")

    elif chart_id == "top_cells":
        frame = eda_logic.top_detection_cells(fires)
        if frame.empty:
            return _empty_notice(chart_id)
        frame = frame.sort_values("detections")
        fig = px.bar(frame, x="detections", y="cell", orientation="h", text="detections",
                     hover_data=["latitude", "longitude", "active_days"],
                     color="detections", color_continuous_scale="Reds")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        _style_figure(fig, title="15 cellules avec le plus de détections",
                      x_title="Détections cumulées", y_title="Cellule NASA POWER", height=430)

    elif chart_id == "geographic_detections":
        frame = eda_logic.detection_scatter(fires)
        if frame.empty:
            return _empty_notice(chart_id)
        fig = px.scatter(
            frame, x="longitude", y="latitude", size="detections", size_max=20,
            color="detections", color_continuous_scale="YlOrRd",
            hover_data={"detections": ":,", "latitude": ":.3f", "longitude": ":.3f"},
            range_x=eda_logic.STUDY_AREA["longitude"],
            range_y=eda_logic.STUDY_AREA["latitude"],
        )
        fig.update_layout(coloraxis_showscale=False)
        _style_figure(fig, title="Répartition géographique des détections FIRMS",
                      x_title="Longitude (°E)", y_title="Latitude (°N)", height=420)

    elif chart_id == "latitude_bands":
        frame = eda_logic.latitude_band_detections(fires)
        fig = px.bar(frame, x="band", y="detections", text="detections",
                     hover_data=["active_cells"], color="detections",
                     color_continuous_scale="Reds")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        _style_figure(fig, title="Détections par bande latitudinale (découpage indicatif)",
                      x_title="Bande", y_title="Nombre de détections")

    elif chart_id == "confidence_distribution":
        frame = eda_logic.confidence_distribution(firms)
        if frame.empty:
            return _empty_notice(chart_id)
        fig = px.bar(frame, x="confidence", y="detections", text="percent",
                     color="confidence",
                     color_discrete_map={
                         "Nominale (n)": COLORS["text_muted"],
                         "Faible (l)": COLORS["risk_moderate"],
                         "Élevée (h)": _FIRE_COLOR,
                         "Non renseignée": COLORS["border_subtle"],
                     })
        fig.update_traces(texttemplate="%{text:.1f} %", textposition="outside")
        _style_figure(fig, title="Niveaux de confiance des détections FIRMS",
                      x_title="Confiance déclarée par le capteur", y_title="Détections")

    elif chart_id == "frp_distribution":
        frame = eda_logic.frp_distribution(firms)
        fig = go.Figure(go.Bar(
            x=frame["frp_bin"], y=frame["detections"], marker_color=_FIRE_COLOR,
            hovertemplate="FRP %{x} MW<br>Détections : %{y:,}<extra></extra>",
        ))
        fig.update_yaxes(type="log", title_text="Détections (échelle log)")
        _style_figure(fig, title="Puissance radiative des feux (FRP, MW)",
                      x_title="Classe de FRP (MW)", y_title="Détections (log)")

    elif chart_id in {"temperature_distribution", "humidity_distribution",
                      "wind_distribution", "precipitation_distribution", "selected_variable"}:
        selected = {
            "temperature_distribution": "T2M_MAX",
            "humidity_distribution": "RH2M",
            "wind_distribution": "WS2M",
            "precipitation_distribution": "PRECTOTCORR",
            "selected_variable": variable,
        }[chart_id]
        frame = eda_logic.weather_histogram(panel, selected)
        if frame.empty:
            return _empty_notice(chart_id)
        meta_var = eda_logic.WEATHER_VARIABLES.get(selected, {"label": selected, "unit": ""})
        y_title = f"Cellules-jours ({meta_var['unit']})" if meta_var["unit"] else "Cellules-jours"
        fig = go.Figure()
        fig.add_trace(go.Bar(x=frame["bin_label"], y=frame["without_detection"],
                             name="Sans détection FIRMS", marker_color=_NEUTRAL_COLOR,
                             hovertemplate="%{x}<br>Sans détection : %{y:,}<extra></extra>"))
        fig.add_trace(go.Bar(x=frame["bin_label"], y=frame["with_detection"],
                             name="Avec détection FIRMS", marker_color=_FIRE_COLOR,
                             hovertemplate="%{x}<br>Avec détection : %{y:,}<extra></extra>"))
        fig.update_layout(barmode="stack")
        _style_figure(fig, title=f"{meta_var['label']} — distribution",
                      x_title=f"{meta_var['label']} ({meta_var['unit']})", y_title=y_title)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=9))

    elif chart_id == "weather_comparison":
        frame = eda_logic.detection_rate_by_weather_class(panel, variable)
        if frame.empty:
            return _empty_notice(chart_id)
        fig = px.bar(frame, x="weather_class", y="fire_rate_pct", text="fire_rate_pct",
                     hover_data=["cell_days"], color="fire_rate_pct",
                     color_continuous_scale="Oranges")
        fig.update_traces(texttemplate="%{text:.2f} %", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        meta_var = eda_logic.WEATHER_VARIABLES.get(variable, {"label": variable, "unit": ""})
        _style_figure(fig, title=f"Taux de détection par classe de {meta_var['label'].lower()}",
                      x_title=f"Classe de {meta_var['label'].lower()} ({meta_var['unit']})",
                      y_title="Cellules-jours avec détection (%)")

    elif chart_id == "correlation_matrix":
        frame = eda_logic.correlation_matrix(panel)
        if frame.empty:
            return _empty_notice(chart_id)
        labels = [eda_logic.WEATHER_VARIABLES.get(c, {}).get("label", c) for c in frame.columns]
        fig = go.Figure(go.Heatmap(
            z=frame.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1,
            colorscale=[[0, "#1E3A8A"], [0.5, "#F8FAFC"], [1, "#DC2626"]],
            text=np.round(frame.to_numpy(), 2), texttemplate="%{text}",
            textfont=dict(size=9), colorbar=dict(title="Pearson r"),
        ))
        _style_figure(fig, title="Corrélations de Pearson (variables numériques)", height=460)

    elif chart_id == "monthly_weather_trend":
        frame = eda_logic.monthly_weather_trend(panel, variable)
        if frame.empty:
            return _empty_notice(chart_id)
        meta_var = eda_logic.WEATHER_VARIABLES.get(variable, {"label": variable, "unit": ""})
        fig = go.Figure()
        fig.add_trace(go.Bar(x=frame["month"], y=frame["fire_rate_pct"], name="Cellules détectées (%)",
                             marker_color="#FCA5A5",
                             hovertemplate="%{x}<br>Détections : %{y:.2f} %<extra></extra>", yaxis="y2"))
        fig.add_trace(go.Scatter(x=frame["month"], y=frame["mean_weather"],
                                 name=f"Moyenne {meta_var['label'].lower()}",
                                 mode="lines", line=dict(color=_WEATHER_COLOR, width=2),
                                 hovertemplate="%{x}<br>Moyenne : %{y:.2f}<extra></extra>"))
        fig.update_layout(yaxis2=dict(title="Cellules détectées (%)", overlaying="y",
                                      side="right", showgrid=False))
        _style_figure(fig, title="Dynamique mensuelle météo / détections",
                      x_title="Mois", y_title=f"{meta_var['label']} ({meta_var['unit']})")

    else:  # pragma: no cover - guarded by the registry
        st.warning(f"Visualisation inconnue : {chart_id}")
        return

    st.plotly_chart(fig, width="stretch", alt=meta.title)
    st.caption(f"{meta.description} {_SOURCE_NOTES[meta.source]}")


def _render_quality_tab(
    panel: pd.DataFrame,
    fires: pd.DataFrame,
    firms: pd.DataFrame,
    raw_firms: pd.DataFrame | None,
    firms_stats: dict[str, Any],
) -> None:
    st.markdown("##### 🔎 Dimensions, valeurs manquantes, doublons et types")
    render_data_quality_summary(
        _cached_quality_reports(),
        title="Rapport qualité des jeux traités (fire_panel, fires_daily)",
        expanded=True,
    )

    st.markdown("##### 📅 / 🗺️ Couverture temporelle et géographique")
    coverage = eda_logic.coverage_frame(panel, fires, firms)
    st.dataframe(coverage, hide_index=True, width="stretch",
                 alt="Temporal and geographic coverage of the three EDA datasets")
    st.caption(
        "Les dates et coordonnées proviennent des données observées. "
        "« outside_study_area » compte les détections situées hors du domaine du modèle "
        "(32,0–37,5°N ; -2,0–9,0°E) : elles sont conservées dans l'EDA historique mais "
        "ne doivent pas être utilisées pour une inférence."
    )

    if raw_firms is not None:
        st.markdown("##### 🧪 Valeurs suspectes dans le fichier FIRMS brut")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Lignes lues", f"{firms_stats['input_rows']:,}")
        col_b.metric("Lignes exploitables", f"{firms_stats['valid_rows']:,}")
        col_c.metric("Lignes exclues", f"{firms_stats['dropped_rows']:,}")
        st.dataframe(column_profile(raw_firms), hide_index=True, width="stretch",
                     alt="Missing values by column in the raw FIRMS file")
        suspicious = invalid_value_counts(
            firms,
            numeric_ranges={
                "latitude": (eda_logic.LAT_MIN - 2, eda_logic.LAT_MAX + 2),
                "longitude": (eda_logic.LON_MIN - 2, eda_logic.LON_MAX + 2),
                "brightness": (180.0, 500.0),
                "frp": (0.0, 2000.0),
            },
        )
        st.dataframe(suspicious, hide_index=True, width="stretch",
                     alt="Suspicious value counts in validated FIRMS detections")
        st.caption(
            "Les lignes exclues sont des lignes incomplètes du fichier brut (aucune date ou "
            "coordonnée exploitable). Les valeurs hors plages sont signalées sans être corrigées."
        )


def render_eda_view() -> None:
    """Render the interactive EDA section inside the existing UI V2 shell."""
    st.html(
        """
        <div style="margin-bottom: 12px;">
            <div style="font-size: 1.5rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">
                📈 Qualité des données & Analyse exploratoire (EDA)
            </div>
            <div style="font-size: 0.9rem; color: #475569; margin-top: 4px;">
                Exploration interactive des observations NASA FIRMS et des réanalyses NASA POWER sur le Nord algérien.
            </div>
        </div>
        """
    )
    st.info(
        "**Périmètre scientifique** — seules des observations historiques sont affichées : "
        "détections satellite FIRMS et conditions météorologiques POWER. "
        "Aucun risque prédit, aucune surface brûlée confirmée et aucun dégât ne sont représentés ici. "
        "Les corrélations décrivent des associations et n'impliquent pas de causalité."
    )

    panel, fires, raw_firms = get_eda_datasets()
    if panel is None and fires is None:
        st.warning("Aucun jeu de données disponible : exécutez le pipeline de préparation avant l'EDA.")
        return

    panel = panel if panel is not None else pd.DataFrame()
    fires = fires if fires is not None else pd.DataFrame()
    firms, firms_stats = eda_logic.prepare_firms(raw_firms if raw_firms is not None else pd.DataFrame())

    # -------------------------------------------------------------------------
    # Interactive filters
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("**Filtres interactifs**")
        date_bounds = eda_logic.date_coverage(fires if not fires.empty else panel)
        fallback_start = date_bounds["start"].date() if date_bounds["start"] is not None else None
        fallback_end = date_bounds["end"].date() if date_bounds["end"] is not None else None
        selected_dates = st.date_input(
            "Période d'observation (détections FIRMS)",
            value=(fallback_start, fallback_end) if fallback_start and fallback_end else None,
            key="eda_date_range",
            help="La plage est inclusive. Une sélection incomplète ou inversée est signalée et neutralisée.",
        )
        if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
            chosen_start, chosen_end = selected_dates
        else:
            chosen_start, chosen_end = selected_dates, None

        lat_range = st.slider(
            "Latitude (°N)",
            min_value=float(eda_logic.STUDY_AREA["latitude"][0]),
            max_value=float(eda_logic.STUDY_AREA["latitude"][1]),
            value=tuple(float(v) for v in eda_logic.STUDY_AREA["latitude"]),
            step=0.1, key="eda_lat_range",
        )
        lon_range = st.slider(
            "Longitude (°E)",
            min_value=float(eda_logic.STUDY_AREA["longitude"][0]),
            max_value=float(eda_logic.STUDY_AREA["longitude"][1]),
            value=tuple(float(v) for v in eda_logic.STUDY_AREA["longitude"]),
            step=0.125, key="eda_lon_range",
        )
        col_choice, col_var, col_export = st.columns([2.0, 1.2, 1.4])
        with col_choice:
            selected_charts = st.multiselect(
                "Visualisations affichées",
                options=list(eda_logic.VISUALIZATION_IDS),
                default=list(eda_logic.VISUALIZATION_IDS),
                format_func=lambda chart_id: eda_logic.VISUALIZATION_TITLES[chart_id],
                key="eda_charts",
            )
        with col_var:
            selected_variable = st.selectbox(
                "Variable météo",
                options=list(eda_logic.WEATHER_VARIABLES),
                format_func=lambda key: (
                    f"{eda_logic.WEATHER_VARIABLES[key]['label']} ({eda_logic.WEATHER_VARIABLES[key]['unit']})"
                ),
                key="eda_variable",
            )
        with col_export:
            export_choice = st.selectbox(
                "Export CSV",
                options=["Détections agrégées (journalières)", "Détections FIRMS brutes validées"],
                key="eda_download_dataset",
            )

    start, end, date_note = eda_logic.normalize_date_range(chosen_start, chosen_end)
    lat_bounds, lat_note = eda_logic.normalize_geo_range(lat_range, "latitude")
    lon_bounds, lon_note = eda_logic.normalize_geo_range(lon_range, "longitude")
    notes = [note for note in (date_note, lat_note, lon_note) if note]
    filters = eda_logic.EdaFilters(
        start=start, end=end, lat_range=lat_bounds, lon_range=lon_bounds,
        notes=tuple(notes),
    )
    for note in notes:
        st.warning(note)

    data = eda_logic.apply_filters(panel, fires, firms, filters)
    period = (
        f"{start.date()} → {end.date()}" if start is not None and end is not None else "période complète"
    )
    st.caption(
        f"**Période analysée :** {period} · **Panel :** {len(data.panel):,} cellules-jours · "
        f"**Détections agrégées :** {len(data.fires):,} lignes · "
        f"**FIRMS brutes validées :** {len(data.firms):,} lignes"
    )

    export = data.fires if export_choice.startswith("Détections agrégées") else data.firms
    st.download_button(
        "⬇️ Télécharger les données filtrées (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=(
            "firms_detections_daily_filtered.csv" if export_choice.startswith("Détections agrégées")
            else "firms_detections_raw_filtered.csv"
        ),
        mime="text/csv",
        key="eda_download_csv",
    )

    tab_charts, tab_quality = st.tabs(["📊 Visualisations", "🔎 Qualité & couverture"])
    with tab_charts:
        if not selected_charts:
            st.info("Sélectionnez au moins une visualisation dans les filtres.")
        else:
            for chart_id in selected_charts:
                _render_chart(chart_id, data, selected_variable)
        with st.expander("🧭 Lecture scientifique (associations, pas de causalité)", expanded=False):
            st.markdown(
                "- La saisonnalité des détections FIRMS est très concentrée sur l'été : "
                "elle reflète l'activité observationnelle et les conditions estivales, pas une preuve de cause.\n"
                "- Les températures élevées, une faible humidité et un sol sec sont associés aux cellules-jours "
                "avec détection ; ces variables sont corrélées entre elles.\n"
                "- Une corrélation de Pearson n'établit pas un mécanisme : la fréquentation, la végétation et "
                "les biais de détection satellite peuvent expliquer une partie des motifs observés.\n"
                "- Les FIRMS brutes sont des détections thermiques, pas des surfaces brûlées mesurées au sol."
            )
    with tab_quality:
        _render_quality_tab(panel, fires, firms, raw_firms, firms_stats)
