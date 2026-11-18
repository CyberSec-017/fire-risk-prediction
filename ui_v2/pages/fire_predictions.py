"""Fire Predictions Cockpit — Flagship Interactive Satellite Map Page.

Features:
  - 2D high-resolution satellite map (Esri World Imagery) with administrative overlay.
  - Visualization of the 209 active NASA POWER grid cells.
  - Point-and-click inference: clicking any location snaps to its cell, verifies
    coverage, and displays the risk evaluation modal (@st.dialog).
  - Session history of evaluated points with risk badges and probabilities.
"""

from __future__ import annotations

import textwrap
from typing import Any
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import app_core
from ui_v2.components.point_context import (
    is_point_in_active_cells,
    snap_to_grid,
)
from ui_v2.components.prediction_modal import render_prediction_dialog
from ui_v2.components.satellite_map import (
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
    build_satellite_map,
)


@st.cache_resource(show_spinner="Chargement du modèle de production...")
def get_model_bundle() -> app_core.ModelBundle:
    """Load and validate the 20-feature production model bundle."""
    return app_core.load_model_bundle()


@st.cache_data(show_spinner="Indexation du maillage NASA POWER...")
def get_fire_panel() -> pd.DataFrame | None:
    """Load fire panel dataset for localized historical statistics."""
    if app_core.FIRE_PANEL_PATH.exists():
        df = pd.read_parquet(app_core.FIRE_PANEL_PATH)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "burned" in df.columns:
            df["burned"] = df["burned"].astype(int)
        return df
    return None


def render_page() -> None:
    """Render the interactive satellite prediction cockpit."""
    bundle = get_model_bundle()
    panel_df = get_fire_panel()

    # Initialize session states
    if "predicted_points" not in st.session_state:
        st.session_state["predicted_points"] = []
    if "active_click" not in st.session_state:
        st.session_state["active_click"] = None
    if "last_handled_click" not in st.session_state:
        st.session_state["last_handled_click"] = None

    # -------------------------------------------------------------------------
    # Page Header
    # -------------------------------------------------------------------------
    st.html(
        textwrap.dedent(
            """
            <div style="margin-bottom: 16px;">
                <div style="font-size: 1.5rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">
                    🛰️ Cockpit de Prédiction Satellite
                </div>
                <div style="font-size: 0.9rem; color: #475569; margin-top: 4px;">
                    Cliquez directement sur la carte satellite pour évaluer le risque d'incendie sur la cellule
                    NASA POWER correspondante (Nord de l'Algérie, 209 cellules actives).
                </div>
            </div>
            """
        )
    )

    # -------------------------------------------------------------------------
    # Operational Status & Controls Toolbar
    # -------------------------------------------------------------------------
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2.0, 1.2, 1.0], vertical_alignment="center")

    with col_ctrl1:
        st.html(
            textwrap.dedent(
                """
                <div style="font-size: 0.85rem; color: #334155;">
                    💡 <b>Mode d'emploi :</b> Cliquez sur la zone d'intérêt forestière ou rurale.
                    Une fenêtre de diagnostic météorologique et de prédiction s'ouvre automatiquement.
                </div>
                """
            )
        )

    with col_ctrl2:
        show_grid = st.checkbox(
            "Afficher les 209 cellules NASA POWER",
            value=True,
            help="Affiche le contour des cellules géographiques d'apprentissage (0.5° × 0.625°)",
            key="toggle_grid_cells",
        )

    with col_ctrl3:
        if st.session_state["predicted_points"]:
            if st.button("Effacer l'historique", type="secondary", use_container_width=True):
                st.session_state["predicted_points"] = []
                st.session_state["active_click"] = None
                st.session_state["last_handled_click"] = None
                st.rerun()

    # -------------------------------------------------------------------------
    # Main Interactive Map & Side Context
    # -------------------------------------------------------------------------
    col_map, col_side = st.columns([3.2, 1.1])

    with col_map:
        m = build_satellite_map(
            predicted_points=st.session_state["predicted_points"],
            selected_point=st.session_state["active_click"],
            show_cells_grid=show_grid,
        )

        map_data = st_folium(
            m,
            returned_objects=["last_clicked"],
            width="100%",
            height=620,
            key="satellite_folium_view",
        )

    # Detect click from Folium
    new_click_detected = False
    if map_data and map_data.get("last_clicked"):
        raw_click = map_data["last_clicked"]
        c_lat = round(float(raw_click["lat"]), 4)
        c_lon = round(float(raw_click["lng"]), 4)
        current_click_pair = (c_lat, c_lon)

        if current_click_pair != st.session_state["last_handled_click"]:
            st.session_state["last_handled_click"] = current_click_pair
            st.session_state["active_click"] = current_click_pair
            new_click_detected = True

    # -------------------------------------------------------------------------
    # Side Panel: Active Point & Session History
    # -------------------------------------------------------------------------
    with col_side:
        active_pt = st.session_state["active_click"]
        if active_pt is not None:
            a_lat, a_lon = active_pt
            is_cov, lat_c, lon_c = is_point_in_active_cells(a_lat, a_lon)

            st.html(
                textwrap.dedent(
                    f"""
                    <div class="fwx-card" style="padding: 14px 16px; margin-bottom: 14px;">
                        <div style="font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase;">
                            Point Actif
                        </div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #0F172A; margin: 4px 0;">
                            {a_lat:.4f}°N, {a_lon:.4f}°E
                        </div>
                        <div style="font-size: 0.8rem; color: {'#059669' if is_cov else '#DC2626'}; font-weight: 600;">
                            {'✅ Couvert (cellule ' + f'{lat_c:.3f}°N, {lon_c:.3f}°E)' if is_cov else '❌ Hors maillage opérationnel'}
                        </div>
                    </div>
                    """
                )
            )

            if st.button("🔍 Évaluer à nouveau", type="primary", use_container_width=True):
                render_prediction_dialog(a_lat, a_lon, bundle, panel_df)
        else:
            st.html(
                textwrap.dedent(
                    """
                    <div class="fwx-card" style="padding: 18px; text-align: center; color: #64748B;">
                        <div style="font-size: 1.8rem; margin-bottom: 6px;">👆</div>
                        <div style="font-weight: 700; color: #334155; margin-bottom: 4px;">Aucun point sélectionné</div>
                        <div style="font-size: 0.8rem;">Cliquez sur la carte pour évaluer une position géographique.</div>
                    </div>
                    """
                )
            )

        # Session history summary
        pts = st.session_state["predicted_points"]
        st.html(
            textwrap.dedent(
                f"""
                <div style="margin-top: 18px; margin-bottom: 8px;">
                    <div style="font-size: 0.82rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">
                        Évaluations de la session ({len(pts)})
                    </div>
                </div>
                """
            )
        )

        if not pts:
            st.html(
                textwrap.dedent(
                    """
                    <div style="font-size: 0.78rem; color: #94A3B8; font-style: italic;">
                        Aucune prédiction effectuée dans cette session.
                    </div>
                    """
                )
            )
        else:
            for i, p in enumerate(reversed(pts[-5:])):
                col_hex = p.get("risk_color", "#EA580C")
                p_wilaya = p.get("wilaya", "")
                p_mode = p.get("mode", "MANUAL_SIMULATION")
                mode_badge = "Observation" if "AUTOMATIC" in p_mode else "Simulation"
                p_ts = p.get("timestamp", p.get("date", ""))
                st.html(
                    textwrap.dedent(
                        f"""
                        <div style="
                            background: #FFFFFF;
                            border-left: 4px solid {col_hex};
                            border-radius: 6px;
                            padding: 8px 12px;
                            margin-bottom: 8px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 800; color: {col_hex}; font-size: 0.82rem;">
                                    {p.get('risk_display', '').upper()}
                                </span>
                                <span style="font-size: 0.85rem; font-weight: 700; color: #0F172A; font-family: 'JetBrains Mono', monospace;">
                                    {p.get('probability_formatted', '—')}
                                </span>
                            </div>
                            <div style="font-size: 0.74rem; color: #334155; margin-top: 3px; font-weight: 600;">
                                {p_wilaya + ' &bull; ' if p_wilaya else ''}{p.get('lat', 0):.3f}°N, {p.get('lon', 0):.3f}°E
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.70rem; color: #64748B; margin-top: 2px;">
                                <span>{mode_badge} &bull; {p.get('horizon', 'H = 3j')}</span>
                                <span>{p_ts}</span>
                            </div>
                        </div>
                        """
                    )
                )

    # -------------------------------------------------------------------------
    # Automatic Dialog Trigger on New Click
    # -------------------------------------------------------------------------
    if new_click_detected and active_pt is not None:
        render_prediction_dialog(active_pt[0], active_pt[1], bundle, panel_df)
