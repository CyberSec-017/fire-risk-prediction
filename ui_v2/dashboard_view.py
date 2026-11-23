"""Level 2 — Interactive Geographical Dashboard for FireWatch AI.

Features:
  - Dominant, high-resolution 2D satellite map of Northern Algeria (Esri World Imagery).
  - Search by preset locations (Wilayas, national parks, forest massifs) or direct coordinates.
  - Visualization of the 209 active NASA POWER grid cells.
  - Interactive click handling opening the Location Details & Prediction Modal.
  - Clean top bar with Return to Landing Page navigation.
"""

from __future__ import annotations

import textwrap
from typing import Any
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import app_core
from ui_v2.components.point_context import (
    ALGERIA_WILAYAS,
    get_elevation_estimate,
    get_wilaya_and_region,
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


def extract_folium_click(map_data: dict[str, Any] | None) -> tuple[float, float] | None:
    """Extract and validate coordinates from last_clicked or last_object_clicked.
    
    Handles:
      - Direct map background clicks -> map_data['last_clicked']
      - Layer / Vector clicks (209 cell rectangles, envelope, markers) -> map_data['last_object_clicked']
    """
    if not map_data or not isinstance(map_data, dict):
        return None

    def _parse(val: Any) -> tuple[float, float] | None:
        if not isinstance(val, dict):
            return None
        lat = val.get("lat")
        lng = val.get("lng")
        if lat is None or lng is None:
            return None
        try:
            return (round(float(lat), 4), round(float(lng), 4))
        except (ValueError, TypeError):
            return None

    map_pt = _parse(map_data.get("last_clicked"))
    obj_pt = _parse(map_data.get("last_object_clicked"))
    obj_count = map_data.get("last_object_clicked_count")

    last_map = st.session_state.get("_last_seen_map_click")
    last_obj = st.session_state.get("_last_seen_obj_click")
    last_obj_count = st.session_state.get("_last_seen_obj_count")

    # Detect if a layer object was clicked
    if obj_pt is not None and (obj_pt != last_obj or (obj_count is not None and obj_count != last_obj_count)):
        st.session_state["_last_seen_obj_click"] = obj_pt
        if obj_count is not None:
            st.session_state["_last_seen_obj_count"] = obj_count
        return obj_pt

    # Detect if map canvas was clicked
    if map_pt is not None and map_pt != last_map:
        st.session_state["_last_seen_map_click"] = map_pt
        return map_pt

    return None


def render_dashboard() -> None:
    """Render the Level 2 Interactive Geographical Dashboard."""
    bundle = get_model_bundle()
    panel_df = get_fire_panel()

    # Initialize session states
    if "predicted_points" not in st.session_state:
        st.session_state["predicted_points"] = []
    if "active_click" not in st.session_state:
        st.session_state["active_click"] = (36.72, 4.05)  # Default: Tizi Ouzou / Djurdjura
    if "_last_seen_map_click" not in st.session_state:
        st.session_state["_last_seen_map_click"] = None
    if "_last_seen_obj_click" not in st.session_state:
        st.session_state["_last_seen_obj_click"] = None
    if "_last_seen_obj_count" not in st.session_state:
        st.session_state["_last_seen_obj_count"] = None
    if "_last_synced_click" not in st.session_state:
        st.session_state["_last_synced_click"] = None
    if "last_preset_selection" not in st.session_state:
        st.session_state["last_preset_selection"] = "-- Sélectionner un massif ou une région remarquable --"
    if "active_basemap" not in st.session_state:
        st.session_state["active_basemap"] = "satellite"

    # -------------------------------------------------------------------------
    # Top Clean Bar: Brand, Return Button, Status Badges
    # -------------------------------------------------------------------------
    col_b1, col_b2 = st.columns([2.5, 1.5], vertical_alignment="center")

    with col_b1:
        st.html(
            textwrap.dedent(
                """
                <div style="display: flex; align-items: center; gap: 12px; padding: 4px 0 10px 0;">
                    <div style="font-size: 2.0rem; line-height: 1;">🔥</div>
                    <div>
                        <div style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; color: #0F172A; line-height: 1.1;">
                            FireWatch <span style="color: #EA580C;">AI</span> &bull; <span style="font-size: 1.05rem; font-weight: 600; color: #475569;">Cockpit Satellite</span>
                        </div>
                        <div style="font-size: 0.78rem; font-weight: 500; color: #64748B;">
                            Système d'Aide à la Décision Incendie &bull; Algérie du Nord (32.0°–37.5°N)
                        </div>
                    </div>
                </div>
                """
            )
        )

    with col_b2:
        col_sub1, col_sub2, col_sub3 = st.columns([1.0, 1.2, 0.9], vertical_alignment="center")
        with col_sub1:
            st.html(
                textwrap.dedent(
                    """
                    <span style="display: inline-flex; align-items: center; gap: 6px; background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; font-size: 0.76rem; font-weight: 700; padding: 5px 12px; border-radius: 9999px;">
                        <span style="width: 8px; height: 8px; background-color: #10B981; border-radius: 50%; display: inline-block;"></span>
                        Modèle Calibré &bull; H=3j
                    </span>
                    """
                )
            )
        with col_sub2:
            if st.button("📈 Données & EDA", type="secondary", use_container_width=True, key="btn_open_eda"):
                st.session_state["app_level"] = "eda"
                st.rerun()

        with col_sub3:
            if st.button("← Accueil", type="secondary", use_container_width=True, key="btn_return_landing"):
                st.session_state["app_level"] = "landing"
                st.rerun()

    st.html(
        textwrap.dedent(
            """
            <div style="height: 1px; background: #E2E8F0; margin: 4px 0 14px 0;"></div>
            """
        )
    )

    # -------------------------------------------------------------------------
    # Geographic Navigation & Location Search Bar
    # -------------------------------------------------------------------------
    # Synchronize input coordinates with active_click if active_click changed externally
    if st.session_state.get("_last_synced_click") != st.session_state["active_click"]:
        st.session_state["coord_input_lat"] = float(st.session_state["active_click"][0])
        st.session_state["coord_input_lon"] = float(st.session_state["active_click"][1])
        st.session_state["_last_synced_click"] = st.session_state["active_click"]

    col_search1, col_search2, col_basemap, col_toggle = st.columns([1.8, 1.3, 1.1, 0.9], vertical_alignment="center")

    with col_search1:
        preset_names = ["-- Sélectionner un massif ou une région remarquable --"] + [w[0] for w in ALGERIA_WILAYAS]
        selected_preset = st.selectbox(
            "Recherche par zone géographique / Wilaya :",
            options=preset_names,
            index=0,
            key="preset_wilaya_selector",
            label_visibility="collapsed",
            help="Sélectionnez une zone pour positionner immédiatement la cible.",
        )
        if selected_preset != st.session_state["last_preset_selection"]:
            st.session_state["last_preset_selection"] = selected_preset
            if selected_preset != preset_names[0]:
                for name, p_lat, p_lon, _ in ALGERIA_WILAYAS:
                    if name == selected_preset:
                        target_coords = (round(p_lat, 4), round(p_lon, 4))
                        if target_coords != st.session_state["active_click"]:
                            st.session_state["active_click"] = target_coords
                            st.rerun()
                        break

    with col_search2:
        col_c_lat, col_c_lon = st.columns(2)
        with col_c_lat:
            inp_lat = st.number_input(
                "Latitude (°N)",
                min_value=18.0,
                max_value=39.0,
                step=0.05,
                format="%.4f",
                key="coord_input_lat",
                label_visibility="collapsed",
            )
        with col_c_lon:
            inp_lon = st.number_input(
                "Longitude (°E)",
                min_value=-9.0,
                max_value=12.5,
                step=0.05,
                format="%.4f",
                key="coord_input_lon",
                label_visibility="collapsed",
            )
        typed_coords = (round(float(inp_lat), 4), round(float(inp_lon), 4))
        if typed_coords != st.session_state["active_click"]:
            st.session_state["active_click"] = typed_coords
            st.session_state["_last_synced_click"] = typed_coords
            st.rerun()

    with col_basemap:
        basemap_options = {
            "🛰️ Satellite": "satellite",
            "🗺️ Carte routière": "osm",
        }
        current_bm = st.session_state.get("active_basemap", "satellite")
        current_idx = 0 if current_bm == "satellite" else 1

        selected_bm = st.selectbox(
            "Fond cartographique :",
            options=list(basemap_options.keys()),
            index=current_idx,
            key="basemap_layer_selector",
            label_visibility="collapsed",
            help="Fond cartographique : Satellite haute résolution Esri ou Carte routière OpenStreetMap",
        )
        target_bm = basemap_options[selected_bm]
        if target_bm != st.session_state["active_basemap"]:
            st.session_state["active_basemap"] = target_bm
            st.rerun()

    with col_toggle:
        show_grid = st.checkbox(
            "Grille (209 cellules)",
            value=True,
            help="Affiche le maillage des cellules d'apprentissage NASA POWER (0.5° × 0.625°)",
            key="toggle_grid_cells_dashboard",
        )

    # -------------------------------------------------------------------------
    # Main Dominant Satellite Map Section
    # -------------------------------------------------------------------------
    col_map, col_details = st.columns([3.3, 1.2])

    with col_map:
        m = build_satellite_map(
            predicted_points=st.session_state["predicted_points"],
            selected_point=st.session_state["active_click"],
            show_cells_grid=show_grid,
            active_basemap=st.session_state["active_basemap"],
        )

        map_data = st_folium(
            m,
            returned_objects=["last_clicked", "last_object_clicked", "last_object_clicked_count"],
            width="100%",
            height=660,
            key="satellite_geospatial_canvas",
        )

    # Detect click from map canvas or layer objects
    new_click = extract_folium_click(map_data)
    if new_click is not None and new_click != st.session_state["active_click"]:
        st.session_state["active_click"] = new_click
        st.rerun()

    # -------------------------------------------------------------------------
    # Side Details Panel (Location Info & Trigger Action)
    # -------------------------------------------------------------------------
    with col_details:
        active_pt = st.session_state["active_click"]
        a_lat, a_lon = active_pt
        is_cov, lat_c, lon_c = is_point_in_active_cells(a_lat, a_lon)
        wilaya = get_wilaya_and_region(a_lat, a_lon)
        altitude = get_elevation_estimate(a_lat, a_lon)

        st.html(
            textwrap.dedent(
                f"""
                <div class="fwx-card" style="padding: 16px 18px; margin-bottom: 14px;">
                    <div style="font-size: 0.76rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em;">
                        Position Ciblée
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 4px 0;">
                        {wilaya}
                    </div>
                    <div style="font-size: 0.82rem; color: #334155; margin-bottom: 8px;">
                        📍 <b>{a_lat:.4f}°N, {a_lon:.4f}°E</b> &bull; ~{altitude} m
                    </div>
                    
                    <div style="border-top: 1px solid #E2E8F0; padding-top: 8px; font-size: 0.78rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: #64748B;">Cellule modèle :</span>
                            <b>{lat_c:.3f}°N, {lon_c:.3f}°E</b>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #64748B;">Statut couverture :</span>
                            <span style="color: {'#059669' if is_cov else '#DC2626'}; font-weight: 700;">
                                {'✅ Couvert' if is_cov else '❌ Hors maillage'}
                            </span>
                        </div>
                    </div>
                </div>
                """
            )
        )

        # Action: Prediction Launch if covered, otherwise clear notice and disabled action
        if is_cov:
            if st.button("🔥 Prédire le Risque d'Incendie", type="primary", use_container_width=True, key="btn_open_predict"):
                render_prediction_dialog(a_lat, a_lon, bundle, panel_df)
        else:
            st.warning(
                "⚠️ **Point hors maillage d'apprentissage (209 cellules)**\n\n"
                "Cette position est située en dehors des 209 cellules géographiques terrestres "
                "d'entraînement du modèle (Nord algérien : 32.0°–37.5°N, -1.875°–8.750°E). "
                "La modélisation est désactivée par mesure de rigueur scientifique."
            )
            st.button(
                "🔒 Prédiction désactivée (Hors couverture)",
                type="primary",
                disabled=True,
                use_container_width=True,
                key="btn_predict_disabled",
                help="Sélectionnez une position dans le domaine opérationnel pour activer la prédiction.",
            )

        # Session history summary
        pts = st.session_state["predicted_points"]
        st.html(
            textwrap.dedent(
                f"""
                <div style="margin-top: 20px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.04em;">
                            Évaluations de la session ({len(pts)})
                        </span>
                        {f'<span style="font-size:0.7rem; color:#64748B;">Dernières</span>' if pts else ''}
                    </div>
                </div>
                """
            )
        )

        if not pts:
            st.html(
                textwrap.dedent(
                    """
                    <div style="font-size: 0.78rem; color: #94A3B8; font-style: italic; background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 6px; padding: 12px; text-align: center;">
                        Cliquez sur la carte pour évaluer un point d'intérêt.
                    </div>
                    """
                )
            )
        else:
            for p in reversed(pts[-5:]):
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
                                <span style="font-weight: 800; color: {col_hex}; font-size: 0.8rem;">
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

            if st.button("Effacer l'historique", type="secondary", use_container_width=True, key="btn_clear_history"):
                st.session_state["predicted_points"] = []
                st.rerun()
