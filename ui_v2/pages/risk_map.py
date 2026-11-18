"""Risk Map Page — FireWatch AI UI V2.

Presents:
  - Spatial mapping of historical fire frequency across the 209 NASA POWER cells.
  - Identification of regional hotspot clusters (Kabylie, Jijel, El Tarf, Tellian Atlas).
  - Ranking table of top high-risk cells based on historical burned days.
"""

from __future__ import annotations

import textwrap
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import app_core
from ui_v2.components.point_context import (
    get_active_209_cells,
    get_cell_bounding_box,
)
from ui_v2.components.satellite_map import (
    ESRI_IMAGERY_ATTR,
    ESRI_IMAGERY_URL,
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
)


@st.cache_data(show_spinner="Calcul de la fréquence spatiale des feux...")
def get_spatial_risk_data() -> pd.DataFrame:
    """Compute pre-test historical fire occurrence per active cell."""
    active_cells = get_active_209_cells()
    records = []

    if app_core.FIRE_PANEL_PATH.exists():
        df = pd.read_parquet(
            app_core.FIRE_PANEL_PATH,
            columns=["date", "burned", "lat_cell", "lon_cell"],
        )
        # Strictly pre-test (2018–2022)
        df = df[df["date"] < "2023-01-01"]
        cell_agg = df.groupby(["lat_cell", "lon_cell"])["burned"].agg(["sum", "count"]).reset_index()
        cell_agg.rename(columns={"sum": "fire_days", "count": "total_days"}, inplace=True)
        cell_agg["fire_rate"] = cell_agg["fire_days"] / cell_agg["total_days"]
        cell_map = {
            (round(r["lat_cell"], 4), round(r["lon_cell"], 4)): (r["fire_days"], r["total_days"], r["fire_rate"])
            for _, r in cell_agg.iterrows()
        }
    else:
        cell_map = {}

    for lat_c, lon_c in sorted(active_cells):
        val = cell_map.get((lat_c, lon_c), (0, 1826, 0.0))
        records.append({
            "lat_cell": lat_c,
            "lon_cell": lon_c,
            "fire_days": int(val[0]),
            "total_days": int(val[1]),
            "fire_rate": float(val[2]),
            "fire_rate_pct": float(val[2] * 100.0),
        })

    return pd.DataFrame(records)


def _get_risk_fill_color(rate: float) -> str:
    """Color scale for historical fire frequency."""
    if rate > 0.03:
        return "#DC2626"  # Rouge vif (très élevé)
    if rate > 0.015:
        return "#EA580C"  # Orange vif (élevé)
    if rate > 0.005:
        return "#F59E0B"  # Ambre (modéré)
    return "#10B981"      # Vert (faible)


def render_page() -> None:
    """Render the Spatial Risk Map page."""
    st.html(
        textwrap.dedent(
            """
            <div style="margin-bottom: 18px;">
                <div style="font-size: 1.5rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">
                    🗺️ Cartographie Spatiale de la Fréquence des Incendies
                </div>
                <div style="font-size: 0.9rem; color: #475569; margin-top: 4px;">
                    Distribution géographique des feux observés (2018–2022) sur les 209 cellules NASA POWER du Nord algérien.
                </div>
            </div>
            """
        )
    )

    risk_df = get_spatial_risk_data()

    col_map, col_table = st.columns([2.5, 1.3])

    with col_map:
        m = folium.Map(
            location=[MAP_CENTER_LAT, MAP_CENTER_LON],
            zoom_start=7,
            tiles=None,
            control_scale=True,
        )

        folium.TileLayer(
            tiles=ESRI_IMAGERY_URL,
            attr=ESRI_IMAGERY_ATTR,
            name="🛰️ Satellite (Esri)",
            overlay=False,
        ).add_to(m)

        folium.TileLayer(
            tiles="OpenStreetMap",
            name="🗺️ OpenStreetMap",
            overlay=False,
        ).add_to(m)

        cells_group = folium.FeatureGroup(name="🔥 Taux historique de feu", show=True)
        for _, r in risk_df.iterrows():
            bbox = get_cell_bounding_box(r["lat_cell"], r["lon_cell"])
            fill_col = _get_risk_fill_color(r["fire_rate"])
            opacity = 0.25 if r["fire_rate"] > 0.005 else 0.08
            folium.Rectangle(
                bounds=bbox,
                color=fill_col,
                weight=1.5,
                fill=True,
                fill_color=fill_col,
                fill_opacity=opacity,
                tooltip=f"Cellule ({r['lat_cell']:.3f}°N, {r['lon_cell']:.3f}°E) : {r['fire_days']} jours d'incendie ({r['fire_rate_pct']:.2f}%)",
                popup=(
                    f"<b>Cellule ({r['lat_cell']:.3f}°N, {r['lon_cell']:.3f}°E)</b><br>"
                    f"Jours d'incendie (2018–2022) : <b>{int(r['fire_days'])}</b><br>"
                    f"Fréquence : <b>{r['fire_rate_pct']:.2f} %</b>"
                ),
            ).add_to(cells_group)

        cells_group.add_to(m)
        folium.LayerControl(position="topright", collapsed=True).add_to(m)

        st_folium(m, width="100%", height=560, key="risk_spatial_map")

    with col_table:
        st.markdown("##### 🏆 Top 10 Cellules les Plus Vulnérables")
        top_cells = risk_df.sort_values(by="fire_days", ascending=False).head(10).copy()
        top_cells.reset_index(drop=True, inplace=True)
        top_cells["Rang"] = range(1, len(top_cells) + 1)
        top_cells["Cellule"] = top_cells.apply(
            lambda row: f"{row['lat_cell']:.3f}°N, {row['lon_cell']:.3f}°E", axis=1
        )
        display_df = top_cells[["Rang", "Cellule", "fire_days", "fire_rate_pct"]]
        display_df.rename(
            columns={
                "fire_days": "Jours de feu",
                "fire_rate_pct": "Taux (%)",
            },
            inplace=True,
        )
        st.dataframe(display_df, hide_index=True, use_container_width=True)

        st.html(
            textwrap.dedent(
                """
                <div class="fwx-card" style="padding: 12px 14px; font-size: 0.8rem; margin-top: 12px; color: #475569;">
                    <b>Légende de vulnérabilité :</b><br>
                    <span style="color: #DC2626; font-weight: 700;">■ Rouge</span> : Fréquence > 3.0 % (Hotspots majeurs : Tizi Ouzou, Bejaia, El Tarf, Jijel)<br>
                    <span style="color: #EA580C; font-weight: 700;">■ Orange</span> : Fréquence 1.5 – 3.0 % (Zones forestières collinaires)<br>
                    <span style="color: #F59E0B; font-weight: 700;">■ Ambre</span> : Fréquence 0.5 – 1.5 % (Zones de piémont et transition)<br>
                    <span style="color: #10B981; font-weight: 700;">■ Vert</span> : Fréquence < 0.5 % (Zones semi-arides ou plaines)
                </div>
                """
            )
        )
