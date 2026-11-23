"""Top navigation bar for FireWatch AI UI V2.

Provides:
  - Brand header with FireWatch AI identity and model status.
  - Horizontal tab-style navigation between all 6 platform pages.
  - Synchronization with st.session_state["current_page"].
"""

from __future__ import annotations

import textwrap
import streamlit as st

PAGE_CONFIGS = [
    {
        "id": "fire_predictions",
        "label": "🛰️ Prédiction sur carte",
        "description": "Exploration satellite 2D et inférence au clic (209 cellules)",
    },
    {
        "id": "overview",
        "label": "📊 Vue d'ensemble",
        "description": "Indicateurs clés et dynamique macro-temporelle",
    },
    {
        "id": "risk_map",
        "label": "🗺️ Carte des risques",
        "description": "Cartographie spatiale des zones à risque élevé",
    },
    {
        "id": "weather",
        "label": "🌦️ Surveillance météo",
        "description": "Diagnostic des facteurs météo et stress hydrique",
    },
    {
        "id": "historical",
        "label": "📜 Analyse historique",
        "description": "Historique des détections feux NASA FIRMS (2018–2022)",
    },
    {
        "id": "eda",
        "label": "📈 Données & EDA",
        "description": "Qualité des données et exploration interactive des observations (15 graphiques)",
    },
    {
        "id": "about",
        "label": "ℹ️ Méthodologie",
        "description": "Protocole scientifique, sources de données et modèle",
    },
]

PAGE_LABELS = {p["id"]: p["label"] for p in PAGE_CONFIGS}
LABEL_TO_ID = {p["label"]: p["id"] for p in PAGE_CONFIGS}


def render_navbar() -> str:
    """Render the top horizontal navigation bar and return the active page id."""
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "fire_predictions"

    current_id = st.session_state["current_page"]
    if current_id not in PAGE_LABELS:
        current_id = "fire_predictions"
        st.session_state["current_page"] = current_id

    # -------------------------------------------------------------------------
    # Top Brand Bar
    # -------------------------------------------------------------------------
    col_brand, col_status = st.columns([2.5, 1.5], vertical_alignment="center")

    with col_brand:
        st.html(
            textwrap.dedent(
                """
                <div style="display: flex; align-items: center; gap: 12px; padding: 4px 0 10px 0;">
                    <div style="font-size: 2.2rem; line-height: 1;">🔥</div>
                    <div>
                        <div style="font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em; color: #0F172A; line-height: 1.1;">
                            FireWatch <span style="color: #EA580C;">AI</span>
                        </div>
                        <div style="font-size: 0.8rem; font-weight: 500; color: #64748B; letter-spacing: 0.02em;">
                            Plateforme Opérationnelle d'Intelligence Incendie &bull; Algérie du Nord
                        </div>
                    </div>
                </div>
                """
            )
        )

    with col_status:
        st.html(
            textwrap.dedent(
                """
                <div style="display: flex; justify-content: flex-end; align-items: center; gap: 10px; padding: 4px 0 10px 0;">
                    <span style="display: inline-flex; align-items: center; gap: 6px; background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; font-size: 0.76rem; font-weight: 600; padding: 5px 12px; border-radius: 9999px;">
                        <span style="width: 8px; height: 8px; background-color: #10B981; border-radius: 50%; display: inline-block;"></span>
                        Modèle Calibré &bull; 20 features
                    </span>
                    <span style="display: inline-flex; align-items: center; background: #EFF6FF; border: 1px solid #BFDBFE; color: #1E40AF; font-size: 0.76rem; font-weight: 600; padding: 5px 12px; border-radius: 9999px;">
                        209 cellules actives
                    </span>
                </div>
                """
            )
        )

    # -------------------------------------------------------------------------
    # Horizontal Navigation Menu (using st.segmented_control)
    # -------------------------------------------------------------------------
    current_label = PAGE_LABELS.get(current_id, PAGE_CONFIGS[0]["label"])
    options = [p["label"] for p in PAGE_CONFIGS]

    selected_label = st.segmented_control(
        label="Navigation Principale",
        options=options,
        default=current_label,
        key="nav_segmented_control",
        label_visibility="collapsed",
    )

    if selected_label and selected_label != current_label:
        new_page_id = LABEL_TO_ID.get(selected_label, "fire_predictions")
        st.session_state["current_page"] = new_page_id
        st.rerun()

    st.html(
        textwrap.dedent(
            """
            <div style="height: 1px; background: linear-gradient(90deg, #E2E8F0 0%, #CBD5E1 50%, #E2E8F0 100%); margin: 10px 0 16px 0;"></div>
            """
        )
    )

    return st.session_state["current_page"]
