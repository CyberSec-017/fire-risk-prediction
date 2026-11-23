"""Level 1 — Professional Welcome & Landing Page for FireWatch AI.

Design:
  - Premium environmental intelligence branding.
  - Atmospheric dark hero section with glowing gradient accents.
  - Headline: "Understand. Monitor. Predict."
  - Wildfire forecasting overview for Northern Algeria.
  - Prominent "Explore FireWatch AI" CTA button navigating to the interactive dashboard.
"""

from __future__ import annotations

import textwrap
import streamlit as st


def render_landing_page() -> None:
    """Render the Level 1 Welcome and Landing Experience."""
    # -------------------------------------------------------------------------
    # Hero Section
    # -------------------------------------------------------------------------
    st.html(
        textwrap.dedent(
            """
            <div class="fwx-hero-container">
                <div class="fwx-hero-badge">
                    <span class="fwx-pulse-dot"></span>
                    SYSTÈME D'INTELLIGENCE ENVIRONNEMENTALE &bull; ALGÉRIE DU NORD
                </div>
                <div class="fwx-hero-brand">
                    <span style="font-size: 2.8rem; line-height: 1;">🔥</span>
                    <span class="fwx-hero-title">FireWatch <span style="color: #EA580C;">AI</span></span>
                </div>
                <h1 class="fwx-hero-headline">
                    Understand. Monitor. Predict.
                </h1>
                <p class="fwx-hero-sub">
                    Plateforme opérationnelle d'anticipation des risques d'incendies forestiers.
                    Fusion des réanalyses météorologiques <b>NASA POWER</b> et de la télédétection orbitale
                    <b>NASA FIRMS</b> pour une modélisation prédictive étalonnée à échéance <b>3 jours</b>.
                </p>
            </div>
            """
        )
    )

    # -------------------------------------------------------------------------
    # Centered CTA Action
    # -------------------------------------------------------------------------
    col_l, col_btn, col_r = st.columns([1.2, 1.6, 1.2])
    with col_btn:
        if st.button("🚀 Explorer FireWatch AI →", type="primary", width="stretch", key="btn_explore_firewatch"):
            st.session_state["app_level"] = "dashboard"
            st.rerun()
        if st.button("📈 Explorer les données (EDA & qualité)", type="secondary",
                     width="stretch", key="btn_explore_eda"):
            st.session_state["app_level"] = "eda"
            st.rerun()
        if st.button("🧪 Validation statistique & benchmark", type="secondary",
                     width="stretch", key="btn_explore_statistics"):
            st.session_state["app_level"] = "statistics"
            st.rerun()
        if st.button("📄 Inférence par lots (CSV)", type="secondary",
                     width="stretch", key="btn_explore_batch"):
            st.session_state["app_level"] = "batch"
            st.rerun()

    # -------------------------------------------------------------------------
    # Three Pillars of Environmental Intelligence
    # -------------------------------------------------------------------------
    st.html(
        textwrap.dedent(
            """
            <div style="margin-top: 48px; margin-bottom: 24px;">
                <div style="text-align: center; font-size: 0.85rem; font-weight: 700; color: #64748B; letter-spacing: 0.1em; text-transform: uppercase;">
                    Architecture & Capacités Prédictives
                </div>
            </div>
            """
        )
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.html(
            textwrap.dedent(
                """
                <div class="fwx-feature-card">
                    <div class="fwx-feature-icon">🛰️</div>
                    <div class="fwx-feature-title">Cockpit Satellite 2D Haute Résolution</div>
                    <div class="fwx-feature-text">
                        Cartographie haute précision (Esri World Imagery) sur <b>209 cellules géographiques actives</b>
                        du Nord algérien (32.0°–37.5°N, -1.875°–8.750°E). Évaluation instantanée au clic.
                    </div>
                </div>
                """
            )
        )

    with p2:
        st.html(
            textwrap.dedent(
                """
                <div class="fwx-feature-card">
                    <div class="fwx-feature-icon">⏱️</div>
                    <div class="fwx-feature-title">Anticipation Calibrée à H = 3 Jours</div>
                    <div class="fwx-feature-text">
                        Purge temporelle stricte sans fuite d'information. Modèle <i>HistGradientBoosting</i>
                        étalonné par régression isotonique garantissant des probabilités fiables pour l'aide à la décision.
                    </div>
                </div>
                """
            )
        )

    with p3:
        st.html(
            textwrap.dedent(
                """
                <div class="fwx-feature-card">
                    <div class="fwx-feature-icon">⚡</div>
                    <div class="fwx-feature-title">Double Mode d'Analyse Météorologique</div>
                    <div class="fwx-feature-text">
                        Récupération automatisée des flux satellitaires en direct <b>NASA POWER</b> ou
                        simulation interactive de scénarios de canicule et de vent (Sirocco, sécheresse du sol).
                    </div>
                </div>
                """
            )
        )

    # -------------------------------------------------------------------------
    # Institutional Footer
    # -------------------------------------------------------------------------
    st.html(
        textwrap.dedent(
            """
            <div style="margin-top: 60px; padding: 24px 0; border-top: 1px solid #E2E8F0; text-align: center; font-size: 0.8rem; color: #64748B;">
                <b>FireWatch AI</b> &bull; Modélisation Prédictive des Feux de Forêts en Algérie du Nord &bull;
                Données NASA POWER & NASA FIRMS &bull;
            </div>
            """
        )
    )
