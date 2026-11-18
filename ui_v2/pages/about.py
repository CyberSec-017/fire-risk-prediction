"""Scientific Methodology & Architecture Page — FireWatch AI UI V2.

Presents:
  - Scientific problem formulation (Horizon H=3 days, spatial grid 0.5° × 0.625°).
  - NASA data pipeline integration (NASA POWER meteorology + NASA FIRMS thermal anomalies).
  - Machine learning methodology: 6 classifiers benchmark, temporal splitting with purge, calibration.
  - Contract specification of the 20 production features and risk bins.
"""

from __future__ import annotations

import textwrap
import streamlit as st

import app_core


def render_page() -> None:
    """Render the Methodology & Models page."""
    st.html(
        textwrap.dedent(
            """
            <div style="margin-bottom: 20px;">
                <div style="font-size: 1.5rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">
                    ℹ️ Méthodologie Scientifique, Données & Modèle
                </div>
                <div style="font-size: 0.9rem; color: #475569; margin-top: 4px;">
                    Documentation détaillée du protocole de modélisation prédictive des feux de forêts (Algérie du Nord).
                </div>
            </div>
            """
        )
    )

    t1, t2, t3, t4 = st.tabs([
        "🔬 Protocole Scientifique",
        "📡 Sources de Données",
        "🤖 Modèle & Benchmark",
        "📋 Contrat des 20 Features",
    ])

    with t1:
        st.markdown(
            textwrap.dedent(
                """
                #### 1. Formulation du Problème Prédictif
                Le système FireWatch AI modélise la probabilité qu'au moins un incendie actif se déclare ou se propage
                dans une cellule géographique donnée à un **horizon d'anticipation de $H = 3$ jours**.

                ##### Principes Fondamentaux :
                - **Résolution spatiale :** Découpage du Nord algérien (32.0°–37.5°N, -1.875°–8.750°E) en **209 cellules terrestres** (0.5° lat × 0.625° lon, soit environ $55 \\times 55\\text{ km}$).
                - **Pas temporel :** Échelle journalière ($t$).
                - **Purge temporelle anti-fuite (Embargo) :** Une fenêtre tampon de 3 jours sépare strictement les données météorologiques d'entrée de l'événement cible ($t + 3$), éliminant toute fuite d'information (*data leakage*).
                - **Validation temporelle stricte :** Folds chronologiques sans mélange aléatoire (Train : 2018–2020, Calibrage : 2021, Test : 2022–2024).
                """
            )
        )

    with t2:
        st.markdown(
            textwrap.dedent(
                """
                #### 2. Intégration des Données Spatiales & Météorologiques NASA

                - **NASA POWER (Prediction Of Worldwide Energy Resources) :**
                  Fournit l'historique quotidien et les flux en quasi-temps réel pour la température (T2M, T2M_MAX),
                  l'humidité relative (RH2M), la vitesse du vent à 2 m (WS2M), les précipitations (PRECTOTCORR),
                  et l'humidité du profil racinaire du sol (GWETROOT).
                - **NASA FIRMS (Fire Information for Resource Management System) :**
                  Observations d'anomalies thermiques haute résolution issues des capteurs orbitaux **MODIS** (Terra/Aqua, 1 km)
                  et **VIIRS** (Suomi-NPP / NOAA-20, 375 m).

                ---

                #### 🗺️ Sources Cartographiques & Attributions Légales

                Conformément aux licences et conditions d'utilisation des services cartographiques intégrés dans la plateforme FireWatch AI :

                - **Imagerie Satellite Haute Résolution (Fond Principal) :**  
                  *Fournisseur :* **Esri (Environmental Systems Research Institute, Inc.)**  
                  *Service :* Esri World Imagery (MapServer tile service)  
                  *Attribution légale :* *Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community.*

                - **Référentiel Administratif & Frontières (Superposition) :**  
                  *Fournisseur :* **Esri World Boundaries and Places**  
                  *Attribution légale :* *Esri Boundaries & Places.*

                - **Cartographie Routière Alternative :**  
                  *Fournisseur :* **OpenStreetMap (OSM)**  
                  *Attribution légale :* *&copy; Les contributeurs d'OpenStreetMap (licence ODbL).*

                - **Moteur de Rendu Cartographique :**  
                  *Bibliothèques :* **Leaflet JS** & **Folium**.
                """
            )
        )

    with t3:
        st.markdown(
            textwrap.dedent(
                """
                #### 3. Benchmark des Modèles & Calibrage Isotonique

                Un benchmark rigoureux de 6 architectures de classification a été conduit sur les plis temporels :
                - **HistGradientBoostingClassifier (Sélectionné en Production)** : Meilleur compromis précision-rappel, robustesse aux valeurs manquantes et rapidité d'inférence.
                - **Calibrage Isotonique (`CalibratedClassifierCV`)** : Les probabilités brutes sont étalonnées afin que $P(\\text{feu}) = 0.20$ corresponde empiriquement à 20 feux sur 100 observations.
                - **Métrique d'arbitrage :** PR-AUC (Precision-Recall Area Under Curve), adaptée au déséquilibre sévère des classes (taux de feu moyen $\\approx 1.0\\%$).

                ##### Paliers de Risque Calibrés :
                - 🟢 **Faible (Low) :** $P < 0.05$
                - 🟡 **Modéré (Moderate) :** $0.05 \\le P < 0.15$
                - 🟠 **Élevé (High) :** $0.15 \\le P < 0.35$
                - 🔴 **Très Élevé (Very High) :** $P \\ge 0.35$
                """
            )
        )

    with t4:
        bundle = app_core.load_model_bundle()
        feature_names = bundle.feature_names
        st.markdown(f"#### 4. Contrat Officiel des {len(feature_names)} Features de Production")
        st.markdown(
            textwrap.dedent(
                """
                Le modèle de production exige strictement l'ordre et les noms de features définis dans
                `models/model_card.json`.
                """
            )
        )
        col_f1, col_f2 = st.columns(2)
        half = len(feature_names) // 2
        with col_f1:
            for idx, f in enumerate(feature_names[:half], start=1):
                st.code(f"{idx:02d}. {f}", language="text")
        with col_f2:
            for idx, f in enumerate(feature_names[half:], start=half + 1):
                st.code(f"{idx:02d}. {f}", language="text")
