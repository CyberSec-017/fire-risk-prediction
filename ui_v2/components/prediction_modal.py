"""Interactive Location Details & Fire Risk Prediction Modal.

Implements:
  - Section A: Automatically retrieved geographic context (Lat/Lon, Altitude, Wilaya/Region, Cell, Coverage).
  - Section B: Meteorological input form with validated sensible ranges.
  - Section C: Dual Weather Modes (Automatic NASA POWER vs Manual Hypothetical Simulation).
  - Section D: "Predict Fire Risk" action executing calibrated 20-feature production model (H=3 days).
"""

from __future__ import annotations

import datetime
import textwrap
from typing import Any
import uuid

import pandas as pd
import streamlit as st

import app_core
from ui_v2.components.point_context import (
    get_cell_statistics,
    get_elevation_estimate,
    get_wilaya_and_region,
    is_point_in_active_cells,
    evaluate_point_prediction,
    evaluate_live_point_prediction,
)


def record_prediction_in_history(res: dict[str, Any]) -> bool:
    """Record a completed prediction in session state with strict deduplication.

    Returns True if newly appended, False if already present in history.
    """
    if "predicted_points" not in st.session_state:
        st.session_state["predicted_points"] = []

    pred_id = res.get("prediction_id")
    if not pred_id:
        pred_id = uuid.uuid4().hex[:12]
        res["prediction_id"] = pred_id

    # Verify that this prediction is not already recorded
    existing_ids = {
        p.get("prediction_id")
        for p in st.session_state["predicted_points"]
        if isinstance(p, dict) and p.get("prediction_id")
    }
    if pred_id in existing_ids:
        return False

    st.session_state["predicted_points"].append(res)
    return True


@st.dialog("Détails Géographiques & Évaluation du Risque", width="large", on_dismiss="rerun")
def render_prediction_dialog(
    lat: float,
    lon: float,
    bundle: app_core.ModelBundle,
    panel_df: pd.DataFrame | None = None,
) -> None:
    """Render the detailed floating inspection dialog for the clicked location."""
    is_covered, lat_cell, lon_cell = is_point_in_active_cells(lat, lon)
    wilaya = get_wilaya_and_region(lat, lon)
    altitude = get_elevation_estimate(lat, lon)

    # -------------------------------------------------------------------------
    # Case 1: Point is outside the 209 active cells coverage
    # -------------------------------------------------------------------------
    if not is_covered:
        st.error("⚠️ **Position en dehors du domaine d'apprentissage (209 cellules actives)**")
        st.html(
            textwrap.dedent(
                f"""
                <div style="font-size: 0.9rem; color: #475569; line-height: 1.6; margin-bottom: 14px;">
                    La position sélectionnée (<b>{lat:.4f}°N, {lon:.4f}°E</b> &bull; {wilaya})
                    ne fait pas partie des <b>209 cellules géographiques terrestres NASA POWER</b> sur lesquelles
                    le modèle de production a été rigoureusement entraîné et étalonné (Nord de l'Algérie : 32.0°–37.5°N, -1.875°–8.750°E).
                </div>
                """
            )
        )
        st.info(
            "🔒 **Prédiction désactivée :** Conformément au protocole de rigueur scientifique, l'évaluation "
            "est désactivée hors du maillage d'apprentissage afin d'éviter toute extrapolation non validée."
        )
        if st.button("Fermer", type="secondary", use_container_width=True):
            st.rerun()
        return

    # -------------------------------------------------------------------------
    # Section A: Automatically Retrieved Geographic Context
    # -------------------------------------------------------------------------
    stats = get_cell_statistics(lat_cell, lon_cell, panel_df=panel_df)

    st.html(
        textwrap.dedent(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 18px; margin-bottom: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px; margin-bottom: 10px;">
                    <div>
                        <span style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">Région / Wilaya</span>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #0F172A;">{wilaya}</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700;">
                            ✅ Domaine Couvert (Modèle Opérationnel)
                        </span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 0.82rem;">
                    <div>
                        <span style="color: #64748B;">Latitude / Longitude :</span><br>
                        <b>{lat:.4f}°N, {lon:.4f}°E</b>
                    </div>
                    <div>
                        <span style="color: #64748B;">Cellule NASA POWER :</span><br>
                        <b>{lat_cell:.3f}°N, {lon_cell:.3f}°E</b>
                    </div>
                    <div>
                        <span style="color: #64748B;">Altitude approximative :</span><br>
                        <b>~{altitude} m (NMM)</b>
                    </div>
                    <div>
                        <span style="color: #64748B;">Fréquence historique (2018–22) :</span><br>
                        <b>{stats['fire_days']} j ({stats['fire_rate']*100:.1f}%)</b>
                    </div>
                </div>
            </div>
            """
        )
    )

    # -------------------------------------------------------------------------
    # Section C: Weather Data Mode Selection (Dual Mode)
    # -------------------------------------------------------------------------
    st.markdown("##### 🌦️ Source des Données Météorologiques")

    weather_mode = st.radio(
        label="Mode météorologique",
        options=[
            "📡 Mode Automatique : Données Réelles Observées (NASA POWER)",
            "⚙️ Mode Simulation : Scénario Météorologique Personnalisé",
        ],
        index=1,
        label_visibility="collapsed",
    )

    # -------------------------------------------------------------------------
    # Mode 1: Automatic NASA POWER Live Observation Mode
    # -------------------------------------------------------------------------
    if "Automatique" in weather_mode:
        st.html(
            textwrap.dedent(
                """
                <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 12px 16px; margin: 10px 0; font-size: 0.82rem; color: #1E3A8A;">
                    <b>Mode Automatique NASA POWER :</b> Interroge l'API spatiale NASA POWER pour extraire l'historique
                    météorologique complet à ces coordonnées, puis calcule fidèlement les cumuls et mémoires réelles (7j, 14j, 30j).
                </div>
                """
            )
        )

        col_d1, col_d2 = st.columns([1.5, 2.0])
        with col_d1:
            ref_date = st.date_input(
                "Date de référence observée",
                value=datetime.date(2022, 8, 15),
                min_value=datetime.date(2018, 1, 1),
                max_value=datetime.date.today(),
                help="Date pour laquelle interroger l'historique satellitaire",
            )
        with col_d2:
            st.write("")
            st.write("")
            fetch_btn = st.button("🛰️ Interroger l'API NASA POWER en direct", type="secondary", use_container_width=True)

        fetched_history = st.session_state.get(f"nasa_history_{lat_cell}_{lon_cell}")

        if fetch_btn:
            with st.spinner("Interrogation des serveurs NASA POWER en cours..."):
                try:
                    df_power = app_core.fetch_power_history(lat, lon, end_date=ref_date)
                    st.session_state[f"nasa_history_{lat_cell}_{lon_cell}"] = df_power
                    fetched_history = df_power
                    st.success(f"✅ {len(df_power)} jours d'observations NASA POWER récupérés avec succès.")
                except Exception as exc:
                    st.warning(f"⚠️ Serveur NASA POWER temporairement inaccessible ({exc}). Utilisation des médianes climatiques.")

        # Action: Predict Fire Risk in Automatic Mode
        st.html("<div style='height: 10px;'></div>")
        if st.button("🔥 Prédire le Risque d'Incendie (Mode Automatique)", type="primary", use_container_width=True):
            if fetched_history is not None and not fetched_history.empty:
                try:
                    res = evaluate_live_point_prediction(bundle, lat, lon, ref_date, fetched_history)
                    record_prediction_in_history(res)
                    _render_prediction_result_card(res)
                except Exception as err:
                    st.error(f"Erreur de calcul du modèle : {err}")
            else:
                # Fallback to localized medians
                try:
                    inputs = {
                        "T2M": stats["t2m_median"],
                        "T2M_MAX": stats["t2m_max_median"],
                        "RH2M": stats["rh2m_median"],
                        "WS2M": stats["ws2m_median"],
                        "PRECTOTCORR": stats["precip_median"],
                        "GWETROOT": stats["gwetroot_median"],
                        "dry_days": stats["dry_days_median"],
                    }
                    res = evaluate_point_prediction(bundle, lat, lon, ref_date, inputs)
                    res["mode"] = "AUTOMATIC_MEDIANS_FALLBACK"
                    res["persistence_note"] = "Médianes climatiques historiques locales de la cellule (2018–2022)."
                    record_prediction_in_history(res)
                    _render_prediction_result_card(res)
                except Exception as err:
                    st.error(f"Erreur d'inférence du modèle : {err}")

    # -------------------------------------------------------------------------
    # Mode 2: Manual Simulation Mode
    # -------------------------------------------------------------------------
    else:
        st.html(
            textwrap.dedent(
                """
                <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 12px 16px; margin: 10px 0; font-size: 0.82rem; color: #92400E;">
                    <b>Mode Simulation Hypothétique :</b> Ajustez manuellement les conditions météorologiques du jour
                    pour tester un scénario prédictif. Les valeurs sont pré-remplies avec les médianes historiques locales de cette cellule.
                </div>
                """
            )
        )

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            sim_date = st.date_input(
                "Date de simulation (cycle saisonnier)",
                value=datetime.date(2024, 7, 25),
                help="Définit les harmoniques saisonnières doy_sin et doy_cos",
            )
            t2m = st.slider(
                "Température moyenne (°C)",
                min_value=10.0,
                max_value=48.0,
                value=float(max(10.0, min(stats["t2m_median"], 48.0))),
                step=0.5,
                help="T2M : Température moyenne de l'air à 2 mètres",
            )
            t2m_max = st.slider(
                "Température maximale (°C)",
                min_value=float(t2m),
                max_value=52.0,
                value=float(max(t2m, min(stats["t2m_max_median"], 52.0))),
                step=0.5,
                help="T2M_MAX : Pic de température de l'après-midi",
            )
            rh2m = st.slider(
                "Humidité relative (%)",
                min_value=5.0,
                max_value=95.0,
                value=float(stats["rh2m_median"]),
                step=1.0,
                help="RH2M : Humidité de l'air (critique si < 25%)",
            )

        with col_f2:
            ws2m = st.slider(
                "Vitesse du vent à 2 m (m/s)",
                min_value=0.5,
                max_value=20.0,
                value=float(stats["ws2m_median"]),
                step=0.5,
                help="WS2M : Vitesse du vent (1 m/s = 3.6 km/h ; 10 m/s = 36 km/h)",
            )
            precip = st.number_input(
                "Précipitations du jour (mm)",
                min_value=0.0,
                max_value=50.0,
                value=float(stats["precip_median"]),
                step=0.5,
                help="PRECTOTCORR : Pluie journalière (mm)",
            )
            gwetroot = st.slider(
                "Humidité racinaire du sol (0 – 1)",
                min_value=0.02,
                max_value=0.90,
                value=float(stats["gwetroot_median"]),
                step=0.02,
                help="GWETROOT : Humidité volumique de la zone racinaire (critique si < 0.20)",
            )
            dry_days = st.slider(
                "Nombre de jours consécutifs sans pluie",
                min_value=0,
                max_value=60,
                value=int(stats["dry_days_median"]),
                step=1,
                help="Jours écoulés depuis la dernière pluie > 1 mm",
            )

        # Persistence note disclosure
        st.html(
            textwrap.dedent(
                """
                <div style="font-size: 0.76rem; color: #64748B; margin-top: 6px; line-height: 1.4;">
                    ℹ️ <i>Hypothèse de persistance (build_manual_features) : Les conditions saisies pour ce jour
                    sont projetées sur les fenêtres glissantes antérieures (pluie 7j/14j/30j et tmax glissant).</i>
                </div>
                """
            )
        )

        # ---------------------------------------------------------------------
        # Section D: Prediction Action (Manual Simulation)
        # ---------------------------------------------------------------------
        st.html("<div style='height: 12px;'></div>")
        if st.button("🔥 Prédire le Risque d'Incendie (Scénario Manuel)", type="primary", use_container_width=True):
            inputs = {
                "T2M": t2m,
                "T2M_MAX": t2m_max,
                "RH2M": rh2m,
                "WS2M": ws2m,
                "PRECTOTCORR": precip,
                "GWETROOT": gwetroot,
                "dry_days": float(dry_days),
            }
            try:
                res = evaluate_point_prediction(bundle, lat, lon, sim_date, inputs)
                record_prediction_in_history(res)
                _render_prediction_result_card(res)
            except Exception as err:
                st.error(f"Erreur d'inférence du modèle : {err}")


def _render_prediction_result_card(res: dict[str, Any]) -> None:
    """Render the official prediction card and ensure persistence into session state."""
    prob = res["probability"]
    prob_pct = prob * 100.0
    risk_color = res["risk_color"]
    risk_disp = res["risk_display"]
    wilaya = res.get("wilaya", "")
    mode = res.get("mode", "MANUAL_SIMULATION")
    horizon = res.get("horizon", "H = 3 jours")
    ts_str = res.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pred_id = res.get("prediction_id", "")
    weather = res.get("weather_inputs", {})

    # Ensure safe recording into session state (with deduplication)
    record_prediction_in_history(res)

    mode_label = "🛰️ Observation NASA POWER" if "AUTOMATIC" in mode else "⚙️ Simulation Personnalisée"

    # Format weather parameters summary string
    weather_parts = []
    if "T2M" in weather:
        weather_parts.append(f"T2M: {weather['T2M']:.1f}°C")
    if "T2M_MAX" in weather:
        weather_parts.append(f"Tmax: {weather['T2M_MAX']:.1f}°C")
    if "RH2M" in weather:
        weather_parts.append(f"RH: {weather['RH2M']:.0f}%")
    if "WS2M" in weather:
        weather_parts.append(f"Vent: {weather['WS2M']:.1f} m/s")
    if "PRECTOTCORR" in weather:
        weather_parts.append(f"Pluie: {weather['PRECTOTCORR']:.1f} mm")
    if "GWETROOT" in weather:
        weather_parts.append(f"Hum. sol: {weather['GWETROOT']:.2f}")
    if "dry_days" in weather:
        weather_parts.append(f"Jours secs: {int(weather['dry_days'])} j")
    weather_summary = " &bull; ".join(weather_parts) if weather_parts else "Paramètres nominaux"

    st.html(
        textwrap.dedent(
            f"""
            <div style="
                background: #FFFFFF;
                border: 2px solid {risk_color};
                border-radius: 12px;
                padding: 18px 22px;
                margin-top: 18px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">
                        Résultat Officiel &bull; Échéance {horizon}
                    </span>
                    <span style="
                        background-color: {risk_color}22;
                        color: {risk_color};
                        border: 1px solid {risk_color};
                        padding: 3px 14px;
                        border-radius: 9999px;
                        font-weight: 800;
                        font-size: 0.82rem;
                    ">
                        NIVEAU DE RISQUE : {risk_disp.upper()}
                    </span>
                </div>
                <div style="display: flex; align-items: baseline; gap: 14px;">
                    <div style="font-size: 2.3rem; font-weight: 800; color: {risk_color}; font-family: 'JetBrains Mono', monospace;">
                        {prob_pct:.2f} %
                    </div>
                    <div style="font-size: 0.85rem; color: #475569;">
                        Probabilité calibrée d'incendie (Source : {res.get('score_source', 'REAL')})
                    </div>
                </div>
                <div style="width: 100%; background: #E2E8F0; border-radius: 9999px; height: 10px; margin: 12px 0; overflow: hidden;">
                    <div style="background: {risk_color}; width: {min(prob_pct * 3.0, 100):.1f}%; height: 100%;"></div>
                </div>
                
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px; margin: 12px 0; font-size: 0.78rem; color: #334155;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span><b>Localisation :</b> {wilaya} ({res.get('lat', 0.0):.4f}°N, {res.get('lon', 0.0):.4f}°E)</span>
                        <span style="color: #64748B;">Cellule: {res.get('lat_cell', 0.0):.3f}°N, {res.get('lon_cell', 0.0):.3f}°E</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span><b>Mode :</b> {mode_label}</span>
                        <span style="color: #64748B;">Horodatage : {ts_str}</span>
                    </div>
                    <div style="border-top: 1px dashed #CBD5E1; padding-top: 4px; margin-top: 4px; color: #475569;">
                        <b>Météo :</b> {weather_summary}
                    </div>
                </div>

                <div style="font-size: 0.78rem; color: #475569; line-height: 1.5; margin-top: 8px;">
                    <b>Horizon prédictif :</b> Cette probabilité quantifie le risque d'occurrence d'au moins un foyer
                    d'incendie actif dans la cellule ({res.get('lat_cell', 0.0):.3f}°N, {res.get('lon_cell', 0.0):.3f}°E) au cours des <b>3 prochains jours</b>.<br>
                    <i>{res.get('persistence_note', '')}</i>
                </div>
            </div>
            """
        )
    )

    # Dedicated immediate close & refresh action button
    st.html("<div style='height: 12px;'></div>")
    if st.button(
        "✅ Fermer et actualiser l'historique",
        type="primary",
        use_container_width=True,
        key=f"btn_close_and_refresh_{pred_id or 'latest'}",
        help="Ferme cette boîte de dialogue et actualise immédiatement l'historique et la carte.",
    ):
        st.rerun()
