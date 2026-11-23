"""Streamlit view for CSV batch inference and honest local explanations.

All validation, scoring and export logic lives in ``src/batch_inference.py``;
this module only renders widgets and existing artifacts.  It never imputes
missing values and never falls back to the DEMO heuristic.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
for candidate in (str(ROOT / "src"), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import batch_inference as batch_logic  # noqa: E402
import app_core  # noqa: E402
from dashboard import predictor  # noqa: E402
from dashboard.model_loader import load_model  # noqa: E402

_IMPORTANCE_FIGURE = ROOT / "report" / "figures" / "07_feature_importance.png"
_SUMMARY_COLUMNS = batch_logic.DERIVED_COLUMNS


def _render_schema(card: dict[str, Any]) -> None:
    st.markdown("##### Schéma CSV attendu")
    st.dataframe(
        batch_logic.feature_schema_frame(card),
        hide_index=True,
        width="stretch",
        alt="Authoritative 20-feature CSV schema with accepted ranges",
    )
    st.caption(
        "Les colonnes doivent porter exactement ces noms et apparaître dans cet ordre "
        "dans le modèle, mais l'ordre des colonnes du CSV est accepté. Les colonnes "
        "supplémentaires sont conservées dans l'export sans être utilisées pour le score."
    )
    st.download_button(
        "Télécharger le modèle de CSV (en-têtes)",
        data=batch_logic.template_csv_bytes(card),
        file_name="firewatch_batch_template.csv",
        mime="text/csv",
        key="batch_template_download",
        width="stretch",
    )


def _render_explanation(
    bundle: app_core.ModelBundle,
    result: batch_logic.BatchInferenceResult,
) -> None:
    st.markdown("##### Explicabilité — ce que l'on peut affirmer")

    st.warning(
        "**SHAP n'est pas utilisé comme dépendance obligatoire.** L'artefact déployé est "
        "un `CalibratedClassifierCV` (étalonnage isotonique) autour d'un "
        "`HistGradientBoostingClassifier`. `TreeExplainer` ne prend pas en charge ce "
        "wrapper : l'appliquer au modèle interne expliquerait des marges non calibrées, "
        "pas la probabilité affichée. Une approche SHAP model-agnostic nécessiterait un "
        "jeu de fond représentatif et resterait approximative. Nous documentons donc "
        "cette limite au lieu de produire des attributions trompeuses."
    )

    if result.accepted_count == 0:
        st.info("Aucune ligne valide : aucune explication locale ne peut être calculée.")
    else:
        selected = st.selectbox(
            "Ligne à expliquer",
            options=result.accepted_row_numbers,
            format_func=lambda number: f"Ligne {number}",
            key="batch_explain_row",
        )
        position = result.accepted_row_numbers.index(selected)
        features = result.accepted_features.iloc[position].to_dict()
        probability = float(result.accepted_probabilities[position])

        col_prob, col_band = st.columns(2)
        col_prob.metric("Probabilité calibrée", f"{probability:.4%}")
        col_band.metric(
            "Catégorie de risque",
            str(result.table.loc[result.table["row_number"] == selected, "risk_category"].iloc[0]),
        )

        sensitivity = predictor.feature_sensitivity(bundle, features, top_n=7)
        st.dataframe(
            sensitivity,
            hide_index=True,
            width="stretch",
            alt="Local one-feature-at-a-time sensitivity for the selected row",
        )
        st.caption(
            "Sensibilité locale : chaque ligne remplace une seule variable par une valeur "
            "de référence et mesure la variation de probabilité, en points de pourcentage. "
            "Ce n'est pas une attribution causale : les variables météo sont corrélées "
            "et une modification isolée peut être irréaliste."
        )

    if _IMPORTANCE_FIGURE.exists():
        st.image(
            str(_IMPORTANCE_FIGURE),
            caption=(
                "Importance globale par permutation calculée dans report/notebook.py sur "
                "un échantillon chronologique de la période de test (top 14)."
            ),
            alt="Global permutation importance of the 14 most influential features",
            width="stretch",
        )
    else:
        st.info("Figure d'importance globale indisponible : lancez report/notebook.py.")

    st.info(
        "L'importance par permutation décrit la sensibilité moyenne du modèle sur un "
        "échantillon donné. Elle dépend du choix de la métrique et des corrélations entre "
        "variables. Ni cette figure ni la sensibilité locale n'établissent un lien de "
        "causalité, et elles ne doivent pas être lues comme une explication météorologique "
        "d'un feu observé."
    )


def render_batch_view() -> None:
    """Render the batch CSV inference workflow in UI V2."""
    st.html(
        """
        <div style="margin-bottom: 12px;">
            <div style="font-size: 1.5rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">
                📄 Inférence par lots (CSV)
            </div>
            <div style="font-size: 0.9rem; color: #475569; margin-top: 4px;">
                Scoring reproductible avec le modèle de production — 20 variables, échéance H = 3 jours.
            </div>
        </div>
        """
    )
    st.info(
        "**Contrat strict** — le CSV doit contenir les 20 variables du modèle. Une valeur "
        "manquante, non numérique ou hors plage entraîne le rejet explicite de la ligne : "
        "aucune valeur n'est imputée et aucune ligne invalide n'est scorée."
    )

    bundle = load_model()
    if bundle.status != "REAL" or bundle.card is None:
        st.error(
            "Modèle de production indisponible : l'inférence par lots est arrêtée. "
            f"{bundle.error or ''}"
        )
        return

    with st.expander("Voir le schéma et télécharger un modèle de CSV", expanded=False):
        _render_schema(bundle.card)

    uploaded = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"],
        key="batch_csv_uploader",
        help="Une ligne = une cellule-jour décrite par les 20 variables du modèle.",
    )
    if uploaded is None:
        st.caption("Importez un CSV pour afficher la table de résultats et l'export.")
        return

    try:
        payload = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        raw = batch_logic.read_csv_bytes(payload)
    except Exception as exc:  # noqa: BLE001 - surface the parser's own message
        st.error(f"Lecture CSV impossible : {exc}")
        return

    if raw.empty:
        st.warning("Le fichier ne contient aucune ligne de données.")
        return

    try:
        result = batch_logic.run_batch_inference(raw, bundle)
    except app_core.ModelNotReadyError as exc:
        st.error(str(exc))
        return

    accepted, rejected = st.columns(2)
    accepted.metric("Lignes acceptées et scorées", result.accepted_count)
    rejected.metric("Lignes rejetées", result.rejected_count)
    st.caption(
        f"Fichier : {uploaded.name} — {len(raw)} lignes — "
        f"empreinte SHA-256 du contenu : {hashlib.sha256(payload).hexdigest()[:16]}…"
    )

    results_tab, rejected_tab, explanation_tab = st.tabs(
        ["Résultats", "Lignes rejetées", "Explicabilité"]
    )
    with results_tab:
        st.dataframe(
            result.table[_SUMMARY_COLUMNS],
            hide_index=True,
            width="stretch",
            alt="Batch inference results with validation status and risk category",
        )
        st.download_button(
            "Télécharger les résultats (CSV)",
            data=batch_logic.to_csv_bytes(result.table),
            file_name="firewatch_batch_predictions.csv",
            mime="text/csv",
            key="batch_download_csv",
            width="stretch",
        )
        with st.expander("Afficher les valeurs d'origine et les colonnes supplémentaires"):
            input_columns = [
                column for column in result.table.columns if column not in _SUMMARY_COLUMNS
            ]
            st.dataframe(
                result.table[input_columns],
                hide_index=True,
                width="stretch",
                alt="Original uploaded values preserved in the export",
            )
    with rejected_tab:
        rejected_rows = result.table[
            result.table["validation_status"] == batch_logic.STATUS_REJECTED
        ]
        if rejected_rows.empty:
            st.success("Aucune ligne rejetée.")
        else:
            st.dataframe(
                rejected_rows[_SUMMARY_COLUMNS],
                hide_index=True,
                width="stretch",
                alt="Rejected rows with explicit reasons",
            )
            st.caption(
                "Les lignes rejetées ne reçoivent ni probabilité ni catégorie de risque. "
                "Corrigez les valeurs signalées puis réimportez le fichier."
            )
    with explanation_tab:
        _render_explanation(bundle, result)
