"""Streamlit view for statistical validation and fair model benchmarking.

All statistical calculations and benchmark harmonisation live in pure modules
(``src/statistical_validation.py`` and ``src/model_benchmark.py``). This view
only loads precomputed artifacts, renders tables/figures and keeps every
p-value attached to its hypotheses, assumptions and scientific caveats.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

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

import model_benchmark as benchmark_logic  # noqa: E402
import statistical_validation as stats_logic  # noqa: E402
from dashboard.config import COLORS  # noqa: E402

_FIRE_COLOR = COLORS["risk_very_high"]
_BLUE_COLOR = COLORS["brand_primary"]
_NEUTRAL_COLOR = COLORS["text_muted"]

_METRIC_OPTIONS: dict[str, str] = {
    "PR-AUC (critère principal)": "pr_auc",
    "ROC-AUC": "roc_auc",
    "F1 au seuil OOF": "f1_optimal",
    "Brier (plus bas est meilleur)": "brier_score",
}

_DISPLAY_COLUMNS = [
    "model_label",
    "family",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "optimal_threshold",
    "precision_optimal",
    "recall_optimal",
    "f1_optimal",
    "accuracy_default_05",
    "threshold_source",
    "source_file",
]


# ---------------------------------------------------------------------------
# Cached artifact loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def get_statistical_report() -> dict[str, Any] | None:
    return stats_logic.load_statistical_report()


@st.cache_data(show_spinner=False)
def get_benchmark_artifacts() -> tuple[pd.DataFrame, dict[str, Any]]:
    return benchmark_logic.load_benchmark_results(), benchmark_logic.load_benchmark_protocol()


# ---------------------------------------------------------------------------
# Pure display helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value is True or value is False:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:,.{digits}f}"


def _assumption_icon(passed: bool | None) -> str:
    if passed is True:
        return "✅"
    if passed is False:
        return "⚠️"
    return "ℹ️"


def test_summary_frame(tests: list[dict[str, Any]], alpha: float) -> pd.DataFrame:
    """Build the compact overview table of the four hypothesis tests."""
    return pd.DataFrame([
        {
            "test": test.get("title", test.get("test_id", "—")),
            "statistic": _fmt(test.get("statistic"), 4),
            "p_value": stats_logic.format_p_value(test.get("p_value")),
            "alpha": f"{alpha:g}",
            "decision": stats_logic.significance_label(test, alpha),
            "status": test.get("status", "—"),
        }
        for test in tests
    ])


def _style(fig: go.Figure, title: str, x_title: str, y_title: str, height: int = 330) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", font=dict(size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=55, r=30, t=48, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color=COLORS["text_secondary"], size=11),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=COLORS["border_light"]),
    )
    fig.update_xaxes(title_text=x_title, gridcolor=COLORS["surface_subtle"], zeroline=False)
    fig.update_yaxes(title_text=y_title, gridcolor=COLORS["surface_subtle"], zeroline=False)
    return fig


def pvalue_chart(tests: list[dict[str, Any]], alpha: float) -> go.Figure | None:
    """Horizontal p-value lollipop with the decision threshold (log scale)."""
    points = [test for test in tests if test.get("p_value") is not None]
    if not points:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[max(float(test["p_value"]), 1e-300) for test in points],
        y=[test.get("title", test.get("test_id", "")) for test in points],
        mode="markers",
        marker=dict(size=12, color=_FIRE_COLOR, line=dict(width=1, color="#7F1D1D")),
        customdata=[stats_logic.format_p_value(test["p_value"]) for test in points],
        hovertemplate="%{y}<br>p = %{customdata}<extra></extra>",
        name="p-valeur observée",
    ))
    fig.add_vline(x=alpha, line_dash="dash", line_color=_BLUE_COLOR,
                  annotation_text=f"α = {alpha:g}", annotation_position="top")
    fig.update_xaxes(type="log", range=[-300.0, 0.0])
    return _style(fig, "Synthèse des p-valeurs", "p-valeur (échelle logarithmique)", "Test")


def contingency_chart(test: Mapping[str, Any]) -> go.Figure | None:
    """Observed FIRMS detection rate by season with counts in hover text."""
    contingency = test.get("contingency")
    if not contingency or not contingency.get("detection_rates"):
        return None
    rows = contingency["rows"]
    rates = [100.0 * float(rate) for rate in contingency["detection_rates"]]
    counts = contingency["counts"]
    detected = [row[-1] for row in counts]
    totals = [sum(row) for row in counts]
    fig = go.Figure(go.Bar(
        x=rows,
        y=rates,
        text=[f"{rate:.2f} %" for rate in rates],
        textposition="outside",
        marker_color=_FIRE_COLOR,
        customdata=np.column_stack([detected, totals]),
        hovertemplate=("%{x}<br>Détections : %{customdata[0]:,}<br>"
                       "Cellules-jours : %{customdata[1]:,}<br>Taux : %{y:.2f} %<extra></extra>"),
    ))
    return _style(fig, "Taux observé de détection FIRMS par saison",
                  "Saison météorologique", "Cellules-jours détectées (%)")


def group_summary_chart(test: Mapping[str, Any], title: str) -> go.Figure | None:
    """Median and interquartile range by group (never a fabricated raw sample)."""
    summaries = test.get("group_summaries") or []
    if not summaries:
        return None
    fig = go.Figure()
    for index, summary in enumerate(summaries):
        fig.add_trace(go.Bar(
            x=[summary["group"]],
            y=[summary["median"]],
            error_y=dict(
                type="data",
                array=[max(float(summary["q3"]) - float(summary["median"]), 0.0)],
                arrayminus=[max(float(summary["median"]) - float(summary["q1"]), 0.0)],
                visible=True,
            ),
            marker_color=[_BLUE_COLOR, _FIRE_COLOR, "#F59E0B"][index % 3],
            customdata=[[summary["n"], summary["q1"], summary["q3"]]],
            hovertemplate=("%{x}<br>Médiane : %{y:.1f} °C<br>"
                           "Q1–Q3 : %{customdata[1]:.1f}–%{customdata[2]:.1f} °C<br>"
                           "n = %{customdata[0]:,}<extra></extra>"),
            name=f"médiane — {summary['group']}",
            showlegend=False,
        ))
    return _style(fig, title, "Groupe", "T2M_MAX — médiane et IQR (°C)")


def normality_histogram(test: Mapping[str, Any]) -> go.Figure | None:
    """Observed sample histogram against a fitted normal expectation curve."""
    histogram = test.get("histogram")
    if not histogram:
        return None
    centers = histogram["bin_centers"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centers, y=histogram["observed_counts"], name="Échantillon observé",
        marker_color=_BLUE_COLOR, opacity=0.65,
        hovertemplate="T2M_MAX ≈ %{x:.1f} °C<br>Effectif : %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=centers, y=histogram["normal_expected_counts"], name="Loi normale ajustée",
        mode="lines+markers", line=dict(color=_FIRE_COLOR, width=2),
        hovertemplate="T2M_MAX ≈ %{x:.1f} °C<br>Attendu : %{y:,.0f}<extra></extra>",
    ))
    return _style(fig, "Normalité de T2M_MAX (échantillon borné)",
                  "T2M_MAX (°C)", "Effectif par classe", height=300)


def benchmark_display_frame(benchmark: pd.DataFrame, *, include_tuned: bool) -> pd.DataFrame:
    """Return the benchmark rows visible in the UI."""
    if benchmark.empty:
        return benchmark
    primary = benchmark[benchmark["experiment"] == benchmark_logic.PRIMARY_EXPERIMENT]
    if not include_tuned:
        return primary.copy()
    tuned = benchmark[benchmark["experiment"] == benchmark_logic.TUNED_EXPERIMENT]
    return pd.concat([primary, tuned], ignore_index=True)


def benchmark_metric_chart(
    benchmark: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    base_rate: float | None,
) -> go.Figure | None:
    """Bar chart of one metric for the comparable primary experiments only."""
    primary = benchmark[benchmark["experiment"] == benchmark_logic.PRIMARY_EXPERIMENT]
    if primary.empty or metric not in primary:
        return None
    frame = primary.sort_values(metric, ascending=(metric == "brier_score"))
    fig = px.bar(
        frame,
        x="model_label",
        y=metric,
        color="family",
        text=frame[metric].map(lambda value: _fmt(value, 4)),
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={"family": True, "source_file": True, metric: True},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    if metric == "pr_auc" and base_rate is not None and np.isfinite(base_rate):
        fig.add_hline(
            y=float(base_rate), line_dash="dash", line_color=_FIRE_COLOR,
            annotation_text=f"aléatoire = {base_rate:.3f}", annotation_position="top left",
        )
    return _style(fig, f"Comparaison — {metric_label}", "Modèle", metric_label, height=380)


def threshold_metrics_chart(benchmark: pd.DataFrame) -> go.Figure | None:
    """Precision / recall / F1 / accuracy at the same decision rules."""
    primary = benchmark[benchmark["experiment"] == benchmark_logic.PRIMARY_EXPERIMENT]
    if primary.empty:
        return None
    metrics = ["precision_optimal", "recall_optimal", "f1_optimal", "accuracy_default_05"]
    labels = ["Précision (seuil OOF)", "Rappel (seuil OOF)", "F1 (seuil OOF)", "Accuracy (seuil 0,5)"]
    long = primary.melt(
        id_vars=["model_label"], value_vars=metrics, var_name="metric", value_name="value"
    )
    long["metric"] = long["metric"].map(dict(zip(metrics, labels)))
    fig = px.bar(
        long, x="model_label", y="value", color="metric", barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={"value": ":.4f"},
    )
    return _style(fig, "Métriques au seuil — la précision et le rappel comptent",
                  "Modèle", "Valeur (0–1)", height=400)


def _kv_frame(mapping: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"paramètre": key, "valeur": str(value)} for key, value in mapping.items()]
    )


def _algorithm_frame(protocol: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for name, details in (protocol.get("algorithms", {}).get("excluded", {}) or {}).items():
        rows.append({
            "algorithme": name,
            "inclus": "non",
            "raison": details.get("reason", "—"),
        })
    for name in protocol.get("algorithms", {}).get("included", []) or []:
        rows.append({"algorithme": name, "inclus": "oui", "raison": "évaluation archivée valide"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_statistics_tab(report: dict[str, Any] | None, alpha: float) -> None:
    if report is None:
        st.warning("Rapport statistique indisponible : lancez scripts/build_statistical_artifacts.py.")
        return
    tests = report.get("tests", [])
    summary = test_summary_frame(tests, alpha)
    significant = sum(1 for test in tests if test.get("status") == "ok"
                      and test.get("p_value") is not None
                      and float(test["p_value"]) < alpha)
    dataset = report.get("dataset", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Tests réalisés", len(tests))
    col2.metric(f"Significatifs à α = {alpha:g}", significant)
    col3.metric("Cellules-jours analysées", f"{dataset.get('rows', 0):,}")

    st.dataframe(summary, hide_index=True, width="stretch",
                 alt="Overview of the four hypothesis tests with p-values and decisions")
    st.caption(
        "Hypothèses, conditions de validité et interprétation sont détaillées ci-dessous. "
        "Une p-valeur ne mesure ni la taille de l'effet ni son utilité opérationnelle."
    )

    p_chart = pvalue_chart(tests, alpha)
    if p_chart is not None:
        st.plotly_chart(p_chart, width="stretch", alt="Statistical test p-values with alpha threshold")

    for test in tests:
        with st.expander(f"{test.get('title', test.get('test_id'))} — {stats_logic.significance_label(test, alpha)}"):
            st.markdown(f"**Question.** {test.get('question', '')}")
            st.markdown(f"**{test.get('null_hypothesis', '')}**")
            st.markdown(f"**{test.get('alternative_hypothesis', '')}**")
            st.markdown(f"**Pourquoi ce test ?** {test.get('why_appropriate', '')}")
            if test.get("warning"):
                st.caption(f"⚠️ {test['warning']}")

            assumptions = pd.DataFrame([
                {
                    "condition": item.get("name", "—"),
                    "vérifiée": _assumption_icon(item.get("passed")),
                    "détail": item.get("detail", ""),
                }
                for item in test.get("assumptions", [])
            ])
            if not assumptions.empty:
                st.dataframe(assumptions, hide_index=True, width="stretch",
                             alt=f"Assumption checks for {test.get('test_id')}")

            effect = test.get("effect_size") or {}
            m1, m2, m3 = st.columns(3)
            m1.metric("Statistique", _fmt(test.get("statistic"), 3))
            m2.metric("p-valeur", stats_logic.format_p_value(test.get("p_value")))
            m3.metric("Taille d'effet", f"{_fmt(effect.get('value'), 3)}"
                      if effect else "—")
            if effect:
                st.caption(f"Taille d'effet : {effect.get('name')} — {effect.get('magnitude')}.")
            st.markdown(stats_logic.interpretation_text(test, alpha))
            if test.get("status") == "insufficient_data":
                st.info("Échantillon insuffisant : aucun résultat n'est rapporté pour ce test.")

            if test.get("test_id") == "chi2_season_detection":
                chart = contingency_chart(test)
                if chart is not None:
                    st.plotly_chart(chart, width="stretch", alt="Detection rate by meteorological season")
                    cont = test["contingency"]
                    counts = pd.DataFrame(cont["counts"], index=cont["rows"], columns=cont["columns"])
                    st.dataframe(counts, alt="Observed contingency table for chi-square test")
            elif test.get("test_id") == "kruskal_tmax_by_season":
                chart = group_summary_chart(test, "Température maximale par saison")
                if chart is not None:
                    st.plotly_chart(chart, width="stretch", alt="Seasonal temperature medians and IQR")
            elif test.get("test_id") == "shapiro_tmax_normality":
                chart = normality_histogram(test)
                if chart is not None:
                    st.plotly_chart(chart, width="stretch", alt="Shapiro sample histogram versus fitted normal curve")
            elif test.get("test_id") == "mannwhitney_tmax_by_detection":
                chart = group_summary_chart(test, "Température maximale selon la détection FIRMS")
                if chart is not None:
                    st.plotly_chart(chart, width="stretch", alt="Temperature medians by FIRMS detection group")

    st.caption("Source : NASA POWER (T2M_MAX) et proxy FIRMS « burned ». Aucune prévision n'est utilisée ici.")


def _render_benchmark_tab(benchmark: pd.DataFrame, protocol: Mapping[str, Any]) -> None:
    if benchmark.empty:
        st.warning("Benchmark indisponible : lancez scripts/build_statistical_artifacts.py.")
        return

    metric_label = st.selectbox(
        "Classer les modèles par",
        options=list(_METRIC_OPTIONS),
        index=0,
        key="stats_benchmark_metric",
    )
    metric = _METRIC_OPTIONS[metric_label]
    include_tuned = st.checkbox(
        "Afficher aussi la référence tuned (seuil ajusté sur validation)",
        value=False,
        key="stats_include_tuned",
        help="Ces lignes ne sont pas classées : leurs métriques de seuil sont optimistes.",
    )

    ranking = benchmark[benchmark["experiment"] == benchmark_logic.PRIMARY_EXPERIMENT]
    ascending = metric == "brier_score"
    ranking = ranking.sort_values(metric, ascending=ascending, kind="mergesort")
    visible = benchmark_display_frame(benchmark, include_tuned=include_tuned)
    if include_tuned:
        tuned = benchmark[benchmark["experiment"] == benchmark_logic.TUNED_EXPERIMENT]
        visible = pd.concat([ranking, tuned], ignore_index=True)
    else:
        visible = ranking.copy()

    base_rate = (protocol.get("class_balance") or {}).get("val_positive_rate")
    col1, col2, col3 = st.columns(3)
    col1.metric("Meilleur PR-AUC",
                _fmt(ranking.sort_values("pr_auc", ascending=False)["pr_auc"].iloc[0], 4))
    col2.metric("Taux de positifs (validation)", f"{float(base_rate):.2%}" if base_rate is not None else "—")
    col3.metric("Modèles comparables", len(ranking))

    st.dataframe(
        visible[_DISPLAY_COLUMNS],
        hide_index=True,
        width="stretch",
        alt="Model benchmark metrics on the common temporal validation split",
    )
    st.caption(
        "Seules les lignes « primary » sont classées. La référence tuned est conservée pour information : "
        "son seuil a été ajusté sur la validation, ce qui rend ses métriques de seuil optimistes."
    )

    chart = benchmark_metric_chart(
        benchmark, metric=metric, metric_label=metric_label, base_rate=base_rate
    )
    if chart is not None:
        st.plotly_chart(chart, width="stretch", alt=f"Benchmark comparison by {metric}")
    threshold_chart = threshold_metrics_chart(benchmark)
    if threshold_chart is not None:
        st.plotly_chart(threshold_chart, width="stretch", alt="Precision recall F1 and accuracy comparison")

    st.markdown("##### Comprendre les métriques")
    for note in protocol.get("interpretation_notes", []) or []:
        st.markdown(f"- {note}")
    st.caption(
        "Toutes les lignes partagent le même découpage temporel, les 20 mêmes features et la graine 42. "
        "Aucun résultat issu d'un autre jeu ou d'une autre période n'est ajouté."
    )


def _render_protocol_tab(report: dict[str, Any] | None, protocol: Mapping[str, Any]) -> None:
    split = protocol.get("split") or {}
    if split:
        st.markdown("##### Découpage et contrôle des fuites")
        st.dataframe(_kv_frame(split), hide_index=True, width="stretch",
                     alt="Temporal split and anti-leakage protocol")
    balance = protocol.get("class_balance") or {}
    if balance:
        st.markdown("##### Déséquilibre des classes")
        st.dataframe(_kv_frame(balance), hide_index=True, width="stretch",
                     alt="Class balance on train and validation")
    algorithms = _algorithm_frame(protocol)
    if not algorithms.empty:
        st.markdown("##### Algorithmes inclus et exclus")
        st.dataframe(algorithms, hide_index=True, width="stretch",
                     alt="Included and excluded benchmark algorithms with reasons")

    st.markdown("##### Protection du modèle déployé")
    contract = protocol.get("model_contract") or {}
    st.info(
        f"Contrat gelé : {contract.get('n_features', 20)} features, graine {protocol.get('seed', 42)}. "
        "Les expériences n'écrivent ni dans models/fire_model.joblib ni dans models/model_card.json "
        "et ne changent pas l'inférence en production."
    )
    st.markdown("##### Limites de l'évaluation statistique")
    limitations = (report or {}).get("limitations") or [
        "La significativité statistique n'implique pas une importance pratique.",
        "Une performance de validation ne garantit pas la performance réelle en conditions opérationnelles.",
    ]
    for item in limitations:
        st.markdown(f"- {item}")

    st.markdown("##### Téléchargements reproductibles")
    benchmark_path = ROOT / "results" / benchmark_logic.BENCHMARK_CSV.name
    report_path = ROOT / "results" / stats_logic.STATISTICAL_REPORT_PATH.name
    if benchmark_path.exists():
        st.download_button(
            "Télécharger le benchmark (CSV)",
            data=benchmark_path.read_bytes(),
            file_name=benchmark_logic.BENCHMARK_CSV.name,
            mime="text/csv",
            key="stats_download_benchmark",
        )
    if report_path.exists():
        st.download_button(
            "Télécharger le rapport statistique (JSON)",
            data=report_path.read_bytes(),
            file_name=stats_logic.STATISTICAL_REPORT_PATH.name,
            mime="application/json",
            key="stats_download_report",
        )


def render_statistics_view() -> None:
    """Render the statistical validation and benchmark section in UI V2."""
    st.html(
        """
        <div style="margin-bottom: 12px;">
            <div style="font-size: 1.5rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">
                🧪 Validation statistique & Benchmark des modèles
            </div>
            <div style="font-size: 0.9rem; color: #475569; margin-top: 4px;">
                Tests d'hypothèses sur les observations et comparaison à découpage temporel identique.
            </div>
        </div>
        """
    )
    st.info(
        "**Périmètre** — les tests décrivent des observations historiques (NASA POWER, proxy FIRMS). "
        "Le benchmark est expérimental et n'utilise ni ne modifie le modèle de production. "
        "Une association statistique n'est pas une preuve de causalité."
    )

    report = get_statistical_report()
    benchmark, protocol = get_benchmark_artifacts()
    tests_tab, benchmark_tab, protocol_tab = st.tabs(
        ["Tests statistiques", "Benchmark des modèles", "Protocole & limites"]
    )
    with tests_tab:
        alpha = st.selectbox(
            "Seuil de significativité α",
            options=[0.01, 0.05, 0.10],
            index=1,
            key="stats_alpha",
            help="Le seuil est documenté et appliqué à tous les tests ; il ne change pas les p-valeurs.",
        )
        _render_statistics_tab(report, float(alpha))
    with benchmark_tab:
        _render_benchmark_tab(benchmark, protocol)
    with protocol_tab:
        _render_protocol_tab(report, protocol)
