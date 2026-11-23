"""Formal, reusable statistical validation for the observed FireWatch panel.

The four tests implemented here are deliberately tied to questions that the
shipped dataset can actually answer:

* chi-square: is FIRMS detection frequency independent of meteorological
  season? (two categorical variables);
* Kruskal-Wallis: does daily maximum temperature differ across seasons?
  (one non-normal continuous variable across four groups);
* Shapiro-Wilk: is the sampled temperature distribution approximately normal?
  (needed to justify why rank-based tests are preferred here);
* Mann-Whitney U: do detection and non-detection cell-days differ in daily
  maximum temperature? (two non-normal, unequal-size groups).

Every result keeps its hypotheses, assumptions, statistic, p-value and a plain
language interpretation together, so the Streamlit layer cannot detach a
p-value from its scientific context. The functions never raise on missing,
short or malformed observational inputs: they return a structured
``insufficient_data`` / ``invalid_input`` result instead.

No model artifact is read, trained or modified by this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from eda import MONTH_TO_SEASON, SEASON_ORDER, WEATHER_VARIABLES

ROOT = Path(__file__).resolve().parent.parent
STATISTICAL_REPORT_PATH = ROOT / "results" / "statistical_validation.json"

DEFAULT_ALPHA = 0.05
SHAPIRO_SAMPLE_CAP = 5000
DEFAULT_SEED = 42
EXPECTED_TEST_IDS = (
    "chi2_season_detection",
    "kruskal_tmax_by_season",
    "shapiro_tmax_normality",
    "mannwhitney_tmax_by_detection",
)

_LIMITATIONS = (
    "Les cellules-jours répètent les mêmes cellules et des jours consécutifs : "
    "l'indépendance stricte des observations n'est pas garantie, donc les p-valeurs "
    "sont à lire comme des indicateurs descriptifs et non comme une preuve causale.",
    "Un résultat statistiquement significatif n'implique pas un effet important "
    "pour la décision opérationnelle ; l'ampleur de l'effet et le contexte comptent.",
    "Une performance de validation, même bonne, ne garantit pas la performance "
    "réelle d'un modèle de prévision des feux de forêt.",
    "Le proxy « burned » désigne une cellule-jour avec au moins une détection FIRMS ; "
    "ce n'est ni une surface brûlée confirmée ni un dégât matériel.",
)


# ---------------------------------------------------------------------------
# Small, pure helpers
# ---------------------------------------------------------------------------


def _safe_alpha(alpha: Any) -> tuple[float, str | None]:
    """Return a usable alpha and an optional warning instead of raising."""
    try:
        value = float(alpha)
    except (TypeError, ValueError):
        return DEFAULT_ALPHA, "alpha illisible : 0,05 utilisé."
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        return DEFAULT_ALPHA, "alpha doit être strictement entre 0 et 1 : 0,05 utilisé."
    return value, None


def _native(value: Any) -> Any:
    """Convert numpy scalars and timestamps into JSON-native values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _finite_numeric(values: Any) -> pd.Series:
    """Coerce to numeric, drop NaN/inf and return a clean Series."""
    series = pd.Series(values).copy()
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna()


def _assumption(name: str, passed: bool | None, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def _base_result(
    test_id: str,
    title: str,
    question: str,
    null_hypothesis: str,
    alternative_hypothesis: str,
    why_appropriate: str,
    *,
    alpha: float,
    warning: str | None = None,
) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "title": title,
        "question": question,
        "null_hypothesis": null_hypothesis,
        "alternative_hypothesis": alternative_hypothesis,
        "why_appropriate": why_appropriate,
        "alpha": alpha,
        "status": "invalid_input",
        "statistic": None,
        "p_value": None,
        "p_display": "—",
        "significant": None,
        "effect_size": None,
        "sample": {},
        "assumptions": [],
        "interpretation": "",
        "contingency": None,
        "group_summaries": [],
        "histogram": None,
        "warning": warning,
    }


def _finalize(
    result: dict[str, Any],
    *,
    status: str,
    interpretation: str,
    alpha: float,
    statistic: float | None = None,
    p_value: float | None = None,
) -> dict[str, Any]:
    result["status"] = status
    result["interpretation"] = interpretation
    result["statistic"] = None if statistic is None else _native(float(statistic))
    result["p_value"] = None if p_value is None else _native(float(p_value))
    result["p_display"] = format_p_value(p_value)
    result["significant"] = (
        bool(float(p_value) < alpha) if p_value is not None and status == "ok" else None
    )
    return result


def _failed(test_id: str, title: str, reason: str, *, alpha: float) -> dict[str, Any]:
    result = _base_result(
        test_id=test_id,
        title=title,
        question="Le test n'a pas pu être exécuté.",
        null_hypothesis="—",
        alternative_hypothesis="—",
        why_appropriate="—",
        alpha=alpha,
        warning=reason,
    )
    result["interpretation"] = reason
    result["status"] = "invalid_input"
    return result


def format_p_value(p_value: Any) -> str:
    """Display very small p-values honestly instead of printing a misleading zero."""
    if p_value is None:
        return "—"
    try:
        value = float(p_value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    if value == 0.0:
        return "< 1e-300"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def significance_label(result: Mapping[str, Any], alpha: Any = None) -> str:
    """Return a stable significance label for a result at the requested alpha."""
    if result.get("p_value") is None or result.get("status") not in {"ok", "assumption_violation"}:
        return "Non évaluable"
    level = _safe_alpha(alpha if alpha is not None else result.get("alpha", DEFAULT_ALPHA))[0]
    if float(result["p_value"]) < level:
        return f"Significatif à α = {level:g}"
    return f"Non significatif à α = {level:g}"


def interpretation_text(result: Mapping[str, Any], alpha: Any = None) -> str:
    """Combine the stored explanation with the significance decision at alpha."""
    base = str(result.get("interpretation", "")).strip()
    label = significance_label(result, alpha)
    if result.get("p_value") is None:
        return base
    decision = f"Décision : {label}."
    if result.get("status") == "assumption_violation":
        decision += " Réserve : au moins une hypothèse du test n'est pas vérifiée."
    return f"{base} {decision}".strip()


def _summarize(values: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "q1": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "q3": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _effect_label(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.01:
        return "négligeable"
    if magnitude < 0.05:
        return "faible"
    if magnitude < 0.10:
        return "modérée"
    if magnitude < 0.25:
        return "importante"
    return "très importante"


# ---------------------------------------------------------------------------
# Test 1 — Chi-square independence
# ---------------------------------------------------------------------------


def chi_square_season_detection(
    frame: pd.DataFrame,
    *,
    category_col: str = "season",
    outcome_col: str = "burned",
    alpha: Any = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Chi-square test of independence between season and FIRMS detection proxy."""
    level, alpha_warning = _safe_alpha(alpha)
    result = _base_result(
        test_id="chi2_season_detection",
        title="Test du χ² — saison et détection FIRMS",
        question="La fréquence de détection FIRMS dépend-elle de la saison météorologique ?",
        null_hypothesis="H0 : la saison et la présence d'une détection sont indépendantes.",
        alternative_hypothesis="H1 : la fréquence de détection diffère selon la saison.",
        why_appropriate=(
            "Les deux variables sont catégorielles (4 saisons × détection oui/non) et "
            "les effectifs attendus sont vérifiés avant interprétation."
        ),
        alpha=level,
        warning=alpha_warning,
    )
    if not isinstance(frame, pd.DataFrame) or category_col not in frame or outcome_col not in frame:
        result["warning"] = result["warning"] or "Colonnes « season » ou « burned » absentes."
        return result

    clean = frame[[category_col, outcome_col]].dropna()
    if clean.empty:
        return _finalize(result, status="insufficient_data",
                         interpretation="Aucune observation exploitable pour le test du χ².", alpha=level)

    table = pd.crosstab(clean[category_col], clean[outcome_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return _finalize(
            result,
            status="insufficient_data",
            interpretation=(
                f" table {table.shape[0]}×{table.shape[1]} : il faut au moins deux saisons "
                "et deux modalités de détection."
            ).strip(),
            alpha=level,
        )

    ordered_index = [s for s in SEASON_ORDER if s in table.index]
    ordered_index += [str(s) for s in table.index if s not in ordered_index]
    table = table.reindex(ordered_index, fill_value=0)
    columns = sorted(table.columns, key=str)
    table = table.reindex(columns=columns, fill_value=0)
    counts = table.to_numpy(dtype=float)

    try:
        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(counts, correction=False)
    except Exception as exc:  # pragma: no cover - SciPy raises only on degenerate tables
        result["warning"] = f"Test du χ² non exécuté : {exc}"
        return result

    min_expected = float(np.min(expected))
    expected_ok = min_expected >= 5.0
    n_total = int(counts.sum())
    cramers_v = float(np.sqrt(float(chi2) / (n_total * min(counts.shape[0] - 1, counts.shape[1] - 1))))
    result["statistic"] = float(chi2)
    result["p_value"] = _native(float(p_value))
    result["p_display"] = format_p_value(p_value)
    result["alpha"] = level
    result["sample"] = {"rows": n_total, "degrees_of_freedom": int(dof)}
    result["effect_size"] = {
        "name": "V de Cramér",
        "value": cramers_v,
        "magnitude": _effect_label(cramers_v),
    }
    result["assumptions"] = [
        _assumption(
            "Effectifs attendus ≥ 5",
            expected_ok,
            f"Effectif attendu minimal : {min_expected:.1f}.",
        ),
        _assumption(
            "Observations indépendantes",
            None,
            "Non vérifiable automatiquement : les cellules-jours se répètent dans l'espace et le temps.",
        ),
    ]

    detection_rates = None
    if table.shape[1] == 2:
        detection_col = columns[-1]
        detection_rates = [
            float(table.loc[idx, detection_col] / row_total) if row_total else 0.0
            for idx, row_total in zip(table.index, counts.sum(axis=1))
        ]
    result["contingency"] = {
        "rows": [str(idx) for idx in table.index],
        "columns": [str(col) for col in columns],
        "counts": counts.astype(int).tolist(),
        "detection_rates": detection_rates,
        "expected_min": min_expected,
    }
    interpretation = (
        "Le test compare les fréquences observées de détection entre saisons. "
        f"V de Cramér = {cramers_v:.3f} ({_effect_label(cramers_v)}). "
        "Une association saisonnière n'établit pas de lien de causalité."
    )
    return _finalize(
        result,
        status="ok" if expected_ok else "assumption_violation",
        interpretation=interpretation,
        alpha=level,
        statistic=chi2,
        p_value=p_value,
    )


# ---------------------------------------------------------------------------
# Test 2 — Kruskal-Wallis
# ---------------------------------------------------------------------------


def kruskal_groups(
    frame: pd.DataFrame,
    *,
    group_col: str = "season",
    value_col: str = "T2M_MAX",
    alpha: Any = DEFAULT_ALPHA,
    min_group_size: int = 3,
) -> dict[str, Any]:
    """Kruskal-Wallis rank test for one continuous variable across groups."""
    level, alpha_warning = _safe_alpha(alpha)
    result = _base_result(
        test_id="kruskal_tmax_by_season",
        title="Test de Kruskal-Wallis — température maximale par saison",
        question="La distribution de T2M_MAX diffère-t-elle entre les quatre saisons ?",
        null_hypothesis="H0 : les distributions de T2M_MAX sont identiques dans toutes les saisons.",
        alternative_hypothesis="H1 : au moins une saison a une distribution de T2M_MAX différente.",
        why_appropriate=(
            "T2M_MAX est une variable continue non normale (voir Shapiro) et les quatre "
            "saisons sont des groupes indépendants au sens du protocole ; le test de rang "
            "évite l'hypothèse de normalité requise par l'ANOVA."
        ),
        alpha=level,
        warning=alpha_warning,
    )
    if not isinstance(frame, pd.DataFrame) or group_col not in frame or value_col not in frame:
        result["warning"] = result["warning"] or f"Colonnes « {group_col} » ou « {value_col} » absentes."
        return result

    clean = frame[[group_col, value_col]].dropna(subset=[group_col])
    grouped = clean.groupby(group_col, sort=True, observed=True)[value_col]
    groups: dict[str, np.ndarray] = {}
    for name, values in grouped:
        numeric = _finite_numeric(values).to_numpy()
        if numeric.size:
            groups[str(name)] = numeric
    if group_col == "season":
        groups = {name: groups[name] for name in SEASON_ORDER if name in groups}
    if len(groups) < 2:
        return _finalize(
            result,
            status="insufficient_data",
            interpretation="Moins de deux groupes exploitables : le test de Kruskal-Wallis n'est pas applicable.",
            alpha=level,
        )

    sizes = [len(values) for values in groups.values()]
    sizes_ok = min(sizes) >= max(3, int(min_group_size))
    result["assumptions"] = [
        _assumption(
            "Échantillon suffisant par groupe",
            sizes_ok,
            f"Effectif minimal : {min(sizes):,} (seuil {max(3, int(min_group_size))}).",
        ),
        _assumption(
            "Mesures continues et groupes comparables",
            True,
            "T2M_MAX est une température (°C) mesurée de la même manière dans chaque saison.",
        ),
        _assumption(
            "Indépendance des groupes",
            None,
            "Les saisons sont distinctes, mais la répétition spatiale/temporelle reste une limite.",
        ),
    ]
    result["group_summaries"] = [
        {"group": name, **_summarize(values)} for name, values in groups.items()
    ]
    result["sample"] = {"groups": len(groups), "rows": int(sum(sizes))}
    if not sizes_ok:
        return _finalize(
            result,
            status="insufficient_data",
            interpretation="Un groupe contient trop peu d'observations pour une comparaison de rangs fiable.",
            alpha=level,
        )

    try:
        statistic, p_value = scipy_stats.kruskal(*groups.values())
    except Exception as exc:  # pragma: no cover - degenerate constant input
        result["warning"] = f"Test de Kruskal-Wallis non exécuté : {exc}"
        return result

    n_total = int(sum(sizes))
    k = len(groups)
    epsilon_sq = float(np.clip((float(statistic) - k + 1) / max(n_total - k, 1), 0.0, 1.0))
    result["effect_size"] = {
        "name": "epsilon²",
        "value": epsilon_sq,
        "magnitude": _effect_label(epsilon_sq),
    }
    medians = ", ".join(f"{name} {_summarize(values)['median']:.1f} °C"
                        for name, values in groups.items())
    interpretation = (
        f"Le test compare les rangs de température entre saisons ({medians}). "
        f"epsilon² = {epsilon_sq:.3f} ({_effect_label(epsilon_sq)}) : il mesure la part "
        "de variation de rang expliquée par la saison, pas une cause."
    )
    return _finalize(
        result,
        status="ok",
        interpretation=interpretation,
        alpha=level,
        statistic=statistic,
        p_value=p_value,
    )


# ---------------------------------------------------------------------------
# Test 3 — Shapiro-Wilk normality
# ---------------------------------------------------------------------------


def shapiro_normality(
    values: Any,
    *,
    alpha: Any = DEFAULT_ALPHA,
    max_sample: int = SHAPIRO_SAMPLE_CAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Shapiro-Wilk test on a reproducible bounded sample of a continuous variable."""
    level, alpha_warning = _safe_alpha(alpha)
    result = _base_result(
        test_id="shapiro_tmax_normality",
        title="Test de Shapiro-Wilk — normalité de T2M_MAX",
        question="La distribution observée de T2M_MAX est-elle compatible avec une loi normale ?",
        null_hypothesis="H0 : les valeurs échantillonnées proviennent d'une distribution normale.",
        alternative_hypothesis="H1 : la distribution s'écarte de la normalité.",
        why_appropriate=(
            "Shapiro-Wilk est conçu pour une variable continue et ici un échantillon "
            "aléatoire reproductible de 5 000 valeurs maximum : sur 500 000 cellules-jours, "
            "le test complet serait surpuissant et coûteux."
        ),
        alpha=level,
        warning=alpha_warning,
    )
    clean = _finite_numeric(values)
    if len(clean) < 3:
        return _finalize(
            result,
            status="insufficient_data",
            interpretation="Moins de trois valeurs finies : le test de normalité n'est pas applicable.",
            alpha=level,
        )

    cap = max(3, int(max_sample))
    rng = np.random.default_rng(int(seed))
    sample = rng.choice(clean.to_numpy(), size=min(cap, len(clean)), replace=False)
    result["sample"] = {
        "input_rows": int(len(clean)),
        "tested_rows": int(sample.size),
        "sampling": "aléatoire sans remplacement",
        "seed": int(seed),
        "cap": cap,
    }
    result["assumptions"] = [
        _assumption(
            "Taille compatible avec Shapiro-Wilk",
            sample.size <= SHAPIRO_SAMPLE_CAP,
            f"{sample.size:,} valeurs testées (plafond recommandé {SHAPIRO_SAMPLE_CAP:,}).",
        ),
        _assumption(
            "Valeurs continues et finies",
            bool(np.unique(sample).size > 2),
            "Les valeurs manquantes et infinies sont exclues avant échantillonnage.",
        ),
        _assumption(
            "Indépendance des observations",
            None,
            "L'échantillonnage est aléatoire mais les cellules-jours restent corrélées.",
        ),
    ]
    try:
        statistic, p_value = scipy_stats.shapiro(sample)
    except Exception as exc:  # pragma: no cover - defensive
        result["warning"] = f"Test de Shapiro-Wilk non exécuté : {exc}"
        return result

    hist, edges = np.histogram(sample, bins=20)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = float(edges[1] - edges[0])
    scale = float(np.std(sample, ddof=0)) or 1e-9
    expected = scipy_stats.norm.pdf(centers, loc=float(np.mean(sample)), scale=scale) * width * sample.size
    result["histogram"] = {
        "bin_centers": centers.tolist(),
        "observed_counts": hist.astype(int).tolist(),
        "normal_expected_counts": expected.tolist(),
        "unit": "°C",
        "note": "La courbe normale est ajustée à l'échantillon : elle sert d'illustration, pas de test visuel.",
    }
    result["statistic"] = float(statistic)
    result["p_value"] = _native(float(p_value))
    result["p_display"] = format_p_value(p_value)
    interpretation = (
        "W mesure la similarité avec une droite quantile-quantile normale. "
        "Avec un grand effectif, un écart même faible peut être détecté : ce test "
        "justifie le choix de tests de rangs plutôt qu'une preuve d'absence de normalité."
    )
    return _finalize(
        result,
        status="ok",
        interpretation=interpretation,
        alpha=level,
        statistic=statistic,
        p_value=p_value,
    )


# ---------------------------------------------------------------------------
# Test 4 — Mann-Whitney U
# ---------------------------------------------------------------------------


def mann_whitney_groups(
    frame: pd.DataFrame,
    *,
    group_col: str = "burned",
    value_col: str = "T2M_MAX",
    group_a: Any = 0,
    group_b: Any = 1,
    alpha: Any = DEFAULT_ALPHA,
    min_group_size: int = 3,
) -> dict[str, Any]:
    """Two-sided Mann-Whitney U test comparing one continuous variable between groups."""
    level, alpha_warning = _safe_alpha(alpha)
    result = _base_result(
        test_id="mannwhitney_tmax_by_detection",
        title="Test de Mann-Whitney U — température et détection",
        question="T2M_MAX se répartit-elle différemment selon la présence d'une détection FIRMS ?",
        null_hypothesis="H0 : les deux distributions de T2M_MAX ont la même position (stochastiquement identiques).",
        alternative_hypothesis="H1 : une des deux groupes présente des valeurs de T2M_MAX plus élevées.",
        why_appropriate=(
            "Deux groupes de tailles très inégales, une variable continue non normale "
            "et un grand effectif : le test de rang non paramétrique est adapté, "
            "contrairement au t de Student."
        ),
        alpha=level,
        warning=alpha_warning,
    )
    if not isinstance(frame, pd.DataFrame) or group_col not in frame or value_col not in frame:
        result["warning"] = result["warning"] or f"Colonnes « {group_col} » ou « {value_col} » absentes."
        return result

    values_a = _finite_numeric(frame.loc[frame[group_col] == group_a, value_col]).to_numpy()
    values_b = _finite_numeric(frame.loc[frame[group_col] == group_b, value_col]).to_numpy()
    result["sample"] = {"group_a_rows": int(values_a.size), "group_b_rows": int(values_b.size)}
    if min(values_a.size, values_b.size) < max(3, int(min_group_size)):
        return _finalize(
            result,
            status="insufficient_data",
            interpretation=(
                f"Effectifs insuffisants : {values_a.size:,} vs {values_b.size:,} valeurs exploitables."
            ),
            alpha=level,
        )

    result["group_summaries"] = [
        {"group": f"groupe {group_a} (sans détection)", **_summarize(values_a)},
        {"group": f"groupe {group_b} (avec détection)", **_summarize(values_b)},
    ]
    result["assumptions"] = [
        _assumption(
            "Deux échantillons exploitables",
            True,
            f"{values_a.size:,} et {values_b.size:,} valeurs finies.",
        ),
        _assumption(
            "Variable continue comparable",
            True,
            "La même variable T2M_MAX (°C) est mesurée dans les deux groupes.",
        ),
        _assumption(
            "Forme des distributions",
            None,
            "L'interprétation porte sur les rangs ; elle n'implique pas une différence de médiane si les formes diffèrent.",
        ),
        _assumption(
            "Indépendance des observations",
            None,
            "Les cellules-jours sont corrélées dans l'espace et le temps ; prudence sur la significativité.",
        ),
    ]
    try:
        statistic, p_value = scipy_stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
    except Exception as exc:  # pragma: no cover - defensive
        result["warning"] = f"Test de Mann-Whitney non exécuté : {exc}"
        return result

    rank_biserial = float((2.0 * float(statistic)) / (values_a.size * values_b.size) - 1.0)
    result["effect_size"] = {
        "name": "rang-bisérial",
        "value": rank_biserial,
        "magnitude": _effect_label(rank_biserial),
    }
    median_a = _summarize(values_a)["median"]
    median_b = _summarize(values_b)["median"]
    if rank_biserial < -0.05:
        direction = "le rang-bisérial négatif indique des températures plus élevées les jours avec détection"
    elif rank_biserial > 0.05:
        direction = "le rang-bisérial positif indique des températures plus élevées les jours sans détection"
    else:
        direction = "le déplacement de rangs est faible"
    interpretation = (
        f"Médianes observées : sans détection {median_a:.1f} °C, avec détection {median_b:.1f} °C ; "
        f"{direction}. U compare les rangs, pas les moyennes ; l'effet est une association observée, pas une cause."
    )
    return _finalize(
        result,
        status="ok",
        interpretation=interpretation,
        alpha=level,
        statistic=statistic,
        p_value=p_value,
    )


# ---------------------------------------------------------------------------
# Report assembly and loading
# ---------------------------------------------------------------------------


def build_statistical_report(
    panel: pd.DataFrame | None,
    *,
    alpha: Any = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build the four-test report from the observed fire panel, safely."""
    level, alpha_warning = _safe_alpha(alpha)
    frame = panel.copy() if isinstance(panel, pd.DataFrame) else pd.DataFrame()

    if "date" in frame:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        frame["season"] = dates.dt.month.map(MONTH_TO_SEASON)
    else:
        frame["season"] = np.nan

    dates = pd.to_datetime(frame["date"], errors="coerce").dropna() if "date" in frame else pd.Series(dtype="datetime64[ns]")
    burned = _finite_numeric(frame["burned"]) if "burned" in frame else pd.Series(dtype=float)
    cells = (
        int(frame[["lat_cell", "lon_cell"]].dropna().drop_duplicates().shape[0])
        if {"lat_cell", "lon_cell"}.issubset(frame.columns) else None
    )
    meta = WEATHER_VARIABLES.get("T2M_MAX", {"label": "T2M_MAX", "unit": "°C"})

    tests = [
        chi_square_season_detection(frame, alpha=level),
        kruskal_groups(frame, alpha=level),
        shapiro_normality(frame["T2M_MAX"] if "T2M_MAX" in frame else [],
                          alpha=level, seed=seed),
        mann_whitney_groups(frame, alpha=level),
    ]
    if alpha_warning:
        for test in tests:
            test["warning"] = alpha_warning if not test.get("warning") else f"{alpha_warning} {test['warning']}"

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "alpha": level,
            "seed": int(seed),
            "shapiro_sample_cap": SHAPIRO_SAMPLE_CAP,
            "scope": (
                "Tests descriptifs sur les observations historiques (NASA POWER + proxy FIRMS). "
                "Aucune donnée prédite ni aucune surface brûlée confirmée n'est utilisée."
            ),
        },
        "dataset": {
            "rows": int(len(frame)),
            "cells": cells,
            "date_min": dates.min().date().isoformat() if not dates.empty else None,
            "date_max": dates.max().date().isoformat() if not dates.empty else None,
            "detected_cell_days": int((burned == 1).sum()),
            "detection_rate": float(burned.mean()) if not burned.empty else None,
            "tested_variable": {"column": "T2M_MAX", "label": meta["label"], "unit": meta["unit"]},
        },
        "tests": tests,
        "limitations": list(_LIMITATIONS),
    }
    return report


def validate_statistical_report(report: Any) -> None:
    """Raise ValueError when a loaded report does not match the expected schema."""
    if not isinstance(report, dict):
        raise ValueError("Le rapport statistique doit être un objet JSON.")
    for key in ("generated_at", "protocol", "dataset", "tests", "limitations"):
        if key not in report:
            raise ValueError(f"Clé de rapport manquante : {key}")
    tests = report["tests"]
    if not isinstance(tests, list) or [t.get("test_id") for t in tests if isinstance(t, dict)] != list(EXPECTED_TEST_IDS):
        raise ValueError("Le rapport doit contenir exactement les quatre tests attendus, dans l'ordre.")
    for test in tests:
        for key in ("title", "null_hypothesis", "alternative_hypothesis", "why_appropriate",
                    "status", "statistic", "p_value", "assumptions", "interpretation"):
            if key not in test:
                raise ValueError(f"Clé manquante dans le test {test.get('test_id')} : {key}")


def load_statistical_report(path: Path | str = STATISTICAL_REPORT_PATH) -> dict[str, Any] | None:
    """Load a precomputed report, or return None when it has not been generated."""
    target = Path(path)
    if not target.exists():
        return None
    with open(target, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    validate_statistical_report(report)
    return report
