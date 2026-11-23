"""Reusable model-benchmark artifacts for FireWatch AI.

This module never trains or modifies the deployed model. It harmonises the
already-valid archived classifier experiments (one common temporal split, one
20-feature contract, seed 42), adds a cheap prior-only baseline computed on the
*same* split, and writes a compact protocol describing what may and may not be
compared.

Key protocol points surfaced in the report:

* archived untuned models used 3 chronological expanding folds with a 3-day
  purge to learn the F1 threshold from TRAIN out-of-fold predictions;
* the tuned archive used the same validation year, but fitted its threshold on
  validation, so its thresholded metrics are kept as a non-ranking reference;
* XGBoost, LightGBM and CatBoost are detected at build time and excluded with
  an explicit reason when unavailable — no placeholder metric is invented.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS_DIR = ROOT / "results"
ARCHIVED_UNTOUCHED_CSV = RESULTS_DIR / "classifier_comparison.csv"
ARCHIVED_TUNED_CSV = RESULTS_DIR / "tuned_classifier_comparison.csv"
BENCHMARK_CSV = RESULTS_DIR / "model_benchmark.csv"
BENCHMARK_PROTOCOL_JSON = RESULTS_DIR / "benchmark_protocol.json"
MODEL_CARD_PATH = ROOT / "models" / "model_card.json"

DEFAULT_SEED = 42
PRIMARY_EXPERIMENT = "primary"
TUNED_EXPERIMENT = "tuned_reference"

EXPECTED_COLUMNS: tuple[str, ...] = (
    "experiment",
    "model",
    "model_label",
    "family",
    "train_samples",
    "val_samples",
    "oof_time_sec",
    "fit_time_sec",
    "eval_time_sec",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "optimal_threshold",
    "f1_optimal",
    "precision_optimal",
    "recall_optimal",
    "f1_default_05",
    "precision_default_05",
    "recall_default_05",
    "accuracy_default_05",
    "mean_cv_pr_auc",
    "threshold_source",
    "threshold_metrics_unbiased",
    "include_in_ranking",
    "source_file",
    "notes",
)

NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    col for col in EXPECTED_COLUMNS
    if col not in {
        "experiment", "model", "model_label", "family",
        "threshold_source", "source_file", "notes",
        "threshold_metrics_unbiased", "include_in_ranking",
    }
)

MODEL_FAMILIES: dict[str, str] = {
    "DummyClassifier (prior)": "Simple baseline",
    "LogisticRegression": "Linear baseline",
    "DecisionTreeClassifier": "Tree baseline",
    "RandomForestClassifier": "Random forest",
    "HistGradientBoostingClassifier": "Gradient boosting",
    "KNeighborsClassifier": "Neighbors",
    "GaussianNB": "Probabilistic baseline",
}

MODEL_LABELS: dict[str, str] = {
    "DummyClassifier (prior)": "Dummy — taux de base",
    "LogisticRegression": "Régression logistique",
    "DecisionTreeClassifier": "Arbre de décision",
    "RandomForestClassifier": "Random Forest",
    "HistGradientBoostingClassifier": "HistGradientBoosting",
    "KNeighborsClassifier": "k plus proches voisins",
    "GaussianNB": "Gaussian Naive Bayes",
}

OPTIONAL_ALGORITHMS: dict[str, str] = {
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "CatBoost": "catboost",
}

_INTERPRETATION_NOTES = (
    "PR-AUC : aire sous la courbe précision-rappel ; la référence aléatoire vaut le taux "
    "de positifs de la validation, pas 0,5. C'est le critère principal pour un événement rare.",
    "ROC-AUC : capacité de classement toutes seuils confondus ; elle peut paraître élevée "
    "même quand la précision reste faible en cas de déséquilibre fort.",
    "Accuracy : trompeuse ici — prédire « aucun feu » pour toutes les cellules donne déjà "
    "plus de 95 % de bonnes réponses. Elle ne doit pas guider le choix du modèle.",
    "Les métriques au seuil optimal des modèles non réglés utilisent un seuil appris "
    "uniquement sur les prédictions hors-pli du train ; la validation reste extérieure.",
)
_TUNED_NOTE = (
    "Seuil F1 ajusté sur la validation : les métriques de seuil sont optimistes et "
    "exclues du classement. ROC-AUC, PR-AUC et Brier restent comparables."
)
_PRIMARY_NOTE = "Seuil F1 appris sur les prédictions hors-pli du train ; évaluation sur validation 2022."


# ---------------------------------------------------------------------------
# Loading, validation and availability
# ---------------------------------------------------------------------------


def validate_benchmark_frame(frame: pd.DataFrame) -> None:
    """Raise ValueError when a benchmark table does not match the agreed schema."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Le benchmark doit être un DataFrame.")
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Colonnes de benchmark manquantes : {', '.join(missing)}")


def load_benchmark_results(path: Path | str = BENCHMARK_CSV) -> pd.DataFrame:
    """Load the generated benchmark, or return a correctly typed empty frame."""
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=list(EXPECTED_COLUMNS))
    frame = pd.read_csv(target)
    validate_benchmark_frame(frame)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("threshold_metrics_unbiased", "include_in_ranking"):
        frame[column] = frame[column].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    return frame


def load_benchmark_protocol(path: Path | str = BENCHMARK_PROTOCOL_JSON) -> dict[str, Any]:
    """Load the generated protocol, with a safe default when it is absent."""
    target = Path(path)
    if not target.exists():
        return {
            "generated_at": None,
            "seed": DEFAULT_SEED,
            "model_contract": {"n_features": 20, "artifacts_modified": False},
            "split": {},
            "class_balance": {},
            "algorithms": {"included": [], "excluded": optional_algorithm_availability()},
            "interpretation_notes": list(_INTERPRETATION_NOTES),
            "missing": True,
        }
    with open(target, "r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if not isinstance(protocol, dict):
        raise ValueError("Le protocole de benchmark doit être un objet JSON.")
    return protocol


def optional_algorithm_availability() -> dict[str, dict[str, Any]]:
    """Report optional boosters without importing or installing them."""
    results: dict[str, dict[str, Any]] = {}
    for label, module_name in OPTIONAL_ALGORITHMS.items():
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            available = False
        results[label] = {
            "available": bool(available),
            "module": module_name,
            "status": "installed" if available else "not installed",
            "reason": (
                "Dépendance compatible détectée ; non exécutée ici pour ne pas mélanger "
                "un nouvel entraînement avec les résultats archivés."
                if available else
                "Dépendance absente de l'environnement et non épinglée dans requirements.txt : "
                "aucun résultat n'est rapporté."
            ),
        }
    return results


def load_production_feature_contract(path: Path | str = MODEL_CARD_PATH) -> list[str]:
    """Read the protected 20-feature contract from the model card."""
    with open(path, "r", encoding="utf-8") as handle:
        card = json.load(handle)
    features = card.get("feature_names")
    if not isinstance(features, list) or not features:
        raise ValueError("La carte du modèle ne contient pas de liste « feature_names ».")
    return [str(name) for name in features]


# ---------------------------------------------------------------------------
# Harmonisation helpers
# ---------------------------------------------------------------------------


def _complete_record(
    record: Mapping[str, Any],
    *,
    experiment: str,
    threshold_source: str,
    threshold_metrics_unbiased: bool,
    include_in_ranking: bool,
    source_file: str,
    notes: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {column: np.nan for column in EXPECTED_COLUMNS}
    row.update({key: value for key, value in record.items() if key in EXPECTED_COLUMNS})
    row["experiment"] = experiment
    row["threshold_source"] = threshold_source
    row["threshold_metrics_unbiased"] = bool(threshold_metrics_unbiased)
    row["include_in_ranking"] = bool(include_in_ranking)
    row["source_file"] = source_file
    row["notes"] = notes
    model = str(row.get("model", ""))
    row["model_label"] = MODEL_LABELS.get(model, model)
    row["family"] = MODEL_FAMILIES.get(model, "Autre")
    for column in NUMERIC_COLUMNS:
        row[column] = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return row


def normalize_comparison_frame(
    frame: pd.DataFrame,
    *,
    experiment: str,
    threshold_source: str,
    threshold_metrics_unbiased: bool,
    include_in_ranking: bool,
    source_file: str,
    notes: str,
) -> pd.DataFrame:
    """Convert an archived comparison CSV into the common benchmark schema."""
    if not isinstance(frame, pd.DataFrame) or "model" not in frame.columns:
        raise ValueError("Le CSV archivé doit contenir une colonne « model ».")
    required_metrics = {"roc_auc", "pr_auc", "brier_score", "optimal_threshold"}
    missing = [column for column in required_metrics if column not in frame.columns]
    if missing:
        raise ValueError(f"Métriques archivées manquantes : {', '.join(missing)}")

    rows = [
        _complete_record(
            record,
            experiment=experiment,
            threshold_source=threshold_source,
            threshold_metrics_unbiased=threshold_metrics_unbiased,
            include_in_ranking=include_in_ranking,
            source_file=source_file,
            notes=notes,
        )
        for record in frame.to_dict(orient="records")
        if str(record.get("model", "")).strip()
    ]
    result = pd.DataFrame(rows, columns=list(EXPECTED_COLUMNS))
    validate_benchmark_frame(result)
    return result


# ---------------------------------------------------------------------------
# Baseline computation on the identical split
# ---------------------------------------------------------------------------


def build_prior_baseline_record(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Train a DummyClassifier(prior) on the same train/validation split.

    This is deliberately cheap: it estimates the positive class rate from the
    training period only, and is scored once on the untouched validation year.
    """
    from sklearn.dummy import DummyClassifier

    from compare_classifiers import build_date_folds, fit_and_evaluate_classifier
    from train_baseline import LABEL

    folds = build_date_folds(train_df, n_splits=3, purge_days=3)
    model = DummyClassifier(strategy="prior")
    return fit_and_evaluate_classifier(
        name="DummyClassifier (prior)",
        model=model,
        train_df=train_df,
        val_df=val_df,
        features=features,
        label=LABEL,
        folds=folds,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------


def _atomic_write_df(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f"{target.suffix}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, target)


def _atomic_write_json(data: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f"{target.suffix}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, target)


def build_benchmark_artifacts(
    *,
    output_dir: Path | str = RESULTS_DIR,
    archived_untuned_path: Path | str = ARCHIVED_UNTOUCHED_CSV,
    archived_tuned_path: Path | str = ARCHIVED_TUNED_CSV,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    features: list[str] | None = None,
    baseline_record: Mapping[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild the benchmark CSV and protocol from valid, documented sources."""
    from train_baseline import LABEL, load_data, time_split

    production_features = load_production_feature_contract()
    if features is None:
        features = production_features
    if [str(name) for name in features] != production_features:
        raise ValueError("Le benchmark doit utiliser exactement les 20 features de production.")

    if train_df is None or val_df is None:
        panel, panel_features = load_data()
        if [str(name) for name in panel_features] != production_features:
            raise ValueError("Le jeu de départ ne respecte pas le contrat 20 features.")
        train_df, val_df, _ = time_split(panel)
    if train_df is None or val_df is None:  # pragma: no cover - defensive
        raise ValueError("Les jeux train/validation sont requis pour le benchmark.")

    untuned = pd.read_csv(archived_untuned_path)
    tuned = pd.read_csv(archived_tuned_path)

    primary = normalize_comparison_frame(
        untuned,
        experiment=PRIMARY_EXPERIMENT,
        threshold_source="train OOF (3 plis temporels purgés)",
        threshold_metrics_unbiased=True,
        include_in_ranking=True,
        source_file=Path(archived_untuned_path).name,
        notes=_PRIMARY_NOTE,
    )
    tuned_rows = normalize_comparison_frame(
        tuned,
        experiment=TUNED_EXPERIMENT,
        threshold_source="validation (2022) — optimiste",
        threshold_metrics_unbiased=False,
        include_in_ranking=False,
        source_file=Path(archived_tuned_path).name,
        notes=_TUNED_NOTE,
    )
    if baseline_record is None:
        baseline_record = build_prior_baseline_record(train_df, val_df, features, seed=seed)
    baseline = normalize_comparison_frame(
        pd.DataFrame([baseline_record]),
        experiment=PRIMARY_EXPERIMENT,
        threshold_source="train OOF (3 plis temporels purgés)",
        threshold_metrics_unbiased=True,
        include_in_ranking=True,
        source_file="computed: DummyClassifier(strategy='prior')",
        notes=_PRIMARY_NOTE,
    )

    combined = pd.concat([baseline, primary, tuned_rows], ignore_index=True)
    experiment_order = {PRIMARY_EXPERIMENT: 0, TUNED_EXPERIMENT: 1}
    combined["_order"] = combined["experiment"].map(experiment_order).fillna(9)
    combined = (
        combined.sort_values(["_order", "pr_auc"], ascending=[True, False], kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    validate_benchmark_frame(combined)

    train_labels = pd.to_numeric(train_df[LABEL], errors="coerce").dropna()
    val_labels = pd.to_numeric(val_df[LABEL], errors="coerce").dropna()
    val_dates = pd.to_datetime(val_df["date"], errors="coerce").dropna()
    train_dates = pd.to_datetime(train_df["date"], errors="coerce").dropna()

    protocol: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "model_contract": {
            "n_features": len(features),
            "features": [str(name) for name in features],
            "artifacts_modified": False,
            "note": "models/fire_model.joblib et models/model_card.json n'ont pas été modifiés.",
        },
        "split": {
            "strategy": "temporal, cell-day atomic, no random split",
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "train_date_min": train_dates.min().date().isoformat() if not train_dates.empty else None,
            "train_date_max": train_dates.max().date().isoformat() if not train_dates.empty else None,
            "val_date_min": val_dates.min().date().isoformat() if not val_dates.empty else None,
            "val_date_max": val_dates.max().date().isoformat() if not val_dates.empty else None,
            "purge_days": 3,
            "oof_folds": "3 chronological expanding date blocks; first block is warm-up",
            "test_set": "2023–2024 untouched by these experiments",
        },
        "class_balance": {
            "train_positive_rate": float(train_labels.mean()) if not train_labels.empty else None,
            "val_positive_rate": float(val_labels.mean()) if not val_labels.empty else None,
            "val_positive_rows": int(val_labels.sum()) if not val_labels.empty else 0,
        },
        "preprocessing": (
            "Pipeline identique par famille : imputation médiane et standardisation uniquement "
            "lorsque l'estimateur l'exige ; HistGradientBoosting reçoit les NaN nativement."
        ),
        "threshold_policy": {
            "primary": "F1 threshold learned from train out-of-fold predictions only",
            "tuned_reference": "F1 threshold fitted on validation; thresholded metrics excluded from ranking",
        },
        "algorithms": {
            "included": [
                "DummyClassifier (prior) — simple baseline",
                "LogisticRegression — linear baseline",
                "DecisionTreeClassifier — tree baseline",
                "RandomForestClassifier — bagged trees",
                "HistGradientBoostingClassifier — gradient boosting",
                "KNeighborsClassifier — neighbors",
                "GaussianNB — probabilistic baseline",
            ],
            "excluded": optional_algorithm_availability(),
        },
        "archived_sources": [
            Path(archived_untuned_path).name,
            Path(archived_tuned_path).name,
            "computed: DummyClassifier(strategy='prior') on the identical split",
        ],
        "interpretation_notes": list(_INTERPRETATION_NOTES),
    }

    output = Path(output_dir)
    _atomic_write_df(combined, output / BENCHMARK_CSV.name)
    _atomic_write_json(protocol, output / BENCHMARK_PROTOCOL_JSON.name)
    return combined, protocol
