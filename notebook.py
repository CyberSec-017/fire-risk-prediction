#!/usr/bin/env python
"""
FIREWATCH AI — Reproducible evaluation notebook (``notebook.py``)
=================================================================

This script is the executable "notebook" for the project.  It loads the
deployed model artifact and the processed fire panel, reproduces the
chronological evaluation exactly as in ``src/train_baseline.py``, prints a
complete statistical report to the console, and exports publication-ready
figures plus a machine-readable summary.

Why a script instead of a .ipynb?
---------------------------------
It runs headless in CI, it is diff-able in Git, and every number in the PDF
report can be regenerated with one command:

    python notebook.py

Outputs
-------
    report/figures/*.png               ten evaluation figures
    report/evaluation_summary.json     all metrics, tables and provenance
    report/EVALUATION.md               Markdown version of the console report

Options
-------
    --outdir report        output directory
    --figure-dpi 150       PNG resolution
    --sample 8000          rows used for permutation importance
    --no-figures           skip plotting (fast metric-only run)
    --seed 42              random seed for the importance sample
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Windows consoles default to cp1252, which cannot encode the box-drawing
# characters used by this report.  Never let formatting crash an evaluation.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_core  # noqa: E402
from dashboard.config import FIRE_PANEL_PATH  # noqa: E402
from dashboard.predictor import DISPLAY_LABELS  # noqa: E402


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------
def rule(title: str = "", width: int = 78) -> None:
    if title:
        print(f"\n{'─' * width}\n{title}\n{'─' * width}")
    else:
        print("─" * width)


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas objects into JSON-safe values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Data + model
# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    if not FIRE_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"{FIRE_PANEL_PATH} is missing. Run the pipeline first:\n"
            "  python src/fetch_power.py && python src/clean_firms.py "
            "&& python src/build_panel.py"
        )
    frame = pd.read_parquet(FIRE_PANEL_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    if "target" not in frame.columns:
        raise ValueError("fire_panel.parquet has no 'target' column.")
    # Matches train_baseline.load_data(): rows without usable weather are dropped.
    frame = frame.dropna(subset=["T2M"]).copy()
    return frame.sort_values(["date", "lat_cell", "lon_cell"]).reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame,
    *,
    val_year: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_start = pd.Timestamp(f"{val_year}-01-01")
    test_start = pd.Timestamp(f"{val_year + 1}-01-01")
    train = frame[frame["date"] < val_start - pd.Timedelta(days=horizon)]
    val = frame[
        (frame["date"] >= val_start)
        & (frame["date"] < test_start - pd.Timedelta(days=horizon))
    ]
    test = frame[frame["date"] >= test_start]
    return train, val, test


def raw_estimator(model: Any) -> Any:
    """Return the uncalibrated estimator inside a calibrated classifier."""
    try:
        inner = model.calibrated_classifiers_[0]
    except AttributeError:
        return model
    estimator = getattr(inner, "estimator", None)
    if estimator is not None:
        return getattr(estimator, "estimator", estimator)
    return getattr(inner, "base_estimator", model)


def climatology_probability(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Per-cell, per-month historical fire rate fitted on train only."""
    climatology = (
        train.assign(month=train["date"].dt.month)
        .groupby(["lat_cell", "lon_cell", "month"])["target"]
        .mean()
        .rename("climatology")
    )
    joined = test.assign(month=test["date"].dt.month).join(
        climatology, on=["lat_cell", "lon_cell", "month"]
    )
    return joined["climatology"].fillna(0.0).to_numpy(dtype=float)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    frame = pd.DataFrame({"y": y_true, "p": probabilities})
    grouped = (
        frame.groupby(pd.qcut(frame["p"], bins, duplicates="drop"), observed=True)
        .agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    )
    return float(
        (grouped["n"] * (grouped["observed"] - grouped["predicted"]).abs()).sum()
        / grouped["n"].sum()
    )


def calibration_slope(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    regression = LogisticRegression(C=np.inf).fit(logit, y_true)
    return float(regression.coef_[0][0]), float(regression.intercept_[0])


def risk_band_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: list[float],
    labels: list[str],
) -> pd.DataFrame:
    bands = pd.cut(probabilities, bins=bins, labels=labels, include_lowest=True)
    frame = pd.DataFrame({"band": bands, "y": y_true, "p": probabilities})
    total_positives = max(int(y_true.sum()), 1)
    table = (
        frame.groupby("band", observed=True)
        .agg(
            cells=("y", "size"),
            predicted_rate=("p", "mean"),
            observed_rate=("y", "mean"),
            positives=("y", "sum"),
        )
        .reset_index()
    )
    table["share_of_all_fires"] = table["positives"] / total_positives
    table["mean_score"] = table["predicted_rate"]
    return table[
        ["band", "cells", "mean_score", "observed_rate", "positives", "share_of_all_fires"]
    ]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "grid.color": "#EEF2F7",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )
    return plt


PALETTE = {
    "ember": "#EA580C",
    "deep": "#7C2D12",
    "risk": "#DC2626",
    "green": "#15803D",
    "slate": "#475569",
    "muted": "#94A3B8",
    "blue": "#0369A1",
    "amber": "#D97706",
}


def save_figures(
    *,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    y_test: np.ndarray,
    probabilities: np.ndarray,
    raw_probabilities: np.ndarray,
    climatology: np.ndarray,
    climatology_calibrated: np.ndarray,
    risk_table: pd.DataFrame,
    importance: pd.DataFrame,
    yearly: pd.DataFrame,
    outdir: Path,
    dpi: int,
) -> dict[str, str]:
    plt = _pyplot()
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def finish(fig, name: str) -> None:
        path = figdir / name
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written[name] = str(path)

    # 1 — seasonality -------------------------------------------------------
    monthly = (
        test.assign(month=test["date"].dt.month)
        .groupby("month")
        .agg(fires=("target", "sum"), rate=("target", "mean"))
        .reindex(range(1, 13), fill_value=0)
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].bar(monthly.index, monthly["fires"], color=PALETTE["ember"])
    axes[0].set_title("Observed fire cell-days by month (test period)")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Fire cell-days")
    axes[0].set_xticks(range(1, 13))
    axes[1].plot(monthly.index, monthly["rate"] * 100, marker="o", color=PALETTE["risk"])
    axes[1].fill_between(monthly.index, monthly["rate"] * 100, color=PALETTE["risk"], alpha=0.12)
    axes[1].set_title("Fire rate by month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Fire rate (%)")
    axes[1].set_xticks(range(1, 13))
    finish(fig, "01_fire_seasonality.png")

    # 2 — precision-recall + cumulative capture -----------------------------
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    precision_c, recall_c, _ = precision_recall_curve(y_test, climatology)
    ap_model = average_precision_score(y_test, probabilities)
    ap_clim = average_precision_score(y_test, climatology)

    order = np.argsort(probabilities)[::-1]
    sorted_y = y_test[order]
    cumulative = np.cumsum(sorted_y) / max(sorted_y.sum(), 1)
    population = np.arange(1, len(sorted_y) + 1) / len(sorted_y)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].plot(recall, precision, color=PALETTE["ember"], lw=2, label=f"Model (AP={ap_model:.3f})")
    axes[0].plot(
        recall_c,
        precision_c,
        color=PALETTE["slate"],
        lw=1.5,
        ls="--",
        label=f"Climatology (AP={ap_clim:.3f})",
    )
    axes[0].axhline(y_test.mean(), color=PALETTE["muted"], ls=":", label="Base rate")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision–Recall curve")
    axes[0].legend(loc="upper right")

    axes[1].plot(population * 100, cumulative * 100, color=PALETTE["deep"], lw=2, label="Model")
    axes[1].plot([0, 100], [0, 100], color=PALETTE["muted"], ls=":", label="Random")
    axes[1].axvline(10, color=PALETTE["amber"], ls="--", lw=1)
    top_share = cumulative[int(0.1 * len(cumulative)) - 1] * 100
    axes[1].annotate(
        f"Top 10% of cells\n→ {top_share:.1f}% of fires",
        xy=(10, top_share),
        xytext=(24, max(8, top_share - 28)),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=PALETTE["amber"]),
    )
    axes[1].set_xlabel("% of scored cell-days (highest risk first)")
    axes[1].set_ylabel("% of observed fires captured")
    axes[1].set_title("Cumulative fire capture")
    axes[1].legend(loc="lower right")
    finish(fig, "02_precision_recall_capture.png")

    # 3 — ROC ---------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(fpr, tpr, color=PALETTE["ember"], lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], ls=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test period)")
    ax.legend(loc="lower right")
    finish(fig, "03_roc_curve.png")

    # 4 — calibration -------------------------------------------------------
    frac_pos, mean_pred = calibration_curve(y_test, probabilities, n_bins=12, strategy="quantile")
    ece = expected_calibration_error(y_test, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].plot([0, max(mean_pred.max(), 0.01)], [0, max(mean_pred.max(), 0.01)], color=PALETTE["muted"], ls=":")
    axes[0].plot(mean_pred, frac_pos, marker="o", color=PALETTE["ember"], lw=1.8)
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed fire rate")
    axes[0].set_title(f"Reliability curve (ECE = {ece:.4f})")
    axes[1].hist(probabilities[y_test == 0], bins=60, color=PALETTE["muted"], alpha=0.75, label="No fire")
    axes[1].hist(probabilities[y_test == 1], bins=60, color=PALETTE["risk"], alpha=0.8, label="Fire")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Cell-days (log scale)")
    axes[1].set_title("Score distribution by observed outcome")
    axes[1].legend()
    finish(fig, "04_calibration.png")

    # 5 — risk bands --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    labels = [str(value) for value in risk_table["band"]]
    observed = risk_table["observed_rate"] * 100
    predicted = risk_table["mean_score"] * 100
    positions = np.arange(len(labels))
    ax.bar(positions - 0.2, observed, width=0.38, color=PALETTE["risk"], label="Observed fire rate")
    ax.bar(positions + 0.2, predicted, width=0.38, color=PALETTE["slate"], label="Mean predicted score")
    for index, value in enumerate(observed):
        ax.annotate(f"{value:.1f}%", (index - 0.2, value), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Risk bands: observed vs. predicted")
    ax.legend()
    finish(fig, "05_risk_bands.png")

    # 6 — yearly performance ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    years = [str(value) for value in yearly["year"]]
    xpos = np.arange(len(years))
    axes[0].bar(xpos - 0.2, yearly["pr_auc_model"], width=0.38, color=PALETTE["ember"], label="Model")
    axes[0].bar(xpos + 0.2, yearly["pr_auc_climatology"], width=0.38, color=PALETTE["slate"], label="Climatology")
    axes[0].set_xticks(xpos, years)
    axes[0].set_ylabel("PR-AUC")
    axes[0].set_title("PR-AUC by test year")
    axes[0].legend()
    axes[1].bar(xpos - 0.2, yearly["base_rate"] * 100, width=0.38, color=PALETTE["risk"], label="Base rate")
    axes[1].bar(xpos + 0.2, yearly["top_decile_share"] * 100, width=0.38, color=PALETTE["green"], label="Fires in top decile")
    axes[1].set_xticks(xpos, years)
    axes[1].set_ylabel("%")
    axes[1].set_title("Base rate vs. top-decile fire capture")
    axes[1].legend()
    finish(fig, "06_yearly_performance.png")

    # 7 — permutation importance -------------------------------------------
    top_importance = importance.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = [PALETTE["ember"] if value >= 0 else PALETTE["muted"] for value in top_importance["importance_mean"]]
    ax.barh(top_importance["feature"], top_importance["importance_mean"], xerr=top_importance["importance_std"], color=colors)
    ax.set_xlabel("Increase in Brier score when shuffled (higher = more important)")
    ax.set_title("Permutation importance (test sample)")
    finish(fig, "07_feature_importance.png")

    # 8 — confusion matrix --------------------------------------------------
    threshold = 0.5
    matrix = confusion_matrix(y_test, (probabilities >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    image = ax.imshow(matrix, cmap="Oranges")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, f"{value:,}", ha="center", va="center", fontsize=11,
                color="white" if value > matrix.max() / 2 else "#1F2937")
    ax.set_xticks([0, 1], ["No fire", "Fire"])
    ax.set_yticks([0, 1], ["No fire", "Fire"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(f"Confusion matrix @ p ≥ {threshold:.2f}")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    finish(fig, "08_confusion_matrix.png")

    # 9 — hotspot map -------------------------------------------------------
    hotspot = (
        test.assign(prediction=probabilities)
        .groupby(["lat_cell", "lon_cell"])
        .agg(fires=("target", "sum"), mean_risk=("prediction", "mean"))
        .reset_index()
    )
    hotspot = hotspot[hotspot["fires"] > 0].nlargest(220, "fires")
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    scatter = ax.scatter(
        hotspot["lon_cell"],
        hotspot["lat_cell"],
        s=18 + hotspot["fires"] * 14,
        c=hotspot["mean_risk"],
        cmap="YlOrRd",
        edgecolor="#7C2D12",
        linewidth=0.4,
        alpha=0.9,
    )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Observed fire hotspots and mean predicted risk (2022–2024)")
    fig.colorbar(scatter, ax=ax, label="Mean predicted probability")
    finish(fig, "09_hotspots.png")

    # 10 — data overview ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(train["T2M_MAX"].dropna(), bins=50, color=PALETTE["amber"], alpha=0.8, label="Train")
    axes[0].hist(test["T2M_MAX"].dropna(), bins=50, color=PALETTE["deep"], alpha=0.55, label="Test")
    axes[0].set_title("Daily maximum temperature distribution")
    axes[0].set_xlabel("°C")
    axes[0].set_ylabel("Cell-days")
    axes[0].legend()
    rain = train.groupby("date")["PRECTOTCORR"].mean()
    axes[1].plot(rain.index, rain.rolling(30, min_periods=1).mean(), color=PALETTE["blue"])
    axes[1].set_title("Spatial mean rainfall (30-day rolling)")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("mm/day")
    finish(fig, "10_data_overview.png")

    return written


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  FIREWATCH AI — model evaluation notebook")
    print("  Wildfire risk prediction · northern Algeria · horizon H = 3 days")
    print("=" * 78)
    print(f"  Python      : {platform.python_version()}")
    print(f"  pandas      : {pd.__version__}")
    print(f"  numpy       : {np.__version__}")
    print(f"  generated   : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")

    rule("1. Artifacts and data")
    bundle = app_core.load_model_bundle()
    if bundle.status != "REAL":
        raise RuntimeError(f"Model is not usable: {bundle.status} — {bundle.error}")
    card = bundle.card or {}
    model = bundle.model
    features = list(card["feature_names"])
    raw_model = raw_estimator(model)
    horizon = int(card.get("metrics", {}).get("horizon_days", 3) or 3)

    panel = load_panel()
    val_year = int(card.get("val_year", app_core.MODEL_CALIBRATION_YEAR))
    train, val, test = chronological_split(panel, val_year=val_year, horizon=horizon)

    print(f"  model        : v{card.get('model_version')} ({bundle.model_path.name})")
    print(f"  features     : {len(features)}")
    print(f"  panel        : {len(panel):,} rows · {panel['date'].min():%Y-%m-%d} → {panel['date'].max():%Y-%m-%d}")
    print(f"  cells        : {panel.groupby(['lat_cell', 'lon_cell']).ngroups:,}")
    print(f"  split        : train {len(train):,} (< {val_year}) · val {len(val):,} ({val_year}) · test {len(test):,} (≥ {val_year + 1})")
    print(f"  base rate    : train {pct(train['target'].mean())} · val {pct(val['target'].mean())} · test {pct(test['target'].mean())}")

    if test.empty:
        raise RuntimeError("The test split is empty; check the panel date range.")

    rule("2. Scoring the test period")
    y_test = test["target"].to_numpy(dtype=int)
    calibrated = np.clip(model.predict_proba(test[features])[:, 1], 0, 1)
    try:
        raw = np.clip(raw_model.predict_proba(test[features])[:, 1], 0, 1)
    except Exception:  # noqa: BLE001 - some artifacts expose only the calibrator
        raw = calibrated
    climatology = climatology_probability(train, test)

    # Calibrated climatology is the fair Brier baseline used by the trainer.
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(climatology_probability(train, val), val["target"])
    climatology_cal = iso.predict(climatology)

    print(f"  scored       : {len(test):,} cell-days")
    print(f"  positives    : {int(y_test.sum()):,} ({pct(y_test.mean())})")

    rule("3. Headline metrics (test period)")
    base_rate = float(y_test.mean())
    pr_auc = float(average_precision_score(y_test, calibrated))
    pr_auc_raw = float(average_precision_score(y_test, raw))
    roc_auc = float(roc_auc_score(y_test, calibrated))
    brier = float(brier_score_loss(y_test, calibrated))
    brier_skill = float(1 - brier / (base_rate * (1 - base_rate)))
    ece = expected_calibration_error(y_test, calibrated)
    slope, intercept = calibration_slope(y_test, calibrated)

    order = np.argsort(raw)[::-1]
    top_count = max(int(0.1 * len(raw)), 1)
    top_share = float(y_test[order[:top_count]].sum() / max(y_test.sum(), 1))

    clim_pr_auc = float(average_precision_score(y_test, climatology))
    clim_brier = float(brier_score_loss(y_test, climatology))
    clim_brier_cal = float(brier_score_loss(y_test, climatology_cal))
    gain_vs_clim = float(1 - brier / clim_brier_cal)

    threshold = float(0.5)
    best_threshold = 0.5
    best_f1 = -1.0
    for candidate in np.linspace(0.005, 0.5, 100):
        score = f1_score(y_test, (calibrated >= candidate).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = float(score), float(candidate)
    predictions = (calibrated >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    precision_at_05 = tp / max(tp + fp, 1)
    recall_at_05 = tp / max(tp + fn, 1)

    print(f"  PR-AUC                 {pr_auc:.4f}   (raw uncalibrated {pr_auc_raw:.4f}; random = {base_rate:.4f})")
    print(f"  ROC-AUC                {roc_auc:.4f}")
    print(f"  Brier score            {brier:.5f}   (lower is better)")
    print(f"  Brier skill score      {brier_skill:+.4f}   (0 = base-rate forecast)")
    print(f"  ECE (10 quantile bins) {ece:.5f}")
    print(f"  Calibration slope      {slope:.3f}   (1.0 = perfect) · intercept {intercept:+.3f}")
    print(f"  Fires in top 10%       {pct(top_share)}   ({top_count:,} highest-risk cell-days)")
    print(f"  Climatology PR-AUC     {clim_pr_auc:.4f}   (model gain {pr_auc - clim_pr_auc:+.4f})")
    print(f"  Brier vs calibrated climatology: {gain_vs_clim:+.2%}")
    print(f"  Decision threshold     {threshold:.2f} → precision {precision_at_05:.3f} · recall {recall_at_05:.3f}")
    print(f"  Best F1 threshold      {best_threshold:.3f} → F1 {best_f1:.3f}")

    rule("4. Confusion matrix @ p ≥ 0.5")
    print(f"  True negatives  {tn:>9,}   False positives {fp:>8,}")
    print(f"  False negatives {fn:>9,}   True positives  {tp:>8,}")

    rule("5. Risk bands (model card contract)")
    risk_table = risk_band_table(
        y_test, calibrated, card["risk_bins"], card["risk_labels"]
    )
    display = risk_table.copy()
    display["band"] = display["band"].map(
        lambda label: DISPLAY_LABELS.get(str(label).upper(), str(label))
    )
    display["mean_score"] = display["mean_score"].map(lambda v: f"{v * 100:.2f}%")
    display["observed_rate"] = display["observed_rate"].map(lambda v: f"{v * 100:.2f}%")
    display["share_of_all_fires"] = display["share_of_all_fires"].map(lambda v: f"{v * 100:.1f}%")
    print(display.to_string(index=False))

    rule("6. Yearly performance")
    yearly_rows: list[dict[str, Any]] = []
    for year, group in test.groupby(test["date"].dt.year):
        ys = group["target"].to_numpy(dtype=int)
        index = group.index.to_numpy()
        position = test.index.get_indexer(index)
        scores = calibrated[position]
        clim_scores = climatology[position]
        top = max(int(0.1 * len(scores)), 1)
        ranking = np.argsort(scores)[::-1][:top]
        yearly_rows.append(
            {
                "year": int(year),
                "rows": int(len(group)),
                "base_rate": float(ys.mean()),
                "pr_auc_model": float(average_precision_score(ys, scores)),
                "pr_auc_climatology": float(average_precision_score(ys, clim_scores)),
                "brier": float(brier_score_loss(ys, scores)),
                "top_decile_share": float(ys[ranking].sum() / max(ys.sum(), 1)),
            }
        )
    yearly = pd.DataFrame(yearly_rows)
    print(
        yearly.assign(
            base_rate=yearly["base_rate"].map(lambda v: f"{v * 100:.2f}%"),
            top_decile_share=yearly["top_decile_share"].map(lambda v: f"{v * 100:.1f}%"),
        ).round(4).to_string(index=False)
    )

    rule("7. Permutation importance (local, sampled)")
    sample_size = min(int(args.sample), len(test))
    rng = np.random.default_rng(args.seed)
    sample_positions = rng.choice(len(test), size=sample_size, replace=False)
    sample = test.iloc[sample_positions]
    importance = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    try:
        result = permutation_importance(
            model,
            sample[features],
            sample["target"],
            scoring="neg_brier_score",
            n_repeats=3,
            random_state=args.seed,
            n_jobs=1,
        )
        importance = pd.DataFrame(
            {
                "feature": features,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        print(
            importance.head(12).assign(
                importance_mean=importance.head(12)["importance_mean"].round(5),
                importance_std=importance.head(12)["importance_std"].round(5),
            ).to_string(index=False)
        )
    except Exception as exc:  # noqa: BLE001 - never fail the report on this
        print(f"  skipped: {exc}")

    rule("8. Dataset statistics")
    stats = {
        "rows": int(len(panel)),
        "cells": int(panel.groupby(["lat_cell", "lon_cell"]).ngroups),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "positive_rate": float(panel["target"].mean()),
        "positive_cell_days": int(panel["target"].sum()),
    }
    for key, value in stats.items():
        print(f"  {key:<18} {value}")

    if not args.no_figures:
        rule("9. Figures")
        written = save_figures(
            train=train,
            val=val,
            test=test,
            y_test=y_test,
            probabilities=calibrated,
            raw_probabilities=raw,
            climatology=climatology,
            climatology_calibrated=climatology_cal,
            risk_table=risk_table,
            importance=importance,
            yearly=yearly,
            outdir=outdir,
            dpi=args.figure_dpi,
        )
        for name, path in written.items():
            print(f"  {name:<34} {path}")

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": "FIREWATCH AI — Wildfire Risk Prediction",
        "study_area": "Northern Algeria (32.0–37.5°N, -2.0–9.0°E)",
        "model": {
            "version": card.get("model_version"),
            "path": str(bundle.model_path),
            "n_features": len(features),
            "features": features,
            "risk_bins": card["risk_bins"],
            "risk_labels": card["risk_labels"],
            "recorded_metrics": card.get("metrics", {}),
        },
        "split": {
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "val_year": val_year,
            "test_years": sorted({int(year) for year in test["date"].dt.year.unique()}),
            "horizon_days": horizon,
        },
        "dataset": stats,
        "metrics": {
            "pr_auc": pr_auc,
            "pr_auc_raw": pr_auc_raw,
            "roc_auc": roc_auc,
            "base_rate": base_rate,
            "brier": brier,
            "brier_skill": brier_skill,
            "ece": ece,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "top_decile_fire_share": top_share,
            "climatology_pr_auc": clim_pr_auc,
            "climatology_brier": clim_brier,
            "climatology_calibrated_brier": clim_brier_cal,
            "brier_gain_vs_climatology": gain_vs_clim,
            "precision_at_0_5": precision_at_05,
            "recall_at_0_5": recall_at_05,
            "f1_at_0_5": float(f1_score(y_test, predictions, zero_division=0)),
            "best_f1": best_f1,
            "best_f1_threshold": best_threshold,
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "risk_bands": _jsonable(risk_table.to_dict(orient="records")),
        "yearly": _jsonable(yearly.to_dict(orient="records")),
        "permutation_importance": _jsonable(importance.to_dict(orient="records")),
    }

    summary_path = outdir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  summary written to {summary_path}")

    markdown_path = outdir / "EVALUATION.md"
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"  markdown report  {markdown_path}")

    rule("10. Interpretation")
    print(
        "  Fires are rare, so accuracy is not a useful metric. The model is judged\n"
        "  on Precision-Recall AUC, calibration (Brier / ECE) and how many fires\n"
        "  land in the highest-risk decile. The geometric baseline (per-cell,\n"
        "  per-month climatology) shows how much of the signal comes from weather\n"
        "  rather than location and season alone."
    )
    print("=" * 78)
    return summary


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def render_markdown(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    split = summary["split"]
    dataset = summary["dataset"]
    lines = [
        "# FIREWATCH AI — Evaluation Report",
        "",
        f"*Generated by `notebook.py` on {summary['generated_at']}.*",
        "",
        "## Model",
        "",
        f"- Version: **{summary['model']['version']}**",
        f"- Features: **{summary['model']['n_features']}**",
        f"- Horizon: **{split['horizon_days']} days**",
        f"- Split: train **{split['train_rows']:,}** / validation **{split['val_rows']:,}** (calibration year {split['val_year']}) / test **{split['test_rows']:,}**",
        "",
        "## Headline metrics (test period)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| PR-AUC | {metrics['pr_auc']:.4f} |",
        f"| Base fire rate | {metrics['base_rate'] * 100:.2f}% |",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
        f"| Brier score | {metrics['brier']:.5f} |",
        f"| Brier skill score | {metrics['brier_skill']:+.4f} |",
        f"| Expected calibration error | {metrics['ece']:.5f} |",
        f"| Fires in top decile | {metrics['top_decile_fire_share'] * 100:.1f}% |",
        f"| Climatology PR-AUC | {metrics['climatology_pr_auc']:.4f} |",
        f"| Brier gain vs. climatology | {metrics['brier_gain_vs_climatology'] * 100:+.2f}% |",
        "",
        "## Risk bands",
        "",
        "| Band | Cell-days | Mean score | Observed rate | Share of all fires |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["risk_bands"]:
        lines.append(
            f"| {row['band']} | {row['cells']:,} | {row['mean_score'] * 100:.2f}% | "
            f"{row['observed_rate'] * 100:.2f}% | {row['share_of_all_fires'] * 100:.1f}% |"
        )

    lines += [
        "",
        "## Yearly performance",
        "",
        "| Year | Base rate | PR-AUC (model) | PR-AUC (climatology) | Brier | Fires in top decile |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["yearly"]:
        lines.append(
            f"| {row['year']} | {row['base_rate'] * 100:.2f}% | {row['pr_auc_model']:.4f} | "
            f"{row['pr_auc_climatology']:.4f} | {row['brier']:.5f} | "
            f"{row['top_decile_share'] * 100:.1f}% |"
        )

    lines += [
        "",
        "## Dataset",
        "",
        f"- Rows: **{dataset['rows']:,}**",
        f"- Grid cells: **{dataset['cells']:,}**",
        f"- Period: **{dataset['date_min']} → {dataset['date_max']}**",
        f"- Positive cell-days: **{dataset['positive_cell_days']:,}** ({dataset['positive_rate'] * 100:.2f}%)",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python notebook.py",
        "```",
        "",
        "Figures are written to `report/figures/`, the machine-readable summary to",
        "`report/evaluation_summary.json`.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="report", help="output directory")
    parser.add_argument("--figure-dpi", type=int, default=150)
    parser.add_argument("--sample", type=int, default=8000)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
