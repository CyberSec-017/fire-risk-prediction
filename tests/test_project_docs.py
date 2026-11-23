"""Checks for the reproducibility and documentation deliverables."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_SHA256 = "d816d2e211cefe06c89512a0bba7dcf4215ab11885f8a84dcd3a3becdf01b2ab"
MODEL_CARD_SHA256 = "8b549faae5ab93717680278c13c7288adbaebf7b94444fca655724c7884b7852"


def test_scientific_report_exists_with_expected_page_count():
    pdf = ROOT / "docs" / "Rapport_Projet.pdf"
    assert pdf.is_file()
    pages = len(re.findall(rb"/Type\s*/Page\b", pdf.read_bytes()))
    assert 15 <= pages <= 30
    assert pdf.stat().st_size > 100_000


def test_docker_files_document_the_runtime_and_entry_point():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "python:3.11-slim" in dockerfile
    assert "libgomp1" in dockerfile
    assert "COPY requirements/requirements.txt" in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert 'CMD ["streamlit", "run", "app.py"' in dockerfile
    assert "_stcore/health" in dockerfile
    assert "data/impact/raw/" in dockerignore
    assert "data/impact_test/" in dockerignore


def test_readme_paths_are_current_and_limitations_are_explicit():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "app_v2.py" not in readme
    assert "python notebook.py" not in readme
    for expected in (
        "python -m streamlit run app.py",
        "python report/notebook.py",
        "docs/Rapport_Projet.pdf",
        "report/build_rapport_projet.py",
        "src/batch_inference.py",
        "Dockerfile",
        ".dockerignore",
        "STREAMLIT_SERVER_PORT",
        "STREAMLIT_SERVER_ADDRESS",
        "docker build -t firewatch-ai .",
        "docker run --rm -p 8501:8501 firewatch-ai",
        "http://localhost:8501",
        "No value is imputed",
        "not an observed fire",
        "not verified locally",
    ):
        assert expected in readme, expected


def test_reproducibility_artifacts_exist():
    required = [
        "app.py",
        "Dockerfile",
        ".dockerignore",
        "docs/Rapport_Projet.pdf",
        "report/build_rapport_projet.py",
        "report/evaluation_summary.json",
        "report/figures/07_feature_importance.png",
        "results/statistical_validation.json",
        "results/benchmark_protocol.json",
        "results/model_benchmark.csv",
        "models/fire_model.joblib",
        "models/model_card.json",
        "data/processed/fire_panel.parquet",
        "src/batch_inference.py",
        "dashboard/batch_view.py",
        "ui_v2/pages/batch.py",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    assert not missing, missing


def test_production_artifacts_keep_pinned_hashes():
    model = ROOT / "models" / "fire_model.joblib"
    card = ROOT / "models" / "model_card.json"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == MODEL_SHA256
    assert hashlib.sha256(card.read_bytes()).hexdigest() == MODEL_CARD_SHA256
