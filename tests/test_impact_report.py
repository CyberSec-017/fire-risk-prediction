"""Tests for the impact report builders.

The reports must be auditable: four evidence-layer definitions, the mandated
scope disclaimer, every source licence and the manifest checksum.  They must
never turn a model output into confirmed damage or affected people.
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from impact import methodology  # noqa: E402
from impact.report import (  # noqa: E402
    BURNED_AREA_NOT_AVAILABLE,
    REQUIRED_DISCLAIMER,
    build_reports,
    compute_manifest_checksum,
)

MANIFEST = {
    "schema_version": "1.0",
    "generated_at": "2026-01-01T00:00:00Z",
    "sources": [
        {
            "name": "ESA WorldCover 2021 v200",
            "version": "v200",
            "license": "CC BY 4.0",
            "attribution": "© ESA WorldCover project 2021",
        },
        {
            "name": "OpenStreetMap — Geofabrik Algeria extract",
            "version": "2026-01",
            "license": "ODbL 1.0",
            "attribution": "© OpenStreetMap contributors",
        },
    ],
    "severity": {"tier_thresholds": [0.1, 0.2, 0.35, 0.55]},
}


def assessed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat_cell": [35.0, 35.5],
            "lon_cell": [3.125, 2.5],
            "probability": [0.12, 0.71],
            "risk": ["MODERATE", "VERY HIGH"],
            "in_model_area": [True, True],
            "burned": [0, 1],
            "iei": [0.22, 0.78],
            "exposure_tier": ["Low", "Very High"],
            "impact_priority": [2, 5],
            "impact_priority_label": ["Low priority", "Critical priority"],
            "pop_cell": [120_000.0, 450_000.0],
            "road_km_total": [1_200.0, 2_500.0],
            "building_count": [8_000.0, 22_000.0],
            "critical_count_total": [2.0, 8.0],
            "admin_name": ["Kabylie", "Algiers"],
        }
    )


def build():
    return build_reports(
        MANIFEST,
        assessed=assessed_frame(),
        date="2019-07-01",
        risk_start="2018-03-02",
        risk_end="2024-12-28",
    )


def all_text(report) -> dict[str, str]:
    # HTML escapes apostrophes and ampersands; compare rendered text so the
    # audit checks content rather than entity encoding.
    return {
        "html": html.unescape(report.html or ""),
        "csv": report.csv or "",
        "json": report.json_text or "",
    }


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------
def test_manifest_checksum_matches_file_bytes(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(MANIFEST, indent=2), encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert compute_manifest_checksum(MANIFEST, path) == expected


def test_manifest_checksum_is_deterministic_without_file():
    first = compute_manifest_checksum(MANIFEST)
    second = compute_manifest_checksum(dict(reversed(list(MANIFEST.items()))))
    assert first == second
    assert first is not None and len(first) == 64
    assert compute_manifest_checksum(None) is None


# ---------------------------------------------------------------------------
# Unavailable state
# ---------------------------------------------------------------------------
def test_reports_without_manifest_are_explicitly_unavailable():
    report = build_reports(None, assessed=assessed_frame())
    assert not report.available
    assert report.status == "UNAVAILABLE"
    assert "manifest" in report.message.lower()
    assert report.html is None
    assert report.csv is None
    assert report.json_text is None
    assert report.manifest_checksum is None


# ---------------------------------------------------------------------------
# Required content in every format
# ---------------------------------------------------------------------------
def test_reports_include_four_evidence_layers_disclaimer_sources_and_checksum():
    report = build()
    assert report.available
    assert report.manifest_checksum

    for label, text in all_text(report).items():
        assert REQUIRED_DISCLAIMER in text, label
        assert "does not predict that a specific area will burn" in text, label
        assert report.manifest_checksum in text, label
        for key in methodology.LAYER_ORDER:
            layer = methodology.EVIDENCE_LAYERS[key]
            assert layer["label"] in text, (label, layer["label"])
            assert layer["definition"] in text, (label, layer["definition"])
        for source in methodology.SOURCES:
            assert source["name"] in text, (label, source["name"])
            assert source["license"] in text, (label, source["license"])


def test_reports_are_valid_html_csv_and_json():
    report = build()
    assert report.html is not None and report.html.startswith("<!DOCTYPE html>")
    assert report.csv is not None and "Evidence layer" in report.csv
    # The CSV preamble is followed by a real per-cell table.
    assert "lat_cell" in report.csv
    payload = json.loads(report.json_text or "{}")
    assert payload["manifest_checksum"] == report.manifest_checksum
    assert len(payload["evidence_layers"]) == 4
    assert len(payload["results"]) == 2
    assert payload["results"][0]["lat_cell"] == 35.0


def test_reports_label_model_outputs_conservatively():
    text = all_text(build())
    for label, body in text.items():
        assert "Predicted fire risk" in body, label
        assert "Potential exposure" in body, label
        assert "Observed fire detections" in body, label
        assert BURNED_AREA_NOT_AVAILABLE in body, label


def test_reports_never_assert_forbidden_claims():
    report = build()
    forbidden = ("will burn", "confirmed damage", "people affected")
    for label, text in all_text(report).items():
        # The mandated disclaimer negates the phrase; remove it before checking.
        audited = text.replace(REQUIRED_DISCLAIMER, "")
        lowered = audited.lower()
        for phrase in forbidden:
            assert phrase not in lowered, (label, phrase)


def test_report_without_results_still_carries_evidence():
    report = build_reports(
        MANIFEST,
        assessed=None,
        date=None,
        risk_start=None,
        risk_end=None,
    )
    assert report.available
    for label, text in all_text(report).items():
        assert REQUIRED_DISCLAIMER in text, label
        assert "Confirmed burned area" in text, label
    assert "No per-cell results" in (report.html or "")
    assert "No per-cell results available" in (report.csv or "")
    assert json.loads(report.json_text or "{}")["results"] == []
