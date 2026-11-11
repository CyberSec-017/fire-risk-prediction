"""HTML, CSV and JSON report builders for the impact assessment.

The builders are deliberately dependency-free beyond pandas: they produce
plain strings so the Streamlit page can offer downloads without adding a
runtime dependency.  Reports never present a model output as confirmed
damage, a count of people affected, or a prediction that an area will burn.

Every available report carries:

* the four evidence-layer definitions from :mod:`impact.methodology`;
* the mandated scope disclaimer;
* the data sources with their licences and attribution;
* the SHA-256 checksum of the provenance manifest.

If the manifest is absent there is no provenance to verify, so
:func:`build_reports` returns an explicit unavailable result instead of a
partially evidenced report.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from impact import config
from impact import methodology
from impact.severity import formula_text, priority_matrix_rows

REPORT_TITLE = "FIREWATCH AI — Wildfire Impact Assessment Report"

# This exact sentence is required in every report.
REQUIRED_DISCLAIMER = (
    "This assessment does not predict that a specific area will burn."
)

BURNED_AREA_NOT_AVAILABLE = "Not available (phase 2)"

# Stable, report-friendly column order.  Missing columns are simply omitted.
REPORT_COLUMNS = (
    "lat_cell",
    "lon_cell",
    "probability",
    "risk",
    "in_model_area",
    "burned",
    "iei",
    "iei_components_used",
    "iei_fuel_missing",
    "exposure_tier",
    "impact_priority",
    "impact_priority_label",
    "fuel_area_km2",
    "fuel_fraction",
    "lc_tree_km2",
    "lc_shrub_km2",
    "lc_grass_km2",
    "lc_crop_km2",
    "lc_bare_km2",
    "lc_built_km2",
    "lc_water_km2",
    "lc_other_km2",
    "pop_cell",
    "pop_5km",
    "pop_10km",
    "pop_25km",
    "pop_density_per_km2",
    "road_km_total",
    "road_density_km_per_km2",
    "building_count",
    "building_footprint_km2",
    "critical_count_total",
    "admin_name",
)


@dataclass(frozen=True)
class ReportResult:
    """Outcome of building the three report formats."""

    status: str
    message: str
    manifest_checksum: str | None = None
    html: str | None = None
    csv: str | None = None
    json_text: str | None = None

    @property
    def available(self) -> bool:
        return self.status == config.STATUS_AVAILABLE

    @property
    def unavailable(self) -> bool:
        return not self.available


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _text(value: Any) -> str:
    """Return a JSON/HTML-safe string for a scalar value."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def _iso_date(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return _text(value) or None


def compute_manifest_checksum(
    manifest: Mapping[str, Any] | None,
    manifest_path: Path | str | None = None,
) -> str | None:
    """Return the manifest SHA-256.

    When a manifest file is available its exact bytes are hashed, so the
    checksum matches an external ``sha256sum``.  Otherwise a deterministic
    SHA-256 of the canonical JSON representation is used.
    """
    if not manifest:
        return None
    if manifest_path is not None:
        path = Path(manifest_path)
        if path.exists():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_layer_payload() -> list[dict[str, str]]:
    """Four evidence layers with their definitions and sources.

    ``not_this`` is intentionally excluded: the report's single mandated
    disclaimer carries the negative scope statement, and repeating the
    phrase in several places makes the text harder to audit.
    """
    rows: list[dict[str, str]] = []
    for key, layer in methodology.EVIDENCE_LAYERS.items():
        rows.append(
            {
                "layer": key,
                "label": str(layer["label"]),
                "definition": str(layer["definition"]),
                "source": str(layer["source"]),
            }
        )
    return rows


def source_payload() -> list[dict[str, str]]:
    """Data sources with role, licence and attribution."""
    keys = ("name", "role", "license", "attribution", "status")
    return [
        {key: str(source.get(key, "")) for key in keys}
        for source in methodology.SOURCES
    ]


def _metadata(
    manifest: Mapping[str, Any],
    checksum: str,
    date: Any,
    risk_start: Any,
    risk_end: Any,
    burned_area_status: str,
    assessed: pd.DataFrame | None = None,
) -> dict[str, Any]:
    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    fuel_missing = False
    if assessed is not None and isinstance(assessed, pd.DataFrame) and not assessed.empty:
        if "iei_fuel_missing" in assessed.columns:
            fuel_missing = bool(assessed["iei_fuel_missing"].iloc[0])
        elif "fuel_area_km2" not in assessed.columns or pd.to_numeric(assessed["fuel_area_km2"], errors="coerce").isna().all():
            fuel_missing = True

    return {
        "title": REPORT_TITLE,
        "generated_at": generated_at,
        "risk_date": _iso_date(date),
        "risk_window_start": _iso_date(risk_start),
        "risk_window_end": _iso_date(risk_end),
        "disclaimer": REQUIRED_DISCLAIMER,
        "manifest_checksum": checksum,
        "manifest_schema_version": _text(manifest.get("schema_version")),
        "manifest_generated_at": _text(manifest.get("generated_at")),
        "evidence_layers": evidence_layer_payload(),
        "sources": source_payload(),
        "severity_method": formula_text(fuel_missing=fuel_missing),
        "fuel_missing": fuel_missing,
        "severity_note": (
            "Severity was computed without the fuel component (WorldCover not ingested; "
            "weights renormalized to population 0.35/0.65 and infrastructure 0.30/0.65)."
            if fuel_missing
            else "Severity was computed with all three components (fuel, population and infrastructure)."
        ),
        "priority_matrix": priority_matrix_rows(),
        "caveats": list(methodology.CAVEATS),
        "confirmed_burned_area": burned_area_status,
        "value_labels": {
            "risk": "Predicted fire risk (3-day detection likelihood)",
            "exposure": "Potential exposure (assets inside a cell or ring)",
            "detections": "Observed fire detections (not burned area)",
            "burned_area": burned_area_status,
        },
    }


def _records(assessed: pd.DataFrame | None) -> list[dict[str, Any]]:
    if assessed is None or not isinstance(assessed, pd.DataFrame) or assessed.empty:
        return []
    columns = [column for column in REPORT_COLUMNS if column in assessed.columns]
    if not columns:
        return []

    records: list[dict[str, Any]] = []
    for row in assessed[columns].to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                clean[key] = None
                continue
            try:
                if pd.isna(value):
                    clean[key] = None
                    continue
            except (TypeError, ValueError):
                pass
            if isinstance(value, (datetime, pd.Timestamp)):
                clean[key] = value.isoformat()
            elif hasattr(value, "item"):
                try:
                    clean[key] = value.item()
                except (TypeError, ValueError):
                    clean[key] = str(value)
            else:
                clean[key] = value
        records.append(clean)
    return records


def _metadata_rows(metadata: Mapping[str, Any]) -> list[tuple[str, str]]:
    rows = [
        ("Generated at (UTC)", _text(metadata.get("generated_at"))),
        ("Risk date", _text(metadata.get("risk_date")) or "Not available"),
        (
            "Risk window",
            (
                f"{_text(metadata.get('risk_window_start')) or 'Not available'} to "
                f"{_text(metadata.get('risk_window_end')) or 'Not available'}"
            ),
        ),
        ("Manifest checksum (SHA-256)", _text(metadata.get("manifest_checksum"))),
        ("Manifest schema", _text(metadata.get("manifest_schema_version"))),
        ("Confirmed burned area", _text(metadata.get("confirmed_burned_area"))),
    ]
    if metadata.get("fuel_missing"):
        rows.append(
            (
                "Fuel component",
                "Not available (severity computed without fuel component; weights renormalized: pop 0.35/0.65, infra 0.30/0.65)",
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_html_report(
    metadata: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> str:
    """Build a self-contained HTML report (no scripts, no external assets)."""
    escape = html.escape

    def table(headers: list[str], rows: list[list[Any]]) -> str:
        head = "".join(f"<th>{escape(str(item))}</th>" for item in headers)
        body = "".join(
            "<tr>"
            + "".join(f"<td>{escape(_text(item))}</td>" for item in row)
            + "</tr>"
            for row in rows
        )
        return (
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        )

    metadata_rows = _metadata_rows(metadata)
    evidence_rows = [
        [layer["label"], layer["definition"], layer["source"]]
        for layer in metadata["evidence_layers"]
    ]
    source_rows = [
        [
            source["name"],
            source["role"],
            source["license"],
            source["attribution"],
        ]
        for source in metadata["sources"]
    ]
    priority_rows = [
        [
            row["risk_band"],
            row["exposure_tier"],
            row["priority"],
            row["priority_label"],
        ]
        for row in metadata["priority_matrix"]
    ]

    if records:
        result_headers = list(records[0].keys())
        result_rows = [[record.get(key) for key in result_headers] for record in records]
        result_html = table(result_headers, result_rows)
    else:
        result_html = (
            "<p>No per-cell results were supplied. The assessment evidence "
            "above still applies.</p>"
        )

    caveats_html = "".join(f"<li>{escape(caveat)}</li>" for caveat in metadata["caveats"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(REPORT_TITLE)}</title>
<style>
  body {{ font-family: Inter, -apple-system, "Segoe UI", Roboto, sans-serif;
         color: #0F172A; background: #F8FAFC; margin: 0; padding: 32px; }}
  main {{ max-width: 1180px; margin: 0 auto; background: #FFFFFF;
          border: 1px solid #E2E8F0; border-radius: 8px; padding: 32px; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 4px 0; }}
  h2 {{ font-size: 1.05rem; margin: 26px 0 8px 0; color: #0F5132; }}
  p, li, td, th {{ font-size: 0.84rem; line-height: 1.5; }}
  table {{ width: 100%; border-collapse: collapse; margin: 8px 0 16px 0; }}
  th, td {{ border: 1px solid #E2E8F0; padding: 6px 8px; text-align: left;
            vertical-align: top; }}
  th {{ background: #F1F5F9; font-weight: 600; }}
  .disclaimer {{ background: #FFFBEB; border: 1px solid #FDE68A;
                 border-left: 4px solid #B45309; border-radius: 6px;
                 padding: 12px 16px; margin: 16px 0; }}
  .muted {{ color: #64748B; }}
  code {{ background: #F1F5F9; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<main>
  <h1>{escape(REPORT_TITLE)}</h1>
  <p class="muted">Risk and exposure evidence for northern Algeria. Model risk is
  date-specific; exposure layers are present-day static snapshots.</p>

  <div class="disclaimer"><strong>Scope disclaimer.</strong>
  {escape(REQUIRED_DISCLAIMER)}</div>

  <h2>Report metadata</h2>
  {table(["Field", "Value"], [[key, value] for key, value in metadata_rows])}

  <h2>Evidence layers</h2>
  <p>The four evidence layers are reported separately and must not be conflated.</p>
  {table(["Layer", "Definition", "Source"], evidence_rows)}

  <h2>Data sources and licences</h2>
  {table(["Source", "Role", "Licence", "Attribution"], source_rows)}

  <h2>Method</h2>
  <p>{escape(_text(metadata.get("severity_method")))}</p>
  <p><strong>Predicted fire risk</strong> is the model probability described
  above. <strong>Potential exposure</strong> counts assets and people inside a
  cell. <strong>Observed fire detections</strong> are thermal anomalies, not
  burned area. <strong>Confirmed burned area</strong> is
  {escape(_text(metadata.get("confirmed_burned_area")))}.</p>

  <h2>Priority matrix (heuristic)</h2>
  {table(["Predicted risk band", "Exposure tier", "Priority (1-5)", "Label"],
         priority_rows)}

  <h2>Per-cell results</h2>
  {result_html}

  <h2>Limitations</h2>
  <ul>{caveats_html}</ul>
</main>
</body>
</html>
"""


def build_csv_report(
    metadata: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> str:
    """Build a CSV report with a metadata preamble and one row per cell."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")

    writer.writerow([REPORT_TITLE])
    for key, value in _metadata_rows(metadata):
        writer.writerow([key, value])
    writer.writerow(["Scope disclaimer", REQUIRED_DISCLAIMER])
    writer.writerow([])

    writer.writerow(["Evidence layer", "Definition", "Source"])
    for layer in metadata["evidence_layers"]:
        writer.writerow([layer["label"], layer["definition"], layer["source"]])
    writer.writerow([])

    writer.writerow(["Source", "Role", "Licence", "Attribution", "Status"])
    for source in metadata["sources"]:
        writer.writerow(
            [
                source["name"],
                source["role"],
                source["license"],
                source["attribution"],
                source["status"],
            ]
        )
    writer.writerow([])

    writer.writerow(["Severity method", _text(metadata.get("severity_method"))])
    for caveat in metadata["caveats"]:
        writer.writerow(["Limitation", caveat])
    writer.writerow([])

    if records:
        headers = list(records[0].keys())
        writer.writerow(headers)
        for record in records:
            writer.writerow([_text(record.get(key)) for key in headers])
    else:
        writer.writerow(["No per-cell results available."])
    return buffer.getvalue()


def build_json_report(
    metadata: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> str:
    """Build a JSON report with metadata plus structured per-cell results."""
    payload = dict(metadata)
    payload["results"] = list(records)
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_reports(
    manifest: Mapping[str, Any] | None,
    *,
    manifest_path: Path | str | None = None,
    assessed: pd.DataFrame | None = None,
    date: Any = None,
    risk_start: Any = None,
    risk_end: Any = None,
    burned_area_status: str = BURNED_AREA_NOT_AVAILABLE,
) -> ReportResult:
    """Build all three report formats from a verified manifest.

    Returns an explicit unavailable result when the manifest is absent,
    because source provenance and the checksum cannot be established.
    """
    if not isinstance(manifest, Mapping) or not manifest:
        return ReportResult(
            status=config.STATUS_UNAVAILABLE,
            message=(
                "Report generation is unavailable: the impact data manifest is "
                "missing, so data sources and the manifest checksum cannot be "
                "verified. Run scripts/build_impact_layers.py to generate "
                "data/impact/manifest.json."
            ),
        )

    checksum = compute_manifest_checksum(manifest, manifest_path)
    if not checksum:
        return ReportResult(
            status=config.STATUS_UNAVAILABLE,
            message=(
                "Report generation is unavailable: a manifest checksum could "
                "not be computed."
            ),
        )

    metadata = _metadata(
        manifest,
        checksum,
        date=date,
        risk_start=risk_start,
        risk_end=risk_end,
        burned_area_status=burned_area_status,
        assessed=assessed,
    )
    records = _records(assessed)

    return ReportResult(
        status=config.STATUS_AVAILABLE,
        message=(
            "HTML, CSV and JSON reports were generated with the four evidence "
            "layers, source licences and the manifest checksum."
        ),
        manifest_checksum=checksum,
        html=build_html_report(metadata, records),
        csv=build_csv_report(metadata, records),
        json_text=build_json_report(metadata, records),
    )


__all__ = [
    "REPORT_TITLE",
    "REQUIRED_DISCLAIMER",
    "BURNED_AREA_NOT_AVAILABLE",
    "REPORT_COLUMNS",
    "ReportResult",
    "compute_manifest_checksum",
    "evidence_layer_payload",
    "source_payload",
    "build_html_report",
    "build_csv_report",
    "build_json_report",
    "build_reports",
]
