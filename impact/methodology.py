"""Single source of truth for impact-assessment definitions and caveats.

Everything the UI and the downloadable report say about *what the numbers
mean* comes from this module.  The wording is intentionally conservative:
a predicted fire risk is not a fire perimeter, exposure is not damage, and
observed thermal detections are not confirmed burned area.
"""

from __future__ import annotations

from collections import OrderedDict

# ---------------------------------------------------------------------------
# Four evidence layers
# ---------------------------------------------------------------------------
EVIDENCE_LAYERS = OrderedDict(
    [
        (
            "predicted_risk",
            {
                "label": "Predicted fire risk",
                "definition": (
                    "The calibrated model's probability that at least one NASA "
                    "FIRMS thermal detection occurs in the same coarse grid cell "
                    "within the next 3 days, given the cell's weather and drought "
                    "features."
                ),
                "not_this": (
                    "Not a fire perimeter, not a spread forecast, and not a "
                    "statement that a fire will burn any specific area."
                ),
                "source": "FireWatch AI model v0.1.0 (unchanged 20-feature contract)",
            },
        ),
        (
            "potential_exposure",
            {
                "label": "Potential exposure",
                "definition": (
                    "The amount of vegetation, population, mapped buildings, "
                    "mapped roads and mapped critical facilities that lie inside "
                    "a selected grid cell or distance ring."
                ),
                "not_this": (
                    "Not affected, burned, damaged or lost assets. Exposure is "
                    "an upper-bound proximity measure, not an impact observation."
                ),
                "source": "ESA WorldCover, WorldPop, OpenStreetMap, GHSL (see sources)",
            },
        ),
        (
            "observed_detections",
            {
                "label": "Observed fire detections",
                "definition": (
                    "Cell-day counts of NASA FIRMS active-fire thermal anomalies "
                    "and their summed Fire Radiative Power."
                ),
                "not_this": (
                    "Not burned area. A detection can be a small fire, "
                    "agricultural burning, or an industrial heat source, and "
                    "cloud or overpass gaps can hide real fires."
                ),
                "source": "NASA FIRMS VIIRS SV-C2 (2018–2024 processed panel)",
            },
        ),
        (
            "confirmed_burned_area",
            {
                "label": "Confirmed burned area",
                "definition": (
                    "Burned area reported by an independent burned-area product, "
                    "for example MODIS/VIIRS (MCD64A1/VNP64A1), ESA CCI Fire or "
                    "Copernicus EMS/GWIS."
                ),
                "not_this": (
                    "Not inferred from FIRMS detections, FRP, model probabilities "
                    "or exposure counts. Until a burned-area product is ingested "
                    "this layer is reported as unavailable."
                ),
                "source": "Not ingested in Stage 1 (phase 2)",
            },
        ),
    ]
)

LAYER_ORDER = tuple(EVIDENCE_LAYERS)

# ---------------------------------------------------------------------------
# Non-negotiable disclaimers
# ---------------------------------------------------------------------------
DISCLAIMER = (
    "Predicted risk is a 3-day likelihood of at least one satellite thermal "
    "detection inside a coarse grid cell (approximately 55 km across). It is "
    "not a fire perimeter, rate of spread or damage estimate, and it does not "
    "claim that a predicted fire will burn any particular area."
)

EXPOSURE_DISCLAIMER = (
    "Exposure counts describe assets and people located inside a cell or "
    "distance ring. They do not mean those assets were affected, burned or "
    "damaged. Cell-wide values are an upper bound because a fire is very "
    "unlikely to burn an entire ~3,000 km² cell."
)

SEVERITY_DISCLAIMER = (
    "Impact severity is a transparent prioritisation heuristic built from "
    "model risk and static exposure. It is not a probability of damage, loss "
    "or casualties, and it has not been validated against damage records."
)

BURNED_AREA_DISCLAIMER = (
    "No confirmed burned-area or damage layer is available in Stage 1. "
    "Burned area must come from an independent burned-area product and is "
    "never derived from FIRMS detections, FRP or model probabilities."
)

# ---------------------------------------------------------------------------
# Methodology narrative (used by the report builder in a later stage)
# ---------------------------------------------------------------------------
METHODOLOGY_SECTIONS = [
    {
        "title": "1. Analysis unit and spatial resolution",
        "body": (
            "All statistics are computed on the trained model's 0.5° × 0.625° "
            "NASA POWER grid (approximately 3,076–3,268 km² per cell, varying "
            "with latitude). Cells are centred on the panel's lat_cell/lon_cell "
            "coordinates and extend half a grid step in each direction. "
            "Distance rings of 5, 10 and 25 km around each cell are reported "
            "separately; ring values overlap between neighbouring cells and are "
            "therefore not additive."
        ),
    },
    {
        "title": "2. Environmental exposure",
        "body": (
            "ESA WorldCover class areas are aggregated per cell. Flammable fuel "
            "area is the sum of tree cover, shrubland, grassland and cropland. "
            "Bare/sparse vegetation, built-up land, water and other classes are "
            "reported as non-fuel. Fuel fraction is fuel area divided by cell "
            "area. Land-cover totals are checked against the cell area and a "
            "coverage flag is reported; missing classes are never treated as "
            "zero."
        ),
    },
    {
        "title": "3. Population exposure",
        "body": (
            "WorldPop gridded population is summed inside each cell and inside "
            "5, 10 and 25 km rings. Values are modelled population estimates, "
            "not census counts. Settlement points come from OpenStreetMap and "
            "carry WorldPop-derived population estimates; a missing settlement "
            "in OSM does not mean a settlement does not exist."
        ),
    },
    {
        "title": "4. Infrastructure exposure",
        "body": (
            "Road length by class, building counts and footprint area, and "
            "critical-facility counts are derived from an OpenStreetMap snapshot "
            "of Algeria. These are *mapped* assets: OSM completeness varies by "
            "region and category, so counts are lower bounds. GHSL built-up "
            "surface is used as a completeness cross-check. Critical facilities "
            "are counted only where an explicit OSM tag or an authoritative "
            "record exists; no facility is inferred or invented."
        ),
    },
    {
        "title": "5. Impact Exposure Index (IEI)",
        "body": (
            "Each cell receives three normalized components in [0, 1]: "
            "fuel_norm = flammable area / cell area; "
            "pop_norm = log1p(population) / log1p(95th percentile population); "
            "infra_norm = 0.45 × road_norm + 0.35 × building_norm + "
            "0.20 × critical_norm, where each sub-component is divided by its "
            "95th percentile. "
            "IEI = 0.35 × fuel_norm + 0.35 × pop_norm + 0.30 × infra_norm."
        ),
    },
    {
        "title": "6. Severity tiers and priority",
        "body": (
            "IEI is classified into Very Low, Low, Moderate, High and Very High "
            "tiers using thresholds frozen in the data manifest (documented "
            "defaults apply when no manifest is present). The model risk band "
            "and the exposure tier are then combined in a 4 × 5 matrix that "
            "returns a priority from 1 (minimal) to 5 (critical). This is a "
            "prioritisation heuristic, not a damage probability."
        ),
    },
    {
        "title": "7. Temporal handling",
        "body": (
            "Risk is date-specific and uses the panel's own date range. Exposure "
            "is a present-day static snapshot (WorldCover 2021, WorldPop 2020, "
            "OpenStreetMap download date) applied retrospectively to historical "
            "risk. Exposure is therefore labelled as present-day asset exposure, "
            "not exposure at the time of a historical fire."
        ),
    },
    {
        "title": "8. What this assessment does not do",
        "body": (
            "It does not estimate burned area, monetary loss, casualties or "
            "structural damage; it does not model fire spread; it does not "
            "assign a fire to a specific point inside a cell; and it does not "
            "claim that a predicted fire will burn a particular area."
        ),
    },
]

# ---------------------------------------------------------------------------
# Sources and licensing
# ---------------------------------------------------------------------------
SOURCES = [
    {
        "name": "ESA WorldCover 2021 v200",
        "role": "Land-cover composition and flammable fuel area",
        "license": "CC BY 4.0",
        "attribution": (
            "© ESA WorldCover project 2021 / Contains modified Copernicus "
            "Sentinel data (2021) processed by ESA WorldCover consortium"
        ),
        "status": "planned",
    },
    {
        "name": "WorldPop 2020 (Algeria, constrained)",
        "role": "Gridded population and settlement population estimates",
        "license": "CC BY 4.0",
        "attribution": "WorldPop (www.worldpop.org), licensed under CC BY 4.0",
        "status": "planned",
    },
    {
        "name": "Global Human Settlement Layer (GHSL R2023A)",
        "role": "Built-up surface and population completeness cross-check",
        "license": "CC BY 4.0",
        "attribution": "European Commission Joint Research Centre, GHSL",
        "status": "planned",
    },
    {
        "name": "OpenStreetMap — Geofabrik Algeria extract",
        "role": "Roads, buildings, settlements and critical-facility tags",
        "license": "ODbL 1.0",
        "attribution": "© OpenStreetMap contributors, available under the Open Database License",
        "status": "planned",
    },
    {
        "name": "OCHA COD-AB Algeria",
        "role": "Administrative boundaries for regional comparison",
        "license": "CC BY 3.0 IGO (verify per release)",
        "attribution": "UN OCHA / HDX",
        "status": "planned",
    },
    {
        "name": "Copernicus EMS / GWIS burned area",
        "role": "Confirmed burned area (phase 2 only)",
        "license": "Copernicus free and open, attribution required",
        "attribution": "Copernicus Emergency Management Service / JRC GWIS",
        "status": "phase_2",
    },
    {
        "name": "NASA MODIS/VIIRS burned area (MCD64A1/VNP64A1)",
        "role": "Confirmed burned area (phase 2 only)",
        "license": "U.S. public domain",
        "attribution": "NASA LP DAAC",
        "status": "phase_2",
    },
    {
        "name": "NASA FIRMS VIIRS SV-C2",
        "role": "Observed active-fire detections (already in the panel)",
        "license": "Open, attribution requested",
        "attribution": "NASA Fire Information for Resource Management System (FIRMS)",
        "status": "available",
    },
    {
        "name": "NASA POWER",
        "role": "Meteorological predictors (already in the panel)",
        "license": "Open, attribution requested",
        "attribution": "NASA Prediction Of Worldwide Energy Resources (POWER)",
        "status": "available",
    },
]

EXCLUDED_SOURCES = [
    {
        "name": "GADM",
        "reason": (
            "Redistribution restrictions make it unsuitable for a "
            "redistributable application; use OCHA COD-AB or Natural Earth."
        ),
    },
    {
        "name": "WDPA / Protected Planet",
        "reason": "Non-commercial licence; omit unless the licence is explicitly handled.",
    },
]

# ---------------------------------------------------------------------------
# Known limitations
# ---------------------------------------------------------------------------
CAVEATS = [
    "Grid cells are coarse (~55 km across, ~3,100 km²); exposure is cell-level potential, not fire-scale impact.",
    "FIRMS detections are thermal anomalies, not burned area; cloud cover, small fires and industrial heat sources bias them.",
    "The panel's 'burned' column is a same-day detection indicator, not confirmed burn scar.",
    "The model target is any detection within the next 3 days, not fire spread or burn probability.",
    "WorldCover 2021, WorldPop 2020 and the OpenStreetMap snapshot are present-day layers applied to 2018–2024 risk.",
    "OSM completeness varies; building and road counts are lower bounds and critical-facility tags are sparse.",
    "WorldPop values are modelled estimates, not census counts, and exclude seasonal or displaced populations.",
    "Distance rings overlap between cells and must not be summed across cells.",
    "Administrative assignment uses cell area-majority; boundary cells can be misallocated.",
    "Severity thresholds and IEI weights are documented design choices, not empirically validated against damage.",
    "No burned-area or damage ground truth is available in Stage 1.",
]


def evidence_layer_rows() -> list[dict[str, str]]:
    """Return the four evidence layers as report-friendly rows."""
    rows = []
    for key in LAYER_ORDER:
        layer = EVIDENCE_LAYERS[key]
        rows.append(
            {
                "layer": key,
                "label": layer["label"],
                "definition": layer["definition"],
                "not_this": layer["not_this"],
                "source": layer["source"],
            }
        )
    return rows


def source_attribution_text() -> str:
    """One-line attribution covering every source actually in use."""
    active = [s for s in SOURCES if s["status"] in {"available", "planned"}]
    names = ", ".join(s["name"] for s in active)
    return (
        f"Data sources: {names}. Licences: CC BY 4.0 (WorldCover, WorldPop, "
        "GHSL), ODbL 1.0 (OpenStreetMap), open NASA products (FIRMS, POWER)."
    )


def methodology_markdown() -> str:
    """Render the methodology as Markdown for a downloadable report."""
    lines = ["# Impact assessment methodology", "", DISCLAIMER, ""]
    for section in METHODOLOGY_SECTIONS:
        lines += [f"## {section['title']}", "", section["body"], ""]
    lines += ["## Sources and licensing", ""]
    for source in SOURCES:
        lines.append(
            f"- **{source['name']}** — {source['role']}. "
            f"Licence: {source['license']}. {source['attribution']}."
        )
    lines += ["", "## Limitations", ""]
    lines += [f"- {caveat}" for caveat in CAVEATS]
    return "\n".join(lines)


def assessment_scope_note(start, end) -> str:
    """Describe the risk window using the panel's own dates."""
    if start is None or end is None:
        return "Risk window: unavailable (no panel date range)."
    return (
        "Risk window: "
        f"{start.date().isoformat() if hasattr(start, 'date') else start} to "
        f"{end.date().isoformat() if hasattr(end, 'date') else end} "
        "(read from fire_panel.parquet, not from dashboard configuration)."
    )


def layer_summary_lines() -> list[str]:
    """Short UI-friendly descriptions of the four layers."""
    return [
        f"{layer['label']}: {layer['definition']} {layer['not_this']}"
        for layer in EVIDENCE_LAYERS.values()
    ]


__all__ = [
    "EVIDENCE_LAYERS",
    "LAYER_ORDER",
    "DISCLAIMER",
    "EXPOSURE_DISCLAIMER",
    "SEVERITY_DISCLAIMER",
    "BURNED_AREA_DISCLAIMER",
    "METHODOLOGY_SECTIONS",
    "SOURCES",
    "EXCLUDED_SOURCES",
    "CAVEATS",
    "evidence_layer_rows",
    "source_attribution_text",
    "methodology_markdown",
    "assessment_scope_note",
    "layer_summary_lines",
]
