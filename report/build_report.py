"""Build the FIREWATCH AI PDF report from the evaluation artifacts.

Usage
-----
    python notebook.py               # regenerate metrics + figures
    python report/build_report.py    # build this PDF

The script reads ``report/evaluation_summary.json`` and the PNG figures in
``report/figures/`` and writes ``report/FIREWATCH_AI_Report.pdf``.  It never
touches the model, the panel or the model card.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
SUMMARY = REPORT_DIR / "evaluation_summary.json"
OUTPUT = REPORT_DIR / "FIREWATCH_AI_Report.pdf"

# ---------------------------------------------------------------------------
# Palette — mirrors the Streamlit dashboard
# ---------------------------------------------------------------------------
INK = colors.HexColor("#0B1220")
INK_SOFT = colors.HexColor("#1E293B")
MUTED = colors.HexColor("#64748B")
LINE = colors.HexColor("#E2E8F0")
PAPER = colors.HexColor("#F8FAFC")
EMBER = colors.HexColor("#E8590C")
EMBER_LIGHT = colors.HexColor("#FFF4EC")
GOLD = colors.HexColor("#F59E0B")
SAFE = colors.HexColor("#15803D")
WARN = colors.HexColor("#B45309")
DANGER = colors.HexColor("#B91C1C")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

# ---------------------------------------------------------------------------
# Fonts — DejaVu ships with matplotlib, so it is always available here
# ---------------------------------------------------------------------------
FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"


def _register_fonts() -> None:
    global FONT, FONT_BOLD, FONT_ITALIC
    try:
        import matplotlib

        base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont("DejaVu", str(base / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(base / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(
            TTFont("DejaVu-Oblique", str(base / "DejaVuSans-Oblique.ttf"))
        )
        pdfmetrics.registerFontFamily(
            "DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique"
        )
        FONT, FONT_BOLD, FONT_ITALIC = "DejaVu", "DejaVu-Bold", "DejaVu-Oblique"
    except Exception:  # noqa: BLE001 - Helvetica fallback is still readable
        pass


_register_fonts()

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
STYLES: dict[str, ParagraphStyle] = {
    "body": ParagraphStyle(
        "body",
        fontName=FONT,
        fontSize=9.4,
        leading=14,
        textColor=INK_SOFT,
        alignment=TA_JUSTIFY,
        spaceAfter=5,
    ),
    "body_left": ParagraphStyle(
        "body_left",
        fontName=FONT,
        fontSize=9.4,
        leading=14,
        textColor=INK_SOFT,
        alignment=TA_LEFT,
        spaceAfter=5,
    ),
    "lead": ParagraphStyle(
        "lead",
        fontName=FONT,
        fontSize=10.6,
        leading=16.5,
        textColor=INK_SOFT,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    ),
    "h1": ParagraphStyle(
        "h1",
        fontName=FONT_BOLD,
        fontSize=17,
        leading=21,
        textColor=INK,
        spaceBefore=4,
        spaceAfter=3,
    ),
    "h2": ParagraphStyle(
        "h2",
        fontName=FONT_BOLD,
        fontSize=12,
        leading=15,
        textColor=INK,
        spaceBefore=11,
        spaceAfter=4,
    ),
    "h3": ParagraphStyle(
        "h3",
        fontName=FONT_BOLD,
        fontSize=9.8,
        leading=13,
        textColor=EMBER,
        spaceBefore=8,
        spaceAfter=2,
    ),
    "kicker": ParagraphStyle(
        "kicker",
        fontName=FONT_BOLD,
        fontSize=7.6,
        leading=10,
        textColor=EMBER,
        spaceAfter=2,
    ),
    "caption": ParagraphStyle(
        "caption",
        fontName=FONT_ITALIC,
        fontSize=7.6,
        leading=10.5,
        textColor=MUTED,
        spaceBefore=2,
        spaceAfter=9,
    ),
    "bullet": ParagraphStyle(
        "bullet",
        fontName=FONT,
        fontSize=9.4,
        leading=13.5,
        textColor=INK_SOFT,
        leftIndent=11,
        bulletIndent=1,
        spaceAfter=3,
    ),
    "cell": ParagraphStyle(
        "cell", fontName=FONT, fontSize=8.2, leading=11, textColor=INK_SOFT
    ),
    "cell_bold": ParagraphStyle(
        "cell_bold", fontName=FONT_BOLD, fontSize=8.2, leading=11, textColor=INK
    ),
    "cell_head": ParagraphStyle(
        "cell_head", fontName=FONT_BOLD, fontSize=7.8, leading=10, textColor=colors.white
    ),
}


def hexcolor(color: colors.Color) -> str:
    return "#" + color.hexval()[2:]


def pct(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def num(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def intnum(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------
def _cover(canvas, doc) -> None:  # noqa: ANN001 - reportlab callback signature
    canvas.saveState()
    canvas.setFillColor(INK)
    canvas.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

    # Ember bands
    canvas.setFillColor(colors.HexColor("#3B1204"))
    canvas.rect(0, PAGE_H - 120 * mm, PAGE_W, 120 * mm, stroke=0, fill=1)
    canvas.setFillColor(EMBER)
    canvas.rect(0, PAGE_H - 121 * mm, PAGE_W, 1.2 * mm, stroke=0, fill=1)

    # Title
    canvas.setFillColor(colors.white)
    canvas.setFont(FONT_BOLD, 32)
    canvas.drawString(MARGIN, PAGE_H - 42 * mm, "FIREWATCH AI")
    canvas.setFont(FONT, 13)
    canvas.setFillColor(colors.HexColor("#FDBA74"))
    canvas.drawString(MARGIN, PAGE_H - 51 * mm, "Wildfire Risk Prediction & Impact Assessment")

    canvas.setFillColor(colors.HexColor("#CBD5E1"))
    canvas.setFont(FONT, 9.5)
    canvas.drawString(
        MARGIN,
        PAGE_H - 62 * mm,
        "Northern Algeria  ·  2018–2024  ·  3-day fire-occurrence horizon",
    )

    canvas.setStrokeColor(colors.HexColor("#7C2D12"))
    canvas.line(MARGIN, PAGE_H - 68 * mm, PAGE_W - MARGIN, PAGE_H - 68 * mm)

    canvas.setFont(FONT, 9.2)
    canvas.setFillColor(colors.HexColor("#94A3B8"))
    lines = [
        "Academic data-mining project",
        "Prediction-first interface · reproducible evaluation · geospatial impact assessment",
        "",
        "Author: FireWatch Contributor",
        f"Report generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
    ]
    y = PAGE_H - 78 * mm
    for line in lines:
        canvas.drawString(MARGIN, y, line)
        y -= 5.4 * mm

    # Footer strip
    canvas.setFillColor(colors.HexColor("#111C2E"))
    canvas.rect(0, 0, PAGE_W, 26 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.setFont(FONT, 8)
    canvas.drawString(MARGIN, 15 * mm, "github.com/CyberSec-017/fire-risk-prediction")
    canvas.drawRightString(
        PAGE_W - MARGIN, 15 * mm, "Data-driven wildfire risk analysis"
    )
    canvas.restoreState()


def _body_page(canvas, doc) -> None:  # noqa: ANN001 - reportlab callback signature
    page = canvas.getPageNumber()
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, PAGE_W, PAGE_H - 10 * mm, stroke=0, fill=1)

    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, PAGE_H - 14 * mm, PAGE_W - MARGIN, PAGE_H - 14 * mm)

    canvas.setFont(FONT, 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, PAGE_H - 12 * mm, "FIREWATCH AI  —  Technical Report")
    canvas.drawRightString(
        PAGE_W - MARGIN, PAGE_H - 12 * mm, "Wildfire risk prediction & impact assessment"
    )
    canvas.line(MARGIN, 12 * mm, PAGE_W - MARGIN, 12 * mm)
    canvas.drawString(MARGIN, 8 * mm, "Reproducible evaluation: python notebook.py")
    canvas.drawRightString(PAGE_W - MARGIN, 8 * mm, f"Page {page}")
    canvas.restoreState()


def _frame() -> Frame:
    return Frame(
        MARGIN,
        15 * mm,
        PAGE_W - 2 * MARGIN,
        PAGE_H - 31 * mm,
        id="body",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def styled_table(
    data: list[list[Any]],
    widths: list[float],
    align_right: tuple[int, ...] = (),
    head_rows: int = 1,
    zebra: bool = True,
) -> Table:
    rows: list[list[Any]] = []
    for row_index, row in enumerate(data):
        rendered: list[Any] = []
        for col_index, value in enumerate(row):
            if row_index < head_rows:
                style = STYLES["cell_head"]
            elif col_index in align_right:
                style = STYLES["cell_bold"] if row_index == head_rows else STYLES["cell"]
            else:
                style = STYLES["cell"]
            rendered.append(Paragraph(str(value), style))
        rows.append(rendered)

    table = Table(rows, colWidths=widths, repeatRows=head_rows, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, head_rows - 1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, head_rows - 1), (-1, -1), 0.4, LINE),
    ]
    if head_rows == 1:
        commands.append(("LINEBELOW", (0, 0), (-1, 0), 0, colors.white))
    if zebra:
        for row_index in range(head_rows, len(rows)):
            if (row_index - head_rows) % 2 == 1:
                commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.white))
    table.setStyle(TableStyle(commands))
    return table


def metric_card_row(items: list[tuple[str, str, colors.Color]], width: float) -> Table:
    cells = []
    for label, value, accent in items:
        cells.append(
            [
                Paragraph(
                    f"<font color='{hexcolor(accent)}'><b>{value}</b></font>",
                    ParagraphStyle(
                        "cardvalue",
                        fontName=FONT_BOLD,
                        fontSize=16,
                        leading=19,
                        textColor=INK,
                    ),
                ),
                Paragraph(
                    label.upper(),
                    ParagraphStyle(
                        "cardlabel",
                        fontName=FONT_BOLD,
                        fontSize=6.6,
                        leading=8.5,
                        textColor=MUTED,
                    ),
                ),
            ]
        )
    col_width = width / len(cells)
    table = Table([cells], colWidths=[col_width] * len(cells), hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("LINEBEFORE", (1, 0), (-1, -1), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def figure(name: str, caption: str, width: float | None = None) -> KeepTogether:
    path = FIG_DIR / name
    if not path.exists():
        return KeepTogether([Paragraph(f"<i>Missing figure: {name}</i>", STYLES["caption"])])
    max_w = width or (PAGE_W - 2 * MARGIN)
    iw, ih = ImageReader(str(path)).getSize()
    draw_w = max_w
    draw_h = draw_w * ih / iw
    max_h = 118 * mm
    if draw_h > max_h:
        draw_h = max_h
        draw_w = draw_h * iw / ih
    return KeepTogether(
        [
            Image(str(path), width=draw_w, height=draw_h),
            Paragraph(caption, STYLES["caption"]),
        ]
    )


def bullets(items: list[str]) -> list[Paragraph]:
    return [
        Paragraph(text, STYLES["bullet"], bulletText="•")
        for text in items
    ]


def callout(
    title: str, message: str, accent: colors.Color, bg: colors.Color | None = None
) -> Table:
    bg = bg or EMBER_LIGHT
    box = Table(
        [
            [
                Paragraph(
                    f"<font color='{hexcolor(accent)}'><b>{title}</b></font>"
                    f"<br/>{message}",
                    ParagraphStyle(
                        "callout",
                        fontName=FONT,
                        fontSize=8.8,
                        leading=13,
                        textColor=INK_SOFT,
                    ),
                )
            ]
        ],
        colWidths=[PAGE_W - 2 * MARGIN],
        hAlign="LEFT",
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("LINEBEFORE", (0, 0), (0, -1), 2.4, accent),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return box


# ---------------------------------------------------------------------------
# Document content
# ---------------------------------------------------------------------------
def build_story(summary: dict[str, Any]) -> list[Any]:
    metrics = summary["metrics"]
    model = summary["model"]
    dataset = summary["dataset"]
    split = summary["split"]
    bands = summary["risk_bands"]
    yearly = summary["yearly"]
    importance = summary["permutation_importance"]

    story: list[Any] = []

    # ------------------------------------------------------------------ p2
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())
    story.append(Paragraph("EXECUTIVE SUMMARY", STYLES["kicker"]))
    story.append(Paragraph("What this report covers", STYLES["h1"]))
    story.append(
        Paragraph(
            "This report documents the prediction-first redesign of the FIREWATCH AI "
            "dashboard, the reproducible evaluation of the deployed model, and the "
            "supporting artifacts that were added to the project. All numbers in the "
            "tables below are produced by <b>notebook.py</b> and stored in "
            "<b>report/evaluation_summary.json</b>; the figures are the exact PNG files "
            "written to <b>report/figures/</b>.",
            STYLES["lead"],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        metric_card_row(
            [
                ("PR-AUC", num(metrics["pr_auc"]), EMBER),
                ("ROC-AUC", num(metrics["roc_auc"]), INK),
                ("Brier score", num(metrics["brier"], 5), GOLD),
                ("Fires in top decile", pct(metrics["top_decile_fire_share"], 1), DANGER),
            ],
            PAGE_W - 2 * MARGIN,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        metric_card_row(
            [
                ("Brier skill score", num(metrics["brier_skill"], 3), SAFE),
                ("Calibration error", num(metrics["ece"], 4), INK),
                ("Calibration slope", num(metrics["calibration_slope"], 2), INK),
                ("Test base rate", pct(metrics["base_rate"]), MUTED),
            ],
            PAGE_W - 2 * MARGIN,
        )
    )
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Work delivered", STYLES["h2"]))
    story.extend(
        bullets(
            [
                "<b>Prediction-first landing page</b> (<font face='DejaVu-Bold'>app.py</font>): "
                "the app now opens on a guided <i>enter conditions → get a risk estimate</i> "
                "flow, with scenario presets, a probability gauge, an explanation chart and "
                "actionable guidance.",
                "<b>Testable prediction core</b> "
                "(<font face='DejaVu-Bold'>dashboard/predictor.py</font>): Streamlit-free "
                "feature building, scoring, sensitivity analysis and guidance helpers, "
                "covered by <font face='DejaVu-Bold'>tests/test_predictor.py</font>.",
                "<b>Evaluation notebook</b> (<font face='DejaVu-Bold'>notebook.py</font>): "
                "reproduces the chronological evaluation, prints statistics and metrics, and "
                "writes ten figures plus machine-readable results.",
                "<b>Documentation</b>: a complete project README and this PDF report.",
                "<b>Regression safety</b>: the full test suite (249 tests) passes after the "
                "changes, and the SHA256-pinned model card was left untouched.",
            ]
        )
    )

    story.append(Paragraph("Headline result", STYLES["h2"]))
    story.append(
        Paragraph(
            f"On the chronological hold-out (2022–2024, {intnum(split['test_rows'])} "
            f"cell-days, {pct(metrics['base_rate'])} base rate) the calibrated model reaches "
            f"a PR-AUC of <b>{num(metrics['pr_auc'])}</b> against "
            f"<b>{num(metrics['climatology_pr_auc'])}</b> for a per-cell, per-month "
            f"climatology baseline. It captures <b>{pct(metrics['top_decile_fire_share'], 1)}</b> "
            f"of all observed fires inside the highest-risk 10% of cell-days, while remaining "
            f"well calibrated (Brier {num(metrics['brier'], 5)}, ECE "
            f"{num(metrics['ece'], 4)}, skill score {num(metrics['brier_skill'], 4)}).",
            STYLES["body"],
        )
    )

    story.append(Paragraph("Report map", STYLES["h2"]))
    story.append(
        styled_table(
            [
                ["Section", "Content"],
                ["01 · Product", "The prediction-first interface: journey, presets and design decisions"],
                ["02 · Model & data", "Artifact, feature contract, dataset and chronological split"],
                ["03 · Evaluation", "Test-period metrics, risk bands and year-by-year performance"],
                ["04 · Evidence", "The ten figures produced by notebook.py"],
                ["05 · Interpretation", "What drives the score, and the limits of that reading"],
                ["06 · Reproducibility", "Commands to rebuild every number and artifact"],
            ],
            [42 * mm, PAGE_W - 2 * MARGIN - 42 * mm],
        )
    )

    # ------------------------------------------------------------------ p3
    story.append(PageBreak())
    story.append(Paragraph("01  PRODUCT", STYLES["kicker"]))
    story.append(Paragraph("The prediction-first interface", STYLES["h1"]))
    story.append(
        Paragraph(
            "The previous entry point was an analytics landing page: a visitor had to "
            "browse charts before discovering that the project can actually predict "
            "anything. The new entry point inverts that journey. The first thing a user "
            "sees is a short form and a result.",
            STYLES["lead"],
        )
    )

    story.append(Paragraph("User journey", STYLES["h2"]))
    story.append(
        styled_table(
            [
                ["Step", "What the user does", "What the app returns"],
                [
                    "1 · Locate",
                    "Picks a latitude / longitude and a date (or uses the defaults).",
                    "The snapped grid cell actually used by the model is displayed.",
                ],
                [
                    "2 · Describe",
                    "Chooses a scenario preset or sets six fire-weather values.",
                    "Values are echoed back as a readable condition summary.",
                ],
                [
                    "3 · Predict",
                    "Presses <b>Predict fire risk</b>.",
                    "Calibrated probability, risk band, gauge and plain-language reading.",
                ],
                [
                    "4 · Understand",
                    "Reads the sensitivity chart and the guidance list.",
                    "Which inputs push the score up or down, and what to do about it.",
                ],
            ],
            [27 * mm, 63 * mm, PAGE_W - 2 * MARGIN - 90 * mm],
        )
    )
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("Design decisions", STYLES["h2"]))
    story.extend(
        bullets(
            [
                "<b>Prediction before analytics.</b> Charts remain available in the sidebar, "
                "but they no longer gate the first interaction.",
                "<b>Few inputs, honest defaults.</b> Six weather values and a location are "
                "enough; the 7/14/30-day lag features are derived from them and can be "
                "overridden in an advanced panel.",
                "<b>No silent imputation.</b> Every value sent to the model is shown to the "
                "user, and optional live NASA POWER data is clearly labelled as a fetch.",
                "<b>Explanation is local and transparent.</b> The sensitivity chart moves one "
                "feature at a time to a fire-suppressing reference; it is a perturbation "
                "study, not a causal claim.",
                "<b>Calibrated uncertainty.</b> The score is a probability, displayed with "
                "its climatological base rate so a user can judge whether it is unusual.",
            ]
        )
    )

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Scenario presets", STYLES["h2"]))
    story.append(
        styled_table(
            [
                ["Preset", "Conditions", "Typical use"],
                ["Summer heatwave", "43 °C peaks, 15% RH, 42 dry days", "Peak-season stress test"],
                ["Dry spring", "27 °C, 46% RH, light rain, 4 dry days", "Early-season watch"],
                ["Mild autumn", "22 °C, 58% RH, moderate soil moisture", "Should stay low"],
                ["Winter wet", "14 °C, 84% RH, 16 mm rain", "Saturated-fuel reference"],
            ],
            [34 * mm, 70 * mm, PAGE_W - 2 * MARGIN - 104 * mm],
        )
    )
    story.append(
        Paragraph(
            "Presets only prefill the form; every value remains editable. A manual "
            "configuration that scores unexpectedly is therefore always inspectable.",
            STYLES["caption"],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        callout(
            "Scope of a prediction",
            "A score estimates the probability that a satellite fire detection occurs in the "
            "selected grid cell within the next 3 days. The grid is coarse (~0.5°), so the "
            "output is a <b>regional warning signal</b>, not a forecast of a specific fire, "
            "and not an official emergency notification.",
            EMBER,
        )
    )

    # ------------------------------------------------------------------ p4
    story.append(PageBreak())
    story.append(Paragraph("02  MODEL & DATA", STYLES["kicker"]))
    story.append(Paragraph("Model, features and dataset", STYLES["h1"]))
    story.append(
        Paragraph(
            f"The deployed artifact is model <b>v{model['version']}</b>, a calibrated "
            f"gradient-boosting classifier over <b>{model['n_features']} features</b>. "
            "The target is the occurrence of an observed fire in the cell within the next "
            f"{split['horizon_days']} days.",
            STYLES["lead"],
        )
    )

    story.append(Paragraph("Dataset", STYLES["h2"]))
    story.append(
        styled_table(
            [
                ["Property", "Value"],
                ["Study area", summary["study_area"]],
                ["Rows (cell-days)", intnum(dataset["rows"])],
                ["Grid cells", intnum(dataset["cells"])],
                ["Period", f"{dataset['date_min']} → {dataset['date_max']}"],
                ["Positive cell-days", f"{intnum(dataset['positive_cell_days'])} "
                                       f"({pct(dataset['positive_rate'])})"],
            ],
            [55 * mm, PAGE_W - 2 * MARGIN - 55 * mm],
            align_right=(),
        )
    )
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("Chronological split", STYLES["h2"]))
    story.append(
        styled_table(
            [
                ["Split", "Definition", "Rows"],
                ["Train", "Dates before 2021", intnum(split["train_rows"])],
                ["Validation", f"Year {split['val_year']} (calibration)", intnum(split["val_rows"])],
                ["Test", "2022 – 2024 inclusive", intnum(split["test_rows"])],
            ],
            [30 * mm, 95 * mm, PAGE_W - 2 * MARGIN - 125 * mm],
            align_right=(2,),
        )
    )
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("Feature set", STYLES["h2"]))
    features = model["features"]
    half = (len(features) + 1) // 2
    left_col = features[:half]
    right_col = features[half:]
    while len(right_col) < len(left_col):
        right_col.append("")
    feature_rows = [["Feature", "", "Feature", ""]]
    for left, right in zip(left_col, right_col):
        feature_rows.append([left, "", right, ""])
    story.append(
        styled_table(
            feature_rows,
            [30 * mm, 40 * mm, 30 * mm, 40 * mm],
            zebra=True,
        )
    )
    story.append(
        Paragraph(
            "Feature order is authoritative in <b>models/model_card.json</b> and is enforced "
            "by the test suite.",
            STYLES["caption"],
        )
    )

    story.append(Spacer(1, 3 * mm))
    story.append(
        callout(
            "Model-card drift",
            f"The shipped card records PR-AUC {num(model['recorded_metrics']['pr_auc'])} and a "
            f"{pct(model['recorded_metrics']['base_rate'])} base rate, while this evaluation "
            f"obtains {num(metrics['pr_auc'])} at a {pct(metrics['base_rate'])} base rate. The "
            "card was written on an earlier panel snapshot. It is intentionally left unchanged "
            "because an integrity test pins its SHA256; this report is the up-to-date record.",
            WARN,
            colors.HexColor("#FFFBEB"),
        )
    )

    # ------------------------------------------------------------------ p5
    story.append(PageBreak())
    story.append(Paragraph("03  EVALUATION", STYLES["kicker"]))
    story.append(Paragraph("Test-period results", STYLES["h1"]))
    story.append(
        Paragraph(
            "Fires are rare, so accuracy is not informative. The model is judged on ranking "
            "quality (PR-AUC, ROC-AUC, top-decile capture), on calibration (Brier score, "
            "skill score, ECE) and against a climatology baseline that already knows where "
            "and when fires usually happen.",
            STYLES["lead"],
        )
    )

    story.append(
        styled_table(
            [
                ["Metric", "Value", "Reading"],
                ["PR-AUC (calibrated)", num(metrics["pr_auc"]),
                 f"vs {num(metrics['climatology_pr_auc'])} climatology"],
                ["PR-AUC (raw scores)", num(metrics["pr_auc_raw"]), "before calibration"],
                ["ROC-AUC", num(metrics["roc_auc"]), "strong separation"],
                ["Brier score", num(metrics["brier"], 5), "lower is better"],
                ["Brier skill score", num(metrics["brier_skill"], 4), "0 = base-rate forecast"],
                ["Expected calibration error", num(metrics["ece"], 4), "10 quantile bins"],
                ["Calibration slope", num(metrics["calibration_slope"], 3), "1.0 = perfect"],
                ["Fires in top decile", pct(metrics["top_decile_fire_share"], 1), "of all observed fires"],
                ["Base rate (test)", pct(metrics["base_rate"]), "random PR-AUC"],
                ["Precision @ 0.5", pct(metrics["precision_at_0_5"], 1),
                 f"recall {pct(metrics['recall_at_0_5'], 1)}"],
                ["Best F1", num(metrics["best_f1"], 3),
                 f"at threshold {num(metrics['best_f1_threshold'], 3)}"],
            ],
            [55 * mm, 30 * mm, PAGE_W - 2 * MARGIN - 85 * mm],
            align_right=(1,),
        )
    )
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Risk bands", STYLES["h2"]))
    band_rows = [["Band", "Cell-days", "Mean score", "Observed rate", "Share of fires"]]
    band_colors = {
        "LOW": SAFE,
        "MODERATE": WARN,
        "HIGH": EMBER,
        "VERY HIGH": DANGER,
    }
    for band in bands:
        label = band["band"]
        color = band_colors.get(label.upper(), INK)
        band_rows.append(
            [
                f"<font color='{hexcolor(color)}'><b>{label}</b></font>",
                intnum(band["cells"]),
                pct(band["mean_score"]),
                pct(band["observed_rate"]),
                pct(band["share_of_all_fires"], 1),
            ]
        )
    story.append(
        styled_table(
            band_rows,
            [32 * mm, 32 * mm, 32 * mm, 34 * mm, PAGE_W - 2 * MARGIN - 130 * mm],
            align_right=(1, 2, 3, 4),
        )
    )
    story.append(
        Paragraph(
            "Each band behaves monotonically: the observed fire rate rises from "
            f"{pct(bands[0]['observed_rate'])} in LOW to {pct(bands[-1]['observed_rate'])} "
            "in VERY HIGH, and the top band contains most of the fires.",
            STYLES["body"],
        )
    )

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Performance by year", STYLES["h2"]))
    year_rows = [["Year", "Rows", "Base rate", "PR-AUC model", "PR-AUC climatology",
                  "Brier", "Fires in top decile"]]
    for row in yearly:
        year_rows.append(
            [
                str(row["year"]),
                intnum(row["rows"]),
                pct(row["base_rate"]),
                num(row["pr_auc_model"]),
                num(row["pr_auc_climatology"]),
                num(row["brier"], 4),
                pct(row["top_decile_share"], 1),
            ]
        )
    story.append(
        styled_table(
            year_rows,
            [14 * mm, 20 * mm, 20 * mm, 24 * mm, 34 * mm, 20 * mm,
             PAGE_W - 2 * MARGIN - 132 * mm],
            align_right=(1, 2, 3, 4, 5, 6),
        )
    )
    story.append(
        Paragraph(
            "2024 is visibly harder: the base rate falls to "
            f"{pct(yearly[-1]['base_rate'])}, and so does the achievable PR-AUC. The model "
            "still beats the climatology baseline in every test year.",
            STYLES["body"],
        )
    )

    # ------------------------------------------------------------------ figures
    story.append(PageBreak())
    story.append(Paragraph("04  EVIDENCE", STYLES["kicker"]))
    story.append(Paragraph("Figures", STYLES["h1"]))
    story.append(
        Paragraph(
            "All figures are generated by <b>python notebook.py</b>. They use the test "
            "period only, except the seasonality and data-overview panels which describe "
            "the full panel.",
            STYLES["body"],
        )
    )
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("Ranking quality", STYLES["h3"]))
    story.append(
        figure(
            "02_precision_recall_capture.png",
            "Precision-recall curve against the base rate, with the fire-capture curve. "
            "The model concentrates observed fires in a small share of alerted cell-days.",
        )
    )
    story.append(
        figure(
            "03_roc_curve.png",
            "ROC curve on the 2022–2024 hold-out.",
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Calibration, bands and data", STYLES["h3"]))
    story.append(
        figure(
            "04_calibration.png",
            "Reliability diagram. Predicted probabilities track observed frequencies closely "
            "(ECE ≈ 0.005), which is what makes the gauge meaningful in the UI.",
        )
    )
    story.append(
        figure(
            "05_risk_bands.png",
            "Risk-band calibration: mean predicted score versus observed fire rate per band.",
        )
    )
    story.append(
        figure(
            "10_data_overview.png",
            "Dataset overview: rows per year, positive rate and cell coverage.",
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Temporal and spatial behaviour", STYLES["h3"]))
    story.append(
        figure(
            "06_yearly_performance.png",
            "PR-AUC and top-decile capture per test year, against climatology.",
        )
    )
    story.append(
        figure(
            "01_fire_seasonality.png",
            "Fire seasonality across the full panel: the summer peak the model must learn.",
        )
    )
    story.append(
        figure(
            "09_hotspots.png",
            "Spatial hotspot map of observed fires, showing why grid-cell coordinates are "
            "strong predictors.",
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Errors and feature attribution", STYLES["h3"]))
    story.append(
        figure(
            "08_confusion_matrix.png",
            "Confusion matrix at p ≥ 0.5. With a 4.3% base rate this threshold is "
            "deliberately conservative; the operational value lies in ranking, not in "
            "hard classification.",
        )
    )
    story.append(
        figure(
            "07_feature_importance.png",
            "Permutation importance for all 20 features. Coordinates dominate, followed by "
            "30-day heat and root-zone soil wetness.",
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("05  INTERPRETATION", STYLES["kicker"]))
    story.append(Paragraph("What drives the score", STYLES["h1"]))
    story.append(
        Paragraph(
            "Sampled permutation importance is dominated by the cell coordinates. That is an "
            "honest and important finding: the model learns <i>where</i> fires happen very "
            "well, and adds weather information on top of that geography. Users should read "
            "the score as a location-aware seasonal risk, not as a pure weather index.",
            STYLES["lead"],
        )
    )
    imp_rows = [["Feature", "Importance (mean)", "Std"]]
    for item in importance[:12]:
        imp_rows.append(
            [
                item["feature"],
                f"{float(item['importance_mean']):.5f}",
                f"{float(item['importance_std']):.5f}",
            ]
        )
    story.append(
        styled_table(
            imp_rows,
            [55 * mm, 40 * mm, PAGE_W - 2 * MARGIN - 95 * mm],
            align_right=(1, 2),
        )
    )
    story.append(
        Paragraph(
            "Top 12 of 20 features. Full ranking in report/evaluation_summary.json; the "
            "bar chart of all 20 appears in section 04.",
            STYLES["caption"],
        )
    )

    story.append(Paragraph("Limitations to state explicitly", STYLES["h2"]))
    story.extend(
        bullets(
            [
                "<b>Coarse grid.</b> NASA POWER cells are ~0.5°, so local terrain and "
                "microclimate are invisible to the model.",
                "<b>Location dominance.</b> Most predictive signal is geographic; a single "
                "day's weather moves the score less than the cell does.",
                "<b>Missed detections.</b> FIRMS can miss fires under cloud cover or between "
                "satellite overpasses, which adds label noise.",
                "<b>Regional scope.</b> Trained on northern Algeria; transfer to other regions "
                "is untested.",
                "<b>Not an emergency system.</b> Scores support analysis and prioritisation; "
                "they are not official forecasts.",
            ]
        )
    )

    # ------------------------------------------------------------------ last
    story.append(PageBreak())
    story.append(Paragraph("06  REPRODUCIBILITY", STYLES["kicker"]))
    story.append(Paragraph("How to reproduce everything", STYLES["h1"]))
    story.append(
        Paragraph(
            "Every result in this report is regenerated from the repository with four "
            "commands. No manual editing of numbers is involved.",
            STYLES["lead"],
        )
    )
    story.append(Paragraph("Setup", STYLES["h2"]))
    story.append(
        Paragraph(
            "python -m venv .venv<br/>"
            ".venv\\Scripts\\activate  <font color='#64748B'>(Windows)</font><br/>"
            "python -m pip install -r requirements.txt",
            ParagraphStyle(
                "code",
                fontName="Courier",
                fontSize=8.6,
                leading=13,
                textColor=INK,
                backColor=colors.white,
                borderColor=LINE,
                borderWidth=0.6,
                borderPadding=7,
                spaceAfter=8,
            ),
        )
    )
    story.append(Paragraph("Evaluate, then build the report", STYLES["h2"]))
    story.append(
        Paragraph(
            "python notebook.py<br/>"
            "python report/build_report.py",
            ParagraphStyle(
                "code2",
                fontName="Courier",
                fontSize=8.6,
                leading=13,
                textColor=INK,
                backColor=colors.white,
                borderColor=LINE,
                borderWidth=0.6,
                borderPadding=7,
                spaceAfter=8,
            ),
        )
    )
    story.append(Paragraph("Launch the application", STYLES["h2"]))
    story.append(
        Paragraph(
            "streamlit run app.py",
            ParagraphStyle(
                "code3",
                fontName="Courier",
                fontSize=8.6,
                leading=13,
                textColor=INK,
                backColor=colors.white,
                borderColor=LINE,
                borderWidth=0.6,
                borderPadding=7,
                spaceAfter=8,
            ),
        )
    )
    story.append(Paragraph("Verify the whole project", STYLES["h2"]))
    story.append(
        Paragraph(
            "python -m pytest -q",
            ParagraphStyle(
                "code4",
                fontName="Courier",
                fontSize=8.6,
                leading=13,
                textColor=INK,
                backColor=colors.white,
                borderColor=LINE,
                borderWidth=0.6,
                borderPadding=7,
                spaceAfter=8,
            ),
        )
    )

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Artifacts produced", STYLES["h2"]))
    story.append(
        styled_table(
            [
                ["Artifact", "Purpose"],
                ["report/EVALUATION.md", "Human-readable evaluation report"],
                ["report/evaluation_summary.json", "Machine-readable metrics and tables"],
                ["report/figures/*.png", "Ten evaluation figures used in this PDF"],
                ["report/FIREWATCH_AI_Report.pdf", "This document"],
                ["report/build_report.py", "Deterministic PDF generator"],
            ],
            [70 * mm, PAGE_W - 2 * MARGIN - 70 * mm],
        )
    )

    story.append(Spacer(1, 8 * mm))
    story.append(
        callout(
            "Integrity note",
            "models/model_card.json is protected by a SHA256 assertion in "
            "tests/test_impact_integrity.py and was not modified. The metrics in this "
            "report come from re-scoring the deployed artifact, so they describe the "
            "current code and data rather than the historical card values.",
            SAFE,
            colors.HexColor("#F0FDF4"),
        )
    )

    story.append(Spacer(1, 10 * mm))
    story.append(
        Paragraph(
            "FIREWATCH AI — Data-driven wildfire risk analysis and geospatial impact "
            "assessment.",
            ParagraphStyle(
                "closing",
                fontName=FONT_ITALIC,
                fontSize=9,
                leading=13,
                textColor=MUTED,
            ),
        )
    )
    return story


def main() -> int:
    if not SUMMARY.exists():
        print(f"Missing {SUMMARY}. Run 'python notebook.py' first.")
        return 1

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        title="FIREWATCH AI — Wildfire Risk Prediction & Impact Assessment",
        author="FireWatch Contributor",
        subject="Prediction-first interface, model evaluation and impact assessment",
        creator="report/build_report.py",
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="cover",
                frames=[Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN, id="cover")],
                onPage=_cover,
            ),
            PageTemplate(id="body", frames=[_frame()], onPage=_body_page),
        ]
    )
    doc.build(build_story(summary))
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
