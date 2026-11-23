"""Build the expanded scientific project report at docs/Rapport_Projet.pdf.

This builder reuses the visual system and layout helpers from
``report/build_report.py`` and reads only existing artifacts.  It never writes
``report/FIREWATCH_AI_Report.pdf`` and never touches models, data or results.
All numbers and figures come from files already present in the repository.
"""

from __future__ import annotations

import csv
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
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
DOCS_DIR = ROOT / "docs"
OUTPUT = DOCS_DIR / "Rapport_Projet.pdf"
if str(REPORT_DIR) not in sys.path:
    sys.path.insert(0, str(REPORT_DIR))

import build_report as br  # noqa: E402

PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 2 * br.MARGIN


def esc(value: Any) -> str:
    """Escape dynamic text before it is inserted into a ReportLab paragraph."""
    return html.escape(str(value), quote=False)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark() -> list[dict[str, str]]:
    with (ROOT / "results" / "model_benchmark.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def section(
    story: list[Any], kicker: str, title: str, intro: str | None = None
) -> None:
    story.append(Paragraph(esc(kicker), br.STYLES["kicker"]))
    story.append(Paragraph(esc(title), br.STYLES["h1"]))
    if intro:
        story.append(Paragraph(esc(intro), br.STYLES["lead"]))
    story.append(Spacer(1, 2 * mm))


def paragraph(text: str, style: str = "body") -> Paragraph:
    return Paragraph(esc(text), br.STYLES[style])


def bullets(items: Iterable[str]) -> list[Any]:
    return br.bullets([esc(item) for item in items])


def kv_table(pairs: Iterable[tuple[str, Any]], left: float = 42 * mm) -> Table:
    data = [["Champ", "Valeur"]]
    data.extend([[esc(key), esc(value)] for key, value in pairs])
    return br.styled_table(data, [left, CONTENT_W - left])


def wide_table(
    headers: list[str], rows: Iterable[Iterable[Any]], widths: list[float]
) -> Table:
    data = [[esc(cell) for cell in headers]]
    data.extend([[esc(cell) for cell in row] for row in rows])
    return br.styled_table(data, widths, align_right=tuple(range(1, len(headers))))


def _figure_cell(name: str, caption: str, width: float) -> Table:
    path = br.FIG_DIR / name
    if not path.exists():
        return Table([[Paragraph(f"Missing figure: {name}", br.STYLES["caption"])]])
    iw, ih = ImageReader(str(path)).getSize()
    draw_w = width
    draw_h = draw_w * ih / iw
    max_h = 95 * mm
    if draw_h > max_h:
        draw_h = max_h
        draw_w = draw_h * iw / ih
    cell = Table(
        [
            [Image(str(path), width=draw_w, height=draw_h)],
            [Paragraph(esc(caption), br.STYLES["caption"])],
        ],
        colWidths=[width],
        hAlign="LEFT",
    )
    cell.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    return cell


def figure_pair(left: tuple[str, str], right: tuple[str, str]) -> Table:
    cell_width = CONTENT_W / 2
    table = Table(
        [
            [
                _figure_cell(left[0], left[1], cell_width - 5 * mm),
                _figure_cell(right[0], right[1], cell_width - 5 * mm),
            ]
        ],
        colWidths=[cell_width, cell_width],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    return table


def next_page(story: list[Any]) -> None:
    story.append(PageBreak())


def _cover(canvas, doc) -> None:  # noqa: ANN001 - ReportLab callback signature
    canvas.saveState()
    canvas.setFillColor(br.INK)
    canvas.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#3B1204"))
    canvas.rect(0, PAGE_H - 118 * mm, PAGE_W, 118 * mm, stroke=0, fill=1)
    canvas.setFillColor(br.EMBER)
    canvas.rect(0, PAGE_H - 119 * mm, PAGE_W, 1.2 * mm, stroke=0, fill=1)

    canvas.setFillColor(colors.white)
    canvas.setFont(br.FONT_BOLD, 30)
    canvas.drawString(br.MARGIN, PAGE_H - 42 * mm, "FIREWATCH AI")
    canvas.setFont(br.FONT_BOLD, 20)
    canvas.setFillColor(colors.HexColor("#FDBA74"))
    canvas.drawString(br.MARGIN, PAGE_H - 53 * mm, "Rapport de projet scientifique")
    canvas.setFont(br.FONT, 11)
    canvas.setFillColor(colors.HexColor("#CBD5E1"))
    canvas.drawString(
        br.MARGIN,
        PAGE_H - 63 * mm,
        "Prédiction du risque d'incendie — Algérie du Nord — 2018–2024",
    )
    canvas.setStrokeColor(colors.HexColor("#7C2D12"))
    canvas.line(br.MARGIN, PAGE_H - 69 * mm, PAGE_W - br.MARGIN, PAGE_H - 69 * mm)

    canvas.setFont(br.FONT, 9.5)
    canvas.setFillColor(colors.HexColor("#94A3B8"))
    lines = [
        "Projet académique de data mining et d'analyse géospatiale",
        "Données NASA FIRMS et NASA POWER · modèle étalonné à échéance 3 jours",
        "",
        "Artifacts sources : report/, results/, models/",
        "Généré par : report/build_rapport_projet.py",
    ]
    y = PAGE_H - 79 * mm
    for line in lines:
        canvas.drawString(br.MARGIN, y, line)
        y -= 5.5 * mm

    canvas.setFillColor(colors.HexColor("#111C2E"))
    canvas.rect(0, 0, PAGE_W, 26 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.setFont(br.FONT, 8)
    canvas.drawString(br.MARGIN, 15 * mm, "github.com/CyberSec-017/fire-risk-prediction")
    canvas.drawRightString(PAGE_W - br.MARGIN, 15 * mm, "15–30 pages · document reproductible")
    canvas.restoreState()


def _body_page(canvas, doc) -> None:  # noqa: ANN001 - ReportLab callback signature
    canvas.saveState()
    canvas.setFillColor(br.PAPER)
    canvas.rect(0, 0, PAGE_W, PAGE_H - 10 * mm, stroke=0, fill=1)
    canvas.setStrokeColor(br.LINE)
    canvas.setLineWidth(0.6)
    canvas.line(br.MARGIN, PAGE_H - 14 * mm, PAGE_W - br.MARGIN, PAGE_H - 14 * mm)
    canvas.setFont(br.FONT, 7.4)
    canvas.setFillColor(br.MUTED)
    canvas.drawString(br.MARGIN, PAGE_H - 12 * mm, "FIREWATCH AI — RAPPORT DE PROJET")
    canvas.drawRightString(PAGE_W - br.MARGIN, PAGE_H - 12 * mm, "Risque d'incendie · Algérie du Nord")
    canvas.line(br.MARGIN, 12 * mm, PAGE_W - br.MARGIN, 12 * mm)
    canvas.drawString(br.MARGIN, 8 * mm, "Généré par report/build_rapport_projet.py")
    canvas.drawRightString(PAGE_W - br.MARGIN, 8 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def build_story(
    summary: dict[str, Any],
    stats: dict[str, Any],
    protocol: dict[str, Any],
    card: dict[str, Any],
    benchmark: list[dict[str, str]],
) -> list[Any]:
    metrics = summary["metrics"]
    dataset = summary["dataset"]
    split = summary["split"]
    bands = summary["risk_bands"]
    yearly = summary["yearly"]
    importance = summary["permutation_importance"]
    stats_dataset = stats["dataset"]
    primary = [row for row in benchmark if row["experiment"] == "primary"]
    primary.sort(key=lambda row: float(row["pr_auc"]), reverse=True)

    story: list[Any] = [NextPageTemplate("body"), PageBreak()]

    # ---------------------------------------------------------------- p2
    section(
        story,
        "Résumé exécutif",
        "Synthèse du projet et des preuves produites",
        "Ce rapport rassemble les livrables reproductibles déjà présents dans le dépôt : "
        "évaluation du modèle déployé, tests statistiques, benchmark, exploration de données, "
        "architecture applicative et limites. Aucun résultat expérimental n'est inventé.",
    )
    story.append(
        br.metric_card_row(
            [
                ("PR-AUC (test)", br.num(metrics["pr_auc"]), br.EMBER),
                ("ROC-AUC (test)", br.num(metrics["roc_auc"]), br.INK),
                ("Brier (test)", br.num(metrics["brier"], 5), br.GOLD),
                ("Part des feux au top décile", br.pct(metrics["top_decile_fire_share"], 1), br.DANGER),
            ],
            CONTENT_W,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        br.metric_card_row(
            [
                ("Taux de base", br.pct(metrics["base_rate"]), br.INK),
                ("ECE", br.num(metrics["ece"], 4), br.SAFE),
                ("Lignes de panel", br.intnum(dataset["rows"]), br.EMBER),
                ("Cellules actives", br.intnum(dataset["cells"]), br.GOLD),
            ],
            CONTENT_W,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.extend(
        bullets(
            [
                "Le modèle de production est un classifieur calibré à 20 variables, "
                "entraîné avec une séparation chronologique et une échéance de 3 jours.",
                "L'évaluation reproductible porte sur 2022–2024 et diffère des métriques "
                "enregistrées dans la fiche modèle ; cette dérive est documentée.",
                "Quatre tests descriptifs et un benchmark à découpage commun ont été ajoutés "
                "sans modifier les artefacts de production.",
                "L'interface propose désormais une inférence par CSV avec validation stricte "
                "et explicabilité prudente, sans dépendance SHAP obligatoire.",
            ]
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        kv_table(
            [
                ("Problématique", "Estimer le risque d'observation d'un feu à H = 3 jours sur une cellule géographique."),
                ("Données", "NASA POWER (météo), NASA FIRMS (détections), couches géospatiales optionnelles."),
                ("Méthode", "Panel cellule-jour, fenêtres historiques, gradient boosting étalonné, validation temporelle."),
                ("Livrables", "Application Streamlit, modèle gelé, métriques, figures, tests et documentation."),
            ]
        )
    )

    # ---------------------------------------------------------------- p3
    next_page(story)
    section(story, "Section 1", "Problématique et objectifs", "Pourquoi modéliser un risque d'incendie à échéance courte ?")
    story.extend(
        bullets(
            [
                "Les incendies de forêt dépendent de conditions météorologiques, de la sécheresse "
                "des sols et de facteurs spatiaux qui varient à l'échelle journalière.",
                "Les événements sont rares : la précision brute et l'accuracy sont des critères "
                "trompeurs sans référence au taux de base.",
                "L'objectif opérationnel n'est pas de certifier un départ de feu, mais de classer "
                "des cellules par niveau de risque afin d'orienter la surveillance.",
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        wide_table(
            ["Objectif", "Description", "Livrable"],
            [
                ["Acquisition", "Collecter météo et détections satellite", "src/fetch_power.py, src/clean_firms.py"],
                ["Préparation", "Aligner les observations sur une grille commune", "data/processed/fire_panel.parquet"],
                ["Features", "Construire des indicateurs antérieurs sans fuite", "src/build_panel.py"],
                ["Modélisation", "Entraîner et étalonner un classifieur", "models/fire_model.joblib"],
                ["Évaluation", "Produire des métriques et figures reproductibles", "report/, results/"],
                ["Interface", "Explorer, prédire et exporter les résultats", "app.py, ui_v2/, dashboard/"],
                ["Impact", "Estimer une exposition relative, pas un dommage", "impact/, data/impact/"],
            ],
            [32 * mm, 78 * mm, CONTENT_W - 110 * mm],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        br.callout(
            "Périmètre du risque",
            "La cible est la présence d'une détection FIRMS dans une cellule-jour sur l'horizon "
            "de 3 jours. Ce n'est ni une surface brûlée confirmée, ni un dégât matériel, ni une "
            "alerte officielle.",
            br.EMBER,
        )
    )

    # ---------------------------------------------------------------- p4
    next_page(story)
    section(story, "Section 2", "Sources de données et zone d'étude", "Sources déjà utilisées par le pipeline et l'application.")
    story.append(
        wide_table(
            ["Source", "Rôle", "Utilisation dans le projet"],
            [
                ["NASA FIRMS", "Détections de feux actifs", "Cible cellule-jour et analyses de points chauds"],
                ["NASA POWER", "Réanalyses météorologiques", "T2M, T2M_MAX, RH2M, WS2M, PRECTOTCORR, GWETROOT"],
                ["WorldPop", "Population", "Exposition potentielles dans le module impact"],
                ["OpenStreetMap", "Routes et bâtiments", "Densité d'infrastructure et settlements"],
                ["ESA WorldCover", "Occupation du sol", "Couche géospatiale complémentaire"],
                ["GHSL", "Habitat humain", "Indicateurs d'exposition"],
                ["OCHA / HDX", "Données humanitaires", "Infrastructures et limites administratives"],
            ],
            [35 * mm, 50 * mm, CONTENT_W - 85 * mm],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        kv_table(
            [
                ("Zone", "Nord de l'Algérie"),
                ("Latitude", "32.0°N à 37.5°N"),
                ("Longitude", "-2.0°E à 9.0°E"),
                ("Grille", "0.5° latitude × 0.625° longitude (pas NASA POWER)"),
                ("Période", "2018-03-02 à 2024-12-28"),
                ("Panel", f"{br.intnum(dataset['rows'])} cellules-jours, {br.intnum(dataset['cells'])} cellules"),
                ("Taux de positifs", br.pct(dataset["positive_rate"])),
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        paragraph(
            "Les six variables météo sont collectées sur la grille complète ; FIRMS fournit les "
            "labels. Les autres jeux ne sont pas utilisés par le modèle de production et restent "
            "confinés au module d'impact ou aux couches optionnelles."
        )
    )

    # ---------------------------------------------------------------- p5
    next_page(story)
    section(story, "Section 3", "Préprocessing et construction du panel", "Une chaîne déterministe, étape par étape.")
    story.append(
        wide_table(
            ["Étape", "Script", "Opérations", "Sortie"],
            [
                ["1", "src/fetch_power.py", "Requêtes tuilées, cache des téléchargements", "data/raw/"],
                ["2", "src/clean_firms.py", "Filtrage qualité, alignement spatial, agrégation journalière", "data/interim/fires_daily.parquet"],
                ["3", "src/build_panel.py", "Jonction grille/météo/labels, features 7/14/30 jours", "data/processed/fire_panel.parquet"],
                ["4", "src/train_baseline.py", "Entraînement, étalonnage, contrat 20 features", "models/fire_model.joblib"],
                ["5", "report/notebook.py", "Split chronologique, métriques, figures", "report/, results/"],
            ],
            [12 * mm, 38 * mm, 72 * mm, CONTENT_W - 122 * mm],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.extend(
        bullets(
            [
                "Une ligne du panel correspond à une cellule et une date : les négatifs sont "
                "donc explicites, et pas seulement les jours avec détection.",
                "Les fenêtres glissantes de 7, 14 et 30 jours utilisent uniquement des "
                "observations antérieures au jour prédit.",
                "Les transformations saisonnières doy_sin et doy_cos encodent la cyclicité "
                "de l'année sans introduire de date future.",
                "La séparation chronologique et la purge de 3 jours réduisent le risque de "
                "fuite à la frontière entre partitions.",
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        br.callout(
            "Valeurs manquantes",
            "Le HistGradientBoostingClassifier gère nativement les NaN. Cependant, l'inférence "
            "par CSV rejette les lignes incomplètes ou invalides : le modèle n'a pas à deviner "
            "une valeur qui n'a pas été fournie.",
            br.SAFE,
        )
    )

    # ---------------------------------------------------------------- p6
    next_page(story)
    section(story, "Section 3 (suite)", "Les 20 variables du contrat de production", "Ordre autoritatif : models/model_card.json.")
    story.append(
        wide_table(
            ["Groupe", "Variables", "Rôle"],
            [
                ["Localisation", "lat_cell, lon_cell", "Position sur la grille NASA POWER"],
                ["Météo quotidienne", "T2M, T2M_MAX, RH2M, WS2M, PRECTOTCORR, GWETROOT", "État du jour"],
                ["Historique 7 j", "rain_7d, tmax_7d, rhmin_7d", "Conditions récentes"],
                ["Historique 14 j", "rain_14d, tmax_14d, rhmin_14d", "Tendance intermédiaire"],
                ["Historique 30 j", "rain_30d, tmax_30d, rhmin_30d", "Stress hydrique prolongé"],
                ["Sécheresse", "dry_days", "Nombre de jours sans précipitation"],
                ["Saisonnalité", "doy_sin, doy_cos", "Cycle annuel continu"],
            ],
            [34 * mm, 82 * mm, CONTENT_W - 116 * mm],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        kv_table(
            [
                ("n_features", card["n_features"]),
                ("feature_hash", card["feature_hash"]),
                ("Ordre des colonnes", " → ".join(card["feature_names"][:5]) + " → …"),
                ("Préprocessing", "Aucun Pipeline/ColumnTransformer : les 20 colonnes brutes sont les entrées du modèle."),
                ("Valeurs manquantes", "Préservées dans le scoring ponctuel, rejetées explicitement en inférence CSV."),
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        br.callout(
            "Contrat gelé",
            "Les 20 noms, leur ordre, les bornes de risque et le comportement de prédiction en "
            "production ne sont pas modifiés par les modules analytiques ou par le lot CSV.",
            br.EMBER,
        )
    )

    # ---------------------------------------------------------------- p7
    next_page(story)
    section(story, "Section 4", "Analyse exploratoire des données", "Figures et contrôles de provenance.")
    story.append(
        br.figure(
            "10_data_overview.png",
            "Figure 1 — Distribution de T2M_MAX et cumul mobile de précipitations sur 30 jours "
            "(figure générée par report/notebook.py).",
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.extend(
        bullets(
            [
                "La distribution de température et la série de pluie mobile décrivent le "
                "régime météorologique du panel, sans utiliser de sortie du modèle.",
                "src/eda.py enregistre 15 visualisations réutilisables, avec des règles de "
                "provenance explicites pour distinguer observations, features et prédictions.",
                "Les mêmes principes sont exposés dans l'interface « Données & EDA » et dans "
                "la vue de qualité, qui vérifient les jeux avant exploration.",
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        kv_table(
            [
                ("Période EDA", f"{stats_dataset['date_min']} à {stats_dataset['date_max']}"),
                ("Variables météo testées", stats_dataset["tested_variable"]["label"] + " (" + stats_dataset["tested_variable"]["unit"] + ")"),
                ("Proxy FIRMS", "cellule-jour avec au moins une détection"),
                ("Cellules", br.intnum(stats_dataset["cells"])),
            ]
        )
    )

    # ---------------------------------------------------------------- p8
    next_page(story)
    section(story, "Section 4 (suite)", "Saisonnalité et description du panel", "Une lecture descriptive avant modélisation.")
    story.append(
        br.figure(
            "01_fire_seasonality.png",
            "Figure 2 — Détections mensuelles et taux de détection sur la période de test "
            "2022–2024 (cap de la figure d'origine, pas l'ensemble du panel).",
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        wide_table(
            ["Indicateur", "Valeur"],
            [
                ["Lignes du panel", br.intnum(dataset["rows"])],
                ["Cellules", br.intnum(dataset["cells"])],
                ["Début", dataset["date_min"]],
                ["Fin", dataset["date_max"]],
                ["Cellules-jours positives", br.intnum(dataset["positive_cell_days"])],
                ["Taux positif", br.pct(dataset["positive_rate"])],
            ],
            [70 * mm, CONTENT_W - 70 * mm],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        paragraph(
            "La figure montre une structure saisonnière et spatia-temporelle ; elle ne démontre "
            "pas une causalité climatique. Les tests statistiques de la section suivante "
            "quantifient certaines associations observées avec leurs hypothèses et limites."
        )
    )

    # ---------------------------------------------------------------- p9
    next_page(story)
    section(story, "Section 5", "Validation statistique — protocole et deux premiers tests", "Tests descriptifs sur les observations historiques.")
    story.append(
        kv_table(
            [
                ("Alpha", stats["protocol"]["alpha"]),
                ("Graine", stats["protocol"]["seed"]),
                ("Plafond Shapiro", br.intnum(stats["protocol"]["shapiro_sample_cap"])),
                ("Périmètre", stats["protocol"]["scope"]),
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))

    def test_block(test: dict[str, Any]) -> list[Any]:
        effect = test.get("effect_size") or {}
        effect_text = (
            f"{effect.get('name')} = {br.num(effect.get('value'), 4)} ({effect.get('magnitude')})"
            if effect
            else "Non applicable"
        )
        rows = [
            ["Question", test["question"]],
            ["H0", test["null_hypothesis"]],
            ["H1", test["alternative_hypothesis"]],
            ["Statistique", br.num(test["statistic"], 4)],
            ["p-valeur", test["p_display"]],
            ["Taille d'effet", effect_text],
            ["Interprétation", test["interpretation"]],
        ]
        return [Paragraph(esc(test["title"]), br.STYLES["h2"]), kv_table(rows), Spacer(1, 2 * mm)]

    for item in stats["tests"][:2]:
        story.extend(test_block(item))
    story.append(
        br.callout(
            "Lecture correcte",
            "Une p-valeur très faible indique une incompatibilité avec H0 sous les hypothèses "
            "du test. Elle ne prouve ni une cause, ni une performance prédictive, ni une "
            "généralisation opérationnelle.",
            br.WARN,
        )
    )

    # ---------------------------------------------------------------- p10
    next_page(story)
    section(story, "Section 5 (suite)", "Tests de normalité, de groupes et limites", "Les tests de rangs sont privilégiés après le test de normalité.")
    for item in stats["tests"][2:]:
        story.extend(test_block(item))
    story.append(Paragraph("Limites déclarées dans le rapport statistique", br.STYLES["h2"]))
    story.extend(bullets(stats["limitations"]))
    story.append(Spacer(1, 2 * mm))
    story.append(
        br.callout(
            "Définitions distinctes",
            "Le proxy statistique « burned » compte 11 247 cellules-jours avec détection, "
            "tandis que la cible du modèle compte 26 135 cellules-jours positifs sur une "
            "construction différente. Ces deux nombres ne sont pas interchangeables.",
            br.DANGER,
        )
    )

    # ---------------------------------------------------------------- p11
    next_page(story)
    section(story, "Section 6", "Sélection de modèles et benchmark", "Comparaison à découpage, features et graine identiques.")
    story.append(
        kv_table(
            [
                ("Split benchmark", "2018–2021 entraînement / 2022 validation"),
                ("Test", "2023–2024 non utilisé par ces expériences"),
                ("Purge", f"{protocol['split']['purge_days']} jours"),
                ("Plis OOF", protocol["split"]["oof_folds"]),
                ("Graine", protocol["seed"]),
                ("Contrat", f"{protocol['model_contract']['n_features']} features, artefacts non modifiés"),
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        wide_table(
            ["Modèle", "PR-AUC", "ROC-AUC", "Brier", "F1 (seuil OOF)"],
            [
                [row["model_label"], br.num(row["pr_auc"]), br.num(row["roc_auc"]), br.num(row["brier_score"], 4), br.num(row["f1_optimal"])]
                for row in primary
            ],
            [52 * mm, 24 * mm, 24 * mm, 24 * mm, CONTENT_W - 124 * mm],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        paragraph(
            "HistGradientBoosting arrive en tête sur le critère principal PR-AUC de cette "
            "comparaison. Le seuil F1 des modèles classés est appris uniquement sur les "
            "prédictions hors-pli de l'entraînement ; la validation 2022 reste extérieure."
        )
    )

    # ---------------------------------------------------------------- p12
    next_page(story)
    section(story, "Section 6 (suite)", "Interprétation du benchmark et expériences V2", "Ce que le benchmark permet — et ne permet pas — de conclure.")
    story.extend(bullets(protocol["interpretation_notes"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Algorithmes exclus (aucun résultat rapporté)", br.STYLES["h2"]))
    story.append(
        wide_table(
            ["Algorithme", "Raison"],
            [
                [name, info["reason"]]
                for name, info in protocol["algorithms"]["excluded"].items()
            ],
            [35 * mm, CONTENT_W - 35 * mm],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Feature Engineering V2 — validation 2022 uniquement", br.STYLES["h2"]))
    story.append(
        wide_table(
            ["Configuration", "Features", "PR-AUC", "Brier"],
            [
                ["A — Baseline", "20", "0.3103", "0.0396"],
                ["B — VPD", "22", "0.3114", "0.0395"],
                ["C — Anomalies", "23", "0.3118", "0.0393"],
                ["D — Indices", "24", "0.3090", "0.0396"],
                ["E — Voisins", "23", "0.3146", "0.0388"],
                ["F — Toutes familles", "32", "0.3159", "0.0386"],
            ],
            [55 * mm, 25 * mm, 30 * mm, CONTENT_W - 110 * mm],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        br.callout(
            "Garde-fou production",
            "Ces gains sont des résultats hors ligne sur 2022. Le modèle déployé conserve "
            "ses 20 variables tant qu'un cycle complet de validation et d'approbation n'a "
            "pas été réalisé.",
            br.WARN,
        )
    )

    # ---------------------------------------------------------------- p13
    next_page(story)
    section(story, "Section 7", "Méthodologie d'évaluation", "Comment les chiffres de référence sont calculés.")
    story.append(
        kv_table(
            [
                ("Entraînement", f"{br.intnum(split['train_rows'])} lignes, avant {split['val_year']}"),
                ("Validation", f"{br.intnum(split['val_rows'])} lignes, année {split['val_year']}"),
                ("Test", f"{br.intnum(split['test_rows'])} lignes, années {', '.join(map(str, split['test_years']))}"),
                ("Horizon", f"{split['horizon_days']} jours"),
                ("Script", "report/notebook.py"),
                ("Sortie", "report/evaluation_summary.json, report/figures/"),
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        wide_table(
            ["Métrique", "Définition et lecture"],
            [
                ["PR-AUC", "Aire sous la courbe précision-rappel ; la référence aléatoire vaut le taux de base."],
                ["ROC-AUC", "Classement toutes seuils confondus ; peut paraître élevé malgré une faible précision."],
                ["Brier / skill", "Erreur quadratique des probabilités et gain relatif à une référence."],
                ["ECE / pente", "Écart de calibration attendu et raideur de la relation score/observation."],
                ["Top décile", "Part des événements observés capturée dans les 10 % de scores les plus élevés."],
                ["Climatologie", "Référence par cellule et mois, calculée sans utiliser le modèle."],
            ],
            [34 * mm, CONTENT_W - 34 * mm],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        paragraph(
            "La séparation chronologique est celle de l'évaluation publiée : elle diffère du "
            "découpage du benchmark (validation 2022), volontairement isolé pour la comparaison "
            "de modèles. Les deux protocoles sont documentés séparément."
        )
    )

    # ---------------------------------------------------------------- p14
    next_page(story)
    section(story, "Section 8", "Modèle de production", "Artefact gelé et contrat d'inférence.")
    story.append(
        kv_table(
            [
                ("Version", card["model_version"]),
                ("Entraîné le", card["trained_at"]),
                ("n_features", card["n_features"]),
                ("feature_hash", card["feature_hash"]),
                ("Année de validation", card["val_year"]),
                ("Bornes de risque", ", ".join(str(x) for x in card["risk_bins"])),
                ("Libellés", ", ".join(card["risk_labels"])),
                ("Taux de base enregistré", br.pct(card["metrics"]["base_rate"])),
                ("PR-AUC enregistré", br.num(card["metrics"]["pr_auc"])),
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.extend(
        bullets(
            [
                "L'artefact joblib est un dictionnaire dont la clé « model » contient un "
                "CalibratedClassifierCV avec étalonnage isotonique autour d'un "
                "HistGradientBoostingClassifier.",
                "Aucun Pipeline ni ColumnTransformer n'est présent : les 20 colonnes du model "
                "card correspondent directement aux entrées de l'estimateur.",
                "app_core.load_model_bundle valide la présence de predict_proba, le nombre de "
                "variables, leurs noms et l'ordre avant toute inférence.",
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        br.callout(
            "Dérive de la fiche modèle",
            "La fiche enregistre PR-AUC = 0.1582 et un taux de base de 2,37 % sur un ancien "
            "snapshot. La réévaluation reproductible donne PR-AUC = 0.2441 et 4,32 % sur le panel "
            "actuel. La fiche reste inchangée pour préserver l'intégrité du dépôt.",
            br.DANGER,
        )
    )

    # ---------------------------------------------------------------- p15
    next_page(story)
    section(story, "Section 9", "Résultats d'évaluation", "Métriques, matrice de confusion et stabilité annuelle.")
    story.append(
        wide_table(
            ["Métrique", "Valeur", "Métrique", "Valeur"],
            [
                ["PR-AUC", br.num(metrics["pr_auc"]), "ROC-AUC", br.num(metrics["roc_auc"])],
                ["Brier", br.num(metrics["brier"], 5), "Brier skill", br.num(metrics["brier_skill"], 4)],
                ["ECE", br.num(metrics["ece"], 4), "Pente calibration", br.num(metrics["calibration_slope"], 4)],
                ["Taux de base", br.pct(metrics["base_rate"]), "Top décile", br.pct(metrics["top_decile_fire_share"], 1)],
                ["Meilleur F1", br.num(metrics["best_f1"], 4), "Seuil", br.num(metrics["best_f1_threshold"], 3)],
            ],
            [38 * mm, 38 * mm, 44 * mm, CONTENT_W - 120 * mm],
        )
    )
    story.append(Spacer(1, 3 * mm))
    cm = summary["confusion_matrix"]
    story.append(
        wide_table(
            ["VN", "FP", "FN", "VP"],
            [[br.intnum(cm["tn"]), br.intnum(cm["fp"]), br.intnum(cm["fn"]), br.intnum(cm["tp"])]],
            [CONTENT_W / 4] * 4,
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(paragraph("Matrice au seuil 0,5 : peu de positifs prédits, mais une précision élevée parmi eux."))
    story.append(Spacer(1, 3 * mm))
    story.append(
        wide_table(
            ["Année", "Lignes", "Taux base", "PR-AUC modèle", "PR-AUC climato", "Brier", "Top décile"],
            [
                [
                    row["year"], br.intnum(row["rows"]), br.pct(row["base_rate"]),
                    br.num(row["pr_auc_model"]), br.num(row["pr_auc_climatology"]),
                    br.num(row["brier"], 4), br.pct(row["top_decile_share"], 1),
                ]
                for row in yearly
            ],
            [18 * mm, 23 * mm, 22 * mm, 30 * mm, 30 * mm, 22 * mm, CONTENT_W - 145 * mm],
        )
    )

    # ---------------------------------------------------------------- p16
    next_page(story)
    section(story, "Section 9 (suite)", "Bandes de risque et importance globale", "Calibration par catégorie et sensibilité moyenne du modèle.")
    story.append(
        wide_table(
            ["Bande", "Cellules", "Score moyen", "Taux observé", "Positifs", "Part des feux"],
            [
                [
                    row["band"], br.intnum(row["cells"]), br.num(row["mean_score"], 5),
                    br.pct(row["observed_rate"]), br.intnum(row["positives"]),
                    br.pct(row["share_of_all_fires"], 1),
                ]
                for row in bands
            ],
            [25 * mm, 24 * mm, 28 * mm, 28 * mm, 24 * mm, CONTENT_W - 129 * mm],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Top 12 — importance par permutation (Brier négatif, échantillon de test)", br.STYLES["h2"]))
    story.append(
        wide_table(
            ["Rang", "Feature", "Importance moyenne", "Écart-type"],
            [
                [index, row["feature"], br.num(row["importance_mean"], 6), br.num(row["importance_std"], 6)]
                for index, row in enumerate(importance[:12], start=1)
            ],
            [16 * mm, 48 * mm, 55 * mm, CONTENT_W - 119 * mm],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        paragraph(
            "Les coordonnées dominent souvent le classement : le modèle est plus efficace pour "
            "ranger spatialement les zones à risque que pour expliquer un jour isolé par la "
            "seule météo. L'importance par permutation n'est pas une attribution causale."
        )
    )

    # ---------------------------------------------------------------- p17
    next_page(story)
    section(story, "Section 10", "Preuves graphiques — classement", "Figures d'origine générées par report/notebook.py.")
    story.append(
        figure_pair(
            (
                "02_precision_recall_capture.png",
                "Figure 3 — Courbe précision-rappel et capture cumulative sur la période de test.",
            ),
            (
                "03_roc_curve.png",
                "Figure 4 — Courbe ROC sur la période de test.",
            ),
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        paragraph(
            "La courbe précision-rappel est le diagnostic principal pour un événement rare : "
            "la référence aléatoire suit le taux de base. La ROC résume le classement toutes "
            "seuils confondus et peut masquer une faible précision absolue."
        )
    )

    # ---------------------------------------------------------------- p18
    next_page(story)
    section(story, "Section 10 (suite)", "Preuves graphiques — calibration", "Fiabilité des probabilités et comportement des bandes.")
    story.append(
        figure_pair(
            (
                "04_calibration.png",
                "Figure 5 — Courbe de fiabilité et distribution des scores.",
            ),
            (
                "05_risk_bands.png",
                "Figure 6 — Score moyen et taux observé par bande de risque.",
            ),
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        paragraph(
            "L'étalonnage isotonique vise à rendre les probabilités exploitables. L'ECE faible "
            "et les bandes croissantes indiquent une cohérence globale, mais les effectifs "
            "positifs restent limités dans les catégories hautes."
        )
    )

    # ---------------------------------------------------------------- p19
    next_page(story)
    section(story, "Section 10 (suite)", "Preuves graphiques — temps et espace", "Variations annuelles et localisation des détections.")
    story.append(
        figure_pair(
            (
                "06_yearly_performance.png",
                "Figure 7 — PR-AUC et capture par année de test.",
            ),
            (
                "09_hotspots.png",
                "Figure 8 — Points chauds de détections sur 2022–2024.",
            ),
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        paragraph(
            "Les performances ne sont pas constantes d'une année à l'autre. Cette variabilité "
            "doit être prise en compte avant toute utilisation opérationnelle et exclut une "
            "interprétation unique du score comme probabilité de départ de feu confirmé."
        )
    )

    # ---------------------------------------------------------------- p20
    next_page(story)
    section(story, "Section 10 (suite)", "Preuves graphiques — erreurs et features", "Ce que le modèle voit et où il se trompe.")
    story.append(
        figure_pair(
            (
                "07_feature_importance.png",
                "Figure 9 — Top 14 des variables par importance (figure d'origine, pas les 20).",
            ),
            (
                "08_confusion_matrix.png",
                "Figure 10 — Matrice de confusion au seuil 0,5.",
            ),
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        paragraph(
            "La figure d'importance ne couvre que les 14 variables les plus influentes ; le "
            "tableau de la section 9 donne le top 12 chiffré. La matrice illustre le compromis "
            "entre précision et rappel dans un régime très déséquilibré."
        )
    )

    # ---------------------------------------------------------------- p21
    next_page(story)
    section(story, "Section 11", "Architecture applicative", "Un point d'entrée unique, des modules séparés.")
    story.append(
        wide_table(
            ["Couche", "Fichiers", "Responsabilité"],
            [
                ["Entrée", "app.py", "Route landing, cockpit, EDA, statistiques, batch"],
                ["Vue V2", "ui_v2/landing.py, dashboard_view.py, pages/", "Navigation et rendu"],
                ["Logique", "dashboard/, src/", "Scoring, validation, statistiques, pipeline"],
                ["Noyau", "app_core.py", "Contrat modèle, probabilités, bandes de risque"],
                ["Données", "data/processed, data/interim", "Panel et agrégats FIRMS"],
                ["Modèle", "models/fire_model.joblib, model_card.json", "Artefact gelé et métadonnées"],
                ["Preuves", "report/, results/", "Métriques, figures, benchmark, tests"],
            ],
            [30 * mm, 65 * mm, CONTENT_W - 95 * mm],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.extend(
        bullets(
            [
                "L'inférence par lots accepte un CSV, vérifie les 20 colonnes, la numericité, "
                "les plages et l'alignement à la grille, puis score uniquement les lignes valides.",
                "L'export contient les valeurs d'origine, le statut de validation, la probabilité, "
                "la catégorie, l'horizon 3 et des motifs de rejet explicites.",
                "Aucune valeur n'est imputée et aucun fallback DEMO n'est utilisé pour un lot.",
                "Le module d'impact reste séparé : son IEI est une exposition relative, pas une "
                "prédiction de dommage.",
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        paragraph(
            "Commande de lancement : python -m streamlit run app.py. Le port par défaut est 8501 "
            "et peut être modifié par STREAMLIT_SERVER_PORT."
        )
    )

    # ---------------------------------------------------------------- p22
    next_page(story)
    section(story, "Section 12", "Explicabilité — décision documentée", "Pourquoi nous ne produisons pas de waterfall SHAP trompeur.")
    story.append(
        br.callout(
            "Limite SHAP",
            "TreeExplainer ne prend pas en charge CalibratedClassifierCV ni FrozenEstimator. "
            "L'appliquer au HistGradientBoosting interne expliquerait des marges pré-isotoniques, "
            "pas la probabilité calibrée affichée par l'application. Le support HGB de SHAP "
            "repose en outre sur des attributs privés de scikit-learn et sur un appariement "
            "de versions fragile.",
            br.DANGER,
        )
    )
    story.extend(
        bullets(
            [
                "Une approche SHAP model-agnostic pourrait cibler la fonction predict_proba "
                "complète, mais exigerait un jeu de fond représentatif et resterait approximative.",
                "SHAP n'est donc pas ajouté comme dépendance obligatoire : l'application "
                "documente la limite plutôt que d'afficher des attributions non additives aux "
                "probabilités montrées.",
                "L'explication locale disponible est une sensibilité à une variable à la fois : "
                "elle remplace une grandeur par une référence et mesure la variation de probabilité.",
                "L'explication globale repose sur l'importance par permutation calculée sur le "
                "modèle calibré complet, avec un scoring et un échantillon déclarés.",
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        br.callout(
            "Non-causalité",
            "Ni SHAP, ni la sensibilité locale, ni l'importance par permutation n'établissent un "
            "lien causal. Les variables météo et les indicateurs glissants sont corrélés ; une "
            "attribution peut répartir ou masquer leur contribution conjointe.",
            br.WARN,
        )
    )

    # ---------------------------------------------------------------- p23
    next_page(story)
    section(story, "Section 13", "Limites et menaces à la validité", "Ce que le système ne permet pas d'affirmer.")
    limitations = [
        "Résolution spatiale : la grille NASA POWER est relativement grossière et ne capture "
        "pas la microtopographie.",
        "Détections FIRMS : nuages, fréquence de survol et limites du capteur peuvent masquer "
        "des feux réels.",
        "Déséquilibre des classes : la majorité des cellules-jours sont négatives ; les métriques "
        "doivent être lues relativement au taux de base.",
        "Transfert géographique : le modèle est calibré pour le nord de l'Algérie uniquement.",
        "Proxy cible : une détection satellite n'est pas une surface brûlée confirmée.",
        "Dominance spatiale : les coordonnées portent une part importante du signal de classement.",
        "Dérive de la fiche modèle : les métriques stockées et la réévaluation diffèrent.",
        "Stabilité interannuelle : les résultats varient selon l'année de test.",
        "Indépendance statistique : les cellules-jours se répètent dans l'espace et le temps.",
        "Explicabilité : aucune méthode utilisée n'est causale et SHAP n'est pas sûr ici sans "
        "précautions fortes.",
    ]
    story.extend(bullets(limitations))
    story.append(Spacer(1, 3 * mm))
    story.append(
        br.callout(
            "Risque prédit ≠ occurrence observée ou confirmée",
            "Une probabilité élevée signifie que le modèle classe une cellule comme à risque "
            "selon des relations apprises sur des détections passées. Elle ne constitue ni la "
            "constatation d'un feu en cours, ni une confirmation terrain, ni une alerte officielle.",
            br.DANGER,
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.extend(bullets(stats["limitations"]))

    # ---------------------------------------------------------------- p24
    next_page(story)
    section(story, "Section 14", "Conclusion et travaux futurs", "Synthèse et perspectives sans surinterprétation.")
    story.extend(
        bullets(
            [
                "Le projet fournit une chaîne complète et reproductible : collecte, panel, "
                "modèle étalonné, évaluation, interface et documentation.",
                "Le modèle déployé reste utile pour classer spatialement le risque à H = 3 jours, "
                "avec une calibration globale satisfaisante et des limites explicites.",
                "Les ajouts récents — tests statistiques, benchmark homogène, EDA et inférence "
                "CSV — renforcent la traçabilité sans modifier le contrat de production.",
            ]
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Perspectives", br.STYLES["h2"]))
    story.extend(
        bullets(
            [
                "Étendre les variables environnementales et évaluer la configuration V2 F sur "
                "la période de test avant toute promotion.",
                "Améliorer la résolution spatiale et intégrer des produits de surface brûlée indépendants.",
                "Renforcer les stratégies de validation spatiale et temporelle.",
                "Industrialiser l'incertitude et l'analyse de dérive des scores.",
                "Régénérer la fiche modèle lors d'un cycle de déploiement officiel, avec mise à jour "
                "contrôlée des empreintes d'intégrité.",
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        paragraph(
            "Toute évolution du modèle doit conserver une séparation claire entre risque prédit, "
            "détection observée et dommage confirmé."
        )
    )

    # ---------------------------------------------------------------- p25
    next_page(story)
    section(story, "Annexe", "Reproductibilité et inventaire des artefacts", "Commandes et fichiers requis.")
    story.append(Paragraph("Environnement", br.STYLES["h2"]))
    story.extend(
        bullets(
            [
                "Python >= 3.11 ; scikit-learn 1.9.1 et pandas 3.0.6 sont épinglés dans "
                "requirements/requirements.txt pour préserver l'environnement du modèle.",
                "Dépendances application et tests : python -m pip install -r requirements/requirements.txt.",
                "Génération PDF : python -m pip install -r requirements/requirements-report.txt.",
                "Aucune clé API ni secret n'est requis au lancement : le modèle et le panel sont "
                "fournis localement.",
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Commandes", br.STYLES["h2"]))
    story.extend(
        bullets(
            [
                "Lancer l'application : python -m streamlit run app.py",
                "Tester : python -m pytest -q",
                "Réévaluer : python report/notebook.py",
                "Construire ce rapport : python report/build_rapport_projet.py",
                "Construire les artefacts statistiques : python scripts/build_statistical_artifacts.py",
            ]
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Fichiers requis", br.STYLES["h2"]))
    story.append(
        wide_table(
            ["Chemin", "Rôle"],
            [
                ["models/fire_model.joblib", "Modèle calibré déployé"],
                ["models/model_card.json", "Contrat 20 features et bandes de risque"],
                ["data/processed/fire_panel.parquet", "Panel cellule-jour"],
                ["report/evaluation_summary.json", "Métriques et figures d'évaluation"],
                ["results/statistical_validation.json", "Quatre tests statistiques"],
                ["results/model_benchmark.csv", "Benchmark à découpage commun"],
                ["results/benchmark_protocol.json", "Protocole du benchmark"],
                ["docs/Rapport_Projet.pdf", "Présent rapport"],
            ],
            [78 * mm, CONTENT_W - 78 * mm],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        br.callout(
            "Docker",
            "Un Dockerfile et un .dockerignore accompagnent le dépôt. La construction de "
            "l'image n'a pas pu être vérifiée dans l'environnement de développement parce que "
            "la commande docker n'y est pas installée.",
            br.WARN,
        )
    )
    return story


def main() -> int:
    required = [
        REPORT_DIR / "evaluation_summary.json",
        ROOT / "results" / "statistical_validation.json",
        ROOT / "results" / "benchmark_protocol.json",
        ROOT / "results" / "model_benchmark.csv",
        ROOT / "models" / "model_card.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        print(f"Missing artifacts: {', '.join(missing)}")
        return 1

    summary = load_json(required[0])
    stats = load_json(required[1])
    protocol = load_json(required[2])
    card = load_json(required[4])
    benchmark = load_benchmark()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        title="FIREWATCH AI — Rapport de projet scientifique",
        author="FireWatch AI",
        subject="Prédiction du risque d'incendie en Algérie du Nord",
        creator="report/build_rapport_projet.py",
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="cover",
                frames=[
                    Frame(
                        br.MARGIN,
                        br.MARGIN,
                        CONTENT_W,
                        PAGE_H - 2 * br.MARGIN,
                        id="cover",
                    )
                ],
                onPage=_cover,
            ),
            PageTemplate(id="body", frames=[br._frame()], onPage=_body_page),
        ]
    )
    story = build_story(summary, stats, protocol, card, benchmark)
    doc.build(story)

    pages = int(doc.page)
    raw_pages = len(re.findall(rb"/Type\s*/Page\b", OUTPUT.read_bytes()))
    print(
        f"Wrote {OUTPUT.relative_to(ROOT)} — {pages} pages "
        f"({OUTPUT.stat().st_size / 1024:.0f} KB, PDF objects: {raw_pages})"
    )
    if not 15 <= pages <= 30:
        print("ERROR: the generated report must contain between 15 and 30 pages.")
        return 1
    if raw_pages != pages:
        print("ERROR: PDF page-object count disagrees with the document page counter.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
