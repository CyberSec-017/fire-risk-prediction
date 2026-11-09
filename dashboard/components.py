"""
FIREWATCH AI — Professional UI Components and Design System Renderer.

Provides consistent, production-grade styling, layout primitives,
headers, KPI metric cards, and feedback elements across the application.
"""

from __future__ import annotations

import streamlit as st
from dashboard.config import COLORS, RISK_LEVELS


def inject_global_css() -> None:
    """Inject the unified professional CSS stylesheet for FIREWATCH AI."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ── Base Workspace & Typography ── */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            color: {COLORS["text_primary"]};
        }}

        .stApp {{
            background-color: {COLORS["workspace_bg"]} !important;
        }}

        /* ── Modern Professional Sidebar ── */
        section[data-testid="stSidebar"] {{
            background-color: {COLORS["sidebar_bg"]} !important;
            border-right: 1px solid {COLORS["sidebar_border"]} !important;
        }}

        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown span,
        section[data-testid="stSidebar"] label {{
            color: {COLORS["sidebar_muted"]} !important;
            font-size: 0.85rem !important;
        }}

        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            color: {COLORS["sidebar_text"]} !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
        }}

        section[data-testid="stSidebar"] hr {{
            border-color: {COLORS["sidebar_border"]} !important;
            opacity: 0.6;
        }}

        /* ── Clean Headings Hierarchy ── */
        h1 {{
            color: {COLORS["text_primary"]} !important;
            font-weight: 700 !important;
            font-size: 1.65rem !important;
            letter-spacing: -0.02em !important;
            margin-bottom: 0.35rem !important;
        }}

        h2 {{
            color: {COLORS["text_primary"]} !important;
            font-weight: 600 !important;
            font-size: 1.25rem !important;
            letter-spacing: -0.01em !important;
            margin-top: 1.2rem !important;
            margin-bottom: 0.3rem !important;
        }}

        h3 {{
            color: {COLORS["text_primary"]} !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            margin-bottom: 0.25rem !important;
        }}

        p, div {{
            color: {COLORS["text_secondary"]};
        }}

        /* ── Metric Cards ── */
        div[data-testid="stMetric"] {{
            background-color: {COLORS["surface_white"]} !important;
            border: 1px solid {COLORS["border_light"]} !important;
            border-radius: 8px !important;
            padding: 16px 20px !important;
            box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04) !important;
        }}

        div[data-testid="stMetric"] label {{
            color: {COLORS["text_muted"]} !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }}

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            color: {COLORS["text_primary"]} !important;
            font-size: 1.65rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }}

        /* ── Form Inputs & Selectors ── */
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stDateInput > div > div,
        .stNumberInput > div > div input {{
            background-color: {COLORS["surface_white"]} !important;
            border: 1px solid {COLORS["border_subtle"]} !important;
            color: {COLORS["text_primary"]} !important;
            border-radius: 6px !important;
            font-size: 0.88rem !important;
        }}

        .stSelectbox > div > div:focus-within,
        .stNumberInput > div > div input:focus {{
            border-color: {COLORS["brand_primary"]} !important;
            box-shadow: 0 0 0 1px {COLORS["brand_primary"]} !important;
        }}

        /* ── Primary Action Buttons ── */
        .stButton > button {{
            background-color: {COLORS["brand_primary"]} !important;
            color: #FFFFFF !important;
            border: 1px solid {COLORS["brand_dark"]} !important;
            border-radius: 6px !important;
            padding: 8px 18px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            letter-spacing: -0.01em !important;
            transition: all 0.15s ease-in-out !important;
        }}

        .stButton > button:hover {{
            background-color: {COLORS["brand_dark"]} !important;
            border-color: {COLORS["brand_dark"]} !important;
            box-shadow: 0 2px 4px rgba(15, 81, 50, 0.18) !important;
        }}

        /* Secondary/Download Buttons */
        .stDownloadButton > button {{
            background-color: {COLORS["surface_white"]} !important;
            color: {COLORS["text_secondary"]} !important;
            border: 1px solid {COLORS["border_subtle"]} !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            font-size: 0.84rem !important;
        }}

        .stDownloadButton > button:hover {{
            color: {COLORS["brand_primary"]} !important;
            border-color: {COLORS["brand_primary"]} !important;
            background-color: {COLORS["brand_subtle"]} !important;
        }}

        /* ── Tabs Styling ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background-color: transparent;
            border-bottom: 1px solid {COLORS["border_light"]};
            margin-bottom: 16px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border: none;
            border-radius: 6px 6px 0 0;
            padding: 8px 16px;
            font-weight: 500;
            font-size: 0.88rem;
            color: {COLORS["text_muted"]} !important;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {COLORS["surface_white"]} !important;
            border-bottom: 2px solid {COLORS["brand_primary"]} !important;
            color: {COLORS["brand_primary"]} !important;
            font-weight: 600 !important;
        }}

        /* ── Tables & DataFrames ── */
        .stDataFrame {{
            border: 1px solid {COLORS["border_light"]} !important;
            border-radius: 6px !important;
            background-color: {COLORS["surface_white"]} !important;
        }}

        /* ── Expanders ── */
        .streamlit-expanderHeader {{
            background-color: {COLORS["surface_white"]} !important;
            border: 1px solid {COLORS["border_light"]} !important;
            border-radius: 6px !important;
            color: {COLORS["text_primary"]} !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
        }}

        /* ── Dividers ── */
        hr {{
            border: none !important;
            border-top: 1px solid {COLORS["border_light"]} !important;
            margin: 24px 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(
    title: str = "FIREWATCH AI",
    subtitle: str = "Intelligent Wildfire Monitoring & Risk Assessment Platform",
    badge: str = "Operational Environment",
) -> None:
    """Render a clean, professional application header."""
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 16px;
            margin-bottom: 20px;
            border-bottom: 1px solid {COLORS['border_light']};
            flex-wrap: wrap;
            gap: 12px;
        ">
            <div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="
                        width: 10px;
                        height: 10px;
                        border-radius: 50%;
                        background-color: {COLORS['brand_primary']};
                    "></div>
                    <span style="
                        font-size: 0.76rem;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 0.08em;
                        color: {COLORS['brand_primary']};
                    ">Environmental Intelligence</span>
                </div>
                <h1 style="margin: 4px 0 2px 0; font-size: 1.7rem; font-weight: 700; color: {COLORS['text_primary']};">
                    {title}
                </h1>
                <p style="margin: 0; font-size: 0.88rem; color: {COLORS['text_muted']};">
                    {subtitle}
                </p>
            </div>
            <div>
                <span style="
                    display: inline-block;
                    background-color: {COLORS['surface_white']};
                    color: {COLORS['text_secondary']};
                    border: 1px solid {COLORS['border_light']};
                    border-radius: 9999px;
                    padding: 4px 12px;
                    font-size: 0.75rem;
                    font-weight: 500;
                    letter-spacing: 0.02em;
                ">
                    ● {badge}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render the refined enterprise sidebar."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 12px 0 20px 0; border-bottom: 1px solid {COLORS['sidebar_border']}; margin-bottom: 18px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="
                        width: 24px;
                        height: 24px;
                        border-radius: 4px;
                        background-color: {COLORS['brand_accent']};
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: #FFFFFF;
                        font-weight: 700;
                        font-size: 13px;
                    ">F</div>
                    <span style="
                        font-size: 1.05rem;
                        font-weight: 700;
                        color: #FFFFFF;
                        letter-spacing: -0.02em;
                    ">FIREWATCH AI</span>
                </div>
                <div style="
                    margin-top: 6px;
                    font-size: 0.72rem;
                    color: {COLORS['sidebar_muted']};
                    line-height: 1.4;
                ">
                    Wildfire Intelligence Platform<br>
                    Study Domain: Northern Algeria
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="
                background-color: {COLORS['sidebar_surface']};
                border: 1px solid {COLORS['sidebar_border']};
                border-radius: 6px;
                padding: 10px 12px;
                margin-bottom: 18px;
            ">
                <div style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: {COLORS['sidebar_muted']};">
                    Active Study Area
                </div>
                <div style="font-size: 0.8rem; font-weight: 500; color: #F8FAFC; margin-top: 2px;">
                    Lat: 32.0° → 37.5°N
                </div>
                <div style="font-size: 0.8rem; font-weight: 500; color: #F8FAFC;">
                    Lon: -2.0° → 9.0°E
                </div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">
                    Temporal: 2024 – 2026
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="
                margin-top: 30px;
                padding-top: 14px;
                border-top: 1px solid {COLORS['sidebar_border']};
                font-size: 0.72rem;
                color: {COLORS['sidebar_muted']};
                line-height: 1.5;
            ">
                <b>Data Fusion Pipeline:</b><br>
                • NASA FIRMS (VIIRS SV-C2)<br>
                • NASA POWER Daily Weather<br>
                <div style="margin-top: 8px; color: #64748B;">
                    Academic Data Science Project
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    status: str = "neutral",
) -> None:
    """Render a clean, professional metric card."""
    status_colors = {
        "neutral": COLORS["brand_primary"],
        "low": COLORS["risk_low"],
        "moderate": COLORS["risk_moderate"],
        "high": COLORS["risk_high"],
        "danger": COLORS["risk_very_high"],
    }
    accent = status_colors.get(status, COLORS["brand_primary"])

    st.markdown(
        f"""
        <div style="
            background-color: {COLORS['surface_white']};
            border: 1px solid {COLORS['border_light']};
            border-top: 3px solid {accent};
            border-radius: 6px;
            padding: 16px 18px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            height: 100%;
        ">
            <div style="
                font-size: 0.72rem;
                font-weight: 600;
                color: {COLORS['text_muted']};
                text-transform: uppercase;
                letter-spacing: 0.04em;
            ">
                {title}
            </div>
            <div style="
                font-size: 1.6rem;
                font-weight: 700;
                color: {COLORS['text_primary']};
                letter-spacing: -0.02em;
                margin: 4px 0 2px 0;
            ">
                {value}
            </div>
            {"<div style='font-size: 0.76rem; color: " + COLORS['text_secondary'] + "; line-height: 1.3;'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(
    title: str,
    subtitle: str = "",
) -> None:
    """Render a clean editorial section header."""
    st.markdown(
        f"""
        <div style="margin: 22px 0 12px 0;">
            <div style="
                font-size: 1.05rem;
                font-weight: 600;
                color: {COLORS['text_primary']};
                letter-spacing: -0.01em;
            ">
                {title}
            </div>
            {"<div style='font-size: 0.8rem; color: " + COLORS['text_muted'] + "; margin-top: 2px;'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_callout(
    title: str,
    message: str,
    variant: str = "info",
) -> None:
    """Render a clean semantic callout banner (info, warning, danger, success)."""
    palette = {
        "info": {
            "border": "#BAE6FD",
            "bg": "#F0F9FF",
            "text": "#0369A1",
            "title": "#0C4A6E",
        },
        "warning": {
            "border": "#FDE68A",
            "bg": "#FFFBEB",
            "text": "#B45309",
            "title": "#78350F",
        },
        "danger": {
            "border": "#FECACA",
            "bg": "#FEF2F2",
            "text": "#B91C1C",
            "title": "#7F1D1D",
        },
        "success": {
            "border": "#BBF7D0",
            "bg": "#F0FDF4",
            "text": "#15803D",
            "title": "#14532D",
        },
    }
    cfg = palette.get(variant, palette["info"])

    st.markdown(
        f"""
        <div style="
            background-color: {cfg['bg']};
            border: 1px solid {cfg['border']};
            border-left: 4px solid {cfg['text']};
            border-radius: 6px;
            padding: 14px 18px;
            margin: 14px 0;
        ">
            <div style="font-size: 0.86rem; font-weight: 600; color: {cfg['title']};">
                {title}
            </div>
            <div style="font-size: 0.82rem; color: {COLORS['text_secondary']}; margin-top: 3px; line-height: 1.5;">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_unavailable(dataset_name: str = "dataset") -> None:
    """Render a clean missing data warning state."""
    st.markdown(
        f"""
        <div style="
            background-color: {COLORS['surface_white']};
            border: 1px solid {COLORS['border_light']};
            border-radius: 8px;
            padding: 36px 24px;
            text-align: center;
            margin: 24px 0;
        ">
            <div style="
                display: inline-block;
                width: 44px;
                height: 44px;
                border-radius: 50%;
                background-color: #FEF3C7;
                color: #B45309;
                font-size: 20px;
                line-height: 44px;
                margin-bottom: 12px;
            ">!</div>
            <div style="font-size: 1rem; font-weight: 600; color: {COLORS['text_primary']};">
                Dataset Not Found: {dataset_name}
            </div>
            <p style="
                max-width: 520px;
                margin: 6px auto 14px auto;
                font-size: 0.84rem;
                color: {COLORS['text_muted']};
                line-height: 1.5;
            ">
                The required processed artifact has not been detected. Please run the preprocessing pipeline to generate the necessary files.
            </p>
            <code style="
                background-color: {COLORS['surface_subtle']};
                color: {COLORS['brand_primary']};
                padding: 6px 14px;
                border-radius: 4px;
                font-size: 0.8rem;
                border: 1px solid {COLORS['border_light']};
                font-family: 'JetBrains Mono', monospace;
            ">python src/build_panel.py</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
