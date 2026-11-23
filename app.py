"""FireWatch AI — Wildfire Risk Intelligence Platform (UI V2).

Two-Level Experience:
  - LEVEL 1: Professional Welcome / Landing Page ("Understand. Monitor. Predict.").
  - LEVEL 2: Interactive Geographical Dashboard (Satellite Map of Algeria, 209 active cells).
"""

from __future__ import annotations

from pathlib import Path
import sys
import streamlit as st

# Configure page settings as first Streamlit call
st.set_page_config(
    page_title="FireWatch AI — Wildfire Intelligence Platform",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Inject custom theme
CSS_FILE = ROOT / "ui_v2" / "styles" / "theme.css"
if CSS_FILE.exists():
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.html(f"<style>{f.read()}</style>")

from ui_v2.dashboard_view import render_dashboard
from ui_v2.landing import render_landing_page
from ui_v2.pages.eda import render_page as render_eda_page
from ui_v2.sidebar import render_v2_sidebar


def main() -> None:
    """Master controller routing between Level 1 (Landing) and Level 2 (Dashboard)."""
    render_v2_sidebar()

    if "app_level" not in st.session_state:
        st.session_state["app_level"] = "landing"

    level = st.session_state["app_level"]

    if level == "landing":
        render_landing_page()
    elif level == "dashboard":
        render_dashboard()
    elif level == "eda":
        render_eda_page()
    else:
        render_landing_page()


if __name__ == "__main__":
    main()
