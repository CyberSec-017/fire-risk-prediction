"""Statistical-validation page — FireWatch AI UI V2.

Thin navigation wrapper around ``dashboard.statistics_view.render_statistics_view``.
It is reachable from the existing landing page and cockpit and never replaces
the satellite cockpit or the EDA page.
"""

from __future__ import annotations

import streamlit as st

from dashboard.statistics_view import render_statistics_view


def render_page() -> None:
    """Render the statistical section with navigation back to existing views."""
    col_nav, col_spacer = st.columns([1.6, 4])
    with col_nav:
        if st.button("← Retour au cockpit", key="stats_back_to_dashboard"):
            st.session_state["app_level"] = "dashboard"
            st.rerun()
        if st.button("← Accueil", key="stats_back_to_landing"):
            st.session_state["app_level"] = "landing"
            st.rerun()

    render_statistics_view()
