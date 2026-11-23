"""Data-quality and exploratory-analysis page — FireWatch AI UI V2.

Thin navigation wrapper around ``dashboard.eda_view.render_eda_view``. It does
not duplicate the satellite cockpit or the legacy multipage analytics pages;
it is a third view reachable from the existing two-level V2 navigation.
"""

from __future__ import annotations

import streamlit as st

from dashboard.eda_view import render_eda_view


def render_page() -> None:
    """Render the EDA page with navigation back to the cockpit and landing."""
    col_nav, col_spacer = st.columns([1.4, 4])
    with col_nav:
        if st.button("← Retour au cockpit", key="eda_back_to_dashboard"):
            st.session_state["app_level"] = "dashboard"
            st.rerun()
        if st.button("← Accueil", key="eda_back_to_landing"):
            st.session_state["app_level"] = "landing"
            st.rerun()

    render_eda_view()
