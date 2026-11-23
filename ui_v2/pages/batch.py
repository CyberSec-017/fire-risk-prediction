"""Batch CSV inference page — FireWatch AI UI V2.

Thin navigation wrapper around ``dashboard.batch_view.render_batch_view``.
It keeps the satellite cockpit and the other analytical pages unchanged.
"""

from __future__ import annotations

import streamlit as st

from dashboard.batch_view import render_batch_view


def render_page() -> None:
    """Render batch inference with navigation back to existing views."""
    col_nav, col_spacer = st.columns([1.6, 4])
    with col_nav:
        if st.button("← Retour au cockpit", key="batch_back_to_dashboard"):
            st.session_state["app_level"] = "dashboard"
            st.rerun()
        if st.button("← Accueil", key="batch_back_to_landing"):
            st.session_state["app_level"] = "landing"
            st.rerun()

    render_batch_view()
