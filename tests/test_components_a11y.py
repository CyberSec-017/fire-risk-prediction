"""Accessibility and design-system contracts for ``dashboard.components``.

These tests patch ``streamlit.markdown`` so they can inspect the generated
HTML/CSS without booting a Streamlit runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import components  # noqa: E402


def render(function) -> str:
    """Run a component and return the HTML it sends to Streamlit."""
    with patch("streamlit.markdown") as markdown:
        function()
    assert markdown.called
    return markdown.call_args.args[0]


def test_global_stylesheet_keeps_focus_visible():
    html = render(components.inject_global_css)

    assert ":focus-visible" in html
    # Keyboard users must get a ring in the dark sidebar too.
    assert "stSidebar" in html
    assert "34D399" in html


def test_global_stylesheet_respects_reduced_motion():
    html = render(components.inject_global_css)

    assert "@media (prefers-reduced-motion: reduce)" in html
    assert "transition-duration: 0.01ms" in html


def test_global_stylesheet_ships_a_screen_reader_helper():
    html = render(components.inject_global_css)

    assert ".fwx-sr-only" in html
    assert "clip: rect(0, 0, 0, 0)" in html


def test_callouts_announce_their_urgency():
    danger = render(lambda: components.render_callout("Critical", "Evacuate", "danger"))
    info = render(lambda: components.render_callout("Note", "All good", "info"))

    assert 'role="alert"' in danger
    assert 'role="status"' in info


def test_missing_data_state_is_announced():
    html = render(lambda: components.render_data_unavailable("fire_panel"))

    assert 'role="status"' in html
    assert "fire_panel" in html
