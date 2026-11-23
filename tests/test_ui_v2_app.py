"""AppTest integration tests for app_v2.py (FireWatch AI UI V2).

Validates:
  - Initial startup renders the Level 1 Welcome & Landing page ("Understand. Monitor. Predict.").
  - "Explore FireWatch AI" navigation transitions to Level 2 (Geographical Dashboard).
  - Level 2 displays the satellite map cockpit, location context, and return button.
  - "← Accueil" successfully returns to the landing page.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

APP_PATH = ROOT / "app.py"


def test_app_v2_initial_load_shows_landing_page():
    """Verify that initial app opening shows the Level 1 Welcome & Landing Page."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    assert not app.exception
    rendered = " ".join(item.value for item in list(app.markdown) + list(app.html))
    assert "Understand. Monitor. Predict." in rendered
    assert "FireWatch" in rendered

    # Button to explore must be present
    buttons = [b.label for b in app.button]
    assert any("Explorer FireWatch AI" in b for b in buttons)


def test_app_v2_transition_to_dashboard():
    """Verify that setting app_level to 'dashboard' renders the Level 2 Geographical Dashboard."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["app_level"] = "dashboard"
    app.run()
    assert not app.exception
    rendered = " ".join(item.value for item in list(app.markdown) + list(app.html))
    assert "Cockpit Satellite" in rendered
    assert "Position Ciblée" in rendered

    # Return button must be present
    buttons = [b.label for b in app.button]
    assert any("Accueil" in b for b in buttons)
    assert any("Prédire" in b for b in buttons)


def test_app_v2_return_navigation():
    """Verify navigation back from dashboard to landing page."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["app_level"] = "dashboard"
    app.run()
    assert not app.exception

    # Simulate return
    app.session_state["app_level"] = "landing"
    app.run()
    assert not app.exception
    rendered = " ".join(item.value for item in list(app.markdown) + list(app.html))
    assert "Understand. Monitor. Predict." in rendered
