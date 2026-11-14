"""Guard against deprecated Streamlit width arguments.

Streamlit 1.67 deprecates ``use_container_width`` in favour of ``width``.
The third-party ``streamlit_folium.st_folium`` still exposes its own
``use_container_width`` parameter, so that name is explicitly allowed.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_FILES = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py"))]
ALLOWED_FUNCTIONS = {"st_folium"}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def test_streamlit_widgets_use_the_width_argument():
    offenders: list[str] = []
    for path in STREAMLIT_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not any(keyword.arg == "use_container_width" for keyword in node.keywords):
                continue
            name = _call_name(node.func)
            if name in ALLOWED_FUNCTIONS:
                continue
            offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} ({name or 'call'})")

    assert not offenders, (
        "deprecated use_container_width found; use width=\"stretch\" instead:\n"
        + "\n".join(offenders)
    )


def test_pages_still_exist_and_are_parseable():
    assert len(STREAMLIT_FILES) > 1
    for path in STREAMLIT_FILES:
        assert path.read_text(encoding="utf-8").strip()
