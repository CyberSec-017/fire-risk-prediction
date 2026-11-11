"""FIREWATCH AI — independent Wildfire Impact Assessment module.

This package is deliberately separate from the fire-risk model pipeline:

* it never imports ``app_core`` or ``dashboard.model_loader``;
* it never writes to ``models/``, ``src/`` or the processed panel;
* it consumes only read-only exposure artifacts plus the existing
  per-cell risk scores produced elsewhere.

Stage 1 provides the data contracts, exposure calculations, severity
index and methodology text.  The Streamlit page, maps, charts and report
builders arrive in later stages.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
