"""Enterprise Sidebar component for FireWatch AI UI V2.

Provides:
  - Brand identity widget matching the Geoaether dark aesthetic.
  - Real-time operational model indicators (H=3 days, 20 features, 209 cells).
  - High-contrast visual anchoring ensuring unified dark identity.
"""

from __future__ import annotations

import textwrap
import streamlit as st


def render_v2_sidebar() -> None:
    """Render the refined dark enterprise sidebar identity for V2."""
    with st.sidebar:
        st.html(
            textwrap.dedent(
                """
                <div style="padding: 14px 16px; background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 10px; margin-top: 14px;">
                    <div style="font-size: 0.70rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">
                        FireWatch AI &bull; Cockpit V2
                    </div>
                    <div style="font-size: 0.88rem; font-weight: 800; color: #F8FAFC; margin-bottom: 4px;">
                        Système d'Aide à la Décision
                    </div>
                    <div style="font-size: 0.76rem; color: #94A3B8; line-height: 1.45;">
                        Plateforme Opérationnelle Incendie &bull; Algérie du Nord (32.0°–37.5°N)
                    </div>
                    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); display: flex; align-items: center; gap: 6px; font-size: 0.74rem; color: #10B981; font-weight: 600;">
                        <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#10B981; box-shadow: 0 0 6px #10B981;"></span>
                        Modèle Calibré &bull; H=3j (209 cellules)
                    </div>
                </div>
                """
            )
        )
