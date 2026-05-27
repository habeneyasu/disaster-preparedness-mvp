"""Geospatial GIS: Folium map rendering for district risk visualization."""

from __future__ import annotations

import logging

import folium

from app.ai_engines.predictive_ml import get_district_coordinates
from app.core.config import settings

logger = logging.getLogger(__name__)


def render_risk_map(
    district: str,
    predicted_risk: str,
    confidence_score: float | None = None,
) -> str:
    """Build an interactive map and return the saved HTML file path."""
    lat, lon = get_district_coordinates(district)
    risk = predicted_risk.lower()
    if risk not in settings.RISK_LABELS:
        logger.warning("Unknown risk %s; using default marker color", risk)
        color = "blue"
    else:
        color = settings.RISK_COLORS[risk]

    popup = f"<b>{district}</b><br>Risk: {risk.title()}"
    if confidence_score is not None:
        popup += f"<br>Confidence: {confidence_score:.0%}"

    risk_map = folium.Map(
        location=[lat, lon],
        zoom_start=settings.DEFAULT_MAP_ZOOM,
    )
    folium.CircleMarker(
        location=[lat, lon],
        radius=settings.RISK_MARKER_RADIUS,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        popup=folium.Popup(popup, max_width=260),
    ).add_to(risk_map)

    settings.ensure_data_dir()
    output = settings.RISK_MAP_HTML
    risk_map.save(output)
    logger.info("Saved risk map to %s", output)
    return str(output)
