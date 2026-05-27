"""Orchestrates NLP → ML → GIS for a single incident analysis."""

from __future__ import annotations

from dataclasses import dataclass

from app.ai_engines.generative_nlp import generate_report_summary
from app.ai_engines.geospatial_gis import render_risk_map
from app.ai_engines.predictive_ml import predict_risk


@dataclass(frozen=True, slots=True)
class IncidentAnalysis:
    summary: str
    predicted_risk: str
    confidence_score: float
    map_path: str


def analyze_incident(district: str, raw_report: str) -> IncidentAnalysis:
    """Run the full AI pipeline for one district and field report."""
    summary = generate_report_summary(raw_report)
    risk, confidence = predict_risk(district, summary)
    map_path = render_risk_map(district, risk, confidence)
    return IncidentAnalysis(
        summary=summary,
        predicted_risk=risk,
        confidence_score=confidence,
        map_path=map_path,
    )
