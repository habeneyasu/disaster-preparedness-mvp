"""Orchestrates NLP → ML → GIS and persists incident records."""

from __future__ import annotations

from dataclasses import dataclass

from app.ai_engines.generative_nlp import generate_report_summary
from app.ai_engines.geospatial_gis import render_risk_map
from app.ai_engines.predictive_ml import predict_risk
from app.repository.database import insert_query_record


@dataclass(frozen=True, slots=True)
class IncidentAnalysis:
    summary: str
    predicted_risk: str
    confidence_score: float
    map_path: str


@dataclass(frozen=True, slots=True)
class IncidentRecord:
    id: int
    district: str
    hazard_type: str | None
    raw_report: str
    analysis: IncidentAnalysis


def analyze_incident(district: str, raw_report: str) -> IncidentAnalysis:
    summary = generate_report_summary(raw_report)
    risk, confidence = predict_risk(district, summary)
    map_path = render_risk_map(district, risk, confidence)
    return IncidentAnalysis(summary, risk, confidence, map_path)


def process_incident(
    district: str,
    raw_report: str,
    hazard_type: str | None = None,
) -> IncidentRecord:
    """Run the AI pipeline and persist one query_log row."""
    analysis = analyze_incident(district, raw_report)
    row_id = insert_query_record(
        district=district,
        hazard_type=hazard_type,
        raw_report=raw_report,
        summary=analysis.summary,
        predicted_risk=analysis.predicted_risk,
        confidence_score=analysis.confidence_score,
        map_path=analysis.map_path,
    )
    return IncidentRecord(row_id, district, hazard_type, raw_report, analysis)
