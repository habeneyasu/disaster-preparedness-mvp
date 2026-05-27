"""REST routes for incident analysis and query history."""

from fastapi import APIRouter, HTTPException, Query

from app.ai_engines.pipeline import IncidentRecord, process_incident
from app.ai_engines.predictive_ml import list_districts
from app.api.schemas import (
    DistrictsResponse,
    HistoryItem,
    HistoryResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.core.config import settings
from app.repository.database import fetch_query_history

router = APIRouter(prefix=settings.API_PREFIX, tags=["incidents"])


def _to_response(record: IncidentRecord) -> SummarizeResponse:
    a = record.analysis
    return SummarizeResponse(
        id=record.id,
        district=record.district,
        hazard_type=record.hazard_type,
        summary=a.summary,
        predicted_risk=a.predicted_risk,
        confidence_score=a.confidence_score,
        map_path=a.map_path,
    )


@router.get("/districts", response_model=DistrictsResponse)
def districts() -> DistrictsResponse:
    return DistrictsResponse(districts=list_districts())


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(body: SummarizeRequest) -> SummarizeResponse:
    try:
        return _to_response(
            process_incident(body.district, body.raw_report, body.hazard_type)
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/history", response_model=HistoryResponse)
def history(
    limit: int | None = Query(default=None, ge=1, le=settings.HISTORY_MAX_LIMIT),
) -> HistoryResponse:
    rows = fetch_query_history(limit=limit)
    return HistoryResponse(items=[HistoryItem(**row) for row in rows])
