"""REST routes for incident analysis and query history."""

from fastapi import APIRouter, HTTPException, Query

from app.ai_engines.pipeline import analyze_incident
from app.api.schemas import HistoryItem, HistoryResponse, SummarizeRequest, SummarizeResponse
from app.core.config import settings
from app.repository.database import fetch_query_history, insert_query_record

router = APIRouter(prefix=settings.API_PREFIX, tags=["incidents"])


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(body: SummarizeRequest) -> SummarizeResponse:
    try:
        result = analyze_incident(body.district, body.raw_report)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    row_id = insert_query_record(
        district=body.district,
        hazard_type=body.hazard_type,
        raw_report=body.raw_report,
        summary=result.summary,
        predicted_risk=result.predicted_risk,
        confidence_score=result.confidence_score,
        map_path=result.map_path,
    )
    return SummarizeResponse(
        id=row_id,
        district=body.district,
        hazard_type=body.hazard_type,
        summary=result.summary,
        predicted_risk=result.predicted_risk,
        confidence_score=result.confidence_score,
        map_path=result.map_path,
    )


@router.get("/history", response_model=HistoryResponse)
def history(
    limit: int | None = Query(default=None, ge=1, le=settings.HISTORY_MAX_LIMIT),
) -> HistoryResponse:
    rows = fetch_query_history(limit=limit)
    return HistoryResponse(items=[HistoryItem(**row) for row in rows])
