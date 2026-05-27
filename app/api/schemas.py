"""API request and response models."""

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    district: str = Field(..., min_length=1)
    raw_report: str = Field(..., min_length=1)
    hazard_type: str | None = None


class SummarizeResponse(BaseModel):
    id: int
    district: str
    summary: str
    predicted_risk: str
    confidence_score: float
    map_path: str
    hazard_type: str | None = None


class HistoryItem(BaseModel):
    id: int
    timestamp: str
    district: str
    hazard_type: str | None
    raw_report: str
    summary: str
    predicted_risk: str
    confidence_score: float
    map_path: str


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
