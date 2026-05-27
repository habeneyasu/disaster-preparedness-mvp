# Disaster Preparedness MVP

Multi-modal disaster triage for Ethiopia: field reports are summarized (BART), classified for risk (RandomForest), mapped (Folium), and audited in SQLite. Exposed via FastAPI and a Gradio dashboard.

## Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Pydantic, Uvicorn |
| UI | Gradio (`/ui`) |
| NLP | Transformers, PyTorch CPU (`facebook/bart-base`) |
| ML | Scikit-Learn, Pandas |
| GIS | Folium |
| Storage | SQLite (`data/query_log.db`) |
| Tooling | [uv](https://docs.astral.sh/uv/), Docker Compose |

## Quick start

```bash
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| URL | Purpose |
|-----|---------|
| http://localhost:8000/ui | Gradio dashboard |
| http://localhost:8000/docs | OpenAPI |
| http://localhost:8000/health | Health check |

## API

```bash
curl http://localhost:8000/api/districts

curl -X POST http://localhost:8000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Gambela Town",
    "hazard_type": "flood",
    "raw_report": "Heavy seasonal downpours caused river overflow and flooded residential lowlands."
  }'

curl "http://localhost:8000/api/history?limit=10"
```

## Pipeline

```text
POST /ui or POST /api/summarize
  → NLP (summary)
  → ML (risk + confidence)
  → GIS (risk_map.html)
  → SQLite (query_log)
```

## Docker

```bash
docker compose up --build
```

## Data

| File | Role |
|------|------|
| `data/districts_data.csv` | District features (source, tracked in git) |
| `data/disaster_model.pkl` | Trained classifier (generated) |
| `data/query_log.db` | Audit log (generated) |
| `data/risk_map.html` | Latest map output (generated) |
