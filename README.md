# Disaster Preparedness MVP

A self-contained, multi-modal MVP Disaster Preparedness & Alert System for localized operations in Ethiopia. It integrates Generative AI (NLP report synthesis) and Predictive AI (hazard classification) with a decoupled FastAPI backend, local SQLite auditing, and interactive Folium geospatial mapping under a single unified Gradio dashboard UI.

> **Status:** Project structure scaffolded. Target implementation phase: AI Engine layer.

---

## 🛠️ Tech Stack & Features

* **API & Core Routing:** FastAPI, Pydantic, Uvicorn (`/api/summarize`, `/api/history`)
* **User Interface:** Gradio (Multi-tab layout mounted on `/ui`)
* **Generative AI:** Transformers & PyTorch CPU (`facebook/bart-base` local synthesis)
* **Predictive ML:** Scikit-Learn, Pandas, NumPy (Hazard risk classifier $\ge 70\%$ accuracy)
* **Geospatial & Storage:** Folium (Dynamic HTML map rendering) & SQLite (`query_log` table)
* **Environment Package Manager:** [uv](https://docs.astral.sh/uv/) & Docker Compose

---

## 📦 Project Directory Tree

```text
├── app/
│   ├── main.py                  # App entry point (FastAPI + Gradio mounting)
│   ├── api/                     # API layer (endpoints.py routing & schemas.py DTOs)
│   ├── ai_engines/              # Intelligence layer (generative_nlp, predictive_ml, geospatial_gis)
│   ├── repository/              # Data access layer (database.py SQLite DAO)
│   └── core/                    # System core configuration (config.py environment & constants)
└── data/                        # Runtime artifact storage (districts_data.csv seed source)