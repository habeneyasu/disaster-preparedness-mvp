from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.core.config import settings
from app.repository.database import init_db
from app.ui import build_ui


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.APP_TITLE, lifespan=lifespan)
app.include_router(router)
settings.ensure_data_dir()
app.mount(
    "/risk-map",
    StaticFiles(directory=str(settings.DATA_DIR)),
    name="risk-map",
)
app = gr.mount_gradio_app(app, build_ui(), path=settings.GRADIO_MOUNT_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
