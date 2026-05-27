from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.core.config import settings
from app.repository.database import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.APP_TITLE, lifespan=lifespan)
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
