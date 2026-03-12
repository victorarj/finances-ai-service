from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.database import check_database_health, dispose_engine
from app.routes.ingest import router as ingest_router
from app.routes.query import router as query_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await dispose_engine()


app = FastAPI(title="finances-ai-service", lifespan=lifespan)
app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/health")
async def health() -> dict[str, str]:
    await check_database_health()
    return {"status": "ok"}
