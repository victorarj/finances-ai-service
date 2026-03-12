import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.database import get_db_session
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_and_store_chunks
from app.ingestion.extractor import extract_text
from app.integrations.node_backend import update_document_status
from app.storage.object_store import ObjectStore


logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    document_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    storage_key: str = Field(min_length=1)
    source_type: str = Field(min_length=1)


async def _safe_update_document_status(
    document_id: int,
    status: str,
    *,
    processed_at: datetime | None = None,
) -> None:
    try:
        await update_document_status(document_id, status, processed_at=processed_at)
    except Exception as exc:  # pragma: no cover - helper already catches real callback failures
        logger.warning(
            "Status callback wrapper failed for document %s status %s: %s",
            document_id,
            status,
            exc,
        )


@router.post("/ingest")
async def ingest_document(
    payload: IngestRequest,
    session: AsyncSession = Depends(get_db_session),
):
    settings = get_settings()
    document_id = int(payload.document_id)
    user_id = int(payload.user_id)
    object_store = ObjectStore()

    await _safe_update_document_status(document_id, "processing")

    try:
        file_bytes = await object_store.download(payload.storage_key)
        text = await extract_text(file_bytes, payload.storage_key)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No chunks generated from extracted document text")

        stored_chunks = await embed_and_store_chunks(
            session,
            document_id=document_id,
            user_id=user_id,
            chunks=chunks,
            settings=settings,
        )

        await _safe_update_document_status(
            document_id,
            "ready",
            processed_at=datetime.now(timezone.utc),
        )
        return {
            "document_id": document_id,
            "user_id": user_id,
            "storage_key": payload.storage_key,
            "source_type": payload.source_type,
            "chunk_count": len(stored_chunks),
            "status": "ready",
        }
    except Exception as exc:
        logger.exception("Ingestion failed for document %s", document_id)
        await _safe_update_document_status(document_id, "failed")
        return JSONResponse(
            status_code=500,
            content={
                "error": "ingestion_failed",
                "detail": str(exc),
                "document_id": document_id,
                "status": "failed",
            },
        )
