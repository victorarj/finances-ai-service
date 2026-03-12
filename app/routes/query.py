from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.database import build_chunk_debug_query, get_db_session
from app.llm.chain import generate_answer, generate_embeddings
from app.retrieval.retriever import retrieve_similar_chunks


router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    document_ids: list[str] | None = None


@router.post("/query")
async def query_documents(
    payload: QueryRequest,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, object]:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    user_id = int(payload.user_id)
    document_ids = [int(document_id) for document_id in payload.document_ids] if payload.document_ids else None
    settings = get_settings()

    [question_embedding] = await generate_embeddings([question], settings=settings)
    chunks = await retrieve_similar_chunks(
        session,
        user_id=user_id,
        question_embedding=question_embedding,
        document_ids=document_ids,
        limit=5,
    )

    return await generate_answer(question=question, chunks=chunks, settings=settings)


@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    user_id: int = Query(...),
    session: AsyncSession = Depends(get_db_session),
) -> list[dict[str, object]]:
    result = await session.execute(build_chunk_debug_query(document_id, user_id))
    return [dict(row._mapping) for row in result]
