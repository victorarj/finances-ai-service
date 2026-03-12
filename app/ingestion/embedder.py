from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.database import document_chunks_table
from app.llm.chain import generate_embeddings


async def embed_and_store_chunks(
    session: AsyncSession,
    *,
    document_id: int,
    user_id: int,
    chunks: list[dict[str, int | str]],
    settings: Settings,
) -> list[dict[str, int | str]]:
    if not chunks:
        return []

    contents = [str(chunk["content"]) for chunk in chunks]
    embeddings = await generate_embeddings(contents, settings=settings)

    if len(embeddings) != len(chunks):
        raise ValueError("Embedding count does not match chunk count")

    rows = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        rows.append(
            {
                "document_id": document_id,
                "user_id": user_id,
                "content": str(chunk["content"]),
                "embedding": embedding,
                "chunk_index": int(chunk["chunk_index"]),
            }
        )

    await session.execute(insert(document_chunks_table), rows)
    await session.commit()
    return rows
