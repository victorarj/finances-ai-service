from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import document_chunks_table, documents_table


async def retrieve_similar_chunks(
    session: AsyncSession,
    *,
    user_id: int,
    question_embedding: list[float],
    document_ids: list[int] | None = None,
    limit: int = 5,
) -> list[dict[str, int | str | float]]:
    distance = document_chunks_table.c.embedding.cosine_distance(question_embedding).label("distance")

    query = (
        select(
            document_chunks_table.c.document_id,
            document_chunks_table.c.chunk_index,
            document_chunks_table.c.content,
            documents_table.c.source_type,
            distance,
        )
        .select_from(
            document_chunks_table.join(
                documents_table,
                document_chunks_table.c.document_id == documents_table.c.id,
            )
        )
        .where(
            document_chunks_table.c.user_id == user_id,
            documents_table.c.user_id == user_id,
            documents_table.c.deleted_at.is_(None),
        )
        .order_by(distance.asc())
        .limit(limit)
    )

    if document_ids:
        query = query.where(document_chunks_table.c.document_id.in_(document_ids))

    result = await session.execute(query)
    rows = [dict(row._mapping) for row in result]

    # TODO: reranker
    return rows
