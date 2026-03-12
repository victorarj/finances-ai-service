from collections.abc import AsyncIterator

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Integer, MetaData, String, Table, Text, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings


metadata = MetaData()

documents_table = Table(
    "documents",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("filename", String(255), nullable=False),
    Column("mime_type", String(100), nullable=False),
    Column("storage_key", String(255), nullable=False),
    Column("source_type", String(50), nullable=False),
    Column("status", String(50), nullable=False),
    Column("uploaded_at", DateTime, nullable=False),
    Column("processed_at", DateTime),
    Column("deleted_at", DateTime),
)

document_chunks_table = Table(
    "document_chunks",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("document_id", Integer, ForeignKey("documents.id"), nullable=False),
    Column("user_id", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("embedding", Vector(1536), nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
)

_settings = get_settings()
engine = create_async_engine(_settings.database_url, future=True, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session


async def check_database_health() -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT 1"))


async def dispose_engine() -> None:
    await engine.dispose()


def build_chunk_debug_query(document_id: int, user_id: int):
    return (
        select(
            document_chunks_table.c.id,
            document_chunks_table.c.document_id,
            document_chunks_table.c.user_id,
            document_chunks_table.c.content,
            document_chunks_table.c.chunk_index,
            document_chunks_table.c.created_at,
        )
        .where(
            document_chunks_table.c.document_id == document_id,
            document_chunks_table.c.user_id == user_id,
        )
        .order_by(document_chunks_table.c.chunk_index.asc(), document_chunks_table.c.id.asc())
    )
