from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities import Chunk, Vector
from src.infrastructure.storage.postgres.mappers import chunk_to_domain
from src.infrastructure.storage.postgres.models import ChunkModel


class PostgresVectorRepository:
    """Реализация VectorStorePort поверх pgvector."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, chunk: Chunk, embedding: Vector) -> None:
        stmt = pg_insert(ChunkModel).values(
            id=chunk.id,
            document_id=chunk.document_id,
            position=chunk.position,
            chunking_strategy=chunk.chunking_strategy,
            text=chunk.text,
            embedding=embedding,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ChunkModel.id],
            set_={"text": stmt.excluded.text, "embedding": stmt.excluded.embedding},
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def search(self, query_embedding: Vector, top_k: int) -> list[Chunk]:
        stmt = (
            select(ChunkModel)
            .where(ChunkModel.embedding.is_not(None))
            .order_by(ChunkModel.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        result = await self._session.execute(stmt)
        return [chunk_to_domain(row) for row in result.scalars().all()]
