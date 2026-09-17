from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities import Chunk
from src.infrastructure.storage.postgres.mappers import chunk_to_domain
from src.infrastructure.storage.postgres.models import ChunkModel


class PostgresFullTextRepository:
    """Реализация FullTextStorePort поверх Postgres FTS (tsvector)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, chunk: Chunk) -> None:
        stmt = pg_insert(ChunkModel).values(
            id=chunk.id,
            document_id=chunk.document_id,
            position=chunk.position,
            chunking_strategy=chunk.chunking_strategy,
            text=chunk.text,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ChunkModel.id],
            set_={"text": stmt.excluded.text},
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def search(self, query_text: str, top_k: int) -> list[Chunk]:
        query = func.plainto_tsquery("russian", query_text)
        stmt = (
            select(ChunkModel)
            .where(ChunkModel.text_search.op("@@")(query))
            .order_by(func.ts_rank(ChunkModel.text_search, query).desc())
            .limit(top_k)
        )
        result = await self._session.execute(stmt)
        return [chunk_to_domain(row) for row in result.scalars().all()]
