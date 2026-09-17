from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities import Document
from src.infrastructure.storage.postgres.models import DocumentModel


class PostgresDocumentRepository:
    """Реализация DocumentStorePort. Нужна, чтобы чанки могли ссылаться
    на документ внешним ключом (chunks.document_id -> documents.id)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, document: Document) -> None:
        stmt = pg_insert(DocumentModel).values(
            id=document.id,
            source_url=document.source_url,
            doc_type=document.doc_type.value,
            raw_text=document.raw_text,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[DocumentModel.id],
            set_={"raw_text": stmt.excluded.raw_text, "source_url": stmt.excluded.source_url},
        )
        await self._session.execute(stmt)
        await self._session.commit()
