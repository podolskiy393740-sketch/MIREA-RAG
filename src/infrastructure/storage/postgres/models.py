from __future__ import annotations

from sqlalchemy import Computed, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from src.infrastructure.storage.postgres.settings import PostgresSettings

_settings = PostgresSettings()


class Base(DeclarativeBase):
    pass


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_url: Mapped[str] = mapped_column(Text)
    doc_type: Mapped[str] = mapped_column(String(32))
    raw_text: Mapped[str] = mapped_column(Text)

    chunks: Mapped[list["ChunkModel"]] = relationship(back_populates="document")


class ChunkModel(Base):
    """chunking_strategy хранит, какой чанкер породил фрагмент
    (structure_aware / recursive) — гибридная стратегия из CLAUDE.md
    выбирается на уровне ChunkerSelector, здесь только фиксируется факт."""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    chunking_strategy: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(_settings.embedding_dim), nullable=True
    )
    # Генерируется БД из text — не выставлять вручную в коде.
    text_search: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('russian', text)", persisted=True),
    )

    document: Mapped["DocumentModel"] = relationship(back_populates="chunks")
