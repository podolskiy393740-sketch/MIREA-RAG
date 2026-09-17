"""init: documents и chunks (pgvector + FTS)

Revision ID: afd21b44b9ed
Revises:
Create Date: 2026-09-17
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

from src.infrastructure.storage.postgres.settings import PostgresSettings

revision = "afd21b44b9ed"
down_revision = None
branch_labels = None
depends_on = None

_embedding_dim = PostgresSettings().embedding_dim


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("doc_type", sa.String(32), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("chunking_strategy", sa.String(32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(_embedding_dim), nullable=True),
        sa.Column(
            "text_search",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('russian', text)", persisted=True),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_chunks_text_search", "chunks", ["text_search"], postgresql_using="gin"
    )
    # Индекс приближённого поиска (ivfflat/hnsw) по embedding сознательно не
    # создаём здесь: подбор параметров зависит от объёма данных и итоговой
    # размерности эмбеддингов (открытый вопрос, см. CLAUDE.md). До тех пор
    # поиск идёт точным перебором — это ORDER BY <-> без индекса.


def downgrade() -> None:
    op.drop_index("ix_chunks_text_search", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
