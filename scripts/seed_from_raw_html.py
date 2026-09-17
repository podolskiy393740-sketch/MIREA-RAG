"""Индексация HTML-страниц, собранных через Wayback Machine.

См. scripts/fetch_real_pages.py (сбор) и docs/data-collection.md
(методология). Манифест по умолчанию — data/raw_html/manifest.json.

Использование (при поднятом Postgres):
    python -m scripts.seed_from_raw_html --manifest data/raw_html/manifest.json
"""

from __future__ import annotations

import argparse
import asyncio

from src.application.use_cases import IngestDocumentUseCase
from src.infrastructure.chunking.chunker_selector import ChunkerSelector
from src.infrastructure.embeddings.qwen_local_embedder import QwenLocalEmbedder
from src.infrastructure.ingestion.raw_html_source import load_documents_from_raw_html_manifest
from src.infrastructure.storage.postgres.document_repository import PostgresDocumentRepository
from src.infrastructure.storage.postgres.fulltext_repository import PostgresFullTextRepository
from src.infrastructure.storage.postgres.session import get_session
from src.infrastructure.storage.postgres.vector_repository import PostgresVectorRepository


async def seed(manifest_path: str) -> None:
    documents = load_documents_from_raw_html_manifest(manifest_path)
    print(f"Загружено страниц из манифеста: {len(documents)}")

    async with get_session() as session:
        document_store = PostgresDocumentRepository(session)
        fulltext_store = PostgresFullTextRepository(session)
        vector_store = PostgresVectorRepository(session)
        embedder = QwenLocalEmbedder()
        use_case = IngestDocumentUseCase(
            chunker=ChunkerSelector(),
            document_store=document_store,
            fulltext_store=fulltext_store,
            vector_store=vector_store,
            embedder=embedder,
        )

        total_chunks = 0
        for document in documents:
            chunks = await use_case.execute(document)
            total_chunks += len(chunks)
            print(f"  {document.source_url}: {len(chunks)} чанков")

        print(f"Проиндексировано чанков: {total_chunks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/raw_html/manifest.json", help="Путь к manifest.json")
    args = parser.parse_args()

    asyncio.run(seed(args.manifest))
