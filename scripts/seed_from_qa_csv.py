"""Индексация документов из CSV вида question,answer,source_url,topic.

Датасет по умолчанию — data/external/mirea_rag_slava_qa_pairs.csv, взятый
из chernenko-s/mirea-rag без разрешения автора (репозиторий без лицензии,
включён как запасной источник "на всякий случай" — см. атрибуцию и риски
в data/external/README.md и docs/data-collection.md).

Использование (при поднятом Postgres):
    python -m scripts.seed_from_qa_csv --csv data/external/mirea_rag_slava_qa_pairs.csv
"""

from __future__ import annotations

import argparse
import asyncio

from src.application.use_cases import IngestDocumentUseCase
from src.infrastructure.chunking.chunker_selector import ChunkerSelector
from src.infrastructure.ingestion.qa_csv_source import load_documents_from_qa_csv
from src.infrastructure.storage.postgres.fulltext_repository import PostgresFullTextRepository
from src.infrastructure.storage.postgres.session import get_session


async def seed(csv_path: str) -> None:
    documents = load_documents_from_qa_csv(csv_path)
    print(f"Загружено документов из CSV: {len(documents)}")

    async with get_session() as session:
        fulltext_store = PostgresFullTextRepository(session)
        # vector_store/embedder не подключены — модель эмбеддингов ещё не
        # выбрана (см. docs/embeddings-comparison.md). Документы уйдут
        # только в FTS-индекс, векторная ветка добавится позже.
        use_case = IngestDocumentUseCase(chunker=ChunkerSelector(), fulltext_store=fulltext_store)

        total_chunks = 0
        for document in documents:
            chunks = await use_case.execute(document)
            total_chunks += len(chunks)

        print(f"Проиндексировано чанков: {total_chunks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Путь к CSV (question,answer,source_url,topic)")
    args = parser.parse_args()

    asyncio.run(seed(args.csv))
