"""Индексация документов из CSV вида question,answer,source_url,topic.

НЕ запускать на data/qa_pairs.csv из chernenko-s/mirea-rag без явного
разрешения автора/куратора — репозиторий без лицензии, датасет похоже
подготовлен вручную (is_generated=TRUE), это не наш сбор данных (см.
обсуждение в docs/embeddings-comparison.md и историю решений проекта).

Использование (после разрешения и при поднятом Postgres):
    python -m scripts.seed_from_qa_csv --csv path/to/qa_pairs.csv
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
