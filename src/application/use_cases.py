from __future__ import annotations

from src.domain.entities import Answer, Chunk, Document, Query
from src.domain.ports import ChunkerPort, EmbedderPort, FullTextStorePort, LLMPort, VectorStorePort


class AnswerQuestionUseCase:
    """Основной сценарий Q&A (см. ARCHITECTURE.md).

    Порты опциональны: пока инфраструктура (Postgres/эмбеддинги/LLM) не
    подключена, use case возвращает human-fallback вместо ошибки — это
    позволяет presentation-слою (Telegram-бот) работать уже сейчас.
    """

    def __init__(
        self,
        vector_store: VectorStorePort | None = None,
        fulltext_store: FullTextStorePort | None = None,
        llm: LLMPort | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._fulltext_store = fulltext_store
        self._llm = llm

    async def execute(self, query: Query) -> Answer:
        if self._vector_store is None or self._fulltext_store is None or self._llm is None:
            return Answer(
                text=(
                    "Пока не подключена база знаний — этот вопрос передан "
                    "техподдержке/куратору."
                ),
                needs_human_fallback=True,
            )

        raise NotImplementedError(
            "Retrieval (RRF) и генерация ответа реализуются отдельной задачей "
            "по инфраструктуре — см. ARCHITECTURE.md"
        )


class IngestDocumentUseCase:
    """Индексация документа: чанкинг -> FTS (всегда) -> векторное хранилище
    (только если подключены embedder и vector_store).

    Модель эмбеддингов — открытый вопрос (см. CLAUDE.md), поэтому векторная
    ветка опциональна: без неё документ всё равно проиндексируется в FTS и
    станет доступен гибридному поиску (RRF) частично уже сейчас, а
    векторная ветка подключается позже без изменений в этом use case.
    """

    def __init__(
        self,
        chunker: ChunkerPort,
        fulltext_store: FullTextStorePort,
        vector_store: VectorStorePort | None = None,
        embedder: EmbedderPort | None = None,
    ) -> None:
        self._chunker = chunker
        self._fulltext_store = fulltext_store
        self._vector_store = vector_store
        self._embedder = embedder

    async def execute(self, document: Document) -> list[Chunk]:
        chunks = self._chunker.chunk(document)

        for chunk in chunks:
            await self._fulltext_store.upsert(chunk)

        if self._vector_store is not None and self._embedder is not None:
            embeddings = await self._embedder.embed([chunk.text for chunk in chunks])
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                await self._vector_store.upsert(chunk, embedding)

        return chunks
