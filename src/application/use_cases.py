from __future__ import annotations

import asyncio

from src.domain.entities import Answer, Chunk, Document, Query
from src.domain.ports import ChunkerPort, EmbedderPort, FullTextStorePort, LLMPort, VectorStorePort
from src.infrastructure.retrieval.rrf import rrf_fuse

_SEARCH_TOP_K = 10
# Сколько чанков после RRF-слияния отдавать в промпт LLM. Обрезка по
# токен-бюджету (см. ARCHITECTURE.md) отложена до выбора конкретной LLM
# (открытый вопрос, см. CLAUDE.md) — там же появится настоящий бюджет.
_MAX_CONTEXT_CHUNKS = 5


class AnswerQuestionUseCase:
    """Основной сценарий Q&A (см. ARCHITECTURE.md).

    Порты опциональны и деградируют по отдельности:
    - без fulltext_store вообще нет поиска -> человеческий фолбек;
    - без vector_store/embedder работает только FTS-ветка гибридного
      поиска (модель эмбеддингов уже выбрана — Qwen3-Embedding, но это
      не значит, что векторное хранилище обязано быть поднято везде);
    - без llm поиск отрабатывает полностью (можно увидеть, что нашёл
      RRF), но финальный ответ не генерируется — LLM ещё не выбрана
      (открытый вопрос, см. CLAUDE.md), нельзя реализовывать как решение
      по умолчанию.
    """

    def __init__(
        self,
        vector_store: VectorStorePort | None = None,
        fulltext_store: FullTextStorePort | None = None,
        embedder: EmbedderPort | None = None,
        llm: LLMPort | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._fulltext_store = fulltext_store
        self._embedder = embedder
        self._llm = llm

    async def execute(self, query: Query) -> Answer:
        if self._fulltext_store is None:
            return Answer(
                text="Пока не подключена база знаний — этот вопрос передан техподдержке/куратору.",
                needs_human_fallback=True,
            )

        context_chunks = await self._retrieve(query)

        if self._llm is None:
            return Answer(
                text="Пока не подключена генерация ответа — этот вопрос передан техподдержке/куратору.",
                sources=context_chunks,
                needs_human_fallback=True,
            )

        answer_text = await self._llm.generate(query.text, context_chunks)
        return Answer(text=answer_text, sources=context_chunks)

    async def _retrieve(self, query: Query) -> list[Chunk]:
        if self._vector_store is not None and self._embedder is not None:
            query_embedding = (await self._embedder.embed([query.text]))[0]
            fts_results, vector_results = await asyncio.gather(
                self._fulltext_store.search(query.text, top_k=_SEARCH_TOP_K),
                self._vector_store.search(query_embedding, top_k=_SEARCH_TOP_K),
            )
        else:
            fts_results = await self._fulltext_store.search(query.text, top_k=_SEARCH_TOP_K)
            vector_results = []

        fused = rrf_fuse(fts_results, vector_results)
        return [retrieved.chunk for retrieved in fused[:_MAX_CONTEXT_CHUNKS]]


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
