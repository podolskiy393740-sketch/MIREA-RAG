from __future__ import annotations

import asyncio

from src.domain.entities import Answer, Chunk, Document, Query
from src.domain.ports import (
    ChunkerPort,
    DocumentStorePort,
    EmbedderPort,
    FullTextStorePort,
    LLMPort,
    VectorStorePort,
)
from src.infrastructure.llm.prompt_templates import INSUFFICIENT_DATA_MARKER, build_prompt
from src.infrastructure.retrieval.rrf import rrf_fuse

_SEARCH_TOP_K = 10
# Грубый предфильтр числом чанков после RRF (до тонкой обрезки по
# токен-бюджету в build_prompt — см. prompt_templates.py).
_MAX_CONTEXT_CHUNKS = 5

_NO_ANSWER_FOUND_TEXT = (
    "К сожалению, не нашёл ответа в документах вуза — этот вопрос передан техподдержке/куратору."
)


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

    Снижение галлюцинаций (требование куратора, см. CLAUDE.md) сделано
    механически, а не только в тексте промпта: если retrieval не нашёл
    ни одного чанка, LLM вообще не вызывается — ей физически не на чём
    галлюцинировать. Если LLM всё равно возвращает маркер "источники не
    содержат ответа" (см. prompt_templates.py), это тоже уходит в
    человеческий фолбек, а не показывается студенту как уверенный ответ.
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

        if not context_chunks:
            return Answer(text=_NO_ANSWER_FOUND_TEXT, needs_human_fallback=True)

        prompt = build_prompt(query.text, context_chunks)
        try:
            answer_text = await self._llm.generate(prompt)
        except Exception:
            # Осознанно широкий except: сеть/лимиты/невалидный ответ
            # внешнего API — любой сбой LLM должен уйти в человеческий
            # фолбек, а не уронить ответ бота студенту.
            return Answer(text=_NO_ANSWER_FOUND_TEXT, sources=context_chunks, needs_human_fallback=True)

        if INSUFFICIENT_DATA_MARKER in answer_text:
            return Answer(text=_NO_ANSWER_FOUND_TEXT, sources=context_chunks, needs_human_fallback=True)

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
    """Индексация документа: документ -> чанкинг -> FTS (всегда) ->
    векторное хранилище (только если подключены embedder и vector_store).

    document_store обязателен: chunks.document_id — внешний ключ на
    documents.id в Postgres, документ должен быть сохранён до его чанков.

    Модель эмбеддингов — открытый вопрос (см. CLAUDE.md), поэтому векторная
    ветка опциональна: без неё документ всё равно проиндексируется в FTS и
    станет доступен гибридному поиску (RRF) частично уже сейчас, а
    векторная ветка подключается позже без изменений в этом use case.
    """

    def __init__(
        self,
        chunker: ChunkerPort,
        document_store: DocumentStorePort,
        fulltext_store: FullTextStorePort,
        vector_store: VectorStorePort | None = None,
        embedder: EmbedderPort | None = None,
    ) -> None:
        self._chunker = chunker
        self._document_store = document_store
        self._fulltext_store = fulltext_store
        self._vector_store = vector_store
        self._embedder = embedder

    async def execute(self, document: Document) -> list[Chunk]:
        await self._document_store.upsert(document)
        chunks = self._chunker.chunk(document)

        for chunk in chunks:
            await self._fulltext_store.upsert(chunk)

        if self._vector_store is not None and self._embedder is not None:
            embeddings = await self._embedder.embed([chunk.text for chunk in chunks])
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                await self._vector_store.upsert(chunk, embedding)

        return chunks
