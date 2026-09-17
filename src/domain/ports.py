from __future__ import annotations

from typing import Protocol

from src.domain.entities import Chunk, Document, UserContext, Vector


class ChunkerPort(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


class EmbedderPort(Protocol):
    async def embed(self, texts: list[str]) -> list[Vector]: ...


class VectorStorePort(Protocol):
    """Хранилище эмбеддингов (pgvector). search() возвращает чанки,
    упорядоченные по векторной близости — ранг = позиция в списке.
    Слияние с FTS-результатами (RRF) — отдельный шаг, см. ARCHITECTURE.md."""

    async def upsert(self, chunk: Chunk, embedding: Vector) -> None: ...

    async def search(self, query_embedding: Vector, top_k: int) -> list[Chunk]: ...


class FullTextStorePort(Protocol):
    """FTS-хранилище (Postgres tsvector). search() возвращает чанки,
    упорядоченные по релевантности — ранг = позиция в списке."""

    async def upsert(self, chunk: Chunk) -> None: ...

    async def search(self, query_text: str, top_k: int) -> list[Chunk]: ...


class LLMPort(Protocol):
    """Генерация ответа по уже собранному промпту (см.
    infrastructure/llm/prompt_templates.py — грaундинг в контекст,
    обрезка по токен-бюджету собираются в application-слое, LLM здесь
    не должна знать про структуру Chunk)."""

    async def generate(self, prompt: str) -> str: ...


class UserContextPort(Protocol):
    """Порт под персонализацию — без реализации до решения по privacy-гипотезе."""

    def resolve(self, student_id: str) -> UserContext | None: ...
