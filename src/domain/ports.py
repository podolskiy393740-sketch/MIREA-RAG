from __future__ import annotations

from typing import Protocol

from src.domain.entities import Chunk, Document, RetrievedChunk, UserContext, Vector


class ChunkerPort(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


class EmbedderPort(Protocol):
    def embed(self, texts: list[str]) -> list[Vector]: ...


class VectorStorePort(Protocol):
    def upsert(self, chunk: Chunk, embedding: Vector) -> None: ...

    def search(self, query_embedding: Vector, top_k: int) -> list[RetrievedChunk]: ...


class FullTextStorePort(Protocol):
    def upsert(self, chunk: Chunk) -> None: ...

    def search(self, query_text: str, top_k: int) -> list[RetrievedChunk]: ...


class LLMPort(Protocol):
    def generate(self, prompt: str, context_chunks: list[Chunk]) -> str: ...


class UserContextPort(Protocol):
    """Порт под персонализацию — без реализации до решения по privacy-гипотезе."""

    def resolve(self, student_id: str) -> UserContext | None: ...
