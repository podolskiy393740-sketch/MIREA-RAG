from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DocumentType(str, Enum):
    STRUCTURED_HTML = "structured_html"
    UNSTRUCTURED_PDF = "unstructured_pdf"
    PLAIN_TEXT = "plain_text"


@dataclass(frozen=True)
class Document:
    id: str
    source_url: str
    doc_type: DocumentType
    raw_text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    position: int
    chunking_strategy: str


Vector = list[float]


@dataclass(frozen=True)
class UserContext:
    """Заготовка под персонализацию. Не реализована: privacy-гипотеза и
    UX для неавторизованных пользователей ещё не согласованы (см. CLAUDE.md)."""

    student_id: str | None = None
    course: int | None = None
    faculty: str | None = None


@dataclass(frozen=True)
class Query:
    text: str
    user_context: UserContext | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    fts_rank: int | None
    vector_rank: int | None
    rrf_score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Chunk] = field(default_factory=list)
    needs_human_fallback: bool = False
