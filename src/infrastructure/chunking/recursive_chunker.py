from __future__ import annotations

from src.domain.entities import Chunk, Document

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveCharacterChunker:
    """Recursive character splitting — для документов без структурной
    разметки (сплошной текст, PDF-регламенты). Дополняющий, не запасной
    вариант гибридной стратегии (см. CLAUDE.md/ARCHITECTURE.md)."""

    def __init__(
        self,
        max_chars: int = 1000,
        overlap: int = 150,
        separators: list[str] | None = None,
    ) -> None:
        if overlap >= max_chars:
            raise ValueError("overlap must be smaller than max_chars")
        self._max_chars = max_chars
        self._overlap = overlap
        self._separators = separators or _DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._split(document.raw_text, self._separators)
        merged = self._merge_with_overlap(pieces)
        return [
            Chunk(
                id=f"{document.id}:{position}",
                document_id=document.id,
                text=text,
                position=position,
                chunking_strategy="recursive",
            )
            for position, text in enumerate(merged)
            if text.strip()
        ]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        text = text.strip()
        if len(text) <= self._max_chars:
            return [text] if text else []

        if not separators:
            return [text[i : i + self._max_chars] for i in range(0, len(text), self._max_chars)]

        separator, *rest = separators
        parts = text.split(separator) if separator else list(text)

        pieces: list[str] = []
        for part in parts:
            if len(part) <= self._max_chars:
                if part:
                    pieces.append(part)
            else:
                pieces.extend(self._split(part, rest))
        return pieces

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Склеивает мелкие куски в чанки размером до max_chars, добавляя
        overlap символов хвоста предыдущего чанка в начало следующего."""
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= self._max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
                tail = current[-self._overlap :]
                current = f"{tail} {piece}".strip()
            else:
                current = piece
        if current:
            chunks.append(current)
        return chunks
