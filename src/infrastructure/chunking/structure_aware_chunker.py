from __future__ import annotations

import re

from src.domain.entities import Chunk, Document

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class StructureAwareChunker:
    """Режет по заголовкам и абзацам — основная стратегия для страниц
    МИРЭА, т.к. они уже структурно размечены (см. CLAUDE.md). Ожидает
    raw_text в markdown-подобном формате (заголовки от '#' до '######'),
    как его должен отдавать ingestion-парсер HTML-страниц.

    Каждый чанк несёт путь заголовков как контекст — иначе абзац, вырванный
    из документа, теряет смысл при retrieval.
    """

    def __init__(self, max_chars: int = 1200) -> None:
        self._max_chars = max_chars

    def chunk(self, document: Document) -> list[Chunk]:
        sections = self._split_into_sections(document.raw_text)
        chunks: list[Chunk] = []
        position = 0
        for heading_path, body in sections:
            for piece in self._split_body(body):
                text = f"{heading_path}\n\n{piece}" if heading_path else piece
                if not text.strip():
                    continue
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{position}",
                        document_id=document.id,
                        text=text,
                        position=position,
                        chunking_strategy="structure_aware",
                    )
                )
                position += 1
        return chunks

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """Возвращает список (путь заголовков, тело секции)."""
        heading_stack: list[tuple[int, str]] = []
        body_lines: list[str] = []
        sections: list[tuple[str, str]] = []

        def flush() -> None:
            body = "\n".join(body_lines).strip()
            if body:
                path = " / ".join(title for _, title in heading_stack)
                sections.append((path, body))
            body_lines.clear()

        for line in text.splitlines():
            match = _HEADER_RE.match(line)
            if match:
                flush()
                level = len(match.group(1))
                title = match.group(2).strip()
                heading_stack[:] = [h for h in heading_stack if h[0] < level]
                heading_stack.append((level, title))
            else:
                body_lines.append(line)
        flush()

        if not sections and text.strip():
            sections.append(("", text.strip()))
        return sections

    def _split_body(self, body: str) -> list[str]:
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        pieces: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= self._max_chars:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = paragraph
        if current:
            pieces.append(current)
        return pieces or [body]
