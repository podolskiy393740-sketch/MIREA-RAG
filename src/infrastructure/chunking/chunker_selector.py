from __future__ import annotations

from src.domain.entities import Chunk, Document, DocumentType
from src.domain.ports import ChunkerPort
from src.infrastructure.chunking.recursive_chunker import RecursiveCharacterChunker
from src.infrastructure.chunking.structure_aware_chunker import StructureAwareChunker


class ChunkerSelector:
    """Выбор стратегии чанкинга по типу документа — гибрид, а не или/или
    (см. CLAUDE.md): structure-aware для размеченных HTML-страниц МИРЭА,
    recursive character splitting — для сплошного текста и PDF-регламентов."""

    def __init__(
        self,
        structure_aware: ChunkerPort | None = None,
        recursive: ChunkerPort | None = None,
    ) -> None:
        self._structure_aware = structure_aware or StructureAwareChunker()
        self._recursive = recursive or RecursiveCharacterChunker()

    def chunk(self, document: Document) -> list[Chunk]:
        if document.doc_type is DocumentType.STRUCTURED_HTML:
            return self._structure_aware.chunk(document)
        return self._recursive.chunk(document)
