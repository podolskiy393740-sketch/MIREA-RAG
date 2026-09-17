from __future__ import annotations

from src.domain.entities import Chunk
from src.infrastructure.storage.postgres.models import ChunkModel


def chunk_to_domain(model: ChunkModel) -> Chunk:
    return Chunk(
        id=model.id,
        document_id=model.document_id,
        text=model.text,
        position=model.position,
        chunking_strategy=model.chunking_strategy,
    )
