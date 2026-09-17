from __future__ import annotations

from src.domain.entities import Chunk, RetrievedChunk

DEFAULT_K = 60


def rrf_fuse(
    fts_results: list[Chunk],
    vector_results: list[Chunk],
    k: int = DEFAULT_K,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion — складывает ранги из FTS- и векторного
    списков, а не взвешенное среднее скоров (см. CLAUDE.md/ARCHITECTURE.md).

    score(d) = sum(1 / (k + rank_i(d))) по спискам, где документ встретился.
    Не требует нормализации разных шкал (ts_rank vs. косинусная близость) и
    даёт устойчивость к опечаткам в запросах: документ может быть высоко в
    FTS-списке даже при слабом векторном сходстве, и наоборот — обе ветки
    вносят вклад в итоговый ранг независимо от того, насколько "уверенно"
    выглядит их родная шкала.
    """
    fts_ranks = {chunk.id: rank for rank, chunk in enumerate(fts_results, start=1)}
    vector_ranks = {chunk.id: rank for rank, chunk in enumerate(vector_results, start=1)}

    chunks_by_id: dict[str, Chunk] = {}
    for chunk in (*fts_results, *vector_results):
        chunks_by_id.setdefault(chunk.id, chunk)

    fused = [
        RetrievedChunk(
            chunk=chunk,
            fts_rank=fts_ranks.get(chunk_id),
            vector_rank=vector_ranks.get(chunk_id),
            rrf_score=_score(fts_ranks.get(chunk_id), vector_ranks.get(chunk_id), k),
        )
        for chunk_id, chunk in chunks_by_id.items()
    ]
    fused.sort(key=lambda retrieved: retrieved.rrf_score, reverse=True)
    return fused


def _score(fts_rank: int | None, vector_rank: int | None, k: int) -> float:
    score = 0.0
    if fts_rank is not None:
        score += 1 / (k + fts_rank)
    if vector_rank is not None:
        score += 1 / (k + vector_rank)
    return score
