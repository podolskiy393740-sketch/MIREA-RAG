from src.domain.entities import Chunk
from src.infrastructure.retrieval.rrf import rrf_fuse


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="doc", text=f"текст {chunk_id}", position=0, chunking_strategy="recursive")


def test_document_in_both_lists_outranks_document_in_one():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    # "a" встречается в обоих списках, "b" и "c" — только в одном каждый.
    fused = rrf_fuse(fts_results=[a, b], vector_results=[a, c])

    assert fused[0].chunk.id == "a"
    assert fused[0].fts_rank == 1
    assert fused[0].vector_rank == 1


def test_chunk_present_only_in_fts_is_kept_with_null_vector_rank():
    a = _chunk("a")
    fused = rrf_fuse(fts_results=[a], vector_results=[])

    assert len(fused) == 1
    assert fused[0].fts_rank == 1
    assert fused[0].vector_rank is None


def test_chunk_present_only_in_vector_is_kept_with_null_fts_rank():
    a = _chunk("a")
    fused = rrf_fuse(fts_results=[], vector_results=[a])

    assert len(fused) == 1
    assert fused[0].vector_rank == 1
    assert fused[0].fts_rank is None


def test_empty_inputs_return_empty_list():
    assert rrf_fuse(fts_results=[], vector_results=[]) == []


def test_score_matches_rrf_formula():
    a = _chunk("a")
    fused = rrf_fuse(fts_results=[a], vector_results=[a], k=60)

    assert fused[0].rrf_score == 1 / 61 + 1 / 61


def test_typo_resilience_example():
    """Демонстрация из CLAUDE.md для презентации куратору: опечатка в
    запросе ломает точное совпадение лексем в FTS (нужный документ
    проваливается вниз tsquery-списка), но векторный эмбеддинг устойчив к
    опечатке и всё равно находит документ по смыслу. Чистый FTS-поиск
    показал бы студенту нерелевантный документ первым; RRF за счёт
    векторной ветки вытаскивает релевантный документ наверх."""
    relevant_despite_typo = _chunk("regulation")
    keyword_coincidence = _chunk("distractor")

    # FTS: опечатка сломала точное совпадение лексемы, релевантный
    # документ найден только по частичному стеммингу и стоит на 5-м месте.
    fts_results = [
        keyword_coincidence,
        _chunk("fts-noise-1"),
        _chunk("fts-noise-2"),
        relevant_despite_typo,
    ]
    # Вектор: эмбеддинг запроса устойчив к опечатке, документ — на первом месте.
    vector_results = [relevant_despite_typo, _chunk("vector-noise-1"), _chunk("vector-noise-2")]

    fused = rrf_fuse(fts_results, vector_results)
    pure_fts_top = fts_results[0]

    assert fused[0].chunk.id == "regulation"
    assert pure_fts_top.id != "regulation"
