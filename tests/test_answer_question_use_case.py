import pytest

from src.application.use_cases import AnswerQuestionUseCase
from src.domain.entities import Chunk, Query, Vector


def _chunk(chunk_id: str, text: str = "текст") -> Chunk:
    return Chunk(id=chunk_id, document_id="doc", text=text, position=0, chunking_strategy="recursive")


class FakeFullTextStore:
    def __init__(self, results: list[Chunk]) -> None:
        self._results = results
        self.last_query: str | None = None

    async def upsert(self, chunk: Chunk) -> None:
        raise NotImplementedError

    async def search(self, query_text: str, top_k: int) -> list[Chunk]:
        self.last_query = query_text
        return self._results


class FakeVectorStore:
    def __init__(self, results: list[Chunk]) -> None:
        self._results = results
        self.last_embedding: Vector | None = None

    async def upsert(self, chunk: Chunk, embedding: Vector) -> None:
        raise NotImplementedError

    async def search(self, query_embedding: Vector, top_k: int) -> list[Chunk]:
        self.last_embedding = query_embedding
        return self._results


class FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[Vector]:
        return [[1.0, 2.0] for _ in texts]


class FakeLLM:
    def __init__(self, answer: str = "готовый ответ") -> None:
        self._answer = answer
        self.received_chunks: list[Chunk] | None = None

    async def generate(self, prompt: str, context_chunks: list[Chunk]) -> str:
        self.received_chunks = context_chunks
        return self._answer


@pytest.mark.asyncio
async def test_no_fulltext_store_returns_human_fallback_immediately():
    use_case = AnswerQuestionUseCase()

    answer = await use_case.execute(Query(text="когда стипендия?"))

    assert answer.needs_human_fallback is True
    assert answer.sources == []


@pytest.mark.asyncio
async def test_fts_only_search_when_no_embedder_or_vector_store():
    fts = FakeFullTextStore([_chunk("a"), _chunk("b")])
    use_case = AnswerQuestionUseCase(fulltext_store=fts)

    answer = await use_case.execute(Query(text="вопрос"))

    assert answer.needs_human_fallback is True  # LLM ещё не подключена
    assert {c.id for c in answer.sources} == {"a", "b"}


@pytest.mark.asyncio
async def test_hybrid_search_fuses_fts_and_vector_results():
    fts = FakeFullTextStore([_chunk("a")])
    vector = FakeVectorStore([_chunk("b")])
    use_case = AnswerQuestionUseCase(fulltext_store=fts, vector_store=vector, embedder=FakeEmbedder())

    answer = await use_case.execute(Query(text="вопрос"))

    assert {c.id for c in answer.sources} == {"a", "b"}
    assert vector.last_embedding == [1.0, 2.0]
    assert fts.last_query == "вопрос"


@pytest.mark.asyncio
async def test_llm_present_generates_final_answer_from_retrieved_chunks():
    fts = FakeFullTextStore([_chunk("a")])
    llm = FakeLLM(answer="Стипендия выплачивается 25 числа.")
    use_case = AnswerQuestionUseCase(fulltext_store=fts, llm=llm)

    answer = await use_case.execute(Query(text="когда стипендия?"))

    assert answer.needs_human_fallback is False
    assert answer.text == "Стипендия выплачивается 25 числа."
    assert llm.received_chunks == [_chunk("a")]


@pytest.mark.asyncio
async def test_context_is_capped_at_max_chunks():
    many_chunks = [_chunk(str(i)) for i in range(10)]
    fts = FakeFullTextStore(many_chunks)
    llm = FakeLLM()
    use_case = AnswerQuestionUseCase(fulltext_store=fts, llm=llm)

    await use_case.execute(Query(text="вопрос"))

    assert len(llm.received_chunks) == 5
