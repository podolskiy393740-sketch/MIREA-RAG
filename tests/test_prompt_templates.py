from src.domain.entities import Chunk
from src.infrastructure.llm.prompt_templates import INSUFFICIENT_DATA_MARKER, build_prompt


def _chunk(text: str, chunk_id: str = "c") -> Chunk:
    return Chunk(id=chunk_id, document_id="doc", text=text, position=0, chunking_strategy="recursive")


def test_prompt_contains_question_and_context():
    prompt = build_prompt("когда стипендия?", [_chunk("Стипендия выплачивается 25 числа.")])

    assert "когда стипендия?" in prompt
    assert "Стипендия выплачивается 25 числа." in prompt


def test_prompt_instructs_insufficient_data_marker_on_no_answer():
    prompt = build_prompt("вопрос", [_chunk("текст")])

    assert INSUFFICIENT_DATA_MARKER in prompt


def test_context_truncated_to_token_budget_keeps_at_least_one_chunk():
    huge_chunk = _chunk("а" * 100_000, chunk_id="huge")
    prompt = build_prompt("вопрос", [huge_chunk], max_context_tokens=10)

    # Бюджет крошечный, но хотя бы один чанк должен попасть в промпт —
    # иначе LLM вообще не на чем строить ответ.
    assert "а" in prompt


def test_context_budget_drops_chunks_beyond_budget():
    chunks = [_chunk("слово " * 200, chunk_id=str(i)) for i in range(20)]
    small_budget_prompt = build_prompt("вопрос", chunks, max_context_tokens=50)
    large_budget_prompt = build_prompt("вопрос", chunks, max_context_tokens=5000)

    assert len(small_budget_prompt) < len(large_budget_prompt)


def test_empty_context_still_produces_valid_prompt():
    prompt = build_prompt("вопрос без источников", [])

    assert "вопрос без источников" in prompt
