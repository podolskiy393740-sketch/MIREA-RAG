import pytest

from src.infrastructure.eval.llm_judge import LLMJudge


class FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.received_prompt: str | None = None

    async def generate(self, prompt: str) -> str:
        self.received_prompt = prompt
        return self._response


@pytest.mark.asyncio
async def test_parses_clean_json_response():
    llm = FakeLLM('{"score": 4, "reason": "в целом верно"}')
    judge = LLMJudge(llm)

    result = await judge.judge(question="в", ideal_answer="а", model_answer="б")

    assert result.score == 4
    assert result.reason == "в целом верно"


@pytest.mark.asyncio
async def test_extracts_json_surrounded_by_extra_text():
    llm = FakeLLM('Конечно! Вот моя оценка:\n{"score": 5, "reason": "точно"}\nНадеюсь, помог.')
    judge = LLMJudge(llm)

    result = await judge.judge(question="в", ideal_answer="а", model_answer="б")

    assert result.score == 5
    assert result.reason == "точно"


@pytest.mark.asyncio
async def test_unparsable_response_returns_none_score():
    llm = FakeLLM("Извините, не могу оценить этот ответ.")
    judge = LLMJudge(llm)

    result = await judge.judge(question="в", ideal_answer="а", model_answer="б")

    assert result.score is None
    assert result.reason is None
    assert result.raw_text == "Извините, не могу оценить этот ответ."


@pytest.mark.asyncio
async def test_score_out_of_range_is_rejected():
    llm = FakeLLM('{"score": 7, "reason": "вне шкалы"}')
    judge = LLMJudge(llm)

    result = await judge.judge(question="в", ideal_answer="а", model_answer="б")

    assert result.score is None


@pytest.mark.asyncio
async def test_prompt_includes_question_and_both_answers():
    llm = FakeLLM('{"score": 3, "reason": "ok"}')
    judge = LLMJudge(llm)

    await judge.judge(question="Когда стипендия?", ideal_answer="25 числа", model_answer="25-го")

    assert "Когда стипендия?" in llm.received_prompt
    assert "25 числа" in llm.received_prompt
    assert "25-го" in llm.received_prompt
