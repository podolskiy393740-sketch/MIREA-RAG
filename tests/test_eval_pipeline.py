import pytest

from src.domain.entities import Answer
from src.infrastructure.eval.llm_judge import LLMJudge
from src.infrastructure.eval.pipeline import EvalCase, evaluate_dataset


class FakeJudgeLLM:
    async def generate(self, prompt: str) -> str:
        return '{"score": 5, "reason": "совпадает"}'


async def _answer_fn(question: str) -> Answer:
    return Answer(text=f"ответ на: {question}", needs_human_fallback=False)


async def _fallback_answer_fn(question: str) -> Answer:
    return Answer(text="передано техподдержке", needs_human_fallback=True)


@pytest.mark.asyncio
async def test_evaluates_each_case_and_preserves_order():
    cases = [EvalCase(question=f"вопрос {i}", ideal_answer=f"ответ на: вопрос {i}") for i in range(3)]
    judge = LLMJudge(FakeJudgeLLM())

    results = await evaluate_dataset(cases, _answer_fn, judge, concurrency=2)

    assert len(results) == 3
    assert [r.question for r in results] == [c.question for c in cases]
    assert all(r.rouge_1 == 1.0 for r in results)  # answer_fn возвращает точное совпадение
    assert all(r.judge_score == 5 for r in results)


@pytest.mark.asyncio
async def test_result_carries_human_fallback_flag():
    cases = [EvalCase(question="q", ideal_answer="a")]
    judge = LLMJudge(FakeJudgeLLM())

    results = await evaluate_dataset(cases, _fallback_answer_fn, judge)

    assert results[0].needs_human_fallback is True


@pytest.mark.asyncio
async def test_latency_is_measured():
    cases = [EvalCase(question="q", ideal_answer="a")]
    judge = LLMJudge(FakeJudgeLLM())

    results = await evaluate_dataset(cases, _fallback_answer_fn, judge)

    assert results[0].latency_ms >= 0
