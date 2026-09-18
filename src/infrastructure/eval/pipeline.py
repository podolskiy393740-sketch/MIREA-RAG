from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from src.domain.entities import Answer
from src.infrastructure.eval.llm_judge import LLMJudge
from src.infrastructure.eval.metrics import rouge_1_f1, rouge_l_f1

AnswerFn = Callable[[str], Awaitable[Answer]]


@dataclass(frozen=True)
class EvalCase:
    question: str
    ideal_answer: str


@dataclass(frozen=True)
class EvalResult:
    question: str
    ideal_answer: str
    model_answer: str
    rouge_1: float | None
    rouge_l: float | None
    judge_score: int | None
    judge_reason: str | None
    latency_ms: float | None
    needs_human_fallback: bool
    error: str | None = None


async def evaluate_dataset(
    cases: list[EvalCase], answer_fn: AnswerFn, judge: LLMJudge, concurrency: int = 1
) -> list[EvalResult]:
    """Прогоняет каждый кейс через реальный AnswerQuestionUseCase (answer_fn
    — замыкание из composition root, скрывающее управление БД-сессиями от
    eval-пайплайна) и оценивает ROUGE + LLM-judge.

    concurrency по умолчанию 1: бесплатный тариф OpenRouter — общий
    перегруженный пул с лимитом запросов в минуту (см. project memory),
    параллельные eval-запросы быстро упираются в 429.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _run(case: EvalCase) -> EvalResult:
        async with semaphore:
            return await _evaluate_case(case, answer_fn, judge)

    return await asyncio.gather(*(_run(case) for case in cases))


async def _evaluate_case(case: EvalCase, answer_fn: AnswerFn, judge: LLMJudge) -> EvalResult:
    """Ошибка на одном кейсе (сеть, исчерпанные ретраи LLM и т.п.) не
    должна ронять весь прогон и терять уже посчитанные результаты
    остальных кейсов — падение "проваливается" в EvalResult.error, а не
    наружу через asyncio.gather (см. evaluate_dataset)."""
    started_at = time.monotonic()
    try:
        answer = await answer_fn(case.question)
    except Exception as exc:
        return EvalResult(
            question=case.question,
            ideal_answer=case.ideal_answer,
            model_answer="",
            rouge_1=None,
            rouge_l=None,
            judge_score=None,
            judge_reason=None,
            latency_ms=None,
            needs_human_fallback=True,
            error=str(exc),
        )
    latency_ms = (time.monotonic() - started_at) * 1000

    judge_score: int | None = None
    judge_reason: str | None = None
    try:
        judged = await judge.judge(
            question=case.question, ideal_answer=case.ideal_answer, model_answer=answer.text
        )
        judge_score = judged.score
        judge_reason = judged.reason
    except Exception:
        pass  # judge недоступен - ROUGE и сам ответ всё равно посчитаны

    return EvalResult(
        question=case.question,
        ideal_answer=case.ideal_answer,
        model_answer=answer.text,
        rouge_1=rouge_1_f1(case.ideal_answer, answer.text),
        rouge_l=rouge_l_f1(case.ideal_answer, answer.text),
        judge_score=judge_score,
        judge_reason=judge_reason,
        latency_ms=latency_ms,
        needs_human_fallback=answer.needs_human_fallback,
    )
