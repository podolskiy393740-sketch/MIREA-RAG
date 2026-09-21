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
    judge_score: int | None
    judge_reason: str | None
    latency_ms: float | None
    needs_human_fallback: bool
    # Ragas-style метрики (основные, рекомендованы куратором вместо ROUGE)
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_recall: float | None = None
    # ROUGE оставлен как дополнительный референс, не основная метрика
    rouge_1: float | None = None
    rouge_l: float | None = None
    error: str | None = None


OnResult = Callable[[int, int, EvalResult], None]


async def evaluate_dataset(
    cases: list[EvalCase],
    answer_fn: AnswerFn,
    judge: LLMJudge,
    concurrency: int = 1,
    on_result: OnResult | None = None,
) -> list[EvalResult]:
    """Прогоняет каждый кейс через реальный AnswerQuestionUseCase (answer_fn
    — замыкание из composition root, скрывающее управление БД-сессиями от
    eval-пайплайна) и оценивает ROUGE + LLM-judge.

    concurrency по умолчанию 1: бесплатный тариф OpenRouter — общий
    перегруженный пул с лимитом запросов в минуту (см. project memory),
    параллельные eval-запросы быстро упираются в 429.

    on_result(completed, total, result) вызывается по мере готовности
    каждого кейса — порядок вызовов может не совпадать с порядком cases
    при concurrency > 1, но итоговый список результатов порядок сохраняет
    (asyncio.gather). Без колбэка долгий прогон (десятки кейсов,
    последовательно) не даёт вообще никакой обратной связи до самого
    конца — неотличимо от зависшего процесса снаружи.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))
    completed = 0
    lock = asyncio.Lock()

    async def _run(case: EvalCase) -> EvalResult:
        nonlocal completed
        async with semaphore:
            result = await _evaluate_case(case, answer_fn, judge)
        if on_result is not None:
            async with lock:
                completed += 1
                on_result(completed, len(cases), result)
        return result

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
            judge_score=None,
            judge_reason=None,
            latency_ms=None,
            needs_human_fallback=True,
            error=str(exc),
        )
    latency_ms = (time.monotonic() - started_at) * 1000

    context_text = "\n\n".join(c.text for c in answer.sources)

    judge_score: int | None = None
    judge_reason: str | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_recall: float | None = None
    try:
        judged, ragas = await asyncio.gather(
            judge.judge(
                question=case.question, ideal_answer=case.ideal_answer, model_answer=answer.text
            ),
            judge.judge_ragas(
                question=case.question,
                ideal_answer=case.ideal_answer,
                model_answer=answer.text,
                context=context_text,
            ),
        )
        judge_score = judged.score
        judge_reason = judged.reason
        faithfulness = ragas.faithfulness
        answer_relevance = ragas.answer_relevance
        context_recall = ragas.context_recall
    except Exception:
        pass  # judge недоступен — latency и fallback-rate всё равно посчитаны

    return EvalResult(
        question=case.question,
        ideal_answer=case.ideal_answer,
        model_answer=answer.text,
        judge_score=judge_score,
        judge_reason=judge_reason,
        latency_ms=latency_ms,
        needs_human_fallback=answer.needs_human_fallback,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_recall=context_recall,
        rouge_1=rouge_1_f1(case.ideal_answer, answer.text),
        rouge_l=rouge_l_f1(case.ideal_answer, answer.text),
    )
