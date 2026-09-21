from __future__ import annotations

import json
from dataclasses import dataclass

from src.domain.ports import LLMPort

_JUDGE_SYSTEM_PROMPT = """Ты — строгий и беспристрастный оценщик качества ответа RAG-помощника МИРЭА.

Тебе даны вопрос студента, эталонный ответ и ответ модели. Оцени ответ
модели по шкале 1-5, ориентируясь на фактическую точность, полноту и
отсутствие выдуманных деталей, которых нет в эталоне:

5 — полностью соответствует эталону, без ошибок и существенных пропусков
4 — в целом верно, но есть небольшие неточности или мелкие пропуски
3 — частично верно, но есть заметные пропуски или неточности
2 — много ошибок или существенная неполнота, ответ вводит в заблуждение
1 — ответ в основном неверный или не отвечает на вопрос

Верни ТОЛЬКО JSON без дополнительного текста: {"score": 1-5, "reason": "короткое объяснение"}"""

_FAITHFULNESS_PROMPT = """Оцени, насколько каждый факт в ответе модели подтверждён предоставленным контекстом.
Факт в ответе, которого нет в контексте — нарушение достоверности.

Контекст:
{context}

Ответ модели:
{answer}

Верни ТОЛЬКО JSON: {{"score": 0.0-1.0, "reason": "..."}}
где 1.0 — все факты из контекста, 0.0 — ответ содержит выдуманные факты."""

_ANSWER_RELEVANCE_PROMPT = """Оцени, насколько ответ модели отвечает на заданный вопрос.
Ответ, который уходит от темы или не решает вопрос студента — менее релевантен.

Вопрос: {question}
Ответ: {answer}

Верни ТОЛЬКО JSON: {{"score": 0.0-1.0, "reason": "..."}}
где 1.0 — ответ полностью и точно отвечает на вопрос, 0.0 — не отвечает."""

_CONTEXT_RECALL_PROMPT = """Оцени, содержит ли контекст информацию, необходимую для правильного ответа.
Сравни с эталонным ответом: все ли ключевые факты из эталона есть в контексте?

Вопрос: {question}
Эталонный ответ: {ideal_answer}
Контекст:
{context}

Верни ТОЛЬКО JSON: {{"score": 0.0-1.0, "reason": "..."}}
где 1.0 — контекст содержит всё необходимое, 0.0 — нужной информации нет."""


@dataclass(frozen=True)
class JudgeResult:
    score: int | None
    reason: str | None
    raw_text: str


@dataclass(frozen=True)
class RagasScore:
    score: float | None
    reason: str | None
    raw_text: str


@dataclass(frozen=True)
class RagasMetrics:
    faithfulness: float | None
    answer_relevance: float | None
    context_recall: float | None


class LLMJudge:
    """LLM-as-a-judge для eval RAG-пайплайна.

    judge() — классическая оценка 1-5 по эталонному ответу.
    judge_ragas() — три Ragas-style метрики (Faithfulness, AnswerRelevance,
    ContextRecall), рекомендованные куратором вместо ROUGE как основные.
    """

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def judge(self, *, question: str, ideal_answer: str, model_answer: str) -> JudgeResult:
        prompt = (
            f"{_JUDGE_SYSTEM_PROMPT}\n\n"
            f"Вопрос:\n{question}\n\n"
            f"Эталонный ответ:\n{ideal_answer}\n\n"
            f"Ответ модели:\n{model_answer}"
        )
        raw_text = await self._llm.generate(prompt)
        score, reason = _parse_judge_json(raw_text)
        return JudgeResult(score=score, reason=reason, raw_text=raw_text)

    async def judge_ragas(
        self, *, question: str, ideal_answer: str, model_answer: str, context: str
    ) -> RagasMetrics:
        """Три Ragas-style метрики одним вызовом (три параллельных LLM-запроса).
        При ошибке любой метрики остальные сохраняются."""
        import asyncio

        async def _faithfulness() -> float | None:
            try:
                raw = await self._llm.generate(
                    _FAITHFULNESS_PROMPT.format(context=context, answer=model_answer)
                )
                score, _ = _parse_float_json(raw)
                return score
            except Exception:
                return None

        async def _answer_relevance() -> float | None:
            try:
                raw = await self._llm.generate(
                    _ANSWER_RELEVANCE_PROMPT.format(question=question, answer=model_answer)
                )
                score, _ = _parse_float_json(raw)
                return score
            except Exception:
                return None

        async def _context_recall() -> float | None:
            try:
                raw = await self._llm.generate(
                    _CONTEXT_RECALL_PROMPT.format(
                        question=question, ideal_answer=ideal_answer, context=context
                    )
                )
                score, _ = _parse_float_json(raw)
                return score
            except Exception:
                return None

        f, ar, cr = await asyncio.gather(_faithfulness(), _answer_relevance(), _context_recall())
        return RagasMetrics(faithfulness=f, answer_relevance=ar, context_recall=cr)


def _parse_float_json(text: str) -> tuple[float | None, str | None]:
    obj_text = _extract_first_json_object(text)
    if obj_text is None:
        return None, None
    try:
        data = json.loads(obj_text)
    except (json.JSONDecodeError, TypeError):
        return None, None
    try:
        score = float(data.get("score"))
    except (TypeError, ValueError):
        score = None
    if score is not None and not (0.0 <= score <= 1.0):
        score = max(0.0, min(1.0, score))  # clamp на случай LLM-округлений
    reason = data.get("reason")
    if reason is not None and not isinstance(reason, str):
        reason = str(reason)
    return score, reason


def _parse_judge_json(text: str) -> tuple[int | None, str | None]:
    obj_text = _extract_first_json_object(text)
    if obj_text is None:
        return None, None

    try:
        data = json.loads(obj_text)
    except (json.JSONDecodeError, TypeError):
        return None, None

    try:
        score = int(float(data.get("score")))
    except (TypeError, ValueError):
        score = None
    if score is not None and not (1 <= score <= 5):
        score = None

    reason = data.get("reason")
    if reason is not None and not isinstance(reason, str):
        reason = str(reason)

    return score, reason


def _extract_first_json_object(text: str) -> str | None:
    """LLM иногда добавляет пояснения вокруг JSON, несмотря на просьбу
    вернуть только его — вытаскиваем первый сбалансированный {...} блок,
    а не полагаемся на то, что весь ответ — чистый JSON."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
