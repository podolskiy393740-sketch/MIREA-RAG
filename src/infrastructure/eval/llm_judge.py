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


@dataclass(frozen=True)
class JudgeResult:
    score: int | None
    reason: str | None
    raw_text: str


class LLMJudge:
    """LLM-as-a-judge — вторая часть eval-методологии из CLAUDE.md (первая
    — ROUGE, см. metrics.py). Использует тот же LLMPort, что и генерация
    ответов, но может быть другой моделью/инстансом клиента."""

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
