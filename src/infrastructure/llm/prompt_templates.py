from __future__ import annotations

from src.domain.entities import Chunk

# LLM должна вернуть ровно эту фразу, если в источниках нет ответа —
# use case проверяет её и переводит на человека вместо того, чтобы
# показать студенту неуверенный/выдуманный ответ (требование куратора:
# снижение галлюцинаций должно быть реализовано, а не заявлено).
INSUFFICIENT_DATA_MARKER = "Недостаточно данных для ответа."

_SYSTEM_INSTRUCTIONS = f"""Ты — помощник по вопросам РТУ МИРЭА для студентов первого курса.
Отвечай ТОЛЬКО на основе текста источников ниже. Не придумывай факты,
которых там нет, и не подставляй общие знания о вузах вместо источников.
Если источники не содержат ответа на вопрос — ответь ровно фразой:
"{INSUFFICIENT_DATA_MARKER}" и больше ничего не пиши.
Если ответ есть — отвечай кратко, по-русски, как будто объясняешь
студенту, без ссылок вида "источник №N"."""

# Точный расчёт токенов появится вместе с выбором конкретной LLM (открытый
# вопрос, см. CLAUDE.md) — до тех пор это единственная разумная оценка без
# привязки к токенайзеру. Работа с контекстным окном обрезкой по
# токен-бюджету (а не по числу чанков) — явное требование куратора.
_CHARS_PER_TOKEN_ESTIMATE = 4
_DEFAULT_MAX_CONTEXT_TOKENS = 2000


def build_prompt(
    question: str, context_chunks: list[Chunk], max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS
) -> str:
    """Собирает промпт: системная инструкция + источники (обрезанные по
    токен-бюджету, приоритет — по порядку в context_chunks, т.е. по
    RRF-рангу) + вопрос студента."""
    context_text = _fit_context_to_budget(context_chunks, max_context_tokens)

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"Источники:\n{context_text}\n\n"
        f"Вопрос студента: {question}\n"
        f"Ответ:"
    )


def _fit_context_to_budget(chunks: list[Chunk], max_tokens: int) -> str:
    budget_chars = max_tokens * _CHARS_PER_TOKEN_ESTIMATE
    pieces: list[str] = []
    used_chars = 0

    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        addition = len(text) + 2  # +разделитель между чанками
        if pieces and used_chars + addition > budget_chars:
            break  # хотя бы один чанк уже влез — дальше режем по бюджету
        pieces.append(text)
        used_chars += addition

    return "\n\n".join(pieces)
