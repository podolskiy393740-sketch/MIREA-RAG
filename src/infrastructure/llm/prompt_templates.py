from __future__ import annotations

from src.domain.entities import Chunk, UserContext

# LLM должна вернуть ровно эту фразу, если в источниках нет ответа —
# use case проверяет её и переводит на человека вместо того, чтобы
# показать студенту неуверенный/выдуманный ответ (требование куратора:
# снижение галлюцинаций должно быть реализовано, а не заявлено).
INSUFFICIENT_DATA_MARKER = "Недостаточно данных для ответа."

_SYSTEM_INSTRUCTIONS = f"""Ты — официальный помощник РТУ МИРЭА. Отвечай вежливо, по-деловому, \
нейтральным тоном — вне зависимости от стиля вопроса. Никогда не копируй \
разговорный стиль, сленг или просторечия из вопроса.

Отвечай ТОЛЬКО на основе текста источников ниже. Не придумывай факты, \
которых там нет, и не подставляй общие знания о вузах вместо источников. \
Особо важно: не указывай конкретные числа, даты, адреса, телефоны и имена, \
если они прямо не присутствуют в тексте источников ниже.
Если источники содержат хоть какую-то релевантную информацию по теме — \
давай ответ на её основе, даже если он неполный.
Фразу "{INSUFFICIENT_DATA_MARKER}" возвращай ТОЛЬКО когда источники \
вообще не затрагивают тему вопроса.

Форматирование: жирным (**) выделяй только критически важные данные — \
конкретные сроки, суммы, адреса, телефоны. Не жирни обычные ключевые слова \
и названия разделов. Без ссылок вида "источник №N"."""

# Точный расчёт токенов появится вместе с выбором конкретной LLM (открытый
# вопрос, см. CLAUDE.md) — до тех пор это единственная разумная оценка без
# привязки к токенайзеру. Работа с контекстным окном обрезкой по
# токен-бюджету (а не по числу чанков) — явное требование куратора.
_CHARS_PER_TOKEN_ESTIMATE = 4
_DEFAULT_MAX_CONTEXT_TOKENS = 2000


def build_prompt(
    question: str,
    context_chunks: list[Chunk],
    user_context: UserContext | None = None,
    max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS,
) -> str:
    """Собирает промпт: системная инструкция + источники (обрезанные по
    токен-бюджету, приоритет — по порядку в context_chunks, т.е. по
    RRF-рангу) + вопрос студента."""
    context_text = _fit_context_to_budget(context_chunks, max_context_tokens)

    profile_hint = ""
    if user_context:
        parts = []
        if user_context.course:
            parts.append(f"{user_context.course} курс")
        if user_context.faculty:
            parts.append(f"факультет {user_context.faculty}")
        if parts:
            profile_hint = f" [Профиль студента: {', '.join(parts)}]"

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"Источники:\n{context_text}\n\n"
        f"Вопрос студента: {question}{profile_hint}\n"
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


# --- Query Expansion ---

_QUERY_EXPANSION_PROMPT = (
    "Сгенерируй два коротких поисковых запроса для базы знаний МИРЭА — "
    "перефразируй вопрос студента, используя синонимы и смежные термины. "
    "Выведи ТОЛЬКО два запроса, каждый на отдельной строке, без нумерации и пояснений.\n\n"
    "Вопрос: {question}"
)


def build_expansion_prompt(question: str) -> str:
    return _QUERY_EXPANSION_PROMPT.format(question=question)


def parse_expanded_queries(raw: str, original: str) -> list[str]:
    """Парсит две строки из ответа LLM, возвращает [original, v1, v2].
    Если LLM вернул мусор или меньше вариантов — просто меньше вариантов."""
    lines = [line.strip("•-–— 1234567890.)").strip() for line in raw.strip().splitlines()]
    variants = [l for l in lines if 4 <= len(l) <= 300][:2]
    return [original] + variants
