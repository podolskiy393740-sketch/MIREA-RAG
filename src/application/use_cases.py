from __future__ import annotations

from src.domain.entities import Answer, Query
from src.domain.ports import FullTextStorePort, LLMPort, VectorStorePort


class AnswerQuestionUseCase:
    """Основной сценарий Q&A (см. ARCHITECTURE.md).

    Порты опциональны: пока инфраструктура (Postgres/эмбеддинги/LLM) не
    подключена, use case возвращает human-fallback вместо ошибки — это
    позволяет presentation-слою (Telegram-бот) работать уже сейчас.
    """

    def __init__(
        self,
        vector_store: VectorStorePort | None = None,
        fulltext_store: FullTextStorePort | None = None,
        llm: LLMPort | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._fulltext_store = fulltext_store
        self._llm = llm

    async def execute(self, query: Query) -> Answer:
        if self._vector_store is None or self._fulltext_store is None or self._llm is None:
            return Answer(
                text=(
                    "Пока не подключена база знаний — этот вопрос передан "
                    "техподдержке/куратору."
                ),
                needs_human_fallback=True,
            )

        raise NotImplementedError(
            "Retrieval (RRF) и генерация ответа реализуются отдельной задачей "
            "по инфраструктуре — см. ARCHITECTURE.md"
        )
