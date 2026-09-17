from __future__ import annotations

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# ВАЖНО: бесплатные модели OpenRouter меняются и пропадают без
# предупреждения. meta-llama/llama-3.3-70b-instruct:free (изначальный
# выбор) исчезла из бесплатного тира между сессиями работы над проектом —
# API в ответ прямо предложил платный слаг. При выборе новой модели
# проверять живьём через GET /api/v1/models (?id.endswith(":free")), а не
# по статьям/документации — та тоже быстро устаревает.
_DEFAULT_MODEL = "inclusionai/ling-3.0-flash-vl:free"


class OpenRouterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str
    openrouter_model: str = _DEFAULT_MODEL


class OpenRouterLLMClient:
    """LLMPort через OpenRouter API — команда выбрала API-путь для
    генерации (в отличие от self-hosted эмбеддингов): self-hosted LLM
    приемлемого для чата качества почти всегда требует GPU, иначе
    задержка неприемлема (явное требование куратора, см. CLAUDE.md).

    Модель по умолчанию — бесплатная (см. _DEFAULT_MODEL), чтобы не
    тратить бюджет на внутреннем тестировании (курсовая — "старт:
    Telegram-бот для внутренних тестов"). Переключить на платную модель,
    когда понадобится больше, можно через OPENROUTER_MODEL без изменений
    в коде.

    Бесплатный тир на практике нестабилен: провайдеры (Google AI Studio,
    Nvidia, сторонние роутеры) регулярно отдают 429/503 "temporarily
    overloaded" на общем пуле бесплатных запросов — это не баг клиента.
    AnswerQuestionUseCase уже перехватывает такие сбои и уходит в
    human-фолбек (см. use_cases.py), так что бот не падает, но стоит
    закладывать, что на бесплатном тире ответы иногда будут "передано
    техподдержке" не из-за отсутствия данных, а из-за перегрузки апстрима.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else OpenRouterSettings().openrouter_api_key
        self._model = model if model is not None else OpenRouterSettings().openrouter_model
        self._client = client or httpx.AsyncClient(timeout=30)

    async def generate(self, prompt: str) -> str:
        response = await self._client.post(
            _API_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "messages": [{"role": "user", "content": prompt}]},
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
