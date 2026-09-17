from __future__ import annotations

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


class OpenRouterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str
    openrouter_model: str = _DEFAULT_MODEL


class OpenRouterLLMClient:
    """LLMPort через OpenRouter API — команда выбрала API-путь для
    генерации (в отличие от self-hosted эмбеддингов): self-hosted LLM
    приемлемого для чата качества почти всегда требует GPU, иначе
    задержка неприемлема (явное требование куратора, см. CLAUDE.md).

    Модель по умолчанию — бесплатный тариф OpenRouter
    (meta-llama/llama-3.3-70b-instruct:free), чтобы не тратить бюджет на
    внутреннем тестировании (курсовая — "старт: Telegram-бот для
    внутренних тестов"). У бесплатных моделей лимит 20 запросов/мин —
    переключить на платную модель, когда понадобится больше, можно через
    OPENROUTER_MODEL, без изменений в коде.
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
