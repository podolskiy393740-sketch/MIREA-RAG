from __future__ import annotations

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.infrastructure.llm.openai_compatible_client import OpenAICompatibleClient

_API_URL = "https://api.openai.com/v1/chat/completions"
# gpt-5.4-nano — текущее поколение, самый дешёвый тир в нём ($0.20/$1.25
# за 1M токенов на сентябрь 2026, проверено веб-поиском при выборе, не по
# памяти — цены/линейки OpenAI меняются). Решение команды: GPT как
# основной путь генерации вместо бесплатного тира OpenRouter (см.
# openrouter_llm.py) — деньги реальные, ключ платный.
_DEFAULT_MODEL = "gpt-5.4-nano"


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    openai_model: str = _DEFAULT_MODEL


class OpenAIClient(OpenAICompatibleClient):
    """LLMPort напрямую через OpenAI API — основной путь генерации (см.
    CLAUDE.md). Платный ключ команды, поэтому в отличие от
    OpenRouterLLMClient (бесплатный тир, нестабильный) должен работать
    предсказуемо, но ретраи на 429/5xx всё равно оставлены: даже платный
    API может кратковременно вернуть 429/503 под нагрузкой у самого OpenAI.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
        model: str | None = None,
        retries: int = 3,
        backoff_seconds: float = 5.0,
    ) -> None:
        resolved_api_key = api_key if api_key is not None else OpenAISettings().openai_api_key
        resolved_model = model if model is not None else OpenAISettings().openai_model
        super().__init__(
            api_url=_API_URL,
            api_key=resolved_api_key,
            model=resolved_model,
            client=client,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
