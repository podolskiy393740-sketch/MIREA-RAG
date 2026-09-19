from __future__ import annotations

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.infrastructure.llm.openai_compatible_client import OpenAICompatibleClient


class LocalLLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama, vLLM, llama.cpp-server и т.п. отдают OpenAI-совместимый
    # /v1/chat/completions — поэтому отдельный протокол не нужен. Дефолт —
    # Ollama на той же машине.
    local_llm_url: str = "http://localhost:11434/v1/chat/completions"
    # Конкретная модель не выбрана куратором ("например, Qwen/Llama") —
    # это плейсхолдер, заменяется через .env.
    local_llm_model: str = "qwen3:8b"
    # Локальные серверы обычно не проверяют ключ, но часть клиентов ждёт
    # заголовок Authorization — отправляем заглушку.
    local_llm_api_key: str = "local"


class LocalLLMClient(OpenAICompatibleClient):
    """LLMPort к локальной модели, развёрнутой на серверах вуза.

    Это ПРОД-путь (ФЗ-152, см. CLAUDE.md): данные личного кабинета —
    курс, факультет, долги по оплате — нельзя отправлять во внешний API,
    даже без ФИО. Внешние клиенты (OpenAIClient, OpenRouterLLMClient) —
    только прототип, выбор между ними делает llm.factory.build_llm().
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        api_url: str | None = None,
        model: str | None = None,
        retries: int = 3,
        backoff_seconds: float = 5.0,
    ) -> None:
        settings = LocalLLMSettings()
        super().__init__(
            api_url=api_url if api_url is not None else settings.local_llm_url,
            api_key=settings.local_llm_api_key,
            model=model if model is not None else settings.local_llm_model,
            # Инференс на CPU медленный — таймаут щедрее, чем у внешних API.
            client=client or httpx.AsyncClient(timeout=180),
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
