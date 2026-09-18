from __future__ import annotations

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.infrastructure.llm.openai_compatible_client import OpenAICompatibleClient

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


class OpenRouterLLMClient(OpenAICompatibleClient):
    """LLMPort через OpenRouter API. Команда переключилась на прямой
    OpenAI (см. openai_llm.py) как основной путь генерации — этот клиент
    остаётся рабочей альтернативой (например, для сравнения на защите).

    Бесплатный тир на практике нестабилен и вдобавок ограничен жёсткой
    дневной квотой (не только "временной перегрузкой" — см. project
    memory про 429 free-models-per-day). AnswerQuestionUseCase перехватывает
    такие сбои и уходит в human-фолбек, так что бот не падает, но на
    бесплатном тире ответы иногда будут "передано техподдержке" не из-за
    отсутствия данных, а из-за исчерпанной квоты/перегрузки апстрима.
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
        model: str | None = None,
        retries: int = 3,
        backoff_seconds: float = 5.0,
    ) -> None:
        resolved_api_key = api_key if api_key is not None else OpenRouterSettings().openrouter_api_key
        resolved_model = model if model is not None else OpenRouterSettings().openrouter_model
        super().__init__(
            api_url=_API_URL,
            api_key=resolved_api_key,
            model=resolved_model,
            client=client,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
