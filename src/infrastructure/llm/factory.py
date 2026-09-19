from __future__ import annotations

import logging
from enum import Enum

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain.ports import LLMPort
from src.infrastructure.llm.local_llm import LocalLLMClient
from src.infrastructure.llm.openai_llm import OpenAIClient, OpenAISettings
from src.infrastructure.llm.openrouter_llm import OpenRouterLLMClient, OpenRouterSettings

logger = logging.getLogger(__name__)


class LLMMode(str, Enum):
    LOCAL = "local"
    PROTOTYPE = "prototype"


class LLMModeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Безопасный дефолт: без явной настройки внешний API не используется
    # никогда. Куратор (ФЗ-152, CLAUDE.md): временное решение не должно
    # попадать в прод-конфигурацию "по умолчанию".
    llm_mode: LLMMode = LLMMode.LOCAL


def build_llm(mode: LLMMode | None = None) -> LLMPort | None:
    """Единственное место, где выбирается LLM для бота.

    LOCAL — прод: локальная модель на серверах вуза. Внешние клиенты не
    создаются вообще, даже если ключи лежат в окружении — так ключ,
    оставшийся в .env после прототипа, не может увести данные ЛКС наружу.

    PROTOTYPE — только для разработки и eval на публичных данных: OpenAI,
    иначе OpenRouter, иначе None (бот работает в режиме human-фолбека).
    """
    mode = mode if mode is not None else LLMModeSettings().llm_mode

    if mode is LLMMode.LOCAL:
        logger.info("LLM mode=local: локальная модель (прод-путь, ФЗ-152)")
        return LocalLLMClient()

    logger.warning("LLM mode=prototype: внешний API — только для прототипа, НЕ для прода (ФЗ-152)")
    try:
        openai_settings = OpenAISettings()
        if openai_settings.openai_api_key:  # пустая строка — валидный str для pydantic
            return OpenAIClient(api_key=openai_settings.openai_api_key, model=openai_settings.openai_model)
    except ValidationError:
        pass

    try:
        openrouter_settings = OpenRouterSettings()
        if openrouter_settings.openrouter_api_key:
            return OpenRouterLLMClient(
                api_key=openrouter_settings.openrouter_api_key, model=openrouter_settings.openrouter_model
            )
    except ValidationError:
        pass

    return None
