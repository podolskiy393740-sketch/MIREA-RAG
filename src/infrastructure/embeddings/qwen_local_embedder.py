from __future__ import annotations

import asyncio
from typing import Protocol

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain.entities import Vector


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qwen_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"


class _Encoder(Protocol):
    def encode(self, texts: list[str]): ...  # sentence-transformers возвращает numpy array


class QwenLocalEmbedder:
    """EmbedderPort: Qwen3-Embedding self-hosted через sentence-transformers.

    Модель выбрана командой (см. docs/embeddings-comparison.md) — self-hosted,
    без внешнего API, без сетевой задержки на каждый чанк/запрос —
    соответствует требованию куратора по снижению задержки.

    encode() у sentence-transformers синхронный и CPU/GPU-bound — весь
    вызов (включая ленивую загрузку весов при первом обращении) уходит в
    отдельный поток через asyncio.to_thread, чтобы не блокировать event
    loop Telegram-бота на время инференса.
    """

    def __init__(self, encoder: _Encoder | None = None, model_name: str | None = None) -> None:
        self._model_name = model_name or EmbeddingSettings().qwen_embedding_model
        self._encoder = encoder

    def _get_encoder(self) -> _Encoder:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    async def embed(self, texts: list[str]) -> list[Vector]:
        return await asyncio.to_thread(self._encode_sync, texts)

    def _encode_sync(self, texts: list[str]) -> list[Vector]:
        vectors = self._get_encoder().encode(texts)
        return [[float(x) for x in vector] for vector in vectors]
