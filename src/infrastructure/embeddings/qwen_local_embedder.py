from __future__ import annotations

import asyncio
import threading
from typing import Protocol

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain.entities import Vector


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qwen_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"


class _Encoder(Protocol):
    def encode(self, texts: list[str], batch_size: int = 32): ...  # sentence-transformers возвращает numpy array


_ENCODE_LOCK = threading.Lock()


class QwenLocalEmbedder:
    """EmbedderPort: Qwen3-Embedding self-hosted через sentence-transformers.

    Модель выбрана командой (см. docs/embeddings-comparison.md) — self-hosted,
    без внешнего API, без сетевой задержки на каждый чанк/запрос —
    соответствует требованию куратора по снижению задержке.

    encode() у sentence-transformers синхронный и CPU/GPU-bound — весь
    вызов (включая ленивую загрузку весов при первом обращении) уходит в
    отдельный поток через asyncio.to_thread, чтобы не блокировать event
    loop Telegram-бота на время инференса.

    float16 + batch_size=1: на серверах с малым объёмом RAM (~1 ГБ)
    загрузка float32 (2.4 ГБ весов) вызывает OOM kill во время инференса.
    float16 вдвое снижает размер весов; batch_size=1 убирает пиковые
    аллокации активационных тензоров при обработке батча.
    """

    # batch_size=1 — защита от OOM на CPU-серверах с малым объёмом RAM.
    _ENCODE_BATCH_SIZE = 1

    def __init__(self, encoder: _Encoder | None = None, model_name: str | None = None) -> None:
        self._model_name = model_name or EmbeddingSettings().qwen_embedding_model
        self._encoder = encoder

    def _get_encoder(self) -> _Encoder:
        if self._encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self._model_name,
                model_kwargs={"torch_dtype": torch.float16},
            )
        return self._encoder

    async def embed(self, texts: list[str]) -> list[Vector]:
        return await asyncio.to_thread(self._encode_sync, texts)

    def _encode_sync(self, texts: list[str]) -> list[Vector]:
        with _ENCODE_LOCK:
            vectors = self._get_encoder().encode(texts, batch_size=self._ENCODE_BATCH_SIZE)
        return [[float(x) for x in vector] for vector in vectors]
