from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

import httpx

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenAICompatibleClient:
    """Общая реализация LLMPort для любого провайдера с OpenAI-совместимым
    chat/completions API (сам OpenAI, OpenRouter и т.п. отдают одинаковый
    формат запроса/ответа) — эндпоинт, ключ и модель у каждого провайдера
    свои, а логика запроса и ретраев — одна и та же, вынесена сюда, чтобы
    не дублировать и не чинить в двух местах при следующей находке."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        client: httpx.AsyncClient | None = None,
        retries: int = 3,
        backoff_seconds: float = 5.0,
    ) -> None:
        self._api_url = api_url
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=30)
        self._retries = retries
        self._backoff_seconds = backoff_seconds

    async def generate(self, prompt: str) -> str:
        """Ретраит 429/5xx с бэкоффом — общие пулы бесплатных/перегруженных
        моделей реально их отдают под нагрузкой, это не баг клиента. Не
        различает "временная перегрузка" от "дневная квота исчерпана"
        (оба — просто 429) — второе ретраи не спасут, см. project memory."""
        last_error: Exception | None = None
        for attempt in range(self._retries):
            try:
                response = await self._client.post(
                    self._api_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self._model, "messages": [{"role": "user", "content": prompt}]},
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRYABLE_STATUS_CODES:
                    raise  # 400/401/404 и т.п. не пройдут повторно без изменений
                last_error = exc
            except httpx.TransportError as exc:
                last_error = exc
            if attempt < self._retries - 1:
                await asyncio.sleep(self._backoff_seconds * (attempt + 1))
        raise RuntimeError(f"{self._api_url} не ответил после {self._retries} попыток") from last_error

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Стриминг токенов через OpenAI-совместимый SSE-формат.
        Не ретраит — reconnect при стриминге сложнее и нецелесообразен."""
        async with self._client.stream(
            "POST",
            self._api_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload)
                    token = chunk["choices"][0]["delta"].get("content") or ""
                    if token:
                        yield token
                except Exception:
                    continue
