from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

_CDX_URL = "https://web.archive.org/cdx/search/cdx"
_USER_AGENT = "MIREA-RAG-coursework/0.1 (contact: podolskiy393740@gmail.com)"


@dataclass(frozen=True)
class Snapshot:
    original_url: str
    timestamp: str

    @property
    def raw_url(self) -> str:
        # "id_" -> отдать архивную копию как есть, без тулбара Wayback Machine.
        return f"https://web.archive.org/web/{self.timestamp}id_/{self.original_url}"


async def find_latest_snapshot(
    client: httpx.AsyncClient, url: str, retries: int = 3, backoff_seconds: float = 5.0
) -> Snapshot | None:
    """Ищет самый свежий доступный снапшот страницы в Wayback Machine.

    Используется вместо прямого обращения к mirea.ru: сайт закрыт
    DDoS-Guard по гео-блокировке для нашей инфраструктуры (см.
    docs/data-collection.md), а web.archive.org отдаёт кэш публично
    опубликованных страниц с отдельного сервера, не задевая mirea.ru.

    С ретраями: архив периодически отвечает "Temporarily Offline"
    страницей вместо JSON (валидный HTTP 200, невалидный JSON) или рвёт
    соединение по таймауту — это временная перегрузка архива, а не
    признак того, что URL не существует.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = await client.get(
                _CDX_URL,
                params={"url": url, "output": "json", "filter": "statuscode:200", "fl": "timestamp", "limit": "-1"},
                headers={"User-Agent": _USER_AGENT},
                timeout=30,
            )
            response.raise_for_status()
            rows = response.json()
            if len(rows) < 2:  # первая строка — заголовок колонки
                return None
            return Snapshot(original_url=url, timestamp=rows[-1][0])
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
        if attempt < retries - 1:
            await asyncio.sleep(backoff_seconds * (attempt + 1))
    raise RuntimeError(f"Не удалось получить список снапшотов для {url}") from last_error


async def fetch_snapshot_html(
    client: httpx.AsyncClient, snapshot: Snapshot, retries: int = 3, backoff_seconds: float = 5.0
) -> str:
    """Скачивает HTML снапшота с ретраями — архив периодически отдаёт
    503/пустые ответы под нагрузкой, это не признак ошибки в URL."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = await client.get(snapshot.raw_url, headers={"User-Agent": _USER_AGENT}, timeout=30)
            response.raise_for_status()
            if len(response.text) > 1000:
                return response.text
            last_error = ValueError(f"Ответ подозрительно короткий ({len(response.text)} байт)")
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
        if attempt < retries - 1:
            await asyncio.sleep(backoff_seconds * (attempt + 1))
    raise RuntimeError(f"Не удалось получить снапшот {snapshot.raw_url}") from last_error
