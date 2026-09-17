from __future__ import annotations

import hashlib


def stable_document_id(source_url: str) -> str:
    """Детерминированный id по source_url — повторный запуск ingestion
    для той же страницы/документа обновляет ту же запись (upsert), а не
    плодит дубликаты."""
    return hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:16]
