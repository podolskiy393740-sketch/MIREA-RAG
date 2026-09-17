from __future__ import annotations

import json
from pathlib import Path

from src.domain.entities import Document, DocumentType
from src.infrastructure.ingestion.document_id import stable_document_id
from src.infrastructure.ingestion.html_parser import html_to_markdown


def load_documents_from_raw_html_manifest(manifest_path: str | Path) -> list[Document]:
    """Строит документы из HTML-страниц, собранных через Wayback Machine
    (см. scripts/fetch_real_pages.py, docs/data-collection.md).

    manifest.json лежит рядом с *.html-файлами (data/raw_html/); каждая
    запись — {name, source_url, wayback_timestamp, raw_url}. HTML-файл
    ищется по <name>.html в той же папке.

    doc_type = STRUCTURED_HTML: это реальные страницы mirea.ru с разметкой
    (заголовки h1/h2), html_to_markdown уже вычленяет из них текст статьи
    (см. html_parser.py) — гибридная стратегия чанкинга направит их в
    StructureAwareChunker.
    """
    manifest_path = Path(manifest_path)
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    html_dir = manifest_path.parent

    documents = []
    for entry in entries:
        html_path = html_dir / f"{entry['name']}.html"
        html = html_path.read_text(encoding="utf-8")
        markdown = html_to_markdown(html)
        documents.append(
            Document(
                id=stable_document_id(entry["source_url"]),
                source_url=entry["source_url"],
                doc_type=DocumentType.STRUCTURED_HTML,
                raw_text=markdown,
            )
        )
    return documents
