from __future__ import annotations

import csv
from pathlib import Path

from src.domain.entities import Document, DocumentType
from src.infrastructure.ingestion.document_id import stable_document_id


def load_documents_from_qa_csv(csv_path: str | Path) -> list[Document]:
    """Строит псевдо-документы из CSV вида question,answer,source_url,topic.

    Формат совпадает с data/qa_pairs.csv из референсного проекта
    chernenko-s/mirea-rag: несколько строк указывают на один и тот же
    source_url (несколько вопросов-ответов по одному официальному
    документу). Группируем по source_url и склеиваем уникальные ответы —
    это приближение к содержимому исходного документа, а не сами вопросы:
    в retrieval должен попадать контент, а не синтетические вопросы.

    doc_type = UNSTRUCTURED_PDF, т.к. source_url в этом датасете — PDF
    (приказы, положения), а не размеченные HTML-страницы. Это не решение
    по эмбеддингам/данным за команду, а просто корректная маршрутизация
    в уже зафиксированную (см. CLAUDE.md) гибридную стратегию чанкинга:
    recursive character splitting для текста без структуры.
    """
    rows = _read_rows(csv_path)

    answers_by_source: dict[str, list[str]] = {}
    order: list[str] = []
    for row in rows:
        source_url = (row.get("source_url") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not source_url or not answer:
            continue
        if source_url not in answers_by_source:
            answers_by_source[source_url] = []
            order.append(source_url)
        if answer not in answers_by_source[source_url]:
            answers_by_source[source_url].append(answer)

    return [
        Document(
            id=stable_document_id(source_url),
            source_url=source_url,
            doc_type=DocumentType.UNSTRUCTURED_PDF,
            raw_text="\n\n".join(answers_by_source[source_url]),
        )
        for source_url in order
    ]


def _read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
