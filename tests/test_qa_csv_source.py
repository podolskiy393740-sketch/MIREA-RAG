from pathlib import Path

from src.domain.entities import DocumentType
from src.infrastructure.chunking.chunker_selector import ChunkerSelector
from src.infrastructure.ingestion.qa_csv_source import load_documents_from_qa_csv

_FIXTURE = Path(__file__).parent / "fixtures" / "sample_qa_pairs.csv"


def test_groups_rows_by_source_url():
    documents = load_documents_from_qa_csv(_FIXTURE)

    assert len(documents) == 2
    urls = {doc.source_url for doc in documents}
    assert urls == {"https://example.test/doc-a.pdf", "https://example.test/doc-b.pdf"}


def test_concatenates_unique_answers_for_the_same_source():
    documents = load_documents_from_qa_csv(_FIXTURE)
    doc_a = next(d for d in documents if d.source_url == "https://example.test/doc-a.pdf")

    assert "Нужен паспорт и документ об образовании." in doc_a.raw_text
    assert "Документ об образовании можно донести позже установленного срока." in doc_a.raw_text


def test_rows_without_source_url_are_skipped():
    documents = load_documents_from_qa_csv(_FIXTURE)

    assert all("не должен попасть" not in doc.raw_text for doc in documents)


def test_documents_are_typed_as_unstructured_pdf():
    documents = load_documents_from_qa_csv(_FIXTURE)

    assert all(doc.doc_type is DocumentType.UNSTRUCTURED_PDF for doc in documents)


def test_document_ids_are_stable_across_calls():
    first_run = {d.source_url: d.id for d in load_documents_from_qa_csv(_FIXTURE)}
    second_run = {d.source_url: d.id for d in load_documents_from_qa_csv(_FIXTURE)}

    assert first_run == second_run


def test_resulting_documents_are_chunkable():
    """Связка qa_csv_source -> ChunkerSelector: UNSTRUCTURED_PDF должен
    маршрутизироваться в recursive-чанкер (см. CLAUDE.md/ARCHITECTURE.md)."""
    documents = load_documents_from_qa_csv(_FIXTURE)

    chunks = ChunkerSelector().chunk(documents[0])

    assert len(chunks) > 0
    assert chunks[0].chunking_strategy == "recursive"
