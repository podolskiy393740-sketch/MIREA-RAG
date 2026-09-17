from pathlib import Path

from src.domain.entities import DocumentType
from src.infrastructure.chunking.chunker_selector import ChunkerSelector
from src.infrastructure.ingestion.raw_html_source import load_documents_from_raw_html_manifest

_MANIFEST = Path(__file__).parent / "fixtures" / "raw_html" / "manifest.json"


def test_loads_one_document_per_manifest_entry():
    documents = load_documents_from_raw_html_manifest(_MANIFEST)

    assert len(documents) == 1
    assert documents[0].source_url == "https://example.test/sample-page/"


def test_document_is_structured_html_type():
    documents = load_documents_from_raw_html_manifest(_MANIFEST)

    assert documents[0].doc_type is DocumentType.STRUCTURED_HTML


def test_mega_menu_excluded_but_content_kept():
    documents = load_documents_from_raw_html_manifest(_MANIFEST)

    assert "Абитуриентам" not in documents[0].raw_text
    assert "Тестовая страница" in documents[0].raw_text
    assert "Текст раздела." in documents[0].raw_text


def test_document_ids_are_stable_across_calls():
    first = {d.source_url: d.id for d in load_documents_from_raw_html_manifest(_MANIFEST)}
    second = {d.source_url: d.id for d in load_documents_from_raw_html_manifest(_MANIFEST)}

    assert first == second


def test_resulting_document_is_chunkable_with_structure_aware():
    documents = load_documents_from_raw_html_manifest(_MANIFEST)

    chunks = ChunkerSelector().chunk(documents[0])

    assert len(chunks) > 0
    assert chunks[0].chunking_strategy == "structure_aware"
