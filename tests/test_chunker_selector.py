from src.domain.entities import Document, DocumentType
from src.infrastructure.chunking.chunker_selector import ChunkerSelector


def test_routes_structured_html_to_structure_aware():
    document = Document(
        id="d1", source_url="u", doc_type=DocumentType.STRUCTURED_HTML, raw_text="# Заголовок\n\nТекст."
    )

    chunks = ChunkerSelector().chunk(document)

    assert chunks[0].chunking_strategy == "structure_aware"


def test_routes_unstructured_pdf_to_recursive():
    document = Document(
        id="d2", source_url="u", doc_type=DocumentType.UNSTRUCTURED_PDF, raw_text="Сплошной текст регламента."
    )

    chunks = ChunkerSelector().chunk(document)

    assert chunks[0].chunking_strategy == "recursive"


def test_routes_plain_text_to_recursive():
    document = Document(id="d3", source_url="u", doc_type=DocumentType.PLAIN_TEXT, raw_text="Просто текст.")

    chunks = ChunkerSelector().chunk(document)

    assert chunks[0].chunking_strategy == "recursive"
