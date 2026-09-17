from src.domain.entities import Document, DocumentType
from src.infrastructure.chunking.structure_aware_chunker import StructureAwareChunker

_STRUCTURED_TEXT = """# Положение о военной кафедре

## Общие сведения

Военная кафедра проводит обучение студентов по программе подготовки
офицеров запаса.

## Порядок зачисления

Зачисление проводится на конкурсной основе по результатам медицинской
комиссии и профессионального отбора.
"""


def _document(text: str) -> Document:
    return Document(
        id="doc-42", source_url="https://mirea.ru/military", doc_type=DocumentType.STRUCTURED_HTML, raw_text=text
    )


def test_splits_by_headings():
    chunker = StructureAwareChunker(max_chars=1000)
    chunks = chunker.chunk(_document(_STRUCTURED_TEXT))

    assert len(chunks) == 2
    assert all(c.chunking_strategy == "structure_aware" for c in chunks)


def test_chunk_carries_heading_path_as_context():
    chunker = StructureAwareChunker(max_chars=1000)
    chunks = chunker.chunk(_document(_STRUCTURED_TEXT))

    assert "Порядок зачисления" in chunks[1].text
    assert "Положение о военной кафедре" in chunks[1].text


def test_long_section_is_split_by_paragraphs():
    long_section = "# Раздел\n\n" + "\n\n".join(f"Абзац номер {i} с содержанием пункта." for i in range(20))
    chunker = StructureAwareChunker(max_chars=150)

    chunks = chunker.chunk(_document(long_section))

    assert len(chunks) > 1


def test_text_without_headings_is_still_chunked():
    chunker = StructureAwareChunker(max_chars=1000)
    chunks = chunker.chunk(_document("Просто абзац без заголовков вообще."))

    assert len(chunks) == 1
    assert chunks[0].text == "Просто абзац без заголовков вообще."
