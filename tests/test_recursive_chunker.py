from src.domain.entities import Document, DocumentType
from src.infrastructure.chunking.recursive_chunker import RecursiveCharacterChunker


def _document(text: str) -> Document:
    return Document(id="doc-1", source_url="https://example.test", doc_type=DocumentType.PLAIN_TEXT, raw_text=text)


def test_short_text_becomes_single_chunk():
    chunker = RecursiveCharacterChunker(max_chars=1000, overlap=100)
    chunks = chunker.chunk(_document("Короткий текст без структуры."))

    assert len(chunks) == 1
    assert chunks[0].chunking_strategy == "recursive"
    assert chunks[0].document_id == "doc-1"


def test_long_text_is_split_within_max_chars():
    paragraph = "Регламент устанавливает порядок действий. " * 50
    chunker = RecursiveCharacterChunker(max_chars=200, overlap=30)

    chunks = chunker.chunk(_document(paragraph))

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 + 30 for c in chunks)


def test_consecutive_chunks_overlap():
    text = "Предложение номер " + " ".join(f"{i}." for i in range(200))
    chunker = RecursiveCharacterChunker(max_chars=100, overlap=20)

    chunks = chunker.chunk(_document(text))

    assert len(chunks) > 1
    tail_of_first = chunks[0].text[-10:]
    assert tail_of_first in chunks[1].text


def test_positions_are_sequential():
    chunker = RecursiveCharacterChunker(max_chars=50, overlap=10)
    chunks = chunker.chunk(_document("а " * 100))

    assert [c.position for c in chunks] == list(range(len(chunks)))
