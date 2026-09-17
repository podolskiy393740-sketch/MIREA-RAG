import pytest

from src.application.use_cases import IngestDocumentUseCase
from src.domain.entities import Chunk, Document, DocumentType, Vector
from src.infrastructure.chunking.chunker_selector import ChunkerSelector


class FakeDocumentStore:
    def __init__(self) -> None:
        self.upserted: list[Document] = []

    async def upsert(self, document: Document) -> None:
        self.upserted.append(document)


class FakeFullTextStore:
    def __init__(self) -> None:
        self.upserted: list[Chunk] = []

    async def upsert(self, chunk: Chunk) -> None:
        self.upserted.append(chunk)

    async def search(self, query_text: str, top_k: int) -> list[Chunk]:
        raise NotImplementedError


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: list[tuple[Chunk, Vector]] = []

    async def upsert(self, chunk: Chunk, embedding: Vector) -> None:
        self.upserted.append((chunk, embedding))

    async def search(self, query_embedding: Vector, top_k: int) -> list[Chunk]:
        raise NotImplementedError


class FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[Vector]:
        return [[float(len(text))] for text in texts]


def _document() -> Document:
    return Document(
        id="doc-1", source_url="https://mirea.ru/faq", doc_type=DocumentType.PLAIN_TEXT,
        raw_text="Первый вопрос про общежитие. Второй вопрос про стипендию.",
    )


@pytest.mark.asyncio
async def test_ingest_without_embedder_only_fills_fulltext_store():
    document_store = FakeDocumentStore()
    fulltext_store = FakeFullTextStore()
    use_case = IngestDocumentUseCase(
        chunker=ChunkerSelector(), document_store=document_store, fulltext_store=fulltext_store
    )

    chunks = await use_case.execute(_document())

    assert len(fulltext_store.upserted) == len(chunks)
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_ingest_saves_document_before_chunks():
    """chunks.document_id — внешний ключ на documents.id в Postgres:
    документ обязан быть сохранён до чанков, иначе будет ForeignKeyViolationError."""
    document_store = FakeDocumentStore()
    document = _document()
    use_case = IngestDocumentUseCase(
        chunker=ChunkerSelector(), document_store=document_store, fulltext_store=FakeFullTextStore()
    )

    await use_case.execute(document)

    assert document_store.upserted == [document]


@pytest.mark.asyncio
async def test_ingest_with_embedder_also_fills_vector_store():
    fulltext_store = FakeFullTextStore()
    vector_store = FakeVectorStore()
    use_case = IngestDocumentUseCase(
        chunker=ChunkerSelector(),
        document_store=FakeDocumentStore(),
        fulltext_store=fulltext_store,
        vector_store=vector_store,
        embedder=FakeEmbedder(),
    )

    chunks = await use_case.execute(_document())

    assert len(vector_store.upserted) == len(chunks)
    assert len(fulltext_store.upserted) == len(chunks)


@pytest.mark.asyncio
async def test_ingest_returns_chunks_from_selected_strategy():
    document = Document(
        id="doc-2", source_url="u", doc_type=DocumentType.STRUCTURED_HTML, raw_text="# Заголовок\n\nТекст раздела."
    )
    use_case = IngestDocumentUseCase(
        chunker=ChunkerSelector(), document_store=FakeDocumentStore(), fulltext_store=FakeFullTextStore()
    )

    chunks = await use_case.execute(document)

    assert chunks[0].chunking_strategy == "structure_aware"
