import pytest

from src.infrastructure.embeddings.qwen_local_embedder import QwenLocalEmbedder


class FakeEncoder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]):
        self.calls.append(texts)
        return [[float(i)] * self.dim for i in range(len(texts))]


@pytest.mark.asyncio
async def test_embed_returns_one_vector_per_text():
    embedder = QwenLocalEmbedder(encoder=FakeEncoder(dim=4))

    vectors = await embedder.embed(["привет", "мир"])

    assert len(vectors) == 2
    assert all(len(v) == 4 for v in vectors)


@pytest.mark.asyncio
async def test_embed_does_not_load_real_model_when_encoder_injected():
    """Инъекция encoder должна полностью обходить ленивую загрузку
    sentence-transformers — тест не должен качать веса из интернета."""
    fake = FakeEncoder()
    embedder = QwenLocalEmbedder(encoder=fake)

    await embedder.embed(["текст"])

    assert fake.calls == [["текст"]]


@pytest.mark.asyncio
async def test_embed_converts_values_to_plain_floats():
    embedder = QwenLocalEmbedder(encoder=FakeEncoder(dim=2))

    vectors = await embedder.embed(["a"])

    assert isinstance(vectors[0][0], float)


@pytest.mark.asyncio
async def test_embed_empty_input_returns_empty_list():
    embedder = QwenLocalEmbedder(encoder=FakeEncoder())

    vectors = await embedder.embed([])

    assert vectors == []
