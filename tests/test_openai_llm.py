import httpx
import pytest

from src.infrastructure.llm.openai_llm import OpenAIClient


@pytest.mark.asyncio
async def test_generate_returns_message_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "Стипендия 25 числа."}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        llm = OpenAIClient(client=http_client, api_key="test-key", model="gpt-5.4-nano")
        answer = await llm.generate("когда стипендия?")

    assert answer == "Стипендия 25 числа."


@pytest.mark.asyncio
async def test_generate_sends_model_prompt_and_auth_header():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = request.content
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        llm = OpenAIClient(client=http_client, api_key="secret-key", model="gpt-5.4-nano")
        await llm.generate("тестовый промпт")

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["auth"] == "Bearer secret-key"
    assert b"gpt-5.4-nano" in captured["body"]
    assert "тестовый промпт".encode() in captured["body"]


@pytest.mark.asyncio
async def test_generate_raises_on_non_retryable_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        llm = OpenAIClient(client=http_client, api_key="bad-key", model="gpt-5.4-nano")
        with pytest.raises(httpx.HTTPStatusError):
            await llm.generate("вопрос")


@pytest.mark.asyncio
async def test_generate_retries_on_429_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        llm = OpenAIClient(client=http_client, api_key="k", model="gpt-5.4-nano", retries=3, backoff_seconds=0)
        answer = await llm.generate("вопрос")

    assert answer == "ok"
    assert calls["count"] == 2


def test_default_model_is_gpt_5_4_nano_when_not_overridden():
    from src.infrastructure.llm.openai_llm import _DEFAULT_MODEL

    assert _DEFAULT_MODEL == "gpt-5.4-nano"
