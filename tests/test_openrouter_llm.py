import httpx
import pytest

from src.infrastructure.llm.openrouter_llm import OpenRouterLLMClient


@pytest.mark.asyncio
async def test_generate_returns_message_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "Стипендия 25 числа."}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        llm = OpenRouterLLMClient(client=http_client, api_key="test-key", model="test-model")
        answer = await llm.generate("когда стипендия?")

    assert answer == "Стипендия 25 числа."


@pytest.mark.asyncio
async def test_generate_sends_model_prompt_and_auth_header():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = request.content
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        llm = OpenRouterLLMClient(client=http_client, api_key="secret-key", model="some/model")
        await llm.generate("тестовый промпт")

    assert captured["auth"] == "Bearer secret-key"
    assert b"some/model" in captured["body"]
    assert "тестовый промпт".encode() in captured["body"]


@pytest.mark.asyncio
async def test_generate_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        llm = OpenRouterLLMClient(client=http_client, api_key="bad-key", model="test-model")
        with pytest.raises(httpx.HTTPStatusError):
            await llm.generate("вопрос")
