from src.infrastructure.llm.factory import LLMMode, LLMModeSettings, build_llm
from src.infrastructure.llm.local_llm import LocalLLMClient
from src.infrastructure.llm.openai_llm import OpenAIClient
from src.infrastructure.llm.openrouter_llm import OpenRouterLLMClient


def test_local_mode_never_builds_external_client_even_with_keys_in_env(monkeypatch):
    """ФЗ-152: ключ, оставшийся в окружении после прототипа, не должен
    приводить к отправке данных во внешний API в прод-режиме."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-leftover-from-prototype")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-leftover")

    llm = build_llm(LLMMode.LOCAL)

    assert isinstance(llm, LocalLLMClient)
    assert not isinstance(llm, (OpenAIClient, OpenRouterLLMClient))


def test_default_mode_is_local(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_MODE", raising=False)
    monkeypatch.chdir(tmp_path)  # чтобы не подхватить реальный .env из корня репозитория

    assert LLMModeSettings().llm_mode is LLMMode.LOCAL


def test_prototype_mode_prefers_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    assert isinstance(build_llm(LLMMode.PROTOTYPE), OpenAIClient)


def test_prototype_mode_falls_back_to_openrouter_without_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    assert isinstance(build_llm(LLMMode.PROTOTYPE), OpenRouterLLMClient)


def test_prototype_mode_without_any_key_returns_none_for_human_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    assert build_llm(LLMMode.PROTOTYPE) is None


def test_mode_is_read_from_env_when_not_passed_explicitly(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "prototype")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert isinstance(build_llm(), OpenAIClient)


def test_local_client_targets_openai_compatible_local_endpoint():
    llm = LocalLLMClient(api_url="http://localhost:9999/v1/chat/completions", model="test-model")

    assert llm._api_url == "http://localhost:9999/v1/chat/completions"
    assert llm._model == "test-model"
