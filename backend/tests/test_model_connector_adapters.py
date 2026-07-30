"""Per-adapter reasoning/thinking-model payload shaping and model-list filtering.

Each adapter must send the right provider-specific reasoning parameter (or none)
based on the selected model's ReasoningStyle, and must never send a parameter an
incompatible model would reject (e.g. temperature alongside Anthropic thinking).
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from domain.model_connector.anthropic.adapter import AnthropicAdapter
from domain.model_connector.deepseek.adapter import DeepSeekAdapter
from domain.model_connector.gemini.adapter import GeminiAdapter
from domain.model_connector.lmstudio.adapter import LMStudioAdapter
from domain.model_connector.ollama.adapter import OllamaAdapter
from domain.model_connector.openai.adapter import OpenAIAdapter
from domain.model_connector.types import ChatMessage, ChatRequest, ProviderConfig, ProviderKind


def _config(kind: ProviderKind, model_id: str) -> ProviderConfig:
    return ProviderConfig(
        id="p1", kind=kind, display_name="test", base_url="http://test", model_id=model_id,
        extra={"api_key": "key"},
    )


def _request(**overrides) -> ChatRequest:
    base = dict(
        provider_id="p1",
        messages=[ChatMessage(role="user", content="hi")],
        max_completion_tokens=2048,
    )
    base.update(overrides)
    return ChatRequest(**base)


def _fake_response(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body, request=httpx.Request("POST", "http://test/x"))


# ── OpenAI ──────────────────────────────────────────────────────────────────

def test_openai_sends_reasoning_effort_for_reasoning_model() -> None:
    adapter = OpenAIAdapter(_config(ProviderKind.OPENAI, "o3-mini"))
    payload = adapter._build_payload(_request(reasoning_effort="high", temperature=0.2))
    assert payload["reasoning_effort"] == "high"
    assert "temperature" not in payload


def test_openai_omits_reasoning_effort_for_regular_model() -> None:
    adapter = OpenAIAdapter(_config(ProviderKind.OPENAI, "gpt-4o"))
    payload = adapter._build_payload(_request(reasoning_effort="high", temperature=0.2))
    assert "reasoning_effort" not in payload
    assert payload["temperature"] == 0.2


# ── DeepSeek ────────────────────────────────────────────────────────────────

def test_deepseek_reasoner_omits_temperature_and_reasoning_effort() -> None:
    adapter = DeepSeekAdapter(_config(ProviderKind.DEEPSEEK, "deepseek-reasoner"))
    payload = adapter._build_payload(_request(reasoning_effort="high", temperature=0.2))
    assert "temperature" not in payload
    assert "reasoning_effort" not in payload


def test_deepseek_chat_keeps_temperature() -> None:
    adapter = DeepSeekAdapter(_config(ProviderKind.DEEPSEEK, "deepseek-chat"))
    payload = adapter._build_payload(_request(temperature=0.3))
    assert payload["temperature"] == 0.3


# ── Anthropic ───────────────────────────────────────────────────────────────

async def test_anthropic_thinking_model_sends_budget_tokens_no_temperature() -> None:
    adapter = AnthropicAdapter(_config(ProviderKind.ANTHROPIC, "claude-opus-4-5"))
    adapter._client.post = AsyncMock(
        return_value=_fake_response({"content": [{"type": "text", "text": "ok"}], "usage": {}})
    )
    await adapter.chat(_request(temperature=0.2, thinking_budget=2048, max_completion_tokens=4096))

    sent = adapter._client.post.call_args.kwargs["json"]
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in sent


async def test_anthropic_non_thinking_model_keeps_temperature() -> None:
    adapter = AnthropicAdapter(_config(ProviderKind.ANTHROPIC, "claude-3-5-haiku-20241022"))
    adapter._client.post = AsyncMock(
        return_value=_fake_response({"content": [{"type": "text", "text": "ok"}], "usage": {}})
    )
    await adapter.chat(_request(temperature=0.2, thinking_budget=2048))

    sent = adapter._client.post.call_args.kwargs["json"]
    assert "thinking" not in sent
    assert sent["temperature"] == 0.2


async def test_anthropic_thinking_budget_clamped_below_max_tokens() -> None:
    adapter = AnthropicAdapter(_config(ProviderKind.ANTHROPIC, "claude-sonnet-4-5"))
    adapter._client.post = AsyncMock(
        return_value=_fake_response({"content": [{"type": "text", "text": "ok"}], "usage": {}})
    )
    await adapter.chat(_request(thinking_budget=99999, max_completion_tokens=1000))

    sent = adapter._client.post.call_args.kwargs["json"]
    assert sent["thinking"]["budget_tokens"] < 1000


# ── Gemini ──────────────────────────────────────────────────────────────────

async def test_gemini_thinking_model_sends_thinking_config() -> None:
    adapter = GeminiAdapter(_config(ProviderKind.GEMINI, "gemini-2.5-pro"))
    adapter._client.post = AsyncMock(
        return_value=_fake_response({"candidates": [], "usageMetadata": {}})
    )
    await adapter.chat(_request(thinking_budget=5000))

    sent = adapter._client.post.call_args.kwargs["json"]
    assert sent["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 5000


async def test_gemini_non_thinking_model_omits_thinking_config() -> None:
    adapter = GeminiAdapter(_config(ProviderKind.GEMINI, "gemini-1.5-pro"))
    adapter._client.post = AsyncMock(
        return_value=_fake_response({"candidates": [], "usageMetadata": {}})
    )
    await adapter.chat(_request(thinking_budget=5000))

    sent = adapter._client.post.call_args.kwargs["json"]
    assert "thinkingConfig" not in sent["generationConfig"]


async def test_gemini_pro_thinking_budget_clamped_to_minimum() -> None:
    adapter = GeminiAdapter(_config(ProviderKind.GEMINI, "gemini-2.5-pro"))
    adapter._client.post = AsyncMock(
        return_value=_fake_response({"candidates": [], "usageMetadata": {}})
    )
    await adapter.chat(_request(thinking_budget=1))  # below Pro's min of 128

    sent = adapter._client.post.call_args.kwargs["json"]
    assert sent["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 128


# ── Ollama ──────────────────────────────────────────────────────────────────

async def test_ollama_reasoning_model_sends_think_true() -> None:
    adapter = OllamaAdapter(_config(ProviderKind.OLLAMA, "deepseek-r1:8b"))
    adapter._client.post = AsyncMock(
        return_value=_fake_response({"message": {"content": "ok"}, "usage": {}})
    )
    await adapter.chat(_request(thinking_budget=1))

    sent = adapter._client.post.call_args.kwargs["json"]
    assert sent["think"] is True


async def test_ollama_regular_model_omits_think() -> None:
    adapter = OllamaAdapter(_config(ProviderKind.OLLAMA, "llama3:8b"))
    adapter._client.post = AsyncMock(
        return_value=_fake_response({"message": {"content": "ok"}, "usage": {}})
    )
    await adapter.chat(_request(thinking_budget=1))

    sent = adapter._client.post.call_args.kwargs["json"]
    assert "think" not in sent


async def test_ollama_list_models_excludes_embedding_models() -> None:
    adapter = OllamaAdapter(_config(ProviderKind.OLLAMA, "llama3:8b"))
    adapter._client.get = AsyncMock(
        return_value=_fake_response_get(
            {"models": [{"name": "llama3:8b"}, {"name": "nomic-embed-text:latest"}]}
        )
    )
    models = await adapter.list_models()
    assert models == ["llama3:8b"]


# ── LM Studio ───────────────────────────────────────────────────────────────

async def test_lmstudio_list_models_excludes_embedding_models() -> None:
    adapter = LMStudioAdapter(_config(ProviderKind.LM_STUDIO, "some-model"))
    adapter._client.get = AsyncMock(
        return_value=_fake_response_get(
            {"data": [{"id": "llama-3-8b-instruct"}, {"id": "text-embedding-bert"}]}
        )
    )
    models = await adapter.list_models()
    assert models == ["llama-3-8b-instruct"]


def _fake_response_get(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body, request=httpx.Request("GET", "http://test/x"))


# ── Base URL Configuration ──────────────────────────────────────────────────

def test_openai_adapter_respects_configured_base_url() -> None:
    config = _config(ProviderKind.OPENAI, "gpt-4o")
    adapter = OpenAIAdapter(config)
    assert str(adapter._client.base_url).rstrip("/") == "http://test"


def test_anthropic_adapter_respects_configured_base_url() -> None:
    config = _config(ProviderKind.ANTHROPIC, "claude-3-5-sonnet")
    adapter = AnthropicAdapter(config)
    assert str(adapter._client.base_url).rstrip("/") == "http://test"


def test_gemini_adapter_respects_configured_base_url() -> None:
    config = _config(ProviderKind.GEMINI, "gemini-1.5-pro")
    adapter = GeminiAdapter(config)
    assert str(adapter._client.base_url).rstrip("/") == "http://test"


def test_deepseek_adapter_respects_configured_base_url() -> None:
    config = _config(ProviderKind.DEEPSEEK, "deepseek-chat")
    adapter = DeepSeekAdapter(config)
    assert str(adapter._client.base_url).rstrip("/") == "http://test"


def test_adapters_fallback_to_default_base_url_when_config_empty() -> None:
    def _empty_config(kind: ProviderKind, model_id: str) -> ProviderConfig:
        return ProviderConfig(
            id="p1",
            kind=kind,
            display_name="test",
            base_url="",  # empty base url to force fallback
            model_id=model_id,
            extra={"api_key": "key"},
        )

    # OpenAI default
    openai_adapter = OpenAIAdapter(_empty_config(ProviderKind.OPENAI, "gpt-4o"))
    assert str(openai_adapter._client.base_url).rstrip("/") == "https://api.openai.com"

    # Anthropic default
    anthropic_adapter = AnthropicAdapter(_empty_config(ProviderKind.ANTHROPIC, "claude-3-5-sonnet"))
    assert str(anthropic_adapter._client.base_url).rstrip("/") == "https://api.anthropic.com"

    # Gemini default
    gemini_adapter = GeminiAdapter(_empty_config(ProviderKind.GEMINI, "gemini-1.5-pro"))
    assert str(gemini_adapter._client.base_url).rstrip("/") == "https://generativelanguage.googleapis.com"

    # DeepSeek default
    deepseek_adapter = DeepSeekAdapter(_empty_config(ProviderKind.DEEPSEEK, "deepseek-chat"))
    assert str(deepseek_adapter._client.base_url).rstrip("/") == "https://api.deepseek.com"


# ── Non-vendor-shaped model lists (3rd-party gateways) ──────────────────────

async def test_openai_list_models_returns_raw_ids_when_prefix_filter_zeroes_out() -> None:
    """A real, non-empty response whose model IDs don't match OpenAI's own naming
    (e.g. a 3rd-party gateway proxying other vendors) must surface those real IDs,
    not silently substitute fake OpenAI preset names."""
    adapter = OpenAIAdapter(_config(ProviderKind.OPENAI, "gpt-4o"))
    adapter._client.get = AsyncMock(
        return_value=_fake_response_get(
            {"data": [{"id": "DeepSeek-V4-Pro"}, {"id": "Gemini-3.5-Flash"}]}
        )
    )
    models = await adapter.list_models()
    assert models == ["DeepSeek-V4-Pro", "Gemini-3.5-Flash"]


async def test_openai_list_models_still_filters_when_some_match() -> None:
    """Existing behavior preserved: a real OpenAI-shaped response with a mix of
    chat and non-chat models still excludes the non-chat ones."""
    adapter = OpenAIAdapter(_config(ProviderKind.OPENAI, "gpt-4o"))
    adapter._client.get = AsyncMock(
        return_value=_fake_response_get(
            {"data": [{"id": "gpt-4o"}, {"id": "text-embedding-3-small"}, {"id": "whisper-1"}]}
        )
    )
    models = await adapter.list_models()
    assert models == ["gpt-4o"]


async def test_openai_list_models_uses_presets_when_response_genuinely_empty() -> None:
    adapter = OpenAIAdapter(_config(ProviderKind.OPENAI, "gpt-4o"))
    adapter._client.get = AsyncMock(return_value=_fake_response_get({"data": []}))
    models = await adapter.list_models()
    assert models == OpenAIAdapter.MODEL_PRESETS


async def test_gemini_list_models_returns_raw_names_when_capability_filter_zeroes_out() -> None:
    """A real, non-empty response missing supportedGenerationMethods (e.g. a 3rd-party
    gateway that doesn't populate it) must surface the real model names, not fake
    Gemini preset names."""
    adapter = GeminiAdapter(_config(ProviderKind.GEMINI, "gemini-1.5-pro"))
    adapter._client.get = AsyncMock(
        return_value=_fake_response_get(
            {"models": [{"name": "models/DeepSeek-V4-Pro"}, {"name": "models/GPT-5.4"}]}
        )
    )
    models = await adapter.list_models()
    assert models == ["DeepSeek-V4-Pro", "GPT-5.4"]


async def test_gemini_list_models_still_filters_when_some_match() -> None:
    adapter = GeminiAdapter(_config(ProviderKind.GEMINI, "gemini-1.5-pro"))
    adapter._client.get = AsyncMock(
        return_value=_fake_response_get(
            {
                "models": [
                    {"name": "models/gemini-1.5-pro", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]},
                ]
            }
        )
    )
    models = await adapter.list_models()
    assert models == ["gemini-1.5-pro"]


async def test_gemini_list_models_uses_presets_when_response_genuinely_empty() -> None:
    adapter = GeminiAdapter(_config(ProviderKind.GEMINI, "gemini-1.5-pro"))
    adapter._client.get = AsyncMock(return_value=_fake_response_get({"models": []}))
    models = await adapter.list_models()
    from domain.model_connector.gemini.adapter import MODEL_PRESETS
    assert models == MODEL_PRESETS

