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
