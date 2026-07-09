"""classify()/thinking_budget_range() — the shared reasoning-model detection table
every provider adapter and the model-list endpoint rely on."""
from __future__ import annotations

import pytest

from domain.model_connector.reasoning import ReasoningStyle, classify, thinking_budget_range
from domain.model_connector.types import ProviderKind


@pytest.mark.parametrize(
    "kind,model_id,expected",
    [
        (ProviderKind.OPENAI, "o3-mini", ReasoningStyle.EFFORT),
        (ProviderKind.OPENAI, "o1", ReasoningStyle.EFFORT),
        (ProviderKind.OPENAI, "gpt-5", ReasoningStyle.EFFORT),
        (ProviderKind.OPENAI, "gpt-4o", ReasoningStyle.NONE),
        (ProviderKind.OPENAI, "gpt-3.5-turbo", ReasoningStyle.NONE),
        (ProviderKind.ANTHROPIC, "claude-opus-4-5", ReasoningStyle.BUDGET_TOKENS),
        (ProviderKind.ANTHROPIC, "claude-sonnet-4-5", ReasoningStyle.BUDGET_TOKENS),
        (ProviderKind.ANTHROPIC, "claude-3-7-sonnet-20250219", ReasoningStyle.BUDGET_TOKENS),
        (ProviderKind.ANTHROPIC, "claude-3-5-sonnet-20241022", ReasoningStyle.NONE),
        (ProviderKind.ANTHROPIC, "claude-3-5-haiku-20241022", ReasoningStyle.NONE),
        (ProviderKind.ANTHROPIC, "claude-3-haiku-20240307", ReasoningStyle.NONE),
        (ProviderKind.GEMINI, "gemini-2.5-pro", ReasoningStyle.THINKING_BUDGET),
        (ProviderKind.GEMINI, "gemini-2.5-flash", ReasoningStyle.THINKING_BUDGET),
        (ProviderKind.GEMINI, "gemini-2.0-flash", ReasoningStyle.NONE),
        (ProviderKind.GEMINI, "gemini-1.5-pro", ReasoningStyle.NONE),
        (ProviderKind.DEEPSEEK, "deepseek-reasoner", ReasoningStyle.TOGGLE),
        (ProviderKind.DEEPSEEK, "deepseek-chat", ReasoningStyle.NONE),
        (ProviderKind.OLLAMA, "deepseek-r1:8b", ReasoningStyle.TOGGLE),
        (ProviderKind.OLLAMA, "qwq:32b", ReasoningStyle.TOGGLE),
        (ProviderKind.OLLAMA, "llama3:8b", ReasoningStyle.NONE),
        (ProviderKind.LM_STUDIO, "deepseek-r1-distill-qwen-7b", ReasoningStyle.TOGGLE),
        (ProviderKind.LM_STUDIO, "llama-3-8b-instruct", ReasoningStyle.NONE),
    ],
)
def test_classify(kind: ProviderKind, model_id: str, expected: ReasoningStyle) -> None:
    assert classify(kind, model_id) == expected


def test_classify_unknown_model_defaults_to_none() -> None:
    assert classify(ProviderKind.OPENAI, "") == ReasoningStyle.NONE
    assert classify(ProviderKind.ANTHROPIC, "some-future-model") == ReasoningStyle.NONE


def test_thinking_budget_range_gemini_flash_can_disable() -> None:
    lo, hi, can_disable = thinking_budget_range(ProviderKind.GEMINI, "gemini-2.5-flash")
    assert (lo, hi) == (0, 24576)
    assert can_disable is True


def test_thinking_budget_range_gemini_pro_cannot_disable() -> None:
    lo, hi, can_disable = thinking_budget_range(ProviderKind.GEMINI, "gemini-2.5-pro")
    assert (lo, hi) == (128, 32768)
    assert can_disable is False


def test_thinking_budget_range_anthropic() -> None:
    lo, hi, can_disable = thinking_budget_range(ProviderKind.ANTHROPIC, "claude-opus-4-5")
    assert lo == 1024
    assert hi >= lo
    assert can_disable is True
