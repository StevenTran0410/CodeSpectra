"""Reasoning/thinking-model classification shared by every provider adapter.

Detection is prefix/substring matching against known model-id conventions —
there is no standardized capability-discovery API across providers for this.
Defaults to NONE when uncertain, which is the safe failure mode (temperature
still applies; no risk of sending a reasoning param an incompatible model
rejects). Written against the 2026 model lineups; new model families need a
one-line addition here when they ship.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from .types import ProviderKind


class ReasoningStyle(str, Enum):
    NONE = "none"                        # regular model — temperature applies as-is
    EFFORT = "effort"                    # OpenAI o-series/gpt-5 — reasoning_effort enum, no temperature
    BUDGET_TOKENS = "budget_tokens"       # Anthropic extended thinking — budget_tokens, no temperature
    THINKING_BUDGET = "thinking_budget"   # Gemini 2.5+ — thinkingConfig.thinkingBudget, temperature OK
    TOGGLE = "toggle"                     # DeepSeek-reasoner / local reasoning models — no tunable budget


class ModelInfo(BaseModel):
    id: str
    reasoning_style: ReasoningStyle


_OPENAI_EFFORT_PREFIXES = ("o1", "o3", "o4", "gpt-5")

# Extended-thinking-capable Claude tiers: 3.7+/4.x/5.x sonnet & opus. Haiku and
# pre-3.7 sonnet/opus (3.5 and earlier) do not support extended thinking.
_ANTHROPIC_THINKING_MARKERS = ("3-7-sonnet", "sonnet-4", "opus-4", "sonnet-5", "opus-5")

_GEMINI_THINKING_PREFIXES = ("gemini-2.5", "gemini-3")

_LOCAL_REASONING_MARKERS = ("deepseek-r1", "qwq", "reasoner", "thinking")


def classify(kind: ProviderKind, model_id: str) -> ReasoningStyle:
    mid = (model_id or "").lower()

    if kind == ProviderKind.OPENAI:
        if any(mid.startswith(p) for p in _OPENAI_EFFORT_PREFIXES):
            return ReasoningStyle.EFFORT
        return ReasoningStyle.NONE

    if kind == ProviderKind.ANTHROPIC:
        if "haiku" in mid:
            return ReasoningStyle.NONE
        if any(marker in mid for marker in _ANTHROPIC_THINKING_MARKERS):
            return ReasoningStyle.BUDGET_TOKENS
        return ReasoningStyle.NONE

    if kind == ProviderKind.GEMINI:
        if any(mid.startswith(p) for p in _GEMINI_THINKING_PREFIXES):
            return ReasoningStyle.THINKING_BUDGET
        return ReasoningStyle.NONE

    if kind == ProviderKind.DEEPSEEK:
        if "reasoner" in mid:
            return ReasoningStyle.TOGGLE
        return ReasoningStyle.NONE

    if kind in (ProviderKind.OLLAMA, ProviderKind.LM_STUDIO):
        if any(marker in mid for marker in _LOCAL_REASONING_MARKERS):
            return ReasoningStyle.TOGGLE
        return ReasoningStyle.NONE

    return ReasoningStyle.NONE


def thinking_budget_range(kind: ProviderKind, model_id: str) -> tuple[int, int, bool]:
    """(min, max, can_disable) for THINKING_BUDGET/BUDGET_TOKENS styles — UI range hints."""
    mid = (model_id or "").lower()

    if kind == ProviderKind.GEMINI:
        if "flash" in mid:
            return (0, 24576, True)
        return (128, 32768, False)  # Pro tier cannot disable thinking

    if kind == ProviderKind.ANTHROPIC:
        return (1024, 32000, True)  # omit the thinking block entirely to disable

    return (0, 0, True)
