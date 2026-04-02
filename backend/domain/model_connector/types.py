"""Core types for the LLM model connector layer."""
from enum import Enum
from typing import Any

from pydantic import BaseModel


class ProviderKind(str, Enum):
    # Local providers
    OLLAMA = "ollama"
    LM_STUDIO = "lmstudio"
    # Cloud providers (BYOK)
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"


CLOUD_KINDS: frozenset[ProviderKind] = frozenset({
    ProviderKind.OPENAI,
    ProviderKind.ANTHROPIC,
    ProviderKind.GEMINI,
    ProviderKind.DEEPSEEK,
})


class ProviderCapabilities(BaseModel):
    streaming: bool = False
    embeddings: bool = False
    max_context_tokens: int = 4096
    supports_system_prompt: bool = True


class ProviderConfig(BaseModel):
    id: str
    kind: ProviderKind
    display_name: str
    base_url: str
    model_id: str
    capabilities: ProviderCapabilities = ProviderCapabilities()
    extra: dict[str, Any] = {}


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    provider_id: str
    messages: list[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.2
    stream: bool = False


class ChatResponse(BaseModel):
    provider_id: str
    model_id: str
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
