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
    model_id: str | None = None
    messages: list[ChatMessage]
    max_completion_tokens: int = 2048
    # None = omit temperature from payload entirely (use provider/model default).
    # Required for models that reject any explicit temperature (o1, o3, gpt-5, etc.)
    temperature: float | None = 0.2
    # When True each adapter enables its native JSON-output mode:
    # OpenAI → response_format=json_object, Gemini → responseMimeType=application/json,
    # Ollama → format=json, LM Studio → response_format=json_object,
    # Anthropic → prefill assistant turn with "{"
    json_mode: bool = False
    stream: bool = False


class ChatResponse(BaseModel):
    provider_id: str
    model_id: str
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
