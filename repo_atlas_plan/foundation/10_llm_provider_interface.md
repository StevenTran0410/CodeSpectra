# 10 — LLM Provider Interface

---

## Design goals

1. All provider-specific logic lives in adapter classes — the analysis pipeline never branches on provider kind.
2. Structured generation (JSON output) is a first-class operation.
3. Connection validation and model listing are uniform across all providers.
4. Error codes are normalized so the UI can give consistent, actionable messages.

---

## Provider kinds

| Kind | Type | Base URL | Auth |
|---|---|---|---|
| `ollama` | Local | `http://localhost:11434` | None |
| `lmstudio` | Local | `http://localhost:1234` | None |
| `openai` | Cloud (BYOK) | `https://api.openai.com` | Bearer API key |
| `anthropic` | Cloud (BYOK) | `https://api.anthropic.com` | `x-api-key` header |
| `gemini` | Cloud (BYOK) | `https://generativelanguage.googleapis.com` | Query param `key=` |
| `deepseek` | Cloud (BYOK) | `https://api.deepseek.com` | Bearer API key |

---

## Adapter protocol

Every adapter implements:

```python
class ProviderAdapter(Protocol):
    async def list_models(self) -> list[str]: ...
    async def chat(self, request: ChatRequest) -> ChatResponse: ...
    async def test_connection(self) -> tuple[bool, str, str | None]:
        """Returns (ok, message, warning_or_None)."""
        ...
```

### `ChatRequest`

```python
class ChatRequest(BaseModel):
    messages: list[ChatMessage]   # role: "system" | "user" | "assistant"
    max_tokens: int = 4096
    temperature: float = 0.2
```

### `ChatResponse`

```python
class ChatResponse(BaseModel):
    provider_id: str
    model_id: str
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
```

---

## Error codes

```python
class ProviderErrorCode(str, Enum):
    CONNECTION_REFUSED = "connection_refused"
    TIMEOUT            = "timeout"
    AUTH_FAILED        = "auth_failed"
    MODEL_NOT_FOUND    = "model_not_found"
    RATE_LIMITED       = "rate_limited"
    UNKNOWN            = "unknown"
```

### HTTP status mapping (FastAPI exception handler)

| `ProviderErrorCode` | HTTP status |
|---|---|
| `connection_refused` | 503 Service Unavailable |
| `timeout` | 503 Service Unavailable |
| `auth_failed` | 401 Unauthorized |
| `model_not_found` | 404 Not Found |
| `rate_limited` | 429 Too Many Requests |
| `unknown` | 500 Internal Server Error |

Response body:
```json
{
  "error": "connection_refused",
  "message": "Cannot reach Ollama at http://localhost:11434. Make sure Ollama is running.",
  "retryable": true
}
```

---

## `test_connection` return contract

The third element of the return tuple is an **optional warning** — used when the connection succeeded but the setup is incomplete:

| Condition | `ok` | `warning` |
|---|---|---|
| Connected, models available | `true` | `None` |
| Connected, no models pulled (Ollama) | `true` | `"No models pulled yet. Run: ollama pull <model>"` |
| Connected, no model loaded (LM Studio) | `true` | `"No model is loaded in LM Studio. Open LM Studio and load a model."` |
| Not connected | `false` | `None` |
| Auth failed | `false` | `None` |

The frontend maps: `ok=true, warning=null` → green; `ok=true, warning≠null` → amber; `ok=false` → red.

---

## Capability matrix

| Capability | Ollama | LM Studio | OpenAI | Anthropic | Gemini | DeepSeek |
|---|---|---|---|---|---|---|
| `streaming` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `embeddings` | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| `supports_system_prompt` | ✓ | ✓ | ✓ | ✓ (separate field) | ✓ (systemInstruction) | ✓ |
| `max_context_tokens` | model-dependent | model-dependent | model-dependent | 200k | 1M | 64k |
