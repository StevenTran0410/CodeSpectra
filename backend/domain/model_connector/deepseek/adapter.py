"""DeepSeek provider adapter — OpenAI-compatible API with DeepSeek base URL."""

from domain.model_connector._cloud_base import CloudAdapterBase
from domain.model_connector.errors import ProviderError, ProviderErrorCode
from domain.model_connector.openai.adapter import OpenAIAdapter
from domain.model_connector.types import ChatRequest, EmbedRequest, EmbedResponse, ProviderConfig

DEEPSEEK_MODEL_PRESETS = [
    "deepseek-chat",
    "deepseek-reasoner",
]


class DeepSeekAdapter(OpenAIAdapter):
    CHAT_MODEL_PREFIXES = None
    MODEL_PRESETS = DEEPSEEK_MODEL_PRESETS
    # deepseek-reasoner ignores temperature/top_p silently (no error) rather than
    # rejecting it outright, but omitting it is still correct — and unlike OpenAI's
    # o-series it has no reasoning_effort equivalent, so _build_payload's
    # reasoning_effort injection (gated on the same flag) is a no-op here too.
    _NO_TEMPERATURE_PREFIXES = ("deepseek-reasoner",)

    def __init__(self, config: ProviderConfig) -> None:
        CloudAdapterBase.__init__(self, config, base_url="https://api.deepseek.com")

    def _build_payload(self, request: ChatRequest) -> dict:
        payload = super()._build_payload(request)
        payload.pop("reasoning_effort", None)
        return payload

    async def embed(self, request: EmbedRequest) -> EmbedResponse:  # noqa: ARG002
        raise ProviderError(
            ProviderErrorCode.UNKNOWN,
            "DeepSeek has no embeddings API — use OpenAI or Gemini for embeddings",
            provider_id=self.config.id,
        )

