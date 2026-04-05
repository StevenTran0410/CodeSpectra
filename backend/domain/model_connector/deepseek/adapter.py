"""DeepSeek provider adapter — OpenAI-compatible API with DeepSeek base URL."""

from domain.model_connector._cloud_base import CloudAdapterBase
from domain.model_connector.openai.adapter import OpenAIAdapter
from domain.model_connector.types import ProviderConfig

DEEPSEEK_MODEL_PRESETS = [
    "deepseek-chat",
    "deepseek-reasoner",
]


class DeepSeekAdapter(OpenAIAdapter):
    CHAT_MODEL_PREFIXES = None
    MODEL_PRESETS = DEEPSEEK_MODEL_PRESETS

    def __init__(self, config: ProviderConfig) -> None:
        CloudAdapterBase.__init__(self, config, base_url="https://api.deepseek.com")
