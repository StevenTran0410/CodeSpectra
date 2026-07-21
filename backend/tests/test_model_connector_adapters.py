"""Tests for model connector adapter configuration."""

import pytest
from domain.model_connector._cloud_base import CloudAdapterBase
from domain.model_connector.types import ProviderConfig, ProviderKind


@pytest.mark.asyncio
async def test_cloud_adapter_base_url_precedence_config_wins():
    """Test that config.base_url takes precedence over caller-provided default."""
    config = ProviderConfig(
        id="test-config",
        kind=ProviderKind.OPENAI,
        display_name="Test OpenAI",
        model_id="gpt-4",
        base_url="https://custom.example",
        extra={},
    )
    caller_default = "https://api.openai.com"

    adapter = CloudAdapterBase(config, base_url=caller_default)

    assert str(adapter._client.base_url).rstrip("/") == "https://custom.example"


@pytest.mark.asyncio
async def test_cloud_adapter_base_url_falls_back_to_caller_default():
    """Test that caller default is used when config.base_url is empty."""
    config = ProviderConfig(
        id="test-config",
        kind=ProviderKind.OPENAI,
        display_name="Test OpenAI",
        model_id="gpt-4",
        base_url="",
        extra={},
    )
    caller_default = "https://api.openai.com"

    adapter = CloudAdapterBase(config, base_url=caller_default)

    assert str(adapter._client.base_url).rstrip("/") == "https://api.openai.com"


@pytest.mark.asyncio
async def test_cloud_adapter_base_url_none_uses_caller_default():
    """Test that None config.base_url falls back to caller default."""
    config = ProviderConfig(
        id="test-config",
        kind=ProviderKind.OPENAI,
        display_name="Test OpenAI",
        model_id="gpt-4",
        base_url="",
        extra={},
    )
    caller_default = "https://api.openai.com"

    adapter = CloudAdapterBase(config, base_url=caller_default)

    assert str(adapter._client.base_url).rstrip("/") == "https://api.openai.com"
