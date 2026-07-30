"""ProviderConfigService.list_models enriches each raw model id with its ReasoningStyle,
computed once server-side so the frontend model picker never re-implements the heuristic."""
from __future__ import annotations

from unittest.mock import AsyncMock

import domain.model_connector.service as service_module
from domain.model_connector.reasoning import ReasoningStyle
from domain.model_connector.service import ProviderConfigService
from domain.model_connector.types import ProviderConfig, ProviderKind


def _config(kind: ProviderKind, model_id: str) -> ProviderConfig:
    return ProviderConfig(
        id="p1", kind=kind, display_name="test", base_url="http://test", model_id=model_id,
    )


async def test_list_models_enriches_with_reasoning_style(monkeypatch) -> None:
    fake_adapter = AsyncMock()
    fake_adapter.list_models = AsyncMock(return_value=["gpt-4o", "o3-mini"])
    fake_adapter.aclose = AsyncMock()

    svc = ProviderConfigService()
    monkeypatch.setattr(
        svc, "_get_by_id_full", AsyncMock(return_value=_config(ProviderKind.OPENAI, "gpt-4o"))
    )
    monkeypatch.setattr(service_module, "_get_adapter", lambda config: fake_adapter)

    result = await svc.list_models("p1")

    assert [m.id for m in result] == ["gpt-4o", "o3-mini"]
    styles = {m.id: m.reasoning_style for m in result}
    assert styles["gpt-4o"] == ReasoningStyle.NONE
    assert styles["o3-mini"] == ReasoningStyle.EFFORT
    fake_adapter.aclose.assert_awaited_once()


async def test_list_models_fallback_on_provider_error(monkeypatch) -> None:
    from domain.model_connector.errors import ProviderError, ProviderErrorCode
    
    fake_adapter = AsyncMock()
    fake_adapter.list_models = AsyncMock(side_effect=ProviderError(ProviderErrorCode.UNKNOWN, "connect fail"))
    fake_adapter.aclose = AsyncMock()

    config = _config(ProviderKind.OPENAI, "gpt-4o")
    config.extra = {"manual_models": ["gpt-5", "gpt-5-mini"]}

    svc = ProviderConfigService()
    monkeypatch.setattr(svc, "_get_by_id_full", AsyncMock(return_value=config))
    monkeypatch.setattr(service_module, "_get_adapter", lambda config: fake_adapter)

    result = await svc.list_models("p1")
    assert [m.id for m in result] == ["gpt-5", "gpt-5-mini"]
    fake_adapter.aclose.assert_awaited_once()


async def test_list_models_raises_when_no_manual_models(monkeypatch) -> None:
    from domain.model_connector.errors import ProviderError, ProviderErrorCode
    import pytest

    fake_adapter = AsyncMock()
    fake_adapter.list_models = AsyncMock(side_effect=ProviderError(ProviderErrorCode.UNKNOWN, "connect fail"))
    fake_adapter.aclose = AsyncMock()

    config = _config(ProviderKind.OPENAI, "gpt-4o")
    config.extra = {}  # no manual models

    svc = ProviderConfigService()
    monkeypatch.setattr(svc, "_get_by_id_full", AsyncMock(return_value=config))
    monkeypatch.setattr(service_module, "_get_adapter", lambda config: fake_adapter)

    with pytest.raises(ProviderError) as exc_info:
        await svc.list_models("p1")
    assert "connect fail" in str(exc_info.value)
    fake_adapter.aclose.assert_awaited_once()


async def test_list_models_ignores_fallback_on_success(monkeypatch) -> None:
    fake_adapter = AsyncMock()
    fake_adapter.list_models = AsyncMock(return_value=["live-model-1"])
    fake_adapter.aclose = AsyncMock()

    config = _config(ProviderKind.OPENAI, "gpt-4o")
    config.extra = {"manual_models": ["manual-model-1"]}

    svc = ProviderConfigService()
    monkeypatch.setattr(svc, "_get_by_id_full", AsyncMock(return_value=config))
    monkeypatch.setattr(service_module, "_get_adapter", lambda config: fake_adapter)

    result = await svc.list_models("p1")
    assert [m.id for m in result] == ["live-model-1"]
    fake_adapter.aclose.assert_awaited_once()


# ── chat() parameter-rejection retries ──────────────────────────────────────

async def test_chat_retries_without_reasoning_effort_on_rejection(monkeypatch) -> None:
    """A provider/gateway that 400s on reasoning_effort must trigger one retry with it
    dropped, not fail the whole call (the gpt-5 prefix heuristic can't know every proxy's
    real capabilities)."""
    from domain.model_connector.errors import ProviderError, ProviderErrorCode
    from domain.model_connector.types import ChatMessage, ChatRequest, ChatResponse

    calls: list[str | None] = []

    async def fake_chat(request: ChatRequest) -> ChatResponse:
        calls.append(request.reasoning_effort)
        if request.reasoning_effort is not None:
            raise ProviderError(
                ProviderErrorCode.UNKNOWN,
                "HTTP 400: Unrecognized request argument supplied: reasoning_effort",
            )
        return ChatResponse(provider_id="p1", model_id="gpt-5.4-mini", content="ok")

    fake_adapter = AsyncMock()
    fake_adapter.chat = fake_chat
    fake_adapter.aclose = AsyncMock()

    svc = ProviderConfigService()
    monkeypatch.setattr(
        svc, "_get_by_id_full", AsyncMock(return_value=_config(ProviderKind.OPENAI, "gpt-5.4-mini"))
    )
    monkeypatch.setattr(service_module, "_get_adapter", lambda config: fake_adapter)

    resp = await svc.chat(
        ChatRequest(
            provider_id="p1",
            messages=[ChatMessage(role="user", content="hi")],
            reasoning_effort="low",
        )
    )
    assert resp.content == "ok"
    assert calls == ["low", None]  # first attempt with it, retry without


async def test_chat_does_not_retry_reasoning_effort_on_unrelated_error(monkeypatch) -> None:
    from domain.model_connector.errors import ProviderError, ProviderErrorCode
    from domain.model_connector.types import ChatMessage, ChatRequest
    import pytest

    fake_adapter = AsyncMock()
    fake_adapter.chat = AsyncMock(
        side_effect=ProviderError(ProviderErrorCode.RATE_LIMITED, "HTTP 429: rate limited")
    )
    fake_adapter.aclose = AsyncMock()

    svc = ProviderConfigService()
    monkeypatch.setattr(
        svc, "_get_by_id_full", AsyncMock(return_value=_config(ProviderKind.OPENAI, "gpt-5.4-mini"))
    )
    monkeypatch.setattr(service_module, "_get_adapter", lambda config: fake_adapter)

    with pytest.raises(ProviderError):
        await svc.chat(
            ChatRequest(
                provider_id="p1",
                messages=[ChatMessage(role="user", content="hi")],
                reasoning_effort="low",
            )
        )
    assert fake_adapter.chat.await_count == 1  # no retry on an unrelated error

