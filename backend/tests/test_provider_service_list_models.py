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
