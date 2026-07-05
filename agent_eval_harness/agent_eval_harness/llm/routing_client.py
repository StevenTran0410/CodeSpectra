"""Routing LLM Client — routes LLM completions based on active component context."""
from __future__ import annotations

import contextvars

from agent_eval_harness.llm.client import LLMClient, LLMMessage, LLMResponse

current_component_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_component_id_var", default=None
)


class RoutingLLMClient:
    """LLMClient implementation that routes calls dynamically to component-specific overrides.

    If no component-specific override matches, it falls back to the default client.
    """

    def __init__(self, default: LLMClient, overrides: dict[str, LLMClient]) -> None:
        self._default = default
        self._overrides = overrides

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float | None = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        # 1. Check ContextVar first (Tier-2 support)
        comp_id = current_component_id_var.get()
        if comp_id and comp_id in self._overrides:
            return await self._overrides[comp_id].complete(
                messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode
            )

        # 2. Check Haystack span (Tier-1 support)
        try:
            from haystack.tracing import tracer as haystack_global_tracer

            span = haystack_global_tracer.current_span()
            if span:
                # Haystack's own automatic component spans tag "haystack.component.name";
                # AEH's manual inner spans (test_targets._shared.tracing_helpers.manual_span,
                # the convention every real component uses to wrap its own LLM call) tag the
                # bare "component_name" instead — the same fallback tier1_haystack.py's own
                # _normalize() already uses. Checking only the first key means every real
                # target's LLM call falls through to the default client, silently ignoring
                # the override (confirmed empirically: writer span kept the default model
                # until this fallback was added).
                comp_name = span.tags.get("haystack.component.name") or span.tags.get(
                    "component_name"
                )
                if comp_name and comp_name in self._overrides:
                    return await self._overrides[comp_name].complete(
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        json_mode=json_mode,
                    )
        except Exception:
            pass

        # 3. Fallback to default client
        return await self._default.complete(
            messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode
        )

    async def aclose(self) -> None:
        if hasattr(self._default, "aclose"):
            await self._default.aclose()
        for client in self._overrides.values():
            if hasattr(client, "aclose"):
                await client.aclose()
