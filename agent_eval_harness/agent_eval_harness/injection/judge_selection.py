"""The judge scoring an agent's output must not share a provider kind with whatever generated the synthetic gold — same-family pairs measurably inflate scores."""
from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger("agent_eval_harness.injection.judge_selection")


class _HasKind(Protocol):
    provider_id: str
    kind: str


def pick_judge_provider(generator: _HasKind, available: list[_HasKind]) -> _HasKind | None:
    """Falls back to the generator's own provider (with a warning) rather than hard-failing when no other kind is configured."""
    if not available:
        return None
    for candidate in available:
        if candidate.kind and candidate.kind != generator.kind:
            return candidate
    logger.warning(
        "pick_judge_provider: no configured provider has a kind different from the "
        "generator's (%s) — falling back to the generator's own provider; scores are more "
        "exposed to generator/judge circularity than with a cross-family judge (CS-289 §5.2).",
        generator.kind,
    )
    for candidate in available:
        if candidate.provider_id == generator.provider_id:
            return candidate
    return available[0]
