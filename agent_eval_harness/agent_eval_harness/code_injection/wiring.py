"""Deterministic (never LLM-guessed) wiring.json builder for CodeSpectra's own backend, the v1 target — component ids come straight from Stage 1's already-harvested SystemMap. Isolated here so a future generic "any target" wiring-locator can drop in as a swap for this function."""
from __future__ import annotations

from typing import Any

from agent_eval_harness.mapping.system_map import SystemMap


def build_wiring_for_codespectra(system_map: SystemMap, plan_id: str) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "entrypoint": {
            "module": "domain.analysis.orchestrator",
            "class": "RunDirectorAgent",
            "method": "run",
        },
        "component_ids": sorted(c.id for c in system_map.components),
    }
