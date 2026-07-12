"""Deterministic (never LLM-guessed) wiring.json builder for CodeSpectra's own backend — the
v1 target. Component ids come straight from Stage 1's already-harvested SystemMap, the same
source Stage 3's gates already trust — no separate discovery step, no guessing. A generic
"any target" version of this function is Stage 4's later wiring-locator work (an LLM step
over the same SystemMap/AgentFlowMap shape already produced for arbitrary repos) — kept
isolated here so that later work is a drop-in swap for this function, not a rewrite of the
injector that calls it."""
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
