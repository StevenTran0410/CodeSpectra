"""Build wiring.json: deterministic mapping of component ids and per-agent invocation data, derived entirely from SystemMap and EvaluationContracts — no guessing, no target-specific code."""
from __future__ import annotations

from typing import Any

from agent_eval_harness.mapping.system_map import SystemMap


def build_wiring(system_map: SystemMap, plan_id: str, dataset_kinds: list[str] | None = None, agent_invocations: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build deterministic wiring dict from SystemMap and contract data."""
    wiring: dict[str, Any] = {
        "plan_id": plan_id,
        "component_ids": sorted(c.id for c in system_map.components),
        # Files the coding agent implements rather than copies; a reinstall must carry these forward from backup.
        "agent_owned_files": ["target_init.py", "retrieval_stub.py", "run_config.json"],
    }

    if dataset_kinds:
        wiring["dataset_kinds"] = dataset_kinds

    if agent_invocations:
        wiring["agent_invocation"] = agent_invocations

    return wiring


