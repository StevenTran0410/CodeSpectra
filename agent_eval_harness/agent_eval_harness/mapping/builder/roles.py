"""Role vocabulary + structural subtraction — the sole owner of VALID_ROLES.

Role classification itself moved to Stage 2.5 enrichment: Stage 2 no longer calls an LLM here.
This module keeps the taxonomy and the two deterministic subtractions that Stage 2.5's hard gate
applies against the LLM's answer — primitives only, no system_map/types import, so it can be
called from either stage without a cycle.
"""
from __future__ import annotations

ROLE_CONFIDENCE_THRESHOLD: float = 0.7
VALID_ROLES = frozenset({
    "input_guard.rule",
    "input_guard.llm",
    "orchestrator",
    "retrieval_agent",
    "tool",
    "validator",
    "writer",
    "worker",
    "unknown",
})


def admissible_roles(is_tool: bool | None, constructor_fanout: int | None) -> frozenset[str]:
    """Structure SUBTRACTS impossible roles from VALID_ROLES; it never assigns one.
    None means "unknown" (e.g. a hand-written or legacy map) — no subtraction fires."""
    admissible = set(VALID_ROLES)
    if is_tool is False:
        admissible.discard("tool")
    if constructor_fanout is not None and constructor_fanout < 2:
        admissible.discard("orchestrator")
    return frozenset(admissible)


def forced_role(is_conditional_source: bool) -> str | None:
    """The ONE path that ASSIGNS a role structurally instead of subtracting from the LLM's
    verdict. Always wins, applied before the admissible/confidence gate. admissible_roles
    can only remove (and its fanout<2 rule would otherwise strip orchestrator right back)."""
    return "orchestrator" if is_conditional_source else None


def structural_facts(fan_in: int, fan_out: int) -> str:
    """Shown as evidence, not as a rule — degree alone cannot separate the semantic axis."""
    return f"fan-in: {fan_in}, fan-out: {fan_out}"
