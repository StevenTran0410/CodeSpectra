"""System-type vocabulary — generic pattern names only, no framework/target literal.
Primitives layer (like roles.VALID_ROLES): importable from discovery/* and mapping/*
without a cycle. Phase 2 keys per-type dispatch tables off these same frozensets."""
from __future__ import annotations

AGENTIC_SYSTEM_TYPES = frozenset({
    "pipeline", "routing", "parallelization", "orchestrator",
    "peer-collaboration", "evaluator-optimizer", "debate", "blackboard",
})
SINGLE_AGENT_TYPES = frozenset({"single-flow", "tool-loop", "plan-execute", "reflection"})
ALL_SYSTEM_TYPES = AGENTIC_SYSTEM_TYPES | SINGLE_AGENT_TYPES
CAPABILITY_TAGS = frozenset({"has_retrieval", "has_tools", "has_memory"})

# Decision source for system_type classification (stored in system_type_signals[decided_by])
DECIDED_BY_STRUCTURAL = "structural"
DECIDED_BY_LLM = "llm"
DECIDED_BY_UNRESOLVED = "unresolved"
