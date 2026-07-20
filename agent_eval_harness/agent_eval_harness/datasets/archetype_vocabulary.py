"""Shared archetype field-vocabulary — single canonical source for kwarg-name mappings, used by synthetic_agent_io.py (dataset generation) and contract_harvest.py (case binding derivation)."""

ARCHETYPE_KWARG_SETS: dict[str, frozenset[str]] = {
    "rag_single_shot:glossary": frozenset({"provider_id", "model_id", "snapshot_id", "profile"}),
    "rag_single_shot:important_files": frozenset({"provider_id", "model_id", "snapshot_id", "graph_summary", "profile"}),
    "rag_mem_ctx:project_identity": frozenset({"provider_id", "model_id", "snapshot_id", "repo_name", "mem_ctx", "profile"}),
    "rag_mem_ctx_participant:architecture": frozenset({"provider_id", "model_id", "snapshot_id", "graph_summary", "arch_bundle", "identity_output", "profile", "folder_tree"}),
    "rag_mem_ctx_participant:structure": frozenset({"provider_id", "model_id", "snapshot_id", "arch_bundle", "folder_tree", "identity_output", "profile"}),
    "rag_query_planning:conventions": frozenset({"provider_id", "model_id", "snapshot_id", "static_convention", "structure_output", "profile"}),
    "rag_query_planning:risk": frozenset({"provider_id", "model_id", "snapshot_id", "static_risk", "profile"}),
    "rag_upstream:violations": frozenset({"provider_id", "model_id", "snapshot_id", "static_convention", "static_risk", "conventions_output", "profile"}),
    "rag_upstream:onboarding": frozenset({"provider_id", "model_id", "snapshot_id", "important_files_output", "profile"}),
    "rag_query_planning_mem_ctx:feature_map": frozenset({"provider_id", "model_id", "snapshot_id", "graph_summary", "identity_output", "architecture_output", "profile", "folder_tree"}),
}

ARCHETYPE_KWARG_SET_VALUES: frozenset[frozenset[str]] = frozenset(ARCHETYPE_KWARG_SETS.values())

KWARG_CASE_KEY_MAPPING: dict[str, str] = {
    "arch_bundle": "bundle",
    "identity_output": "project_identity_output",
    "folder_tree": "folder_tree",
    "graph_summary": "graph_summary",
    "repo_name": "repo_name",
    "doc_ctx": "doc_ctx",
    "manifest_ctx": "manifest_ctx",
    "static_convention": "static_convention",
    "static_risk": "static_risk",
    "conventions_output": "conventions_output",
    "important_files_output": "important_files_output",
    "structure_output": "structure_output",
    "architecture_output": "architecture_output",
    "mem_ctx": "mem_ctx",
}

# The case field holding a case's retrieval evidence — agents that retrieve internally never receive it as a kwarg, so it reaches them through the retrieval stub instead.
EVIDENCE_CASE_KEY = "bundle"

# Stand-in for a required identifier pointing at real state a synthetic case cannot have; visibly fake on purpose so a value showing up in output means it was used, not resolved.
SYNTHETIC_ID_PLACEHOLDER = "aeh-synthetic"
