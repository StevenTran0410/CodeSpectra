"""Shared archetype field-vocabulary — single canonical source for kwarg-name mappings, used by synthetic_agent_io.py (dataset generation) and contract_harvest.py (case binding derivation)."""

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

# Prefix for case_binding entries that reference virtual inputs (harvested from constructor deps)
VIRTUAL_BINDING_PREFIX = "virtual:"

# Stand-in for a required identifier pointing at real state a synthetic case cannot have; visibly fake on purpose so a value showing up in output means it was used, not resolved.
SYNTHETIC_ID_PLACEHOLDER = "aeh-synthetic"
