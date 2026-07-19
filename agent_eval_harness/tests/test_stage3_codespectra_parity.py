"""CS-303 Slice 0 — CodeSpectra parity fixture (no production behavior asserted to change).

Snapshots two deterministic facts per CodeSpectra kwarg shape in
`_KNOWN_SHAPE_KWARG_SETS`: (1) the `_archetype_for` classification, and (2) whether the
generic path's field-descriptor key-set (raw kwarg names minus config-bound ones) equals
what the corresponding archetype builder actually renders. That equality is the go/no-go
signal for Slice 3c's removal of `_KNOWN_SHAPE_KWARG_SETS` — see the PAUSE note next to that
dict in synthetic_agent_io.py. This file must stay green through every later slice; a flip in
the parity assertion means the Slice 3c decision needs revisiting.
"""
from __future__ import annotations

import pytest

from agent_eval_harness.datasets.fulfillment import _archetype_for
from agent_eval_harness.datasets.generator_utils import config_kwarg_names_from_case_binding
from agent_eval_harness.datasets.generators.synthetic_agent_io import _KNOWN_SHAPE_KWARG_SETS
from agent_eval_harness.planning.contract import EvaluationContract, InvocationContract, KwargSpec

_CONFIG_KWARGS = frozenset({"provider_id", "model_id"})

# Upstream agent-output kwargs per shape — mirrors what harvest_contracts' second pass (and,
# before its removal, the old _LEGACY_OVERRIDE) supplies as upstream_context_specs. Needed so
# the builder-side field-set below is computed the same way production does it.
_UPSTREAM_KWARGS_BY_SHAPE: dict[str, set[str]] = {
    "rag_upstream:violations": {"conventions_output"},
    "rag_upstream:onboarding": {"important_files_output"},
    "rag_mem_ctx_participant:architecture": {"identity_output"},
    "rag_mem_ctx_participant:structure": {"identity_output"},
    "rag_query_planning:conventions": {"structure_output"},
    "rag_query_planning_mem_ctx:feature_map": {"identity_output", "architecture_output"},
}


def _builder_field_keys(shape_key: str, archetype: str) -> frozenset[str]:
    """What the archetype builder actually renders as its case-input field set — hand-derived
    from _rag_writer_prompt's fixed shape (a "bundle" evidence block always; string_field_specs
    and upstream_specs vary per archetype wrapper). The builders are untouched by this ticket,
    so this mapping is a stable snapshot, not a moving target."""
    upstream = _UPSTREAM_KWARGS_BY_SHAPE.get(shape_key, set())
    if archetype == "rag_single_shot":
        return frozenset({"bundle"})
    if archetype == "rag_upstream":
        return frozenset({"bundle"}) | upstream
    if archetype == "rag_mem_ctx":
        return frozenset({"bundle", "folder_tree", "doc_ctx", "manifest_ctx", "repo_name"})
    if archetype == "rag_mem_ctx_participant":
        return frozenset({"bundle", "folder_tree"}) | upstream
    if archetype == "rag_query_planning":
        return frozenset({"bundle"}) | upstream
    if archetype == "rag_query_planning_mem_ctx":
        return frozenset({"bundle", "folder_tree"}) | upstream
    raise AssertionError(f"no builder-field mapping for archetype {archetype!r}")


def _contract_for_shape(agent_id: str, archetype: str, kwarg_names: frozenset[str]) -> EvaluationContract:
    kwargs = [KwargSpec(name=n, annotation="str") for n in sorted(kwarg_names)]
    case_binding = {
        n: (f"config:{n}" if n in _CONFIG_KWARGS else f"case:$.input.{n}")
        for n in kwarg_names
    }
    return EvaluationContract(
        agent_id=agent_id,
        invocation=InvocationContract(
            kwargs=kwargs, case_binding=case_binding, constructor_deps=["RetrievalService"],
        ),
        has_retrieval_signal=True,
        query_planning_subcall=archetype.startswith("rag_query_planning"),
    )


@pytest.mark.parametrize("shape_key", sorted(_KNOWN_SHAPE_KWARG_SETS))
def test_codespectra_shape_archetype_and_generic_field_keys(shape_key):
    archetype, agent_id = shape_key.split(":", 1)
    kwarg_names = _KNOWN_SHAPE_KWARG_SETS[shape_key]
    contract = _contract_for_shape(agent_id, archetype, kwarg_names)

    # (1) archetype classification snapshot — must stay stable across Slices 2-6.
    assert _archetype_for(contract) == archetype

    # (2) generic-path field-descriptor key-set: _generate_generic always excludes exactly the
    # case_binding "config:"-bound names (D9) — for CodeSpectra's real shapes that is always
    # {"provider_id", "model_id"}, so this formula matches the generator's real output key-set.
    config_names = config_kwarg_names_from_case_binding(contract.invocation.case_binding)
    generic_field_keys = frozenset(kwarg_names) - config_names
    builder_field_keys = _builder_field_keys(shape_key, archetype)

    # PARITY CHECK (Slice 3c gate): the generic path exposes raw kwarg names; the hand-crafted
    # archetype builders semantically EXPAND some of those names (e.g. "mem_ctx" -> bundle +
    # folder_tree + doc_ctx + manifest_ctx + repo_name) into a richer, differently-keyed shape.
    # These are NOT equal for any of CodeSpectra's 10 real shapes -> Slice 3c retains
    # _KNOWN_SHAPE_KWARG_SETS rather than routing everyone through generic. If this assertion
    # ever starts failing (i.e. the sets become equal), the Slice 3c removal decision should be
    # revisited — generic would then be a safe, lossless replacement for the builder.
    assert generic_field_keys != builder_field_keys, (
        f"{shape_key}: generic={sorted(generic_field_keys)} builder={sorted(builder_field_keys)} "
        "— parity now holds; revisit the Slice 3c PAUSE note in synthetic_agent_io.py"
    )
