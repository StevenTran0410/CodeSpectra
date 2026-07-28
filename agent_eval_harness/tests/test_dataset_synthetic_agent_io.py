"""Tests for the synthetic_agent_io generator's fan_in_judge archetype."""
import json

import pytest

from agent_eval_harness.datasets.generators.synthetic_agent_io import (
    _extract_text_values,
    generate,
    is_dataset_stale,
    schema_hash,
)
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient

# Local copy of CodeSpectra shapes (original was in synthetic_agent_io.py, deleted in Slice 4)
_KNOWN_SHAPE_KWARG_SETS: dict[str, frozenset[str]] = {
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

_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_confidence": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["overall_confidence", "notes"],
}


def _config(count: int = 2) -> dict:
    return {
        "dataset_name": "test_synth_k",
        "agent_id": "auditor",
        "archetype": "fan_in_judge",
        "contract": {
            "output": {"json_schema": _SCHEMA},
            "field_downstream_consumers": {
                "A": ["confidence", "blind_spots", "purpose"],
                "B": ["confidence", "main_layers"],
            },
        },
        "count": count,
    }


def _valid_case(i: int) -> dict:
    return {
        "input": {
            "A": {"confidence": "high", "blind_spots": [], "purpose": f"repo {i}"},
            "B": {"confidence": "medium", "main_layers": ["api"]},
        },
        "gold": {"overall_confidence": "high", "notes": f"looks fine {i}"},
    }


def _invalid_case(i: int) -> dict:
    return {
        "input": {
            "A": {"confidence": "high", "blind_spots": [], "purpose": f"repo {i}"},
            "B": {"confidence": "medium", "main_layers": ["api"]},
        },
        "gold": {"notes": f"missing overall_confidence {i}"},  # violates required
    }


async def test_generate_fan_in_judge_shape_and_self_validation():
    payload = json.dumps([_valid_case(1), _valid_case(2)])
    llm_client = FakeLLMClient(LLMResponse(content=payload, model="fake"))

    cases = await generate(_config(count=2), llm_client)

    assert len(cases) == 2
    for case in cases:
        assert case.kind == "synthetic_agent_io"
        assert case.input["shape"] == "all_sections"
        assert set(case.input["all_sections"]) == {"A", "B"}
        assert case.expected["overall_confidence"] in ("high",)
        assert case.labels["agent_id"] == "auditor"
        assert case.labels["archetype"] == "fan_in_judge"
        assert case.labels["schema_hash"]
        assert case.provenance == "synthetic"


async def test_generate_rejects_schema_invalid_gold_after_retry_ceiling():
    # Every round returns only schema-invalid golds -> 2 consecutive dry rounds stops it.
    bad_payload = json.dumps([_invalid_case(1), _invalid_case(2)])
    llm_client = FakeLLMClient(
        [LLMResponse(content=bad_payload, model="fake")] * 3
    )

    cases = await generate(_config(count=2), llm_client)

    assert cases == []  # dropped, never kept as best-effort
    assert len(llm_client.calls) == 2  # _MAX_CONSECUTIVE_DRY_ROUNDS(2), never a 3rd identical call


async def test_generate_retries_and_fills_remaining_count():
    first = json.dumps([_valid_case(1), _invalid_case(2)])
    second = json.dumps([_valid_case(3)])
    llm_client = FakeLLMClient(
        [LLMResponse(content=first, model="fake"), LLMResponse(content=second, model="fake")]
    )

    cases = await generate(_config(count=2), llm_client)

    assert len(cases) == 2
    assert len(llm_client.calls) == 2


async def test_fan_in_judge_with_no_output_schema_flags_needs_human():
    """The fan_in_judge early-return path must flag needs_human when no output schema was harvestable — cases are flagged, never silently stamped valid ([])."""
    config = _config(count=2)
    config["contract"]["output"] = {}  # no json_schema harvestable
    payload = json.dumps([_valid_case(1), _valid_case(2)])
    llm_client = FakeLLMClient(LLMResponse(content=payload, model="fake"))

    cases = await generate(config, llm_client)

    assert len(cases) == 2  # not dropped — flagged, not silently kept as validated
    for case in cases:
        assert case.labels.get("schema_source") == "none"
        assert "needs_human" in case.labels


async def test_schema_missing_flag_carries_contract_harvest_note():
    """The 'no output schema statically harvestable' harvest note flows to the case label rather than a generic placeholder."""
    config = _config(count=1)
    config["contract"]["output"] = {}
    config["contract"]["needs_human"] = ["auditor: no output schema statically harvestable"]
    payload = json.dumps([_valid_case(1)])
    llm_client = FakeLLMClient(LLMResponse(content=payload, model="fake"))

    cases = await generate(config, llm_client)

    assert cases[0].labels["needs_human"] == "auditor: no output schema statically harvestable"


async def test_generate_unimplemented_archetype_raises():
    config = _config()
    config["archetype"] = "unimplemented"
    llm_client = FakeLLMClient(LLMResponse(content="[]", model="fake"))

    with pytest.raises(ValueError, match="unimplemented"):
        await generate(config, llm_client)


async def test_generate_requires_llm_client():
    with pytest.raises(ValueError, match="LLM client is required"):
        await generate(_config(), None)


_G_SCHEMA = {
    "type": "object",
    "properties": {"entrypoint": {"type": "object"}, "confidence": {"type": "string"}},
    "required": ["entrypoint", "confidence"],
}


def _rag_config(agent_id: str, archetype: str, count: int = 1) -> dict:
    """These RAG-writer-archetype tests exist to exercise the archetype builders themselves, so give them the real CodeSpectra kwarg shape (routing through the fast-path) whenever one is known."""
    contract: dict = {"output": {"json_schema": _G_SCHEMA}}
    shape = _KNOWN_SHAPE_KWARG_SETS.get(f"{archetype}:{agent_id}")
    if shape:
        contract["invocation"] = {
            "kwargs": [{"name": n} for n in sorted(shape)],
            "case_binding": {n: f"config:{n}" for n in ("provider_id", "model_id", "snapshot_id", "profile") if n in shape},
        }
    return {
        "dataset_name": f"test_synth_{agent_id}",
        "agent_id": agent_id,
        "archetype": archetype,
        "contract": contract,
        "virtual_inputs": [
            {"name": "bundle"}, {"name": "folder_tree"}, {"name": "doc_ctx"},
            {"name": "manifest_ctx"}, {"name": "mem_ctx"}, {"name": "arch_bundle"},
            {"name": "repo_name"}, {"name": "query"},
        ],
        "count": count,
    }


def _bundle_evidence() -> dict:
    return {"evidences": [{"rel_path": "src/main.py", "excerpt": "def main(): pass", "score": 0.9}]}


def _gold() -> dict:
    return {"entrypoint": {"file": "src/main.py", "reason": "main"}, "confidence": "high"}


async def test_generate_rag_single_shot_shape():
    payload = json.dumps([{"input": {"bundle": _bundle_evidence()}, "gold": _gold()}])
    llm_client = FakeLLMClient(LLMResponse(content=payload, model="fake"))

    cases = await generate(_rag_config("important_files", "rag_single_shot"), llm_client)

    assert len(cases) == 1
    assert cases[0].input["shape"] == "generic"
    assert "bundle" in cases[0].input
    assert cases[0].labels["archetype"] == "rag_single_shot"


async def test_generate_rag_upstream_shape_includes_upstream_field():
    candidate = {
        "input": {"bundle": _bundle_evidence(), "important_files_output": {"entrypoint": {"file": "a.py"}}},
        "gold": _gold(),
    }
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))

    cases = await generate(_rag_config("onboarding", "rag_upstream"), llm_client)

    assert len(cases) == 1
    assert cases[0].input["shape"] == "generic"
    assert cases[0].input["important_files_output"]["entrypoint"]["file"] == "a.py"


async def test_generate_rag_mem_ctx_shape_includes_four_artifacts():
    candidate = {
        "input": {
            "bundle": _bundle_evidence(), "folder_tree": "src/\n  main.py\n",
            "doc_ctx": "readme", "manifest_ctx": "deps", "repo_name": "myrepo",
        },
        "gold": _gold(),
    }
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))

    cases = await generate(_rag_config("project_identity", "rag_mem_ctx"), llm_client)

    assert len(cases) == 1
    assert cases[0].input["shape"] == "generic"
    assert cases[0].input["repo_name"] == "myrepo"
    assert cases[0].input["doc_ctx"] == "readme"


async def test_generate_rag_mem_ctx_participant_shape():
    candidate = {
        "input": {
            "bundle": _bundle_evidence(), "folder_tree": "src/\n",
            "identity_output": {"domain": "web"},
        },
        "gold": _gold(),
    }
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))

    cases = await generate(_rag_config("architecture", "rag_mem_ctx_participant"), llm_client)

    assert len(cases) == 1
    assert cases[0].input["shape"] == "generic"
    assert cases[0].input["identity_output"]["domain"] == "web"


async def test_generate_folder_tree_archetypes_all_get_faithfulness_guard():
    """Regression test: only rag_mem_ctx used to carry the CRITICAL faithfulness instruction, so other folder_tree archetypes generated evidence paths absent from their own folder_tree. The instruction must fire for every archetype that includes folder_tree."""
    candidate = {"input": {"bundle": _bundle_evidence(), "folder_tree": "src/\n"}, "gold": _gold()}
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))

    await generate(_rag_config("architecture", "rag_mem_ctx_participant"), llm_client)

    assert "CRITICAL (faithfulness guard)" in llm_client.calls[0][0].content


async def test_generate_rag_query_planning_shape():
    candidate = {
        "input": {"bundle": _bundle_evidence(), "structure_output": {"folders": []}},
        "gold": _gold(),
    }
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))

    cases = await generate(_rag_config("conventions", "rag_query_planning"), llm_client)

    assert len(cases) == 1
    assert cases[0].input["shape"] == "generic"
    assert cases[0].input["structure_output"] == {"folders": []}


async def test_generate_rag_query_planning_no_upstream_for_risk():
    """J (risk) has no upstream agent-output field in _UPSTREAM_SPECS_BY_AGENT."""
    candidate = {"input": {"bundle": _bundle_evidence()}, "gold": _gold()}
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))

    cases = await generate(_rag_config("risk", "rag_query_planning"), llm_client)

    assert len(cases) == 1
    assert "structure_output" not in cases[0].input


async def test_generate_rag_query_planning_mem_ctx_shape():
    candidate = {
        "input": {
            "bundle": _bundle_evidence(), "folder_tree": "src/\n",
            "identity_output": {"domain": "web"}, "architecture_output": {"main_layers": []},
        },
        "gold": _gold(),
    }
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))

    cases = await generate(_rag_config("feature_map", "rag_query_planning_mem_ctx"), llm_client)

    assert len(cases) == 1
    assert cases[0].input["shape"] == "generic"
    assert cases[0].input["architecture_output"] == {"main_layers": []}


async def test_generate_rag_archetype_rejects_invalid_gold_after_retry_ceiling():
    """Proves the shared _generate_validated_cases retry/reject discipline applies to the RAG-writer archetypes too, not just fan_in_judge."""
    bad_candidate = {"input": {"bundle": _bundle_evidence()}, "gold": {"confidence": "high"}}  # missing entrypoint
    llm_client = FakeLLMClient([LLMResponse(content=json.dumps([bad_candidate]), model="fake")] * 3)

    cases = await generate(_rag_config("glossary", "rag_single_shot"), llm_client)

    assert cases == []
    assert len(llm_client.calls) == 2  # _MAX_CONSECUTIVE_DRY_ROUNDS(2), never a 3rd identical call


async def test_generate_recovers_cases_from_truncated_response_via_json_repair():
    """Regression test: a completion cut off mid-object used to parse as [] and waste the whole round — json_repair should recover whichever leading array elements are structurally complete."""
    complete = json.dumps({"input": {"bundle": _bundle_evidence()}, "gold": _gold()})
    truncated = f'[{complete}, {{"input": {{"bundle": {{"evidences": [{{"rel_path": "x.py"'
    llm_client = FakeLLMClient(LLMResponse(content=truncated, model="fake"))

    cases = await generate(_rag_config("glossary", "rag_single_shot", count=1), llm_client)

    assert len(cases) == 1
    assert cases[0].expected == _gold()


def test_is_dataset_stale_false_when_hash_matches_current_schema():
    current = {"type": "object", "properties": {"a": {"type": "string"}}}
    cases = [{"labels_json": json.dumps({"schema_hash": schema_hash(current)})}]
    assert is_dataset_stale(cases, current) is False


def test_is_dataset_stale_true_when_hash_differs_from_current_schema():
    old_schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    current_schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
    cases = [{"labels_json": json.dumps({"schema_hash": schema_hash(old_schema)})}]
    assert is_dataset_stale(cases, current_schema) is True


def test_is_dataset_stale_false_when_no_cases_or_no_labels():
    current = {"type": "object"}
    assert is_dataset_stale([], current) is False
    assert is_dataset_stale([{"labels_json": None}], current) is False
    assert is_dataset_stale([{"labels_json": json.dumps({"agent_id": "x"})}], current) is False


async def test_generate_requests_in_small_batches_for_large_counts():
    """Regression test: asking for the full count (default 20) in ONE completion risked truncation for rich archetypes, silently surfacing as "generated 0 cases". Batches must be capped, never all-at-once."""

    def _batch(n: int) -> str:
        return json.dumps([{"input": {"bundle": _bundle_evidence()}, "gold": _gold()} for _ in range(n)])

    # 20 cases needed, capped at 3/call -> 7 calls (3*6 + 2), never a single 20-case request.
    responses = [LLMResponse(content=_batch(n), model="fake") for n in (3, 3, 3, 3, 3, 3, 2)]
    llm_client = FakeLLMClient(responses)

    cases = await generate(_rag_config("glossary", "rag_single_shot", count=20), llm_client)

    assert len(cases) == 20
    assert len(llm_client.calls) == 7


def _generic_config(agent_id: str = "retriever", archetype: str = "rag_upstream") -> dict:
    """Config for a non-CodeSpectra agent whose kwarg set is not in _KNOWN_SHAPE_KWARG_SETS."""
    return {
        "dataset_name": "test_generic",
        "agent_id": agent_id,
        "archetype": archetype,
        "contract": {
            "output": {"json_schema": _G_SCHEMA},
            "invocation": {
                "kwargs": [
                    {"name": "query", "annotation": "str"},
                    {"name": "documents", "annotation": "list[str]"},
                ],
            },
        },
        "count": 1,
    }


async def test_d9_generic_builder_fires_for_unknown_kwarg_set():
    """An agent with a non-CodeSpectra kwarg set bypasses fast-path and uses _generate_generic."""
    candidate = {
        "input": {"query": "what is X?", "documents": ["doc1"]},
        "gold": {"entrypoint": {"file": "main.py", "reason": "entry"}, "confidence": "high"},
    }
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))

    cases = await generate(_generic_config(), llm_client)

    assert len(cases) == 1
    assert cases[0].input["shape"] == "generic"
    assert cases[0].labels["archetype"] == "rag_upstream"
    assert "query" in cases[0].input
    assert "documents" in cases[0].input


async def test_d9_generic_builder_tags_schema_source_none_when_no_output_schema():
    """When contract.output.json_schema is None, cases are tagged schema_source=none."""
    config = _generic_config()
    config["contract"]["output"] = {}  # no json_schema
    candidate = {
        "input": {"query": "what is X?", "documents": ["doc1"]},
        "gold": {},
    }
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))

    cases = await generate(config, llm_client)

    assert len(cases) == 1
    assert cases[0].labels.get("schema_source") == "none"
    assert "needs_human" in cases[0].labels


async def test_d9_known_kwarg_set_routes_through_generic():
    """CS-310 Slice 4: CodeSpectra-shaped kwarg sets now route through the generic path (all archetype builders deleted)."""
    config = _rag_config("glossary", "rag_single_shot")
    # glossary's exact kwarg set was in _KNOWN_SHAPE_KWARG_SETS; now all retrieval agents use generic
    config["contract"]["invocation"] = {
        "kwargs": [
            {"name": "provider_id"}, {"name": "model_id"},
            {"name": "snapshot_id"}, {"name": "profile"},
        ]
    }
    candidate = {"input": {"bundle": _bundle_evidence()}, "gold": _gold()}
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))

    cases = await generate(config, llm_client)

    assert len(cases) == 1
    assert cases[0].input["shape"] == "generic"  # All retrieval agents use generic path now


async def test_dispatch_fan_in_judge_routes_by_shape_not_agent_id():
    """A NON-CodeSpectra agent_id with field_downstream_consumers still dispatches to the fan_in_judge builder — the gate is contract shape, never an agent-id table."""
    config = {
        "dataset_name": "test_synth_foreign_fan_in",
        "agent_id": "some_foreign_reviewer",
        "archetype": "fan_in_judge",
        "contract": {
            "output": {"json_schema": _SCHEMA},
            "field_downstream_consumers": {"X": ["confidence"], "Y": ["confidence"]},
        },
        "count": 1,
    }
    candidate = {
        "input": {
            "X": {"confidence": "high"},
            "Y": {"confidence": "low"},
        },
        "gold": {"overall_confidence": "medium", "notes": "mixed"},
    }
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))

    cases = await generate(config, llm_client)

    assert len(cases) == 1
    assert cases[0].input["shape"] == "all_sections"
    assert set(cases[0].input["all_sections"]) == {"X", "Y"}


async def test_dispatch_empty_kwargs_routes_to_generic_not_a_codespectra_builder():
    """An agent whose harvest yields NO kwargs at all must degrade to the generic path, never fall into a CodeSpectra archetype builder that would render a fixed 'bundle.evidences' shape unrelated to the real (empty) contract."""
    config = _generic_config(agent_id="bare_tool_agent", archetype="rag_single_shot")
    config["contract"]["invocation"] = {"kwargs": []}
    candidate = {"input": {}, "gold": {"entrypoint": {"file": "main.py", "reason": "entry"}, "confidence": "high"}}
    llm_client = FakeLLMClient(LLMResponse(content=json.dumps([candidate]), model="fake"))

    cases = await generate(config, llm_client)

    assert len(cases) == 1
    assert cases[0].input["shape"] == "generic"  # not "retrieval_only" (the builder's shape)


# target-specific: uses fan_in_judge inputs (letter-keyed dicts) which have no top-level string
# values, so the dedup degrades to a no-op — which is the correct behavior per the spec.


class _FakeEmbeddingClient:
    """Returns distinct unit vectors for different texts, identical vector for same text."""

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}
        self._counter = 0

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            if text not in self._cache:
                # Orthogonal unit vectors: slot counter in a 10-dim space
                vec = [0.0] * 10
                vec[self._counter % 10] = 1.0
                self._cache[text] = vec
                self._counter += 1
            result.append(self._cache[text])
        return result


class _IdenticalEmbeddingClient:
    """Always returns the same unit vector, regardless of text — simulates near-duplicate."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * len(texts)


async def test_d6_no_false_positive_identical_field_names_different_values():
    """Two cases with identical field names but different string values must NOT be treated as near-duplicates — embedding whole JSON would inflate cosine-sim via shared keys, so text-values-only embedding must score them as distinct."""
    candidate_a = {
        "input": {"query": "Python packaging conventions", "repo_name": "my-python-lib"},
        "gold": {"entrypoint": {"file": "setup.py", "reason": "entry"}, "confidence": "high"},
    }
    candidate_b = {
        "input": {"query": "Kubernetes deployment patterns", "repo_name": "k8s-operator"},
        "gold": {"entrypoint": {"file": "main.go", "reason": "entry"}, "confidence": "medium"},
    }
    payload = json.dumps([candidate_a, candidate_b])
    llm_client = FakeLLMClient(LLMResponse(content=payload, model="fake"))
    embed_client = _FakeEmbeddingClient()

    cases = await generate(
        _rag_config("important_files", "rag_single_shot", count=2),
        llm_client,
        embedding_client=embed_client,
    )

    assert len(cases) == 2, "distinct cases must not be auto-dropped by dedup"
    assert not any(c.labels.get("near_duplicate") for c in cases), (
        "cases with different text values must not be flagged near-duplicate"
    )


def test_d6_extracts_text_from_nested_inputs():
    """Regression: real generated inputs are nested (letter-keyed objects, bundle.evidences), so a top-level-only walk returned '' for every case and collapsed them all to sim 1.0."""
    letter_keyed_a = {
        "A": {"summary": "auth module is well tested", "confidence": "high"},
        "B": {"summary": "db layer has no tests", "confidence": "low"},
    }
    letter_keyed_b = {
        "A": {"summary": "payments flow is untested", "confidence": "low"},
        "B": {"summary": "unrelated content entirely", "confidence": "high"},
    }
    text_a = _extract_text_values(letter_keyed_a)
    text_b = _extract_text_values(letter_keyed_b)
    assert text_a and text_b, "nested string leaves must be reached, not skipped"
    assert text_a != text_b, "distinct nested cases must not collapse to the same text"

    bundled = {
        "shape": "kwargs",
        "bundle": {"evidences": [{"rel_path": "src/x.py", "excerpt": "def foo(): pass", "score": 0.9}]},
    }
    text_c = _extract_text_values(bundled)
    assert "def foo(): pass" in text_c, "evidence excerpts distinguish cases and must be embedded"
    assert "kwargs" not in text_c, "the constant 'shape' discriminator must stay out of the text"


async def test_d6_near_duplicate_flagged_not_dropped():
    """Cases scoring 0.80–0.93 are accepted with near_duplicate=True, not auto-dropped."""
    # Uses _FakeEmbeddingClient (orthogonal vectors), so this only verifies distinct-enough
    # cases carry no near_duplicate flag — true near-duplicate flagging needs a partial-match client.
    candidate_a = {
        "input": {"query": "topic A", "repo_name": "repo-a"},
        "gold": {"entrypoint": {"file": "a.py", "reason": "r"}, "confidence": "high"},
    }
    candidate_b = {
        "input": {"query": "topic B", "repo_name": "repo-b"},
        "gold": {"entrypoint": {"file": "b.py", "reason": "r"}, "confidence": "high"},
    }
    payload = json.dumps([candidate_a, candidate_b])
    llm_client = FakeLLMClient(LLMResponse(content=payload, model="fake"))

    cases = await generate(
        _rag_config("glossary", "rag_single_shot", count=2),
        llm_client,
        embedding_client=_FakeEmbeddingClient(),
    )

    assert len(cases) == 2
    assert all("max_sim" in c.labels for c in cases[1:]), "accepted cases after first carry max_sim"


async def test_d6_no_embedding_client_behavior_unchanged():
    """With embedding_client=None, generation behavior is unchanged."""
    payload = json.dumps([_valid_case(1), _valid_case(2)])
    llm_client = FakeLLMClient(LLMResponse(content=payload, model="fake"))

    cases = await generate(_config(count=2), llm_client, embedding_client=None)

    assert len(cases) == 2
    assert not any(c.labels.get("near_duplicate") for c in cases)
    assert not any(c.labels.get("max_sim") is not None for c in cases)


def test_cs327_validate_input_gate():
    """CS-327: _validate_input cross-checks input against contract required fields, foreign fields, and types."""
    from agent_eval_harness.datasets.generators.synthetic_agent_io import (
        SyntheticAgentIOConfig,
        _validate_input,
    )

    parsed_config = SyntheticAgentIOConfig.model_validate({
        "dataset_name": "ds1",
        "agent_id": "agent_a",
        "archetype": "rag_single_shot",
        "contract": {
            "invocation": {
                "kwargs": [
                    {"name": "query", "required": True, "type_hint": "str"},
                    {"name": "context", "required": False, "resolved_schema": {"type": "object", "properties": {"title": {"type": "string"}}}},
                ]
            }
        },
        "count": 2,
    })

    # Valid input
    valid_in = {"query": "test query", "context": {"title": "doc1"}}
    assert _validate_input(valid_in, parsed_config) == []

    # Missing required field
    missing_in = {"context": {"title": "doc1"}}
    errs = _validate_input(missing_in, parsed_config)
    assert any("missing required input field 'query'" in e for e in errs)

    # Foreign field
    foreign_in = {"query": "test", "fake_param": 123}
    errs = _validate_input(foreign_in, parsed_config)
    assert any("foreign input field 'fake_param'" in e for e in errs)

    # Type mismatch (int instead of object dict)
    type_err_in = {"query": "test", "context": 12345}
    errs = _validate_input(type_err_in, parsed_config)
    assert any("expected object dict" in e for e in errs)


def test_cs327_validate_schema_enum_values():
    """CS-327: _validate_gold rejects gold containing enum values outside schema_enum_values domain."""
    from agent_eval_harness.datasets.generators.synthetic_agent_io import (
        _validate_gold,
        _validate_schema_enum_values,
    )

    enums = {
        "severity": ["low", "medium", "high"],
        "folders[].role": ["src", "tests"],
    }

    # Valid gold
    valid_gold = {"severity": "high", "folders": [{"role": "src"}, {"role": "tests"}]}
    assert _validate_schema_enum_values(valid_gold, enums) == []

    # Invalid enum value in root
    invalid_gold_root = {"severity": "critical", "folders": [{"role": "src"}]}
    errs = _validate_schema_enum_values(invalid_gold_root, enums)
    assert len(errs) == 1
    assert "not in allowed enum domain" in errs[0]

    # Invalid enum value in list item
    invalid_gold_item = {"severity": "low", "folders": [{"role": "unknown"}]}
    errs = _validate_schema_enum_values(invalid_gold_item, enums)
    assert len(errs) == 1
    assert "not in allowed enum domain" in errs[0]

    # Integrated _validate_gold test
    errs = _validate_gold(invalid_gold_root, None, schema_enum_values=enums)
    assert len(errs) > 0


async def test_cs327_coerce_stringified_json_object_kwarg():
    """CS-327: _generate_generic coerces stringified JSON object kwargs to python dicts."""
    from agent_eval_harness.datasets.generators.synthetic_agent_io import generate

    cfg = {
        "dataset_name": "ds_coerce",
        "agent_id": "agent_b",
        "archetype": "rag_single_shot",
        "contract": {
            "invocation": {
                "kwargs": [
                    {"name": "meta", "required": True, "resolved_schema": {"type": "object", "properties": {"version": {"type": "string"}}}}
                ]
            },
            "output": {"json_schema": {"type": "object", "properties": {"status": {"type": "string"}}}}
        },
        "count": 1,
    }

    # Model returns stringified JSON for object kwarg 'meta'
    case_with_stringified_kwarg = {
        "input": {"meta": '{"version": "1.0"}'},
        "gold": {"status": "ok"},
    }
    payload = json.dumps([case_with_stringified_kwarg])
    llm_client = FakeLLMClient(LLMResponse(content=payload, model="fake"))

    cases = await generate(cfg, llm_client)
    assert len(cases) == 1
    assert isinstance(cases[0].input["meta"], dict)
    assert cases[0].input["meta"]["version"] == "1.0"


def test_fix1_validate_input_rejects_fabricated_bundle():
    """FIX 1 reject path: contract declares only query; candidate with unharvested bundle MUST be rejected."""
    from agent_eval_harness.datasets.generators.synthetic_agent_io import (
        SyntheticAgentIOConfig,
        _validate_input,
    )

    parsed_config = SyntheticAgentIOConfig.model_validate({
        "dataset_name": "ds_strict",
        "agent_id": "strict_agent",
        "archetype": "rag_single_shot",
        "contract": {
            "invocation": {
                "kwargs": [
                    {"name": "query", "required": True, "type_hint": "str"}
                ]
            }
        },
        "count": 1,
    })

    bad_candidate_input = {"query": "how to build", "bundle": {"evidences": []}}
    errs = _validate_input(bad_candidate_input, parsed_config)
    assert len(errs) == 1
    assert "foreign input field 'bundle'" in errs[0]


async def test_fix2_fan_in_judge_coerces_stringified_input():
    """FIX 2: _generate_fan_in_judge coerces stringified JSON input values to dicts."""
    from agent_eval_harness.datasets.generators.synthetic_agent_io import generate

    cfg = _config(count=1)
    case_with_stringified_fan_in = {
        "input": {
            "A": '{"confidence": "high", "blind_spots": [], "purpose": "p1"}',
            "B": {"confidence": "medium", "main_layers": ["api"]},
        },
        "gold": {"overall_confidence": "high", "notes": "ok"},
    }
    payload = json.dumps([case_with_stringified_fan_in])
    llm_client = FakeLLMClient(LLMResponse(content=payload, model="fake"))

    cases = await generate(cfg, llm_client)
    assert len(cases) == 1
    sections = cases[0].input["all_sections"]
    assert isinstance(sections["A"], dict)
    assert sections["A"]["confidence"] == "high"


async def test_fix4_input_kind_strategy_signal():
    """FIX 4: observability.input_kind ('query' vs 'structured') alters prompt strategy."""
    from agent_eval_harness.datasets.generators.synthetic_agent_io import _enum_and_strategy_prompt_block, SyntheticAgentIOConfig

    cfg_query = SyntheticAgentIOConfig.model_validate({
        "dataset_name": "d1", "agent_id": "a1", "archetype": "rag_single_shot",
        "contract": {"observability": {"input_kind": "query"}},
    })
    block_q = _enum_and_strategy_prompt_block(cfg_query)
    assert "Natural language user query" in block_q

    cfg_struct = SyntheticAgentIOConfig.model_validate({
        "dataset_name": "d2", "agent_id": "a2", "archetype": "rag_single_shot",
        "contract": {"observability": {"input_kind": "structured"}},
    })
    block_s = _enum_and_strategy_prompt_block(cfg_struct)
    assert "Structured data parameters" in block_s


def test_fix5_cardinality_and_streamed_output_validation():
    """FIX 5c & cardinality: _validate_gold enforces cardinality='object' and handles streamed output flag."""
    from agent_eval_harness.datasets.generators.synthetic_agent_io import _validate_gold

    # Array gold when object expected -> rejected
    errs = _validate_gold([{"a": 1}], {"type": "object"}, cardinality="object")
    assert len(errs) > 0
    assert "gold is not a JSON object" in errs[0]

    # Object gold when array expected -> rejected
    errs_arr = _validate_gold({"a": 1}, {"type": "array"}, cardinality="array")
    assert len(errs_arr) > 0
    assert "gold is not a JSON array" in errs_arr[0]


