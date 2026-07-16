"""Tests for the synthetic_agent_io generator's fan_in_judge archetype (CS-289 Phase 0)."""
import json

import pytest

from agent_eval_harness.datasets.generators.synthetic_agent_io import generate, is_dataset_stale, schema_hash
from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.llm.fake_client import FakeLLMClient

pytestmark = pytest.mark.asyncio

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


async def test_generate_unimplemented_archetype_raises():
    config = _config()
    config["archetype"] = "unimplemented"
    llm_client = FakeLLMClient(LLMResponse(content="[]", model="fake"))

    with pytest.raises(ValueError, match="unimplemented"):
        await generate(config, llm_client)


async def test_generate_requires_llm_client():
    with pytest.raises(ValueError, match="LLM client is required"):
        await generate(_config(), None)


# --- RAG-writer archetypes (Phases 1-5) ---

_G_SCHEMA = {
    "type": "object",
    "properties": {"entrypoint": {"type": "object"}, "confidence": {"type": "string"}},
    "required": ["entrypoint", "confidence"],
}


def _rag_config(agent_id: str, archetype: str, count: int = 1) -> dict:
    return {
        "dataset_name": f"test_synth_{agent_id}",
        "agent_id": agent_id,
        "archetype": archetype,
        "contract": {"output": {"json_schema": _G_SCHEMA}},
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
    assert cases[0].input["shape"] == "retrieval_only"
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
    assert cases[0].input["shape"] == "retrieval_and_upstream"
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
    assert cases[0].input["shape"] == "mem_ctx_and_retrieval"
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
    assert cases[0].input["shape"] == "mem_ctx_participant"
    assert cases[0].input["identity_output"]["domain"] == "web"


async def test_generate_folder_tree_archetypes_all_get_faithfulness_guard():
    """Regression test for a real bug: only rag_mem_ctx (project_identity) carried the
    CRITICAL faithfulness instruction, so rag_mem_ctx_participant (architecture/structure)
    and rag_query_planning_mem_ctx (feature_map) generated evidence paths absent from their
    own folder_tree. The instruction must fire for every archetype that includes folder_tree."""
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
    assert cases[0].input["shape"] == "query_planning"
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
    assert cases[0].input["shape"] == "query_planning_mem_ctx"
    assert cases[0].input["architecture_output"] == {"main_layers": []}


async def test_generate_rag_archetype_rejects_invalid_gold_after_retry_ceiling():
    """Proves the shared _generate_validated_cases retry/reject discipline applies to the
    RAG-writer archetypes too, not just fan_in_judge."""
    bad_candidate = {"input": {"bundle": _bundle_evidence()}, "gold": {"confidence": "high"}}  # missing entrypoint
    llm_client = FakeLLMClient([LLMResponse(content=json.dumps([bad_candidate]), model="fake")] * 3)

    cases = await generate(_rag_config("glossary", "rag_single_shot"), llm_client)

    assert cases == []
    assert len(llm_client.calls) == 2  # _MAX_CONSECUTIVE_DRY_ROUNDS(2), never a 3rd identical call


async def test_generate_recovers_cases_from_truncated_response_via_json_repair():
    """Regression test for a real bug: a completion cut off mid-object (e.g. hit an internal
    stop condition before finishing the array) used to parse as [] and waste the whole round.
    json_repair should recover whichever leading array elements are structurally complete."""
    complete = json.dumps({"input": {"bundle": _bundle_evidence()}, "gold": _gold()})
    truncated = f'[{complete}, {{"input": {{"bundle": {{"evidences": [{{"rel_path": "x.py"'
    llm_client = FakeLLMClient(LLMResponse(content=truncated, model="fake"))

    cases = await generate(_rag_config("glossary", "rag_single_shot", count=1), llm_client)

    assert len(cases) == 1
    assert cases[0].expected == _gold()


# --- Phase 6: staleness / schema-hash versioning ---

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
    """Regression test for a real bug: asking for the full count (default 20) in ONE
    completion risked truncation for rich archetypes (e.g. rag_mem_ctx's 4-artifact cases),
    silently surfacing as "generated 0 cases". Batches must be capped, never all-at-once."""

    def _batch(n: int) -> str:
        return json.dumps([{"input": {"bundle": _bundle_evidence()}, "gold": _gold()} for _ in range(n)])

    # 20 cases needed, capped at 3/call -> 7 calls (3*6 + 2), never a single 20-case request.
    responses = [LLMResponse(content=_batch(n), model="fake") for n in (3, 3, 3, 3, 3, 3, 2)]
    llm_client = FakeLLMClient(responses)

    cases = await generate(_rag_config("glossary", "rag_single_shot", count=20), llm_client)

    assert len(cases) == 20
    assert len(llm_client.calls) == 7


# --- D9: generic builder equivalence (AC4) ---


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
    """AC4: an agent with a non-CodeSpectra kwarg set bypasses fast-path and uses _generate_generic."""
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
    """AC4: when contract.output.json_schema is None, cases are tagged schema_source=none."""
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


async def test_d9_known_kwarg_set_still_uses_fast_path():
    """AC4 inverse: a CodeSpectra-shaped kwarg set still routes through the archetype builder."""
    config = _rag_config("glossary", "rag_single_shot")
    # glossary's exact kwarg set is in _KNOWN_SHAPE_KWARG_SETS
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
    assert cases[0].input["shape"] == "retrieval_only"  # fast-path builder shape, not "generic"
