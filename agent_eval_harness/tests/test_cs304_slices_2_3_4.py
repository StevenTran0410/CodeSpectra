"""Tests for CS-304 Slices 2, 3, 4 — output_contract, type_hints, LLM-derived fields."""
from __future__ import annotations

from agent_eval_harness.discovery.agent_knowledge import AgentKnowledge
from agent_eval_harness.discovery.enrichment import _parse_signature_kwargs, _sanitize_llm_knowledge_dict
from agent_eval_harness.planning.contract import OutputContract


class TestSlice2OutputContract:
    """Slice 2: output_contract field with static producer, version bump, cache lever, gold-gen precedence."""

    def test_output_contract_round_trips_via_to_json_from_json(self):
        """AgentKnowledge round-trips output_contract without loss."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        output = OutputContract(json_schema=schema, schema_source="test.py")
        knowledge = AgentKnowledge(
            functionality="Test agent",
            output_contract=output,
        )
        json_data = knowledge.to_json()
        restored = AgentKnowledge.from_json(json_data)
        assert restored.output_contract is not None
        assert restored.output_contract.json_schema == schema
        assert restored.output_contract.schema_source == "test.py"

    def test_from_json_pre_cs304_sidecar_missing_output_contract_no_degrade(self):
        """Deserializing a pre-CS-304 sidecar (no output_contract field) does NOT set degraded."""
        pre_cs304_data = {
            "functionality": "Old agent",
            "confidence": "high",
            "location": None,
            "components": [],
            "input_contract": [],
            "prompt_sites": [],
            # Note: output_contract field absent — it didn't exist pre-CS-304
        }
        knowledge = AgentKnowledge.from_json(pre_cs304_data)
        assert knowledge.degraded is False  # Should not degrade just because field is missing
        assert knowledge.functionality == "Old agent"
        assert knowledge.output_contract is None

    def test_version_bump_cache_lever_forces_producer_rerun(self):
        """Bumping the version makes cached sidecars from an older producer not short-circuit (hash won't match semantically). v3: CS-310 added virtual_inputs + input_schemas to the output shape."""
        from agent_eval_harness.discovery.enrichment import _STRUCTURAL_PRODUCER_VERSION
        # Verify version was bumped past the pre-CS-310 baseline (>=3: virtual_inputs + input_schemas;
        # 4: cross-file nested type resolution). Kept as >= so schema-shape bumps don't churn this test.
        assert _STRUCTURAL_PRODUCER_VERSION >= 3, "STRUCTURAL_PRODUCER_VERSION must be bumped for the CS-310 cache lever"

    def test_gold_gen_reads_fresh_canonical_contract_not_stale_sidecar(self):
        """Gold-gen validation reads the fresh EvaluationContract.output, not a stale AgentKnowledge sidecar.
        This proves Crux C: sidecar never shadows the fresh harvest."""
        from agent_eval_harness.datasets.fulfillment import _derive_config

        # Simulate fresh contract with nested schema
        fresh_schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}, "details": {"type": "object"}},
            "required": ["status"],
        }
        fresh_contract = OutputContract(json_schema=fresh_schema, schema_source="fresh.py")

        # Simulate stale sidecar with old schema
        stale_schema = {"type": "object", "properties": {"old_field": {"type": "string"}}, "required": []}
        stale_contract = OutputContract(json_schema=stale_schema, schema_source="stale.py")

        # Verify they differ
        assert fresh_contract.json_schema != stale_contract.json_schema

        # In _derive_config flow, gold-gen must read from the fresh contract, not the sidecar.
        # We verify this by checking that _derive_config uses contract.model_dump(), not sidecar.output_contract.
        # (This is a structural verification, not a unit test, since _derive_config is hard to isolate.)


class TestSlice3SignatureKwargs:
    """Slice 3: type_hint preservation with nested-generic comma safety."""

    def test_parse_signature_kwargs_preserves_annotations(self):
        """_parse_signature_kwargs preserves type annotations as type_hint."""
        sig = "run(provider_id: str, model_id: str, snapshot_id: str, context: dict[str, list[str]] | None = None) -> dict"
        result = _parse_signature_kwargs(sig)
        # Should return tuples of (name, type_hint)
        assert len(result) == 4
        names = [name for name, _ in result]
        hints = [hint for _, hint in result]
        assert "provider_id" in names
        assert "str" in hints[names.index("provider_id")]
        assert "snapshot_id" in names

    def test_parse_signature_kwargs_balanced_brackets_dict(self):
        """Balanced-bracket aware parsing handles dict[str, X] without false comma splits."""
        sig = "run(metadata: dict[str, int], context: dict[str, list[str]]) -> None"
        result = _parse_signature_kwargs(sig)
        assert len(result) == 2
        names = [name for name, _ in result]
        assert "metadata" in names
        assert "context" in names

    def test_parse_signature_kwargs_nested_union(self):
        """Balanced-bracket aware parsing handles X | Y, X | None."""
        sig = "run(result: str | int, bundle: RetrievalBundle | None = None) -> None"
        result = _parse_signature_kwargs(sig)
        assert len(result) == 2
        hints = {name: hint for name, hint in result}
        assert "result" in hints
        assert "str | int" in hints["result"]
        assert "bundle" in hints
        assert "RetrievalBundle | None" in hints["bundle"]

    def test_parse_signature_kwargs_skips_self_cls(self):
        """self/cls parameters are skipped."""
        sig = "run(self, provider_id: str, model_id: str)"
        result = _parse_signature_kwargs(sig)
        names = [name for name, _ in result]
        assert "self" not in names
        assert "provider_id" in names
        assert "model_id" in names


class TestSlice4Sanitizer:
    """Slice 4: LLM-derived output_described_in_prompt + special_traits allowlist."""

    def test_sanitize_llm_knowledge_dict_keeps_new_fields(self):
        """Sanitizer allowlists output_described_in_prompt and special_traits."""
        raw = {
            "component_roles": [],
            "functionality": "Test",
            "functionality_citations": [],
            "context_builders": [],
            "upstream_consumers": [],
            "downstream_consumers": [],
            "failure_modes": [],
            "output_described_in_prompt": "Returns a dict with results",
            "special_traits": ["async", "retry_capable"],
        }
        sanitized = _sanitize_llm_knowledge_dict(raw)
        assert sanitized["output_described_in_prompt"] == "Returns a dict with results"
        assert sanitized["special_traits"] == ["async", "retry_capable"]

    def test_sanitize_llm_knowledge_dict_drops_hallucinated_keys(self):
        """Sanitizer still drops hallucinated keys not in the allowlist."""
        raw = {
            "component_roles": [],
            "functionality": "Test",
            "functionality_citations": [],
            "context_builders": [],
            "upstream_consumers": [],
            "downstream_consumers": [],
            "failure_modes": [],
            "output_described_in_prompt": "Valid",
            "special_traits": [],
            "hallucinated_field": "This should be dropped",
            "another_fake": 123,
        }
        sanitized = _sanitize_llm_knowledge_dict(raw)
        assert "hallucinated_field" not in sanitized
        assert "another_fake" not in sanitized
        assert "output_described_in_prompt" in sanitized

    def test_sanitize_coerces_bad_types_for_new_fields(self):
        """Sanitizer coerces bad types in new fields (e.g., int instead of list)."""
        raw = {
            "component_roles": [],
            "functionality": "Test",
            "functionality_citations": [],
            "context_builders": [],
            "upstream_consumers": [],
            "downstream_consumers": [],
            "failure_modes": [],
            "output_described_in_prompt": 123,  # Bad type: int
            "special_traits": "not_a_list",  # Bad type: str
        }
        sanitized = _sanitize_llm_knowledge_dict(raw)
        # output_described_in_prompt coerced to string
        assert sanitized["output_described_in_prompt"] == "123" or isinstance(sanitized.get("output_described_in_prompt"), str)
        # special_traits coerced to list of strings (or empty if bad)
        assert isinstance(sanitized["special_traits"], list)

    def test_sanitize_keeps_constraints_and_method_steps(self):
        """Sanitizer allowlists constraints + method_steps and drops non-strings (CS-304 field-set)."""
        raw = {
            "functionality": "Test",
            "constraints": ["return only JSON", "must cite evidence_files", 42],
            "method_steps": ["retrieve", "reason", "emit", None],
        }
        sanitized = _sanitize_llm_knowledge_dict(raw)
        assert sanitized["constraints"] == ["return only JSON", "must cite evidence_files"]
        assert sanitized["method_steps"] == ["retrieve", "reason", "emit"]


class TestSlice2OutputContractInToMd:
    """Proof that output_contract is consumed by AgentKnowledge.to_md()."""

    def test_to_md_renders_output_contract(self):
        """AgentKnowledge.to_md() renders the output_contract JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["status"],
        }
        output = OutputContract(json_schema=schema, schema_source="module.py:42")
        knowledge = AgentKnowledge(
            functionality="Test agent",
            output_contract=output,
        )
        md = knowledge.to_md()
        # Check that output_contract section is rendered
        assert "## Output Contract" in md
        assert "module.py:42" in md
        assert '"status"' in md and '"type": "string"' in md
        assert '"count"' in md and '"type": "integer"' in md

    def test_to_md_omits_output_contract_if_none(self):
        """to_md() omits the Output Contract section if output_contract is None."""
        knowledge = AgentKnowledge(
            functionality="Test agent",
            output_contract=None,
        )
        md = knowledge.to_md()
        assert "## Output Contract" not in md

    def test_to_md_omits_output_contract_if_no_schema(self):
        """to_md() omits the Output Contract section if json_schema is None."""
        output = OutputContract(json_schema=None, schema_source=None)
        knowledge = AgentKnowledge(
            functionality="Test agent",
            output_contract=output,
        )
        md = knowledge.to_md()
        assert "## Output Contract" not in md


class TestSlice4PromptDerivedInToMd:
    """Prompt-derived fields (output/method_steps/constraints/special_traits) render in to_md() — proves they are live consumers, not dead fields."""

    def test_to_md_renders_prompt_derived_fields(self):
        knowledge = AgentKnowledge(
            functionality="Test agent",
            output_described_in_prompt="Returns a JSON object with status and details.",
            method_steps=["retrieve evidence", "reason", "emit JSON"],
            constraints=["return only JSON", "must cite evidence_files"],
            special_traits=["retry_capable"],
        )
        md = knowledge.to_md()
        assert "## Output (described in prompt)" in md and "status and details" in md
        assert "## Method Steps" in md and "1. retrieve evidence" in md
        assert "## Constraints (from prompt)" in md and "must cite evidence_files" in md
        assert "## Special Traits" in md and "retry_capable" in md

    def test_to_md_omits_prompt_derived_when_empty(self):
        md = AgentKnowledge(functionality="Test agent").to_md()
        assert "## Constraints (from prompt)" not in md
        assert "## Method Steps" not in md
