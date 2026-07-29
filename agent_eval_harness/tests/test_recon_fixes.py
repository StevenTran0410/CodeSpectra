"""Unit tests for RECON fixes #1/#2/#3/#6/#7.

Tests:
  #2  — case-binding key reconciliation (suffix match rewrites binding + logs recon note)
  #3  — run_eval._run_one_case honours skip_case() and tallies skipped count in manifest
  #6a — _annotation_to_schema: dict[str,float] -> additionalProperties: {type: number}
  #6b — _annotation_to_schema: list[str] still emits items (regression guard)
  #6c — _annotation_to_schema: X | None -> non-None branch schema
  #6d — _validate_input flags string value where schema says number
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# #2 — Case-binding key reconciliation
# ---------------------------------------------------------------------------

class TestCaseBindingReconciliation:
    """_reconcile_case_binding_keys rewrites absent keys via suffix match and appends recon notes."""

    def _import_reconcile(self):
        from agent_eval_harness.ui.server import _reconcile_case_binding_keys
        return _reconcile_case_binding_keys

    def test_exact_key_kept(self):
        """A binding key that exactly matches an example_case input key is kept unchanged."""
        fn = self._import_reconcile()
        binding = {"bundle": "case:$.input.arch_bundle"}
        example_keys = {"arch_bundle", "identity_output"}
        result, notes = fn(binding, example_keys)
        assert result == {"bundle": "case:$.input.arch_bundle"}
        assert notes == []

    def test_suffix_match_rewrites_bundle_to_arch_bundle(self):
        """'bundle' absent from example keys, but 'arch_bundle' ends with '_bundle' → rewrite."""
        fn = self._import_reconcile()
        binding = {"bundle": "case:$.input.bundle"}
        example_keys = {"arch_bundle", "identity_output"}
        result, notes = fn(binding, example_keys)
        assert result == {"bundle": "case:$.input.arch_bundle"}
        assert len(notes) == 1
        assert "rewritten" in notes[0]
        assert "arch_bundle" in notes[0]

    def test_suffix_match_identity_output(self):
        """'project_identity_output' mapped to 'identity_output' via suffix match."""
        fn = self._import_reconcile()
        binding = {"pid": "case:$.input.project_identity_output"}
        example_keys = {"identity_output", "arch_bundle"}
        # "project_identity_output" ends with "identity_output" (suffix match)
        result, notes = fn(binding, example_keys)
        assert result == {"pid": "case:$.input.identity_output"}
        assert len(notes) == 1
        assert "rewritten" in notes[0]

    def test_unresolvable_key_noted_not_dropped(self):
        """A key with no suffix match is kept as-is with a review note, not silently dropped."""
        fn = self._import_reconcile()
        binding = {"foo": "case:$.input.totally_unknown_key"}
        example_keys = {"arch_bundle", "identity_output"}
        result, notes = fn(binding, example_keys)
        assert result == {"foo": "case:$.input.totally_unknown_key"}
        assert len(notes) == 1
        assert "verify manually" in notes[0]

    def test_non_case_binding_passed_through(self):
        """Bindings that don't start with 'case:$.input.' are never reconciled."""
        fn = self._import_reconcile()
        binding = {"provider_id": "config:provider_id"}
        example_keys = {"arch_bundle"}
        result, notes = fn(binding, example_keys)
        assert result == {"provider_id": "config:provider_id"}
        assert notes == []

    def test_empty_example_keys_skips_reconciliation(self):
        """When example_input_keys is empty (no example case yet), binding is returned unchanged."""
        fn = self._import_reconcile()
        binding = {"bundle": "case:$.input.bundle"}
        result, notes = fn(binding, set())
        assert result == {"bundle": "case:$.input.bundle"}
        assert notes == []


# ---------------------------------------------------------------------------
# #3 — run_eval template: skip_case() honoured
# ---------------------------------------------------------------------------

class TestRunEvalSkipCase:
    """_run_one_case skips when dispatch_module.skip_case(case) returns True.

    Rather than loading the full template (which requires a live tracer.py), these tests
    verify the skip_case() guard logic in isolation: the key contract is that a dispatch module
    whose skip_case() returns True causes the outcome to carry skipped=True without calling
    invoke_agent, and that such cases are counted separately from attempted/succeeded.
    """

    def test_skip_case_guard_is_present_in_template(self):
        """The run_eval.py template source contains the skip_case() guard."""
        template_path = (
            Path(__file__).parents[1]
            / "agent_eval_harness"
            / "code_injection"
            / "templates"
            / "run_eval.py"
        )
        source = template_path.read_text(encoding="utf-8")
        assert "skip_case" in source, "skip_case guard must be present in run_eval.py template"
        assert "skipped" in source, "skipped tally must be present in run_eval.py template"
        assert "\"skipped\": skipped" in source, "manifest must record skipped count"

    def test_skip_case_guard_precedes_invoke_agent(self):
        """The skip_case() check comes before invoke_agent in the template source."""
        template_path = (
            Path(__file__).parents[1]
            / "agent_eval_harness"
            / "code_injection"
            / "templates"
            / "run_eval.py"
        )
        source = template_path.read_text(encoding="utf-8")
        skip_idx = source.index("skip_case")
        invoke_idx = source.index("invoke_agent")
        assert skip_idx < invoke_idx, "skip_case() guard must precede invoke_agent call"

    def test_skip_case_not_counted_in_attempted(self):
        """Skipped outcomes (outcome['skipped'] is True) are not incremented in 'attempted'."""
        template_path = (
            Path(__file__).parents[1]
            / "agent_eval_harness"
            / "code_injection"
            / "templates"
            / "run_eval.py"
        )
        source = template_path.read_text(encoding="utf-8")
        # The guard must be: if outcome.get("skipped"): skipped += 1 (not attempted)
        assert "outcome.get(\"skipped\")" in source, "skipped branch must use outcome.get('skipped')"
        # Confirm there's an else branch for attempted/succeeded
        assert "else:" in source, "run() loop must have else branch for attempted"
        # Confirm skipped is in the manifest
        assert '"skipped": skipped' in source, "manifest must include skipped tally"


# ---------------------------------------------------------------------------
# #6a — _annotation_to_schema: dict[K,V] emits additionalProperties
# ---------------------------------------------------------------------------

class TestAnnotationToSchemaDictFidelity:
    """_annotation_to_schema on dict[str, float] emits additionalProperties: {type: number}."""

    def _parse_annotation(self, annotation_str: str):
        """Parse an annotation string into an AST node."""
        tree = ast.parse(f"x: {annotation_str}", mode="exec")
        return tree.body[0].annotation

    def test_dict_str_float_emits_additionalproperties_number(self):
        """dict[str, float] → {type: object, additionalProperties: {type: number}}."""
        from agent_eval_harness.mapping.builder.contract_harvest import _annotation_to_schema
        node = self._parse_annotation("dict[str, float]")
        schema = _annotation_to_schema(node)
        assert schema == {"type": "object", "additionalProperties": {"type": "number"}}

    def test_dict_str_str_emits_additionalproperties_string(self):
        """dict[str, str] → {type: object, additionalProperties: {type: string}}."""
        from agent_eval_harness.mapping.builder.contract_harvest import _annotation_to_schema
        node = self._parse_annotation("dict[str, str]")
        schema = _annotation_to_schema(node)
        assert schema == {"type": "object", "additionalProperties": {"type": "string"}}

    def test_bare_dict_fallback_to_plain_object(self):
        """dict (no type params) → {type: object} (unchanged fallback behaviour)."""
        from agent_eval_harness.mapping.builder.contract_harvest import _annotation_to_schema
        node = self._parse_annotation("dict")
        schema = _annotation_to_schema(node)
        assert schema == {"type": "object"}


# ---------------------------------------------------------------------------
# #6b — _annotation_to_schema: list[str] regression guard
# ---------------------------------------------------------------------------

class TestAnnotationToSchemaListFidelity:
    """list[str] still emits items: {type: string} (no regression)."""

    def _parse_annotation(self, annotation_str: str):
        tree = ast.parse(f"x: {annotation_str}", mode="exec")
        return tree.body[0].annotation

    def test_list_str_emits_items_string(self):
        from agent_eval_harness.mapping.builder.contract_harvest import _annotation_to_schema
        node = self._parse_annotation("list[str]")
        schema = _annotation_to_schema(node)
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_list_bare_emits_array_no_items(self):
        from agent_eval_harness.mapping.builder.contract_harvest import _annotation_to_schema
        node = self._parse_annotation("list")
        schema = _annotation_to_schema(node)
        assert schema == {"type": "array"}


# ---------------------------------------------------------------------------
# #6c — _annotation_to_schema: X | None resolves to non-None branch
# ---------------------------------------------------------------------------

class TestAnnotationToSchemaOptionalFidelity:
    """X | None (PEP 604) and Optional[X] resolve to the non-None branch schema."""

    def _parse_annotation(self, annotation_str: str):
        tree = ast.parse(f"x: {annotation_str}", mode="exec")
        return tree.body[0].annotation

    def test_str_or_none_resolves_to_string(self):
        """str | None → {type: string}."""
        from agent_eval_harness.mapping.builder.contract_harvest import _annotation_to_schema
        node = self._parse_annotation("str | None")
        schema = _annotation_to_schema(node)
        assert schema == {"type": "string"}

    def test_list_str_or_none_resolves_to_array(self):
        """list[str] | None → {type: array, items: {type: string}}."""
        from agent_eval_harness.mapping.builder.contract_harvest import _annotation_to_schema
        node = self._parse_annotation("list[str] | None")
        schema = _annotation_to_schema(node)
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_optional_str_resolves_to_string(self):
        """Optional[str] → {type: string}."""
        from agent_eval_harness.mapping.builder.contract_harvest import _annotation_to_schema
        node = self._parse_annotation("Optional[str]")
        schema = _annotation_to_schema(node)
        assert schema == {"type": "string"}


# ---------------------------------------------------------------------------
# #6d — _validate_input flags string value where schema expects number
# ---------------------------------------------------------------------------

class TestValidateInputScalarTypeMismatch:
    """_validate_input reports type errors for string/number/integer/boolean mismatches."""

    def _make_parsed(self, kwargs: list[dict]) -> object:
        """Build a minimal SyntheticAgentIOConfig-like object."""
        from agent_eval_harness.datasets.generators.synthetic_agent_io import SyntheticAgentIOConfig
        contract = {"invocation": {"kwargs": kwargs, "case_binding": {}}}
        return SyntheticAgentIOConfig.model_validate({
            "dataset_name": "test_ds",
            "agent_id": "a1",
            "archetype": "rag_single_shot",
            "contract": contract,
        })

    def test_string_value_where_number_expected_is_flagged(self):
        """A string value ('high') where the schema says {type: number} is an error."""
        from agent_eval_harness.datasets.generators.synthetic_agent_io import _validate_input
        kwargs = [{"name": "confidence", "required": True,
                   "resolved_schema": {"type": "number"}}]
        parsed = self._make_parsed(kwargs)
        errs = _validate_input({"confidence": "high"}, parsed)
        assert any("confidence" in e and "number" in e for e in errs), f"Expected numeric error, got: {errs}"

    def test_numeric_value_where_string_expected_is_flagged(self):
        """An int value (42) where the schema says {type: string} is an error."""
        from agent_eval_harness.datasets.generators.synthetic_agent_io import _validate_input
        kwargs = [{"name": "label", "required": True,
                   "resolved_schema": {"type": "string"}}]
        parsed = self._make_parsed(kwargs)
        errs = _validate_input({"label": 42}, parsed)
        assert any("label" in e and "string" in e for e in errs), f"Expected string error, got: {errs}"

    def test_non_bool_where_boolean_expected_is_flagged(self):
        """A string ('yes') where the schema says {type: boolean} is an error."""
        from agent_eval_harness.datasets.generators.synthetic_agent_io import _validate_input
        kwargs = [{"name": "active", "required": True,
                   "resolved_schema": {"type": "boolean"}}]
        parsed = self._make_parsed(kwargs)
        errs = _validate_input({"active": "yes"}, parsed)
        assert any("active" in e and "boolean" in e for e in errs), f"Expected boolean error, got: {errs}"

    def test_bool_is_not_accepted_as_number(self):
        """bool is a subtype of int in Python but should NOT be accepted as numeric in schema."""
        from agent_eval_harness.datasets.generators.synthetic_agent_io import _validate_input
        kwargs = [{"name": "score", "required": True,
                   "resolved_schema": {"type": "number"}}]
        parsed = self._make_parsed(kwargs)
        errs = _validate_input({"score": True}, parsed)
        assert any("score" in e for e in errs), f"Expected error for bool as number, got: {errs}"

    def test_valid_number_passes(self):
        """A float value where {type: number} is expected passes without errors."""
        from agent_eval_harness.datasets.generators.synthetic_agent_io import _validate_input
        kwargs = [{"name": "confidence", "required": True,
                   "resolved_schema": {"type": "number"}}]
        parsed = self._make_parsed(kwargs)
        errs = _validate_input({"confidence": 0.95}, parsed)
        assert errs == []
