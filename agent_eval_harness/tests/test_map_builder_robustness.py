"""Robustness and negative control tests for map builder."""
import json
import tempfile
from pathlib import Path

import pytest

from agent_eval_harness.llm.client import LLMResponse
from agent_eval_harness.mapping.builder.pipeline import SystemMapBuilder
from tests._stubs import KeyedFakeLLMClient


class TestRobustness:
    def test_stripped_t2_degrades_toward_unknown(self, target_root: Path):
        """Stripped docstrings should degrade confidence, resulting in 'unknown' roles."""
        import ast
        import asyncio

        async def run_test():
            t2_dir = target_root / "multi_agent"

            # Copy T2 to a temp dir and strip docstrings
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_t2 = Path(tmp_dir) / "multi_agent"
                tmp_t2.mkdir()

                # Copy all Python files and strip docstrings
                for py_file in t2_dir.glob("**/*.py"):
                    rel_path = py_file.relative_to(t2_dir)
                    dest_file = tmp_t2 / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)

                    source = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(source)

                    # Strip docstrings by setting them to empty strings
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            if (
                                node.body
                                and isinstance(node.body[0], ast.Expr)
                                and isinstance(node.body[0].value, ast.Constant)
                            ):
                                node.body[0].value.value = ""

                    dest_file.write_text(ast.unparse(tree), encoding="utf-8")

                default_response = LLMResponse(
                    content=json.dumps({
                        "role": "unknown",
                        "confidence": 0.5,
                        "reasoning": "Low confidence due to stripped docs."
                    }),
                    model="fake-gpt",
                )

                builder = SystemMapBuilder(KeyedFakeLLMClient({}, default=default_response))
                system_map, summary = await builder.build(tmp_t2)

                # All components should be classified as unknown (confidence < 0.7 threshold)
                for component in system_map.components:
                    assert component.role == "unknown", (
                        f"Component {component.id} should be 'unknown'"
                    )

        asyncio.run(run_test())

    def test_stripped_t2_no_confident_misclassification(self, target_root: Path):
        """Stripped T2 should not have confident wrong classifications."""
        import asyncio

        async def run_test():
            t2_dir = target_root / "multi_agent"

            # Use a default response with confidence below threshold
            default_response = LLMResponse(
                content=json.dumps({
                    "role": "tool",  # Intentionally wrong
                    "confidence": 0.5,
                    "reasoning": "Misguessed, but low confidence"
                }),
                model="fake-gpt",
            )

            builder = SystemMapBuilder(KeyedFakeLLMClient({}, default=default_response))
            system_map, summary = await builder.build(t2_dir)

            # Below-threshold confidence must degrade to "unknown", never keep the wrong guess
            for component in system_map.components:
                assert component.role == "unknown", (
                    f"Component {component.id} should be 'unknown', not a confident wrong guess"
                )

        asyncio.run(run_test())

    def test_negative_control_store_module_empty_map(self, target_root: Path):
        """Non-agentic code (store module) should produce empty map with special message."""
        import asyncio

        async def run_test():
            store_dir = target_root.parent / "agent_eval_harness" / "store"
            if not store_dir.exists():
                pytest.skip("Store module not found")

            # Use a dummy LLM client (should never be called)
            default_response = LLMResponse(
                content=json.dumps({"role": "unknown", "confidence": 0.0, "reasoning": ""}),
                model="fake-gpt",
            )

            builder = SystemMapBuilder(KeyedFakeLLMClient({}, default=default_response))
            system_map, summary = await builder.build(store_dir)

            # Should have no components
            assert len(system_map.components) == 0

            # Summary should contain the special message
            assert "no agentic components identified" in summary

        asyncio.run(run_test())
