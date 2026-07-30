"""Tests for Slice 1 fixture compatibility: old fixtures remain unchanged."""
import ast
from pathlib import Path

import pytest

from agent_eval_harness.mapping.builder.contract_harvest import (
    _harvest_constructor_dep_bindings,
)
from agent_eval_harness.mapping.builder.types import parse_python_source


def test_linear_rag_retriever_has_no_virtual_inputs():
    """Verify linear_rag RetrieverComponent (pure-kwarg) has no virtual inputs."""
    components_file = Path(__file__).parent.parent / "test_targets" / "linear_rag" / "components.py"
    parsed = parse_python_source(components_file)
    assert parsed is not None
    tree = parsed[1]

    cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RetrieverComponent":
            cls = node
            break

    assert cls is not None
    bindings = _harvest_constructor_dep_bindings(cls)

    # linear_rag RetrieverComponent takes corpus and top_k as params, NOT retrieval dep
    assert not any(
        "retriev" in (b.annotation or "").lower() for b in bindings
    ), f"linear_rag should have no retrieval deps, but got: {bindings}"


def test_self_retrieve_annotation_tier_has_retrieval_binding():
    """Verify self_retrieve fixture's annotation-tier agent has retrieval binding."""
    agents_file = Path(__file__).parent.parent / "test_targets" / "self_retrieve" / "agents.py"
    parsed = parse_python_source(agents_file)
    assert parsed is not None
    tree = parsed[1]

    cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AnnotationTierAgent":
            cls = node
            break

    assert cls is not None
    bindings = _harvest_constructor_dep_bindings(cls)

    assert len(bindings) > 0
    assert any(b.param == "retriever" for b in bindings)


def test_self_retrieve_usage_tier_has_retrieval_binding():
    """Verify self_retrieve fixture's usage-tier agent has retrieval binding."""
    agents_file = Path(__file__).parent.parent / "test_targets" / "self_retrieve" / "agents.py"
    parsed = parse_python_source(agents_file)
    assert parsed is not None
    tree = parsed[1]

    cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "UsageTierAgent":
            cls = node
            break

    assert cls is not None
    bindings = _harvest_constructor_dep_bindings(cls)

    assert len(bindings) > 0
    assert any(b.param == "retriever" for b in bindings)
