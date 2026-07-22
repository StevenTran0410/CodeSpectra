"""Tests for Slice 1: static virtual input harvest."""
import ast
from pathlib import Path

import pytest

from agent_eval_harness.mapping.builder.contract_harvest import (
    ConstructorDepBinding,
    DepCallSite,
    _harvest_constructor_dep_bindings,
    _harvest_dep_call_sites,
    _dep_role_for_annotation,
)
from agent_eval_harness.mapping.builder.types import parse_python_source


def test_harvest_constructor_dep_bindings():
    """Test extraction of param->attr bindings from __init__."""
    agents_file = Path(__file__).parent.parent / "test_targets" / "self_retrieve" / "agents.py"
    parsed = parse_python_source(agents_file)
    assert parsed is not None
    tree = parsed[1]

    # Find AnnotationTierAgent class
    cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AnnotationTierAgent":
            cls = node
            break

    assert cls is not None
    bindings = _harvest_constructor_dep_bindings(cls)

    assert len(bindings) > 0
    assert any(b.param == "retriever" for b in bindings)

    retriever_binding = next(b for b in bindings if b.param == "retriever")
    assert retriever_binding.attr == "_retriever"
    assert "SelfRetriever" in (retriever_binding.annotation or "")


def test_harvest_dep_call_sites():
    """Test detection of method calls on deps."""
    agents_file = Path(__file__).parent.parent / "test_targets" / "self_retrieve" / "agents.py"
    parsed = parse_python_source(agents_file)
    assert parsed is not None
    tree = parsed[1]

    cls = None
    entry = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AnnotationTierAgent":
            cls = node
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name == "run":
                    entry = item
            break

    assert cls is not None
    assert entry is not None

    bindings = _harvest_constructor_dep_bindings(cls)
    call_sites = _harvest_dep_call_sites(entry, bindings, tree, agents_file, {agents_file: tree}, None)

    assert len(call_sites) > 0
    assert any(cs.method == "search" for cs in call_sites)


def test_dep_role_for_annotation():
    """Test role keyword matching."""
    keywords_by_role = {
        "retrieval": ["RetrievalService"],
        "llm_provider": ["ProviderConfigService", "LLMClient"],
    }

    assert _dep_role_for_annotation("SelfRetriever", keywords_by_role) == "unknown"
    assert _dep_role_for_annotation("RetrievalService", keywords_by_role) == "retrieval"
    assert _dep_role_for_annotation("LLMClient", keywords_by_role) == "llm_provider"
    assert _dep_role_for_annotation(None, keywords_by_role) == "unknown"
