"""LCELScanner — pipe idiom (synthetic fixture) + factory idiom (real cloned LangChain app dogfood)."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval_harness.mapping.builder.lcel_scanner import LCELScanner

_FIXTURE = Path(__file__).parent.parent / "test_targets" / "lcel_chain" / "chain.py"
# Real LangChain conversational-RAG app cloned for the factory-idiom dogfood (external target).
_CLONE = Path(
    r"D:/Program Files (x86)/Python VS Code/test_repo/conversational-rag-chatbot"
)


def _by_name(cands):
    return {c.class_name: c for c in cands}


def test_pipe_idiom_library_objects_degrade_explicitly():
    """Every pure-library link degrades explicitly (is_library_object=True) instead of vanishing."""
    cands = _by_name(LCELScanner().scan([_FIXTURE]))
    for lib in ("ChatOpenAI", "StrOutputParser"):
        assert lib in cands, f"library link {lib} not surfaced"
        assert cands[lib].is_library_object is True


def test_pipe_idiom_runnable_lambda_unwraps_to_user_function():
    """RunnableLambda(postprocess) must surface the wrapped USER function as harvestable, not the wrapper."""
    cands = _by_name(LCELScanner().scan([_FIXTURE]))
    assert "postprocess" in cands
    assert cands["postprocess"].is_library_object is False
    assert cands["postprocess"].entry_kind == "function"
    assert "RunnableLambda" not in cands


def test_pipe_idiom_guard_rejects_type_and_flag_unions():
    """The PEP-604 / flag-enum shapes in the same file must not become candidates."""
    cands = _by_name(LCELScanner().scan([_FIXTURE]))
    assert "RetryPolicy" not in cands
    assert "backoff_strategy" not in cands
    assert "_verbose_levels" not in cands


@pytest.mark.skipif(not _CLONE.exists(), reason="cloned LangChain app not present on this machine")
def test_factory_idiom_dogfood_maps_real_chain_graph():
    """Dogfood on a REAL LangChain app: the entire chain is built with the factory-function idiom
    (zero `|`), so this proves the dominant production idiom is mapped, not a strawman."""
    files = list(_CLONE.glob("**/*.py"))
    cands = _by_name(LCELScanner().scan(files))

    expected = {"retriever", "history_aware_retriever", "question_answer_chain", "rag_chain"}
    found = expected & set(cands)
    # Report measured coverage (the acceptance is the real graph, not a fixture).
    assert found == expected, f"factory-idiom coverage {len(found)}/{len(expected)}: missing {expected - found}"
    for name in expected:
        assert cands[name].is_library_object is False
        assert cands[name].entry_kind == "function"
