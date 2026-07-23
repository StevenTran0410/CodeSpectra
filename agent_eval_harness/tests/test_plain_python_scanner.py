"""PlainPythonScanner — signal-based detection, proven on the synthetic fixture, the real ask-mode
agent (dogfood), the precision floor (Haystack fixtures), and the wiring_block llm_fallback path."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval_harness.discovery.wiring import WiringBlock, WiringNode
from agent_eval_harness.mapping.builder.plain_python_scanner import PlainPythonScanner
from agent_eval_harness.mapping.builder.scanners import scan_all

_FIXTURE = Path(__file__).parent.parent / "test_targets" / "plain_agent" / "plain_agent.py"
_TARGETS = Path(__file__).parent.parent / "test_targets"
# Real ask-mode agent (provider-abstraction shape) — the one real plain-python target we have.
_QA_AGENT = Path(
    r"D:/Program Files (x86)/Python VS Code/CodeSpectra/backend/domain/qa/agent.py"
)


def _by_name(cands):
    return {c.class_name: c for c in cands}


def test_provider_abstraction_agent_detected():
    """A `*Agent` class corroborated by an entry method is emitted, even with NO raw SDK call."""
    cands = _by_name(PlainPythonScanner().scan([_FIXTURE]))
    assert "ResearchAgent" in cands
    assert cands["ResearchAgent"].entry_kind == "class"


def test_bare_agent_class_without_corroborator_rejected():
    """Precision guard: a `*Agent` class with no entry method / LLM call / prompt ref is NOT emitted."""
    cands = _by_name(PlainPythonScanner().scan([_FIXTURE]))
    assert "DisabledAgent" not in cands


def test_raw_sdk_tool_list_secondary_heuristic():
    """SECONDARY: a tools=[...] list of module-level functions surfaces tool components."""
    cands = _by_name(PlainPythonScanner().scan([_FIXTURE]))
    for tool in ("tool_search", "tool_summarize"):
        assert tool in cands
        assert cands[tool].is_tool is True
        assert cands[tool].owner_class_name == "ToolAgent"


def test_wiring_block_llm_fallback_path():
    """The wiring_block(llm_fallback) escape hatch resolves names the AST heuristic missed to real
    defs, and dedups against AST hits (no doubles). Exercised through scan(..., wiring_block=)."""
    wb = WiringBlock(
        nodes=[
            WiringNode(alias="a", callee_name="ResearchAgent", source_hint_file="plain_agent.py"),
            WiringNode(alias="s", callee_name="tool_search", source_hint_file="plain_agent.py"),
        ],
        edges=[],
        framework="llm_inferred",
        source="llm_fallback",
    )
    cands = PlainPythonScanner().scan([_FIXTURE], wiring_block=wb)
    names = [c.class_name for c in cands]
    assert names.count("ResearchAgent") == 1  # deduped against the AST-heuristic hit
    assert "tool_search" in names


def test_precision_floor_no_new_candidates_on_haystack_fixtures():
    """Registering PlainPythonScanner into scan_all must not add candidates to the Haystack fixtures."""
    expected = {"multi_agent": 7, "linear_rag": 2, "t3_reranker": 3}
    for tgt, count in expected.items():
        files = list((_TARGETS / tgt).glob("**/*.py"))
        cands, _label = scan_all(files)
        assert len(cands) == count, f"{tgt}: expected {count}, got {len(cands)}"


@pytest.mark.skipif(not _QA_AGENT.exists(), reason="backend QA agent not present")
def test_dogfood_real_ask_mode_agent():
    """Dogfood on the REAL ask-mode agent: provider-abstraction shape (self._call_stream via a base
    class), entry `run`, no raw SDK / no tools list. The scanner must still surface its class."""
    cands = _by_name(PlainPythonScanner().scan([_QA_AGENT]))
    agent_classes = [name for name, c in cands.items() if c.entry_kind == "class" and name.endswith("Agent")]
    assert agent_classes, f"no *Agent class surfaced from the real ask-mode agent; got {list(cands)}"
