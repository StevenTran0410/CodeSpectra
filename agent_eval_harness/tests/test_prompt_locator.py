from __future__ import annotations

from pathlib import Path

from agent_eval_harness.code_injection.prompt_locator import locate_prompt_reference


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_finds_relative_import_from_sibling_prompts_module(tmp_path: Path) -> None:
    _write(tmp_path / "backend/domain/analysis/prompts.py", "AGENT_A_SYSTEM = 'you are...'\n")
    _write(
        tmp_path / "backend/domain/analysis/agents/agent_project_identity.py",
        "from ..prompts import AGENT_A_SCHEMA_STR, AGENT_A_SYSTEM, render_bundle\n\n"
        "def run(): pass\n",
    )

    result = locate_prompt_reference(
        "backend/domain/analysis/agents/agent_project_identity.py", tmp_path
    )

    assert result == "backend/domain/analysis/prompts.py (imported as AGENT_A_SCHEMA_STR, AGENT_A_SYSTEM, render_bundle)"


def test_returns_none_when_no_prompt_shaped_import_exists(tmp_path: Path) -> None:
    _write(
        tmp_path / "backend/domain/analysis/agents/agent_x.py",
        "from ..retrieval import RetrievalService\nimport os\n",
    )

    result = locate_prompt_reference("backend/domain/analysis/agents/agent_x.py", tmp_path)

    assert result is None


def test_absolute_prompt_import_reports_module_name_unresolved(tmp_path: Path) -> None:
    _write(
        tmp_path / "backend/domain/analysis/agents/agent_x.py",
        "from some_prompt_registry import SYSTEM_PROMPT\n",
    )

    result = locate_prompt_reference("backend/domain/analysis/agents/agent_x.py", tmp_path)

    assert result is not None
    assert "some_prompt_registry" in result
    assert "could not resolve" in result


def test_returns_none_when_component_file_missing(tmp_path: Path) -> None:
    result = locate_prompt_reference("backend/does/not/exist.py", tmp_path)

    assert result is None


def test_returns_none_when_component_file_is_blank(tmp_path: Path) -> None:
    result = locate_prompt_reference("", tmp_path)

    assert result is None
