"""CS-300 B4: both prompt_site_scan.py snippet bugs must die, or AC3 is theatre."""
from __future__ import annotations

from pathlib import Path

from agent_eval_harness.discovery.prompt_site_scan import scan_for_prompt_sites


class TestModuleAssignmentSnippets:
    """Bug 2: _extract_snippet only unwrapped Assign+Constant; JoinedStr and AugAssign both
    silently returned ''."""

    def test_augassign_snippet_is_resolved_not_empty(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(
            'WRITER_PROMPT = "Write an answer."\nWRITER_PROMPT += " Cite every source."\n',
            encoding="utf-8",
        )
        sites = scan_for_prompt_sites(tmp_path, ["agent.py"])["agent.py"]
        augassign_site = next(s for s in sites if s.line == 2)
        assert augassign_site.snippet == "Write an answer. Cite every source."

    def test_joinedstr_snippet_is_resolved_not_empty(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(
            "ROLE = 'planner'\nY_SYSTEM = f'You are {ROLE}.'\n", encoding="utf-8"
        )
        sites = scan_for_prompt_sites(tmp_path, ["agent.py"])["agent.py"]
        site = next(s for s in sites if s.kind == "module_assignment")
        assert "You are" in site.snippet
        assert "{ROLE}" in site.snippet


class TestPromptImportSites:
    """Bug 1: the prompt_import branch hardcoded snippet='' unconditionally, and gated on the
    MODULE name containing 'prompt'/'system' — a real assumption forbidden by Nguyên tắc số 0.
    Fixed to gate on the imported ALIAS name (generic) and resolve real text."""

    def test_import_from_a_module_not_named_prompts_still_resolves(self, tmp_path: Path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "agent_texts.py").write_text(
            "AGENT_K_SYSTEM = 'You are a critical auditor reviewing the outputs of 10 code "
            "analysis agents (sections A-J).'\n",
            encoding="utf-8",
        )
        (pkg / "agent.py").write_text(
            "from pkg.agent_texts import AGENT_K_SYSTEM\n", encoding="utf-8"
        )

        sites = scan_for_prompt_sites(tmp_path, ["pkg/agent.py"])["pkg/agent.py"]
        import_sites = [s for s in sites if s.kind == "prompt_import"]
        assert len(import_sites) == 1
        assert "critical auditor reviewing the outputs of 10 code analysis agents" in import_sites[0].snippet

    def test_one_site_per_matching_alias_non_matching_alias_ignored(self, tmp_path: Path):
        pkg = tmp_path / "pkg2"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "helpers.py").write_text(
            "PLANNER_SYSTEM = 'planner prompt'\nhelper_function = None\n", encoding="utf-8"
        )
        (pkg / "agent.py").write_text(
            "from pkg2.helpers import PLANNER_SYSTEM, helper_function\n", encoding="utf-8"
        )

        sites = scan_for_prompt_sites(tmp_path, ["pkg2/agent.py"])["pkg2/agent.py"]
        import_sites = [s for s in sites if s.kind == "prompt_import"]
        assert len(import_sites) == 1
        assert import_sites[0].snippet == "planner prompt"

    def test_relative_import_resolves(self, tmp_path: Path):
        pkg = tmp_path / "pkg3"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "agent_texts.py").write_text("JUDGE_SYSTEM = 'judge prompt'\n", encoding="utf-8")
        (pkg / "agent.py").write_text("from .agent_texts import JUDGE_SYSTEM\n", encoding="utf-8")

        sites = scan_for_prompt_sites(tmp_path, ["pkg3/agent.py"])["pkg3/agent.py"]
        import_sites = [s for s in sites if s.kind == "prompt_import"]
        assert len(import_sites) == 1
        assert import_sites[0].snippet == "judge prompt"

    def test_unresolvable_import_degrades_to_empty_snippet_not_exception(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(
            "from some_third_party_sdk import SYSTEM_PROMPT\n", encoding="utf-8"
        )
        sites = scan_for_prompt_sites(tmp_path, ["agent.py"])["agent.py"]
        import_sites = [s for s in sites if s.kind == "prompt_import"]
        assert len(import_sites) == 1
        assert import_sites[0].snippet == ""
