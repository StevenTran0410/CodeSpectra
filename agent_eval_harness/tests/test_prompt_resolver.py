"""CS-300 AC3: prompt_resolver generalizes — no module-name assumption, no name pattern."""
from __future__ import annotations

import ast
from pathlib import Path

from agent_eval_harness.discovery.prompt_resolver import (
    build_module_constants,
    resolve_constant,
    resolve_import_site,
)


def _consts(source: str) -> dict[str, ast.expr]:
    return build_module_constants(ast.parse(source))


class TestResolveConstant:
    def test_plain_constant_resolves_verbatim(self):
        consts = _consts("X_SYSTEM = 'literal'\n")
        assert resolve_constant(consts["X_SYSTEM"], consts) == "literal"

    def test_joinedstr_is_mandatory_not_optional(self):
        """THE MANDATORY FAILING TEST (Judge spec §9 AC3): a Constant-only resolver returns ''
        here, silently dropping ~46% of real prompts — the exact bug this ticket exists to fix."""
        consts = _consts(
            "ROLE = 'planner'\nWORK = 'plan'\nY_SYSTEM = f'You are {ROLE}. Do {WORK}.'\n"
        )
        value = resolve_constant(consts["Y_SYSTEM"], consts)
        assert value
        assert "You are" in value
        assert "{ROLE}" in value  # placeholder retained, not resolved through the f-string
        assert "{WORK}" in value

    def test_augassign_folds_into_consts(self):
        consts = _consts(
            'WRITER_PROMPT = "Write an answer."\nWRITER_PROMPT += " Cite every source."\n'
        )
        assert resolve_constant(consts["WRITER_PROMPT"], consts) == "Write an answer. Cite every source."

    def test_binop_add_concatenates_both_sides(self):
        consts = _consts("A_PROMPT = 'a' + 'b'\n")
        assert resolve_constant(consts["A_PROMPT"], consts) == "ab"

    def test_binop_add_with_unresolvable_side_is_none(self):
        consts = _consts("A_PROMPT = 'a' + get_suffix()\n")
        assert resolve_constant(consts["A_PROMPT"], consts) is None

    def test_name_resolves_through_another_module_constant(self):
        consts = _consts("_BASE = 'base text'\nB_SYSTEM = _BASE\n")
        assert resolve_constant(consts["B_SYSTEM"], consts) == "base text"

    def test_name_cycle_terminates(self):
        """A→B→A must not infinite-loop — the visited set is required, not decorative."""
        consts = _consts("A_PROMPT = B_PROMPT\nB_PROMPT = A_PROMPT\n")
        assert resolve_constant(consts["A_PROMPT"], consts) is None
        assert resolve_constant(consts["B_PROMPT"], consts) is None

    def test_call_never_guessed(self):
        consts = _consts("X_SYSTEM = build_prompt()\n")
        assert resolve_constant(consts["X_SYSTEM"], consts) is None

    def test_none_node_is_none(self):
        assert resolve_constant(None, {}) is None


class TestResolveImportSite:
    def test_absolute_and_relative_imports_resolve_one_entry_per_alias(self, tmp_path: Path):
        """Nguyên tắc số 0: the module is deliberately named 'agent_texts.py', NOT 'prompts.py' —
        the resolver must not assume where prompts live."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "agent_texts.py").write_text(
            "PLANNER_SYSTEM = 'You are a planning component. Decompose the query into intents.'\n"
            "_RUBRIC = 'Score 1-5.'\n"
            "JUDGE_SYSTEM = f\"You are a critical judge reviewing another component's output. {_RUBRIC}\"\n",
            encoding="utf-8",
        )
        abs_file = pkg / "agent_abs.py"
        abs_file.write_text("from pkg.agent_texts import PLANNER_SYSTEM, JUDGE_SYSTEM\n", encoding="utf-8")
        rel_file = pkg / "agent_rel.py"
        rel_file.write_text("from .agent_texts import PLANNER_SYSTEM\n", encoding="utf-8")

        abs_tree = ast.parse(abs_file.read_text(encoding="utf-8"))
        abs_import = next(n for n in ast.walk(abs_tree) if isinstance(n, ast.ImportFrom))
        abs_resolved = resolve_import_site(abs_import, abs_file, tmp_path)
        assert len(abs_resolved) == 2  # one entry PER ALIAS
        assert abs_resolved["PLANNER_SYSTEM"] == "You are a planning component. Decompose the query into intents."
        assert "critical judge reviewing another component's output" in (abs_resolved["JUDGE_SYSTEM"] or "")

        rel_tree = ast.parse(rel_file.read_text(encoding="utf-8"))
        rel_import = next(n for n in ast.walk(rel_tree) if isinstance(n, ast.ImportFrom))
        rel_resolved = resolve_import_site(rel_import, rel_file, tmp_path)
        assert rel_resolved["PLANNER_SYSTEM"] == "You are a planning component. Decompose the query into intents."

    def test_aliased_import_keyed_by_local_name(self, tmp_path: Path):
        pkg = tmp_path / "pkg2"
        pkg.mkdir()
        (pkg / "texts.py").write_text("FOO_SYSTEM = 'foo'\n", encoding="utf-8")
        importing = pkg / "agent.py"
        importing.write_text("from pkg2.texts import FOO_SYSTEM as BAR_SYSTEM\n", encoding="utf-8")

        tree = ast.parse(importing.read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
        resolved = resolve_import_site(node, importing, tmp_path)
        assert resolved == {"BAR_SYSTEM": "foo"}

    def test_unresolvable_third_party_import_degrades_to_none_no_exception(self, tmp_path: Path):
        importing_file = tmp_path / "agent.py"
        importing_file.write_text("from some_third_party_sdk import SYSTEM_PROMPT\n", encoding="utf-8")
        tree = ast.parse(importing_file.read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
        resolved = resolve_import_site(node, importing_file, tmp_path)
        assert resolved == {"SYSTEM_PROMPT": None}
