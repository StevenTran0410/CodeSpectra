"""Pass 4: Mine constraints from source code (Phase A) and LLM prompts (Phase B)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.mapping.system_map import Constraint

from .types import CandidateComponent


def mine_constraints(
    source_files: list[Path],
    candidates: list[CandidateComponent],
    repo_root: Path | None = None,
) -> dict[str, list[Constraint]]:
    """Phase A (sync): Mine static constraints from class attributes."""
    constraints: dict[str, list[Constraint]] = {}

    if repo_root is None:
        repo_root = Path(".")

    for file in source_files:
        try:
            source = file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file))
        except SyntaxError:
            continue

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_name = node.name

                for item in node.body:
                    if isinstance(item, ast.Assign):
                        # Check if target is a simple Name with UPPER_CASE pattern
                        if len(item.targets) == 1:
                            target = item.targets[0]
                            if isinstance(target, ast.Name):
                                name = target.id
                                letters_only = "".join(ch for ch in name if ch.isalpha())
                                if name[0].isupper() and letters_only.isupper():
                                    if isinstance(item.value, ast.Constant) and isinstance(
                                        item.value.value, int
                                    ):
                                        # Found a constraint
                                        try:
                                            rel_path = file.relative_to(repo_root)
                                        except ValueError:
                                            rel_path = file

                                        citation = f"{rel_path.as_posix()}:{class_name}.{name}"
                                        constraint = Constraint(
                                            name=name.lower(),
                                            value=item.value.value,
                                            source=citation,
                                        )

                                        # Attach to all candidates from this class
                                        for candidate in candidates:
                                            if candidate.class_name == class_name:
                                                if candidate.candidate_id not in constraints:
                                                    constraints[candidate.candidate_id] = []
                                                constraints[candidate.candidate_id].append(
                                                    constraint
                                                )

    return constraints


async def mine_constraints_llm_phase(
    candidates: list[CandidateComponent],
    llm_client: LLMClient,
    phase_a_results: dict[str, list[Constraint]],
) -> dict[str, list[Constraint]]:
    """Phase B (async): Mine constraints from LLM prompts using keyword pre-filter."""
    from .prompts import CONSTRAINT_EXTRACTION_SYSTEM

    results = dict(phase_a_results)  # Start with Phase A results

    # Pre-filter keywords
    KEYWORDS = ("max", "retry", "retries", "at most", "fan-out", "limit")

    # Cache ASTs by file to avoid re-parsing the same file for each candidate
    ast_cache: dict[Path, ast.Module | None] = {}

    for candidate in candidates:
        # Skip tool candidates
        if candidate.is_tool:
            continue

        # Collect string literals from methods
        literals = []
        for hint in candidate.manual_span_hints:
            # Get or parse the candidate's file, using cache to avoid re-parsing
            if candidate.file not in ast_cache:
                try:
                    source = candidate.file.read_text(encoding="utf-8")
                    ast_cache[candidate.file] = ast.parse(source, filename=str(candidate.file))
                except SyntaxError:
                    ast_cache[candidate.file] = None
                    continue

            tree = ast_cache[candidate.file]
            if tree is None:
                continue

            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == candidate.class_name:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            for child in ast.walk(item):
                                if isinstance(child, ast.Constant) and isinstance(
                                    child.value, str
                                ):
                                    if len(child.value) > 30:
                                        literals.append(child.value)

        # Pre-filter by keywords
        filtered_literals = []
        for literal in literals:
            if any(
                keyword.lower() in literal.lower() for keyword in KEYWORDS
            ):
                filtered_literals.append(literal)

        if not filtered_literals:
            continue

        # Call LLM
        user_prompt = "Extract constraints from these literals:\n\n" + "\n---\n".join(
            filtered_literals
        )

        response = await llm_client.complete(
            [
                LLMMessage(role="system", content=CONSTRAINT_EXTRACTION_SYSTEM),
                LLMMessage(role="user", content=user_prompt),
            ],
            json_mode=True,
        )

        try:
            parsed = json.loads(response.content)
            if not isinstance(parsed, list):
                continue

            for item in parsed:
                if not isinstance(item, dict):
                    continue

                name = item.get("name", "")
                value = item.get("value")
                quote = item.get("quote", "")

                # Validate: quote must not be empty and must be in one of the literals
                if not quote or not any(quote in literal for literal in filtered_literals):
                    continue

                # Accept the constraint
                if value is not None and isinstance(value, (int, float)):
                    quoted = quote[:60]
                    constraint = Constraint(
                        name=name,
                        value=int(value),
                        source=(
                            f"{candidate.file}:{candidate.class_name} "
                            f"(LLM-mined, quoted: \"{quoted}\")"
                        ),
                    )

                    if candidate.candidate_id not in results:
                        results[candidate.candidate_id] = []
                    results[candidate.candidate_id].append(constraint)
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    return results
