"""AST-based scanner for prompt sites: module constants, SDK calls, and prompt imports."""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_eval_harness.discovery.prompt_resolver import (
    build_module_constants,
    resolve_constant,
    resolve_import_site,
)

logger = logging.getLogger("agent_eval_harness.discovery.prompt_site_scan")

_SDK_CALL_SNIPPET_CHARS = 120


@dataclass
class PromptSite:
    """A location where a prompt or prompt-related string is found."""
    file: str
    line: int
    kind: Literal['module_assignment', 'sdk_call', 'prompt_import']
    snippet: str


def scan_for_prompt_sites(repo_root: Path, accepted_files: list[str]) -> dict[str, list[PromptSite]]:
    """Scan accepted files for prompt sites: constants, SDK calls, prompt imports.

    Returns dict keyed by file path (relative to repo_root) for O(1) per-agent lookup.

    Three detectors:
    1. module_assignment: ast.Assign/AugAssign where target matches *_SYSTEM|*PROMPT*|*TEMPLATE*
    2. sdk_call: ast.Call where func matches LLM SDK patterns and first arg is string literal
    3. prompt_import: ast.ImportFrom where an IMPORTED ALIAS (not the module name) matches
       *_SYSTEM|*PROMPT*|*TEMPLATE* — no module-name assumption, the constant can live in any
       file (e.g. a target that doesn't call its prompts module `prompts.py`)
    """
    result: dict[str, list[PromptSite]] = {}

    for file_rel in accepted_files:
        file_path = repo_root / file_rel
        if not file_path.is_file():
            continue

        try:
            source_code = file_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Could not read {file_rel}: {e}")
            continue

        sites = []

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.warning(f"SyntaxError in {file_rel}: {e}")
            continue

        consts = build_module_constants(tree)

        for node in ast.walk(tree):
            # Detector 1: Module-level prompt constants
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    target_name = _get_name(target)
                    if target_name and _matches_prompt_pattern(target_name):
                        snippet = _extract_snippet(node, consts)
                        sites.append(PromptSite(
                            file=file_rel,
                            line=node.lineno,
                            kind='module_assignment',
                            snippet=snippet
                        ))

            # Detector 2: LLM SDK calls with string literal prompts
            elif isinstance(node, ast.Call):
                func_name = _get_call_func_name(node)
                if func_name and _matches_sdk_pattern(func_name):
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        snippet = node.args[0].value[:_SDK_CALL_SNIPPET_CHARS]
                        sites.append(PromptSite(
                            file=file_rel,
                            line=node.lineno,
                            kind='sdk_call',
                            snippet=snippet
                        ))

            # Detector 3: Prompt imports — one PromptSite per matching alias, snippet
            # resolved from the target module.
            elif isinstance(node, ast.ImportFrom):
                prompt_aliases = [
                    alias for alias in node.names
                    if _matches_prompt_pattern(alias.asname or alias.name)
                ]
                if prompt_aliases:
                    resolved = resolve_import_site(node, file_path, repo_root)
                    for alias in prompt_aliases:
                        local_name = alias.asname or alias.name
                        sites.append(PromptSite(
                            file=file_rel,
                            line=node.lineno,
                            kind='prompt_import',
                            snippet=resolved.get(local_name) or ''
                        ))

        result[file_rel] = sites

    return result


def _get_name(node: ast.expr) -> str | None:
    """Extract name from ast node (Name, Attribute, or Subscript)."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return node.attr
    elif isinstance(node, ast.Tuple) or isinstance(node, ast.List):
        return None
    return None


def _matches_prompt_pattern(name: str) -> bool:
    """Check if name matches *_SYSTEM, *PROMPT*, or *TEMPLATE* patterns (case-insensitive)."""
    upper = name.upper()
    return (
        upper.endswith('_SYSTEM') or
        'PROMPT' in upper or
        'TEMPLATE' in upper
    )


def _get_call_func_name(call: ast.Call) -> str | None:
    """Extract function name from a Call node."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    elif isinstance(call.func, ast.Attribute):
        # e.g., client.messages.create -> "create"
        return call.func.attr
    return None


def _matches_sdk_pattern(func_name: str) -> bool:
    """Check if function name matches known LLM SDK patterns."""
    patterns = ['ChatCompletion', 'messages', 'create', 'complete', 'chat']
    return any(p.lower() in func_name.lower() for p in patterns)


def _extract_snippet(node: ast.Assign | ast.AugAssign, consts: dict[str, ast.expr]) -> str:
    """Resolve the assigned/augmented value via the shared resolver. A module-level name is
    looked up in `consts` (already folded across any AugAssign, in source order); anything else
    falls back to resolving this node's own right-hand side directly."""
    target = node.targets[0] if isinstance(node, ast.Assign) else node.target
    name = target.id if isinstance(target, ast.Name) else None
    if name is not None and name in consts:
        value = resolve_constant(consts[name], consts)
        if value is not None:
            return value
    value = resolve_constant(node.value, consts)
    return value if value is not None else ''
