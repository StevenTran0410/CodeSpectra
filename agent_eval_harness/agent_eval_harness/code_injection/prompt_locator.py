"""Best-effort static heuristic: point at where an agent's prompt likely lives, by scanning
its own file's imports for anything that looks prompt-related. Never guaranteed — when it
finds nothing, the caller should tell the coding agent to search manually instead of
claiming a location that wasn't actually found."""
from __future__ import annotations

import re
from pathlib import Path

_IMPORT_RE = re.compile(r"^\s*from\s+([.\w]+)\s+import\s+(.+?)\s*(?:#.*)?$", re.MULTILINE)
_PROMPT_HINT = re.compile(r"prompt", re.IGNORECASE)


def locate_prompt_reference(component_file: str, repo_root: Path) -> str | None:
    """component_file is repo-root-relative (e.g. "backend/domain/analysis/agents/agent_x.py").
    Returns a human-readable pointer, e.g. "backend/domain/analysis/prompts.py (imported as
    AGENT_A_SYSTEM)", or None if no prompt-shaped import was found in this file."""
    abs_path = repo_root / component_file
    if not component_file or not abs_path.is_file():
        return None
    try:
        text = abs_path.read_text(encoding="utf-8")
    except OSError:
        return None

    for match in _IMPORT_RE.finditer(text):
        module, names = match.group(1), match.group(2)
        if not _PROMPT_HINT.search(module):
            continue
        names_clean = ", ".join(n.strip().split(" as ")[0] for n in names.split(","))
        resolved = _resolve_relative_module(module, abs_path.parent, repo_root)
        if resolved is None:
            return f"module '{module}' (imported as {names_clean}) — could not resolve to a file, check import manually"
        rel = resolved.relative_to(repo_root) if _is_within(resolved, repo_root) else resolved
        return f"{rel.as_posix()} (imported as {names_clean})"
    return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_relative_module(module: str, file_dir: Path, repo_root: Path) -> Path | None:
    """Resolves a Python relative import ('.'/'..'/... prefix) to a .py file path. Absolute
    (non-relative) imports return None — this heuristic only follows an agent's own package,
    not third-party or top-level absolute imports."""
    dots = 0
    while dots < len(module) and module[dots] == ".":
        dots += 1
    if dots == 0:
        return None
    rest = module[dots:]
    base = file_dir
    for _ in range(dots - 1):
        base = base.parent
    candidate = base / (rest.replace(".", "/") + ".py") if rest else base / "__init__.py"
    return candidate if candidate.is_file() else None
