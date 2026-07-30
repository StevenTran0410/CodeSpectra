"""LLM residue pass: decides makes_model_call for a component whose closure resolved completely
enough to walk but stopped at a genuinely unresolved call with no boundary reached anywhere in it --
the one case static analysis leaves undecided (pipeline.py's _compute_model_call_verdict). Static
still owns the call closure; the LLM only judges, over the exact source that closure already
resolved, whether it reaches a model/embedding provider -- cited, verified against the bundle sent,
and cached by content hash so an unrelated commit or re-run spends zero tokens.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_eval_harness.discovery.wiring import CallSite, NodeCallTarget
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.mapping.system_map import Component

from .resolution_primitives import LLM_UNVERIFIED_DEFAULT, LLM_VERIFIED

logger = logging.getLogger("agent_eval_harness.mapping.builder.boundary_llm")

_PROMPT_VERSION = "1"
_BUNDLE_CHAR_BUDGET = 8000  # per residue component; deepest (closest to the seam) frames win when over budget
_RESIDUE_MAX_TOKENS = 2048

_RESIDUE_SYSTEM_PROMPT = (
    "You are given, per component, the complete source of every method reachable in its call "
    "closure that static analysis was able to resolve, plus a note for any call static analysis "
    "could not resolve further. Decide, using only that source, whether the component's own "
    "closure ultimately constructs and sends a request to an external language-model or "
    "embedding provider.\n"
    'Respond with a JSON object shaped exactly as {"verdicts": [...]}, where "verdicts" is an '
    "array with one entry per component you can decide with real evidence, each entry shaped "
    "exactly as:\n"
    '{"component_id": "...", "makes_model_call": true or false, '
    '"citation": {"file": "...", "line": 0}, "evidence_kind": "..."}\n'
    "citation.file and citation.line must be an EXACT file path and line number copied from the "
    "source you were given -- never invented -- and that line must itself be a call expression, "
    "the single strongest line of evidence for your answer either way. "
    "Omit a component entirely, rather than guessing, when its source gives no real evidence."
)


@dataclass
class ResidueCandidate:
    """One component eligible for the LLM residue pass: its own entry line, plus the closure's
    call sites and resolved targets, scoped exactly as pipeline.py's static tiers scoped them."""
    component_id: str
    rel_file: str
    entry_line: int
    call_sites: list[CallSite]
    targets: list[NodeCallTarget]


@dataclass
class _Snippet:
    file: str
    start_line: int
    end_line: int
    text: str


@dataclass
class _Bundle:
    component_id: str
    snippets: list[_Snippet] = field(default_factory=list)
    unresolved_notes: list[str] = field(default_factory=list)

    def contains(self, file: str, line: int) -> bool:
        return any(s.file == file and s.start_line <= line <= s.end_line for s in self.snippets)


def _read_snippet(
    package_root: Path, rel_file: str, lineno: int,
    file_cache: dict[str, tuple[str, ast.Module] | None],
) -> _Snippet | None:
    """The def/class node whose OWN line is `lineno`, full body, uncut -- never a caller's reference
    to it. `file_cache` is shared across one whole residue batch so a file shared by several
    components' closures is only read/parsed once."""
    if rel_file not in file_cache:
        try:
            text = (package_root / rel_file).read_text(encoding="utf-8")
            file_cache[rel_file] = (text, ast.parse(text, filename=rel_file))
        except (OSError, SyntaxError, UnicodeDecodeError):
            file_cache[rel_file] = None
    entry = file_cache[rel_file]
    if entry is None:
        return None
    text, tree = entry
    node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.lineno == lineno),
        None,
    )
    if node is None:
        return None
    end = getattr(node, "end_lineno", None) or node.lineno
    lines = text.splitlines()
    return _Snippet(file=rel_file, start_line=node.lineno, end_line=end,
                     text="\n".join(lines[node.lineno - 1:end]))


def _line_has_call(tree: ast.Module, lineno: int) -> bool:
    """True if `lineno` is itself a call, or lies inside a def/method whose body contains one --
    a citation landing on the signature of a call-dispatching helper (rather than its exact call
    line) still counts, so long as the enclosing function genuinely reaches a call somewhere."""
    enclosing = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.lineno <= lineno <= (getattr(n, "end_lineno", None) or n.lineno)
    ]
    if enclosing:
        innermost = min(enclosing, key=lambda n: (getattr(n, "end_lineno", None) or n.lineno) - n.lineno)
        return any(isinstance(m, ast.Call) for m in ast.walk(innermost))
    return any(
        isinstance(n, ast.Call) and n.lineno <= lineno <= (getattr(n, "end_lineno", None) or n.lineno)
        for n in ast.walk(tree)
    )


def _build_bundle(
    residue: ResidueCandidate, package_root: Path,
    file_cache: dict[str, tuple[str, ast.Module] | None],
) -> _Bundle:
    frames: list[tuple[int, str, int]] = []
    seen: set[tuple[str, int]] = set()

    def add(priority: int, file: str, line: int) -> None:
        key = (file, line)
        if key not in seen:
            seen.add(key)
            frames.append((priority, file, line))

    add(0, residue.rel_file, residue.entry_line)
    for i, target in enumerate(residue.targets, start=1):
        add(i, target.defining_file, target.line)
    frames.sort(key=lambda f: -f[0])

    unresolved_lines = {(cs.file, cs.lineno) for cs in residue.call_sites if cs.outcome == "unresolved"}

    def is_evidence(snippet: _Snippet) -> bool:
        return any(
            snippet.file == file and snippet.start_line <= line <= snippet.end_line
            for file, line in unresolved_lines
        )

    evidence: list[_Snippet] = []
    rest: list[_Snippet] = []
    for _, file, line in frames:
        snippet = _read_snippet(package_root, file, line, file_cache)
        if snippet is None:
            continue
        (evidence if is_evidence(snippet) else rest).append(snippet)

    snippets: list[_Snippet] = list(evidence)
    total = sum(len(s.text) for s in snippets)
    for snippet in rest:
        if snippets and total + len(snippet.text) > _BUNDLE_CHAR_BUDGET:
            continue
        snippets.append(snippet)
        total += len(snippet.text)

    unresolved_notes = sorted({
        f"{cs.file}:{cs.lineno} ({cs.callee_text})"
        for cs in residue.call_sites if cs.outcome == "unresolved"
    })
    return _Bundle(component_id=residue.component_id, snippets=snippets, unresolved_notes=unresolved_notes)


def _render_prompt(bundles: list[_Bundle]) -> str:
    parts = []
    for bundle in bundles:
        lines = [f"component_id: {bundle.component_id}"]
        for note in bundle.unresolved_notes:
            lines.append(f"# static analysis could not resolve further: {note}")
        for snippet in bundle.snippets:
            lines.append(f"# {snippet.file} lines {snippet.start_line}-{snippet.end_line}")
            lines.append(snippet.text)
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def _cache_key(bundle: _Bundle, boundary_hash: str, model_id: str) -> str:
    parts = sorted(f"{s.file}:{s.start_line}-{s.end_line}\n{s.text}" for s in bundle.snippets)
    digest_input = "\x1e".join(parts) + "\x1f" + boundary_hash + "\x1f" + model_id + "\x1f" + _PROMPT_VERSION
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


def _client_model_id(llm_client: LLMClient) -> str:
    return str(getattr(llm_client, "model_id", None) or getattr(llm_client, "_model_id", None) or "unspecified")


def _provider_boundary_hash() -> str:
    from agent_eval_harness.discovery.engine import provider_boundaries_content_hash
    return provider_boundaries_content_hash()


async def _cache_get(cache_key: str) -> dict[str, Any] | None:
    from agent_eval_harness.store import repository
    try:
        return await repository.get_model_call_verdict(cache_key)
    except RuntimeError:
        return None 


async def _cache_put(
    cache_key: str, makes_model_call: bool, source: str, citation: str, evidence_kind: str,
) -> None:
    from agent_eval_harness.store import repository
    try:
        await repository.upsert_model_call_verdict(cache_key, makes_model_call, source, citation, evidence_kind)
    except RuntimeError:
        pass


def _extract_verdict(
    entry: Any, bundle: _Bundle, file_cache: dict[str, tuple[str, ast.Module] | None],
) -> tuple[bool, str, str] | None:
    """Validate one LLM-returned entry against the bundle actually sent for its component: the
    citation must be inside that bundle, and its enclosing def/method must itself reach a call
    somewhere in its body (the LLM may cite a call-dispatching helper's signature rather than its
    exact call line). Any failure -> None, so the caller leaves the component undecided rather
    than promoting it."""
    if not isinstance(entry, dict):
        return None
    makes_model_call = entry.get("makes_model_call")
    citation = entry.get("citation")
    evidence_kind = entry.get("evidence_kind")
    if not isinstance(makes_model_call, bool) or not isinstance(citation, dict) or not evidence_kind:
        return None
    file = citation.get("file")
    line = citation.get("line")
    if not isinstance(file, str) or not isinstance(line, int):
        return None
    if not bundle.contains(file, line):
        return None
    cached_file = file_cache.get(file)
    if cached_file is None or not _line_has_call(cached_file[1], line):
        return None
    return makes_model_call, f"{file}:{line}", str(evidence_kind)


def _unwrap_verdicts(parsed: Any) -> list[Any]:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                return value
    return []


async def decide_residue_verdicts(
    residue: list[ResidueCandidate], components: list[Component], package_root: Path,
    llm_client: LLMClient, target_system_id: str,
) -> None:
    """Mutates `components` in place for every residue candidate: a cache hit, a verified LLM
    citation, or -- when neither decides it -- a defaulted `makes_model_call = False` (never True,
    so an unresolved seam is never promoted to an agent). Every outcome, including a default, is
    written to the boundary debug dump only, never surfaced as a SystemMap discrepancy -- that
    list is shared with genuine doc-vs-code reconciliation and must stay free of this pass's noise.
    A no-op, with zero LLM calls, when `residue` is empty -- the two guardrail fixtures never reach
    this branch."""
    if not residue:
        return
    by_id = {c.id: c for c in components}
    file_cache: dict[str, tuple[str, ast.Module] | None] = {}
    boundary_hash = _provider_boundary_hash()
    model_id = _client_model_id(llm_client)

    bundles: dict[str, _Bundle] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    to_call: list[tuple[ResidueCandidate, _Bundle, str]] = []
    for r in residue:
        bundle = _build_bundle(r, package_root, file_cache)
        bundles[r.component_id] = bundle
        cache_key = _cache_key(bundle, boundary_hash, model_id)
        cached = await _cache_get(cache_key)
        if cached is None:
            to_call.append((r, bundle, cache_key))
            continue
        component = by_id[r.component_id]
        component.makes_model_call = bool(cached["makes_model_call"])
        component.model_call_source = cached["source"]
        component.model_call_citation = cached["citation"]
        outcomes[r.component_id] = {"outcome": "cache_hit", **cached}

    system_prompt = ""
    user_prompt = ""
    raw_response = ""
    if to_call:
        system_prompt = _RESIDUE_SYSTEM_PROMPT
        user_prompt = _render_prompt([b for _, b, _ in to_call])
        response = await llm_client.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            max_tokens=_RESIDUE_MAX_TOKENS, temperature=0.0, json_mode=True,
        )
        raw_response = response.content
        try:
            parsed = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        parsed_list = _unwrap_verdicts(parsed)
        parsed_by_id: dict[str, Any] = {}
        for entry in parsed_list:
            if isinstance(entry, dict) and isinstance(entry.get("component_id"), str):
                parsed_by_id.setdefault(entry["component_id"], entry)

        for r, bundle, cache_key in to_call:
            component = by_id[r.component_id]
            result = _extract_verdict(parsed_by_id.get(r.component_id), bundle, file_cache)
            if result is None:
                component.makes_model_call = False
                component.model_call_source = LLM_UNVERIFIED_DEFAULT
                component.model_call_citation = None
                outcomes[r.component_id] = {
                    "outcome": "defaulted_to_step",
                    "reason": "llm residue pass returned no verifiable verdict",
                }
                continue
            makes_model_call, citation, evidence_kind = result
            component.makes_model_call = makes_model_call
            component.model_call_source = LLM_VERIFIED
            component.model_call_citation = citation
            outcomes[r.component_id] = {
                "outcome": "llm_verified", "makes_model_call": makes_model_call,
                "citation": citation, "evidence_kind": evidence_kind,
            }
            await _cache_put(cache_key, makes_model_call, LLM_VERIFIED, citation, evidence_kind)
