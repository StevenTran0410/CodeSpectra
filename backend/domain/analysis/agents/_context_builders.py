"""Typed context assembly helpers shared across section agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from domain.retrieval.service import RetrievalService
from domain.retrieval.types import RetrievalBundle, RetrievalMode, RetrievalSection, RetrieveRequest
from infrastructure.db.database import get_db

from ..static_convention import ConventionReport
from ..static_risk import RiskReport

_MANIFEST_PATTERNS = (
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "pom.xml",
    "build.gradle",
    "composer.json",
    "Gemfile",
    "Package.swift",
)

_DOC_PATTERNS = (
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "ARCHITECTURE.md",
)


async def _fetch_files_by_pattern(
    snapshot_id: str,
    patterns: tuple[str, ...],
    char_limit: int,
    max_rows: int = 6,
) -> str:
    """Fetch file content from retrieval_chunks matching any of the given filename patterns."""
    db = get_db()
    conditions = " OR ".join("rc.rel_path LIKE ?" for _ in patterns)
    params: list[Any] = [snapshot_id] + [f"%{p}" for p in patterns]
    rows = []
    try:
        async with db.execute(
            f"""SELECT rc.rel_path, rc.content
                FROM retrieval_chunks rc
                WHERE rc.snapshot_id=? AND ({conditions})
                ORDER BY LENGTH(rc.rel_path) ASC
                LIMIT {max_rows}""",
            params,
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        pass
    if not rows:
        return ""
    merged: dict[str, list[str]] = {}
    for row in rows:
        path = row["rel_path"]
        merged.setdefault(path, []).append(row["content"] or "")
    parts = []
    for path, chunks in merged.items():
        full = "\n".join(chunks)
        if char_limit and len(full) > char_limit:
            full = full[:char_limit] + "\n...(truncated)"
        parts.append(f"=== {path} ===\n{full}")
    return "\n\n".join(parts)


@dataclass
class PipelineMemoryContext:
    arch_bundle: RetrievalBundle
    folder_tree: str
    doc_files: str
    manifest_files: str


async def prefetch_pipeline_context(
    retrieval: RetrievalService,
    snapshot_id: str,
    mode: RetrievalMode = RetrievalMode.HYBRID,
) -> PipelineMemoryContext:
    """Pre-fetch shared retrieval data before pipeline construction.

    Runs 4 DB/retrieval tasks in parallel. arch_bundle is shared by B and C.
    folder_tree, doc_files, manifest_files are used by A (and C for folder_tree).
    D and E are NOT pre-fetched — they have semantically orthogonal CONVENTIONS queries.
    """
    arch_task = retrieval.retrieve(
        RetrieveRequest(
            snapshot_id=snapshot_id,
            query=(
                "framework entrypoint bootstrap layer service router handler middleware "
                "folder module package structure boundary domain"
            ),
            section=RetrievalSection.ARCHITECTURE,
            mode=mode,
            max_results=30,
        )
    )
    tree_task = fetch_folder_tree(snapshot_id)
    doc_task = _fetch_files_by_pattern(snapshot_id, _DOC_PATTERNS, char_limit=0, max_rows=4)
    manifest_task = _fetch_files_by_pattern(snapshot_id, _MANIFEST_PATTERNS, char_limit=3000)
    arch, tree, docs, manifests = await asyncio.gather(
        arch_task, tree_task, doc_task, manifest_task
    )
    return PipelineMemoryContext(
        arch_bundle=arch,
        folder_tree=tree,
        doc_files=docs,
        manifest_files=manifests,
    )


def extract_a_identity_context(a_output: dict[str, Any] | None) -> str:
    """Format A's identity fields as a context block for B, C, and F."""
    if not a_output:
        return ""
    tech_stack = a_output.get("tech_stack") or []
    tech_str = ", ".join(str(t) for t in tech_stack) if isinstance(tech_stack, list) else ""
    domain = str(a_output.get("domain", "") or "")
    runtime_type = str(a_output.get("runtime_type", "") or "")
    if not any([tech_str, domain, runtime_type]):
        return ""
    lines = ["Project identity context (from ProjectIdentityAgent):"]
    if tech_str:
        lines.append(f"  Tech stack: {tech_str}")
    if domain:
        lines.append(f"  Domain: {domain}")
    if runtime_type:
        lines.append(f"  Runtime type: {runtime_type}")
    return "\n".join(lines)


def extract_b_arch_context(b_output: dict[str, Any] | None) -> str:
    """Format B's architecture fields as a context block for F."""
    if not b_output:
        return ""
    layers = b_output.get("main_layers") or []
    services = b_output.get("main_services") or []
    layers_str = ", ".join(str(layer) for layer in layers) if isinstance(layers, list) else ""
    if not layers_str and not services:
        return ""
    lines = ["Architecture context (from ArchitectureAgent — do not re-list these services):"]
    if layers_str:
        lines.append(f"  Layers: {layers_str}")
    if isinstance(services, list) and services:
        lines.append("  Known services:")
        for svc in services[:6]:
            if isinstance(svc, dict):
                name = str(svc.get("name", "") or "")
                role = str(svc.get("role", "") or "")
                lines.append(f"    - {name} ({role})" if role else f"    - {name}")
            else:
                lines.append(f"    - {svc}")
    return "\n".join(lines)


async def fetch_folder_tree(snapshot_id: str, max_files: int = 60) -> str:
    """Compact top-level file listing from manifest_files for a snapshot."""
    db = get_db()
    rows = []
    try:
        async with db.execute(
            """SELECT rel_path FROM manifest_files
               WHERE snapshot_id=?
               ORDER BY rel_path ASC
               LIMIT ?""",
            (snapshot_id, max_files),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        pass
    return "\n".join(row["rel_path"] for row in rows) if rows else ""


def build_convention_block(report: ConventionReport | None) -> str:
    if report is None or not report.signals:
        return ""
    return report.as_context_text()


def build_risk_block(report: RiskReport | None, categories: list[str] | None = None) -> str:
    cats = tuple(categories) if categories is not None else ("blast_radius", "anti_pattern")
    if report is None:
        return ""
    rows = [f for f in report.findings if f.category in cats]
    if not rows:
        return ""
    lines: list[str] = ["Blast radius and boundary violations (FACTS — do not contradict):"]
    for f in rows:
        lines.append(f"RISK [{f.category}] {f.title}: {f.rationale}")
    return "\n".join(lines)


def extract_d_hint_context(agent_d_output: dict[str, Any] | None) -> str:
    """D→E contract: Section D exposes `signals`, never `rules`."""
    if not agent_d_output:
        return ""
    raw = agent_d_output.get("signals", [])
    if not isinstance(raw, list):
        return ""
    out_lines: list[str] = []
    for item in raw[:5]:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category", "") or "")
        hint = str(
            item.get("pattern", "") or item.get("description", "") or item.get("evidence", "") or ""
        )[:90]
        if not hint and not cat:
            continue
        if cat:
            out_lines.append(f"{cat}: {hint}".strip())
        else:
            out_lines.append(hint)
    if not out_lines:
        return ""
    prefix = (
        "Team patterns already confirmed (use for negative-space inference — "
        "look for violations):\n"
    )
    return prefix + "\n".join(out_lines)
