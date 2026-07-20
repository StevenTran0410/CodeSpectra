"""Plan renderer: compile→render driver for the 4-file .md handoff (AGENTS.md, TASKS.md, REFERENCE.md, CODE.md)."""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

from agent_eval_harness.code_injection.facts import compile_plan_facts
from agent_eval_harness.code_injection.template_engine import resolve, clear_discovery_tasks
from agent_eval_harness.mapping.system_map import SystemMap
from agent_eval_harness.planning.report import EvaluationPlanReport

logger = logging.getLogger("agent_eval_harness.code_injection.plan_renderer")

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


_PLAN_FILES: list[tuple[str, str]] = [
    ("AGENTS.md", "agents_md.tpl"),
    ("TASKS.md", "tasks_md.tpl"),
    ("REFERENCE.md", "reference_md.tpl"),
    ("CODE.md", "code_md.tpl"),
    ("RECON.md", "recon_md.tpl"),
]

# Written only when absent — holds the coding agent's own findings; re-rendering used to wipe them.
PRESERVED_PLAN_FILES: frozenset[str] = frozenset({"RECON.md"})


def render_eval_plan_files(
    system_map: SystemMap,
    wiring: dict,
    dataset_summaries: list[dict],
    session_id: str,
    branch_name: str,
    plan_report: EvaluationPlanReport | None = None,
    repo_root: Path | None = None,
    base_ref: str = "main",
    knowledge_dir: Path | None = None,
) -> dict[str, str]:
    """Compile facts once and render the 4 self-contained handoff files.

    Returns {filename: content}; caller writes each as its own file.
    """
    clear_discovery_tasks()

    facts = compile_plan_facts(
        system_map=system_map,
        wiring=wiring,
        dataset_summaries=dataset_summaries,
        session_id=session_id,
        branch_name=branch_name,
        plan_report=plan_report,
        repo_root=repo_root,
        knowledge_dir=knowledge_dir,
    )

    out: dict[str, str] = {}
    for filename, tpl_name in _PLAN_FILES:
        tpl_path = _TEMPLATES_DIR / tpl_name
        if not tpl_path.exists():
            raise ValueError(f"Master template file missing: {tpl_path}")
        out[filename] = resolve(tpl_path.read_text(encoding="utf-8"), facts)
    return out


def render_eval_plan_md(
    system_map: SystemMap,
    wiring: dict,
    dataset_summaries: list[dict],
    session_id: str,
    branch_name: str,
    plan_report: EvaluationPlanReport | None = None,
    repo_root: Path | None = None,
    base_ref: str = "main",
    knowledge_dir: Path | None = None,
) -> str:
    """Backward-compat single-document view: the 4 files concatenated; prefer render_eval_plan_files() for the real handoff."""
    files = render_eval_plan_files(
        system_map, wiring, dataset_summaries, session_id, branch_name,
        plan_report=plan_report, repo_root=repo_root, base_ref=base_ref,
        knowledge_dir=knowledge_dir,
    )
    parts = [
        "# AEH Eval Plan (4-File, concatenated view)",
        f"Session: {session_id}",
        f"Branch: {branch_name}",
        f"Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}Z",
        "",
        "**This is a reference concatenation. The real handoff is 4 separate files.**",
        "**Read AGENTS.md first, then TASKS.md as the working checklist.**",
        "",
        "---",
        "",
    ]
    for filename, content in files.items():
        parts.append(f"## {filename}\n")
        parts.append(content)
        parts.append("\n---\n")
    return "\n".join(parts)
