"""Renders an AEH Eval Plan markdown document for handoff to an external coding agent."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from agent_eval_harness.code_injection.prompt_locator import locate_prompt_reference
from agent_eval_harness.mapping.system_map import SystemMap
from agent_eval_harness.planning.report import AgentPlanReport, EvaluationPlanReport

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def render_eval_plan_md(
    system_map: SystemMap,
    wiring: dict,
    dataset_summaries: list[dict],
    session_id: str,
    branch_name: str,
    plan_report: EvaluationPlanReport | None = None,
    repo_root: Path | None = None,
) -> str:
    """Build a self-contained briefing document for an external coding agent.

    Args:
        system_map: SystemMap instance — read field names from system_map.py before using.
        wiring: dict from build_wiring_for_codespectra().
        dataset_summaries: list of dicts with keys dataset_id, kind, case_count, gate_ids,
            and optionally example_case (one DatasetCase.model_dump(), for a peek at real shape).
        session_id: AEH expansion session id.
        branch_name: eval branch name (e.g. aeh/eval-<session_id>).
        plan_report: Stage 3's EvaluationPlanReport (per-agent data_profile/contract/gates),
            already computed and saved to plan_report_path — read it, never re-derive it.
            When None, falls back to a thin wiring table (legacy sessions with no report).
        repo_root: required when plan_report is given, to resolve prompt-location references
            against real files on disk.
    """
    sections: list[str] = []

    # Section 1: Header
    sections.append(
        f"# AEH Eval Plan\n"
        f"Target system: {system_map.target_system_id}\n"
        f"Session: {session_id}\n"
        f"Branch: {branch_name}\n"
        f"plan_id: {wiring['plan_id']}\n"
        f"Generated: {datetime.datetime.utcnow().isoformat()}Z\n\n"
        f"This document is a self-contained briefing for an external coding agent. "
        f"It contains all file content, per-agent context, and instructions needed "
        f"to instrument the target system for AEH evaluation. "
        f"Do not run AEH to create these files — implement them directly from this document."
    )

    # Section 2: Agents — rich per-agent context pulled from Stage 3's already-computed
    # plan_report, falling back to a thin component table when no report is available.
    if plan_report is not None:
        if repo_root is None:
            raise ValueError("repo_root is required when plan_report is provided")
        agent_sections = [
            _render_agent_section(agent_report, system_map, dataset_summaries, repo_root)
            for agent_report in plan_report.agents
        ]
        sections.append("## Agents\n\n" + "\n\n".join(agent_sections))
    else:
        component_rows = "\n".join(
            f"| {c.id} | {c.entry_point} |" for c in system_map.components
        )
        sections.append(
            f"## Wiring Summary\n"
            f"| Component ID | Entry Point |\n"
            f"|---|---|\n"
            f"{component_rows}"
        )
        dataset_rows = "\n".join(
            f"| {d['dataset_id']} | {d['kind']} | {d['case_count']} | {', '.join(d['gate_ids'])} |"
            for d in dataset_summaries
        )
        sections.append(
            f"## Datasets\n"
            f"| Dataset ID | Kind | Cases | Gate IDs |\n"
            f"|---|---|---|---|\n"
            f"{dataset_rows}"
        )

    # Section 3: File 1/4 — tracer.py
    tracer_content = (_TEMPLATES_DIR / "tracer.py").read_text(encoding="utf-8")
    sections.append(
        f"## File 1/4: backend/.aeh/tracer.py\n"
        f"```python\n{tracer_content}\n```"
    )

    # Section 4: File 2/4 — run_eval.py
    run_eval_content = (_TEMPLATES_DIR / "run_eval.py").read_text(encoding="utf-8")
    sections.append(
        f"## File 2/4: backend/.aeh/run_eval.py\n"
        f"```python\n{run_eval_content}\n```"
    )

    # Section 5: File 3/4 — wiring.json
    sections.append(
        f"## File 3/4: backend/.aeh/wiring.json\n"
        f"```json\n{json.dumps(wiring, indent=2)}\n```"
    )

    # Section 6: File 4/4 — aeh_eval.py
    aeh_eval_content = (_TEMPLATES_DIR / "aeh_eval.py").read_text(encoding="utf-8")
    sections.append(
        f"## File 4/4: backend/api/aeh_eval.py\n"
        f"```python\n{aeh_eval_content}\n```"
    )

    # Section 7: main.py edits — read from template, not hardcoded
    main_py_diff = (_TEMPLATES_DIR / "main_py_diff.txt").read_text(encoding="utf-8")
    sections.append(
        f"## backend/main.py edits\n"
        f"```python\n{main_py_diff}\n```"
    )

    # Section 8: Verify
    sections.append(
        "## Step: Verify\n"
        "After writing the files, restart the CodeSpectra backend and confirm "
        "`/aeh/run-eval` appears in FastAPI docs at `/docs`. Then call "
        "`POST /aeh/run-eval?verify=true` (from `/docs` or curl) BEFORE running the full "
        "eval — this runs exactly ONE case and confirms it actually succeeded. "
        "Do not proceed to the full run until this returns success; if it fails, the "
        "wiring (component ids in `wiring.json`, or the `main.py` hook placement) is "
        "wrong and needs fixing first — running the full set against broken wiring "
        "just wastes every case."
    )

    # Section 9: Run
    sections.append(
        "## Step: Run\n"
        "Once `?verify=true` succeeds, run `POST /aeh/run-eval` (no query param) from "
        "`/docs` or curl to execute the full eval driver against every case. "
        "Results are written to `backend/.aeh/manifest.json`."
    )

    # Section 10: Hand Back
    sections.append(
        "## Step: Hand Back\n"
        "Once `backend/.aeh/manifest.json` exists, return to the AEH UI and click "
        "**Load Results** (Stage 5). AEH reads from the fixed path automatically — "
        "no path picker needed."
    )

    return "\n\n---\n\n".join(sections)


def _render_agent_section(
    agent_report: AgentPlanReport,
    system_map: SystemMap,
    dataset_summaries: list[dict],
    repo_root: Path,
) -> str:
    lines = [f"### Agent: {agent_report.agent_id} (role: {agent_report.role})"]

    contract = agent_report.contract
    primary_component_id = contract.component_id if contract else ""
    owned_ids = {g.component for g in agent_report.gates if g.component}
    if primary_component_id:
        owned_ids.add(primary_component_id)
    primary_component = (
        system_map.component_by_id(primary_component_id) if primary_component_id else None
    )
    if primary_component is None and owned_ids:
        # No contract-declared primary component — fall back to the first gate's component.
        primary_component = system_map.component_by_id(sorted(owned_ids)[0])

    if primary_component is not None:
        lines.append(f"- **Code**: `{primary_component.file or 'file not tracked'}`")
        prompt_ref = (
            locate_prompt_reference(primary_component.file, repo_root)
            if primary_component.file
            else None
        )
        if prompt_ref:
            lines.append(f"- **Prompt**: `{prompt_ref}`")
        else:
            lines.append(
                f"- **Prompt**: not auto-detected — search `{primary_component.file}` "
                "for a prompt-related import or constant (e.g. a `*_SYSTEM`/`*_PROMPT` "
                "name imported from a sibling `prompts.py`), or check whether this agent "
                "stores its prompt outside the codebase (DB, config)."
            )
    else:
        lines.append("- **Code**: no component resolved for this agent in the system map.")

    if len(owned_ids) > 1:
        lines.append(f"- **Other owned components**: {sorted(owned_ids - {primary_component_id})}")

    profile = agent_report.data_profile
    if profile is not None:
        lines.append(f"- **Input**: {profile.input_data or 'not described'}")
        lines.append(f"- **Output**: {profile.output_data or 'not described'}")
        if profile.internal_tools:
            lines.append(f"- **Tools/helpers used**: {'; '.join(profile.internal_tools)}")
        if profile.failure_modes:
            lines.append(f"- **Known failure modes**: {'; '.join(profile.failure_modes)}")
        if profile.consistency_notes:
            lines.append(f"- **Consistency notes**: {'; '.join(profile.consistency_notes)}")

    if contract and contract.invocation:
        inv = contract.invocation
        kwarg_names = [k.name for k in inv.kwargs]
        citation = f", cited at {inv.citations[0]}" if inv.citations else ""
        lines.append(
            f"- **Invocation**: `{inv.callable}.{inv.method}` "
            f"kwargs={kwarg_names} (source: {inv.source}{citation})"
        )

    agent_gate_ids = {g.id for g in agent_report.gates}
    agent_datasets = [
        d for d in dataset_summaries if agent_gate_ids & set(d.get("gate_ids", []))
    ]
    if agent_datasets:
        rows = "\n".join(
            f"  | {d['dataset_id']} | {d['kind']} | {d['case_count']} | "
            f"{', '.join(d['gate_ids'])} |"
            for d in agent_datasets
        )
        lines.append(
            "- **Datasets**:\n\n"
            "  | Dataset ID | Kind | Cases | Gate IDs |\n"
            "  |---|---|---|---|\n"
            f"{rows}"
        )
        for d in agent_datasets:
            if d.get("example_case"):
                lines.append(
                    f"  Example case from `{d['dataset_id']}`:\n"
                    f"  ```json\n{json.dumps(d['example_case'], indent=2)}\n  ```"
                )

    if agent_report.needs_human:
        lines.append(f"- **Needs human review**: {', '.join(agent_report.needs_human)}")

    return "\n".join(lines)
