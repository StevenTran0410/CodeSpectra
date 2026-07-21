"""Template engine: resolves {{type:arg}} slots in plan templates using PlanFacts.

Five slot types: scalar, table, transclude, block, marker. Missing facts emit both an inline marker and a discovery task.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel

from agent_eval_harness.code_injection.facts import (
    AgentFacts,
    PlanFacts,
    Resolved,
)

logger = logging.getLogger("agent_eval_harness.code_injection.template_engine")

_BLOCK_FLAGS = {
    "retrieval_stub_warning": "retrieval_stub_warning.md",
    "anchor_recovery": "anchor_recovery.md",
    "unresolved_binding_skip": "unresolved_binding_skip.md",
    "multi_llm_call_sequence": "multi_llm_call_sequence.md",
    "troubleshooting_null_tracer_check": "troubleshooting_null_tracer_check.md",
    "case_boundary_sentinel_required": "case_boundary_sentinel_required.md",
    "troubleshooting_stale_data_pointer": "troubleshooting_stale_data_pointer.md",
    "runbook_missing": "runbook_missing.md",
    "provider_missing": "provider_missing.md",
}

_BLOCKS_DIR = Path(__file__).resolve().parent / "blocks"

_DISCOVERY_TASKS: list[dict] = []


class DiscoveryTask(BaseModel):
    """A task to discover/clarify a missing fact."""
    task_id: str
    fact_id: str
    title: str
    instructions: str
    constraint: str | None = None


def resolve(text: str, facts: PlanFacts, agent: AgentFacts | None = None, depth: int = 0) -> str:
    """Resolve {{type:arg}} slots in template text; depth is capped at 2 to prevent runaway recursion."""
    if depth > 2:
        raise ValueError(f"Template recursion depth exceeded (>{depth})")

    pattern = r"\{\{(\w+):([^\}]+)\}\}"

    def slot_replacer(match: re.Match) -> str:
        slot_type = match.group(1)
        arg = match.group(2).strip()

        try:
            if slot_type == "scalar":
                return _resolve_scalar(arg, facts, agent)
            elif slot_type == "table":
                return _resolve_table(arg, facts, agent)
            elif slot_type == "transclude":
                return _resolve_transclude(arg, facts, agent)
            elif slot_type == "block":
                return _resolve_block(arg, facts, agent, depth + 1)
            elif slot_type == "marker":
                return _resolve_marker(arg, facts, agent)
            elif slot_type == "code":
                return _resolve_code(arg, facts)
            elif slot_type == "sha256":
                return _resolve_sha256(arg, facts)
            else:
                raise ValueError(f"Unknown slot type: {slot_type}")
        except Exception as e:
            logger.error(f"Failed to resolve {{{{{slot_type}:{arg}}}}}: {e}")
            raise

    resolved = re.sub(pattern, slot_replacer, text)
    return resolved


def _resolve_scalar(arg: str, facts: PlanFacts, agent: AgentFacts | None) -> str:
    """Resolve {{scalar:field}} — simple value from facts."""
    parts = arg.split(".")

    if agent and parts[0] == "agent":
        if parts[1] == "id":
            return agent.agent_id
        elif parts[1] == "role":
            return agent.role
        else:
            return f"[UNKNOWN_SCALAR: agent.{parts[1]}]"

    if parts[0] == "plan":
        if parts[1] == "session_id":
            return facts.session_id
        elif parts[1] == "target_system_id":
            return facts.target_system_id
        elif parts[1] == "branch_name":
            return facts.branch_name
        elif parts[1] == "plan_id":
            return facts.plan_id
        elif parts[1] == "db_path":
            return facts.db_path
        elif parts[1] == "total_agents":
            return str(len(facts.agents))
        elif parts[1] == "batch_size":
            return str(facts.batch_size)
        elif parts[1] == "dataset_ids":
            return json.dumps(facts.dataset_ids)
        elif parts[1] == "main_entry_path":
            if facts.main_entry_path is None:
                return "[NEEDS CLARIFICATION]"
            elif isinstance(facts.main_entry_path, Resolved):
                return facts.main_entry_path.value or "[NEEDS CLARIFICATION]"
            else:
                return "[NEEDS CLARIFICATION]"
        else:
            return f"[UNKNOWN_SCALAR: plan.{parts[1]}]"

    if parts[0] == "runbook":
        if parts[1] == "start_command":
            return facts.runbook.start_command
        elif parts[1] == "sha256_posix":
            return facts.runbook.sha256_command_posix
        elif parts[1] == "sha256_windows":
            return facts.runbook.sha256_command_windows
        else:
            return f"[UNKNOWN_SCALAR: runbook.{parts[1]}]"

    return f"[UNKNOWN_SCALAR: {arg}]"


def _resolve_table(arg: str, facts: PlanFacts, agent: AgentFacts | None) -> str:
    """Resolve {{table:arg}} — markdown table from facts."""
    if arg == "agents":
        lines = ["| Agent ID | Role |", "| --- | --- |"]
        for a in facts.agents:
            lines.append(f"| {a.agent_id} | {a.role} |")
        return "\n".join(lines)

    if arg == "datasets":
        runnable = [d for d in facts.dataset_summaries if d.get("case_count", 0) > 0]
        empty = [d for d in facts.dataset_summaries if d.get("case_count", 0) == 0]
        lines = ["| Dataset ID | Kind | Cases |", "| --- | --- | --- |"]
        for ds in runnable:
            lines.append(f"| `{ds.get('dataset_id', '')}` | {ds.get('kind', '')} | {ds.get('case_count', 0)} |")
        if not runnable:
            lines.append("| _(none)_ | | 0 |")
        if empty:
            kinds = sorted({d.get("kind", "") or "unknown" for d in empty})
            lines.append("")
            lines.append(
                f"**Ignore these {len(empty)} dataset(s)** — kind(s) {', '.join(f'`{k}`' for k in kinds)} — "
                "they hold no reviewed cases, so the driver's provenance filter skips them and no "
                "dispatch module will ever see one. They are listed in `wiring.json` only so a later "
                "review can activate them; do not treat their absence from the run as a failure."
            )
            lines.append("")
            lines.append("<details><summary>Ignored dataset ids</summary>\n")
            for ds in empty:
                lines.append(f"- `{ds.get('dataset_id', '')}` ({ds.get('kind', '')})")
            lines.append("\n</details>")
        return "\n".join(lines)

    if agent and arg == "components":
        if agent.components and isinstance(agent.components, Resolved):
            lines = ["| Component | Role | Location |", "| --- | --- | --- |"]
            for comp in agent.components.value:
                lines.append(f"| {comp.id} | {comp.role} | {comp.file}:{comp.line} |")
            return "\n".join(lines)
        return "(no components found)"

    if agent and arg == "inputs":
        if agent.input_contract:
            lines = ["| Kwarg | Type | Source |", "| --- | --- | --- |"]
            for arg_fact in agent.input_contract:
                if isinstance(arg_fact, Resolved):
                    lines.append(f"| {arg_fact.value.kwarg} | {arg_fact.value.type_hint} | {arg_fact.value.source_kind} |")
            return "\n".join(lines)
        return "(no inputs found)"

    if arg == "target_overview":
        roles: dict[str, int] = {}
        for a in facts.agents:
            roles[a.role or "unknown"] = roles.get(a.role or "unknown", 0) + 1
        total_cases = sum(ds.get("case_count", 0) for ds in facts.dataset_summaries)
        lines = ["| Property | Value |", "| --- | --- |",
                 f"| Agents | {len(facts.agents)} |",
                 f"| Roles | {', '.join(f'{r}×{n}' for r, n in sorted(roles.items())) or '(none)'} |",
                 f"| Datasets | {len(facts.dataset_summaries)} |",
                 f"| Total cases | {total_cases} |"]
        return "\n".join(lines)

    if arg == "endpoints":
        eps = facts.runbook.endpoints
        if not eps:
            return "(none harvested — see runbook recovery below)"
        return "\n".join(f"- `{e}`" for e in eps)

    if arg == "provider_listing":
        pl = facts.provider_listing
        if isinstance(pl, Resolved):
            return str(pl.value)
        return "(not harvested — the coding agent must list providers at runtime; see discovery task)"

    if arg == "agent_cards":
        if not facts.agents:
            return "(no agents)"
        return "\n\n".join(_render_agent_card(a, facts) for a in facts.agents)

    if arg == "agents_json":
        data = [{"agent_id": a.agent_id, "role": a.role,
                 "invocation_mode": a.invocation_mode, "location": _location_str(a)}
                for a in facts.agents]
        return json.dumps(data, indent=2)

    if arg == "discovery_tasks":
        # Rendered after the markers that register them (resolve() runs left-to-right).
        if not _DISCOVERY_TASKS:
            return "(none — every fact the plan needs was harvested)"
        rows = []
        for t in _DISCOVERY_TASKS:
            rows.append(f"- [ ] {t['task_id']} {t['title']} — {t['instructions']}")
            rows.append(
                "      Record the answer in § RECON before continuing. "
                "Never read or print a credential value while investigating."
            )
        return "\n".join(rows)

    if arg == "skeleton_tasks":
        return _render_skeleton_tasks(facts)

    if arg == "batch_headers":
        return _render_batches(facts)

    if arg == "finish_tasks":
        return _render_finish_tasks(facts)

    if arg == "dispatch_modules":
        if not facts.dispatch_modules:
            return "(no dispatch modules)"
        parts = []
        for dm in facts.dispatch_modules:
            parts.append(
                f"### `{dm['filename']}`  (agent: {dm['agent_id']})\n\n"
                f"**SHA256**: `{dm['sha']}`\n\n"
                f"```python\n{dm['code']}\n```"
            )
        return "\n\n".join(parts)

    if arg == "main_py_edits":
        return _render_entrypoint_edits()

    return f"[UNKNOWN_TABLE: {arg}]"


def _resolve_transclude(arg: str, facts: PlanFacts, agent: AgentFacts | None) -> str:
    """Resolve {{transclude:field}} — embed verbatim prose from agent knowledge."""
    if agent and arg in agent.transclude:
        return agent.transclude[arg]
    return f"(no {arg} found)"


def _resolve_block(arg: str, facts: PlanFacts, agent: AgentFacts | None, depth: int) -> str:
    """Resolve {{block:name}}; degrade blocks collapse to empty once their fact resolves."""
    if arg == "runbook_missing" and facts.runbook.start_command:
        return ""
    if arg == "provider_missing" and isinstance(facts.provider_listing, Resolved):
        return ""
    if arg not in _BLOCK_FLAGS:
        raise ValueError(f"Unknown block: {arg}")

    block_file = _BLOCKS_DIR / _BLOCK_FLAGS[arg]
    if not block_file.exists():
        logger.warning(f"Block file not found: {block_file}")
        return f"[BLOCK_NOT_FOUND: {arg}]"

    block_text = block_file.read_text(encoding="utf-8")

    resolved_block = resolve(block_text, facts, agent, depth)

    return resolved_block


def _resolve_marker(arg: str, facts: PlanFacts, agent: AgentFacts | None) -> str:
    """Resolve {{marker:fact_id}} — emit marker + discovery task for missing fact."""
    # D-prefixed: these are conditional lookups, not steps in the T-numbered sequence.
    task_id = f"D{len(_DISCOVERY_TASKS) + 1}"
    task = DiscoveryTask(
        task_id=task_id,
        fact_id=arg,
        title=f"Clarify {arg}",
        instructions=f"Determine and document: {arg}",
    )
    _DISCOVERY_TASKS.append(task.model_dump())

    return f"\n[NEEDS CLARIFICATION: {arg} — see {task_id}]\n"


def get_discovery_tasks() -> list[dict]:
    """Get all discovery tasks discovered during rendering."""
    return _DISCOVERY_TASKS.copy()


def clear_discovery_tasks() -> None:
    """Clear the discovery task list for next render."""
    global _DISCOVERY_TASKS
    _DISCOVERY_TASKS = []


def _render_entrypoint_edits() -> str:
    """Describe the entrypoint edits by semantic position — literal anchors go stale."""
    return "\n".join([
        "Insert three marker-wrapped blocks into the server entrypoint file. Find each",
        "position yourself, then record the file and line you used in TASKS.md § RECON.",
        "Markers make the edits idempotent and removable — keep them exactly as written.",
        "",
        "**Block 1 — enable tracing** · position: the very top of the file, **before any",
        "framework or application import** (the tracer must be enabled before the framework",
        "is first imported, or no spans are captured).",
        "",
        "```python",
        "# aeh:begin aeh_tracer",
        "import os",
        'os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")',
        "# aeh:end aeh_tracer",
        "```",
        "",
        "**Block 2 — import the eval router** · position: beside the file's other router /",
        "blueprint imports. `.aeh` is not an importable package name, so either move",
        "`aeh_eval.py` next to the app's other route modules, or keep it in `.aeh/` and put",
        "that directory on `sys.path` first. Record which you chose in § RECON.",
        "",
        "```python",
        "# aeh:begin aeh_eval_import",
        "from aeh_eval import router as aeh_eval_router",
        "# aeh:end aeh_eval_import",
        "```",
        "",
        "**Block 3 — register the eval router** · position: beside the other route",
        "registrations, using the same registration call and indentation the file already uses.",
        "",
        "```python",
        "# aeh:begin aeh_eval_route",
        'app.include_router(aeh_eval_router, prefix="/aeh")',
        "# aeh:end aeh_eval_route",
        "```",
        "",
        "> If the entrypoint does not use this framework's API, adapt the *registration call*",
        "> to whatever it does use — but keep the markers and the route path `/aeh`.",
        "> If you cannot find a registration site at all, stop and report it as a [Conflict].",
    ])


def _location_str(a: AgentFacts) -> str:
    """One-line file:line for an agent, or empty if unresolved."""
    if isinstance(a.location, Resolved):
        loc = a.location.value
        return f"{loc.file}:{loc.line_start}"
    return ""


def _render_agent_card(a: AgentFacts, facts: PlanFacts) -> str:
    """Render one per-agent card: functionality + WHERE + WIRE + retrieval warning."""
    lines = [f"### Agent: {a.agent_id}  (role: {a.role or 'unknown'})"]

    func = (a.transclude.get("functionality") or "").strip()
    if func:
        lines.append(f"\n> {func}")

    where = ["\n**WHERE** (from code index):", "", "| What | Location |", "| --- | --- |"]
    if isinstance(a.location, Resolved):
        loc = a.location.value
        where.append(f"| Class/entry | `{loc.file}:{loc.line_start}-{loc.line_end}` |")
        where.append(f"| Entry method | `{loc.entry_method}()` @ line {loc.entry_line} |")
    else:
        where.append("| Location | [NEEDS CLARIFICATION] |")
    if isinstance(a.prompt_sites, Resolved) and a.prompt_sites.value:
        ps = a.prompt_sites.value[0]
        where.append(f"| Prompt | `{ps.file}:{ps.line}` |")
    if isinstance(a.components, Resolved):
        for c in a.components.value:
            where.append(f"| Component `{c.id}` | `{c.file}:{c.line}` |")
    lines.append("\n".join(where))

    lines.append(f"\n**Invocation mode**: `{a.invocation_mode}`")
    if isinstance(a.case_binding, Resolved) and a.case_binding.value:
        wire = ["\n**WIRE** (case field → kwarg):", "", "| case field | → kwarg | kwarg type |", "| --- | --- | --- |"]
        for kwarg, binding in a.case_binding.value.items():
            annotation = a.typed_bindings.get(kwarg)
            wire.append(f"| `{binding}` | `{kwarg}` | {f'`{annotation}`' if annotation else 'plain'} |")
        lines.append("\n".join(wire))
        if a.typed_bindings:
            typed = ", ".join(f"`{k}`: `{v}`" for k, v in a.typed_bindings.items())
            lines.append(
                f"\n⚠️ **Typed kwarg**: {typed} — the case supplies a raw JSON value, but the "
                "entry method annotates a project type it dereferences by attribute. Build that "
                "type from the raw value in your dispatch region instead of passing the dict "
                "through; read the real constructor from the target's source."
            )
    else:
        lines.append("\n**WIRE**: no case_binding harvested — dispatch module uses empty kwargs.")

    if a.unsatisfiable_bindings:
        fields = ", ".join(f"`{f}`" for f in a.unsatisfiable_bindings)
        lines.append(
            f"\n🛑 **Unsatisfiable binding**: {fields} — bound to a case field that **no case "
            "carries**, so the kwarg arrives as `None`. If the agent validates it, the call "
            "fails before any stub is consulted; if the agent swallows that error, the case "
            "still reports success. Stop and report this rather than inventing a value."
        )

    if a.degraded_bindings:
        fields = ", ".join(f"`{f}`" for f in a.degraded_bindings)
        lines.append(
            f"\n⚠️ **Optional binding never populated**: {fields} — bound, but absent from the "
            "sample case, so the kwarg falls back to its default. The call still runs; it just "
            "sees less context than the binding implies. Confirm against the real cases "
            "(T004c) and note it in `RECON.md` if it holds for all of them."
        )

    if a.unconsumed_case_fields:
        fields = ", ".join(f"`{f}`" for f in a.unconsumed_case_fields)
        lines.append(
            f"\n⚠️ **Unusable case fields**: {fields} — the entry method has no parameter for "
            "these, so the agent builds that context itself and the case's version is ignored. "
            "Nothing you write in the dispatch can change that. Expect this agent's output to "
            "reflect the real environment for those inputs, and say so when you hand back."
        )

    if a.has_retrieval_signal:
        block_file = _BLOCKS_DIR / _BLOCK_FLAGS["retrieval_stub_warning"]
        if block_file.exists():
            lines.append("\n" + resolve(block_file.read_text(encoding="utf-8"), facts, a, depth=1))

    return "\n".join(lines)


# Only target-independent harness plumbing may be sha-locked; anything depending on the target's own interfaces belongs below, where the agent can still correct it.
_STATIC_ARTIFACTS: list[tuple[str, str]] = [
    ("tracer", ".aeh/tracer.py"),
    ("run_eval", ".aeh/run_eval.py"),
    ("aeh_eval", ".aeh/aeh_eval.py"),
]

# Generated from the target's own harvested signatures, so a harvest mistake must be fixable in place; sha-locking it would leave no legal way to correct a bad binding.
_TARGET_DERIVED_ARTIFACTS: list[tuple[str, str]] = [
    ("wiring", ".aeh/wiring.json"),
]

_BATCH_FIRST_MILESTONE = 2  # M0 = setup, M1 = skeleton, batches start at M2
_BATCH_FIRST_TASK = 20      # T001-T009 reserved for setup + skeleton


def _batch_count(facts: PlanFacts) -> int:
    bs = max(1, facts.batch_size)
    return (len(facts.agents) + bs - 1) // bs


def _render_skeleton_tasks(facts: PlanFacts) -> str:
    """M1 — install the harness files verbatim, then prove the service still starts."""
    lines = [
        "## M1 — Skeleton: install the harness files",
        "> Điều kiện vào: [x] 🛑 GATE M0. These files are copied verbatim — do not edit them.",
        "",
        "- [ ] T004e Determine the target's Python import root — the directory on `sys.path` "
        "when the app runs, from which its own package imports resolve — and create `.aeh/` "
        "directly inside it. `run_eval.py` puts its own parent on `sys.path`, so if `.aeh/` "
        "sits anywhere else the target's imports fail. Record the resolved path in `RECON.md`; "
        "every `python .aeh/run_eval.py …` command below is relative to it.",
        "      Verify: `python -c \"import <target_top_level_package>\"` succeeds from `.aeh/`'s parent",
        "",
        "- [ ] T005 If `.aeh/` already exists from an earlier plan, move it aside — "
        "`mv .aeh .aeh.bak-<timestamp>` — then start clean. Do not merge old and new installs, "
        "and never delete it: its `out/*.jsonl` and `manifest.json` are untracked, so git "
        "cannot recover them. Report what the old manifest claimed (`attempted`/`succeeded`) "
        "in RECON.md rather than summarising it from memory or from any doc.",
        "      Verify: `.aeh/` is absent or empty before you copy anything into it",
        "",
        "- [ ] T006 If you moved an install aside, copy every file listed under "
        "`agent_owned_files` in the OLD `.aeh.bak-*/wiring.json` back into the fresh `.aeh/`. "
        "Those files hold your implementations, not generated content — a reinstall that skips "
        "this step silently reverts them and the run regresses while the gates still pass. "
        "Add any further files you author to that list so the next reinstall keeps them too.",
        "      Verify: every path in the old `agent_owned_files` exists in the new `.aeh/`, "
        "or the list was empty",
        "",
    ]
    tnum = 7
    for key, path in _STATIC_ARTIFACTS:
        if not facts.code_artifacts.get(key):
            continue  # nothing generated for this target
        sha = facts.code_shas.get(key, "")
        lines.append(f"- [ ] T{tnum:03d} Create `{path}` — copy the matching block from CODE.md **verbatim**")
        lines.append(f"      Verify: sha256 of `{path}` == `{sha or '(see CODE.md)'}`")
        tnum += 1
    for key, path in _TARGET_DERIVED_ARTIFACTS:
        if not facts.code_artifacts.get(key):
            continue
        lines.append(
            f"- [ ] T{tnum:03d} Create `{path}` from CODE.md. Generated from this target's "
            "harvested signatures, so it is **not** sha-locked: if a binding contradicts the "
            "real signature, correct it here and record the change in RECON.md as `[Conflict]`."
        )
        lines.append(f"      Verify: `{path}` parses as JSON and every agent id in it has a dispatch entry")
        tnum += 1
    if facts.code_artifacts.get("retrieval_stub"):
        lines.append(
            f"- [ ] T{tnum:03d} Create `.aeh/retrieval_stub.py` from CODE.md, then **implement its "
            "`=== IMPLEMENT THIS ===` region**: give the stub the real retrieval dependency's "
            "method names, signatures, return types and async-ness, reading them from the target's "
            "source. Not sha-locked — its interface belongs to the target, not to this plan."
        )
        lines.append(
            "      Verify: `grep -c \"IMPLEMENT THIS\" .aeh/retrieval_stub.py` → 0, and one real "
            "call through it returns content derived from the case's evidence (a shape mismatch "
            "can be swallowed by the caller and leave the agent running on nothing)"
        )
        tnum += 1

    lines.append(
        f"- [ ] T{tnum:03d} Locate the target's server entrypoint (the file that registers its routes) "
        "and apply the three marker blocks from CODE.md § \"Server Entrypoint Edits\""
    )
    lines.append(
        "      Record the file you edited in § RECON. "
        "Verify: `grep -c \"aeh:begin\" <that file>` → 3, and the service still starts"
    )
    lines.append(
        f"- [ ] T{tnum + 1:03d} Create `.aeh/target_init.py` from CODE.md, then **implement its "
        "`=== IMPLEMENT THIS ===` region** — the target's once-per-process setup "
        "(database/connection init, config bootstrap). Not sha-locked: it is yours, and a "
        "newer plan can ship a newer `run_eval.py` without erasing it."
    )
    lines.append(
        "      Verify: `python .aeh/run_eval.py --verify` gets past setup without an "
        "\"not initialised\"-style error"
    )
    lines.append(
        "\n🛑 GATE M1: start the target service (REFERENCE.md § Runbook) and confirm the eval route "
        "answers. Fail → stop, report the task id; do not start wiring agents."
    )
    return "\n".join(lines)


def _render_batches(facts: PlanFacts) -> str:
    """Split agents into batches and render the per-agent wiring milestones."""
    agents = facts.agents
    if not agents:
        return "(no agents to wire)"
    bs = max(1, facts.batch_size)
    total = len(agents)
    n_batches = _batch_count(facts)
    out: list[str] = []
    tnum = _BATCH_FIRST_TASK
    done = 0
    for bi in range(n_batches):
        batch = agents[bi * bs:(bi + 1) * bs]
        m = bi + _BATCH_FIRST_MILESTONE
        prev_gate = f"M{m - 1}"
        remaining = total - done - len(batch)
        block = [
            f"## M{m} — Batch {bi + 1}/{n_batches}: wire agents {bi * bs + 1}-{bi * bs + len(batch)}",
            f"> Tổng: {total} · Xong: {done} · Batch này: {', '.join(a.agent_id for a in batch)} · Còn: {remaining}",
            f"> Điều kiện vào: [x] 🛑 GATE {prev_gate}. New session → re-run the {prev_gate} Verify command first.",
            "",
        ]
        for a in batch:
            block.append(
                f"- [ ] T{tnum:03d} [P] Create `.aeh/dispatch/{a.agent_id}.py` from its CODE.md block, "
                f"then **implement its `=== IMPLEMENT THIS ===` region** (the generated kwargs above it are final — do not edit them)"
            )
            block.append(
                f"      Verify: `python .aeh/run_eval.py --verify --agent {a.agent_id}` → exit 0, ≥1 span, "
                f"and `grep -c \"IMPLEMENT THIS\" .aeh/dispatch/{a.agent_id}.py` → 0 leftover NotImplementedError"
            )
            tnum += 1
        block.append(
            f"\n🛑 GATE M{m}: `python .aeh/run_eval.py --verify --batch {bi + 1}` → {len(batch)}/{len(batch)} agents ≥1 span, 0 error. Fail → stop, report the task id."
        )
        out.append("\n".join(block))
        done += len(batch)
    return "\n\n---\n\n".join(out)


def _render_finish_tasks(facts: PlanFacts) -> str:
    """Final milestone — full run, then hand the ledger back to AEH."""
    m = _BATCH_FIRST_MILESTONE + _batch_count(facts)
    return "\n".join([
        f"## M{m} — Run & hand back",
        f"> Điều kiện vào: [x] 🛑 GATE M{m - 1} (every agent batch green).",
        "",
        "- [ ] T900 Run the full evaluation: `python .aeh/run_eval.py`",
        "      Verify: `.aeh/out/manifest.json` exists and reports 0 errored cases",
        "- [ ] T901 Hand back to AEH: the manifest path, this TASKS.md (with your ticks and any",
        "      `[Drift]` / `[Conflict]` notes), and the RECON line.",
        "",
        f"🛑 GATE M{m}: manifest exists and every non-skipped agent produced ≥1 span.",
    ])


def _resolve_code(arg: str, facts: PlanFacts) -> str:
    """Resolve {{code:name}} — a fenced code block from a compiled artifact."""
    code = facts.code_artifacts.get(arg)
    if code is None:
        return f"[UNKNOWN_CODE: {arg}]"
    if not code:
        return "(not applicable for this target)"
    lang = "json" if arg == "wiring" else "python"
    return f"```{lang}\n{code}\n```"


def _resolve_sha256(arg: str, facts: PlanFacts) -> str:
    """Resolve {{sha256:name}} — the sha256 of a compiled code artifact."""
    return facts.code_shas.get(arg, "") or "(n/a)"
