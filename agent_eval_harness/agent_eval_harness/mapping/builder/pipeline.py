"""SystemMapBuilder: orchestrates all 6 passes to generate a system_map.yaml."""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.mapping.system_map import CallSiteRecord, Component, SpanMatchBlock, SystemMap

from agent_eval_harness.discovery.wiring import (
    CallSite,
    NodeCallTarget,
    StepResolution,
    WiringBlock,
    WiringEdge,
    WiringNode,
    resolve_generic_call_closure,
    resolve_langgraph_node_calls_full,
    resolve_unscanned_steps,
    wiring_identity,
)

from .boundary_llm import ResidueCandidate, decide_residue_verdicts
from .constraints import mine_constraints, mine_constraints_llm_phase
from .resolution_primitives import (
    DECLARED_BOUNDARY,
    DERIVED_BOUNDARY,
    LLM_UNVERIFIED_DEFAULT,
    STRUCTURAL_CLOSURE,
)
from .roles import ROLE_CONFIDENCE_THRESHOLD
from .scanners import FrameworkScanner, get_scanner, scan_all
from .topology import extract_topology
from .types import CandidateComponent, TopologyEdges, parse_python_source

logger = logging.getLogger("agent_eval_harness.mapping.builder.pipeline")


@dataclass
class _ResidueEvidence:
    call_sites: list[CallSite]
    targets: list[NodeCallTarget]


@dataclass
class _ModelCallVerdict:
    makes_model_call: bool | None
    source: str | None
    citation: str | None
    residue: "_ResidueEvidence | None" = None


class SystemMapBuilder:
    """Build a system_map.yaml from a target's source code."""

    def __init__(
        self, llm_client: LLMClient, *, confidence_threshold: float = ROLE_CONFIDENCE_THRESHOLD,
        framework: str | None = None,
    ) -> None:
        # Local import: discovery.engine loads mapping.builder.system_types at module scope, which
        # (via mapping/__init__.py importing this module) would otherwise be a load-time cycle.
        from agent_eval_harness.discovery.engine import load_provider_boundaries

        self._llm_client = llm_client
        self._threshold = confidence_threshold
        self._provider_boundaries = load_provider_boundaries()
        # A caller-declared pure framework (from a per-system candidate) is authoritative for the map
        # label — it must win over scan_all's union of frameworks that merely CO-LOCATE in the file
        # set, so a per-system map is never mislabeled 'a+b'. None => fall back to the scan union.
        self._explicit_framework = framework if framework and "+" not in framework else None
        self._scanner: FrameworkScanner = get_scanner(framework)

    async def build(
        self, target_path: Path, docs_path: Path | None = None
    ) -> tuple[SystemMap, str]:
        """Build system map through all 6 passes."""
        # Discover source files (track original target files for filtering)
        target = Path(target_path)
        if not target.is_dir():
            target = target.parent
        original_target_files = set(target.glob("**/*.py"))

        files, composed_import_names = self._discover_source_files(target_path)

        # Pass 1: Scan with every registered scanner (mixed clusters), keeping only candidates
        # defined inside target dir or referenced compositionally.
        all_candidates, framework_label = scan_all(files)
        candidates = [
            c for c in all_candidates
            if c.file in original_target_files or c.class_name in composed_import_names
        ]

        package_root = self._find_package_root(target_path)
        return await self._build_from_candidates(
            candidates, files, package_root, target_path.name, docs_path,
            framework_label=framework_label,
        )

    async def build_from_files(
        self, files: list[Path], package_root: Path, target_system_id: str,
        docs_path: Path | None = None, wiring_block: WiringBlock | None = None,
        scope_framework: str | None = None, exclude_component_classes: set[str] | None = None,
        retrieval_client: object | None = None, snapshot_id: str | None = None,
        resolver_skip_import_for: frozenset[str] = frozenset(),
        system_type: str | None = None,
    ) -> tuple[SystemMap, str]:
        """Same as build(), but the caller supplies the exact file set directly instead of a directory
        glob. When a community was split into per-system candidates, `scope_framework` keeps only the
        components produced by that system's scanner (so co-located sibling classes don't bleed in),
        and `exclude_component_classes` drops any class a sibling's wiring_block already claims (for the
        wireless plain-python candidate). Both default off => a non-split candidate keeps every
        component, unchanged.

        For a plain-python system, `retrieval_client` + `snapshot_id` (both optional) let the CS-318
        unscanned-step resolver find the DEFINING file of a step composed from an imported helper or an
        inherited base method and admit it to the scan set — so the step becomes a real component
        instead of vanishing. Absent => resolver runs only its free static layers (still resolves
        importable/inherited steps whose files are on disk) and skips retrieval. `resolver_skip_import_for`
        forces a named callee past the import layer (acceptance-harness only, to exercise live RRF)."""
        unresolved: list[StepResolution] = []
        if wiring_block is not None and "plain_python" in (wiring_block.framework or ""):
            resolutions = await resolve_unscanned_steps(
                wiring_block, files, package_root, retrieval_client, snapshot_id,
                skip_import_layer_for=resolver_skip_import_for,
            )
            file_set = set(files)
            for r in resolutions:
                if r.defining_file:
                    admitted = package_root / r.defining_file
                    if admitted.is_file() and admitted not in file_set:
                        files = files + [admitted]
                        file_set.add(admitted)
                        logger.info(
                            "resolver admitted %s for step '%s' via %s layer",
                            r.defining_file, r.callee, r.layer,
                        )
                else:
                    unresolved.append(r)

        # CS-319: a LangGraph node method delegates to an intra-node call graph (the "gray" layer). Walk
        # each node body (+ transitively the agent's own-class helpers), resolve every repo-local call to
        # its defining file via free AST layers, and inject the resolved targets as call-target components
        # + node->target "call" edges so the map carries the node's real dependencies, not a bare skeleton.
        langgraph_call_targets: list[NodeCallTarget] = []
        closure_info: dict[str, tuple[bool, list[CallSite], list[NodeCallTarget]]] = {}
        if wiring_block is not None and "langgraph" in (wiring_block.framework or ""):
            langgraph_call_targets, call_sites_by_key, complete_by_key, targets_by_key = (
                await resolve_langgraph_node_calls_full(
                    wiring_block, package_root, retrieval_client, snapshot_id,
                )
            )
            closure_info.update({
                key: (complete_by_key[key], call_sites_by_key.get(key, []), targets_by_key.get(key, []))
                for key in complete_by_key
            })

        # scan_all sees the ORIGINAL block only — augmenting it first would make PlainPythonScanner
        # re-emit the call targets as plain_python candidates (wrong framework tag, then scope-dropped).
        all_candidates, framework_label = scan_all(files, wiring_block=wiring_block)
        if langgraph_call_targets:
            all_candidates = all_candidates + self._call_target_candidates(
                langgraph_call_targets, package_root, {c.candidate_id for c in all_candidates},
                framework="langgraph",
            )
            # Augment the block AFTER scanning so topology builds the node->target "call" edges.
            wiring_block = self._augment_langgraph_wiring(wiring_block, langgraph_call_targets)
        if scope_framework:
            all_candidates = [c for c in all_candidates if c.framework == scope_framework]
        if exclude_component_classes:
            all_candidates = [c for c in all_candidates if c.class_name not in exclude_component_classes]

        # The SAME gray-layer call closure, generalized to any framework whose components expose a
        # recognized entry method — LangGraph's own node methods are excluded here naturally, since
        # their names aren't in the recognized set.
        generic_targets, call_sites_by_caller, complete_by_caller, targets_by_caller = (
            await resolve_generic_call_closure(
                [c for c in all_candidates if c.framework in ("haystack", "plain_python", "langchain")],
                package_root, retrieval_client, snapshot_id,
            )
        )
        closure_info.update({
            key: (complete_by_caller[key], call_sites_by_caller.get(key, []), targets_by_caller.get(key, []))
            for key in complete_by_caller
        })
        generic_call_pairs: list[tuple[str, str]] = []
        if generic_targets:
            all_candidates = all_candidates + self._call_target_candidates(
                generic_targets, package_root, {c.candidate_id for c in all_candidates},
                framework=scope_framework or framework_label or "unknown",
            )
            generic_call_pairs = [(t.caller_alias, t.callee) for t in generic_targets]

        system_map, summary = await self._build_from_candidates(
            all_candidates, files, package_root, target_system_id, docs_path, wiring_block,
            framework_label=framework_label,
            system_type=system_type,
            generic_call_pairs=generic_call_pairs,
            closure_info=closure_info,
        )
        if unresolved:
            self._append_needs_human_components(system_map, unresolved)
            summary = self._build_summary(system_map)
        return system_map, summary

    def _append_needs_human_components(
        self, system_map: SystemMap, unresolved: list[StepResolution]
    ) -> None:
        """A wiring step no resolver layer could place still must not vanish: emit a placeholder
        component flagged needs_human so the call-flow shape survives and the gap is reviewable. Skips
        any step a scanner did resolve after all (defensive)."""
        existing = {c.id for c in system_map.components}
        for r in unresolved:
            comp_id = (r.alias or r.callee).lower()
            if comp_id in existing:
                continue
            existing.add(comp_id)
            system_map.components.append(Component.model_validate({
                "id": comp_id,
                "role": "unknown",
                "entry_point": "",
                "entry_kind": r.entry_kind,
                "needs_human": True,
                "file": "",
            }))

    def _augment_langgraph_wiring(
        self, wiring_block: WiringBlock, targets: list[NodeCallTarget]
    ) -> WiringBlock:
        """Return a copy of the LangGraph wiring block with the resolved intra-node call targets added
        as nodes and node/helper->target "call" edges — so topology builds the gray call graph. A
        target invoked by >=2 callers is ONE node (dedup by alias) with multiple inbound call edges."""
        nodes = {n.alias: n for n in wiring_block.nodes}
        edges = list(wiring_block.edges)
        existing = {(e.src, e.dst) for e in edges}
        for t in targets:
            nodes.setdefault(t.alias, WiringNode(
                alias=t.alias, callee_name=t.callee, source_hint_file=t.defining_file,
                entry_kind=t.entry_kind, owner_class=t.owner_class, framework="langgraph",
            ))
            if (t.caller_alias, t.alias) not in existing:
                existing.add((t.caller_alias, t.alias))
                edges.append(WiringEdge(src=t.caller_alias, dst=t.alias, kind="call"))
        return WiringBlock(
            nodes=list(nodes.values()), edges=edges,
            framework=wiring_block.framework, source=wiring_block.source,
        )

    def _call_target_candidates(
        self, targets: list[NodeCallTarget], package_root: Path, existing_ids: set[str], framework: str,
    ) -> list[CandidateComponent]:
        """Emit one CandidateComponent per unique resolved gray call target (no scanner emits a bare
        module fn / injected service class) — shared by the LangGraph-specific path and the generic
        one, tagged with whichever framework the caller resolved these targets under. Deduped by
        candidate_id so a shared hub is ONE component; skips any id a scanner already produced
        (defensive)."""
        out: list[CandidateComponent] = []
        seen: set[str] = set(existing_ids)
        for t in targets:
            cand = CandidateComponent(
                file=package_root / t.defining_file, line=t.line, class_name=t.callee,
                entry_kind=t.entry_kind, owner_class_name=t.owner_class, framework=framework,
            )
            cid = cand.candidate_id
            if cid in seen:
                continue
            seen.add(cid)
            out.append(cand)
        return out

    async def _build_from_candidates(
        self,
        candidates: list[CandidateComponent],
        files: list[Path],
        package_root: Path,
        target_system_id: str,
        docs_path: Path | None,
        wiring_block: WiringBlock | None = None,
        framework_label: str | None = None,
        system_type: str | None = None,
        generic_call_pairs: list[tuple[str, str]] | None = None,
        closure_info: dict[str, tuple[bool, list[CallSite], list[NodeCallTarget]]] | None = None,
    ) -> tuple[SystemMap, str]:
        """Passes 2-6: structural mining through system map assembly, shared by build() and
        build_from_files(). Role classification moved to Stage 2.5 enrichment — every component
        leaves Stage 2 with role='unknown'; is_tool/constructor_fanout are persisted as data for
        Stage 2.5's hard gate. framework_label is the union-scan's contributed-frameworks string;
        when absent (direct callers), fall back to the advisory single-scanner framework.
        `closure_info` is keyed the same way class_to_candidate_ids is: bound-method candidates by
        wiring_identity(owner, callee), everything else by class_name."""
        topology_map = extract_topology(
            files, candidates, wiring_block=wiring_block, generic_call_pairs=generic_call_pairs,
        )
        constraint_map = mine_constraints(files, candidates, package_root)
        constraint_map = await mine_constraints_llm_phase(
            candidates, self._llm_client, constraint_map
        )
        components, residue_candidates = self._assemble_components(
            candidates, topology_map, constraint_map, package_root, closure_info or {},
        )
        await decide_residue_verdicts(
            residue_candidates, components, package_root, self._llm_client, target_system_id,
        )
        discrepancies = await self._reconcile_docs(docs_path, components) if docs_path else []
        # Framework precedence: a supplied pure wiring_block (the system's ground truth) > a caller-
        # declared pure framework > scan_all's co-location union > the advisory single scanner. This
        # keeps each per-system map single-framework even when other systems' classes co-locate.
        if wiring_block is not None and wiring_block.framework and "+" not in wiring_block.framework:
            resolved_framework = wiring_block.framework
        elif self._explicit_framework:
            resolved_framework = self._explicit_framework
        else:
            resolved_framework = framework_label if framework_label else self._scanner.framework
        system_map = SystemMap.model_validate({
            "target_system_id": target_system_id,
            "components": [c.model_dump() for c in components],
            "framework": resolved_framework,
            "system_type": system_type,
            "discrepancies": discrepancies,
        })
        summary = self._build_summary(system_map)
        return system_map, summary

    def _discover_source_files(self, target_path: Path) -> tuple[list[Path], set[str]]:
        """Find all .py files in target directory, following only composed imports."""
        target = Path(target_path)
        if not target.is_dir():
            target = target.parent

        # Initial glob
        files = set(target.glob("**/*.py"))
        package_root = self._find_package_root(target)
        resolved_files = set(files)

        # Collect imports and constructor annotations, mapping re-exported module paths deterministically.
        imported_names: dict[str, set[str]] = {}  # {name: {module_path, ...}}
        imported_in_composed_context: set[str] = set()

        for file in sorted(files):
            parsed = parse_python_source(file)
            if parsed is None:
                continue
            _, tree = parsed

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.names:
                        for alias in node.names:
                            import_name = alias.asname or alias.name
                            imported_names.setdefault(import_name, set()).add(node.module)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                            for arg in item.args.args:
                                if arg.annotation:
                                    for name_node in ast.walk(arg.annotation):
                                        if isinstance(name_node, ast.Name):
                                            if name_node.id in imported_names:
                                                imported_in_composed_context.add(
                                                    name_node.id
                                                )

        # Second pass: resolve every module path seen for each composed import (not just one)
        for import_name, module_paths in sorted(imported_names.items()):
            if import_name not in imported_in_composed_context:
                continue

            for module_path in sorted(module_paths):
                module_parts = module_path.split(".")
                # Try as package first (module/__init__.py)
                potential_file = package_root
                for part in module_parts:
                    potential_file = potential_file / part
                init_file = potential_file / "__init__.py"
                if init_file.exists():
                    resolved_files.add(init_file)
                # Try as module (.py file)
                py_file = potential_file.with_suffix(".py")
                if py_file.exists():
                    resolved_files.add(py_file)

        return sorted(resolved_files), set(imported_in_composed_context)

    def _find_package_root(self, target_path: Path) -> Path:
        """Find the first ancestor WITHOUT __init__.py."""
        current = target_path if target_path.is_dir() else target_path.parent
        while (current / "__init__.py").exists():
            current = current.parent
        return current

    def _resolve_entry_point(self, candidate: CandidateComponent, package_root: Path) -> str:
        """Resolve entry_point as module:class_name."""
        try:
            rel_path = candidate.file.relative_to(package_root)
        except ValueError:
            rel_path = candidate.file

        module_path = rel_path.with_suffix("").as_posix().replace("/", ".")
        suffix = (
            wiring_identity(candidate.owner_class_name, candidate.class_name)
            if candidate.entry_kind == "bound_method"
            else candidate.class_name
        )
        return f"{module_path}:{suffix}"

    def _generate_span_match(
        self, candidate: CandidateComponent, all_candidates: list[CandidateComponent]
    ) -> list[SpanMatchBlock]:
        """Generate span_match blocks for a candidate."""
        # Tool candidates: match by aeh.tool.name tag
        if candidate.is_tool:
            return [
                SpanMatchBlock(
                    tags={"aeh.tool.name": candidate.registered_name}
                )
            ]

        if candidate.tag_suffix is not None:
            if candidate.manual_span_hints:
                hint = candidate.manual_span_hints[0]
                for key, value in hint.tags.items():
                    if value == candidate.tag_suffix:
                        return [
                            SpanMatchBlock(
                                component_name=hint.component_name,
                                tags={key: value}
                            )
                        ]
            # Fallback
            return [SpanMatchBlock(component_name=candidate.class_name.lower())]

        # Regular candidates: check if they own tool candidates
        if candidate.manual_span_hints:
            owns_tools = False
            for tool_candidate in all_candidates:
                if tool_candidate.is_tool:
                    if tool_candidate.owner_class_name == candidate.class_name:
                        owns_tools = True
                        break

            if owns_tools:
                for hint in candidate.manual_span_hints:
                    is_string_literal = (
                        hint.op_name.startswith("'") and hint.op_name.endswith("'")
                    ) or (hint.op_name.startswith('"') and hint.op_name.endswith('"'))
                    if is_string_literal:
                        # It's a string literal, not a Name reference to a shared constant
                        op_name = hint.op_name[1:-1]
                        return [
                            SpanMatchBlock(
                                span_name_pattern=f"^{re.escape(op_name)}$"
                            )
                        ]

            # Otherwise use component_name
            if candidate.manual_span_hints:
                return [
                    SpanMatchBlock(
                        component_name=candidate.manual_span_hints[0].component_name
                    )
                ]

        # Fallback: use haystack_name or class_name.lower()
        component_name = candidate.haystack_name or candidate.class_name.lower()
        return [SpanMatchBlock(component_name=component_name)]

    def _compute_model_call_verdict(
        self, candidate: CandidateComponent, rel_file: str,
        call_sites: list[CallSite], targets: list[NodeCallTarget], closure_walked: bool,
    ) -> _ModelCallVerdict:
        """Tiers, all static: (1) a Resolved callee that is a declared/derived provider boundary ->
        True, citing the resolving file:line; a closure with no Unresolved site and no boundary
        reached -> False. A candidate whose closure was never walked at all (`closure_walked` False)
        or whose split-candidate span has no bounded evidence region stays terminally undecided --
        zero call sites then means "did not look", not "looked and found nothing". Anything else
        neither tier decides is handed to the caller as residue: a closure that DID resolve enough
        to walk, but stopped at a genuinely unresolved call with no boundary reached anywhere in it.
        For a split candidate (guard_rule/guard_llm sharing one entry method), evidence is scoped to
        its OWN manual_span line range first, so the two siblings can receive opposite verdicts."""
        from agent_eval_harness.discovery.engine import (
            match_provider_boundary_entry,
            match_provider_boundary_import,
        )

        if not closure_walked:
            return _ModelCallVerdict(None, None, None)
        scoped_sites = call_sites
        scoped_targets = targets
        if candidate.tag_suffix is not None:
            bounded = [h for h in candidate.manual_span_hints if h.end_line]
            if not bounded:
                return _ModelCallVerdict(None, None, None)
            lo, hi = min(h.line for h in bounded), max(h.end_line for h in bounded)
            scoped_sites = [cs for cs in call_sites if cs.file == rel_file and lo <= cs.lineno <= hi]
            scoped_site_ids = {cs.site_id for cs in scoped_sites}
            scoped_targets = [t for t in targets if t.site_id in scoped_site_ids]

        for t in scoped_targets:
            row = match_provider_boundary_entry(
                self._provider_boundaries, defining_file=t.defining_file,
                owner_class=t.owner_class, resolved_symbol=t.resolved_symbol,
            )
            if row is not None:
                return _ModelCallVerdict(True, DERIVED_BOUNDARY, f"{t.defining_file}:{t.line}")
        for cs in scoped_sites:
            if cs.outcome == "out_of_scope" and cs.reason == "external_import":
                row = match_provider_boundary_import(self._provider_boundaries, cs.external_module)
                if row is not None:
                    return _ModelCallVerdict(True, DECLARED_BOUNDARY, f"{cs.file}:{cs.lineno}")

        if not any(cs.outcome == "unresolved" for cs in scoped_sites):
            return _ModelCallVerdict(False, STRUCTURAL_CLOSURE, None)
        return _ModelCallVerdict(None, None, None, residue=_ResidueEvidence(scoped_sites, scoped_targets))

    def _assemble_components(
        self,
        candidates: list[CandidateComponent],
        topology_map: dict[str, TopologyEdges],
        constraint_map: dict[str, list],
        package_root: Path,
        closure_info: dict[str, tuple[bool, list[CallSite], list[NodeCallTarget]]] | None = None,
    ) -> tuple[list[Component], list[ResidueCandidate]]:
        """Assemble final Component objects. role='unknown' -- Stage 2.5 enrichment assigns it;
        is_tool/constructor_fanout are persisted as the structural evidence it gates on. Second
        return value: components whose closure resolved enough to walk but stopped at a genuinely
        unresolved call with no boundary reached anywhere in it -- the caller runs the LLM residue
        pass over these, async, after this method returns."""
        closure_info = closure_info or {}
        components = []
        residue_candidates: list[ResidueCandidate] = []
        for candidate in candidates:
            entry_point = self._resolve_entry_point(candidate, package_root)
            span_match = self._generate_span_match(candidate, candidates)
            constraints = constraint_map.get(candidate.candidate_id, [])
            topology = topology_map.get(candidate.candidate_id, TopologyEdges())

            try:
                rel_file = candidate.file.relative_to(package_root).as_posix()
            except ValueError:
                rel_file = candidate.file.as_posix() if candidate.file else ""

            closure_key = (
                wiring_identity(candidate.owner_class_name, candidate.class_name)
                if candidate.entry_kind == "bound_method"
                else candidate.class_name
            )
            closure = closure_info.get(closure_key)
            closure_complete = closure[0] if closure else None
            call_sites_raw = closure[1] if closure else []
            call_sites = [
                CallSiteRecord(
                    site_id=cs.site_id, file=cs.file, lineno=cs.lineno,
                    callee_text=cs.callee_text, source_line=cs.source_line,
                    outcome=cs.outcome, reason=cs.reason, external_module=cs.external_module,
                )
                for cs in call_sites_raw
            ]
            verdict = self._compute_model_call_verdict(
                candidate, rel_file, call_sites_raw, closure[2] if closure else [], closure is not None,
            )
            if verdict.makes_model_call is None and closure is not None and verdict.residue is None:
                logger.debug(
                    "%s: makes_model_call undecided (unresolved call site in closure) -- defaulted to step",
                    candidate.candidate_id,
                )
                verdict = _ModelCallVerdict(False, LLM_UNVERIFIED_DEFAULT, None)

            # Infra-helper flagging (fix-plan #15) REVERTED to pass-through: a fanout-only rule
            # falsely flags single/isolated real agents (e.g. a lone QA agent with fan 0), and
            # the prompt-site/LLM-verb evidence needed for a safe rule isn't available here.
            is_lib = candidate.is_library_object

            component = Component(
                id=candidate.candidate_id,
                role="unknown",
                role_confidence=None,
                role_source=None,
                is_tool=candidate.is_tool,
                is_library_object=is_lib,
                constructor_fanout=len(topology.constructor_downstream),
                constructor_downstream=list(topology.constructor_downstream),
                model=None,
                entry_point=entry_point,
                entry_kind=candidate.entry_kind,
                file=rel_file,
                span_match=span_match,
                constraints=constraints,
                upstream=topology.upstream,
                downstream=topology.downstream,
                call_downstream=list(topology.call_downstream),
                conditional_downstream=list(topology.conditional_downstream),
                motif=topology.motif,
                closure_complete=closure_complete,
                call_sites=call_sites,
                makes_model_call=verdict.makes_model_call,
                model_call_source=verdict.source,
                model_call_citation=verdict.citation,
            )
            components.append(component)
            if verdict.residue is not None:
                residue_candidates.append(ResidueCandidate(
                    component_id=candidate.candidate_id, rel_file=rel_file, entry_line=candidate.line,
                    call_sites=verdict.residue.call_sites, targets=verdict.residue.targets,
                ))

        return components, residue_candidates

    def _build_summary(self, system_map: SystemMap) -> str:
        """Build a fixed-header summary. Role is pending Stage 2.5 enrichment — Stage 2 no
        longer classifies it, so an all-'unknown' map here is expected, not a failure."""
        if not system_map.components:
            return (
                "=== AEH System Map Summary ===\n"
                f"target: {system_map.target_system_id}\n"
                "no agentic components identified\n"
            )

        total = len(system_map.components)
        unknown_count = sum(1 for c in system_map.components if c.role == "unknown")
        library_object_count = sum(1 for c in system_map.components if c.is_library_object)

        summary = "=== AEH System Map Summary ===\n"
        summary += f"target:           {system_map.target_system_id}\n"
        summary += f"components_found: {total}\n"
        summary += "role:             pending Stage 2.5 enrichment\n"
        summary += f"unknown:          {unknown_count}  (role classification happens in Stage 2.5, not here)\n"
        summary += f"library_objects:  {library_object_count}  (explicit degrade - framework/library links in a chain, never harvestable)\n"
        summary += f"discrepancies:    {len(system_map.discrepancies)}\n"

        return summary

    async def _reconcile_docs(
        self, docs_path: Path | None, components: list[Component]
    ) -> list[str]:
        """Check if discovered components are mentioned in the hand-written docs."""
        if not docs_path or not docs_path.exists():
            return []
        try:
            content = docs_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Could not read docs file %s: %s", docs_path, e)
            return [f"Could not read docs file '{docs_path.name}': {e}"]

        content_lower = content.lower()
        discrepancies = []
        for comp in components:
            if comp.id.lower() not in content_lower:
                discrepancies.append(
                    f"Component '{comp.id}' is discovered in code but not mentioned "
                    f"in the documentation '{docs_path.name}'"
                )
        return discrepancies
