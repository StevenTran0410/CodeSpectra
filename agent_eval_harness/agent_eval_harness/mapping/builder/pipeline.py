"""SystemMapBuilder: orchestrates all 6 passes to generate a system_map.yaml."""
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.mapping.system_map import Component, SpanMatchBlock, SystemMap

from agent_eval_harness.discovery.wiring import WiringBlock, wiring_identity

from .constraints import mine_constraints, mine_constraints_llm_phase
from .roles import ROLE_CONFIDENCE_THRESHOLD
from .scanners import FrameworkScanner, get_scanner, scan_all
from .topology import extract_topology
from .types import CandidateComponent, TopologyEdges, parse_python_source

logger = logging.getLogger("agent_eval_harness.mapping.builder.pipeline")


class SystemMapBuilder:
    """Build a system_map.yaml from a target's source code."""

    def __init__(
        self, llm_client: LLMClient, *, confidence_threshold: float = ROLE_CONFIDENCE_THRESHOLD,
        framework: str | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._threshold = confidence_threshold
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
    ) -> tuple[SystemMap, str]:
        """Same as build(), but the caller supplies the exact file set directly instead of a directory
        glob. When a community was split into per-system candidates, `scope_framework` keeps only the
        components produced by that system's scanner (so co-located sibling classes don't bleed in),
        and `exclude_component_classes` drops any class a sibling's wiring_block already claims (for the
        wireless plain-python candidate). Both default off => a non-split candidate keeps every
        component, unchanged."""
        all_candidates, framework_label = scan_all(files, wiring_block=wiring_block)
        if scope_framework:
            all_candidates = [c for c in all_candidates if c.framework == scope_framework]
        if exclude_component_classes:
            all_candidates = [c for c in all_candidates if c.class_name not in exclude_component_classes]
        return await self._build_from_candidates(
            all_candidates, files, package_root, target_system_id, docs_path, wiring_block,
            framework_label=framework_label,
        )

    async def _build_from_candidates(
        self,
        candidates: list[CandidateComponent],
        files: list[Path],
        package_root: Path,
        target_system_id: str,
        docs_path: Path | None,
        wiring_block: WiringBlock | None = None,
        framework_label: str | None = None,
    ) -> tuple[SystemMap, str]:
        """Passes 2-6: structural mining through system map assembly, shared by build() and
        build_from_files(). Role classification moved to Stage 2.5 enrichment — every component
        leaves Stage 2 with role='unknown'; is_tool/constructor_fanout are persisted as data for
        Stage 2.5's hard gate. framework_label is the union-scan's contributed-frameworks string;
        when absent (direct callers), fall back to the advisory single-scanner framework."""
        topology_map = extract_topology(files, candidates, wiring_block=wiring_block)
        constraint_map = mine_constraints(files, candidates, package_root)
        constraint_map = await mine_constraints_llm_phase(
            candidates, self._llm_client, constraint_map
        )
        components = self._assemble_components(
            candidates, topology_map, constraint_map, package_root
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

    def _assemble_components(
        self,
        candidates: list[CandidateComponent],
        topology_map: dict[str, TopologyEdges],
        constraint_map: dict[str, list],
        package_root: Path,
    ) -> list[Component]:
        """Assemble final Component objects. role='unknown' — Stage 2.5 enrichment assigns it;
        is_tool/constructor_fanout are persisted as the structural evidence it gates on."""
        components = []
        for candidate in candidates:
            entry_point = self._resolve_entry_point(candidate, package_root)
            span_match = self._generate_span_match(candidate, candidates)
            constraints = constraint_map.get(candidate.candidate_id, [])
            topology = topology_map.get(candidate.candidate_id, TopologyEdges())

            # Model hints
            model = candidate.model_hints[0] if candidate.model_hints else None

            try:
                rel_file = candidate.file.relative_to(package_root).as_posix()
            except ValueError:
                rel_file = candidate.file.as_posix() if candidate.file else ""

            component = Component(
                id=candidate.candidate_id,
                role="unknown",
                role_confidence=None,
                role_source=None,
                is_tool=candidate.is_tool,
                is_library_object=candidate.is_library_object,
                constructor_fanout=len(topology.constructor_downstream),
                constructor_downstream=list(topology.constructor_downstream),
                model=model,
                entry_point=entry_point,
                entry_kind=candidate.entry_kind,
                file=rel_file,
                span_match=span_match,
                constraints=constraints,
                upstream=topology.upstream,
                downstream=topology.downstream,
            )
            components.append(component)

        return components

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
