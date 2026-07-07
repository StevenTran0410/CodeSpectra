"""SystemMapBuilder: orchestrates all 6 passes to generate a system_map.yaml."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from agent_eval_harness.llm.client import LLMClient
from agent_eval_harness.mapping.system_map import Component, SpanMatchBlock, SystemMap

from .constraints import mine_constraints, mine_constraints_llm_phase
from .roles import ROLE_CONFIDENCE_THRESHOLD, classify_roles
from .scanners import FrameworkScanner, HaystackScanner
from .topology import extract_topology
from .types import CandidateComponent, RoleClassification, TopologyEdges


class SystemMapBuilder:
    """Build a system_map.yaml from a target's source code."""

    def __init__(
        self, llm_client: LLMClient, *, confidence_threshold: float = ROLE_CONFIDENCE_THRESHOLD
    ) -> None:
        self._llm_client = llm_client
        self._threshold = confidence_threshold
        self._scanner: FrameworkScanner = HaystackScanner()

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

        # Pass 1: Scan
        all_candidates = self._scanner.scan(files)

        # Filter candidates: keep defined inside target dir or referenced compositionally.
        candidates = [
            c for c in all_candidates
            if c.file in original_target_files or c.class_name in composed_import_names
        ]

        # Pass 2: Classify roles
        role_classifications = await classify_roles(candidates, self._llm_client, self._threshold)

        # Pass 3: Extract topology
        topology_map = extract_topology(files, candidates)

        # Pass 4a: Mine static constraints
        package_root = self._find_package_root(target_path)
        constraint_map = mine_constraints(files, candidates, package_root)

        # Pass 4b: Mine LLM-based constraints
        constraint_map = await mine_constraints_llm_phase(
            candidates, self._llm_client, constraint_map
        )

        # Assemble components
        components = self._assemble_components(
            candidates, role_classifications, topology_map, constraint_map, package_root
        )

        # Reconcile docs (if provided)
        discrepancies = await self._reconcile_docs(docs_path, components) if docs_path else []

        # Create and validate system map
        system_map = SystemMap.model_validate({
            "target_system_id": target_path.name,
            "components": [c.model_dump() for c in components],
            "discrepancies": discrepancies,
        })

        summary = self._build_summary(system_map, role_classifications)

        return system_map, summary

    async def build_from_files(
        self, files: list[Path], package_root: Path, target_system_id: str, docs_path: Path | None = None
    ) -> tuple[SystemMap, str]:
        """Same as build(), but the caller supplies the exact file set directly instead of a directory glob."""
        all_candidates = self._scanner.scan(files)

        role_classifications = await classify_roles(all_candidates, self._llm_client, self._threshold)
        topology_map = extract_topology(files, all_candidates)
        constraint_map = mine_constraints(files, all_candidates, package_root)
        constraint_map = await mine_constraints_llm_phase(
            all_candidates, self._llm_client, constraint_map
        )
        components = self._assemble_components(
            all_candidates, role_classifications, topology_map, constraint_map, package_root
        )
        discrepancies = await self._reconcile_docs(docs_path, components) if docs_path else []
        system_map = SystemMap.model_validate({
            "target_system_id": target_system_id,
            "components": [c.model_dump() for c in components],
            "discrepancies": discrepancies,
        })
        summary = self._build_summary(system_map, role_classifications)
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
            try:
                source = file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(file))
            except (SyntaxError, OSError):
                continue

            # Collect ImportFrom statements
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.names:
                        for alias in node.names:
                            import_name = alias.asname or alias.name
                            imported_names.setdefault(import_name, set()).add(node.module)

            # Check if any imported names are used in constructor annotations
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
        return f"{module_path}:{candidate.class_name}"

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

        # Split candidates: match by component_name + distinguishing tag
        if candidate.tag_suffix is not None:
            # Find the manual_span hint to get the component_name
            if candidate.manual_span_hints:
                hint = candidate.manual_span_hints[0]
                # Find the tag key whose value matches the tag_suffix
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
            # Check if any tool candidate is owned by this candidate's class
            owns_tools = False
            for tool_candidate in all_candidates:
                if tool_candidate.is_tool:
                    if tool_candidate.owner_class_name == candidate.class_name:
                        owns_tools = True
                        break

            if owns_tools:
                # Check for either single or double quote characters on string literals.
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
        role_classifications: list[RoleClassification],
        topology_map: dict[str, TopologyEdges],
        constraint_map: dict[str, list],
        package_root: Path,
    ) -> list[Component]:
        """Assemble final Component objects."""
        # Build lookup
        roles_by_id = {rc.candidate_id: rc for rc in role_classifications}

        components = []
        for candidate in candidates:
            role_class = roles_by_id.get(candidate.candidate_id)
            if not role_class:
                continue

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
                role=role_class.role,
                model=model,
                entry_point=entry_point,
                file=rel_file,
                span_match=span_match,
                constraints=constraints,
                upstream=topology.upstream,
                downstream=topology.downstream,
            )
            components.append(component)

        return components

    def _build_summary(
        self, system_map: SystemMap, role_classifications: list[RoleClassification]
    ) -> str:
        """Build a fixed-header summary."""
        if not system_map.components:
            return (
                "=== AEH System Map Summary ===\n"
                f"target: {system_map.target_system_id}\n"
                "no agentic components identified\n"
            )

        total = len(system_map.components)
        unknown_count = sum(
            1 for rc in role_classifications if rc.role == "unknown"
        )
        classified_count = total - unknown_count
        pct = int(100 * classified_count / total) if total > 0 else 0

        review_required = []
        for rc in role_classifications:
            if rc.confidence < 1.0:
                review_required.append(f"  {rc.candidate_id}   confidence={rc.confidence:.2f}")

        summary = "=== AEH System Map Summary ===\n"
        summary += f"target:           {system_map.target_system_id}\n"
        summary += f"components_found: {total}\n"
        summary += f"classified:       {classified_count}  ({pct}%)\n"
        summary += f"unknown:          {unknown_count}\n"
        summary += f"discrepancies:    {len(system_map.discrepancies)}\n"

        if review_required:
            summary += "review_required:\n"
            summary += "\n".join(review_required)
            summary += "\n"

        return summary

    async def _reconcile_docs(
        self, docs_path: Path | None, components: list[Component]
    ) -> list[str]:
        """Stub for doc reconciliation (not implemented for MVP)."""
        return []
