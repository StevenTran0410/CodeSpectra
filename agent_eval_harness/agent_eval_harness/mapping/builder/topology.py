"""Pass 3: Extract topology edges from source code (sync, no LLM)."""
from __future__ import annotations

import ast
from pathlib import Path

from agent_eval_harness.mapping.builder.types import CandidateComponent, TopologyEdges


def extract_topology(
    source_files: list[Path],
    candidates: list[CandidateComponent],
) -> dict[str, TopologyEdges]:
    """Extract topology edges from source code.

    Returns dict[candidate_id, TopologyEdges] for all candidates (including splits,
    which inherit the class's topology).
    """
    connect_edges: dict[str, set[str]] = {}
    constructor_edges: dict[str, set[str]] = {}

    # Build mapping from class_name to all candidate_ids for that class (including splits)
    class_to_candidate_ids: dict[str, list[str]] = {}
    for candidate in candidates:
        if candidate.class_name not in class_to_candidate_ids:
            class_to_candidate_ids[candidate.class_name] = []
        class_to_candidate_ids[candidate.class_name].append(candidate.candidate_id)

    # Phase A: Extract connect() edges
    # First pass: collect add_component_names mapping
    add_component_names: dict[str, str] = {}  # {haystack_name: class_name}

    for file in source_files:
        try:
            source = file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "add_component":
                    if (
                        len(node.args) >= 2
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and isinstance(node.args[1], ast.Call)
                        and isinstance(node.args[1].func, ast.Name)
                    ):
                        haystack_name = node.args[0].value
                        class_name = node.args[1].func.id
                        add_component_names[haystack_name] = class_name

    # Second pass: extract connect() calls
    for file in source_files:
        try:
            source = file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "connect":
                    if len(node.args) >= 2:
                        if (
                            isinstance(node.args[0], ast.Constant)
                            and isinstance(node.args[0].value, str)
                            and isinstance(node.args[1], ast.Constant)
                            and isinstance(node.args[1].value, str)
                        ):
                            source_str = node.args[0].value
                            dest_str = node.args[1].value

                            # Split on first '.'
                            source_parts = source_str.split(".", 1)
                            dest_parts = dest_str.split(".", 1)

                            if source_parts and dest_parts:
                                source_node_name = source_parts[0]
                                dest_node_name = dest_parts[0]

                                # Map through add_component_names
                                source_class_name = add_component_names.get(source_node_name)
                                dest_class_name = add_component_names.get(dest_node_name)

                                if source_class_name and dest_class_name:
                                    source_ids = class_to_candidate_ids.get(
                                        source_class_name, []
                                    )
                                    dest_ids = class_to_candidate_ids.get(
                                        dest_class_name, []
                                    )

                                    # Apply edge to all variants of the class
                                    for src_id in source_ids:
                                        for dest_id in dest_ids:
                                            if src_id not in connect_edges:
                                                connect_edges[src_id] = set()
                                            connect_edges[src_id].add(dest_id)

    # Phase B: Extract constructor injection edges
    for file in source_files:
        try:
            source = file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file))
        except SyntaxError:
            continue

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                # Find all candidate_ids for this class
                class_candidate_ids = class_to_candidate_ids.get(node.name, [])

                if not class_candidate_ids:
                    continue

                # Look at __init__ parameters
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        for arg in item.args.args:
                            if arg.annotation:
                                # Collect all Name nodes from the annotation
                                for name_node in ast.walk(arg.annotation):
                                    if isinstance(name_node, ast.Name):
                                        # Only include if this is a known candidate class
                                        injected_ids = class_to_candidate_ids.get(
                                            name_node.id, []
                                        )

                                        if injected_ids:
                                            # Add edge from all variants of this class
                                            for class_id in class_candidate_ids:
                                                for injected_id in injected_ids:
                                                    if class_id not in constructor_edges:
                                                        constructor_edges[class_id] = set()
                                                    constructor_edges[class_id].add(
                                                        injected_id
                                                    )

    # Phase C: Combine and invert to build full topology
    edges: dict[str, TopologyEdges] = {}

    # Collect all edges
    all_edges: dict[str, set[str]] = {}
    for source, targets in connect_edges.items():
        if source not in all_edges:
            all_edges[source] = set()
        all_edges[source].update(targets)

    for source, targets in constructor_edges.items():
        if source not in all_edges:
            all_edges[source] = set()
        all_edges[source].update(targets)

    # Initialize topology for all candidates
    for candidate in candidates:
        edges[candidate.candidate_id] = TopologyEdges()

    # Populate upstream/downstream
    for source, targets in all_edges.items():
        if source not in edges:
            edges[source] = TopologyEdges()
        for target in targets:
            if target not in edges:
                edges[target] = TopologyEdges()

            edges[source].downstream.append(target)
            edges[target].upstream.append(source)

    # Dedupe
    for topology in edges.values():
        topology.upstream = list(dict.fromkeys(topology.upstream))
        topology.downstream = list(dict.fromkeys(topology.downstream))

    return edges
