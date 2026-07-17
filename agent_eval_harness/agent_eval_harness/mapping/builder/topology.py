"""Pass 3: Extract topology edges from source code (sync, no LLM)."""
from __future__ import annotations

import ast
import logging
from pathlib import Path

from agent_eval_harness.mapping.builder.types import (
    CandidateComponent,
    TopologyEdges,
    parse_python_source,
)

logger = logging.getLogger("agent_eval_harness.mapping.builder.topology")


def extract_topology(
    source_files: list[Path],
    candidates: list[CandidateComponent],
) -> dict[str, TopologyEdges]:
    """Extract topology edges from source code."""
    connect_edges: dict[str, set[str]] = {}
    constructor_edges: dict[str, set[str]] = {}

    # Build mapping from class_name to all candidate_ids for that class (including splits)
    class_to_candidate_ids: dict[str, list[str]] = {}
    for candidate in candidates:
        if candidate.class_name not in class_to_candidate_ids:
            class_to_candidate_ids[candidate.class_name] = []
        class_to_candidate_ids[candidate.class_name].append(candidate.candidate_id)

    # Cache ASTs to avoid re-parsing files multiple times across three passes
    asts: dict[Path, ast.Module] = {}
    for file in source_files:
        parsed = parse_python_source(file)
        if parsed is not None:
            asts[file] = parsed[1]

    # Phase A: Extract connect() edges using unified wiring detector. Read separately from the
    # ast cache above — the wiring detector works on raw text and must still see files that fail
    # to ast.parse.
    file_contents = {}
    for file in source_files:
        try:
            file_contents[str(file)] = file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("skipping unreadable file %s: %s", file, exc)
            continue

    from agent_eval_harness.discovery.wiring import detect_wiring_block_static
    wiring_block = detect_wiring_block_static(file_contents)

    add_component_names: dict[str, str] = {}
    if wiring_block:
        for w_node in wiring_block.nodes:
            add_component_names[w_node.alias] = w_node.class_name

        for w_edge in wiring_block.edges:
            source_class_name = add_component_names.get(w_edge.src)
            dest_class_name = add_component_names.get(w_edge.dst)

            if source_class_name and dest_class_name:
                source_ids = class_to_candidate_ids.get(source_class_name, [])
                dest_ids = class_to_candidate_ids.get(dest_class_name, [])

                # Apply edge to all variants of the class
                for src_id in source_ids:
                    for dest_id in dest_ids:
                        if src_id not in connect_edges:
                            connect_edges[src_id] = set()
                        connect_edges[src_id].add(dest_id)

    # Phase B: Extract constructor injection edges
    for file, tree in asts.items():
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

    # Populate constructor_downstream before the merge below discards edge-kind (orchestrator signal)
    for source, targets in constructor_edges.items():
        if source not in edges:
            edges[source] = TopologyEdges()
        edges[source].constructor_downstream = list(dict.fromkeys(targets))

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
