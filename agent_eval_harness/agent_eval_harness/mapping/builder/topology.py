"""Pass 3: Extract topology edges from source code (sync, no LLM)."""
from __future__ import annotations

import ast
import logging
from pathlib import Path

from agent_eval_harness.discovery.wiring import WiringBlock, wiring_identity
from agent_eval_harness.mapping.builder.types import (
    CandidateComponent,
    TopologyEdges,
    parse_python_source,
)

logger = logging.getLogger("agent_eval_harness.mapping.builder.topology")


def _merge_missing_wiring(base: WiringBlock, fresh: WiringBlock) -> WiringBlock:
    """Additively merge into `base` only the nodes (by alias) and edges (by src/dst) that `fresh`
    detection found and `base` lacks — so a stale/partial persisted block can't orphan scanned nodes.
    Returns `base` unchanged when nothing is missing."""
    base_aliases = {n.alias for n in base.nodes}
    base_edges = {(e.src, e.dst) for e in base.edges}
    extra_nodes = [n for n in fresh.nodes if n.alias not in base_aliases]
    extra_edges = [e for e in fresh.edges if (e.src, e.dst) not in base_edges]
    if not extra_nodes and not extra_edges:
        return base
    frameworks = sorted(set(base.framework.split("+")) | set(fresh.framework.split("+")))
    return WiringBlock(
        nodes=list(base.nodes) + extra_nodes,
        edges=list(base.edges) + extra_edges,
        framework="+".join(frameworks),
        source=base.source,
    )


def extract_topology(
    source_files: list[Path],
    candidates: list[CandidateComponent],
    wiring_block: WiringBlock | None = None,
    generic_call_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, TopologyEdges]:
    """Extract topology edges from source code. A wiring_block persisted at Stage 1 is reused as-is;
    otherwise it is re-detected statically (CLI/dir-mode has no Stage 1 to supply one).
    `generic_call_pairs` are (caller_class_name, callee_class_name) pairs from the
    framework-independent call-closure walk — resolved through the SAME class_to_candidate_ids
    translation as a wiring "call" edge, so call-pair emission is not conditional on framework."""
    connect_edges: dict[str, set[str]] = {}
    constructor_edges: dict[str, set[str]] = {}
    # CS-319: candidate-id pairs whose wiring edge is a gray "call" edge / a conditional router branch.
    call_pairs: set[tuple[str, str]] = set()
    conditional_pairs: set[tuple[str, str]] = set()

    class_to_candidate_ids: dict[str, list[str]] = {}
    for candidate in candidates:
        key = (
            wiring_identity(candidate.owner_class_name, candidate.class_name)
            if candidate.entry_kind == "bound_method"
            else candidate.class_name
        )
        class_to_candidate_ids.setdefault(key, []).append(candidate.candidate_id)

    # Cache ASTs to avoid re-parsing files multiple times across three passes
    asts: dict[Path, ast.Module] = {}
    for file in source_files:
        parsed = parse_python_source(file)
        if parsed is not None:
            asts[file] = parsed[1]

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
    if wiring_block is None:
        wiring_block = detect_wiring_block_static(file_contents)
    elif file_contents:
        # Safety-net (Gate D defense-in-depth): a wiring_block persisted before union detection can
        # be missing an entire framework's wiring even though scan_all harvested those components,
        # which would orphan the scanned nodes. Re-detect from source and merge any nodes/edges the
        # passed block lacks. No-op when the passed block is already complete, or when there are no
        # source files to re-scan (the supplied block, incl. an llm_fallback one, is used as-is).
        fresh = detect_wiring_block_static(file_contents)
        if fresh is not None:
            wiring_block = _merge_missing_wiring(wiring_block, fresh)

    add_component_names: dict[str, str] = {}
    if wiring_block:
        for w_node in wiring_block.nodes:
            add_component_names[w_node.alias] = (
                wiring_identity(w_node.owner_class, w_node.callee_name)
                if w_node.entry_kind == "bound_method"
                else w_node.callee_name
            )

        for w_edge in wiring_block.edges:
            source_class_name = add_component_names.get(w_edge.src)
            dest_class_name = add_component_names.get(w_edge.dst)

            if source_class_name and dest_class_name:
                source_ids = class_to_candidate_ids.get(source_class_name, [])
                dest_ids = class_to_candidate_ids.get(dest_class_name, [])

                for src_id in source_ids:
                    for dest_id in dest_ids:
                        if src_id not in connect_edges:
                            connect_edges[src_id] = set()
                        connect_edges[src_id].add(dest_id)
                        # CS-319 edge typing (additive; default hard/non-conditional => unchanged for
                        # every pre-CS-319 wiring edge, so the Haystack map is byte-identical).
                        if getattr(w_edge, "kind", "hard") == "call":
                            call_pairs.add((src_id, dest_id))
                        if getattr(w_edge, "conditional", False):
                            conditional_pairs.add((src_id, dest_id))

    for caller_key, callee_key in (generic_call_pairs or []):
        source_ids = class_to_candidate_ids.get(caller_key, [])
        dest_ids = class_to_candidate_ids.get(callee_key, [])
        for src_id in source_ids:
            for dest_id in dest_ids:
                connect_edges.setdefault(src_id, set()).add(dest_id)
                call_pairs.add((src_id, dest_id))

    for file, tree in asts.items():
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_candidate_ids = class_to_candidate_ids.get(node.name, [])

                if not class_candidate_ids:
                    continue

                # Look at __init__ parameters
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        for arg in item.args.args:
                            if arg.annotation:
                                for name_node in ast.walk(arg.annotation):
                                    if isinstance(name_node, ast.Name):
                                        # Only include if this is a known candidate class
                                        injected_ids = class_to_candidate_ids.get(
                                            name_node.id, []
                                        )

                                        if injected_ids:
                                            for class_id in class_candidate_ids:
                                                for injected_id in injected_ids:
                                                    if class_id not in constructor_edges:
                                                        constructor_edges[class_id] = set()
                                                    constructor_edges[class_id].add(
                                                        injected_id
                                                    )

    edges: dict[str, TopologyEdges] = {}

    # Build all_edges including both connect and constructor edges for upstream/downstream.
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

    # Populate upstream/downstream from all_edges (includes constructors)
    for source, targets in all_edges.items():
        if source not in edges:
            edges[source] = TopologyEdges()
        for target in targets:
            if target not in edges:
                edges[target] = TopologyEdges()

            edges[source].downstream.append(target)
            edges[target].upstream.append(source)

    for topology in edges.values():
        topology.upstream = list(dict.fromkeys(topology.upstream))
        topology.downstream = list(dict.fromkeys(topology.downstream))

    # CS-319: tag the "call" and "conditional" subsets of each source's downstream (both empty on any
    # map with no such wiring edges, i.e. every non-LangGraph-node-body map).
    for src, dst in call_pairs:
        if src in edges and dst in edges[src].downstream:
            edges[src].call_downstream.append(dst)
    for src, dst in conditional_pairs:
        if src in edges and dst in edges[src].downstream:
            edges[src].conditional_downstream.append(dst)
    for topology in edges.values():
        # Sorted, not just deduped: an upstream set (e.g. a resolver's visited-set) can iterate in a
        # different order per process (CPython string hash randomization), which a single CI run won't
        # catch -- sorting here makes the emitted order deterministic regardless of how it got built.
        topology.call_downstream = sorted(dict.fromkeys(topology.call_downstream))
        topology.conditional_downstream = sorted(dict.fromkeys(topology.conditional_downstream))

    # CS-321: classify motif per component over the topology downstream graph (connect edges only).
    _compute_motifs(edges, connect_edges)

    return edges


def _compute_motifs(edges: dict[str, TopologyEdges], connect_edges: dict[str, set[str]]) -> None:
    """Classify per-component motif (linear/branch/loop) based on topology. Branch = has conditional_downstream.
    Loop = member of a cycle (from connect edges only, excluding constructor edges).
    Linear = remainder. Modifies edges in place."""
    # Find all strongly connected components (SCCs) via Tarjan's algorithm over connect_edges only.
    sccs: list[set[str]] = []
    rec_stack: set[str] = set()
    ids: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    id_counter = [0]
    stack: list[str] = []

    def _tarjan_scc(cid: str) -> None:
        ids[cid] = lowlinks[cid] = id_counter[0]
        id_counter[0] += 1
        stack.append(cid)
        rec_stack.add(cid)

        # Walk connect_edges only (exclude constructor edges) for cycle detection.
        for dst in connect_edges.get(cid, set()):
            if dst not in ids:
                _tarjan_scc(dst)
                lowlinks[cid] = min(lowlinks[cid], lowlinks[dst])
            elif dst in rec_stack:
                lowlinks[cid] = min(lowlinks[cid], ids[dst])

        if ids[cid] == lowlinks[cid]:
            scc: set[str] = set()
            while True:
                node = stack.pop()
                rec_stack.discard(node)
                scc.add(node)
                if node == cid:
                    break
            if len(scc) > 1 or (len(scc) == 1 and cid in connect_edges.get(cid, set())):
                sccs.append(scc)

    for candidate_id in edges:
        if candidate_id not in ids:
            _tarjan_scc(candidate_id)

    cycle_members = set()
    for scc in sccs:
        cycle_members.update(scc)

    # Assign motifs
    for cid, topology in edges.items():
        if cid in cycle_members:
            topology.motif = "loop"
        elif topology.conditional_downstream:
            topology.motif = "branch"
        else:
            topology.motif = "linear"
