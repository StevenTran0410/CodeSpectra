from __future__ import annotations

import ast
import json
import logging
from agent_eval_harness.llm.client import LLMClient, LLMMessage
from agent_eval_harness.discovery.client import CodeSpectraClient
from agent_eval_harness.discovery.analysis_context import ProjectContext

logger = logging.getLogger("agent_eval_harness.discovery.expansion")


def extract_symbol_snippet(content: str, symbol_identifier: str) -> str:
    """Return the source snippet for symbol_identifier (e.g. 'MyClass.run', 'MyClass',
    'my_function'), or empty string if it can't be parsed or found."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ""

    lines = content.splitlines()

    parts = symbol_identifier.split(".")
    class_name = None
    func_name = None
    if len(parts) == 2:
        class_name, func_name = parts[0], parts[1]
    elif len(parts) == 1:
        class_name = parts[0]
    else:
        return ""

    matched_node = None

    if class_name and func_name:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for sub_node in node.body:
                    if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub_node.name == func_name:
                        matched_node = sub_node
                        break
                if matched_node:
                    break
    elif class_name:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                matched_node = node
                break
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == class_name:
                matched_node = node
                break

    if not matched_node:
        return ""

    start_line = matched_node.lineno - 1
    end_line = getattr(matched_node, "end_lineno", None)
    if end_line is None:
        end_line = start_line + 50

    snippet_lines = lines[start_line:end_line]
    if len(snippet_lines) > 100:
        snippet_lines = snippet_lines[:100]
    return "\n".join(snippet_lines)


MAX_CLASSIFY_BATCH_SIZE = 6

VALID_ROLE_HINTS = {"orchestrator", "agent_core", "prompt", "tool", "context_builder", "model_client", "config", "util"}

async def _classify_nodes_batch(
    items: list[tuple[str, str, str | None]],
    llm_client: LLMClient,
    candidate: dict,
    project_context: ProjectContext | None = None,
) -> dict[str, dict]:
    """Classify a batch of Python code chunks separately using a single LLM call."""
    name = candidate.get("name", "unknown")
    frameworks = ", ".join(candidate.get("frameworks", []))

    system_prompt = (
        "You are an expert AI software architect. You will be given several independent Python "
        "source code chunks from a target codebase, each from a different file. Classify EACH ONE "
        "separately and carefully — a verdict on one chunk must never be influenced by the content "
        "of another chunk in this batch. Judge every chunk purely on its own content.\n"
        f"The candidate agentic system we are expanding is named \"{name}\" and uses frameworks: "
        f"[{frameworks}].\n"
        "For each chunk, determine if it is:\n"
        "1. \"accept\" - a core agentic component (agent loops, tool orchestrators, custom agents, "
        "prompt templates, main entry points).\n"
        "2. \"boundary\" - pure non-agentic infrastructure, library utility, database helper, "
        "logging, test files, third-party libraries, or configuration. Do not expand its neighbors.\n"
        "3. \"expand\" - an intermediary connector module that bridges or imports agentic systems "
        "(e.g. API routers, connector files, shared models). Accept it and continue expanding.\n\n"
        "Valid role_hint values are: orchestrator, agent_core, prompt, tool, context_builder, "
        "model_client, config, util. Assign a role_hint only to chunks classified as 'accept'.\n\n"
        "Respond ONLY in raw JSON, one entry per chunk, in this exact schema:\n"
        "{\"verdicts\": [{\"id\": \"<unique chunk id, copied exactly from the input>\", "
        "\"verdict\": \"accept\"|\"boundary\"|\"expand\", \"reason\": \"brief\", "
        "\"role_hint\": \"<one of the valid role_hint values or null>\", "
        "\"key_symbols\": [\"symbol1\", \"symbol2\"], "
        "\"follow\": false, \"skip\": false}]}"
    )

    if project_context and project_context.feature_map:
        system_prompt += "\n\nProject feature map:\n" + project_context.feature_map.as_context_block()

    user_prompt = "\n\n".join(
        f"=== ID: {path}::{chunk_id} ===\n{content}" for path, content, chunk_id in items
    )

    result: dict[str, dict] = {
        f"{path}::{chunk_id}": {
            "verdict": "boundary",
            "reason": "",
            "role_hint": None,
            "key_symbols": [],
            "follow": False,
            "skip": False,
        }
        for path, _, chunk_id in items
    }
    try:
        response = await llm_client.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            json_mode=True,
        )
        data = json.loads(response.content)
        for entry in data.get("verdicts", []):
            unique_id = entry.get("id")
            verdict = entry.get("verdict")
            if unique_id in result and verdict in ("accept", "boundary", "expand"):
                role_hint = entry.get("role_hint")
                # Coerce role_hint to None if not in valid set
                if role_hint and role_hint not in VALID_ROLE_HINTS:
                    role_hint = None
                result[unique_id] = {
                    "verdict": verdict,
                    "reason": entry.get("reason", ""),
                    "role_hint": role_hint,
                    "key_symbols": entry.get("key_symbols", []),
                    "follow": bool(entry.get("follow", False)),
                    "skip": bool(entry.get("skip", False)),
                }
    except Exception as e:
        logger.warning(f"Batch classification failed: {e}. Defaulting batch to 'boundary'.")
    return result


async def expand_candidate(
    snapshot_id: str,
    candidate: dict,
    client: CodeSpectraClient,
    llm_client: LLMClient,
    *,
    node_budget: int = 100,
    hop_cap: int = 3,
    project_context: ProjectContext | None = None,
) -> dict:
    # 1. Chunks Extraction for Seeds
    seeds = []
    chunked_files = set()
    evidence_list = candidate.get("evidence", [])
    excluded = set(candidate.get("excluded_files", []))
    for ev in evidence_list:
        file_path = ev.get("file")
        if file_path in excluded:
            continue
        chunk_id = ev.get("chunk_id")
        snippet = ev.get("snippet")
        if file_path and chunk_id:
            seeds.append((file_path, chunk_id, snippet))
            chunked_files.add(file_path)

    # For files in cluster_files or hub_paths that don't have chunk evidence, add them as whole-file fallbacks (chunk_id = None)
    if candidate.get("matched_files"):
        fallback_files = set(candidate["matched_files"]) - excluded
    else:
        fallback_files = (set(candidate.get("cluster_files", [])) | set(candidate.get("hub_paths", []))) - excluded
    for f in sorted(fallback_files):
        if f not in chunked_files:
            seeds.append((f, None, None))

    # Frontier elements: list of (file_path, chunk_id, snippet)
    frontier = seeds[:]
    accepted_chunks = set()
    accepted_files = set()
    boundary_files = set()
    visited_chunks = set()
    raw_file_edges: set[tuple[str, str]] = set()

    # hops_from_seed tracks distance at (file_path, chunk_id) granularity
    hops_from_seed = {}
    for f_path, c_id, _ in frontier:
        hops_from_seed[(f_path, c_id)] = 0

    # annotations tracks verdict data for each file
    annotations = {}
    # boundary_reasons collects reason strings for all boundary verdicts (for A9 log)
    boundary_reasons: list[str] = []

    while frontier:
        if len(accepted_chunks) >= node_budget:
            accepted_edges = sorted(
                {(s, d) for s, d in raw_file_edges if s in accepted_files and d in accepted_files}
            )
            edges_out = [{"src": s, "dst": d} for s, d in accepted_edges]
            accepted_list = [
                {
                    "file": f,
                    "role_hint": annotations.get(f, {}).get("role_hint"),
                    "key_symbols": annotations.get(f, {}).get("key_symbols", []),
                    "follow": annotations.get(f, {}).get("follow", False),
                }
                for f in sorted(accepted_files)
            ]
            return {
                "accepted": accepted_list,
                "boundary": sorted(list(boundary_files - accepted_files)),
                "boundary_reasons": boundary_reasons,
                "stop_reason": "node_budget",
                "accepted_edges": edges_out
            }

        level = []
        while frontier and len(level) < MAX_CLASSIFY_BATCH_SIZE:
            item = frontier.pop(0)
            if (item[0], item[1]) in visited_chunks:
                continue
            visited_chunks.add((item[0], item[1]))
            level.append(item)

        if not level:
            continue

        to_classify = []
        resolved = {}
        for file_path, chunk_id, snippet in level:
            if snippet is not None:
                content = snippet
            else:
                try:
                    file_resp = await client.read_file(snapshot_id, file_path)
                    full_content = file_resp.get("content", "")
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
                    full_content = ""

                if chunk_id is not None:
                    content = extract_symbol_snippet(full_content, chunk_id)
                    if not content:
                        content = full_content
                else:
                    content = full_content

            wb = candidate.get("wiring_block")
            skip_classify = False
            if wb and isinstance(wb, dict):
                nodes = wb.get("nodes") or []
                for n in nodes:
                    if n.get("source_hint_file") == file_path:
                        skip_classify = True
                        break

            resolved[(file_path, chunk_id)] = (content, skip_classify)
            if not skip_classify:
                to_classify.append((file_path, content, chunk_id))

        verdicts = await _classify_nodes_batch(to_classify, llm_client, candidate, project_context) if to_classify else {}

        for file_path, chunk_id, snippet in level:
            content, skip_classify = resolved[(file_path, chunk_id)]
            verdict_data = verdicts.get(f"{file_path}::{chunk_id}", {
                "verdict": "boundary",
                "reason": "",
                "role_hint": None,
                "key_symbols": [],
                "follow": False,
                "skip": False,
            })
            if isinstance(verdict_data, str):
                # Handle old format compatibility (shouldn't happen in phase 1, but be safe)
                verdict = verdict_data
                verdict_data = {
                    "verdict": verdict,
                    "reason": "",
                    "role_hint": None,
                    "key_symbols": [],
                    "follow": False,
                    "skip": False,
                }
            else:
                verdict = verdict_data.get("verdict", "boundary")

            if skip_classify:
                verdict = "expand"

            chunk_key = (file_path, chunk_id)

            if verdict == "boundary":
                boundary_files.add(file_path)
                boundary_reasons.append(verdict_data.get("reason", ""))
                continue

            # Track annotation data for accepted files
            annotations[file_path] = verdict_data

            accepted_chunks.add(chunk_key)
            accepted_files.add(file_path)

            if verdict == "expand" and hops_from_seed.get(chunk_key, 0) < hop_cap:
                try:
                    edges_resp = await client.get_symbol_edges(snapshot_id, file_path)

                    for edge in edges_resp.get("outgoing", []):
                        src_sym = edge.get("src_symbol")
                        dst_sym = edge.get("dst_symbol")
                        if src_sym and dst_sym and "::" in src_sym and "::" in dst_sym:
                            sf, src_id = src_sym.split("::", 1)
                            df, dst_id = dst_sym.split("::", 1)
                            if chunk_id is None or src_id == chunk_id:
                                if df != file_path:
                                    raw_file_edges.add((file_path, df))
                                neighbor = (df, dst_id)
                                if neighbor not in visited_chunks:
                                    frontier.append((df, dst_id, None))
                                    hops_from_seed[neighbor] = hops_from_seed.get(chunk_key, 0) + 1

                    for edge in edges_resp.get("incoming", []):
                        src_sym = edge.get("src_symbol")
                        dst_sym = edge.get("dst_symbol")
                        if src_sym and dst_sym and "::" in src_sym and "::" in dst_sym:
                            sf, src_id = src_sym.split("::", 1)
                            df, dst_id = dst_sym.split("::", 1)
                            if chunk_id is None or dst_id == chunk_id:
                                if sf != file_path:
                                    raw_file_edges.add((sf, file_path))
                                neighbor = (sf, src_id)
                                if neighbor not in visited_chunks:
                                    frontier.append((sf, src_id, None))
                                    hops_from_seed[neighbor] = hops_from_seed.get(chunk_key, 0) + 1
                except Exception as e:
                    logger.warning(f"Failed to get symbol edges for {file_path}: {e}")

        # BFS frontier sort: partition frontier within same hop level
        # Items matching follow=True key_symbols go to front
        follow_symbols = set()
        for annotation in annotations.values():
            if annotation.get("follow", False):
                follow_symbols.update(annotation.get("key_symbols", []))

        if follow_symbols and frontier:
            # Partition frontier: matching items first, then rest
            matching = []
            rest = []
            for item in frontier:
                file_path, chunk_id, snippet = item
                item_str = f"{file_path}::{chunk_id}" if chunk_id else file_path
                if any(symbol in item_str for symbol in follow_symbols):
                    matching.append(item)
                else:
                    rest.append(item)
            frontier = matching + rest

    accepted_edges = sorted(
        {(s, d) for s, d in raw_file_edges if s in accepted_files and d in accepted_files}
    )
    edges_out = [{"src": s, "dst": d} for s, d in accepted_edges]
    accepted_list = [
        {
            "file": f,
            "role_hint": annotations.get(f, {}).get("role_hint"),
            "key_symbols": annotations.get(f, {}).get("key_symbols", []),
            "follow": annotations.get(f, {}).get("follow", False),
        }
        for f in sorted(accepted_files)
    ]
    return {
        "accepted": accepted_list,
        "boundary": sorted(list(boundary_files - accepted_files)),
        "boundary_reasons": boundary_reasons,
        "stop_reason": "frontier_exhausted",
        "accepted_edges": edges_out
    }
