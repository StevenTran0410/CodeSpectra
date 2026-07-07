import ast
from dataclasses import dataclass, asdict
import json
import logging
from typing import Any
from agent_eval_harness.llm.client import RateLimitExceeded, LLMMessage

logger = logging.getLogger("agent_eval_harness.discovery.wiring")

@dataclass
class WiringNode:
    alias: str
    class_name: str
    source_hint_file: str

@dataclass
class WiringEdge:
    src: str
    dst: str

@dataclass
class WiringBlock:
    nodes: list[WiringNode]
    edges: list[WiringEdge]
    framework: str  # "haystack" | "langgraph" | "langchain" | "llm_inferred"
    source: str  # "static" | "llm_fallback"

    def to_dict(self) -> dict:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "framework": self.framework,
            "source": self.source,
        }


def _build_self_attr_classes(tree: ast.AST) -> dict[str, str]:
    """Map self.<attr> -> ClassName for every `self.<attr> = ClassName(...)`
    assignment found anywhere in this file. Some wrapper patterns construct the
    real component once (typically in __init__) and merely reference it later via
    a closure/lambda inside add_component()'s arguments — e.g.
    `self._conventions = ConventionsAgent(...)` earlier, then
    `add_component("conventions", _SectionAgentComponent(..., lambda c, d:
    self._conventions.run(...), ...))` later. Tracing the closure back to this
    assignment is the only way to recover the real per-component class in that case."""
    attr_to_class: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and isinstance(node.value, ast.Call)
            ):
                call = node.value
                class_name = None
                if isinstance(call.func, ast.Name):
                    class_name = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    class_name = call.func.attr
                if class_name and class_name[0].isupper():
                    attr_to_class[target.attr] = class_name
    return attr_to_class


def _find_self_attr_reference(node: ast.AST) -> str | None:
    """Walk node's subtree (including inside nested lambdas) for the first
    `self.<attr>` attribute access."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "self"
        ):
            return sub.attr
    return None


def _find_class_definition_file(class_name: str, file_contents: dict[str, str]) -> str | None:
    """Search every file already available (the candidate's cluster files) for
    where class_name is actually DEFINED (a `class ClassName` statement) — far more
    robust than tracing import statements through package re-exports or dynamic
    lazy-loading (e.g. an `__init__.py` that resolves names via `importlib` at
    runtime from a name->submodule dict), since it looks for the ground truth:
    where the class body itself lives, not how it happened to be imported.
    Returns None (not a guess) if zero or more than one file defines it."""
    matches = []
    for fpath, content in file_contents.items():
        try:
            other_tree = ast.parse(content, filename=fpath)
        except SyntaxError:
            continue
        if any(isinstance(n, ast.ClassDef) and n.name == class_name for n in ast.walk(other_tree)):
            matches.append(fpath)
    return matches[0] if len(matches) == 1 else None


def _resolve_component_class_and_file(
    arg1: ast.expr, file: str, self_attr_classes: dict[str, str], file_contents: dict[str, str]
) -> tuple[str, str]:
    """Given add_component()/add_node()'s constructor-call argument, return the most
    specific class name available and the file it's actually defined in.

    Factory/wrapper patterns (e.g. `_SectionAgentComponent(agent=ConventionsAgent())`,
    or the closure form `_SectionAgentComponent(..., lambda c, d: self._conventions.run(...))`)
    register every component from one orchestrator file via one shared wrapper class —
    naively using the wrapper's own class name/file makes every node in the pipeline
    look identical (same class_name, same source_hint_file) even though they are
    completely different real components. Try, in order: (1) a nested constructor
    call directly among arg1's own positional/keyword arguments, (2) a `self.<attr>`
    reference inside arg1 (including inside lambdas) traced back to a
    `self.<attr> = ClassName(...)` assignment elsewhere in the same file. Falls back
    to the outer wrapper class/this file when neither pattern is found — the exact
    same behavior as before this fix for any ordinary, unwrapped add_component() call.
    """
    outer_name = ""
    if isinstance(arg1, ast.Call):
        if isinstance(arg1.func, ast.Name):
            outer_name = arg1.func.id
        elif isinstance(arg1.func, ast.Attribute):
            outer_name = arg1.func.attr

    inner_name = None
    if isinstance(arg1, ast.Call):
        for sub in list(arg1.args) + [kw.value for kw in arg1.keywords]:
            if isinstance(sub, ast.Call):
                candidate_name = None
                if isinstance(sub.func, ast.Name):
                    candidate_name = sub.func.id
                elif isinstance(sub.func, ast.Attribute):
                    candidate_name = sub.func.attr
                # Only treat this as the real wrapped component if it looks like a
                # class instantiation (PascalCase) — a lowercase/underscore-prefixed
                # call here is almost always a plain helper/data function passed as
                # an argument (e.g. `RetrieverComponent(_load_corpus())`), not a
                # nested component. Without this guard, that helper call gets
                # mistaken for "the real class" and both class_name and file end up
                # wrong even for a completely ordinary, unwrapped add_component().
                if candidate_name and candidate_name[0].isupper():
                    inner_name = candidate_name
                    break

        if inner_name is None:
            attr = _find_self_attr_reference(arg1)
            if attr and attr in self_attr_classes:
                inner_name = self_attr_classes[attr]

    if inner_name:
        resolved_file = _find_class_definition_file(inner_name, file_contents) or file
        return inner_name, resolved_file

    # No wrapper detected — keep today's exact behavior: the file where
    # add_component()/add_node() itself is called.
    return outer_name, file


def _detect_haystack(file_contents: dict[str, str]) -> WiringBlock | None:
    nodes = {}
    edges = []

    for file, content in file_contents.items():
        try:
            tree = ast.parse(content, filename=file)
        except SyntaxError:
            continue

        self_attr_classes = _build_self_attr_classes(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "add_component":
                    if len(node.args) >= 2:
                        arg0 = node.args[0]
                        arg1 = node.args[1]

                        alias = None
                        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                            alias = arg0.value
                        elif isinstance(arg0, ast.Str):
                            alias = arg0.s

                        class_name, source_hint_file = _resolve_component_class_and_file(
                            arg1, file, self_attr_classes, file_contents
                        )

                        if alias:
                            nodes[alias] = WiringNode(
                                alias=alias,
                                class_name=class_name,
                                source_hint_file=source_hint_file
                            )
                            
    for file, content in file_contents.items():
        try:
            tree = ast.parse(content, filename=file)
        except SyntaxError:
            continue
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "connect":
                    if len(node.args) >= 2:
                        arg0 = node.args[0]
                        arg1 = node.args[1]
                        
                        src_str = None
                        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                            src_str = arg0.value
                        elif isinstance(arg0, ast.Str):
                            src_str = arg0.s
                            
                        dst_str = None
                        if isinstance(arg1, ast.Constant) and isinstance(arg1.value, str):
                            dst_str = arg1.value
                        elif isinstance(arg1, ast.Str):
                            dst_str = arg1.s
                            
                        if src_str and dst_str:
                            src_parts = src_str.split(".", 1)
                            dst_parts = dst_str.split(".", 1)
                            if src_parts and dst_parts:
                                src_name = src_parts[0]
                                dst_name = dst_parts[0]
                                edges.append(WiringEdge(src=src_name, dst=dst_name))
                                
    if nodes or edges:
        return WiringBlock(nodes=list(nodes.values()), edges=edges, framework="haystack", source="static")
    return None


def _detect_langgraph(file_contents: dict[str, str]) -> WiringBlock | None:
    nodes = {}
    edges = []
    
    for file, content in file_contents.items():
        try:
            tree = ast.parse(content, filename=file)
        except SyntaxError:
            continue

        self_attr_classes = _build_self_attr_classes(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "add_node":
                    if len(node.args) >= 2:
                        arg0 = node.args[0]
                        arg1 = node.args[1]

                        alias = None
                        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                            alias = arg0.value
                        elif isinstance(arg0, ast.Str):
                            alias = arg0.s

                        class_name = ""
                        source_hint_file = file
                        if isinstance(arg1, ast.Name):
                            class_name = arg1.id
                        elif isinstance(arg1, ast.Attribute):
                            class_name = arg1.attr
                        elif isinstance(arg1, ast.Call):
                            class_name, source_hint_file = _resolve_component_class_and_file(
                                arg1, file, self_attr_classes, file_contents
                            )

                        if alias:
                            nodes[alias] = WiringNode(
                                alias=alias,
                                class_name=class_name,
                                source_hint_file=source_hint_file
                            )
                            
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "add_edge":
                    if len(node.args) >= 2:
                        arg0 = node.args[0]
                        arg1 = node.args[1]
                        
                        src = None
                        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                            src = arg0.value
                        elif isinstance(arg0, ast.Str):
                            src = arg0.s
                            
                        dst = None
                        if isinstance(arg1, ast.Constant) and isinstance(arg1.value, str):
                            dst = arg1.value
                        elif isinstance(arg1, ast.Str):
                            dst = arg1.s
                            
                        if src and dst:
                            edges.append(WiringEdge(src=src, dst=dst))
                            
    if nodes or edges:
        return WiringBlock(nodes=list(nodes.values()), edges=edges, framework="langgraph", source="static")
    return None


def _flatten_bitor(node: ast.AST, sub_nodes_set: set[ast.AST]) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        sub_nodes_set.add(node)
        return _flatten_bitor(node.left, sub_nodes_set) + _flatten_bitor(node.right, sub_nodes_set)
    return [node]


_BUILTIN_TYPE_NAMES = frozenset({
    "str", "int", "float", "bool", "dict", "list", "set", "tuple",
    "bytes", "bytearray", "complex", "object", "frozenset", "type", "Any", "None",
})


def _looks_like_type_union(operands: list) -> bool:
    """PEP 604 unions (`str | None`) parse to the same BinOp/BitOr AST shape as a
    LangChain LCEL chain — reject anything containing a bare constant (the `None`
    in `X | None`) or a builtin type name, since a real pipe chain never does."""
    for op in operands:
        if isinstance(op, ast.Constant):
            return True
        if isinstance(op, ast.Name) and op.id in _BUILTIN_TYPE_NAMES:
            return True
    return False


def _detect_langchain_lcel(file_contents: dict[str, str]) -> WiringBlock | None:
    nodes = {}
    edges = []
    
    for file, content in file_contents.items():
        try:
            tree = ast.parse(content, filename=file)
        except SyntaxError:
            continue
            
        processed_binops = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                if node in processed_binops:
                    continue
                    
                operands = _flatten_bitor(node, processed_binops)
                if len(operands) >= 2 and not _looks_like_type_union(operands):
                    operand_nodes = []
                    for idx, op in enumerate(operands):
                        alias = None
                        class_name = ""
                        
                        if isinstance(op, ast.Name):
                            alias = op.id
                            class_name = op.id
                        elif isinstance(op, ast.Call):
                            if isinstance(op.func, ast.Name):
                                class_name = op.func.id
                            elif isinstance(op.func, ast.Attribute):
                                class_name = op.func.attr
                            alias = class_name
                        elif isinstance(op, ast.Attribute):
                            alias = op.attr
                            class_name = op.attr
                        
                        if not alias:
                            alias = f"step_{idx}"
                            class_name = "unknown"
                            
                        unique_alias = alias
                        suffix = 1
                        while unique_alias in nodes and nodes[unique_alias].class_name != class_name:
                            unique_alias = f"{alias}_{suffix}"
                            suffix += 1
                            
                        node_obj = WiringNode(
                            alias=unique_alias,
                            class_name=class_name,
                            source_hint_file=file
                        )
                        nodes[unique_alias] = node_obj
                        operand_nodes.append(node_obj)
                        
                    for i in range(len(operand_nodes) - 1):
                        edges.append(WiringEdge(
                            src=operand_nodes[i].alias,
                            dst=operand_nodes[i + 1].alias
                        ))
                        
    if nodes or edges:
        return WiringBlock(nodes=list(nodes.values()), edges=edges, framework="langchain", source="static")
    return None


def detect_wiring_block_static(file_contents: dict[str, str]) -> WiringBlock | None:
    for detector in (_detect_haystack, _detect_langgraph, _detect_langchain_lcel):
        result = detector(file_contents)
        if result is not None:
            return result
    return None


async def detect_wiring_block(
    file_contents: dict[str, str],
    llm_client: Any = None,
) -> WiringBlock | None:
    result = detect_wiring_block_static(file_contents)
    if result is not None:
        return result
    if llm_client is None:
        return None
    return await _detect_via_llm(file_contents, llm_client)


async def _detect_via_llm(file_contents: dict[str, str], llm_client: Any) -> WiringBlock | None:
    context_parts = []
    for path, content in file_contents.items():
        truncated = content[:6000]
        context_parts.append(f"=== File: {path} ===\n{truncated}\n")
    
    files_context = "\n".join(context_parts)
    
    system_prompt = (
        "You are an expert AI software architect. Analyze the provided file contents and determine if there is an agentic pipeline or call chain (custom orchestrator or undocumented framework).\n"
        "If you find a pipeline/agent orchestration structure, respond ONLY in raw JSON format with the exact schema:\n"
        "{\n"
        '  "framework": "string (the framework name or \'llm_inferred\')",\n'
        '  "nodes": [\n'
        '    {"alias": "string", "class_name": "string", "source_hint_file": "string"}\n'
        '  ],\n'
        '  "edges": [\n'
        '    {"src": "string", "dst": "string"}\n'
        '  ]\n'
        "}\n"
        "If no agentic pipeline/orchestration is present, return empty lists."
    )
    
    user_prompt = f"File contents:\n{files_context}\n\nAnalyze the files and return the wiring block JSON."
    
    try:
        from agent_eval_harness.llm.client import LLMMessage
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        response = await llm_client.complete(messages, max_tokens=1024, json_mode=True)
        content = response.content.strip()
        
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
            
        parsed = json.loads(content)
        nodes_raw = parsed.get("nodes") or []
        edges_raw = parsed.get("edges") or []
        
        if not nodes_raw:
            return None
            
        nodes = []
        for n in nodes_raw:
            nodes.append(WiringNode(
                alias=n.get("alias") or "",
                class_name=n.get("class_name") or "",
                source_hint_file=n.get("source_hint_file") or "",
            ))
        edges = []
        for e in edges_raw:
            edges.append(WiringEdge(
                src=e.get("src") or "",
                dst=e.get("dst") or "",
            ))
            
        return WiringBlock(
            nodes=nodes,
            edges=edges,
            framework=parsed.get("framework") or "llm_inferred",
            source="llm_fallback",
        )
    except RateLimitExceeded:
        raise
    except Exception as exc:
        logger.warning("LLM fallback wiring detection failed: %s", exc)
        return None
