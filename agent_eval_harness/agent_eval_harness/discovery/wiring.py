import ast
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from agent_eval_harness.llm.client import LLMMessage, RateLimitExceeded

logger = logging.getLogger("agent_eval_harness.discovery.wiring")

_LLM_FALLBACK_MAX_TOKENS = 1024
_LLM_FALLBACK_FILE_CHAR_LIMIT = 6000


def strip_json_code_fence(content: str) -> str:
    """Strip a ```/```json markdown fence an LLM sometimes wraps a json_mode response in."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content


def wiring_identity(owner: str | None, callee: str) -> str:
    """Topology/entry key: `Owner.callee` for a bound method, bare `callee` for a class/function."""
    return f"{owner}.{callee}" if owner else callee


def parse_entry_suffix(suffix: str) -> tuple[str | None, str]:
    """Inverse of wiring_identity: `Owner.method` -> (Owner, method); bare -> (None, bare)."""
    if "." in suffix:
        owner, _, name = suffix.partition(".")
        return owner, name
    return None, suffix


def enclosing_class_name(target: ast.AST, tree: ast.Module) -> str | None:
    """Name of the ClassDef whose subtree contains `target`, or None — a parent-pointer pass without mutating ast. Helper for CS-312; not wired here."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(sub is target for sub in ast.walk(node)):
            return node.name
    return None


@dataclass
class WiringNode:
    alias: str
    callee_name: str
    source_hint_file: str
    entry_kind: str = "class"  # "class" | "function" | "bound_method"
    owner_class: str | None = None


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

    @staticmethod
    def from_dict(data: dict) -> "WiringBlock":
        return WiringBlock(
            nodes=[WiringNode(**n) for n in data.get("nodes", [])],
            edges=[WiringEdge(**e) for e in data.get("edges", [])],
            framework=data.get("framework", "llm_inferred"),
            source=data.get("source", "static"),
        )


def _iter_parsed_files(
    file_contents: dict[str, str], cache: dict[str, ast.AST | None] | None = None
):
    """Parse each file once, silently skipping any with a syntax error; `cache` (when given) is
    shared across the 3 static detectors and nested class-resolution lookups so a file within one
    detection call is never re-parsed."""
    for file, content in file_contents.items():
        if cache is not None and file in cache:
            tree = cache[file]
            if tree is not None:
                yield file, tree
            continue
        try:
            tree = ast.parse(content, filename=file)
        except SyntaxError as e:
            logger.debug(f"Skipping {file} for wiring detection, syntax error: {e}")
            if cache is not None:
                cache[file] = None
            continue
        if cache is not None:
            cache[file] = tree
        yield file, tree


def _const_str(node: ast.expr) -> str | None:
    """Extract a string literal from a Constant (or legacy Str) AST node, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    return None


def _call_func_name(call: ast.Call) -> str | None:
    """Return a Call node's callee name: `Foo(...)` -> "Foo", `mod.Foo(...)` -> "Foo"."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _build_self_attr_classes(tree: ast.AST) -> dict[str, str]:
    """Map self.<attr> -> ClassName for `self.<attr> = ClassName(...)` assignments, so a later
    closure/lambda that only references self.<attr> can still be traced back to its real class."""
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
                class_name = _call_func_name(node.value)
                if class_name and class_name[0].isupper():
                    attr_to_class[target.attr] = class_name
    return attr_to_class


def _find_self_attr_reference(node: ast.AST) -> str | None:
    """Return the first `self.<attr>` access anywhere in node's subtree, including inside lambdas."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "self"
        ):
            return sub.attr
    return None


def _find_class_definition_file(
    class_name: str, file_contents: dict[str, str], cache: dict[str, ast.AST | None] | None = None
) -> str | None:
    """Return the file that actually DEFINES class_name (ground truth, robust to re-exports/lazy
    imports); None if zero or more than one file defines it."""
    matches = [
        fpath for fpath, tree in _iter_parsed_files(file_contents, cache)
        if any(isinstance(n, ast.ClassDef) and n.name == class_name for n in ast.walk(tree))
    ]
    return matches[0] if len(matches) == 1 else None


def _resolve_component_class_and_file(
    arg1: ast.expr,
    file: str,
    self_attr_classes: dict[str, str],
    file_contents: dict[str, str],
    cache: dict[str, ast.AST | None] | None = None,
) -> tuple[str, str]:
    """Resolve add_component()/add_node()'s constructor arg to the real (possibly wrapper-hidden)
    component class and the file it's defined in, via a nested constructor call or a self.<attr>
    closure reference; falls back to the outer wrapper class/this file if neither is found."""
    outer_name = _call_func_name(arg1) or "" if isinstance(arg1, ast.Call) else ""

    inner_name = None
    if isinstance(arg1, ast.Call):
        for sub in list(arg1.args) + [kw.value for kw in arg1.keywords]:
            if isinstance(sub, ast.Call):
                candidate_name = _call_func_name(sub)
                # PascalCase only — a lowercase helper call (e.g. RetrieverComponent(_load_corpus()))
                # is data, not a component.
                if candidate_name and candidate_name[0].isupper():
                    inner_name = candidate_name
                    break

        if inner_name is None:
            attr = _find_self_attr_reference(arg1)
            if attr and attr in self_attr_classes:
                inner_name = self_attr_classes[attr]

    if inner_name:
        resolved_file = _find_class_definition_file(inner_name, file_contents, cache) or file
        return inner_name, resolved_file

    # No wrapper detected — the file where add_component()/add_node() itself is called.
    return outer_name, file


def _detect_haystack(
    file_contents: dict[str, str], cache: dict[str, ast.AST | None] | None = None
) -> WiringBlock | None:
    nodes = {}
    edges = []
    parsed_files = list(_iter_parsed_files(file_contents, cache))

    for file, tree in parsed_files:
        self_attr_classes = _build_self_attr_classes(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "add_component":
                    if len(node.args) >= 2:
                        arg0 = node.args[0]
                        arg1 = node.args[1]

                        alias = _const_str(arg0)
                        class_name, source_hint_file = _resolve_component_class_and_file(
                            arg1, file, self_attr_classes, file_contents, cache
                        )

                        if alias:
                            nodes[alias] = WiringNode(
                                alias=alias,
                                callee_name=class_name,
                                source_hint_file=source_hint_file
                            )

    for file, tree in parsed_files:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "connect":
                    if len(node.args) >= 2:
                        src_str = _const_str(node.args[0])
                        dst_str = _const_str(node.args[1])

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


def _detect_langgraph(
    file_contents: dict[str, str], cache: dict[str, ast.AST | None] | None = None
) -> WiringBlock | None:
    nodes = {}
    edges = []

    for file, tree in _iter_parsed_files(file_contents, cache):
        self_attr_classes = _build_self_attr_classes(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "add_node":
                    if len(node.args) >= 2:
                        arg0 = node.args[0]
                        arg1 = node.args[1]

                        alias = _const_str(arg0)

                        class_name = ""
                        source_hint_file = file
                        if isinstance(arg1, ast.Name):
                            class_name = arg1.id
                        elif isinstance(arg1, ast.Attribute):
                            class_name = arg1.attr
                        elif isinstance(arg1, ast.Call):
                            class_name, source_hint_file = _resolve_component_class_and_file(
                                arg1, file, self_attr_classes, file_contents, cache
                            )

                        if alias:
                            nodes[alias] = WiringNode(
                                alias=alias,
                                callee_name=class_name,
                                source_hint_file=source_hint_file
                            )

                elif isinstance(node.func, ast.Attribute) and node.func.attr == "add_edge":
                    if len(node.args) >= 2:
                        src = _const_str(node.args[0])
                        dst = _const_str(node.args[1])

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


def _detect_langchain_lcel(
    file_contents: dict[str, str], cache: dict[str, ast.AST | None] | None = None
) -> WiringBlock | None:
    nodes = {}
    edges = []

    for file, tree in _iter_parsed_files(file_contents, cache):
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
                            class_name = _call_func_name(op) or ""
                            alias = class_name
                        elif isinstance(op, ast.Attribute):
                            alias = op.attr
                            class_name = op.attr

                        if not alias:
                            alias = f"step_{idx}"
                            class_name = "unknown"

                        unique_alias = alias
                        suffix = 1
                        while unique_alias in nodes and nodes[unique_alias].callee_name != class_name:
                            unique_alias = f"{alias}_{suffix}"
                            suffix += 1

                        node_obj = WiringNode(
                            alias=unique_alias,
                            callee_name=class_name,
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
    # Shared across all 3 detectors so a file isn't re-parsed for each one.
    cache: dict[str, ast.AST | None] = {}
    for detector in (_detect_haystack, _detect_langgraph, _detect_langchain_lcel):
        result = detector(file_contents, cache)
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
        truncated = content[:_LLM_FALLBACK_FILE_CHAR_LIMIT]
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
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        response = await llm_client.complete(
            messages, max_tokens=_LLM_FALLBACK_MAX_TOKENS, json_mode=True
        )
        content = strip_json_code_fence(response.content)

        parsed = json.loads(content)
        nodes_raw = parsed.get("nodes") or []
        edges_raw = parsed.get("edges") or []

        if not nodes_raw:
            return None

        nodes = []
        for n in nodes_raw:
            nodes.append(WiringNode(
                alias=n.get("alias") or "",
                callee_name=n.get("class_name") or "",
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
