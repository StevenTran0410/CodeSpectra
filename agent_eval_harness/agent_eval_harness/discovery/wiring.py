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


def _detect_haystack(file_contents: dict[str, str]) -> WiringBlock | None:
    nodes = {}
    edges = []
    
    for file, content in file_contents.items():
        try:
            tree = ast.parse(content, filename=file)
        except SyntaxError:
            continue
            
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
                            
                        class_name = ""
                        if isinstance(arg1, ast.Call):
                            if isinstance(arg1.func, ast.Name):
                                class_name = arg1.func.id
                            elif isinstance(arg1.func, ast.Attribute):
                                class_name = arg1.func.attr
                        
                        if alias:
                            nodes[alias] = WiringNode(
                                alias=alias,
                                class_name=class_name,
                                source_hint_file=file
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
                        if isinstance(arg1, ast.Name):
                            class_name = arg1.id
                        elif isinstance(arg1, ast.Attribute):
                            class_name = arg1.attr
                        elif isinstance(arg1, ast.Call):
                            if isinstance(arg1.func, ast.Name):
                                class_name = arg1.func.id
                            elif isinstance(arg1.func, ast.Attribute):
                                class_name = arg1.func.attr
                                
                        if alias:
                            nodes[alias] = WiringNode(
                                alias=alias,
                                class_name=class_name,
                                source_hint_file=file
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
