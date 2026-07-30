import ast
import builtins
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_eval_harness.llm.client import LLMMessage, RateLimitExceeded

logger = logging.getLogger("agent_eval_harness.discovery.wiring")

_LLM_FALLBACK_MAX_TOKENS = 1024
_LLM_FALLBACK_FILE_CHAR_LIMIT = 6000
_RESOLVER_RRF_HIT_LIMIT = 10  # RRF fusion hits inspected per unresolved step before giving up


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
    """Name of the ClassDef whose subtree contains `target`, or None. Thin lookup over
    `_build_enclosing_class_map` so there is one enclosing-class traversal, not two."""
    return _build_enclosing_class_map(tree).get(id(target))


@dataclass
class WiringNode:
    alias: str
    callee_name: str
    source_hint_file: str
    entry_kind: str = "class"  # "class" | "function" | "bound_method"
    owner_class: str | None = None
    # Which detector produced this node. Defaulted so pre-union wiring_block_json rows (whose nodes
    # lack the key) still deserialize via WiringNode(**n); the union stamps the real framework.
    framework: str = "unknown"


@dataclass
class WiringEdge:
    src: str
    dst: str
    # CS-319 edge typing (additive; defaults reproduce every pre-CS-319 edge byte-for-byte):
    # "hard" = framework control-flow edge (StateGraph add_edge / add_conditional_edges dst);
    # "call" = intra-node call to a resolved gray target (node-body call graph).
    kind: str = "hard"
    # True for a hard edge whose source routes via add_conditional_edges(src, router, ...).
    conditional: bool = False


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


def _build_enclosing_class_map(tree: ast.AST) -> dict[int, str]:
    """id(node) -> name of the innermost enclosing ClassDef, for every node inside a class body.
    ast has no parent pointer, so this is one explicit ordered pass via ast.iter_child_nodes
    (ast.walk loses nesting order). Lets a bare self.<method> reference recover its owner class."""
    owner: dict[int, str] = {}

    def _walk(node: ast.AST, current: str | None) -> None:
        if isinstance(node, ast.ClassDef):
            current = node.name
        if current is not None:
            owner[id(node)] = current
        for child in ast.iter_child_nodes(node):
            _walk(child, current)

    _walk(tree, None)
    return owner


def _build_stategraph_var_names(tree: ast.AST) -> set[str]:
    """Names (local `graph` or `self.<attr>` encoded as "self.<attr>") assigned from a StateGraph(...)
    constructor. Receiver-guard for add_node/add_edge/add_conditional_edges: trust <name>.add_node(...)
    only when <name> traces back to a StateGraph() construction, not any object with a same-named method."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.value, ast.Call)
            and _call_func_name(node.value) == "StateGraph"
        ):
            target = node.targets[0]
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                names.add(f"self.{target.attr}")
    return names


def _receiver_name(func_value: ast.expr) -> str | None:
    """Receiver of a `<name>.method(...)` call in the same encoding as _build_stategraph_var_names
    ("graph" or "self.<attr>"); None if it isn't a plain name or self-attribute."""
    if isinstance(func_value, ast.Name):
        return func_value.id
    if (
        isinstance(func_value, ast.Attribute)
        and isinstance(func_value.value, ast.Name)
        and func_value.value.id == "self"
    ):
        return f"self.{func_value.attr}"
    return None


def _imports_langgraph(tree: ast.AST) -> bool:
    """True if the file imports anything from the langgraph package — a cheap guard for
    create_react_agent/@task/@entrypoint, which have no receiver to track and names generic
    enough (especially @task) to need an import gate."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "langgraph":
            return True
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "langgraph" for a in node.names):
            return True
    return False


def _decorator_effective_name(decorator: ast.expr) -> str | None:
    """Callee name of a decorator whether bare (@task) or Call-wrapped (@entrypoint(checkpointer=...));
    the Haystack decorator helper only unwraps Name/Attribute, not Call."""
    if isinstance(decorator, ast.Call):
        return _call_func_name(decorator)
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return None


def _const_str_or_sentinel(node: ast.expr) -> str | None:
    """_const_str plus the ast.Name sentinels START/END -> "__start__"/"__end__" (the string names
    the Pregel runtime uses for the virtual start/end nodes)."""
    s = _const_str(node)
    if s is not None:
        return s
    if isinstance(node, ast.Name):
        if node.id == "START":
            return "__start__"
        if node.id == "END":
            return "__end__"
    return None


def _resolve_node_target(
    arg1: ast.expr,
    enclosing_class_map: dict[int, str],
    self_attr_classes: dict[str, str],
    known_class_names: set[str],
    file: str,
    file_contents: dict[str, str],
    cache: dict[str, ast.AST | None] | None = None,
) -> tuple[str, str, str | None]:
    """Resolve add_node()'s second arg to (callee_name, entry_kind, owner_class). Single source of
    truth shared by Stage-1 `_detect_langgraph` and Stage-2 `LangGraphScanner` so the bound-method
    owner resolution can never drift between the two."""
    # self.<method> — a bound method; recover its owner class from the enclosing-class map.
    if (
        isinstance(arg1, ast.Attribute)
        and isinstance(arg1.value, ast.Name)
        and arg1.value.id == "self"
    ):
        return arg1.attr, "bound_method", enclosing_class_map.get(id(arg1))
    if isinstance(arg1, ast.Name):
        if arg1.id in known_class_names:
            return arg1.id, "class", None
        return arg1.id, "function", None
    if isinstance(arg1, ast.Attribute):
        # Non-self attribute (e.g. mod.fn) — best effort, owner unknown.
        return arg1.attr, "function", None
    if isinstance(arg1, ast.Call):
        callee, _resolved_file = _resolve_component_class_and_file(
            arg1, file, self_attr_classes, file_contents, cache
        )
        return callee, "class", None
    return "", "function", None


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


def _langgraph_conditional_edges(node: ast.Call) -> list[WiringEdge]:
    """Edges from an add_conditional_edges(src, router, {path: dst}) call: over-approximate to an
    edge into every dict destination (router itself is not a node). List-form path_map and non-dict
    forms yield nothing (clean degrade). Same deliberate over-approximation as the BitOr LCEL path."""
    src = _const_str_or_sentinel(node.args[0]) if node.args else None
    if not src:
        return []
    path_map = node.args[2] if len(node.args) >= 3 else next(
        (kw.value for kw in node.keywords if kw.arg == "path_map"), None
    )
    if not isinstance(path_map, ast.Dict):
        return []
    out = []
    for value in path_map.values:
        dst = _const_str_or_sentinel(value)
        if dst:
            # CS-319: a router source's outgoing StateGraph edges are conditional branches.
            out.append(WiringEdge(src=src, dst=dst, kind="hard", conditional=True))
    return out


def _langgraph_react_agent(
    tree: ast.AST, file: str, nodes: dict[str, WiringNode], edges: list[WiringEdge]
) -> None:
    """create_react_agent(...) call site -> one function-entry node named after the assigned var,
    plus an edge to each literal tool in tools=[...]. Import-gated (no receiver to track)."""
    if not _imports_langgraph(tree):
        return
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and _call_func_name(node.value) == "create_react_agent"
        ):
            var = node.targets[0].id
            nodes[var] = WiringNode(
                alias=var, callee_name="create_react_agent",
                source_hint_file=file, entry_kind="function", owner_class=None,
            )
            tools = next((kw.value for kw in node.value.keywords if kw.arg == "tools"), None)
            if isinstance(tools, ast.List):
                for elt in tools.elts:
                    tool = elt.id if isinstance(elt, ast.Name) else (
                        elt.attr if isinstance(elt, ast.Attribute) else None
                    )
                    if tool:
                        edges.append(WiringEdge(src=var, dst=tool))


def _langgraph_functional_api(
    tree: ast.AST, file: str, nodes: dict[str, WiringNode], edges: list[WiringEdge]
) -> None:
    """@task / @entrypoint functional API -> one function-entry node per decorated top-level def,
    plus an edge from each entrypoint to every @task it calls. Import-gated (@task is generic)."""
    if not _imports_langgraph(tree):
        return
    task_names: set[str] = set()
    entrypoint_names: set[str] = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = {_decorator_effective_name(d) for d in node.decorator_list}
            if "task" in names:
                task_names.add(node.name)
            if "entrypoint" in names:
                entrypoint_names.add(node.name)
    for name in task_names | entrypoint_names:
        nodes[name] = WiringNode(
            alias=name, callee_name=name, source_hint_file=file,
            entry_kind="function", owner_class=None,
        )
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in entrypoint_names:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and _call_func_name(sub) in task_names:
                    edges.append(WiringEdge(src=node.name, dst=_call_func_name(sub)))


def _detect_langgraph(
    file_contents: dict[str, str], cache: dict[str, ast.AST | None] | None = None
) -> WiringBlock | None:
    nodes: dict[str, WiringNode] = {}
    edges: list[WiringEdge] = []

    for file, tree in _iter_parsed_files(file_contents, cache):
        self_attr_classes = _build_self_attr_classes(tree)
        stategraph_names = _build_stategraph_var_names(tree)
        enclosing_class_map = _build_enclosing_class_map(tree)
        known_class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            attr = node.func.attr
            # Receiver-guard: only trust a call whose receiver traces to a StateGraph() construction.
            if attr in ("add_node", "add_edge", "add_conditional_edges"):
                if _receiver_name(node.func.value) not in stategraph_names:
                    continue
            if attr == "add_node" and len(node.args) >= 2:
                alias = _const_str(node.args[0])
                if not alias:
                    continue
                callee, entry_kind, owner = _resolve_node_target(
                    node.args[1], enclosing_class_map, self_attr_classes,
                    known_class_names, file, file_contents, cache,
                )
                nodes[alias] = WiringNode(
                    alias=alias, callee_name=callee, source_hint_file=file,
                    entry_kind=entry_kind, owner_class=owner,
                )
            elif attr == "add_edge" and len(node.args) >= 2:
                src = _const_str_or_sentinel(node.args[0])
                dst = _const_str_or_sentinel(node.args[1])
                if src and dst:
                    edges.append(WiringEdge(src=src, dst=dst))
            elif attr == "add_conditional_edges":
                edges.extend(_langgraph_conditional_edges(node))

        _langgraph_react_agent(tree, file, nodes, edges)
        _langgraph_functional_api(tree, file, nodes, edges)

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

_NUMPY_DTYPE_ATTRS = frozenset({
    "float16", "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool_", "complex64", "complex128",
})


def _looks_like_type_union(operands: list) -> bool:
    """PEP 604 unions (`str | None`) parse to the same BinOp/BitOr AST shape as a
    LangChain LCEL chain — reject anything containing a bare constant (the `None`
    in `X | None`), a builtin type name, a numpy-dtype union (`np.float32 | np.int64`),
    or a flag-enum OR (`Flags.RED | Flags.BLUE`, by ALL-CAPS attr convention), since a
    real pipe chain never does."""
    for op in operands:
        if isinstance(op, ast.Constant):
            return True
        if isinstance(op, ast.Name) and op.id in _BUILTIN_TYPE_NAMES:
            return True
        if isinstance(op, ast.Attribute) and (
            op.attr in _NUMPY_DTYPE_ATTRS or op.attr.isupper()
        ):
            return True
    return False


# LangChain chain/runnable factory + wrapper constructors: each RETURNS a Runnable and takes prior
# Runnables/retrievers/llms as args, so the composed graph is built by nested calls, not `|`. This is
# the dominant production LCEL idiom. The set is LangChain-idiom-generic (not any one target's names).
_LANGCHAIN_CHAIN_FACTORIES = frozenset({
    "create_retrieval_chain", "create_history_aware_retriever", "create_stuff_documents_chain",
    "create_sql_query_chain", "create_openai_tools_agent", "create_openai_functions_agent",
    "create_tool_calling_agent", "create_react_agent",
    "RunnableWithMessageHistory", "RunnableParallel", "RunnablePassthrough", "RunnableLambda",
    "RunnableBranch", "RunnableSequence", "RunnableMap", "RunnableAssign",
})
# A `<vectorstore>.as_retriever(...)` call also produces a Runnable (a retriever) that later factories
# consume; matched by method name (attribute call), import-gated below.
_RETRIEVER_FACTORY_METHOD = "as_retriever"


def _imports_langchain(tree: ast.AST) -> bool:
    """True if the file imports from any langchain* package — a cheap gate so the factory idiom
    (esp. the generic `.as_retriever()` method) never false-positives on non-LangChain code."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0].startswith("langchain"):
            return True
        if isinstance(node, ast.Import) and any(a.name.split(".")[0].startswith("langchain") for a in node.names):
            return True
    return False


def _langchain_chain_factories(
    tree: ast.AST, file: str, nodes: dict, edges: list
) -> None:
    """Factory-function LCEL idiom: `V = create_retrieval_chain(a, b)` / `V = x.as_retriever(...)`.
    Each produced variable is a Runnable node; edges come from call ARGS that name a prior produced
    Runnable (linkage is by argument, not `|`). Walks module AND function bodies, so chains built
    inside a `build_chain()`-style factory function and returned are seen. Import-gated."""
    if not _imports_langchain(tree):
        return

    def _produced_var(node: ast.AST) -> tuple[str, ast.Call] | None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            callee = _call_func_name(node.value)
            if callee in _LANGCHAIN_CHAIN_FACTORIES or callee == _RETRIEVER_FACTORY_METHOD:
                return node.targets[0].id, node.value
        return None

    produced: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        pv = _produced_var(node)
        if pv is not None:
            produced[pv[0]] = pv[1]

    for var, call in produced.items():
        nodes.setdefault(var, WiringNode(
            alias=var, callee_name=var, source_hint_file=file,
            entry_kind="function", owner_class=None,
        ))
        arg_exprs = list(call.args) + [kw.value for kw in call.keywords]
        for arg in arg_exprs:
            if isinstance(arg, ast.Name) and arg.id in produced and arg.id != var:
                edges.append(WiringEdge(src=var, dst=arg.id))


def _detect_langchain_lcel(
    file_contents: dict[str, str], cache: dict[str, ast.AST | None] | None = None
) -> WiringBlock | None:
    nodes = {}
    edges = []

    for file, tree in _iter_parsed_files(file_contents, cache):
        _langchain_chain_factories(tree, file, nodes, edges)
        # The pipe idiom (a | b) is only LCEL when the file imports langchain — otherwise a plain
        # `set(...) | set(...)` / bitwise-or is a false positive (factory path already self-gates).
        if not _imports_langchain(tree):
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


# Pure-python agents use no framework DSL — the entry method orchestrates a sequence of collaborator
# calls (planning, retrieval, LLM, tools). The wiring IS that call flow.
_PLAIN_ENTRY_METHODS = (
    "run", "run_async", "arun", "__call__", "answer", "ask", "execute", "invoke", "chat", "respond",
)
# Core agentic operations only — data-prep/formatting verbs (load, match, render, extract, inject,
# fetch) are deliberately excluded so the flow shows the agentic components, not internal plumbing.
_AGENTIC_CALL_VERBS = frozenset({
    "plan", "retrieve", "search", "query", "call", "chat", "complete", "generate", "answer",
    "synthesize", "classify", "rank", "rerank", "embed", "route", "dispatch", "invoke", "decompose",
    "summarize", "analyze", "stream", "reason", "act", "observe", "think", "critique", "reflect",
    "score", "judge",
})


def _imports_recognized_framework(tree: ast.AST) -> bool:
    """A file importing Haystack/LangGraph/LangChain is owned by that framework's detector — the
    plain-python detector skips it so the two never compete on the same file."""
    if _imports_langgraph(tree) or _imports_langchain(tree):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "haystack":
            return True
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "haystack" for a in node.names):
            return True
    return False


def _is_agentic_call_name(name: str) -> bool:
    # Token-exact (split on _) — substring matching mis-fires (e.g. "extract" contains "act").
    return any(tok in _AGENTIC_CALL_VERBS for tok in name.lower().lstrip("_").split("_"))


def _detect_plain_python(
    file_contents: dict[str, str],
    cache: dict[str, ast.AST | None] | None = None,
    claimed_files: set[str] | None = None,
    claimed_classes: set[str] | None = None,
) -> WiringBlock | None:
    """Framework-free ('pure python') agent: a *Agent class whose entry method orchestrates a
    sequence of collaborator calls. The wiring graph IS that flow — the agent node feeds a linear
    chain of the agentic-verb calls it makes, in source order. Generic (verb + shape based); never
    keys on a target-specific symbol name.

    `claimed_files`/`claimed_classes` are the source files and component class names already owned by
    a framework detector (e.g. a Haystack agent wired via `add_component` in a pipeline file). Those
    are Haystack/LangGraph components, NOT standalone plain-python agents, so they're excluded — this
    is the real Haystack-vs-plain-python distinction: a plain agent is orchestrated by nothing."""
    claimed_files = claimed_files or set()
    claimed_classes = claimed_classes or set()
    nodes: dict[str, WiringNode] = {}
    edges: list[WiringEdge] = []
    for file, tree in _iter_parsed_files(file_contents, cache):
        if _imports_recognized_framework(tree) or file in claimed_files:
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            if not cls.name.endswith("Agent") or cls.name in claimed_classes:
                continue
            entry = next(
                (m for m in cls.body
                 if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and m.name in _PLAIN_ENTRY_METHODS),
                None,
            )
            if entry is None:
                continue
            seen: set[str] = set()
            steps: list[str] = []
            for sub in sorted(
                (n for n in ast.walk(entry) if isinstance(n, ast.Call)),
                key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)),
            ):
                name = _call_func_name(sub)
                if not name or not _is_agentic_call_name(name):
                    continue
                key = name.lstrip("_")
                if key in seen:
                    continue
                seen.add(key)
                steps.append(name)
            if len(steps) < 2:  # too few calls to be a real orchestration — skip base/abstract classes
                continue
            agent_alias = cls.name
            nodes[agent_alias] = WiringNode(
                alias=agent_alias, callee_name=cls.name, source_hint_file=file,
                entry_kind="class", framework="plain_python",
            )
            prev = agent_alias
            for step in steps:
                alias = step.lstrip("_")
                nodes.setdefault(alias, WiringNode(
                    alias=alias, callee_name=step, source_hint_file=file,
                    entry_kind="function", framework="plain_python",
                ))
                edges.append(WiringEdge(src=prev, dst=alias))
                prev = alias
    if nodes:
        return WiringBlock(nodes=list(nodes.values()), edges=edges, framework="plain_python", source="static")
    return None


def detect_wiring_block_static(file_contents: dict[str, str]) -> WiringBlock | None:
    # Run EVERY static detector and UNION the results (per-node framework tag) rather than
    # first-hit-wins: a mixed-framework cluster (e.g. Haystack + LangGraph in one system) must not
    # let one detector's block starve the others. Shared cache so a file isn't re-parsed per detector.
    cache: dict[str, ast.AST | None] = {}
    blocks: list[WiringBlock] = []
    claimed_files: set[str] = set()
    claimed_classes: set[str] = set()
    for detector in (_detect_haystack, _detect_langgraph, _detect_langchain_lcel):
        result = detector(file_contents, cache)
        if result is not None:
            blocks.append(result)
            for n in result.nodes:
                if n.source_hint_file:
                    claimed_files.add(n.source_hint_file)
                if n.callee_name:
                    claimed_classes.add(n.callee_name)
    # Plain-python runs LAST with what the framework detectors already claimed, so a Haystack agent
    # (wired via add_component from another file) is never mis-detected as a standalone plain agent.
    pp = _detect_plain_python(file_contents, cache, claimed_files, claimed_classes)
    if pp is not None:
        blocks.append(pp)
    return union_wiring_blocks(blocks)


def union_wiring_blocks(blocks: list[WiringBlock]) -> WiringBlock | None:
    """Merge per-detector blocks into one, stamping each node with its producing detector's framework.
    A single contributor is returned with its own framework preserved (keeps single-framework behavior
    identical). Genuine mixes get framework='a+b' and, on a cross-framework alias collision, the later
    alias is namespaced ('alias@framework') with that block's edges rewritten so topology still joins."""
    if not blocks:
        return None
    if len(blocks) == 1:
        only = blocks[0]
        for n in only.nodes:
            n.framework = only.framework
        return only

    merged_nodes: list[WiringNode] = []
    merged_edges: list[WiringEdge] = []
    alias_owner: dict[str, str] = {}  # alias -> framework currently owning it
    for blk in blocks:
        remap: dict[str, str] = {}
        for n in blk.nodes:
            n.framework = blk.framework
            if n.alias in alias_owner and alias_owner[n.alias] != blk.framework:
                new_alias = f"{n.alias}@{blk.framework}"
                remap[n.alias] = new_alias
                n.alias = new_alias
            alias_owner[n.alias] = blk.framework
            merged_nodes.append(n)
        for e in blk.edges:
            merged_edges.append(WiringEdge(src=remap.get(e.src, e.src), dst=remap.get(e.dst, e.dst)))

    framework = "+".join(sorted({blk.framework for blk in blocks}))
    return WiringBlock(nodes=merged_nodes, edges=merged_edges, framework=framework, source="static")


def _split_block_components(block: WiringBlock) -> list[WiringBlock]:
    """Split one PURE-framework block into its connected components (union-find over edges; a node
    touched by no edge is its own singleton). Each component is a distinct system, so a community
    holding two independent same-framework pipelines splits correctly — not merged, and not special-
    cased to one-per-framework. Node objects are preserved (framework tag / entry_kind / owner_class
    / source_hint_file). Edges whose endpoints reference no harvested node (e.g. START/END sentinels
    alone) carry no component and are dropped."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    nodes_by_alias = {n.alias: n for n in block.nodes}
    for alias in nodes_by_alias:
        find(alias)
    # Sentinels (START/END) and dangling aliases legitimately bridge real nodes of the SAME block,
    # keeping one graph as one component; they never merge across blocks (each framework is a
    # separate block already).
    for e in block.edges:
        union(e.src, e.dst)

    comp_nodes: dict[str, list[WiringNode]] = {}
    for alias, node in nodes_by_alias.items():
        comp_nodes.setdefault(find(alias), []).append(node)

    comp_edges: dict[str, list[WiringEdge]] = {}
    for e in block.edges:
        anchor = e.src if e.src in nodes_by_alias else (e.dst if e.dst in nodes_by_alias else None)
        if anchor is not None:
            comp_edges.setdefault(find(anchor), []).append(e)

    return [
        WiringBlock(nodes=ns, edges=comp_edges.get(root, []),
                    framework=block.framework, source=block.source)
        for root, ns in comp_nodes.items()
    ]


def classify_system_type(block: WiringBlock) -> dict:
    """Structural classification of a wiring block into a system type + confidence.
    Deterministic; reuses graph structures already computed. Returns a dict with:
    {kind, type, confidence, candidate_types[], capability_tags[], signals:{...}}
    """
    from agent_eval_harness.mapping.builder.system_types import (
        AGENTIC_SYSTEM_TYPES, SINGLE_AGENT_TYPES, ALL_SYSTEM_TYPES, CAPABILITY_TAGS
    )

    nodes = block.nodes
    edges = block.edges

    # --- Compute structural signals ---
    n_nodes = len(nodes)
    n_edges = len(edges)

    nodes_by_alias = {n.alias: n for n in nodes}
    has_conditional_edge = any(e.conditional for e in edges)
    conditional_sources = sorted({e.src for e in edges if e.conditional})

    # Detect cycle via DFS (back-edge = edge to an ancestor)
    has_cycle = False
    self_loop = any(e.src == e.dst for e in edges)

    if not self_loop and edges:
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs_has_cycle(node: str) -> bool:
            nonlocal has_cycle
            if has_cycle:
                return True
            visited.add(node)
            rec_stack.add(node)
            for e in edges:
                if e.src == node:
                    if e.dst not in visited:
                        if dfs_has_cycle(e.dst):
                            return True
                    elif e.dst in rec_stack:
                        has_cycle = True
                        return True
            rec_stack.discard(node)
            return False

        for n in nodes:
            if n.alias not in visited:
                if dfs_has_cycle(n.alias):
                    break

    # Degree statistics
    from collections import Counter
    out_degrees = Counter(e.src for e in edges)
    in_degrees = Counter(e.dst for e in edges)
    max_out_degree = max(out_degrees.values()) if out_degrees else 0
    max_in_degree = max(in_degrees.values()) if in_degrees else 0

    # Root count (entry points: nodes never a dst)
    dsts = {e.dst for e in edges}
    roots = [n.alias for n in nodes if n.alias not in dsts]
    root_count = len(roots) if roots else len(nodes)

    # Idiom detection from node properties
    react_node = any(
        n.callee_name == "create_react_agent" for n in nodes
    )
    functional_api = any(
        n.entry_kind == "function" and n.callee_name in {n.alias for n in nodes}
        for n in nodes
    )
    plain_python_chain = any(
        n.entry_kind == "class" and n.callee_name.endswith("Agent")
        for n in nodes
    )

    # Retrieval capability detection
    retrieval_verbs = {"retrieve", "search", "rag", "vector", "embed", "query", "lookup", "recall"}
    has_retrieval = any(
        any(verb in (n.callee_name or "").lower().split("_") for verb in retrieval_verbs)
        or any(verb in (n.alias or "").lower().split("_") for verb in retrieval_verbs)
        for n in nodes
    )
    has_retrieval = has_retrieval or any(
        any(verb in (e.dst or "").lower().split("_") for verb in retrieval_verbs)
        for e in edges
    )

    # Tool detection (react agent OR explicit tool-dispatch edges)
    has_tools = react_node or any(
        e.kind == "call" for e in edges  # call edges from node body
    )

    capability_tags = []
    if has_retrieval:
        capability_tags.append("has_retrieval")
    if has_tools:
        capability_tags.append("has_tools")

    signals = {
        "has_conditional_edge": has_conditional_edge,
        "has_cycle": has_cycle,
        "self_loop": self_loop,
        "root_count": root_count,
        "max_out_degree": max_out_degree,
        "max_in_degree": max_in_degree,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "conditional_sources": conditional_sources,
    }

    # --- Classification logic ---
    # Single-agent vs multi-component
    if n_nodes == 1:
        # SINGLE AGENT
        kind = "agent"

        if self_loop or (has_cycle and n_edges >= 1):
            # tool-loop: while-loop of LLM → tool → append
            type_val = "tool-loop"
            confidence = "high"
            candidate_types = []
        elif not has_conditional_edge and not has_cycle:
            # single-flow: one LLM call, no loop
            type_val = "single-flow"
            confidence = "high"
            candidate_types = []
        else:
            # Ambiguous: plan-execute or reflection
            type_val = None
            confidence = "low"
            candidate_types = ["plan-execute", "reflection"]
    else:
        # MULTI-COMPONENT SYSTEM
        kind = "system"

        if not has_cycle:
            # DAG
            if has_conditional_edge:
                # Tie-break: routing vs orchestrator
                # Orchestrator if: 2+ conditional sources OR 1 conditional source with reconvergence
                if len(conditional_sources) >= 2:
                    type_val = "orchestrator"
                    confidence = "high"
                    candidate_types = []
                elif len(conditional_sources) == 1:
                    # Check for reconvergence: any node with max_in_degree >= 2
                    if max_in_degree >= 2:
                        type_val = "orchestrator"
                        confidence = "high"
                        candidate_types = []
                    else:
                        type_val = "routing"
                        confidence = "high"
                        candidate_types = []
                else:
                    # No conditional sources (shouldn't happen if has_conditional_edge=True)
                    type_val = "pipeline"
                    confidence = "high"
                    candidate_types = []
            else:
                # No conditional edge → pipeline
                type_val = "pipeline"
                confidence = "high"
                candidate_types = []
        else:
            # Has cycle
            if has_conditional_edge:
                type_val = "orchestrator"
                confidence = "high"
                candidate_types = []
            else:
                # Cycle but no conditional → evaluator-optimizer, peer-collaboration, debate
                type_val = None
                confidence = "low"
                candidate_types = ["evaluator-optimizer", "peer-collaboration", "debate"]

    return {
        "kind": kind,
        "type": type_val,
        "confidence": confidence,
        "candidate_types": candidate_types,
        "capability_tags": capability_tags,
        "signals": signals,
    }


def detect_wiring_systems(
    file_contents: dict[str, str], cache: dict[str, ast.AST | None] | None = None
) -> list[WiringBlock]:
    """Partition a (possibly mixed) file set into one PURE-framework WiringBlock per distinct agentic
    system = (framework) x (connected wiring component). Runs every static detector over a shared
    parse cache, then splits each detector's pure block into connected components. Returns [] when no
    static wiring exists (caller falls back to the single-candidate / LLM path). Partitions strictly
    by framework + wiring connectivity — never by any target-specific symbol name."""
    if cache is None:
        cache = {}
    systems: list[WiringBlock] = []
    claimed_files: set[str] = set()
    claimed_classes: set[str] = set()
    for detector in (_detect_haystack, _detect_langgraph, _detect_langchain_lcel):
        block = detector(file_contents, cache)
        if block is None:
            continue
        for n in block.nodes:
            n.framework = block.framework
            if n.source_hint_file:
                claimed_files.add(n.source_hint_file)
            if n.callee_name:
                claimed_classes.add(n.callee_name)
        systems.extend(_split_block_components(block))
    # Plain-python LAST, excluding framework-claimed files/classes (a Haystack agent wired via
    # add_component is not a standalone plain agent).
    pp = _detect_plain_python(file_contents, cache, claimed_files, claimed_classes)
    if pp is not None:
        for n in pp.nodes:
            n.framework = pp.framework
        systems.extend(_split_block_components(pp))
    return systems


# ---------------------------------------------------------------------------
# Unscanned-step resolver (CS-318): a plain-python agent composes imported helpers and inherited
# base-class methods, so a wiring step's callee is frequently defined in a file OUTSIDE the scanned
# set and the AST scanner can't see it. Rather than drop the step (a silent shrink) or emit a
# placeholder, resolve its DEFINING file through ordered layers — import (free), base-class/MRO
# (free), then RRF retrieval + AST-confirm (the fallback tail) — and admit that file to the scan set
# so a rescan yields a real component. A step no layer can place is flagged needs_human, never lost.
# Generic: keys on import statements / base classes / symbol names, never any target's own symbols.
# ---------------------------------------------------------------------------


@dataclass
class StepResolution:
    """One plain-python wiring step's resolution outcome. `layer` records WHICH layer placed it,
    so an acceptance run can prove the free layers did the work rather than RRF masking a regression."""
    alias: str
    callee: str
    defining_file: str | None  # repo-relative posix path admitted to the scan set; None when unresolved
    entry_kind: str  # "function" | "bound_method"
    owner_class: str | None
    layer: str  # "import" | "mro" | "rrf" | "unresolved"


def _iter_definition_names(tree: ast.AST) -> set[str]:
    """Every function/class name defined anywhere in the tree (top-level OR nested method)."""
    return {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _file_defines_symbol(tree: ast.AST, name: str) -> bool:
    """AST-confirm the tree defines a top-level-or-nested def/class of exactly `name`. This is the
    noise guard: an RRF hit that merely mentions the symbol but doesn't define it never passes."""
    return name in _iter_definition_names(tree)


def _file_defines_method(tree: ast.AST, method: str, class_name: str) -> bool:
    """AST-confirm `class_name` in this tree has a method named `method` (the MRO-layer confirm)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return any(
                isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == method
                for m in ast.walk(node)
            )
    return False


def _collect_imports(tree: ast.AST) -> dict[str, str]:
    """local_name -> module string for `from M import x`/`x as y` and `import M`/`M as y`."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                out[a.asname or a.name] = node.module
        elif isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = a.name
    return out


def _class_bases(tree: ast.AST, class_name: str) -> list[str]:
    """Base-class NAMES of `class_name` (bare `Base` or dotted `mod.Base` -> "Base")."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            names: list[str] = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    names.append(b.id)
                elif isinstance(b, ast.Attribute):
                    names.append(b.attr)
            return names
    return []


def _match_scanned(scanned_files: list[Path], rel: str) -> Path | None:
    """The scanned Path whose posix path ends with `rel` (or its basename) — locates the agent
    file on disk from a repo-relative wiring hint."""
    if not rel:
        return None
    rel_posix = rel.replace("\\", "/")
    base = rel_posix.rsplit("/", 1)[-1]
    for f in scanned_files:
        fp = f.as_posix()
        if fp.endswith(rel_posix) or fp.rsplit("/", 1)[-1] == base:
            return f
    return None


async def _read_repo_source(
    rel: str, package_root: Path, retrieval: Any, snapshot_id: str | None
) -> str | None:
    """Read a repo file by relative path: disk first (app-path files + fixtures live on disk), then
    the retrieval client's read_file as a fallback. Never raises."""
    try:
        p = package_root / rel
        if p.is_file():
            return p.read_text(encoding="utf-8")
    except OSError:
        pass
    if retrieval is not None and snapshot_id:
        try:
            res = await retrieval.read_file(snapshot_id, rel)
            return res.get("content") if isinstance(res, dict) else None
        except Exception as exc:  # retrieval read is best-effort
            logger.debug("resolver read_file failed for %s: %s", rel, exc)
    return None


def _safe_parse(source: str | None) -> ast.AST | None:
    if not source:
        return None
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


async def _symbol_edge_files(
    retrieval: Any, snapshot_id: str, agent_rel: str, symbol: str
) -> list[str]:
    """Outgoing symbol-graph edges from the agent file whose destination symbol IS `symbol`, mapped
    to their defining file (`file::symbol` -> file). Free (retrieval service, no chat tokens)."""
    try:
        res = await retrieval.get_symbol_edges(snapshot_id, agent_rel)
    except Exception as exc:
        logger.debug("resolver symbol-edges failed for %s: %s", agent_rel, exc)
        return []
    out: list[str] = []
    for e in (res.get("outgoing") or []):
        dst = e.get("dst_symbol") or ""
        if "::" in dst:
            f, s = dst.split("::", 1)
            if s == symbol and f:
                out.append(f.replace("\\", "/"))
    return out


def _module_to_rel(module: str, agent_rel: str, package_root: Path) -> str | None:
    """Map an import module ('domain.analysis.agents.base') to an on-disk repo-relative file, tolerant
    of a source-root prefix ('backend/') the import namespace omits. Tries the module suffix under each
    ancestor prefix of the agent file's own path — deterministic and bounded, never a repo-wide glob."""
    suffix = module.replace(".", "/") + ".py"
    parts = agent_rel.replace("\\", "/").split("/")
    for i in range(len(parts)):
        prefix = "/".join(parts[:i])
        cand = f"{prefix}/{suffix}".lstrip("/") if prefix else suffix
        if (package_root / cand).is_file():
            return cand
    return None


async def _locate_definition_file(
    symbol: str, module_hint: str | None, agent_rel: str, package_root: Path,
    retrieval: Any, snapshot_id: str | None,
) -> str | None:
    """Find a repo file that AST-DEFINES `symbol`, free-first: (1) the import module resolved to a
    disk path (source-root tolerant); (2) a symbol-graph edge from the agent file. Returns the
    AST-confirmed relative path, or None. No repo-wide glob — precision over recall."""
    candidates: list[str] = []
    if module_hint:
        rel = _module_to_rel(module_hint, agent_rel, package_root)
        if rel:
            candidates.append(rel)
    if retrieval is not None and snapshot_id:
        candidates.extend(await _symbol_edge_files(retrieval, snapshot_id, agent_rel, symbol))
    seen: set[str] = set()
    for rel in candidates:
        if rel in seen:
            continue
        seen.add(rel)
        tree = _safe_parse(await _read_repo_source(rel, package_root, retrieval, snapshot_id))
        if tree is not None and _file_defines_symbol(tree, symbol):
            return rel
    return None


async def _rrf_definition_file(
    callee: str, package_root: Path, retrieval: Any, snapshot_id: str,
    require_method_of: list[str] | None = None,
) -> str | None:
    """RRF fallback: fuse a symbol-name query, then AST-CONFIRM each hit actually DEFINES the callee.
    This is where RRF noise dies — a loosely-related hit that doesn't define the symbol is dropped.
    When `require_method_of` is given (an inherited step), a hit must define the callee as a METHOD of
    one of those base classes, so a same-named method on an UNRELATED class (a common ambiguity) is
    rejected rather than admitted. Free (retrieval service, no chat tokens)."""
    try:
        res = await retrieval.search_retrieval(snapshot_id, callee, symbol_chunks_only=True)
    except Exception as exc:
        logger.debug("resolver RRF query failed for %s: %s", callee, exc)
        return None
    hits = (res.get("final") or res.get("fused") or []) if isinstance(res, dict) else []
    seen: set[str] = set()
    for h in hits[:_RESOLVER_RRF_HIT_LIMIT]:
        rel = (h.get("rel_path") or "").replace("\\", "/")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        tree = _safe_parse(await _read_repo_source(rel, package_root, retrieval, snapshot_id))
        if tree is None:
            continue
        if require_method_of:
            if any(_file_defines_method(tree, callee, b) for b in require_method_of):
                return rel
        elif _file_defines_symbol(tree, callee):
            return rel
    return None


async def resolve_unscanned_steps(
    wiring_block: "WiringBlock | None", scanned_files: list[Path], package_root: Path,
    retrieval: Any = None, snapshot_id: str | None = None,
    *, skip_import_layer_for: frozenset[str] = frozenset(),
) -> list[StepResolution]:
    """For a plain-python system, resolve each wiring step whose callee is defined in NO scanned
    file, through ordered layers: import -> base-class/MRO (both free) -> RRF+AST-confirm (fallback)
    -> unresolved. Returns one StepResolution per such step (empty for non-plain systems). Read-only
    on retrieval; spends ZERO chat-provider tokens. `skip_import_layer_for` forces a named callee past
    the import layer (used only by the acceptance harness to exercise the live RRF path deterministically)."""
    if wiring_block is None or "plain_python" not in (wiring_block.framework or ""):
        return []

    scanned_defs: set[str] = set()
    for f in scanned_files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        scanned_defs |= _iter_definition_names(tree)

    agent_nodes = [n for n in wiring_block.nodes if n.entry_kind == "class"]
    step_nodes = [n for n in wiring_block.nodes if n.entry_kind != "class"]

    agent_rel = agent_nodes[0].source_hint_file if agent_nodes else (
        step_nodes[0].source_hint_file if step_nodes else ""
    )
    agent_tree = _safe_parse(await _read_repo_source(agent_rel, package_root, retrieval, snapshot_id)) \
        if agent_rel else None
    imports = _collect_imports(agent_tree) if agent_tree is not None else {}
    base_names: list[str] = []
    if agent_tree is not None:
        for an in agent_nodes:
            base_names.extend(_class_bases(agent_tree, an.callee_name))
    base_names = list(dict.fromkeys(base_names))

    results: list[StepResolution] = []
    for node in step_nodes:
        callee = node.callee_name
        if callee in scanned_defs:
            continue  # the AST scanner already sees it — nothing to resolve

        defining: str | None = None
        layer = "unresolved"
        entry_kind = "function"
        owner: str | None = None

        # Layer 1 — import (free)
        if callee not in skip_import_layer_for and callee in imports:
            defining = await _locate_definition_file(
                callee, imports[callee], agent_rel, package_root, retrieval, snapshot_id
            )
            if defining:
                layer = "import"

        # Layer 2 — base-class / MRO (free). Candidate base files come from resolving each base
        # NAME (via its import) and from the method's own symbol-graph edge; confirm the method
        # actually lives on one of the agent's base classes.
        if defining is None and base_names:
            base_files: list[str] = []
            for base in base_names:
                bf = await _locate_definition_file(
                    base, imports.get(base), agent_rel, package_root, retrieval, snapshot_id
                )
                if bf:
                    base_files.append(bf)
            if retrieval is not None and snapshot_id:
                base_files.extend(await _symbol_edge_files(retrieval, snapshot_id, agent_rel, callee))
            for base_file in dict.fromkeys(base_files):
                tree = _safe_parse(await _read_repo_source(base_file, package_root, retrieval, snapshot_id))
                if tree is None:
                    continue
                match = next((b for b in base_names if _file_defines_method(tree, callee, b)), None)
                if match is not None:
                    defining, layer, entry_kind, owner = base_file, "mro", "bound_method", match
                    break

        # Layer 3 — RRF + AST-confirm (fallback tail). An inherited-shaped step (not imported, agent
        # has bases) must resolve to a method of one of those bases, so RRF cannot mis-pick a
        # same-named method on an unrelated class.
        if defining is None and retrieval is not None and snapshot_id:
            inherited_shaped = callee not in imports and bool(base_names)
            defining = await _rrf_definition_file(
                callee, package_root, retrieval, snapshot_id,
                require_method_of=base_names if inherited_shaped else None,
            )
            if defining:
                layer = "rrf"
                if inherited_shaped:
                    entry_kind, owner = "bound_method", next(
                        (b for b in base_names), None
                    )

        results.append(StepResolution(node.alias, callee, defining, entry_kind, owner, layer))

    return results


# ---------------------------------------------------------------------------
# LangGraph node-body call graph (CS-319): each StateGraph node method delegates to a call graph of
# helpers/collaborators — the "gray" layer. Discover it by walking each node method's AST body, then
# transitively following the agent's OWN-class helper methods (bounded, finite), and RESOLVE every
# call to its repo-defining file via free AST layers: own_class -> same_module -> import -> mro ->
# injected_service. Emission is resolvability-primary (a call that AST-resolves to a repo def is a gray
# component), NOT verb-gated — several real gray targets carry no agentic verb. Excluded: builtins,
# stdlib, pure-data constructors, dunders, nested local defs, and attribute-calls on local vars. No
# RRF here (all targets are repo-local and AST-confirmed). Reuses the CS-318 leaf helpers
# (_collect_imports / _class_bases / _module_to_rel / _file_defines_symbol / _file_defines_method).
# Generic: keys on call SHAPE + import/base structure, never any target's own symbol names.
# ---------------------------------------------------------------------------

_BUILTIN_NAMES = frozenset(dir(builtins))
_PURE_DATA_BASES = frozenset({"TypedDict", "NamedTuple", "BaseModel"})

# Return-annotation wrappers a `with expr as x:` binding unwraps ONE level to find its yielded type
# (Iterator[T]/Generator[T,...] -> T) -- typing/collections.abc vocabulary, not a target convention.
_CONTEXT_MANAGER_WRAPPERS = frozenset({
    "Iterator", "Generator", "AsyncIterator", "AsyncGenerator", "ContextManager", "AsyncContextManager",
})


@dataclass(frozen=True)
class Resolved:
    """The callee resolves to a repo-defined symbol, validated to actually exist there."""
    defining_file: str    # repo-relative posix path (AST-confirmed to define the resolved symbol)
    line: int             # the resolved symbol's OWN def line (method line, not its owning class's)
    entry_kind: str       # "function" | "bound_method" | "class"
    owner_class: str | None
    layer: str            # same_module | own_class | import | mro | injected_service | declared_boundary
    target_name: str      # display/graph alias for the edge (method/function name, or class name for
                           # an injected-service resolution) -- decoupled from the walk-recursion target
    recurse_node: ast.AST | None = None  # the actual def node, when fetched (recursion + return-type reads)


@dataclass(frozen=True)
class OutOfScope:
    """Looked, and the target is deliberately not evidence either way (builtin/external/pure-data/
    untypeable receiver) -- never a confident guess, never silently dropped from the ledger."""
    reason: str


@dataclass(frozen=True)
class Unresolved:
    """Did not resolve at all: the receiver's type (or the callee itself) could not be determined."""


CallResolution = Resolved | OutOfScope | Unresolved


@dataclass
class NodeCallTarget:
    """One resolved gray call target of a LangGraph node (or of a transitively-followed own helper).
    `caller_alias` is the wiring alias of the DIRECT caller (a node or an own-class helper), so the
    emitted edge is the honest call graph, not a flattened node->leaf shortcut."""
    caller_alias: str
    alias: str            # target's wiring alias == its callee name (avoids collision with short node aliases)
    callee: str
    defining_file: str    # repo-relative posix path (AST-confirmed to define `callee`)
    line: int
    entry_kind: str       # "function" | "bound_method" | "class"
    owner_class: str | None
    layer: str            # own_class | same_module | import | mro | injected_service
    resolved_symbol: str = ""  # the actual validated method/function name (== callee, except for an
                               # injected-service edge, where `callee`/`alias` display the CLASS)
    site_id: str | None = None  # the CallSite that produced this edge (span-scoped boundary lookups)


@dataclass
class CallSite:
    """One enumerated call site in a component's call closure — a receipt ledger a downstream
    verdict layer can cite as evidence."""
    site_id: str
    file: str
    lineno: int
    callee_text: str
    source_line: str
    outcome: str = "unresolved"        # "resolved" | "out_of_scope" | "unresolved"
    reason: str | None = None          # set for "out_of_scope" (why); the OutOfScope.reason value
    external_module: str | None = None  # import module string, only for reason == "external_import"


def _module_level_defs(tree: ast.AST) -> dict[str, ast.AST]:
    """Top-level def/class nodes of a module, by name (same-module resolution + pure-data check)."""
    return {
        n.name: n for n in getattr(tree, "body", [])
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _class_def_node(tree: ast.AST, class_name: str) -> ast.ClassDef | None:
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            return n
    return None


def _own_class_methods(tree: ast.AST, class_name: str) -> dict[str, ast.AST]:
    """Direct method defs of `class_name` (bounds the transitive closure to the agent's own class)."""
    cls = _class_def_node(tree, class_name)
    if cls is None:
        return {}
    return {
        m.name: m for m in cls.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _is_pure_data_classdef(node: ast.AST) -> bool:
    """A ClassDef that only carries data (dataclass / TypedDict / NamedTuple / pydantic BaseModel) —
    a constructor call on it is state, not an agentic step, so it is excluded from the gray layer."""
    if not isinstance(node, ast.ClassDef):
        return False
    for d in node.decorator_list:
        if _decorator_effective_name(d) == "dataclass":
            return True
    for b in node.bases:
        name = b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else None)
        if name in _PURE_DATA_BASES:
            return True
    return False


def _self_attr_types(tree: ast.AST, class_name: str) -> dict[str, str]:
    """Map self.<attr> -> injected class NAME for a class's __init__: both `self.x = param` where the
    param is annotated with a class, and `self.x = Class(...)`. Powers injected-service resolution."""
    cls = _class_def_node(tree, class_name)
    if cls is None:
        return {}
    init = next((m for m in cls.body
                 if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == "__init__"), None)
    if init is None:
        return {}
    param_types: dict[str, str] = {}
    for arg in init.args.args + init.args.kwonlyargs:
        if arg.annotation is not None:
            ann = arg.annotation
            nm = ann.id if isinstance(ann, ast.Name) else (ann.attr if isinstance(ann, ast.Attribute) else None)
            if nm:
                param_types[arg.arg] = nm
    out: dict[str, str] = {}
    for node in ast.walk(init):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == "self"):
            attr = node.targets[0].attr
            val = node.value
            if isinstance(val, ast.Name) and val.id in param_types:
                out[attr] = param_types[val.id]
            elif isinstance(val, ast.Call):
                cn = _call_func_name(val)
                if cn:
                    out[attr] = cn
    return out


def _self_attr_containers(tree: ast.AST, class_name: str) -> dict[str, list[tuple[str, str]]]:
    """Map self.<attr> -> [(element_name, shape), ...] for a `self.<attr> = [...]` / `{...}` / `(...)`
    literal built in __init__, where each element is a bare Name ('bare') or a self.<method> ref
    ('self') — a dispatch table built in the agent's own scope, subscripted and called later, e.g.
    `self._h = [self._a, self._b]` then `self._h[i](...)`."""
    cls = _class_def_node(tree, class_name)
    if cls is None:
        return {}
    init = next((m for m in cls.body
                 if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == "__init__"), None)
    if init is None:
        return {}
    out: dict[str, list[tuple[str, str]]] = {}
    for node in ast.walk(init):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == "self"
                and isinstance(node.value, (ast.List, ast.Tuple, ast.Dict))):
            continue
        attr = node.targets[0].attr
        elts = node.value.values if isinstance(node.value, ast.Dict) else node.value.elts
        elements: list[tuple[str, str]] = []
        for elt in elts:
            if isinstance(elt, ast.Name):
                elements.append((elt.id, "bare"))
            elif (isinstance(elt, ast.Attribute) and isinstance(elt.value, ast.Name)
                  and elt.value.id == "self"):
                elements.append((elt.attr, "self"))
        if elements:
            out[attr] = elements
    return out


def _container_call_sites(
    method_node: ast.AST, self_containers: dict[str, list[tuple[str, str]]],
) -> list[tuple[str, str, str | None, ast.Call]]:
    """Calls shaped `self.<attr>[<key>](...)` where `<attr>` is a tracked container literal: expand
    to one (callee, shape, receiver, call_node) record PER element — every element is a possible
    dispatch target, a deliberate over-approximation (same idiom as the LCEL BitOr path)."""
    out: list[tuple[str, str, str | None, ast.Call]] = []
    for node in ast.walk(method_node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)):
            continue
        sub = node.func.value
        if not (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self"):
            continue
        for name, shape in self_containers.get(sub.attr, []):
            out.append((name, shape, None, node))
    return out


def _untracked_container_call_sites(
    method_node: ast.AST, self_containers: dict[str, list[tuple[str, str]]],
) -> list[tuple[str, str, str | None, ast.Call]]:
    """`self.<attr>[<key>](...)` where `<attr>` is NOT a tracked container literal (e.g. a dict handed
    in through the constructor): the element set isn't statically knowable, so this is recorded as
    ONE 'container_dispatch' site per distinct attr rather than silently vanishing from the ledger."""
    out: list[tuple[str, str, str | None, ast.Call]] = []
    seen: set[str] = set()
    for node in ast.walk(method_node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)):
            continue
        sub = node.func.value
        if not (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self"):
            continue
        if sub.attr in self_containers or sub.attr in seen:
            continue
        seen.add(sub.attr)
        out.append((sub.attr, "container_dispatch", None, node))
    return out


def _classify_call_shape(
    node: ast.Call, nested_defs: frozenset[str] = frozenset(),
) -> tuple[str, str, str | None] | None:
    """One call node -> (callee, shape, receiver), or None when the shape isn't a resolvable one.
    Shapes: 'bare' (f()), 'self' (self.f()), 'name_attr' (X.f(), X a Name), 'self_attr'
    (self.<recv>.f()), 'super_attr' (super().f())."""
    f = node.func
    if isinstance(f, ast.Name):
        return (f.id, "bare", None) if f.id not in nested_defs else None
    if isinstance(f, ast.Attribute):
        v = f.value
        if isinstance(v, ast.Name) and v.id == "self":
            return (f.attr, "self", None)
        if isinstance(v, ast.Name):
            return (f.attr, "name_attr", v.id)
        if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id == "self":
            return (f.attr, "self_attr", v.attr)
        if isinstance(v, ast.Call) and _call_func_name(v) == "super":
            return (f.attr, "super_attr", None)
    return None


def _iter_method_calls(method_node: ast.AST) -> list[tuple[str, str, str | None, ast.Call]]:
    """Every call SITE inside a method body, classified by shape (nested local-def names excluded as
    callees) -- container-subscript calls are handled separately by _container_call_sites /
    _untracked_container_call_sites."""
    nested_defs = frozenset(
        n.name for n in ast.walk(method_node)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not method_node
    )
    out: list[tuple[str, str, str | None, ast.Call]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for node in ast.walk(method_node):
        if not isinstance(node, ast.Call):
            continue
        shape_key = _classify_call_shape(node, nested_defs)
        if shape_key is not None and shape_key not in seen:
            seen.add(shape_key)
            out.append((*shape_key, node))
    return out


def _annotation_type_name(annotation: ast.expr) -> str | None:
    """Base type name from an annotation: `T` -> "T", `mod.T` -> "T", `list[str]` -> "list"
    (peels one Subscript level so a builtin container annotation is still recognizable)."""
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    if isinstance(annotation, ast.Subscript):
        return _annotation_type_name(annotation.value)
    return None


def _local_var_types(method_node: ast.AST) -> dict[str, str]:
    """Receiver type names written down directly in this method's own scope: its OWN parameter
    annotations, `x: T = ...`, and `x = T(...)`. Downstream validation (does T actually exist and
    declare the callee) is what filters a non-type name out -- same contract as _self_attr_types."""
    out: dict[str, str] = {}
    args = getattr(method_node, "args", None)
    if args is not None:
        for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            if arg.arg in ("self", "cls") or arg.annotation is None:
                continue
            nm = _annotation_type_name(arg.annotation)
            if nm:
                out[arg.arg] = nm
    for node in ast.walk(method_node):
        if node is method_node:
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            nm = _annotation_type_name(node.annotation)
            if nm:
                out[node.target.id] = nm
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)):
            cn = _call_func_name(node.value)
            if cn:
                out[node.targets[0].id] = cn
    return out


def _context_manager_yield_type_name(returns: ast.expr | None) -> str | None:
    """The type a `with expr as x:` binds `x` to, read from `expr`'s own declared return annotation:
    peel ONE Iterator/Generator/ContextManager wrapper level, then take the bare type name. `Any`
    (spelled out or absent) means genuinely nothing is known -- returns None, never a guess."""
    if returns is None:
        return None
    target = returns
    if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
            and target.value.id in _CONTEXT_MANAGER_WRAPPERS):
        sl = target.slice
        target = sl.elts[0] if isinstance(sl, ast.Tuple) and sl.elts else sl
    name = _annotation_type_name(target)
    if name is None or name == "Any":
        return None
    return name


async def _confirm_def_line(
    rel: str, symbol: str, package_root: Path, retrieval: Any, snapshot_id: str | None,
    *, as_method_of: str | None = None,
) -> tuple[bool, int]:
    """AST-confirm `rel` defines `symbol` (as a top-level/nested def OR, when as_method_of is given, as
    a method of that class) and return (True, lineno). No confirmation -> (False, 0)."""
    tree = _safe_parse(await _read_repo_source(rel, package_root, retrieval, snapshot_id))
    if tree is None:
        return False, 0
    if as_method_of is not None:
        cls = _class_def_node(tree, as_method_of)
        if cls is None:
            return False, 0
        for m in ast.walk(cls):
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == symbol:
                return True, m.lineno
        return False, 0
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == symbol:
            return True, n.lineno
    return False, 0


async def _resolve_via_mro(
    callee: str, base_names: list[str], *,
    agent_rel: str, package_root: Path, module_defs: dict[str, ast.AST], imports: dict[str, str],
    retrieval: Any, snapshot_id: str | None,
) -> "Resolved | Unresolved":
    """`callee` as a method of one of `base_names` (inherited via `self.f()` when not own-class, or
    via `super().f()`) -- first base that declares it wins. A base defined in the SAME module is
    checked directly (module_defs), same as self_attr's same-module fallback; only when it isn't
    found there does the base name get treated as an import to resolve cross-file."""
    for base in base_names:
        if base in module_defs and isinstance(module_defs[base], ast.ClassDef):
            for m in ast.walk(module_defs[base]):
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == callee:
                    return Resolved(agent_rel, m.lineno, "bound_method", base, "mro", callee)
            continue
        brel = _module_to_rel(imports.get(base, base), agent_rel, package_root)
        if not brel:
            continue
        ok, line = await _confirm_def_line(brel, callee, package_root, retrieval, snapshot_id, as_method_of=base)
        if ok:
            return Resolved(brel, line, "bound_method", base, "mro", callee)
    return Unresolved()


async def _resolve_method_on_type(
    cls_name: str, callee: str, *,
    agent_rel: str, package_root: Path, module_defs: dict[str, ast.AST], imports: dict[str, str],
    retrieval: Any, snapshot_id: str | None,
) -> "Resolved | OutOfScope | Unresolved":
    """Validate-then-emit for a receiver already typed to `cls_name` (self_attr injected service, or a
    tier-c local/with-as binding): the method must actually exist on that type, or the answer is
    Unresolved rather than a guessed edge to the class. `target_name` (the display alias) is always
    the CLASS, never the method -- the method is the walk-recursion target, kept separate."""
    if cls_name in _BUILTIN_TYPE_NAMES:
        return OutOfScope("builtin_receiver")
    if cls_name in module_defs:
        node = module_defs[cls_name]
        if not isinstance(node, ast.ClassDef):
            return Unresolved()
        if _is_pure_data_classdef(node):
            return OutOfScope("pure_data_classdef")
        method = next(
            (m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == callee),
            None,
        )
        if method is None:
            return Unresolved()
        return Resolved(agent_rel, method.lineno, "class", cls_name, "injected_service", cls_name, method)
    if cls_name in imports:
        rel = _module_to_rel(imports[cls_name], agent_rel, package_root)
        if rel is None:
            return OutOfScope("external_import")
        ok, line = await _confirm_def_line(rel, callee, package_root, retrieval, snapshot_id, as_method_of=cls_name)
        if not ok:
            return Unresolved()
        dtree = _safe_parse(await _read_repo_source(rel, package_root, retrieval, snapshot_id))
        cls_node = _class_def_node(dtree, cls_name) if dtree else None
        if cls_node is not None and _is_pure_data_classdef(cls_node):
            return OutOfScope("pure_data_classdef")
        method = next(
            (m for m in cls_node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == callee),
            None,
        ) if cls_node is not None else None
        return Resolved(rel, line, "class", cls_name, "injected_service", cls_name, method)
    return Unresolved()


async def resolve_call_target(
    callee: str, shape: str, receiver: str | None, *,
    agent_rel: str, package_root: Path, module_defs: dict[str, ast.AST], own_methods: dict[str, ast.AST],
    imports: dict[str, str], base_names: list[str], self_types: dict[str, str],
    receiver_types: dict[str, str] | None = None, receiver_unknowable: frozenset[str] = frozenset(),
    owner: str | None = None, retrieval: Any = None, snapshot_id: str | None = None,
) -> "Resolved | OutOfScope | Unresolved":
    """Framework-independent call-target resolution: same_module -> own_class -> import -> mro ->
    injected_service, plus tier-c receiver typing (name_attr on a local var) and super_attr (mro).
    Module-level (not nested inside a framework-specific function) so call-pair emission is not
    conditional on framework. `receiver_types` (name_attr locals) and `receiver_unknowable`
    (with-as locals whose bound call declares no concrete type) come from the caller's own scope."""
    receiver_types = receiver_types or {}
    if shape == "bare":
        if callee in _BUILTIN_NAMES:
            return OutOfScope("builtin")
        if callee in module_defs:  # same-module top-level def/class
            node = module_defs[callee]
            if _is_pure_data_classdef(node):
                return OutOfScope("pure_data_classdef")
            ek = "class" if isinstance(node, ast.ClassDef) else "function"
            return Resolved(agent_rel, node.lineno, ek, None, "same_module", callee, node)
        if callee in imports:  # imported module-level fn/class
            rel = _module_to_rel(imports[callee], agent_rel, package_root)
            if rel is None:
                return OutOfScope("external_import")
            ok, line = await _confirm_def_line(rel, callee, package_root, retrieval, snapshot_id)
            if not ok:
                return Unresolved()
            dtree = _safe_parse(await _read_repo_source(rel, package_root, retrieval, snapshot_id))
            dnode = _module_level_defs(dtree).get(callee) if dtree else None
            if dnode is not None and _is_pure_data_classdef(dnode):
                return OutOfScope("pure_data_classdef")
            ek = "class" if isinstance(dnode, ast.ClassDef) else "function"
            return Resolved(rel, line, ek, None, "import", callee, dnode)
        return Unresolved()
    if shape == "self":
        if callee in own_methods:  # own-class helper -> gray component + recurse
            method = own_methods[callee]
            return Resolved(agent_rel, method.lineno, "bound_method", owner, "own_class", callee, method)
        return await _resolve_via_mro(
            callee, base_names, agent_rel=agent_rel, package_root=package_root,
            module_defs=module_defs, imports=imports, retrieval=retrieval, snapshot_id=snapshot_id,
        )
    if shape == "super_attr":
        return await _resolve_via_mro(
            callee, base_names, agent_rel=agent_rel, package_root=package_root,
            module_defs=module_defs, imports=imports, retrieval=retrieval, snapshot_id=snapshot_id,
        )
    if shape == "container_dispatch":
        return OutOfScope("container_dispatch_unknowable")
    if shape == "name_attr":
        if receiver in receiver_unknowable:
            return OutOfScope("receiver_type_unknowable")
        if receiver in receiver_types:
            return await _resolve_method_on_type(
                receiver_types[receiver], callee, agent_rel=agent_rel, package_root=package_root,
                module_defs=module_defs, imports=imports, retrieval=retrieval, snapshot_id=snapshot_id,
            )
        if receiver in imports:  # imported module: mod.f()
            rel = _module_to_rel(imports[receiver], agent_rel, package_root)
            if rel is None:
                return OutOfScope("external_import")
            ok, line = await _confirm_def_line(rel, callee, package_root, retrieval, snapshot_id)
            if not ok:
                return Unresolved()
            return Resolved(rel, line, "function", None, "import", callee)
        return Unresolved()  # attribute call on an untyped local var -- genuinely unresolvable
    if shape == "self_attr":
        cls_name = self_types.get(receiver)  # injected service class (same-module or imported)
        if cls_name is None:
            return Unresolved()
        return await _resolve_method_on_type(
            cls_name, callee, agent_rel=agent_rel, package_root=package_root,
            module_defs=module_defs, imports=imports, retrieval=retrieval, snapshot_id=snapshot_id,
        )
    return Unresolved()


async def _with_as_receiver_types(
    method_node: ast.AST, *,
    agent_rel: str, package_root: Path, module_defs: dict[str, ast.AST], own_methods: dict[str, ast.AST],
    imports: dict[str, str], base_names: list[str], self_types: dict[str, str], owner: str | None,
    retrieval: Any, snapshot_id: str | None,
) -> tuple[dict[str, str], frozenset[str]]:
    """`with expr as x:` bindings in this method -> (concrete types, unknowable names). `expr` is
    resolved through the SAME layers as any other call; a concrete type comes from a constructor
    call's class or a factory function's own return annotation (peeled one context-manager level) --
    Any or an unresolved `expr` means the binding is unknowable, never guessed."""
    types_out: dict[str, str] = {}
    unknowable: set[str] = set()
    for node in ast.walk(method_node):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            if not (isinstance(item.context_expr, ast.Call) and isinstance(item.optional_vars, ast.Name)):
                continue
            var_name = item.optional_vars.id
            shape_key = _classify_call_shape(item.context_expr)
            if shape_key is None:
                unknowable.add(var_name)
                continue
            c_callee, c_shape, c_receiver = shape_key
            resolved = await resolve_call_target(
                c_callee, c_shape, c_receiver, agent_rel=agent_rel, package_root=package_root,
                module_defs=module_defs, own_methods=own_methods, imports=imports,
                base_names=base_names, self_types=self_types, owner=owner,
                retrieval=retrieval, snapshot_id=snapshot_id,
            )
            if not isinstance(resolved, Resolved):
                unknowable.add(var_name)
                continue
            if resolved.entry_kind == "class":
                types_out[var_name] = c_callee
                continue
            def_node = resolved.recurse_node
            yield_name = (
                _context_manager_yield_type_name(getattr(def_node, "returns", None))
                if isinstance(def_node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
            )
            if yield_name is None:
                unknowable.add(var_name)
            else:
                types_out[var_name] = yield_name
    return types_out, frozenset(unknowable)


@dataclass
class _ClassCtx:
    """Everything the closure walk needs about ONE (file, class) pair -- built once per pair and
    cached, since cross-class recursion (injected-service methods) visits several of these."""
    rel: str
    tree: ast.AST
    class_name: str | None
    imports: dict[str, str]
    base_names: list[str]
    module_defs: dict[str, ast.AST]
    own_methods: dict[str, ast.AST]
    self_types: dict[str, str]
    self_containers: dict[str, list[tuple[str, str]]]


async def walk_call_closure(
    entry_points: list[tuple[str, ast.AST]], *, owner: str | None, agent_tree: ast.AST, agent_rel: str,
    package_root: Path, retrieval: Any = None, snapshot_id: str | None = None,
) -> tuple[list[NodeCallTarget], list["CallSite"], bool]:
    """Shared closure walk: from `entry_points` (alias, method/function node), transitively over the
    owner's own-class helpers AND any resolved injected-service method (cross-class recursion),
    resolving every call via resolve_call_target (incl. container literals and tier-c receiver
    typing). The visited-set -- keyed on (file, class, method), since two unrelated classes both
    define e.g. `run_async` -- IS the termination condition, no hop-count constant. Returns
    (targets, call_sites, complete): `complete` is False the moment any call site is genuinely
    Unresolved (an OutOfScope site does not count against it); `call_sites` is the enumerated
    receipt ledger, each carrying its own resolution outcome."""
    tree_cache: dict[str, ast.AST | None] = {agent_rel: agent_tree}
    ctx_cache: dict[tuple[str, str], _ClassCtx] = {}

    def ctx_for(rel: str, tree: ast.AST, class_name: str | None) -> _ClassCtx:
        key = (rel, class_name or "")
        ctx = ctx_cache.get(key)
        if ctx is None:
            ctx = _ClassCtx(
                rel=rel, tree=tree, class_name=class_name,
                imports=_collect_imports(tree),
                base_names=_class_bases(tree, class_name) if class_name else [],
                module_defs=_module_level_defs(tree),
                own_methods=_own_class_methods(tree, class_name) if class_name else {},
                self_types=_self_attr_types(tree, class_name) if class_name else {},
                self_containers=_self_attr_containers(tree, class_name) if class_name else {},
            )
            ctx_cache[key] = ctx
        return ctx

    async def tree_for(rel: str) -> ast.AST | None:
        if rel not in tree_cache:
            tree_cache[rel] = _safe_parse(await _read_repo_source(rel, package_root, retrieval, snapshot_id))
        return tree_cache[rel]

    root_ctx = ctx_for(agent_rel, agent_tree, owner)

    results: list[NodeCallTarget] = []
    call_sites: list[CallSite] = []
    emitted_edges: set[tuple[str, str]] = set()
    complete = True
    site_counter = 0

    worklist: list[tuple[str, ast.AST, _ClassCtx]] = [
        (alias, node, root_ctx) for alias, node in entry_points
    ]
    walked: set[tuple[str, str, str]] = {
        (root_ctx.rel, root_ctx.class_name or "", alias) for alias, _ in entry_points
    }

    while worklist:
        caller_alias, method_node, ctx = worklist.pop(0)
        calls = (
            _iter_method_calls(method_node)
            + _container_call_sites(method_node, ctx.self_containers)
            + _untracked_container_call_sites(method_node, ctx.self_containers)
        )
        receiver_types = _local_var_types(method_node)
        with_types, with_unknowable = await _with_as_receiver_types(
            method_node, agent_rel=ctx.rel, package_root=package_root,
            module_defs=ctx.module_defs, own_methods=ctx.own_methods, imports=ctx.imports,
            base_names=ctx.base_names, self_types=ctx.self_types, owner=ctx.class_name,
            retrieval=retrieval, snapshot_id=snapshot_id,
        )
        receiver_types.update(with_types)

        for callee, shape, receiver, call_node in calls:
            site_counter += 1
            site_id = f"{caller_alias}#{site_counter}"
            lineno = getattr(call_node, "lineno", 0)
            resolved = await resolve_call_target(
                callee, shape, receiver, agent_rel=ctx.rel, package_root=package_root,
                module_defs=ctx.module_defs, own_methods=ctx.own_methods, imports=ctx.imports,
                base_names=ctx.base_names, self_types=ctx.self_types,
                receiver_types=receiver_types, receiver_unknowable=with_unknowable,
                owner=ctx.class_name, retrieval=retrieval, snapshot_id=snapshot_id,
            )
            if isinstance(resolved, Unresolved):
                call_sites.append(CallSite(
                    site_id=site_id, file=ctx.rel, lineno=lineno, callee_text=callee,
                    source_line="", outcome="unresolved",
                ))
                complete = False
                if _is_agentic_call_name(callee):  # coordinator refinement: warn, don't silently drop
                    logger.warning(
                        "call-closure target unresolved by any layer -- dropped: caller=%s callee=%s shape=%s",
                        caller_alias, callee, shape,
                    )
                continue
            if isinstance(resolved, OutOfScope):
                external_module = None
                if resolved.reason == "external_import":
                    external_module = ctx.imports.get(receiver if shape == "name_attr" else callee)
                call_sites.append(CallSite(
                    site_id=site_id, file=ctx.rel, lineno=lineno, callee_text=callee,
                    source_line="", outcome="out_of_scope", reason=resolved.reason,
                    external_module=external_module,
                ))
                continue

            # Resolved.
            target_alias = resolved.target_name
            if (caller_alias, target_alias) not in emitted_edges:
                emitted_edges.add((caller_alias, target_alias))
                results.append(NodeCallTarget(
                    caller_alias=caller_alias, alias=target_alias, callee=target_alias,
                    defining_file=resolved.defining_file, line=resolved.line,
                    entry_kind=resolved.entry_kind, owner_class=resolved.owner_class,
                    layer=resolved.layer, resolved_symbol=callee, site_id=site_id,
                ))
            call_sites.append(CallSite(
                site_id=site_id, file=ctx.rel, lineno=lineno, callee_text=callee,
                source_line="", outcome="resolved",
            ))

            # Transitively follow own-class helpers and resolved injected-service methods (finite:
            # each (file, class, method) walked once, across the whole closure).
            recurse_node = resolved.recurse_node
            if (resolved.layer in ("own_class", "injected_service")
                    and isinstance(recurse_node, (ast.FunctionDef, ast.AsyncFunctionDef))):
                next_class = ctx.class_name if resolved.layer == "own_class" else resolved.owner_class
                walk_key = (resolved.defining_file, next_class or "", callee)
                if walk_key not in walked:
                    walked.add(walk_key)
                    next_tree = ctx.tree if resolved.defining_file == ctx.rel else await tree_for(resolved.defining_file)
                    if next_tree is not None:
                        next_ctx = ctx_for(resolved.defining_file, next_tree, next_class)
                        worklist.append((target_alias, recurse_node, next_ctx))

    return results, call_sites, complete


async def resolve_langgraph_node_calls(
    wiring_block: "WiringBlock | None", package_root: Path,
    retrieval: Any = None, snapshot_id: str | None = None,
) -> list[NodeCallTarget]:
    """Walk each LangGraph StateGraph node method's body + transitively its agent's own-class helpers,
    resolving every repo-local call to a gray component (resolvability-primary; AST-confirmed; no RRF).
    Returns one NodeCallTarget per (caller, resolved target). Empty for non-langgraph systems. Spends
    ZERO chat tokens; retrieval is only a read-file fallback when a file is off-disk. Thin wrapper over
    resolve_langgraph_node_calls_full for callers that don't need the per-node closure info."""
    targets, _call_sites_by_node, _complete_by_node, _targets_by_node = await resolve_langgraph_node_calls_full(
        wiring_block, package_root, retrieval, snapshot_id,
    )
    return targets


async def resolve_langgraph_node_calls_full(
    wiring_block: "WiringBlock | None", package_root: Path,
    retrieval: Any = None, snapshot_id: str | None = None,
) -> tuple[list[NodeCallTarget], dict[str, list[CallSite]], dict[str, bool], dict[str, list[NodeCallTarget]]]:
    """Full variant: also returns, PER NODE ALIAS, its own call-sites ledger, closure-completeness
    bit, and resolved-target list -- each node is walked independently (a shared helper is walked once
    per caller, redundant but cheap: pure AST, no network) so all three are genuinely per-component,
    not merged across every node of the owning class."""
    if wiring_block is None or "langgraph" not in (wiring_block.framework or ""):
        return [], {}, {}, {}
    node_methods = [n for n in wiring_block.nodes if n.entry_kind == "bound_method" and n.owner_class]
    if not node_methods:
        return [], {}, {}, {}

    results: list[NodeCallTarget] = []
    call_sites_by_node: dict[str, list[CallSite]] = {}
    complete_by_node: dict[str, bool] = {}
    targets_by_node: dict[str, list[NodeCallTarget]] = {}

    # Group nodes by (owner_class, agent_file); parse each agent file once.
    by_agent: dict[tuple[str, str], list[WiringNode]] = {}
    for n in node_methods:
        by_agent.setdefault((n.owner_class, n.source_hint_file), []).append(n)

    for (owner, agent_rel), nodes in by_agent.items():
        agent_tree = _safe_parse(await _read_repo_source(agent_rel, package_root, retrieval, snapshot_id))
        if agent_tree is None:
            continue
        own_methods = _own_class_methods(agent_tree, owner)
        for n in nodes:
            entry_node = own_methods.get(n.callee_name)
            if entry_node is None:
                continue
            targets, call_sites, complete = await walk_call_closure(
                [(n.alias, entry_node)], owner=owner, agent_tree=agent_tree, agent_rel=agent_rel,
                package_root=package_root, retrieval=retrieval, snapshot_id=snapshot_id,
            )
            results.extend(targets)
            # Keyed by wiring_identity(owner, callee_name) -- the SAME key extract_topology's
            # class_to_candidate_ids uses for a bound-method candidate, so the caller can attach
            # this info to the right Component without a second alias namespace.
            component_key = wiring_identity(owner, n.callee_name)
            call_sites_by_node[component_key] = call_sites
            complete_by_node[component_key] = complete
            targets_by_node[component_key] = targets

    return results, call_sites_by_node, complete_by_node, targets_by_node


async def resolve_generic_call_closure(
    candidates: list[Any], package_root: Path,
    retrieval: Any = None, snapshot_id: str | None = None,
) -> tuple[list[NodeCallTarget], dict[str, list[CallSite]], dict[str, bool], dict[str, list[NodeCallTarget]]]:
    """Framework-independent counterpart to resolve_langgraph_node_calls: walk the call closure of
    any candidate whose own class exposes a recognized entry method, or of a bare module-level
    function candidate -- regardless of framework. A LangGraph node method (entry_kind='bound_method')
    is left to resolve_langgraph_node_calls, which already covers it; this widens the SAME mechanism
    to Haystack/plain-python-shaped classes and functions. `caller_alias` on each returned target is
    the CALLING candidate's own class_name (or bare function name) -- the same key extract_topology's
    class_to_candidate_ids already resolves wiring edges through, so no new alias-namespace is needed.
    Returns (targets, call_sites_by_caller, complete_by_caller, targets_by_caller)."""
    from agent_eval_harness.mapping.builder.resolution_primitives import recognized_entry_methods

    results: list[NodeCallTarget] = []
    call_sites_by_caller: dict[str, list[CallSite]] = {}
    complete_by_caller: dict[str, bool] = {}
    targets_by_caller: dict[str, list[NodeCallTarget]] = {}
    trees: dict[str, ast.AST | None] = {}

    for cand in candidates:
        if getattr(cand, "entry_kind", None) not in ("class", "function") or not getattr(cand, "file", None):
            continue
        try:
            agent_rel = cand.file.relative_to(package_root).as_posix()
        except ValueError:
            agent_rel = cand.file.as_posix()
        if agent_rel not in trees:
            trees[agent_rel] = _safe_parse(await _read_repo_source(agent_rel, package_root, retrieval, snapshot_id))
        agent_tree = trees[agent_rel]
        if agent_tree is None:
            continue

        caller_alias = cand.class_name
        if cand.entry_kind == "class":
            owner = cand.class_name
            # Walk EVERY recognized entry method present, not just the first name that matches -- a
            # sync `run` that only raises NotImplementedError alongside the real `run_async` is a
            # common async-only idiom, and picking one by fixed priority would silently walk the stub.
            entry_points = [
                (caller_alias, m_node) for m_node in recognized_entry_methods(agent_tree, owner)
            ]
        else:  # "function"
            owner = None
            entry_node = _module_level_defs(agent_tree).get(cand.class_name)
            entry_points = [(caller_alias, entry_node)] if entry_node is not None else []
        if not entry_points:
            continue

        targets, call_sites, complete = await walk_call_closure(
            entry_points, owner=owner, agent_tree=agent_tree, agent_rel=agent_rel,
            package_root=package_root, retrieval=retrieval, snapshot_id=snapshot_id,
        )
        results.extend(targets)
        call_sites_by_caller[caller_alias] = call_sites
        complete_by_caller[caller_alias] = complete
        targets_by_caller[caller_alias] = targets

    return results, call_sites_by_caller, complete_by_caller, targets_by_caller


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
