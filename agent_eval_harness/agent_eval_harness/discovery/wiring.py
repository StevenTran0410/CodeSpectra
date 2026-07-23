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
            out.append(WiringEdge(src=src, dst=dst))
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
