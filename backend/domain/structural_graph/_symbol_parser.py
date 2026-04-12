"""Symbol parser — CS-202.

Extracts symbol definitions, imports, call sites, and attribute assignments
from Python and TypeScript source files.

All parsing is syntax-only; no type inference is performed here.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImportInfo:
    """A single imported name visible in a file's namespace.

    Attributes:
        local_name:  The name as it is referenced in this file.
        src_file:    The file that exports the symbol (resolved best-effort).
                     None if unresolvable (e.g. stdlib or unrecognised path).
        orig_name:   The exported name in the source file.
                     None for ``import module`` style (local_name == module).
        is_star:     True when this entry comes from a ``from X import *``
                     statement.  The actual exported names are unknown.
    """

    local_name: str
    src_file: str | None
    orig_name: str | None
    is_star: bool = False


@dataclass
class SymbolInfo:
    """A definition (function, method, or class) extracted from a file.

    Attributes:
        fqn:          Fully-qualified name in ``file::Class.method`` form.
        filename:     Source file path as given to parse_file().
        class_name:   Enclosing class name, or None for module-level symbols.
        method_name:  Simple name of the function/method/class.
        kind:         ``"function"``, ``"method"``, or ``"class"``.
        line_start:   1-based line number of the definition.
        base_classes: Direct base class names (populated for classes only).
    """

    fqn: str
    filename: str
    class_name: str | None
    method_name: str
    kind: str
    line_start: int
    base_classes: list[str] = field(default_factory=list)


@dataclass
class AttributeAssign:
    """A ``self.attr = SomeClass()`` assignment observed inside a class.

    Attributes:
        class_name:    Class in which the assignment appears.
        attr_name:     Attribute name (``self.<attr_name>``).
        assigned_type: Name of the class being constructed (right-hand side).
        line:          1-based line number of the assignment.
    """

    class_name: str
    attr_name: str
    assigned_type: str
    line: int


@dataclass
class CallSite:
    """A call expression observed inside a function or method body.

    Attributes:
        caller_fqn:   FQN of the enclosing function/method.
        receiver:     Attribute-access receiver, e.g. ``self._graph`` for
                      ``self._graph.build()``.  None for bare calls.
        method_name:  The called function/method name.
        line:         1-based line number of the call.
        is_self_call: True when the call is ``self.method()``, meaning
                      receiver resolution should use the caller's class.
    """

    caller_fqn: str
    receiver: str | None
    method_name: str
    line: int
    is_self_call: bool = False


@dataclass
class ParsedFile:
    """All symbol information extracted from a single source file.

    Attributes:
        filename:              Source file path.
        language:              ``"python"`` or ``"typescript"``.
        definitions:           Map from FQN to SymbolInfo.
        imports:               Map from local_name to ImportInfo.
        call_sites:            All call sites found in the file.
        attribute_assignments: All ``self.attr = Type()`` assignments.
        inheritance:           Map from class name to list of base class names.
    """

    filename: str
    language: str
    definitions: dict[str, SymbolInfo] = field(default_factory=dict)
    imports: dict[str, ImportInfo] = field(default_factory=dict)
    call_sites: list[CallSite] = field(default_factory=list)
    attribute_assignments: list[AttributeAssign] = field(default_factory=list)
    inheritance: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(filename: str, source: str) -> ParsedFile | None:
    """Parse a source file and return extracted symbol information.

    Args:
        filename: File path (used for FQN construction and language detection).
        source:   Full source text.

    Returns:
        A :class:`ParsedFile`, or ``None`` if the file could not be parsed.
    """
    try:
        lower = filename.lower()
        if lower.endswith(".py"):
            return _parse_python(filename, source)
        if lower.endswith((".ts", ".tsx")):
            return _parse_typescript(filename, source)
        return None
    except Exception:
        logger.exception("parse_file failed for %s", filename)
        return None


# ---------------------------------------------------------------------------
# Python parser (ast module)
# ---------------------------------------------------------------------------

def _parse_python(filename: str, source: str) -> ParsedFile:
    result = ParsedFile(filename=filename, language="python")
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return result

    # Pass 1 — collect imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname if alias.asname else alias.name
                result.imports[local] = ImportInfo(
                    local_name=local,
                    src_file=_module_to_file(alias.name),
                    orig_name=alias.name,
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            src_file = _module_to_file(module)
            for alias in node.names:
                if alias.name == "*":
                    result.imports[f"*:{module}"] = ImportInfo(
                        local_name=f"*:{module}",
                        src_file=src_file,
                        orig_name=None,
                        is_star=True,
                    )
                else:
                    local = alias.asname if alias.asname else alias.name
                    result.imports[local] = ImportInfo(
                        local_name=local,
                        src_file=src_file,
                        orig_name=alias.name,
                    )

    # Pass 2 — collect class/function definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = [_ast_name(b) for b in node.bases if _ast_name(b)]
            fqn = f"{filename}::{node.name}"
            sym = SymbolInfo(
                fqn=fqn,
                filename=filename,
                class_name=None,
                method_name=node.name,
                kind="class",
                line_start=node.lineno,
                base_classes=bases,
            )
            result.definitions[fqn] = sym
            result.inheritance[node.name] = bases

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mfqn = f"{filename}::{node.name}.{item.name}"
                    result.definitions[mfqn] = SymbolInfo(
                        fqn=mfqn,
                        filename=filename,
                        class_name=node.name,
                        method_name=item.name,
                        kind="method",
                        line_start=item.lineno,
                    )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Module-level functions only (skip those inside classes)
            parent_is_class = any(
                isinstance(p, ast.ClassDef)
                for p in ast.walk(tree)
                if isinstance(p, ast.ClassDef) and any(
                    isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c is node
                    for c in ast.walk(p)
                    if c is not p
                )
            )
            if not parent_is_class:
                fqn = f"{filename}::{node.name}"
                if fqn not in result.definitions:
                    result.definitions[fqn] = SymbolInfo(
                        fqn=fqn,
                        filename=filename,
                        class_name=None,
                        method_name=node.name,
                        kind="function",
                        line_start=node.lineno,
                    )

    # Pass 3 — collect call sites and attribute assignments using a visitor
    visitor = _PythonVisitor(filename, result.definitions)
    visitor.visit(tree)
    result.call_sites = visitor.call_sites
    result.attribute_assignments = visitor.attribute_assignments

    return result


class _PythonVisitor(ast.NodeVisitor):
    """Extracts call sites and attribute assignments from a Python AST."""

    def __init__(self, filename: str, definitions: dict[str, SymbolInfo]) -> None:
        self.filename = filename
        self.definitions = definitions
        self.call_sites: list[CallSite] = []
        self.attribute_assignments: list[AttributeAssign] = []
        self._class_stack: list[str] = []
        self._func_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._func_stack.append(node.name)
        # Scan for self.attr = Type() assignments
        if self._class_stack:
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and isinstance(stmt.value, ast.Call)
                        ):
                            ctor_name = _ast_call_name(stmt.value)
                            if ctor_name:
                                self.attribute_assignments.append(
                                    AttributeAssign(
                                        class_name=self._class_stack[-1],
                                        attr_name=target.attr,
                                        assigned_type=ctor_name,
                                        line=stmt.lineno,
                                    )
                                )
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        caller_fqn = self._current_fqn()
        if caller_fqn is None:
            self.generic_visit(node)
            return

        func = node.func
        if isinstance(func, ast.Attribute):
            receiver_str = _ast_expr_str(func.value)
            is_self = isinstance(func.value, ast.Name) and func.value.id == "self"
            self.call_sites.append(
                CallSite(
                    caller_fqn=caller_fqn,
                    receiver=receiver_str,
                    method_name=func.attr,
                    line=node.lineno,
                    is_self_call=is_self,
                )
            )
        elif isinstance(func, ast.Name):
            self.call_sites.append(
                CallSite(
                    caller_fqn=caller_fqn,
                    receiver=None,
                    method_name=func.id,
                    line=node.lineno,
                )
            )

        self.generic_visit(node)

    def _current_fqn(self) -> str | None:
        if not self._func_stack:
            return None
        if self._class_stack:
            return f"{self.filename}::{self._class_stack[-1]}.{self._func_stack[-1]}"
        return f"{self.filename}::{self._func_stack[-1]}"


# ---------------------------------------------------------------------------
# TypeScript parser (tree-sitter)
# ---------------------------------------------------------------------------

def _parse_typescript(filename: str, source: str) -> ParsedFile:
    result = ParsedFile(filename=filename, language="typescript")
    try:
        from domain.repo_map._loaders import _load_ts_language, _ts_parse, _node_text
    except ImportError:
        logger.warning("tree-sitter not available; skipping TS parse for %s", filename)
        return result

    lang = _load_ts_language("typescript")
    if lang is None:
        return result

    root = _ts_parse(source, lang)
    if root is None:
        return result

    lines = source.splitlines()

    def text_at(node: object) -> str:
        return _node_text(node) or ""

    def node_line(node: object) -> int:
        try:
            return node.start_point[0] + 1  # type: ignore[union-attr]
        except Exception:
            return 0

    # Track interface names so we can apply EC-7 logic later
    interface_names: set[str] = set()

    # Pass 1 — imports
    for node in _ts_walk(root):
        try:
            if node.type == "import_statement":
                _parse_ts_import(node, filename, result, text_at)
        except Exception:
            pass

    # Pass 2 — definitions (class, interface, function)
    _current_class: list[str] = []

    for node in _ts_walk(root):
        try:
            if node.type in ("class_declaration", "abstract_class_declaration"):
                cname_node = node.child_by_field_name("name")
                cname = text_at(cname_node) if cname_node else None
                if cname:
                    bases: list[str] = []
                    # heritage clause: implements / extends
                    for child in node.children:
                        if child.type in ("class_heritage", "implements_clause", "extends_clause"):
                            for t in _ts_walk(child):
                                if t.type == "type_identifier":
                                    bases.append(text_at(t))
                    fqn = f"{filename}::{cname}"
                    result.definitions[fqn] = SymbolInfo(
                        fqn=fqn,
                        filename=filename,
                        class_name=None,
                        method_name=cname,
                        kind="class",
                        line_start=node_line(node),
                        base_classes=bases,
                    )
                    result.inheritance[cname] = bases

            elif node.type == "interface_declaration":
                iname_node = node.child_by_field_name("name")
                iname = text_at(iname_node) if iname_node else None
                if iname:
                    interface_names.add(iname)
                    fqn = f"{filename}::{iname}"
                    result.definitions[fqn] = SymbolInfo(
                        fqn=fqn,
                        filename=filename,
                        class_name=None,
                        method_name=iname,
                        kind="interface",
                        line_start=node_line(node),
                    )

            elif node.type in ("function_declaration", "function"):
                fname_node = node.child_by_field_name("name")
                fname = text_at(fname_node) if fname_node else None
                if fname:
                    fqn = f"{filename}::{fname}"
                    if fqn not in result.definitions:
                        result.definitions[fqn] = SymbolInfo(
                            fqn=fqn,
                            filename=filename,
                            class_name=None,
                            method_name=fname,
                            kind="function",
                            line_start=node_line(node),
                        )
        except Exception:
            pass

    # Pass 3 — methods inside classes
    for node in _ts_walk(root):
        try:
            if node.type in ("class_declaration", "abstract_class_declaration"):
                cname_node = node.child_by_field_name("name")
                cname = text_at(cname_node) if cname_node else None
                if not cname:
                    continue
                body = node.child_by_field_name("body")
                if body is None:
                    continue
                for child in body.children:
                    if child.type in ("method_definition", "public_field_definition"):
                        mname_node = child.child_by_field_name("name")
                        mname = text_at(mname_node) if mname_node else None
                        if mname:
                            mfqn = f"{filename}::{cname}.{mname}"
                            result.definitions[mfqn] = SymbolInfo(
                                fqn=mfqn,
                                filename=filename,
                                class_name=cname,
                                method_name=mname,
                                kind="method",
                                line_start=node_line(child),
                            )
        except Exception:
            pass

    # Pass 4 — call sites and constructor assignments
    _parse_ts_calls(root, filename, result, text_at, node_line, interface_names)

    return result


def _parse_ts_import(
    node: object,
    filename: str,
    result: ParsedFile,
    text_at: object,
) -> None:
    """Extract import bindings from a TypeScript import_statement node."""
    source_node = None
    clause_node = None
    for child in node.children:  # type: ignore[union-attr]
        if child.type == "string":
            source_node = child
        elif child.type == "import_clause":
            clause_node = child

    if source_node is None:
        return

    raw_path = text_at(source_node).strip("'\"")  # type: ignore[operator]
    src_file = _ts_module_to_file(raw_path)

    if clause_node is None:
        return

    for child in _ts_walk(clause_node):
        if child.type == "namespace_import":
            # import * as X — treat as star
            alias_node = child.child_by_field_name("name")
            local = text_at(alias_node) if alias_node else None
            if local:
                result.imports[local] = ImportInfo(
                    local_name=local,
                    src_file=src_file,
                    orig_name=None,
                    is_star=True,
                )
        elif child.type == "named_imports":
            for spec in child.children:
                if spec.type == "import_specifier":
                    name_node = spec.child_by_field_name("name")
                    alias_node = spec.child_by_field_name("alias")
                    orig = text_at(name_node) if name_node else None
                    local = text_at(alias_node) if alias_node else orig
                    if local and orig:
                        result.imports[local] = ImportInfo(
                            local_name=local,
                            src_file=src_file,
                            orig_name=orig,
                        )
        elif child.type == "identifier":
            # default import: import X from '...'
            local = text_at(child)
            if local:
                result.imports[local] = ImportInfo(
                    local_name=local,
                    src_file=src_file,
                    orig_name="default",
                )


def _parse_ts_calls(
    root: object,
    filename: str,
    result: ParsedFile,
    text_at: object,
    node_line: object,
    interface_names: set[str],
) -> None:
    """Walk the TS AST to collect call sites and constructor assignments."""
    # Build a map: method body node -> caller FQN
    # We do a full walk keeping track of class/method context
    class_stack: list[str] = []
    method_stack: list[str] = []

    def caller_fqn() -> str | None:
        if not method_stack:
            return None
        if class_stack:
            return f"{filename}::{class_stack[-1]}.{method_stack[-1]}"
        return f"{filename}::{method_stack[-1]}"

    def walk_node(node: object) -> None:
        try:
            ntype = node.type  # type: ignore[union-attr]
        except Exception:
            return

        pushed_class = False
        pushed_method = False

        try:
            if ntype in ("class_declaration", "abstract_class_declaration"):
                cname_node = node.child_by_field_name("name")  # type: ignore[union-attr]
                cname = text_at(cname_node) if cname_node else None
                if cname:
                    class_stack.append(cname)
                    pushed_class = True

            elif ntype in ("method_definition", "function_declaration", "function", "arrow_function"):
                mname_node = node.child_by_field_name("name")  # type: ignore[union-attr]
                mname = text_at(mname_node) if mname_node else None
                if mname:
                    method_stack.append(mname)
                    pushed_method = True

            elif ntype == "call_expression":
                fqn = caller_fqn()
                if fqn:
                    func_node = node.child_by_field_name("function")  # type: ignore[union-attr]
                    if func_node and func_node.type == "member_expression":
                        obj_node = func_node.child_by_field_name("object")
                        prop_node = func_node.child_by_field_name("property")
                        receiver = text_at(obj_node) if obj_node else None
                        method_name = text_at(prop_node) if prop_node else None
                        if method_name:
                            is_self = receiver in ("this", "self")
                            result.call_sites.append(
                                CallSite(
                                    caller_fqn=fqn,
                                    receiver=receiver,
                                    method_name=method_name,
                                    line=node_line(node),  # type: ignore[operator]
                                    is_self_call=is_self,
                                )
                            )
                    elif func_node and func_node.type == "identifier":
                        mname_called = text_at(func_node)
                        if mname_called:
                            result.call_sites.append(
                                CallSite(
                                    caller_fqn=fqn,
                                    receiver=None,
                                    method_name=mname_called,
                                    line=node_line(node),  # type: ignore[operator]
                                )
                            )

            elif ntype == "new_expression":
                # this.x = new ClassName() in constructor
                constructor_node = node.child_by_field_name("constructor")  # type: ignore[union-attr]
                ctor_type = text_at(constructor_node) if constructor_node else None
                if ctor_type and class_stack and method_stack and method_stack[-1] == "constructor":
                    # Look for parent assignment: this.x = new ...
                    pass  # handled via expression_statement walk below

        except Exception:
            pass

        try:
            for child in node.children:  # type: ignore[union-attr]
                walk_node(child)
        except Exception:
            pass

        if pushed_method:
            if method_stack:
                method_stack.pop()
        if pushed_class:
            if class_stack:
                class_stack.pop()

    # Also scan for this.x = new ClassName() in constructor methods
    def scan_ctor_assignments(node: object, class_name: str) -> None:
        try:
            if node.type in ("expression_statement",):  # type: ignore[union-attr]
                for child in node.children:  # type: ignore[union-attr]
                    if child.type == "assignment_expression":
                        left = child.child_by_field_name("left")
                        right = child.child_by_field_name("right")
                        if (
                            left is not None
                            and right is not None
                            and left.type == "member_expression"
                        ):
                            obj = left.child_by_field_name("object")
                            prop = left.child_by_field_name("property")
                            if obj and text_at(obj) in ("this", "self") and prop:
                                attr = text_at(prop)
                                if right.type == "new_expression":
                                    ctor_node = right.child_by_field_name("constructor")
                                    ctor_type = text_at(ctor_node) if ctor_node else None
                                    if attr and ctor_type:
                                        result.attribute_assignments.append(
                                            AttributeAssign(
                                                class_name=class_name,
                                                attr_name=attr,
                                                assigned_type=ctor_type,
                                                line=node_line(node),  # type: ignore[operator]
                                            )
                                        )
            for child in node.children:  # type: ignore[union-attr]
                scan_ctor_assignments(child, class_name)
        except Exception:
            pass

    # Run constructor assignment scan per class
    for node in _ts_walk(root):
        try:
            if node.type in ("class_declaration", "abstract_class_declaration"):
                cname_node = node.child_by_field_name("name")
                cname = text_at(cname_node) if cname_node else None
                if cname:
                    body = node.child_by_field_name("body")
                    if body:
                        for child in body.children:
                            if child.type == "method_definition":
                                mname_node = child.child_by_field_name("name")
                                mname = text_at(mname_node) if mname_node else None
                                if mname == "constructor":
                                    scan_ctor_assignments(child, cname)
        except Exception:
            pass

    walk_node(root)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_walk(node: object):
    """Breadth-first walk of a tree-sitter node."""
    queue = [node]
    while queue:
        current = queue.pop(0)
        yield current
        try:
            queue.extend(current.children)  # type: ignore[union-attr]
        except Exception:
            pass


def _module_to_file(module: str) -> str | None:
    """Best-effort conversion of a Python module name to a file path."""
    if not module:
        return None
    return module.replace(".", "/") + ".py"


def _ts_module_to_file(path: str) -> str | None:
    """Best-effort conversion of a TS import path to a file path."""
    if not path:
        return None
    if path.startswith("."):
        for ext in (".ts", ".tsx", ".js"):
            if path.endswith(ext):
                return path
        return path + ".ts"
    return None


def _ast_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _ast_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _ast_expr_str(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _ast_expr_str(node.value)
        return f"{inner}.{node.attr}" if inner else node.attr
    return None
