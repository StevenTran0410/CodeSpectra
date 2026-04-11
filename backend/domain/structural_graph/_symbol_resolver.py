"""Symbol resolver — CS-202.

Resolves call sites to concrete SymbolEdge objects using import namespace
tracking, constructor-assignment analysis, and MRO-style inheritance walking.

Confidence levels:
  CONF_HIGH  — exactly one unambiguous resolution path
  CONF_LOW   — multiple candidates or interface-typed receiver
  CONF_NONE  — unresolvable; caller MUST NOT emit an edge for these

The public entry point is :func:`resolve_edges`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .symbol_graph import SymbolEdge
from ._symbol_parser import (
    AttributeAssign,
    CallSite,
    ImportInfo,
    ParsedFile,
    SymbolInfo,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence constants
# ---------------------------------------------------------------------------

CONF_HIGH = "high"
CONF_LOW = "low"
CONF_NONE = "none"

# Maximum MRO depth to prevent infinite loops in pathological inheritance
_MAX_MRO_DEPTH = 20


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def build_definition_index(
    parsed_files: dict[str, ParsedFile],
) -> dict[str, SymbolInfo]:
    """Return a flat O(1) lookup from FQN to SymbolInfo across all files.

    Args:
        parsed_files: All parsed files keyed by filename.

    Returns:
        Dict mapping FQN strings to :class:`~._symbol_parser.SymbolInfo`.
    """
    index: dict[str, SymbolInfo] = {}
    for pf in parsed_files.values():
        index.update(pf.definitions)
    return index


def build_constructor_index(
    parsed_files: dict[str, ParsedFile],
) -> dict[tuple[str, str, str], list[str]]:
    """Return an index from (filename, class_name, attr_name) to assigned types.

    Multiple entries in the list indicate the attribute is reassigned in more
    than one method — making it ambiguous (EC-3).

    Args:
        parsed_files: All parsed files keyed by filename.

    Returns:
        Dict mapping ``(filename, class_name, attr_name)`` to a list of
        assigned constructor type names.
    """
    index: dict[tuple[str, str, str], list[str]] = {}
    for pf in parsed_files.values():
        for assign in pf.attribute_assignments:
            key = (pf.filename, assign.class_name, assign.attr_name)
            index.setdefault(key, [])
            if assign.assigned_type not in index[key]:
                index[key].append(assign.assigned_type)
    return index


def build_inheritance_index(
    parsed_files: dict[str, ParsedFile],
) -> dict[str, list[str]]:
    """Return a flat map from class name to list of direct base class names.

    Collects inheritance across ALL parsed files.

    Args:
        parsed_files: All parsed files keyed by filename.

    Returns:
        Dict mapping class name to base class name list.
    """
    index: dict[str, list[str]] = {}
    for pf in parsed_files.values():
        for cls, bases in pf.inheritance.items():
            index.setdefault(cls, [])
            for b in bases:
                if b not in index[cls]:
                    index[cls].append(b)
    return index


# ---------------------------------------------------------------------------
# Call site resolution
# ---------------------------------------------------------------------------

def resolve_call_site(
    call: CallSite,
    caller_file: ParsedFile,
    all_files: dict[str, ParsedFile],
    def_index: dict[str, SymbolInfo],
    ctor_index: dict[tuple[str, str, str], list[str]],
    inh_index: dict[str, list[str]],
) -> list[SymbolEdge]:
    """Resolve a single call site to zero or more SymbolEdge objects.

    Confidence decision tree:

    1. ``receiver`` is None → bare function call.
       - Look up ``call.method_name`` in the caller file's import namespace.
       - If found with a known src_file, look for exactly one definition
         ``src_file::method_name`` → CONF_HIGH.
       - If from a star import → return [] (EC-9).
       - If not found in imports, search def_index for same-filename
         definitions only.

    2. ``is_self_call`` is True → ``self.method()`` call.
       - Determine caller's class from caller_fqn.
       - Look for method definition in that class first.
       - Walk MRO via inh_index until found or exhausted.
       - CONF_HIGH if single base found in def_index, CONF_LOW otherwise.

    3. ``receiver`` is ``self.attr`` style.
       - Look up ``(caller_file.filename, class_name, attr)`` in ctor_index.
       - 0 types → CONF_NONE (EC-6 duck typing also falls here).
       - 1 type → resolve to that type's method → CONF_HIGH (EC-2).
       - 2+ types → emit one edge per type with CONF_LOW (EC-3).

    4. Any case where receiver type resolves to an interface name →
       collect all implementing classes → CONF_LOW (EC-7).

    Returns:
        A list of :class:`~.symbol_graph.SymbolEdge` objects.
        Unresolvable call sites return an empty list (never CONF_NONE edges).

    Args:
        call:        The call site to resolve.
        caller_file: The ParsedFile containing this call site.
        all_files:   All ParsedFile objects keyed by filename.
        def_index:   Output of :func:`build_definition_index`.
        ctor_index:  Output of :func:`build_constructor_index`.
        inh_index:   Output of :func:`build_inheritance_index`.
    """
    try:
        return _resolve(call, caller_file, all_files, def_index, ctor_index, inh_index)
    except Exception:
        logger.exception(
            "resolve_call_site failed for %s in %s",
            call.method_name,
            caller_file.filename,
        )
        return []


def _resolve(
    call: CallSite,
    caller_file: ParsedFile,
    all_files: dict[str, ParsedFile],
    def_index: dict[str, SymbolInfo],
    ctor_index: dict[tuple[str, str, str], list[str]],
    inh_index: dict[str, list[str]],
) -> list[SymbolEdge]:
    # Collect interface names from all files for EC-7 detection
    interface_names: set[str] = set()
    for pf in all_files.values():
        for sym in pf.definitions.values():
            if sym.kind == "interface":
                interface_names.add(sym.method_name)

    # ------------------------------------------------------------------ #
    # Case 1 — bare function call (no receiver)
    # ------------------------------------------------------------------ #
    if call.receiver is None:
        return _resolve_bare_call(call, caller_file, def_index)

    # ------------------------------------------------------------------ #
    # Case 2 — self.method() call (direct method, inheritance)
    # ------------------------------------------------------------------ #
    if call.is_self_call and call.receiver in ("self", "this"):
        return _resolve_self_call(call, caller_file, def_index, inh_index)

    # ------------------------------------------------------------------ #
    # Case 3 — self.attr.method() call (attribute receiver)
    # ------------------------------------------------------------------ #
    # Extract the attribute name from a "self.attr" receiver
    attr_name = _extract_self_attr(call.receiver)
    if attr_name:
        class_name = _caller_class(call.caller_fqn)
        if class_name is None:
            return []
        key = (caller_file.filename, class_name, attr_name)
        assigned_types = ctor_index.get(key, [])
        if not assigned_types:
            return []  # EC-6: no type trace

        edges: list[SymbolEdge] = []
        confidence = CONF_HIGH if len(assigned_types) == 1 else CONF_LOW

        for type_name in assigned_types:
            # Resolve the type name via imports if needed
            resolved_type = _resolve_type_name(type_name, caller_file, def_index)
            if resolved_type is None:
                continue

            # EC-7: if resolved_type is an interface, find implementors
            if resolved_type in interface_names:
                implementors = _find_implementors(resolved_type, all_files, def_index)
                for impl_class in implementors:
                    target_fqn = _find_method_fqn(impl_class, call.method_name, def_index, inh_index)
                    if target_fqn:
                        edges.append(
                            SymbolEdge(
                                src_symbol=call.caller_fqn,
                                dst_symbol=target_fqn,
                                edge_type="calls",
                                confidence=CONF_LOW,
                                evidence_lines=[call.line],
                            )
                        )
                continue

            target_fqn = _find_method_fqn(resolved_type, call.method_name, def_index, inh_index)
            if target_fqn:
                edges.append(
                    SymbolEdge(
                        src_symbol=call.caller_fqn,
                        dst_symbol=target_fqn,
                        edge_type="calls",
                        confidence=confidence,
                        evidence_lines=[call.line],
                    )
                )
        return edges

    # ------------------------------------------------------------------ #
    # Case 4 — other attribute receiver (e.g. module.func())
    # ------------------------------------------------------------------ #
    return _resolve_module_call(call, caller_file, def_index)


def _resolve_bare_call(
    call: CallSite,
    caller_file: ParsedFile,
    def_index: dict[str, SymbolInfo],
) -> list[SymbolEdge]:
    method = call.method_name

    # Check for star import that covers this name
    for imp in caller_file.imports.values():
        if imp.is_star:
            # Under a star import we can't know — return empty (EC-9)
            return []

    # Look in import namespace
    imp_info = caller_file.imports.get(method)
    if imp_info:
        if imp_info.is_star:
            return []
        target_fqn = _lookup_import_def(imp_info, method, def_index)
        if target_fqn:
            return [
                SymbolEdge(
                    src_symbol=call.caller_fqn,
                    dst_symbol=target_fqn,
                    edge_type="calls",
                    confidence=CONF_HIGH,
                    evidence_lines=[call.line],
                )
            ]
        return []

    # Not imported — look in same file only
    candidates = [
        fqn
        for fqn, sym in def_index.items()
        if sym.filename == caller_file.filename and sym.method_name == method
    ]
    if len(candidates) == 1:
        return [
            SymbolEdge(
                src_symbol=call.caller_fqn,
                dst_symbol=candidates[0],
                edge_type="calls",
                confidence=CONF_HIGH,
                evidence_lines=[call.line],
            )
        ]
    return []


def _resolve_self_call(
    call: CallSite,
    caller_file: ParsedFile,
    def_index: dict[str, SymbolInfo],
    inh_index: dict[str, list[str]],
) -> list[SymbolEdge]:
    class_name = _caller_class(call.caller_fqn)
    if class_name is None:
        return []

    target_fqn = _find_method_fqn(class_name, call.method_name, def_index, inh_index)
    if target_fqn:
        return [
            SymbolEdge(
                src_symbol=call.caller_fqn,
                dst_symbol=target_fqn,
                edge_type="calls",
                confidence=CONF_HIGH,
                evidence_lines=[call.line],
            )
        ]
    return []


def _resolve_module_call(
    call: CallSite,
    caller_file: ParsedFile,
    def_index: dict[str, SymbolInfo],
) -> list[SymbolEdge]:
    """Handle module.func() style calls via import namespace."""
    if call.receiver is None:
        return []
    # receiver might be an imported module name
    imp_info = caller_file.imports.get(call.receiver)
    if imp_info and not imp_info.is_star and imp_info.src_file:
        target_fqn = _lookup_import_def(
            ImportInfo(
                local_name=call.method_name,
                src_file=imp_info.src_file,
                orig_name=call.method_name,
            ),
            call.method_name,
            def_index,
        )
        if target_fqn:
            return [
                SymbolEdge(
                    src_symbol=call.caller_fqn,
                    dst_symbol=target_fqn,
                    edge_type="calls",
                    confidence=CONF_HIGH,
                    evidence_lines=[call.line],
                )
            ]
    return []


# ---------------------------------------------------------------------------
# Public resolve_edges entry point
# ---------------------------------------------------------------------------

def resolve_edges(parsed_files: dict[str, ParsedFile]) -> list[SymbolEdge]:
    """Resolve all call sites in all files to SymbolEdge objects.

    Args:
        parsed_files: All parsed files keyed by filename.

    Returns:
        Deduplicated list of :class:`~.symbol_graph.SymbolEdge` objects.
    """
    def_index = build_definition_index(parsed_files)
    ctor_index = build_constructor_index(parsed_files)
    inh_index = build_inheritance_index(parsed_files)

    seen: set[tuple[str, str, str, str]] = set()
    edges: list[SymbolEdge] = []

    for pf in parsed_files.values():
        for call in pf.call_sites:
            try:
                new_edges = resolve_call_site(
                    call, pf, parsed_files, def_index, ctor_index, inh_index
                )
                for edge in new_edges:
                    key = (edge.src_symbol, edge.dst_symbol, edge.edge_type, edge.confidence)
                    if key not in seen:
                        seen.add(key)
                        edges.append(edge)
            except Exception:
                logger.exception(
                    "resolve_edges: failed call site %s in %s",
                    call.method_name,
                    pf.filename,
                )

    return edges


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _caller_class(fqn: str) -> str | None:
    """Extract class name from ``file::ClassName.method`` FQN."""
    if "::" not in fqn:
        return None
    _, rest = fqn.split("::", 1)
    if "." in rest:
        return rest.split(".")[0]
    return None


def _extract_self_attr(receiver: str | None) -> str | None:
    """Return ``attr`` from ``self.attr`` receiver string, else None."""
    if receiver is None:
        return None
    if receiver.startswith("self."):
        return receiver[5:]
    if receiver.startswith("this."):
        return receiver[5:]
    return None


def _find_method_fqn(
    class_name: str,
    method_name: str,
    def_index: dict[str, SymbolInfo],
    inh_index: dict[str, list[str]],
    depth: int = 0,
) -> str | None:
    """Walk the MRO of class_name looking for method_name.

    Returns the first FQN found, or None if not found within _MAX_MRO_DEPTH.
    """
    if depth > _MAX_MRO_DEPTH:
        return None

    # Search def_index for any file that defines class_name.method_name
    for fqn, sym in def_index.items():
        if sym.class_name == class_name and sym.method_name == method_name:
            return fqn

    # Walk base classes
    for base in inh_index.get(class_name, []):
        result = _find_method_fqn(base, method_name, def_index, inh_index, depth + 1)
        if result:
            return result

    return None


def _lookup_import_def(
    imp_info: ImportInfo,
    method_name: str,
    def_index: dict[str, SymbolInfo],
) -> str | None:
    """Return the FQN for a symbol given its ImportInfo."""
    if imp_info.src_file is None:
        return None
    # Try exact file match across all known file variants
    for fqn, sym in def_index.items():
        if (
            sym.method_name == (imp_info.orig_name or method_name)
            and _file_matches(sym.filename, imp_info.src_file)
            and sym.class_name is None  # module-level only for bare imports
        ):
            return fqn
    return None


def _file_matches(actual: str, expected: str) -> bool:
    """Check whether two file paths refer to the same file (suffix match)."""
    a = actual.replace("\\", "/")
    e = expected.replace("\\", "/")
    return a == e or a.endswith("/" + e) or e.endswith("/" + a)


def _resolve_type_name(
    type_name: str,
    caller_file: ParsedFile,
    def_index: dict[str, SymbolInfo],
) -> str | None:
    """Resolve a bare type name to the definitive class name in def_index.

    Checks the caller file's import namespace first, then falls back to the
    type_name as-is if a class with that name exists anywhere in def_index.
    """
    # Check imports
    imp_info = caller_file.imports.get(type_name)
    if imp_info and not imp_info.is_star:
        return imp_info.orig_name or type_name

    # Look for class defined in same file
    for sym in def_index.values():
        if sym.method_name == type_name and sym.kind in ("class", "interface"):
            return type_name

    return type_name


def _find_implementors(
    interface_name: str,
    all_files: dict[str, ParsedFile],
    def_index: dict[str, SymbolInfo],
) -> list[str]:
    """Return class names that implement or extend the given interface."""
    implementors: list[str] = []
    for pf in all_files.values():
        for cls_name, bases in pf.inheritance.items():
            if interface_name in bases:
                implementors.append(cls_name)
    return implementors
