"""PlantUML C4 syntax parser → {nodes, edges} JSON (CS-219).

Converts LLM-generated PlantUML C4 text into the structured dict format
consumed by the React Flow + ELK renderer on the frontend.

Supported macros
----------------
Nodes : Person, Person_Ext, System, System_Ext, Container, ContainerDb,
        ContainerQueue, Component, ComponentDb
Edges : Rel, BiRel, Rel_D, Rel_U, Rel_L, Rel_R,
        Rel_Down, Rel_Up, Rel_Left, Rel_Right

Argument conventions (PlantUML C4 standard)
--------------------------------------------
Person / System_Ext / System : (alias, label, ?description)
Container / Component        : (alias, label, ?technology, ?description)
Rel / BiRel                  : (from, to, label, ?technology)
"""

from __future__ import annotations

import re
from typing import Any

# ── node type map ─────────────────────────────────────────────────────────────

_NODE_TYPE: dict[str, str] = {
    "Person":         "person",
    "Person_Ext":     "person",
    "System":         "container",
    "SystemDb":       "containerDb",
    "System_Ext":     "systemExt",
    "SystemExt":      "systemExt",
    "Container":      "container",
    "ContainerDb":    "containerDb",
    "ContainerQueue": "container",
    "Component":      "component",
    "ComponentDb":    "containerDb",
}

_NODE_MACROS   = frozenset(_NODE_TYPE)
_REL_MACROS    = frozenset({
    "Rel", "BiRel",
    "Rel_D", "Rel_U", "Rel_L", "Rel_R",
    "Rel_Down", "Rel_Up", "Rel_Left", "Rel_Right",
})
_BIREL_MACROS  = frozenset({"BiRel"})

# macros whose arg order is (alias, label, description)   — no technology slot
_NO_TECH_NODES = frozenset({"Person", "Person_Ext", "System", "System_Ext", "SystemExt"})

# ── argument parser ───────────────────────────────────────────────────────────

def _parse_args(raw: str) -> list[str]:
    """Split comma-separated args, respecting double-quoted strings."""
    args: list[str] = []
    buf = ""
    in_q = False
    for ch in raw:
        if ch == '"' and not in_q:
            in_q = True
        elif ch == '"' and in_q:
            in_q = False
        elif ch == "," and not in_q:
            args.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        args.append(buf.strip())
    return args


# ── main parser ───────────────────────────────────────────────────────────────

# Matches:  MacroName(anything not containing a closing paren)
# We allow multi-char macro names including underscores.
_MACRO_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(([^)]*)\)", re.MULTILINE)


def parse_plantuml_c4(text: str) -> dict[str, Any]:
    """Parse a PlantUML C4 block into ``{"nodes": [...], "edges": [...]}``.

    Unknown / unsupported macros are silently skipped.
    Duplicate node aliases are ignored (first wins).
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    edge_idx = 0

    for m in _MACRO_RE.finditer(text):
        macro = m.group(1)
        args  = _parse_args(m.group(2))

        # ── node ─────────────────────────────────────────────────────────────
        if macro in _NODE_MACROS and len(args) >= 2:
            alias = args[0]
            label = args[1]
            if not alias or alias in seen:
                continue
            seen.add(alias)

            if macro in _NO_TECH_NODES:
                technology  = None
                description = args[2] if len(args) > 2 else None
            else:
                technology  = args[2] if len(args) > 2 else None
                description = args[3] if len(args) > 3 else None

            node: dict[str, Any] = {
                "id":    alias,
                "type":  _NODE_TYPE[macro],
                "label": label[:28],
            }
            if technology:
                node["technology"]  = technology[:22]
            if description:
                node["description"] = description[:44]
            nodes.append(node)

        # ── edge ─────────────────────────────────────────────────────────────
        elif macro in _REL_MACROS and len(args) >= 2:
            source = args[0]
            target = args[1]
            label  = args[2] if len(args) > 2 else None
            if not source or not target:
                continue
            edge_idx += 1
            edge: dict[str, Any] = {
                "id":     f"e{edge_idx}",
                "source": source,
                "target": target,
            }
            if label:
                edge["label"] = label[:18]
            if macro in _BIREL_MACROS:
                edge["bidirectional"] = True
            edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def parse_three_sections(llm_output: str) -> dict[str, dict[str, Any]]:
    """Extract sections B / C / F from LLM output delimited by markers.

    Expected format::

        === SECTION B ===
        @startuml
        ...
        @enduml

        === SECTION C ===
        ...

        === SECTION F ===
        ...

    Returns ``{"B": {...}, "C": {...}, "F": {...}}``.
    Missing sections get ``None``.
    """
    result: dict[str, dict[str, Any] | None] = {"B": None, "C": None, "F": None}
    # Split on the section markers (case-insensitive)
    splitter = re.compile(r"===\s*SECTION\s+([BCF])\s*===", re.IGNORECASE)
    parts = splitter.split(llm_output)
    # parts = [preamble, key, block, key, block, ...]
    i = 1
    while i + 1 < len(parts):
        key   = parts[i].strip().upper()
        block = parts[i + 1]
        if key in result:
            parsed = parse_plantuml_c4(block)
            if parsed["nodes"]:          # only keep non-empty diagrams
                result[key] = parsed
        i += 2
    return result  # type: ignore[return-value]
