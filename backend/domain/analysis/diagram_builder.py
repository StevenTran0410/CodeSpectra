"""
C4 model diagram builders for analysis sections B, C, F.

Pure functions: accept validated section data and a frozenset of known manifest paths.
No DB access — the caller fetches valid_paths from the manifest.

Diagram levels per section:
  Section B → C4Container  (C2: runtime processes, external systems, DB)
  Section C → C4Component  (C3: folder/module structure grouped into boundaries)
  Section F → C4Component  (C3: feature boundaries with entrypoint + key files)

Node labels are sanitized via _safe_label().  Node aliases are sanitized via _safe_alias().
File paths are validated via _is_valid_path() — only manifest-confirmed paths become nodes.
"""
from __future__ import annotations

import posixpath
import re
from typing import Any


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_path_hints(layer_name: str) -> list[str]:
    """Extract path prefixes from parentheses in a layer name.

    e.g. "Backend domain layer (backend/domain)" -> ["backend/domain"]
    """
    return re.findall(r'\(([^)]+/[^)]*)\)', layer_name)


def _strip_path_hint(layer_name: str) -> str:
    """Remove the parenthetical path suffix from a layer name."""
    return re.sub(r'\s*\([^)]+/[^)]*\)\s*', '', layer_name).strip()


def _safe_alias(text: str, prefix: str = "N") -> str:
    """Convert arbitrary text to a valid Mermaid C4 node alias.

    Aliases must match [a-zA-Z][a-zA-Z0-9_]* — no slashes, dots, or spaces.
    """
    if not text or not text.strip():
        return f"{prefix}0"
    alias = re.sub(r'[^a-zA-Z0-9]', '_', text.strip())
    alias = re.sub(r'_+', '_', alias).strip('_')
    if not alias:
        return f"{prefix}0"
    if alias[0].isdigit():
        alias = f"{prefix}_{alias}"
    return alias[:28]


def _safe_label(text: str, max_len: int = 40) -> str:
    """Sanitize a string for use as a Mermaid C4 node label.

    - Empty / whitespace input returns 'unknown'
    - Unicode punctuation is normalised to ASCII equivalents
    - Characters that break Mermaid parsing are stripped / replaced
    - Truncated to max_len with '...' suffix when over limit
    """
    if not text or not text.strip():
        return "unknown"
    sanitised = (
        text
        .replace('\u2013', '-')
        .replace('\u2014', '-')
        .replace('\u2019', "'")
        .replace('\u201c', "'")
        .replace('\u201d', "'")
        .replace('\u2192', '->')
        .replace('\u2190', '<-')
        .replace('\ufffd', '')
    )
    sanitised = sanitised.encode('ascii', errors='ignore').decode('ascii')
    sanitised = (
        sanitised
        .replace('"', "'")
        .replace('[', '(')
        .replace(']', ')')
        .replace('{', '(')
        .replace('}', ')')
        .replace('\n', ' ')
        .replace('\r', '')
    )
    sanitised = sanitised.strip()
    if not sanitised:
        return "unknown"
    if len(sanitised) > max_len:
        sanitised = sanitised[: max_len - 3] + "..."
    return sanitised


def _is_valid_path(path: str, valid_paths: frozenset[str]) -> bool:
    """Return True only if path exists in the snapshot manifest."""
    if not path:
        return False
    normalised = posixpath.normpath(path.replace("\\", "/"))
    return path in valid_paths or normalised in valid_paths


def _file_tech(filename: str) -> str:
    """Infer a short technology label from a file name."""
    name = filename.lower()
    if name.endswith(('.tsx', '.jsx')):
        return "React / TS"
    if name.endswith('.ts'):
        return "TypeScript"
    if name.endswith('.js'):
        return "JavaScript"
    if name.endswith('.py'):
        return "Python"
    if name.endswith(('.sql', '.db', '.sqlite')):
        return "SQL"
    if name.endswith(('.yaml', '.yml', '.toml', '.json', '.env')):
        return "Config"
    if name.endswith(('.md', '.rst', '.txt')):
        return "Docs"
    return "File"


def _infer_container_tech(layer_name: str, path_hint: str, frameworks: list[str]) -> str:
    """Heuristically map a layer to its technology stack.

    Uses path hint first (most reliable), then framework list, then name keywords.
    """
    name_lc = (layer_name or "").lower()
    path_lc = (path_hint or "").lower()
    fw_map: dict[str, str] = {f.lower(): f for f in frameworks}

    # Renderer / UI
    if any(k in path_lc for k in ("renderer", "frontend", "client")) or \
       any(k in name_lc for k in ("renderer", "frontend", "ui layer")):
        react = fw_map.get("react", "")
        ts = fw_map.get("typescript", "")
        if react and ts:
            return f"{react} + {ts}"
        return react or ts or "TypeScript / React"

    # Electron preload / IPC bridge
    if "preload" in path_lc or ("ipc" in name_lc and "bridge" in name_lc):
        return "contextBridge / Node.js"

    # Electron main
    if any(k in path_lc for k in ("src/main", "electron/main")) or \
       ("electron" in name_lc and "main" in name_lc):
        electron = fw_map.get("electron", "Electron")
        return f"{electron} + Node.js"

    # Backend / API
    if any(k in path_lc for k in ("backend", "api", "server", "service")) or \
       any(k in name_lc for k in ("backend", "api layer", "service layer")):
        for fw_key, fw_lang in (
            ("fastapi", "Python"),
            ("django", "Python"),
            ("flask", "Python"),
            ("express", "Node.js"),
            ("nestjs", "Node.js"),
            ("rails", "Ruby"),
        ):
            if fw_key in fw_map:
                return f"{fw_map[fw_key]} / {fw_lang}"
        return "Python" if "python" in name_lc + path_lc else "API Server"

    # Database / storage
    if any(k in path_lc + name_lc for k in ("database", " db", "sqlite", "postgres", "mysql", "mongo")):
        for fw_key in ("sqlite", "postgresql", "mysql", "mongodb", "redis"):
            if fw_key in fw_map:
                return fw_map[fw_key]
        return "Database"

    # Worker / background
    if any(k in name_lc for k in ("worker", "queue", "scheduler", "cron")):
        return "Worker / asyncio"

    # Backend sub-layers (domain, infra, shared, utilities) — likely Python
    if any(k in name_lc for k in ("domain", "infrastructure", "infra", "shared", "util", "common")):
        return "Python"

    # Generic fallback: pick a meaningful framework (skip UI-only ones for backend layers)
    _ui_only = {"react", "vue", "angular", "svelte", "vite", "tailwind"}
    non_ui = [f for f in frameworks if f.lower() not in _ui_only]
    return non_ui[0] if non_ui else (frameworks[0] if frameworks else "unknown")


def _infer_container_type(layer_name: str, path_hint: str) -> str:
    """Return 'db', 'queue', or 'standard' to select the right C4 shape."""
    combined = (layer_name + " " + path_hint).lower()
    if any(k in combined for k in ("database", "sqlite", "postgres", "mysql", "mongo", " db", "redis", "storage")):
        return "db"
    if any(k in combined for k in ("queue", "broker", "kafka", "rabbitmq", "sqs")):
        return "queue"
    return "standard"


def _c2_relation_lines(
    containers: list[tuple[str, str, str]],
    ext_aliases: list[str],
) -> list[str]:
    """Infer Rel() lines for a C2 Container diagram.

    containers: [(alias, path_hint, name_lower), ...]
    ext_aliases: aliases for System_Ext nodes.
    """
    renderer = main_proc = backend = None
    db_nodes: list[str] = []

    for alias, path, name in containers:
        combined = path + " " + name
        if any(k in combined for k in ("renderer", "frontend", "client", "ui")):
            if renderer is None:
                renderer = alias
        elif "preload" in combined or ("ipc" in name and "bridge" in name):
            pass  # bridge — not a primary relation source/target
        elif any(k in combined for k in ("main", "electron")):
            if main_proc is None:
                main_proc = alias
        elif any(k in combined for k in ("backend", "api", "server", "service")):
            if backend is None:
                backend = alias
        elif any(k in combined for k in ("database", "db", "sqlite", "storage")):
            db_nodes.append(alias)

    lines: list[str] = []
    ui = renderer or main_proc or (containers[0][0] if containers else None)
    if ui:
        lines.append(f'    Rel(user, {ui}, "Uses")')
    if renderer and main_proc:
        lines.append(f'    Rel({renderer}, {main_proc}, "IPC")')
    if main_proc and backend:
        lines.append(f'    Rel({main_proc}, {backend}, "HTTP REST")')
    elif renderer and backend and not main_proc:
        lines.append(f'    Rel({renderer}, {backend}, "HTTP")')
    src_db = backend or main_proc
    for db in db_nodes:
        if src_db:
            lines.append(f'    Rel({src_db}, {db}, "Read / Write")')
    src_ext = backend or main_proc
    for ext in ext_aliases:
        if src_ext:
            lines.append(f'    Rel({src_ext}, {ext}, "API calls")')
    return lines


# ── Section B → C4Container ───────────────────────────────────────────────────

def build_architecture_diagram(section_b: dict[str, Any], valid_paths: frozenset[str]) -> str:
    """Generate a C4Container Mermaid diagram for Section B (Architecture Overview).

    Maps main_layers → Container/ContainerDb nodes, external_integrations →
    System_Ext, database_hints → ContainerDb, then infers Rel() connections
    from layer path patterns (renderer → main → backend → db → external).

    Args:
        section_b: Validated Section B dict.
        valid_paths: Frozenset of rel_paths in the snapshot manifest (unused here
                     but kept for API symmetry — paths are layer-level, not file-level).

    Returns:
        A C4Container Mermaid string, or empty string if no layers exist.
    """
    layers: list[str] = section_b.get("main_layers") or []
    frameworks: list[str] = section_b.get("frameworks") or []
    external: list[str] = section_b.get("external_integrations") or []
    db_hints: list[str] = section_b.get("database_hints") or []

    if not layers:
        return ""

    # C4 diagrams use hardcoded colours that clash with Mermaid dark theme.
    # Injecting the init directive forces light theme regardless of renderer settings.
    lines: list[str] = [
        "%%{init: {'theme': 'default'}}%%",
        "C4Container",
        '    title Container Diagram',
        "",
    ]

    # Developer persona — always present
    lines.append('    Person(user, "Developer", "Analyzes code repositories")')
    lines.append("")

    # Layers → Container / ContainerDb nodes
    containers: list[tuple[str, str, str]] = []  # (alias, path_hint, name_lower)
    seen_aliases: set[str] = set()
    layer_aliases: list[str] = []

    for i, layer in enumerate(layers[:7]):
        hints = _extract_path_hints(layer)
        path_hint = hints[0] if hints else ""
        base_name = _strip_path_hint(layer)
        tech = _infer_container_tech(base_name, path_hint, frameworks)
        ctype = _infer_container_type(base_name, path_hint)

        label = _safe_label(base_name, max_len=32)
        desc = _safe_label(path_hint or base_name, max_len=36)
        raw_alias = f"C{i}_{_safe_alias(path_hint or base_name, 'L')}"
        alias = raw_alias if raw_alias not in seen_aliases else f"C{i}"
        seen_aliases.add(alias)

        if ctype == "db":
            lines.append(f'    ContainerDb({alias}, "{label}", "{tech}", "{desc}")')
        elif ctype == "queue":
            lines.append(f'    ContainerQueue({alias}, "{label}", "{tech}", "{desc}")')
        else:
            lines.append(f'    Container({alias}, "{label}", "{tech}", "{desc}")')

        containers.append((alias, path_hint.lower(), base_name.lower()))
        layer_aliases.append(alias)

    # DB hints not already represented in layers
    db_layer_keywords = {"db", "database", "sqlite", "postgres", "mysql", "mongo", "redis"}
    layer_text_lc = " ".join(layers).lower()
    for j, hint in enumerate(db_hints[:2]):
        if any(k in hint.lower() for k in db_layer_keywords) and any(k in layer_text_lc for k in db_layer_keywords):
            continue  # already covered by a layer
        alias = f"DB{j}"
        tech = _infer_container_tech(hint, "database", frameworks)
        label = _safe_label(hint, max_len=28)
        lines.append(f'    ContainerDb({alias}, "{label}", "{tech}", "Data storage")')
        containers.append((alias, "db", hint.lower()))

    lines.append("")

    # External integrations → System_Ext
    # Skip items that are internal frameworks, IPC bridges, or already represented as containers
    _internal_substrings = (
        "fastapi", "react", "electron", "sqlite", "django", "flask", "express",
        "ipc", "ipc bridge", "subprocess", "http server", "fastapi http",
        "python subprocess", "node.js", "vite", "zustand", "tailwind",
    )
    ext_aliases: list[str] = []
    ext_idx = 0
    for ext in external[:5]:
        ext_lc = ext.lower()
        if any(kw in ext_lc for kw in _internal_substrings):
            continue
        alias = f"EXT{ext_idx}"
        ext_idx += 1
        label = _safe_label(ext, max_len=28)
        lines.append(f'    System_Ext({alias}, "{label}", "External service")')
        ext_aliases.append(alias)

    lines.append("")

    # Relationships
    rel_lines = _c2_relation_lines(containers, ext_aliases)
    lines.extend(rel_lines)

    lines.append("")
    n = min(max(len(layer_aliases) + len(ext_aliases), 3), 5)
    lines.append(f'    UpdateLayoutConfig($c4ShapeInRow = "{n}", $c4BoundaryInRow = "1")')

    return "\n".join(lines)


# ── Section C → C4Component ───────────────────────────────────────────────────

def build_structure_diagram(section_c: dict[str, Any], valid_paths: frozenset[str]) -> str:
    """Generate a C4Component Mermaid diagram for Section C (Repo Structure).

    Folders are grouped into Container_Boundary blocks by their first-level
    directory prefix. Only folders containing at least one path in valid_paths
    are rendered — this prevents hallucinated folder names from appearing.

    Args:
        section_c: Validated Section C dict.
        valid_paths: Frozenset of rel_paths in the snapshot manifest.

    Returns:
        A C4Component Mermaid string, or empty string if no valid folders exist.
    """
    folders: list[dict[str, Any]] = section_c.get("folders") or []
    if not folders:
        return ""

    # Validate + sort folders by path depth
    valid_folders: list[dict[str, Any]] = []
    for folder in folders:
        if not isinstance(folder, dict):
            continue
        path = str(folder.get("path") or "").strip("/")
        if not path:
            continue
        prefix = path.rstrip("/") + "/"
        if any(p.startswith(prefix) or p == path for p in valid_paths):
            valid_folders.append({**folder, "_path": path})

    if not valid_folders:
        return ""

    valid_folders.sort(key=lambda f: (len(f["_path"].split("/")), f["_path"]))

    # Group by top-level directory (first path segment)
    boundaries: dict[str, list[dict[str, Any]]] = {}
    for folder in valid_folders:
        path = folder["_path"]
        top = path.split("/")[0]
        boundaries.setdefault(top, []).append(folder)

    lines: list[str] = [
        "%%{init: {'theme': 'default'}}%%",
        "C4Component",
        '    title Component Diagram',
        "",
    ]

    alias_counter = 0
    boundary_aliases: dict[str, str] = {}
    component_aliases: dict[str, str] = {}

    for top_dir, group in boundaries.items():
        b_alias = f"BND{alias_counter}"
        alias_counter += 1
        boundary_aliases[top_dir] = b_alias

        # Use the top-level folder's own description for the boundary label if available
        top_folder = next((f for f in group if f["_path"] == top_dir), None)
        b_desc = str(top_folder.get("description") or "") if top_folder else ""
        b_label_text = f"{top_dir} — {b_desc}" if b_desc else top_dir
        b_label = _safe_label(b_label_text, max_len=48)
        lines.append(f'    Container_Boundary({b_alias}, "{b_label}") {{')

        for folder in group:
            path = folder["_path"]
            # Skip the folder that IS the boundary itself — it's already the boundary label
            if path == top_dir:
                # Still register it so parent-child Rel() can reference it
                component_aliases[path] = b_alias
                continue
            role = str(folder.get("role") or "unknown")
            desc = str(folder.get("description") or "")
            # Show just the sub-path relative to top_dir for readability
            rel_path = path[len(top_dir):].lstrip("/") or path
            label = _safe_label(rel_path, max_len=34)
            role_label = _safe_label(role, max_len=16)
            desc_label = _safe_label(desc, max_len=40) if desc else role_label
            c_alias = f"CMP{alias_counter}"
            alias_counter += 1
            component_aliases[path] = c_alias
            lines.append(f'        Component({c_alias}, "{label}", "{role_label}", "{desc_label}")')

        lines.append("    }")
        lines.append("")

    # Relationships: connect components to their nearest registered parent
    rels_added: set[tuple[str, str]] = set()
    for folder in valid_folders:
        path = folder["_path"]
        parts = path.split("/")
        child_alias = component_aliases.get(path)
        if not child_alias:
            continue
        # Walk up to find nearest parent component or boundary
        for depth in range(len(parts) - 1, 0, -1):
            parent_path = "/".join(parts[:depth])
            parent_alias = component_aliases.get(parent_path)
            if parent_alias:
                key = (parent_alias, child_alias)
                if key not in rels_added:
                    rels_added.add(key)
                    lines.append(f'    Rel({parent_alias}, {child_alias}, "contains")')
                break

    if rels_added:
        lines.append("")

    n_boundaries = len(boundaries)
    shapes_per_row = min(max(max(len(g) for g in boundaries.values()), 2), 4)
    bounds_per_row = min(n_boundaries, 3)
    lines.append(
        f'    UpdateLayoutConfig($c4ShapeInRow = "{shapes_per_row}", $c4BoundaryInRow = "{bounds_per_row}")'
    )

    return "\n".join(lines)


# ── Section F → C4Component (feature boundaries) ─────────────────────────────

def build_feature_diagram(section_f: dict[str, Any], valid_paths: frozenset[str]) -> str:
    """Generate a C4Component Mermaid diagram for Section F (Feature Map).

    Each feature is rendered as a Container_Boundary containing its entrypoint
    and up to 3 key files (validated against valid_paths). Features with no
    valid files are skipped. Capped at 6 features for readability.

    Args:
        section_f: Validated Section F dict.
        valid_paths: Frozenset of rel_paths in the snapshot manifest.

    Returns:
        A C4Component Mermaid string, or empty string if no renderable features.
    """
    features: list[dict[str, Any]] = section_f.get("features") or []
    if not features:
        return ""

    lines: list[str] = [
        "%%{init: {'theme': 'default'}}%%",
        "C4Component",
        '    title Feature Map',
        "",
    ]

    alias_counter = 0
    rel_lines: list[str] = []
    rendered = 0

    for feat in features:
        if not isinstance(feat, dict):
            continue
        if rendered >= 6:
            break

        name = str(feat.get("name") or "").strip()
        if not name:
            continue

        entrypoint = str(feat.get("entrypoint") or "").strip()
        key_files: list[str] = feat.get("key_files") or []

        # Validate entrypoint
        ep_valid = _is_valid_path(entrypoint, valid_paths) if entrypoint else False
        valid_kfs = [kf for kf in key_files[:4] if _is_valid_path(str(kf).strip(), valid_paths)]

        # Skip feature if no valid files at all
        if not ep_valid and not valid_kfs:
            continue

        b_alias = f"FEAT{alias_counter}"
        alias_counter += 1
        feat_label = _safe_label(name, max_len=36)
        lines.append(f'    Container_Boundary({b_alias}, "{feat_label}") {{')

        ep_alias: str | None = None
        if ep_valid:
            ep_alias = f"EP{alias_counter}"
            alias_counter += 1
            ep_filename = entrypoint.split("/")[-1]
            ep_label = _safe_label(ep_filename, max_len=30)
            ep_tech = _file_tech(ep_filename)
            lines.append(f'        Component({ep_alias}, "{ep_label}", "{ep_tech}", "entrypoint")')

        kf_aliases: list[str] = []
        for kf in valid_kfs[:3]:
            kf_str = str(kf).strip()
            # Skip if same as entrypoint
            if kf_str == entrypoint:
                continue
            kf_alias = f"KF{alias_counter}"
            alias_counter += 1
            kf_filename = kf_str.split("/")[-1]
            kf_label = _safe_label(kf_filename, max_len=30)
            kf_tech = _file_tech(kf_filename)
            lines.append(f'        Component({kf_alias}, "{kf_label}", "{kf_tech}", "key file")')
            kf_aliases.append(kf_alias)

        lines.append("    }")
        lines.append("")

        # Relationships inside the feature boundary
        if ep_alias:
            for kf_alias in kf_aliases:
                rel_lines.append(f'    Rel({ep_alias}, {kf_alias}, "uses")')
        elif kf_aliases:
            for i in range(len(kf_aliases) - 1):
                rel_lines.append(f'    Rel({kf_aliases[i]}, {kf_aliases[i+1]}, "depends")')

        rendered += 1

    if rendered == 0:
        return ""

    lines.extend(rel_lines)
    if rel_lines:
        lines.append("")

    shapes_per_row = 3
    bounds_per_row = min(rendered, 3)
    lines.append(
        f'    UpdateLayoutConfig($c4ShapeInRow = "{shapes_per_row}", $c4BoundaryInRow = "{bounds_per_row}")'
    )

    return "\n".join(lines)


# ── Section F → per-feature sequence diagrams ────────────────────────────────

def build_feature_sequence_diagrams(section_f: dict[str, Any]) -> list[dict[str, Any] | None]:
    """Build a React Flow + ELK sequence diagram for each feature in Section F.

    Uses ``reading_order`` as the ordered steps. Skips features with fewer
    than 2 steps. Returns a list (one entry per feature) of ``{nodes, edges}``
    dicts in C4DiagramData format, or ``None`` for features without enough data.
    """
    features: list[dict[str, Any]] = section_f.get("features") or []
    result: list[dict[str, Any] | None] = []

    for feat in features:
        if not isinstance(feat, dict):
            result.append(None)
            continue

        steps: list[str] = [str(s).strip() for s in (feat.get("reading_order") or []) if s]
        entrypoint = str(feat.get("entrypoint") or "").strip()

        if len(steps) < 2:
            result.append(None)
            continue

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        for i, path in enumerate(steps[:8]):  # cap at 8 for readability
            filename = path.split("/")[-1] or path
            node: dict[str, Any] = {
                "id":    f"s{i}",
                "type":  "step",
                "label": filename[:26],
            }
            if path == entrypoint or i == 0:
                node["description"] = "entrypoint"
            tech = _file_tech(filename)
            if tech != "File":
                node["technology"] = tech
            nodes.append(node)

            if i > 0:
                edges.append({"id": f"e{i}", "source": f"s{i - 1}", "target": f"s{i}"})

        result.append({"nodes": nodes, "edges": edges} if len(nodes) >= 2 else None)

    return result
