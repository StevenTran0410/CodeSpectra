"""Tests for the structural graph pipeline (CS-102, CS-107).

Covers three sub-systems that all live in domain.structural_graph:
  - Centrality scoring & BFS neighbor expansion (Python fallbacks)
  - SCC cycle detection & Louvain community detection (Python + C++ native)
  - Import resolution: suffix-index, '..' path normalisation, __init__.py fallback
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from domain.structural_graph._louvain_fallback import compute_louvain_python
from domain.structural_graph._scc_fallback import compute_scc_python
from domain.structural_graph.service import _compute_scores_python, _expand_neighbors_python


# ── Pure-Python helpers mirroring service logic (no async / no DB) ────────────
# Local copies avoid importing the async service module in test context.
# If service.py changes these functions, update here too.

def _build_py_suffix_index(file_set: set[str]) -> dict[str, str]:
    """Suffix-lookup table for Python files.

    Enables resolving absolute imports like "domain.x" → "backend/domain/x.py"
    regardless of the source-root prefix used in the project layout.
    First-seen suffix wins (insertion order = iteration order of the input).
    """
    index: dict[str, str] = {}
    for f in file_set:
        if f.endswith(".py"):
            parts = f.split("/")
            for i in range(len(parts)):
                suffix = "/".join(parts[i:])
                if suffix not in index:
                    index[suffix] = f
    return index


def _resolve_py_import(imp: str, index: dict[str, str]) -> str | None:
    """Convert a Python import string to a file path via the suffix index.

    Tries module.py first, then package/__init__.py as fallback
    (mirrors Bug-2 fix in service.build()).
    """
    py_guess = imp.replace(".", "/") + ".py"
    result = index.get(py_guess)
    if not result:
        pkg_init = imp.replace(".", "/") + "/__init__.py"
        result = index.get(pkg_init)
    return result


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _resolve_relative_import(src_rel_path: str, target: str, candidates: set[str]) -> str | None:
    """Resolve a relative TS/JS import string to a file path.

    Mirrors service._resolve_relative_import including the Bug-1 fix:
    uses os.path.normpath to collapse '..' segments so '../../store/foo'
    resolves correctly instead of staying as a literal unresolved string.
    """
    base = Path(src_rel_path).parent
    # os.path.normpath collapses '..' — Path.as_posix() does NOT.
    candidate = _normalize(os.path.normpath(str(base / target)))

    options = [
        candidate,
        f"{candidate}.py",
        f"{candidate}.ts",
        f"{candidate}.tsx",
        f"{candidate}.js",
        f"{candidate}.jsx",
        _normalize(str(Path(candidate) / "index.ts")),
        _normalize(str(Path(candidate) / "index.tsx")),
        _normalize(str(Path(candidate) / "index.js")),
    ]
    for opt in options:
        if opt in candidates:
            return opt
    return None


def _resolve_community_edges(
    edge_rows: list[dict],
    node_id_set: set[str],
    py_suffix_index: dict[str, str],
) -> list[tuple[str, str, float]]:
    """Replicate detect_communities() edge-dedup + re-resolution logic.

    Includes all three re-resolution branches from service.detect_communities():
    - already-resolved internal edges
    - unresolved Python absolute imports (suffix index + __init__.py fallback)
    - unresolved relative TS/JS imports (normpath-based resolution)
    """
    edge_set: set[tuple[str, str]] = set()
    edge_tuples: list[tuple[str, str, float]] = []

    for r in edge_rows:
        src, dst, is_external = r["src_path"], r["dst_path"], r["is_external"]
        if src not in node_id_set or src == dst:
            continue
        if not is_external:
            if dst in node_id_set and (src, dst) not in edge_set:
                edge_tuples.append((src, dst, 1.0))
                edge_set.add((src, dst))
        elif src.endswith(".py") and "/" not in dst and not dst.startswith("."):
            # Python absolute import re-resolution (also try __init__.py — Bug 2)
            py_guess = dst.replace(".", "/") + ".py"
            resolved = py_suffix_index.get(py_guess)
            if not resolved:
                pkg_init = dst.replace(".", "/") + "/__init__.py"
                resolved = py_suffix_index.get(pkg_init)
            if resolved and resolved != src and (src, resolved) not in edge_set:
                edge_tuples.append((src, resolved, 1.0))
                edge_set.add((src, resolved))
        elif dst.startswith("."):
            # Relative TS/JS import re-resolution (normpath fix — Bug 1)
            resolved = _resolve_relative_import(src, dst, node_id_set)
            if resolved and resolved != src and (src, resolved) not in edge_set:
                edge_tuples.append((src, resolved, 1.0))
                edge_set.add((src, resolved))

    return edge_tuples


def _make_edge(src: str, dst: str, *, is_external: bool) -> dict:
    return {"src_path": src, "dst_path": dst, "is_external": is_external}


# ── Realistic CodeSpectra file set (ground truth for import-resolution tests) ─

CODESPECTRA_FILES: set[str] = {
    # backend API layer
    "backend/api/structural_graph.py",
    "backend/api/local_repo.py",
    "backend/api/analysis.py",
    "backend/api/workspace.py",
    # domain — structural_graph
    "backend/domain/structural_graph/__init__.py",
    "backend/domain/structural_graph/service.py",
    "backend/domain/structural_graph/types.py",
    "backend/domain/structural_graph/_louvain_fallback.py",
    "backend/domain/structural_graph/_scc_fallback.py",
    # domain — analysis (package __init__ included for Bug-2 tests)
    "backend/domain/analysis/__init__.py",
    "backend/domain/analysis/agents/__init__.py",
    "backend/domain/analysis/static_risk.py",
    "backend/domain/analysis/orchestrator.py",
    "backend/domain/analysis/agent_pipeline.py",
    # domain — repo_map
    "backend/domain/repo_map/service.py",
    # infrastructure
    "backend/infrastructure/db/database.py",
    # shared
    "backend/shared/logger.py",
    "backend/shared/utils.py",
    "backend/shared/errors.py",
    # entrypoints
    "backend/main.py",
    # frontend renderer (for Bug-1 TS relative import tests)
    "src/renderer/src/screens/analysis/index.tsx",
    "src/renderer/src/screens/index-overview/index.tsx",
    "src/renderer/src/store/local-repo.store.ts",
    "src/renderer/src/store/job.store.ts",
    "src/renderer/src/hooks/useTheme.ts",
    "src/renderer/src/components/layout/AppShell.tsx",
    "src/renderer/src/App.tsx",
    "src/renderer/src/main.tsx",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Centrality scoring & BFS neighbor expansion
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_scores_python_ranking_and_tiebreak() -> None:
    nodes = ["a.py", "b.py", "c.py", "d.py", "e.py"]
    edges = [
        ("a.py", "b.py", "import", 0),
        ("a.py", "c.py", "import", 0),
        ("d.py", "b.py", "import", 0),
        ("e.py", "b.py", "import", 1),  # external import still counts for centrality
    ]

    scored = _compute_scores_python(nodes, edges)
    by_path = {str(x["rel_path"]): x for x in scored}

    assert int(by_path["b.py"]["indegree"]) == 3
    assert int(by_path["b.py"]["outdegree"]) == 0
    assert int(by_path["b.py"]["score"]) == 9

    assert int(by_path["a.py"]["indegree"]) == 0
    assert int(by_path["a.py"]["outdegree"]) == 2
    assert int(by_path["a.py"]["score"]) == 2

    # d.py and e.py both score 1; alphabetical rel_path tie-break applies.
    rank = [str(x["rel_path"]) for x in scored]
    assert rank.index("d.py") < rank.index("e.py")


def test_expand_neighbors_python_internal_bounded_bfs() -> None:
    edges = [
        ("src/a.py", "src/b.py", "import", 0),
        ("src/a.py", "src/c.py", "import", 0),
        ("src/b.py", "src/d.py", "import", 0),
        ("src/c.py", "src/e.py", "import", 0),
        ("src/e.py", "src/f.py", "import", 0),
        ("src/a.py", "requests", "import", 1),  # external edge should be ignored
    ]

    out = _expand_neighbors_python(seed="src/a.py", edge_inputs=edges, hops=2, limit=300)
    nodes = set(out["nodes"])
    out_edges = set(tuple(e) for e in out["edges"])

    assert "src/a.py" in nodes
    assert "src/b.py" in nodes
    assert "src/c.py" in nodes
    assert "src/d.py" in nodes
    assert "src/e.py" in nodes
    assert "src/f.py" not in nodes  # 3 hops away
    assert "requests" not in nodes  # external-only node excluded

    assert ("src/a.py", "src/b.py", "import") in out_edges
    assert ("src/a.py", "src/c.py", "import") in out_edges
    assert ("src/b.py", "src/d.py", "import") in out_edges
    assert ("src/c.py", "src/e.py", "import") in out_edges
    assert ("src/e.py", "src/f.py", "import") not in out_edges


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — SCC cycle detection (Python fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def test_scc_fallback_simple_cycle() -> None:
    """a→b→a forms a 2-cycle."""
    result = compute_scc_python([("a", "b"), ("b", "a")])
    assert len(result) == 1
    assert sorted(result[0]) == ["a", "b"]


def test_scc_fallback_two_independent_cycles() -> None:
    """Two separate 2-cycles → 2 SCCs."""
    edges = [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")]
    result = compute_scc_python(edges)
    assert len(result) == 2
    paths = [sorted(s) for s in result]
    assert ["a", "b"] in paths
    assert ["c", "d"] in paths


def test_scc_fallback_no_cycle() -> None:
    """Acyclic graph → no SCCs returned."""
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    result = compute_scc_python(edges)
    assert result == []


def test_scc_fallback_larger_scc() -> None:
    """3-node cycle a→b→c→a."""
    edges = [("a", "b"), ("b", "c"), ("c", "a")]
    result = compute_scc_python(edges)
    assert len(result) == 1
    assert sorted(result[0]) == ["a", "b", "c"]


def test_scc_fallback_bridge_between_two_sccs() -> None:
    """Two cliques joined by a one-way bridge; bridge must not merge SCCs."""
    edges = [
        ("a", "b"), ("b", "a"),           # SCC 1
        ("c", "d"), ("d", "c"),           # SCC 2
        ("b", "c"),                        # one-way bridge — does not merge
    ]
    result = compute_scc_python(edges)
    assert len(result) == 2


def test_scc_fallback_empty() -> None:
    result = compute_scc_python([])
    assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Louvain community detection (Python fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def test_louvain_two_cliques_two_communities() -> None:
    """Two cliques with a weak bridge → two distinct communities."""
    adj = [
        ("a", "b", 1.0), ("b", "c", 1.0), ("c", "a", 1.0),   # clique 1
        ("d", "e", 1.0), ("e", "f", 1.0), ("f", "d", 1.0),   # clique 2
        ("c", "d", 0.05),                                        # weak bridge
    ]
    nodes = ["a", "b", "c", "d", "e", "f"]
    result = compute_louvain_python(adj, nodes)

    assert result["a"] == result["b"] == result["c"], "clique 1 must share a community"
    assert result["d"] == result["e"] == result["f"], "clique 2 must share a community"
    assert result["a"] != result["d"], "the two cliques must be in different communities"


def test_louvain_star_single_community() -> None:
    """Hub + spokes with no inter-spoke edges — all may end up in one community."""
    adj = [(f"leaf{i}", "hub", 1.0) for i in range(5)]
    adj += [("hub", f"leaf{i}", 1.0) for i in range(5)]
    nodes = ["hub"] + [f"leaf{i}" for i in range(5)]
    result = compute_louvain_python(adj, nodes)
    assert len(set(result.values())) <= 2


def test_louvain_disconnected_components() -> None:
    """Fully disconnected components must end up in separate communities."""
    adj = [("a", "b", 1.0), ("c", "d", 1.0)]
    nodes = ["a", "b", "c", "d"]
    result = compute_louvain_python(adj, nodes)
    assert result["a"] == result["b"]
    assert result["c"] == result["d"]
    assert result["a"] != result["c"]


def test_louvain_isolated_nodes_get_communities() -> None:
    """Nodes with no edges still appear in result."""
    adj: list[tuple[str, str, float]] = []
    nodes = ["x", "y", "z"]
    result = compute_louvain_python(adj, nodes)
    assert set(result.keys()) == {"x", "y", "z"}


def test_louvain_returns_int_community_ids() -> None:
    adj = [("a", "b", 1.0), ("b", "c", 1.0), ("c", "a", 1.0)]
    nodes = ["a", "b", "c"]
    result = compute_louvain_python(adj, nodes)
    for v in result.values():
        assert isinstance(v, int)


def test_louvain_community_ids_start_at_zero() -> None:
    adj = [("a", "b", 1.0), ("b", "a", 1.0)]
    nodes = ["a", "b"]
    result = compute_louvain_python(adj, nodes)
    assert min(result.values()) == 0


# ── C++ native Louvain (skipped when native module is not built) ──────────────

def _native_available() -> bool:
    try:
        importlib.import_module("domain.structural_graph._native_graph")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _native_available(), reason="C++ native not built")
def test_native_louvain_two_cliques() -> None:
    """Native compute_louvain must separate two dense cliques."""
    native = importlib.import_module("domain.structural_graph._native_graph")
    adj = [
        ("a", "b", 1.0), ("b", "c", 1.0), ("c", "a", 1.0),
        ("d", "e", 1.0), ("e", "f", 1.0), ("f", "d", 1.0),
        ("c", "d", 0.05),
    ]
    nodes = ["a", "b", "c", "d", "e", "f"]
    result: dict[str, int] = native.compute_louvain(adj, nodes, 1.0, 42)

    assert result["a"] == result["b"] == result["c"]
    assert result["d"] == result["e"] == result["f"]
    assert result["a"] != result["d"]


@pytest.mark.skipif(not _native_available(), reason="C++ native not built")
def test_native_louvain_matches_python_on_simple_graph() -> None:
    """Native and Python fallback must produce structurally equivalent partitions."""
    native = importlib.import_module("domain.structural_graph._native_graph")
    adj = [
        ("a", "b", 1.0), ("b", "c", 1.0), ("c", "a", 1.0),
        ("d", "e", 1.0), ("e", "f", 1.0), ("f", "d", 1.0),
        ("c", "d", 0.05),
    ]
    nodes = ["a", "b", "c", "d", "e", "f"]

    native_result = native.compute_louvain(adj, nodes, 1.0, 42)
    python_result = compute_louvain_python(adj, nodes, 1.0)

    assert native_result["a"] == native_result["b"] == native_result["c"]
    assert python_result["a"] == python_result["b"] == python_result["c"]
    assert native_result["a"] != native_result["d"]
    assert python_result["a"] != python_result["d"]


@pytest.mark.skipif(not _native_available(), reason="C++ native not built")
def test_native_louvain_isolated_nodes() -> None:
    """Native must include isolated nodes in output."""
    native = importlib.import_module("domain.structural_graph._native_graph")
    result: dict[str, int] = native.compute_louvain([], ["x", "y", "z"], 1.0, 42)
    assert set(result.keys()) == {"x", "y", "z"}


@pytest.mark.skipif(not _native_available(), reason="C++ native not built")
def test_native_louvain_empty_graph() -> None:
    native = importlib.import_module("domain.structural_graph._native_graph")
    result = native.compute_louvain([], [], 1.0, 42)
    assert result == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Python suffix-index construction
# ═══════════════════════════════════════════════════════════════════════════════

def test_suffix_index_direct_match() -> None:
    """A file is reachable by its full path as well as any suffix."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    assert idx.get("backend/shared/logger.py") == "backend/shared/logger.py"
    assert idx.get("shared/logger.py") == "backend/shared/logger.py"
    assert idx.get("logger.py") == "backend/shared/logger.py"


def test_suffix_index_first_match_wins() -> None:
    """When two files share a suffix, whichever is iterated first wins (set order is
    not guaranteed). The test only asserts the result is one of the valid candidates."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    result = idx.get("service.py")
    assert result in {
        "backend/domain/structural_graph/service.py",
        "backend/domain/repo_map/service.py",
    }


def test_suffix_index_skips_non_python() -> None:
    """Non-.py files are never indexed."""
    files = {"backend/main.py", "src/renderer/src/index.tsx", "README.md"}
    idx = _build_py_suffix_index(files)
    assert "index.tsx" not in idx
    assert "README.md" not in idx
    assert "main.py" in idx


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Python absolute import resolution
# ═══════════════════════════════════════════════════════════════════════════════

def test_resolve_api_imports_domain() -> None:
    """backend/api/structural_graph.py imports domain.structural_graph.service."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    assert _resolve_py_import("domain.structural_graph.service", idx) == \
        "backend/domain/structural_graph/service.py"
    assert _resolve_py_import("domain.structural_graph.types", idx) == \
        "backend/domain/structural_graph/types.py"


def test_resolve_domain_imports_infra() -> None:
    """Service imports infrastructure.db.database."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    assert _resolve_py_import("infrastructure.db.database", idx) == \
        "backend/infrastructure/db/database.py"


def test_resolve_shared_imports() -> None:
    """Any module importing shared.logger / shared.utils gets the right file."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    assert _resolve_py_import("shared.logger", idx) == "backend/shared/logger.py"
    assert _resolve_py_import("shared.utils", idx) == "backend/shared/utils.py"
    assert _resolve_py_import("shared.errors", idx) == "backend/shared/errors.py"


def test_resolve_louvain_fallback_import() -> None:
    """domain.structural_graph._louvain_fallback resolves correctly."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    assert _resolve_py_import(
        "domain.structural_graph._louvain_fallback", idx
    ) == "backend/domain/structural_graph/_louvain_fallback.py"


def test_external_imports_return_none() -> None:
    """Standard library / third-party imports must NOT resolve."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    for imp in ["fastapi", "asyncio", "json", "pydantic", "networkx", "aiosqlite"]:
        assert _resolve_py_import(imp, idx) is None, f"{imp!r} should be external"


def test_old_logic_fails_without_prefix() -> None:
    """Demonstrate why the original direct-match logic produced zero internal edges."""
    file_set = CODESPECTRA_FILES
    imp = "domain.structural_graph.service"
    py_guess = imp.replace(".", "/") + ".py"        # "domain/structural_graph/service.py"
    assert py_guess not in file_set, (
        "Old direct-match found the file — prefix logic not needed in this fixture"
    )
    idx = _build_py_suffix_index(file_set)
    assert idx.get(py_guess) == "backend/domain/structural_graph/service.py"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Bug 2: Python package __init__.py fallback
# ═══════════════════════════════════════════════════════════════════════════════

def test_package_import_resolves_to_init() -> None:
    """'from domain.analysis.agents import X' targets the agents package.
    The suffix index must resolve it to 'backend/domain/analysis/agents/__init__.py'
    via the __init__.py fallback when 'domain/analysis/agents.py' is absent.
    """
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    result = _resolve_py_import("domain.analysis.agents", idx)
    assert result == "backend/domain/analysis/agents/__init__.py"


def test_package_import_prefers_module_over_init() -> None:
    """When both 'domain/analysis.py' and 'domain/analysis/__init__.py' exist,
    the direct .py match wins (first lookup wins)."""
    files = {
        "backend/domain/analysis.py",
        "backend/domain/analysis/__init__.py",
    }
    idx = _build_py_suffix_index(files)
    result = _resolve_py_import("domain.analysis", idx)
    assert result == "backend/domain/analysis.py"


def test_package_import_top_level_package() -> None:
    """'from domain.analysis import X' resolves to 'backend/domain/analysis/__init__.py'."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    result = _resolve_py_import("domain.analysis", idx)
    assert result == "backend/domain/analysis/__init__.py"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — Bug 1: relative TS import resolution with '..' segments
# ═══════════════════════════════════════════════════════════════════════════════

def test_relative_import_double_dot_dot() -> None:
    """'../../store/local-repo.store' from 'src/renderer/src/screens/analysis/index.tsx'
    must resolve to 'src/renderer/src/store/local-repo.store.ts'.

    This was broken when using Path.as_posix() which left '..' unresolved.
    The fix uses os.path.normpath() which collapses '..' correctly.
    """
    result = _resolve_relative_import(
        "src/renderer/src/screens/analysis/index.tsx",
        "../../store/local-repo.store",
        CODESPECTRA_FILES,
    )
    assert result == "src/renderer/src/store/local-repo.store.ts"


def test_relative_import_single_dot_dot_from_nested() -> None:
    """'../../hooks/useTheme' from AppShell.tsx (2 levels deep under src/renderer/src/)
    must resolve to 'src/renderer/src/hooks/useTheme.ts'."""
    result = _resolve_relative_import(
        "src/renderer/src/components/layout/AppShell.tsx",
        "../../hooks/useTheme",
        CODESPECTRA_FILES,
    )
    assert result == "src/renderer/src/hooks/useTheme.ts"


def test_relative_import_same_dir() -> None:
    """Same-directory import (no '..') resolves to the correct file."""
    result = _resolve_relative_import(
        "src/renderer/src/screens/analysis/index.tsx",
        "./index",
        CODESPECTRA_FILES,
    )
    assert result == "src/renderer/src/screens/analysis/index.tsx"


def test_relative_import_unresolvable_returns_none() -> None:
    """An import that points outside the file set returns None (stays external)."""
    result = _resolve_relative_import(
        "src/renderer/src/screens/analysis/index.tsx",
        "../../nonexistent/module",
        CODESPECTRA_FILES,
    )
    assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8 — detect_communities() edge re-resolution
# ═══════════════════════════════════════════════════════════════════════════════

def test_community_edge_reresolution_recovers_broken_python_data() -> None:
    """Simulate the bad existing DB state (Python imports mis-labelled external)
    and verify that detect_communities() recovers the correct internal edges."""
    node_id_set = CODESPECTRA_FILES
    idx = _build_py_suffix_index(node_id_set)

    bad_rows = [
        _make_edge("backend/api/structural_graph.py",
                   "domain.structural_graph.service", is_external=True),
        _make_edge("backend/api/structural_graph.py",
                   "domain.structural_graph.types", is_external=True),
        _make_edge("backend/domain/structural_graph/service.py",
                   "infrastructure.db.database", is_external=True),
        _make_edge("backend/domain/structural_graph/service.py",
                   "shared.logger", is_external=True),
        # Genuinely external — must NOT appear in output
        _make_edge("backend/api/structural_graph.py", "fastapi", is_external=True),
        _make_edge("backend/domain/structural_graph/service.py", "asyncio", is_external=True),
    ]

    edges = _resolve_community_edges(bad_rows, node_id_set, idx)
    srcs_dsts = {(s, d) for s, d, _ in edges}

    assert ("backend/api/structural_graph.py",
            "backend/domain/structural_graph/service.py") in srcs_dsts
    assert ("backend/api/structural_graph.py",
            "backend/domain/structural_graph/types.py") in srcs_dsts
    assert ("backend/domain/structural_graph/service.py",
            "backend/infrastructure/db/database.py") in srcs_dsts
    assert ("backend/domain/structural_graph/service.py",
            "backend/shared/logger.py") in srcs_dsts
    assert not any(d in {"fastapi", "asyncio"} for _, d, _ in edges)


def test_ts_relative_import_reresolution_in_community_edges() -> None:
    """Unresolved TS relative imports stored pre-Bug-1-fix (dst starts with '.')
    are recovered by detect_communities() re-resolution without a graph rebuild."""
    node_id_set = CODESPECTRA_FILES
    idx = _build_py_suffix_index(node_id_set)

    bad_rows = [
        _make_edge(
            "src/renderer/src/screens/analysis/index.tsx",
            "../../store/local-repo.store",   # raw import string, starts with '.'
            is_external=True,
        ),
        _make_edge(
            "src/renderer/src/screens/index-overview/index.tsx",
            "../../store/job.store",
            is_external=True,
        ),
        _make_edge(
            "src/renderer/src/screens/analysis/index.tsx",
            "react",
            is_external=True,
        ),
    ]

    edges = _resolve_community_edges(bad_rows, node_id_set, idx)
    srcs_dsts = {(s, d) for s, d, _ in edges}

    assert ("src/renderer/src/screens/analysis/index.tsx",
            "src/renderer/src/store/local-repo.store.ts") in srcs_dsts
    assert ("src/renderer/src/screens/index-overview/index.tsx",
            "src/renderer/src/store/job.store.ts") in srcs_dsts
    assert not any(d == "react" for _, d, _ in edges)


def test_community_edge_no_self_loops() -> None:
    """Self-loop edges (src == dst) must be dropped."""
    node_id_set = {"backend/shared/utils.py"}
    idx = _build_py_suffix_index(node_id_set)
    rows = [_make_edge("backend/shared/utils.py", "backend/shared/utils.py", is_external=False)]
    edges = _resolve_community_edges(rows, node_id_set, idx)
    assert edges == []


def test_community_edge_dedup() -> None:
    """Duplicate edges (same src+dst from multiple rows) appear only once."""
    node_id_set = {
        "backend/api/structural_graph.py",
        "backend/domain/structural_graph/service.py",
    }
    idx = _build_py_suffix_index(node_id_set)
    rows = [
        _make_edge("backend/api/structural_graph.py",
                   "backend/domain/structural_graph/service.py", is_external=False),
        _make_edge("backend/api/structural_graph.py",
                   "backend/domain/structural_graph/service.py", is_external=False),
        # Same pair via re-resolution — should still deduplicate
        _make_edge("backend/api/structural_graph.py",
                   "domain.structural_graph.service", is_external=True),
    ]
    edges = _resolve_community_edges(rows, node_id_set, idx)
    pairs = [(s, d) for s, d, _ in edges]
    assert pairs.count(
        ("backend/api/structural_graph.py", "backend/domain/structural_graph/service.py")
    ) == 1


def test_community_edge_src_must_be_known_node() -> None:
    """Edges whose src is not in the manifest node set are silently dropped."""
    node_id_set = {"backend/domain/structural_graph/service.py"}
    idx = _build_py_suffix_index(node_id_set)
    rows = [
        _make_edge("some/unknown/file.py",
                   "backend/domain/structural_graph/service.py", is_external=False),
    ]
    edges = _resolve_community_edges(rows, node_id_set, idx)
    assert edges == []
