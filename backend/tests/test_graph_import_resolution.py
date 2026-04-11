"""Tests for structural graph import resolution (CS-102 bug fix).

Ground truth: the CodeSpectra repo itself — files live under backend/ and src/,
but Python modules are imported without the backend/ prefix.
The suffix-index must bridge that gap.
"""
from __future__ import annotations

# ── helpers mirroring service.py logic ────────────────────────────────────────

def _build_py_suffix_index(file_set: set[str]) -> dict[str, str]:
    """Replicate the suffix-index construction from StructuralGraphService.build()."""
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
    """Convert a Python import string to a file path via the suffix index."""
    py_guess = imp.replace(".", "/") + ".py"
    return index.get(py_guess)


def _resolve_community_edges(
    edge_rows: list[dict],
    node_id_set: set[str],
    py_suffix_index: dict[str, str],
) -> list[tuple[str, str, float]]:
    """Replicate detect_communities() edge-dedup + re-resolution logic."""
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
            py_guess = dst.replace(".", "/") + ".py"
            resolved = py_suffix_index.get(py_guess)
            if resolved and resolved != src and (src, resolved) not in edge_set:
                edge_tuples.append((src, resolved, 1.0))
                edge_set.add((src, resolved))

    return edge_tuples


# ── realistic CodeSpectra file set ────────────────────────────────────────────

CODESPECTRA_FILES: set[str] = {
    # backend API layer
    "backend/api/structural_graph.py",
    "backend/api/local_repo.py",
    "backend/api/analysis.py",
    "backend/api/workspace.py",
    # domain — structural_graph
    "backend/domain/structural_graph/service.py",
    "backend/domain/structural_graph/types.py",
    "backend/domain/structural_graph/_louvain_fallback.py",
    "backend/domain/structural_graph/_scc_fallback.py",
    # domain — analysis
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
}


# ── suffix index tests ────────────────────────────────────────────────────────

def test_suffix_index_direct_match():
    """A file is reachable by its full path as well as any suffix."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    assert idx.get("backend/shared/logger.py") == "backend/shared/logger.py"
    assert idx.get("shared/logger.py") == "backend/shared/logger.py"
    assert idx.get("logger.py") == "backend/shared/logger.py"


def test_suffix_index_first_match_wins():
    """When two files share a suffix, the first (alphabetically sorted) wins.
    Concretely: 'service.py' is ambiguous; the index picks one deterministically."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    # 'service.py' is a suffix for both structural_graph/service.py and
    # repo_map/service.py — whichever wins, it must be one of them.
    result = idx.get("service.py")
    assert result in {
        "backend/domain/structural_graph/service.py",
        "backend/domain/repo_map/service.py",
    }


def test_suffix_index_skips_non_python():
    """Non-.py files are never indexed."""
    files = {"backend/main.py", "src/renderer/src/index.tsx", "README.md"}
    idx = _build_py_suffix_index(files)
    assert "index.tsx" not in idx
    assert "README.md" not in idx
    assert "main.py" in idx


# ── Python absolute import resolution ────────────────────────────────────────

def test_resolve_api_imports_domain():
    """backend/api/structural_graph.py imports domain.structural_graph.service."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    assert _resolve_py_import("domain.structural_graph.service", idx) == \
        "backend/domain/structural_graph/service.py"
    assert _resolve_py_import("domain.structural_graph.types", idx) == \
        "backend/domain/structural_graph/types.py"


def test_resolve_domain_imports_infra():
    """Service imports infrastructure.db.database."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    assert _resolve_py_import("infrastructure.db.database", idx) == \
        "backend/infrastructure/db/database.py"


def test_resolve_shared_imports():
    """Any module importing shared.logger / shared.utils gets the right file."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    assert _resolve_py_import("shared.logger", idx) == "backend/shared/logger.py"
    assert _resolve_py_import("shared.utils", idx) == "backend/shared/utils.py"
    assert _resolve_py_import("shared.errors", idx) == "backend/shared/errors.py"


def test_resolve_louvain_fallback_import():
    """domain.structural_graph._louvain_fallback resolves correctly."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    assert _resolve_py_import(
        "domain.structural_graph._louvain_fallback", idx
    ) == "backend/domain/structural_graph/_louvain_fallback.py"


def test_external_imports_return_none():
    """Standard library / third-party imports must NOT resolve."""
    idx = _build_py_suffix_index(CODESPECTRA_FILES)
    for imp in ["fastapi", "asyncio", "json", "pydantic", "networkx", "aiosqlite"]:
        assert _resolve_py_import(imp, idx) is None, f"{imp!r} should be external"


def test_old_logic_fails_without_prefix():
    """Demonstrate why the original direct-match logic produced zero internal edges."""
    file_set = CODESPECTRA_FILES
    # Old logic: py_guess must be an exact key in file_set
    imp = "domain.structural_graph.service"
    py_guess = imp.replace(".", "/") + ".py"        # "domain/structural_graph/service.py"
    assert py_guess not in file_set, (
        "Old direct-match found the file — prefix logic not needed in this fixture"
    )
    # New suffix-index logic finds it
    idx = _build_py_suffix_index(file_set)
    assert idx.get(py_guess) == "backend/domain/structural_graph/service.py"


# ── detect_communities edge re-resolution ────────────────────────────────────

def _make_edge(src: str, dst: str, *, is_external: bool) -> dict:
    return {"src_path": src, "dst_path": dst, "is_external": is_external}


def test_community_edge_reresolution_recovers_broken_data():
    """Simulate the bad existing DB state (all Python imports mis-labelled external)
    and verify that detect_communities() recovers the correct internal edges."""
    node_id_set = CODESPECTRA_FILES
    idx = _build_py_suffix_index(node_id_set)

    # These edges exist in DB but are wrongly flagged is_external=1
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
        _make_edge("backend/api/structural_graph.py",
                   "fastapi", is_external=True),
        _make_edge("backend/domain/structural_graph/service.py",
                   "asyncio", is_external=True),
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

    # Genuine externals excluded
    assert not any(d in {"fastapi", "asyncio"} for _, d, _ in edges)


def test_community_edge_no_self_loops():
    """Self-loop edges (src == dst) must be dropped."""
    node_id_set = {"backend/shared/utils.py"}
    idx = _build_py_suffix_index(node_id_set)
    rows = [_make_edge("backend/shared/utils.py", "backend/shared/utils.py", is_external=False)]
    edges = _resolve_community_edges(rows, node_id_set, idx)
    assert edges == []


def test_community_edge_dedup():
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
        # Same pair via re-resolution
        _make_edge("backend/api/structural_graph.py",
                   "domain.structural_graph.service", is_external=True),
    ]
    edges = _resolve_community_edges(rows, node_id_set, idx)
    pairs = [(s, d) for s, d, _ in edges]
    assert pairs.count(
        ("backend/api/structural_graph.py", "backend/domain/structural_graph/service.py")
    ) == 1


def test_community_edge_src_must_be_known_node():
    """Edges whose src is not in the manifest node set are silently dropped."""
    node_id_set = {"backend/domain/structural_graph/service.py"}
    idx = _build_py_suffix_index(node_id_set)
    rows = [
        _make_edge("some/unknown/file.py",
                   "backend/domain/structural_graph/service.py", is_external=False),
    ]
    edges = _resolve_community_edges(rows, node_id_set, idx)
    assert edges == []
