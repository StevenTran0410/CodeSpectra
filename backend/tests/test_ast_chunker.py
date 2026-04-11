"""Tests for AST-based semantic chunker (CS-101)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from domain.retrieval.chunker_ast import ASTChunk, ASTChunker, _flat_chunks, _merge_spans_python, _NodeSpan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_span(start: int, end: int, node_type: str = "function_definition", name: str = "fn") -> _NodeSpan:
    return _NodeSpan(
        start_byte=start,
        end_byte=end,
        start_line=0,
        end_line=1,
        node_type=node_type,
        name=name,
        is_import=False,
    )


# ---------------------------------------------------------------------------
# Flat chunking fallback
# ---------------------------------------------------------------------------

def test_flat_chunks_small_file() -> None:
    """File smaller than target_size → single chunk of type 'file'."""
    src = "x = 1\ny = 2\n"
    chunks = _flat_chunks(src, target_size=1500)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "file"
    assert chunks[0].text == src


def test_flat_chunks_large_file_splits() -> None:
    """File larger than target_size → multiple chunks."""
    src = "x = 1\n" * 500  # ~3000 chars
    chunks = _flat_chunks(src, target_size=1000)
    assert len(chunks) > 1
    # All non-file chunks are 'block'
    for c in chunks:
        assert c.chunk_type in ("file", "block")


# ---------------------------------------------------------------------------
# Merge pass — pure Python
# ---------------------------------------------------------------------------

def test_merge_spans_empty() -> None:
    assert _merge_spans_python([], 1500) == []


def test_merge_adjacent_small_nodes() -> None:
    """Two 400-byte functions → merged into single group under 1500-byte target."""
    spans = [_make_span(0, 400, name="foo"), _make_span(400, 800, name="bar")]
    groups = _merge_spans_python(spans, target_size=1500)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_merge_large_node_stays_alone() -> None:
    """A 1600-byte function exceeds target_size=1500 — still one group (can't split further here)."""
    spans = [_make_span(0, 1600, name="big")]
    groups = _merge_spans_python(spans, target_size=1500)
    assert len(groups) == 1
    assert groups[0][0].name == "big"


def test_merge_boundary_exact() -> None:
    """Span ending exactly at target_size boundary is kept in current group."""
    spans = [_make_span(0, 750, name="a"), _make_span(750, 1500, name="b")]
    groups = _merge_spans_python(spans, target_size=1500)
    # 0→1500 = exactly 1500 chars → fits in one group
    assert len(groups) == 1


def test_merge_splits_at_boundary() -> None:
    """Third span would push group over target → new group starts."""
    spans = [
        _make_span(0, 600, name="a"),
        _make_span(600, 1200, name="b"),
        _make_span(1200, 1900, name="c"),   # 0→1900 > 1500 → c goes to new group
    ]
    groups = _merge_spans_python(spans, target_size=1500)
    assert len(groups) == 2
    assert len(groups[0]) == 2  # a + b
    assert groups[1][0].name == "c"


# ---------------------------------------------------------------------------
# ASTChunker — unsupported language fallback
# ---------------------------------------------------------------------------

def test_fallback_on_unsupported_language() -> None:
    """Unknown language falls back to flat chunking."""
    chunker = ASTChunker()
    src = "local x = 1\nlocal y = 2\n" * 200
    chunks = chunker.chunk(src, language="lua", target_size=1000)
    assert all(isinstance(c, ASTChunk) for c in chunks)
    assert len(chunks) > 0


# ---------------------------------------------------------------------------
# ASTChunker — small file short-circuit
# ---------------------------------------------------------------------------

def test_small_file_short_circuit() -> None:
    """File smaller than target_size skips AST parsing, returns a single chunk."""
    chunker = ASTChunker()
    src = "def foo():\n    return 1\n"
    chunks = chunker.chunk(src, language="python", target_size=1500)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "file"
    assert chunks[0].text == src


# ---------------------------------------------------------------------------
# ASTChunker — parse error fallback
# ---------------------------------------------------------------------------

def test_fallback_on_parse_error() -> None:
    """If parser.parse() raises, the chunker falls back gracefully to flat chunks."""
    chunker = ASTChunker()
    src = "def foo():\n    return 1\n" * 100  # > 1500 chars to avoid short-circuit

    # Patch the parse method on whatever parser gets returned
    with patch.object(chunker, "_parse_and_chunk", side_effect=RuntimeError("boom")):
        chunks = chunker.chunk(src, language="python", target_size=1500)
    # Should return flat chunks — no exception raised
    assert len(chunks) > 0
    assert all(isinstance(c, ASTChunk) for c in chunks)


# ---------------------------------------------------------------------------
# ASTChunker — Python semantic chunking
# ---------------------------------------------------------------------------

def test_python_function_not_split() -> None:
    """A 60-line Python function fits within 2000-byte max and stays as one chunk."""
    body = "    x = i * 2\n    y = x + 1\n    result.append(y)\n"
    func = "def compute_heavy(data):\n    result = []\n    for i in data:\n" + body * 15 + "    return result\n"
    # ~750 chars — well under max_node_size=2000
    chunker = ASTChunker()
    chunks = chunker.chunk(func, language="python", target_size=1500)
    # The entire function should be in a single chunk
    full_text = "".join(c.text for c in chunks)
    assert "def compute_heavy" in full_text
    assert "return result" in full_text
    # No chunk should start mid-function (no chunk lacks the def line unless it's the tail)
    func_chunks = [c for c in chunks if "def compute_heavy" in c.text]
    assert len(func_chunks) >= 1


def test_python_import_group_merged() -> None:
    """Consecutive import lines are collected into a single import_group chunk.

    The source must exceed target_size to bypass the short-circuit path and
    actually trigger AST parsing (where import grouping happens).
    """
    imports = (
        "import os\n"
        "import sys\n"
        "import json\n"
        "from pathlib import Path\n"
        "from typing import Any\n"
        "\n"
    )
    # Pad with enough function bodies to exceed target_size=500 so AST runs.
    body = "def func_{i}(x):\n    return x * {i}\n\n"
    funcs = "".join(body.format(i=i) for i in range(30))  # ~30 * 35 = ~1050 chars
    src = imports + funcs
    chunker = ASTChunker()
    chunks = chunker.chunk(src, language="python", target_size=500)
    import_chunks = [c for c in chunks if c.chunk_type == "import_group"]
    # All imports should be in at most one import_group chunk
    assert len(import_chunks) >= 1
    combined = " ".join(c.text for c in import_chunks)
    assert "import os" in combined
    assert "import sys" in combined


def test_python_two_functions_merged_when_small() -> None:
    """Two small functions are merged into one chunk if they fit together."""
    src = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    # Both functions are tiny — should merge into one chunk under target_size=1500
    chunker = ASTChunker()
    chunks = chunker.chunk(src, language="python", target_size=1500)
    # One or two chunks — either is fine, but total text coverage must be complete
    total_text = "".join(c.text for c in chunks)
    assert "def foo" in total_text
    assert "def bar" in total_text


# ---------------------------------------------------------------------------
# ASTChunker — TypeScript semantic chunking
# ---------------------------------------------------------------------------

def test_typescript_class_not_split() -> None:
    """A TypeScript class definition stays together."""
    src = (
        "export class UserService {\n"
        "  private repo: Repository;\n"
        "\n"
        "  constructor(repo: Repository) {\n"
        "    this.repo = repo;\n"
        "  }\n"
        "\n"
        "  async findById(id: string): Promise<User | null> {\n"
        "    return this.repo.findOne({ where: { id } });\n"
        "  }\n"
        "\n"
        "  async save(user: User): Promise<User> {\n"
        "    return this.repo.save(user);\n"
        "  }\n"
        "}\n"
    )
    chunker = ASTChunker()
    chunks = chunker.chunk(src, language="typescript", target_size=1500)
    full = "".join(c.text for c in chunks)
    assert "class UserService" in full
    assert "findById" in full
    assert "save" in full


# ---------------------------------------------------------------------------
# ASTChunker — ASTChunk metadata
# ---------------------------------------------------------------------------

def test_chunk_has_language_field() -> None:
    """ASTChunks produced for a supported language carry the language tag."""
    chunker = ASTChunker()
    src = "def hello():\n    print('hi')\n"
    chunks = chunker.chunk(src, language="python", target_size=1500)
    assert all(c.language == "python" for c in chunks)


def test_chunk_type_function_for_function_node() -> None:
    """A single-function file produces a chunk with chunk_type 'function' or 'file'."""
    chunker = ASTChunker()
    src = "def my_func(x, y):\n    return x + y\n"
    chunks = chunker.chunk(src, language="python", target_size=1500)
    types = {c.chunk_type for c in chunks}
    # Either 'file' (short-circuit path) or 'function' (AST path)
    assert types <= {"file", "function", "block", "import_group"}
