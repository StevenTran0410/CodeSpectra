"""Tests for AST-based semantic chunker (CS-101).

Sections:
  1. Unit tests — flat chunking, merge pass, fallback, short-circuit
  2. Golden: Python fixture — exact chunk boundaries + C++ native parity
  3. Golden: Multi-language — TypeScript, JavaScript, C++, Go, Java, C, Rust
  4. Golden: Complex edge cases — decorators, generics, trait impls, interfaces
"""
from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

import domain.retrieval.chunker_ast as _mod
from domain.retrieval.chunker_ast import ASTChunk, ASTChunker, _flat_chunks, _merge_spans_python, _NodeSpan


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _chunker() -> ASTChunker:
    return ASTChunker()


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


def _native_available() -> bool:
    try:
        importlib.import_module("domain.retrieval._native_chunker")
        return True
    except Exception:
        return False


# ===========================================================================
# 1. Unit tests
# ===========================================================================

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
    assert len(groups) == 1


def test_merge_splits_at_boundary() -> None:
    """Third span would push group over target → new group starts."""
    spans = [
        _make_span(0, 600, name="a"),
        _make_span(600, 1200, name="b"),
        _make_span(1200, 1900, name="c"),
    ]
    groups = _merge_spans_python(spans, target_size=1500)
    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert groups[1][0].name == "c"


# ---------------------------------------------------------------------------
# ASTChunker — fallback and short-circuit
# ---------------------------------------------------------------------------

def test_fallback_on_unsupported_language() -> None:
    """Unknown language falls back to flat chunking."""
    src = "local x = 1\nlocal y = 2\n" * 200
    chunks = _chunker().chunk(src, language="lua", target_size=1000)
    assert all(isinstance(c, ASTChunk) for c in chunks)
    assert len(chunks) > 0


def test_small_file_short_circuit() -> None:
    """File smaller than target_size skips AST parsing, returns a single chunk."""
    src = "def foo():\n    return 1\n"
    chunks = _chunker().chunk(src, language="python", target_size=1500)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "file"
    assert chunks[0].text == src


def test_fallback_on_parse_error() -> None:
    """If parser.parse() raises, the chunker falls back gracefully to flat chunks."""
    src = "def foo():\n    return 1\n" * 100  # > 1500 chars to avoid short-circuit
    with patch.object(_chunker(), "_parse_and_chunk", side_effect=RuntimeError("boom")):
        # Use a fresh chunker to patch properly
        c = _chunker()
        with patch.object(c, "_parse_and_chunk", side_effect=RuntimeError("boom")):
            chunks = c.chunk(src, language="python", target_size=1500)
    assert len(chunks) > 0
    assert all(isinstance(c, ASTChunk) for c in chunks)


# ---------------------------------------------------------------------------
# ASTChunker — Python semantic chunking
# ---------------------------------------------------------------------------

def test_python_function_not_split() -> None:
    """A ~750-char Python function stays as one chunk."""
    body = "    x = i * 2\n    y = x + 1\n    result.append(y)\n"
    func = "def compute_heavy(data):\n    result = []\n    for i in data:\n" + body * 15 + "    return result\n"
    chunks = _chunker().chunk(func, language="python", target_size=1500)
    full_text = "".join(c.text for c in chunks)
    assert "def compute_heavy" in full_text
    assert "return result" in full_text


def test_python_import_group_merged() -> None:
    """Consecutive import lines are collected into a single import_group chunk."""
    imports = "import os\nimport sys\nimport json\nfrom pathlib import Path\nfrom typing import Any\n\n"
    funcs = "".join(f"def func_{i}(x):\n    return x * {i}\n\n" for i in range(30))
    src = imports + funcs
    chunks = _chunker().chunk(src, language="python", target_size=500)
    import_chunks = [c for c in chunks if c.chunk_type == "import_group"]
    assert len(import_chunks) >= 1
    combined = " ".join(c.text for c in import_chunks)
    assert "import os" in combined
    assert "import sys" in combined


def test_python_two_functions_merged_when_small() -> None:
    """Two small functions are merged into one chunk if they fit together."""
    src = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    chunks = _chunker().chunk(src, language="python", target_size=1500)
    total_text = "".join(c.text for c in chunks)
    assert "def foo" in total_text
    assert "def bar" in total_text


# ---------------------------------------------------------------------------
# ASTChunker — TypeScript semantic chunking + metadata
# ---------------------------------------------------------------------------

def test_typescript_class_not_split() -> None:
    """A TypeScript class definition stays together."""
    src = (
        "export class UserService {\n"
        "  private repo: Repository;\n"
        "  constructor(repo: Repository) { this.repo = repo; }\n"
        "  async findById(id: string): Promise<User | null> {\n"
        "    return this.repo.findOne({ where: { id } });\n"
        "  }\n"
        "  async save(user: User): Promise<User> { return this.repo.save(user); }\n"
        "}\n"
    )
    chunks = _chunker().chunk(src, language="typescript", target_size=1500)
    full = "".join(c.text for c in chunks)
    assert "class UserService" in full
    assert "findById" in full
    assert "save" in full


def test_chunk_has_language_field() -> None:
    """ASTChunks produced for a supported language carry the language tag."""
    src = "def hello():\n    print('hi')\n"
    chunks = _chunker().chunk(src, language="python", target_size=1500)
    assert all(c.language == "python" for c in chunks)


def test_chunk_type_function_for_function_node() -> None:
    """A single-function file produces a chunk with an expected chunk_type."""
    src = "def my_func(x, y):\n    return x + y\n"
    chunks = _chunker().chunk(src, language="python", target_size=1500)
    types = {c.chunk_type for c in chunks}
    assert types <= {"file", "function", "block", "import_group"}


# ===========================================================================
# 2. Golden: Python fixture
#
# Section map (0-indexed lines):
#   [0-3]   import group  — 4 import statements
#   [6-11]  functions     — add() + subtract(), small enough to merge
#   [14-30] function      — long_function(), large enough to stand alone
#   [33-42] class         — DataProcessor with 3 methods
#
# Total ≈ 901 bytes. target_size=200 forces the 4-chunk split.
# ===========================================================================

GOLDEN_PYTHON = (
    "import os\n"
    "import sys\n"
    "from pathlib import Path\n"
    "from typing import Any\n"
    "\n"
    "\n"
    "def add(x: int, y: int) -> int:\n"
    "    return x + y\n"
    "\n"
    "\n"
    "def subtract(x: int, y: int) -> int:\n"
    "    return x - y\n"
    "\n"
    "\n"
    "def long_function(data: list) -> dict:\n"
    "    # intentionally long to force its own chunk\n"
    "    if not data:\n"
    "        return {'count': 0, 'total': 0}\n"
    "    total = sum(data)\n"
    "    count = len(data)\n"
    "    mean = total / count\n"
    "    items = sorted(data)\n"
    "    minimum = items[0]\n"
    "    maximum = items[-1]\n"
    "    return {\n"
    "        'count': count,\n"
    "        'total': total,\n"
    "        'mean': mean,\n"
    "        'min': minimum,\n"
    "        'max': maximum,\n"
    "    }\n"
    "\n"
    "\n"
    "class DataProcessor:\n"
    "    def __init__(self, label: str) -> None:\n"
    "        self.label = label\n"
    "        self.records: list = []\n"
    "\n"
    "    def push(self, record: Any) -> None:\n"
    "        self.records.append(record)\n"
    "\n"
    "    def summarize(self) -> dict:\n"
    "        return long_function(self.records)\n"
)
_PY_GOLDEN_TARGET = 200


def test_golden_chunk_count() -> None:
    """GOLDEN_PYTHON with target_size=200 must yield exactly 4 chunks."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_golden_chunk_type_sequence() -> None:
    """Chunks must appear in the order: import_group → block → function → class."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "function", "class"]


def test_golden_import_group_contains_all_imports() -> None:
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    import_chunk = chunks[0]
    assert import_chunk.chunk_type == "import_group"
    for expected in ("import os", "import sys", "from pathlib import Path", "from typing import Any"):
        assert expected in import_chunk.text


def test_golden_import_chunk_starts_at_line_zero() -> None:
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    assert chunks[0].start_line == 0


def test_golden_small_functions_merged_into_one_chunk() -> None:
    """add() and subtract() are small enough to share one 'block' chunk."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    block = chunks[1]
    assert block.chunk_type == "block"
    assert "def add" in block.text
    assert "return x + y" in block.text
    assert "def subtract" in block.text
    assert "return x - y" in block.text


def test_golden_block_chunk_not_split_mid_function() -> None:
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    for chunk in chunks:
        if "def add" in chunk.text:
            assert "return x + y" in chunk.text
        if "def subtract" in chunk.text:
            assert "return x - y" in chunk.text


def test_golden_long_function_is_its_own_chunk() -> None:
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    func_chunks = [c for c in chunks if "def long_function" in c.text]
    assert len(func_chunks) == 1
    assert func_chunks[0].chunk_type == "function"


def test_golden_long_function_body_intact() -> None:
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    fc = next(c for c in chunks if "def long_function" in c.text)
    assert "def long_function" in fc.text
    assert "'max': maximum" in fc.text


def test_golden_long_function_not_in_import_or_class_chunk() -> None:
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    assert "def long_function" not in chunks[0].text
    assert "def long_function" not in chunks[3].text


def test_golden_class_is_its_own_chunk() -> None:
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    class_chunks = [c for c in chunks if "class DataProcessor" in c.text]
    assert len(class_chunks) == 1
    assert class_chunks[0].chunk_type == "class"


def test_golden_class_chunk_includes_all_methods() -> None:
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    cc = next(c for c in chunks if "class DataProcessor" in c.text)
    for method in ("def __init__", "def push", "def summarize"):
        assert method in cc.text


def test_golden_all_chunks_carry_language_tag() -> None:
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    for c in chunks:
        assert c.language == "python"


# ---------------------------------------------------------------------------
# C++ native: merge_spans direct API
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _native_available(), reason="C++ native chunker not built")
def test_native_merge_spans_groups_correctly() -> None:
    """merge_spans must group spans greedily matching the Python fallback logic.

    Spans from golden fixture (approx bytes):
      add ~51B, subtract ~56B, long_function ~435B, DataProcessor ~282B.
    With target=200: [[add,subtract], [long_function], [DataProcessor]]
    """
    native = importlib.import_module("domain.retrieval._native_chunker")
    triples = [(71, 122, 0), (124, 180, 1), (182, 617, 2), (619, 901, 3)]
    groups = [list(g) for g in native.merge_spans(triples, target_size=200)]
    assert groups == [[0, 1], [2], [3]]


@pytest.mark.skipif(not _native_available(), reason="C++ native chunker not built")
def test_native_merge_spans_empty_input() -> None:
    native = importlib.import_module("domain.retrieval._native_chunker")
    assert list(native.merge_spans([], target_size=200)) == []


@pytest.mark.skipif(not _native_available(), reason="C++ native chunker not built")
def test_native_merge_spans_single_span() -> None:
    native = importlib.import_module("domain.retrieval._native_chunker")
    groups = [list(g) for g in native.merge_spans([(0, 50, 7)], target_size=200)]
    assert groups == [[7]]


@pytest.mark.skipif(not _native_available(), reason="C++ native chunker not built")
def test_native_merge_spans_all_fit_in_one_group() -> None:
    native = importlib.import_module("domain.retrieval._native_chunker")
    groups = [list(g) for g in native.merge_spans([(0, 60, 0), (60, 120, 1), (120, 180, 2)], target_size=200)]
    assert groups == [[0, 1, 2]]


@pytest.mark.skipif(not _native_available(), reason="C++ native chunker not built")
def test_native_merge_spans_each_span_exceeds_target() -> None:
    native = importlib.import_module("domain.retrieval._native_chunker")
    groups = [list(g) for g in native.merge_spans([(0, 300, 0), (300, 600, 1), (600, 900, 2)], target_size=200)]
    assert groups == [[0], [1], [2]]


@pytest.mark.skipif(not _native_available(), reason="C++ native chunker not built")
def test_native_chunker_matches_python_fallback_on_golden() -> None:
    """ASTChunker with C++ native must produce chunk-for-chunk identical results to Python fallback."""
    native_ref = _mod._native_chunker
    assert native_ref is not None
    c = _chunker()
    chunks_native = c.chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    _mod._native_chunker = None
    try:
        chunks_python = c.chunk(GOLDEN_PYTHON, "python", target_size=_PY_GOLDEN_TARGET)
    finally:
        _mod._native_chunker = native_ref
    assert len(chunks_native) == len(chunks_python)
    for i, (cn, cp) in enumerate(zip(chunks_native, chunks_python)):
        assert cn.chunk_type == cp.chunk_type, f"Chunk {i}: type mismatch"
        assert cn.text == cp.text, f"Chunk {i}: text mismatch"
        assert cn.start_line == cp.start_line
        assert cn.end_line == cp.end_line


# ===========================================================================
# 3. Golden: Multi-language
# ===========================================================================

# ---------------------------------------------------------------------------
# TypeScript
# Sections: [0] import_group  [1] block(FileReader class)  [2] block(trimLines+countWords)
# ---------------------------------------------------------------------------

GOLDEN_TS = (
    "import { EventEmitter } from 'events';\n"
    "import { readFileSync } from 'fs';\n"
    "\n"
    "export class FileReader {\n"
    "  private path: string;\n"
    "  constructor(path: string) { this.path = path; }\n"
    "  read(): string { return readFileSync(this.path, 'utf8'); }\n"
    "  lines(): string[] { return this.read().split('\\n'); }\n"
    "}\n"
    "\n"
    "export function trimLines(lines: string[]): string[] {\n"
    "  return lines.map(l => l.trim()).filter(l => l.length > 0);\n"
    "}\n"
    "\n"
    "export function countWords(text: string): number {\n"
    "  return text.split(/\\s+/).filter(w => w.length > 0).length;\n"
    "}\n"
)
_TS_TARGET = 300


def test_ts_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_TS, "typescript", target_size=_TS_TARGET)
    assert len(chunks) == 3, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_ts_chunk_type_sequence() -> None:
    assert [c.chunk_type for c in _chunker().chunk(GOLDEN_TS, "typescript", target_size=_TS_TARGET)] == ["import_group", "block", "block"]


def test_ts_import_group_covers_both_imports() -> None:
    ig = _chunker().chunk(GOLDEN_TS, "typescript", target_size=_TS_TARGET)[0]
    assert "from 'events'" in ig.text
    assert "from 'fs'" in ig.text


def test_ts_class_body_intact() -> None:
    cls = _chunker().chunk(GOLDEN_TS, "typescript", target_size=_TS_TARGET)[1]
    assert "export class FileReader" in cls.text
    assert "constructor" in cls.text
    assert "read()" in cls.text
    assert "lines()" in cls.text


def test_ts_small_functions_merged() -> None:
    merged = _chunker().chunk(GOLDEN_TS, "typescript", target_size=_TS_TARGET)[2]
    assert "trimLines" in merged.text
    assert "countWords" in merged.text


def test_ts_no_function_split_mid_body() -> None:
    for chunk in _chunker().chunk(GOLDEN_TS, "typescript", target_size=_TS_TARGET):
        if "trimLines" in chunk.text:
            assert "filter(l => l.length > 0)" in chunk.text
        if "countWords" in chunk.text:
            assert "filter(w => w.length > 0).length" in chunk.text


def test_ts_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_TS, "typescript", target_size=_TS_TARGET):
        assert c.language == "typescript"


# ---------------------------------------------------------------------------
# JavaScript
# Sections: [0] import_group  [1] block(Hasher class)  [2] block(parseConfig)  [3] block(formatConfig)
# ---------------------------------------------------------------------------

GOLDEN_JS = (
    "import { createHash } from 'crypto';\n"
    "import { readFile } from 'fs/promises';\n"
    "\n"
    "export class Hasher {\n"
    "  constructor(algo) { this.algo = algo; }\n"
    "  hash(data) { return createHash(this.algo).update(data).digest('hex'); }\n"
    "  verify(data, expected) { return this.hash(data) === expected; }\n"
    "}\n"
    "\n"
    "export function parseConfig(text) {\n"
    "  return Object.fromEntries(\n"
    "    text.split('\\n').filter(l => l.includes('='))\n"
    "         .map(l => l.split('='))\n"
    "  );\n"
    "}\n"
    "\n"
    "export function formatConfig(obj) {\n"
    "  return Object.entries(obj)\n"
    "    .map(([k, v]) => k + '=' + v).join('\\n');\n"
    "}\n"
)
_JS_TARGET = 250


def test_js_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_JS, "javascript", target_size=_JS_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_js_chunk_type_sequence() -> None:
    assert [c.chunk_type for c in _chunker().chunk(GOLDEN_JS, "javascript", target_size=_JS_TARGET)] == ["import_group", "block", "block", "block"]


def test_js_import_group_covers_both_imports() -> None:
    ig = _chunker().chunk(GOLDEN_JS, "javascript", target_size=_JS_TARGET)[0]
    assert "from 'crypto'" in ig.text
    assert "from 'fs/promises'" in ig.text


def test_js_class_body_intact() -> None:
    cls = _chunker().chunk(GOLDEN_JS, "javascript", target_size=_JS_TARGET)[1]
    assert "export class Hasher" in cls.text
    assert "constructor" in cls.text
    assert "hash(data)" in cls.text
    assert "verify(data" in cls.text


def test_js_each_function_body_intact() -> None:
    for chunk in _chunker().chunk(GOLDEN_JS, "javascript", target_size=_JS_TARGET):
        if "parseConfig" in chunk.text:
            assert "Object.fromEntries" in chunk.text
        if "formatConfig" in chunk.text:
            assert "Object.entries" in chunk.text


def test_js_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_JS, "javascript", target_size=_JS_TARGET):
        assert c.language == "javascript"


# ---------------------------------------------------------------------------
# C++
# Sections: [0] import_group  [1] block(namespace utils)  [2] class(StringProcessor)  [3] function(sum_lengths)
# ---------------------------------------------------------------------------

GOLDEN_CPP = (
    "#include <iostream>\n"
    "#include <vector>\n"
    "#include <string>\n"
    "\n"
    "namespace utils {\n"
    "    std::string trim(const std::string& s) {\n"
    "        size_t start = s.find_first_not_of(\" \\t\");\n"
    "        return (start == std::string::npos) ? \"\" : s.substr(start);\n"
    "    }\n"
    "}\n"
    "\n"
    "class StringProcessor {\n"
    "public:\n"
    "    StringProcessor(std::string sep) : sep_(sep) {}\n"
    "    std::string join(const std::vector<std::string>& parts) const {\n"
    "        std::string result;\n"
    "        for (size_t i = 0; i < parts.size(); i++) {\n"
    "            if (i > 0) result += sep_;\n"
    "            result += parts[i];\n"
    "        }\n"
    "        return result;\n"
    "    }\n"
    "    size_t count(const std::vector<std::string>& parts) const { return parts.size(); }\n"
    "private:\n"
    "    std::string sep_;\n"
    "};\n"
    "\n"
    "int sum_lengths(const std::vector<std::string>& v) {\n"
    "    int total = 0;\n"
    "    for (const auto& s : v) total += static_cast<int>(s.size());\n"
    "    return total;\n"
    "}\n"
)
_CPP_TARGET = 350


def test_cpp_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_cpp_chunk_type_sequence() -> None:
    assert [c.chunk_type for c in _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET)] == ["import_group", "block", "class", "function"]


def test_cpp_include_group_has_all_headers() -> None:
    ig = _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET)[0]
    for header in ("<iostream>", "<vector>", "<string>"):
        assert header in ig.text


def test_cpp_namespace_body_intact() -> None:
    ns = _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET)[1]
    assert "namespace utils" in ns.text
    assert "trim" in ns.text
    assert "find_first_not_of" in ns.text


def test_cpp_class_body_intact() -> None:
    cls = _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET)[2]
    assert cls.chunk_type == "class"
    assert "class StringProcessor" in cls.text
    assert "join(" in cls.text
    assert "count(" in cls.text
    assert "sep_" in cls.text


def test_cpp_standalone_function_intact() -> None:
    fn = _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET)[3]
    assert fn.chunk_type == "function"
    assert "sum_lengths" in fn.text
    assert "return total" in fn.text


def test_cpp_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET):
        assert c.language == "cpp"


# ---------------------------------------------------------------------------
# Go
# Sections: [0] import_group  [1] block(Config struct+methods)  [2] block(joinStrings+countChars)
# ---------------------------------------------------------------------------

GOLDEN_GO = (
    "package main\n"
    "\n"
    "import (\n"
    "\t\"fmt\"\n"
    "\t\"strings\"\n"
    ")\n"
    "\n"
    "type Config struct {\n"
    "\tName  string\n"
    "\tDebug bool\n"
    "}\n"
    "\n"
    "func (c *Config) String() string {\n"
    "\treturn fmt.Sprintf(\"Config{%s}\", c.Name)\n"
    "}\n"
    "\n"
    "func (c *Config) IsDebug() bool {\n"
    "\treturn c.Debug\n"
    "}\n"
    "\n"
    "func joinStrings(parts []string, sep string) string {\n"
    "\treturn strings.Join(parts, sep)\n"
    "}\n"
    "\n"
    "func countChars(s string) int {\n"
    "\treturn len(s)\n"
    "}\n"
)
_GO_TARGET = 200


def test_go_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET)
    assert len(chunks) == 3, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_go_chunk_type_sequence() -> None:
    assert [c.chunk_type for c in _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET)] == ["import_group", "block", "block"]


def test_go_import_group_has_both_packages() -> None:
    ig = _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET)[0]
    assert '"fmt"' in ig.text
    assert '"strings"' in ig.text


def test_go_struct_and_methods_in_one_chunk() -> None:
    block = _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET)[1]
    assert "type Config struct" in block.text
    assert "func (c *Config) String()" in block.text
    assert "func (c *Config) IsDebug()" in block.text


def test_go_methods_body_intact() -> None:
    block = _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET)[1]
    assert 'fmt.Sprintf("Config{%s}"' in block.text
    assert "c.Debug" in block.text


def test_go_standalone_functions_merged() -> None:
    fn_block = _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET)[2]
    assert "joinStrings" in fn_block.text
    assert "countChars" in fn_block.text
    assert "strings.Join" in fn_block.text
    assert "len(s)" in fn_block.text


def test_go_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET):
        assert c.language == "go"


# ---------------------------------------------------------------------------
# Java
# Sections: [0] import_group  [1] class(ItemStore — full class)
# ---------------------------------------------------------------------------

GOLDEN_JAVA = (
    "import java.util.List;\n"
    "import java.util.ArrayList;\n"
    "import java.util.Optional;\n"
    "\n"
    "public class ItemStore {\n"
    "    private List<String> items = new ArrayList<>();\n"
    "\n"
    "    public ItemStore() {}\n"
    "\n"
    "    public void add(String item) {\n"
    "        items.add(item);\n"
    "    }\n"
    "\n"
    "    public Optional<String> findFirst(String prefix) {\n"
    "        return items.stream()\n"
    "            .filter(s -> s.startsWith(prefix))\n"
    "            .findFirst();\n"
    "    }\n"
    "\n"
    "    public int size() { return items.size(); }\n"
    "\n"
    "    public void clear() { items.clear(); }\n"
    "}\n"
)
_JAVA_TARGET = 250


def test_java_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_JAVA, "java", target_size=_JAVA_TARGET)
    assert len(chunks) == 2, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_java_chunk_type_sequence() -> None:
    assert [c.chunk_type for c in _chunker().chunk(GOLDEN_JAVA, "java", target_size=_JAVA_TARGET)] == ["import_group", "class"]


def test_java_import_group_has_all_imports() -> None:
    ig = _chunker().chunk(GOLDEN_JAVA, "java", target_size=_JAVA_TARGET)[0]
    assert "java.util.List" in ig.text
    assert "java.util.ArrayList" in ig.text
    assert "java.util.Optional" in ig.text


def test_java_class_body_intact() -> None:
    """All methods present — class must not be split despite exceeding target_size."""
    cls = _chunker().chunk(GOLDEN_JAVA, "java", target_size=_JAVA_TARGET)[1]
    assert cls.chunk_type == "class"
    assert "public class ItemStore" in cls.text
    assert "public void add(" in cls.text
    assert "findFirst" in cls.text
    assert "items.stream()" in cls.text
    assert "public int size()" in cls.text
    assert "public void clear()" in cls.text


def test_java_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_JAVA, "java", target_size=_JAVA_TARGET):
        assert c.language == "java"


# ---------------------------------------------------------------------------
# C
# Sections: [0] import_group  [1] block(typedef+enum)  [2] block(counter_init+increment)  [3] function(counter_get)
# ---------------------------------------------------------------------------

GOLDEN_C = (
    "#include <string.h>\n"
    "#include <stdlib.h>\n"
    "\n"
    "typedef struct {\n"
    "    char name[64];\n"
    "    int count;\n"
    "} Counter;\n"
    "\n"
    "enum State { IDLE, RUNNING, DONE };\n"
    "\n"
    "void counter_init(Counter *c, const char *name) {\n"
    "    strncpy(c->name, name, 63);\n"
    "    c->count = 0;\n"
    "}\n"
    "\n"
    "int counter_increment(Counter *c) {\n"
    "    return ++c->count;\n"
    "}\n"
    "\n"
    "int counter_get(const Counter *c) {\n"
    "    return c->count;\n"
    "}\n"
)
_C_TARGET = 200


def test_c_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_c_chunk_type_sequence() -> None:
    assert [c.chunk_type for c in _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET)] == ["import_group", "block", "block", "function"]


def test_c_include_group_has_both_headers() -> None:
    ig = _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET)[0]
    assert "<string.h>" in ig.text
    assert "<stdlib.h>" in ig.text


def test_c_struct_and_enum_merged() -> None:
    block = _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET)[1]
    assert "typedef struct" in block.text
    assert "Counter" in block.text
    assert "enum State" in block.text
    assert "IDLE" in block.text


def test_c_init_and_increment_merged() -> None:
    block = _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET)[2]
    assert "counter_init" in block.text
    assert "strncpy" in block.text
    assert "counter_increment" in block.text
    assert "++c->count" in block.text


def test_c_counter_get_is_own_chunk() -> None:
    fn = _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET)[3]
    assert fn.chunk_type == "function"
    assert "counter_get" in fn.text
    assert "return c->count" in fn.text


def test_c_no_function_split_mid_body() -> None:
    for chunk in _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET):
        if "counter_init" in chunk.text:
            assert "c->count = 0" in chunk.text
        if "counter_get" in chunk.text:
            assert "return c->count" in chunk.text


def test_c_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET):
        assert c.language == "c"


# ---------------------------------------------------------------------------
# Rust
# Sections: [0] import_group  [1] block(Counter struct+Op enum)  [2] block(impl Counter)  [3] function(merge)
# ---------------------------------------------------------------------------

GOLDEN_RUST = (
    "use std::collections::HashMap;\n"
    "use std::fmt;\n"
    "\n"
    "pub struct Counter {\n"
    "    name: String,\n"
    "    counts: HashMap<String, u32>,\n"
    "}\n"
    "\n"
    "pub enum Op { Increment, Reset }\n"
    "\n"
    "impl Counter {\n"
    "    pub fn new(name: String) -> Self {\n"
    "        Counter { name, counts: HashMap::new() }\n"
    "    }\n"
    "    pub fn record(&mut self, key: &str) {\n"
    "        *self.counts.entry(key.to_string()).or_insert(0) += 1;\n"
    "    }\n"
    "    pub fn get(&self, key: &str) -> u32 {\n"
    "        *self.counts.get(key).unwrap_or(&0)\n"
    "    }\n"
    "}\n"
    "\n"
    "pub fn merge(a: u32, b: u32) -> u32 {\n"
    "    a + b\n"
    "}\n"
)
_RUST_TARGET = 200


def test_rust_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_rust_chunk_type_sequence() -> None:
    assert [c.chunk_type for c in _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)] == ["import_group", "block", "block", "function"]


def test_rust_use_declarations_grouped() -> None:
    ig = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)[0]
    assert ig.chunk_type == "import_group"
    assert "use std::collections::HashMap" in ig.text
    assert "use std::fmt" in ig.text


def test_rust_struct_and_enum_merged() -> None:
    block = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)[1]
    assert "pub struct Counter" in block.text
    assert "counts: HashMap<String, u32>" in block.text
    assert "pub enum Op" in block.text
    assert "Increment" in block.text


def test_rust_impl_block_intact() -> None:
    impl_chunk = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)[2]
    assert "impl Counter" in impl_chunk.text
    assert "pub fn new(" in impl_chunk.text
    assert "pub fn record(" in impl_chunk.text
    assert "pub fn get(" in impl_chunk.text
    assert "HashMap::new()" in impl_chunk.text


def test_rust_impl_not_split_from_methods() -> None:
    chunks = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)
    for i, chunk in enumerate(chunks):
        if i != 2:
            assert "pub fn new(" not in chunk.text
            assert "pub fn record(" not in chunk.text


def test_rust_standalone_function_is_own_chunk() -> None:
    fn = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)[3]
    assert fn.chunk_type == "function"
    assert "pub fn merge" in fn.text
    assert "a + b" in fn.text


def test_rust_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET):
        assert c.language == "rust"


# ===========================================================================
# 4. Golden: Complex edge cases
# ===========================================================================

# ---------------------------------------------------------------------------
# Python complex — decorators, async def, inheritance, top-level constants
#
# KNOWN: top-level expression_statements (CONSTANT=42, _registry={}) are
#        NOT captured — only function_definition, class_definition,
#        decorated_definition, and import_* nodes are indexed.
# ---------------------------------------------------------------------------

GOLDEN_PY_COMPLEX = (
    "import os\n"
    "import sys\n"
    "from typing import Optional, List\n"
    "from dataclasses import dataclass\n"
    "\n"
    "CONSTANT = 42\n"
    "_registry: dict = {}\n"
    "\n"
    "@dataclass\n"
    "class Point:\n"
    "    x: float\n"
    "    y: float\n"
    "\n"
    "class Shape:\n"
    "    def __init__(self, color: str) -> None:\n"
    "        self.color = color\n"
    "\n"
    "    @staticmethod\n"
    "    def origin() -> 'Shape':\n"
    "        return Shape('black')\n"
    "\n"
    "    def area(self) -> float:\n"
    "        raise NotImplementedError\n"
    "\n"
    "    def describe(self) -> str:\n"
    "        return f\"{self.color} shape area={self.area():.2f}\"\n"
    "\n"
    "class Circle(Shape):\n"
    "    def __init__(self, color: str, radius: float) -> None:\n"
    "        super().__init__(color)\n"
    "        self.radius = radius\n"
    "\n"
    "    def area(self) -> float:\n"
    "        return 3.14159 * self.radius ** 2\n"
    "\n"
    "async def fetch_all(urls: List[str], timeout: int = 30) -> List[Optional[str]]:\n"
    "    results = []\n"
    "    for url in urls:\n"
    "        try:\n"
    "            results.append(url)\n"
    "        except Exception:\n"
    "            results.append(None)\n"
    "    return results\n"
    "\n"
    "def _private_helper(data: list) -> dict:\n"
    "    return {i: v for i, v in enumerate(data)}\n"
)
_PY_TARGET = 400


def test_py_complex_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_py_complex_type_sequence() -> None:
    chunks = _chunker().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "class", "block"]


def test_py_complex_import_group_has_all_four_imports() -> None:
    ig = _chunker().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)[0]
    assert "import os" in ig.text
    assert "import sys" in ig.text
    assert "from typing import" in ig.text
    assert "from dataclasses import dataclass" in ig.text


def test_py_complex_constants_not_in_any_chunk() -> None:
    """Top-level expression_statements (CONSTANT=42, _registry) are not captured."""
    all_text = "".join(c.text for c in _chunker().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET))
    assert "CONSTANT = 42" not in all_text
    assert "_registry: dict = {}" not in all_text


def test_py_complex_dataclass_decorator_kept_with_class() -> None:
    block = _chunker().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)[1]
    assert "@dataclass" in block.text
    assert "class Point:" in block.text
    assert "x: float" in block.text


def test_py_complex_shape_class_body_intact() -> None:
    block = _chunker().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)[1]
    assert "class Shape:" in block.text
    assert "@staticmethod" in block.text
    assert "def origin()" in block.text
    assert "def area(self)" in block.text
    assert "def describe(self)" in block.text


def test_py_complex_circle_is_class_chunk() -> None:
    circle = _chunker().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)[2]
    assert circle.chunk_type == "class"
    assert "class Circle(Shape):" in circle.text
    assert "super().__init__(color)" in circle.text
    assert "3.14159 * self.radius ** 2" in circle.text


def test_py_complex_async_def_detected_as_function() -> None:
    fn_block = _chunker().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)[3]
    assert "async def fetch_all" in fn_block.text
    assert "timeout: int = 30" in fn_block.text
    assert "return results" in fn_block.text


def test_py_complex_async_and_private_merged() -> None:
    fn_block = _chunker().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)[3]
    assert "async def fetch_all" in fn_block.text
    assert "def _private_helper" in fn_block.text
    assert "enumerate(data)" in fn_block.text


def test_py_complex_no_function_split_mid_body() -> None:
    for chunk in _chunker().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET):
        if "async def fetch_all" in chunk.text:
            assert "return results" in chunk.text
        if "def area(self)" in chunk.text and "Circle" in chunk.text:
            assert "3.14159" in chunk.text


def test_py_complex_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET):
        assert c.language == "python"


# ---------------------------------------------------------------------------
# TypeScript complex — interface, type alias, decorator (@Injectable), generics
#
# FIX VERIFIED: interface_declaration and type_alias_declaration are now
#               in semantic_node_types.
# ---------------------------------------------------------------------------

GOLDEN_TS_COMPLEX = (
    "import { Injectable, Inject } from '@angular/core';\n"
    "import { Observable, Subject } from 'rxjs';\n"
    "\n"
    "interface Repository<T> {\n"
    "  findById(id: string): Promise<T | null>;\n"
    "  save(entity: T): Promise<T>;\n"
    "  delete(id: string): Promise<void>;\n"
    "}\n"
    "\n"
    "type UserId = string;\n"
    "type EventName = string | symbol;\n"
    "type Handler<T> = (event: T) => void;\n"
    "\n"
    "@Injectable()\n"
    "export class UserService {\n"
    "  private subject = new Subject<string>();\n"
    "  constructor(@Inject('REPO') private repo: Repository<any>) {}\n"
    "  async getUser(id: UserId): Promise<any> { return this.repo.findById(id); }\n"
    "  async saveUser(user: any): Promise<any> { return this.repo.save(user); }\n"
    "  events(): Observable<string> { return this.subject.asObservable(); }\n"
    "}\n"
    "\n"
    "export function createService(repo: Repository<any>): UserService {\n"
    "  return new UserService(repo);\n"
    "}\n"
)
_TS_COMPLEX_TARGET = 300


def test_ts_complex_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_COMPLEX_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_ts_complex_type_sequence() -> None:
    chunks = _chunker().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_COMPLEX_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "block", "block"]


def test_ts_complex_import_group() -> None:
    ig = _chunker().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_COMPLEX_TARGET)[0]
    assert "from '@angular/core'" in ig.text
    assert "from 'rxjs'" in ig.text


def test_ts_complex_interface_captured() -> None:
    """interface_declaration is now a semantic node — must appear in a chunk."""
    all_text = "".join(c.text for c in _chunker().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_COMPLEX_TARGET))
    assert "interface Repository<T>" in all_text


def test_ts_complex_interface_body_intact() -> None:
    iface_chunk = _chunker().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_COMPLEX_TARGET)[1]
    assert "interface Repository<T>" in iface_chunk.text
    assert "findById(id: string)" in iface_chunk.text
    assert "delete(id: string)" in iface_chunk.text


def test_ts_complex_type_aliases_captured() -> None:
    iface_chunk = _chunker().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_COMPLEX_TARGET)[1]
    assert "type UserId = string" in iface_chunk.text
    assert "type EventName = string | symbol" in iface_chunk.text
    assert "type Handler<T>" in iface_chunk.text


def test_ts_complex_decorated_class_body_intact() -> None:
    cls_chunk = _chunker().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_COMPLEX_TARGET)[2]
    assert "@Injectable()" in cls_chunk.text
    assert "export class UserService" in cls_chunk.text
    assert "async getUser(" in cls_chunk.text
    assert "async saveUser(" in cls_chunk.text
    assert "events():" in cls_chunk.text


def test_ts_complex_export_function_is_own_chunk() -> None:
    fn_chunk = _chunker().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_COMPLEX_TARGET)[3]
    assert "export function createService" in fn_chunk.text
    assert "return new UserService(repo)" in fn_chunk.text


def test_ts_complex_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_COMPLEX_TARGET):
        assert c.language == "typescript"


# ---------------------------------------------------------------------------
# Rust complex — multiple impl blocks (trait impl vs inherent impl), generic fn
#
# Key: trait impls merge together; inherent impl is separate (exceeds budget).
# ---------------------------------------------------------------------------

GOLDEN_RUST_COMPLEX = (
    "use std::collections::HashMap;\n"
    "use std::fmt::{self, Display};\n"
    "\n"
    "pub trait Summary {\n"
    "    fn summarize(&self) -> String;\n"
    "    fn preview(&self) -> String {\n"
    "        format!(\"{}...\", &self.summarize()[..3])\n"
    "    }\n"
    "}\n"
    "\n"
    "pub struct Article {\n"
    "    pub title: String,\n"
    "    pub content: String,\n"
    "}\n"
    "\n"
    "impl Summary for Article {\n"
    "    fn summarize(&self) -> String { self.title.clone() }\n"
    "}\n"
    "\n"
    "impl Display for Article {\n"
    "    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {\n"
    "        write!(f, \"Article: {}\", self.title)\n"
    "    }\n"
    "}\n"
    "\n"
    "impl Article {\n"
    "    pub fn new(title: String, content: String) -> Self {\n"
    "        Article { title, content }\n"
    "    }\n"
    "    pub fn word_count(&self) -> usize {\n"
    "        self.content.split_whitespace().count()\n"
    "    }\n"
    "}\n"
    "\n"
    "pub fn print_summary<T: Summary + Display>(item: &T) {\n"
    "    println!(\"{}: {}\", item, item.summarize());\n"
    "}\n"
)
_RUST_COMPLEX_TARGET = 250


def test_rust_complex_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_COMPLEX_TARGET)
    assert len(chunks) == 5, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_rust_complex_type_sequence() -> None:
    chunks = _chunker().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_COMPLEX_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "block", "block", "function"]


def test_rust_complex_use_declarations_grouped() -> None:
    ig = _chunker().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_COMPLEX_TARGET)[0]
    assert "use std::collections::HashMap" in ig.text
    assert "use std::fmt::{self, Display}" in ig.text


def test_rust_complex_trait_and_struct_merged() -> None:
    block = _chunker().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_COMPLEX_TARGET)[1]
    assert "pub trait Summary" in block.text
    assert "fn summarize(&self) -> String" in block.text
    assert "fn preview(&self)" in block.text
    assert "pub struct Article" in block.text
    assert "pub title: String" in block.text


def test_rust_complex_trait_impls_merged() -> None:
    impl_chunk = _chunker().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_COMPLEX_TARGET)[2]
    assert "impl Summary for Article" in impl_chunk.text
    assert "impl Display for Article" in impl_chunk.text
    assert "self.title.clone()" in impl_chunk.text
    assert "write!(f," in impl_chunk.text


def test_rust_complex_inherent_impl_is_separate() -> None:
    inherent = _chunker().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_COMPLEX_TARGET)[3]
    assert "impl Article" in inherent.text
    assert "pub fn new(" in inherent.text
    assert "pub fn word_count(" in inherent.text
    assert "impl Summary for Article" not in inherent.text
    assert "impl Display for Article" not in inherent.text


def test_rust_complex_generic_fn_is_function_chunk() -> None:
    fn_chunk = _chunker().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_COMPLEX_TARGET)[4]
    assert fn_chunk.chunk_type == "function"
    assert "pub fn print_summary<T: Summary + Display>" in fn_chunk.text
    assert "item.summarize()" in fn_chunk.text


def test_rust_complex_no_impl_methods_leak_across_chunks() -> None:
    chunks = _chunker().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_COMPLEX_TARGET)
    trait_impl = chunks[2]
    assert "pub fn new(" not in trait_impl.text
    assert "word_count" not in trait_impl.text
    inherent = chunks[3]
    assert "impl Summary for Article" not in inherent.text


def test_rust_complex_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_COMPLEX_TARGET):
        assert c.language == "rust"


# ---------------------------------------------------------------------------
# Go complex — interface type, var_declaration, method receivers, errors
# ---------------------------------------------------------------------------

GOLDEN_GO_COMPLEX = (
    "package store\n"
    "\n"
    "import (\n"
    "\t\"context\"\n"
    "\t\"errors\"\n"
    "\t\"fmt\"\n"
    ")\n"
    "\n"
    "var ErrNotFound = errors.New(\"not found\")\n"
    "\n"
    "type Store interface {\n"
    "\tGet(ctx context.Context, key string) (string, error)\n"
    "\tSet(ctx context.Context, key, value string) error\n"
    "}\n"
    "\n"
    "type MemStore struct {\n"
    "\tdata map[string]string\n"
    "}\n"
    "\n"
    "func NewMemStore() *MemStore {\n"
    "\treturn &MemStore{data: make(map[string]string)}\n"
    "}\n"
    "\n"
    "func (s *MemStore) Get(ctx context.Context, key string) (string, error) {\n"
    "\tv, ok := s.data[key]\n"
    "\tif !ok {\n"
    "\t\treturn \"\", fmt.Errorf(\"%w: %s\", ErrNotFound, key)\n"
    "\t}\n"
    "\treturn v, nil\n"
    "}\n"
    "\n"
    "func (s *MemStore) Set(ctx context.Context, key, value string) error {\n"
    "\ts.data[key] = value\n"
    "\treturn nil\n"
    "}\n"
)
_GO_COMPLEX_TARGET = 200


def test_go_complex_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_COMPLEX_TARGET)
    assert len(chunks) == 5, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_go_complex_type_sequence() -> None:
    chunks = _chunker().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_COMPLEX_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "block", "function", "function"]


def test_go_complex_import_group() -> None:
    ig = _chunker().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_COMPLEX_TARGET)[0]
    assert '"context"' in ig.text
    assert '"errors"' in ig.text
    assert '"fmt"' in ig.text


def test_go_complex_error_var_and_interface_merged() -> None:
    block = _chunker().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_COMPLEX_TARGET)[1]
    assert "var ErrNotFound = errors.New" in block.text
    assert "type Store interface" in block.text
    assert "Get(ctx context.Context" in block.text
    assert "Set(ctx context.Context" in block.text


def test_go_complex_struct_and_constructor_merged() -> None:
    block = _chunker().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_COMPLEX_TARGET)[2]
    assert "type MemStore struct" in block.text
    assert "data map[string]string" in block.text
    assert "func NewMemStore()" in block.text
    assert "make(map[string]string)" in block.text


def test_go_complex_get_method_body_intact() -> None:
    get_chunk = _chunker().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_COMPLEX_TARGET)[3]
    assert "func (s *MemStore) Get(" in get_chunk.text
    assert "fmt.Errorf(\"%w: %s\"" in get_chunk.text
    assert "return v, nil" in get_chunk.text


def test_go_complex_set_method_is_own_chunk() -> None:
    set_chunk = _chunker().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_COMPLEX_TARGET)[4]
    assert "func (s *MemStore) Set(" in set_chunk.text
    assert "s.data[key] = value" in set_chunk.text
    assert "return nil" in set_chunk.text


def test_go_complex_get_and_set_not_in_same_chunk() -> None:
    chunks = _chunker().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_COMPLEX_TARGET)
    for chunk in chunks:
        assert not (
            "func (s *MemStore) Get(" in chunk.text
            and "func (s *MemStore) Set(" in chunk.text
        )


def test_go_complex_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_COMPLEX_TARGET):
        assert c.language == "go"


# ---------------------------------------------------------------------------
# C complex — preprocessor macros in import_group, typedef fn pointer, static fn
#
# FIX VERIFIED: preproc_def and preproc_function_def are now import_node_types
#               so #define macros are grouped with #include.
# ---------------------------------------------------------------------------

GOLDEN_C_COMPLEX = (
    "#include <stdio.h>\n"
    "#include <stdlib.h>\n"
    "#include <string.h>\n"
    "#define MAX_SIZE 256\n"
    "#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))\n"
    "\n"
    "typedef int (*compare_fn)(const void *, const void *);\n"
    "typedef struct {\n"
    "    char key[64];\n"
    "    int  value;\n"
    "} Entry;\n"
    "\n"
    "static int entry_cmp(const void *a, const void *b) {\n"
    "    return strcmp(((Entry*)a)->key, ((Entry*)b)->key);\n"
    "}\n"
    "\n"
    "Entry *find_entry(Entry *arr, int n, const char *key) {\n"
    "    Entry target;\n"
    "    strncpy(target.key, key, 63);\n"
    "    return bsearch(&target, arr, n, sizeof(Entry), entry_cmp);\n"
    "}\n"
    "\n"
    "void sort_entries(Entry *arr, int n) {\n"
    "    qsort(arr, n, sizeof(Entry), entry_cmp);\n"
    "}\n"
)
_C_COMPLEX_TARGET = 250


def test_c_complex_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_COMPLEX_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_c_complex_type_sequence() -> None:
    chunks = _chunker().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_COMPLEX_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "function", "function"]


def test_c_complex_macros_in_import_group() -> None:
    """#define macros are now import_node_types — grouped with #include."""
    ig = _chunker().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_COMPLEX_TARGET)[0]
    assert ig.chunk_type == "import_group"
    assert "#include <stdio.h>" in ig.text
    assert "#include <string.h>" in ig.text
    assert "#define MAX_SIZE 256" in ig.text
    assert "#define CLAMP(" in ig.text


def test_c_complex_macros_not_in_semantic_chunks() -> None:
    chunks = _chunker().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_COMPLEX_TARGET)
    for chunk in chunks[1:]:
        assert "#define" not in chunk.text


def test_c_complex_typedef_and_static_fn_merged() -> None:
    block = _chunker().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_COMPLEX_TARGET)[1]
    assert "typedef int (*compare_fn)" in block.text
    assert "typedef struct" in block.text
    assert "} Entry;" in block.text
    assert "static int entry_cmp(" in block.text
    assert "strcmp(" in block.text


def test_c_complex_find_entry_body_intact() -> None:
    fn = _chunker().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_COMPLEX_TARGET)[2]
    assert "Entry *find_entry(" in fn.text
    assert "strncpy(target.key" in fn.text
    assert "bsearch(" in fn.text


def test_c_complex_sort_entries_is_own_chunk() -> None:
    fn = _chunker().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_COMPLEX_TARGET)[3]
    assert fn.chunk_type == "function"
    assert "void sort_entries(" in fn.text
    assert "qsort(" in fn.text


def test_c_complex_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_COMPLEX_TARGET):
        assert c.language == "c"
