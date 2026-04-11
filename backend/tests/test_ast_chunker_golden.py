"""Golden tests for the AST chunker (CS-101).

Predefined Python source with clearly labelled logical sections.
Each test asserts exact chunk boundaries (count, type, content coverage).
C++ native merge_spans is also exercised directly and compared against
the pure-Python fallback to guarantee identical output.
"""
from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

import domain.retrieval.chunker_ast as _mod
from domain.retrieval.chunker_ast import ASTChunker

# ---------------------------------------------------------------------------
# Golden fixture — byte-stable Python source with 4 distinct logical sections
#
# Section map (0-indexed lines):
#   [0-3]   import group  — 4 import statements
#   [6-11]  functions     — add() + subtract(), small enough to merge
#   [14-30] function      — long_function(), large enough to stand alone
#   [33-42] class         — DataProcessor with 3 methods
#
# Total ≈ 901 bytes; well above any target_size used in these tests.
# ---------------------------------------------------------------------------

GOLDEN_PYTHON = (
    "import os\n"                                           # line 0
    "import sys\n"                                          # line 1
    "from pathlib import Path\n"                            # line 2
    "from typing import Any\n"                              # line 3
    "\n"                                                    # line 4
    "\n"                                                    # line 5
    "def add(x: int, y: int) -> int:\n"                    # line 6
    "    return x + y\n"                                    # line 7
    "\n"                                                    # line 8
    "\n"                                                    # line 9
    "def subtract(x: int, y: int) -> int:\n"               # line 10
    "    return x - y\n"                                    # line 11
    "\n"                                                    # line 12
    "\n"                                                    # line 13
    "def long_function(data: list) -> dict:\n"             # line 14
    "    # intentionally long to force its own chunk\n"    # line 15
    "    if not data:\n"                                    # line 16
    "        return {'count': 0, 'total': 0}\n"            # line 17
    "    total = sum(data)\n"                               # line 18
    "    count = len(data)\n"                               # line 19
    "    mean = total / count\n"                            # line 20
    "    items = sorted(data)\n"                            # line 21
    "    minimum = items[0]\n"                              # line 22
    "    maximum = items[-1]\n"                             # line 23
    "    return {\n"                                        # line 24
    "        'count': count,\n"                             # line 25
    "        'total': total,\n"                             # line 26
    "        'mean': mean,\n"                               # line 27
    "        'min': minimum,\n"                             # line 28
    "        'max': maximum,\n"                             # line 29
    "    }\n"                                               # line 30
    "\n"                                                    # line 31
    "\n"                                                    # line 32
    "class DataProcessor:\n"                               # line 33
    "    def __init__(self, label: str) -> None:\n"        # line 34
    "        self.label = label\n"                          # line 35
    "        self.records: list = []\n"                    # line 36
    "\n"                                                    # line 37
    "    def push(self, record: Any) -> None:\n"           # line 38
    "        self.records.append(record)\n"                 # line 39
    "\n"                                                    # line 40
    "    def summarize(self) -> dict:\n"                    # line 41
    "        return long_function(self.records)\n"          # line 42
)

# target_size chosen so that:
#   add (~51 B) + subtract (~56 B) combined (~109 B) fit → 1 merged "block" chunk
#   long_function (~435 B) alone exceeds remaining room → its own "function" chunk
#   DataProcessor (~282 B) alone exceeds remaining room → its own "class" chunk
_TARGET = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunker() -> ASTChunker:
    return ASTChunker()


def _native_available() -> bool:
    try:
        importlib.import_module("domain.retrieval._native_chunker")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Golden — chunk count and ordering
# ---------------------------------------------------------------------------

def test_golden_chunk_count() -> None:
    """GOLDEN_PYTHON with target_size=200 must yield exactly 4 chunks."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    assert len(chunks) == 4, (
        f"Expected 4 chunks, got {len(chunks)}: "
        + str([(c.chunk_type, c.start_line, c.end_line) for c in chunks])
    )


def test_golden_chunk_type_sequence() -> None:
    """Chunks must appear in the order: import_group → block → function → class."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    types = [c.chunk_type for c in chunks]
    assert types == ["import_group", "block", "function", "class"], types


# ---------------------------------------------------------------------------
# Golden — section 1: imports
# ---------------------------------------------------------------------------

def test_golden_import_group_contains_all_imports() -> None:
    """The import_group chunk must cover all four import statements."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    import_chunk = chunks[0]
    assert import_chunk.chunk_type == "import_group"
    for expected in ("import os", "import sys", "from pathlib import Path", "from typing import Any"):
        assert expected in import_chunk.text, (
            f"Expected '{expected}' in import_group, got: {import_chunk.text!r}"
        )


def test_golden_import_chunk_starts_at_line_zero() -> None:
    """Import group must begin on line 0 (first line of the file)."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    assert chunks[0].start_line == 0


# ---------------------------------------------------------------------------
# Golden — section 2: merged small functions (add + subtract)
# ---------------------------------------------------------------------------

def test_golden_small_functions_merged_into_one_chunk() -> None:
    """add() and subtract() are small enough to share one 'block' chunk."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    block = chunks[1]
    assert block.chunk_type == "block"
    assert "def add" in block.text
    assert "return x + y" in block.text
    assert "def subtract" in block.text
    assert "return x - y" in block.text


def test_golden_block_chunk_not_split_mid_function() -> None:
    """Neither add nor subtract may appear without its return statement."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    # Verify no chunk has 'def add' without 'return x + y'
    for chunk in chunks:
        if "def add" in chunk.text:
            assert "return x + y" in chunk.text, (
                f"'def add' found without its return in chunk: {chunk.text!r}"
            )
        if "def subtract" in chunk.text:
            assert "return x - y" in chunk.text, (
                f"'def subtract' found without its return in chunk: {chunk.text!r}"
            )


# ---------------------------------------------------------------------------
# Golden — section 3: long_function alone
# ---------------------------------------------------------------------------

def test_golden_long_function_is_its_own_chunk() -> None:
    """long_function must appear in exactly one 'function' chunk, never shared."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    func_chunks = [c for c in chunks if "def long_function" in c.text]
    assert len(func_chunks) == 1, f"Expected 1 chunk with long_function, got {len(func_chunks)}"
    fc = func_chunks[0]
    assert fc.chunk_type == "function"


def test_golden_long_function_body_intact() -> None:
    """long_function's opening line and final return must be in the same chunk."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    fc = next(c for c in chunks if "def long_function" in c.text)
    assert "def long_function" in fc.text
    assert "'max': maximum" in fc.text, (
        "Last key of the return dict missing — function was split mid-body"
    )


def test_golden_long_function_not_in_import_or_class_chunk() -> None:
    """long_function must not bleed into the import_group or class chunks."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    assert "def long_function" not in chunks[0].text  # import chunk
    assert "def long_function" not in chunks[3].text  # class chunk


# ---------------------------------------------------------------------------
# Golden — section 4: DataProcessor class
# ---------------------------------------------------------------------------

def test_golden_class_is_its_own_chunk() -> None:
    """DataProcessor must be in exactly one 'class' chunk."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    class_chunks = [c for c in chunks if "class DataProcessor" in c.text]
    assert len(class_chunks) == 1
    assert class_chunks[0].chunk_type == "class"


def test_golden_class_chunk_includes_all_methods() -> None:
    """All three methods (__init__, push, summarize) must be in the class chunk."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    cc = next(c for c in chunks if "class DataProcessor" in c.text)
    for method in ("def __init__", "def push", "def summarize"):
        assert method in cc.text, f"Method '{method}' missing from class chunk"


# ---------------------------------------------------------------------------
# Golden — chunk language tag
# ---------------------------------------------------------------------------

def test_golden_all_chunks_carry_language_tag() -> None:
    """Every chunk must report language='python'."""
    chunks = _chunker().chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    for c in chunks:
        assert c.language == "python", f"Missing language tag on chunk: {c.chunk_type}"


# ---------------------------------------------------------------------------
# C++ native: merge_spans direct API
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _native_available(), reason="C++ native chunker not built")
def test_native_merge_spans_groups_correctly() -> None:
    """merge_spans must group spans greedily matching the Python fallback logic.

    Spans derived from the golden fixture byte offsets (approx):
      index 0 (add):        start=71,  end=122  size=51
      index 1 (subtract):   start=124, end=180  size=56
      index 2 (long_func):  start=182, end=617  size=435
      index 3 (DataProc):   start=619, end=901  size=282

    With target_size=200:
      - 0+1: candidate=180-71=109 ≤ 200 → merge
      - +2:  candidate=617-71=546 > 200 → flush [0,1], start new
      - +3:  candidate=901-182=719 > 200 → flush [2], start new
      → groups: [[0,1], [2], [3]]
    """
    native = importlib.import_module("domain.retrieval._native_chunker")
    triples = [
        (71, 122, 0),    # add
        (124, 180, 1),   # subtract
        (182, 617, 2),   # long_function
        (619, 901, 3),   # DataProcessor
    ]
    groups = native.merge_spans(triples, target_size=200)
    groups_as_lists = [list(g) for g in groups]

    assert groups_as_lists == [[0, 1], [2], [3]], (
        f"Unexpected native grouping: {groups_as_lists}"
    )


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
    triples = [(0, 60, 0), (60, 120, 1), (120, 180, 2)]
    groups = [list(g) for g in native.merge_spans(triples, target_size=200)]
    assert groups == [[0, 1, 2]], f"All spans fit → 1 group, got {groups}"


@pytest.mark.skipif(not _native_available(), reason="C++ native chunker not built")
def test_native_merge_spans_each_span_exceeds_target() -> None:
    """Each span alone exceeds target_size — each becomes its own group."""
    native = importlib.import_module("domain.retrieval._native_chunker")
    triples = [(0, 300, 0), (300, 600, 1), (600, 900, 2)]
    groups = [list(g) for g in native.merge_spans(triples, target_size=200)]
    assert groups == [[0], [1], [2]], f"Each span alone → 3 groups, got {groups}"


# ---------------------------------------------------------------------------
# C++ native: full ASTChunker output matches Python fallback
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _native_available(), reason="C++ native chunker not built")
def test_native_chunker_matches_python_fallback_on_golden() -> None:
    """ASTChunker with C++ native must produce chunk-for-chunk identical results
    to the pure-Python fallback on the golden Python fixture.

    This guards against any future divergence between the two merge paths.
    """
    native_ref = _mod._native_chunker
    assert native_ref is not None, "Native module loaded but _native_chunker is None"

    chunker = ASTChunker()

    # Run with native C++ merge pass active
    chunks_native = chunker.chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)

    # Swap native out → force Python fallback
    _mod._native_chunker = None
    try:
        chunks_python = chunker.chunk(GOLDEN_PYTHON, "python", target_size=_TARGET)
    finally:
        _mod._native_chunker = native_ref  # always restore

    assert len(chunks_native) == len(chunks_python), (
        f"Native returned {len(chunks_native)} chunks, Python {len(chunks_python)}"
    )

    for i, (cn, cp) in enumerate(zip(chunks_native, chunks_python)):
        assert cn.chunk_type == cp.chunk_type, (
            f"Chunk {i}: native type={cn.chunk_type!r}, python type={cp.chunk_type!r}"
        )
        assert cn.text == cp.text, (
            f"Chunk {i} ({cn.chunk_type}): text differs between native and Python paths"
        )
        assert cn.start_line == cp.start_line, (
            f"Chunk {i}: native start_line={cn.start_line}, python={cp.start_line}"
        )
        assert cn.end_line == cp.end_line, (
            f"Chunk {i}: native end_line={cn.end_line}, python={cp.end_line}"
        )
