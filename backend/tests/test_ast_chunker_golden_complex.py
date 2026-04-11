"""Complex golden tests for the AST chunker — real-world edge cases (CS-101).

These tests exercise scenarios that the simple fixtures miss:
  - Python: decorators (@dataclass, @staticmethod), async def, inheritance,
            top-level constants (not captured — documented as known behavior)
  - TypeScript: generics, interface, type alias, decorator (@Injectable),
                all correctly captured after fix
  - Rust: multiple impl blocks (trait impl + inherent impl as separate spans),
          generic functions with trait bounds
  - Go: interface type, var_declaration, method receiver, multiple return values
  - C: preprocessor macros (#define) grouped with #include in import_group,
       typedef function pointer, static functions

Each fixture was run through the chunker with real output captured and pinned.
"""
from __future__ import annotations

from domain.retrieval.chunker_ast import ASTChunker


def _c() -> ASTChunker:
    return ASTChunker()


# ===========================================================================
# Python — complex
#
# Spans collected (target=400):
#   import_group : import os/sys/typing/dataclasses      (lines  0-3)
#   decorated_definition(@dataclass Point) + class Shape → merged block (8-25)
#   class Circle(Shape)                                  (27-33) → class chunk
#   async def fetch_all + def _private_helper → merged block (35-45)
#
# KNOWN: top-level expression_statements (CONSTANT=42, _registry={}) are
#        NOT captured by the AST chunker — they are expression_statement nodes
#        which are neither semantic nor import types. Files with only top-level
#        assignments fall back to flat chunking.
# ===========================================================================

GOLDEN_PY_COMPLEX = (
    "import os\n"                                                               # 0
    "import sys\n"                                                              # 1
    "from typing import Optional, List\n"                                       # 2
    "from dataclasses import dataclass\n"                                       # 3
    "\n"                                                                        # 4
    "CONSTANT = 42\n"                                                           # 5
    "_registry: dict = {}\n"                                                    # 6
    "\n"                                                                        # 7
    "@dataclass\n"                                                              # 8
    "class Point:\n"                                                            # 9
    "    x: float\n"                                                            # 10
    "    y: float\n"                                                            # 11
    "\n"                                                                        # 12
    "class Shape:\n"                                                            # 13
    "    def __init__(self, color: str) -> None:\n"                            # 14
    "        self.color = color\n"                                              # 15
    "\n"                                                                        # 16
    "    @staticmethod\n"                                                       # 17
    "    def origin() -> 'Shape':\n"                                            # 18
    "        return Shape('black')\n"                                           # 19
    "\n"                                                                        # 20
    "    def area(self) -> float:\n"                                            # 21
    "        raise NotImplementedError\n"                                       # 22
    "\n"                                                                        # 23
    "    def describe(self) -> str:\n"                                          # 24
    "        return f\"{self.color} shape area={self.area():.2f}\"\n"          # 25
    "\n"                                                                        # 26
    "class Circle(Shape):\n"                                                    # 27
    "    def __init__(self, color: str, radius: float) -> None:\n"             # 28
    "        super().__init__(color)\n"                                         # 29
    "        self.radius = radius\n"                                            # 30
    "\n"                                                                        # 31
    "    def area(self) -> float:\n"                                            # 32
    "        return 3.14159 * self.radius ** 2\n"                               # 33
    "\n"                                                                        # 34
    "async def fetch_all(urls: List[str], timeout: int = 30) -> List[Optional[str]]:\n"  # 35
    "    results = []\n"                                                        # 36
    "    for url in urls:\n"                                                    # 37
    "        try:\n"                                                            # 38
    "            results.append(url)\n"                                         # 39
    "        except Exception:\n"                                               # 40
    "            results.append(None)\n"                                        # 41
    "    return results\n"                                                      # 42
    "\n"                                                                        # 43
    "def _private_helper(data: list) -> dict:\n"                               # 44
    "    return {i: v for i, v in enumerate(data)}\n"                          # 45
)
_PY_TARGET = 400


def test_py_complex_chunk_count() -> None:
    chunks = _c().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_py_complex_type_sequence() -> None:
    chunks = _c().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "class", "block"]


def test_py_complex_import_group_has_all_four_imports() -> None:
    chunks = _c().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)
    ig = chunks[0]
    assert "import os" in ig.text
    assert "import sys" in ig.text
    assert "from typing import" in ig.text
    assert "from dataclasses import dataclass" in ig.text


def test_py_complex_constants_not_in_any_chunk() -> None:
    """Top-level expression_statements (CONSTANT=42, _registry) are not captured.

    This is known AST chunker behavior: only function_definition, class_definition,
    decorated_definition, and import_* node types are indexed.
    Files with ONLY top-level assignments would fall back to flat chunking.
    """
    chunks = _c().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)
    all_text = "".join(c.text for c in chunks)
    assert "CONSTANT = 42" not in all_text
    assert "_registry: dict = {}" not in all_text


def test_py_complex_dataclass_decorator_kept_with_class() -> None:
    """@dataclass decorator must stay with its class — not split across chunks."""
    chunks = _c().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)
    block = chunks[1]
    assert "@dataclass" in block.text
    assert "class Point:" in block.text
    assert "x: float" in block.text


def test_py_complex_shape_class_body_intact() -> None:
    """Shape class including @staticmethod method must all be in the same chunk."""
    chunks = _c().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)
    block = chunks[1]
    assert "class Shape:" in block.text
    assert "@staticmethod" in block.text
    assert "def origin()" in block.text
    assert "def area(self)" in block.text
    assert "def describe(self)" in block.text


def test_py_complex_circle_is_class_chunk() -> None:
    """Circle(Shape) is large enough to get its own chunk with type 'class'."""
    chunks = _c().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)
    circle = chunks[2]
    assert circle.chunk_type == "class"
    assert "class Circle(Shape):" in circle.text
    assert "super().__init__(color)" in circle.text
    assert "3.14159 * self.radius ** 2" in circle.text


def test_py_complex_async_def_detected_as_function() -> None:
    """async def is captured by function_definition node type in tree-sitter-python >=0.25."""
    chunks = _c().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)
    fn_block = chunks[3]
    assert "async def fetch_all" in fn_block.text
    assert "timeout: int = 30" in fn_block.text
    assert "return results" in fn_block.text


def test_py_complex_async_and_private_merged() -> None:
    """fetch_all + _private_helper are both small enough to share one chunk."""
    chunks = _c().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET)
    fn_block = chunks[3]
    assert "async def fetch_all" in fn_block.text
    assert "def _private_helper" in fn_block.text
    assert "enumerate(data)" in fn_block.text


def test_py_complex_no_function_split_mid_body() -> None:
    for chunk in _c().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET):
        if "async def fetch_all" in chunk.text:
            assert "return results" in chunk.text, "fetch_all body was split"
        if "def area(self)" in chunk.text and "Circle" in chunk.text:
            assert "3.14159" in chunk.text, "Circle.area body was split"


def test_py_complex_language_tag() -> None:
    for c in _c().chunk(GOLDEN_PY_COMPLEX, "python", target_size=_PY_TARGET):
        assert c.language == "python"


# ===========================================================================
# TypeScript — complex (interface, type alias, decorator, generics)
#
# Spans collected (target=300):
#   import_group        : 2 import statements              (lines  0-1)
#   block               : interface Repository<T>          merged with
#                         type UserId, EventName, Handler  (lines  3-11)
#   block               : @Injectable() export class       (lines 13-20)
#   block               : export function createService    (lines 22-24)
#
# FIX VERIFIED: interface_declaration and type_alias_declaration are now
#               in semantic_node_types — previously [skip], now captured.
# ===========================================================================

GOLDEN_TS_COMPLEX = (
    "import { Injectable, Inject } from '@angular/core';\n"     # 0
    "import { Observable, Subject } from 'rxjs';\n"             # 1
    "\n"                                                         # 2
    "interface Repository<T> {\n"                               # 3
    "  findById(id: string): Promise<T | null>;\n"              # 4
    "  save(entity: T): Promise<T>;\n"                          # 5
    "  delete(id: string): Promise<void>;\n"                    # 6
    "}\n"                                                        # 7
    "\n"                                                         # 8
    "type UserId = string;\n"                                   # 9
    "type EventName = string | symbol;\n"                       # 10
    "type Handler<T> = (event: T) => void;\n"                   # 11
    "\n"                                                         # 12
    "@Injectable()\n"                                           # 13
    "export class UserService {\n"                              # 14
    "  private subject = new Subject<string>();\n"              # 15
    "  constructor(@Inject('REPO') private repo: Repository<any>) {}\n"  # 16
    "  async getUser(id: UserId): Promise<any> { return this.repo.findById(id); }\n"  # 17
    "  async saveUser(user: any): Promise<any> { return this.repo.save(user); }\n"    # 18
    "  events(): Observable<string> { return this.subject.asObservable(); }\n"        # 19
    "}\n"                                                        # 20
    "\n"                                                         # 21
    "export function createService(repo: Repository<any>): UserService {\n"   # 22
    "  return new UserService(repo);\n"                         # 23
    "}\n"                                                        # 24
)
_TS_TARGET = 300


def test_ts_complex_chunk_count() -> None:
    chunks = _c().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_ts_complex_type_sequence() -> None:
    chunks = _c().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "block", "block"]


def test_ts_complex_import_group() -> None:
    chunks = _c().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_TARGET)
    ig = chunks[0]
    assert "from '@angular/core'" in ig.text
    assert "from 'rxjs'" in ig.text


def test_ts_complex_interface_captured() -> None:
    """interface_declaration is now a semantic node — must appear in a chunk."""
    chunks = _c().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_TARGET)
    all_text = "".join(c.text for c in chunks)
    assert "interface Repository<T>" in all_text, (
        "interface_declaration was not captured — check semantic_node_types for typescript"
    )


def test_ts_complex_interface_body_intact() -> None:
    chunks = _c().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_TARGET)
    iface_chunk = chunks[1]
    assert "interface Repository<T>" in iface_chunk.text
    assert "findById(id: string)" in iface_chunk.text
    assert "delete(id: string)" in iface_chunk.text


def test_ts_complex_type_aliases_captured() -> None:
    """type_alias_declaration nodes are captured and merged with the interface chunk."""
    chunks = _c().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_TARGET)
    iface_chunk = chunks[1]
    assert "type UserId = string" in iface_chunk.text
    assert "type EventName = string | symbol" in iface_chunk.text
    assert "type Handler<T>" in iface_chunk.text


def test_ts_complex_decorated_class_body_intact() -> None:
    """@Injectable() decorator must stay attached to UserService class."""
    chunks = _c().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_TARGET)
    cls_chunk = chunks[2]
    assert "@Injectable()" in cls_chunk.text
    assert "export class UserService" in cls_chunk.text
    assert "async getUser(" in cls_chunk.text
    assert "async saveUser(" in cls_chunk.text
    assert "events():" in cls_chunk.text


def test_ts_complex_export_function_is_own_chunk() -> None:
    chunks = _c().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_TARGET)
    fn_chunk = chunks[3]
    assert "export function createService" in fn_chunk.text
    assert "return new UserService(repo)" in fn_chunk.text


def test_ts_complex_language_tag() -> None:
    for c in _c().chunk(GOLDEN_TS_COMPLEX, "typescript", target_size=_TS_TARGET):
        assert c.language == "typescript"


# ===========================================================================
# Rust — complex (multiple impl blocks, trait impl, generic fn)
#
# Spans collected (target=250):
#   import_group : 2 use declarations                     (lines  0-1)
#   block        : pub trait Summary + pub struct Article  (lines  3-11)
#   block        : impl Summary for Article
#                  + impl Display for Article merged       (lines 13-21)
#   block        : impl Article (inherent impl) alone      (lines 23-26)
#   function     : pub fn print_summary<T: ...>            (lines 28-30)
#
# Key: different impl blocks are separate spans — trait impls merge together,
#      inherent impl is separate because adding it would exceed target_size.
# ===========================================================================

GOLDEN_RUST_COMPLEX = (
    "use std::collections::HashMap;\n"                                          # 0
    "use std::fmt::{self, Display};\n"                                          # 1
    "\n"                                                                        # 2
    "pub trait Summary {\n"                                                     # 3
    "    fn summarize(&self) -> String;\n"                                      # 4
    "    fn preview(&self) -> String {\n"                                       # 5
    "        format!(\"{}...\", &self.summarize()[..3])\n"                     # 6
    "    }\n"                                                                   # 7
    "}\n"                                                                       # 8
    "\n"                                                                        # 9
    "pub struct Article {\n"                                                    # 10
    "    pub title: String,\n"                                                  # 11
    "    pub content: String,\n"                                                # 12
    "}\n"                                                                       # 13
    "\n"                                                                        # 14
    "impl Summary for Article {\n"                                              # 15
    "    fn summarize(&self) -> String { self.title.clone() }\n"               # 16
    "}\n"                                                                       # 17
    "\n"                                                                        # 18
    "impl Display for Article {\n"                                              # 19
    "    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {\n"             # 20
    "        write!(f, \"Article: {}\", self.title)\n"                         # 21
    "    }\n"                                                                   # 22
    "}\n"                                                                       # 23
    "\n"                                                                        # 24
    "impl Article {\n"                                                          # 25
    "    pub fn new(title: String, content: String) -> Self {\n"               # 26
    "        Article { title, content }\n"                                      # 27
    "    }\n"                                                                   # 28
    "    pub fn word_count(&self) -> usize {\n"                                # 29
    "        self.content.split_whitespace().count()\n"                        # 30
    "    }\n"                                                                   # 31
    "}\n"                                                                       # 32
    "\n"                                                                        # 33
    "pub fn print_summary<T: Summary + Display>(item: &T) {\n"                 # 34
    "    println!(\"{}: {}\", item, item.summarize());\n"                       # 35
    "}\n"                                                                       # 36
)
_RUST_TARGET = 250


def test_rust_complex_chunk_count() -> None:
    chunks = _c().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_TARGET)
    assert len(chunks) == 5, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_rust_complex_type_sequence() -> None:
    chunks = _c().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_TARGET)
    assert [c.chunk_type for c in chunks] == [
        "import_group", "block", "block", "block", "function"
    ]


def test_rust_complex_use_declarations_grouped() -> None:
    chunks = _c().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_TARGET)
    ig = chunks[0]
    assert "use std::collections::HashMap" in ig.text
    assert "use std::fmt::{self, Display}" in ig.text


def test_rust_complex_trait_and_struct_merged() -> None:
    """pub trait Summary + pub struct Article are both small — merged into one block."""
    chunks = _c().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_TARGET)
    block = chunks[1]
    assert "pub trait Summary" in block.text
    assert "fn summarize(&self) -> String" in block.text
    assert "fn preview(&self)" in block.text
    assert "pub struct Article" in block.text
    assert "pub title: String" in block.text


def test_rust_complex_trait_impls_merged() -> None:
    """impl Summary for Article and impl Display for Article merge into one block."""
    chunks = _c().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_TARGET)
    impl_chunk = chunks[2]
    assert "impl Summary for Article" in impl_chunk.text
    assert "impl Display for Article" in impl_chunk.text
    assert "self.title.clone()" in impl_chunk.text
    assert "write!(f," in impl_chunk.text


def test_rust_complex_inherent_impl_is_separate() -> None:
    """impl Article (inherent impl) exceeds budget after trait impls — separate chunk."""
    chunks = _c().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_TARGET)
    inherent = chunks[3]
    assert "impl Article" in inherent.text
    assert "pub fn new(" in inherent.text
    assert "pub fn word_count(" in inherent.text
    # Must NOT contain trait impl content
    assert "impl Summary for Article" not in inherent.text
    assert "impl Display for Article" not in inherent.text


def test_rust_complex_generic_fn_is_function_chunk() -> None:
    chunks = _c().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_TARGET)
    fn_chunk = chunks[4]
    assert fn_chunk.chunk_type == "function"
    assert "pub fn print_summary<T: Summary + Display>" in fn_chunk.text
    assert "item.summarize()" in fn_chunk.text


def test_rust_complex_no_impl_methods_leak_across_chunks() -> None:
    """Methods of inherent impl must not appear in trait impl chunk and vice versa."""
    chunks = _c().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_TARGET)
    trait_impl = chunks[2]
    assert "pub fn new(" not in trait_impl.text
    assert "word_count" not in trait_impl.text

    inherent = chunks[3]
    assert "impl Summary for Article" not in inherent.text


def test_rust_complex_language_tag() -> None:
    for c in _c().chunk(GOLDEN_RUST_COMPLEX, "rust", target_size=_RUST_TARGET):
        assert c.language == "rust"


# ===========================================================================
# Go — complex (interface type, var_declaration, method receivers, errors)
#
# Spans collected (target=200):
#   import_group : block import (context, errors, fmt)    (lines  2-6)
#   block        : var ErrNotFound + type Store interface  (lines  8-13)
#   block        : type MemStore struct + func NewMemStore (lines 15-21)
#   function     : func (s *MemStore) Get(...)            (lines 23-29)
#   function     : func (s *MemStore) Set(...)            (lines 31-34)
# ===========================================================================

GOLDEN_GO_COMPLEX = (
    "package store\n"                                                           # 0
    "\n"                                                                        # 1
    "import (\n"                                                                # 2
    "\t\"context\"\n"                                                           # 3
    "\t\"errors\"\n"                                                            # 4
    "\t\"fmt\"\n"                                                               # 5
    ")\n"                                                                       # 6
    "\n"                                                                        # 7
    "var ErrNotFound = errors.New(\"not found\")\n"                            # 8
    "\n"                                                                        # 9
    "type Store interface {\n"                                                  # 10
    "\tGet(ctx context.Context, key string) (string, error)\n"                 # 11
    "\tSet(ctx context.Context, key, value string) error\n"                    # 12
    "}\n"                                                                       # 13
    "\n"                                                                        # 14
    "type MemStore struct {\n"                                                  # 15
    "\tdata map[string]string\n"                                                # 16
    "}\n"                                                                       # 17
    "\n"                                                                        # 18
    "func NewMemStore() *MemStore {\n"                                          # 19
    "\treturn &MemStore{data: make(map[string]string)}\n"                       # 20
    "}\n"                                                                       # 21
    "\n"                                                                        # 22
    "func (s *MemStore) Get(ctx context.Context, key string) (string, error) {\n"  # 23
    "\tv, ok := s.data[key]\n"                                                  # 24
    "\tif !ok {\n"                                                              # 25
    "\t\treturn \"\", fmt.Errorf(\"%w: %s\", ErrNotFound, key)\n"              # 26
    "\t}\n"                                                                     # 27
    "\treturn v, nil\n"                                                         # 28
    "}\n"                                                                       # 29
    "\n"                                                                        # 30
    "func (s *MemStore) Set(ctx context.Context, key, value string) error {\n" # 31
    "\ts.data[key] = value\n"                                                   # 32
    "\treturn nil\n"                                                            # 33
    "}\n"                                                                       # 34
)
_GO_TARGET = 200


def test_go_complex_chunk_count() -> None:
    chunks = _c().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_TARGET)
    assert len(chunks) == 5, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_go_complex_type_sequence() -> None:
    chunks = _c().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_TARGET)
    assert [c.chunk_type for c in chunks] == [
        "import_group", "block", "block", "function", "function"
    ]


def test_go_complex_import_group() -> None:
    chunks = _c().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_TARGET)
    ig = chunks[0]
    assert '"context"' in ig.text
    assert '"errors"' in ig.text
    assert '"fmt"' in ig.text


def test_go_complex_error_var_and_interface_merged() -> None:
    """var ErrNotFound and type Store interface are small — merged into one block."""
    chunks = _c().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_TARGET)
    block = chunks[1]
    assert "var ErrNotFound = errors.New" in block.text
    assert "type Store interface" in block.text
    assert "Get(ctx context.Context" in block.text
    assert "Set(ctx context.Context" in block.text


def test_go_complex_struct_and_constructor_merged() -> None:
    """type MemStore struct + func NewMemStore() fit together under target_size."""
    chunks = _c().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_TARGET)
    block = chunks[2]
    assert "type MemStore struct" in block.text
    assert "data map[string]string" in block.text
    assert "func NewMemStore()" in block.text
    assert "make(map[string]string)" in block.text


def test_go_complex_get_method_body_intact() -> None:
    """Get method with multi-return and error wrapping must not be split."""
    chunks = _c().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_TARGET)
    get_chunk = chunks[3]
    assert "func (s *MemStore) Get(" in get_chunk.text
    assert "fmt.Errorf(\"%w: %s\"" in get_chunk.text
    assert "return v, nil" in get_chunk.text


def test_go_complex_set_method_is_own_chunk() -> None:
    chunks = _c().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_TARGET)
    set_chunk = chunks[4]
    assert "func (s *MemStore) Set(" in set_chunk.text
    assert "s.data[key] = value" in set_chunk.text
    assert "return nil" in set_chunk.text


def test_go_complex_get_and_set_not_in_same_chunk() -> None:
    """Get is large enough to prevent Set from merging in."""
    chunks = _c().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_TARGET)
    for chunk in chunks:
        assert not (
            "func (s *MemStore) Get(" in chunk.text
            and "func (s *MemStore) Set(" in chunk.text
        ), "Get and Set must be in separate chunks"


def test_go_complex_language_tag() -> None:
    for c in _c().chunk(GOLDEN_GO_COMPLEX, "go", target_size=_GO_TARGET):
        assert c.language == "go"


# ===========================================================================
# C — complex (preprocessor macros in import_group, typedef fn pointer,
#              static helper, multiple typedef structs)
#
# Spans collected (target=250):
#   import_group : #include x3 + #define x2 all grouped   (lines  0-5)
#   block        : typedef compare_fn + typedef Entry
#                  + static entry_cmp merged               (lines  6-14)
#   function     : Entry *find_entry(...)                  (lines 16-20)
#   function     : void sort_entries(...)                  (lines 22-24)
#
# FIX VERIFIED: preproc_def and preproc_function_def are now import_node_types
#               so #define macros are grouped with #include in import_group.
# ===========================================================================

GOLDEN_C_COMPLEX = (
    "#include <stdio.h>\n"                                                       # 0
    "#include <stdlib.h>\n"                                                      # 1
    "#include <string.h>\n"                                                      # 2
    "#define MAX_SIZE 256\n"                                                     # 3
    "#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))\n"  # 4
    "\n"                                                                         # 5
    "typedef int (*compare_fn)(const void *, const void *);\n"                  # 6
    "typedef struct {\n"                                                         # 7
    "    char key[64];\n"                                                        # 8
    "    int  value;\n"                                                          # 9
    "} Entry;\n"                                                                 # 10
    "\n"                                                                         # 11
    "static int entry_cmp(const void *a, const void *b) {\n"                   # 12
    "    return strcmp(((Entry*)a)->key, ((Entry*)b)->key);\n"                  # 13
    "}\n"                                                                        # 14
    "\n"                                                                         # 15
    "Entry *find_entry(Entry *arr, int n, const char *key) {\n"                 # 16
    "    Entry target;\n"                                                        # 17
    "    strncpy(target.key, key, 63);\n"                                       # 18
    "    return bsearch(&target, arr, n, sizeof(Entry), entry_cmp);\n"          # 19
    "}\n"                                                                        # 20
    "\n"                                                                         # 21
    "void sort_entries(Entry *arr, int n) {\n"                                  # 22
    "    qsort(arr, n, sizeof(Entry), entry_cmp);\n"                            # 23
    "}\n"                                                                        # 24
)
_C_TARGET = 250


def test_c_complex_chunk_count() -> None:
    chunks = _c().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_c_complex_type_sequence() -> None:
    chunks = _c().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_TARGET)
    assert [c.chunk_type for c in chunks] == [
        "import_group", "block", "function", "function"
    ]


def test_c_complex_macros_in_import_group() -> None:
    """#define macros are now import_node_types — grouped with #include."""
    chunks = _c().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_TARGET)
    ig = chunks[0]
    assert ig.chunk_type == "import_group"
    assert "#include <stdio.h>" in ig.text
    assert "#include <string.h>" in ig.text
    assert "#define MAX_SIZE 256" in ig.text, (
        "#define was not captured — check preproc_def in import_node_types for C"
    )
    assert "#define CLAMP(" in ig.text, (
        "#define CLAMP macro was not captured — check preproc_function_def"
    )


def test_c_complex_macros_not_in_semantic_chunks() -> None:
    """Macros belong only in import_group, never mixed into semantic chunks."""
    chunks = _c().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_TARGET)
    for chunk in chunks[1:]:
        assert "#define" not in chunk.text


def test_c_complex_typedef_and_static_fn_merged() -> None:
    """typedef compare_fn, typedef struct Entry, and static entry_cmp all merge."""
    chunks = _c().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_TARGET)
    block = chunks[1]
    assert "typedef int (*compare_fn)" in block.text
    assert "typedef struct" in block.text
    assert "} Entry;" in block.text
    assert "static int entry_cmp(" in block.text
    assert "strcmp(" in block.text


def test_c_complex_find_entry_body_intact() -> None:
    chunks = _c().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_TARGET)
    fn = chunks[2]
    assert "Entry *find_entry(" in fn.text
    assert "strncpy(target.key" in fn.text
    assert "bsearch(" in fn.text


def test_c_complex_sort_entries_is_own_chunk() -> None:
    chunks = _c().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_TARGET)
    fn = chunks[3]
    assert fn.chunk_type == "function"
    assert "void sort_entries(" in fn.text
    assert "qsort(" in fn.text


def test_c_complex_language_tag() -> None:
    for c in _c().chunk(GOLDEN_C_COMPLEX, "c", target_size=_C_TARGET):
        assert c.language == "c"
