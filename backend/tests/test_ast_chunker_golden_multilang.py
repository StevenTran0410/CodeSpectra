"""Golden tests for AST chunker — all supported languages except Python (CS-101).

Python is already covered in test_ast_chunker_golden.py.
Each fixture is a compact but realistic source file with clearly-labelled
logical sections. Chunk count, type sequence, and symbol-level assertions
are all pinned to verified actual chunker output.

Languages covered: TypeScript, JavaScript, C++, Go, Java, C, Rust.
"""
from __future__ import annotations

from domain.retrieval.chunker_ast import ASTChunker


def _chunker() -> ASTChunker:
    return ASTChunker()


# ---------------------------------------------------------------------------
# TypeScript
#
# Fixture sections:
#   [0] import_group  — 2 import statements
#   [1] block         — export class FileReader (export_statement wrapping class)
#   [2] block         — trimLines + countWords merged (both export_statements,
#                       combined ~210B < target 300)
# ---------------------------------------------------------------------------

GOLDEN_TS = (
    "import { EventEmitter } from 'events';\n"              # line 0
    "import { readFileSync } from 'fs';\n"                  # line 1
    "\n"                                                    # line 2
    "export class FileReader {\n"                           # line 3
    "  private path: string;\n"                             # line 4
    "  constructor(path: string) { this.path = path; }\n"  # line 5
    "  read(): string { return readFileSync(this.path, 'utf8'); }\n"  # line 6
    "  lines(): string[] { return this.read().split('\\n'); }\n"      # line 7
    "}\n"                                                   # line 8
    "\n"                                                    # line 9
    "export function trimLines(lines: string[]): string[] {\n"        # line 10
    "  return lines.map(l => l.trim()).filter(l => l.length > 0);\n"  # line 11
    "}\n"                                                   # line 12
    "\n"                                                    # line 13
    "export function countWords(text: string): number {\n"            # line 14
    "  return text.split(/\\s+/).filter(w => w.length > 0).length;\n" # line 15
    "}\n"                                                   # line 16
)
_TS_TARGET = 300


def test_ts_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_TS, "typescript", target_size=_TS_TARGET)
    assert len(chunks) == 3, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_ts_chunk_type_sequence() -> None:
    chunks = _chunker().chunk(GOLDEN_TS, "typescript", target_size=_TS_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "block"]


def test_ts_import_group_covers_both_imports() -> None:
    chunks = _chunker().chunk(GOLDEN_TS, "typescript", target_size=_TS_TARGET)
    ig = chunks[0]
    assert ig.chunk_type == "import_group"
    assert "from 'events'" in ig.text
    assert "from 'fs'" in ig.text


def test_ts_class_body_intact() -> None:
    """FileReader class must not be split — constructor, read, and lines all present."""
    chunks = _chunker().chunk(GOLDEN_TS, "typescript", target_size=_TS_TARGET)
    cls = chunks[1]
    assert "export class FileReader" in cls.text
    assert "constructor" in cls.text
    assert "read()" in cls.text
    assert "lines()" in cls.text


def test_ts_small_functions_merged() -> None:
    """trimLines and countWords are small enough to share one chunk."""
    chunks = _chunker().chunk(GOLDEN_TS, "typescript", target_size=_TS_TARGET)
    merged = chunks[2]
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
#
# Fixture sections:
#   [0] import_group  — 2 import statements
#   [1] block         — export class Hasher
#   [2] block         — export function parseConfig (alone, ~200B ≈ target 250)
#   [3] block         — export function formatConfig (alone)
# ---------------------------------------------------------------------------

GOLDEN_JS = (
    "import { createHash } from 'crypto';\n"                # line 0
    "import { readFile } from 'fs/promises';\n"             # line 1
    "\n"                                                    # line 2
    "export class Hasher {\n"                               # line 3
    "  constructor(algo) { this.algo = algo; }\n"           # line 4
    "  hash(data) { return createHash(this.algo).update(data).digest('hex'); }\n"  # line 5
    "  verify(data, expected) { return this.hash(data) === expected; }\n"          # line 6
    "}\n"                                                   # line 7
    "\n"                                                    # line 8
    "export function parseConfig(text) {\n"                 # line 9
    "  return Object.fromEntries(\n"                        # line 10
    "    text.split('\\n').filter(l => l.includes('='))\n" # line 11
    "         .map(l => l.split('='))\n"                    # line 12
    "  );\n"                                                # line 13
    "}\n"                                                   # line 14
    "\n"                                                    # line 15
    "export function formatConfig(obj) {\n"                 # line 16
    "  return Object.entries(obj)\n"                        # line 17
    "    .map(([k, v]) => k + '=' + v).join('\\n');\n"     # line 18
    "}\n"                                                   # line 19
)
_JS_TARGET = 250


def test_js_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_JS, "javascript", target_size=_JS_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_js_chunk_type_sequence() -> None:
    chunks = _chunker().chunk(GOLDEN_JS, "javascript", target_size=_JS_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "block", "block"]


def test_js_import_group_covers_both_imports() -> None:
    chunks = _chunker().chunk(GOLDEN_JS, "javascript", target_size=_JS_TARGET)
    ig = chunks[0]
    assert "from 'crypto'" in ig.text
    assert "from 'fs/promises'" in ig.text


def test_js_class_body_intact() -> None:
    chunks = _chunker().chunk(GOLDEN_JS, "javascript", target_size=_JS_TARGET)
    cls = chunks[1]
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
#
# Fixture sections:
#   [0] import_group  — 3 #include directives
#   [1] block         — namespace utils (namespace_definition → no class/fn match)
#   [2] class         — class StringProcessor
#   [3] function      — int sum_lengths(...)
# ---------------------------------------------------------------------------

GOLDEN_CPP = (
    "#include <iostream>\n"                                             # line 0
    "#include <vector>\n"                                               # line 1
    "#include <string>\n"                                               # line 2
    "\n"                                                                # line 3
    "namespace utils {\n"                                              # line 4
    "    std::string trim(const std::string& s) {\n"                   # line 5
    "        size_t start = s.find_first_not_of(\" \\t\");\n"          # line 6
    "        return (start == std::string::npos) ? \"\" : s.substr(start);\n"  # line 7
    "    }\n"                                                           # line 8
    "}\n"                                                              # line 9
    "\n"                                                               # line 10
    "class StringProcessor {\n"                                        # line 11
    "public:\n"                                                        # line 12
    "    StringProcessor(std::string sep) : sep_(sep) {}\n"           # line 13
    "    std::string join(const std::vector<std::string>& parts) const {\n"  # line 14
    "        std::string result;\n"                                    # line 15
    "        for (size_t i = 0; i < parts.size(); i++) {\n"           # line 16
    "            if (i > 0) result += sep_;\n"                        # line 17
    "            result += parts[i];\n"                               # line 18
    "        }\n"                                                      # line 19
    "        return result;\n"                                         # line 20
    "    }\n"                                                          # line 21
    "    size_t count(const std::vector<std::string>& parts) const { return parts.size(); }\n"  # line 22
    "private:\n"                                                       # line 23
    "    std::string sep_;\n"                                         # line 24
    "};\n"                                                             # line 25
    "\n"                                                               # line 26
    "int sum_lengths(const std::vector<std::string>& v) {\n"          # line 27
    "    int total = 0;\n"                                             # line 28
    "    for (const auto& s : v) total += static_cast<int>(s.size());\n"  # line 29
    "    return total;\n"                                              # line 30
    "}\n"                                                              # line 31
)
_CPP_TARGET = 350


def test_cpp_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_cpp_chunk_type_sequence() -> None:
    chunks = _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "class", "function"]


def test_cpp_include_group_has_all_headers() -> None:
    chunks = _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET)
    ig = chunks[0]
    for header in ("<iostream>", "<vector>", "<string>"):
        assert header in ig.text


def test_cpp_namespace_body_intact() -> None:
    chunks = _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET)
    ns = chunks[1]
    assert "namespace utils" in ns.text
    assert "trim" in ns.text
    assert "find_first_not_of" in ns.text


def test_cpp_class_body_intact() -> None:
    chunks = _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET)
    cls = chunks[2]
    assert cls.chunk_type == "class"
    assert "class StringProcessor" in cls.text
    assert "StringProcessor(std::string sep)" in cls.text
    assert "join(" in cls.text
    assert "count(" in cls.text
    assert "sep_" in cls.text


def test_cpp_standalone_function_intact() -> None:
    chunks = _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET)
    fn = chunks[3]
    assert fn.chunk_type == "function"
    assert "sum_lengths" in fn.text
    assert "return total" in fn.text


def test_cpp_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_CPP, "cpp", target_size=_CPP_TARGET):
        assert c.language == "cpp"


# ---------------------------------------------------------------------------
# Go
#
# Fixture sections:
#   [0] import_group  — block import declaration
#   [1] block         — type Config struct + methods String + IsDebug merged
#                       (type_declaration ~45B + method ~70B + method ~45B = ~160B < 200)
#   [2] block         — joinStrings + countChars merged (~100B < 200)
# ---------------------------------------------------------------------------

GOLDEN_GO = (
    "package main\n"                                                    # line 0
    "\n"                                                                # line 1
    "import (\n"                                                        # line 2
    "\t\"fmt\"\n"                                                       # line 3
    "\t\"strings\"\n"                                                   # line 4
    ")\n"                                                               # line 5
    "\n"                                                                # line 6
    "type Config struct {\n"                                            # line 7
    "\tName  string\n"                                                  # line 8
    "\tDebug bool\n"                                                    # line 9
    "}\n"                                                               # line 10
    "\n"                                                                # line 11
    "func (c *Config) String() string {\n"                             # line 12
    "\treturn fmt.Sprintf(\"Config{%s}\", c.Name)\n"                   # line 13
    "}\n"                                                               # line 14
    "\n"                                                                # line 15
    "func (c *Config) IsDebug() bool {\n"                              # line 16
    "\treturn c.Debug\n"                                                # line 17
    "}\n"                                                               # line 18
    "\n"                                                                # line 19
    "func joinStrings(parts []string, sep string) string {\n"          # line 20
    "\treturn strings.Join(parts, sep)\n"                              # line 21
    "}\n"                                                               # line 22
    "\n"                                                                # line 23
    "func countChars(s string) int {\n"                                # line 24
    "\treturn len(s)\n"                                                 # line 25
    "}\n"                                                               # line 26
)
_GO_TARGET = 200


def test_go_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET)
    assert len(chunks) == 3, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_go_chunk_type_sequence() -> None:
    chunks = _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "block"]


def test_go_import_group_has_both_packages() -> None:
    chunks = _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET)
    ig = chunks[0]
    assert '"fmt"' in ig.text
    assert '"strings"' in ig.text


def test_go_struct_and_methods_in_one_chunk() -> None:
    """Config struct + both methods are merged into one block chunk."""
    chunks = _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET)
    block = chunks[1]
    assert "type Config struct" in block.text
    assert "func (c *Config) String()" in block.text
    assert "func (c *Config) IsDebug()" in block.text


def test_go_methods_body_intact() -> None:
    chunks = _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET)
    block = chunks[1]
    assert 'fmt.Sprintf("Config{%s}"' in block.text
    assert "c.Debug" in block.text


def test_go_standalone_functions_merged() -> None:
    chunks = _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET)
    fn_block = chunks[2]
    assert "joinStrings" in fn_block.text
    assert "countChars" in fn_block.text
    assert "strings.Join" in fn_block.text
    assert "len(s)" in fn_block.text


def test_go_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_GO, "go", target_size=_GO_TARGET):
        assert c.language == "go"


# ---------------------------------------------------------------------------
# Java
#
# Fixture sections:
#   [0] import_group  — 3 import declarations
#   [1] class         — public class ItemStore (full class, single semantic unit)
# ---------------------------------------------------------------------------

GOLDEN_JAVA = (
    "import java.util.List;\n"                                          # line 0
    "import java.util.ArrayList;\n"                                     # line 1
    "import java.util.Optional;\n"                                      # line 2
    "\n"                                                                # line 3
    "public class ItemStore {\n"                                        # line 4
    "    private List<String> items = new ArrayList<>();\n"             # line 5
    "\n"                                                                # line 6
    "    public ItemStore() {}\n"                                       # line 7
    "\n"                                                                # line 8
    "    public void add(String item) {\n"                             # line 9
    "        items.add(item);\n"                                        # line 10
    "    }\n"                                                           # line 11
    "\n"                                                                # line 12
    "    public Optional<String> findFirst(String prefix) {\n"         # line 13
    "        return items.stream()\n"                                   # line 14
    "            .filter(s -> s.startsWith(prefix))\n"                 # line 15
    "            .findFirst();\n"                                      # line 16
    "    }\n"                                                           # line 17
    "\n"                                                                # line 18
    "    public int size() { return items.size(); }\n"                 # line 19
    "\n"                                                               # line 20
    "    public void clear() { items.clear(); }\n"                     # line 21
    "}\n"                                                              # line 22
)
_JAVA_TARGET = 250


def test_java_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_JAVA, "java", target_size=_JAVA_TARGET)
    assert len(chunks) == 2, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_java_chunk_type_sequence() -> None:
    chunks = _chunker().chunk(GOLDEN_JAVA, "java", target_size=_JAVA_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "class"]


def test_java_import_group_has_all_imports() -> None:
    chunks = _chunker().chunk(GOLDEN_JAVA, "java", target_size=_JAVA_TARGET)
    ig = chunks[0]
    assert "java.util.List" in ig.text
    assert "java.util.ArrayList" in ig.text
    assert "java.util.Optional" in ig.text


def test_java_class_body_intact() -> None:
    """All methods must be present — class must not be split despite exceeding target_size."""
    chunks = _chunker().chunk(GOLDEN_JAVA, "java", target_size=_JAVA_TARGET)
    cls = chunks[1]
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
#
# Fixture sections:
#   [0] import_group  — 2 #include directives
#   [1] block         — typedef struct Counter + enum State merged (~110B < 200)
#   [2] block         — counter_init + counter_increment merged (~140B < 200;
#                       adding counter_get would push to ~230B > 200)
#   [3] function      — counter_get alone
# ---------------------------------------------------------------------------

GOLDEN_C = (
    "#include <string.h>\n"                                             # line 0
    "#include <stdlib.h>\n"                                             # line 1
    "\n"                                                                # line 2
    "typedef struct {\n"                                                # line 3
    "    char name[64];\n"                                              # line 4
    "    int count;\n"                                                  # line 5
    "} Counter;\n"                                                      # line 6
    "\n"                                                                # line 7
    "enum State { IDLE, RUNNING, DONE };\n"                            # line 8
    "\n"                                                                # line 9
    "void counter_init(Counter *c, const char *name) {\n"              # line 10
    "    strncpy(c->name, name, 63);\n"                                # line 11
    "    c->count = 0;\n"                                               # line 12
    "}\n"                                                               # line 13
    "\n"                                                                # line 14
    "int counter_increment(Counter *c) {\n"                            # line 15
    "    return ++c->count;\n"                                          # line 16
    "}\n"                                                               # line 17
    "\n"                                                                # line 18
    "int counter_get(const Counter *c) {\n"                            # line 19
    "    return c->count;\n"                                            # line 20
    "}\n"                                                               # line 21
)
_C_TARGET = 200


def test_c_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_c_chunk_type_sequence() -> None:
    chunks = _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "block", "function"]


def test_c_include_group_has_both_headers() -> None:
    chunks = _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET)
    ig = chunks[0]
    assert "<string.h>" in ig.text
    assert "<stdlib.h>" in ig.text


def test_c_struct_and_enum_merged() -> None:
    """typedef struct Counter and enum State are both small — merged into one block."""
    chunks = _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET)
    block = chunks[1]
    assert "typedef struct" in block.text
    assert "Counter" in block.text
    assert "enum State" in block.text
    assert "IDLE" in block.text


def test_c_init_and_increment_merged() -> None:
    """counter_init and counter_increment fit together under target_size."""
    chunks = _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET)
    block = chunks[2]
    assert "counter_init" in block.text
    assert "strncpy" in block.text
    assert "counter_increment" in block.text
    assert "++c->count" in block.text


def test_c_counter_get_is_own_chunk() -> None:
    chunks = _chunker().chunk(GOLDEN_C, "c", target_size=_C_TARGET)
    fn = chunks[3]
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
#
# Fixture sections:
#   [0] import_group  — 2 use declarations
#   [1] block         — pub struct Counter + pub enum Op merged (~105B < 200)
#   [2] block         — impl Counter block (~250B alone; adding pub fn merge
#                       would push to ~300B > 200 → impl alone)
#   [3] function      — pub fn merge alone
# ---------------------------------------------------------------------------

GOLDEN_RUST = (
    "use std::collections::HashMap;\n"                                  # line 0
    "use std::fmt;\n"                                                   # line 1
    "\n"                                                                # line 2
    "pub struct Counter {\n"                                            # line 3
    "    name: String,\n"                                               # line 4
    "    counts: HashMap<String, u32>,\n"                              # line 5
    "}\n"                                                               # line 6
    "\n"                                                                # line 7
    "pub enum Op { Increment, Reset }\n"                               # line 8
    "\n"                                                                # line 9
    "impl Counter {\n"                                                  # line 10
    "    pub fn new(name: String) -> Self {\n"                         # line 11
    "        Counter { name, counts: HashMap::new() }\n"              # line 12
    "    }\n"                                                           # line 13
    "    pub fn record(&mut self, key: &str) {\n"                      # line 14
    "        *self.counts.entry(key.to_string()).or_insert(0) += 1;\n" # line 15
    "    }\n"                                                           # line 16
    "    pub fn get(&self, key: &str) -> u32 {\n"                      # line 17
    "        *self.counts.get(key).unwrap_or(&0)\n"                    # line 18
    "    }\n"                                                           # line 19
    "}\n"                                                               # line 20
    "\n"                                                                # line 21
    "pub fn merge(a: u32, b: u32) -> u32 {\n"                         # line 22
    "    a + b\n"                                                       # line 23
    "}\n"                                                               # line 24
)
_RUST_TARGET = 200


def test_rust_chunk_count() -> None:
    chunks = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)
    assert len(chunks) == 4, [(c.chunk_type, c.start_line, c.end_line) for c in chunks]


def test_rust_chunk_type_sequence() -> None:
    chunks = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)
    assert [c.chunk_type for c in chunks] == ["import_group", "block", "block", "function"]


def test_rust_use_declarations_grouped() -> None:
    chunks = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)
    ig = chunks[0]
    assert ig.chunk_type == "import_group"
    assert "use std::collections::HashMap" in ig.text
    assert "use std::fmt" in ig.text


def test_rust_struct_and_enum_merged() -> None:
    """pub struct Counter and pub enum Op are both small — merged into one block."""
    chunks = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)
    block = chunks[1]
    assert "pub struct Counter" in block.text
    assert "counts: HashMap<String, u32>" in block.text
    assert "pub enum Op" in block.text
    assert "Increment" in block.text


def test_rust_impl_block_intact() -> None:
    """impl Counter must stay together — all three methods present."""
    chunks = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)
    impl_chunk = chunks[2]
    assert "impl Counter" in impl_chunk.text
    assert "pub fn new(" in impl_chunk.text
    assert "pub fn record(" in impl_chunk.text
    assert "pub fn get(" in impl_chunk.text
    assert "HashMap::new()" in impl_chunk.text


def test_rust_impl_not_split_from_methods() -> None:
    """No method may appear outside the impl block chunk."""
    chunks = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)
    # Only chunk[2] should contain impl methods
    for i, chunk in enumerate(chunks):
        if i != 2:
            assert "pub fn new(" not in chunk.text
            assert "pub fn record(" not in chunk.text


def test_rust_standalone_function_is_own_chunk() -> None:
    chunks = _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET)
    fn = chunks[3]
    assert fn.chunk_type == "function"
    assert "pub fn merge" in fn.text
    assert "a + b" in fn.text


def test_rust_language_tag() -> None:
    for c in _chunker().chunk(GOLDEN_RUST, "rust", target_size=_RUST_TARGET):
        assert c.language == "rust"
