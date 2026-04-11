"""AST-based semantic chunker for code files (CS-101).

Replaces flat line-count chunking with syntax-aware chunking via Tree-sitter.
Functions, classes, and other logical units are never split mid-body.

Supported languages: python, typescript, javascript, cpp, go, java.
Falls back to flat chunking for unsupported languages or on any parse error.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native merge-pass hotspot (optional — graceful fallback to Python)
# ---------------------------------------------------------------------------

def _load_native_chunker() -> Any:
    try:
        return importlib.import_module("domain.retrieval._native_chunker")
    except Exception:
        return None


_native_chunker = _load_native_chunker()

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class ASTChunk:
    """A semantically coherent code chunk produced by the AST chunker."""

    text: str
    chunk_type: str              # 'function' | 'class' | 'import_group' | 'block' | 'file'
    node_names: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    language: str = ""


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LanguageConfig:
    get_language: Callable[[], Any]       # lazy factory returning tree_sitter.Language
    semantic_node_types: frozenset[str]
    import_node_types: frozenset[str]


def _make_python_lang() -> Any:
    import tree_sitter_python as tspython
    from tree_sitter import Language
    return Language(tspython.language())


def _make_typescript_lang() -> Any:
    import tree_sitter_typescript as tsts
    from tree_sitter import Language
    return Language(tsts.language_typescript())


def _make_javascript_lang() -> Any:
    import tree_sitter_javascript as tsjs
    from tree_sitter import Language
    return Language(tsjs.language())


def _make_cpp_lang() -> Any:
    import tree_sitter_cpp as tscpp
    from tree_sitter import Language
    return Language(tscpp.language())


def _make_go_lang() -> Any:
    import tree_sitter_go as tsgo
    from tree_sitter import Language
    return Language(tsgo.language())


def _make_java_lang() -> Any:
    import tree_sitter_java as tsjava
    from tree_sitter import Language
    return Language(tsjava.language())


def _make_c_lang() -> Any:
    import tree_sitter_c as tsc
    from tree_sitter import Language
    return Language(tsc.language())


def _make_rust_lang() -> Any:
    import tree_sitter_rust as tsrust
    from tree_sitter import Language
    return Language(tsrust.language())


LANGUAGE_CONFIGS: dict[str, LanguageConfig] = {
    "python": LanguageConfig(
        get_language=_make_python_lang,
        semantic_node_types=frozenset({
            "function_definition",
            "async_function_definition",
            "class_definition",
            "decorated_definition",
        }),
        import_node_types=frozenset({
            "import_statement",
            "import_from_statement",
        }),
    ),
    "typescript": LanguageConfig(
        get_language=_make_typescript_lang,
        semantic_node_types=frozenset({
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "method_definition",
            "export_statement",
            "lexical_declaration",
            "variable_declaration",
        }),
        import_node_types=frozenset({
            "import_statement",
        }),
    ),
    "javascript": LanguageConfig(
        get_language=_make_javascript_lang,
        semantic_node_types=frozenset({
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "method_definition",
            "export_statement",
            "lexical_declaration",
            "variable_declaration",
        }),
        import_node_types=frozenset({
            "import_statement",
        }),
    ),
    "cpp": LanguageConfig(
        get_language=_make_cpp_lang,
        semantic_node_types=frozenset({
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
            "template_declaration",
        }),
        import_node_types=frozenset({
            "preproc_include",
        }),
    ),
    "go": LanguageConfig(
        get_language=_make_go_lang,
        semantic_node_types=frozenset({
            "function_declaration",
            "method_declaration",
            "type_declaration",
            "var_declaration",
            "const_declaration",
        }),
        import_node_types=frozenset({
            "import_declaration",
        }),
    ),
    "java": LanguageConfig(
        get_language=_make_java_lang,
        semantic_node_types=frozenset({
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "annotation_type_declaration",
            "method_declaration",
            "constructor_declaration",
            "field_declaration",
        }),
        import_node_types=frozenset({
            "import_declaration",
        }),
    ),
    "c": LanguageConfig(
        get_language=_make_c_lang,
        semantic_node_types=frozenset({
            "function_definition",
            "struct_specifier",
            "enum_specifier",
            "union_specifier",
            "type_definition",
        }),
        import_node_types=frozenset({
            "preproc_include",
        }),
    ),
    "rust": LanguageConfig(
        get_language=_make_rust_lang,
        semantic_node_types=frozenset({
            "function_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "mod_item",
            "type_item",
        }),
        import_node_types=frozenset({
            "use_declaration",
        }),
    ),
}

# ---------------------------------------------------------------------------
# Internal span type
# ---------------------------------------------------------------------------

@dataclass
class _NodeSpan:
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    node_type: str
    name: str          # best-effort symbol name, empty if not determinable
    is_import: bool


# ---------------------------------------------------------------------------
# Parser cache
# ---------------------------------------------------------------------------

_parser_cache: dict[str, Any] = {}


def _get_parser(language: str) -> Any | None:
    """Return a cached tree_sitter.Parser for the given language, or None."""
    if language in _parser_cache:
        return _parser_cache[language]
    cfg = LANGUAGE_CONFIGS.get(language)
    if cfg is None:
        return None
    try:
        from tree_sitter import Parser
        lang = cfg.get_language()
        parser = Parser(lang)
        _parser_cache[language] = parser
        return parser
    except Exception as exc:
        logger.warning("[chunker_ast] failed to load parser for %s: %s", language, exc)
        _parser_cache[language] = None
        return None


# ---------------------------------------------------------------------------
# Node collection (top-down DFS)
# ---------------------------------------------------------------------------

def _extract_name(node: Any, src_bytes: bytes) -> str:
    """Extract the best-effort symbol name from an AST node."""
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier", "type_identifier"):
            try:
                return src_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            except Exception:
                return ""
    return ""


def _collect_nodes(
    root: Any,
    src_bytes: bytes,
    cfg: LanguageConfig,
    max_size: int,
) -> list[_NodeSpan]:
    """
    Top-down traversal over the AST using an explicit stack (no Python recursion).

    - Semantic nodes (functions, classes) that fit within max_size are emitted
      as a single span.
    - Semantic nodes larger than max_size are recursively split at their children.
    - Import nodes are collected together into a pending buffer, coalesced, and
      flushed as a single span when a non-import node is encountered.
    - All other nodes fall through to their children.

    Uses an iterative stack to avoid Python recursion limits and reduce
    per-call overhead compared to nested closures.
    """
    semantic_types = cfg.semantic_node_types
    import_types = cfg.import_node_types

    result: list[_NodeSpan] = []
    import_buffer: list[_NodeSpan] = []

    def flush_imports() -> None:
        if import_buffer:
            result.extend(import_buffer)
            import_buffer.clear()

    # Iterative DFS using an explicit stack.
    # Use named_children (skips punctuation/whitespace tokens) for ~5x speedup
    # on the root-level children access vs .children.
    # The stack starts with root's named children in order.
    stack: list[Any] = list(reversed(root.named_children))

    while stack:
        node = stack.pop()
        node_type = node.type

        if node_type in import_types:
            import_buffer.append(_NodeSpan(
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                node_type="import_group",
                name="",
                is_import=True,
            ))
            continue

        if node_type in semantic_types:
            flush_imports()
            size = node.end_byte - node.start_byte
            if size <= max_size:
                result.append(_NodeSpan(
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    node_type=node_type,
                    name=_extract_name(node, src_bytes),
                    is_import=False,
                ))
                continue
            # Node too large — push named children in reverse to maintain order.
            stack.extend(reversed(node.named_children))
            continue

        # Non-semantic node — flush imports and descend into named children.
        flush_imports()
        stack.extend(reversed(node.named_children))

    flush_imports()

    # Coalesce adjacent import_group spans into a single merged span so they
    # travel through the merge pass as one unit and don't bleed into function groups.
    coalesced: list[_NodeSpan] = []
    pending_imports: list[_NodeSpan] = []

    def flush_import_spans() -> None:
        if not pending_imports:
            return
        merged = _NodeSpan(
            start_byte=pending_imports[0].start_byte,
            end_byte=pending_imports[-1].end_byte,
            start_line=pending_imports[0].start_line,
            end_line=pending_imports[-1].end_line,
            node_type="import_group",
            name="",
            is_import=True,
        )
        coalesced.append(merged)
        pending_imports.clear()

    for span in result:
        if span.is_import:
            pending_imports.append(span)
        else:
            flush_import_spans()
            coalesced.append(span)
    flush_import_spans()

    return coalesced


# ---------------------------------------------------------------------------
# Merge pass
# ---------------------------------------------------------------------------

def _merge_spans_python(spans: list[_NodeSpan], target_size: int) -> list[list[_NodeSpan]]:
    """
    Pure Python fallback merge: greedy accumulation.

    Rules:
    - Import group spans are never merged with non-import spans (imports keep
      their own group to preserve chunk_type='import_group' semantics).
    - Non-import spans are greedily accumulated while their byte range fits
      within target_size.
    """
    if not spans:
        return []
    groups: list[list[_NodeSpan]] = []
    current: list[_NodeSpan] = []
    group_start = 0

    def flush() -> None:
        if current:
            groups.append(list(current))
            current.clear()

    for span in spans:
        if span.is_import:
            # Imports always get their own group — flush any in-progress group first.
            flush()
            groups.append([span])
            continue
        # Non-import span: attempt to merge into current group.
        if not current:
            current.append(span)
            group_start = span.start_byte
        else:
            candidate_len = span.end_byte - group_start
            if candidate_len > target_size:
                flush()
                current.append(span)
                group_start = span.start_byte
            else:
                current.append(span)

    flush()
    return groups


def _merge_spans(spans: list[_NodeSpan], target_size: int) -> list[list[_NodeSpan]]:
    """
    Merge spans into groups.

    Import spans are always emitted as their own group (never merged with
    non-import spans). Non-import segments are merged greedily up to target_size,
    using the native C++ hotspot when available.
    """
    if not spans:
        return []

    # Split the span list at import boundaries, preserving order.
    # Each segment is either a single import span or a run of non-import spans.
    segments: list[list[_NodeSpan]] = []
    current_seg: list[_NodeSpan] = []

    for span in spans:
        if span.is_import:
            if current_seg:
                segments.append(current_seg)
                current_seg = []
            segments.append([span])  # import always its own segment
        else:
            current_seg.append(span)

    if current_seg:
        segments.append(current_seg)

    result: list[list[_NodeSpan]] = []
    for seg in segments:
        if not seg:
            continue
        if seg[0].is_import:
            # Import segment: emit as-is (already a single coalesced span).
            result.append(seg)
            continue
        # Non-import segment: merge with native or Python fallback.
        if _native_chunker is not None:
            try:
                # Build local index within this segment.
                triples = [(s.start_byte, s.end_byte, i) for i, s in enumerate(seg)]
                raw_groups = _native_chunker.merge_spans(triples, target_size)
                for group_indices in raw_groups:
                    result.append([seg[idx] for idx in group_indices])
                continue
            except Exception as exc:
                logger.debug(
                    "[chunker_ast] native merge_spans failed, using Python fallback: %s", exc
                )
        # Python fallback for this segment.
        result.extend(_merge_spans_python(seg, target_size))

    return result


# ---------------------------------------------------------------------------
# Flat chunking fallback (mirrors _split_chunks in service.py)
# ---------------------------------------------------------------------------

def _flat_chunks(source: str, target_size: int) -> list[ASTChunk]:
    """Flat sliding-window fallback for unsupported languages or parse failures."""
    clean = source.replace("\r\n", "\n")
    if len(clean) <= target_size:
        return [ASTChunk(text=clean, chunk_type="file", start_line=0,
                         end_line=clean.count("\n"), language="")]
    overlap = max(120, target_size // 8)
    result: list[ASTChunk] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + target_size)
        piece = clean[start:end]
        result.append(ASTChunk(text=piece, chunk_type="block",
                                start_line=clean[:start].count("\n"),
                                end_line=clean[:end].count("\n"),
                                language=""))
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return result


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class ASTChunker:
    """
    Produces semantically coherent code chunks from source text.

    Usage:
        chunker = ASTChunker()
        chunks = chunker.chunk(source, language="python", target_size=1500)
    """

    # Maximum size of a single node before we recursively split its children.
    _MAX_NODE_SIZE = 2000

    def chunk(
        self,
        source: str,
        language: str,
        target_size: int = 1500,
    ) -> list[ASTChunk]:
        """
        Return AST-based chunks for supported languages.
        Falls back to flat chunking for unsupported languages or on any error.
        """
        lang_key = language.lower() if language else ""

        if lang_key not in LANGUAGE_CONFIGS:
            return _flat_chunks(source, target_size)

        # Short-circuit: file fits in a single chunk — skip AST overhead.
        if len(source) <= target_size:
            line_count = source.count("\n")
            return [ASTChunk(
                text=source,
                chunk_type="file",
                start_line=0,
                end_line=line_count,
                language=lang_key,
            )]

        parser = _get_parser(lang_key)
        if parser is None:
            return _flat_chunks(source, target_size)

        try:
            return self._parse_and_chunk(source, lang_key, parser, target_size)
        except Exception as exc:
            logger.warning(
                "[chunker_ast] parse/chunk failed for language=%s, falling back to flat: %s",
                lang_key, exc,
            )
            return _flat_chunks(source, target_size)

    def _parse_and_chunk(
        self,
        source: str,
        language: str,
        parser: Any,
        target_size: int,
    ) -> list[ASTChunk]:
        cfg = LANGUAGE_CONFIGS[language]
        src_bytes = source.encode("utf-8", errors="replace")
        tree = parser.parse(src_bytes)
        root = tree.root_node

        spans = _collect_nodes(root, src_bytes, cfg, self._MAX_NODE_SIZE)

        if not spans:
            # No semantic nodes found (e.g. script with only top-level expressions).
            return _flat_chunks(source, target_size)

        groups = _merge_spans(spans, target_size)

        chunks: list[ASTChunk] = []
        for group in groups:
            first = group[0]
            last = group[-1]
            text = src_bytes[first.start_byte:last.end_byte].decode("utf-8", errors="replace")

            # Determine dominant chunk_type for the group.
            types_in_group = [s.node_type for s in group]
            if all(t == "import_group" for t in types_in_group):
                chunk_type = "import_group"
            elif len(group) == 1:
                t = group[0].node_type
                if "function" in t or "method" in t:
                    chunk_type = "function"
                elif "class" in t or "struct" in t or "interface" in t:
                    chunk_type = "class"
                else:
                    chunk_type = "block"
            else:
                chunk_type = "block"

            names = [s.name for s in group if s.name]
            chunks.append(ASTChunk(
                text=text,
                chunk_type=chunk_type,
                node_names=names,
                start_line=first.start_line,
                end_line=last.end_line,
                language=language,
            ))

        return chunks
