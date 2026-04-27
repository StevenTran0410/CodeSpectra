"""RetrievalQuality computation — cheap heuristics for retrieval signal assessment."""

import math
import re
from collections import Counter

from .types import RankedChunk, RetrievalQuality

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
    "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that",
    "the", "to", "was", "will", "with", "how", "what", "where", "which", "who", "why"
}

_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_DEF_PATTERN = re.compile(r"\b(def |class |function )\w")


def _tokenize_for_coverage(text: str) -> set[str]:
    """Extract lowercased tokens, excluding stopwords."""
    tokens = _WORD.findall(text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _compute_coverage(query_terms: list[str], ranked_chunks: list[RankedChunk]) -> float:
    """Fraction of non-stopword query tokens appearing in any chunk's content."""
    if not query_terms:
        return 1.0

    query_tokens = {t.lower() for t in query_terms if t.lower() not in _STOPWORDS and len(t) > 1}
    if not query_tokens:
        return 1.0

    all_content_tokens: set[str] = set()
    for chunk in ranked_chunks:
        all_content_tokens.update(_tokenize_for_coverage(chunk.excerpt))

    hits = len(query_tokens & all_content_tokens)
    return hits / len(query_tokens)


def _compute_path_entropy(ranked_chunks: list[RankedChunk]) -> float:
    """Shannon entropy H = -Σ p·log2(p) of top-level directory distribution."""
    if not ranked_chunks:
        return 0.0

    top_dirs: list[str] = []
    for chunk in ranked_chunks:
        parts = chunk.rel_path.split("/")
        top_dirs.append(parts[0] if parts else "")

    if not top_dirs:
        return 0.0

    counts = Counter(top_dirs)
    total = len(top_dirs)

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def _has_definition(ranked_chunks: list[RankedChunk]) -> bool:
    """Check if any chunk is a function/class/method or content matches def/class/function pattern."""
    for chunk in ranked_chunks:
        if chunk.chunk_type in ("function", "class", "method"):
            return True
        if _DEF_PATTERN.search(chunk.excerpt):
            return True
    return False


def compute_retrieval_quality(
    query_terms: list[str],
    ranked_chunks: list[RankedChunk],
    top_score: float,
) -> RetrievalQuality:
    """Compute RetrievalQuality signals from ranked chunks.

    Args:
        query_terms: Query terms (will be lowercased and stopword-filtered)
        ranked_chunks: Ranked chunk results
        top_score: Top BM25 score from the retrieval pipeline

    Returns:
        RetrievalQuality with computed flags and label
    """
    coverage = _compute_coverage(query_terms, ranked_chunks)
    path_entropy = _compute_path_entropy(ranked_chunks)
    has_definition = _has_definition(ranked_chunks)
    score_gap_ok = top_score > 10.0

    flags: list[str] = []
    if coverage < 0.5:
        flags.append("low_coverage")
    if path_entropy < 0.5:
        flags.append("mono_dir")
    if not has_definition:
        flags.append("no_def")
    if not score_gap_ok:
        flags.append("flat_score")

    if len(flags) >= 2 or path_entropy < 0.5:
        quality_label = "weak"
    elif len(flags) == 1:
        quality_label = "mixed"
    else:
        quality_label = "strong"

    return RetrievalQuality(
        score_gap_ok=score_gap_ok,
        coverage=coverage,
        path_entropy=path_entropy,
        has_definition=has_definition,
        flags=flags,
        quality_label=quality_label,
    )
