"""BM25 scorer for retrieval (CS-201, CS-227).

Pre-computed IDF is loaded from retrieval_bm25_stats at query time.
Corpus tokenization uses the same _WORD regex and identifier splitting
to guarantee token-set alignment between build time and query time.
"""
from __future__ import annotations

import importlib
import json
import math
import re
from typing import Any

_WORD = re.compile(r"[A-Za-z0-9_]+")

_NATIVE_BM25 = None
_NATIVE_BM25_LOADED = False

# Chunk type weights for code-aware retrieval (CS-227, CS-229)
CHUNK_TYPE_WEIGHT: dict[str, float] = {
    "function":     1.5,
    "class":        1.5,
    "block":        1.0,
    "file":         1.0,
    "import_group": 0.2,
    "barrel":       0.3,
}


def _load_native_bm25():
    """Lazy loader for the native BM25 extension. Returns module or None."""
    global _NATIVE_BM25, _NATIVE_BM25_LOADED
    if _NATIVE_BM25_LOADED:
        return _NATIVE_BM25
    _NATIVE_BM25_LOADED = True
    try:
        _NATIVE_BM25 = importlib.import_module("domain.retrieval._native_bm25")
    except Exception:
        _NATIVE_BM25 = None
    return _NATIVE_BM25


def split_identifier(token: str) -> list[str]:
    """Split camelCase and snake_case identifiers into sub-tokens.

    Examples:
      'getUserById' → ['getuserbyid', 'get', 'user', 'by', 'id']
      'get_user_by_id' → ['get_user_by_id', 'get', 'user', 'by', 'id']
      'HTTPServer' → ['httpserver', 'http', 'server']

    Single-char tokens are dropped.
    """
    if not token or len(token) <= 1:
        return [token] if token else []

    # Original token (lowercased)
    lower = token.lower()
    result = [lower]

    # Snake_case split
    if "_" in token:
        parts = token.lower().split("_")
        for p in parts:
            if len(p) > 1:
                result.append(p)
        return result

    # camelCase / PascalCase split: insert markers before uppercase letters
    # Then split on markers
    marked = ""
    for i, ch in enumerate(token):
        if ch.isupper() and i > 0:
            marked += "|" + ch
        else:
            marked += ch

    sub_tokens = []
    for part in marked.split("|"):
        sub = part.lower()
        if len(sub) > 1:
            sub_tokens.append(sub)

    # Add unique sub-tokens
    for st in sub_tokens:
        if st not in result:
            result.append(st)

    return result


def _query_terms(q: str) -> list[str]:
    """Extract and deduplicate query terms with identifier splitting (CS-227).

    Returns lowercased terms in order, with sub-tokens from split_identifier().
    """
    # Extract raw tokens
    raw_tokens = [w.lower() for w in _WORD.findall(q) if len(w) > 1]

    # Expand with split_identifier
    expanded: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        for sub in split_identifier(token):
            if sub and sub not in seen:
                expanded.append(sub)
                seen.add(sub)

    return expanded


class BM25Scorer:
    def __init__(self, idf: dict[str, float], avgdl: float, k1: float = 2.0, b: float = 0.75) -> None:
        self._idf = idf
        self._avgdl = avgdl
        self._k1 = k1
        self._b = b

    @classmethod
    def from_stats_row(cls, row: Any) -> "BM25Scorer | None":
        if row is None:
            return None
        return cls(
            idf=json.loads(row["idf_json"]),
            avgdl=float(row["avgdl"]),
            k1=float(row["k1"]),
            b=float(row["b"]),
        )

    def score(
        self,
        terms: list[str],
        content_low: str,
        path_low: str,
        chunk_type: str = "block",
    ) -> float:
        """BM25 score for a chunk against query terms, with type weighting (CS-227).

        Terms must already be lowercased (same as _query_terms() output).
        content_low and path_low must be lowercased.
        Path hits are treated as a synthetic document of length avgdl with TF=1.
        chunk_type: used to apply CHUNK_TYPE_WEIGHT multiplier.
        """
        if not terms:
            return 0.0

        # Tokenize content the same way as build_index() corpus tokenization
        tokens = _WORD.findall(content_low)
        dl = len(tokens)
        if dl == 0:
            return 0.0

        # Build term frequency map for this chunk
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        k1 = self._k1
        b = self._b
        avgdl = self._avgdl if self._avgdl > 0 else 1.0
        length_norm = 1 - b + b * dl / avgdl

        total = 0.0
        for term in terms:
            idf = self._idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            # Content contribution
            f = tf.get(term, 0)
            content_contrib = idf * (f * (k1 + 1)) / (f + k1 * length_norm)
            # Path contribution: treat as synthetic doc of length avgdl, TF=1
            path_contrib = idf * 1.0 if term in path_low else 0.0
            total += content_contrib + path_contrib

        # Apply chunk type weight multiplier (CS-227)
        weight = CHUNK_TYPE_WEIGHT.get(chunk_type, 1.0)
        return total * weight

    def score_batch(
        self,
        chunks: list[tuple[str, str, str] | tuple[str, str, str, str]],
        terms: list[str],
        min_score_abs: float = 0.0,
        min_score_relative: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Score multiple chunks with type weighting and cutoff (CS-227).

        chunks: list of (chunk_id, content, path) or (chunk_id, content, path, chunk_type).
                If 3-tuple, chunk_type defaults to "block".
        terms:  pre-lowercased query terms (same as score() input).
        min_score_abs: absolute floor (e.g. 0.5).
        min_score_relative: relative fraction of top score (e.g. 0.25).

        Returns: list of (chunk_id, score) — only survivors (score >= cutoff).
        """
        native = _load_native_bm25()
        if native is not None:
            try:
                # Convert 3-tuples to 4-tuples for native call
                chunks_4tuple = []
                for chunk in chunks:
                    if len(chunk) == 3:
                        chunk_id, content, path = chunk
                        chunks_4tuple.append((chunk_id, content, path, "block"))
                    else:
                        chunks_4tuple.append(chunk)
                return list(
                    native.batch_score(
                        chunks_4tuple,
                        terms,
                        self._idf,
                        self._avgdl,
                        self._k1,
                        self._b,
                        CHUNK_TYPE_WEIGHT,
                        min_score_abs,
                        min_score_relative,
                    )
                )
            except Exception:
                pass

        # Python fallback: compute all scores, apply cutoff
        scored: list[tuple[str, str, str, str, float]] = []
        for chunk in chunks:
            if len(chunk) == 3:
                chunk_id, content, path = chunk
                chunk_type = "block"
            else:
                chunk_id, content, path, chunk_type = chunk
            s = self.score(terms, content.lower(), path.lower(), chunk_type)
            scored.append((chunk_id, content, path, chunk_type, s))

        # Find top score and compute cutoff
        if not scored:
            return []
        top_score = max(s[4] for s in scored)
        cutoff = max(min_score_abs, min_score_relative * top_score) if top_score > 0 else min_score_abs

        # Return only survivors
        results: list[tuple[str, float]] = []
        for chunk_id, _, _, _, score in scored:
            if score >= cutoff:
                results.append((chunk_id, score))
        return results

    @staticmethod
    def build_idf(tokenized_corpus: list[list[str]], n_docs: int) -> dict[str, float]:
        """Compute BM25 IDF for all terms in corpus.

        Uses BM25 IDF formula: log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
        The +1 ensures IDF >= 0 for very common terms.
        """
        doc_freq: dict[str, int] = {}
        for tokens in tokenized_corpus:
            for tok in set(tokens):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        idf: dict[str, float] = {}
        for term, df in doc_freq.items():
            # Skip hapax legomena (appears in only 1 doc) to reduce IDF blob size
            if df < 2:
                continue
            idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        return idf
