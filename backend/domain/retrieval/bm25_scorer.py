"""BM25 scorer for retrieval (CS-201).

Pre-computed IDF is loaded from retrieval_bm25_stats at query time.
Corpus tokenization uses the same _WORD regex as _query_terms() in service.py
to guarantee token-set alignment between build time and query time.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

_WORD = re.compile(r"[A-Za-z0-9_]+")


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

    def score(self, terms: list[str], content_low: str, path_low: str) -> float:
        """BM25 score for a chunk against query terms.

        Terms must already be lowercased (same as _query_terms() output).
        content_low and path_low must be lowercased.
        Path hits are treated as a synthetic document of length avgdl with TF=1.
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

        return total

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
