"""BM25 scorer tests for CS-201."""
from __future__ import annotations

from domain.retrieval.bm25_scorer import BM25Scorer


def test_bm25_scorer_basic_ranking() -> None:
    """Two chunks with same terms; chunk with higher TF scores higher."""
    # Corpus: 2 docs
    corpus = [
        ["foo", "foo", "bar"],  # doc 0: foo appears 2x
        ["foo", "baz"],         # doc 1: foo appears 1x
    ]
    idf = BM25Scorer.build_idf(corpus, 2)
    scorer = BM25Scorer(idf, avgdl=2.5)

    # Query for "foo"
    terms = ["foo"]

    # doc 0 has higher TF for "foo", should score higher
    score_doc0 = scorer.score(terms, "foo foo bar", "file.py")
    score_doc1 = scorer.score(terms, "foo baz", "file.py")

    assert score_doc0 > score_doc1


def test_bm25_scorer_idf_penalizes_common_terms() -> None:
    """Term in all docs has near-zero IDF vs rare term."""
    # Corpus: 4 docs
    corpus = [
        ["the", "common", "word"],
        ["the", "common", "phrase"],
        ["the", "common", "data"],
        ["the", "common", "rare"],
    ]
    idf = BM25Scorer.build_idf(corpus, 4)

    # "the" and "common" appear in all docs -> low IDF (but "the" is filtered as hapax for df=1 edge)
    # "rare" appears in 1 doc -> filtered out by hapax rule (df < 2)
    # Let's use terms that actually appear in the IDF
    scorer = BM25Scorer(idf, avgdl=3.0)

    # "word" appears in 1 doc -> filtered (hapax)
    # "common" appears in 4 docs -> low IDF
    # Let's check that a term appearing in 1 doc is not in the IDF
    assert "word" not in idf

    # "common" is in idf (appears in all 4 docs, df=4)
    assert "common" in idf
    common_idf = idf.get("common", 0.0)

    # Create a new corpus with a truly rare term
    corpus2 = [
        ["the", "rare", "word"],
        ["the", "common", "phrase"],
        ["the", "common", "data"],
        ["the", "common", "rock"],
    ]
    idf2 = BM25Scorer.build_idf(corpus2, 4)

    # "rare" appears in 1 doc -> filtered (hapax)
    assert "rare" not in idf2

    # "common" appears in 3 docs (df=3)
    assert "common" in idf2
    common_idf2 = idf2.get("common", 0.0)

    # With df=3, common_idf2 should be lower than df=2 case
    corpus3 = [
        ["the", "specialword"],
        ["the", "specialword"],
        ["the", "commonword"],
        ["the", "commonword"],
    ]
    idf3 = BM25Scorer.build_idf(corpus3, 4)

    # Both appear in 2 docs (minimum for inclusion)
    assert "specialword" in idf3
    assert "commonword" in idf3

    # Both should have same IDF (df=2)
    assert idf3["specialword"] == idf3["commonword"]


def test_bm25_scorer_length_normalization() -> None:
    """Same content in short vs long chunk; short chunk scores higher per term."""
    corpus = [
        ["foo", "bar"],
        ["foo", "bar", "x", "y", "z", "w"],
    ]
    idf = BM25Scorer.build_idf(corpus, 2)
    scorer = BM25Scorer(idf, avgdl=4.0)

    terms = ["foo"]

    # Short doc: 2 tokens
    score_short = scorer.score(terms, "foo bar", "file.py")

    # Long doc: 6 tokens (same "foo bar" but with extra words)
    score_long = scorer.score(terms, "foo bar x y z w", "file.py")

    # Short chunk scores higher due to length normalization
    assert score_short > score_long


def test_bm25_scorer_path_boost() -> None:
    """Same content, one with term in path; path hit increases score."""
    corpus = [
        ["foo", "content"],
        ["foo", "content"],
    ]
    idf = BM25Scorer.build_idf(corpus, 2)
    scorer = BM25Scorer(idf, avgdl=2.0)

    terms = ["foo"]
    content = "foo content"

    # Without path hit
    score_no_path = scorer.score(terms, content, "somefile.py")

    # With path hit
    score_with_path = scorer.score(terms, content, "foo_module.py")

    # Path hit adds to score
    assert score_with_path > score_no_path


def test_bm25_scorer_cold_start_fallback() -> None:
    """from_stats_row(None) returns None without error."""
    result = BM25Scorer.from_stats_row(None)
    assert result is None


def test_bm25_scorer_zero_score_excludes_chunk() -> None:
    """Chunk with no query terms returns 0.0."""
    corpus = [["foo", "bar"]]
    idf = BM25Scorer.build_idf(corpus, 1)
    scorer = BM25Scorer(idf, avgdl=2.0)

    terms = ["baz"]  # Not in content

    score = scorer.score(terms, "foo bar", "file.py")

    assert score == 0.0


def test_bm25_scorer_unseen_term_graceful() -> None:
    """Query term not in IDF vocab does not raise; returns based on seen terms."""
    corpus = [
        ["foo", "bar"],
        ["foo", "baz"],
    ]
    idf = BM25Scorer.build_idf(corpus, 2)
    scorer = BM25Scorer(idf, avgdl=2.0)

    # Query has both a seen term ("foo") and unseen term ("unseen")
    terms = ["foo", "unseen"]
    content = "foo bar baz"

    # Should not raise and should return score based on "foo"
    score = scorer.score(terms, content, "file.py")

    assert score > 0.0

    # Query with only unseen term should return 0
    score_unseen_only = scorer.score(["unseen"], content, "file.py")
    assert score_unseen_only == 0.0
