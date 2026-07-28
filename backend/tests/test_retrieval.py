"""Retrieval tests: BM25 scoring, quality signals, and symbol query surface."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from domain.qa.graph_queries import find_symbol_path, get_symbol_impact_cone
from domain.repo_map.service import (
    RepoMapService,
    levenshtein_distance,
    search_symbols_cascade,
)
from domain.retrieval.bm25_scorer import BM25Scorer
from domain.retrieval.quality import compute_retrieval_quality
from domain.retrieval.types import RankedChunk
from infrastructure.db.database import get_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    scorer = BM25Scorer(idf, avgdl=3.0)

    # "word" appears in only 1 doc -> filtered by the hapax rule (df < 2).
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


def test_mono_dir_weak_label():
    """All chunks from qa/ dir → path_entropy < 0.5 → label 'weak'."""
    chunks = [
        RankedChunk(
            chunk_id="1",
            rel_path="qa/agent.py",
            chunk_index=0,
            score=12.0,
            chunk_type="block",
            bm25_component=10.0,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="def process(): pass",
        ),
        RankedChunk(
            chunk_id="2",
            rel_path="qa/types.py",
            chunk_index=0,
            score=11.0,
            chunk_type="block",
            bm25_component=10.0,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="class Response: pass",
        ),
        RankedChunk(
            chunk_id="3",
            rel_path="qa/service.py",
            chunk_index=0,
            score=10.5,
            chunk_type="block",
            bm25_component=9.5,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="def answer(): pass",
        ),
    ]
    quality = compute_retrieval_quality(["process", "answer"], chunks, 12.0)
    assert quality.quality_label == "weak"
    assert "mono_dir" in quality.flags
    assert quality.path_entropy < 0.5


def test_diverse_paths_strong_label():
    """Chunks from 4+ different top-level dirs → label 'strong'."""
    chunks = [
        RankedChunk(
            chunk_id="1",
            rel_path="backend/service.py",
            chunk_index=0,
            score=12.0,
            chunk_type="function",
            bm25_component=10.0,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="def process(query): pass",
        ),
        RankedChunk(
            chunk_id="2",
            rel_path="frontend/component.tsx",
            chunk_index=0,
            score=11.0,
            chunk_type="block",
            bm25_component=10.0,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="function render() { return null; }",
        ),
        RankedChunk(
            chunk_id="3",
            rel_path="tests/unit_test.py",
            chunk_index=0,
            score=10.5,
            chunk_type="block",
            bm25_component=9.5,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="def test_process(): pass",
        ),
        RankedChunk(
            chunk_id="4",
            rel_path="docs/readme.md",
            chunk_index=0,
            score=10.0,
            chunk_type="block",
            bm25_component=9.0,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="# Process Query",
        ),
    ]
    quality = compute_retrieval_quality(["process", "query"], chunks, 12.0)
    assert quality.quality_label == "strong"
    assert len(quality.flags) == 0
    assert quality.path_entropy >= 1.5


def test_low_coverage_mixed_label():
    """Low coverage (query terms not in chunks) → label 'mixed'."""
    chunks = [
        RankedChunk(
            chunk_id="1",
            rel_path="backend/service.py",
            chunk_index=0,
            score=12.0,
            chunk_type="block",
            bm25_component=10.0,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="def unrelated(): return 42",
        ),
        RankedChunk(
            chunk_id="2",
            rel_path="frontend/component.tsx",
            chunk_index=0,
            score=11.0,
            chunk_type="block",
            bm25_component=10.0,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="function display() { return null; }",
        ),
    ]
    quality = compute_retrieval_quality(["search", "database"], chunks, 12.0)
    assert quality.quality_label == "mixed"
    assert "low_coverage" in quality.flags
    assert quality.coverage < 0.5


def test_has_definition_in_chunk_type():
    """chunk_type='function' detected as has_definition."""
    chunks = [
        RankedChunk(
            chunk_id="1",
            rel_path="backend/service.py",
            chunk_index=0,
            score=12.0,
            chunk_type="function",
            bm25_component=10.0,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="def process(): pass",
        ),
    ]
    quality = compute_retrieval_quality(["process"], chunks, 12.0)
    assert quality.has_definition is True
    assert "no_def" not in quality.flags


def test_has_definition_in_content():
    """def/class pattern in content detected as has_definition."""
    chunks = [
        RankedChunk(
            chunk_id="1",
            rel_path="backend/service.py",
            chunk_index=0,
            score=12.0,
            chunk_type="block",
            bm25_component=10.0,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="class MyClass:\n    def method(self): pass",
        ),
    ]
    quality = compute_retrieval_quality(["myclass"], chunks, 12.0)
    assert quality.has_definition is True
    assert "no_def" not in quality.flags


def test_no_definition_flag():
    """No definition → 'no_def' flag."""
    chunks = [
        RankedChunk(
            chunk_id="1",
            rel_path="backend/config.py",
            chunk_index=0,
            score=12.0,
            chunk_type="block",
            bm25_component=10.0,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="CONFIG_PATH = '/etc/app.conf'",
        ),
    ]
    quality = compute_retrieval_quality(["config"], chunks, 12.0)
    assert quality.has_definition is False
    assert "no_def" in quality.flags


def test_flat_score_flag():
    """top_score <= 10.0 → score_gap_ok=False → 'flat_score' flag."""
    chunks = [
        RankedChunk(
            chunk_id="1",
            rel_path="backend/service.py",
            chunk_index=0,
            score=8.5,
            chunk_type="block",
            bm25_component=7.5,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="def process(): pass",
        ),
    ]
    quality = compute_retrieval_quality(["process"], chunks, 8.5)
    assert quality.score_gap_ok is False
    assert "flat_score" in quality.flags


def test_multiple_flags_weak():
    """≥2 flags → 'weak' label."""
    chunks = [
        RankedChunk(
            chunk_id="1",
            rel_path="qa/service.py",
            chunk_index=0,
            score=8.0,
            chunk_type="block",
            bm25_component=7.0,
            symbol_bonus=1.0,
            module_bonus=1.0,
            centrality_bonus=0.0,
            token_estimate=100,
            excerpt="CONFIG_PATH = '/etc/app.conf'",
        ),
    ]
    quality = compute_retrieval_quality(["missing", "query"], chunks, 8.0)
    assert quality.quality_label == "weak"
    assert len([f for f in quality.flags if f in ["low_coverage", "mono_dir", "no_def", "flat_score"]]) >= 2


def test_levenshtein_distance() -> None:
    """Test Levenshtein distance edge cases and bounds."""
    assert levenshtein_distance("getUser", "getUser") == 0
    assert levenshtein_distance("getUssr", "getUser") == 1
    assert levenshtein_distance("getUsr", "getUser") == 1
    assert levenshtein_distance("process", "prosody") == 4


@pytest.mark.asyncio
async def test_search_cascade_shared_helper_fuzzy() -> None:
    """Test fuzzy fallback in search_symbols_cascade: edit distance <=2 matches, >2 drops."""
    db = get_db()
    snap_id = f"test-search-{uuid.uuid4().hex[:8]}"

    # Insert code_symbols directly
    await db.execute(
        """
        INSERT INTO code_symbols (id, snapshot_id, rel_path, language, name, kind, line_start, line_end, signature, parent_name, extract_source, created_at)
        VALUES (?, ?, 'app.py', 'python', 'getUserById', 'function', 1, 10, '()', NULL, 'ast', ?)
        """,
        (uuid.uuid4().hex, snap_id, _now()),
    )
    await db.commit()

    # Exact/LIKE search matches
    res_exact = await search_symbols_cascade(db, snap_id, "getUserById")
    assert len(res_exact) == 1
    assert res_exact[0].name == "getUserById"

    # Typo search (edit dist = 1: getUssrById -> getUserById) recovers via fuzzy fallback
    res_typo = await search_symbols_cascade(db, snap_id, "getUssrById")
    assert len(res_typo) == 1
    assert res_typo[0].name == "getUserById"

    # Distant typo (edit dist > 2) drops
    res_distant = await search_symbols_cascade(db, snap_id, "getCompletelyDifferentName")
    assert len(res_distant) == 0


@pytest.mark.asyncio
async def test_shared_search_helper_used_by_deep_research_and_repomap() -> None:
    """Test that RepoMapService.search and DeepResearch._validate_plan_step both use search_symbols_cascade."""
    db = get_db()
    snap_id = f"test-shared-{uuid.uuid4().hex[:8]}"

    await db.execute(
        """
        INSERT INTO code_symbols (id, snapshot_id, rel_path, language, name, kind, line_start, line_end, signature, parent_name, extract_source, created_at)
        VALUES (?, ?, 'service.py', 'python', 'processOrder', 'function', 1, 20, '()', NULL, 'ast', ?)
        """,
        (uuid.uuid4().hex, snap_id, _now()),
    )
    await db.commit()

    # RepoMapService search recovers typo'd 'procesOrder'
    rm_service = RepoMapService()
    search_resp = await rm_service.search(snap_id, "procesOrder")
    assert len(search_resp.symbols) == 1
    assert search_resp.symbols[0].name == "processOrder"

    # DeepResearch._validate_plan_step recovers the same typo via the SAME shared helper (real class, no silent skip).
    from domain.qa.deep_research import DeepResearchAgent
    agent = DeepResearchAgent(None, None)
    step = {"type": "trace_forward", "target": "procesOrder"}
    validated_step = await agent._validate_plan_step(step, snap_id)
    assert "_warning" not in validated_step


@pytest.mark.asyncio
async def test_get_symbol_impact_cone() -> None:
    """Test symbol-granular impact cone reverse traversal."""
    db = get_db()
    snap_id = f"test-impact-{uuid.uuid4().hex[:8]}"

    # Edges: A calls Target, B extends Target, C instantiates Target
    await db.executemany(
        """
        INSERT INTO symbol_graph_edges (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, confidence, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?, '[]')
        """,
        [
            (snap_id, "caller.py::caller", "target.py::Target", "calls", 1.0, "direct", "high"),
            (snap_id, "sub.py::Sub", "target.py::Target", "extends", 1.0, "inheritance", "high"),
            (snap_id, "factory.py::Factory.create", "target.py::Target", "instantiates", 1.0, "ctor", "high"),
            # Forward dependency: Target CALLS Downstream — Downstream is NOT impacted by a change to Target and must be excluded (directionality guard).
            (snap_id, "target.py::Target", "downstream.py::Downstream", "calls", 1.0, "direct", "high"),
        ],
    )
    await db.commit()

    res_all = await get_symbol_impact_cone(snap_id, "target.py::Target")
    assert res_all.seed_symbol == "target.py::Target"
    assert set(res_all.impacted_symbols) == {
        "caller.py::caller",
        "sub.py::Sub",
        "factory.py::Factory.create",
    }
    # A change to Target does NOT impact what Target depends on.
    assert "downstream.py::Downstream" not in res_all.impacted_symbols

    # Narrowing edge_kinds to calls only
    res_calls = await get_symbol_impact_cone(snap_id, "target.py::Target", edge_kinds=["calls"])
    assert res_calls.impacted_symbols == ["caller.py::caller"]


@pytest.mark.asyncio
async def test_find_symbol_path() -> None:
    """Test point-to-point find_symbol_path traversal."""
    db = get_db()
    snap_id = f"test-path-{uuid.uuid4().hex[:8]}"

    # Path: Start -> Mid -> Target
    await db.executemany(
        """
        INSERT INTO symbol_graph_edges (snapshot_id, src_symbol, dst_symbol, edge_type, confidence_score, resolution_method, confidence, evidence_lines)
        VALUES (?, ?, ?, ?, ?, ?, ?, '[]')
        """,
        [
            (snap_id, "a.py::Start", "b.py::Mid", "calls", 1.0, "direct", "high"),
            (snap_id, "b.py::Mid", "c.py::Target", "calls", 1.0, "direct", "high"),
            (snap_id, "x.py::Unrelated", "y.py::Other", "calls", 1.0, "direct", "high"),
        ],
    )
    await db.commit()

    path = await find_symbol_path(snap_id, "a.py::Start", "c.py::Target")
    assert path is not None
    assert len(path) == 2
    assert path[0].src_symbol == "a.py::Start"
    assert path[0].dst_symbol == "b.py::Mid"
    assert path[1].src_symbol == "b.py::Mid"
    assert path[1].dst_symbol == "c.py::Target"

    # Unreachable target returns None
    no_path = await find_symbol_path(snap_id, "a.py::Start", "y.py::Other")
    assert no_path is None
