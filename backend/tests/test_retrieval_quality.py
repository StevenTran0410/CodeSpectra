"""Tests for RetrievalQuality signals computation."""

import pytest

from domain.retrieval.quality import compute_retrieval_quality
from domain.retrieval.types import RankedChunk


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
